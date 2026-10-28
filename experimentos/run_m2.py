#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner M2 - Markov + refinamento do texto de regras por LLM.

O M2 NÃO muda a estrutura da carta: tipo, subtipos, custo, CMC, P/T e
palavras-chave continuam vindo do pipeline (idênticos ao M1 para a mesma
carta, porque a semente é a mesma). Só `texto_regras` é substituído pela
versão refinada pelo `gemma3:4b`.

Isso é o que D-004 §7.2 e D-005 §8.4 exigem: o contraste M1 × M2 mede
exclusivamente o efeito do texto.

Uso:
    python experimentos/run_m2.py

Requer o Ollama no ar (`ollama serve`) com o modelo baixado. O arquivo
`resultados_m2.jsonl` é append-only e o runner retoma de onde parou.
"""

import json
import sys
import time
import urllib.error
import urllib.request

from comum import (
    EXPERIMENTOS,
    RAIZ,
    campo_carta,
    carregar_itens,
    cor_do_pipeline,
    criar_log,
    ordens_concluidas,
)

sys.path.insert(0, str(RAIZ / "src"))
from pipeline import gerar_carta  # noqa: E402

OUT = EXPERIMENTOS / "resultados_m2.jsonl"
LOG = EXPERIMENTOS / "log_m2.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

# Prompt deliberadamente restrito: mexe só no texto de regras. Instruções
# explícitas para não tocar em nome/tipo/custo/P-T/keywords, porque o LLM
# tende a "melhorar" a carta inteira se não for contido.
SISTEMA = (
    "Você é um editor de regras de Magic: The Gathering. "
    "Sua ÚNICA tarefa é refinar o TEXTO DE REGRAS desta carta. "
    "NUNCA mude nome, tipo, subtipo, custo de mana, poder, resistência, "
    "lealdade ou palavras-chave. "
    "Mantenha o número de parágrafos, a mecânica e o sentido originais. "
    "Use terminologia correta de MTG, seja conciso e direto. "
    "Retorne APENAS o texto de regras refinado, sem explicações, sem JSON, "
    "sem markdown."
)


def refinar_texto(texto: str, tipo_linha: str, seed: int, log) -> str | None:
    """Devolve o texto refinado, ou None se a chamada falhar.

    Falha aqui NÃO invalida a carta: o M2 mantém o texto do Markov e o
    registro guarda `refinado: false`. Assim a taxa de falha do refino vira
    métrica (D-002), em vez de sumir silenciosamente."""
    if not texto or not texto.strip():
        return None

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"TIPO: {tipo_linha}\n\nTEXTO ATUAL:\n{texto}\n\nTEXTO REFINADO:"},
        ],
        "stream": False,
        # `seed` no Ollama torna a amostragem reproduzível para o mesmo input.
        "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 256, "seed": seed},
    }
    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=300) as resposta:
            dados = json.loads(resposta.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log(f"  [WARN] refino falhou: {exc}")
        return None

    refinado = dados.get("message", {}).get("content", "").strip()
    if not refinado:
        return None
    # Alguns modelos devolvem o texto entre aspas ou com cercas de código.
    if refinado.startswith('"') and refinado.endswith('"') and len(refinado) > 1:
        refinado = refinado[1:-1].strip()
    if refinado.startswith("```"):
        linhas = [l for l in refinado.splitlines() if not l.strip().startswith("```")]
        refinado = "\n".join(linhas).strip()
    return refinado or None


def main() -> None:
    log = criar_log(LOG)
    itens = carregar_itens()
    concluidas = ordens_concluidas(OUT)

    refinados = 0
    sem_refino = 0
    erros = 0
    # Recompoe os contadores a partir do que já está no arquivo, para o
    # resumo final refletir a execução inteira e não só a retomada.
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for linha in f:
                try:
                    registro = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if registro.get("modelo") != "M2" or "erro" in registro:
                    continue
                if registro.get("refinado"):
                    refinados += 1
                else:
                    sem_refino += 1

    log(f"Iniciando M2: {len(itens)} cartas | {len(concluidas)} já concluídas")

    with open(OUT, "a", encoding="utf-8") as fout:
        for item in itens:
            if item["ordem"] in concluidas:
                continue
            try:
                inicio = time.time()
                carta = gerar_carta(
                    nome=item["nome"],
                    cor=cor_do_pipeline(item["cor"]),
                    tipo=item["tipo_forcado"],
                    seed=item["seed"],
                    usar_arte=False,
                )
                texto_markov = carta.texto_regras
                texto_refinado = refinar_texto(
                    texto_markov, carta.tipo_linha, item["seed"], log
                )

                refinado = bool(texto_refinado) and texto_refinado != texto_markov
                if refinado:
                    carta.texto_regras = texto_refinado
                    refinados += 1
                else:
                    sem_refino += 1

                registro = {
                    "modelo": "M2",
                    "ordem": item["ordem"],
                    "carta_id": item["carta_id"],
                    "tipo_experimento": item["tipo_experimento"],
                    "nome_entrada": item["nome"],
                    "cor_entrada": item["cor"],
                    "tipo_forcado": item["tipo_forcado"],
                    "seed": item["seed"],
                    **campo_carta(carta),
                    "refinado": refinado,
                    "texto_regras_markov": texto_markov,
                    "gabarito": item["gabarito"],
                    "tempo_seg": round(time.time() - inicio, 2),
                }
                fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as exc:  # noqa: BLE001 - falha por carta é métrica
                erros += 1
                log(f"ERRO carta {item['ordem']} ({item['nome']}): {exc}")
                fout.write(
                    json.dumps(
                        {
                            "modelo": "M2",
                            "ordem": item["ordem"],
                            "carta_id": item["carta_id"],
                            "tipo_experimento": item["tipo_experimento"],
                            "erro": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

            if (item["ordem"] + 1) % 25 == 0:
                log(
                    f"Progresso: {item['ordem'] + 1}/{len(itens)} | "
                    f"refinados={refinados} sem_refino={sem_refino} erros={erros}"
                )

    log(
        f"Concluido M2. refinados={refinados} sem_refino={sem_refino} "
        f"erros={erros}. Resultado em {OUT}"
    )


if __name__ == "__main__":
    main()
