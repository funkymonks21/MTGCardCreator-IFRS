#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner do Modo B — inferência de tipo a partir do nome.

Segue D-004 §7.3 e D-005:

- **Entrada: apenas o nome.** Nada de cor e nada de tipo — para todos os
  braços. Dar cor só ao M3 tornaria a comparação inválida (D-005 §8.1).
- **M1/M2:** `inferir_tipo_por_nome` (POS tagging + WordNet + listas
  curadas). Se ele se abstém (`tipo is None`), o pipeline sorteia pela
  frequência real do corpus, com o MESMO `rng` semeado. O refino do M2 é só
  de texto, então M1 ≡ M2; as duas previsões são idênticas por construção e
  não precisam ser geradas duas vezes.
- **M3:** o `gemma3:4b` escolhe entre os 6 tipos, com enum no prompt +
  JSON schema (`format`) e 1 retry.
- **Amostra:** 400 cartas sorteadas das 1.000, com semente fixa.

O runner grava `resultados_modo_b.jsonl` em modo append e retoma pelas
`ordem` já concluídas, como os runners do Modo A.

Uso:
    python experimentos/run_modo_b.py

Requer o Ollama no ar com `gemma3:4b`.
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request

from comum import (
    EXPERIMENTOS,
    RAIZ,
    carregar_itens,
    criar_log,
    ordens_concluidas,
)

sys.path.insert(0, str(RAIZ / "src"))
from pipeline.generator import _obter_modelo  # noqa: E402
from pipeline.tipo_por_nome import inferir_tipo_por_nome  # noqa: E402

OUT = EXPERIMENTOS / "resultados_modo_b.jsonl"
LOG = EXPERIMENTOS / "log_modo_b.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

AMOSTRA_N = 400
SEED_AMOSTRA = 20261025

TIPOS = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land"]

SISTEMA = (
    "Você classifica nomes de cartas de Magic: The Gathering. "
    "Recebe APENAS um nome e deve escolher exatamente UM dos 6 tipos: "
    "Creature, Instant, Sorcery, Artifact, Enchantment, Land. "
    "Retorne APENAS JSON válido no schema, com o campo card_type. "
    "Não explique e não invente outros tipos."
)

SCHEMA = {
    "type": "object",
    "required": ["card_type"],
    "additionalProperties": False,
    "properties": {
        "card_type": {"type": "string", "enum": TIPOS},
    },
}


def selecionar(itens: list[dict]) -> list[dict]:
    """400 cartas sorteadas uma única vez, com semente fixa (D-004 §7.3)."""
    rng = random.Random(SEED_AMOSTRA)
    amostra = rng.sample(itens, AMOSTRA_N)
    return sorted(amostra, key=lambda item: item["ordem"])


def prever_m1(nome: str, seed: int) -> dict:
    """Réplica exata do caminho `_TIPO_AUTO` de `gerar_carta`.

    Não chamamos `gerar_carta` inteiro porque o Modo B só precisa do tipo: o
    pipeline gastaria Markov/subtipos/regressão para produzir uma carta que
    não será avaliada. O `rng` é o mesmo objeto usado na inferência e no
    fallback, como em `generator.py` (linhas 321–330).
    """
    rng = random.Random(seed)
    inferencia = inferir_tipo_por_nome(nome, rng=rng)
    if inferencia.tipo is None:
        tipo = _obter_modelo().sortear_tipo(rng=rng)
        return {
            "tipo": tipo,
            "absteve": True,
            "motivo": inferencia.motivo,
        }
    return {
        "tipo": inferencia.tipo,
        "absteve": False,
        "motivo": inferencia.motivo,
    }


def prever_m3(nome: str, seed: int) -> dict:
    """Uma chamada ao Ollama. Devolve o JSON decodificado ou `__erro__`."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"Nome da carta: {nome}"},
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {
            # Tarefa discriminativa: temperatura 0 para medir a melhor
            # decisão do modelo, não a variância de amostragem.
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 24,
            "seed": seed,
        },
    }
    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=300) as resposta:
            dados = json.loads(resposta.read())
        conteudo = dados.get("message", {}).get("content", "").strip()
        if not conteudo:
            return {"__erro__": "resposta vazia"}
        return json.loads(conteudo)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"__erro__": str(exc)}


def main() -> None:
    log = criar_log(LOG)
    itens = carregar_itens()
    amostra = selecionar(itens)
    concluidas = ordens_concluidas(OUT)

    log(
        f"Iniciando Modo B: {len(amostra)} cartas "
        f"(semente da amostra {SEED_AMOSTRA}) | {len(concluidas)} já concluídas"
    )

    with open(OUT, "a", encoding="utf-8") as fout:
        for item in amostra:
            if item["ordem"] in concluidas:
                continue
            inicio = time.time()
            try:
                m1 = prever_m1(item["nome"], item["seed"])
            except Exception as exc:  # noqa: BLE001 - falha por carta é métrica
                log(f"ERRO M1 ordem {item['ordem']} ({item['nome']}): {exc}")
                fout.write(
                    json.dumps(
                        {
                            "experimento": "modo_b",
                            "ordem": item["ordem"],
                            "carta_id": item["carta_id"],
                            "erro": f"M1: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
                continue

            m3 = None
            motivo = "sem tentativa"
            tentativas = 0
            bruto_txt = None
            for tentativa in (1, 2):
                tentativas = tentativa
                bruta = prever_m3(item["nome"], item["seed"] + tentativa)
                if "__erro__" in bruta:
                    motivo = f"erro de rede/JSON: {bruta['__erro__']}"
                    log(
                        f"  [M3] {item['nome']} tentativa {tentativa}: {motivo}"
                    )
                    continue
                tipo = bruta.get("card_type")
                if tipo not in TIPOS:
                    motivo = f"card_type fora do enum: {tipo!r}"
                    log(
                        f"  [M3] {item['nome']} tentativa {tentativa}: {motivo}"
                    )
                    continue
                m3 = {"tipo": tipo.lower()}
                bruto_txt = json.dumps(bruta, ensure_ascii=False)
                break

            if m3 is None:
                m3 = {"tipo": None, "erro": motivo}

            registro = {
                "experimento": "modo_b",
                "ordem": item["ordem"],
                "carta_id": item["carta_id"],
                "tipo_experimento": item["tipo_experimento"],
                "nome_entrada": item["nome"],
                # Registrada para auditoria; NÃO é usada na previsão.
                "cor_entrada": item["cor"],
                "tipo_real": item["tipo_forcado"],
                "seed": item["seed"],
                "m1": m1,
                "m3": {**m3, "tentativas": tentativas, "bruto": bruto_txt},
                "tempo_seg": round(time.time() - inicio, 2),
            }
            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
            fout.flush()

            if (len(concluidas) + 1) % 25 == 0 or item is amostra[-1]:
                feitas = len(concluidas) + 1
                log(f"Progresso: {feitas}/{len(amostra)}")
            concluidas.add(item["ordem"])

    log(f"Concluido Modo B. Resultado em {OUT}")


if __name__ == "__main__":
    main()
