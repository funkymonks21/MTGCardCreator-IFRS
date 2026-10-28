#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner M3 - LLM pura (`gemma3:4b`) com schema restrito aos 6 tipos.

Segue o D-003 (defesa em profundidade):
  1. prompt com enum explícito dos 6 tipos;
  2. decodificação estruturada (`format` com JSON schema no Ollama) — é a
     camada que de fato impede tipo fora do enum;
  3. validação pós-geração com **1 retry** e registro de falha.

Segue o D-004 Modo A: a entrada é nome + cor + **tipo real**. O tipo é DADO
(não é a métrica do Modo A), então uma resposta com tipo diferente do pedido
conta como falha de geração daquela carta — não como "erro de conteúdo" a ser
mascarado. O tipo nunca é corrigido à força.

Uso:
    python experimentos/run_m3.py

Requer o Ollama no ar com o modelo baixado. `resultados_m3.jsonl` é
append-only e o runner retoma de onde parou.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request

from comum import (
    EXPERIMENTOS,
    RAIZ,
    carregar_itens,
    cmc_do_custo,
    criar_log,
    normalizar_custo,
    ordens_concluidas,
)

OUT = EXPERIMENTOS / "resultados_m3.jsonl"
LOG = EXPERIMENTOS / "log_m3.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

# Exatamente os 6 tipos do escopo (README §1.1). A ordem é a mesma do enum
# do schema, para prompt e gramática falarem a mesma língua.
TIPOS_VALIDOS = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land"]

# `type` do config -> valor do enum. Os tipos não previstos aqui não existem
# no config_v1.json (que só contém os 6 do escopo).
TIPO_PARA_ENUM = {t.lower(): t for t in TIPOS_VALIDOS}
TIPO_PARA_ENUM["land"] = "Land"

SCHEMA = {
    "type": "object",
    "required": [
        "name",
        "card_type",
        "mana_cost",
        "cmc",
        "subtypes",
        "keywords",
        "text",
        "power",
        "toughness",
    ],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 3, "maxLength": 60},
        "card_type": {"type": "string", "enum": TIPOS_VALIDOS},
        "mana_cost": {"type": "string"},
        "cmc": {"type": "number"},
        "subtypes": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "string", "minLength": 10},
        # `["string","null"]` fazia o gemma3:4b emitir lixo (`'{2}'`, `'<3'`,
        # `'/>2'`) ao alternar entre os dois ramos do union. `string` puro com
        # `maxLength` produz valores limpos; não-criatura usa "" (None depois).
        "power": {"type": "string", "maxLength": 5},
        "toughness": {"type": "string", "maxLength": 5},
    },
}

CORES_EN = {
    "W": "white",
    "U": "blue",
    "B": "black",
    "R": "red",
    "G": "green",
}

SISTEMA = (
    "Você é um gerador de cartas Magic: The Gathering para um experimento "
    "controlado. Retorne APENAS JSON válido seguindo o schema. "
    "Respeite ESTRITAMENTE o tipo pedido e os 6 tipos permitidos: Creature, "
    "Instant, Sorcery, Artifact, Enchantment, Land. "
    "Nunca invente outros tipos. "
    "Use nomes e texto de regras curtos e jogáveis, no estilo clássico de MTG. "
    "O campo mana_cost usa o formato Oracle, com cada símbolo entre chaves: "
    "por exemplo \"{2}{R}\" ou \"{1}{W}{W}\"; nunca escreva \"2R\". "
    "Terreno não tem custo de mana: use a string vazia \"\". "
    "Para criaturas, power e toughness são números inteiros simples, escritos "
    "como string (ex.: \"2\"), nunca símbolos de mana. "
    "Para qualquer não-criatura (instantânea, feitiço, artefato, encantamento, "
    "terreno), power e toughness devem ser a string vazia \"\"."
)


def instrucao_tipo(tipo: str) -> str:
    """Lembrete por tipo repetido no turno do usuário.

    O system prompt é diluído em 1000+ chamadas; sem este reforço o
    `gemma3:4b` preenchia `mana_cost` em **100%** dos terrenos (observado em
    6/6 sondagens) e a carta era rejeitada no lint. Isto é formato de saída,
    não ensino de mecânica: a regra já está no system prompt e no schema.
    """
    tipo = tipo.lower()
    if tipo == "land":
        return 'Este é um TERRENO: "mana_cost" DEVE ser exatamente "" (string vazia).'
    if tipo == "creature":
        return (
            'Esta é uma CRIATURA: "power" e "toughness" DEVEM ser inteiros '
            'escritos como string (ex.: "3" e "4").'
        )
    return 'Esta é uma NÃO-CRIATURA: "power" e "toughness" DEVEM ser "" (string vazia).'


def log_refinado(log, msg):
    log(f"  {msg}")


def _int_ou_none(valor) -> int | None:
    """Normaliza P/T: o modelo pode devolver 2, "2", "2*" ou null."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto:
        return None
    achado = re.search(r"-?\d+", texto)
    return int(achado.group()) if achado else None


def normalizar(carta: dict) -> dict:
    """Converte a saída crua do LLM para o formato de registro, sem
    'consertar' conteúdo: só normaliza tipos (P/T para int) e deriva o CMC do
    `mana_cost` (mesma regra do M1/M2, para a métrica ser comparável)."""
    poder = _int_ou_none(carta.get("power"))
    resistencia = _int_ou_none(carta.get("toughness"))
    custo = normalizar_custo(carta.get("mana_cost") or "")
    return {
        "nome_gerado": (carta.get("name") or "").strip(),
        "tipo": carta.get("card_type"),
        "tipo_linha": None,  # o LLM não devolve type_line; subtipos ficam abaixo
        "custo_mana": custo,
        "cmc": cmc_do_custo(custo),
        "cmc_declarado": carta.get("cmc"),
        "texto_regras": (carta.get("text") or "").strip(),
        "palavras_chave": [k for k in carta.get("keywords", []) if isinstance(k, str)],
        "poder": poder,
        "resistencia": resistencia,
        "subtipos": [s for s in carta.get("subtypes", []) if isinstance(s, str)],
    }


def validar(carta: dict, tipo_esperado: str) -> tuple[bool, str]:
    """Checagens ESTRUTURAIS (D-003 camada 3). Devolve (ok, motivo).

    Só rejeita o que impede comparar a carta com o gabarito: tipo fora do
    enum, tipo diferente do pedido, campos ausentes/incoerentes. Alucinação
    de mecânica NÃO rejeita — vira métrica na avaliação (decisão registrada:
    rejeitar esconderia a falha e inflaria o M3)."""
    if not isinstance(carta, dict):
        return False, "resposta não é objeto JSON"

    tipo = carta.get("card_type")
    if tipo not in TIPOS_VALIDOS:
        return False, f"card_type fora do enum: {tipo!r}"
    if tipo.lower() != tipo_esperado.lower():
        return False, f"tipo ignorado: pedido {tipo_esperado!r}, veio {tipo!r}"

    if not isinstance(carta.get("name"), str) or len(carta["name"].strip()) < 3:
        return False, "name inválido"
    if not isinstance(carta.get("text"), str) or len(carta["text"].strip()) < 10:
        return False, "text ausente/curto"
    if not isinstance(carta.get("subtypes"), list):
        return False, "subtypes não é lista"
    if not isinstance(carta.get("keywords"), list):
        return False, "keywords não é lista"

    poder = _int_ou_none(carta.get("power"))
    resistencia = _int_ou_none(carta.get("toughness"))
    tipo_min = tipo.lower()

    if tipo_min in {"instant", "sorcery"} and (poder is not None or resistencia is not None):
        return False, "instant/feitiço com P/T"
    if tipo_min == "creature" and (poder is None or resistencia is None):
        return False, "criatura sem P/T"
    if tipo_min == "land" and normalizar_custo(carta.get("mana_cost") or ""):
        return False, "terreno com custo de mana"

    return True, "ok"


def gerar_com_llm(nome: str, cor: str, tipo_esperado: str, seed: int) -> dict | None:
    """Uma chamada ao Ollama. Devolve o JSON decodificado ou None."""
    cor_en = CORES_EN.get(cor, str(cor).lower())
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SISTEMA},
            {
                "role": "user",
                "content": (
                    "Crie UMA carta de Magic.\n"
                    f"Cor: {cor_en}\n"
                    f"Tipo OBRIGATÓRIO: {tipo_esperado}\n"
                    f"Inspire-se no nome: {nome}\n"
                    f"{instrucao_tipo(tipo_esperado)}\n"
                    "Responda no formato JSON do schema."
                ),
            },
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 320, "seed": seed},
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
            return None
        return json.loads(conteudo)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"__erro__": str(exc)}


def main() -> None:
    log = criar_log(LOG)
    itens = carregar_itens()
    concluidas = ordens_concluidas(OUT)

    ok = 0
    falhas = 0
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for linha in f:
                try:
                    registro = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if registro.get("modelo") != "M3":
                    continue
                if "erro" in registro:
                    falhas += 1
                else:
                    ok += 1

    log(f"Iniciando M3: {len(itens)} cartas | {len(concluidas)} já concluídas")

    with open(OUT, "a", encoding="utf-8") as fout:
        for item in itens:
            if item["ordem"] in concluidas:
                continue
            inicio = time.time()
            tipo_esperado = TIPO_PARA_ENUM[item["tipo_forcado"].lower()]

            carta_llm = None
            motivo = "sem tentativa"
            tentativas = 0
            # 1 tentativa + 1 retry = no máximo 2 chamadas (D-003 camada 3).
            for tentativa in (1, 2):
                tentativas = tentativa
                bruta = gerar_com_llm(
                    item["nome"], item["cor"], tipo_esperado, item["seed"] + tentativa
                )
                if bruta is None:
                    motivo = "resposta vazia"
                    continue
                if "__erro__" in bruta:
                    motivo = f"erro de rede/JSON: {bruta['__erro__']}"
                    log_refinado(log, f"[M3] {item['nome']} tentativa {tentativa}: {motivo}")
                    continue
                valido, motivo = validar(bruta, tipo_esperado)
                if valido:
                    carta_llm = bruta
                    break
                log_refinado(
                    log, f"[M3] {item['nome']} tentativa {tentativa}: {motivo}"
                )

            if carta_llm is None:
                falhas += 1
                fout.write(
                    json.dumps(
                        {
                            "modelo": "M3",
                            "ordem": item["ordem"],
                            "carta_id": item["carta_id"],
                            "tipo_experimento": item["tipo_experimento"],
                            "nome_entrada": item["nome"],
                            "cor_entrada": item["cor"],
                            "tipo_forcado": item["tipo_forcado"],
                            "erro": motivo,
                            "tentativas": tentativas,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
            else:
                ok += 1
                fout.write(
                    json.dumps(
                        {
                            "modelo": "M3",
                            "ordem": item["ordem"],
                            "carta_id": item["carta_id"],
                            "tipo_experimento": item["tipo_experimento"],
                            "nome_entrada": item["nome"],
                            "cor_entrada": item["cor"],
                            "tipo_forcado": item["tipo_forcado"],
                            "seed": item["seed"],
                            "tentativas": tentativas,
                            **normalizar(carta_llm),
                            "carta_llm_bruta": carta_llm,
                            "gabarito": item["gabarito"],
                            "tempo_seg": round(time.time() - inicio, 2),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

            if (item["ordem"] + 1) % 10 == 0:
                log(f"Progresso: {item['ordem'] + 1}/{len(itens)} | ok={ok} falhas={falhas}")

    log(f"Concluido M3. ok={ok} falhas={falhas}. Resultado em {OUT}")


if __name__ == "__main__":
    main()
