#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Braços M2 e M3 do estudo, reutilizáveis pela interface.

Por que este módulo existe
==========================
O estudo compara três formas de gerar a MESMA carta:

- **M1** — o pipeline (`pipeline.generator.gerar_carta`): Markov + regressão.
- **M2** — o M1 com o **texto de regras** refinado por LLM. Tipo, subtipos,
  custo, CMC, P/T e palavras-chave continuam vindo do pipeline; só
  `texto_regras` é substituído. É o que torna o contraste M1 × M2 um teste
  do texto, e não da estrutura.
- **M3** — LLM pura (`gemma3:4b` via Ollama), com o schema restrito aos seis
  tipos do escopo (D-003).

Os prompts, temperaturas, schemas e a validação são **cópias fiéis** de
`experimentos/run_m2.py` e `experimentos/run_m3.py`. A demo e o experimento
precisam falar a mesma língua: se o prompt da interface fosse "melhorado", o
que aparece na apresentação deixaria de ser o que foi medido.

Pré-requisitos para M2/M3
=========================
Ollama no ar (`ollama serve`) com o modelo baixado:

    ollama pull gemma3:4b

A URL e o modelo podem ser trocados por ambiente:

    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=gemma3:4b

No Colab, o notebook `notebooks/app_colab.ipynb` sobe o Ollama no próprio
runtime. Sem Ollama (ou sem endpoint configurado), as funções levantam
`OllamaIndisponivel` com uma mensagem explicando o que falta. O M1 funciona em
qualquer lugar.

Uso
===
    from pipeline.llm import gerar_com_modelo, ollama_disponivel

    carta = gerar_com_modelo("M2", "Serra Angel", "Branco", seed=42)
    carta = gerar_com_modelo("M3", "Serra Angel", "Branco", seed=42)

    # refinar uma carta já gerada (o botão "Refinar texto" do app):
    from pipeline.llm import refinar_como_m2

    refinada = refinar_como_m2(carta_m1)
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Optional

from .generator import (
    PESO_COR_PADRAO,
    Carta,
    gerar_carta,
    renderizar_carta,
    resolver_tipo,
)
from .tipos import TIPOS_DISPLAY

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODELO_LLM = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
ENDPOINT_CHAT = f"{OLLAMA_URL}/api/chat"

MODELOS = ("M1", "M2", "M3")
ROTULO_MODELO = {
    "M1": "M1 — pipeline (Markov + regressão)",
    "M2": "M2 — pipeline + texto refinado por LLM",
    "M3": "M3 — LLM pura",
}

# Tipo interno do pipeline -> valor do enum do schema do M3 (D-003).
TIPO_PARA_ENUM = {
    "creature": "Creature",
    "instant": "Instant",
    "sorcery": "Sorcery",
    "artifact": "Artifact",
    "enchantment": "Enchantment",
    "land": "Land",
}
TIPOS_VALIDOS = list(TIPO_PARA_ENUM.values())

CORES_EN = {
    "Branco": "white",
    "Azul": "blue",
    "Preto": "black",
    "Vermelho": "red",
    "Verde": "green",
}


class OllamaIndisponivel(RuntimeError):
    """O Ollama não respondeu. A mensagem já diz o que fazer."""


class FalhaGeracao(RuntimeError):
    """O M3 não produziu uma carta válida nem depois do retry (D-003)."""


# ---------------------------------------------------------------------------
# M2 — refino do texto de regras
# ---------------------------------------------------------------------------

# Prompt deliberadamente restrito: mexe só no texto de regras. Instruções
# explícitas para não tocar em nome/tipo/custo/P-T/keywords, porque o LLM
# tende a "melhorar" a carta inteira se não for contido.
SISTEMA_M2 = (
    "Você é um editor de regras de Magic: The Gathering. "
    "Sua ÚNICA tarefa é refinar o TEXTO DE REGRAS desta carta. "
    "NUNCA mude nome, tipo, subtipo, custo de mana, poder, resistência, "
    "lealdade ou palavras-chave. "
    "Mantenha o número de parágrafos, a mecânica e o sentido originais. "
    "Use terminologia correta de MTG, seja conciso e direto. "
    "Retorne APENAS o texto de regras refinado, sem explicações, sem JSON, "
    "sem markdown."
)


def refinar_texto(
    texto: str, tipo_linha: str, seed: int = 0, avisar=print
) -> Optional[str]:
    """Devolve o texto refinado pelo LLM, ou None se a chamada falhar.

    Falha aqui **não** invalida a carta: o M2 mantém o texto do Markov e
    registra `refinado=False`. É a mesma regra do runner do estudo, para que
    a demo não esconda a taxa de falha do refino.
    """
    if not texto or not texto.strip():
        return None

    payload = {
        "model": MODELO_LLM,
        "messages": [
            {"role": "system", "content": SISTEMA_M2},
            {
                "role": "user",
                "content": f"TIPO: {tipo_linha}\n\nTEXTO ATUAL:\n{texto}\n\nTEXTO REFINADO:",
            },
        ],
        "stream": False,
        # `seed` no Ollama torna a amostragem reproduzível para o mesmo input.
        "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 256, "seed": seed},
    }
    try:
        resposta = _post_chat(payload)
    except OllamaIndisponivel:
        raise
    except (TimeoutError, json.JSONDecodeError) as exc:
        avisar(f"Refino falhou ({exc}); mantendo o texto do pipeline.")
        return None

    refinado = resposta.get("message", {}).get("content", "").strip()
    if not refinado:
        avisar("Refino devolveu texto vazio; mantendo o texto do pipeline.")
        return None
    # Alguns modelos devolvem o texto entre aspas ou com cercas de código.
    if refinado.startswith('"') and refinado.endswith('"') and len(refinado) > 1:
        refinado = refinado[1:-1].strip()
    if refinado.startswith("```"):
        linhas = [l for l in refinado.splitlines() if not l.strip().startswith("```")]
        refinado = "\n".join(linhas).strip()
    return refinado or None


def gerar_carta_m2(
    nome: str,
    cor: str,
    peso_cor: float = PESO_COR_PADRAO,
    tipo: Optional[str] = None,
    pesos_ngrama=None,
    n_paragrafos: Optional[int] = None,
    usar_arte: bool = False,
    seed_arte: Optional[int] = None,
    avisar_arte=print,
    seed: Optional[int] = None,
) -> Carta:
    """M2 = M1 com o `texto_regras` refinado.

    A estrutura (tipo, subtipos, custo, P/T, keywords) é a do pipeline, com a
    mesma semente. Depois do refino a imagem é re-renderizada, porque o texto
    mudou — inclusive a arte, quando pedida.
    """
    carta = gerar_carta(
        nome,
        cor,
        peso_cor=peso_cor,
        tipo=tipo,
        pesos_ngrama=pesos_ngrama,
        n_paragrafos=n_paragrafos,
        usar_arte=False,
        seed=seed,
    )
    texto_pipeline = carta.texto_regras
    texto_refinado = refinar_texto(
        texto_pipeline, carta.tipo_linha, seed=seed or 0, avisar=avisar_arte
    )
    refinado = bool(texto_refinado) and texto_refinado != texto_pipeline
    if refinado:
        carta.texto_regras = texto_refinado
    carta.modelo = "M2"
    carta.refinado = refinado
    _renderizar(carta, cor, usar_arte, seed_arte, avisar_arte)
    return carta


# ---------------------------------------------------------------------------
# M3 — LLM pura
# ---------------------------------------------------------------------------

# Schema do D-003. `power`/`toughness` são `string` puro (e não
# ["string","null"]): o gemma3:4b emitia lixo (`'{2}'`, `'<3'`, `'/>2'`) ao
# alternar entre os ramos do union. Não-criatura usa "" (None depois).
SCHEMA_M3 = {
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
        "power": {"type": "string", "maxLength": 5},
        "toughness": {"type": "string", "maxLength": 5},
    },
}

SISTEMA_M3 = (
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
    """Lembrete por tipo repetido no turno do usuário (ver run_m3.py).

    O system prompt é diluído em muitas chamadas; sem este reforço o
    `gemma3:4b` preenchia `mana_cost` em **100%** dos terrenos. É formato de
    saída, não ensino de mecânica.
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


def _int_ou_none(valor) -> Optional[int]:
    """Normaliza P/T: o modelo pode devolver 2, "2", "2*" ou null."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto:
        return None
    achado = re.search(r"-?\d+", texto)
    return int(achado.group()) if achado else None


_SIMBOLO_MANA = re.compile(r"\{([^}]*)\}")


def _normalizar_custo(custo_mana: str) -> str:
    """Converte notações compactas (`2W`, `10WU`) para o formato Oracle.

    Cópia da regra de `experimentos/comum.py`: só acrescenta as chaves, não
    mexe nos símbolos. Sem isso, `2W` viraria CMC 0 e o terreno seria
    validado errado — um artefato de formato, não de conteúdo.
    """
    texto = (custo_mana or "").strip()
    if not texto:
        return ""
    if texto.lower() in {"none", "null", "n/a", "na", "-", "—", "sem custo", "no cost"}:
        return ""
    if "{" in texto:
        return re.sub(r"\s+", "", texto)
    simbolos = re.findall(r"\d+|[WUBRGX]", texto.upper())
    if not simbolos or "".join(simbolos) != re.sub(r"[\s,;/]+", "", texto.upper()):
        return texto
    return "".join(f"{{{s}}}" for s in simbolos)


def validar(carta: dict, tipo_esperado: str) -> tuple:
    """Checagens ESTRUTURAIS do D-003. Devolve `(ok, motivo)`.

    Só rejeita o que impede comparar a carta com o pedido: tipo fora do enum,
    tipo diferente do pedido, campos ausentes/incoerentes. Alucinação de
    mecânica NÃO rejeita — ela vira métrica na avaliação.
    """
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
    if tipo_min == "land" and _normalizar_custo(carta.get("mana_cost") or ""):
        return False, "terreno com custo de mana"

    return True, "ok"


def _pedir_carta_llm(nome: str, cor: str, tipo_en: str, seed: int) -> dict:
    """Uma chamada ao Ollama. Devolve o JSON decodificado.

    Levanta `OllamaIndisponivel` se o servidor não responder; devolve
    `{"__erro__": ...}` para falhas de JSON, que o retry trata.
    """
    cor_en = CORES_EN.get(cor, str(cor).lower())
    payload = {
        "model": MODELO_LLM,
        "messages": [
            {"role": "system", "content": SISTEMA_M3},
            {
                "role": "user",
                "content": (
                    "Crie UMA carta de Magic.\n"
                    f"Cor: {cor_en}\n"
                    f"Tipo OBRIGATÓRIO: {tipo_en}\n"
                    f"Inspire-se no nome: {nome}\n"
                    f"{instrucao_tipo(tipo_en)}\n"
                    "Responda no formato JSON do schema."
                ),
            },
        ],
        "format": SCHEMA_M3,
        "stream": False,
        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 320, "seed": seed},
    }
    try:
        resposta = _post_chat(payload)
    except (TimeoutError, json.JSONDecodeError) as exc:
        return {"__erro__": str(exc)}
    conteudo = resposta.get("message", {}).get("content", "").strip()
    if not conteudo:
        return {"__erro__": "resposta vazia"}
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError as exc:
        return {"__erro__": f"JSON inválido: {exc}"}


def _post_chat(payload: dict, timeout: int = 300) -> dict:
    """POST em `/api/chat`. Só falha de rede vira `OllamaIndisponivel`."""
    requisicao = urllib.request.Request(
        ENDPOINT_CHAT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=timeout) as resposta:
            return json.loads(resposta.read())
    except urllib.error.HTTPError as exc:
        # HTTPError é subclasse de URLError: precisa vir antes. O caso comum é
        # 404 quando o modelo não foi baixado.
        detalhe = ""
        try:
            detalhe = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001 - mensagem de erro é best-effort
            pass
        raise OllamaIndisponivel(
            f"O Ollama respondeu HTTP {exc.code} em {ENDPOINT_CHAT}. "
            f"Confira se o modelo existe: `ollama pull {MODELO_LLM}`. "
            f"({detalhe})"
        ) from exc
    except urllib.error.URLError as exc:
        raise OllamaIndisponivel(
            f"Ollama não respondeu em {OLLAMA_URL}. Inicie com `ollama serve` "
            f"e baixe o modelo com `ollama pull {MODELO_LLM}`. "
            f"(detalhe: {exc})"
        ) from exc


def gerar_carta_m3(
    nome: str,
    cor: str,
    tipo: Optional[str] = None,
    seed: Optional[int] = None,
    usar_arte: bool = False,
    seed_arte: Optional[int] = None,
    avisar_arte=print,
) -> Carta:
    """M3 = LLM pura, com 1 tentativa + 1 retry (D-003).

    O tipo é resolvido pelo **pipeline** (mesma regra do M1) e imposto ao
    LLM. Não é "ensinar Magic" ao modelo: é o desenho do Modo A do estudo,
    em que tipo, nome e cor são DADOS e o LLM gera o resto. Sem isso, o demo
    compararia cartas de tipos diferentes.

    Levanta `FalhaGeracao` se as duas tentativas forem inválidas.

    Sem semente (`None`), sorteia uma: o LLM é estocástico, e fixar 0 aqui
    faria a demo devolver sempre a mesma carta. A semente sorteada aparece em
    `Carta.semente_usada`, então o resultado continua reproduzível.
    """
    if cor not in CORES_EN:
        raise ValueError(f"Cor inválida: {cor!r}. Use uma de {list(CORES_EN)}.")
    nome = (nome or "").strip() or "Carta Sem Nome"
    seed_base = random.randrange(1, 2**31 - 1) if seed is None else int(seed)
    tipo_chave, tipo_origem, tipo_motivo = resolver_tipo(nome, tipo, seed=seed_base)
    tipo_en = TIPO_PARA_ENUM[tipo_chave]

    carta_llm = None
    motivo = "sem tentativa"
    tentativas = 0
    for tentativa in (1, 2):
        tentativas = tentativa
        bruta = _pedir_carta_llm(nome, cor, tipo_en, seed_base + tentativa)
        if "__erro__" in bruta:
            motivo = f"erro de rede/JSON: {bruta['__erro__']}"
            continue
        valido, motivo = validar(bruta, tipo_en)
        if valido:
            carta_llm = bruta
            break

    if carta_llm is None:
        raise FalhaGeracao(
            f"O M3 não produziu carta válida para {nome!r} em {tentativas} "
            f"tentativa(s): {motivo}"
        )

    custo = _normalizar_custo(carta_llm.get("mana_cost") or "")
    subtipos = [s for s in carta_llm.get("subtypes", []) if isinstance(s, str)]
    poder = _int_ou_none(carta_llm.get("power"))
    resistencia = _int_ou_none(carta_llm.get("toughness"))
    texto = (carta_llm.get("text") or "").strip()
    tipo_linha = TIPO_PARA_ENUM[tipo_chave]
    if subtipos:
        # "Creature — human soldier" -> "Creature — Human Soldier"
        tipo_linha += " — " + " ".join(s[:1].upper() + s[1:] for s in subtipos)

    carta = Carta(
        nome=(carta_llm.get("name") or nome).strip(),
        cor=cor,
        custo_mana=custo,
        tipo=TIPOS_DISPLAY[tipo_chave],
        texto_regras=texto,
        imagem_final=None,  # preenchida por _renderizar
        tipo_linha=tipo_linha,
        subtipos=subtipos,
        poder=poder,
        resistencia=resistencia,
        palavras_chave=[k for k in carta_llm.get("keywords", []) if isinstance(k, str)],
        n_paragrafos=len([p for p in texto.split("\n\n") if p.strip()]),
        tipo_origem="llm",
        tipo_motivo=f"tipo {tipo_en} imposto ao LLM (resolvido por: {tipo_origem} — {tipo_motivo})",
        semente_usada=f"LLM seed={seed_base + tentativas}",
        modelo="M3",
    )
    _renderizar(carta, cor, usar_arte, seed_arte, avisar_arte)
    return carta


# ---------------------------------------------------------------------------
# Dispatch usado pela interface
# ---------------------------------------------------------------------------


def gerar_com_modelo(
    modelo: str,
    nome: str,
    cor: str,
    peso_cor: float = PESO_COR_PADRAO,
    tipo: Optional[str] = None,
    pesos_ngrama=None,
    n_paragrafos: Optional[int] = None,
    usar_arte: bool = False,
    seed_arte: Optional[int] = None,
    avisar_arte=print,
    seed: Optional[int] = None,
) -> Carta:
    """Gera a carta no braço escolhido ("M1", "M2" ou "M3").

    Os parâmetros de texto do M1 (`peso_cor`, `pesos_ngrama`,
    `n_paragrafos`) só têm efeito no M1 e no M2. No M3, o texto vem inteiro
    do LLM — os sliders ficam inertes, e a interface avisa isso.
    """
    modelo = (modelo or "M1").strip().upper()
    if modelo == "M1":
        return gerar_carta(
            nome,
            cor,
            peso_cor=peso_cor,
            tipo=tipo,
            pesos_ngrama=pesos_ngrama,
            n_paragrafos=n_paragrafos,
            usar_arte=usar_arte,
            seed_arte=seed_arte,
            avisar_arte=avisar_arte,
            seed=seed,
        )
    if modelo == "M2":
        return gerar_carta_m2(
            nome,
            cor,
            peso_cor=peso_cor,
            tipo=tipo,
            pesos_ngrama=pesos_ngrama,
            n_paragrafos=n_paragrafos,
            usar_arte=usar_arte,
            seed_arte=seed_arte,
            avisar_arte=avisar_arte,
            seed=seed,
        )
    if modelo == "M3":
        return gerar_carta_m3(
            nome,
            cor,
            tipo=tipo,
            seed=seed,
            usar_arte=usar_arte,
            seed_arte=seed_arte,
            avisar_arte=avisar_arte,
        )
    raise ValueError(f"Modelo inválido: {modelo!r}. Use um de {list(MODELOS)}.")


def _renderizar(carta: Carta, cor: str, usar_arte: bool, seed_arte, avisar_arte) -> None:
    """Re-renderiza a imagem depois que o texto da carta mudou."""
    renderizar_carta(
        carta, usar_arte=usar_arte, seed_arte=seed_arte, avisar_arte=avisar_arte
    )


def refinar_como_m2(
    carta: Carta, seed: Optional[int] = None, avisar=print
) -> Carta:
    """Aplica o refino do M2 a uma carta JÁ gerada (tipicamente do M1).

    Diferente de `gerar_carta_m2`, não regera a estrutura: usa exatamente o
    nome, tipo, custo, P/T e texto da carta recebida e troca só
    `texto_regras`. É o "refinar esta carta" da demo -- o usuário vê o texto
    do M1 e decide se manda o LLM reescrevê-lo.

    Devolve uma cópia (a carta original fica intacta) com `modelo="M2"`;
    se o refino falhar, a cópia mantém o texto do pipeline e `refinado=False`,
    a mesma regra do runner do estudo.
    """
    if carta.modelo == "M3":
        raise ValueError(
            "O refino do M2 é do pipeline; cartas do M3 já vêm inteiras do LLM."
        )
    nova = replace(carta)
    texto = refinar_texto(
        nova.texto_regras,
        nova.tipo_linha or nova.tipo,
        seed=0 if seed is None else int(seed),
        avisar=avisar,
    )
    nova.refinado = bool(texto) and texto != nova.texto_regras
    if nova.refinado:
        nova.texto_regras = texto
    nova.modelo = "M2"
    # Mantém a decisão de arte da carta original: quem saiu sem arte continua
    # sem arte (o botão "Gerar arte" do app é que liga a ilustração).
    renderizar_carta(
        nova, usar_arte=nova.arte_da_api is not None, avisar_arte=avisar
    )
    return nova


def ollama_disponivel(timeout: float = 1.5) -> bool:
    """True se o Ollama responde em `OLLAMA_URL` (não checa o modelo)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as resposta:
            return resposta.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
