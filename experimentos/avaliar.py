#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Avaliação dos três braços do estudo (M1, M2, M3) contra o gabarito.

O que este script faz
=====================
Lê os três `.jsonl` produzidos pelos runners e calcula as métricas do
`README.md` desta pasta, separando sempre **cartas reais** (têm gabarito) de
**cartas fictícias** (não têm — servem para criatividade/coerência e como
controle de memorização).

Blocos de métricas
------------------
1. **Estruturais** (só reais): CMC (MAE/RMSE/exato/Spearman), P/T (MAE/exato),
   palavras-chave (Jaccard/exato), coerência cor × custo, validade de terreno.
2. **Texto** (todas): tamanho, diversidade lexical (distinct-1/2), novidade de
   5-gramas contra o corpus, memória (Jaccard de 5-gramas com a carta real
   pareada) e um lint de regras explícito.
3. **M3**: consistência interna do CMC declarado × derivado, taxa de retry e
   taxa de falha de geração.
4. **Comparações pareadas**: M1 × M2 (mesma semente, então a estrutura TEM de
   ser idêntica — serve de teste de integridade), M1/M2 × M3 (bootstrap
   pareado para CMC, McNemar para acerto exato).

Saídas
------
- `experimentos/metricas.json` — todos os números, formato máquina.
- `experimentos/relatorio_metricas.md` — tabelas em português.

Uso:
    python experimentos/avaliar.py
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
EXPERIMENTOS = RAIZ / "experimentos"
CORPUS = RAIZ / "data" / "processed" / "cards.parquet"

sys.path.insert(0, str(EXPERIMENTOS))
from comum import cmc_do_custo, normalizar_custo  # noqa: E402

ARQUIVOS = {
    "M1": EXPERIMENTOS / "resultados_m1.jsonl",
    "M2": EXPERIMENTOS / "resultados_m2.jsonl",
    "M3": EXPERIMENTOS / "resultados_m3.jsonl",
}

# `Carta.tipo` (pipeline) -> `type` do config/Gabarito (Scryfall).
TIPO_PT = {
    "Criatura": "creature",
    "Instantânea": "instant",
    "Feitiço": "sorcery",
    "Artefato": "artifact",
    "Encantamento": "enchantment",
    "Terreno": "land",
}

COR_PIP = {"W": "{W}", "U": "{U}", "B": "{B}", "R": "{R}", "G": "{G}"}

# Tipos fora do escopo: aparecer no TEXTO é sinal de alucinação de mecânica.
TIPOS_PROIBIDOS = re.compile(
    r"\b(planeswalker|battle|commander|vanguard|conspiracy|phenomenon|"
    r"planechase|scheme|dungeon)\b",
    re.IGNORECASE,
)

PALAVRA = re.compile(r"[a-zà-ÿ0-9']+")


# --------------------------------------------------------------------------
# Carregamento e normalização
# --------------------------------------------------------------------------
def carregar(modelo: str) -> Tuple[List[dict], List[dict]]:
    """Devolve (registros_ok, registros_com_erro)."""
    caminho = ARQUIVOS[modelo]
    ok: List[dict] = []
    erros: List[dict] = []
    if not caminho.exists():
        return ok, erros
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            (erros if "erro" in registro else ok).append(registro)
    return ok, erros


def tipo_normalizado(registro: dict) -> Optional[str]:
    tipo = registro.get("tipo")
    if not tipo:
        return None
    tipo = str(tipo)
    if tipo in TIPO_PT:
        return TIPO_PT[tipo]
    return tipo.lower()


def tokens(texto: str) -> List[str]:
    return PALAVRA.findall((texto or "").lower())


def ngramas(toks: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(toks) - n + 1):
        yield tuple(toks[i : i + n])


def hash_ngramas(toks: Sequence[str], n: int) -> set:
    """Conjunto de hashes dos n-gramas.

    `hash()` é salgado por processo, mas aqui os dois lados (corpus e saída)
    são calculados no MESMO processo, então a comparação é válida. Usamos
    CRC-32 para não depender nem disso.
    """
    return {
        zlib.crc32(" ".join(grama).encode("utf-8"))
        for grama in ngramas(toks, n)
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def inteiro_ou_none(valor) -> Optional[int]:
    if valor is None:
        return None
    achado = re.search(r"-?\d+", str(valor))
    return int(achado.group()) if achado else None


# --------------------------------------------------------------------------
# Vocabulário de referência (corpus)
# --------------------------------------------------------------------------
def vocabulario_corpus() -> Tuple[set, set, Dict[str, int]]:
    """(palavras, 5-gramas hasheados, vocabulário de keywords)."""
    df = pd.read_parquet(CORPUS, columns=["oracle_text", "keywords", "type_line"])
    palavras = set()
    gramas = set()
    for texto in df["oracle_text"].fillna(""):
        toks = tokens(texto)
        palavras.update(toks)
        gramas |= hash_ngramas(toks, 5)
    keywords = set()
    for lista in df["keywords"]:
        if lista is None:
            continue
        for kw in lista:
            if isinstance(kw, str) and kw.strip():
                keywords.add(kw.strip().lower())
    return palavras, gramas, keywords


# --------------------------------------------------------------------------
# Lint de regras (explícito e auditável)
# --------------------------------------------------------------------------
def lint(registro: dict, cor_entrada: str) -> List[str]:
    """Devolve a lista de falhas DURAS de estrutura/coerência.

    Só entram aqui regras que qualquer carta de MTG violaria — não julgamento
    estético. Alucinação de mecânica fica em `alertas`, porque é métrica de
    conteúdo, não motivo de descarte.
    """
    falhas: List[str] = []
    tipo = tipo_normalizado(registro)
    custo = normalizar_custo(registro.get("custo_mana") or "")
    texto = registro.get("texto_regras") or ""
    poder = inteiro_ou_none(registro.get("poder"))
    resistencia = inteiro_ou_none(registro.get("resistencia"))

    if not texto.strip():
        falhas.append("texto_vazio")
    if tipo == "land":
        if custo:
            falhas.append("terreno_com_custo")
        if poder is not None or resistencia is not None:
            falhas.append("terreno_com_pt")
    if tipo in {"instant", "sorcery"} and (poder is not None or resistencia is not None):
        falhas.append("instant_feitico_com_pt")
    if tipo == "creature" and (poder is None or resistencia is None):
        falhas.append("criatura_sem_pt")
    if tipo != "land" and cor_entrada in COR_PIP and custo and COR_PIP[cor_entrada] not in custo:
        falhas.append("custo_sem_cor")
    return falhas


def alertas(registro: dict) -> List[str]:
    """Sinais de conteúdo que NÃO invalidam a carta (viram métrica)."""
    avisos: List[str] = []
    texto = registro.get("texto_regras") or ""
    keywords = registro.get("palavras_chave") or []
    minusculo = texto.lower()

    if TIPOS_PROIBIDOS.search(texto):
        avisos.append("tipo_proibido_no_texto")
    for kw in keywords:
        if isinstance(kw, str) and kw.strip() and kw.strip().lower() not in minusculo:
            avisos.append("keyword_ausente_no_texto")
            break
    return avisos


# --------------------------------------------------------------------------
# Métricas por modelo
# --------------------------------------------------------------------------
def resumo_len(valores: Sequence[float]) -> Dict[str, float]:
    if not valores:
        return {"n": 0, "media": None, "mediana": None, "desvio": None}
    arr = np.asarray(valores, dtype=float)
    return {
        "n": int(arr.size),
        "media": float(arr.mean()),
        "mediana": float(np.median(arr)),
        "desvio": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def metricas_modelo(
    modelo: str,
    registros: List[dict],
    erros: List[dict],
    gramas_corpus: set,
    vocab_keywords: set,
    oracle_por_nome: Dict[str, str],
) -> Dict[str, object]:
    reais = [r for r in registros if r.get("tipo_experimento") == "real"]
    ficticias = [r for r in registros if r.get("tipo_experimento") == "ficticio"]
    todas = registros

    # ---------- CMC ----------
    cmc_erros, cmc_exatos = [], []
    cmc_gerado, cmc_gabarito = [], []
    for r in reais:
        g = r.get("gabarito") or {}
        alvo = g.get("cmc")
        gerado = r.get("cmc")
        if alvo is None or gerado is None:
            continue
        alvo = float(alvo)
        cmc_erros.append(abs(float(gerado) - alvo))
        cmc_exatos.append(int(round(float(gerado))) == int(round(alvo)))
        cmc_gerado.append(float(gerado))
        cmc_gabarito.append(alvo)

    spearman = None
    if len(cmc_gerado) > 2 and len(set(cmc_gerado)) > 1 and len(set(cmc_gabarito)) > 1:
        spearman = float(stats.spearmanr(cmc_gerado, cmc_gabarito).statistic)

    # ---------- P/T (só criaturas reais) ----------
    pt_poder, pt_res, pt_exatos = [], [], []
    for r in reais:
        if r.get("tipo_forcado") != "creature":
            continue
        g = r.get("gabarito") or {}
        alvo_p = inteiro_ou_none(g.get("power"))
        alvo_r = inteiro_ou_none(g.get("toughness"))
        ger_p = inteiro_ou_none(r.get("poder"))
        ger_r = inteiro_ou_none(r.get("resistencia"))
        if None in (alvo_p, alvo_r, ger_p, ger_r):
            continue
        pt_poder.append(abs(ger_p - alvo_p))
        pt_res.append(abs(ger_r - alvo_r))
        pt_exatos.append(ger_p == alvo_p and ger_r == alvo_r)

    # ---------- Keywords ----------
    kw_jaccard, kw_exato = [], []
    kw_jaccard_nv, kw_exato_nv = [], []
    ambos_vazios = 0
    kw_declaradas, kw_validas, kw_no_texto = 0, 0, 0
    for r in reais:
        g = r.get("gabarito") or {}
        alvo = {str(k).strip().lower() for k in (g.get("keywords") or [])}
        gerado = {str(k).strip().lower() for k in (r.get("palavras_chave") or [])}
        kw_jaccard.append(jaccard(gerado, alvo))
        iguais = gerado == alvo
        kw_exato.append(iguais)
        if not gerado and not alvo:
            ambos_vazios += 1
        else:
            # Sem os dois-vazios o "exato" não é inflado por cartas sem keyword.
            kw_jaccard_nv.append(jaccard(gerado, alvo))
            kw_exato_nv.append(iguais)
    for r in todas:
        texto = (r.get("texto_regras") or "").lower()
        for kw in r.get("palavras_chave") or []:
            if not isinstance(kw, str):
                continue
            kw_declaradas += 1
            if kw.strip().lower() in vocab_keywords:
                kw_validas += 1
            if kw.strip().lower() in texto:
                kw_no_texto += 1

    # ---------- Texto ----------
    tamanhos = [len(tokens(r.get("texto_regras") or "")) for r in todas]
    todas_palavras: List[str] = []
    gramas_modelo = set()
    for r in todas:
        toks = tokens(r.get("texto_regras") or "")
        todas_palavras.extend(toks)
        gramas_modelo |= hash_ngramas(toks, 5)
    total_palavras = len(todas_palavras)
    distinct1 = len(set(todas_palavras)) / total_palavras if total_palavras else 0.0
    bigramas = Counter()
    for r in todas:
        for grama in ngramas(tokens(r.get("texto_regras") or ""), 2):
            bigramas[zlib.crc32(" ".join(grama).encode("utf-8"))] += 1
    distinct2 = len(bigramas) / sum(bigramas.values()) if bigramas else 0.0

    # Novidade: fração dos 5-gramas do modelo que NÃO existem no corpus.
    if gramas_modelo:
        novidade = 1.0 - len(gramas_modelo & gramas_corpus) / len(gramas_modelo)
    else:
        novidade = 0.0

    # Memória: Jaccard de 5-gramas com a carta real pareada (quando há texto).
    memoria = []
    for r in reais:
        texto_real = oracle_por_nome.get(r.get("nome_entrada", ""))
        if not texto_real:
            continue
        a = hash_ngramas(tokens(r.get("texto_regras") or ""), 5)
        b = hash_ngramas(tokens(texto_real), 5)
        memoria.append(jaccard(a, b))

    # ---------- Lint e alertas ----------
    falhas_por_tipo = Counter()
    com_falha = 0
    for r in todas:
        f = lint(r, r.get("cor_entrada", ""))
        if f:
            com_falha += 1
            falhas_por_tipo.update(f)
    alertas_por_tipo = Counter()
    for r in todas:
        alertas_por_tipo.update(alertas(r))

    # Textos duplicados dentro do braço.
    contagem_textos = Counter((r.get("texto_regras") or "").strip() for r in todas)
    duplicados = sum(1 for t, c in contagem_textos.items() if c > 1 and t)
    duplicados_top = [
        {"texto": t, "vezes": c}
        for t, c in contagem_textos.most_common(5)
        if c > 1 and t
    ]
    por_tipo = Counter(r.get("tipo_forcado") for r in todas)
    erros_por_tipo = Counter(r.get("tipo_forcado") for r in erros)

    # ---------- M3: consistência do CMC declarado ----------
    cmc_declarado_ok = None
    if modelo == "M3":
        pares = [
            (float(r.get("cmc_declarado")), float(r.get("cmc")))
            for r in todas
            if r.get("cmc_declarado") is not None and r.get("cmc") is not None
        ]
        if pares:
            cmc_declarado_ok = sum(
                1 for declarado, derivado in pares
                if int(round(declarado)) == int(round(derivado))
            ) / len(pares)

    tentativas = Counter(r.get("tentativas") for r in todas)

    return {
        "modelo": modelo,
        "n_registros": len(todas),
        "n_erros": len(erros),
        "n_reais": len(reais),
        "n_ficticias": len(ficticias),
        "taxa_falha_geracao": len(erros) / (len(todas) + len(erros)) if (todas or erros) else 0.0,
        "cmc": {
            "n": len(cmc_erros),
            "mae": float(np.mean(cmc_erros)) if cmc_erros else None,
            "rmse": float(math.sqrt(np.mean(np.square(cmc_erros)))) if cmc_erros else None,
            "vies": float(np.mean(np.array(cmc_gerado) - np.array(cmc_gabarito))) if cmc_erros else None,
            "exato": float(np.mean(cmc_exatos)) if cmc_exatos else None,
            "spearman": spearman,
        },
        "pt": {
            "n": len(pt_poder),
            "mae_poder": float(np.mean(pt_poder)) if pt_poder else None,
            "mae_resistencia": float(np.mean(pt_res)) if pt_res else None,
            "exato": float(np.mean(pt_exatos)) if pt_exatos else None,
        },
        "keywords": {
            "jaccard_medio": float(np.mean(kw_jaccard)) if kw_jaccard else None,
            "exato": float(np.mean(kw_exato)) if kw_exato else None,
            "ambos_vazios": ambos_vazios,
            "jaccard_nao_vazios": float(np.mean(kw_jaccard_nv)) if kw_jaccard_nv else None,
            "exato_nao_vazios": float(np.mean(kw_exato_nv)) if kw_exato_nv else None,
            "declaradas": kw_declaradas,
            "taxa_valida_corpus": (kw_validas / kw_declaradas) if kw_declaradas else None,
            "taxa_no_texto": (kw_no_texto / kw_declaradas) if kw_declaradas else None,
        },
        "texto": {
            "palavras": resumo_len(tamanhos),
            "distinct1": distinct1,
            "distinct2": distinct2,
            "novidade_5grama": novidade,
            "memoria_jaccard_media": float(np.mean(memoria)) if memoria else None,
            "memoria_jaccard_max": float(np.max(memoria)) if memoria else None,
            "textos_duplicados": duplicados,
            "duplicados_top": duplicados_top,
        },
        "lint": {
            "cartas_com_falha": com_falha,
            "taxa_ok": 1 - com_falha / len(todas) if todas else None,
            "por_regra": dict(falhas_por_tipo),
        },
        "alertas": dict(alertas_por_tipo),
        "por_tipo": {str(k): v for k, v in sorted(por_tipo.items(), key=lambda kv: str(kv[0]))},
        "erros_por_tipo": {str(k): v for k, v in sorted(erros_por_tipo.items(), key=lambda kv: str(kv[0]))},
        "m3": {
            "cmc_declarado_consistente": cmc_declarado_ok,
            "tentativas": {str(k): v for k, v in tentativas.items()},
            "falhas": [
                {
                    "ordem": r.get("ordem"),
                    "nome": r.get("nome_entrada"),
                    "tipo": r.get("tipo_forcado"),
                    "motivo": r.get("erro"),
                }
                for r in erros
            ],
        } if modelo == "M3" else None,
    }


# --------------------------------------------------------------------------
# Comparações pareadas
# --------------------------------------------------------------------------
def pares(registros: List[dict]) -> Dict[int, dict]:
    return {r["ordem"]: r for r in registros if "ordem" in r}


def bootstrap_diferenca(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 20261025):
    """IC 95% da diferença de médias pareadas (a - b), por bootstrap."""
    if a.size == 0 or a.size != b.size:
        return None
    rng = np.random.default_rng(seed)
    dif = a - b
    amostras = rng.choice(dif, size=(n, dif.size), replace=True).mean(axis=1)
    return {
        "n": int(dif.size),
        "diferenca": float(dif.mean()),
        "ic95": [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))],
    }


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    if b01 + b10 == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    p = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)
    return {"b01": b01, "b10": b10, "p": p}


def comparar(m1: List[dict], m2: List[dict], m3: List[dict], corpus_txt: Dict[str, str]):
    """Comparações pareadas por `ordem`."""
    p1, p2, p3 = pares(m1), pares(m2), pares(m3)
    ordens = sorted(set(p1) & set(p2))
    comparacoes: Dict[str, object] = {}

    # Integridade: M1 e M2 devem ser idênticos fora do texto (mesma semente).
    campos = ["custo_mana", "cmc", "poder", "resistencia", "palavras_chave", "subtipos", "tipo"]
    divergencias = defaultdict(int)
    for o in ordens:
        for campo in campos:
            if p1[o].get(campo) != p2[o].get(campo):
                divergencias[campo] += 1
    comparacoes["integridade_m1_m2"] = {
        "cartas_pareadas": len(ordens),
        "divergencias": dict(divergencias),
    }

    # Efeito do refinamento no texto (M2 tem os dois textos lado a lado).
    delta_palavras, delta_novidade = [], []
    lint_m1, lint_m2 = [], []
    alterados = 0
    for o in sorted(p2):
        r = p2[o]
        antes = r.get("texto_regras_markov")
        depois = r.get("texto_regras")
        if antes is None or depois is None:
            continue
        if depois != antes:
            alterados += 1
        ta, td = tokens(antes), tokens(depois)
        delta_palavras.append(len(td) - len(ta))
        # novidade local: 5-gramas que aparecem no texto refinado e não no Markov
        ga, gd = hash_ngramas(ta, 5), hash_ngramas(td, 5)
        delta_novidade.append(len(gd - ga) / len(gd) if gd else 0.0)
        lint_m1.append(not lint({**r, "texto_regras": antes}, r.get("cor_entrada", "")))
        lint_m2.append(not lint(r, r.get("cor_entrada", "")))

    if delta_palavras:
        arr = np.asarray(delta_palavras, dtype=float)
        comparacoes["m2_refino_texto"] = {
            "n": int(arr.size),
            "delta_palavras_medio": float(arr.mean()),
            "delta_palavras_ic95": [
                float(np.mean(arr) - 1.96 * arr.std(ddof=1) / math.sqrt(arr.size)),
                float(np.mean(arr) + 1.96 * arr.std(ddof=1) / math.sqrt(arr.size)),
            ],
            "fracao_texto_alterado": alterados / len(delta_palavras) if delta_palavras else None,
            "novos_5gramas_fracao": float(np.mean(delta_novidade)),
            "mcnemar_lint": mcnemar(lint_m1, lint_m2),
        }

    # M3 × M1: CMC (só reais com os dois lados).
    ordens3 = sorted(set(p1) & set(p3))
    e1, e3, ex1, ex3 = [], [], [], []
    for o in ordens3:
        g = p1[o].get("gabarito") or {}
        if g.get("cmc") is None or p1[o].get("cmc") is None or p3[o].get("cmc") is None:
            continue
        alvo = float(g["cmc"])
        e1.append(abs(float(p1[o]["cmc"]) - alvo))
        e3.append(abs(float(p3[o]["cmc"]) - alvo))
        ex1.append(int(round(float(p1[o]["cmc"]))) == int(round(alvo)))
        ex3.append(int(round(float(p3[o]["cmc"]))) == int(round(alvo)))
    if e1:
        comparacoes["m3_vs_m1"] = {
            "n": len(e1),
            "cmc_mae_m1": float(np.mean(e1)),
            "cmc_mae_m3": float(np.mean(e3)),
            "bootstrap_diferenca_mae": bootstrap_diferenca(np.array(e3), np.array(e1)),
            "mcnemar_cmc_exato": mcnemar(ex1, ex3),
        }

    # M3 × M1: P/T (criaturas reais em que os três lados existem).
    pe1, pe3, pex1, pex3 = [], [], [], []
    for o in ordens3:
        r1, r3 = p1[o], p3[o]
        if r1.get("tipo_forcado") != "creature" or r1.get("tipo_experimento") != "real":
            continue
        g = r1.get("gabarito") or {}
        alvo_p = inteiro_ou_none(g.get("power"))
        alvo_r = inteiro_ou_none(g.get("toughness"))
        p1p, p1r = inteiro_ou_none(r1.get("poder")), inteiro_ou_none(r1.get("resistencia"))
        p3p, p3r = inteiro_ou_none(r3.get("poder")), inteiro_ou_none(r3.get("resistencia"))
        if None in (alvo_p, alvo_r, p1p, p1r, p3p, p3r):
            continue
        pe1.append(abs(p1p - alvo_p) + abs(p1r - alvo_r))
        pe3.append(abs(p3p - alvo_p) + abs(p3r - alvo_r))
        pex1.append(p1p == alvo_p and p1r == alvo_r)
        pex3.append(p3p == alvo_p and p3r == alvo_r)
    if pe1:
        comparacoes["m3_vs_m1_pt"] = {
            "n": len(pe1),
            "erro_soma_medio_m1": float(np.mean(pe1)),
            "erro_soma_medio_m3": float(np.mean(pe3)),
            "bootstrap_diferenca_erro": bootstrap_diferenca(np.array(pe3), np.array(pe1)),
            "mcnemar_pt_exato": mcnemar(pex1, pex3),
        }

    # Memória: M1/M2/M3 contra o texto real pareado.
    def memoria_media(registros: Dict[int, dict]) -> Optional[float]:
        valores = []
        for o, r in registros.items():
            if r.get("tipo_experimento") != "real":
                continue
            texto_real = corpus_txt.get(r.get("nome_entrada", ""))
            if not texto_real:
                continue
            valores.append(
                jaccard(
                    hash_ngramas(tokens(r.get("texto_regras") or ""), 5),
                    hash_ngramas(tokens(texto_real), 5),
                )
            )
        return float(np.mean(valores)) if valores else None

    comparacoes["memoria_por_modelo"] = {
        "M1": memoria_media(p1),
        "M2": memoria_media(p2),
        "M3": memoria_media(p3),
    }
    return comparacoes


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------
def fmt(valor, casas: int = 3) -> str:
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, float):
        if math.isnan(valor):
            return "—"
        return f"{valor:.{casas}f}"
    return str(valor)


def tabela(cabecalho: Sequence[str], linhas: Sequence[Sequence[str]]) -> str:
    saida = ["| " + " | ".join(cabecalho) + " |"]
    saida.append("|" + "|".join(["---"] * len(cabecalho)) + "|")
    for linha in linhas:
        saida.append("| " + " | ".join(linha) + " |")
    return "\n".join(saida)


def gerar_relatorio(metricas: Dict[str, dict], comparacoes: Dict[str, object]) -> str:
    partes: List[str] = []
    partes.append("# Resultados — M1, M2 e M3\n")
    partes.append(
        "Gerado por `experimentos/avaliar.py` a partir dos três `.jsonl`. "
        "Os números abaixo são reprodutíveis: as sementes são SHA-256 do "
        "`carta_id` (ver `comum.py`) e M1/M2 compartilham a mesma semente.\n"
    )

    partes.append("## 1. Volume e falhas\n")
    linhas = []
    for m in ("M1", "M2", "M3"):
        d = metricas[m]
        linhas.append([
            m,
            fmt(d["n_registros"]),
            fmt(d["n_reais"]),
            fmt(d["n_ficticias"]),
            fmt(d["n_erros"]),
            fmt(d["taxa_falha_geracao"], 4),
        ])
    partes.append(tabela(
        ["Braço", "registros", "reais", "fictícias", "erros", "taxa de falha"],
        linhas,
    ))

    partes.append("\n## 2. Estruturais (cartas reais)\n")
    linhas = []
    for m in ("M1", "M2", "M3"):
        d = metricas[m]
        linhas.append([
            m,
            fmt(d["cmc"]["mae"]),
            fmt(d["cmc"]["rmse"]),
            fmt(d["cmc"]["vies"], 2),
            fmt(d["cmc"]["exato"]),
            fmt(d["cmc"]["spearman"]),
            fmt(d["pt"]["mae_poder"]),
            fmt(d["pt"]["mae_resistencia"]),
            fmt(d["pt"]["exato"]),
            fmt(d["keywords"]["jaccard_medio"]),
            fmt(d["keywords"]["exato"]),
            fmt(d["keywords"]["jaccard_nao_vazios"]),
            fmt(d["keywords"]["exato_nao_vazios"]),
            fmt(d["keywords"]["taxa_valida_corpus"]),
            fmt(d["keywords"]["taxa_no_texto"]),
            fmt(d["keywords"]["ambos_vazios"]),
        ])
    partes.append(tabela(
        [
            "Braço", "CMC MAE", "CMC RMSE", "viés CMC", "CMC exato", "Spearman",
            "P MAE", "T MAE", "P/T exato", "KW Jaccard", "KW exato",
            "KW Jaccard*", "KW exato*", "KW válida", "KW no texto", "KW abstenção",
        ],
        linhas,
    ))
    partes.append(
        "\n> `KW válida` = fração das palavras-chave declaradas que existe no "
        "vocabulário de keywords do corpus. `KW no texto` = fração das keywords "
        "declaradas que aparece no próprio texto da carta (grounding). "
        "`KW Jaccard*`/`KW exato*` excluem as cartas em que **os dois lados** "
        "não têm keyword; `KW abstenção` mostra quantas são essas — sem esse "
        "corte, o acerto exato fica inflado por acordos vazios.\n"
    )

    partes.append("\n## 3. Texto (todas as cartas)\n")
    linhas = []
    for m in ("M1", "M2", "M3"):
        d = metricas[m]["texto"]
        linhas.append([
            m,
            fmt(d["palavras"]["media"], 1),
            fmt(d["palavras"]["mediana"], 1),
            fmt(d["distinct1"]),
            fmt(d["distinct2"]),
            fmt(d["novidade_5grama"]),
            fmt(d["memoria_jaccard_media"], 4),
            fmt(d["memoria_jaccard_max"], 3),
            fmt(d["textos_duplicados"]),
        ])
    partes.append(tabela(
        [
            "Braço", "palavras (média)", "palavras (mediana)", "distinct-1",
            "distinct-2", "novidade 5-grama", "memória média", "memória máx.",
            "textos duplicados",
        ],
        linhas,
    ))
    partes.append(
        "\n> `novidade 5-grama` = fração de 5-gramas do braço ausentes no corpus "
        "(maior = menos cópia). `memória` = Jaccard de 5-gramas com a carta real "
        "pareada; não é métrica de qualidade — é detector de cópia.\n"
    )
    top = metricas["M3"]["texto"]["duplicados_top"]
    if top:
        exemplos = "; ".join(
            f"{item['vezes']}× `{item['texto'][:70]}`" for item in top[:3]
        )
        partes.append(f"> M3, duplicados mais frequentes: {exemplos}.\n")

    partes.append("\n## 4. Lint de regras\n")
    linhas = []
    for m in ("M1", "M2", "M3"):
        d = metricas[m]["lint"]
        regras = ", ".join(f"{k}={v}" for k, v in sorted(d["por_regra"].items())) or "—"
        linhas.append([m, fmt(d["cartas_com_falha"]), fmt(d["taxa_ok"]), regras])
    partes.append(tabela(["Braço", "cartas com falha", "taxa OK", "falhas por regra"], linhas))

    linhas = []
    for m in ("M1", "M2", "M3"):
        d = metricas[m]["alertas"]
        regras = ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "—"
        linhas.append([m, regras])
    partes.append("\n**Alertas (não invalidam a carta):**\n")
    partes.append(tabela(["Braço", "alertas"], linhas))

    partes.append("\n## 5. M3 — consistência e retry\n")
    d = metricas["M3"]
    if d["m3"]:
        partes.append(
            f"- CMC declarado consistente com o custo: "
            f"**{fmt(d['m3']['cmc_declarado_consistente'])}**\n"
            f"- Chamadas por carta: `{json.dumps(d['m3']['tentativas'], ensure_ascii=False)}`\n"
            f"- Cartas geradas por tipo: `{json.dumps(d['por_tipo'], ensure_ascii=False)}`\n"
        )
        if d["erros_por_tipo"]:
            partes.append(
                f"- Falhas por tipo: `{json.dumps(d['erros_por_tipo'], ensure_ascii=False)}`\n"
            )
        for falha in d["m3"]["falhas"]:
            partes.append(
                f"- Falha: ordem {falha['ordem']} — {falha['nome']} "
                f"({falha['tipo']}): {falha['motivo']}\n"
            )

    partes.append("\n## 6. Comparações pareadas\n")
    integ = comparacoes["integridade_m1_m2"]
    partes.append(
        f"**Integridade M1 × M2:** {integ['cartas_pareadas']} cartas pareadas; "
        f"divergências fora do texto: `{json.dumps(integ['divergencias'], ensure_ascii=False)}` "
        "(esperado: tudo zero, pois a semente é a mesma).\n"
    )
    refino = comparacoes.get("m2_refino_texto")
    if refino:
        partes.append(
            f"**Efeito do refinamento (M2):** Δpalavras médio = "
            f"{fmt(refino['delta_palavras_medio'], 2)} "
            f"(IC95 {fmt(refino['delta_palavras_ic95'][0], 2)} a "
            f"{fmt(refino['delta_palavras_ic95'][1], 2)}); "
            f"fração de 5-gramas novos = {fmt(refino['novos_5gramas_fracao'])}; "
            f"McNemar do lint: p = {fmt(refino['mcnemar_lint']['p'], 4)} "
            f"(b01={refino['mcnemar_lint']['b01']}, b10={refino['mcnemar_lint']['b10']}).\n"
        )
    m3c = comparacoes.get("m3_vs_m1")
    if m3c:
        boot = m3c["bootstrap_diferenca_mae"]
        partes.append(
            f"**M3 × M1 no CMC:** MAE M3 = {fmt(m3c['cmc_mae_m3'])} vs M1 = "
            f"{fmt(m3c['cmc_mae_m1'])}; Δ = {fmt(boot['diferenca'])} "
            f"(IC95 {fmt(boot['ic95'][0])} a {fmt(boot['ic95'][1])}); "
            f"McNemar do acerto exato: p = {fmt(m3c['mcnemar_cmc_exato']['p'], 4)}.\n"
        )
    m3pt = comparacoes.get("m3_vs_m1_pt")
    if m3pt:
        boot = m3pt["bootstrap_diferenca_erro"]
        partes.append(
            f"**M3 × M1 no P/T:** erro absoluto somado (P+T) M3 = "
            f"{fmt(m3pt['erro_soma_medio_m3'])} vs M1 = "
            f"{fmt(m3pt['erro_soma_medio_m1'])}; Δ = {fmt(boot['diferenca'])} "
            f"(IC95 {fmt(boot['ic95'][0])} a {fmt(boot['ic95'][1])}); "
            f"McNemar do acerto exato: p = {fmt(m3pt['mcnemar_pt_exato']['p'], 4)} "
            f"(b01={m3pt['mcnemar_pt_exato']['b01']}, "
            f"b10={m3pt['mcnemar_pt_exato']['b10']}).\n"
        )
    mem = comparacoes["memoria_por_modelo"]
    partes.append(
        f"**Memória média por braço:** M1={fmt(mem['M1'], 4)}, "
        f"M2={fmt(mem['M2'], 4)}, M3={fmt(mem['M3'], 4)}.\n"
    )

    partes.append("\n## 7. Leitura rápida\n")
    partes.append(
        "- M1 e M2 compartilham estrutura por construção; qualquer diferença "
        "estrutural é bug de execução, não resultado.\n"
        "- O M2 só pode ganhar/perder em **texto** (comprimento, novidade, "
        "lint, memória) — é o que o desenho isola.\n"
        "- O M3 é o único braço que pode errar estrutura (CMC, P/T, terreno, "
        "coerência cor × custo) — e é isso que o lint mede.\n"
        "- Métricas de texto não têm alvo pareado: valem como distribuição, "
        "não como acerto.\n"
    )
    kw1, kw3 = metricas["M1"]["keywords"], metricas["M3"]["keywords"]
    if kw3.get("exato") is not None:
        partes.append(
            f"- **Keywords:** o acerto exato bruto do M1 ({fmt(kw1['exato'])}) "
            f"inclui {kw1['ambos_vazios']} cartas em que os dois lados são vazios; "
            f"o M3 abstém-se em apenas {kw3['ambos_vazios']}. Restringindo às cartas "
            f"com keyword em algum lado, os dois caem para "
            f"{fmt(kw1['exato_nao_vazios'])} (M1) e {fmt(kw3['exato_nao_vazios'])} "
            f"(M3) — nenhum dos dois resolve o campo.\n"
            f"- **Grounding do M3:** só {fmt(kw3['taxa_no_texto'])} das keywords "
            f"declaradas aparecem no próprio texto da carta; no M1/M2 o pipeline "
            f"garante a consistência por construção.\n"
        )
    if metricas["M3"]["m3"]:
        partes.append(
            f"- **CMC declarado do M3:** bate com o custo em "
            f"{fmt(metricas['M3']['m3']['cmc_declarado_consistente'])} dos casos — "
            f"por isso a métrica usa o CMC derivado, nunca o declarado.\n"
        )
    return "\n".join(partes)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    print("Carregando corpus para vocabulário e n-gramas...", flush=True)
    _, gramas_corpus, vocab_keywords = vocabulario_corpus()
    df = pd.read_parquet(CORPUS, columns=["name", "oracle_text"])
    oracle_por_nome = {
        str(nome): (texto or "") for nome, texto in zip(df["name"], df["oracle_text"])
    }

    registros: Dict[str, List[dict]] = {}
    erros: Dict[str, List[dict]] = {}
    for modelo in ARQUIVOS:
        ok, err = carregar(modelo)
        registros[modelo] = ok
        erros[modelo] = err
        print(f"{modelo}: {len(ok)} registros, {len(err)} erros", flush=True)

    faltando = [m for m in ARQUIVOS if not registros[m]]
    if faltando:
        print(f"AVISO: sem resultados para {faltando}; o relatório sai incompleto.")

    metricas = {
        m: metricas_modelo(
            m,
            registros[m],
            erros[m],
            gramas_corpus,
            vocab_keywords,
            oracle_por_nome,
        )
        for m in ARQUIVOS
    }
    comparacoes = comparar(
        registros["M1"], registros["M2"], registros["M3"], oracle_por_nome
    )

    saida = {"metricas": metricas, "comparacoes": comparacoes}
    caminho_json = EXPERIMENTOS / "metricas.json"
    caminho_json.write_text(
        json.dumps(saida, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    relatorio = gerar_relatorio(metricas, comparacoes)
    caminho_md = EXPERIMENTOS / "relatorio_metricas.md"
    caminho_md.write_text(relatorio, encoding="utf-8")

    print(f"OK: {caminho_json}")
    print(f"OK: {caminho_md}")


if __name__ == "__main__":
    main()
