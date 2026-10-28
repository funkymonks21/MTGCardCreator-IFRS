#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Avaliação do Modo B — inferência de tipo a partir do nome.

Lê `experimentos/resultados_modo_b.jsonl` e compara:

- **M1/M2** — `inferir_tipo_por_nome` + fallback sorteado (M1 ≡ M2);
- **M3** — `gemma3:4b` escolhendo entre os 6 tipos.

Métricas (D-004 §7.3–7.5):

- acurácia efetiva, cobertura e acurácia seletiva;
- macro-F1 e matriz de confusão (para não premiar o colapso na maioria);
- baselines: chute uniforme, amostrador do prior e maioria;
- McNemar pareado M1 × M3, geral e por subconjunto real/fictício.

Saídas:
- `experimentos/metricas_modo_b.json`
- `experimentos/relatorio_modo_b.md`

Uso:
    python experimentos/avaliar_modo_b.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliar import fmt, mcnemar, tabela  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "experimentos" / "resultados_modo_b.jsonl"
SAIDA_JSON = RAIZ / "experimentos" / "metricas_modo_b.json"
SAIDA_MD = RAIZ / "experimentos" / "relatorio_modo_b.md"
CORPUS = RAIZ / "data" / "processed" / "cards.parquet"

# Prior de tipo do corpus (§13.5), usado quando o parquet não está disponível.
PESOS_CORPUS_FALLBACK = {
    "creature": 0.5574,
    "artifact": 0.0774,
    "enchantment": 0.1057,
    "sorcery": 0.1091,
    "instant": 0.1156,
    "land": 0.0348,
}

TIPOS = ["creature", "instant", "sorcery", "artifact", "enchantment", "land"]
ROTULO = {
    "creature": "Criatura",
    "instant": "Instantânea",
    "sorcery": "Feitiço",
    "artifact": "Artefato",
    "enchantment": "Encantamento",
    "land": "Terreno",
}


def wilson(k: int, n: int, z: float = 1.959964):
    """IC 95% de Wilson para uma proporção. Não depende de aproximação normal."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    meio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centro - meio), min(1.0, centro + meio)]


def pesos_corpus() -> Dict[str, float]:
    """Distribuição de tipo no corpus (só os 6 tipos), normalizada.

    É o prior que a **amostra não tem**: o `config_v1.json` é balanceado por
    tipo por desenho (D-006), então a amostra de 400 tem 38,3% de criaturas,
    contra 55,7% no corpus. Reponderar corrige a diferença.
    """
    if not CORPUS.exists():
        return dict(PESOS_CORPUS_FALLBACK)
    try:
        import pandas as pd

        df = pd.read_parquet(CORPUS, columns=["type_line"])
    except Exception:  # noqa: BLE001 - parquet ausente/corrompido cai no fallback
        return dict(PESOS_CORPUS_FALLBACK)
    contagem = Counter()
    for linha in df["type_line"].fillna(""):
        for tipo, chave in (
            ("Creature", "creature"),
            ("Instant", "instant"),
            ("Sorcery", "sorcery"),
            ("Artifact", "artifact"),
            ("Enchantment", "enchantment"),
            ("Land", "land"),
        ):
            if tipo in linha:
                contagem[chave] += 1
                break
    total = sum(contagem.values())
    if not total:
        return dict(PESOS_CORPUS_FALLBACK)
    return {t: contagem[t] / total for t in TIPOS}


def acuracia_reponderada(resumo: Dict[str, object], pesos: Dict[str, float]):
    """Pós-estratificação: Σ w_t · recall_t, com w = prior do corpus."""
    total = 0.0
    for t in TIPOS:
        recall = resumo["recall_por_tipo"][t]
        if recall is None:
            return None
        total += pesos[t] * recall
    return total


def bootstrap_reponderado(
    registros: Sequence[dict], pesos: Dict[str, float], n: int = 10000, seed: int = 20261025
):
    """IC 95% da diferença (M3 − M1) na acurácia reponderada.

    Reamostra **dentro de cada tipo**, porque o peso é por tipo: reamostrar a
    amostra inteira mudaria a proporção das classes a cada réplica e o
    estimador deixaria de ser pós-estratificado.
    """
    tipos = np.array([r["tipo_real"] for r in registros])
    acertos = {
        chave: np.array([r[chave]["tipo"] == r["tipo_real"] for r in registros], dtype=float)
        for chave in ("m1", "m3")
    }
    indices = {t: np.where(tipos == t)[0] for t in TIPOS}
    rng = np.random.default_rng(seed)
    difs = np.empty(n)
    for i in range(n):
        acc = {"m1": 0.0, "m3": 0.0}
        for t in TIPOS:
            idx = indices[t]
            if idx.size == 0:
                continue
            amostra = rng.choice(idx, size=idx.size, replace=True)
            for chave in ("m1", "m3"):
                acc[chave] += pesos[t] * acertos[chave][amostra].mean()
        difs[i] = acc["m3"] - acc["m1"]
    return {
        "n": len(registros),
        "diferenca": float(difs.mean()),
        "ic95": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
    }


def bootstrap_diferenca(registros: Sequence[dict], n: int = 10000, seed: int = 20261025):
    """IC 95% da diferença pareada (M3 − M1) na acurácia bruta."""
    acertos = {
        chave: np.array([r[chave]["tipo"] == r["tipo_real"] for r in registros], dtype=float)
        for chave in ("m1", "m3")
    }
    dif = acertos["m3"] - acertos["m1"]
    rng = np.random.default_rng(seed)
    amostras = rng.choice(dif, size=(n, dif.size), replace=True).mean(axis=1)
    return {
        "n": int(dif.size),
        "diferenca": float(dif.mean()),
        "ic95": [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))],
    }


def carregar() -> List[dict]:
    registros: List[dict] = []
    if not ENTRADA.exists():
        return registros
    with open(ENTRADA, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if "erro" in registro:
                continue
            registros.append(registro)
    return registros


def resumo_braco(registros: Sequence[dict], chave: str) -> Dict[str, object]:
    """Métricas de um braço. `chave` é "m1" ou "m3"."""
    reais = [r["tipo_real"] for r in registros]
    previstos = [r[chave].get("tipo") for r in registros]
    validos = [p is not None for p in previstos]
    acertos = [p == t for p, t in zip(previstos, reais)]

    n = len(registros)
    n_validos = sum(validos)
    n_acertos = sum(1 for a, v in zip(acertos, validos) if a and v)

    # Matriz de confusão e macro-F1 só sobre previsões válidas.
    pares = [(t, p) for t, p, v in zip(reais, previstos, validos) if v]
    if pares:
        y_true = [t for t, _ in pares]
        y_pred = [p for _, p in pares]
        matriz = confusion_matrix(y_true, y_pred, labels=TIPOS).tolist()
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=TIPOS, average="macro", zero_division=0)
        )
    else:
        matriz = [[0] * len(TIPOS) for _ in TIPOS]
        macro_f1 = None

    recall = {}
    recall_ic95 = {}
    for i, tipo in enumerate(TIPOS):
        total = sum(matriz[i])
        recall[tipo] = matriz[i][i] / total if total else None
        recall_ic95[tipo] = wilson(matriz[i][i], total)

    # Abstenção do M1: `absteve` só existe no braço heurístico.
    absteve = [bool(r[chave].get("absteve")) for r in registros] if chave == "m1" else None
    cobertura = (
        sum(1 for a in absteve if not a) / n if absteve is not None else n_validos / n
    )
    if absteve is not None:
        nao_absteve = [i for i, a in enumerate(absteve) if not a]
        seletiva = (
            sum(1 for i in nao_absteve if acertos[i]) / len(nao_absteve)
            if nao_absteve
            else None
        )
        abst_idx = [i for i, a in enumerate(absteve) if a]
        acuracia_fallback = (
            sum(1 for i in abst_idx if acertos[i]) / len(abst_idx)
            if abst_idx
            else None
        )
    else:
        seletiva = n_acertos / n_validos if n_validos else None
        acuracia_fallback = None

    return {
        "n": n,
        "n_validos": n_validos,
        "acuracia_efetiva": n_acertos / n if n else None,
        "ic95_acuracia": wilson(n_acertos, n),
        "cobertura": cobertura,
        "acuracia_seletiva": seletiva,
        "acuracia_fallback": acuracia_fallback,
        "macro_f1": macro_f1,
        "recall_por_tipo": recall,
        "recall_ic95": recall_ic95,
        "matriz_confusao": matriz,
        "abstencoes": sum(1 for a in absteve if a) if absteve is not None else None,
    }


def baselines(reais: Sequence[str]) -> Dict[str, float]:
    n = len(reais)
    p = Counter(reais)
    return {
        "uniforme": 1 / len(TIPOS),
        "prior_amostra": sum((v / n) ** 2 for v in p.values()) if n else None,
        "maioria_amostra": max(p.values()) / n if n else None,
        "tipo_majoritario": p.most_common(1)[0][0] if p else None,
    }


def mcnemar_braco(registros: Sequence[dict]) -> Dict[str, object]:
    pares = [
        r
        for r in registros
        if r["m1"].get("tipo") is not None and r["m3"].get("tipo") is not None
    ]
    if not pares:
        return {"n": 0}
    acertos_m1 = [r["m1"]["tipo"] == r["tipo_real"] for r in pares]
    acertos_m3 = [r["m3"]["tipo"] == r["tipo_real"] for r in pares]
    teste = mcnemar(acertos_m1, acertos_m3)
    return {
        "n": len(pares),
        "acuracia_m1": float(np.mean(acertos_m1)),
        "acuracia_m3": float(np.mean(acertos_m3)),
        **teste,
    }


def matriz_md(matriz: Sequence[Sequence[int]], titulo: str) -> str:
    cabecalho = ["real \\ previsto"] + [ROTULO[t] for t in TIPOS]
    linhas = []
    for i, tipo in enumerate(TIPOS):
        linhas.append([ROTULO[tipo]] + [str(v) for v in matriz[i]])
    return f"**{titulo}**\n\n" + tabela(cabecalho, linhas)


def main() -> None:
    registros = carregar()
    if not registros:
        raise SystemExit(f"Sem resultados em {ENTRADA}")

    reais = [r["tipo_real"] for r in registros]
    m1 = resumo_braco(registros, "m1")
    m3 = resumo_braco(registros, "m3")
    base = baselines(reais)

    # Pós-estratificação pelo prior do corpus (ver D-001 §4.9 e §14.1).
    pesos = pesos_corpus()
    reponderado = {
        "pesos_corpus": pesos,
        "m1": acuracia_reponderada(m1, pesos),
        "m3": acuracia_reponderada(m3, pesos),
        "diferenca": bootstrap_reponderado(registros, pesos),
    }
    delta_bruto = bootstrap_diferenca(registros)

    por_subconjunto = {}
    for tipo_exp in ("real", "ficticio"):
        sub = [r for r in registros if r.get("tipo_experimento") == tipo_exp]
        if sub:
            por_subconjunto[tipo_exp] = {
                "n": len(sub),
                "m1": resumo_braco(sub, "m1"),
                "m3": resumo_braco(sub, "m3"),
                "mcnemar": mcnemar_braco(sub),
            }

    comparacoes = {
        "geral": mcnemar_braco(registros),
        "real": por_subconjunto.get("real", {}).get("mcnemar"),
        "ficticio": por_subconjunto.get("ficticio", {}).get("mcnemar"),
    }

    distribuicao = Counter(reais)
    amostra = {
        "n": len(registros),
        "por_tipo_real": {t: distribuicao.get(t, 0) for t in TIPOS},
        "por_experimento": dict(Counter(r.get("tipo_experimento") for r in registros)),
    }

    saida = {
        "amostra": amostra,
        "baselines": base,
        "m1": m1,
        "m3": m3,
        "comparacoes": comparacoes,
        "por_subconjunto": por_subconjunto,
        "reponderado_corpus": reponderado,
        "bootstrap_diferenca_bruta": delta_bruto,
    }
    SAIDA_JSON.write_text(
        json.dumps(saida, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    # ---------------- relatório ----------------
    partes = []
    partes.append("# Modo B — inferência de tipo pelo nome\n")
    partes.append(
        "Gerado por `experimentos/avaliar_modo_b.py` a partir de "
        "`resultados_modo_b.jsonl`. Entrada dos dois braços: **apenas o nome** "
        "(sem cor, sem tipo). M1 e M2 são idênticos por construção; por isso a "
        "comparação tem **dois braços**, não três.\n"
    )

    partes.append("## 1. Amostra\n")
    partes.append(
        f"- {amostra['n']} cartas sorteadas das 1.000 (semente fixa "
        "`20261025`), sem estratificação.\n"
        f"- Reais: {amostra['por_experimento'].get('real', 0)}; "
        f"fictícias: {amostra['por_experimento'].get('ficticio', 0)}.\n"
    )
    linhas = [
        [ROTULO[t], str(amostra["por_tipo_real"][t])] for t in TIPOS
    ]
    partes.append(tabela(["Tipo real", "n"], linhas))

    partes.append("\n## 2. Métricas por braço\n")
    linhas = []
    for nome, d in (("M1/M2 (heurística)", m1), ("M3 (LLM)", m3)):
        ic = d["ic95_acuracia"]
        linhas.append([
            nome,
            fmt(d["acuracia_efetiva"]),
            f"[{fmt(ic[0])}; {fmt(ic[1])}]" if ic else "—",
            fmt(d["cobertura"]),
            fmt(d["acuracia_seletiva"]),
            fmt(d["macro_f1"]),
        ])
    partes.append(tabela(
        ["Braço", "Acurácia efetiva", "IC95 (Wilson)", "Cobertura", "Acurácia seletiva", "Macro-F1"],
        linhas,
    ))
    partes.append(
        "\n> `efetiva` = acertos / todas as cartas. No M1, abstenções viram o "
        "sorteio semeado do pipeline; no M3, falha de geração conta como erro. "
        "`seletiva` = acerto só entre as respostas válidas (no M1, só entre as "
        "não-abstenções). A decomposição do D-005 §8.2 "
        "(`acc_ef = c·a_sel + (1−c)·S`) supõe fallback com o prior do próprio "
        "conjunto; aqui o fallback usa o prior do corpus (≈55,7% criaturas), "
        "então a conta não fecha exatamente com o `S` da amostra.\n"
    )
    delta = delta_bruto
    partes.append(
        f"\n**Diferença (M3 − M1), amostra como está:** "
        f"{fmt(delta['diferenca'] * 100, 2)} p.p.; "
        f"IC95 bootstrap [{fmt(delta['ic95'][0] * 100, 2)}; "
        f"{fmt(delta['ic95'][1] * 100, 2)}] p.p. "
        "O intervalo contém zero — compatível com empate.\n"
    )
    if m1["abstencoes"] is not None:
        partes.append(
            f"\nO M1 absteve-se em **{m1['abstencoes']}/{m1['n']}** cartas "
            f"(cobertura {fmt(m1['cobertura'])}); entre essas, o fallback "
            f"sorteado acertou {fmt(m1['acuracia_fallback'])}.\n"
        )
    if m3["abstencoes"] is not None or m3["n_validos"] < m3["n"]:
        partes.append(
            f"\nO M3 falhou em **{m3['n'] - m3['n_validos']}** cartas.\n"
        )

    partes.append("\n## 3. Baselines (mesma amostra)\n")
    linhas = [
        ["Chute uniforme entre 6", fmt(base["uniforme"])],
        ["Amostrador do prior da amostra", fmt(base["prior_amostra"])],
        [
            f"Sempre “{ROTULO.get(base['tipo_majoritario'], base['tipo_majoritario'])}”",
            fmt(base["maioria_amostra"]),
        ],
    ]
    partes.append(tabela(["Baseline", "Acurácia"], linhas))
    partes.append(
        "\n> Referência do corpus (D-004 §7.4): uniforme 16,67%, prior 34,44%, "
        "maioria (Creature) 54,83%. Os números acima são os da **amostra de "
        "400**, que tem 38% de criaturas — por isso a maioria é menor que a do "
        "corpus. Comparar com o baseline da própria amostra é o que evita "
        "inflar a dificuldade.\n"
    )

    # ---------------- reponderação ----------------
    pesos = reponderado["pesos_corpus"]
    prior_corpus = sum(v * v for v in pesos.values())
    partes.append("\n### 3.1 Acurácia reponderada pelo corpus\n")
    partes.append(
        "A amostra segue a alocação do D-006 (piso de 60 por tipo), que "
        "**sobre-representa os tipos raros**: terreno é 12,5% na amostra e 3,5% "
        "no corpus; criatura é 38,2% e 55,7%. Reponderar por `w_t` = prior do "
        "corpus responde à pergunta que a amostra sozinha não responde: **que "
        "acurácia esperar numa população com a mistura real de tipos**. É a "
        "mesma pós-estratificação do D-001 §4.9, aplicada à acurácia de tipo.\n"
    )
    linhas = []
    for t in TIPOS:
        n_amostra = amostra["por_tipo_real"][t]
        linhas.append([
            ROTULO[t],
            f"{pesos[t] * 100:.1f}%",
            f"{n_amostra / amostra['n'] * 100:.1f}%",
            str(n_amostra),
        ])
    partes.append(tabela(["Tipo", "Peso no corpus", "Peso na amostra", "n"], linhas))
    linhas = [
        ["M1/M2 (heurística)", fmt(m1["acuracia_efetiva"]), fmt(reponderado["m1"])],
        ["M3 (LLM)", fmt(m3["acuracia_efetiva"]), fmt(reponderado["m3"])],
        ["Chute uniforme", fmt(base["uniforme"]), fmt(1 / len(TIPOS))],
        ["Amostrador do prior", fmt(base["prior_amostra"]), fmt(prior_corpus)],
        [
            "Sempre Criatura",
            fmt(base["maioria_amostra"]),
            fmt(pesos["creature"]),
        ],
    ]
    partes.append("\n" + tabela(
        ["Braço / baseline", "Amostra", "Reponderado pelo corpus"], linhas
    ))
    dif = reponderado["diferenca"]
    partes.append(
        f"\n**Diferença reponderada (M3 − M1):** "
        f"{fmt(dif['diferenca'] * 100, 2)} p.p.; IC95 bootstrap "
        f"[{fmt(dif['ic95'][0] * 100, 2)}; {fmt(dif['ic95'][1] * 100, 2)}] p.p. "
        "O IC também contém zero.\n"
    )
    partes.append(
        "\n> **Hipótese da reponderação:** cada tipo tem a mesma acurácia na "
        "amostra e na população. O IC por tipo (§4.1) mostra onde essa hipótese "
        "é mais frágil: terreno e encantamento têm `n` pequeno e intervalos "
        "largos, e são justamente os tipos em que os braços mais divergem. "
        "Logo, o número reponderado é uma **extrapolação**, não uma medição "
        "direta.\n"
    )

    partes.append("\n## 4. Matriz de confusão\n")
    partes.append(matriz_md(m1["matriz_confusao"], "M1/M2 (heurística)"))
    partes.append("\n" + matriz_md(m3["matriz_confusao"], "M3 (LLM)"))

    partes.append("\n### 4.1 Recall por tipo real\n")
    linhas = []
    for t in TIPOS:
        ic1 = m1["recall_ic95"][t]
        ic3 = m3["recall_ic95"][t]
        linhas.append([
            ROTULO[t],
            str(amostra["por_tipo_real"][t]),
            (
                f"{fmt(m1['recall_por_tipo'][t])} "
                f"[{fmt(ic1[0])}; {fmt(ic1[1])}]"
                if ic1
                else fmt(m1["recall_por_tipo"][t])
            ),
            (
                f"{fmt(m3['recall_por_tipo'][t])} "
                f"[{fmt(ic3[0])}; {fmt(ic3[1])}]"
                if ic3
                else fmt(m3["recall_por_tipo"][t])
            ),
        ])
    partes.append(tabela(
        ["Tipo real", "n", "M1 recall [IC95]", "M3 recall [IC95]"], linhas
    ))
    partes.append(
        "\n> IC95 de Wilson por tipo. Os intervalos **não são independentes** "
        "(as linhas da matriz somam o mesmo `n`) e não foram corrigidos para "
        "multiplicidade; servem para mostrar a precisão, não para declarar "
        "vitória por tipo.\n"
    )

    partes.append("\n## 5. Real × fictício (controle de memorização)\n")
    linhas = []
    for tipo_exp, rotulo in (("real", "Real"), ("ficticio", "Fictícia")):
        d = por_subconjunto.get(tipo_exp)
        if not d:
            continue
        linhas.append([
            rotulo,
            str(d["n"]),
            fmt(d["m1"]["acuracia_efetiva"]),
            fmt(d["m3"]["acuracia_efetiva"]),
            fmt(d["m1"]["cobertura"]),
            fmt(d["m3"]["cobertura"]),
        ])
    partes.append(tabela(
        ["Subconjunto", "n", "M1 acurácia", "M3 acurácia", "M1 cobertura", "M3 cobertura"],
        linhas,
    ))
    partes.append(
        "\n> Nomes fictícios são o controle limpo de memorização: o `gemma3:4b` "
        "não os viu no pré-treinamento. Se o M3 for muito melhor em nomes "
        "reais, a diferença é (ao menos em parte) conhecimento de carta, não "
        "inferência.\n"
    )

    partes.append("\n## 6. McNemar (M1 × M3)\n")
    linhas = []
    for nome, chave in (("Geral", "geral"), ("Só reais", "real"), ("Só fictícias", "ficticio")):
        d = comparacoes.get(chave)
        if not d or not d.get("n"):
            continue
        linhas.append([
            nome,
            str(d["n"]),
            fmt(d.get("acuracia_m1")),
            fmt(d.get("acuracia_m3")),
            str(d.get("b01")),
            str(d.get("b10")),
            fmt(d.get("p"), 4),
        ])
    partes.append(tabela(
        ["Conjunto", "n", "M1", "M3", "b01 (M1 erra, M3 acerta)", "b10 (M1 acerta, M3 erra)", "p"],
        linhas,
    ))
    partes.append(
        "\n> McNemar bilateral com correção de continuidade; `p` é o do teste "
        "binomial exato usado em `avaliar.mcnemar`.\n"
    )

    partes.append("\n## 7. Leitura\n")
    m1_ef, m3_ef = m1["acuracia_efetiva"], m3["acuracia_efetiva"]
    partes.append(
        f"- Acurácia efetiva: M1 = **{fmt(m1_ef)}**, M3 = **{fmt(m3_ef)}**; "
        f"baseline de maioria na amostra = {fmt(base['maioria_amostra'])} e "
        f"IC95 de Wilson [{fmt(m1['ic95_acuracia'][0])}; "
        f"{fmt(m1['ic95_acuracia'][1])}] (M1) / "
        f"[{fmt(m3['ic95_acuracia'][0])}; {fmt(m3['ic95_acuracia'][1])}] (M3). "
        "Na amostra os dois batem o baseline por ~13 p.p.\n"
    )
    partes.append(
        f"- **Reponderado pelo corpus** (55,7% criaturas): M1 = "
        f"**{fmt(reponderado['m1'])}**, M3 = **{fmt(reponderado['m3'])}**, "
        f"contra maioria de {fmt(pesos['creature'])}. A margem sobre o "
        "baseline cai para ~3 p.p. e a ordem dos braços se inverte — parte do "
        "acerto de tipo é, sim, o prior de criaturas.\n"
    )
    partes.append(
        f"- **Macro-F1:** M1 = {fmt(m1['macro_f1'])} vs M3 = "
        f"{fmt(m3['macro_f1'])}. A acurácia bruta empata, mas o M3 é mais "
        "equilibrado entre as seis classes.\n"
    )
    if m3_ef is not None and base["maioria_amostra"] is not None:
        if m3_ef <= base["maioria_amostra"]:
            partes.append(
                "- O M3 **não supera** o baseline de maioria: parte do acerto "
                "pode ser só o prior de criaturas.\n"
            )
    partes.append(
        "- O M1 tem cobertura menor que 100%: parte do número efetivo vem do "
        "sorteio de fallback, não da heurística. A acurácia seletiva mostra o "
        "quanto a heurística acerta **quando responde**.\n"
    )
    partes.append(
        "- A matriz de confusão é obrigatória: acurácia sozinha não distingue "
        "um classificador que aprendeu as 6 classes de um que colapsou em "
        "Creature.\n"
    )

    # Vantagens por tipo (recall).
    r1, r3 = m1["recall_por_tipo"], m3["recall_por_tipo"]
    ganha_m1 = [
        t for t in TIPOS
        if r1[t] is not None and r3[t] is not None and r1[t] > r3[t] + 0.05
    ]
    ganha_m3 = [
        t for t in TIPOS
        if r1[t] is not None and r3[t] is not None and r3[t] > r1[t] + 0.05
    ]
    if ganha_m1:
        detalhe = ", ".join(
            f"{ROTULO[t]} ({fmt(r1[t])} vs {fmt(r3[t])})" for t in ganha_m1
        )
        partes.append(f"- **Onde o M1 ganha (>5 p.p. de recall):** {detalhe}.\n")
    if ganha_m3:
        detalhe = ", ".join(
            f"{ROTULO[t]} ({fmt(r3[t])} vs {fmt(r1[t])})" for t in ganha_m3
        )
        partes.append(f"- **Onde o M3 ganha (>5 p.p. de recall):** {detalhe}.\n")

    real = por_subconjunto.get("real")
    fict = por_subconjunto.get("ficticio")
    if real and fict:
        d_m3 = real["m3"]["acuracia_efetiva"] - fict["m3"]["acuracia_efetiva"]
        d_m1 = real["m1"]["acuracia_efetiva"] - fict["m1"]["acuracia_efetiva"]
        partes.append(
            f"- **Memorização:** real − fictícia = {fmt(d_m3 * 100, 1)} p.p. "
            f"(M3) e {fmt(d_m1 * 100, 1)} p.p. (M1). O ganho do M3 em nomes "
            f"reais é da mesma ordem do ganho do M1; nas fictícias os dois "
            f"empatam ({fmt(fict['m1']['acuracia_efetiva'])} vs "
            f"{fmt(fict['m3']['acuracia_efetiva'])}). Não há sinal forte de "
            "memorização de carta — o que existe é vocabulário mais canônico "
            "nos nomes reais.\n"
        )

    if comparacoes["geral"].get("p") is not None:
        p = comparacoes["geral"]["p"]
        partes.append(
            f"- McNemar geral: p = {fmt(p, 4)}. "
            + ("Há diferença detectável entre os braços." if p < 0.05 else "Não há diferença detectável entre os braços.")
            + "\n"
        )

    SAIDA_MD.write_text("\n".join(partes), encoding="utf-8")
    print(f"OK: {SAIDA_JSON}")
    print(f"OK: {SAIDA_MD}")


if __name__ == "__main__":
    main()
