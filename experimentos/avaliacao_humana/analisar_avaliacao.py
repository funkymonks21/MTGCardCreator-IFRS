#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analisa as respostas da avaliação humana cega (P-04).

Entrada: `experimentos/avaliacao_humana/respostas/*.csv`, um por avaliador, com
as colunas `avaliador,item_id,realismo,coerencia,criatividade,existe`. O
`formulario.html` exporta nesse formato.

O pareamento é por **bloco**: cada bloco tem 4 cartas com o mesmo nome (M1, M2,
M3 e a real). Toda inferência é feita no nível do bloco, com bootstrap por
bloco — as 4 cartas de um bloco não são independentes, e tratá-las como tal
subestimaria o erro padrão.

Saídas:
- `experimentos/avaliacao_humana/relatorio_avaliacao_humana.md`
- `experimentos/avaliacao_humana/metricas_avaliacao_humana.json`

Uso:
    python experimentos/avaliacao_humana/analisar_avaliacao.py
    python experimentos/avaliacao_humana/analisar_avaliacao.py --respostas DIR
    python experimentos/avaliacao_humana/analisar_avaliacao.py --simular 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parents[2]
DIR = RAIZ / "experimentos" / "avaliacao_humana"
GABARITO = DIR / "gabarito.json"

sys.path.insert(0, str(RAIZ / "experimentos"))
from avaliar import fmt, tabela  # noqa: E402

BRACOS = ["M1", "M2", "M3", "REAL"]
ROTULO = {"M1": "M1 (Markov)", "M2": "M2 (Markov+refino)", "M3": "M3 (LLM)", "REAL": "Carta real"}
DIMENSOES = ["realismo", "coerencia", "criatividade"]
ROTULO_DIM = {
    "realismo": "Realismo",
    "coerencia": "Coerência",
    "criatividade": "Criatividade",
}
N_BOOT = 10000
SEED = 20261026
COMPARACOES = [("M2", "M1"), ("M3", "M1"), ("M3", "M2"), ("M1", "REAL"), ("M2", "REAL"), ("M3", "REAL")]


def carregar_gabarito() -> Dict[str, dict]:
    if not GABARITO.exists():
        raise SystemExit(f"Gabarito ausente: {GABARITO}. Rode gerar_amostra.py.")
    dados = json.loads(GABARITO.read_text(encoding="utf-8"))
    return dados["itens"]


def normalizar_existe(valor: str) -> Optional[int]:
    v = (valor or "").strip().lower()
    if v in {"sim", "s", "1", "true", "yes", "y"}:
        return 1
    if v in {"nao", "não", "n", "0", "false", "no"}:
        return 0
    return None


def carregar_respostas(pasta: Path, gabarito: Dict[str, dict]):
    registros: List[dict] = []
    problemas: List[str] = []
    arquivos = sorted(pasta.glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum CSV em {pasta}")
    for arquivo in arquivos:
        with open(arquivo, encoding="utf-8-sig", newline="") as f:
            for numero, linha in enumerate(csv.DictReader(f), start=2):
                item_id = (linha.get("item_id") or "").strip()
                if not item_id:
                    continue
                if item_id not in gabarito:
                    problemas.append(f"{arquivo.name}:{numero}: item_id desconhecido {item_id!r}")
                    continue
                avaliador = (linha.get("avaliador") or arquivo.stem).strip()
                valores = {}
                for dim in DIMENSOES:
                    bruto = (linha.get(dim) or "").strip().replace(",", ".")
                    try:
                        valor = int(float(bruto))
                    except ValueError:
                        problemas.append(f"{arquivo.name}:{numero}: {dim}={bruto!r}")
                        continue
                    if not 1 <= valor <= 5:
                        problemas.append(f"{arquivo.name}:{numero}: {dim} fora de 1–5 ({valor})")
                        continue
                    valores[dim] = valor
                existe = normalizar_existe(linha.get("existe") or "")
                if existe is None:
                    problemas.append(f"{arquivo.name}:{numero}: existe={linha.get('existe')!r}")
                if len(valores) != len(DIMENSOES):
                    continue
                registros.append(
                    {
                        "avaliador": avaliador,
                        "item_id": item_id,
                        "bloco": gabarito[item_id]["ordem"],
                        "braco": gabarito[item_id]["braco"],
                        "existe": existe,
                        **valores,
                    }
                )
    # Reenvio do mesmo avaliador: a última linha vence (mesma regra dos runners).
    unicos: Dict[tuple, dict] = {}
    for registro in registros:
        unicos[(registro["avaliador"], registro["item_id"])] = registro
    if len(unicos) != len(registros):
        problemas.append(
            f"{len(registros) - len(unicos)} resposta(s) duplicada(s) por "
            "(avaliador, item) — mantida a última"
        )
    return list(unicos.values()), problemas, arquivos


def media_ic(valores_por_bloco: Dict[int, List[float]], n_boot: int = N_BOOT):
    """Média com IC95 por bootstrap de blocos."""
    blocos = [np.array(v, dtype=float) for v in valores_por_bloco.values() if v]
    if not blocos:
        return None
    tudo = np.concatenate(blocos)
    rng = np.random.default_rng(SEED)
    replicas = np.empty(n_boot)
    for i in range(n_boot):
        escolhidos = rng.integers(0, len(blocos), size=len(blocos))
        replicas[i] = np.concatenate([blocos[j] for j in escolhidos]).mean()
    return {
        "n_obs": int(tudo.size),
        "n_blocos": len(blocos),
        "media": float(tudo.mean()),
        "ic95": [float(np.percentile(replicas, 2.5)), float(np.percentile(replicas, 97.5))],
    }


def por_bloco(registros: Sequence[dict], braco: str, dim: str) -> Dict[int, List[float]]:
    saida: Dict[int, List[float]] = defaultdict(list)
    for r in registros:
        if r["braco"] == braco and r.get(dim) is not None:
            saida[r["bloco"]].append(float(r[dim]))
    return saida


def contraste(registros: Sequence[dict], a: str, b: str, dim: str):
    """Diferença pareada por bloco (a − b), IC bootstrap e teste t pareado."""
    blocos_a = por_bloco(registros, a, dim)
    blocos_b = por_bloco(registros, b, dim)
    comuns = sorted(set(blocos_a) & set(blocos_b))
    if not comuns:
        return None
    difs = np.array([np.mean(blocos_a[k]) - np.mean(blocos_b[k]) for k in comuns])
    rng = np.random.default_rng(SEED)
    replicas = np.array([difs[rng.integers(0, difs.size, difs.size)].mean() for _ in range(N_BOOT)])
    p = None
    if len(comuns) >= 2:
        p = float(stats.ttest_rel(
            [np.mean(blocos_a[k]) for k in comuns],
            [np.mean(blocos_b[k]) for k in comuns],
        ).pvalue)
    return {
        "n_blocos": len(comuns),
        "diferenca": float(difs.mean()),
        "ic95": [float(np.percentile(replicas, 2.5)), float(np.percentile(replicas, 97.5))],
        "p": p,
        "blocos_a_favor": int(np.sum(difs > 0)),
        "blocos_b_favor": int(np.sum(difs < 0)),
        "empates": int(np.sum(difs == 0)),
    }


def icc2k(matriz: np.ndarray):
    """ICC(2,k): two-way random, absolute agreement, média de k avaliadores."""
    n, k = matriz.shape
    if n < 2 or k < 2:
        return None
    grande = matriz.mean()
    ss_total = ((matriz - grande) ** 2).sum()
    ss_itens = k * ((matriz.mean(axis=1) - grande) ** 2).sum()
    ss_avaliadores = n * ((matriz.mean(axis=0) - grande) ** 2).sum()
    ss_erro = ss_total - ss_itens - ss_avaliadores
    ms_itens = ss_itens / (n - 1)
    ms_avaliadores = ss_avaliadores / (k - 1)
    ms_erro = ss_erro / ((n - 1) * (k - 1))
    denominador = ms_itens + (ms_avaliadores - ms_erro) / n
    if denominador == 0:
        return None
    return float((ms_itens - ms_erro) / denominador)


def concordancia(registros: Sequence[dict], dim: str):
    avaliadores = sorted({r["avaliador"] for r in registros})
    itens = sorted({r["item_id"] for r in registros})
    mapa = {(r["item_id"], r["avaliador"]): r.get(dim) for r in registros}
    linhas = [
        [mapa[(i, a)] for a in avaliadores]
        for i in itens
        if all(mapa.get((i, a)) is not None for a in avaliadores)
    ]
    if not linhas:
        return {"n_itens": 0, "n_avaliadores": len(avaliadores), "icc2k": None}
    return {
        "n_itens": len(linhas),
        "n_avaliadores": len(avaliadores),
        "icc2k": icc2k(np.array(linhas, dtype=float)),
    }


def proporcao_existe(registros: Sequence[dict], braco: str):
    por = por_bloco(registros, braco, "existe")
    return media_ic(por)


def analisar(registros: Sequence[dict], arquivos: Sequence[Path], problemas: Sequence[str]):
    avaliadores = sorted({r["avaliador"] for r in registros})
    blocos = sorted({r["bloco"] for r in registros})

    medias: Dict[str, Dict[str, dict]] = {}
    for braco in BRACOS:
        medias[braco] = {}
        for dim in DIMENSOES:
            resultado = media_ic(por_bloco(registros, braco, dim))
            if resultado:
                medias[braco][dim] = resultado

    contrastes = {}
    for a, b in COMPARACOES:
        chave = f"{a}-{b}"
        contrastes[chave] = {}
        for dim in DIMENSOES:
            resultado = contraste(registros, a, b, dim)
            if resultado:
                contrastes[chave][dim] = resultado

    existe = {braco: proporcao_existe(registros, braco) for braco in BRACOS}
    existe_contrastes = {}
    for a, b in COMPARACOES:
        blocos_a = por_bloco(registros, a, "existe")
        blocos_b = por_bloco(registros, b, "existe")
        comuns = sorted(set(blocos_a) & set(blocos_b))
        if not comuns:
            continue
        difs = np.array([np.mean(blocos_a[k]) - np.mean(blocos_b[k]) for k in comuns])
        rng = np.random.default_rng(SEED)
        replicas = np.array([difs[rng.integers(0, difs.size, difs.size)].mean() for _ in range(N_BOOT)])
        existe_contrastes[f"{a}-{b}"] = {
            "diferenca": float(difs.mean()),
            "ic95": [float(np.percentile(replicas, 2.5)), float(np.percentile(replicas, 97.5))],
        }

    concordancia_dim = {dim: concordancia(registros, dim) for dim in DIMENSOES}

    integridade = {
        "n_avaliadores": len(avaliadores),
        "avaliadores": avaliadores,
        "n_respostas": len(registros),
        "n_blocos": len(blocos),
        "arquivos": [a.name for a in arquivos],
        "problemas": list(problemas),
        "respostas_por_avaliador": dict(Counter(r["avaliador"] for r in registros)),
    }

    return {
        "integridade": integridade,
        "medias": medias,
        "contrastes": contrastes,
        "existe": existe,
        "existe_contrastes": existe_contrastes,
        "concordancia": concordancia_dim,
    }


def relatorio(metricas: dict, simulado: bool = False) -> str:
    integ = metricas["integridade"]
    medias = metricas["medias"]
    partes = ["# Avaliação humana cega — relatório\n"]
    if simulado:
        partes.append(
            "> **DADOS SINTÉTICOS — NÃO É RESULTADO.** Gerado por `--simular` "
            "para testar o pipeline. Não citar, não copiar, não versionar.\n"
        )
    partes.append(
        "Gerado por `experimentos/avaliacao_humana/analisar_avaliacao.py`. "
        "Cada bloco tem 4 cartas com o mesmo nome: M1, M2, M3 e a **carta "
        "real** (âncora). Toda inferência é no nível do bloco, com bootstrap "
        "por bloco — as 4 cartas de um bloco não são independentes.\n"
    )

    partes.append("\n## 1. Participação e integridade\n")
    partes.append(tabela(
        ["Item", "Valor"],
        [
            ["Avaliadores", str(integ["n_avaliadores"])],
            ["Blocos cobertos", str(integ["n_blocos"])],
            ["Respostas válidas", str(integ["n_respostas"])],
        ],
    ))
    por_aval = ", ".join(
        f"{nome}: {n}" for nome, n in sorted(integ["respostas_por_avaliador"].items())
    )
    partes.append(f"\nRespostas por avaliador — {por_aval}.\n")
    if integ["problemas"]:
        partes.append(f"\n> **{len(integ['problemas'])} problema(s) de leitura** "
                      "(linhas ignoradas ou corrigidas). Primeiros: "
                      + "; ".join(f"`{p}`" for p in integ["problemas"][:5]) + "\n")
    else:
        partes.append("\nNenhum problema de leitura: todos os `item_id` existem "
                      "no gabarito e todas as notas estão em 1–5.\n")

    partes.append("\n## 2. Médias por braço\n")
    for dim in DIMENSOES:
        linhas = []
        for braco in BRACOS:
            d = medias[braco].get(dim)
            if not d:
                linhas.append([ROTULO[braco], "—", "—", "—"])
                continue
            linhas.append([
                ROTULO[braco],
                fmt(d["media"], 2),
                f"[{fmt(d['ic95'][0], 2)}; {fmt(d['ic95'][1], 2)}]",
                str(d["n_obs"]),
            ])
        partes.append(f"\n**{ROTULO_DIM[dim]}** (1–5, maior é melhor)\n")
        partes.append(tabela(["Braço", "Média", "IC95", "n"], linhas))

    partes.append("\n## 3. Contrastes pareados por bloco\n")
    partes.append(
        "Diferença média dentro do bloco (a − b). Positivo = **a** melhor. "
        "`p` é do teste t pareado sobre as médias de bloco; o IC é bootstrap "
        "por bloco.\n"
    )
    for chave in [f"{a}-{b}" for a, b in COMPARACOES]:
        dados = metricas["contrastes"].get(chave, {})
        if not dados:
            continue
        linhas = []
        for dim in DIMENSOES:
            d = dados.get(dim)
            if not d:
                continue
            linhas.append([
                ROTULO_DIM[dim],
                f"{d['diferenca']:+.2f}",
                f"[{d['ic95'][0]:+.2f}; {d['ic95'][1]:+.2f}]",
                fmt(d["p"], 4) if d.get("p") is not None else "—",
                f"{d['blocos_a_favor']}/{d['blocos_b_favor']}/{d['empates']}",
            ])
        if linhas:
            partes.append(f"\n**{chave}**\n")
            partes.append(tabela(
                ["Dimensão", "Δ", "IC95", "p", "blocos a/b/empate"], linhas
            ))

    partes.append("\n## 4. “Poderia existir?”\n")
    linhas = []
    for braco in BRACOS:
        d = metricas["existe"].get(braco)
        if not d:
            continue
        linhas.append([
            ROTULO[braco],
            fmt(d["media"] * 100, 1) + "%",
            f"[{fmt(d['ic95'][0] * 100, 1)}%; {fmt(d['ic95'][1] * 100, 1)}%]",
            str(d["n_obs"]),
        ])
    partes.append(tabela(["Braço", "Sim", "IC95", "n"], linhas))
    linhas = []
    for chave in [f"{a}-{b}" for a, b in COMPARACOES]:
        d = metricas["existe_contrastes"].get(chave)
        if not d:
            continue
        linhas.append([
            chave,
            f"{d['diferenca'] * 100:+.1f} p.p.",
            f"[{d['ic95'][0] * 100:+.1f}; {d['ic95'][1] * 100:+.1f}]",
        ])
    if linhas:
        partes.append("\n" + tabela(["Contraste", "Δ", "IC95"], linhas))

    partes.append("\n## 5. Concordância entre avaliadores\n")
    partes.append(
        "ICC(2,k) — two-way random effects, absolute agreement, média dos "
        "avaliadores. `≥ 0,75` boa, `0,50–0,75` moderada, `< 0,50` fraca.\n"
    )
    linhas = []
    for dim in DIMENSOES:
        d = metricas["concordancia"][dim]
        icc = d.get("icc2k")
        linhas.append([
            ROTULO_DIM[dim],
            fmt(icc, 3) if icc is not None else "—",
            str(d["n_itens"]),
            str(d["n_avaliadores"]),
        ])
    partes.append(tabela(["Dimensão", "ICC(2,k)", "Itens completos", "Avaliadores"], linhas))
    partes.append(
        "\n> O ICC usa só as linhas completas (item com nota de **todos** os "
        "avaliadores). Com menos de 2 avaliadores ele não é calculável.\n"
    )

    partes.append("\n## 6. Leitura\n")
    m_real = {dim: medias["REAL"].get(dim, {}).get("media") for dim in DIMENSOES}
    for dim in DIMENSOES:
        if m_real[dim] is None:
            continue
        texto = [f"- **{ROTULO_DIM[dim]}:** real = {fmt(m_real[dim], 2)}"]
        for braco in ("M1", "M2", "M3"):
            valor = medias[braco].get(dim, {}).get("media")
            if valor is not None:
                texto.append(f"{braco} = {fmt(valor, 2)} ({valor - m_real[dim]:+.2f} vs real)")
        partes.append("; ".join(texto) + ".\n")
    partes.append(
        "- A âncora real é o que dá escala ao julgamento: **estar perto dela** é "
        "mais informativo que a nota absoluta.\n"
    )
    partes.append(
        "- Diferenças com IC contendo zero não devem ser tratadas como "
        "vantagem, mesmo que a média favoreça um braço.\n"
    )
    partes.append("\n## 7. Limitações\n")
    partes.append(
        "- **Blocking por nome:** ver as 4 versões do mesmo nome reduz "
        "variância, mas pode ancorar o julgamento. A ordem dentro do bloco é "
        "aleatória e o rótulo do braço não aparece.\n"
        "- **Amostra de conveniência:** 3–5 avaliadores não são uma amostra "
        "aleatória de jogadores; os ICs refletem itens e blocos, não a "
        "população de avaliadores.\n"
        "- **Recorte do M2:** só entram blocos em que o refino mudou o texto. "
        "Isso mede o efeito do refino **quando ele acontece**, não o M2 como "
        "entregue (11,2% dos casos ficam de fora).\n"
        "- **Sem mascarar o nome:** o nome é o mesmo nas 4 cartas; a carta real "
        "pode ser reconhecida por jogadores experientes. É um risco aceito e "
        "declarado.\n"
    )
    return "\n".join(partes)


def simular(pasta: Path, n_avaliadores: int = 3) -> None:
    """Gera respostas sintéticas para testar o pipeline de análise."""
    gabarito = carregar_gabarito()
    pasta.mkdir(parents=True, exist_ok=True)
    base = {"REAL": 4.2, "M2": 3.2, "M3": 3.4, "M1": 2.7}
    rng = np.random.default_rng(SEED)
    for indice in range(1, n_avaliadores + 1):
        nome = f"simulado{indice}"
        vies = rng.normal(0, 0.25)
        with open(pasta / f"{nome}.csv", "w", encoding="utf-8", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["avaliador", "item_id", "realismo", "coerencia", "criatividade", "existe"])
            for item_id, info in gabarito.items():
                braco = info["braco"]
                notas = [
                    int(np.clip(round(base[braco] + vies + rng.normal(0, 0.7)), 1, 5)),
                    int(np.clip(round(base[braco] + vies + rng.normal(0, 0.7)), 1, 5)),
                    int(np.clip(round(3.5 + rng.normal(0, 0.8)), 1, 5)),
                ]
                existe = "sim" if rng.random() < {"REAL": 0.85, "M2": 0.55, "M3": 0.6, "M1": 0.3}[braco] else "nao"
                escritor.writerow([nome, item_id, *notas, existe])
    print(f"OK: {n_avaliadores} avaliadores simulados em {pasta}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respostas", type=Path, default=DIR / "respostas")
    parser.add_argument("--simular", type=int, default=0,
                        help="gera N avaliadores sintéticos e analisa (teste do pipeline)")
    args = parser.parse_args()

    pasta = args.respostas
    if args.simular:
        pasta = DIR / "respostas_simuladas"
        simular(pasta, args.simular)

    gabarito = carregar_gabarito()
    registros, problemas, arquivos = carregar_respostas(pasta, gabarito)
    metricas = analisar(registros, arquivos, problemas)

    if args.simular:
        saida_json = pasta / "metricas_simuladas.json"
        saida_md = pasta / "relatorio_simulado.md"
    else:
        saida_json = DIR / "metricas_avaliacao_humana.json"
        saida_md = DIR / "relatorio_avaliacao_humana.md"

    saida_json.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saida_md.write_text(relatorio(metricas, simulado=bool(args.simular)), encoding="utf-8")
    print(f"OK: {saida_json}")
    print(f"OK: {saida_md}")


if __name__ == "__main__":
    main()
