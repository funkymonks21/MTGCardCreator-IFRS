import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path('.').resolve()
DATA = ROOT / "data" / "processed" / "cards.parquet"
OUT_JSON = ROOT / "experimentos" / "config_v1.json"
OUT_MD = ROOT / "experimentos" / "resumo_amostragem.md"

TIPOS = ["creature", "instant", "sorcery", "enchantment", "artifact", "land"]
CORES = ["W", "U", "B", "R", "G"]
NOMES_COR = {"W": "Branco", "U": "Azul", "B": "Preto", "R": "Vermelho", "G": "Verde"}

POOL_850 = {
    "creature": {"W": 86, "U": 86, "B": 86, "R": 86, "G": 86},
    "instant": {"W": 21, "U": 21, "B": 21, "R": 21, "G": 21},
    "sorcery": {"W": 20, "U": 20, "B": 20, "R": 19, "G": 19},
    "enchantment": {"W": 19, "U": 19, "B": 19, "R": 20, "G": 20},
    "artifact": {"W": 12, "U": 12, "B": 12, "R": 12, "G": 12},
    "land": {"W": 12, "U": 12, "B": 12, "R": 12, "G": 12},
}
EVAL_500 = {t: {c: (40 if t == "creature" else 12) for c in CORES} for t in TIPOS}
SEMENTES = {"pool": 123456789, "eval": 987654321, "ficticios": 555444333}


def tipo_chave(type_line: str):
    tl = str(type_line)
    for chave, valor in (
        ("Creature", "creature"), ("Instant", "instant"), ("Sorcery", "sorcery"),
        ("Artifact", "artifact"), ("Enchantment", "enchantment"), ("Land", "land"),
    ):
        if chave in tl:
            return valor
    return None


def _como_lista(valor):
    if valor is None:
        return []
    if isinstance(valor, str):
        return [valor] if valor else []
    try:
        return [x for x in valor if x]
    except TypeError:
        return []


def cor_mono(row):
    tipo = tipo_chave(row["type_line"])
    campo = "color_identity" if tipo == "land" else "colors"
    cores = _como_lista(row.get(campo))
    return cores[0] if len(cores) == 1 else None


def montar_pool(mono: pd.DataFrame, rng):
    por_celula = defaultdict(list)
    for idx, row in mono.iterrows():
        por_celula[(row["_t"], row["_cor"])].append(idx)
    escolhidos = []
    for tipo in TIPOS:
        for cor in CORES:
            precisa = POOL_850[tipo][cor]
            cand = por_celula[(tipo, cor)][:]
            if len(cand) < precisa:
                raise RuntimeError(f"pool {tipo}/{cor}")
            rng.shuffle(cand)
            escolhidos.extend(cand[:precisa])
    return escolhidos


def montar_avaliadas(pool_df: pd.DataFrame, rng):
    por_celula = defaultdict(list)
    for idx, row in pool_df.iterrows():
        por_celula[(row["_t"], row["_cor"])].append(idx)
    escolhidos = []
    for tipo in TIPOS:
        for cor in CORES:
            precisa = EVAL_500[tipo][cor]
            cand = por_celula[(tipo, cor)][:]
            if len(cand) < precisa:
                raise RuntimeError(f"eval {tipo}/{cor}")
            rng.shuffle(cand)
            escolhidos.extend(cand[:precisa])
    return escolhidos


def _carregar_vocabularios(df: pd.DataFrame):
    TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")
    STOP = {"the", "of", "a", "an", "and", "in", "to", "for", "from", "at", "on",
            "with", "by", "into", "over", "under", "s"}
    PROIB_HEAD = {"up","down","out","away","end","past","dead","class","artifact",
                  "three","one","two","everythingamajig","superlatorium","you",
                  "ages","room","roads","off","on","back","car","empires","self","upkeep"}
    PROIB_POR_TIPO_HEAD = {
        "land": {"forest", "island", "swamp", "mountain", "plains", "wastes",
                "dragon", "triome", "snarl", "verge", "guildgate", "campus",
                "commons", "gate", "isle", "marsh", "bog", "moor", "cataract",
                "village", "ruins", "keep", "volcano", "spires", "ridge", "grove",
                "woodland", "garden", "thicket", "lodge", "encampment", "steppe",
                "meadow", "cho", "bridge", "outpost", "trokair", "giants",
                "conclave", "edge", "sanctum", "stronghold", "gorge", "battlements",
                "valley", "fens", "crossroads", "raceway", "foundry", "fen", "quarry"},
        "artifact": {"bobblehead", "cluestone", "keyrune", "spellbomb", "suit",
                     "cannon", "anchor", "gauntlet", "lens", "map", "wand"},
        "enchantment": {"zendikon", "infestation"},
    }

    entradas = []
    for _, r in df.iterrows():
        t = tipo_chave(r["type_line"])
        if t not in TIPOS:
            continue
        campo = "color_identity" if t == "land" else "colors"
        cs = _como_lista(r.get(campo))
        if len(cs) != 1 or cs[0] not in CORES:
            continue
        entradas.append((t, cs[0], r["name"]))

    mods_cnt = defaultdict(Counter)
    heads_cnt = defaultdict(Counter)

    for t, c, nome in entradas:
        ps = [p for p in TOKEN.findall(str(nome)) if p.lower() not in STOP and len(p) >= 3]
        if len(ps) != 2:
            continue
        modp, headp = ps[0], ps[1]
        if len(modp) > 12 or len(headp) > 12:
            continue
        if "'" in modp or "-" in modp or "'" in headp or "-" in headp:
            continue
        mods_cnt[(t, c)][modp] += 1
        heads_cnt[(t, c)][headp] += 1

    mods_por_celula = {}
    heads_por_celula = {}
    for t in TIPOS:
        for c in CORES:
            mc = mods_cnt[(t, c)]
            hc = heads_cnt[(t, c)]
            mods_por_celula[(t, c)] = [w for w, n in mc.most_common(800) if w.lower() not in PROIB_HEAD]
            heads_por_celula[(t, c)] = [w for w, n in hc.most_common(800) if w.lower() not in PROIB_HEAD | PROIB_POR_TIPO_HEAD.get(t, set())]
            if len(mods_por_celula[(t, c)]) < 15 or len(heads_por_celula[(t, c)]) < 15:
                raise RuntimeError(f"vocab {t}/{c} {len(mods_por_celula[(t,c)])}/{len(heads_por_celula[(t,c)])}")
    return mods_por_celula, heads_por_celula


def gerar_nomes_ficticios(df: pd.DataFrame, rng, total_por_celula):
    existentes = {str(n).lower() for n in df["name"].dropna()}
    mods_por_celula, heads_por_celula = _carregar_vocabularios(df)
    usados = set()
    saida = []
    for tipo in TIPOS:
        for cor in CORES:
            precisa = total_por_celula[(tipo, cor)]
            mods = mods_por_celula[(tipo, cor)]
            heads = heads_por_celula[(tipo, cor)]
            for _ in range(precisa):
                nome = None
                for _ in range(20000):
                    mod = rng.choice(mods)
                    head = rng.choice(heads)
                    cand = f"{mod} {head}"
                    if len(cand.split()) != 2:
                        continue
                    chave = cand.lower()
                    if chave in existentes or chave in usados:
                        continue
                    nome = cand
                    break
                if nome is None:
                    raise RuntimeError(f"nome {tipo}/{cor}")
                usados.add(nome.lower())
                saida.append({"name": nome, "type": tipo, "cor": cor})
    return saida


def tabela_md(tab: pd.DataFrame):
    cabecalho = "| " + " | ".join([str(tab.index.name or "")] + [str(c) for c in tab.columns]) + " |"
    linha = "|" + "---|" * (len(tab.columns) + 1)
    corpo = ["| " + " | ".join([str(rotulo)] + [str(v) for v in linha_valores]) + " |" for rotulo, linha_valores in zip(tab.index, tab.values)]
    return "\n".join([cabecalho, linha] + corpo)


def main():
    import random
    rng_pool = random.Random(SEMENTES["pool"])
    rng_eval = random.Random(SEMENTES["eval"])
    rng_fic = random.Random(SEMENTES["ficticios"])

    df = pd.read_parquet(DATA)
    df["_t"] = df["type_line"].map(tipo_chave)
    df = df[df["_t"].isin(TIPOS)].copy()
    df["_cor"] = df.apply(cor_mono, axis=1)
    mono = df[df["_cor"].notna()].copy()

    pool_ids = montar_pool(mono, rng_pool)
    pool_df = mono.loc[pool_ids].copy()
    eval_ids = montar_avaliadas(pool_df, rng_eval)
    eval_set = set(eval_ids)
    reserva_ids = sorted(set(pool_ids) - eval_set)
    eval_df = mono.loc[eval_ids].copy()
    reserva_df = mono.loc[reserva_ids].copy()

    perfil_fic = {(t, c): EVAL_500[t][c] for t in TIPOS for c in CORES}
    ficticios = gerar_nomes_ficticios(df, rng_fic, perfil_fic)

    total_por_tipo = {t: sum(EVAL_500[t].values()) for t in TIPOS}

    def registros(frame: pd.DataFrame, incluir_gabarito: bool):
        out = []
        for _, row in frame.iterrows():
            item = {"name": row["name"], "type": row["_t"], "cor": row["_cor"]}
            if incluir_gabarito:
                item.update({
                    "oracle_id": row.get("oracle_id"),
                    "type_line": row.get("type_line"),
                    "cmc": float(row["cmc"]) if pd.notna(row.get("cmc")) else None,
                    "power": row.get("power"),
                    "toughness": row.get("toughness"),
                    "colors": _como_lista(row.get("colors")),
                    "color_identity": _como_lista(row.get("color_identity")),
                    "keywords": _como_lista(row.get("keywords")),
                })
            out.append(item)
        return out

    config = {
        "versao": "v1",
        "data": "2026-10-23",
        "decisao": "D-006",
        "sementes": SEMENTES,
        "universo_monocor": int(len(mono)),
        "alocacao": {
            "pool_850": {"total": 850, "por_tipo": {t: sum(POOL_850[t].values()) for t in TIPOS}, "por_cor": {c: 170 for c in CORES}},
            "avaliadas_reais": {"total": 500, "por_tipo": total_por_tipo, "por_cor": {c: 100 for c in CORES}},
            "ficticias": {"total": 500, "por_tipo": total_por_tipo, "por_cor": {c: 100 for c in CORES}},
            "reserva": {"total": 350},
        },
        "cartas": {"pool": registros(pool_df, False), "avaliadas_reais": registros(eval_df, True), "reserva": registros(reserva_df, False)},
        "ficticias": ficticios,
    }
    OUT_JSON.write_text(__import__('json').dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK -> " + str(OUT_JSON.relative_to(ROOT)))
    print("pool %d avaliadas %d reserva %d ficticias %d" % (len(pool_ids), len(eval_ids), len(reserva_ids), len(ficticios)))

    linhas = ["# Resumo da amostragem (D-006)\n\n",
              "- Sementes: " + repr(SEMENTES) + "\n",
              "- Universo monocor: " + f"{len(mono):,}" + "\n",
              "- Pool (holdout): " + str(len(pool_ids)) + "\n",
              "- Avaliadas reais: " + str(len(eval_ids)) + "\n",
              "- Reserva: " + str(len(reserva_ids)) + "\n",
              "- Ficticias: " + str(len(ficticios)) + "\n\n",
              "## Avaliadas reais - tipo x cor\n\n"]
    aval = pd.crosstab(eval_df["_t"], eval_df["_cor"]).reindex(index=TIPOS, columns=CORES).fillna(0).astype(int).rename(index=str, columns=NOMES_COR).rename_axis("tipo")
    linhas.append(tabela_md(aval))
    linhas.append("\n\n## Ficticias - tipo x cor\n\n")
    fic_tab = pd.crosstab([f["type"] for f in ficticios], [f["cor"] for f in ficticios]).reindex(index=TIPOS, columns=CORES).fillna(0).astype(int).rename(index=str, columns=NOMES_COR).rename_axis("tipo")
    linhas.append(tabela_md(fic_tab))
    linhas.append("\n\n## Amostra de nomes ficticias\n\n")
    for f in ficticios[:15]:
        linhas.append("- " + f["name"] + " (" + f["type"] + ", " + NOMES_COR[f["cor"]] + ")\n")
    linhas.append("\n## Ficticias por tipo (5 exemplos)\n\n")
    for tipo in TIPOS:
        exemplos = [f["name"] for f in ficticios if f["type"] == tipo][:5]
        linhas.append("- **" + tipo + "**: " + ", ".join(exemplos) + "\n")
    OUT_MD.write_text("".join(linhas), encoding="utf-8")
    print("OK -> " + str(OUT_MD.relative_to(ROOT)))


if __name__ == "__main__":
    main()
