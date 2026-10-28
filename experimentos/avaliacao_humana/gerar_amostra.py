#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gera a amostra cega da avaliação humana (P-04).

Desenho: **blocos de 4 cartas com o mesmo nome** — três geradas (M1, M2, M3) e
uma **carta real** como âncora. O avaliador não sabe qual é qual; a única coisa
que muda entre as quatro é o conteúdo da carta.

Por que a âncora real: sem ela, "nota 3,8" não significa nada. Com ela, dá para
dizer se um braço fica acima ou abaixo da carta verdadeira no mesmo bloco.

Por que só nomes reais: o bloco precisa de uma carta real para a âncora, e
nomes fictícios não têm oracle text. Além disso, o julgamento é sobre a carta,
não sobre o nome.

Por que só blocos em que o M2 refiniu: em 112/1000 o refino devolveu o texto do
Markov. Incluí-los deixaria M1 e M2 idênticos em ~11% dos blocos e diluiria o
efeito que o M2 existe para medir. O protocolo declara isso como recorte.

Saídas em `experimentos/avaliacao_humana/`:

- `itens_cegos.json` — o que o avaliador vê (sem rótulo de braço);
- `gabarito.json` — o mapa item → braço. **Não abrir antes de coletar.**
- `formulario.html` — formulário offline, autocontido, com exportação CSV;
- `modelo_respostas.csv` — cabeçalho vazio, se preferirem Google Forms.

Uso:
    python experimentos/avaliacao_humana/gerar_amostra.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
EXP = RAIZ / "experimentos"
DIR = EXP / "avaliacao_humana"
CORPUS = RAIZ / "data" / "processed" / "cards.parquet"

SEED = 20261026
N_BLOCOS = 40
BRACOS = ("M1", "M2", "M3")

TIPO_EN = {
    "creature": "Creature",
    "instant": "Instant",
    "sorcery": "Sorcery",
    "artifact": "Artifact",
    "enchantment": "Enchantment",
    "land": "Land",
}

# Campos que o avaliador vê. `texto` pode ter parágrafos.
CAMPOS_CARTA = ("nome", "custo", "tipo_linha", "texto", "pt")


def carregar(modelo: str) -> Dict[int, dict]:
    """Última linha de cada `ordem` vence (mesma regra de `avaliar.py`)."""
    caminho = EXP / f"resultados_{modelo.lower()}.jsonl"
    registros: Dict[int, dict] = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if "erro" in registro or "ordem" not in registro:
                continue
            registros[registro["ordem"]] = registro
    return registros


def carregar_corpus() -> Dict[str, dict]:
    df = pd.read_parquet(
        CORPUS,
        columns=["name", "oracle_text", "type_line", "mana_cost", "power", "toughness"],
    )
    df = df.dropna(subset=["oracle_text"]).drop_duplicates("name")
    return {
        linha["name"]: {
            "custo": linha["mana_cost"] if isinstance(linha["mana_cost"], str) else "",
            "tipo_linha": linha["type_line"] or "",
            "texto": linha["oracle_text"] or "",
            "pt": (
                f"{linha['power']}/{linha['toughness']}"
                if isinstance(linha["power"], str) and isinstance(linha["toughness"], str)
                else ""
            ),
        }
        for _, linha in df.iterrows()
    }


def carta_gerada(registro: dict) -> dict:
    """Monta um item legível a partir de um registro de braço."""
    tipo = TIPO_EN.get(registro.get("tipo_forcado"), registro.get("tipo", ""))
    subtipos = registro.get("subtipos") or []
    tipo_linha = tipo + (f" — {' '.join(subtipos)}" if subtipos else "")
    poder, resistencia = registro.get("poder"), registro.get("resistencia")
    pt = (
        f"{poder}/{resistencia}"
        if poder is not None and resistencia is not None
        else ""
    )
    return {
        "nome": registro["nome_entrada"],
        "custo": registro.get("custo_mana") or "",
        "tipo_linha": tipo_linha,
        "texto": (registro.get("texto_regras") or "").strip(),
        "pt": pt,
    }


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    m1, m2, m3 = (carregar(b) for b in BRACOS)
    corpus = carregar_corpus()

    comuns = sorted(set(m1) & set(m2) & set(m3))
    candidatos: List[int] = []
    for ordem in comuns:
        if m1[ordem].get("tipo_experimento") != "real":
            continue
        if not m2[ordem].get("refinado"):
            continue
        nome = m1[ordem]["nome_entrada"]
        if nome not in corpus:
            continue
        if any(not (r.get("texto_regras") or "").strip() for r in (m1[ordem], m2[ordem], m3[ordem])):
            continue
        if any(r.get("tipo_linha") is None and not r.get("subtipos") for r in (m1[ordem], m2[ordem], m3[ordem])):
            continue
        candidatos.append(ordem)

    if len(candidatos) < N_BLOCOS:
        raise SystemExit(f"Poucos blocos elegíveis: {len(candidatos)}")

    rng = random.Random(SEED)
    blocos_ordens = sorted(rng.sample(candidatos, N_BLOCOS))
    rng.shuffle(blocos_ordens)

    itens: List[dict] = []
    mapa: Dict[str, dict] = {}
    blocos_meta: List[dict] = []
    seq = 0
    for indice, ordem in enumerate(blocos_ordens, start=1):
        cartas = {
            "M1": carta_gerada(m1[ordem]),
            "M2": carta_gerada(m2[ordem]),
            "M3": carta_gerada(m3[ordem]),
            "REAL": {**corpus[m1[ordem]["nome_entrada"]], "nome": m1[ordem]["nome_entrada"]},
        }
        ordem_itens = list(cartas)
        rng.shuffle(ordem_itens)
        ids = []
        for braco in ordem_itens:
            seq += 1
            item_id = f"i{seq:03d}"
            item = {
                "item_id": item_id,
                "bloco": indice,
                "nome": cartas[braco]["nome"],
                **{campo: cartas[braco][campo] for campo in CAMPOS_CARTA if campo != "nome"},
            }
            itens.append(item)
            mapa[item_id] = {"braco": braco, "ordem": ordem, "nome": cartas[braco]["nome"]}
            ids.append(item_id)
        blocos_meta.append({"bloco": indice, "ordem": ordem, "itens": ids})

    (DIR / "itens_cegos.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_blocos": N_BLOCOS,
                "instrucoes": (
                    "Cada bloco traz 4 versões do MESMO nome de carta: três "
                    "geradas por modelos e uma que é a carta real. Não há "
                    "identificação de qual é qual."
                ),
                "itens": itens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (DIR / "gabarito.json").write_text(
        json.dumps(
            {"seed": SEED, "n_blocos": N_BLOCOS, "blocos": blocos_meta, "itens": mapa},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (DIR / "modelo_respostas.csv").write_text(
        "avaliador,item_id,realismo,coerencia,criatividade,existe\n", encoding="utf-8"
    )
    (DIR / "formulario.html").write_text(
        _html(itens, N_BLOCOS), encoding="utf-8"
    )

    por_braco = {b: sum(1 for v in mapa.values() if v["braco"] == b) for b in (*BRACOS, "REAL")}
    print(f"OK: {N_BLOCOS} blocos, {len(itens)} cartas")
    print(f"OK: por braço {por_braco}")
    print(f"OK: {DIR / 'formulario.html'}")
    print(f"OK: {DIR / 'gabarito.json'} (não abrir antes de coletar)")


def _html(itens: List[dict], n_blocos: int) -> str:
    """Formulário offline, autocontido, com exportação CSV."""
    dados = json.dumps(itens, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Avaliação cega de cartas geradas</title>
<style>
  :root {{ --bg:#f4f1ea; --card:#fffdf8; --ink:#22201c; --muted:#6b6459;
           --line:#d8d0c2; --accent:#1f5f4a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:15px/1.5 "Segoe UI", system-ui, sans-serif; }}
  header {{ position:sticky; top:0; z-index:5; background:var(--bg);
            border-bottom:1px solid var(--line); padding:14px 20px;
            display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
  h1 {{ font-size:17px; margin:0; }}
  .barra {{ flex:1; height:8px; background:#e4ded2; border-radius:99px; min-width:120px; }}
  .barra > i {{ display:block; height:100%; width:0; background:var(--accent);
                border-radius:99px; transition:width .2s; }}
  main {{ max-width:1180px; margin:0 auto; padding:22px 18px 80px; }}
  .bloco {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }}
  .carta {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.05); }}
  .carta h3 {{ margin:0 0 8px; font-size:15px; }}
  .custo {{ float:right; font-weight:600; }}
  .tipo {{ color:var(--muted); font-size:12.5px; margin:0 0 8px; }}
  .texto {{ white-space:pre-wrap; font-size:13.5px; min-height:76px;
            border-top:1px solid var(--line); border-bottom:1px solid var(--line);
            padding:8px 0; margin:0 0 8px; }}
  .pt {{ float:right; font-weight:700; }}
  .campo {{ display:flex; align-items:center; justify-content:space-between;
            gap:8px; margin:6px 0; font-size:13px; }}
  .escala {{ display:flex; gap:4px; }}
  .escala label {{ cursor:pointer; }}
  .escala input {{ position:absolute; opacity:0; }}
  .escala span {{ display:block; width:26px; height:26px; line-height:26px;
                  text-align:center; border:1px solid var(--line); border-radius:6px;
                  font-size:12px; }}
  .escala input:checked + span {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .simnao {{ display:flex; gap:6px; }}
  .simnao label {{ cursor:pointer; }}
  .simnao input {{ position:absolute; opacity:0; }}
  .simnao span {{ display:block; padding:3px 10px; border:1px solid var(--line);
                  border-radius:6px; font-size:12.5px; }}
  .simnao input:checked + span {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .acoes {{ display:flex; gap:10px; margin-top:18px; align-items:center; }}
  button {{ font:inherit; padding:9px 16px; border-radius:8px; border:1px solid var(--accent);
            background:var(--accent); color:#fff; cursor:pointer; }}
  button.sec {{ background:transparent; color:var(--accent); }}
  button:disabled {{ opacity:.4; cursor:not-allowed; }}
  .dica {{ color:var(--muted); font-size:13px; max-width:760px; }}
  .aviso {{ background:#fff6d8; border:1px solid #e5cf8a; padding:10px 12px;
            border-radius:8px; font-size:13px; }}
  .fim {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }}
  code {{ background:#eee9dd; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<header>
  <h1>Avaliação cega — cartas geradas</h1>
  <div class="barra"><i id="progresso"></i></div>
  <span id="contador" class="dica"></span>
</header>
<main id="app"></main>
<script>
const ITENS = {dados};
const N_BLOCOS = {n_blocos};
const CHAVE = "avaliacao-cartas-v1";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
const DIMENSOES = [
  ["realismo", "Parece uma carta real de Magic?"],
  ["coerencia", "O texto é internamente coerente?"],
  ["criatividade", "A mecânica é interessante/não óbvia?"]
];
const ROTULOS = ["1", "2", "3", "4", "5"];
let estado = carregar();

function carregar() {{
  try {{
    const bruto = JSON.parse(localStorage.getItem(CHAVE) || "null");
    if (bruto && bruto.respostas) return bruto;
  }} catch (e) {{}}
  return {{ avaliador: "", indice: 0, respostas: {{}} }};
}}
function salvar() {{ localStorage.setItem(CHAVE, JSON.stringify(estado)); }}
function blocos() {{
  const mapa = new Map();
  for (const item of ITENS) {{
    if (!mapa.has(item.bloco)) mapa.set(item.bloco, []);
    mapa.get(item.bloco).push(item);
  }}
  return [...mapa.entries()].sort((a,b) => a[0]-b[0]).map(([n, itens]) => ({{ n, itens }}));
}}
function completo(item) {{
  const r = estado.respostas[item.item_id] || {{}};
  return DIMENSOES.every(([d]) => r[d] >= 1 && r[d] <= 5) && (r.existe === "sim" || r.existe === "nao");
}}
function atualizarBarra() {{
  const total = ITENS.length;
  const feitos = ITENS.filter(completo).length;
  document.getElementById("progresso").style.width = (100*feitos/total).toFixed(1) + "%";
  document.getElementById("contador").textContent = feitos + "/" + total + " cartas";
}}
function inicio() {{
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="fim">
      <h2 style="margin-top:0">Antes de começar</h2>
      <p class="dica">Você vai avaliar <b>${N_BLOCOS} blocos</b>. Em cada bloco há
      <b>4 versões do mesmo nome de carta</b>: três foram geradas por modelos e
      uma é a carta real. Você não sabe qual é qual — e não precisa saber.</p>
      <p class="dica">Para cada carta, dê três notas de 1 a 5 e diga se ela
      poderia existir em uma coleção de Magic. Não existe resposta "certa":
      julgue como jogador.</p>
      <p class="dica">O progresso é salvo neste navegador. Pode fechar e voltar
      depois no mesmo computador/navegador.</p>
      <p class="aviso">Se você já avaliou cartas deste projeto antes, use o
      mesmo nome para retomar.</p>
      <div class="acoes">
        <input id="nome" placeholder="Seu nome (ou apelido)" value="${{estado.avaliador.replace(/"/g,"&quot;")}}"
               style="font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:8px">
        <button id="comecar">Começar / retomar</button>
      </div>
    </div>`;
  document.getElementById("comecar").onclick = () => {{
    const nome = document.getElementById("nome").value.trim();
    if (!nome) return alert("Informe seu nome ou apelido.");
    estado.avaliador = nome;
    salvar();
    render();
  }};
}}
function render() {{
  if (!estado.avaliador) return inicio();
  const lista = blocos();
  if (estado.indice >= lista.length) return fim();
  const {{ n, itens }} = lista[estado.indice];
  const app = document.getElementById("app");
  app.innerHTML = `
    <h2 style="margin:0 0 4px">Bloco ${{estado.indice+1}} de ${{lista.length}}</h2>
    <p class="dica">Quatro versões de <b>${{esc(itens[0].nome)}}</b>. Avalie cada
    uma separadamente.</p>
    <div class="bloco">${{itens.map(cartaHTML).join("")}}</div>
    <div class="acoes">
      <button id="voltar" class="sec" ${{estado.indice === 0 ? "disabled" : ""}}>Voltar</button>
      <button id="proxima">Próximo bloco</button>
      <span class="dica" id="pendente"></span>
    </div>`;
  document.getElementById("voltar").onclick = () => {{ estado.indice--; salvar(); render(); }};
  document.getElementById("proxima").onclick = () => {{
    if (!itens.every(completo)) {{
      document.getElementById("pendente").textContent = "Faltam notas neste bloco.";
      return;
    }}
    estado.indice++; salvar(); render();
  }};
  app.querySelectorAll("input").forEach(entrada => {{
    entrada.onchange = () => {{
      const {{ item, dimensao, valor }} = entrada.dataset;
      const resp = estado.respostas[item] || {{}};
      resp[dimensao] = dimensao === "existe" ? valor : Number(valor);
      estado.respostas[item] = resp;
      salvar();
      atualizarBarra();
      document.getElementById("pendente").textContent = "";
    }};
  }});
  atualizarBarra();
  window.scrollTo(0,0);
}}
function cartaHTML(item) {{
  const resp = estado.respostas[item.item_id] || {{}};
  const grupos = DIMENSOES.map(([dim, rotulo]) => `
    <div class="campo"><span>${{rotulo}}</span>
      <div class="escala">${{ROTULOS.map(v => `
        <label><input type="radio" name="${{item.item_id}}-${{dim}}" data-item="${{item.item_id}}"
          data-dimensao="${{dim}}" data-valor="${{v}}" ${{resp[dim] === Number(v) ? "checked" : ""}}>
        <span>${{v}}</span></label>`).join("")}}</div>
    </div>`).join("");
  return `
    <article class="carta">
      <h3>${{esc(item.nome)}} <span class="custo">${{esc(item.custo)}}</span></h3>
      <p class="tipo">${{esc(item.tipo_linha)}} <span class="pt">${{esc(item.pt)}}</span></p>
      <p class="texto">${{esc(item.texto)}}</p>
      ${{grupos}}
      <div class="campo"><span>Poderia existir?</span>
        <div class="simnao">
          ${{["sim","nao"].map(v => `
            <label><input type="radio" name="${{item.item_id}}-existe" data-item="${{item.item_id}}"
              data-dimensao="existe" data-valor="${{v}}" ${{resp.existe === v ? "checked" : ""}}>
            <span>${{v === "sim" ? "Sim" : "Não"}}</span></label>`).join("")}}
        </div>
      </div>
    </article>`;
}}
function fim() {{
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="fim">
      <h2 style="margin-top:0">Terminou — obrigado!</h2>
      <p class="dica">Clique para baixar o CSV e envie para quem coordenou o
      estudo. O arquivo inclui só suas notas, sem identificação das versões.</p>
      <div class="acoes">
        <button id="baixar">Baixar minhas respostas (CSV)</button>
        <button id="rever" class="sec">Rever respostas</button>
      </div>
    </div>`;
  document.getElementById("baixar").onclick = baixar;
  document.getElementById("rever").onclick = () => {{ estado.indice = 0; salvar(); render(); }};
}}
function baixar() {{
  const linhas = [["avaliador","item_id","realismo","coerencia","criatividade","existe"]];
  for (const item of ITENS) {{
    const r = estado.respostas[item.item_id] || {{}};
    linhas.push([estado.avaliador, item.item_id, r.realismo ?? "", r.coerencia ?? "",
                 r.criatividade ?? "", r.existe ?? ""]);
  }}
  const csv = "\\ufeff" + linhas.map(l => l.map(v => `"${{String(v).replace(/"/g,'""')}}"`).join(",")).join("\\n");
  const url = URL.createObjectURL(new Blob([csv], {{type:"text/csv;charset=utf-8"}}));
  const a = document.createElement("a");
  a.href = url;
  a.download = `respostas_${{estado.avaliador.replace(/[^a-z0-9]+/gi,"_")}}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}}
inicio();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
