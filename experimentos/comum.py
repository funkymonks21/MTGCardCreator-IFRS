#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilitários compartilhados pelos runners do estudo (M1, M2 e M3).

Por que este módulo existe
==========================
Os três runners precisam das MESMAS convenções. Duplicar essas regras em
três arquivos foi a origem de dois bugs reais no primeiro teste:

1. `config_v1.json` guarda a cor como código Scryfall (`W/U/B/R/G`), mas
   `pipeline.gerar_carta` espera o nome em português (`Branco`, ...). Sem o
   mapa, todas as cartas falhavam com "Cor inválida".
2. `Carta` não expõe `cmc`. O custo de mana previsto está em
   `Carta.custo_mana` (formato Oracle, ex.: `{2}{W}{W}`); o CMC tem de ser
   derivado dele. Acessar `carta.cmc` gerava `AttributeError` em 100% das
   cartas.

Além disso, a semente usada no primeiro teste era `hash((nome, modelo, i))`.
`hash()` de `str` no CPython é **salgado por processo** (PYTHONHASHSEED
aleatório), então a mesma carta recebia sementes diferentes a cada execução
— o experimento deixava de ser reproduzível. Aqui a semente vem de um hash
criptográfico do identificador da carta, que é estável entre máquinas e
execuções.

Decisão de pareamento (M1 × M2)
===============================
D-002 menciona `seed = hash(carta_id, modelo)`. Na prática, usar modelos
diferentes no hash daria sementes diferentes para a MESMA carta, e então a
diferença entre M1 e M2 misturaria (a) o efeito do refinamento e (b) ruído
de amostragem do Markov. Como o desenho quer isolar o efeito do texto
(O M2 = M1 com o texto refinado), os dois braços usam a MESMA semente para
a mesma carta. Assim a base Markov é idêntica e a única diferença
sistemática é o refino. Ver a nota de revisão no `README.md` desta pasta.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# Diretórios do projeto (este arquivo vive em experimentos/).
RAIZ = Path(__file__).resolve().parents[1]
EXPERIMENTOS = RAIZ / "experimentos"
CONFIG = EXPERIMENTOS / "config_v1.json"

# Código de cor do Scryfall -> nome usado pelo pipeline.
MAP_COR = {
    "W": "Branco",
    "U": "Azul",
    "B": "Preto",
    "R": "Vermelho",
    "G": "Verde",
}

# Custo de mana -> CMC. O pipeline só emite `{N}` genérico e `{C}` colorido
# (ver estrutura._formatar_custo), então a conversão é exata e sem ambiguidade.
_SIMBOLO_MANA = re.compile(r"\{([^}]*)\}")


def cor_do_pipeline(cor: str) -> str:
    """Traduz `W/U/B/R/G` para o nome em português aceito por `gerar_carta`.

    Aceita também o nome em português (devolve inalterado) para que chamadas
    antigas continuem funcionando."""
    if cor in MAP_COR:
        return MAP_COR[cor]
    if cor in MAP_COR.values():
        return cor
    raise ValueError(
        f"Cor desconhecida: {cor!r}. Esperado um de {sorted(MAP_COR)} "
        f"ou {sorted(MAP_COR.values())}."
    )


def normalizar_custo(custo_mana: str) -> str:
    """Converte notações compactas de custo para o formato Oracle.

    O M1/M2 já emitem `{2}{W}`; o LLM às vezes emite `2W`, `10WU` ou `2 G`.
    Sem normalizar, `cmc_do_custo("2W")` daria 0 e o CMC do M3 ficaria
    artificialmente errado — um artefato de FORMATO, não de conteúdo.

    A função só acrescenta as chaves; **não** mexe nos símbolos. Se a string
    não casar com o padrão esperado, é devolvida como veio (e o lint acusa).
    """
    texto = (custo_mana or "").strip()
    if not texto:
        return ""
    if texto.lower() in {"none", "null", "n/a", "na", "-", "—", "sem custo", "no cost"}:
        return ""
    if "{" in texto:
        # Já está no formato Oracle: só remove espaço entre símbolos.
        return re.sub(r"\s+", "", texto)
    simbolos = re.findall(r"\d+|[WUBRGX]", texto.upper())
    if not simbolos or "".join(simbolos) != re.sub(r"[\s,;/]+", "", texto.upper()):
        return texto
    return "".join(f"{{{s}}}" for s in simbolos)


def cmc_do_custo(custo_mana: str) -> int:
    """Calcula o custo de mana convertido a partir do texto Oracle.

    `{2}{W}{W}` -> 4; `{W}` -> 1; `""` (terreno) -> 0. Símbolos que o
    pipeline não gera (`{X}`, híbridos) são ignorados em vez de quebrar —
    a função é usada só para registrar o CMC previsto nas métricas."""
    if not custo_mana:
        return 0
    total = 0
    for simbolo in _SIMBOLO_MANA.findall(custo_mana):
        if simbolo.isdigit():
            total += int(simbolo)
        elif simbolo in "WUBRG":
            total += 1
    return total


def semente(carta_id: str) -> int:
    """Semente determinística e estável entre processos/máquinas.

    `hash()` embutido NÃO serve: é salgado por processo. Usamos SHA-256 do
    identificador e ficamos com 32 bits, que é o que `random.Random` precisa.
    O MESMO `carta_id` deve ser passado por M1 e M2 (ver docstring do módulo).
    """
    digest = hashlib.sha256(carta_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def criar_log(caminho: Path):
    """Devolve `log(msg)` que escreve com carimbo de hora no stdout e no
    arquivo indicado (append). Centraliza o formato dos três runners."""
    caminho.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        carimbo = datetime.now().strftime("%H:%M:%S")
        linha = f"[{carimbo}] {msg}"
        print(linha, flush=True)
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(linha + "\n")

    return log


def carregar_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def _itens_reais(cfg: dict) -> Iterator[dict]:
    """500 cartas reais (gabarito completo, retiradas do treino — D-006)."""
    for i, carta in enumerate(cfg["cartas"]["avaliadas_reais"]):
        yield {
            "ordem": i,
            "carta_id": f"real::{carta['name']}",
            "tipo_experimento": "real",
            "nome": carta["name"],
            "cor": carta["cor"],
            "tipo_forcado": carta["type"],
            "gabarito": carta,
        }


def _itens_ficticios(cfg: dict) -> Iterator[dict]:
    """500 cartas fictícias: mesmo perfil (tipo x cor), nome inventado.

    Não têm gabarito de CMC/P-T por construção (D-006 §9.7) — servem para
    criatividade/coerência e como controle de memorização."""
    for j, ficticia in enumerate(cfg["ficticias"]):
        yield {
            "ordem": 500 + j,
            "carta_id": f"ficticio::{ficticia['name']}",
            "tipo_experimento": "ficticio",
            "nome": ficticia["name"],
            "cor": ficticia["cor"],
            "tipo_forcado": ficticia["type"],
            "gabarito": None,
        }


def carregar_itens() -> List[dict]:
    """Lista única de 1000 itens (500 reais + 500 fictícios), na ordem em que
    aparecem no `config_v1.json`. Cada item traz a semente já resolvida."""
    cfg = carregar_config()
    itens = list(_itens_reais(cfg)) + list(_itens_ficticios(cfg))
    for item in itens:
        item["seed"] = semente(item["carta_id"])
    return itens


def ordens_concluidas(caminho: Path) -> set:
    """Lê um `.jsonl` de resultados e devolve as `ordem` já concluídas com
    sucesso (linhas com `erro` não contam). Base da retomada dos runners."""
    concluidas = set()
    if not caminho.exists():
        return concluidas
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if "ordem" in registro and "erro" not in registro:
                concluidas.add(registro["ordem"])
    return concluidas


def campo_carta(carta) -> Dict[str, Optional[object]]:
    """Extrai de um `Carta` só os campos que existem de fato.

    Mantido num único lugar para que M1 e M2 registrem exatamente as mesmas
    colunas — condição necessária para a comparação pareada. `cmc` é
    DERIVADO de `custo_mana` (ver `cmc_do_custo`), nunca lido do objeto.
    """
    return {
        "nome_gerado": carta.nome,
        "tipo": carta.tipo,
        "tipo_linha": carta.tipo_linha,
        "custo_mana": carta.custo_mana,
        "cmc": cmc_do_custo(carta.custo_mana),
        "texto_regras": carta.texto_regras,
        "palavras_chave": list(carta.palavras_chave),
        "poder": carta.poder,
        "resistencia": carta.resistencia,
        "subtipos": list(carta.subtipos),
    }
