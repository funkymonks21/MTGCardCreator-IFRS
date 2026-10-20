"""
Regras de texto que dependem do tipo/subtipo da carta.

O modelo de Markov aprende o formato geral de cada tipo, mas não garante
100% das regras estruturais do Magic. Este módulo aplica os "lints" que o
projeto pediu:

- Aura (Encantamento — Aura): sempre precisa ter uma linha "Enchant ...".
  Se o texto gerado já tiver uma, ela é preservada; senão, o alvo é
  sorteado das Auras reais do corpus.
- Equipamento (Artefato — Equipment): sempre precisa ter uma habilidade
  "Equip {custo}". Se já existir, é preservada; senão, o custo é sorteado
  dos Equipamentos reais (compatível com a cor da carta).
- Terreno: terrenos com tipos básicos (Plains/Island/Swamp/Mountain/Forest)
  ganham a habilidade de mana correspondente; terrenos sem tipo básico
  ganham uma habilidade de mana da cor escolhida se ainda não tiverem
  nenhuma.

Também há `ajustar_subtipos_por_texto`, que mantém o type_line coerente
com o texto: se o modelo gerou "Enchant ..." em um encantamento, o subtipo
"Aura" é garantido; se gerou "Equip ..." em um artefato, "Equipment" é
garantido.
"""

from __future__ import annotations

import random
import re
from typing import List, TYPE_CHECKING

from .subtipos import (
    MAX_SUBTIPOS,
    TERRENOS_BASICOS,
)

if TYPE_CHECKING:
    from .subtipos import ModeloSubtipos

TEXTO_VAZIO = "(nenhum texto gerado)"

_RE_ENCHANT = re.compile(r"(?i)^enchant\b")
_RE_ENCHANT_VALIDO = re.compile(r"(?i)^enchant\s+\S+")
_RE_ENCHANTED = re.compile(r"(?i)\benchanted\b")
_RE_AURA_SELF = re.compile(r"(?i)\bthis aura\b")
_RE_EQUIP = re.compile(r"(?i)^equip\b")
_RE_EQUIP_VALIDO = re.compile(r"(?i)^equip\s+\{")
_RE_EQUIPPED = re.compile(r"(?i)\bequipped\b")
_RE_EQUIP_SELF = re.compile(r"(?i)\bthis equipment\b")
_RE_MANA_SIMPLES = re.compile(r"(?i)^\{T\}:\s*Add\b")
_RE_MANA_QUALQUER = re.compile(r"(?i)^\{T\}[^:\n]*:\s*Add\b")


def _linhas(texto: str) -> List[str]:
    """Quebra o texto em linhas não vazias; trata o placeholder do modelo
    de texto como texto vazio."""
    texto = str(texto or "").strip()
    if not texto or texto == TEXTO_VAZIO:
        return []
    return [linha.strip() for linha in texto.split("\n") if linha.strip()]


def _juntar(linhas: List[str]) -> str:
    return "\n".join(linhas)


def _formatar_habilidade_mana(cores: List[str]) -> str:
    """Monta "{T}: Add {G}." / "{T}: Add {G} or {U}." / com vírgulas."""
    cores = list(dict.fromkeys(cores))
    if len(cores) == 1:
        return f"{{T}}: Add {{{cores[0]}}}."
    if len(cores) == 2:
        return f"{{T}}: Add {{{cores[0]}}} or {{{cores[1]}}}."
    return (
        "{T}: Add "
        + ", ".join(f"{{{c}}}" for c in cores[:-1])
        + f", or {{{cores[-1]}}}."
    )


def ajustar_subtipos_por_texto(
    tipo: str,
    subtipos: List[str],
    texto: str,
) -> List[str]:
    """Garante coerência entre o texto gerado e os subtipos.

    - Encantamento com linha "Enchant ..." -> subtipo "Aura".
    - Artefato com linha "Equip ..." -> subtipo "Equipment".

    Não mexe em subtipos quando não há evidência no texto.
    """
    subtipos = list(subtipos)
    linhas = _linhas(texto)
    texto_bruto = str(texto or "")

    if tipo == "enchantment":
        # Além da linha "Enchant ...", textos com "Enchanted creature/land"
        # ou "this Aura" também só fazem sentido em Auras.
        tem_enchant = (
            any(_RE_ENCHANT.match(linha) for linha in linhas)
            or bool(_RE_ENCHANTED.search(texto_bruto))
            or bool(_RE_AURA_SELF.search(texto_bruto))
        )
        if tem_enchant and "Aura" not in subtipos:
            if len(subtipos) < MAX_SUBTIPOS["enchantment"]:
                subtipos.append("Aura")
            elif subtipos:
                subtipos[-1] = "Aura"

    if tipo == "artifact":
        # "Equipped creature ..." e "this Equipment" também caracterizam
        # um Equipamento.
        tem_equip = (
            any(_RE_EQUIP.match(linha) for linha in linhas)
            or bool(_RE_EQUIPPED.search(texto_bruto))
            or bool(_RE_EQUIP_SELF.search(texto_bruto))
        )
        if tem_equip and "Equipment" not in subtipos:
            if len(subtipos) < MAX_SUBTIPOS["artifact"]:
                subtipos.append("Equipment")
            elif subtipos:
                subtipos[0] = "Equipment"

    # Remove duplicatas preservando a ordem.
    vistos = set()
    resultado = []
    for subtipo in subtipos:
        if subtipo not in vistos:
            vistos.add(subtipo)
            resultado.append(subtipo)
    return resultado


def garantir_enchant(
    texto: str,
    subtipos: List[str],
    cor: str,
    peso_cor: float,
    modelo: "ModeloSubtipos",
    rng: random.Random,
) -> str:
    """Garante uma única linha "Enchant ..." quando a carta é Aura."""
    if "Aura" not in subtipos:
        return texto

    linhas = _linhas(texto)
    validas = [
        i for i, linha in enumerate(linhas) if _RE_ENCHANT_VALIDO.match(linha)
    ]

    if validas:
        # Mantém a primeira linha válida e remove eventuais duplicatas.
        indice = validas[0]
        linhas = [
            linha
            for i, linha in enumerate(linhas)
            if i == indice or not _RE_ENCHANT.match(linha)
        ]
        return _juntar(linhas)

    # Remove "Enchant" malformado e coloca um alvo válido no topo.
    linhas = [linha for linha in linhas if not _RE_ENCHANT.match(linha)]
    alvo = modelo.sortear_alvo_enchant(cor, peso_cor, rng)
    linhas.insert(0, alvo)
    return _juntar(linhas)


def garantir_equip(
    texto: str,
    subtipos: List[str],
    cor: str,
    modelo: "ModeloSubtipos",
    rng: random.Random,
) -> str:
    """Garante uma única habilidade "Equip {custo}" em Equipamentos."""
    if "Equipment" not in subtipos:
        return texto

    linhas = _linhas(texto)
    validas = [i for i, linha in enumerate(linhas) if _RE_EQUIP_VALIDO.match(linha)]

    if validas:
        indice = validas[0]
        linhas = [
            linha
            for i, linha in enumerate(linhas)
            if i == indice or not _RE_EQUIP.match(linha)
        ]
        return _juntar(linhas)

    linhas = [linha for linha in linhas if not _RE_EQUIP.match(linha)]
    custo = modelo.sortear_custo_equip(cor, rng)
    linhas.append(f"Equip {custo}")
    return _juntar(linhas)


def garantir_habilidade_mana(
    texto: str,
    subtipos: List[str],
    cor: str,
) -> str:
    """Garante habilidade de mana em terrenos.

    - Com tipo básico (Forest, Island...): a linha simples "{T}: Add ..."
      é substituída/ajustada para produzir exatamente as cores
      correspondentes aos subtipos.
    - Sem tipo básico: se não houver nenhuma habilidade de mana, adiciona
      "{T}: Add {cor}." (a cor escolhida pelo usuário).
    """
    linhas = _linhas(texto)

    cores_basicas = list(
        dict.fromkeys(
            TERRENOS_BASICOS[s] for s in subtipos if s in TERRENOS_BASICOS
        )
    )

    if cores_basicas:
        # Um terreno com tipo básico (Forest, Island...) deve produzir
        # exatamente as cores desses tipos. Removemos habilidades de mana
        # simples antigas (que podem ter vindo de outra cor) e deixamos uma
        # linha coerente com os subtipos.
        linhas = [linha for linha in linhas if not _RE_MANA_SIMPLES.match(linha)]
        linhas.insert(0, _formatar_habilidade_mana(cores_basicas))
        return _juntar(linhas)

    # Terreno sem tipo básico: só adiciona se não houver NENHUMA
    # habilidade de mana (não sobrescreve habilidades de utilidade).
    tem_mana = any(_RE_MANA_QUALQUER.match(linha) for linha in linhas)
    if not tem_mana:
        linhas.insert(0, _formatar_habilidade_mana([cor]))
    return _juntar(linhas)


def aplicar_regras_texto(
    texto: str,
    tipo: str,
    subtipos: List[str],
    cor: str,
    peso_cor: float,
    modelo: "ModeloSubtipos",
    rng: random.Random,
) -> str:
    """Aplica o lint de texto específico do tipo (Aura/Equipamento/Terreno)."""
    if tipo == "enchantment":
        return garantir_enchant(texto, subtipos, cor, peso_cor, modelo, rng)
    if tipo == "artifact":
        return garantir_equip(texto, subtipos, cor, modelo, rng)
    if tipo == "land":
        return garantir_habilidade_mana(texto, subtipos, cor)
    return texto
