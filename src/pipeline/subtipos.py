"""
Modelo estatístico de subtipos de carta (types/subtypes do `type_line`).

O que este módulo resolve
=========================
Até agora o pipeline gerava só o tipo base (Criatura, Artefato, Terreno...).
Cartas reais também têm SUBTIPOS: "Creature — Human Warrior",
"Artifact — Equipment", "Land — Forest Island", "Enchantment — Aura" etc.

As quantidades e quais subtipos aparecem NÃO são regras fixas: são
probabilidades estimadas diretamente do corpus da Scryfall
(data/processed/cards.parquet), condicionadas pelo tipo e pela cor e
misturadas com um modelo global do mesmo jeito que o modelo de texto
(ver `texto_markov._misturar`). O `peso_cor` controla essa mistura:
perto de 1 usa mais a distribuição da cor escolhida; perto de 0 usa mais a
distribuição global.

Regras de quantidade (pedido do projeto)
========================================
- Criatura: 1 a 3 subtipos de criatura (nunca 0).
- Artefato: 0 ou 1 subtipo de artefato.
- Terreno: 0 a 2 subtipos de terreno.
- Encantamento: 0 a 2 subtipos (o corpus não passa de 2).
- Instantânea/Feitiço: 0 ou 1 subtipo de mágica.

Os limites são aplicados durante o treino: cartas fora do limite são
"encostadas" no limite mais próximo (ex.: uma criatura com 4 subtipos conta
como 3; um artefato com 2 conta como 1). Assim as probabilidades continuam
vindo dos dados, mas respeitam as regras do projeto.

Separação entre tipos de criatura e de artefato
===============================================
Cartas que são "Artifact Creature" têm subtipos de criatura (Golem, Wall),
não subtipos de artefato. Como o gerador produz um tipo-base por vez,
montamos os pools assim:
1. extraímos os subtipos das cartas "puras" de cada tipo-base;
2. retiramos do pool de artefato/encantamento/instantânea/feitiço qualquer
   subtipo que também seja um tipo de criatura (isso limpa contaminações de
   cartas "Kindred Artifact — Wizard Equipment", por exemplo);
3. para terrenos mantemos o pool puro (Locus, por exemplo, é um subtipo de
   terreno legítimo mesmo aparecendo em alguma criatura).

Subtipos com espaço
===================
O `type_line` da Scryfall separa subtipos por espaço, mas alguns subtipos
têm espaço no nome ("Time Lord", "Infinity Stone"). Eles são protegidos
antes da tokenização para não virarem dois subtipos errados.

Artefato treinado
=================
`models/modelos_subtipos.pkl` (pickle), gerado por
`src/pipeline/treinar_modelos_subtipos.py`. Ele também guarda as
distribuições usadas pelas regras de texto:
- alvos de "Enchant ..." (para Auras);
- custos de "Equip ..." (para Equipamentos).
"""

from __future__ import annotations

import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .tipos import (
    TIPOS_USAM_COLOR_IDENTITY,
    TIPOS_VALIDOS,
    tipo_da_carta,
)

CORES_VALIDAS = ["W", "U", "B", "R", "G"]

# Nome do tipo base como aparece no type_line real (Oracle, em inglês).
TIPOS_BASE_EN = {
    "creature": "Creature",
    "instant": "Instant",
    "sorcery": "Sorcery",
    "artifact": "Artifact",
    "enchantment": "Enchantment",
    "land": "Land",
}

# Limites pedidos para a quantidade de subtipos por tipo.
MIN_SUBTIPOS = {
    "creature": 1,
    "artifact": 0,
    "land": 0,
    "enchantment": 0,
    "instant": 0,
    "sorcery": 0,
}
MAX_SUBTIPOS = {
    "creature": 3,
    "artifact": 1,
    "land": 2,
    "enchantment": 2,
    "instant": 1,
    "sorcery": 1,
}

# Subtipos que, quando sorteados, "fecham" o type_line: eles nunca aparecem
# junto de outro subtipo em cartas reais (ex.: uma Saga não é uma Aura --
# "Enchantment — Saga Aura" não existe). Se o primeiro subtipo sorteado for
# um destes, a quantidade cai para 1.
SUBTIPOS_SOLITARIOS = {"Saga"}

# Subtipos oficiais com espaço. Protegidos antes de dividir o type_line.
SUBTIPOS_MULTIPALAVRA = ("Time Lord", "Infinity Stone")

# Tipos básicos de terreno -> cor de mana que eles produzem.
TERRENOS_BASICOS = {
    "Plains": "W",
    "Island": "U",
    "Swamp": "B",
    "Mountain": "R",
    "Forest": "G",
}
_BASICO_POR_COR = {cor: nome for nome, cor in TERRENOS_BASICOS.items()}

TIPOS_BASE = ("Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land")

_PLACEHOLDER_ESPACO = "\u0001"
_PADRAO_EQUIP = re.compile(r"(?im)\bEquip\s+((?:\{[^}]+\}\s*)+)")
_PADRAO_SIMBOLO = re.compile(r"\{([^}]+)\}")
_PADRAO_LETRA_COR = re.compile(r"[WUBRG]")


def _cor_em(cores, cor: str) -> bool:
    """`cor in cores`, tolerando list/ndarray/None (mesma lógica do modelo
    de texto: a coluna do Parquet volta como numpy.ndarray)."""
    try:
        return cor in cores
    except TypeError:
        return False


def _misturar(dist_cor: Counter, dist_global: Counter, peso_cor: float) -> Counter:
    """Mistura distribuições cor/global com peso `peso_cor`, caindo para o
    lado que tiver dados quando um dos dois está vazio. Cópia local da
    mesma ideia de `texto_markov._misturar` para manter este módulo
    independente do modelo de texto."""
    tem_cor, tem_global = bool(dist_cor), bool(dist_global)
    if tem_cor and tem_global:
        peso_cor_ef, peso_global_ef = peso_cor, 1 - peso_cor
    elif tem_cor:
        peso_cor_ef, peso_global_ef = 1.0, 0.0
    elif tem_global:
        peso_cor_ef, peso_global_ef = 0.0, 1.0
    else:
        return Counter()

    total_cor = sum(dist_cor.values()) or 1
    total_global = sum(dist_global.values()) or 1

    resultado: Counter = Counter()
    for item, freq in dist_cor.items():
        resultado[item] += peso_cor_ef * (freq / total_cor)
    for item, freq in dist_global.items():
        resultado[item] += peso_global_ef * (freq / total_global)
    return resultado


def _separar_type_line(type_line) -> Tuple[List[str], List[str]]:
    """Devolve (tipos-base, subtipos) de um `type_line`.

    Ex.: "Legendary Artifact Creature — Time Lord Rogue" ->
         (["Creature", "Artifact"], ["Time Lord", "Rogue"]).
    """
    linha = str(type_line or "")
    if "\u2014" in linha:
        base, sufixo = linha.split("\u2014", 1)
    else:
        base, sufixo = linha, ""

    tipos_base = [t for t in TIPOS_BASE if t in base]

    # Protege subtipos com espaço antes de dividir.
    for frase in SUBTIPOS_MULTIPALAVRA:
        sufixo = sufixo.replace(frase, frase.replace(" ", _PLACEHOLDER_ESPACO))
    subtipos = [
        token.replace(_PLACEHOLDER_ESPACO, " ")
        for token in sufixo.split()
        if token.strip()
    ]
    return tipos_base, subtipos


def _limitar_quantidade(tipo: str, n: int) -> int:
    """Aplica os limites do projeto (1-3 criatura, 0-1 artefato etc.)."""
    return max(MIN_SUBTIPOS[tipo], min(MAX_SUBTIPOS[tipo], n))


def _extrair_alvos_enchant(oracle_text: str) -> List[str]:
    """Extrai linhas válidas de "Enchant ..." de uma Aura real."""
    alvos = []
    for linha in str(oracle_text or "").split("\n"):
        linha = re.sub(r"\([^)]*\)", "", linha).strip()
        if re.match(r"(?i)^Enchant\b", linha):
            linha = linha.rstrip(". ").strip()
            if linha:
                alvos.append(linha)
    return alvos


def _extrair_custos_equip(oracle_text: str) -> List[str]:
    """Extrai custos de linhas "Equip {2}", inclusive quando a linha tem um
    prefixo de nome ("Lightbringer — Equip {4}"). Normaliza espaços."""
    texto = str(oracle_text or "")
    custos = []
    for bruto in _PADRAO_EQUIP.findall(texto):
        custo = "".join(bruto.split())
        if custo:
            custos.append(custo)
    return custos


# Linha "Equipped creature ..." (a habilidade que um Equipamento concede à
# criatura equipada). É quase universal em Equipamentos reais (99% do
# corpus) e por isso é garantida na geração (ver regras_tipo).
_PADRAO_EQUIPPED_CREATURE = re.compile(r"(?i)^equipped creature\b")


def _extrair_equipped_creature(oracle_text: str) -> List[str]:
    """Extrai linhas "Equipped creature ..." de um Equipamento real."""
    linhas = []
    for linha in str(oracle_text or "").split("\n"):
        linha = re.sub(r"\([^)]*\)", "", linha).strip()
        if _PADRAO_EQUIPPED_CREATURE.match(linha):
            linha = linha.strip()
            if linha:
                linhas.append(linha)
    return linhas


# Linha "Crew N" (habilitar um Veículo: vira criatura até o fim do turno).
# Quase todo Veículo real (96% do corpus) tem uma, com N entre 1 e 8.
_PADRAO_CREW = re.compile(r"(?im)\bCrew\s+(\d+)")


def _extrair_custos_crew(oracle_text: str) -> List[str]:
    """Extrai os valores de "Crew N" de um Veículo real (devolve ["2"], ...)."""
    return _PADRAO_CREW.findall(str(oracle_text or ""))


def _cor_para_veiculo(colors) -> str:
    """Reduz a lista de cores de um Veículo a uma chave (para condicionar o
    P/T por cor): "" (incolor) ou a cor única; multicolor vira a primeira
    cor em ordem WUBRG (Veículos são quase sempre incolores ou de uma cor)."""
    try:
        presentes = [c for c in CORES_VALIDAS if c in colors]
    except TypeError:
        return ""
    return presentes[0] if presentes else ""


# Linha de capítulo de Saga: começa com numeral romano ("I — ...",
# "II — ...") ou uma lista deles ("I, II, III — ..."). Ver regras_tipo para
# como as Sagas geradas recebem capítulos numerados.
_PADRAO_CAPITULO_SAGA = re.compile(
    r"(?i)^(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3})(?:\s*,\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}))*\s*(?:[—\-–]\s*|\s+)"
)


def _contar_capitulos_saga(oracle_text: str) -> int:
    """Conta quantas linhas de capítulo (numeral romano inicial) uma Saga
    real tem. Reminder text entre parênteses é ignorado."""
    n = 0
    for linha in str(oracle_text or "").split("\n"):
        linha = re.sub(r"\([^)]*\)", "", linha).strip()
        if _PADRAO_CAPITULO_SAGA.match(linha):
            n += 1
    return n


def _custo_equip_compativel(custo: str, cor: str) -> bool:
    """Aceita custos genéricos e custos coloridos só com a cor da carta
    (evita um Equipamento vermelho com "Equip {2}{G}", por exemplo)."""
    for simbolo in _PADRAO_SIMBOLO.findall(custo):
        letras = set(_PADRAO_LETRA_COR.findall(simbolo))
        if letras and not letras.issubset({cor}):
            return False
    return True


class ModeloSubtipos:
    """Probabilidades de quantidade/quais subtipos + estatísticas das
    regras de Aura/Equipamento/Terreno."""

    def __init__(self, df_cartas: pd.DataFrame):
        # Frequência de subtipos: global e por cor.
        # frequencias_por_cor[tipo][cor][subtipo] = contagem
        self.frequencias_global: Dict[str, Counter] = {
            tipo: Counter() for tipo in TIPOS_VALIDOS
        }
        self.frequencias_por_cor: Dict[str, Dict[str, Counter]] = {
            tipo: {cor: Counter() for cor in CORES_VALIDAS}
            for tipo in TIPOS_VALIDOS
        }

        # Quantidade de subtipos: global e por cor.
        self.contagem_global: Dict[str, Counter] = {
            tipo: Counter() for tipo in TIPOS_VALIDOS
        }
        self.contagem_por_cor: Dict[str, Dict[str, Counter]] = {
            tipo: {cor: Counter() for cor in CORES_VALIDAS}
            for tipo in TIPOS_VALIDOS
        }

        # Co-ocorrência: depois de escolher o subtipo X, quais subtipos Y
        # aparecem junto? Usado para montar combos plausíveis (ex.:
        # "Human Warrior" em vez de sortear cada um isoladamente).
        self.coocorrencias_global: Dict[str, Dict[str, Counter]] = {
            tipo: defaultdict(Counter) for tipo in TIPOS_VALIDOS
        }
        self.coocorrencias_por_cor: Dict[str, Dict[str, Dict[str, Counter]]] = {
            tipo: {cor: defaultdict(Counter) for cor in CORES_VALIDAS}
            for tipo in TIPOS_VALIDOS
        }

        # Ordem média do subtipo dentro do type_line (para exibir
        # "Human Warrior" em vez de "Warrior Human").
        self._soma_posicao: Dict[str, Counter] = {
            tipo: Counter() for tipo in TIPOS_VALIDOS
        }
        self._contagem_posicao: Dict[str, Counter] = {
            tipo: Counter() for tipo in TIPOS_VALIDOS
        }
        self.ordem_media: Dict[str, Dict[str, float]] = {}

        # Regras de texto.
        self.alvos_enchant_global: Counter = Counter()
        self.alvos_enchant_por_cor: Dict[str, Counter] = {
            cor: Counter() for cor in CORES_VALIDAS
        }
        self.custos_equip_global: Counter = Counter()
        self.custos_equip_por_cor: Dict[str, Counter] = {
            cor: Counter() for cor in CORES_VALIDAS
        }
        # Linhas "Equipped creature ..." reais, para garantir essa habilidade
        # nos Equipamentos gerados.
        self.equipped_creature_global: Counter = Counter()
        self.equipped_creature_por_cor: Dict[str, Counter] = {
            cor: Counter() for cor in CORES_VALIDAS
        }
        # Custo de "Crew N" de Veículos reais (a habilidade que torna o
        # Veículo uma criatura). Garantida nos Veículos gerados.
        self.crew_global: Counter = Counter()
        self.crew_por_cor: Dict[str, Counter] = {
            cor: Counter() for cor in CORES_VALIDAS
        }
        # Poder/resistência de Veículos reais (eles NÃO passam pelo modelo de
        # regressão de P/T, que é treinado só em criaturas). Chave: "" para
        # incolor, ou a cor única. Valor: Counter de (poder, resistencia).
        self.pt_veiculo_por_cor: Dict[str, Counter] = {}
        self.pt_veiculo_global: Counter = Counter()
        # Quantos capítulos uma Saga real tem (1, 2, 3...), para sortear a
        # quantidade de parágrafos numerados de uma Saga gerada.
        self.capitulos_saga_global: Counter = Counter()
        self.capitulos_saga_por_cor: Dict[str, Counter] = {
            cor: Counter() for cor in CORES_VALIDAS
        }

        # Pools finais (sem contaminação entre criatura/artefato etc.).
        self.pools: Dict[str, List[str]] = {}

        self._treinar(df_cartas)

    # -- treino -----------------------------------------------------

    def _treinar(self, df_cartas: pd.DataFrame) -> None:
        # 1) Parse de todas as cartas uma vez.
        registros = []
        for _, carta in df_cartas.iterrows():
            tipo = tipo_da_carta(carta.get("type_line", ""))
            if tipo is None:
                continue
            bases, subtipos = _separar_type_line(carta.get("type_line", ""))
            registros.append(
                {
                    "tipo": tipo,
                    "bases": bases,
                    "subtipos": subtipos,
                    "colors": carta.get("colors"),
                    "color_identity": carta.get("color_identity"),
                    "oracle_text": carta.get("oracle_text", ""),
                    "power": carta.get("power"),
                    "toughness": carta.get("toughness"),
                }
            )

        # 2) Pools iniciais: só cartas "puras" (um tipo-base) para não
        # misturar, por exemplo, subtipos de criatura de um Artifact
        # Creature no pool de artefato.
        pools_iniciais: Dict[str, set] = {tipo: set() for tipo in TIPOS_VALIDOS}
        for reg in registros:
            if len(reg["bases"]) != 1:
                continue
            tipo = reg["tipo"]
            pools_iniciais[tipo].update(reg["subtipos"])

        # 3) Remove tipos de criatura dos pools que não são criatura.
        tipos_criatura = pools_iniciais["creature"]
        self.pools["creature"] = sorted(tipos_criatura)
        for tipo in ("artifact", "enchantment", "instant", "sorcery"):
            self.pools[tipo] = sorted(pools_iniciais[tipo] - tipos_criatura)
        # Terreno mantém o pool puro (Locus etc. são tipos de terreno).
        self.pools["land"] = sorted(pools_iniciais["land"])

        # 4) Acumula contagens já filtradas pelo pool final.
        pools_por_tipo = {tipo: set(pool) for tipo, pool in self.pools.items()}
        for reg in registros:
            tipo = reg["tipo"]
            if len(reg["bases"]) != 1:
                # Cartas com múltiplos tipos-base não entram nas
                # distribuições de quantidade (evita, por exemplo, contar
                # um Artifact Creature como artefato e como criatura).
                continue

            pool = pools_por_tipo[tipo]
            subtipos = []
            vistos = set()
            for subtipo in reg["subtipos"]:
                if subtipo in pool and subtipo not in vistos:
                    vistos.add(subtipo)
                    subtipos.append(subtipo)

            n = _limitar_quantidade(tipo, len(subtipos))

            coluna_cor = (
                "color_identity" if tipo in TIPOS_USAM_COLOR_IDENTITY else "colors"
            )
            cores_carta = [
                cor
                for cor in CORES_VALIDAS
                if _cor_em(reg[coluna_cor], cor)
            ]

            self.contagem_global[tipo][n] += 1
            for s in subtipos:
                self.frequencias_global[tipo][s] += 1
            for i, s in enumerate(subtipos):
                self._soma_posicao[tipo][s] += i
                self._contagem_posicao[tipo][s] += 1
            for a in subtipos:
                for b in subtipos:
                    if a != b:
                        self.coocorrencias_global[tipo][a][b] += 1

            for cor in cores_carta:
                self.contagem_por_cor[tipo][cor][n] += 1
                for s in subtipos:
                    self.frequencias_por_cor[tipo][cor][s] += 1
                for a in subtipos:
                    for b in subtipos:
                        if a != b:
                            self.coocorrencias_por_cor[tipo][cor][a][b] += 1

            # Regras de texto (Aura/Equipamento) usam todas as cartas
            # puras do tipo, com cor quando houver.
            if tipo == "enchantment" and "Aura" in subtipos:
                for alvo in _extrair_alvos_enchant(reg["oracle_text"]):
                    self.alvos_enchant_global[alvo] += 1
                    for cor in cores_carta:
                        self.alvos_enchant_por_cor[cor][alvo] += 1

            if tipo == "artifact" and "Equipment" in subtipos:
                for custo in _extrair_custos_equip(reg["oracle_text"]):
                    self.custos_equip_global[custo] += 1
                    for cor in cores_carta:
                        self.custos_equip_por_cor[cor][custo] += 1
                for linha in _extrair_equipped_creature(reg["oracle_text"]):
                    self.equipped_creature_global[linha] += 1
                    for cor in cores_carta:
                        self.equipped_creature_por_cor[cor][linha] += 1

            if tipo == "enchantment" and "Saga" in subtipos:
                n_caps = _contar_capitulos_saga(reg["oracle_text"])
                if n_caps > 0:
                    self.capitulos_saga_global[n_caps] += 1
                    for cor in cores_carta:
                        self.capitulos_saga_por_cor[cor][n_caps] += 1

            if tipo == "artifact" and "Vehicle" in subtipos:
                # Custo de Crew (habilidade obrigatória do Veículo gerado).
                for valor in _extrair_custos_crew(reg["oracle_text"]):
                    self.crew_global[valor] += 1
                    for cor in cores_carta:
                        self.crew_por_cor[cor][valor] += 1

                # P/T de Veículo (não passa pela regressão, treinada só em
                # criaturas -- ver pipeline.estrutura).
                poder, resistencia = reg["power"], reg["toughness"]
                if (
                    str(poder).isdigit()
                    and str(resistencia).isdigit()
                ):
                    chave = _cor_para_veiculo(reg["colors"])
                    pt = (int(poder), int(resistencia))
                    self.pt_veiculo_global[pt] += 1
                    self.pt_veiculo_por_cor.setdefault(chave, Counter())[pt] += 1

        # 5) Ordem média.
        for tipo in TIPOS_VALIDOS:
            self.ordem_media[tipo] = {
                s: self._soma_posicao[tipo][s] / self._contagem_posicao[tipo][s]
                for s in self._soma_posicao[tipo]
                if self._contagem_posicao[tipo][s]
            }

    # -- geração ----------------------------------------------------

    def _distribuicao_contagem(
        self, tipo: str, cor: str, peso_cor: float
    ) -> Counter:
        return _misturar(
            self.contagem_por_cor[tipo].get(cor, Counter()),
            self.contagem_global[tipo],
            peso_cor,
        )

    def _distribuicao_subtipos(
        self, tipo: str, cor: str, peso_cor: float
    ) -> Counter:
        return _misturar(
            self.frequencias_por_cor[tipo].get(cor, Counter()),
            self.frequencias_global[tipo],
            peso_cor,
        )

    def _distribuicao_coocorrencia(
        self, tipo: str, cor: str, ultimo: str, peso_cor: float
    ) -> Counter:
        dist_cor = self.coocorrencias_por_cor[tipo].get(cor, {}).get(
            ultimo, Counter()
        )
        dist_global = self.coocorrencias_global[tipo].get(ultimo, Counter())
        return _misturar(dist_cor, dist_global, peso_cor)

    def sortear(
        self,
        tipo: str,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Sorteia os subtipos de uma carta do `tipo` e da `cor`.

        A quantidade vem das probabilidades observadas no corpus (já
        limitadas pelas regras do projeto); os subtipos vêm das frequências
        daquela cor misturadas com o global. Para quantidades > 1, o
        próximo subtipo é condicionado ao anterior pelas co-ocorrências
        reais (ex.: Human -> Warrior), com fallback para a frequência
        global se não houver dado.
        """
        if tipo not in self.pools:
            raise ValueError(f"Tipo inválido: {tipo!r}. Use um de {TIPOS_VALIDOS}.")
        if cor not in CORES_VALIDAS:
            raise ValueError(f"Cor inválida: {cor!r}. Use uma de {CORES_VALIDAS}.")

        rng = rng or random.Random()
        dist_contagem = self._distribuicao_contagem(tipo, cor, peso_cor)
        if not dist_contagem:
            n = MIN_SUBTIPOS[tipo]
        else:
            n = rng.choices(
                list(dist_contagem.keys()),
                weights=list(dist_contagem.values()),
                k=1,
            )[0]
        n = _limitar_quantidade(tipo, int(n))
        if n <= 0:
            return []

        dist_sub = self._distribuicao_subtipos(tipo, cor, peso_cor)
        pool = self.pools[tipo]
        escolhidos: List[str] = []

        for indice in range(n):
            if indice == 0:
                dist = Counter(
                    {s: w for s, w in dist_sub.items() if s in pool}
                )
            else:
                # A partir do 2º subtipo, nunca sorteamos um "solitarário"
                # (ex.: Saga): ele não combina com outro subtipo.
                ultimo = escolhidos[-1]
                dist = Counter(
                    {
                        s: w
                        for s, w in self._distribuicao_coocorrencia(
                            tipo, cor, ultimo, peso_cor
                        ).items()
                        if s in pool and s not in escolhidos
                        and s not in SUBTIPOS_SOLITARIOS
                    }
                )
                if not dist:
                    dist = Counter(
                        {
                            s: w
                            for s, w in dist_sub.items()
                            if s not in escolhidos and s not in SUBTIPOS_SOLITARIOS
                        }
                    )

            if not dist:
                # Último fallback: qualquer subtipo do pool ainda não
                # escolhido (menos os solitários, que não se combinam),
                # com peso igual.
                restantes = [
                    s
                    for s in pool
                    if s not in escolhidos and s not in SUBTIPOS_SOLITARIOS
                ]
                if not restantes:
                    break
                dist = Counter({s: 1.0 for s in restantes})

            escolhido = rng.choices(
                list(dist.keys()), weights=list(dist.values()), k=1
            )[0]
            escolhidos.append(escolhido)

            # Subtipo "solitarário" (ex.: Saga) fecha o type_line: não
            # combina com mais nada, então paramos de sortear.
            if escolhido in SUBTIPOS_SOLITARIOS:
                break

        # Coerência de terrenos: se há tipo(s) básico(s), pelo menos um deve
        # produzir a cor escolhida pelo usuário. Ex.: um terreno verde não
        # vira "Mountain Plains" (R/W); vira "Forest Plains" (G/W).
        if tipo == "land" and escolhidos:
            basicos = [s for s in escolhidos if s in TERRENOS_BASICOS]
            if basicos and not any(TERRENOS_BASICOS[s] == cor for s in basicos):
                escolhidos[escolhidos.index(basicos[0])] = _BASICO_POR_COR[cor]

        # Ordena pela posição média observada no corpus.
        ordem = self.ordem_media.get(tipo, {})
        escolhidos.sort(key=lambda s: ordem.get(s, 999.0))
        return escolhidos

    def sortear_alvo_enchant(
        self,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Sorteia uma linha "Enchant ..." para uma Aura."""
        rng = rng or random.Random()
        dist = _misturar(
            self.alvos_enchant_por_cor.get(cor, Counter()),
            self.alvos_enchant_global,
            peso_cor,
        )
        if not dist:
            return "Enchant creature"
        return rng.choices(
            list(dist.keys()), weights=list(dist.values()), k=1
        )[0]

    def sortear_custo_equip(
        self,
        cor: str,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Sorteia um custo de Equip compatível com a cor da carta."""
        rng = rng or random.Random()
        compativeis = Counter(
            {
                custo: freq
                for custo, freq in self.custos_equip_global.items()
                if _custo_equip_compativel(custo, cor)
            }
        )
        if not compativeis:
            return "{2}"
        return rng.choices(
            list(compativeis.keys()),
            weights=list(compativeis.values()),
            k=1,
        )[0]

    def sortear_equipped_creature(
        self,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Sorteia uma linha real "Equipped creature ..." para um
        Equipamento, misturando a cor da carta com o global. Cai num texto
        genérico se o corpus não tiver nenhuma amostra."""
        rng = rng or random.Random()
        dist = _misturar(
            self.equipped_creature_por_cor.get(cor, Counter()),
            self.equipped_creature_global,
            peso_cor,
        )
        if not dist:
            return "Equipped creature gets +1/+1."
        return rng.choices(
            list(dist.keys()), weights=list(dist.values()), k=1
        )[0]

    def sortear_crew(
        self,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Sorteia o valor de "Crew N" para um Veículo, misturando a cor da
        carta com o global. Usa a distribuição real (2 e 3 são os mais
        comuns). Cai em "2" se o corpus não tiver nenhuma amostra."""
        rng = rng or random.Random()
        dist = _misturar(
            self.crew_por_cor.get(cor, Counter()),
            self.crew_global,
            peso_cor,
        )
        if not dist:
            return "2"
        return rng.choices(
            list(dist.keys()), weights=list(dist.values()), k=1
        )[0]

    def sortear_pt_veiculo(
        self,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> Optional[tuple]:
        """Sorteia (poder, resistência) de um Veículo a partir dos Veículos
        reais. Veículos não passam pela regressão de P/T (treinada só em
        criaturas), então o P/T vem direto do corpus. Mistura a cor da carta
        com o global; devolve None se não houver dados."""
        rng = rng or random.Random()
        dist = _misturar(
            self.pt_veiculo_por_cor.get(cor, Counter()),
            self.pt_veiculo_global,
            peso_cor,
        )
        if not dist:
            return None
        return rng.choices(
            list(dist.keys()), weights=list(dist.values()), k=1
        )[0]

    def sortear_capitulos_saga(
        self,
        cor: str,
        peso_cor: float = 0.7,
        rng: Optional[random.Random] = None,
        minimo: int = 1,
        maximo: int = 5,
    ) -> int:
        """Sorteia quantos capítulos uma Saga terá, usando a distribuição
        real (2 e 3 são os mais comuns). Preso a [minimo, maximo]."""
        rng = rng or random.Random()
        dist = _misturar(
            self.capitulos_saga_por_cor.get(cor, Counter()),
            self.capitulos_saga_global,
            peso_cor,
        )
        validos = {n: f for n, f in dist.items() if minimo <= n <= maximo and f > 0}
        if not validos:
            return minimo
        return rng.choices(
            list(validos.keys()), weights=list(validos.values()), k=1
        )[0]

    @staticmethod
    def montar_tipo_linha(tipo: str, subtipos: List[str]) -> str:
        """Monta o type_line no formato Oracle: "Creature — Human Warrior"."""
        base = TIPOS_BASE_EN[tipo]
        if not subtipos:
            return base
        return f"{base} \u2014 {' '.join(subtipos)}"

    # -- persistência ----------------------------------------------

    def salvar(self, caminho: str = "models/modelos_subtipos.pkl") -> Path:
        destino = Path(caminho)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "wb") as f:
            pickle.dump(self, f)
        return destino

    @classmethod
    def carregar(
        cls, caminho: str = "models/modelos_subtipos.pkl"
    ) -> "ModeloSubtipos":
        origem = Path(caminho)
        if not origem.exists():
            raise FileNotFoundError(
                f"{origem} não encontrado. Treine o modelo antes com:\n"
                f"  python src/pipeline/treinar_modelos_subtipos.py"
            )
        with open(origem, "rb") as f:
            return pickle.load(f)
