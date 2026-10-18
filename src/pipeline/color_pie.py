"""
Regras de color pie: o que cada cor do Magic pode e não pode fazer.

É uma versão simplificada do color pie oficial (o jogo real tem várias
exceções pontuais), mas suficiente para o objetivo do projeto: impedir
combinações claramente fora do espírito das cores -- o exemplo que guiou
o design foi "o usuário não deveria conseguir criar uma carta azul que
destrói um artefato; a carta deveria dar bounce".
"""

from typing import Dict, Set

# Para cada mecânica, o conjunto de cores que legitimamente a usam no
# color pie. Mecânicas ausentes daqui (não deveria acontecer, mas por
# segurança) são tratadas como permitidas em nenhuma cor.
MECANICAS_POR_COR: Dict[str, Set[str]] = {
    "destroy": {"Preto", "Branco"},
    "exile": {"Branco", "Preto"},
    "bounce": {"Azul"},
    "counter": {"Azul"},
    "draw": {"Azul"},
    "damage": {"Vermelho", "Preto"},
    "sacrifice": {"Preto"},
    "tap": {"Azul", "Branco"},
    "discard": {"Preto"},
    "pump": {"Verde", "Branco", "Vermelho"},
}

# Mecânicas de "remoção de permanente" são tratadas como uma categoria à
# parte: se o usuário pede uma delas numa cor que não a possui, a carta
# NÃO vira um pump genérico -- ela recebe o "sabor" de remoção típico
# daquela cor, preservando a intenção original do usuário (remover algo).
MECANICAS_DE_REMOCAO = {"destroy", "exile", "bounce", "damage"}

SABOR_DE_REMOCAO_POR_COR = {
    "Branco": "exile",
    "Azul": "bounce",
    "Preto": "destroy",
    "Vermelho": "damage",
    "Verde": "damage",  # aproximação do "fight" verde
}


def ajustar_mecanica_para_cor(mecanica: str, cor: str) -> str:
    """Garante que a mecânica final respeita o color pie da cor pedida.

    - Se `mecanica` já é legal para `cor`, devolve ela mesma, sem mudança.
    - Se for uma mecânica de remoção ilegal para a cor (ex: destroy +
      Azul), troca pelo "sabor" de remoção típico daquela cor (bounce).
    - Caso contrário (mecânica não-remoção fora do color pie da cor),
      cai para "pump", que serve como mecânica genérica de último recurso.
    """
    cores_permitidas = MECANICAS_POR_COR.get(mecanica, set())

    if cor in cores_permitidas:
        return mecanica

    if mecanica in MECANICAS_DE_REMOCAO:
        return SABOR_DE_REMOCAO_POR_COR[cor]

    return "pump"


# Templates simples de texto de regras por mecânica. Ainda bem diretos de
# propósito -- a próxima etapa do projeto vai enriquecer isso (variações
# de frase, números escalados pelo custo de mana etc.); por enquanto o
# foco é a mecânica e a cor estarem sempre coerentes entre si.
TEMPLATES = {
    "destroy": "Destrua a criatura ou planeswalker alvo.",
    "exile": "Exile a criatura ou planeswalker alvo.",
    "bounce": "Devolva a criatura ou planeswalker alvo para a mão do seu dono.",
    "counter": "Anule a mágica alvo.",
    "draw": "Compre uma carta.",
    "damage": "Esta carta causa 3 pontos de dano à criatura, planeswalker ou jogador alvo.",
    "sacrifice": "Sacrifique uma criatura.",
    "tap": "Vire a criatura alvo.",
    "discard": "O jogador alvo descarta uma carta.",
    "pump": "A criatura alvo recebe +2/+2 até o final do turno.",
}
