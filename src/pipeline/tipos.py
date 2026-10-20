"""
Tipos de carta suportados pelo gerador e regras associadas.

A geração agora é organizada em etapas: primeiro o TIPO é definido (pelo
usuário, por inferência a partir do nome ou por sorteio), depois o texto de
habilidades é gerado com um modelo treinado só naquele tipo (ver
`texto_markov.ModeloTextoPorTipo`), e por fim custo de mana e poder/
resistência são previstos (ver `estrutura.ModeloEstrutural`).

Este módulo centraliza a classificação do `type_line` da Scryfall e as
regras que dependem do tipo, para que `texto_markov`, `estrutura` e
`generator` usem exatamente os mesmos conceitos.
"""

from __future__ import annotations

from typing import List, Optional

# Chaves internas (estáveis, usadas nos modelos e nos artefatos salvos).
TIPOS_VALIDOS: List[str] = [
    "creature",
    "instant",
    "sorcery",
    "artifact",
    "enchantment",
    "land",
]

# Nomes exibidos na interface (português).
TIPOS_DISPLAY = {
    "creature": "Criatura",
    "instant": "Instantânea",
    "sorcery": "Feitiço",
    "artifact": "Artefato",
    "enchantment": "Encantamento",
    "land": "Terreno",
}

# Tipos que NÃO podem ter habilidades ativadas (custo + ":" + efeito).
# Instants e sorceries não são permanentes, então não têm habilidades
# ativadas "de campo" como "{T}: ...". O corpus tem pouquíssimas
# exceções (habilidades ativadas a partir do cemitério/mão, ~30 linhas);
# elas são filtradas no treino e evitadas na geração.
TIPOS_SEM_HABILIDADE_ATIVADA = {"instant", "sorcery"}

# Só criaturas têm poder/resistência. Artefatos, encantamentos e terrenos
# não-criatura (e instants/sorceries) não têm P/T.
TIPOS_COM_PODER_RESISTENCIA = {"creature"}

# Terrenos não têm custo de mana -- o campo fica vazio no Oracle.
TIPOS_SEM_CUSTO_MANA = {"land"}

# Terrenos são incolores (colors == []), mas têm uma identidade de cor
# (color_identity) que diz quais cores de mana eles produzem. Para o
# modelo de texto de terreno usamos color_identity como sinal de cor;
# assim um terreno verde aprende o vocabulário de "Forest", um azul o de
# "Island" etc. (ver texto_markov.ModeloMarkovPorCor).
TIPOS_USAM_COLOR_IDENTITY = {"land"}


def tipo_da_carta(type_line) -> Optional[str]:
    """Classifica um `type_line` da Scryfall em um dos TIPOS_VALIDOS.

    Prioridade: Creature > Instant > Sorcery > Land > Artifact >
    Enchantment. Assim, "Artifact Creature", "Enchantment Creature" e
    "Land Creature" contam como criatura; "Artifact Land" e
    "Enchantment Land" contam como terreno; "Artifact Enchantment"
    (não-criatura) conta como artefato. Devolve None para tipos não
    suportados (planeswalker, batalha etc.).
    """
    linha = str(type_line or "")
    if "Creature" in linha:
        return "creature"
    if "Instant" in linha:
        return "instant"
    if "Sorcery" in linha:
        return "sorcery"
    if "Land" in linha:
        return "land"
    if "Artifact" in linha:
        return "artifact"
    if "Enchantment" in linha:
        return "enchantment"
    return None
