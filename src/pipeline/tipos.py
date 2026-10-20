"""
Tipos de carta suportados pelo gerador e regras associadas.

A geração agora é organizada em etapas: primeiro o TIPO é definido, depois o
texto de habilidades é gerado com um modelo treinado só naquele tipo (ver
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
]

# Nomes exibidos na interface (português).
TIPOS_DISPLAY = {
    "creature": "Criatura",
    "instant": "Instantânea",
    "sorcery": "Feitiço",
    "artifact": "Artefato",
    "enchantment": "Encantamento",
}

# Tipos que NÃO podem ter habilidades ativadas (custo + ":" + efeito).
# Instants e sorceries não são permanentes, então não têm habilidades
# ativadas "de campo" como "{T}: ...". O corpus tem pouquíssimas
# exceções (habilidades ativadas a partir do cemitério/mão, ~30 linhas);
# elas são filtradas no treino e evitadas na geração.
TIPOS_SEM_HABILIDADE_ATIVADA = {"instant", "sorcery"}

# Só criaturas têm poder/resistência. Artefatos e encantamentos
# não-criatura (e instants/sorceries) não têm P/T.
TIPOS_COM_PODER_RESISTENCIA = {"creature"}


def tipo_da_carta(type_line) -> Optional[str]:
    """Classifica um `type_line` da Scryfall em um dos TIPOS_VALIDOS.

    Prioridade: Creature > Instant > Sorcery > Artifact > Enchantment.
    Assim, "Artifact Creature" e "Enchantment Creature" contam como
    criatura; "Artifact Enchantment" (não-criatura) conta como artefato.
    Devolve None para tipos não suportados (terreno, planeswalker, batalha
    etc.).
    """
    linha = str(type_line or "")
    if "Creature" in linha:
        return "creature"
    if "Instant" in linha:
        return "instant"
    if "Sorcery" in linha:
        return "sorcery"
    if "Artifact" in linha:
        return "artifact"
    if "Enchantment" in linha:
        return "enchantment"
    return None
