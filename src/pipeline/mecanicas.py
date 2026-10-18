"""
Inferência da mecânica de jogo mais provável a partir do nome digitado
pelo usuário.

Abordagem híbrida em camadas (da mais para a menos confiável), conforme
discutido no relatório do projeto:

    1. Dicionário curado: termos que JÁ SÃO jargão de Magic (ex:
       "destroy", "exile"), mapeados manualmente com 100% de precisão.
    2. Similaridade semântica via WordNet (wup_similarity): usada quando
       a palavra não está no dicionário, mas é semanticamente próxima de
       alguma mecânica (ex: "murder", "slay", "eliminate" -> "destroy").
    3. Fallback final (MECANICA_PADRAO): usado quando o nome não tem
       relação semântica clara com nada (ex: nomes aleatórios como
       "jasdhahs12123"), garantindo que o sistema sempre gere algo.

O WordNet só entende inglês comum, por isso as âncoras e o dicionário
são em inglês (mesma língua do oracle_text das cartas) -- o nome digitado
pelo usuário também é comparado em inglês.
"""

from typing import Tuple

import nltk
from nltk.corpus import wordnet as wn

# Camada 1: termos que já são o próprio jargão do jogo. Propositalmente
# curto e preciso -- cobrir sinônimos é trabalho da camada 2 (WordNet),
# não deste dicionário.
DICIONARIO_MECANICAS = {
    "destroy": ["destroy"],
    "exile": ["exile"],
    "bounce": ["return", "bounce"],
    "counter": ["counter"],
    "draw": ["draw"],
    "damage": ["damage"],
    "sacrifice": ["sacrifice"],
    "tap": ["tap"],
    "discard": ["discard"],
    "pump": ["pump", "boost"],
}

# Palavra-âncora em inglês comum usada na camada 2 (similaridade via
# WordNet). Nem toda mecânica tem uma âncora boa (ex: "pump" e "tap" não
# são verbos comuns do inglês do dia a dia) -- essas ficam de fora da
# camada de similaridade e só são alcançadas via dicionário ou fallback.
ANCORAS = {
    "destroy": "destroy",
    "exile": "banish",
    "bounce": "return",
    "counter": "cancel",
    "draw": "draw",
    "damage": "hurt",
    "sacrifice": "sacrifice",
    "discard": "discard",
}

MECANICA_PADRAO = "pump"

# Abaixo deste score de similaridade (escala 0 a 1 do wup_similarity),
# consideramos que não há relação semântica confiável o suficiente e
# caímos no fallback. Ajustável -- vale testar empiricamente no notebook.
LIMIAR_SIMILARIDADE = 0.5


def _garantir_wordnet():
    """Garante que o corpus do WordNet está baixado (idempotente)."""
    for recurso in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(recurso)
        except LookupError:
            nome = recurso.split("/")[-1]
            nltk.download(nome, quiet=True)


def _mapa_palavra_para_mecanica() -> dict:
    """Inverte DICIONARIO_MECANICAS: palavra -> mecânica (lookup O(1))."""
    mapa = {}
    for mecanica, termos in DICIONARIO_MECANICAS.items():
        for termo in termos:
            mapa[termo] = mecanica
    return mapa


_MAPA_PALAVRA_MECANICA = _mapa_palavra_para_mecanica()


def _tokens_do_nome(nome: str) -> list:
    _garantir_wordnet()
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return [t.lower() for t in nltk.word_tokenize(nome) if t.isalpha()]


def _similaridade_wordnet(palavra: str, ancora: str) -> float:
    """Maior wup_similarity entre os synsets verbais de `palavra` e de
    `ancora`. Retorna 0.0 se alguma das duas não existir no WordNet."""
    syns_palavra = wn.synsets(palavra, pos=wn.VERB)
    syns_ancora = wn.synsets(ancora, pos=wn.VERB)

    melhor = 0.0
    for sp in syns_palavra:
        for sa in syns_ancora:
            score = sp.wup_similarity(sa)
            if score and score > melhor:
                melhor = score
    return melhor


def inferir_mecanica(nome: str) -> Tuple[str, str]:
    """Infere a mecânica mais provável a partir do nome da carta.

    Retorna (mecanica, origem), onde origem é 'dicionario', 'similaridade'
    ou 'padrao' -- registra qual camada decidiu, para o sistema conseguir
    explicar sua própria escolha (útil para debug e para a defesa do
    projeto: dá pra mostrar exatamente por que cada carta ficou do jeito
    que ficou).
    """
    tokens = _tokens_do_nome(nome)

    # Camada 1: dicionário curado
    for token in tokens:
        if token in _MAPA_PALAVRA_MECANICA:
            return _MAPA_PALAVRA_MECANICA[token], "dicionario"

    # Camada 2: similaridade semântica via WordNet
    melhor_mecanica, melhor_score = None, 0.0
    for token in tokens:
        for mecanica, ancora in ANCORAS.items():
            score = _similaridade_wordnet(token, ancora)
            if score > melhor_score:
                melhor_score, melhor_mecanica = score, mecanica

    if melhor_mecanica and melhor_score >= LIMIAR_SIMILARIDADE:
        return melhor_mecanica, "similaridade"

    # Camada 3: nada bateu (ex: nome aleatório "jasdhahs12123")
    return MECANICA_PADRAO, "padrao"
