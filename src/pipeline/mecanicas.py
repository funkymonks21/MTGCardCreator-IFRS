"""
Inferência da mecânica de jogo mais provável a partir do nome digitado
pelo usuário.

Abordagem híbrida em camadas (da mais para a menos confiável):

    1. Dicionário curado: termos que JÁ SÃO jargão de Magic (ex:
       "destroy", "exile"), mapeados manualmente com 100% de precisão.
    2. Expansão determinística via WordNet: para cada mecânica, expandimos
       sua palavra-âncora (ex: "kill" para a mecânica "destroy") pegando
       os SINÔNIMOS DIRETOS (mesmo synset) e os HIPÔNIMOS DIRETOS dela
       (ex: "murder", "slay", "assassinate" são hipônimos de "kill" --
       formas mais específicas do mesmo conceito). Isso cobre variações
       como "murder" -> "destroy" sem precisar de um score numérico.
    3. Fallback final (MECANICA_PADRAO): usado quando o nome não bate em
       nada acima (ex: nomes aleatórios como "jasdhahs12123"), garantindo
       que o sistema sempre gere algo.

Por que não usar similaridade numérica (ex: wup_similarity) como camada 2
=========================================================================
Essa foi a primeira versão deste módulo, e os testes no Colab mostraram um
problema real: a hierarquia de VERBOS do WordNet é muito mais rasa e
fragmentada que a de substantivos (não existe uma raiz única como
"entity"), o que torna o wup_similarity pouco confiável para verbos.
Resultado prático observado: "murder" não batia com "destroy" (ficavam em
"árvores" diferentes da hierarquia), enquanto "bolt" (de "Lightning
Bolt") batia com "draw" por pura coincidência de hierarquia, mesmo sem
nenhuma relação de sentido real.

A troca por sinônimos + hipônimos diretos resolve os dois problemas ao
mesmo tempo: é uma relação lexical exata do WordNet (não um score
aproximado), então não existe "quase bateu" nem limiar pra ajustar --
ou a palavra está numa relação direta e explicável com a âncora, ou não
está. "murder" é hipônimo DIRETO de "kill" (é literalmente o exemplo
clássico de livro-texto de troponímia verbal), então essa relação é
capturada com segurança.

O WordNet só entende inglês comum, por isso as âncoras e o dicionário
são em inglês (mesma língua do oracle_text das cartas) -- o nome digitado
pelo usuário também é comparado em inglês.
"""

from typing import Dict, List, Tuple

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

# Palavra(s)-âncora em inglês comum usada(s) na camada 2 (expansão via
# WordNet). Escolhidas pela posição delas na hierarquia de verbos, não só
# pelo significado -- "kill" (não "destroy") é a âncora de "destroy"
# porque é dela que "murder"/"slay"/"assassinate" descendem no WordNet.
# Mecânicas sem uma âncora boa em inglês comum (ex: "tap", "pump") ficam
# de fora desta camada e só são alcançadas via dicionário ou fallback.
ANCORAS: Dict[str, List[str]] = {
    "destroy": ["kill"],
    "exile": ["banish"],
    "bounce": ["return"],
    "counter": ["negate", "cancel"],
    "draw": ["draw"],
    "damage": ["hurt", "injure"],
    "sacrifice": ["sacrifice"],
    "discard": ["discard"],
}

MECANICA_PADRAO = "pump"


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

# Cache do vocabulário expandido via WordNet (camada 2). É construído sob
# demanda, na primeira chamada que precisar dele, porque depende do
# corpus do WordNet já estar baixado -- e fica em cache porque consultar
# o WordNet synset a synset não é instantâneo.
_vocabulario_expandido_cache = None


def _construir_vocabulario_expandido() -> Dict[str, str]:
    """Constrói o mapa palavra -> mecânica da camada 2: para cada âncora,
    coleta os lemmas (sinônimos diretos) de todos os seus synsets verbais
    e os lemmas dos hipônimos diretos desses synsets (um nível abaixo na
    hierarquia -- é aqui que "murder" aparece como hipônimo de "kill")."""
    _garantir_wordnet()
    vocabulario: Dict[str, str] = {}

    for mecanica, ancoras in ANCORAS.items():
        for ancora in ancoras:
            for synset in wn.synsets(ancora, pos=wn.VERB):
                palavras_do_synset = [l.name().replace("_", " ").lower() for l in synset.lemmas()]
                for hiponimo in synset.hyponyms():
                    palavras_do_synset += [
                        l.name().replace("_", " ").lower() for l in hiponimo.lemmas()
                    ]

                for palavra in palavras_do_synset:
                    # Em caso de uma palavra aparecer em mais de uma
                    # mecânica (raro, mas possível), mantém a primeira
                    # associação encontrada em vez de sobrescrever --
                    # evita que a ordem de iteração do dict mude o
                    # resultado de forma não-óbvia.
                    vocabulario.setdefault(palavra, mecanica)

    return vocabulario


def _obter_vocabulario_expandido() -> Dict[str, str]:
    global _vocabulario_expandido_cache
    if _vocabulario_expandido_cache is None:
        _vocabulario_expandido_cache = _construir_vocabulario_expandido()
    return _vocabulario_expandido_cache


def _tokens_do_nome(nome: str) -> list:
    _garantir_wordnet()
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return [t.lower() for t in nltk.word_tokenize(nome) if t.isalpha()]


def inferir_mecanica(nome: str) -> Tuple[str, str]:
    """Infere a mecânica mais provável a partir do nome da carta.

    Retorna (mecanica, origem), onde origem é 'dicionario',
    'wordnet_expandido' ou 'padrao' -- registra qual camada decidiu, para
    o sistema conseguir explicar sua própria escolha (útil para debug e
    para a defesa do projeto: dá pra mostrar exatamente por que cada
    carta ficou do jeito que ficou).
    """
    tokens = _tokens_do_nome(nome)

    # Camada 1: dicionário curado (jargão exato do jogo)
    for token in tokens:
        if token in _MAPA_PALAVRA_MECANICA:
            return _MAPA_PALAVRA_MECANICA[token], "dicionario"

    # Camada 2: sinônimos/hipônimos diretos da âncora, via WordNet
    vocabulario = _obter_vocabulario_expandido()
    for token in tokens:
        if token in vocabulario:
            return vocabulario[token], "wordnet_expandido"

    # Camada 3: nada bateu (ex: nome aleatório "jasdhahs12123")
    return MECANICA_PADRAO, "padrao"
