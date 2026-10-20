"""
Geração de texto de regras via modelo estatístico de linguagem (cadeia de
Markov sobre bigramas/trigramas), treinado separadamente por TIPO e por cor.

Substitui a abordagem anterior (mecânica fixa -> template fixo, com regras
de color pie bloqueando combinações) por algo propositalmente mais
"caótico": o texto nasce de amostragem estatística sobre sequências de
palavras extraídas do oracle_text real das cartas da Scryfall. Não há
filtro de color pie aqui -- é uma decisão de design deliberada (caos >
garantia), documentada no relatório do projeto.

Ideia geral
===========
Para cada cor (W/U/B/R/G), treinamos um modelo que sabe, estatisticamente,
"depois da palavra X, qual palavra tende a vir?" (bigramas) e "depois das
palavras X e Y, qual palavra tende a vir?" (trigramas). Um segundo modelo,
treinado com TODAS as cores juntas, serve de pano de fundo ("global").

Na hora de gerar, a distribuição de probabilidade da próxima palavra é uma
MISTURA ponderada entre o modelo da cor escolhida e o modelo global:

    prob_final(palavra) = peso_cor * prob_cor(palavra)
                         + (1 - peso_cor) * prob_global(palavra)

`peso_cor` perto de 1 => o vocabulário típico da cor domina a geração.
`peso_cor` perto de 0 => mais caos geral, ignorando a cor quase por
completo. É o "dial" que o projeto pediu para controlar o peso da cor.
A mesma mistura (função `_misturar`) também é usada para sortear as
keywords da carta -- ver abaixo.

Três correções importantes feitas após os primeiros testes
===========================================================
1. KEYWORDS SEPARADAS DO CORPO DO TEXTO
   No oracle_text real, uma carta como "Serra Angel" tem o texto em DUAS
   partes bem distintas: uma linha só com keywords ("Flying, vigilance")
   e, em outra linha, o texto de habilidade "de verdade". A primeira
   versão deste módulo juntava tudo numa string só antes de tokenizar,
   então a cadeia de Markov misturava keywords com texto corrido de forma
   confusa. Agora, durante o treino, cada LINHA do oracle_text é
   classificada: se a linha inteira é só uma lista de keywords
   conhecidas separadas por vírgula, ela vira uma entrada nos contadores
   `keywords_por_cor`/`keywords_global` (não entra na cadeia de texto). Se
   não, ela é tratada como texto de habilidade normal. Na geração, as
   keywords são sorteadas separadamente e ficam numa linha própria, no
   topo do texto, como nas cartas reais.

2. AUTORREFERÊNCIA ("this creature") DOMINANDO O TEXTO
   Desde a atualização de regras de 2023, o Oracle Text da Scryfall usa
   "this creature" / "this artifact" / etc. no lugar de repetir o nome da
   própria carta -- e isso faz esse bigrama aparecer em uma fração enorme
   de TODAS as cartas do jogo, dominando estatisticamente a cadeia.
   A correção: durante o treino, qualquer ocorrência de "this <tipo>"
   (creature/artifact/enchantment/planeswalker/permanent/land/spell) OU
   do nome literal daquela carta específica é substituída por um token
   especial "~" ANTES de entrar na cadeia. Na hora de gerar, esse "~" é
   substituído pelo nome real que o usuário digitou -- então a carta
   "fala de si mesma" usando o próprio nome, como nas cartas antigas,
   só que sem o viés estatístico do "this creature" genérico.

3. HABILIDADES ATIVADAS E CUSTOS COM SÍMBOLOS ({T}, {2}, {W}...)
   Habilidades ativadas no Oracle Text aparecem como
   "{T}: Add {G}." ou "{1}, {T}: Draw a card." -- custo (símbolos de
   tap/mana/genérico, às vezes com itens não-mana separados por
   vírgula), dois-pontos, efeito. A primeira versão deste módulo usava um
   filtro `^[a-zA-Z']+$` na tokenização, que DESCARTAVA tudo que não fosse
   letra: os símbolos {T}/{W}/{2}, os numerais e a pontuação sumiam, então
   "{T}: Add {G}." virava só ["add"] e a cadeia nunca aprendia a estrutura
   de custo. Agora:
   a) os símbolos {...} viram tokens atômicos preservados verbatim
      ({T}, {W}, {2}, {W/U}, {X}... -- o que o corpus tiver);
   b) vírgula e dois-pontos também são tokens (separadores de custo);
   c) na junção, símbolos adjacentes se agrudam ({2}{U}) e , / : grudam
      no token anterior, como no Oracle real;
   d) um "lint" (_corrigir_habilidade_ativada) garante a regra do usuário:
      linha iniciada por custo precisa de ":" antes do efeito -- e de ","
      entre itens de custo distintos ({T}, {2}: ...). Sem isso, a cadeia
      (mistura cor/global + backoff) às vezes gerava "{T} draw a card".
   ATENÇÃO: o modelo pré-treinado (models/modelo_markov.pkl) precisa ser
   re-treinado depois desta mudança, senão o vocabulário não tem os
   símbolos e a geração nunca os emite.

4. NÚMEROS E MODIFICADORES DE P/T
   O filtro antigo também descartava números e sinais: "deals 3 damage"
   virava "deals damage", e "gets +2/+2 until end of turn" virava "gets
   until end of turn". Agora números ("3", "-1") e modificadores de
   poder/resistência ("+1/+1", "-1/-1", "2/2", "+X/+X") são preservados
   como tokens atômicos. Assim a cadeia aprende "deals <n> damage",
   "gets <modificador> until end of turn" e "put a +1/+1 counter on ...".
   Também normalizamos o sinal de menos Unicode (−) para "-" antes de
   tokenizar. Esse modelo também precisa ser re-treinado após a mudança.
   Por fim, um lint (_corrigir_frases_incompletas) cobre os casos em que
   a cadeia ainda gera "gets until..." ou "deals damage to..." sem valor:
   ele insere um modificador de P/T ou um número, mas preserva frases
   válidas como "deals damage equal to ...".

Modelos por tipo de carta
=========================
O modelo antigo só olhava a cor, então o texto de uma criatura podia
"vazar" para um instantâneo (e vice-versa). Como o projeto agora gera
Criatura, Instantânea, Feitiço, Artefato, Encantamento e Terreno, o treino
é feito por TIPO: um ModeloMarkovPorCor separado para cada tipo de carta,
usando só as cartas daquele tipo (ver pipeline/tipos.py). O artefato salvo
é um ModeloTextoPorTipo -- um wrapper que escolhe o modelo do tipo pedido
(ou sorteia um tipo, ponderado pela frequência real no corpus).

Terrenos são incolores, então o modelo de terreno usa `color_identity`
(que diz quais cores de mana ele produz) no lugar de `colors`; o modelo
global de terreno inclui também os terrenos incolores de utilidade.

Consequência prática: instants/sorceries aprendem textos de mágica
(efeitos que não viram permanentes) e artefatos/encantamentos/terrenos
aprendem textos de permanente -- incluindo habilidades ativadas. Instants
e sorceries não têm habilidades ativadas ("custo + ':' + efeito"), então
essas linhas são filtradas no treino e evitadas na geração.

Por que misturar com um modelo global (e não só usar o da cor)
================================================================
Com poucos milhares de cartas por cor, um modelo de TRIGRAMAS treinado só
com os textos daquela cor é esparso. Por isso o modelo faz "stupid
backoff": tenta trigrama -> se não há dados suficientes, cai para bigrama
-> se ainda não há, cai para unigrama -- e em cada nível mistura com o
modelo global (mais denso) para preencher lacunas em vez de travar.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import pandas as pd

from .mecanicas import DICIONARIO_MECANICAS, inferir_mecanica
from .tipos import (
    TIPOS_SEM_HABILIDADE_ATIVADA,
    TIPOS_USAM_COLOR_IDENTITY,
    TIPOS_VALIDOS,
    tipo_da_carta,
)

TOKEN_INICIO = "<INICIO>"
TOKEN_FIM = "<FIM>"
TOKEN_AUTORREFERENCIA = "~"

CORES_VALIDAS = ["W", "U", "B", "R", "G"]

# Keywords conhecidas (lado esquerdo = como aparecem em minúsculo no
# oracle_text; lado direito = como devem aparecer no texto final da
# carta gerada). Não é uma lista exaustiva de TODAS as keywords que já
# existiram em Magic -- é um conjunto curado das mais comuns, suficiente
# para o escopo do projeto. Fácil de estender depois.
CANONICAL_KEYWORD_DISPLAY = {
    "flying": "Flying", "trample": "Trample", "vigilance": "Vigilance",
    "haste": "Haste", "lifelink": "Lifelink", "deathtouch": "Deathtouch",
    "first strike": "First strike", "double strike": "Double strike",
    "menace": "Menace", "reach": "Reach", "defender": "Defender",
    "indestructible": "Indestructible", "hexproof": "Hexproof",
    "flash": "Flash", "prowess": "Prowess", "flashback": "Flashback",
    "convoke": "Convoke", "delve": "Delve", "afterlife": "Afterlife",
    "myriad": "Myriad", "changeling": "Changeling", "shroud": "Shroud",
    "fear": "Fear", "intimidate": "Intimidate", "skulk": "Skulk",
    "unleash": "Unleash", "bushido": "Bushido", "rampage": "Rampage",
    "exalted": "Exalted", "annihilator": "Annihilator", "infect": "Infect",
    "wither": "Wither", "persist": "Persist", "undying": "Undying",
    "flanking": "Flanking", "banding": "Banding", "cascade": "Cascade",
    "riot": "Riot", "evolve": "Evolve", "outlast": "Outlast",
    "amass": "Amass", "mentor": "Mentor",
}

# Keywords "parametrizadas", onde o número/cor varia carta a carta (ex:
# "Ward {2}", "Protection from red") -- reconhecidas pelo prefixo, e o
# texto original (sem o reminder text) é preservado tal como apareceu.
KEYWORDS_PARAMETRIZADAS = ("ward", "protection from")

# Tipos de permanente/objeto usados no padrão "this <tipo>" que o Oracle
# Text moderno usa pra autorreferência (ver item 2 no docstring acima).
_TIPOS_AUTORREFERENCIA = (
    "legendary artifact creature", "legendary enchantment creature",
    "artifact creature", "enchantment creature", "creature", "artifact",
    "enchantment", "planeswalker", "permanent", "land", "spell", "card",
)
_PADRAO_AUTORREFERENCIA_GENERICA = re.compile(
    r"\bthis (" + "|".join(_TIPOS_AUTORREFERENCIA) + r")\b", re.IGNORECASE
)

# Símbolos de custo/mana do Oracle Text ({T}, {W}, {U}, {B}, {R}, {G},
# {C}, {X}, {2}, {W/U}, {W/P}...). Cada "{...}" vira UM token atômico,
# preservado verbatim -- ver item 3 no docstring do módulo.
_PADRAO_SIMBOLO = re.compile(r"\{[^{}]*\}")

# Símbolos de girar/desgirar. Quando encostam em outro símbolo ({T}{2}),
# o Oracle real separa com vírgula ("{T}, {2}"), porque são itens de custo
# DISTINTOS; já símbolos de mana puros se agrudam ("{2}{U}" = um custo
# único). Usado por _corrigir_habilidade_ativada.
_SIMBOLOS_TAP = {"{T}", "{Q}"}


def _eh_simbolo(token: str) -> bool:
    """True se o token é um símbolo de custo/mana do Oracle ({T}, {2},
    {W/U}...)."""
    return bool(_PADRAO_SIMBOLO.fullmatch(token))


def _cor_em(cs, cor: str) -> bool:
    """Verifica se `cor` está no valor da coluna `colors` de uma carta.

    Importante: NÃO usamos `isinstance(cs, list)` aqui. A coluna `colors`
    é uma lista de strings quando o DataFrame vem direto do JSON, mas
    depois de salvar/ler via Parquet (pandas + pyarrow), cada valor volta
    como `numpy.ndarray`, não `list`. Usamos `in` dentro de um try/except
    em vez de checar o tipo: funciona pra list, tuple, ndarray, e trata
    com segurança valores ausentes (None, NaN) que não são iteráveis."""
    try:
        return cor in cs
    except TypeError:
        return False


def _garantir_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _dividir_em_linhas(texto: str) -> List[str]:
    """Quebra o oracle_text em linhas (cada ability/linha do card real é
    preservada como uma unidade separada), removendo reminder text entre
    parênteses e espaços em excesso de cada linha individualmente."""
    if not isinstance(texto, str):
        return []
    linhas = []
    for linha in texto.split("\n"):
        linha = re.sub(r"\([^)]*\)", "", linha)
        linha = re.sub(r"\s+", " ", linha).strip()
        if linha:
            linhas.append(linha)
    return linhas


def _eh_keyword(segmento: str) -> Optional[str]:
    """Se `segmento` (um pedaço de linha separado por vírgula) é uma
    keyword reconhecida, devolve como ela deve aparecer no texto final.
    Caso contrário, devolve None."""
    seg_limpo = segmento.strip().rstrip(".")
    seg_lower = seg_limpo.lower()

    if seg_lower in CANONICAL_KEYWORD_DISPLAY:
        return CANONICAL_KEYWORD_DISPLAY[seg_lower]

    for prefixo in KEYWORDS_PARAMETRIZADAS:
        if seg_lower.startswith(prefixo):
            return seg_limpo

    return None


def _linha_e_so_keywords(linha: str) -> Optional[List[str]]:
    """Se TODOS os pedaços de `linha` (separados por vírgula) são
    keywords reconhecidas, devolve a lista de keywords (no formato de
    exibição). Caso contrário (é texto de habilidade normal), devolve
    None -- mesmo que a linha mencione uma keyword no meio de uma frase
    (ex: "As long as ~ has flying, ..."), já que aí ela não é uma linha
    SÓ de keywords."""
    segmentos = [s.strip() for s in linha.split(",")]
    resultado = []
    for seg in segmentos:
        kw = _eh_keyword(seg)
        if kw is None:
            return None
        resultado.append(kw)
    return resultado


def _normalizar_autorreferencia(linha: str, nome_carta: str) -> str:
    """Substitui tanto o padrão genérico "this creature/artifact/..."
    quanto o nome literal desta carta específica pelo token especial
    TOKEN_AUTORREFERENCIA, ANTES da tokenização. Isso evita que "this
    creature" (hiperfrequente no corpus pós-2023) domine as estatísticas
    da cadeia, e evita que o nome de uma carta real vaze pro texto de uma
    carta gerada completamente diferente."""
    linha = _PADRAO_AUTORREFERENCIA_GENERICA.sub(TOKEN_AUTORREFERENCIA, linha)
    if nome_carta:
        linha = re.sub(re.escape(nome_carta), TOKEN_AUTORREFERENCIA, linha, flags=re.IGNORECASE)
    return linha


def _tokenizar(texto: str) -> List[str]:
    """Tokeniza uma linha do oracle_text, separando-a em dois tipos de
    token: símbolos de custo/mana ({T}, {2}, {W/U}...), preservados
    VERBATIM como tokens atômicos, e o texto ao redor, tokenizado pelo
    NLTK. Ver item 3 no docstring do módulo -- sem isso os símbolos e a
    pontuação de custo eram descartados e a estrutura de habilidade
    ativada se perdia."""
    _garantir_nltk()
    resultado: List[str] = []
    pos = 0
    for m in _PADRAO_SIMBOLO.finditer(texto):
        resultado.extend(_tokenizar_texto(texto[pos:m.start()]))
        resultado.append(m.group(0))
        pos = m.end()
    resultado.extend(_tokenizar_texto(texto[pos:]))
    return resultado


# Números ("3", "-1") e modificadores de P/T ("+1/+1", "-1/-1", "2/2",
# "+X/+X") precisam sobreviver à tokenização: sem eles a cadeia aprende
# "deals damage" e "gets until end of turn", sem o valor. Ver item 4 no
# docstring do módulo.
_PADRAO_NUMERO = re.compile(r"[+-]?\d+")
_PADRAO_PT = re.compile(r"[+-]?(?:\d+|[Xx])/[+-]?(?:\d+|[Xx])")


def _eh_numero_ou_pt(token: str) -> bool:
    return bool(
        _PADRAO_NUMERO.fullmatch(token) or _PADRAO_PT.fullmatch(token)
    )


def _tokenizar_texto(texto: str) -> List[str]:
    """Tokeniza um trecho de oracle_text SEM símbolos {...}. Mantém:
      - palavras (minúsculas; o token especial de autorreferência "~" é
        preservado como está, ver _normalizar_autorreferencia);
      - números e modificadores de P/T ("3", "-1", "+1/+1", "2/2",
        "+X/+X"), que antes eram descartados e deixavam frases como
        "deals damage" / "gets until end of turn";
      - vírgula e dois-pontos, que são os separadores de custo/efeito das
        habilidades ativadas ("{1}, {T}: ...") e também aparecem no meio
        das frases normais.
    Qualquer outra pontuação (ponto final, aspas, parênteses etc.) é
    descartada -- o ponto final é re-adicionado na saída final."""
    # Alguns oracle_text usam o sinal de menos Unicode (−). Normalizamos
    # para ASCII antes de tokenizar, para "-1/-1" cair no mesmo padrão.
    texto = texto.replace("\u2212", "-")
    tokens = nltk.word_tokenize(texto)
    resultado = []
    for t in tokens:
        if t in (",", ":"):
            resultado.append(t)
        elif t == TOKEN_AUTORREFERENCIA or re.match(r"^[a-zA-Z']+$", t):
            resultado.append(t if t == TOKEN_AUTORREFERENCIA else t.lower())
        elif _eh_numero_ou_pt(t):
            resultado.append(t)
    return resultado


_MODIFICADORES_PT = (
    "+1/+1", "+2/+2", "+1/+0", "+2/+0", "+3/+3", "+4/+4", "-1/-1", "-2/-2",
)
_VALORES_DANO = ("1", "2", "3", "4", "5")


def _corrigir_frases_incompletas(
    tokens: List[str], rng: random.Random
) -> List[str]:
    """Insere valor em dois padrões que a cadeia às vezes gera incompletos:

      - "gets until/for/and..." -> insere um modificador de P/T antes
        ("gets +1/+1 until ...");
      - "deals damage to..." sem "equal" por perto e sem contexto de
        gatilho ("whenever ...") -> insere um número antes de "damage"
        ("deals 3 damage to ...").

    A segunda regra é conservadora: "deals damage equal to ..." é uma
    construção válida e não é tocada. O objetivo é só garantir que os
    textos não saiam com "gets" vazio nem com dano sem valor.
    """
    tokens = list(tokens)
    resultado: List[str] = []
    for i, token in enumerate(tokens):
        resultado.append(token)

        if token == "gets":
            proximo = tokens[i + 1] if i + 1 < len(tokens) else None
            if proximo in {None, "until", "for", "and", ",", ":"}:
                resultado.append(rng.choice(_MODIFICADORES_PT))

        elif token == "deals":
            if (
                i + 1 < len(tokens)
                and tokens[i + 1] == "damage"
                and "equal" not in tokens[i + 2 : i + 14]
                and not any(
                    marcador in {"whenever", "when", "if", "time", "while", "as", "long"}
                    for marcador in tokens[max(0, i - 8) : i]
                )
            ):
                resultado.append(rng.choice(_VALORES_DANO))

    return resultado


def _corrigir_habilidade_ativada(tokens: List[str]) -> List[str]:
    """Aplica a regra de formatação de habilidades ativadas (ver item 3 no
    docstring do módulo): se a linha COMEÇA com um custo (símbolos), ele
    tem que ser seguido de ":" antes do efeito -- a menos que a linha já
    tenha um ":" em algum lugar. Também separa com "," os itens de custo
    distintos ({T}{2} -> "{T}, {2}"), mantendo agrupados os símbolos de um
    mesmo custo ({2}{U} continua "{2}{U}").

    É um "lint" de segurança: como a cadeia mistura o modelo da cor com o
    global e faz backoff até unigrama, ela às vezes gera sequências como
    "{T} draw a card" (sem ":") -- o corpo do texto gerado é ajustado aqui
    antes de virar a linha final."""
    tokens = list(tokens)

    # Tira pontuação pendurada do fim (a cadeia pode ter terminado no meio
    # do custo) -- o ponto final é adicionado depois, na montagem da linha.
    while tokens and tokens[-1] in (",", ":"):
        tokens.pop()

    if not tokens or not _eh_simbolo(tokens[0]):
        return tokens  # não é habilidade ativada; nada a corrigir

    # Cluster de custo no começo: símbolos e vírgulas consecutivos.
    fim_custo = 0
    while fim_custo < len(tokens) and (
        _eh_simbolo(tokens[fim_custo]) or tokens[fim_custo] == ","
    ):
        fim_custo += 1

    custo = tokens[:fim_custo]
    resto = tokens[fim_custo:]

    custo_ajustado: List[str] = []
    for tok in custo:
        anterior = custo_ajustado[-1] if custo_ajustado else None
        if (
            anterior is not None
            and _eh_simbolo(anterior)
            and _eh_simbolo(tok)
            and (anterior in _SIMBOLOS_TAP or tok in _SIMBOLOS_TAP)
        ):
            # Girar ({T}/{Q}) não se agruma a outro símbolo: são itens de
            # custo distintos, separados por vírgula no Oracle real.
            custo_ajustado.append(",")
        custo_ajustado.append(tok)

    if ":" not in tokens and resto:
        custo_ajustado.append(":")

    resultado = custo_ajustado + resto

    # Normaliza repetições de pontuação vindas de custos quebrados, para
    # não sair coisas como "{T},:" ou "{T},," no texto final:
    #   - vírgula logo antes de ":" desaparece ("{T}, :" -> "{T}:");
    #   - vírgulas/dois-pontos consecutivos colapsam em um só.
    limpo: List[str] = []
    for tok in resultado:
        if tok in (",", ":") and limpo and limpo[-1] in (",", ":"):
            if tok == ":":
                limpo[-1] = ":"  # a vírgula antes do ":" vira o próprio ":"
            continue
        limpo.append(tok)
    return limpo


def _juntar_tokens(tokens: List[str]) -> str:
    """Junta os tokens na formatação do Oracle real:
      - vírgula e dois-pontos grudam no token anterior ("{T}," e "{T}:");
      - símbolos de custo adjacentes se agrudam ("{2}{U}", não "{2} {U}");
      - o restante é separado por espaço.
    O ponto final não entra aqui -- é adicionado por _gerar_corpo."""
    texto = ""
    anterior: Optional[str] = None
    for tok in tokens:
        if not texto:
            texto = tok
        elif tok in (",", ":"):
            texto += tok  # gruda no token anterior
        elif _eh_simbolo(tok) and _eh_simbolo(anterior):
            texto += tok  # símbolos do mesmo custo se agrudam ({2}{U})
        else:
            texto += " " + tok
        anterior = tok
    return texto


def _capitalizar(texto: str) -> str:
    """Deixa maiúscula a primeira letra do texto e também a primeira letra
    do EFEITO de uma habilidade ativada (depois do ":"), como no Oracle
    real ("{T}: Add {G}.", "{1}, Sacrifice an artifact: Draw a card.").
    Os símbolos {...} são maiúsculos e não são afetados."""
    if not texto:
        return texto
    # Só a PRIMEIRA posição: o texto pode começar com o nome da carta já
    # capitalizado (ex: "Teste de Terreno enters tapped."). Procurar a
    # primeira letra minúscula em qualquer lugar transformaria "Teste" em
    # "TEste".
    if texto[0].islower():
        texto = texto[0].upper() + texto[1:]
    m = re.search(r":[^a-zA-Z]*([a-z])", texto)
    if m:
        texto = texto[: m.start(1)] + m.group(1).upper() + texto[m.end(1):]
    return texto


def _misturar(dist_cor: Counter, dist_global: Counter, peso_cor: float) -> Counter:
    """Mistura duas distribuições (cor e global) ponderadas por
    `peso_cor`, com fallback automático pro lado que tiver dado quando o
    outro estiver vazio (ver explicação no módulo-level docstring, item
    sobre o bug de peso_cor=1.0 + cor sem dado)."""
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


def _amostra_sem_reposicao(rng: random.Random, candidatos: List, pesos: List[float], n: int) -> List:
    """Sorteia até `n` itens distintos de `candidatos`, ponderados por
    `pesos`, sem repetir nenhum."""
    candidatos = list(candidatos)
    pesos = list(pesos)
    escolhidos = []
    for _ in range(min(n, len(candidatos))):
        if not candidatos or sum(pesos) <= 0:
            break
        escolhido = rng.choices(candidatos, weights=pesos, k=1)[0]
        idx = candidatos.index(escolhido)
        candidatos.pop(idx)
        pesos.pop(idx)
        escolhidos.append(escolhido)
    return escolhidos


def _palavra_semente(nome: Optional[str]) -> Optional[str]:
    """Usa o pipeline de inferência de mecânica (dicionário + WordNet, já
    existente em pipeline.mecanicas) só como ponto de partida opcional
    para a cadeia -- não como regra. Se o nome não ativar nenhuma
    mecânica reconhecível, retorna None e a cadeia começa "do zero"."""
    if not nome:
        return None
    mecanica, origem = inferir_mecanica(nome)
    if origem == "padrao":
        return None
    termos = DICIONARIO_MECANICAS.get(mecanica)
    return termos[0] if termos else None


class ModeloMarkovPorCor:
    """Modelos de bigramas/trigramas (texto de habilidade) e de
    frequência de keywords, treinados por cor, mais um modelo global
    (todas as cores), usados em conjunto na geração.

    Cada instância é treinada com as cartas de UM tipo (creature, instant,
    sorcery, artifact, enchantment, land -- ver pipeline/tipos.py). O
    parâmetro `tipo` também define as regras de estrutura: instants/
    sorceries não podem ter habilidades ativadas (custo + ":"), então
    essas linhas são ignoradas no treino e evitadas na geração.

    Para terrenos, `usar_color_identity=True` faz o treino usar
    `color_identity` no lugar de `colors`: terrenos são incolores, mas a
    identidade de cor diz quais mana eles produzem (um "Forest" tem
    identidade G), então o modelo de terreno aprende o vocabulário certo
    para cada cor. O modelo global é sempre treinado com TODAS as cartas do
    tipo, uma vez cada (inclusive terrenos incolores)."""

    def __init__(
        self,
        df_cartas: pd.DataFrame,
        tipo: str = "creature",
        usar_color_identity: bool = False,
    ):
        self.tipo = tipo
        self.usar_color_identity = usar_color_identity
        self.permitir_habilidade_ativada = tipo not in TIPOS_SEM_HABILIDADE_ATIVADA
        self.bigramas_por_cor: Dict[str, Dict[str, Counter]] = {}
        self.trigramas_por_cor: Dict[str, Dict[Tuple[str, str], Counter]] = {}
        self.unigramas_por_cor: Dict[str, Counter] = {}

        self.bigramas_global: Dict[str, Counter] = defaultdict(Counter)
        self.trigramas_global: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.unigramas_global: Counter = Counter()

        # keywords_por_cor[cor]: Counter{"Flying": 812, "Trample": 340, ...}
        self.keywords_por_cor: Dict[str, Counter] = {}
        self.keywords_global: Counter = Counter()

        # contagem_keywords_por_cor[cor]: Counter{0: 4200, 1: 1500, 2: 300, ...}
        # -- quantas cartas daquela cor tinham 0, 1, 2... keywords no total.
        # Usado pra sortear QUANTAS keywords a carta gerada deve ter.
        self.contagem_keywords_por_cor: Dict[str, Counter] = {}
        self.contagem_keywords_global: Counter = Counter()

        self._treinar(df_cartas)

    # -- treino -----------------------------------------------------

    def _processar_cartas(
        self,
        cartas: pd.DataFrame,
        bigramas: Dict[str, Counter],
        trigramas: Dict[Tuple[str, str], Counter],
        unigramas: Counter,
        keywords: Counter,
        contagem_keywords: Counter,
    ) -> None:
        """Acumula os contadores de texto/keywords de `cartas` nos
        contadores passados. Usado tanto para o modelo de cada cor quanto
        para o modelo global."""
        for _, carta in cartas.iterrows():
            nome_carta = carta.get("name", "") or ""
            linhas = _dividir_em_linhas(carta.get("oracle_text", ""))

            total_keywords_desta_carta = 0

            for linha in linhas:
                # Instants/sorceries não têm habilidades ativadas
                # ("custo: efeito"). O corpus tem pouquíssimas
                # exceções (ativadas do cemitério/mão), que excluímos
                # do treino para o modelo não aprender esse formato.
                if (
                    not self.permitir_habilidade_ativada
                    and _PADRAO_SIMBOLO.match(linha)
                ):
                    continue

                kws_da_linha = _linha_e_so_keywords(linha)

                if kws_da_linha is not None:
                    # Linha é só keywords -- não entra na cadeia de
                    # texto, só nos contadores de keyword.
                    for kw in kws_da_linha:
                        keywords[kw] += 1
                    total_keywords_desta_carta += len(kws_da_linha)
                    continue

                # Linha de texto de habilidade "normal": normaliza
                # autorreferência e treina a cadeia NESTA linha
                # isoladamente (não encadeia com outras linhas/outras
                # habilidades da mesma carta, que são conceitualmente
                # independentes).
                linha_normalizada = _normalizar_autorreferencia(linha, nome_carta)
                tokens = [TOKEN_INICIO] + _tokenizar(linha_normalizada) + [TOKEN_FIM]
                if len(tokens) <= 2:
                    continue  # linha vazia depois de tokenizar

                unigramas.update(tokens)
                for w1, w2 in nltk.bigrams(tokens):
                    bigramas[w1][w2] += 1
                for w1, w2, w3 in nltk.trigrams(tokens):
                    trigramas[(w1, w2)][w3] += 1

            contagem_keywords[total_keywords_desta_carta] += 1

    def _treinar(self, df_cartas: pd.DataFrame):
        # Terrenos são incolores; para eles a cor vem da identidade de cor
        # (ver docstring da classe).
        coluna_cor = "color_identity" if self.usar_color_identity else "colors"

        for cor in CORES_VALIDAS:
            cartas_da_cor = df_cartas[
                df_cartas[coluna_cor].apply(lambda cs: _cor_em(cs, cor))
            ]

            bigramas: Dict[str, Counter] = defaultdict(Counter)
            trigramas: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
            unigramas: Counter = Counter()
            keywords: Counter = Counter()
            contagem_keywords: Counter = Counter()

            self._processar_cartas(
                cartas_da_cor, bigramas, trigramas, unigramas, keywords, contagem_keywords
            )

            self.bigramas_por_cor[cor] = bigramas
            self.trigramas_por_cor[cor] = trigramas
            self.unigramas_por_cor[cor] = unigramas

            self.keywords_por_cor[cor] = keywords
            self.contagem_keywords_por_cor[cor] = contagem_keywords

        # Modelo global: TODAS as cartas do tipo, uma vez cada. Fazer isso
        # separadamente (em vez de somar os modelos por cor) evita duplicar
        # cartas multicolor e -- crucialmente -- inclui cartas incolores,
        # como os terrenos de utilidade que não têm identidade de cor.
        self._processar_cartas(
            df_cartas,
            self.bigramas_global,
            self.trigramas_global,
            self.unigramas_global,
            self.keywords_global,
            self.contagem_keywords_global,
        )

    # -- geração: texto de habilidade ---------------------------------

    def _distribuicao_proxima_palavra(
        self, cor: str, contexto: Tuple[str, ...], peso_cor: float
    ) -> Counter:
        """Mistura a distribuição condicional do modelo da COR com a do
        modelo GLOBAL, ponderada por `peso_cor`. Faz backoff
        trigrama -> bigrama -> unigrama quando o contexto mais específico
        não tem dados em NENHUM dos dois modelos."""
        dist_cor, dist_global = Counter(), Counter()

        if len(contexto) >= 2:
            par = tuple(contexto[-2:])
            dist_cor = self.trigramas_por_cor[cor].get(par, Counter())
            dist_global = self.trigramas_global.get(par, Counter())

        if not dist_cor and not dist_global and contexto:
            ultima = contexto[-1]
            dist_cor = self.bigramas_por_cor[cor].get(ultima, Counter())
            dist_global = self.bigramas_global.get(ultima, Counter())

        if not dist_cor and not dist_global:
            dist_cor = self.unigramas_por_cor[cor]
            dist_global = self.unigramas_global

        return _misturar(dist_cor, dist_global, peso_cor)

    def _gerar_corpo(
        self, cor: str, peso_cor: float, nome: Optional[str], max_palavras: int, rng: random.Random
    ) -> Tuple[str, Optional[str]]:
        semente = _palavra_semente(nome)
        vocabulario_conhecido = set(self.unigramas_por_cor[cor]) | set(self.unigramas_global)

        if semente and semente in vocabulario_conhecido:
            palavras_geradas = [semente]
            contexto = [semente]
        else:
            semente = None
            palavras_geradas = []
            contexto = [TOKEN_INICIO]

        while len(palavras_geradas) < max_palavras:
            dist = self._distribuicao_proxima_palavra(cor, tuple(contexto), peso_cor)
            dist.pop(TOKEN_INICIO, None)
            if not dist:
                break

            candidatos = list(dist.keys())
            pesos = list(dist.values())
            if not self.permitir_habilidade_ativada and not palavras_geradas:
                # Instants/sorceries não têm habilidades ativadas: impede
                # que a linha comece com um custo ("{...}: efeito").
                sem_custo = [
                    (c, w)
                    for c, w in zip(candidatos, pesos)
                    if not _eh_simbolo(c)
                ]
                if sem_custo:
                    candidatos = [c for c, _ in sem_custo]
                    pesos = [w for _, w in sem_custo]

            proxima = rng.choices(candidatos, weights=pesos, k=1)[0]

            if proxima == TOKEN_FIM:
                break

            palavras_geradas.append(proxima)
            contexto.append(proxima)

        # Substitui o token de autorreferência pelo nome real da carta
        # ANTES de juntar tudo -- é aqui que a carta gerada passa a
        # "falar de si mesma" pelo nome que o usuário escolheu.
        nome_para_substituir = nome if nome else "This creature"
        palavras_finais = [nome_para_substituir if p == TOKEN_AUTORREFERENCIA else p for p in palavras_geradas]

        # Completa padrões que saíram sem valor ("gets until...",
        # "deals damage to..." sem "equal") antes de formatar.
        palavras_finais = _corrigir_frases_incompletas(palavras_finais, rng)

        # Ajusta a estrutura de habilidade ativada (custo + ":" + efeito,
        # ver item 3 do docstring do módulo) e junta os tokens na
        # formatação do Oracle real (símbolos agrudados, "," / ":" colados).
        palavras_finais = _corrigir_habilidade_ativada(palavras_finais)
        texto = _juntar_tokens(palavras_finais)
        texto_formatado = f"{_capitalizar(texto)}." if texto else ""
        return texto_formatado, semente

    # -- geração: keywords ---------------------------------------------

    def _escolher_keywords(self, cor: str, peso_cor: float, rng: random.Random) -> List[str]:
        dist_contagem = _misturar(
            self.contagem_keywords_por_cor[cor], self.contagem_keywords_global, peso_cor
        )
        if not dist_contagem:
            return []

        contagens = list(dist_contagem.keys())
        pesos_contagem = list(dist_contagem.values())
        n_keywords = rng.choices(contagens, weights=pesos_contagem, k=1)[0]
        if n_keywords <= 0:
            return []

        dist_keywords = _misturar(self.keywords_por_cor[cor], self.keywords_global, peso_cor)
        if not dist_keywords:
            return []

        return _amostra_sem_reposicao(
            rng, list(dist_keywords.keys()), list(dist_keywords.values()), n_keywords
        )

    # -- geração: ponto de entrada --------------------------------------

    def gerar_texto(
        self,
        cor: str,
        peso_cor: float = 0.7,
        nome: Optional[str] = None,
        max_palavras: int = 20,
        seed: Optional[int] = None,
    ) -> dict:
        """Gera o texto de uma carta para a `cor` dada: uma linha de
        keywords (se houver) seguida do corpo de habilidade gerado pela
        cadeia de Markov. Retorna um dict com o texto final (já com
        keywords + corpo combinados), as keywords separadas, o corpo
        separado e a palavra-semente usada -- para fins de transparência.
        """
        if cor not in CORES_VALIDAS:
            raise ValueError(f"Cor inválida: {cor!r}. Use uma de {CORES_VALIDAS}.")

        rng = random.Random(seed)

        palavras_chave = self._escolher_keywords(cor, peso_cor, rng)
        corpo, semente = self._gerar_corpo(cor, peso_cor, nome, max_palavras, rng)

        partes = []
        if palavras_chave:
            partes.append(", ".join(palavras_chave))
        if corpo:
            partes.append(corpo)

        texto_final = "\n".join(partes) if partes else "(nenhum texto gerado)"

        return {
            "texto": texto_final,
            "palavras_chave": palavras_chave,
            "corpo": corpo,
            "semente": semente,
        }


class ModeloTextoPorTipo:
    """Guarda um `ModeloMarkovPorCor` por tipo de carta (creature, instant,
    sorcery, artifact, enchantment, land -- ver pipeline/tipos.py).

    O pipeline decide o tipo ANTES de gerar o texto: como cada modelo é
    treinado só com cartas daquele tipo, as habilidades geradas combinam
    com ele (um instantâneo gera efeitos de mágica, um terreno gera texto
    de terreno com habilidades de mana etc.)."""

    def __init__(self, df_cartas: pd.DataFrame, tipos: Optional[List[str]] = None):
        tipos = list(tipos) if tipos is not None else list(TIPOS_VALIDOS)
        self.tipos = tipos
        self.modelos: Dict[str, ModeloMarkovPorCor] = {}
        self.contagem_por_tipo: Counter = Counter()

        tipo_de_cada_carta = df_cartas["type_line"].map(tipo_da_carta)
        for tipo in tipos:
            cartas_do_tipo = df_cartas[tipo_de_cada_carta == tipo]
            if cartas_do_tipo.empty:
                continue
            self.contagem_por_tipo[tipo] = len(cartas_do_tipo)
            self.modelos[tipo] = ModeloMarkovPorCor(
                cartas_do_tipo,
                tipo=tipo,
                usar_color_identity=tipo in TIPOS_USAM_COLOR_IDENTITY,
            )

    def sortear_tipo(self, rng: Optional[random.Random] = None) -> str:
        """Sorteia um tipo suportado, ponderado pela frequência real no
        corpus (criaturas são as mais comuns)."""
        rng = rng or random.Random()
        tipos = list(self.modelos)
        pesos = [self.contagem_por_tipo[t] for t in tipos]
        return rng.choices(tipos, weights=pesos, k=1)[0]

    def gerar_texto(
        self,
        cor: str,
        tipo: Optional[str] = None,
        peso_cor: float = 0.7,
        nome: Optional[str] = None,
        max_palavras: int = 20,
        seed: Optional[int] = None,
    ) -> dict:
        """Gera o texto para (cor, tipo). Se `tipo` for None, sorteia um
        tipo ponderado pela frequência no corpus. O dict de retorno é o
        mesmo de `ModeloMarkovPorCor.gerar_texto`, com a chave extra
        "tipo"."""
        if tipo is None:
            tipo = self.sortear_tipo(random.Random(seed))
        if tipo not in self.modelos:
            raise ValueError(
                f"Tipo inválido: {tipo!r}. Use um de {sorted(self.modelos)}."
            )

        resultado = self.modelos[tipo].gerar_texto(
            cor,
            peso_cor=peso_cor,
            nome=nome,
            max_palavras=max_palavras,
            seed=seed,
        )
        resultado["tipo"] = tipo
        return resultado


def carregar_modelo(caminho: str = "models/modelo_markov.pkl") -> "ModeloTextoPorTipo":
    """Carrega o modelo pré-treinado do disco. Levanta um erro claro se o
    artefato ainda não foi gerado (ver treinar_modelo_markov.py)."""
    import pickle

    caminho_path = Path(caminho)
    if not caminho_path.exists():
        raise FileNotFoundError(
            f"{caminho_path} não encontrado. Treine o modelo antes com:\n"
            f"  python src/pipeline/treinar_modelo_markov.py"
        )
    with open(caminho_path, "rb") as f:
        return pickle.load(f)
