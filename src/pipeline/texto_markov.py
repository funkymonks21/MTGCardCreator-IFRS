"""
Geração de texto de regras via modelo estatístico de linguagem (cadeia de
Markov sobre bigramas/trigramas), treinado separadamente por cor.

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

Duas correções importantes feitas após os primeiros testes
============================================================
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
    _garantir_nltk()
    tokens = nltk.word_tokenize(texto)
    resultado = []
    for t in tokens:
        if t == TOKEN_AUTORREFERENCIA or re.match(r"^[a-zA-Z']+$", t):
            resultado.append(t.lower() if t != TOKEN_AUTORREFERENCIA else t)
    return resultado


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
    (todas as cores), usados em conjunto na geração."""

    def __init__(self, df_cartas: pd.DataFrame):
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

    def _treinar(self, df_cartas: pd.DataFrame):
        for cor in CORES_VALIDAS:
            cartas_da_cor = df_cartas[df_cartas["colors"].apply(lambda cs: _cor_em(cs, cor))]

            bigramas: Dict[str, Counter] = defaultdict(Counter)
            trigramas: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
            unigramas: Counter = Counter()
            keywords: Counter = Counter()
            contagem_keywords: Counter = Counter()

            for _, carta in cartas_da_cor.iterrows():
                nome_carta = carta.get("name", "") or ""
                linhas = _dividir_em_linhas(carta.get("oracle_text", ""))

                total_keywords_desta_carta = 0

                for linha in linhas:
                    kws_da_linha = _linha_e_so_keywords(linha)

                    if kws_da_linha is not None:
                        # Linha é só keywords -- não entra na cadeia de
                        # texto, só nos contadores de keyword.
                        for kw in kws_da_linha:
                            keywords[kw] += 1
                            self.keywords_global[kw] += 1
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
                        self.bigramas_global[w1][w2] += 1
                    for w1, w2, w3 in nltk.trigrams(tokens):
                        trigramas[(w1, w2)][w3] += 1
                        self.trigramas_global[(w1, w2)][w3] += 1

                contagem_keywords[total_keywords_desta_carta] += 1

            self.bigramas_por_cor[cor] = bigramas
            self.trigramas_por_cor[cor] = trigramas
            self.unigramas_por_cor[cor] = unigramas
            self.unigramas_global.update(unigramas)

            self.keywords_por_cor[cor] = keywords
            self.contagem_keywords_por_cor[cor] = contagem_keywords
            self.contagem_keywords_global.update(contagem_keywords)

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

        texto = " ".join(palavras_finais)
        texto_formatado = f"{texto[:1].upper()}{texto[1:]}." if texto else ""
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


def carregar_modelo(caminho: str = "models/modelo_markov.pkl") -> "ModeloMarkovPorCor":
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
