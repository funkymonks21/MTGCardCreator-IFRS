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

Por que misturar com um modelo global (e não só usar o da cor)
================================================================
Com ~5-6 mil cartas por cor, um modelo de TRIGRAMAS treinado só com os
textos daquela cor é esparso: muitos contextos de 2 palavras não se
repetem o suficiente para gerar continuações. Por isso o modelo faz
"stupid backoff": tenta trigrama -> se não há dados suficientes, cai para
bigrama -> se ainda não há, cai para unigrama. A mistura com o modelo
global (mais denso, porque tem todas as cores) ajuda a preencher essas
lacunas em vez de travar a geração.
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

CORES_VALIDAS = ["W", "U", "B", "R", "G"]


def _cor_em(cs, cor: str) -> bool:
    """Verifica se `cor` está no valor da coluna `colors` de uma carta.

    Importante: NÃO usamos `isinstance(cs, list)` aqui. A coluna `colors`
    é uma lista de strings quando o DataFrame vem direto do JSON, mas
    depois de salvar/ler via Parquet (pandas + pyarrow), cada valor volta
    como `numpy.ndarray`, não `list` -- um `isinstance(cs, (list, tuple))`
    deixa de bater com TODAS as linhas nesse caso, silenciosamente (foi
    exatamente o bug que zerou o treino: 0 cartas em todas as 5 cores).
    Usamos `in` dentro de um try/except em vez de checar o tipo: funciona
    pra list, tuple, ndarray, e trata com segurança valores ausentes
    (None, NaN) que não são iteráveis."""
    try:
        return cor in cs
    except TypeError:
        return False


def _garantir_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _limpar_oracle_text(texto: str) -> str:
    """Remove reminder text entre parênteses (ex: "(This creature can't
    be blocked except by...)") e normaliza espaços em branco."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"\([^)]*\)", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _tokenizar(texto: str) -> List[str]:
    _garantir_nltk()
    return [t.lower() for t in nltk.word_tokenize(texto) if re.match(r"^[a-zA-Z']+$", t)]


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
    """Modelos de bigramas/trigramas treinados por cor, mais um modelo
    global (todas as cores), usados em conjunto na geração."""

    def __init__(self, df_cartas: pd.DataFrame):
        self.bigramas_por_cor: Dict[str, Dict[str, Counter]] = {}
        self.trigramas_por_cor: Dict[str, Dict[Tuple[str, str], Counter]] = {}
        self.unigramas_por_cor: Dict[str, Counter] = {}

        self.bigramas_global: Dict[str, Counter] = defaultdict(Counter)
        self.trigramas_global: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.unigramas_global: Counter = Counter()

        self._treinar(df_cartas)

    # -- treino -----------------------------------------------------

    def _treinar(self, df_cartas: pd.DataFrame):
        for cor in CORES_VALIDAS:
            textos = df_cartas[df_cartas["colors"].apply(lambda cs: _cor_em(cs, cor))]["oracle_text"]

            bigramas: Dict[str, Counter] = defaultdict(Counter)
            trigramas: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
            unigramas: Counter = Counter()

            for texto in textos.dropna():
                tokens = [TOKEN_INICIO] + _tokenizar(_limpar_oracle_text(texto)) + [TOKEN_FIM]
                unigramas.update(tokens)

                for w1, w2 in nltk.bigrams(tokens):
                    bigramas[w1][w2] += 1
                    self.bigramas_global[w1][w2] += 1

                for w1, w2, w3 in nltk.trigrams(tokens):
                    trigramas[(w1, w2)][w3] += 1
                    self.trigramas_global[(w1, w2)][w3] += 1

            self.bigramas_por_cor[cor] = bigramas
            self.trigramas_por_cor[cor] = trigramas
            self.unigramas_por_cor[cor] = unigramas
            self.unigramas_global.update(unigramas)

    # -- geração ------------------------------------------------------

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

        # peso_cor só faz sentido como MISTURA quando os dois lados têm
        # dado pro mesmo contexto. Se só um dos dois tiver dado aqui (ex:
        # "destroy" nunca aparece em nenhum bigrama da cor Azul, mas
        # aparece no global), ele deve carregar a probabilidade inteira
        # -- multiplicar por um peso fixo e deixar o outro lado em zero
        # faria a distribuição inteira zerar (foi o bug encontrado nos
        # testes: peso_cor=1.0 + cor sem dado = ValueError no sorteio).
        tem_cor, tem_global = bool(dist_cor), bool(dist_global)
        if tem_cor and tem_global:
            peso_cor_efetivo, peso_global_efetivo = peso_cor, 1 - peso_cor
        elif tem_cor:
            peso_cor_efetivo, peso_global_efetivo = 1.0, 0.0
        elif tem_global:
            peso_cor_efetivo, peso_global_efetivo = 0.0, 1.0
        else:
            return Counter()  # nem a cor nem o global têm nada (vocabulário vazio)

        total_cor = sum(dist_cor.values()) or 1
        total_global = sum(dist_global.values()) or 1

        dist_final: Counter = Counter()
        for palavra, freq in dist_cor.items():
            dist_final[palavra] += peso_cor_efetivo * (freq / total_cor)
        for palavra, freq in dist_global.items():
            dist_final[palavra] += peso_global_efetivo * (freq / total_global)

        return dist_final

    def gerar_texto(
        self,
        cor: str,
        peso_cor: float = 0.7,
        nome: Optional[str] = None,
        max_palavras: int = 20,
        seed: Optional[int] = None,
    ) -> dict:
        """Gera um texto de regras para a `cor` dada. Retorna um dict com
        o texto final e a palavra-semente usada (se houver), para fins de
        transparência (mostrar como a cadeia começou)."""
        if cor not in CORES_VALIDAS:
            raise ValueError(f"Cor inválida: {cor!r}. Use uma de {CORES_VALIDAS}.")

        rng = random.Random(seed)
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

        texto = " ".join(palavras_geradas)
        texto_formatado = f"{texto[:1].upper()}{texto[1:]}." if texto else "(nenhum texto gerado)"

        return {"texto": texto_formatado, "semente": semente}


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
