"""
Geração dos atributos "estruturais" da carta via regressão (scikit-learn):
custo de mana e poder/resistência.

Por que regressão?
==================
O texto de regras (gerado pela cadeia de Markov, ver texto_markov.py) é a
principal pista sobre o "tamanho" de uma carta: um texto curto e simples
tende a custar pouco; um texto longo, com muitos efeitos, tende a custar
mais. A ideia aqui é aprender essa relação diretamente das cartas reais da
Scryfall, em vez de inventar uma regra fixa (a versão anterior do
`generator.py` usava um placeholder bobo baseado no tamanho do nome).

Como funciona
=============
1. CUSTO DE MANA (`ModeloEstrutural.prever_custo_mana`)
   Treinado só com cartas MONOCOR e com custo "simples" (nenhum híbrido
   {W/U}, phyrexiano {W/P} ou {X}), para que o alvo seja limpo. Para cada
   carta extraímos dois números do `mana_cost`:
     - `cmc`: o custo de mana convertido (valor total);
     - `pips`: quantos símbolos da cor aparecem (ex: "{2}{W}{W}" -> 2).
   Um `MultiOutputRegressor(Ridge)` prevê esses dois valores a partir de:
     - TF-IDF (1-2 grams) do oracle_text;
     - features numéricas (nº de palavras, nº de linhas);
     - one-hot da cor da carta;
     - one-hot do TIPO de carta (creature, instant, sorcery, artifact,
       enchantment -- ver pipeline/tipos.py).
   O tipo entra como feature porque o custo típico depende dele: feitiços
   e instantâneas costumam ser mais baratos que criaturas do mesmo
   "tamanho" de texto, artefatos baratos são comuns etc.
   Na hora de gerar, o custo é remontado no formato do Oracle:
   `{generico}{W}{W}`, garantindo pelo menos um símbolo colorido quando o
   custo previsto é >= 1 (carta colorida sempre tem ao menos um pip).

2. PODER/RESISTÊNCIA (`ModeloEstrutural.prever_poder_resistencia`)
   Treinado só com CRIATURAS com P/T numérico. Prevê o par (poder,
   resistência) a partir do texto, da cor e do CUSTO DE MANA previsto --
   o custo entra como feature porque é o sinal mais forte de "quão grande"
   a criatura deve ser (o mesmo texto de uma carta barata e de uma cara
   produz corpos diferentes). Os valores previstos são arredondados e
   presos ao intervalo [0, 20]. Este modelo NÃO é usado para instants,
   sorceries, artefatos ou encantamentos não-criatura: o `generator` só
   chama `prever_poder_resistencia` quando o tipo é criatura (ver
   pipeline/tipos.py, TIPOS_COM_PODER_RESISTENCIA).

Artefato
========
Os dois modelos treinados são salvos juntos via joblib em
`models/modelos_estruturais.pkl` (versionado no git, como o modelo de
texto). Se os dados da Scryfall mudarem, retreine com
`python src/pipeline/treinar_modelos_estruturais.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .tipos import tipo_da_carta

CORES_VALIDAS = ["W", "U", "B", "R", "G"]

# Regularização do Ridge. Valores em torno de 1.0 tiveram o melhor erro
# médio no holdout (ver métricas impressas pelo script de treino).
ALPHA_RIDGE = 1.0

CAMINHO_PADRAO = "models/modelos_estruturais.pkl"

# Custo de mana "simples": apenas símbolos genéricos {0}..{n} e letras de
# cor {W}{U}{B}{R}{G}. Exclui híbridos, phyrexiano, {X}, {C}, {S} etc.,
# que complicam a contagem de pips -> cmc.
_MANA_SIMPLES = re.compile(r"^(?:\{[0-9]+\}|\{[WUBRG]\})+$")
_SIMBOLO = re.compile(r"\{([^}]*)\}")

# Colunas de features usadas por cada modelo. Manter como listas facilita
# garantir que treino e inferência montem exatamente o mesmo DataFrame.
FEATURES_CUSTO = ["texto", "n_palavras", "n_linhas", "cor", "tipo"]
FEATURES_PT = ["texto", "n_palavras", "n_linhas", "cmc", "cor"]
NUMERICAS_CUSTO = ["n_palavras", "n_linhas"]
NUMERICAS_PT = ["n_palavras", "n_linhas", "cmc"]


def _limpar_texto(texto) -> str:
    """Normaliza o oracle_text para servir de entrada do TF-IDF: remove
    reminder text entre parênteses, troca quebras de linha por espaço,
    colapsa espaços e passa para minúsculas. Símbolos de mana ({T}, {2}...)
    são mantidos (em minúsculo: {t}, {2}...), tanto no treino quanto na
    inferência -- o texto gerado pela cadeia de Markov passa pela mesma
    função antes de ser vetorizado."""
    texto = re.sub(r"\([^)]*\)", " ", str(texto))
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


def _cor_da_carta(colors) -> Optional[str]:
    """Extrai a única cor de uma carta monocolor. Devolve None para cartas
    incolores ou multicolor (não usadas no treino do custo). Funciona com
    list, tuple e numpy.ndarray (ver comentário em texto_markov._cor_em)."""
    try:
        return colors[0] if len(colors) == 1 else None
    except TypeError:
        return None


def _cor_para_modelo_pt(colors) -> str:
    """Para o modelo de P/T usamos todas as criaturas: cor única quando
    existe, "M" para multicolor e "C" para incolor. A one-hot lida com
    essas categorias sem problema."""
    try:
        if len(colors) == 1:
            return colors[0]
        return "M" if len(colors) > 1 else "C"
    except TypeError:
        return "C"


def _contar_pips(mana_cost: str) -> int:
    """Conta quantos símbolos de cor ({W}/{U}/{B}/{R}/{G}) há no custo."""
    return sum(1 for s in _SIMBOLO.findall(mana_cost) if s in "WUBRG")


def preparar_dados_custo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra e monta as features/alvos do modelo de custo de mana.

    Retorna (X, y): X com FEATURES_CUSTO e y com as colunas "cmc" e "pips".
    """
    dados = pd.DataFrame(
        {
            "texto": df["oracle_text"].map(_limpar_texto),
            "n_palavras": df["oracle_text"].map(
                lambda t: len(_limpar_texto(t).split())
            ),
            "n_linhas": df["oracle_text"].map(lambda t: str(t).count("\n") + 1),
            "cor": df["colors"].map(_cor_da_carta),
            "tipo": df["type_line"].map(tipo_da_carta),
            "mana_cost": df["mana_cost"].fillna(""),
            "cmc": df["cmc"],
        }
    )
    dados = dados[
        dados["cor"].notna()
        & dados["tipo"].notna()
        & dados["mana_cost"].str.fullmatch(_MANA_SIMPLES)
        & dados["cmc"].notna()
    ].copy()
    dados["pips"] = dados["mana_cost"].map(_contar_pips)
    return dados[FEATURES_CUSTO], dados[["cmc", "pips"]]


def preparar_dados_pt(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra criaturas com P/T numérico e monta features/alvos do modelo
    de poder/resistência. Retorna (X, y), com y = ["poder", "resistencia"]."""
    criaturas = df[
        df["type_line"].fillna("").str.contains("Creature")
        & df["power"].astype(str).str.fullmatch(r"\d+")
        & df["toughness"].astype(str).str.fullmatch(r"\d+")
    ].copy()

    dados = pd.DataFrame(
        {
            "texto": criaturas["oracle_text"].map(_limpar_texto),
            "n_palavras": criaturas["oracle_text"].map(
                lambda t: len(_limpar_texto(t).split())
            ),
            "n_linhas": criaturas["oracle_text"].map(
                lambda t: str(t).count("\n") + 1
            ),
            "cmc": criaturas["cmc"].fillna(0),
            "cor": criaturas["colors"].map(_cor_para_modelo_pt),
            "poder": criaturas["power"].astype(int),
            "resistencia": criaturas["toughness"].astype(int),
        }
    )
    return dados[FEATURES_PT], dados[["poder", "resistencia"]]


def _construir_pipeline(numericas: List[str], categoricas: List[str]) -> Pipeline:
    """Pipeline scikit-learn compartilhado: TF-IDF do texto + escalonamento
    das features numéricas + one-hot das categóricas (cor e, no custo,
    tipo), alimentando um Ridge multi-saída (prevê os dois alvos de uma
    vez)."""
    pre_processamento = ColumnTransformer(
        [
            (
                "texto",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=4000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                "texto",
            ),
            ("numericas", StandardScaler(), numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )
    return Pipeline(
        [
            ("pre_processamento", pre_processamento),
            ("regressao", MultiOutputRegressor(Ridge(alpha=ALPHA_RIDGE))),
        ]
    )


def construir_pipeline_custo() -> Pipeline:
    """Pipeline de regressão do custo de mana (alvos: cmc e pips; features
    incluem cor e tipo)."""
    return _construir_pipeline(NUMERICAS_CUSTO, ["cor", "tipo"])


def construir_pipeline_pt() -> Pipeline:
    """Pipeline de regressão de poder/resistência (alvos: poder e
    resistência; só criaturas, então não há feature de tipo)."""
    return _construir_pipeline(NUMERICAS_PT, ["cor"])


def _features_inferencia(
    texto: str,
    cor: str,
    tipo: Optional[str] = None,
    cmc: Optional[float] = None,
) -> pd.DataFrame:
    """Monta o DataFrame de UMA linha com as mesmas colunas usadas no
    treino, a partir do texto gerado pela cadeia de Markov."""
    texto_limpo = _limpar_texto(texto)
    linha = {
        "texto": texto_limpo,
        "n_palavras": len(texto_limpo.split()),
        "n_linhas": str(texto).count("\n") + 1,
        "cor": cor,
    }
    if tipo is not None:
        linha["tipo"] = tipo
    if cmc is not None:
        linha["cmc"] = float(cmc)
    return pd.DataFrame([linha])


def _formatar_custo(cmc: int, pips: int, cor: str) -> str:
    """Remonta o custo no formato do Oracle: genérico primeiro, depois os
    símbolos coloridos. Ex: cmc=4, pips=2, cor="W" -> "{2}{W}{W}"."""
    generico = cmc - pips
    return (f"{{{generico}}}" if generico > 0 else "") + f"{{{cor}}}" * pips


class ModeloEstrutural:
    """Agrupa os dois modelos de regressão (custo de mana e P/T) e oferece
    a interface usada pelo pipeline: prever custo, depois prever P/T usando
    o custo previsto."""

    def __init__(self, modelo_custo: Pipeline, modelo_pt: Pipeline):
        self.modelo_custo = modelo_custo
        self.modelo_pt = modelo_pt

    def prever_custo_mana(self, texto: str, cor: str, tipo: str) -> Tuple[str, int]:
        """Prevê o custo de mana de uma carta da cor `cor` e do tipo `tipo`
        a partir do seu texto. Retorna (custo_formatado, cmc_previsto)."""
        if cor not in CORES_VALIDAS:
            raise ValueError(f"Cor inválida: {cor!r}. Use uma de {CORES_VALIDAS}.")

        previsao = self.modelo_custo.predict(
            _features_inferencia(texto, cor, tipo=tipo)
        )[0]
        cmc = max(0, int(round(float(previsao[0]))))
        pips = max(0, int(round(float(previsao[1]))))

        if cmc == 0:
            return "{0}", 0

        # Carta colorida precisa de pelo menos um símbolo da sua cor, e os
        # pips não podem passar do custo total (genérico >= 0).
        pips = max(1, min(pips, cmc))
        return _formatar_custo(cmc, pips, cor), cmc

    def prever_poder_resistencia(
        self, texto: str, cor: str, cmc: int
    ) -> Tuple[int, int]:
        """Prevê (poder, resistência) de uma criatura da cor `cor`, usando
        o texto e o custo de mana previsto. Valores presos a [0, 20]."""
        previsao = self.modelo_pt.predict(
            _features_inferencia(texto, cor, cmc=cmc)
        )[0]
        poder = int(max(0, min(20, round(float(previsao[0])))))
        resistencia = int(max(0, min(20, round(float(previsao[1])))))
        return poder, resistencia

    def salvar(self, caminho: str = CAMINHO_PADRAO) -> Path:
        """Salva o modelo completo (os dois pipelines) com joblib."""
        destino = Path(caminho)
        destino.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destino)
        return destino

    @classmethod
    def carregar(cls, caminho: str = CAMINHO_PADRAO) -> "ModeloEstrutural":
        """Carrega o artefato salvo. Levanta um erro claro se ele ainda não
        foi treinado."""
        origem = Path(caminho)
        if not origem.exists():
            raise FileNotFoundError(
                f"{origem} não encontrado. Treine os modelos antes com:\n"
                f"  python src/pipeline/treinar_modelos_estruturais.py"
            )
        try:
            return joblib.load(origem)
        except Exception as exc:  # artefato de versão incompatível, etc.
            raise RuntimeError(
                f"Falha ao carregar {origem} ({exc}). Isso costuma acontecer "
                f"quando o artefato foi gerado com outra versão do "
                f"scikit-learn. Retreine com:\n"
                f"  python src/pipeline/treinar_modelos_estruturais.py"
            ) from exc
