"""
Núcleo do pipeline de geração de cartas.

Este módulo é o único lugar onde a LÓGICA de geração deve morar. Tanto o
notebook do Colab (notebooks/demo.ipynb) quanto uma eventual interface web
futura apenas IMPORTAM e CHAMAM `gerar_carta` — eles não reimplementam
nada. Isso garante que exista um único "motor" testável e que as próximas
etapas do projeto (scikit-learn, PyTorch, OpenCV) entrem aqui dentro, sem
mexer na parte de interface.

Decisão de design: o texto de regras é gerado por uma cadeia de Markov
treinada por cor (ver pipeline.texto_markov), SEM filtro de color pie.
A primeira versão deste projeto usava inferência de mecânica + regras
rígidas de color pie (pipeline.mecanicas / pipeline.color_pie, ainda no
repositório como registro do processo) -- trocamos por geração
estatística "caótica" de propósito: o peso de cada cor na geração é
ajustável via `peso_cor`, mas nada impede, com baixa probabilidade, que
saia uma combinação fora do color pie oficial. Essa troca é documentada
no relatório do projeto.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image, ImageDraw

from .estrutura import ModeloEstrutural
from .texto_markov import carregar_modelo

# Mapeamento das 5 cores do Magic (WUBRG) para abreviação e uma cor de
# fundo aproximada, só para o placeholder visual por enquanto.
COLOR_PIE = {
    "Branco": {"abrev": "W", "hex": "#F8F6D8"},
    "Azul": {"abrev": "U", "hex": "#C1D7E9"},
    "Preto": {"abrev": "B", "hex": "#BAB1AB"},
    "Vermelho": {"abrev": "R", "hex": "#E49977"},
    "Verde": {"abrev": "G", "hex": "#A3C095"},
}

# Peso padrão do modelo da cor na mistura cor/global (ver
# pipeline.texto_markov.ModeloMarkovPorCor). Exposto aqui porque é um
# parâmetro que faz sentido o notebook poder sobrescrever.
PESO_COR_PADRAO = 0.7

# Carregamento "preguiçoso": o modelo só é lido do disco na primeira
# chamada de gerar_carta, e fica em cache depois disso -- evita o custo
# de carregar o pickle se o módulo for só importado sem ser usado.
_modelo_cache = None

# Mesma lógica para os modelos de regressão (custo de mana + P/T, ver
# pipeline.estrutura).
_modelos_estruturais_cache: Optional[ModeloEstrutural] = None


def _obter_modelo():
    global _modelo_cache
    if _modelo_cache is None:
        _modelo_cache = carregar_modelo()
    return _modelo_cache


def _obter_modelos_estruturais() -> ModeloEstrutural:
    global _modelos_estruturais_cache
    if _modelos_estruturais_cache is None:
        _modelos_estruturais_cache = ModeloEstrutural.carregar()
    return _modelos_estruturais_cache


@dataclass
class Carta:
    """Representa o resultado final de uma geração: todos os campos que a
    interface (Colab, web, o que for) precisa para exibir a carta."""
    nome: str
    cor: str
    custo_mana: str
    tipo: str
    texto_regras: str
    imagem_final: Image.Image
    # Poder e resistência da criatura (previstos pela regressão do
    # pipeline.estrutura). Nulos se a carta não for criatura.
    poder: Optional[int] = None
    resistencia: Optional[int] = None
    # Campos de transparência: mostram COMO o sistema chegou no resultado.
    peso_cor: float = 0.0
    semente_usada: str = ""
    palavras_chave: List[str] = field(default_factory=list)


def gerar_carta(nome: str, cor: str, peso_cor: float = PESO_COR_PADRAO) -> Carta:
    """Ponto de entrada único do pipeline.

    Recebe o nome e a cor escolhidos pelo usuário (e, opcionalmente, o
    peso da cor na geração) e devolve um objeto `Carta` pronto para ser
    exibido.

    TODO (cada etapa substitui uma das funções `_placeholder` abaixo):
        1. [FEITO] cadeia de Markov (bigramas/trigramas) por cor, com
           peso ajustável cor/global -- ver pipeline.texto_markov
        2. [FEITO] texto gerado -> regressão (scikit-learn, treinada nas
           cartas reais da Scryfall) -> custo de mana + poder/resistência
           -- ver pipeline.estrutura
        3. texto gerado -> classificador PyTorch -> mede o quão "on-color"
           o texto gerado ficou (métrica/diagnóstico, não filtro)
        4. nome + tipo + texto -> prompt -> API de imagem -> arte
        5. arte + textos -> composição final (OpenCV) -> imagem da carta
    """
    if cor not in COLOR_PIE:
        raise ValueError(f"Cor inválida: {cor!r}. Use uma de {list(COLOR_PIE)}.")

    nome = nome.strip() or "Carta Sem Nome"
    cor_abrev = COLOR_PIE[cor]["abrev"]

    modelo = _obter_modelo()
    resultado = modelo.gerar_texto(cor_abrev, peso_cor=peso_cor, nome=nome)
    texto_regras = resultado["texto"]

    # Regressão (scikit-learn): o custo de mana sai do texto, e o
    # poder/resistência sai do texto + custo previsto.
    modelos_estruturais = _obter_modelos_estruturais()
    custo_mana, cmc_previsto = modelos_estruturais.prever_custo_mana(
        texto_regras, cor_abrev
    )
    poder, resistencia = modelos_estruturais.prever_poder_resistencia(
        texto_regras, cor_abrev, cmc_previsto
    )

    imagem = _imagem_placeholder(nome, cor, custo_mana, texto_regras, poder, resistencia)

    return Carta(
        nome=nome,
        cor=cor,
        custo_mana=custo_mana,
        tipo="Criatura — Placeholder",
        texto_regras=texto_regras,
        imagem_final=imagem,
        poder=poder,
        resistencia=resistencia,
        peso_cor=peso_cor,
        semente_usada=resultado["semente"] or "",
        palavras_chave=resultado["palavras_chave"],
    )


def _imagem_placeholder(
    nome: str,
    cor: str,
    custo_mana: str,
    texto_regras: str,
    poder: Optional[int] = None,
    resistencia: Optional[int] = None,
) -> Image.Image:
    """Será substituído pela composição final com OpenCV, usando a arte
    gerada pela API de imagem (etapas 4-5 do TODO acima). Por enquanto,
    desenha um retângulo simples só para a interface ter uma imagem real
    para exibir."""
    largura, altura = 480, 680
    img = Image.new("RGB", (largura, altura), color=COLOR_PIE[cor]["hex"])
    draw = ImageDraw.Draw(img)

    # A fonte bitmap padrão do PIL não renderiza acentos corretamente;
    # como isso é só o placeholder visual (será substituído por uma
    # composição de verdade via OpenCV + fonte própria na etapa final),
    # removemos os acentos apenas para o texto desenhado na imagem.
    nome_img = _remover_acentos(nome)
    texto_img = _remover_acentos(texto_regras)

    draw.rectangle([10, 10, largura - 10, altura - 10], outline="black", width=4)
    draw.text((30, 25), nome_img, fill="black")
    # Custo de mana alinhado à direita (o custo pode ser longo tipo
    # "{2}{W}{W}", herdo do formato do Oracle).
    largura_custo = draw.textlength(custo_mana)
    draw.text((largura - 30 - largura_custo, 25), custo_mana, fill="black")
    draw.rectangle([30, 420, largura - 30, altura - 60], fill="white", outline="black")
    draw.multiline_text((40, 435), _quebrar_linhas(texto_img, 55), fill="black")
    # Poder/resistência no rodapé, como nas cartas reais (só se houver).
    if poder is not None and resistencia is not None:
        draw.text((largura - 75, altura - 45), f"{poder}/{resistencia}", fill="black")

    return img


def _remover_acentos(texto: str) -> str:
    """Normaliza para ASCII só para desenho na imagem placeholder (ver
    comentário acima); o texto "de verdade" retornado em Carta.texto_regras
    continua com acentuação normal."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _quebrar_linhas(texto: str, largura_max: int) -> str:
    """Util simples para não estourar a caixa de texto do placeholder.

    Importante: respeita as quebras de linha que já vêm em `texto` (ex: a
    linha de keywords separada do corpo, ver pipeline.texto_markov) --
    cada linha original é quebrada/reflow independentemente, nunca
    juntada com a linha seguinte. Um `texto.split()` ingênuo trataria
    "\n" como só mais um espaço e desfaria essa separação visual."""
    linhas_finais = []
    for linha_original in texto.split("\n"):
        palavras = linha_original.split()
        if not palavras:
            linhas_finais.append("")
            continue
        linha_atual = ""
        for palavra in palavras:
            if len(linha_atual) + len(palavra) + 1 <= largura_max:
                linha_atual = f"{linha_atual} {palavra}".strip()
            else:
                linhas_finais.append(linha_atual)
                linha_atual = palavra
        linhas_finais.append(linha_atual)
    return "\n".join(linhas_finais)
