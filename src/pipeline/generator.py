"""
Núcleo do pipeline de geração de cartas.

Este módulo é o único lugar onde a LÓGICA de geração deve morar. Tanto o
notebook do Colab (notebooks/demo.ipynb) quanto uma eventual interface web
futura apenas IMPORTAM e CHAMAM `gerar_carta` — eles não reimplementam
nada. Isso garante que exista um único "motor" testável e que as próximas
etapas do projeto (NLTK, Word2Vec, scikit-learn, PyTorch, OpenCV) entrem
aqui dentro, sem mexer na parte de interface.

Status atual: PLACEHOLDER. As funções privadas (prefixo `_`) abaixo serão
substituídas, uma a uma, pelo pipeline real descrito no TODO de
`gerar_carta`.
"""

import unicodedata
from dataclasses import dataclass
from PIL import Image, ImageDraw

from .mecanicas import inferir_mecanica
from .color_pie import ajustar_mecanica_para_cor, TEMPLATES

# Mapeamento das 5 cores do Magic (WUBRG) para abreviação e uma cor de
# fundo aproximada, só para o placeholder visual por enquanto.
COLOR_PIE = {
    "Branco": {"abrev": "W", "hex": "#F8F6D8"},
    "Azul": {"abrev": "U", "hex": "#C1D7E9"},
    "Preto": {"abrev": "B", "hex": "#BAB1AB"},
    "Vermelho": {"abrev": "R", "hex": "#E49977"},
    "Verde": {"abrev": "G", "hex": "#A3C095"},
}


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
    # Campos de transparência: mostram COMO o sistema chegou no resultado.
    # Úteis para debug e para explicar o funcionamento (ex: "por que essa
    # carta azul deu bounce em vez de destroy?").
    mecanica_inferida: str = ""
    mecanica_final: str = ""
    origem_inferencia: str = ""


def gerar_carta(nome: str, cor: str) -> Carta:
    """Ponto de entrada único do pipeline.

    Recebe o nome e a cor escolhidos pelo usuário e devolve um objeto
    `Carta` pronto para ser exibido.

    TODO (cada etapa substitui uma das funções `_placeholder` abaixo):
        1. [FEITO] nome -> tokenização + WordNet -> mecânica mais provável
           (ex: "murder" -> "destroy") -- ver pipeline.mecanicas
        2. [FEITO] mecânica + cor -> validação contra o color pie (bloqueia
           combinações ilegais, ex: Azul + "destroy" vira "bounce") --
           ver pipeline.color_pie
        3. [PRÓXIMO] camada de embeddings (Word2Vec treinado no corpus da
           Scryfall) como segunda opinião / desempate da camada WordNet
        4. texto gerado -> regressão (scikit-learn, treinada nas cartas
           reais da Scryfall) -> custo de mana
        5. texto gerado -> classificador PyTorch -> checagem de coerência
           com a cor pedida (gera de novo se a confiança for baixa)
        6. nome + tipo + texto -> prompt -> API de imagem -> arte
        7. arte + textos -> composição final (OpenCV) -> imagem da carta
    """
    if cor not in COLOR_PIE:
        raise ValueError(f"Cor inválida: {cor!r}. Use uma de {list(COLOR_PIE)}.")

    nome = nome.strip() or "Carta Sem Nome"

    mecanica_inferida, origem = inferir_mecanica(nome)
    mecanica_final = ajustar_mecanica_para_cor(mecanica_inferida, cor)
    texto_regras = TEMPLATES[mecanica_final]

    custo_mana = _custo_placeholder(nome, cor)
    imagem = _imagem_placeholder(nome, cor, custo_mana, texto_regras)

    return Carta(
        nome=nome,
        cor=cor,
        custo_mana=custo_mana,
        tipo="Criatura — Placeholder",
        texto_regras=texto_regras,
        imagem_final=imagem,
        mecanica_inferida=mecanica_inferida,
        mecanica_final=mecanica_final,
        origem_inferencia=origem,
    )


def _custo_placeholder(nome: str, cor: str) -> str:
    """Será substituído pela regressão treinada em scikit-learn
    (etapa 4 do TODO acima). Por enquanto, só uma aproximação boba a
    partir do tamanho do nome, para a interface já ter algo para mostrar."""
    abrev = COLOR_PIE[cor]["abrev"]
    tamanho = max(1, len(nome) % 4)
    return f"{tamanho}{abrev}"


def _imagem_placeholder(nome: str, cor: str, custo_mana: str, texto_regras: str) -> Image.Image:
    """Será substituído pela composição final com OpenCV, usando a arte
    gerada pela API de imagem (etapas 6-7 do TODO acima). Por enquanto,
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
    draw.text((largura - 60, 25), custo_mana, fill="black")
    draw.rectangle([30, 420, largura - 30, altura - 60], fill="white", outline="black")
    draw.multiline_text((40, 435), _quebrar_linhas(texto_img, 55), fill="black")

    return img


def _remover_acentos(texto: str) -> str:
    """Normaliza para ASCII só para desenho na imagem placeholder (ver
    comentário acima); o texto "de verdade" retornado em Carta.texto_regras
    continua com acentuação normal."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _quebrar_linhas(texto: str, largura_max: int) -> str:
    """Util simples para não estourar a caixa de texto do placeholder."""
    palavras = texto.split()
    linhas, linha_atual = [], ""
    for palavra in palavras:
        if len(linha_atual) + len(palavra) + 1 <= largura_max:
            linha_atual = f"{linha_atual} {palavra}".strip()
        else:
            linhas.append(linha_atual)
            linha_atual = palavra
    linhas.append(linha_atual)
    return "\n".join(linhas)
