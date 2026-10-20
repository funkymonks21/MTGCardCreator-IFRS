"""
Núcleo do pipeline de geração de cartas.

Este módulo é o único lugar onde a LÓGICA de geração deve morar. Tanto o
notebook do Colab (notebooks/demo.ipynb) quanto uma eventual interface web
futura apenas IMPORTAM e CHAMAM `gerar_carta` — eles não reimplementam
nada. Isso garante que exista um único "motor" testável e que as próximas
etapas do projeto (scikit-learn, PyTorch, OpenCV) entrem aqui dentro, sem
mexer na parte de interface.

Decisão de design: o texto de regras é gerado por uma cadeia de Markov
treinada por TIPO e por cor (ver pipeline.texto_markov), SEM filtro de
color pie. O pipeline é organizado em etapas explícitas:

    1. decide o TIPO da carta (Criatura, Instantânea, Feitiço, Artefato,
       Encantamento ou Terreno) -- escolhido pelo usuário, inferido a
       partir do nome (ver pipeline.tipo_por_nome) ou sorteado;
    2. gera as HABILIDADES com o modelo de texto treinado só naquele tipo
       (instants/sorceries não geram habilidades ativadas);
    3. prevê CUSTO DE MANA (regressão com o tipo como feature) e, apenas
       para criaturas, PODER/RESISTÊNCIA. Terrenos não têm custo nem P/T.

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
from .tipo_por_nome import inferir_tipo_por_nome
from .tipos import TIPOS_COM_PODER_RESISTENCIA, TIPOS_DISPLAY, TIPOS_VALIDOS

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

# Opções de tipo expostas para a interface.
#
# "Automático" tenta inferir o tipo pelo nome (ver
# pipeline.tipo_por_nome.inferir_tipo_por_nome) e, se não houver sinal
# suficiente ou houver empate, sorteia. "Aleatório" pula a inferência e
# sorteia direto, ponderado pela frequência real no corpus (ver
# ModeloTextoPorTipo.sortear_tipo).
TIPO_AUTOMATICO = "Automático (pelo nome)"
TIPO_ALEATORIO = "Aleatório"

# Sentinelas internas devolvidas por _resolver_tipo (e usadas como valores
# de TIPOS_DISPONIVEIS para as duas opções especiais).
_TIPO_AUTO = "__auto__"
_TIPO_SORTEAR = "__aleatorio__"

TIPOS_DISPONIVEIS = {
    TIPO_AUTOMATICO: _TIPO_AUTO,
    TIPO_ALEATORIO: _TIPO_SORTEAR,
    **{TIPOS_DISPLAY[tipo]: tipo for tipo in TIPOS_VALIDOS},
}

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
    # Como o tipo foi decidido: "escolhido", "nome" (inferência) ou
    # "sorteio". `tipo_motivo` explica a pista usada na inferência.
    tipo_origem: str = "escolhido"
    tipo_motivo: str = ""


def _resolver_tipo(tipo: Optional[str]) -> str:
    """Converte a opção escolhida na interface para a chave interna usada
    pelos modelos (ex: "Instantânea" -> "instant"). Aceita também a própria
    chave interna.

    None/vazio/"Automático" devolvem a sentinela `_TIPO_AUTO` (o pipeline
    tenta inferir pelo nome e, se não conseguir, sorteia). "Aleatório"
    devolve `_TIPO_SORTEAR` (sorteia sem tentar inferir).

    A comparação ignora acentos e maiúsculas, então "Instantanea" e
    "instantanea" também funcionam."""
    if tipo is None:
        return _TIPO_AUTO
    if tipo in {_TIPO_AUTO, _TIPO_SORTEAR}:
        # Aceita também os valores de TIPOS_DISPONIVEIS, para quem passar
        # TIPOS_DISPONIVEIS[opcao] em vez do texto da opção.
        return tipo

    normalizado = _remover_acentos(str(tipo)).strip().casefold()
    if normalizado in {"", "auto", "automatico", _remover_acentos(TIPO_AUTOMATICO).casefold()}:
        return _TIPO_AUTO
    if normalizado in {"aleatorio", "random", _remover_acentos(TIPO_ALEATORIO).casefold()}:
        return _TIPO_SORTEAR

    for chave in TIPOS_VALIDOS:
        if normalizado == chave.casefold():
            return chave
        if normalizado == _remover_acentos(TIPOS_DISPLAY[chave]).casefold():
            return chave

    opcoes = ", ".join([TIPO_AUTOMATICO, TIPO_ALEATORIO, *TIPOS_DISPLAY.values()])
    raise ValueError(f"Tipo inválido: {tipo!r}. Use um de: {opcoes}.")


def gerar_carta(
    nome: str,
    cor: str,
    peso_cor: float = PESO_COR_PADRAO,
    tipo: Optional[str] = None,
) -> Carta:
    """Ponto de entrada único do pipeline.

    Recebe o nome, a cor e (opcionalmente) o tipo e o peso da cor na
    geração, e devolve um objeto `Carta` pronto para ser exibido.

    Se `tipo` for None/vazio/"Automático", o pipeline tenta inferir o tipo
    pelo nome (ver pipeline.tipo_por_nome) e, se não houver sinal ou
    houver empate, sorteia. Se for "Aleatório", sorteia direto.

    Ordem das etapas (importante: cada uma depende da anterior):
        1. [FEITO] decide o TIPO (Criatura, Instantânea, Feitiço, Artefato,
           Encantamento ou Terreno) -- escolhido, inferido pelo nome ou
           sorteado;
        2. [FEITO] gera as HABILIDADES com a cadeia de Markov treinada só
           naquele tipo (bigramas/trigramas por tipo e cor) -- ver
           pipeline.texto_markov. Instants/sorceries não têm habilidades
           ativadas;
        3. [FEITO] texto gerado -> regressão (scikit-learn, treinada nas
           cartas reais da Scryfall) -> custo de mana; poder/resistência
           apenas para criaturas; terrenos não têm custo nem P/T -- ver
           pipeline.estrutura;
        4. texto gerado -> classificador PyTorch -> mede o quão "on-color"
           o texto gerado ficou (métrica/diagnóstico, não filtro);
        5. nome + tipo + texto -> prompt -> API de imagem -> arte;
        6. arte + textos -> composição final (OpenCV) -> imagem da carta.
    """
    if cor not in COLOR_PIE:
        raise ValueError(f"Cor inválida: {cor!r}. Use uma de {list(COLOR_PIE)}.")

    nome = nome.strip() or "Carta Sem Nome"
    cor_abrev = COLOR_PIE[cor]["abrev"]
    decisao = _resolver_tipo(tipo)

    modelo = _obter_modelo()

    # Etapa 1: decide o tipo. A ordem de preferência é: escolha explícita
    # do usuário > inferência pelo nome > sorteio ponderado pela frequência
    # real no corpus.
    if decisao == _TIPO_SORTEAR:
        tipo_chave = modelo.sortear_tipo()
        tipo_origem = "sorteio"
        tipo_motivo = "tipo sorteado (opção Aleatório)"
    elif decisao == _TIPO_AUTO:
        inferencia = inferir_tipo_por_nome(nome)
        if inferencia.tipo is None:
            tipo_chave = modelo.sortear_tipo()
            tipo_origem = "sorteio"
            tipo_motivo = inferencia.motivo
        else:
            tipo_chave = inferencia.tipo
            tipo_origem = "nome"
            tipo_motivo = inferencia.motivo
    else:
        tipo_chave = decisao
        tipo_origem = "escolhido"
        tipo_motivo = "tipo escolhido manualmente"

    tipo_display = TIPOS_DISPLAY[tipo_chave]

    # Etapa 2: habilidades condicionadas ao tipo escolhido.
    resultado = modelo.gerar_texto(
        cor_abrev, tipo=tipo_chave, peso_cor=peso_cor, nome=nome
    )
    texto_regras = resultado["texto"]

    # Etapa 3: custo de mana (o tipo entra como feature) e, só para
    # criaturas, poder/resistência (ver pipeline.tipos).
    modelos_estruturais = _obter_modelos_estruturais()
    custo_mana, cmc_previsto = modelos_estruturais.prever_custo_mana(
        texto_regras, cor_abrev, tipo_chave
    )
    poder: Optional[int] = None
    resistencia: Optional[int] = None
    if tipo_chave in TIPOS_COM_PODER_RESISTENCIA:
        poder, resistencia = modelos_estruturais.prever_poder_resistencia(
            texto_regras, cor_abrev, cmc_previsto
        )

    imagem = _imagem_placeholder(
        nome, cor, tipo_display, custo_mana, texto_regras, poder, resistencia
    )

    return Carta(
        nome=nome,
        cor=cor,
        custo_mana=custo_mana,
        tipo=tipo_display,
        texto_regras=texto_regras,
        imagem_final=imagem,
        poder=poder,
        resistencia=resistencia,
        peso_cor=peso_cor,
        semente_usada=resultado["semente"] or "",
        palavras_chave=resultado["palavras_chave"],
        tipo_origem=tipo_origem,
        tipo_motivo=tipo_motivo,
    )


def _imagem_placeholder(
    nome: str,
    cor: str,
    tipo: str,
    custo_mana: str,
    texto_regras: str,
    poder: Optional[int] = None,
    resistencia: Optional[int] = None,
) -> Image.Image:
    """Será substituído pela composição final com OpenCV, usando a arte
    gerada pela API de imagem (etapas 5-6 do TODO acima). Por enquanto,
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
    tipo_img = _remover_acentos(tipo)
    texto_img = _remover_acentos(texto_regras)

    draw.rectangle([10, 10, largura - 10, altura - 10], outline="black", width=4)
    draw.text((30, 25), nome_img, fill="black")
    # Linha de tipo logo abaixo do nome (ex: "Criatura" / "Instantânea").
    draw.text((30, 45), tipo_img, fill="black")
    # Custo de mana alinhado à direita (o custo pode ser longo tipo
    # "{2}{W}{W}", herdo do formato do Oracle). Terrenos não têm custo --
    # nesse caso o campo vem vazio e simplesmente não desenhamos nada.
    if custo_mana:
        largura_custo = draw.textlength(custo_mana)
        draw.text((largura - 30 - largura_custo, 25), custo_mana, fill="black")
    draw.rectangle([30, 420, largura - 30, altura - 60], fill="white", outline="black")
    draw.multiline_text((40, 435), _quebrar_linhas(texto_img, 55), fill="black")
    # Poder/resistência no rodapé, como nas cartas reais (só se houver --
    # instants, sorceries, artefatos, encantamentos não-criatura e terrenos
    # não têm).
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
