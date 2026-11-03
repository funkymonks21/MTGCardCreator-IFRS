"""
Arte e composição visual da carta: prompt -> API de imagem -> moldura
("frame") realista -> imagem final.

Este módulo implementa as etapas 7 e 8 prometidas no docstring de
`generator.py`:

    7. nome + tipo + texto -> PROMPT -> API de imagem (arte);
    8. arte + textos -> COMPOSIÇÃO final -> imagem da carta.

Decisões de design
==================
- **API de imagem: Pollinations.ai.** É o único serviço realmente gratuito
  e SEM chave/token: a arte é obtida montando a URL
  `https://image.pollinations.ai/prompt/{prompt}` (com `width`, `height`,
  `seed`, `nologo`, `model`) e baixando o PNG/JPEG resultante. Funciona
  direto no Colab, sem cadastro. A interface é isolada em
  `ArtePollinations`, então trocar de provedor (Hugging Face, Stability...)
  é implementar a mesma assinatura `gerar(prompt, ...) -> Image.Image`.
- **Fallback local.** Se a API falhar (sem internet, timeout, serviço fora
  do ar), `gerar_arte` cai num degradê procedural determinístico derivado
  do nome da carta -- a geração NUNCA quebra por causa da arte.
- **Moldura em código.** O frame é desenhado com Pillow (bordas, painéis,
  gradiente da cor, caixa de texto translúcida, selo de P/T, rodapé). Não
  usamos imagens/templates oficiais da Wizards (material protegido): o
  visual é uma releitura vetorial própria.
- **Símbolos desenhados.** Os símbolos de custo e de habilidade também são
  desenhados em código (`pipeline.simbolos`), em vez de baixar os SVGs
  oficiais. Assim o app é distribuível sem carregar marca de terceiros.
- **Fontes livres.** Título em Cinzel e corpo em EB Garamond, ambas SIL OFL
  (uso livre), versionadas em `assets/fonts/`. A Beleren/Matrix oficial é
  proprietária e não é distribuída aqui. Se as fontes não estiverem
  presentes, cai na fonte padrão do Pillow sem quebrar.

Uso
===
Normalmente não se chama este módulo direto: `generator.gerar_carta(...,
usar_arte=True)` monta a carta inteira. As funções ficam expostas para
teste e para quem quiser reusar só a arte ou só o frame:

    from pipeline.imagem import gerar_arte, compor_frame

    arte = gerar_arte("Serra Angel", "Creature — Angel", "Flying, vigilance.")
    carta = compor_frame(arte, nome=..., tipo_linha=..., texto_regras=..., ...)
"""

from __future__ import annotations

import hashlib
import io
import math
import random
import time
import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .simbolos import desenhar_custo, desenhar_simbolo, dividir

# ---------------------------------------------------------------------------
# Constantes de layout. A carta final tem a proporção "real" do Magic
# (~63x88 mm, ou seja 2.5 x 3.5 polegadas). Usamos 630x880 px (10x) para
# ter resolução confortável de impressão/tela.
# ---------------------------------------------------------------------------
LARGURA_PADRAO = 630
ALTURA_PADRAO = 880
# Raio dos cantos arredondados (cartas reais têm cantos levemente
# arredondados).
RAIO_CANTO = 26

# Fração da altura ocupada pela arte (janela da ilustração), medida da
# borda superior do frame até a caixa de tipo. Números aproximados das
# cartas modernas.
MARGEM_EXTERNA = 18          # borda preta externa
MARGEM_INTERNA = 12          # respiro entre a borda e os painéis

# Altura do painel do título (nome + custo). A janela de arte começa logo
# abaixo dele.
ALTURA_TITULO = 62

# ---------------------------------------------------------------------------
# Geometria da JANELA DE ARTE. Estes números são a ÚNICA fonte de verdade
# sobre onde a ilustração cai na carta: tanto `compor_frame` (que recorta e
# cola a arte) quanto `gerar_arte` (que PEDE a imagem à API já nesta
# proporção) usam `tamanho_janela_arte()`. Antes, `gerar_arte` pedia uma
# imagem 630x880 (retrato) e o frame recortava para ~594x372 (paisagem) --
# o `ImageOps.fit` jogava fora mais da metade da imagem, e era isso que
# fazia a arte parecer "cortada" dentro da moldura.
# ---------------------------------------------------------------------------
FRACAO_ALTURA_ARTE = 0.50    # fração da altura total reservada à arte


def tamanho_janela_arte(
    largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO
) -> Tuple[int, int]:
    """Devolve (largura, altura) em pixels da JANELA DE ARTE -- a área útil
    onde a ilustração é exibida, já descontando as margens e o painel do
    título. É a proporção que a API de imagem deve receber, para a arte não
    ser esticada/cortada na composição."""
    largura_util = largura - 2 * MARGEM_EXTERNA
    y_arte0 = MARGEM_EXTERNA + ALTURA_TITULO + 6
    y_arte1 = MARGEM_EXTERNA + int(altura * FRACAO_ALTURA_ARTE)
    return largura_util, max(1, y_arte1 - y_arte0)


def _geometria(
    largura: int = LARGURA_PADRAO, altura: int = ALTURA_PADRAO
) -> Dict[str, int]:
    """Mesma geometria vertical usada por `compor_frame`, isolada para que
    composição e cálculo da janela de arte nunca fiquem em desacordo."""
    largura_util = largura - 2 * MARGEM_EXTERNA
    y_titulo0 = MARGEM_EXTERNA
    y_titulo1 = y_titulo0 + ALTURA_TITULO
    y_arte0 = y_titulo1 + 6
    y_arte1 = MARGEM_EXTERNA + int(altura * FRACAO_ALTURA_ARTE)
    y_tipo0 = y_arte1 + 8
    y_tipo1 = y_tipo0 + 44
    y_texto0 = y_tipo1 + 10
    y_texto1 = altura - MARGEM_EXTERNA - 46
    return {
        "largura_util": largura_util,
        "y_titulo0": y_titulo0,
        "y_titulo1": y_titulo1,
        "y_arte0": y_arte0,
        "y_arte1": y_arte1,
        "altura_arte": y_arte1 - y_arte0,
        "y_tipo0": y_tipo0,
        "y_tipo1": y_tipo1,
        "y_texto0": y_texto0,
        "y_texto1": y_texto1,
    }

FONTE_TITULO = "assets/fonts/Cinzel.ttf"
FONTE_TEXTO = "assets/fonts/EBGaramond.ttf"

# Cores-base por cor do Magic (WUBRG). São tons de partida para o
# gradiente do painel; ajustados para ficarem legíveis com texto escuro.
CORES_FRAME: Dict[str, Dict[str, str]] = {
    "W": {"claro": "#F3EED0", "escuro": "#D8CFA6", "texto": "#1A1A14"},
    "U": {"claro": "#BFD9EC", "escuro": "#7FA8C9", "texto": "#0F1B26"},
    "B": {"claro": "#B9B0AA", "escuro": "#6E6560", "texto": "#141110"},
    "R": {"claro": "#EEB08B", "escuro": "#C9724A", "texto": "#26120A"},
    "G": {"claro": "#B9D2A9", "escuro": "#7FA36A", "texto": "#12200C"},
    # Incolor (artefatos/terrenos sem cor) -- cinza pedra.
    "C": {"claro": "#D8D5CE", "escuro": "#A9A49A", "texto": "#1A1917"},
}

# Cache em disco das artes baixadas: evita repetir a chamada para o mesmo
# prompt (nome + tipo + texto) e deixa o notebook responsivo ao recalcular
# a mesma carta. Fica fora do git (ver .gitignore).
CACHE_DIR = Path("assets/cache_arte")

# Timeout padrão da chamada à API de imagem (segundos). Pollinations pode
# demorar bastante na primeira chamada de um prompt novo.
TIMEOUT_API = 120


# ===========================================================================
# 1. PROMPT
# ===========================================================================

# Descrições de estilo por TIPO de carta. Servem de "direção de arte" para a
# API; mantemos tudo em inglês (os modelos de imagem respondem melhor) e
# sempre pedimos arte de alta fantasia, sem texto, em enquadramento de carta.
_ESTILO_POR_TIPO: Dict[str, str] = {
    "creature": "a majestic fantasy creature, dynamic pose, detailed anatomy",
    "instant": "a dramatic magical effect bursting in an instant, energy and motion",
    "sorcery": "a powerful ritual of magic unfolding, sweeping arcane energy",
    "artifact": "an ancient magical artifact, intricate craftsmanship, mysterious glow",
    "enchantment": "an ethereal aura of enchantment, glowing runes and light",
    "land": "a sweeping fantasy landscape, epic vista, atmospheric depth",
}

# Direção de arte por cor do Magic. Ajuda a arte a "combinar" com a carta.
_ESTILO_POR_COR: Dict[str, str] = {
    "W": "holy radiant light, white and gold palette, order and purity",
    "U": "blue and teal palette, mist and water, knowledge and illusion",
    "B": "dark palette, black and violet, decay and shadow, ominous",
    "R": "fiery red and orange palette, lava and flame, chaos and passion",
    "G": "lush green palette, forest and growth, primal nature",
    "C": "neutral stone and metal palette, colorless, ancient",
}

# Sufixo fixo de qualidade/estilo aplicado a todo prompt.
_SUFIXO_ESTILO = (
    "digital painting, fantasy trading card art, highly detailed, "
    "dramatic lighting, rich colors, centered composition, no text, "
    "no watermark, no border"
)


def montar_prompt(
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
) -> str:
    """Monta o prompt de imagem a partir do NOME, TIPO e TEXTO da carta.

    - `nome` é a semente temática principal ("Serra Angel" -> anjo);
    - o TIPO define a categoria de cena (criatura, terreno, artefato...);
    - a COR dá a paleta (a chave deve ser "W"/"U"/"B"/"R"/"G"/"C");
    - o TEXTO entra reduzido a umas poucas palavras-chave, para não poluir
      o prompt com frases do Oracle (que confundem o modelo de imagem).

    O texto do prompt é sempre em inglês e termina com um sufixo fixo de
    estilo para padronizar a "cara" das artes.
    """
    chave_tipo = _chave_tipo(tipo)
    estilo_tipo = _ESTILO_POR_TIPO.get(chave_tipo, _ESTILO_POR_TIPO["creature"])
    estilo_cor = _ESTILO_POR_COR.get(cor, _ESTILO_POR_COR["C"])

    # Frases de habilidade só ajudam se forem curtas e concretas. Extraímos
    # no máximo 3 palavras significativas, ignorando símbolos de mana e
    # termos genéricos que confundem o modelo.
    detalhe = _palavras_tematicas(texto_regras)

    partes = [f"fantasy art of {nome}", estilo_tipo]
    if detalhe:
        partes.append(detalhe)
    partes.extend([estilo_cor, _SUFIXO_ESTILO])
    return ", ".join(p for p in partes if p)


def _chave_tipo(tipo: str) -> str:
    """Aceita tanto a chave interna ("creature") quanto a linha de tipo
    completa ("Creature — Human Warrior") e devolve a categoria usada em
    `_ESTILO_POR_TIPO`. Cai em "creature" para o que não reconhecer."""
    t = (tipo or "").lower()
    if "creature" in t or "criatura" in t:
        return "creature"
    if "land" in t or "terreno" in t:
        return "land"
    if "artifact" in t or "artefato" in t:
        return "artifact"
    if "enchantment" in t or "encantamento" in t:
        return "enchantment"
    if "instant" in t or "instant" in t:
        return "instant"
    if "sorcery" in t or "feiti" in t:
        return "sorcery"
    return "creature"


# Palavras que só atrapalham o prompt de imagem (jargão de regras).
_STOPWORDS_TEMA = {
    "target", "creature", "player", "opponent", "control", "card", "cards",
    "spell", "spells", "turn", "end", "until", "each", "your", "you", "this",
    "that", "with", "from", "into", "then", "when", "whenever", "if", "add",
    "gets", "gain", "gains", "draw", "damage", "counter", "mana", "value",
    "instead", "other", "another", "up", "to", "the", "a", "an", "and", "or",
    "of", "for", "on", "in", "it", "its", "may", "you", "control",
}


def _palavras_tematicas(texto_regras: str, limite: int = 3) -> str:
    """Extrai até `limite` palavras "concretas" do texto de regras, para
    dar um toque temático à arte sem transformar o prompt num amontoado de
    frases de regra. Ignora símbolos de mana ({t}, {2}...), palavras
    curtas e o jargão de regras (_STOPWORDS_TEMA)."""
    if not texto_regras:
        return ""
    palavras: List[str] = []
    for bruto in texto_regras.replace("\n", " ").split():
        p = bruto.strip(".,:;()[]{}\"'").lower()
        if not p or not p.isalpha() or len(p) < 4:
            continue
        if p in _STOPWORDS_TEMA:
            continue
        if p not in palavras:
            palavras.append(p)
        if len(palavras) >= limite:
            break
    return " ".join(palavras)


# ===========================================================================
# 2. ARTE (API de imagem)
# ===========================================================================

# Motivos de falha da API (usados para AVISAR o usuário em vez de cair no
# fallback silenciosamente -- foi o que fez o fundo liso aparecer sem
# explicação).
MOTIVO_LIMITE = "limite"
MOTIVO_REDE = "rede"
MOTIVO_SERVIDOR = "servidor"
MOTIVO_RESPOSTA = "resposta"

# Plano grátis da Pollinations: 1 imagem a cada ~45s por IP. Quando o
# servidor responde 402, ele manda um header `Retry-After` (segundos);
# esperamos esse tempo (limitado a um teto) e tentamos de novo.
_MAX_ESPERA_LIMITE = 50


@dataclass
class ResultadoArte:
    """Resultado da tentativa de gerar a arte: a imagem (sempre presente --
    cai no fallback se a API falhar), se veio da API ou do fallback, e o
    motivo quando falhou."""

    imagem: Image.Image
    da_api: bool
    motivo: Optional[str] = None
    detalhe: str = ""


def gerar_arte_detalhada(
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
    *,
    seed: Optional[int] = None,
    usar_cache: bool = True,
    timeout: int = TIMEOUT_API,
    tentar_limite: bool = True,
    avisar=print,
) -> ResultadoArte:
    """Como `gerar_arte`, mas devolve um `ResultadoArte` dizendo se a arte
    veio da API ou do fallback e, no segundo caso, o MOTIVO.

    É o que permite a interface avisar "a API está no limite, tente de novo
    em Ns" em vez de mostrar um fundo liso sem explicação. `avisar` é uma
    função de log (default `print`); passe `avisar=lambda *_: None` para
    silenciar.
    """
    prompt = montar_prompt(nome, tipo, texto_regras, cor)
    if seed is None:
        seed = _seed_do_prompt(prompt)

    # A arte é pedida DENTRO DA PROPORÇÃO DA JANELA (não da carta inteira):
    # assim o frame não precisa recortar nada e a ilustração não "corta".
    largura_arte, altura_arte = tamanho_janela_arte()

    if usar_cache:
        em_cache = _ler_cache(prompt, largura_arte, altura_arte, seed)
        if em_cache is not None:
            return ResultadoArte(em_cache, da_api=True, detalhe="cache")

    imagem, motivo, detalhe = _arte_via_api(
        prompt, largura_arte, altura_arte, seed, timeout,
        tentar_limite=tentar_limite, avisar=avisar,
    )
    if imagem is not None:
        if usar_cache:
            _salvar_cache(imagem, prompt, largura_arte, altura_arte, seed)
        return ResultadoArte(imagem, da_api=True)

    # Fallback: fundo procedural. Avisamos o motivo para o usuário entender
    # por que não veio arte de verdade.
    _avisar_motivo(motivo, detalhe, avisar)
    fallback = arte_fallback(
        nome, cor, largura_arte, altura_arte, seed
    )
    return ResultadoArte(fallback, da_api=False, motivo=motivo, detalhe=detalhe)


def _avisar_motivo(motivo: Optional[str], detalhe: str, avisar) -> None:
    """Emite a mensagem adequada ao motivo da falha, quando há um log
    (`avisar`) configurado."""
    if avisar is None:
        return
    if motivo == MOTIVO_LIMITE:
        avisar(
            f"[arte] A API de imagem atingiu o limite gratuito (1 imagem a "
            f"cada ~45s por IP). Usando arte provisória; tente de novo em "
            f"instantes. {detalhe}".strip()
        )
    elif motivo == MOTIVO_REDE:
        avisar(
            f"[arte] Sem conexão com a API de imagem ({detalhe}). Usando "
            f"arte provisória local."
        )
    elif motivo == MOTIVO_SERVIDOR:
        avisar(
            f"[arte] A API de imagem respondeu com erro ({detalhe}). Usando "
            f"arte provisória local."
        )
    else:
        avisar(
            f"[arte] Resposta inesperada da API de imagem ({detalhe}). "
            f"Usando arte provisória local."
        )


def gerar_arte(
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
    *,
    seed: Optional[int] = None,
    usar_cache: bool = True,
    timeout: int = TIMEOUT_API,
    avisar=print,
) -> Image.Image:
    """Gera a ILUSTRAÇÃO da carta (sem moldura) a partir do nome/tipo/texto.

    Tenta a API de imagem (Pollinations.ai) e, se ela falhar, cai num
    fallback procedural local -- nunca levanta exceção por causa de rede.
    A imagem devolvida já vem na PROPORÇÃO DA JANELA DE ARTE (a área onde a
    ilustração aparece na carta, ver `tamanho_janela_arte`), então a
    composição não precisa recortá-la.

    `seed` fixa a arte (mesma seed -> mesma imagem); se None, a seed é
    derivada do prompt, o que também a torna reprodutível para o mesmo
    nome/tipo/texto. `usar_cache` reaproveita o arquivo em
    `assets/cache_arte/` quando existir. Para saber se a arte veio da API
    ou do fallback, use `gerar_arte_detalhada`.
    """
    return gerar_arte_detalhada(
        nome, tipo, texto_regras, cor,
        seed=seed, usar_cache=usar_cache, timeout=timeout, avisar=avisar,
    ).imagem


def _arte_via_api(
    prompt: str,
    largura: int,
    altura: int,
    seed: int,
    timeout: int,
    tentar_limite: bool = True,
    avisar=print,
) -> Tuple[Optional[Image.Image], Optional[str], str]:
    """Baixa a arte da Pollinations.ai.

    Devolve `(imagem, motivo, detalhe)`. Em caso de sucesso a imagem vem
    preenchida e o motivo é None. Em QUALQUER falha a imagem é None, e o
    motivo diz o que houve (limite/rede/servidor/resposta) para o chamador
    decidir o que avisar.

    Quando o servidor responde 402 (limite do plano grátis), respeitamos o
    header `Retry-After` e tentamos de novo uma vez (se `tentar_limite`).
    """
    prompt_url = urllib.parse.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{prompt_url}"
    params = {
        "width": int(largura),
        "height": int(altura),
        "seed": int(seed),
        "nologo": "true",
    }
    try:
        resposta = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:  # rede, timeout, DNS...
        return None, MOTIVO_REDE, f"{type(exc).__name__}: {exc}"

    if resposta.status_code == 402:
        espera = _retry_apos(resposta)
        if tentar_limite and espera is not None and 0 <= espera <= _MAX_ESPERA_LIMITE:
            time.sleep(espera + 1)
            try:
                resposta = requests.get(url, params=params, timeout=timeout)
            except Exception as exc:
                return None, MOTIVO_REDE, f"{type(exc).__name__}: {exc}"
            # Se a nova tentativa passou do limite, seguimos o fluxo normal
            # (a resposta pode ser uma imagem válida). Só desistimos se
            # ainda estiver bloqueada.
            if resposta.status_code == 402:
                return None, MOTIVO_LIMITE, (
                    f"limite de {espera}s respeitado, ainda bloqueado"
                )
        else:
            return None, MOTIVO_LIMITE, resposta.text[:160]

    if resposta.status_code >= 500:
        return None, MOTIVO_SERVIDOR, f"HTTP {resposta.status_code}"

    try:
        resposta.raise_for_status()
    except Exception as exc:
        return None, MOTIVO_RESPOSTA, f"{type(exc).__name__}: {exc}"

    if not resposta.headers.get("content-type", "").startswith("image/"):
        return None, MOTIVO_RESPOSTA, f"content-type={resposta.headers.get('content-type')}"

    try:
        imagem = Image.open(io.BytesIO(resposta.content)).convert("RGB")
    except Exception as exc:
        return None, MOTIVO_RESPOSTA, f"imagem inválida: {exc}"

    # A API nem sempre respeita exatamente o tamanho pedido: ajustamos para
    # o tamanho da janela SEM distorcer (recorte centralizado). Como já
    # pedimos a proporção certa, o recorte é pequeno (só bordas).
    return ImageOps.fit(imagem, (largura, altura), Image.LANCZOS), None, ""


def _retry_apos(resposta) -> Optional[int]:
    """Lê o header `Retry-After` (segundos) da resposta, se houver."""
    valor = resposta.headers.get("Retry-After") or resposta.headers.get("retry-after")
    if not valor:
        return None
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def arte_fallback(
    nome: str,
    cor: str = "C",
    largura: Optional[int] = None,
    altura: Optional[int] = None,
    seed: int = 0,
) -> Image.Image:
    """Arte procedural determinística, usada quando a API não está
    disponível (ou quando o usuário pede a carta SEM arte).

    É um fundo abstrato "de mana": gradiente radial na cor da carta,
    manchas de luz desfocadas, uma marca d'água grande do símbolo da cor,
    vinheta e um grão leve. Não tenta imitar uma ilustração de verdade --
    assume o papel de arte provisória, mas com o mesmo acabamento da
    moldura (antes o caminho sem arte caía num retângulo liso e destoava).

    Por padrão gera já na proporção da JANELA DE ARTE (não da carta
    inteira), para ficar igual ao que a API devolveria. É determinística:
    mesma `seed` + mesmo `nome` -> mesma imagem.
    """
    if largura is None or altura is None:
        largura, altura = tamanho_janela_arte()
    base = CORES_FRAME.get(cor, CORES_FRAME["C"])
    rng = random.Random((seed or 0) ^ _hash_int(nome))
    escuro = _hex_para_rgb(base["escuro"])
    claro = _hex_para_rgb(base["claro"])

    # 1) Fundo: gradiente radial (claro no centro, escuro nas bordas).
    gradiente = Image.radial_gradient("L").resize((largura, altura), Image.LANCZOS)
    gradiente = ImageOps.invert(gradiente)
    img = ImageOps.colorize(gradiente, black=escuro, white=claro).convert("RGBA")

    # 2) Manchas de luz desfocadas, para dar profundidade.
    manchas = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    mdraw = ImageDraw.Draw(manchas)
    for _ in range(7):
        rx = rng.randint(max(8, largura // 9), max(12, largura // 3))
        ry = int(rx * rng.uniform(0.55, 1.15))
        cx = rng.randint(0, largura)
        cy = rng.randint(0, altura)
        if rng.random() < 0.3:
            tom = tuple(max(0, int(v * 0.65)) for v in escuro)
        else:
            tom = tuple(min(255, int(v * rng.uniform(1.15, 1.45))) for v in claro)
        mdraw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=tom + (rng.randint(22, 52),),
        )
    manchas = manchas.filter(ImageFilter.GaussianBlur(radius=max(6, largura // 16)))
    img = Image.alpha_composite(img, manchas)

    # 3) Raios de luz diagonais, bem suaves -- quebram a simetria do
    #    gradiente e dão a sensação de "cena" em vez de fundo de slide.
    raios = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(raios)
    origem = (largura * rng.uniform(0.30, 0.70), -altura * 0.08)
    for _ in range(rng.randint(4, 6)):
        ang = math.radians(rng.uniform(55, 125))
        comprimento = max(largura, altura) * 1.5
        meia_largura = rng.uniform(0.020, 0.055) * largura
        dx, dy = math.cos(ang) * comprimento, math.sin(ang) * comprimento
        px, py = -math.sin(ang) * meia_largura, math.cos(ang) * meia_largura
        rdraw.polygon(
            [
                origem,
                (origem[0] + dx + px, origem[1] + dy + py),
                (origem[0] + dx - px, origem[1] + dy - py),
            ],
            fill=(255, 255, 255, rng.randint(10, 24)),
        )
    raios = raios.filter(ImageFilter.GaussianBlur(radius=max(2, largura // 40)))
    img = Image.alpha_composite(img, raios)

    # 4) Marca d'água discreta do símbolo da cor: lembra que a arte é
    #    procedural, sem parecer uma ilustração "de verdade".
    chave_cor = cor if cor in CORES_FRAME else "C"
    lado_marca = int(min(largura, altura) * 0.62)
    marca = desenhar_simbolo(chave_cor, lado_marca).copy()
    marca.putalpha(marca.getchannel("A").point(lambda a: int(a * 0.18)))
    marca = marca.filter(ImageFilter.GaussianBlur(radius=1.5))
    img.alpha_composite(
        marca, ((largura - marca.width) // 2, (altura - marca.height) // 2)
    )

    # 5) Vinheta: escurece as bordas e puxa o olho para o centro.
    vinheta = Image.radial_gradient("L").resize((largura, altura), Image.LANCZOS)
    sombra = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    sombra.putalpha(vinheta.point(lambda v: int(v * 0.55)))
    img = Image.alpha_composite(img, sombra)

    # 6) Grão sutil: evita o aspecto de "degradê de PowerPoint".
    ruido = Image.effect_noise((largura, altura), 18).convert("L")
    grao = Image.merge(
        "RGBA", [ruido, ruido, ruido, Image.new("L", (largura, altura), 12)]
    )
    img = Image.alpha_composite(img, grao)
    return img.convert("RGB")


class ArtePollinations:
    """Cliente fino para a API de imagem da Pollinations.ai.

    Existe para deixar explícito o "contrato" de um provedor de arte:
    `gerar(prompt, largura, altura, seed) -> Optional[Image.Image]`.
    Para trocar de serviço (Hugging Face, Stability...), basta
    implementar a mesma assinatura e passar a instância para
    `gerar_arte_por_provedor`.

    Note que `largura`/`altura` são as da JANELA DE ARTE (ver
    `tamanho_janela_arte`), não da carta inteira.
    """

    URL = "https://image.pollinations.ai/prompt/"

    def __init__(self, timeout: int = TIMEOUT_API):
        self.timeout = timeout

    def gerar(
        self, prompt: str, largura: int, altura: int, seed: int
    ) -> Optional[Image.Image]:
        imagem, _motivo, _detalhe = _arte_via_api(
            prompt, largura, altura, seed, self.timeout, avisar=None
        )
        return imagem


def gerar_arte_por_provedor(
    provedor,
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
    *,
    seed: Optional[int] = None,
) -> Image.Image:
    """Igual a `gerar_arte`, mas permite injetar outro PROVEDOR de imagem
    (qualquer objeto com o método `gerar`). Útil para testes e para trocar
    de API sem mexer no resto do pipeline. A arte é pedida na proporção da
    janela; se o provedor devolver None, cai no fallback local."""
    prompt = montar_prompt(nome, tipo, texto_regras, cor)
    if seed is None:
        seed = _seed_do_prompt(prompt)
    largura, altura = tamanho_janela_arte()
    arte = provedor.gerar(prompt, largura, altura, seed)
    if arte is None:
        return arte_fallback(nome, cor, largura, altura, seed)
    return ImageOps.fit(arte, (largura, altura), Image.LANCZOS)


# -- cache em disco -----------------------------------------------------

def _seed_do_prompt(prompt: str) -> int:
    """Seed reprodutível derivada do prompt: mesma carta -> mesma arte."""
    return _hash_int(prompt) % (2**31 - 1)


def _hash_int(texto: str) -> int:
    return int(hashlib.sha256(texto.encode("utf-8")).hexdigest()[:8], 16)


def _caminho_cache(prompt: str, largura: int, altura: int, seed: int) -> Path:
    chave = hashlib.sha256(
        f"{prompt}|{largura}x{altura}|{seed}".encode("utf-8")
    ).hexdigest()[:24]
    return CACHE_DIR / f"{chave}.png"


def _ler_cache(prompt: str, largura: int, altura: int, seed: int) -> Optional[Image.Image]:
    caminho = _caminho_cache(prompt, largura, altura, seed)
    if caminho.exists():
        try:
            return Image.open(caminho).convert("RGB")
        except Exception:
            return None
    return None


def _salvar_cache(
    imagem: Image.Image, prompt: str, largura: int, altura: int, seed: int
) -> None:
    caminho = _caminho_cache(prompt, largura, altura, seed)
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        imagem.save(caminho, "PNG")
    except Exception:
        pass


# ===========================================================================
# 3. COMPOSIÇÃO (frame/molduras)
# ===========================================================================

@dataclass
class _Estilo:
    """Cores e medidas derivadas da cor da carta, para desenhar o frame."""

    claro: Tuple[int, int, int]
    escuro: Tuple[int, int, int]
    texto: Tuple[int, int, int]


def compor_frame(
    arte: Image.Image,
    nome: str,
    tipo_linha: str,
    texto_regras: str,
    custo_mana: str = "",
    poder: Optional[int] = None,
    resistencia: Optional[int] = None,
    cor: str = "C",
    largura: int = LARGURA_PADRAO,
    altura: int = ALTURA_PADRAO,
) -> Image.Image:
    """Compõe a carta final: arte + molduras + textos no estilo Magic.

    A moldura é desenhada em código (nada de template oficial):
      - borda preta externa, segundo contorno fino e cantos arredondados;
      - painel do TÍTULO com degradê da cor, nome à esquerda e custo em
        SÍMBOLOS desenhados (`pipeline.simbolos`) à direita;
      - janela da ARTE com uma sombra interna sutil;
      - caixa de TIPO, caixa de TEXTO tipo pergaminho e selo de P/T;
      - rodapé discreto com sigilo próprio e o aviso "não oficial";
      - tudo em fontes livres (Cinzel/EB Garamond), caindo na fonte padrão
        do Pillow se os .ttf não estiverem presentes.

    Devolve a imagem final (RGB) já no tamanho pedido.
    """
    estilo = _estilo_para(cor)
    base = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    # --- fundo (moldura) com cantos arredondados --------------------------
    # Preenche o retângulo arredondado com o tom claro da cor, depois
    # desenha a borda preta externa por cima.
    draw.rounded_rectangle(
        [0, 0, largura - 1, altura - 1],
        radius=RAIO_CANTO,
        fill=estilo.claro + (255,),
        outline=(15, 15, 15, 255),
        width=MARGEM_EXTERNA // 2,
    )
    # Segundo contorno, fino e discreto: dá o "acabamento" de carta real
    # sem competir com a borda preta externa.
    draw.rounded_rectangle(
        [6, 6, largura - 7, altura - 7],
        radius=max(4, RAIO_CANTO - 4),
        outline=_misturar(estilo.escuro, (0, 0, 0), 0.35) + (255,),
        width=2,
    )

    x0 = MARGEM_EXTERNA
    x1 = largura - MARGEM_EXTERNA
    largura_util = x1 - x0

    # --- geometria vertical dos painéis (compartilhada com a arte) --------
    # Usamos `_geometria` para que a janela de arte seja EXATAMENTE a mesma
    # que `gerar_arte` usa ao pedir a imagem à API -- assim a ilustração
    # entra sem recorte.
    g = _geometria(largura, altura)
    y_titulo0, y_titulo1 = g["y_titulo0"], g["y_titulo1"]
    y_arte0, y_arte1 = g["y_arte0"], g["y_arte1"]
    y_tipo0, y_tipo1 = g["y_tipo0"], g["y_tipo1"]
    y_texto0, y_texto1 = g["y_texto0"], g["y_texto1"]

    # --- painel do título (gradiente horizontal da cor) ------------------
    painel_titulo = _gradiente_horizontal(
        (largura_util, y_titulo1 - y_titulo0), estilo.escuro, estilo.claro
    )
    base.paste(painel_titulo, (x0, y_titulo0), _mascara_arredondada(
        painel_titulo.size, radius=12
    ))
    # Brilho no topo e sombra na base do painel: o degradê sozinho ficava
    # chapado. Cores opacas (blend) para não abrir buracos no RGBA.
    draw.line(
        [(x0 + 12, y_titulo0 + 2), (x1 - 12, y_titulo0 + 2)],
        fill=_misturar(estilo.claro, (255, 255, 255), 0.55) + (255,),
        width=2,
    )
    draw.line(
        [(x0 + 12, y_titulo1 - 2), (x1 - 12, y_titulo1 - 2)],
        fill=_misturar(estilo.escuro, (0, 0, 0), 0.25) + (255,),
        width=2,
    )

    # --- janela da arte ---------------------------------------------------
    arte_ajustada = ImageOps.fit(
        arte.convert("RGB"), (largura_util, y_arte1 - y_arte0), Image.LANCZOS
    )
    base.paste(arte_ajustada, (x0, y_arte0), _mascara_arredondada(
        arte_ajustada.size, radius=8
    ))
    # Sombra interna sutil no topo/base da arte, para "assentar" a imagem.
    _aplicar_vinheta(base, (x0, y_arte0, x1, y_arte1))

    # --- moldura interna fina em volta da arte ---------------------------
    draw.rounded_rectangle(
        [x0, y_arte0, x1, y_arte1],
        radius=8,
        outline=(20, 20, 20, 220),
        width=3,
    )

    # --- caixa de tipo (gradiente + textos) ------------------------------
    painel_tipo = _gradiente_horizontal(
        (largura_util, y_tipo1 - y_tipo0), estilo.escuro, estilo.claro
    )
    base.paste(painel_tipo, (x0, y_tipo0), _mascara_arredondada(
        painel_tipo.size, radius=8
    ))
    draw = ImageDraw.Draw(base)

    # --- caixa de texto (fundo pergaminho translúcido) -------------------
    caixa_texto = Image.new("RGBA", (largura_util, y_texto1 - y_texto0), (255, 255, 255, 0))
    cdraw = ImageDraw.Draw(caixa_texto)
    cdraw.rounded_rectangle(
        [0, 0, largura_util - 1, (y_texto1 - y_texto0) - 1],
        radius=8,
        fill=(250, 246, 233, 238),
        outline=(30, 30, 30, 200),
        width=2,
    )
    base.paste(caixa_texto, (x0, y_texto0), caixa_texto)

    # --- fontes -----------------------------------------------------------
    f_texto = _carregar_fonte(FONTE_TEXTO, 23)
    f_pt = _carregar_fonte(FONTE_TITULO, 34)

    # --- custo de mana (calculado antes do nome, para reservar o espaço) --
    custo_img: Optional[Image.Image] = None
    if custo_mana:
        # Se o custo for muito longo, encolhe para não invadir o nome.
        tamanho_custo = 36
        custo_img = desenhar_custo(custo_mana, tamanho_custo)
        while custo_img.width > largura_util * 0.42 and tamanho_custo > 20:
            tamanho_custo -= 2
            custo_img = desenhar_custo(custo_mana, tamanho_custo)

    # --- nome: encolhe (e, no limite, trunca) se o custo ocupar espaço ----
    disponivel_nome = largura_util - 36
    if custo_img is not None:
        disponivel_nome -= custo_img.width + 12
    tamanho_nome = 34
    f_nome = _carregar_fonte(FONTE_TITULO, tamanho_nome)
    while tamanho_nome > 20 and draw.textlength(nome, font=f_nome) > disponivel_nome:
        tamanho_nome -= 2
        f_nome = _carregar_fonte(FONTE_TITULO, tamanho_nome)
    nome_desenho = nome
    if draw.textlength(nome_desenho, font=f_nome) > disponivel_nome:
        while (
            nome_desenho
            and draw.textlength(nome_desenho + "...", font=f_nome) > disponivel_nome
        ):
            nome_desenho = nome_desenho[:-1]
        nome_desenho = nome_desenho.rstrip() + "..."

    # --- textos -----------------------------------------------------------
    draw.text(
        (x0 + 18, y_titulo0 + 12 + (34 - tamanho_nome) // 2),
        nome_desenho,
        font=f_nome,
        fill=estilo.texto,
    )
    if custo_img is not None:
        base.alpha_composite(
            custo_img,
            (
                x1 - 18 - custo_img.width,
                y_titulo0 + (ALTURA_TITULO - custo_img.height) // 2,
            ),
        )

    # --- linha de tipo: mesma ideia, encolhe/trunca se for longa ----------
    disponivel_tipo = largura_util - 36
    tamanho_tipo = 24
    f_tipo = _carregar_fonte(FONTE_TITULO, tamanho_tipo)
    while tamanho_tipo > 16 and draw.textlength(tipo_linha, font=f_tipo) > disponivel_tipo:
        tamanho_tipo -= 1
        f_tipo = _carregar_fonte(FONTE_TITULO, tamanho_tipo)
    tipo_desenho = tipo_linha
    if draw.textlength(tipo_desenho, font=f_tipo) > disponivel_tipo:
        while (
            tipo_desenho
            and draw.textlength(tipo_desenho + "...", font=f_tipo) > disponivel_tipo
        ):
            tipo_desenho = tipo_desenho[:-1]
        tipo_desenho = tipo_desenho.rstrip() + "..."
    draw.text(
        (x0 + 18, y_tipo0 + 8 + (24 - tamanho_tipo) // 2),
        tipo_desenho,
        font=f_tipo,
        fill=estilo.texto,
    )

    # Corpo do texto: quebra em linhas já contando a largura dos símbolos
    # (que não são texto), reduz a fonte se não couber e desenha tudo.
    linhas, f_texto, d_simbolo, espaco_simbolo, altura_linha = _layout_regras(
        texto_regras,
        f_texto,
        draw,
        largura_util - 36,
        (y_texto1 - y_texto0) - 24,
    )
    _desenhar_regras(
        base,
        draw,
        x0 + 18,
        y_texto0 + 12,
        linhas,
        f_texto,
        d_simbolo,
        espaco_simbolo,
        altura_linha,
    )

    # --- selo de P/T no rodapé -------------------------------------------
    if poder is not None and resistencia is not None:
        pt = f"{poder}/{resistencia}"
        caixa_pt = _caixa_pt(draw, pt, f_pt, estilo)
        x_pt = x1 - 14 - (caixa_pt.width)
        y_pt = altura - MARGEM_EXTERNA - 40
        base.paste(caixa_pt, (x_pt, y_pt), caixa_pt)

    # --- rodapé: sigilo próprio + aviso de "não oficial" -----------------
    _rodape_carta(base, x0, altura, estilo)

    # --- soma final (aplica a máscara arredondada da carta) --------------
    final = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    final.paste(base, (0, 0), _mascara_arredondada((largura, altura), radius=RAIO_CANTO))
    return final.convert("RGB")


def _estilo_para(cor: str) -> _Estilo:
    base = CORES_FRAME.get(cor, CORES_FRAME["C"])
    return _Estilo(
        claro=_hex_para_rgb(base["claro"]),
        escuro=_hex_para_rgb(base["escuro"]),
        texto=_hex_para_rgb(base["texto"]),
    )


def _caixa_pt(draw: ImageDraw.ImageDraw, texto: str, fonte, estilo: _Estilo) -> Image.Image:
    """Desenha o selo de poder/resistência (retângulo com o valor), como nas
    cartas reais."""
    largura_txt = draw.textlength(texto, font=fonte)
    pad = 16
    w = int(largura_txt + pad * 2)
    h = 44
    caixa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(caixa)
    cdraw.rounded_rectangle(
        [0, 0, w - 1, h - 1],
        radius=8,
        fill=estilo.claro + (255,),
        outline=(20, 20, 20, 220),
        width=2,
    )
    cdraw.text((pad, 4), texto, font=fonte, fill=estilo.texto)
    return caixa


def _unidades_por_paragrafo(texto: str) -> List[List[str]]:
    """Cada parágrafo (linha do texto original) vira uma lista de unidades.

    Uma unidade é uma palavra ou um grupo de símbolos colados (``{2}{W}``):
    é a menor coisa que não pode ser quebrada no meio. Parágrafos vazios
    viram lista vazia, para preservar o espaçamento original.
    """
    paragrafos: List[List[str]] = []
    for linha in str(texto).split("\n"):
        paragrafos.append(linha.split() if linha.strip() else [])
    return paragrafos


def _medir_unidade(
    unidade: str,
    fonte,
    draw: ImageDraw.ImageDraw,
    d_simbolo: int,
    espaco_simbolo: int,
) -> float:
    """Largura, em pixels, de uma unidade mista (texto + símbolos)."""
    total = 0.0
    anterior_simbolo = False
    for tipo, valor in dividir(unidade):
        if tipo == "simbolo":
            if anterior_simbolo:
                total += espaco_simbolo
            total += d_simbolo
            anterior_simbolo = True
        else:
            total += draw.textlength(valor, font=fonte)
            anterior_simbolo = False
    return total


def _layout_regras(
    texto: str,
    fonte_inicial,
    draw: ImageDraw.ImageDraw,
    largura_max: int,
    altura_max: int,
) -> Tuple[List[Optional[List[str]]], object, int, int, int]:
    """Quebra o texto em linhas e escolhe o tamanho da fonte que cabe.

    Devolve ``(linhas, fonte, diâmetro_do_símbolo, espaço_entre_símbolos,
    altura_da_linha)``. Cada linha é uma lista de unidades; ``None`` marca
    uma linha em branco (parágrafo vazio). Se o texto não couber na altura,
    a fonte (e os símbolos, junto) encolhem até caber ou até 13 px -- o
    mesmo truque das cartas reais.
    """
    tamanho = getattr(fonte_inicial, "size", 23)
    while True:
        fonte = _carregar_fonte(FONTE_TEXTO, tamanho)
        d_simbolo = max(12, int(round(tamanho * 0.96)))
        espaco_simbolo = max(1, d_simbolo // 8)
        espaco = draw.textlength(" ", font=fonte)
        ascent, descent = fonte.getmetrics()
        altura_linha = ascent + descent + 5

        linhas: List[Optional[List[str]]] = []
        for unidades in _unidades_por_paragrafo(texto):
            if not unidades:
                linhas.append(None)
                continue
            atual: List[str] = []
            largura_atual = 0.0
            for unidade in unidades:
                largura_unidade = _medir_unidade(
                    unidade, fonte, draw, d_simbolo, espaco_simbolo
                )
                extra = largura_unidade if not atual else espaco + largura_unidade
                if atual and largura_atual + extra > largura_max:
                    linhas.append(atual)
                    atual, largura_atual = [unidade], largura_unidade
                else:
                    atual.append(unidade)
                    largura_atual += extra
            if atual:
                linhas.append(atual)

        if len(linhas) * altura_linha <= altura_max or tamanho <= 13:
            return linhas, fonte, d_simbolo, espaco_simbolo, altura_linha
        tamanho -= 1


def _desenhar_regras(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    linhas: List[Optional[List[str]]],
    fonte,
    d_simbolo: int,
    espaco_simbolo: int,
    altura_linha: int,
) -> None:
    """Desenha as linhas de `_layout_regras`, colando os símbolos RGBA.

    `draw` e `base` precisam apontar para a mesma imagem (o texto é
    desenhado pelo `draw` e os símbolos são colados na `base`).
    """
    ascent, descent = fonte.getmetrics()
    espaco = draw.textlength(" ", font=fonte)
    for linha in linhas:
        if linha is None:
            y += altura_linha
            continue
        x_atual = x
        anterior_simbolo = False
        for i, unidade in enumerate(linha):
            if i:
                x_atual += espaco
            for tipo, valor in dividir(unidade):
                if tipo == "simbolo":
                    if anterior_simbolo:
                        x_atual += espaco_simbolo
                    simbolo = desenhar_simbolo(valor, d_simbolo)
                    y_simbolo = int(
                        round(y + (ascent + descent - simbolo.height) / 2)
                    )
                    base.alpha_composite(
                        simbolo, (int(round(x_atual)), y_simbolo)
                    )
                    x_atual += simbolo.width
                    anterior_simbolo = True
                else:
                    # Pontuação logo depois de um símbolo ("{T}:") fica
                    # colada nele; sem esse ajuste o espaço interno do
                    # símbolo + o lado esquerdo da vírgula/dois-pontos
                    # abriam um vão estranho.
                    if anterior_simbolo and valor[:1] in ",.;:!?)]}":
                        x_atual -= d_simbolo * 0.07
                    draw.text(
                        (x_atual, y), valor, font=fonte, fill=(20, 20, 20)
                    )
                    x_atual += draw.textlength(valor, font=fonte)
                    anterior_simbolo = False
        y += altura_linha


def _rodape_carta(
    base: Image.Image, x0: int, altura: int, estilo: _Estilo
) -> None:
    """Rodapé discreto: sigilo próprio + aviso de que a carta é fan-made.

    Ocupa o canto inferior esquerdo (o selo de P/T fica à direita), no
    lugar onde cartas reais trazem o símbolo do set e o crédito do artista.
    """
    cor = _misturar(estilo.texto, estilo.claro, 0.35)
    desenho = ImageDraw.Draw(base)
    y = altura - MARGEM_EXTERNA - 20
    cx, cy, r = x0 + 7, y - 4, 5
    desenho.polygon(
        [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
        fill=cor + (255,),
    )
    desenho.text(
        (x0 + 18, y - 13),
        "não oficial",
        font=_carregar_fonte(FONTE_TITULO, 13),
        fill=cor + (255,),
    )


def _misturar(
    cor1: Tuple[int, int, int],
    cor2: Tuple[int, int, int],
    fator: float,
) -> Tuple[int, int, int]:
    """Interpola duas cores RGB (0 = cor1, 1 = cor2)."""
    return tuple(
        int(round(cor1[i] * (1 - fator) + cor2[i] * fator)) for i in range(3)
    )


def _gradiente_horizontal(
    tamanho: Tuple[int, int],
    esquerda: Tuple[int, int, int],
    direita: Tuple[int, int, int],
) -> Image.Image:
    """Gera um degradê horizontal (esquerda -> direita) do tamanho pedido."""
    largura, altura = tamanho
    img = Image.new("RGB", (largura, altura))
    draw = ImageDraw.Draw(img)
    for x in range(largura):
        f = x / max(1, largura - 1)
        cor = tuple(int(esquerda[i] * (1 - f) + direita[i] * f) for i in range(3))
        draw.line([(x, 0), (x, altura)], fill=cor)
    return img


def _mascara_arredondada(tamanho: Tuple[int, int], radius: int) -> Image.Image:
    """Máscara RGBA (branco opaco) com cantos arredondados, para colar
    painéis com o mesmo raio da carta."""
    mascara = Image.new("L", tamanho, 0)
    ImageDraw.Draw(mascara).rounded_rectangle(
        [0, 0, tamanho[0] - 1, tamanho[1] - 1], radius=radius, fill=255
    )
    return mascara


def _aplicar_vinheta(
    base: Image.Image, caixa: Tuple[int, int, int, int]
) -> None:
    """Escurece levemente o topo e a base da janela de arte, dando
    profundidade e destacando o recorte. Modifica `base` in-place."""
    x0, y0, x1, y1 = caixa
    altura = y1 - y0
    faixa = Image.new("RGBA", (x1 - x0, altura), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(faixa)
    passos = max(1, altura // 6)
    for i in range(passos):
        alfa = int(70 * (1 - i / passos))
        fdraw.line([(0, i), (x1 - x0, i)], fill=(0, 0, 0, alfa))
        fdraw.line([(0, altura - 1 - i), (x1 - x0, altura - 1 - i)], fill=(0, 0, 0, alfa))
    base.alpha_composite(faixa, (x0, y0))


# -- auxiliares de fonte/cor -------------------------------------------

@lru_cache(maxsize=64)
def _carregar_fonte(caminho: str, tamanho: int):
    """Carrega o .ttf pedido; se não existir (repo clonado sem as fontes),
    devolve a fonte padrão do Pillow em vez de quebrar.

    O cache importa: `_layout_regras` testa vários tamanhos até o texto
    caber, e sem ele o arquivo .ttf seria lido dezenas de vezes por carta.
    """
    try:
        return ImageFont.truetype(caminho, tamanho)
    except Exception:
        try:
            return ImageFont.load_default(size=tamanho)
        except TypeError:  # Pillow antigo sem o parâmetro size
            return ImageFont.load_default()


def _hex_para_rgb(valor: str) -> Tuple[int, int, int]:
    valor = valor.lstrip("#")
    return tuple(int(valor[i : i + 2], 16) for i in (0, 2, 4))
