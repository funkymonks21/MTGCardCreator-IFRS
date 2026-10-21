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
  gradiente da cor, caixa de texto translúcida, selo de P/T, símbolo de
  custo). Não usamos imagens/templates oficiais da Wizards (material
  protegido): o visual é uma releitura vetorial própria.
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
import random
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

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

def gerar_arte(
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
    *,
    largura: int = LARGURA_PADRAO,
    altura: int = ALTURA_PADRAO,
    seed: Optional[int] = None,
    usar_cache: bool = True,
    timeout: int = TIMEOUT_API,
) -> Image.Image:
    """Gera a ILUSTRAÇÃO da carta (sem moldura) a partir do nome/tipo/texto.

    Tenta a API de imagem (Pollinations.ai) e, se ela falhar, cai num
    fallback procedural local -- nunca levanta exceção por causa de rede.
    O resultado é uma `PIL.Image.Image` já no tamanho pedido (por padrão
    630x880, a proporção da carta).

    `seed` fixa a arte (mesma seed -> mesma imagem); se None, a seed é
    derivada do prompt, o que também a torna reprodutível para o mesmo
    nome/tipo/texto. `usar_cache` reaproveita o arquivo em
    `assets/cache_arte/` quando existir.
    """
    prompt = montar_prompt(nome, tipo, texto_regras, cor)
    if seed is None:
        seed = _seed_do_prompt(prompt)

    if usar_cache:
        em_cache = _ler_cache(prompt, largura, altura, seed)
        if em_cache is not None:
            return em_cache

    arte = _arte_via_api(prompt, largura, altura, seed, timeout)
    if arte is None:
        arte = arte_fallback(nome, cor, largura, altura, seed)

    if usar_cache:
        _salvar_cache(arte, prompt, largura, altura, seed)
    return arte


def _arte_via_api(
    prompt: str,
    largura: int,
    altura: int,
    seed: int,
    timeout: int,
) -> Optional[Image.Image]:
    """Baixa a arte da Pollinations.ai. Devolve None em QUALQUER falha
    (rede, status ruim, corpo não-imagem), para o chamador usar o
    fallback."""
    prompt_url = urllib.parse.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{prompt_url}"
    params = {
        "width": int(largura),
        "height": int(altura),
        "seed": int(seed),
        "nologo": "true",
        "model": "flux",
    }
    try:
        resposta = requests.get(url, params=params, timeout=timeout)
        resposta.raise_for_status()
        if not resposta.headers.get("content-type", "").startswith("image/"):
            return None
        imagem = Image.open(io.BytesIO(resposta.content)).convert("RGB")
        # A API nem sempre respeita exatamente o tamanho pedido: garantimos
        # o tamanho final recortando/preenchendo (sem distorcer).
        return ImageOps.fit(imagem, (largura, altura), Image.LANCZOS)
    except Exception:
        return None


def arte_fallback(
    nome: str,
    cor: str = "C",
    largura: int = LARGURA_PADRAO,
    altura: int = ALTURA_PADRAO,
    seed: int = 0,
) -> Image.Image:
    """Arte procedural determinística, usada quando a API não está
    disponível. Desenha um degradê de duas cores (derivadas da cor da
    carta e do nome) com algumas formas suaves, só para a carta não ficar
    sem ilustração. Não é arte "de verdade" -- é o plano B."""
    base = CORES_FRAME.get(cor, CORES_FRAME["C"])
    rng = random.Random((seed or 0) ^ _hash_int(nome))

    # Degradê vertical entre o tom escuro e o claro da cor.
    topo = _hex_para_rgb(base["escuro"])
    baixo = _hex_para_rgb(base["claro"])
    img = Image.new("RGB", (largura, altura), topo)
    draw = ImageDraw.Draw(img)
    for y in range(altura):
        f = y / max(1, altura - 1)
        draw.line(
            [(0, y), (largura, y)],
            fill=tuple(int(topo[i] * (1 - f) + baixo[i] * f) for i in range(3)),
        )

    # Formas suaves semi-transparentes, para dar alguma textura.
    overlay = Image.new("RGBA", (largura, altura), (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    for _ in range(9):
        r = rng.randint(largura // 6, largura // 2)
        cx = rng.randint(0, largura)
        cy = rng.randint(0, altura)
        alfa = rng.randint(18, 55)
        tom = tuple(rng.randint(200, 255) for _ in range(3))
        odraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tom + (alfa,))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=largura // 20))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


class ArtePollinations:
    """Cliente fino para a API de imagem da Pollinations.ai.

    Existe para deixar explícito o "contrato" de um provedor de arte:
    `gerar(prompt, largura, altura, seed) -> Optional[Image.Image]`.
    Para trocar de serviço (Hugging Face, Stability...), basta
    implementar a mesma assinatura e passar a instância para
    `gerar_arte_por_provedor`.
    """

    URL = "https://image.pollinations.ai/prompt/"

    def __init__(self, modelo: str = "flux", timeout: int = TIMEOUT_API):
        self.modelo = modelo
        self.timeout = timeout

    def gerar(
        self, prompt: str, largura: int, altura: int, seed: int
    ) -> Optional[Image.Image]:
        return _arte_via_api(prompt, largura, altura, seed, self.timeout)


def gerar_arte_por_provedor(
    provedor,
    nome: str,
    tipo: str,
    texto_regras: str = "",
    cor: str = "C",
    *,
    largura: int = LARGURA_PADRAO,
    altura: int = ALTURA_PADRAO,
    seed: Optional[int] = None,
) -> Image.Image:
    """Igual a `gerar_arte`, mas permite injetar outro PROVEDOR de imagem
    (qualquer objeto com o método `gerar`). Útil para testes e para trocar
    de API sem mexer no resto do pipeline."""
    prompt = montar_prompt(nome, tipo, texto_regras, cor)
    if seed is None:
        seed = _seed_do_prompt(prompt)
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
      - borda preta externa e cantos arredondados;
      - painel do TÍTULO com degradê da cor, nome à esquerda e custo à
        direita;
      - janela da ARTE com uma sombra interna sutil;
      - caixa de TIPO, caixa de TEXTO translúcida e selo de P/T;
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

    x0 = MARGEM_EXTERNA
    x1 = largura - MARGEM_EXTERNA
    largura_util = x1 - x0

    # --- geometria vertical dos painéis ----------------------------------
    y_titulo0, y_titulo1 = x0, x0 + 62
    y_arte0 = y_titulo1 + 6
    y_arte1 = y_titulo0 + int(altura * 0.50)
    y_tipo0, y_tipo1 = y_arte1 + 8, y_arte1 + 8 + 44
    y_texto0 = y_tipo1 + 10
    # A caixa de texto vai até deixar espaço para o selo de P/T no rodapé.
    y_texto1 = altura - MARGEM_EXTERNA - 46

    # --- painel do título (gradiente horizontal da cor) ------------------
    painel_titulo = _gradiente_horizontal(
        (largura_util, y_titulo1 - y_titulo0), estilo.escuro, estilo.claro
    )
    base.paste(painel_titulo, (x0, y_titulo0), _mascara_arredondada(
        painel_titulo.size, radius=12
    ))

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

    # --- caixa de texto (fundo claro translúcido) ------------------------
    caixa_texto = Image.new("RGBA", (largura_util, y_texto1 - y_texto0), (255, 255, 255, 0))
    cdraw = ImageDraw.Draw(caixa_texto)
    cdraw.rounded_rectangle(
        [0, 0, largura_util - 1, (y_texto1 - y_texto0) - 1],
        radius=8,
        fill=(255, 255, 255, 210),
        outline=(30, 30, 30, 200),
        width=2,
    )
    base.paste(caixa_texto, (x0, y_texto0), caixa_texto)

    # --- fontes -----------------------------------------------------------
    f_nome = _carregar_fonte(FONTE_TITULO, 34)
    f_custo = _carregar_fonte(FONTE_TITULO, 30)
    f_tipo = _carregar_fonte(FONTE_TITULO, 24)
    f_texto = _carregar_fonte(FONTE_TEXTO, 23)
    f_pt = _carregar_fonte(FONTE_TITULO, 34)

    # --- textos -----------------------------------------------------------
    draw.text(
        (x0 + 18, y_titulo0 + 12),
        nome,
        font=f_nome,
        fill=estilo.texto,
    )
    if custo_mana:
        largura_custo = draw.textlength(custo_mana, font=f_custo)
        draw.text(
            (x1 - 18 - largura_custo, y_titulo0 + 16),
            custo_mana,
            font=f_custo,
            fill=estilo.texto,
        )

    draw.text(
        (x0 + 18, y_tipo0 + 8),
        tipo_linha,
        font=f_tipo,
        fill=estilo.texto,
    )

    # Corpo do texto: quebra as linhas para caber na caixa.
    texto_formatado = _quebrar_para_caixa(
        texto_regras,
        f_texto,
        largura_util - 36,
        draw,
        altura_max=(y_texto1 - y_texto0) - 24,
        tamanho_fonte=23,
    )
    draw.multiline_text(
        (x0 + 18, y_texto0 + 12),
        texto_formatado,
        font=f_texto,
        fill=(20, 20, 20),
        spacing=5,
    )

    # --- selo de P/T no rodapé -------------------------------------------
    if poder is not None and resistencia is not None:
        pt = f"{poder}/{resistencia}"
        caixa_pt = _caixa_pt(draw, pt, f_pt, estilo)
        x_pt = x1 - 14 - (caixa_pt.width)
        y_pt = altura - MARGEM_EXTERNA - 40
        base.paste(caixa_pt, (x_pt, y_pt), caixa_pt)

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


def _quebrar_para_caixa(
    texto: str,
    fonte,
    largura_max: int,
    draw: ImageDraw.ImageDraw,
    altura_max: int,
    tamanho_fonte: int,
) -> str:
    """Quebra o texto em linhas que caibam na largura da caixa, respeitando
    as quebras de linha já existentes ("\\n" separa keywords/parágrafos).
    Se o texto não couber na altura, reduz a fonte até caber (até um
    limite) -- o mesmo truque das cartas reais, que encolhem o texto."""
    tamanho = tamanho_fonte
    while True:
        fonte_atual = _carregar_fonte(FONTE_TEXTO, tamanho)
        linhas = _reflow(texto, fonte_atual, largura_max, draw)
        altura = draw.multiline_textbbox((0, 0), linhas, font=fonte_atual, spacing=5)[3]
        if altura <= altura_max or tamanho <= 13:
            return linhas
        tamanho -= 1


def _reflow(texto: str, fonte, largura_max: int, draw: ImageDraw.ImageDraw) -> str:
    """Reorganiza cada linha original em linhas que caibam em `largura_max`
    pixels. Nunca junta uma linha original com a seguinte."""
    linhas_finais: List[str] = []
    for linha_original in str(texto).split("\n"):
        palavras = linha_original.split()
        if not palavras:
            continue
        atual = ""
        for palavra in palavras:
            teste = f"{atual} {palavra}".strip()
            if draw.textlength(teste, font=fonte) <= largura_max:
                atual = teste
            else:
                if atual:
                    linhas_finais.append(atual)
                atual = palavra
        if atual:
            linhas_finais.append(atual)
    return "\n".join(linhas_finais)


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

def _carregar_fonte(caminho: str, tamanho: int):
    """Carrega o .ttf pedido; se não existir (repo clonado sem as fontes),
    devolve a fonte padrão do Pillow em vez de quebrar."""
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
