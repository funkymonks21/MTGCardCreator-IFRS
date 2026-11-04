"""
Símbolos de mana e de habilidade da carta, em discos coloridos.

Duas camadas de desenho
=======================
1. **Fonte Mana** (`assets/fonts/mana.ttf`): quando o arquivo existe, o
   glifo de cada símbolo vem dessa fonte livre e o módulo só desenha o
   disco colorido por baixo. É o que deixa o resultado mais próximo de
   uma carta de verdade (sol, gota, caveira, chama, árvore...).
2. **Desenho em código** (`_glifo_*`): se a fonte não estiver presente
   (clone sem os assets, por exemplo), os mesmos símbolos são desenhados
   com elipses e polígonos do Pillow. Fica um pouco mais simples, mas a
   carta continua sendo gerada.

Sobre licença: a fonte Mana é distribuída sob SIL OFL 1.1 (ver
`assets/fonts/mana-LICENSE.txt`), mas os DESENHOS dos símbolos de mana são
marca da Wizards of the Coast. Aqui eles são usados num projeto acadêmico,
fan-made e sem fins comerciais; sem a fonte, o app usa a camada 2, que é
inteiramente original.

O que é suportado
=================
- custo de mana no formato do Oracle: ``{2}{W}``, ``{X}{R}``, ``{T}``;
- símbolos coloridos W/U/B/R/G, incolor ``{C}``, neve ``{S}``;
- genéricos (``{0}``..``{20}``), ``{X}``;
- híbridos ``{W/U}``, ``{2/W}`` (metade a metade);
- ``{T}`` (virar) e ``{Q}`` (desvirar);
- qualquer coisa desconhecida vira um disco com o texto, sem sumir.

Uso
===
    from pipeline.simbolos import dividir, desenhar_custo, desenhar_simbolo

    dividir("Discard a card: {2}{W}.")   # -> texto/símbolo intercalados
    desenhar_custo("{2}{W}", 34)          # -> imagem RGBA da fileira
    desenhar_simbolo("W", 22)             # -> um símbolo só

As imagens devolvidas são RGBA e ficam em cache (`lru_cache`); quem chama
não deve modificá-las, só colar/alpha_composite.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# A mesma fonte livre da moldura (Cinzel, SIL OFL). Se o repo for clonado
# sem `assets/fonts/`, cai na fonte padrão do Pillow sem quebrar.
FONTE_SIMBOLO = "assets/fonts/Cinzel.ttf"

# Fonte Mana (SIL OFL 1.1, https://mana.andrewgioia.com): traz o glifo de
# cada símbolo no lugar de desenharmos um por um. O disco colorido e o
# contorno continuam sendo desenhados aqui.
FONTE_MANA = "assets/fonts/mana.ttf"

# Código de cada símbolo na fonte Mana. Os genéricos seguem uma sequência
# (0..15 = U+E605..U+E614, 16..20 = U+E62A..U+E62E). Híbridos e phyrexianos
# não existem como glifo único: esses caem no desenho em código.
_MAPA_FONTE: Dict[str, str] = {
    "W": "\ue600",
    "U": "\ue601",
    "B": "\ue602",
    "R": "\ue603",
    "G": "\ue604",
    "C": "\ue904",
    "S": "\ue619",
    "X": "\ue615",
    "Y": "\ue616",
    "Z": "\ue617",
    "T": "\ue61a",
    "Q": "\ue61b",
}
_MAPA_FONTE.update({str(n): chr(0xE605 + n) for n in range(16)})
_MAPA_FONTE.update({str(n): chr(0xE62A + n - 16) for n in range(16, 21)})

_PADRAO = re.compile(r"\{([^{}]+)\}")

# Cores por símbolo. `fundo` é o disco, `borda` o contorno e `glifo` o
# desenho/texto. Os tons foram escolhidos para ficar legíveis no tamanho
# pequeno do texto de regras (~22 px).
_CORES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "W": {"fundo": (247, 241, 214), "borda": (120, 104, 62), "glifo": (110, 88, 32)},
    "U": {"fundo": (43, 120, 168), "borda": (16, 60, 88), "glifo": (234, 246, 255)},
    "B": {"fundo": (58, 52, 52), "borda": (18, 15, 15), "glifo": (233, 226, 214)},
    "R": {"fundo": (190, 62, 42), "borda": (86, 22, 12), "glifo": (255, 232, 214)},
    "G": {"fundo": (47, 105, 63), "borda": (18, 52, 30), "glifo": (228, 242, 219)},
    "C": {"fundo": (156, 150, 138), "borda": (58, 54, 46), "glifo": (246, 242, 231)},
    "S": {"fundo": (198, 216, 226), "borda": (62, 88, 104), "glifo": (30, 58, 74)},
    "T": {"fundo": (206, 202, 192), "borda": (70, 66, 58), "glifo": (45, 42, 36)},
    "Q": {"fundo": (206, 202, 192), "borda": (70, 66, 58), "glifo": (45, 42, 36)},
    "GENERICO": {"fundo": (202, 196, 182), "borda": (62, 58, 48), "glifo": (34, 31, 26)},
}

# Tom interno da chama, para o símbolo vermelho não ficar chapado.
_BRILHO = {"R": (244, 175, 70)}


def dividir(texto: str) -> List[Tuple[str, str]]:
    """Divide ``texto`` em pedaços de texto e símbolos, na ordem original.

    Devolve uma lista de ``("texto", "...")`` e ``("simbolo", "W")``. A
    chave do símbolo já vem em maiúsculas e sem espaços nas bordas. Texto
    sem ``{...}`` devolve um único pedaço de texto -- quem chama não
    precisa tratar o caso "sem símbolos" à parte.
    """
    if not texto:
        return []
    partes: List[Tuple[str, str]] = []
    ultimo = 0
    for achado in _PADRAO.finditer(texto):
        if achado.start() > ultimo:
            partes.append(("texto", texto[ultimo : achado.start()]))
        partes.append(("simbolo", achado.group(1).strip().upper()))
        ultimo = achado.end()
    if ultimo < len(texto):
        partes.append(("texto", texto[ultimo:]))
    return partes


def tem_simbolos(texto: str) -> bool:
    """True se houver pelo menos um ``{...}`` em ``texto``."""
    return bool(_PADRAO.search(texto or ""))


def _cores_para(chave: str) -> Dict[str, Tuple[int, int, int]]:
    return _CORES.get(chave, _CORES["GENERICO"])


@lru_cache(maxsize=256)
def desenhar_simbolo(chave: str, tamanho: int) -> Image.Image:
    """Desenha UM símbolo (``"W"``, ``"2"``, ``"T"``, ``"W/U"``...) como
    imagem RGBA quadrada de ``tamanho`` pixels de lado.

    O desenho é feito em resolução 4x e reduzido com LANCZOS, o que dá
    contornos suaves nos tamanhos pequenos usados no texto de regras.
    """
    chave = (chave or "?").strip().upper()
    tamanho = max(8, int(tamanho))
    escala = 4 if tamanho <= 48 else 3
    lado = tamanho * escala
    img = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(img)

    partes = [p for p in chave.split("/") if p]
    if len(partes) == 2:
        _desenhar_hibrido(img, desenho, partes[0], partes[1], lado)
    else:
        _desenhar_simples(img, desenho, chave, lado)

    return img.resize((tamanho, tamanho), Image.LANCZOS)


def desenhar_custo(custo: str, tamanho: int = 34, espaco: int = 4) -> Image.Image:
    """Desenha um custo de mana inteiro como uma fileira horizontal RGBA.

    Texto solto no meio do custo (raro, mas possível em nomes de símbolo
    desconhecido) é renderizado na fonte da moldura. Devolve uma imagem de
    altura ``tamanho``; se ``custo`` estiver vazio, devolve 1x1 transparente.
    """
    pecas: List[Image.Image] = []
    for tipo, valor in dividir(custo):
        if tipo == "simbolo":
            pecas.append(desenhar_simbolo(valor, tamanho))
        else:
            texto = valor.strip()
            if texto:
                pecas.append(_texto_em_imagem(texto, int(tamanho * 0.88)))

    if not pecas:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    largura = sum(p.width for p in pecas) + espaco * (len(pecas) - 1)
    altura = max(p.height for p in pecas)
    linha = Image.new("RGBA", (max(1, largura), altura), (0, 0, 0, 0))
    x = 0
    for peca in pecas:
        linha.alpha_composite(peca, (x, (altura - peca.height) // 2))
        x += peca.width + espaco
    return linha


# ---------------------------------------------------------------------------
# Desenho de cada símbolo
# ---------------------------------------------------------------------------


def _desenhar_simples(
    img: Image.Image, desenho: ImageDraw.ImageDraw, chave: str, lado: int
) -> None:
    """Disco colorido + glifo da cor/tipo.

    O glifo vem da fonte Mana quando ela está disponível; senão é desenhado
    em código (`_desenhar_glifo`), para o app nunca depender do arquivo.
    """
    cores = _cores_para(chave if not chave.isdigit() else "GENERICO")
    margem = lado * 0.045
    caixa = [margem, margem, lado - margem, lado - margem]
    desenho.ellipse(
        caixa,
        fill=cores["fundo"],
        outline=cores["borda"],
        width=max(1, int(lado * 0.034)),
    )
    if not _colar_glifo_fonte(img, chave, lado, cores["glifo"]):
        _desenhar_glifo(desenho, chave, lado, cores)


@lru_cache(maxsize=32)
def _fonte_mana(tamanho: int):
    """Fonte Mana no tamanho pedido, ou None se o arquivo não existir."""
    try:
        return ImageFont.truetype(FONTE_MANA, max(8, int(tamanho)))
    except Exception:
        return None


@lru_cache(maxsize=512)
def _mascara_glifo(chave: str, lado: int, fator: float):
    """Máscara "L" do glifo de ``chave`` na fonte Mana, cortada e ajustada.

    Devolve None quando a fonte não existe ou quando o símbolo não tem
    glifo próprio (híbridos, phyrexianos, texto desconhecido). O desenho é
    feito em resolução alta e reduzido, que é o que mantém a borda limpa
    nos 22 px do texto de regras.
    """
    caractere = _MAPA_FONTE.get(chave)
    if not caractere:
        return None
    grande = lado * 3
    fonte = _fonte_mana(grande)
    if fonte is None:
        return None
    tela = Image.new("L", (grande * 2, grande * 2), 0)
    ImageDraw.Draw(tela).text(
        (grande, grande), caractere, font=fonte, fill=255, anchor="mm"
    )
    caixa = tela.getbbox()
    if not caixa:
        return None
    recorte = tela.crop(caixa)
    alvo = max(2, int(round(lado * fator)))
    escala = min(alvo / recorte.width, alvo / recorte.height)
    tamanho = (
        max(1, int(round(recorte.width * escala))),
        max(1, int(round(recorte.height * escala))),
    )
    return recorte.resize(tamanho, Image.LANCZOS)


def _colar_glifo_fonte(
    img: Image.Image,
    chave: str,
    lado: int,
    cor: Tuple[int, int, int],
    fator: float = 0.60,
    centro: Tuple[float, float] = (0.5, 0.5),
    metade: str = "",
) -> bool:
    """Cola o glifo da fonte Mana em ``img``, pintado de ``cor``.

    Devolve False se não houver glifo para a chave (quem chamou desenha o
    próprio). ``centro`` é relativo ao lado (0..1), para posicionar o
    glifo na metade certa de um símbolo híbrido; ``metade`` (``"esquerda"``
    ou ``"direita"``) recorta o glifo na divisória, para um não invadir o
    lado do outro.
    """
    mascara = _mascara_glifo(chave, lado, fator)
    if mascara is None:
        return False
    x = int(round(centro[0] * lado - mascara.width / 2))
    y = int(round(centro[1] * lado - mascara.height / 2))

    if metade == "esquerda" and x + mascara.width > lado // 2:
        corte = x + mascara.width - lado // 2
        mascara = mascara.crop((0, 0, mascara.width - corte, mascara.height))
    elif metade == "direita" and x < lado // 2:
        corte = lado // 2 - x
        mascara = mascara.crop((corte, 0, mascara.width, mascara.height))
        x += corte
    if mascara.width < 1 or mascara.height < 1:
        return False

    pintada = Image.new("RGBA", mascara.size, cor + (255,))
    pintada.putalpha(mascara)
    img.alpha_composite(pintada, (max(0, x), max(0, y)))
    return True


def _desenhar_glifo(
    desenho: ImageDraw.ImageDraw,
    chave: str,
    lado: int,
    cores: Dict[str, Tuple[int, int, int]],
) -> None:
    c = lado / 2
    cor = cores["glifo"]

    if chave == "W":
        _glifo_sol(desenho, c, lado, cor)
    elif chave == "U":
        _glifo_gota(desenho, c, lado, cor)
    elif chave == "B":
        _glifo_lua(desenho, c, lado, cor, cores["fundo"])
    elif chave == "R":
        _glifo_chama(desenho, c, lado, cor, _BRILHO.get("R"))
    elif chave == "G":
        _glifo_arvore(desenho, c, lado, cor)
    elif chave == "C":
        _glifo_losango(desenho, c, lado, cor)
    elif chave == "S":
        _glifo_floco(desenho, c, lado, cor)
    elif chave == "T":
        _glifo_seta(desenho, c, lado, cor, diagonal=True)
    elif chave == "Q":
        _glifo_seta(desenho, c, lado, cor, diagonal=False)
    else:
        texto = chave if (chave.isdigit() or chave in {"X", "Y", "Z"}) else chave
        _texto_central(desenho, texto, lado, cor)


def _glifo_sol(desenho, c: float, lado: int, cor) -> None:
    """Sol da cor branca: disco central + oito raios."""
    raio = lado * 0.105
    desenho.ellipse([c - raio, c - raio, c + raio, c + raio], fill=cor)
    largura = max(1, int(lado * 0.042))
    for i in range(8):
        ang = math.radians(i * 45)
        x0, y0 = c + math.cos(ang) * lado * 0.19, c + math.sin(ang) * lado * 0.19
        x1, y1 = c + math.cos(ang) * lado * 0.30, c + math.sin(ang) * lado * 0.30
        desenho.line([(x0, y0), (x1, y1)], fill=cor, width=largura)


def _glifo_gota(desenho, c: float, lado: int, cor) -> None:
    """Gota da cor azul."""
    pontos = [
        (c, c - lado * 0.30),
        (c + lado * 0.13, c - lado * 0.07),
        (c + lado * 0.19, c + lado * 0.08),
        (c + lado * 0.16, c + lado * 0.20),
        (c + lado * 0.06, c + lado * 0.27),
        (c - lado * 0.06, c + lado * 0.27),
        (c - lado * 0.16, c + lado * 0.20),
        (c - lado * 0.19, c + lado * 0.08),
        (c - lado * 0.13, c - lado * 0.07),
    ]
    desenho.polygon(pontos, fill=cor)


def _glifo_lua(desenho, c: float, lado: int, cor, fundo) -> None:
    """Lua crescente da cor preta: um disco com outro deslocado por cima."""
    raio = lado * 0.245
    desenho.ellipse([c - raio, c - raio, c + raio, c + raio], fill=cor)
    deslocamento = lado * 0.105
    desenho.ellipse(
        [c - raio + deslocamento, c - raio - deslocamento * 0.45,
         c + raio + deslocamento, c + raio - deslocamento * 0.45],
        fill=fundo,
    )


def _glifo_chama(desenho, c: float, lado: int, cor, brilho) -> None:
    """Chama da cor vermelha, com um núcleo mais claro."""
    pontos = [
        (c, c - lado * 0.32),
        (c + lado * 0.12, c - lado * 0.10),
        (c + lado * 0.23, c + lado * 0.06),
        (c + lado * 0.18, c + lado * 0.24),
        (c, c + lado * 0.30),
        (c - lado * 0.18, c + lado * 0.24),
        (c - lado * 0.23, c + lado * 0.06),
        (c - lado * 0.10, c - lado * 0.08),
        (c - lado * 0.02, c - lado * 0.16),
    ]
    desenho.polygon(pontos, fill=cor)
    if brilho:
        interno = [
            (c, c - lado * 0.12),
            (c + lado * 0.09, c + lado * 0.04),
            (c + lado * 0.10, c + lado * 0.16),
            (c, c + lado * 0.22),
            (c - lado * 0.10, c + lado * 0.16),
            (c - lado * 0.09, c + lado * 0.04),
        ]
        desenho.polygon(interno, fill=brilho)


def _glifo_arvore(desenho, c: float, lado: int, cor) -> None:
    """Árvore estilizada da cor verde (copa em dois triângulos + tronco)."""
    desenho.polygon(
        [(c, c - lado * 0.31), (c + lado * 0.23, c + lado * 0.03),
         (c - lado * 0.23, c + lado * 0.03)],
        fill=cor,
    )
    desenho.polygon(
        [(c, c - lado * 0.15), (c + lado * 0.28, c + lado * 0.17),
         (c - lado * 0.28, c + lado * 0.17)],
        fill=cor,
    )
    desenho.rectangle(
        [c - lado * 0.045, c + lado * 0.14, c + lado * 0.045, c + lado * 0.29],
        fill=cor,
    )


def _glifo_losango(desenho, c: float, lado: int, cor) -> None:
    """Losango da cor incolor."""
    desenho.polygon(
        [(c, c - lado * 0.28), (c + lado * 0.22, c),
         (c, c + lado * 0.28), (c - lado * 0.22, c)],
        fill=cor,
    )


def _glifo_floco(desenho, c: float, lado: int, cor) -> None:
    """Floco de neve (seis braços) para o símbolo de mana nevada."""
    largura = max(1, int(lado * 0.038))
    for i in range(3):
        ang = math.radians(i * 60)
        dx, dy = math.cos(ang) * lado * 0.27, math.sin(ang) * lado * 0.27
        desenho.line([(c - dx, c - dy), (c + dx, c + dy)], fill=cor, width=largura)


def _glifo_seta(desenho, c: float, lado: int, cor, diagonal: bool) -> None:
    """Seta de virar (diagonal) e de desvirar (vertical)."""
    largura = max(1, int(lado * 0.07))
    if diagonal:
        desenho.line(
            [(c - lado * 0.14, c + lado * 0.14), (c + lado * 0.09, c - lado * 0.09)],
            fill=cor,
            width=largura,
        )
        ponta = [
            (c + lado * 0.22, c - lado * 0.22),
            (c + lado * 0.02, c - lado * 0.17),
            (c + lado * 0.17, c - lado * 0.02),
        ]
    else:
        desenho.line(
            [(c, c + lado * 0.16), (c, c - lado * 0.10)],
            fill=cor,
            width=largura,
        )
        ponta = [
            (c, c - lado * 0.24),
            (c - lado * 0.11, c - lado * 0.07),
            (c + lado * 0.11, c - lado * 0.07),
        ]
    desenho.polygon(ponta, fill=cor)


def _texto_central(desenho, texto: str, lado: int, cor) -> None:
    """Número/genérico/X: texto centralizado, encolhido até caber."""
    tamanho = int(lado * 0.58)
    fonte = _fonte(tamanho)
    while tamanho > 6:
        caixa = desenho.textbbox((0, 0), texto, font=fonte)
        if (caixa[2] - caixa[0]) <= lado * 0.62 and (caixa[3] - caixa[1]) <= lado * 0.62:
            break
        tamanho -= max(1, int(lado * 0.02))
        fonte = _fonte(tamanho)
    desenho.text((lado / 2, lado / 2), texto, font=fonte, fill=cor, anchor="mm")


def _desenhar_hibrido(
    img: Image.Image,
    desenho: ImageDraw.ImageDraw,
    primeira: str,
    segunda: str,
    lado: int,
) -> None:
    """Símbolo híbrido (``{W/U}``, ``{2/W}``): disco dividido ao meio.

    Com a fonte Mana, cada metade recebe o glifo do seu símbolo, recortado
    na divisória -- que é como as cartas de verdade fazem. Sem a fonte,
    caem as iniciais (``W``/``U``) desenhadas em código.
    """
    cores_1 = _cores_para(primeira if not primeira.isdigit() else "GENERICO")
    cores_2 = _cores_para(segunda if not segunda.isdigit() else "GENERICO")
    margem = lado * 0.045
    caixa = [margem, margem, lado - margem, lado - margem]

    desenho.pieslice(caixa, 90, 270, fill=cores_1["fundo"])
    desenho.pieslice(caixa, 270, 90, fill=cores_2["fundo"])
    desenho.ellipse(
        caixa,
        outline=cores_1["borda"],
        width=max(1, int(lado * 0.034)),
    )
    desenho.line(
        [(lado / 2, margem), (lado / 2, lado - margem)],
        fill=(24, 22, 19),
        width=max(1, int(lado * 0.018)),
    )

    if _colar_glifo_fonte(
        img, primeira, lado, cores_1["glifo"], fator=0.40,
        centro=(0.27, 0.5), metade="esquerda",
    ) and _colar_glifo_fonte(
        img, segunda, lado, cores_2["glifo"], fator=0.40,
        centro=(0.73, 0.5), metade="direita",
    ):
        return

    # Sem a fonte: letras pequenas em cada metade. Sem elas, um híbrido
    # como {2/W} ficava só "meio cinza, meio branco" e não dava para ler.
    _texto_na_metade(
        desenho, primeira, lado * 0.27, lado / 2, lado, cores_1["glifo"]
    )
    _texto_na_metade(
        desenho, segunda, lado * 0.73, lado / 2, lado, cores_2["glifo"]
    )


def _texto_na_metade(
    desenho: ImageDraw.ImageDraw,
    texto: str,
    centro_x: float,
    centro_y: float,
    lado: int,
    cor: Tuple[int, int, int],
) -> None:
    """Texto pequeno centralizado numa das metades de um símbolo híbrido."""
    texto = texto if len(texto) <= 2 else texto[:2]
    tamanho = int(lado * 0.30)
    fonte = _fonte(max(6, tamanho))
    while tamanho > 6 and desenho.textlength(texto, font=fonte) > lado * 0.34:
        tamanho -= 1
        fonte = _fonte(tamanho)
    desenho.text((centro_x, centro_y), texto, font=fonte, fill=cor, anchor="mm")


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _texto_em_imagem(texto: str, tamanho: int) -> Image.Image:
    """Renderiza um trecho de texto solto (fora de ``{...}``) como imagem."""
    fonte = _fonte(max(8, int(tamanho)))
    medidor = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    caixa = medidor.textbbox((0, 0), texto, font=fonte)
    largura = max(1, caixa[2] - caixa[0])
    altura = max(1, caixa[3] - caixa[1])
    img = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (-caixa[0], -caixa[1]), texto, font=fonte, fill=(42, 38, 30, 255)
    )
    return img


@lru_cache(maxsize=64)
def _fonte(tamanho: int):
    """Cinzel no tamanho pedido; fonte padrão do Pillow se o .ttf faltar."""
    try:
        return ImageFont.truetype(FONTE_SIMBOLO, tamanho)
    except Exception:
        try:
            return ImageFont.load_default(size=tamanho)
        except TypeError:  # Pillow antigo sem o parâmetro size
            return ImageFont.load_default()
