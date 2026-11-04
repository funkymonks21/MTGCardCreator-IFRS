#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""App da demo: a mesma interface para a aula e para quem quiser brincar.

Por que Gradio
==============
O notebook (`notebooks/demo.ipynb`) é ótimo para a apresentação, mas exige
alguém com o Colab aberto executando células. O Gradio transforma a mesma
geração em uma página web: roda localmente (`python app.py`) ou no Colab com
`share=True`, que devolve um link público temporário -- ninguém precisa
instalar Python, clonar o repositório ou ter o Ollama na máquina.

O app NÃO reimplementa nada: chama `pipeline.llm.gerar_com_modelo`, o mesmo
dispatch que o notebook usa. Os prompts, temperaturas e validações do M2/M3
continuam sendo os medidos no estudo.

Três braços, um seletor
=======================
- **M1** — pipeline (Markov + regressões). Offline, instantâneo.
- **M2** — o M1 com `texto_regras` refinado por LLM. Mesma semente = mesma
  estrutura do M1, o que deixa a diferença de texto visível.
- **M3** — LLM pura (`gemma3:4b`), com o tipo resolvido pelo pipeline e
  imposto ao modelo (desenho do Modo A do estudo).

O M2/M3 exigem o Ollama. Sem ele, o app abre normalmente e só o M1 funciona;
o aviso na tela diz exatamente o que subir.

Como rodar
==========
Local (com Ollama no ar):

    pip install -r requirements-app.txt
    python app.py

Local, link público temporário (túnel do Gradio, ~72h):

    python app.py --share

Colab: ver `notebooks/app_colab.ipynb` (instala o Ollama, baixa o
`gemma3:4b` e sobe o app com link público).

Configuração por ambiente (mesmas variáveis do pipeline):

    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=gemma3:4b
    GRADIO_SHARE=1          # equivale a --share
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# O pipeline carrega os modelos por caminho relativo ("models/..."): o app
# precisa rodar da raiz do repositório, mesmo que seja chamado de outro
# diretório.
RAIZ = Path(__file__).resolve().parent
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ / "src"))

import gradio as gr  # noqa: E402

from pipeline import (  # noqa: E402
    COLOR_PIE,
    MODELOS,
    PESO_COR_PADRAO,
    PESOS_NGRAMA_PADRAO,
    ROTULO_MODELO,
    TIPO_AUTOMATICO,
    TIPOS_DISPONIVEIS,
    FalhaGeracao,
    OllamaIndisponivel,
    gerar_com_modelo,
    ollama_disponivel,
    refinar_como_m2,
    renderizar_carta,
)

# Valores de semente usados pelos exemplos: os mesmos do estudo, para que a
# carta que aparece no app seja a mesma dos resultados publicados.
SEED_EXEMPLO = 2634650529

DICA_MODELO = {
    "M1": "Texto por cadeia de Markov; custo e P/T pelas regressões. "
          "Roda offline.",
    "M2": "A estrutura é a do pipeline (mesma semente = mesma carta do M1); "
          "só o texto de regras passa pelo refino do LLM.",
    "M3": "O texto vem inteiro do LLM. Peso da cor, parágrafos e n-gramas "
          "não se aplicam; nome, cor e tipo (resolvido pelo pipeline) são "
          "impostos ao modelo.",
}


# ---------------------------------------------------------------------------
# Texto auxiliar
# ---------------------------------------------------------------------------


def _resumo_ollama() -> str:
    """Badge de status do Ollama (HTML, porque Markdown não colore)."""
    if ollama_disponivel():
        return (
            "<span style='color:#1a7f37;font-weight:600'>● Ollama no ar</span> "
            "<span style='color:#6b6459'>— M1, M2 e M3 prontos.</span>"
        )
    return (
        "<span style='color:#b35900;font-weight:600'>● Ollama indisponível</span> "
        "<span style='color:#6b6459'>— só o M1 funciona. Para liberar M2/M3: "
        "<code>ollama serve</code> e <code>ollama pull gemma3:4b</code>.</span>"
    )


def _detalhes(carta, avisos, nome_pedido: str = "") -> str:
    """Ficha da carta em Markdown -- o que a apresentação precisa mostrar."""
    linhas = [f"### {carta.nome}", ""]
    pedido = (nome_pedido or "").strip()
    if pedido and pedido != carta.nome:
        # O M3 trata o nome como TEMA: o schema pede que o modelo crie o
        # próprio nome (é o desenho medido no estudo). Mostrar o pedido
        # evita a impressão de que a entrada foi ignorada.
        linhas += [f"**Nome pedido:** {pedido}", ""]
    linhas += [
        f"**Modelo:** {ROTULO_MODELO.get(carta.modelo, carta.modelo)}",
        f"**Tipo:** {carta.tipo_linha or carta.tipo}",
        f"**Custo de mana:** {carta.custo_mana or '— (terreno)'}",
    ]
    if carta.poder is not None:
        linhas.append(f"**Poder/Resistência:** {carta.poder}/{carta.resistencia}")
    if carta.palavras_chave:
        linhas.append(f"**Palavras-chave:** {', '.join(carta.palavras_chave)}")
    linhas += [
        f"**Semente:** {carta.semente_usada or '—'}",
        "",
        f"**Como o tipo foi decidido:** {carta.tipo_origem} — {carta.tipo_motivo}",
    ]
    if carta.modelo == "M2":
        estado = "sim" if carta.refinado else "não (texto do pipeline mantido)"
        linhas.append(f"**Refino aplicado:** {estado}")
    if carta.arte_da_api is not None:
        origem = "API (Pollinations)" if carta.arte_da_api else "fallback local"
        detalhe = f" — {carta.arte_motivo}" if carta.arte_motivo else ""
        linhas.append(f"**Arte:** {origem}{detalhe}")
    else:
        linhas.append("**Arte:** procedural local (a API não foi chamada)")
    if avisos:
        linhas += ["", f"> ⚠️ {' '.join(avisos)}"]
    linhas += [
        "",
        "---",
        "",
        "**Texto de regras**",
        "",
        (carta.texto_regras or "_sem texto_").replace("\n", "  \n"),
    ]
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------


def _seed(valor):
    """-1 (ou vazio) sorteia; qualquer outro inteiro vira semente fixa."""
    try:
        n = int(valor)
    except (TypeError, ValueError):
        return None
    return None if n < 0 else n


def _pesos_ngrama(*valores):
    return {ordem: float(v) for ordem, v in enumerate(valores)}


def gerar(
    modelo, nome, cor, tipo, seed, peso_cor, n_paragrafos, usar_arte, *ngramas
):
    """Callback do botão: gera e devolve (imagem, ficha).

    As falhas de M2/M3 viram `gr.Error`, que aparece como toast -- em vez de
    derrubar a interface ou deixar a coluna da carta vazia sem explicação.
    """
    avisos = []
    try:
        carta = gerar_com_modelo(
            modelo,
            nome,
            cor,
            peso_cor=float(peso_cor),
            tipo=tipo,
            pesos_ngrama=_pesos_ngrama(*ngramas),
            n_paragrafos=None if n_paragrafos in (None, "", "Automatico") else int(n_paragrafos),
            usar_arte=bool(usar_arte),
            avisar_arte=avisos.append,
            seed=_seed(seed),
        )
    except OllamaIndisponivel as exc:
        raise gr.Error(
            "O M2/M3 precisa do Ollama, e a chamada não funcionou.\n\n"
            f"{exc}\n\n"
            "Sem Ollama, use o M1: a geração continua funcionando offline."
        )
    except FalhaGeracao as exc:
        raise gr.Error(
            f"O M3 não produziu uma carta válida nesta tentativa.\n\n{exc}\n\n"
            "A amostragem do LLM é estocástica: clique de novo para tentar outra."
        )
    return (
        carta.imagem_final,
        _detalhes(carta, avisos, nome),
        (carta, nome),
        gr.update(interactive=carta.modelo == "M1"),
        gr.update(interactive=True),
    )


def refinar(estado):
    """Refina o texto de uma carta do M1 já exibida -- vira a versão M2.

    É o caminho "gerar no M1, ler o texto e só então mandar refinar": a
    estrutura e o resto da ficha não mudam, só `texto_regras`.
    """
    if not estado:
        raise gr.Error("Gere uma carta primeiro.")
    carta, pedido = estado
    if carta.modelo != "M1":
        raise gr.Error(
            "O refino do M2 parte do texto do pipeline. Gere uma carta no M1 "
            "para refinar — cartas do M3 já vêm inteiras do LLM."
        )

    avisos = []
    try:
        nova = refinar_como_m2(carta, avisar=avisos.append)
    except OllamaIndisponivel as exc:
        raise gr.Error(
            "O refino precisa do Ollama, e a chamada não funcionou.\n\n"
            f"{exc}"
        )
    return (
        nova.imagem_final,
        _detalhes(nova, avisos, pedido),
        (nova, pedido),
        gr.update(interactive=False),
    )


def gerar_arte(estado):
    """Gera a arte de uma carta já exibida, de qualquer modelo.

    A carta nasce sem arte (rápido, offline); este botão é o "gostei do
    texto, quero ver com ilustração". Se a API falhar, o fallback procedural
    entra e o motivo aparece na ficha.
    """
    if not estado:
        raise gr.Error("Gere uma carta primeiro.")
    carta, pedido = estado
    avisos = []
    renderizar_carta(carta, usar_arte=True, avisar_arte=avisos.append)
    return carta.imagem_final, _detalhes(carta, avisos, pedido), (carta, pedido)


def comparar(nome, cor, seed, usar_arte, peso_cor, n_paragrafos, *ngramas):
    """Gera os três braços com a MESMA entrada, lado a lado.

    É o argumento da apresentação em uma tela: o M1 e o M2 compartilham a
    semente (estrutura idêntica), e o M3 recebe o mesmo nome, cor e tipo.

    Os pesos (cor e n-gramas) e o número de parágrafos só afetam M1/M2; no
    M3 o texto vem inteiro do LLM. Os valores usados aparecem no rodapé.
    """
    imagens, linhas = [], []
    avisos = []
    seed_int = _seed(seed)
    pesos = _pesos_ngrama(*ngramas)
    paragrafos = None if n_paragrafos in (None, "", "Automatico") else int(n_paragrafos)
    linhas += [f"**Nome pedido:** {nome} · **cor:** {cor}", ""]

    for modelo in MODELOS:
        try:
            carta = gerar_com_modelo(
                modelo,
                nome,
                cor,
                peso_cor=float(peso_cor),
                tipo=TIPO_AUTOMATICO,
                pesos_ngrama=pesos,
                n_paragrafos=paragrafos,
                usar_arte=bool(usar_arte),
                avisar_arte=avisos.append,
                seed=seed_int,
            )
        except OllamaIndisponivel as exc:
            raise gr.Error(
                "A comparação precisa dos três braços, e o M2/M3 exigem o "
                f"Ollama.\n\n{exc}"
            )
        except FalhaGeracao as exc:
            raise gr.Error(f"O M3 falhou nesta rodada.\n\n{exc}\n\nTente outra semente.")
        imagens.append((carta.imagem_final, modelo))
        linhas.append(
            f"**{modelo}** — {carta.tipo_linha or carta.tipo} · "
            f"custo `{carta.custo_mana or '—'}`"
            + (f" · {carta.poder}/{carta.resistencia}" if carta.poder is not None else "")
        )
        linhas.append("")
        linhas.append(carta.texto_regras or "_sem texto_")
        linhas.append("")

    if avisos:
        linhas.append(f"> ⚠️ {' '.join(avisos)}")
    pesos_txt = ", ".join(f"{ordem}: {v:.2f}" for ordem, v in sorted(pesos.items()))
    linhas += [
        "",
        f"_Parâmetros usados: tipo automático · peso da cor {float(peso_cor):.2f} · "
        f"parágrafos {paragrafos or 'automático'} · n-gramas {{{pesos_txt}}}_",
    ]
    return imagens, "\n".join(linhas)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def criar_app() -> gr.Blocks:
    """Monta a página. Separado de `main()` para o Colab reaproveitar."""
    with gr.Blocks(title="Gerador de cartas — M1/M2/M3") as demo:
        # Guarda a última carta gerada (por sessão) para as ações "depois do
        # fato": refinar o texto e gerar a arte sem regerar a carta.
        estado = gr.State()
        gr.Markdown(
            "# Gerador de cartas de Magic\n"
            "O **mesmo** pipeline em três braços: **M1** (Markov + regressões), "
            "**M2** (M1 com o texto refinado por LLM) e **M3** (LLM pura). "
            "Os prompts e a validação são os do estudo — o que aparece aqui é "
            "o que foi medido."
        )

        with gr.Tab("Gerar carta"):
            with gr.Row():
                with gr.Column(scale=5):
                    status = gr.HTML(_resumo_ollama())
                    botao_status = gr.Button("Verificar Ollama", size="sm")

                    modelo = gr.Radio(
                        choices=[(ROTULO_MODELO[m], m) for m in MODELOS],
                        value="M1",
                        label="Modelo",
                    )
                    dica = gr.Markdown(DICA_MODELO["M1"])

                    nome = gr.Textbox(
                        value="Serra Angel",
                        label="Nome da carta",
                        placeholder="ex.: Murder, Serra Angel, Mind Stone, ...",
                    )
                    with gr.Row():
                        cor = gr.Dropdown(
                            choices=list(COLOR_PIE), value="Branco", label="Cor"
                        )
                        tipo = gr.Dropdown(
                            choices=list(TIPOS_DISPONIVEIS),
                            value=TIPO_AUTOMATICO,
                            label="Tipo",
                        )

                    with gr.Row():
                        seed = gr.Number(
                            value=-1, precision=0, label="Semente (-1 = aleatória)"
                        )
                        peso_cor = gr.Slider(
                            0.0, 1.0, value=PESO_COR_PADRAO, step=0.1,
                            label="Peso da cor",
                        )
                    n_paragrafos = gr.Dropdown(
                        choices=[
                            ("Automático (pelos dados)", "Automatico"),
                            ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"),
                        ],
                        value="Automatico",
                        label="Parágrafos de habilidade",
                    )
                    usar_arte = gr.Checkbox(
                        value=True,
                        label="Arte pela API (Pollinations, ~10–30s)",
                        info="Sem chave. Desligado: arte procedural local, "
                             "sem internet. Se o limite gratuito for atingido, "
                             "a carta usa o fallback e avisa.",
                    )

                    with gr.Accordion("Pesos dos n-gramas (só M1/M2)", open=False):
                        gr.Markdown(
                            "Peso de cada ordem na escolha da próxima palavra. "
                            "Os pesos são relativos entre as ordens disponíveis."
                        )
                        ngramas = [
                            gr.Slider(
                                0.0, 1.0, value=PESOS_NGRAMA_PADRAO.get(o, 0.0),
                                step=0.05, label=rotulo,
                            )
                            for o, rotulo in {
                                0: "Unigrama (0)", 1: "Bigrama (1)", 2: "Trigrama (2)",
                                3: "Tetragrama (3)", 4: "Pentagrama (4)",
                            }.items()
                        ]

                    botao = gr.Button("Gerar carta", variant="primary", size="lg")

                with gr.Column(scale=6):
                    imagem = gr.Image(
                        label="Carta gerada", type="pil", height=520,
                        buttons=["download", "fullscreen"],
                    )
                    with gr.Row():
                        botao_refinar = gr.Button("Refinar texto (M2)", interactive=False)
                        botao_arte = gr.Button("Gerar arte", interactive=False)
                    gr.Markdown(
                        "<small>**Refinar texto** só vale para cartas do M1. "
                        "**Gerar arte** vale para qualquer modelo.</small>"
                    )
                    ficha = gr.Markdown("_Clique em **Gerar carta**._")

            entradas = [modelo, nome, cor, tipo, seed, peso_cor, n_paragrafos, usar_arte, *ngramas]
            saidas = [imagem, ficha, estado, botao_refinar, botao_arte]
            botao.click(gerar, inputs=entradas, outputs=saidas)
            nome.submit(gerar, inputs=entradas, outputs=saidas)
            botao_refinar.click(
                refinar,
                inputs=estado,
                outputs=[imagem, ficha, estado, botao_refinar],
            )
            botao_arte.click(
                gerar_arte,
                inputs=estado,
                outputs=[imagem, ficha, estado],
            )
            botao_status.click(lambda: _resumo_ollama(), outputs=status)

            def ao_trocar_modelo(m):
                # No M3 o texto vem inteiro do LLM: os controles de texto do
                # pipeline ficam inertes (e visivelmente apagados).
                ativo = m != "M3"
                return (
                    DICA_MODELO.get(m, ""),
                    gr.update(interactive=ativo),   # peso da cor
                    gr.update(interactive=ativo),   # parágrafos
                    *[gr.update(interactive=ativo) for _ in ngramas],
                )

            modelo.change(
                ao_trocar_modelo,
                inputs=modelo,
                outputs=[dica, peso_cor, n_paragrafos, *ngramas],
            )

        with gr.Tab("Comparar os três"):
            gr.Markdown(
                "A mesma entrada nos três braços. **M1 e M2 compartilham a "
                "semente**, então a estrutura é idêntica — a diferença que "
                "aparece é a do texto. O M3 recebe o mesmo nome, cor e tipo.\n\n"
                "_Os pesos e o número de parágrafos valem só para **M1/M2**: "
                "no M3 o texto vem inteiro do LLM. Esta aba tem controles "
                "próprios, independentes da aba **Gerar carta**._"
            )
            with gr.Row():
                c_nome = gr.Textbox(value="Serra Angel", label="Nome da carta")
                c_cor = gr.Dropdown(choices=list(COLOR_PIE), value="Branco", label="Cor")
                c_seed = gr.Number(value=SEED_EXEMPLO, precision=0, label="Semente")
            with gr.Row():
                c_peso = gr.Slider(
                    0.0, 1.0, value=PESO_COR_PADRAO, step=0.1,
                    label="Peso da cor (M1/M2)",
                )
                c_parag = gr.Dropdown(
                    choices=[
                        ("Automático (pelos dados)", "Automatico"),
                        ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"),
                    ],
                    value="Automatico",
                    label="Parágrafos de habilidade (M1/M2)",
                )
                c_arte = gr.Checkbox(
                    value=False,
                    label="Arte pela API (desligado = procedural)",
                )
            with gr.Accordion("Pesos dos n-gramas (M1/M2)", open=False):
                c_ngramas = [
                    gr.Slider(
                        0.0, 1.0, value=PESOS_NGRAMA_PADRAO.get(o, 0.0),
                        step=0.05, label=rotulo,
                    )
                    for o, rotulo in {
                        0: "Unigrama (0)", 1: "Bigrama (1)", 2: "Trigrama (2)",
                        3: "Tetragrama (3)", 4: "Pentagrama (4)",
                    }.items()
                ]
            c_botao = gr.Button("Gerar os três", variant="primary")
            c_galeria = gr.Gallery(
                label="M1 · M2 · M3", columns=3, height=420,
                object_fit="contain", buttons=["download", "fullscreen"],
            )
            c_textos = gr.Markdown()

            c_botao.click(
                comparar,
                inputs=[c_nome, c_cor, c_seed, c_arte, c_peso, c_parag, *c_ngramas],
                outputs=[c_galeria, c_textos],
            )

        gr.Markdown(
            "---\n"
            "**Fan-made / não oficial.** A moldura e a arte procedural são "
            "desenhadas em código e as fontes são livres (SIL OFL). Os "
            "desenhos dos símbolos de mana são marca da Wizards of the "
            "Coast e aparecem aqui via fonte Mana (OFL), como acabamento de "
            "um trabalho acadêmico sem fins comerciais. Nenhum template ou "
            "arte oficial é usado.\n\n"
            "Estudo: comparação M1/M2/M3 com cartas reais × fictícias, "
            "split holdout/reserva (D-006) e avaliação humana. "
            "Código e números em `experimentos/`."
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo M1/M2/M3 em Gradio.")
    parser.add_argument("--share", action="store_true", help="cria um link público temporário")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    compartilhar = args.share or os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "sim")
    demo = criar_app()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=compartilhar,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
