#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner M1 - Markov puro + regressão, sem LLM.

Gera as 1000 cartas do estudo (500 reais + 500 fictícias) usando apenas o
pipeline local. É o braço de controle: qualquer diferença entre M1 e M2 vem
exclusivamente do refinamento de texto do M2 (mesma semente -> mesma base).

Uso:
    python experimentos/run_m1.py

O arquivo `resultados_m1.jsonl` é gravado em modo append e o runner retoma de
onde parou: rode de novo se a execução for interrompida.
"""

import json
import sys

from comum import (
    EXPERIMENTOS,
    RAIZ,
    campo_carta,
    carregar_itens,
    cor_do_pipeline,
    criar_log,
    ordens_concluidas,
)

sys.path.insert(0, str(RAIZ / "src"))
from pipeline import gerar_carta  # noqa: E402

OUT = EXPERIMENTOS / "resultados_m1.jsonl"
LOG = EXPERIMENTOS / "log_m1.txt"


def main() -> None:
    log = criar_log(LOG)
    itens = carregar_itens()
    concluidas = ordens_concluidas(OUT)
    log(f"Iniciando M1: {len(itens)} cartas | {len(concluidas)} já concluídas")

    erros = 0
    with open(OUT, "a", encoding="utf-8") as fout:
        for item in itens:
            if item["ordem"] in concluidas:
                continue
            try:
                carta = gerar_carta(
                    nome=item["nome"],
                    cor=cor_do_pipeline(item["cor"]),
                    tipo=item["tipo_forcado"],
                    seed=item["seed"],
                    usar_arte=False,
                )
                registro = {
                    "modelo": "M1",
                    "ordem": item["ordem"],
                    "carta_id": item["carta_id"],
                    "tipo_experimento": item["tipo_experimento"],
                    "nome_entrada": item["nome"],
                    "cor_entrada": item["cor"],
                    "tipo_forcado": item["tipo_forcado"],
                    "seed": item["seed"],
                    **campo_carta(carta),
                    "gabarito": item["gabarito"],
                }
                fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as exc:  # noqa: BLE001 - falha por carta é métrica
                erros += 1
                log(f"ERRO carta {item['ordem']} ({item['nome']}): {exc}")
                fout.write(
                    json.dumps(
                        {
                            "modelo": "M1",
                            "ordem": item["ordem"],
                            "carta_id": item["carta_id"],
                            "tipo_experimento": item["tipo_experimento"],
                            "erro": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

            if (item["ordem"] + 1) % 50 == 0:
                log(f"Progresso: {item['ordem'] + 1}/{len(itens)} | erros={erros}")

    log(f"Concluido M1. erros={erros}. Resultado em {OUT}")


if __name__ == "__main__":
    main()
