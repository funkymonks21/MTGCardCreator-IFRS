"""
Script de treino do modelo de subtipos e das estatísticas das regras de
Aura/Equipamento/Terreno.

Uso (rodar da raiz do repositório, com o .venv ativado):
    python src/pipeline/treinar_modelos_subtipos.py

Pré-requisito: já ter rodado `python src/data/fetch_scryfall_data.py`,
que gera data/processed/cards.parquet.

O que o script faz:
    1. Carrega data/processed/cards.parquet
    2. Estima, a partir do corpus, as distribuições de:
       - quantidade de subtipos por tipo (com os limites do projeto:
         criatura 1-3, artefato 0-1, terreno 0-2 etc.);
       - quais subtipos aparecem, por tipo e por cor;
       - co-ocorrência entre subtipos (para combos plausíveis);
       - alvos de "Enchant ..." em Auras;
       - custos de "Equip ..." em Equipamentos.
    3. Salva o modelo em models/modelos_subtipos.pkl (versionado no git).

Como os outros artefatos, esse arquivo é carregado pelo notebook/Colab sem
precisar baixar a base da Scryfall nem treinar nada.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Adiciona src/ ao sys.path para importar através do pacote `pipeline`
# (mesmo motivo dos outros scripts de treino: imports relativos).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.subtipos import (  # noqa: E402
    MAX_SUBTIPOS,
    MIN_SUBTIPOS,
    ModeloSubtipos,
    TIPOS_BASE_EN,
)

DADOS_PROCESSADOS = Path("data/processed/cards.parquet")
MODELOS_DIR = Path("models")
SAIDA = MODELOS_DIR / "modelos_subtipos.pkl"


def main():
    if not DADOS_PROCESSADOS.exists():
        raise FileNotFoundError(
            f"{DADOS_PROCESSADOS} não encontrado. Rode antes:\n"
            f"  python src/data/fetch_scryfall_data.py"
        )

    print("Carregando cartas processadas...")
    df = pd.read_parquet(DADOS_PROCESSADOS)
    print(f"{len(df)} cartas carregadas. Estimando distribuições de subtipos...")

    modelo = ModeloSubtipos(df)
    destino = modelo.salvar(SAIDA)

    tamanho_kb = destino.stat().st_size / 1e3
    print(f"\nModelo salvo em {destino} ({tamanho_kb:.1f} kB)")

    print("\nResumo por tipo de carta:")
    for tipo in TIPOS_BASE_EN:
        contagem = modelo.contagem_global[tipo]
        total = sum(contagem.values())
        if not total:
            continue
        dist = ", ".join(
            f"{n}: {100 * freq / total:.1f}%"
            for n, freq in sorted(contagem.items())
        )
        print(
            f"  {tipo:<12} {total:>5} cartas puras | "
            f"{len(modelo.pools[tipo]):>3} subtipos | "
            f"n={MIN_SUBTIPOS[tipo]}..{MAX_SUBTIPOS[tipo]} ({dist})"
        )
        top = modelo.frequencias_global[tipo].most_common(8)
        print(f"               top: {top}")

    print("\nRegras de texto:")
    print(f"  alvos de Enchant: {modelo.alvos_enchant_global.most_common(8)}")
    print(f"  custos de Equip: {modelo.custos_equip_global.most_common(8)}")


if __name__ == "__main__":
    main()
