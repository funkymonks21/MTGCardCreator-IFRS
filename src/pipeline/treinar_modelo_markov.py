"""
Script de treino do modelo de geração de texto (cadeia de Markov por cor).

Uso (rodar da raiz do repositório, com o .venv ativado):
    python src/pipeline/treinar_modelo_markov.py

Pré-requisito: já ter rodado `python src/data/fetch_scryfall_data.py`,
que gera data/processed/cards.parquet.

O que o script faz:
    1. Carrega data/processed/cards.parquet
    2. Treina um ModeloMarkovPorCor (bigramas/trigramas por cor + modelo
       global, ver pipeline/texto_markov.py)
    3. Salva o modelo treinado em models/modelo_markov.pkl

Esse artefato treinado (ao contrário de data/) É versionado no git -- é
ele que o notebook do Colab carrega para gerar cartas, sem que o usuário
final precise baixar a base inteira da Scryfall nem treinar nada.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

## Adiciona a pasta `src/` (duas pastas acima deste arquivo: src/pipeline/..)
## ao sys.path, e importa `texto_markov` ATRAVÉS do pacote `pipeline`
## (`from pipeline.texto_markov import ...`), não como módulo solto.
## Isso é necessário porque texto_markov.py faz `from .mecanicas import
## ...` (import relativo) -- esse import só funciona se o Python souber
## que texto_markov faz parte do pacote `pipeline`, o que exige importar
## via `pipeline.texto_markov`, nunca `import texto_markov` direto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.texto_markov import ModeloMarkovPorCor  # noqa: E402

DADOS_PROCESSADOS = Path("data/processed/cards.parquet")
MODELOS_DIR = Path("models")
SAIDA = MODELOS_DIR / "modelo_markov.pkl"


def main():
    if not DADOS_PROCESSADOS.exists():
        raise FileNotFoundError(
            f"{DADOS_PROCESSADOS} não encontrado. Rode antes:\n"
            f"  python src/data/fetch_scryfall_data.py"
        )

    print("Carregando cartas processadas...")
    df = pd.read_parquet(DADOS_PROCESSADOS)
    print(f"{len(df)} cartas carregadas. Treinando modelo por cor (pode levar um tempo)...")

    modelo = ModeloMarkovPorCor(df)

    MODELOS_DIR.mkdir(exist_ok=True)
    with open(SAIDA, "wb") as f:
        pickle.dump(modelo, f)

    tamanho_mb = SAIDA.stat().st_size / 1e6
    print(f"\nModelo salvo em {SAIDA} ({tamanho_mb:.1f} MB)")
    print("\nResumo por cor:")
    for cor, unigramas in modelo.unigramas_por_cor.items():
        total_tokens = sum(unigramas.values())
        print(f"  {cor}: {total_tokens} tokens, vocabulário de {len(unigramas)} palavras")


if __name__ == "__main__":
    main()
