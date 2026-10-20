"""
Script de treino dos modelos "estruturais" (scikit-learn): custo de mana e
poder/resistência.

Uso (rodar da raiz do repositório, com o .venv ativado):
    python src/pipeline/treinar_modelos_estruturais.py

Pré-requisito: já ter rodado `python src/data/fetch_scryfall_data.py`,
que gera data/processed/cards.parquet.

O que o script faz:
    1. Carrega data/processed/cards.parquet
    2. Separa treino/teste (holdout) e imprime métricas (MAE e R²) dos
       dois modelos -- úteis para o relatório do projeto
    3. Re-treina cada modelo com TODOS os dados disponíveis
    4. Salva os dois pipelines juntos em models/modelos_estruturais.pkl
       (joblib), que é versionado no git e carregado pelo notebook.

Como o modelo de texto (treinar_modelo_markov.py), esse artefato É
versionado no git -- é ele que o Colab carrega, sem precisar baixar a base
da Scryfall nem treinar nada.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

## Adiciona `src/` ao sys.path para importar através do pacote `pipeline`
## (mesmo motivo do treinar_modelo_markov.py: os imports relativos só
## funcionam se o módulo for importado como parte do pacote).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.estrutura import (  # noqa: E402
    ModeloEstrutural,
    construir_pipeline_custo,
    construir_pipeline_pt,
    preparar_dados_custo,
    preparar_dados_pt,
)

DADOS_PROCESSADOS = Path("data/processed/cards.parquet")
MODELOS_DIR = Path("models")
SAIDA = MODELOS_DIR / "modelos_estruturais.pkl"


def _avaliar(nome: str, pipeline, X, y, alvos) -> None:
    """Treina `pipeline` no split de treino, avalia no split de teste e
    imprime MAE/R² por alvo (holdout de 20%)."""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_treino, y_treino)
    previsao = pipeline.predict(X_teste)

    print(f"\n[{nome}] holdout de {len(X_teste)} cartas")
    for i, alvo in enumerate(alvos):
        mae = mean_absolute_error(y_teste.iloc[:, i], previsao[:, i])
        r2 = r2_score(y_teste.iloc[:, i], previsao[:, i])
        print(f"  {alvo:<12} MAE = {mae:5.3f}   R² = {r2:5.3f}")


def main():
    if not DADOS_PROCESSADOS.exists():
        raise FileNotFoundError(
            f"{DADOS_PROCESSADOS} não encontrado. Rode antes:\n"
            f"  python src/data/fetch_scryfall_data.py"
        )

    print("Carregando cartas processadas...")
    df = pd.read_parquet(DADOS_PROCESSADOS)
    print(f"{len(df)} cartas carregadas.")

    X_custo, y_custo = preparar_dados_custo(df)
    X_pt, y_pt = preparar_dados_pt(df)
    print(f"Cartas usadas no modelo de custo: {len(X_custo)} (monocolor, mana simples)")
    print(f"  por tipo: {X_custo['tipo'].value_counts().to_dict()}")
    print(f"Criaturas usadas no modelo de P/T: {len(X_pt)} (P/T numérico)")

    # 1) Avaliação em holdout, só para diagnóstico/relatório.
    _avaliar("custo de mana", construir_pipeline_custo(), X_custo, y_custo, ["cmc", "pips"])
    _avaliar("poder/resistência", construir_pipeline_pt(), X_pt, y_pt, ["poder", "resistência"])

    # 2) Treino final com todos os dados (é o modelo que vai para o artefato).
    print("\nTreinando os modelos finais com todos os dados...")
    modelo_custo = construir_pipeline_custo().fit(X_custo, y_custo)
    modelo_pt = construir_pipeline_pt().fit(X_pt, y_pt)

    modelo = ModeloEstrutural(modelo_custo, modelo_pt)
    destino = modelo.salvar(SAIDA)

    tamanho_mb = destino.stat().st_size / 1e6
    print(f"\nModelos salvos em {destino} ({tamanho_mb:.1f} MB)")


if __name__ == "__main__":
    main()
