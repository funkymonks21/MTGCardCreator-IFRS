"""
Script de obtenção dos dados brutos de cartas de Magic: The Gathering
via Bulk Data API da Scryfall (https://scryfall.com/docs/api/bulk-data).

Uso:
    python src/data/fetch_scryfall_data.py

O que o script faz:
    1. Consulta o endpoint de metadados de bulk data da Scryfall
    2. Localiza o arquivo "Oracle Cards" (uma carta por Oracle ID, sem
       reimpressões duplicadas)
    3. Baixa o arquivo JSONL comprimido (.jsonl.gz) para
       data/raw/oracle_cards.jsonl.gz
    4. Filtra tipos de carta que não interessam ao projeto (tokens,
       emblemas, cartas de plano, Vanguard, cartas sem oracle_text)
    5. Salva um dataset limpo em data/processed/cards.csv e
       data/processed/cards.parquet, prontos para as próximas etapas
       (NLTK, Word2Vec, scikit-learn, PyTorch)

Nota sobre o formato: a Bulk Data API da Scryfall passou a servir os
arquivos como JSON Lines comprimido em gzip (campo `jsonl_download_uri`,
extensão .jsonl.gz) em vez de um único array JSON (`download_uri`, .json)
como em versões anteriores da API. O script já lida com o formato atual.
"""

import gzip
import json
from pathlib import Path

import pandas as pd
import requests

BULK_DATA_ENDPOINT = "https://api.scryfall.com/bulk-data"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# A Scryfall pede um User-Agent identificável nas requisições.
# Não é obrigatório, mas é boa prática e evita bloqueios por uso excessivo.
HEADERS = {
    "User-Agent": "MTGCardGenerator/0.1 (projeto academico)",
    "Accept": "application/json",
}

# Colunas do objeto "Card" da Scryfall que interessam ao projeto.
# Documentação completa do objeto Card: https://scryfall.com/docs/api/cards
RELEVANT_COLUMNS = [
    "id",
    "oracle_id",
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "oracle_text",
    "colors",
    "color_identity",
    "keywords",
    "power",
    "toughness",
    "loyalty",
    "rarity",
    "set",
    "set_name",
    "legalities",
]


def get_bulk_data_url(bulk_type: str = "oracle_cards") -> str:
    """Consulta o endpoint de metadados de bulk data e retorna a URL de
    download do tipo pedido (campo `jsonl_download_uri`: JSONL comprimido
    em gzip). As URLs mudam de timestamp a cada atualização, por isso
    sempre são obtidas dinamicamente, nunca hardcoded."""
    resp = requests.get(BULK_DATA_ENDPOINT, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    for item in payload["data"]:
        if item["type"] == bulk_type:
            return item["jsonl_download_uri"]

    raise ValueError(f"Tipo de bulk data '{bulk_type}' não encontrado na resposta da API.")


def download_bulk_file(url: str, destino: Path) -> Path:
    """Baixa o arquivo .jsonl.gz de bulk data (o arquivo comprimido do
    'oracle_cards' costuma ter uns 25 MB), mostrando progresso no
    terminal. O arquivo é salvo tal como vem, ainda comprimido — a
    descompressão acontece em `carregar_para_dataframe`."""
    destino.parent.mkdir(parents=True, exist_ok=True)

    print(f"Baixando dados de:\n  {url}")
    resp = requests.get(url, headers=HEADERS, timeout=180, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    baixado = 0

    with open(destino, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            baixado += len(chunk)
            if total:
                pct = baixado / total * 100
                print(f"\r  {pct:5.1f}% ({baixado / 1e6:.1f} MB)", end="")
    print(f"\nDownload concluído: {destino}")
    return destino


def carregar_para_dataframe(caminho_jsonl_gz: Path) -> pd.DataFrame:
    """Lê o arquivo .jsonl.gz (um objeto JSON por linha, comprimido em
    gzip) e retorna um DataFrame contendo apenas as colunas relevantes
    para o projeto (ignora preços, URLs de imagem em alta resolução,
    rulings, legalidades por formato detalhadas etc.).

    Processar linha a linha e já extrair só as colunas relevantes de cada
    carta (em vez de montar a lista completa de dicts e só filtrar depois)
    evita segurar na memória campos pesados que não vamos usar, como as
    URLs de imagem em várias resoluções."""
    registros = []
    with gzip.open(caminho_jsonl_gz, "rt", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            carta = json.loads(linha)
            registros.append({coluna: carta.get(coluna) for coluna in RELEVANT_COLUMNS})

    return pd.DataFrame(registros)


def filtrar_cartas_jogaveis(df: pd.DataFrame) -> pd.DataFrame:
    """Remove categorias de carta que não interessam a um gerador de
    cartas "normais": tokens, emblemas, cartas de plano, cartas de
    Vanguard, e qualquer carta sem oracle_text (texto de regras)."""
    tipos_excluir = "Token|Emblem|Plane|Scheme|Vanguard|Dungeon"
    df = df[~df["type_line"].fillna("").str.contains(tipos_excluir, regex=True)]
    df = df[df["oracle_text"].notna() & (df["oracle_text"].str.strip() != "")]

    return df.reset_index(drop=True)


def main():
    raw_path = RAW_DIR / "oracle_cards.jsonl.gz"
    processed_csv = PROCESSED_DIR / "cards.csv"
    processed_parquet = PROCESSED_DIR / "cards.parquet"

    url = get_bulk_data_url("oracle_cards")
    download_bulk_file(url, raw_path)

    print("Carregando e processando...")
    df = carregar_para_dataframe(raw_path)
    print(f"Total de cartas no arquivo bruto: {len(df)}")

    df = filtrar_cartas_jogaveis(df)
    print(f"Total de cartas após filtro: {len(df)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    df.to_parquet(processed_parquet, index=False)

    print(f"\nDados processados salvos em:\n  {processed_csv}\n  {processed_parquet}")


if __name__ == "__main__":
    main()
