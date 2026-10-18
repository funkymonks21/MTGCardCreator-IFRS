# MTG Card Generator

Projeto acadêmico (disciplina de IA e Bibliotecas de Dados para Python) que
gera cartas de *Magic: The Gathering* originais a partir de um **nome** e
uma **cor** escolhidos pelo usuário. A carta gerada é estatisticamente
coerente com o restante do jogo (custo de mana, estilo de texto) e respeita
o *color pie* (ex: azul não destrói permanentes, só dá bounce/counter).

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkymonks21/MTGCardCreator-IFRS/blob/main/notebooks/demo.ipynb)

**Como testar:** clique no badge acima, depois em *Ambiente de execução >
Executar tudo* (`Ctrl+F9`). Não é necessário instalar nada localmente nem
treinar nenhum modelo — o notebook usa os artefatos já treinados que ficam
versionados em `models/`.

> Status atual: obtenção dos dados (Scryfall) + inferência de mecânica a
> partir do nome (dicionário curado + WordNet) + regras de color pie já
> funcionando. Custo de mana e imagem final ainda são placeholders. As
> próximas etapas (embeddings treinados no corpus, regressão do custo em
> scikit-learn, classificador de cor em PyTorch e composição final em
> OpenCV) vão preencher o que falta em `src/pipeline/`, sem alterar a
> interface do notebook.

## Stack

- **Dados**: [Scryfall API](https://scryfall.com/docs/api) + Pandas / Numpy
- **Texto**: NLTK, Gensim (Word2Vec)
- **Estatística / ML**: Scikit-Learn
- **Rede neural**: PyTorch
- **Grafos**: NetworkX
- **Visualização**: Matplotlib
- **Imagem final da carta**: OpenCV
- **Arte**: API de geração de imagem (LLM)

## Estrutura do repositório

```
MTGCardCreator-IFRS/
├── data/
│   ├── raw/              # JSON bruto baixado da Scryfall (gitignored)
│   └── processed/        # cards.csv / cards.parquet prontos para uso (gitignored)
├── models/                 # artefatos pré-treinados (word2vec, regressor, classificador PyTorch...)
├── notebooks/
│   └── demo.ipynb            # interface do projeto: roda no Colab, chama o pipeline
├── src/
│   ├── data/
│   │   └── fetch_scryfall_data.py   # etapa 1: obtenção dos dados
│   └── pipeline/
│       └── generator.py              # TODA a lógica de geração mora aqui (gerar_carta)
├── requirements.txt
└── README.md
```

A separação entre `src/pipeline/` (lógica) e `notebooks/demo.ipynb`
(interface) é proposital: o notebook nunca reimplementa nada, só importa e
chama `gerar_carta(nome, cor)`. Isso deixa claro no código onde está cada
parte do trabalho e permite, no futuro, reaproveitar o mesmo pipeline numa
interface diferente sem duplicar lógica.

Os dados **não ficam versionados no repositório** (o arquivo bruto da
Scryfall passa de 100 MB). Qualquer pessoa que clonar o repositório
precisa rodar o script de obtenção localmente — veja abaixo.

## Como rodar (desenvolvimento)

> Isto é para quem vai **trabalhar no projeto** (gerar dados, treinar os
> modelos que vão para `models/`). Para só **testar o gerador de cartas**,
> use o badge do Colab lá em cima — não precisa de nada disto.

### 1. Clonar o repositório

```bash
git clone https://github.com/funkymonks21/MTGCardCreator-IFRS.git
cd MTGCardCreator-IFRS
```

### 2. Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar e processar os dados das cartas

```bash
python src/data/fetch_scryfall_data.py
```

Isso vai:
1. Consultar a [Bulk Data API da Scryfall](https://scryfall.com/docs/api/bulk-data)
   para pegar a URL atual do arquivo **Oracle Cards** (uma carta por
   Oracle ID, sem duplicar reimpressões)
2. Baixar o JSON bruto em `data/raw/oracle_cards.json`
3. Filtrar tokens, emblemas, cartas de plano/Vanguard e cartas sem texto
   de regras
4. Salvar o resultado limpo em `data/processed/cards.csv` e
   `data/processed/cards.parquet`

O processo pode levar alguns minutos dependendo da conexão (o arquivo
bruto tem ~150 MB). Ao final você deve ver algo como:

```
Total de cartas no arquivo bruto: 34000+
Total de cartas após filtro: 28000+
Dados processados salvos em:
  data/processed/cards.csv
  data/processed/cards.parquet
```

### 5. Conferir rapidamente os dados

```python
import pandas as pd
df = pd.read_parquet("data/processed/cards.parquet")
print(df.shape)
print(df[["name", "mana_cost", "colors", "oracle_text"]].head())
```

## Uso e atribuição dos dados

Os dados de cartas são fornecidos pela Scryfall sob a
[Fan Content Policy](https://company.wizards.com/en/legal/fancontent) da
Wizards of the Coast, para fins de criação de software, pesquisa e
conteúdo de comunidade sobre Magic: The Gathering. Este projeto é
acadêmico, sem fins comerciais, e os dados brutos não são redistribuídos
— cada usuário baixa sua própria cópia via `fetch_scryfall_data.py`.

## Próximos passos

- [x] Mapeamento de mecânicas a partir do nome (dicionário curado +
      similaridade via WordNet) — `src/pipeline/mecanicas.py`
- [x] Regras de color pie por cor — `src/pipeline/color_pie.py`
- [ ] Pré-processamento do `oracle_text` (limpeza, tokenização, remoção
      de reminder text)
- [ ] Treinar Word2Vec no corpus de cartas (segunda opinião além do
      WordNet, para o jargão específico do jogo)
- [ ] Regressão do custo de mana (Scikit-Learn)
- [ ] Classificador de cor em PyTorch (validação de coerência)
- [ ] Grafo de relações semânticas (NetworkX) + visualizações (Matplotlib)
- [ ] Geração da arte via API de imagem
- [ ] Montagem final da carta (OpenCV)
