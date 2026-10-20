# MTG Card Generator

Projeto acadêmico (disciplina de IA e Bibliotecas de Dados para Python) que
gera cartas de *Magic: The Gathering* originais a partir de um **nome**, uma
**cor** e um **tipo** (Criatura, Instantânea, Feitiço, Artefato ou
Encantamento) escolhidos pelo usuário. O pipeline é organizado em etapas:
primeiro o tipo é decidido, depois as habilidades são geradas por uma
cadeia de Markov (bigramas/trigramas) treinada separadamente por **tipo e
cor** sobre o `oracle_text` real das cartas da Scryfall, e por fim custo de
mana e poder/resistência são previstos por regressão. O peso ajustável
(`peso_cor`) controla o quanto o vocabulário típico da cor domina a geração
(perto de 1 = fiel à cor, perto de 0 = mais caótico/genérico).

Instants e sorceries não têm habilidades ativadas nem poder/resistência;
artefatos e encantamentos não-criatura também não têm P/T. Só criaturas
recebem poder/resistência.

> **Decisão de design:** não há filtro de color pie na geração atual —
> isso é proposital (ver `src/pipeline/generator.py`). Uma primeira versão
> do projeto usava inferência de mecânica + regras rígidas de color pie
> (`src/pipeline/mecanicas.py` e `color_pie.py`, mantidos no repositório
> como registro do processo), mas optamos por geração estatística livre
> para privilegiar originalidade/surpresa sobre garantias determinísticas.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkymonks21/MTGCardCreator-IFRS/blob/main/notebooks/demo.ipynb)

**Como testar:** clique no badge acima, depois em *Ambiente de execução >
Executar tudo* (`Ctrl+F9`). Não é necessário instalar nada localmente nem
treinar nenhum modelo — o notebook usa os artefatos já treinados que ficam
versionados em `models/`.

> Status atual: obtenção dos dados (Scryfall), geração de texto via
> cadeia de Markov por tipo e cor (`src/pipeline/texto_markov.py`),
> criação de cartas de 5 tipos (criatura, instantânea, feitiço, artefato,
> encantamento) e geração de custo de mana + poder/resistência via
> regressão scikit-learn (`src/pipeline/estrutura.py`) já funcionando, com
> peso da cor ajustável. A imagem final ainda é um placeholder (Pillow).
> As próximas etapas (classificador de cor em PyTorch como métrica de
> diagnóstico, grafo de transições com NetworkX/Matplotlib, arte via API de
> imagem e composição final em OpenCV) vão preencher o que falta em
> `src/pipeline/`, sem alterar a interface do notebook.

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
├── models/
│   ├── modelo_markov.pkl          # cadeia de Markov por tipo e cor, já treinada (versionado no git)
│   └── modelos_estruturais.pkl    # regressões de custo de mana e P/T (versionado no git)
├── notebooks/
│   └── demo.ipynb              # interface do projeto: roda no Colab, chama o pipeline
├── src/
│   ├── data/
│   │   └── fetch_scryfall_data.py      # obtenção dos dados brutos
│   └── pipeline/
│       ├── generator.py                # ponto de entrada (tipo -> habilidades -> custo/P/T)
│       ├── tipos.py                    # tipos suportados e regras (P/T, habilidades ativadas)
│       ├── texto_markov.py             # geração de texto via cadeia de Markov por tipo e cor
│       ├── treinar_modelo_markov.py    # script de treino -> gera models/modelo_markov.pkl
│       ├── estrutura.py                # regressão de custo de mana e P/T (scikit-learn)
│       ├── treinar_modelos_estruturais.py  # script de treino -> models/modelos_estruturais.pkl
│       ├── mecanicas.py                # [não usado na geração] inferência de mecânica, usado só como semente opcional
│       └── color_pie.py                # [não usado na geração] regras de color pie, mantido como registro
├── requirements.txt
└── README.md
```

A separação entre `src/pipeline/` (lógica) e `notebooks/demo.ipynb`
(interface) é proposital: o notebook nunca reimplementa nada, só importa e
chama `gerar_carta(nome, cor, tipo=...)`. Isso deixa claro no código onde
está cada parte do trabalho e permite, no futuro, reaproveitar o mesmo
pipeline numa interface diferente sem duplicar lógica.

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

### 6. Treinar os modelos

```bash
python src/pipeline/treinar_modelo_markov.py
python src/pipeline/treinar_modelos_estruturais.py
```

O primeiro treina um modelo de Markov (bigramas/trigramas) para CADA tipo
de carta (criatura, instantânea, feitiço, artefato, encantamento), por cor,
sobre o `oracle_text` de `data/processed/cards.parquet`, e salva o resultado
em `models/modelo_markov.pkl` (um `ModeloTextoPorTipo`, que agrupa os
modelos de cada tipo). O segundo treina as duas regressões scikit-learn
(custo de mana -- com o tipo como feature -- e poder/resistência -- só
criaturas; ver `src/pipeline/estrutura.py`), imprime métricas de MAE/R² no
holdout e salva `models/modelos_estruturais.pkl`.

**Esses arquivos precisam ser commitados** (ao contrário de `data/`, a
pasta `models/` não está no `.gitignore`) — são eles que o Colab carrega,
sem precisar rodar os passos 4-6 de novo.

```bash
git add models/modelo_markov.pkl models/modelos_estruturais.pkl
git commit -m "Treina modelos de texto (Markov) e estruturais (custo/P-T)"
git push
```

### 7. Testar a geração localmente (opcional)

```python
import sys
sys.path.insert(0, "src")
from pipeline import gerar_carta

# tipo pode ser "Criatura", "Instantânea", "Feitiço", "Artefato",
# "Encantamento", "Aleatório" (ou None, que sorteia)
carta = gerar_carta("Murder", "Preto", peso_cor=0.7, tipo="Instantânea")
print("tipo:", carta.tipo)
print("custo de mana:", carta.custo_mana)
if carta.poder is not None:
    print("poder/resistência:", carta.poder, "/", carta.resistencia)
print("texto:", carta.texto_regras)
print("semente:", carta.semente_usada)
```

## Uso e atribuição dos dados

Os dados de cartas são fornecidos pela Scryfall sob a
[Fan Content Policy](https://company.wizards.com/en/legal/fancontent) da
Wizards of the Coast, para fins de criação de software, pesquisa e
conteúdo de comunidade sobre Magic: The Gathering. Este projeto é
acadêmico, sem fins comerciais, e os dados brutos não são redistribuídos
— cada usuário baixa sua própria cópia via `fetch_scryfall_data.py`.

## Próximos passos

- [x] ~~Mapeamento de mecânicas a partir do nome~~ — mantido em
      `src/pipeline/mecanicas.py`, hoje usado só como semente opcional
- [x] ~~Regras de color pie por cor~~ — mantido em `color_pie.py`, não
      usado na geração atual (ver decisão de design no topo deste README)
- [x] Geração de texto via cadeia de Markov por tipo e cor, com peso
      ajustável — `src/pipeline/texto_markov.py`
- [x] Tipos de carta (criatura, instantânea, feitiço, artefato,
      encantamento) com texto condicionado ao tipo; instants/sorceries sem
      habilidades ativadas e sem P/T — `src/pipeline/tipos.py`
- [x] Regressão do custo de mana (com tipo como feature) e de
      poder/resistência (só criaturas) (Scikit-Learn)
      — `src/pipeline/estrutura.py`
- [ ] Classificador de cor em PyTorch — não mais como filtro, mas como
      *métrica de diagnóstico* ("o quão on-color este texto gerado
      ficou?"), compatível com a escolha de deixar a geração livre
- [ ] Grafo de transições de palavras por cor (NetworkX) + visualizações
      (Matplotlib) — bom complemento visual pro relatório, mostrando o
      "espaço" de onde a cadeia de Markov sorteia
- [ ] Geração da arte via API de imagem
- [ ] Montagem final da carta (OpenCV)
