# MTG Card Generator

Projeto acadêmico (disciplina de IA e Bibliotecas de Dados para Python) que
gera cartas de *Magic: The Gathering* originais a partir de um **nome**, uma
**cor** e um **tipo** (Criatura, Instantânea, Feitiço, Artefato,
Encantamento ou Terreno). O tipo pode ser escolhido pelo usuário, inferido
a partir do nome ou sorteado. O pipeline é organizado em etapas: primeiro o
tipo e os **subtipos** são decididos, depois as habilidades são geradas por
uma cadeia de Markov (unigramas até pentagramas) treinada separadamente por
**tipo e cor** sobre o `oracle_text` real das cartas da Scryfall, e por fim
custo de mana e poder/resistência são previstos por regressão. Os pesos
ajustáveis controlam a geração do texto: `peso_cor` decide o quanto o
vocabulário típico da cor domina a geração (perto de 1 = fiel à cor, perto
de 0 = mais caótico/genérico), e `pesos_ngrama` dá um peso para **cada ordem
de n-grama** (unigrama, bigrama, trigrama, tetragrama e pentagrama). Contextos
mais longos só existem quando aquela sequência apareceu no corpus, então são
o sinal mais forte de coerência — por isso o padrão
(`PESOS_NGRAMA_PADRAO = {4: 1.00, 3: 0.70, 2: 0.40, 1: 0.15, 0: 0.02}`)
favorece tetragrama/pentagrama. O atalho legado `peso_trigrama` (interpola só
trigrama e bigrama) continua aceito para compatibilidade.

Os subtipos também vêm dos dados: as quantidades e os nomes são estimados
do corpus, condicionados pelo tipo e pela cor. Criaturas recebem 1–3
subtipos de criatura, artefatos 0–1 subtipo de artefato, terrenos 0–2
subtipos de terreno, encantamentos 0–2, e instantâneas/feitiços 0–1.
Auras sempre ganham uma linha `Enchant ...`; Equipamentos sempre ganham uma
linha `Equipped creature ...` (o que a criatura equipada recebe) e a linha
`Equip ...` (o custo de equipar); **Veículos** sempre ganham `Crew N` (N
sorteado dos Veículos reais) **e** poder/resistência; **Sagas** têm cada
parágrafo transformado num capítulo com numeral romano incremental (`I — ...`,
`II — ...`), com a quantidade de capítulos sorteada da distribuição real das
Sagas (2 e 3 são os mais comuns); e terrenos com tipo básico (Plains, Island,
Swamp, Mountain, Forest) ganham a habilidade de mana correspondente. Os
subtipos ficam em inglês, como no `type_line` original da Scryfall, para
manter a coerência com o texto Oracle gerado (ex.: `Creature — Human
Warrior`). Assim como no texto, as distribuições de subtipos são misturadas
com um modelo global pelo mesmo `peso_cor`.

> **Veículos e P/T:** Veículos (`Artifact — Vehicle`) têm poder/resistência,
> mas **não** passam pela regressão de P/T (que é treinada só em cartas com
> `Creature` no type_line). O P/T do Veículo é sorteado direto dos Veículos
> reais do corpus.
>
> **Sagas:** o modelo de Markov às vezes já emite um numeral romano no início
> de um parágrafo (aprendeu o formato real das Sagas); o lint remove esse
> prefixo antes de renumerar, e a linha solta de keywords é descartada para
> não virar um capítulo.

A **quantidade de parágrafos** de habilidade e a **quantidade de keywords**
de cada carta também vêm dos dados: o número de parágrafos é sorteado da
distribuição de probabilidade real do tipo (ex.: criaturas costumam ter 1–2,
terrenos 2–3) e cada parágrafo é gerado como uma habilidade independente. Os
keywords (Flying, Trample etc.) são sorteados a partir da frequência real,
com um fator (`FATOR_KEYWORDS`) que deixa a carta com mais keywords que a
média do corpus, já que o número real é baixo.

Instants e sorceries não têm habilidades ativadas nem poder/resistência;
artefatos, encantamentos e terrenos não-criatura também não têm P/T, e
terrenos não têm custo de mana. Só criaturas recebem poder/resistência.

> **Artefatos incolores:** a maioria dos artefatos de Magic (76% do corpus) é
> incolor e, portanto, não pertence a nenhuma cor — eles só entram no modelo
> **global**. Para o tipo `artifact`, o `peso_cor` pedido é reduzido
> (`AJUSTE_PESO_COR_POR_TIPO`) para dar mais voz ao modelo global, que é o
> vocabulário mais representativo de artefato.

**Arte e moldura** (`src/pipeline/imagem.py`): com `usar_arte=True` (o
checkbox do notebook), a carta ganha **ilustração** e **moldura** de verdade.
O pipeline monta um prompt a partir do **nome + tipo + texto** (a cor define a
paleta; o tipo define a cena), chama uma API de imagem gratuita e **sem
chave** ([Pollinations.ai](https://pollinations.ai/)) e desenha a moldura em
código — borda preta com cantos arredondados, painel do título com gradiente
da cor, janela da arte com vinheta, caixa de tipo, caixa de texto translúcida
e selo de poder/resistência. As fontes são livres (Cinzel para títulos, EB
Garamond para o corpo — ambas SIL OFL, em `assets/fonts/`); a Beleren oficial
é proprietária e **não** é distribuída aqui.

- **Cache:** artes já baixadas ficam em `assets/cache_arte/` (fora do git), o
  que torna a mesma carta instantânea ao recalcular.
- **Fallback:** se a API falhar (sem internet, timeout, serviço fora do ar), a
  carta ainda é gerada com um fundo procedural determinístico — a geração
  **nunca** quebra por causa da arte.
- **Seed:** a arte é reprodutível (mesma seed/prompt → mesma imagem); o
  parâmetro `seed_arte` de `gerar_carta` permite fixá-la.
- **Trocando de provedor:** `ArtePollinations` isola a chamada HTTP no
  contrato `gerar(prompt, largura, altura, seed)`; trocar por Hugging Face,
  Stability etc. é implementar a mesma assinatura.

Sem `usar_arte`, a figura sai num placeholder simples (rápido, offline), que é
o que a geração em lote usa por padrão.

**Inferência do tipo pelo nome** (`src/pipeline/tipo_por_nome.py`): com a
opção `Automático (pelo nome)`, o sistema usa POS tagging do NLTK + WordNet
(com listas curadas em inglês e português) para achar pistas no nome:
verbos/ações puxam instantânea/feitiço, lugares puxam terreno,
profissões/nomes próprios puxam criatura, objetos puxam artefato e conceitos
abstratos puxam encantamento. Em caso de empate ou de nenhuma pista, o tipo
é sorteado. É uma heurística, não um classificador treinado: nomes ambíguos
podem cair num tipo diferente do que a carta real de Magic teria.

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
> criação de cartas de 6 tipos (criatura, instantânea, feitiço, artefato,
> encantamento, terreno), inferência do tipo pelo nome
> (`src/pipeline/tipo_por_nome.py`), subtipos data-driven
> (`src/pipeline/subtipos.py`), regras de Aura/Equipamento/Veículo/Saga/Terreno
> (`src/pipeline/regras_tipo.py`), geração de custo de mana +
> poder/resistência via regressão scikit-learn (`src/pipeline/estrutura.py`)
> e **arte via API de imagem + moldura realista em código**
> (`src/pipeline/imagem.py`) já funcionando, com peso da cor ajustável.
> As próximas etapas (classificador de cor em PyTorch como métrica de
> diagnóstico, grafo de transições com NetworkX/Matplotlib) vão preencher o
> que falta em `src/pipeline/`, sem alterar a interface do notebook.

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
│   ├── modelos_estruturais.pkl    # regressões de custo de mana e P/T (versionado no git)
│   └── modelos_subtipos.pkl       # distribuições de subtipos + regras de Aura/Equip/Equipped/Crew/Saga (versionado no git)
├── notebooks/
│   └── demo.ipynb              # interface do projeto: roda no Colab, chama o pipeline
├── src/
│   ├── data/
│   │   └── fetch_scryfall_data.py      # obtenção dos dados brutos
│   └── pipeline/
│       ├── generator.py                # ponto de entrada (tipo -> subtipos -> habilidades -> regras -> custo/P/T)
│       ├── tipos.py                    # tipos suportados e regras (P/T, habilidades ativadas, terrenos)
│       ├── tipo_por_nome.py            # inferência do tipo pelo nome (NLTK: POS tagging + WordNet)
│       ├── texto_markov.py             # geração de texto via cadeia de Markov por tipo e cor
│       ├── treinar_modelo_markov.py    # script de treino -> gera models/modelo_markov.pkl
│       ├── subtipos.py                 # modelo estatístico de subtipos (quantidade, nomes, co-ocorrência)
│       ├── treinar_modelos_subtipos.py # script de treino -> gera models/modelos_subtipos.pkl
│       ├── regras_tipo.py              # Enchant (Aura), Equipped creature + Equip (Equipamento), Crew (Veículo), Saga (capítulos) e mana de terrenos
│       ├── estrutura.py                # regressão de custo de mana e P/T (scikit-learn)
│       ├── treinar_modelos_estruturais.py  # script de treino -> models/modelos_estruturais.pkl
│       ├── imagem.py                   # arte via API de imagem + moldura realista (frame) desenhada em código
│       ├── mecanicas.py                # [não usado na geração] inferência de mecânica, usado só como semente opcional
│       └── color_pie.py                # [não usado na geração] regras de color pie, mantido como registro
├── assets/
│   └── fonts/                          # fontes livres (Cinzel, EB Garamond — SIL OFL) usadas na moldura
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

> O `requirements.txt` fixa a versão do **scikit-learn** (`1.9.1`) usada para
> treinar `models/modelos_estruturais.pkl`. O pickle dos estimadores é
> sensível à versão: rodar com outra versão dispara
> `InconsistentVersionWarning` (silenciado no carregamento, ver
> `src/pipeline/estrutura.py`) e pode dar resultados inválidos. Ao
> retreinar os modelos, atualize essa versão no `requirements.txt`.

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
python src/pipeline/treinar_modelos_subtipos.py
python src/pipeline/treinar_modelos_estruturais.py
```

O primeiro treina um modelo de Markov (unigramas até pentagramas) para CADA tipo
de carta (criatura, instantânea, feitiço, artefato, encantamento, terreno),
por cor, sobre o `oracle_text` de `data/processed/cards.parquet`, e salva o
resultado em `models/modelo_markov.pkl` (um `ModeloTextoPorTipo`, que agrupa
os modelos de cada tipo). Terrenos são incolores: para eles o treino usa
`color_identity` no lugar de `colors` (um "Forest" tem identidade G), e o
modelo global inclui também os terrenos incolores de utilidade. O treino
também acumula as distribuições de **quantos parágrafos** e **quantas
keywords** cada carta tinha (por tipo e por cor), usadas depois para sortear
esses números na geração.

O segundo estima as distribuições de subtipos diretamente do corpus
(`src/pipeline/subtipos.py`): quantidade por tipo (com os limites do
projeto), frequência de cada subtipo por tipo e cor, co-ocorrência entre
subtipos (para combos como "Human Warrior"), alvos de `Enchant ...`, custos de
`Equip ...`, linhas `Equipped creature ...`, custos de `Crew N`, P/T de
Veículos e número de capítulos de Sagas. O resultado vai para
`models/modelos_subtipos.pkl`. Artefatos e criaturas têm pools separados:
um `Artifact Creature` usa subtipos de criatura, e o pool de artefato
remove qualquer nome que também seja tipo de criatura (ex.: `Wizard`).

O terceiro treina as duas regressões scikit-learn (custo de mana -- com o
tipo como feature -- e poder/resistência -- só criaturas; ver
`src/pipeline/estrutura.py`), imprime métricas de MAE/R² no holdout e salva
`models/modelos_estruturais.pkl`. Terrenos ficam de fora do modelo de custo
(não têm `mana_cost`) e não recebem P/T.

**Esses arquivos precisam ser commitados** (ao contrário de `data/`, a
pasta `models/` não está no `.gitignore`) — são eles que o Colab carrega,
sem precisar rodar os passos 4-6 de novo.

```bash
git add models/modelo_markov.pkl models/modelos_subtipos.pkl models/modelos_estruturais.pkl
git commit -m "Treina modelos de texto (Markov), subtipos e estruturais (custo/P-T)"
git push
```

### 7. Testar a geração localmente (opcional)

```python
import sys
sys.path.insert(0, "src")
from pipeline import gerar_carta

# tipo pode ser "Criatura", "Instantânea", "Feitiço", "Artefato",
# "Encantamento", "Terreno", "Aleatório" ou "Automático (pelo nome)".
# None/vazio também significa automático; se a inferência não tiver
# certeza (empate ou nome sem pista), o tipo é sorteado.
carta = gerar_carta("Murder", "Preto", peso_cor=0.7)  # automático
# pesos_ngrama dá um peso por ORDEM de n-grama (0=unigrama ... 4=pentagrama);
# os pesos são relativos entre as ordens que tiverem dados no contexto atual.
# carta = gerar_carta("Murder", "Preto", peso_cor=0.7,
#                     pesos_ngrama={4: 1.0, 3: 0.7, 2: 0.4, 1: 0.15, 0: 0.02})
# Atalho legado: peso_trigrama (0..1) interpola só trigrama e bigrama.
# carta = gerar_carta("Murder", "Preto", peso_cor=0.7, peso_trigrama=0.9)
print("tipo:", carta.tipo)
print("linha de tipo:", carta.tipo_linha)  # ex.: "Creature — Human Warrior"
print("subtipos:", carta.subtipos)
print("como o tipo foi decidido:", carta.tipo_origem, "-", carta.tipo_motivo)
print("custo de mana:", carta.custo_mana or "(nenhum -- terreno)")
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
      ajustável da cor (`peso_cor`) e por ordem de n-grama
      (`pesos_ngrama`: unigrama, bigrama, trigrama, tetragrama e
      pentagrama), preservando números e modificadores de P/T (`3`,
      `+1/+1`, `-1/-1`, `2/2`, `+X/+X`) e completando padrões incompletos
      (“gets until...”, “deals damage...” sem valor) — `src/pipeline/texto_markov.py`
- [x] Tipos de carta (criatura, instantânea, feitiço, artefato,
      encantamento, terreno) com texto condicionado ao tipo;
      instants/sorceries sem habilidades ativadas e sem P/T; terrenos sem
      custo nem P/T — `src/pipeline/tipos.py`
- [x] Subtipos data-driven (criatura 1–3, artefato 0–1, terreno 0–2,
      encantamento 0–2, mágica 0–1), com pools separados para criaturas e
      artefatos e coerência de cor nos terrenos — `src/pipeline/subtipos.py`
- [x] Número de **parágrafos** de habilidade e de **keywords** sorteado da
      distribuição real do tipo (keywords com fator para ficar acima da
      média do corpus) — `src/pipeline/texto_markov.py`
- [x] Ajuste do peso cor/global para **artefatos incolores** (a maioria do
      corpus), dando mais voz ao modelo global — `AJUSTE_PESO_COR_POR_TIPO`
- [x] Regras de Aura (`Enchant ...`), Equipamento (`Equipped creature ...` +
      `Equip ...`), Veículo (`Crew N` + P/T) e Saga (capítulos numerados em
      romano), além da habilidade de mana de terrenos com tipo básico
      — `src/pipeline/regras_tipo.py`
- [x] Inferência do tipo pelo nome (POS tagging + WordNet) com fallback
      para sorteio em caso de empate/sem pista — `src/pipeline/tipo_por_nome.py`
- [x] Regressão do custo de mana (com tipo como feature) e de
      poder/resistência (só criaturas) (Scikit-Learn)
      — `src/pipeline/estrutura.py`
- [x] **Arte via API de imagem** (Pollinations.ai, sem chave) com prompt
      montado a partir do nome/tipo/texto, cache em disco e fallback
      procedural local — `src/pipeline/imagem.py`
- [x] **Moldura realista em código** (frame): painel do título com
      gradiente da cor, janela da arte, caixa de tipo, caixa de texto
      translúcida e selo de P/T, com fontes livres (Cinzel/EB Garamond) —
      `src/pipeline/imagem.py`
- [ ] Classificador de cor em PyTorch — não mais como filtro, mas como
      *métrica de diagnóstico* ("o quão on-color este texto gerado
      ficou?"), compatível com a escolha de deixar a geração livre
- [ ] Grafo de transições de palavras por cor (NetworkX) + visualizações
      (Matplotlib) — bom complemento visual pro relatório, mostrando o
      "espaço" de onde a cadeia de Markov sorteia
