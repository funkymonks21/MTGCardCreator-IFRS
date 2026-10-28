# Estudo comparativo de 3 modelos — caderno de metodologia

Este documento registra **cada decisão metodológica** do estudo comparativo, com
a justificativa matemática correspondente. Ele é separado do `README.md` da raiz
(que descreve o gerador) porque aqui o assunto não é *como gerar cartas*, e sim
*como provar qual modelo gera melhor*.

> **Regra do documento:** nenhuma decisão entra aqui sem (a) o problema que ela
> resolve, (b) a alternativa descartada, e (c) a base quantitativa. Se um número
> não puder ser derivado do corpus ou de teoria estatística, ele é marcado como
> **premissa** — e premissas precisam ser validadas depois.

---

## 1. Contexto do experimento

O objetivo final é responder, com dados, três perguntas sobre os modelos:

| Pergunta | Métrica central |
|---|---|
| Qual é o **mais próximo dos dados reais**? | erro contra o gabarito (MAE/acurácia) + divergência distribucional |
| Qual é o **mais criativo**? | novidade de n-gramas, diversidade lexical, entropia |
| Qual é o **mais assertivo/realista**? | coerência interna, validade de regras, avaliação humana |

Os três braços:

| Braço | Texto | Tipo / custo / P-T |
|---|---|---|
| **M1** | Markov n-gramas 1–5 (pesos por ordem) | regressão Ridge (TF-IDF + metadados) |
| **M2** | Markov + refino por LLM | mesma regressão do M1 |
| **M3** | LLM pura | LLM decide tudo |

O ponto crítico: **M1 e M2 compartilham a regressão**, então qualquer diferença
entre eles vem do texto. E o **M3 não é treinado por nós** — logo, não é possível
descontaminá-lo (ver §4.12).

### 1.1 Escopo fechado: exatamente 6 tipos

Todos os braços — inclusive o M3 — produzem **apenas** `Creature`, `Instant`,
`Sorcery`, `Artifact`, `Enchantment` e `Land`. Não há `Battle`, `Kindred`,
`Planeswalker` nem os tipos não-jogáveis (`Stickers`, `Conspiracy`,
`Phenomenon`, `Hero`, `Event`, `Boss`).

Isso **já estava embutido** no D-001: a população de referência exclui as 502
cartas não-jogáveis, e o holdout de 1.200 é sorteado só dentro dos 33.031
cartões dos 6 tipos. Portanto **nenhum número precisa ser recalculado** — a
restrição confirma o desenho em vez de alterá-lo.

O que ela **acrescenta** são duas definições que faltavam: como o M3 é
*impedido* de sair dos 6 tipos (D-003) e como a escolha de tipo é avaliada
(D-004).

---

## 2. Estado herdado (decisões já implementadas)

Registradas aqui de forma resumida; a fundamentação detalhada de cada uma está no
`README.md` da raiz e no código indicado.

| # | Decisão | Onde vive |
|---|---|---|
| H-01 | Texto combina ordens 0–4 por interpolação ponderada, com pesos `{4:1.00, 3:0.70, 2:0.40, 1:0.15, 0:0.02}` | `texto_markov.py` |
| H-02 | Nº de parágrafos sorteado da distribuição real por tipo/cor | `texto_markov.py` |
| H-03 | Keywords infladas por `FATOR_KEYWORDS = 2.0`, teto 4 | `texto_markov.py` |
| H-04 | Artefatos: ajuste de peso de cor `0.4` (favorece o global) | `texto_markov.py` |
| H-05 | Veículos: `Crew N` obrigatório + P/T sorteado dos Veículos reais | `regras_tipo.py`, `subtipos.py` |
| H-06 | Sagas: capítulos com numeral romano incremental, 1–5 | `regras_tipo.py` |
| H-07 | Aura → `Enchant`; Equipamento → `Equipped creature...` + `Equip` | `regras_tipo.py` |
| H-08 | Terreno sem custo de mana, sem P/T; mágica instantânea/feitiço sem habilidade ativada | `regras_tipo.py` |
| H-09 | Arte via Pollinations (grátis), moldura desenhada, fonte Cinzel + EB Garamond | `imagem.py` |

---

## 3. Como ler as decisões

Cada decisão tem: **ID**, **data**, **status** (`proposta` / `aceita` /
`validada` / `revista`) e as seções *Problema → Decisão → Base matemática →
Consequências → Alternativas descartadas* (a estrutura se adapta ao tipo de
decisão; o que não pode faltar é a justificativa quantitativa).

---

## 4. D-001 — Conjunto de avaliação (holdout) de 1.200 cartas

- **Data:** 2026-10-23
- **Status:** **revista** — a parte numérica (1.200, alocação por tipo) foi
  substituída pelo **D-006** (pool de 850 + 500 reais + 500 fictícios). O
  raciocínio de *por que existe holdout* (§4.2) e o *método de alocação*
  (§4.7, maiores restos) continuam válidos e são reusados no D-006.

> **Nota de revisão:** esta seção é mantida como registro histórico da decisão
> original. Os números abaixo (1.200, 658/137/129/125/91/60) **não** são mais
> os usados no experimento. Leia o D-006 para o desenho vigente.

### 4.1 Problema

Para afirmar que um modelo "chegou mais perto dos dados reais", é preciso
comparar a saída dele com a **verdade** de cartas que ele **nunca viu**. Se as
cartas de avaliação estiverem no treino, a medição deixa de medir generalização e
passa a medir **memorização**.

O risco não é teórico — é o comportamento esperado do M1. Ver §4.2.

### 4.2 Por que o holdout é obrigatório: a matemática da memorização

O gerador de texto interpola as ordens de n-grama:

$$P(w \mid h) = \sum_{k=0}^{4} \lambda_k \, P_k(w \mid h_k)$$

com os pesos padrão `{4: 1.00, 3: 0.70, 2: 0.40, 1: 0.15, 0: 0.02}`. A soma é
**2,27**, então a massa efetiva de cada ordem depois da normalização é:

| Ordem $k$ | n-grama | Peso bruto | Massa normalizada $\lambda_k$ |
|---|---|---|---|
| 4 | pentagrama | 1,00 | **44,05%** |
| 3 | tetragrama | 0,70 | 30,84% |
| 2 | trigrama | 0,40 | 17,62% |
| 1 | bigrama | 0,15 | 6,61% |
| 0 | unigrama | 0,02 | 0,88% |

Considere um pentagrama $(w_1, w_2, w_3, w_4, w_5)$ que existe **no treino**.
Ao gerar a posição 5 com contexto $(w_1..w_4)$:

- se essa sequência tem uma **única** continuação no corpus, então
  $P_4(w_5 \mid h) = 1$ e, portanto,

$$P(w_5 \mid h) \;\ge\; \lambda_4 = 0{,}4405$$

- como o tetragrama $(w_2,w_3,w_4,w_5)$ também é, em geral, único, soma-se
  $\lambda_3$:

$$P(w_5 \mid h) \;\ge\; \lambda_4 + \lambda_3 = 0{,}4405 + 0{,}3084 = 0{,}7489$$

Ou seja: **cada passo tem ≥ 44% de chance de repetir a continuação memorizada, e
tipicamente ≥ 75%**. Como o próximo contexto volta a ser um n-grama do mesmo
texto, o modelo reentra no caminho memorizado a cada passo — o texto **converge
para o original** em vez de se afastar dele.

Se a carta foi retirada do treino, esses n-gramas simplesmente **não existem** e
a probabilidade cai para o que o modelo generalizou. É essa diferença que o
holdout mede.

> **Consequência direta:** avaliar no treino **inflaria o M1 por construção** e
> tornaria a comparação com M2/M3 inválida — o M1 pareceria mais fiel só porque
> está copiando.

### 4.3 Decisão

- **1.200 cartas** retiradas do corpus de treino.
- **Mínimo de 60 por tipo**.
- **O restante distribuído proporcionalmente** ao tamanho de cada tipo.

### 4.4 População de referência

O corpus tem **33.533 cartas**, mas 502 são não-jogáveis (`Card`, `Stickers`,
`Conspiracy`, `Phenomenon`, `Hero`, `Event`, `Boss`, `Summon ...`) e ficam fora
do escopo do gerador. A população modelada é:

$$P = 18\,411 + 3\,817 + 3\,605 + 3\,491 + 2\,558 + 1\,149 = 33\,031$$

| Tipo | $n_i$ | Proporção $n_i/P$ |
|---|---:|---:|
| Creature | 18.411 | 55,739% |
| Instant | 3.817 | 11,556% |
| Sorcery | 3.605 | 10,914% |
| Enchantment | 3.491 | 10,569% |
| Artifact | 2.558 | 7,745% |
| Land | 1.149 | 3,479% |
| **Total** | **33.031** | **100%** |

### 4.5 Por que 1.200 (e não 1.000 ou 2.000)

O holdout precisa ser **≥ o tamanho da amostra de geração**. O protocolo é gerar
**1.000 cartas válidas por modelo** (ver D-002); as 1.200 deixam **200 de
reserva** para repor falhas, para completar cotas finas e para reuso em análises
secundárias.

A precisão de uma proporção estimada é dada por

$$\text{MoE} = z_{1-\alpha/2}\sqrt{\frac{p(1-p)}{n}}, \qquad
\text{máximo em } p = 0{,}5.$$

Com $z_{0{,}975} = 1{,}96$:

| $n$ | MoE (95%, $p=0{,}5$) |
|---:|---:|
| 1.000 | ± 3,10 p.p. |
| **1.200** | **± 2,83 p.p.** |

Ou seja, 1.200 já está na faixa de ±3 p.p. — abaixo disso o ganho por carta
adicional é marginal ($\propto 1/\sqrt{n}$), e acima disso o custo de geração
cresce sem retorno proporcional.

**Custo no treino:** $1.200/33.031 = 3{,}63\%$. O treino mantém 96,37% do
corpus. Para um modelo com dezenas de milhares de n-gramas, uma remoção de 3,6%
é muito menor que a variabilidade entre sementes de treino — ou seja, é
**estatisticamente indistinguível** de usar o corpus inteiro, mas elimina o
vazamento.

### 4.6 Por que um piso de 60 por tipo

Sem piso, a alocação proporcional dá a Land apenas

$$q_{\text{land}} = 1200 \times \frac{1149}{33031} = 41{,}74 \text{ cartas.}$$

Com $n = 41{,}74$, o intervalo de confiança de 95% para uma proporção é

$$1{,}96\sqrt{\frac{0{,}25}{41{,}74}} = \pm 15{,}17 \text{ p.p.}$$

Com o piso de 60, cai para

$$1{,}96\sqrt{\frac{0{,}25}{60}} = \pm 12{,}65 \text{ p.p.}$$

O piso compra **2,5 p.p. de precisão** no tipo mais raro e, mais importante,
garante que **nenhum tipo fique com estimativa inutilizável**. É o mesmo
raciocínio de alocação estratificada com mínimo: prefere-se perder um pouco de
representatividade global em troca de poder inferencial em todas as células.

### 4.7 Método de alocação (maiores restos)

A alocação é calculada em 4 passos, de forma reprodutível:

1. **Quotas proporcionais puras:** $q_i = N \cdot n_i / P$.
2. **Aplicar o piso:** todo $q_i < 60$ vira 60 (aqui, só Land).
3. **Redistribuir o restante** entre os tipos não-afetados, proporcionalmente às
   suas contagens: $q'_i = (N - 60 \cdot |\text{afetados}|) \cdot n_i / \sum_{\text{livres}} n_j$.
4. **Fechar em inteiros** pelo método dos **maiores restos**: arredonda para
   baixo e distribui as unidades que faltam aos maiores restos fracionários.

Justificativa do passo 4: o arredondamento ingênuo não fecha a soma em $N$; o
método dos maiores restos é o que **minimiza o erro de representação** (é o mesmo
usado em repartição de cadeiras — cota de Hamilton).

Cálculo intermediário (passo 3), após fixar Land em 60:

| Tipo | $q'_i$ exato |
|---|---:|
| Creature | 658,319 |
| Instant | 136,484 |
| Sorcery | 128,903 |
| Enchantment | 124,827 |
| Artifact | 91,466 |
| **Soma** | **1.140,000** |

### 4.8 Alocação final

| Tipo | Cartas | % do holdout | % do próprio tipo |
|---|---:|---:|---:|
| Creature | **658** | 54,83% | 3,57% |
| Instant | **137** | 11,42% | 3,59% |
| Sorcery | **129** | 10,75% | 3,58% |
| Enchantment | **125** | 10,42% | 3,58% |
| Artifact | **91** | 7,58% | 3,56% |
| Land | **60** | 5,00% | 5,22% |
| **Total** | **1.200** | 100% | 3,63% |

Observação: a fração retirada é praticamente constante (~3,58%) em todos os
tipos, **exceto Land (5,22%)** — efeito direto do piso. É essa distorção que
exige reponderação (§4.9).

### 4.9 Reponderação (pesos de desenho)

Como Land foi **superamostrado** (5,00% do holdout vs 3,48% da população),
estatísticas que pretendam falar sobre "a população de cartas reais" precisam de
pesos. O peso de desenho é

$$w_i = \frac{n_i / P}{a_i / N}$$

| Tipo | $w_i$ |
|---|---:|
| Creature | 1,0165 |
| Instant | 1,0122 |
| Sorcery | 1,0153 |
| Enchantment | 1,0146 |
| Artifact | 1,0212 |
| **Land** | **0,6957** |

Leitura: cada carta de Land "vale" 0,696 de uma carta comum; as demais valem
~1,01–1,02. **Onde usar:**

- **Comparações entre modelos:** dispensam peso — os três recebem as mesmas
  entradas (amostra pareada), então o viés de amostragem se cancela.
- **Afirmações sobre a população** ("a distribuição de CMC do M1 desvia X% da
  real"): **exigem** peso.

### 4.10 Poder para comparação pareada

Como cada modelo recebe a **mesma** carta, as diferenças são pareadas. A
diferença mínima detectável (80% de poder, $\alpha = 0{,}05$) é

$$\text{MDD} = (z_{0{,}975} + z_{0{,}80})\frac{\sigma_d}{\sqrt{n}}
= 2{,}8016 \cdot \frac{\sigma_d}{\sqrt{1000}}.$$

| Desvio-padrão das diferenças $\sigma_d$ | MDD (CMC) |
|---:|---:|
| 1,0 | 0,089 |
| 2,0 | 0,177 |
| 3,0 | 0,266 |

Como o erro de CMC do pipeline atual tem ordem de ~1,0 (MAE), espera-se
$\sigma_d \lesssim 2$ — logo diferenças de **~0,2 de CMC** entre modelos já são
detectáveis. Isso é suficiente para separar M1 de M3 se houver diferença real.

### 4.11 O que exatamente sai do treino

As 1.200 cartas são removidas **dos três artefatos**, não só do Markov:

| Artefato | Por quê |
|---|---|
| `modelo_markov.pkl` | é o que memoriza n-gramas (§4.2) |
| `modelos_subtipos.pkl` | subtipos condicionados por tipo/cor também memorizam |
| `modelos_estruturais.pkl` | a regressão vê o `oracle_text` real como feature |

Manter a exclusão nos três é o caminho conservador: garante que nenhuma etapa do
pipeline "conheceu" a carta avaliada. O custo é o mesmo retreino, uma vez só.

### 4.12 O caso do M3 (não tem solução completa)

O Gemma foi pré-treinado pela Google e provavelmente **conhece Magic**. Não é
possível remover cartas do pré-treinamento. Portanto:

- o M3 pode **lembrar** de cartas reais;
- isso não é vazamento *do nosso experimento*, é contaminação *externa*;
- a resposta é **medir e reportar** a taxa de memorização (§4.14), não esconder.

Um resultado do tipo "o M3 reproduziu 12% das cartas reais quase literalmente"
é um **achado legítimo** e deve aparecer no relatório.

### 4.13 Alternativas rejeitadas

| Alternativa | Por que foi rejeitada |
|---|---|
| Avaliar no próprio treino | mede cópia, não generalização (§4.2) |
| Nomes inventados, sem holdout | perde o gabarito de CMC/P-T/tipo — a métrica mais forte |
| Proporcional puro (sem piso) | Land com $n=41{,}7$ → IC de ±15,2 p.p. |
| 2.000 de holdout | só se justifica com 2.000 gerações; aqui seria desperdício de corpus |
| Holdout por conjunto/edição | melhor contra contaminação temática, mas exige `released_at` (ausente em `RELEVANT_COLUMNS`) e reduz a diversidade |

> **Nota sobre holdout por conjunto:** a alternativa "separar edições inteiras"
> é metodologicamente superior contra vazamento temático, porém o
> `fetch_scryfall_data.py` hoje não coleta `released_at` e a partição por edição
> distorceria as cotas por tipo. Fica registrada como **melhoria futura**, caso
> sobre tempo.

### 4.14 Validação opcional (sonda de memorização)

Para quantificar §4.2 empiricamente, pode-se treinar **um** modelo extra com as
1.200 cartas **incluídas** (só para a sonda) e medir a similaridade (ex.:
ROUGE-L / sobreposição de 5-gramas) entre texto gerado e texto real da mesma
carta. Se a similaridade for muito maior que a do modelo com holdout, a
memorização está confirmada e medida. **Status: opcional**, não bloqueia o
resto.

---

## 5. D-002 — Protocolo de geração

- **Data:** 2026-10-23
- **Status:** aceita (revisada pelo D-006 nos números)

- **1.000 gerações válidas por modelo** (3.000 no total), divididas em
  **500 com nome real** + **500 com nome fictício** (D-006).
- As 500 reais são sorteadas **uma única vez**, com semente fixa, do pool de
  850 e o **mesmo conjunto de 500 é usado pelos três braços** (pareamento).
- Os 350 restantes do pool **não são usados para escolher as "melhores"**:
  servem apenas de **reserva** para repor falhas.
- Entrada no **Modo A** (D-004): nome, cor e tipo real da carta. Para as
  fictícias, o tipo e a cor são os do perfil sorteado, e o nome é inventado.
- Semente de geração por carta: `seed = hash(carta_id, modelo)`. **Revista pelo
  §12.1** (hash salgado por processo → SHA-256; M1 e M2 com a mesma semente).
- Uma geração só conta se passar no **lint de regras** (Veículo com Crew+P/T,
  Saga com capítulos, Aura com `Enchant`, terreno sem custo/P-T, etc.); a taxa
  de falha por braço é reportada como métrica, não escondida.

### 5.1 Por que subconjunto aleatório (e não "as melhores 500")

Escolher as melhores entre as disponíveis é **seleção sobre a saída**: a amostra
deixa de ser representativa da distribuição do modelo e o erro medido passa a
ser o de uma amostra truncada. Formalmente, se o critério de descarte é
correlacionado com a qualidade, o estimador $\widehat{\text{MAE}}$ fica
**enviesado para baixo** e o modelo parece melhor do que é.

O subconjunto aleatório garante que cada carta tem a mesma probabilidade de
entrar, independente da saída — a amostra é **não-enviesada** por construção.
O custo é aceitar que falhas dentro das 500 contam contra o modelo, o que é
exatamente o comportamento desejado numa comparação.

---

## 6. D-003 — M3 restrito aos 6 tipos

- **Data:** 2026-10-23
- **Status:** aceita

### 6.1 Problema

Uma LLM sem restrição pode responder `Planeswalker`, `Battle`, `Kindred`, ou
inventar categorias. Se isso acontecer, a saída não é comparável com o gabarito
e contamina as métricas. É preciso **impedir** a saída inválida, não apenas
detectá-la depois.

### 6.2 Decisão: três camadas de garantia

| Camada | Mecanismo | O que elimina |
|---|---|---|
| 1. Prompt | enum explícito dos 6 tipos + instrução "escolha exatamente um" | erro grosseiro |
| 2. Decodificação | saída estruturada (JSON schema / gramática) — no Ollama, `format` com o schema | tipo fora do enum |
| 3. Validação | pós-processamento: se o tipo não pertence ao enum → **1 retry** → fallback determinístico + registro de falha | resíduo |

A camada 2 é a que realmente garante: **gramática restringe o espaço de
decodificação**, então o modelo *não consegue* emitir um token fora do enum.
As camadas 1 e 3 existem porque nem todo runner implementa gramática (no Colab,
por exemplo, pode-se usar `llama-cpp-python` com GBNF). É "defesa em
profundidade": se uma camada falhar, a seguinte segura.

### 6.3 Por que não extrair o tipo por regex do texto livre

Porque "Creature — Human Wizard" e "Artifact Creature — Golem" contêm mais de
um tipo, e a ordem/ambiguidade gera erro de parsing. Com enum na decodificação,
o problema deixa de existir — o modelo devolve um campo `tipo` com valor único.

### 6.4 Consequência para as métricas

Restringir o M3 ao enum **não** torna a acurácia de tipo trivial: ele ainda tem
que **escolher** entre 6 (ver D-004). O que a restrição faz é garantir que o
erro seja do tipo "escolheu o errado", nunca "inventou uma categoria" — que é um
erro de formato, não de conteúdo.

---

## 7. D-004 — Dois modos de avaliação

- **Data:** 2026-10-23
- **Status:** aceita

### 7.1 O problema de justiça

O M1 e o M2 **recebem** o tipo (ele é uma entrada do pipeline), então não faz
sentido pontuá-los por "acertar o tipo". Já o M3, na definição original do
estudo, decide tudo — inclusive o tipo. Comparar a acurácia de tipo do M3 com a
do M1 seria comparar um **classificador** com um **oráculo**.

A solução é separar o estudo em dois modos, com perguntas diferentes.

### 7.2 Modo A — condicionado (comparação principal)

| Item | Definição |
|---|---|
| Entrada | nome + cor + **tipo real** |
| O que se avalia | texto, custo de mana, P/T, coerência, criatividade |
| Tipo | **não é avaliado** (é dado) |
| Amostra | 1.000 (D-002) |
| Uso | é o *head-to-head* que responde "qual é o mais próximo dos dados reais" |

Este é o modo principal, porque isola o que realmente difere entre os braços: a
**geração de conteúdo**, não a decisão de tipo.

### 7.3 Modo B — inferência de tipo (análise secundária)

| Item | Definição |
|---|---|
| Entrada | **apenas o nome** (sem cor, sem tipo — ver D-005) |
| M1 / M2 | usam `inferir_tipo_por_nome` (POS tagging + WordNet + listas curadas); como o refino do M2 é só de texto, **M1 ≡ M2** |
| M3 | escolhe entre os 6 via LLM |
| Métrica | acurácia contra `tipo_real` do gabarito + cobertura, acurácia seletiva e macro-F1 |
| Amostra | **400 cartas** (subconjunto aleatório das 1.000) |

**Transparência obrigatória no relatório:** a decisão de tipo do M1/M2 **não é
Markov** — é um módulo heurístico separado (`tipo_por_nome.py`). Chamar isso de
"desempenho do M1" seria impreciso; o correto é "desempenho da etapa de
inferência de tipo do pipeline".

### 7.4 Baselines (sem os quais a acurácia não significa nada)

No conjunto de avaliação (n = 1.200, alocação do §4.8):

| Baseline | Acurácia esperada |
|---|---:|
| Chute uniforme entre 6 | **16,67%** |
| Amostrador da distribuição real do corpus | **34,44%** |
| Sempre responder "Creature" (maioria) | **54,83%** |

Uma acurácia de 60% parece boa contra 16,67%, mas é quase indistinguível de
"chutar Creature". Por isso o relatório deve citar **os três** baselines e,
idealmente, reportar também **macro-F1** e a **matriz de confusão** — que
penalizam o colapso na classe majoritária, coisa que a acurácia não faz.

### 7.5 Abstenção: cobertura e acurácia seletiva

`inferir_tipo_por_nome` pode devolver `None` (sem sinal/empate). Pontuar isso
como "erro" puro esconde informação. O correto é reportar a curva
**risco × cobertura**:

- **Cobertura** $= \frac{\#\{\text{tipo} \neq \text{None}\}}{\#\text{total}}$
- **Acurácia seletiva** $= \frac{\#\{\text{acertos entre os não-abstidos}\}}{\#\{\text{não-abstidos}\}}$
- **Acurácia efetiva** $= \frac{\#\text{acertos}}{\#\text{total}}$ (com `None` → sorteio semeado, como o pipeline faz de fato)

Assim o relatório mostra o trade-off: o M1/M2 pode ser muito preciso **quando
responde** e abster-se no resto. Comparar só a acurácia efetiva puniria a
abstenção honesta.

### 7.6 Custo

O Modo B custa **+400 gerações apenas do M3** (M1/M2 são instantâneos). Com
$n = 400$, a margem de erro de uma proporção é

$$1{,}96\sqrt{\frac{0{,}25}{400}} = \pm 4{,}90 \text{ p.p.},$$

suficiente para separar o M3 do baseline de maioria (54,83%) se houver diferença
real de ~10 p.p.

### 7.7 Ressalva (a mesma do §4.12)

No Modo B o M3 leva vantagem de **memorização**: nomes de cartas reais
("Serra Angel", "Llanowar Elves") podem estar no pré-treinamento, e o modelo
"acerta o tipo" sem inferir nada. Esse número deve ser lido junto com a taxa de
memorização (§4.12) — e, se possível, cruzado com nomes que o M3 provavelmente
não conhece.

---

## 8. D-005 — Mecânica do Modo B para M1/M2

- **Data:** 2026-10-23
- **Status:** aceita

### 8.1 Entrada: apenas o nome (sem cor)

`inferir_tipo_por_nome(nome)` **não recebe cor** — a assinatura é só isso. Como a
cor é informativa sobre o tipo (terrenos são incolores, artefatos quase sempre),
dar a cor apenas ao M3 confundiria a comparação: a diferença medida poderia ser
"o M3 viu a cor", não "o M3 infere melhor".

**Decisão:** no Modo B, **todos** recebem apenas o nome. É também o que o
pipeline faz de fato quando o usuário deixa o tipo em "Automático".

### 8.2 Abstenção e a decomposição exata da acurácia efetiva

O módulo devolve `None` quando há empate ou nenhum sinal (ex.: `"Day of
Judgment"` → empate entre instantânea/feitiço e encantamento). O `gerar_carta`
então **sorteia** o tipo pela frequência real do corpus.

Isso permite escrever a acurácia efetiva como uma decomposição exata:

$$\text{acc}_{\text{ef}} = c \cdot a_{\text{sel}} + (1-c)\cdot S, \qquad
S = \sum_i p_i^2 = 0{,}3444$$

onde $c$ é a **cobertura** (fração de cartas em que o módulo não se absteve),
$a_{\text{sel}}$ é a **acurácia seletiva** (acerto entre as não-abstenções) e
$S$ é a acurácia esperada de um sorteio da distribuição real — que é
exatamente o baseline "amostrador do prior" (34,44%).

Consequência prática: **se a acurácia efetiva do M1 ficar em ~34%, a inferência
não agregou nada acima do sorteio**. E mesmo com cobertura de 100% e acurácia
seletiva de 50%, a acurácia efetiva (50%) ainda perde para o baseline de maioria
(54,83%). É por isso que os três números — cobertura, seletiva e efetiva — têm
que aparecer juntos.

### 8.3 Problema encontrado: a decisão de tipo não era reprodutível (RESOLVIDO)

> **Status:** corrigido em 2026-10-23. Ver §8.3.1.

Auditando o caminho real do código, havia **duas fontes de aleatoriedade não
semeadas** quando `tipo=None`:

| Onde | O quê | Linha |
|---|---|---|
| `generator.py` | `inferir_tipo_por_nome(nome)` chamado **sem** `rng` | 315 |
| `generator.py` | `modelo.sortear_tipo()` chamado **sem** `rng` (fallback) | 317 |
| `tipo_por_nome.py` | desempate instantânea/feitiço usa `rng.choice(...)` | 636 |
| `texto_markov.py` | `sortear_tipo` usa `rng or random.Random()` | 1176 |

Ou seja: em `"Counterspell"` (que o módulo manda para **sorcery**, quando o real
é *instant*) e em `"Day of Judgment"` (abstenção), o resultado **muda entre
execuções**. Para um experimento isso é inaceitável — duas rodadas do mesmo
modelo dariam acurácias diferentes.

**Ação necessária (quando o código for liberado):** passar
`rng=random.Random(seed_da_carta)` nos dois pontos de chamada do
`generator.py`. É uma mudança de 2 linhas, mas sem ela o Modo B mede ruído.

> Observação: o Modo A **não** é afetado, porque recebe o tipo explicitamente e
> não passa por esse caminho.

#### 8.3.1 Correção aplicada

`gerar_carta` ganhou um parâmetro **`seed: Optional[int] = None`** (o último da
assinatura, para não quebrar chamadas posicionais existentes). Com ele:

- **um único** `rng = random.Random(seed)` é criado no início e usado em todas
  as decisões aleatórias (tipo, subtipos, Saga, Crew, P/T de Veículo);
- `inferir_tipo_por_nome(nome, rng=rng)` e `modelo.sortear_tipo(rng=rng)`
  recebem esse `rng` — a decisão de tipo fica determinística;
- `modelo.gerar_texto(..., seed=seed)` recebe a semente, então o texto também
  é determinístico;
- o caminho `"Aleatório"` (`_TIPO_SORTEAR`) foi corrigido junto — antes ele
  também sorteava sem semente.

Com `seed=None` o comportamento é **idêntico ao anterior** (não reprodutível),
então a interface pública continua compatível.

**Verificação feita:**

```
mesmo seed  -> cartas idênticas (tipo, custo, texto, P/T, keywords)  ✓
seed diferente -> cartas diferentes                                   ✓
tipo automático com abstenção -> determinístico por seed              ✓
```

Isso fecha o **P-09**. A partir de agora, toda geração do experimento deve
passar `seed=hash(carta_id, modelo)`.

### 8.4 M1 ≡ M2 no Modo B (e por que isso é bom)

O M2 é definido como *Markov + refino de texto por LLM*, **mantendo a mesma
regressão**. O escopo do refino foi **confirmado: só o texto** — a etapa de
decisão de tipo não é tocada. Logo, no Modo B, M1 e M2 produzem **exatamente a
mesma previsão** (dado o mesmo seed).

Isso não é um defeito: é um **teste de sanidade**. Confirma que o ganho do M2
(se houver) vem do texto, e não de uma vantagem escondida na escolha de tipo.
Mas precisa ser dito no relatório: **o Modo B compara dois braços, não três** —
"módulo heurístico (M1/M2)" contra "LLM (M3)".

Como a equivalência é **por construção**, não faz sentido rodar o M2 duas vezes
no Modo B: as 400 previsões de tipo são geradas uma vez e reaproveitadas para
os dois. Isso também economiza 400 gerações.

**Teste estatístico adequado:** como as previsões são pareadas e binárias, usar
**McNemar**:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \qquad \text{gl} = 1,$$

onde $b$ = cartas em que M1 erra e M3 acerta, e $c$ = o contrário. A correção
de continuidade (−1) é obrigatória aqui, porque $b + c$ pode ser pequeno.
McNemar é o teste certo porque ignora os pares em que os dois acertam ou os dois
erram — que são a maioria.

---

## 9. D-006 — Desenho híbrido: pool 850, 500 reais + 500 fictícios

- **Data:** 2026-10-23
- **Status:** aceita
- **Substitui:** a parte numérica do D-001

### 9.1 Motivação

Três problemas motivaram a revisão:

1. **Memorização não era mensurável.** Avaliando só com nomes reais, não havia
   como separar "o modelo generaliza" de "o modelo lembra da carta". Uma metade
   com nomes inventados dá o contraste controlado.
2. **Criatividade não era mensurável.** Sem nomes novos, todo teste premiava
   quem se parecesse com cartas existentes.
3. **Desequilíbrio de cor.** O corpus é dominado por criaturas; sem
   balanceamento, "acertar sempre verde/branco" poderia mascarar diferenças.

### 9.2 Três conjuntos que não podem ser confundidos

Esta é a distinção que evita erro de dimensionamento:

| Conjunto | Tamanho | Papel |
|---|---:|---|
| **Pool (holdout)** | **850** | cartas reais **retiradas do treino**; é o universo de onde saem as amostras |
| **Amostra real avaliada** | **500** | subconjunto do pool efetivamente gerado e pontuado |
| **Reserva** | **350** | o resto do pool; repõe falhas, **nunca** escolhido por qualidade |

E, separadamente, a **metade fictícia** (500) não sai do corpus: os nomes são
inventados, mas o perfil (tipo × cor) **espelha** o da metade real.

Total por modelo: **500 + 500 = 1.000** gerações (3.000 no total).

### 9.3 Universo de amostragem

O universo é o de cartas **monocor** do corpus modelado, com a mesma regra que
o pipeline usa internamente (`colors` para os tipos comuns, `color_identity`
para Terreno — ver `tipos.TIPOS_USAM_COLOR_IDENTITY`):

$$P_{\text{mono}} = 24\,528 \text{ cartas} \quad (74{,}3\% \text{ dos } 33\,031)$$

Ficam de fora **8.503 cartas**: 4.994 multicor (15,1%) e 3.509 incolores
(10,6%).

> ⚠️ **Limitação registrada:** a metade real não cobre artefatos incolores
> (1.962, ou 76,7% dos artefatos) nem terrenos de identidade vazia/multicor.
> A avaliação de artefatos usa só os **523 artefatos monocor** e a de terrenos
> só os **252 terrenos de identidade monocor**. Isso é consequência direta de
> exigir 100 cartas por cor: o gerador só aceita 5 cores, então uma carta sem
> cor não tem entrada correspondente. A alternativa (6º estrato "incolor")
> teria de reduzir a amostra para ~83 por estrato — ver §9.9.

### 9.4 Alocação do pool (850)

Método: **IPF (Iterative Proportional Fitting)** para casar duas margens —
linhas (tipo) e colunas (cor) — seguido de arredondamento por **maiores restos**.
Alvos de linha: piso de **60 por tipo** (regra do D-001 §4.7) para os tipos
raros, restante proporcional. Alvos de coluna: **170 por cor**.

| Tipo | W | U | B | R | G | Total |
|---|---:|---:|---:|---:|---:|---:|
| Creature | 86 | 86 | 86 | 86 | 86 | **430** |
| Instant | 21 | 21 | 21 | 21 | 21 | **105** |
| Sorcery | 20 | 20 | 20 | 19 | 19 | **98** |
| Enchantment | 19 | 19 | 19 | 20 | 20 | **97** |
| Artifact | 12 | 12 | 12 | 12 | 12 | **60** |
| Land | 12 | 12 | 12 | 12 | 12 | **60** |
| **Total** | **170** | **170** | **170** | **170** | **170** | **850** |

Verificações: todas as células respeitam a capacidade do corpus; as margens
fecham exatamente; o `n_eff` (Kish) do pool é **757/850 (89%)**.

### 9.5 Alocação da amostra real avaliada (500)

Alvos: **100 por cor** e **≥60 por tipo**. Com 500 cartas e 6 tipos, o piso de
60 consome 300 e sobram 200 para Creature — é a única forma de garantir o
mínimo em todos os tipos nesse tamanho.

| Tipo | W | U | B | R | G | Total |
|---|---:|---:|---:|---:|---:|---:|
| Creature | 40 | 40 | 40 | 40 | 40 | **200** |
| Instant | 12 | 12 | 12 | 12 | 12 | **60** |
| Sorcery | 12 | 12 | 12 | 12 | 12 | **60** |
| Enchantment | 12 | 12 | 12 | 12 | 12 | **60** |
| Artifact | 12 | 12 | 12 | 12 | 12 | **60** |
| Land | 12 | 12 | 12 | 12 | 12 | **60** |
| **Total** | **100** | **100** | **100** | **100** | **100** | **500** |

A matriz das 500 é **célula a célula menor ou igual** à das 850 (verificado),
então a amostra pode ser extraída do pool sem reposição.

Perfil por cor de cada modelo: **100 reais + 100 fictícias = 200** — que é
exatamente o número que você pediu.

### 9.6 Metade fictícia (500)

- Espelha **exatamente** a matriz de 9.5: 40 criaturas + 12 de cada outro tipo
  por cor, 100 por cor.
- Isso é essencial: o contraste real × fictício fica **pareado por tipo e cor**,
  então a única variável que muda é a origem do nome.
- Nomes precisam satisfazer: (a) **não existir no corpus** (checagem
  case-insensitive contra `cards.parquet`), (b) ter 2–3 palavras no estilo
  Magic, (c) ser únicos entre si.
- A geração dos nomes é determinística (semente fixa) e registrada em
  `experimentos/config_v1.json`.

### 9.7 O que cada metade pode (e não pode) medir

| Métrica | Metade real (500) | Metade fictícia (500) |
|---|---|---|
| Tipo (Modo B) | ✅ tem gabarito | ❌ não tem tipo verdadeiro |
| CMC (MAE/R²) | ✅ tem gabarito | ❌ não tem gabarito |
| P/T (MAE) | ✅ tem gabarito | ❌ não tem gabarito |
| Coerência / validade de regras | ✅ | ✅ |
| Criatividade (novidade de n-gramas) | ✅ | ✅ (melhor) |
| **Memorização** | ✅ via similaridade com a carta real | ✅ **controle** |
| Avaliação humana | ✅ | ✅ (mais justa) |

**Consequência dura:** as métricas de fidelidade numérica repousam em **n=500**,
não em 1.000. As 500 fictícias **não substituem** as reais — elas respondem
outra pergunta.

### 9.8 Poder estatístico

| Medida | n | Precisão (95%) |
|---|---:|---|
| Proporção geral (real) | 500 | ± 4,38 p.p. |
| Proporção por cor | 100 | ± 9,80 p.p. |
| Proporção por tipo | 60 | ± 12,65 p.p. |
| Diferença pareada de CMC, $\sigma_d = 2$ | 500 | MDD ≈ 0,251 CMC |

Comparado com o desenho de 1.200 (MoE ±2,83, MDD 0,177), perde-se ~40% de
sensibilidade — o preço do contraste real × fictício e do balanceamento por cor.

### 9.9 Pesos de desenho e `n_eff`

Como o desenho **superamostra** artefatos e terrenos (60 de 523 e 60 de 252,
respectivamente), afirmações sobre a **população** exigem reponderação:

$$w_i = \frac{n_i / P_{\text{mono}}}{a_i / 500}$$

| Tipo | $w_i$ (média nas 5 cores) |
|---|---:|
| Creature | 1,427 |
| Instant | 1,165 |
| Sorcery | 1,081 |
| Enchantment | 1,068 |
| Artifact | 0,178 |
| **Land** | **0,086** |

Eficiência: **`n_eff` = 391/500 (78%)**. Ou seja, para estimativas
populacionais, 500 cartas valem ~391. Comparações **entre modelos** continuam
pareadas e **não** precisam de peso.

### 9.10 Alternativas rejeitadas

| Alternativa | Por quê |
|---|---|
| Pool de 700 (proposta original) | não fecha ~170/cor com ≥60/tipo sem pesos extremos |
| 6º estrato "incolor" | reduziria para ~83 por estrato e quebraria "100 por cor" |
| 50% real / 50% fictício com 2.000 no total | dobra o custo de geração sem ganho proporcional de validade |
| Metade fictícia só para criaturas | perderia o pareamento por tipo |
| Escolher as 500 "melhores" do pool | viés de seleção (D-002 §5.1) |

---

## 10. Glossário

| Termo | Definição |
|---|---|
| **Gabarito** | Valores verdadeiros (`tipo`, `cmc`, `poder`, `resistência`, `texto`, `keywords`) das cartas do holdout. Nunca entra no prompt nem no treino. |
| **Holdout** | Conjunto de cartas **retiradas do treino** para avaliar generalização. |
| **Vazamento (leakage)** | A carta avaliada (ou informação dela) participou do treino. |
| **Memorização** | O modelo reproduz o texto de uma carta vista no treino. |
| **MDD** | Menor diferença que o teste consegue detectar com o poder desejado. |
| **MoE** | Margem de erro de uma estimativa. |
| **Peso de desenho** | Fator que corrige amostragem estratificada não-proporcional. |

### 10.1 Os dois papéis do gabarito (importante)

| Campo | Papel do gabarito | Métrica |
|---|---|---|
| Tipo | **alvo direto** | acurácia, matriz de confusão, macro-F1 |
| CMC | **alvo direto** | MAE, R² |
| Poder / Resistência | **alvo direto** | MAE, Spearman |
| Texto | **NÃO é alvo** | detectar memorização + comparar distribuições com o corpus |

O texto é o ponto sutil: **acertar o texto real não é acertar** — seria copiar.
Para texto, a referência de qualidade é o **conjunto** dos textos reais
(distribuição), não a carta pareada. A carta pareada serve só como **detector de
cópia**: similaridade alta é sinal de memorização, não de qualidade.

---

## 11. Pendências abertas

| # | Pergunta | Impacto | Status |
|---|---|---|---|
| P-01 | O M3 recebe o tipo ou escolhe o tipo? | define se a acurácia de tipo do M3 é mensurável | **resolvida** → D-004 (Modo B: escolhe entre os 6) |
| P-02 | As 500 reais são subconjunto aleatório do pool? | viés de seleção | **resolvida** → D-002 (sim, sorteio semeado) |
| P-03 | VRAM da GPU local | duração da Fase 5 | **resolvida na prática** → rodada oficial completou na GPU local (§13) |
| P-04 | 3–5 colegas para avaliação humana cega? | validade da métrica "realista" | **protocolo pronto** → §15 e `avaliacao_humana/protocolo.md`; falta coleta |
| P-05 | `released_at` no fetch (holdout por edição) | melhoria futura, opcional | aberta |
| P-06 | Runner do M3: Ollama (`format`/JSON) local e `llama-cpp-python` (GBNF) no Colab — os dois aceitam o mesmo schema? | garante a camada 2 do D-003 nos dois ambientes | **parcial** → Ollama validado (§12.4); GBNF no Colab em aberto |
| P-07 | Orçamento de retry por carta (1 tentativa? 2?) | taxa de falha vs custo de geração | **resolvida** → §12.4 (1 tentativa + 1 retry) |
| P-08 | Modo B: usar nomes reais (gabarito) ou nomes sintéticos (sem gabarito, mas sem memorização)? | validade da acurácia de tipo do M3 | **resolvida na prática** → §14: as duas metades entram, e o contraste real × fictício é o controle |
| P-09 | Semear o `rng` no `generator.py` (D-005 §8.3) | sem isso o Modo B não é reprodutível | **resolvida** → `seed` adicionado e testado |
| P-10 | Confirmar que o refino do M2 **não** toca o tipo (mantém M1 ≡ M2 no Modo B) | define se o Modo B tem 2 ou 3 braços | **resolvida** → refino só de texto; M1 ≡ M2 confirmado |
| P-11 | Como gerar os 500 nomes fictícios (vocabulário, comprimento, garantia de não-colisão)? | validade do contraste real × fictício | **resolvida** → D-006 (nomes gerados e congelados no `config_v1.json`) |
| P-12 | O estrado incolor fica fora do escopo (artefatos incolores = 76,7% do tipo)? | cobertura do tipo Artifact | aberta — **decisão de escopo**, ver D-006 §9.3 |

---

## 12. Revisão de execução — 2026-10-25

Auditoria dos runners antes da rodada oficial. Os quatro pontos abaixo são
**defeitos de implementação**, não mudanças de desenho: cada um invalidava uma
propriedade que o protocolo promete (reprodutibilidade, pareamento,
rastreabilidade). Todos foram corrigidos e verificados.

### 12.1 Semente determinística (revê D-002)

D-002 pedia `seed = hash(carta_id, modelo)`. O `hash()` de `str` do CPython é
**salgado por processo** (`PYTHONHASHSEED` aleatório): a mesma carta recebia
sementes diferentes a cada execução, então a rodada **não era reproduzível**.
A correção:

- `seed = int.from_bytes(SHA-256(carta_id)[:4], "big")`, em `comum.py`;
- **M1 e M2 recebem a MESMA semente para a mesma carta**. A leitura literal de
  D-002 (`hash(carta_id, modelo)`) faria a diferença M2 − M1 misturar o efeito
  do refino com ruído de amostragem do Markov. Com a mesma semente, a base
  Markov é idêntica e a única diferença sistemática é o texto refinado — que é
  exatamente o contraste que o D-004 §7.2 quer medir;
- o M3 usa `seed + tentativa` (1 ou 2), para o retry não repetir a mesma
  amostra.

O pareamento é **verificado** por `avaliar.py`: ele confere, carta a carta, que
M1 e M2 têm `custo_mana`, `cmc`, P/T, keywords, subtipos e tipo idênticos. Se
qualquer divergência aparecer, é bug de execução e o relatório mostra o campo.

### 12.2 Campos inexistentes e códigos de cor

| Defeito | Efeito | Correção |
|---|---|---|
| `carta.cmc` não existe em `Carta` | `AttributeError` em 100% das cartas | CMC derivado de `custo_mana` (`{N}` + 1 por pip colorido), em `comum.cmc_do_custo` |
| `config_v1.json` usa `W/U/B/R/G`; `gerar_carta` espera `Branco/Azul/...` | `ValueError: Cor inválida` em todas | `comum.cor_do_pipeline` |

O CMC derivado é o mesmo nos três braços — inclusive no M3, onde é recalculado
a partir do `mana_cost` devolvido pelo LLM. O valor **declarado** pelo M3 fica
em `cmc_declarado` e a diferença vira métrica de consistência interna
(`avaliar.py` §5 do relatório), não é sobrescrita.

### 12.3 Retomada dos runners

Os três runners gravam `.jsonl` em modo **append** e pulam as `ordem` já
concluídas com sucesso. Motivo concreto: o timeout de 120 s do shell matou a
execução várias vezes no meio da amostra (a M2 chega a ~3,5 s/carta com o
`gemma3:4b`). Sem retomada, cada interrupção descartava o trabalho e, pior,
podia deixar um arquivo truncado no meio de uma linha. Linhas com `erro` **não**
contam como concluídas, então a carta é tentada de novo — o que preserva a
regra de "1.000 gerações válidas por braço".

### 12.4 Validação do M3 e orçamento de retry (resolve P-07)

O M3 roda no **Modo A** (D-004 §7.2): recebe nome, cor e **tipo real**. A
validação pós-geração rejeita apenas falhas **estruturais**:

1. `card_type` fora do enum dos 6;
2. `card_type` diferente do tipo pedido (no Modo A o tipo é dado, não é a
   métrica — devolver outro tipo é falha de geração, não "acerto criativo");
3. criatura sem P/T; instantânea/feitiço com P/T; terreno com custo de mana;
4. campos obrigatórios ausentes ou vazios.

**Alucinação de mecânica não rejeita.** Se o modelo inventa uma habilidade
inexistente, isso conta **contra** o braço na avaliação (lint/alertas). A
alternativa — rejeitar e pedir de novo — esconderia justamente a falha que o
estudo quer medir e inflaria o M3.

Orçamento: **1 tentativa + 1 retry** (no máximo 2 chamadas por carta). Com taxa
de falha estrutural $p$, o resíduo após um retry é $p^2$: para $p = 10\%$, sobra
$1\%$; para $p = 3\%$, $0{,}09\%$. Um segundo retry reduziria ainda mais, mas
triplicaria o custo de GPU para um ganho marginal — e a taxa de falha já é
reportada como métrica, não escondida. O número de tentativas fica em
`tentativas`, por carta.

No Ollama, a camada 2 do D-003 é o `format` com JSON schema (validada nesta
rodada). O caminho `llama-cpp-python` + GBNF do Colab não foi exercitado aqui;
por isso P-06 fica **parcialmente** resolvida.

### 12.5 Esquema do M3: P/T como string e terreno sem custo

Duas falhas de **formato de saída** foram encontradas em sondagem e corrigidas
antes da rodada oficial. Nenhuma muda o que é medido; as duas evitavam que uma
carta boa fosse rejeitada por um detalhe do schema.

| Defeito | Evidência | Correção |
|---|---|---|
| `power`/`toughness` como `["string","null"]` | o `gemma3:4b` emitia `'{2}'`, `'<3'`, `'/>2'` ao alternar entre os ramos do union; criaturas viravam "sem P/T" e eram rejeitadas | `{"type":"string","maxLength":5}`; não-criatura usa `""`, que o runner converte em `None`. Sondagem: `'2'/'2'` limpo em 8/8 tipos |
| terreno preenchia `mana_cost` | 6/6 terrenos (reais e fictícios) recebiam `{1}{U}`, `{2}{R}`, etc., e caíam no lint | lembrete por tipo repetido no turno do usuário (`"mana_cost" DEVE ser ""`) — é formato, não ensino de mecânica; 4/4 terrenos passaram depois |

O `mana_cost` devolvido é normalizado para o formato Oracle (`2W` → `{2}{W}`,
`none` → `""`) por `comum.normalizar_custo`. A normalização só acrescenta
chaves e trata o vazio: **não** corrige cor nem símbolo, então um custo
incoerente continua contando contra o M3. O valor bruto fica em
`carta_llm_bruta`, para auditoria.

### 12.6 Avaliação

`experimentos/avaliar.py` lê os três `.jsonl` e produz
`experimentos/metricas.json` (máquina) e `experimentos/relatorio_metricas.md`
(tabelas). Ele separa reais de fictícias em todas as métricas com gabarito,
mede novidade de 5-gramas contra `data/processed/cards.parquet`, detecta cópia
(Jaccard de 5-gramas com a carta real pareada) e aplica o lint de regras. As
comparações usam bootstrap pareado (10.000 reamostragens) para diferença de
médias e McNemar para acerto binário — ambos pareados por `ordem`.

---

## 13. Resultados da rodada oficial — 2026-10-25

A rodada usou o `config_v1.json` congelado (500 reais + 500 fictícias) e o
`gemma3:4b` via Ollama na GPU local. Os três `.jsonl` estão completos:

| Braço | Registros | Reais | Fictícias | Erros | Falha |
|---|---|---|---|---|---|
| M1 | 1.000 | 500 | 500 | 0 | 0,0% |
| M2 | 1.000 | 500 | 500 | 0 | 0,0% |
| M3 | 998 | 499 | 499 | 2 | 0,2% |

O M2 refinou 888 textos e manteve o texto Markov em 112 (11,2%) — seja por
resposta vazia, seja por o LLM devolver o mesmo texto. O M3 fechou 995 cartas
em 1 chamada e 3 em 2 chamadas; as 2 falhas restantes esgotaram as duas
tentativas e são estruturais: pedido `Artifact`, o modelo devolveu
`Enchantment` nas duas (`Susurian Dirgecraft`, ordem 411; `Ripclaw Wrappings`,
ordem 911).

**Falhas não são repostas pela reserva.** Repor uma carta do M3 exigiria
gerar M1/M2 para o mesmo `carta_id`, que não estão no conjunto pareado; isso
quebraria o pareamento para ganhar 0,2% de volume. As 2 falhas ficam no
relatório como falha medida (e ambas são do mesmo tipo: artefato →
encantamento).

> Nota operacional: ao retomar um runner, uma ordem que falhou pode gerar mais
> de uma linha de erro. Antes de avaliar, o `.jsonl` do M3 foi consolidado
> mantendo **a última linha de cada `ordem`** (1002 → 1000 linhas).

### 13.1 Estrutura (cartas reais)

| Braço | CMC MAE | CMC exato | viés CMC | Spearman | P MAE | T MAE | P/T exato | KW Jaccard | KW exato |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 1,272 | 33,0% | +0,03 | 0,385 | 1,503 | 1,497 | 10,1% | 0,438 | 43,0% |
| M2 | 1,272 | 33,0% | +0,03 | 0,385 | 1,503 | 1,497 | 10,1% | 0,438 | 43,0% |
| M3 | 1,351 | 31,7% | −0,82 | 0,353 | 1,216 | 1,367 | 11,1% | 0,108 | 9,2% |

- **CMC:** M3 empata com M1 (Δ = 0,080; IC95 −0,028 a 0,186; McNemar
  p = 0,606). O viés −0,82 mostra que o LLM **subestima** o custo, mas erra
  para cima e para baixo o suficiente para o MAE ficar igual. O CMC que o M3
  **declara** bate com o custo que ele mesmo escreveu em apenas **17,8%** dos
  casos; por isso a métrica usa o CMC derivado (`comum.cmc_do_custo`), nunca o
  campo `cmc`.
- **P/T:** o acerto exato é estatisticamente igual (11,1% vs 10,1%;
  McNemar p = 0,871), mas o **erro absoluto somado (P+T)** do M3 é menor:
  2,583 vs 3,000 (Δ = −0,417; IC95 −0,693 a −0,146). O LLM acerta a
  **magnitude** com mais frequência, não o par exato.
- **Keywords:** o número bruto do M1 engana. Das 500 cartas reais, **214**
  têm gabarito **e** geração vazios (o M1 abstém-se e acerta); o M3 abstém-se
  em apenas **40**. Restringindo às cartas em que pelo menos um lado tem
  keyword, o acerto exato desaba para **0,3% (M1)** e **1,3% (M3)** — nenhum
  dos dois resolve o campo. O M3 ainda tem dois problemas próprios: só
  **9,8%** das keywords declaradas aparecem no texto da própria carta
  (grounding) e **16,5%** não existem no vocabulário de keywords do corpus.

### 13.2 Texto (todas as cartas)

| Braço | Palavras (média) | distinct-1 | distinct-2 | Novidade 5-grama | Memória média | Memória máx. | Textos duplicados |
|---|---|---|---|---|---|---|---|
| M1 | 20,9 | 0,070 | 0,322 | 0,548 | 0,0012 | 0,143 | 4 |
| M2 | 22,1 | 0,071 | 0,325 | 0,681 | 0,0013 | 0,143 | 3 |
| M3 | 15,2 | 0,078 | 0,233 | 0,759 | 0,0025 | 0,360 | 33 |

- **M2:** o refino acrescenta 1,18 palavra em média (IC95 0,87 a 1,50) e
  **53,3%** dos 5-gramas do texto refinado não existiam no texto Markov. A
  novidade sobe de 0,548 para 0,681 sem piorar o lint (McNemar p = 1,0;
  b01 = b10 = 0). É o efeito que o desenho queria isolar.
- **M3:** é o mais curto e o mais "novo", mas a diversidade local cai
  (distinct-2 0,233 vs 0,322) e há **33 textos duplicados**, concentrados em
  terrenos e remoções: 17× `Tap: Add {U} to your mana pool.`, 15× `Target
  player discards a card. You gain 2 life.`, 14× `Target player discards a
  card.`. Ou seja: novidade alta no agregado, **colapso de template** na
  cauda.
- **Memória:** nenhum braço copia. O máximo do M3 (0,360, em `Beneath the
  Sands`) é paráfrase de uma carta real, não decalque; a média fica em 0,25%.

### 13.3 Lint e alertas

| Braço | Cartas com falha dura | Falhas | Alertas |
|---|---|---|---|
| M1 | 2 (0,2%) | `custo_sem_cor=2` | `tipo_proibido_no_texto=32` |
| M2 | 2 (0,2%) | `custo_sem_cor=2` | `keyword_ausente_no_texto=14`, `tipo_proibido_no_texto=32` |
| M3 | 12 (1,2%) | `custo_sem_cor=10`, `terreno_com_pt=2` | `keyword_ausente_no_texto=823`, `tipo_proibido_no_texto=6` |

As 2 falhas do M1 são custos `{0}` em cartas coloridas — um artefato raro do
regressor, não do Markov. As 10 do M3 são custos com a cor errada; as 2 de
terreno são terrenos com P/T (1/1 e 0/0). O alerta de keyword ausente no M3
(823/998) é o mesmo problema de grounding medido acima, visto por carta.

### 13.4 Respostas às três perguntas

| Pergunta | Resposta curta | Evidência |
|---|---|---|
| **Mais próximo dos dados reais** | **empate estrutural** entre M1 e M3, com vantagens locais | CMC: p = 0,606; P/T: M3 erra menos em magnitude (IC95 exclui 0), mas acerta o par exato igual; texto: M1/M2 têm comprimento médio mais próximo (20,9/22,1 vs 15,2 palavras) |
| **Mais criativo** | **M3**, com ressalva séria | novidade 0,759 e distinct-1 0,078, mas 33 duplicados e distinct-2 0,233 — novidade de vocabulário, não de estrutura |
| **Mais realista/assertivo** | **M1/M2** | lint 99,8% vs 98,8%; M3 tem CMC declarado incoerente (17,8%) e 82% das cartas com keyword fora do texto; M2, porém, muda mecânicas ao refinar |

**Síntese:** não há vencedor em todas as dimensões. O M2 é o melhor
**custo-benefício de texto** (fluência e novidade sem custo estrutural), mas o
refino não é fiel à mecânica. O M3 surpreende na estrutura (CMC e P/T
comparáveis a um pipeline treinado), mas falha em **coerência interna**:
declara o que não usa, repete templates e, em 2 casos, ignora o tipo pedido.
O M1 é o mais **consistente por construção**, e o menos fluente.

### 13.5 Ameaças à validade

- **Um único LLM e um único prompt** (`gemma3:4b`, temperatura 0,9 no M3 e 0,4
  no M2). Não há análise de sensibilidade a prompt/modelo; os números valem
  para esta configuração.
- **Modo B medido à parte.** Aqui o tipo foi dado ao M3; a acurácia de tipo
  (Modo B) está no §14 e não reusa nenhum número do §13.
- **Seis tipos, artefatos coloridos.** Artefatos incolores (76,7% do tipo no
  corpus) ficam fora por desenho (P-12); as 2 falhas do M3 são justamente
  artefato → encantamento.
- **Fictícias sem gabarito.** Elas só entram em métricas de texto; não há
  verdade para CMC/P/T/keywords.
- **Keywords são um alvo frouxo.** O campo `keywords` da Scryfall não é
  exaustivo e a checagem de grounding é lexical (substring), não semântica.
- **Sem avaliação humana** (P-04). "Realista" aqui é lint + distribuição, não
  julgamento de jogador.
- **Uma geração por carta.** A semente estabiliza o pipeline, mas o Ollama não
  garante o mesmo sample entre versões do modelo.

### 13.6 Reprodutibilidade

```bash
python experimentos/run_m1.py     # Markov + regressão
python experimentos/run_m2.py     # M1 + refino de texto (Ollama)
python experimentos/run_m3.py     # LLM pura (Ollama, gemma3:4b)
python experimentos/avaliar.py    # metricas.json + relatorio_metricas.md
```

Os runners são append-only e retomam pelas `ordem` concluídas; `avaliar.py`
produz as tabelas deste §13. O `.jsonl` do M3 precisa estar consolidado por
`ordem` (ver nota no início do §13) antes da avaliação.

---

## 14. Resultados do Modo B — inferência de tipo pelo nome

Rodada de 2026-10-26, com o `gemma3:4b` via Ollama. Aqui **nada do §13 é
reaproveitado**: os dois braços recebem apenas o nome e precisam decidir o
tipo. O M2 não entra como braço separado — como o refino dele só toca
`texto_regras`, a decisão de tipo é idêntica à do M1 (P-10), então o desenho
tem **dois braços**: heurística (`tipo_por_nome.py`) × LLM.

> **Transparência metodológica:** o acerto do M1/M2 aqui é da **etapa de
> inferência de tipo**, não do Markov. O gerador Markov não participa desta
> decisão; ele só entra no fallback, quando a heurística se abstém (D-005 §8.2).

### 14.1 Amostra e baselines

400 cartas sorteadas das 1.000 com semente fixa `20261025`
(`random.Random(20261025).sample`), sem estratificação. Saíram 198 reais e 202
fictícias; por tipo: 153 criatura, 60 artefato, 50 terreno, 49 instantânea, 47
feitiço, 41 encantamento. As 400 `ordem` são únicas e todos os registros têm
previsão válida dos dois braços (M3 sem falhas de geração).

| Baseline (na própria amostra) | Acurácia |
|---|---|
| Chute uniforme entre 6 | 0,167 |
| Amostrador do prior da amostra | 0,224 |
| Sempre “Criatura” | 0,383 |

A amostra tem **38,2%** de criaturas, bem abaixo dos 55,7% do corpus: o
`config_v1.json` impõe piso de 60 por tipo (D-006 §9.5), então os tipos raros
ficam sobre-representados (terreno: 12,5% na amostra, 3,5% no corpus). O
baseline de maioria **na amostra** é 0,383, não 0,548. Os números do corpus
(16,67% / 34,44% / 54,83%) ficam como referência, mas comparar contra o baseline
da amostra é o que evita inflar a dificuldade.

### 14.2 Resultado principal

| Braço | Acurácia efetiva | IC95 (Wilson) | Cobertura | Acurácia seletiva | Macro-F1 |
|---|---|---|---|---|---|
| M1/M2 (heurística) | 0,512 | [0,464; 0,561] | 0,897 | 0,524 | 0,393 |
| M3 (LLM) | 0,520 | [0,471; 0,569] | 1,000 | 0,520 | **0,431** |

- **Empate técnico na acurácia.** Δ = +0,75 p.p. para o M3, IC95 bootstrap
  [−5,00; +6,50] p.p.; McNemar pareado p = 0,864 (b01 = 70, b10 = 67). O
  intervalo contém zero nas duas leituras.
- **O M3 é mais equilibrado.** Macro-F1 0,431 vs 0,393: a acurácia bruta
  esconde que o M1 acerta muito em criatura/artefato e quase nada em
  instantânea, enquanto o M3 distribui melhor os erros.
- **A cobertura do M1 não é 100%.** Ele se absteve em **41/400** (10,2%);
  nessas cartas o fallback sorteado do pipeline acertou 0,415. A decomposição
  do D-005 §8.2 fecha exatamente:
  `0,8975 × 0,5237 + 0,1025 × 0,4146 = 0,5125`. Ou seja, ~4 p.p. da acurácia
  efetiva do M1 vêm do sorteio, não da heurística. A coluna "seletiva" mostra
  o que a heurística faz **quando responde**.

### 14.2.1 Reponderação pelo corpus — a conta que muda a leitura

A amostra segue a alocação do D-006 (piso de 60 por tipo), que
**sobre-representa os tipos raros**: terreno é 12,5% na amostra e 3,5% no
corpus; criatura é 38,2% e 55,7%. Comparar 0,512/0,520 com a maioria **da
amostra** (0,383) superestima a margem: na população real 55,7% das cartas são
criaturas, e criatura é a classe mais fácil para os dois. A pós-estratificação
do D-001 §4.9 aplicada à acurácia
(Σ `w_t` · recall_t) dá:

| Braço / baseline | Amostra | Reponderado pelo corpus |
|---|---|---|
| M1/M2 (heurística) | 0,512 | **0,590** |
| M3 (LLM) | 0,520 | **0,584** |
| Chute uniforme | 0,167 | 0,167 |
| Amostrador do prior | 0,224 | 0,354 |
| Sempre Criatura | 0,383 | 0,557 |

- A margem sobre "sempre criatura" **cai de ~13 p.p. para ~3 p.p.**
- A ordem **se inverte** (M1 0,590 vs M3 0,584; Δ = −0,53 p.p., IC95
  [−6,32; +5,15]) — mas o IC é largo o bastante para não significar nada.
- Conclusão honesta: **parte do acerto de tipo é o prior de criaturas**. O
  ganho real sobre chutar a classe majoritária é modesto nos dois braços.
- **Hipótese frágil:** a reponderação supõe que a acurácia por tipo se
  transfere para a população. Terreno e encantamento têm `n` pequeno (50 e 41)
  e ICs largos, e são justamente onde os braços mais divergem. É
  extrapolação, não medição direta.

### 14.3 Onde cada braço ganha

| Tipo real | n | M1 recall | M3 recall | Leitura |
|---|---|---|---|---|
| Criatura | 153 | **0,850** | 0,797 | ambos fortes; M1 um pouco melhor |
| Instantânea | 49 | 0,102 | **0,388** | M1 praticamente não reconhece |
| Feitiço | 47 | **0,298** | 0,255 | os dois muito fracos |
| Artefato | 60 | **0,450** | 0,283 | M1 usa pistas de "objeto" |
| Encantamento | 41 | 0,220 | **0,244** | os dois muito fracos |
| Terreno | 50 | 0,400 | **0,560** | M3 reconhece topônimos melhor |

Matrizes completas em `relatorio_modo_b.md` §4. Os padrões de erro são
diferentes e explicam o empate:

- **M1** tem pistas de "objeto" e por isso acerta 27/60 artefatos, mas nome
  próprio/animal/pessoa puxa para **criatura** e vence em 24/60; o mesmo
  mecanismo manda 19/50 terrenos para criatura. Ele ainda acerta 20/50
  terrenos quando o nome é um lugar explícito ("Island").
- **M3** manda 16/60 artefatos para encantamento e 10/50 terrenos para
  encantamento, e 18/47 feitiços para instantânea. É o mesmo erro que o Modo A
  já tinha flagrado no artefato (`Artefato → Encantamento`, §13), agora
  aparecendo como **confusão sistemática de classe**, não como falha isolada.
- **Instantânea × feitiço × encantamento** é um bloco que **nenhum dos dois
  separa**: 137/400 = 34,3% da amostra, e o melhor recall no bloco é 0,388
  (instantânea, no M3). O nome sozinho quase não carrega a diferença entre
  esses tipos — ela está no texto de regras, que o Modo B não fornece.

### 14.4 Controle de memorização (real × fictício)

| Subconjunto | n | M1 acurácia | M3 acurácia | M1 cobertura | M3 cobertura |
|---|---|---|---|---|---|
| Real | 198 | 0,535 | 0,551 | 0,929 | 1,000 |
| Fictícia | 202 | 0,490 | 0,490 | 0,866 | 1,000 |

O `gemma3:4b` nunca viu os nomes fictícios no pré-treinamento; se ele
"reconhecesse cartas", a diferença real − fictícia seria grande e exclusiva
dele. Não é o caso:

- **M3:** +6,0 p.p. em nomes reais (McNemar real p = 0,810; fictícias
  p = 1,000, com b01 = b10 = 34).
- **M1:** +4,5 p.p. na mesma direção.
- Nas fictícias os dois **empatam exatamente** (0,490).

A leitura é que os nomes reais usam vocabulário mais canônico (sufixos,
topônimos, "Angel", "Elves"), o que ajuda qualquer método — não que o LLM
esteja recuperando a carta. Isso fecha a P-08 na prática: usar as duas metades
dá o gabarito **e** o controle.

### 14.5 Leitura e limitações

1. **Decidir tipo pelo nome tem teto baixo.** ~52% na amostra, ~59% reponderado
   pelo corpus — mas o baseline "sempre criatura" reponderado é 55,7%, então o
   ganho real é de ~3 p.p. A informação que separa
   instantânea/feitiço/encantamento está no texto de regras.
2. **M3 e M1 são equivalentes em acurácia** (p = 0,864; IC95 da diferença
   contém zero nas duas ponderações). A vantagem do M3 é de **equilíbrio entre
   classes** (macro-F1), não de acerto total; a vantagem do M1 é em artefato e
   criatura. Trocar a heurística pelo LLM nesta etapa custaria 400 chamadas
   para não ganhar acurácia.
3. **O LLM não "sabe" artefato.** A confusão artefato → encantamento é o mesmo
   vício do Modo A e deve entrar como limitação do M3, não como ruído.
4. **Amostra não é o corpus.** 38,2% de criaturas contra 55,7%; por isso a
   coluna reponderada é obrigatória, e ela reduz a margem sobre o baseline de
   13 p.p. para 3 p.p. Por tipo, os ICs mais largos são terreno (n = 50) e
   encantamento (n = 41) — justamente onde os braços mais divergem.
5. **Um modelo, um prompt, temperatura 0.** Não há análise de sensibilidade;
   temperatura 0 mede a melhor decisão do modelo, não a variância.
6. **Sem cor e sem custo de mana.** Por desenho (D-005 §8.1). Um humano
   usaria a cor como pista; nenhum braço recebeu.

### 14.6 Reprodutibilidade

```bash
python experimentos/run_modo_b.py      # 400 cartas, M1 + M3 (Ollama)
python experimentos/avaliar_modo_b.py  # metricas_modo_b.json + relatorio_modo_b.md
```

`run_modo_b.py` grava `resultados_modo_b.jsonl` em append-only e retoma pelas
`ordem` concluídas, como os runners do Modo A. A amostra é reconstruída pela
semente `20261025` — não é preciso congelar a lista de 400 no repositório. O
relatório detalhado (matrizes, recalls, McNemar) é gerado por
`avaliar_modo_b.py` em `relatorio_modo_b.md`.

---

## 15. Avaliação humana cega — protocolo (P-04)

**Status: protocolo e instrumento prontos; coleta não iniciada.** Não há
resultado humano neste caderno. O que existe é o desenho, a amostra congelada e
o formulário — para que a coleta não vire um experimento improvisado.

Por que ela é necessária: "realista" no §13 é lint + distribuição + grounding.
Nenhuma dessas métricas sabe se um jogador olharia a carta e diria "isso não é
Magic". A avaliação humana é o único teste que responde isso.

### 15.1 Desenho: blocos de 4 com âncora real

Cada bloco tem **4 cartas com o mesmo nome**, em ordem aleatória e sem rótulo:
M1, M2, M3 e **a carta real**. A real é a âncora: sem ela, "nota 3,8" não tem
escala; com ela, a pergunta vira "quantos pontos abaixo da carta verdadeira".

| Dimensão | Escala | O que captura |
|---|---|---|
| Realismo | 1–5 | parece uma carta real? |
| Coerência | 1–5 | o texto funciona como regra? |
| Criatividade | 1–5 | a mecânica é interessante/não óbvia? |
| Poderia existir? | sim/não | o "passa ou não passa" |

Blocos pareados reduzem variância e permitem detectar ~0,45 ponto com 80% de
poder em **40 blocos**. O custo é ancoragem (ver 15.4).

### 15.2 Amostra

- **40 blocos = 160 cartas**, sorteadas com semente `20261026` de 440 cartas
  reais elegíveis (500 reais menos as que o M2 não refinou ou sem oracle text).
- Tipos: 22 criatura, 7 artefato, 5 instantânea, 3 feitiço, 2 encantamento,
  1 terreno. Segue a alocação do D-006 — **não** é balanceada, e o estudo não
  estima nota por tipo.
- **Recorte declarado:** só entra bloco em que o refino do M2 mudou o texto.
  Isso mede o efeito do refino quando ele acontece, não o M2 como entregue.

### 15.3 Instrumento e análise

- `experimentos/avaliacao_humana/formulario.html`: offline, autocontido, um
  bloco por tela, progresso em `localStorage`, exporta o CSV que o analisador
  espera.
- `analisar_avaliacao.py`: média por braço/dimensão com IC95, contrastes
  pareados (M2−M1, M3−M1, M3−M2 e cada um − real), "poderia existir" em p.p. e
  ICC(2,k) entre avaliadores.
- **Toda inferência é no nível do bloco**, com bootstrap de blocos: as 4 cartas
  de um bloco não são independentes.
- `--simular N` roda o pipeline com respostas sintéticas (marcadas como
  **DADOS SINTÉTICOS**) para testar o código sem humano.

### 15.4 Ameaças e mitigações

| Ameaça | Mitigação |
|---|---|
| Ancoragem ao ver 4 versões do mesmo nome | ordem aleatória; sem rótulo; efeito atinge as 4 igualmente |
| Reconhecer a carta real pelo nome | declarado; a leitura usa a distância à real, não a nota absoluta |
| Fadiga | 40 blocos em até 2 sessões; ordem de blocos aleatória |
| M2 selecionado | declarado; não é o M2 "como entregue" |
| 3–5 avaliadores | ICC + leitura qualitativa; sem generalização |
| Vazamento de rótulo | `gabarito.json` separado; `itens_cegos.json` não tem campo de braço |

### 15.5 Comandos

```bash
python experimentos/avaliacao_humana/gerar_amostra.py    # 40 blocos + formulário
# ... coleta ...
python experimentos/avaliacao_humana/analisar_avaliacao.py
```

Protocolo completo, instruções para os avaliadores e critérios de nota em
`experimentos/avaliacao_humana/protocolo.md`. Mensagens prontas de convite,
canais de envio (arquivo, link, servidor local) e checklist de coleta em
`experimentos/avaliacao_humana/convite.md`.

---

## 16. Da experimentação para a interface

A demo (`notebooks/demo.ipynb`) passou a gerar os três braços, e para isso os
prompts do M2 e do M3 saíram dos runners e foram para
`src/pipeline/llm.py`. A cópia é **literal** — mesmo system prompt, mesmas
temperaturas (`M2`: 0,4 / top_p 0,9; `M3`: 0,9 / top_p 0,95), mesmo schema e
mesma regra de 1 retry. Se o prompt da interface fosse "melhorado", o que
aparece na apresentação deixaria de ser o que foi medido.

Dois detalhes de desenho do demo:

- **O tipo do M3 é resolvido pelo pipeline e imposto ao LLM.** É o mesmo
  desenho do Modo A: tipo, nome e cor são dados; o LLM gera o resto. Sem isso,
  os três braços produziriam tipos diferentes e a comparação lado a lado não
  faria sentido.
- **M1 e M2 compartilham a semente.** Com a mesma semente, a estrutura é
  idêntica e a única diferença visível é o texto — exatamente o contraste que
  o M2 existe para medir.

Quem tem Ollama local pode conferir a reprodutibilidade: para a carta
`real::Longbow Archer` (semente `2634650529`, cor `Branco`, tipo `Criatura`),
`gerar_carta_m2` e `gerar_carta_m3` devolvem o **mesmo texto** registrado em
`resultados_m2.jsonl` e `resultados_m3.jsonl`.



