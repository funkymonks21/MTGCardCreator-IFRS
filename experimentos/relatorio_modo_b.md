# Modo B — inferência de tipo pelo nome

Gerado por `experimentos/avaliar_modo_b.py` a partir de `resultados_modo_b.jsonl`. Entrada dos dois braços: **apenas o nome** (sem cor, sem tipo). M1 e M2 são idênticos por construção; por isso a comparação tem **dois braços**, não três.

## 1. Amostra

- 400 cartas sorteadas das 1.000 (semente fixa `20261025`), sem estratificação.
- Reais: 198; fictícias: 202.

| Tipo real | n |
|---|---|
| Criatura | 153 |
| Instantânea | 49 |
| Feitiço | 47 |
| Artefato | 60 |
| Encantamento | 41 |
| Terreno | 50 |

## 2. Métricas por braço

| Braço | Acurácia efetiva | IC95 (Wilson) | Cobertura | Acurácia seletiva | Macro-F1 |
|---|---|---|---|---|---|
| M1/M2 (heurística) | 0.512 | [0.464; 0.561] | 0.897 | 0.524 | 0.393 |
| M3 (LLM) | 0.520 | [0.471; 0.569] | 1.000 | 0.520 | 0.431 |

> `efetiva` = acertos / todas as cartas. No M1, abstenções viram o sorteio semeado do pipeline; no M3, falha de geração conta como erro. `seletiva` = acerto só entre as respostas válidas (no M1, só entre as não-abstenções). A decomposição do D-005 §8.2 (`acc_ef = c·a_sel + (1−c)·S`) supõe fallback com o prior do próprio conjunto; aqui o fallback usa o prior do corpus (≈55,7% criaturas), então a conta não fecha exatamente com o `S` da amostra.


**Diferença (M3 − M1), amostra como está:** 0.75 p.p.; IC95 bootstrap [-5.00; 6.50] p.p. O intervalo contém zero — compatível com empate.


O M1 absteve-se em **41/400** cartas (cobertura 0.897); entre essas, o fallback sorteado acertou 0.415.


## 3. Baselines (mesma amostra)

| Baseline | Acurácia |
|---|---|
| Chute uniforme entre 6 | 0.167 |
| Amostrador do prior da amostra | 0.224 |
| Sempre “Criatura” | 0.383 |

> Referência do corpus (D-004 §7.4): uniforme 16,67%, prior 34,44%, maioria (Creature) 54,83%. Os números acima são os da **amostra de 400**, que tem 38% de criaturas — por isso a maioria é menor que a do corpus. Comparar com o baseline da própria amostra é o que evita inflar a dificuldade.


### 3.1 Acurácia reponderada pelo corpus

A amostra segue a alocação do D-006 (piso de 60 por tipo), que **sobre-representa os tipos raros**: terreno é 12,5% na amostra e 3,5% no corpus; criatura é 38,2% e 55,7%. Reponderar por `w_t` = prior do corpus responde à pergunta que a amostra sozinha não responde: **que acurácia esperar numa população com a mistura real de tipos**. É a mesma pós-estratificação do D-001 §4.9, aplicada à acurácia de tipo.

| Tipo | Peso no corpus | Peso na amostra | n |
|---|---|---|---|
| Criatura | 55.7% | 38.2% | 153 |
| Instantânea | 11.6% | 12.2% | 49 |
| Feitiço | 10.9% | 11.8% | 47 |
| Artefato | 7.7% | 15.0% | 60 |
| Encantamento | 10.6% | 10.2% | 41 |
| Terreno | 3.5% | 12.5% | 50 |

| Braço / baseline | Amostra | Reponderado pelo corpus |
|---|---|---|
| M1/M2 (heurística) | 0.512 | 0.590 |
| M3 (LLM) | 0.520 | 0.584 |
| Chute uniforme | 0.167 | 0.167 |
| Amostrador do prior | 0.224 | 0.354 |
| Sempre Criatura | 0.383 | 0.557 |

**Diferença reponderada (M3 − M1):** -0.53 p.p.; IC95 bootstrap [-6.32; 5.15] p.p. O IC também contém zero.


> **Hipótese da reponderação:** cada tipo tem a mesma acurácia na amostra e na população. O IC por tipo (§4.1) mostra onde essa hipótese é mais frágil: terreno e encantamento têm `n` pequeno e intervalos largos, e são justamente os tipos em que os braços mais divergem. Logo, o número reponderado é uma **extrapolação**, não uma medição direta.


## 4. Matriz de confusão

**M1/M2 (heurística)**

| real \ previsto | Criatura | Instantânea | Feitiço | Artefato | Encantamento | Terreno |
|---|---|---|---|---|---|---|
| Criatura | 130 | 5 | 2 | 3 | 8 | 5 |
| Instantânea | 12 | 5 | 11 | 6 | 13 | 2 |
| Feitiço | 11 | 7 | 14 | 1 | 13 | 1 |
| Artefato | 24 | 3 | 2 | 27 | 2 | 2 |
| Encantamento | 10 | 2 | 11 | 4 | 9 | 5 |
| Terreno | 19 | 2 | 0 | 8 | 1 | 20 |

**M3 (LLM)**

| real \ previsto | Criatura | Instantânea | Feitiço | Artefato | Encantamento | Terreno |
|---|---|---|---|---|---|---|
| Criatura | 122 | 7 | 1 | 7 | 12 | 4 |
| Instantânea | 10 | 19 | 8 | 0 | 11 | 1 |
| Feitiço | 5 | 18 | 12 | 2 | 9 | 1 |
| Artefato | 15 | 5 | 2 | 17 | 16 | 5 |
| Encantamento | 10 | 10 | 7 | 0 | 10 | 4 |
| Terreno | 11 | 0 | 0 | 1 | 10 | 28 |

### 4.1 Recall por tipo real

| Tipo real | n | M1 recall [IC95] | M3 recall [IC95] |
|---|---|---|---|
| Criatura | 153 | 0.850 [0.785; 0.898] | 0.797 [0.727; 0.853] |
| Instantânea | 49 | 0.102 [0.044; 0.218] | 0.388 [0.264; 0.528] |
| Feitiço | 47 | 0.298 [0.187; 0.440] | 0.255 [0.153; 0.395] |
| Artefato | 60 | 0.450 [0.331; 0.575] | 0.283 [0.185; 0.408] |
| Encantamento | 41 | 0.220 [0.120; 0.367] | 0.244 [0.138; 0.393] |
| Terreno | 50 | 0.400 [0.276; 0.538] | 0.560 [0.423; 0.688] |

> IC95 de Wilson por tipo. Os intervalos **não são independentes** (as linhas da matriz somam o mesmo `n`) e não foram corrigidos para multiplicidade; servem para mostrar a precisão, não para declarar vitória por tipo.


## 5. Real × fictício (controle de memorização)

| Subconjunto | n | M1 acurácia | M3 acurácia | M1 cobertura | M3 cobertura |
|---|---|---|---|---|---|
| Real | 198 | 0.535 | 0.551 | 0.929 | 1.000 |
| Fictícia | 202 | 0.490 | 0.490 | 0.866 | 1.000 |

> Nomes fictícios são o controle limpo de memorização: o `gemma3:4b` não os viu no pré-treinamento. Se o M3 for muito melhor em nomes reais, a diferença é (ao menos em parte) conhecimento de carta, não inferência.


## 6. McNemar (M1 × M3)

| Conjunto | n | M1 | M3 | b01 (M1 erra, M3 acerta) | b10 (M1 acerta, M3 erra) | p |
|---|---|---|---|---|---|---|
| Geral | 400 | 0.512 | 0.520 | 70 | 67 | 0.8644 |
| Só reais | 198 | 0.535 | 0.551 | 36 | 33 | 0.8099 |
| Só fictícias | 202 | 0.490 | 0.490 | 34 | 34 | 1.0000 |

> McNemar bilateral com correção de continuidade; `p` é o do teste binomial exato usado em `avaliar.mcnemar`.


## 7. Leitura

- Acurácia efetiva: M1 = **0.512**, M3 = **0.520**; baseline de maioria na amostra = 0.383 e IC95 de Wilson [0.464; 0.561] (M1) / [0.471; 0.569] (M3). Na amostra os dois batem o baseline por ~13 p.p.

- **Reponderado pelo corpus** (55,7% criaturas): M1 = **0.590**, M3 = **0.584**, contra maioria de 0.557. A margem sobre o baseline cai para ~3 p.p. e a ordem dos braços se inverte — parte do acerto de tipo é, sim, o prior de criaturas.

- **Macro-F1:** M1 = 0.393 vs M3 = 0.431. A acurácia bruta empata, mas o M3 é mais equilibrado entre as seis classes.

- O M1 tem cobertura menor que 100%: parte do número efetivo vem do sorteio de fallback, não da heurística. A acurácia seletiva mostra o quanto a heurística acerta **quando responde**.

- A matriz de confusão é obrigatória: acurácia sozinha não distingue um classificador que aprendeu as 6 classes de um que colapsou em Creature.

- **Onde o M1 ganha (>5 p.p. de recall):** Criatura (0.850 vs 0.797), Artefato (0.450 vs 0.283).

- **Onde o M3 ganha (>5 p.p. de recall):** Instantânea (0.388 vs 0.102), Terreno (0.560 vs 0.400).

- **Memorização:** real − fictícia = 6.0 p.p. (M3) e 4.5 p.p. (M1). O ganho do M3 em nomes reais é da mesma ordem do ganho do M1; nas fictícias os dois empatam (0.490 vs 0.490). Não há sinal forte de memorização de carta — o que existe é vocabulário mais canônico nos nomes reais.

- McNemar geral: p = 0.8644. Não há diferença detectável entre os braços.
