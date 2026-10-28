# Resultados — M1, M2 e M3

Gerado por `experimentos/avaliar.py` a partir dos três `.jsonl`. Os números abaixo são reprodutíveis: as sementes são SHA-256 do `carta_id` (ver `comum.py`) e M1/M2 compartilham a mesma semente.

## 1. Volume e falhas

| Braço | registros | reais | fictícias | erros | taxa de falha |
|---|---|---|---|---|---|
| M1 | 1000 | 500 | 500 | 0 | 0.0000 |
| M2 | 1000 | 500 | 500 | 0 | 0.0000 |
| M3 | 998 | 499 | 499 | 2 | 0.0020 |

## 2. Estruturais (cartas reais)

| Braço | CMC MAE | CMC RMSE | viés CMC | CMC exato | Spearman | P MAE | T MAE | P/T exato | KW Jaccard | KW exato | KW Jaccard* | KW exato* | KW válida | KW no texto | KW abstenção |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 1.272 | 1.858 | 0.03 | 0.330 | 0.385 | 1.503 | 1.497 | 0.101 | 0.438 | 0.430 | 0.017 | 0.003 | 0.933 | 1.000 | 214 |
| M2 | 1.272 | 1.858 | 0.03 | 0.330 | 0.385 | 1.503 | 1.497 | 0.101 | 0.438 | 0.430 | 0.017 | 0.003 | 0.933 | 0.940 | 214 |
| M3 | 1.351 | 2.012 | -0.82 | 0.317 | 0.353 | 1.216 | 1.367 | 0.111 | 0.108 | 0.092 | 0.030 | 0.013 | 0.835 | 0.098 | 40 |

> `KW válida` = fração das palavras-chave declaradas que existe no vocabulário de keywords do corpus. `KW no texto` = fração das keywords declaradas que aparece no próprio texto da carta (grounding). `KW Jaccard*`/`KW exato*` excluem as cartas em que **os dois lados** não têm keyword; `KW abstenção` mostra quantas são essas — sem esse corte, o acerto exato fica inflado por acordos vazios.


## 3. Texto (todas as cartas)

| Braço | palavras (média) | palavras (mediana) | distinct-1 | distinct-2 | novidade 5-grama | memória média | memória máx. | textos duplicados |
|---|---|---|---|---|---|---|---|---|
| M1 | 20.9 | 19.0 | 0.070 | 0.322 | 0.548 | 0.0012 | 0.143 | 4 |
| M2 | 22.1 | 20.0 | 0.071 | 0.325 | 0.681 | 0.0013 | 0.143 | 3 |
| M3 | 15.2 | 14.0 | 0.078 | 0.233 | 0.759 | 0.0025 | 0.360 | 33 |

> `novidade 5-grama` = fração de 5-gramas do braço ausentes no corpus (maior = menos cópia). `memória` = Jaccard de 5-gramas com a carta real pareada; não é métrica de qualidade — é detector de cópia.

> M3, duplicados mais frequentes: 17× `Tap: Add {U} to your mana pool.`; 15× `Target player discards a card. You gain 2 life.`; 14× `Target player discards a card.`.


## 4. Lint de regras

| Braço | cartas com falha | taxa OK | falhas por regra |
|---|---|---|---|
| M1 | 2 | 0.998 | custo_sem_cor=2 |
| M2 | 2 | 0.998 | custo_sem_cor=2 |
| M3 | 12 | 0.988 | custo_sem_cor=10, terreno_com_pt=2 |

**Alertas (não invalidam a carta):**

| Braço | alertas |
|---|---|
| M1 | tipo_proibido_no_texto=32 |
| M2 | keyword_ausente_no_texto=14, tipo_proibido_no_texto=32 |
| M3 | keyword_ausente_no_texto=823, tipo_proibido_no_texto=6 |

## 5. M3 — consistência e retry

- CMC declarado consistente com o custo: **0.178**
- Chamadas por carta: `{"1": 995, "2": 3}`
- Cartas geradas por tipo: `{"artifact": 118, "creature": 400, "enchantment": 120, "instant": 120, "land": 120, "sorcery": 120}`

- Falhas por tipo: `{"artifact": 2}`

- Falha: ordem 411 — Susurian Dirgecraft (artifact): tipo ignorado: pedido 'Artifact', veio 'Enchantment'

- Falha: ordem 911 — Ripclaw Wrappings (artifact): tipo ignorado: pedido 'Artifact', veio 'Enchantment'


## 6. Comparações pareadas

**Integridade M1 × M2:** 1000 cartas pareadas; divergências fora do texto: `{}` (esperado: tudo zero, pois a semente é a mesma).

**Efeito do refinamento (M2):** Δpalavras médio = 1.18 (IC95 0.87 a 1.50); fração de 5-gramas novos = 0.533; McNemar do lint: p = 1.0000 (b01=0, b10=0).

**M3 × M1 no CMC:** MAE M3 = 1.351 vs M1 = 1.271; Δ = 0.080 (IC95 -0.028 a 0.186); McNemar do acerto exato: p = 0.6057.

**M3 × M1 no P/T:** erro absoluto somado (P+T) M3 = 2.583 vs M1 = 3.000; Δ = -0.417 (IC95 -0.693 a -0.146); McNemar do acerto exato: p = 0.8714 (b01=20, b10=18).

**Memória média por braço:** M1=0.0012, M2=0.0013, M3=0.0025.


## 7. Leitura rápida

- M1 e M2 compartilham estrutura por construção; qualquer diferença estrutural é bug de execução, não resultado.
- O M2 só pode ganhar/perder em **texto** (comprimento, novidade, lint, memória) — é o que o desenho isola.
- O M3 é o único braço que pode errar estrutura (CMC, P/T, terreno, coerência cor × custo) — e é isso que o lint mede.
- Métricas de texto não têm alvo pareado: valem como distribuição, não como acerto.

- **Keywords:** o acerto exato bruto do M1 (0.430) inclui 214 cartas em que os dois lados são vazios; o M3 abstém-se em apenas 40. Restringindo às cartas com keyword em algum lado, os dois caem para 0.003 (M1) e 0.013 (M3) — nenhum dos dois resolve o campo.
- **Grounding do M3:** só 0.098 das keywords declaradas aparecem no próprio texto da carta; no M1/M2 o pipeline garante a consistência por construção.

- **CMC declarado do M3:** bate com o custo em 0.178 dos casos — por isso a métrica usa o CMC derivado, nunca o declarado.
