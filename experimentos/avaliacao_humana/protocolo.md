# Avaliação humana cega — protocolo (P-04)

Status: **protocolo e instrumento prontos; coleta não iniciada.**
Abre a pendência P-04 do caderno (`experimentos/README.md` §11). É o único
teste do estudo que não pode ser automatizado: "realista" e "coerente" são
julgamentos de jogador, e as métricas do §13 são proxies (lint, distribuição,
grounding), não substitutos.

## 1. Pergunta

> Um jogador experiente, **sem saber qual braço gerou a carta**, julga as
> cartas geradas como mais ou menos parecidas com uma carta real? E quão longe
> elas ficam da carta verdadeira?

Três dimensões separadas, porque elas divergem no §13:

| Dimensão | Pergunta ao avaliador | O que o §13 mede hoje |
|---|---|---|
| **Realismo** | "Parece uma carta real de Magic?" | lint, CMC, P/T, comprimento |
| **Coerência** | "O texto é internamente coerente?" | lint, grounding de keyword, CMC declarado |
| **Criatividade** | "A mecânica é interessante/não óbvia?" | novidade de 5-grama, distinct-n |

Além disso, um binário: **"poderia existir?"** (sim/não). É a pergunta que um
jogador responde sem pensar em nota e que captura o "passa ou não passa".

## 2. Desenho: blocos com âncora real

Cada bloco tem **4 cartas com o mesmo nome**:

- M1 (Markov + regressão);
- M2 (M1 + refino de texto);
- M3 (LLM pura);
- **a carta real** — âncora de calibração, sem rótulo.

O avaliador vê as 4 em ordem aleatória, sem identificação. Por que a âncora:

- nota absoluta não tem escala ("3,8 é bom?");
- com a real no mesmo bloco, a pergunta vira "quantos pontos abaixo da carta
  verdadeira este braço fica?";
- dá para medir **taxa de vitória contra a real** por bloco.

Por que os mesmos 4 nomes no bloco (e não cartas independentes):

- remove a variância de "este nome é fácil/difícil";
- permite comparação pareada, que é o que dá poder com 40 blocos;
- **risco:** ancoragem — ver §8.

Por que só nomes reais: o bloco precisa da carta real, e nomes fictícios não têm
`oracle_text` no corpus.

## 3. Amostra

- **40 blocos** = 160 cartas = 40 × 4.
- Sorteio com semente fixa `20261026` de `resultados_m1/m2/m3.jsonl`, exigindo:
  - `tipo_experimento == "real"`;
  - `refinado == True` no M2;
  - texto não vazio nos três braços;
  - nome presente no corpus (para a âncora).
- Universo elegível: **440** das 500 cartas reais.
- Distribuição dos 40 nomes por tipo: 22 criatura, 7 artefato, 5 instantânea,
  3 feitiço, 2 encantamento, 1 terreno. Cores: 14 R, 8 W, 7 B, 6 U, 5 G.
- A amostra é **de conveniência por tipo** (segue a alocação do D-006, não é
  balanceada): o estudo **não** estima nota por tipo, e sim a média geral e o
  contraste entre braços.

**Recorte declarado:** só entram blocos em que o refino do M2 mudou o texto.
Isso mede o efeito do refino **quando ele acontece**, não o M2 como entregue
(11,2% dos casos ficam de fora e não seriam distinguíveis do M1).

## 4. Instrumento

Formulário offline autocontido: `formulario.html` (abre no navegador, sem
servidor, sem rede). Ele:

- mostra um bloco por tela, 4 cartas lado a lado;
- pede as três notas (1–5) e o sim/não para cada carta;
- exige o bloco completo para avançar;
- salva o progresso em `localStorage` (pode fechar e voltar);
- exporta CSV com `avaliador,item_id,realismo,coerencia,criatividade,existe`.

Definições na tela (versão curta):

| Nota | Realismo | Coerência | Criatividade |
|---|---|---|---|
| 5 | indistinguível de carta real | texto funciona como regra, sem ambiguidade | mecânica original, não óbvia |
| 4 | parece carta real com um detalhe estranho | funciona, com uma frase confusa | ideia incomum, execução simples |
| 3 | parece carta de um jogo parecido | dá para entender, mas precisa de interpretação | variação de algo conhecido |
| 2 | parece texto gerado | frases se contradizem ou não fecham | clichê com uma troca de palavra |
| 1 | não parece uma carta | impossível de executar | nenhuma ideia própria |

O formulário **não** mostra keywords, CMC, cor do frame nem nome do braço.
Mostra só o que um jogador leria na carta: nome, custo, tipo, texto e P/T.

Se preferirem Google Forms, `modelo_respostas.csv` tem o cabeçalho exato que o
analisador espera.

## 5. Procedimento

1. Convidar **3 a 5** pessoas que jogam Magic (não envolvidas no projeto).
2. Enviar `formulario.html` + uma frase de contexto (sem revelar os braços).
3. Cada avaliador faz os 40 blocos, em uma ou duas sessões (~45–60 min no
   total; ~15–20 s por carta).
4. Receber os CSVs e colocá-los em `experimentos/avaliacao_humana/respostas/`
   (a pasta é ignorada pelo git).
5. Rodar `python experimentos/avaliacao_humana/analisar_avaliacao.py`.
6. Só então abrir `gabarito.json` — na prática o analisador é quem abre.

Instrução-padrão para os avaliadores (sugestão):

> Você vai avaliar 40 blocos. Em cada bloco há quatro versões do mesmo nome de
> carta: três foram geradas por modelos e uma é a carta real. Você não sabe
> qual é qual. Dê notas de 1 a 5 para realismo, coerência e criatividade, e
> diga se a carta poderia existir. Não existe resposta certa — julgue como
> jogador. O progresso é salvo no navegador; se precisar parar, volte pelo
> mesmo navegador.

Mensagens prontas de convite (WhatsApp e e-mail), checklist de envio,
alternativas de canal (arquivo, link, servidor local) e o que fazer com os
CSVs recebidos estão em `experimentos/avaliacao_humana/convite.md`.

## 6. Poder e concordância

- **Poder:** com 40 blocos, o contraste pareado detecta ~0,45 ponto (escala
  1–5) com 80% de poder, assumindo desvio-padrão da diferença ≈ 1,0 e α = 0,05
  (bilateral). Para 0,5 ponto, ~32 blocos bastariam; 40 dá margem.
- **Concordância:** ICC(2,k) por dimensão (`≥ 0,75` boa, `0,50–0,75` moderada,
  `< 0,50` fraca). Se o ICC de criatividade ficar baixo, a dimensão entra no
  relatório como **não confiável**, não como resultado.
- **Amostra de avaliadores:** 3–5 é conveniência. Os ICs refletem itens e
  blocos, não a população de jogadores.

## 7. Análise (o que o script faz)

- Toda inferência é no **nível do bloco**, com 10.000 reamostragens de blocos.
  As 4 cartas de um bloco não são independentes; tratar cada carta como
  observação independente subestimaria o erro padrão.
- Médias por braço e dimensão, com IC95 por bootstrap de blocos.
- Contrastes pareados M2−M1, M3−M1, M3−M2 e cada braço − REAL: Δ, IC95 e teste
  t pareado sobre as médias de bloco; contagem de blocos a favor/b/empate.
- "Poderia existir?": proporção com IC e contraste em p.p.
- ICC(2,k) entre avaliadores.
- Saídas: `relatorio_avaliacao_humana.md` e `metricas_avaliacao_humana.json`.

Teste do pipeline sem humanos:

```bash
python experimentos/avaliacao_humana/analisar_avaliacao.py --simular 3
```

Isso gera respostas sintéticas em `respostas_simuladas/` e roda a análise
inteira. Serve para verificar o código — **não** para produzir resultado.

## 8. Ameaças à validade e mitigações

| Ameaça | Mitigação |
|---|---|
| Ancoragem: ver 4 versões do mesmo nome | ordem aleatória dentro do bloco; rótulo nunca aparece; análise é pareada por bloco, então a âncora afeta os quatro igualmente |
| Reconhecer a carta real pelo nome | aceito e declarado; um jogador pode lembrar "essa é a Serra Angel"; a taxa de vitória contra a real é lida com isso em mente |
| Fadiga nas últimas cartas | progresso salvo; 40 blocos cabem em 2 sessões; a ordem dos blocos é aleatória, então fadiga não favorece um braço |
| M2 selecionado (só refinados) | declarado no §3; não é o M2 "como entregue" |
| 3–5 avaliadores | ICC + leitura qualitativa; sem generalização para "jogadores" |
| Escalas sem treino | âncora real dá referência concreta; instruções com âncoras de nota |
| Vazamento de rótulo | `gabarito.json` separado; nenhum campo de braço em `itens_cegos.json` (verificável por `grep`) |

## 9. Arquivos

| Arquivo | Papel |
|---|---|
| `gerar_amostra.py` | sorteia os 40 blocos e escreve os insumos |
| `convite.md` | mensagens prontas, canais de envio e checklist de coleta |
| `itens_cegos.json` | o que o avaliador vê (sem braço) |
| `formulario.html` | instrumento offline + exportação CSV |
| `gabarito.json` | item → braço; **não abrir antes de coletar** |
| `modelo_respostas.csv` | cabeçalho para Google Forms |
| `analisar_avaliacao.py` | análise e relatório |
| `respostas/` | CSVs recebidos (ignorado pelo git) |

## 10. Reprodutibilidade

```bash
python experimentos/avaliacao_humana/gerar_amostra.py   # 40 blocos, 160 cartas
# ... coleta ...
python experimentos/avaliacao_humana/analisar_avaliacao.py
```

A amostra é reconstruída pela semente `20261026` — não é preciso versionar a
lista de blocos. `itens_cegos.json` e `gabarito.json` são versionados para
auditoria do sorteio e para congelar o instrumento depois que a coleta começar.
