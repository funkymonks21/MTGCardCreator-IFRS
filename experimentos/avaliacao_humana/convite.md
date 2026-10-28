# Como enviar para os avaliadores

Guia curto para o coordenador. O instrumento é **um único arquivo**,
`formulario.html` (54 KB, sem internet, sem servidor, sem instalação).

> **Nunca envie** `gabarito.json`, `itens_cegos.json`, `resultados_*.jsonl` ou
> qualquer arquivo do estudo. O avaliador só precisa do `formulario.html` —
> ele já contém as 160 cartas embutidas.

## 1. Escolha do canal

| Situação | Como enviar | Observação |
|---|---|---|
| **Mesma sala/laboratório** | copie só o `formulario.html` para uma pasta vazia e rode `python -m http.server 8000` nela; passe `http://SEU-IP:8000/formulario.html` | Melhor UX: abre no navegador de qualquer máquina da rede, sem baixar nada. **Não** sirva a pasta `avaliacao_humana/` inteira: ela tem o `gabarito.json`. As respostas continuam no navegador de cada um. |
| **Remoto, 3–5 pessoas** | Anexar o `.html` no WhatsApp/Telegram/Discord ou link do Drive | Rápido e suficiente. **Não** mande por Outlook corporativo: ele bloqueia `.html`. |
| **Remoto, quer um link** | Subir **só** o `formulario.html` no Netlify Drop, Cloudflare Pages ou GitHub Pages | Melhor para quem vai responder no celular. Só o formulário vai ao ar; o gabarito fica local. |
| **E-mail** | Anexar **zipado** (`formulario.zip`) | Outlook bloqueia `.html`; o zip passa. Avise que precisa extrair antes de abrir. |

**Evite Google Forms/Microsoft Forms.** Seriam 160 cartas × 4 perguntas
montadas à mão. O `modelo_respostas.csv` existe só como plano B, não como
caminho recomendado.

## 2. Antes de enviar

- [ ] O avaliador **joga Magic** (é o critério do estudo).
- [ ] O avaliador **não participou do projeto**.
- [ ] O arquivo abre no **Chrome, Edge ou Firefox no desktop**. No iPhone, o
      visualizador de arquivos (Quick Look) costuma abrir `.html` **sem rodar
      JavaScript** — nesse caso, use a opção de link (tabela acima).
- [ ] Você definiu **prazo** e **canal de devolução** (ex.: "até sexta, por
      e-mail").
- [ ] Você não vai abrir o `gabarito.json` antes de receber todas as respostas.

## 3. Mensagem para colar (WhatsApp/Telegram)

> Oi! Topa participar de um teste cego de umas cartas de Magic geradas por
> computador? São 40 blocos, ~45–60 min, dá para fazer em duas sessões.
>
> Em cada bloco aparecem **quatro versões do mesmo nome de carta**: três foram
> geradas por modelos e uma é a carta real. Você não sabe qual é qual — e não
> precisa saber. Para cada uma, dê notas de 1 a 5 para **realismo**,
> **coerência** e **criatividade**, e diga se ela **poderia existir**.
>
> É só abrir o arquivo `formulario.html` no Chrome/Edge/Firefox do computador
> (não abra pela pré-visualização do WhatsApp/e-mail; salve e abra). O progresso
> é salvo sozinho, pode parar e voltar no mesmo navegador.
>
> No fim, clique em **Baixar minhas respostas (CSV)** e me mande o arquivo.
> Sem pressa e sem resposta "certa": julgue como jogador. Valeu!

## 4. Mensagem para colar (e-mail)

> Assunto: Teste cego de cartas geradas — 40 blocos (~45 min)
>
> Oi, [nome],
>
> Estou fechando a avaliação do gerador de cartas e preciso de jogadores para
> um teste cego. São 40 blocos; em cada um há quatro versões do mesmo nome de
> carta — três geradas por modelos, uma real — e você não sabe qual é qual.
>
> Para cada carta: notas de 1 a 5 em realismo, coerência e criatividade, e um
> sim/não em "poderia existir?". Leva ~45–60 min no total e pode ser feito em
> duas sessões (o progresso fica salvo no navegador).
>
> Instruções:
> 1. Baixe o anexo `formulario.html` e abra no Chrome, Edge ou Firefox do
>    computador. Não preencha pela pré-visualização do e-mail.
> 2. Preencha os blocos. Pode fechar e voltar no mesmo navegador.
> 3. No fim, clique em "Baixar minhas respostas (CSV)" e responda este e-mail
>    com o arquivo.
>
> Prazo: [data]. Qualquer dúvida, me chama. Obrigado!

## 5. Depois que as respostas chegarem

1. Salvar os CSVs em `experimentos/avaliacao_humana/respostas/` (a pasta é
   ignorada pelo git).
2. Conferir que cada arquivo tem **161 linhas** (cabeçalho + 160 cartas):
   ```powershell
   Get-ChildItem experimentos\avaliacao_humana\respostas\*.csv |
     ForEach-Object { "$($_.Name): $((Get-Content $_).Count) linhas" }
   ```
3. Rodar:
   ```bash
   python experimentos/avaliacao_humana/analisar_avaliacao.py
   ```
4. Ler a seção **"Participação e integridade"** do relatório. Se aparecer
   "problema(s) de leitura", o script diz o arquivo e a linha — peça o reenvio
   só se faltar resposta (respostas incompletas são reportadas como cobertura
   por avaliador, não como erro).

## 6. Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| A página abre em branco / sem estilo | abriu pela pré-visualização do e-mail ou do Drive | salvar no disco e abrir no navegador |
| Botão "Próximo bloco" não avança | falta nota em alguma das 4 cartas | o formulário avisa "Faltam notas neste bloco" |
| Progresso sumiu | trocou de navegador, usou aba anônima ou limpou dados | usar sempre o mesmo navegador/perfil; não usar anônima |
| Não abre no celular | Quick Look do iOS não executa JS | usar a versão hospedada por URL |
| CSV vem vazio/estranho no Excel | separador/acentuação | o arquivo é UTF-8 com vírgula; abrir no Excel via "Dados > De texto/CSV" |
| Outlook recusa o anexo | `.html` é bloqueado | enviar `.zip` ou usar link |

## 7. Privacidade e consentimento

- O formulário pede só um **nome ou apelido**; não coleta IP, e-mail nem
  qualquer dado pessoal.
- As respostas ficam no `localStorage` do navegador do avaliador até ele
  exportar o CSV — nada é enviado automaticamente.
- Avise no convite que as respostas serão usadas **agregadas** no relatório do
  estudo e que o nome/apelido serve só para controlar quem já respondeu.
