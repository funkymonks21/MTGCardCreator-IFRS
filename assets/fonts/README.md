# Fontes da moldura

Fontes usadas por `src/pipeline/imagem.py` para desenhar a carta. Todas são
**livres** (SIL Open Font License), então podem ser redistribuídas neste
repositório. A fonte oficial das cartas de Magic (Beleren/Matrix) é
proprietária da Wizards of the Coast e **não** é usada aqui — estas são
escolhas livres de aparência próxima.

| Arquivo            | Fonte         | Uso                    | Licença | Origem |
|--------------------|---------------|------------------------|---------|--------|
| `Cinzel.ttf`       | Cinzel        | títulos (nome, tipo, P/T) | SIL OFL 1.1 | [Google Fonts](https://fonts.google.com/specimen/Cinzel) |
| `EBGaramond.ttf`   | EB Garamond   | corpo do texto         | SIL OFL 1.1 | [Google Fonts](https://fonts.google.com/specimen/EB+Garamond) |

São arquivos **variáveis** (`wght`), baixados do repositório
[google/fonts](https://github.com/google/fonts). Se os arquivos não estiverem
presentes (por exemplo, um clone sem os assets), `_carregar_fonte` cai na
fonte padrão do Pillow e a carta continua sendo gerada, só com tipografia
menos elaborada.
