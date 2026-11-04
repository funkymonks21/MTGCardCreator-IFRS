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
| `mana.ttf`         | Mana          | símbolos de mana/habilidade | SIL OFL 1.1 | [mana.andrewgioia.com](https://mana.andrewgioia.com) |

A `mana.ttf` é usada por `src/pipeline/simbolos.py` só para o **glifo** de
cada símbolo (sol, gota, caveira, chama, árvore, seta...). O disco colorido
e o contorno são desenhados em código. Os desenhos dos símbolos de mana são
marca da Wizards of the Coast: aqui entram apenas como acabamento visual de
um projeto acadêmico e fan-made, sem fins comerciais. Se o arquivo for
removido, `simbolos.py` cai no desenho próprio em Pillow e nada quebra.

São arquivos **variáveis** (`wght`), baixados do repositório
[google/fonts](https://github.com/google/fonts). Se os arquivos não estiverem
presentes (por exemplo, um clone sem os assets), `_carregar_fonte` cai na
fonte padrão do Pillow e a carta continua sendo gerada, só com tipografia
menos elaborada.
