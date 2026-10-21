"""Compara esquemas de peso pela 'coerencia': fracao dos n-gramas gerados
que existem no corpus. Usa o MESMO tokenizador do modelo para montar o
vocabulario de n-gramas, para a comparacao ser valida."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from pipeline.texto_markov import (  # noqa: E402
    TOKEN_AUTORREFERENCIA,
    carregar_modelo,
)

m = carregar_modelo()
sub = m.modelos["creature"]

# Reconstroi os n-gramas gerados a partir da MESMA tokenizacao que o modelo
# usa internamente: em vez de re-tokenizar o texto final, pedimos ao modelo
# os tokens crus. Fazemos isso monkeypatchando _juntar_tokens para capturar
# a lista de tokens. Mais simples: usar o proprio modelo para gerar e
# tokenizar com a funcao interna _tokenizar.

from pipeline.texto_markov import _tokenizar  # noqa: E402


def vocab_ngram(tam):
    v = set()
    mapas = {
        2: (sub.bigramas_por_cor, sub.bigramas_global),
        3: (sub.trigramas_por_cor, sub.trigramas_global),
        4: (sub.tetragramas_por_cor, sub.tetragramas_global),
        5: (sub.pentagramas_por_cor, sub.pentagramas_global),
    }
    por_cor, glob = mapas[tam]
    for d in por_cor.values():
        v.update(d.keys())
    v.update(glob.keys())
    return v


vocabs = {t: vocab_ngram(t) for t in (2, 3, 4, 5)}

esquemas = {
    "legado so-bigrama": dict(peso_trigrama=0.0),
    "legado so-trigrama": dict(peso_trigrama=1.0),
    "peso_trigrama=0.85 (antigo)": dict(peso_trigrama=0.85),
    "PESOS_NGRAMA_PADRAO (novo)": dict(),
    "so 3-5 gramas": dict(pesos_ngrama={4: 1.0, 3: 1.0, 2: 1.0}),
}

N = 150
print(f"{'esquema':<30} {'2g':>7} {'3g':>7} {'4g':>7} {'5g':>7}")
for rotulo, kwargs in esquemas.items():
    totais = {t: 0 for t in (2, 3, 4, 5)}
    batidas = {t: 0 for t in (2, 3, 4, 5)}
    for i in range(N):
        corpo = sub.gerar_texto("R", peso_cor=0.7, seed=5000 + i, **kwargs)["corpo"]
        # substitui o nome de autorreferencia de volta pelo token, para casar
        # com o vocabulario do corpus
        tokens = _tokenizar(corpo.replace("This creature", TOKEN_AUTORREFERENCIA))
        for t in (2, 3, 4, 5):
            for j in range(len(tokens) - (t - 1)):
                gram = tuple(tokens[j:j + t])
                totais[t] += 1
                if gram in vocabs[t]:
                    batidas[t] += 1
    linha = " ".join(
        f"{(batidas[t] / totais[t] if totais[t] else 0):6.1%}" for t in (2, 3, 4, 5)
    )
    print(f"{rotulo:<30} {linha}")
