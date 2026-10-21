"""Teste direto do mecanismo multi-ordem (sem proxy fragil)."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from pipeline.texto_markov import (  # noqa: E402
    PESOS_NGRAMA_PADRAO,
    _combinar_ordens,
    carregar_modelo,
)

# 1) _combinar_ordens: pesos normalizados so sobre ordens disponiveis
d_unig = Counter({"a": 1})
d_bigr = Counter({"b": 1})
d_penta = Counter({"c": 1})
comb = _combinar_ordens({0: d_unig, 1: d_bigr, 4: d_penta}, {4: 0.5, 1: 0.3, 0: 0.2})
tot = sum(comb.values())
assert abs(comb["c"] / tot - 0.5) < 1e-9
assert abs(comb["b"] / tot - 0.3) < 1e-9
assert abs(comb["a"] / tot - 0.2) < 1e-9
print("ok: _combinar_ordens normaliza entre as ordens disponiveis")

# ordem ausente nao entra na conta; restantes renormalizam
comb2 = _combinar_ordens({1: d_bigr, 4: Counter()}, {4: 0.5, 1: 0.3, 0: 0.2})
assert set(comb2) == {"b"}
print("ok: ordem sem dados e ignorada")

# 2) modelo real: acha um contexto de 4 tokens que EXISTA no pentagrama e
#    confirma que, com peso so-pentagrama, a proxima palavra fica restrita a ele
m = carregar_modelo()
sub = m.modelos["creature"]
penta = sub.pentagramas_global
ctx = None
for k in penta:
    if isinstance(k, tuple) and len(k) == 4 and sum(penta[k].values()) > 0:
        ctx = k
        break
assert ctx is not None, "nenhum pentagrama encontrado"

esperado = set(penta[ctx]) | set(sub.pentagramas_por_cor["R"].get(ctx, {}))
dist = sub._distribuicao_proxima_palavra("R", ctx, peso_cor=0.7, pesos_ngrama={4: 1.0})
assert set(dist).issubset(esperado), set(dist) - esperado
print(f"ok: contexto de 4 tokens {ctx} -> proxima palavra restrita ao pentagrama")

# contexto SEM pentagrama: ordem 4 e ignorada e caem as demais
ctx2 = tuple(list(ctx)[:3])  # contexto de 3 tokens
dist3 = sub._distribuicao_proxima_palavra("R", ctx2, 0.7, pesos_ngrama={4: 1.0})
assert not dist3, "ordem 4 sem dados deveria dar dist vazia quando e a unica com peso"
print("ok: contexto sem pentagrama e ignorado quando so ele tem peso")

# 3) padrao: usando PESOS_NGRAMA_PADRAO a distribuicao inclui massas de
#    varias ordens (mais palavras candidatas que so-trigrama)
dist_padrao = sub._distribuicao_proxima_palavra("R", ctx, 0.7)
dist_tri = sub._distribuicao_proxima_palavra("R", ctx, 0.7, peso_trigrama=1.0)
print(f"ok: candidatos padrao={len(dist_padrao)} vs so-trigrama={len(dist_tri)}")
print("PESOS_NGRAMA_PADRAO:", PESOS_NGRAMA_PADRAO)
print("\nTODOS OS TESTES PASSARAM")
