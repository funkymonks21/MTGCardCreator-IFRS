import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from pipeline.texto_markov import carregar_modelo

m = carregar_modelo()
sub = m.modelos["creature"]
p = sub.pentagramas_global
print("tipo de p:", type(p))
ks = list(p.keys())[:5]
print("primeiras chaves:", ks)
print("len de cada chave:", [len(k) if hasattr(k, "__len__") else "?" for k in ks])

# quantas chaves de tamanho 4?
n4 = [k for k in p if isinstance(k, tuple) and len(k) == 4]
print("chaves com len 4:", len(n4), "de", len(p))
cands = [k for k in n4 if sum(p[k].values()) > 0]
print("com dados:", len(cands))
ctx = cands[0]
print("ctx:", ctx)
print("sucessores globais:", dict(p[ctx]))
print("sucessores corR:", dict(sub.pentagramas_por_cor["R"].get(ctx, {})))

from pipeline.texto_markov import _combinar_ordens
dists = {o: sub._distribuicao_ordem("R", ctx, o, 0.7) for o in range(0, 5)}
for o, d in dists.items():
    print(f"ordem {o}: {len(d)} candidatos")
comb = _combinar_ordens(dists, {4: 1.0})
print("combinado com {4:1.0}:", len(comb), "candidatos")
print("chaves combinadas (10):", list(comb)[:10])
esperado = set(p[ctx]) | set(sub.pentagramas_por_cor["R"].get(ctx, {}))
print("esperado:", len(esperado))
print("extras:", len(set(comb) - esperado))
