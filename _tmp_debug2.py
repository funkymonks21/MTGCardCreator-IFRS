import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
import pipeline.texto_markov as tm

orig = tm._combinar_ordens


def espiao(dists, pesos):
    print(">> _combinar_ordens chamado com pesos =", pesos)
    print(">> ordens disponiveis:", {o: len(d) for o, d in dists.items() if d})
    return orig(dists, pesos)


tm._combinar_ordens = espiao

m = tm.carregar_modelo()
sub = m.modelos["creature"]
p = sub.pentagramas_global
ctx = ("<INICIO>", "landfall", "whenever", "a")

# Reimplementa a distribuicao inline para ver de onde vem o problema
print("--- chamando _distribuicao_proxima_palavra com {4:1.0} ---")
d = sub._distribuicao_proxima_palavra("R", ctx, 0.7, pesos_ngrama={4: 1.0})
print("resultado:", len(d), "candidatos")

# Checa PESOS_NGRAMA_PADRAO
print("PESOS_NGRAMA_PADRAO =", tm.PESOS_NGRAMA_PADRAO)

# Reproduz manualmente a logica
dist_ordem = {o: sub._distribuicao_ordem("R", ctx, o, 0.7) for o in range(0, 5)}
disp = {o: dd for o, dd in dist_ordem.items() if dd}
pesos = {4: 1.0}
pesos_ef = {o: pesos.get(o, 0.0) for o in disp}
print("pesos_ef manual:", pesos_ef, "soma:", sum(pesos_ef.values()))
