import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from pipeline import PESOS_NGRAMA_PADRAO, gerar_carta  # noqa: E402
from pipeline.texto_markov import carregar_modelo  # noqa: E402

print("PESOS_NGRAMA_PADRAO =", PESOS_NGRAMA_PADRAO)

m = carregar_modelo()

# checa que os contadores novos existem
sub = m.modelos["creature"]
for attr in ("tetragramas_por_cor", "pentagramas_por_cor", "tetragramas_global", "pentagramas_global"):
    assert hasattr(sub, attr), attr
    print(f"ok: {attr} presente")

# mesma seed, varios esquemas de peso
print("\n=== mesma palavra-semente, esquemas de peso diferentes ===")
for rotulo, kwargs in [
    ("legado peso_trigrama=1.0 (so trigrama)", dict(peso_trigrama=1.0)),
    ("legado peso_trigrama=0.0 (so bigrama)", dict(peso_trigrama=0.0)),
    ("padrao PESOS_NGRAMA_PADRAO (1..5)", dict()),
    ("so pentagrama+tetragrama", dict(pesos_ngrama={4: 0.5, 3: 0.5})),
]:
    r = m.gerar_texto("R", tipo="creature", seed=7, **kwargs)
    print(f"\n--- {rotulo} ---")
    print(r["corpo"])

# carta completa com o padrao novo
c = gerar_carta("Carta Ngrama", "Verde", tipo="Criatura")
print("\n=== carta completa (padrao) ===")
print("pesos usados:", c.pesos_ngrama)
print(c.texto_regras)
