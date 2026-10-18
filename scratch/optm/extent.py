"""Body bounding box against the box faces (scratch/optm, 2026-10-16)."""
import dataclasses, json, sys
import jax.numpy as jnp, numpy as np
sys.path.insert(0, "examples/mesh")
import optimize as O
from jax_dsmc.mesh import levelset as L
tag = sys.argv[1]
d = f"examples/mesh/results/optimize/{tag}"
h = json.load(open(f"{d}/history.json"))
grid, ls = O.initial_levelset(h["config"])
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
xs = np.arange(lo[0], hi[0], 0.05); ys = np.arange(lo[1], hi[1], 0.05)
xx, yy = np.meshgrid(xs, ys, indexing="ij"); pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], 1))
print("box", lo, hi)
for k in list(range(0, len(h["rows"]) + 1, 10)) + [int(a) for a in sys.argv[2:]]:
    c = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{d}/coef_{k}.npy")))
    inside = np.asarray(L.value(c, pts)).reshape(xx.shape) < 0
    X, Y = xx[inside], yy[inside]
    # thickness: min over x-columns of the body's vertical extent, in the middle 80 %
    cols = [inside[i].sum() * 0.05 for i in range(len(xs)) if inside[i].any()]
    print(f"{k:3d} x {X.min():6.2f}..{X.max():6.2f}  y {Y.min():5.2f}..{Y.max():5.2f}  max thick {max(cols):.2f}")
