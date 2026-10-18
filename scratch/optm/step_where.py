"""Where does each step of a run move the surface (2026-10-17)?  Per iterate k: the knot with
the largest |coef_{k+1} - coef_k| (x, y, its phi), the share of the step's squared size within
1.5 m of that knot, and the body's nose x (min x of phi < 0)."""

import dataclasses, json, sys
import jax.numpy as jnp, numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O
from jax_dsmc.mesh import levelset as L

tag = sys.argv[1]
d = f"examples/mesh/results/optimize/{tag}"
h = json.load(open(f"{d}/history.json"))
grid, ls = O.initial_levelset(h["config"])
x = np.asarray(L.knots(ls))
n = len(h["rows"])
out = []
prev = np.load(f"{d}/coef_0.npy").ravel()
for k in range(n):
    nxt = np.load(f"{d}/coef_{k + 1}.npy").ravel()
    dc = nxt - prev
    c = dataclasses.replace(ls, coef=jnp.asarray(prev.reshape(ls.coef.shape)))
    phi = np.asarray(L.value(c, jnp.asarray(x)))
    i = int(np.argmax(np.abs(dc)))
    near = np.linalg.norm(x - x[i], axis=1) < 1.5
    body = x[phi < 0]
    nose = body[np.argmin(body[:, 0])]
    # signed normal motion near the nose: mean dc over band knots with x < nose + 1.5 (dc > 0 = surface recedes)
    band = (np.abs(phi) < 0.5) & (x[:, 0] < nose[0] + 1.5)
    out.append(
        {
            "k": k,
            "x": x[i].tolist(),
            "phi": float(phi[i]),
            "dc": float(dc[i]),
            "local": float((dc[near] ** 2).sum() / (dc**2).sum()),
            "nose": nose.tolist(),
            "nose_dc": float(dc[band].mean()),
        }
    )
    prev = nxt
json.dump(out, open(f"scratch/optm/step_where_{tag}.json", "w"))
for o in out:
    print(
        f"{o['k']:3d} max-move knot x {o['x'][0]:6.2f} y {o['x'][1]:5.2f} phi {o['phi']:+.2f} local {o['local']:.2f} nose x {o['nose'][0]:5.2f} y {o['nose'][1]:4.2f} nose_dc {o['nose_dc'] * 1e3:+6.1f} mm"
    )
