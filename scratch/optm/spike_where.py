"""Where do the Kn 0.5 M 10 gradient spikes sit?  Adam's second moment v (beta2 0.999: the
running sum of squared gradients) at its largest knots, against the body (2026-10-16)."""
import dataclasses, json, sys
import jax.numpy as jnp, numpy as np
sys.path.insert(0, "examples/mesh")
import optimize as O
from jax_dsmc.mesh import levelset as L
for tag in sys.argv[1:]:
    d = f"examples/mesh/results/optimize/{tag}"
    h = json.load(open(f"{d}/history.json"))
    grid, ls = O.initial_levelset(h["config"])
    n = len(h["rows"])
    c = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{d}/coef_{n}.npy")))
    x = np.asarray(L.knots(c))
    phi = np.asarray(L.value(c, jnp.asarray(x)))
    v = np.load(f"{d}/adam_state.npz")["v"].ravel()
    order = np.argsort(v)[::-1]
    tot = v.sum()
    print(tag, "iterates", n, "v top-10 share", v[order[:10]].sum() / tot, "top-50", v[order[:50]].sum() / tot)
    inside = phi < 0
    xb = x[inside]
    print("  body x", xb[:, 0].min(), xb[:, 0].max(), "y", xb[:, 1].min(), xb[:, 1].max())
    for i in order[:12]:
        print(f"  knot x {x[i,0]:6.2f} y {x[i,1]:5.2f} phi {phi[i]:+.3f} v/median-band {v[i]/np.median(v[np.abs(phi)<0.5]):.0f}")
