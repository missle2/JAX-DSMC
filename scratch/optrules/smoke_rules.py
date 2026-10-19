"""Smoke test of optimize.rule_step + after_step for tr / lbfgs / mma on circle_long's level
set with a synthetic objective (no DSMC): J = frontal height proxy = int y^2 over the body,
its coefficient gradient by JAX.  Checks the state round-trip, area hold and J decrease."""

import dataclasses
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402
import optimizers as OZ  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402


def objective(coef, ls, lo, hi):
    ls = dataclasses.replace(ls, coef=coef)
    box = np.asarray(hi) - np.asarray(lo)
    n = [int(np.ceil(box[a] / ls.h)) * 2 for a in range(2)]
    xs = [lo[a] + box[a] * (jnp.arange(n[a]) + 0.5) / n[a] for a in range(2)]
    gx, gy = jnp.meshgrid(*xs, indexing="ij")
    x = jnp.stack([gx.ravel(), gy.ravel()], 1)
    inside = 0.5 * (1 - jnp.tanh(L.value(ls, x) / (ls.h / 2)))
    return jnp.sum(inside * (x[:, 1] - 5.0) ** 2) * box.prod() / (n[0] * n[1])


cfg = dict(
    O.default_config(),
    case="circle_long",
    fnum_scale=2.0,
    levelset_h=0.0625,
    smooth=8.0,
    band=1.0,
    step=0.05,
    area_mode="tangent",
    keepout=1.0,
)
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
area0 = float(L.body_volume(ls0, lo, hi, 4))
f = jax.jit(jax.value_and_grad(objective), static_argnums=(2, 3))
for rule in ("tr", "lbfgs", "mma"):
    cfg["optimizer"] = rule
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "opt_state.npz"
        ls, state, js = ls0, {"area0": area0}, []
        for k in range(6):
            j, g = f(ls.coef, ls, tuple(lo), tuple(hi))
            j, g = float(j), np.asarray(g)
            js.append(j)
            _a, g_area = O.area_and_gradient(ls, lo, hi)
            g_full = g
            g, _nu = O.tangent_component(g, g_area)
            base, delta, pred, state, extra = O.rule_step(cfg, ls, g, g_full, g_area, j, 1e-6,
                                                          state, k, lo, hi)  # fmt: skip
            OZ.save_state(path, state)
            state = OZ.load_state(path)
            stepped = dataclasses.replace(base, coef=base.coef + jnp.asarray(delta))
            ls, shift, clipped = O.after_step(stepped, "tangent", area0, lo, hi, 1.0)
            area = float(L.body_volume(ls, lo, hi, 4))
            keep = {k2: v for k2, v in extra.items() if k2 in ("accepted", "rho", "radius",
                                                                "lbfgs_pairs", "step_max")}  # fmt: skip
            print(
                f"{rule} {k}: J {j:.4f} pred {pred:+.4f} area err {area / area0 - 1:+.1e} {keep}",
                flush=True,
            )
        assert js[-1] < js[0], (rule, js)
print("ok")
