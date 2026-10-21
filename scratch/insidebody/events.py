"""Every wall event of the culprit slot in its fatal move (scratch/insidebody/culprit.py's
cached pre-step state), by the stencil tracker with traced events (equal to the lane's)."""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import wall_adjoint  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402

RUN = "examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_tr2_tan_sm8_free_g32_w500"
SEED, STEP, SLOT = 18, 163, 20515
with open(f"{RUN}/history.json") as fh:
    cfg = dict(O.default_config(), **json.load(fh)["config"])
_g, ls = O.initial_levelset(cfg)
ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{RUN}/coef_68.npy")))
grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
    cfg, "radius", EM.nominal(cfg, "radius"), cfg["steps"], ls
)
z = np.load(f"scratch/insidebody/state_{SEED}_{STEP}.npz")
x0, v0, a0 = (jnp.asarray(z[k]) for k in ("x", "v", "alive"))
key = key_for(jax.random.key(cfg["seed_start"] + SEED), cfg["warmup"] + STEP, STAGE_MOVE)
for tracker in ("lane", "stencil"):
    out = wall_adjoint.advance(
        x0, v0, a0, grid, faces, wall, sp.mass[0], params.dt, key, epsilon=params.epsilon,
        max_bounces=static.K_bounce, return_events=True, mesh=mesh, tracker=tracker,
        max_walk=static.K_walk, trace=tracker == "stencil", lane_size=static.lane_size,
    )  # fmt: skip
    x1, v1, alive1, counts = out[0], out[1], out[2], out[3]
    print(tracker, "counts", counts, "slot end", np.asarray(x1[SLOT]), "alive",
          bool(alive1[SLOT]), "phi end", float(L.value(ls, x1[SLOT:SLOT + 1])[0]))  # fmt: skip
    if tracker == "stencil":
        ev = out[-1]
        print("K_bounce", static.K_bounce)
        for b in range(ev.face.shape[0]):
            f = int(ev.face[b, SLOT])
            if f < 0:
                continue
            xh = np.asarray(ev.x_hit[b, SLOT])
            print(f" bounce {b}: face {f} t {float(ev.time[b, SLOT]):.4e} x_hit {xh} "
                  f"phi(x_hit) {float(L.value(ls, jnp.asarray(xh[None]))[0]):.2e} "
                  f"normal {np.asarray(ev.normal[b, SLOT])} v_in {np.asarray(ev.v_in[b, SLOT])[:2]} "
                  f"v_out {np.asarray(ev.v_out[b, SLOT])[:2]}")  # fmt: skip
