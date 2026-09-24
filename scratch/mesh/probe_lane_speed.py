"""Uncaptured forward (700 + 300 steps, seed 32) on the roughest optc shape (S1 gd
coef_8): stencil K=4 vs lane K=4/16/32 -- compiled time, stuck, lane overflow, drag; then
the lane occupancy (slots with a pass-0 event) and the deepest pass used, on 20 moves of
the final state."""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
import forward as F
from jax_dsmc import wall_adjoint
from jax_dsmc.rng import STAGE_MOVE, key_for

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load("examples/mesh/results/optimize/circle_coarse_S1_gd/coef_8.npy")))
final = None
for tracker, kb in [(a, int(b)) for a, b in (x.split(":") for x in sys.argv[1:])]:
    F.K_BOUNCE = kb
    cfg["tracker"] = tracker
    for rep in range(2):
        t0 = time.time()
        s0, final, diag, rec, tallies, s = EM.forward(cfg, "radius", None, jax.random.key(32), False, body=ls)
        jax.block_until_ready(final)
        drag = float(jnp.sum(tallies.momentum[100:, 4:, 0]))
        print(f"{tracker} K={kb} run {rep}: {time.time()-t0:.0f}s  stuck {int(diag.surface_stuck.sum())}  "
              f"lane_overflow {int(diag.lane_overflow.sum())}  alive {int(final.alive.sum())}  drag-sum {drag:.6e}", flush=True)
# lane occupancy on the steady state
g, mesh, st = s["grid"], s["mesh"], s["static"]
x, v, alive = final.x, final.v, final.alive
occ, deep = [], 0
for k in range(20):
    out = wall_adjoint.advance(x, v, alive, g, s["faces"], s["wall"], s["sp"].mass[0], s["params"].dt,
                               key_for(jax.random.key(7), k, STAGE_MOVE), mesh=mesh, max_bounces=32,
                               return_events=True, tracker="lane", lane_size=st.N)
    ev = out[5]
    occ.append(int(jnp.sum(ev.slot >= 0)))
    rows = np.nonzero(np.asarray(jnp.any(ev.tail.face >= 0, axis=1)))[0]
    deep = max(deep, int(rows.max()) + 2 if rows.size else 1)
    x, v, alive = out[0], out[1], out[2]
print(f"slots {st.N}, alive {int(alive.sum())}: lane occupancy per step mean {np.mean(occ):.0f} max {max(occ)} "
      f"({max(occ)/st.N:.2%} of slots); deepest recorded pass {deep}", flush=True)
