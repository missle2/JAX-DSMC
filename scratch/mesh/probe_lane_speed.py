"""Uncaptured forward (700 + 300 steps, seed 32) on the roughest optc shape (S1 gd
coef_8): stencil K=4 vs lane K=4 / 32 -- compiled time, stuck, lane overflow, drag."""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
import forward as F

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load("examples/mesh/results/optimize/circle_coarse_S1_gd/coef_8.npy")))
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
