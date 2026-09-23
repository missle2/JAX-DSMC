"""Stage-1 checks on the failing optc shape (S1 gd coef_8): (a) full uncaptured forward at
K_bounce 4 and 8 -- stuck counts and time; (b) shape_terms with K_bounce 1 (many deletions)
at fnum x16, short window: the replay must reproduce the forward (REPLAY_TOL 1e-12)."""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
import forward as F
from jax_dsmc import ensemble_backward

ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load("examples/mesh/results/optimize/circle_coarse_S1_gd/coef_8.npy")))
mode = sys.argv[1]
if mode == "forward":
    for kb in (4, 8):
        F.K_BOUNCE = kb
        for rep in range(2):   # second pass = compiled time
            t0 = time.time()
            s0, final, diag, *_ = EM.forward(cfg, "radius", None, jax.random.key(32), False, body=ls)
            jax.block_until_ready(final)
            print(f"K_bounce {kb} run {rep}: {time.time()-t0:.0f}s  stuck(last 300) {int(diag.surface_stuck.sum())}  "
                  f"inside {int(diag.inside_body.sum())}  recrossed {int(diag.wall_recrossed.sum())}", flush=True)
else:
    F.K_BOUNCE = 1
    F.DT = float(sys.argv[2]) if len(sys.argv) > 2 else F.DT  # larger steps: more multi-event moves
    cfg.update(fnum_scale=16.0, warmup=100, steps=60, window=40)
    t0 = time.time()
    terms, info, _ = EM.shape_terms(cfg, jax.random.key(32), ls, "radius", 32, 0.2, ("along_pool",))
    t = terms["along_pool"]
    print(f"K_bounce 1: stuck {info['stuck']} hits {info['hits']}  replay_error {float(t.replay_error):.2e}  "
          f"|g| {float(jnp.linalg.norm(t.pathwise + t.score)):.3g}  {time.time()-t0:.0f}s", flush=True)
