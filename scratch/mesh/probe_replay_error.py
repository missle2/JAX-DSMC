"""The shape VJP's replay error on the S4 adam iterate-25/26 shape (seeds 32..35, fnum x4
on the laptop GPU): lane K=32 vs stencil K=4, tolerance lifted so every value prints."""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
import forward as F
from jax_dsmc import ensemble_backward

EM.REPLAY_TOL = 1.0
ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=float(sys.argv[1]), grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
k = int(sys.argv[2])
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"examples/mesh/results/optimize/circle_coarse_S4_adam/coef_{k}.npy")))
for tracker, kb in (("lane", 32), ("stencil", 4)):
    F.K_BOUNCE = kb
    cfg["tracker"] = tracker
    for seed in (32, 33, 34, 35):
        t0 = time.time()
        terms, info, _ = EM.shape_terms(cfg, jax.random.key(seed), ls, "radius", 32, 0.2, ("along_pool",))
        t = terms["along_pool"]
        print(f"{tracker} K={kb} seed {seed}: gradient replay error {float(t.replay_error):.2e}  stuck {info['stuck']}  hits {info['hits']}  {time.time()-t0:.0f}s", flush=True)
