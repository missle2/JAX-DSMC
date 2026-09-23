"""Uncaptured forward (warm-up + steps, seed 32) of the saved optc Adam iterate-1 shape and
of the re-smoothed one: does `wall_recrossed` reproduce / go away?"""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
for name, path in [("saved adam coef_1", "examples/mesh/results/optimize/circle_coarse_S1_adam/coef_1.npy"),
                   ("adam + smooth 2", "scratch/mesh/adam1_adam__smooth_2.0.npy")]:
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(path)))
    t0 = time.time()
    try:
        _s0, final, diag, *_ = EM.forward(cfg, "radius", None, jax.random.key(32), False, body=ls)
        d = {k: float(v) for k, v in diag._asdict().items() if np.ndim(v) == 0} if hasattr(diag, "_asdict") else diag
        print(name, "OK", f"{time.time()-t0:.0f}s", {k: v for k, v in d.items() if v} if isinstance(d, dict) else d, flush=True)
    except RuntimeError as e:
        print(name, "FAILED", f"{time.time()-t0:.0f}s", e, flush=True)
