"""Real-scale gradient gate: shape_terms (seed 32, fnum x4) on the roughest optc shape
(S1 gd coef_8) with the dense stencil and the lane at the same budget (identical events:
the gradients must agree to roundoff), then the lane at 32 (no deletions)."""
import dataclasses, json, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
import forward as F
from jax_dsmc import ensemble_backward

ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=4.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load("examples/mesh/results/optimize/circle_coarse_S1_gd/coef_8.npy")))
out = {}
for tracker, kb in (("stencil", 4), ("lane", 4), ("lane", 32)):
    F.K_BOUNCE = kb
    cfg["tracker"] = tracker
    t0 = time.time()
    terms, info, _ = EM.shape_terms(cfg, jax.random.key(32), ls, "radius", 32, 0.2, ("along_pool",))
    t = terms["along_pool"]
    g = np.asarray(t.pathwise + t.score)
    out[(tracker, kb)] = (g, info)
    print(f"{tracker} K={kb}: J {info['objective']:.10e} hits {info['hits']} stuck {info['stuck']} "
          f"replay {float(t.replay_error):.1e} |g| {np.linalg.norm(g):.6e}  {time.time()-t0:.0f}s", flush=True)
a, b, c = (out[k][0] for k in (("stencil", 4), ("lane", 4), ("lane", 32)))
np.savez("scratch/mesh/lane_gradient.npz", stencil4=a, lane4=b, lane32=c,
         objective=np.array([out[k][1]["objective"] for k in (("stencil", 4), ("lane", 4), ("lane", 32))]),
         hits=np.array([out[k][1]["hits"] for k in (("stencil", 4), ("lane", 4), ("lane", 32))]))
print(f"lane4 vs stencil4: max|dg|/|g| {np.abs(b - a).max() / np.abs(a).max():.2e}; "
      f"lane32 vs stencil4: {np.abs(c - a).max() / np.abs(a).max():.2e}, "
      f"|g| ratio {np.linalg.norm(c) / np.linalg.norm(a):.6f}", flush=True)
