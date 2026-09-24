"""The optc shape that raised SplitCellError (S1 gd coef_16): mesh build with sub-cells,
then one seed of the full shape_terms chain (fnum x4 so it fits the laptop GPU)."""
import dataclasses, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward
from jax_dsmc.mesh import cut

ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=float(sys.argv[1]) if len(sys.argv) > 1 else 4.0, grid_scale=1,
           levelset_h=0.0625, levelset_samples=16, grazing=0.05)
grid, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load("examples/mesh/results/optimize/circle_coarse_S1_gd/coef_16.npy")))
m = cut.build_mesh(ls, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=1)
split = np.nonzero(np.asarray(m.split_of) >= 0)[0] if m.n_sub else []
vol = np.asarray(m.volume)
print(f"n_sub {m.n_sub}, split cells {list(split)}, sub-cell parents {np.asarray(m.sub_parent).tolist()}")
for j, c in enumerate(np.asarray(m.sub_parent)):
    if vol[400 + j] > 0:
        print(f"   cell {c}: parent-region volume {vol[c]:.4f}, sub-cell {400 + j} volume {vol[400 + j]:.4f}")
t0 = time.time()
terms, info, s = EM.shape_terms(cfg, jax.random.key(32), ls, "radius", cfg["neighbors"], cfg["timing_band"], (cfg["variant"],))
t = terms[cfg["variant"]]
g = np.asarray(t.pathwise + t.score)
print(f"shape_terms ok: J {info['objective']:.4e} hits {info['hits']} stuck {info['stuck']} replay_error {float(t.replay_error):.1e} "
      f"|g| {np.linalg.norm(g):.3e} finite {np.isfinite(g).all()}  {time.time()-t0:.0f}s")
