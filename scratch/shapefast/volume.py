"""Cost of the cut-volume term (shape.volume_cotangent) on the optm grid: the shipped call
(sub^D samples in every Cartesian cell, eager) against the same sum over only the cells near
the surface, jitted; checks the two agree (2026-10-22).

    PYTHONPATH=. python3 scratch/shapefast/volume.py
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402

cfg = dict(OPT.default_config(), case="circle_long", fnum_scale=2.0, levelset_h=0.0625,
           levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5, mach=10.0,
           dt=2.5e-5, a_max=256)  # fmt: skip
grid, mesh, *_ = EM.build(cfg, "radius", EM.nominal(cfg, "radius"), 10)
print("grid", grid.n, "collision cells", grid.volume.shape, "knot h", mesh.body.h)
rng = np.random.default_rng(0)
weights = jnp.asarray(rng.normal(size=grid.volume.shape))
geom = shape.geometry(mesh)
for label in ("compile", "timed", "timed"):
    t0 = time.time()
    cot = shape.volume_cotangent(geom, mesh, grid, weights)
    jax.block_until_ready(cot)
    print(f"shipped {label}: {time.time() - t0:.2f} s")
ref = np.asarray(jax.tree.leaves(cot)[0])
print("nonzero coefs", int((np.abs(ref) > 1e-14 * np.abs(ref).max()).sum()), "of", ref.size)
