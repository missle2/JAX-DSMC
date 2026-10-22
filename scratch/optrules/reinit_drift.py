"""Does reinitialisation alone roughen a shape?  Smooth early iterates of the rough runs,
``L.reinitialise`` applied N times with no step, the contour's total turning
(scratch/optrules/roughness.py) after each block.

    PYTHONPATH=. python3 scratch/optrules/reinit_drift.py
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optimize as O  # noqa: E402
from roughness import turning  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

R = "examples/mesh/results/optimize/"
CASES = (("M10 S4 Adam @ 41", "circle_long_kn0.5_M10_S4_adam_tan_sm8_free_g32", 41),
         ("M10 S1 TR @ 82", "circle_long_kn0.5_M10_S1_tr_tan_sm8_free_g32_w500", 82),
         ("deck S4 TR2 @ 153", "circle_long_kn0.5_S4_tr2_tan_sm8_free_g32_w500", 153))  # fmt: skip
for name, tag, k in CASES:
    cfg = dict(O.default_config(), **json.load(open(f"{R}{tag}/history.json"))["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{R}{tag}/coef_{k}.npy")))
    out = []
    for n in range(0, 61):
        if n % 20 == 0:
            segs = O.surface_polyline(ls, lo, hi, n=1500)
            tot, conc = turning(max(segs, key=len))
            area = float(L.body_volume(ls, lo, hi, 4))
            out.append(f"{n}: {tot:.2f} ({conc:.2f}) A {area:.3f}")
        ls = L.reinitialise(ls)
    print(f"{name:18s}", " | ".join(out), flush=True)
