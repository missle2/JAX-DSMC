"""Find the first NaN in a training-window gradient (jax_debug_nans)."""
import argparse

import jax
import numpy as np

from examples.learned_collision import train_scalar as ts
from examples.learned_collision import windows

jax.config.update("jax_debug_nans", True)
args = argparse.Namespace(dx=4.0, ppc=10, blocks=5, burn=2)
ref, meta = ts.load("reference_thermal_L64_v1.npz")
collide_fn, theta, _ = ts.operator("esfp")
case = windows.coarse_case(meta["wave"], meta["length_mfp"], 4.0, 10, ref["snap_x"].shape[2])
fns = ts.compiled(case, collide_fn, args, ts.FIELD[meta["wave"]])
rng = np.random.default_rng(0)
for i in range(40):
    loss, g = ts.window(ref, meta, case, rng, fns, theta, args)
    print(i, float(loss), {k: float(v) for k, v in g.items()}, flush=True)
