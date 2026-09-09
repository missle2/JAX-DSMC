"""Run one Bird normal-shock case and SAVE THE RAW SNAPSHOT PROFILES.

The expensive part is the simulation; the shock-frame alignment is pure
post-processing.  Saving the raw (n_snapshots, n_x) profiles means every
alignment estimator can be compared on *identical* data, which is the only way
to separate an alignment artefact from a physics error.
"""

import argparse
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from examples.shock.normal import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--mach", type=float, default=2.0)
ap.add_argument("--dim", type=int, default=1)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--settle-mfp", type=float, default=25.0)
ap.add_argument("--sample-mfp", type=float, default=25.0)
ap.add_argument("--domain-mfp", type=float, default=320.0)
ap.add_argument("--cells-per-mfp", type=float, default=4.0)
ap.add_argument("--npc", type=int, default=150)
ap.add_argument("--transverse-cells", type=int, default=4)
ap.add_argument("--a-max", type=int, default=64)
ap.add_argument("--sample-stride", type=int, default=1)
a = ap.parse_args()

t0 = time.time()
r, t, u, info = run(
    mach=a.mach,
    seeds=a.seeds,
    settle_mfp=a.settle_mfp,
    sample_mfp=a.sample_mfp,
    domain_mfp=a.domain_mfp,
    cells_per_mfp=a.cells_per_mfp,
    npc=a.npc,
    dim=a.dim,
    transverse_cells=a.transverse_cells,
    a_max=a.a_max,
    sample_stride=a.sample_stride,
    verbose=True,
)
wall = time.time() - t0
info["wall"] = wall
np.savez_compressed(a.out + ".npz", rho=r, temp=t, ux=u)
with open(a.out + ".pkl", "wb") as f:
    pickle.dump(info, f)
print(json.dumps({k: v for k, v in info.items() if isinstance(v, (int, float, str))}, indent=1))
print(f"snapshots {r.shape}  wall {wall:.1f}s")
