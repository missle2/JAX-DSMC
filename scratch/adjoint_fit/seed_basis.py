"""The two value backwards of one optimiser seed (scratch/perf/profile_seed.py's settings:
optc at fnum x 4, virtual 1 x 2 pull-back) timed with each fit basis -- the end-to-end
cost of the quadratic_full fit against the cheaper bases.  The forward and replay run
once; `_backwards` is re-run per basis (first call compiles, the second is timed).
    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/adjoint_fit/seed_basis.py
"""
import functools
import json
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, ensemble_device  # noqa: E402

ensemble_backward.configure("virtual", 1, 2, "staged")
KN = float(os.environ.get("KN", "0"))  # 0: the collisionless optc seed (fits weighted by 0)
cfg = dict(
    O.default_config(), fnum_scale=4.0, levelset_h=0.0625, levelset_samples=16, grazing=0.05,
    knudsen=KN, a_max=64 if KN else 0,
)
_grid, ls = O.initial_levelset(cfg)
captured = {}
_orig = EM._backwards


def _capture(*args):
    captured["args"] = args
    return _orig(*args)


EM._backwards = _capture
EM.shape_terms(cfg, jax.random.key(32), ls, "radius", 32, 0.2, ("along_pool",))
args = captured["args"]
virtual = ensemble_device.backward_virtual


def _with_basis(*a, basis, **k):
    """`device_backward` passes the basis positionally (argument 10)."""
    a = list(a)
    a[10] = basis
    return virtual(*a, **k)


rows = {}  # run index -> row
order = sys.argv[1:] or ("quadratic_full", "affine", "quadratic_full", "affine")
for i, basis in enumerate(order):
    ensemble_device.backward_virtual = functools.partial(_with_basis, basis=basis)
    times, lam = [], None
    for _ in range(1 if basis in order[:i] else 2):  # compiled already: time once
        t0 = time.perf_counter()
        out = _orig(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    lam = np.asarray(out[0].lam)
    row = {"basis": basis, "seconds": times[-1], "lam_rms": float(np.sqrt(np.mean(lam**2)))}
    if i:
        ref = rows[0]["lam"]
        row["lam_rel_diff_vs_first"] = float(np.sqrt(np.mean((lam - ref) ** 2) / np.mean(ref**2)))
    print(json.dumps(row), flush=True)
    row["lam"] = lam
    rows[i] = row
ensemble_device.backward_virtual = virtual
os.makedirs("scratch/adjoint_fit", exist_ok=True)
json.dump(
    [{k: v for k, v in r.items() if k != "lam"} for r in rows.values()],
    open(f"scratch/adjoint_fit/seed_basis_kn{KN:g}.json", "w"), indent=1,
)
