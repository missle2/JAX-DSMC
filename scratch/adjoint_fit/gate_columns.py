"""Gate for the two-column value backward: one optimiser seed (optc at fnum x 4, collisional
at Kn = 0.5 so the fits matter; virtual 1 x 2 pull-back), its two value backwards
(`EM._backwards`) saved with the current code (`save`) and compared with the new
(`compare`): lam / lambda0 / kernel of both fields, and the time (first call compiles,
the second is timed).
    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/adjoint_fit/gate_columns.py save|compare
"""
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

mode = sys.argv[1]
if os.environ.get("LU"):  # isolate the solver: the column sweep with the old batched LU
    import jax.numpy as jnp  # noqa: E402

    from jax_dsmc import ensemble_device  # noqa: E402

    ensemble_device._spd_solve = lambda m, rhs: jnp.linalg.solve(m, rhs)
KN = float(os.environ.get("KN", "0.5"))
PULL = tuple(int(a) for a in os.environ.get("PULL", "1x2").split("x"))
ensemble_backward.configure("virtual", *PULL, "staged")
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
times = []
for _ in range(2):
    t0 = time.perf_counter()
    out = _orig(*args)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)
arrays = {
    f"{name}_{field}": np.asarray(getattr(dev, field))
    for name, dev in zip(("hits", "timing"), out, strict=True)
    for field in ("lambda0", "lam", "kernel")
}
path = f"scratch/adjoint_fit/gate_kn{KN:g}_{PULL[0]}x{PULL[1]}.npz"
print(f"backwards: first {times[0]:.1f} s (incl. compile), second {times[1]:.1f} s")
if mode == "save":
    np.savez(path, seconds=times[1], **arrays)
    print("saved", path)
else:
    ref = np.load(path)
    print(f"reference: {float(ref['seconds']):.1f} s")
    worst = 0.0
    for key, a in arrays.items():
        b = ref[key]
        scale = np.sqrt(np.mean(b**2)) or 1.0
        rel = float(np.max(np.abs(a - b)) / scale)
        worst = max(worst, rel)
        print(f"  {key:16s} max |diff| / rms {rel:.2e}")
    print("GATE", "PASS" if worst < 1e-8 else "FAIL", f"(worst {worst:.2e}, tolerance 1e-8)")
    # growth of the difference back from the last captured step: rounding amplified by the
    # backward recursion grows smoothly from ~1e-15
    for name in ("hits_lam", "timing_lam"):
        a, b = arrays[name], ref[name]
        m = a.shape[0]
        steps = [m - 1, m - 2, m - 5, m - 20, m - 100, m // 2, 0]
        rel = [np.max(np.abs(a[k] - b[k])) / (np.sqrt(np.mean(b[k] ** 2)) or 1.0) for k in steps]
        print(f"  {name} by step back:", "  ".join(f"k={k}: {r:.1e}" for k, r in zip(steps, rel)))
