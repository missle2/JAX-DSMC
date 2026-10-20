"""How fast does the drag re-equilibrate after a shape change?  (The optimiser's warm start.)

Per seed: run iterate A of an optm run cold (``warmup`` + ``steps``), switch to iterate B
carrying the gas over (particles inside B removed) and record the per-step drag for ``n``
steps; the reference is B run cold for the same ``n`` steps (the same absolute step indices,
so the same random numbers).  Writes per-seed per-step drag series to OUT.npz.

    PYTHONPATH=. python3 scratch/warmstart/relax.py OUT.npz RUN_DIR A B SEEDS N [overrides]
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402


def drag_series(tallies, dim, fnum, dt):
    """Per-step drag (the windowed objective's integrand, per step)."""
    return np.asarray(jnp.sum(tallies.momentum[:, 2 * dim :, 0], axis=1)) * fnum / dt


def body(cfg, run, it):
    _grid, ls = OPT.initial_levelset(cfg)
    return dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))


def main():
    out, run, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seeds, n = int(sys.argv[5]), int(sys.argv[6])
    over = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
    with open(f"{run}/history.json") as fh:
        cfg = dict(OPT.default_config(), **json.load(fh)["config"])
    cfg.update(over)
    ls_a, ls_b = body(cfg, run, a), body(cfg, run, b)
    warm_rows, cold_rows = [], []
    for i in range(seeds):
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        _s0, _f, diag, _r, _t, s_a = EM.forward(cfg, "radius", 0.0, key, False, ls_a)
        EM._check(diag)
        long = dict(cfg, steps=n)
        _s0, _f, diag, _r, tw, s_w = EM.forward(long, "radius", 0.0, key, False, ls_b,
                                                restart=s_a["final"])  # fmt: skip
        EM._check(diag)
        _s0, _f, diag, _r, tc, s_c = EM.forward(long, "radius", 0.0, key, False, ls_b)
        EM._check(diag)
        fnum, dt = float(s_c["params"].fnum[0]), float(s_c["params"].dt)
        warm_rows.append(drag_series(tw, s_c["static"].D, fnum, dt))
        cold_rows.append(drag_series(tc, s_c["static"].D, fnum, dt))
        w, c = warm_rows[-1], cold_rows[-1]
        print(f"seed {i}: {time.time() - t0:.0f} s; warm/cold mean over blocks of 100: "
              + " ".join(f"{w[k:k + 100].mean() / c.mean():.3f}" for k in range(0, n, 100)),
              flush=True)  # fmt: skip
        np.savez(out, warm=np.stack(warm_rows), cold=np.stack(cold_rows), a=a, b=b,
                 config=json.dumps(cfg))  # fmt: skip


if __name__ == "__main__":
    main()
