"""Why the long-box optimisers died on ``inflow_shortfall`` (2026-10-12 cluster runs): the
forward (uncaptured: the record does not fit 4 GB) of one seed at a run's iterate, its shortfall split into the inflow's
causes, and the slot occupancy against N.

    PYTHONPATH=. python3 scratch/cellfit/shortfall_check.py RUN_DIR ITER SEED [optimize args]
"""

import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

FIELDS = (
    "inflow_requested",
    "inflow_injected",
    "inflow_escaped",
    "inflow_shortfall",
    "inflow_sampler_failed",
    "inflow_candidate_overflow",
    "inflow_storage_overflow",
    "inflow_tracking_exhausted",
    "attempts_clipped",
)


def main():
    run, it, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = OPT.default_config()
    args = sys.argv[4:]
    for flag, val in zip(args[::2], args[1::2], strict=True):
        k = flag[2:].replace("-", "_")
        cfg[k] = type(cfg[k])(val)
    _grid, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    t0 = time.time()
    _, final, diag, _, _, s = EM.forward(
        cfg, "radius", EM.nominal(cfg, "radius"), jax.random.key(cfg["seed_start"] + seed), False,
        ls
    )
    st = s["static"]
    print(f"{run} it {it} seed {seed}: N {st.N}, steps {st.n_steps}, "
          f"alive at end {int(final.alive.sum())} ({time.time() - t0:.0f} s)")
    for name in FIELDS:
        v = np.asarray(getattr(diag, name))
        nz = np.nonzero(v)[0]
        print(f"  {name:28s} sum {v.sum():10.0f}  steps nonzero {len(nz):4d}"
              + (f"  first {nz[0]} last {nz[-1]}" if len(nz) else ""))


if __name__ == "__main__":
    main()
