"""Production-density NTC attempt budget of the optckn runs: the optc level sets at Kn 0.5,
fnum x1 on the coarse grid, forward only (no capture), at a_max 64 and 32 -- a clipped
attempt stops a run (ensemble_mesh._check), so the budget needs margin over the shapes the
optimiser may reach.

    PYTHONPATH=. python3 scratch/cellfit/attempts_check.py
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

RUNS = {
    "c0": ("circle_coarse_S1_adam", 0),
    "a100": ("circle_coarse_S1_adam", 100),
    "a200": ("circle_coarse_S1_adam", 200),
}


def main():
    for a_max in (32, 64):
        cfg = OPT.default_config()
        cfg.update(fnum_scale=1, grid_scale=1, levelset_h=0.0625, levelset_samples=16,
                   grazing=0.05, knudsen=0.5, a_max=a_max)
        _grid, ls0 = OPT.initial_levelset(cfg)
        for name, (run, it) in RUNS.items():
            coef = np.load(f"examples/mesh/results/optimize/{run}/coef_{it}.npy")
            ls = dataclasses.replace(ls0, coef=jnp.asarray(coef))
            t0 = time.time()
            _, final, diag, _, _, s = EM.forward(
                cfg, "radius", EM.nominal(cfg, "radius"), jax.random.key(32), False, ls
            )
            clipped = float(jnp.sum(diag.attempts_clipped))
            short = float(jnp.sum(diag.inflow_shortfall))
            print(f"a_max {a_max:3d} {name:5s}: attempts_clipped {clipped:.0f}  inflow_shortfall "
                  f"{short:.0f}  alive {int(final.alive.sum())} of {final.alive.shape[0]} slots"
                  f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
