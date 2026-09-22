"""Stability of the virtual-partner backward against the partner count K.

    python3 -m examples.gradient_bench.k_stability --partners 2 4 8 16 32

One seed of the Rankine-Hugoniot case (80 cells x 40 particles, 300 steps, review
section 6): the rms and max of the carried adjoint over the alive particles at every
step of the backward sweep, per K.  The quadrature noise each substep injects into the
carried values is smoothed by the next fit; when the net gain exceeds one the field
grows geometrically towards the initial time (review section 5.12).  Writes
``results/k_stability.json``; ``report.py`` draws ``figures/k_stability.png``.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_inlet_shock as eis
from examples.gradient_bench import RESULTS
from jax_dsmc import ensemble_device
from jax_dsmc.constants import K_BOLTZ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partners", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--particles", type=int, default=3200)
    ap.add_argument("--cells", type=int, default=80)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--attempts", type=int, default=192)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=32)
    ap.add_argument("--output", default=os.path.join(RESULTS, "k_stability.json"))
    args = ap.parse_args(argv)
    geometry = eis.inlet_shock_geometry(args.particles, args.cells, args.dt)
    forward = eis.compiled_forward(
        geometry, args.cells, args.attempts, args.steps, "certified", True
    )
    t = eis.controls("mach", jnp.float64(1.0))
    state0, out, _diag, rec = forward(t, jax.random.key(args.seed))
    length = geometry["length"]
    totals = eis._window_counts(out.x, out.alive, length)
    phi = eis.objective_terminal("density-ratio", out.x, out.v, length, args.particles, totals)
    grid = geometry["grid"]
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    v_scale = float(np.sqrt(K_BOLTZ * eis.GAS.tref / eis.GAS.mass))
    scale = jnp.concatenate((jnp.asarray(dx), jnp.full(3, v_scale)))
    sp = eis._species_of(t)
    alive = np.asarray(rec.alive)
    npc_max = 4 * int(geometry["capacity"]) // args.cells
    occupancy = eis.ensemble_backward.max_occupancy(rec, grid)
    result = {
        "config": {k: v for k, v in vars(args).items() if k not in ("output", "partners")},
        "max_occupancy": occupancy,
        "mean_alive": float(alive.sum(axis=1).mean()),
        "runs": [],
    }
    for k in args.partners:
        t0 = time.time()
        dev = ensemble_device.backward_virtual(
            rec,
            state0.alive,
            out,
            phi,
            grid,
            scale,
            sp,
            jax.random.key(1),
            npc_max,
            args.neighbors,
            "quadratic_full",
            4096,
            args.substeps,
            k,
            "staged",
            None,
        )
        lam = np.asarray(dev.lam)
        rms = [float(np.sqrt(np.mean(lam[s][alive[s]] ** 2))) for s in range(lam.shape[0])]
        mx = [float(np.abs(lam[s][alive[s]]).max()) for s in range(lam.shape[0])]
        row = {"partners": k, "seconds": time.time() - t0, "rms": rms, "max": mx}
        result["runs"].append(row)
        print(
            f"K={k}: {row['seconds']:.0f}s  rms at steps {args.steps - 1}/200/100/0: "
            f"{rms[-1]:.3g} / {rms[200]:.3g} / {rms[100]:.3g} / {rms[0]:.3g}",
            flush=True,
        )
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)


if __name__ == "__main__":
    main()
