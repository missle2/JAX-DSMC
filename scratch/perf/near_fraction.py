"""Gate M2 of scratch/perf/forward.md: the fraction of rays the near-body cull of
``levelset.hit_body`` keeps (``_near``: phi_start <= G L), for the pass-0 search (L = |v| dt)
and the reach-length query (L = reach), and how tight the Lipschitz bound G is (against the
sampled max |grad phi| over the box), on the optc level sets: the initial circle and later
iterates.  Population: uniform over the flow domain, thermal velocities + the stream (the
free-molecular field is close to it; the windward excess only changes the fraction by O(1)).

    JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/perf/near_fraction.py
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import forward as F  # noqa: E402
import optimize as O  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

RUNS = os.path.join(ROOT, "examples", "mesh", "results", "optimize")


def main():
    cfg = dict(O.default_config(), levelset_h=0.0625, levelset_samples=16)
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    dt = float(F.DT) if hasattr(F, "DT") else 1e-4
    key = jax.random.PRNGKey(0)
    n = 400_000
    x = lo + jax.random.uniform(key, (n, 2)) * (hi - lo)
    vth = np.sqrt(1.380649e-23 * F.TEMP / 6.63e-26)
    v = vth * jax.random.normal(jax.random.fold_in(key, 1), (n, 3))
    v = v.at[:, 0].add(F.VSTREAM)
    speed = jnp.linalg.norm(v[:, :2], axis=1)
    cases = {"initial": ls0}
    for run, k in (
        ("circle_coarse_S1_adam", 40),
        ("circle_coarse_S8_gd", -1),
        ("circle_coarse_S4_adam", 60),
    ):
        files = sorted(
            (int(f[5:-4]) for f in os.listdir(os.path.join(RUNS, run)) if f.startswith("coef_")),
        )
        it = files[k] if k < 0 else k
        coef = np.load(os.path.join(RUNS, run, f"coef_{it}.npy"))
        cases[f"{run}@{it}"] = ls0.__class__(
            jnp.asarray(coef), ls0.lo, ls0.h, ls0.n_sample, ls0.n_bisect, ls0.grazing
        )
    rows = {}
    for name, ls in cases.items():
        phi, g = L.value_and_grad(ls, x)
        flow = np.asarray(phi > 0)
        bound = float(L.gradient_bound(ls))
        gmax = float(jnp.max(jnp.linalg.norm(g, axis=1)))
        gnear = float(jnp.max(jnp.where(jnp.abs(phi) < 0.5, jnp.linalg.norm(g, axis=1), 0.0)))
        pass0 = np.asarray(L._near(phi, speed * dt, ls, x))[flow].mean()
        reach = np.asarray(L._near(phi, jnp.full(n, ls.reach), ls, x))[flow].mean()
        local = L.local_gradient_bounds(ls)
        rows[name] = {
            "G_local_median_max": [float(jnp.median(local)), float(jnp.max(local))],
            "G_bound": bound,
            "max_grad_sampled": gmax,
            "max_grad_near_surface": gnear,
            "near_fraction_pass0": float(pass0),
            "near_fraction_reach": float(reach),
            "reach": ls.reach,
            "mean_step_length": float(jnp.mean(speed) * dt),
        }
        print(name, json.dumps(rows[name]), flush=True)
    with open(os.path.join(ROOT, "scratch", "perf", "near_fraction.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
