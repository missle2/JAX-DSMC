"""Run a Couette study and write its results to a log.

    python3 -m examples.couette --study gate       # the viscosity gate
    python3 -m examples.couette --study knudsen    # the Kn -> 0 extrapolation
    python3 -m examples.couette --study converge   # cell-size convergence
    python3 -m examples.couette --study walls      # every wall treatment
    python3 -m examples.couette --study dims       # D = 1, 2, 3
    python3 -m examples.couette --study all

Each study appends a machine-readable table to ``examples/couette/results/<study>.log``
and prints it.  ``examples.couette.report`` then **parses those logs** to build the
figures and ``RESULTS.md`` -- the logs stay the single source of truth, so a figure can
never disagree with the run that produced it, and a truncated run raises rather than
quietly plotting a short curve.

Two properties of this driver are load-bearing, and both exist because their absence
produced published-looking numbers that were wrong:

**Every case is an ensemble.**  ``mu_meas/mu_CE`` carries several per cent of Monte
Carlo scatter at any affordable particle count, so a single draw of it is not a
measurement.  Four successive explanations of an apparent viscosity deficit were built
on single draws before an ensemble showed the quantity simply scatters.  There is no
one-seed mode: ``--seeds`` has a minimum of 2, and every ratio is reported as mean and
standard error with the seed count beside it.  Note that an SE from 2 seeds is itself a
very noisy estimate -- prefer 4 or more for anything quoted.

**Warm-up is measured in diffusion times, never in steps.**  Momentum crosses the gap on
``L^2/nu``, so a study that widens the gap changes the time to steady state as ``L^2``.
A constant step budget therefore runs the widest case for a small fraction of a
diffusion time and produces a clean-looking but meaningless trend -- exactly how an
earlier Knudsen sweep manufactured one.  ``--warm-diff`` sets warm-up as a multiple of
``diffusion_steps(case)``, per case; ``--warm`` overrides it with a raw count and should
only be used for smoke tests.
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import numpy as np

from examples.couette import Case, diffusion_steps, profile, slip_prediction

RESULTS = "examples/couette/results"

#: Warm-up as a multiple of the gap's momentum-diffusion time.  The shear rate itself
#: settles by ~0.3 of a diffusion time, so 3 is a deliberate safety factor rather than a
#: threshold -- moving the gate from 1.0 to 3.0 shifted the ratio by 1.7 %, which is the
#: measurement that says 1.0 was not enough.
WARM_DIFFUSION_TIMES = 3.0

COLUMNS = (
    "case",
    "dim",
    "n_cells",
    "npc",
    "acc",
    "eps_rel",
    "specular",
    "dx_over_mfp",
    "knudsen",
    "du_dy",
    "du_dy_se",
    "tau_bulk",
    "mu_meas",
    "mu_ce",
    "ratio",
    "ratio_se",
    "seeds",
    "warm",
    "slip",
    "slip_theory",
    "temp_bulk",
    "n_part",
    "recrossed",
    "clipped",
)


def _mean_se(values):
    """Mean and standard error of the mean.  ``ddof=1`` -- the sample spread, not the
    population one; with n=2 this is the honest (large) number rather than the flattering
    one."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def run_case(case: Case, sample, seeds, warm=None, warm_diff=WARM_DIFFUSION_TIMES):
    """Run ``case`` over ``seeds`` independent streams.  Returns the seed-0 result (for
    the saved profile) plus the ensemble statistics."""
    if seeds < 2:
        raise ValueError(f"seeds={seeds}: a single draw of a noisy ratio is not a measurement")
    n_warm = int(warm) if warm else round(warm_diff * diffusion_steps(case))
    results = [
        profile(case, key=jax.random.PRNGKey(1000 + s), n_warm=n_warm, n_sample=sample)
        for s in range(seeds)
    ]
    ratio, ratio_se = _mean_se([r["ratio"] for r in results])
    du_dy, du_dy_se = _mean_se([r["du_dy"] for r in results])
    stats = {
        "ratio": ratio,
        "ratio_se": ratio_se,
        "du_dy": du_dy,
        "du_dy_se": du_dy_se,
        "seeds": seeds,
        "warm": n_warm,
        # the remaining scalars are ensemble means too; they are diagnostics, not gates
        "tau_bulk": float(np.mean([r["tau_bulk"] for r in results])),
        "mu_meas": float(np.mean([r["mu_meas"] for r in results])),
        "slip": float(np.mean([r["slip"] for r in results])),
        "temp_bulk": float(np.mean([np.mean(r["temp"]) for r in results])),
        # failure counters are summed, never averaged: one clipped attempt anywhere
        # invalidates the ensemble, so averaging it away would hide exactly the thing
        # the counter exists to catch
        "recrossed": int(sum(r["recrossed"] for r in results)),
        "clipped": int(sum(r["clipped"] for r in results)),
    }
    return results[0], stats


def _row(name, case: Case, r, st):
    return (
        f"{name:<22} {case.dim:>3d} {case.n_cells:>7d} {case.npc:>5d} {case.acc:>5.2f} "
        f"{case.epsilon_rel:>7.2f} {int(case.specular):>8d} {r['dx_over_mfp']:>11.4f} "
        f"{r['knudsen']:>8.5f} {st['du_dy']:>12.5e} {st['du_dy_se']:>11.4e} "
        f"{st['tau_bulk']:>12.5e} {st['mu_meas']:>12.5e} {r['mu_ce']:>12.5e} "
        f"{st['ratio']:>8.4f} {st['ratio_se']:>9.4f} {st['seeds']:>6d} {st['warm']:>8d} "
        f"{st['slip']:>8.2f} {slip_prediction(r):>11.2f} {st['temp_bulk']:>10.3f} "
        f"{r['n_part']:>8d} {st['recrossed']:>10d} {st['clipped']:>8d}"
    )


def run_study(name, sample, seeds, warm=None, warm_diff=WARM_DIFFUSION_TIMES, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.log")
    header = "# " + " ".join(COLUMNS)
    lines = [
        header,
        f"# sample={sample} seeds={seeds} warm_diff={warm_diff} "
        f"generated={time.strftime('%Y-%m-%d')}",
    ]
    print(header)
    for label, case in STUDIES[name]:
        t0 = time.time()
        r, st = run_case(case, sample, seeds, warm=warm, warm_diff=warm_diff)
        np.savez(
            os.path.join(out_dir, f"{name}__{label.replace(' ', '_').replace('/', '')}.npz"),
            y=r["y"],
            u_s=r["u_s"],
            tau=r["tau"],
            temp=r["temp"],
            nrho=r["nrho"],
            mfp=r["mfp"],
            length=r["length"],
            u_wall=r["u_wall"],
        )
        line = _row(label, case, r, st)
        lines.append(line)
        print(f"{line}   ({time.time() - t0:.0f}s)", flush=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")
    return path


STUDIES = {
    # dx/mfp = 0.3, which is NOT resolution-converged: the cell-size study measures a
    # 1.35 % low bias here against its own dx -> 0 extrapolation.  It is kept anyway
    # because the `knudsen` sweep runs at this same dx/mfp, so the bias is a constant
    # offset across that sweep -- it moves the intercept and not the slope, and
    # `report.combined_gate` then removes it explicitly.  Running the gate at 0.15
    # instead would cost 4x and still need the same correction.
    "gate": [("dx/mfp=0.30", Case(n_cells=100, a_max=128))],
    "converge": [
        ("50 cells", Case(n_cells=50, a_max=256)),
        ("100 cells", Case(n_cells=100, a_max=128)),
        ("200 cells", Case(n_cells=200)),
    ],
    "walls": [
        ("diffuse acc=1.0", Case(n_cells=100, a_max=128)),
        ("diffuse acc=0.5", Case(n_cells=100, a_max=128, acc=0.5)),
        ("diffuse acc=0.1", Case(n_cells=100, a_max=128, acc=0.1)),
        ("specular", Case(n_cells=100, a_max=128, specular=True)),
        ("regularised eps=0.1", Case(n_cells=100, a_max=128, epsilon_rel=0.1)),
        ("regularised eps=0.3", Case(n_cells=100, a_max=128, epsilon_rel=0.3)),
    ],
    "knudsen": [
        # Chapman-Enskog is a CONTINUUM limit, so the gate has to be taken as Kn -> 0.
        # dx/lambda is held at 0.3 throughout, so cell size is NOT the variable here --
        # only the gap in mean free paths is.  The range stops at Kn = 1/30 because
        # warm-up costs L^2: one more halving of Kn is a 4x longer run for a point the
        # fit does not need.
        ("Kn=1/10", Case(n_cells=33, gap_mfp=10.0, a_max=128)),
        ("Kn=1/15", Case(n_cells=50, gap_mfp=15.0, a_max=128)),
        ("Kn=1/22", Case(n_cells=73, gap_mfp=22.0, a_max=128)),
        ("Kn=1/30", Case(n_cells=100, gap_mfp=30.0, a_max=128)),
    ],
    "dims": [
        ("D=1", Case(dim=1, n_cells=100, npc=200, a_max=128)),
        ("D=2", Case(dim=2, n_cells=100, npc=60, n_span=4, a_max=128)),
        ("D=3", Case(dim=3, n_cells=100, npc=20, n_span=4, a_max=128)),
    ],
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m examples.couette", description=__doc__)
    ap.add_argument("--study", default="gate", choices=[*STUDIES, "all"])
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=4, help="independent streams per case (min 2)")
    ap.add_argument(
        "--warm-diff",
        type=float,
        default=WARM_DIFFUSION_TIMES,
        help="warm-up in momentum-diffusion times L^2/nu (the only correct unit)",
    )
    ap.add_argument(
        "--warm", type=int, default=0, help="raw warm-up step override -- smoke tests only"
    )
    args = ap.parse_args(argv)
    names = list(STUDIES) if args.study == "all" else [args.study]
    for name in names:
        print(f"\n=== {name} ===")
        run_study(name, args.sample, args.seeds, warm=args.warm, warm_diff=args.warm_diff)


if __name__ == "__main__":
    main()
