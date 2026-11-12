"""The MLP design sweep's plan (``lccv``, 2026-11-11): an orthogonal array over the network
and its training, cross-validated, sized to a GPU-hour budget from measured timings.

**Design**: an L9(3^4) orthogonal array -- 9 runs covering every pair of levels of four
three-level factors once, so each factor's main effect is estimated balanced over the others
(the screening pass; the promising region is swept densely after):

| factor | levels |
|---|---|
| hidden layers | 1, 2, 3 |
| width | 8, 32, 128 |
| weight decay | 0, 1e-4, 1e-2 |
| learning rate | 0.003, 0.01, 0.03 |

**Runs per design point**: four leave-one-case-out folds (train on three of S3 S4 K1 K8,
validate forward on the fourth and on the three) and two all-case runs with different seeds
(training-seed variance; their test-case error comes from ``lceval`` later).  Every run's
forward validation (:func:`.train.validate`) is written with it.

**Sizing**: per-iteration seconds per case and per-case validation seconds from a timing run
(``lctime``: :mod:`.train` with ``--validate-train``) give each run's cost as a function of
the iteration count; one iteration count for all runs fills ``--hours`` of GPU time, and
the runs are packed into ``--jobs`` chains by longest-processing-time first.  No JAX import:
this runs on a login node.

    python3 -m examples.learned_collision.sweep --timing train_linear_lctime.json

Writes ``examples/learned_collision/results/sweep_plan_<tag>.json`` (read by
``examples/mesh/cluster/study.py --group lccv``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TRAIN = ("S3", "S4", "K1", "K8")
L9 = ((0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2), (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1),
      (2, 0, 2, 1), (2, 1, 0, 2), (2, 2, 1, 0))  # fmt: skip
"""The standard L9(3^4) array (rows = runs, columns = factors, entries = level indices)."""
FACTORS = {
    "depth": (1, 2, 3),
    "width": (8, 32, 128),
    "weight_decay": (0.0, 1e-4, 1e-2),
    "lr": (0.003, 0.01, 0.03),
}


def design():
    """The nine design points as ``{factor: level}`` dicts."""
    names = list(FACTORS)
    return [{n: FACTORS[n][i] for n, i in zip(names, row, strict=True)} for row in L9]


def runs(points, seeds_all=(0, 1)):
    """``(tag, cases, validate, seed, point)`` of every run."""
    out = []
    for p_i, point in enumerate(points):
        for held in TRAIN:
            cases = [c for c in TRAIN if c != held]
            out.append((f"cv_p{p_i}_loco{held}", cases, [held], 0, point))
        for seed in seeds_all:
            out.append((f"cv_p{p_i}_all_s{seed}", list(TRAIN), [], seed, point))
    return out


def run_cost(cases, validate, iters, timing):
    """Seconds of one run: the bias fill, ``iters`` iterations and the forward validation
    (on ``validate`` and the training cases)."""
    it = sum(timing["iter"][c] for c in cases)
    fill = sum(timing["fill"][c] for c in cases)
    val = sum(timing["validate"][c] for c in list(validate) + list(cases))
    return timing["startup"] + fill + iters * it + val


def timing_of(path):
    """Per-case seconds from a :mod:`.train` timing run (``--validate-train``): an
    iteration's (median over iterations), the bias fill's (scaled per window) and the
    validation's."""
    log = json.loads(path.read_text())
    rows = log["log"]
    per_iter = {c: sorted(r["case_seconds"][c] for r in rows[1:] or rows)[len(rows[1:] or rows) // 2]
                for c in rows[0]["case_seconds"]}  # fmt: skip
    batch = {c: (log["args"]["shear_batch"] if c.startswith("S") else log["args"]["batch"])
             for c in per_iter}  # fmt: skip
    per_window = {c: per_iter[c] / batch[c] for c in per_iter}
    validate = {c: v["seconds"] for c, v in log.get("validation", {}).items()}
    return {"iter": per_iter, "fill": {c: per_window[c] * log["args"]["bias_init"]
                                       for c in per_iter},
            "validate": validate, "startup": 120.0}  # fmt: skip


def pack(costs, jobs):
    """Longest-processing-time packing: ``[[run indices] per job]``, the job loads."""
    order = sorted(range(len(costs)), key=lambda i: -costs[i])
    bins, loads = [[] for _ in range(jobs)], [0.0] * jobs
    for i in order:
        j = loads.index(min(loads))
        bins[j].append(i)
        loads[j] += costs[i]
    return bins, loads


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timing", default="train_linear_lctime.json")
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--min-iters", type=int, default=40)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    timing = timing_of(RESULTS / args.timing)
    plan = runs(design())
    budget = args.hours * 3600.0
    fixed = sum(run_cost(c, v, 0, timing) for _, c, v, _, _ in plan)
    per_iter = sum(run_cost(c, v, 1, timing) - run_cost(c, v, 0, timing) for _, c, v, _, _ in plan)
    iters = math.floor((budget - fixed) / per_iter)
    if iters < args.min_iters:
        raise SystemExit(f"{args.hours} GPU-h allow only {iters} iterations per run (fixed "
                         f"{fixed / 3600:.1f} h): raise --hours or cut the design")  # fmt: skip
    iters = min(iters, args.max_iters)
    costs = [run_cost(c, v, iters, timing) for _, c, v, _, _ in plan]
    bins, loads = pack(costs, args.jobs)
    out = {"iters": iters, "hours": sum(costs) / 3600, "job_hours": [x / 3600 for x in loads],
           "timing": timing, "design": design(),
           "runs": [{"tag": t, "cases": c, "validate": v, "seed": s, **p}
                    for t, c, v, s, p in plan],
           "jobs": bins}  # fmt: skip
    (RESULTS / f"sweep_plan_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{len(plan)} runs x {iters} iterations = {out['hours']:.1f} GPU-h over {args.jobs} "
        f"jobs ({', '.join(f'{h:.1f}' for h in out['job_hours'])} h)"
    )


if __name__ == "__main__":
    main()
