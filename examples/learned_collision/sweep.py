"""The MLP design sweep's plan (``lccv``, 2026-11-11): an orthogonal array over the network
and its training, cross-validated, sized to a GPU-hour budget from measured timings.

**Design**: an L9(3^4) orthogonal array -- 9 runs covering every pair of levels of four
three-level factors once, so each factor's main effect is estimated balanced over the others
(the screening pass; the promising region is swept densely after):

| factor | levels |
|---|---|
| hidden layers | 1, 2, 3 |
| width | 8, 32, 128 |
| weight decay (per-iteration shrink) | 0, 1e-3, 1e-2 |
| learning rate (per-iteration output change) | 0.001, 0.003, 0.01 |

The learning rate is :mod:`.train`'s fan-in-normalised step (2026-11-12): the correction moves
~``0.5 lr`` per iteration against the ~0.1 it needs (``mu / mu_ref`` 1.06-1.12), so the levels
reach it in ~200 / ~60 / ~20 iterations -- the lowest is the insurance point, the highest the
fastest that keeps the step-to-step jitter ~1 %.  (The first levels, 0.003-0.03 of plain Adam,
moved ``dlog_tau`` ~0.03-0.3 per step: lctime's 6 steps at 0.02 overshot to ``mu`` 0.5x.)
Weight decay is decoupled and independent of the learning rate: 1e-2 shrinks untrained
weights ~20x over 300 iterations (the network pulled towards its linear small-signal limit);
the first levels (AdamW's ``lr x wd``) shrank them < 3 %, no effect to measure.

**Runs per design point**: five leave-one-case-out folds (train on four of S3 S4 K1 K4 K8,
validate forward on the fifth and on the four) and two all-case runs with different seeds
(training-seed variance; their test-case error comes from ``lceval`` later).  Every run's
forward validation (:func:`.train.validate`) is written with it.

**Sizing**: per-iteration seconds per case and per-case validation seconds from a timing run
(``lctime``: :mod:`.train` with ``--validate-train``) give each run's cost as a function of
the iteration count; one iteration count for all runs fills ``--hours`` of GPU time, and
the runs are packed into ``--jobs`` chains by longest-processing-time first.  No JAX import:
this runs on a login node.

    python3 -m examples.learned_collision.sweep [--hours 60] [--jobs 8]

Writes ``examples/learned_collision/results/sweep_plan_<tag>.json`` (read by
``examples/mesh/cluster/study.py --group lccv``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TRAIN = ("S3", "S4", "K1", "K4", "K8")
L9 = ((0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2), (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1),
      (2, 0, 2, 1), (2, 1, 0, 2), (2, 2, 1, 0))  # fmt: skip
"""The standard L9(3^4) array (rows = runs, columns = factors, entries = level indices)."""
FACTORS = {
    "depth": (1, 2, 3),
    "width": (8, 32, 128),
    "weight_decay": (0.0, 1e-3, 1e-2),
    "lr": (0.001, 0.003, 0.01),
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


def run_cost(cases, validate, iters, timing, point=None):
    """Seconds of one run: the bias fill, ``iters`` iterations and the forward validation
    (on ``validate`` and the training cases).  An MLP ``point``'s windows cost more than the
    linear model's, interpolated in ``depth x width`` up to the largest network's measured
    ratio (``timing["mlp_ratio"]``)."""
    factor = 1.0
    if point is not None:
        top = max(FACTORS["depth"]) * max(FACTORS["width"])
        factor += (timing["mlp_ratio"] - 1.0) * point["depth"] * point["width"] / top
    it = factor * sum(timing["iter"][c] for c in cases)
    fill = factor * sum(timing["fill"][c] for c in cases)
    val = sum(timing["validate"][c] for c in list(validate) + list(cases))
    return timing["startup"] + fill + iters * it + val


def _per_iter(rows):
    """Per-case median iteration seconds (the first, compiling, iteration dropped)."""
    rows = rows[1:] or rows
    return {c: sorted(r["case_seconds"][c] for r in rows)[len(rows) // 2]
            for c in rows[0]["case_seconds"]}  # fmt: skip


def timing_of(path, mlp_path=None):
    """Per-case seconds from a :mod:`.train` timing run (``--validate-train``): an
    iteration's (median over iterations), the bias fill's (scaled per window) and the
    validation's; with ``mlp_path`` (the largest network's timing run) the ratio of its
    iteration time to the linear model's."""
    log = json.loads(path.read_text())
    per_iter = _per_iter(log["log"])
    batch = {c: (log["args"]["shear_batch"] if c.startswith("S") else log["args"]["batch"])
             for c in per_iter}  # fmt: skip
    per_window = {c: per_iter[c] / batch[c] for c in per_iter}
    validate = {c: v["seconds"] for c, v in log.get("validation", {}).items()}
    ratio = 1.0
    if mlp_path is not None:
        mlp = _per_iter(json.loads(mlp_path.read_text())["log"])
        ratio = sum(mlp.values()) / sum(per_iter[c] for c in mlp)
    fill = {c: per_window[c] * log["args"]["bias_init"] for c in per_iter}
    # the fixed cost a run pays besides its steady iterations, fill windows and validation:
    # the compiles (the fill's forward, the first iteration's gradient) and loading the
    # references -- measured as the time to the end of iteration 0 beyond one steady
    # iteration and the fill, plus the import / reference load before the clock starts
    # (lctime 2026-11-12: ~340 s a run, which the first sizing's fixed 120 s left out)
    startup = log["log"][0]["seconds"] - sum(per_iter.values()) - sum(fill.values()) + 40.0
    return {"iter": per_iter, "fill": fill, "validate": validate,
            "startup": max(startup, 120.0), "mlp_ratio": ratio}  # fmt: skip


def pack(costs, jobs):
    """Longest-processing-time packing: ``[[run indices] per job]``, the job loads."""
    order = sorted(range(len(costs)), key=costs.__getitem__, reverse=True)
    bins, loads = [[] for _ in range(jobs)], [0.0] * jobs
    for i in order:
        j = loads.index(min(loads))
        bins[j].append(i)
        loads[j] += costs[i]
    return bins, loads


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timing", default="train_linear_lctime3.json")
    ap.add_argument(
        "--timing-mlp",
        default="train_mlp_lctime3_mlp.json",
        help="the largest network's timing run ('' = cost as the linear model)",
    )
    ap.add_argument("--hours", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--min-iters", type=int, default=40)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--tag", default="v2")
    ap.add_argument(
        "--train-args",
        default="",
        help="extra train.py options of every run (the loss chosen by lcloss, e.g. "
        "'--shock-modes 3')",
    )
    args = ap.parse_args(argv)
    timing = timing_of(RESULTS / args.timing,
                       RESULTS / args.timing_mlp if args.timing_mlp else None)  # fmt: skip
    plan = runs(design())
    budget = args.hours * 3600.0
    fixed = sum(run_cost(c, v, 0, timing, p) for _, c, v, _, p in plan)
    per_iter = sum(run_cost(c, v, 1, timing, p) - run_cost(c, v, 0, timing, p)
                   for _, c, v, _, p in plan)  # fmt: skip
    iters = math.floor((budget - fixed) / per_iter)
    if iters < args.min_iters:
        raise SystemExit(f"{args.hours} GPU-h allow only {iters} iterations per run (fixed "
                         f"{fixed / 3600:.1f} h): raise --hours or cut the design")  # fmt: skip
    iters = min(iters, args.max_iters)
    costs = [run_cost(c, v, iters, timing, p) for _, c, v, _, p in plan]
    bins, loads = pack(costs, args.jobs)
    out = {"iters": iters, "hours": sum(costs) / 3600, "job_hours": [x / 3600 for x in loads],
           "timing": timing, "design": design(),
           "runs": [{"tag": t, "cases": c, "validate": v, "seed": s, **p}
                    for t, c, v, s, p in plan],
           "jobs": bins, "train_args": args.train_args}  # fmt: skip
    (RESULTS / f"sweep_plan_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{len(plan)} runs x {iters} iterations = {out['hours']:.1f} GPU-h over {args.jobs} "
        f"jobs ({', '.join(f'{h:.1f}' for h in out['job_hours'])} h)"
    )


if __name__ == "__main__":
    main()
