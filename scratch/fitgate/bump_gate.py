"""Where on the body the Kn 0.5 diffuse excess sits (2026-11-03): the diffuse M 10 circle's
adjoint against central differences under three localised controls -- ``bump_nose`` (the
stagnation region), ``bump_side`` (both sides: the silhouette) and ``bump_base`` (the wake),
``ensemble_mesh.BUMP_CENTRES`` -- with every term projected onto each from ONE forward and
backward per seed (``EM.control_gradient`` of the same cotangents), so the term that carries
the excess on the bump that carries it is read directly.  After silfix the cos 2 theta drag
reads 1.22-1.27 and the heat 1.46-1.67 while the radius passes: a weight misplaced over the
surface.  Rows to ``examples/mesh/results/fitgate/bump_<objective>_s<a>-<b>.jsonl``.

    PYTHONPATH=. python3 scratch/fitgate/bump_gate.py --objective drag|heat --seeds 0 4
    PYTHONPATH=. python3 scratch/fitgate/bump_gate.py --report

``--report``: per objective and bump, the seed-mean FD, the adjoint total per variant, their
ratio (+- 2 s.e., paired) and the term means.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

OUT = Path("examples/mesh/results/fitgate")
BUMPS = ("bump_nose", "bump_side", "bump_base")
GATES = {"drag": "c10_diff_m2", "heat": "c10_diff_heat_m2"}  # their settings, not their FDs
STEP = 0.1
TERMS = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette", "order")


def run(objective, seeds, variants, smoke=False):
    cfg, _par, _steps, seed_start, _rows = V.gate_config(GATES[objective])
    tag = ""
    if smoke:  # the plumbing only (variant_gate --smoke's sizes)
        cfg = {**cfg, "fnum_scale": 16.0 * cfg["fnum_scale"], "warmup": 150, "steps": 40,
               "window": 20}  # fmt: skip
        tag = "_smoke"
    out = OUT / f"bump_{objective}{tag}_s{seeds[0]}-{seeds[1]}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines()}
    ensemble_backward.configure("virtual", 4, 8, "staged", V.BASES["base"])
    for seed in range(*seeds):
        if seed in done:
            continue
        key = jax.random.key(seed_start + seed)
        t0 = time.time()
        all_terms, info, s = EM.shape_terms(
            cfg, key, None, BUMPS[0], 32, cfg.get("timing_band", 10.0), tuple(variants)
        )
        row = {"objective": objective, "seed": seed, "J": info["objective"], "step": STEP,
               "controls": {}}  # fmt: skip
        for par in BUMPS:
            terms, volume = V.terms_row(cfg, par, all_terms, s)
            row["controls"][par] = {"terms": terms, "volume": volume}
        t_adj = time.time() - t0
        for par in BUMPS:
            row["controls"][par]["fd"] = EM.finite_differences(cfg, par, key, [STEP])[0]
        row["seconds_adjoint"], row["seconds"] = t_adj, time.time() - t0
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        fds = {p: round(row["controls"][p]["fd"], 5) for p in BUMPS}
        print(f"{objective} seed {seed}: J {info['objective']:.6g}  FD {fds}  "
              f"{row['seconds']:.0f}s", flush=True)  # fmt: skip


def report():
    for objective in GATES:
        rows = {}
        for path in sorted(OUT.glob(f"bump_{objective}_s*.jsonl")):
            if "_smoke" in path.name:  # bump_<objective>_smoke_* matches the glob too
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[r["seed"]] = r
        if not rows:
            continue
        rows = [rows[k] for k in sorted(rows)]
        n = len(rows)
        print(f"\n{objective}: {n} seeds")
        for par in BUMPS:
            fd = np.array([r["controls"][par]["fd"] for r in rows])
            print(f"  {par:10s} FD {fd.mean():+.4g} +- {2 * fd.std(ddof=1) / n**0.5:.2g}")
            for v in rows[0]["controls"][par]["terms"]:
                c = [r["controls"][par] for r in rows]
                a = np.array([sum(x["terms"][v][k] for k in TERMS) + x["volume"] for x in c])
                d = a - fd
                ratio = a.mean() / fd.mean()
                se = 2 * d.std(ddof=1) / n**0.5 / abs(fd.mean())
                means = {k: np.mean([x["terms"][v][k] for x in c]) for k in TERMS}
                means["volume"] = np.mean([x["volume"] for x in c])
                parts = "  ".join(f"{k[:9]} {m:+.4g}" for k, m in means.items() if abs(m) > 1e-9)
                print(f"    {v:24s} adj {a.mean():+.4g}  ratio {ratio:.3f} +- {se:.3f}  "
                      f"excess {a.mean() - fd.mean():+.4g}   [{parts}]")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=list(GATES))
    ap.add_argument("--seeds", type=int, nargs=2)
    ap.add_argument("--variants", nargs="+",
                    default=["cell_along_pool_plane", "cell_pool_plane"])  # fmt: skip
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fnum x 16, 150 + 40 steps, *_smoke")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        run(args.objective, args.seeds, args.variants, args.smoke)


if __name__ == "__main__":
    main()
