#!/usr/bin/env python3
"""sparta_jax against SPARTA on the three bench decks: per-move event rates over seeds.

    # SPARTA's side (any CPU; serial SPARTA, seconds to minutes) -- done once, committed as
    # bench/validation/sparta_serial_rows.jsonl (SPARTA af66be39, 10K and 100K, 32 seeds):
    python3 bench/validate.py sparta --exe /path/to/spa_serial --out bench/results/validate
    # sparta_jax's side (a GPU job: `python3 bench/matrix.py validate --platform ...`, or local):
    python3 bench/validate.py jax --out bench/results/validate
    # compare:
    python3 bench/validate.py compare --out bench/results/validate

Both sides run the decks at the bench README sizes 10K and 100K with 8 seeds and record the
timed run's totals.  The comparison is the z-score of the difference of the per-move rates
(attempts, collisions, surface checks/collisions, boundary exits/collides, final particle
count), z = (mean_jax - mean_sparta) / sqrt(se_jax^2 + se_sparta^2).  |z| < 3 everywhere is
the gate for timing runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ("10K", "100K")
DECKS = ("free", "collide", "sphere")
SEEDS = 8
# SPARTA's side: 32 serial seeds (cheap).  The 8-seed reference was unlucky on collide 100K
# nbound (mean low and spread 2.6x below the 32-seed one) and failed sparta_jax at z = 3.4 on
# A100 and MI100 alike; against 32 seeds the same rows give z = 1.5 (2026-11-28).
SPARTA_SEEDS = 32
RATES = ("nattempt", "ncoll", "nscheck", "nscoll", "nexit", "nbound")


def run_side(code, a):
    for deck in DECKS:
        for n in SIZES:
            cmd = [sys.executable, os.path.join(HERE, "run_bench.py"), code, "--deck", deck, "--particles", n,
                   "--machine", a.machine, "--variant", "validate", "--reps", str(SPARTA_SEEDS if code == "sparta" else SEEDS), "--out", a.out]
            if code == "sparta":
                cmd += ["--exe", a.exe]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


def compare(a):
    files = a.rows or [os.path.join(a.out, "rows.jsonl")]
    rows = [json.loads(ln) for f in files for ln in open(f) if ln.strip()]
    worst = 0.0
    print(f"{'deck':8s} {'size':>6s} {'quantity':10s} {'sparta_jax':>12s} {'SPARTA':>12s} {'z':>7s}")
    for deck in DECKS:
        for n in SIZES:
            per = {}
            for code in ("jax", "sparta"):
                rs = [r for r in rows if r["code"] == code and r["deck"] == deck and r["variant"] == "validate"
                      and r["particles"] == int(n[:-1]) * (1000 if n.endswith("K") else 10**6)]
                if not rs:
                    continue
                vals = {}
                for q in RATES:
                    vals[q] = np.array([r["totals"].get(q, 0) / r["particle_moves"] for r in rs])
                vals["np_end"] = np.array([r.get("np_end", r["totals"].get("np_end", 0)) or 0 for r in rs], float)
                per[code] = vals
            if len(per) < 2:
                print(f"{deck:8s} {n:>6s} (missing a side)")
                continue
            for q in RATES:
                xj, xs = per["jax"][q], per["sparta"][q]
                if xj.mean() == 0 and xs.mean() == 0:
                    continue
                se = np.sqrt(xj.var(ddof=1) / len(xj) + xs.var(ddof=1) / len(xs))
                z = (xj.mean() - xs.mean()) / se if se > 0 else (0.0 if xj.mean() == xs.mean() else np.inf)
                worst = max(worst, abs(z))
                print(f"{deck:8s} {n:>6s} {q:10s} {xj.mean():12.6g} {xs.mean():12.6g} {z:7.2f}")
    print(f"max |z| = {worst:.2f}: {'PASS' if worst < 3 else 'CHECK'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("sparta", "jax", "compare"))
    p.add_argument("--exe")
    p.add_argument("--machine", default="validate")
    p.add_argument("--out", default=os.path.join(HERE, "results", "validate"))
    p.add_argument("--rows", nargs="+", help="row files to compare (default <out>/rows.jsonl); SPARTA's serial "
                   "reference rows are committed in bench/validation/sparta_serial_rows.jsonl")
    a = p.parse_args()
    if a.what == "compare":
        compare(a)
    else:
        run_side(a.what, a)


if __name__ == "__main__":
    main()
