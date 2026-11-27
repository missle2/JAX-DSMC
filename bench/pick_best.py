#!/usr/bin/env python3
"""Turn a platform's tuning rows into bench/configs/<platform>.json (read by matrix.py).

    python3 bench/pick_best.py --platform pm-gpu --rows $RESULTS_DIR/rows.jsonl

Per code, the variant with the best geometric-mean speed-up over the decks it ran (relative to
the baseline variant: SPARTA "kk_default" / "mpi128", JAX "jax" / "jaxcpu_8x16"), among
variants that completed every deck the baseline completed.  A variant must beat the baseline by
more than ``--min-gain`` (default 3 %, about the rep-to-rep spread) to replace it.  The settings
of a variant are recovered from its plan line (env, accel, extra), recorded in the rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--platform", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--min-gain", type=float, default=0.03)
    a = p.parse_args()
    rows = matrix.load_rows(os.path.dirname(a.rows))
    gpu = matrix.PLATFORMS[a.platform]["gpu"]
    tune = {r.variant: r for r in matrix.runs_tune(a.platform, [], {})}
    cfg = {}
    for code, base in (("sparta", "kk_default" if gpu else "mpi128"), ("jax", "jax" if gpu else "jaxcpu_8x16")):
        med = defaultdict(list)
        for r in rows:
            want = 1 if matrix.single_gpu_only(a.platform) else tune[r["variant"]].nodes * tune[r["variant"]].ranks_per_node \
                if r["variant"] in tune else -1
            if r["code"] == code and r["variant"] in tune and r["procs"] == want:
                med[(r["variant"], r["deck"])].append(r["mpsps"])
        med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
        decks = {d for (v, d) in med if v == base}
        if not decks:
            print(f"{code}: no baseline rows ({base}); skipped")
            continue
        scores = {}
        for v in {v for (v, _d) in med}:
            if all((v, d) in med for d in decks):
                scores[v] = math.exp(sum(math.log(med[(v, d)] / med[(base, d)]) for d in decks) / len(decks))
        best = max(scores, key=scores.get)
        if scores[best] < 1.0 + a.min_gain:
            best = base
        r = tune[best]
        print(f"{code}: best {best} ({scores[best]:.3f}x baseline); " + ", ".join(f"{v} {s:.3f}" for v, s in sorted(scores.items(), key=lambda t: -t[1])))
        entry = {"tag": best, "env": r.env, "extra": r.extra}
        if code == "sparta":
            entry.update(accel=r.accel, build=r.sparta_build)
        elif not gpu:
            entry["cpu_layout"] = [r.ranks_per_node, r.threads]
        cfg[code] = entry
        if code == "sparta" and not gpu:
            cfg["cpu_ranks_per_node"] = r.ranks_per_node
    out = os.path.join(HERE, "configs", f"{a.platform}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
