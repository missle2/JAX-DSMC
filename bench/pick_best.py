#!/usr/bin/env python3
"""Turn a platform's tuning rows into bench/configs/<platform>.json (read by matrix.py).

    python3 bench/pick_best.py --platform pm-gpu        # rows found as matrix.py finds them

Per code and per tuned deck (collide, sphere), the fastest variant relative to the baseline
(SPARTA "kk_default" / "mpi128", JAX "jax" / "jaxcpu_8x16"); decks without tuning rows (free)
take the best geometric mean over the tuned decks.  Multi-GPU JAX layout variants (jax_node*)
are scored against jax_node, separately from the single-GPU knobs, and applied on top of them.  A variant must beat the baseline by
at least ``--min-gain`` (default 3 %, about the rep-to-rep spread) to replace it.  The settings
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
    p.add_argument("--results", default=matrix.default_results(),
                   help="results root (rows in <root>/<platform>/rows.jsonl) or a platform's results dir")
    p.add_argument("--rows", help="(old) a rows.jsonl file: its directory is used as --results")
    p.add_argument("--min-gain", type=float, default=0.03)
    a = p.parse_args()
    root = os.path.dirname(a.rows) if a.rows else a.results
    rows = []
    for d in {root, os.path.join(root, a.platform)}:  # (2026-11-28: a root path silently found nothing)
        rows += matrix.load_rows(d)
    rows = [r for r in rows if r.get("machine", "").startswith(a.platform)]
    print(f"{len(rows)} {a.platform} rows under {root}")
    if not rows:
        sys.exit(f"no rows for {a.platform}: pass --results <dir holding {a.platform}/rows.jsonl>")
    gpu = matrix.PLATFORMS[a.platform]["gpu"]
    tune = {r.variant: r for r in matrix.runs_tune(a.platform, [], {})}
    out = os.path.join(HERE, "configs", f"{a.platform}.json")
    # merge: a code without tuning rows here keeps its existing entry (e.g. pm-gpu starts from
    # alp-a100's single-GPU JAX tuning -- same GPU architecture -- and tunes only the node level)
    cfg = json.load(open(out)) if os.path.exists(out) else {}
    devices = matrix.PLATFORMS[a.platform]["devices"]

    def medians(code, variants):
        med = defaultdict(list)
        for r in rows:
            v = r["variant"]
            if r["code"] != code or v not in variants:
                continue
            want = 1 if matrix.single_gpu_only(a.platform) else tune[v].nodes * tune[v].ranks_per_node
            if r["procs"] == want:
                med[(v, r["deck"])].append(r["mpsps"])
        return {k: sorted(x)[len(x) // 2] for k, x in med.items()}

    def choose(med, base, label):
        """Per deck: the fastest variant (> min-gain over the baseline, else the baseline);
        overall (decks without their own tuning rows, e.g. free): best geometric mean."""
        decks = sorted({d for (v, d) in med if v == base})
        if not decks:
            print(f"{label}: no baseline rows ({base}); skipped")
            return None, {}
        per = {}
        for d in decks:
            ratio = {v: med[(v, d)] / med[(base, d)] for (v, dd) in med if dd == d}
            best = max(ratio, key=ratio.get)
            per[d] = best if ratio[best] >= 1.0 + a.min_gain else base
            top = ", ".join(f"{v} {x:.3f}" for v, x in sorted(ratio.items(), key=lambda t: -t[1])[:6])
            print(f"{label} {d}: {per[d]} ({ratio[per[d]]:.3f}x {base}); top: {top}")
        scores = {v: math.exp(sum(math.log(med[(v, d)] / med[(base, d)]) for d in decks) / len(decks))
                  for v in {v for (v, _d) in med} if all((v, d) in med for d in decks)}
        overall = max(scores, key=scores.get)
        if scores[overall] < 1.0 + a.min_gain:
            overall = base
        print(f"{label} overall (other decks): {overall} ({scores[overall]:.3f}x)")
        return overall, per

    def entry(code, v):
        r = tune[v]
        e = {"tag": v, "env": r.env, "extra": r.extra}
        if code == "sparta":
            e.update(accel=r.accel, build=r.sparta_build)
            if r.wrap:  # MPS daemon / PrgEnv wrapper; nodes filled in per run
                e["wrap"] = r.wrap.replace("with_mps.sh 1 --", "with_mps.sh {nodes} --")
            if gpu and r.ranks_per_node > devices:
                e["ranks_per_device"] = r.ranks_per_node // devices
            if not gpu:
                e["ranks_per_node"] = r.ranks_per_node
        elif not gpu:
            e["cpu_layout"] = [r.ranks_per_node, r.threads]
        return e

    for code, base in (("sparta", "kk_default" if gpu else "mpi128"), ("jax", "jax" if gpu else "jaxcpu_8x16")):
        # global particle/reorder competes like any setting: SPARTA's own GPU decks use it, and it
        # changes only the particles' memory order (2026-11-28; earlier excluded as a deck edit)
        cands = {v for v, r in tune.items() if r.code == code}
        node_level = set()
        if code == "jax" and gpu:  # device-level knobs vs the 1-GPU baseline; layout vs jax_node
            node_level = {v for v in cands if tune[v].ranks_per_node > 1}
        overall, per = choose(medians(code, cands - node_level), base, code)
        if overall is None:
            continue
        cfg[code] = entry(code, overall)
        cfg[code]["decks"] = {d: entry(code, v) for d, v in per.items()}
        if code == "sparta" and not gpu:
            cfg["cpu_ranks_per_node"] = tune[overall].ranks_per_node
        if node_level:
            nbest, _ = choose(medians(code, node_level), "jax_node", "jax multi-GPU")
            if nbest is not None:  # applied on top of the deck's device-level entry (rpn > 1)
                cfg[code]["node"] = {"suffix": nbest[len("jax_node"):], "env": tune[nbest].env}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
