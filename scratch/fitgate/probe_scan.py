"""Probe-gate scan: per probe, the seed-mean FD (each eps) vs the seed-mean prediction."""
import json, sys
from pathlib import Path
import numpy as np
R = Path("examples/mesh/results/fitgate")
for tag in ("probe_c10_spec_s", "probe_c10_spec_kn10_s"):
    rows = [json.loads(l) for p in sorted(R.glob(tag + "*.jsonl")) for l in p.read_text().splitlines()]
    rows = [r for r in rows if not r.get("smoke")]
    n = len(rows)
    print(tag, "n", n, "hits", np.mean([r["hits"] for r in rows]), "J", np.mean([r["objective"] for r in rows]))
    for k in ("xn", "vs", "vr"):
        fd = np.array([r["fd"][k] for r in rows])
        print(f"  {k} eps={rows[0]['eps'][k]} FD per est: " + "  ".join(
            f"{m:+.4f}±{2*s/np.sqrt(n):.4f}" for m, s in zip(fd.mean(0), fd.std(0, ddof=1))))
        for v in rows[0]["pred"]:
            p = np.array([r["pred"][v][k] for r in rows])
            print(f"     {v:24s} pred {p.mean():+.4f}±{2*p.std(ddof=1)/np.sqrt(n):.4f}")
