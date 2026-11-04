"""Diffuse production check (2026-11-03): adjoint vs FD of drag (mode2), total heat rate (mode2)
and D - L (mode2s: dD = 0 by symmetry, so it is -dL) on the c10 diffuse circle, the three wall
fits.  Prints per gate and variant: mean adjoint, FD_h / FD_h2 means, ratio +- 2 s.e., paired t.

    PYTHONPATH=. python3 scratch/fitgate/diffchk.py [gate ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitgate")
from analyze import TERMS  # noqa: E402

RESULTS = Path("examples/mesh/results/fitgate")
GATES = sys.argv[1:] or ["c10_diff_m2", "c10_diff_heat_m2", "c10_diff_lift_m2s", "c10_diff_kn10_m2",
                         "c10_spec_m2", "c10_diff_m2_gz001", "c10_diff_m2_sil2"]


def rows_of(gate):
    by = {}
    for path in sorted(RESULTS.glob(f"{gate}_base*_s*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            by[r["seed"]] = r
    return [by[s] for s in sorted(by)]


for gate in GATES:
    rows = rows_of(gate)
    if not rows:
        print(f"\n{gate}: no rows yet")
        continue
    fd = np.array([r["fd"][0] for r in rows])
    fd2 = np.array([r["fd"][1] for r in rows])
    n = len(rows)
    print(f"\n{gate}: {n} seeds  FD_h {fd.mean():+.4f} +- {2 * fd.std(ddof=1) / n**0.5:.4f}"
          f"  FD_h/2 {fd2.mean():+.4f} +- {2 * fd2.std(ddof=1) / n**0.5:.4f}"
          f"  J {np.mean([r['objective'] for r in rows]):.4f}")
    for v in rows[0]["terms"]:
        a = np.array([sum(r["terms"][v][k] for k in TERMS) + r["volume"] for r in rows])
        d = a - fd
        se = d.std(ddof=1) / n**0.5
        terms = {k: np.mean([r["terms"][v][k] for r in rows]) for k in TERMS}
        big = "  ".join(f"{k} {x:+.4f}" for k, x in terms.items() if abs(x) > 1e-4)
        print(f"  {v:24s} adj {a.mean():+.4f}  ratio {a.mean() / fd.mean():.3f} +- "
              f"{2 * se / abs(fd.mean()):.3f}  t {d.mean() / se:+.1f}  "
              f"ratio_h2 {a.mean() / fd2.mean():.3f}  sd/|FD| {a.std(ddof=1) / abs(fd.mean()):.2f}"
              f"   [{big}  volume {np.mean([r['volume'] for r in rows]):+.4f}]")

# the derivative-only m2chk gates against c10_diff_m2 seed by seed (the same forward and FD):
# the paired change of the total and of each term, membership (reported, not summed) if any
PAIRS = {"c10_diff_m2_gz001": "c10_diff_m2", "c10_diff_m2_sil2": "c10_diff_m2",
         "c10_diff_m2_ord128": "c10_diff_m2", "c10_diff_m2_sf": "c10_diff_m2",
         "c10_diff_lift_m2s_sf": "c10_diff_lift_m2s", "c10_diff_heat_m2_sf": "c10_diff_heat_m2"}
for gate in GATES:
    if gate not in PAIRS:
        continue
    mine = {r["seed"]: r for r in rows_of(gate)}
    base = {r["seed"]: r for r in rows_of(PAIRS[gate])}
    seeds = sorted(set(mine) & set(base))
    if not seeds:
        continue
    fd = np.mean([base[s]["fd"][0] for s in seeds])
    print(f"\n{gate} - {PAIRS[gate]}, paired over seeds {seeds}:")
    for v in mine[seeds[0]]["terms"]:
        if v not in base[seeds[0]]["terms"]:
            continue
        d = np.array([sum(mine[s]["terms"][v][k] - base[s]["terms"][v][k] for k in TERMS)
                      + mine[s]["volume"] - base[s]["volume"] for s in seeds])  # fmt: skip
        dt = {k: np.mean([mine[s]["terms"][v][k] - base[s]["terms"][v][k] for s in seeds])
              for k in TERMS}  # fmt: skip
        mem = [mine[s]["terms"][v].get("membership") for s in seeds]
        extra = f"  membership {np.mean(mem) / fd:+.3f}" if None not in mem else ""
        print(f"  {v:24s} d(total)/FD {d.mean() / fd:+.3f} +- {2 * d.std(ddof=1) / len(d)**0.5 / abs(fd):.3f}  "
              + " ".join(f"{k} {x / fd:+.3f}" for k, x in dt.items() if abs(x) > 1e-4) + extra)
