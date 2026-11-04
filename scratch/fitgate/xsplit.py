"""The specular pathwise_x split vs Kn (2026-11-03): along_pool_plane's pathwise_x = perp (the
`_perp` variant) + free-flight timing (`_free` - `_perp`) + collisional L* V (full - `_free`);
the excess = adjoint total - FD (the FD of the parent gate's every seed, FD_FROM).

    PYTHONPATH=. python3 scratch/fitgate/xsplit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitgate")
from analyze import TERMS  # noqa: E402

RESULTS = Path("examples/mesh/results/fitgate")
SPLITS = (("Kn 0.5", "c10_spec_xsplit", "c10_spec"), ("Kn 0.5 h1000", "c10_spec_h1000_xsplit",
          "c10_spec"), ("Kn 1", "c10_spec_kn1_xsplit", "c10_spec_kn1_xsplit"),
          ("Kn 2", "c10_spec_kn2_xsplit", "c10_spec_kn2"),
          ("Kn 10", "c10_spec_kn10_xsplit", "c10_spec_kn10"),
          ("diff radius", "c10_diff_xsplit", "c10_diff"),
          ("diff cos2", "c10_diff_m2_xsplit", "c10_diff_m2"),
          ("diff heat cos2", "c10_diff_heat_m2_xsplit", "c10_diff_heat_m2"))  # fmt: skip


def rows_of(gate):
    by = {}
    for path in sorted(RESULTS.glob(f"{gate}_base*_s*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["gate"] == gate and r.get("basis", "base") == "base":
                by[r["seed"]] = r
    return [by[s] for s in sorted(by)]


def tot(r, v):
    return sum(r["terms"][v][k] for k in TERMS) + r["volume"]


print(f"{'':14s} {'n':>3s} {'FD':>7s} {'+-2se':>6s} | {'excess/FD (along_pool_plane, cellxv, cell_pool)':>40s} | "
      "perp  timing  L*V  (/FD)  | pathwise_v  volume")
for label, gate, parent in SPLITS:
    rows = rows_of(gate)
    if not rows:
        continue
    prow = rows_of(parent)
    fd_all = np.array([r["fd"][0] for r in prow])
    fd = fd_all.mean()
    fd_se = fd_all.std(ddof=1) / len(fd_all) ** 0.5
    n = len(rows)
    ex = []
    for v in ("along_pool_plane", "cellxv_along_pool_plane", "cell_pool_plane"):
        if v not in rows[0]["terms"]:
            v = "cell_along_pool_plane"  # noqa: PLW2901 -- the diffuse splits carry it instead
        a = np.array([tot(r, v) for r in rows])
        se = np.hypot(a.std(ddof=1) / n**0.5, fd_se)
        ex.append(f"{(a.mean() - fd) / fd:+.3f}+-{2 * se / fd:.3f}")
    px = lambda v: np.mean([r["terms"][v]["pathwise_x"] for r in rows])  # noqa: E731
    perp, free, full = px("along_pool_plane_perp"), px("along_pool_plane_free"), px("along_pool_plane")
    pv = np.mean([r["terms"]["along_pool_plane"]["pathwise_v"] for r in rows])
    vol = np.mean([r["volume"] for r in rows])
    print(f"{label:14s} {n:3d} {fd:7.4f} {2 * fd_se:6.4f} | {'  '.join(ex):>40s} | "
          f"{perp / fd:+.3f} {(free - perp) / fd:+.3f} {(full - free) / fd:+.3f} | "
          f"{pv / fd:+.3f} {vol / fd:+.3f}")
