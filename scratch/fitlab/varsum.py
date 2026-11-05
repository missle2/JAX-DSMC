"""Bias and variance of each fit (2026-11-05): per case and control, the adjoint / FD ratio, its
excess in standard errors and the per-seed spread of the adjoint total relative to production's
(``cell_along_pool_plane``): the same seeds, the same other terms, only pathwise_x differs, so
the spread ratio is the fit's own variance cost.

    PYTHONPATH=. python3 scratch/fitlab/varsum.py out/win_drag.json out/knn_drag.json \\
        [--fits win_k64_s0.1_v0.5 knn_k64_d0.03_s0.1_v0.5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402
import summary  # noqa: E402


def totals(rows, c, name):
    return np.array([r["base_total"][c] - r["pathwise_x"][c][lab.BASE_VARIANT]
                     + r["pathwise_x"][c][name] for r in rows])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fits", nargs="*", default=None)
    args = ap.parse_args()
    for p in args.paths:
        data = json.loads(Path(p).read_text())
        case, rows = data["case"], data["rows"]
        fds = lab.finite_differences(case)
        controls = [c for c in summary.ORDER if c in fds and c in rows[0]["pathwise_x"]
                    and c != "bump_base"]  # fmt: skip
        print(f"\n{case} ({Path(p).stem}, {len(rows)} seeds): ratio (excess/se) [spread / "
              "production's spread]")  # fmt: skip
        print(f"  {'fit':30s} " + " ".join(f"{c:>23s}" for c in controls))
        for name in rows[0]["pathwise_x"][controls[0]]:
            if args.fits and name not in args.fits and name != lab.BASE_VARIANT:
                continue
            cells = []
            for c in controls:
                ratio, t = summary.closure(rows, fds, c, name)
                spread = totals(rows, c, name).std(ddof=1)
                base = totals(rows, c, lab.BASE_VARIANT).std(ddof=1)
                head = f"({t:+5.1f})" if np.isnan(ratio) else f"{ratio:5.2f} ({t:+5.1f})"
                cells.append(f"{head} [{spread / base:4.2f}]")
            print(f"  {name[:30]:30s} " + " ".join(f"{x:>23s}" for x in cells))


if __name__ == "__main__":
    main()
