"""One line per fit: the adjoint / FD ratio (excess / 1 se in brackets: the adjoint's and the
FD's seed-mean s.e. in quadrature) of every control with an FD, from the lab's closure files (scratch/fitlab/out/{ref,self,cand}_<case><tag>.json).

    PYTHONPATH=. python3 scratch/fitlab/summary.py out/self_drag_*.json [--fits ref cell_pool_plane]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

ORDER = ("radius", "mode2", "mode2s", "bump_nose", "bump_side", "bump_base")


def closure(rows, fds, c, name):
    n = len(rows)
    tot = np.array([r["base_total"][c] - r["pathwise_x"][c][lab.BASE_VARIANT]
                    + r["pathwise_x"][c][name] for r in rows])  # fmt: skip
    fdv = np.array(list(fds[c].values()))
    fd_se = fdv.std(ddof=1) / len(fdv) ** 0.5 if len(fdv) > 1 else 0.0
    se = np.hypot(tot.std(ddof=1) / n**0.5, fd_se)
    excess = tot.mean() - fdv.mean()
    return tot.mean() / fdv.mean() if abs(fdv.mean()) > 1e-12 else np.nan, excess / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fits", nargs="*", default=None)
    args = ap.parse_args()
    by_case = {}
    for p in args.paths:
        data = json.loads(Path(p).read_text())
        by_case.setdefault(data["case"], []).append((Path(p).stem, data["rows"]))
    for case, files in by_case.items():
        fds = lab.finite_differences(case)
        controls = [c for c in ORDER if c in fds and c in files[0][1][0]["pathwise_x"]]
        print(f"\n{case}: ratio (excess / se); mode2s: excess / se only (FD 0 by symmetry)")
        print(f"  {'file':28s} {'fit':24s} " + " ".join(f"{c:>15s}" for c in controls))
        for stem, rows in files:
            names = rows[0]["pathwise_x"][controls[0]]
            for name in names:
                if args.fits and name not in args.fits:
                    continue
                cells = []
                for c in controls:
                    ratio, t = closure(rows, fds, c, name)
                    cells.append(f"({t:+5.1f})" if np.isnan(ratio) else f"{ratio:6.3f} ({t:+5.1f})")
                print(f"  {stem[:28]:28s} {name[:24]:24s} " + " ".join(f"{x:>15s}" for x in cells))


if __name__ == "__main__":
    main()
