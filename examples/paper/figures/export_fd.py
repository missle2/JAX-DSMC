"""Data of the paper's Appendix A figure (adjoint against paired central differences) as CSV
for pgfplots, from the gate runs' artifacts (examples/paper/results/fd/<name>.json) or, for a
run whose artifact was not written or not pulled, the ``seed=`` lines of its Slurm log
(examples/mesh/cluster/logs/paper_fd_<name>_<job>.out):

* ``fd_<name>.csv`` (``seed, adjoint, fd_large, fd_small``): per seed, the adjoint estimate and
  the two central differences (steps ``--fd-steps``, larger first), all over the seeds' mean
  objective's |dJ/dp| scale-free form (divided by the mean of the large-step differences);
* ``fd_<name>_band.csv`` (``x, adjoint, adjoint_lo, adjoint_hi, fd_lo, fd_hi``): the normalised
  adjoint mean and its +- 2 standard error band over the seeds (gross outliers, robust z of
  the adjoint above :data:`OUTLIER_Z`, left out and listed), and the large-step FD's band,
  at the panel's two ends;
* ``fd_summary.csv``: per gate the means and standard errors, raw and normalised.

    python3 examples/paper/figures/export_fd.py --out docs/paper/figures/data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FD = REPO / "examples" / "paper" / "results" / "fd"
LOGS = REPO / "examples" / "mesh" / "cluster" / "logs"
GATES = ("af_drag_radius", "af_drag_mode2", "af_heat_radius", "nw_fm_radius", "nw_drag_radius",
         "nw_heat_radius")  # fmt: skip
OUTLIER_Z = 10.0
"""Robust z (median absolute deviation) of a seed's adjoint above which it is left out of the
band: seed 4 of the airfoil drag gates (z ~ 25, one pathwise-x event in 16 seeds)."""
SEED = re.compile(r"^seed=(\d+) J=(\S+) ensemble=(\S+) .* FD=\[(\S+), (\S+)\]")


def _from_json(path):
    rows = json.loads(path.read_text())["rows"]
    return [(r["seed"], r["objective"], r["gradient"], *r["fd"][:2]) for r in rows]


def _from_log(name):
    rows = {}
    for log in sorted(LOGS.glob(f"paper_fd_{name}_*.out")):
        for line in log.read_text().splitlines():
            m = SEED.match(line)
            if m:
                rows[int(m[1])] = tuple(float(m[i]) for i in range(2, 6))
    return [(s, *v) for s, v in sorted(rows.items())]


def rows_of(name):
    path = FD / f"{name}.json"
    rows = _from_json(path) if path.exists() else _from_log(name)
    return np.array(rows, float).reshape(-1, 5)


def _mean_se(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def outliers(x):
    """Indices whose robust z exceeds :data:`OUTLIER_Z`."""
    mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return np.nonzero(np.abs(x - np.median(x)) > OUTLIER_Z * mad)[0] if mad > 0 else np.array([])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in GATES:
        r = rows_of(name)
        if not len(r):
            print(f"{name}: no seeds")
            continue
        seed, adj, fd1, fd2 = r[:, 0], r[:, 2], r[:, 3], r[:, 4]
        scale = abs(fd1.mean())
        np.savetxt(args.out / f"fd_{name}.csv", np.c_[seed, adj / scale, fd1 / scale, fd2 / scale],
                   delimiter=",", header="seed,adjoint,fd_large,fd_small", comments="",
                   fmt="%.6g")  # fmt: skip
        (a, sa), (f1, s1), (f2, s2) = _mean_se(adj), _mean_se(fd1), _mean_se(fd2)
        ma = float(np.median(adj))
        out = outliers(adj)
        kept = np.delete(adj, out)
        k, sk = _mean_se(kept)
        x = [-0.6, len(r) - 0.4]
        band = [[xi, k / scale, (k - 2 * sk) / scale, (k + 2 * sk) / scale,
                 (f1 - 2 * s1) / scale, (f1 + 2 * s1) / scale] for xi in x]  # fmt: skip
        np.savetxt(args.out / f"fd_{name}_band.csv", np.array(band), delimiter=",",
                   header="x,adjoint,adjoint_lo,adjoint_hi,fd_lo,fd_hi", comments="",
                   fmt="%.6g")  # fmt: skip
        summary.append([name, len(r), a, sa, ma, f1, s1, f2, s2, a / f1,
                        ";".join(str(int(seed[i])) for i in out), k / f1, 2 * sk / abs(f1)])  # fmt: skip
        print(
            f"{name}: {len(r)} seeds  adjoint {a:.4g} +- {sa:.2g} (median {ma:.4g})  "
            f"FD {f1:.4g} +- {s1:.2g} / {f2:.4g} +- {s2:.2g}  ratio {a / f1:.3f}  "
            f"without seeds {seed[out].astype(int).tolist()}: {k / f1:.3f} +- {2 * sk / abs(f1):.3f}"
        )
    with open(args.out / "fd_summary.csv", "w") as fh:
        fh.write(
            "gate,seeds,adjoint,adjoint_se,adjoint_median,fd_large,fd_large_se,fd_small,"
            "fd_small_se,ratio,outlier_seeds,ratio_kept,ratio_kept_2se\n"
        )
        for row in summary:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
