"""Data of the paper's figures 5-7 from the 2-D lsfilter / optrc / ldopt and 3-D lsfilter3d run
histories (examples/mesh/results/optimize/<tag>/history.json), as CSV for pgfplots:

* ``hist_<problem>_<arm>.csv`` (``iterate, value, mean``): the objective (J / J0 for drag, L / D
  for the lift-to-drag runs) and its 25-iterate running mean, for the arms ``hz_adam``
  (Helmholtz, Adam), ``hz_mma`` (Helmholtz, MMA) and ``band_adam`` (band filter, reinitialised,
  Adam: the earlier formulation), cut at a non-finite objective;
* ``cost.csv``: the median seconds per phase of an iterate (iterates 20 on) for four runs.

    python3 examples/paper/figures/export_histories.py --out docs/paper/figures/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "mesh" / "results" / "optimize"
C, W = "circle_long_kn0.5", "free_g32_w500"
RUNNING = 25
# problem -> (lift?, {arm: tag})
PROBLEMS = {
    "m10_drag": (False, {"hz_adam": f"{C}_M10_S1_ox_adam_hz0_tan_sm8_{W}",
                         "hz_mma": f"{C}_M10_S1_mma_s_hz0_tan_sm8_{W}",
                         "band_adam": f"{C}_M10_S1_ox_adam_rc_tan_sm8_{W}"}),
    "kn_drag": (False, {"hz_adam": f"{C}_S1_ox_adam_hz0_tan_sm8_{W}",
                        "hz_mma": f"{C}_S1_mma_s_hz0_tan_sm8_{W}",
                        "band_adam": f"{C}_S1_ox_adam_fix_tan_sm8_{W}"}),
    "kn_ld": (True, {"hz_adam": f"{C}_S1_ox_adam_ldhz0_floor_sm8_{W}",
                     "hz_mma": f"{C}_S1_mma_s_ldhz0_floor_sm8_{W}",
                     "band_adam": f"{C}_S1_ox_adam_ldfl_floor_sm8_{W}"}),
    "m10_ld": (True, {"hz_adam": f"{C}_M10_S1_ox_adam_ldhz0_floor_sm8_{W}",
                      "hz_mma": f"{C}_M10_S1_mma_s_ldhz0_floor_sm8_{W}",
                      "band_adam": f"{C}_M10_S1_ox_adam_ldfl_floor_sm8_{W}"}),
}  # fmt: skip
PHASES = ("build", "warmup", "capture", "replay", "backward", "shape")
COST = {
    "2-D Kn 0.5": f"{C}_S1_ox_adam_hz0_tan_sm8_{W}",
    "2-D Ma 10": f"{C}_M10_S1_ox_adam_hz0_tan_sm8_{W}",
    "3-D Kn 0.5": "sphere_long_kn0.5_S1_ox_adam_drag_hz0_free_w300",
    "3-D Ma 10": "sphere_long_kn0.5_M10_S1_ox_adam_ldfl_hz0_free_w300",
}


def running(x, w=RUNNING):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


def history(tag, lift):
    rows = json.loads((ROOT / tag / "history.json").read_text())["rows"]
    j = np.array([r["objective"] for r in rows], float)
    if not np.isfinite(j).all():
        j = j[: int(np.argmin(np.isfinite(j)))]
    value = -j if lift else j / j[0]
    return value, running(value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for problem, (lift, arms) in PROBLEMS.items():
        for arm, tag in arms.items():
            if not (ROOT / tag / "history.json").exists():
                print(f"missing {tag}")
                continue
            value, mean = history(tag, lift)
            it = np.arange(len(value))
            np.savetxt(args.out / f"hist_{problem}_{arm}.csv", np.c_[it, value, mean],
                       delimiter=",", header="iterate,value,mean", comments="", fmt="%.6g")
            print(f"{problem} {arm}: {len(value)} iterates, last mean {mean[-1]:.4g}")
    rows = []
    for label, tag in COST.items():
        hist = json.loads((ROOT / tag / "history.json").read_text())["rows"][20:]
        phases = np.array([[r["phase_seconds"].get(p, np.nan) for p in PHASES] for r in hist])
        med = np.nanmedian(phases, axis=0)
        total = float(np.median([r["seconds"] for r in hist]))
        rows.append([label, *med.round(2), round(total - float(np.nansum(med)), 2), round(total, 1)])
    with open(args.out / "cost.csv", "w") as fh:
        fh.write("case," + ",".join(PHASES) + ",other,total\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
