"""Score the kNN wall fit's reference-scale sweep (2026-11-27, paper plan phase A; study group
knntune -> scratch/fitlab/cluster/knn3_<case>_T_cm<c>.json).

Each fit ``k3_C<c_mfp>_<cell>_0.033_<v>_o0[_k<k>]`` (wall-distance scale min(c_mfp x the wall
mean free path, cell x the cell size), velocity scale v x the thermal spread, k neighbours;
k = 64 unmarked) is scored on the diffuse drag cases (radius and cos 2 theta) by the summary
figure's per-seed error (figures._seed_error: sqrt(bias^2 + var / N_eff) / |FD|, N_eff 10),
RMS over the 12 columns; also per case the adjoint / FD ratios and the cosine between the
seed-mean adjoint and FD vectors over every control with an FD (the direction a step takes).
The nose bump (drag only) is listed, not scored.  Production (per-cell fit) for reference.

    PYTHONPATH=. python3 scratch/fitlab/tune.py [--top 15]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import figures  # noqa: E402
import lab  # noqa: E402
import summary  # noqa: E402

CLUSTER = Path("scratch/fitlab/cluster")
CASES = ("drag", "m03_diff", "kn10", "hot", "drag_g2", "m25_diff")
COLS = ("radius", "mode2")
PATTERN = re.compile(r"k3_C([\d.]+)_([\d.]+)_0\.033_([\d.]+)_o0(?:_k(\d+))?$")


def load(case):
    """``{fit: rows}`` merged over the c_mfp chunks (same seeds in each)."""
    out = {}
    for path in sorted(CLUSTER.glob(f"knn3_{case}_T_cm*.json")):
        rows = json.loads(path.read_text())["rows"]
        if not rows:
            continue
        for name in rows[0]["pathwise_x"]["radius"]:
            out[name] = rows
    return out


def adjoint_mean(rows, c, name):
    return np.mean([r["base_total"][c] - r["pathwise_x"][c][lab.BASE_VARIANT]
                    + r["pathwise_x"][c][name] for r in rows])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    data = {case: load(case) for case in CASES}
    fds = {case: lab.finite_differences(case) for case in CASES}
    fits = sorted(set.intersection(*(set(d) for d in data.values() if d)))
    fits = [f for f in fits if PATTERN.match(f)] + [lab.BASE_VARIANT]
    table = []
    for name in fits:
        errs, ratios, cosines, nose = [], {}, {}, None
        for case in CASES:
            rows = data[case].get(name) or next(iter(data[case].values()))
            for c in COLS:
                errs.append(figures._seed_error(rows, fds[case], c, name))
                ratios[f"{case}:{c}"] = summary.closure(rows, fds[case], c, name)[0]
            ctl = [c for c in rows[0]["pathwise_x"] if c in fds[case] and c != "mode2s"]
            a = np.array([adjoint_mean(rows, c, name) for c in ctl])
            f = np.array([np.mean(list(fds[case][c].values())) for c in ctl])
            cosines[case] = float(a @ f / np.linalg.norm(a) / np.linalg.norm(f))
            if case == "drag":
                nose = summary.closure(rows, fds[case], "bump_nose", name)[0]
        m = PATTERN.match(name)
        params = ({"c_mfp": float(m[1]), "cell": float(m[2]), "v": float(m[3]),
                   "k": int(m[4] or 64)} if m else {})  # fmt: skip
        table.append({"fit": name, **params, "rms": float(np.sqrt(np.mean(np.square(errs)))),
                      "worst": float(np.max(errs)), "ratios": ratios, "cosine": cosines,
                      "min_cosine": min(cosines.values()), "nose": nose})  # fmt: skip
    table.sort(key=lambda r: r["rms"])
    head = "".join(f"{c.split(':')[0][:7]:>8s}{'r' if c.endswith('radius') else 'm':1s}"
                   for c in table[0]["ratios"])  # fmt: skip
    print(f"{'fit':34s}{'rms':>7s}{'worst':>7s}{'cos_min':>8s}{'nose':>7s} {head}")
    ref = {"k3_C1_0.25_0.033_0.3_o0", lab.BASE_VARIANT}
    for i, r in enumerate(table):
        if i < args.top or r["fit"] in ref:
            cells = "".join(f"{v:9.3f}" for v in r["ratios"].values())
            print(f"{i + 1:3d} {r['fit']:30s}{r['rms']:7.3f}{r['worst']:7.3f}"
                  f"{r['min_cosine']:8.4f}{r['nose']:7.2f} {cells}")
    # sensitivity: the RMS spread along each parameter around the best fit's others
    best = table[0]
    print(f"\nbest {best['fit']}: sensitivity (rms) along each parameter, the others at best")
    for p in ("c_mfp", "cell", "v", "k"):
        line = sorted((r[p], r["rms"]) for r in table if r.get(p) is not None and all(
            r[q] == best[q] for q in ("c_mfp", "cell", "v", "k") if q != p))  # fmt: skip
        print(f"  {p:6s} " + "  ".join(f"{v:g}: {e:.3f}" for v, e in line))
    out = lab.OUT / "tune.json"
    out.write_text(json.dumps(table, indent=1))
    print(f"\n{len(table) - 1} fits scored; wrote {out}")


if __name__ == "__main__":
    main()
