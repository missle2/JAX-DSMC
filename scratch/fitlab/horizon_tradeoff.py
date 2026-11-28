"""Bias against noise of the captured horizon (2026-11-28): would the optimiser rather have
the 300-step capture's lower noise or a longer capture's smaller bias?

From a long-capture dump (c10_spec_h1000: window 200 + 800 steps of history), per seed and per
surface sector the production pathwise (x + v) contribution of the hits at most L steps before
the window -- the surface field a capture of window + L steps would give.  The radius control
moves every hit by a unit normal displacement, so its per-hit contributions binned by sector are
the gradient with respect to each sector's own normal displacement (the optimiser's local
controls, here without the normal rotation; cos 2 theta's add it).  Per L:

  bias^2   = sum_s (mean_L(s) - mean_ref(s))^2      (ref: the longest L, all seeds)
  var      = sum_s var_seeds(L, s)                  (ONE seed's noise, the optimiser's S1)
  mse      = bias^2 + var, relative to sum_s mean_ref(s)^2, and the expected cosine of one
             seed's field with the reference field.

The other terms (tally, volume, timing) do not depend on L: they add the same noise to every L,
so the DIFFERENCE between L's is what decides.  Finer sectors (more, smaller controls, as in 3-D)
raise the variance at fixed bias.

    python3 scratch/fitlab/horizon_tradeoff.py [--case spec_h1000] [--sectors 36 144]
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

LAGS = (0, 50, 100, 150, 200, 300, 400, 600)


def fields(case, n_sec, control):
    """``[seeds, lags, sectors]``: the per-sector pathwise total with history <= L."""
    out = []
    for _seed, path in lab.seed_files(lab.CASES[case]):
        meta = json.loads(path.with_suffix(".json").read_text())
        z = lab.load(path, ("hit_step", "hit_x", "hit_dx", "hit_dv", "hit_gx", "hit_gv"))
        prod = meta["variants"].index(lab.BASE_VARIANT)
        i = meta["controls"].index(control)
        n, window = len(meta["steps"]), meta["config"]["window"]
        lag = (n - window) - 1 - z["hit_step"].astype(np.int64)  # < 0: inside the window
        g = (np.sum(z["hit_gx"][prod] * z["hit_dx"][i], axis=1)
             + np.sum(z["hit_gv"][prod] * z["hit_dv"][i], axis=1))  # fmt: skip
        rel = z["hit_x"] - lab.CENTRE
        sec = (np.floor((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (2 * np.pi) * n_sec)
               .astype(int) % n_sec)  # fmt: skip
        rows = []
        for big_l in LAGS:
            keep = lag < big_l
            rows.append(np.bincount(sec[keep], weights=g[keep], minlength=n_sec))
        out.append(rows)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="spec_h1000")
    ap.add_argument("--sectors", type=int, nargs="+", default=[36, 144])
    ap.add_argument("--controls", nargs="+", default=["radius", "mode2"])
    args = ap.parse_args()
    res = {}
    for control in args.controls:
        for n_sec in args.sectors:
            f = fields(args.case, n_sec, control)
            ref = f[:, -1].mean(axis=0)
            norm = float(np.sum(ref**2))
            print(f"\n{args.case} {control}, {n_sec} sectors, {len(f)} seeds "
                  f"(ref: L = {LAGS[-1]})")  # fmt: skip
            print(f"  {'L':>5s} {'bias2/ref2':>11s} {'var1/ref2':>10s} {'mse1/ref2':>10s} "
                  f"{'E cos (1 seed)':>15s}")  # fmt: skip
            rows = []
            for j, big_l in enumerate(LAGS):
                mean = f[:, j].mean(axis=0)
                bias2 = float(np.sum((mean - ref) ** 2))
                var = float(np.sum(f[:, j].var(axis=0, ddof=1)))
                cos = np.mean([np.dot(x, ref) / np.linalg.norm(x) / np.linalg.norm(ref)
                               for x in f[:, j]])  # fmt: skip
                rows.append({"L": big_l, "bias2": bias2 / norm, "var": var / norm,
                             "mse": (bias2 + var) / norm, "cos": float(cos)})  # fmt: skip
                print(f"  {big_l:5d} {bias2 / norm:11.4f} {var / norm:10.4f} "
                      f"{(bias2 + var) / norm:10.4f} {cos:15.4f}")  # fmt: skip
            res[f"{control}_{n_sec}"] = rows
    path = lab.OUT / f"horizon_tradeoff_{args.case}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
