"""The specular horizon gate (2026-11-28, paper plan phase B): is the specular excess the
capture's truncation?

The objective is the drag over the last ``window`` captured steps; a hit at step k feeds it
through lambda_k, its value-to-go to the window's end.  In a stationary flow the hits BEFORE
the capture (held fixed by the adjoint) feed it too, and the FD (the shape changed for the whole
run) sees them.  Per captured step k the production pathwise contribution
G(k) = sum_{hits at k} (g_x . dx + g_v . dv) (the dump's hit_gx / hit_gv, variant 0, against
hit_dx / hit_dv) is profiled against the lag ``window start - k``; the cumulative sum from the
window's end back to lag L is the pathwise total a capture of ``window + L`` steps would give.
With the 1000-step capture (c10_spec_h1000) the profile shows how far back the hits matter
and the converged total; c10_spec (300 steps) is the same forward captured 100 steps before
its window.

    python3 scratch/fitlab/spec_lag.py [--cases spec spec_h1000]
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

CONTROLS = ("radius", "mode2")
BIN = 25  # steps per profile bin


def profile(case):
    """``{control: [seeds, captured steps]}`` per-step pathwise sums, steps counted back from
    the capture's end (index 0 = the last step), and the seeds' meta."""
    out = {c: [] for c in CONTROLS}
    meta0 = None
    for _seed, path in lab.seed_files(lab.CASES[case]):
        meta = json.loads(path.with_suffix(".json").read_text())
        meta0 = meta0 or meta
        z = lab.load(path, ("hit_step", "hit_dx", "hit_dv", "hit_gx", "hit_gv"))
        prod = meta["variants"].index(lab.BASE_VARIANT)
        n = len(meta["steps"])
        back = n - 1 - z["hit_step"].astype(np.int64)  # hit_step: the captured-step index
        for c in CONTROLS:
            i = meta["controls"].index(c)
            g = (np.sum(z["hit_gx"][prod] * z["hit_dx"][i], axis=1)
                 + np.sum(z["hit_gv"][prod] * z["hit_dv"][i], axis=1))  # fmt: skip
            out[c].append(np.bincount(back, weights=g, minlength=n))
    return {c: np.array(v) for c, v in out.items()}, meta0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["spec", "spec_h1000"])
    args = ap.parse_args()
    res = {}
    for case in args.cases:
        prof, meta = profile(case)
        window = meta["config"]["window"]
        fds = lab.finite_differences(case)
        res[case] = {}
        print(f"\n{case}: {len(meta['steps'])} captured steps, window {window}, "
              f"{len(prof['radius'])} seeds")  # fmt: skip
        for c in CONTROLS:
            g = prof[c]
            cum = np.cumsum(g, axis=1)  # pathwise total of a capture of (index + 1) steps
            n_seed = len(g)
            fd_m = np.mean(list(fds[c].values())) if c in fds else np.nan
            print(f"  {c}: FD {fd_m:+.4g}; pathwise (x + v) of a capture of window + L steps")
            rows = []
            for lag in range(0, g.shape[1] - window + 1, BIN):
                h = window + lag
                v = cum[:, h - 1]
                se = v.std(ddof=1) / n_seed**0.5
                # the bin's own share: the hits LAG..LAG+BIN steps before the window
                lo, hi = h, min(h + BIN, g.shape[1])
                share = g[:, lo:hi].sum(axis=1).mean() if hi > lo else 0.0
                rows.append({"lag": lag, "total": float(v.mean()), "se": float(se),
                             "next_bin": float(share)})  # fmt: skip
                print(f"    L {lag:4d} ({lag * meta['dt'] * 1e3:5.2f} ms)  pathwise "
                      f"{v.mean():+.4f} +- {se:.4f}   next {BIN} steps {share:+.4f}")  # fmt: skip
            res[case][c] = {"fd": fd_m, "rows": rows}
    path = lab.OUT / "spec_lag.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
