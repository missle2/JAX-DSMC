"""Score the fits against the shift FD (2026-11-05): ``scratch/fitlab/shiftfd.py``'s forward
differences (the fit-free pathwise_x under each shift) against every fit's prediction
``sum_hits grad_x lambda . (w dir)`` on the same seeds' dumps -- the production variants (the
dumps' ``hit_gx``) and the lab's pooled references (``out/coef_<case><tag>.npz``, the hits a
reference has no group for keep cell_pool_plane's gradient).  Per objective, (control, mode)
and eps: the FD's seed mean +- 2 s.e., each fit's prediction and the paired (same seed)
difference.

    PYTHONPATH=. python3 scratch/fitlab/shiftcmp.py [--refs "" _k11 _cp_k5 _slab0.03]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

SHIFTS = Path("scratch/fitlab/cluster")
BUMPS = {"bump_nose": (np.pi,), "bump_side": (0.5 * np.pi, -0.5 * np.pi), "bump_base": (0.0,)}
FALLBACK = "cell_pool_plane"


def weight(theta, control):
    if control == "radius":
        return np.ones_like(theta)
    return sum(np.maximum(0.0, np.cos(theta - t0)) ** 4 for t0 in BUMPS[control])


def directions(h, control, mode):
    """``w dir`` at the dumped hits, as :func:`shiftfd._shifted_apply_faces` builds it."""
    rel = h["hit_x"] - lab.CENTRE
    n = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
    w = weight(np.arctan2(rel[:, 1], rel[:, 0]), control)
    vin_n = np.sum(h["hit_v_pre"][:, :2] * n, axis=1)
    incoming = vin_n < -1e-12
    full = (h["hit_v"][:, :2] - h["hit_v_pre"][:, :2]) / np.where(incoming, -vin_n, 1.0)[:, None]
    d = n if mode == "normal" else full
    return np.where(incoming[:, None], w[:, None] * d, 0.0)


def reference_gradients(case, tag, h, fallback):
    z = np.load(lab.OUT / f"coef_{case}{tag}.npz")
    coef = dict(zip(z["ids"].tolist(), z["coef"], strict=True))
    settings = dict(lab.SETTINGS)
    for k, v in json.loads((lab.OUT / f"ref_{case}{tag}.json").read_text())["settings"].items():
        settings[k] = tuple(v) if isinstance(v, list) else v
    g, _value, _gid, _gv = lab.model_gradient(h["hit_x"], h["hit_v"], h["hit_step"], coef,
                                              settings)  # fmt: skip
    return np.where(np.isnan(g), fallback, g)


def load_shifts():
    rows = {}
    for p in sorted(SHIFTS.glob("shift_s*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows[r["seed"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="*", default=["", "_k11", "_cp_k5", "_slab0.03"])
    args = ap.parse_args()
    shifts = load_shifts()
    if not shifts:
        raise SystemExit(f"no shift rows in {SHIFTS}")
    for case in ("drag", "heat"):
        files = dict(lab.seed_files(lab.CASES[case]))
        seeds = sorted(s for s in shifts if s in files)
        keys = sorted({(s["control"], s["mode"], s["eps"]) for s in shifts[seeds[0]]["shifts"]})
        fd = {k: [] for k in keys}
        pred = {}
        for seed in seeds:
            path = files[seed]
            meta = json.loads(path.with_suffix(".json").read_text())
            if meta["key"] != shifts[seed]["key"]:
                raise RuntimeError(f"seed {seed}: dump key {meta['key']} != shift key")
            h = lab.load(path, ("hit_step", "hit_x", "hit_v", "hit_v_pre", "hit_gx"))
            fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
            for tag in args.refs:
                if not (lab.OUT / f"coef_{case}{tag}.npz").exists():
                    continue
                fits[f"ref{tag}"] = reference_gradients(case, tag, h, fits[FALLBACK])
            for s in shifts[seed]["shifts"]:
                fd[(s["control"], s["mode"], s["eps"])].append(s["slope"][case])
            for control, mode in sorted({(k[0], k[1]) for k in keys}):
                dirs = directions(h, control, mode)
                for name, g in fits.items():
                    pred.setdefault((control, mode), {}).setdefault(name, []).append(
                        float(np.sum(g * dirs))
                    )
            print(f"  {case} seed {seed} done", flush=True)
        n = len(seeds)
        print(f"\n{case}: {n} seeds ({seeds}); FD slope mean +- 2 se, then per fit: prediction, "
              f"paired FD - prediction +- 2 se")  # fmt: skip
        for control, mode in sorted({(k[0], k[1]) for k in keys}):
            print(f"  {control} / {mode}")
            for eps in sorted({k[2] for k in keys if k[:2] == (control, mode)}):
                f = np.array(fd[(control, mode, eps)])
                se = 2 * f.std(ddof=1) / n**0.5 if n > 1 else np.nan
                print(f"    FD eps {eps:.0e}: {f.mean():+.4g} +- {se:.3g}")
            for name, p in pred[(control, mode)].items():
                p = np.array(p)
                parts = []
                for eps in sorted({k[2] for k in keys if k[:2] == (control, mode)}):
                    d = np.array(fd[(control, mode, eps)]) - p
                    se = 2 * d.std(ddof=1) / n**0.5 if n > 1 else np.nan
                    parts.append(f"{eps:.0e}: {d.mean():+.3g}+-{se:.2g}")
                print(f"      {name:24s} pred {p.mean():+.4g}   FD - pred  " + "  ".join(parts))


if __name__ == "__main__":
    main()
