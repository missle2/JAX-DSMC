"""The 3-D Newtonian reinit / radius battery (scratch/reinit/battery3d.sh, 2026-10-26): per
run and every ``--every`` iterates, the objective (J / J0, or L / D), the body's piece count,
its thinnest feature in knots (drag: the nose radius 1 knot behind the tip; L / D: the median
half-thickness on the medial surface), the instability signature (one extra
``levelset.reinitialise`` pass's rms coefficient change near the surface, whether or not the
run reinitialises) and the surface |grad phi| (1 for a signed distance).

    PYTHONPATH=. python3 scratch/reinit/analyse3d.py [--every 25]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

L = O.L
ARMS = ("band_re1", "band_re0", "hz_re1", "hz_re0", "hz2_re0")


def thinnest(phi, h, lift):
    if lift:
        inner = -phi
        ridge = (inner > 0) & (inner == ndimage.maximum_filter(inner, size=3))
        return float(np.median(inner[ridge]) / h) if ridge.any() else float("nan")
    inside = np.argwhere(phi < 0)
    if not len(inside):
        return float("nan")
    i = inside[:, 0].min() + 1  # one knot behind the tip
    return float(np.sqrt((phi[i] < 0).sum() / np.pi))  # equivalent radius, knots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args()
    summary = {}
    for prob in ("drag", "ld"):
        for arm in ARMS:
            tag = f"n3d_{prob}_{arm}"
            out = Path(O.RESULTS) / tag
            if not (out / "history.json").exists():
                continue
            hist = json.loads((out / "history.json").read_text())
            rows, cfg = hist["rows"], dict(O.default_config(), **hist["config"])
            _g, ls0 = O.initial_levelset(cfg)
            h = float(ls0.h)
            lift = cfg["objective"] == "lift_drag"
            j = np.array([r["objective"] for r in rows])
            value = -j if lift else j / j[0]
            print(f"{tag}  ({len(rows)} iterates)")
            lines = []
            for k in sorted({*range(0, len(rows) + 1, args.every), len(rows)}):
                p = out / f"coef_{k}.npy"
                if not p.exists():
                    continue
                c = np.load(p)
                ls = dataclasses.replace(ls0, coef=jnp.asarray(c))
                phi = np.asarray(L.value(ls, L.knots(ls))).reshape(c.shape)
                pieces = int(ndimage.label(phi < 0)[1])
                near = np.abs(phi) < 0.3 * 2  # 3 knots at h = 0.2
                try:
                    r = np.asarray(L.reinitialise(ls).coef) - c
                    signature = float(np.sqrt(np.mean(r[near] ** 2)) / h)
                except ValueError:
                    signature = float("nan")
                v = float(value[min(k, len(value) - 1)])
                gn = rows[min(k, len(rows) - 1)].get("surface_grad_norm", float("nan"))
                lines.append((k, v, pieces, thinnest(phi, h, lift), signature, gn))
                print(f"  k {k:3d}  {'L/D' if lift else 'J/J0'} {v:.3f}  pieces {pieces}  "
                      f"thinnest {lines[-1][3]:.2f} knots  reinit-pass change "
                      f"{signature:.4f} h  |grad phi| {gn:.2f}", flush=True)  # fmt: skip
            summary[tag] = lines
    Path("scratch/reinit/analyse3d.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
