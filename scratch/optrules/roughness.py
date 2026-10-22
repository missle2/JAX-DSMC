"""Roughness of the optrules shapes (2026-10-21): the total absolute turning of the zero
contour, int |kappa| ds / (2 pi) -- 1 for a convex body, each concave dent or wiggle adds --
per run along its iterates, plus the largest concave turning in one 0.25 m window.

    PYTHONPATH=. python3 scratch/optrules/roughness.py
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optimize as O  # noqa: E402
from noise import REGIMES, ROOT, RUNS, rows  # noqa: E402


def turning(seg, arc=0.05):
    """Total |turning| / 2 pi and the excess concave turning of a closed polyline, after
    resampling at ``arc`` spacing (the contour's own vertices are finer than its noise)."""
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    t = np.arange(0.0, s[-1], arc)
    p = np.c_[np.interp(t, s, seg[:, 0]), np.interp(t, s, seg[:, 1])]
    e = np.diff(np.vstack([p, p[:1]]), axis=0)
    ang = np.arctan2(e[:, 1], e[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))
    total = np.sum(turn)  # +-2 pi: the orientation
    concave = -np.sign(total) * turn
    return np.sum(np.abs(turn)) / (2 * np.pi), np.sum(np.clip(concave, 0, None)) / (2 * np.pi)


def main():
    for regime in REGIMES:
        cfg = dict(O.default_config(),
                   **json.load(open(f"{ROOT}/{RUNS[0][1].format(r=regime)}/history.json"))["config"])  # fmt: skip
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        print("==", REGIMES[regime], " columns: iterate: total turning/2pi (concave part)")
        for label, tag, _seeds, _c, _ls in RUNS:
            tag = tag.format(r=regime)
            try:
                n = len(rows(tag))
            except FileNotFoundError:
                continue
            out = []
            for k in sorted({0, n // 4, n // 2, 3 * n // 4, n}):
                c = np.load(f"{ROOT}/{tag}/coef_{k}.npy")
                segs = O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi,
                                          n=1500)  # fmt: skip
                seg = max(segs, key=len)
                tot, conc = turning(seg)
                out.append(f"{k}: {tot:.2f} ({conc:.2f}){'' if len(segs) == 1 else f' [{len(segs)} pieces]'}")
            print(f"  {label:16s} " + "   ".join(out))


if __name__ == "__main__":
    main()
