"""Where the optrules shapes are rough (2026-10-21): latest (or named) shapes of the runs with
the most turning (scratch/optrules/roughness.py), the concave stretches of the contour in red,
a zoom on the roughest 1 m window.

    PYTHONPATH=. python3 scratch/optrules/roughness_fig.py
"""

import dataclasses
import json
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optimize as O  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from noise import ROOT  # noqa: E402

CASES = (
    ("Kn 0.5 deck, S4 Adam @ 254", "circle_long_kn0.5_S4_adam_tan_sm8_free_g32", 254),
    ("Kn 0.5 deck, S4 TR reworked @ 153", "circle_long_kn0.5_S4_tr2_tan_sm8_free_g32_w500", 153),
    ("Kn 0.5 deck, S1 MMA @ 392", "circle_long_kn0.5_S1_mma_tan_sm8_free_g32_w500", 392),
    ("Kn 0.5 deck, S1 L-BFGS @ 582", "circle_long_kn0.5_S1_lbfgs_tan_sm8_free_g32_w500", 582),
    ("M 10, S4 Adam @ 128 (best)", "circle_long_kn0.5_M10_S4_adam_tan_sm8_free_g32", 128),
    ("M 10, S4 Adam @ 166", "circle_long_kn0.5_M10_S4_adam_tan_sm8_free_g32", 166),
    ("M 10, S1 TR @ 330", "circle_long_kn0.5_M10_S1_tr_tan_sm8_free_g32_w500", 330),
    ("M 10, S1 Adam @ 93", "circle_long_kn0.5_M10_S1_adam_tan_sm8_free_g32_w500", 93),
)


def concave_mask(seg):
    e = np.diff(np.vstack([seg, seg[:1]]), axis=0)
    ang = np.arctan2(e[:, 1], e[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))
    return -np.sign(turn.sum()) * turn > 1e-3


def main():
    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for i, (label, tag, k) in enumerate(CASES):
        with open(f"{ROOT}/{tag}/history.json") as fh:
            cfg = dict(O.default_config(), **json.load(fh)["config"])
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{ROOT}/{tag}/coef_{k}.npy")))
        segs = O.surface_polyline(ls, lo, hi, n=1500)
        ax, zx = axes[2 * (i // 4), i % 4], axes[2 * (i // 4) + 1, i % 4]
        worst, wpos = -1.0, None
        for seg in segs:
            mask = concave_mask(seg)
            for a in (ax, zx):
                a.plot(seg[:, 0], seg[:, 1], color="k", lw=0.8)
                a.plot(seg[mask, 0], seg[mask, 1], ".", color="#D55E00", ms=1.5)
            # the 1 m window with the most concave vertices
            for p in seg[mask][:: max(1, mask.sum() // 200)]:
                near = np.sum(np.all(np.abs(seg[mask] - p) < 0.5, axis=1))
                if near > worst:
                    worst, wpos = near, p
        ax.set_title(f"{label}  ({len(segs)} piece{'s' if len(segs) > 1 else ''})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 14)
        ax.set_ylim(1.5, 8.5)
        if wpos is not None:
            ax.add_patch(plt.Rectangle(wpos - 0.5, 1.0, 1.0, fill=False, ec="#0072B2", lw=1))
            zx.set_xlim(wpos[0] - 0.5, wpos[0] + 0.5)
            zx.set_ylim(wpos[1] - 0.5, wpos[1] + 0.5)
        zx.set_aspect("equal")
        zx.set_title("zoom: roughest 1 m (red = concave)", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("optc", "optrules_roughness.png"), dpi=110)


if __name__ == "__main__":
    main()
