"""The 2-D sharp-start lsfilter runs (2026-10-28): M 10 drag from a diamond (fixed area, against
the circle start) and M 10 L / D from a flat plate (area >= 8 m^2, against the circle start at
0.5 A0).  Top: the objective (25-iterate running mean, raw faint); bottom: bodies at a few
iterates.  Writes ``examples/mesh/figures/optc/lsfilter/lsfilter_sharp.png``.

    PYTHONPATH=. python3 scratch/lsfilter/sharp.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "scratch/lsfilter")
import figures as F  # noqa: E402
import optimize as O  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from style import INK, MUTED  # noqa: E402

M = f"{F.C}_M10_S1_ox_adam"
CIRCLE, SHARP = "#8a8a8a", "#e34948"
PANELS = [
    ("M 10 drag at fixed area", f"{M}_hz0_tan_sm8_{F.W}", f"{M}_dmhz0_tan_sm8_{F.W}", "diamond",
     (0, 85, 140, 185)),
    ("M 10 L / D", f"{M}_ldhz0_floor_sm8_{F.W}", f"{M}_plhz0_floor_sm8_{F.W}", "plate",
     (0, 80, 160, 227)),
]  # fmt: skip


def raw(run):
    j = np.array([r["objective"] for r in run.rows])
    return -j if run.lift else j


def main():
    mosaic = [["c0"] * 4 + ["c1"] * 4, [f"b0{i}" for i in range(4)] + [f"b1{i}" for i in range(4)]]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16.0, 7.0), height_ratios=(1.4, 1.0))
    for col, (title, circle_tag, sharp_tag, start, ks) in enumerate(PANELS):
        circle, sharp = F.Run(circle_tag), F.Run(sharp_tag)
        ax = axes[f"c{col}"]
        for run, colour, name in ((circle, CIRCLE, "circle start"), (sharp, SHARP, start)):
            y = raw(run)
            ax.plot(y, color=colour, lw=0.6, alpha=0.35)
            ax.plot(F.running(y), color=colour, lw=2.0,
                    label=f"{name}: {F.running(y)[-1]:.3g} @ {len(y) - 1}")  # fmt: skip
        ax.set_title(title + ("" if sharp.lift else " (J, absolute)") + ": 25-iterate mean, raw "
                     "faint", loc="left", fontsize=10, color=INK)  # fmt: skip
        ax.set_xlabel("iterate", fontsize=9)
        if not sharp.lift:
            ax.set_yscale("log")
        ax.legend(fontsize=9, frameon=False)
        F._style(ax)
        grid, ls0 = O.initial_levelset(sharp.cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        for i, k in enumerate(ks):
            ax = axes[f"b{col}{i}"]
            for seg in sharp.body(k, ls0, lo, hi):
                ax.fill(seg[:, 0], seg[:, 1], color=SHARP, alpha=0.35, lw=0)
                ax.plot(seg[:, 0], seg[:, 1], color=SHARP, lw=1.2)
            for seg in circle.body(k, ls0, lo, hi):
                ax.plot(seg[:, 0], seg[:, 1], color=CIRCLE, lw=1.0)
            ax.set_xlim(-6.0, 11.0)
            ax.set_ylim(1.0, 9.0)
            ax.set_aspect("equal")
            ax.set_title(f"{start} (red) / circle (grey) @ {k}", fontsize=8, color=MUTED)
            ax.set_yticks([])
            ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("lsfilter sharp starts, Helmholtz + Adam, Kn 0.5 M 10 (flow ->)", fontsize=11,
                 color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc/lsfilter", "lsfilter_sharp.png")
    fig.savefig(path, dpi=100, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
