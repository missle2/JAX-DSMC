"""Every Kn 0.5 squash run's history (2026-10-15): the quadratic-fit runs (2026-10-11) and the
cubic-in-v runs (``squashkn2``, 2026-10-13), at M 0.32 (S = 4, 8) and M 10 (S = 2, 4).

    python3 scratch/optab/squash_histories.py

Rows: the regimes.  Columns: the objective ``J = D - lam V`` (drag at the area's fixed price), the
aspect ratio ``s``, the gradient ``J' = dD/ds - lam dV/ds`` the optimiser steps on (0 at the
optimum; band +-2 s.e. of ``dD/ds``).  A convergence stop is marked with a dot.  Writes
``figures/squash/squashkn2_histories.png``.
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402

RES = "examples/mesh/results/optimize_squash"
REGIMES = {"M0.32": "Kn 0.5, M 0.32 (subsonic)", "M10": "Kn 0.5, M 10"}
FIT_LABEL = {"quad": "quadratic fit", "cubicv": "cubic-in-v fit"}
# Okabe-Ito: the fit by hue (quadratic warm, cubic cool), the seed count by shade
COLOURS = {
    ("quad", "lo"): "#E69F00",
    ("quad", "hi"): "#D55E00",
    ("cubicv", "lo"): "#56B4E9",
    ("cubicv", "hi"): "#0072B2",
}
STYLE = {"quad": "--", "cubicv": "-"}


def runs():
    out = {k: [] for k in REGIMES}
    for path in sorted(glob.glob(f"{RES}/ellipse_kn0.5_*cell-*/history.json")):
        tag = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            h = json.load(fh)
        fit = "cubicv" if "cubicv" in tag else "quad"
        regime = "M10" if "_M10_" in tag else "M0.32"
        seeds = int(tag.split("_S")[-1].split("_")[0])
        out[regime].append((fit, seeds, h))
    for regime, rs in out.items():
        lo = min(s for _f, s, _h in rs)
        out[regime] = [
            (f, s, "lo" if s == lo else "hi", h)
            for f, s, h in sorted(rs, key=lambda t: (t[0] != "quad", t[1]))
        ]
    return out


def main():
    data = runs()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    for row, (regime, title) in enumerate(REGIMES.items()):
        for fit, seeds, shade, h in data[regime]:
            r = h["rows"]
            it = np.array([x["iterate"] for x in r])
            colour, ls = COLOURS[fit, shade], STYLE[fit]
            stop = " (converged)" if h.get("converged") else ""
            label = f"{FIT_LABEL[fit]}, {seeds} seeds per iterate: {len(r)} iterates{stop}"
            jp = np.array([x["J_prime"] for x in r])
            se = np.array([x["dD_ds_se"] for x in r])
            for ax, y in zip(
                axes[row], ([x["J"] for x in r], [x["s"] for x in r], jp), strict=True
            ):
                ax.plot(it, y, ls, color=colour, lw=1.3, label=label)
                if h.get("converged"):
                    ax.plot(it[-1], y[-1], "o", color=colour, ms=5)
            axes[row, 2].fill_between(it, jp - 2 * se, jp + 2 * se, color=colour, alpha=0.12, lw=0)
        axes[row, 2].axhline(0.0, color="0.4", lw=0.8)
        # a single-iterate spike would flatten the panel: clip to the bulk, label what is off it
        allj = np.concatenate([[x["J_prime"] for x in h["rows"]] for *_r, h in data[regime]])
        lim = 1.6 * np.percentile(np.abs(allj), 99)
        axes[row, 2].set_ylim(-lim, lim)
        for fit, seeds, shade, h in data[regime]:
            for x in h["rows"]:
                if abs(x["J_prime"]) > lim:
                    y = np.sign(x["J_prime"]) * 0.93 * lim
                    axes[row, 2].annotate(
                        f"{FIT_LABEL[fit]}, S {seeds}, iterate {x['iterate']}: "
                        f"J' = {x['J_prime']:.3g} (off scale)",
                        (x["iterate"], y),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=7,
                        va="center",
                        color="0.25",
                        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.6},
                    )
        heads = (
            f"{title}: objective J = drag - λ·volume",
            f"{title}: aspect ratio s (height / width)",
            f"{title}: gradient J' = dJ/ds (±2 s.e.)",
        )
        for ax, head in zip(axes[row], heads, strict=True):
            ax.set_title(head, fontsize=9.5, loc="left")
            ax.set_xlabel("optimisation iterate", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=8)
        axes[row, 1].legend(fontsize=7.5, frameon=False)
    fig.suptitle(
        "Squash (aspect-ratio) optimisation at Kn 0.5, every run: quadratic (dashed, 2026-10-11) "
        "vs cubic-in-v backward fit (solid, 2026-10-13), Adam from the circle (s = 1); J' = 0 at the optimum",
        fontsize=11,
    )
    fig.tight_layout()
    path = figure_path("squash", "squashkn2_histories.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    main()
