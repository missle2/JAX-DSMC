"""Shapes of the optrules2 / optrules1seed runs (2026-10-21, partial): per run its best shape
(lowest 10-iterate mean of J for S = 1) filled and its latest outline, against the S = 4 Adam
reference's best; the Mach 10 row adds a zoom on the nose (where the iterate-68 trapped
particle crept along a concave stretch).

    PYTHONPATH=. python3 scratch/optrules/noise_shapes.py

Writes ``examples/mesh/figures/optc/optrules_noise_shapes.png``.
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
from noise import REGIMES, ROOT, RUNS, rows, smooth  # noqa: E402


def best_and_last(tag, seeds):
    h = rows(tag)
    j = np.array([r["objective"] for r in h])
    x, js = smooth(j) if seeds == 1 else (np.arange(len(j)), j)
    return int(x[int(np.argmin(js))]), len(h), float(js.min())


def main():
    shown = [r for r in RUNS if "10-19" not in r[0]]
    fig, axes = plt.subplots(2, len(shown) + 1, figsize=(3.0 * (len(shown) + 1), 6.4))
    for row, regime in enumerate(REGIMES):
        ref = rows(shown[0][1].format(r=regime))
        cfg = dict(O.default_config(), **json.load(open(
            f"{ROOT}/{shown[0][1].format(r=regime)}/history.json"))["config"])  # fmt: skip
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        j0 = ref[0]["objective"]

        def line(tag, k, ls0=ls0, lo=lo, hi=hi):
            c = np.load(f"{ROOT}/{tag}/coef_{k}.npy")
            return O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi)

        ref_tag = shown[0][1].format(r=regime)
        kb, _n, _jb = best_and_last(ref_tag, 4)
        ref_line = line(ref_tag, kb)
        for col, (label, tag, seeds, colour, _ls) in enumerate(shown):
            ax = axes[row, col]
            tag = tag.format(r=regime)
            try:
                kb, n, jb = best_and_last(tag, seeds)
            except FileNotFoundError:
                ax.set_axis_off()
                continue
            for seg in ref_line:
                ax.plot(seg[:, 0], seg[:, 1], color="#8a8a86", lw=0.8, ls="--")
            for seg in line(tag, kb):
                ax.fill(seg[:, 0], seg[:, 1], color=colour, alpha=0.3, lw=0)
                ax.plot(seg[:, 0], seg[:, 1], color=colour, lw=1.2)
            for seg in line(tag, n):
                ax.plot(seg[:, 0], seg[:, 1], color="k", lw=0.7)
            ax.set_title(f"{label}\nbest J/J0 {jb / j0:.3f} @ {kb}; latest = {n}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlim(-1.0, 14.0)
            ax.set_ylim(1.5, 8.5)
            ax.tick_params(labelsize=6)
        ax = axes[row, -1]
        for label, tag, seeds, colour, _ls in shown:
            tag = tag.format(r=regime)
            try:
                _kb, n, _jb = best_and_last(tag, seeds)
            except FileNotFoundError:
                continue
            for seg in line(tag, n):
                ax.plot(seg[:, 0], seg[:, 1], color=colour, lw=1.0, label=label)
                label = None
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(4.0, 6.0)
        ax.set_title("latest shapes: nose zoom", fontsize=8)
        ax.legend(fontsize=5)
        axes[row, 0].set_ylabel(REGIMES[regime], fontsize=9)
    fig.suptitle("filled: best shape; black: latest; grey dashed: S4 Adam's best "
                 "(flow from the left, area held)", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(figure_path("optc", "optrules_noise_shapes.png"), dpi=130)


if __name__ == "__main__":
    main()
