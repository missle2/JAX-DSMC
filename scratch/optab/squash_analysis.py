"""The 2026-10-13 ``squashkn2`` runs (cubic-in-v backward fit, run to convergence) against the
quadratic-fit runs of 2026-10-11, Kn 0.5 at M 0.32 and M 10.

    python3 scratch/optab/squash_analysis.py

(a, b) ``s`` per iterate (the convergence stop marked); (c) the slope ratio
``(dD/ds) / (lam dV/ds)`` binned in ``s`` (1 = the optimum): how flat the objective is, and
how far the fit moves the crossing.  Writes ``figures/squash/squashkn2_cubic.png``.
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
FITS = {"cubicv": "cubic-in-v fit (2026-10-13)", "quad": "quadratic fit (2026-10-11)"}
# Okabe-Ito, by seeds per iterate
COLOURS = {2: "#E69F00", 4: "#0072B2", 8: "#D55E00"}
BINS = np.arange(0.30, 1.0001, 0.05)


def runs():
    out = {}
    for path in sorted(glob.glob(f"{RES}/ellipse_kn0.5_*cell-*/history.json")):
        tag = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            h = json.load(fh)
        fit = "cubicv" if "cubicv" in tag else "quad"
        regime = "M10" if "_M10_" in tag else "M0.32"
        out[tag] = (fit, regime, h)
    return out


def summary(h, tail=30):
    s = np.array([r["s"] for r in h["rows"]])
    return {
        "iterates": len(s),
        "converged": bool(h.get("converged")),
        "s_last": float(s[-1]),
        "s_tail_mean": float(s[-tail:].mean()),
        "s_tail_sd": float(s[-tail:].std(ddof=1)),
    }


def ratio_bins(h):
    s = np.array([r["s"] for r in h["rows"]])
    d = np.array([r["dD_ds"] for r in h["rows"]])
    lam_dv = d - np.array([r["J_prime"] for r in h["rows"]])
    q = d / lam_dv
    idx = np.digitize(s, BINS)
    x, y, e = [], [], []
    for b in range(1, len(BINS)):
        m = idx == b
        if m.sum() >= 3:
            x.append(0.5 * (BINS[b - 1] + BINS[b]))
            y.append(q[m].mean())
            e.append(q[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(x), np.array(y), np.array(e)


def main():
    rs = runs()
    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for tag, (fit, regime, h) in rs.items():
        S = h["config"]["seeds_per_iterate"]
        sm = summary(h)
        report[tag] = sm
        print(f"{tag:72s} {sm['iterates']:3d} it  converged {sm['converged']!s:5s}  s last "
              f"{sm['s_last']:.3f}  last 30: {sm['s_tail_mean']:.3f} +- sd {sm['s_tail_sd']:.3f}")
        ax = axes[0 if regime == "M0.32" else 1]
        ls = "-" if fit == "cubicv" else "--"
        s = [r["s"] for r in h["rows"]]
        ax.plot(s, ls, color=COLOURS[S], lw=1.4, label=f"{FITS[fit]}, S = {S}")
        if sm["converged"]:
            ax.plot(len(s) - 1, s[-1], "o", color=COLOURS[S], ms=7, mec="white")
        x, y, e = ratio_bins(h)
        axes[2].errorbar(x + (0.004 if regime == "M10" else 0), y, e, fmt=("o" if regime ==
                         "M0.32" else "s") + ls, color=COLOURS[S], ms=4, lw=1.1, capsize=2,
                         mfc=COLOURS[S] if fit == "cubicv" else "white",
                         label=f"{regime}, {fit}, S = {S}")
    for ax, regime in zip(axes[:2], ("Kn 0.5, M 0.32", "Kn 0.5, M 10"), strict=True):
        ax.axhline(0.35, color="0.5", ls=":", lw=1, label="free-molecular reference s* = 0.35")
        ax.set_title(f"({'ab'[axes.tolist().index(ax)]}) {regime}: s per iterate "
                     "(dot: converged)", fontsize=10, loc="left")
        ax.set_xlabel("iterate")
        ax.set_ylabel("y semi-axis factor s")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[2].axhline(1.0, color="0.4", lw=1)
    axes[2].set_title("(c) (dD/ds) / (λ dV/ds) binned in s (1 = optimum; ± se)", fontsize=10,
                      loc="left")
    axes[2].set_xlabel("y semi-axis factor s")
    axes[2].legend(fontsize=6.5, frameon=False, ncol=2)
    axes[2].grid(alpha=0.25, lw=0.5)
    fig.suptitle("squashkn2 (2026-10-13, rtx/h200): the cubic-in-v backward fit vs the "
                 "quadratic one, Adam lr 0.05 to convergence", fontsize=11)
    fig.tight_layout()
    path = figure_path("squash", "squashkn2_cubic.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)
    with open("scratch/optab/squash_analysis.json", "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
