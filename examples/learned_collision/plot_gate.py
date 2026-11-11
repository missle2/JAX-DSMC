"""The Phase 1 gate figure: ``mu_eff / mu_CE`` and ``kappa_eff / kappa_CE`` against the cell
size for every operator, from the :mod:`.shear_wave` result files.

    PYTHONPATH=. python3 -m examples.learned_collision.plot_gate \
        [--shear shear_wave_gate.json ...] [--thermal thermal_wave_gate.json ...] [--out gate.png]

Rows of later files replace earlier rows of the same (operator, dx, ppc).
"""

from __future__ import annotations

import argparse
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.learned_collision.shear_wave import RESULTS

STYLE = {
    "ntc": ("NTC (uniform partners)", "#0072B2", "o"),
    "ntc_adjacent": ("NTC, x-adjacent partners", "#E69F00", "s"),
    "esfp": ("ES-FP, cell-mean target", "#009E73", "^"),
    "esfp_recon": ("ES-FP, reconstructed target", "#CC79A7", "D"),
}
"""Operator -> (label, colour, marker); Okabe-Ito hues in a fixed order."""
SHEAR = ("shear_wave_gate.json", "shear_wave_gate_recon.json", "shear_wave_gate_recon4.json",
         "shear_wave_gate_adjacent.json")  # fmt: skip
THERMAL = ("thermal_wave_gate.json",)
LOW_PPC = ("shear_wave_ppc10.json", "shear_wave_ppc10_ntc.json")


def load(names, ppc=50):
    """``{operator: {dx: (ratio, se)}}`` from result files (missing files are skipped)."""
    out = {}
    for name in names:
        path = RESULTS / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            if r["ppc"] != ppc:
                continue
            ratio = r.get("ratio", r.get("mu_ratio"))
            se = r.get("ratio_se", r.get("mu_ratio_se"))
            out.setdefault(r["operator"], {})[r["dx_mfp"]] = (ratio, se)
    return out


def panel(ax, rows, ylabel, theory, low=None):
    """One panel: each operator's ratio with its standard error, plus the small-cell theory and
    (hollow, dashed) the same operators at fewer particles per cell, ``low``."""
    dx = np.geomspace(0.2, 5.0, 100)
    if theory:
        ax.plot(dx, 1.0 + 16.0 / (45.0 * np.pi) * dx**2, color="#888888", lw=1.5, ls="--",
                label="Alexander et al. 1998, $1 + 0.113 (\\Delta x/\\lambda)^2$")  # fmt: skip
    ax.axhline(1.0, color="#bbbbbb", lw=1.0, zorder=0)
    for i, (op, (label, colour, marker)) in enumerate(STYLE.items()):
        if op not in rows:
            continue
        d = np.array(sorted(rows[op]))
        y = np.array([rows[op][x][0] for x in d])
        e = np.array([rows[op][x][1] for x in d])
        shift = 2.0 ** (0.06 * (i - 1.5))  # side by side, so the error bars stay readable
        ax.errorbar(d * shift, y, yerr=e, color=colour, marker=marker, ms=6, lw=2, capsize=3,
                    label=label)  # fmt: skip
        if low and op in low:
            d = np.array(sorted(low[op]))
            y = np.array([low[op][x][0] for x in d])
            e = np.array([low[op][x][1] for x in d])
            ax.errorbar(d * shift, y, yerr=e, color=colour, marker=marker, ms=6, lw=1.2,
                        ls="--", mfc="white", capsize=3)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
    ax.set_xlabel("cell size $\\Delta x / \\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.85, 2.3)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shear", nargs="+", default=list(SHEAR))
    ap.add_argument("--thermal", nargs="+", default=list(THERMAL))
    ap.add_argument("--ppc", type=int, default=50)
    ap.add_argument("--low", nargs="+", default=list(LOW_PPC))
    ap.add_argument("--low-ppc", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS / "gate.png"))
    args = ap.parse_args(argv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    low = load(args.low, args.low_ppc)
    panel(axes[0], load(args.shear, args.ppc), "effective / Chapman-Enskog", theory=True, low=low)
    axes[0].set_title(f"viscosity (shear-wave decay); hollow, dashed: {args.low_ppc} per cell")
    panel(axes[1], load(args.thermal, args.ppc), "", theory=True)
    axes[1].set_title("conductivity (entropy-wave decay)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"Coarse-cell transport error, {args.ppc} particles per cell, "
                 "dt = 0.05 $\\lambda / v_{th}$", fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
