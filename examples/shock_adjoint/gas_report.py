"""Argon vs helium: does the alpha-sensitivity survive in the 2-D shock?

`examples/p2_replication` established the mechanism in 0-D homogeneous relaxation.
Anisotropy relaxes at a rate set by the second Legendre moment `<cos^2 chi>`, whose
alpha-derivative

    d/d(alpha) <cos^2 chi> = 8/(alpha+2)^2 - 4/(alpha+1)^2

vanishes at exactly `alpha = sqrt(2) = 1.41421`.  Argon's tabulated `alpha = 1.40` sits
1.005 % away, so a temperature-like objective has a structurally near-zero alpha
sensitivity *for argon specifically*.  Helium (1.26) is 12.6x better conditioned.

0-D confirmed it: helium resolved at 4.7 sigma where argon managed 0.3.  But the
verification target is the 2-D reflected shock, which adds transport, a specular wall and
a developing flow -- none of which the 0-D argument covers, and all of which add noise.
This module reports whether the 12.6x survives that.

    python3 -m examples.shock_adjoint.gas_report

Parses `scratch/adjoint/gas_alpha.log` (shock, from `scratch/adjoint/probe_gas_alpha.py`)
and `scratch/adjoint/p2_gases.log` (0-D, from `examples.p2_replication --part alpha`).
Like the other report modules here it **parses the logs** rather than restating their
numbers, so the logs stay the single source of truth.
"""

from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np

from jax_dsmc.species_data import ALPHA_STATIONARY, MONATOMIC
from jax_dsmc.viz import (
    INK_SECONDARY,
    REFERENCE,
    SERIES_ANALYTIC,
    SERIES_ERROR,
    SERIES_MEASURED,
    make_figure,
    savefig,
)

SCRATCH = "scratch/adjoint"
FIGURES = "examples/shock_adjoint/figures"

_F = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
_BY_ALPHA = {round(s.alpha, 2): s for s in MONATOMIC}
_BY_NAME = {s.name: s for s in MONATOMIC}


def d_legendre_d_alpha(alpha):
    """``d/d(alpha) <cos^2 chi>`` -- the analytic conditioning number.  Zero at sqrt(2)."""
    return 8.0 / (np.asarray(alpha) + 2.0) ** 2 - 4.0 / (np.asarray(alpha) + 1.0) ** 2


def parse_shock(path):
    """``scratch/adjoint/gas_alpha.log`` -> ``[{gas, alpha, N, a_max, ad, ad_se, ...}]``."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    rows = []
    for line in lines:
        m = re.match(
            rf"\s*([A-Za-z]+)\s+({_F})\s+(\d+)\s+(\d+)\s+({_F})\s+({_F})\s+({_F})\s+({_F})"
            rf"\s+({_F})\s+({_F})\s+({_F})x\s*$",
            line,
        )
        if m:
            g = m.groups()
            rows.append(
                {
                    "gas": g[0],
                    "alpha": float(g[1]),
                    "N": int(g[2]),
                    "a_max": int(g[3]),
                    "ad": float(g[4]),
                    "ad_se": float(g[5]),
                    "fd": float(g[6]),
                    "fd_se": float(g[7]),
                    "z": float(g[8]),
                    "snr": float(g[9]),
                    "cond": float(g[10]),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no gas rows parsed (is the run still going?)")
    return rows


def parse_zerod(path):
    """``scratch/adjoint/p2_gases.log`` -> ``{alpha_rounded: row}`` for the 0-D baseline."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    out = {}
    for line in lines:
        m = re.match(
            rf"\s*(\d+)\s+({_F})\s+(\d+)\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s*$",
            line,
        )
        if m:
            g = m.groups()
            out[round(float(g[1]), 2)] = {
                "alpha": float(g[1]),
                "N": int(g[2]),
                "ad": float(g[3]),
                "ad_se": float(g[4]),
                "fd": float(g[5]),
                "fd_se": float(g[6]),
                "z": float(g[7]),
                "snr": float(g[8]),
            }
    if not out:
        raise ValueError(f"{path}: no alpha rows parsed")
    return out


def print_comparison(shock, zerod):
    """The one table the whole gas question reduces to."""
    print("\n=== dT_y/d(alpha): 0-D relaxation vs the 2-D reflected shock ================")
    print("   `|AD|/SE` asks whether the gradient is resolvable from zero at all.  It is")
    print("   the number that was ~0 for argon and motivated the gas comparison; `|z|`")
    print("   asks the separate question of whether AD and FD agree.\n")
    print(
        f"   {'gas':>6} {'alpha':>6} {'cond':>6} | {'0-D N':>7} {'0-D AD':>11} {'+-SE':>9} "
        f"{'|AD|/SE':>8} | {'shock N':>8} {'shock AD':>11} {'+-SE':>9} {'|AD|/SE':>8} {'|z|':>5}"
    )
    for r in sorted(shock, key=lambda r: r["alpha"]):
        s = _BY_NAME.get(r["gas"])
        sym = s.symbol if s else r["gas"][:2]
        z = zerod.get(round(r["alpha"], 2))
        left = (
            f"{z['N']:>7} {z['ad']:>11.5f} {z['ad_se']:>9.5f} {z['snr']:>8.1f}" if z else " " * 38
        )
        print(
            f"   {sym:>6} {r['alpha']:>6.2f} {r['cond']:>5.1f}x | {left} | "
            f"{r['N']:>8} {r['ad']:>11.5f} {r['ad_se']:>9.5f} {r['snr']:>8.1f} {r['z']:>5.2f}"
        )

    pairs = [
        (r, zerod.get(round(r["alpha"], 2)))
        for r in shock
        if zerod.get(round(r["alpha"], 2)) is not None
    ]
    if len(pairs) >= 2:
        best = max(pairs, key=lambda p: abs(d_legendre_d_alpha(p[0]["alpha"])))
        worst = min(pairs, key=lambda p: abs(d_legendre_d_alpha(p[0]["alpha"])))
        cond = abs(d_legendre_d_alpha(best[0]["alpha"]) / d_legendre_d_alpha(worst[0]["alpha"]))
        g0 = best[1]["snr"] / worst[1]["snr"] if worst[1]["snr"] > 0 else math.inf
        g1 = best[0]["snr"] / worst[0]["snr"] if worst[0]["snr"] > 0 else math.inf
        print(
            f"\n   conditioning ratio {best[0]['gas']}/{worst[0]['gas']} = {cond:.1f}x analytic; "
            f"realised {g0:.1f}x in 0-D, {g1:.1f}x in the shock."
        )
    print("\n   The AD and 0-D gradient columns are in DIFFERENT units -- [P2]'s 0-D T_y is")
    print("   dimensionless, the shock's is in kelvin -- so only |AD|/SE compares across the")
    print("   two.  The shock's extra noise sources (transport, specular wall, developing flow)")
    print("   are absent from the 0-D argument, so the shock column is the one that decides")
    print("   whether alpha is a usable verification parameter for the milestone cases.")


def figure_resolvability(shock, zerod, path):
    """``|AD|/SE`` side by side: 0-D against the shock, per gas."""
    rows = sorted(shock, key=lambda r: -abs(d_legendre_d_alpha(r["alpha"])))
    labels, s0, s1 = [], [], []
    for r in rows:
        s = _BY_NAME.get(r["gas"])
        labels.append(f"{s.symbol if s else r['gas'][:2]}\n" + rf"$\alpha={r['alpha']:.2f}$")
        z = zerod.get(round(r["alpha"], 2))
        s0.append(z["snr"] if z else 0.0)
        s1.append(r["snr"])

    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    x = np.arange(len(rows))
    ax[0].bar(x - 0.19, s0, width=0.36, color=SERIES_ANALYTIC, label="0-D relaxation")
    ax[0].bar(x + 0.19, s1, width=0.36, color=SERIES_MEASURED, label="2-D reflected shock")
    ax[0].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[0].text(-0.44, 2.1, "resolvable above here", color=INK_SECONDARY, fontsize=8.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set(ylabel=r"$|\overline{AD}|\,/\,SE$")
    ax[0].set_title(r"is $\partial T_y/\partial\alpha$ resolvable from zero?", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    fine = np.linspace(1.15, 1.55, 300)
    ax[1].plot(
        fine,
        np.abs(d_legendre_d_alpha(fine)),
        "-",
        color=REFERENCE,
        linewidth=1.8,
        label=r"$|\partial_\alpha\langle\cos^2\chi\rangle|$",
    )
    ax[1].axvline(ALPHA_STATIONARY, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].text(
        ALPHA_STATIONARY + 0.006,
        ax[1].get_ylim()[1] * 0.9,
        r"$\sqrt{2}$",
        color=INK_SECONDARY,
        fontsize=9,
    )
    for r, colour in zip(rows, (SERIES_ANALYTIC, SERIES_MEASURED, SERIES_ERROR), strict=False):
        s = _BY_NAME.get(r["gas"])
        a = r["alpha"]
        ax[1].plot([a], [abs(d_legendre_d_alpha(a))], "o", color=colour, markersize=8)
        ax[1].annotate(
            s.symbol if s else r["gas"][:2],
            (a, abs(d_legendre_d_alpha(a))),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax[1].set(xlabel=r"VSS $\alpha$", ylabel=r"analytic conditioning")
    ax[1].set_title(r"why: the angular moment is stationary at $\alpha=\sqrt{2}$", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def figure_shock_gradient(shock, path):
    """AD against FD in the shock, and which of the two actually resolves the signal.

    The 0-D and shock gradients are **not** plotted against each other in physical
    units: 0-D `T_y` is dimensionless in [P2]'s scaling while the shock's is in kelvin,
    so only the dimensionless `|grad|/SE` is comparable across the two regimes.
    """
    rows = sorted(shock, key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    ad = np.array([r["ad"] for r in rows])
    ase = np.array([r["ad_se"] for r in rows])
    fd = np.array([r["fd"] for r in rows])
    fse = np.array([r["fd_se"] for r in rows])

    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    fine = np.linspace(1.15, 1.55, 300)
    fp_at = d_legendre_d_alpha(a)
    k = float(np.sum(fp_at * ad) / np.sum(fp_at * fp_at))
    ax[0].plot(
        fine,
        k * d_legendre_d_alpha(fine),
        "-",
        color=REFERENCE,
        linewidth=1.8,
        label=(
            rf"$k\,\partial_\alpha\langle\cos^2\chi\rangle$, $k={k:.3g}$ "
            rf"(1 constant, {len(rows)} points)"
        ),
    )
    ax[0].errorbar(
        a,
        ad,
        yerr=ase,
        fmt="o",
        color=SERIES_ANALYTIC,
        markersize=7,
        capsize=4,
        label="adjoint (AD)",
    )
    ax[0].errorbar(
        a,
        fd,
        yerr=fse,
        fmt="s",
        color=SERIES_MEASURED,
        markersize=6,
        capsize=4,
        label="finite difference",
    )
    ax[0].axhline(0.0, color=INK_SECONDARY, linewidth=1.0)
    ax[0].axvline(ALPHA_STATIONARY, color=REFERENCE, linestyle="--", linewidth=1.4)
    for r in rows:
        s = _BY_NAME.get(r["gas"])
        ax[0].annotate(
            s.symbol if s else r["gas"][:2],
            (r["alpha"], r["ad"]),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax[0].set(xlabel=r"VSS $\alpha$", ylabel=r"$\partial T_y/\partial\alpha$  (shock)")
    ax[0].set_title("2-D reflected shock: AD vs FD", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    labels = [(_BY_NAME[r["gas"]].symbol if r["gas"] in _BY_NAME else r["gas"][:2]) for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(
        x - 0.19,
        np.abs(ad) / ase,
        width=0.36,
        color=SERIES_ANALYTIC,
        label="adjoint (AD)",
    )
    ax[1].bar(
        x + 0.19,
        np.abs(fd) / fse,
        width=0.36,
        color=SERIES_MEASURED,
        label="finite difference",
    )
    ax[1].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].text(-0.44, 2.1, "resolvable above here", color=INK_SECONDARY, fontsize=8.5)
    for i in range(len(rows)):
        ax[1].text(
            x[i],
            max(abs(ad[i]) / ase[i], abs(fd[i]) / fse[i]) + 0.12,
            rf"$SE_{{FD}}/SE_{{AD}}={fse[i] / ase[i]:.1f}\times$",
            ha="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set(ylabel=r"$|\overline{grad}|\,/\,SE$")
    ax[1].set_title("helium: AD resolves it, FD does not.  argon: neither, correctly", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.shock_adjoint.gas_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--figures", default=FIGURES)
    ap.add_argument("--shock-log", default="gas_alpha.log")
    ap.add_argument("--zerod-log", default="p2_gases.log")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    shock = parse_shock(os.path.join(args.scratch, args.shock_log))
    zerod = parse_zerod(os.path.join(args.scratch, args.zerod_log))
    print_comparison(shock, zerod)
    if not args.no_plot:
        made = [
            figure_resolvability(shock, zerod, os.path.join(args.figures, "gas_resolvability.png")),
            figure_shock_gradient(shock, os.path.join(args.figures, "gas_shock_alpha.png")),
        ]
        print("\n=== figures ================================================================")
        for p in made:
            print(f"   {p}")


if __name__ == "__main__":
    main()
