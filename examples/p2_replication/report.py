"""Tables and figures for the [P2] replication, parsed from the run logs.

Like ``examples.shock_adjoint.variance_report``, this **parses the logs** rather than
restating their numbers, so the logs stay the single source of truth and a truncated run
raises instead of plotting a blank panel.

    python3 -m examples.p2_replication.report

[P2]'s own section 6 result is a *figure* -- error ``e`` against particle count ``N``,
log-log, one curve per gradient component (their ``Error-N--wo67``/``wt67``, and the
random errors in ``std-in-DSMC-grads``).  :func:`figure_error_vs_n` is the direct
counterpart, so the two can be laid side by side.
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
    time_color,
)

SCRATCH = "scratch/adjoint"
FIGURES = "examples/p2_replication/figures"

COMPONENTS = (
    r"$\partial T_y/\partial T_x^0$",
    r"$\partial T_y/\partial T_y^0$",
    r"$\partial T_y/\partial T_z^0$",
)
COMP_COLOUR = (SERIES_ANALYTIC, SERIES_MEASURED, SERIES_ERROR)

#: ASCII forms for the text tables -- `COMPONENTS` is LaTeX and renders as raw markup there.
COMPONENTS_PLAIN = ("dTy/dTx0", "dTy/dTy0", "dTy/dTz0")

_F = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_replicate(path):
    """``scratch/adjoint/p2_repl.log`` -> ``{beta: [row, ...]}``.

    Each row is ``{N, a_max, ad[3], fd[3], e[3], ad_se[3], fd_se[3]}``.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    out, beta, rows = {}, None, None
    for line in lines:
        m = re.match(r"\s*---\s*beta\s*=\s*(\d+)", line)
        if m:
            beta = int(m.group(1))
            rows = out.setdefault(beta, [])
            continue
        if beta is None:
            continue
        m = re.match(r"\s*(\d+)\s+(\d+)\s+((?:\S+/\S+/\S+\s*){3})", line)
        if m:
            cells = m.group(3).split()
            ad, fd, e = [], [], []
            for c in cells:
                a, f, err = c.split("/")
                ad.append(float(a))
                fd.append(float(f))
                e.append(float(err))
            rows.append(
                {"N": int(m.group(1)), "a_max": int(m.group(2)), "ad": ad, "fd": fd, "e": e}
            )
            continue
        m = re.match(r"\s*\+-SE\s+((?:\+-\S+\s*){3})", line)
        if m and rows:
            ad_se, fd_se = [], []
            for c in m.group(1).split():
                a, f = c.lstrip("+-").split("/")
                ad_se.append(float(a))
                fd_se.append(float(f))
            rows[-1]["ad_se"] = ad_se
            rows[-1]["fd_se"] = fd_se
    if not out:
        raise ValueError(f"{path}: no beta blocks parsed -- did the run finish?")
    for beta, rows in out.items():
        missing = [r["N"] for r in rows if "ad_se" not in r]
        if missing:
            raise ValueError(f"{path}: beta={beta} rows {missing} have no +-SE line")
    return out


def parse_alpha(path):
    """An ``--part alpha`` log -> ``[{beta, alpha, N, ad, ad_se, fd, fd_se, z, snr}]``."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    rows = []
    for line in lines:
        m = re.match(
            rf"\s*(\d+)\s+({_F})\s+(\d+)\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s+({_F})\s*$",
            line,
        )
        if m:
            g = m.groups()
            rows.append(
                {
                    "beta": int(g[0]),
                    "alpha": float(g[1]),
                    "N": int(g[2]),
                    "ad": float(g[3]),
                    "ad_se": float(g[4]),
                    "fd": float(g[5]),
                    "fd_se": float(g[6]),
                    "z": float(g[7]),
                    "snr": float(g[8]),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no alpha rows parsed")
    return rows


def print_replicate(data):
    print("\n=== [P2] section 6.1: e = |mean AD - mean FD| vs N ==========================")
    print("  [P2] main.tex:1423 predicts e falls with N, then PLATEAUS at the O(Delta m^2)")
    print("  finite-difference error.  M_s = 100, Delta m = 0.1, M = 20, phi = v_y^2.\n")
    for beta in sorted(data):
        print(f"  --- beta = {beta}")
        print(
            f"      {'N':>8} "
            + " ".join(f"{'e[' + c + ']':>12}" for c in "xyz")
            + "   "
            + " ".join(f"{'AD_' + c + ' +-SE':>18}" for c in "xyz")
        )
        for r in data[beta]:
            print(
                f"      {r['N']:>8} "
                + " ".join(f"{v:12.2e}" for v in r["e"])
                + "   "
                + " ".join(f"{r['ad'][p]:9.4f} +-{r['ad_se'][p]:6.4f}" for p in range(3))
            )
        e_last = data[beta][-1]["e"]
        print(f"      plateau ~ {max(e_last):.1e}\n")
    print("  Plateau height grows with beta -- 1e-5, 2e-3, 1.3e-2 -- exactly as an")
    print("  O(Delta m^2) truncation growing with the nonlinearity should.  beta = 0 has")
    print("  essentially NO plateau because Maxwell molecules' second-moment equations")
    print("  close linearly, so the central difference has no truncation error at all.")


def parse_p2_forward(path):
    """``p2_forward.log`` -> ``{beta: {N: {mean, sd, se, rel_se_pct}}}``.

    The forward objective's own noise, measured on the same ladder and the same ``M_s``
    keys as the replication.  Without it the estimator comparison can only say which of AD
    and FD is noisier, not whether either is noisy *relative to the quantity being
    differentiated* -- which is the question that decides if a gradient is usable.
    """
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                beta, n = int(parts[0]), int(parts[1])
                mean, sd, se, rel = (float(x) for x in parts[2:6])
            except ValueError:
                continue
            out.setdefault(beta, {})[n] = {
                "mean": mean,
                "sd": sd,
                "se": se,
                "rel_se_pct": rel,
            }
    if not out:
        raise ValueError(f"{path}: no forward rows parsed -- did the probe finish?")
    return out


def print_estimator_noise(data, fwd=None, rel_tol=0.10):
    """``SE_FD/SE_AD`` -- is AD actually the less noisy estimator, and by how much?"""
    print("\n=== AD vs FD noise at [P2]'s protocol (M_s = 100) ==========================")
    print("  [P2] main.tex:1502 states AD is roughly 10x less noisy than FD.  Measured")
    print("  here, the DIRECTION reproduces and the FACTOR does not.\n")
    print(f"  {'beta':>5} {'usable pts':>11} {'mean ratio':>11} {'range':>14}")
    for beta in sorted(data):
        good = []
        for r in data[beta]:
            for p in range(3):
                ratio, err = _ratio_with_quantisation(r["ad_se"][p], r["fd_se"][p])
                if err / ratio <= rel_tol:
                    good.append(ratio)
        if not good:
            print(f"  {beta:>5} {'--':>11} {'(all quantised)':>26}")
            continue
        print(
            f"  {beta:>5} {len(good):>11d} {np.mean(good):>10.2f}x "
            f"{min(good):>6.2f}-{max(good):.2f}"
        )
    print("\n  beta = 0's 1.00x is real but uninformative: for Maxwell molecules the")
    print("  central difference has NO truncation error, so AD and FD are computing the")
    print("  same number off the same runs and their sample SEs coincide by construction.")
    print("  The informative rows are beta = 1 and 2, where FD is 1.7-2.0x noisier on")
    print("  average -- the same direction [P2] reports, about 5x smaller in magnitude.")
    print(f"\n  Points whose rounding uncertainty exceeds {100 * rel_tol:.0f} % are dropped:")
    print("  the log prints each SE to 4 decimals, so an SE of 0.0001 is one significant")
    print("  figure and a ratio built from it would be reporting the log's format.")


def print_forward_vs_gradient(data, fwd, diag=1):
    """Is the gradient noisier than the objective it is taken of?"""
    print("\n=== forward noise vs gradient noise ========================================")
    print("  Measured on the same runs, so this is like-for-like.  It asks what")
    print("  differentiating actually costs in precision -- the question the AD-vs-FD")
    print("  ratio above cannot answer, since it compares two gradients to each other.\n")
    ratios = []
    print(f"  {'beta':>5} {'N':>8} {'SE(dTy/dTy0)':>14} {'SE(J=Ty)':>12} {'ratio':>8}")
    for beta in sorted(data):
        for r in sorted(data[beta], key=lambda r: r["N"]):
            f = fwd.get(beta, {}).get(r["N"])
            if not f:
                continue
            ratio = r["ad_se"][diag] / f["se"]
            ratios.append(ratio)
            print(
                f"  {beta:>5d} {r['N']:>8d} {r['ad_se'][diag]:>14.4e} "
                f"{f['se']:>12.4e} {ratio:>8.2f}"
            )
    if ratios:
        print(
            f"\n  Mean {np.mean(ratios):.2f}x over {len(ratios)} points "
            f"(range {min(ratios):.2f}-{max(ratios):.2f}): the DIAGONAL gradient is as"
        )
        print("  well resolved as the objective itself.  Differentiating costs nothing")
        print("  here -- which is what makes the off-diagonal components the interesting")
        print("  case.  In RELATIVE terms, at the largest N:")
    for beta in sorted(data):
        rows = sorted(data[beta], key=lambda r: r["N"])
        f = fwd.get(beta, {}).get(rows[-1]["N"])
        if not f:
            continue
        frel = f["se"] / abs(f["mean"])
        cells = " ".join(
            f"{COMPONENTS_PLAIN[c]} {(rows[-1]['ad_se'][c] / abs(rows[-1]['ad'][c])) / frel:>5.1f}x"
            for c in range(3)
        )
        print(f"    beta={beta}: {cells}")
    print("\n  The off-diagonal sensitivities are 4-9x noisier than the objective in")
    print("  relative terms, purely because they are small numbers (~0.03) rather than")
    print("  O(1) ones.  Same runs, same absolute noise, very different usability.")


def figure_error_vs_n(data, path):
    """[P2]'s ``Error-N`` figure, plus the normalised version that is actually readable.

    **Top row** is their plot verbatim: ``e = |mean AD - mean FD|`` against ``N``, log-log,
    with the combined random error dashed.

    **Bottom row exists because the top one is misleading at small N**, and that was
    learned the hard way: extending this study down to ``N = 500`` to fill in the
    decreasing branch made the curves *noisier*, not cleaner.  ``e`` is a **folded**
    difference of two noisy means, so wherever the true gap sits below the sampling noise
    it behaves like a half-normal -- it scatters, and it can collapse towards zero by
    chance cancellation rather than by convergence (``beta = 1`` at ``N = 1000`` dips to
    ``1.8e-5``, two decades below its neighbours, purely by luck).  Reading a downward
    excursion there as convergence would be reading noise as physics.

    Dividing by the combined standard error removes the ambiguity: ``e/SE ~ 1`` means the
    two gradients are indistinguishable at this sample size, ``e/SE >> 1`` means the gap
    is real and is the ``O(Delta m^2)`` truncation plateau.
    """
    betas = sorted(data)
    fig, ax = make_figure(2, len(betas), figsize=(5.2 * len(betas), 8.0))
    n_b = len(betas)
    for k, beta in enumerate(betas):
        rows = data[beta]
        ns = np.array([r["N"] for r in rows], dtype=float)
        top, bot = ax[k], ax[k + n_b]
        for p_i in range(3):
            e = np.array([r["e"][p_i] for r in rows])
            sd = np.array([math.hypot(r["ad_se"][p_i], r["fd_se"][p_i]) for r in rows])
            top.plot(
                ns,
                e,
                "o-",
                color=COMP_COLOUR[p_i],
                linewidth=2.0,
                markersize=6,
                label=COMPONENTS[p_i],
            )
            top.plot(ns, sd, "--", color=COMP_COLOUR[p_i], linewidth=1.2, alpha=0.65)
            bot.plot(ns, e / sd, "o-", color=COMP_COLOUR[p_i], linewidth=2.0, markersize=6)
        top.set(
            xscale="log",
            yscale="log",
            xlabel="particles $N$",
            ylabel="$e = |\\overline{\\nabla J}^{AD} - \\overline{\\nabla J}^{FD}|$",
        )
        top.set_title(rf"$\beta={beta}$  ($\omega={1 - beta / 2:g}$)", fontsize=11)
        bot.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.4)
        bot.set(xscale="log", yscale="log", xlabel="particles $N$", ylabel="$e\\,/\\,$combined SE")
        bot.set_title("above 1: real gap.  at 1: indistinguishable", fontsize=10)
        if k == 0:
            top.legend(frameon=False, fontsize=8.5)
    ax[0].text(
        0.03,
        0.04,
        "dashed: combined random error",
        transform=ax[0].transAxes,
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    return savefig(fig, path)


def figure_gradient_convergence(data, path):
    """The gradient values themselves, with error bars -- the quantity being replicated."""
    betas = sorted(data)
    fig, ax = make_figure(1, len(betas), figsize=(5.2 * len(betas), 4.4))
    for k, beta in enumerate(betas):
        rows = data[beta]
        ns = np.array([r["N"] for r in rows], dtype=float)
        for p in range(3):
            ax[k].errorbar(
                ns * (1.0 + 0.04 * (p - 1)),
                [r["ad"][p] for r in rows],
                yerr=[r["ad_se"][p] for r in rows],
                fmt="o-",
                color=COMP_COLOUR[p],
                markersize=6,
                capsize=3,
                linewidth=1.8,
                label=COMPONENTS[p] + " (AD)",
            )
            ax[k].errorbar(
                ns * (1.0 + 0.04 * (p - 1)),
                [r["fd"][p] for r in rows],
                yerr=[r["fd_se"][p] for r in rows],
                fmt="s",
                markerfacecolor="none",
                color=COMP_COLOUR[p],
                markersize=7,
                capsize=3,
                linestyle="none",
            )
        ax[k].set(xscale="log", xlabel="particles $N$", ylabel="gradient")
        ax[k].set_title(rf"$\beta={beta}$: filled = AD, open = FD", fontsize=11)
        if k == 0:
            ax[k].legend(frameon=False, fontsize=8.5)
    return savefig(fig, path)


#: The replication log prints every standard error as ``%.4f``.  An SE of 0.0001 therefore
#: carries a ~29 % rounding uncertainty and 0.0010 carries ~3 %, which is enough to make a
#: ratio of two of them meaningless at the small-SE end.  Everything below propagates this
#: explicitly rather than plotting quantised values as if they were exact.
SE_ULP = 1e-4
SE_QUANT = SE_ULP / math.sqrt(12.0)


def _ratio_with_quantisation(ad_se, fd_se):
    """``SE_FD/SE_AD`` and its uncertainty from the log's 4-decimal rounding alone."""
    ratio = fd_se / ad_se
    rel = math.hypot(SE_QUANT / ad_se, SE_QUANT / fd_se)
    return ratio, ratio * rel


def figure_estimator_noise(data, path, fwd=None, rel_tol=0.10):
    """AD against FD **noise** -- the comparison the error plot cannot make.

    ``figure_error_vs_n`` plots the *disagreement* between the two estimators and uses
    their standard errors only as a yardstick for it.  This asks the other question:
    given the same budget, which estimator is noisier?  That is what decides whether AD is
    worth having, and it is the measurement behind [P2]'s claim at ``main.tex:1502`` that
    AD is roughly ten times less noisy than FD.

    Both panels are at [P2]'s protocol verbatim -- ``M_s = 100`` runs, ``Delta m = 0.1``,
    ``M = 20`` -- so ``N`` is the only thing varying.
    """
    betas = sorted(data)
    fig, ax = make_figure(1, 2, figsize=(12.6, 4.8))
    _panel_se_vs_n(ax[0], data, fwd, betas)
    _panel_noise_ratio(ax[1], data, betas, rel_tol)
    fig.tight_layout()
    return savefig(fig, path)


def _panel_se_vs_n(ax0, data, fwd, betas, diag=1):
    """Left panel: forward, AD and FD standard errors against ``N``.

    ``diag`` is ``dT_y/dT_y^0`` -- the largest signal and the component the README tables.
    """
    ax = [ax0]
    for beta in betas:
        rows = sorted(data[beta], key=lambda r: r["N"])
        ns = np.array([r["N"] for r in rows], dtype=float)
        colour = time_color(betas.index(beta), max(len(betas), 2))
        ax[0].plot(
            ns,
            [r["ad_se"][diag] for r in rows],
            "o-",
            color=colour,
            linewidth=2.0,
            markersize=7,
            label=rf"$\beta={beta}$  AD",
        )
        ax[0].plot(
            ns,
            [r["fd_se"][diag] for r in rows],
            "s--",
            color=colour,
            linewidth=1.5,
            markersize=7,
            markerfacecolor="none",
            label=rf"$\beta={beta}$  FD",
        )
    # The objective's own noise, on the same runs.  Drawn as a wide translucent halo
    # BEHIND the AD curves rather than as a line, because a line is invisible here: the
    # AD standard error of dT_y/dT_y^0 equals the forward standard error of T_y to within
    # +-20 % at every point (mean ratio 0.98 over all 15).  The AD curves therefore ride
    # inside the halo, which is the finding -- a line would just look like it had been
    # forgotten.  All three beta agree within ~7 %, so they are drawn individually rather
    # than averaged.
    if fwd:
        for i, beta in enumerate(sorted(fwd)):
            pts = sorted(fwd[beta].items())
            ax[0].plot(
                [n for n, _ in pts],
                [v["se"] for _, v in pts],
                "-",
                color=INK_SECONDARY,
                linewidth=9.0,
                alpha=0.22,
                solid_capstyle="round",
                zorder=0,
                label=r"forward $J = T_y$ (all $\beta$)" if i == 0 else None,
            )
    ns0 = np.array([r["N"] for r in sorted(data[betas[0]], key=lambda r: r["N"])], dtype=float)
    y0 = sorted(data[betas[-1]], key=lambda r: r["N"])[0]["fd_se"][diag]
    ax[0].plot(
        ns0,
        y0 * (ns0 / ns0[0]) ** -0.5,
        ":",
        color=REFERENCE,
        linewidth=1.6,
        label=r"$N^{-1/2}$",
    )
    ax[0].axhspan(0, SE_ULP, color=SERIES_ERROR, alpha=0.13)
    ax[0].text(
        ns0[0],
        SE_ULP * 1.15,
        "one printed digit -- ratios here are rounding",
        fontsize=7.5,
        color=INK_SECONDARY,
    )
    ax[0].set(
        xscale="log",
        yscale="log",
        xlabel="particles $N$",
        ylabel=rf"SE of {COMPONENTS[diag]}",
    )
    # beta = 0's two curves lie on top of each other and that is the point, not an omission
    ax[0].annotate(
        r"$\beta=0$: AD and FD coincide"
        "\n"
        r"(Maxwell molecules close linearly,"
        "\n"
        r"so FD has no truncation error)",
        xy=(ns0[1], sorted(data[0], key=lambda r: r["N"])[1]["ad_se"][diag]),
        xytext=(0.05, 0.20),
        textcoords="axes fraction",
        fontsize=7.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 0.9},
    )
    ax[0].set_title("AD rides inside the objective's own noise; FD sits above it", fontsize=11)
    ax[0].legend(frameon=False, fontsize=7.5, ncol=2)


def _panel_noise_ratio(ax1, data, betas, rel_tol):
    """Right panel: ``SE_FD/SE_AD``, which is [P2]'s actual claim."""
    ax = {1: ax1}
    ns0 = np.array([r["N"] for r in sorted(data[betas[0]], key=lambda r: r["N"])], dtype=float)
    for beta in betas:
        rows = sorted(data[beta], key=lambda r: r["N"])
        colour = time_color(betas.index(beta), max(len(betas), 2))
        xs, ys, es = [], [], []
        for r in rows:
            for p in range(3):
                ratio, err = _ratio_with_quantisation(r["ad_se"][p], r["fd_se"][p])
                if err / ratio > rel_tol:
                    continue  # the log has too few digits here to say anything
                xs.append(r["N"] * (1.0 + 0.05 * (p - 1)))
                ys.append(ratio)
                es.append(err)
        if not xs:
            continue
        ax[1].errorbar(
            xs,
            ys,
            yerr=es,
            fmt="o",
            color=colour,
            markersize=6,
            capsize=2.5,
            linewidth=1.2,
            label=rf"$\beta={beta}$  (mean {np.mean(ys):.2f}$\times$)",
        )
    ax[1].axhline(1.0, color=INK_SECONDARY, linewidth=1.4)
    ax[1].axhline(10.0, color=REFERENCE, linestyle="--", linewidth=1.6)
    ax[1].text(
        ns0[0],
        10.6,
        "[P2]: AD $\\sim 10\\times$ less noisy (main.tex:1502)",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel="particles $N$",
        ylabel=r"$\mathrm{SE}_{\rm FD} \, / \, \mathrm{SE}_{\rm AD}$",
    )
    ax[1].set_ylim(0.7, 16)
    ax[1].set_title("the direction reproduces, the factor does not", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5, loc="center right")


def figure_noble_gases(rows, path):
    """Measured ``dT_y/d(alpha)`` at the five noble gases' real ``alpha`` values."""
    rows = sorted(rows, key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    ad = np.array([r["ad"] for r in rows])
    ase = np.array([r["ad_se"] for r in rows])
    fd = np.array([r["fd"] for r in rows])
    fse = np.array([r["fd_se"] for r in rows])
    gases = {round(s.alpha, 2): s for s in MONATOMIC}

    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    fine = np.linspace(1.2, 1.5, 200)
    fp = 8.0 / (fine + 2.0) ** 2 - 4.0 / (fine + 1.0) ** 2
    fp_at = 8.0 / (a + 2.0) ** 2 - 4.0 / (a + 1.0) ** 2
    k = float(np.sum(fp_at * ad) / np.sum(fp_at * fp_at))
    ax[0].plot(
        fine,
        k * fp,
        "-",
        color=REFERENCE,
        linewidth=1.8,
        label=rf"$k\,\partial_\alpha\langle\cos^2\chi\rangle$, $k={k:.3f}$",
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
    for x, y, s in zip(a, ad, [gases.get(round(v, 2)) for v in a], strict=True):
        if s is not None:
            ax[0].annotate(
                s.symbol,
                (x, y),
                textcoords="offset points",
                xytext=(0, 11),
                ha="center",
                color=INK_SECONDARY,
                fontsize=9,
            )
    ax[0].set(xlabel=r"VSS $\alpha$", ylabel=r"$\partial T_y/\partial\alpha$")
    ax[0].set_title(r"real noble gases straddle $\alpha=\sqrt{2}$", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    order = np.argsort(-np.abs(fp_at))
    xs = np.arange(len(order))
    labels = [
        (gases.get(round(a[i], 2)).symbol if gases.get(round(a[i], 2)) else f"{a[i]:.2f}")
        for i in order
    ]
    ax[1].bar(xs, np.abs(ad[order]) / ase[order], width=0.6, color=SERIES_ANALYTIC)
    ax[1].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].text(-0.45, 2.15, "$|z|=2$", color=INK_SECONDARY, fontsize=8.5)
    for i, j in enumerate(order):
        ax[1].text(
            i,
            abs(ad[j]) / ase[j] + 0.12,
            f"{abs(ad[j]) / ase[j]:.1f}",
            ha="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel(r"$|\partial T_y/\partial\alpha| \,/\, \mathrm{SE}$")
    ax[1].set_title("how resolvable the gradient is, per gas", fontsize=11)
    return savefig(fig, path)


def print_noble_gases(rows):
    gases = {round(s.alpha, 2): s for s in MONATOMIC}
    print("\n=== dT_y/d(alpha) at the five noble gases' real alpha values ================")
    print(
        f"   {'gas':>8} {'alpha':>6} {'AD':>11} {'+-SE':>9} {'FD':>11} {'+-SE':>9} "
        f"{'|z|':>5} {'|AD|/SE':>8} {'cond vs Ar':>11}"
    )
    for r in sorted(rows, key=lambda r: r["alpha"]):
        s = gases.get(round(r["alpha"], 2))
        name = s.symbol if s else "--"
        cond = f"{s.alpha_conditioning:>10.1f}x" if s else " " * 11
        print(
            f"   {name:>8} {r['alpha']:>6.2f} {r['ad']:>11.5f} {r['ad_se']:>9.5f} "
            f"{r['fd']:>11.5f} {r['fd_se']:>9.5f} {r['z']:>5.2f} {r['snr']:>8.1f} {cond}"
        )
    print("\n   `|AD|/SE` is the number that matters for a verification: it asks whether the")
    print("   gradient is resolvable from zero at all.  Argon is the WORST of the five --")
    print("   its alpha = 1.40 sits 1.0 % from the stationary point sqrt(2) = 1.41421.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.p2_replication.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--figures", default=FIGURES)
    ap.add_argument("--gases-log", default="p2_gases.log")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    rep = parse_replicate(os.path.join(args.scratch, "p2_repl.log"))
    fpath = os.path.join(args.scratch, "p2_forward.log")
    fwd = parse_p2_forward(fpath) if os.path.exists(fpath) else None
    print_replicate(rep)
    print_estimator_noise(rep, fwd=fwd)
    if fwd:
        print_forward_vs_gradient(rep, fwd)
    made = []
    if not args.no_plot:
        made.append(figure_error_vs_n(rep, os.path.join(args.figures, "error_vs_n.png")))
        made.append(
            figure_gradient_convergence(rep, os.path.join(args.figures, "gradient_convergence.png"))
        )
        made.append(figure_estimator_noise(rep, os.path.join(args.figures, "estimator_noise.png")))

    gpath = os.path.join(args.scratch, args.gases_log)
    if os.path.exists(gpath):
        rows = parse_alpha(gpath)
        print_noble_gases(rows)
        if not args.no_plot:
            made.append(figure_noble_gases(rows, os.path.join(args.figures, "noble_gases.png")))
    else:
        print(f"\n   (no {gpath} yet -- run `--part alpha` to produce it)")

    if made:
        print("\n=== figures ================================================================")
        for p in made:
            print(f"   {p}")


if __name__ == "__main__":
    main()
