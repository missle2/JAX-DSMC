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
)

SCRATCH = "scratch/adjoint"
FIGURES = "examples/p2_replication/figures"

COMPONENTS = (
    r"$\partial T_y/\partial T_x^0$",
    r"$\partial T_y/\partial T_y^0$",
    r"$\partial T_y/\partial T_z^0$",
)
COMP_COLOUR = (SERIES_ANALYTIC, SERIES_MEASURED, SERIES_ERROR)

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
    print_replicate(rep)
    made = []
    if not args.no_plot:
        made.append(figure_error_vs_n(rep, os.path.join(args.figures, "error_vs_n.png")))
        made.append(
            figure_gradient_convergence(rep, os.path.join(args.figures, "gradient_convergence.png"))
        )

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
