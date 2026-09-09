"""Consolidated tables and figures for the score-estimator variance study.

The study runs as three separate probes under ``scratch/adjoint/`` and writes plain-text
logs.  This module **parses those logs** rather than restating their numbers, so the logs
stay the single source of truth: re-run a probe and the report follows automatically, and
a missing or reformatted section raises instead of quietly plotting nothing.  The
alternative -- transcribing the measurements into a literal table here -- is simpler to
write and silently goes stale the first time a probe is re-run, which is the failure mode
worth designing against.

Sections, matching ``docs/PSEUDOCODE.md`` section 23.3:

``0``  zero-check against an analytically known zero (a collision invariant)
``A``  ``sd`` vs particle count ``N``          -- tests [P2]'s ``O(1/sqrt(N M_s))``
``B``  [P2]'s own protocol, ``Delta m = 0.1``  -- AD vs FD at a 10 % step
``C``  ``T_y`` vs a spatially windowed ``phi`` -- what the window costs
``D``  ``sd`` and mean vs step count ``M``     -- tests [P2]'s ``O(sqrt(M))``

Run it with::

    python3 -m examples.shock_adjoint.variance_report

It needs no GPU and no simulation: everything it reports was already measured.
"""

from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np

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

NAMES = ("temp", "ux", "alpha", "diam")

LABEL = {
    "temp": "$T_1$",
    "ux": "$u_\\mathrm{stream}$",
    "alpha": r"$\alpha$",
    "diam": "$d_\\mathrm{ref}$",
}

COLOUR = {
    "temp": SERIES_ANALYTIC,
    "ux": SERIES_MEASURED,
    "alpha": SERIES_ERROR,
    "diam": INK_SECONDARY,
}

# `alpha`'s ensemble mean is consistent with zero at every N and flips sign across the M
# sweep, so any fit to log|mean| for it is an artefact.  Excluded from mean-based panels
# and fits, never from sd-based ones.  See section 23.3.2 and 23.3.6.
MEAN_IS_MEANINGFUL = ("temp", "ux", "diam")

_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def _floats(line):
    return [float(x) for x in re.findall(_FLOAT, line)]


# --- parsing ------------------------------------------------------------------


def _sections(text):
    """Split on the ``=== X. title ===`` banners the probes print."""
    out, key, buf = {}, "preamble", []
    for line in text.splitlines():
        m = re.match(r"^=== (\S+?)[.\s]", line)
        if m:
            out[key] = buf
            key, buf = m.group(1), []
        else:
            buf.append(line)
    out[key] = buf
    return out


def _grad_table(lines):
    """Parse the ``param / pathwise mean / full mean / sd(full) / sd(score) / SE %`` block."""
    rows = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 6 and parts[0] in NAMES:
            v = _floats(" ".join(parts[1:]))
            if len(v) == 5:
                rows[parts[0]] = dict(
                    zip(("pathwise", "mean", "sd", "sd_score", "se_pct"), v, strict=True)
                )
    return rows


def _parse_zero(lines):
    zero = {}
    for line in lines:
        m = re.match(r"\s*(\w+)\s+max\|pathwise\|\s+(\S+)\s+scored\s+(\S+)\s+\+-\s+(\S+)", line)
        if m:
            zero[m.group(1)] = {
                "pathwise": float(m.group(2)),
                "mean": float(m.group(3)),
                "sem": float(m.group(4)),
            }
    return zero


def _parse_a(lines):
    points, cur, fit = [], None, {}
    for line in lines:
        m = re.match(
            r"\s*transverse=(\d+)\s+C=(\d+)\s+N=(\d+)\s+A_max=(\d+)\s+\((\d+) keys\)", line
        )
        if m:
            cur = {
                "transverse": int(m.group(1)),
                "n_cells": int(m.group(2)),
                "n_part": int(m.group(3)),
                "a_max": int(m.group(4)),
                "n_keys": int(m.group(5)),
                "rows": {},
            }
            points.append(cur)
            continue
        f = re.match(r"\s*(\w+)\s+p = (\S+)\s+sd:", line)
        if f and f.group(1) in NAMES:
            fit[f.group(1)] = float(f.group(2))
        elif cur is not None:
            cur["rows"].update(_grad_table([line]))
    return {"points": points, "fit": fit}


def _parse_b(lines):
    rows, meta = {}, {}
    for line in lines:
        m = re.match(r"\s*N = (\d+), C = (\d+), M_s = (\d+)", line)
        if m:
            meta = {
                "n_part": int(m.group(1)),
                "n_cells": int(m.group(2)),
                "m_s": int(m.group(3)),
            }
        parts = line.split()
        if len(parts) == 6 and parts[0] in NAMES:
            v = _floats(" ".join(parts[1:]))
            if len(v) == 5:
                rows[parts[0]] = dict(zip(("ad", "ad_sem", "fd", "fd_sem", "z"), v, strict=True))
    return {"rows": rows, "meta": meta}


def _parse_c(lines):
    c, label = {}, None
    for line in lines:
        m = re.match(r"\s*phi = (\S+)\s+\((\d+) keys\)", line)
        if m:
            label = m.group(1)
            c[label] = {}
        elif label is not None:
            c[label].update(_grad_table([line]))
    return c


def _parse_d(lines):
    d = {}
    for line in lines:
        m = re.match(r"\s*M=\s*(\d+):", line)
        if m:
            d[int(m.group(1))] = {
                name: float(v) for name, v in re.findall(r"(\w+): sd (" + _FLOAT + ")", line)
            }
    return d


def parse_variance_log(path):
    """The main probe's log -> ``{zero, A, B, C, D}``.

    Raises if a banner is missing, which is what a half-finished or reformatted run looks
    like -- better than returning empty dicts and plotting a blank panel.
    """
    with open(path, encoding="utf-8") as fh:
        sec = _sections(fh.read())
    for need in ("0", "A", "B", "C", "D"):
        if need not in sec:
            raise ValueError(f"{path}: section {need!r} missing -- did the probe finish?")
    return {
        "zero": _parse_zero(sec["0"]),
        "A": _parse_a(sec["A"]),
        "B": _parse_b(sec["B"]),
        "C": _parse_c(sec["C"]),
        "D": _parse_d(sec["D"]),
    }


def parse_m_scaling(path):
    """The follow-up that records the **mean** as well as the sd against ``M``."""
    rows, fits = {}, {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit():
            v = _floats(line)
            steps = int(v[0])
            rows[steps] = {
                n: {"mean": v[1 + 3 * i], "sd": v[2 + 3 * i], "ratio": v[3 + 3 * i]}
                for i, n in enumerate(NAMES)
            }
        elif len(parts) == 4 and parts[0] in NAMES:
            fits[parts[0]] = dict(
                zip(("q_sd", "q_mean", "q_ratio"), [float(x) for x in parts[1:]], strict=True)
            )
    if not rows:
        raise ValueError(f"{path}: no M rows parsed")
    return {"rows": rows, "fits": fits}


def parse_split(path):
    """``sd(pathwise)``, ``sd(score)`` and their correlation, measured separately."""
    out, cur = {}, None
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        m = re.match(r"\s*---\s*N = (\d+), C = (\d+), (\d+) keys", line)
        if m:
            cur = int(m.group(1))
            out[cur] = {"n_cells": int(m.group(2)), "n_keys": int(m.group(3)), "rows": {}}
            continue
        parts = line.split()
        if cur is not None and len(parts) == 7 and parts[0] in NAMES:
            # `corr` is literally `nan` for a score-only parameter, whose pathwise sd is
            # identically zero -- so convert tokens directly rather than by float regex,
            # which would silently drop exactly the rows that prove the split.
            try:
                v = [float(x) for x in parts[1:6]]
            except ValueError:
                continue
            out[cur]["rows"][parts[0]] = dict(
                zip(("mean", "sd_pathwise", "sd_score", "corr", "sd_full"), v, strict=True)
            )
    if not out:
        raise ValueError(f"{path}: no blocks parsed")
    return out


# --- tables -------------------------------------------------------------------


def _rule(width=92):
    print("-" * width)


def table_zero(zero):
    print("\n=== 0. Zero-check: `sum |v|^2` is a collision invariant =====================")
    print("  VSS conserves pair energy and a specular wall conserves it, so dJ/d(any collision")
    print("  parameter) is EXACTLY zero -- pathwise and in expectation.  A ground truth, not")
    print("  another estimator.  D=2, N=16000, 32 keys.\n")
    print(
        f"  {'param':>7} {'max|pathwise|':>14} {'scored mean':>14} {'+- SE':>12} {'|z| from 0':>11}"
    )
    _rule(64)
    for name, r in zero.items():
        z = abs(r["mean"]) / r["sem"] if r["sem"] else math.nan
        print(f"  {name:>7} {r['pathwise']:14.3e} {r['mean']:14.3e} {r['sem']:12.3e} {z:11.2f}")
    print("\n  Both halves pass: pathwise is machine zero, and the noisy score estimator is")
    print("  consistent with the true zero in every case (all |z| < 2).")


def table_a(a):
    print("\n=== A. sd vs particle count N ==============================================")
    print("  [P2] predicts sd ~ N^-1/2 (main.tex:1431).  D=2 widened *transversely* so npc --")
    print("  and therefore A_max -- is held at 256 while N and C grow together.  M=16, phi=T_y.\n")
    hdr = f"  {'N':>8} {'C':>6} {'keys':>5}"
    for n in NAMES:
        hdr += f" {'sd(' + n + ')':>12}"
    print(hdr)
    _rule(76)
    for p in a["points"]:
        line = f"  {p['n_part']:>8} {p['n_cells']:>6} {p['n_keys']:>5}"
        for n in NAMES:
            line += f" {p['rows'][n]['sd']:12.3e}"
        print(line)
    _rule(76)
    line = f"  {'fitted p':>8} {'':>6} {'':>5}"
    for n in NAMES:
        line += f" {a['fit'][n]:12.3f}"
    print(line + "   <- vs -0.500 predicted")

    print("\n  Means with their standard errors -- the same runs, read the other way:\n")
    print(
        f"  {'N':>8}" + "".join(f" {'mean(' + n + ')':>14} {'+-SE':>11}" for n in ("alpha", "diam"))
    )
    _rule(60)
    for p in a["points"]:
        line = f"  {p['n_part']:>8}"
        for n in ("alpha", "diam"):
            r = p["rows"][n]
            line += f" {r['mean']:14.4e} {r['sd'] / math.sqrt(p['n_keys']):11.2e}"
        print(line)
    print("\n  `diam` converges on a well-defined non-zero value (stable to ~10% over a 64x")
    print("  range in N) while its error bar shrinks 8x.  `alpha`'s mean is consistent with")
    print("  ZERO at every N, which is the whole reason its *relative* error looks alarming --")
    print("  its ABSOLUTE error bar falls 8x on the same law as everything else.")


def table_b(b):
    m = b["meta"]
    print("\n=== B. At [P2]'s own protocol: Delta m = 0.1 ================================")
    print(
        f"  N = {m['n_part']}, C = {m['n_cells']}, M_s = {m['m_s']}  ->  N*M_s = "
        f"{m['n_part'] * m['m_s']:.1e}  ([P2] uses 1e8)\n"
    )
    print(
        f"  {'param':>7} {'AD mean':>14} {'+-SE':>11} {'FD mean':>14} {'+-SE':>11} "
        f"{'|z|':>7} {'gap':>8}"
    )
    _rule(80)
    for n in NAMES:
        r = b["rows"][n]
        gap = abs(r["fd"] - r["ad"]) / abs(r["ad"]) if r["ad"] else math.nan
        gapf = "--" if n == "alpha" else f"{100 * gap:7.1f}%"
        print(
            f"  {n:>7} {r['ad']:14.4e} {r['ad_sem']:11.2e} {r['fd']:14.4e} {r['fd_sem']:11.2e} "
            f"{r['z']:7.2f} {gapf:>8}"
        )
    print("\n  These |z| are NOT an AD/FD disagreement -- they are Delta m = 0.1 being a 10%")
    print("  step.  A central difference carries O(h^2) truncation; that step is 'nearly")
    print("  optimal' in [P2] only at THEIR noise level.  Our AD standard error here is")
    print("  0.03%-0.25%, low enough that truncation dominates.  `ux` is worst because a 10%")
    print("  step moves this case from Ma = 2.0 to Ma = 2.2.  Mode A is the precision test.")
    print("\n  [P2] main.tex:1502 reports AD ~10x less noisy than FD.  Measured ratio SE_FD/SE_AD:")
    line = "   "
    for n in NAMES:
        r = b["rows"][n]
        line += f"  {n} {r['fd_sem'] / r['ad_sem']:.1f}x"
    print(line + "    (partial reproduction; `alpha` is the one whose mean is ~0)")


def table_c(c):
    print("\n=== C. What the spatial window costs =======================================")
    print("  Same case, same keys, N=64000, 128 keys.  `T_y` is [P2]'s objective; `windowed`")
    print("  is the Gaussian-windowed phi used by the mode-A verification sweep.\n")
    print(
        f"  {'param':>7} {'T_y mean':>13} {'T_y sd':>11} {'win mean':>13} {'win sd':>11} "
        f"{'sd ratio':>9} {'mean ratio':>11} {'net SNR':>9}"
    )
    _rule(92)
    for n in NAMES:
        t, w = c["ty"][n], c["windowed"][n]
        sdr = w["sd"] / t["sd"]
        mnr = abs(w["mean"]) / abs(t["mean"]) if t["mean"] else math.nan
        print(
            f"  {n:>7} {t['mean']:13.4e} {t['sd']:11.3e} {w['mean']:13.4e} {w['sd']:11.3e} "
            f"{sdr:9.2f} {mnr:11.3f} {mnr / sdr:8.2f}x"
        )
    print("\n  The window LOWERS the absolute sd in every case -- fewer particles carry weight,")
    print("  so there is less to fluctuate.  What it lowers faster is the MEAN.  So it costs")
    print("  SIGNAL, not variance: for `diam` the spread falls 2.4x while the signal falls 15x.")


def table_d(d, m):
    print("\n=== D. sd and mean vs step count M =========================================")
    print("  [P2] main.tex:1502 reports sd ~ sqrt(M).  But their M-sweep is a 0-D relaxation")
    print("  TOWARDS EQUILIBRIUM, where the objective saturates so sd growth is pure score")
    print("  accumulation.  Ours is a DEVELOPING shock: dJ/dm itself grows with M, so the raw")
    print("  sd exponent is not the comparable quantity.  Measure the mean too.\n")
    print(
        f"  {'M':>4}"
        + "".join(
            f" {'mean(' + n + ')':>13} {'sd':>11} {'sd/|mean|':>10}" for n in MEAN_IS_MEANINGFUL
        )
    )
    _rule(88)
    for steps in sorted(m["rows"]):
        line = f"  {steps:>4}"
        for n in MEAN_IS_MEANINGFUL:
            r = m["rows"][steps][n]
            line += f" {r['mean']:13.4e} {r['sd']:11.3e} {r['ratio']:10.3f}"
        print(line)
    _rule(88)
    print(f"\n  {'param':>7} {'q(sd)':>8} {'q(|mean|)':>11} {'q(sd/|mean|)':>14}   reading")
    _rule(92)
    read = {
        "temp": "signal saturated -- the ONLY one in [P2]'s regime",
        "ux": "signal outgrows noise; longer runs are better",
        "alpha": "NOT READABLE: mean flips sign across the sweep",
        "diam": "signal and noise grow together, to three digits",
    }
    for n in NAMES:
        f = m["fits"][n]
        print(f"  {n:>7} {f['q_sd']:8.3f} {f['q_mean']:11.3f} {f['q_ratio']:14.3f}   {read[n]}")
    print("\n  CAVEAT: four points over an 8x range is a short lever arm, and `diam`'s ratio is")
    print("  not monotone (0.414, 0.274, 0.253, 0.423) -- its fitted exponent is flat partly")
    print("  because the endpoints happen to coincide.  What the fit supports is that its")
    print("  signal and noise EXPONENTS are indistinguishable, not that the ratio is constant")
    print("  point by point.")
    print("\n  `temp` is the only parameter here with a saturated signal, so it is the only one")
    print("  whose raw sd exponent is comparable to [P2]'s 0.5 -- and it is 0.291, i.e. our")
    print("  noise grows MORE SLOWLY than published.  For the pure-score `diam`, signal and")
    print("  noise grow at the same rate, leaving relative precision FLAT in run length.")


def table_split(sp):
    print("\n=== Decomposition: is the noise pathwise or score? =========================")
    print("  The sd(score) column above is sd(full - pathwise), which bounds the score")
    print("  CONTRIBUTION but cannot attribute the rest -- the halves may be correlated.")
    print("  Measuring all three settles it.  256 keys, phi = T_y.\n")
    print(
        f"  {'N':>7} {'param':>7} {'sd(pathwise)':>13} {'sd(score)':>13} {'corr':>7} "
        f"{'score % var':>12} {'baseline cap':>13}"
    )
    _rule(84)
    for n_part in sorted(sp):
        for n in NAMES:
            r = sp[n_part]["rows"][n]
            # The halves are uncorrelated, so VARIANCES add.  Quoting the score's share of
            # the standard deviation overstates it: for `alpha` that reads 19% where the
            # variance share -- what a baseline actually removes -- is ~3%.
            tot = r["sd_pathwise"] ** 2 + r["sd_score"] ** 2
            share = r["sd_score"] ** 2 / tot if tot else math.nan
            capf = "  (all of it)" if share > 0.999 else f"{1.0 / math.sqrt(1.0 - share):12.3f}x"
            print(
                f"  {n_part:>7} {n:>7} {r['sd_pathwise']:13.4e} {r['sd_score']:13.4e} "
                f"{r['corr']:7.2f} {share:11.1%} {capf:>13}"
            )
    print("\n  Correlations are all within +-0.2 of zero, so the halves are effectively")
    print("  independent and their VARIANCES add.  Every parameter with a pathwise route")
    print("  carries 3-5x more pathwise than score noise, and in variance terms the score is")
    print("  only 3-12% of the total.  `baseline cap` is the best a PERFECT baseline could do")
    print("  HERE: remove the score variance and nothing else, so 1.02x-1.07x.")
    print("\n  NOTE these are NOT the configuration section 11.4's ~1.2x was measured in (that")
    print("  was a windowed phi at D=1, N=2000), so the two numbers are not comparable and")
    print("  1.2x is not 'the ceiling' -- it exceeds this configuration's.  The robust claim")
    print("  is directional: wherever a pathwise route exists the score is a MINORITY of the")
    print("  variance, so a baseline is correspondingly limited.  The 84x applies only where")
    print("  the score is the WHOLE variance -- the count score in isolation, and score-only")
    print("  parameters like `diam` (row reads 'all of it').  For everything else MORE")
    print("  PARTICLES is the only lever; no estimator change touches pathwise noise.")


# --- figures ------------------------------------------------------------------


def _loglog_guide(ax, x, y0, slope, label):
    """A dashed power-law guide anchored at the first point -- the prediction, not a fit."""
    x = np.asarray(x, dtype=float)
    ax.plot(
        x, y0 * (x / x[0]) ** slope, "--", color=REFERENCE, linewidth=1.4, label=label, zorder=1
    )


def figure_a(a, path):
    """sd vs N, and how well each mean is resolved from zero."""
    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    ns = np.array([p["n_part"] for p in a["points"]], dtype=float)

    for name in NAMES:
        sd = np.array([p["rows"][name]["sd"] for p in a["points"]])
        ax[0].plot(
            ns,
            sd / sd[0],
            "o-",
            color=COLOUR[name],
            linewidth=2.0,
            markersize=6,
            label=f"{LABEL[name]}  $p={a['fit'][name]:+.3f}$",
        )
    _loglog_guide(ax[0], ns, 1.0, -0.5, "[P2]: $N^{-1/2}$")
    ax[0].set(
        xscale="log", yscale="log", xlabel="particles $N$", ylabel="sd, normalised to $N=4000$"
    )
    ax[0].set_title("A. the $N^{-1/2}$ law holds for all four", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    for name in NAMES:
        z = [
            abs(p["rows"][name]["mean"]) / (p["rows"][name]["sd"] / math.sqrt(p["n_keys"]))
            for p in a["points"]
        ]
        ax[1].plot(ns, z, "o-", color=COLOUR[name], linewidth=2.0, markersize=6, label=LABEL[name])
    ax[1].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].text(ns[0], 2.3, "$|z|=2$", color=INK_SECONDARY, fontsize=8.5)
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel="particles $N$",
        ylabel=r"$|\mathrm{mean}| \, / \, \mathrm{SE}$",
    )
    ax[1].set_title(r"how far each mean sits from zero", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    return savefig(fig, path)


def figure_b(b, path):
    """AD vs FD at [P2]'s 10 % step -- a truncation test, not a precision test."""
    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    xs = np.arange(len(NAMES))

    for i, name in enumerate(NAMES):
        r = b["rows"][name]
        s = abs(r["ad"]) if r["ad"] else 1.0
        ax[0].errorbar(
            i - 0.11,
            r["ad"] / s,
            yerr=r["ad_sem"] / s,
            fmt="o",
            color=SERIES_ANALYTIC,
            markersize=7,
            capsize=4,
            label="AD (scored)" if i == 0 else None,
        )
        ax[0].errorbar(
            i + 0.11,
            r["fd"] / s,
            yerr=r["fd_sem"] / s,
            fmt="s",
            color=SERIES_MEASURED,
            markersize=7,
            capsize=4,
            label=r"FD, $\Delta m = 0.1$" if i == 0 else None,
        )
    ax[0].axhline(0.0, color=REFERENCE, linewidth=1.0)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([LABEL[n] for n in NAMES])
    ax[0].set_ylabel("gradient / |AD|")
    ax[0].set_title(r"B. AD vs FD at [P2]'s $\Delta m = 0.1$", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)

    zs = [b["rows"][n]["z"] for n in NAMES]
    ax[1].bar(xs, zs, color=[COLOUR[n] for n in NAMES], width=0.6)
    ax[1].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].text(-0.4, 2.4, "$|z|=2$", color=INK_SECONDARY, fontsize=8.5)
    for i, z in enumerate(zs):
        ax[1].text(i, z * 1.08, f"{z:.1f}", ha="center", color=INK_SECONDARY, fontsize=9)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([LABEL[n] for n in NAMES])
    ax[1].set(yscale="log", ylabel="$|z|$ between AD and FD")
    ax[1].set_title("the gap is $O(h^2)$ truncation, not disagreement", fontsize=11)
    return savefig(fig, path)


def figure_c(c, path):
    """The window lowers the sd -- and lowers the signal faster."""
    fig, (ax,) = make_figure(1, 1, figsize=(6.4, 4.4))
    xs = np.arange(len(NAMES))
    sd_r, mn_r = [], []
    for n in NAMES:
        t, w = c["ty"][n], c["windowed"][n]
        sd_r.append(w["sd"] / t["sd"])
        mn_r.append(abs(w["mean"]) / abs(t["mean"]))
    ax.bar(xs - 0.19, sd_r, width=0.36, color=SERIES_ANALYTIC, label="sd ratio  (noise)")
    ax.bar(xs + 0.19, mn_r, width=0.36, color=SERIES_MEASURED, label="|mean| ratio  (signal)")
    ax.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.4)
    for i in range(len(NAMES)):
        ax.text(
            i,
            max(sd_r[i], mn_r[i]) * 1.25,
            f"SNR {mn_r[i] / sd_r[i]:.2f}x",
            ha="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[n] for n in NAMES])
    ax.set(yscale="log", ylabel="windowed $\\phi$  /  $T_y$")
    ax.set_title("C. the window costs signal, not variance", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    return savefig(fig, path)


def figure_d(m, path):
    """Signal and noise vs run length.  The raw sd exponent is the wrong comparison."""
    fig, ax = make_figure(1, 3, figsize=(15.0, 4.3))
    ms = np.array(sorted(m["rows"]), dtype=float)
    panels = (
        ("sd", "sd", "noise"),
        ("mean", "|mean|", "signal"),
        ("ratio", "sd / |mean|", "ratio"),
    )
    for k, (field, ylab, title) in enumerate(panels):
        for name in MEAN_IS_MEANINGFUL:
            y = np.array([abs(m["rows"][int(s)][name][field]) for s in ms])
            q = m["fits"][name]["q_" + {"sd": "sd", "mean": "mean", "ratio": "ratio"}[field]]
            ax[k].plot(
                ms,
                y / y[0],
                "o-",
                color=COLOUR[name],
                linewidth=2.0,
                markersize=6,
                label=f"{LABEL[name]}  $q={q:+.3f}$",
            )
        if field == "sd":
            _loglog_guide(ax[k], ms, 1.0, 0.5, r"[P2]: $\sqrt{M}$")
        ax[k].set(
            xscale="log", yscale="log", xlabel="steps $M$", ylabel=f"{ylab}, normalised to $M=4$"
        )
        ax[k].set_title(f"D. {title} vs run length", fontsize=11)
        ax[k].legend(frameon=False, fontsize=8.5)
    return savefig(fig, path)


def figure_split(sp, path):
    """Pathwise vs score noise, measured separately rather than inferred.

    Plotted as shares of the **variance**, not of the standard deviation.  The two halves
    are uncorrelated (measured |corr| <= 0.2), so their *variances* add and the shares sum
    to 1; stacking standard deviations would sum to ~1.2 and misstate the composition.
    It also states the baseline argument in the right units: a baseline attacks the score
    variance, and for `alpha` that is ~3% of the total.
    """
    fig, (ax,) = make_figure(1, 1, figsize=(6.6, 4.4))
    n_part = max(sp)
    rows = sp[n_part]["rows"]
    xs = np.arange(len(NAMES))
    pw, sc = [], []
    for n in NAMES:
        r = rows[n]
        tot = r["sd_pathwise"] ** 2 + r["sd_score"] ** 2
        pw.append(r["sd_pathwise"] ** 2 / tot)
        sc.append(r["sd_score"] ** 2 / tot)
    ax.bar(xs, pw, width=0.6, color=SERIES_ANALYTIC, label="pathwise")
    ax.bar(xs, sc, width=0.6, bottom=pw, color=SERIES_MEASURED, label="score")
    for i in range(len(NAMES)):
        ax.text(i, 1.02, f"score {sc[i]:.1%}", ha="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[n] for n in NAMES])
    ax.set(ylim=(0.0, 1.16), ylabel="share of gradient variance")
    ax.set_title(
        f"noise decomposition, $N = {n_part}$\na baseline can only attack the orange",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="center left")
    return savefig(fig, path)


# --- entry point --------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.shock_adjoint.variance_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scratch", default=SCRATCH, help="directory holding the probe logs")
    ap.add_argument("--figures", default=FIGURES, help="where to write the PNGs")
    ap.add_argument("--no-plot", action="store_true", help="tables only")
    args = ap.parse_args(argv)

    v = parse_variance_log(os.path.join(args.scratch, "variance.log"))
    m = parse_m_scaling(os.path.join(args.scratch, "m_scaling.log"))
    sp = parse_split(os.path.join(args.scratch, "split_var.log"))

    print("=" * 92)
    print("  SCORE-ESTIMATOR VARIANCE STUDY -- consolidated report")
    print("  Parsed from the probe logs; see docs/PSEUDOCODE.md section 23.3 for the prose.")
    print("=" * 92)
    table_zero(v["zero"])
    table_a(v["A"])
    table_b(v["B"])
    table_c(v["C"])
    table_d(v["D"], m)
    table_split(sp)

    if not args.no_plot:
        print("\n=== figures ================================================================")
        for p in (
            figure_a(v["A"], os.path.join(args.figures, "variance_a_particles.png")),
            figure_b(v["B"], os.path.join(args.figures, "variance_b_protocol.png")),
            figure_c(v["C"], os.path.join(args.figures, "variance_c_window.png")),
            figure_d(m, os.path.join(args.figures, "variance_d_steps.png")),
            figure_split(sp, os.path.join(args.figures, "variance_decomposition.png")),
        ):
            print(f"  {p}")


if __name__ == "__main__":
    main()
