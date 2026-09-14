"""Tables and figures for the Couette studies, parsed from the run logs.

    python3 -m examples.couette.report

Like ``examples.shock_adjoint.variance_report`` and ``examples.p2_replication.report``,
this **parses the logs** rather than restating their numbers, so the logs stay the single
source of truth: a figure cannot disagree with the run that produced it, and a truncated
study raises instead of quietly plotting a short curve.

Writes ``examples/couette/RESULTS.md`` and the figures in ``figures/``.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from examples.couette import TEMP, slip_coefficient
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

RESULTS = "examples/couette/results"
FIGURES = "examples/couette/figures"

FIELDS = (
    ("case", str),
    ("dim", int),
    ("n_cells", int),
    ("npc", int),
    ("acc", float),
    ("eps_rel", float),
    ("specular", int),
    ("dx_over_mfp", float),
    ("knudsen", float),
    ("du_dy", float),
    ("du_dy_se", float),
    ("tau_bulk", float),
    ("mu_meas", float),
    ("mu_ce", float),
    ("ratio", float),
    ("ratio_se", float),
    ("seeds", int),
    ("warm", int),
    ("slip", float),
    ("slip_theory", float),
    ("temp_bulk", float),
    ("n_part", int),
    ("recrossed", int),
    ("clipped", int),
)


def parse(study, results=RESULTS):
    """``results/<study>.log`` -> list of row dicts.  Raises if the study is missing."""
    path = os.path.join(results, f"{study}.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python3 -m examples.couette --study {study}`")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            # the case label may contain spaces; it is the fixed-width first 22 columns
            label, rest = line[:22].strip(), line[22:].split()
            if len(rest) != len(FIELDS) - 1:
                raise ValueError(f"{path}: malformed row with {len(rest)} fields: {line!r}")
            row = {"case": label}
            for (name, cast), tok in zip(FIELDS[1:], rest, strict=True):
                row[name] = cast(tok)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_profile(study, label, results=RESULTS):
    path = os.path.join(results, f"{study}__{label.replace(' ', '_').replace('/', '')}.npz")
    return np.load(path) if os.path.exists(path) else None


def _assert_clean(rows, study):
    """A study with clipped attempts or stranded particles is not a measurement."""
    for r in rows:
        if r["clipped"]:
            raise ValueError(
                f"{study}/{r['case']}: {r['clipped']} clipped attempts -- A_max is "
                "undersized, so the collision rate and every transport number below it "
                "are wrong (gap U4's failure mode)"
            )
        if r["recrossed"]:
            raise ValueError(f"{study}/{r['case']}: {r['recrossed']} particles recrossed")


# --- tables ----------------------------------------------------------------


def dx_extrapolation(rows):
    """Fit ``ratio(dx)`` at fixed Kn and extrapolate to ``dx -> 0``.

    DSMC's cell-size error is expected to be second order, and it measures that way here:
    an ``O(dx^2)`` fit leaves a residual at the coarsest point several times smaller than
    an ``O(dx)`` fit does.  The extrapolated intercept is what the Knudsen study's points
    would read if they were resolution-converged too -- they are all taken at one
    ``dx/lambda``, so this offset shifts that study's intercept without touching its slope.
    """
    rows = sorted(rows, key=lambda r: r["dx_over_mfp"])
    h = np.array([r["dx_over_mfp"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    se = np.array([max(r["ratio_se"], 1e-6) for r in rows])
    if h.size < 3:
        raise ValueError(f"dx extrapolation needs >= 3 points, got {h.size}")
    out = {}
    for key, x in (("quad", h**2), ("lin", h)):
        design = np.vstack([np.ones_like(x), x]).T
        w = 1.0 / se
        beta = np.linalg.lstsq(design * w[:, None], ratio * w, rcond=None)[0]
        out[key] = (float(beta[0]), ratio - design @ beta)
    intercept, resid = out["quad"]
    gap = float(ratio[0] - ratio[1])  # finest minus next-finest; rows sorted by dx
    gap_se = float(math.hypot(se[0], se[1]))
    gate = next((r for r in rows if abs(r["dx_over_mfp"] - 0.30) < 0.02), rows[1])
    # Richardson order from the three measured values alone.  Fitting an intercept under
    # an assumed O(dx^2) and then "measuring" order 2 from the residuals would be nearly
    # circular; this uses only successive differences and assumes nothing.
    order = order_lo = order_hi = math.nan
    if h.size >= 3 and abs(h[1] / h[0] - 2.0) < 0.2 and abs(h[2] / h[1] - 2.0) < 0.2:
        d_lo, d_hi = ratio[1] - ratio[0], ratio[2] - ratio[1]
        if d_lo and d_hi and (d_hi / d_lo) > 0:
            q = d_hi / d_lo
            rel = math.hypot(
                math.hypot(se[1], se[2]) / abs(d_hi), math.hypot(se[0], se[1]) / abs(d_lo)
            )
            order = math.log2(q)
            order_lo = math.log2(max(q * (1.0 - rel), 1e-6))
            order_hi = math.log2(q * (1.0 + rel))
    return {
        "order": order,
        "order_lo": order_lo,
        "order_hi": order_hi,
        "intercept": intercept,
        "bias": intercept - gate["ratio"],
        "gate_dx": gate["dx_over_mfp"],
        "resid_coarse": float(resid[-1]),
        "resid_coarse_linear": float(out["lin"][1][-1]),
        "spread_pct": float(100 * (ratio.max() - ratio.min()) / ratio.mean()),
        "range": float(h.max() / h.min()),
        "finest_gap": gap,
        "finest_gap_se": gap_se,
        "finest_z": abs(gap / gap_se) if gap_se else math.inf,
        "rows": rows,
    }


def table_converge(rows):
    print("\n=== cell-size convergence ==============================================")
    print("  Kn is held at 1/30 throughout, so the GAP is fixed and only the cell size")
    print("  varies.  This asks one question: is the gate resolved in dx?\n")
    print(
        f"  {'case':<12} {'dx/mfp':>8} {'N':>9} {'seeds':>6} {'mu_meas':>12} "
        f"{'mu_CE':>12} {'ratio':>16}"
    )
    for r in sorted(rows, key=lambda r: -r["dx_over_mfp"]):
        print(
            f"  {r['case']:<12} {r['dx_over_mfp']:>8.3f} {r['n_part']:>9d} {r['seeds']:>6d} "
            f"{r['mu_meas']:>12.4e} {r['mu_ce']:>12.4e} {r['ratio']:>8.4f} +- "
            f"{r['ratio_se']:.4f}"
        )
    fit = dx_extrapolation(rows)
    print(
        f"\n  Spread {fit['spread_pct']:.1f} % over a {fit['range']:.0f}x range of cell size"
        " -- NOT flat."
    )
    print(
        f"  The two finest points differ by {fit['finest_gap']:+.4f} +- "
        f"{fit['finest_gap_se']:.4f} (|z| = {fit['finest_z']:.2f}): a real"
    )
    print("  resolution dependence, not scatter.  O(dx^2) describes it -- residual at the")
    print(
        f"  coarsest point {fit['resid_coarse']:+.4f} against "
        f"{fit['resid_coarse_linear']:+.4f} for O(dx) -- and"
    )
    print(f"  extrapolates to {fit['intercept']:.4f} as dx -> 0, so the gate's own")
    print(
        f"  dx/mfp = {fit['gate_dx']:.2f} sits {100 * fit['bias']:+.2f} % low from "
        "resolution alone."
    )
    if not math.isnan(fit["order"]):
        print(
            f"\n  Richardson order from the three values alone: p = {fit['order']:.2f} "
            f"(1-sigma [{fit['order_lo']:+.2f}, {fit['order_hi']:+.2f}])."
        )
        print("  Consistent with the expected 2, but only loosely: the coarsest point's")
        print("  2-seed error bar dominates, so the ORDER is far less well determined than")
        print("  the INTERCEPT -- which rests on the two fine points and is +-0.004.")
    print("\n  This CORRECTS an earlier reading of this study as flat, which came from")
    print("  single draws at a constant 4000-step warm-up.  It matters for the Knudsen")
    print("  study: every point there is at dx/mfp = 0.30 and carries this same offset,")
    print("  which biases that study's INTERCEPT but leaves its SLOPE untouched.")
    print(_ensemble_note(rows))


def table_walls(rows):
    print("\n=== wall treatments ====================================================")
    print("  Read the shear rate and the slip, NOT the ratio: at this resolution the")
    print("  ratio carries several per cent of Monte Carlo scatter, and for the")
    print("  specular row it divides two numbers that are both at noise level.\n")
    print(f"  {'wall':<22} {'du_s/dn':>12} {'slip':>8} {'Maxwell':>9} {'tau':>12}")
    for r in rows:
        print(
            f"  {r['case']:<22} {r['du_dy']:>12.4e} {r['slip']:>8.1f} "
            f"{r['slip_theory']:>9.1f} {r['tau_bulk']:>12.4e}"
        )
    diff = next(
        (r for r in rows if r["acc"] == 1.0 and not r["specular"] and r["eps_rel"] == 0.0), None
    )
    spec = next((r for r in rows if r["specular"]), None)
    if diff and spec and spec["du_dy"] != 0.0:
        print(
            f"\n  Specular drives {abs(diff['du_dy'] / spec['du_dy']):.0f}x less shear "
            "than diffuse: it exerts no tangential force at all."
        )
    for r in rows:
        if r["eps_rel"] > 0 and diff:
            err = abs(r["du_dy"] - diff["du_dy"]) / abs(diff["du_dy"]) * 100.0
            print(
                f"  Regularised eps={r['eps_rel']:.1f} dt reproduces the deterministic "
                f"shear rate to {err:.1f} %."
            )


def table_dims(rows):
    print("\n=== dimension-agnostic check ===========================================")
    print("  The gap is resolved identically in every case; the periodic directions add")
    print("  samples, not structure.  So these should agree to Monte Carlo error, and a")
    print("  disagreement would mean the wall physics is not dimension-agnostic.\n")
    print(
        f"  {'case':<8} {'N':>9} {'seeds':>6} {'du_s/dn':>12} {'+-SE':>11} "
        f"{'slip':>8} {'ratio':>8} {'recross':>8}"
    )
    for r in rows:
        print(
            f"  {r['case']:<8} {r['n_part']:>9d} {r['seeds']:>6d} {r['du_dy']:>12.4e} "
            f"{r['du_dy_se']:>11.3e} {r['slip']:>8.1f} {r['ratio']:>8.4f} {r['recrossed']:>8d}"
        )
    shear = np.array([r["du_dy"] for r in rows])
    print(f"\n  Shear rate spread across D: {100 * shear.std() / abs(shear.mean()):.1f} %.")
    print(_ensemble_note(rows))


def kn_extrapolation(rows):
    """Fit ``mu_meas/mu_CE = a + b*Kn`` and extrapolate to ``Kn -> 0``.

    Chapman-Enskog is a *continuum* limit, so comparing a single finite-Knudsen run
    against it is not the gate -- the gate is the intercept.  The uncertainty on that
    intercept is taken from the **residual scatter of the fit**, not from the per-point
    standard errors: with a handful of seeds those SEs are themselves so noisy that
    points land 5-9 sigma off a visually good line.  Using them to weight the fit would
    manufacture false precision, which is the exact failure this study exists to undo.
    """
    rows = sorted(rows, key=lambda r: r["knudsen"])
    kn = np.array([r["knudsen"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    if kn.size < 3:
        raise ValueError(f"Kn extrapolation needs >= 3 points, got {kn.size}")
    design = np.vstack([np.ones_like(kn), kn]).T
    beta, *_ = np.linalg.lstsq(design, ratio, rcond=None)
    resid = ratio - design @ beta
    dof = kn.size - 2
    cov = (resid**2).sum() / dof * np.linalg.inv(design.T @ design)
    return {
        "intercept": float(beta[0]),
        "intercept_se": float(np.sqrt(cov[0, 0])),
        "slope": float(beta[1]),
        "slope_se": float(np.sqrt(cov[1, 1])),
        "rms_resid": float(np.sqrt((resid**2).mean())),
        "dof": dof,
        "kn": kn,
        "ratio": ratio,
        "ratio_se": np.array([r["ratio_se"] for r in rows]),
        "rows": rows,
    }


def combined_gate(kn_rows, conv_rows):
    """The gate with BOTH discretisations extrapolated away.

    Chapman-Enskog is the continuum, resolution-converged limit, so a defensible
    comparison has to remove two finite parameters, not one:

    * finite Knudsen number -- removed by the Kn sweep's intercept;
    * finite cell size -- removed by the dx sweep's intercept.

    They compose cleanly *only* because the Kn sweep holds ``dx/lambda`` fixed at 0.30 for
    every point.  That makes the cell-size error a constant offset on that sweep: it moves
    the intercept and leaves the slope alone, so the two corrections are independent and
    add.  Had the Kn sweep varied ``dx/lambda`` as well, the two effects would be
    confounded and neither extrapolation would mean anything on its own.
    """
    kn = kn_extrapolation(kn_rows)
    dx = dx_extrapolation(conv_rows)
    # uncertainty on the dx correction: the two finest points bound it, and they are what
    # the O(dx^2) fit is anchored to
    bias_se = dx["finest_gap_se"]
    value = kn["intercept"] + dx["bias"]
    se = math.hypot(kn["intercept_se"], bias_se)
    return {
        "kn_intercept": kn["intercept"],
        "kn_intercept_se": kn["intercept_se"],
        "dx_bias": dx["bias"],
        "dx_bias_se": bias_se,
        "value": value,
        "se": se,
        "t": abs(value - 1.0) / se if se else math.inf,
    }


def _ensemble_note(rows):
    seeds = sorted({r["seeds"] for r in rows})
    warm = sorted({r["warm"] for r in rows})
    return (
        f"  Ensembles of {seeds[0] if len(seeds) == 1 else min(seeds)}+ seeds; warm-up "
        f"{warm[0] if len(warm) == 1 else f'{min(warm)}-{max(warm)}'} steps "
        "(3 momentum-diffusion times per case)."
    )


def table_gate(rows):
    print("\n=== the viscosity gate =================================================")
    print("  One configuration, run as an ensemble.  A single draw of this ratio is not")
    print("  a measurement: it carries several per cent of Monte Carlo scatter, and four")
    print("  successive explanations of an apparent deficit were built on single draws")
    print("  before an ensemble showed the quantity simply scatters.\n")
    print(f"  {'case':<14} {'N':>9} {'seeds':>6} {'mu_meas':>12} {'mu_CE':>12} {'ratio':>16}")
    for r in rows:
        print(
            f"  {r['case']:<14} {r['n_part']:>9d} {r['seeds']:>6d} {r['mu_meas']:>12.4e} "
            f"{r['mu_ce']:>12.4e} {r['ratio']:>8.4f} +- {r['ratio_se']:.4f}"
        )
    r = rows[0]
    dev, se = 100.0 * (r["ratio"] - 1.0), 100.0 * r["ratio_se"]
    z = abs(dev / se) if se > 0 else float("inf")
    print(f"\n  Deviation from Chapman-Enskog: {dev:+.2f} % +- {se:.2f} %  ->  |z| = {z:.2f}")
    print(f"  at Kn = {r['knudsen']:.4f}.  This is a FINITE-Knudsen point; the continuum")
    print("  gate is the Kn -> 0 intercept in the knudsen study, not this number.")
    print(_ensemble_note(rows))


def table_knudsen(rows):
    print("\n=== Chapman-Enskog as Kn -> 0 ==========================================")
    print("  dx/lambda is held at 0.30 in every row, so cell size is NOT the variable --")
    print("  only the gap in mean free paths is.  Warm-up is scaled per case as")
    print("  L^2/nu: widening the gap lengthens the time to steady state as L^2, so a")
    print("  constant step budget silently under-converges the wide cases and")
    print("  manufactures a trend.  An earlier sweep did exactly that.\n")
    fit = kn_extrapolation(rows)
    print(f"  {'case':<10} {'Kn':>8} {'gap/mfp':>8} {'warm':>8} {'seeds':>6} {'ratio':>16}")
    for r in fit["rows"]:
        print(
            f"  {r['case']:<10} {r['knudsen']:>8.4f} {1.0 / r['knudsen']:>8.1f} "
            f"{r['warm']:>8d} {r['seeds']:>6d} {r['ratio']:>8.4f} +- {r['ratio_se']:.4f}"
        )
    print(
        f"\n  Linear fit:  ratio = {fit['intercept']:.4f} +- {fit['intercept_se']:.4f}  "
        f"{fit['slope']:+.3f} +- {fit['slope_se']:.3f} * Kn"
    )
    dev = 100.0 * (fit["intercept"] - 1.0)
    t = abs(dev / (100.0 * fit["intercept_se"])) if fit["intercept_se"] > 0 else float("inf")
    print(
        f"  Extrapolated to Kn -> 0: {fit['intercept']:.4f}, i.e. {dev:+.2f} % from Chapman-Enskog"
    )
    print(
        f"  -> |t| = {t:.2f} on {fit['dof']} dof: "
        f"{'CONSISTENT' if t < 2.0 else 'INCONSISTENT'} with the continuum limit."
    )
    print("  The apparent viscosity falls with Kn, which is the sign first-order slip")
    print("  theory predicts -- the Knudsen layers eat into the sheared bulk.")
    print("\n  NOTE this intercept is still at dx/mfp = 0.30 and so carries the cell-size")
    print("  offset the convergence study measures.  See the combined gate below.")
    return fit


def table_combined(kn_rows, conv_rows):
    """The gate with both finite parameters extrapolated away."""
    c = combined_gate(kn_rows, conv_rows)
    print("\n=== THE GATE: Chapman-Enskog with both discretisations removed ============")
    print("  Chapman-Enskog is the continuum AND resolution-converged limit, so two")
    print("  finite parameters have to go, not one.  They add rather than confound only")
    print("  because the Kn sweep holds dx/mfp fixed at 0.30, making the cell-size error")
    print("  a constant offset on it -- shifting the intercept, not the slope.\n")
    print(
        f"    Kn -> 0 intercept (at dx/mfp = 0.30)   {c['kn_intercept']:.4f} "
        f"+- {c['kn_intercept_se']:.4f}"
    )
    print(
        f"    cell-size correction (dx -> 0)         {c['dx_bias']:+.4f} +- {c['dx_bias_se']:.4f}"
    )
    print(f"    {'-' * 62}")
    print(f"    mu_meas / mu_CE                        {c['value']:.4f} +- {c['se']:.4f}")
    print(
        f"\n  {100 * (c['value'] - 1.0):+.2f} % from Chapman-Enskog, |t| = {c['t']:.2f}: "
        f"{'CONSISTENT' if c['t'] < 2.0 else 'INCONSISTENT'}."
    )


# --- figures ---------------------------------------------------------------


def figure_profiles(study, rows, path, label_key="case"):
    """Velocity, shear stress and temperature across the gap, against the analytic forms."""
    fig, ax = make_figure(1, 3, figsize=(16.0, 4.4))
    plotted = 0
    for i, r in enumerate(rows):
        data = load_profile(study, r["case"])
        if data is None:
            continue
        y, u_s, tau, temp = data["y"], data["u_s"], data["tau"], data["temp"]
        length, u_wall = float(data["length"]), float(data["u_wall"])
        colour = time_color(i, max(len(rows), 2))
        yn = y / length
        ax[0].plot(yn, u_s, "-", color=colour, linewidth=1.6, label=r[label_key])
        ax[1].plot(yn, tau, "-", color=colour, linewidth=1.6)
        ax[2].plot(yn, temp, "-", color=colour, linewidth=1.6)
        plotted += 1
    if not plotted:
        raise ValueError(f"{study}: no saved profiles; re-run the study")

    # analytic slip solution for the reference case
    ref = rows[0]
    data = load_profile(study, ref["case"])
    length, u_wall, mfp = float(data["length"]), float(data["u_wall"]), float(data["mfp"])
    zeta = slip_coefficient(ref["acc"]) if not ref["specular"] else math.inf
    yn = np.linspace(0.0, 1.0, 200)
    if np.isfinite(zeta):
        # u(y) = u_wall (2y/L - 1) / (1 + 2 zeta lambda / L): linear with Maxwell slip
        u_lin = u_wall * (2.0 * yn - 1.0) / (1.0 + 2.0 * zeta * mfp / length)
        ax[0].plot(
            yn, u_lin, "--", color=REFERENCE, linewidth=1.8, label="continuum + Maxwell slip"
        )
    ax[0].axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax[0].set(xlabel="$y/L$", ylabel="$u_s$  [m/s]")
    ax[0].set_title("velocity profile", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xlabel="$y/L$", ylabel=r"$\tau_{ns}$  [Pa]")
    ax[1].set_title("shear stress (constant in the bulk)", fontsize=11)
    ax[2].axhline(TEMP, color=REFERENCE, linestyle="--", linewidth=1.4, label="wall $T$")
    ax[2].set(xlabel="$y/L$", ylabel="$T$  [K]")
    ax[2].set_title("temperature: viscous heating", fontsize=11)
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return savefig(fig, path)


def figure_convergence(rows, path):
    """``mu_meas/mu_CE`` against cell size, and the convergence it actually shows.

    Rebuilt after the ensemble re-run: the earlier version was drawn for a curve believed
    to be flat, with an ``O(h^2)`` guide anchored on the assumption that the error went to
    zero at ``dx -> 0``.  It does not -- the ratio converges to 0.9863 at this Knudsen
    number, not to 1 -- so the guide overshot the coarse point by 4x and the "under-
    resolved" annotation fell outside the axes once the y-range changed.
    """
    fit = dx_extrapolation(rows)
    rows = fit["rows"]
    h = np.array([r["dx_over_mfp"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    se = np.array([r["ratio_se"] for r in rows])

    fig, ax = make_figure(1, 2, figsize=(12.0, 4.6))

    grid = np.linspace(0.0, h.max() * 1.08, 120)
    ax[0].plot(
        grid,
        fit["intercept"] + (ratio[0] - fit["intercept"]) * (grid / h[0]) ** 2,
        "-",
        color=SERIES_ANALYTIC,
        linewidth=1.8,
        label=r"$O(\Delta x^2)$ fit",
    )
    ax[0].errorbar(
        h,
        ratio,
        yerr=se,
        fmt="o",
        color=SERIES_MEASURED,
        markersize=8,
        capsize=3,
        linewidth=1.4,
        label=r"DSMC (mean $\pm$ SE)",
    )
    ax[0].errorbar(
        [0.0],
        [fit["intercept"]],
        yerr=[fit["finest_gap_se"]],
        fmt="D",
        color=SERIES_ERROR,
        markersize=9,
        capsize=4,
        linewidth=1.6,
        label=rf"$\Delta x\!\to\!0$: {fit['intercept']:.4f}",
    )
    ax[0].axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.6, label="Chapman-Enskog")
    ax[0].axvspan(0.5, h.max() * 1.12, color=SERIES_ERROR, alpha=0.10)
    ax[0].annotate(
        "cell $>$ half a mfp:\nDSMC under-resolved",
        xy=(0.52, 0.04),
        xycoords=("data", "axes fraction"),
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax[0].set_xlim(-0.02, h.max() * 1.12)
    ax[0].set(xlabel=r"$\Delta x/\lambda$", ylabel=r"$\mu_{\rm meas}/\mu_{\rm CE}$")
    ax[0].set_title(r"at fixed $Kn=1/30$: converges to 0.986, not to 1", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")

    # Error measured against the EXTRAPOLATED limit, not against 1: the residual Knudsen
    # offset is not a discretisation error and including it would flatten the slope.
    err = np.abs(ratio - fit["intercept"])
    ax[1].loglog(h, err, "o", color=SERIES_MEASURED, markersize=9)
    ref = np.linspace(h.min() * 0.85, h.max() * 1.15, 40)
    ax[1].loglog(
        ref,
        err[0] * (ref / h[0]) ** 2,
        "--",
        color=REFERENCE,
        linewidth=1.6,
        label=r"$O(\Delta x^2)$",
    )
    for hi, ei, r in zip(h, err, rows, strict=True):
        ax[1].annotate(
            f"{r['n_cells']} cells",
            (hi, ei),
            textcoords="offset points",
            xytext=(7, -12),
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax[1].set(
        xlabel=r"$\Delta x/\lambda$",
        ylabel=r"$|\mu_{\rm meas}/\mu_{\rm CE} - $ limit$|$",
    )
    title = "error against the extrapolated limit"
    if not math.isnan(fit["order"]):
        # `\n` must stay a real newline, so the LaTeX half is a separate raw string
        title += "\n" + rf"Richardson $p={fit['order']:.2f}$ "
        title += rf"(1$\sigma$ [{fit['order_lo']:+.1f}, {fit['order_hi']:+.1f}])"
    ax[1].set_title(title, fontsize=10)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def figure_walls(rows, path):
    """Shear rate and slip across the wall treatments."""
    labels = [r["case"] for r in rows]
    x = np.arange(len(rows))
    shear = np.array([abs(r["du_dy"]) for r in rows])
    slip = np.array([r["slip"] for r in rows])
    theory = np.array([r["slip_theory"] for r in rows])

    fig, ax = make_figure(1, 2, figsize=(12.4, 4.6))
    ax[0].bar(x, shear, width=0.6, color=SERIES_MEASURED)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax[0].set(ylabel=r"$|du_s/dn|$  [1/s]")
    ax[0].set_title("momentum transfer by wall type (log scale)", fontsize=11)

    ax[1].bar(x - 0.19, slip, width=0.36, color=SERIES_MEASURED, label="measured")
    finite = np.isfinite(theory) & (theory < 1e4)
    ax[1].bar(
        x[finite] + 0.19,
        theory[finite],
        width=0.36,
        color=SERIES_ANALYTIC,
        label=r"Maxwell $\zeta\lambda\,du/dn$",
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax[1].set(ylabel="wall slip  [m/s]")
    ax[1].set_title("slip velocity vs first-order theory", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def figure_knudsen(rows, path):
    """The continuum gate: ``mu_meas/mu_CE`` against Kn, extrapolated to Kn -> 0."""
    fit = kn_extrapolation(rows)
    kn, ratio, se = fit["kn"], fit["ratio"], fit["ratio_se"]

    fig, ax = make_figure(1, 2, figsize=(12.0, 4.5))

    grid = np.linspace(0.0, kn.max() * 1.08, 100)
    line = fit["intercept"] + fit["slope"] * grid
    band = fit["intercept_se"] + abs(fit["slope_se"] * grid)
    ax[0].fill_between(grid, line - band, line + band, color=SERIES_ANALYTIC, alpha=0.18)
    ax[0].plot(grid, line, "-", color=SERIES_ANALYTIC, linewidth=1.8, label="linear fit")
    ax[0].errorbar(
        kn,
        ratio,
        yerr=se,
        fmt="o",
        color=SERIES_MEASURED,
        markersize=8,
        capsize=3,
        linewidth=1.4,
        label="DSMC (mean $\\pm$ SE)",
    )
    ax[0].errorbar(
        [0.0],
        [fit["intercept"]],
        yerr=[fit["intercept_se"]],
        fmt="D",
        color=SERIES_ERROR,
        markersize=9,
        capsize=4,
        linewidth=1.6,
        label=rf"$Kn\!\to\!0$: {fit['intercept']:.4f}$\pm${fit['intercept_se']:.4f}",
    )
    ax[0].axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.6, label="Chapman-Enskog")
    ax[0].set(xlabel=r"$Kn = \lambda/L$", ylabel=r"$\mu_{\rm meas}/\mu_{\rm CE}$")
    ax[0].set_xlim(left=-0.004)
    ax[0].set_title("the gate is the intercept, not any single point", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")

    # The per-point SEs are shown above but deliberately NOT used to weight the fit:
    # this panel is why.  With too few seeds the SE is itself a very noisy estimate --
    # the 2-seed version of this same sweep put points at 5.6 and 9.2 "sigma" off a
    # visually excellent line, purely because two draws had landed close together.  The
    # title reports what the current seed count actually delivers rather than asserting
    # an outcome, so this panel stays a check and does not become another stale claim.
    resid = ratio - (fit["intercept"] + fit["slope"] * kn)
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = np.where(se > 0, resid / se, np.nan)
    ax[1].axhspan(-1.0, 1.0, color=SERIES_ANALYTIC, alpha=0.15)
    ax[1].axhline(0.0, color=INK_SECONDARY, linewidth=0.9)
    ax[1].plot(kn, pulls, "o", color=SERIES_ERROR, markersize=9)
    for k, pl, r in zip(kn, pulls, fit["rows"], strict=True):
        ax[1].annotate(
            f"{r['seeds']} seeds",
            (k, pl),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax[1].set(xlabel=r"$Kn = \lambda/L$", ylabel=r"fit residual / quoted SE")
    worst = float(np.nanmax(np.abs(pulls)))
    verdict = (
        "quoted SEs are consistent with the fit"
        if worst < 3.0
        else "quoted SEs are UNRELIABLE at this seed count"
    )
    ax[1].set_title(
        f"{verdict}: worst pull {worst:.1f}$\\sigma$\n"
        f"(rms residual {fit['rms_resid']:.4f} in ratio units)",
        fontsize=10,
    )
    fig.tight_layout()
    return savefig(fig, path)


def figure_dimensions(rows, path):
    """The same wall physics in 1, 2 and 3 spatial dimensions."""
    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    for i, r in enumerate(rows):
        data = load_profile("dims", r["case"])
        if data is None:
            continue
        colour = time_color(i, max(len(rows), 2))
        ax[0].plot(
            data["y"] / float(data["length"]),
            data["u_s"],
            "-",
            color=colour,
            linewidth=1.8,
            label=f"{r['case']}  (N={r['n_part']:,})",
        )
    ax[0].axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax[0].set(xlabel="$y/L$", ylabel="$u_s$  [m/s]")
    ax[0].set_title("velocity profile is dimension-agnostic", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    x = np.arange(len(rows))
    shear = np.array([abs(r["du_dy"]) for r in rows])
    ax[1].bar(x, shear, width=0.55, color=SERIES_MEASURED)
    ax[1].axhline(
        float(shear.mean()),
        color=REFERENCE,
        linestyle="--",
        linewidth=1.6,
        label=f"mean {shear.mean():.3e}",
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["case"] for r in rows])
    ax[1].set(ylabel=r"$|du_s/dn|$  [1/s]")
    ax[1].set_title(f"spread {100 * shear.std() / shear.mean():.1f} %", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return savefig(fig, path)


def write_results_md(available, made, path="examples/couette/RESULTS.md"):
    """A single page of results, regenerated from the logs."""
    lines = [
        "# Couette results",
        "",
        "Generated by `python3 -m examples.couette.report`, which **parses**",
        "`results/*.log`. Do not edit by hand -- re-run the study instead.",
        "",
        "Every ratio below is an **ensemble mean over independent seeds, with its",
        "standard error**. `mu_meas/mu_CE` carries several per cent of Monte Carlo",
        "scatter at any affordable particle count, so a single draw of it is not a",
        "measurement -- and warm-up is set per case as a multiple of the momentum",
        "diffusion time `L^2/nu`, never as a constant step count.",
        "",
    ]
    if "knudsen" in available and "converge" in available:
        c = combined_gate(available["knudsen"], available["converge"])
        lines += [
            "## THE GATE: Chapman-Enskog with both discretisations removed",
            "",
            "Chapman-Enskog is the continuum *and* resolution-converged limit, so two",
            "finite parameters have to be extrapolated away, not one. They add rather than",
            "confound only because the Knudsen sweep holds `dx/lambda` fixed at 0.30,",
            "which makes the cell-size error a constant offset on it -- shifting that",
            "sweep's intercept and leaving its slope alone.",
            "",
            "| term | value |",
            "|---|---|",
            f"| `Kn -> 0` intercept (at `dx/lambda = 0.30`) | {c['kn_intercept']:.4f} ± "
            f"{c['kn_intercept_se']:.4f} |",
            f"| cell-size correction (`dx -> 0`) | {c['dx_bias']:+.4f} ± {c['dx_bias_se']:.4f} |",
            f"| **`mu_meas/mu_CE`** | **{c['value']:.4f} ± {c['se']:.4f}** |",
            "",
            f"**{100 * (c['value'] - 1.0):+.2f} % from Chapman-Enskog, `|t| = {c['t']:.2f}`:"
            f" {'consistent' if c['t'] < 2.0 else 'INCONSISTENT'}.**",
            "",
        ]
    if "knudsen" in available:
        fit = kn_extrapolation(available["knudsen"])
        dev = 100.0 * (fit["intercept"] - 1.0)
        t = abs(dev / (100.0 * fit["intercept_se"])) if fit["intercept_se"] > 0 else float("inf")
        lines += [
            "## The gate: Chapman-Enskog as `Kn -> 0`",
            "",
            "Chapman-Enskog is a *continuum* limit, so the gate is the `Kn -> 0`",
            "intercept, not any single finite-Knudsen run.",
            "",
            f"**ratio(Kn -> 0) = {fit['intercept']:.4f} +- {fit['intercept_se']:.4f}** "
            f"({dev:+.2f} % from Chapman-Enskog, |t| = {t:.2f} on {fit['dof']} dof: "
            f"{'consistent' if t < 2.0 else 'INCONSISTENT'}).",
            "",
            f"Slope `{fit['slope']:+.3f} +- {fit['slope_se']:.3f}` per unit Kn -- the",
            "apparent viscosity *falls* with Kn, the sign first-order slip theory",
            "predicts, as the Knudsen layers eat into the sheared bulk.",
            "",
            "`dx/lambda` is held at 0.30 in every row, so cell size is not the variable.",
            "",
            "| case | `Kn` | gap/`lambda` | warm-up (steps) | seeds | ratio |",
            "|---|---|---|---|---|---|",
        ]
        for r in fit["rows"]:
            lines.append(
                f"| {r['case']} | {r['knudsen']:.4f} | {1.0 / r['knudsen']:.0f} | "
                f"{r['warm']:,} | {r['seeds']} | **{r['ratio']:.4f} ± {r['ratio_se']:.4f}** |"
            )
        lines += [
            "",
            "The fit's uncertainty comes from its **residual scatter**, not from those",
            "per-point SEs: at these seed counts the SE is itself so noisy that points",
            "land several 'sigma' off a perfectly good line. Weighting by them would",
            "manufacture precision that is not there.",
            "",
        ]
    if "gate" in available:
        r = available["gate"][0]
        dev, se = 100.0 * (r["ratio"] - 1.0), 100.0 * r["ratio_se"]
        lines += [
            "## Single-configuration gate",
            "",
            f"`{r['case']}`, N = {r['n_part']:,}, {r['seeds']} seeds, warm-up {r['warm']:,} steps:",
            "",
            f"**ratio = {r['ratio']:.4f} ± {r['ratio_se']:.4f}** "
            f"({dev:+.2f} % ± {se:.2f} %) at `Kn = {r['knudsen']:.4f}`.",
            "",
            "This is a finite-Knudsen point and is *expected* to sit below 1; read it",
            "against the extrapolation above, not against Chapman-Enskog directly.",
            "",
        ]
    if "converge" in available:
        rows = sorted(available["converge"], key=lambda r: r["dx_over_mfp"])
        dx = dx_extrapolation(rows)
        lines += [
            "## Cell-size convergence",
            "",
            "Held at `Kn = 1/30`, varying only `dx/lambda`. The ratio is **not flat**:",
            f"it spans {dx['spread_pct']:.1f} % over a {dx['range']:.0f}x range, and the two",
            f"finest points differ by {dx['finest_gap']:+.4f} ± {dx['finest_gap_se']:.4f}",
            f"(`|z| = {dx['finest_z']:.2f}`) -- a real resolution dependence, not scatter.",
            "",
            f"`O(dx^2)` describes it (residual {dx['resid_coarse']:+.4f} at the coarsest",
            f"point against {dx['resid_coarse_linear']:+.4f} for `O(dx)`) and extrapolates",
            f"to **{dx['intercept']:.4f} as `dx -> 0`**, so the gate's own",
            f"`dx/lambda = {dx['gate_dx']:.2f}` sits {100 * dx['bias']:+.2f} % low from",
            "resolution alone.",
            "",
            "This **corrects an earlier reading of this study as flat**, which came from",
            "single draws at a constant 4 000-step warm-up.",
            "",
            "| case | `dx/lambda` | N | seeds | `mu` measured | `mu` CE | ratio |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['case']} | {r['dx_over_mfp']:.3f} | {r['n_part']:,} | {r['seeds']} | "
                f"{r['mu_meas']:.4e} | {r['mu_ce']:.4e} | **{r['ratio']:.4f} ± "
                f"{r['ratio_se']:.4f}** |"
            )
        lines += ["", ""]
    if "walls" in available:
        lines += [
            "## Wall treatments",
            "",
            "Read the shear rate and slip, not the ratio -- the specular row divides two",
            "numbers that are both at noise level.",
            "",
            "| wall | `du_s/dn` (1/s) | slip (m/s) | Maxwell slip | `tau` (Pa) |",
            "|---|---|---|---|---|",
        ]
        for r in available["walls"]:
            theory = "--" if r["slip_theory"] > 1e4 else f"{r['slip_theory']:.1f}"
            lines.append(
                f"| {r['case']} | {r['du_dy']:.4e} ± {r['du_dy_se']:.1e} | {r['slip']:.1f} | "
                f"{theory} | {r['tau_bulk']:.4e} |"
            )
        lines.append("")
    if "dims" in available:
        rows = available["dims"]
        shear = np.array([r["du_dy"] for r in rows])
        lines += [
            "## Dimension-agnostic",
            "",
            f"Shear rate spread across `D = 1, 2, 3`: "
            f"**{100 * shear.std() / abs(shear.mean()):.1f} %**.",
            "",
            "| case | N | seeds | `du_s/dn` (1/s) | slip (m/s) | recrossed |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['case']} | {r['n_part']:,} | {r['seeds']} | {r['du_dy']:.4e} ± "
                f"{r['du_dy_se']:.1e} | {r['slip']:.1f} | {r['recrossed']} |"
            )
        lines.append("")
    if made:
        lines += ["## Figures", ""] + [
            f"- `{os.path.relpath(p, 'examples/couette')}`" for p in made
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


STUDY_ORDER = ("knudsen", "gate", "converge", "walls", "dims")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m examples.couette.report", description=__doc__)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--figures", default=FIGURES)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.figures, exist_ok=True)

    available, made = {}, []
    for study in STUDY_ORDER:
        try:
            rows = parse(study, args.results)
        except FileNotFoundError as exc:
            print(f"   (skipping {study}: {exc})")
            continue
        _assert_clean(rows, study)
        available[study] = rows

    def plot(fn, *fn_args):
        if not args.no_plot:
            made.append(fn(*fn_args))

    if "knudsen" in available:
        table_knudsen(available["knudsen"])
        plot(figure_knudsen, available["knudsen"], os.path.join(args.figures, "knudsen.png"))
        plot(
            figure_profiles,
            "knudsen",
            available["knudsen"],
            os.path.join(args.figures, "knudsen_profiles.png"),
        )
    if "knudsen" in available and "converge" in available:
        table_combined(available["knudsen"], available["converge"])
    if "gate" in available:
        table_gate(available["gate"])
    if "converge" in available:
        table_converge(available["converge"])
        plot(
            figure_convergence,
            available["converge"],
            os.path.join(args.figures, "convergence.png"),
        )
        plot(
            figure_profiles,
            "converge",
            available["converge"],
            os.path.join(args.figures, "profiles.png"),
        )
    if "walls" in available:
        table_walls(available["walls"])
        plot(figure_walls, available["walls"], os.path.join(args.figures, "wall_treatments.png"))
        plot(
            figure_profiles,
            "walls",
            available["walls"],
            os.path.join(args.figures, "wall_profiles.png"),
        )
    if "dims" in available:
        table_dims(available["dims"])
        plot(figure_dimensions, available["dims"], os.path.join(args.figures, "dimensions.png"))

    path = write_results_md(available, made)
    print(f"\n=== wrote {path} ===")
    for p in made:
        print(f"   {p}")


if __name__ == "__main__":
    main()
