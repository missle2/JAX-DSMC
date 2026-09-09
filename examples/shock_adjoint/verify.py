"""Parameter sweeps comparing AD against finite differences, and the figures.

Two sweeps, because section 13.4 requires two modes and they answer different
questions.  Both are run over ``D = 1, 2, 3`` on the reflected-shock case of
:mod:`examples.shock_adjoint`.

``sweep_pathwise``
    Fixed base key.  Compares ``jax.grad`` of the **plain** objective against a
    central difference whose step is bisected down until no frozen decision flips.
    Expected agreement: round-off.  Blind to the score terms by construction.

``sweep_ensemble``
    Many base keys.  Compares the mean **scored** gradient against a central
    difference of ``E[J]`` at a step large enough for decisions to flip.  This is
    the only mode that can see ``diam``, ``omega``, ``tref`` or ``nrho`` at all.
"""

from __future__ import annotations

import math

import jax
import numpy as np

from examples.shock_adjoint import (
    HEADLINE,
    LABEL,
    SCORE_ONLY,
    Case,
    calibrate,
    case,
    ensemble_grads,
    fd_central,
    fd_ensemble,
    largest_flip_free_h,
    mean_sem,
    reference_theta,
    value_and_grad,
)

SPAN = 0.15
"""Half-width of every sweep, as a fraction of the reference value."""

N_POINTS = 7
H_ENSEMBLE = 0.06
"""Relative step for the ensemble difference.

Large on purpose.  ``E[J]`` is smooth, but each realisation is a step function of a
score-carried parameter, so a small step flips nothing, returns exactly zero for
every seed, and reports a confident ``0.0``.  The price is an ``O(h^2)`` truncation
error, which the sweep figure shows rather than hides.
"""


def _sweep_values(theta0, name, n_points=N_POINTS, span=SPAN):
    return [theta0[name] * f for f in np.linspace(1.0 - span, 1.0 + span, n_points)]


def sweep_pathwise(cs: Case, key, name, theta0=None, n_points=N_POINTS, span=SPAN):
    """AD (pathwise) vs fixed-key FD, over a sweep in ``theta[name]``.

    The flip-free step is re-bisected at **every** sweep point: section 13.4.1
    measured ``h_crit`` varying up to 30x between seeds, so a step that was safe at
    the reference is not guaranteed safe 15% away from it.
    """
    theta0 = reference_theta() if theta0 is None else theta0
    rows = []
    for val in _sweep_values(theta0, name, n_points, span):
        theta = dict(theta0)
        theta[name] = val
        _, grad = value_and_grad(theta, cs, key, False)
        h = largest_flip_free_h(theta, cs, key, name)
        if h is None:
            rows.append({"value": float(val), "ad": float(grad[name]), "fd": math.nan, "h": None})
            continue
        fd, flipped, floor = fd_central(theta, cs, key, name, h)
        rows.append(
            {
                "value": float(val),
                "ad": float(grad[name]),
                "fd": math.nan if flipped else fd,
                "h": h,
                "floor": floor,
            }
        )
    return rows


def sweep_ensemble(
    cs: Case, keys, name, theta0=None, n_points=5, span=SPAN, h=H_ENSEMBLE, chunk=32, keys_fd=None
):
    """Mean scored AD gradient vs an ensemble FD of ``E[J]``, over a sweep.

    ``keys_fd`` defaults to ``keys`` but is worth making **larger**: a reverse pass
    costs 17x a forward one (measured on the GPU: 44.8 vs 2.7 ms/key at ``D = 1``),
    so at a fixed budget the two sides should not get the same sample count.
    Minimising ``sigma_a^2/K_a + sigma_f^2/K_f`` at fixed cost puts
    ``K_a/K_f = (sigma_a/sigma_f) sqrt(c_f/c_a)``, which at the measured variances is
    about ``1:4``.
    """
    theta0 = reference_theta() if theta0 is None else theta0
    keys_fd = keys if keys_fd is None else keys_fd
    rows = []
    for val in _sweep_values(theta0, name, n_points, span):
        theta = dict(theta0)
        theta[name] = val
        grads = ensemble_grads(theta, cs, keys, chunk)
        ad, ad_sem = mean_sem(grads[name])
        fd, fd_sem = fd_ensemble(theta, cs, keys_fd, name, h, chunk)
        rows.append(
            {
                "value": float(val),
                "ad": ad,
                "ad_sem": ad_sem,
                "fd": fd,
                "fd_sem": fd_sem,
                "n_ad": len(keys),
                "n_fd": len(keys_fd),
            }
        )
    return rows


def relative_error(rows):
    """``|FD - AD| / |AD|`` at each sweep point, ``nan`` where FD is unavailable."""
    return np.array(
        [
            abs(r["fd"] - r["ad"]) / abs(r["ad"])
            if r["ad"] != 0 and np.isfinite(r["fd"])
            else (0.0 if r["ad"] == 0 and r["fd"] == 0 else math.nan)
            for r in rows
        ]
    )


def scaled_error(rows):
    """``|FD - AD| / median(|AD|)`` over the sweep -- the scale-robust companion to
    :func:`relative_error`.

    Dividing by the *local* ``|AD|`` is the natural per-point measure, but it is a harsh
    one for a parameter whose sweep crosses a near-stationary point of ``J``: there
    ``|AD|`` is small by construction and the ratio blows up without anything being wrong
    with the gradient.  That is measured, not hypothetical -- across the 21 ``alpha``
    sweep points the relative error correlates with ``|AD|`` at Spearman ``rho = -0.835``
    (``p < 1e-4``) and with the step ``h`` at ``rho = +0.025`` (``p = 0.92``), i.e. it
    tracks the denominator and not the truncation.  ``temp`` and ``ux`` never sample such
    a point (their ``|AD|`` stays in ``[165, 234]``), which is the whole reason they look
    two decades cleaner.  Normalising by the sweep's median makes the three comparable.
    """
    ad = np.array([abs(r["ad"]) for r in rows])
    scale = np.median(ad[ad > 0]) if np.any(ad > 0) else math.nan
    return np.array(
        [
            abs(r["fd"] - r["ad"]) / scale if np.isfinite(r["fd"]) and scale else math.nan
            for r in rows
        ]
    )


def floor_ratio(rows):
    """``|FD - AD|`` in units of the finite difference's own **cancellation noise**.

    Relative error alone is the wrong yardstick when ``dJ/dtheta`` is small next to
    ``J``.  ``alpha`` is the case in point: ``dJ/dalpha ~ 20`` against ``J ~ 8e4``,
    so differencing ``J`` at ``h ~ 1e-6`` loses roughly
    ``eps * J / (h * theta) ~ 2e-5`` to round-off -- which is ``1e-6`` *relative to
    the gradient*, and is exactly the size of the "disagreement" reported for it.

    A ratio of order 1 means the two agree as closely as the difference can
    resolve.  That is a pass, and calling it a 4-decade-worse result would be
    reading round-off as physics.
    """
    return np.array(
        [
            abs(r["fd"] - r["ad"]) / r["floor"]
            if r.get("floor") and np.isfinite(r["fd"])
            else math.nan
            for r in rows
        ]
    )


def z_scores(rows):
    """``(FD - AD) / sqrt(sem_FD^2 + sem_AD^2)`` -- the right statistic for mode B.

    An ensemble comparison cannot be judged by relative error: both sides are noisy,
    and the question is whether they differ by more than their combined uncertainty.
    """
    return np.array(
        [(r["fd"] - r["ad"]) / math.hypot(r["fd_sem"], r["ad_sem"]) for r in rows],
        dtype=np.float64,
    )


def run_pathwise(dims=(1, 2, 3), names=HEADLINE, seed=3, **kw):
    """The whole mode-A study.  Returns ``{(dim, name): rows}`` plus a case map."""
    out, cases = {}, {}
    for dim in dims:
        # calibrate BEFORE differentiating: `run_adaptive` cannot be used inside a
        # gradient, so `A_max` has to be settled here or the run silently clips.
        cs = cases[dim] = calibrate(case(dim, **kw))
        for name in names:
            out[dim, name] = sweep_pathwise(cs, jax.random.PRNGKey(seed), name)
    return out, cases


CHUNK_BY_DIM = {1: 128, 2: 64, 3: 32}
"""Keys per ``vmap`` batch.  Larger overruns a 4 GB card; the reverse pass of a
512-key batch at ``D = 1`` asks for 4.0 GiB and rematerialisation cannot get it
below 2.85 GiB."""


def run_ensemble(
    dims=(1, 2, 3), names=HEADLINE, n_keys=1024, fd_factor=4, seed=101, chunk=None, **kw
):
    """The whole mode-B study.  Returns ``{(dim, name): rows}`` plus a case map."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_keys)
    keys_fd = jax.random.split(jax.random.PRNGKey(seed + 1), n_keys * fd_factor)
    out, cases = {}, {}
    for dim in dims:
        cs = cases[dim] = calibrate(case(dim, **kw))
        ch = CHUNK_BY_DIM[dim] if chunk is None else chunk
        for name in names:
            out[dim, name] = sweep_ensemble(cs, keys, name, chunk=ch, keys_fd=keys_fd)
    return out, cases


def keys_for_target_se(rows, target=0.10):
    """Sample count that would put the AD standard error at ``target`` of the mean.

    This is gap U2's gate, stated in its own terms: *"report the sample count
    needed"*.  Scaling is ``K ~ (se/mean / target)^2 * K_now``.
    """
    mid = rows[len(rows) // 2]
    if mid["ad"] == 0.0:
        return math.inf
    frac = abs(mid["ad_sem"] / mid["ad"])
    return mid["n_ad"] * (frac / target) ** 2


# --- reporting -------------------------------------------------------------


def print_pathwise(res, cases):
    print("\n=== mode A: pathwise gradient vs fixed-key finite difference ===")
    print("    (zero decision flips enforced; `h` re-bisected at every sweep point)\n")
    print(
        f"{'D':>2} {'param':>7} {'N*C*M':>9} {'h* range':>19} {'max rel err':>12} "
        f"{'err/medAD':>11} {'err/FDnoise':>12} {'verdict':>10}"
    )
    for (dim, name), rows in sorted(res.items()):
        hs = [r["h"] for r in rows if r["h"] is not None]
        rel = relative_error(rows)
        worst = np.nanmax(rel) if np.any(np.isfinite(rel)) else math.nan
        sc = scaled_error(rows)
        worst_sc = np.nanmax(sc) if np.any(np.isfinite(sc)) else math.nan
        fr = floor_ratio(rows)
        worst_fr = np.nanmax(fr) if np.any(np.isfinite(fr)) else math.nan
        vacuous = all(r["ad"] == 0.0 for r in rows)
        # PASS on ANY of: a tight per-point relative match; a tight match against the
        # sweep's own gradient scale (see `scaled_error` -- this is what stops a
        # near-stationary sweep point from failing a correct gradient); or agreement
        # within the difference's own round-off, which covers a small derivative on a
        # large J.
        ok = worst < 1e-5 or worst_sc < 1e-5 or worst_fr < 20.0
        verdict = "VACUOUS" if vacuous else ("PASS" if ok else "FAIL")
        span = f"{min(hs):.1e}-{max(hs):.1e}" if hs else "none"
        print(
            f"{dim:>2} {name:>7} {cases[dim].work:9.1e} {span:>19} {worst:12.2e} "
            f"{worst_sc:11.2e} {worst_fr:12.1f} {verdict:>10}"
        )
    if any(all(r["ad"] == 0.0 for r in rows) for rows in res.values()):
        print(
            "\n  VACUOUS: both sides are exactly 0.0.  The parameter reaches J only through a\n"
            "  frozen decision, so at fixed randomness there is nothing for FD to measure and\n"
            "  nothing for the pathwise gradient to carry.  Mode B is the test that applies."
        )


def print_ensemble(res, cases, n_keys):
    print(f"\n=== mode B: scored gradient vs ensemble finite difference ({n_keys} keys) ===")
    print(f"    (h = {H_ENSEMBLE:.0%} relative, common random numbers)\n")
    print(
        f"{'D':>2} {'param':>7} {'AD dE[J]/dth':>15} {'+-':>10} {'FD':>15} {'+-':>10} "
        f"{'|z|':>6} {'K for 10%':>10}"
    )
    for (dim, name), rows in sorted(res.items()):
        z = z_scores(rows)
        mid = rows[len(rows) // 2]
        print(
            f"{dim:>2} {name:>7} {mid['ad']:15.6e} {mid['ad_sem']:10.2e} "
            f"{mid['fd']:15.6e} {mid['fd_sem']:10.2e} {np.max(np.abs(z)):6.2f}"
            f"  {keys_for_target_se(rows):>10.0f}"
        )
    del cases
    print(
        "\n  |z| is the discrepancy in units of the combined standard error, maximised over\n"
        "  the sweep.  |z| < 2-3 is agreement; a large |z| at a small standard error is a\n"
        "  real disagreement, not noise."
    )


# --- figures ---------------------------------------------------------------


def figure_pathwise(res, cases, path="examples/shock_adjoint/figures/adjoint_pathwise.png"):
    """One panel per (D, parameter): AD curve with FD markers on top."""
    from jax_dsmc.viz import (  # noqa: PLC0415
        INK,
        SERIES_ANALYTIC,
        SERIES_MEASURED,
        make_figure,
        savefig,
        style_axes,
    )

    names = sorted({n for _, n in res}, key=HEADLINE.index)
    dims = sorted({d for d, _ in res})
    fig, axes = make_figure(len(dims), len(names), figsize=(3.6 * len(names), 3.0 * len(dims)))
    # `make_figure` returns a flat list whatever the layout (house convention)
    axes = np.asarray(axes, dtype=object).reshape(len(dims), len(names))
    for r, dim in enumerate(dims):
        for c, name in enumerate(names):
            ax = axes[r, c]
            rows = res[dim, name]
            x = np.array([q["value"] for q in rows])
            ad = np.array([q["ad"] for q in rows])
            fd = np.array([q["fd"] for q in rows])
            ax.plot(x, ad, "-", color=SERIES_ANALYTIC, lw=2.0, label="AD (pathwise)")
            ax.plot(x, fd, "o", color=SERIES_MEASURED, ms=5.5, mfc="none", mew=1.6, label="FD")
            rel = relative_error(rows)
            worst = np.nanmax(rel) if np.any(np.isfinite(rel)) else math.nan
            fr = np.nanmax(floor_ratio(rows))
            if np.allclose(ad, 0.0):
                note = "both exactly 0\n(score-only parameter)"
            else:
                note = f"max rel err {worst:.1e}\n= {fr:.1f}x the FD noise floor"
            ax.set_title(f"D = {dim}   {LABEL[name]}", color=INK, fontsize=10)
            ax.annotate(
                note,
                xy=(0.04, 0.06),
                xycoords="axes fraction",
                fontsize=8,
                color=INK,
            )
            if r == len(dims) - 1:
                ax.set_xlabel(LABEL[name])
            if c == 0:
                ax.set_ylabel(r"$\partial J / \partial \theta$")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            style_axes(ax)
    fig.suptitle(
        "Mode A: pathwise gradient vs fixed-key finite difference, zero decision flips\n"
        "each sweep point is a different realisation, so the curve is jagged by design; "
        "what is tested is that FD tracks AD at every point",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, path)


def figure_ensemble(res, cases, path="examples/shock_adjoint/figures/adjoint_ensemble.png"):
    """Same grid for mode B, with error bars on both sides."""
    from jax_dsmc.viz import (  # noqa: PLC0415
        INK,
        SERIES_ANALYTIC,
        SERIES_MEASURED,
        make_figure,
        savefig,
        style_axes,
    )

    names = sorted({n for _, n in res}, key=HEADLINE.index)
    dims = sorted({d for d, _ in res})
    fig, axes = make_figure(len(dims), len(names), figsize=(3.6 * len(names), 3.0 * len(dims)))
    # `make_figure` returns a flat list whatever the layout (house convention)
    axes = np.asarray(axes, dtype=object).reshape(len(dims), len(names))
    for r, dim in enumerate(dims):
        for c, name in enumerate(names):
            ax = axes[r, c]
            rows = res[dim, name]
            x = np.array([q["value"] for q in rows])
            ax.errorbar(
                x,
                [q["ad"] for q in rows],
                yerr=[q["ad_sem"] for q in rows],
                fmt="-",
                color=SERIES_ANALYTIC,
                lw=2.0,
                capsize=3,
                label="AD (scored)",
            )
            ax.errorbar(
                x,
                [q["fd"] for q in rows],
                yerr=[q["fd_sem"] for q in rows],
                fmt="o",
                color=SERIES_MEASURED,
                ms=5.5,
                mfc="none",
                mew=1.6,
                capsize=3,
                label="FD of E[J]",
            )
            z = np.max(np.abs(z_scores(rows)))
            tag = "score-carried" if name in SCORE_ONLY else "pathwise + score"
            ax.set_title(f"D = {dim}   {LABEL[name]}", color=INK, fontsize=10)
            ax.annotate(
                f"max |z| = {z:.1f}\n{tag}",
                xy=(0.04, 0.06),
                xycoords="axes fraction",
                fontsize=8,
                color=INK,
            )
            if r == len(dims) - 1:
                ax.set_xlabel(LABEL[name])
            if c == 0:
                ax.set_ylabel(r"$\partial \mathbb{E}[J] / \partial \theta$")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            style_axes(ax)
    del cases
    fig.suptitle(
        "Mode B: scored gradient vs ensemble finite difference of E[J]", color=INK, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, path)


def figure_diagnostics(res, cases, path="examples/shock_adjoint/figures/adjoint_diagnostics.png"):
    """Two things the overlay plots cannot show: how well, and how narrow.

    Left -- the mode-A relative error per (D, parameter) on a log axis.  The overlay
    figure shows the curves lying on top of each other, which is the *claim*; this
    shows the size of the residual, which is the *evidence*.

    Right -- the largest flip-free step actually found, against section 13.4.1's
    fitted ``h_crit ~ 5.8 / (N C M)``.  The three cases here sit at ``N C M`` of
    ``1.3e6``, ``5.1e6`` and ``2.0e7``, which is an independent check of that fit at
    a point above the ``3e6`` it was originally fitted over (gap U8).
    """
    from jax_dsmc.viz import (  # noqa: PLC0415
        INK,
        REFERENCE,
        make_figure,
        savefig,
        style_axes,
        time_color,
    )

    names = sorted({n for _, n in res}, key=HEADLINE.index)
    dims = sorted({d for d, _ in res})
    fig, axes = make_figure(1, 2, figsize=(11.5, 4.2))

    width = 0.8 / len(dims)
    for i, dim in enumerate(dims):
        vals, pos = [], []
        for c, name in enumerate(names):
            rel = relative_error(res[dim, name])
            worst = np.nanmax(rel) if np.any(np.isfinite(rel)) else np.nan
            # an exact zero cannot be drawn on a log axis; it is the VACUOUS case
            vals.append(worst if worst > 0 else np.nan)
            pos.append(c + (i - (len(dims) - 1) / 2) * width)
        axes[0].bar(pos, vals, width=width, color=time_color(i, len(dims)), label=f"D = {dim}")
    axes[0].axhline(1e-6, color=REFERENCE, ls="--", lw=1.2)
    axes[0].text(len(names) - 0.6, 1.4e-6, "test tolerance", fontsize=8, color=INK, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([LABEL[n].split("  ")[0] for n in names])
    axes[0].set_ylabel(r"$|{\rm FD} - {\rm AD}| \, / \, |{\rm AD}|$")
    axes[0].set_title(
        "mode A residual\n(bars absent = both sides exactly 0)", color=INK, fontsize=10
    )
    axes[0].legend(frameon=False, fontsize=8)

    for i, dim in enumerate(dims):
        work = cases[dim].work
        hs = [r["h"] for name in names for r in res[dim, name] if r["h"] is not None]
        if hs:
            axes[1].plot(
                [work] * len(hs),
                hs,
                "o",
                color=time_color(i, len(dims)),
                ms=6,
                alpha=0.65,
                label=f"D = {dim}",
            )
    w = np.array([cases[d].work for d in dims], dtype=float)
    grid = np.logspace(np.log10(w.min()) - 0.3, np.log10(w.max()) + 0.3, 32)
    axes[1].plot(grid, 5.8 / grid, color=REFERENCE, ls="--", lw=1.4)
    axes[1].text(
        grid[2], 5.8 / grid[2] * 1.6, r"$h_{\rm crit} \sim 5.8 / (NCM)$", fontsize=8, color=INK
    )
    axes[1].axhline(1e-9, color=REFERENCE, ls=":", lw=1.2)
    axes[1].text(grid[-1], 1.3e-9, "round-off floor", fontsize=8, color=INK, ha="right")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$N \cdot C \cdot M$")
    axes[1].set_ylabel(r"largest flip-free $h$")
    axes[1].set_title(
        "the FD window, and where it closes\n(section 13.4.1, gap U8)", color=INK, fontsize=10
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    return savefig(fig, path)
