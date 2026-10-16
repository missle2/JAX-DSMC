"""The 2026-10-15 ``optab`` A/B runs with the tangent area mode (study.py --group optab, tags
``circle_long_S4_adam_tan_ab_<name>``): step smoothing 2 / 4 / 8 knots against a coarser level
set, optc2 S = 4 Adam, free-molecular, long box.  The first ~110-129 of 300 iterates (pulled
mid-run, 2026-10-15).

    run_locked.sh cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/optab/analysis_tan.py

The area holds to ~1e-12 relative in tangent mode, so the drag itself is the comparison.  Per run:

* drag / initial drag, and drag / control run's drag (a 10-iterate running mean);
* shape roughness of the zero contour (``analysis.contour_metrics``: total turning / 2 pi,
  smallest radius of curvature, concave fraction) every 5 iterates, and the outlines at the last
  iterate all runs reached;
* the adjoint's apparent error: the observed drag change of a step (common random numbers: the
  same 4 seeds every iterate) against its first-order prediction ``g . dc``, as the slope of
  ``dD_k = a g . dc_k + b shift_k`` and as running ratios ``sum dD / sum g . dc``.  A slope
  below 1 has two causes that rise together near the optimum: the step's own noise (``g . dc``
  carries the noise's power, ``E[a] = S / (S + rho)``; Adam's momentum cuts this, the step is
  mostly past gradients) and the step's second-order term (a fixed step size overshoots once the
  gradient shrinks);
* the per-iterate wall time against the wall clock (all five ran side by side on one rtx node,
  separate MIG 1g.24gb slices) and against the particle hits.

Writes ``figures/optc/optab_tan_{drag,shapes,adjoint,timing}.png`` and
``scratch/optab/analysis_tan.json``.
"""

import dataclasses
import datetime
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optab")
import optimize as O  # noqa: E402
from analysis import _fit, contour_metrics, eta, load  # noqa: E402
from figure_paths import figure_path  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

RES = "examples/mesh/results/optimize"
LOGS = "examples/mesh/cluster/logs"
AB = ("ctl", "smooth4", "smooth8", "lsh0.125", "lsh0.25")
JOB = {"ctl": 34223187, "smooth4": 34223188, "smooth8": 34223189, "lsh0.125": 34223190,
       "lsh0.25": 34223191}  # fmt: skip
LABEL = {
    "ctl": "control: step smoothed over 2 knots (12.5 cm); level-set spacing h = 6.25 cm",
    "smooth4": "step smoothed over 4 knots (25 cm); h = 6.25 cm",
    "smooth8": "step smoothed over 8 knots (50 cm); h = 6.25 cm",
    "lsh0.125": "coarser level set h = 12.5 cm; smoothed over 2 knots (25 cm)",
    "lsh0.25": "coarser level set h = 25 cm; smoothed over 2 knots (50 cm)",
}
SHORT = {
    "ctl": "control (smooth 12.5 cm)",
    "smooth4": "smooth 25 cm",
    "smooth8": "smooth 50 cm",
    "lsh0.125": "h 12.5 cm",
    "lsh0.25": "h 25 cm",
}
REF = "circle_long_S4_adam"  # same run, the older projection area constraint
COLOURS = {  # Okabe-Ito, as optab_overview.png
    "ctl": "#000000",
    "smooth4": "#0072B2",
    "smooth8": "#56B4E9",
    "lsh0.125": "#D55E00",
    "lsh0.25": "#E69F00",
}
STYLE = {"lsh0.125": "--", "lsh0.25": "--"}
EVERY = 5
RUNNING = 10
WINDOW = 20


def tag(key):
    return f"circle_long_S4_adam_tan_ab_{key}"


def running_mean(x, n):
    c = np.cumsum(np.r_[0.0, x])
    return (c[n:] - c[:-n]) / n


def outline(ls, lo, hi, dx=0.02):
    """Zero-contour segments of ``ls`` (grid ``dx``)."""
    xs = np.arange(lo[0], hi[0], dx)
    ys = np.arange(lo[1], hi[1], dx)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    phi = np.asarray(L.value(ls, jnp.asarray(np.stack([xx.ravel(), yy.ravel()], 1))))
    fig, ax = plt.subplots()
    cs = ax.contour(xx, yy, phi.reshape(xx.shape), levels=[0.0])
    segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 4]
    plt.close(fig)
    return segs


def start_time(key):
    """The job's start from its log's first line (``[Thu Oct 15 14:20:10 MDT 2026] ...``)."""
    with open(f"{LOGS}/optab_{key}_{JOB[key]}.out") as fh:
        stamp = fh.readline().split("]")[0].strip("[").replace(" MDT", "")
    return datetime.datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y")


def change_point(s):
    """The single mean shift of ``s[1:]`` (iterate 0 compiles) minimising the squared error."""
    best = (np.inf, 0)
    for k in range(10, len(s) - 10):
        a, b = s[1:k], s[k:]
        v = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        best = min(best, (v, k))
    k = best[1]
    return k, float(s[1:k].mean()), float(s[k:].mean())


def adjoint(rows):
    J = np.array([r["objective"] for r in rows])
    p = np.array([r["predicted_decrease"] for r in rows])[:-1]
    sh = np.array([r["area_shift"] for r in rows])[:-1]
    obs = np.diff(J)
    n = len(obs)
    half = np.arange(n) < n // 2
    a, e = _fit(obs, np.c_[p, sh])
    a1, e1 = _fit(obs[half], np.c_[p, sh][half])
    a2, e2 = _fit(obs[~half], np.c_[p, sh][~half])
    return {
        "a": a, "a_se": e, "a_first": a1, "a_first_se": e1, "a_second": a2, "a_second_se": e2,
        "ratio_total": float(obs.sum() / p.sum()),
        "ratio_last30": float(obs[-30:].sum() / p[-30:].sum()),
        "corr": float(np.corrcoef(obs, p)[0, 1]),
        "obs": obs.tolist(), "pred": p.tolist(),
    }  # fmt: skip


def main():
    hist = {k: load(tag(k)) for k in AB}
    ref = load(REF)
    n_common = min(len(h["rows"]) for h in hist.values())
    out = {"n_common": n_common}
    outlines = {}
    for key, h in hist.items():
        rows = h["rows"]
        grid, ls = O.initial_levelset(h["config"])
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        n = len(rows)
        shapes = {}
        for k in sorted(set(range(0, n + 1, EVERY)) | {n_common, n}):
            path = f"{RES}/{tag(key)}/coef_{k}.npy"
            if os.path.exists(path):
                shapes[k] = contour_metrics(
                    dataclasses.replace(ls, coef=jnp.asarray(np.load(path))), lo, hi
                )
        cn = dataclasses.replace(
            ls, coef=jnp.asarray(np.load(f"{RES}/{tag(key)}/coef_{n_common}.npy"))
        )
        outlines[key] = (outline(ls, lo, hi), outline(cn, lo, hi))
        s = np.array([r["seconds"] for r in rows])
        k_jump, before, after = change_point(s)
        t0 = start_time(key)
        t_jump = t0 + datetime.timedelta(seconds=float(s[:k_jump].sum()))
        J = np.array([r["objective"] for r in rows])
        A = np.array([r["area"] for r in rows])
        rec = {
            "iterates": n,
            "drag_rel": {str(k): float(J[k] / J[0]) for k in (25, 50, 75, 100, n_common - 1)},
            "drag_rel_last": float(J[-1] / J[0]),
            "drag_rel_last10": float(J[-10:].mean() / J[0]),
            "area_rel_max_err": float(np.abs(A / A[0] - 1).max()),
            "shapes": {str(k): v for k, v in shapes.items()},
            "adjoint": adjoint(rows),
            "timing": {
                "jump_iterate": k_jump,
                "jump_clock": t_jump.strftime("%H:%M"),
                "before": before,
                "after": after,
                "s_per_khit_before": float(s[1:k_jump].mean() / np.mean([r["hits"] for r in rows[1:k_jump]]) * 1e3),
                "s_per_khit_after": float(s[k_jump:].mean() / np.mean([r["hits"] for r in rows[k_jump:]]) * 1e3),
            },
        }  # fmt: skip
        out[key] = rec
        last = shapes[max(shapes)]
        q, t = rec["adjoint"], rec["timing"]
        print(
            f"{key:9s} n {n:3d} drag/D0 @100 {rec['drag_rel']['100']:.3f} last10 "
            f"{rec['drag_rel_last10']:.3f}  turning {last['turning']:.2f} r_min "
            f"{last['r_min'] * 100:.0f} cm concave {last['concave']:.2f}  adjoint a "
            f"{q['a']:.2f}+-{q['a_se']:.2f} (halves {q['a_first']:.2f} {q['a_second']:.2f}) "
            f"last30 {q['ratio_last30']:.2f}  time {t['before']:.0f}->{t['after']:.0f} s at it "
            f"{t['jump_iterate']} ({t['jump_clock']})",
            flush=True,
        )
    out["reference_projection_S4"] = {"eta": eta(ref).tolist()}
    with open("scratch/optab/analysis_tan.json", "w") as fh:
        json.dump(out, fh)
    plot_drag(hist, ref)
    plot_shapes(hist, out, outlines)
    plot_adjoint(out)
    plot_timing(hist, out)


def _style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def plot_drag(hist, ref):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axes[0]
    for key, h in hist.items():
        J = np.array([r["objective"] for r in h["rows"]])
        it = np.arange(len(J))
        lab = f"{LABEL[key]}: {J[-10:].mean() / J[0]:.2f} (last 10 of {len(J)})"
        ax.plot(it, J / J[0], STYLE.get(key, "-"), color=COLOURS[key], lw=1.5, label=lab)
    e = eta(ref)[: max(len(h["rows"]) for h in hist.values())]
    ax.plot(e, ":", color="0.55", lw=1.3,
            label="earlier run, same settings but area held by projection (Adam + shift)")  # fmt: skip
    ax.set_ylim(0, 1.05)
    _style(
        ax,
        "(a) drag of the optimised body / drag of the starting circle (area held fixed)",
        "optimisation iterate",
        "drag / initial drag",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax = axes[1]
    Jc = np.array([r["objective"] for r in hist["ctl"]["rows"]])
    sec = np.array([r["objective_se"] for r in hist["ctl"]["rows"]])
    for key, h in hist.items():
        J = np.array([r["objective"] for r in h["rows"]])
        se = np.array([r["objective_se"] for r in h["rows"]])
        n = min(len(J), len(Jc))
        rr = running_mean(J[:n] / Jc[:n], RUNNING)
        it = np.arange(RUNNING - 1, n)
        ax.plot(it, rr, STYLE.get(key, "-"), color=COLOURS[key], lw=1.5, label=SHORT[key])
        if key != "ctl":
            band = 2 * running_mean(np.hypot(se[:n] / J[:n], sec[:n] / Jc[:n]), RUNNING)
            band /= np.sqrt(RUNNING)
            ax.fill_between(it, rr - band, rr + band, color=COLOURS[key], alpha=0.15, lw=0)
    _style(
        ax,
        f"(b) drag / control run's drag at the same iterate ({RUNNING}-iterate running mean,"
        " band ±2 s.e.)",
        "optimisation iterate",
        "drag relative to control",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    fig.suptitle(
        "Drag minimisation at fixed area, free-molecular flow, Adam with 4 DSMC seeds per "
        "iterate: which step regularisation? (optab A/B, tangent area constraint, rtx, "
        "2026-10-15)",
        fontsize=11,
    )
    fig.tight_layout()
    path = figure_path("optc", "optab_tan_drag.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


def plot_shapes(hist, out, outlines):
    n = out["n_common"]
    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 15)
    pts = np.concatenate([s for pair in outlines.values() for segs in pair for s in segs])
    (x0, y0), (x1, y1) = pts.min(0) - 0.3, pts.max(0) + 0.3
    for j, key in enumerate(AB):
        ax = fig.add_subplot(gs[0, 3 * j : 3 * j + 3])
        init, now = outlines[key]
        for s in init:
            ax.plot(s[:, 0], s[:, 1], color="0.6", lw=0.8, ls=":")
        for s in now:
            ax.plot(s[:, 0], s[:, 1], color=COLOURS[key], lw=1.4)
            ax.fill(s[:, 0], s[:, 1], color=COLOURS[key], alpha=0.12, lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.tick_params(labelsize=7)
        d = out[key]["drag_rel"][str(n - 1)]
        ax.set_title(f"{SHORT[key]}\ndrag/D0 = {d:.2f}", fontsize=9)
        if j == 0:
            ax.set_ylabel("y [m] (flow left to right)", fontsize=8)
    titles = (
        ("turning", "(b) total turning / 2π (1 = convex)", 1.0),
        ("r_min", "(c) smallest radius of curvature [cm]", 100.0),
        ("concave", "(d) concave fraction of the perimeter", 1.0),
    )
    for j, (field, title, scale) in enumerate(titles):
        ax = fig.add_subplot(gs[1, 5 * j : 5 * j + 5])
        for key in AB:
            sh = out[key]["shapes"]
            ks = sorted(int(k) for k in sh)
            ax.plot(ks, [sh[str(k)][field] * scale for k in ks], STYLE.get(key, "-"),
                    color=COLOURS[key], lw=1.4, label=SHORT[key])  # fmt: skip
        if field == "r_min":
            ax.set_yscale("log")
        _style(ax, title, "optimisation iterate")
        if j == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"(a) body outlines at iterate {n} (dotted: starting circle) and their roughness "
        "-- optab A/B, tangent area constraint",
        fontsize=11,
    )
    fig.tight_layout()
    path = figure_path("optc", "optab_tan_shapes.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


def plot_adjoint(out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    lim = 0.0
    for key in AB:
        q = out[key]["adjoint"]
        p, o = np.array(q["pred"]) * 1e22, np.array(q["obs"]) * 1e22
        lim = max(lim, np.abs(p).max(), np.abs(o).max())
        ax.plot(p, o, "o", ms=3, color=COLOURS[key], alpha=0.6, mew=0, label=SHORT[key])
    ax.plot([-lim, 0.2 * lim], [-lim, 0.2 * lim], color="0.4", lw=1, label="observed = predicted")
    _style(
        ax,
        "(a) drag change of each step: observed vs adjoint",
        "adjoint prediction g·Δc  [1e-22]",
        "observed ΔD (same 4 seeds)  [1e-22]",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[1]
    for key in AB:
        q = out[key]["adjoint"]
        p, o = np.array(q["pred"]), np.array(q["obs"])
        r = running_mean(o, WINDOW) / running_mean(p, WINDOW)
        ax.plot(np.arange(WINDOW - 1, len(o)), r, STYLE.get(key, "-"), color=COLOURS[key],
                lw=1.4, label=SHORT[key])  # fmt: skip
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.set_ylim(-0.2, 1.4)
    _style(
        ax,
        f"(b) Σ observed / Σ predicted over the last {WINDOW} steps",
        "optimisation iterate",
    )
    ax = axes[2]
    x = np.arange(len(AB))
    for dx, field, lab, mfc in (
        (-0.12, "first", "first half", None),
        (0.12, "second", "second half", "white"),
    ):
        for j, key in enumerate(AB):
            q = out[key]["adjoint"]
            ax.errorbar(x[j] + dx, q[f"a_{field}"], 2 * q[f"a_{field}_se"], fmt="o", ms=6,
                        color=COLOURS[key], mfc=mfc or COLOURS[key], capsize=2,
                        label=lab if j == 0 else None)  # fmt: skip
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.axhline(4 / (4 + 4.8), color="0.3", lw=1, ls="--",
               label="earlier projection runs at S = 4: S/(S+4.8) = 0.45")  # fmt: skip
    ax.set_xticks(x, [SHORT[k].replace(" (", "\n(") for k in AB], fontsize=7.5)
    ax.set_ylim(0, 1.3)
    _style(ax, "(c) fitted slope a of ΔD = a g·Δc + b shift (±2 s.e.)", "")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(
        "Adjoint check along the trajectory: does the drag fall as much as the gradient "
        "predicts? (1 = exact first order)",
        fontsize=11,
    )
    fig.tight_layout()
    path = figure_path("optc", "optab_tan_adjoint.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


def plot_timing(hist, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for key, h in hist.items():
        s = np.array([r["seconds"] for r in h["rows"]])
        hits = np.array([r["hits"] for r in h["rows"]])
        t0 = start_time(key)
        clock = [t0 + datetime.timedelta(seconds=float(c)) for c in np.cumsum(s)]
        axes[0].plot(clock[1:], s[1:], STYLE.get(key, "-"), color=COLOURS[key], lw=1.2,
                     label=SHORT[key])  # fmt: skip
        axes[1].plot(hits[1:] / 1e3, s[1:], "o", ms=3, color=COLOURS[key], mew=0, alpha=0.7)
    import matplotlib.dates as mdates

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _style(
        axes[0],
        "(a) seconds per iterate against the wall clock (all 5 jobs on c3gpu-e7-u5, own MIG slices)",
        "end of iterate, MDT 2026-10-15",
        "seconds per iterate (4 seeds)",
    )
    axes[0].legend(fontsize=8, frameon=False)
    _style(axes[1], "(b) seconds per iterate against the body's particle hits",
           "hits per seed [thousands]")  # fmt: skip
    fig.suptitle("Iterate timings (iterate 0 = compile omitted)", fontsize=11)
    fig.tight_layout()
    path = figure_path("optc", "optab_tan_timing.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    main()
