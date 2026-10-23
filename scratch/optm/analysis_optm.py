"""The 2026-10-15..17 ``optm`` runs (study.py --group optm): drag minimisation at fixed area in
free space (``--boundary free``: every box face a free-stream face), S = 4 Adam, step smoothed
over 8 knots, tangent area mode, long box (x -10..20 m, y 0..10 m, 1 m keep-out from every
face), one H200 each.  Pulled 2026-10-17: Kn 0.5 M 0.32 300 iterates, free-molecular M 10 300,
Kn 0.5 M 10 247 (the 2026-10-16 pull had 236 / 300 / 68).

    run_locked.sh cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/optm/step_where.py \
        circle_long_kn0.5_M10_S4_adam_tan_sm8_free
    run_locked.sh cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/optm/analysis_optm.py

``optm_kn0.5_M10_oscillation.png`` (from ``step_where.py``'s per-step motion of the nose): the
Kn 0.5 M 10 drag's saw-tooth against the nose's normal motion and the gradient spikes.

Per run (``figures/optc/optm/optm_<run>.png``): drag / starting drag (±2 s.e.), the outlines, the
adjoint check (observed drag change of each step, common random numbers -- the same 4 seeds
every iterate -- against its first-order prediction ``g . dc``; the fitted slope of
``dD = a g . dc + b shift`` per phase, 1 = exact first order), the gradient norm, and the
body's length / thickness.  ``optm_drag.png`` sets the three side by side, ``optm_timing.png``
the seconds per iterate against the iterate and the particle hits.  Numbers to
``scratch/optm/analysis_optm.json``.
"""

import dataclasses
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
from analysis import _fit, contour_metrics  # noqa: E402
from analysis_tan import outline, running_mean  # noqa: E402
from figure_paths import figure_path  # noqa: E402

RES = "examples/mesh/results/optimize"
LOGS = "examples/mesh/cluster/logs"
RUNS = ("kn0.5", "M10", "kn0.5_M10")
JOB = {"kn0.5": 34248393, "M10": 34248394, "kn0.5_M10": 34248395}
TITLE = {
    "kn0.5": "Kn 0.5, M 0.32 (subsonic, collisional)",
    "M10": "free-molecular, M 10",
    "kn0.5_M10": "Kn 0.5, M 10 (hypersonic, collisional)",
}
COLOURS = {"kn0.5": "#0072B2", "M10": "#D55E00", "kn0.5_M10": "#009E73"}  # Okabe-Ito
# adjoint-check phases (first iterate of each; the last runs to the end)
PHASES = {
    "kn0.5": ((0, "iterates 0-79"), (80, "80-159"), (160, "160+")),
    "M10": ((0, "0-149"), (150, "150-253"), (254, "254+: tail at the keep-out")),
    "kn0.5_M10": ((0, "0-35"), (36, "36+: nose gradient spikes")),  # one fit: the saw-tooth
}
EVERY = 5
WINDOW = 20
SPIKE = 5.0  # a gradient spike: |g| above SPIKE x the median of the last 20 iterates
NOSE_MOVE = 0.005  # m: a step that moves the nose region by more than this is advancing / receding
MARGIN = 1.0


def tag(run):
    return f"circle_long_{run}_S4_adam_tan_sm8_free"


def load(run):
    with open(f"{RES}/{tag(run)}/history.json") as fh:
        return json.load(fh)


def node(run):
    with open(f"{LOGS}/optm_{run}_{JOB[run]}.out") as fh:
        return fh.readline().split(" on ")[1].split(":")[0]


def adjoint(rows, phases):
    J0 = rows[0]["objective"]
    J = np.array([r["objective"] for r in rows]) / J0
    p = np.array([r["predicted_decrease"] for r in rows])[:-1] / J0
    sh = np.array([r["area_shift"] for r in rows])[:-1]
    obs = np.diff(J)
    out = {"obs": obs.tolist(), "pred": p.tolist(), "phases": []}
    starts = [s for s, _l in phases] + [len(obs)]
    for (s, lab), e in zip(phases, starts[1:], strict=True):
        if e - s < 8:
            continue
        a, se = _fit(obs[s:e], np.c_[p[s:e], sh[s:e]])
        out["phases"].append({
            "label": lab, "start": s, "end": e, "a": a, "a_se": se,
            "ratio": float(obs[s:e].sum() / p[s:e].sum()),
            "corr": float(np.corrcoef(obs[s:e], p[s:e])[0, 1]),
        })  # fmt: skip
    return out


def geometry(run, h):
    grid, ls = O.initial_levelset(h["config"])
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n = len(h["rows"])
    shapes, outlines = {}, {}
    marks = sorted({0, n // 4, n // 2, 3 * n // 4, n})
    for k in sorted(set(range(0, n + 1, EVERY)) | set(marks)):
        c = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{RES}/{tag(run)}/coef_{k}.npy")))
        shapes[k] = contour_metrics(c, lo, hi)
        if k in marks:
            outlines[k] = outline(c, lo, hi, dx=0.04)
    return lo, hi, shapes, outlines


def _style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def keepout_span(ax, rows):
    k = np.array([r["keepout_knots"] for r in rows])
    if (k > 0).any():
        first = int(np.argmax(k > 0))
        ax.axvspan(first, len(rows) - 1, color="0.85", alpha=0.6, lw=0,
                   label=f"body held at the 1 m keep-out (from iterate {first})")  # fmt: skip


def plot_run(run, h, geo, adj):  # noqa: PLR0915 -- one six-panel figure
    rows = h["rows"]
    lo, hi, shapes, outlines = geo
    colour = COLOURS[run]
    J0 = rows[0]["objective"]
    J = np.array([r["objective"] for r in rows]) / J0
    se = np.array([r["objective_se"] for r in rows]) / J0
    it = np.arange(len(J))
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5), width_ratios=(1, 1.45, 1))
    # (a) drag
    ax = axes[0, 0]
    keepout_span(ax, rows)
    ax.fill_between(it, J - 2 * se, J + 2 * se, color=colour, alpha=0.2, lw=0)
    ax.plot(it, J, color=colour, lw=1.4, label="drag / starting drag (band ±2 s.e., 4 seeds)")
    kmin = int(np.argmin(J))
    ax.plot(kmin, J[kmin], "v", color="0.2", ms=7, label=f"lowest: {J[kmin]:.3f} at iterate {kmin}")
    ax.plot(it[-1], J[-1], "o", color="0.2", ms=6, label=f"last: {J[-1]:.3f} at iterate {it[-1]}")
    ax.set_ylim(0, 1.05)
    _style(ax, "(a) drag of the body / drag of the starting circle (area fixed)",
           "optimisation iterate", "drag / initial drag")  # fmt: skip
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    # (b) outlines, light -> dark with the iterate
    ax = axes[0, 1]
    ks = sorted(outlines)
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 1.0, len(ks)))
    for k, sh in zip(ks, shades, strict=True):
        for j, s in enumerate(outlines[k]):
            ax.plot(s[:, 0], s[:, 1], color=sh, lw=1.3, ls=":" if k == 0 else "-",
                    label=f"iterate {k}" if j == 0 else None)  # fmt: skip
    for s in outlines[ks[-1]]:
        ax.fill(s[:, 0], s[:, 1], color=colour, alpha=0.18, lw=0)
    ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, color="0.2", lw=1.2))
    ax.add_patch(plt.Rectangle(lo + MARGIN, *(hi - lo - 2 * MARGIN), fill=False, color="0.5",
                               lw=0.8, ls="--"))  # fmt: skip
    ax.annotate("", xy=(lo[0] + 2.5, hi[1] - 1.2), xytext=(lo[0] + 0.5, hi[1] - 1.2),
                arrowprops={"arrowstyle": "->", "color": "0.3"})  # fmt: skip
    ax.text(lo[0] + 0.5, hi[1] - 0.95, "flow", fontsize=8, color="0.3")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    _style(ax, "(b) outlines (box solid, 1 m keep-out dashed; every face free stream)",
           "x [m]", "y [m]")  # fmt: skip
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=len(ks))
    # (c) observed vs predicted per step
    ax = axes[0, 2]
    obs, pred = np.array(adj["obs"]), np.array(adj["pred"])
    lim = 1.15 * np.percentile(np.abs(np.r_[obs, pred]), 97)
    off = int(np.sum((np.abs(pred) > lim) | (np.abs(obs) > lim)))
    sc = ax.scatter(np.clip(pred, -lim, lim), np.clip(obs, -lim, lim), c=np.arange(len(obs)),
                    cmap="viridis", s=12, lw=0)  # fmt: skip
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1, label="observed = predicted")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("iterate", fontsize=8)
    lines = [f"{q['label']}: slope {q['a']:.2f} ± {q['a_se']:.2f}, ΣΔD/Σpred {q['ratio']:.2f}"
             for q in adj["phases"]]  # fmt: skip
    if off:
        lines.append(f"{off} steps off scale, drawn at the edge")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.2,
            bbox={"fc": "white", "ec": "0.8", "alpha": 0.9})  # fmt: skip
    _style(ax, "(c) each step's drag change: observed vs the adjoint's prediction",
           "predicted g·Δc / initial drag", "observed ΔD (same 4 seeds) / initial drag")  # fmt: skip
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    # (d) running ratio
    ax = axes[1, 0]
    keepout_span(ax, rows)
    r = running_mean(obs, WINDOW) / running_mean(pred, WINDOW)
    ax.plot(np.arange(WINDOW - 1, len(obs)), r, color=colour, lw=1.4,
            label=f"Σ observed / Σ predicted, last {WINDOW} steps")  # fmt: skip
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label="1 = the step does what the adjoint says")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylim(-1.0, 1.6)
    _style(ax, "(d) adjoint check along the run (< 1: step noise or overshoot;\n< 0: the drag "
           "rose while the adjoint predicted a fall)", "optimisation iterate")  # fmt: skip
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    # (e) gradient norm
    ax = axes[1, 1]
    keepout_span(ax, rows)
    g = np.array([x["grad_norm"] for x in rows])
    ax.semilogy(it, g / g[0], color=colour, lw=1.2, label="|gradient| / starting |gradient|")
    _style(
        ax,
        "(e) gradient size (every step moves the largest knot exactly 0.05 m)",
        "optimisation iterate",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    # (f) body geometry
    ax = axes[1, 2]
    keepout_span(ax, rows)
    ks = sorted(shapes)
    ax.plot(ks, [shapes[k]["length"] for k in ks], color=colour, lw=1.4, label="length (x) [m]")
    ax.plot(ks, [shapes[k]["height"] for k in ks], color=colour, lw=1.4, ls="--",
            label="thickness (y) [m]")  # fmt: skip
    _style(
        ax,
        f"(f) body length and thickness (area fixed; the keep-out allows {hi[0] - lo[0] - 2 * MARGIN:.0f} m)",
        "optimisation iterate",
        "m",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    fig.suptitle(
        f"Drag minimisation at fixed area in free space, {TITLE[run]}: Adam, 4 DSMC seeds per "
        f"iterate, {len(rows)} iterates so far (optm, H200, pulled 2026-10-17)",
        fontsize=11.5,
    )
    fig.tight_layout()
    path = figure_path("optc/optm", f"optm_{run}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def spikes(rows):
    g = np.array([r["grad_norm"] for r in rows])
    med = np.array([np.median(g[max(0, k - 20) : k + 1]) for k in range(len(g))])
    return np.flatnonzero(g > SPIKE * med)


def plot_oscillation(run, h):
    """The saw-tooth: the drag's rises against the nose's motion (``step_where.py``: the mean
    coefficient change over the surface knots within 1.5 m of the nose, + = the surface recedes)
    and the gradient spikes."""
    path = f"scratch/optm/step_where_{tag(run)}.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        where = json.load(fh)
    rows = h["rows"]
    J0 = rows[0]["objective"]
    J = np.array([r["objective"] for r in rows]) / J0
    se = np.array([r["objective_se"] for r in rows]) / J0
    g = np.array([r["grad_norm"] for r in rows])
    n = min(len(where), len(J) - 1)
    nose = np.array([w["nose_dc"] for w in where])[:n] * 1e3
    obs = np.diff(J)[:n]
    spk = spikes(rows)
    it = np.arange(len(J))
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.6), width_ratios=(2.3, 1))
    ax = axes[0, 0]
    adv = nose < -NOSE_MOVE * 1e3
    for k in np.flatnonzero(adv):
        ax.axvspan(k, k + 1, color="#CC79A7", alpha=0.18, lw=0)
    ax.fill_between(it, J - 2 * se, J + 2 * se, color=COLOURS[run], alpha=0.25, lw=0)
    ax.plot(it, J, color=COLOURS[run], lw=1.5, label="drag / starting drag (±2 s.e.)")
    for k in spk:
        ax.axvline(k, color="0.25", lw=0.7, ls=":")
    ax.plot(
        [],
        [],
        color="#CC79A7",
        alpha=0.4,
        lw=6,
        label="steps that move the nose forward (upstream)",
    )
    ax.plot(
        [],
        [],
        color="0.25",
        lw=0.7,
        ls=":",
        label=f"gradient spike (|g| > {SPIKE:g}x its running median)",
    )
    ax.set_ylim(0.1, 1.02)
    _style(ax, "(a) drag: the rises happen while the steps push the nose upstream; the largest start "
           "at a gradient spike (iterates 109, 156, 232)", "optimisation iterate", "drag / initial drag")  # fmt: skip
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax = axes[1, 0]
    ax.bar(np.arange(n), nose, width=1.0, color=np.where(nose < 0, "#CC79A7", "#56B4E9"), lw=0)
    for k in spk:
        ax.axvline(k, color="0.25", lw=0.7, ls=":")
    ax.axhline(0, color="0.4", lw=0.6)
    _style(ax, "(b) the step's motion of the nose region (mean coefficient change of the surface "
           "knots within 1.5 m of the nose; + recedes, − advances)", "optimisation iterate",
           "mm per iterate")  # fmt: skip
    ax2 = axes[0, 1]
    ax2.semilogy(it, g / g[0], color=COLOURS[run], lw=1.0)
    ax2.semilogy(spk, g[spk] / g[0], "o", color="0.25", ms=4, label="spike")
    _style(ax2, "(c) |gradient| / starting |gradient|", "optimisation iterate")
    ax2.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    c = np.corrcoef(obs[36:], nose[36:])[0, 1]
    ax.scatter(nose, obs, c=np.arange(n), cmap="viridis", s=12, lw=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    _style(ax, f"(d) drag change of a step vs its nose motion (correlation {c:.2f} after iterate 36)",
           "nose motion [mm] (+ recedes)", "observed ΔD / initial drag")  # fmt: skip
    fig.suptitle(
        f"Why the drag saw-tooths, {TITLE[run]}: the nose's gradient is spiky and flips sign; Adam's "
        "momentum keeps each direction for 5-25 iterates and the full 5 cm step drives the nose that way",
        fontsize=11.5,
    )
    fig.tight_layout()
    out = figure_path("optc/optm", f"optm_{run}_oscillation.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)
    return {"corr_obs_nose_after36": float(c), "spikes": spk.tolist(),
            "mean_dD_nose_advancing": float(obs[adv].mean()),
            "mean_dD_nose_receding": float(obs[nose > NOSE_MOVE * 1e3].mean())}  # fmt: skip


def plot_drag(hist):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for run, h in hist.items():
        J = np.array([r["objective"] for r in h["rows"]])
        J /= J[0]
        ax.plot(J, color=COLOURS[run], lw=1.5,
                label=f"{TITLE[run]}: {J[-10:].mean():.3f} (mean of the last 10 of {len(J)}), "
                f"lowest {J.min():.3f}")  # fmt: skip
    ax.set_ylim(0, 1.05)
    _style(ax, "Drag / starting drag at fixed area, free-space boundary, S = 4 Adam, smooth 8 "
           "knots (optm, pulled 2026-10-17)", "optimisation iterate", "drag / initial drag")  # fmt: skip
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = figure_path("optc/optm", "optm_drag.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def plot_timing(hist, nodes):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for run, h in hist.items():
        s = np.array([r["seconds"] for r in h["rows"]])
        hits = np.array([r["hits"] for r in h["rows"]])
        lab = (f"{TITLE[run]} ({nodes[run]}): iterate 0 {s[0]:.0f} s (compile), then "
               f"median {np.median(s[1:]):.0f} s")  # fmt: skip
        axes[0].plot(np.arange(1, len(s)), s[1:], color=COLOURS[run], lw=1.2, label=lab)
        axes[1].plot(hits[1:] / 1e3, s[1:], "o", ms=3, color=COLOURS[run], mew=0, alpha=0.7)
    axes[0].set_ylim(0, None)
    _style(axes[0], "(a) seconds per iterate (4 seeds) after the first", "optimisation iterate",
           "seconds")  # fmt: skip
    axes[0].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0, 0.62))
    # the ~1200 s iterates: a full recompile of the backward and the shape gradient -- the
    # densest cell crossed a 64-particle step of npc_max (ensemble_mesh.NPC_BUCKET), a static
    # argument of both
    axes[0].text(0.99, 0.35, "M 10 spikes at iterates 127, 139, 192: recompiles of the backward + shape gradient\n(the "
                 "densest cell crossed a 64-particle npc_max step)", transform=axes[0].transAxes,
                 ha="right", fontsize=7.5, color="0.3")  # fmt: skip
    axes[1].set_ylim(0, None)
    _style(axes[1], "(b) seconds per iterate against the body's particle hits per seed",
           "hits per seed [thousands]")  # fmt: skip
    fig.suptitle("optm step timings, one H200 per run (shared node c3gpu-g5-u13 for M10 and "
                 "Kn 0.5 M10)", fontsize=11)  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc/optm", "optm_timing.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main():
    hist = {run: load(run) for run in RUNS}
    nodes = {run: node(run) for run in RUNS}
    out = {}
    for run, h in hist.items():
        rows = h["rows"]
        adj = adjoint(rows, PHASES[run])
        geo = geometry(run, h)
        J = np.array([r["objective"] for r in rows]) / rows[0]["objective"]
        s = np.array([r["seconds"] for r in rows])
        hits = np.array([r["hits"] for r in rows])
        k = np.array([r["keepout_knots"] for r in rows])
        out[run] = {
            "iterates": len(rows), "node": nodes[run], "drag_last10": float(J[-10:].mean()),
            "drag_min": float(J.min()), "drag_argmin": int(J.argmin()),
            "keepout_first": int(np.argmax(k > 0)) if (k > 0).any() else None,
            "area_rel_max_err": float(np.abs(np.array([r["area"] for r in rows]) / rows[0]["area"] - 1).max()),
            "adjoint_phases": adj["phases"],
            "seconds": {"first": float(s[0]), "median": float(np.median(s[1:])),
                        "p5": float(np.percentile(s[1:], 5)), "p95": float(np.percentile(s[1:], 95)),
                        "max": float(s[1:].max()), "argmax": int(s[1:].argmax() + 1),
                        "corr_hits": float(np.corrcoef(s[1:], hits[1:])[0, 1])},
            "shapes": {str(i): v for i, v in geo[2].items()},
        }  # fmt: skip
        print(run, json.dumps({k: v for k, v in out[run].items() if k != "shapes"}), flush=True)
        plot_run(run, h, geo, adj)
        osc = plot_oscillation(run, h)
        if osc:
            out[run]["oscillation"] = osc
            print(run, "oscillation", json.dumps(osc), flush=True)
    with open("scratch/optm/analysis_optm.json", "w") as fh:
        json.dump(out, fh, indent=1)
    plot_drag(hist)
    plot_timing(hist, nodes)


if __name__ == "__main__":
    main()
