"""Collect the cluster study's artifacts into the convergence figures and a table.

    python3 -m examples.mesh.cluster.collect

* ``figures/optc/optimize_convergence.png`` (``opt``) and ``optimize_convergence_coarse.png``
  (``optc``): drag vs iterate and vs cost (``S x iterate`` seeds evaluated) for every
  ``(S, optimiser)`` run, area check.
* ``figures/optc/optc_drag.png``: the optc drag histories with the first shapes crossing the
  inflow / outflow face and the held-out drag (``results/optimize/heldout_drag.json``, if
  present); ``figures/optc/optc_body_extent.png``: the body's gaps to those faces per iterate
  (per-run cache ``results/optimize/<tag>/contact.json``; this part imports jax for
  :func:`jax_dsmc.mesh.levelset.value`).
* ``figures/optc/gradient_seed_convergence.png``: the ``cos 2 theta`` gradient's standard error
  vs number of seeds (bootstrap over the 32-seed artifact) against the CRN FD;
  ``figures/gates/mode2_gate.png``: that artifact's per-term, per-seed and per-variant gate.
* the sphere gate row (ensemble vs FD, z) if its artifact exists.
* ``squash`` / ``squashkn``: ``figures/squash/squash_gradient_gate.png`` (``dD/ds`` of every
  ``grad_s`` artifact against its CRN FD and the free-molecular reference, per estimator) and
  ``figures/squash/squash_optimisation.png`` (the completed ``optimize_squash`` runs: ``s``
  per iterate, and the slope ratio ``(dD/ds) / (lam dV/ds)`` whose crossing of 1 is the
  optimum); the per-run panels are ``optimize_squash.py --plot``.
The per-seed files of the seed jobs (``..._dev_seed{i}.json``) are merged into the usual
artifacts first; the report (incl. the optc run accounting from the Slurm logs) goes to
``study_results.json``.  Everything comes from the artifacts; nothing is typed in.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
RESULTS_OPT = os.path.join(MESH, "results", "optimize")
FIGURES = os.path.join(MESH, "figures")
sys.path.insert(0, MESH)
from figure_paths import figure_path  # noqa: E402

GRADIENT_RESULTS = os.path.join(os.path.dirname(MESH), "gradient_bench", "results")
SEED_START = 32


def merge_seed_artifacts() -> dict[str, int]:
    """The per-seed files of the ``sphere`` / ``mode2`` groups (``<artifact>_seed{i}.json``,
    one row each with ``seed = 0``) -> the usual artifact ``<artifact>.json`` with the rows
    renumbered by their file index; rewritten from the per-seed files every time.  Returns
    ``{artifact: seeds merged}``."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(GRADIENT_RESULTS, "ensemble_mesh_*_dev_seed*.json"))):
        stem, _, idx = os.path.basename(path)[: -len(".json")].rpartition("_seed")
        target = os.path.join(GRADIENT_RESULTS, stem + ".json")
        with open(path) as fh:
            d = json.load(fh)
        if not d["rows"]:
            continue
        row = dict(d["rows"][0], seed=int(idx))
        if target in merged:
            with open(target) as fh:
                combined = json.load(fh)
        else:
            combined = dict(d, rows=[])
            combined["config"]["seed_start"] = SEED_START
        combined["rows"] = [r for r in combined["rows"] if r["seed"] != row["seed"]] + [row]
        combined["rows"].sort(key=lambda r: r["seed"])
        with open(target, "w") as fh:
            json.dump(combined, fh, indent=1)
        merged[target] = len(combined["rows"])
    return {os.path.basename(k): v for k, v in merged.items()}


def load_histories(label="fine"):
    """``history.json`` of the ``circle_<label>_S*`` runs (``fine``: opt, ``coarse``: optc)."""
    out = {}
    pattern = os.path.join(RESULTS_OPT, f"circle_{label}_S*_*", "history.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            h = json.load(fh)
        tag = os.path.basename(os.path.dirname(path))
        out[tag] = h
    return out


def figure_optimize(hist, path_fig):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for h in hist.values():
        rows = h["rows"]
        s = h["config"]["seeds_per_iterate"]
        opt = h["config"]["optimizer"]
        it = np.array([r["iterate"] for r in rows])
        j = np.array([r["objective"] for r in rows])
        se = np.array(
            [r["objective_se"] if r["objective_se"] == r["objective_se"] else 0.0 for r in rows]
        )
        ls = "-" if opt == "gd" else "--"
        axes[0].errorbar(it, j, yerr=se, fmt="o" + ls, ms=3, capsize=2, label=f"S={s} {opt}")
        axes[1].plot((it + 1) * s, j, "o" + ls, ms=3, label=f"S={s} {opt}")
        axes[2].plot(
            it,
            np.array([r["area"] for r in rows]) / h["area_target"] - 1.0,
            ls,
            label=f"S={s} {opt}",
        )
    axes[0].set_xlabel("iterate")
    axes[0].set_ylabel("windowed drag J (N/m)")
    axes[0].set_title("per iterate (CRN seeds; error bar = seed spread / sqrt S)", fontsize=8)
    axes[1].set_xlabel("seeds evaluated (S x iterates)")
    axes[1].set_ylabel("J")
    axes[1].set_title("per unit cost", fontsize=8)
    axes[2].set_xlabel("iterate")
    axes[2].set_ylabel("area / target - 1")
    axes[2].set_title("constraint", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def gradient_seed_convergence(path_fig):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    paths = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_levelset*grid2*_mode2_*_dev.json")
        )
    )
    if not paths:
        return None
    with open(paths[0]) as fh:
        d = json.load(fh)
    g = np.array([r["gradient"] for r in d["rows"]])
    fd = np.array([r["fd"][0] for r in d["rows"]])
    n = len(g)
    rng = np.random.default_rng(0)
    counts = [c for c in (1, 2, 4, 8, 16, 32) if c <= n]
    se = []
    for c in counts:
        means = [g[rng.choice(n, c, replace=True)].mean() for _ in range(2000)]
        se.append(float(np.std(means)))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(counts, se, "o-", label="SE of the ensemble mean (bootstrap)")
    ax.loglog(counts, se[0] / np.sqrt(np.array(counts)), color="0.5", lw=1, label="1 / sqrt(S)")
    ax.axhline(abs(fd.mean()) * 0.01, color="tab:red", ls="--", lw=1, label="1 % of |FD|")
    ax.set_xlabel("seeds S")
    ax.set_ylabel("standard error of dJ/dθ")
    ax.set_title(
        f"cos 2θ mode, {n} seeds: ensemble {g.mean():.3e} ± {g.std(ddof=1) / math.sqrt(n):.1e}, FD {fd.mean():.3e}",
        fontsize=7,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    return {
        "seeds": n,
        "counts": counts,
        "se": se,
        "ensemble": float(g.mean()),
        "fd": float(fd.mean()),
        "artifact": os.path.basename(paths[0]),
    }


def sphere_gate():
    paths = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_sphere_levelset*_radius_*_dev.json")
        )
    )
    if not paths:
        return None
    with open(paths[0]) as fh:
        d = json.load(fh)
    g = np.array([r["gradient"] for r in d["rows"]])
    fd = np.array([r["fd"][0] for r in d["rows"]])
    n = len(g)
    z = (
        (g.mean() - fd.mean()) / math.sqrt((g.var(ddof=1) + fd.var(ddof=1)) / n)
        if n > 1
        else float("nan")
    )
    return {
        "seeds": n,
        "ensemble": float(g.mean()),
        "ensemble_se": float(g.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "fd": float(fd.mean()),
        "fd_se": float(fd.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "z": float(z),
        "artifact": os.path.basename(paths[0]),
    }


# the categorical slots of `jax_dsmc.viz` (blue, orange, aqua, yellow), fixed per seed count
SEED_COLOURS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
REFERENCE = "#8a8a86"
GRID = "#d9d8d4"
LOGS = os.path.join(HERE, "logs")
END_PATTERNS = (
    ("time limit", "DUE TO TIME LIMIT"),
    ("host OOM kill", "oom_kill"),
    ("cancelled", "CANCELLED AT"),
    ("GPU OOM", "RESOURCE_EXHAUSTED"),
)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _stat(a):
    a = np.asarray(a, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return {"mean": float(a.mean()), "se": sd / math.sqrt(len(a)), "sd": sd}


def _job_id(path: str) -> int:
    return int(path.rsplit("_", 1)[1][: -len(".out")])


def run_done(h) -> bool:
    """An optimisation history has finished: converged (``optimize.window_converged``) or
    at its iterate cap."""
    return bool(h.get("converged")) or len(h["rows"]) >= h["config"]["iterations"]


def job_end(text: str, iterates: int, target: int) -> str:
    """How one Slurm log ended: ``complete`` when the run reached its target, else the
    first matching reason (a Python error's last line when there is one)."""
    lines = [ln for ln in text.splitlines() if ln.startswith(("RuntimeError", "ValueError"))]
    if lines:
        return lines[-1].split(": ", 1)[-1][:70]
    for label, pattern in END_PATTERNS:
        if pattern in text:
            return label
    return "complete" if iterates >= target else "running / unknown"


def run_accounting(hist):
    """Per run of the study, from its ``history.json`` and every Slurm log of its job
    name: jobs submitted, iterates per job, how the last job ended, and timing (median
    seconds per iterate over the history, per seed = that / S)."""
    import re  # noqa: PLC0415

    out = {}
    for tag, h in hist.items():
        cfg = h["config"]
        s, opt = cfg["seeds_per_iterate"], cfg["optimizer"]
        logs = sorted(
            glob.glob(os.path.join(LOGS, f"optc_opt_S{s}_{opt}_*.out")),
            key=_job_id,
        )
        jobs = []
        for path in logs:
            with open(path) as fh:
                text = fh.read()
            its = [int(m) for m in re.findall(r"^iterate (\d+):", text, re.M)]
            jobs.append(
                {
                    "job": _job_id(path),
                    "iterates": [its[0], its[-1]] if its else None,
                    "end": "converged"
                    if "converged after" in text
                    else job_end(text, its[-1] + 1 if its else 0, cfg["iterations"]),
                }
            )
        sec = np.array([r["seconds"] for r in h["rows"]])
        out[tag] = {
            "S": s,
            "optimizer": opt,
            "iterates": len(h["rows"]),
            "target": cfg["iterations"],
            "converged": bool(h.get("converged")),
            "jobs": jobs,
            "last_end": jobs[-1]["end"] if jobs else "no log",
            "seconds_per_iterate_median": float(np.median(sec)),
            "seconds_per_iterate_p90": float(np.percentile(sec, 90)),
            "seconds_per_seed": float(np.median(sec) / s),
            "gpu_hours": float(sec.sum() / 3600.0),
            "J_first": h["rows"][0]["objective"],
            "J_last": h["rows"][-1]["objective"],
            "J_min": min(r["objective"] for r in h["rows"]),
            "stuck_total": float(sum(r.get("stuck", 0.0) for r in h["rows"])),
            "area_rel_max": float(
                max(abs(r["area_after_step"] / h["area_target"] - 1.0) for r in h["rows"])
            ),
        }
    return out


# the circle case's box (``examples.mesh.forward.CASES["circle"]``)
CIRCLE_BOX = (0.0, 10.0)
FACE_SAMPLES = 4001  # points along each of the inflow / outflow faces (2.5 mm apart)
EXTENT_SPACING = 0.01  # the grid on which the body's bounding box is read (m)


def _phi(ls, pts):
    """``phi`` of the level set at ``pts`` (``f64[P, 2]``), in chunks (host memory)."""
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.mesh import levelset as L  # noqa: PLC0415

    return np.concatenate(
        [
            np.asarray(L.value(ls, jnp.asarray(pts[a : a + 200_000])))
            for a in range(0, len(pts), 200_000)
        ]
    )


def shape_contact(coef, h):
    """For one level-set iterate: ``min phi`` along the inflow face ``x = lo`` and along the
    outflow face ``x = hi`` (``< 0``: the body crosses that face; evaluated by
    :func:`levelset.value` at :data:`FACE_SAMPLES` points), and the body's bounding box
    ``(x_min, x_max, y_min, y_max)`` from ``phi < 0`` on an :data:`EXTENT_SPACING` grid
    over the box widened by 0.5 m in ``x`` (so a protrusion through a face is measured)."""
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.mesh import levelset as L  # noqa: PLC0415
    from jax_dsmc.mesh.types import LevelSet  # noqa: PLC0415

    lo, hi = CIRCLE_BOX
    origin, _shape = L.grid_for_box(np.array([lo, lo]), np.array([hi, hi]), h)
    ls = LevelSet(coef=jnp.asarray(coef), lo=jnp.asarray(origin), h=float(h))
    y = np.linspace(lo, hi, FACE_SAMPLES)
    faces = [np.stack([np.full_like(y, xf), y], axis=1) for xf in (lo, hi)]
    face_min = [float(_phi(ls, f).min()) for f in faces]
    xs = np.arange(lo - 0.5, hi + 0.5 + EXTENT_SPACING / 2, EXTENT_SPACING)
    ys = np.arange(lo, hi + EXTENT_SPACING / 2, EXTENT_SPACING)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    body = (_phi(ls, np.stack([xx.ravel(), yy.ravel()], axis=1)) < 0.0).reshape(xx.shape)
    bx, by = xx[body], yy[body]
    return {
        "inflow_phi_min": face_min[0],
        "outflow_phi_min": face_min[1],
        "extent": [float(bx.min()), float(bx.max()), float(by.min()), float(by.max())],
    }


def contacts(tag, h):
    """:func:`shape_contact` of every saved ``coef_k`` of a run, cached in
    ``<run>/contact.json`` (recomputed when the run has new iterates).  Returns the list
    (index ``k`` = the shape entering iterate ``k``, i.e. the shape of history row ``k``)."""
    out_dir = os.path.join(RESULTS_OPT, tag)
    n = 0
    while os.path.exists(os.path.join(out_dir, f"coef_{n}.npy")):
        n += 1
    cache = os.path.join(out_dir, "contact.json")
    rows = []
    if os.path.exists(cache):
        with open(cache) as fh:
            rows = json.load(fh)["rows"]
    for k in range(len(rows), n):
        rows.append(shape_contact(np.load(os.path.join(out_dir, f"coef_{k}.npy")), h))
    with open(cache, "w") as fh:
        json.dump(
            {"face_samples": FACE_SAMPLES, "extent_spacing": EXTENT_SPACING, "rows": rows}, fh
        )
    return rows


def first_contact(rows, face):
    """The first shape index ``k`` whose body crosses ``face`` (``inflow`` / ``outflow``),
    or ``None``."""
    hit = [k for k, r in enumerate(rows) if r[f"{face}_phi_min"] < 0.0]
    return hit[0] if hit else None


def figure_extent_optc(hist, path_fig):
    """The gap between the body and the inflow face (x = lo) and the outflow face
    (x = hi) vs iterate for every optc run (colours and line styles of
    :func:`figure_drag_optc`); returns the final extents."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    final = {}
    for tag, h in sorted(hist.items()):
        s, opt = h["config"]["seeds_per_iterate"], h["config"]["optimizer"]
        e = np.array([r["extent"] for r in contacts(tag, h["config"]["levelset_h"])])
        final[tag] = dict(zip(("x_min", "x_max", "y_min", "y_max"), e[-1].tolist(), strict=True))
        ls = "-" if opt == "gd" else "--"
        for ax, gap in zip(axes, (e[:, 0] - CIRCLE_BOX[0], CIRCLE_BOX[1] - e[:, 1]), strict=True):
            ax.plot(np.arange(len(e)), gap, ls, color=SEED_COLOURS[s], lw=2.0, label=f"S={s} {opt}")
    for ax, name in zip(axes, ("inflow face (x = 0)", "outflow face (x = 10)"), strict=True):
        _style(ax)
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.axhline(0.5, color=REFERENCE, lw=0.8, ls=":")
        ax.axhspan(-0.55, -0.5, color=GRID, lw=0)
        ax.set_xlabel("iterate", fontsize=9, color=INK)
        ax.set_title(f"gap between the body and the {name}", fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("gap [m]   (dotted: one DSMC cell)", fontsize=9, color=INK)
    fig.text(
        0.99, 0.01, "negative: the body protrudes through the face; measured only to 0.5 m "
        "beyond it (grey band = at least 0.5 m)", ha="right", fontsize=7, color=INK_SECONDARY,
    )  # fmt: skip
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)
    return final


def _draw_run(ax, tag, h, heldout):
    """One run's curve of :func:`figure_drag_optc`: drag, SE band, end marker, first
    boundary contact, held-out drag of the last shape, direct label."""
    s, opt = h["config"]["seeds_per_iterate"], h["config"]["optimizer"]
    it = np.array([r["iterate"] for r in h["rows"]])
    j = np.array([r["objective"] for r in h["rows"]]) * 1e20
    se = np.nan_to_num(np.array([r["objective_se"] for r in h["rows"]], dtype=float)) * 1e20
    colour = SEED_COLOURS[s]
    ls = "-" if opt == "gd" else "--"
    if s > 1:
        ax.fill_between(it, j - se, j + se, color=colour, alpha=0.18, lw=0)
    ax.plot(it, j, ls, color=colour, lw=2.0)
    ax.plot(it[-1], j[-1], "o", color=colour, ms=5, mec="white", mew=1.0)
    rows = contacts(tag, h["config"]["levelset_h"])
    for face, marker in (("inflow", "<"), ("outflow", ">")):
        k = first_contact(rows, face)
        if k is None:
            continue
        if k < len(it):  # the crossing shape was evaluated: mark its own drag
            ax.plot(it[k], j[k], marker, color=INK, ms=7, mec="white", mew=0.8, zorder=6)
        else:  # the run stopped on the crossing shape: hollow, at the last drag
            ax.plot(k, j[-1], marker, mfc="white", mec=INK, ms=7, mew=1.2, zorder=6)
    if heldout and tag in heldout:
        v = np.array(heldout[tag]["drag"]) * 1e20
        k = heldout[tag]["iterate"]
        ax.errorbar(
            k, v.mean(), yerr=v.std(ddof=1) / math.sqrt(len(v)), fmt="D", ms=5,
            mfc="white", mec=colour, ecolor=colour, mew=1.5, capsize=2, zorder=5,
        )  # fmt: skip
    ax.annotate(
        f"S={s} {opt}", (it[-1], j[-1]), xytext=(5, 0), textcoords="offset points",
        fontsize=7, color=INK_SECONDARY, va="center",
    )  # fmt: skip


def figure_drag_optc(hist, path_fig, heldout=None):
    """Windowed drag vs iterate for every optc run: colour = seeds per iterate, solid =
    GD, dashed = Adam; band = seed SE (none at S = 1); the held-out drag of the circle
    and of each run's last shape (independent seeds, ``heldout_drag.json``) if present."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    _style(ax)
    order = sorted(
        hist.items(),
        key=lambda kv: (kv[1]["config"]["optimizer"], kv[1]["config"]["seeds_per_iterate"]),
    )
    for tag, h in order:
        _draw_run(ax, tag, h, heldout)
    if heldout and "circle" in heldout:
        v = np.array(heldout["circle"]["drag"]) * 1e20
        ax.axhline(v.mean(), color=REFERENCE, lw=1.0, ls=":")
        ax.annotate(
            "circle (held-out seeds)", (0, v.mean()), xytext=(2, 3), textcoords="offset points",
            fontsize=7, color=INK_SECONDARY,
        )  # fmt: skip
    handles = [
        Line2D([], [], color=SEED_COLOURS[s], lw=2.0, label=f"S = {s}") for s in (1, 2, 4, 8)
    ]
    handles += [
        Line2D([], [], color=INK_SECONDARY, lw=2.0, ls="-", label="gradient descent"),
        Line2D([], [], color=INK_SECONDARY, lw=2.0, ls="--", label="Adam"),
    ]
    handles += [
        Line2D([], [], ls="", marker="<", color=INK, mec="white", ms=7,
               label="first shape crossing the inflow face"),
        Line2D([], [], ls="", marker=">", color=INK, mec="white", ms=7,
               label="first shape crossing the outflow face"),
        Line2D([], [], ls="", marker="<", mfc="white", mec=INK, ms=7,
               label="crossing shape never evaluated (run stopped)"),
    ]  # fmt: skip
    if heldout:
        handles.append(
            Line2D([], [], ls="", marker="D", mfc="white", mec=INK_SECONDARY, mew=1.5,
                   label="last shape, held-out seeds")
        )  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax.set_xlabel("iterate", color=INK, fontsize=9)
    ax.set_ylabel("windowed drag J  [1e-20 N/m]", color=INK, fontsize=9)
    ax.set_title(
        "Coarse-circle shape optimisation at fixed area (optc): drag on each run's own CRN seeds",
        fontsize=9, color=INK, loc="left",
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def mode2_summary():
    """The fine (``grid2``, fnum x 1) ``cos 2 theta`` gradient against its CRN FD: per-term
    means and SEs over the seeds, per-seed paired differences, the estimator variants."""
    paths = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_levelset*grid2*_mode2_*_dev.json")
        )
    )
    if not paths:
        return None
    with open(paths[0]) as fh:
        d = json.load(fh)
    rows = d["rows"]
    n = len(rows)

    fd = np.array([r["fd"] for r in rows])
    out = {"artifact": os.path.basename(paths[0]), "seeds": n, "config": d["config"]}
    for k in (
        "gradient",
        "pathwise_tally",
        "pathwise_x",
        "pathwise_v",
        "timing",
        "objective",
        "seconds",
        "seconds_backward",
        "seconds_shape",
    ):
        out[k] = _stat([r[k] for r in rows])
    out["fd"] = [_stat(fd[:, i]) for i in range(fd.shape[1])]
    fd_mean = fd.mean(axis=1)
    for k in ("gradient", "pathwise_tally"):
        diff = np.array([r[k] for r in rows]) - fd_mean
        out[f"{k}_minus_fd"] = dict(
            _stat(diff), z=float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)))
        )
    out["variants"] = {}
    for v in rows[0]["variants"]:
        diff = np.array([r["variants"][v]["gradient"] for r in rows]) - fd_mean
        out["variants"][v] = dict(
            _stat(diff), z=float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)))
        )
    coarse = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_levelset0.125_mode2_f4_*_dev.json")
        )
    )
    if coarse:
        with open(coarse[0]) as fh:
            c = json.load(fh)["rows"]
        out["coarse"] = {
            "artifact": os.path.basename(coarse[0]),
            "seeds": len(c),
            **{
                k: _stat([r[k] for r in c])
                for k in ("gradient", "pathwise_tally", "pathwise_x", "pathwise_v")
            },
            "fd": _stat([np.mean(r["fd"]) for r in c]),
        }
    return out, rows


def figure_mode2(summary, rows, path_fig):
    """The fine cos 2 theta gate: (a) the terms of the adjoint estimate against the CRN FD
    (mean +- SE over the seeds; the coarse run's parts as hollow markers); (b) per seed,
    the full estimate and its tally term against that seed's FD; (c) every fitted-slope
    estimator's paired difference from the FD."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for ax in axes:
        _style(ax)
    _mode2_terms(axes[0], summary)
    _mode2_seeds(axes[1], rows)
    _mode2_variants(axes[2], summary)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def _mode2_terms(ax, summary):
    """Panel (a) of :func:`figure_mode2`."""
    blue = SEED_COLOURS[1]
    labels = ["tally (fixed hit set)", "x-part  (grad_x lambda)", "v-part  (grad_v lambda)",
              "adjoint total", "FD  h = 0.1", "FD  h = 0.05"]  # fmt: skip
    stats = [summary["pathwise_tally"], summary["pathwise_x"], summary["pathwise_v"],
             summary["gradient"], summary["fd"][0], summary["fd"][1]]  # fmt: skip
    y = np.arange(len(labels))[::-1]
    for yi, lab, st in zip(y, labels, stats, strict=True):
        colour = REFERENCE if lab.startswith("FD") else blue
        ax.barh(yi, st["mean"] * 1e20, height=0.6, color=colour, alpha=0.85)
        ax.errorbar(st["mean"] * 1e20, yi, xerr=st["se"] * 1e20, color=INK, capsize=2, lw=1)
    if "coarse" in summary:
        c = summary["coarse"]
        for yi, k in zip(
            y[:4], ("pathwise_tally", "pathwise_x", "pathwise_v", "gradient"), strict=True
        ):
            ax.plot(c[k]["mean"] * 1e20, yi, "o", mfc="white", mec=INK, ms=6, mew=1.2)
        ax.plot(c["fd"]["mean"] * 1e20, y[4], "o", mfc="white", mec=INK, ms=6, mew=1.2,
                label=f"coarse run (grid 1, fnum x4, {c['seeds']} seeds)")  # fmt: skip
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.axvline(0.0, color=REFERENCE, lw=0.8)
    left = min(st["mean"] - st["se"] for st in stats) * 1e20
    ax.set_xlim(1.1 * left, 0.1 * abs(left))
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("dJ/d(mode amplitude)  [1e-20]", fontsize=9, color=INK)
    ax.set_title(
        f"(a) terms, {summary['seeds']} seeds (grid 2, fnum x1)", fontsize=9, loc="left", color=INK
    )


def _mode2_seeds(ax, rows):
    """Panel (b) of :func:`figure_mode2`."""
    blue, orange = SEED_COLOURS[1], SEED_COLOURS[2]
    fd = np.array([np.mean(r["fd"]) for r in rows]) * 1e20
    total = np.array([r["gradient"] for r in rows]) * 1e20
    tally = np.array([r["pathwise_tally"] for r in rows]) * 1e20
    lo, hi = min(fd.min(), total.min(), tally.min()), max(fd.max(), total.max(), tally.max())
    pad = 0.05 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad], color=REFERENCE, lw=1, ls=":", label="y = FD"
    )
    ax.plot(fd, total, "o", color=blue, ms=5, mec="white", mew=0.8, label="adjoint total")
    ax.plot(fd, tally, "o", color=orange, ms=5, mec="white", mew=0.8, label="tally term alone")
    ax.set_xlabel("CRN FD of the seed  [1e-20]", fontsize=9, color=INK)
    ax.set_ylabel("estimate  [1e-20]", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(b) seed by seed", fontsize=9, loc="left", color=INK)


def _mode2_variants(ax, summary):
    """Panel (c) of :func:`figure_mode2`."""
    blue, orange = SEED_COLOURS[1], SEED_COLOURS[2]
    names = list(summary["variants"])
    yv = np.arange(len(names))[::-1]
    for yi, v in zip(yv, names, strict=True):
        st = summary["variants"][v]
        ax.errorbar(
            st["mean"] * 1e20, yi, xerr=2 * st["se"] * 1e20, fmt="o", color=blue, capsize=3, ms=6
        )
        ax.annotate(f"z = {st['z']:.0f}", (st["mean"] * 1e20, yi), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7, color=INK_SECONDARY)  # fmt: skip
    st = summary["pathwise_tally_minus_fd"]
    ax.errorbar(
        st["mean"] * 1e20, -1, xerr=2 * st["se"] * 1e20, fmt="o", color=orange, capsize=3, ms=6
    )
    ax.annotate(f"z = {st['z']:+.1f}", (st["mean"] * 1e20, -1), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=7, color=INK_SECONDARY)  # fmt: skip
    ax.axvline(0.0, color=REFERENCE, lw=0.8)
    ax.set_yticks([*yv, -1], [*names, "tally only"], fontsize=8)
    ax.set_ylim(-1.7, len(names) - 0.3)
    ax.set_xlabel("estimate - FD  [1e-20]  (paired, +-2 SE)", fontsize=9, color=INK)
    ax.set_title("(c) grad-lambda estimators", fontsize=9, loc="left", color=INK)


SQUASH_RESULTS = os.path.join(MESH, "results", "optimize_squash")
SQUASH_VARIANTS = {  # the four grad-lambda estimators, colour and label
    "one": ("#1baf7a", "one-sided"),
    "along": ("#eda100", "along-flight"),
    "one_pool": ("#eb6834", "pooled"),
    "along_pool": ("#2a78d6", "pooled + along-flight (used by the optimiser)"),
}


def _squash_module():
    """``optimize_squash`` (its free-molecular ``reference``; imports jax via ensemble_mesh)."""
    import optimize_squash  # noqa: PLC0415

    return optimize_squash


def squash_gate():
    """The ``grad_s`` artifacts (ellipse, ``squash_y``): per ``s``, the CRN FD, the reference
    ``dD/ds``, and per estimator the adjoint and its paired difference from the FD."""
    paths = [
        p for p in glob.glob(os.path.join(GRADIENT_RESULTS, "ensemble_mesh_ellipse_*_squash_y_*_dev.json"))
        if "_dev_seed" not in p
    ]  # fmt: skip
    if not paths:
        return None
    osq = _squash_module()
    out = []
    for path in paths:
        with open(path) as fh:
            art = json.load(fh)
        rows, cfg = art["rows"], art["config"]
        s = float(cfg["squash"])
        fd = np.array([np.mean(r["fd"]) for r in rows])
        tally = np.array([r["pathwise_tally"] for r in rows])
        ref_cfg = dict(
            osq.default_config(), knudsen=cfg.get("knudsen", 0.0), mach=cfg.get("mach", 0.0)
        )
        entry = {
            "s": s,
            "seeds": len(rows),
            "complete": art.get("complete", False),
            "fd": _stat(fd),
            "reference": float(osq.reference(ref_cfg, s)[1]),
            "tally_minus_fd": _stat(tally - fd),
            "variants": {},
        }
        for name in rows[0]["variants"]:
            g = np.array([r["variants"][name]["gradient"] for r in rows])
            d = _stat(g - fd)
            entry["variants"][name] = {
                "gradient": _stat(g),
                "minus_fd": {**d, "z": d["mean"] / d["se"]},
            }
        out.append(entry)
    return sorted(out, key=lambda e: e["s"])


def figure_squash_gate(gate, path_fig):
    """(a) ``dD/ds`` against ``s``: the free-molecular reference, the CRN FD and every
    estimator (+-2 SE, offset in ``s`` for legibility); (b) each estimator's paired
    difference from the FD per ``s``."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    osq = _squash_module()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax in axes:
        _style(ax)
    ax = axes[0]
    grid = np.linspace(0.2, 1.05, 120)
    cfg = osq.default_config()
    ax.plot(grid, [osq.reference(cfg, x)[1] * 1e20 for x in grid], color=REFERENCE, lw=1.5,
            label="free-molecular reference")  # fmt: skip
    s = np.array([e["s"] for e in gate])
    ax.errorbar(s, [e["fd"]["mean"] * 1e20 for e in gate], yerr=[2 * e["fd"]["se"] * 1e20 for e in gate],
                fmt="s", color=INK, ms=6, capsize=3, label="CRN finite difference")  # fmt: skip
    for j, (name, (colour, label)) in enumerate(SQUASH_VARIANTS.items()):
        dx = (j - 1.5) * 0.012
        g = [e["variants"][name]["gradient"] for e in gate]
        ax.errorbar(s + dx, [x["mean"] * 1e20 for x in g], yerr=[2 * x["se"] * 1e20 for x in g],
                    fmt="o", color=colour, ms=4, capsize=2, label=f"adjoint, {label}")  # fmt: skip
    ax.set_xlabel("y semi-axis factor s", fontsize=9, color=INK)
    ax.set_ylabel("dD/ds  [1e-20 N/m]  (+-2 SE)", fontsize=9, color=INK)
    ax.set_title(
        f"(a) dD/ds, {gate[0]['seeds']} CRN seeds per point", fontsize=9, loc="left", color=INK
    )
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[1]
    for j, (name, (colour, label)) in enumerate(SQUASH_VARIANTS.items()):
        dx = (j - 1.5) * 0.012
        d = [e["variants"][name]["minus_fd"] for e in gate]
        ax.errorbar(s + dx, [x["mean"] * 1e21 for x in d], yerr=[2 * x["se"] * 1e21 for x in d],
                    fmt="o", color=colour, ms=5, capsize=2, label=label)  # fmt: skip
    d = [e["tally_minus_fd"] for e in gate]
    ax.errorbar(s + 0.035, [x["mean"] * 1e21 for x in d], yerr=[2 * x["se"] * 1e21 for x in d],
                fmt="D", mfc="none", color=INK_SECONDARY, ms=5, capsize=2, label="tally term alone")  # fmt: skip
    ax.axhline(0.0, color=REFERENCE, lw=0.8)
    ax.set_xlabel("y semi-axis factor s", fontsize=9, color=INK)
    ax.set_ylabel("adjoint - FD  [1e-21 N/m]  (paired, +-2 SE)", fontsize=9, color=INK)
    ax.set_title("(b) paired error of each estimator", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def load_squash_histories():
    """``{tag: history}`` of the ``optimize_squash`` runs."""
    out = {}
    for path in sorted(glob.glob(os.path.join(SQUASH_RESULTS, "*", "history.json"))):
        with open(path) as fh:
            out[os.path.basename(os.path.dirname(path))] = json.load(fh)
    return out


def _squash_label(h):
    c = h["config"]
    regime = "free-molecular"
    if c.get("knudsen", 0.0) > 0.0:
        regime = f"Kn {c['knudsen']:g}"
        regime += f", M {c['mach']:g}" if c.get("mach", 0.0) > 0.0 else ", M 0.32"
    return f"{regime}, S = {c['seeds_per_iterate']}"


def _run_label(h):
    """``S = n``, with the progress when the run is not finished."""
    n, target = len(h["rows"]), h["config"]["iterations"]
    label = f"S = {h['config']['seeds_per_iterate']}"
    return label if run_done(h) else f"{label} (running, {n}/{target})"


def _iterate_panel(ax, h, key, key_se=None, factor=1.0, marker="o"):
    """One run's ``key`` per iterate (+- 1 SE when the run has several seeds)."""
    rows = h["rows"]
    it = [r["iterate"] for r in rows]
    y = np.array([r[key] for r in rows]) * factor
    colour = SEED_COLOURS[h["config"]["seeds_per_iterate"]]
    ax.plot(it, y, "-", marker=marker, color=colour, ms=3.5, lw=1.4, label=_run_label(h))
    if key_se is not None:
        se = np.nan_to_num(np.array([r[key_se] for r in rows])) * factor
        ax.fill_between(it, y - se, y + se, color=colour, alpha=0.18, lw=0)


def _power(values):
    top = max(abs(float(v)) for v in values)
    return math.floor(math.log10(top)) if top > 0 else 0


def _slope_panel(ax, osq, c, lam_dv, gate):
    """Panel (c) of :func:`figure_squash_optimisation`: the reference and the gate's FD."""
    grid = np.linspace(0.2, 1.05, 120)
    ax.plot(grid, [osq.reference(c, x)[1] / lam_dv for x in grid], color=REFERENCE, lw=1.5,
            label="free-molecular reference")  # fmt: skip
    if gate:
        ax.errorbar([e["s"] for e in gate], [e["fd"]["mean"] / lam_dv for e in gate],
                    yerr=[2 * e["fd"]["se"] / lam_dv for e in gate], fmt="s", color=INK, ms=6,
                    capsize=3, label="CRN FD (gradient gate, 8 seeds, +-2 SE)")  # fmt: skip
    ax.axhline(1.0, color=INK_SECONDARY, lw=0.8)
    ax.set_xlabel("y semi-axis factor s", fontsize=9, color=INK)
    ax.set_ylabel("(dD/ds) / (lam dV/ds)", fontsize=9, color=INK)
    ax.set_title("(c) slope ratio: the optimum is where it is 1", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def figure_squash_optimisation(hists, gate, path_fig):
    """The free-molecular squash optimisations (the analytic reference holds): (a) the
    objective ``J = D - lam V`` per iterate against the reference's ``J`` at the same ``s``;
    (b) ``s`` per iterate; (c) the slope ratio ``(dD/ds) / (lam dV/ds)`` at each iterate
    (1 = the optimum), with the reference and the FD of the gradient gate."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    osq = _squash_module()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax in axes:
        _style(ax)
    power = _power([r["J"] for h in hists.values() for r in h["rows"]])
    lam_dv = None
    for h in hists.values():
        c, rows = h["config"], h["rows"]
        colour = SEED_COLOURS[c["seeds_per_iterate"]]
        lam_dv = h["lam"] * osq.reference(c, 1.0)[3]
        _iterate_panel(axes[0], h, "J", "drag_se", 10.0**-power)
        j_ref = [(r["D_ref"] - lam_dv * r["s"]) * 10.0**-power for r in rows]
        axes[0].plot([r["iterate"] for r in rows], j_ref, ":", color=colour, lw=1.2,
                     label=f"reference J at the same s ({_run_label(h)})")  # fmt: skip
        _iterate_panel(axes[1], h, "s")
        s = np.array([r["s"] for r in rows])
        ratio = np.array([r["dD_ds"] for r in rows]) / lam_dv
        axes[2].plot(s, ratio, ls="none", marker="o", color=colour, ms=4, label=_run_label(h))
    axes[0].set_xlabel("iterate", fontsize=9, color=INK)
    axes[0].set_ylabel(f"J = D - lam V  [1e{power} N/m]  (+-1 SE)", fontsize=9, color=INK)
    axes[0].set_title("(a) objective (common random numbers)", fontsize=9, loc="left", color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].axhline(0.35, color=REFERENCE, ls=":", lw=1.2, label="reference optimum s* = 0.35")
    axes[1].set_xlabel("iterate", fontsize=9, color=INK)
    axes[1].set_ylabel("y semi-axis factor s", fontsize=9, color=INK)
    axes[1].set_title("(b) Adam iterates (lr 0.05, from s = 1)", fontsize=9, loc="left", color=INK)
    axes[1].legend(fontsize=7, frameon=False)
    _slope_panel(axes[2], osq, next(iter(hists.values()))["config"], lam_dv, gate)
    fig.suptitle("Squash, free-molecular (Kn ~ 1e18, M 0.32)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def figure_squash_set(hists, title, path_fig):
    """One set of squash runs in the same flow (no analytic reference): (a) the objective
    ``J = D - lam V`` per iterate, (b) ``s`` per iterate, (c) the adjoint ``J' = dD/ds -
    lam dV/ds`` per iterate (0 at the optimum)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax in axes:
        _style(ax)
    pj = _power([r["J"] for h in hists.values() for r in h["rows"]])
    pg = _power([r["J_prime"] for h in hists.values() for r in h["rows"]])
    for h in hists.values():
        _iterate_panel(axes[0], h, "J", "drag_se", 10.0**-pj)
        _iterate_panel(axes[1], h, "s")
        _iterate_panel(axes[2], h, "J_prime", "dD_ds_se", 10.0**-pg)
    axes[2].axhline(0.0, color=INK_SECONDARY, lw=0.8)
    labels = (
        ("(a) objective (common random numbers)", f"J = D - lam V  [1e{pj} N/m]  (+-1 SE)"),
        ("(b) Adam iterates (lr 0.05, from s = 1)", "y semi-axis factor s"),
        ("(c) adjoint gradient of J", f"J' = dD/ds - lam dV/ds  [1e{pg} N/m]  (+-1 SE)"),
    )
    for ax, (t, y) in zip(axes, labels, strict=True):
        ax.set_xlabel("iterate", fontsize=9, color=INK)
        ax.set_ylabel(y, fontsize=9, color=INK)
        ax.set_title(t, fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def squash_outputs():
    """The squash figures and report entries: the gradient gate, the free-molecular runs
    (completed ones, against the reference) and one figure per collisional set of runs."""
    report = {}
    gate = squash_gate()
    if gate:
        figure_squash_gate(gate, figure_path("squash", "squash_gradient_gate.png"))
        report["squash_gate"] = gate
        for e in gate:
            v = e["variants"]
            print(
                f"squash s={e['s']:g}: FD {e['fd']['mean']:.4g} ± {e['fd']['se']:.2g}, reference "
                f"{e['reference']:.4g}; adjoint - FD z: "
                + ", ".join(f"{n} {v[n]['minus_fd']['z']:+.1f}" for n in v)
            )
    hists = {t: h for t, h in load_squash_histories().items() if t.startswith("ellipse_")}
    report["squash_runs"] = {
        t: {"iterates": len(h["rows"]), "target": h["config"]["iterations"],
            "converged": bool(h.get("converged")),
            "s_last": h["rows"][-1]["s_next"] if h["rows"] else None,
            "hours_per_iterate": float(np.mean([r["seconds"] for r in h["rows"]])) / 3600 if h["rows"] else None}
        for t, h in hists.items()
    }  # fmt: skip
    sets = {}
    for t, h in hists.items():
        c = h["config"]
        sets.setdefault((c.get("knudsen", 0.0), c.get("mach", 0.0)), {})[t] = h
    for (kn, mach), runs in sorted(sets.items()):
        if kn == 0.0:
            done = {t: h for t, h in runs.items() if run_done(h)}
            if done:
                figure_squash_optimisation(
                    done, gate, figure_path("squash", "squash_optimisation.png")
                )
            continue
        regime = f"Kn {kn:g}, M {mach:g}" if mach > 0.0 else f"Kn {kn:g}, M 0.32"
        name = f"squashkn_kn{kn:g}" + (f"_M{mach:g}" if mach > 0.0 else "") + "_optimisation.png"
        figure_squash_set(runs, f"Squash, {regime}", figure_path("squash", name))
    return report


def optc_outputs(hist):
    """The optc drag figure (with the held-out drag if ``heldout_drag.json`` exists) and the
    per-run accounting; returns the report entries."""
    path_heldout = os.path.join(RESULTS_OPT, "heldout_drag.json")
    heldout = None
    if os.path.exists(path_heldout):
        with open(path_heldout) as fh:
            heldout = json.load(fh)
    figure_drag_optc(hist, figure_path("optc", "optc_drag.png"), heldout)
    final = figure_extent_optc(hist, figure_path("optc", "optc_body_extent.png"))
    return {"optc_runs": run_accounting(hist), "optc_heldout": heldout, "optc_extent": final}


def main():
    os.makedirs(FIGURES, exist_ok=True)
    report = {}
    merged = merge_seed_artifacts()
    for name, n in merged.items():
        print(f"merged {n} seed files -> {name}")
    report["merged"] = merged
    report["optimize"] = {}
    for label, suffix in (("fine", ""), ("coarse", "_coarse")):
        hist = load_histories(label)
        if not hist:
            continue
        figure_optimize(hist, figure_path("optc", f"optimize_convergence{suffix}.png"))
        if label == "coarse":
            report.update(optc_outputs(hist))
        runs = {
            tag: {
                "seeds_per_iterate": h["config"]["seeds_per_iterate"],
                "optimizer": h["config"]["optimizer"],
                "iterates": len(h["rows"]),
                "J_first": h["rows"][0]["objective"],
                "J_last": h["rows"][-1]["objective"],
                "seconds": sum(r["seconds"] for r in h["rows"]),
            }
            for tag, h in hist.items()
        }
        report["optimize"].update(runs)
        print("| run | S | optimiser | iterates | J first | J last | change | GPU s |")
        print("|---|---|---|---|---|---|---|---|")
        for tag, r in runs.items():
            print(
                f"| {tag} | {r['seeds_per_iterate']} | {r['optimizer']} | {r['iterates']} | {r['J_first']:.4g} | {r['J_last']:.4g} | {(r['J_last'] / r['J_first'] - 1) * 100:+.2f} % | {r['seconds']:.0f} |"
            )
    gs = gradient_seed_convergence(figure_path("optc", "gradient_seed_convergence.png"))
    if gs:
        report["gradient_seed_convergence"] = gs
        print(
            "gradient SE vs seeds:",
            dict(zip(gs["counts"], [f"{v:.2e}" for v in gs["se"]], strict=True)),
        )
    m2 = mode2_summary()
    if m2:
        summary, rows = m2
        figure_mode2(summary, rows, figure_path("gates", "mode2_gate.png"))
        report["mode2"] = summary
        print(
            f"mode2 ({summary['seeds']} seeds): adjoint {summary['gradient']['mean']:.4g} ± "
            f"{summary['gradient']['se']:.2g}, tally {summary['pathwise_tally']['mean']:.4g}, "
            f"FD {summary['fd'][0]['mean']:.4g} / {summary['fd'][1]['mean']:.4g}; "
            f"adjoint - FD z = {summary['gradient_minus_fd']['z']:+.1f}"
        )
    report.update(squash_outputs())
    sg = sphere_gate()
    if sg:
        report["sphere"] = sg
        print(
            f"sphere radius: ensemble {sg['ensemble']:.4g} ± {sg['ensemble_se']:.2g} vs FD {sg['fd']:.4g} ± {sg['fd_se']:.2g}, z = {sg['z']:+.2f} ({sg['seeds']} seeds)"
        )
    with open(os.path.join(HERE, "study_results.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(HERE, "study_results.json"))


if __name__ == "__main__":
    main()
