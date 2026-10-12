"""Aspect-ratio optimisation with an analytic reference: a circle (2-D ``ellipse`` case) or
sphere (3-D ``spheroid`` case) squashed in ``y`` to semi-axes ``(R, s R[, R])``, cost

    J(s) = D(s) - lam V(s)

(``D`` the windowed drag, ``V`` the body's volume -- its area in 2-D -- ``pi R^2 s`` or
``4/3 pi R^3 s``).  Every area measure grows with ``s`` like the drag, so the volume enters
with a minus sign (area is rewarded, drag is paid for); the optimum is where
``dD/ds = lam dV/ds``.  The free-molecular reference ``D_ref(s)``
(:mod:`ellipsoid_reference`, half-range stream as the inlet produces it) fixes ``lam`` so
that the reference optimum is ``--s-target`` (default 0.35, where ``D`` is strongly
curved; the drag is nearly linear in ``s`` above ~0.6, so an optimum there would be
ill-conditioned).

    python3 -m examples.mesh.optimize_squash --iterations 25 --seeds-per-iterate 4
    python3 -m examples.mesh.optimize_squash --knudsen 0.5 --a-max 64 --seeds-per-iterate 4

Each iterate evaluates ``dD/ds`` by the ensemble adjoint (``examples.ensemble_mesh``
``shape_terms`` on the level-set body, one ``grad_x lambda`` variant, chained to ``s``
through :func:`ensemble_mesh.control_gradient`), adds the exact ``-lam dV/ds``, and takes
an Adam step ``s <- s - lr m_hat / sqrt(v_hat)`` (``optimize.adam_direction``; its step has
unit scale, so ``--learning-rate`` is the largest change of ``s`` per iterate).  Common
random numbers across iterates.

``--knudsen K`` (``examples.ensemble_mesh.knudsen_density``) raises the gas density, at the
same particle count, to the free-stream Knudsen number ``K`` on the body diameter ``2 R``:
the transitional regime, where NTC collides on every step.  ``lam`` is still fixed by the
free-molecular reference at that density (the same problem with the same weight on the
volume), but the reference no longer describes the drag: its optimum is not the answer
there, and the run maps where the collisional optimum moves to.  The collision pull-back is
then the shipped virtual one (``--substeps 4 --partners 8``; ``1 x 2`` when collisionless).
``--mach M`` sets the stream to ``M`` times the free-stream sound speed (the deck's 100 m/s
is M 0.32); a hypersonic stream needs a shorter ``--dt`` and more ``--slots`` (the study's
M 10 settings in ``examples/mesh/cluster/study.py``).

Artifacts: ``results/optimize_squash/<tag>/history.json`` (per iterate ``s``, the drag and
gradient means and SEs, Adam's moments, the reference ``D_ref``, ``dD_ref/ds`` and
``J'_ref`` at the same ``s``); ``--plot`` draws ``figures/squash/optimize_squash_<tag>.png``.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ellipsoid_reference as E
import forward as F
from optimize import adam_direction

from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "optimize_squash")
from figure_paths import figure_path  # noqa: E402


def default_config():
    cfg = EM.default_config()
    cfg.update(
        {
            "case": "ellipse",
            "body": "levelset",
            "fnum_scale": 1.0,
            "levelset_h": 0.0625,
            "levelset_samples": 16,
            "grazing": 0.05,
            # the stratified fit biases the mode2 gradient (2026-09-25 diagnostic); the
            # unstratified pool matched FD there
            "value_class": "none",
            "s0": 1.0,
            "s_target": 0.35,
            # Adam (the only optimiser here): `learning_rate` is the largest change of s per
            # iterate; 0.65 from the circle to s* takes ~13 iterates at full speed
            "learning_rate": 0.05,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "iterations": 25,
            "seeds_per_iterate": 4,
            "seed_start": 32,
            # pooled one-sided fit without the nodes behind the hit's tangent plane, plus the
            # along-flight identity (2026-10-02 gates: within 1 % of FD at s = 0.35 .. 1)
            "variant": "along_pool_plane",
            "neighbors": 32,
            # the window-edge timing term's band, in whole steps per edge (ensemble_mesh
            # .timing_band_setting): a 0.2-step band is far noisier for the same mean
            "timing_band": 10.0,
            # the field movie of the first seed's forward per iterate (`frames_<k>.npz`, a
            # snapshot every `frames_every` warm-up steps; 0: none), for examples.mesh.movie
            "frames_every": 50,
            "tag": "",
        }
    )
    return cfg


def reference(cfg, s):
    """``(D_ref, dD_ref/ds, d2D_ref/ds2, dV/ds)`` of the case at ``s``."""
    c = F.CASES[cfg["case"]]
    nrho = F.NRHO * EM.knudsen_density(cfg)
    args = (c["radius"], nrho, E.MASS_AR, F.TEMP, EM.stream_speed(cfg))
    fn = E.drag_2d if c["dim"] == 2 else E.drag
    d, dd = fn(s, *args)
    h = 1e-4
    d2 = (fn(s + h, *args)[1] - fn(s - h, *args)[1]) / (2 * h)
    dv = E.area_2d(s, c["radius"])[1] if c["dim"] == 2 else E.volume(s, c["radius"])[1]
    return d, dd, d2, dv


def _se(a):
    a = np.asarray(a, dtype=float)
    return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")


def seed_gradient(cfg, s, seed):
    """One seed's windowed drag and ``dD/ds`` at ``s`` (the chosen variant: pathwise +
    score + silhouette, the runner's reported gradient); seed 0 also records the field
    movie of its forward (``info["frames"]``)."""
    run = dict(cfg, squash=float(s))
    all_terms, info, st = EM.shape_terms(
        run,
        jax.random.key(cfg["seed_start"] + seed),
        None,
        "squash_y",
        cfg["neighbors"],
        cfg["timing_band"],
        (cfg["variant"],),
        cfg.get("frames_every", 0) if seed == 0 else 0,
    )
    info["frames"] = st["frames"]
    terms = all_terms[cfg["variant"]]
    pick = functools.partial(EM.control_gradient, run, "squash_y", st["grid"])
    parts = {k: pick(getattr(terms, k)) for k in ("pathwise_tally", "pathwise_x", "pathwise_v")}
    parts["volume"] = 0.0 if st["volume"] is None else pick(st["volume"])
    parts["timing"] = pick(terms.timing)
    parts["order"] = 0.0 if terms.order is None else pick(terms.order)
    grad = pick(terms.pathwise) + pick(terms.score) + pick(terms.silhouette) + parts["volume"]
    grad = grad + parts["timing"] + parts["order"]
    return info["objective"], grad, parts, info


def run(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "history.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    rows = saved["rows"] if saved else []
    _d, dd_t, _d2, dv = reference(cfg, cfg["s_target"])
    lam = dd_t / dv  # reference optimum at s_target
    s = rows[-1]["s_next"] if rows else cfg["s0"]
    moments = (rows[-1]["adam_m"], rows[-1]["adam_v"]) if rows else (0.0, 0.0)
    while len(rows) < cfg["iterations"]:
        t0 = time.time()
        per = [seed_gradient(cfg, s, i) for i in range(cfg["seeds_per_iterate"])]
        drag = np.array([p[0] for p in per])
        grad = np.array([p[1] for p in per])
        d_ref, dd_ref, _, _ = reference(cfg, s)
        jprime = float(grad.mean()) - lam * dv
        direction, moments = adam_direction(
            jprime, moments, len(rows), cfg["adam_beta1"], cfg["adam_beta2"]
        )
        s_next = s - cfg["learning_rate"] * float(direction)
        row = {
            "iterate": len(rows),
            "s": s,
            "drag": float(drag.mean()),
            "drag_se": _se(drag),
            "dD_ds": float(grad.mean()),
            "dD_ds_se": _se(grad),
            "parts": {k: float(np.mean([p[2][k] for p in per])) for k in per[0][2]},
            "J": float(drag.mean()) - lam * (dv * s),
            "J_prime": jprime,
            "D_ref": d_ref,
            "dD_ref_ds": dd_ref,
            "J_prime_ref": dd_ref - lam * dv,
            "s_next": s_next,
            "adam_m": float(moments[0]),
            "adam_v": float(moments[1]),
            "hits": float(np.mean([p[3]["hits"] for p in per])),
            "stuck": float(np.sum([p[3]["stuck"] for p in per])),
            "seconds": time.time() - t0,
        }
        if per[0][3]["frames"] is not None:
            np.savez(out / f"frames_{row['iterate']}.npz", **per[0][3]["frames"])
        rows.append(row)
        path.write_text(
            json.dumps(
                {
                    "config": cfg,
                    "lam": lam,
                    "knudsen_density": EM.knudsen_density(cfg),
                    "s_star_ref": cfg["s_target"],
                    "rows": rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(
            f"iterate {row['iterate']}: s {s:.4f}  D {row['drag']:.4e} (ref {d_ref:.4e})  "
            f"dD/ds {row['dD_ds']:.4e} ± {row['dD_ds_se']:.1e} (ref {dd_ref:.4e})  "
            f"J' {jprime:+.3e}  -> s {s_next:.4f}  {row['seconds']:.0f}s",
            flush=True,
        )
        s = s_next


def backfill_frames(out: Path):
    """``frames_<k>.npz`` for every iterate of a finished history that has none: the forward
    of the iterate's first seed (key ``seed_start``, the same as the run's) at its ``s``,
    with a snapshot every ``frames_every`` warm-up steps -- forward only, no gradient."""
    h = json.loads((out / "history.json").read_text())
    cfg = h["config"]
    every = cfg.get("frames_every") or 50
    for row in h["rows"]:
        path = out / f"frames_{row['iterate']}.npz"
        if path.exists():
            continue
        t0 = time.time()
        run_cfg = dict(cfg, squash=float(row["s"]))
        key = jax.random.key(cfg["seed_start"])
        *_, setup = EM.forward(run_cfg, "squash_y", float(row["s"]), key, False, None, every)
        np.savez(path, **setup["frames"])
        print(f"frames {row['iterate']}: s {row['s']:.4f}  {time.time() - t0:.0f}s", flush=True)


def _unit(values):
    """``(factor, label)``: the power of ten that brings ``values`` to order one (the drag is
    ~1e-20 N/m in the deck's gas, ~1e-2 at Kn 0.5 with fnum scaled by the density)."""
    top = max(abs(float(v)) for v in values)
    power = math.floor(math.log10(top)) if top > 0 else 0
    return 10.0**-power, f"1e{power}"


def _finish(fig, axes, h, free, path_fig):
    """Spines, title and file of :func:`plot`."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cfg = h["config"]
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    regime = f", Kn = {cfg['knudsen']:g}" if cfg.get("knudsen", 0.0) > 0.0 else ""
    regime += f", M = {cfg['mach']:g}" if cfg.get("mach", 0.0) > 0.0 else ""
    optimum = f", free-molecular optimum s* = {h['s_star_ref']}" if free else ""
    fig.suptitle(
        f"{cfg['case']} squashed in y{regime}, Adam lr {cfg['learning_rate']:g}: "
        f"J = D - lam V{optimum}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def plot(out: Path, path_fig: str):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    h = json.loads((out / "history.json").read_text())
    cfg, rows, lam = h["config"], h["rows"], h["lam"]
    it = np.array([r["iterate"] for r in rows])
    s = np.array([r["s"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    blue, orange, grey = "#2a78d6", "#eb6834", "#8a8a86"
    # the free-molecular reference describes only the collisionless runs; a collisional
    # run's drag behaves differently and is shown on its own
    free = cfg.get("knudsen", 0.0) <= 0.0
    ax = axes[0]
    ax.plot(it, s, "o-", color=blue, ms=4, label="iterate")
    if free:
        ax.axhline(h["s_star_ref"], color=grey, ls="--", lw=1, label="reference optimum")
    ax.set_xlabel("iterate")
    ax.set_ylabel("y semi-axis factor s")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    grid = np.linspace(min(0.15, s.min()), max(1.0, s.max()), 200)
    k, unit = _unit([r["dD_ds"] for r in rows])
    if free:
        ref = np.array([reference(cfg, x)[1] for x in grid])
        ax.plot(grid, ref * k, color=grey, lw=1.5, label="free-molecular reference dD/ds")
    ax.errorbar(
        s, [r["dD_ds"] * k for r in rows], yerr=[2 * np.nan_to_num(r["dD_ds_se"]) * k for r in rows],
        fmt="o", color=blue, ms=4, capsize=2, label="adjoint dD/ds (+-2 SE)",
    )  # fmt: skip
    ax.axhline(lam * reference(cfg, 1.0)[3] * k, color=orange, lw=1, ls=":", label="lam dV/ds")
    ax.set_xlabel("s")
    ax.set_ylabel(f"dD/ds  [{unit}]")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    k, unit = _unit([r["drag"] for r in rows])
    ax.plot(s, [r["drag"] * k for r in rows], "o", color=blue, ms=4, label="in-box drag")
    if free:
        ax.plot(
            grid, [reference(cfg, x)[0] * k for x in grid], color=grey, lw=1.5, label="reference"
        )
    ax.set_xlabel("s")
    ax.set_ylabel(f"windowed drag  [{unit} N/m]")
    ax.legend(fontsize=7, frameon=False)
    _finish(fig, axes, h, free, path_fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cfg = default_config()
    for k, v in cfg.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--pullback", default="virtual", choices=("virtual", "sequential"))
    # default: 1 x 2 collisionless (the virtual pull-back's partners do no work there,
    # optimize.py), the shipped 4 x 8 with --knudsen
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--partners", type=int, default=None)
    p.add_argument(
        "--backward-basis",
        default=ensemble_backward.CELL_BASIS,
        help="field fit of the virtual pull-back (jax_dsmc.cell_fit; quadratic_full: per query)",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--frames", action="store_true", help="re-run the forwards of a finished run for its movie"
    )
    args = p.parse_args(argv)
    cfg = {k: getattr(args, k) for k in cfg}
    collisional = cfg["knudsen"] > 0.0
    substeps = args.substeps or (4 if collisional else 1)
    partners = args.partners or (8 if collisional else 2)
    ensemble_backward.configure(args.pullback, substeps, partners, "staged", args.backward_basis)
    regime = f"_kn{cfg['knudsen']:g}" if cfg["knudsen"] > 0.0 else ""
    regime += f"_M{cfg['mach']:g}" if cfg["mach"] > 0.0 else ""
    # the collisional runs depend on the backward's field fit (the free-molecular ones do
    # not: no collision term): its tag keeps a per-cell run out of a per-query run's
    # directory (the kn0.5 runs of 2026-10-01 are the per-query fit's)
    regime += ensemble_backward.basis_tag(args.backward_basis) if collisional else ""
    tag = cfg["tag"] or (
        f"{cfg['case']}{regime}_S{cfg['seeds_per_iterate']}_st{cfg['s_target']:g}_adam"
    )
    out = Path(RESULTS) / tag
    if args.plot:
        plot(out, figure_path("squash", f"optimize_squash_{tag}.png"))
        return
    if args.frames:
        backfill_frames(out)
        return
    run(cfg, out)


if __name__ == "__main__":
    main()
