"""Level-set shape optimisation of the SPARTA circle: minimum windowed drag at fixed area.

    python3 -m examples.mesh.optimize --iterations 12 --seeds-per-iterate 2

The first optimisation case of the plan (phase E): the body is a C1 level set
(:mod:`jax_dsmc.mesh.levelset`) started from the circle; each iterate averages the
ensemble-adjoint shape terms of ``examples.ensemble_mesh.shape_terms`` over a few seeds
(``dJ/dcoef`` on the B-spline coefficients, pathwise + kernel score + silhouette +
window-edge timing + cut-volume, the ``along_pool_plane`` variant), smooths them on the coefficient grid (a Gaussian of ``--smooth`` knots: the
Sobolev / H1 gradient), restricts them to the band of coefficients whose knot lies within
``--band`` of the surface, and steps ``phi <- phi - eta g`` with ``eta`` chosen so that the
largest coefficient change -- the largest normal displacement, since ``|grad phi| = 1`` --
is ``--step``.  The area constraint is enforced exactly by a uniform shift of ``phi``
(bisection on :func:`levelset.body_volume`) after ``phi`` is reinitialised to a signed
distance; the cut-cell tables are rebuilt on the host at every iterate.  Common random numbers across
iterates (the same seeds every iterate) make the objective history comparable.

``--area-mode lagrangian`` replaces the projection (a step the gradient never saw) by an
augmented Lagrangian that is differentiated with the drag:

    L = D + nu (A - A0) + mu / 2 (A - A0)^2,   grad L = grad D + (nu + mu (A - A0)) grad A

(``A`` :func:`levelset.body_volume`, differentiable in ``coef``), the multiplier updated after
each step, ``nu <- nu + mu (A_new - A0)``.  ``nu`` starts at minus the area price of the
squash runs (:func:`area_price`: ``dD_ref/ds / dV/ds`` of the free-molecular ellipse at
``s = 0.35``, at this run's density and stream speed) and ``mu = --area-penalty * price / A0``
(a 1 % area error pulls as hard as the price at the default 100).  The squash cost
``D - price V`` alone (``mu = 0``, fixed ``nu``) is not usable here: the price is 4x the
circle's own ``dD/dA`` under uniform growth (2.4e-21 vs 6.0e-22 at R = 3, free-molecular),
and with ``D ~ R`` against ``A ~ R^2`` the body would grow without bound.

The multiplier update is unstable with this step: the step has a fixed size (``--step``,
GD normalised, Adam unit-scale), so the area overshoots whatever the pull, and the update
integrates the overshoot -- in the 2026-10-13 ``optab`` runs ``nu`` wound up to ~400x the
price within 100 iterates and the area swung between 4 % and 220 % of ``A0``, the steps
spent on the area instead of the drag.  ``--area-mode penalty`` keeps the same objective
with constant coefficients, ``nu = --area-linear * price`` (default 0) and ``mu`` as above:
a fixed function of the shape, nothing to wind up.  Its optimum misses ``A0`` by
``-(dD/dA + nu) / mu`` -- with ``nu = 0`` and the default ``mu`` about -0.25 % of ``A0``
(the circle's ``dD/dA`` is ~price / 4) -- and a fixed-size step jitters the area by up to
one step's area change around it (~0.05 m x the perimeter).

``--area-mode tangent`` holds the area exactly with a Lagrange multiplier per iterate
instead (projected gradient): the gradient loses its component along ``grad A``,
``g_t = g - nu grad A`` with ``nu = g . grad A / |grad A|^2`` (the KKT gradient on the
constant-area set; ``nu`` is recorded, ~``dD/dA``), and so does the final step direction
(Adam's elementwise scaling and the smoothing do not keep it orthogonal) -- the step then
changes the area only at second order, which the uniform shift of the projection restores
(:func:`tangent_component`).  Unlike ``projection``, whose shift undid whatever area the
step itself changed (a move the gradient never saw: Adam's first steps are a uniform
growth), the shift here is ``O(step^2)``.

``--workers W`` (cluster) evaluates the seeds of an iterate in ``W`` subprocesses, one
per visible GPU (``ROCR_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` round-robin), each a
re-invocation of this script in worker mode (``--eval-coef`` ...); the combination is the
same mean, so ``W`` changes the wall time only.

The run stops when it has converged -- the objective (the Lagrangian with
``--area-mode lagrangian``) of the last ``--converge-window`` iterates spread by at most
``--converge-tol`` of its mean (:func:`window_converged`) -- or at ``--iterations``,
whichever comes first; a resumed run that has converged does nothing.

Artifacts under ``examples/mesh/results/optimize/<tag>/``: ``history.json`` (one row per
iterate: objective mean and SE over the seeds, area, step, gradient norms, seconds),
``coef_<k>.npy`` and ``frames_<k>.npz`` (the field movie of the first seed's forward at
iterate ``k``: per-cell ``nrho``, ``u``, ``temp`` every ``--frames-every`` warm-up steps
and at the end, float32 -- ~1 MB per iterate on the finer circle; ``0`` disables);
``--plot`` draws ``figures/optc/optimize_<tag>.png`` (shape sequence, drag and area
histories) from them and ``examples.mesh.movie`` strings the frames into a GIF or an MP4
(fill of the flow around every iterate, then the shape change).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ellipsoid_reference as E
import forward as F

from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh.types import LevelSet

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "optimize")
from figure_paths import figure_path  # noqa: E402


def default_config():
    cfg = EM.default_config()
    cfg["body"] = "levelset"
    cfg.update(
        {
            # the cap; the run stops earlier when converged (window_converged): the objective
            # of the last `converge_window` iterates within `converge_tol` of their mean.  On
            # the optc histories 0.125 % stopped only the S1 / S2 GD runs (at 87 / 97 % of their
            # eventual drag reduction); 0.25 % also stopped S1 Adam (90 %) and S4 GD (80 %),
            # 1 % fired on plateaus (41-82 %)
            "iterations": 300,
            "converge_window": 5,
            "converge_tol": 0.00125,
            "seeds_per_iterate": 2,
            "seed_start": 32,
            "step": 0.05,
            "smooth": 2.0,
            "band": 1.0,
            "variant": "along_pool_plane",
            "neighbors": 32,
            # the window-edge timing term's band, in whole steps per edge (ensemble_mesh
            # .timing_band_setting): a 0.2-step band is far noisier for the same mean
            "timing_band": 10.0,
            "tag": "circle",
            "optimizer": "gd",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "workers": 1,
            "frames_every": 50,
            # the body's smallest distance to the box faces (keep_inside; 0 = unbounded,
            # the optc runs of 2026-09-23 which grew through the outflow face)
            "keepout": 0.0,
            # the area constraint: "projection" (uniform shift after the step, the runs up to
            # 2026-10-12), "lagrangian" (differentiated augmented Lagrangian, module doc; its
            # multiplier update winds up), "penalty" (the same objective, constant
            # coefficients: nu = area_linear * price, mu = area_penalty * price / A0) or
            # "tangent" (Lagrange multiplier per iterate: the step is projected onto the
            # constant-area set, the shift restores the second-order remainder)
            "area_mode": "projection",
            "area_penalty": 100.0,
            "area_linear": 0.0,
        }
    )
    return cfg


SQUASH_TARGET = 0.35  # optimize_squash's default s_target: where its area price is set


def area_price(cfg):
    """The squash runs' price of area (drag per unit area): ``dD_ref/ds / dV/ds`` of the
    free-molecular specular ellipse (``F.CASES["ellipse"]``) at ``s = SQUASH_TARGET``, at this
    run's gas density (``knudsen``) and stream speed (``mach``) -- ``optimize_squash.run``'s
    ``lam``, for any case.  2-D only."""
    c = F.CASES["ellipse"]
    nrho = F.NRHO * EM.knudsen_density(cfg)
    _d, dd = E.drag_2d(SQUASH_TARGET, c["radius"], nrho, E.MASS_AR, F.TEMP, EM.stream_speed(cfg))
    return float(dd / E.area_2d(SQUASH_TARGET, c["radius"])[1])


def _area_of_coef(coef, ls, lo, hi):
    return L.body_volume(dataclasses.replace(ls, coef=coef), lo, hi, 4)


def area_and_gradient(ls, lo, hi):
    """``(A, dA/dcoef)`` of :func:`levelset.body_volume` (the measure the constraint uses)."""
    a, g = jax.value_and_grad(_area_of_coef)(ls.coef, ls, lo, hi)
    return float(a), np.asarray(g)


def initial_levelset(cfg):
    grid, *_ = F.build(cfg["case"], "analytic", cfg["tracker"], "Ar", 0.0, cfg["radius_cells"])
    _coef, ls = EM.levelset_coef(jnp.float64(EM.nominal(cfg, "radius")), cfg, "radius", grid)
    return grid, ls


def _volume_at_shift(c, ls, lo, hi):
    return L.body_volume(dataclasses.replace(ls, coef=ls.coef + c), lo, hi, 4)


def project_area(ls: LevelSet, target: float, lo, hi, n_bisect: int = 40) -> tuple[LevelSet, float]:
    """The uniform shift ``c`` of ``phi`` (a uniform normal displacement) that restores the
    body area ``target``; bisection on the monotone ``body_volume``."""
    vol = jax.jit(functools.partial(_volume_at_shift, ls=ls, lo=lo, hi=hi))
    a, b = -ls.h * 4.0, ls.h * 4.0
    # body_volume decreases with c (phi up = body shrinks)
    for _ in range(n_bisect):
        mid = 0.5 * (a + b)
        if float(vol(mid)) > target:
            a = mid
        else:
            b = mid
    c = 0.5 * (a + b)
    return dataclasses.replace(ls, coef=ls.coef + c), c


def tangent_component(v: np.ndarray, g_area: np.ndarray) -> tuple[np.ndarray, float]:
    """``v`` without its component along ``g_area`` (coefficient space: ``g_area . dc`` is the
    first-order area change of a step ``dc``), and the removed coefficient."""
    nu = float(np.sum(v * g_area) / np.sum(g_area * g_area))
    return v - nu * g_area, nu


def smoothed_direction(grad: np.ndarray, ls: LevelSet, sigma: float, band: float) -> np.ndarray:
    """H1-smoothed gradient restricted to the coefficient band near the surface."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    mask = np.asarray(L.coefficient_band(ls, band))
    g = gaussian_filter(grad * mask, sigma=sigma, mode="nearest")
    return g * mask


def adam_direction(g, state, k, beta1, beta2, eps=1e-30):
    """Adam's normalised step from the (smoothed, band-restricted) gradient ``g``: the
    bias-corrected first moment over the root of the bias-corrected second moment,
    elementwise.  ``state = (m, v)`` arrays shaped like ``g`` (zeros at ``k = 0``)."""
    m, v = state
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m / (1.0 - beta1 ** (k + 1))
    v_hat = v / (1.0 - beta2 ** (k + 1))
    return m_hat / (np.sqrt(v_hat) + eps), (m, v)


FRAME_KEYS = ("step", "nrho", "u", "temp")


def window_converged(values, window: int, tol: float) -> bool:
    """True when the last ``window`` of ``values`` spread (max - min) by at most ``tol`` times
    their mean magnitude: the optimisers' stopping test (``--iterations`` is the cap)."""
    if window < 2 or len(values) < window:
        return False
    w = np.asarray(values[-window:], dtype=float)
    return bool(np.ptp(w) <= tol * abs(w.mean()))


def converged_objective(cfg, history) -> bool:
    """:func:`window_converged` on the rows' Lagrangian (the objective in projection mode)."""
    values = [r.get("lagrangian", r["objective"]) for r in history]
    return window_converged(values, cfg["converge_window"], cfg["converge_tol"])


PHASES = ("build", "warmup", "capture", "replay", "backward", "shape")
"""The per-seed phases of :func:`EM.shape_terms` whose wall-clock seconds the history keeps
(``phase_seconds``, summed over the seeds)."""


def seed_terms(cfg, ls, seeds, frames_for=None):
    """Per-seed coefficient gradient (pathwise + score), objective, hits and seconds of the
    seeds ``seeds`` (indices; key ``seed_start + i``), stacked.  The seed ``frames_for``
    (if in ``seeds``) also records the field movie of its forward (``cfg["frames_every"]``;
    keys ``frame_<name>``)."""
    grads, objs, hits, secs, stuck, phases = [], [], [], [], [], []
    out = {}
    for i in seeds:
        key = jax.random.key(cfg["seed_start"] + int(i))
        t0 = time.time()
        every = cfg["frames_every"] if frames_for is not None and int(i) == frames_for else 0
        all_terms, info, s = EM.shape_terms(
            cfg, key, ls, "radius", cfg["neighbors"], cfg["timing_band"], (cfg["variant"],), every
        )
        terms = all_terms[cfg["variant"]]
        # the window-edge timing term is part of the windowed objective's derivative
        grad = terms.pathwise + terms.score + terms.silhouette + terms.timing
        if terms.order is not None:
            grad = grad + terms.order  # the collision-order term (zero without collisions)
        if s["volume"] is not None:
            grad = grad + s["volume"]  # the cut-volume term (zero without collisions)
        grads.append(np.asarray(grad))
        objs.append(info["objective"])
        hits.append(info["hits"])
        stuck.append(info["stuck"])
        secs.append(time.time() - t0)
        phases.append([info.get(f"seconds_{name}", np.nan) for name in PHASES])
        if s["frames"] is not None:
            out.update({f"frame_{name}": s["frames"][name] for name in FRAME_KEYS})
    out.update(
        {
            "gradient": np.stack(grads),
            "objective": np.asarray(objs, dtype=float),
            "hits": np.asarray(hits, dtype=float),
            "stuck": np.asarray(stuck, dtype=float),
            "seconds": np.asarray(secs, dtype=float),
            "phase_seconds": np.asarray(phases, dtype=float),
        }
    )
    return out


def _combine(parts):
    grads = np.concatenate([p["gradient"] for p in parts])
    objs = np.concatenate([p["objective"] for p in parts])
    n = len(objs)
    frames = [p for p in parts if "frame_step" in p]
    return {
        "frames": {name: frames[0][f"frame_{name}"] for name in FRAME_KEYS} if frames else None,
        "gradient": grads.mean(axis=0),
        "gradient_seed_sd": float(grads.std(axis=0, ddof=1).mean()) if n > 1 else float("nan"),
        "objective": float(objs.mean()),
        "objective_se": float(objs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "hits": float(np.concatenate([p["hits"] for p in parts]).mean()),
        # particles deleted in concave pockets (SPARTA stuck), summed over the seeds
        "stuck": float(np.concatenate([p.get("stuck", np.zeros(1)) for p in parts]).sum()),
        "seconds": float(np.concatenate([p["seconds"] for p in parts]).sum()),
        "phase_seconds": dict(zip(PHASES, np.concatenate(
            [p.get("phase_seconds", np.full((len(p["seconds"]), len(PHASES)), np.nan))
             for p in parts]).sum(axis=0).tolist(), strict=True)),
    }  # fmt: skip


def averaged_terms(cfg, ls, seeds):
    """Mean over the seeds of the coefficient gradient and the objective, in this process;
    the field movie of the first seed."""
    return _combine([seed_terms(cfg, ls, seeds, seeds[0])])


def visible_devices():
    """The GPU ids this job may use: SLURM's ``ROCR_VISIBLE_DEVICES`` (AMD) /
    ``CUDA_VISIBLE_DEVICES`` (NVIDIA) or, unset, every device JAX sees."""
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(var):
            return os.environ[var].split(",")
    try:
        n = len(jax.devices("gpu"))
    except RuntimeError:
        n = 1
    return [str(i) for i in range(max(n, 1))]


def parallel_terms(cfg, ls, seeds, out: Path, k: int, argv):
    """:func:`averaged_terms` with the seeds split over ``cfg["workers"]`` subprocesses, one
    GPU each (``visible_devices()`` round-robin): every worker re-runs this script with the
    same arguments plus ``--eval-coef/--eval-seeds/--eval-out`` and writes its
    :func:`seed_terms` to an ``.npz``; the parent combines them.  The seeds a worker gets
    are fixed by its index, so the CRN property across iterates is kept."""
    workers = min(cfg["workers"], len(seeds))
    devices = visible_devices()
    coef_path = out / f"coef_{k}.npy"
    np.save(coef_path, np.asarray(ls.coef))
    procs = []
    for w in range(workers):
        mine = [str(i) for i in seeds[w::workers]]
        part = out / f"eval_{k}_w{w}.npz"
        env = dict(os.environ)
        dev = devices[w % len(devices)]
        for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            env[var] = dev
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        cmd = [
            sys.executable,
            "-m",
            "examples.mesh.optimize",
            *argv,
            "--workers",
            "1",
            "--eval-coef",
            str(coef_path),
            "--eval-out",
            str(part),
            "--eval-frames-for",
            str(seeds[0]),
            "--eval-seeds",
            *mine,
        ]
        log = open(out / f"eval_{k}_w{w}.log", "w")  # noqa: SIM115 -- closed after wait
        procs.append(
            (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, part)
        )
    parts = []
    for proc, log, part in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            raise RuntimeError(f"worker failed (exit {code}), see {log.name}")
        z = np.load(part)
        parts.append({key: z[key] for key in z.files})
        part.unlink()
    return _combine(parts)


def evaluate_worker(cfg, coef_path: str, seeds, out_path: str, frames_for=None):
    """Worker entry: the level set of ``coef_path`` on the seeds ``seeds`` -> ``out_path``
    (the field movie too if ``frames_for`` is one of them)."""
    _grid, ls = initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(coef_path)))
    np.savez(out_path, **seed_terms(cfg, ls, seeds, frames_for))


def surface_polyline(ls: LevelSet, lo, hi, n: int = 400):
    """Zero-set contour of ``phi`` (2-D) as a list of ``[M, 2]`` arrays, for the figures."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    phi = np.asarray(L.value(ls, pts)).reshape(n, n)
    fig, ax = plt.subplots()
    cs = ax.contour(xx, yy, phi, levels=[0.0])
    paths = [np.asarray(seg) for seg in cs.allsegs[0]]
    plt.close(fig)
    return paths


def keep_inside(ls: LevelSet, lo, hi, margin: float) -> tuple[LevelSet, int]:
    """The level set with the body kept ``margin`` away from every box face: ``phi`` at the
    knots floored at ``margin - d_face`` (``d_face`` the distance to the nearest face,
    negative outside the box), so the zero set cannot come nearer a face than ``margin``.
    The optc runs of 2026-09-23 had no such bound: the Adam bodies grew through the outflow
    face and one reached the inflow face (``inflow_shortfall``, S8 iterate 37).  Returns
    the new level set and the number of knots the floor moved (0: inactive)."""
    x = L.knots(ls)
    d_face = jnp.min(jnp.minimum(x - jnp.asarray(lo), jnp.asarray(hi) - x), axis=1)
    values = L.value(ls, x)
    floored = jnp.maximum(values, margin - d_face)
    moved = int(jnp.sum(floored > values))
    if not moved:
        return ls, 0
    coef = L.interpolate(floored.reshape(ls.coef.shape))
    return dataclasses.replace(ls, coef=coef), moved


def constrain(ls, area0, lo, hi, margin, shift, passes: int = 12):
    """Area and face margin together: alternate :func:`keep_inside` and
    :func:`project_area` (the floor removes area next to a face, the uniform shift restores
    it elsewhere) until the floor no longer moves a knot, ending on the floor (the margin is
    exact, the area to the last pass's residual).  Returns ``(ls, total shift, knots the
    first floor moved)``."""
    ls, first = keep_inside(ls, lo, hi, margin)
    moved = first
    for _ in range(passes):
        if not moved:
            break
        ls, extra = project_area(ls, area0, lo, hi)
        shift += extra
        ls, moved = keep_inside(ls, lo, hi, margin)
    return ls, shift, first


def after_step(stepped, mode, area0, lo, hi, keepout):
    """Reinitialise the stepped level set, then the constraints: ``projection`` restores the
    area by a uniform shift (with the face margin, :func:`constrain`); ``lagrangian`` /
    ``penalty`` only apply the face margin (the area is in the differentiated objective).  Returns
    ``(ls, area shift, knots the face floor moved)``."""
    # reinitialise first: the uniform shift of the projection keeps |grad phi| = 1
    new = L.reinitialise(stepped)
    if mode in ("lagrangian", "penalty"):
        clipped = 0
        if keepout > 0.0:
            new, clipped = keep_inside(new, lo, hi, keepout)
        return new, 0.0, clipped
    new, shift = project_area(new, area0, lo, hi)
    if keepout > 0.0:
        return constrain(new, area0, lo, hi, keepout, shift)
    return new, shift, 0


def run(cfg, out: Path, argv=()):  # noqa: PLR0912, PLR0915 -- one optimiser loop, kept whole
    out.mkdir(parents=True, exist_ok=True)
    hist_path = out / "history.json"
    saved = json.loads(hist_path.read_text()) if hist_path.exists() else None
    history = saved["rows"] if saved else []
    grid, ls = initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    if history:
        # coef_k.npy is the level set *entering* iterate k
        ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(out / f"coef_{len(history)}.npy")))
    else:
        np.save(out / "coef_0.npy", np.asarray(ls.coef))
    area0 = saved["area_target"] if saved else float(L.body_volume(ls, lo, hi, 4))
    seeds = list(range(cfg["seeds_per_iterate"]))
    t_start = time.time()
    k = len(history)
    adam_state = (np.zeros(np.asarray(ls.coef).shape), np.zeros(np.asarray(ls.coef).shape))
    if history and cfg["optimizer"] == "adam" and (out / "adam_state.npz").exists():
        z = np.load(out / "adam_state.npz")
        adam_state = (z["m"], z["v"])
    mode = cfg["area_mode"]
    if saved and saved["config"].get("area_mode", "projection") != mode:
        raise ValueError(f"{out} was run with area_mode {saved['config'].get('area_mode')}")
    if mode not in ("projection", "lagrangian", "penalty", "tangent"):
        raise ValueError(f"area_mode {mode}")
    price = area_price(cfg)
    mu = cfg["area_penalty"] * price / area0
    nu = history[-1]["area_multiplier_next"] if history and mode == "lagrangian" else -price
    if mode == "penalty":
        nu = cfg["area_linear"] * price
    while k < cfg["iterations"] and not converged_objective(cfg, history):
        t0 = time.time()
        # the mesh of the current iterate (an uncovered box raises here; split cells get
        # sub-cells since 2026-09-23)
        cut.build_mesh(
            ls, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=cfg["radius_cells"]
        )
        if cfg["workers"] > 1:
            avg = parallel_terms(cfg, ls, seeds, out, k, argv)
        else:
            avg = averaged_terms(cfg, ls, seeds)
        g = avg["gradient"]
        area_err = 0.0
        if mode in ("lagrangian", "penalty"):
            area_now, g_area = area_and_gradient(ls, lo, hi)
            area_err = area_now - area0
            g = g + (nu + mu * area_err) * g_area
        elif mode == "tangent":
            area_now, g_area = area_and_gradient(ls, lo, hi)
            area_err = area_now - area0
            # the KKT gradient: Adam's moments then track the constrained gradient too
            g, nu = tangent_component(g, g_area)
        direction = smoothed_direction(g, ls, cfg["smooth"], cfg["band"] + ls.h)
        if cfg["optimizer"] == "adam":
            # Adam's elementwise-normalised step has unit scale: `step` is its learning
            # rate, i.e. the largest coefficient change per iterate, like the GD step
            direction, adam_state = adam_direction(
                direction, adam_state, k, cfg["adam_beta1"], cfg["adam_beta2"]
            )
            np.savez(out / "adam_state.npz", m=adam_state[0], v=adam_state[1])
            # the elementwise normalisation undoes the H1 smoothing (at k = 0 it is exactly
            # sign(g): +-step jumps where g changes sign, overhangs a particle bounces in
            # until the tracker's event budget runs out -- `wall_recrossed` on every optc Adam
            # run, 2026-09-23), so the step itself is smoothed again
            direction = smoothed_direction(direction, ls, cfg["smooth"], cfg["band"] + ls.h)
        if mode == "tangent":
            # the step itself: first-order area preserving
            direction, _ = tangent_component(direction, g_area)
        scale = float(np.abs(direction).max())
        eta = cfg["step"] / max(scale, 1e-300)
        stepped = dataclasses.replace(ls, coef=ls.coef - eta * jnp.asarray(direction))
        new, shift, clipped = after_step(stepped, mode, area0, lo, hi, cfg["keepout"])
        area = float(L.body_volume(new, lo, hi, 4))
        nu_next = nu + mu * (area - area0) if mode == "lagrangian" else nu
        # predicted first-order decrease: g . delta coef (before the area projection)
        predicted = float(-eta * np.sum(g * direction))
        row = {
            "iterate": k,
            "optimizer": cfg["optimizer"],
            "objective": avg["objective"],
            "objective_se": avg["objective_se"],
            "gradient_seed_sd": avg["gradient_seed_sd"],
            "hits": avg["hits"],
            "stuck": avg["stuck"],
            "area": float(L.body_volume(ls, lo, hi, 4)),
            "area_after_step": area,
            "grad_norm": float(np.linalg.norm(g)),
            "direction_max": scale,
            "eta": eta,
            "area_shift": shift,
            "keepout_knots": clipped,
            "predicted_decrease": predicted,
            "seconds": time.time() - t0,
            "phase_seconds": avg["phase_seconds"],
        }
        if mode == "tangent":
            row.update(area_multiplier=nu, area_error=area_err)
        if mode in ("lagrangian", "penalty"):
            row.update(
                area_multiplier=nu,
                area_multiplier_next=nu_next,
                area_penalty_mu=mu,
                area_error=area_err,
                lagrangian=avg["objective"] + nu * area_err + 0.5 * mu * area_err**2,
            )
        history.append(row)
        np.save(out / f"coef_{k + 1}.npy", np.asarray(new.coef))
        if avg["frames"] is not None:
            np.savez(out / f"frames_{k}.npz", **avg["frames"])
        hist_path.write_text(
            json.dumps(
                {
                    "config": cfg,
                    "backward_basis": ensemble_backward.VIRTUAL_BASIS,
                    "area_target": area0,
                    "converged": converged_objective(cfg, history),
                    "rows": history,
                    "elapsed": time.time() - t_start,
                },
                indent=1,
            )
            + "\n"
        )
        print(
            f"iterate {k}: J = {row['objective']:.5g} ± {row['objective_se']:.2g}  area {row['area']:.4f} "
            f"-> {area:.4f} (shift {shift:+.4f})  |g| {row['grad_norm']:.3g}  predicted dJ {predicted:+.3g}  "
            f"hits {row['hits']:.0f}  stuck {row['stuck']:.0f}  {row['seconds']:.0f}s",
            flush=True,
        )
        ls, nu = new, nu_next
        k += 1
    if converged_objective(cfg, history):
        print(f"converged after {len(history)} iterates", flush=True)


def plot(cfg, out: Path, path_fig: str):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    hist = json.loads((out / "history.json").read_text())
    rows = hist["rows"]
    grid, ls0 = initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    cmap = plt.get_cmap("viridis")
    shown = sorted({0, n, *range(0, n + 1, max(1, n // 6))})
    for k in shown:
        p = out / f"coef_{k}.npy"
        if not p.exists():
            continue
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(p)))
        for j, seg in enumerate(surface_polyline(ls, lo, hi)):
            axes[0].plot(
                seg[:, 0],
                seg[:, 1],
                color=cmap(k / max(n, 1)),
                lw=1.2,
                label=f"iterate {k}" if j == 0 else None,
            )
    axes[0].set_aspect("equal")
    axes[0].set_xlim(lo[0], hi[0])
    axes[0].set_ylim(lo[1], hi[1])
    axes[0].set_title("zero set of phi (flow from the left)", fontsize=9)
    axes[0].legend(fontsize=6)
    it = [r["iterate"] for r in rows]
    axes[1].errorbar(
        it,
        [r["objective"] for r in rows],
        yerr=[r["objective_se"] for r in rows],
        fmt="o-",
        ms=3,
        capsize=2,
    )
    axes[1].set_xlabel("iterate")
    axes[1].set_ylabel("windowed drag J (N/m)")
    axes[1].set_title(f"drag, {cfg['seeds_per_iterate']} CRN seeds per iterate", fontsize=9)
    axes[2].plot(it, [r["area"] for r in rows], "o-", ms=3, label="area")
    axes[2].axhline(hist["area_target"], color="0.5", ls="--", label="target")
    axes[2].set_xlabel("iterate")
    axes[2].set_ylabel("body area")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    print("wrote", path_fig)


PRE_CUBIC_BASIS = "cell:quad:adapt1.5:align0.9:both"
"""The backward basis of the histories written before ``backward_basis`` was recorded
(2026-10-11 .. 10-12; older finished runs used the per-query ``quadratic_full``)."""


def resumed_basis(out: Path, asked):
    """The backward basis of run ``out``: a resumed run keeps the one it started with (its
    history's ``backward_basis``, :data:`PRE_CUBIC_BASIS` if it predates the record) and
    asking for another raises; a new run takes ``asked`` or the shipped default."""
    path = out / "history.json"
    if not path.exists():
        return asked or ensemble_backward.CELL_BASIS
    basis = json.loads(path.read_text()).get("backward_basis", PRE_CUBIC_BASIS)
    if asked not in (None, basis):
        raise ValueError(f"{out} was run with backward basis {basis}, not {asked}")
    return basis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cfg = default_config()
    for k, v in cfg.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--pullback", default="virtual", choices=("virtual", "sequential"))
    # the SPARTA circle is collisionless (Natt = 0): the virtual pull-back's partners do no
    # work there, so the cheap setting of the phase-D artifacts (1 x 2) is the default
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--partners", type=int, default=2)
    p.add_argument(
        "--backward-basis",
        default=None,
        help="field fit of the virtual pull-back (jax_dsmc.cell_fit; quadratic_full: per query); "
        "default: a resumed run's own, else ensemble_backward.CELL_BASIS",
    )
    p.add_argument("--plot", action="store_true", help="only draw the figure from the artifacts")
    # worker mode of `parallel_terms` (internal): evaluate one level set on some seeds
    p.add_argument("--eval-coef", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-seeds", type=int, nargs="*", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-frames-for", type=int, default=None, help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    cfg = {k: getattr(args, k) for k in cfg}
    out = Path(RESULTS) / cfg["tag"]
    basis = resumed_basis(out, args.backward_basis)
    ensemble_backward.configure(args.pullback, args.substeps, args.partners, "staged", basis)
    if args.eval_coef is not None:
        evaluate_worker(cfg, args.eval_coef, args.eval_seeds, args.eval_out, args.eval_frames_for)
        return
    if args.plot:
        plot(cfg, out, figure_path("optc", f"optimize_{cfg['tag']}.png"))
        return
    run(cfg, out, argv)


if __name__ == "__main__":
    main()
