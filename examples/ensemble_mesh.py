"""Shape gradients of the embedded-body cases (plan section 5): analytic circle first.

    python3 -m examples.ensemble_mesh --case circle --parameter radius --seeds 64

The SPARTA circle duplicate of ``examples/mesh/forward.py`` (argon, ``vstream`` 100 m/s,
specular body, inflow at ``x = 0``) at ``fnum x --fnum-scale`` (fewer particles per run),
warmed up for ``--warmup`` uncaptured steps, then ``--steps`` captured steps of which the
last ``--window`` form the objective: the mean drag on the body in N/m,

    J = fnum / (W dt) * sum_{k in window} sum_{hits} m (v_in - v_out) . e_x,

a per-hit reward (the traced replay, :func:`jax_dsmc.shape.replay`) whose pathwise part
(the specular ``v_out`` depends on the normal) enters through the tally cotangent.
Controls: ``radius`` (analytic ``dF/dr = F/r`` for a specular cylinder in free-molecular
flow is the reference for the value; blockage is what it is at this box), ``centre_x``,
``mode2`` (a ``cos 2 theta`` normal displacement of the surface, level set only), ``width``
(the half-width of the ``plate`` case's capsule) and ``squash_y`` (the ``y`` semi-axis
factor ``s`` of the ``ellipse`` / ``spheroid`` cases, evaluated at ``--squash``; reference
``examples/mesh/ellipsoid_reference.py``).
Bodies (``--body``): the analytic circle, or the **level set** interpolating its signed
distance (``jax_dsmc.mesh.levelset``, LS1/LS2 of the plan): the shape VJP then lands on
the B-spline coefficients and is chained through ``coef(theta)`` -- for ``radius`` that is
``d coef / d r = -1`` at every knot, so the level-set radius gradient must reproduce the
analytic one (LS2 replaces the polygon's GE gate).
Estimator: value-form ensemble backward + :func:`jax_dsmc.shape.gradient`.  Arbiter:
common-random-number central differences at several ``h`` on the same seed (body rebuilt
on the host, warm-up included).  Artifacts in the ``ensemble_couette`` row format.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh"))
import forward as F

from examples.ensemble_inlet_shock import device_backward
from examples.gradient_bench import RESULTS
from jax_dsmc import ensemble_backward, inflow, initialize, sample, shape, simulate, sort
from jax_dsmc import species_data as sd
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import n_cartesian, num_cells
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as levelset_mod
from jax_dsmc.mesh.types import AnalyticBody
from jax_dsmc.state import FlowParams
from jax_dsmc.validation import require_valid

PARAMETERS = ("radius", "centre_x", "mode2", "width", "squash_y")
BODIES = ("analytic", "levelset")
# the replay's continuous error: round-off of one arithmetic compiled twice, amplified by
# grazing reflections on concave walls to ~1e-12 on optimised shapes; a discrete divergence
# (a flipped outcome) reports 1 (`shape._mismatch`), so 1e-9 separates the two widely
REPLAY_TOL = 1e-9


def default_config():
    return {
        "case": "circle",
        "fnum_scale": 4.0,
        "warmup": 700,
        "steps": 300,
        "window": 200,
        "radius_cells": 1,
        # SPARTA's move loop: pass 0 over all, later passes on the particles with an event
        "tracker": "lane",
        "body": "analytic",
        "levelset_h": 0.125,
        "levelset_samples": 8,
        "acc": 0.0,
        "silhouette_band": 0.0,
        "grazing": 0.0,
        "grid_scale": 1,
        "value_class": "hit_ahead",
        # the y semi-axis factor s of the `ellipse` / `spheroid` cases (semi-axes R, s R[, R]):
        # the evaluation point of the control `squash_y`
        "squash": 1.0,
        # the free-stream Knudsen number lambda / (2 R) on the case's body diameter (R its
        # radius; :func:`knudsen_density`): 0 keeps the deck's nrho = 1 (Kn ~ 1e18, no
        # collisions).  The density and fnum are scaled alike, so the particle count is the
        # same; `a_max` (0: forward.py's 4) bounds the NTC attempts per cell and step
        "knudsen": 0.0,
        "a_max": 0,
        # the free-stream Mach number (:func:`stream_speed`; 0: the deck's 100 m/s, M 0.32),
        # with the time step `dt` (0: the deck's 1e-4 s) and the slot count `slots` (0: the
        # case's capacity / fnum_scale) a faster, fuller box needs
        "mach": 0.0,
        "dt": 0.0,
        "slots": 0,
        # the captured record: "lean" drops v_post (rebuilt bitwise from v and the pair
        # record) and born_v (host adjoint only) -- 48 of ~122 bytes per slot and step
        "record": "lean",
    }


def _mode_shape(x, centre):
    """``-cos(2 theta)`` about ``centre``: the level-set increment of a unit outward
    normal displacement ``cos(2 theta)`` (``delta phi = -delta_n``)."""
    rel = x - jnp.asarray(centre)[None, :]
    theta = jnp.arctan2(rel[:, 1], rel[:, 0])
    return -jnp.cos(2.0 * theta)


def _capsule_distance(centre, half_width, rho, x):
    """Signed distance to a stadium: the segment ``centre +- half_width e_y`` thickened by
    ``rho`` (a plate normal to the stream with rounded ends), ``f64[N]``."""
    rel = x - centre[None, :]
    t = jnp.clip(rel[:, 1], -half_width, half_width)
    foot = jnp.stack((jnp.zeros_like(t), t), axis=1)
    d2 = jnp.sum((rel - foot) ** 2, axis=1)
    # knots on the segment itself: sqrt(0) has no gradient (the analytic.py:53 trap)
    dist = jnp.where(d2 > 0.0, jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0)), 0.0)
    return dist - rho


def _ellipsoid_distance(centre, semi, x):
    """``(q - 1) / |grad q|`` with ``q = |(x - centre) / semi|``: zero on the ellipsoid, unit
    gradient on it and ``d phi / d semi`` there equal to minus the surface's normal speed (the
    exact shape derivative), a Lipschitz-bounded signed pseudo-distance elsewhere, ``f64[N]``.
    Written as ``(q^2 - q) / |(x - centre) / semi^2|`` so the centre (``q = 0``) is finite."""
    rel = (x - centre[None, :]) / semi[None, :]
    q2 = jnp.sum(rel * rel, axis=1)
    g2 = jnp.sum((rel / semi[None, :]) ** 2, axis=1)
    inner = g2 > 1e-24
    q = jnp.sqrt(jnp.where(inner, q2, 1.0))
    g = jnp.sqrt(jnp.where(inner, g2, 1.0))
    return jnp.where(inner, (q2 - q) / g, -jnp.min(semi))


def levelset_coef(theta, cfg, parameter, grid):
    """Coefficients of the level-set body at control value ``theta`` (JAX, differentiable:
    the chain ``d coef / d theta`` of the LS2 gate is its VJP)."""
    c = F.CASES[cfg["case"]]
    centre = jnp.asarray(c["centre"], dtype=jnp.float64)
    radius = jnp.float64(c["radius"])
    if parameter == "radius":
        radius = theta
    elif parameter == "centre_x":
        centre = centre.at[0].set(theta)
    elif parameter not in ("mode2", "width", "squash_y"):
        raise ValueError(f"parameter must be one of {PARAMETERS}")
    h = cfg["levelset_h"]
    lo, shape = levelset_mod.grid_for_box(grid.lo, grid.hi, h)
    if c.get("kind") == "ellipse":
        # the aspect-ratio test: semi-axes (R, s R[, R]), s the control `squash_y`
        if parameter != "squash_y":
            raise ValueError("the ellipse / spheroid cases take the control squash_y")
        dim = len(c["centre"])
        semi = jnp.full((dim,), radius).at[1].set(radius * theta)
        template = levelset_mod.LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(lo), h=float(h))
        sdf = _ellipsoid_distance(centre, semi, levelset_mod.knots(template))
        ls = levelset_mod.from_values(sdf.reshape(shape), lo, h, n_sample=cfg["levelset_samples"])
        return ls.coef, ls
    if parameter == "squash_y":
        raise ValueError("squash_y is the control of the ellipse / spheroid cases")
    if c.get("kind") == "capsule":
        # the GP plate: half-width is the control, thickness rho = the case's radius
        half_width = theta if parameter == "width" else jnp.float64(c["half_width"])
        template = levelset_mod.LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(lo), h=float(h))
        sdf = _capsule_distance(centre, half_width, radius, levelset_mod.knots(template))
        ls = levelset_mod.from_values(sdf.reshape(shape), lo, h, n_sample=cfg["levelset_samples"])
        return ls.coef, ls
    if parameter == "width":
        raise ValueError("width is the capsule's control")
    ls = levelset_mod.from_analytic(centre, radius, lo, h, shape, n_sample=cfg["levelset_samples"])
    coef = ls.coef
    if parameter == "mode2":
        mode = levelset_mod.interpolate(_mode_shape(levelset_mod.knots(ls), centre).reshape(shape))
        coef = coef + theta * mode
    return coef, ls


def body_of(cfg, parameter, value, grid):
    if cfg["body"] == "levelset":
        coef, ls = levelset_coef(jnp.float64(value), cfg, parameter, grid)
        return dataclasses.replace(ls, coef=coef)
    if parameter in ("mode2", "width", "squash_y"):
        raise ValueError(f"{parameter} is a level-set control")
    c = F.CASES[cfg["case"]]
    centre = np.array(c["centre"], dtype=np.float64)
    radius = float(c["radius"])
    if parameter == "radius":
        radius = value
    elif parameter == "centre_x":
        centre[0] = value
    else:
        raise ValueError(f"parameter must be one of {PARAMETERS}")
    return AnalyticBody(centre=jnp.asarray(centre), radius=jnp.float64(radius))


def nominal(cfg, parameter):
    c = F.CASES[cfg["case"]]
    if parameter == "radius":
        return float(c["radius"])
    if parameter == "centre_x":
        return float(c["centre"][0])
    if parameter == "width":
        return float(c["half_width"])
    if parameter == "squash_y":
        return float(cfg["squash"])
    return 0.0


def control_gradient(cfg, parameter, grid, geom_cotangent):
    """``dJ/d theta`` from a cotangent on the geometry: the analytic body's own field, or
    the level-set coefficients chained through ``levelset_coef``."""
    if cfg["body"] != "levelset":
        return float(geom_cotangent.radius if parameter == "radius" else geom_cotangent.centre[0])
    theta0 = jnp.float64(nominal(cfg, parameter))
    _coef, vjp = jax.vjp(
        functools.partial(_coef_of, cfg=cfg, parameter=parameter, grid=grid), theta0
    )
    return float(vjp(geom_cotangent)[0])


def _coef_of(theta, cfg, parameter, grid):
    return levelset_coef(theta, cfg, parameter, grid)[0]


def mean_free_path(nrho, gas="Ar"):
    """The VHS free-stream mean free path ``(T / T_ref)^(omega - 1/2) / (sqrt 2 pi d^2 n)``
    at the deck's temperature (Bird 1994, eq. 4.65 with the VHS diameter at ``T``), m."""
    sp = sd.BY_SYMBOL[gas]
    sigma = np.pi * sp.diam**2 * (F.TEMP / sp.tref) ** (0.5 - sp.omega)
    return 1.0 / (np.sqrt(2.0) * sigma * nrho)


def knudsen_density(cfg):
    """The factor on the deck's ``nrho`` (and on ``fnum``) that gives ``cfg["knudsen"]`` on
    the body diameter ``2 R``; 1 when ``knudsen`` is 0."""
    if cfg.get("knudsen", 0.0) <= 0.0:
        return 1.0
    length = 2.0 * F.CASES[cfg["case"]]["radius"]
    return mean_free_path(F.NRHO) / (cfg["knudsen"] * length)


def stream_speed(cfg, gas="Ar"):
    """The free-stream speed: ``mach`` times the sound speed ``sqrt(5/3 k T / m)`` of the
    monatomic gas at the deck's temperature, or the deck's ``VSTREAM`` when ``mach`` is 0."""
    if cfg.get("mach", 0.0) <= 0.0:
        return F.VSTREAM
    return cfg["mach"] * float(np.sqrt(5.0 / 3.0 * K_BOLTZ * F.TEMP / sd.BY_SYMBOL[gas].mass))


def build(cfg, parameter, value, steps, body=None):
    """The forward's objects at control ``value`` (``F.build`` with a scaled ``fnum``), or
    at an explicit ``body`` (the optimiser's iterates)."""
    grid, mesh, sp, params, static, faces, res, wall, _target, capacity = F.build(
        cfg["case"], "analytic", cfg["tracker"], "Ar", cfg["acc"], cfg["radius_cells"]
    )
    k_grid = int(cfg["grid_scale"])
    if k_grid > 1:
        # a finer DSMC grid at the same particles per cell: fnum / k^D, N x k^D
        grid = dataclasses.replace(grid, n=tuple(k_grid * n for n in grid.n), volume=jnp.zeros(0))
        params = dataclasses.replace(
            params, fnum=jnp.full((num_cells(grid),), float(params.fnum[0]) / k_grid**static.D)
        )
        static = dataclasses.replace(static, N=static.N * k_grid**static.D)
        capacity = capacity * k_grid**static.D
    if body is None:
        body = body_of(cfg, parameter, value, grid)
    if cfg["grazing"] > 0.0:
        body = dataclasses.replace(body, grazing=float(cfg["grazing"]))
    mesh = cut.build_mesh(
        body, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=cfg["radius_cells"]
    )
    grid = cut.with_volume(grid, mesh)
    # uniform weights on every collision cell, split cells' sub-cells included; a denser gas
    # (`knudsen`) at the same particle count weighs each particle as much more
    density = knudsen_density(cfg)
    fnum = float(params.fnum[0]) * cfg["fnum_scale"] * density
    params = dataclasses.replace(params, fnum=jnp.full((num_cells(grid),), fnum))
    if density != 1.0:
        res = dataclasses.replace(res, nrho=res.nrho * density)
    n = int(cfg.get("slots", 0)) or int(static.N / cfg["fnum_scale"]) + 256
    a_max = int(cfg.get("a_max", 0)) or static.A_max
    static = dataclasses.replace(
        static, N=n, A_max=a_max, n_steps=steps, record=cfg.get("record", "lean")
    )
    capacity = int(capacity / cfg["fnum_scale"]) + 64
    if cfg.get("mach", 0.0) > 0.0 or cfg.get("dt", 0.0) > 0.0:
        # the inflow's per-step budget follows the stream speed and the step (F.build's rule)
        if cfg.get("dt", 0.0) > 0.0:
            params = dataclasses.replace(params, dt=jnp.float64(cfg["dt"]))
        res = dataclasses.replace(
            res,
            vstream=res.vstream.at[:, 0].set(
                jnp.where(res.vstream[:, 0] != 0.0, stream_speed(cfg), 0.0)
            ),
        )
        target = inflow.target_count(res, grid, 0, 0, sp.mass[0], params.fnum[0], params.dt)
        capacity = int(2 * float(target) + 64)
    return grid, mesh, sp, params, static, faces, res, wall, capacity


# `step0` is traced (integer arithmetic on the step index only): the frames mode's warm-up
# blocks and the captured run at `warmup` share one compilation per block length
@functools.partial(jax.jit, static_argnames=("faces", "static", "capacity", "capture"))
def _run(
    state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, step0, capture
):
    return simulate.run(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        key,
        faces=faces,
        wall=wall,
        res=res,
        capacity=capacity,
        emit_moments="none",
        step0=step0,
        mesh=mesh,
        capture=capture,
        emit_surface=True,
    )


@jax.jit
def _snapshot(state, grid, params, sp):
    """One :class:`CellMoments` of a state (the frame of a field movie)."""
    return sample.moments(state, sort.build_table(state, grid), grid, params, sp)


def _frame(step, state, grid, params, sp):
    m = _snapshot(state, grid, params, sp)
    # the Cartesian cells (a split cell shows its largest region): every iterate's frames
    # keep one shape whatever sub-cells its shape needs
    c0 = n_cartesian(grid)
    return {
        "step": step,
        "nrho": np.asarray(m.nrho[:c0], dtype=np.float32),
        "u": np.asarray(m.u[:c0], dtype=np.float32),
        "temp": np.asarray(m.temp[:c0], dtype=np.float32),
    }


def stack_frames(frames):
    """A list of :func:`_frame` dicts -> one dict of stacked arrays (``step`` leading)."""
    return {key: np.stack([f[key] for f in frames]) for key in ("step", "nrho", "u", "temp")}


def forward(cfg, parameter, value, key, capture, body=None, frames_every=0):
    """Empty start, ``warmup`` uncaptured steps, then the captured steps.  Returns
    ``(state0, final, diag, rec, tallies, setup)`` with ``setup`` the run's objects.

    ``frames_every > 0`` (a divisor of ``warmup``) runs the warm-up in blocks of that
    many steps -- the RNG contract makes the run identical -- and puts a field snapshot of
    the state at every block end plus the final state (the fill of the flow around this
    body, then its steady state) into ``setup["frames"]`` (:func:`stack_frames`; per-cell
    ``nrho``, ``u``, ``temp`` in float32, a few kB per frame).  The optimiser's movie."""
    grid, mesh, sp, params, static, faces, res, wall, capacity = build(
        cfg, parameter, value, cfg["steps"], body
    )
    flow = FlowParams(
        nrho=jnp.float64(F.NRHO * knudsen_density(cfg)),
        temp=jnp.float64(F.TEMP),
        vstream=jnp.array([stream_speed(cfg), 0.0, 0.0]),
    )
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    common = (grid, sp, params, key, faces, wall, res, mesh)
    frames = []
    if frames_every > 0:
        if cfg["warmup"] % frames_every:
            raise ValueError(f"frames_every {frames_every} must divide warmup {cfg['warmup']}")
        frames.append(_frame(0, state, grid, params, sp))
        warm = dataclasses.replace(static, n_steps=frames_every)
        for k0 in range(0, cfg["warmup"], frames_every):
            (state, cstate), aux = _run(state, cstate, *common, warm, capacity, k0, False)
            require_valid(aux[1])
            frames.append(_frame(k0 + frames_every, state, grid, params, sp))
    elif cfg["warmup"]:
        warm = dataclasses.replace(static, n_steps=cfg["warmup"])
        (state, cstate), aux = _run(state, cstate, *common, warm, capacity, 0, False)
        require_valid(aux[1])
    state0 = state
    (final, _), aux = _run(state0, cstate, *common, static, capacity, cfg["warmup"], capture)
    if frames_every > 0:
        frames.append(_frame(cfg["warmup"] + cfg["steps"], final, grid, params, sp))
    setup = {
        "grid": grid,
        "mesh": mesh,
        "sp": sp,
        "params": params,
        "static": static,
        "faces": faces,
        "wall": wall,
        "frames": stack_frames(frames) if frames else None,
    }
    return state0, final, aux[1], (aux[2] if capture else None), aux[-1], setup


NPC_BUCKET = 64


@functools.lru_cache(maxsize=16)
def drag_rewards(mass: float, dim: int, first: int, last: int, scale: float):
    """The windowed-drag per-hit reward and its window-edge jump as ``functools.partial``
    objects, one pair per distinct setting.  They are static arguments of the jitted
    ``shape.replay`` and ``shape.gradient``, and a partial hashes by identity: building new
    ones for every seed recompiled both (~40 s per seed, 2026-09-28), so they are cached
    on the (hashable) values that define them."""
    reward = functools.partial(_drag_reward, mass=mass, dim=dim, first=first, scale=scale)
    jump = functools.partial(_drag_jump, mass=mass, dim=dim, first=first, last=last, scale=scale)
    return reward, jump


def _drag_reward(events, tallies, rec_k, step, mass, dim, first, scale):
    """Momentum along ``x`` given to the body by each element hit of a window step."""
    del tallies, rec_k
    r = mass * (events.v_in[..., 0] - events.v_out[..., 0])
    return jnp.where((events.face >= 2 * dim) & (step >= first), r * scale, 0.0)


def _drag_jump(events, rec_k, step, mass, dim, first, last, scale):
    """Window-edge jumps of the per-hit drag reward (see ``ensemble_plane._shear_jump``)."""
    del rec_k
    r = mass * (events.v_in[..., 0] - events.v_out[..., 0]) * scale
    sign = jnp.where(step == last, 1.0, jnp.where(step == first - 1, -1.0, 0.0))
    return jnp.where(events.face >= 2 * dim, r * sign, 0.0)


def objective_value(cfg, tallies, dim, scale):
    first = cfg["steps"] - cfg["window"]
    return float(jnp.sum(tallies.momentum[first:, 2 * dim :, 0]) * scale)


def _check(diag):
    require_valid(diag)
    for name in (
        "stencil_overrun",
        "inside_body",
        "wall_recrossed",
        "walk_exhausted",
        "lane_overflow",
        "record_overflow",
    ):
        if int(getattr(diag, name).sum()):
            raise RuntimeError(f"{name} fired")


def _backwards(rec, state0, final, grid, v_scale, sp, key, neighbors, npc_max, hits, timing):
    """The two value-form device backwards of :func:`shape_terms`: the objective's hit
    rewards (``lam``) and the timing rewards (``lam_timing``, the along-flight identity's
    value); the host bookkeeping is skipped (only the device's ``lam`` is used)."""
    n = rec.x.shape[1]
    out = []
    for rewards in (hits, timing):
        _, dev = device_backward(
            rec,
            state0,
            final,
            jnp.zeros(n),
            1.0,
            grid,
            v_scale,
            sp,
            jax.random.fold_in(key, 12345),
            neighbors,
            npc_max=npc_max,
            hit_reward=rewards,
            return_device=True,
            host_adjoint=False,
        )
        out.append(dev)
    return tuple(out)


def shape_terms(
    cfg,
    key,
    body=None,
    parameter="radius",
    neighbors=32,
    timing_band=0.2,
    variants=None,
    frames_every=0,
):
    """One seed: forward, traced replay (hit rewards), value-form backward, shape terms
    on the geometry (all four ``grad_x lambda`` variants, or the ``variants`` named).
    Returns ``(all_terms, info, setup)``; ``info`` carries the objective, the hit count and
    the timings; ``setup["frames"]`` the field movie of the forward (``frames_every``)."""
    value0 = nominal(cfg, parameter)
    state0, final, diag, rec, tallies, s = forward(
        cfg, parameter, value0, key, True, body, frames_every
    )
    _check(diag)
    static, grid, mesh, wall, sp, params = (
        s["static"],
        s["grid"],
        s["mesh"],
        s["wall"],
        s["sp"],
        s["params"],
    )
    dim, m, w = static.D, cfg["steps"], cfg["window"]
    mass = sp.mass[0]
    dt = params.dt
    scale_r = float(params.fnum[0]) / (w * float(dt))
    first = cfg["warmup"] + m - w
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    reward, jump = drag_rewards(float(mass), dim, first, cfg["warmup"] + m - 1, scale_r)
    rep = shape.replay(
        rec,
        steps,
        state0,
        key,
        grid,
        s["faces"],
        wall,
        mass,
        dt,
        mesh,
        0.0,
        0.0,
        static,
        reward=reward,
        timing_jump=jump,
        timing_band=timing_band,
    )
    if float(rep.replay_error) > REPLAY_TOL:
        raise RuntimeError(f"replay error {float(rep.replay_error):g}")
    hit_reward = rep.hit_reward
    value = float(jnp.sum(rep.hit_total))
    # the timing rewards enter only the second backward, which needs each slot's per-step
    # total: keep that and let the per-event form go (the replay's outputs otherwise stay
    # alive through both backwards and the shape VJP -- the chain's memory peak)
    timing_total = rep.timing_total[:, None]
    del rep
    check = objective_value(cfg, tallies, dim, scale_r)
    newborn_share = (check - value) / max(abs(check), 1e-300)
    n_surf = 2 * dim + mesh.n_elements
    tally_cot = tallies.__class__(
        None,
        jnp.zeros((n_surf, 3)).at[2 * dim :, 0].set(scale_r),
        jnp.zeros(n_surf),
        jnp.zeros(n_surf),
    )
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    v_scale = float(np.sqrt(K_BOLTZ * F.TEMP / float(mass)))
    fit_scale = jnp.concatenate((jnp.asarray(dx), jnp.full(3, v_scale)))
    # the node tables' width: the densest cell of the record (a narrower table would drop
    # nodes; checked by the backward) -- the fits' and partner draws' cost scales with it.
    # A static argument of the jitted backward and gradient: rounded up so that seeds and
    # iterates with slightly different occupancies share one compilation (the padded rows
    # are masked out of every fit and partner draw, so the result does not depend on it)
    npc_max = int(ensemble_backward.max_occupancy(rec, grid)) + 8
    npc_max = -(-npc_max // NPC_BUCKET) * NPC_BUCKET
    t0 = time.time()
    dev, dev_t = _backwards(
        rec, state0, final, grid, v_scale, sp, key, neighbors, npc_max, hit_reward, timing_total
    )
    del timing_total
    t_backward = time.time() - t0
    t0 = time.time()
    all_terms = shape.gradient_variants(
        dev_t.lam,
        names=variants,
        records=rec,
        lam=dev.lam,
        steps=steps,
        state0=state0,
        base_key=key,
        grid=grid,
        faces=s["faces"],
        wall=wall,
        mass=mass,
        dt=dt,
        mesh=mesh,
        epsilon=0.0,
        acc_delta=0.0,
        static=static,
        scale=fit_scale,
        npc_max=npc_max,
        hit_reward=hit_reward,
        tally_cotangent=tally_cot,
        tally_weight=(steps >= first).astype(jnp.float64),
        timing_jump=jump,
        timing_band=timing_band,
        silhouette_band=float(cfg["silhouette_band"]),
        reward=reward if cfg["silhouette_band"] > 0.0 else None,
        value_class=cfg.get("value_class", "hit_ahead"),
        # epsilon = 0 above: only the slots with a body hit have a geometry-dependent
        # post-move state, so the grad-lambda fits run there only (shape.gradient)
        hit_fits=True,
    )
    t_shape = time.time() - t0
    for t in all_terms.values():
        if float(t.replay_error) > REPLAY_TOL:
            raise RuntimeError(f"replay error {float(t.replay_error):g}")
    lam = np.asarray(dev.lam)
    alive = np.asarray(rec.alive)
    info = {
        "objective": check,
        "hits": int(diag.surface_hits.sum()),
        "stuck": int(diag.surface_stuck.sum()),
        "newborn_hit_share": newborn_share,
        "alive_final": int(final.alive.sum()),
        "lambda_rms": [float(np.sqrt(np.mean(lam[k][alive[k]] ** 2))) for k in (0, m // 2, m - 1)],
        "seconds_backward": t_backward,
        "seconds_shape": t_shape,
    }
    return all_terms, info, s


def ensemble_gradient(cfg, parameter, key, neighbors=32, timing_band=0.2, variant="along_pool"):
    """One seed's scalar gradient of the control ``parameter`` (``shape_terms`` reduced by
    :func:`control_gradient`), in the artifact row format."""
    all_terms, info, s = shape_terms(cfg, key, None, parameter, neighbors, timing_band)
    grid = s["grid"]
    terms = all_terms[variant]
    check, value0 = info["objective"], nominal(cfg, parameter)
    pick = functools.partial(control_gradient, cfg, parameter, grid)
    pathwise, score, timing = pick(terms.pathwise), pick(terms.score), pick(terms.timing)
    silhouette = pick(terms.silhouette)
    parts = {k: pick(getattr(terms, k)) for k in ("pathwise_x", "pathwise_v", "pathwise_tally")}
    variants = {
        name: {
            "pathwise": pick(t.pathwise),
            "pathwise_x": pick(t.pathwise_x),
            "pathwise_v": pick(t.pathwise_v),
            "gradient": pick(t.pathwise) + score + silhouette,
        }
        for name, t in all_terms.items()
    }
    return {
        **info,
        # the hit-timing term of a per-hit reward lives at the window's two edges and has
        # zero mean on a steady window (G3a); it is reported, and left out of the estimate
        "gradient": pathwise + score + silhouette,
        "gradient_with_timing": pathwise + score + silhouette + timing,
        "pathwise": pathwise,
        **parts,
        "score": score,
        "silhouette": silhouette,
        "timing": timing,
        "variants": variants,
        "analytic_F_over_r": check / value0 if parameter == "radius" else 0.0,
    }


def finite_differences(cfg, parameter, key, hs):
    value0 = nominal(cfg, parameter)
    out = []
    for h in hs:
        values = []
        for sign in (1.0, -1.0):
            _, _, diag, _, tallies, s = forward(cfg, parameter, value0 + sign * h, key, False)
            _check(diag)
            scale_r = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
            values.append(objective_value(cfg, tallies, s["static"].D, scale_r))
        out.append((values[0] - values[1]) / (2 * h))
    return out


def artifact_tag(cfg):
    """The body / model part of the artifact name (empty for the analytic circle)."""
    tag = "" if cfg["body"] == "analytic" else f"_{cfg['body']}{cfg['levelset_h']:g}"
    if cfg["acc"] > 0.0:
        tag += f"_acc{cfg['acc']:g}"
    if cfg["silhouette_band"] > 0.0:
        tag += f"_sil{cfg['silhouette_band']:g}"
    if cfg["grazing"] > 0.0:
        tag += f"_gz{cfg['grazing']:g}"
    if cfg["grid_scale"] > 1:
        tag += f"_grid{cfg['grid_scale']}"
    if F.CASES[cfg["case"]].get("kind") == "ellipse":
        tag += f"_s{cfg['squash']:g}"  # the evaluation point of squash_y
    if cfg.get("knudsen", 0.0) > 0.0:
        tag += f"_kn{cfg['knudsen']:g}"
    if cfg.get("mach", 0.0) > 0.0:
        tag += f"_M{cfg['mach']:g}"
    if cfg.get("value_class", "hit_ahead") != "none":
        # stratified grad-lambda fits (2026-09-22); untagged artifacts are the mixed pool
        tag += "_vchit"
    return tag


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cfg = default_config()
    for k, v in cfg.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--parameter", default="radius", choices=PARAMETERS)
    p.add_argument("--neighbors", type=int, default=32)
    p.add_argument("--timing-band", type=float, default=0.2, help="hit-timing band, x dt")
    p.add_argument(
        "--variant",
        default="along_pool",
        choices=list(shape.VARIANTS),
        help="estimator of grad_x lambda reported at the top level (all four are stored)",
    )
    p.add_argument("--pullback", default="virtual", choices=("virtual", "sequential"))
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--partners", type=int, default=8)
    p.add_argument("--fd-steps", type=float, nargs="+", default=[0.1, 0.05])
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--seed-start", type=int, default=32)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = {k: getattr(args, k) for k in cfg}
    if cfg["body"] not in BODIES:
        raise SystemExit(f"--body must be one of {BODIES}")
    tag = ensemble_backward.configure(args.pullback, args.substeps, args.partners, "staged")
    body_tag = artifact_tag(cfg)
    out = args.output or Path(RESULTS) / (
        f"ensemble_mesh_{cfg['case']}{body_tag}_{args.parameter}_f{cfg['fnum_scale']:g}"
        f"_H{cfg['warmup']}_M{cfg['steps']}_W{cfg['window']}{tag}_dev.json"
    )
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    t0 = time.time()
    for i in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + i)
        t1 = time.time()
        row = ensemble_gradient(
            cfg, args.parameter, key, args.neighbors, args.timing_band, args.variant
        )
        row["fd"] = finite_differences(cfg, args.parameter, key, args.fd_steps)
        row["seed"] = i
        row["seconds"] = time.time() - t1
        rows.append(row)
        print(
            f"seed={i} J={row['objective']:.5g} ensemble={row['gradient']:.5g} "
            f"(pathwise {row['pathwise']:.4g} = x {row['pathwise_x']:.3g} + v {row['pathwise_v']:.3g} "
            f"+ tally {row['pathwise_tally']:.3g}; score {row['score']:.4g}, silhouette "
            f"{row['silhouette']:.4g}, timing {row['timing']:.4g}) "
            f"F/r={row['analytic_F_over_r']:.5g} "
            f"FD={row['fd']} hits={row['hits']} {row['seconds']:.0f}s",
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "estimator": "ensemble adjoint + replayed wall-map VJP (jax_dsmc.shape)",
                    "config": {
                        **cfg,
                        "parameter": args.parameter,
                        "seed_start": args.seed_start,
                        "timing_band": args.timing_band,
                        "variant": args.variant,
                    },
                    "requested_seeds": args.seeds,
                    "complete": len(rows) == args.seeds,
                    "fd_steps": args.fd_steps,
                    "rows": rows,
                    "elapsed": time.time() - t0,
                },
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
