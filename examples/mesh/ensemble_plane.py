"""Gate G3: the first mesh-path shape gradient -- a Couette wall as an embedded plane.

    python3 -m examples.mesh.ensemble_plane --objective energy-window --seeds 64

Two-dimensional box ``[0, L] x [0, 1]`` (``y`` periodic), a stationary diffuse box face at
``x = 0`` and, embedded, a rectangle whose left face at ``x = g0 + theta`` is the moving
diffuse wall (drift ``U`` along ``y``); its other three faces are specular and never
reached.  Unit mass with ``k_B = 1`` and ``T = 1`` as the Couette benchmark.  The control
``theta`` translates the rectangle along ``x``.  The gas fills ``[0, g0 + theta]`` and runs
``--warmup`` uncaptured steps first (the plan's capture strategy: the window sits far past
the mixing time, so the captured start state's own ``theta`` dependence -- what the
truncated adjoint cannot see, gate G6 -- has decayed; a run from an empty gap instead put
the first hits into nearly empty cells, where the fitted ``grad_x lambda`` is noise).
Objectives over the last ``W`` captured steps: the mean tangential energy
(``energy-window``, a per-step reward) and the mean specific momentum given to the plane
(``shear-window``, a per-hit reward from the traced replay).  Collisionless by default
(``--diameter 0``): the cut-volume term of the collision step is not in the estimator
(plan section 2, gate G5).

Estimator: ensemble backward with value-form rewards -> ``lambda`` at every post-move
state -> :func:`jax_dsmc.shape.gradient` (pathwise VJP of the replayed wall map, the
score terms and the hit-timing band term -- ``v_y^2`` jumps at every hit, the shear
reward only at the window's edges; all reported).  Arbiter: common-random-number central differences at
several ``h`` on the same seed (the body is rebuilt on the host for each ``theta``; the
warm-up runs at the perturbed ``theta`` too, so the FD is the full derivative).
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.ensemble_inlet_shock import device_backward
from examples.gradient_bench import RESULTS
from jax_dsmc import ensemble_backward, initialize, shape, simulate
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.collide import species_params
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import num_cells
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import surface as S
from jax_dsmc.state import FlowParams, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid

MASS = K_BOLTZ  # unit mass with k_B = 1, as the Couette benchmark
OBJECTIVES = ("energy-window", "shear-window")
REPLAY_TOL = 1e-12


def default_config():
    return {
        "length": 1.6,
        "gap0": 1.1,
        "body_right": 1.5,
        "cells_x": 8,
        "cells_y": 4,
        "particles": 4096,
        "warmup": 48,
        "steps": 24,
        "window": 12,
        "dt": 0.05,
        "speed": 1.0,
        "diameter": 0.0,
        "attempts": 8,
    }


def body_points(cfg, theta):
    """Rectangle ``[g0 + theta, right] x [0, 1]`` listed so the normals point out of it."""
    g = cfg["gap0"] + theta
    r = cfg["body_right"]
    return np.array([[g, 0.0], [g, 1.0], [r, 1.0], [r, 0.0]])


ELEMENTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
MODELS = np.array([0, 1, 1, 1])  # the left face is the moving diffuse wall, the rest specular


def build(cfg, theta):
    grid = GridSpec(
        lo=jnp.zeros(2), hi=jnp.array([cfg["length"], 1.0]), n=(cfg["cells_x"], cfg["cells_y"])
    )
    surface = S.make_surface(body_points(cfg, theta), ELEMENTS, MODELS)
    if float(S.body_volume(surface)) <= 0.0:
        raise ValueError("the rectangle is inside-out")
    # 0.05 x |v| per step against h = 0.2: radius 3 covers 12 thermal speeds
    mesh = cut.build_mesh(surface, grid, radius=3, sub=8)
    return cut.with_volume(grid, mesh), mesh


def wall_params(cfg):
    return (
        WallParams.uniform(2, 1.0, acc=1.0, n_models=2)
        .with_model(2, 0, temp=1.0, acc=1.0, vwall=jnp.array([0.0, cfg["speed"], 0.0]))
        .with_model(2, 1, temp=1.0, acc=0.0)
    )


def faces():
    return Faces((BC.DIFFUSE, BC.SPECULAR, BC.PERIODIC, BC.PERIODIC))


def species(cfg):
    d = max(cfg["diameter"], 1e-12)  # a vanishing diameter is a collisionless gas
    return species_params(
        jnp.array([MASS]),
        jnp.array([[d]]),
        jnp.array([[0.81]]),
        jnp.ones((1, 1)),
        jnp.array([[1.4]]),
    )


def static_of(cfg, steps=None):
    n = cfg["particles"]
    steps = cfg["steps"] if steps is None else steps
    return SimStatic(n, 2, 1, 1, cfg["attempts"], 8, steps, majorant_mode="certified")


def initial(cfg, theta, key):
    """Uniform in ``[0, g0 + theta] x [0, 1]`` at ``T = 1``."""
    n = cfg["particles"]
    kx, kv = jax.random.split(jax.random.fold_in(key, 999))
    x = jax.random.uniform(kx, (n, 2), dtype=jnp.float64)
    x = x.at[:, 0].multiply(cfg["gap0"] + theta)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    return ParticleState(x, v, jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n))


def forward(cfg, theta, key, capture):
    """Warm-up then the captured steps at ``theta``; returns ``(state0, final, diag, rec,
    tallies, grid, mesh)`` with ``state0`` the state at the start of the captured steps
    (``diag`` / ``tallies`` are the captured steps' only)."""
    grid, mesh = build(cfg, theta)
    sp = species(cfg)
    c = num_cells(grid)
    params = SimParams(
        jnp.float64(cfg["dt"]), jnp.full(c, 1.0 / cfg["particles"]), jnp.float64(0.0)
    )
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(1.0), vstream=jnp.zeros(3))
    cstate = initialize.initial_collide_state(flow, grid, sp, static_of(cfg))
    state = initial(cfg, theta, key)
    common = (grid, sp, params, key, faces(), wall_params(cfg), mesh)
    if cfg["warmup"]:
        (state, cstate), aux = _run(state, cstate, *common, static_of(cfg, cfg["warmup"]), 0, False)
        require_valid(aux[1])
    state0 = state
    (final, _), aux = _run(state0, cstate, *common, static_of(cfg), cfg["warmup"], capture)
    diag = aux[1]
    rec = aux[2] if capture else None
    return state0, final, diag, rec, aux[-1], grid, mesh


@functools.partial(jax.jit, static_argnames=("faces", "static", "step0", "capture"))
def _run(state0, cstate, grid, sp, params, key, faces, wall, mesh, static, step0, capture):
    return simulate.run(
        state0,
        cstate,
        grid,
        sp,
        params,
        static,
        key,
        faces=faces,
        wall=wall,
        mesh=mesh,
        emit_moments="none",
        step0=step0,
        capture=capture,
        emit_surface=True,
    )


def _shear_reward(events, tallies, rec_k, step, first, scale):
    del tallies, rec_k
    r = events.v_in[..., 1] - events.v_out[..., 1]
    return jnp.where((events.face == 4) & (step >= first), r * scale, 0.0)


def _shear_jump(events, rec_k, step, first, last, scale):
    """A per-hit reward moves across the window's edges only: ``+r`` when a hit of the last
    window step slips into the next step, ``-r`` when one of the step before the window
    slips in."""
    del rec_k
    r = (events.v_in[..., 1] - events.v_out[..., 1]) * scale
    sign = jnp.where(step == last, 1.0, jnp.where(step == first - 1, -1.0, 0.0))
    return jnp.where(events.face == 4, r * sign, 0.0)


def _energy_jump(events, rec_k, step, first, scale):
    """``v_y^2`` at the step end changes from ``v_in``'s to ``v_out``'s when the hit is in
    this step rather than the next (any recorded hit: the fixed wall's too)."""
    del rec_k
    jump = (events.v_out[..., 1] ** 2 - events.v_in[..., 1] ** 2) * scale
    return jnp.where((events.face >= 0) & (step >= first), jump, 0.0)


def objective_value(cfg, objective, rec, tallies):
    n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
    first = m - w
    if objective == "energy-window":
        vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0)
        return float(jnp.sum(vy2[first:]) / (n * w))
    return float(jnp.sum(tallies.momentum[first:, 4, 1]) / (MASS * n * w))


def _check(diag):
    require_valid(diag)
    if int(diag.surface_hits.sum()) == 0:
        raise RuntimeError("no plane hits")
    for name in ("stencil_overrun", "inside_body", "wall_recrossed"):
        if int(getattr(diag, name).sum()):
            raise RuntimeError(f"{name} fired")


def ensemble_gradient(  # noqa: PLR0915 -- one seed, kept whole
    cfg, objective, key, neighbors=32, timing_band=0.2, variant="along_pool"
):
    """One seed: forward, value-form backward, shape terms.  Returns a result row."""
    n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
    first = cfg["warmup"] + m - w
    scale_r = 1.0 / (n * w)
    state0, final, diag, rec, tallies, grid, mesh = forward(cfg, 0.0, key, True)
    _check(diag)
    static = static_of(cfg)
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    wall = wall_params(cfg)
    dt = jnp.float64(cfg["dt"])
    phi_step, hit_reward, tally_cot = None, None, None
    last = cfg["warmup"] + m - 1
    if objective == "energy-window":
        jump = functools.partial(_energy_jump, first=first, scale=scale_r)
        reward = None
    else:
        jump = functools.partial(_shear_jump, first=first, last=last, scale=scale_r)
        reward = functools.partial(_shear_reward, first=first, scale=scale_r)
    rep = shape.replay(
        rec,
        steps,
        state0,
        key,
        grid,
        faces(),
        wall,
        MASS,
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
    if objective == "energy-window":
        vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0) * scale_r
        phi_step = jnp.where((steps >= first)[:, None], vy2, 0.0)
        value = float(jnp.sum(phi_step))
        check = objective_value(cfg, objective, rec, tallies)
        if abs(value - check) > 1e-9 * max(1.0, abs(value)):
            raise RuntimeError(f"step rewards {value} != objective {check}")
    else:
        hit_reward = rep.hit_reward
        value = float(jnp.sum(rep.hit_total))
        check = objective_value(cfg, objective, rec, tallies)
        if abs(value - check) > 1e-9 * max(1.0, abs(value)):
            raise RuntimeError(f"hit rewards {value} != tallies {check}")
        # the pathwise part of the reward: d r / d v_out through the specular... none here
        # (the plane is diffuse: v_out is a kernel sample), but the tally weight is what a
        # specular element would need, so it is wired and reported
        n_surf = 4 + mesh.n_elements
        tally_cot = tallies.__class__(
            None,
            jnp.zeros((n_surf, 3)).at[4, 1].set(scale_r / MASS),
            jnp.zeros(n_surf),
            jnp.zeros(n_surf),
        )
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    fit_scale = jnp.concatenate((jnp.asarray(dx), jnp.ones(3)))
    npc_max = max(64, 4 * n // num_cells(grid))
    _, dev = device_backward(
        rec,
        state0,
        final,
        jnp.zeros(n),
        n,
        grid,
        1.0,
        species(cfg),
        jax.random.fold_in(key, 12345),
        neighbors,
        npc_max=npc_max,
        phi_step=phi_step,
        hit_reward=hit_reward,
        return_device=True,
    )
    # the along-flight identity's value: the timing rewards carried backward
    _, dev_t = device_backward(
        rec,
        state0,
        final,
        jnp.zeros(n),
        n,
        grid,
        1.0,
        species(cfg),
        jax.random.fold_in(key, 12345),
        neighbors,
        npc_max=npc_max,
        hit_reward=rep.timing_reward,
        return_device=True,
    )
    all_terms = shape.gradient_variants(
        dev_t.lam,
        records=rec,
        lam=dev.lam,
        steps=steps,
        state0=state0,
        base_key=key,
        grid=grid,
        faces=faces(),
        # G3's artifacts were measured with the mixed pool (the one-sided rule alone
        # separates the populations at a flat wall); kept so provenance reproduces them
        value_class="none",
        wall=wall,
        mass=MASS,
        dt=dt,
        mesh=mesh,
        epsilon=0.0,
        acc_delta=0.0,
        static=static,
        scale=fit_scale,
        npc_max=npc_max,
        hit_reward=hit_reward,
        tally_cotangent=tally_cot,
        tally_weight=(steps >= first).astype(jnp.float64) if tally_cot is not None else None,
        timing_jump=jump,
        timing_band=timing_band,
    )
    terms = all_terms[variant]
    if float(terms.replay_error) > REPLAY_TOL:
        raise RuntimeError(f"replay error {float(terms.replay_error):g}")
    # d/dtheta of a translation along x: the sum over the points' x-cotangents
    pathwise = float(jnp.sum(terms.pathwise[:, 0]))
    score = float(jnp.sum(terms.score[:, 0]))
    timing = float(jnp.sum(terms.timing[:, 0]))
    variants = {}
    for name, t in all_terms.items():
        pw = float(jnp.sum(t.pathwise[:, 0]))
        variants[name] = {
            "pathwise": pw,
            "pathwise_x": float(jnp.sum(t.pathwise_x[:, 0])),
            "pathwise_v": float(jnp.sum(t.pathwise_v[:, 0])),
            "gradient": pw + score + timing,
        }
    return {
        "objective": value,
        "gradient": pathwise + score + timing,
        "pathwise": pathwise,
        "score": score,
        "timing": timing,
        "variants": variants,
        "hits": int(diag.surface_hits.sum()),
        "lambda_rms0": float(jnp.sqrt(jnp.mean(dev.lambda0**2))),
    }


def finite_differences(cfg, objective, key, hs):
    out = []
    for h in hs:
        values = []
        for theta in (h, -h):
            _, _, diag, rec, tallies, _, _ = forward(cfg, theta, key, objective == "energy-window")
            require_valid(diag)
            values.append(objective_value(cfg, objective, rec, tallies))
        out.append((values[0] - values[1]) / (2 * h))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cfg = default_config()
    for k, v in cfg.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--objective", default="energy-window", choices=OBJECTIVES)
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
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=32)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = {k: getattr(args, k) for k in cfg}
    tag = ensemble_backward.configure(args.pullback, args.substeps, args.partners, "staged")
    out = args.output or Path(RESULTS) / (
        f"ensemble_mesh_plane_{args.objective}{cfg['window']}_N{cfg['particles']}"
        f"_H{cfg['warmup']}_M{cfg['steps']}_d{cfg['diameter']:g}{tag}_dev.json"
    )
    hs = [0.04, 0.02, 0.01]
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    t0 = time.time()
    for i in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + i)
        row = ensemble_gradient(
            cfg, args.objective, key, args.neighbors, args.timing_band, args.variant
        )
        row["fd"] = finite_differences(cfg, args.objective, key, hs)
        row["seed"] = i
        rows.append(row)
        print(
            f"seed={i} J={row['objective']:.6g} ensemble={row['gradient']:.6g} "
            f"(pathwise {row['pathwise']:.4g}, score {row['score']:.4g}, timing {row['timing']:.4g}) "
            f"FD={row['fd']}",
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "estimator": "ensemble adjoint + replayed wall-map VJP (jax_dsmc.shape)",
                    "config": {
                        **cfg,
                        "objective": args.objective,
                        "seed_start": args.seed_start,
                        "timing_band": args.timing_band,
                        "variant": args.variant,
                    },
                    "requested_seeds": args.seeds,
                    "complete": len(rows) == args.seeds,
                    "fd_steps": hs,
                    "rows": rows,
                    "elapsed": time.time() - t0,
                },
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
