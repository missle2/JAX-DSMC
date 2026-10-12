"""Ensemble adjoint of the Couette case (the `gradient_bench` triad arms), route (b).

    python3 -m examples.ensemble_couette --parameter speed --seeds 384 --seed-start 32

Same forward as ``examples.gradient_convergence --flow couette`` (``couette_run``),
captured with ``simulate.run(capture=True)`` and pulled back on the host.  The diffuse
wall is a kernel: the realized re-emission is its unbiased sample, so the value passes
through unchanged and the wall parameters enter ONLY through the score of the
re-emission density at each hit,

    b_f(v) = acc * [ v_n exp(-m v_n^2/2kT) / (kT/m) ] * [ exp(-m|v_t - u_f|^2/2kT) / (2 pi kT/m) ]
             on the diffuse branch,   (1 - acc) on the specular branch,

weighted by lambda at the post-move state.  No epsilon regularisation is required for
these parameters (the hit set carries no theta-dependence); the forward may still use
one, and it is parameter-free randomness to this estimator.

Objectives (``--objective``): ``energy`` is the terminal mean tangential energy of the
original benchmark; ``energy-window`` and ``shear-window`` are the value-form windowed
objectives of the mesh plan (gates G1 and G2) -- the mean of ``v_y^2`` over the last
``--window`` steps and the mean momentum per particle and step given to the moving wall
(face 1) over the same window, the latter as a per-hit reward read off the traced replay
of the wall map (:func:`jax_dsmc.shape.replay`).  Both need ``--device``.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from examples.ensemble_inlet_shock import device_backward
from examples.gradient_bench import RESULTS
from examples.gradient_convergence import (
    COUETTE_CONTROLS,
    couette_run,
    couette_species,
    couette_walls,
)
from jax_dsmc import ensemble_backward, shape
from jax_dsmc.boundary import BC, Faces
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import SimStatic
from jax_dsmc.validation import require_valid

MASS = K_BOLTZ  # the Couette case uses unit mass with k_B = 1 (thermal variances = T)
NORMALS = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
OBJECTIVES = ("energy", "energy-window", "shear-window")
REPLAY_TOL = 1e-12  # relative; the replay is the forward's arithmetic in another compilation


def log_wall_kernel(v_out, diffuse, face, wall, mass):
    """Log of the accommodation-mixed diffuse re-emission density, for the realized branch."""
    temp = wall.temp[face]
    u = wall.vwall[face]
    n = NORMALS[face]
    v_n = jnp.dot(v_out, n)
    v_t = v_out - v_n * n
    kt = K_BOLTZ * temp / mass
    log_diffuse = (
        jnp.log(wall.acc[face])
        + jnp.log(v_n)
        - v_n**2 / (2 * kt)
        - jnp.sum((v_t - u) ** 2) / (2 * kt)
        - 2.0 * jnp.log(temp)
    )
    return jnp.where(diffuse, log_diffuse, jnp.log(1.0 - wall.acc[face]))


def _tangential_energy(x, v):
    del x
    return v[:, 1] ** 2


def _shear_reward(events, tallies, rec_k, step, first, scale):
    """Momentum per unit mass along ``y`` given to the moving wall (face 1) by each hit of
    a window step, times ``scale = 1 / (N W)``."""
    del tallies, rec_k
    r = events.v_in[..., 1] - events.v_out[..., 1]
    return jnp.where((events.face == 1) & (step >= first), r * scale, 0.0)


def objective_terms(objective, window, cfg, t, key, state0, out, rec, tallies):
    """``(J, phi_final, phi_step, hit_reward)`` of one captured forward.

    ``energy``: terminal, as before.  ``energy-window``: ``phi_step`` is ``v_y^2 / (N W)``
    at the post-collision state of every window step.  ``shear-window``: ``hit_reward``
    is the per-hit momentum to face 1 over the window from the traced replay, whose sum
    is checked against the forward's own tallies.
    """
    n, m = cfg["particles"], cfg["steps"]
    if objective == "energy":
        phi = _tangential_energy(out.x, out.v)
        return float(jnp.mean(phi)), phi / n, None, None
    first = m - window
    scale = 1.0 / (n * window)
    if objective == "energy-window":
        vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0) * scale
        phi_step = jnp.where((jnp.arange(m) >= first)[:, None], vy2, 0.0)
        return float(jnp.sum(phi_step)), jnp.zeros(n), phi_step, None
    if objective == "shear-window":
        static = SimStatic(n, 1, 1, 1, cfg["attempts"], 8, m, majorant_mode=cfg["majorant"])
        grid = _grid(cfg)
        rep = shape.replay(
            rec,
            jnp.arange(m, dtype=jnp.int32),
            state0,
            key,
            grid,
            Faces.uniform(BC.DIFFUSE, 1),
            couette_walls(t),
            MASS,
            jnp.float64(cfg["dt"]),
            None,
            jnp.float64(cfg["epsilon"]),
            0.0,
            static,
            functools.partial(_shear_reward, first=first, scale=scale),
        )
        r, err = rep.hit_reward, rep.replay_error
        if float(err) > REPLAY_TOL:
            raise RuntimeError(f"the replayed wall map differs from the record by {float(err):g}")
        value = float(jnp.sum(r))
        from_tallies = _shear_from_tallies(tallies, first, scale)
        if abs(value - from_tallies) > 1e-9 * max(1.0, abs(value)):
            raise RuntimeError(f"replayed hit rewards {value} != tallies {from_tallies}")
        return value, jnp.zeros(n), None, r
    raise ValueError(f"objective must be one of {OBJECTIVES}")


def _shear_from_tallies(tallies, first, scale):
    return float(jnp.sum(tallies.momentum[first:, 1, 1]) * scale / MASS)


def _grid(cfg):
    from jax_dsmc.state import GridSpec  # noqa: PLC0415 -- only the windowed objectives need it

    return GridSpec(jnp.zeros(1), jnp.ones(1), (cfg["cells"],))


_species_of = couette_species


def controls(parameter, c):
    return {**COUETTE_CONTROLS, parameter: COUETTE_CONTROLS[parameter] * c}


@functools.cache
def _compiled_forward(
    particles, cells, steps, attempts, dt, epsilon, majorant, capture, emit_surface=False
):
    return jax.jit(
        functools.partial(
            couette_run,
            particles=particles,
            cells=cells,
            steps=steps,
            attempts=attempts,
            dt=dt,
            epsilon=epsilon,
            majorant=majorant,
            score_mode="paper",
            capture=capture,
            emit_surface=emit_surface,
        )
    )


def ensemble_gradient(
    parameter, key, cfg, neighbors=32, device=False, objective="energy", window=1, one_sided=False
):
    """One seed: forward with capture, backward (host or device), wall-score assembly."""
    t = controls(parameter, jnp.float64(1.0))
    tallies = None
    if objective == "shear-window":
        forward = _compiled_forward(**cfg, capture=True, emit_surface=True)
        state0, out, diag, rec, grid, tallies = forward(t, key)
    else:
        forward = _compiled_forward(**cfg, capture=True)
        state0, out, diag, rec, grid = forward(t, key)
    n = cfg["particles"]
    value, phi_final, phi_step, hit_reward = objective_terms(
        objective, window, cfg, t, key, state0, out, rec, tallies
    )
    sp = _species_of(controls(parameter, 1.0))
    if device:
        normals = None
        if one_sided:
            # the nearest diffuse face within one cell of every slot: one-sided fits there
            normals = shape.local_normals(rec, grid, None, Faces.uniform(BC.DIFFUSE, 1))
        adj = device_backward(
            rec,
            state0,
            out,
            phi_final,
            n,
            grid,
            1.0,
            sp,
            jax.random.fold_in(key, 12345),
            neighbors,
            npc_max=4 * n // cfg["cells"],
            phi_step=phi_step,
            hit_reward=hit_reward,
            normals=normals,
        )
    else:
        if objective != "energy":
            raise ValueError("the windowed objectives run on the device backward (--device)")
        adj = ensemble_backward.backward(
            rec, state0, out, _tangential_energy, n, grid, 1.0, neighbors, sp=sp
        )

    def score(c, v_out, diffuse, face):
        return log_wall_kernel(v_out, diffuse, face, couette_walls(controls(parameter, c)), MASS)

    d_score = jax.jit(jax.vmap(jax.grad(score), in_axes=(None, 0, 0, 0)))
    gradient = 0.0
    for h in adj.hits:
        s = d_score(1.0, jnp.asarray(h.v_out), jnp.asarray(h.diffuse), jnp.asarray(h.faces))
        gradient += float(jnp.sum(jnp.asarray(h.values) * s))
    return value, gradient, diag


def _objective_value(parameter, key, cfg, objective, window, c):
    n, m = cfg["particles"], cfg["steps"]
    if objective == "energy":
        _, out, diag, _, _ = _compiled_forward(**cfg, capture=False)(controls(parameter, c), key)
        require_valid(diag)
        return float(jnp.mean(out.v[:, 1] ** 2))
    if objective == "energy-window":
        _, _, diag, rec, _ = _compiled_forward(**cfg, capture=True)(controls(parameter, c), key)
        require_valid(diag)
        vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0)
        return float(jnp.sum(vy2[m - window :]) / (n * window))
    _, _, diag, _, _, tallies = _compiled_forward(**cfg, capture=False, emit_surface=True)(
        controls(parameter, c), key
    )
    require_valid(diag)
    return _shear_from_tallies(tallies, m - window, 1.0 / (n * window))


def finite_differences(parameter, key, cfg, hs, objective="energy", window=1):
    value = functools.partial(_objective_value, parameter, key, cfg, objective, window)
    return [(value(1 + h) - value(1 - h)) / (2 * h) for h in hs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter", default="speed", choices=list(COUETTE_CONTROLS))
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--attempts", type=int, default=256)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.005)
    p.add_argument("--majorant", default="certified", choices=("thermal", "certified"))
    p.add_argument("--neighbors", type=int, default=32)
    p.add_argument("--pullback", default="virtual", choices=("virtual", "sequential", "linearised"))
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--partners", type=int, default=8)
    p.add_argument(
        "--backward-basis",
        default=ensemble_backward.CELL_BASIS,
        help="field fit of the virtual pull-back (jax_dsmc.cell_fit; quadratic_full: per query)",
    )
    p.add_argument("--nodes", default="staged", choices=("staged", "post"))
    p.add_argument("--device", action="store_true", help="backward on the device (JIT)")
    p.add_argument("--objective", default="energy", choices=OBJECTIVES)
    p.add_argument("--window", type=int, default=1, help="steps of the windowed objectives")
    p.add_argument(
        "--one-sided", action="store_true", help="one-sided fits next to the walls (device)"
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=32)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    tag = ensemble_backward.configure(
        args.pullback, args.substeps, args.partners, args.nodes, args.backward_basis
    )
    cfg = {
        k: getattr(args, k)
        for k in ("particles", "cells", "steps", "attempts", "dt", "epsilon", "majorant")
    }
    obj = "" if args.objective == "energy" else f"_{args.objective}{args.window}"
    obj += "_1s" if args.one_sided else ""
    out = args.output or Path(RESULTS) / (
        f"ensemble_couette_{args.parameter}_N{args.particles}_M{args.steps}_eps{args.epsilon:g}"
        f"{tag}{'_dev' if args.device else ''}{obj}.json"
    )
    hs = [0.1, 0.05, 0.025, 0.0125]
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    t0 = time.time()
    for i in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + i)
        j, g, diag = ensemble_gradient(
            args.parameter,
            key,
            cfg,
            args.neighbors,
            args.device,
            args.objective,
            args.window,
            args.one_sided,
        )
        require_valid(diag)
        fd = finite_differences(args.parameter, key, cfg, hs, args.objective, args.window)
        rows.append({"seed": i, "objective": j, "gradient": g, "fd": fd})
        print(f"seed={i} J={j:.6g} ensemble={g:.6g} FD={fd}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "estimator": "ensemble adjoint, route (b), host backward, wall kernel scores",
                    "config": {
                        **cfg,
                        "parameter": args.parameter,
                        "seed_start": args.seed_start,
                        "objective": args.objective,
                        "window": args.window,
                        "one_sided": args.one_sided,
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
