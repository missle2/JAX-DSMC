"""Ensemble adjoint for collision-kernel parameters on a 0-D anisotropy relaxation.

    python3 -m examples.ensemble_relaxation --parameter diameter --seeds 192

An anisotropic Maxwellian (T_x = 2, T_y = T_z = 1/2) relaxes under VSS collisions
only (``collide_only``); ``J = <v_x^2>`` after ``M`` steps.  The kernel parameters
enter the adjoint through the collision operator's explicit dependence,

    dJ/dtheta = sum_k sum_{accepted pairs} d_theta log q(g) * [Lam(v_i') + Lam(v_j') - Lam(v_i) - Lam(v_j)],

with ``q = collision_rate`` differentiated by ``jax.grad`` through ``species_params``.
The accepted pairs sample ``q f f`` whatever the majorant, so no attempt-count term
appears.  Nondimensional units: unit mass, ``k_B = 1``, ``T_ref = 1``, unit density.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.gradient_bench import RESULTS
from jax_dsmc import ensemble_backward, ensemble_device, simulate
from jax_dsmc.collide import species_params
from jax_dsmc.collide.vss import collision_rate
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid

CONTROLS = {"diameter": 0.4, "omega": 0.81, "alpha": 1.4}


def species(t):
    return species_params(
        jnp.array([K_BOLTZ]),
        jnp.array([[t["diameter"]]]),
        jnp.array([[t["omega"]]]),
        jnp.ones((1, 1)),
        jnp.array([[t["alpha"]]]),
    )


def controls(parameter, c):
    return {**CONTROLS, parameter: CONTROLS[parameter] * c}


def relaxation_run(t, key, particles, steps, attempts, dt, majorant, capture=False):
    n = particles
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (1,))
    static = SimStatic(n, 1, 1, 1, attempts, 1, steps, majorant_mode=majorant)
    params = SimParams(jnp.float64(dt), jnp.full(1, 1.0 / n), jnp.float64(0.0))
    kx, kv = jax.random.split(jax.random.fold_in(key, 7))
    sigma = jnp.array([jnp.sqrt(2.0), jnp.sqrt(0.5), jnp.sqrt(0.5)])
    state = ParticleState(
        jax.random.uniform(kx, (n, 1), dtype=jnp.float64),
        sigma * jax.random.normal(kv, (n, 3), dtype=jnp.float64),
        jnp.zeros(n, jnp.int32),
        jnp.ones(n, bool),
        jnp.zeros(n),
    )
    (out, _), aux = simulate.run(
        state,
        CollideState(jnp.zeros((1, 1, 1))),
        grid,
        species(t),
        params,
        static,
        key,
        collide_only=True,
        emit_moments="none",
        capture=capture,
    )
    return state, out, aux[1], (aux[2] if capture else None), grid


@functools.cache
def _compiled_forward(particles, steps, attempts, dt, majorant, capture):
    return jax.jit(
        functools.partial(
            relaxation_run,
            particles=particles,
            steps=steps,
            attempts=attempts,
            dt=dt,
            majorant=majorant,
            capture=capture,
        )
    )


def _vx2(x, v):
    del x
    return v[:, 0] ** 2


def _log_rate(c, vi, vj, parameter):
    return jnp.log(collision_rate(vi, vj, 0, 0, species(controls(parameter, c))))


@functools.cache
def _pair_score(parameter):
    """``d/dc log q(v_i, v_j)`` at ``c = 1`` -- cached so the device backward compiles once."""
    return functools.partial(_d_log_rate_at_one, parameter=parameter)


def _d_log_rate_at_one(vi, vj, parameter):
    return jax.grad(_log_rate)(1.0, vi, vj, parameter)


def ensemble_gradient(parameter, key, cfg, neighbors=32, device=False):
    """One seed.  ``device`` runs the backward with :mod:`jax_dsmc.ensemble_device` (the
    pull-back selected by ``ensemble_backward.configure``) instead of the host sweep."""
    forward = _compiled_forward(**cfg, capture=True)
    state0, out, diag, rec, grid = forward(controls(parameter, jnp.float64(1.0)), key)
    n = cfg["particles"]
    value = float(jnp.mean(_vx2(out.x, out.v)))
    sp = species(controls(parameter, 1.0))
    if device:
        phi_final = _vx2(out.x, out.v) / n
        scale = jnp.ones(4)
        if ensemble_backward.VIRTUAL is None:
            dev = ensemble_device.backward(
                rec, state0.alive, out, phi_final, grid, scale, n, neighbors
            )
            accept = np.asarray(rec.pairs.accept) & (np.asarray(rec.pairs.pi) < n)
            score = jax.vmap(_pair_score(parameter))(
                jnp.asarray(rec.pairs.vi)[accept], jnp.asarray(rec.pairs.vj)[accept]
            )
            gradient = float(jnp.sum(jnp.asarray(dev.four_term)[accept] * score))
        else:
            substeps, partners = ensemble_backward.VIRTUAL
            dev = ensemble_device.backward_virtual(
                rec,
                state0.alive,
                out,
                phi_final,
                grid,
                scale,
                sp,
                jax.random.fold_in(key, 12345),
                n,
                neighbors,
                ensemble_backward.VIRTUAL_BASIS,
                min(4096, 1 << (n - 1).bit_length()),
                substeps,
                partners,
                ensemble_backward.VIRTUAL_NODES,
                _pair_score(parameter),
            )
            gradient = float(jnp.sum(dev.kernel))
        return value, gradient, diag
    adj = ensemble_backward.backward(rec, state0, out, _vx2, n, grid, 1.0, neighbors, sp=sp)
    d_log_rate = jax.jit(
        jax.vmap(jax.grad(functools.partial(_log_rate, parameter=parameter)), in_axes=(None, 0, 0))
    )
    gradient = 0.0
    for pr in adj.pairs:
        if len(pr.i):
            w = d_log_rate(1.0, jnp.asarray(pr.vi), jnp.asarray(pr.vj))
            weight = 1.0 if pr.weight is None else jnp.asarray(pr.weight)
            gradient += float(jnp.sum(weight * w * jnp.asarray(pr.four_term)))
    return value, gradient, diag


def finite_differences(parameter, key, cfg, hs):
    forward = _compiled_forward(**cfg, capture=False)

    def value(c):
        _, out, diag, _, _ = forward(controls(parameter, c), key)
        require_valid(diag)
        return float(jnp.mean(out.v[:, 0] ** 2))

    return [(value(1 + h) - value(1 - h)) / (2 * h) for h in hs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter", default="diameter", choices=list(CONTROLS))
    p.add_argument("--particles", type=int, default=2000)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--attempts", type=int, default=2048)
    p.add_argument("--dt", type=float, default=0.2)
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
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    tag = ensemble_backward.configure(
        args.pullback, args.substeps, args.partners, args.nodes, args.backward_basis
    )
    cfg = {k: getattr(args, k) for k in ("particles", "steps", "attempts", "dt", "majorant")}
    out = args.output or Path(RESULTS) / (
        f"ensemble_relax_{args.parameter}_N{args.particles}_M{args.steps}{tag}"
        f"{'_dev' if args.device else ''}.json"
    )
    hs = [0.1, 0.05, 0.025, 0.0125]
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    t0 = time.time()
    for i in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + i)
        j, g, diag = ensemble_gradient(args.parameter, key, cfg, args.neighbors, args.device)
        require_valid(diag)
        fd = finite_differences(args.parameter, key, cfg, hs)
        rows.append({"seed": i, "objective": j, "gradient": g, "fd": fd})
        print(f"seed={i} J={j:.6g} ensemble={g:.6g} FD={fd}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "estimator": "ensemble adjoint, route (b), kernel-rate scores",
                    "config": {**cfg, "parameter": args.parameter, "seed_start": args.seed_start},
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
