"""Ensemble adjoint of the inlet-shock case (the `gradient_bench` inlet arms), route (b).

    python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 384 --seed-start 32

Same forward as ``examples.gradient_convergence --flow inlet-shock`` (same keys, same
objective), captured with ``simulate.run(capture=True)`` and pulled back on the host by
:mod:`jax_dsmc.ensemble_backward`.  The parameter gradient is assembled from four
smooth pieces, none of which is a count score:

* source density score  ``sum_births lam * d/dc log b_f(v_sampled)``
* source intensity      ``sum_{k,f} dN_f/dc * mean_births lam``
* initial density score ``sum_initial lam_0 * d/dc log f_0(v)``
* initial intensity     ``sum_sides dN0_s/dc * mean_initial lam_0``

``c`` is the control multiplying ``INLET_CONTROLS[parameter]``; ``b_f`` is the drifting
half-Maxwellian flux density with normalisation ``Z = pi^{3/2} vscale^4 flux_integral(s)``.
Rows are written in the ``gradient_convergence`` artifact format so
``examples.gradient_bench.report`` can compare them with the score-estimator arms.
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
from examples.gradient_convergence import (
    INLET_CONTROLS,
    inlet_shock_geometry,
    inlet_shock_reservoirs,
    inlet_shock_reward,
    inlet_shock_run,
)
from jax_dsmc import ensemble_backward, ensemble_device, inflow, species_data
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.shape import HitRewards
from jax_dsmc.validation import require_valid

GAS = species_data.ARGON


def log_flux_density(v, temp, stream, normal, mass):
    """Log of the normalised inward flux density ``(v.n) exp(-m|v-u|^2/2kT) / Z``."""
    vscale = jnp.sqrt(2.0 * K_BOLTZ * temp / mass)
    s = jnp.dot(stream, normal) / vscale
    vn = jnp.dot(v, normal)
    log_z = 4.0 * jnp.log(vscale) + jnp.log(inflow.flux_integral(s))
    return jnp.log(vn) - mass * jnp.sum((v - stream) ** 2) / (2.0 * K_BOLTZ * temp) - log_z


def log_maxwellian(v, temp, stream, mass):
    return -mass * jnp.sum((v - stream) ** 2) / (2.0 * K_BOLTZ * temp) - 1.5 * jnp.log(temp)


def _species_of(t):
    from jax_dsmc.collide import species_params  # noqa: PLC0415 -- keep the module light

    return species_params(
        jnp.array([GAS.mass]),
        jnp.array([[t["diam"]]]),
        jnp.array([[t["omega"]]]),
        jnp.array([[GAS.tref]]),
        jnp.array([[t["alpha"]]]),
    )


def controls(parameter, c):
    return {**INLET_CONTROLS, parameter: INLET_CONTROLS[parameter] * c}


GROWTH_LIMIT = 100.0
"""Largest tolerated growth of rms lambda over the horizon (device backward)."""

EDGE = 0.1
"""Plateau windows of the density-ratio objective: the first / last tenth of the domain
(`examples/shock/gradients.py` uses 8 of 80 cells)."""


def _window_counts(x, alive, length):
    """Particles in the upstream and downstream plateau windows."""
    s = x[:, 0] / length
    up = jnp.sum(alive & (s < EDGE))
    down = jnp.sum(alive & (s > 1.0 - EDGE))
    return up, down


def objective_value(objective, x, v, alive, length, n_nominal):
    """``energy-window``: windowed kinetic energy per nominal particle (the bench objective);
    ``density-ratio``: downstream / upstream plateau counts (Rankine-Hugoniot's rho_2/rho_1)."""
    if objective == "energy-window":
        return jnp.sum(jnp.where(alive, inlet_shock_reward(x, v, length), 0.0)) / n_nominal
    up, down = _window_counts(x, alive, length)
    return down / jnp.maximum(up, 1)


def objective_terminal(objective, x, v, length, n_nominal, totals):
    """Per-particle terminal adjoint ``dJ/d(particle i's contribution)``: the reward itself
    for a sum, the delta-method weights ``1[down]/N_up - N_down/N_up^2 1[up]`` for the ratio
    (``totals = (N_up, N_down)`` frozen at the realized values: the linearisation is exact
    for the expectation's derivative up to O(1/N))."""
    if objective == "energy-window":
        return inlet_shock_reward(x, v, length) / n_nominal
    up, down = totals
    s = x[:, 0] / length
    up_f = jnp.maximum(up, 1).astype(jnp.float64)
    return jnp.where(s > 1.0 - EDGE, 1.0 / up_f, 0.0) - jnp.where(s < EDGE, down / up_f**2, 0.0)


def _terminal_phi(x, v, objective, length, n_nominal, totals, normalization):
    """The host backward divides ``phi`` by ``normalization``; undo that here."""
    return objective_terminal(objective, x, v, length, n_nominal, totals) * normalization


def _forward(t, key, dynamic, static, cells, attempts, steps, majorant, capture):
    """``inlet_shock_run`` with the geometry split into traced arrays and static sizes."""
    return inlet_shock_run(
        t, key, {**dynamic, **static}, cells, attempts, steps, majorant, "paper", capture=capture
    )


@functools.cache
def _compiled_forward(cells, attempts, steps, majorant, capture, capacity, source_capacity):
    """One compiled forward per configuration: ``(t, key, dynamic_geometry) -> ...``."""
    static = {"capacity": capacity, "source_capacity": source_capacity}
    return jax.jit(
        functools.partial(
            _forward,
            static=static,
            cells=cells,
            attempts=attempts,
            steps=steps,
            majorant=majorant,
            capture=capture,
        )
    )


def compiled_forward(geometry, cells, attempts, steps, majorant, capture):
    fn = _compiled_forward(
        cells,
        attempts,
        steps,
        majorant,
        capture,
        int(geometry["capacity"]),
        int(geometry["source_capacity"]),
    )
    dynamic = {k: geometry[k] for k in ("length", "weight", "grid", "dt")}
    return lambda t, key: fn(t, key, dynamic)


def device_backward(
    rec,
    state0,
    out,
    phi_final,
    normalization,
    grid,
    v_scale,
    sp,
    key,
    neighbors,
    npc_max,
    chunk=None,
    phi_step=None,
    hit_reward=None,
    return_device=False,
    normals=None,
    host_adjoint=True,
):
    """The device backward under the pull-back selected by ``ensemble_backward.configure``,
    returned as the host :class:`ensemble_backward.Adjoint` (births and wall hits read off
    ``lam``).  ``npc_max`` must cover the densest cell of the run (checked).
    ``phi_step`` / ``hit_reward`` are the value-form rewards of
    :func:`ensemble_device.backward`; ``return_device=True`` also returns the raw
    :class:`ensemble_device.Backward` (its ``lam`` feeds the shape adjoint); ``normals``
    (:func:`jax_dsmc.shape.local_normals`) switches the virtual pull-back's fits to the
    one-sided form.  ``host_adjoint=False`` skips the host bookkeeping of births and
    wall hits (:func:`ensemble_backward.from_device`) and returns ``(None, dev)`` -- the
    shape adjoint reads only the device's ``lam``."""
    # a lane-tracked replay's rewards (`shape.HitRewards`): the device's move pull-back
    # needs each slot's per-step total only; the host bookkeeping gets the per-event form
    dev_reward = hit_reward.totals()[:, None] if isinstance(hit_reward, HitRewards) else hit_reward
    worst = ensemble_backward.max_occupancy(rec, grid)
    if worst > npc_max:
        raise ValueError(f"npc_max={npc_max} below the densest cell ({worst} particles)")
    if chunk is None:
        # queries per lax.map pass: the population, up to 4096 (padding is wasted work)
        chunk = min(4096, 1 << (int(rec.x.shape[1]) - 1).bit_length())
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    scale = jnp.concatenate((jnp.asarray(dx, dtype=jnp.float64), jnp.full(3, v_scale)))
    if ensemble_backward.VIRTUAL is None:
        dev = ensemble_device.backward(
            rec,
            state0.alive,
            out,
            phi_final,
            grid,
            scale,
            npc_max,
            neighbors,
            phi_step=phi_step,
            hit_reward=dev_reward,
        )
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
            key,
            npc_max,
            neighbors,
            "quadratic_full",
            chunk,
            substeps,
            partners,
            ensemble_backward.VIRTUAL_NODES,
            phi_step=phi_step,
            hit_reward=dev_reward,
            normals=normals,
        )
    # the virtual pull-back's quadrature noise feeds back through the fits; with too few
    # partners for the cell occupancy it grows geometrically over a long horizon (K = 2
    # at 40 particles/cell blew up 1e5x over 300 steps where K = 8 decayed) -- refuse
    # rather than poison the ensemble silently
    alive = np.asarray(rec.alive)
    lam = np.asarray(dev.lam)
    rms = np.array(
        [
            np.sqrt(np.mean(lam[k][alive[k]] ** 2)) if alive[k].any() else 0.0
            for k in range(len(lam))
        ]
    )
    if rms[-1] > 0 and rms.max() > GROWTH_LIMIT * rms[-1]:
        raise RuntimeError(
            f"unstable backward: rms lambda grew {rms.max() / rms[-1]:.3g}x over the horizon; "
            "raise --partners (review section 5.12)"
        )
    if not host_adjoint:
        return None, dev
    adj = ensemble_backward.from_device(rec, dev, normalization, hit_reward=hit_reward)
    return (adj, dev) if return_device else adj


def ensemble_gradient(
    parameter,
    key,
    geometry,
    cells,
    attempts,
    steps,
    majorant,
    n_nominal,
    neighbors=32,
    exact_last=False,
    collision_term=True,
    device=False,
    objective="energy-window",
):
    """One seed: forward with capture, backward (host, or device with ``device=True``),
    score assembly.  Returns (J, dJ/dc, diag)."""
    t = controls(parameter, jnp.float64(1.0))
    forward = compiled_forward(geometry, cells, attempts, steps, majorant, True)
    state0, out, diag, rec = forward(t, key)
    length = geometry["length"]
    value = float(objective_value(objective, out.x, out.v, out.alive, length, n_nominal))
    totals = _window_counts(out.x, out.alive, length)
    v_scale = float(jnp.sqrt(K_BOLTZ * GAS.tref / GAS.mass))
    sp = _species_of(controls(parameter, 1.0))
    if device:
        adj = device_backward(
            rec,
            state0,
            out,
            objective_terminal(objective, out.x, out.v, length, n_nominal, totals),
            n_nominal,
            geometry["grid"],
            v_scale,
            sp,
            jax.random.fold_in(key, 12345),
            neighbors,
            npc_max=4 * int(geometry["capacity"]) // cells,
        )
    else:
        # the host backward divides phi by the normalization itself
        adj = ensemble_backward.backward(
            rec,
            state0,
            out,
            functools.partial(
                _terminal_phi,
                objective=objective,
                length=length,
                n_nominal=n_nominal,
                totals=totals,
                normalization=n_nominal,
            ),
            n_nominal,
            geometry["grid"],
            v_scale,
            neighbors,
            exact_last,
            collision_term,
            sp=sp,
        )
    # --- sources ---------------------------------------------------------------
    normals = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    def source_score(c, v, face):
        _, temperature, stream = inlet_shock_reservoirs(controls(parameter, c))
        return log_flux_density(v, temperature[face], stream[face], normals[face], GAS.mass)

    def source_intensity(c, face):
        density, temperature, stream = inlet_shock_reservoirs(controls(parameter, c))
        res = inflow.InflowParams(density, temperature, stream)
        return inflow.target_count(
            res, geometry["grid"], 0, face, GAS.mass, geometry["weight"], geometry["dt"]
        )

    d_score = jax.jit(jax.vmap(jax.grad(source_score), in_axes=(None, 0, 0)))
    d_intensity = jax.jit(jax.grad(source_intensity), static_argnums=1)
    gradient = 0.0
    for b in adj.births:
        gradient += float(
            jnp.sum(
                jnp.asarray(b.values)
                * d_score(1.0, jnp.asarray(b.velocities), jnp.asarray(b.faces))
            )
        )
        for face in (0, 1):
            on = b.faces == face
            if on.any():
                gradient += float(d_intensity(1.0, face)) * float(np.mean(b.values[on]))
    # --- initial condition -----------------------------------------------------
    half = geometry["capacity"] // 2
    side = (np.arange(geometry["capacity"]) >= half).astype(int)

    def initial_score(c, v, s):
        _, temperature, stream = inlet_shock_reservoirs(controls(parameter, c))
        return log_maxwellian(v, temperature[s], stream[s], GAS.mass)

    def initial_intensity(c, s):
        density, _, _ = inlet_shock_reservoirs(controls(parameter, c))
        return density[s] * length / (2 * geometry["weight"])

    alive0 = np.asarray(state0.alive)
    lam0 = adj.lambda0
    d_initial = jax.jit(jax.vmap(jax.grad(initial_score), in_axes=(None, 0, 0)))
    gradient += float(
        jnp.sum(
            jnp.asarray(lam0[alive0])
            * d_initial(1.0, jnp.asarray(state0.v)[alive0], jnp.asarray(side[alive0]))
        )
    )
    for s in (0, 1):
        on = alive0 & (side == s)
        gradient += float(jax.grad(initial_intensity)(1.0, s)) * float(np.mean(lam0[on]))
    return value, gradient, diag


def finite_differences(
    parameter,
    key,
    geometry,
    cells,
    attempts,
    steps,
    majorant,
    n_nominal,
    hs,
    objective="energy-window",
):
    forward = compiled_forward(geometry, cells, attempts, steps, majorant, False)

    def value(c):
        _, out, diag, _ = forward(controls(parameter, c), key)
        require_valid(diag)
        return float(
            objective_value(objective, out.x, out.v, out.alive, geometry["length"], n_nominal)
        )

    return [(value(1 + h) - value(1 - h)) / (2 * h) for h in hs]


def main(argv=None):  # noqa: PLR0915 -- one CLI
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter", default="temp", choices=list(INLET_CONTROLS))
    p.add_argument("--particles", type=int, default=128)
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--attempts", type=int, default=512)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--majorant", default="certified", choices=("thermal", "certified"))
    p.add_argument("--neighbors", type=int, default=32)
    p.add_argument(
        "--exact-last", action="store_true", help="diagnostic: exact ghost field on the last step"
    )
    p.add_argument(
        "--no-collision-term", action="store_true", help="diagnostic: drop the partner terms"
    )
    p.add_argument("--pullback", default="virtual", choices=("virtual", "sequential", "linearised"))
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--partners", type=int, default=8)
    p.add_argument("--nodes", default="staged", choices=("staged", "post"))
    p.add_argument("--device", action="store_true", help="backward on the device (JIT)")
    p.add_argument(
        "--objective", default="energy-window", choices=("energy-window", "density-ratio")
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=32)
    p.add_argument(
        "--fd-steps",
        default="0.1,0.05,0.025,0.0125",
        help="relative FD steps (each costs two forwards per seed)",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    tag = ensemble_backward.configure(args.pullback, args.substeps, args.partners, args.nodes)
    geometry = inlet_shock_geometry(args.particles, args.cells, args.dt)
    out = args.output or Path(RESULTS) / (
        f"ensemble_{args.parameter}_N{args.particles}_M{args.steps}"
        f"{'_exactlast' if args.exact_last else ''}"
        f"{'_nocoll' if args.no_collision_term else ''}{tag}{'_dev' if args.device else ''}"
        f"{'_rho' if args.objective == 'density-ratio' else ''}.json"
    )
    hs = [float(h) for h in args.fd_steps.split(",")]
    rows = []
    if out.exists():
        previous = json.loads(out.read_text())
        if previous["fd_steps"] != hs:
            raise ValueError(f"{out} was produced with FD steps {previous['fd_steps']}")
        rows = previous["rows"]
    t0 = time.time()
    for i in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + i)
        j, g, diag = ensemble_gradient(
            args.parameter,
            key,
            geometry,
            args.cells,
            args.attempts,
            args.steps,
            args.majorant,
            args.particles,
            args.neighbors,
            args.exact_last,
            not args.no_collision_term,
            args.device,
            args.objective,
        )
        require_valid(diag)
        fd = finite_differences(
            args.parameter,
            key,
            geometry,
            args.cells,
            args.attempts,
            args.steps,
            args.majorant,
            args.particles,
            hs,
            args.objective,
        )
        rows.append({"seed": i, "objective": j, "gradient": g, "fd": fd})
        print(f"seed={i} J={j:.6g} ensemble={g:.6g} FD={fd}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "estimator": "ensemble adjoint, route (b), host backward",
                    "config": {k: v for k, v in vars(args).items() if k not in ("output", "seeds")},
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
