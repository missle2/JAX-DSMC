"""Production NTC transient AD/FD experiments, with resumable per-seed records.

No result from this runner alone certifies mesh/steady convergence. Run the same
case at several --steps/--dt/--cells/--particles and inspect paired refinements.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples import shock_adjoint as sa
from examples.shock.normal import rankine_hugoniot
from examples.shock_adjoint import mfp_vss as mean_free_path
from jax_dsmc import adjoint, inflow, sampling, simulate, species_data
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.collide import species_params
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.gradient_validation import Acceptance, paired_report
from jax_dsmc.safe import safe_log
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid


def scored(args, value, phi, state, rho=1.0):
    """The estimator the run asked for; both have the primal value of ``objective``.

    ``trajectory`` scores the complete scalar with the once-per-event log likelihood;
    ``paper`` is the terminal-label recursion of [P2] (each particle's own accumulator).
    A fixed ``args.baseline`` applies to both and must come from independent seeds.
    """
    if args.score_mode == "trajectory":
        return adjoint.trajectory_surrogate(value, state.logh_acc[0], args.baseline)
    return adjoint.surrogate(phi, state.logh_acc, state.alive, rho, args.baseline)


COUETTE_CONTROLS = {
    "temperature_left": 1.0,
    "temperature_right": 1.0,
    "speed": 1.0,
    "diameter": 0.25,
    "omega": 0.81,
    "alpha": 1.4,
    "accommodation": 0.8,
}


def couette_walls(t):
    """Wall parameters from the controls; differentiable in every entry of ``t``."""
    return WallParams(
        jnp.array([t["temperature_left"], t["temperature_right"]]),
        jnp.array([[0.0, -t["speed"], 0.0], [0.0, t["speed"], 0.0]]),
        jnp.full(2, t["accommodation"]),
    )


def couette_species(t):
    """Single-species VSS table of the Couette case (unit mass, ``k_B = 1``)."""
    return species_params(
        jnp.array([K_BOLTZ]),
        jnp.array([[t["diameter"]]]),
        jnp.array([[t["omega"]]]),
        jnp.ones((1, 1)),
        jnp.array([[t["alpha"]]]),
    )


def couette_run(
    t,
    key,
    particles,
    cells,
    steps,
    attempts,
    dt,
    epsilon,
    majorant,
    score_mode,
    capture=False,
    emit_surface=False,
):
    """The forward of the Couette case (unit gap, unit mass, k_B = 1); ``capture`` adds
    records, ``emit_surface`` appends the per-step face tallies as a sixth result."""
    n = particles
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (cells,))
    sp = couette_species(t)
    static = SimStatic(n, 1, 1, 1, attempts, 8, steps, majorant_mode=majorant)
    params = SimParams(jnp.float64(dt), jnp.full(cells, 1 / n), jnp.float64(epsilon))
    kx, kv = jax.random.split(jax.random.fold_in(key, 999))
    state = ParticleState(
        jax.random.uniform(kx, (n, 1), dtype=jnp.float64),
        jax.random.normal(kv, (n, 3), dtype=jnp.float64),
        jnp.zeros(n, jnp.int32),
        jnp.ones(n, bool),
        jnp.zeros(n),
    )
    cs = CollideState(jnp.zeros((cells, 1, 1)))
    (out, _), aux = simulate.run(
        state,
        cs,
        grid,
        sp,
        params,
        static,
        key,
        faces=Faces.uniform(BC.DIFFUSE, 1),
        wall=couette_walls(t),
        emit_moments="none",
        score_mode=score_mode,
        capture=capture,
        emit_surface=emit_surface,
    )
    result = (state, out, aux[1], (aux[2] if capture else None), grid)
    return (*result, aux[-1]) if emit_surface else result


def couette(args):
    if args.parameter not in COUETTE_CONTROLS:
        raise ValueError(f"Couette parameters: {list(COUETTE_CONTROLS)}")

    def evaluate(control, key):
        t = {**COUETTE_CONTROLS, args.parameter: COUETTE_CONTROLS[args.parameter] * control}
        _, out, diag, _, _ = couette_run(
            t,
            key,
            args.particles,
            args.cells,
            args.steps,
            args.attempts,
            args.dt,
            args.epsilon,
            args.majorant,
            args.score_mode,
        )
        phi = out.v[:, 1] ** 2
        return scored(args, jnp.mean(phi), phi, out), diag

    return evaluate, "mean tangential velocity squared", 1.0


def shock(args):
    cs = sa.case(
        1,
        npc=max(1, args.particles // args.cells),
        n_x=args.cells,
        n_steps=args.steps,
        a_max=args.attempts,
        dt_scale=(args.dt / 0.05) * (args.cells / 16),
    )
    nominal = sa.reference_theta()
    if args.parameter not in nominal:
        raise ValueError(f"Shock parameters: {list(nominal)}")

    def evaluate(control, key):
        theta = {**nominal, args.parameter: nominal[args.parameter] * control}
        return sa.evaluate(
            theta, cs, key, scored=True, baseline=args.baseline, score_mode=args.score_mode
        )

    return (
        evaluate,
        "reflected shock smooth total velocity squared",
        float(3 * K_BOLTZ * nominal["temp"] / cs.gas.mass + nominal["ux"] ** 2),
    )


INLET_CONTROLS = {
    "temp": species_data.ARGON.tref,
    "nrho": 1e21,
    "mach": 2.0,
    "diam": species_data.ARGON.diam,
    "omega": species_data.ARGON.omega,
    "alpha": species_data.ARGON.alpha,
}


def inlet_shock_geometry(particles, cells, dt):
    """Nominal geometry and simulator weight; never varies with the differentiated control."""
    gas = species_data.ARGON
    lam = mean_free_path(1e21)
    length = 60 * lam
    nominal_ratio = rankine_hugoniot(2.0, 5 / 3)["rho"]
    weight = 1e21 * (1 + nominal_ratio) * length / (2 * particles)
    grid = GridSpec(jnp.zeros(1), jnp.array([length]), (cells,))
    return {
        "length": length,
        "weight": weight,
        "capacity": 4 * particles,
        "source_capacity": particles,
        "grid": grid,
        "dt": dt * lam / np.sqrt(K_BOLTZ * gas.tref / gas.mass),
    }


def inlet_shock_reservoirs(t):
    """Both reservoirs from the upstream state by Rankine-Hugoniot at ``t["mach"]``.

    Differentiable in every entry of ``t`` (the ensemble adjoint scores through it).
    Returns ``(density[2], temperature[2], stream[2, 3])``.
    """
    gas = species_data.ARGON
    gamma = 5 / 3
    m2 = t["mach"] ** 2
    ratio = (gamma + 1) * m2 / ((gamma - 1) * m2 + 2)
    pressure = (2 * gamma * m2 - (gamma - 1)) / (gamma + 1)
    density = jnp.array([t["nrho"], t["nrho"] * ratio])
    temperature = jnp.array([t["temp"], t["temp"] * pressure / ratio])
    speed = t["mach"] * jnp.sqrt(gamma * K_BOLTZ * t["temp"] / gas.mass)
    stream = jnp.array([[speed, 0.0, 0.0], [speed / ratio, 0.0, 0.0]])
    return density, temperature, stream


def inlet_shock_initial_state(key, density, temperature, stream, geometry):
    """Two half-domains of Maxwellian gas with stochastically rounded counts.

    Returns ``(state, initial_score, invalid)``; the score is the count rounding's.
    """
    gas = species_data.ARGON
    length, weight, capacity = geometry["length"], geometry["weight"], geometry["capacity"]
    xs, vs, active = [], [], []
    initial_score = jnp.float64(0.0)
    invalid_initial = jnp.bool_(False)
    for side in (0, 1):
        keys = jax.random.split(jax.random.fold_in(key, 100 + side), 4)
        mean = density[side] * length / (2 * weight)
        u = jax.random.uniform(keys[0], (), dtype=jnp.float64)
        number = sampling.stochastic_round(mean, u).astype(jnp.int32)
        fraction = mean - jnp.floor(mean)
        initial_score += jnp.where(u < fraction, safe_log(fraction), safe_log(1 - fraction))
        half = capacity // 2
        invalid_initial |= number > half
        xs.append(length * (side + jax.random.uniform(keys[1], (half, 1), dtype=jnp.float64)) / 2)
        vs.append(
            stream[side]
            + jnp.sqrt(K_BOLTZ * temperature[side] / gas.mass)
            * jax.random.normal(keys[2], (half, 3), dtype=jnp.float64)
        )
        active.append(jnp.arange(half) < number)
    state = ParticleState(
        jnp.concatenate(xs),
        jnp.concatenate(vs),
        jnp.zeros(capacity, jnp.int32),
        jnp.concatenate(active),
        jnp.full(capacity, initial_score),
    )
    return state, initial_score, invalid_initial


def inlet_shock_reward(x, v, length):
    return jnp.sum(v**2, axis=1) * jnp.exp(-36 * (x[:, 0] / length - 0.5) ** 2)


def inlet_shock_run(t, key, geometry, cells, attempts, steps, majorant, score_mode, capture=False):
    """The forward of the inlet-shock case; ``capture`` adds the event records."""
    gas = species_data.ARGON
    density, temperature, stream = inlet_shock_reservoirs(t)
    sp = species_params(
        jnp.array([gas.mass]),
        jnp.array([[t["diam"]]]),
        jnp.array([[t["omega"]]]),
        jnp.array([[gas.tref]]),
        jnp.array([[t["alpha"]]]),
    )
    state, _, invalid_initial = inlet_shock_initial_state(
        key, density, temperature, stream, geometry
    )
    cs = CollideState(jnp.zeros((cells, 1, 1)))
    params = SimParams(
        jnp.float64(geometry["dt"]), jnp.full(cells, geometry["weight"]), jnp.float64(0.0)
    )
    static = SimStatic(geometry["capacity"], 1, 1, 1, attempts, 8, steps, majorant_mode=majorant)
    (out, _), aux = simulate.run(
        state,
        cs,
        geometry["grid"],
        sp,
        params,
        static,
        key,
        faces=Faces.uniform(BC.INFLOW, 1),
        res=inflow.InflowParams(density, temperature, stream),
        capacity=geometry["source_capacity"],
        emit_moments="none",
        score_mode=score_mode,
        capture=capture,
    )
    diag = aux[1]
    # Initial population capacity failures share the explicit source failure channel.
    diag = dataclasses.replace(diag, inflow_shortfall=diag.inflow_shortfall + invalid_initial)
    return state, out, diag, (aux[2] if capture else None)


def inlet_shock(args):
    """Two-sided physical reservoir shock, fixed simulator weight and random IC counts."""
    if args.parameter not in INLET_CONTROLS:
        raise ValueError(f"Inlet shock parameters: {list(INLET_CONTROLS)}")
    geometry = inlet_shock_geometry(args.particles, args.cells, args.dt)

    def evaluate(control, key):
        t = {**INLET_CONTROLS, args.parameter: INLET_CONTROLS[args.parameter] * control}
        _, out, diag, _ = inlet_shock_run(
            t, key, geometry, args.cells, args.attempts, args.steps, args.majorant, args.score_mode
        )
        phi = inlet_shock_reward(out.x, out.v, geometry["length"])
        value = jnp.sum(jnp.where(out.alive, phi, 0.0)) / args.particles
        # fixed normalisation: rho = N_alive / N_nominal turns the live mean into sum/N_nominal
        rho = jax.lax.stop_gradient(jnp.sum(out.alive) / args.particles)
        return scored(args, value, phi, out, rho), diag

    gas = species_data.ARGON
    scale = (
        3 * K_BOLTZ * gas.tref / gas.mass
        + INLET_CONTROLS["mach"] ** 2 * (5 / 3) * K_BOLTZ * gas.tref / gas.mass
    )
    return evaluate, "inlet shock fixed-normalization windowed total velocity squared", scale


def main(argv=None):  # noqa: PLR0915 -- keep the mathematical scan together
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--flow", choices=("couette", "shock", "inlet-shock"), required=True)
    p.add_argument("--parameter", required=True)
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--attempts", type=int, default=256)
    p.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Couette dt; inlet dt in mean-free-flight units; reflected-shock dt scale relative to .05",
    )
    p.add_argument("--epsilon", type=float, default=0.005, help="Couette only")
    p.add_argument(
        "--score-mode",
        choices=("paper", "trajectory"),
        default="trajectory",
        help="terminal-label (paper) or once-per-event full-trajectory likelihood",
    )
    p.add_argument(
        "--majorant",
        choices=("thermal", "certified"),
        default="certified",
        help="NTC majorant: the thermal heuristic or the closed-cell energy bound",
    )
    p.add_argument(
        "--time", action="store_true", help="record wall-clock per gradient and per FD pair"
    )
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Independent objective baseline, fixed before these seeds",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if min(args.particles, args.cells, args.steps, args.attempts, args.seeds) < 1:
        p.error("Sizes must be positive")
    evaluate, objective, scale = {"couette": couette, "shock": shock, "inlet-shock": inlet_shock}[
        args.flow
    ](args)
    value = jax.jit(evaluate)
    gradient = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
    config = {k: v for k, v in vars(args).items() if k not in ("output", "seeds", "time")}
    steps = [0.1, 0.05, 0.025, 0.0125]
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in [
        *sorted((root / "jax_dsmc").rglob("*.py")),
        Path(__file__),
        root / "examples/shock_adjoint/__init__.py",
    ]:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    implementation = digest.hexdigest()
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous["config"] != config
            or previous["fd_steps"] != steps
            or previous.get("implementation_sha256") != implementation
            or previous.get("jax_version") != jax.__version__
        ):
            raise ValueError("Cannot resume a different experiment into the same artifact")
        rows = previous["rows"]
    start = time.monotonic()
    for seed in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + seed)
        t0 = time.monotonic()
        (j, diag), g = gradient(jnp.float64(1), key)
        g = float(g)  # block until the device is done, so the timing is real
        t1 = time.monotonic()
        require_valid(diag, gradients=True)
        fd = []
        for h in steps:
            hi, dh = value(jnp.float64(1 + h), key)
            lo, dl = value(jnp.float64(1 - h), key)
            require_valid(dh)
            require_valid(dl)
            fd.append(float((hi - lo) / (2 * h)))
        t2 = time.monotonic()
        row = {
            "seed": seed,
            "objective": float(j),
            "gradient": g,
            "fd": fd,
            "max_attempts": int(jnp.max(diag.max_attempts)),
            "min_majorant_margin": float(jnp.min(diag.min_majorant_margin)),
        }
        if args.time:
            # the first seed includes compilation; report.py drops it from timing stats
            row["seconds_gradient"] = t1 - t0
            row["seconds_fd_pair"] = (t2 - t1) / len(steps)
        rows.append(row)
        result = {
            "implementation_sha256": implementation,
            "jax_version": jax.__version__,
            "requested_seeds": args.seeds,
            "complete": len(rows) == args.seeds,
            "config": config,
            "backend": jax.default_backend(),
            "objective": objective,
            "objective_scale": scale,
            "fd_steps": steps,
            "rows": rows,
            "elapsed_this_invocation": time.monotonic() - start,
            "derivative_target": "finite-horizon expected numerical objective",
            "limitations": [
                "hard membership sensitivity not established",
                "majorant dependence conditional; refinement required",
                "no stationary certification; all cases require refinement",
            ],
        }
        if len(rows) > 1:
            result["comparisons" if len(rows) == args.seeds else "provisional_comparisons"] = [
                paired_report(
                    [r["gradient"] for r in rows],
                    [r["fd"][i] for r in rows],
                    scale=1 / scale,
                    acceptance=Acceptance(comparisons=len(steps)),
                )
                for i in range(len(steps))
            ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)
        print(f"seed={seed} J={float(j):.6g} AD={float(g):.6g} FD={fd}", flush=True)


if __name__ == "__main__":
    main()
