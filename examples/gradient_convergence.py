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

from jax_dsmc import adjoint, simulate
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.collide import species_params
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.gradient_validation import Acceptance, paired_report
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid


def couette(args):
    controls = {
        "temperature_left": 1.0,
        "temperature_right": 1.0,
        "speed": 1.0,
        "diameter": 0.25,
        "omega": 0.81,
        "alpha": 1.4,
        "accommodation": 0.8,
    }
    if args.parameter not in controls:
        raise ValueError(f"Couette parameters: {list(controls)}")

    def evaluate(control, key):
        t = {**controls, args.parameter: controls[args.parameter] * control}
        n = args.particles
        grid = GridSpec(jnp.zeros(1), jnp.ones(1), (args.cells,))
        sp = species_params(
            jnp.array([K_BOLTZ]),
            jnp.array([[t["diameter"]]]),
            jnp.array([[t["omega"]]]),
            jnp.ones((1, 1)),
            jnp.array([[t["alpha"]]]),
        )
        static = SimStatic(n, 1, 1, 1, args.attempts, 8, args.steps, majorant_mode="certified")
        params = SimParams(
            jnp.float64(args.dt), jnp.full(args.cells, 1 / n), jnp.float64(args.epsilon)
        )
        kx, kv = jax.random.split(jax.random.fold_in(key, 999))
        state = ParticleState(
            jax.random.uniform(kx, (n, 1), dtype=jnp.float64),
            jax.random.normal(kv, (n, 3), dtype=jnp.float64),
            jnp.zeros(n, jnp.int32),
            jnp.ones(n, bool),
            jnp.zeros(n),
        )
        cs = CollideState(jnp.zeros((args.cells, 1, 1)))
        wall = WallParams(
            jnp.array([t["temperature_left"], t["temperature_right"]]),
            jnp.array([[0.0, -t["speed"], 0.0], [0.0, t["speed"], 0.0]]),
            jnp.full(2, t["accommodation"]),
        )
        (out, _), (_, diag) = simulate.run(
            state,
            cs,
            grid,
            sp,
            params,
            static,
            key,
            faces=Faces.uniform(BC.DIFFUSE, 1),
            wall=wall,
            emit_moments="none",
            score_mode="trajectory",
        )
        value = jnp.mean(out.v[:, 1] ** 2)
        return adjoint.trajectory_surrogate(value, out.logh_acc[0], args.baseline), diag

    return evaluate, "mean tangential velocity squared", 1.0


def shock(args):
    from examples import shock_adjoint as sa  # noqa: PLC0415 -- optional reference dependency

    cs = sa.case(
        1,
        npc=max(1, args.particles // args.cells),
        n_x=args.cells,
        n_steps=args.steps,
        a_max=args.attempts,
        dt_scale=(args.dt/.05)*(args.cells/16),
    )
    nominal = sa.reference_theta()
    if args.parameter not in nominal:
        raise ValueError(f"Shock parameters: {list(nominal)}")

    def evaluate(control, key):
        theta = {**nominal, args.parameter: nominal[args.parameter] * control}
        return sa.evaluate(theta, cs, key, scored=True, baseline=args.baseline)

    return (
        evaluate,
        "reflected shock smooth total velocity squared",
        float(3 * K_BOLTZ * nominal["temp"] / cs.gas.mass + nominal["ux"] ** 2),
    )


def inlet_shock(args):  # noqa: PLR0915 -- keep the mathematical scan together
    """Two-sided physical reservoir shock, fixed simulator weight and random IC counts."""
    from examples.shock.normal import (
        rankine_hugoniot,
    )
    from jax_dsmc import (  # noqa: PLC0415 -- optional reference dependency
        inflow,
        sampling,
        species_data,
    )
    from jax_dsmc.safe import safe_log  # noqa: PLC0415 -- optional reference dependency

    gas = species_data.ARGON
    controls = {
        "temp": gas.tref,
        "nrho": 1e21,
        "mach": 2.0,
        "diam": gas.diam,
        "omega": gas.omega,
        "alpha": gas.alpha,
    }
    if args.parameter not in controls:
        raise ValueError(f"Inlet shock parameters: {list(controls)}")
    # Nominal geometry and weight never change with the differentiated control.
    from examples.shock_adjoint import (
        mfp_vss as mean_free_path,
    )

    lam = mean_free_path(1e21)
    length = 60 * lam
    nominal_ratio = rankine_hugoniot(2.0, 5 / 3)["rho"]
    weight = 1e21 * (1 + nominal_ratio) * length / (2 * args.particles)
    capacity = 4 * args.particles
    source_capacity = args.particles
    grid = GridSpec(jnp.zeros(1), jnp.array([length]), (args.cells,))
    dt = args.dt * lam / np.sqrt(K_BOLTZ * gas.tref / gas.mass)

    def evaluate(control, key):
        t = {**controls, args.parameter: controls[args.parameter] * control}
        gamma = 5 / 3
        m2 = t["mach"] ** 2
        ratio = (gamma + 1) * m2 / ((gamma - 1) * m2 + 2)
        pressure = (2 * gamma * m2 - (gamma - 1)) / (gamma + 1)
        density = jnp.array([t["nrho"], t["nrho"] * ratio])
        temperature = jnp.array([t["temp"], t["temp"] * pressure / ratio])
        speed = t["mach"] * jnp.sqrt(gamma * K_BOLTZ * t["temp"] / gas.mass)
        stream = jnp.array([[speed, 0.0, 0.0], [speed / ratio, 0.0, 0.0]])
        sp = species_params(
            jnp.array([gas.mass]),
            jnp.array([[t["diam"]]]),
            jnp.array([[t["omega"]]]),
            jnp.array([[gas.tref]]),
            jnp.array([[t["alpha"]]]),
        )
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
            xs.append(
                length * (side + jax.random.uniform(keys[1], (half, 1), dtype=jnp.float64)) / 2
            )
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
        cs = CollideState(jnp.zeros((args.cells, 1, 1)))
        params = SimParams(jnp.float64(dt), jnp.full(args.cells, weight), jnp.float64(0.0))
        static = SimStatic(
            capacity, 1, 1, 1, args.attempts, 8, args.steps, majorant_mode="certified"
        )
        (out, _), (_, diag) = simulate.run(
            state,
            cs,
            grid,
            sp,
            params,
            static,
            key,
            faces=Faces.uniform(BC.INFLOW, 1),
            res=inflow.InflowParams(density, temperature, stream),
            capacity=source_capacity,
            emit_moments="none",
            score_mode="trajectory",
        )
        phi = jnp.sum(out.v**2, axis=1) * jnp.exp(-36 * (out.x[:, 0] / length - 0.5) ** 2)
        value = jnp.sum(jnp.where(out.alive, phi, 0.0)) / args.particles
        # Initial population capacity failures share the explicit source failure channel.
        diag = dataclasses.replace(diag, inflow_shortfall=diag.inflow_shortfall + invalid_initial)
        return adjoint.trajectory_surrogate(value, out.logh_acc[0], args.baseline), diag

    scale = (
        3 * K_BOLTZ * gas.tref / gas.mass
        + controls["mach"] ** 2 * (5 / 3) * K_BOLTZ * gas.tref / gas.mass
    )
    return evaluate, "inlet shock fixed-normalization windowed total velocity squared", scale


def main():  # noqa: PLR0915 -- keep the mathematical scan together
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--flow", choices=("couette", "shock", "inlet-shock"), required=True)
    p.add_argument("--parameter", required=True)
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--attempts", type=int, default=256)
    p.add_argument(
        "--dt", type=float, default=0.05, help="Couette dt; inlet dt in mean-free-flight units; reflected-shock dt scale relative to .05"
    )
    p.add_argument("--epsilon", type=float, default=0.005, help="Couette only")
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Independent objective baseline, fixed before these seeds",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.particles, args.cells, args.steps, args.attempts, args.seeds) < 1:
        p.error("Sizes must be positive")
    evaluate, objective, scale = {"couette": couette, "shock": shock, "inlet-shock": inlet_shock}[
        args.flow
    ](args)
    value = jax.jit(evaluate)
    gradient = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
    config = {k: v for k, v in vars(args).items() if k not in ("output", "seeds")}
    steps = [0.1, 0.05, 0.025, 0.0125]
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in sorted((root/"jax_dsmc").rglob("*.py")) + [Path(__file__), root/"examples/shock_adjoint/__init__.py"]:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    implementation = digest.hexdigest()
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous["config"] != config or previous["fd_steps"] != steps
                or previous.get("implementation_sha256") != implementation
                or previous.get("jax_version") != jax.__version__):
            raise ValueError("Cannot resume a different experiment into the same artifact")
        rows = previous["rows"]
    start = time.monotonic()
    for seed in range(len(rows), args.seeds):
        key = jax.random.key(args.seed_start + seed)
        (j, diag), g = gradient(jnp.float64(1), key)
        require_valid(diag, gradients=True)
        fd = []
        for h in steps:
            hi, dh = value(jnp.float64(1 + h), key)
            lo, dl = value(jnp.float64(1 - h), key)
            require_valid(dh)
            require_valid(dl)
            fd.append(float((hi - lo) / (2 * h)))
        rows.append({"seed": seed, "objective": float(j), "gradient": float(g), "fd": fd})
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
