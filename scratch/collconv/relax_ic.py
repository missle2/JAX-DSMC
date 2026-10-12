"""Path 1b: a collisional gate of the FITTED SLOPE grad_v lambda (what the shape gradient's
pathwise x / v parts consume) -- no earlier gate checks a slope of the collisional value.

0-D anisotropy relaxation (examples.ensemble_relaxation, J = <v_x^2> after M steps) with a
reparameterised initial condition: v_0 = (c_tx sigma_x u_x + c_ux, sigma_y u_y, sigma_z u_z).
The adjoint gradient is the fitted slope of lambda_0 at the initial population,

    dJ/dc = sum_i  d_vx lambda_0(v_i) * d v_ix / dc ,

with lambda_0 from backward_virtual (S x K, staged) and the slope from
ensemble_device.node_gradients (the shape adjoint's fit: k neighbours, quadratic_full).
Arbiter: CRN central differences (the initial draws are common; the collisions follow).

    PYTHONPATH=. python3 scratch/collconv/relax_ic.py OUT.jsonl SEED0 SEED1 '{overrides}'
overrides: control (tx|ux), particles, steps, dt, attempts, virtual [[S, K], ...],
neighbors [k, ...], bases [backward fit basis, ...] (jax_dsmc.cell_fit), backward_keys (the
backward repeated with that many partner keys on the same forward: its own MC noise), fd_steps.
"""

import functools
import itertools
import json
import sys
import time

import jax
import jax.numpy as jnp

from examples import ensemble_relaxation as R
from jax_dsmc import ensemble_device, simulate
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic
from jax_dsmc.validation import require_valid

SIGMA = (2.0**0.5, 0.5**0.5, 0.5**0.5)


def run(c, key, control, particles, steps, attempts, dt, capture):
    n = particles
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (1,))
    static = SimStatic(n, 1, 1, 1, attempts, 1, steps, majorant_mode="certified")
    params = SimParams(jnp.float64(dt), jnp.full(1, 1.0 / n), jnp.float64(0.0))
    kx, kv = jax.random.split(jax.random.fold_in(key, 7))
    u = jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    v = jnp.asarray(SIGMA) * u
    if control == "tx":
        v = v.at[:, 0].multiply(c)
    else:
        v = v.at[:, 0].add(c - 1.0)
    state = ParticleState(
        jax.random.uniform(kx, (n, 1), dtype=jnp.float64), v, jnp.zeros(n, jnp.int32),
        jnp.ones(n, bool), jnp.zeros(n),
    )  # fmt: skip
    (out, _), aux = simulate.run(
        state, CollideState(jnp.zeros((1, 1, 1))), grid, R.species(R.CONTROLS), params, static,
        key, collide_only=True, emit_moments="none", capture=capture,
    )  # fmt: skip
    return state, out, aux[1], (aux[2] if capture else None), grid


@functools.cache
def compiled(control, particles, steps, attempts, dt, capture):
    return jax.jit(functools.partial(
        run, control=control, particles=particles, steps=steps, attempts=attempts, dt=dt,
        capture=capture,
    ))  # fmt: skip


def gradient(key, cfg, sk, ks, bases=("quadratic_full",), backward_keys=1):
    fwd = compiled(cfg["control"], cfg["particles"], cfg["steps"], cfg["attempts"], cfg["dt"], True)
    state0, out, diag, rec, grid = fwd(jnp.float64(1.0), key)
    require_valid(diag)
    n = cfg["particles"]
    phi = out.v[:, 0] ** 2 / n
    scale = jnp.ones(4)
    sp = R.species(R.CONTROLS)
    dvdc = state0.v[:, 0] if cfg["control"] == "tx" else jnp.ones(n)
    res = {}
    for (s, k_partners), basis, r in itertools.product(sk, bases, range(backward_keys)):
        t0 = time.time()
        dev = ensemble_device.backward_virtual(
            rec, state0.alive, out, phi, grid, scale, sp, jax.random.fold_in(key, 12345 + r), n,
            32, basis, min(4096, 1 << (n - 1).bit_length()), s, k_partners,
            "staged", None,
        )  # fmt: skip
        dev.lambda0.block_until_ready()
        tag = "" if basis == "quadratic_full" else f"_{basis}"
        tag += f"_r{r}" if r else ""
        res[f"{s}x{k_partners}{tag}_seconds"] = time.time() - t0
        for k in ks:
            _, gv = ensemble_device.node_gradients(
                state0.x, state0.v, state0.alive, dev.lambda0, grid, n, scale, k
            )
            res[f"{s}x{k_partners}{tag}_k{k}"] = float(jnp.sum(gv[:, 0] * dvdc))
    return float(jnp.mean(out.v[:, 0] ** 2)), res


def fd(key, cfg, hs):
    fwd = compiled(cfg["control"], cfg["particles"], cfg["steps"], cfg["attempts"], cfg["dt"], False)

    def value(c):
        _, out, diag, _, _ = fwd(jnp.float64(c), key)
        require_valid(diag)
        return float(jnp.mean(out.v[:, 0] ** 2))

    return [(value(1 + h) - value(1 - h)) / (2 * h) for h in hs]


def main():
    path = sys.argv[1]
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    cfg = {"control": "tx", "particles": 2000, "steps": 10, "attempts": 2048, "dt": 0.2}
    sk = over.pop("virtual", [[4, 8]])
    ks = over.pop("neighbors", [32])
    bases = over.pop("bases", ["quadratic_full"])
    backward_keys = over.pop("backward_keys", 1)
    hs = over.pop("fd_steps", [0.05, 0.025])
    cfg.update(over)
    for i in range(int(sys.argv[2]), int(sys.argv[3])):
        t0 = time.time()
        key = jax.random.key(i)
        j, g = gradient(key, cfg, sk, ks, bases, backward_keys)
        row = {"seed": i, "config": cfg, "objective": j, "adjoint": g,
               "fd": fd(key, cfg, hs) if hs else [],
               "fd_steps": hs, "seconds": time.time() - t0}
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
