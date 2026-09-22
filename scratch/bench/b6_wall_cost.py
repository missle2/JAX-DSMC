"""B6: cost of the wall path -- forward and reverse wall-clock per step, compiled memory.

Runs in either worktree (main repo = old fold path, branch = chronological tracker), so
the two can be compared on identical cases.  ``K_bounce`` only matters on the branch.
Case: Couette-like box with DIFFUSE walls on axis 0, PERIODIC elsewhere, argon at 273 K,
N particles, M steps; the reverse pass differentiates the wall temperature.
"""

import argparse
import functools
import json
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_dsmc import simulate  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams  # noqa: E402
from jax_dsmc.collide import species_params  # noqa: E402
from jax_dsmc.collide.ntc import vremax_init  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.species_data import ARGON  # noqa: E402
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic  # noqa: E402


def build(dim, n, steps, k_bounce, cells=8):
    gas = ARGON
    sp = species_params(
        jnp.array([gas.mass]), jnp.array([[gas.diam]]), jnp.array([[gas.omega]]),
        jnp.array([[gas.tref]]), jnp.array([[gas.alpha]]),
    )
    length = 0.01
    grid = GridSpec(lo=jnp.zeros(dim), hi=jnp.full(dim, length), n=(cells,) * dim,
                    transverse=jnp.float64(length ** (3 - dim)))
    nrho = 1e21
    fnum = nrho * length**3 / n
    vth = jnp.sqrt(K_BOLTZ * gas.tref / gas.mass)
    dt = 0.1 * (length / cells) / float(vth)
    params = SimParams(jnp.float64(dt), jnp.full(cells**dim, fnum), jnp.float64(0.0))
    static = SimStatic(n, dim, 1, 1, 96, k_bounce, steps)
    key = jax.random.PRNGKey(0)
    kx, kv = jax.random.split(key)
    state = ParticleState(
        jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * length,
        vth * jax.random.normal(kv, (n, 3), dtype=jnp.float64),
        jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n),
    )
    cstate = CollideState(vremax=vremax_init(jnp.full((cells**dim,), gas.tref), sp.omega[0, 0],
                                             sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0],
                                             cells**dim, 1))
    bcs = [BC.DIFFUSE, BC.DIFFUSE] + [BC.PERIODIC] * (2 * dim - 2)
    faces = Faces(tuple(bcs))
    return state, cstate, grid, sp, params, static, faces, key


def objective(temp, state, cstate, grid, sp, params, static, faces, key):
    wall = WallParams(temp=jnp.full(2 * len(grid.n), 1.0) * temp,
                      vwall=jnp.zeros((2 * len(grid.n), 3)), acc=jnp.ones(2 * len(grid.n)))
    (out, _), (_, diag) = simulate.run(state, cstate, grid, sp, params, static, key,
                                       faces=faces, wall=wall, emit_moments="none")
    return jnp.mean(jnp.sum(out.v**2, axis=1)), diag


def timed(fn, *args, reps=3):
    result = fn(*args)
    jax.block_until_ready(result)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t.append(time.perf_counter() - t0)
    return min(t), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dims", default="1,2,3")
    ap.add_argument("--bounces", default="2,4,8")
    ap.add_argument("--label", default="branch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for dim in map(int, args.dims.split(",")):
        for kb in map(int, args.bounces.split(",")):
            state, cstate, grid, sp, params, static, faces, key = build(dim, args.n, args.steps, kb)
            f = jax.jit(functools.partial(objective, state=state, cstate=cstate, grid=grid, sp=sp,
                                          params=params, static=static, faces=faces, key=key))
            g = jax.jit(jax.value_and_grad(functools.partial(
                objective, state=state, cstate=cstate, grid=grid, sp=sp, params=params,
                static=static, faces=faces, key=key), has_aux=True))
            t0 = time.perf_counter()
            fwd, (j, diag) = timed(f, jnp.float64(273.15))
            t_fwd_compile = time.perf_counter() - t0
            t0 = time.perf_counter()
            rev, ((jg, diag_g), grad) = timed(g, jnp.float64(273.15))
            t_rev_compile = time.perf_counter() - t0
            mem = g.lower(jnp.float64(273.15)).compile().memory_analysis()
            row = {
                "label": args.label, "dim": dim, "k_bounce": kb, "n": args.n, "steps": args.steps,
                "forward_ms_per_step": 1e3 * fwd / args.steps,
                "reverse_ms_per_step": 1e3 * rev / args.steps,
                "forward_compile_s": t_fwd_compile, "reverse_compile_s": t_rev_compile,
                "reverse_temp_bytes": int(mem.temp_size_in_bytes) if mem else None,
                "reverse_argument_bytes": int(mem.argument_size_in_bytes) if mem else None,
                "objective": float(j), "gradient": float(grad),
                "recrossed": int(jnp.sum(diag.wall_recrossed)),
                "clipped": int(jnp.sum(diag.attempts_clipped)),
                "backend": jax.default_backend(), "jax": jax.__version__,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
