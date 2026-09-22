"""Cost of the device ensemble backward vs the forward (with and without capture)
vs the pathwise VJP, on a Couette-like case.  Reports ms/step and compiled memory."""

import argparse
import functools
import json
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from examples.gradient_convergence import (  # noqa: E402
    COUETTE_CONTROLS,
    couette_run,
    couette_species,
)
from jax_dsmc import ensemble_device  # noqa: E402


def timed(fn, *args, reps=3):
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def main():  # noqa: PLR0915 -- one timing script
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--attempts", type=int, default=64)
    ap.add_argument("--npc-max", type=int, default=256)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--pullback", default="sequential", choices=("sequential", "virtual"))
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--partners", type=int, default=8)
    ap.add_argument("--nodes", default="staged", choices=("post", "staged"))
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t = dict(COUETTE_CONTROLS)
    key = jax.random.key(0)
    run = functools.partial(
        couette_run,
        particles=args.n,
        cells=args.cells,
        steps=args.steps,
        attempts=args.attempts,
        dt=0.05,
        epsilon=0.0,
        majorant="certified",
        score_mode="paper",
    )
    fwd = jax.jit(lambda t, key: run(t, key, capture=False)[1].v)
    fwd_cap = jax.jit(lambda t, key: run(t, key, capture=True))
    t_fwd, _ = timed(fwd, t, key)
    t_cap, (state0, out, diag, rec, grid) = timed(fwd_cap, t, key)
    phi_final = out.v[:, 1] ** 2 / args.n
    dx = (grid.hi - grid.lo) / jnp.asarray(grid.n)
    scale = jnp.concatenate((dx, jnp.ones(3)))
    if args.pullback == "virtual":
        bwd = functools.partial(
            ensemble_device.backward_virtual,
            npc_max=args.npc_max,
            k=args.k,
            chunk=args.chunk,
            substeps=args.substeps,
            partners=args.partners,
            nodes=args.nodes,
        )
        bwd_args = (rec, state0.alive, out, phi_final, grid, scale, couette_species(t), key)
    else:
        bwd = functools.partial(ensemble_device.backward, npc_max=args.npc_max, k=args.k)
        bwd_args = (rec, state0.alive, out, phi_final, grid, scale)
    t0 = time.perf_counter()
    t_bwd, res = timed(bwd, *bwd_args)
    compile_bwd = time.perf_counter() - t0
    mem = jax.jit(bwd).lower(*bwd_args).compile().memory_analysis()

    # pathwise VJP of the same forward for reference
    def objective(speed):
        tt = {**t, "speed": speed}
        _, o, _, _, _ = run(tt, key, capture=False)
        return jnp.mean(o.v[:, 1] ** 2)

    g = jax.jit(jax.grad(objective))
    t_vjp, _ = timed(g, jnp.float64(1.0))
    row = {
        "n": args.n,
        "cells": args.cells,
        "npc": args.n / args.cells,
        "steps": args.steps,
        "attempts": args.attempts,
        "npc_max": args.npc_max,
        "k": args.k,
        "pullback": args.pullback,
        "substeps": args.substeps if args.pullback == "virtual" else None,
        "partners": args.partners if args.pullback == "virtual" else None,
        "nodes": args.nodes if args.pullback == "virtual" else None,
        "forward_ms_per_step": 1e3 * t_fwd / args.steps,
        "forward_capture_ms_per_step": 1e3 * t_cap / args.steps,
        "ensemble_backward_ms_per_step": 1e3 * t_bwd / args.steps,
        "pathwise_vjp_ms_per_step": 1e3 * t_vjp / args.steps,
        "backward_compile_s": compile_bwd,
        "backward_temp_MB": mem.temp_size_in_bytes / 1e6 if mem else None,
        "backward_argument_MB": mem.argument_size_in_bytes / 1e6 if mem else None,
        "accepted_pairs_per_step": float(jnp.mean(jnp.sum(rec.pairs.accept, axis=(1, 2)))),
        "max_attempts": int(jnp.max(diag.max_attempts)),
        "clipped": int(jnp.sum(diag.attempts_clipped)),
        "lambda0_finite": bool(jnp.all(jnp.isfinite(res.lambda0))),
        "backend": jax.default_backend(),
        "jax": jax.__version__,
    }
    print(json.dumps(row, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=1)


if __name__ == "__main__":
    main()
