"""M1 BKW convergence, seed-averaged, in ONE process.

Two reasons this is not a shell loop over `python -m examples.bkw`:

1. **A single seed cannot show convergence.** The fitted decay rate is a random
   variable; a one-seed sweep gave 4.13% / 7.50% / 2.17% at N = 20k / 45k / 80k,
   which is noise, not a trend.  We need mean +- sd over seeds.
2. **Each separate process pays a full recompile.**  That is most of the wall
   time at these sizes and it is what makes GPU utilisation look spiky.  Here the
   compile is timed once per shape and reused across seeds.
"""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import bkw
from examples.bkw import _relax, initial_state, measured_excess, setup

SEEDS = 8


def fitted_rate(em, dt, nu):
    t = np.arange(em.size) * dt * nu
    return -np.polyfit(t, np.log(em), 1)[0]


def sweep(n_part, n_steps, dt_nu, seeds=SEEDS):
    sp, grid, static, params, cstate, nu, dt, temp = setup(n_part, n_steps, steps_per_tau=dt_nu)
    fn = jax.jit(_relax, static_argnames=("static",))
    rates, e0s, run_times = [], [], []
    n_viol = n_clip = 0
    t_compile = None
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        state = initial_state(key, static, grid, temp)
        t0 = time.perf_counter()
        (_f, _), (em, viol, clip) = fn(state, cstate, grid, sp, params, static,
                                       jax.random.fold_in(key, 99))
        em.block_until_ready()
        elapsed = time.perf_counter() - t0
        if t_compile is None:
            t_compile = elapsed          # first call includes tracing + XLA compile
        else:
            run_times.append(elapsed)    # subsequent calls hit the compiled kernel
        e = np.concatenate([[float(measured_excess(state.v))], np.asarray(em)])
        rates.append(fitted_rate(e, dt, nu))
        e0s.append(math.exp(np.polyfit(np.arange(e.size) * dt * nu, np.log(e), 1)[1]))
        n_viol += int(np.sum(viol))
        n_clip += int(np.sum(clip))
    return (np.array(rates), np.array(e0s), n_viol, n_clip, static.A_max,
            t_compile, float(np.mean(run_times)))


EXACT = 1.0 / 3.0
print(f"BKW M1 convergence, {SEEDS} seeds per point.  exact rate = {EXACT:.6f} nu\n")
print("=== statistical convergence: N sweep at dt*nu = 0.15 ===")
print(f"{'N':>9} {'A_max':>8} {'rate mean':>11} {'rate sd':>10} {'bias %':>9} "
      f"{'sd %':>8} {'viol':>6} {'clip':>6} {'compile s':>10} {'run s':>8}")
prev = None
for N in (10_000, 20_000, 40_000):
    r, e0, v, c, amax, tc, tr = sweep(N, 30, 0.15)
    bias = (r.mean() - EXACT) / EXACT * 100
    sd = r.std(ddof=1) / EXACT * 100
    print(f"{N:>9,} {amax:>8,} {r.mean():>11.6f} {r.std(ddof=1):>10.6f} {bias:>9.3f} "
          f"{sd:>8.3f} {v:>6} {c:>6} {tc:>10.2f} {tr:>8.3f}")
    if prev is not None:
        print(f"{'':>9} {'':>8} sd ratio vs previous N = {prev / r.std(ddof=1):.3f}  "
              f"(1/sqrt(2) scaling predicts 1.414)")
    prev = r.std(ddof=1)

print("\n=== discretisation: dt sweep at N = 20000, same total time t*nu = 4.5 ===")
print(f"{'dt*nu':>8} {'steps':>7} {'A_max':>8} {'rate mean':>11} {'rate sd':>10} {'bias %':>9}")
for d in (0.30, 0.15, 0.075):
    steps = int(round(4.5 / d))
    r, e0, v, c, amax, tc, tr = sweep(20_000, steps, d)
    print(f"{d:>8.3f} {steps:>7} {amax:>8,} {r.mean():>11.6f} {r.std(ddof=1):>10.6f} "
          f"{(r.mean() - EXACT) / EXACT * 100:>9.3f}")
