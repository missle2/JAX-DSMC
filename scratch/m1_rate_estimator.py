"""Is the M1 rate bias physics, or an artefact of how the rate is FITTED?

`m1_convergence.py` fits `log E(t)` by ordinary least squares over the whole
trajectory.  That is suspect: `E` decays by ~4.5x over the window while the
absolute Monte Carlo noise on it stays roughly constant, so late points have a
much larger *relative* error -- and `log` of a noisy small number is biased
downward by `-sigma^2 / 2E^2`, which steepens the fit and reads the rate HIGH.

Three estimators on the SAME trajectories separate the possibilities:
  ols_full  -- what the convergence study used
  ols_half  -- OLS over the first half only (E still large, noise small)
  ratio     -- log(E(t1)/E(t0)) / (t1 - t0), two early points, no fit at all
If the bias tracks the estimator, it is a fitting artefact.  If all three agree,
it is in the solver.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from examples.bkw import _relax, initial_state, measured_excess, setup

EXACT = 1.0 / 3.0


def trajectories(n_part, n_steps, dt_nu, seeds):
    sp, grid, static, params, cstate, nu, dt, temp = setup(n_part, n_steps, steps_per_tau=dt_nu)
    fn = jax.jit(_relax, static_argnames=("static",))
    out = []
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        state = initial_state(key, static, grid, temp)
        (_f, _), (em, _v, _c) = fn(state, cstate, grid, sp, params, static,
                                   jax.random.fold_in(key, 99))
        e = np.concatenate([[float(measured_excess(state.v))], np.asarray(em)])
        out.append(e)
    return np.array(out), dt * nu


def estimators(e, dtn):
    t = np.arange(e.size) * dtn
    half = e.size // 2
    ols_full = -np.polyfit(t, np.log(e), 1)[0]
    ols_half = -np.polyfit(t[:half], np.log(e[:half]), 1)[0]
    i0, i1 = 1, max(2, half // 2)
    ratio = -(np.log(e[i1]) - np.log(e[i0])) / (t[i1] - t[i0])
    return ols_full, ols_half, ratio


SEEDS = 16
print(f"{SEEDS} seeds per point.  exact rate = {EXACT:.6f} nu")
print("bias % of each estimator; +- is the standard error of the mean\n")

print("=== N sweep, dt*nu = 0.15 ===")
print(f"{'N':>9} | {'ols_full':>17} {'ols_half':>17} {'ratio':>17}")
for N in (10_000, 20_000, 40_000):
    E, dtn = trajectories(N, 30, 0.15, SEEDS)
    cols = []
    for k in range(3):
        v = np.array([estimators(e, dtn)[k] for e in E])
        b = (v.mean() - EXACT) / EXACT * 100
        se = v.std(ddof=1) / np.sqrt(SEEDS) / EXACT * 100
        cols.append(f"{b:+7.2f} +- {se:5.2f}")
    print(f"{N:>9,} | " + " ".join(f"{c:>17}" for c in cols))

print("\n=== dt sweep, N = 20000, same total time ===")
print(f"{'dt*nu':>9} | {'ols_full':>17} {'ols_half':>17} {'ratio':>17}")
for d in (0.30, 0.15, 0.075):
    steps = int(round(4.5 / d))
    E, dtn = trajectories(20_000, steps, d, SEEDS)
    cols = []
    for k in range(3):
        v = np.array([estimators(e, dtn)[k] for e in E])
        b = (v.mean() - EXACT) / EXACT * 100
        se = v.std(ddof=1) / np.sqrt(SEEDS) / EXACT * 100
        cols.append(f"{b:+7.2f} +- {se:5.2f}")
    print(f"{d:>9.3f} | " + " ".join(f"{c:>17}" for c in cols))
