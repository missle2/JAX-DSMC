"""Where does a full timestep's time go, and what breaks first at 1e10-1e11?

Profiles the three stages of `simulate._step` separately -- `sort.build_table`,
`sample.moments`, `collide` -- because the aggregate number hides which one is
the scaling problem.  Then projects memory to the target scale.

Run on an IDLE device.  Concurrent GPU work has already produced one phantom
"36x cliff" in this project (spec 8.6).
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sample, sampling
from jax_dsmc.collide import make_prefactor
from jax_dsmc.collide.ntc import collide, vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.sort import build_table
from jax_dsmc.state import (CollideState, GridSpec, ParticleState, SimParams,
                            SimStatic, SpeciesParams)

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def build(npc, side, A_max, T=300.0):
    """A production-shaped case: `npc` particles per cell, `side**3` cells."""
    n = (side, side, side)
    C = side ** 3
    N = C * npc
    mass = jnp.array([AR_MASS]); mr = jnp.array([[AR_MASS / 2]])
    diam = jnp.array([[AR_DIAM]]); om = jnp.array([[0.81]])
    tref = jnp.array([[AR_TREF]]); al = jnp.array([[1.4]])
    sp = SpeciesParams(mass=mass, diam=diam, omega=om, tref=tref, alpha=al, mr=mr,
                       prefactor=make_prefactor(diam, om, tref, mr))
    g = GridSpec(lo=jnp.zeros(3), hi=jnp.full(3, 1e-4), n=n)
    st = SimStatic(N=N, D=3, S=1, G=1, A_max=A_max, K_bounce=1, n_steps=1)
    pr = SimParams(dt=jnp.float64(1e-9), fnum=jnp.ones(C), epsilon=jnp.float64(0.0))
    k = jax.random.PRNGKey(0)
    x = sampling.sample_positions(jax.random.fold_in(k, 1), N, g)
    v = sampling.sample_maxwellian(jax.random.fold_in(k, 2), N, jnp.float64(T),
                                   mass[0], jnp.zeros(3))
    state = ParticleState(x=x, v=v, species=jnp.zeros(N, jnp.int32),
                          alive=jnp.ones(N, bool), logh_acc=jnp.zeros(N))
    cs = CollideState(vremax=vremax_init(jnp.full((C,), T), om[0, 0], tref[0, 0],
                                         mr[0, 0], sp.prefactor[0, 0], C, 1))
    return state, cs, g, sp, pr, st, jnp.full((C,), T), k, C, N


def bench(fn, *a, reps=7, **kw):
    out = fn(*a, **kw)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


J_SORT = jax.jit(build_table)
J_MOM = jax.jit(sample.moments)
J_COL = jax.jit(collide, static_argnames=("static",))

print("=== stage breakdown, production shape: npc = 32, A_max = 64 ===")
print(f"{'N':>12} {'cells':>9} {'sort ms':>9} {'moments ms':>11} {'collide ms':>11} "
      f"{'total ms':>10} {'ns/part':>9}")
rows = []
for side in (8, 16, 32, 48):
    state, cs, g, sp, pr, st, T, k, C, N = build(32, side, 64)
    t_sort = bench(J_SORT, state, g)
    tb = build_table(state, g)
    t_mom = bench(J_MOM, state, tb, g, pr, sp)
    t_col = bench(J_COL, state, tb, cs, g, sp, pr, st, T, 0, k)
    tot = t_sort + t_mom + t_col
    rows.append((N, C, t_sort, t_mom, t_col))
    print(f"{N:>12,} {C:>9,} {t_sort * 1e3:>9.2f} {t_mom * 1e3:>11.2f} {t_col * 1e3:>11.2f} "
          f"{tot * 1e3:>10.2f} {tot / N * 1e9:>9.1f}")

print("\n  share of the step, largest case:")
N, C, ts, tm, tc = rows[-1]
tot = ts + tm + tc
for name, t in (("sort", ts), ("moments", tm), ("collide", tc)):
    print(f"    {name:<9} {t / tot * 100:5.1f} %")

print("\n  scaling exponent (time ~ N^p; 1.0 is ideal, >1 is a scaling defect):")
Ns = np.array([r[0] for r in rows], float)
for i, name in enumerate(("sort", "moments", "collide")):
    y = np.array([r[2 + i] for r in rows])
    p = np.polyfit(np.log(Ns), np.log(y), 1)[0]
    print(f"    {name:<9} N^({p:.3f})")

print("\n=== how much of `moments` is wasted? ===")
print("`simulate._step` calls the full `moments` every step but uses ONLY `temp`.")
print("The pressure tensor builds an (N,3,3) intermediate -- 9 f64 per particle.\n")


def temp_only(state, table, grid, params, species):
    """What the majorant actually needs: count, bulk velocity, temperature."""
    from jax_dsmc.grid import cell_of  # noqa: PLC0415
    from jax_dsmc.safe import safe_recip_count  # noqa: PLC0415
    from jax_dsmc.constants import K_BOLTZ  # noqa: PLC0415
    C = num_cells(grid)
    c = cell_of(state.x, grid)
    w = state.alive.astype(jnp.float64)
    cnt = jax.ops.segment_sum(w, c, num_segments=C)
    inv = safe_recip_count(cnt)
    u = jax.ops.segment_sum(w[:, None] * state.v, c, num_segments=C) * inv[:, None]
    cpec = state.v - u[c]
    e = jax.ops.segment_sum(w * jnp.sum(cpec * cpec, -1), c, num_segments=C) * inv
    return species.mass[0] * e / (3.0 * K_BOLTZ)


J_TEMP = jax.jit(temp_only)
print(f"{'N':>12} {'moments ms':>11} {'temp-only ms':>13} {'speedup':>9}")
for side in (16, 32, 48):
    state, cs, g, sp, pr, st, T, k, C, N = build(32, side, 64)
    tb = build_table(state, g)
    tm = bench(J_MOM, state, tb, g, pr, sp)
    tt = bench(J_TEMP, state, tb, g, pr, sp)
    # agreement check -- the optimisation must not change the answer
    a = np.asarray(J_MOM(state, tb, g, pr, sp).temp)
    b = np.asarray(J_TEMP(state, tb, g, pr, sp))
    assert np.allclose(a, b, rtol=1e-12), "temp-only disagrees with moments!"
    print(f"{N:>12,} {tm * 1e3:>11.2f} {tt * 1e3:>13.2f} {tm / tt:>9.2f}x")

print("\n=== memory footprint of the particle state ===")
per = {"x (N,3) f64": 24, "v (N,3) f64": 24, "species (N,) i32": 4,
       "alive (N,) bool": 1, "logh_acc (N,) f64": 8}
tot_b = sum(per.values())
print(f"  per particle: {tot_b} bytes  (" + ", ".join(f"{k}={v}" for k, v in per.items()) + ")")
print(f"\n{'N':>14} {'state GB':>10} {'+scratch copy':>14} {'sort tables GB':>15} {'total GB':>10}")
for N in (1e8, 1e9, 1e10, 1e11):
    st_gb = N * tot_b / 1e9
    scratch = N * 32 / 1e9          # the (N+1,3) v copy plus (N+1) logh copy per step
    tables = N * 8 / 1e9            # order i32 + cell i32
    print(f"{N:>14,.0f} {st_gb:>10.1f} {scratch:>14.1f} {tables:>15.1f} "
          f"{st_gb + scratch + tables:>10.1f}")
print("\n  (single device HBM today is ~80-192 GB, so 1e9 is the single-device ceiling")
print("   and 1e10+ requires sharding -- decision 7, deferred but load-bearing on layout.)")
