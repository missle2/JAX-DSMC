"""Where does the collide step's time actually go?

Hypothesis: each attempt slot does `v.at[pi].set(...)` on the full (N+1, 3)
velocity array.  `pi` holds only C indices, so the *update* is O(C) -- but if XLA
materialises a fresh (N+1, 3) buffer per scan iteration instead of updating the
carry in place, the per-slot cost becomes O(N).  In the 0D milestone-1 case
A_max ~ N and C = 1, so that would make one step O(N^2) with zero parallelism.

Three measurements separate the possibilities:
  A. fix A_max, sweep N   -> per-slot cost vs N.  Flat = in-place, linear = copy.
  B. fix N, sweep A_max   -> per-slot cost floor (kernel launch latency).
  C. fix N and A_max, sweep C -> does work across cells actually parallelise?
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc.collide import make_prefactor
from jax_dsmc.collide.ntc import collide, vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.sort import build_table
from jax_dsmc.state import (CollideState, GridSpec, ParticleState, SimParams,
                            SimStatic, SpeciesParams)
from jax_dsmc import sampling

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def case(N, n, A_max, T=300.0):
    mass = jnp.array([AR_MASS]); mr = jnp.array([[AR_MASS / 2]])
    diam = jnp.array([[AR_DIAM]]); om = jnp.array([[1.0]])
    tref = jnp.array([[AR_TREF]]); al = jnp.array([[1.0]])
    sp = SpeciesParams(mass=mass, diam=diam, omega=om, tref=tref, alpha=al, mr=mr,
                       prefactor=make_prefactor(diam, om, tref, mr))
    g = GridSpec(lo=jnp.zeros(3), hi=jnp.full(3, 1e-4), n=n)
    C = num_cells(g)
    st = SimStatic(N=N, D=3, S=1, G=1, A_max=A_max, K_bounce=1, n_steps=1)
    pr = SimParams(dt=jnp.float64(1e-8), fnum=jnp.ones(C), epsilon=jnp.float64(0.0))
    k = jax.random.PRNGKey(0)
    x = sampling.sample_positions(jax.random.fold_in(k, 1), N, g)
    v = sampling.sample_maxwellian(jax.random.fold_in(k, 2), N, jnp.float64(T), mass[0], jnp.zeros(3))
    state = ParticleState(x=x, v=v, species=jnp.zeros(N, jnp.int32),
                          alive=jnp.ones(N, bool), logh_acc=jnp.zeros(N))
    cs = CollideState(vremax=vremax_init(jnp.full((C,), T), om[0, 0], tref[0, 0],
                                         mr[0, 0], sp.prefactor[0, 0], C, 1))
    return state, cs, g, sp, pr, st, jnp.full((C,), T), k, C


FN = jax.jit(collide, static_argnames=("static",))


def timeit(N, n, A_max, reps=5):
    state, cs, g, sp, pr, st, T, k, C = case(N, n, A_max)
    tb = build_table(state, g)
    t0 = time.perf_counter()
    out = FN(state, tb, cs, g, sp, pr, st, T, 0, k)
    out[0].v.block_until_ready()
    t_compile = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = FN(state, tb, cs, g, sp, pr, st, T, 0, k)
        out[0].v.block_until_ready()
        ts.append(time.perf_counter() - t0)
    return t_compile, float(np.median(ts)), C


if __name__ == "__main__":
    print("=== A: fix A_max = 512, sweep N.  Per-slot cost should be O(C)=O(1), NOT O(N). ===")
    print(f"{'N':>10} {'C':>5} {'compile s':>10} {'run s':>9} {'us/slot':>9} {'vs N=8k':>9}")
    base = None
    for N in (8_000, 32_000, 128_000, 512_000):
        tc, tr, C = timeit(N, (1, 1, 1), 512)
        per = tr / 512 * 1e6
        base = per if base is None else base
        print(f"{N:>10,} {C:>5} {tc:>10.2f} {tr:>9.4f} {per:>9.1f} {per / base:>9.2f}x")

    print("\n=== B: fix N = 32000, sweep A_max.  Slope = per-slot cost, intercept = fixed overhead. ===")
    print(f"{'A_max':>10} {'run s':>9} {'us/slot':>9}")
    for A in (128, 512, 2048, 8192):
        tc, tr, C = timeit(32_000, (1, 1, 1), A)
        print(f"{A:>10,} {tr:>9.4f} {tr / A * 1e6:>9.1f}")

    print("\n=== C: fix N = 128000 and A_max = 512, sweep cell count.  Cells are the parallel axis. ===")
    print(f"{'cells C':>10} {'run s':>9} {'us/slot':>9} {'vs C=1':>9}")
    base = None
    for n in ((1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 16, 16)):
        tc, tr, C = timeit(128_000, n, 512)
        per = tr / 512 * 1e6
        base = per if base is None else base
        print(f"{C:>10,} {tr:>9.4f} {per:>9.1f} {per / base:>9.2f}x")
