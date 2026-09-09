"""Audit for structural performance defects of the same class as the RNG-in-scan bug.

Not a scaling study.  The question is narrow: **is any work sitting somewhere it
does not belong?**  Four suspects, each isolated by ablation so the answer is a
measurement rather than an opinion.

  A. Does the attempt-slot scan copy the whole (N+1,3) velocity carry per
     iteration, or update it in place?  If it copies, the step is O(N*A_max)
     instead of O(C*A_max) -- the same shape of defect as the RNG draws.
  B. `simulate._step` calls the full `sample.moments` every step and uses only
     `temp`.  The pressure tensor builds an (N,3,3) intermediate.
  C. `moments` recomputes `cell_of` although `table.cell` already holds it.
  D. `collide` concatenates a scratch row onto v and logh every call.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sample, sampling
from jax_dsmc.collide import make_prefactor
from jax_dsmc.collide.ntc import collide, vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_of, num_cells
from jax_dsmc.safe import safe_recip_count
from jax_dsmc.sort import build_table
from jax_dsmc.state import (CollideState, GridSpec, ParticleState, SimParams,
                            SimStatic, SpeciesParams)

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def build(N, side, A_max, T=300.0):
    n = (side, side, side)
    C = side ** 3
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
    return state, cs, g, sp, pr, st, jnp.full((C,), T), k, C


def bench(fn, *a, reps=7, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------- A ---------
# Ablation: the same scan shape, but the body either scatters into an (N+1,3)
# carry (as `collide` does) or accumulates into a (C,) carry only.  If the
# scatter version's per-slot cost grows with N while the other's does not, the
# carry is being copied rather than updated in place.

def _body_scatter(carry, xs, C):
    v, acc = carry
    idx, u = xs
    vi = v[idx]
    v = v.at[idx].set(vi + u[:, None])
    return (v, acc + jnp.sum(vi)), None


def _body_nocarry(carry, xs, C):
    v, acc = carry
    idx, u = xs
    vi = v[idx]
    return (v, acc + jnp.sum(vi) + jnp.sum(u)), None


def run_ablation(N, C, A, body):
    v = jnp.zeros((N + 1, 3))
    idx = jnp.zeros((A, C), jnp.int32)
    u = jnp.ones((A, C))
    import functools  # noqa: PLC0415
    f = functools.partial(body, C=C)
    return lax.scan(f, (v, jnp.float64(0.0)), (idx, u))[0][1]


J_SC = jax.jit(run_ablation, static_argnums=(0, 1, 2, 3))
print("=== A. is the (N+1,3) velocity carry updated in place, or copied? ===")
print("Same scan, C=1, A_max=256.  'scatter' writes the carry; 'read-only' does not.\n")
print(f"{'N':>10} {'scatter us/slot':>17} {'read-only us/slot':>19} {'ratio':>8}")
for N in (8_000, 64_000, 512_000, 2_000_000):
    t1 = bench(J_SC, N, 1, 256, _body_scatter)
    t2 = bench(J_SC, N, 1, 256, _body_nocarry)
    print(f"{N:>10,} {t1 / 256 * 1e6:>17.1f} {t2 / 256 * 1e6:>19.1f} {t1 / t2:>8.2f}x")
print("\n  If 'scatter' grows with N and 'read-only' does not, the carry is copied.")

# ---------------------------------------------------------------- B, C ------
def temp_only(state, table, grid, params, species):
    """Only what the majorant needs.  Reuses `table.cell` instead of recomputing."""
    C = num_cells(grid)
    c = table.cell
    w = state.alive.astype(jnp.float64)
    cnt = jax.ops.segment_sum(w, c, num_segments=C)
    inv = safe_recip_count(cnt)
    u = jax.ops.segment_sum(w[:, None] * state.v, c, num_segments=C) * inv[:, None]
    cpec = state.v - u[c]
    e = jax.ops.segment_sum(w * jnp.sum(cpec * cpec, -1), c, num_segments=C) * inv
    return species.mass[0] * e / (3.0 * K_BOLTZ)


def temp_recompute(state, table, grid, params, species):
    """Same, but recomputing `cell_of` the way `moments` does."""
    C = num_cells(grid)
    c = cell_of(state.x, grid)
    w = state.alive.astype(jnp.float64)
    cnt = jax.ops.segment_sum(w, c, num_segments=C)
    inv = safe_recip_count(cnt)
    u = jax.ops.segment_sum(w[:, None] * state.v, c, num_segments=C) * inv[:, None]
    cpec = state.v - u[c]
    e = jax.ops.segment_sum(w * jnp.sum(cpec * cpec, -1), c, num_segments=C) * inv
    return species.mass[0] * e / (3.0 * K_BOLTZ)


J_MOM = jax.jit(sample.moments)
J_T = jax.jit(temp_only)
J_TR = jax.jit(temp_recompute)

print("\n=== B/C. `_step` calls full `moments` every step but uses only `temp` ===")
print(f"{'N':>10} {'cells':>7} {'moments ms':>11} {'temp+reuse ms':>14} {'temp+recompute':>15} "
      f"{'B: saving':>10} {'C: saving':>10}")
for side, npc in ((16, 16), (24, 24), (32, 32)):
    C = side ** 3
    N = C * npc
    state, cs, g, sp, pr, st, T, k, _C = build(N, side, 32)
    tb = build_table(state, g)
    tm = bench(J_MOM, state, tb, g, pr, sp)
    tt = bench(J_T, state, tb, g, pr, sp)
    tr = bench(J_TR, state, tb, g, pr, sp)
    a = np.asarray(J_MOM(state, tb, g, pr, sp).temp)
    b = np.asarray(J_T(state, tb, g, pr, sp))
    assert np.allclose(a, b, rtol=1e-11), "temp-only disagrees with moments"
    print(f"{N:>10,} {C:>7,} {tm * 1e3:>11.2f} {tt * 1e3:>14.2f} {tr * 1e3:>15.2f} "
          f"{tm / tt:>9.2f}x {tr / tt:>9.2f}x")

# ---------------------------------------------------------------- D ---------
print("\n=== D. cost of the per-call scratch-row concatenate ===")


def with_concat(v, logh):
    a = jnp.concatenate([v, jnp.zeros((1, 3))])
    b = jnp.concatenate([logh, jnp.zeros((1,))])
    return jnp.sum(a) + jnp.sum(b)


def without(v, logh):
    return jnp.sum(v) + jnp.sum(logh)


JC, JW = jax.jit(with_concat), jax.jit(without)
print(f"{'N':>10} {'concat ms':>10} {'baseline ms':>12} {'overhead ms':>12}")
for N in (100_000, 1_000_000, 4_000_000):
    v = jnp.zeros((N, 3)); lg = jnp.zeros(N)
    tc = bench(JC, v, lg)
    tw = bench(JW, v, lg)
    print(f"{N:>10,} {tc * 1e3:>10.3f} {tw * 1e3:>12.3f} {(tc - tw) * 1e3:>12.3f}")
