"""Replicate step 4 for both theta and diff EVERY intermediate array."""
import sys; sys.path.insert(0, ".")
import dataclasses, functools
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax import random
from jax_dsmc import sample, sort
from jax_dsmc.boundary import BC
from jax_dsmc.collide import vremax_init
from jax_dsmc.collide.ntc import attempt_count, pair_from_draws, thermal_majorant
from jax_dsmc.collide.vss import acceptance_prob, collision_rate, frame_is_well_defined
from jax_dsmc.grid import cell_volume, num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.move import move_with_decisions
from jax_dsmc.rng import STAGE_ACCEPT, STAGE_NTC_ROUND, STAGE_PAIR, key_for
from jax_dsmc.simulate import run
from jax_dsmc.state import CollideState
from examples import shock_adjoint as sa

cs = sa.case(1, npc=20, n_steps=4, n_x=8, transverse=2, a_max=256)
th = sa.reference_theta()
key = jax.random.PRNGKey(0)


def after(delta, steps):
    theta = sa.perturb(th, "temp", delta)
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    st0 = initial_state(key, flow, grid, sp, static)
    C = num_cells(grid)
    cst = CollideState(vremax=vremax_init(jnp.full((C,), theta["temp"]), sp.omega[0, 0],
                                          sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0], C, 1))
    if steps:
        stat = dataclasses.replace(static, n_steps=steps)
        (st0, cst), _ = run(st0, cst, grid, sp, params, stat, key, bc=BC.SPECULAR,
                            emit_moments="none")
    return st0, cst, grid, sp, params, static


def probe(delta, k=3):
    st, cst, grid, sp, params, static = after(delta, k)
    st, fold = move_with_decisions(st, grid, params.dt, BC.SPECULAR)
    tab = sort.build_table(st, grid)
    m = sample.moments(st, tab, grid, params, sp)
    sigma = jnp.maximum(cst.vremax[:, 0, 0],
                        thermal_majorant(m.temp, sp.omega[0, 0], sp.tref[0, 0],
                                         sp.mr[0, 0], sp.prefactor[0, 0]))
    n, _ = attempt_count(tab.counts, sigma, cell_volume(grid), params.dt, params.fnum,
                         key_for(key, k, STAGE_NTC_ROUND))
    A, C = static.A_max, tab.counts.shape[0]
    npc_safe = jnp.maximum(tab.counts, 2)
    k_pi, k_pr = random.split(key_for(key, k, STAGE_PAIR))
    u_i = random.randint(k_pi, (A, C), 0, npc_safe)
    u_r = random.randint(k_pr, (A, C), 0, npc_safe - 1)
    u_acc = random.uniform(key_for(key, k, STAGE_ACCEPT), (A, C), dtype=jnp.float64)
    i, j = pair_from_draws(u_i, u_r, tab.counts)
    active = (jnp.arange(A)[:, None] < n) & (tab.counts >= 2)
    pi = jnp.where(active, tab.order[tab.starts + i], static.N)
    pj = jnp.where(active, tab.order[tab.starts + j], static.N)
    v0 = jnp.concatenate([st.v, jnp.zeros((1, 3))])
    q = collision_rate(v0[pi], v0[pj], 0, 0, sp)
    accept = active & (u_acc < acceptance_prob(q, sigma))
    ok = frame_is_well_defined(v0[pi], v0[pj])
    return dict(fold=fold, cell=tab.cell, counts=tab.counts, order=tab.order,
                starts=tab.starts, n=n, sigma=sigma, u_i=u_i, u_r=u_r, pi=pi, pj=pj,
                active=active, accept=accept, ok=ok, q=q)


A, B = probe(-7.5e-5), probe(-5.0e-5)
print(f"{'array':>10} {'identical':>10} {'n differ':>9}")
for name in A:
    a, b = np.asarray(A[name]), np.asarray(B[name])
    same = np.array_equal(a, b)
    nd = int(np.sum(a != b))
    print(f"{name:>10} {str(same):>10} {nd:9d}")
    if not same and name in ("pi", "pj", "accept", "ok", "active", "n", "cell", "fold"):
        idx = np.argwhere(a != b)[:5]
        print(f"           first diffs at {idx.tolist()} : {a[tuple(idx.T)]} vs {b[tuple(idx.T)]}")
