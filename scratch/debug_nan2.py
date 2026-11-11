"""Reproduce the NaN window (5th draw, rng seed 0) and bisect the step that makes it."""
import argparse
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import train_scalar as ts
from examples.learned_collision import windows
from examples.learned_collision.shear_wave import _stack
from jax_dsmc import simulate, sort
from jax_dsmc.collide import relax

args = argparse.Namespace(dx=4.0, ppc=10, blocks=5, burn=2)
ref, meta = ts.load("reference_thermal_L64_v1.npz")
collide_fn, theta, _ = ts.operator("esfp")
case = windows.coarse_case(meta["wave"], meta["length_mfp"], 4.0, 10, ref["snap_x"].shape[2])
rng = np.random.default_rng(0)
snaps = ref["snap_x"].shape[1]
snap_every = int(np.round((ref["snap_t"][1] - ref["snap_t"][0]) / (ref["t"][1] - ref["t"][0])))
last = (len(ref["t"]) - 1 - args.burn - args.blocks) // snap_every
for i in range(5):
    s = int(rng.integers(0, min(snaps, last + 1)))
    seed = int(rng.integers(0, ref["snap_x"].shape[0]))
    k = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
k_thin, k_run = jax.random.split(k)
inits = [windows.initial_state(ref["snap_x"][seed, s], ref["snap_v"][seed, s], case,
                               jax.random.fold_in(k_thin, r)) for r in range(2)]
states = jax.tree.map(_stack, *[x[0] for x in inits])
cstates = jax.tree.map(_stack, *[x[1] for x in inits])
keys = jnp.stack([jax.random.fold_in(k_run, r) for r in range(2)])
burn = jax.vmap(functools.partial(windows.burn_in, case=case, collide_fn=collide_fn,
                                  blocks=args.burn), in_axes=(None, 0, 0, 0))
states, cstates = burn(theta, states, cstates, keys)
one = dataclasses.replace(case.static, n_steps=1)


def step(st, cs, key, k0):
    (s2, c2), _ = simulate.run(st, cs, case.grid, case.sp, case.params, one, key, step0=k0,
                               emit_moments="none", collide_fn=collide_fn,
                               collision_params=theta)
    return s2, c2


def diag(st):
    tb = sort.build_table(st, case.grid)
    seg = jnp.where(st.alive, tb.cell, case.n_coarse)
    n, u, pi, th = relax.cell_moments(st.v, st.alive, seg, case.n_coarse)
    pid = pi - th[:, None, None] * jnp.eye(3)
    w = jnp.linalg.eigvalsh(pid)
    gap = jnp.min(jnp.stack([w[:, 2] - w[:, 1], w[:, 1] - w[:, 0]], 1), 1) / th
    a, _ = relax.step_coefficients(n, pi, th, case.grid, case.sp, case.params, theta, False)
    return n.min(), gap.min(), th.min(), w[:, 2].max() / th[jnp.argmax(w[:, 2])]


jstep, jdiag = jax.jit(step), jax.jit(diag)
for r in range(2):
    st = jax.tree.map(lambda a: a[r], states)  # noqa: E731
    cs = jax.tree.map(lambda a: a[r], cstates)  # noqa: E731
    for k in range(60):
        nmin, gmin, tmin, pdmax = (float(x) for x in jdiag(st))
        if k >= 35 or gmin < 1e-6:
            print(f"r{r} step {40 + k}: min count {nmin:.0f}  min eig gap/theta {gmin:.2e}  "
                  f"theta min {tmin:.3e}  max pd/theta {pdmax:.3f}", flush=True)
        st, cs = jstep(st, cs, keys[r], 40 + k)
