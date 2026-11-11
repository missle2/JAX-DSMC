"""One relax step from the state after 6 steps: AD vs FD of a random projection of v."""
import dataclasses
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
import test_learned_window as tw
from jax_dsmc import simulate, sort
from jax_dsmc.collide import relax

states, cstates, keys, ref, case = tw._window()
state = jax.tree.map(lambda a: a[0], states)  # noqa: E731
cstate = jax.tree.map(lambda a: a[0], cstates)  # noqa: E731
key = keys[0]
theta0 = {"log_tau": jnp.float64(0.0), "log_pr": jnp.float64(np.log(2 / 3))}
for k0 in range(0, 9):
    st6 = dataclasses.replace(case.static, n_steps=1)
    w = jax.random.normal(jax.random.PRNGKey(99), state.v.shape)

    def f(log_tau, state=state, cstate=cstate, k0=k0):
        th = {"log_tau": log_tau, "log_pr": theta0["log_pr"]}
        (s, _), _ = simulate.run(state, cstate, case.grid, case.sp, case.params, st6, key,
                                 step0=k0, emit_moments="none", collide_fn=relax.collide,
                                 collision_params=th)
        return jnp.sum(jnp.where(s.alive[:, None], s.v * w, 0.0))

    g = float(jax.grad(f)(0.0))
    fd = float((f(1e-5) - f(-1e-5)) / 2e-5)
    table = sort.build_table(state, case.grid)
    counts = np.asarray(table.counts)
    print(f"step {k0}: AD {g:.6e} FD {fd:.6e}  counts min {counts.min()} max {counts.max()}", flush=True)
    (state, cstate), _ = simulate.run(state, cstate, case.grid, case.sp, case.params, st6, key,
                                      step0=k0, emit_moments="none", collide_fn=relax.collide,
                                      collision_params=theta0)
