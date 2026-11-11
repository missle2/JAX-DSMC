"""Locate the cell whose relax-step derivative is wrong at step 8 and print its state."""
import dataclasses
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
lp = jnp.float64(np.log(2 / 3))
st1 = dataclasses.replace(case.static, n_steps=1)
for k0 in range(8):
    (state, cstate), _ = simulate.run(state, cstate, case.grid, case.sp, case.params, st1, key,
                                      step0=k0, emit_moments="none", collide_fn=relax.collide,
                                      collision_params={"log_tau": 0.0, "log_pr": lp})
table = sort.build_table(state, case.grid)


def collide_v(log_tau):
    (s, _), _ = simulate.run(state, cstate, case.grid, case.sp, case.params, st1, key, step0=8,
                             emit_moments="none", collide_fn=relax.collide,
                             collision_params={"log_tau": log_tau, "log_pr": lp})
    return s.v


_, jv = jax.jvp(collide_v, (jnp.float64(0.0),), (jnp.float64(1.0),))
fd = (collide_v(1e-6) - collide_v(-1e-6)) / 2e-6
rel = np.abs(np.asarray(jv - fd)).max(axis=1) / np.abs(np.asarray(fd)).max()
(s8, _), _ = simulate.run(state, cstate, case.grid, case.sp, case.params, st1, key, step0=8,
                          emit_moments="none", collide_fn=relax.collide,
                          collision_params={"log_tau": 0.0, "log_pr": lp})
t8 = sort.build_table(s8, case.grid)
cell = np.asarray(sort.build_table(s8, case.grid).cell)
for cc in range(case.n_coarse):
    print(cc, "count", int(t8.counts[cc]), "max rel err", rel[cell == cc].max())
bad = int(np.argmax(rel))
print("worst particle", bad, "cell", cell[bad], "alive", bool(s8.alive[bad]))
print("jvp", np.asarray(jv[bad]), "fd", np.asarray(fd[bad]))
