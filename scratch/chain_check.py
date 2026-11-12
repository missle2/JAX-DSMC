"""The chains advance (each window's end state is the chain's next start) and refresh."""

import sys

import jax
import numpy as np

from examples.learned_collision import train

args = train.parser().parse_args(sys.argv[1:])
theta = train.operators.operator("linear", "shock", **train.model_options(args))[1]
r = train.CaseRunner("K3", "linear", args)
x0 = np.asarray(r.chains[0].x)
v0 = np.asarray(r.chains[0].v)
rng = np.random.default_rng(0)
_, _, _, _ = r.windows(theta, rng, "pair", 2)
moved = [not np.array_equal(np.asarray(r.chains[0].v[i]), v0[i]) for i in range(len(v0))]
print("chains changed:", moved, "refreshed:", r.refreshed)
print("positions home", r.x_home, "now", r.positions(r.chains[0]))
# the same windows again from the end states: a pure continuation
st, cs, keys, tg, rows, idx = r.draw_batch(rng, 2)
print("drawn chains", np.asarray(idx), "x equal to chain:",
      bool(np.array_equal(np.asarray(st.x).reshape(-1, *x0.shape[1:]), np.asarray(r.chains[0].x)[np.asarray(idx)])))
print("cursor", r.cursor, jax.tree.structure(r.chains) == jax.tree.structure(r.chains))
