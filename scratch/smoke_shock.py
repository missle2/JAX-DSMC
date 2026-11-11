"""Smoke: a small coarse M3 shock slab holds its position (shortfall, clipping, x_s)."""
import time
import jax.numpy as jnp
import numpy as np
import jax
from examples.learned_collision import flows, moments, shear_wave

ref = dict(flows.REFERENCES["shock_M3"])
state, cstate, grid, sp, params, static, key, c, bc = flows.shock_setup(3.0, 1.0, 20, 0, 4.0)
faces, cap, res, wall = flows.run_kwargs(bc)
states = jax.tree.map(flows.stack, state, state)
cstates = jax.tree.map(flows.stack, cstate, cstate)
keys = jnp.stack([key, jax.random.fold_in(key, 9)])
print("N", static.N, "A_max", static.A_max, "capacity", cap, "expected", c["n_part"], "dt", c["dt"])
xc = (np.arange(c["cells"]) + 0.5) * c["lambda"]
t0 = time.time()
for b in range(60):
    states, cstates, clip, short = flows.advance(states, cstates, keys, grid, sp, params, b * 20,
        None, res, wall, static=static, collide_fn=None, faces=faces, capacity=cap)
    if b % 10 == 9:
        cnt = np.stack([np.asarray(moments.raw_sums(states.x[i, :, 0], states.v[i], states.alive[i], 0.0, c["length"], c["cells"])[0]) for i in range(2)])
        xs = flows.shock_position(cnt, xc, c["ratio"], c["ppc_up"]) / c["lambda"]
        print(b, "alive", np.asarray(states.alive.sum(1)), "x_s", np.round(xs, 2), "clip", int(clip), "short", float(short), f"{time.time()-t0:.0f}s")
