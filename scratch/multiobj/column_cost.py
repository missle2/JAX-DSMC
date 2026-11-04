"""Cost of the virtual backward against its number of value columns (2026-11-03: separate
drag / lift / heat gradients from one sweep = 2 R columns, hit + timing per objective).
Collisional relaxation (several collisions per particle per step), the shipped per-cell fit,
staged nodes, 4 x 8; compile excluded (each R timed on its second call).

    PYTHONPATH=. python3 scratch/multiobj/column_cost.py
"""

import time

import jax
import jax.numpy as jnp

from examples.ensemble_relaxation import CONTROLS, relaxation_run, species
from jax_dsmc import ensemble_backward as host
from jax_dsmc import ensemble_device as device

jax.config.update("jax_enable_x64", True)

n, steps = 4000, 20
t = dict(CONTROLS)
state0, out, _diag, rec, grid = relaxation_run(
    t, jax.random.key(3), n, steps, 2048, 0.2, "certified", capture=True
)
base = (grid, jnp.ones(4), species(t), jax.random.key(0), 1024, 32, host.CELL_BASIS, 4096, 4, 8)
for r in (1, 2, 4, 6):
    rewards = jnp.stack([out.v[:, i % 3] * (1.0 + i) / n for i in range(r)], axis=1)
    if r == 1:
        rewards = rewards[:, 0]
    for attempt in range(2):
        t0 = time.time()
        dev = device.backward_virtual(rec, state0.alive, out, rewards, *base, "staged", None,
                                      collision_out=True, columns_out=r > 1)  # fmt: skip
        jax.block_until_ready(dev.lam)
        dt = time.time() - t0
    print(f"columns {r}: {dt:.2f} s", flush=True)
