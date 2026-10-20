"""One step's slope fits of the shape adjoint (ensemble_device.node_gradients at the hit
queries): own cell k32 vs the pooled 3x3 block k96 (+ half-space), circle_long-like sizes."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import ensemble_device
from jax_dsmc.state import GridSpec

N, NPC, Q = 100_000, 256, 512
rng = np.random.default_rng(0)
grid = GridSpec(jnp.zeros(2), jnp.asarray([60.0, 20.0]), (60, 20))
x = jnp.asarray(rng.uniform(0, 1, (N, 2)) * np.array([60.0, 20.0]))
v = jnp.asarray(rng.normal(0, 1, (N, 3)))
alive = jnp.ones(N, bool)
lam = jnp.asarray(rng.normal(size=N))
scale = jnp.asarray([0.5, 0.5, 1.0, 1.0, 1.0])
# hit queries clustered round a body (cells near the centre), the rest unused (-1)
near = np.where(np.linalg.norm(np.asarray(x) - [30.0, 10.0], axis=1) < 3.5)[0]
queries = jnp.asarray(np.concatenate([rng.choice(near, Q // 2, replace=False),
                                      -np.ones(Q // 2, int)]))  # fmt: skip
direction = jnp.asarray(np.tile([1.0, 0.0, 0.0], (N, 1)))


def fits(pool, k, half):
    return ensemble_device.node_gradients(x, v, alive, lam, grid, NPC, scale, k, "quadratic_full",
                                          4096, direction, pool, None, queries, half)  # fmt: skip


for pool, k, half in ((0, 32, None), (1, 96, None), (1, 96, 0.25)):
    fn = jax.jit(lambda pool=pool, k=k, half=half: fits(pool, k, half))
    jax.block_until_ready(fn())
    t0 = time.time()
    for _ in range(20):
        out = fn()
    jax.block_until_ready(out)
    print(f"pool {pool} k {k} halfspace {half}: {(time.time() - t0) / 20 * 1e3:7.2f} ms / step",
          flush=True)  # fmt: skip
