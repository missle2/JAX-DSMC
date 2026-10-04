"""XLA:CPU fusion control: how the code keeps a computed array from being recomputed.

XLA fuses a cheap producer into each of its consumers.  When a small table (per cell,
per block) is read by an ``N``-sized consumer through a gather, the fused producer is
evaluated once per *consumer element* -- a per-cell logarithm becomes a per-particle one.
``optimization_barrier``, the intended remedy, is expanded away before fusion on XLA:CPU,
so the code uses :func:`materialise` instead.  (A ``lax.cond`` branch boundary does the
same; ``wall_adjoint._dense_pass`` uses one where skipping the work is also wanted.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def materialise(t: jax.Array) -> jax.Array:
    """``t`` itself, stored as its own array: an identity scatter onto zeros, exact bit for
    bit (``-0`` included).  A scatter is always the root of its fusion, so ``t``'s producer
    runs once, over ``t``'s own size, whatever reads it afterwards."""
    return jnp.zeros_like(t).at[jnp.arange(t.shape[0], dtype=jnp.int32)].set(t)
