"""The per-step body-hit mask of a record (every wall-event layout), as shape.hit_ahead folds it."""

import jax
import jax.numpy as jnp

from jax_dsmc.wall_adjoint import RecordEvents, SplitEvents, particle_sum


def per_step_hits(records):
    dim = records.x.shape[-1]
    walls = records.walls
    if isinstance(walls, RecordEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)
        n = head.shape[1]
        body = (walls.face >= 2 * dim) & (walls.slot >= 0)
        target = jnp.where(body, walls.slot, n)
        steps = jnp.arange(head.shape[0])[:, None]
        return head.at[steps, target].set(True, mode="drop")
    if isinstance(walls, SplitEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)
        tail = jnp.any(walls.tail.face >= 2 * dim, axis=1).astype(jnp.float64)
        folded = jax.vmap(particle_sum)(
            jnp.zeros_like(head, jnp.float64)[:, None], tail[:, None], walls.slot
        )
        return head | (folded > 0)
    return jnp.any(walls.face >= 2 * dim, axis=1)
