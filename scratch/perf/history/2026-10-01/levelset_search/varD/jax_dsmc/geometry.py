"""Cartesian geometry operations shared by sources and chronological tracking.

The particle/collision driver consumes these measures and event queries. A future
mesh adapter supplies equivalent operations with static connectivity and traced
coordinates; no spatial trees or collision subcells are introduced here.
"""

import jax.numpy as jnp


def surface_measure(grid, axis):
    extent = grid.hi - grid.lo
    return grid.transverse * jnp.prod(jnp.where(jnp.arange(len(grid.n)) == axis, 1.0, extent))


def sample_surface(grid, axis, side, uniform):
    x = grid.lo + uniform * (grid.hi - grid.lo)
    return x.at[:, axis].set(grid.lo[axis] if side == 0 else grid.hi[axis])


def first_hit(x, v, grid):
    """Return time and surface ID; simultaneous hits prefer lower axis ID."""
    vd = v[:, : x.shape[1]]
    side = vd > 0
    bound = jnp.where(side, grid.hi, grid.lo)
    time = jnp.where(vd != 0, jnp.maximum((bound - x) / jnp.where(vd != 0, vd, 1.0), 0.0), jnp.inf)
    axis = jnp.argmin(time, axis=1)
    return jnp.min(time, axis=1), 2 * axis + side[jnp.arange(len(x)), axis].astype(jnp.int32)


def averaged_planar_impulse(velocities, inward_normal, gap, duration, mass, area=1.0):
    """First-hit specular wall force, averaged over uniform initial wall distance.

    Independent collisionless shape reference: the opposite face is absorbing,
    and particles initially approach this wall. Integrating the distance first
    accounts for the moving-support term that fixed-hit pathwise AD omits.
    Output is force on the wall per initial particle per area. This is NOT a
    general collisional shape estimator or a repeated-reflection approximation.
    """
    speed = jnp.maximum(-jnp.sum(velocities * inward_normal, axis=-1), 0.0)
    hit_probability = jnp.minimum(duration * speed / gap, 1.0)
    magnitude = 2 * mass * speed * hit_probability / (duration * area)
    return -magnitude[..., None] * inward_normal
