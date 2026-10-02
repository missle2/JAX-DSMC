"""Extensive particle-to-boundary transfer tallies in laboratory coordinates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SurfaceTallies(NamedTuple):
    hits: jax.Array
    momentum: jax.Array
    energy: jax.Array
    heat: jax.Array


def zero(surfaces):
    return SurfaceTallies(
        jnp.zeros(surfaces, jnp.int32),
        jnp.zeros((surfaces, 3)),
        jnp.zeros(surfaces),
        jnp.zeros(surfaces),
    )


def rates(tallies, particle_weight, measures, duration):
    """Force/area, total energy flux, heat flux; positive transfer is to surface."""
    factor = particle_weight / (measures * duration)
    return SurfaceTallies(
        tallies.hits,
        tallies.momentum * factor[:, None],
        tallies.energy * factor,
        tallies.heat * factor,
    )
