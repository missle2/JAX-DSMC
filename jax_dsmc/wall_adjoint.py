"""P3 random-time walls, with the conditional-time derivative made explicit.

A realized 1-D boundary history restricts the Gaussian travel time to an
interval. Reparameterizing the truncated Gaussian in that interval and scoring
the interval probability gives a consistent derivative of the regularized
transition, including position observables. Merely scoring the hit probability
while freezing an unconditional travel time omits this conditional derivative.
"""

import functools

import jax
import jax.numpy as jnp

from jax_dsmc.geometry import first_hit
from jax_dsmc.safe import safe_log
from jax_dsmc.surface import SurfaceTallies, zero


def conditional_time(raw, z, lower, upper, dt, epsilon):
    """Value-preserving conditional Gaussian sample and interval log probability."""
    eps = jnp.where(epsilon > 0.0, epsilon, 1.0)
    finite_lo, finite_hi = jnp.isfinite(lower), jnp.isfinite(upper)
    zl = (jnp.where(finite_lo, lower, dt) - dt) / eps
    zu = (jnp.where(finite_hi, upper, dt) - dt) / eps
    fl = jnp.where(finite_lo, jax.scipy.special.ndtr(zl), 0.0)
    fu = jnp.where(finite_hi, jax.scipy.special.ndtr(zu), 1.0)
    probability = jnp.maximum(fu - fl, 1e-30)
    within = jax.lax.stop_gradient(
        jnp.clip((jax.scipy.special.ndtr(z) - fl) / probability, 0.0, 1.0)
    )
    quantile = jnp.clip(fl + within * probability, 1e-15, 1.0 - 1e-15)
    sampled = dt + eps * jax.scipy.special.ndtri(quantile)
    # Preserve the original random realization exactly, but use its conditional tangent.
    corrected = jax.lax.stop_gradient(raw) + (sampled - jax.lax.stop_gradient(sampled))
    time = jnp.maximum(jnp.where(epsilon > 0.0, corrected, dt), 0.0)
    logp = jnp.where(epsilon > 0.0, safe_log(probability), 0.0)
    return time, logp


def next_hit(x, v, lo, hi):
    vd = v[:, 0]
    moving = vd != 0.0
    side = vd > 0.0
    bound = jnp.where(side, hi, lo)
    distance = (bound - x[:, 0]) / jnp.where(moving, vd, 1.0)
    return jnp.where(moving, jnp.maximum(distance, 0.0), jnp.inf), side


def _advance_impl(  # noqa: PLR0915 -- chronological event loop remains replayable
    x, v, alive, grid, faces, wall, mass, dt, key, epsilon=0.0, acc_delta=0.0, max_bounces=4, return_tallies=False
):
    """1-D first-event tracking and P3 wall gradient; returns the apply_faces tuple.

    Specular, diffuse, periodic and absorbing faces participate in the same
    chronological event search. ``recrossed`` reports exhausted event capacity.
    All parameters use physical units, including the wall temperature in kelvin.
    """
    from jax_dsmc.boundary import (  # noqa: PLC0415 -- break the boundary dispatch import cycle
        BC,
        _diffuse_velocity,
    )

    n = x.shape[0]
    dim = x.shape[1]
    z = jax.random.normal(jax.random.fold_in(key, 123), (n,), dtype=jnp.float64)
    raw = dt + epsilon * z
    budget = jnp.maximum(raw, 0.0)
    elapsed = jnp.zeros(n)
    logp = jnp.zeros(n)
    had_hit = jnp.zeros(n, bool)
    originally_alive = alive
    tally = zero(2*dim)

    def bounce(k, carry):
        x, v, alive, elapsed, logp, had_hit, tally = carry
        distance, surface = first_hit(x, v, grid)
        hit = alive & (elapsed + distance <= budget)
        used = jnp.where(hit, distance, 0.0)
        x_at = x + used[:, None] * v[:, :dim]
        for axis in range(dim):
            for s in (0, 1):
                selected = hit & (surface == 2*axis+s)
                rule = faces.of(axis, s)
                incident = v
                if rule is BC.DIFFUSE:
                    new_v, lp = _diffuse_velocity(
                        v, wall, mass, axis, s, jax.random.fold_in(key, 2*dim*k+2*axis+s), acc_delta
                    )
                    v = jnp.where(selected[:, None], new_v, v)
                    logp = logp + jnp.where(selected, lp, 0.0)
                elif rule is BC.SPECULAR:
                    v = v.at[:, axis].set(jnp.where(selected, -v[:, axis], v[:, axis]))
                elif rule is BC.PERIODIC:
                    x_at = x_at.at[:, axis].set(jnp.where(selected, grid.lo[axis] if s else grid.hi[axis], x_at[:, axis]))
                else:
                    alive = alive & ~selected
                f = 2*axis+s
                outgoing = jnp.where(alive[:,None], v, 0.)
                impulse = jnp.where(selected[:,None], mass*(incident-outgoing), 0.).sum(axis=0)
                energy = jnp.where(selected, .5*mass*(jnp.sum(incident**2,axis=1)-jnp.sum(outgoing**2,axis=1)), 0.).sum()
                velocity_wall = wall.vwall[f] if rule is BC.DIFFUSE else jnp.zeros(3)
                tally = SurfaceTallies(tally.hits.at[f].add(selected.sum(dtype=jnp.int32)),
                                       tally.momentum.at[f].add(impulse),
                                       tally.energy.at[f].add(energy),
                                       tally.heat.at[f].add(energy-jnp.dot(velocity_wall,impulse)))
        x = jnp.where(hit[:, None], x_at, x)
        elapsed = elapsed + used
        had_hit = had_hit | hit
        return x, v, alive, elapsed, logp, had_hit, tally

    x, v, alive, elapsed, logp, had_hit, tally = jax.lax.fori_loop(
        0, max_bounces, bounce, (x, v, alive, elapsed, logp, had_hit, tally)
    )
    distance, _ = first_hit(x, v, grid)
    exhausted = alive & (elapsed + distance <= budget)
    lower = jnp.where(had_hit, elapsed, -jnp.inf)
    upper = jnp.where(alive, elapsed + distance, jnp.inf)
    time, interval_score = conditional_time(raw, z, lower, upper, dt, epsilon)
    x = x + jnp.where(alive, time - elapsed, 0.0)[:, None] * v[:, :dim]
    logp = logp + interval_score
    result = (
        jnp.clip(x, grid.lo, grid.hi),
        v,
        alive,
        jnp.sum(exhausted),
        jnp.where(originally_alive, logp, 0.0),
    )

    return (*result, tally) if return_tallies else result


_advance_jitted = jax.jit(_advance_impl, static_argnames=("faces", "max_bounces", "return_tallies"))


@functools.wraps(_advance_impl)
def advance(
    x, v, alive, grid, faces, wall, mass, dt, key, epsilon=0.0, acc_delta=0.0, max_bounces=4, return_tallies=False
):
    # Eager calls need a stable compilation boundary; an enclosing jit/grad already
    # supplies tracing. Bypass the extra boundary there to keep its AD graph small.
    leaves = jax.tree.leaves((x, v, alive, grid, wall, mass, dt, key, epsilon, acc_delta))
    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
    kernel = _advance_impl if traced else _advance_jitted
    return kernel(
        x,
        v,
        alive,
        grid,
        faces,
        wall,
        mass,
        dt,
        key,
        epsilon=epsilon,
        acc_delta=acc_delta,
        max_bounces=max_bounces,
        return_tallies=return_tallies,
    )
