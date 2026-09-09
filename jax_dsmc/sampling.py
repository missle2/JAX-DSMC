"""Reparameterised distribution draws (spec section 5).

Every sampler here is a **deterministic differentiable function of a
parameter-independent random draw** — the reparameterisation trick, which is
what makes ``d/dtemp`` and ``d/dvstream`` pathwise-exact.  [P3] states the
requirement explicitly: initial particles are generated as
``(x_0i, v_0i) = Psi(xi_i; m)`` where the distribution of ``xi_i`` does not
depend on ``m``.

Keys are *arguments* here; they are never derived inside these functions.  The
RNG contract of section 1 (keys from ``(base_key, step, stage_id)`` only) is the
caller's responsibility — see ``jax_dsmc.initialize``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, random

from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def thermal_speed(temp: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Per-component Maxwellian sigma, ``sqrt(k T / m)``.  Both args TRACED.

    Reference: [DF] ``dsmcCloud.C:1043-1051``.  No ``safe_*`` guard is needed:
    physical ``temp`` and ``mass`` are strictly positive, and ``d/dtemp`` of the
    composite in :func:`sample_maxwellian` is the exact pathwise derivative.
    """
    return jnp.sqrt(K_BOLTZ * temp / mass)


def sample_maxwellian(
    key: jax.Array,
    n: int,
    temp: jnp.ndarray,
    mass: jnp.ndarray,
    vstream: jnp.ndarray,
) -> jnp.ndarray:
    """Shifted Maxwellian velocities, ``f64[n, 3]`` — Gaussian form (section 5.1).

    Args:
        key: PRNG key from ``key_for`` (STATIC stream, section 1).
        n: number of velocities.  STATIC (it is an array shape).
        temp: translational temperature, scalar.  TRACED.
        mass: species mass, scalar.  TRACED.
        vstream: bulk velocity ``f64[3]``.  TRACED.

    Velocities are **always 3-component** regardless of the spatial dimension
    ``D`` (section 0.2).

    We follow [DF] ``dsmcCloud.C:1043-1051`` (three independent standard normals)
    rather than SPARTA ``create_particles.cpp:761-780`` (Rayleigh magnitude plus
    uniform phase).  The two are statistically identical — SPARTA's
    ``vscale*sqrt(-log U)*cos(theta)`` also has variance ``kT/m`` — but the
    Gaussian form is trig-free, needs one draw instead of four, and removes
    ``sqrt(-log U)`` (and its singular gradient at ``U = 1``) from the
    initialisation path entirely.

    Gradient paths: ``d/dtemp`` flows through :func:`thermal_speed` into all
    three components; ``d/dvstream`` is the identity.  Both exact, both pathwise.
    """
    xi = random.normal(key, (n, 3), dtype=jnp.float64)
    return vstream + thermal_speed(temp, mass) * xi


def sample_positions(key: jax.Array, n: int, grid: GridSpec) -> jnp.ndarray:
    """Uniform positions in the domain box, ``f64[n, D]`` (section 5.2).

    Args:
        key: PRNG key from ``key_for``.
        n: number of positions.  STATIC.
        grid: ``GridSpec``; ``lo``/``hi`` TRACED, ``n`` STATIC.

    ``lo`` and ``hi`` enter linearly, so ``d/dlo`` and ``d/dhi`` are exact and
    the same uniform draw is reused across parameter values (common random
    numbers, section 1 rule 4).

    SPARTA's axisymmetric variant ``sqrt(lo_y^2 + U*(hi_y^2 - lo_y^2))``
    (``create_particles.cpp:698``) is out of scope.
    """
    D = len(grid.n)
    u = random.uniform(key, (n, D), dtype=jnp.float64)
    return grid.lo + u * (grid.hi - grid.lo)


def stochastic_round(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Round ``x`` up with probability ``frac(x)``, else down.  Returns ``i32``.

    Args:
        x: real-valued count(s).  TRACED (but see below).
        u: uniform draw(s) on ``[0, 1)``, broadcastable to ``x``.

    Reference: SPARTA ``create_particles.cpp:672-677`` and the default (neither
    ``mcflag`` nor ``remainflag``) attempt-count path of
    ``collide_vss.cpp:137-161``.

    Unbiased: ``E[round(x)] = floor(x) + P(u < frac) = x``.  It preserves
    fractional counts in expectation across the domain instead of truncating them
    away cell by cell.

    This is a **frozen decision** (section 0.5): the result is an integer and
    carries no tangent.  It is nevertheless load-bearing for the gradient —
    section 8.1 scores the decision with ``log P(n | f)``, and that score is the
    *only* path by which ``dt``, ``fnum`` and ``cell_volume`` reach the
    objective.  Callers that need the score must recompute
    ``frac = x - floor(x)`` themselves; this function deliberately returns only
    the decision.
    """
    lower = jnp.floor(x)
    up = u < (x - lower)
    return lax.stop_gradient(lower + up).astype(jnp.int32)


# --------------------------------------------------------------------------
# Flux-weighted draws — M4 (diffuse wall) and M5 (inflow).  Stubs only.
#
# These are genuinely a *different distribution*, not a different construction
# of the Maxwellian, so the Rayleigh closed form survives here and only here
# (section 5.1).  Both need `safe_sqrt` from `jax_dsmc.safe`, because the draw
# is singular at one end of the unit interval.
# --------------------------------------------------------------------------


def sample_flux_normal(key: jax.Array, n: int, temp_wall, mass) -> jnp.ndarray:
    """Wall-normal speed of a diffusely re-emitted particle.  **M4 stub.**

    The implementation is [DF] ``dsmcDiffuseWallPatch.C:106-111``::

        u = random.uniform(key, (n,))
        v_perp = thermal_speed(temp_wall, mass) * safe_sqrt(-2.0 * log(1.0 - u))

    with the tangential components staying Gaussian.  [DF] writes ``log(1 - R)``
    rather than ``log(R)``; both are correct and the choice only moves which end
    of the unit interval is singular.  ``safe_sqrt`` covers it either way.
    """
    raise NotImplementedError("flux-weighted wall re-emission is milestone 4")


def mol_inflow(s):
    """Bird eq. 4.22 inflow number flux for speed ratio ``s``.  **M5 stub.**

    ``fix_emit.cpp:166-174``::

        (exp(-s * s) + sqrt(pi) * s * (1 + erf(s))) / (2 * sqrt(pi))
    """
    raise NotImplementedError("inflow flux is milestone 5")


def sample_inflow_velocity(key: jax.Array, temp, mass, vstream, normal) -> jnp.ndarray:
    """Flux-weighted inflow velocity.  **M5 stub.**

    ``fix_emit_face.cpp:576-583``: rejection sampling with acceptance probability
    proportional to ``(beta + s) exp(-beta^2)``.

    NOTE for whoever implements it: this is a rejection *loop*.  It must be a
    bounded ``lax.scan`` with a static trip count and an acceptance mask, not a
    ``while_loop``, or the M8 inflow work cannot differentiate it (section 12).
    """
    raise NotImplementedError("inflow sampling is milestone 5")
