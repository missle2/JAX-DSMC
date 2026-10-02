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
import numpy as np
from jax import lax, random

from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.safe import safe_sqrt
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
    """Wall-normal speed of a diffusely re-emitted particle, ``f64[n] >= 0``.

    [DF] ``dsmcDiffuseWallPatch.C:106-111``::

        v_perp = sqrt(k T / m) * sqrt(-2 log(1 - u))

    which is SPARTA's ``vrm * sqrt(-log u)`` (``collide`` uses the most-probable
    speed ``vrm = sqrt(2kT/m)``) written in the per-component sigma this codebase
    uses everywhere else -- the two differ only by which end of the unit interval
    is singular.  ``log1p(-u)`` keeps the ``u -> 0`` end accurate, and
    :func:`safe_sqrt` keeps the gradient finite there, which is the end that
    matters: the flux weighting makes ``v_perp = 0`` the *likely* value.

    Reparameterised in ``temp_wall``, so ``dJ/dT_wall`` is pathwise -- the reason
    a thermal wall's *parameter* gradient is easy even though the *event* of
    hitting it is not (section 9.4).
    """
    u = jax.random.uniform(key, (n,), dtype=jnp.float64)
    return thermal_speed(temp_wall, mass) * safe_sqrt(-2.0 * jnp.log1p(-u))


def standard_diffuse_draws(key, n):
    """The parameter-free randomness of one diffuse re-emission: ``(z, r)``.

    ``z`` is ``f64[n, 3]`` standard normal and ``r`` is ``f64[n]`` standard Rayleigh
    (``sqrt(-2 log(1 - u))``).  :func:`diffuse_velocity_from_draws` turns them into a
    re-emitted velocity for any face, so a bounce pass that may hit one of ``2D``
    faces draws once rather than once per face -- a particle hits at most one face
    per pass, so reusing the draw across faces leaves every marginal law unchanged.
    """
    return diffuse_draws_from_uniforms(*standard_diffuse_uniforms(key, n))


NORMAL_LO = np.nextafter(np.float64(-1.0), np.float64(0.0))
"""Lower end of the uniform ``jax.random.normal`` transforms (``_normal_real``)."""


def standard_diffuse_uniforms(key, n):
    """The uniforms behind :func:`standard_diffuse_draws`: ``(u_z f64[n, 3], u_r f64[n])``.
    ``u_z`` is the uniform ``jax.random.normal`` draws internally for the same key and
    shape (on ``(nextafter(-1, 0), 1)``) -- the transform to normals
    (:func:`diffuse_draws_from_uniforms`) can then run on the few rows that use them.
    ``test_split_diffuse_draws_are_the_draws`` pins the identity to the installed JAX."""
    k_perp, k_tan = jax.random.split(key)
    u_z = jax.random.uniform(k_tan, (n, 3), dtype=jnp.float64, minval=NORMAL_LO, maxval=1.0)
    u_r = jax.random.uniform(k_perp, (n,), dtype=jnp.float64)
    return u_z, u_r


def diffuse_draws_from_uniforms(u_z, u_r):
    """``(z, r)`` of :func:`standard_diffuse_draws` from its uniforms, row for row:
    ``z = sqrt(2) erfinv(u_z)`` (``jax.random.normal``'s own transform) and the standard
    Rayleigh ``r = sqrt(-2 log(1 - u_r))``."""
    z = jax.lax.mul(np.array(np.sqrt(2), np.float64), jax.lax.erf_inv(u_z))
    return z, safe_sqrt(-2.0 * jnp.log1p(-u_r))


def diffuse_velocity_from_draws(z, r, temp_wall, mass, axis, inward, vwall):
    """Diffuse re-emission from standard draws; reparameterised in ``temp_wall``.

    ``temp_wall`` is a scalar or three per-component temperatures ([P3] Test 3);
    the normal component uses ``temp_wall[axis]``.
    """
    tangential = thermal_speed(temp_wall, mass) * z
    normal_temp = temp_wall if jnp.ndim(temp_wall) == 0 else temp_wall[axis]
    v_perp = inward * thermal_speed(normal_temp, mass) * r
    return tangential.at[:, axis].set(v_perp) + vwall


def sample_diffuse_velocity(key, n, temp_wall, mass, axis, inward, vwall):
    """Full post-wall velocity for an **axis-aligned** diffuse wall, ``f64[n, 3]``.

    ``inward`` is ``+1.0`` on a low face and ``-1.0`` on a high face: the normal
    component must point back into the domain.  ``vwall`` is added afterwards and
    is required to be **tangential** -- see :func:`assert_tangential_wall`.

    SPARTA draws the two tangential components in polar form
    (``vtangent * sin/cos(2 pi u)``, ``surf_collide_diffuse.cpp:233-236``), which is
    Box-Muller for a pair of ``N(0, kT/m)`` variates.  We draw them directly as
    Gaussians (decision 11): statistically identical, trig-free, and one fewer
    singular expression.  On an axis-aligned face the tangent directions are just
    the other two velocity axes, so SPARTA's ``tangent1``/``tangent2``
    cross-product construction -- and its ``lensq3(tangent1) == 0`` degeneracy --
    does not arise at all.
    """
    z, r = standard_diffuse_draws(key, n)
    return diffuse_velocity_from_draws(z, r, temp_wall, mass, axis, inward, vwall)


def sample_tau(key, n: int, dt, epsilon):
    """The regularised timestep ``tau ~ N(dt, epsilon^2)``, ``f64[n]``.  [P3] section 4.

    ``epsilon == 0`` returns ``dt`` **exactly** (``dt + 0.0 * z == dt`` in IEEE), so the
    whole M1-M5 validation suite is unaffected by the existence of this knob -- that
    exactness is the reason the regularisation can be shipped switched off rather than
    kept on a branch.

    [P3] requires ``epsilon << dt/3`` so that ``P(tau < 0)`` is negligible; a negative
    ``tau`` would advect backwards.  Rather than trust the caller, values are clamped at
    zero and :func:`negative_tau_fraction` reports how often that bit.

    The point of the randomisation is not physical.  It makes the *event* of hitting a
    wall probabilistic, so the discontinuous boundary map is replaced by a smooth
    dependence on ``log p_l`` -- an ill-defined pathwise derivative becomes a
    well-defined weak derivative in expectation.
    """
    z = jax.random.normal(key, (n,), dtype=jnp.float64)
    return jnp.maximum(dt + epsilon * z, 0.0)


def negative_tau_fraction(dt, epsilon):
    """``P(tau < 0)`` for the current settings -- the quantity [P3] asks to keep negligible."""
    dt = jnp.asarray(dt, dtype=jnp.float64)
    epsilon = jnp.asarray(epsilon, dtype=jnp.float64)
    safe = jnp.where(epsilon > 0.0, epsilon, 1.0)
    return jnp.where(epsilon > 0.0, 0.5 * jax.scipy.special.erfc(dt / (safe * jnp.sqrt(2.0))), 0.0)


def assert_tangential_wall(vwall, axis, tol=1e-12):
    """A wall may slide, not breathe.  Raises if ``vwall`` has a normal component.

    SPARTA supports a wall whose velocity has a component along its own normal, but
    only via a rejection loop (``surf_collide_diffuse.cpp:265-276``) whose trip count
    is data-dependent -- inadmissible here (section 12, no ``while_loop``).  For
    planar Couette the wall slides *in its own plane*, so the normal component is
    exactly zero and the loop would never execute anyway.  This asserts the
    restriction rather than silently producing the wrong flux.
    """
    vn = float(jnp.abs(jnp.asarray(vwall)[..., axis]).max())
    if vn > tol:
        raise ValueError(
            f"wall velocity has a normal component {vn:.3e} on axis {axis}; only "
            "tangential wall motion is supported (SPARTA's normal-motion sampler is "
            "a data-dependent rejection loop)"
        )


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


def _tangent_frame(normal: jnp.ndarray, v_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Orthonormal tangents ``(t1, t2)`` of a unit ``normal``, ``f64[n, 3]`` each.

    ``t1`` is the incident velocity's tangential part (SPARTA
    ``surf_collide_diffuse.cpp:237-246``); when that vanishes (normal incidence) the
    fallback is the perpendicular ``normal x e_k`` with ``k`` the smallest normal
    component -- never parallel to ``normal``, so never degenerate -- chosen with
    ``jnp.where`` rather than SPARTA's random vector, so the frame is deterministic.
    ``t2 = normal x t1``.
    """
    vn = jnp.sum(v_in * normal, axis=1, keepdims=True)
    t1 = v_in - vn * normal
    norm1 = jnp.sqrt(jnp.sum(t1 * t1, axis=1, keepdims=True))
    scale = jnp.sqrt(jnp.sum(v_in * v_in, axis=1, keepdims=True))
    degenerate = norm1 <= 1e-12 * scale
    k = jnp.argmin(jnp.abs(normal), axis=1)
    e_k = jax.nn.one_hot(k, 3, dtype=jnp.float64)
    alt = jnp.cross(normal, e_k)
    alt = alt / jnp.sqrt(jnp.sum(alt * alt, axis=1, keepdims=True))
    t1 = jnp.where(degenerate, alt, t1 / jnp.where(degenerate, 1.0, norm1))
    return t1, jnp.cross(normal, t1)


def _log_with_floored_derivative(vn, floor):
    """``log(vn)`` with the exact primal and the derivative ``1 / max(vn, floor)``."""
    den = jax.lax.stop_gradient(jnp.maximum(jnp.maximum(vn, floor), 1e-300))
    primal = jax.lax.stop_gradient(jnp.log(jnp.maximum(vn, 1e-300)))
    return primal + (vn - jax.lax.stop_gradient(vn)) / den


def log_diffuse_density(v_out, normal, temp, mass, vwall, floor=0.0):
    """Log density of the diffuse re-emission law of :func:`diffuse_velocity_general` at
    the realised ``v_out`` (``f64[n, 3]``): the flux-weighted Maxwellian in the frame of
    the unit ``normal`` (``f64[n, 3]``, into the flow), tangential drift ``vwall``,

        q(v | n) = v_n / s^2 exp(-v_n^2 / 2 s^2) * exp(-|v_t - u|^2 / 2 s^2) / (2 pi s^2),
        s^2 = k T / m,   v_n = v . n,   v_t = v - v_n n.

    The value is frame-independent (the tangent frame of the sampler only reparameterises
    the isotropic tangential Gaussian), so its gradients in ``temp``, ``vwall`` and
    ``normal`` are the kernel scores of the ensemble adjoint (review section 5.3): the
    wall parameters and -- through ``normal`` -- the shape enter a diffuse hit only here.
    ``temp`` is a scalar or ``f64[n]``.  ``-inf`` where ``v_n <= 0``.

    The normal score carries ``v_t / v_n`` from ``log v_n`` -- a Rayleigh ``v_n`` makes
    its variance diverge logarithmically at grazing emission.  ``floor`` (scalar or
    ``f64[n]``, ``0`` = exact) is the grazing regularisation of that term: the primal is
    unchanged and only the derivative of ``log v_n`` is evaluated at ``max(v_n, floor)``
    (the same cut-off as :func:`jax_dsmc.mesh.levelset.regularised_quotient` on the hit
    time; bias ``O(floor / s)``).
    """
    s2 = K_BOLTZ * temp / mass
    s2 = jnp.broadcast_to(jnp.asarray(s2), (v_out.shape[0],))
    vn = jnp.sum(v_out * normal, axis=1)
    vt = v_out - vn[:, None] * normal
    tangential = jnp.sum((vt - vwall) ** 2, axis=1)
    return jnp.where(
        vn > 0.0,
        _log_with_floored_derivative(vn, floor)
        - vn**2 / (2.0 * s2)
        - tangential / (2.0 * s2)
        - 2.0 * jnp.log(s2)
        - jnp.log(2.0 * jnp.pi),
        -jnp.inf,
    )


def diffuse_velocity_general(z, r, normal, temp, mass, vwall, v_in):
    """Diffuse re-emission from a surface of arbitrary unit ``normal``, ``f64[n, 3]``.

    The same law as :func:`diffuse_velocity_from_draws`, written in the ``(n, t1, t2)``
    frame of SPARTA ``surf_collide_diffuse.cpp:213-245``: the normal component is the
    flux-weighted ``sqrt(kT/m) * r`` (``r`` standard Rayleigh), the two tangential
    components are ``sqrt(kT/m) * z``; ``vwall`` is added afterwards and is required to
    be tangential (see :func:`assert_tangential_wall`).  ``z, r`` are the draws of
    :func:`standard_diffuse_draws` (``z[:, 2]`` is unused here); ``normal`` is
    ``f64[n, 3]`` pointing into the flow, zero-padded when ``D < 3``; ``v_in`` (the
    incident velocity) only fixes the tangent frame, see :func:`_tangent_frame`.
    ``temp`` is a scalar (a per-axis triple has no meaning off the axes).
    Reparameterised in ``temp`` and in ``normal``: ``d/dtemp`` and ``d/dnormal`` are
    pathwise.  The tangential Gaussian is isotropic, so the law does not depend on the
    frame and any smooth frame field is a valid reparameterisation; the derivative uses
    the frame **parallel-transported** from the primal one (``t1(n) = normalise(a - (a .
    n) n)`` with ``a`` the frozen primal ``t1``), whose derivative ``-(a . dn) n`` is
    bounded -- the incident-velocity frame's own derivative carries ``1 / |v_t,in|`` at
    normal incidence, a log-divergent variance for no change in the law.
    """
    t1, _t2 = _tangent_frame(normal, v_in)
    a = jax.lax.stop_gradient(t1)
    t1 = a - jnp.sum(a * normal, axis=1, keepdims=True) * normal
    t1 = t1 / jnp.sqrt(jnp.sum(t1 * t1, axis=1, keepdims=True))
    t2 = jnp.cross(normal, t1)
    sigma = thermal_speed(temp, mass)
    return sigma * (r[:, None] * normal + z[:, :1] * t1 + z[:, 1:2] * t2) + vwall
