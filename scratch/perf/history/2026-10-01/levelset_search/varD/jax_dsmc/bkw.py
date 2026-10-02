"""Bobylev-Krook-Wu (BKW) exact solution — the Milestone 1 physics gate.

The spatially homogeneous Boltzmann equation

    df/dt = Q(f, f)

admits, for **Maxwell molecules with isotropic scattering**, a closed-form
isotropic similarity solution found independently by Bobylev (1975) and by
Krook & Wu (1977).  It is one of very few exact solutions of the nonlinear
Boltzmann equation, and it is the standard acceptance test for a DSMC
collision operator: Bird uses it for exactly this purpose, and it is the same
Maxwell-molecule regime the adjoint-DSMC papers ([P1]/[P2]) run their numerics
in, so the M6 adjoint oracle reuses this setup.

The solution
------------
With ``vth**2 = k*T/m`` (so that ``<|v|^2> = 3*vth**2``) and a similarity
variable ``K(t) in (3/5, 1]``,

                   nrho                 /  |v|^2  \\  / 5K - 3    1 - K   |v|^2 \\
    f(v, t) = ---------------- * exp| - ------- | * | ------- + ------- * ----- |
              (2*pi*K*vth^2)^(3/2)      \\ 2K vth^2/  \\   2K       2K^2    vth^2/

    K(t) = 1 - (1 - K0) * exp(-nu * t / 6)

``f`` is isotropic in ``v`` and depends on it only through ``|v|``.  ``nu`` is
the per-particle collision rate (below).  The nondimensional statement found in
the literature is the ``nrho = m = k = T = 1``, ``nu = 1`` case, usually written

    f(t, v) = 1/(2*(2*pi*K)^(3/2)) * ((5K-3)/K + (1-K)/K^2 * |v|^2)
              * exp(-|v|^2/(2K)),      K(t) = 1 - exp(-t/6),

which is the form used as the benchmark in the spectral-method literature
(e.g. Hu & Wang, *Burnett Spectral Method for the Spatially Homogeneous
Boltzmann Equation*, arXiv:1810.07804, section on the BKW solution; and the
same expression in arXiv:2605.28475 section 5.2.1).  That ``K(t)`` is the
``K0 = 3/5`` member of the family above, shifted so that ``K(0) = 0`` is
irrelevant: ``K(t) = 1 - exp(-t/6)`` first reaches the positivity floor
``K = 3/5`` at ``t0 = 6*ln(5/2) ~ 5.4931``, which is why the benchmark is
always started at ``t0``.  Here that offset is absorbed into ``K0`` instead.

Positivity
----------
``f >= 0`` requires the constant term ``(5K-3)/(2K) >= 0``, i.e.
``K >= 3/5 =`` :data:`K_MIN`.  At ``K = 3/5`` exactly, ``f(0) = 0``: the
distribution has a hole at the origin.  ``K -> 1`` is the Maxwellian.

Validity regime — read this before using it as a gate
-----------------------------------------------------
The solution holds **only** for

* **Maxwell molecules**: ``sigma_T(v_r) * v_r = const``.  In VHS/VSS terms this
  is ``omega = 1`` exactly (spec section 7.1: ``q = prefactor * (vr2)^(1-omega)``
  is constant iff ``omega == 1``).  Not ``omega = 0.5`` (hard spheres), and not
  any intermediate value — the eigenvalue spectrum of the linearised operator
  is ``|v_r|``-independent only at ``omega = 1``, which is the whole reason a
  closed form exists.
* **isotropic scattering** in the centre-of-mass frame.  In VSS terms
  ``alpha = 1`` (which reduces VSS to VHS).  The rate constant ``1/6`` comes
  from the angular average ``<(1 - cos^2 theta)/4>`` over the unit sphere
  (see :func:`_bkw_rate_derivation` in ``tests/test_bkw.py``); a VSS
  ``alpha != 1`` reweights that average and changes the constant.
* a **single species**, spatially homogeneous, no external forcing, and an
  initial condition that is itself a member of the family (``K = K0``).

Anything else and this module is not an oracle for it.

The time constant
-----------------
Everything is expressed through the per-particle collision rate

    nu = nrho * (sigma_T * v_r)                                 [1/s]

(the loss term of ``Q`` is ``-f(v) * nrho * int b dsigma = -f(v) * nu``).  In
the literature's nondimensionalisation ``nrho = 1`` and ``sigma_T*v_r = 1``, so
``nu = 1`` and ``K = 1 - (1-K0)*exp(-t/6)`` reads directly.

**The rate constant was not taken on trust.**  ``tests/test_bkw.py`` verifies it
executably by Fourier-transforming the analytic ``f`` numerically and checking
that it satisfies Bobylev's Fourier-space form of the Maxwell-molecule
collision operator to ~1e-12.  That check is independent of this module's
algebra and of any transcription: it would fail for a wrong exponent, a wrong
polynomial coefficient, or a wrong normalisation.

Moments
-------
Mass and temperature are **constants of the motion** (they must be — they are
collision invariants):

    <1>       = nrho
    <v>       = 0
    <|v|^2>   = 3 * vth^2          =>  T constant

The moment that relaxes, and the one a DSMC run is compared against, is the
fourth:

    <|v|^4> / vth^4 = 15 * (1 - (1 - K)^2)
    => 1 - <|v|^4>/(15 * vth^4) = (1 - K)^2 = (1 - K0)^2 * exp(-nu*t/3)

i.e. the normalised fourth-moment excess is a **pure exponential with rate
nu/3**, twice the rate of ``K`` itself.  :func:`moment_excess` returns it.
The sixth moment relaxes too but as a mixture of ``exp(-nu*t/3)`` and
``exp(-nu*t/2)``, so it is not the clean gate quantity.

References
----------
* A. V. Bobylev, "Exact solutions of the Boltzmann equation", Dokl. Akad. Nauk
  SSSR 225 (1975) 1296-1299.
* M. Krook, T. T. Wu, "Exact solutions of the Boltzmann equation",
  Phys. Fluids 20 (1977) 1589-1595.
* G. A. Bird, *Molecular Gas Dynamics and the Direct Simulation of Gas Flows*
  — BKW as the DSMC relaxation validation case.
* J. Hu, L. Wang, arXiv:1810.07804 — the benchmark form quoted above.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_dsmc.constants import K_BOLTZ

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout

K_MIN = 0.6
"""Positivity floor for the similarity variable, ``K >= 3/5``.

Below it the constant term ``(5K-3)/(2K)`` goes negative and ``f`` is not a
density.  Equivalent to the literature's starting time ``t0 = 6*ln(5/2)`` for
the ``K(t) = 1 - exp(-t/6)`` normalisation.
"""

T0_ND = 6.0 * math.log(2.5)
"""Nondimensional time at which ``1 - exp(-t/6)`` reaches :data:`K_MIN`.

~5.4931.  Provided so a run can be compared against a published figure that
uses the shifted convention.
"""

_GAMMA_3_2 = 0.5 * math.sqrt(math.pi)
"""``gamma(2.5 - omega)`` at ``omega = 1``; see spec section 7.1."""


class BKWMoments(NamedTuple):
    """Exact moments of the BKW solution at one time.

    All fields are ``f64`` scalars (or arrays, if ``t`` was an array).

    Attributes:
        nrho: number density; constant.
        u: bulk velocity magnitude; identically zero.
        temp: translational temperature; constant.
        v2: ``<|v|^2>``; equals ``3 k T / m``, constant.
        v4: ``<|v|^4>``; relaxes to ``15 (kT/m)^2``.
        excess: ``1 - <|v|^4>/(15 vth^4) == (1-K)^2``; the pure exponential.
        K: the similarity variable at this time.
    """

    nrho: jnp.ndarray
    u: jnp.ndarray
    temp: jnp.ndarray
    v2: jnp.ndarray
    v4: jnp.ndarray
    excess: jnp.ndarray
    K: jnp.ndarray


# --- the Maxwell-molecule rate ------------------------------------------------


def maxwell_sigma_vr(diam: float, tref: float, mr: float) -> jnp.ndarray:
    """``sigma_T * v_r`` for a Maxwell molecule, in m^3/s.  TRACED throughout.

    Specialisation of the VHS/VSS prefactor of spec section 7.1
    (``collide_vss.cpp:122-125``) to ``omega = 1``::

        prefactor = pi*d^2 * (2 k tref / mr)^(omega - 1/2) / gamma(5/2 - omega)
        q         = prefactor * (vr2)^(1 - omega)

    At ``omega = 1`` the ``vr2`` factor is ``vr2^0 == 1``, so ``q`` is the
    constant returned here and the gas is genuinely Maxwellian.  This is the
    single number that sets the BKW clock.

    Args:
        diam: VHS reference diameter ``d_ref`` [m].
        tref: reference temperature ``T_ref`` [K].
        mr: reduced mass ``m/2`` for a single species [kg].

    Returns:
        ``sigma_T * v_r`` [m^3/s], constant in ``v_r``.
    """
    cxs = math.pi * diam**2
    return cxs * jnp.sqrt(2.0 * K_BOLTZ * tref / mr) / _GAMMA_3_2


def collision_rate(nrho: float, sigma_vr: float) -> jnp.ndarray:
    """Per-particle collision rate ``nu = nrho * sigma_T * v_r`` [1/s].  TRACED.

    This is the *only* place time enters the BKW solution.  It is the loss-term
    coefficient of the Boltzmann collision operator: a given molecule collides
    ``nu`` times per second.  The number of collision *events* per unit volume
    per unit time is ``nrho * nu / 2``, which is what a DSMC ``ncollide``
    counter measures — do not confuse the two when calibrating a run.
    """
    return jnp.asarray(nrho, dtype=jnp.float64) * jnp.asarray(sigma_vr, dtype=jnp.float64)


def thermal_variance(temp: float, mass: float) -> jnp.ndarray:
    """``vth^2 = k*T/m`` [m^2/s^2], the per-component velocity variance.  TRACED.

    Note this is the *statistician's* thermal speed (``<v_x^2> = vth^2``), not
    the most-probable speed ``sqrt(2kT/m)``.  ``<|v|^2> = 3 vth^2``.
    """
    return K_BOLTZ * jnp.asarray(temp, dtype=jnp.float64) / jnp.asarray(mass, dtype=jnp.float64)


def similarity_K(t: jnp.ndarray, nu: float, K0: float = K_MIN) -> jnp.ndarray:
    """The BKW similarity variable ``K(t)``.  TRACED in ``t``, ``nu``, ``K0``.

    ``K(t) = 1 - (1 - K0) * exp(-nu*t/6)``, monotonically increasing to 1.

    Args:
        t: time [s], any shape.
        nu: per-particle collision rate from :func:`collision_rate`.
        K0: ``K`` at ``t = 0``; must lie in ``[K_MIN, 1]`` for ``f >= 0``.
            The default :data:`K_MIN` is the canonical most-nonequilibrium
            start, matching the literature's ``t0 = 6 ln(5/2)``.

    Returns:
        ``K``, same shape as ``t``.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    return 1.0 - (1.0 - K0) * jnp.exp(-nu * t / 6.0)


def time_of_K(K: jnp.ndarray, nu: float, K0: float = K_MIN) -> jnp.ndarray:
    """Inverse of :func:`similarity_K`: the time at which the state is ``K``."""
    K = jnp.asarray(K, dtype=jnp.float64)
    return -(6.0 / nu) * jnp.log((1.0 - K) / (1.0 - K0))


def moment_relaxation_rate(nu: float) -> jnp.ndarray:
    """Decay rate of the fourth-moment excess, ``nu/3`` [1/s].

    ``K`` itself relaxes at ``nu/6``; the excess is quadratic in ``1-K`` and so
    relaxes at ``nu/3``.  This is the number the M1 gate fits against.
    """
    return jnp.asarray(nu, dtype=jnp.float64) / 3.0


# --- the distribution ---------------------------------------------------------


def distribution_K(
    vmag: jnp.ndarray,
    K: jnp.ndarray,
    temp: float,
    mass: float,
    nrho: float = 1.0,
) -> jnp.ndarray:
    """``f(|v|)`` at similarity variable ``K``, in s^3/m^6 (times ``nrho``).

    Parametrised by ``K`` rather than by ``t`` so that the shape and the clock
    can be tested separately — a transcription error in one does not hide
    behind the other.

    Args:
        vmag: speed ``|v|`` [m/s], any shape.  ``f`` is isotropic, so this is
            the whole dependence on ``v``.
        K: similarity variable, broadcastable against ``vmag``.
        temp: temperature [K]; constant in time.
        mass: molecular mass [kg].
        nrho: number density [1/m^3]; scales ``f`` linearly.  Pass ``1.0`` for
            a probability density.

    Returns:
        ``f``, broadcast of ``vmag`` and ``K``.  ``int f d^3v = nrho``.
    """
    vmag = jnp.asarray(vmag, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    vth2 = thermal_variance(temp, mass)
    s2 = K * vth2  # variance of the Gaussian factor
    x = vmag * vmag / vth2  # |v|^2 / vth^2, dimensionless
    gauss = jnp.exp(-0.5 * x / K) / (2.0 * math.pi * s2) ** 1.5
    poly = (5.0 * K - 3.0) / (2.0 * K) + (1.0 - K) / (2.0 * K * K) * x
    return nrho * gauss * poly


def distribution(
    vmag: jnp.ndarray,
    t: jnp.ndarray,
    nu: float,
    temp: float,
    mass: float,
    nrho: float = 1.0,
    K0: float = K_MIN,
) -> jnp.ndarray:
    """``f(|v|, t)``, the BKW solution.  See the module docstring for the form.

    Thin composition of :func:`similarity_K` and :func:`distribution_K`.
    """
    return distribution_K(vmag, similarity_K(t, nu, K0), temp, mass, nrho)


def speed_pdf_K(
    vmag: jnp.ndarray,
    K: jnp.ndarray,
    temp: float,
    mass: float,
) -> jnp.ndarray:
    """Probability density of the *speed* ``|v|``: ``4*pi*|v|^2 * f / nrho``.

    Integrates to 1 over ``|v| in [0, inf)``.  This is what a histogram of DSMC
    particle speeds estimates, so it is the right thing to overlay in a figure.
    """
    vmag = jnp.asarray(vmag, dtype=jnp.float64)
    return 4.0 * math.pi * vmag * vmag * distribution_K(vmag, K, temp, mass, 1.0)


def maxwellian(vmag: jnp.ndarray, temp: float, mass: float, nrho: float = 1.0) -> jnp.ndarray:
    """The equilibrium Maxwellian ``f`` at the same ``(nrho, temp, mass)``.

    Identical to ``distribution_K(vmag, 1.0, ...)``; written out separately so
    the ``K -> 1`` limit can be tested against an independently coded formula
    rather than against the same expression with a substitution.
    """
    vmag = jnp.asarray(vmag, dtype=jnp.float64)
    vth2 = thermal_variance(temp, mass)
    return nrho * jnp.exp(-0.5 * vmag * vmag / vth2) / (2.0 * math.pi * vth2) ** 1.5


# --- exact moments ------------------------------------------------------------


def moment_excess(t: jnp.ndarray, nu: float, K0: float = K_MIN) -> jnp.ndarray:
    """The normalised fourth-moment excess ``1 - <|v|^4>/(15 vth^4) = (1-K)^2``.

    A **pure exponential**, ``(1-K0)^2 * exp(-nu*t/3)``.  This is the M1
    acceptance quantity: a DSMC run at ``omega = 1`` must reproduce both its
    amplitude and its rate.  Dimensionless and independent of ``temp``,
    ``mass`` and ``nrho``, which is exactly what makes it a clean gate.
    """
    K = similarity_K(t, nu, K0)
    d = 1.0 - K
    return d * d


def moment_v4_K(K: jnp.ndarray, temp: float, mass: float) -> jnp.ndarray:
    """``<|v|^4>`` [m^4/s^4] at similarity variable ``K``.

    ``= 15 * vth^4 * (1 - (1-K)^2)``, rising to the Maxwellian ``15 vth^4``.
    """
    K = jnp.asarray(K, dtype=jnp.float64)
    vth2 = thermal_variance(temp, mass)
    d = 1.0 - K
    return 15.0 * vth2 * vth2 * (1.0 - d * d)


def moment_v6_K(K: jnp.ndarray, temp: float, mass: float) -> jnp.ndarray:
    """``<|v|^6>`` [m^6/s^6] at ``K``: ``105 * vth^6 * (1 - 3(1-K)^2 + 2(1-K)^3)``.

    Provided for completeness and as a second, harder check on the polynomial
    coefficients.  It is **not** a single exponential — it mixes ``exp(-nu t/3)``
    and ``exp(-nu t/2)`` — so :func:`moment_excess` is the gate quantity, not
    this.
    """
    K = jnp.asarray(K, dtype=jnp.float64)
    vth2 = thermal_variance(temp, mass)
    d = 1.0 - K
    return 105.0 * vth2**3 * (1.0 - 3.0 * d * d + 2.0 * d**3)


def moments(
    t: jnp.ndarray,
    nu: float,
    temp: float,
    mass: float,
    nrho: float = 1.0,
    K0: float = K_MIN,
) -> BKWMoments:
    """All exact moments of the BKW solution at time ``t``.

    ``nrho``, ``u`` and ``temp`` are collision invariants and come back
    constant by construction; that is a statement about the physics, not a
    computation, and the test suite checks it against numerical quadrature of
    :func:`distribution` rather than against these expressions.
    """
    K = similarity_K(t, nu, K0)
    vth2 = thermal_variance(temp, mass)
    zeros = jnp.zeros_like(K)
    return BKWMoments(
        nrho=jnp.full_like(K, nrho),
        u=zeros,
        temp=jnp.full_like(K, temp),
        v2=jnp.full_like(K, 3.0) * vth2,
        v4=moment_v4_K(K, temp, mass),
        excess=(1.0 - K) ** 2,
        K=K,
    )


# --- exact sampling of the initial condition ----------------------------------


def sample_velocities(
    key: jax.Array,
    n: int,
    K: float,
    temp: float,
    mass: float,
) -> jnp.ndarray:
    """Draw ``n`` velocities exactly from the BKW distribution at ``K``.

    Exact, not rejection-based, so the DSMC initial condition carries no bias
    to confound the gate.  The construction: with ``s^2 = K vth^2``, write

        f/nrho = a * N(0, s^2 I)  +  (1-a) * [ isotropic, |v| = s * chi_5 ]

    where ``a = (5K-3)/(2K)`` and ``1-a = 3(1-K)/(2K)``.  The second component
    has radial density ``prop r^4 exp(-r^2/2s^2)``, which is exactly a chi
    distribution with 5 degrees of freedom scaled by ``s`` — because the
    ``|v|^2`` weight on a 3D Gaussian adds two to the radial power.  Both
    components are sampled in closed form.

    Args:
        key: PRNG key.  This is host-side setup, not a step-stage draw, so it
            is outside the section 1 ``fold_in`` contract.
        n: number of particles (static).
        K: similarity variable in ``[K_MIN, 1]``; must be a Python float, since
            it selects nothing at trace time but appears in the mixture weight.
        temp: temperature [K].
        mass: molecular mass [kg].

    Returns:
        ``f64[n, 3]`` velocities.  Spec section 0.2: velocities are always
        3-component regardless of the spatial dimension ``D``.
    """
    s = jnp.sqrt(K * thermal_variance(temp, mass))
    a = (5.0 * K - 3.0) / (2.0 * K)

    k_pick, k_gauss, k_chi, k_dir = jax.random.split(key, 4)
    pick_gauss = jax.random.uniform(k_pick, (n,), dtype=jnp.float64) < a

    v_gauss = s * jax.random.normal(k_gauss, (n, 3), dtype=jnp.float64)

    chi5 = jnp.sqrt(jnp.sum(jax.random.normal(k_chi, (n, 5), dtype=jnp.float64) ** 2, axis=1))
    dirs = jax.random.normal(k_dir, (n, 3), dtype=jnp.float64)
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    v_chi = (s * chi5)[:, None] * dirs

    return jnp.where(pick_gauss[:, None], v_gauss, v_chi)


# --- Fourier representation (used by the verification test) -------------------


def fourier_transform_K(
    kmag: jnp.ndarray,
    K: jnp.ndarray,
    temp: float,
    mass: float,
) -> jnp.ndarray:
    """``phi(k) = int f(v) exp(-i k.v) d^3v / nrho`` for the BKW solution.

    Real and isotropic.  In the dimensionless variable ``y = |k|^2 vth^2 / 2``,

        phi = (1 - (1-K) * y) * exp(-K * y).

    This is the object Bobylev's Fourier-space form of the Maxwell-molecule
    collision operator acts on, and ``tests/test_bkw.py`` uses it to verify the
    ``exp(-nu t/6)`` rate constant against the Boltzmann equation itself rather
    than against a remembered formula.  It is also checked there against a
    direct numerical transform of :func:`distribution_K`, so the two are not
    allowed to be consistently wrong together.
    """
    kmag = jnp.asarray(kmag, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    y = 0.5 * kmag * kmag * thermal_variance(temp, mass)
    return (1.0 - (1.0 - K) * y) * jnp.exp(-K * y)
