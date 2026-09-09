"""Verification of the BKW analytic solution, independent of the DSMC solver.

Nothing here imports a solver module.  The point of these tests is that the
oracle is right *before* anything is compared against it: a transcription error
in ``bkw.py`` would otherwise silently invalidate the whole M1 acceptance gate
by moving the target rather than by failing.

The load-bearing test is :func:`test_bobylev_fourier_residual`, which checks
that the analytic ``f`` actually solves the spatially homogeneous Boltzmann
equation for Maxwell molecules — via Bobylev's Fourier-space form, evaluated by
quadrature, with the time derivative taken by AD.  It is sensitive to every
coefficient in the closed form *and* to the ``exp(-nu t/6)`` rate constant, and
it does not reuse any of the algebra it is checking.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import bkw
from jax_dsmc.constants import K_BOLTZ

jax.config.update("jax_enable_x64", True)

# A deliberately non-unit, non-round physical configuration: unit values would
# hide a dimensional error (e.g. vth vs sqrt(2)*vth) behind a factor of 1.
MASS = 6.63e-26  # kg, argon
TEMP = 273.0  # K
NRHO = 1.0e20  # 1/m^3
DIAM = 4.17e-10  # m, argon VHS reference diameter
TREF = 273.0  # K

SIGMA_VR = float(bkw.maxwell_sigma_vr(DIAM, TREF, MASS / 2.0))
NU = float(bkw.collision_rate(NRHO, SIGMA_VR))
VTH = math.sqrt(K_BOLTZ * TEMP / MASS)


# --- quadrature helpers (module level: spec 0.6 forbids nested defs) ---------
#
# `K` is always wrapped in `jnp.asarray` before it reaches `bkw.*`.  A Python float
# is a weakly-typed scalar that XLA bakes in as a *literal*, so every distinct `K`
# compiles its own kernel: the 13-point sweep below measured 1.53 s that way against
# 0.01 s with an array argument.  Same numbers, 150x the wall time.


@functools.cache
def _gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for ``int_a^b`` by ``n``-point Gauss-Legendre.

    Cached because the nodes depend only on ``(a, b, n)``, all of which are fixed
    constants here, while `leggauss` costs 0.53 s at ``n = 400`` and 2.57 s at
    ``n = 800`` -- it was recomputed on every call and accounted for essentially
    all of this file's runtime.  The returned arrays are set read-only so a
    caller cannot mutate the shared copy.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


def _radial_moment(K: float, power: int, n: int = 400, span: float = 16.0) -> float:
    """``int_0^inf |v|^power * 4 pi |v|^2 f(|v|) d|v|`` by quadrature.

    Integrates the analytic ``f`` numerically rather than using any closed-form
    moment from ``bkw.py``; the closed forms are what is under test.
    """
    v, w = _gauss_legendre(0.0, span * VTH, n)
    f = np.asarray(bkw.distribution_K(v, jnp.asarray(K), TEMP, MASS, NRHO))
    return float(np.sum(w * 4.0 * math.pi * v**2 * v**power * f))


def _cartesian_moments(K: float, n: int = 90, span: float = 12.0) -> tuple[float, float, float]:
    """``(int f, int v_x f, int |v|^2 f)`` on a 3D tensor-product GL grid.

    The 3D form is what establishes ``<v> == 0``; a radial integral cannot,
    because it has already assumed isotropy.
    """
    g, w = _gauss_legendre(-span * VTH, span * VTH, n)
    vx = g[:, None, None]
    vy = g[None, :, None]
    vz = g[None, None, :]
    wt = w[:, None, None] * w[None, :, None] * w[None, None, :]
    v2 = vx**2 + vy**2 + vz**2
    f = np.asarray(bkw.distribution_K(np.sqrt(v2), jnp.asarray(K), TEMP, MASS, NRHO))
    return float(np.sum(wt * f)), float(np.sum(wt * vx * f)), float(np.sum(wt * v2 * f))


def _phi_normalised(y: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """``phi(k)/nrho`` re-expressed in ``y = |k|^2 vth^2 / 2``."""
    kmag = jnp.sqrt(2.0 * y / (K_BOLTZ * TEMP / MASS))
    return bkw.fourier_transform_K(kmag, jnp.asarray(K), TEMP, MASS)


def _phi_of_t(t: jnp.ndarray, y: float, K0: float) -> jnp.ndarray:
    """``phi(y, t)/nrho`` as a function of time alone, for AD in ``t``."""
    return _phi_normalised(jnp.asarray(y), bkw.similarity_K(t, NU, K0))


def _bobylev_rhs(y: float, K: float, n_mu: int = 96) -> float:
    """Bobylev's Fourier-space Maxwell-molecule collision operator at ``k``.

    For a kernel ``b(cos theta)`` with ``int b dsigma = sigma_T v_r`` and an
    isotropic ``f``, the Fourier transform ``phi`` obeys

        d phi(k)/dt = nu * (1/2) int_{-1}^{1} [phi(k+) phi(k-) - phi(k)] dmu

    with ``|k+|^2 = |k|^2 (1+mu)/2`` and ``|k-|^2 = |k|^2 (1-mu)/2``, i.e.
    ``y+ = y(1+mu)/2`` and ``y- = y(1-mu)/2`` in the reduced variable.  The
    ``nu`` prefactor is ``nrho * sigma_T v_r``; ``phi`` here is normalised so
    ``phi(0) = 1``.

    Reference: A. V. Bobylev, "Fourier transform method in the theory of the
    Boltzmann equation for Maxwell molecules", Dokl. Akad. Nauk SSSR 225 (1975).
    """
    mu, w = _gauss_legendre(-1.0, 1.0, n_mu)
    yp = y * (1.0 + mu) / 2.0
    ym = y * (1.0 - mu) / 2.0
    phi_p = np.asarray(_phi_normalised(jnp.asarray(yp), jnp.asarray(K)))
    phi_m = np.asarray(_phi_normalised(jnp.asarray(ym), jnp.asarray(K)))
    phi_k = float(_phi_normalised(jnp.asarray(y), jnp.asarray(K)))
    return NU * 0.5 * float(np.sum(w * (phi_p * phi_m - phi_k)))


def _numerical_fourier_transform(kmag: float, K: float, n: int = 800) -> float:
    """``phi(k)/nrho`` by direct quadrature of the analytic ``f``.

    For isotropic ``f``, ``phi(k) = (4 pi / k) int_0^inf f(r) sin(k r) r dr``.
    """
    r, w = _gauss_legendre(0.0, 16.0 * VTH, n)
    f = np.asarray(bkw.distribution_K(r, jnp.asarray(K), TEMP, MASS, 1.0))
    return float(4.0 * math.pi / kmag * np.sum(w * f * np.sin(kmag * r) * r))


# --- 1. it is a normalised probability density -------------------------------


@pytest.mark.parametrize("K", [0.6, 0.7, 0.85, 0.99, 1.0])
def test_normalised_density(K):
    """``int f d^3v == nrho`` at every ``K``, by radial quadrature."""
    assert _radial_moment(K, 0) == pytest.approx(NRHO, rel=1e-12)


@pytest.mark.parametrize("K", [0.6, 0.75, 0.95])
def test_nonnegative(K):
    """``f >= 0`` everywhere for ``K >= K_MIN``; the floor is where it bites."""
    v = jnp.linspace(0.0, 12.0 * VTH, 4001)
    f = bkw.distribution_K(v, jnp.asarray(K), TEMP, MASS, NRHO)
    assert float(jnp.min(f)) >= -1e-30 * NRHO


def test_positivity_floor_is_tight():
    """Just below ``K_MIN`` the density goes negative — the floor is not slack."""
    f0 = bkw.distribution_K(jnp.asarray(0.0), bkw.K_MIN - 1e-3, TEMP, MASS, NRHO)
    assert float(f0) < 0.0
    f0_at_floor = bkw.distribution_K(jnp.asarray(0.0), bkw.K_MIN, TEMP, MASS, NRHO)
    assert float(f0_at_floor) == pytest.approx(0.0, abs=1e-12 * NRHO / VTH**3)


# --- 2. mass, zero mean velocity, correct temperature at all times ------------


@pytest.mark.parametrize("K", [0.6, 0.8, 1.0])
def test_collision_invariants(K):
    """3D quadrature: mass ``nrho``, ``<v> == 0``, ``T`` constant and correct."""
    mass_int, mx_int, e_int = _cartesian_moments(K)
    assert mass_int == pytest.approx(NRHO, rel=1e-10)
    assert abs(mx_int) < 1e-10 * NRHO * VTH
    temp = MASS * (e_int / mass_int) / (3.0 * K_BOLTZ)
    assert temp == pytest.approx(TEMP, rel=1e-10)


def test_temperature_constant_in_time():
    """``<|v|^2>`` is time-independent — the exact statement of energy conservation."""
    times = np.linspace(0.0, 40.0 / NU, 9)
    ks = np.asarray(bkw.similarity_K(times, NU))
    v2 = np.array([_radial_moment(float(k), 2) / NRHO for k in ks])
    assert np.allclose(v2, 3.0 * VTH**2, rtol=1e-10)


def test_moments_helper_agrees_with_quadrature():
    """The closed-form :func:`bkw.moments` matches numerical integration."""
    t = 3.0 / NU
    m = bkw.moments(t, NU, TEMP, MASS, NRHO)
    K = float(m.K)
    assert float(m.v2) == pytest.approx(_radial_moment(K, 2) / NRHO, rel=1e-10)
    assert float(m.v4) == pytest.approx(_radial_moment(K, 4) / NRHO, rel=1e-10)
    assert float(bkw.moment_v6_K(K, TEMP, MASS)) == pytest.approx(
        _radial_moment(K, 6) / NRHO, rel=1e-9
    )


# --- 3. relaxation to the Maxwellian ------------------------------------------


def test_relaxes_to_maxwellian():
    """As ``t -> inf`` the solution converges pointwise to the Maxwellian."""
    v = jnp.linspace(0.05 * VTH, 6.0 * VTH, 200)
    eq = bkw.maxwellian(v, TEMP, MASS, NRHO)
    prev = np.inf
    for t_nd in (2.0, 10.0, 30.0, 60.0, 100.0):
        f = bkw.distribution(v, t_nd / NU, NU, TEMP, MASS, NRHO)
        err = float(jnp.max(jnp.abs(f - eq) / eq))
        assert err < prev  # monotone in this range
        prev = err
    assert prev < 1e-6


def test_K_limit_is_the_maxwellian():
    """``K == 1`` reproduces the independently coded Maxwellian exactly."""
    v = jnp.linspace(0.0, 8.0 * VTH, 500)
    got = bkw.distribution_K(v, 1.0, TEMP, MASS, NRHO)
    want = bkw.maxwellian(v, TEMP, MASS, NRHO)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-14, atol=0.0)


# --- 4. the analytically known relaxation rate -------------------------------
# This is the check that catches a transcription error in the time constant.


def test_fourth_moment_excess_is_a_pure_exponential():
    """``1 - <|v|^4>/(15 vth^4)`` measured by quadrature decays at exactly nu/3.

    The fourth moment is integrated numerically from the analytic ``f`` at a
    sequence of times, so the fitted rate is a property of the *distribution*,
    not of :func:`bkw.moment_excess`.
    """
    t = np.linspace(0.0, 12.0, 13) / NU
    ks = np.asarray(bkw.similarity_K(t, NU))
    v4 = np.array([_radial_moment(float(k), 4) / NRHO for k in ks])
    excess = 1.0 - v4 / (15.0 * VTH**4)

    # matches the closed form
    assert np.allclose(excess, np.asarray(bkw.moment_excess(t, NU)), rtol=1e-9)

    # and its log is exactly linear with slope -nu/3
    slope, intercept = np.polyfit(t, np.log(excess), 1)
    assert slope == pytest.approx(-NU / 3.0, rel=1e-9)
    assert math.exp(intercept) == pytest.approx((1.0 - bkw.K_MIN) ** 2, rel=1e-9)
    assert float(bkw.moment_relaxation_rate(NU)) == pytest.approx(NU / 3.0, rel=1e-14)

    # guard against the classic confusion: the moment rate is NOT the K rate
    assert abs(slope + NU / 6.0) > 0.1 * NU

    # residual of the linear fit is at quadrature noise, i.e. it really is a
    # single exponential and not a sum of two
    resid = np.log(excess) - (slope * t + intercept)
    assert np.max(np.abs(resid)) < 1e-9


def test_similarity_variable_rate():
    """``1 - K`` decays at ``nu/6``, and ``time_of_K`` inverts it."""
    t = np.linspace(0.0, 20.0, 21) / NU
    K = np.asarray(bkw.similarity_K(t, NU))
    slope, _ = np.polyfit(t, np.log(1.0 - K), 1)
    assert slope == pytest.approx(-NU / 6.0, rel=1e-9)
    assert np.allclose(np.asarray(bkw.time_of_K(K, NU)), t, atol=1e-12 / NU)


def test_literature_normalisation_reproduced():
    """In the ``nu = 1``, ``vth = 1`` convention this is ``K = 1 - exp(-t/6)``.

    Cross-check of the published form: starting from ``K_MIN`` at ``t = 0`` is
    the same trajectory as ``1 - exp(-t/6)`` started at ``t0 = 6 ln(5/2)``.
    """
    t = np.linspace(0.0, 15.0, 31)
    shifted = 1.0 - np.exp(-(t + bkw.T0_ND) / 6.0)
    assert np.allclose(np.asarray(bkw.similarity_K(t, 1.0)), shifted, rtol=1e-13)
    assert bkw.T0_ND == pytest.approx(6.0 * math.log(2.5))
    assert 1.0 - math.exp(-bkw.T0_ND / 6.0) == pytest.approx(bkw.K_MIN)


# --- 5. it actually solves the Boltzmann equation ----------------------------


@pytest.mark.parametrize("K", [0.6, 0.72, 0.9, 0.999])
def test_fourier_transform_matches_direct_quadrature(K):
    """The closed-form ``phi`` is the numerical Fourier transform of ``f``.

    Without this, the Bobylev residual test below could pass on a ``phi`` that
    is self-consistent but is not the transform of the ``f`` we ship.
    """
    for kk in (0.3, 1.0, 2.5, 4.0):
        kmag = kk / VTH
        got = float(bkw.fourier_transform_K(jnp.asarray(kmag), jnp.asarray(K), TEMP, MASS))
        want = _numerical_fourier_transform(kmag, K)
        assert got == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_phi_at_zero_is_unity():
    """``phi(0) = int f/nrho = 1`` — the normalisation the residual test assumes."""
    for K in (0.6, 0.8, 1.0):
        assert float(
            bkw.fourier_transform_K(jnp.asarray(0.0), jnp.asarray(K), TEMP, MASS)
        ) == pytest.approx(1.0)


@pytest.mark.parametrize("t_nd", [0.0, 1.5, 4.0, 9.0])
@pytest.mark.parametrize("y", [0.05, 0.4, 1.3, 3.0])
def test_bobylev_fourier_residual(t_nd, y):
    """**The verification test.**  ``d phi/dt == Q_hat(phi, phi)`` for Maxwell molecules.

    Left side: automatic differentiation of the analytic solution in ``t``,
    which exercises the ``exp(-nu t / 6)`` clock.
    Right side: Gauss-Legendre quadrature of Bobylev's collision integral,
    which knows nothing about BKW.

    If the polynomial coefficients, the ``1/6``, the ``nrho`` normalisation or
    the ``vth`` scaling were wrong, these two would not agree.
    """
    t = t_nd / NU
    dphi_dt = float(jax.grad(_phi_of_t)(t, y, bkw.K_MIN))
    K = float(bkw.similarity_K(t, NU))
    rhs = _bobylev_rhs(y, K)
    scale = NU * max(abs(rhs), 1e-3)
    assert dphi_dt == pytest.approx(rhs, abs=1e-11 * scale / NU, rel=1e-10)


def test_bobylev_residual_rejects_a_wrong_rate():
    """The residual test has teeth: a 1/5 instead of a 1/6 breaks it.

    Guards the guard.  Without this, a residual test that happened to be
    trivially satisfied would look like verification.
    """
    y = 1.0
    t = 3.0 / NU
    K_wrong = 1.0 - (1.0 - bkw.K_MIN) * math.exp(-NU * t / 5.0)
    dphi_dt_wrong = (
        -(NU / 5.0)
        * (1.0 - K_wrong)
        * float(jax.grad(_phi_normalised, argnums=1)(jnp.asarray(y), jnp.asarray(K_wrong)))
    )
    rhs = _bobylev_rhs(y, K_wrong)
    assert abs(dphi_dt_wrong - rhs) > 0.05 * abs(rhs)


# --- 6. exact sampling --------------------------------------------------------


@pytest.mark.parametrize("K", [0.6, 0.8])
def test_sample_velocities_reproduces_the_moments(K):
    """Monte Carlo draws match the analytic moments within sampling error."""
    n = 400_000
    v = bkw.sample_velocities(jax.random.key(7), n, K, TEMP, MASS)
    assert v.shape == (n, 3)
    assert v.dtype == jnp.float64

    v2 = np.asarray(jnp.sum(v * v, axis=1))
    mean_v = np.asarray(jnp.mean(v, axis=0))

    # <v> = 0 to within the standard error of the mean
    se = math.sqrt(VTH**2 / n)
    assert np.all(np.abs(mean_v) < 5.0 * se)

    want_v2 = 3.0 * VTH**2
    assert v2.mean() == pytest.approx(want_v2, rel=6.0 * v2.std() / math.sqrt(n) / want_v2)

    want_v4 = float(bkw.moment_v4_K(K, TEMP, MASS))
    got_v4 = (v2**2).mean()
    se4 = (v2**2).std() / math.sqrt(n)
    assert abs(got_v4 - want_v4) < 5.0 * se4


def test_sample_velocities_is_isotropic():
    """Each Cartesian component carries the same variance ``vth^2``."""
    v = bkw.sample_velocities(jax.random.key(11), 200_000, 0.65, TEMP, MASS)
    var = np.asarray(jnp.var(v, axis=0))
    assert np.allclose(var, VTH**2, rtol=0.02)


# --- 7. the Maxwell-molecule rate itself --------------------------------------


def test_sigma_vr_matches_the_vss_prefactor_at_omega_one():
    """:func:`bkw.maxwell_sigma_vr` is spec 7.1's prefactor with ``omega = 1``.

    Recomputed here from the general formula rather than from the specialised
    one, so a wrong ``gamma(2.5 - omega)`` would show up.
    """
    mr = MASS / 2.0
    omega = 1.0
    prefactor = (
        math.pi * DIAM**2 * (2.0 * K_BOLTZ * TREF / mr) ** (omega - 0.5) / math.gamma(2.5 - omega)
    )
    assert SIGMA_VR == pytest.approx(prefactor, rel=1e-14)
    # and it really is velocity-independent: q = prefactor * (vr2)^(1-omega)
    for vr2 in (1.0, 1e4, 1e8):
        assert prefactor * vr2 ** (1.0 - omega) == pytest.approx(prefactor, rel=1e-14)


def test_collision_rate_is_linear_in_density():
    assert float(bkw.collision_rate(2.0 * NRHO, SIGMA_VR)) == pytest.approx(2.0 * NU, rel=1e-14)


def test_everything_is_f64():
    """Spec 0.3: no f32 anywhere in the oracle."""
    v = jnp.linspace(0.0, VTH, 8)
    assert bkw.distribution_K(v, 0.7, TEMP, MASS, NRHO).dtype == jnp.float64
    assert bkw.similarity_K(jnp.asarray(1.0), NU).dtype == jnp.float64
    assert bkw.moment_excess(jnp.asarray(1.0), NU).dtype == jnp.float64
