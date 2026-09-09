"""Tests for ``jax_dsmc.safe`` — spec section 2.4, plus the U1 gate of section 19.

The point of this file is stated in section 2.3: revision 2 tested only that
gradients are *finite*, which passed a derivative of ``5e14``.  Every test here
asserts a **stated numerical bound**:

===========================  ===============================================
function                     derivative bound
===========================  ===============================================
``safe_sqrt`` via norm       ``1.0``  (structural, independent of ``eps``)
``safe_sqrt`` bare           ``1 / (2 * sqrt(eps))``
``safe_pow`` (w.r.t. |vr|)   ``2 * |1 - omega| * eps**(0.5 - omega)``
``safe_log`` reject branch   ``1 / (Sigma - max q)``
===========================  ===============================================

No nested function definitions anywhere (section 0.6): every differentiated
integrand is a module-level function, bound with ``functools.partial``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma

from jax_dsmc.constants import EPS, EPS_REL_VR2, K_BOLTZ
from jax_dsmc.safe import (
    safe_div,
    safe_log,
    safe_norm,
    safe_normalize,
    safe_pow,
    safe_recip_count,
    safe_sqrt,
)

jax.config.update("jax_enable_x64", True)

# A real species, so the relative clamp has a physical scale: N2 at its
# reference temperature (SPARTA ``data.vss``: d = 4.17e-10 m, omega = 0.74).
MASS_N2 = 4.65e-26  # kg
MR_N2 = MASS_N2 / 2.0  # reduced mass of a like pair
TREF = 273.0  # K
VR2_SCALE = 2.0 * K_BOLTZ * TREF / MR_N2  # ~3.24e5 m^2/s^2
VTH = float(np.sqrt(VR2_SCALE))  # ~569 m/s

# Fourteen decades of |v|, as section 2.3 measured.
DECADES = np.logspace(-14.0, 2.0, 17)


# --- differentiated integrands (module level, section 0.6) ----------------


def _naive_sqrt(x):
    """Unguarded ``sqrt``; the thing ``safe_sqrt`` must match away from 0."""
    return jnp.sqrt(x)


def _naive_norm(v):
    """Unguarded Euclidean norm; ``NaN`` gradient at the origin."""
    return jnp.sqrt(jnp.sum(v * v))


def _naive_div(num, den):
    """Unguarded division."""
    return num / den


def _naive_pow_of_speed(vr, p):
    """Unguarded ``(vr**2)**p``, as a function of the speed ``|vr|``."""
    return (vr * vr) ** p


def _pow_of_speed(vr, p, eps):
    """``safe_pow(vr**2, p, eps)`` as a function of the speed ``|vr|``.

    This is the composition the collision rate actually uses (section 7.2), and
    the one section 2.4's bound is stated for: ``d/d|vr| (vr^2)^p =
    2 p (vr^2)^(p - 0.5)``, maximal at the clamp for ``p < 0.5``.
    """
    return safe_pow(vr * vr, p, eps)


def _q_of_speed(vr, prefactor, omega, eps):
    """The physical collision rate ``q = prefactor * (vr^2)^(1-omega)`` (section 7.2)."""
    return prefactor * safe_pow(vr * vr, 1.0 - omega, eps)


def _reject_score(q, sigma):
    """The NTC reject-branch score ``log(1 - q/Sigma)`` ([P2] eq. ``back_eta``)."""
    return safe_log(1.0 - q / sigma)


def _div_by(den, num):
    """``safe_div`` with the arguments ordered for differentiation in ``den``."""
    return safe_div(num, den)


def _normalize_component(v, i):
    """Component ``i`` of ``safe_normalize(v)``; scalar output for ``jax.grad``."""
    return safe_normalize(v)[i]


def _pow_bound(p, eps):
    """Section 2.4's bound on ``|d/d|vr| safe_pow(vr^2, p, eps)|``."""
    return 2.0 * abs(p) * eps ** (p - 0.5)


# --- safe_div -------------------------------------------------------------


def test_safe_div_primal_and_fallback():
    assert float(safe_div(2.0, 4.0)) == pytest.approx(0.5)
    assert float(safe_div(1.0, 0.0)) == 0.0
    assert float(safe_div(1.0, 0.0, fallback=7.0)) == 7.0
    assert float(safe_div(1.0, -0.0)) == 0.0


def test_safe_div_gradient_is_finite_at_zero_denominator():
    g = jax.grad(_div_by)(0.0, 1.0)
    assert np.isfinite(float(g))
    assert float(g) == 0.0  # the guard branch is constant in `den`
    assert np.isfinite(float(jax.grad(jax.grad(_div_by))(0.0, 1.0)))


def test_safe_div_matches_naive_away_from_the_singularity():
    for den in (1e-3, 1.0, 5.0):
        assert float(safe_div(3.0, den)) == float(_naive_div(3.0, den))
        got = float(jax.grad(_div_by)(den, 3.0))
        want = float(jax.grad(_naive_div, argnums=1)(3.0, den))
        assert got == want  # bit-identical: the guard changes nothing here


# --- safe_sqrt ------------------------------------------------------------


def test_safe_sqrt_primal_matches_the_analytic_limit():
    assert float(safe_sqrt(0.0)) == 0.0
    assert float(safe_sqrt(4.0)) == 2.0
    for x in DECADES:
        assert float(safe_sqrt(x * x)) == pytest.approx(float(x), rel=1e-15)


@pytest.mark.parametrize("eps", [1e-30, 1e-16, 1e-8])
def test_safe_sqrt_bare_derivative_bound(eps):
    """Bare ``safe_sqrt`` is bounded by ``1/(2 sqrt(eps))`` — large, but stated."""
    bound = 1.0 / (2.0 * np.sqrt(eps))
    xs = jnp.asarray(np.concatenate([[0.0, eps], eps * np.logspace(0.0, 12.0, 40)]))
    grads = jax.vmap(partial(jax.grad(safe_sqrt), eps=eps))(xs)
    grads = np.asarray(grads)
    assert np.all(np.isfinite(grads))
    assert np.max(np.abs(grads)) <= bound * (1.0 + 1e-12)
    # and the bound is attained just above the clamp, so it is not vacuous
    assert np.max(np.abs(grads)) >= 0.5 * bound


def test_safe_sqrt_gradient_is_zero_below_the_clamp():
    assert float(jax.grad(safe_sqrt)(0.0)) == 0.0
    assert float(jax.grad(safe_sqrt)(0.5 * EPS)) == 0.0
    assert np.isfinite(float(jax.grad(jax.grad(safe_sqrt))(0.0)))


def test_safe_sqrt_matches_naive_away_from_the_singularity():
    for x in (1e-6, 1.0, 1e6):
        assert float(safe_sqrt(x)) == float(_naive_sqrt(x))
        assert float(jax.grad(safe_sqrt)(x)) == float(jax.grad(_naive_sqrt)(x))


# --- safe_norm: the structural bound of 1.0 -------------------------------


def test_safe_norm_derivative_is_bounded_by_one_across_fourteen_decades():
    """Section 2.3: ``||d ||v|| / dv|| == 1`` exactly, independent of ``eps``."""
    for scale in DECADES:
        v = jnp.asarray([scale, -2.0 * scale, 0.5 * scale], dtype=jnp.float64)
        g = np.asarray(jax.grad(safe_norm)(v))
        assert np.all(np.isfinite(g))
        assert np.linalg.norm(g) <= 1.0 + 1e-12
        assert np.linalg.norm(g) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("eps", [1e-30, 1e-16, 1e-8])
def test_safe_norm_bound_is_independent_of_eps(eps):
    """`eps` is the wrong knob here — raising it does not change the bound."""
    v = jnp.asarray([1e-13, 0.0, 0.0], dtype=jnp.float64)
    g = np.asarray(jax.grad(partial(safe_norm, axis=-1, eps=eps))(v))
    assert np.linalg.norm(g) <= 1.0 + 1e-12


def test_safe_norm_at_the_origin_is_zero_with_a_zero_gradient():
    v = jnp.zeros(3, dtype=jnp.float64)
    assert float(safe_norm(v)) == 0.0
    g = np.asarray(jax.grad(safe_norm)(v))
    assert np.all(np.isfinite(g))
    assert np.all(g == 0.0)
    # the naive form is exactly the failure this guards against
    assert np.all(np.isnan(np.asarray(jax.grad(_naive_norm)(v))))


def test_safe_norm_second_derivative_is_finite_at_the_origin():
    v = jnp.zeros(3, dtype=jnp.float64)
    h = np.asarray(jax.hessian(safe_norm)(v))
    assert np.all(np.isfinite(h))


def test_safe_norm_matches_naive_away_from_the_origin():
    v = jnp.asarray([3.0, 4.0, 12.0], dtype=jnp.float64)
    assert float(safe_norm(v)) == float(_naive_norm(v))
    assert np.array_equal(np.asarray(jax.grad(safe_norm)(v)), np.asarray(jax.grad(_naive_norm)(v)))


# --- safe_pow -------------------------------------------------------------


def test_safe_pow_zero_exponent_falls_back_to_one_not_zero():
    """The Maxwell-molecule case: ``x**0 == 1`` for every ``x``, including 0.

    Returning a blanket ``0.0`` would make ``q = prefactor * vr2**0`` vanish for a
    coincident pair at ``omega == 1`` — precisely Oracle 1's configuration
    (sections 2.2, 13.1).
    """
    eps = EPS_REL_VR2 * VR2_SCALE
    assert float(safe_pow(0.0, 0.0, eps)) == 1.0
    assert float(safe_pow(0.5 * eps, 0.0, eps)) == 1.0
    assert float(safe_pow(1e3 * eps, 0.0, eps)) == pytest.approx(1.0, rel=1e-15)
    # non-zero exponent keeps the 0.0 fallback
    assert float(safe_pow(0.0, -0.2, eps)) == 0.0
    assert float(safe_pow(0.0, 0.25, eps)) == 0.0


def test_safe_pow_primal_matches_the_analytic_value_above_the_clamp():
    eps = EPS_REL_VR2 * VR2_SCALE
    for x in (2.0 * eps, 1.0, VR2_SCALE):
        for p in (-0.5, -0.2, 0.25, 0.5, 1.0):
            assert float(safe_pow(x, p, eps)) == pytest.approx(x**p, rel=1e-13)


def test_safe_pow_gradient_is_zero_below_the_clamp():
    eps = EPS_REL_VR2 * VR2_SCALE
    for p in (-0.5, 0.0, 0.25):
        g = jax.grad(_pow_of_speed)(0.5 * float(np.sqrt(eps)), p, eps)
        assert float(g) == 0.0
    assert np.isfinite(float(jax.grad(jax.grad(_pow_of_speed))(0.0, -0.2, eps)))


def test_safe_pow_matches_naive_away_from_the_clamp():
    eps = EPS_REL_VR2 * VR2_SCALE
    for vr in (VTH, 10.0 * VTH):
        for p in (-0.2, 0.25):
            got = float(jax.grad(_pow_of_speed)(vr, p, eps))
            want = float(jax.grad(_naive_pow_of_speed)(vr, p))
            assert got == pytest.approx(want, rel=1e-13)


# --- safe_log -------------------------------------------------------------


def test_safe_log_primal_and_floor():
    assert float(safe_log(1.0)) == 0.0
    assert float(safe_log(0.0)) == pytest.approx(float(np.log(EPS)))
    assert float(safe_log(0.5)) == pytest.approx(float(np.log(0.5)), rel=1e-15)


def test_safe_log_reject_branch_bound_is_the_majorant_gap():
    """``|d log(1 - q/Sigma) / dq| <= 1/(Sigma - max q)`` (section 2.3)."""
    sigma = 2.0
    max_q = 1.98  # a 1.01x margin: deliberately tight
    bound = 1.0 / (sigma - max_q)
    qs = jnp.asarray(np.linspace(0.0, max_q, 200))
    grads = np.asarray(jax.vmap(partial(jax.grad(_reject_score), sigma=sigma))(qs))
    assert np.all(np.isfinite(grads))
    assert np.max(np.abs(grads)) <= bound * (1.0 + 1e-12)
    assert np.max(np.abs(grads)) >= 0.5 * bound  # attained, so not vacuous


def test_safe_log_gradient_is_finite_at_zero():
    assert np.isfinite(float(jax.grad(safe_log)(0.0)))
    assert float(jax.grad(safe_log)(0.0)) == 0.0  # the maximum branch is flat
    assert np.isfinite(float(jax.grad(jax.grad(safe_log))(0.0)))


# --- safe_normalize -------------------------------------------------------


def test_safe_normalize_primal_and_origin():
    v = jnp.asarray([3.0, 4.0, 0.0], dtype=jnp.float64)
    assert np.allclose(np.asarray(safe_normalize(v)), [0.6, 0.8, 0.0], rtol=1e-15)
    z = jnp.zeros(3, dtype=jnp.float64)
    assert np.all(np.asarray(safe_normalize(z)) == 0.0)


def test_safe_normalize_gradient_is_finite_at_the_origin():
    z = jnp.zeros(3, dtype=jnp.float64)
    for i in range(3):
        g = np.asarray(jax.grad(partial(_normalize_component, i=i))(z))
        assert np.all(np.isfinite(g))
        assert np.all(g == 0.0)


def test_safe_normalize_gradient_is_bounded_by_one_over_the_norm():
    """``d(v/|v|)/dv`` has spectral norm ``1/|v|``; check it at a realistic scale."""
    v = jnp.asarray([VTH, 0.0, 0.0], dtype=jnp.float64)
    j = np.asarray(jax.jacobian(safe_normalize)(v))
    assert np.all(np.isfinite(j))
    assert np.linalg.norm(j, 2) <= 1.0 / VTH * (1.0 + 1e-12)


# --- safe_recip_count -----------------------------------------------------


def test_safe_recip_count_handles_empty_cells():
    c = jnp.asarray([0, 1, 2, 7], dtype=jnp.int32)
    got = np.asarray(safe_recip_count(c))
    assert np.array_equal(got, [0.0, 1.0, 0.5, 1.0 / 7.0])
    assert got.dtype == np.float64


# --- U1 (section 19): does EPS_REL_VR2 = 1e-8 bound d q / d vr2? ----------
#
# Gate: sweep ``omega in [0.5, 1.5]`` and ``|vr|`` down to the clamp; assert
# ``max |dq/d|vr||  <=  2 |1-omega| eps**(0.5-omega)``; and confirm the bound is
# TIGHT rather than vacuous.

OMEGA_SWEEP = np.linspace(0.5, 1.5, 11)


def _sweep_speeds(eps):
    """``|vr|`` from just above the clamp up to ten thermal speeds."""
    vr_clamp = np.sqrt(eps)
    just_above = vr_clamp * (1.0 + np.logspace(-12.0, 0.0, 40))
    decades = np.logspace(np.log10(vr_clamp), np.log10(10.0 * VTH), 400)
    return jnp.asarray(np.concatenate([just_above, decades]))


def _max_abs_dq_dvr(omega, eps):
    """``max |d/d|vr| safe_pow(vr^2, 1-omega, eps)|`` over the sweep."""
    grads = jax.vmap(partial(jax.grad(_pow_of_speed), p=1.0 - omega, eps=eps))(_sweep_speeds(eps))
    return float(np.max(np.abs(np.asarray(grads))))


@pytest.mark.parametrize("omega", OMEGA_SWEEP)
def test_u1_safe_pow_derivative_bound_holds_and_is_tight(omega):
    """U1: the section 2.4 bound holds, and is attained (ratio ~ 1)."""
    eps = EPS_REL_VR2 * VR2_SCALE
    p = 1.0 - omega
    bound = _pow_bound(p, eps)
    achieved = _max_abs_dq_dvr(omega, eps)
    assert np.isfinite(achieved)
    assert achieved <= bound * (1.0 + 1e-9), f"omega={omega}: {achieved:.6e} > {bound:.6e}"
    if abs(p) < 1e-15:  # omega == 1: the exponent is 0, the derivative is exactly 0
        assert achieved == 0.0
        assert bound == 0.0
        return
    ratio = bound / achieved
    # "not vacuous": a bound 1e10x larger than what is achieved would be useless.
    assert ratio <= 1.05, f"omega={omega}: bound is loose by {ratio:.3e}x"


def test_u1_bound_is_attained_exactly_at_the_clamp_boundary():
    """The maximum sits at ``vr2 = eps``, which is why the bound is tight."""
    eps = EPS_REL_VR2 * VR2_SCALE
    for omega in (0.75, 1.2, 1.5):
        p = 1.0 - omega
        at_clamp = abs(float(jax.grad(_pow_of_speed)(float(np.sqrt(eps)) * (1.0 + 1e-12), p, eps)))
        assert at_clamp == pytest.approx(_pow_bound(p, eps), rel=1e-9)


def test_u1_gradient_vanishes_below_the_clamp():
    eps = EPS_REL_VR2 * VR2_SCALE
    below = jnp.asarray(np.sqrt(eps) * np.linspace(0.0, 0.999, 20))
    for omega in (0.5, 1.2, 1.5):
        grads = jax.vmap(partial(jax.grad(_pow_of_speed), p=1.0 - omega, eps=eps))(below)
        assert np.all(np.asarray(grads) == 0.0)


def test_u1_absolute_eps_would_leave_the_derivative_unbounded():
    """Why the clamp must be RELATIVE: section 2.3's 2.5e16 at omega = 1.2.

    Regression evidence that ``EPS = 1e-30`` is the wrong clamp for ``safe_pow``,
    even though it is the right one for ``safe_norm``.
    """
    unclamped = abs(float(jax.grad(_pow_of_speed)(1e-12, 1.0 - 1.2, EPS)))
    assert unclamped > 1e16
    assert unclamped == pytest.approx(2.5e16, rel=0.05)
    clamped = _max_abs_dq_dvr(1.2, EPS_REL_VR2 * VR2_SCALE)
    assert clamped < 1e3
    assert unclamped / clamped > 1e13


def test_u1_physical_collision_rate_gradient_is_bounded():
    """The same bound, carried through the physical ``q`` with its prefactor."""
    eps = EPS_REL_VR2 * VR2_SCALE
    for omega in (0.75, 1.0, 1.2):
        prefactor = float(np.pi * 4.17e-10**2 * VR2_SCALE ** (omega - 0.5) / gamma(2.5 - omega))
        bound = prefactor * _pow_bound(1.0 - omega, eps)
        grads = jax.vmap(partial(jax.grad(_q_of_speed), prefactor=prefactor, omega=omega, eps=eps))(
            _sweep_speeds(eps)
        )
        assert np.max(np.abs(np.asarray(grads))) <= bound * (1.0 + 1e-9)


@pytest.mark.parametrize("omega", [0.75, 1.2, 1.5])
def test_u1_residual_dominance_over_a_thermal_pair(omega):
    """What the clamp still costs: ``EPS_REL_VR2**(0.5 - omega)``.

    The bound holds and is tight, but "bounded" is not "harmless" — section 2.3's
    concern is *dominance*, one pair's cotangent swamping the rest.  The residual
    dominance of an at-the-clamp pair over a thermal-speed pair is exactly
    ``EPS_REL_VR2**(0.5 - omega)``: ``2.7e2`` at ``omega = 0.75``, ``4.0e5`` at
    ``omega = 1.2``, ``1.0e8`` at ``omega = 1.5``, with ``EPS_REL_VR2 = 1e-8``.
    Measured against the *rms* over an equilibrium ensemble instead of a single
    thermal pair (``scratch/gate_U1_clamp_bound.py`` part 3) the same quantity is
    ``1.2e5`` at ``omega = 1.2`` and ``3.4e4`` at ``omega = 1.1`` — the latter is
    where section 2.3's "~3.4e4" reproduces, not at the ``omega = 1.2`` it is
    attributed to.  Asserted here so that changing ``EPS_REL_VR2`` moves a number
    a reader can see; see the U1 report for the recommendation to raise it.
    """
    eps = EPS_REL_VR2 * VR2_SCALE
    p = 1.0 - omega
    at_clamp = _pow_bound(p, eps)
    at_thermal = abs(float(jax.grad(_pow_of_speed)(VTH, p, eps)))
    dominance = at_clamp / at_thermal
    assert dominance == pytest.approx(EPS_REL_VR2 ** (0.5 - omega), rel=1e-6)
    assert dominance <= 1e9  # documents the residual; see the report on U1


def _pow_of_vr2(vr2, p, eps):
    """``safe_pow`` differentiated in ``vr2`` itself, not in ``|vr|``."""
    return safe_pow(vr2, p, eps)


def test_u1_bound_is_stated_for_d_dvr_not_d_dvr2():
    """Spec note: section 19's LHS is mislabelled; the bound is exact for ``d/d|vr|``.

    Section 19 writes the gate as ``max|dq/dvr2| <= 2|1-omega| eps**(0.5-omega)``.
    Read literally that is FALSE: ``d/dvr2 (vr2)**(1-omega) = |1-omega|
    eps**(-omega)`` at the clamp, which exceeds the stated right-hand side by
    ``1/(2 sqrt(eps))``.  The formula is the exact, attained supremum of
    ``d/d|vr|`` — the AD-reachable gradient, since ``vr2 = sum(vr*vr)`` — and
    section 2.3's own table (5.0e+01 at ``omega=0.75, |vr|=1e-4``) is that
    quantity, not ``d/dvr2``.
    """
    eps = EPS_REL_VR2 * VR2_SCALE
    omega = 1.2
    p = 1.0 - omega
    stated = _pow_bound(p, eps)
    wrt_vr2 = abs(float(jax.grad(_pow_of_vr2)(eps * (1.0 + 1e-12), p, eps)))
    wrt_vr = abs(float(jax.grad(_pow_of_speed)(float(np.sqrt(eps)) * (1.0 + 1e-12), p, eps)))
    assert wrt_vr == pytest.approx(stated, rel=1e-9)  # exact for d/d|vr|
    assert wrt_vr2 > stated  # false for d/dvr2, as written
    assert wrt_vr2 / stated == pytest.approx(1.0 / (2.0 * np.sqrt(eps)), rel=1e-9)
    # section 2.3's table is d/d|vr|, confirming the intended reading
    assert abs(float(jax.grad(_pow_of_speed)(1e-4, 1.0 - 0.75, EPS))) == pytest.approx(
        50.0, rel=1e-9
    )


def test_safe_pow_exponent_gradient_is_finite_below_the_clamp():
    """``omega`` is traced too: ``dq/domega`` must survive a clamped pair."""
    eps = EPS_REL_VR2 * VR2_SCALE
    g = jax.grad(safe_pow, argnums=1)(0.0, -0.2, eps)
    assert np.isfinite(float(g))
    assert float(g) == 0.0  # the fallback is locally constant in p
    above = jax.grad(safe_pow, argnums=1)(VR2_SCALE, -0.2, eps)
    assert float(above) == pytest.approx(VR2_SCALE**-0.2 * np.log(VR2_SCALE), rel=1e-12)
