"""Regression assertions for the section-19 gaps closed so far: U1 and U3.

Spec: PSEUDOCODE.md section 19 (unverified-claims register), section 2.3/2.4
(clamp bounds), section 7.2 (collision rate, majorant, stale-max rule),
section 8.1/8.2 (attempt count, ``A_max``).

The full numerical studies live in ``scratch/gate_U1_clamp_bound.py`` and
``scratch/gate_U3_majorant_margin.py``; this file is the fast, deterministic
distillate that must keep passing.  Two of these tests assert that a claim in
the spec is **false** -- they are written that way on purpose, so that a later
"fix" to the spec that is not also a fix to the numbers fails loudly:

* :func:`test_u1_section19_inequality_on_dq_dvr2_is_false`
* :func:`test_u3_sparta_seed_alone_permits_violations`
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_dsmc.constants import EPS_REL_VR2, K_BOLTZ  # noqa: E402
from jax_dsmc.safe import safe_pow  # noqa: E402

# --- argon VSS: SPARTA data/ar.vss + data/ar.species ------------------------
AR_DIAM = 4.11e-10
AR_OMEGA = 0.81
AR_TREF = 273.15
AR_MASS = 6.63e-26
AR_MR = AR_MASS / 2.0

#: The value an earlier U1 pass proposed raising ``EPS_REL_VR2`` to, on a
#: peak-ratio ``D = sup/rms`` argument.  **Measured and REJECTED** -- see spec
#: 17.4.  Kept here as a named constant only so the tests below can state, as
#: executable assertions, why it was rejected.
EPS_REL_VR2_REJECTED = 1e-6

OMEGAS = tuple(round(0.5 + 0.1 * i, 6) for i in range(11))


# --- shared helpers (deliberately duplicated from the probes: tests must not
#     depend on scratch/) ---------------------------------------------------


def cxs():
    return math.pi * AR_DIAM * AR_DIAM


def prefactor(omega):
    """SPARTA collide_vss.cpp:122-125."""
    return cxs() * (2.0 * K_BOLTZ * AR_TREF / AR_MR) ** (omega - 0.5) / math.gamma(2.5 - omega)


def v_ref2(temp):
    """``(2 k T / m_r)`` -- the squared reference thermal speed for a pair."""
    return 2.0 * K_BOLTZ * temp / AR_MR


def eps_abs(eps_rel=EPS_REL_VR2):
    """The absolute clamp handed to ``safe_pow`` (section 2.2)."""
    return eps_rel * v_ref2(AR_TREF)


def q_of_vr2(vr2, omega, eps, pref):
    return pref * safe_pow(vr2, 1.0 - omega, eps)


def q_of_vrvec(vrvec, omega, eps, pref):
    """The composition AD sees in ``vss.py``: ``vr2 = sum(vr*vr)`` then ``safe_pow``."""
    return q_of_vr2(jnp.sum(vrvec * vrvec), omega, eps, pref)


_grad_vr2 = jax.jit(jax.vmap(jax.grad(q_of_vr2), in_axes=(0, None, None, None)))
_grad_vec = jax.jit(jax.vmap(jax.grad(q_of_vrvec), in_axes=(0, None, None, None)))


def stated_bound(omega, eps, pref=1.0):
    """The section-2.4 formula ``2 |1-omega| eps^(0.5-omega)``."""
    return pref * 2.0 * abs(1.0 - omega) * eps ** (0.5 - omega)


def sweep_u(eps, n=200, u_hi=5.0):
    """Log grid of ``|vr|`` from just above the clamp up to ``u_hi``, in units of ``v_ref``."""
    u_c = math.sqrt(eps)
    shoulder = u_c * (1.0 + np.geomspace(1e-14, 1e-3, 12))
    return np.concatenate([shoulder, np.geomspace(u_c * (1.0 + 1e-12), u_hi, n)])


def achieved_maxima(omega, eps_rel):
    """(max |dq/dvr2|, max ||grad_vr q||) over the sweep, prefactor-free."""
    u = sweep_u(eps_rel)
    g2 = np.abs(np.asarray(_grad_vr2(jnp.asarray(u * u), omega, eps_rel, 1.0)))
    vec = jnp.stack([jnp.asarray(u), jnp.zeros(u.size), jnp.zeros(u.size)], axis=-1)
    gv = np.linalg.norm(np.asarray(_grad_vec(vec, omega, eps_rel, 1.0)), axis=-1)
    return float(g2.max()), float(gv.max())


def maxwell_pair_cdf(u_c):
    """P(|vr| < u_c * v_ref) for an equilibrium pair.

    The relative velocity of two like particles is Maxwellian with the reduced
    mass, so in ``u = |vr| / sqrt(2 k T / m_r)`` the density is
    ``4/sqrt(pi) u^2 exp(-u^2)`` -- free of both species and temperature.
    """
    return math.erf(u_c) - 2.0 / math.sqrt(math.pi) * u_c * math.exp(-u_c * u_c)


def truncated_fraction(eps_rel, t_over_tref=1.0):
    """Fraction of equilibrium pairs the clamp removes, at gas temperature ``T``."""
    return maxwell_pair_cdf(math.sqrt(eps_rel / t_over_tref))


def pop_rms_grad(omega):
    """rms of ``||grad_vr q||`` over an unclamped Maxwellian; finite iff ``omega < 1.25``."""
    a = (5.0 - 4.0 * omega) / 2.0
    if a <= 0.0:
        return math.nan
    return 2.0 * abs(1.0 - omega) * math.sqrt(2.0 / math.sqrt(math.pi) * math.gamma(a))


def dominance(omega, eps_rel):
    """Worst-case (at-the-clamp) gradient over the equilibrium rms."""
    return stated_bound(omega, eps_rel) / pop_rms_grad(omega)


# =========================== U1 =============================================


@pytest.mark.parametrize("omega", [w for w in OMEGAS if w != 1.0])
def test_u1_bound_is_exact_for_the_ad_reachable_gradient(omega):
    """``2 |1-omega| eps^(0.5-omega)`` is the attained sup of ``||grad_vr q||``.

    This is the quantity that actually flows in reverse mode: ``vr2`` is not an
    input, ``vr`` is, and ``||grad_vr q|| = |dq/dvr2| * 2 |vr|``.  The bound is
    reached exactly on the clamp shoulder, so it is tight, not vacuous.
    """
    _, achieved = achieved_maxima(omega, EPS_REL_VR2)
    bound = stated_bound(omega, EPS_REL_VR2)
    assert achieved <= bound * (1.0 + 1e-9)
    assert bound / achieved == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("omega", [0.5, 0.81, 1.2, 1.5])
def test_u1_section19_inequality_on_dq_dvr2_is_false(omega):
    """The gate text asserts the bound against ``dq/dvr2``.  That assertion fails.

    ``sup |dq/dvr2| = |1-omega| eps^(-omega)`` while the stated bound is
    ``2 |1-omega| eps^(0.5-omega)``; their ratio is ``2 sqrt(eps)``, which is
    below 1 for any clamp small enough to be worth having -- and, being
    dimensional, changes with the unit system.  The bound is right; the
    left-hand side in section 19 is the wrong quantity.
    """
    achieved, _ = achieved_maxima(omega, EPS_REL_VR2)
    bound = stated_bound(omega, EPS_REL_VR2)
    assert bound < achieved, "section-19 inequality unexpectedly holds"
    assert bound / achieved == pytest.approx(2.0 * math.sqrt(EPS_REL_VR2), rel=1e-6)


def test_u1_correct_dq_dvr2_bound():
    """The right bound for ``dq/dvr2`` is ``|1-omega| eps^(-omega)``, also attained."""
    for omega in OMEGAS:
        if omega == 1.0:
            continue
        achieved, _ = achieved_maxima(omega, EPS_REL_VR2)
        bound = abs(1.0 - omega) * EPS_REL_VR2 ** (-omega)
        assert achieved <= bound * (1.0 + 1e-9)
        assert bound / achieved == pytest.approx(1.0, rel=1e-6)


def test_u1_gradient_is_zero_below_the_clamp():
    """Below the clamp ``safe_pow`` is constant, so the tangent is exactly 0 -- not NaN."""
    u = jnp.asarray(np.geomspace(1e-12, math.sqrt(EPS_REL_VR2) * 0.99, 40))
    g = _grad_vr2(u * u, 1.2, EPS_REL_VR2, 1.0)
    assert np.all(np.asarray(g) == 0.0)
    vec = jnp.stack([u, jnp.zeros(u.size), jnp.zeros(u.size)], axis=-1)
    gv = np.asarray(_grad_vec(vec, 1.2, EPS_REL_VR2, 1.0))
    assert np.all(np.isfinite(gv)) and np.all(gv == 0.0)


def test_u1_omega_one_fallback_is_one():
    """``x**0 == 1`` for every ``x`` including 0 -- Oracle 1's configuration (section 13.1)."""
    assert float(safe_pow(jnp.asarray(0.0), jnp.asarray(0.0), eps_abs())) == 1.0
    assert float(q_of_vr2(jnp.asarray(0.0), 1.0, eps_abs(), prefactor(1.0))) == prefactor(1.0)


def test_u1_clamp_truncates_essentially_no_equilibrium_pairs():
    """The 'too loose' end is not binding: truncation costs ``~eps^1.5``.

    Even 100x looser than the current value, and even in a gas 100x colder than
    ``T_ref`` (which is what widens the clamp in local thermal units), fewer
    than one pair in a million is removed.
    """
    assert truncated_fraction(EPS_REL_VR2) < 1e-12
    assert truncated_fraction(EPS_REL_VR2_REJECTED) < 1e-9
    assert truncated_fraction(EPS_REL_VR2_REJECTED, t_over_tref=0.01) < 1e-6
    # leading-order form, used for the scaling argument in the report
    u_c = math.sqrt(EPS_REL_VR2_REJECTED)
    assert maxwell_pair_cdf(u_c) == pytest.approx(
        4.0 / (3.0 * math.sqrt(math.pi)) * u_c**3, rel=1e-5
    )


def test_u1_monte_carlo_never_reaches_the_clamp_in_equilibrium():
    """2e6 argon pairs at 300 K, and how much headroom each candidate eps leaves.

    This is one of the reasons ``1e-6`` was rejected: it puts the clamp within
    ~57x of the closest pair drawn from only 2e6 samples, so at the 1e9-1e10
    pair counts of a production run it would genuinely be reached.  ``1e-8``
    keeps it nearly four decades clear.
    """
    key = jax.random.PRNGKey(20240903)
    k1, k2 = jax.random.split(key)
    scale = math.sqrt(K_BOLTZ * 300.0 / AR_MASS)
    vr2 = jnp.sum(
        (
            scale * jax.random.normal(k1, (2_000_000, 3))
            - scale * jax.random.normal(k2, (2_000_000, 3))
        )
        ** 2,
        axis=-1,
    )
    vr2_min = float(jnp.min(vr2))
    assert vr2_min / eps_abs(EPS_REL_VR2) > 1e3  # adopted value: far clear
    assert vr2_min / eps_abs(EPS_REL_VR2_REJECTED) < 1e2  # rejected value: within reach


def test_u1_larger_eps_cuts_worst_case_dominance_but_it_does_not_decide():
    """``D = sup/rms`` does improve as ``eps^(0.5-omega)`` -- and is still not the criterion.

    The arithmetic below is correct and is kept.  What was wrong is the claim
    attached to it: that "a pair sitting on the clamp -- a duplicated parcel, a
    zero-T IC" realises ``D``.  It does not.  ``safe_pow`` falls back to a
    *constant* below the clamp, so every one of those configurations sits in the
    flat region and contributes exactly zero gradient
    (:func:`test_u1_degenerate_pair_injects_exactly_no_gradient`).  ``D`` is
    attained only in a thin shell at ``|vr| = sqrt(eps) v_ref``, of probability
    ``~eps^1.5`` -- and the variance contribution of that shell is unmeasurable
    (spec 17.4).  So ``D`` is a real bound on a hypothetical, not a reason to move
    the constant.
    """
    for omega in (0.81, 0.99, 1.2):
        d_now = dominance(omega, EPS_REL_VR2)
        d_new = dominance(omega, EPS_REL_VR2_REJECTED)
        assert d_new < d_now
        assert d_new / d_now == pytest.approx(100.0 ** (0.5 - omega), rel=1e-6)
    assert dominance(0.81, EPS_REL_VR2) == pytest.approx(272.9, rel=1e-3)
    assert dominance(0.81, EPS_REL_VR2_REJECTED) == pytest.approx(65.5, rel=1e-3)


def test_u1_second_moment_diverges_above_omega_1p25():
    """For ``omega >= 1.25`` the clamp is the only thing making the gradient variance finite."""
    assert math.isfinite(pop_rms_grad(1.2))
    assert math.isnan(pop_rms_grad(1.25))
    assert math.isnan(pop_rms_grad(1.5))


def test_u1_degenerate_pair_injects_exactly_no_gradient():
    """THE test that decided U1.

    ``safe_pow`` is a double-``where`` onto a *constant* fallback, not a
    ``maximum(x, eps)`` clamp, so below the clamp both value and gradient are
    identically zero.  Every configuration the rejected ``D`` argument named --
    a duplicated parcel, a zero-temperature IC, a cold freestream, and the
    inactive scratch-row slots of section 8.4 (where ``pi == pj`` so ``vr == 0``
    exactly) -- lands there and contributes *nothing*, at either candidate eps.
    """
    for omega in (0.75, AR_OMEGA, 1.0, 1.2):
        for eps_rel in (EPS_REL_VR2, EPS_REL_VR2_REJECTED):
            eps = eps_abs(eps_rel)
            g = jax.grad(q_of_vrvec)(jnp.zeros(3), omega, eps, prefactor(omega))
            assert jnp.all(jnp.isfinite(g))
            assert float(jnp.linalg.norm(g)) == 0.0
            # and the second derivative is finite too -- needed for grad-of-grad
            h = jax.hessian(q_of_vrvec)(jnp.zeros(3), omega, eps, prefactor(omega))
            assert jnp.all(jnp.isfinite(h))


def test_u1_raising_eps_widens_the_value_discontinuity():
    """The clamp is a step in ``q`` of size ``eps^(1-omega)``; a larger eps is a bigger step.

    A second measured cost of the rejected value, over the whole physical VHS
    range.  At ``omega = 1`` there is no step at all, which is why the M1 BKW
    gate is insensitive to this constant either way.
    """
    for omega in (0.60, 0.75, AR_OMEGA, 0.95):
        jumps = {}
        for eps_rel in (EPS_REL_VR2, EPS_REL_VR2_REJECTED):
            eps = eps_abs(eps_rel)
            hi = float(q_of_vr2(jnp.asarray(eps * (1 + 1e-7)), omega, eps, 1.0))
            lo = float(q_of_vr2(jnp.asarray(eps * (1 - 1e-7)), omega, eps, 1.0))
            jumps[eps_rel] = hi - lo
        assert jumps[EPS_REL_VR2_REJECTED] > jumps[EPS_REL_VR2]
        ratio = jumps[EPS_REL_VR2_REJECTED] / jumps[EPS_REL_VR2]
        assert ratio == pytest.approx(100.0 ** (1.0 - omega), rel=1e-4)


def test_u1_maxwell_molecules_are_insensitive_to_the_clamp():
    """``omega == 1`` gives ``dq/dv == 0`` identically, so U1 cannot bind at M1.

    ``q = prefactor * vr2^0 = prefactor`` for every ``vr``, clamped or not.  The
    M1 BKW acceptance gate runs at exactly ``omega = 1``, so this constant first
    becomes live at M3/M4 argon.
    """
    for eps_rel in (EPS_REL_VR2, EPS_REL_VR2_REJECTED):
        eps = eps_abs(eps_rel)
        for vr in (jnp.zeros(3), jnp.array([1e-9, 0.0, 0.0]), jnp.array([300.0, -50.0, 7.0])):
            assert float(q_of_vrvec(vr, 1.0, eps, 1.0)) == pytest.approx(1.0, rel=1e-15)
            g = jax.grad(q_of_vrvec)(vr, 1.0, eps, 1.0)
            assert float(jnp.linalg.norm(g)) == 0.0
    assert stated_bound(1.0, EPS_REL_VR2) == 0.0


def test_u1_verdict_the_constant_stays_at_1e_8():
    """Records the U1 decision so a future edit has to argue with a failing test."""
    assert EPS_REL_VR2 == 1e-8, "spec 17.4: the raise to 1e-6 was measured and rejected"


# =========================== U3 =============================================

N_U3 = 400
A_MAX_U3 = 1500
STEPS_U3 = 12
K_EV = 5.0


def sparta_vremax_init(temp_thermal):
    """SPARTA collide_vss.cpp:107-133 -- ``2 * cxs * max(vscale)``.

    ``vscale[i] = sqrt(2 k T_thermal / m_i)`` (mixture.cpp:229-237) uses the
    *species* mass; the relative-speed scale uses the *reduced* mass, so the
    seed is short by ``sqrt(2)`` before any extreme-value margin is considered.
    """
    return 2.0 * cxs() * math.sqrt(2.0 * K_BOLTZ * temp_thermal / AR_MASS)


def thermal_majorant(temp, omega, k_ev=K_EV, factor=1.0):
    """An a-priori majorant from the cell temperature (the proposed replacement).

    ``q`` is monotone in ``vr``, so its supremum on the reachable interval
    ``[sqrt(eps), k_ev * v_ref(T)]`` is at an endpoint: the top for ``omega < 1``,
    the clamp for ``omega > 1``.
    """
    pref = prefactor(omega)
    hi = pref * (k_ev * k_ev * v_ref2(temp)) ** (1.0 - omega)
    lo = pref * eps_abs() ** (1.0 - omega)
    return factor * max(hi, lo)


def temperature(v):
    return AR_MASS / (3.0 * K_BOLTZ) * jnp.mean(jnp.sum((v - jnp.mean(v, axis=0)) ** 2, axis=-1))


def max_pair_q(v, omega, pref):
    """Exact max of ``q`` over all ``N(N-1)/2`` pairs."""
    g = v @ v.T
    d = jnp.diag(g)
    vr2 = d[:, None] + d[None, :] - 2.0 * g
    vr2 = jnp.where(jnp.eye(v.shape[0], dtype=bool), jnp.nan, vr2)
    return jnp.maximum(
        q_of_vr2(jnp.nanmax(vr2), omega, eps_abs(), pref),
        q_of_vr2(jnp.nanmin(vr2), omega, eps_abs(), pref),
    )


def _slot(carry, xs, omega, pref, sigma):
    """One NTC attempt slot.  Module level + functools.partial (section 0.6)."""
    v, nviol, minmarg, qmax = carry
    i, j, u_acc, nhat, active = xs
    vr = v[i] - v[j]
    vr2 = jnp.sum(vr * vr)
    q = q_of_vr2(vr2, omega, eps_abs(), pref)
    p = jnp.clip(q / sigma, 0.0, 1.0)
    acc = active & (u_acc < p)
    vcm = 0.5 * (v[i] + v[j])
    half = 0.5 * jnp.sqrt(vr2) * nhat
    v = v.at[i].set(jnp.where(acc, vcm + half, v[i])).at[j].set(jnp.where(acc, vcm - half, v[j]))
    marg = jnp.where(active & (q > 0.0), sigma / jnp.where(q > 0.0, q, 1.0), jnp.inf)
    return (
        v,
        nviol + (active & (q > sigma)).astype(jnp.int64),
        jnp.minimum(minmarg, marg),
        jnp.maximum(qmax, jnp.where(active, q, 0.0)),
    ), None


def ntc_step(v, sigma, key, omega, pref, rate_const):
    """One step under the [DF] stale-max rule: ``sigma`` is frozen throughout (section 7.2)."""
    kp, ku, kn, kr = jax.random.split(key, 4)
    f = 0.5 * N_U3 * (N_U3 - 1.0) * sigma * rate_const
    frac = f - jnp.floor(f)
    n_att = (jnp.floor(f) + (jax.random.uniform(kr, ()) < frac)).astype(jnp.int64)
    ij = jax.random.randint(kp, (A_MAX_U3, 2), 0, N_U3)
    nrm = jax.random.normal(kn, (A_MAX_U3, 3))
    xs = (
        ij[:, 0],
        ij[:, 1],
        jax.random.uniform(ku, (A_MAX_U3,)),
        nrm / jnp.linalg.norm(nrm, axis=-1, keepdims=True),
        (jnp.arange(A_MAX_U3) < jnp.minimum(n_att, A_MAX_U3)) & (ij[:, 0] != ij[:, 1]),
    )
    body = functools.partial(_slot, omega=omega, pref=pref, sigma=sigma)
    (v, nviol, minmarg, qmax), _ = jax.lax.scan(body, (v, jnp.int64(0), jnp.inf, 0.0), xs)
    return v, nviol, minmarg, qmax, n_att


_ntc_step = jax.jit(ntc_step)


def run_u3(sigma0, omega, use_thermal, factor, beams, steps=STEPS_U3):
    """Short 0D relaxation; returns (total violations, min margin, min Sigma/max_q_all)."""
    pref = prefactor(omega)
    key = jax.random.PRNGKey(7)
    kic, krun = jax.random.split(key)
    if beams:
        v = math.sqrt(K_BOLTZ * 20.0 / AR_MASS) * jax.random.normal(kic, (N_U3, 3))
        v = v.at[:, 0].add(jnp.where(jnp.arange(N_U3) < N_U3 // 2, 700.0, -700.0))
    else:
        v = math.sqrt(K_BOLTZ * AR_TREF / AR_MASS) * jax.random.normal(kic, (N_U3, 3))
    rate_const = 400.0 / (0.5 * N_U3 * (N_U3 - 1.0) * sparta_vremax_init(AR_TREF))
    sigma = jnp.asarray(sigma0)
    tot, mm, tm = 0, math.inf, math.inf
    for k in range(steps):
        tm = min(tm, float(sigma / max_pair_q(v, omega, pref)))
        temp = float(temperature(v))
        v, nviol, minmarg, qmax, _n = _ntc_step(
            v, sigma, jax.random.fold_in(krun, k), omega, pref, rate_const
        )
        tot += int(nviol)
        mm = min(mm, float(minmarg))
        sigma = jnp.maximum(sigma, factor * qmax)
        if use_thermal:
            sigma = jnp.maximum(sigma, thermal_majorant(temp, omega, factor=factor))
    return tot, mm, tm


def test_u3_sparta_seed_is_below_max_q_at_cold_start():
    """SPARTA's ``vremax_init`` is *not* a majorant: ``Sigma / max q < 1`` from step 0.

    It is the right seed for SPARTA, which ratchets ``vremax`` live inside the
    candidate loop, so ``q > vremax`` is impossible by construction.  Under the
    [DF] stale-max rule we adopted (section 18.1) it is a violation.
    """
    key = jax.random.PRNGKey(11)
    v = math.sqrt(K_BOLTZ * AR_TREF / AR_MASS) * jax.random.normal(key, (2000, 3))
    ratio = sparta_vremax_init(AR_TREF) / float(max_pair_q(v, AR_OMEGA, prefactor(AR_OMEGA)))
    assert ratio < 1.0
    assert 0.7 < ratio < 0.95


@pytest.mark.parametrize("beams", [False, True])
def test_u3_sparta_seed_alone_permits_violations(beams):
    """The claim 'a 2-5x safety factor eliminates violations' fails on the seed.

    5x the SPARTA seed is still short of ``max q`` for a counter-streaming IC,
    because the seed is built from ``T_ref`` and knows nothing about the cell.
    """
    tot, _mm, tmin = run_u3(sparta_vremax_init(AR_TREF), AR_OMEGA, False, 1.0, beams)
    assert tot > 0
    assert tmin < 1.0


def test_u3_five_times_sparta_seed_works_but_is_not_the_rule_we_want():
    """``5x`` the seed does hold a beam IC -- the objection to it is cost, not violations.

    An earlier draft of this test asserted ``5x`` still violated.  It does not:
    measured zero violations here and in ``scratch/gate_U3_margin_numpy.py``
    (min margin 1.83 on beams).  The real objection is that a constant multiplier
    is **not adaptive** -- it pays a flat 5x in attempts even in a cold gas, where
    the a-priori thermal rule costs 1.0x and still holds a larger margin.  That is
    why 7.2.3 adopts the thermal majorant rather than simply inflating the seed.
    """
    tot, _mm, tmin = run_u3(5.0 * sparta_vremax_init(AR_TREF), AR_OMEGA, False, 1.0, beams=True)
    assert tot == 0
    assert tmin > 1.0
    # ... but it is a blunt instrument: 5x the seed regardless of the gas state.
    assert 5.0 * sparta_vremax_init(AR_TREF) > 3.0 * thermal_majorant(AR_TREF, AR_OMEGA, factor=1.0)


@pytest.mark.parametrize("beams", [False, True])
@pytest.mark.parametrize("omega", [0.7, AR_OMEGA, 1.0, 1.2])
def test_u3_thermal_rule_holds_the_margin(omega, beams):
    """The proposed rule: zero violations and ``min Sigma/q >= 2`` on both transients.

    ``Sigma = 2 * prefactor * (25 * v_ref(T_cell)^2)^(1-omega)``, floored by the
    stale ratchet.  ``T_cell`` is known at the *start* of the step, so unlike a
    ratchet on the observed max this does not lag a transient by one step.
    """
    sigma0 = thermal_majorant(AR_TREF, omega, factor=2.0)
    tot, mm, tmin = run_u3(sigma0, omega, True, 2.0, beams)
    assert tot == 0
    assert mm >= 2.0
    assert tmin >= 2.0


def test_u3_thermal_majorant_bounds_every_pair():
    """``K = 5`` covers the extreme-value tail: ``P(u > 5) * N^2/2 ~ 1.6e-4``."""
    key = jax.random.PRNGKey(3)
    for omega in (0.7, AR_OMEGA, 1.0, 1.2):
        v = math.sqrt(K_BOLTZ * AR_TREF / AR_MASS) * jax.random.normal(key, (4000, 3))
        assert thermal_majorant(AR_TREF, omega) >= float(max_pair_q(v, omega, prefactor(omega)))


def test_u3_margin_costs_attempts_one_for_one():
    """``n_attempt ~ Sigma`` (section 8.1), so a 2x margin is a 2x attempt load.

    This is the tension in section 8.2: ``A_max`` sized at cold start is
    overrun by exactly the factor the majorant is inflated by.
    """
    base = sparta_vremax_init(AR_TREF)
    f_cold = 0.5 * N_U3 * (N_U3 - 1.0) * base
    f_warm = 0.5 * N_U3 * (N_U3 - 1.0) * thermal_majorant(AR_TREF, AR_OMEGA, factor=2.0)
    assert f_warm / f_cold == pytest.approx(
        thermal_majorant(AR_TREF, AR_OMEGA, factor=2.0) / base, rel=1e-12
    )
    assert 2.0 < f_warm / f_cold < 6.0
