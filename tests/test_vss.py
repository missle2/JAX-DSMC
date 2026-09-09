"""VSS collision model (spec section 7.6).

The conservation tests are the load-bearing ones: momentum and energy are
conserved *structurally* here (mass-weighted offsets about a shared ``vcm``, and
``|v_r'| == |v_r|`` by construction), so any deviation beyond round-off means the
CM decomposition has been broken, not that a tolerance needs loosening.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from jax_dsmc.collide.vss import (  # noqa: E402
    acceptance_prob,
    collision_rate,
    log_h,
    make_prefactor,
    scatter_key,
)
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.state import SpeciesParams  # noqa: E402

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def species(omega=0.81, alpha=1.4, mass_ratio=1.0):
    mi, mj = AR_MASS, AR_MASS * mass_ratio
    mass = jnp.array([mi, mj])
    mr = jnp.array(
        [[mi * mi / (mi + mi), mi * mj / (mi + mj)], [mi * mj / (mi + mj), mj * mj / (mj + mj)]]
    )
    diam = jnp.full((2, 2), AR_DIAM)
    om = jnp.full((2, 2), omega)
    tref = jnp.full((2, 2), AR_TREF)
    al = jnp.full((2, 2), alpha)
    return SpeciesParams(
        mass=mass,
        diam=diam,
        omega=om,
        tref=tref,
        alpha=al,
        mr=mr,
        prefactor=make_prefactor(diam, om, tref, mr),
    )


def random_pairs(n, key, scale=400.0):
    ki, kj = jax.random.split(key)
    return (scale * jax.random.normal(ki, (n, 3)), scale * jax.random.normal(kj, (n, 3)))


OMEGA_REF = 0.81
"""Bird's argon.  Only used where the quantity under test does not depend on omega."""


# --------------------------- conservation -----------------------------------


@pytest.mark.parametrize("alpha", [1.0, 1.4, 2.0])
@pytest.mark.parametrize("mass_ratio", [1.0, 10.0, 100.0])
def test_momentum_and_energy_conserved(alpha, mass_ratio):
    """Structural, so the tolerance is round-off (1e-14), not a physics tolerance.

    Not parametrised over ``omega``: :func:`scatter` never reads it -- ``omega``
    enters the *rate*, not the deflection -- so the four-value omega axis this test
    used to carry ran four bit-identical copies of every case.  ``alpha`` (the
    deflection law) and ``mass_ratio`` (the centre-of-mass weights) are the two
    axes the scattered velocities actually depend on.
    """
    sp = species(OMEGA_REF, alpha, mass_ratio)
    n = 20_000
    vi, vj = random_pairs(n, jax.random.PRNGKey(3))
    isp = jnp.zeros(n, jnp.int32)
    jsp = jnp.ones(n, jnp.int32)
    vi_n, vj_n = scatter_key(vi, vj, isp, jsp, sp, jax.random.PRNGKey(4))
    mi, mj = sp.mass[isp][:, None], sp.mass[jsp][:, None]

    p0, p1 = mi * vi + mj * vj, mi * vi_n + mj * vj_n
    assert float(jnp.max(jnp.abs(p1 - p0)) / jnp.max(jnp.abs(p0))) < 1e-14

    e0 = jnp.sum(0.5 * mi * vi**2 + 0.5 * mj * vj**2, axis=-1)
    e1 = jnp.sum(0.5 * mi * vi_n**2 + 0.5 * mj * vj_n**2, axis=-1)
    assert float(jnp.max(jnp.abs(e1 - e0) / e0)) < 1e-14


def test_relative_speed_is_preserved():
    """``|v_r'| == |v_r|``: no internal energy yet, so ``scale`` is identically 1."""
    sp = species()
    vi, vj = random_pairs(5000, jax.random.PRNGKey(5))
    z = jnp.zeros(5000, jnp.int32)
    vi_n, vj_n = scatter_key(vi, vj, z, z, sp, jax.random.PRNGKey(6))
    r0 = jnp.linalg.norm(vi - vj, axis=-1)
    r1 = jnp.linalg.norm(vi_n - vj_n, axis=-1)
    assert float(jnp.max(jnp.abs(r1 - r0) / r0)) < 1e-14


# --------------------------- the angular law --------------------------------


def cos_chi(vi, vj, vi_n, vj_n):
    vr, vr_n = vi - vj, vi_n - vj_n
    return jnp.sum(vr * vr_n, -1) / (jnp.linalg.norm(vr, axis=-1) * jnp.linalg.norm(vr_n, axis=-1))


@pytest.mark.parametrize("alpha", [1.0, 1.25, 1.5, 2.0])
def test_cos_chi_matches_the_vss_law(alpha):
    """Empirical ``cos X`` must match ``2 U^(1/alpha) - 1`` by KS test."""
    sp = species(alpha=alpha)
    n = 40_000
    vi, vj = random_pairs(n, jax.random.PRNGKey(7))
    z = jnp.zeros(n, jnp.int32)
    vi_n, vj_n = scatter_key(vi, vj, z, z, sp, jax.random.PRNGKey(8))
    c = np.asarray(cos_chi(vi, vj, vi_n, vj_n))
    # CDF of 2U^(1/a)-1 is ((c+1)/2)^a on [-1, 1]
    ks = stats.kstest(c, lambda x: np.clip((x + 1.0) / 2.0, 0.0, 1.0) ** alpha)
    assert ks.pvalue > 1e-3, f"alpha={alpha} KS p={ks.pvalue}"


def test_alpha_one_is_isotropic():
    """``alpha == 1`` must give ``mean(cos X) -> 0`` and ``var(cos X) -> 1/3``."""
    sp = species(alpha=1.0)
    n = 200_000
    vi, vj = random_pairs(n, jax.random.PRNGKey(9))
    z = jnp.zeros(n, jnp.int32)
    vi_n, vj_n = scatter_key(vi, vj, z, z, sp, jax.random.PRNGKey(10))
    c = np.asarray(cos_chi(vi, vj, vi_n, vj_n))
    assert abs(c.mean()) < 4.0 / math.sqrt(n)
    assert c.var() == pytest.approx(1.0 / 3.0, rel=0.02)


# --------------------------- degeneracies -----------------------------------

DEGENERATE = {
    "identical velocities": (jnp.array([100.0, 0.0, 0.0]), jnp.array([100.0, 0.0, 0.0])),
    "v_r exactly along x": (jnp.array([100.0, 5.0, 7.0]), jnp.array([-100.0, 5.0, 7.0])),
    "v_r exactly along y": (jnp.array([5.0, 100.0, 7.0]), jnp.array([5.0, -100.0, 7.0])),
    "v_r exactly along z": (jnp.array([5.0, 7.0, 100.0]), jnp.array([5.0, 7.0, -100.0])),
    "both zero": (jnp.zeros(3), jnp.zeros(3)),
}


@pytest.mark.parametrize("name", sorted(DEGENERATE))
def test_degenerate_inputs_give_finite_values_and_gradients(name):
    """Section 7.6: finite values *and* finite gradients at every degeneracy.

    The gradient half is the part a naive ``jnp.where`` guard fails -- both
    branches are evaluated, so the unselected one still poisons the cotangent.
    """
    vi, vj = DEGENERATE[name]
    sp = species()
    z = jnp.zeros((), jnp.int32)

    out = scatter_key(vi, vj, z, z, sp, jax.random.PRNGKey(11))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in out)

    g = jax.grad(
        lambda a, b: jnp.sum(scatter_key(a, b, z, z, sp, jax.random.PRNGKey(11))[0]), argnums=(0, 1)
    )(vi, vj)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), f"{name}: non-finite scatter grad"

    q = collision_rate(vi, vj, z, z, sp)
    assert bool(jnp.isfinite(q))
    gq = jax.grad(lambda a: collision_rate(a, vj, z, z, sp))(vi)
    assert bool(jnp.all(jnp.isfinite(gq))), f"{name}: non-finite rate grad"


def test_scatter_is_scale_invariant():
    """The frame threshold is RELATIVE, so a change of units must not change physics.

    With the absolute ``d > 1e-6`` of ``collide_vss.cpp:429`` this differs by order
    unity between SI and nondimensional units (section 7.4).
    """
    sp = species()
    z = jnp.zeros((), jnp.int32)
    theta = 1e-7
    for scale in (1.0, 340.0, 3.4e4):
        vi = jnp.array([scale, theta * scale, 0.0])
        vj = jnp.zeros(3)
        vi_n, _ = scatter_key(vi, vj, z, z, sp, jax.random.PRNGKey(12))
        got = vi_n / scale
        if scale == 1.0:
            ref = got
        assert float(jnp.max(jnp.abs(got - ref))) < 1e-12


# --------------------------- rate, acceptance, score ------------------------


def test_maxwell_molecules_have_a_constant_rate():
    """``omega == 1`` => ``q == prefactor`` for every ``vr``, so ``dq/dv == 0``."""
    sp = species(omega=1.0)
    z = jnp.zeros((), jnp.int32)
    for vr in (jnp.zeros(3), jnp.array([1e-9, 0, 0]), jnp.array([500.0, -30.0, 7.0])):
        q = collision_rate(vr, jnp.zeros(3), z, z, sp)
        assert float(q) == pytest.approx(float(sp.prefactor[0, 0]), rel=1e-15)
        assert float(jnp.linalg.norm(jax.grad(collision_rate)(vr, jnp.zeros(3), z, z, sp))) == 0.0


def test_rate_matches_the_analytic_power_law_away_from_the_clamp():
    sp = species(omega=0.81)
    z = jnp.zeros((), jnp.int32)
    for s in (1.0, 50.0, 500.0):
        vr = jnp.array([s, 0.0, 0.0])
        got = float(collision_rate(vr, jnp.zeros(3), z, z, sp))
        want = float(sp.prefactor[0, 0]) * (s * s) ** (1.0 - 0.81)
        assert got == pytest.approx(want, rel=1e-12)


def test_acceptance_probabilities_partition_unity():
    """Section 7.6: ``h_accept + h_reject == 1`` for every candidate."""
    sp = species()
    vi, vj = random_pairs(10_000, jax.random.PRNGKey(13))
    z = jnp.zeros(10_000, jnp.int32)
    q = collision_rate(vi, vj, z, z, sp)
    sigma = 2.0 * jnp.max(q)
    p = acceptance_prob(q, sigma)
    assert float(jnp.max(jnp.abs(p + (1.0 - p) - 1.0))) < 1e-15
    assert bool(jnp.all((p >= 0.0) & (p <= 1.0)))


def test_acceptance_stops_the_gradient_through_sigma():
    """``Sigma`` is an algorithmic device, not physics (section 7.2)."""
    q = jnp.float64(1.0)
    g = jax.grad(acceptance_prob, argnums=1)(q, jnp.float64(4.0))
    assert float(g) == 0.0


def test_log_h_both_branches_are_differentiable_in_v():
    """The ``j=3`` reject branch contributes too -- omitting it biases the gradient."""
    sp = species()
    z = jnp.zeros((), jnp.int32)
    vi, vj = jnp.array([300.0, 10.0, -5.0]), jnp.array([-100.0, 4.0, 2.0])
    q = collision_rate(vi, vj, z, z, sp)
    sigma = 3.0 * q
    for accepted in (True, False):
        g = jax.grad(lambda a: log_h(collision_rate(a, vj, z, z, sp), sigma, accepted))(vi)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0


def test_log_h_gradient_matches_finite_differences_in_vss_params():
    """``log h`` is differentiable in ``diam``, ``omega``, ``tref`` via ``prefactor``."""
    z = jnp.zeros((), jnp.int32)
    vi, vj = jnp.array([300.0, 10.0, -5.0]), jnp.array([-100.0, 4.0, 2.0])

    def f(diam, omega):
        sp = species()
        sp2 = SpeciesParams(
            mass=sp.mass,
            diam=diam,
            omega=omega,
            tref=sp.tref,
            alpha=sp.alpha,
            mr=sp.mr,
            prefactor=make_prefactor(diam, omega, sp.tref, sp.mr),
        )
        q = collision_rate(vi, vj, z, z, sp2)
        return log_h(q, 1e-15, True)

    d0 = jnp.full((2, 2), AR_DIAM)
    o0 = jnp.full((2, 2), 0.81)
    gd, go = jax.grad(f, argnums=(0, 1))(d0, o0)
    for arg, ana, h in ((0, gd[0, 0], AR_DIAM * 1e-6), (1, go[0, 0], 1e-6)):
        base = [d0, o0]
        up, dn = list(base), list(base)
        up[arg] = base[arg].at[0, 0].add(h)
        dn[arg] = base[arg].at[0, 0].add(-h)
        fd = (f(*up) - f(*dn)) / (2 * h)
        assert float(ana) == pytest.approx(float(fd), rel=1e-5)


def test_prefactor_matches_sparta_formula():
    """``collide_vss.cpp:122-125``."""
    sp = species(omega=0.81)
    cxs = math.pi * AR_DIAM**2
    want = (
        cxs * (2 * K_BOLTZ * AR_TREF / float(sp.mr[0, 0])) ** (0.81 - 0.5) / math.gamma(2.5 - 0.81)
    )
    assert float(sp.prefactor[0, 0]) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("omega", [0.5, 0.6, 0.75, 0.81, 0.9, 1.0])
def test_mean_collision_rate_matches_hard_sphere_at_tref(omega):
    """``<sigma_T v_r>`` at ``T = T_ref`` must equal the hard-sphere rate, every omega.

    This is the calibration that DEFINES the VHS reference diameter, and it is
    exact rather than approximate::

        <sigma_T v_r> = pi d^2 (2kT_ref/m_r)^(w-1/2) / Gamma(5/2-w)
                        * (2kT/m_r)^(1-w) Gamma(5/2-w) / Gamma(3/2)

    -- the ``Gamma(5/2-w)`` cancels, and at ``T = T_ref`` the powers combine to
    ``pi d^2 (2kT/m_r)^(1/2) / Gamma(3/2)``, which is the hard-sphere mean rate
    ``pi d^2 <v_r>`` with no residual ``omega`` dependence.

    **Why this test exists.**  The M1 BKW gate validates the collision rate only at
    ``omega == 1``, where :func:`collision_rate` collapses to ``prefactor`` and the
    ``(v_r^2)^(1-omega)`` factor -- and hence the whole ``Gamma(5/2-omega)``
    normalisation -- is never exercised.  Every shock and transport result uses
    ``omega = 0.81``.  A wrong Gamma argument, or a sign error in the exponent,
    would leave M1 passing and silently scale every transport coefficient.
    """
    m = 6.63e-26
    d = 4.11e-10
    tref = 273.15
    mr = m / 2.0
    pref = float(
        make_prefactor(
            jnp.array([[d]]), jnp.array([[omega]]), jnp.array([[tref]]), jnp.array([[mr]])
        )[0, 0]
    )
    # <(v_r^2)^(1-w)> for a Maxwellian relative velocity, in closed form
    kt_mr = 2.0 * K_BOLTZ * tref / mr
    moment = kt_mr ** (1.0 - omega) * math.exp(math.lgamma(2.5 - omega) - math.lgamma(1.5))
    rate = pref * moment
    vr_bar = math.sqrt(8.0 * K_BOLTZ * tref / (math.pi * mr))
    rate_hs = math.pi * d * d * vr_bar
    assert rate == pytest.approx(rate_hs, rel=1e-12)
