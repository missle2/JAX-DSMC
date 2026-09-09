"""The [P2] section 6 mapping, and the analytic facts the replication rests on.

These are cheap, deterministic checks.  The expensive ensemble study lives in
``examples/p2_replication`` and its results are recorded in that directory's README.
"""

import math

import jax.numpy as jnp
import pytest
from jax import random

from examples import p2_replication as p2
from jax_dsmc import species_data as sd


def test_beta_maps_to_omega():
    """``q ~ |v_r|^beta`` against ``sigma_T v_r = pre (v_r^2)^(1-omega)``."""
    assert p2.omega_of(0) == pytest.approx(1.0)  # Maxwell molecules
    assert p2.omega_of(1) == pytest.approx(0.5)  # hard sphere
    assert p2.omega_of(2) == pytest.approx(0.0)


def test_collision_fraction_matches_the_paper():
    """[P2] main.tex:1440 -- ``N_c/N = dt Sigma_v / eps``: 1 %, 10 %, 100 %."""
    assert p2.expected_collision_fraction(0) == pytest.approx(0.01)
    assert p2.expected_collision_fraction(1) == pytest.approx(0.10)
    assert p2.expected_collision_fraction(2) == pytest.approx(1.00)


def test_ntc_attempt_count_reduces_to_the_papers_pair_count():
    """Our ``0.5 npc (npc-1) Sigma dt fnum / v_cell`` must equal [P2]'s ``N_c / 2``.

    This is the load-bearing identity of the whole replication: if it fails the two
    codes are running different collision rates and nothing downstream means anything.
    """
    for beta in (0, 1, 2):
        cs = p2.case(beta, 64_000, npc_target=1000)
        ours = p2.attempts_per_cell(cs) * cs.n_cells
        theirs = 0.5 * cs.n_part * p2.expected_collision_fraction(beta)
        # (npc-1) vs npc is the only difference, so agreement is to 1/npc.
        assert ours == pytest.approx(theirs, rel=2.0 / cs.npc)


@pytest.mark.parametrize("beta", [1, 2])
def test_majorant_is_frozen_at_the_papers_value(beta):
    """``Sigma`` must not drift: [P2]'s derivation assumes it fixed."""
    sigma, thermal, ok = p2.check_majorant_is_frozen(p2.case(beta, 8000))
    assert ok, f"thermal majorant {thermal:.3e} exceeds [P2]'s Sigma {sigma:.3e}"


def test_beta_zero_majorant_deviation_is_known_and_bounded():
    """The one known deviation, pinned so it cannot silently grow.

    At ``omega = 1`` the a-priori thermal majorant's ``F_UP = 2`` multiplies a
    temperature factor of exactly 1, so we run at 2x [P2]'s ``Sigma``.  Documented in
    ``majorant_used``; asserted here so a change to ``F_UP`` shows up as a test failure
    rather than as a silently different replication.
    """
    cs = p2.case(0, 8000)
    sigma, thermal, ok = p2.check_majorant_is_frozen(cs)
    assert not ok
    assert thermal / sigma == pytest.approx(2.0, rel=1e-9)


def test_legendre_moment_is_one_third_at_alpha_one_and_two():
    """``<cos^2 chi> = 1/3`` at both ends -- isotropic at ``alpha = 1``, and again at 2."""
    assert float(p2.p2_legendre_moment(1.0)) == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert float(p2.p2_legendre_moment(2.0)) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_angular_moment_is_stationary_at_exactly_sqrt_two():
    """**Why ``dJ/d(alpha)`` measures zero on the argon shock case.**

    ``d<cos^2 chi>/d(alpha) = 8/(a+2)^2 - 4/(a+1)^2`` vanishes at ``a = sqrt(2)``, and
    argon's tabulated VSS ``alpha = 1.4`` is 1.0 % away from it.  A temperature-like
    objective is therefore at a **stationary point in alpha** there -- the near-zero
    ensemble gradient is the right answer, not a defect.  Measured confirmation (sign
    change between 1.4 and 1.5, ``r = 0.992`` against this curve) is in the example's
    README.
    """
    assert float(p2.d_legendre_d_alpha(p2.ALPHA_STATIONARY)) == pytest.approx(0.0, abs=1e-15)
    assert p2.ALPHA_STATIONARY == pytest.approx(math.sqrt(2.0))
    # sign change across it, and argon sits inside 1 %
    assert float(p2.d_legendre_d_alpha(1.0)) < 0.0
    assert float(p2.d_legendre_d_alpha(2.0)) > 0.0
    # 1.005 % -- close enough that a temperature objective is effectively stationary
    assert abs(1.4 - p2.ALPHA_STATIONARY) / p2.ALPHA_STATIONARY < 0.011


def test_energy_is_conserved_and_relaxation_is_partial():
    """Forward physics: the mean temperature is a collision invariant, for every beta."""
    means = []
    for beta in (0, 1, 2):
        cs = p2.case(beta, 2000, npc_target=500)
        m = p2.moments_of(p2.T0_REF, cs, random.PRNGKey(0))
        means.append(float(jnp.mean(m)))
        # relaxing towards isotropy, but [P2] chooses T = 2 so it stays partial
        assert float(m[2]) > p2.T0_REF[2]
        assert float(m[2]) < p2.T_MAXWELL
    assert means[0] == pytest.approx(means[1], rel=1e-12)
    assert means[0] == pytest.approx(means[2], rel=1e-12)


def test_euler_identity_is_exact_at_beta_zero_and_fails_otherwise():
    """The analytic gradient check, with its own negative control.

    At ``beta = 0`` the rate is velocity-independent, the dynamics are scale-invariant,
    ``T_l`` is homogeneous of degree 1 in ``m``, and Euler's theorem is exact.  The
    right-hand side is a FORWARD quantity, so an adjoint error cannot cancel from both
    sides -- which is what makes this stronger than an AD-vs-FD comparison.
    """
    cs = p2.case(0, 2000, npc_target=500)
    _, _, res = p2.euler_identity_residual(p2.T0_REF, cs, random.PRNGKey(7))
    assert res < 1e-12, f"beta=0 Euler identity should be exact, got {res:.2e}"

    # negative control: scale invariance is broken once the rate depends on |v_r|
    cs2 = p2.case(2, 2000, npc_target=500)
    _, _, res2 = p2.euler_identity_residual(p2.T0_REF, cs2, random.PRNGKey(7))
    assert res2 > 1e-3, f"beta=2 identity should NOT hold, got {res2:.2e}"


def test_argon_is_the_worst_monatomic_gas_for_an_alpha_study():
    """Argon's ``alpha = 1.40`` sits 1.0 % from the stationary point ``sqrt(2)``.

    That makes it the **worst** of the five noble gases Bird tabulates for measuring an
    ``alpha`` sensitivity, and helium (``alpha = 1.26``) the best by 12.6x.  Pinned here
    because the shock example's choice of argon is a physics choice, and anyone adding an
    ``alpha`` verification needs to know it is the ill-conditioned one.
    """
    assert sd.best_for_alpha_study() is sd.HELIUM
    worst = min(sd.MONATOMIC, key=lambda s: abs(s.d_legendre_d_alpha))
    assert worst is sd.ARGON
    assert sd.HELIUM.alpha_conditioning > 12.0
    assert sd.ARGON.alpha_conditioning == pytest.approx(1.0)


def test_bird_table_a3_matches_sparta_ar_vss():
    """Anchor for the transcription: Bird's Ar row must equal SPARTA's shipped data.

    SPARTA ``data/ar.vss`` reads ``Ar 4.11e-10 0.81 273.15 1.4``.  Table A3's name column
    is scrambled by PDF extraction, so rows were assigned by matching diameters against
    Table A2 -- this asserts the assignment landed on the row two independent sources
    agree about.
    """
    assert sd.ARGON.diam == pytest.approx(4.11e-10)
    assert sd.ARGON.omega == pytest.approx(0.81)
    assert sd.ARGON.alpha == pytest.approx(1.4)
    # and the noble-gas diameters must stay below Table A2's VHS values, which is the
    # ordering fact the row assignment relied on
    for s, vhs in (
        (sd.HELIUM, 2.33e-10),
        (sd.NEON, 2.77e-10),
        (sd.ARGON, 4.17e-10),
        (sd.KRYPTON, 4.76e-10),
        (sd.XENON, 5.74e-10),
    ):
        assert s.diam < vhs
        assert s.diam > 0.95 * vhs
