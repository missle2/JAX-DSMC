"""Structural gates for wall-driven Couette flow (spec section 9.4).

The **quantitative** gate -- measured shear viscosity against Chapman-Enskog -- is a
multi-minute run and lives in ``examples/couette``, matching how the BKW and
Taylor-Green gates are handled.  What is asserted here is cheap and still
falsifiable: that each wall treatment does the qualitatively right thing, and in
particular that a specular wall drives **no** shear at all, which is the sharpest
single statement distinguishing the two wall models.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from examples import couette as cu  # noqa: E402

WARM = 600
SAMPLE = 600
CELLS = 40
NPC = 120


def _run(**kw):
    return cu.profile(
        cu.Case(n_cells=CELLS, npc=NPC, a_max=256, **kw),
        n_warm=WARM,
        n_sample=SAMPLE,
        sample_stride=20,
    )


@pytest.fixture(scope="module")
def diffuse():
    return _run()


def test_no_silent_attempt_or_wall_truncation(diffuse):
    assert diffuse["clipped"] == 0
    assert diffuse["recrossed"] == 0


def test_diffuse_walls_drive_a_shear_with_the_right_sign(diffuse):
    """Walls sliding in opposite directions must set up a monotone velocity profile."""
    assert diffuse["du_dy"] > 0.0, "streamwise velocity must increase toward the +wall"
    assert diffuse["tau_bulk"] < 0.0, "momentum flows down the gradient"
    u_s = diffuse["u_s"]
    assert u_s[0] < 0.0 < u_s[-1], "the two walls must drag the gas opposite ways"
    # monotone up to Monte Carlo scatter
    assert np.corrcoef(diffuse["y"], u_s)[0, 1] > 0.95


def test_specular_walls_drive_no_shear_at_all(diffuse):
    """The sharpest wall-model discriminator: specular transfers no tangential momentum.

    A specular wall mirrors the normal component and leaves the tangential ones alone,
    so however fast it is declared to slide, it cannot drag the gas.  A diffuse wall at
    the same speed does.  If the wall velocity ever leaked into the specular branch this
    is what would catch it.
    """
    spec = _run(specular=True)
    u_s = spec["u_s"]
    scale = float(np.std(u_s))
    assert abs(float(np.mean(u_s))) < 3.0 * scale / np.sqrt(len(u_s)) + 1.0
    assert abs(spec["du_dy"]) < 0.05 * abs(diffuse["du_dy"]), "specular wall drove a shear"


def test_lowering_accommodation_lowers_the_momentum_transfer(diffuse):
    """``acc`` interpolates specular and diffuse, so the shear must fall with it."""
    partial = _run(acc=0.3)
    assert abs(partial["du_dy"]) < abs(diffuse["du_dy"]), (
        f"acc=0.3 sheared {partial['du_dy']:.3e} vs acc=1 {diffuse['du_dy']:.3e}"
    )
    assert abs(partial["tau_bulk"]) < abs(diffuse["tau_bulk"])


def test_the_regularised_wall_reproduces_the_deterministic_one(diffuse):
    """``epsilon > 0`` must not change the transport it is regularising.

    [P3] is explicit that ``J_eps != J`` even in expectation, so this is a statement
    about magnitude: at ``epsilon = 0.1 dt`` the shear rate must sit within Monte Carlo
    scatter of the deterministic run, not merely have the same sign.
    """
    reg = _run(epsilon_rel=0.1)
    assert reg["du_dy"] == pytest.approx(diffuse["du_dy"], rel=0.15)
    assert reg["tau_bulk"] == pytest.approx(diffuse["tau_bulk"], rel=0.20)


def test_the_measured_viscosity_is_the_right_order(diffuse):
    """A coarse sanity bound only -- the 0.24% gate needs the resolved run.

    At ``dx/lambda = 0.75`` (what this suite can afford) DSMC's own cell-size bias is
    large and *positive*; section 9.4 measures 2.09x at ``dx/lambda = 0.6``.  So this
    asserts only that the measurement is in the right decade and has the right sign,
    and deliberately does **not** pretend to be the physics gate.
    """
    assert diffuse["mu_meas"] > 0.0
    assert 0.3 < diffuse["ratio"] < 6.0, f"ratio {diffuse['ratio']:.3f} is not even close"


def test_wall_temperature_sets_the_bulk_temperature_scale(diffuse):
    """Viscous heating raises ``T`` above the wall, and the walls still bound it."""
    temp = diffuse["temp"]
    assert float(np.min(temp)) > 0.5 * cu.TEMP
    assert float(np.mean(temp)) > cu.TEMP, "shear work must heat the gas"
