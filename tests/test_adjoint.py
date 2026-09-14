"""Gradients through the **production** solve: `move -> sort -> collide` (section 13.4).

``tests/test_adjoint_oracle.py`` validates the adjoint *construction* against the
papers' own recursions on a hand-rolled Algorithm-1 forward pass.  This file
validates the thing that actually runs: the NTC driver, the cell sort, the specular
wall and the time loop, at ``D = 1, 2, 3``.

Two modes, and the distinction is the point (section 13.4):

* **fixed key** -- validates the *pathwise* gradient against a central difference
  whose step is bisected until no frozen decision flips.  Agreement to round-off.
* **structural** -- for a parameter that reaches ``J`` **only** through a frozen
  decision, asserts the pathwise gradient is exactly zero *and* the scored gradient
  is not.  That pair is the regression guard for the section-8.1 defect, where
  ``dt``, ``fnum`` and ``cell_volume`` read exactly ``0.0`` and nothing failed.

The full ensemble check of ``dE[J]/dtheta`` lives in ``examples/shock_adjoint``;
it needs hundreds of keys and is too slow for the unit suite.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import shock_adjoint as sa
from jax_dsmc import adjoint
from jax_dsmc.boundary import BC
from jax_dsmc.collide import make_prefactor, reduced_mass, species_params, vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.state import CollideState, SpeciesParams

jax.config.update("jax_enable_x64", True)

DIMS = [1, 2, 3]


def tiny(dim):
    """A case small enough for the unit suite and still inside the FD window.

    ``a_max=128`` against a measured peak of 73 attempts.  The obvious ``64`` runs
    with ``attempts_clipped = 9`` every step -- a silent under-collision, gap U4's
    exact failure mode -- which :func:`test_case_does_not_silently_under_collide`
    now refuses to let past.
    """
    return sa.case(dim, npc=20, n_steps=4, n_x=8, transverse=2, a_max=128)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", sa.PATHWISE)
def test_pathwise_gradient_matches_flip_free_finite_difference(dim, name):
    """``jax.grad`` of the plain objective == central FD, with zero decision flips."""
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(0)
    _, grad = sa.value_and_grad(theta, cs, key, False)
    h = sa.largest_flip_free_h(theta, cs, key, name)
    assert h is not None, f"no flip-free step exists at D={dim}, N*C*M={cs.work:.1e}"
    fd, flipped, floor = sa.fd_central(theta, cs, key, name, h)
    assert not flipped
    ad = float(grad[name])
    # tolerance = relative agreement OR the difference's own cancellation noise,
    # whichever is larger.  The second term binds when dJ/dtheta is small next to J.
    tol = max(1e-6 * abs(ad), 20.0 * floor)
    assert abs(fd - ad) <= tol, f"FD {fd:.9e} vs AD {ad:.9e} at h={h:.1e} (tol {tol:.2e})"


@pytest.mark.parametrize("dim", DIMS)
def test_case_does_not_silently_under_collide(dim):
    """``A_max`` must not clip, or every gradient below is measured on a different gas."""
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(0)
    _, diag = sa.evaluate(theta, cs, key, True)
    clipped = int(jnp.sum(diag.attempts_clipped))
    peak = int(jnp.max(diag.max_attempts))
    assert clipped == 0, f"{clipped} clipped attempts; peak {peak} needs A_max > {cs.a_max}"
    assert int(jnp.sum(diag.majorant_violations)) == 0


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", sa.SCORE_ONLY)
def test_score_only_parameters_are_zero_pathwise_and_nonzero_when_scored(dim, name):
    """The section-8.1 regression guard, in its sharpest form.

    ``diam``, ``omega``, ``tref`` reach ``J`` only through ``q``, which is consumed
    by the accept/reject decision; ``nrho`` reaches it only through ``fnum``, which
    is consumed by the attempt count.  Both are frozen, so the *pathwise* gradient
    must be exactly ``0.0`` -- and the *scored* one must not be, because that is
    where the whole sensitivity lives.

    A failure of the second half is what revision 2 shipped: three headline
    gradients reading exactly zero, with no error and no ``NaN``.
    """
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(0)
    _, path = sa.value_and_grad(theta, cs, key, False)
    j, full = sa.value_and_grad(theta, cs, key, True)
    assert float(path[name]) == 0.0
    scale = abs(float(j) / float(theta[name]))
    assert abs(float(full[name])) > 1e-6 * scale, (
        f"{name} carries no score gradient: {float(full[name]):.3e} against a scale of {scale:.3e}"
    )


@pytest.mark.parametrize("dim", DIMS)
def test_scored_and_plain_objectives_have_identical_values(dim):
    """``magic == 1``, so the surrogate is reportable as the objective itself."""
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(1)
    j_plain, _ = sa.value_and_grad(theta, cs, key, False)
    j_scored, _ = sa.value_and_grad(theta, cs, key, True)
    np.testing.assert_allclose(float(j_scored), float(j_plain), rtol=1e-14)


@pytest.mark.parametrize("dim", DIMS)
def test_no_nan_in_any_gradient(dim):
    """Every traced control, every dimension, finite."""
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(2)
    _, grad = sa.value_and_grad(theta, cs, key, True)
    for name, g in grad.items():
        assert np.isfinite(float(g)), f"{name} gradient is {float(g)}"


def test_hand_built_species_params_silently_loses_the_prefactor_path():
    """Why :func:`species_params` exists, asserted rather than commented.

    ``prefactor`` is a *derived* field but a stored leaf.  Differentiate through a
    ``SpeciesParams`` assembled by hand and ``diam`` keeps only its (nonexistent)
    direct path, so the model-calibration gradient reads exactly zero -- the same
    silent-zero failure shape as section 8.1.
    """
    mass = jnp.array([sa.AR_MASS])
    mr = reduced_mass(mass)

    def by_hand(diam):
        sp = SpeciesParams(
            mass=mass,
            diam=jnp.reshape(diam, (1, 1)),
            omega=jnp.array([[sa.AR_OMEGA]]),
            tref=jnp.array([[sa.AR_TREF]]),
            alpha=jnp.array([[sa.AR_ALPHA]]),
            mr=mr,
            prefactor=make_prefactor(  # frozen at the reference diameter
                jnp.array([[sa.AR_DIAM]]), jnp.array([[sa.AR_OMEGA]]), jnp.array([[sa.AR_TREF]]), mr
            ),
        )
        return sp.prefactor[0, 0]

    def properly(diam):
        sp = species_params(
            mass,
            jnp.reshape(diam, (1, 1)),
            jnp.array([[sa.AR_OMEGA]]),
            jnp.array([[sa.AR_TREF]]),
            jnp.array([[sa.AR_ALPHA]]),
        )
        return sp.prefactor[0, 0]

    d = jnp.float64(sa.AR_DIAM)
    assert float(jax.grad(by_hand)(d)) == 0.0
    # d(prefactor)/d(diam) = 2 prefactor / diam, since prefactor ~ diam^2
    np.testing.assert_allclose(
        float(jax.grad(properly)(d)), 2.0 * float(properly(d)) / float(d), rtol=1e-12
    )


@pytest.mark.parametrize("dim", DIMS)
def test_windowed_objective_is_causal_and_value_neutral(dim):
    """``sum_k magic(L_{<=k}) r_k`` has the value of the plain windowed sum (section 11.5).

    The magic box is value-neutral at every step, so accumulating it inside the loop
    must not move the primal at all -- only the gradient, which now scores each
    reward against the decisions that *preceded* it rather than all of them.
    """
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(4)
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    state = initial_state(key, flow, grid, sp, static)
    n_cells = num_cells(grid)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((n_cells,), theta["temp"]),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            n_cells,
            1,
        )
    )
    phi = functools.partial(
        adjoint.phi_windowed,
        inner=adjoint.phi_kinetic,
        axis=0,
        center=cs.window_mfp * lam,
        width=cs.width_mfp * lam,
    )
    args = (state, cstate, grid, sp, params, static, key, phi)
    j_dice, _ = adjoint.windowed_objective(*args, rho=1.0, bc=BC.SPECULAR)
    j_plain, _ = adjoint.windowed_objective(*args, rho=1.0, bc=BC.SPECULAR, baseline=3.0)
    # the baseline is also value-neutral, so all three agree
    np.testing.assert_allclose(float(j_dice), float(j_plain), rtol=1e-13)
    assert np.isfinite(float(j_dice)) and float(j_dice) > 0.0


@pytest.mark.parametrize("dim", DIMS)
def test_collision_invariant_has_exactly_zero_collision_parameter_gradient(dim):
    """`sum |v|^2` is a collision invariant, so its VSS-parameter gradients must vanish.

    VSS conserves the pair's kinetic energy exactly (``|v_r'| == |v_r|`` and the CM
    velocity is untouched), and the specular wall only flips a sign, so the total
    kinetic energy of the gas is *conserved by construction* over the whole run.  It
    therefore cannot depend on ``alpha``, ``diam``, ``omega`` or ``tref`` -- neither
    pathwise nor in expectation.

    That makes it a **zero-check with an analytically known answer**, needing no
    finite difference and no ensemble: a much stronger statement than "FD agrees".
    It catches any leak of a collision parameter into the energy budget, which is the
    failure a conservation bug would produce.

    It is also why this objective cannot be used to *measure* score-term variance --
    the signal is identically zero.  ``examples/shock_adjoint`` and [P2] both use a
    directional moment (`v_y^2`, [P2]'s `T_y`) instead, which collisions do change.
    """
    cs, theta, key = tiny(dim), sa.reference_theta(), jax.random.PRNGKey(11)
    flow, sp, grid, params, static, _ = sa.build(theta, cs)
    del flow, params
    n_cells = num_cells(grid)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((n_cells,), theta["temp"]),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            n_cells,
            1,
        )
    )

    def energy(th):
        f, s, g, pa, st, _ = sa.build(th, cs)
        p0 = initial_state(key, f, g, s, st)
        j, _ = adjoint.terminal_objective(
            p0,
            cstate,
            g,
            s,
            pa,
            st,
            key,
            adjoint.phi_kinetic,
            rho=1.0,
            bc=BC.SPECULAR,
            scored=False,
        )
        return j

    j0 = float(energy(theta))
    grad = jax.grad(energy)(theta)
    # scale by theta so the comparison is dimensionless: d(ln J)/d(ln theta)
    for name in ("alpha", "diam", "omega", "tref"):
        rel = abs(float(grad[name]) * float(theta[name]) / j0)
        assert rel < 1e-10, f"d(ln J)/d(ln {name}) = {rel:.3e}, expected 0 for an invariant"
