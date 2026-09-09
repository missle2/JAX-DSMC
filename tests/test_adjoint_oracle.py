"""The papers' backward recursions, hand-coded, versus ``jax.grad`` (spec section 13).

This is the load-bearing test of the whole architecture.  The project's bet is that
the adjoint DSMC algorithms of [P1] and [P2] *are* reverse-mode AD of the forward
solve plus one DiCE magic box, so that no hand-written backward pass is needed in
production.  These two tests make that bet falsifiable:

* **Oracle 1** transcribes [P1] Algorithm 2 (``alg:adj_DSMC1``) from
  ``arXiv-2009.01363v3/revision_v1_clean_version.tex:1171-1186``.  It is *fully
  independent* -- the matrix ``A`` is written from eq. ``AB_def`` analytically, not
  obtained from AD.
* **Oracle 2** transcribes [P2] Algorithm 3 (``alg:adjoint_VHS``) from
  ``arXiv-2207.11579v2/main.tex:980-994``, including the ``eta`` score terms of
  eq. ``back_eta`` **with all three branches**.

Both run under :func:`pair_perm` rather than the production :func:`pair_ntc`.  The
papers pair *without* replacement -- [P2] eq. ``factorizationEa`` is a product over
``N/2`` disjoint pairs -- so an oracle built on with-replacement pairing would be
validating our recursion against a different algorithm from the published one.

**Scope, stated so it is not overclaimed.**  These oracles drive a faithful
Algorithm-1 forward pass assembled from the *production* kernel (``vss.scatter``,
``vss.collision_rate``, ``vss.log_h``) and the *production* surrogate
(``adjoint.surrogate``).  They therefore validate the seeding, the recursion, the
``eta`` terms, the both-branches requirement, the ``(phi_i + phi_i1)`` reward
attribution and the accumulator bookkeeping.  They do **not** exercise
``collide``'s NTC driver, whose gradients are checked by finite differences in
``tests/test_adjoint.py`` and ``examples/shock_adjoint``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jax_dsmc import adjoint
from jax_dsmc.collide import pair_perm, species_params
from jax_dsmc.collide.vss import collision_rate, log_h, scatter
from jax_dsmc.constants import K_BOLTZ

jax.config.update("jax_enable_x64", True)

N_PART = 64
"""Even, so ``pair_perm`` produces a perfect matching with no leftover particle."""

N_STEPS = 5
MASS = 6.63e-26
TREF = 273.15
DIAM = 4.11e-10
RHO = 1.0


# --- shared forward pass ---------------------------------------------------


def _params(omega, alpha):
    return species_params(
        jnp.array([MASS]),
        jnp.array([[DIAM]]),
        jnp.array([[omega]]),
        jnp.array([[TREF]]),
        jnp.array([[alpha]]),
    )


def _draws(key, n_steps, n_pair):
    """All randomness for the run, drawn up front -- the RNG contract, in miniature.

    Returned as one stacked pytree so the forward pass is a pure function of
    ``(v0, sp, draws)``, which is what lets ``jax.grad`` and the hand-coded
    recursion see *identical* frozen decisions.
    """
    k_perm, k_chi, k_eps, k_acc = random.split(key, 4)
    return {
        "perm": jnp.stack(
            [random.permutation(random.fold_in(k_perm, k), N_PART) for k in range(n_steps)]
        ),
        "chi": random.uniform(k_chi, (n_steps, n_pair)),
        "eps": random.uniform(k_eps, (n_steps, n_pair)),
        "acc": random.uniform(k_acc, (n_steps, n_pair)),
    }


def _labframe_scatter(vi, vj, sp, u_chi, u_eps):
    """[P1]'s own collision map: ``sigma`` drawn in the **lab frame**.

    ``cos(theta) = 2 u_chi - 1``, ``azimuth = 2 pi u_eps``, giving a direction that
    is uniform on the sphere and -- crucially -- *independent of the pre-collision
    velocities*.  That independence is [P1]'s stated assumption
    (``revision_v1_clean_version.tex:468``: "the collision parameter vector sigma is
    chosen independently of v and v1 ... there is no need to perturb sigma; i.e.
    delta sigma = 0"), and it is what makes eq. ``AB_def`` the exact Jacobian.

    Statistically identical to :func:`scatter` at ``omega = 1, alpha = 1`` -- both
    sample an isotropic post-collision direction -- but **not** identical as a map
    at frozen randomness.  See
    :func:`test_production_scatter_is_not_p1_A_and_that_is_expected`.
    """
    del sp
    cos_t = 2.0 * u_chi - 1.0
    sin_t = jnp.sqrt(jnp.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
    azi = 2.0 * jnp.pi * u_eps
    s_hat = jnp.stack([cos_t, sin_t * jnp.cos(azi), sin_t * jnp.sin(azi)], axis=-1)
    vcm = 0.5 * (vi + vj)
    vrm = jnp.linalg.norm(vi - vj, axis=-1, keepdims=True)
    half = 0.5 * vrm * s_hat
    return vcm + half, vcm - half


def _one_step(v, logh, draws, k, sp, sigma, kernel=scatter):
    """One Algorithm-1 step: pair everyone off, test, scatter the accepted pairs.

    ``sigma`` is the majorant, held **fixed** for the whole run (the stale-max rule
    of section 7.2 and [P2]'s fixed ``Sigma``).  Returns the updated velocities,
    the updated score accumulator, and the frozen decisions the backward
    recursions need.
    """
    perm = draws["perm"][k]
    i, j = perm[0::2], perm[1::2]
    vi, vj = v[i], v[j]

    q = collision_rate(vi, vj, 0, 0, sp)
    accept = draws["acc"][k] < q / sigma  # FROZEN
    vi_n, vj_n = kernel(vi, vj, sp, draws["chi"][k], draws["eps"][k])

    v = v.at[i].set(jnp.where(accept[:, None], vi_n, vi))
    v = v.at[j].set(jnp.where(accept[:, None], vj_n, vj))
    lh = log_h(q, sigma, accept)
    logh = logh.at[i].add(lh).at[j].add(lh)
    # `vr_post` is recorded rather than recomputed: it is [P1]'s `sigma` direction,
    # and re-deriving it later from the draws would risk silently pairing a step
    # with another step's uniforms.
    return v, logh, (i, j, accept, vi, vj, vi_n - vj_n)


def _vss_kernel(vi, vj, sp, u_chi, u_eps):
    """The production VSS scatter, with the species indices bound."""
    return scatter(vi, vj, 0, 0, sp, u_chi, u_eps)


def _forward(v0, sp, draws, sigma, n_steps, kernel=_vss_kernel):
    """``n_steps`` of Algorithm 1.  Returns ``(v_M, L, history)``."""
    v, logh, hist = v0, jnp.zeros((N_PART,)), []
    for k in range(n_steps):
        v, logh, rec = _one_step(v, logh, draws, k, sp, sigma, kernel)
        hist.append(rec)
    return v, logh, hist


def _phi(x, v, component, power):
    del x
    return v[:, component] ** power


def _surrogate_of(v0, sp, draws, sigma, n_steps, phi, scored, kernel=_vss_kernel):
    """The objective as a function of the initial velocities -- what AD differentiates."""
    v, logh, _ = _forward(v0, sp, draws, sigma, n_steps, kernel)
    alive = jnp.ones((N_PART,), dtype=bool)
    vals = phi(None, v)
    if scored:
        return adjoint.surrogate(vals, logh, alive, rho=RHO)
    return adjoint.objective(vals, alive, rho=RHO)


def _initial_velocities(key, temp=TREF):
    return jnp.sqrt(K_BOLTZ * temp / MASS) * random.normal(key, (N_PART, 3), dtype=jnp.float64)


# --- Oracle 1: [P1] Algorithm 2, Maxwell molecules -------------------------


def _p1_matrix(sigma_hat, alpha_hat):
    """``A(sigma, alpha)`` from [P1] eq. ``AB_def``, written analytically.

    ``A = 0.5 * [[I + sigma alpha^T, I - sigma alpha^T],
                 [I - sigma alpha^T, I + sigma alpha^T]]``

    Deliberately **not** obtained from ``jax.jacobian``: an oracle that asked AD
    for the very Jacobian it is meant to be checking would be circular.
    """
    p = jnp.outer(sigma_hat, alpha_hat)
    eye = jnp.eye(3)
    return 0.5 * jnp.block([[eye + p, eye - p], [eye - p, eye + p]])


def _unit(u):
    return u / jnp.linalg.norm(u, axis=-1, keepdims=True)


def test_oracle1_p1_algorithm2_matches_jax_grad():
    """[P1] Algorithm 2 == ``jax.grad``, Maxwell molecules, every collision accepted.

    ``omega = 1`` makes ``q`` a constant, so setting ``Sigma`` to exactly that
    constant gives ``p == 1``: every candidate collides, ``log h == 0``, and no
    score term arises.  The adjoint is then purely pathwise, which is precisely
    [P1]'s regime.

    **This runs [P1]'s own collision map** (:func:`_labframe_scatter`), not
    SPARTA's.  That is not a weakening of the test -- it is what makes it a test of
    [P1]'s *algorithm*.  Eq. ``AB_def`` is derived under ``delta sigma = 0``, and
    only a lab-frame sampler satisfies that; the companion test
    :func:`test_production_scatter_is_not_p1_A_and_that_is_expected` measures what
    changes when the production sampler is substituted, and why.

    Sign convention (section 11.1): [P1] seeds ``gamma_F = -dr/dv`` and computes
    ``dJ/dv_0 = -(1/N) gamma_0``.  We follow [P2]'s unsigned convention in
    production, so the comparison carries the negation explicitly here.
    """
    key = random.PRNGKey(0)
    sp = _params(omega=1.0, alpha=1.0)
    sigma = sp.prefactor[0, 0]  # q is constant at omega == 1, so p == 1 exactly
    v0 = _initial_velocities(random.fold_in(key, 1))
    draws = _draws(random.fold_in(key, 2), N_STEPS, N_PART // 2)
    phi = functools.partial(_phi, component=0, power=2)

    _, logh, hist = _forward(v0, sp, draws, sigma, N_STEPS, _labframe_scatter)
    assert float(jnp.max(jnp.abs(logh))) == 0.0, "p == 1 must give log h == 0 everywhere"
    assert all(bool(jnp.all(rec[2])) for rec in hist), "p == 1 must accept every candidate"

    # --- the paper's recursion, seeded per eq. gammaFinal ---------------------
    v_m = _forward(v0, sp, draws, sigma, N_STEPS, _labframe_scatter)[0]
    gamma = -jax.grad(lambda v: jnp.sum(phi(None, v)))(v_m)
    for i, j, accept, vi, vj, vr_post in reversed(hist):
        # Equal masses, so v_i' - v_j' is exactly v_r'; alpha and sigma are the
        # pre- and post-collision relative directions of [P1] eq. AB_def.
        alpha_hat, sigma_hat = _unit(vi - vj), _unit(vr_post)
        gi, gj = gamma[i], gamma[j]
        sym = 0.5 * (gi + gj)
        dot = jnp.sum((gi - gj) * sigma_hat, axis=-1)[:, None]
        gamma = gamma.at[i].set(jnp.where(accept[:, None], sym + 0.5 * dot * alpha_hat, gi))
        gamma = gamma.at[j].set(jnp.where(accept[:, None], sym - 0.5 * dot * alpha_hat, gj))

    # dJ/dv_0 = -(1/N) gamma_0, [P1] eq. DalphaEqtn
    expected = -gamma / N_PART * RHO
    got = jax.grad(_surrogate_of)(v0, sp, draws, sigma, N_STEPS, phi, True, _labframe_scatter)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-11, atol=1e-25)


def test_production_scatter_is_not_p1_A_and_that_is_expected():
    """SPARTA's sampler has a **rank-6** pair Jacobian; [P1]'s ``A`` is rank 4.

    This is the reason :func:`test_oracle1_p1_algorithm2_matches_jax_grad` runs
    [P1]'s own map, and it is a genuine property of the discretisation rather than
    a defect -- so it is asserted, not worked around.

    ``scatter`` builds the post-collision direction **in the frame aligned to**
    ``v_r`` (``collide_vss.cpp:429-437``).  At frozen ``(u_chi, u_eps)`` the map
    ``v_r -> v_r'`` is therefore a rotation *about the ``v_r`` axis*, so ``v'``
    depends on the direction of ``v_r`` and not only on ``v_cm`` and ``|v_r|``.
    That is two extra degrees of freedom: rank 6 instead of 4.

    Both maps push forward the same distribution, so both are valid Maxwell-molecule
    DSMC and both give unbiased pathwise gradients -- they are different
    *reparameterisations*, exactly as with a Gaussian.  Section 7.5 notes that AD
    picks the extra terms up with no special handling; what is recorded here is
    that the divergence is already present at ``omega = 1, alpha = 1``, where [P1]
    would predict none.

    **Consequence for section 17.1**: the ``rank(A) == 4`` structural assertion is
    about the analytic matrix of eq. ``AB_def``, *not* about our collision map.  A
    test asserting rank 4 of ``jax.jacobian(scatter)`` would fail today.
    """
    key = random.PRNGKey(0)
    sp = _params(omega=1.0, alpha=1.0)
    for t in range(6):
        k1, k2, k3, k4 = random.split(random.fold_in(key, t), 4)
        vi = 300.0 * random.normal(k1, (3,))
        vj = 300.0 * random.normal(k2, (3,))
        u_chi, u_eps = random.uniform(k3), random.uniform(k4)

        vi_n, vj_n = _vss_kernel(vi, vj, sp, u_chi, u_eps)
        a_hat, s_hat = _unit(vi - vj), _unit(vi_n - vj_n)
        a_mat = _p1_matrix(s_hat, a_hat)

        j_sparta = _pair_jacobian(vi, vj, sp, u_chi, u_eps).T  # un-transpose: d(out)/d(in)
        j_lab = jnp.transpose(_pair_jacobian_of(_labframe_scatter, vi, vj, sp, u_chi, u_eps))

        # energy is conserved by both, so |v_r| is untouched
        np.testing.assert_allclose(
            float(jnp.linalg.norm(vi_n - vj_n)), float(jnp.linalg.norm(vi - vj)), rtol=1e-14
        )
        # [P1]'s A is the EXACT Jacobian of [P1]'s map ...
        lab_i, lab_j = _labframe_scatter(vi, vj, sp, u_chi, u_eps)
        s_hat_lab = _unit(lab_i - lab_j)
        np.testing.assert_allclose(
            np.asarray(j_lab), np.asarray(_p1_matrix(s_hat_lab, a_hat)), atol=1e-9
        )
        assert np.linalg.matrix_rank(np.asarray(j_lab), tol=1e-8) == 4
        # ... and NOT of ours.
        assert np.linalg.matrix_rank(np.asarray(j_sparta), tol=1e-8) == 6
        rel = float(jnp.linalg.norm(j_sparta - a_mat) / jnp.linalg.norm(a_mat))
        assert rel > 0.1, f"expected a large divergence, got {rel:.2e}"


def _pair_jacobian_of(kernel, vi, vj, sp, u_chi, u_eps):
    """``[d(v', v_1')/d(v, v_1)]^T`` for an arbitrary pair kernel."""
    jac = jax.jacobian(kernel, argnums=(0, 1))(vi, vj, sp, u_chi, u_eps)
    top = jnp.concatenate([jac[0][0], jac[1][0]], axis=1)
    bot = jnp.concatenate([jac[0][1], jac[1][1]], axis=1)
    return jnp.concatenate([top, bot], axis=0).T


def test_oracle1_matrix_is_rank_four():
    """``rank(A) == 4`` and ``svd(A) == [1, 1, 1, 1, 0, 0]`` (section 17.1).

    A structural assertion about the collision map, not about the adjoint.  ``A``
    acts as the identity on the 3-D symmetric subspace ``(u, u)`` and as the rank-1
    projector ``sigma alpha^T`` on the antisymmetric one, so four singular values
    are 1 and two are 0.

    This is also the standing correction to [P1], which writes ``B = A^T = A^-1``:
    ``A`` is singular, so ``A^-1`` does not exist.  ``B = A^T`` is correct and is
    all the adjoint needs.  If a refactor ever makes ``A`` full rank, the collision
    map has been altered and the physics has changed.
    """
    key = random.PRNGKey(3)
    for k in range(8):
        s_hat = _unit(random.normal(random.fold_in(key, 2 * k), (3,)))
        a_hat = _unit(random.normal(random.fold_in(key, 2 * k + 1), (3,)))
        a = np.asarray(_p1_matrix(s_hat, a_hat))
        sv = np.linalg.svd(a, compute_uv=False)
        assert np.linalg.matrix_rank(a) == 4
        np.testing.assert_allclose(sv, [1, 1, 1, 1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(a @ a.T @ a, a, atol=1e-12)


# --- Oracle 2: [P2] Algorithm 3, general VSS kernel ------------------------


def _pair_jacobian(vi, vj, sp, u_chi, u_eps):
    """``D = [d(v', v_1')/d(v, v_1)]^T`` for one pair, [P2] eq. ``back_D``, ``j=2``.

    Obtained from ``jax.jacobian`` of ``scatter``, which section 13.2 permits and
    flags: VSS has no simple closed form for ``D`` -- which is exactly why [P2]
    notes that its ``C`` and [P1]'s ``A`` "may not be the same".  So this oracle
    validates the *recursion structure*, not ``scatter``'s own Jacobian; that is
    covered by the finite-difference tests in ``tests/test_vss.py``.
    """
    jac = jax.jacobian(_vss_kernel, argnums=(0, 1))(vi, vj, sp, u_chi, u_eps)
    # jac[a][b] has shape (2, 3, 3): d(out_a)/d(in_b) stacked over the two outputs
    top = jnp.concatenate([jac[0][0], jac[1][0]], axis=1)  # d(v', v1')/dv
    bot = jnp.concatenate([jac[0][1], jac[1][1]], axis=1)  # d(v', v1')/dv1
    return jnp.concatenate([top, bot], axis=0).T


def _eta_term(vi, vj, sp, sigma, accept, reward):
    """``eta`` from [P2] eq. ``back_eta``, **all three branches**.

    ``d log h / dv_i`` times ``(phi_i^M + phi_i1^M)``.  ``log h`` is
    ``log(q/Sigma)`` on a real collision and ``log(1 - q/Sigma)`` on a virtual one
    that did not become real -- the second branch is the one that is easy to drop,
    and dropping it gives a subtly biased gradient that still looks healthy.

    Both partners receive the derivative with respect to *their own* velocity, and
    both receive the *same* reward ``phi_i + phi_i1``: that pairing is the whole
    content of the ``(phi_i^M + phi_i1^M)`` factor.
    """
    g_i, g_j = jax.grad(_log_h_of, argnums=(0, 1))(vi, vj, sp, sigma, accept)
    return (RHO / N_PART) * g_i * reward, (RHO / N_PART) * g_j * reward


def _log_h_of(vi, vj, sp, sigma, accept):
    return log_h(collision_rate(vi, vj, 0, 0, sp), sigma, accept)


@pytest.mark.parametrize("power", [2, 4])
def test_oracle2_p2_algorithm3_matches_jax_grad(power):
    """[P2] Algorithm 3 == ``jax.grad`` of the surrogate, general VSS kernel.

    ``omega != 1`` and ``alpha != 1``, with ``Sigma`` a genuine strict majorant so
    that accepted *and* rejected candidates both occur -- which is what makes the
    ``eta`` terms live.

    This is the test that catches: dropping the reject-branch score term;
    ``stop_gradient``-ing the wrong thing; accumulating ``log h`` onto only one
    partner; and letting ``Sigma`` vary within the step.
    """
    key = random.PRNGKey(11)
    sp = _params(omega=0.81, alpha=1.4)
    v0 = _initial_velocities(random.fold_in(key, 1))
    draws = _draws(random.fold_in(key, 2), N_STEPS, N_PART // 2)
    phi = functools.partial(_phi, component=1, power=power)

    # a strict majorant: 2x the largest q any pair will ever present
    q_all = collision_rate(v0[:, None, :], v0[None, :, :], 0, 0, sp)
    sigma = 2.0 * jnp.max(q_all)

    v_m, logh, hist = _forward(v0, sp, draws, sigma, N_STEPS)
    n_acc = int(sum(int(jnp.sum(rec[2])) for rec in hist))
    n_tot = N_STEPS * (N_PART // 2)
    assert 0 < n_acc < n_tot, f"need both branches live; got {n_acc}/{n_tot} accepted"

    phi_m = phi(None, v_m)  # phi_i^M, the reward the score terms are weighted by
    gamma = (RHO / N_PART) * jax.grad(lambda v: jnp.sum(phi(None, v)))(v_m)

    for k in reversed(range(N_STEPS)):
        i, j, accept, vi, vj, _ = hist[k]
        reward = (phi_m[i] + phi_m[j])[:, None]
        gi, gj = gamma[i], gamma[j]

        d_mat = jax.vmap(_pair_jacobian, in_axes=(0, 0, None, 0, 0))(
            vi, vj, sp, draws["chi"][k], draws["eps"][k]
        )
        stacked = jnp.concatenate([gi, gj], axis=-1)[:, :, None]
        transported = jnp.squeeze(d_mat @ stacked, -1)
        # j=1 and j=3 of eq. back_D are the identity; only a real collision has D
        transported = jnp.where(accept[:, None], transported, jnp.concatenate([gi, gj], -1))

        eta_i, eta_j = jax.vmap(_eta_term, in_axes=(0, 0, None, None, 0, 0))(
            vi, vj, sp, sigma, accept, reward
        )
        gamma = gamma.at[i].set(transported[:, :3] + eta_i)
        gamma = gamma.at[j].set(transported[:, 3:] + eta_j)

    got = jax.grad(_surrogate_of)(v0, sp, draws, sigma, N_STEPS, phi, True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(gamma), rtol=1e-10, atol=1e-25)


def test_oracle2_dropping_the_reject_branch_is_detected():
    """The both-branches requirement is real, not decorative.

    Rebuilds the surrogate with the ``j=3`` ("virtual, not real") score term
    suppressed and asserts the result **disagrees** with Algorithm 3.  Without this
    the previous test could pass for a version of the code that scores only
    accepted candidates -- which is the single most likely way to get a plausible
    but biased gradient (section 7.3).
    """
    key = random.PRNGKey(11)
    sp = _params(omega=0.81, alpha=1.4)
    v0 = _initial_velocities(random.fold_in(key, 1))
    draws = _draws(random.fold_in(key, 2), N_STEPS, N_PART // 2)
    phi = functools.partial(_phi, component=1, power=2)
    q_all = collision_rate(v0[:, None, :], v0[None, :, :], 0, 0, sp)
    sigma = 2.0 * jnp.max(q_all)

    full = jax.grad(_surrogate_of)(v0, sp, draws, sigma, N_STEPS, phi, True)
    accept_only = jax.grad(_surrogate_accept_only)(v0, sp, draws, sigma, N_STEPS, phi)
    rel = float(jnp.linalg.norm(full - accept_only) / jnp.linalg.norm(full))
    assert rel > 1e-3, f"suppressing the reject branch must change the gradient; rel={rel:.2e}"


def _surrogate_accept_only(v0, sp, draws, sigma, n_steps, phi):
    """Deliberately wrong surrogate: scores only the accepted candidates."""
    v, logh = v0, jnp.zeros((N_PART,))
    for k in range(n_steps):
        perm = draws["perm"][k]
        i, j = perm[0::2], perm[1::2]
        vi, vj = v[i], v[j]
        q = collision_rate(vi, vj, 0, 0, sp)
        accept = draws["acc"][k] < q / sigma
        vi_n, vj_n = _vss_kernel(vi, vj, sp, draws["chi"][k], draws["eps"][k])
        v = v.at[i].set(jnp.where(accept[:, None], vi_n, vi))
        v = v.at[j].set(jnp.where(accept[:, None], vj_n, vj))
        lh = jnp.where(accept, log_h(q, sigma, accept), 0.0)  # the defect
        logh = logh.at[i].add(lh).at[j].add(lh)
    return adjoint.surrogate(phi(None, v), logh, jnp.ones((N_PART,), dtype=bool), rho=RHO)


def test_pair_perm_is_a_perfect_matching():
    """The oracles' pairing must be what the papers assume: disjoint, no self-pairs."""
    for seed in range(16):
        i, j = pair_perm(random.PRNGKey(seed), N_PART)
        used = np.concatenate([np.asarray(i), np.asarray(j)])
        assert sorted(used.tolist()) == list(range(N_PART))
        assert not np.any(np.asarray(i) == np.asarray(j))


def test_magic_box_is_value_neutral():
    """``magic(L) == 1`` in value, ``dL`` in gradient -- so ``J~`` can be reported as ``J``."""
    logh = jnp.array([-3.0, 0.0, 1.5, -0.25])
    np.testing.assert_allclose(np.asarray(adjoint.magic(logh)), np.ones(4), rtol=0, atol=0)
    got = jax.grad(lambda x: jnp.sum(jnp.array([2.0, 3.0, 5.0, 7.0]) * adjoint.magic(x)))(logh)
    np.testing.assert_allclose(np.asarray(got), [2.0, 3.0, 5.0, 7.0], rtol=1e-14)


def test_baseline_is_value_neutral_and_changes_no_gradient_in_expectation():
    """A constant baseline shifts neither the value nor the *pathwise* gradient.

    ``E[grad L] == 0`` makes the score half unbiased under any constant ``b``, so
    the baseline is a pure variance knob (section 11.4).  What is checkable
    deterministically is the other half: the value is exactly unchanged, and the
    difference between the two gradients is ``b * sum(w * grad L)`` -- i.e. it lies
    entirely in the score direction.
    """
    key = random.PRNGKey(5)
    phi_v = random.normal(random.fold_in(key, 0), (N_PART,))
    logh = 0.1 * random.normal(random.fold_in(key, 1), (N_PART,))
    alive = jnp.ones((N_PART,), dtype=bool)
    j0 = adjoint.surrogate(phi_v, logh, alive, RHO, None)
    jb = adjoint.surrogate(phi_v, logh, alive, RHO, "mean")
    np.testing.assert_allclose(float(jb), float(j0), rtol=1e-14)
    np.testing.assert_allclose(float(j0), float(adjoint.objective(phi_v, alive, RHO)), rtol=1e-14)

    g0 = jax.grad(adjoint.surrogate, argnums=1)(phi_v, logh, alive, RHO, None)
    gb = jax.grad(adjoint.surrogate, argnums=1)(phi_v, logh, alive, RHO, "mean")
    b = float(jnp.mean(phi_v))
    expected = -b / N_PART * RHO * np.ones(N_PART)
    np.testing.assert_allclose(np.asarray(gb - g0), expected, rtol=1e-12)


# --- Oracle 3: 1D two-step, to pin the intra-step ordering (section 13.3) ---
#
# Oracles 1 and 2 are 0-D, so the position cotangent `alpha` is identically zero and
# both candidate adjoint forms collapse to `beta_k = B_k beta_{k+1}`.  They therefore
# *provably cannot* tell `move -> collide` from `collide -> move` -- that is gap U5,
# and this section is what closes it.
#
# Section 12.1 records the divergence.  We run `move -> sort -> collide`, whose adjoint
# carries the transport term OUTSIDE `B_k`:
#
#     beta_k = B_k beta_{k+1} + dt P* alpha_{k+1},      alpha_k = alpha_{k+1}
#
# [P3] `main.tex:721` runs `collide -> move`, which puts it INSIDE:
#
#     beta_k = B_k (beta_{k+1} + dt P* alpha_{k+1}).
#
# The minimum configuration that separates them is 1 dimension, 2 cells, 2 steps,
# periodic.  Two steps is the minimum that exercises the coupling: over a single step
# the seed carries no transport contribution into `B_k`, so the two forms differ only
# in a term that has not yet been through a collision.

O3_N = 32
O3_CELLS = 2
O3_STEPS = 2
O3_L = 1.0e-3
O3_DT = 1.0e-6
O3_VTH2 = K_BOLTZ * TREF / MASS


def _o3_terminal(x, v):
    """``sum v_x^2 + vth^2 (x/L)^2`` -- depends on **both** endpoints.

    The `x` term is the entire point: it is what makes ``alpha`` nonzero.  With a
    velocity-only objective ``alpha == 0``, the transport term vanishes and this
    oracle degenerates into Oracle 1.  The ``vth^2`` weight puts the two
    contributions within an order of magnitude of one another.
    """
    return jnp.sum(v[:, 0] ** 2 + O3_VTH2 * (x[:, 0] / O3_L) ** 2)


def _o3_pad(alpha):
    """``P*``: adjoint of ``v -> v[:, :D]``, i.e. zero-pad ``R^D -> R^3``."""
    return jnp.pad(alpha, ((0, 0), (0, 3 - alpha.shape[1])))


def _o3_advect(x, v):
    """``x += dt v_x`` then the periodic fold.  ``dW/dx == 1`` almost everywhere."""
    return jnp.mod(x + O3_DT * v[:, :1], O3_L)


def _o3_collide(v, i, j, sp, u_chi, u_eps):
    """Every pair collides: ``omega == 1`` with ``Sigma == q`` gives ``p == 1``."""
    vi_n, vj_n = _labframe_scatter(v[i], v[j], sp, u_chi, u_eps)
    return v.at[i].set(vi_n).at[j].set(vj_n)


def _o3_pairs(cells, key):
    """Disjoint pairs *within each cell* -- ``pair_perm`` semantics, applied per cell."""
    cells = np.asarray(cells)
    ii, jj = [], []
    for c in range(O3_CELLS):
        members = np.flatnonzero(cells == c)
        order = np.asarray(random.permutation(random.fold_in(key, c), len(members)))
        members = members[order]
        half = len(members) // 2
        ii.append(members[: 2 * half : 2])
        jj.append(members[1 : 2 * half : 2])
    return jnp.asarray(np.concatenate(ii)), jnp.asarray(np.concatenate(jj))


def _o3_forward(x0, v0, plan, sp, collide_first):
    """Two steps in the requested order, replaying the frozen decisions in ``plan``."""
    x, v = x0, v0
    for i, j, u_chi, u_eps in plan:
        if collide_first:
            v = _o3_collide(v, i, j, sp, u_chi, u_eps)
            x = _o3_advect(x, v)
        else:
            x = _o3_advect(x, v)
            v = _o3_collide(v, i, j, sp, u_chi, u_eps)
    return x, v


def _o3_plan_and_history(key, x0, v0, sp, collide_first):
    """One concrete pass: discover cell membership, freeze the decisions, record ``B_k``.

    ``plan`` is what the differentiated forward replays -- pair indices and scattering
    uniforms, frozen exactly as production freezes them.  ``history`` additionally
    carries the pre- and post-collision relative directions each step, which are the
    ``alpha`` and ``sigma`` of [P1] eq. ``AB_def`` and hence all the backward pass
    needs to apply ``B_k`` analytically.
    """
    plan, history, x, v = [], [], x0, v0
    dx = O3_L / O3_CELLS
    for k in range(O3_STEPS):
        k_perm, k_chi, k_eps = random.split(random.fold_in(key, k), 3)
        if not collide_first:
            x = _o3_advect(x, v)
        cell = jnp.clip(jnp.floor((x[:, 0] - 0.0) / dx).astype(jnp.int32), 0, O3_CELLS - 1)
        i, j = _o3_pairs(cell, k_perm)
        u_chi = random.uniform(k_chi, (i.size,))
        u_eps = random.uniform(k_eps, (i.size,))
        plan.append((i, j, u_chi, u_eps))

        vi, vj = v[i], v[j]
        v = _o3_collide(v, i, j, sp, u_chi, u_eps)
        history.append((i, j, _unit(vi - vj), _unit(v[i] - v[j])))
        if collide_first:
            x = _o3_advect(x, v)
    return plan, history


def _o3_apply_B(gamma, rec):
    """``B_k = A^T`` from [P1] eq. ``AB_def``, applied to the pair cotangents.

    Analytic, never ``jax.jacobian``: asking AD for the Jacobian this oracle is meant
    to be checking would be circular.  Every pair collided, so there is no reject
    branch to mask.
    """
    i, j, alpha_hat, sigma_hat = rec
    gi, gj = gamma[i], gamma[j]
    sym = 0.5 * (gi + gj)
    dot = jnp.sum((gi - gj) * sigma_hat, axis=-1)[:, None]
    return gamma.at[i].set(sym + 0.5 * dot * alpha_hat).at[j].set(sym - 0.5 * dot * alpha_hat)


def _o3_backward(x_m, v_m, history, transport_inside):
    """The hand-coded recursion, in whichever of the two forms is requested."""
    alpha = jax.grad(_o3_terminal, argnums=0)(x_m, v_m)
    beta = jax.grad(_o3_terminal, argnums=1)(x_m, v_m)
    for rec in reversed(history):
        if transport_inside:  # [P3]: beta_k = B_k (beta_{k+1} + dt P* alpha_{k+1})
            beta = _o3_apply_B(beta + O3_DT * _o3_pad(alpha), rec)
        else:  # ours:  beta_k = B_k beta_{k+1} + dt P* alpha_{k+1}
            beta = _o3_apply_B(beta, rec) + O3_DT * _o3_pad(alpha)
        # alpha_k = alpha_{k+1} in both: the periodic fold has unit derivative a.e.
    return alpha, beta


def _o3_setup(collide_first):
    key = random.PRNGKey(17)
    sp = _params(omega=1.0, alpha=1.0)
    x0 = O3_L * random.uniform(random.fold_in(key, 1), (O3_N, 1), dtype=jnp.float64)
    v0 = jnp.sqrt(O3_VTH2) * random.normal(random.fold_in(key, 2), (O3_N, 3), dtype=jnp.float64)
    plan, history = _o3_plan_and_history(random.fold_in(key, 3), x0, v0, sp, collide_first)
    x_m, v_m = _o3_forward(x0, v0, plan, sp, collide_first)
    got = jax.grad(
        lambda x, v: _o3_terminal(*_o3_forward(x, v, plan, sp, collide_first)), argnums=(0, 1)
    )(x0, v0)
    return sp, plan, history, x_m, v_m, got


def _o3_mismatch(got, expected):
    """Relative discrepancy, on the velocity cotangent where the two forms differ."""
    return float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected))


def test_oracle3_transport_term_sits_outside_B_k():
    """**The M2 gate (gap U5).**  ``beta_k = B_k beta_{k+1} + dt P* alpha_{k+1}``.

    1-D, 2 cells, 2 steps, periodic, with ``B_k`` written analytically from [P1]
    eq. ``AB_def`` -- never from ``jax.jacobian``, which would be circular.  This is
    the first oracle in which the position cotangent ``alpha`` is nonzero, and so the
    first that can see the intra-step ordering at all.
    """
    _, _, history, x_m, v_m, (got_a, got_b) = _o3_setup(collide_first=False)
    alpha, beta = _o3_backward(x_m, v_m, history, transport_inside=False)

    # the test is vacuous unless the position cotangent actually survives to t=0
    assert float(jnp.max(jnp.abs(alpha))) > 0.0, "alpha == 0: the objective ignores x"
    np.testing.assert_allclose(np.asarray(got_b), np.asarray(beta), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(got_a), np.asarray(alpha), rtol=1e-12, atol=0.0)


def test_oracle3_rejects_the_p3_inside_B_form():
    """Negative control 1: substituting [P3]'s ``B_k(beta + dt P* alpha)`` must FAIL.

    Without this the oracle above could pass for the wrong reason -- if the transport
    term were negligible, both forms would agree and the test would assert nothing.
    """
    _, _, history, x_m, v_m, (_, got_b) = _o3_setup(collide_first=False)
    _, beta_inside = _o3_backward(x_m, v_m, history, transport_inside=True)
    mismatch = _o3_mismatch(got_b, beta_inside)
    assert mismatch > 1e-3, (
        f"[P3]'s inside-B form is indistinguishable here ({mismatch:.2e}); "
        "the oracle has lost its discriminating power"
    )


def test_oracle3_rejects_the_reordered_loop():
    """Negative control 2: reordering the forward loop must swap which form is right.

    Run ``collide -> move`` instead and the OUTSIDE recursion breaks while [P3]'s
    INSIDE one becomes exact.  Asserting both directions is what makes this a
    discriminator rather than a coincidence -- an oracle that merely failed on the
    reordered loop could be failing for any reason at all.
    """
    _, _, history, x_m, v_m, (_, got_b) = _o3_setup(collide_first=True)
    _, beta_outside = _o3_backward(x_m, v_m, history, transport_inside=False)
    _, beta_inside = _o3_backward(x_m, v_m, history, transport_inside=True)

    assert _o3_mismatch(got_b, beta_outside) > 1e-3, "our form must NOT match [P3]'s loop"
    np.testing.assert_allclose(np.asarray(got_b), np.asarray(beta_inside), rtol=1e-12, atol=0.0)


def test_oracle3_position_cotangent_is_unchanged_by_a_step():
    """``alpha_k = alpha_{k+1}`` in **both** orderings -- the periodic fold is unit slope."""
    for collide_first in (False, True):
        _, _, history, x_m, v_m, (got_a, _) = _o3_setup(collide_first)
        seed = jax.grad(_o3_terminal, argnums=0)(x_m, v_m)
        alpha, _ = _o3_backward(x_m, v_m, history, transport_inside=collide_first)
        np.testing.assert_allclose(np.asarray(alpha), np.asarray(seed), rtol=1e-14)
        np.testing.assert_allclose(np.asarray(got_a), np.asarray(seed), rtol=1e-12)


def test_oracle3_one_step_would_already_discriminate():
    """Spec correction: section 13.3 says two steps is the *minimum* that separates them.

    Measured, that is not so -- the two forms differ by ``dt (I - B) P* alpha``, which
    is already nonzero after a single step (2.7e-2 here against 9.8e-2 at two steps).
    Two steps remains the right configuration, because it is the first that pushes a
    *transported* cotangent back through a second ``B_k`` and so tests the composition
    rather than one application, and because the margin is ~4x wider.  Recorded here
    rather than left in the spec as an unverified rationale.
    """
    global O3_STEPS  # noqa: PLW0603
    saved = O3_STEPS
    try:
        O3_STEPS = 1
        _, _, history, x_m, v_m, (_, got_b) = _o3_setup(collide_first=False)
        _, beta_out = _o3_backward(x_m, v_m, history, transport_inside=False)
        _, beta_in = _o3_backward(x_m, v_m, history, transport_inside=True)
        np.testing.assert_allclose(np.asarray(got_b), np.asarray(beta_out), rtol=1e-12)
        assert _o3_mismatch(got_b, beta_in) > 1e-3
    finally:
        O3_STEPS = saved


def test_production_step_moves_before_it_collides():
    """Oracle 3's premise, asserted against the **production** step.

    :func:`test_oracle3_transport_term_sits_outside_B_k` validates the recursion
    against its own faithful 1-D forward pass; on its own it would not notice if
    ``simulate._step`` were reordered, and then the whole oracle would be pinning the
    adjoint of a loop we no longer run.  This closes that gap behaviourally rather
    than by reading the source: under ``move -> collide`` the new positions are built
    from the **pre**-collision velocities, so

        x_1 == wrap(x_0 + dt * v_0[:, :D])

    exactly.  Under ``collide -> move`` they would use ``v_1`` instead and this fails.
    """
    from jax_dsmc import sampling, simulate  # noqa: PLC0415
    from jax_dsmc.boundary import BC  # noqa: PLC0415
    from jax_dsmc.collide import make_prefactor, vremax_init  # noqa: PLC0415
    from jax_dsmc.grid import num_cells  # noqa: PLC0415
    from jax_dsmc.state import (  # noqa: PLC0415
        CollideState,
        GridSpec,
        ParticleState,
        SimParams,
        SimStatic,
    )

    n, N, temp, dt = (4,), 2000, 300.0, 1e-6
    sp = _params(omega=1.0, alpha=1.0)
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.full(1, 1e-4), n=n)
    C = num_cells(grid)
    static = SimStatic(N=N, D=1, S=1, G=1, A_max=400, K_bounce=3, n_steps=1)
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.full((C,), 1e12), epsilon=jnp.float64(0.0))
    key = random.PRNGKey(0)
    x0 = sampling.sample_positions(random.fold_in(key, 1), N, grid)
    v0 = sampling.sample_maxwellian(
        random.fold_in(key, 2), N, jnp.float64(temp), sp.mass[0], jnp.zeros(3)
    )
    state = ParticleState(
        x=x0,
        v=v0,
        species=jnp.zeros(N, jnp.int32),
        alive=jnp.ones(N, bool),
        logh_acc=jnp.zeros(N),
    )
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((C,), temp),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            make_prefactor(sp.diam, sp.omega, sp.tref, sp.mr)[0, 0],
            C,
            1,
        )
    )

    (new, _), _ = simulate._step(
        (state, cstate),
        0,
        grid,
        sp,
        params,
        static,
        key,
        BC.PERIODIC,
        False,
        "none",
        None,
    )

    # the check is vacuous unless collisions actually changed velocities this step
    assert float(jnp.max(jnp.abs(new.v - v0))) > 0.0, "no collisions: nothing to discriminate"

    lo, hi = grid.lo[0], grid.hi[0]
    expected = lo + jnp.mod(x0 + dt * v0[:, :1] - lo, hi - lo)
    np.testing.assert_allclose(np.asarray(new.x), np.asarray(expected), rtol=1e-13, atol=0.0)
