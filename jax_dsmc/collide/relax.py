"""Relaxation collision operators for coarse cells: a moment-exact ES-Fokker-Planck step
(branch ``neural-collision``; a ``simulate.run(collide_fn=)`` replacement for NTC).

Within each collision cell every particle's peculiar velocity ``c = v - u_c`` takes one
Gaussian Ornstein-Uhlenbeck step (the ES-Fokker-Planck form, Mathiaud & Mieussens 2016)::

    c' = a c + chol(Sigma) xi,
    a = exp(-r_q dt / 3),
    Sigma = theta (1 - a^2) I + (exp(-r_s dt) - a^2) Pi_dev,

``Pi`` the cell's kinetic tensor (``sum c c^T / n``, per unit mass), ``theta = tr Pi / 3``,
``Pi_dev = Pi - theta I``.  Then ``Pi' = a^2 Pi + Sigma = theta I + exp(-r_s dt) Pi_dev`` and
the heat flux ``q' = a^3 q = exp(-r_q dt) q``: the stress and heat flux relax at exactly
``r_s`` and ``r_q`` over any step, so the step is stable and consistent at any ``dt``.
``r_s = 1 / tau_s = p / mu`` with the gas's VSS viscosity ``mu(T)`` (Bird 4.64) gives the
viscosity, ``r_q = Pr r_s`` with ``Pr = 2/3`` the monatomic Prandtl number (the conductivity).

A *frozen-coefficient* ES-FP step (``D = (1 - nu) theta I + nu Pi`` held over the step) is
not usable here: its stress factor ``a^2 + (1 - a^2) nu`` tends to ``nu = -5/4`` at large
``dt`` -- a growing oscillation for ``dt / tau > ln 3`` (2026-10-21, ``tests/test_relax.py``).

``Sigma`` must be positive definite: where a strongly anisotropic cell would break it, the
stress factor is raised toward ``a^2`` just enough (:func:`stress_factor_bound`).  This is the
paper's own limiter (Mathiaud & Mieussens 2016, eq. 35: ``nu = max(-5/4, -RT / (lambda_max -
RT))``, so the diffusion tensor stays positive definite) in exactly integrated form: with
``nu`` clamped the frozen-coefficient step's stress factor is ``a^2 + (1 - a^2) nu = a^2 -
(1 - a^2) theta / pd_max``, the bound here up to :data:`SIGMA_FLOOR`.  It leaves the heat-flux
rate alone and slows the stress relaxation; at ``Pr = 2/3`` it acts once ``pd_max > 0.8
theta`` (:data:`BOUND_ONSET`, the small-``dt`` limit) -- strong shock cores.  The cell's
momentum and kinetic energy are then restored exactly --
``c'`` minus its cell mean, scaled by ``alpha^2 = sum |c|^2 / sum |c'|^2``.

``theta`` (the run's ``collision_params``) holds the traced coefficients: ``log_tau`` (a log
multiplier of ``tau_s``) and ``log_pr`` (``log Pr``); :data:`BASELINE` is the physical model.
A cell with fewer than two particles is left unchanged.  Noise: stage ``STAGE_LEARNED``, one
normal triple per slot (:func:`jax_dsmc.rng.indexed_normal`), so the rematerialised backward
replays it.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.collide.ntc import replacement_diagnostics
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_volume, num_cells
from jax_dsmc.rng import STAGE_LEARNED, indexed_normal, indexed_uniform, key_for

PRANDTL = 2.0 / 3.0
"""The monatomic Prandtl number: the heat-flux to stress relaxation-rate ratio."""

BOUND_ONSET = 1.0 / (3.0 / (2.0 * PRANDTL) - 1.0)
"""``pd_max / theta`` (``pd_max`` the largest eigenvalue of ``Pi_dev``) above which the
positive-definiteness bound (:func:`stress_factor_bound`) acts at the base ``Pr = 2/3`` and
small ``dt``: ES-FP's ``nu = 1 - 3 / (2 Pr) = -5/4`` needs ``pd_max < theta / |nu| = 0.8
theta`` (Mathiaud & Mieussens 2016, eqs. 7, 35)."""

SIGMA_FLOOR = 1e-3
"""Smallest eigenvalue of ``Sigma / (theta (1 - a^2))`` the stress-factor guard keeps."""

BASELINE = {"log_tau": 0.0, "log_pr": math.log(PRANDTL)}
"""The physical coefficients (``collision_params`` of the untrained operator)."""


def vss_viscosity(temp, sp):
    """Chapman-Enskog VSS viscosity of the single species, Bird (4.64) [Pa s]."""
    a, w = sp.alpha[0, 0], sp.omega[0, 0]
    d, tref, m = sp.diam[0, 0], sp.tref[0, 0], sp.mass[0]
    pref = (
        5.0
        * (a + 1.0)
        * (a + 2.0)
        * jnp.sqrt(math.pi * m * K_BOLTZ * tref)
        / (4.0 * a * (5.0 - 2.0 * w) * (7.0 - 2.0 * w) * math.pi * d * d)
    )
    return pref * (temp / tref) ** w


def cell_sums(v, alive, seg, n_cells):
    """Per-cell ``(n, sum v, sum v v^T)`` of the alive particles (``seg`` the cell, dead
    slots at the sentinel ``n_cells``): ``f64[C]``, ``f64[C, 3]``, ``f64[C, 3, 3]``."""
    w = alive.astype(v.dtype)
    n = jax.ops.segment_sum(w, seg, n_cells + 1)[:n_cells]
    s1 = jax.ops.segment_sum(w[:, None] * v, seg, n_cells + 1)[:n_cells]
    vv = w[:, None, None] * v[:, :, None] * v[:, None, :]
    s2 = jax.ops.segment_sum(vv, seg, n_cells + 1)[:n_cells]
    return n, s1, s2


def stress_factor_bound(e_s, a2, theta, pd_max):
    """The stress factor ``e_s`` raised where ``Sigma = theta (1 - a^2) I + (e_s - a^2)
    Pi_dev`` would lose positive definiteness: with ``e_s < a^2`` its smallest eigenvalue is
    ``theta (1 - a^2) - (a^2 - e_s) pd_max`` (``pd_max`` the largest eigenvalue of
    ``Pi_dev``), kept at ``SIGMA_FLOOR theta (1 - a^2)`` or above.

    The gradient flows through the bound (``a^2``, ``theta`` and ``pd_max``): where it is
    active, ``Sigma``'s smallest eigenvalue is ``SIGMA_FLOOR theta (1 - a^2)`` exactly, and
    the Cholesky factor's derivative divides by its square root.  A stop-gradient bound gave
    that eigenvalue a derivative ~``(1 + pd_max / theta) / SIGMA_FLOOR`` times too large
    (AD 5518 against FD 9.5 for one particle, 2026-11-05); cells hitting the bound (5 % at 20
    particles per cell) then broke every window gradient longer than a few steps."""
    room = (1.0 - SIGMA_FLOOR) * theta * (1.0 - a2) / jnp.maximum(pd_max, 1e-300)
    return jnp.where(pd_max > 0.0, jnp.maximum(e_s, a2 - room), e_s)


def largest_deviatoric(pi_dev, theta, n):
    """``f64[C]``: the largest eigenvalue of ``Pi_dev``, with a gradient everywhere.

    ``eigvalsh``'s derivative is NaN wherever two eigenvalues coincide (its eigenvector term
    divides by their gap, and a zero cotangent times that infinity is NaN), and a cell's
    ``Pi_dev`` is degenerate exactly when it holds fewer than three particles: two give
    ``(2 theta, -theta, -theta)``.  So two-particle cells take that closed form, cells with
    fewer than two (where ``Sigma`` is the identity anyway) a fixed value, and only cells of
    three or more -- generically nondegenerate -- reach ``eigvalsh``, on a safe matrix
    elsewhere."""
    many = n >= 3.0
    safe = jnp.diag(jnp.array([1.0, 0.0, -1.0]))[None] * theta[:, None, None]
    top = jnp.linalg.eigvalsh(jnp.where(many[:, None, None], pi_dev, safe))[:, -1]
    return jnp.where(many, top, 2.0 * theta)


def cell_moments(v, alive, seg, n_cells):
    """``(n, u, Pi, theta)`` per cell: count, mean velocity ``f64[C, 3]``, kinetic tensor
    ``f64[C, 3, 3]`` about the cell mean (per unit mass) and ``theta = tr Pi / 3``."""
    n, s1, s2 = cell_sums(v, alive, seg, n_cells)
    n_safe = jnp.maximum(n, 1.0)
    u = s1 / n_safe[:, None]
    pi = s2 / n_safe[:, None, None] - u[:, :, None] * u[:, None, :]
    theta_c = jnp.maximum(jnp.trace(pi, axis1=1, axis2=2) / 3.0, 1e-300)
    return n, u, pi, theta_c


def safe_cells(n, pi, theta_c):
    """``(live, n, pi, theta_c)`` with the cells of fewer than two particles given safe inputs
    -- the mean ``theta`` of the live cells, an isotropic ``Pi`` and ``n = 2`` -- *before*
    anything is computed from them, not just masked after: a masked overflow (``tau_s =
    mu(T -> 0) / p``) still sends ``0 * inf = NaN`` back through the ``where`` (the NaN window
    gradients of 2026-11-05, at about 10 particles per cell)."""
    live = n >= 2.0
    theta_ref = jax.lax.stop_gradient(
        jnp.sum(jnp.where(live, theta_c, 0.0)) / jnp.maximum(jnp.sum(live), 1.0)
    )
    theta_c = jnp.where(live, theta_c, theta_ref)
    pi = jnp.where(live[:, None, None], pi, theta_c[:, None, None] * jnp.eye(3))
    return live, jnp.where(live, n, 2.0), pi, theta_c


def stress_time(n, theta_c, grid, sp, params):
    """``f64[C]``: the stress relaxation time ``tau_s = mu(T) / p`` of each cell's state."""
    temp = sp.mass[0] * theta_c / K_BOLTZ
    pressure = n * params.fnum / cell_volume(grid) * K_BOLTZ * temp
    return vss_viscosity(temp, sp) / jnp.maximum(pressure, 1e-300)


def rates(n, theta_c, grid, sp, params, theta, splitting, delta=None):
    """``(r_s, r_q)`` (``f64[C]``): each cell's stress and heat-flux relaxation rates --
    ``r_s = exp(-log_tau) / tau_s``, ``r_q = Pr r_s`` -- times ``exp(delta[0])`` and
    ``exp(delta[1])`` with a learned per-cell correction ``delta`` (:func:`correction`; the
    *rate basis*, 2026-11-12: one row per transport coefficient, so a correction of the heat
    flux reads as one).  ``n``, ``theta_c``: safe cells (:func:`safe_cells`)."""
    tau_s = stress_time(n, theta_c, grid, sp, params)
    if splitting:
        tau_s = jnp.maximum(tau_s - 0.5 * params.dt, 1e-3 * tau_s)
    r_s = jnp.exp(-theta["log_tau"]) / tau_s
    r_q = jnp.exp(theta["log_pr"]) * r_s
    if delta is not None:
        r_s, r_q = r_s * jnp.exp(delta[0]), r_q * jnp.exp(delta[1])
    return r_s, r_q


def stress_factors(n, pi, theta_c, grid, sp, params, theta, splitting, delta=None):
    """``(live, theta_c, pi_dev, a, e_s, e_bound)`` of the step: the drift factor ``a``, the
    stress factor ``exp(-r_s dt)`` and the same after :func:`stress_factor_bound` (equal where
    the bound is inactive), on safe cells."""
    live, n, pi, theta_c = safe_cells(n, pi, theta_c)
    r_s, r_q = rates(n, theta_c, grid, sp, params, theta, splitting, delta)
    a = jnp.exp(-r_q * params.dt / 3.0)
    pi_dev = pi - theta_c[:, None, None] * jnp.eye(3)
    e_s = jnp.exp(-r_s * params.dt)
    pd_max = largest_deviatoric(pi_dev, theta_c, n)
    return live, theta_c, pi_dev, a, e_s, stress_factor_bound(e_s, a**2, theta_c, pd_max)


def step_coefficients(n, pi, theta_c, grid, sp, params, theta, splitting, delta=None):
    """The step's drift factor ``a`` (``f64[C]``) and noise factor ``chol(Sigma)``
    (``f64[C, 3, 3]``; the identity in cells with fewer than two particles).  ``delta``: an
    optional per-cell ``(dlog r_s, dlog r_q)`` (``f64[C]`` each, :func:`rates`) -- a learned
    correction (:func:`correction`)."""
    live, theta_c, pi_dev, a, _, e_s = stress_factors(n, pi, theta_c, grid, sp, params, theta,
                                                      splitting, delta)  # fmt: skip
    a2 = a**2
    eye = jnp.eye(3)
    cov = (theta_c * (1.0 - a2))[:, None, None] * eye + (e_s - a2)[:, None, None] * pi_dev
    cov = cov + 1e-12 * theta_c[:, None, None] * eye
    cov = jnp.where(live[:, None, None], cov, eye)
    return a, jnp.linalg.cholesky(cov)


def bound_activity(n, pi, theta_c, grid, sp, params, theta, splitting=False):
    """Per cell ``(pd_ratio, slowdown)``: ``pd_max / theta`` and ``log(e_bound / e_s)`` (> 0
    where :func:`stress_factor_bound` acts: the stress relaxes slower than ``r_s`` by that
    much per step), at ``theta`` without a correction -- the ``lcbound`` diagnostic."""
    live, theta_c, pi_dev, _, e_s, e_b = stress_factors(n, pi, theta_c, grid, sp, params, theta,
                                                        splitting)  # fmt: skip
    n_s = jnp.where(live, n, 2.0)
    ratio = largest_deviatoric(pi_dev, theta_c, n_s) / theta_c
    slow = jnp.log(jnp.maximum(e_b, 1e-300)) - jnp.log(jnp.maximum(e_s, 1e-300))
    return jnp.where(live, ratio, 0.0), jnp.where(live, slow, 0.0)


def conserve(c, c_new, alive, seg, row, n, n_cells):
    """``(alpha c_new - mean, ok)``: ``c_new`` made zero-mean in its cell and scaled to the
    cell's old kinetic energy ``sum |c|^2`` -- exact momentum and energy conservation.  ``ok``
    (``bool[C]``) marks the cells that were updated (two or more particles, nonzero energy)."""
    _, m1, _ = cell_sums(c_new, alive, seg, n_cells)
    c_new = c_new - (m1 / jnp.maximum(n, 1.0)[:, None])[row]
    e_old = jax.ops.segment_sum(jnp.where(alive, jnp.sum(c**2, axis=1), 0.0), seg, n_cells + 1)
    e_new = jax.ops.segment_sum(jnp.where(alive, jnp.sum(c_new**2, axis=1), 0.0), seg,
                                n_cells + 1)  # fmt: skip
    ok = (n >= 2.0) & (e_new[:n_cells] > 0.0)
    alpha = jnp.sqrt(jnp.where(ok, e_old[:n_cells], 1.0) / jnp.where(ok, e_new[:n_cells], 1.0))
    return alpha[row][:, None] * c_new, ok


def neighbour_rows(n, periodic):
    """Host tables of the uniform grid ``n`` (STATIC): ``minus``, ``plus`` (``i32[C, D]``, the
    neighbouring cell on each side of each axis) and ``span`` (``f64[C, D]``, cells between
    them: 2, or 1 where a non-periodic box face makes the difference one-sided)."""
    shape = tuple(int(k) for k in n)
    multi = np.stack(np.unravel_index(np.arange(int(np.prod(shape))), shape), axis=-1)
    minus = np.empty_like(multi[:, 0:1].repeat(len(shape), axis=1))
    plus = np.empty_like(minus)
    span = np.full(minus.shape, 2.0)
    for axis, size in enumerate(shape):
        lo, hi = multi.copy(), multi.copy()
        lo[:, axis] -= 1
        hi[:, axis] += 1
        if periodic[axis]:
            lo[:, axis] %= size
            hi[:, axis] %= size
        else:
            span[:, axis] -= (lo[:, axis] < 0) + (hi[:, axis] >= size)
            lo[:, axis] = np.maximum(lo[:, axis], 0)
            hi[:, axis] = np.minimum(hi[:, axis], size - 1)
        minus[:, axis] = np.ravel_multi_index(tuple(lo.T), shape)
        plus[:, axis] = np.ravel_multi_index(tuple(hi.T), shape)
    return minus, plus, np.maximum(span, 1.0)


def cell_gradient(f, n, grid, periodic):
    """``f64[C, K, D]``: the central difference of a per-cell field ``f[C, K]`` over the
    neighbouring cells (one-sided at a non-periodic face; an empty neighbour contributes the
    cell's own value).  Uniform grids without split cells only."""
    if int(grid.n_sub):
        raise ValueError("cell gradients need a grid without split cells")
    minus, plus, span = neighbour_rows(grid.n, periodic)
    dx = (grid.hi - grid.lo) / jnp.asarray(grid.n)
    filled = n > 0.0
    f_m = jnp.where(filled[minus][..., None], f[minus], f[:, None, :])  # [C, D, K]
    f_p = jnp.where(filled[plus][..., None], f[plus], f[:, None, :])
    grad = (f_p - f_m) / (jnp.asarray(span) * dx)[..., None]  # [C, D, K]
    return jnp.swapaxes(grad, 1, 2)


def velocity_gradient(u, n, grid, periodic):
    """``f64[C, 3, D]``: :func:`cell_gradient` of the cell-mean velocity."""
    return cell_gradient(u, n, grid, periodic)


def conserve_about(c, c_new, u_loc, v, alive, seg, row, n, n_cells):
    """``(alpha c_new', ok)`` with ``c_new'`` zero-mean in its cell and ``alpha`` solving
    ``sum |u_loc + alpha c_new'|^2 = sum |v|^2``: exact energy about a *reconstructed* local
    velocity (momentum is exact because ``u_loc`` averages to the cell mean).  Falls back to
    no update (``ok`` false) where the quadratic has no positive root."""
    _, m1, _ = cell_sums(c_new, alive, seg, n_cells)
    c_new = c_new - (m1 / jnp.maximum(n, 1.0)[:, None])[row]
    w = alive.astype(v.dtype)
    s_cc = jax.ops.segment_sum(w * jnp.sum(c_new**2, axis=1), seg, n_cells + 1)[:n_cells]
    s_uc = jax.ops.segment_sum(w * jnp.sum(u_loc * c_new, axis=1), seg, n_cells + 1)[:n_cells]
    k = jax.ops.segment_sum(w * (jnp.sum(v**2, axis=1) - jnp.sum(u_loc**2, axis=1)), seg,
                            n_cells + 1)[:n_cells]  # fmt: skip
    disc = s_uc**2 + s_cc * k
    ok = (n >= 2.0) & (s_cc > 0.0) & (disc > 0.0)
    s_cc_safe = jnp.where(ok, s_cc, 1.0)
    alpha = (-s_uc + jnp.sqrt(jnp.where(ok, disc, 1.0))) / s_cc_safe
    ok = ok & (alpha > 0.0)
    return jnp.where(ok, alpha, 1.0)[row][:, None] * c_new, ok


FEATURES = ("dx2", "dx4", "dt2", "dt1", "inv_n", "dx2_inv_n", "kn_u2", "kn_t2", "dx2_kn2", "bound")
"""The theory basis of the learned corrections (:func:`cell_features`), with ``r = h /
lambda`` the cell size in local mean free paths:

- ``r^2``, ``r^4``: the cell-size error of uniform in-cell collisions and its next order
  (Alexander, Garcia & Alder 1998: ``mu_eff / mu - 1 = (16 / 45 pi) r^2``);
- ``(dt / tau)^2``, ``dt / tau``: the time-step error and any residual first-order splitting;
- ``1 / N_c``, ``r^2 / N_c``: finite-particle biases of the cell state, alone and in the
  cell-size term;
- ``Kn_u^2 = (lambda |S| / v)^2``, ``Kn_T^2 = (lambda |grad ln T|)^2``: Burnett-order
  (Chapman-Enskog O(Kn^2)) corrections to the stress and heat-flux closures;
- ``r^2 (Kn_u^2 + Kn_T^2)``: the cell-size error growing with the local gradients (the
  reconstruction is only linear);
- ``max(0, pd_max / theta - 0.8)``: how far the cell's anisotropy is past ES-FP's
  positive-definiteness limit (:data:`BOUND_ONSET`), where the base step's stress relaxation
  is slowed (:func:`stress_factor_bound`) -- a defect of the Gaussian model at ``Pr = 2/3``,
  not of the coarse cell (2026-11-12: the bound couples the stress rate to the heat-flux rate
  there, which ``log_pr``-only corrections traded heat flux for)."""

FEATURE_SCALE = np.array([4.0, 16.0, 0.01, 0.1, 0.1, 0.4, 0.01, 0.01, 0.04, 0.2])
"""Nominal size of each feature at the training conditions (``r = 2``, ``dt / tau = 0.1``,
``N_c = 10``, ``Kn ~ 0.1``, a shock core ``pd_max ~ theta``): the features are divided by it so the coefficients are O(1)."""

CORRECTION_BOUND = 2.0
"""Bound on each correction (``|dlog| < 2``, a factor e^2 on ``r_s`` or ``r_q``): the raw
linear or MLP output is squashed by ``B tanh(x / B)``, so an input far outside the training
range (the circle's cold wall layer, ``r ~ 30``) cannot drive ``tau`` to 0 or infinity."""

CORRECTIONS = ("linear", "mlp")


def lambda_factor(omega):
    """``K`` in ``lambda = K tau_s sqrt(kT / m)``: Bird's VSS mean free path (4.52) in terms of
    the Chapman-Enskog viscosity (4.64), ``K = 4 (5 - 2w)(7 - 2w) / (30 sqrt(2 pi))``."""
    return 4.0 * (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega) / (30.0 * math.sqrt(2.0 * math.pi))


def cell_features(n, n_safe, u, pi, theta_c, grid, sp, params, periodic):
    """``f64[C, 10]``: the :data:`FEATURES` of each cell, divided by :data:`FEATURE_SCALE`.
    ``n`` the particle counts (empty neighbours drop out of the gradients), ``n_safe``,
    ``pi`` and ``theta_c`` the safe ones (:func:`safe_cells`); ``u`` the cell means (the
    gradients come from neighbouring cells, so a coarse cell sees its own resolved shear)."""
    tau = stress_time(n_safe, theta_c, grid, sp, params)
    lam = lambda_factor(sp.omega[0, 0]) * tau * jnp.sqrt(theta_c)
    dx = (grid.hi - grid.lo) / jnp.asarray(grid.n)
    h = jnp.prod(dx) ** (1.0 / dx.shape[0])
    r2 = (h / lam) ** 2
    dim = dx.shape[0]
    g = jnp.zeros((n.shape[0], 3, 3)).at[:, :, :dim].set(cell_gradient(u, n, grid, periodic))
    strain = 0.5 * (g + jnp.swapaxes(g, 1, 2))
    strain = strain - jnp.trace(strain, axis1=1, axis2=2)[:, None, None] * jnp.eye(3) / 3.0
    kn_u2 = lam**2 * jnp.sum(strain**2, axis=(1, 2)) / theta_c
    g_t = cell_gradient(jnp.log(theta_c)[:, None], n, grid, periodic)[:, 0, :]
    kn_t2 = lam**2 * jnp.sum(g_t**2, axis=1)
    dt_tau = params.dt / tau
    inv_n = 1.0 / n_safe
    pi_dev = pi - theta_c[:, None, None] * jnp.eye(3)
    over = jax.nn.relu(largest_deviatoric(pi_dev, theta_c, n_safe) / theta_c - BOUND_ONSET)
    phi = jnp.stack([r2, r2**2, dt_tau**2, dt_tau, inv_n, r2 * inv_n, kn_u2, kn_t2,
                     r2 * (kn_u2 + kn_t2), over], axis=1)  # fmt: skip
    # a cell with no volume (inside a body) has tau = 0: its features are infinite, and a
    # zero coefficient times infinity would be NaN -- cap them (the bounded correction
    # saturates there anyway, and the cell holds no particle)
    return jnp.minimum(jnp.nan_to_num(phi, nan=0.0, posinf=1e6), 1e6) / jnp.asarray(FEATURE_SCALE)


def _standardised(theta, phi):
    """The features over ``theta["phi_scale"]`` (their spread over the training cells,
    fixed: no gradient), when the parameters carry one.  A scale only, no centring: a
    feature that is zero (a resolved, equilibrium cell) stays zero, so a correction without
    an intercept vanishes there -- and an L1 penalty on the coefficients then weighs every
    term by its observed spread, not by :data:`FEATURE_SCALE`'s nominal guess."""
    if "phi_scale" not in theta:
        return phi
    return phi / jax.lax.stop_gradient(theta["phi_scale"])


def correction_linear(theta, phi):
    """``f64[2, C]``: the raw linear correction ``A phi`` (``theta["A"]``: ``f64[2, F]``)."""
    return theta["A"] @ _standardised(theta, phi).T


def mlp_layers(theta):
    """The number of weight layers of an MLP parameter set (``W0`` ... ``W{L-1}``)."""
    return sum(1 for k in theta if k.startswith("W"))


def _mlp(theta, x):
    layers = mlp_layers(theta)
    for i in range(layers - 1):
        x = jnp.tanh(x @ theta[f"W{i}"] + theta[f"b{i}"])
    return x @ theta[f"W{layers - 1}"] + theta[f"b{layers - 1}"]


def correction_mlp(theta, phi):
    """``f64[2, C]``: a tanh MLP of the standardised features, **minus its value at zero
    features** -- no correction in a resolved, equilibrium cell (the fine-cell limit gives
    the base operator back whatever the weights), only the finite-particle and gradient
    terms there."""
    x = _standardised(theta, phi)
    return (_mlp(theta, x) - _mlp(theta, jnp.zeros((1, x.shape[1])))).T


def correction(theta, phi, kind):
    """The bounded per-cell ``(dlog r_s, dlog r_q)`` of correction ``kind`` (:func:`rates`)."""
    raw = correction_linear(theta, phi) if kind == "linear" else correction_mlp(theta, phi)
    return CORRECTION_BOUND * jnp.tanh(raw / CORRECTION_BOUND)


def init_correction(kind, key=None, width=64, depth=2, phi_scale=None):
    """The trainable parameters of correction ``kind`` added to :data:`BASELINE`, zero output
    at initialisation (so the corrected operator starts as the base, bitwise).  ``mlp``:
    ``depth`` hidden tanh layers of ``width`` (``1 / sqrt(fan-in)`` weights, a zero last
    layer).  ``phi_scale`` (``f64[F]``): the features' training spread (:func:`_standardised`)."""
    theta = {k: jnp.float64(v) for k, v in BASELINE.items()}
    k = len(FEATURES)
    if kind == "linear":
        theta["A"] = jnp.zeros((2, k))
    elif kind == "mlp":
        sizes = [k] + [width] * depth + [2]
        keys = jax.random.split(jax.random.PRNGKey(0) if key is None else key, depth + 1)
        for i in range(depth + 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            last = i == depth
            theta[f"W{i}"] = (
                jnp.zeros((fan_in, fan_out))
                if last
                else jax.random.normal(keys[i], (fan_in, fan_out)) / math.sqrt(fan_in)
            )
            theta[f"b{i}"] = jnp.zeros(fan_out)
    elif kind is not None:
        raise ValueError(f"unknown correction {kind}")
    if kind is not None and phi_scale is not None:
        theta["phi_scale"] = jnp.asarray(phi_scale, jnp.float64)
    return theta


def local_state(state, table, grid, periodic):
    """The per-cell state an operator relaxes toward: ``(seg, row, n, u, u_loc, pi,
    theta_c)``.  With ``periodic`` (a tuple of bools per axis) the target velocity is the
    reconstructed ``u(x) = u_c + G_c (x - xbar_c)`` and ``Pi`` is taken about it; with
    ``None`` the cell mean."""
    n_cells = num_cells(grid)
    v, alive = state.v, state.alive
    seg = jnp.where(alive, jnp.minimum(table.cell, n_cells), n_cells)
    row = jnp.minimum(seg, n_cells - 1)  # a gather index for every slot (dead ones masked)
    n, u, pi, theta_c = cell_moments(v, alive, seg, n_cells)
    if periodic is None:
        return seg, row, n, u, u[row], pi, theta_c
    dim = state.x.shape[1]
    n_x, xs, _ = cell_sums(jnp.pad(state.x, ((0, 0), (0, 3 - dim))), alive, seg, n_cells)
    xbar = (xs / jnp.maximum(n_x, 1.0)[:, None])[:, :dim]
    grad = velocity_gradient(u, n, grid, periodic)
    u_loc = u[row] + jnp.einsum("nij,nj->ni", grad[row], state.x - xbar[row])
    # the kinetic tensor about the reconstructed velocity: no in-cell shear in it
    _, _, s2 = cell_sums(v - u_loc, alive, seg, n_cells)
    pi = s2 / jnp.maximum(n, 1.0)[:, None, None]
    theta_c = jnp.maximum(jnp.trace(pi, axis1=1, axis2=2) / 3.0, 1e-300)
    return seg, row, n, u, u_loc, pi, theta_c


def _finish(state, table, cstate, c, c_new, u_loc, seg, row, n, periodic):
    """Restore the cell's momentum and energy (:func:`conserve` / :func:`conserve_about`) and
    write the new velocities of the updated cells."""
    n_cells = n.shape[0]
    v, alive = state.v, state.alive
    if periodic is None:
        c_new, ok = conserve(c, c_new, alive, seg, row, n, n_cells)
    else:
        c_new, ok = conserve_about(c, c_new, u_loc, v, alive, seg, row, n, n_cells)
    moved = alive & ok[row]
    out = dataclasses.replace(state, v=jnp.where(moved[:, None], u_loc + c_new, v))
    return out, cstate, replacement_diagnostics(state, table)


def collide(state, table, cstate, grid, sp, params, static, cell_temp, step, base_key, theta,
            *, splitting=False, periodic=None, corrected=None):  # fmt: skip
    """One moment-exact ES-FP step over every cell (module docstring).

    STATIC options:
    - ``splitting=True`` shortens the stress relaxation time by ``dt / 2``, the first-order
      transport/collision splitting correction (as lattice Boltzmann's ``tau - 1/2``).
    - ``periodic`` (a tuple of bools per axis) turns on the **position-aware** target: each
      particle relaxes toward ``u(x) = u_c + G_c (x - xbar_c)`` (``G_c`` the cells' velocity
      gradient, :func:`velocity_gradient`; ``xbar_c`` the cell's particle centroid, so the
      cell momentum is unchanged) instead of the cell mean -- what a coarse cell needs to
      stop relaxing every particle toward one velocity (the O((dx / lambda)^2) excess
      viscosity of uniform in-cell collisions).  ``None``: the cell-mean target.
    - ``corrected`` (``"linear"`` / ``"mlp"``, needs ``periodic``): a learned per-cell
      ``(dlog r_s, dlog r_q)`` from the theory features (:func:`cell_features`,
      :func:`correction`; parameters from :func:`init_correction`).
    """
    del static, cell_temp
    v = state.v
    seg, row, n, u, u_loc, pi, theta_c = local_state(state, table, grid, periodic)
    delta = None
    if corrected is not None:
        if periodic is None:
            raise ValueError("a corrected operator needs periodic= (its features use gradients)")
        _, n_s, pi_s, theta_s = safe_cells(n, pi, theta_c)
        phi = cell_features(n, n_s, u, pi_s, theta_s, grid, sp, params, periodic)
        delta = correction(theta, phi, corrected)
    a, chol = step_coefficients(n, pi, theta_c, grid, sp, params, theta, splitting, delta)
    xi = indexed_normal(key_for(base_key, step, STAGE_LEARNED), v.shape[0], (3,))
    c = v - u_loc
    c_new = a[row][:, None] * c + jnp.einsum("nij,nj->ni", chol[row], xi)
    return _finish(state, table, cstate, c, c_new, u_loc, seg, row, n, periodic)


def bgk_collide(state, table, cstate, grid, sp, params, static, cell_temp, step, base_key,
                theta, *, es=True, periodic=None):  # fmt: skip
    """One particle BGK (``es=False``, ``Pr = 1``) or ES-BGK (Holway; ``Pr = exp(log_pr)``,
    2/3 at :data:`BASELINE`) step: each particle relaxes with probability ``1 - exp(-nu dt)``
    -- ``nu = Pr / tau_s`` (the stress then relaxes at ``1 / tau_s``, the heat flux at
    ``Pr / tau_s``) -- to a fresh draw from ``N(u(x_i), Lambda)``, ``Lambda = theta I + b
    Pi_dev`` with ``b = 1 - 1 / Pr`` (``Lambda = theta I`` for BGK); then the cell's momentum
    and energy are restored over all its particles as in :func:`collide`.  ``periodic`` as
    there (the reconstructed target).

    A forward baseline: whether a particle relaxes is a discrete draw, so the pathwise
    gradient misses ``tau``'s effect on it -- train through :func:`collide` instead."""
    del static, cell_temp
    v = state.v
    seg, row, n, _, u_loc, pi, theta_c = local_state(state, table, grid, periodic)
    live, n_s, pi_s, theta_s = safe_cells(n, pi, theta_c)
    tau_s = jnp.exp(theta["log_tau"]) * stress_time(n_s, theta_s, grid, sp, params)
    pr = jnp.exp(theta["log_pr"]) if es else jnp.float64(1.0)
    p_relax = 1.0 - jnp.exp(-pr * params.dt / tau_s)
    eye = jnp.eye(3)
    pi_dev = pi_s - theta_s[:, None, None] * eye
    b = jnp.broadcast_to(1.0 - 1.0 / pr, theta_s.shape)
    pd_max = largest_deviatoric(pi_dev, theta_s, n_s)
    b = jnp.maximum(b, -(1.0 - SIGMA_FLOOR) * theta_s / jnp.maximum(pd_max, 1e-300))
    cov = theta_s[:, None, None] * eye + b[:, None, None] * pi_dev
    cov = jnp.where(live[:, None, None], cov + 1e-12 * theta_s[:, None, None] * eye, eye)
    chol = jnp.linalg.cholesky(cov)
    key = key_for(base_key, step, STAGE_LEARNED)
    xi = indexed_normal(key, v.shape[0], (3,))
    draw = indexed_uniform(jax.random.fold_in(key, 1), v.shape[0])
    c = v - u_loc
    relax = (draw < p_relax[row])[:, None]
    c_new = jnp.where(relax, jnp.einsum("nij,nj->ni", chol[row], xi), c)
    return _finish(state, table, cstate, c, c_new, u_loc, seg, row, n, periodic)
