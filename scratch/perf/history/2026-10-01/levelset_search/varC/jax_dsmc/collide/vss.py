"""Variable Soft Sphere collision model (spec section 7).

Three deliberate deviations from SPARTA's ``collide_vss.cpp``, each documented at
its site:

1. the ``vr2`` clamp is **relative** to the pair's reference thermal energy, not
   the absolute ``1e-30`` a naive translation would use (section 2.3);
2. the frame-degeneracy threshold is **relative** to ``|vr|``, where SPARTA's
   ``if (d > 1.0e-6)`` is dimensional and unit-system dependent (section 7.4);
3. SPARTA's ``fabs(alpha_r - 1.0) < 0.001`` fast path is dropped, because
   ``alpha`` is a traced parameter and a Python branch on it is illegal.

Every stochastic branch here exposes a differentiable ``log h`` -- that is what
makes the score half of the adjoint (section 11) obtainable from AD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, random

from jax_dsmc.constants import EPS_D, EPS_REL_VR2, EPS_U, K_BOLTZ
from jax_dsmc.safe import safe_log, safe_norm, safe_pow, safe_sqrt
from jax_dsmc.state import SpeciesParams

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def make_prefactor(diam, omega, tref, mr):
    """VSS cross-section prefactor, ``f64[S, S]``.  TRACED in all four arguments.

    ``collide_vss.cpp:122-125``::

        cxs * (2 k T_ref / m_r)^(omega - 0.5) / Gamma(5/2 - omega)

    so that ``sigma_T * v_r = prefactor * (v_r^2)^(1 - omega)``.

    Traced in ``diam``, ``tref`` and ``omega``, so ``dJ/d(VSS params)`` has a path
    here and a second one through the deflection law in :func:`scatter`.  This is
    the model-calibration gradient.

    Spec: section 7.1.
    """
    cxs = jnp.pi * diam * diam
    return (
        cxs
        * (2.0 * K_BOLTZ * tref / mr) ** (omega - 0.5)
        / jnp.exp(jax.scipy.special.gammaln(2.5 - omega))
    )


def vr2_clamp(sp: SpeciesParams, isp, jsp):
    """Absolute clamp handed to :func:`safe_pow`, scaled to the pair's thermal energy.

    ``EPS_REL_VR2 * (2 k T_ref / m_r)`` -- a fixed fraction of the reference
    thermal speed squared, **not** an absolute ``1e-30``.  Section 2.3 measures
    that the absolute form leaves ``dq/d|vr|`` unbounded at ``~2.5e16``; U1
    (section 17.4) confirms this relative form makes it exactly
    ``2|1-omega| eps^(0.5-omega)`` and settles the constant at ``1e-8``.
    """
    return EPS_REL_VR2 * (2.0 * K_BOLTZ * sp.tref[isp, jsp] / sp.mr[isp, jsp])


def collision_rate(vi, vj, isp, jsp, sp: SpeciesParams):
    """``q = sigma_T(v_r) * v_r``, what SPARTA calls ``vre``.  TRACED.

    Args:
        vi, vj: ``f64[..., 3]`` pre-collision velocities.
        isp, jsp: integer species indices, broadcastable against the batch.
        sp: ``SpeciesParams``.

    Returns:
        ``f64[...]``.

    ``collide_vss.cpp:200-232``.  ``omega == 1`` (Maxwell molecules, the M1 BKW
    gate) makes this exactly ``prefactor``, independent of ``vr`` -- and hence
    ``dq/dv == 0`` identically, which is why the clamp constant cannot bind at M1.
    """
    vr = vi - vj
    vr2 = jnp.sum(vr * vr, axis=-1)
    return sp.prefactor[isp, jsp] * safe_pow(vr2, 1.0 - sp.omega[isp, jsp], vr2_clamp(sp, isp, jsp))


def acceptance_prob(q, sigma):
    """NTC acceptance probability ``clip(q / Sigma, 0, 1)``.

    ``Sigma`` is ``stop_gradient``-ed.  Under the stale-max rule (section 7.2 /
    decision 9) it enters the step as carried state from the *previous* step, so a
    cross-step path would otherwise exist.  We cut it because ``Sigma`` is an
    algorithmic device for exact rejection sampling, not physics.

    **The clip is a bias hazard, not a safety net.**  It saturates when
    ``q > Sigma``, and at saturation ``log h = log(1) = 0`` with zero derivative,
    so a violated attempt contributes *literally nothing* and biases the gradient
    (section 7.2.1: 14% low at a 1% violation rate, 75% low at 15%).  The
    ``majorant_violations`` diagnostic counts these, and section 7.2.3's a-priori
    thermal majorant is what keeps the count at zero.
    """
    return jnp.clip(q / lax.stop_gradient(sigma), 0.0, 1.0)


def log_h(q, sigma, accepted):
    """Score contribution of one attempted pair, ``f64[...]``.

    ``log(q/Sigma)`` when the virtual collision became real, ``log(1 - q/Sigma)``
    when it did not.  **Both branches contribute** -- the ``j=3`` "virtual but not
    real" row of section 7.3 is easy to miss, and omitting it gives a subtly biased
    gradient that still passes a naive smoke test.  The Algorithm-3 oracle
    (section 13.2) is what catches it.

    Differentiable in ``vi``, ``vj`` through ``q``, and in ``diam``, ``omega``,
    ``tref`` through ``prefactor``.  Both paths are wanted.
    """
    p = acceptance_prob(q, sigma)
    return jnp.where(accepted, safe_log(p), safe_log(1.0 - p))


def scatter_key(vi, vj, isp, jsp, sp: SpeciesParams, key):
    """:func:`scatter`, drawing its two uniforms from ``key``.

    Convenience wrapper for callers outside the attempt-slot scan (tests, oracles).
    **The scan must not use this**: drawing inside the loop body costs a kernel
    launch per slot, which measured as the dominant cost of a collision step
    (section 8.6).  Section 1 rule 3 requires one batched draw per stage.
    """
    k_chi, k_eps = random.split(key)
    shape = jnp.broadcast_shapes(vi.shape[:-1], vj.shape[:-1])
    return scatter(
        vi,
        vj,
        isp,
        jsp,
        sp,
        random.uniform(k_chi, shape, dtype=jnp.float64),
        random.uniform(k_eps, shape, dtype=jnp.float64),
    )


def frame_is_well_defined(vi, vj):
    """The **frozen branch** of :func:`scatter`: is the ``v_r``-aligned frame usable?

    ``bool[...]``, exactly the ``ok`` of :func:`scatter`.  Exposed as its own
    function for two reasons: :func:`scatter` uses it, so the two can never drift
    apart, and the collision driver folds it into
    ``Diagnostics.decision_checksum``.

    **It has to be in that checksum, and this was found the hard way.**  Crossing
    the threshold swaps the post-collision velocity to the degenerate fallback --
    an ``O(|v_r|)`` jump.  A finite-difference check whose flip guard covered cells,
    attempt counts, pairs, accept bits and the boundary fold still reported "no
    flips" across a jump of 87 m/s in a single pair, because this branch was
    missing from it (``scratch/adjoint/probe_locate.py``).  It is a genuine
    discontinuity of the collision map, not a numerical artefact.
    """
    vr = vi - vj
    d = safe_norm(vr[..., 1:3], axis=-1)
    return d > EPS_D * safe_norm(vr, axis=-1)


def scatter(vi, vj, isp, jsp, sp: SpeciesParams, u_chi, u_eps):
    """VSS post-collision velocities.  ``collide_vss.cpp:405-452``.

    Args:
        vi, vj: ``f64[..., 3]`` pre-collision velocities.
        isp, jsp: integer species indices.
        sp: ``SpeciesParams``.
        u_chi, u_eps: uniforms on ``[0, 1)``, broadcastable against the batch.
            Passed in rather than drawn here so the caller can draw them once for
            every attempt slot (section 1 rule 3) -- and so this function is a pure
            deterministic map, which is what makes it directly differentiable.

    Returns:
        ``(vi_new, vj_new)``, each ``f64[..., 3]``.

    **Conservation is structural, not approximate.**  Momentum: the two offsets are
    mass-weighted and opposite about a shared ``vcm``, so
    ``mi*vi' + mj*vj' == mi*vi + mj*vj`` identically.  Energy: ``|v_r'| == |v_r|``
    by construction, and relative kinetic energy in the CM frame depends only on
    ``|v_r|``.  Both hold to round-off regardless of ``alpha`` or ``omega``.
    """
    mi, mj = sp.mass[isp], sp.mass[jsp]
    a_inv = 1.0 / sp.alpha[isp, jsp]
    eps_az = 2.0 * jnp.pi * u_eps

    # u_chi is a dimensionless uniform on [0,1), so a plain small clamp is right
    # here -- the relative-to-thermal-scale form of `collision_rate` does not apply.
    cos_x = 2.0 * safe_pow(u_chi, a_inv, EPS_U) - 1.0  # VSS angular law
    sin_x = safe_sqrt(1.0 - cos_x * cos_x)

    vr = vi - vj
    vr2 = jnp.sum(vr * vr, axis=-1)
    vrm = safe_sqrt(vr2)

    # No internal energy yet, so etrans is unchanged and |v_r'| == |v_r|.  `scale`
    # is identically 1.0; kept as the hook for Larsen-Borgnakke, where it becomes
    # sqrt(2 * etrans_post / (mr * vr2)) and `sp.mr[isp, jsp]` comes back into use.
    scale = 1.0

    # frame aligned to v_r; collide_vss.cpp:429-437
    d = safe_norm(vr[..., 1:3], axis=-1)
    # RELATIVE threshold.  SPARTA's `if (d > 1.0e-6)` is dimensional and only works
    # because SPARTA runs in SI with thermal speeds ~1e2 m/s; under any
    # nondimensionalisation it fires on different physical configurations, changing
    # the post-collision velocity by order unity (section 7.4).  `vrm` is already
    # computed one line above, so the relative form costs nothing.
    ok = frame_is_well_defined(vi, vj)
    d_safe = jnp.where(ok, d, 1.0)  # denominator made safe FIRST (section 2)

    vr0, vr1, vr2c = vr[..., 0], vr[..., 1], vr[..., 2]
    sin_e, cos_e = jnp.sin(eps_az), jnp.cos(eps_az)

    ua_g = cos_x * vr0 + sin_x * d * sin_e
    vb_g = cos_x * vr1 + sin_x * (vrm * vr2c * cos_e - vr0 * vr1 * sin_e) / d_safe
    wc_g = cos_x * vr2c - sin_x * (vrm * vr1 * cos_e + vr0 * vr2c * sin_e) / d_safe

    # degenerate fallback: v_r lies along x, so the aligned frame is undefined
    ua_d = cos_x * vr0
    vb_d = sin_x * vr0 * cos_e
    wc_d = sin_x * vr0 * sin_e

    vr_new = scale * jnp.stack(
        [jnp.where(ok, ua_g, ua_d), jnp.where(ok, vb_g, vb_d), jnp.where(ok, wc_g, wc_d)],
        axis=-1,
    )

    # mass-weighted CM decomposition; collide_vss.cpp:446-452
    msum = mi + mj
    vcm = (mi[..., None] * vi + mj[..., None] * vj) / msum[..., None]
    return vcm + (mj / msum)[..., None] * vr_new, vcm - (mi / msum)[..., None] * vr_new


def reduced_mass(mass):
    """Pair table ``mr[i, j] = m_i m_j / (m_i + m_j)``, ``f64[S, S]``.  TRACED."""
    return mass[:, None] * mass[None, :] / (mass[:, None] + mass[None, :])


def species_params(mass, diam, omega, tref, alpha) -> SpeciesParams:
    """Assemble :class:`SpeciesParams` from the five **primitive** VSS constants.

    ``mr`` and ``prefactor`` are *derived* fields, but they are stored leaves of the
    dataclass -- so a gradient taken with respect to a hand-built ``SpeciesParams``
    treats ``prefactor`` as independent and **silently loses** the ``diam``,
    ``omega`` and ``tref`` paths through it.  That is the entire model-calibration
    gradient, and it fails to zero rather than to an error.

    Build the container through this function whenever it is differentiated, so the
    derived quantities stay downstream of the primitives.

    Spec: sections 3.1 and 7.1.
    """
    mr = reduced_mass(mass)
    return SpeciesParams(
        mass=mass,
        diam=diam,
        omega=omega,
        tref=tref,
        alpha=alpha,
        mr=mr,
        prefactor=make_prefactor(diam, omega, tref, mr),
    )
