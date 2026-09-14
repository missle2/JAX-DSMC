"""An exact replication of [P2]'s section 6 numerical tests, in our production solver.

[P2] is ``papers_reference/arXiv-2207.11579v2/main.tex``.  Its section 6 (``:1401``)
validates the adjoint DSMC method on a **0-D homogeneous** relaxation, and it is the only
published, external check available for the whole estimator.  Reproducing it here tests
the forward solve, the pathwise gradient and the score terms together against numbers we
did not generate.

The mapping onto our VSS solver, which is exact for the angle-independent suite:

===================================  ==================================  ==================
[P2]                                 ours                                mapping
===================================  ==================================  ==================
``q = C_k(theta) |v-v_1|^beta``      ``sigma_T v_r = pre (v_r^2)^(1-w)``  ``omega = 1 - beta/2``
``kappa = 0`` (isotropic)            VSS deflection law                  ``alpha = 1``
``A_kappa = 1/epsilon``              ``prefactor``                       ``pre = 1/eps = 0.1``
``Sigma_v = 10^beta``                ``vremax``                          ``Sigma_v/eps``
``rho = 1``                          number density ``N/V``              ``V = N``, ``fnum = 1``
``v^0 = (sqrt(T_p) N_p)_p``          built here, not by ``initialize``   anisotropic
===================================  ==================================  ==================

Under that mapping our NTC attempt count
``0.5 npc (npc-1) Sigma dt fnum / v_cell`` reduces to ``0.5 N rho Sigma dt``, which is
exactly [P2]'s ``N_c / 2`` pairs per step with ``N_c/N = dt Sigma_v / eps`` (``:1440``).

**Sigma is frozen without touching production code.**  ``collide`` takes
``Sigma = max(cstate.vremax, thermal_majorant)``, a monotone ratchet.  Seeding
``vremax`` at [P2]'s a-priori bound ``10^beta / eps`` -- which their ``|v_r|`` never
attains, since ``T^0 <= 1`` bounds ``max|v_i - v_j|`` by 10 (``:1447``) -- means the
ratchet never fires and ``Sigma`` stays fixed for the whole run, as their derivation
assumes.  :func:`check_majorant_is_frozen` asserts it rather than trusting it.

Two deviations are unavoidable and are recorded rather than hidden:

1. **Spatial cells as a computational device.**  [P2] is a single 0-D box.  A single
   cell here would make the sequential attempt-slot scan ``O(N)`` long.  We split the
   box into cells at fixed ``npc``, which leaves the total collision rate unchanged
   (``npc`` and the cell volume fall together and cancel -- the same argument the BKW
   example uses) but restricts pairing to within a cell.  For a homogeneous gas the two
   agree in distribution; they are not bit-identical at finite ``N``.
2. **The angle-dependent suite (section 6.2, ``kappa = {1,2,5}``) is NOT replicable
   here.**  Their ``C_kappa(theta) ~ (1+cos theta)^kappa`` is a different scattering law
   from VSS's ``cos chi = 2 u^(1/alpha) - 1``; no ``alpha`` reproduces it.  Replicating
   that section means implementing a second kernel, which is a physics addition, not a
   configuration.  Only ``kappa = 0`` is claimed exact.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jax_dsmc import adjoint
from jax_dsmc.collide import vremax_init
from jax_dsmc.collide.ntc import thermal_majorant
from jax_dsmc.grid import GridSpec, cell_volume, num_cells
from jax_dsmc.state import CollideState, ParticleState, SimParams, SimStatic, SpeciesParams
from jax_dsmc.viz import (
    INK_SECONDARY,
    REFERENCE,
    SERIES_ANALYTIC,
    SERIES_MEASURED,
    make_figure,
    savefig,
)

EPS = 10.0
"""[P2]'s ``epsilon``, which sets collisions per unit time (``:1476``)."""

T0_REF = (1.0, 1.0, 0.5)
"""``m = (T_x^0, T_y^0, T_z^0)`` -- [P2] eq. (IC), ``:1452``."""

T_MAXWELL = sum(T0_REF) / 3.0
"""0.8333(3): the isotropic temperature the gas relaxes to (``:1453``)."""

DT = 0.1
N_STEPS = 20
"""``Delta t = 0.1``, ``M = 20``, so ``T = M dt = 2`` (``:1454``)."""

DELTA_M = 0.1
"""[P2]'s finite-difference step, ``:1431`` -- 'nearly optimal for N = 1e6, M_s = 100'."""

M_S = 100
"""i.i.d. runs averaged on **both** sides before the error is formed (``:1425``)."""

MASS = 1.0
"""Irrelevant to the physics here: equal masses make VSS scatter mass-free, and the
prefactor is set directly rather than derived from ``(diam, tref, mr)``."""


def omega_of(beta):
    """``beta = 2(1 - omega)`` inverted.  beta 0/1/2 -> omega 1 / 0.5 / 0."""
    return 1.0 - 0.5 * beta


def sigma_v(beta):
    """[P2]'s majorant on ``|v_r|^beta``: ``10^beta`` (``:1447``)."""
    return 10.0**beta


def majorant(beta):
    """Our ``vremax``, i.e. the majorant on ``sigma_T v_r``: ``Sigma_v / eps``."""
    return sigma_v(beta) / EPS


def species(beta, alpha=1.0):
    """``SpeciesParams`` with the prefactor set **directly** to [P2]'s ``A_kappa``.

    Normally :func:`jax_dsmc.collide.vss.species_params` must build the prefactor from
    ``(diam, omega, tref, mr)`` or the gradient path through those three is silently
    lost.  Here that is the point: [P2] specifies the kernel by its *value*
    ``1/(40 pi) |v_r|^beta``, not by a molecular diameter, and the parameters we
    differentiate are ``T^0`` and ``alpha``, neither of which routes through the
    prefactor.  ``diam`` and ``tref`` are set to NaN-free dummies that are never read.
    """
    f = jnp.float64
    return SpeciesParams(
        mass=jnp.array([MASS]),
        diam=jnp.array([[1.0]]),
        omega=jnp.array([[f(omega_of(beta))]]),
        tref=jnp.array([[1.0]]),
        alpha=jnp.array([[f(alpha)]]),
        mr=jnp.array([[MASS / 2.0]]),
        prefactor=jnp.array([[f(1.0 / EPS)]]),
    )


@dataclasses.dataclass(frozen=True)
class Case:
    """A [P2] configuration.  Hashable so it can be a ``static_argnum``."""

    beta: int
    n_part: int
    side: int
    a_max: int
    n_steps: int = N_STEPS

    @property
    def n_cells(self):
        return self.side**3

    @property
    def npc(self):
        return self.n_part / self.n_cells

    @property
    def volume(self):
        """``V = N`` makes the number density ``rho = 1``, as [P2] assumes."""
        return float(self.n_part)


def case(beta, n_part, npc_target=1000, a_max=None, n_steps=N_STEPS):
    """Size the cell grid and ``A_max`` for a given ``(beta, N)``.

    ``A_max`` is per cell and the attempt count is ``~ 0.5 npc Sigma dt``, so it scales
    with ``npc`` and with ``Sigma = 10^(beta-1)`` -- at ``beta = 2`` that is 100x the
    ``beta = 0`` value, which is why ``npc_target`` has to come down as ``beta`` rises.
    """
    side = max(1, round((n_part / npc_target) ** (1.0 / 3.0)))
    n_cells = side**3
    npc = n_part / n_cells
    if a_max is None:
        expected = 0.5 * npc * majorant(beta) * DT
        a_max = max(8, math.ceil(1.4 * expected + 4.0 * math.sqrt(expected + 1.0)))
    return Case(beta=beta, n_part=n_part, side=side, a_max=a_max, n_steps=n_steps)


def build(cs: Case, alpha=1.0):
    """``(sp, grid, params, static, cstate)`` for a case.  No traced parameters here."""
    sp = species(cs.beta, alpha)
    box = cs.volume ** (1.0 / 3.0)
    grid = GridSpec(lo=jnp.zeros(3), hi=jnp.full(3, box), n=(cs.side, cs.side, cs.side))
    static = SimStatic(N=cs.n_part, D=3, S=1, G=1, A_max=cs.a_max, K_bounce=1, n_steps=cs.n_steps)
    params = SimParams(dt=jnp.float64(DT), fnum=jnp.ones(cs.n_cells), epsilon=jnp.float64(0.0))
    cstate = CollideState(vremax=jnp.full((cs.n_cells, 1, 1), jnp.float64(majorant(cs.beta))))
    return sp, grid, params, static, cstate


def initial_velocities(key, n_part, t0):
    """[P2]'s sampler verbatim (``:1460``): ``v^j = sqrt(T_j^0) N(0,1)``.

    Differentiable in ``t0`` by construction, giving their
    ``d v^j / d T_p^0 = v^j / (2 T_p^0) delta_{jp}`` -- the pathwise estimator, and the
    reason ``d/dT^0`` needs no score term.
    """
    z = random.normal(key, (n_part, 3), dtype=jnp.float64)
    return z * jnp.sqrt(jnp.asarray(t0, dtype=jnp.float64))


def initial_state(key, cs: Case, t0):
    """Uniform positions, anisotropic-Gaussian velocities.

    ``initialize.initial_state`` cannot be used: it takes a **scalar** temperature, so
    it can only make an isotropic Maxwellian, and [P2]'s whole setup is anisotropic.
    """
    k_x, k_v = random.split(key)
    box = cs.volume ** (1.0 / 3.0)
    x = random.uniform(k_x, (cs.n_part, 3), dtype=jnp.float64, maxval=box)
    return ParticleState(
        x=x,
        v=initial_velocities(k_v, cs.n_part, t0),
        species=jnp.zeros((cs.n_part,), dtype=jnp.int32),
        alive=jnp.ones((cs.n_part,), dtype=bool),
        logh_acc=jnp.zeros((cs.n_part,), dtype=jnp.float64),
    )


def check_majorant_is_frozen(cs: Case, t0=T0_REF):
    """Assert the a-priori thermal majorant cannot lift ``Sigma`` above [P2]'s value.

    If it could, ``collide`` would silently run at a larger ``Sigma`` than [P2] specifies
    -- same physics, different variance and different score terms -- and the replication
    would be of something else.  Returns ``(sigma_p2, thermal, ok)``.
    """
    sp = species(cs.beta)
    th = float(
        thermal_majorant(
            jnp.full((1,), jnp.float64(max(t0))),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
        )[0]
    )
    s = majorant(cs.beta)
    return s, th, th <= s


def expected_collision_fraction(beta):
    """[P2] ``:1440``: ``N_c / N = dt Sigma_v / eps``.  1% / 10% / 100% for beta 0/1/2."""
    return DT * sigma_v(beta) / EPS


def attempts_per_cell(cs: Case):
    """What our NTC will actually schedule, for sizing ``A_max`` and cross-checking."""
    npc = cs.npc
    v_cell = cs.volume / cs.n_cells
    return 0.5 * npc * (npc - 1.0) * majorant(cs.beta) * DT / v_cell


def majorant_used(cs: Case, t0=T0_REF):
    """The ``Sigma`` ``collide`` will actually use -- not always [P2]'s.

    At ``beta = 0`` (Maxwell molecules) ``q = prefactor`` is velocity-independent, so
    [P2] can take ``Sigma`` exactly equal to it.  Our a-priori thermal majorant carries
    a fixed ``F_UP = 2`` safety factor (section 7.2.3) which for ``omega = 1`` multiplies
    a temperature factor of ``(...)^0 = 1``, leaving ``Sigma = 2 x prefactor``.  It cannot
    be lowered from outside ``collide`` because the majorant enters as a ``max`` floor.

    Consequence, stated rather than hidden: at ``beta = 0`` we run NTC at **twice** [P2]'s
    majorant -- twice the attempts at half the acceptance.  Both are valid fixed-``Sigma``
    NTC schemes and both are unbiased for ``dE[J]/dm``, so the **gradient values and the
    ``e`` vs ``N`` convergence still replicate**; what does not is their *random-error*
    curve at ``beta = 0``, which is a function of ``Sigma``.  ``beta = 1`` and ``beta = 2``
    are exact, their thermal majorants being ~12 and ~22 orders of magnitude below
    ``Sigma_v/eps``.
    """
    s, th, _ = check_majorant_is_frozen(cs, t0)
    return max(s, th)


def _phi_t_l(x, v, component=1):
    """[P2]'s ``phi(v) = v_l^2``, so ``J = (1/N) sum_i (v_i^l)^2 = T_l``."""
    del x
    return v[:, component] ** 2


def value(t0, cs: Case, key, component=1, alpha=1.0, scored=True):
    """``T_l`` at ``t = M dt``, differentiable in ``t0`` (and in ``alpha``)."""
    sp, grid, params, static, cstate = build(cs, alpha)
    st = initial_state(key, cs, t0)
    j, _ = adjoint.terminal_objective(
        st,
        cstate,
        grid,
        sp,
        params,
        static,
        key,
        functools.partial(_phi_t_l, component=component),
        rho=1.0,
        collide_only=True,
        scored=scored,
        score_mode="paper",  # Retain the estimator compared to P2's recursion.
    )
    return j


def moments_of(t0, cs: Case, key, alpha=1.0):
    """``(T_x, T_y, T_z)`` at the final time -- the forward-physics sanity check.

    They must relax **towards** ``T_MAXWELL = 0.8333`` without reaching it: [P2] chooses
    ``T = 2`` precisely so that 'only partial relaxation is attained' (``:1454``).
    """
    return jnp.array([value(t0, cs, key, component=c, alpha=alpha) for c in (0, 1, 2)])


_grad_t0 = jax.grad(value, argnums=0)


def ad_gradient(t0, cs: Case, key, component=1, alpha=1.0, scored=True):
    """``dT_l / dm`` for ``m = (T_x^0, T_y^0, T_z^0)`` -- one forward + one reverse pass."""
    return _grad_t0(jnp.asarray(t0, dtype=jnp.float64), cs, key, component, alpha, scored)


def fd_gradient(t0, cs: Case, key, component=1, delta=DELTA_M, alpha=1.0):
    """[P2] eq. FD_gradient (``:1418``): central difference at a **shared** seed.

    The shared seed is not an optimisation -- without it the two forward runs are
    independent samples and the difference is pure noise.
    """
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    out = []
    for p in range(3):
        e = jnp.zeros(3).at[p].set(delta)
        hi = value(t0 + e, cs, key, component, alpha, scored=False)
        lo = value(t0 - e, cs, key, component, alpha, scored=False)
        out.append((hi - lo) / (2.0 * delta))
    return jnp.stack(out)


def euler_identity_residual(t0, cs: Case, key, component=1):
    """An **analytic** check on the gradient that needs no finite difference.

    At ``beta = 0`` the collision rate ``q = prefactor`` is velocity-independent, so
    scaling every velocity by ``s`` leaves the collision *process* untouched and simply
    scales the whole trajectory.  ``T_l`` is then homogeneous of degree 1 in
    ``m = (T_x^0, T_y^0, T_z^0)``, and Euler's theorem gives the exact identity

        ``sum_p T_p^0 dT_l/dT_p^0 = T_l``.

    This holds for **beta = 0 only** -- at ``beta > 0`` the rate ``|v_r|^beta`` changes
    under the scaling, the dynamics are not scale-invariant, and the identity fails by a
    margin that grows with ``beta``.  Returns ``(lhs, rhs, relative residual)``.

    It is a stronger check than AD-vs-FD because the right-hand side is a *forward*
    quantity: an error in the adjoint cannot cancel out of both sides.
    """
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    g = ad_gradient(t0, cs, key, component=component)
    lhs = float(jnp.sum(t0 * g))
    rhs = float(value(t0, cs, key, component=component))
    return lhs, rhs, abs(lhs - rhs) / abs(rhs)


@functools.cache
def _batched(cs: Case, component, alpha, delta):
    """``keys -> (ad[K,3], fd[K,3])``, jitted once per configuration.

    Cached on the case because tracing dominates otherwise: a single un-jitted
    ``ad + fd`` pair measured 35 s at ``N = 8000``, nearly all of it retracing.
    """

    def one(key):
        ad = ad_gradient(T0_REF, cs, key, component=component, alpha=alpha, scored=True)
        fd = fd_gradient(T0_REF, cs, key, component=component, delta=delta, alpha=alpha)
        return ad, fd

    return jax.jit(jax.vmap(one))


def ensemble(cs: Case, keys, component=1, alpha=1.0, delta=DELTA_M, chunk=None):
    """[P2]'s ``M_s`` i.i.d. runs, each with its own random initial condition.

    Returns ``(ad[K,3], fd[K,3])``.  Halves ``chunk`` on OOM rather than losing the run.
    """
    f = _batched(cs, component, alpha, delta)
    chunk = len(keys) if chunk is None else chunk
    while chunk >= 1:
        try:
            outs = [
                jax.block_until_ready(f(keys[i : i + chunk])) for i in range(0, len(keys), chunk)
            ]
            return (
                jnp.concatenate([o[0] for o in outs]),
                jnp.concatenate([o[1] for o in outs]),
            )
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("a single key does not fit on this device")


def error_metric(ad, fd):
    """[P2] eq. error (``:1429``): ``e = |mean(grad^AD) - mean(grad^FD)|``, componentwise.

    Also returns each side's random error, which [P2] forms the same way -- the standard
    deviation over the ``M_s`` runs rescaled by ``1/sqrt(M_s)`` (``:1433``).
    """
    n = ad.shape[0]
    ad_m, fd_m = jnp.mean(ad, axis=0), jnp.mean(fd, axis=0)
    return (
        jnp.abs(ad_m - fd_m),
        ad_m,
        fd_m,
        jnp.std(ad, axis=0, ddof=1) / math.sqrt(n),
        jnp.std(fd, axis=0, ddof=1) / math.sqrt(n),
    )


def _value_alpha(alpha, t0, cs: Case, key, component=1, scored=True):
    """:func:`value` with ``alpha`` first, so ``jax.grad`` can take it."""
    return value(t0, cs, key, component=component, alpha=alpha, scored=scored)


_grad_alpha = jax.grad(_value_alpha, argnums=0)


def ad_gradient_alpha(alpha, cs: Case, key, component=1, t0=T0_REF, scored=True):
    """``dT_l / d(alpha)``: the VSS scattering exponent, which [P2] never differentiates.

    **This is a positive control, and it exists because of a real doubt.**  On the 2-D
    shock the ensemble mean of ``dT_y/d(alpha)`` came out consistent with zero at every
    ``N`` while its per-run spread stayed large -- and although the finite difference
    independently agreed (section 23.3.4), 'both say zero' is a weak result to rest on.
    What was missing was a case where ``dJ/d(alpha)`` is unambiguously **large**, to show
    the machinery detects an ``alpha`` sensitivity when there is one to detect.

    This 0-D relaxation is that case: ``alpha`` sets the scattering anisotropy, which
    directly sets the rate at which ``T_y`` relaxes towards ``T_M``, with no transport,
    no boundary and no shock structure in the way.
    """
    return float(_grad_alpha(jnp.float64(alpha), t0, cs, key, component, scored))


def fd_gradient_alpha(alpha, cs: Case, key, component=1, delta=DELTA_M, t0=T0_REF):
    """Central difference in ``alpha`` at a shared seed -- the independent check."""
    hi = value(t0, cs, key, component, alpha=alpha + delta, scored=False)
    lo = value(t0, cs, key, component, alpha=alpha - delta, scored=False)
    return float((hi - lo) / (2.0 * delta))


@functools.cache
def _batched_alpha(cs: Case, component, alpha, delta):
    def one(key):
        return (
            _grad_alpha(jnp.float64(alpha), T0_REF, cs, key, component, True),
            (
                value(T0_REF, cs, key, component, alpha=alpha + delta, scored=False)
                - value(T0_REF, cs, key, component, alpha=alpha - delta, scored=False)
            )
            / (2.0 * delta),
        )

    return jax.jit(jax.vmap(one))


def ensemble_alpha(cs: Case, keys, component=1, alpha=1.4, delta=DELTA_M, chunk=None):
    """``M_s`` i.i.d. runs of ``dT_l/d(alpha)``, AD and FD.  Returns ``(ad[K], fd[K])``."""
    f = _batched_alpha(cs, component, alpha, delta)
    chunk = len(keys) if chunk is None else chunk
    while chunk >= 1:
        try:
            outs = [
                jax.block_until_ready(f(keys[i : i + chunk])) for i in range(0, len(keys), chunk)
            ]
            return (
                jnp.concatenate([o[0] for o in outs]),
                jnp.concatenate([o[1] for o in outs]),
            )
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("a single key does not fit on this device")


def p2_legendre_moment(alpha):
    """``<cos^2 chi>`` for the VSS deflection law, in closed form.

    With ``cos chi = 2 u^(1/alpha) - 1`` and ``u ~ U(0,1)``,

        ``<cos^2 chi> = 4a/(a+2) - 4a/(a+1) + 1``.

    Anisotropy relaxation is driven by the second Legendre moment of the deflection,
    so this -- not ``alpha`` itself -- is the quantity a temperature objective feels.
    """
    a = jnp.asarray(alpha, dtype=jnp.float64)
    return 4.0 * a / (a + 2.0) - 4.0 * a / (a + 1.0) + 1.0


def d_legendre_d_alpha(alpha):
    """``d<cos^2 chi>/d(alpha) = 8/(a+2)^2 - 4/(a+1)^2``.

    **Zero at exactly ``alpha = sqrt(2)``**: ``8/(a+2)^2 = 4/(a+1)^2`` gives
    ``sqrt(2)(a+1) = a+2``, so ``a = (2-sqrt2)/(sqrt2-1) = sqrt2``.  The moment is
    ``1/3`` at both ``alpha = 1`` and ``alpha = 2`` with a minimum between them.
    """
    a = jnp.asarray(alpha, dtype=jnp.float64)
    return 8.0 / (a + 2.0) ** 2 - 4.0 / (a + 1.0) ** 2


ALPHA_STATIONARY = math.sqrt(2.0)
"""``sqrt(2) = 1.41421``.  Argon's tabulated VSS ``alpha = 1.4`` is 1.0 % below it,
which is why ``dT_l/d(alpha)`` measures ~zero on the argon shock case: the objective
sits at a **stationary point in alpha**, not at a broken gradient.  See the README."""


def figure_alpha_sweep(alphas, ad, ad_se, fd, fd_se, path):
    """The money plot: measured ``dT_y/d(alpha)`` against the analytic angular moment."""
    a = np.asarray(alphas, dtype=float)
    ad, fd = np.asarray(ad), np.asarray(fd)
    fp = np.asarray(d_legendre_d_alpha(a))
    k = float(np.sum(fp * ad) / np.sum(fp * fp))

    fig, ax = make_figure(1, 2, figsize=(11.6, 4.4))
    fine = np.linspace(min(a.min(), 1.0), max(a.max(), 2.0), 200)
    ax[0].plot(
        fine,
        k * np.asarray(d_legendre_d_alpha(fine)),
        "-",
        color=REFERENCE,
        linewidth=1.8,
        label=rf"$k\,\partial_\alpha\langle\cos^2\chi\rangle$, $k={k:.3f}$",
    )
    ax[0].errorbar(
        a,
        ad,
        yerr=ad_se,
        fmt="o",
        color=SERIES_ANALYTIC,
        markersize=7,
        capsize=4,
        label="adjoint (AD)",
    )
    ax[0].errorbar(
        a,
        fd,
        yerr=fd_se,
        fmt="s",
        color=SERIES_MEASURED,
        markersize=6,
        capsize=4,
        label="finite difference",
    )
    ax[0].axhline(0.0, color=INK_SECONDARY, linewidth=1.0)
    ax[0].axvline(ALPHA_STATIONARY, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[0].text(
        ALPHA_STATIONARY + 0.02,
        ax[0].get_ylim()[1] * 0.55,
        r"$\alpha=\sqrt{2}$",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax[0].set(xlabel=r"VSS $\alpha$", ylabel=r"$\partial T_y/\partial\alpha$")
    ax[0].set_title(r"$dT_y/d\alpha$ changes sign at $\alpha=\sqrt{2}$", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)

    ax[1].plot(
        fine, np.asarray(p2_legendre_moment(fine)), "-", color=SERIES_ANALYTIC, linewidth=2.0
    )
    ax[1].axvline(ALPHA_STATIONARY, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax[1].axvline(1.4, color=SERIES_MEASURED, linestyle=":", linewidth=1.8)
    ax[1].text(
        1.405,
        float(p2_legendre_moment(1.7)),
        "argon $\\alpha=1.4$",
        color=SERIES_MEASURED,
        fontsize=9,
        rotation=90,
        va="center",
    )
    ax[1].set(xlabel=r"VSS $\alpha$", ylabel=r"$\langle\cos^2\chi\rangle$")
    ax[1].set_title(r"the moment is stationary at $\sqrt{2}$ — and argon sits there", fontsize=11)
    return savefig(fig, path)
