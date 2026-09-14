"""Adjoint verification on the Bird normal-shock case (spec sections 11 and 13.4).

Takes the reflected-shock configuration that reproduced Bird's thickness in
``examples/shock`` and asks a different question of it: **are the gradients right?**

The answer has to be given twice, because the parameters split into two groups that
require *different* verification methods, and running the wrong one produces a
confident wrong answer rather than an error.

**Group 1 -- parameters with a pathwise route.**  ``temp``, ``vstream`` and
``alpha`` change the velocities themselves: ``temp`` and ``vstream`` through the
reparameterised Maxwellian, ``alpha`` through the VSS deflection law
``cos X = 2 U^(1/alpha) - 1``.  At frozen randomness ``J`` is a smooth function of
them, so a central finite difference converges to the pathwise derivative and
agreement should be at round-off.

**Group 2 -- parameters that reach ``J`` only through a frozen decision.**
``diam``, ``omega`` and ``tref`` enter *only* through ``q = sigma_T v_r``, which is
used to accept or reject a candidate; ``nrho`` enters only through ``fnum``, which
sets the attempt count.  Perturb any of them by less than it takes to flip a
decision and ``J`` does not move **at all**.  Their fixed-key finite difference is
exactly ``0.0``, and so is the pathwise gradient -- they agree, vacuously, on the
wrong number.  The real sensitivity is the *score* term, and it estimates
``dE[J]/dtheta``, so it can only be checked against an **ensemble** finite
difference over many base keys.

That split is not a quirk of this example; it is the reason section 13.4 requires
the harness to run in two modes and to name them differently.

**The flip guard is what makes mode A trustworthy.**  Section 13.4.1 measured the
failure: at ``h`` a factor of three too large, one particle changing cell rebinds
60 pair indices and the FD estimate is wrong by a factor of 11 -- with no error and
no ``NaN``.  Every fixed-key difference here compares
``Diagnostics.decision_checksum`` between the two evaluations and refuses to report
a number unless they match exactly.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import adjoint, species_data
from jax_dsmc.boundary import BC
from jax_dsmc.collide import species_params, vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.state import (
    CollideState,
    FlowParams,
    GridSpec,
    SimParams,
    SimStatic,
)

jax.config.update("jax_enable_x64", True)

# Argon, SPARTA's `data/ar.vss`: `Ar 4.11e-10 0.81 273.15 1.4`
AR_MASS = 6.63e-26
AR_DIAM = 4.11e-10
AR_OMEGA = 0.81
AR_TREF = 273.15
AR_ALPHA = 1.4

DEFAULT_GAS = species_data.ARGON
"""The gas every case uses unless told otherwise.  Argon keeps the published shock
comparison (Bird section 12.11) intact.

**But argon is the worst possible choice for an `alpha` study**, and that is measured,
not a guess: anisotropy relaxation is driven by `<cos^2 chi>`, whose derivative in
`alpha` vanishes at exactly `sqrt(2) = 1.41421`, and argon's tabulated `alpha = 1.40`
sits 1.0 % away.  `examples/p2_replication` measures the resulting resolvability across
Bird's five monatomic gases -- `|AD|/SE` of He 4.7, Ne 3.1, Kr 2.6, Ar 0.3, Xe 0.0.
Pass `gas=species_data.HELIUM` (12.6x argon's conditioning) when `alpha` is the
parameter under test.  See `jax_dsmc.species_data.best_for_alpha_study`.
"""

MACH = 2.0
NRHO = 1e21
GAMMA = 5.0 / 3.0

PATHWISE = ("temp", "ux", "alpha")
"""Parameters with a pathwise route to ``J`` -- checkable by a fixed-key FD."""

SCORE_ONLY = ("diam", "omega", "tref", "nrho")
"""Parameters reachable **only** through a frozen decision -- ensemble FD only."""

HEADLINE = ("temp", "ux", "alpha")
"""The three swept in the figures: two IC (``temp``, ``ux``) and one VSS (``alpha``).

Exactly the parameters that have a pathwise route, which is what [P1] and [P2]
verify.  [P1] is Maxwell molecules with ``Sigma = q``, so ``p == 1``, no candidate is
ever rejected and **no score term exists at all** -- purely pathwise.  [P2] adds the
``eta`` terms but still reports sensitivities with respect to the *initial
distribution* parameter, which is ``temp`` here.

``SCORE_ONLY`` is deliberately not in this list.  Those parameters are real and their
gradients are computed, but checking them needs the ensemble machinery of mode B,
whose cost and variance are under investigation rather than being a verification
claim.  See the README."""

LABEL = {
    "temp": r"$T_1$  (K)",
    "ux": r"$u_{\rm stream}$  (m/s)",
    "nrho": r"$n_1$  (m$^{-3}$)",
    "diam": r"$d_{\rm ref}$  (m)",
    "omega": r"$\omega$",
    "tref": r"$T_{\rm ref}$  (K)",
    "alpha": r"$\alpha$",
}


def mfp_vss(nrho, alpha=None, diam=None, gas=None):
    """Bird eqn 4.52.  See ``examples/shock/README.md`` -- this one is easy to get wrong."""
    gas = DEFAULT_GAS if gas is None else gas
    alpha = gas.alpha if alpha is None else alpha
    diam = gas.diam if diam is None else diam
    hs = 1.0 / (math.sqrt(2.0) * math.pi * diam * diam * nrho)
    return (alpha + 1.0) * (alpha + 2.0) / (6.0 * alpha) * hs


def piston_speed(mach=MACH, temp=None, gas=None):
    """``u_p`` for a shock reflected off a specular wall, and the sound speed."""
    gas = DEFAULT_GAS if gas is None else gas
    temp = gas.tref if temp is None else temp
    a1 = math.sqrt(GAMMA * K_BOLTZ * temp / gas.mass)
    r = (GAMMA + 1.0) * mach * mach / ((GAMMA - 1.0) * mach * mach + 2.0)
    return mach * a1 * (1.0 - 1.0 / r), a1


def reference_theta(gas=None):
    """The point in parameter space the sweeps are centred on.

    Defaults to Bird's Mach-2 argon.  Pass ``gas=species_data.HELIUM`` for an ``alpha``
    study -- see :data:`DEFAULT_GAS` for why argon cannot resolve one.
    """
    gas = DEFAULT_GAS if gas is None else gas
    u_p, _ = piston_speed(gas=gas)
    return {
        "temp": jnp.float64(gas.tref),
        "ux": jnp.float64(-u_p),
        "nrho": jnp.float64(NRHO),
        "diam": jnp.float64(gas.diam),
        "omega": jnp.float64(gas.omega),
        "tref": jnp.float64(gas.tref),
        "alpha": jnp.float64(gas.alpha),
    }


@dataclasses.dataclass(frozen=True)
class Case:
    """Static shape of one verification case.  Hashable, so it can be a static arg."""

    dim: int
    n_x: int
    transverse: int
    npc: int
    n_steps: int
    a_max: int
    domain_mfp: float
    window_mfp: float
    width_mfp: float
    dt_scale: float = 1.0
    remat: str = "nothing_saveable"
    """``jax.checkpoint_policies`` attribute used by the time loop.

    Exposed on the case so the reverse-pass cost can be measured against it.  The
    default rematerialises everything, which is the right choice at production
    scale (memory) and the wrong one for a small benchmark (recompute)."""

    gas: species_data.Species = DEFAULT_GAS
    """Which gas.  **On the Case, not passed separately, on purpose.**

    ``mass`` and ``dt`` are derived from it, so two gases give genuinely different
    compiled programs.  The jitted wrappers here are ``functools.cache``d on the case,
    so if the gas lived outside it two gases would silently share one cache entry and
    the second would run with the first's ``dt``."""

    @property
    def n_cells(self):
        return self.n_x * self.transverse ** (self.dim - 1)

    @property
    def n_part(self):
        return self.n_cells * self.npc

    @property
    def work(self):
        """``N * C * M`` -- the quantity section 13.4.1 fits ``h_crit ~ 5.8 / work`` to."""
        return self.n_part * self.n_cells * self.n_steps


def case(
    dim,
    npc=50,
    n_steps=16,
    n_x=40,
    transverse=2,
    a_max=256,
    domain_mfp=20.0,
    remat="nothing_saveable",
    gas=None,
    dt_scale=1.0,
):
    """A small reflected-shock case, sized so a finite-difference window still exists.

    Deliberately far below the production shock of ``examples/shock`` (which runs
    ``N * C * M ~ 1e11``).  Section 13.4.1 measured ``h_crit ~ 5.8 / (N C M)``
    against a round-off floor near ``1e-9``, so at production scale there is **no
    valid step size at all** -- the FD gate is only available on small cases, and
    that is a property of the method, not of this code.
    """
    return Case(
        dim=dim,
        n_x=n_x,
        transverse=transverse,
        npc=npc,
        n_steps=n_steps,
        a_max=a_max,
        domain_mfp=domain_mfp,
        window_mfp=6.0,
        width_mfp=3.0,
        remat=remat,
        dt_scale=dt_scale,
        gas=DEFAULT_GAS if gas is None else gas,
    )


def calibrate(cs: Case, theta=None, key=None, headroom=1.6, max_rounds=6):
    """Return ``cs`` with an ``A_max`` the flow actually fits in.

    **Why this is not automatic.**  ``jax_dsmc.simulate.run_adaptive`` already grows
    ``A_max`` whenever ``attempts_clipped > 0`` -- but it does so with *host-side
    Python control flow*, re-running the offending chunk after a recompile, so it
    cannot appear inside a differentiated call.  Its own docstring says as much:
    calibrate with ``run_adaptive``, then take gradients through ``run`` at the
    settled capacity.  ``adjoint.terminal_objective`` therefore calls plain ``run``,
    and the calibration has to happen **here**, once, before any gradient is taken.

    Skipping it is not a small matter.  A hand-picked ``A_max = 64`` against a peak
    of 73 ran with ``attempts_clipped = 9`` every step: a silently under-collided
    gas, gap U4's exact failure mode, and every gradient measured on it would have
    been correct for the wrong physics.

    Uses the diagnostics rather than ``run_adaptive`` itself, because all that is
    needed is ``max_attempts`` -- one forward run per round, no re-running.
    """
    theta = reference_theta() if theta is None else theta
    key = jax.random.PRNGKey(0) if key is None else key
    for _ in range(max_rounds):
        _, diag = _jit_evaluate(theta, cs, key, False)
        if int(jnp.sum(diag.attempts_clipped)) == 0:
            return cs
        peak = int(jnp.max(diag.max_attempts))
        cs = dataclasses.replace(cs, a_max=max(cs.a_max + 1, int(headroom * peak)))
    raise RuntimeError(f"A_max did not settle; last try {cs.a_max}")


def build(theta, cs: Case):
    """Assemble the run from the primitive controls.  Everything here is TRACED.

    ``species_params`` rather than a hand-built :class:`SpeciesParams`: ``mr`` and
    ``prefactor`` are *derived*, and a container built by hand would leave
    ``prefactor`` as an independent leaf -- silently zeroing the ``diam``,
    ``omega`` and ``tref`` gradients, which is the entire model-calibration signal.

    ``fnum`` is derived as ``nrho * V / N`` at fixed simulated particle count, which
    is what gives ``nrho`` a gradient path at all: it enters the NTC attempt rate,
    and so is carried by the count score of section 8.1.
    """
    lam = mfp_vss(NRHO)
    length = cs.domain_mfp * lam
    dx = length / cs.n_x
    trans_len = dx * cs.transverse
    grid = GridSpec(
        lo=jnp.zeros(cs.dim),
        hi=jnp.array([length] + [trans_len] * (cs.dim - 1)),
        n=(cs.n_x,) + (cs.transverse,) * (cs.dim - 1),
        transverse=jnp.float64(trans_len ** (3 - cs.dim)),
    )
    n_cells = num_cells(grid)
    sp = species_params(
        jnp.array([cs.gas.mass]),
        jnp.reshape(theta["diam"], (1, 1)),
        jnp.reshape(theta["omega"], (1, 1)),
        jnp.reshape(theta["tref"], (1, 1)),
        jnp.reshape(theta["alpha"], (1, 1)),
    )
    volume = length * trans_len**2
    fnum = jnp.full((n_cells,), theta["nrho"] * volume / cs.n_part)
    # dt is held FIXED across the sweep on purpose.  Deriving it from `temp` would
    # make a `temp` sweep vary two things at once, and the resulting gradient would
    # not be the one being checked (section 22.6: a control must vary one thing).
    # Thermal speed sets the CFL-like step, so it must follow the GAS, not argon:
    # helium is 16x lighter, hence 4x faster, hence needs a 4x smaller dt.
    dt = cs.dt_scale * 0.15 * dx / math.sqrt(K_BOLTZ * cs.gas.tref / cs.gas.mass)
    static = SimStatic(
        N=cs.n_part,
        D=cs.dim,
        S=1,
        G=1,
        A_max=cs.a_max,
        K_bounce=1,
        n_steps=cs.n_steps,
        remat_policy=cs.remat,
    )
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    flow = FlowParams(
        nrho=theta["nrho"],
        temp=theta["temp"],
        vstream=jnp.stack([theta["ux"], jnp.float64(0.0), jnp.float64(0.0)]),
    )
    return flow, sp, grid, params, static, lam


def _phi(cs: Case, lam):
    """Objective density: kinetic energy in a Gaussian window a few mfp off the wall.

    A *smooth* window, not an indicator: ``lo < x < hi`` has zero derivative almost
    everywhere and a delta at each edge, which would freeze the transport half of
    the gradient out exactly the way ``floor`` freezes the attempt count
    (section 8.1).  The window sits where the reflected shock is during the run, so
    ``J`` responds to the wave's strength and position rather than to the far field.
    """
    return functools.partial(
        adjoint.phi_windowed,
        inner=adjoint.phi_kinetic,
        axis=0,
        center=cs.window_mfp * lam,
        width=cs.width_mfp * lam,
    )


def evaluate(theta, cs: Case, base_key, scored=True, baseline=None):
    """``J`` at one parameter point and one base key.  Returns ``(J, diag)``.

    ``scored`` picks which gradient this carries -- :func:`jax_dsmc.adjoint.surrogate`
    (``dE[J]/dtheta``, includes the score terms) or
    :func:`jax_dsmc.adjoint.objective` (pathwise only).  **The value is identical
    either way**, which is what makes the comparison fair: both modes difference
    the same numbers.
    """
    flow, sp, grid, params, static, lam = build(theta, cs)
    state = initial_state(base_key, flow, grid, sp, static)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((num_cells(grid),), theta["temp"]),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            num_cells(grid),
            1,
        )
    )
    j, (_, _, diag) = adjoint.terminal_objective(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        base_key,
        _phi(cs, lam),
        rho=1.0,
        bc=BC.SPECULAR,
        baseline=baseline,
        scored=scored,
    )
    return j, diag


def _value(theta, cs, base_key, scored):
    return evaluate(theta, cs, base_key, scored)[0]


def _value_and_grad(theta, cs, base_key, scored):
    return jax.value_and_grad(_value)(theta, cs, base_key, scored)


# `cs` and `scored` are STATIC: `cs` fixes every shape and loop trip count, and
# `scored` selects a code path at trace time.  Both are hashable, so each distinct
# case compiles once and every subsequent call -- and the FD sweeps make hundreds --
# reuses it.  Without this the harness re-traces a scan-inside-a-scan under
# `jax.checkpoint` on every single evaluation.
_jit_value = jax.jit(_value, static_argnums=(1, 3))
_jit_value_and_grad = jax.jit(_value_and_grad, static_argnums=(1, 3))
_jit_evaluate = jax.jit(evaluate, static_argnums=(1, 3))


def value_and_grad(theta, cs: Case, base_key, scored=True):
    """``(J, dJ/dtheta)`` at one base key.  ``theta`` is a plain dict pytree."""
    return _jit_value_and_grad(theta, cs, base_key, scored)


def fingerprint(diag):
    """The frozen-decision fingerprint of a whole run, as a NumPy array.

    Two evaluations that produced identical arrays here took **identical discrete
    paths**: same cell membership, same attempt counts, same pair choices, same
    accept bits.  That is the precondition for a fixed-key finite difference to
    mean anything.
    """
    return np.asarray(diag.decision_checksum, dtype=np.float64)


def perturb(theta, name, delta):
    """``theta`` with one entry multiplied by ``1 + delta``.  Relative, so unit-free."""
    out = dict(theta)
    out[name] = theta[name] * (1.0 + delta)
    return out


def fd_central(theta, cs, base_key, name, h, scored=False):
    """Central difference in ``theta[name]``, with the flip guard and a noise floor.

    Returns ``(estimate, flipped, floor)``.

    ``flipped`` is ``True`` when the two evaluations took different discrete paths,
    in which case the estimate is a difference quotient across a discontinuity and
    must not be reported.

    ``floor`` is the **cancellation noise** of the difference itself,
    ``eps * |J| / (h * |theta|)``.  It is not decoration: when the derivative is
    small compared with ``J`` -- ``dJ/dalpha ~ 20`` against ``J ~ 8e4`` at ``D = 3``
    -- this floor is what limits agreement, and judging that case by relative error
    alone reports a spurious ``1e-7`` discrepancy that is entirely round-off.
    """
    jp, dp = _jit_evaluate(perturb(theta, name, h), cs, base_key, scored)
    jm, dm = _jit_evaluate(perturb(theta, name, -h), cs, base_key, scored)
    flipped = not np.array_equal(fingerprint(dp), fingerprint(dm))
    denom = 2.0 * h * float(theta[name])
    scale = max(abs(float(jp)), abs(float(jm)))
    return float(jp - jm) / denom, flipped, np.finfo(np.float64).eps * scale / abs(denom)


def largest_flip_free_h(theta, cs, base_key, name, h0=1e-5, h_min=1e-10, factor=2.0):
    """Bisect ``h`` downward until no frozen decision flips between the two sides.

    Section 13.4.1 measured that this is a **cliff, not a slope**: FD relative error
    goes ``3.3e-13 -> 11.1`` across a factor of ~3 in ``h``, and the critical value
    varies up to 30x between seeds.  So a single globally-chosen ``h`` is unsafe
    even where a window exists, and the per-seed search is not optional.

    **``h0`` caps the search at ``1e-5``, and that cap is load-bearing.**  Taking the
    largest flip-free step is wrong when the window is *wide*: the search then stops
    somewhere truncation still dominates, and reports a "disagreement" that is
    nothing but the ``O(h^2)`` error of the central difference.

    Measured on ``alpha`` at ``D = 1`` (``scratch/adjoint/probe_alpha.py``), where the
    flip-free window is wide because perturbing the deflection exponent flips fewer
    decisions than perturbing a velocity:

    ======== ============ ==============================
    ``h``    ``|FD-AD|``  in units of the FD noise floor
    ======== ============ ==============================
    1.0e-04  1.28e-05     193.7
    3.3e-05  1.15e-06     5.8
    1.1e-05  5.97e-08     **0.1**
    ======== ============ ==============================

    A clean ``h^2``, and at ``1.1e-05`` it reaches the floor -- exactly where
    ``temp`` already was. An earlier cap of ``1e-4`` left ``alpha`` reading a
    ``4.4e-6`` relative error that was *entirely* truncation, and was briefly
    mis-attributed to cancellation. Where the case is large enough that ``h_crit``
    is below this cap, the bisection binds instead and the cap costs nothing.

    Returns ``None`` when the search reaches ``h_min`` without finding one, which is
    the honest answer for a case whose ``N * C * M`` has closed the window.
    """
    h = h0
    while h >= h_min:
        if not fd_central(theta, cs, base_key, name, h)[1]:
            return h
        h /= factor
    return None


# --- ensemble mode: the only way to check a score-carried parameter ---------


def _value_for_vmap(theta, base_key, cs, scored):
    """Argument order flipped so ``vmap`` can map the key and bind the rest."""
    return _value(theta, cs, base_key, scored)


def _grad_for_vmap(theta, base_key, cs):
    return jax.grad(_value_for_vmap)(theta, base_key, cs, True)


@functools.cache
def _vmapped(cs: Case, scored: bool, want_grad: bool):
    """``vmap`` over base keys, compiled once per ``(case, mode)``.

    ``cs`` and ``scored`` are bound into a ``partial`` rather than passed through
    ``vmap``: they are a Python dataclass and a bool, which are not array leaves,
    and ``in_axes`` only describes *positional* arguments.  The cache gives one
    compilation per case instead of one per call -- and the sweeps make hundreds.
    """
    inner = _grad_for_vmap if want_grad else _value_for_vmap
    bound = (
        functools.partial(inner, cs=cs)
        if want_grad
        else functools.partial(inner, cs=cs, scored=scored)
    )
    return jax.jit(jax.vmap(bound, in_axes=(None, 0)))


def _chunked(fn, theta, keys, chunk):
    """Apply a ``vmap``-ed ``fn`` over ``keys`` in host-side chunks.

    ``vmap`` over the whole ensemble at once is the fast path but its peak memory is
    ``n_keys`` times one run's: a 512-key reverse pass at ``D = 1`` asks for 4.0 GiB
    and rematerialisation cannot get it below 2.85 GiB, which is an out-of-memory on
    a 4 GB card.  The chunk loop is Python, outside any jit, so it costs one dispatch
    per chunk.
    """
    outs = [fn(theta, keys[i : i + chunk]) for i in range(0, len(keys), chunk)]
    return (
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs) if len(outs) > 1 else outs[0]
    )


def ensemble_values(theta, cs: Case, keys, chunk=32):
    """``f64[n_keys]`` of ``J``, one per base key."""
    return _chunked(_vmapped(cs, True, False), theta, keys, chunk)


def ensemble_grads(theta, cs: Case, keys, chunk=32):
    """Per-key ``dJ~/dtheta`` -- the **scored** gradient, an estimator of ``dE[J]/dtheta``."""
    return _chunked(_vmapped(cs, True, True), theta, keys, chunk)


def mean_sem(x):
    """``(mean, standard error)`` of a sample.  Everything below reports both."""
    a = np.asarray(x, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size))


def fd_ensemble(theta, cs: Case, keys, name, h, chunk=32):
    """Central difference of ``E[J]`` in ``theta[name]``, with common random numbers.

    Returns ``(estimate, standard_error)``.

    **``h`` must be large here, and that is the opposite of mode A.**  ``E[J]`` is
    smooth in every parameter, but each *realisation* is a step function, so a small
    ``h`` flips no decisions, returns exactly zero for every seed, and reports a
    confident ``0.0``.  ``h`` therefore has to be big enough that decisions flip
    often -- a few percent -- and the price is an ``O(h^2)`` truncation error that
    the sweep plots make visible.

    The two sides share base keys (common random numbers), which correlates them and
    cuts the variance of the difference well below ``2 sigma_J^2 / M``.
    """
    jp = np.asarray(ensemble_values(perturb(theta, name, h), cs, keys, chunk))
    jm = np.asarray(ensemble_values(perturb(theta, name, -h), cs, keys, chunk))
    scale = 2.0 * h * float(theta[name])
    return mean_sem((jp - jm) / scale)
