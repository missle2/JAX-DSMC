"""Bird's normal-shock-structure validation (Molecular Gas Dynamics section 12.11).

This is a far sharper test than the Riemann problem in ``__init__.py``.  A Riemann
solution is *inviscid*, so it pins down wave positions and plateau values but says
nothing about the one quantity that is purely kinetic -- the **thickness** of the
shock.  Bird's normal-shock case measures exactly that, and it is anchored to
experiment (Alsmeyer 1976), so it catches errors in the collision rate, the
transport coefficients and the VSS model that an Euler comparison cannot see.

THE TARGET.  Bird defines the maximum-slope thickness

    L = (rho_2 - rho_1) / max|d rho / dx|

and reports, for argon at Mach 2 with VSS ``omega = 0.81``:

    L / lambda_1 = 6.2        (Bird section 12.11, DSMC)
    L / lambda_1 = 4.8        (Alsmeyer 1976, experiment, HARD-SPHERE mean free path)

Those agree: Bird's mean free path (eqn 4.52) is a fixed fraction of the hard-sphere
one (eqn 4.56),

    lambda_VSS / lambda_HS = alpha (5-2w)(7-2w) / [4 (alpha+1)(alpha+2)]

which is **0.780** for argon (``w = 0.81``, ``alpha = 1.40``), so ``4.8 / 0.780 = 6.2``.
**Getting this factor wrong misreports the thickness by 28 %**, and it is the single
easiest way to appear to validate against Bird while actually disagreeing with him.
Both numbers are reported below so the definition can never be ambiguous.

THE SET-UP, and how it differs from Bird's.  Bird uses a piston or stream boundary
(``DSMC1S``) to hold a *steady* shock in the frame of the wave.  We have neither
inflow nor a moving wall yet -- M2 has periodic and specular boundaries only -- so we
use the equivalent **reflected-shock** problem, which needs only a specular wall:

    uniform gas, moving at -u_p, strikes a stationary specular wall at x = 0.

The gas is brought to rest at the wall and a shock runs back into the oncoming
stream.  Its *structure* is the same steady structure Bird computes; only the frame
differs.  The upstream state is simply the undisturbed initial gas, so no inflow is
needed, and the measurement is made before the shock reaches the far boundary.

Choosing ``u_p`` for a target shock Mach number ``M``: the shock moves at ``u_s`` into
gas approaching at ``u_p``, so the upstream speed in the shock frame is ``u_s + u_p =
M a_1``; the post-shock gas is at rest in this frame, so mass conservation gives
``u_s = M a_1 / r`` with ``r`` the Rankine-Hugoniot density ratio.  Hence

    u_p = M a_1 (1 - 1/r).

Bird's own resolution requirements, all reported by :func:`report`: cells small
compared with ``lambda_1``, a timestep well below the mean collision time, and enough
particles that the density change across the wave is many standard deviations of the
statistical scatter (he notes that at Mach 1.2 it is only about two, which is why
weak waves are so much noisier).
"""

from __future__ import annotations

import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc.constants import K_BOLTZ  # noqa: E402

# Bird, Appendix A Table A3 -- and identical to SPARTA's `data/ar.vss`.
AR_MASS = 6.63e-26
AR_DIAM = 4.11e-10
AR_OMEGA = 0.81
AR_ALPHA = 1.40
AR_TREF = 273.15

# Bird section 12.11 / Alsmeyer 1976, for argon at Mach 2.
BIRD_L_OVER_LAMBDA_VSS = 6.2
ALSMEYER_L_OVER_LAMBDA_HS = 4.8


def mfp_hard_sphere(nrho, diam=AR_DIAM):
    """``1 / (sqrt(2) pi d^2 n)`` with ``d`` the **VHS reference diameter**.

    This is *not* the mean free path Alsmeyer quotes -- see :func:`mfp_alsmeyer`.
    It is the bookkeeping quantity the other two are built from.
    """
    return 1.0 / (math.sqrt(2.0) * math.pi * diam * diam * nrho)


def cross_section_ratio(omega=AR_OMEGA, alpha=AR_ALPHA):
    """``mu_HS(d_ref) / mu_VSS = (d_HS / d_ref)^2``.

        a (5-2w)(7-2w) / [4 (a+1)(a+2)]

    **This is a cross-section ratio, not a mean-free-path ratio, and confusing the
    two cost a 24 % "discrepancy" against Bird.**  It is 0.8244 at
    ``w = 0.75, a = 1.5325`` and 0.7751 at ``w = 0.81, a = 1.6625`` -- values that
    appear in Bird's text near his mean-free-path discussion and are easy to read
    as being that ratio.  They are not: the mean-free-path ratio is
    :func:`mfp_vss` / :func:`mfp_alsmeyer`, which is 0.8021 and 0.7577.  Bird's
    "76 %" is the latter; his "0.8244" is this.

    Its real job is defining the equivalent hard-sphere diameter: a hard-sphere gas
    reproduces the VSS gas's viscosity when ``d_HS^2 = d_ref^2 * this``.
    """
    return alpha * (5 - 2 * omega) * (7 - 2 * omega) / (4 * (alpha + 1) * (alpha + 2))


def mfp_vss(nrho, omega=AR_OMEGA, alpha=AR_ALPHA, diam=AR_DIAM):
    """Bird eqn (4.52), the mean free path his shock thicknesses are quoted in.

    Bird defines it through the viscosity,

        lambda = 2 mu (5-2w)(7-2w) / [15 rho sqrt(2 pi R T)]

    and substituting the VSS viscosity (Bird 4.64) collapses the whole ``omega``
    dependence, leaving

        lambda_4.52 = [(a+1)(a+2) / (6a)] * 1/(sqrt(2) pi d_ref^2 n)

    -- a function of ``alpha`` **only**, equal to the hard-sphere value at
    ``a = 1``.  Verified three ways in ``scratch/``: it reduces to
    ``1/(sqrt(2) pi d^2 n)`` exactly at ``w = 1/2, a = 1``; the closed form matches
    the viscosity route to six figures at every ``(w, a)`` tried; and it reproduces
    Bird's stated "76 % of the hard sphere mean free path" at his
    ``w = 0.81, a = 1.6625``.

    **The trap.**  This used to be ``cross_section_ratio * mfp_hard_sphere``, which
    is 1.2455x too small for argon and made every shock read 24 % too thick against
    Bird -- while still reproducing two numbers quoted in his text, because those
    numbers are the cross-section ratio (see :func:`cross_section_ratio`).  Two
    plausible formulas agreeing with the book at one point each is not
    verification; the check that caught it was deriving the mean free path from the
    measured viscosity instead.
    """
    del omega  # eqn (4.52) genuinely has no omega dependence -- see above
    return (alpha + 1.0) * (alpha + 2.0) / (6.0 * alpha) * mfp_hard_sphere(nrho, diam)


def mfp_alsmeyer(nrho, omega=AR_OMEGA, alpha=AR_ALPHA, diam=AR_DIAM):
    """The hard-sphere mean free path Alsmeyer (1976) normalises his data by.

    A hard-sphere gas matched to argon's *viscosity*, not to the VHS reference
    diameter: ``d_HS^2 = d_ref^2 * cross_section_ratio``, so

        lambda_HS = mfp_hard_sphere(d_ref) / cross_section_ratio

    which is **larger** than ``mfp_hard_sphere`` for argon (1.71e-3 vs 1.33e-3 at
    ``n = 1e21``).  Quoting a thickness against ``mfp_hard_sphere`` instead
    understates it by 28 %.
    """
    return mfp_hard_sphere(nrho, diam) / cross_section_ratio(omega, alpha)


def rankine_hugoniot(mach, gamma=5.0 / 3.0):
    """Density, pressure and temperature ratios across a normal shock."""
    m2 = mach * mach
    rho = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)
    p = 1.0 + 2.0 * gamma / (gamma + 1.0) * (m2 - 1.0)
    return {"rho": rho, "p": p, "temp": p / rho}


def piston_speed(mach, temp=AR_TREF, gamma=5.0 / 3.0):
    """Approach speed that produces a shock of Mach number ``mach`` off a wall."""
    a1 = math.sqrt(gamma * K_BOLTZ * temp / AR_MASS)
    r = rankine_hugoniot(mach, gamma)["rho"]
    return mach * a1 * (1.0 - 1.0 / r), mach * a1 / r, a1


def max_slope_thickness(x, rho, rho1, rho2, smooth_cells=5):
    """``L = (rho_2 - rho_1) / max|d rho / dx|``, Bird's maximum-slope definition.

    The gradient is taken from a lightly smoothed profile.  DSMC scatter makes the
    raw cell-to-cell difference an estimate of the *noise* rather than of the
    gradient, and the maximum of a noisy derivative is biased high -- which reports
    the shock as artificially **thin**, in the direction that would flatter the
    result.  ``smooth_cells`` is fixed in cells and must stay small compared with the
    shock width in cells (reported by :func:`report`), so it cannot manufacture the
    answer; the sensitivity to it is printed so the reader can judge.
    """
    dx = float(x[1] - x[0])
    k = int(smooth_cells)
    kern = np.ones(k) / k
    sm = np.convolve(rho, kern, mode="same")
    grad = np.abs(np.gradient(sm, dx))
    grad[:k] = 0.0
    grad[-k:] = 0.0
    if grad.max() <= 0:
        return math.inf, 0
    return abs(rho2 - rho1) / grad.max(), int(np.argmax(grad))


def tanh_thickness(x, rho, rho1, rho2, fit_halfwidth=12.0, lam=1.0):
    """Thickness from a ``tanh`` fit -- an estimator with no smoothing parameter.

    :func:`max_slope_thickness` has to smooth before differentiating, and the
    answer then depends on how much: Monte-Carlo noise biases the maximum of a
    noisy derivative *high* (reporting the wave too thin), while smoothing biases
    it *low* (too thick).  Extrapolating the smoothing to zero handles that, but it
    is an extrapolation.

    A shock profile is close to

        rho_n(x) = 0.5 * (1 - tanh((x - x0) / delta))

    for which ``max|d rho_n/dx| = 0.5/delta`` exactly, so ``L = 2 delta``.  Fitting
    ``delta`` uses every point in the wave at once, needs no derivative and no
    smoothing, and averages the noise instead of chasing its maximum.  Agreement
    between the two estimators is the evidence that the number is a property of the
    profile rather than of the method; disagreement says the profile is not
    tanh-like, which is itself worth knowing (Bird notes real profiles are
    asymmetric, increasingly so at high Mach).

    Returns ``(L, delta, rms_residual)``, ``L`` in the same units as ``x``.
    """
    from scipy.optimize import curve_fit  # noqa: PLC0415

    rn = (np.asarray(rho) - rho1) / (rho2 - rho1)
    xx = np.asarray(x, dtype=float)
    m = np.abs(xx) < fit_halfwidth * lam
    if m.sum() < 8:
        return math.nan, math.nan, math.nan

    def model(xv, x0, delta):
        return 0.5 * (1.0 - np.tanh((xv - x0) / delta))

    try:
        popt, _ = curve_fit(model, xx[m], rn[m], p0=[0.0, 2.0 * lam], maxfev=20000)
    except Exception:
        return math.nan, math.nan, math.nan
    resid = float(np.sqrt(np.mean((rn[m] - model(xx[m], *popt)) ** 2)))
    delta = abs(float(popt[1]))
    return 2.0 * delta, delta, resid


def normalised(profile, v1, v2):
    """Bird eqn (12.32): ``(q - q_1) / (q_2 - q_1)``, so every field runs 0 -> 1."""
    return (profile - v1) / (v2 - v1)


def setup(
    mach=2.0,
    nrho=1e21,
    temp=AR_TREF,
    cells_per_mfp=4.0,
    domain_mfp=320.0,
    npc=150,
    cfl=0.15,
    a_max=64,
    dim=1,
    transverse_cells=4,
):
    """Build the reflected-shock case, sized in units of the upstream mean free path.

    Everything is expressed in ``lambda_1`` because that is what Bird's result is
    quoted in: the domain is ``domain_mfp`` mean free paths long and each cell is
    ``1/cells_per_mfp`` of one, so a 6-mfp shock is resolved by ~24 cells whatever
    the density.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.collide import make_prefactor  # noqa: PLC0415
    from jax_dsmc.collide.ntc import vremax_init  # noqa: PLC0415
    from jax_dsmc.grid import num_cells  # noqa: PLC0415
    from jax_dsmc.state import (  # noqa: PLC0415
        CollideState,
        GridSpec,
        SimParams,
        SimStatic,
        SpeciesParams,
    )

    lam1 = mfp_vss(nrho)
    length = domain_mfp * lam1
    n_x = round(domain_mfp * cells_per_mfp)
    n = (n_x,) + (transverse_cells,) * (dim - 1)

    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2.0]])
    diam = jnp.array([[AR_DIAM]])
    omega = jnp.array([[AR_OMEGA]])
    tref = jnp.array([[AR_TREF]])
    alpha = jnp.array([[AR_ALPHA]])
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=omega,
        tref=tref,
        alpha=alpha,
        mr=mr,
        prefactor=make_prefactor(diam, omega, tref, mr),
    )

    # The transverse box is sized so a cell is CUBIC (dx the same on every axis):
    # a shock is a 1-D structure, and the only role of the extra dimensions is to
    # exercise the D-generic code paths on cells of the same shape as the 1-D run.
    trans_len = (length / n_x) * transverse_cells
    hi = jnp.array([length] + [trans_len] * (dim - 1))
    grid = GridSpec(
        lo=jnp.zeros(dim),
        hi=hi,
        n=n,
        transverse=jnp.float64(trans_len ** (3 - dim)),
    )
    C = num_cells(grid)
    vth = math.sqrt(K_BOLTZ * temp / AR_MASS)
    dx = length / n_x
    dt = cfl * dx / vth
    u_p, u_s, a1 = piston_speed(mach, temp)
    # `npc` is particles per CELL, so the total follows C -- not n_x.  Sizing off
    # n_x instead silently divides the per-cell count by `transverse_cells**(D-1)`
    # in 2-D and 3-D, which shows up as a thicker, noisier wave rather than as an
    # error, and would make the cross-dimension comparison meaningless.
    n_part = C * npc
    # D real transverse cells plus (3-D) suppressed ones multiply back to the same
    # physical box whatever D is, so `nrho` reads identically across dimensions.
    volume = length * trans_len**2
    fnum = jnp.full((C,), nrho * volume / n_part)

    static = SimStatic(N=n_part, D=dim, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=1)
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((C,), temp), omega[0, 0], tref[0, 0], mr[0, 0], sp.prefactor[0, 0], C, 1
        )
    )
    rh = rankine_hugoniot(mach)
    mct = lam1 / vth
    # The far boundary is not passive.  The gas streams AWAY from x = L, so an
    # expansion fan runs back from it at u_p + a_1 -- FASTER than the shock, so it
    # always catches up eventually.  Everything must be measured before it does.
    t_meet = length / (u_p + a1 + u_s)
    info = {
        "mach": mach,
        "nrho": nrho,
        "temp": temp,
        "lam1": lam1,
        "lam1_hs": mfp_alsmeyer(nrho, AR_OMEGA, AR_ALPHA),
        "L": length,
        "n": n,
        "C": C,
        "N": n_part,
        "dx": dx,
        "dt": dt,
        "u_p": u_p,
        "u_s": u_s,
        "a1": a1,
        "npc": npc,
        "vth": vth,
        "mct": mct,
        "dt_over_mct": dt / mct,
        "cells_per_mfp": lam1 / dx,
        "rho2_over_rho1": rh["rho"],
        "t2_over_t1": rh["temp"],
        "dim": dim,
        "t_meet": t_meet,
        "steps_to_meet": int(t_meet / dt),
    }
    return sp, grid, static, params, cstate, info


def run(  # noqa: PLR0915
    mach=2.0, seeds=4, settle_mfp=25.0, sample_mfp=25.0, verbose=False, sample_stride=1, **kw
):
    """Run the reflected shock and return shock-frame-averaged profiles.

    Two phases, following Bird: the wave is allowed to travel ``settle_mfp`` mean
    free paths so its structure reaches steady state, then profiles are sampled over
    a further ``sample_mfp``.  Because the wave is *moving*, each sample is shifted
    to its own measured shock position before averaging -- averaging in the lab frame
    would smear the profile by exactly the distance travelled, which is the effect
    Bird's ``STABIL`` routine exists to prevent in his stationary-wave set-up.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.boundary import BC  # noqa: PLC0415
    from jax_dsmc.simulate import run_adaptive  # noqa: PLC0415
    from jax_dsmc.state import ParticleState  # noqa: PLC0415

    sp, grid, static, params, cstate, info = setup(mach=mach, **kw)
    import dataclasses  # noqa: PLC0415

    n_settle = math.ceil(settle_mfp * info["lam1"] / info["u_s"] / info["dt"])
    n_sample = math.ceil(sample_mfp * info["lam1"] / info["u_s"] / info["dt"])
    static = dataclasses.replace(static, n_steps=n_settle + n_sample)
    info["steps"] = static.n_steps
    info["n_settle"] = n_settle
    # Snapshots from every seed are concatenated on return.  The linear
    # shock-position fit in `shock_frame_average` MUST know where one seed's time
    # series ends: fitting a single line through the concatenation sees the
    # position sawtooth back to its starting value at each seed boundary, which
    # drags the fitted shock speed down by roughly the number of seeds.
    info["seeds"] = seeds
    # Snapshots are kept every `sample_stride` steps.  A long run cannot keep all
    # of them: `run_adaptive` concatenates the per-step CellMoments, whose pressure
    # tensor alone is `steps * C * 9 * 8` bytes -- 1.2 GB for a 4500-step, 3600-cell
    # run, which is an out-of-memory on a 4 GB card.  The shock profile needs only
    # nrho, temp and u_x, and nothing about a shock changes in one timestep.
    n_snap = static.n_steps // sample_stride
    n_settle_snap = math.ceil(n_settle / sample_stride)
    info["sample_stride"] = sample_stride
    info["snap_dt"] = info["dt"] * sample_stride
    info["n_per_seed"] = n_snap - n_settle_snap
    # The settling curve (one long run sliced by travel distance) shows L flat from
    # about 20 mean free paths onward; below that the wave is still forming and
    # reads thin.  Bird settles for "approximately 20 shock widths".
    if settle_mfp < 15.0:
        print(
            f"  ** settle_mfp = {settle_mfp:.0f} is short: the shock structure is not steady "
            "below ~15-20 mean free paths, and the thickness will read low. **"
        )
    if static.n_steps >= info["steps_to_meet"]:
        raise RuntimeError(
            f"the expansion from the far boundary reaches the shock at step "
            f"{info['steps_to_meet']}, before the run ends at {static.n_steps}.  It "
            f"travels at u_p + a_1 = {info['u_p'] + info['a1']:.0f} m/s against the "
            f"shock's {info['u_s']:.0f} m/s, so it ALWAYS catches up -- lengthen the "
            f"domain (domain_mfp) or shorten settle_mfp + sample_mfp."
        )

    # `run_adaptive` applies the stride within each chunk, so the chunk length must
    # be a whole number of strides or the kept steps drift relative to the clock.
    chunk = max(32, static.n_steps // 8)
    chunk = max(sample_stride, chunk - chunk % sample_stride)

    prof_r, prof_t, prof_u = [], [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(4000 + s)
        # uniform gas streaming at -u_p into the specular wall at x = 0
        x = jax.random.uniform(key, (static.N, static.D), dtype=jnp.float64) * jnp.asarray(
            np.asarray(grid.hi)
        )
        v = info["vth"] * jax.random.normal(
            jax.random.fold_in(key, 1), (static.N, 3), dtype=jnp.float64
        )
        v = v.at[:, 0].add(-info["u_p"])
        state = ParticleState(
            x=x,
            v=v,
            species=jnp.zeros((static.N,), jnp.int32),
            alive=jnp.ones((static.N,), bool),
            logh_acc=jnp.zeros((static.N,), jnp.float64),
        )
        (_fs, _fc), (mom, diag), _h = run_adaptive(
            state,
            cstate,
            grid,
            sp,
            params,
            static,
            jax.random.fold_in(key, 2),
            bc=BC.SPECULAR,
            chunk=chunk,
            sample_stride=sample_stride,
            verbose=verbose,
        )
        info["clipped"] = int(jnp.sum(diag.attempts_clipped))
        info["violations"] = int(jnp.sum(diag.majorant_violations))
        nn = info["n"]
        ns = n_settle_snap
        r = x_profile(np.asarray(mom.nrho)[ns:], nn)
        t = x_profile(np.asarray(mom.temp)[ns:], nn)
        u = x_profile(np.asarray(mom.u[:, :, 0])[ns:], nn)
        prof_r.append(r)
        prof_t.append(t)
        prof_u.append(u)
    return (np.concatenate(prof_r), np.concatenate(prof_t), np.concatenate(prof_u), info)


def x_profile(field, n):
    """Average a per-cell field down to a profile along x.

    Cells are raveled row-major with **x on axis 0** (``grid._row_major_strides``),
    so ``reshape(n)`` puts the transverse axes last and they average away.  The
    shock is a 1-D structure; the transverse cells exist only to exercise the
    D-generic paths, and averaging over them is what makes the 2-D and 3-D profiles
    directly comparable with the 1-D one at equal ``npc``.
    """
    lead = field.shape[:-1]
    return field.reshape(*lead, *n).mean(axis=tuple(range(len(lead) + 1, len(lead) + len(n))))


def shock_position_crossing(r, xc, half, dx):
    """Locate the shock at the outermost half-density crossing.  ONE cell's noise.

    Kept only as a comparison for :func:`shock_position_integral` -- it is what a
    first implementation naturally does, and it is *biased*: the estimate is pinned
    to a single interpolated crossing, so the ensemble average has an artificial
    cusp exactly at ``xi = 0``, which is exactly where ``max|d rho/dx|`` is read.
    """
    idx = np.where((r[:-1] >= half) & (r[1:] < half))[0]
    if idx.size == 0:
        return None
    i = idx[-1]
    f = (r[i] - half) / max(r[i] - r[i + 1], 1e-30)
    return xc[i] + f * dx


def shock_position_integral(r, xc, n1, n2, dx, window=None):
    """Locate the shock by the equal-area (integral) rule.  ALL cells' information.

    If ``rn = (rho - rho_1)/(rho_2 - rho_1)`` were a perfect step down at ``x0`` on a
    window ``[a, b]``, then ``int_a^b rn dx = x0 - a`` exactly.  So

        x0 = a + int_a^b rn dx

    is an unbiased estimator that uses every cell in the window rather than the two
    cells straddling one level.  Its variance falls like the number of cells in the
    window instead of staying at the single-crossing value, and -- the reason it
    matters here -- it introduces no cusp, because nothing forces the averaged
    profile to pass through the half level at ``xi = 0``.

    **The window is the whole trap.**  The identity above holds only if ``rn == 1``
    at the window's left edge and ``rn == 0`` at its right, so the window must
    bracket the wave.  Trimming it as a fraction of the *domain* is wrong: this
    problem runs the wave a few tens of mean free paths into a domain hundreds of
    mean free paths long (the far boundary has to be kept away, see :func:`setup`),
    so a domain-relative window misses the shock entirely and the estimator
    silently returns the window's left edge -- a constant, whose fitted "shock
    speed" is zero.  :func:`shock_frame_average` therefore locates the wave
    roughly first and centres the window on it.
    """
    rn = np.clip((r - n1) / (n2 - n1), 0.0, 1.0)
    lo, hi = (0, rn.size) if window is None else window
    return xc[lo] - 0.5 * dx + float(np.sum(rn[lo:hi])) * dx


def shock_frame_average(rho, temp, ux, info, align="fit", trim=15.0):  # noqa: PLR0915
    """Shift every snapshot to the shock position, then average.

    ``align`` selects the estimator, and the choice is *not* cosmetic -- it moved the
    measured thickness by more than 20 %:

    ``"crossing"``
        each snapshot shifted to its own half-density crossing.  Biased: see
        :func:`shock_position_crossing`.
    ``"integral"``
        each snapshot shifted to its own equal-area position.  Unbiased, but the
        per-snapshot position still carries Monte-Carlo noise, and random shift
        errors of rms ``s`` convolve the ensemble profile with a kernel of that
        width, which reports the shock **thicker** than it is.
    ``"fit"`` (default)
        the shock travels at a constant speed, so ``x_0(t)`` is a straight line.
        Fitting that line through all the snapshots and shifting by the *fitted*
        position removes essentially all of the per-snapshot noise -- the residual
        scatter is reduced by ``sqrt(n_samples)``.  The fitted slope is also a free
        validation quantity: it is the shock speed, and Rankine-Hugoniot predicts it
        independently (reported as ``u_s_fit`` vs ``info["u_s"]``).

    ``trim`` is the half-width, **in mean free paths**, of the integration window
    centred on a rough (crossing-based) shock location.  It must be comfortably
    wider than the wave and narrower than the distance to the wall.
    """
    n1 = info["nrho"]
    n2 = n1 * info["rho2_over_rho1"]
    half = 0.5 * (n1 + n2)
    nsnap, nx = rho.shape
    xc = (np.arange(nx) + 0.5) * info["dx"]
    dx = info["dx"]
    half_w = round(trim * info["lam1"] / dx)  # window half-width, in cells

    pos = np.full(nsnap, np.nan)
    for k in range(nsnap):
        rough = shock_position_crossing(rho[k], xc, half, dx)
        if rough is None:
            continue
        if align == "crossing":
            pos[k] = rough
            continue
        c = int(rough / dx)
        lo_i = max(0, c - half_w)
        hi_i = min(nx, c + half_w)
        pos[k] = shock_position_integral(rho[k], xc, n1, n2, dx, window=(lo_i, hi_i))
    ok = np.isfinite(pos)
    if not ok.any():
        raise RuntimeError("no snapshot yielded a shock position")

    per = int(info.get("n_per_seed", nsnap))
    if align == "fit" and per >= 3:
        t = np.arange(per) * info.get("snap_dt", info["dt"])
        slopes, resids = [], []
        for s0 in range(0, nsnap, per):
            sl = slice(s0, min(s0 + per, nsnap))
            m = ok[sl]
            if m.sum() < 3:
                continue
            tt = t[: sl.stop - sl.start]
            slope, icept = np.polyfit(tt[m], pos[sl][m], 1)
            fitted = icept + slope * tt
            resids.append(float(np.std(pos[sl][m] - fitted[m])))
            slopes.append(float(slope))
            pos[sl] = fitted
            ok[sl] = True
        # wall at x = 0, gas approaches along -x, so the wave runs towards +x
        info["u_s_fit"] = float(np.mean(slopes)) if slopes else float("nan")
        info["align_resid_mfp"] = float(np.mean(resids)) / info["lam1"] if resids else float("nan")
    else:
        info["u_s_fit"] = float("nan")
        info["align_resid_mfp"] = float("nan")
    info["align"] = align

    grid_out = np.arange(-40.0, 40.0, 0.25)  # in units of lambda_1
    acc_r = np.zeros_like(grid_out)
    acc_t = np.zeros_like(grid_out)
    acc_u = np.zeros_like(grid_out)
    used = 0
    for k in range(nsnap):
        if not ok[k]:
            continue
        xi = (xc - pos[k]) / info["lam1"]
        acc_r += np.interp(grid_out, xi, rho[k])
        acc_t += np.interp(grid_out, xi, temp[k])
        acc_u += np.interp(grid_out, xi, ux[k])
        used += 1
    return grid_out, acc_r / used, acc_t / used, acc_u / used, used


def report(xi, rho, temp, ux, info, used):  # noqa: PLR0915
    """Print Bird's measurements and compare with his published values."""
    n1 = info["nrho"]
    n2 = n1 * info["rho2_over_rho1"]
    t1 = info["temp"]
    t2 = t1 * info["t2_over_t1"]

    print(f"=== Bird section 12.11 normal shock, argon, Mach {info['mach']:.2f} ===")
    print(
        f"  {info['C']} cells, N = {info['N']:,} ({info['npc']} per cell), "
        f"{info['steps']} steps, {used} shock-frame samples"
    )
    print(
        f"  lambda_1 (VSS, Bird 4.52) = {info['lam1']:.4e} m   "
        f"lambda_1 (Alsmeyer hard sphere) = {info['lam1_hs']:.4e} m"
    )
    # The shock-frame average is only meaningful if the wave was actually tracked.
    # At low `npc` the half-density crossing drowns in per-cell noise, the linear
    # fit of x0(t) collapses, and every downstream number is a smear of the whole
    # domain -- which still PRINTS, and prints something plausible: npc = 30 gave
    # L/lambda = 20.7 with no error anywhere.  The fitted shock speed is the
    # honest check, because Rankine-Hugoniot predicts it independently.
    u_s_fit = info.get("u_s_fit", float("nan"))
    resid = info.get("align_resid_mfp", float("nan"))
    speed_err = abs(u_s_fit / info["u_s"] - 1.0) if info["u_s"] else float("inf")
    if not (speed_err < 0.05) or not (resid < 1.0):
        print(
            f"\n  ** SHOCK TRACKING FAILED **  fitted speed {u_s_fit:.0f} m/s against "
            f"{info['u_s']:.0f} from Rankine-Hugoniot, alignment residual {resid:.2f} "
            "lambda_1 (needs < 1).\n     Every number below is a smear of the whole "
            "domain, not a shock profile.  Raise npc (>= ~120 at this resolution)."
        )

    print("\n  resolution (Bird's own requirements):")
    print(f"    cells per lambda_1     = {info['cells_per_mfp']:.2f}   (needs >~ 3)")
    print(f"    dt / mean collision t  = {info['dt_over_mct']:.4f}   (Bird uses ~0.05)")
    print(f"    clipped {info['clipped']}   majorant violations {info['violations']}")
    print(
        f"    shock speed (fitted)   = {u_s_fit:.1f} m/s vs {info['u_s']:.1f} exact"
        f"   alignment residual {resid:.3f} lambda_1"
    )

    print("\n  Rankine-Hugoniot (the state the wave must connect):")
    # xi < 0 is BEHIND the shock (compressed, at rest against the wall) = state 2;
    # xi > 0 is the undisturbed oncoming stream = state 1.  Average over plateaus a
    # few mfp clear of the wave rather than reading the array ends, which sit near
    # the domain boundaries where the profile is not the shock's business.
    dn = (xi > -25) & (xi < -10)
    up = (xi > 10) & (xi < 25)
    r_meas = float(rho[dn].mean() / rho[up].mean())
    t_meas = float(temp[dn].mean() / temp[up].mean())
    print(
        f"    rho2/rho1  exact {info['rho2_over_rho1']:.4f}   measured {r_meas:.4f}"
        f"   ({abs(r_meas / info['rho2_over_rho1'] - 1) * 100:.2f} %)"
    )
    print(
        f"    T2/T1      exact {info['t2_over_t1']:.4f}   measured {t_meas:.4f}"
        f"   ({abs(t_meas / info['t2_over_t1'] - 1) * 100:.2f} %)"
    )
    u_meas = float(ux[up].mean())
    print(
        f"    u upstream exact {-info['u_p']:.1f}   measured {u_meas:.1f} m/s"
        f"   ({abs(u_meas / -info['u_p'] - 1) * 100:.2f} %)"
    )

    thick, _i_max = max_slope_thickness(xi * info["lam1"], rho, n1, n2)
    l_vss = thick / info["lam1"]
    l_hs = thick / info["lam1_hs"]
    print("\n  maximum-slope thickness  L = (rho2-rho1) / max|drho/dx| :")
    print(f"    L / lambda_1 (VSS)  = {l_vss:6.2f}     Bird 12.11 DSMC:  {BIRD_L_OVER_LAMBDA_VSS}")
    print(
        f"    L / lambda_1 (HS)   = {l_hs:6.2f}     Alsmeyer 1976:    {ALSMEYER_L_OVER_LAMBDA_HS}"
    )
    print(
        f"    -> {abs(l_vss - BIRD_L_OVER_LAMBDA_VSS) / BIRD_L_OVER_LAMBDA_VSS * 100:.1f} % "
        f"from Bird, {abs(l_hs - ALSMEYER_L_OVER_LAMBDA_HS) / ALSMEYER_L_OVER_LAMBDA_HS * 100:.1f} % "
        "from Alsmeyer"
    )
    span = l_vss * info["cells_per_mfp"]
    if span >= 20:
        print(f"    (shock spans {span:.0f} cells; the 5-cell smoothing is well inside that)")
    else:
        print(
            f"    ** the shock spans only {span:.0f} cells, so the 5-cell smoothing is a "
            f"large fraction of it and biases L upward.  Raise --cells-per-mfp. **"
        )

    rn = normalised(rho, n1, n2)
    tn = normalised(temp, t1, t2)
    # Both normalised profiles DECREASE with xi (state 2 is at negative xi), and
    # `np.interp` requires an increasing x-coordinate -- passing them as-is silently
    # returns nonsense, which is exactly what a separation of precisely 0.00 was.
    x_r = float(np.interp(0.5, rn[::-1], xi[::-1]))
    x_t = float(np.interp(0.5, tn[::-1], xi[::-1]))
    print(
        f"\n  temperature leads density by {x_t - x_r:.2f} lambda_1  "
        f"(T half-point at {x_t:+.2f}, rho half-point at {x_r:+.2f})"
    )
    print("    (Bird: the temperature profile LEADS the density profile in every case)")
    return {
        "L_over_lam_vss": l_vss,
        "L_over_lam_hs": l_hs,
        "separation": x_t - x_r,
        "rho_ratio": r_meas,
        "temp_ratio": t_meas,
    }


def main():  # noqa: PLR0915
    """CLI: run Bird's normal shock and report against Bird and Alsmeyer."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dims", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--settle-mfp", type=float, default=25.0)
    ap.add_argument("--sample-mfp", type=float, default=40.0)
    ap.add_argument("--domain-mfp", type=float, default=320.0)
    ap.add_argument("--cells-per-mfp", type=float, default=4.0)
    ap.add_argument("--npc", type=int, default=120, help="particles per cell; >= ~120 is needed")
    ap.add_argument("--transverse-cells", type=int, default=3)
    ap.add_argument("--a-max", type=int, default=160)
    ap.add_argument("--sample-stride", type=int, default=4)
    ap.add_argument("--align", default="fit", choices=["fit", "integral", "crossing"])
    ap.add_argument("--plot", default=None, help="write a figure to this path")
    a = ap.parse_args()

    out = {}
    for dim in a.dims:
        rho, temp, ux, info = run(
            mach=a.mach,
            seeds=a.seeds,
            settle_mfp=a.settle_mfp,
            sample_mfp=a.sample_mfp,
            domain_mfp=a.domain_mfp,
            cells_per_mfp=a.cells_per_mfp,
            npc=a.npc,
            dim=dim,
            transverse_cells=a.transverse_cells,
            a_max=a.a_max,
            sample_stride=a.sample_stride,
            verbose=True,
        )
        xi, r, t, u, used = shock_frame_average(rho, temp, ux, info, align=a.align)
        out[dim] = (xi, r, t, u, info, report(xi, r, t, u, info, used))
        n1 = info["nrho"]
        n2 = n1 * info["rho2_over_rho1"]
        lt, _, res = tanh_thickness(xi, r, n1, n2, lam=1.0)
        print(f"    tanh fit (no smoothing):  L / lambda_1 = {lt:.2f}   (rms residual {res:.4f})")
        print()

    if a.plot:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        from jax_dsmc.viz import INK, make_figure, savefig, style_axes, time_color  # noqa: PLC0415

        fig, axes = make_figure(1, 1, figsize=(6.2, 4.4))
        ax = axes[0]
        for i, dim in enumerate(a.dims):
            xi, r, t, _u, info, _res = out[dim]
            n1 = info["nrho"]
            n2 = n1 * info["rho2_over_rho1"]
            t1 = info["temp"]
            t2 = t1 * info["t2_over_t1"]
            m = np.abs(xi) < 12
            c = time_color(i, max(len(a.dims), 2))
            ax.plot(xi[m], normalised(r, n1, n2)[m], color=c, lw=1.6, label=f"D = {dim}")
            ax.plot(xi[m], normalised(t, t1, t2)[m], color=c, lw=1.1, ls="--")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\xi = (x - x_{\rm shock}) / \lambda_1$")
        ax.set_ylabel(r"normalised $\rho$ (solid), $T$ (dashed)")
        ax.set_title(f"Bird Fig. 12.30 form: argon, Mach {a.mach}", color=INK)
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
        print(savefig(fig, a.plot, dpi=150))
    return out


if __name__ == "__main__":
    main()
