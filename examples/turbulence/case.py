"""Set up and run a decaying-turbulence case in D in {1, 2, 3}."""

from __future__ import annotations

import dataclasses
import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.shock.normal import (  # noqa: E402
    AR_ALPHA,
    AR_DIAM,
    AR_MASS,
    AR_OMEGA,
    AR_TREF,
    mfp_vss,
)
from examples.turbulence import (  # noqa: E402
    broadband,
    kinematic_viscosity,
    sound_speed,
    taylor_green,
)
from jax_dsmc.constants import K_BOLTZ  # noqa: E402


def setup(
    dim=2,
    cells_per_side=64,
    cells_per_mfp=1.0,
    npc=100,
    mach=0.3,
    temp=AR_TREF,
    nrho=1e21,
    cfl=0.15,
    a_max=64,
    n_steps=1,
    ic="taylor-green",
    k_max=4,
    ic_phase_seed=20260905,
):
    """Build a periodic Taylor-Green box.

    The box is sized in mean free paths -- ``cells_per_side / cells_per_mfp`` of
    them -- because every dimensionless group that matters (Reynolds number, Knudsen
    number, the signal-to-noise ratio of the velocity field) is set by that ratio
    and by ``npc``, not by the absolute density.
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

    lam = mfp_vss(nrho)
    box = cells_per_side / cells_per_mfp * lam
    n = (cells_per_side,) * dim
    C = num_cells(GridSpec(lo=jnp.zeros(dim), hi=jnp.full(dim, box), n=n))

    sp = SpeciesParams(
        mass=jnp.array([AR_MASS]),
        diam=jnp.array([[AR_DIAM]]),
        omega=jnp.array([[AR_OMEGA]]),
        tref=jnp.array([[AR_TREF]]),
        alpha=jnp.array([[AR_ALPHA]]),
        mr=jnp.array([[AR_MASS / 2.0]]),
        prefactor=make_prefactor(
            jnp.array([[AR_DIAM]]),
            jnp.array([[AR_OMEGA]]),
            jnp.array([[AR_TREF]]),
            jnp.array([[AR_MASS / 2.0]]),
        ),
    )
    grid = GridSpec(
        lo=jnp.zeros(dim),
        hi=jnp.full(dim, box),
        n=n,
        transverse=jnp.float64(box ** (3 - dim)),
    )
    vth = math.sqrt(K_BOLTZ * temp / AR_MASS)
    a1 = sound_speed(temp)
    u0 = mach * a1
    dx = box / cells_per_side
    dt = cfl * dx / vth
    n_part = C * npc
    fnum = jnp.full((C,), nrho * box**3 / n_part)

    nu = kinematic_viscosity(temp, nrho)
    kbox = 2.0 * math.pi / box
    ell = box / (2.0 * math.pi)  # Taylor-Green length scale
    static = SimStatic(N=n_part, D=dim, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=n_steps)
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((C,), temp),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            C,
            1,
        )
    )
    # Per-cell velocity noise is v_th/sqrt(npc) per component, and it is WHITE --
    # flat in k -- while the signal is concentrated at k = 1.  This ratio is the
    # single number that decides whether a spectrum is measurable.
    snr = u0 / (vth / math.sqrt(npc))
    info = {
        "dim": dim,
        "box": box,
        "box_mfp": box / lam,
        "lam": lam,
        "n": n,
        "C": C,
        "N": n_part,
        "npc": npc,
        "dx": dx,
        "dx_over_mfp": dx / lam,
        "dt": dt,
        "temp": temp,
        "nrho": nrho,
        "vth": vth,
        "a1": a1,
        "u0": u0,
        "mach": mach,
        "nu": nu,
        "kbox": kbox,
        "ell": ell,
        "Re": u0 * ell / nu,
        "Kn": lam / ell,
        "t_eddy": ell / u0,
        "tau_visc": 1.0 / (2.0 * nu * kbox * kbox),
        "snr_per_cell": snr,
        "mct": lam / vth,
        "dt_over_mct": dt / (lam / vth),
        "ic": ic,
        "k_max": k_max,
        "ic_phase_seed": ic_phase_seed,
    }
    return sp, grid, static, params, cstate, info


def initial_particles(key, grid, static, info):
    """Uniform positions, Maxwellian plus the chosen mean field.

    ``info["ic"]`` selects it.  ``"taylor-green"`` is the default and is what the
    *viscosity* gate needs -- in 2-D it is an exact Navier-Stokes solution, which is
    the entire basis of that measurement.  ``"broadband"`` is what a *spectrum* needs:
    a single Fourier mode imprints its own symmetry on ``E(k)`` (measured: a factor-12
    odd/even comb in 1-D, at shells 90x clear of the noise floor), and no shell slope
    can be fitted through that.  See :func:`examples.turbulence.broadband` and spec
    22.4.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from jax_dsmc.state import ParticleState  # noqa: PLC0415

    x = jax.random.uniform(key, (static.N, static.D), dtype=jnp.float64) * grid.hi
    v = info["vth"] * jax.random.normal(
        jax.random.fold_in(key, 1), (static.N, 3), dtype=jnp.float64
    )
    if info.get("ic", "taylor-green") == "broadband":
        # The mode phases come from `ic_phase_seed`, NOT from `key`, so every seed
        # starts from the SAME mean field and differs only in thermal and collision
        # noise.  That is the premise `run_ensemble` rests on: derive the phases from
        # the run key instead and averaging M realisations averages M different flows,
        # driving the mean field toward zero while every diagnostic still reports
        # cleanly.  `test_broadband_ic_is_identical_across_seeds` pins it.
        v = v + broadband(
            x,
            info["u0"],
            info["kbox"],
            static.D,
            jax.random.PRNGKey(info["ic_phase_seed"]),
            k_max=info["k_max"],
        )
    else:
        v = v + taylor_green(x, info["u0"], info["kbox"], static.D)
    return ParticleState(
        x=x,
        v=v,
        species=jnp.zeros((static.N,), jnp.int32),
        alive=jnp.ones((static.N,), bool),
        logh_acc=jnp.zeros((static.N,), jnp.float64),
    )


def run(
    dim=2,
    seed=0,
    n_eddy=4.0,
    n_steps=None,
    sample_stride=1,
    frame_every=None,
    movie_dir=None,
    planes=None,
    verbose=True,
    **kw,
):
    """Run one case, returning per-step moments plus the info dict.

    ``frame_every`` writes a field PNG every that many steps (see
    :class:`jax_dsmc.viz.FieldMovie`); the frames are named so that

        ffmpeg -framerate 20 -pattern_type glob -i 'field_*.png' out.gif

    stitches them in order.  Images are written **host-side, outside the compiled
    step**, so they cost only the frames actually requested.
    """
    from jax_dsmc.boundary import BC  # noqa: PLC0415
    from jax_dsmc.simulate import run_adaptive  # noqa: PLC0415

    sp, grid, static, params, cstate, info = setup(dim=dim, **kw)
    if n_steps is None:
        n_steps = max(1, math.ceil(n_eddy * info["t_eddy"] / info["dt"]))
    static = dataclasses.replace(static, n_steps=n_steps)
    # A full CellMoments per step is `steps * C * 18 * 8` bytes -- 3.2 GB for a
    # 1343-step run on a 128^2 grid, which does not fit on a 4 GB card.  Nothing
    # about a decaying vortex changes in one timestep, so keep every stride-th.
    chunk = max(16, n_steps // 12)
    chunk = max(sample_stride, chunk - chunk % sample_stride)
    info["sample_stride"] = sample_stride
    info["snap_dt"] = info["dt"] * sample_stride
    info["steps"] = n_steps
    info["n_eddy"] = n_eddy
    if verbose:
        summarise(info)

    state = initial_particles(jax.random.PRNGKey(7000 + seed), grid, static, info)
    (_fs, _fc), (mom, diag), _h = run_adaptive(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        jax.random.fold_in(jax.random.PRNGKey(7000 + seed), 2),
        bc=BC.PERIODIC,
        chunk=chunk,
        sample_stride=sample_stride,
        verbose=verbose,
    )
    info["clipped"] = int(np.sum(np.asarray(diag.attempts_clipped)))
    info["violations"] = int(np.sum(np.asarray(diag.majorant_violations)))
    if movie_dir is not None and frame_every:
        write_frames(mom, info, movie_dir, frame_every, planes)
    return mom, info


def write_frames(mom, info, directory, every, planes=None):
    """Write the field PNGs for a completed run."""
    from jax_dsmc.viz import PLANES_MID, FieldMovie  # noqa: PLC0415

    n = info["n"]
    pl = planes if planes is not None else (PLANES_MID if len(n) == 3 else None)
    movie = FieldMovie(
        directory, n=n, lo=np.zeros(len(n)), hi=np.full(len(n), info["box"]), planes=pl
    )
    u0 = info["u0"]
    movie.set_limits(
        {
            "u_x": (-u0, u0),
            "u_y": (-u0, u0),
            "nrho": (0.9 * info["nrho"], 1.1 * info["nrho"]),
            "temp": (0.98 * info["temp"], 1.06 * info["temp"]),
        }
    )
    ux = np.asarray(mom.u[:, :, 0])
    uy = np.asarray(mom.u[:, :, 1])
    nr = np.asarray(mom.nrho)
    tp = np.asarray(mom.temp)
    out = []
    for k in range(0, ux.shape[0], every):
        out += movie.add(
            k,
            {"u_x": ux[k], "u_y": uy[k], "nrho": nr[k], "temp": tp[k]},
            time=k * info["dt"],
        )
    return out


def summarise(info):
    ic = info.get("ic", "taylor-green")
    label = "Taylor-Green" if ic == "taylor-green" else f"broadband (|k| <= {info['k_max']})"
    print(f"=== {label}, D = {info['dim']}, argon ===")
    print(f"  grid {info['n']}  C = {info['C']:,}  N = {info['N']:,}  ({info['npc']} per cell)")
    print(
        f"  box = {info['box_mfp']:.1f} mfp   cell = {info['dx_over_mfp']:.2f} mfp   (needs <~ 1)"
    )
    print(f"  Ma = {info['mach']:.2f}   Re = {info['Re']:.1f}   Kn = {info['Kn']:.4f}")
    print(f"  dt / mean collision time = {info['dt_over_mct']:.4f}   (needs <~ 0.2)")
    print(f"  eddy time = {info['t_eddy']:.3e} s   viscous time = {info['tau_visc']:.3e} s")
    print(f"  per-cell velocity SNR u0/(vth/sqrt(npc)) = {info['snr_per_cell']:.2f}")
    if info["snr_per_cell"] < 2.0:
        print("    ** below ~2 the mean field is buried in per-cell noise; raise npc **")
    if ic == "taylor-green":
        print(
            "  IC is a single Fourier mode: right for the VISCOSITY gate (exact NS in 2-D),\n"
            "  wrong for a spectrum -- it combs E(k) with its own symmetry.  Use --ic broadband."
        )


class MeanFields:
    """Ensemble-averaged per-cell fields, shaped like :class:`CellMoments` in time.

    Carries the members the validators and the ``--save`` path read; it is
    deliberately not a ``CellMoments``, because the averaged object is a mean over
    realisations and not a moment of any single one.  ``nrho`` and ``temp`` are
    carried only so a completed ensemble can be saved and re-analysed -- omitting
    them cost one 42-minute run, which finished, printed its numbers and then died
    with ``AttributeError`` before writing anything.
    """

    def __init__(self, u, count, nrho=None, temp=None):
        self.u = u
        self.count = count
        self.nrho = nrho
        self.temp = temp


def run_ensemble(seeds=4, at_frac=1.0, full=False, **kw):
    """Run ``seeds`` independent realisations and average the VELOCITY FIELD.

    The initial condition is deterministic -- Taylor-Green, or a broadband field whose
    phases come from ``info["ic_phase_seed"]`` rather than from the run key -- so every
    seed evolves the same mean flow and differs only in its collision and
    initialisation noise.  Averaging ``M`` fields is therefore
    exactly equivalent to running one case at ``M * npc``, at identical total cost,
    with two practical advantages: peak memory stays that of a single run, and the
    realisations are completely independent, so they spread across a multi-GPU node
    with no communication at all.

    This is the right way to get a measurement that does not lean on a noise
    correction.  Subtracting the floor is safe on a scalar and **not** safe under a
    logarithm near the crossover (spec section 22); lowering the floor is safe
    everywhere.

    Args:
        seeds: number of realisations.
        at_frac: which point in the run to sample the field at, as a fraction of the
            total, when ``full`` is False.
        full: average the **whole time series** rather than one snapshot, so the
            energy-decay fit also sees the lowered floor.

    Returns ``(fields, info)`` where ``fields`` is a ``f64[C, 3]`` array when
    ``full`` is False and a :class:`MeanFields` when it is.  ``info["n_ensemble"]``
    is ``seeds``, which is what lets the validators lower the quoted floor.
    """
    acc_u = acc_c = acc_n = acc_t = None
    info = None
    clipped = violations = 0
    for s in range(seeds):
        mom, info = run(seed=s, verbose=(s == 0), **kw)
        # diagnostics are per-run; report the TOTAL, or a single bad seed hides
        clipped += info["clipped"]
        violations += info["violations"]
        u = np.asarray(mom.u)
        c = np.asarray(mom.count)
        n_r = np.asarray(mom.nrho) if full else None
        t_r = np.asarray(mom.temp) if full else None
        if not full:
            at = min(u.shape[0] - 1, max(0, round(at_frac * (u.shape[0] - 1))))
            u, c = u[at], c[at]
        acc_u = u if acc_u is None else acc_u + u
        acc_c = c if acc_c is None else acc_c + c
        if full:
            acc_n = n_r if acc_n is None else acc_n + n_r
            acc_t = t_r if acc_t is None else acc_t + t_r
        print(f"  seed {s} done")
    info["n_ensemble"] = seeds
    info["npc_effective"] = info["npc"] * seeds
    info["clipped"] = clipped
    info["violations"] = violations
    if full:
        return MeanFields(acc_u / seeds, acc_c / seeds, acc_n / seeds, acc_t / seeds), info
    return acc_u / seeds, info


def run_ic_ensemble(realisations=8, seeds=1, at_frac=1.0, **kw):
    """Run ``realisations`` INDEPENDENT flows and return their final fields.

    Returns ``(fields, info)`` with ``fields`` a list of ``f64[C, 3]``.

    This is the other ensemble, and in 1-D it is the one a spectrum needs.
    :func:`run_ensemble` averages *fields* across noise seeds of the **same** flow,
    which lowers the particle-noise floor and leaves the flow untouched.  This
    function varies ``ic_phase_seed``, so each realisation is a **different** flow
    drawn from the same statistics; averaging their spectra reduces the realisation
    scatter that dominates a 1-D ``E(k)`` (see
    :func:`examples.turbulence.validate.fit_slope`).

    The two compose: ``seeds > 1`` runs a noise ensemble inside each realisation.
    In 1-D that is usually wasted -- the floor is already far below the signal -- and
    ``realisations`` is where the budget should go.
    """
    base = kw.pop("ic_phase_seed", 20260905)
    fields, info = [], None
    clipped = violations = 0
    for r in range(realisations):
        kw_r = dict(kw, ic_phase_seed=base + 7919 * r)
        if seeds > 1:
            u, info = run_ensemble(seeds=seeds, at_frac=at_frac, full=False, **kw_r)
        else:
            mom, info = run(verbose=(r == 0), **kw_r)
            u = np.asarray(mom.u)
            at = min(u.shape[0] - 1, max(0, round(at_frac * (u.shape[0] - 1))))
            u = u[at]
        clipped += info["clipped"]
        violations += info["violations"]
        fields.append(np.asarray(u))
        print(f"  realisation {r} done")
    info["n_realisations"] = realisations
    info["n_ensemble"] = seeds
    info["clipped"] = clipped
    info["violations"] = violations
    return fields, info
