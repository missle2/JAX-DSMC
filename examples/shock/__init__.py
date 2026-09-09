"""Milestone 2: periodic transport and shock formation, in 1-D, 2-D and 3-D.

M1 exercised the collision operator alone.  This exercises **transport**: particles
advecting between cells under periodic boundaries, with collisions in the loop.

The physics.  A compressible gas given a smooth initial velocity
``u_x(x, 0) = U0 sin(2 pi x / L)``, uniform in the other directions, steepens: fast
fluid overtakes slow fluid, the profile becomes a sawtooth, and a shock forms at
``t_s = L / (2 pi U0)``.  Dissipation then limits the front to a finite thickness
of order the mean free path -- which is the DSMC result a continuum solver cannot
give you, since Navier-Stokes has no intrinsic thickness scale of its own.

**The acceptance gate is cross-dimensional agreement.**  The initial condition
varies only in ``x``, so the problem is one-dimensional physics whatever ``D`` is.
Run as ``D = 1``, ``D = 2`` and ``D = 3`` with the same ``dx`` and the same number
of particles per ``x``-slab, the measured ``u(x)`` profiles must agree to within
Monte Carlo error.  That single test covers a lot:

  * the ``GridSpec.transverse`` extent, without which ``cell_volume`` is a length
    at ``D = 1`` and an area at ``D = 2``, corrupting both the collision rate and
    ``nrho``;
  * advection and periodic wrapping in each dimension;
  * that the ``v[:, :D]`` slice in ``move`` really does leave the physics alone.

A second, physical check is that the front thickness comes out at a few mean free
paths and stays there as the density is varied -- i.e. it is set by the gas, not by
the grid.

Spectra (the decaying-turbulence ``E(k)`` batch) reuse this same driver with a
different initial condition and different post-processing; ``sample.energy_spectrum_1d``
is already in place for them.

Run from the repository root::

    python3 -m examples.shock --dims 1 2 3 --plot
"""

from __future__ import annotations

import argparse
import functools
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import make_prefactor  # noqa: E402
from jax_dsmc.collide.ntc import vremax_init  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.simulate import run_adaptive  # noqa: E402
from jax_dsmc.state import (  # noqa: E402
    CollideState,
    FlowParams,
    GridSpec,
    SimParams,
    SimStatic,
    SpeciesParams,
)

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15
FIGURE_DIR = pathlib.Path(__file__).resolve().parent / "figures"


def mean_free_path(nrho, diam=AR_DIAM):
    """VHS mean free path ``1 / (sqrt(2) pi d^2 n)``."""
    return 1.0 / (math.sqrt(2.0) * math.pi * diam * diam * nrho)


def sound_speed(temp=AR_TREF, mass=AR_MASS, gamma=5.0 / 3.0):
    return math.sqrt(gamma * K_BOLTZ * temp / mass)


def setup(
    dim=1,
    n_x=128,
    n_trans=2,
    npc_slab=400,
    nrho=1e23,
    temp=AR_TREF,
    length=1e-3,
    mach=2.0,
    cfl=0.2,
    a_max=64,
):
    """Build the same 1-D shock problem embedded in ``dim`` dimensions.

    ``n_x`` cells along x and ``n_trans`` along each of the ``dim - 1`` other axes.
    ``npc_slab`` is particles per **x-slab**, held fixed across ``dim`` so that the
    x-profile carries identical statistics however many transverse cells share it.

    ``a_max`` is only a starting guess: :func:`run_adaptive` grows it from the
    measured ``attempts_clipped``, so it never has to be right.
    """
    n_part = n_x * npc_slab
    n = (n_x,) + (n_trans,) * (dim - 1)

    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2.0]])
    diam = jnp.array([[AR_DIAM]])
    omega = jnp.array([[0.81]])
    tref = jnp.array([[AR_TREF]])
    alpha = jnp.array([[1.4]])
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=omega,
        tref=tref,
        alpha=alpha,
        mr=mr,
        prefactor=make_prefactor(diam, omega, tref, mr),
    )

    # The domain is always a physical 3-D box of side `length`.  Axes that carry
    # cells are described by lo/hi; the remaining `3 - dim` are folded into
    # `transverse`, so `cell_volume` is a true volume in every dimension.
    grid = GridSpec(
        lo=jnp.zeros(dim),
        hi=jnp.full(dim, length),
        n=n,
        transverse=jnp.float64(length ** (3 - dim)),
    )
    C = num_cells(grid)

    vth = math.sqrt(K_BOLTZ * temp / AR_MASS)
    dx = length / n_x
    dt = cfl * dx / vth
    u0 = mach * sound_speed(temp)
    mfp = mean_free_path(nrho)
    t_shock = length / (2.0 * math.pi * u0)
    steps = math.ceil(1.6 * t_shock / dt)

    fnum = jnp.full((C,), nrho * length**3 / n_part)
    static = SimStatic(N=n_part, D=dim, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=steps)
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    flow = FlowParams(nrho=jnp.float64(nrho), temp=jnp.float64(temp), vstream=jnp.zeros(3))
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((C,), temp), omega[0, 0], tref[0, 0], mr[0, 0], sp.prefactor[0, 0], C, 1
        )
    )
    info = {
        "dim": dim,
        "n": n,
        "C": C,
        "N": n_part,
        "npc_slab": npc_slab,
        "npc_cell": n_part // C,
        "dx": dx,
        "dt": dt,
        "mfp": mfp,
        "Kn": mfp / length,
        "mfp_over_dx": mfp / dx,
        "dt_over_mft": dt / (mfp / vth),
        "u0": u0,
        "mach": mach,
        "vth": vth,
        "t_shock": t_shock,
        "steps": steps,
        "L": length,
    }
    return sp, grid, static, params, flow, cstate, info


def sine_vstream(x, u0, length):
    """``u_x = U0 sin(2 pi x / L)``; uniform in every other direction."""
    ux = u0 * jnp.sin(2.0 * jnp.pi * x[:, 0] / length)
    return jnp.stack([ux, jnp.zeros_like(ux), jnp.zeros_like(ux)], axis=-1)


def x_profile(field, count, info):
    """Collapse a per-cell field to a profile along x, weighted by occupancy.

    Cell indices are row-major with ``x`` as the leading axis (``grid.n`` is
    ``(n_x, n_trans, ...)``), so a reshape puts x first and the transverse axes
    after it.  The average is **count-weighted**, not a plain mean: cells that
    happen to hold fewer particles carry a noisier velocity and must not be given
    equal say.
    """
    shape = tuple(info["n"])
    f = np.asarray(field).reshape(shape)
    w = np.asarray(count).reshape(shape)
    axes = tuple(range(1, len(shape)))
    if not axes:
        return f
    return (f * w).sum(axis=axes) / np.maximum(w.sum(axis=axes), 1e-30)


def run_case(dim, seeds=4, verbose=False, **kw):
    """Run the shock in ``dim`` dimensions; return the seed-averaged x-profiles."""
    from jax_dsmc import sample, sort  # noqa: PLC0415

    sp, grid, static, params, flow, cstate, info = setup(dim=dim, **kw)
    vs = functools.partial(sine_vstream, u0=info["u0"], length=info["L"])
    ux, rho, tmp, hist = [], [], [], []
    t0 = time.perf_counter()
    for s in range(seeds):
        key = jax.random.PRNGKey(500 + s)
        state = initial_state(key, flow, grid, sp, static, vstream_fn=vs)
        (fs, _), (_mom, diag), h = run_adaptive(
            state,
            cstate,
            grid,
            sp,
            params,
            static,
            jax.random.fold_in(key, 3),
            bc=BC.PERIODIC,
            chunk=max(8, info["steps"] // 4),
            verbose=verbose,
        )
        hist += h
        table = sort.build_table(fs, grid)
        m = sample.moments(fs, table, grid, params, sp)
        ux.append(x_profile(m.u[:, 0], m.count, info))
        rho.append(x_profile(m.nrho, m.count, info))
        tmp.append(x_profile(m.temp, m.count, info))
        info["clipped"] = int(jnp.sum(diag.attempts_clipped))
        info["violations"] = int(jnp.sum(diag.majorant_violations))
        info["peak_attempts"] = int(jnp.max(diag.max_attempts))
    info["wall_s"] = time.perf_counter() - t0
    info["resizes"] = len(hist)
    return np.array(ux), np.array(rho), np.array(tmp), info


def front_thickness(u_profile, info):
    """Front thickness in mean free paths, from ``(u_max - u_min) / max|du/dx|``.

    A sinusoid of the same amplitude gives ``L / pi`` by this measure, so a value
    well below that means the profile really has steepened; the DSMC answer should
    land at a few mean free paths.
    """
    du = np.abs(np.diff(u_profile, append=u_profile[:1]))
    grad = du.max() / info["dx"]
    if grad <= 0:
        return math.inf
    return (u_profile.max() - u_profile.min()) / grad / info["mfp"]


def report(info, ux, rho, tmp):
    """Print the case, the resolution checks, and the front measurement."""
    u = ux.mean(axis=0)
    r = rho.mean(axis=0)
    print(
        f"=== D = {info['dim']}   cells {info['n']} = {info['C']:,}   "
        f"N = {info['N']:,} ({info['npc_slab']} per x-slab, {info['npc_cell']} per cell) ==="
    )
    print(
        f"  Kn = {info['Kn']:.4f}   mfp/dx = {info['mfp_over_dx']:.2f}   "
        f"dt/mft = {info['dt_over_mft']:.3f}   steps = {info['steps']}"
    )
    print(
        f"  Mach {info['mach']:.1f}, U0 = {info['u0']:.0f} m/s   t_shock = {info['t_shock']:.3e} s"
    )
    print(
        f"  A_max resizes {info['resizes']}, settled peak {info['peak_attempts']:,}, "
        f"clipped {info['clipped']}, violations {info['violations']}"
    )
    print(f"  wall {info['wall_s']:.1f} s")
    thick = front_thickness(u, info)
    sine = info["L"] / math.pi / info["mfp"]
    print(f"  front thickness  = {thick:.2f} mfp   (an un-steepened sine gives {sine:.1f})")
    print(f"  density ratio    = {r.max() / r.min():.2f}")
    print(f"  u range          = [{u.min():.1f}, {u.max():.1f}] m/s")
    return thick


def compare_dims(profiles, infos):
    """The M2 gate: the same 1-D physics must give the same profile in every D."""
    print("\n=== cross-dimensional agreement (the M2 gate) ===")
    ref_d = infos[0]["dim"]
    ref = profiles[0].mean(axis=0)
    scale = np.abs(ref).max()
    # MC error on a cell velocity is v_th / sqrt(npc_slab); the profile is a mean
    # over `seeds` realisations, so the expected disagreement between two runs is
    # about sqrt(2) times that.
    expect = (
        math.sqrt(2.0) * infos[0]["vth"] / math.sqrt(infos[0]["npc_slab"] * profiles[0].shape[0])
    )
    print(f"  expected MC disagreement ~ {expect:.1f} m/s ({expect / scale * 100:.1f} % of |u|max)")
    ok = True
    for prof, info in zip(profiles[1:], infos[1:], strict=True):
        d = np.abs(prof.mean(axis=0) - ref)
        rms = float(np.sqrt(np.mean(d**2)))
        good = rms < 3.0 * expect
        ok &= good
        print(
            f"  D={info['dim']} vs D={ref_d}: rms diff {rms:7.2f} m/s "
            f"({rms / scale * 100:5.2f} % of |u|max)  max {d.max():7.2f}  "
            f"[{'ok' if good else 'FAIL'}]"
        )
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[1],
        help="dimensions to run; give several to check they agree",
    )
    ap.add_argument("--n-x", type=int, default=128)
    ap.add_argument("--n-trans", type=int, default=2, help="cells along each transverse axis")
    ap.add_argument("--npc-slab", type=int, default=400, help="particles per x-slab")
    ap.add_argument("--nrho", type=float, default=1e23)
    ap.add_argument("--mach", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--frames", type=int, default=6, help="field snapshots to render")
    ap.add_argument("--plot", action="store_true", help="write field PNGs to examples/figures")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    kw = {
        "n_x": args.n_x,
        "n_trans": args.n_trans,
        "npc_slab": args.npc_slab,
        "nrho": args.nrho,
        "mach": args.mach,
    }
    profiles, infos = [], []
    for d in args.dims:
        ux, rho, tmp, info = run_case(d, seeds=args.seeds, verbose=args.verbose, **kw)
        report(info, ux, rho, tmp)
        profiles.append(ux)
        infos.append(info)
        if args.plot:
            plot_profiles(ux, rho, tmp, info)
        print()
    if len(profiles) > 1:
        compare_dims(profiles, infos)
    return profiles, infos


def plot_profiles(ux, rho, tmp, info):
    """Write the x-profiles as a PNG.  Fields only -- never particle state.

    The field path is what a long run should use: a frame is tens of kilobytes and
    its size does not depend on ``N`` at all, whereas one subsampled particle frame
    at ``N = 1e6`` is already ~1.7 MB and the full state is two orders worse.
    ``jax_dsmc.viz.FieldMovie`` renders 2-D and 3-D fields the same way.
    """
    from jax_dsmc import viz  # noqa: PLC0415

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x = (np.arange(info["n"][0]) + 0.5) * info["dx"] / info["L"]
    fig, axes = viz.make_figure(1, 3, figsize=(13.5, 3.9))
    for ax, (data, lab) in zip(
        axes,
        ((ux, "$u_x$  [m/s]"), (rho, r"$n$  [m$^{-3}$]"), (tmp, "$T$  [K]")),
        strict=True,
    ):
        for row in data:
            ax.plot(x, row, color=viz.REFERENCE, lw=0.7, alpha=0.5)
        ax.plot(x, data.mean(axis=0), color=viz.SERIES_MEASURED, lw=viz.LINE_WIDTH)
        ax.set_xlabel("$x / L$")
        ax.set_ylabel(lab)
    fig.suptitle(
        f"D={info['dim']}  Mach {info['mach']:.1f}  Kn={info['Kn']:.4f}  "
        f"t={1.6 * info['t_shock']:.2e} s  ({info['seeds'] if 'seeds' in info else len(ux)} seeds)",
        color=viz.INK,
    )
    return viz.savefig(fig, str(FIGURE_DIR / f"shock_d{info['dim']}.png"))


if __name__ == "__main__":
    main()


# ============================ the Riemann (Sod) case =========================
#
# The sine case above shows a shock forming, but it has no closed-form solution to
# check against.  A Riemann problem does: an initial discontinuity in density and
# pressure develops a rarefaction, a contact and a shock whose positions and
# plateau values are known exactly (`examples.shock.exact`).  The domain stays
# periodic -- there are simply two interfaces, and the comparison is made before
# waves from one reach the other.


def sod_state(key, grid, sp, static, nrho_l, temp_l, nrho_r, temp_r, length):
    """Two-region initial state: a density and temperature jump at ``x = L/2``.

    Density is set by *how many particles go in each half*, since ``fnum`` is
    uniform: with equal cell volumes, ``nrho ~ npc``.  Both halves are drawn from
    equilibrium Maxwellians at their own temperature, at rest.
    """
    n_tot = static.N
    frac_l = nrho_l / (nrho_l + nrho_r)  # equal half-volumes
    n_l = round(frac_l * n_tot)
    n_r = n_tot - n_l
    k1, k2, k3, k4 = jax.random.split(key, 4)

    xl = jax.random.uniform(k1, (n_l, static.D), dtype=jnp.float64) * jnp.asarray(
        [length / 2] + [length] * (static.D - 1)
    )
    xr = jax.random.uniform(k2, (n_r, static.D), dtype=jnp.float64) * jnp.asarray(
        [length / 2] + [length] * (static.D - 1)
    ) + jnp.asarray([length / 2] + [0.0] * (static.D - 1))
    vth_l = math.sqrt(K_BOLTZ * temp_l / AR_MASS)
    vth_r = math.sqrt(K_BOLTZ * temp_r / AR_MASS)
    vl = vth_l * jax.random.normal(k3, (n_l, 3), dtype=jnp.float64)
    vr = vth_r * jax.random.normal(k4, (n_r, 3), dtype=jnp.float64)

    from jax_dsmc.state import ParticleState  # noqa: PLC0415

    return ParticleState(
        x=jnp.concatenate([xl, xr]),
        v=jnp.concatenate([vl, vr]),
        species=jnp.zeros((n_tot,), jnp.int32),
        alive=jnp.ones((n_tot,), bool),
        logh_acc=jnp.zeros((n_tot,), jnp.float64),
    )


def setup_sod(
    dim=1,
    n_x=1024,
    n_trans=2,
    npc_slab=200,
    nrho_l=1e23,
    temp_l=AR_TREF,
    density_ratio=0.125,
    pressure_ratio=0.125,
    length=1e-2,
    cfl=0.2,
    t_frac=0.10,
    a_max=64,
):
    """A Sod-type Riemann problem sized so the DSMC shock is genuinely resolved.

    ``length`` is deliberately 10x the sine case and ``n_x`` scales with it, which
    keeps ``dx`` at the mean free path while pushing ``Kn`` down by 10x.  That costs
    almost nothing: cells are the *parallel* axis (section 8.6), and the per-cell
    attempt count depends on ``dx``, not on ``L``.  The point is to separate the
    waves by many shock thicknesses so the inviscid reference is meaningful.

    ``t_frac`` is the run time as a fraction of ``L / c``; the waves must not reach
    the *other* interface, or the exact single-Riemann solution stops applying.
    """
    nrho_r = density_ratio * nrho_l
    temp_r = pressure_ratio / density_ratio * temp_l  # p = n k T

    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2.0]])
    diam = jnp.array([[AR_DIAM]])
    omega = jnp.array([[0.81]])
    tref = jnp.array([[AR_TREF]])
    alpha = jnp.array([[1.4]])
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=omega,
        tref=tref,
        alpha=alpha,
        mr=mr,
        prefactor=make_prefactor(diam, omega, tref, mr),
    )

    n = (n_x,) + (n_trans,) * (dim - 1)
    grid = GridSpec(
        lo=jnp.zeros(dim),
        hi=jnp.full(dim, length),
        n=n,
        transverse=jnp.float64(length ** (3 - dim)),
    )
    C = num_cells(grid)

    vth = math.sqrt(K_BOLTZ * temp_l / AR_MASS)
    dx = length / n_x
    dt = cfl * dx / vth
    c0 = sound_speed(temp_l)
    steps = math.ceil(t_frac * length / c0 / dt)
    n_part = n_x * npc_slab

    fnum = jnp.full((C,), nrho_l * length**3 / (n_part / (0.5 * (1 + density_ratio))))
    static = SimStatic(N=n_part, D=dim, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=steps)
    params = SimParams(dt=jnp.float64(dt), fnum=fnum, epsilon=jnp.float64(0.0))
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((C,), temp_l), omega[0, 0], tref[0, 0], mr[0, 0], sp.prefactor[0, 0], C, 1
        )
    )
    info = {
        "dim": dim,
        "n": n,
        "C": C,
        "N": n_part,
        "dx": dx,
        "dt": dt,
        "L": length,
        "nrho_l": nrho_l,
        "nrho_r": nrho_r,
        "temp_l": temp_l,
        "temp_r": temp_r,
        "mfp": mean_free_path(nrho_l),
        "Kn": mean_free_path(nrho_l) / length,
        "mfp_over_dx": mean_free_path(nrho_l) / dx,
        "steps": steps,
        "t_end": steps * dt,
        "vth": vth,
        "c0": c0,
        "npc_slab": npc_slab,
    }
    return sp, grid, static, params, cstate, info


def run_sod(dim=1, seeds=4, verbose=False, **kw):
    """Run the Riemann problem and return seed-averaged profiles plus the exact solution."""
    from jax_dsmc import sample as smp  # noqa: PLC0415
    from jax_dsmc import sort  # noqa: PLC0415

    sp, grid, static, params, cstate, info = setup_sod(dim=dim, **kw)
    rho, ux, tmp = [], [], []
    t0 = time.perf_counter()
    for s in range(seeds):
        key = jax.random.PRNGKey(900 + s)
        state = sod_state(
            key,
            grid,
            sp,
            static,
            info["nrho_l"],
            info["temp_l"],
            info["nrho_r"],
            info["temp_r"],
            info["L"],
        )
        (fs, _), (_m, diag), _h = run_adaptive(
            state,
            cstate,
            grid,
            sp,
            params,
            static,
            jax.random.fold_in(key, 5),
            bc=BC.PERIODIC,
            chunk=max(16, info["steps"] // 6),
            verbose=verbose,
        )
        table = sort.build_table(fs, grid)
        m = smp.moments(fs, table, grid, params, sp)
        rho.append(x_profile(m.nrho, m.count, info))
        ux.append(x_profile(m.u[:, 0], m.count, info))
        tmp.append(x_profile(m.temp, m.count, info))
        info["clipped"] = int(jnp.sum(diag.attempts_clipped))
        info["violations"] = int(jnp.sum(diag.majorant_violations))
    info["wall_s"] = time.perf_counter() - t0
    return np.array(rho), np.array(ux), np.array(tmp), info


def sod_exact(info, n_pts=2000):
    """The exact Riemann solution for this case, in DSMC units, about ``x = L/2``.

    The solver works in ``(rho, u, p)``; we hand it mass density and pressure built
    from the DSMC state (``p = n k T``), then convert the result back to number
    density and temperature so the comparison is like for like.
    """
    from examples.shock import exact  # noqa: PLC0415

    m = AR_MASS
    left = (info["nrho_l"] * m, 0.0, info["nrho_l"] * K_BOLTZ * info["temp_l"])
    right = (info["nrho_r"] * m, 0.0, info["nrho_r"] * K_BOLTZ * info["temp_r"])
    x = np.linspace(0.0, info["L"], n_pts)
    xi = (x - info["L"] / 2.0) / info["t_end"]
    r, u, p = exact.sample(xi, left, right, gamma=5.0 / 3.0)
    return x, r / m, u, p / (K_BOLTZ * (r / m)), exact.wave_speeds(left, right, 5.0 / 3.0)


def measure_waves(rho, info):
    """Locate the shock and contact in the DSMC density profile.

    Both are found as the steepest descent within their own search window, which is
    robust to the finite width DSMC gives them.  Position is the physically sharp
    thing to compare: the exact solution fixes where a wave *is* exactly, while its
    *width* is a genuine difference between Boltzmann and Euler and carries no
    information about solver correctness.
    """
    xc = (np.arange(info["n"][0]) + 0.5) * info["dx"]
    r = rho.mean(axis=0)
    _x, _r, _u, _t, w = sod_exact(info)
    xs, t = info["L"] / 2.0, info["t_end"]
    out = {}
    for name, speed in (("contact", w["contact"]), ("shock", w["right"])):
        x_ex = xs + speed * t
        win = np.abs(xc - x_ex) < 0.25 * (w["right"] - w["contact"]) * t + 4 * info["dx"]
        if win.sum() < 3:
            continue
        grad = np.gradient(r, info["dx"])
        i = np.argmin(np.where(win, grad, 0.0))
        out[name] = {
            "measured": xc[i],
            "exact": x_ex,
            "err_mfp": (xc[i] - x_ex) / info["mfp"],
            "err_pct_of_travel": (xc[i] - x_ex) / (speed * t) * 100.0,
        }
    return out


def compare_sod(rho, ux, tmp, info):
    """Report DSMC against the exact solution, away from the discontinuities.

    The exact solution is inviscid: its shock has zero thickness and its contact is
    perfectly sharp, while DSMC gives both a physical width of order the mean free
    path.  So the comparison is made on the **plateau** values and the **wave
    positions**, and cells within a few mean free paths of a discontinuity are
    excluded -- comparing there would be measuring the difference between Euler and
    Boltzmann, which is not what this test is for.
    """
    x, r_ex, u_ex, t_ex, w = sod_exact(info)
    xc = (np.arange(info["n"][0]) + 0.5) * info["dx"]
    r_e = np.interp(xc, x, r_ex)
    u_e = np.interp(xc, x, u_ex)
    t_e = np.interp(xc, x, t_ex)

    # mask out a few mfp either side of each wave, and the second interface
    guard = 6.0 * info["mfp"]
    xs = info["L"] / 2.0
    waves = [
        xs + w["left"] * info["t_end"],
        xs + w["contact"] * info["t_end"],
        xs + w["right"] * info["t_end"],
        xs,
    ]
    ok = np.ones_like(xc, dtype=bool)
    for xw in waves:
        ok &= np.abs(xc - xw) > guard
    ok &= (xc > 0.15 * info["L"]) & (xc < 0.85 * info["L"])

    print("\n=== DSMC vs exact Riemann solution ===")
    print(
        f"  p*/p_L = {w['p_star'] / (info['nrho_l'] * K_BOLTZ * info['temp_l']):.4f}   "
        f"u* = {w['u_star']:.1f} m/s"
    )
    print(
        f"  wave speeds [m/s]: left {w['left']:.1f}   contact {w['contact']:.1f}   "
        f"right {w['right']:.1f}"
    )
    print(
        f"  comparing {int(ok.sum())} of {ok.size} cells "
        f"(excluding {guard / info['mfp']:.0f} mfp either side of each wave)"
    )
    out = {}
    for name, meas, ref, unit in (
        ("nrho", rho.mean(0), r_e, "m^-3"),
        ("u_x", ux.mean(0), u_e, "m/s"),
        ("temp", tmp.mean(0), t_e, "K"),
    ):
        scale = np.abs(ref[ok]).max()
        rms = float(np.sqrt(np.mean((meas[ok] - ref[ok]) ** 2)))
        out[name] = rms / scale
        print(
            f"  {name:>5}: rms error {rms:12.4e} {unit:>6}  =  {rms / scale * 100:5.2f} % of range"
        )

    # Wave POSITIONS are the sharp test; wave WIDTHS are physics, not error.
    waves_m = measure_waves(rho, info)
    print("\n  wave positions (what the exact solution pins down exactly):")
    for nm, dd in waves_m.items():
        print(
            f"    {nm:>8}: DSMC {dd['measured'] / info['L']:.4f} L   "
            f"exact {dd['exact'] / info['L']:.4f} L   "
            f"off by {dd['err_mfp']:+.1f} mfp ({dd['err_pct_of_travel']:+.1f}% of travel)"
        )
    out["waves"] = waves_m
    print("\n  The rms above is NOT all solver error: DSMC's contact diffuses and its shock")
    print("  has a finite thickness of order the mean free path.  Euler has neither, so any")
    print("  such structure inside the mask contributes Boltzmann-vs-Euler difference.")
    return out, (x, r_ex, u_ex, t_ex, w)
