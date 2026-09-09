"""Milestone 1 acceptance gate: 0D BKW relaxation (spec section 13 / PLAN phase 2).

Isolates ``collide()`` -- one cell, no grid, no advection -- at ``omega = 1``
(Maxwell molecules), where the spatially homogeneous Boltzmann equation has the
exact Bobylev-Krook-Wu solution.

The gate quantity is the **normalised fourth-moment excess**

    E(t) = 1 - <|v|^4> / (15 vth^4) = (1 - K(t))^2 = (1 - K0)^2 exp(-nu t / 3)

a pure exponential, dimensionless, and independent of ``temp``, ``mass`` and
``nrho``.  A correct solver must reproduce **both** its amplitude and its rate;
matching only the rate would pass a solver with the wrong collision cross-section,
and matching only the amplitude would pass one that does not relax at all.

Run::

    python -m examples.bkw
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
from jax import lax

jax.config.update("jax_enable_x64", True)


def enable_compilation_cache(path=None):
    """Persist compiled binaries to disk across processes.

    A change of ``N`` changes every array shape, so XLA **must** produce a new
    program -- that part is not avoidable and is not a defect.  What is avoidable
    is paying for it *again* on the next invocation: with this enabled, the second
    run of any size already seen skips compilation entirely, which is what makes
    re-running a sweep from the shell cheap.

    Off by default in library use; the CLI turns it on unless ``--no-cache``.
    """
    path = pathlib.Path(path or (pathlib.Path(__file__).resolve().parents[1] / ".jax_cache"))
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path


from jax_dsmc import bkw, sample, sort  # noqa: E402
from jax_dsmc.collide import make_prefactor  # noqa: E402
from jax_dsmc.collide.ntc import collide, vremax_init  # noqa: E402
from jax_dsmc.state import (  # noqa: E402
    CollideState,
    GridSpec,
    ParticleState,
    SimParams,
    SimStatic,
    SpeciesParams,
)

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def setup(
    n_part=200_000,
    n_steps=60,
    nrho=1e21,
    temp=273.15,
    steps_per_tau=0.15,
    a_max=None,
    side=None,
    npc_target=1000,
):
    """Build the 0D case.  Returns everything ``run`` needs plus the analytic ``nu``."""
    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2.0]])
    diam = jnp.array([[AR_DIAM]])
    omega = jnp.array([[1.0]])  # Maxwell molecules -- the BKW regime
    tref = jnp.array([[AR_TREF]])
    alpha = jnp.array([[1.0]])  # isotropic scattering, as BKW assumes
    sp = SpeciesParams(
        mass=mass,
        diam=diam,
        omega=omega,
        tref=tref,
        alpha=alpha,
        mr=mr,
        prefactor=make_prefactor(diam, omega, tref, mr),
    )

    # The box volume is set so that nrho is the requested density; with fnum == 1
    # the parcel count IS the molecule count.
    volume = n_part * 1.0 / nrho
    box = volume ** (1.0 / 3.0)
    # Cells per axis follow N so that particles-per-cell stays ~constant.  This is
    # the single most important knob for speed here, and it is a *shape* choice,
    # not a physics one: `A_max` is PER CELL, so pinning the cell count makes the
    # SEQUENTIAL attempt-slot scan grow with N while the PARALLEL cell axis stays
    # fixed -- exactly backwards.  Measured at N = 640,000: 26.4 s with 8 cells,
    # 0.68 s with npc held at ~1000 (39x), and 1e7 particles becomes reachable at
    # all (section 8.6).
    if side is None:
        side = max(1, round((n_part / npc_target) ** (1.0 / 3.0)))
    grid = GridSpec(lo=jnp.zeros(3), hi=jnp.full(3, box), n=(side, side, side))
    n_cells = side**3

    nu = float(bkw.collision_rate(nrho, bkw.maxwell_sigma_vr(AR_DIAM, AR_TREF, float(mr[0, 0]))))
    dt = steps_per_tau / nu

    # A_max must be sized from the INFLATED majorant (section 8.2, gap U4): the
    # attempt count is proportional to Sigma, and Sigma carries the F_UP factor.
    sigma = float(
        vremax_init(
            jnp.full((1,), temp), omega[0, 0], tref[0, 0], mr[0, 0], sp.prefactor[0, 0], 1, 1
        )[0, 0, 0]
    )
    # A_max is PER CELL, so it follows the per-cell occupancy, not N.  Splitting a
    # homogeneous gas across `n_cells` leaves the total collision rate unchanged --
    # `npc` falls by `n_cells` and the cell volume falls with it, and the two
    # cancel -- so this is the same physics with a scan `n_cells` times shorter.
    npc = n_part / n_cells
    v_cell = volume / n_cells
    f = 0.5 * npc * (npc - 1.0) * sigma * dt * 1.0 / v_cell
    a_max = max(8, math.ceil(1.3 * f)) if a_max is None else a_max

    static = SimStatic(N=n_part, D=3, S=1, G=1, A_max=a_max, K_bounce=1, n_steps=n_steps)
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.ones(n_cells), epsilon=jnp.float64(0.0))
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((n_cells,), temp),
            omega[0, 0],
            tref[0, 0],
            mr[0, 0],
            sp.prefactor[0, 0],
            n_cells,
            1,
        )
    )
    return sp, grid, static, params, cstate, nu, dt, temp


def initial_state(key, static, grid, temp, k0=bkw.K_MIN):
    """Particles drawn **exactly** from the BKW distribution at ``K = K0``.

    Exact rather than rejection-sampled, so the initial condition carries no bias
    of its own into the gate.
    """
    v = bkw.sample_velocities(key, static.N, k0, temp, AR_MASS)
    x = jax.random.uniform(jax.random.fold_in(key, 1), (static.N, 3)) * (grid.hi - grid.lo)
    return ParticleState(
        x=x,
        v=v,
        species=jnp.zeros(static.N, jnp.int32),
        alive=jnp.ones(static.N, bool),
        logh_acc=jnp.zeros(static.N),
    )


def measured_excess(v):
    """``1 - <|v|^4>/(15 vth^4)`` from particle velocities, with ``vth`` measured.

    ``vth`` is taken from the sample's own second moment rather than from the
    nominal temperature, so the gate tests the *shape* of the distribution and is
    insensitive to any drift in total energy (which is conserved to round-off
    anyway -- see ``tests/test_ntc.py``).
    """
    v2 = jnp.sum(v * v, axis=-1)
    vth2 = jnp.mean(v2) / 3.0
    return 1.0 - jnp.mean(v2 * v2) / (15.0 * vth2 * vth2)


def _relax_step(carry, k, grid, sp, params, static, base_key):
    """One collision step that also emits the fourth-moment excess.

    0D, so there is no advection: a single cell with no walls means ``move`` is
    identically the identity, and running it would only add round-off.
    """
    state, cstate = carry
    table = sort.build_table(state, grid)
    # `cell_temperature`, not `moments`: the gate needs only the fourth-moment
    # excess and the majorant's temperature, so the full moment set would
    # materialise an (N,3,3) pressure tensor every step for nothing (section 8.6).
    temp = sample.cell_temperature(state, table, grid, sp)
    state, cstate, diag = collide(state, table, cstate, grid, sp, params, static, temp, k, base_key)
    return (state, cstate), (
        measured_excess(state.v),
        diag.majorant_violations,
        diag.attempts_clipped,
    )


def _relax(state, cstate, grid, sp, params, static, base_key):
    body = functools.partial(
        _relax_step, grid=grid, sp=sp, params=params, static=static, base_key=base_key
    )
    return lax.scan(body, (state, cstate), jnp.arange(static.n_steps))


#: Jitted ONCE at module level.  Every `jax.jit(...)` call builds a fresh wrapper
#: with its own cache, so creating it inside `run_seeds` meant two calls at the
#: same problem size recompiled instead of hitting the cache.  Hoisted, every
#: repeat of a shape already seen in this process is free.
_RELAX_JIT = jax.jit(_relax, static_argnames=("static",))


# =============================== the driver =================================


def run_seeds(n_part, n_steps, dt_nu, seeds, seed0=0, verbose=False, side=None, npc_target=1000):
    """Relax the gas once per seed and return every trajectory.

    Args:
        n_part, n_steps, dt_nu: the case.
        seeds: how many seeds to run, numbered ``seed0 .. seed0 + seeds - 1``.
        seed0: first seed.  Distinct seeds give statistically independent runs.

    Returns:
        ``(E, t_nu, info)`` where ``E`` is ``f64[seeds, n_steps + 1]`` -- the
        measured fourth-moment excess, including the initial condition in column
        0 -- ``t_nu`` is the matching ``f64[n_steps + 1]`` time axis in units of
        the mean collision time, and ``info`` is a dict of run metadata.

    The jit cache is keyed on shape, so the compile is paid **once** and every
    subsequent seed reuses it.  That matters: at these sizes the compile is a
    large fraction of a single run's wall time, which is why sweeping seeds in a
    shell loop over separate processes is roughly an order of magnitude slower
    and made the GPU look idle between bursts (spec 8.6).
    """
    sp, grid, static, params, cstate, nu, dt, temp = setup(
        n_part, n_steps, steps_per_tau=dt_nu, side=side, npc_target=npc_target
    )
    fn = _RELAX_JIT
    traj, viol, clip = [], 0, 0
    t_compile = None
    for s in range(seed0, seed0 + seeds):
        key = jax.random.PRNGKey(s)
        state = initial_state(key, static, grid, temp)
        t0 = time.perf_counter()
        (_f, _), (em, v, c) = fn(
            state, cstate, grid, sp, params, static, jax.random.fold_in(key, 99)
        )
        em.block_until_ready()
        elapsed = time.perf_counter() - t0
        if t_compile is None:
            t_compile = elapsed
        traj.append(np.concatenate([[float(measured_excess(state.v))], np.asarray(em)]))
        viol += int(np.sum(v))
        clip += int(np.sum(c))
        if verbose:
            print(f"    seed {s:>3}: rate = {rate_ols(traj[-1], dt * nu):.6f}  ({elapsed:.2f} s)")
    E = np.array(traj)
    t_nu = np.arange(E.shape[1]) * dt * nu
    info = {
        "nu": nu,
        "dt": dt,
        "dt_nu": dt * nu,
        "A_max": static.A_max,
        "N": static.N,
        "n_steps": n_steps,
        "violations": viol,
        "clipped": clip,
        "compile_s": t_compile,
        "cells": int(np.prod(grid.n)),
        "npc": static.N // int(np.prod(grid.n)),
    }
    return E, t_nu, info


# --- rate estimators.  Three, because they fail differently. -----------------
#
# `E` decays ~4.5x over the window while the absolute Monte Carlo noise on it
# stays roughly constant, so late points carry much larger *relative* error, and
# `log` of a noisy small number is biased downward by `-sigma^2 / 2E^2` -- which
# steepens an OLS fit and reads the rate HIGH.  Agreement between the three is
# evidence that a measured deviation is physical rather than an artefact of the
# fit; disagreement localises it to the estimator.


def rate_ols(e, dtn, frac=1.0):
    """Decay rate from OLS on ``log E`` over the first ``frac`` of the trajectory."""
    n = max(3, round(frac * e.size))
    t = np.arange(n) * dtn
    return -np.polyfit(t, np.log(e[:n]), 1)[0]


def rate_ratio(e, dtn):
    """Two-point log-ratio over the first quarter.  No fit, so no fitting bias."""
    i0, i1 = 1, max(2, e.size // 4)
    return -(np.log(e[i1]) - np.log(e[i0])) / ((i1 - i0) * dtn)


ESTIMATORS = {
    "ols_full": lambda e, d: rate_ols(e, d, 1.0),
    "ols_half": lambda e, d: rate_ols(e, d, 0.5),
    "ratio": rate_ratio,
}

EXACT_RATE = 1.0 / 3.0


def summarise(E, dtn):
    """Per-estimator mean, sd and standard error of the mean, in % bias."""
    out = {}
    for name, fn in ESTIMATORS.items():
        v = np.array([fn(e, dtn) for e in E])
        sd = v.std(ddof=1) if v.size > 1 else float("nan")
        out[name] = {
            "mean": v.mean(),
            "sd": sd,
            "sem": sd / np.sqrt(v.size) if v.size > 1 else float("nan"),
            "bias_pct": (v.mean() - EXACT_RATE) / EXACT_RATE * 100,
            "sem_pct": sd / np.sqrt(v.size) / EXACT_RATE * 100 if v.size > 1 else float("nan"),
            "values": v,
        }
    return out


# =============================== reporting ==================================


def report_single(E, t_nu, info):
    """Print one case: the trajectory against exact, and the fitted rate."""
    nu = info["nu"]
    ee = np.asarray(bkw.moment_excess(jnp.asarray(t_nu / nu), nu))
    em = E.mean(axis=0)
    print(f"  {'step':>5} {'t*nu':>8} {'E measured':>12} {'E exact':>12} {'rel err':>10}")
    for k in range(0, em.size, max(1, (em.size - 1) // 10)):
        print(
            f"  {k:>5} {t_nu[k]:>8.3f} {em[k]:>12.6f} {ee[k]:>12.6f} "
            f"{abs(em[k] - ee[k]) / ee[k]:>10.2e}"
        )
    stats = summarise(E, info["dt_nu"])
    print(f"\n  exact decay rate   = {EXACT_RATE:.6f} nu")
    for name, st in stats.items():
        sem = "" if E.shape[0] < 2 else f" +- {st['sem_pct']:.2f}"
        print(f"  {name:<10} rate = {st['mean']:.6f}   bias = {st['bias_pct']:+.3f} %{sem}")
    print(f"  max rel err in <E> = {np.max(np.abs(em - ee) / ee):.4f}")
    print(f"  majorant violations= {info['violations']}")
    print(f"  attempts clipped   = {info['clipped']}")
    return stats, ee


def report_convergence(rows):
    """Print the N sweep and check the sd against the 1/sqrt(N) law."""
    print(f"\n{'N':>10} {'A_max':>8} {'seeds':>6} | " + " | ".join(f"{n:>22}" for n in ESTIMATORS))
    print(f"{'':>10} {'':>8} {'':>6} | " + " | ".join(f"{'bias % +- sem':>22}" for _ in ESTIMATORS))
    print("-" * (28 + 25 * len(ESTIMATORS)))
    for N, info, stats in rows:
        cells = " | ".join(
            f"{stats[n]['bias_pct']:>+11.2f} +- {stats[n]['sem_pct']:>6.2f}" for n in ESTIMATORS
        )
        print(f"{N:>10,} {info['A_max']:>8,} {len(stats['ols_full']['values']):>6} | {cells}")

    print("\n  1/sqrt(N) check -- sd of the rate estimate should halve per 4x in N:")
    print(f"  {'N':>10} {'sd (ols_half)':>15} {'ratio to prev':>15} {'predicted':>11}")
    prev_sd = prev_n = None
    for N, _info, stats in rows:
        sd = stats["ols_half"]["sd"]
        if prev_sd is None:
            print(f"  {N:>10,} {sd:>15.6f} {'-':>15} {'-':>11}")
        else:
            print(f"  {N:>10,} {sd:>15.6f} {prev_sd / sd:>15.3f} {np.sqrt(N / prev_n):>11.3f}")
        prev_sd, prev_n = sd, N

    # fitted exponent of sd ~ N^p; the Monte Carlo law says p = -1/2
    Ns = np.array([r[0] for r in rows], dtype=float)
    sds = np.array([r[2]["ols_half"]["sd"] for r in rows])
    p = np.polyfit(np.log(Ns), np.log(sds), 1)[0]
    print(f"\n  fitted sd ~ N^({p:.3f})   Monte Carlo predicts N^(-0.500)")
    return p


# =============================== plotting ===================================


FIGURE_DIR = pathlib.Path(__file__).resolve().parent / "figures"


def _fig_path(name):
    """Figures live beside the example, not in whatever directory it was run from."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIGURE_DIR / name)


def _mpl():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def plot_relaxation(E, t_nu, info, path=None):
    """Measured excess against the exact BKW curve, log scale."""
    path = _fig_path("bkw_relaxation.png") if path is None else path
    plt = _mpl()
    ee = np.asarray(bkw.moment_excess(jnp.asarray(t_nu / info["nu"]), info["nu"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if E.shape[0] > 1:
        lo, hi = E.min(axis=0), E.max(axis=0)
        ax.fill_between(
            t_nu, lo, hi, alpha=0.2, color="C0", label=f"DSMC spread, {E.shape[0]} seeds"
        )
    ax.semilogy(t_nu, E.mean(axis=0), "o", ms=4, mfc="none", color="C0", label="DSMC mean")
    ax.semilogy(t_nu, ee, "k-", lw=2, label="BKW exact")
    ax.set_xlabel(r"$t\,\nu$")
    ax.set_ylabel(r"$1-\langle |v|^4\rangle/15\,v_{th}^4$")
    ax.set_title(f"M1: BKW relaxation, Maxwell molecules (N={info['N']:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


def plot_convergence(rows, path=None):
    """sd of the fitted rate vs N on log-log, against the 1/sqrt(N) reference.

    This is the plot that demonstrates the error is Monte Carlo scatter rather
    than a defect: a slope of -1/2 is the signature of independent sampling, and
    a flattening slope would mean a systematic floor had been hit.
    """
    path = _fig_path("bkw_convergence.png") if path is None else path
    plt = _mpl()
    Ns = np.array([r[0] for r in rows], dtype=float)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for name, mk in zip(ESTIMATORS, ("o", "s", "^"), strict=True):
        sds = np.array([r[2][name]["sd"] for r in rows])
        ax.loglog(Ns, sds, mk + "-", ms=6, mfc="none", label=name)
    ref = np.array([r[2]["ols_half"]["sd"] for r in rows])[0] * np.sqrt(Ns[0] / Ns)
    ax.loglog(Ns, ref, "k--", lw=1.5, label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("particles $N$")
    ax.set_ylabel("sd of fitted decay rate  [$\\nu$]")
    ax.set_title("Monte Carlo convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    for name, mk in zip(ESTIMATORS, ("o", "s", "^"), strict=True):
        b = np.array([r[2][name]["bias_pct"] for r in rows])
        e = np.array([r[2][name]["sem_pct"] for r in rows])
        ax2.errorbar(Ns, b, yerr=e, fmt=mk + "-", ms=6, mfc="none", capsize=3, label=name)
    ax2.axhline(0.0, color="k", lw=1.5, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("particles $N$")
    ax2.set_ylabel("bias in decay rate  [%]")
    ax2.set_title("Bias, with standard error of the mean")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


# =============================== entry point ================================


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-part", type=int, default=40_000, help="particles (single case)")
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument(
        "--dt-nu", type=float, default=0.15, help="timestep in units of the mean collision time"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of seeds to average over, numbered from --seed0",
    )
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument(
        "--sweep-n",
        type=int,
        nargs="+",
        default=None,
        help="particle counts to sweep for the convergence study, "
        "e.g. --sweep-n 5000 10000 20000 40000",
    )
    ap.add_argument("--plot", action="store_true", help="write bkw_relaxation.png")
    ap.add_argument(
        "--plot-convergence",
        action="store_true",
        help="write bkw_convergence.png (needs --sweep-n)",
    )
    ap.add_argument("--verbose", action="store_true", help="print each seed's rate")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the on-disk compilation cache (examples/.jax_cache). "
        "A change of N changes every array shape, so XLA must build a new "
        "program; the cache means you only pay for it once per size, ever.",
    )
    ap.add_argument(
        "--particles-per-cell",
        type=int,
        default=1000,
        help="target particles per cell; the cell count is DERIVED from it so that it stays "
        "~constant as N grows (default 1000).  This is the main speed knob: A_max is PER "
        "CELL, so pinning the cell count instead makes the sequential attempt-slot scan "
        "grow with N while the parallel cell axis stays fixed.  At N=640,000 that is "
        "26.4 s versus 0.68 s (section 8.6).",
    )
    ap.add_argument(
        "--cells-per-side",
        type=int,
        default=None,
        help="override the derived cell count with an explicit cells-per-axis.  The gas is "
        "homogeneous, so this changes no physics -- npc falls by the cell count and the "
        "cell volume falls with it, and the two cancel.",
    )
    args = ap.parse_args(argv)
    if not args.no_cache:
        enable_compilation_cache()

    if args.sweep_n:
        print(
            f"M1 BKW convergence sweep: seeds={args.seeds}  dt*nu={args.dt_nu}  "
            f"steps={args.n_steps}"
        )
        print("NOTE: a rate from a handful of seeds is NOT evidence -- see the module docstring.\n")
        rows = []
        for N in args.sweep_n:
            E, t_nu, info = run_seeds(
                N,
                args.n_steps,
                args.dt_nu,
                args.seeds,
                args.seed0,
                args.verbose,
                args.cells_per_side,
                args.particles_per_cell,
            )
            rows.append((N, info, summarise(E, info["dt_nu"])))
            print(
                f"  N={N:,} done  (cells={info['cells']:,}, npc={info['npc']:,}, "
                f"A_max={info['A_max']:,}, compile {info['compile_s']:.1f} s, "
                f"violations {info['violations']}, clipped {info['clipped']})"
            )
        report_convergence(rows)
        if args.plot_convergence:
            plot_convergence(rows)
        return rows

    E, t_nu, info = run_seeds(
        args.n_part,
        args.n_steps,
        args.dt_nu,
        args.seeds,
        args.seed0,
        args.verbose,
        args.cells_per_side,
        args.particles_per_cell,
    )
    print(
        f"M1 BKW gate: N={info['N']:,}  cells={info['cells']:,} (npc={info['npc']:,})  "
        f"steps={info['n_steps']}  A_max={info['A_max']:,}  seeds={args.seeds}"
    )
    print(f"  nu = {info['nu']:.6e} 1/s   dt = {info['dt']:.6e} s   dt*nu = {info['dt_nu']:.4f}")
    print(f"  K0 = {bkw.K_MIN}  =>  E(0) = {(1 - bkw.K_MIN) ** 2:.6f}\n")
    stats, _ = report_single(E, t_nu, info)
    if args.seeds < 8:
        print(
            "\n  WARNING: fewer than 8 seeds.  The fitted rate has a wide spread "
            "(sd ~ 13% at N=1e4);\n  a small sweep will show a convincing-looking "
            "trend that is pure noise.  Use --seeds 16+."
        )
    if args.plot:
        plot_relaxation(E, t_nu, info)
    return stats


if __name__ == "__main__":
    main()
