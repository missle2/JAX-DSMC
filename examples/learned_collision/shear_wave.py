"""The effective viscosity and conductivity of a collision operator against the cell size:
periodic shear- and temperature-wave decay (the Phase 1 decision gate of the learned-collision
plan, 2026-10-21).

A periodic 1-D box of length ``L`` (argon at the Couette example's 273.15 K, n = 1e21 m^-3)
starts as a Maxwellian carrying the transverse wave ``u_y = A sin(k x)``, ``k = 2 pi / L``.
Its projected amplitude ``A(t) = (2 / N) sum v_y sin(k x)`` decays as ``exp(-nu k^2 t)``,
``nu = mu / rho`` (Navier-Stokes; Burnett corrections are O((k lambda)^2), the same for every
cell size here).  No walls and no Knudsen layers; ``dt`` is fixed in physical units
(``dt = DT_MFP lambda / v_th``), so only the cell size varies -- the clean replacement of the
Couette study, whose ``dt`` scaled with the cell (``examples/couette/__init__.py``).

``--wave thermal``: the box starts isobaric with ``T = T0 (1 + eps sin(k x))`` and
``n ~ 1 / T`` -- the entropy mode -- whose amplitude decays as ``exp(-chi k^2 t)``,
``chi = kappa / (rho c_p)``.  It is measured as ``-(2 / N) sum (1 - 3 e_i / 5) sin(k x_i)``,
``e_i = |v_i|^2 / (3 v_th^2)``: the density mode minus ``1 / gamma`` of the pressure mode,
which a linear sound wave (``delta rho / rho = delta p / (gamma p)``) does not enter.  The
density mode alone carried the sound that the initial shot noise in the pressure mode
launches (+-0.05 per seed at 1600 particles, the seed mean oscillating at the box's acoustic
period).  With the
Chapman-Enskog ``kappa = (15 / 4)(k_B / m) mu`` (Pr = 2/3), ``chi = 3 nu / 2``.

    PYTHONPATH=. python3 -m examples.learned_collision.shear_wave [--wave shear|thermal] \
        [--dx-mfp 0.25 0.5 1 2 4] [--operators ntc ntc_adjacent esfp esfp_recon] [--particles 4e5]
        [--ppc 50]

Writes ``examples/learned_collision/results/<wave>_wave_<tag>.json`` (one row per operator and
cell size: ``ratio`` = ``mu_eff / mu_CE`` or ``kappa_eff / kappa_CE`` from the seed-mean
amplitude, its bootstrap standard error ``ratio_se``, and the mean series).  Every cell size gets the same particle-runs (``--particles``): a coarse box
has fewer particles, so it runs more seeds.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.couette import GAS, NRHO, TEMP, mean_free_path, viscosity_ce
from jax_dsmc import sampling, simulate
from jax_dsmc.collide import relax, species_params, vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic

LENGTH_MFP = 64.0
"""Box length in mean free paths (``k lambda = 0.098``)."""
DT_MFP = 0.05
"""Time step in ``lambda / v_th``, fixed for every cell size."""
AMPLITUDE = 0.2
"""Shear-wave amplitude in ``v_th``."""
THERMAL_AMPLITUDE = 0.1
"""Temperature-wave amplitude ``eps`` (``kappa ~ T^omega`` makes the decay nonlinear at
O(eps^2))."""
RATE_CE = {"shear": 1.0, "thermal": 1.5}
"""Chapman-Enskog decay rate of each wave in ``nu k^2``."""
BLOCK = 20
"""Steps between amplitude samples."""
RESULTS = Path(__file__).resolve().parent / "results"
OPERATORS = ("ntc", "ntc_adjacent", "esfp", "esfp_recon")
_RECON = functools.partial(relax.collide, periodic=(True,))
"""ES-FP with the reconstructed velocity target: one object, since ``advance`` keys its compile
cache on ``collide_fn`` and a ``partial`` hashes by identity (a fresh one per seed recompiled
every seed until the JIT ran out of code memory)."""


def operator(name):
    """``(collide_fn, collision_params, pairing)`` of an operator name (``None`` collide_fn =
    NTC, whose partner selection is ``pairing``)."""
    theta = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
    if name == "ntc":
        return None, None, "uniform"
    if name == "ntc_adjacent":
        return None, None, "adjacent"
    if name == "esfp":
        return relax.collide, theta, "uniform"
    if name == "esfp_recon":
        return _RECON, theta, "uniform"
    raise ValueError(f"unknown operator {name}")


def isobaric_positions(key, n_part, length, k, eps):
    """``n_part`` positions with density ``~ 1 / (1 + eps sin(k x))`` on ``[0, length)``, by
    rejection from twice as many uniform draws (acceptance ``sqrt((1 - eps) / (1 + eps))``)."""
    kx, ka = jax.random.split(key)
    x = jax.random.uniform(kx, (2 * n_part,), maxval=length)
    keep = jax.random.uniform(ka, x.shape) * (1.0 + eps * jnp.sin(k * x)) < 1.0 - eps
    return x[jnp.nonzero(keep, size=n_part)[0]][:, None]


def auto_a_max(ppc):
    """Attempt slots per cell and step: about ``0.25 ppc`` attempts on average at ``dt =
    DT_MFP``, six-sigma headroom (measured maxima: 5 at ppc 10, 15 at ppc 50; the slot scan
    costs ``A_max`` whatever the need, so 64 slots ran 4-12x too many).  Clipping is counted."""
    mean = 0.25 * ppc
    return math.ceil(mean + 6.0 * math.sqrt(mean) + 6.0)


def setup(dx_mfp, ppc, seed, a_max, wave="shear", length_mfp=LENGTH_MFP, ny=1,
          amplitude=AMPLITUDE):  # fmt: skip
    """The box, its gas and the initial wave: ``(state, cstate, grid, sp, params, static,
    key, constants)``.  ``ny > 1``: a 2-D box ``ny`` cells high (periodic, the wave still
    along ``x``), so the particles per ``lambda`` of ``x`` grow ``ny``-fold at the same cells
    and particles per cell.  ``amplitude``: the shear wave's, in ``v_th``."""
    a_max = a_max or auto_a_max(ppc)
    lam = mean_free_path()
    length = length_mfp * lam
    cells = round(length_mfp / dx_mfp)
    n_cells = cells * ny
    n_part = n_cells * ppc
    if ny == 1:
        grid = GridSpec(lo=jnp.zeros(1), hi=jnp.array([length]), n=(cells,))
    else:
        grid = GridSpec(lo=jnp.zeros(2), hi=jnp.array([length, ny * dx_mfp * lam]),
                        n=(cells, ny))  # fmt: skip
    sp = species_params(
        jnp.array([GAS.mass]), jnp.array([[GAS.diam]]), jnp.array([[GAS.omega]]),
        jnp.array([[GAS.tref]]), jnp.array([[GAS.alpha]]),
    )  # fmt: skip
    vth = math.sqrt(K_BOLTZ * TEMP / GAS.mass)
    dt = DT_MFP * lam / vth
    volume = float(jnp.prod(grid.hi - grid.lo))
    fnum = NRHO * volume / n_part
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.full((n_cells,), fnum),
                       epsilon=jnp.float64(0))  # fmt: skip
    static = SimStatic(N=n_part, D=len(grid.n), S=1, G=1, A_max=a_max, K_bounce=3,
                       n_steps=BLOCK)  # fmt: skip
    key = jax.random.PRNGKey(seed)
    k = 2.0 * math.pi / length
    x = sampling.sample_positions(jax.random.fold_in(key, 1), n_part, grid)
    if wave == "thermal":
        x0 = isobaric_positions(jax.random.fold_in(key, 3), n_part, length, k, THERMAL_AMPLITUDE)
        x = x.at[:, 0].set(x0[:, 0])
    v = sampling.sample_maxwellian(jax.random.fold_in(key, 2), n_part, jnp.float64(TEMP),
                                   sp.mass[0], jnp.zeros(3))  # fmt: skip
    if wave == "shear":
        v = v.at[:, 1].add(amplitude * vth * jnp.sin(k * x[:, 0]))
    else:
        v = v * jnp.sqrt(1.0 + THERMAL_AMPLITUDE * jnp.sin(k * x[:, :1]))
    state = ParticleState(x=x, v=v, species=jnp.zeros(n_part, jnp.int32),
                          alive=jnp.ones(n_part, bool), logh_acc=jnp.zeros(n_part))  # fmt: skip
    cstate = CollideState(vremax=vremax_init(
        jnp.full((n_cells,), TEMP), sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0],
        sp.prefactor[0, 0], n_cells, 1,
    ))  # fmt: skip
    consts = {"lambda": lam, "length": length, "k": k, "vth": vth, "dt": dt, "cells": cells,
              "ny": ny, "n_part": n_part, "mu_ce": viscosity_ce(TEMP),
              "rho": NRHO * GAS.mass}  # fmt: skip
    return state, cstate, grid, sp, params, static, key, consts


def _advance(state, cstate, key, grid, sp, params, step0, theta, static, collide_fn):
    """One block of ``static.n_steps`` steps of one seed from step ``step0``."""
    (state, cstate), (_, diag) = simulate.run(
        state, cstate, grid, sp, params, static, key, step0=step0, emit_moments="none",
        collide_fn=collide_fn, collision_params=theta,
    )  # fmt: skip
    return state, cstate, jnp.sum(diag.attempts_clipped)


@functools.partial(jax.jit, static_argnames=("static", "collide_fn"))
def advance(states, cstates, keys, grid, sp, params, step0, theta, static, collide_fn):
    """One block for a batch of seeds (leading axis of ``states``, ``cstates``, ``keys``)."""
    body = functools.partial(_advance, grid=grid, sp=sp, params=params, step0=step0,
                             theta=theta, static=static, collide_fn=collide_fn)  # fmt: skip
    states, cstates, clipped = jax.vmap(body)(states, cstates, keys)
    return states, cstates, jnp.sum(clipped)


def _stack(*leaves):
    return jnp.stack(leaves)


def amplitude(state, k, vth, wave="shear"):
    """The wave's amplitude over the alive particles: ``(2 / N) sum v_y sin(k x)`` (shear) or
    the entropy mode ``-(2 / N) sum (1 - 3 e / 5) sin(k x)`` (thermal; see the module
    docstring)."""
    w = state.alive.astype(jnp.float64)
    if wave == "shear":
        f = state.v[:, 1]
    else:
        f = 0.2 * jnp.sum(state.v**2, axis=1) / vth**2 - 1.0
    return 2.0 * jnp.sum(w * f * jnp.sin(k * state.x[:, 0])) / jnp.sum(w)


def fit_decay(t, a, t0):
    """``gamma`` of ``A0 exp(-gamma t)`` fitted to the seed-mean amplitude ``a`` over
    ``t >= t0`` by Gauss-Newton on ``A`` itself (a fit of ``log A`` is biased where the
    signal-to-noise ratio is low).  Each step changes ``gamma`` by at most half its value; a
    fit that still fails (a noisy bootstrap resample) returns the log-linear start."""
    keep = t >= t0
    tt, aa = t[keep], a[keep]
    start = -np.polyfit(tt, np.log(np.maximum(aa, 1e-3 * aa[0])), 1)[0]
    gamma, a0 = start, aa[0] * np.exp(start * tt[0])
    for _ in range(50):
        model = a0 * np.exp(-gamma * tt)
        jac = np.stack([model / a0, -tt * model], axis=1)
        step = np.linalg.lstsq(jac, aa - model, rcond=None)[0]
        step *= min(1.0, 0.5 * abs(gamma) / max(abs(step[1]), 1e-300))
        a0, gamma = a0 + step[0], gamma + step[1]
        if not (np.isfinite(gamma) and gamma > 0.0):
            return start
        if abs(step[1]) < 1e-12 * abs(gamma):
            break
    return gamma


def run_batch(name, dx_mfp, ppc, seeds, decays, a_max, wave="shear", length_mfp=LENGTH_MFP,
              ny=1):  # fmt: skip
    """The amplitude series of the seeds ``seeds`` run as one vmapped batch:
    ``(t, A[seeds, T], attempts_clipped, constants)``."""
    cases = [setup(dx_mfp, ppc, s, a_max, wave, length_mfp, ny) for s in seeds]
    _, _, grid, sp, params, static, _, c = cases[0]
    states = jax.tree.map(_stack, *[case[0] for case in cases])
    cstates = jax.tree.map(_stack, *[case[1] for case in cases])
    keys = jnp.stack([case[6] for case in cases])
    collide_fn, theta, pairing = operator(name)
    static = dataclasses.replace(static, pairing=pairing)
    amp = jax.jit(jax.vmap(functools.partial(amplitude, k=c["k"], vth=c["vth"], wave=wave)))
    t_end = decays / (RATE_CE[wave] * c["mu_ce"] / c["rho"] * c["k"] ** 2)
    blocks = math.ceil(t_end / (c["dt"] * BLOCK))
    amps, clipped = [np.asarray(amp(states))], 0
    for b in range(blocks):
        states, cstates, clip = advance(states, cstates, keys, grid, sp, params, b * BLOCK,
                                        theta, static=static, collide_fn=collide_fn)  # fmt: skip
        clipped += int(clip)
        amps.append(np.asarray(amp(states)))
    t = c["dt"] * BLOCK * np.arange(blocks + 1)
    return t, np.stack(amps, axis=1), clipped, c


def measure(name, dx_mfp, ppc, seeds, decays=2.0, a_max=0, n_boot=200, wave="shear",
            length_mfp=LENGTH_MFP, ny=1, seed0=0, batch_particles=1e5):  # fmt: skip
    """``mu_eff / mu_CE`` (shear) or ``kappa_eff / kappa_CE`` (thermal) of an operator at one
    cell size: the decay of the seed-mean amplitude (skipping the first 20 mean collision
    times), its bootstrap standard error over the seeds, and the series.  Seeds run in
    vmapped batches of about ``batch_particles`` particles."""
    t0 = time.time()
    n_part = round(length_mfp / dx_mfp) * ny * ppc
    batch = max(1, min(seeds, int(batch_particles // n_part)))
    series, clipped = [], 0
    for first in range(seed0, seed0 + seeds, batch):
        ids = range(first, min(first + batch, seed0 + seeds))
        t, a, clip, c = run_batch(name, dx_mfp, ppc, ids, decays, a_max, wave, length_mfp, ny)
        series.append(a)
        clipped += clip
    series = np.concatenate(series)
    start = 20.0 * c["lambda"] / c["vth"]
    to_mu = c["rho"] / c["k"] ** 2 / c["mu_ce"] / RATE_CE[wave]
    ratio = fit_decay(t, series.mean(axis=0), start) * to_mu
    rng = np.random.default_rng(0)
    boot = [fit_decay(t, series[rng.integers(0, seeds, seeds)].mean(axis=0), start) * to_mu
            for _ in range(n_boot)]  # fmt: skip
    return {"wave": wave, "length_mfp": length_mfp, "ny": ny, "operator": name,
            "dx_mfp": dx_mfp, "ppc": ppc, "seeds": seeds, "seed0": seed0,
            "cells": c["cells"], "n_part": c["n_part"], "ratio": ratio,
            "ratio_se": float(np.std(boot)), "attempts_clipped": clipped,
            "seconds": time.time() - t0, "t": t.tolist(),
            "amplitude_mean": series.mean(axis=0).tolist(),
            "amplitude_var": series.var(axis=0, ddof=1).tolist()}  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wave", default="shear", choices=tuple(RATE_CE))
    ap.add_argument("--dx-mfp", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--operators", nargs="+", default=list(OPERATORS), choices=OPERATORS)
    ap.add_argument(
        "--particles",
        type=float,
        default=4e5,
        help="particle-runs per cell size: seeds = particles / N (at least 4)",
    )
    ap.add_argument("--ppc", type=int, default=50)
    ap.add_argument("--length-mfp", type=float, default=LENGTH_MFP, help="box length / lambda")
    ap.add_argument("--ny", type=int, default=1, help="2-D box this many cells high")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (to extend a result)")
    ap.add_argument("--batch-particles", type=float, default=1e5)
    ap.add_argument("--a-max", type=int, default=0, help="0: auto_a_max(ppc)")
    ap.add_argument("--tag", default="gate")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.wave}_wave_{args.tag}.json"
    label = "mu/mu_CE" if args.wave == "shear" else "kappa/kappa_CE"
    rows = []
    for name in args.operators:
        for dx in args.dx_mfp:
            n_part = round(args.length_mfp / dx) * args.ny * args.ppc
            seeds = max(4, round(args.particles / n_part))
            row = measure(name, dx, args.ppc, seeds, a_max=args.a_max, wave=args.wave,
                          length_mfp=args.length_mfp, ny=args.ny, seed0=args.seed0,
                          batch_particles=args.batch_particles)  # fmt: skip
            rows.append(row)
            print(f"{name:12s} dx/lambda {dx:5.2f} ({seeds} seeds): {label} "
                  f"{row['ratio']:.4f} +- {row['ratio_se']:.4f}  clipped "
                  f"{row['attempts_clipped']}  {row['seconds']:.0f} s", flush=True)  # fmt: skip
            out.write_text(json.dumps(rows))
    print("wrote", out)


if __name__ == "__main__":
    main()
