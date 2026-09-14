"""Steady normal shock held by two-sided inflow ([P3] test3 geometry).

The companion to :mod:`examples.shock.normal`, which drives its shock with a piston off
a specular wall.  Here the shock is held stationary by **inflow at both ends**: the left
face injects the upstream state, the right face the Rankine-Hugoniot post-shock state.
Both are ``BC.INFLOW`` -- the downstream stream points *out* of the domain, so
``mol_inflow(s)`` with ``s << 0`` supplies only the small inward-moving tail, which is
exactly the boundary condition a steady shock needs.

This is the verification case for the inflow boundary condition, and it checks two
different things at once:

* the inflow BC delivers the state it was asked for (the **upstream** rows), and
* the interior physics reproduces Rankine-Hugoniot from it (the **downstream** rows).

Measured at Mach 2 in argon, ``L = 60 lambda``, 200 cells, 40 000 particles, 20 000
warm-up steps:

| | measured | Rankine-Hugoniot | error |
|---|---|---|---|
| ``nrho`` upstream | 9.9814e+20 | 1.0000e+21 | -0.19 % |
| ``T`` upstream | 2.7128e+02 | 2.7315e+02 | -0.68 % |
| ``ux`` upstream | 6.1614e+02 | 6.1580e+02 | +0.05 % |
| ``nrho`` downstream | 2.2705e+21 | 2.2857e+21 | -0.66 % |
| ``T`` downstream | 5.6614e+02 | 5.6764e+02 | -0.26 % |
| ``ux`` downstream | 2.7042e+02 | 2.6941e+02 | +0.37 % |

with the mass flux ``nrho * ux`` conserved end to end to 0.17 %.

> **Initialise with the shock already in the middle.**  Filling the box with the upstream
> state instead makes the shock form at the outlet and never travel back: measured, it
> parked at ``x/L = 0.900`` with only **6.0 mfp** of downstream run, which is less than a
> Mach-2 shock is thick.  The "downstream plateau" was then inside the shock structure and
> read 9 % off Rankine-Hugoniot -- with the mass flux still conserved to 0.07 %, which is
> what showed the solver was fine and the *layout* was not.  Placing the shock at
> mid-domain gives 16.5 mfp of run and the table above.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.shock.normal import mfp_vss, rankine_hugoniot  # noqa: E402
from jax_dsmc import inflow, sample, sampling, simulate, sort  # noqa: E402
from jax_dsmc import species_data as sd  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams  # noqa: E402
from jax_dsmc.collide import species_params, vremax_init  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.state import (  # noqa: E402
    CollideState,
    GridSpec,
    ParticleState,
    SimParams,
    SimStatic,
)

GAS = sd.ARGON
TEMP1 = 273.15
NRHO1 = 1.0e21
GAMMA = 5.0 / 3.0
BLOCK = 200


@dataclasses.dataclass(frozen=True)
class Case:
    mach: float = 2.0
    n_cells: int = 200
    npc: int = 200
    gap_mfp: float = 60.0
    a_max: int = 128
    capacity: int = 256
    """Static per-face injection bound.  ``inflow_shortfall`` reports if it binds."""


def states(case: Case):
    """Upstream and Rankine-Hugoniot downstream states."""
    rh = rankine_hugoniot(case.mach, GAMMA)
    a1 = math.sqrt(GAMMA * K_BOLTZ * TEMP1 / GAS.mass)
    u1 = case.mach * a1
    return {
        "rh": rh,
        "a1": a1,
        "u1": u1,
        "nrho2": NRHO1 * rh["rho"],
        "temp2": TEMP1 * rh["temp"],
        "u2": u1 / rh["rho"],
    }


def build(case: Case):
    st = states(case)
    lam = mfp_vss(NRHO1)
    length = case.gap_mfp * lam
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.full(1, length), n=(case.n_cells,))
    n_cell = num_cells(grid)
    n_part = case.npc * n_cell
    sp = species_params(
        jnp.array([GAS.mass]),
        jnp.array([[GAS.diam]]),
        jnp.array([[GAS.omega]]),
        jnp.array([[GAS.tref]]),
        jnp.array([[GAS.alpha]]),
    )
    vth = math.sqrt(K_BOLTZ * TEMP1 / GAS.mass)
    dt = 0.2 * (length / case.n_cells) / max(vth, st["u1"])
    fnum = NRHO1 * length / n_part * 1.5
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.full((n_cell,), fnum), epsilon=jnp.float64(0.0))
    static = SimStatic(N=n_part, D=1, S=1, G=1, A_max=case.a_max, K_bounce=3, n_steps=BLOCK)
    faces = Faces((BC.INFLOW, BC.INFLOW))
    res = inflow.InflowParams(
        nrho=jnp.array([NRHO1, st["nrho2"]]),
        temp=jnp.array([TEMP1, st["temp2"]]),
        vstream=jnp.array([[st["u1"], 0.0, 0.0], [st["u2"], 0.0, 0.0]]),
    )
    return grid, sp, params, static, faces, res, lam, length, fnum, st


def initial_state(key, grid, sp, n_part, length, st):
    """Shock **already at mid-domain** -- see the module docstring for why."""
    x = sampling.sample_positions(jax.random.fold_in(key, 1), n_part, grid)
    left = x[:, 0] < 0.5 * length
    v_up = sampling.sample_maxwellian(
        jax.random.fold_in(key, 2),
        n_part,
        jnp.float64(TEMP1),
        sp.mass[0],
        jnp.array([st["u1"], 0.0, 0.0]),
    )
    v_dn = sampling.sample_maxwellian(
        jax.random.fold_in(key, 3),
        n_part,
        jnp.float64(st["temp2"]),
        sp.mass[0],
        jnp.array([st["u2"], 0.0, 0.0]),
    )
    u = jax.random.uniform(jax.random.fold_in(key, 4), (n_part,), dtype=jnp.float64)
    frac_up = 1.0 / (1.5 * st["rh"]["rho"])
    alive = jnp.where(left, u < frac_up, u < frac_up * st["rh"]["rho"])
    return ParticleState(
        x=x,
        v=jnp.where(left[:, None], v_up, v_dn),
        species=jnp.zeros(n_part, jnp.int32),
        alive=alive,
        logh_acc=jnp.zeros(n_part),
    )


def profile(case: Case, key=None, n_warm=20_000, n_sample=10_000):
    """Run to steady state; return the profiles and the plateau comparison."""
    key = jax.random.PRNGKey(0) if key is None else key
    grid, sp, params, static, faces, res, lam, length, fnum, st = build(case)
    n_cell = num_cells(grid)
    state = initial_state(key, grid, sp, static.N, length, st)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((n_cell,), max(TEMP1, st["temp2"])),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            n_cell,
            1,
        )
    )
    wall = WallParams.uniform(1, TEMP1)
    run = jax.jit(
        lambda s, c, k0: simulate.run(
            s,
            c,
            grid,
            sp,
            params,
            static,
            key,
            faces=faces,
            wall=wall,
            res=res,
            capacity=case.capacity,
            emit_moments=False,
            step0=k0,
        )
    )
    diag = None
    for blk in range(n_warm // BLOCK):
        (state, cstate), (_, diag) = run(state, cstate, blk * BLOCK)

    acc = {k: np.zeros(n_cell) for k in ("n", "ux", "temp")}
    blocks = n_sample // BLOCK
    for blk in range(blocks):
        (state, cstate), _ = run(state, cstate, n_warm + blk * BLOCK)
        table = sort.build_table(state, grid)
        moments = sample.moments(state, table, grid, params, sp)
        cell = np.asarray(table.cell)
        w = np.asarray(state.alive, dtype=float)
        cnt = np.bincount(cell, weights=w, minlength=n_cell + 1)[:n_cell]
        acc["n"] += cnt
        acc["ux"] += np.bincount(cell, weights=w * np.asarray(state.v[:, 0]), minlength=n_cell + 1)[
            :n_cell
        ]
        acc["temp"] += np.asarray(moments.temp) * cnt

    dx = length / n_cell
    nrho = acc["n"] / blocks * fnum / dx
    ux = acc["ux"] / np.maximum(acc["n"], 1)
    temp = acc["temp"] / np.maximum(acc["n"], 1)
    edge = n_cell // 5
    ratio = nrho / NRHO1
    half = 0.5 * (ratio[:edge].mean() + ratio[-edge:].mean())
    i_shock = int(np.argmax(ratio > half))
    flux = nrho * ux
    return {
        "x": (np.arange(n_cell) + 0.5) * dx,
        "nrho": nrho,
        "ux": ux,
        "temp": temp,
        "mfp": lam,
        "length": length,
        "shock_frac": i_shock / n_cell,
        "downstream_mfp": (n_cell - i_shock) * dx / lam,
        "flux_ratio": float(flux[-edge:].mean() / flux[:edge].mean()),
        "shortfall": float(jnp.max(diag.inflow_shortfall)),
        "clipped": int(jnp.sum(diag.attempts_clipped)),
        "measured": {
            "nrho1": float(nrho[:edge].mean()),
            "temp1": float(temp[:edge].mean()),
            "ux1": float(ux[:edge].mean()),
            "nrho2": float(nrho[-edge:].mean()),
            "temp2": float(temp[-edge:].mean()),
            "ux2": float(ux[-edge:].mean()),
        },
        "target": {
            "nrho1": NRHO1,
            "temp1": TEMP1,
            "ux1": st["u1"],
            "nrho2": st["nrho2"],
            "temp2": st["temp2"],
            "ux2": st["u2"],
        },
    }
