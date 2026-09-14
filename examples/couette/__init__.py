"""Planar Couette flow: the M4 physics gate for wall boundary conditions.

Two diffuse walls slide in opposite directions and shear the gas between them.  The
gate is quantitative and external: the **measured** shear viscosity

    mu = -tau_ns / (du_s/dn)

taken in the bulk, against the Chapman-Enskog VSS formula (Bird 4.64).  Nothing in the
solver is told that number, so it is a target rather than an input -- the same standard
as the BKW gate at M1 and the Taylor-Green viscosity gate in ``examples/turbulence``.

**Geometry, and why it looks transposed.**  The spatial domain is 1-D and *is* the gap,
so the gap direction is spatial axis 0, which is velocity component 0.  The streamwise
direction must therefore be a velocity component that does **not** advect -- component 1.
Getting this backwards asks for a wall sliding along its own normal, which
``WallParams.with_wall_velocity`` refuses (it would need SPARTA's data-dependent
rejection sampler).

**Resolution matters more than anything else here.**  Measured, at `L = 30 lambda`:

| ``dx/lambda`` | 0.60 | 0.30 | **0.15** |
|---|---|---|---|
| ``mu_measured / mu_CE`` | 2.09 | 1.028 | **0.998** |

The coarse point is outside DSMC's own requirement that a cell be well under a mean free
path, so it is not on the asymptotic curve and no convergence order is claimed from it.
What the table does establish is that the gate is resolution-limited, not model-limited.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sample, sampling, simulate, sort  # noqa: E402
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
TEMP = 273.15
NRHO = 1.0e21


def gap_axis(dim: int) -> int:
    """The wall-normal spatial axis: always the **last** one, periodic on the rest."""
    return dim - 1


def stream_comp(dim: int) -> int:
    """The velocity component the walls drag.  Never the gap axis.

    In 1-D the single spatial axis *is* the gap and is velocity component 0, so the
    streamwise component has to be one that does not advect -- component 1.  In 2-D and
    3-D the gap is the last axis and the streamwise direction is component 0, which does
    advect and is periodic: the ordinary planar-Couette arrangement, and the one [DF]'s
    ``couette_N2-O2`` case uses.
    """
    return 1 if gap_axis(dim) == 0 else 0


BLOCK = 200
"""Steps per jitted call.  Chunking does not change a single random draw (section 1)."""


def mean_free_path(nrho=NRHO, gas=GAS):
    """Bird 4.52 for a VSS gas: ``(a+1)(a+2)/(6a) / (sqrt(2) pi d^2 n)``."""
    a, d = gas.alpha, gas.diam
    return (a + 1.0) * (a + 2.0) / (6.0 * a) / (math.sqrt(2.0) * math.pi * d * d * nrho)


def viscosity_ce(temp, gas=GAS):
    """Chapman-Enskog VSS viscosity, Bird 4.64 [Pa s].  The **target** of this example."""
    a, w, d, tref, m = gas.alpha, gas.omega, gas.diam, gas.tref, gas.mass
    pref = (
        5.0
        * (a + 1.0)
        * (a + 2.0)
        * math.sqrt(math.pi * m * K_BOLTZ * tref)
        / (4.0 * a * (5.0 - 2.0 * w) * (7.0 - 2.0 * w) * math.pi * d * d)
    )
    return pref * (temp / tref) ** w


@dataclasses.dataclass(frozen=True)
class Case:
    dim: int = 1
    """Spatial dimensions.  The gap is resolved in every case; the periodic directions
    only add statistics, so ``D = 2`` and ``D = 3`` are a *check* that the wall physics is
    dimension-agnostic rather than a different experiment."""
    n_cells: int = 200
    npc: int = 400
    gap_mfp: float = 30.0
    u_wall: float = 200.0
    acc: float = 1.0
    epsilon_rel: float = 0.0
    a_max: int = 64
    specular: bool = False
    """Specular walls transfer no tangential momentum, so they must drive no shear."""

    n_span: int = 4
    """Cells along each periodic direction (``D >= 2`` only)."""

    @property
    def dx_over_mfp(self):
        return self.gap_mfp / self.n_cells

    @property
    def gap(self):
        return gap_axis(self.dim)

    @property
    def stream(self):
        return stream_comp(self.dim)


def build(case: Case):
    lam = mean_free_path()
    length = case.gap_mfp * lam
    dim, gap = case.dim, case.gap
    # the gap is fully resolved; the periodic directions are short and coarse, since they
    # exist only to add samples, not structure
    span = case.n_span * (length / case.n_cells)
    hi = [span] * dim
    n = [case.n_span] * dim
    hi[gap] = length
    n[gap] = case.n_cells
    grid = GridSpec(lo=jnp.zeros(dim), hi=jnp.asarray(hi), n=tuple(n))
    n_cell = num_cells(grid)
    n_part = case.npc * n_cell
    sp = species_params(
        jnp.array([GAS.mass]),
        jnp.array([[GAS.diam]]),
        jnp.array([[GAS.omega]]),
        jnp.array([[GAS.tref]]),
        jnp.array([[GAS.alpha]]),
    )
    vth = math.sqrt(K_BOLTZ * TEMP / GAS.mass)
    dt = 0.2 * (length / case.n_cells) / vth
    volume = float(np.prod(np.asarray(hi)))
    fnum = NRHO * volume / n_part
    params = SimParams(
        dt=jnp.float64(dt),
        fnum=jnp.full((n_cell,), fnum),
        epsilon=jnp.float64(case.epsilon_rel * dt),
    )
    static = SimStatic(N=n_part, D=dim, S=1, G=1, A_max=case.a_max, K_bounce=3, n_steps=BLOCK)
    bc = BC.SPECULAR if case.specular else BC.DIFFUSE
    rules = [BC.PERIODIC] * (2 * dim)
    rules[2 * gap] = bc
    rules[2 * gap + 1] = bc
    faces = Faces(tuple(rules))
    wall = WallParams.uniform(dim, TEMP, acc=case.acc)
    drag = np.zeros(3)
    drag[case.stream] = case.u_wall
    wall = wall.with_wall_velocity(gap, 0, jnp.asarray(-drag)).with_wall_velocity(
        gap, 1, jnp.asarray(drag)
    )
    return grid, sp, params, static, faces, wall, lam, length, fnum


def initial_state(key, grid, sp, n_part):
    return ParticleState(
        x=sampling.sample_positions(jax.random.fold_in(key, 1), n_part, grid),
        v=sampling.sample_maxwellian(
            jax.random.fold_in(key, 2), n_part, jnp.float64(TEMP), sp.mass[0], jnp.zeros(3)
        ),
        species=jnp.zeros(n_part, jnp.int32),
        alive=jnp.ones(n_part, bool),
        logh_acc=jnp.zeros(n_part),
    )


def _accumulate(run, state, cstate, grid, gap, stream, n_slab, dy, n_warm, blocks):
    """Bin particle moments onto gap-axis slabs.  Split out to keep `profile` readable."""
    acc = {k: np.zeros(n_slab) for k in ("n", "us", "un", "uns", "v2")}
    for blk in range(blocks):
        (state, cstate), _ = run(state, cstate, n_warm + blk * BLOCK)
        j = np.clip(
            ((np.asarray(state.x[:, gap]) - float(grid.lo[gap])) / dy).astype(int),
            0,
            n_slab - 1,
        )
        w = np.asarray(state.alive, dtype=float)
        vs = np.asarray(state.v[:, stream])
        vn = np.asarray(state.v[:, gap])
        acc["n"] += np.bincount(j, weights=w, minlength=n_slab)
        acc["us"] += np.bincount(j, weights=w * vs, minlength=n_slab)
        acc["un"] += np.bincount(j, weights=w * vn, minlength=n_slab)
        acc["uns"] += np.bincount(j, weights=w * vn * vs, minlength=n_slab)
        acc["v2"] += np.bincount(
            j, weights=w * np.asarray(np.sum(state.v**2, axis=1)), minlength=n_slab
        )
    return acc


def profile(case: Case, key=None, n_warm=None, n_sample=6000):
    """Run to steady state and return the gap-resolved profiles and bulk transport.

    Binning is on the **gap-axis slab**, not the flat cell index, so the periodic
    directions are averaged over automatically and the same code serves ``D = 1, 2, 3``.
    All moments are formed from the particles here rather than read from
    :func:`sample.moments`, so the stress and the temperature come from one consistent
    definition of the peculiar velocity.

    ``n_warm=None`` means three momentum-diffusion times for *this* case
    (:func:`diffusion_steps`), which is the only correct default: the time to steady
    state goes as ``L^2``, so any constant step count silently under-converges wide gaps.
    A constant default here is what let an earlier Knudsen sweep run its widest case for
    1.7 % of a diffusion time and report the resulting artefact as physics.  Pass an
    explicit small ``n_warm`` only for structural tests that do not read the transport
    coefficients.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    n_warm = 3 * diffusion_steps(case) if n_warm is None else int(n_warm)
    grid, sp, params, static, faces, wall, lam, length, fnum = build(case)
    n_cell = num_cells(grid)
    gap, stream = case.gap, case.stream
    n_slab = case.n_cells
    dy = length / n_slab

    state = initial_state(key, grid, sp, static.N)
    cstate = CollideState(
        vremax=vremax_init(
            jnp.full((n_cell,), TEMP),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            n_cell,
            1,
        )
    )
    run = jax.jit(
        lambda st, cs, k0: simulate.run(
            st,
            cs,
            grid,
            sp,
            params,
            static,
            key,
            faces=faces,
            wall=wall,
            emit_moments=False,
            step0=k0,
        )
    )
    diag = None
    for blk in range(n_warm // BLOCK):
        (state, cstate), (_, diag) = run(state, cstate, blk * BLOCK)

    blocks = n_sample // BLOCK
    acc = _accumulate(run, state, cstate, grid, gap, stream, n_slab, dy, n_warm, blocks)

    cnt = np.maximum(acc["n"], 1.0)
    u_s = acc["us"] / cnt
    u_n = acc["un"] / cnt
    # peculiar-velocity correlation = the off-diagonal pressure tensor
    corr = acc["uns"] / cnt - u_n * u_s
    temp = GAS.mass * (acc["v2"] / cnt - (u_s**2 + u_n**2)) / (3.0 * K_BOLTZ)
    # number density per slab: the slab volume is dy times the periodic cross-section
    cross = (
        float(np.prod([float(grid.hi[a] - grid.lo[a]) for a in range(case.dim) if a != gap]))
        if case.dim > 1
        else 1.0
    )
    nrho = acc["n"] / blocks * fnum / (dy * cross)
    tau = GAS.mass * nrho * corr
    y = (np.arange(n_slab) + 0.5) * dy

    core = slice(n_slab // 4, 3 * n_slab // 4)
    slope = float(np.polyfit(y[core], u_s[core], 1)[0])
    tau_bulk = float(tau[core].mean())
    mu_meas = -tau_bulk / slope if slope != 0.0 else float("nan")
    mu_ce = viscosity_ce(float(temp[core].mean()))
    return {
        "dim": case.dim,
        "y": y,
        "u_s": u_s,
        "tau": tau,
        "temp": temp,
        "nrho": nrho,
        "mfp": lam,
        "length": length,
        "du_dy": slope,
        "tau_bulk": tau_bulk,
        "mu_meas": mu_meas,
        "mu_ce": mu_ce,
        "ratio": mu_meas / mu_ce,
        "slip": case.u_wall - abs(float(u_s[-1])),
        "knudsen": lam / length,
        "dx_over_mfp": case.dx_over_mfp,
        "u_wall": case.u_wall,
        "acc": case.acc,
        "epsilon_rel": case.epsilon_rel,
        "specular": case.specular,
        "n_part": static.N,
        "recrossed": int(jnp.sum(diag.wall_recrossed)),
        "clipped": int(jnp.sum(diag.attempts_clipped)),
    }


def diffusion_steps(case: Case) -> int:
    """Steps for momentum to diffuse across the gap, ``L^2/nu`` in units of ``dt``.

    **The only relevant timescale for reaching steady state**, and the one this example
    originally ignored -- warm-up was chosen as a step count, so a Knudsen sweep that
    widened the gap 8x silently ran the widest case for 1.7 % of a diffusion time and
    produced a clean-looking but meaningless trend.  Scale warm-up by this, never by a
    constant.

    Note the shear *rate* settles well before ``L^2/nu`` (measured: converged at 0.28 of
    it), so this is a conservative bound rather than a threshold.
    """
    lam = mean_free_path()
    length = case.gap_mfp * lam
    nu = viscosity_ce(TEMP) / (GAS.mass * NRHO)
    vth = math.sqrt(K_BOLTZ * TEMP / GAS.mass)
    dt = 0.2 * (length / case.n_cells) / vth
    return int(length * length / nu / dt)


def slip_coefficient(acc=1.0):
    """Maxwell's first-order slip coefficient ``zeta`` for accommodation ``acc``.

    ``u_slip = zeta * lambda * du/dn`` with ``zeta = (2 - acc)/acc * 1.1466`` -- the
    1.1466 is the Cercignani hard-sphere value for full accommodation.  It diverges as
    ``acc -> 0``, which is the correct statement that a specular wall imposes no slip
    condition at all because it exerts no tangential force.
    """
    return (2.0 - acc) / acc * 1.1466


def slip_prediction(result):
    """Predicted wall slip from Maxwell's condition, for the measured shear rate."""
    return slip_coefficient(result["acc"]) * result["mfp"] * abs(result["du_dy"])
