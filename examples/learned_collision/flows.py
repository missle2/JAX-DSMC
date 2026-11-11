"""The round-1 flows (plan 2026-11-11): four 2-D shear waves and four 2-D normal shocks, each
a planar flow in a slab periodic in ``y``, and the twelve coarse cases built on them.

* **Shear waves** (:mod:`.shear_wave`, a 2-D slab ``SHEAR_HEIGHT`` mean free paths high):
  amplitude ``u0 / v_th`` in {0.1, 0.5} x wavelength ``L / lambda`` in {32, 128}, coarse
  cells ``dx_c = 2 lambda`` (8x the fine ``lambda / 4``).
* **Normal shocks** held by inflow at both ends (the 1-D :mod:`examples.shock.inflow_driven`
  made a slab): Mach {1.5, 3, 6, 10}, ``SHOCK_LENGTH`` upstream mean free paths long with
  the shock started at ``SHOCK_X``; coarse cells ``dx_c / lambda_1`` in {1, 4}, bracketing
  the 8x ``2 lambda_1`` -- a shock has no Knudsen number of its own (its structure scales
  with ``lambda_1``), so the second axis is the resolution.  One fine reference per Mach
  serves both resolutions.

Train on the extreme corners (S2, S3, K1, K8), test on the other eight.

Every slab is tall enough for ``N_SLAB`` particles per ``lambda`` of ``x`` at the coarse
particles per cell: the 1-D collective (long-time-tail) transport of few simulators per
``lambda`` grew the shear-wave viscosity with box length (1.42 / 1.43 / 1.70 at 32 / 64 /
128 lambda, 5 particles per lambda) -- not a local error, so kept out of the training data.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from examples.couette import GAS, NRHO, TEMP, mean_free_path
from examples.learned_collision import shear_wave
from examples.learned_collision.shear_wave import BLOCK, auto_a_max
from examples.shock.normal import rankine_hugoniot
from jax_dsmc import inflow, sampling, simulate
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.collide import species_params, vremax_init
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic

GAMMA = 5.0 / 3.0
DX_FINE = 0.25
"""Fine reference cells in (upstream) mean free paths."""
PPC_FINE = 20
PPC_COARSE = 20
N_SLAB = 50.0
"""Smallest coarse particle count per ``lambda`` of ``x`` (summed over the slab height)."""
SHEAR_HEIGHT = 10.0
SHOCK_LENGTH = 60.0
SHOCK_X = 25.0
SHOCK_HEIGHT = 40.0

REFERENCES = {
    "shear_a0.1_L32": {"kind": "shear", "amplitude": 0.1, "length": 32.0},
    "shear_a0.1_L128": {"kind": "shear", "amplitude": 0.1, "length": 128.0},
    "shear_a0.5_L32": {"kind": "shear", "amplitude": 0.5, "length": 32.0},
    "shear_a0.5_L128": {"kind": "shear", "amplitude": 0.5, "length": 128.0},
    "shock_M1.5": {"kind": "shock", "mach": 1.5},
    "shock_M3": {"kind": "shock", "mach": 3.0},
    "shock_M6": {"kind": "shock", "mach": 6.0},
    "shock_M10": {"kind": "shock", "mach": 10.0},
}
"""The fine runs: one per flow, whatever the coarse resolutions read from it."""
for _ref in REFERENCES.values():
    _ref["height"] = SHEAR_HEIGHT if _ref["kind"] == "shear" else SHOCK_HEIGHT
    if _ref["kind"] == "shock":
        _ref["length"] = SHOCK_LENGTH

CASES = {
    "S1": ("shear_a0.1_L32", 2.0, "test"),
    "S2": ("shear_a0.1_L128", 2.0, "train"),
    "S3": ("shear_a0.5_L32", 2.0, "train"),
    "S4": ("shear_a0.5_L128", 2.0, "test"),
    "K1": ("shock_M1.5", 1.0, "train"),
    "K2": ("shock_M1.5", 4.0, "test"),
    "K3": ("shock_M3", 1.0, "test"),
    "K4": ("shock_M3", 4.0, "test"),
    "K5": ("shock_M6", 1.0, "test"),
    "K6": ("shock_M6", 4.0, "test"),
    "K7": ("shock_M10", 1.0, "test"),
    "K8": ("shock_M10", 4.0, "train"),
}
"""``name -> (reference, dx_c / lambda, role)``."""


def coarse_resolutions(ref_name):
    """The coarse cell sizes the cases read from reference ``ref_name``."""
    return tuple(sorted({dx for ref, dx, _ in CASES.values() if ref == ref_name}))


def thermal_speed():
    """``sqrt(k T_1 / m)`` [m/s]."""
    return math.sqrt(K_BOLTZ * TEMP / GAS.mass)


def shock_states(mach):
    """Upstream speed and the Rankine-Hugoniot downstream state (SI)."""
    rh = rankine_hugoniot(mach, GAMMA)
    u1 = mach * math.sqrt(GAMMA) * thermal_speed()
    return {"ratio": rh["rho"], "u1": u1, "u2": u1 / rh["rho"], "nrho2": NRHO * rh["rho"],
            "temp2": TEMP * rh["temp"]}  # fmt: skip


def dt_mfp(ref):
    """The time step in ``lambda_1 / v_th`` (the same for the fine and coarse runs): the
    shear waves' 0.05, and ``0.05 / (1 + u_1 / v_th)`` for a shock, so the hot compressed
    gas behind a Mach 10 shock still sees ``dt / tau_2 ~ 0.03``."""
    if ref["kind"] == "shear":
        return shear_wave.DT_MFP
    return shear_wave.DT_MFP / (1.0 + shock_states(ref["mach"])["u1"] / thermal_speed())


def window_height(dx_c, ppc=PPC_COARSE, height=SHOCK_HEIGHT):
    """The coarse window's slab height [lambda]: the smallest multiple of ``dx_c`` with
    ``ppc H / dx_c^2 >= N_SLAB`` (the slab is periodic in ``y``, so a window may take any
    stripe of it), at most ``height``."""
    rows = math.ceil(N_SLAB * dx_c / ppc - 1e-9)
    return min(rows * dx_c, height)


def thinning(dx_c, ppc_c=PPC_COARSE, dx_f=DX_FINE, ppc_f=PPC_FINE, dim=2):
    """The fraction of fine particles a coarse run keeps (equal number density)."""
    return (ppc_c / dx_c**dim) / (ppc_f / dx_f**dim)


def _flux(nrho, temp, u):
    """Number flux [1/(m^2 s)] of a drifting Maxwellian through a face, into the box when
    ``u > 0`` (the half-range moment)."""
    v_mp = math.sqrt(2.0 * K_BOLTZ * temp / GAS.mass)
    s = u / v_mp
    return nrho * v_mp / (2.0 * math.sqrt(math.pi)) * (
        math.exp(-s * s) + math.sqrt(math.pi) * s * (1.0 + math.erf(s))
    )  # fmt: skip


def _maxwellian(key, n, temp, u, sp):
    return sampling.sample_maxwellian(key, n, jnp.float64(temp), sp.mass[0],
                                      jnp.array([u, 0.0, 0.0]))  # fmt: skip


def shock_setup(mach, dx_mfp, ppc, seed, height_mfp=SHOCK_HEIGHT, a_max=0,
                length_mfp=SHOCK_LENGTH, x_shock=SHOCK_X):  # fmt: skip
    """A 2-D normal-shock slab: ``(state, cstate, grid, sp, params, static, key, consts,
    bc)``, ``bc`` the :func:`jax_dsmc.simulate.run` boundary keywords (inflow of the
    upstream state at ``x = 0`` and of the Rankine-Hugoniot state at ``x = L``, periodic in
    ``y``).  The shock starts as a step at ``x_shock`` (``ppc`` particles per upstream cell)."""
    st = shock_states(mach)
    lam, vth = mean_free_path(), thermal_speed()
    nx, ny = round(length_mfp / dx_mfp), round(height_mfp / dx_mfp)
    length, height = nx * dx_mfp * lam, ny * dx_mfp * lam
    grid = GridSpec(lo=jnp.zeros(2), hi=jnp.array([length, height]), n=(nx, ny))
    sp = species_params(
        jnp.array([GAS.mass]), jnp.array([[GAS.diam]]), jnp.array([[GAS.omega]]),
        jnp.array([[GAS.tref]]), jnp.array([[GAS.alpha]]),
    )  # fmt: skip
    dt = dt_mfp({"kind": "shock", "mach": mach}) * lam / vth
    cell_area = (dx_mfp * lam) ** 2
    fnum = NRHO * cell_area / ppc
    x_s = x_shock * lam
    n_up = round(NRHO * x_s * height / fnum)
    n_dn = round(st["nrho2"] * (length - x_s) * height / fnum)
    expected = n_up + n_dn
    cap = math.ceil(1.05 * expected + 6.0 * math.sqrt(expected) + 64)
    # NTC attempts scale with n^2 sigma g dt: about ratio^2 sqrt(T2/T1) behind the shock
    mean = 0.25 * ppc * st["ratio"] ** 2 * math.sqrt(st["temp2"] / TEMP)
    mean *= dt_mfp({"kind": "shock", "mach": mach}) / shear_wave.DT_MFP
    a_max = a_max or max(auto_a_max(ppc), math.ceil(mean + 6.0 * math.sqrt(mean) + 6.0))
    inject = _flux(NRHO, TEMP, st["u1"]) * height * dt / fnum
    capacity = math.ceil(inject + 6.0 * math.sqrt(inject) + 32)
    params = SimParams(dt=jnp.float64(dt), fnum=jnp.full((nx * ny,), fnum),
                       epsilon=jnp.float64(0))  # fmt: skip
    static = SimStatic(N=cap, D=2, S=1, G=1, A_max=a_max, K_bounce=3, n_steps=BLOCK)
    key = jax.random.PRNGKey(seed)
    ku = jax.random.fold_in(key, 1)
    kd = jax.random.fold_in(key, 2)
    pos_up = jax.random.uniform(ku, (n_up, 2)) * jnp.array([x_s, height])
    pos_dn = jax.random.uniform(kd, (n_dn, 2)) * jnp.array([length - x_s, height])
    pos_dn = pos_dn + jnp.array([x_s, 0.0])
    x = jnp.zeros((cap, 2)).at[:n_up].set(pos_up).at[n_up:expected].set(pos_dn)
    v_up = _maxwellian(jax.random.fold_in(key, 3), cap, TEMP, st["u1"], sp)
    v_dn = _maxwellian(jax.random.fold_in(key, 4), cap, st["temp2"], st["u2"], sp)
    idx = jnp.arange(cap)
    v = jnp.where((idx < n_up)[:, None], v_up, v_dn)
    state = ParticleState(x=x, v=v, species=jnp.zeros(cap, jnp.int32), alive=idx < expected,
                          logh_acc=jnp.zeros(cap))  # fmt: skip
    n_cells = nx * ny
    cstate = CollideState(vremax=vremax_init(
        jnp.full((n_cells,), max(TEMP, st["temp2"])), sp.omega[0, 0], sp.tref[0, 0],
        sp.mr[0, 0], sp.prefactor[0, 0], n_cells, 1,
    ))  # fmt: skip
    bc = {
        "faces": Faces((BC.INFLOW, BC.INFLOW, BC.PERIODIC, BC.PERIODIC)),
        "wall": WallParams.uniform(2, TEMP),
        "res": inflow.InflowParams(
            nrho=jnp.array([NRHO, st["nrho2"], NRHO, NRHO]),
            temp=jnp.array([TEMP, st["temp2"], TEMP, TEMP]),
            vstream=jnp.array(
                [[st["u1"], 0.0, 0.0], [st["u2"], 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        ),
        "capacity": capacity,
    }
    consts = {"lambda": lam, "vth": vth, "dt": dt, "cells": nx, "ny": ny, "n_part": expected,
              "ppc_up": ppc * ny, "ratio": st["ratio"], "x_shock": x_s, "length": length,
              "height": height, **st}  # fmt: skip
    return state, cstate, grid, sp, params, static, key, consts, bc


def shear_setup(amplitude, length_mfp, dx_mfp, ppc, seed, height_mfp=SHEAR_HEIGHT, a_max=0):
    """:func:`.shear_wave.setup` as a 2-D slab ``height_mfp`` high, with empty ``bc``."""
    ny = round(height_mfp / dx_mfp)
    out = shear_wave.setup(dx_mfp, ppc, seed, a_max, "shear", length_mfp, ny, amplitude)
    out[-1]["ppc_up"] = ppc * ny
    return (*out, {})


def setup(ref, dx_mfp, ppc, seed, height_mfp=None, a_max=0):
    """:func:`shock_setup` or :func:`shear_setup` of a :data:`REFERENCES` entry."""
    height = ref["height"] if height_mfp is None else height_mfp
    if ref["kind"] == "shock":
        return shock_setup(ref["mach"], dx_mfp, ppc, seed, height, a_max)
    return shear_setup(ref["amplitude"], ref["length"], dx_mfp, ppc, seed, height, a_max)


def run_kwargs(bc):
    """``(faces, capacity)`` (static) and ``(res, wall)`` (traced) of a ``bc`` dict."""
    return bc.get("faces"), bc.get("capacity", 0), bc.get("res"), bc.get("wall")


def _advance(state, cstate, key, grid, sp, params, step0, theta, res, wall, static,
             collide_fn, faces, capacity):  # fmt: skip
    (state, cstate), (_, diag) = simulate.run(
        state, cstate, grid, sp, params, static, key, step0=step0, emit_moments="none",
        faces=faces, wall=wall, res=res, capacity=capacity, collide_fn=collide_fn,
        collision_params=theta,
    )  # fmt: skip
    return state, cstate, jnp.sum(diag.attempts_clipped), jnp.max(diag.inflow_shortfall)


@functools.partial(jax.jit, static_argnames=("static", "collide_fn", "faces", "capacity"))
def advance(states, cstates, keys, grid, sp, params, step0, theta, res, wall, static,
            collide_fn, faces, capacity):  # fmt: skip
    """One block of ``static.n_steps`` steps for a batch of seeds (leading axis of
    ``states``, ``cstates``, ``keys``): ``(states, cstates, clipped, shortfall)``."""
    body = functools.partial(_advance, grid=grid, sp=sp, params=params, step0=step0,
                             theta=theta, res=res, wall=wall, static=static,
                             collide_fn=collide_fn, faces=faces, capacity=capacity)  # fmt: skip
    states, cstates, clipped, short = jax.vmap(body)(states, cstates, keys)
    return states, cstates, jnp.sum(clipped), jnp.max(short)


def shock_position(count, x_centres, ratio, ppc_up):
    """The shock's equivalent-jump position [same units as ``x_centres``] from per-bin
    counts along ``x`` (summed over ``y``): ``x_0 + integral (ratio - n) / (ratio - 1) dx``
    with ``n`` the density relative to upstream -- the mass integral, which averages the
    noise of every bin (a level crossing would pick up one bin's)."""
    count = np.asarray(count, dtype=float)
    dx = float(x_centres[1] - x_centres[0])
    n_rel = count / (ppc_up * 1.0)
    x0 = float(x_centres[0]) - 0.5 * dx
    return x0 + np.sum((ratio - n_rel) / (ratio - 1.0), axis=-1) * dx


def stack(*leaves):
    """``jnp.stack`` (a module-level body for ``jax.tree.map``)."""
    return jnp.stack(leaves)
