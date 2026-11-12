"""The round-1 generalisation test (plan 2026-11-11): a Mach 10 argon circle at
``Kn_D = lambda / D = 0.042`` (``lambda = 0.25 m``, ``D = 6 m``), forward only.

Three run settings, so each place a coarse-cell collision operator can go wrong shows up
separately (2026-11-11: at 8x on the specular wall every operator's drag was within 2 % --
M 10 drag is mostly inviscid momentum flux and hides the transport error):

* ``--cells``: ``fine`` (the ``circle_xlong`` box on 0.0625 m cells, ``dx = lambda / 4``:
  the reference), ``c8`` (its own 0.5 m cells, ``2 lambda``, the training's 8x) or ``c16``
  (1 m cells, ``4 lambda``, the K-shock cases' coarsest; ``circle_xlong_h1``).  Every run at
  ``--ppc`` particles per free-stream cell and the fine run's time step.
* ``--wall``: ``specular`` (the decks') or ``diffuse`` (full accommodation at ``T_WALL =
  T_0 / 2``, half the stagnation temperature, so there is a real heat load -- a cold wall
  at M 10 makes a stagnation layer ~100x the free-stream density that even the fine run
  cannot resolve; at ``T_0 / 2`` the layer's ``lambda_w ~ 0.08 m``, ~1 fine cell).
* ``--operator`` / ``--theta``: the collision operator, in every cell (no multi-level cells
  yet), so the near-wall ``dx / lambda_local`` is outside the training range (histogram).

Measured over ``--sample`` flow-through times after ``--warm``: drag and (diffuse) heat
load per unit span (block means, standard error over blocks); the cell fields (``n``,
``|u|``, ``T``) on the run's own cells and, for the fine run, also restricted onto the 0.5 m
and 1 m cells (fluid-volume weighted), so :mod:`.circle_report` compares like with like; the
stagnation-line profiles at the run's own resolution (the two cell rows either side of
``y = y_c``).

    PYTHONPATH=. python3 -m examples.learned_collision.circle --cells fine --wall diffuse
    PYTHONPATH=. python3 -m examples.learned_collision.circle --cells c16 --operator esfp_recon
    PYTHONPATH=. python3 -m examples.learned_collision.circle --cells c8 --theta train_linear_v1.json

Writes ``examples/learned_collision/results/circle_<label>_<cells>_<wall>_<tag>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_mesh as em
from examples.learned_collision import operators
from examples.learned_collision.evaluate import load_theta
from examples.learned_collision.shear_wave import RESULTS
from examples.mesh import forward as F
from jax_dsmc import initialize, sample, simulate, sort
from jax_dsmc import species_data as sd
from jax_dsmc.grid import cell_volume, n_cartesian, num_cells
from jax_dsmc.state import FlowParams

LAMBDA = 0.25
MACH = 10.0
GAMMA = 5.0 / 3.0
T_WALL = 0.5 * F.TEMP * (1.0 + 0.5 * (GAMMA - 1.0) * MACH**2)
"""The diffuse wall's temperature: half the stagnation temperature (~4690 K)."""
DT = 2.5e-5 / 8
"""The decks' M 10 step on 0.5 m cells divided by the fine run's 8 (a free-stream move of
0.15 fine cell), for every run."""
CELLS = {"fine": ("circle_xlong", 8), "c8": ("circle_xlong", 1), "c16": ("circle_xlong_h1", 1)}
"""``cells -> (forward.CASES case, grid_scale)``."""
CELL_SIZE = {"fine": 0.0625, "c8": 0.5, "c16": 1.0}
BLOCK = 50


def config(cells, wall):
    """The :mod:`examples.ensemble_mesh` configuration of a run."""
    case, scale = CELLS[cells]
    cfg = em.default_config()
    cfg.update(case=case, knudsen=LAMBDA / (2.0 * F.CASES[case]["radius"]), mach=MACH,
               boundary="free", acc=1.0 if wall == "diffuse" else 0.0,
               t_wall=T_WALL if wall == "diffuse" else 0.0, grid_scale=scale,
               fnum_scale=1.0, dt=DT, steps=BLOCK, warmup=0)  # fmt: skip
    return cfg


def setup(cells, wall, ppc, a_max=32):
    """``(grid, mesh, sp, params, static, faces, res, wall, capacity, nrho, cfg)``."""
    cfg = config(cells, wall)
    nrho = F.NRHO * em.knudsen_density(cfg)
    grid, mesh, sp, params, static, faces, res, wall_p, _ = em.build(cfg, "radius", 3.0, BLOCK)
    area = float(np.prod(np.asarray(grid.hi - grid.lo))) / n_cartesian(grid)
    fnum = nrho * area / ppc
    params = dataclasses.replace(params, fnum=jnp.full((num_cells(grid),), fnum))
    slots = int(2.5 * ppc * n_cartesian(grid)) + 4096
    static = dataclasses.replace(static, N=slots, A_max=a_max)
    capacity = F.injection_bound(res, grid, faces, sp.mass[0], fnum, params.dt) + 64
    return grid, mesh, sp, params, static, faces, res, wall_p, capacity, nrho, cfg


@functools.partial(jax.jit, static_argnames=("faces", "static", "capacity", "collide_fn"))
def _run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, step0,
         theta, collide_fn):  # fmt: skip
    return simulate.run(state, cstate, grid, sp, params, static, key, faces=faces, wall=wall,
                        res=res, capacity=capacity, emit_moments="none", step0=step0,
                        mesh=mesh, emit_surface=True, collide_fn=collide_fn,
                        collision_params=theta)  # fmt: skip


@jax.jit
def _fields(state, grid, params, sp):
    m = sample.moments(state, sort.build_table(state, grid), grid, params, sp)
    return m.nrho, m.u, m.temp


def restrict(field, weight, grid, factor):
    """A field on the Cartesian cells averaged (``weight``-weighted) onto ``factor``-fold
    coarser cells: ``f64[nx / factor, ny / factor]`` (``factor = 1``: just reshaped)."""
    nx, ny = (int(k) for k in grid.n)
    f = np.asarray(field)[: nx * ny].reshape(nx, ny)
    w = np.asarray(weight)[: nx * ny].reshape(nx, ny)
    shape = (nx // factor, factor, ny // factor, factor)
    num = (f * w).reshape(shape).sum(axis=(1, 3))
    den = w.reshape(shape).sum(axis=(1, 3))
    return num / np.maximum(den, 1e-300)


def _advance(state, cstate, setup_, key, b, theta, fn, static):
    """One block, its ``A_max`` regrown (and the block redone) while it clips."""
    grid, mesh, sp, params, _, faces, res, wall, capacity = setup_[:9]
    regrown = 0
    while True:
        (new, new_c), aux = _run(state, cstate, grid, sp, params, key, faces, wall, res, mesh,
                                 static, capacity, b * BLOCK, theta, fn)  # fmt: skip
        diag = aux[1]
        if not int(jnp.sum(diag.attempts_clipped)):
            break
        peak = int(jnp.max(diag.max_attempts))
        static = dataclasses.replace(static, A_max=max(static.A_max + 1, math.ceil(1.4 * peak)))
        regrown += 1
    if float(jnp.max(diag.inflow_shortfall)) > 0:
        raise RuntimeError(f"inflow shortfall at block {b}: raise the slots")
    return new, new_c, aux[-1], static, regrown


def run(cells, wall, operator_name, theta, ppc, warm, sample_ft, seed=0):
    """One circle run (module docstring)."""
    t0 = time.time()
    s = setup(cells, wall, ppc)
    grid, _, sp, params, static, *_, nrho, cfg = s
    speed = em.stream_speed(cfg)
    fn, theta0, pairing = operators.operator(operator_name, "circle")
    theta = theta0 if theta is None else theta
    static = dataclasses.replace(static, pairing=pairing)
    key = jax.random.PRNGKey(seed)
    flow = FlowParams(nrho=jnp.float64(nrho), temp=jnp.float64(F.TEMP),
                      vstream=jnp.array([speed, 0.0, 0.0]))  # fmt: skip
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    flow_through = float(grid.hi[0] - grid.lo[0]) / speed
    n_warm = math.ceil(warm * flow_through / (DT * BLOCK))
    n_sample = math.ceil(sample_ft * flow_through / (DT * BLOCK))
    fnum = float(params.fnum[0])
    n_cells = num_cells(grid)
    acc = {"nrho": np.zeros(n_cells), "mom": np.zeros((n_cells, 3)), "temp_n": np.zeros(n_cells)}
    drag, heat, regrown = [], [], 0
    for b in range(n_warm + n_sample):
        state, cstate, tallies, static, k = _advance(state, cstate, s, key, b, theta, fn, static)
        regrown += k
        if b >= n_warm:
            body = slice(2 * static.D, None)
            drag.append(float(jnp.sum(tallies.momentum[:, body, 0])) * fnum / (BLOCK * DT))
            heat.append(float(jnp.sum(tallies.heat[:, body])) * fnum / (BLOCK * DT))
            n, u, temp = (np.asarray(a) for a in _fields(state, grid, params, sp))
            acc["nrho"] += n
            acc["mom"] += n[:, None] * u
            acc["temp_n"] += n * temp
    n = acc["nrho"] / n_sample
    u = acc["mom"] / np.maximum(acc["nrho"], 1e-300)[:, None]
    temp = acc["temp_n"] / np.maximum(acc["nrho"], 1e-300)
    out = _summary(grid, cells, n, u, temp, nrho, np.array(drag), np.array(heat))
    return out | {"operator": operator_name, "cells": cells, "wall": wall, "ppc": ppc,
                  "t_wall": T_WALL if wall == "diffuse" else None, "A_max": static.A_max,
                  "a_max_regrown": regrown, "slots": static.N,
                  "alive_end": int(state.alive.sum()), "warm_blocks": n_warm,
                  "sample_blocks": n_sample, "dt": DT,
                  "seconds": time.time() - t0}  # fmt: skip


def _fields_on(grid, n, speed, temp, factor, nrho_inf):
    """``n / n_inf``, ``|u|``, ``T / T_inf`` restricted ``factor``-fold (fluid-volume and
    mass weighted)."""
    vol = np.asarray(cell_volume(grid))[: n_cartesian(grid)]
    vol = np.broadcast_to(vol, (n_cartesian(grid),)) if vol.ndim == 0 else vol
    n_c = restrict(n, vol, grid, factor) / nrho_inf
    mass = n[: vol.shape[0]] * vol
    return {"nrho": n_c, "speed": restrict(speed, mass, grid, factor),
            "temp": restrict(temp, mass, grid, factor) / F.TEMP}  # fmt: skip


def _summary(grid, cells, n, u, temp, nrho_inf, drag, heat):
    """Loads, fields at the comparison resolutions, the stagnation line, ``dx / lambda``."""
    speed = np.linalg.norm(u, axis=1)
    native = _fields_on(grid, n, speed, temp, 1, nrho_inf)
    out = {"drag": float(drag.mean()), "drag_se": float(drag.std(ddof=1) / math.sqrt(len(drag))),
           "heat": float(heat.mean()), "heat_se": float(heat.std(ddof=1) / math.sqrt(len(heat))),
           "cell": CELL_SIZE[cells]}  # fmt: skip
    factors = {"c8": 8, "c16": 16} if cells == "fine" else {cells: 1}
    for name, factor in factors.items():
        f = _fields_on(grid, n, speed, temp, factor, nrho_inf)
        out[f"fields_{name}"] = {k: v.astype(np.float32).tolist() for k, v in f.items()}
    ny = int(grid.n[1])
    rows = slice(ny // 2 - 1, ny // 2 + 1)  # the two rows either side of y = 5 m
    out["stagnation"] = {k: v[:, rows].mean(axis=1).tolist() for k, v in native.items()}
    out["x"] = (float(grid.lo[0]) + (np.arange(int(grid.n[0])) + 0.5) * CELL_SIZE[cells]).tolist()
    gas = sd.BY_SYMBOL["Ar"]
    n_abs = np.maximum(native["nrho"], 1e-30) * nrho_inf
    lam = em.mean_free_path(n_abs) * (np.maximum(native["temp"], 1e-3)) ** (gas.omega - 0.5)
    fluid = native["nrho"] > 0.05
    hist, edges = np.histogram(np.log2(CELL_SIZE[cells] / lam[fluid]),
                               bins=np.arange(-4.0, 7.5, 0.5))  # fmt: skip
    out["log2_dx_lambda_hist"], out["log2_dx_lambda_edges"] = hist.tolist(), edges.tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", default="c8", choices=tuple(CELLS))
    ap.add_argument("--wall", default="specular", choices=("specular", "diffuse"))
    ap.add_argument("--operator", default="ntc",
                    choices=(*operators.BASELINES, *operators.MODELS))  # fmt: skip
    ap.add_argument("--theta", default=None, help="a train_*.json (overrides --operator)")
    ap.add_argument("--ppc", type=int, default=20)
    ap.add_argument("--warm", type=float, default=3.0, help="flow-through times")
    ap.add_argument("--sample", type=float, default=1.0, help="flow-through times")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args(argv)
    theta, name, label = None, args.operator, args.operator
    if args.theta:
        theta, name = load_theta(args.theta)
        label = args.theta.removesuffix(".json")
    out = run(args.cells, args.wall, name, theta, args.ppc, args.warm, args.sample, args.seed)
    out["label"] = label
    path = RESULTS / f"circle_{label}_{args.cells}_{args.wall}_{args.tag}.json"
    path.write_text(json.dumps(out))
    print(
        f"wrote {path}: drag {out['drag']:.4e} +- {out['drag_se']:.1e} N/m, heat "
        f"{out['heat']:.4e} +- {out['heat_se']:.1e} W/m, {out['seconds']:.0f} s, "
        f"A_max {out['A_max']}"
    )


if __name__ == "__main__":
    main()
