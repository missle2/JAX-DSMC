"""The round-1 generalisation test (plan 2026-11-11): a Mach 10 argon circle at
``Kn_D = lambda / D = 0.042`` (``lambda = 0.25 m``, ``D = 6 m``), forward only.

The coarse run is the ``circle_xlong`` box on its own 0.5 m cells (``dx = 2 lambda``, the
training's 8x); the fine reference the same box on 8x finer cells (``grid_scale 8``,
``dx = lambda / 4``), both at ``--ppc`` particles per free-stream cell and the fine run's time
step.  Free-stream faces all round (the body in free space) and the decks' specular wall: a
cold *diffuse* wall at M 10 would put a stagnation layer ~100x the free-stream density
(``lambda_w ~ 2 mm``) against the body, unresolved even by the fine run.

Every cell runs the chosen operator -- next to the wall too: there are no multi-level cells
yet, so this is the plain "how wrong is it" test, with the near-body ``dx / lambda_local``
outside the training range (reported as a histogram).

Measured over ``--sample`` flow-through times after ``--warm``: the drag per unit span
(block means, standard error over blocks), the cell fields restricted to the 0.5 m cells
(``nrho``, ``|u|``, ``T``), the stagnation-line profiles and the shock stand-off (where
the stagnation line's density first exceeds twice the free stream's).

    PYTHONPATH=. python3 -m examples.learned_collision.circle --scale 8          # fine NTC
    PYTHONPATH=. python3 -m examples.learned_collision.circle --operator esfp_recon
    PYTHONPATH=. python3 -m examples.learned_collision.circle --theta train_linear_v1.json

Writes ``examples/learned_collision/results/circle_<label>_s<scale>_<tag>.json``.
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
from jax_dsmc.collide import relax
from jax_dsmc.grid import cell_volume, n_cartesian, num_cells
from jax_dsmc.state import FlowParams

CASE = "circle_xlong"
LAMBDA = 0.25
MACH = 10.0
DT_COARSE_CELL = 2.5e-5
"""The decks' M 10 step on 0.5 m cells (a free-stream move of 0.15 cell); the runs here use
it divided by the finest scale, 8 (the fine run's)."""
FINE = 8
BLOCK = 50


def config(scale):
    """The :mod:`examples.ensemble_mesh` configuration of a run on ``scale``-fold cells."""
    cfg = em.default_config()
    cfg.update(case=CASE, knudsen=LAMBDA / (2.0 * F.CASES[CASE]["radius"]), mach=MACH,
               boundary="free", acc=0.0, grid_scale=scale, fnum_scale=1.0,
               dt=DT_COARSE_CELL / FINE, steps=BLOCK, warmup=0)  # fmt: skip
    return cfg


def setup(scale, ppc, a_max=32):
    """``(grid, mesh, sp, params, static, faces, res, wall, capacity, nrho)``."""
    cfg = config(scale)
    nrho = F.NRHO * em.knudsen_density(cfg)
    grid, mesh, sp, params, static, faces, res, wall, _ = em.build(cfg, "radius", 3.0, BLOCK)
    area = float(np.prod(np.asarray(grid.hi - grid.lo))) / n_cartesian(grid)
    fnum = nrho * area / ppc
    params = dataclasses.replace(params, fnum=jnp.full((num_cells(grid),), fnum))
    slots = int(2.0 * ppc * n_cartesian(grid)) + 4096
    static = dataclasses.replace(static, N=slots, A_max=a_max)
    capacity = F.injection_bound(res, grid, faces, sp.mass[0], fnum, params.dt) + 64
    return grid, mesh, sp, params, static, faces, res, wall, capacity, nrho


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


def restrict(field, weight, grid, scale):
    """A fine field on the Cartesian cells, averaged (``weight``-weighted) onto the
    ``scale``-fold coarser cells: ``f64[nx / scale, ny / scale]``."""
    nx, ny = (int(k) for k in grid.n)
    f = np.asarray(field)[: nx * ny].reshape(nx, ny)
    w = np.asarray(weight)[: nx * ny].reshape(nx, ny)
    shape = (nx // scale, scale, ny // scale, scale)
    num = (f * w).reshape(shape).sum(axis=(1, 3))
    den = w.reshape(shape).sum(axis=(1, 3))
    return num / np.maximum(den, 1e-300)


def run(scale, operator_name, theta, ppc, warm, sample_ft, seed=0):
    """One circle run (module docstring)."""
    t0 = time.time()
    grid, mesh, sp, params, static, faces, res, wall, capacity, nrho = setup(scale, ppc)
    speed = em.stream_speed(config(scale))
    fn, theta0, pairing = operators.operator(operator_name, "circle")
    theta = theta0 if theta is None else theta
    static = dataclasses.replace(static, pairing=pairing)
    key = jax.random.PRNGKey(seed)
    flow = FlowParams(nrho=jnp.float64(nrho), temp=jnp.float64(F.TEMP),
                      vstream=jnp.array([speed, 0.0, 0.0]))  # fmt: skip
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    dt = float(params.dt)
    flow_through = float(grid.hi[0] - grid.lo[0]) / speed
    n_warm = math.ceil(warm * flow_through / (dt * BLOCK))
    n_sample = math.ceil(sample_ft * flow_through / (dt * BLOCK))
    fnum = float(params.fnum[0])
    n_cells = num_cells(grid)
    acc = {"nrho": np.zeros(n_cells), "mom": np.zeros((n_cells, 3)), "temp_n": np.zeros(n_cells)}
    drag, regrown = [], 0
    for b in range(n_warm + n_sample):
        while True:
            (new, new_c), aux = _run(state, cstate, grid, sp, params, key, faces, wall, res,
                                     mesh, static, capacity, b * BLOCK, theta, fn)  # fmt: skip
            diag, tallies = aux[1], aux[-1]
            if not int(jnp.sum(diag.attempts_clipped)):
                break
            peak = int(jnp.max(diag.max_attempts))
            static = dataclasses.replace(static, A_max=max(static.A_max + 1, math.ceil(1.4 * peak)))
            regrown += 1
        if float(jnp.max(diag.inflow_shortfall)) > 0:
            raise RuntimeError(f"inflow shortfall at block {b}: raise the slots")
        state, cstate = new, new_c
        if b >= n_warm:
            mom = float(jnp.sum(tallies.momentum[:, 2 * static.D :, 0]))
            drag.append(mom * fnum / (BLOCK * dt))
            n, u, temp = (np.asarray(a) for a in _fields(state, grid, params, sp))
            acc["nrho"] += n
            acc["mom"] += n[:, None] * u
            acc["temp_n"] += n * temp
    n = acc["nrho"] / n_sample
    u = acc["mom"] / np.maximum(acc["nrho"], 1e-300)[:, None]
    temp = acc["temp_n"] / np.maximum(acc["nrho"], 1e-300)
    return _summary(grid, scale, n, u, temp, nrho, np.array(drag), sp) | {
        "operator": operator_name, "scale": scale, "ppc": ppc, "A_max": static.A_max,
        "a_max_regrown": regrown, "slots": static.N, "alive_end": int(state.alive.sum()),
        "warm_blocks": n_warm, "sample_blocks": n_sample, "dt": dt,
        "seconds": time.time() - t0}  # fmt: skip


def _summary(grid, scale, n, u, temp, nrho_inf, drag, sp):
    """Drag, the fields on the 0.5 m cells, the stagnation line and ``dx / lambda_local``."""
    coarse = scale  # restrict onto the 0.5 m cells
    n_c = restrict(n, np.ones_like(n), grid, coarse)
    speed = np.linalg.norm(u, axis=1)
    u_c = restrict(speed, n, grid, coarse)
    t_c = restrict(temp, n, grid, coarse)
    gas = sd.BY_SYMBOL["Ar"]
    lam_local = em.mean_free_path(np.maximum(n_c, 1e-30 * nrho_inf)) * (np.maximum(t_c, 1.0) / F.TEMP) ** (
        gas.omega - 0.5)  # fmt: skip
    dx_c = float(grid.hi[0] - grid.lo[0]) / (int(grid.n[0]) // coarse)
    ratio = dx_c / lam_local
    fluid = n_c > 0.05 * nrho_inf
    row = n_c.shape[1] // 2
    line = n_c[:, row] / nrho_inf
    x = float(grid.lo[0]) + (np.arange(n_c.shape[0]) + 0.5) * dx_c
    front = x < F.CASES[CASE]["centre"][0] - F.CASES[CASE]["radius"]
    jump = np.nonzero(front & (line > 2.0))[0]
    standoff = float(F.CASES[CASE]["centre"][0] - F.CASES[CASE]["radius"] - x[jump[0]]) if len(
        jump) else float("nan")  # fmt: skip
    blocks = len(drag)
    del sp
    hist, edges = np.histogram(np.log2(ratio[fluid]), bins=np.arange(-2.0, 7.5, 0.5))
    return {
        "drag": float(drag.mean()), "drag_se": float(drag.std(ddof=1) / math.sqrt(blocks)),
        "standoff": standoff, "stagnation_n": line.tolist(),
        "stagnation_T": (t_c[:, row] / F.TEMP).tolist(), "x": x.tolist(),
        "nrho": (n_c / nrho_inf).astype(np.float32).tolist(),
        "speed": u_c.astype(np.float32).tolist(), "temp": (t_c / F.TEMP).astype(np.float32).tolist(),
        "log2_dx_lambda_hist": hist.tolist(), "log2_dx_lambda_edges": edges.tolist(),
        "cut_cell_volume_min": float(np.min(np.asarray(cell_volume(grid)))),
    }  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scale", type=int, default=1, choices=(1, FINE), help="1 coarse, 8 fine")
    ap.add_argument("--operator", default="ntc",
                    choices=(*operators.BASELINES, *operators.MODELS))  # fmt: skip
    ap.add_argument("--theta", default=None, help="a train_*.json (overrides --operator)")
    ap.add_argument("--ppc", type=int, default=20)
    ap.add_argument("--warm", type=float, default=3.0, help="flow-through times")
    ap.add_argument("--sample", type=float, default=1.0, help="flow-through times")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    theta, name, label = None, args.operator, args.operator
    if args.theta:
        theta, name = load_theta(args.theta)
        label = args.theta.removesuffix(".json")
    out = run(args.scale, name, theta, args.ppc, args.warm, args.sample, args.seed)
    out["label"] = label
    out["relax_bound"] = relax.CORRECTION_BOUND
    path = RESULTS / f"circle_{label}_s{args.scale}_{args.tag}.json"
    path.write_text(json.dumps(out))
    print(
        f"wrote {path}: drag {out['drag']:.4e} +- {out['drag_se']:.1e} N/m, stand-off "
        f"{out['standoff']:.2f} m, {out['seconds']:.0f} s, A_max {out['A_max']}"
    )


if __name__ == "__main__":
    main()
