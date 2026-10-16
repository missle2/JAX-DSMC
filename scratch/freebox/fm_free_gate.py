"""Gate for ``forward.build(boundary="free")`` (2026-10-15): free-molecular drag of the specular
circle in the 10 m box with every face a free-stream face, against the full-Maxwellian closed
form (``free_molecular.reference_drag``; Cd 12.22 at the deck's M 0.32).  The channel deck
(inflow / outflow / specular sides) holds a half-range stream and matched the half-range
reference instead (17.17; free_molecular_circle.json: 17.69 / 17.33 / 17.35 at blockage
0.3 / 0.15 / 0.075).  Free space has no blockage, so the two radii should agree.

    run_locked.sh gpu env PYTHONPATH=. python3 scratch/freebox/fm_free_gate.py [--radii 1.5 0.75]
"""

import argparse
import dataclasses
import json
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "examples/mesh")
import free_molecular as FM  # noqa: E402
from forward import BLOCK, CASES, DT, NRHO, TEMP, VSTREAM, build  # noqa: E402

from jax_dsmc import initialize, simulate  # noqa: E402

from jax_dsmc.state import FlowParams, ParticleState  # noqa: E402


def full_start(key, grid, sp, static, mesh):
    """The drifting Maxwellian outside the body at t = 0 (FM._halfrange_start without the cut)."""
    D = static.D
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n_full = round(NRHO * float(np.prod(hi - lo)) / CASES["circle"]["fnum"])
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    sigma = math.sqrt(FM.K_BOLTZ * TEMP / float(sp.mass[0]))
    v = rng.normal(size=(n_full, 3)) * sigma
    v[:, 0] += VSTREAM
    x = lo + rng.random((n_full, D)) * (hi - lo)
    c = np.asarray(mesh.body.centre)
    keep = np.sum((x - c) ** 2, axis=1) > float(mesh.body.radius) ** 2
    x, v = x[keep], v[keep]
    n = x.shape[0]
    alive = np.zeros(static.N, bool)
    alive[:n] = True
    xs = np.broadcast_to(lo, (static.N, D)).copy()
    xs[:n] = x
    vs = np.zeros((static.N, 3))
    vs[:n] = v
    return ParticleState(
        x=jnp.asarray(xs), v=jnp.asarray(vs), species=jnp.zeros(static.N, jnp.int32),
        alive=jnp.asarray(alive), logh_acc=jnp.zeros(static.N),
    )  # fmt: skip


def run(radius, boundary, steps, window, seed):
    grid, mesh, sp, params, static, faces, res, wall, _t, capacity = build(
        "circle", "analytic", "stencil", "Ar", 0.0, 1, body_radius=radius, boundary=boundary
    )
    volume = float(np.prod(np.asarray(grid.hi) - np.asarray(grid.lo)))
    static = dataclasses.replace(static, N=int(1.4 * NRHO * volume / CASES["circle"]["fnum"]))
    key = jax.random.PRNGKey(seed)
    flow = FlowParams(
        nrho=jnp.float64(NRHO), temp=jnp.float64(TEMP), vstream=jnp.array([VSTREAM, 0.0, 0.0])
    )
    state = full_start(key, grid, sp, static, mesh)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    step = jax.jit(
        lambda s, c, k0: simulate.run(
            s, c, grid, sp, params, static, key, faces=faces, wall=wall, res=res,
            capacity=capacity, emit_moments=False, step0=k0, mesh=mesh, emit_surface=True,
        )
    )  # fmt: skip
    D = static.D
    fx, bad, t0 = [], 0, time.time()
    alive_trace = []
    for k0 in range(0, steps, BLOCK):
        (state, cstate), (_, diag, tallies) = step(state, cstate, k0)
        bad += int(
            diag.inflow_shortfall.sum() + diag.stencil_overrun.sum() + diag.inside_body.sum()
        )
        alive_trace.append(int(jnp.sum(state.alive)))
        if k0 >= steps - window:
            fx.append(float(np.asarray(tallies.momentum)[:, 2 * D, 0].sum()))
    blocks = np.asarray(fx) * CASES["circle"]["fnum"] / (BLOCK * DT)
    means = np.asarray([b.mean() for b in np.array_split(blocks, 10)])
    force, se = float(blocks.mean()), float(means.std(ddof=1) / math.sqrt(len(means)))
    mass = float(sp.mass[0])
    cd = FM.drag_coefficient(force, D, radius, NRHO, mass, VSTREAM)
    cd_se = cd * se / force
    ref = FM.reference_drag(D, radius, NRHO, mass, TEMP, VSTREAM, TEMP, False)
    cd_ref = FM.drag_coefficient(ref, D, radius, NRHO, mass, VSTREAM)
    return {
        "radius": radius, "boundary": boundary, "cd": cd, "cd_se": cd_se,
        "cd_reference_full_maxwellian": cd_ref, "z": (cd - cd_ref) / cd_se,
        "alive_start_end": [alive_trace[0], alive_trace[-1]], "bad": bad,
        "seconds": time.time() - t0,
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.5, 0.75])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--window", type=int, default=3000)
    args = ap.parse_args()
    rows = []
    for r in args.radii:
        row = run(r, "free", args.steps, args.window, 0)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open("scratch/freebox/fm_free_gate.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
