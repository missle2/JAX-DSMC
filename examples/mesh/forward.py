"""Forward duplicates of SPARTA's ``circle`` (2-D) and ``sphere`` (3-D) examples in argon.

Same geometry (``read_sparta_surf``), box, grid, inlet (``emit/face xlo``: reservoir
``nrho 1 fnum 0.001`` at 273.15 K streaming at 100 m/s), timestep 1e-4 and 1000 steps from
an empty box as ``examples/mesh/sparta/in.{circle,sphere}_ar``; the surface is
``diffuse 300 K`` with accommodation ``--acc`` (0 = specular, SPARTA's shipped setting).
Per run one JSON of stats every ``--stats`` steps (``Np``, ``Nscoll``, the tracker
diagnostics), the window-averaged (steps ``n_steps - window + 1 .. n_steps``) per-element
loads in SPARTA's ``compute surf`` units, the window-averaged cell fields, snapshot PNGs of
the 2-D fields at ``--snapshots`` and the VTK files (``.vtr`` with ``volume_fraction``,
``.vtp`` surface with loads) for ParaView.

    JAX_PLATFORMS=cpu python3 examples/mesh/forward.py --case circle --seed 0
    python3 examples/mesh/forward.py --case sphere --tracker walk --body analytic
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

# XLA's CUDA-graph command buffers crash inside the WSL CUDA driver
# (`cuGraphExecMemcpyNodeSetParams`) on the polygon-mesh step with an inflow source
# (2026-09-17, jax 0.7 / RTX 3050): disable them for these runs.  The flag is a no-op on CPU.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import inflow, initialize, output, simulate, viz  # noqa: E402
from jax_dsmc import species_data as sd  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams  # noqa: E402
from jax_dsmc.collide import species_params  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import cell_volume, num_cells  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402
from jax_dsmc.mesh import surface as S  # noqa: E402
from jax_dsmc.mesh.types import AnalyticBody  # noqa: E402
from jax_dsmc.state import FlowParams, GridSpec, SimParams, SimStatic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPARTA = os.path.join(HERE, "sparta")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

CASES = {
    "circle": {
        "dim": 2,
        "lo": [0.0, 0.0],
        "hi": [10.0, 10.0],
        "n": (20, 20),
        "surf": "data.circle",
        "centre": [5.0, 5.0],
        "radius": 3.0,
        "fnum": 1.0e-3,
        "capacity": 60_000,
    },
    "plate": {
        # GP (plan section 4): a diffuse plate normal to the stream -- a stadium (capsule)
        # level set of half-width `half_width` and thickness `radius`; only reachable
        # through `examples.ensemble_mesh --body levelset --case plate --acc 1`
        "dim": 2,
        "lo": [0.0, 0.0],
        "hi": [10.0, 10.0],
        "n": (20, 20),
        "surf": "data.circle",
        "centre": [5.0, 5.0],
        "radius": 0.4,
        "half_width": 1.0,
        "kind": "capsule",
        "fnum": 1.0e-3,
        "capacity": 60_000,
    },
    "sphere": {
        "dim": 3,
        "lo": [-2.0, -2.0, -2.0],
        "hi": [2.0, 2.0, 2.0],
        "n": (20, 20, 20),
        "surf": "data.sphere",
        "centre": [0.0, 0.0, 0.0],
        "radius": 1.0,
        "fnum": 5.0e-4,
        "capacity": 100_000,
    },
}
CASES_FNUM = {c["n"]: c["fnum"] for c in CASES.values()}
NRHO = 1.0
TEMP = 273.15
VSTREAM = 100.0
T_SURF = 300.0
DT = 1.0e-4
# event passes per step.  A particle still unresolved after them -- repeated reflections
# in a concave pocket of an optimised shape -- is deleted and counted (`surface_stuck`,
# SPARTA MAXSTUCK).  Measured 2026-09-23 on the roughest optc shape (S1 gd iterate 8): 4
# passes delete 1 particle in the last 300 steps (~1e-7 of particle-steps); 8 delete none
# but cost +66 % forward time (every pass runs the hit search on all particles), so 4
K_BOUNCE = 4
BLOCK = 10  # SPARTA `stats 10`


def build(
    case: str,
    body: str,
    tracker: str,
    gas: str,
    acc: float,
    radius_cells: int,
    body_radius: float | None = None,
):
    """Everything a run needs; ``body_radius`` overrides the analytic body's radius."""
    c = CASES[case]
    D = c["dim"]
    grid = GridSpec(lo=jnp.array(c["lo"]), hi=jnp.array(c["hi"]), n=c["n"])
    if body == "polygon":
        surf = S.read_sparta_surf(os.path.join(SPARTA, c["surf"]), D)
    else:
        r = c["radius"] if body_radius is None else body_radius
        surf = AnalyticBody(centre=jnp.array(c["centre"]), radius=jnp.float64(r))
    mesh = cut.build_mesh(surf, grid, radius=radius_cells)
    grid = cut.with_volume(grid, mesh)
    species = sd.BY_SYMBOL[gas]
    sp = species_params(
        jnp.array([species.mass]),
        jnp.array([[species.diam]]),
        jnp.array([[species.omega]]),
        jnp.array([[species.tref]]),
        jnp.array([[species.alpha]]),
    )
    C = num_cells(grid)
    params = SimParams(dt=jnp.float64(DT), fnum=jnp.full((C,), c["fnum"]), epsilon=jnp.float64(0.0))
    static = SimStatic(
        N=c["capacity"],
        D=D,
        S=1,
        G=1,
        A_max=4,
        K_bounce=K_BOUNCE,
        n_steps=BLOCK,
        tracker=tracker,
        K_walk=8,
    )
    faces = Faces((BC.INFLOW, BC.OUTFLOW) + (BC.SPECULAR,) * (2 * D - 2))
    res = inflow.InflowParams.uniform(D, NRHO, TEMP, [VSTREAM, 0.0, 0.0])
    wall = WallParams.uniform(D, T_SURF, acc=0.0, n_models=1).with_model(D, 0, temp=T_SURF, acc=acc)
    target = float(inflow.target_count(res, grid, 0, 0, sp.mass[0], params.fnum[0], params.dt))
    capacity = int(2 * target + 64)
    return grid, mesh, sp, params, static, faces, res, wall, target, capacity


def surface_loads(momentum, energy, hits, mesh, fnum, window_steps, dt):
    """SPARTA ``compute surf`` values from the window's extensive tallies (elements only).

    ``press``: normal force per unit area, positive when compressive; ``sh``: tangential
    force per unit area (vector); ``f``: total force on the element (vector); ``ke``:
    energy flux to the surface per unit area.  All from ``fnum * sum m (v_in - v_out)``
    over the window divided by ``area * window * dt`` (``compute_surf.cpp:263-648``).
    """
    D = mesh.normals.shape[1]
    n = np.asarray(mesh.normals)
    area = np.asarray(mesh.areas)
    dim = 2 * D
    p = np.asarray(momentum)[dim:, :D]
    factor = fnum / (window_steps * dt)
    force = p * factor
    pn = np.sum(force * n, axis=1)
    return {
        "hits": np.asarray(hits)[dim:].astype(int),
        "press": -pn / area,
        "shear": (force - pn[:, None] * n) / area[:, None],
        "force": force,
        "ke": np.asarray(energy)[dim:] * factor / area,
    }


def run(args) -> dict:  # noqa: PLR0915 -- one run, kept whole so the bookkeeping is readable
    case = args.case
    grid, mesh, sp, params, static, faces, res, wall, target, capacity = build(
        case, args.body, args.tracker, args.gas, args.acc, args.radius
    )
    D = static.D
    key = jax.random.PRNGKey(args.seed)
    flow = FlowParams(
        nrho=jnp.float64(NRHO), temp=jnp.float64(TEMP), vstream=jnp.array([VSTREAM, 0.0, 0.0])
    )
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    step = jax.jit(
        lambda s, c, k0, moments: simulate.run(
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
            capacity=capacity,
            emit_moments=moments,
            step0=k0,
            mesh=mesh,
            emit_surface=True,
        ),
        static_argnums=(3,),
    )
    n_steps = args.steps
    window = args.window
    first_window = n_steps - window
    E = mesh.n_elements
    tally_sum = {
        "momentum": np.zeros((2 * D + E, 3)),
        "energy": np.zeros(2 * D + E),
        "hits": np.zeros(2 * D + E),
    }
    field_sum = {
        "nrho": np.zeros(num_cells(grid)),
        "u": np.zeros((num_cells(grid), 3)),
        "temp": np.zeros(num_cells(grid)),
        "temp_kinetic": np.zeros(num_cells(grid)),
        "count": np.zeros(num_cells(grid)),
    }
    mass = float(sp.mass[0])
    field_n = 0
    stats = []
    snapshots = {}
    overlay = _overlay(mesh)
    mask = cut.flow_mask(mesh, grid, sub=8 if D == 2 else 4)
    series = _Series(tag_of(args), grid, mesh, mask, args.vtk_every) if args.vtk_every else None
    t0 = time.time()
    diag_totals = {}
    for k0 in range(0, n_steps, BLOCK):
        in_window = k0 >= first_window
        want_snap = any(k0 < s <= k0 + BLOCK for s in args.snapshots)
        want_frame = series is not None and any(
            (k0 + j) % args.vtk_every == 0 for j in range(1, BLOCK + 1)
        )
        (state, cstate), (out, diag, tallies) = step(
            state, cstate, k0, in_window or want_snap or want_frame
        )
        if series is not None:
            series.add_block(k0, out, tallies, mass)
        d = {f: np.asarray(getattr(diag, f)) for f in _DIAG}
        for f, v in d.items():
            diag_totals[f] = diag_totals.get(f, 0.0) + float(v.sum())
        n_alive = int(jnp.sum(state.alive))
        stats.append(
            {
                "step": k0 + BLOCK,
                "np": n_alive,
                "nscoll": int(d["surface_hits"].sum()),
                "nscoll_step": int(d["surface_hits"][-1]),
                "injected": int(d["inflow_injected"].sum()),
                "shortfall": float(d["inflow_shortfall"].sum()),
                "stencil_overrun": int(d["stencil_overrun"].sum()),
                "walk_exhausted": int(d["walk_exhausted"].sum()),
                "inside_body": int(d["inside_body"].sum()),
                "recrossed": int(d["wall_recrossed"].sum()),
                "attempts_clipped": int(d["attempts_clipped"].sum()),
                "ncoll": int(np.asarray(getattr(diag, "collisions", np.zeros(1))).sum())
                if hasattr(diag, "collisions")
                else -1,
                "seconds": time.time() - t0,
            }
        )
        if in_window:
            tally_sum["momentum"] += np.asarray(tallies.momentum).sum(axis=0)
            tally_sum["energy"] += np.asarray(tallies.energy).sum(axis=0)
            tally_sum["hits"] += np.asarray(tallies.hits).sum(axis=0)
            # SPARTA's `fix ave/grid` averages over particle-steps (sum of m v over every
            # particle in every step of the window, divided by the summed count), so `u`
            # and the temperatures are count-weighted here; a plain mean of per-step cell
            # means would count an empty cut cell's zero as a sample.  `temp_kinetic` is
            # SPARTA's `compute grid temp`, the un-centred (m / 3k) <|v|^2>.
            cnt = np.asarray(out.count, dtype=float)
            u_step = np.asarray(out.u)
            t_step = np.asarray(out.temp)
            field_sum["nrho"] += np.asarray(out.nrho).sum(axis=0)
            field_sum["count"] += cnt.sum(axis=0)
            field_sum["u"] += (cnt[:, :, None] * u_step).sum(axis=0)
            field_sum["temp"] += (cnt * t_step).sum(axis=0)
            field_sum["temp_kinetic"] += (
                cnt * (t_step + mass * np.sum(u_step**2, axis=2) / (3.0 * K_BOLTZ))
            ).sum(axis=0)
            field_n += BLOCK
        for s in args.snapshots:
            if k0 < s <= k0 + BLOCK:
                i = s - k0 - 1
                snap = {
                    "nrho": np.asarray(out.nrho[i]),
                    "u": np.asarray(out.u[i]),
                    "temp": np.asarray(out.temp[i]),
                }
                snapshots[s] = snap
        if (k0 + BLOCK) % 100 == 0:
            r = stats[-1]
            print(
                f"{case} step {r['step']:5d}  Np {r['np']:6d}  Nscoll/step {r['nscoll'] / BLOCK:6.1f}  "
                f"overrun {r['stencil_overrun']} walk_exh {r['walk_exhausted']} inside {r['inside_body']}  "
                f"{r['seconds']:.1f} s",
                flush=True,
            )
    cnt = np.maximum(field_sum["count"], 1.0)
    fields = {
        "nrho": field_sum["nrho"] / max(field_n, 1),
        "count": field_sum["count"] / max(field_n, 1),
        "u": field_sum["u"] / cnt[:, None],
        "temp": field_sum["temp"] / cnt,
        "temp_kinetic": field_sum["temp_kinetic"] / cnt,
    }
    loads = surface_loads(
        tally_sum["momentum"],
        tally_sum["energy"],
        tally_sum["hits"],
        mesh,
        CASES[case]["fnum"],
        window,
        DT,
    )
    total_force = loads["force"].sum(axis=0)
    tag = tag_of(args)
    result = {
        "case": case,
        "body": args.body,
        "tracker": args.tracker,
        "gas": args.gas,
        "acc": args.acc,
        "seed": args.seed,
        "steps": n_steps,
        "window": window,
        "dt": DT,
        "fnum": CASES[case]["fnum"],
        "capacity": static.N,
        "inlet_capacity": capacity,
        "target_count": target,
        "stencil_radius": args.radius,
        "backend": jax.default_backend(),
        "seconds": time.time() - t0,
        "cells": [int((np.asarray(mesh.cell_type) == k).sum()) for k in (0, 1, 2)],
        "flow_volume": float(mesh.volume.sum()),
        "diagnostics_total": diag_totals,
        "stats": stats,
        "total_force": total_force.tolist(),
        "total_hits_window": int(loads["hits"].sum()),
        "loads": {k: np.asarray(v).tolist() for k, v in loads.items()},
        "fields": {k: np.asarray(v).tolist() for k, v in fields.items()},
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"forward_{tag}.json")
    with open(path, "w") as fh:
        json.dump(result, fh)
    print(
        f"wrote {path}: Np {stats[-1]['np']}, mean Nscoll/step over window "
        f"{loads['hits'].sum() / window:.1f}, total force {total_force}, {result['seconds']:.0f} s"
    )
    _write_vtk(tag, grid, mesh, mask, fields, loads)
    if series is not None:
        series.close()
    if snapshots:
        _write_snapshots(tag, grid, snapshots, overlay, mask, fields, D)
    return result


def tag_of(args) -> str:
    return f"{args.case}_{args.body}_{args.tracker}_{args.gas}_acc{args.acc:g}_{args.seed}"


class _Series:
    """A ParaView time series every ``every`` steps: fields ``.vtr`` (per cell) and the
    clipped ``.vtu`` (flow region only), the surface ``.vtp`` with the loads accumulated
    over the ``every`` steps before the frame; ``fields.pvd`` / ``flow.pvd`` /
    ``surface.pvd`` bind them to the simulation time."""

    def __init__(self, tag, grid, mesh, mask, every):
        self.dir = os.path.join(RESULTS, "vtk", f"{tag}_series")
        os.makedirs(self.dir, exist_ok=True)
        self.grid, self.mesh, self.mask, self.every = grid, mesh, mask, every
        full = float(cell_volume(dataclasses.replace(grid, volume=jnp.zeros(0)))[0])
        self.static = {
            "volume_fraction": np.asarray(mesh.volume) / full,
            "cell_type": np.asarray(mesh.cell_type),
        }
        self.entries = {"fields": [], "flow": [], "surface": []}
        D = mesh.normals.shape[1]
        E = mesh.n_elements
        self.acc = {
            "momentum": np.zeros((2 * D + E, 3)),
            "energy": np.zeros(2 * D + E),
            "hits": np.zeros(2 * D + E),
        }
        self.acc_steps = 0

    def add_block(self, k0, out, tallies, mass):
        mom = np.asarray(tallies.momentum)
        en = np.asarray(tallies.energy)
        hits = np.asarray(tallies.hits)
        for j in range(BLOCK):
            step = k0 + j + 1
            self.acc["momentum"] += mom[j]
            self.acc["energy"] += en[j]
            self.acc["hits"] += hits[j]
            self.acc_steps += 1
            if step % self.every:
                continue
            fields = {
                "nrho": np.asarray(out.nrho[j]),
                "u": np.asarray(out.u[j]),
                "temp": np.asarray(out.temp[j]),
                "count": np.asarray(out.count[j]),
                **self.static,
            }
            fields["temp_kinetic"] = fields["temp"] + mass * np.sum(fields["u"] ** 2, axis=1) / (
                3.0 * K_BOLTZ
            )
            t = step * DT
            p = output.write_vtr(
                os.path.join(self.dir, f"fields_{step:05d}"),
                self.grid.lo,
                self.grid.hi,
                self.grid.n,
                fields,
            )
            self.entries["fields"].append((t, p))
            p = output.write_vtu_flow(
                os.path.join(self.dir, f"flow_{step:05d}"),
                self.grid.lo,
                self.grid.hi,
                self.grid.n,
                self.mask,
                fields,
            )
            self.entries["flow"].append((t, p))
            if self.mesh.kind == "surface":
                loads = surface_loads(
                    self.acc["momentum"],
                    self.acc["energy"],
                    self.acc["hits"],
                    self.mesh,
                    CASES_FNUM[self.grid.n],
                    self.acc_steps,
                    DT,
                )
                p = output.write_vtp_surface(
                    os.path.join(self.dir, f"surface_{step:05d}"),
                    self.mesh.body.points,
                    self.mesh.body.elements,
                    {k: np.asarray(v) for k, v in loads.items()},
                )
                self.entries["surface"].append((t, p))
            for k in self.acc:
                self.acc[k][...] = 0.0
            self.acc_steps = 0

    def close(self):
        for name, entries in self.entries.items():
            if entries:
                output.write_pvd(os.path.join(self.dir, f"{name}.pvd"), entries)
        print(f"wrote {len(self.entries['fields'])} frames to {self.dir}")


_DIAG = (
    "surface_hits",
    "inflow_injected",
    "inflow_shortfall",
    "stencil_overrun",
    "walk_exhausted",
    "inside_body",
    "wall_recrossed",
    "attempts_clipped",
    "inflow_tracking_exhausted",
)


def _overlay(mesh):
    if mesh.kind == "surface":
        return (np.asarray(mesh.body.points), np.asarray(mesh.body.elements))
    return (np.asarray(mesh.body.centre)[None, :], float(mesh.body.radius))


def _write_vtk(tag, grid, mesh, mask, fields, loads):
    """Window averages: ``.vtr`` (every cell, honest per-cell values), ``.vtu`` (the same
    values on the flow region only, the body carved out at sub-cell resolution) and the
    surface ``.vtp`` with the loads."""
    vtk_dir = os.path.join(RESULTS, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    full = float(cell_volume(dataclasses.replace(grid, volume=jnp.zeros(0)))[0])
    cell_fields = dict(fields)
    cell_fields["volume_fraction"] = np.asarray(mesh.volume) / full
    cell_fields["cell_type"] = np.asarray(mesh.cell_type)
    output.write_vtr(os.path.join(vtk_dir, f"{tag}_fields"), grid.lo, grid.hi, grid.n, cell_fields)
    output.write_vtu_flow(
        os.path.join(vtk_dir, f"{tag}_flow"), grid.lo, grid.hi, grid.n, mask, cell_fields
    )
    if mesh.kind == "surface":
        output.write_vtp_surface(
            os.path.join(vtk_dir, f"{tag}_surface"),
            mesh.body.points,
            mesh.body.elements,
            {k: np.asarray(v) for k, v in loads.items()},
        )


def _write_snapshots(tag, grid, snapshots, overlay, mask, steady, D):
    """Field PNGs with the body carved out (``mask``) and the surface overlaid; 3-D runs
    are shown on the z = 0 plane."""
    os.makedirs(FIGURES, exist_ok=True)
    vmax_n = float(np.percentile(steady["nrho"], 99)) * 1.1
    limits = {"nrho": (0.0, vmax_n), "|u|": (0.0, 400.0), "temp": (0.0, 400.0)}
    plane = viz.Plane(2, fraction=0.5) if D == 3 else None
    for step, snap in sorted(snapshots.items()):
        speed = np.sqrt(np.sum(snap["u"] ** 2, axis=1))
        viz.field_snapshot(
            os.path.join(FIGURES, f"{tag}_t{step:04d}.png"),
            {"nrho": snap["nrho"], "|u|": speed, "temp": snap["temp"]},
            grid.n,
            grid.lo,
            grid.hi,
            title=f"{tag}  step {step}",
            limits=limits,
            surface=overlay,
            mask=mask,
            plane=plane,
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), default="circle")
    ap.add_argument("--body", choices=("polygon", "analytic"), default="polygon")
    ap.add_argument("--tracker", choices=("stencil", "walk"), default="stencil")
    ap.add_argument("--gas", default="Ar", choices=sorted(sd.BY_SYMBOL))
    ap.add_argument("--acc", type=float, default=0.0, help="surface accommodation (0 = specular)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--window", type=int, default=300, help="averaging window (last steps)")
    ap.add_argument("--radius", type=int, default=1, help="stencil dilation in cells")
    ap.add_argument("--snapshots", type=int, nargs="*", default=[100, 300, 1000])
    ap.add_argument(
        "--vtk-every",
        type=int,
        default=0,
        help="write a ParaView time series (.vtr, clipped .vtu, surface .vtp) every N steps",
    )
    args = ap.parse_args(argv)
    if args.steps % BLOCK or args.window % BLOCK:
        raise SystemExit(f"--steps and --window must be multiples of {BLOCK}")
    if args.vtk_every and BLOCK % args.vtk_every:
        raise SystemExit(f"--vtk-every must divide {BLOCK}")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
