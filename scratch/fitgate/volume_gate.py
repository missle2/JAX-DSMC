"""The collision pull-back in situ on the fitgate gates (2026-11-01): scale the NTC flow volume
of a set of collision cells by (1 + eps) -- nothing else changes -- and compare the CRN central
difference of the windowed objective in eps with the backward's prediction

    dJ/deps = sum_{c in set} g_c V_c,   g_c = shape.volume_weights (the cells' collision
                                        increments C_kc of the virtual pull-back, -sum_k C_kc / V_c)

(the squash version: scratch/bias/volume_scale.py, 2026-10-02).  Every NTC probability of a cell
scales as 1 / V_c, so this is the adjoint's sensitivity to the collision rates themselves: the
weights of the cut-volume term (the M 10 bias's suspects, with the slope-fit terms fed by the
same pull-back), free of the geometry, the fits of grad lambda and the wall terms.  Sets:
``all`` (every collision cell: the whole shock layer) and ``cut`` (the body's cut cells: what the
cut-volume term weighs); two eps per set (linearity: the FD's own curvature).

    PYTHONPATH=. python3 scratch/fitgate/volume_gate.py --gate c10_spec --seeds 0 4

``--field NAME --reuse-fd`` (2026-11-01, after the diffuse circle read pred / FD ~1.4): the
backward's collision field fit swapped (:data:`FIELDS`) and only the prediction recomputed,
the FD read from the base run's rows (the scaled forwards do not depend on the backward).
One backward per seed; rows to ``volume_<gate>_<field>_s*.jsonl``.  A gate in
``variant_gate.FD_FROM`` (``c10_diff_h600`` / ``_h1000``: the same forward, a longer captured
horizon) reads the FD of the gate it shares the forward with; ``fd_objective`` is that row's J.

``--virtual S P`` (2026-11-01, after every field read the diffuse circle's 1.48): the virtual
pull-back's substeps x partners (default 4 x 8; its linearisation error is O(rate dt / S),
validated to rate dt 0.16).  Each row also carries the collision rate per particle per step
of every collision cell (accepted collisions x 2 / occupancy, over the captured steps) binned
in :data:`RATE_EDGES`, with the prediction summed per bin -- where the substeps move it --
and ``ratemap_<gate>[_<field>][_v<S>x<P>]_s<seed>.npz`` (rate, g_c V_c, cell centres).  Rows to
``volume_<gate>[_<field>]_v<S>x<P>_s*.jsonl``.  (The sequential pull-back has no
per-cell increments, so it cannot be gated here.)

``--sets cut near2`` (2026-11-03, the specular split: the bias sits in the collisional part
of grad_x lambda at the wall, the whole-box volume gate passes 1.01): the sets scaled, among
``all``, ``cut`` and ``near<K>`` -- the cut cells and every collision cell within ``K`` lattice
layers of one (max-norm over the cell centres, the cut cells' split sub-cells included): the
collision cells whose rates the wall's L* lambda reads, a bigger signal than the cut cells
alone.  A non-default set list tags the rows (``volume_<gate>_<sets>_s*.jsonl``).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, grid as grid_mod, shape  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path("examples/mesh/results/fitgate")
EPS = (0.2, 0.1)
FIELDS = {
    "base": ensemble_backward.CELL_BASIS,
    # the leverage guard off / tighter (it lowers the order at gap queries: biased toward the mean)
    "noguard": "cell:cubicv:adapt1.5:align0.9:both",
    "guard8": "cell:cubicv:adapt1.5:guard8:align0.9:both",
    # the full cubic (x-v cross terms) and the quadratic
    "cubic": "cell:cubic:adapt1.5:guard32:align0.9:both",
    "quad": "cell:quad:adapt1.5:guard32:align0.9:both",
    # pooled over the 3^D block
    "pool1": ensemble_backward.CELL_BASIS + ":pool1",
    # the per-query k-nearest quadratic (local in velocity: a bimodal cell is two clouds)
    "knn": "quadratic_full",
}
RATE_EDGES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, float("inf"))
"""Bins of collisions per particle per step (the virtual default was validated to 0.16)."""
_BUILD = EM.build
_VOLUME_TERM = EM._volume_term
SCALE = {}
CAPTURED = {}


def build_scaled(cfg, parameter, value, steps, body=None):
    """``EM.build`` with the flow volume of ``SCALE["mask"]``'s cells times ``1 + eps``."""
    out = _BUILD(cfg, parameter, value, steps, body)
    if SCALE:
        grid = out[0]
        vol = grid.volume * jnp.where(SCALE["mask"], 1.0 + SCALE["eps"], 1.0)
        out = (dataclasses.replace(grid, volume=vol), *out[1:])
    return out


def grab_volume_term(dev, rec, grid, mesh):
    """``EM._volume_term``, keeping the weights and the volumes it saw."""
    CAPTURED["w"] = np.asarray(shape.volume_weights(dev.cell_collision, grid))
    CAPTURED["v"] = np.asarray(grid.volume)
    CAPTURED["rate"] = np.asarray(collision_rate(rec.x, rec.alive, rec.pairs, grid))
    return _VOLUME_TERM(dev, rec, grid, mesh)


def step_occupancy(xs, grid):
    """One captured step's alive particles per collision cell, ``f64[C]``."""
    x, alive = xs
    n_cells = grid.volume.shape[0]
    cell = grid_mod.collision_cell(x, grid)
    ok = alive & (cell >= 0) & (cell < n_cells)
    return jnp.zeros(n_cells).at[jnp.where(ok, cell, 0)].add(ok.astype(jnp.float64))


@jax.jit
def collision_rate(x, alive, pairs, grid):
    """Accepted collisions per particle per step of every collision cell over the captured
    steps: ``2 sum_k accepted_kc / sum_k occupancy_kc`` (0 in empty cells)."""
    count = jnp.sum(jax.lax.map(functools.partial(step_occupancy, grid=grid), (x, alive)), axis=0)
    if hasattr(pairs, "n_att"):  # the compact lean record (2026-11-03): accepted entries
        n_cells = count.shape[0]
        cell = jnp.where(pairs.slot >= 0, pairs.cell, n_cells).reshape(-1)
        accepted = jnp.zeros(n_cells + 1).at[cell].add(1.0)[:n_cells]
    else:
        real = pairs.accept & (pairs.pi < x.shape[1])  # attempt slots on the scratch slot: none
        accepted = jnp.sum(real, axis=(0, 1)).astype(jnp.float64)
    return jnp.where(count > 0, 2.0 * accepted / jnp.maximum(count, 1.0), 0.0)


def objective(cfg, parameter, value, key):
    _a, _b, diag, _r, tallies, st = EM.forward(cfg, parameter, value, key, False)
    EM._check(diag)
    scale_r = float(st["params"].fnum[0]) / (cfg["window"] * float(st["params"].dt))
    return EM.objective_value(cfg, tallies, st["static"].D, scale_r)


def masks(cfg, parameter, value, names=("all", "cut")):
    """``{name: bool[C]}`` of the scaled sets ``names`` (module doc: ``all``, ``cut``,
    ``near<K>``)."""
    grid, mesh, *_ = _BUILD(cfg, parameter, value, 1)
    cells = jnp.zeros(grid.volume.shape, bool)
    cells = cells.at[: mesh.cell_type.shape[0]].set(jnp.asarray(mesh.cell_type) == cut.CUT)
    flow = np.asarray(grid.volume) > 0.0
    out = {"all": jnp.asarray(flow), "cut": cells}
    for name in names:
        if name.startswith("near"):
            out[name] = jnp.asarray(near_cut(grid, np.asarray(cells), int(name[4:])) & flow)
    return {name: out[name] for name in names}


def near_cut(grid, cut_mask, layers):
    """``bool[C]``: the cells whose centre lies within ``layers`` lattice cells (max-norm, per
    axis in units of the cell width) of a cut cell's centre, the cut cells included."""
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    centres = np.asarray(grid_mod.cell_centers(grid)) / dx
    anchors = centres[cut_mask]
    near = np.zeros(len(centres), bool)
    for start in range(0, len(centres), 4096):
        block = centres[start : start + 4096]
        gap = np.abs(block[:, None, :] - anchors[None, :, :]).max(axis=-1).min(axis=1)
        near[start : start + 4096] = gap <= layers + 1e-6
    return near


def base_fds(gate):
    """``{seed: row}`` of the base field's rows of ``gate`` (their FDs), or of the gate whose
    forward it shares (``variant_gate.FD_FROM``: the captured-horizon gates)."""
    paths = sorted(OUT_DIR.glob(f"volume_{V.FD_FROM.get(gate, gate)}_s*.jsonl"))
    return {r["seed"]: r for p in paths for r in map(json.loads, p.read_text().splitlines())}


def rate_columns(mask, name, seed, cfg, parameter, value):
    """The row's collision-rate columns (prediction and cells per :data:`RATE_EDGES` bin) and
    the seed's rate map, from the backward just run (:data:`CAPTURED`)."""
    rate, wv = CAPTURED["rate"], CAPTURED["w"] * CAPTURED["v"]
    m = np.asarray(mask) & (CAPTURED["v"] > 0.0)
    bins = np.digitize(rate, RATE_EDGES[1:-1])
    grid = _BUILD(cfg, parameter, value, 1)[0]
    centres = np.asarray(grid_mod.cell_centers(grid))
    np.savez(OUT_DIR / f"ratemap_{name}_s{seed}.npz",
             rate=rate, wv=wv, mask=m, centres=centres)  # fmt: skip
    return {
        "rate_edges": [e if np.isfinite(e) else None for e in RATE_EDGES],
        "pred_by_rate": [float(np.sum(wv[m & (bins == b)])) for b in range(len(RATE_EDGES) - 1)],
        "cells_by_rate": [int(np.sum(m & (bins == b))) for b in range(len(RATE_EDGES) - 1)],
        "rate_quantiles": [float(q) for q in np.quantile(rate[m], (0.5, 0.9, 0.99, 1.0))],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    ap.add_argument("--field", default="base", choices=sorted(FIELDS))
    ap.add_argument("--reuse-fd", action="store_true", help="read the FD from the base rows")
    ap.add_argument("--virtual", type=int, nargs=2, default=None, metavar=("S", "P"),
                    help="virtual pull-back substeps, partners (default 4 8, untagged)")  # fmt: skip
    ap.add_argument("--sets", nargs="+", default=["all", "cut"],
                    help="scaled sets: all, cut, near<K> (module doc)")  # fmt: skip
    args = ap.parse_args(argv)
    cfg, parameter, _steps, seed_start, _rows = V.gate_config(args.gate)
    value = float(EM.nominal(cfg, parameter))
    tag = "" if args.field == "base" else f"_{args.field}"
    virtual = (4, 8) if args.virtual is None else tuple(args.virtual)
    if args.virtual is not None:
        tag += f"_v{virtual[0]}x{virtual[1]}"
    if args.sets != ["all", "cut"]:
        if args.reuse_fd:
            raise ValueError("--reuse-fd reads the base rows' all / cut FDs only")
        tag += "_" + "-".join(args.sets)
    out = OUT_DIR / f"volume_{args.gate}{tag}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines()}
    reused = base_fds(args.gate) if args.reuse_fd else {}
    ensemble_backward.configure("virtual", *virtual, "staged", FIELDS[args.field])
    EM.build = build_scaled
    EM._volume_term = grab_volume_term
    sets = masks(cfg, parameter, value, tuple(args.sets))
    print({name: int(np.sum(np.asarray(m))) for name, m in sets.items()}, "cells", flush=True)
    flow = sets["all"] if "all" in sets else masks(cfg, parameter, value, ("all",))["all"]
    for seed in range(*args.seeds):
        if seed in done or (args.reuse_fd and seed not in reused):
            continue
        key = jax.random.key(seed_start + seed)
        t0 = time.time()
        CAPTURED.clear()
        _t, info, _s = EM.shape_terms(
            cfg, key, None, parameter, 32, cfg.get("timing_band", 10.0), ("cell_pool_plane",)
        )
        row = {"gate": args.gate, "field": args.field, "virtual": list(virtual), "seed": seed,
               "objective": info["objective"], "eps": list(EPS)}  # fmt: skip
        row |= rate_columns(flow, f"{args.gate}{tag}", seed, cfg, parameter, value)
        for name, mask in sets.items():
            m = np.asarray(mask)
            row[f"pred_{name}"] = float(np.sum(CAPTURED["w"][m] * CAPTURED["v"][m]))
            if args.reuse_fd:
                # the shared forward: its objective must be the FD rows' (checked below)
                row["fd_objective"] = reused[seed]["objective"]
                row[f"fd_{name}"] = reused[seed][f"fd_{name}"]
                continue
            fds = []
            for eps in EPS:
                vals = []
                for sign in (1.0, -1.0):
                    SCALE.update(mask=mask, eps=sign * eps)
                    vals.append(objective(cfg, parameter, value, key))
                    SCALE.clear()
                fds.append((vals[0] - vals[1]) / (2 * eps))
            row[f"fd_{name}"] = fds
        if args.reuse_fd and abs(row["objective"] - row["fd_objective"]) > 1e-9 * abs(
            row["objective"]
        ):
            print(f"WARNING seed {seed}: J {row['objective']} != the FD rows' "
                  f"{row['fd_objective']} -- not the same forward", flush=True)  # fmt: skip
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
