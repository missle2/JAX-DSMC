"""The fit lab's data (2026-11-04): one gate seed's forward and backward, then every captured
step's near-wall particles and body hits saved for offline grad-lambda fits.

The drag cos 2 theta excess sits in pathwise_x at the stagnation nose (bumpchk): there the
diffuse hit's dx_post ~ delta n, so the term is the wall-normal derivative of the re-emitted
particle's value, fitted per cell with one x-gradient for every velocity.  The lab tests fits
of that derivative against (1) a reference pooled over every step and seed (the steady window:
lambda(x, v, t) ~ c(t) + V(x, v)) and (2) closure: any fit's pathwise_x recomputed exactly from
the saved hit Jacobians, plus the bump gate's other terms, against the FD.

Saved per seed (``scratch/fitlab/cache/<gate>_s<seed>.npz`` + ``.json``):

* nodes: every alive slot in the near-wall cells (cell centre within ``--width`` cells of the
  wall), every captured step -- step, slot, cell, x, v, lambda, the timing value T, the
  collision increment dt L* lambda, steps since the slot's last body hit (-1 none / newborn)
  and that hit's normal;
* hits: every surviving slot with a body hit -- step, slot, cell, x_post, v_post, the
  pre-move v, the last hit normal, lambda, T, dt L* lambda, d x_post / d theta and
  d v_post / d theta per control (forward-mode through the wall map), and each production
  variant's (grad_x, grad_v) at the query (after the along-flight identity);
* json: the gate config, controls, variants, the production terms per control
  (``variant_gate.terms_row``) and the reproduction check: sum over hits of
  gx . dx + gv . dv against pathwise_x + pathwise_v.

    PYTHONPATH=. python3 scratch/fitlab/dump.py --gate c10_diff_m2 --seeds 0 4 [--smoke]
"""

import argparse
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
from jax_dsmc import ensemble_backward, shape  # noqa: E402
from jax_dsmc.collide.ntc import step_v_post  # noqa: E402
from jax_dsmc.ensemble_device import node_gradients  # noqa: E402
from jax_dsmc.grid import cell_centers, cell_of, spacing  # noqa: E402
from jax_dsmc.mesh import levelset as levelset_mod  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402

OUT = Path("scratch/fitlab/cache")
CONTROLS = ("radius", "mode2", "mode2s", "bump_nose", "bump_side", "bump_base")
VARIANTS = ("cell_along_pool_plane", "cell_pool_plane", "along_pool_plane")

_STASH = {}
_ORIGINAL = shape.gradient_variants


def _stashing(lam_timing, names=None, collision=None, **kwargs):
    """``shape.gradient_variants`` that keeps its inputs (the record, values, setup)."""
    _STASH.clear()
    _STASH.update(kwargs, lam_timing=lam_timing, collision=collision)
    return _ORIGINAL(lam_timing, names=names, collision=collision, **kwargs)


def _moved(geom, x, v, alive, key, setup_map):
    x_post, v_post, *_ = shape.wall_map(geom, x, v, alive, key, **setup_map)
    return x_post, v_post


def _tangent(t, geom, f):
    return jax.jvp(f, (geom,), (t,))[1]


def _fit_list(npc_max, scale):
    fits = []
    for name in VARIANTS:
        e = shape.VARIANTS[name]
        fits.append({
            "npc_max": npc_max, "scale": scale, "k": e[1],
            "basis": e[4] if len(e) > 4 else "quadratic_full", "chunk": 4096, "pool": e[0],
            "halfspace": e[3] if len(e) > 3 else None, "along": e[2],
        })  # fmt: skip
    return tuple(fits)


def _step(geom, tangents, x, v, alive, rec_k, lam_k, lt_k, coll_k, step, base_key, setup, fits):
    """One captured step: the hit mask, the hits' tangents and each variant's fits."""
    setup_map = setup["map"]
    key = key_for(base_key, step, STAGE_MOVE)
    _x, _v, _alive, _logp, _tallies, events = shape.wall_map(geom, x, v, alive, key, **setup_map)
    dim = x.shape[1]
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    queries, lost = shape._body_hit_slots(events, survivors, dim)
    n = x.shape[0]
    hit = jnp.zeros(n, bool).at[jnp.where(queries >= 0, queries, n)].set(True, mode="drop")
    direction = shape._last_normal(events)
    f = functools.partial(_moved, x=x, v=v, alive=alive, key=key, setup_map=setup_map)
    dx, dv = jax.vmap(functools.partial(_tangent, geom=geom, f=f))(tangents)
    gxs, gvs = [], []
    for fit in fits:
        settings = {k: val for k, val in fit.items() if k != "along"}
        gx, gv = node_gradients(
            rec_k.x, rec_k.v, rec_k.alive, lam_k, setup_map["grid"], direction=direction,
            klass=None, queries=queries, **settings,
        )  # fmt: skip
        if fit["along"]:
            gx = shape.along_flight(gx, rec_k.v, lt_k + coll_k / setup_map["dt"])
        gxs.append(jnp.where(survivors[:, None], gx, 0.0))
        gvs.append(jnp.where(survivors[:, None], gv, 0.0))
    return {
        "hit": hit, "lost": lost, "direction": direction, "dx": dx, "dv": dv,
        "gx": jnp.stack(gxs), "gv": jnp.stack(gvs), "survivors": survivors,
        "cell": cell_of(rec_k.x, setup_map["grid"]),
    }  # fmt: skip


def near_wall_cells(ls, grid, width):
    """``bool[C]``: cells whose centre is within ``width`` cells of the wall (outside it,
    or inside by at most one cell: the cut cells)."""
    dx = float(jnp.max(spacing(grid)))
    phi = np.asarray(levelset_mod.value(ls, cell_centers(grid)))
    return (phi < width * dx) & (phi > -dx)


def dump_seed(gate, seed, smoke, width):
    cfg, _par, _steps, seed_start, _rows = V.gate_config(gate)
    tag = ""
    if smoke:
        cfg = {**cfg, "fnum_scale": 16.0 * cfg["fnum_scale"], "warmup": 150, "steps": 40,
               "window": 20}  # fmt: skip
        tag = "_smoke"
    ensemble_backward.configure("virtual", 4, 8, "staged", V.BASES["base"])
    key = jax.random.key(seed_start + seed)
    t0 = time.time()
    shape.gradient_variants = _stashing
    try:
        all_terms, info, s = EM.shape_terms(
            cfg, key, None, CONTROLS[0], 32, cfg.get("timing_band", 10.0), VARIANTS
        )
    finally:
        shape.gradient_variants = _ORIGINAL
    production = {}
    for par in CONTROLS:
        terms, volume = V.terms_row(cfg, par, all_terms, s)
        production[par] = {"terms": terms, "volume": volume}
    t_gate = time.time() - t0
    st = dict(_STASH)
    grid, mesh, rec = st["grid"], st["mesh"], st["records"]
    geom = shape.geometry(mesh)
    tangents = []
    for par in CONTROLS:
        coef_of = functools.partial(EM._coef_of, cfg=cfg, parameter=par, grid=grid)
        base, t = jax.jvp(coef_of, (jnp.float64(EM.nominal(cfg, par)),), (jnp.float64(1.0),))
        gap = float(jnp.max(jnp.abs(base - geom)))
        if gap > 1e-12:
            raise RuntimeError(f"{par}: nominal coefficients differ from the run's by {gap:g}")
        tangents.append(t)
    tangents = jnp.stack(tangents)
    setup = shape._setup(
        st["base_key"], grid, st["faces"], st["wall"], st["mass"], st["dt"], mesh,
        st["epsilon"], st["acc_delta"], st["static"],
    )  # fmt: skip
    fits = _fit_list(st["npc_max"], st["scale"])
    step_fn = jax.jit(functools.partial(_step, base_key=st["base_key"], setup=setup, fits=fits))
    region = near_wall_cells(mesh.body, grid, width)
    lam, lt, coll = st["lam"], st["lam_timing"], st["collision"]
    m, n = lam.shape[0], lam.shape[1]
    since = np.full(n, -1, np.int64)
    last_normal = np.zeros((n, 3))
    nodes = {k: [] for k in ("step", "slot", "cell", "x", "v", "lam", "lt", "coll", "since",
                             "normal")}  # fmt: skip
    hits = {k: [] for k in ("step", "slot", "cell", "x", "v", "v_pre", "normal", "lam", "lt",
                            "coll", "dx", "dv", "gx", "gv")}  # fmt: skip
    x, v, alive = st["state0"].x, st["state0"].v, st["state0"].alive
    lost_total = 0
    for k in range(m):
        rec_k = jax.tree.map(functools.partial(_take, k=k), rec)
        out = step_fn(geom, tangents, x, v, alive, rec_k, lam[k], lt[k], coll[k],
                      st["steps"][k])  # fmt: skip
        out = jax.tree.map(np.asarray, out)
        lost_total += int(out["lost"])
        r_alive = np.asarray(rec_k.alive)
        born = np.asarray(rec_k.born_face) >= 0
        since = np.where(since >= 0, since + 1, since)
        since[born | ~r_alive] = -1
        h = out["hit"]
        since[h] = 0
        last_normal[h] = out["direction"][h]
        cells = out["cell"]
        rx, rv = np.asarray(rec_k.x), np.asarray(rec_k.v)
        lam_k, lt_k, coll_k = (np.asarray(a[k]) for a in (lam, lt, coll))
        keep = r_alive & region[cells]
        idx = np.nonzero(keep)[0]
        for name, val in (("step", np.full(idx.size, k)), ("slot", idx), ("cell", cells[idx]),
                          ("x", rx[idx]), ("v", rv[idx]), ("lam", lam_k[idx]),
                          ("lt", lt_k[idx]), ("coll", coll_k[idx]), ("since", since[idx]),
                          ("normal", last_normal[idx])):  # fmt: skip
            nodes[name].append(val)
        idx = np.nonzero(h)[0]
        for name, val in (("step", np.full(idx.size, k)), ("slot", idx), ("cell", cells[idx]),
                          ("x", rx[idx]), ("v", rv[idx]), ("v_pre", np.asarray(v)[idx]),
                          ("normal", out["direction"][idx]), ("lam", lam_k[idx]),
                          ("lt", lt_k[idx]), ("coll", coll_k[idx]),
                          ("dx", out["dx"][:, idx]), ("dv", out["dv"][:, idx]),
                          ("gx", out["gx"][:, idx]), ("gv", out["gv"][:, idx])):  # fmt: skip
            hits[name].append(val)
        x, v, alive = rec_k.x, step_v_post(rec_k), rec_k.alive
    axis = {"dx": 1, "dv": 1, "gx": 1, "gv": 1}
    arrays = {f"node_{k}": np.concatenate(val) for k, val in nodes.items()}
    arrays.update({f"hit_{k}": np.concatenate(val, axis=axis.get(k, 0))
                   for k, val in hits.items()})  # fmt: skip
    # the reproduction check: the production pathwise_x + pathwise_v per control and variant
    check = {}
    for c, par in enumerate(CONTROLS):
        check[par] = {}
        for j, name in enumerate(VARIANTS):
            px = float(np.sum(arrays["hit_gx"][j] * arrays["hit_dx"][c]))
            pv = float(np.sum(arrays["hit_gv"][j] * arrays["hit_dv"][c]))
            ref = production[par]["terms"][name]
            check[par][name] = {"pathwise_x": [px, ref["pathwise_x"]],
                                "pathwise_v": [pv, ref["pathwise_v"]]}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{gate}{tag}_s{seed}"
    np.savez_compressed(f"{stem}.npz", **arrays)
    meta = {
        "gate": gate, "seed": seed, "key": seed_start + seed, "config": cfg,
        "controls": CONTROLS, "variants": VARIANTS, "width": width,
        "objective": info["objective"], "production": production, "check": check,
        "lost_hits": lost_total, "dt": float(st["dt"]), "scale": np.asarray(st["scale"]).tolist(),
        "npc_max": int(st["npc_max"]), "grid": {"lo": np.asarray(grid.lo).tolist(),
        "hi": np.asarray(grid.hi).tolist(), "n": list(grid.n)},
        "steps": np.asarray(st["steps"]).tolist(), "region_cells": int(region.sum()),
        "rows": {"nodes": int(arrays["node_step"].size), "hits": int(arrays["hit_step"].size)},
        "seconds_gate": t_gate, "seconds": time.time() - t0,
    }  # fmt: skip
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=1, default=str))
    worst = max(
        abs(a - b) / max(abs(b), 1e-12)
        for d in check.values() for e in d.values() for a, b in e.values()
    )  # fmt: skip
    size = Path(f"{stem}.npz").stat().st_size / 1e6
    print(f"{gate} seed {seed}: nodes {meta['rows']['nodes']} hits {meta['rows']['hits']} "
          f"cells {meta['region_cells']} lost {lost_total}  {size:.0f} MB  reproduction "
          f"max rel {worst:.2e}  {meta['seconds']:.0f}s", flush=True)  # fmt: skip


def _take(a, k):
    return a[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--width", type=float, default=2.5, help="near-wall band, cells")
    ap.add_argument("--smoke", action="store_true", help="fnum x 16, 150 + 40 steps, *_smoke")
    args = ap.parse_args()
    for seed in range(*args.seeds):
        dump_seed(args.gate, seed, args.smoke, args.width)


if __name__ == "__main__":
    main()
