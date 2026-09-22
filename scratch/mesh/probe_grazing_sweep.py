"""Grazing regularisation of the tally term: bias and variance vs the cut-off.

For each seed the analytic-circle forward is run once and the per-particle JVP of the
replayed step's ``v_x,post`` w.r.t. the radius is summed over the captured window
(the pathwise tally term up to the constant ``-m fnum / (W dt)``) for every cut-off
``eps`` of ``AnalyticBody.grazing``; the per-hit maximum and the count of hits below the
cut-off are kept.  Output ``probe_grazing_sweep.json`` (rows per seed x eps) for the
report's ``grazing_regularisation.png``.

    python3 scratch/mesh/probe_grazing_sweep.py --seeds 8 --eps 0 0.01 0.03 0.1 0.3
"""

import argparse
import dataclasses
import functools
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402


def _vx_post(geom, x, v, alive, key, setup):
    _x1, v1, _a, _lp, _tal, ev = shape.wall_map(geom, x, v, alive, key, **setup)
    return v1[:, 0], ev


def _jvp(geom, tangent, fn, x, v, alive, key):
    return jax.jvp(functools.partial(fn, x=x, v=v, alive=alive, key=key), (geom,), (tangent,))


def sweep_seed(cfg, key, eps_values, first, last):
    state0, _final, diag, rec, tallies, s = EM.forward(
        cfg, "radius", EM.nominal(cfg, "radius"), key, True
    )
    EM._check(diag)
    mesh, grid = s["mesh"], s["grid"]
    mass = float(s["sp"].mass[0])
    scale = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
    setup = {
        "grid": grid,
        "faces": s["faces"],
        "wall": s["wall"],
        "mass": s["sp"].mass[0],
        "dt": s["params"].dt,
        "mesh": mesh,
        "epsilon": 0.0,
        "acc_delta": 0.0,
        "max_bounces": s["static"].K_bounce,
    }
    out = {}
    for eps in eps_values:
        body = dataclasses.replace(mesh.body, grazing=float(eps))
        setup_e = dict(setup, mesh=dataclasses.replace(mesh, body=body))
        geom = shape.geometry(setup_e["mesh"])
        tangent = geom.__class__(
            centre=jnp.zeros_like(geom.centre), radius=jnp.ones_like(geom.radius), grazing=geom.grazing
        )
        fn = jax.jit(functools.partial(_vx_post, setup=setup_e))
        total, max_abs, n_hits, n_below, sum_sq = 0.0, 0.0, 0, 0, 0.0
        t0 = time.time()
        for k in range(first, last):
            x = state0.x if k == 0 else rec.x[k - 1]
            v = state0.v if k == 0 else rec.v_post[k - 1]
            alive = state0.alive if k == 0 else rec.alive[k - 1]
            kk = key_for(key, cfg["warmup"] + k, STAGE_MOVE)
            (_vx, ev), (dvx, _dev) = _jvp(geom, tangent, fn, x, v, alive, kk)
            hit = np.asarray(ev.face[0] >= 4)
            d = np.asarray(dvx)[hit]
            n = np.asarray(ev.normal[0])[hit, :2]
            vin = np.asarray(ev.v_in[0])[hit, :2]
            cos = np.abs(np.sum(n * vin, axis=1)) / np.maximum(np.linalg.norm(vin, axis=1), 1e-300)
            total += float(d.sum())
            sum_sq += float((d * d).sum())
            max_abs = max(max_abs, float(np.abs(d).max()) if d.size else 0.0)
            n_hits += int(hit.sum())
            n_below += int((cos < eps).sum())
        # the tally term of the drag objective: -m (v_x,post derivative) x fnum / (W dt)
        out[str(eps)] = {
            "tally": -mass * total * scale,
            "sum_dvx": total,
            "rms_per_hit": float(np.sqrt(sum_sq / max(n_hits, 1))),
            "max_abs_per_hit": max_abs,
            "hits": n_hits,
            "hits_below_cutoff": n_below,
            "seconds": time.time() - t0,
        }
        print(
            f"  eps {eps}: tally {out[str(eps)]['tally']:.5g}  max per hit {max_abs:.4g}  "
            f"rms {out[str(eps)]['rms_per_hit']:.4g}  hits {n_hits} (below cut-off {n_below})  "
            f"{out[str(eps)]['seconds']:.0f}s",
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=32)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--first", type=int, default=100)
    ap.add_argument("--last", type=int, default=300)
    args = ap.parse_args()
    cfg = EM.default_config()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_grazing_sweep.json")
    rows = json.load(open(path))["rows"] if os.path.exists(path) else []
    for i in range(len(rows), args.seeds):
        print(f"seed {i}", flush=True)
        key = jax.random.key(args.seed_start + i)
        rows.append({"seed": i, "eps": sweep_seed(cfg, key, args.eps, args.first, args.last)})
        with open(path, "w") as fh:
            json.dump({"config": cfg, "eps": args.eps, "rows": rows}, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
