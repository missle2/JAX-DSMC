"""Why does one seed's tally term differ by 4 % between the analytic and the level-set
circle when the objective, the hits and the FD agree to 1e-5?

Per-particle, per-step derivative of the reflected x-momentum with respect to the radius
(forward-mode JVP of the replayed wall map: tangent ``dr = 1`` for the analytic body,
``dcoef = -1`` for the level set) on the captured window of one seed; the largest
discrepancies are listed with the hit's ``v . n / |v|``.  A grazing hit's derivative is
``O(1 / (v . n))`` and amplifies the level set's 1e-7 surface error.

    python3 scratch/mesh/probe_grazing_tally.py --seed 1
"""

import argparse
import functools
import json
import os
import sys

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


def per_particle(cfg, key, body_kind, first, last):
    cfg = dict(cfg, body=body_kind)
    state0, _final, diag, rec, _tallies, s = EM.forward(cfg, "radius", EM.nominal(cfg, "radius"), key, True)
    EM._check(diag)
    mesh, grid = s["mesh"], s["grid"]
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
    geom = shape.geometry(mesh)
    if body_kind == "levelset":
        tangent = -jnp.ones_like(geom)
    else:
        tangent = geom.__class__(centre=jnp.zeros_like(geom.centre), radius=jnp.ones_like(geom.radius))
    fn = jax.jit(functools.partial(_vx_post, setup=setup))
    out = {}
    for k in range(first, last):
        x = state0.x if k == 0 else rec.x[k - 1]
        v = state0.v if k == 0 else rec.v_post[k - 1]
        alive = state0.alive if k == 0 else rec.alive[k - 1]
        kk = key_for(key, cfg["warmup"] + k, STAGE_MOVE)
        (_vx, ev), (dvx, _dev) = jax.jvp(lambda g: fn(g, x, v, alive, kk), (geom,), (tangent,))
        hit = np.asarray(ev.face[0] >= 4)
        n = np.asarray(ev.normal[0])[:, :2]
        vin = np.asarray(ev.v_in[0])[:, :2]
        cos = np.abs(np.sum(n * vin, axis=1)) / np.maximum(np.linalg.norm(vin, axis=1), 1e-300)
        out[k] = (np.asarray(dvx), hit, cos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first", type=int, default=100)
    ap.add_argument("--last", type=int, default=300)
    args = ap.parse_args()
    cfg = EM.default_config()
    key = jax.random.key(32 + args.seed)  # ensemble_mesh --seed-start default
    res = {kind: per_particle(cfg, key, kind, args.first, args.last) for kind in ("analytic", "levelset")}
    rows = []
    for k in range(args.first, args.last):
        da, ha, ca = res["analytic"][k]
        dl, hl, cl = res["levelset"][k]
        both = ha & hl
        diff = np.where(both, dl - da, 0.0)
        i = int(np.argmax(np.abs(diff)))
        rows.append(
            {
                "step": k,
                "sum_analytic": float(da[ha].sum()),
                "sum_levelset": float(dl[hl].sum()),
                "hits": int(ha.sum()),
                "hits_differ": int((ha != hl).sum()),
                "max_abs_diff": float(abs(diff[i])),
                "at_cos": float(ca[i]),
                "d_analytic": float(da[i]),
                "d_levelset": float(dl[i]),
            }
        )
    tot_a = sum(r["sum_analytic"] for r in rows)
    tot_l = sum(r["sum_levelset"] for r in rows)
    print(f"sum over window of d v_x,post / dr: analytic {tot_a:.6g} levelset {tot_l:.6g} rel diff {(tot_l - tot_a) / tot_a:+.3e}")
    rows.sort(key=lambda r: -r["max_abs_diff"])
    print("largest per-particle discrepancies (step, |diff|, cos of incidence, d_analytic, d_levelset):")
    for r in rows[:8]:
        print(f"  step {r['step']}: |diff| {r['max_abs_diff']:.4g}  cos {r['at_cos']:.2e}  {r['d_analytic']:.4g} vs {r['d_levelset']:.4g}  (hits {r['hits']}, hit sets differ {r['hits_differ']})")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_grazing_tally_seed{args.seed}.json")
    with open(path, "w") as fh:
        json.dump({"seed": args.seed, "total_analytic": tot_a, "total_levelset": tot_l, "rows": rows}, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
