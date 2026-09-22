"""Where is the one-sided fit needed, and what does it cost?  Plane case (collisionless:
replica-exact values and gradients are available at any state).

Particle classes at the post-move state of a step: (hit) reflected off the plane this
step; (near-in) within one cell of the plane, approaching, no hit; (near-out) within one
cell, receding, no hit; (wall0-in/out) within one cell of the fixed diffuse wall at
x = 0; (interior) everywhere else.  For each class and each fit variant -- two-sided,
one-sided by the sign of v . n_local (n_local = normal of the nearest surface of the
particle's cell; the shape adjoint's "last hit normal" coincides with it for the hit
class) -- the fitted value / grad_x / grad_v are compared with 1024-replica truths:
bias (mean error), rms error, and the sd of the fitted quantity (its variance cost).
"""

import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from examples.ensemble_inlet_shock import device_backward
from examples.mesh import ensemble_plane as P
from jax_dsmc import ensemble_device, shape
from jax_dsmc.grid import num_cells
from jax_dsmc.rng import STAGE_MOVE, key_for

SEEDS = [int(a) for a in sys.argv[1:]] or [0]
REPLICAS = int(os.environ.get("REPLICAS", 1024))
PER_CLASS = int(os.environ.get("PER_CLASS", 32))
STEP_STRIDE = int(os.environ.get("STEP_STRIDE", 3))
DELTA_X, DELTA_V = 0.005, 0.02
cfg = P.default_config()
n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
first = cfg["warmup"] + m - w
dt = jnp.float64(cfg["dt"])
faces, wall = P.faces(), P.wall_params(cfg)
static = P.static_of(cfg)
scale_r = 1.0 / (n * w)
GAP = cfg["gap0"]
H = cfg["length"] / cfg["cells_x"]


def _propagate_step(carry, k, key, k_start, grid, mesh):
    x, v, total = carry
    step = cfg["warmup"] + k
    kk = key_for(key, step, STAGE_MOVE)
    alive = jnp.ones(x.shape[0], bool)
    x1, v1, _, _, _, ev = shape.wall_map(
        shape.geometry(mesh), x, v, alive, kk, grid, faces, wall, P.MASS, dt, mesh, max_bounces=8
    )
    active = k > k_start
    r = jnp.where(ev.face == 4, (ev.v_in[..., 1] - ev.v_out[..., 1]) * scale_r, 0.0)
    total = total + jnp.where(active & (step >= first), jnp.sum(r, axis=0), 0.0)
    return (jnp.where(active, x1, x), jnp.where(active, v1, v), total), None


@jax.jit
def propagate(x0, v0, key, k_start, grid, mesh):
    body = functools.partial(_propagate_step, key=key, k_start=k_start, grid=grid, mesh=mesh)
    (_, _, total), _ = jax.lax.scan(body, (x0, v0, jnp.zeros(x0.shape[0])), jnp.arange(m))
    return total


def truth(xs, vs, k, key, grid, mesh):
    """Replica value, grad_x (q, 2), grad_v (q, 3)."""
    q = xs.shape[0]
    xr, vr = jnp.repeat(xs, REPLICAS, axis=0), jnp.repeat(vs, REPLICAS, axis=0)
    value = np.asarray(propagate(xr, vr, key, k, grid, mesh)).reshape(q, REPLICAS).mean(axis=1)
    out = {}
    for name, delta, dim in (("x", DELTA_X, 2), ("v", DELTA_V, 3)):
        g = np.zeros((q, dim))
        for axis in range(dim):
            vals = []
            for sign in (1.0, -1.0):
                xp, vp = xr, vr
                if name == "x":
                    xp = xr.at[:, axis].add(sign * delta)
                else:
                    vp = vr.at[:, axis].add(sign * delta)
                vals.append(np.asarray(propagate(xp, vp, key, k, grid, mesh)).reshape(q, REPLICAS))
            g[:, axis] = (vals[0] - vals[1]).mean(axis=1) / (2 * delta)
        out[name] = g
    return value, out["x"], out["v"]


def classify(x, v, hit):
    xx, vx = np.asarray(x[:, 0]), np.asarray(v[:, 0])
    near_plane = xx > GAP - H
    near_wall0 = xx < H
    cls = np.full(x.shape[0], "interior", dtype=object)
    cls[near_wall0 & (vx < 0)] = "wall0-in"
    cls[near_wall0 & (vx > 0)] = "wall0-out"
    cls[near_plane & (vx > 0)] = "near-in"
    cls[near_plane & (vx < 0)] = "near-out"
    cls[hit] = "hit"
    return cls


def local_normal(x):
    """Normal (into the flow) of the nearest surface: the plane for x > gap/2, the fixed
    wall otherwise (both along x here)."""
    return jnp.where(
        (x[:, 0] > GAP / 2)[:, None], jnp.array([-1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])
    )


results = []
for seed in SEEDS:
    key = jax.random.key(32 + seed)
    state0, final, diag, rec, tallies, grid, mesh = P.forward(cfg, 0.0, key, True)
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    hit_reward, _, err, _, _ = shape.replay(
        rec,
        steps,
        state0,
        key,
        grid,
        faces,
        wall,
        P.MASS,
        dt,
        mesh,
        0.0,
        0.0,
        static,
        functools.partial(P._shear_reward, first=first, scale=scale_r),
    )
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    fit_scale = jnp.concatenate((jnp.asarray(dx), jnp.ones(3)))
    npc_max = max(64, 4 * n // num_cells(grid))
    _, dev = device_backward(
        rec,
        state0,
        final,
        jnp.zeros(n),
        n,
        grid,
        1.0,
        P.species(cfg),
        jax.random.fold_in(key, 12345),
        32,
        npc_max=npc_max,
        hit_reward=hit_reward,
        return_device=True,
    )
    x, v, alive = state0.x, state0.v, state0.alive
    rows = []
    rng = np.random.default_rng(seed)
    for k in range(0, m - 1, STEP_STRIDE):
        kk = key_for(key, int(steps[k]), STAGE_MOVE)
        _, _, _, _, _, ev = shape.wall_map(
            shape.geometry(mesh),
            x,
            v,
            alive,
            kk,
            grid,
            faces,
            wall,
            P.MASS,
            dt,
            mesh,
            max_bounces=8,
        )
        hit = np.asarray(ev.face[0] == 4) & np.asarray(alive)
        cls = classify(rec.x[k], rec.v[k], hit)
        idx = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(cls == c), size=min(PER_CLASS, (cls == c).sum()), replace=False
                )
                for c in np.unique(cls)
            ]
        ).astype(int)
        # fits: two-sided; one-sided by the query's own side of the local normal
        nrm = local_normal(rec.x[k])
        side = jnp.sign(jnp.sum(rec.v[k] * nrm, axis=1))[:, None] * nrm
        fits = {}
        for name, direction in (("two", None), ("one", side)):
            ids, valid, cell, _, count = ensemble_device.cell_nodes(
                rec.x[k], rec.alive[k], grid, npc_max
            )
            c = jnp.minimum(cell, count.shape[0] - 1)
            tables = (rec.x[k][ids], rec.v[k][ids], valid, dev.lam[k][ids])
            val, gx, gv = ensemble_device.cell_query(
                tables,
                c,
                rec.x[k],
                rec.v[k],
                fit_scale,
                32,
                "quadratic_full",
                4096,
                gradients=True,
                direction=direction,
            )
            fits[name] = (np.asarray(val), np.asarray(gx), np.asarray(gv))
        tv, tgx, tgv = truth(
            rec.x[k][idx], rec.v[k][idx], k, jax.random.fold_in(key, 777 + k), grid, mesh
        )
        for j, i in enumerate(idx):
            rows.append(
                {
                    "k": k,
                    "cls": str(cls[i]),
                    "lam": float(dev.lam[k][i]),
                    "true": [float(tv[j]), tgx[j].tolist(), tgv[j].tolist()],
                    **{
                        f"fit_{nm}": [float(f[0][i]), f[1][i].tolist(), f[2][i].tolist()]
                        for nm, f in fits.items()
                    },
                }
            )
        x, v, alive = rec.x[k], rec.v_post[k], rec.alive[k]
        print(f"seed {seed} step {k}: {idx.size} particles", flush=True)
    # summary per class
    summary = {}
    for c in sorted({r["cls"] for r in rows}):
        rr = [r for r in rows if r["cls"] == c]
        tv = np.array([r["true"][0] for r in rr])
        tgx = np.array([r["true"][1] for r in rr])
        tgv = np.array([r["true"][2] for r in rr])
        entry = {
            "n": len(rr),
            "scale_value": float(np.sqrt(np.mean(tv**2))),
            "scale_gx": float(np.sqrt(np.mean(tgx**2))),
            "scale_gv": float(np.sqrt(np.mean(tgv**2))),
        }
        for nm in ("two", "one"):
            fv = np.array([r[f"fit_{nm}"][0] for r in rr])
            fgx = np.array([r[f"fit_{nm}"][1] for r in rr])
            fgv = np.array([r[f"fit_{nm}"][2] for r in rr])
            for q, f, t in (("value", fv, tv), ("gx", fgx, tgx), ("gv", fgv, tgv)):
                e = f - t
                entry[f"{nm}_{q}_bias"] = (
                    float(e.mean()) if e.ndim == 1 else e.mean(axis=0).tolist()
                )
                entry[f"{nm}_{q}_bias_se"] = (
                    float(e.std(ddof=1) / np.sqrt(len(e)))
                    if e.ndim == 1
                    else (e.std(axis=0, ddof=1) / np.sqrt(len(e))).tolist()
                )
                entry[f"{nm}_{q}_rms"] = float(np.sqrt(np.mean(e**2)))
                entry[f"{nm}_{q}_sd"] = (
                    float(f.std()) if f.ndim == 1 else float(np.sqrt(np.mean(f.var(axis=0))))
                )
        summary[c] = entry
    print(json.dumps({"seed": seed, "summary": summary}, indent=1), flush=True)
    results.append({"seed": seed, "summary": summary, "rows": rows})
json.dump(results, open("examples/mesh/results/plane_fit_classes.json", "w"))
