"""Which estimate of ``grad lambda`` is best?  The collisionless plane case lets the TRUE
value gradient be computed: a particle's future depends only on its own state and its own
future wall draws, so V(z) = E[future window rewards | z] is a replica average of one
particle propagated from z, and grad_x V / grad_v V are CRN central differences of it.

For every plane hit of the captured steps (a subsample per step) this compares the
fitted slopes of several estimators -- two-sided, one-sided, pooled one-sided, and the
along-flight identity (``v . grad_x V = -T``, ``T`` the value of the timing rewards from
a second backward) with fitted or pooled perpendicular parts -- with the replica truth,
weighted by the sensitivity dx_post/dtheta the shape VJP applies, i.e. the x/v-part of
the estimator with each gradient.  Bias = sum_i (fit_i - true_i) . dx_i with its SE over
hits; the identity itself is checked per particle (``v . grad_x V`` exact vs ``-T``).
OBJECTIVE=shear-window|energy-window; REPLICAS, PER_STEP env.
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

SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
OBJECTIVE = os.environ.get("OBJECTIVE", "shear-window")
REPLICAS = int(__import__("os").environ.get("REPLICAS", 1024))
PER_STEP = int(__import__("os").environ.get("PER_STEP", 48))
DELTA_X = 0.005
DELTA_V = 0.02
cfg = P.default_config()
n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
first = cfg["warmup"] + m - w
dt = jnp.float64(cfg["dt"])
faces, wall = P.faces(), P.wall_params(cfg)
static = P.static_of(cfg)
scale_r = 1.0 / (n * w)


def _propagate_step(carry, k, key, k_start, grid, mesh):
    x, v, total = carry
    step = cfg["warmup"] + k
    kk = key_for(key, step, STAGE_MOVE)
    alive = jnp.ones(x.shape[0], bool)
    x1, v1, _, _, _, ev = shape.wall_map(
        shape.geometry(mesh), x, v, alive, kk, grid, faces, wall, P.MASS, dt, mesh, max_bounces=8
    )
    active = k > k_start
    if OBJECTIVE == "energy-window":
        r = v1[:, 1] ** 2 * scale_r
    else:
        r = jnp.sum(
            jnp.where(ev.face == 4, (ev.v_in[..., 1] - ev.v_out[..., 1]) * scale_r, 0.0), axis=0
        )
    total = total + jnp.where(active & (step >= first), r, 0.0)
    x = jnp.where(active, x1, x)
    v = jnp.where(active, v1, v)
    return (x, v, total), None


@jax.jit
def propagate(x0, v0, key, k_start, grid, mesh):
    """Future window rewards of a batch of particles from the post-move state of step
    ``k_start`` (a scan over all captured steps, inactive before ``k_start``)."""
    body = functools.partial(_propagate_step, key=key, k_start=k_start, grid=grid, mesh=mesh)
    (_, _, total), _ = jax.lax.scan(body, (x0, v0, jnp.zeros(x0.shape[0])), jnp.arange(m))
    return total


def true_gradients_fast(xs, vs, k, key, grid, mesh):
    """Same as above, every replica with its own key, vectorised over particles."""
    q = xs.shape[0]
    out = {}
    for name, delta, dim in (("x", DELTA_X, 2), ("v", DELTA_V, 3)):
        g = np.zeros((q, dim))
        for axis in range(dim):
            vals = []
            for sign in (1.0, -1.0):
                xr = jnp.repeat(xs, REPLICAS, axis=0)
                vr = jnp.repeat(vs, REPLICAS, axis=0)
                if name == "x":
                    xr = xr.at[:, axis].add(sign * delta)
                else:
                    vr = vr.at[:, axis].add(sign * delta)
                # one key per replica index, shared by all particles and both signs (CRN)
                tot = propagate(xr, vr, key, k, grid, mesh)
                vals.append(np.asarray(tot).reshape(q, REPLICAS))
            g[:, axis] = (vals[0] - vals[1]).mean(axis=1) / (2 * delta)
        out[name] = g
    return out["x"], out["v"]


VARIANTS = {
    # name: (one-sided, pool, k, along-flight)
    "two": (False, 0, 32, False),
    "one": (True, 0, 32, False),
    "one_pool": (True, 1, 96, False),
    "along": (True, 0, 32, True),
    "along_pool": (True, 1, 96, True),
}
JUMP = P._energy_jump if OBJECTIVE == "energy-window" else P._shear_jump


def backward(rec, state0, final, grid, key, npc_max, hit_reward=None, phi_step=None):
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
        phi_step=phi_step,
        return_device=True,
    )
    return dev


results = []
for seed in SEEDS:
    key = jax.random.key(32 + seed)
    state0, final, diag, rec, tallies, grid, mesh = P.forward(cfg, 0.0, key, True)
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    last = cfg["warmup"] + m - 1
    if OBJECTIVE == "energy-window":
        jump = functools.partial(P._energy_jump, first=first, scale=scale_r)
        reward = None
    else:
        jump = functools.partial(P._shear_jump, first=first, last=last, scale=scale_r)
        reward = functools.partial(P._shear_reward, first=first, scale=scale_r)
    rep = shape.replay(
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
        reward=reward,
        timing_jump=jump,
        timing_band=0.2,
    )
    hit_reward, phi_step = None, None
    if OBJECTIVE == "energy-window":
        vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0) * scale_r
        phi_step = jnp.where((steps >= first)[:, None], vy2, 0.0)
    else:
        hit_reward = rep.hit_reward
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    fit_scale = jnp.concatenate((jnp.asarray(dx), jnp.ones(3)))
    npc_max = max(64, 4 * n // num_cells(grid))
    dev = backward(rec, state0, final, grid, key, npc_max, hit_reward, phi_step)
    # the along-flight identity's value: the timing rewards carried backward
    dev_t = backward(rec, state0, final, grid, key, npc_max, hit_reward=rep.timing_reward)
    x, v, alive = state0.x, state0.v, state0.alive
    rows = []
    rng = np.random.default_rng(seed)
    for k in range(m - 1):  # the last step's post-move value is 0 (no future)
        kk = key_for(key, int(steps[k]), STAGE_MOVE)
        f = functools.partial(
            shape.wall_map,
            x=x,
            v=v,
            alive=alive,
            key=kk,
            grid=grid,
            faces=faces,
            wall=wall,
            mass=P.MASS,
            dt=dt,
            mesh=mesh,
            max_bounces=8,
        )

        def g(theta):
            out = f(shape.geometry(mesh) + theta * jnp.array([1.0, 0.0]))
            return out[0], out[1], out[5]

        (x_post, v_post, ev), (dxp, dvp, _) = jax.jvp(g, (0.0,), (1.0,))
        hit = np.asarray(ev.face[0] == 4) & np.asarray(alive)
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            x, v, alive = rec.x[k], rec.v_post[k], rec.alive[k]
            continue
        idx = rng.choice(idx, size=min(PER_STEP, idx.size), replace=False)
        direction = shape._last_normal(ev)
        fits = {}
        for name, (one_sided, pool, kk_fit, along) in VARIANTS.items():
            gx, gv = ensemble_device.node_gradients(
                rec.x[k],
                rec.v[k],
                rec.alive[k],
                dev.lam[k],
                grid,
                npc_max,
                fit_scale,
                kk_fit,
                direction=direction if one_sided else None,
                pool=pool,
            )
            if along:
                gx = shape.along_flight(gx, rec.v[k], dev_t.lam[k])
            fits[name] = (np.asarray(gx), np.asarray(gv))
        xs, vs = rec.x[k][idx], rec.v[k][idx]
        tgx, tgv = true_gradients_fast(xs, vs, k, jax.random.fold_in(key, 777 + k), grid, mesh)
        vd = np.asarray(rec.v[k][idx, :2])
        speed = np.linalg.norm(vd, axis=1)
        for j, i in enumerate(idx):
            row = {
                "k": k,
                "i": int(i),
                "dx": np.asarray(dxp[i]).tolist(),
                "dv": np.asarray(dvp[i]).tolist(),
                "true_x": tgx[j].tolist(),
                "true_v": tgv[j].tolist(),
                "lam": float(dev.lam[k][i]),
                # the identity: v . grad_x V (exact) against -T
                "vgrad_true": float(np.dot(tgx[j], vd[j])),
                "minus_T": float(-dev_t.lam[k][i]),
                "speed": float(speed[j]),
            }
            for name, (gx, gv) in fits.items():
                row[f"{name}_x"] = gx[i].tolist()
                row[f"{name}_v"] = gv[i].tolist()
            rows.append(row)
        x, v, alive = rec.x[k], rec.v_post[k], rec.alive[k]
        print(f"seed {seed} step {k}: {idx.size} hits sampled of {hit.sum()}", flush=True)
    n_hits_total = int(diag.surface_hits.sum())
    sampled = len(rows)
    dxs = np.array([r["dx"] for r in rows])
    dvs = np.array([r["dv"] for r in rows])

    def part(key_x, key_v):
        gx = np.array([r[key_x] for r in rows])
        gv = np.array([r[key_v] for r in rows])
        return np.sum(gx * dxs, axis=1) + np.sum(gv * dvs, axis=1)

    upscale = n_hits_total / sampled

    def summ(c):
        return upscale * c.sum(), upscale * c.std(ddof=1) * np.sqrt(len(c))

    c_true = part("true_x", "true_v")
    res = {"seed": seed, "hits": n_hits_total, "sampled": sampled, "x_part_true": summ(c_true)}
    for name in VARIANTS:
        c = part(f"{name}_x", f"{name}_v")
        res[f"x_part_{name}"] = summ(c)
        res[f"bias_{name}"] = summ(c - c_true)
        res[f"rms_{name}"] = float(np.sqrt(np.mean((c - c_true) ** 2)))
    vg = np.array([r["vgrad_true"] for r in rows])
    mt = np.array([r["minus_T"] for r in rows])
    res["identity"] = {
        "mean_vgrad_true": float(vg.mean()),
        "mean_minus_T": float(mt.mean()),
        "bias": float((mt - vg).mean()),
        "bias_se": float((mt - vg).std(ddof=1) / np.sqrt(len(vg))),
        "rms_diff": float(np.sqrt(np.mean((mt - vg) ** 2))),
        "scale": float(np.sqrt(np.mean(vg**2))),
        "corr": float(np.corrcoef(vg, mt)[0, 1]),
    }
    print(json.dumps(res), flush=True)
    results.append({**res, "rows": rows})
suffix = "" if OBJECTIVE == "shear-window" else "_energy"
json.dump(results, open(f"examples/mesh/results/plane_fit_variants{suffix}.json", "w"))
