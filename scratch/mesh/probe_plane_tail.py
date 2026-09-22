"""Which hits carry the G3 pathwise term?  Per-particle contributions of one seed."""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

from examples.ensemble_inlet_shock import device_backward
from examples.mesh import ensemble_plane as P
from jax_dsmc import ensemble_device, shape
from jax_dsmc.grid import num_cells
from jax_dsmc.rng import STAGE_MOVE, key_for

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
objective = sys.argv[2] if len(sys.argv) > 2 else "shear-window"
cfg = P.default_config()
key = jax.random.key(32 + seed)
n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
state0, final, diag, rec, tallies, grid, mesh = P.forward(cfg, 0.0, key, True)
steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
first = cfg["warmup"] + m - w
phi_step, hit_reward = None, None
if objective == "energy-window":
    vy2 = jnp.where(rec.alive, rec.v_post[..., 1] ** 2, 0.0) / (n * w)
    phi_step = jnp.where((steps >= first)[:, None], vy2, 0.0)
else:
    hit_reward, _, err, _, _ = shape.replay(
        rec, steps, state0, key, grid, P.faces(), P.wall_params(cfg), P.MASS, jnp.float64(cfg["dt"]),
        mesh, 0.0, 0.0, P.static_of(cfg), functools.partial(P._shear_reward, first=first, scale=1.0 / (n * w)))
dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
scale = jnp.concatenate((jnp.asarray(dx), jnp.ones(3)))
npc_max = max(64, 4 * n // num_cells(grid))
_, dev = device_backward(rec, state0, final, jnp.zeros(n), n, grid, 1.0, P.species(cfg),
                         jax.random.fold_in(key, 12345), 32, npc_max=npc_max, phi_step=phi_step,
                         hit_reward=hit_reward, return_device=True)
static = P.static_of(cfg)
wall = P.wall_params(cfg)
geom = shape.geometry(mesh)


def per_particle(k, x, v, alive, lam_k):
    keyk = key_for(key, cfg["warmup"] + k, STAGE_MOVE)
    f = functools.partial(shape.wall_map, x=x, v=v, alive=alive, key=keyk, grid=grid,
                          faces=P.faces(), wall=wall, mass=P.MASS, dt=jnp.float64(cfg["dt"]),
                          mesh=mesh, max_bounces=8)
    # jacobian of the scalar sum_i gx_i.x_post_i + gv_i.v_post_i split per particle: use
    # per-particle cotangent rows via vmap over one-hot masks is too costly; instead take the
    # jvp along d/dtheta (translation): dx_post/dtheta for every particle
    def g(theta):
        out = f(geom + theta * jnp.array([1.0, 0.0]))
        return out[0], out[1], out[5]
    (x_post, v_post, ev), (dxp, dvp, _) = jax.jvp(g, (0.0,), (1.0,))
    gx, gv = ensemble_device.node_gradients(rec.x[k], rec.v[k], rec.alive[k], lam_k, grid, npc_max, scale, 32)
    contrib = jnp.sum(gx * dxp, axis=1) + jnp.sum(gv * dvp, axis=1)
    return contrib, dxp, ev, gx, lam_k


x, v, alive = state0.x, state0.v, state0.alive
rows = []
for k in range(m):
    c, dxp, ev, gx, lam_k = per_particle(k, x, v, alive, dev.lam[k])
    c = np.asarray(c); dxp = np.asarray(dxp); gx = np.asarray(gx); lam_k = np.asarray(lam_k)
    hit = np.asarray(ev.face[0] >= 4)
    vin = np.asarray(ev.v_in[0])
    for i in np.flatnonzero(hit):
        rows.append((k, i, c[i], vin[i, 0], np.linalg.norm(dxp[i]), gx[i, 0], lam_k[i]))
    x, v, alive = rec.x[k], rec.v_post[k], rec.alive[k]
rows = np.array(rows)
print("total", rows[:, 2].sum(), "hits", len(rows))
order = np.argsort(-np.abs(rows[:, 2]))
print(" step  slot   contrib     v_x,in   |dx_post/dtheta|   dV/dx      lam")
for r in rows[order[:12]]:
    print(f"{int(r[0]):5d} {int(r[1]):5d} {r[2]:+10.5f} {r[3]:+9.4f} {r[4]:10.2f} {r[5]:+10.3e} {r[6]:+10.3e}")
print("lam scale (rms over hits):", np.sqrt(np.mean(rows[:, 6] ** 2)), " |dV/dx| median:", np.median(np.abs(rows[:, 5])))
c = rows[:, 2]
print("share of |contrib| in top 5 %:", np.sort(np.abs(c))[::-1][: len(c) // 20].sum() / np.abs(c).sum())
print("corr(|contrib|, 1/|v_x|):", np.corrcoef(np.abs(c), 1 / np.abs(rows[:, 3]))[0, 1])
