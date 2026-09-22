"""Device virtual pull-back vs the host one on the relaxation case (different partner draws)."""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples import ensemble_relaxation as er
from jax_dsmc import ensemble_backward as host
from jax_dsmc import ensemble_device as device
from jax_dsmc.collide.vss import collision_rate

nodes = sys.argv[1] if len(sys.argv) > 1 else "post"
S, K = int(sys.argv[2]) if len(sys.argv) > 2 else 2, int(sys.argv[3]) if len(sys.argv) > 3 else 8
cfg = dict(particles=2000, steps=10, attempts=2048, dt=0.2, majorant="certified")
sp = er.species(er.controls("diameter", 1.0))


def log_rate(c, vi, vj):
    return jnp.log(collision_rate(vi, vj, 0, 0, er.species(er.controls("diameter", c))))


def pair_score(vi, vj):
    return jax.grad(log_rate)(1.0, vi, vj)


forward = er._compiled_forward(**cfg, capture=True)
for seed in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
    key = jax.random.PRNGKey(seed)
    state0, out, diag, rec, grid = forward(er.controls("diameter", jnp.float64(1.0)), key)
    host.configure("virtual", S, K, nodes)
    t0 = time.time()
    adj = host.backward(rec, state0, out, er._vx2, 2000, grid, 1.0, 32, sp=sp)
    g_host = sum(float(np.sum(p.weight * np.asarray(jax.vmap(pair_score)(jnp.asarray(p.vi), jnp.asarray(p.vj))) * p.four_term)) for p in adj.pairs if len(p.i))
    th = time.time() - t0
    # staged velocities: exact replay must agree bit for bit
    if nodes == "staged":
        for k in range(cfg["steps"]):
            n = 2000
            ids, valid, cell, rank, count = device.cell_nodes(rec.x[k], rec.alive[k], grid, 2048)
            st_dev = np.asarray(device.staged_velocities(jax.tree.map(lambda a: a[k], rec.pairs), rec.v[k], rec.v_post[k], cell, S))
            st_host = host._staged_velocities(rec.pairs, k, np.asarray(rec.v[k]), np.asarray(rec.v_post[k]), n, S)
            assert np.array_equal(st_dev, st_host), f"staged velocities differ at step {k}"
    scale = jnp.concatenate((jnp.ones(1), jnp.ones(3)))
    t0 = time.time()
    dev = device.backward_virtual(rec, state0.alive, out, er._vx2(out.x, out.v) / 2000, grid, scale, sp, jax.random.key(seed), 2048, 32, "quadratic_full", 4096, S, K, nodes, pair_score)
    g_dev = float(jnp.sum(dev.kernel))
    td = time.time() - t0
    print(f"seed {seed}: host {g_host:.5f} ({th:.1f}s)  device {g_dev:.5f} ({td:.1f}s)  lambda0 corr {np.corrcoef(adj.lambda0, np.asarray(dev.lambda0))[0,1]:.6f}  rel diff {abs(g_dev-g_host)/abs(g_host):.3%}", flush=True)
