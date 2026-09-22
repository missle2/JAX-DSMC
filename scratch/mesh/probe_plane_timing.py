"""The hit-timing boundary term of a velocity-dependent state reward (G3 energy-window).

For hits at t_hit within a band Delta of the step end, the reward collected at the end of
the step jumps from phi(v_in) to phi(v_out) when the hit moves across the step boundary;
moving the plane by delta removes the hits with t_hit in (dt - delta / v_x, dt].  The
band estimate of the boundary term is  - sum_hits 1{t_hit > dt - Delta} jump / (Delta v_x).
Prints it next to the pathwise term and the seed's CRN FD.
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from examples.mesh import ensemble_plane as P
from jax_dsmc import shape

cfg = P.default_config()
n, m, w = cfg["particles"], cfg["steps"], cfg["window"]
first = cfg["warmup"] + m - w
dt = cfg["dt"]
art = json.load(open(sys.argv[1]))
out = []
for row in art["rows"][: int(sys.argv[2]) if len(sys.argv) > 2 else 8]:
    key = jax.random.key(art["config"]["seed_start"] + row["seed"])
    state0, final, diag, rec, tallies, grid, mesh = P.forward(cfg, 0.0, key, True)
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    x, v, alive = state0.x, state0.v, state0.alive
    bands = {0.1: 0.0, 0.2: 0.0, 0.4: 0.0}
    for k in range(m):
        from jax_dsmc.rng import STAGE_MOVE, key_for
        keyk = key_for(key, int(steps[k]), STAGE_MOVE)
        _, _, _, _, _, ev = shape.wall_map(
            shape.geometry(mesh), x, v, alive, keyk, grid, P.faces(), P.wall_params(cfg), P.MASS,
            jnp.float64(dt), mesh, max_bounces=8)
        if int(steps[k]) >= first:
            hit = np.asarray(ev.face[0] >= 4) & np.asarray(alive)
            xh = np.asarray(ev.x_hit[0])[hit]
            vin = np.asarray(ev.v_in[0])[hit]
            vout = np.asarray(ev.v_out[0])[hit]
            t_hit = (xh[:, 0] - np.asarray(x)[hit][:, 0]) / vin[:, 0]
            jump = (vout[:, 1] ** 2 - vin[:, 1] ** 2) / (n * w)
            for b in bands:
                band = t_hit > dt * (1 - b)
                bands[b] += -np.sum(jump[band] / (b * dt * vin[band, 0]))
        x, v, alive = rec.x[k], rec.v_post[k], rec.alive[k]
    res = {"seed": row["seed"], "pathwise": row["pathwise"], "fd": row["fd"], **{f"band{b}": bands[b] for b in bands}}
    out.append(res)
    print(res, flush=True)
o = np.array([r["pathwise"] for r in out]); f = np.array([r["fd"][0] for r in out])
for b in bands:
    t = np.array([r[f"band{b}"] for r in out])
    tot = o + t
    print(f"band {b}: boundary {t.mean():+.3f} ± {t.std(ddof=1)/np.sqrt(len(t)):.3f}; pathwise+boundary {tot.mean():+.3f} ± {tot.std(ddof=1)/np.sqrt(len(t)):.3f}; FD(h=0.04) {f.mean():+.3f} ± {f.std(ddof=1)/np.sqrt(len(f)):.3f}")
