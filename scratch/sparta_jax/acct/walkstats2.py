"""Work counts of the new sparta_jax move at steady state (steps 1000-1039, seed 7)."""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
from sparta_jax import sim  # noqa: E402

cfg, consts, geo = sim.make()
C = cfg.ncell
st = sim.empty_state(cfg)
st, _ = sim.run(st, 7, jnp.int32(0), consts, cfg, 1000)
seed = jnp.uint64(7)


@jax.jit
def first(state, istep):
    state, slot, n_in, *_ = sim.emit(state, seed, istep, consts, cfg)
    new = sim.dense_move(state, cfg)
    near = new.cell > C
    rank = sim.cumsum(near.astype(jnp.int32)) - 1
    big = cfg._replace(lane=8192) if hasattr(cfg, "_replace") else None
    slots, w = sim._lane_gather(new, 0, rank, near, consts, LCFG)
    alive = jnp.sum(state.cell != C)
    cellw = (w.ij & 0xFFFF) + (w.ij >> 16) * cfg.nx
    act = sim._status(w.flags) == sim.ACTIVE
    incut = jnp.sum(act & (consts.nsurf[cellw] > 0))
    return w, slots, alive, jnp.sum(near), incut, n_in, jnp.sum(new.cell == C) - jnp.sum(state.cell == C)


import dataclasses
LCFG = dataclasses.replace(cfg, lane=8192)


@jax.jit
def stage(w, slots, it, istep):
    act = sim._status(w.flags) == sim.ACTIVE
    cellw = (w.ij & 0xFFFF) + (w.ij >> 16) * cfg.nx
    useful = jnp.sum(jnp.where(act, consts.nsurf[cellw], 0))
    incut = jnp.sum(act & (consts.nsurf[cellw] > 0))
    w2, cnt, _ = sim._walk_stage(w, slots, seed, istep, it, jnp.zeros((cfg.nlines, 3)), consts, cfg)
    return w2, jnp.sum(act), useful, incut, cnt


rows = []
for k in range(40):
    istep = jnp.int32(1000 + k)
    w, slots, alive, nnear, ninc, n_in, ndense_exit = first(st, istep)
    acts, incs, useful, it = [], [], 0, 0
    while True:
        w, a, u, ic, cnt = stage(w, slots, jnp.int32(it), istep)
        if int(a) == 0:
            break
        acts.append(int(a)); incs.append(int(ic)); useful += int(u); it += 1
    rows.append(dict(alive=int(alive), near=int(nnear), active=acts, active_in_cut=incs,
                     useful=useful, n_in=int(n_in)))
    st, _ = sim.run(st, 7, istep, consts, cfg, 1)
S = max(len(r["active"]) for r in rows)
pad = lambda key: np.array([r[key] + [0] * (S - len(r[key])) for r in rows])
out = dict(alive=np.mean([r["alive"] for r in rows]), near_mean=np.mean([r["near"] for r in rows]),
           near_max=int(np.max([r["near"] for r in rows])), inserts=np.mean([r["n_in"] for r in rows]),
           active_per_stage=pad("active").mean(0).tolist(), active_max_per_stage=pad("active").max(0).tolist(),
           active_in_cut_per_stage=pad("active_in_cut").mean(0).tolist(),
           line_tests_useful=np.mean([r["useful"] for r in rows]), K=cfg.K,
           lanes=[cfg.lane, cfg.lane2, cfg.lane3])
print(json.dumps(out, indent=1))
json.dump(out, open("/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct/walkstats2.json", "w"), indent=1)
