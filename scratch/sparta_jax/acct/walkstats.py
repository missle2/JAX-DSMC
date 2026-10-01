"""Work counts of the sparta_jax move at steady state: alive, near (lane entries), walk stages,
active entries per stage, line tests (useful = nsurf of the cell per active entry)."""
import functools
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
from sparta_jax import sim  # noqa: E402

cfg, consts, geo = sim.make()
st = sim.empty_state(cfg)
st, _ = sim.run(st, 7, jnp.int32(0), consts, cfg, 1000)


@jax.jit
def first(state, istep):
    state, slot, *_ = sim.emit(state, jnp.uint64(7), istep, consts, cfg)
    new, near, _, _ = sim.dense_move(state, consts, cfg)
    rank = jnp.cumsum(near.astype(jnp.int32)) - 1
    N, B = cfg.capacity, 8192
    slots = jnp.full(B, N, jnp.int32).at[jnp.where(near, rank, B)].set(jnp.arange(N, dtype=jnp.int32), mode="drop")
    valid = slots < N
    g = jnp.minimum(slots, N - 1)
    x, y, vx, vy, dtr = state.x[g], state.y[g], state.vx[g], state.vy[g], state.dtr[g]
    w = sim.Walk(x=x, y=y, vx=vx, vy=vy, vz=state.vz[g], xn=x + dtr * vx, yn=y + dtr * vy,
                 dtrem=dtr, icell=jnp.where(valid, state.cell[g], 0),
                 exclude=jnp.full(B, -1, jnp.int32), stuck=jnp.zeros(B, jnp.int32),
                 status=jnp.where(valid, sim.ACTIVE, sim.DONE).astype(jnp.int32), slot=slots)
    alive = jnp.sum(state.cell < cfg.ncell)
    ninc = jnp.sum((state.cell < cfg.ncell) & (consts.nsurf[jnp.minimum(state.cell, cfg.ncell - 1)] > 0))
    return w, alive, jnp.sum(near), ninc, slot


@jax.jit
def stage(w, it, istep):
    act = w.status == sim.ACTIVE
    useful = jnp.sum(jnp.where(act, consts.nsurf[w.icell], 0))
    incut = jnp.sum(act & (consts.nsurf[w.icell] > 0))
    w2, cnt, _ = sim._walk_stage(w, jnp.uint64(7), istep, it, jnp.zeros((cfg.nlines, 3)), consts, cfg)
    return w2, jnp.sum(act), useful, incut, cnt


rows = []
for k in range(40):
    istep = jnp.int32(1000 + k)
    w, alive, nnear, ninc, _ = first(st, istep)
    act_per_stage, useful, incut = [], 0, []
    it = 0
    while True:
        w, a, u, ic, cnt = stage(w, jnp.int32(it), istep)
        a = int(a)
        if a == 0:
            break
        act_per_stage.append(a)
        incut.append(int(ic))
        useful += int(u)
        it += 1
    rows.append(dict(alive=int(alive), near=int(nnear), in_cut=int(ninc), active=act_per_stage,
                     active_in_cut=incut, line_tests_useful=useful))
    st, _ = sim.run(st, 7, istep, consts, cfg, 1)
alive = np.mean([r["alive"] for r in rows])
near = np.mean([r["near"] for r in rows])
S = max(len(r["active"]) for r in rows)
act = np.array([r["active"] + [0] * (S - len(r["active"])) for r in rows]).mean(0)
inc = np.array([r["active_in_cut"] + [0] * (S - len(r["active_in_cut"])) for r in rows]).mean(0)
out = dict(alive=alive, near=near, in_cut_cells=np.mean([r["in_cut"] for r in rows]),
           active_per_stage=act.tolist(), active_in_cut_per_stage=inc.tolist(),
           stages_mean=np.mean([len(r["active"]) for r in rows]),
           stages_max=S, line_tests_useful=np.mean([r["line_tests_useful"] for r in rows]),
           K=cfg.K, lane=cfg.lane, lane2=cfg.lane2, fixed_stages=cfg.fixed_stages)
print(json.dumps(out, indent=1))
json.dump(out, open("/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct/walkstats.json", "w"), indent=1)
