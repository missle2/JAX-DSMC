import functools

import jax
import jax.numpy as jnp

from sparta_jax import sim
from sparta_jax.sim import _starts, cumsum, materialise


def _partition_one_scatter(knew, stay, cs, state, size, consts, cfg, csfn=cumsum):
    N, C, M = cfg.capacity, cfg.ncell, size
    pos = jnp.arange(N, dtype=jnp.int32)
    slot = state.pkey & 0xFFFF
    mval = (
        jnp.full(M, (C + 1) * 65536, jnp.int32)
        .at[jnp.where(stay, M, pos - cs)]
        .set(knew * 65536 + slot, mode="drop")
    )
    cs_end = jnp.concatenate([cs, (cs[-1] + stay[-1]).astype(jnp.int32)[None]])
    nstay = cs_end[state.start[1:]] - cs_end[state.start[: C + 1]]
    hist = jnp.zeros(C + 2, jnp.int32).at[mval >> 16].add(1)[: C + 1]
    start = _starts(nstay + hist)
    offs = materialise(start[: C + 1] - _starts(nstay)[: C + 1])
    dest_stay = jnp.where(stay, offs[knew] + cs, N)
    ms = jnp.sort(mval)
    mk = ms >> 16
    ok = mk <= C
    mkc = jnp.minimum(mk, C)
    mfirst = _starts(hist)[: C + 1]
    base = materialise(start[: C + 1] + nstay - mfirst)
    dest_mov = jnp.where(ok, base[mkc] + jnp.arange(M, dtype=jnp.int32), N)
    pkey = (
        jnp.zeros(N, jnp.int32)
        .at[jnp.concatenate([dest_stay, dest_mov])]
        .set(jnp.concatenate([state.pkey, ms]), mode="drop")
    )
    mslot = jnp.where(ok, ms & 0xFFFF, N)
    scode = state.scode.at[mslot].set(consts.safe[mkc], mode="drop")
    return pkey, start, scode


def make_sort(partition, csfn=cumsum):
    def sort(state, consts, cfg):
        N = cfg.capacity
        slot = state.pkey & 0xFFFF
        knew = state.cell[slot]
        stay = knew == state.pkey >> 16
        stay_i = stay.astype(jnp.int32)
        cs = csfn(stay_i) - stay_i
        nmov = N - (cs[-1] + stay_i[-1])
        sizes = jnp.asarray(cfg.mover_sizes, jnp.int32)
        branch = jnp.sum(nmov > sizes, dtype=jnp.int32)
        branches = [functools.partial(partition, size=m, consts=consts, cfg=cfg) for m in cfg.mover_sizes]
        branches.append(functools.partial(sim._slow_sort, consts=consts, cfg=cfg))
        pkey, start, scode = jax.lax.switch(branch, branches, knew, stay, cs, state)
        return state._replace(pkey=pkey, start=start, scode=scode)
    return sort


def _jcumsum(x):
    return jnp.cumsum(x, dtype=jnp.int32)


VARIANTS = {
    "same_rebuilt": make_sort(sim._partition),
    "one_scatter": make_sort(_partition_one_scatter),
    "jnp_cumsum": make_sort(sim._partition, _jcumsum),
}


def _partition_search(knew, stay, cs, state, size, consts, cfg, stable=False):
    """Movers located by binary search on the stayer count (M searches of log N steps)
    instead of an N-wide compaction scatter; unstable sort of the unique packed keys."""
    N, C, M = cfg.capacity, cfg.ncell, size
    slot = state.pkey & 0xFFFF
    # mover rank r is at the first position p with (p + 1 - stayers through p) > r
    movers_through = jnp.arange(1, N + 1, dtype=jnp.int32) - (cs + stay.astype(jnp.int32))
    want = jnp.arange(1, M + 1, dtype=jnp.int32)
    mpos = jnp.searchsorted(movers_through, want, side="left", method="scan").astype(jnp.int32)
    have = mpos < N
    mp = jnp.minimum(mpos, N - 1)
    mval = jnp.where(have, knew[mp] * 65536 + slot[mp], (C + 1) * 65536)
    cs_end = jnp.concatenate([cs, (cs[-1] + stay[-1]).astype(jnp.int32)[None]])
    nstay = cs_end[state.start[1:]] - cs_end[state.start[: C + 1]]
    hist = jnp.zeros(C + 2, jnp.int32).at[mval >> 16].add(1)[: C + 1]
    start = _starts(nstay + hist)
    offs = materialise(start[: C + 1] - _starts(nstay)[: C + 1])
    dest_stay = jnp.where(stay, offs[knew] + cs, N)
    pkey = jnp.zeros(N, jnp.int32).at[dest_stay].set(state.pkey, mode="drop")
    ms = jax.lax.sort(mval, is_stable=stable)
    mk = ms >> 16
    ok = mk <= C
    mkc = jnp.minimum(mk, C)
    mfirst = _starts(hist)[: C + 1]
    base = materialise(start[: C + 1] + nstay - mfirst)
    dest_mov = jnp.where(ok, base[mkc] + jnp.arange(M, dtype=jnp.int32), N)
    pkey = pkey.at[dest_mov].set(ms, mode="drop")
    mslot = jnp.where(ok, ms & 0xFFFF, N)
    scode = state.scode.at[mslot].set(consts.safe[mkc], mode="drop")
    return pkey, start, scode


def _partition_unstable(knew, stay, cs, state, size, consts, cfg):
    N, C, M = cfg.capacity, cfg.ncell, size
    pos = jnp.arange(N, dtype=jnp.int32)
    slot = state.pkey & 0xFFFF
    mval = (
        jnp.full(M, (C + 1) * 65536, jnp.int32)
        .at[jnp.where(stay, M, pos - cs)]
        .set(knew * 65536 + slot, mode="drop")
    )
    cs_end = jnp.concatenate([cs, (cs[-1] + stay[-1]).astype(jnp.int32)[None]])
    nstay = cs_end[state.start[1:]] - cs_end[state.start[: C + 1]]
    hist = jnp.zeros(C + 2, jnp.int32).at[mval >> 16].add(1)[: C + 1]
    start = _starts(nstay + hist)
    offs = materialise(start[: C + 1] - _starts(nstay)[: C + 1])
    dest_stay = jnp.where(stay, offs[knew] + cs, N)
    pkey = jnp.zeros(N, jnp.int32).at[dest_stay].set(state.pkey, mode="drop")
    ms = jax.lax.sort(mval, is_stable=False)
    mk = ms >> 16
    ok = mk <= C
    mkc = jnp.minimum(mk, C)
    mfirst = _starts(hist)[: C + 1]
    base = materialise(start[: C + 1] + nstay - mfirst)
    dest_mov = jnp.where(ok, base[mkc] + jnp.arange(M, dtype=jnp.int32), N)
    pkey = pkey.at[dest_mov].set(ms, mode="drop")
    mslot = jnp.where(ok, ms & 0xFFFF, N)
    scode = state.scode.at[mslot].set(consts.safe[mkc], mode="drop")
    return pkey, start, scode


VARIANTS["search"] = make_sort(_partition_search)
VARIANTS["unstable"] = make_sort(_partition_unstable)
