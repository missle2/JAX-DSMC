"""3-D brick domain decomposition of ``sim3d`` over devices (and processes), with ``shard_map``.

Generalises main's ``jax_dsmc/decomp.py`` (a 1-D cut along x with two neighbours) to a
Px x Py x Pz brick layout, which the strong-scaling points need: 8M particles are ~93^3 cells,
which cannot be cut into 128 x-slabs.  The pieces kept from main: fixed per-device capacities
whose overflow is an error, not a warning (``overflow`` in the stats; the driver raises), the
compaction of arrivals before they are placed, and per-device random streams.

Per device ("rank"; one per GPU, one process per GPU in the cluster runs):

* it owns the cells of its brick (``Consts.offset``/``extent``; local cell index with the
  static padded strides ``cfg.bdim``) and the particles in them;
* geometry is replicated: the grid is implicit and the cut-cell tables are small, so the device
  finishes every move of its own particles, even one that walks into a neighbour's brick, and
  the particle then migrates to the owner of its end cell (``LEFT`` code carries the global
  cell, so the owner is exact on cell faces);
* migration: the leavers are compacted once (one N-sized prefix sum), then exchanged along x,
  then y, then z -- two ``ppermute`` per axis, fixed-size buffers -- so an edge or corner
  neighbour is reached in two or three hops (SPARTA's ``Comm::migrate_particles``, which sends
  directly; a move longer than a brick is counted in ``far``);
* departures push their slots onto the free stack (``State.ftop``), arrivals pop from it.

Load balance: the bricks are static here (uniform cuts).  The sphere deck's ``fix balance``
is not reproduced; the per-device particle counts are reported so that the imbalance can be
measured before a rebalancer is built.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import geometry3d as G3
from . import rng
from . import sim3d as S

AXIS = "dev"

try:  # jax >= 0.4.35
    from jax import shard_map as _shard_map

    def shard_map(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)

except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map

    def shard_map(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


# ---------------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Layout:
    """Px x Py x Pz bricks; rank r has coordinates (r % Px, (r // Px) % Py, r // (Px Py))."""

    P: tuple
    cuts: tuple  # per axis the cell boundaries, length P[a] + 1

    @property
    def nrank(self):
        return self.P[0] * self.P[1] * self.P[2]

    @property
    def bdim(self):
        return tuple(int(max(np.diff(self.cuts[a]))) for a in range(3))

    def coords(self, r):
        Px, Py, _ = self.P
        return (r % Px, (r // Px) % Py, r // (Px * Py))

    def offset(self, r):
        c = self.coords(r)
        return tuple(int(self.cuts[a][c[a]]) for a in range(3))

    def extent(self, r):
        c = self.coords(r)
        return tuple(int(self.cuts[a][c[a] + 1] - self.cuts[a][c[a]]) for a in range(3))

    def stride(self, a):
        return (1, self.P[0], self.P[0] * self.P[1])[a]

    def perm(self, a, up):
        """ppermute pairs sending to the neighbour along axis a (+1 if up, else -1)."""
        pairs = []
        for r in range(self.nrank):
            c = self.coords(r)[a]
            if up and c < self.P[a] - 1:
                pairs.append((r, r + self.stride(a)))
            if not up and c > 0:
                pairs.append((r, r - self.stride(a)))
        return pairs


def uniform_cuts(n, p):
    return tuple(int(round(k * n / p)) for k in range(p + 1))


def choose_layout(n, nrank, min_cells=3) -> Layout:
    """The factorisation of ``nrank`` into Px Py Pz with the least brick surface (cells on brick
    faces, the migration volume), every brick at least ``min_cells`` cells per axis."""
    best = None
    for px in range(1, nrank + 1):
        if nrank % px:
            continue
        for py in range(1, nrank // px + 1):
            if (nrank // px) % py:
                continue
            pz = nrank // (px * py)
            p = (px, py, pz)
            w = [n[a] / p[a] for a in range(3)]
            if min(w) < min_cells:
                continue
            surface = sum((p[a] - 1) * (n[(a + 1) % 3] * n[(a + 2) % 3]) for a in range(3))
            key = (surface, -min(w))
            if best is None or key < best[0]:
                best = (key, p)
    if best is None:
        raise ValueError(f"{nrank} ranks cannot split {n} with >= {min_cells} cells per brick")
    p = best[1]
    return Layout(P=p, cuts=tuple(uniform_cuts(n[a], p[a]) for a in range(3)))


# ---------------------------------------------------------------------------------------------
# per-rank host tables
# ---------------------------------------------------------------------------------------------


def _rank_cells(grid: G3.Grid, off, ext, bdim):
    return S.brick_cells(grid, off, ext, bdim)


def rank_tables(deck, geo: G3.Geometry3, layout: Layout, r):
    """Host arrays of rank r: local flow volume, clearance, emit tasks; and its particle count."""
    grid = geo.grid
    off, ext, bdim = layout.offset(r), layout.extent(r), layout.bdim
    gid, valid = _rank_cells(grid, off, ext, bdim)
    vol = np.where(valid, geo.flow_volume()[gid], 0.0)
    clear = None if geo.clearance is None else np.where(valid, geo.clearance[gid], -1).astype(np.int8)
    tasks = None
    if deck.emit:
        ijk = grid.unravel(geo.task_gid)
        inb = np.all((ijk >= np.array(off)) & (ijk < np.array(off) + np.array(ext)), axis=1)
        lij = ijk[inb] - np.array(off)
        lcell = lij[:, 0] + bdim[0] * (lij[:, 1] + bdim[1] * lij[:, 2])
        tasks = (lcell.astype(np.int32), geo.task_face[inb].astype(np.int32), geo.task_rate[inb])
    return {"gid": gid, "valid": valid, "volume": vol, "clear": clear, "tasks": tasks}


# ---------------------------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------------------------


def _owner_coord(g, cuts):
    """Brick coordinate along an axis of global cell index g (static cuts)."""
    c = jnp.asarray(np.asarray(cuts[1:-1], np.int32))
    return jnp.sum((g[:, None] >= c[None, :]).astype(jnp.int32), axis=1) if len(cuts) > 2 else jnp.zeros_like(g)


def _compact_rows(mask, size):
    """Indices of the first ``size`` set entries (len(mask) where there are fewer), overflow."""
    n = mask.shape[0]
    r = jnp.cumsum(mask.astype(jnp.int32)) - 1
    idx = (
        jnp.full(size, n, jnp.int32)
        .at[jnp.where(mask & (r < size), r, size)]
        .set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )
    return idx, jnp.maximum(jnp.sum(mask.astype(jnp.int32), dtype=jnp.int32) - size, 0)


def migrate(state: S.State, coords, layout: Layout, L: int, M: int, consts: S.Consts, cfg: S.Config):
    """Move every LEFT particle to the device that owns its cell.  Inside ``shard_map``.

    Buffers: ``L`` leavers per device, ``M`` per direction per axis.  Rows are
    (x, y, z, vx, vy, vz, gid, valid) in float64 (gid < 2^53 exactly)."""
    N, Cl = cfg.capacity, cfg.ncell_local
    left = state.cell >= cfg.LEFT0
    idx, over_l = _compact_rows(left, L)
    ok = idx < N
    g = jnp.minimum(idx, N - 1)
    gid = jnp.where(ok, state.cell[g] - cfg.LEFT0, 0)
    rows = jnp.stack(
        [state.x[g], state.y[g], state.z[g], state.vx[g], state.vy[g], state.vz[g], gid.astype(jnp.float64), ok.astype(jnp.float64)],
        axis=1,
    )
    rows = jnp.where(ok[:, None], rows, 0.0)
    n_dep = jnp.sum(ok.astype(jnp.int32), dtype=jnp.int32)
    # the leavers' slots die and go on the free stack (all leavers: an overflowing one is lost)
    cell = state.cell.at[jnp.where(left, jnp.arange(N, dtype=jnp.int32), N)].set(cfg.DEAD, mode="drop")
    ftop = state.ftop - n_dep
    order = state.order.at[jnp.where(ok, ftop + jnp.arange(L, dtype=jnp.int32), N)].set(idx, mode="drop")
    state = state._replace(cell=cell, order=order, ftop=ftop)
    over = over_l
    far = jnp.int32(0)
    for a in range(3):
        if layout.P[a] == 1:
            continue
        valid = rows[:, 7] > 0.5
        ga = S._unravel_gid(rows[:, 6].astype(jnp.int32), cfg)[a]
        d = _owner_coord(ga, layout.cuts[a]) - coords[a]
        far = far + jnp.sum((valid & (jnp.abs(d) > 1)).astype(jnp.int32), dtype=jnp.int32)
        up = valid & (d > 0)
        down = valid & (d < 0)
        iu, ou = _compact_rows(up, M)
        idn, od = _compact_rows(down, M)
        send_u = jnp.take(rows, iu, axis=0, mode="fill", fill_value=0.0)
        send_d = jnp.take(rows, idn, axis=0, mode="fill", fill_value=0.0)
        recv_lo = jax.lax.ppermute(send_u, AXIS, layout.perm(a, True))  # from the lower neighbour
        recv_hi = jax.lax.ppermute(send_d, AXIS, layout.perm(a, False))
        stay = jnp.where((valid & (d == 0))[:, None], rows, 0.0)
        allr = jnp.concatenate([stay, recv_lo, recv_hi], axis=0)
        keep, ok_ = _compact_rows(allr[:, 7] > 0.5, L)
        rows = jnp.take(allr, keep, axis=0, mode="fill", fill_value=0.0)
        over = over + ou + od + ok_
    # arrivals: every remaining row is owned here
    valid = rows[:, 7] > 0.5
    n_arr = jnp.sum(valid.astype(jnp.int32), dtype=jnp.int32)
    g3 = S._unravel_gid(rows[:, 6].astype(jnp.int32), cfg)
    lc, inb = S._local_index(g3, consts, cfg)
    wrong = jnp.sum((valid & ~inb).astype(jnp.int32), dtype=jnp.int32)  # far jumps that did not arrive
    q = jnp.arange(L, dtype=jnp.int32)
    pos = state.ftop + q
    okw = valid & inb & (pos < N)
    slot = jnp.where(okw, state.order[jnp.minimum(pos, N - 1)], N)
    put = functools.partial(S._put, slot)
    state = state._replace(
        x=put(state.x, rows[:, 0]),
        y=put(state.y, rows[:, 1]),
        z=put(state.z, rows[:, 2]),
        vx=put(state.vx, rows[:, 3]),
        vy=put(state.vy, rows[:, 4]),
        vz=put(state.vz, rows[:, 5]),
        cell=put(state.cell, lc.astype(jnp.int32)),
        ftop=state.ftop + jnp.sum(okw.astype(jnp.int32), dtype=jnp.int32),
    )
    lost = over + wrong + jnp.sum((valid & inb & (pos >= N)).astype(jnp.int32), dtype=jnp.int32)
    return state, {"overflow": lost, "sent": n_dep, "received": jnp.sum(okw.astype(jnp.int32), dtype=jnp.int32), "far": far, "arrived": n_arr}


# ---------------------------------------------------------------------------------------------
# the decomposed run
# ---------------------------------------------------------------------------------------------

RANK_FIELDS = ("offset", "extent", "volume", "clear", "task_lcell", "task_face", "task_rate")
STATE_SCALARS = ("ftop",)


@dataclasses.dataclass
class Decomp:
    deck: object
    geo: G3.Geometry3
    layout: Layout
    cfg: S.Config
    consts: S.Consts  # per-rank fields stacked along a leading rank axis, others replicated
    mesh: Mesh
    L: int
    M: int
    host: dict  # per-rank host summaries (counts)


def _devices(nrank):
    devs = jax.devices()
    if len(devs) < nrank:
        raise ValueError(f"{nrank} ranks but {len(devs)} devices")
    return np.array(devs[:nrank])


def build(deck, nrank, geo=None, layout=None, headroom=None, **cfg_overrides) -> Decomp:
    """Static configuration, per-rank constants and the device mesh of a decomposed run."""
    geo = geo or G3.build(deck)
    grid = geo.grid
    layout = layout or choose_layout(grid.n, nrank)
    bdim = layout.bdim
    counts = G3.create_counts(geo, geo.n_create, seed=0)
    tables = [rank_tables(deck, geo, layout, r) for r in range(nrank)]
    n_rank = np.array([int(counts[t["gid"][t["valid"]]].sum()) for t in tables])
    if headroom is None:
        headroom = 1.5 if deck.emit else 1.06
    cap = int(n_rank.max() * headroom + 8 * math.sqrt(n_rank.max() + 1) + 1024)
    cap = S._round_up(cap, 1024)
    # migration buffers from the largest per-step move across a brick face
    dt = max(r.dt for r in deck.runs)
    vscale = geo.vscale
    frac = [
        (abs(deck.vstream[a]) + 2.0 * vscale) * dt / (grid.h[a] * bdim[a]) if layout.P[a] > 1 else 0.0
        for a in range(3)
    ]
    L = S._round_up(int(cap * min(1.0, 2.0 * sum(frac)) + 1024), 256)
    M = S._round_up(int(cap * min(1.0, 2.0 * max(frac + [0.0])) + 512), 256)
    M = min(M, L)
    # uniform static config: from one brick, then the maxima over ranks
    T = max((0 if t["tasks"] is None else len(t["tasks"][0])) for t in tables)
    cfg, consts1, _ = S.make(deck, geo, capacity=cap, brick=(layout.offset(0), bdim), **cfg_overrides)
    if deck.emit:
        per_task = int(math.floor(float(geo.task_rate.max()) * dt)) + 1
        ins = max(float(t["tasks"][2].sum()) * dt for t in tables)
        ins_cap = int(ins + 10.0 * math.sqrt(ins + 1.0) + 256)
        cfg = dataclasses.replace(cfg, ntask=max(T, 1), per_task=per_task, ins_cap=ins_cap)
    devices = _devices(nrank)
    mesh = Mesh(devices, (AXIS,))
    rank_sh = NamedSharding(mesh, P(AXIS))
    repl = NamedSharding(mesh, P())

    def stacked(fn, dtype):
        arrs = [np.asarray(fn(r), dtype) for r in range(nrank)]
        shape = (nrank,) + arrs[0].shape
        return jax.make_array_from_callback(shape, rank_sh, lambda idx: np.stack(arrs)[idx])

    def pad_tasks(r, k, fill):
        t = tables[r]["tasks"]
        a = np.full(cfg.ntask, fill, (np.int32, np.int32, np.float64)[k])
        if t is not None:
            a[: len(t[k])] = t[k]
        return a

    consts = S.Consts(
        offset=stacked(layout.offset, np.int32),
        extent=stacked(layout.extent, np.int32),
        volume=stacked(lambda r: tables[r]["volume"], np.float64),
        clear=None if geo.clearance is None else stacked(lambda r: tables[r]["clear"], np.int8),
        cut_gid=None if consts1.cut_gid is None else jax.device_put(consts1.cut_gid, repl),
        cut_tri=None if consts1.cut_tri is None else jax.device_put(consts1.cut_tri, repl),
        cut_ids=None if consts1.cut_ids is None else jax.device_put(consts1.cut_ids, repl),
        cut_nsurf=None if consts1.cut_nsurf is None else jax.device_put(consts1.cut_nsurf, repl),
        task_lcell=None if not deck.emit else stacked(lambda r: pad_tasks(r, 0, 0), np.int32),
        task_face=None if not deck.emit else stacked(lambda r: pad_tasks(r, 1, 0), np.int32),
        task_rate=None if not deck.emit else stacked(lambda r: pad_tasks(r, 2, 0.0), np.float64),
    )
    host = {"n_rank": n_rank, "capacity": cap, "L": L, "M": M, "layout": layout.P}
    return Decomp(deck=deck, geo=geo, layout=layout, cfg=cfg, consts=consts, mesh=mesh, L=L, M=M, host=host)


def _consts_spec(consts: S.Consts):
    return S.Consts(
        **{
            f: (None if getattr(consts, f) is None else (P(AXIS) if f in RANK_FIELDS else P()))
            for f in S.Consts._fields
        }
    )


def _local_consts(c: S.Consts):
    return c._replace(**{f: (None if getattr(c, f) is None else getattr(c, f)[0]) for f in RANK_FIELDS})


def _state_spec():
    return S.State(**{f: P(AXIS) for f in S.State._fields})


def _local_state(st: S.State):
    return st._replace(ftop=st.ftop[0])


def _global_state(st: S.State):
    return st._replace(ftop=st.ftop[None])


def create_particles(dec: Decomp, seed):
    """Every rank creates the particles of its brick on its own device; assembled into one
    global (rank-sharded) State."""
    cfg, geo = dec.cfg, dec.geo
    sharding = NamedSharding(dec.mesh, P(AXIS))
    per_rank = {}
    for r, dev in enumerate(dec.mesh.devices.ravel()):
        if dev.process_index != jax.process_index():
            continue
        with jax.default_device(dev):
            c_r = jax.tree.map(lambda a: a, _rank_consts_host(dec, r))
            st = S.create_particles(cfg, c_r, geo, dec.deck.temp, dec.deck.vstream, seed)
        per_rank[r] = jax.device_put(st, dev)
    fields = {}
    for f in S.State._fields:
        shards = [per_rank[r] for r in sorted(per_rank)]
        arrs = [jnp.atleast_1d(getattr(s, f)) for s in shards]
        shape = (arrs[0].shape[0] * dec.layout.nrank,) + arrs[0].shape[1:]
        fields[f] = jax.make_array_from_single_device_arrays(shape, sharding, arrs)
    return S.State(**fields)


def _rank_consts_host(dec: Decomp, r):
    """Rank r's constants as single-device arrays (for creation)."""
    c = dec.consts

    def pick(a, f):
        if a is None:
            return None
        if f in RANK_FIELDS:
            return jnp.asarray(np.asarray(a.addressable_data(_shard_pos(a, r)))[0])
        return jnp.asarray(np.asarray(a.addressable_data(0)))

    return S.Consts(**{f: pick(getattr(c, f), f) for f in S.Consts._fields})


def _shard_pos(arr, r):
    for i, s in enumerate(arr.addressable_shards):
        if (s.index[0].start or 0) == r:
            return i
    raise KeyError(r)


def _inner_run(state, consts, seeds, step0, dt, cfg, layout, L, M, nsteps):
    st = _local_state(state)
    c = _local_consts(consts)
    rank = jax.lax.axis_index(AXIS)
    coords = (rank % layout.P[0], (rank // layout.P[0]) % layout.P[1], rank // (layout.P[0] * layout.P[1]))
    seed = seeds[0]
    mig = functools.partial(migrate, coords=coords, layout=layout, L=L, M=M, consts=c, cfg=cfg)
    body = functools.partial(S.step, dt=dt, seed=seed, consts=c, cfg=cfg, migrate=mig)
    steps = step0 + 1 + jnp.arange(nsteps, dtype=jnp.int32)
    st, stats = jax.lax.scan(lambda s, i: body(s, i), st, steps)
    stats = jax.tree.map(lambda a: a[:, None], stats)  # (nsteps, 1) per rank
    return _global_state(st), stats


@functools.lru_cache(maxsize=None)
def _run_fn(cfg, layout, L, M, nsteps, mesh, consts_spec):
    inner = functools.partial(_inner_run, cfg=cfg, layout=layout, L=L, M=M, nsteps=nsteps)
    stats_spec = S.Stats(**{f: P(None, AXIS) for f in S.Stats._fields})
    f = shard_map(
        inner,
        mesh=mesh,
        in_specs=(_state_spec(), consts_spec, P(AXIS), P(), P()),
        out_specs=(_state_spec(), stats_spec),
    )
    return jax.jit(f, donate_argnums=(0,))


def _spec_key(consts):
    return S.Consts(*[None if v is None else (P(AXIS) if f in RANK_FIELDS else P()) for f, v in zip(S.Consts._fields, consts)])


def run(dec: Decomp, state, step0, dt, seeds, nsteps):
    """``nsteps`` steps of every rank; returns (state, Stats with a trailing rank axis)."""
    fn = _run_fn(dec.cfg, dec.layout, dec.L, dec.M, nsteps, dec.mesh, _spec_key(dec.consts))
    return fn(state, dec.consts, seeds, jnp.asarray(step0, jnp.int32), jnp.asarray(dt, jnp.float64))


def rank_seeds(dec: Decomp, seed):
    """Independent per-rank stream seeds (rng.mix64 of (seed, rank))."""
    s = np.array([int(rng.mix64(np.uint64(seed) * rng.GOLDEN + np.uint64(r + 1))) for r in range(dec.layout.nrank)], np.uint64)
    return jax.make_array_from_callback((dec.layout.nrank,), NamedSharding(dec.mesh, P(AXIS)), lambda idx: s[idx])


def _inner_diag(state, cfg):
    st = _local_state(state)
    Cl = cfg.ncell_local
    alive = st.cell < Cl
    v2 = jnp.where(alive, st.vx**2 + st.vy**2 + st.vz**2, 0.0)
    n = jnp.sum(alive.astype(jnp.int32), dtype=jnp.int32)
    npc = st.start[1 : Cl + 1] - st.start[:Cl]
    out = {
        "np": jax.lax.psum(n, AXIS),
        "sum_v2": jax.lax.psum(jnp.sum(v2), AXIS),
        "maxpercell": jax.lax.pmax(jnp.max(npc), AXIS),
        "np_rank_max": jax.lax.pmax(n, AXIS),
    }
    return out


@functools.lru_cache(maxsize=None)
def _diag_fn(cfg, mesh):
    f = shard_map(
        functools.partial(_inner_diag, cfg=cfg),
        mesh=mesh,
        in_specs=(_state_spec(),),
        out_specs={"np": P(), "sum_v2": P(), "maxpercell": P(), "np_rank_max": P()},
    )
    return jax.jit(f)


def diagnostics(dec: Decomp, state):
    d = _diag_fn(dec.cfg, dec.mesh)(state)
    n = d["np"]
    temp = dec.cfg.mass * d["sum_v2"] / (3.0 * G3.BOLTZ * jnp.maximum(n, 1))
    return {"np": n, "temp": temp, "maxpercell": d["maxpercell"], "np_rank_max": d["np_rank_max"]}


@functools.lru_cache(maxsize=None)
def _reset_fn(cfg, mesh):
    def inner(state):
        st = _local_state(state)
        return _global_state(S.reset_vremax(st, cfg))

    return jax.jit(shard_map(inner, mesh=mesh, in_specs=(_state_spec(),), out_specs=_state_spec()))


def reset_vremax(dec: Decomp, state):
    return _reset_fn(dec.cfg, dec.mesh)(state)


def ranks_of(nrank):
    """All layouts with this many ranks (for tests)."""
    return [p for p in itertools.product(range(1, nrank + 1), repeat=3) if p[0] * p[1] * p[2] == nrank]
