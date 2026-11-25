"""SPARTA's per-step algorithm for its 3-D benchmark decks (in.free, in.collide, in.sphere), in JAX.

Forward only, float64, not differentiable.  It follows ``sim.py`` (the validated 2-D circle
port) and the dimension-agnostic pieces of main (grid indexing, per-face boundaries, emit per
face, the cell walk); see ``sparta_jax/README.md``.  One step, in SPARTA's order
(``Update::run``):

1. collide_modify vremax: reset vremax/remain every ``vre_every`` steps;
2. move -- ``Update::move<3,0,0>`` -- split in three:
   a. the dense straight move of every particle whose segment provably meets no surface: its
      end cell lies inside the cut-free Chebyshev ring of its start cell (``clear``).  Global
      boundaries: OUTFLOW deletes; REFLECT mirrors (only in decks without surfaces, where a
      mirrored path cannot reach one);
   b. emit -- ``FixEmitFace::perform_task_onepass`` (start_of_step fix): new particles on the
      boundary faces with ``dtremain = dt U``, advanced by their own dtremain in the same way;
   c. SPARTA's cell-by-cell walk (``line_tri_intersect``, ``SurfCollideDiffuse``,
      ``Domain::collide``) for the rest, on a compacted lane;
3. sort (if collisions): per-cell slot lists, ``order`` + ``start``;
4. collide: NTC with per-cell vremax and remain (``CollideVSS``).

Layout: SoA, fixed capacity ``N``; slots never move.  ``cell`` holds the LOCAL cell index of
the device's brick (``Brick``; the whole grid on one device), or a code:
``DEAD = Cl``, ``NEAR = Cl + 1 + c`` (start cell c, awaiting the walk) and
``LEFT = 2 Cl + 1 + gid`` (moved out of the brick into global cell gid: migrated by
``decomp3d``).  ``order`` is the slot list sorted by
cell with dead slots last; its tail is the free list emit writes into.

Particles carry no id, species (one species) or internal energy (argon), as in the decks.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry3d as G3
from . import rng

EPSSQNEG = -1.0e-16
MAXSTUCK = 20
# walk status
DONE, ACTIVE, DEADW = 0, 1, 2
# outface (SPARTA order XLO XHI YLO YHI ZLO ZHI)
INTERIOR = -1
# RNG streams (rng.step_key(seed, step, stream))
S_NINSERT, S_POS, S_BETA, S_ACCEPT, S_THETA, S_VR, S_DT = range(7)
S_SURF, S_DIFF1, S_DIFF2, S_DIFF3, S_DIFF4 = range(10, 15)
S_COLL = 20


@dataclasses.dataclass(frozen=True)
class Config:
    """Static configuration (hashable; a jit static argument)."""

    n: tuple  # global cells per axis
    lo: tuple
    hi: tuple
    h: tuple
    bdim: tuple  # local brick cells per axis (static); == n on one device
    boundary: str  # "r" | "o"
    mass: float
    vscale: float
    vstream: tuple
    fnum: float
    collide: bool
    omega: float
    alpha: float
    prefactor: float
    vremax0: float
    vre_every: int
    has_surf: bool
    K: int  # max triangles per cut cell
    ncut: int
    acc: float
    tsurf: float
    emit: bool
    ntask: int
    per_task: int  # max insertions per task per step
    ins_cap: int  # max insertions per step (compacted buffer)
    trials: int = 32  # vectorised accept-reject trials of the emit normal-velocity sampler
    capacity: int = 1 << 16
    lane: int = 4096  # cell-walk lane, first stage (chunked if more particles are near)
    lane2: int = 1024
    lane3: int = 128
    max_walk: int = 1000
    sort: str = "pack"  # "pack": one int64 key (cell << 32 | slot); "full": (cell, slot) pairs
    starts: str = "scatter"  # bucket starts: "scatter" (histogram + prefix sum) | "search" (binary search)
    # "all": attempt k on every cell (masked) | "prefix": only on the cells with > k attempts
    # (cells ordered by attempt count; a lax.switch over static prefix sizes).  Same results.
    collide_mode: str = "all"
    collide_unroll: int = 0  # attempts run unconditionally before the while_loop (no predicate)

    @property
    def ncell_local(self):
        return self.bdim[0] * self.bdim[1] * self.bdim[2]

    @property
    def DEAD(self):
        return self.ncell_local

    @property
    def NEAR0(self):
        return self.ncell_local + 1

    @property
    def LEFT0(self):
        return 2 * self.ncell_local + 1


class Consts(NamedTuple):
    """Traced constants.  Per-cell arrays are LOCAL (brick) cells; geometry is global."""

    offset: jax.Array  # (3,) int32 global index of the brick's first cell
    extent: jax.Array  # (3,) int32 cells of the brick per axis (<= cfg.bdim: padded strides)
    volume: jax.Array  # (Cl,) flow volume
    clear: jax.Array | None  # (Cl,) int8 cut-free Chebyshev radius (has_surf)
    cut_gid: jax.Array | None  # (M,) int32 sorted global ids of cut cells
    cut_tri: jax.Array | None  # (M, K, 12) p1 p2 p3 norm of each of the cell's triangles
    cut_ids: jax.Array | None  # (M, K) int32 triangle ids, -1 padded
    cut_nsurf: jax.Array | None  # (M,) int32
    task_lcell: jax.Array | None  # (T,) local cell of each emit task
    task_face: jax.Array | None  # (T,) int32 face id
    task_rate: jax.Array | None  # (T,) ntarget / dt


class State(NamedTuple):
    x: jax.Array
    y: jax.Array
    z: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array
    cell: jax.Array  # int32 local cell or code (module docstring)
    order: jax.Array  # (N,) int32 slots sorted by cell, dead last
    start: jax.Array  # (Cl + 2,) int32 bucket starts; start[Cl] = alive count at the sort
    ftop: jax.Array  # () int32: order[ftop:] are free slots (a stack: emit/arrivals pop, departures push)
    vremax: jax.Array  # (Cl,)
    remain: jax.Array  # (Cl,)


class Stats(NamedTuple):
    np: jax.Array
    nattempt: jax.Array
    ncoll: jax.Array
    nscoll: jax.Array
    nscheck: jax.Array
    nexit: jax.Array
    nbound: jax.Array
    ninsert: jax.Array
    nlane: jax.Array
    nwalk: jax.Array
    nstuck: jax.Array
    emit_fail: jax.Array
    overflow: jax.Array  # capacity or buffer exceeded (particles lost) -- must stay 0


# ---------------------------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------------------------


def _vss(sp: G3.Species):
    mr = sp.mass * sp.mass / (sp.mass + sp.mass)
    cxs = sp.diam * sp.diam * math.pi
    prefactor = cxs * (2.0 * G3.BOLTZ * sp.tref / mr) ** (sp.omega - 0.5) / math.gamma(2.5 - sp.omega)
    return cxs, prefactor


def make(deck, geo: G3.Geometry3 | None = None, capacity=None, brick=None, lane_scale=1.0, **overrides):
    """(Config, Consts, Geometry3) for a deck on one device (``brick=None``: the whole grid) or
    for one brick ``(offset (3,), bdim (3,))`` of a decomposition (``decomp3d``)."""
    geo = geo or G3.build(deck)
    grid = geo.grid
    sp = deck.species
    cxs, prefactor = _vss(sp)
    offset, bdim = (np.zeros(3, np.int64), tuple(grid.n)) if brick is None else brick
    offset = np.asarray(offset, np.int64)
    gid_local = _brick_gids(grid, offset, bdim)
    vol_all = geo.flow_volume()
    has_surf = geo.cut is not None
    dt_max = max(r.dt for r in deck.runs)
    # emit tasks of this brick
    if deck.emit:
        in_brick = np.isin(geo.task_gid, gid_local)
        t_gid = geo.task_gid[in_brick]
        t_face = geo.task_face[in_brick].astype(np.int32)
        t_rate = geo.task_rate[in_brick]
        t_lcell = _global_to_local(grid, t_gid, offset, bdim)
        mean = float(t_rate.sum() * dt_max)
        per_task = int(math.floor(float(t_rate.max(initial=0.0)) * dt_max)) + 1
        ins_cap = int(min(len(t_gid) * per_task, mean + 10.0 * math.sqrt(mean + 1.0) + 256))
        ins_cap = max(ins_cap, 1)
    else:
        t_gid = t_face = t_rate = t_lcell = None
        per_task = ins_cap = 0
    n_local = int(np.prod(bdim))
    if capacity is None:
        capacity = _default_capacity(geo, deck, gid_local)
    cfg = Config(
        n=tuple(int(v) for v in grid.n),
        lo=tuple(float(v) for v in grid.lo),
        hi=tuple(float(v) for v in grid.hi),
        h=tuple(float(v) for v in grid.h),
        bdim=tuple(int(v) for v in bdim),
        boundary=deck.boundary,
        mass=sp.mass,
        vscale=geo.vscale,
        vstream=tuple(float(v) for v in deck.vstream),
        fnum=deck.fnum,
        collide=deck.collide,
        omega=sp.omega,
        alpha=sp.alpha,
        prefactor=prefactor,
        vremax0=2.0 * cxs * geo.vscale,
        vre_every=deck.vre_every,
        has_surf=has_surf,
        K=int(geo.cut.csurfs.shape[1]) if has_surf else 0,
        ncut=int(len(geo.cut.gid)) if has_surf else 0,
        acc=deck.acc,
        tsurf=deck.tsurf,
        emit=deck.emit,
        ntask=0 if t_gid is None else len(t_gid),
        per_task=per_task,
        ins_cap=ins_cap,
        capacity=int(capacity),
        # walk lanes: ~1 % of the slots in stage 1 (a near-surface share above that just runs
        # more lane chunks), then 1/4 and 1/32 of it for the re-compacted later stages
        lane=_round_up(max(1024, capacity // 96), 256) if has_surf else 256,
        lane2=_round_up(max(256, capacity // 384), 128) if has_surf else 128,
        lane3=_round_up(max(64, capacity // 3072), 64) if has_surf else 64,
    )
    if lane_scale != 1.0:
        cfg = dataclasses.replace(
            cfg,
            lane=_round_up(cfg.lane * lane_scale, 256),
            lane2=_round_up(cfg.lane2 * lane_scale, 128),
            lane3=_round_up(cfg.lane3 * lane_scale, 64),
        )
    cfg = dataclasses.replace(cfg, **overrides)
    if has_surf:
        cut = geo.cut
        ids = cut.csurfs
        s = np.maximum(ids, 0)
        S = geo.surf
        tri = np.concatenate([S.p1[s], S.p2[s], S.p3[s], S.norm[s]], axis=2)  # (M, K, 12)
        tri = np.where((ids >= 0)[:, :, None], tri, 0.0)
        cut_gid = jnp.asarray(cut.gid, jnp.int32)
        cut_tri = jnp.asarray(tri)
        cut_ids = jnp.asarray(ids, jnp.int32)
        cut_nsurf = jnp.asarray(cut.nsurf, jnp.int32)
        clear = jnp.asarray(geo.clearance[gid_local], jnp.int8)
    else:
        cut_gid = cut_tri = cut_ids = cut_nsurf = clear = None
    consts = Consts(
        offset=jnp.asarray(offset, jnp.int32),
        extent=jnp.asarray(bdim, jnp.int32),
        volume=jnp.asarray(vol_all[gid_local]),
        clear=clear,
        cut_gid=cut_gid,
        cut_tri=cut_tri,
        cut_ids=cut_ids,
        cut_nsurf=cut_nsurf,
        task_lcell=None if t_lcell is None else jnp.asarray(t_lcell, jnp.int32),
        task_face=None if t_face is None else jnp.asarray(t_face, jnp.int32),
        task_rate=None if t_rate is None else jnp.asarray(t_rate),
    )
    assert 2 * n_local + 1 + grid.ncell < 2**31, "int32 cell codes"
    return cfg, consts, geo


def _round_up(v, m):
    return int((v + m - 1) // m * m)


def brick_cells(grid: G3.Grid, offset, extent, bdim):
    """Global id of every local cell (strides ``bdim``, i fastest) and whether it lies inside
    the brick's ``extent`` (padding cells beyond it get gid 0 and valid False)."""
    nx, ny, _ = grid.n
    i = np.arange(bdim[0])[None, None, :]
    j = np.arange(bdim[1])[None, :, None]
    k = np.arange(bdim[2])[:, None, None]
    valid = (i < extent[0]) & (j < extent[1]) & (k < extent[2])
    gid = (i + offset[0]) + nx * ((j + offset[1]) + ny * (k + offset[2]))
    return np.where(valid, gid, 0).ravel(), np.broadcast_to(valid, gid.shape).ravel()


def _brick_gids(grid: G3.Grid, offset, bdim):
    return brick_cells(grid, offset, bdim, bdim)[0]


def _global_to_local(grid: G3.Grid, gid, offset, bdim):
    ijk = grid.unravel(gid) - np.asarray(offset)[None, :]
    return ijk[:, 0] + bdim[0] * (ijk[:, 1] + bdim[1] * ijk[:, 2])


def _default_capacity(geo, deck, gid_local):
    """Particle slots: the expected count in the brick x 1.25 (+ emit headroom), rounded."""
    vol = geo.flow_volume()
    frac = float(vol[gid_local].sum() / vol.sum())
    n = geo.n_create * frac
    head = 1.25 if deck.emit else 1.0
    return _round_up(int(n * head + 8 * math.sqrt(n + 1) + 1024), 1024)


def empty_state(cfg: Config) -> State:
    N, Cl = cfg.capacity, cfg.ncell_local
    z = jnp.zeros(N)
    return State(
        x=z,
        y=z,
        z=z,
        vx=z,
        vy=z,
        vz=z,
        cell=jnp.full(N, cfg.DEAD, jnp.int32),
        order=jnp.arange(N, dtype=jnp.int32),
        start=jnp.zeros(Cl + 2, jnp.int32).at[Cl + 1].set(N),
        ftop=jnp.int32(0),
        vremax=jnp.full(Cl, cfg.vremax0),
        remain=jnp.zeros(Cl),
    )


def create_particles(cfg: Config, consts: Consts, geo: G3.Geometry3, temp, vstream, seed):
    """CreateParticles: per-cell counts (``geometry3d.create_counts``), uniform positions in the
    cell, rejected inside the body in cut cells (``Grid::outside_surfs`` with the cell's flow
    point), Maxwellian velocities as SPARTA draws them (vn cos theta1, vr cos/sin theta2).
    Returns a sorted State.  Runs on the device; the counts come from the host."""
    gid_local, valid = brick_cells(geo.grid, np.asarray(consts.offset), np.asarray(consts.extent), cfg.bdim)
    counts_all = G3.create_counts(geo, geo.n_create, seed=seed)
    counts = np.where(valid, counts_all[gid_local], 0)
    n = int(counts.sum())
    if n > cfg.capacity:
        raise ValueError(f"capacity {cfg.capacity} < {n} particles")
    state = empty_state(cfg)
    cell = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    cut_part = cut_row = None
    if cfg.has_surf:
        row = np.searchsorted(geo.cut.gid, gid_local)
        row = np.minimum(row, len(geo.cut.gid) - 1)
        is_cut = geo.cut.gid[row] == gid_local
        cut_part = np.nonzero(is_cut[cell])[0]
        cut_row = row[cell[cut_part]]
    vscale = math.sqrt(2.0 * G3.BOLTZ * temp / cfg.mass)
    x, v = _create_device(
        jnp.asarray(cell),
        consts,
        None if cut_part is None else jnp.asarray(cut_part, jnp.int32),
        None if cut_row is None else jnp.asarray(cut_row, jnp.int32),
        jnp.asarray(geo.cut.xout) if cfg.has_surf else None,
        cfg,
        vscale,
        tuple(float(a) for a in vstream),
        int(seed),
    )
    N = cfg.capacity
    pad = N - n
    full = lambda a, fill: jnp.concatenate([a, jnp.full(pad, fill, a.dtype)])  # noqa: E731
    state = state._replace(
        x=full(x[0], 0.0),
        y=full(x[1], 0.0),
        z=full(x[2], 0.0),
        vx=full(v[0], 0.0),
        vy=full(v[1], 0.0),
        vz=full(v[2], 0.0),
        cell=full(jnp.asarray(cell), cfg.DEAD),
    )
    return sort_full(state, cfg)


@functools.partial(jax.jit, static_argnames=("cfg", "vscale", "vstream", "seed"))
def _create_device(cell, consts, cut_part, cut_row, cut_xout, cfg, vscale, vstream, seed):
    n = cell.shape[0]
    key = functools.partial(rng.step_key, seed, 2**31 - 1)  # a step no run reaches
    q = jnp.arange(n, dtype=jnp.int64)
    ijk = _unravel_local(cell, cfg)
    g = [ijk[a] + consts.offset[a] for a in range(3)]
    lo = [cfg.lo[a] + g[a].astype(jnp.float64) * cfg.h[a] for a in range(3)]
    x = [lo[a] + rng.uniform(key(100 + a), q) * cfg.h[a] for a in range(3)]
    if cfg.has_surf:
        # rejection inside the body for the particles of cut cells, redrawn until outside
        # (at most 1000 rounds; a sliver cell can need many)
        lo_c = [lo[a][cut_part] for a in range(3)]
        xc = [x[a][cut_part] for a in range(3)]
        xo = cut_xout[cut_row]
        qc = q[cut_part]
        bad = ~_outside_cell(xc, xo, cut_row, consts, cfg)
        carry = (xc, bad, jnp.int32(1))
        body = functools.partial(_redraw, lo_c=lo_c, xo=xo, cut_row=cut_row, qc=qc, key=key, consts=consts, cfg=cfg)
        xc, bad, _t = jax.lax.while_loop(_redraw_cond, body, carry)
        x = [x[a].at[cut_part].set(xc[a]) for a in range(3)]
    vn = vscale * jnp.sqrt(-jnp.log1p(-rng.uniform(key(400), q)))
    vr = vscale * jnp.sqrt(-jnp.log1p(-rng.uniform(key(401), q)))
    t1 = 2.0 * math.pi * rng.uniform(key(402), q)
    t2 = 2.0 * math.pi * rng.uniform(key(403), q)
    v = [vstream[0] + vn * jnp.cos(t1), vstream[1] + vr * jnp.cos(t2), vstream[2] + vr * jnp.sin(t2)]
    return x, v


def _redraw(carry, lo_c, xo, cut_row, qc, key, consts, cfg):
    xc, bad, t = carry
    xn = [lo_c[a] + rng.uniform(key(100 + a), _ctr(qc, 1 << 12, t)) * cfg.h[a] for a in range(3)]
    xc = [jnp.where(bad, xn[a], xc[a]) for a in range(3)]
    return xc, bad & ~_outside_cell(xc, xo, cut_row, consts, cfg), t + 1


def _redraw_cond(carry):
    _xc, bad, t = carry
    return jnp.any(bad) & (t < 1000)


def _outside_cell(x, xo, ci, consts: Consts, cfg: Config):
    """Grid::outside_surfs: the segment from x to the cell's flow point crosses an even number
    of the cell's triangles (x is in the flow)."""
    tri = consts.cut_tri[ci]  # (n, K, 12)
    ids = consts.cut_ids[ci]
    col = functools.partial(jnp.expand_dims, axis=1)
    s = [col(x[a]) for a in range(3)]
    e = [col(xo[:, a]) for a in range(3)]
    hit, _param, _side = _line_tri(s, e, tri)
    cnt = jnp.sum((hit & (ids >= 0)).astype(jnp.int32), axis=1)
    return cnt % 2 == 0


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _isum(a):
    return jnp.sum(a, dtype=jnp.int32)


def _put(slot, arr, val):
    return arr.at[slot].set(val, mode="drop")


def _floor_index(u, n):
    return jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 1)


def _per_cell(u, h):
    """u / h as SPARTA's cell index computes it; a multiplication when 1/h is exact."""
    return u * (1.0 / h) if math.frexp(h)[0] == 0.5 else u / h


def _unravel_local(c, cfg: Config):
    bx, by, _ = cfg.bdim
    return c % bx, (c // bx) % by, c // (bx * by)


def _local_index(g, consts: Consts, cfg: Config):
    """Local cell of global indices g = (i, j, k) and whether it lies in the brick."""
    li = [g[a] - consts.offset[a] for a in range(3)]
    inb = (
        (li[0] >= 0)
        & (li[0] < consts.extent[0])
        & (li[1] >= 0)
        & (li[1] < consts.extent[1])
        & (li[2] >= 0)
        & (li[2] < consts.extent[2])
    )
    return li[0] + cfg.bdim[0] * (li[1] + cfg.bdim[1] * li[2]), inb


def _gid(g, cfg: Config):
    return g[0] + cfg.n[0] * (g[1] + cfg.n[1] * g[2])


def _unravel_gid(gid, cfg: Config):
    nx, ny, _ = cfg.n
    return gid % nx, (gid // nx) % ny, gid // (nx * ny)


def _ctr(slot, stride, extra):
    """uint64 RNG counter slot * stride + extra (no int32 overflow for large slots)."""
    return slot.astype(jnp.uint64) * np.uint64(stride) + jnp.asarray(extra).astype(jnp.uint64)


# ---------------------------------------------------------------------------------------------
# 2a. dense move
# ---------------------------------------------------------------------------------------------


def _straight(pos, vel, dtv, cfg: Config):
    """End point of a straight move with the global boundary rule.

    Returns (xn, v, exit, mirror_count, outside).  REFLECT mirrors as Domain::collide does when
    the walk reaches the face (xnew -> 2 bound - xnew, v -> -v), repeated while the point is
    still outside (a move longer than the box; three passes, ``outside`` flags anything left);
    OUTFLOW flags exits."""
    xn, vn = [], []
    exit_ = jnp.zeros(pos[0].shape, bool)
    outside = jnp.zeros(pos[0].shape, bool)
    nb = jnp.zeros(pos[0].shape, jnp.int32)
    for a in range(3):
        xa = pos[a] + dtv * vel[a]
        va = vel[a]
        lo, hi = cfg.lo[a], cfg.hi[a]
        if cfg.boundary == "r":
            for _ in range(3):
                below, above = xa < lo, xa >= hi
                xa = jnp.where(below, 2.0 * lo - xa, jnp.where(above, 2.0 * hi - xa, xa))
                va = jnp.where(below | above, -va, va)
                nb = nb + (below | above).astype(jnp.int32)
            outside = outside | (xa < lo) | (xa > hi)
        else:
            exit_ = exit_ | (xa < lo) | (xa >= hi)
        xn.append(xa)
        vn.append(va)
    return xn, vn, exit_, nb, outside


def _end_cell(xn, cfg: Config):
    return [_floor_index(_per_cell(xn[a] - cfg.lo[a], cfg.h[a]), cfg.n[a]) for a in range(3)]


def _safe(start_g, end_g, start_lcell, consts: Consts, cfg: Config):
    """The straight segment between the start and end cells meets no surface: the end cell is
    within the start cell's cut-free Chebyshev ring."""
    if not cfg.has_surf:
        return jnp.ones(start_lcell.shape, bool)
    r = consts.clear[jnp.minimum(start_lcell, cfg.ncell_local - 1)].astype(jnp.int32)
    d = jnp.maximum(
        jnp.maximum(jnp.abs(end_g[0] - start_g[0]), jnp.abs(end_g[1] - start_g[1])),
        jnp.abs(end_g[2] - start_g[2]),
    )
    return d <= r


def dense_move(state: State, dt, consts: Consts, cfg: Config):
    """Straight move of every alive slot that provably meets no surface; NEAR code for the rest.

    Returns (state, nbound).  Near slots keep their start point and velocity."""
    Cl = cfg.ncell_local
    cell = state.cell
    alive = cell < Cl
    pos = (state.x, state.y, state.z)
    vel = (state.vx, state.vy, state.vz)
    xn, vn, exit_, nb, outside = _straight(pos, vel, dt, cfg)
    end_g = _end_cell(xn, cfg)
    sc = _unravel_local(jnp.minimum(cell, Cl - 1), cfg)
    start_g = [sc[a] + consts.offset[a] for a in range(3)]
    safe = _safe(start_g, end_g, cell, consts, cfg)
    if cfg.has_surf:
        safe = safe & ~outside
    if cfg.has_surf and cfg.boundary == "r":
        safe = safe & (nb == 0)  # a mirrored path near a surface: walk
    lc, inb = _local_index(end_g, consts, cfg)
    moved = jnp.where(exit_, cfg.DEAD, jnp.where(inb, lc, cfg.LEFT0 + _gid(end_g, cfg)))
    new_cell = jnp.where(alive, jnp.where(safe, moved, cfg.NEAR0 + cell), cell)
    keep = ~(alive & safe)
    st = state._replace(
        x=jnp.where(keep, state.x, xn[0]),
        y=jnp.where(keep, state.y, xn[1]),
        z=jnp.where(keep, state.z, xn[2]),
        vx=jnp.where(keep, state.vx, vn[0]),
        vy=jnp.where(keep, state.vy, vn[1]),
        vz=jnp.where(keep, state.vz, vn[2]),
        cell=new_cell.astype(jnp.int32),
    )
    nbound = _isum(jnp.where(alive & safe, nb, 0))
    return st, nbound


# ---------------------------------------------------------------------------------------------
# 2b. emit/face (onepass)
# ---------------------------------------------------------------------------------------------


def emit(state: State, dt, seed, istep, consts: Consts, cfg: Config):
    """FixEmitFace::perform_task_onepass for every task (perspecies, one species).

    ninsert = int(ntarget + U) per task; per particle a uniform point on the task face, the
    normal velocity by SPARTA's accept-reject sampler (Bird eq. 12.5; the inner do-while is a
    uniform draw on [-s, 3)), tangential velocities vr sin/cos theta, dtremain = dt U.  The
    inserts are compacted into a buffer of ``ins_cap``, written to the dead tail of ``order``,
    and advanced by their dtremain (straight when safe, else NEAR for the walk).

    Returns (state, inserts (slot, dtremain) for the walk, counts)."""
    T, Mt, R, I = cfg.ntask, cfg.per_task, cfg.trials, cfg.ins_cap
    N, Cl = cfg.capacity, cfg.ncell_local
    key = functools.partial(rng.step_key, seed, istep)
    tq = jnp.arange(T, dtype=jnp.int32)
    ntarget = consts.task_rate * dt + rng.uniform(key(S_NINSERT), tq)
    ninsert = jnp.minimum(jnp.floor(ntarget).astype(jnp.int32), Mt)
    cs = jnp.cumsum(ninsert, dtype=jnp.int32)
    total = cs[-1]
    q = jnp.arange(I, dtype=jnp.int32)
    valid = q < total
    task = jnp.minimum(jnp.searchsorted(cs, q, side="right").astype(jnp.int32), T - 1)
    lcell = consts.task_lcell[task]
    face = consts.task_face[task]
    axis = face // 2
    hi_side = (face % 2) == 1
    sc = _unravel_local(lcell, cfg)
    g = [sc[a] + consts.offset[a] for a in range(3)]
    pos = []
    for a in range(3):
        clo = cfg.lo[a] + g[a].astype(jnp.float64) * cfg.h[a]
        u = rng.uniform(key(S_POS), _ctr(q, 3, a))
        on = axis == a
        pos.append(jnp.where(on, jnp.where(hi_side, clo + cfg.h[a], clo), clo + u * cfg.h[a]))
    sign = jnp.where(hi_side, -1.0, 1.0)
    vs = jnp.asarray(cfg.vstream)
    vstream_n = vs[axis]
    s = sign * vstream_n / cfg.vscale  # scosine = indot / vscale
    r = jnp.sqrt(s * s + 2.0)
    beta1, fail = _normal_beta(s, r, q, valid, key, cfg)
    theta = 2.0 * math.pi * rng.uniform(key(S_THETA), q)
    vr = cfg.vscale * jnp.sqrt(-jnp.log1p(-rng.uniform(key(S_VR), q)))
    # normal axis a, tangential axes (pdim, qdim) = the other two in order
    p_axis = jnp.where(axis == 0, 1, 0)
    q_axis = jnp.where(axis == 2, 1, 2)
    vel = []
    for a in range(3):
        vn_a = beta1 * cfg.vscale * sign + cfg.vstream[a]
        vp_a = vr * jnp.sin(theta) + cfg.vstream[a]
        vq_a = vr * jnp.cos(theta) + cfg.vstream[a]
        vel.append(jnp.where(axis == a, vn_a, jnp.where(p_axis == a, vp_a, vq_a)))
    dtr = dt * rng.uniform(key(S_DT), q)
    # advance by dtremain
    xn, vn, exit_, nb, outside = _straight(pos, vel, dtr, cfg)
    end_g = _end_cell(xn, cfg)
    safe = _safe(g, end_g, lcell, consts, cfg)
    if cfg.has_surf:
        safe = safe & ~outside
    lc, inb = _local_index(end_g, consts, cfg)
    moved = jnp.where(exit_, cfg.DEAD, jnp.where(inb, lc, cfg.LEFT0 + _gid(end_g, cfg)))
    cell = jnp.where(safe, moved, cfg.NEAR0 + lcell)
    xw = [jnp.where(safe, xn[a], pos[a]) for a in range(3)]
    vw = [jnp.where(safe, vn[a], vel[a]) for a in range(3)]
    # add_particle: the next free slots are the dead tail of the sorted slot list
    slot_pos = state.ftop + q
    ok_slot = valid & (slot_pos < N)
    slot = jnp.where(ok_slot, state.order[jnp.minimum(slot_pos, N - 1)], N)
    put = functools.partial(_put, slot)
    state = state._replace(
        x=put(state.x, xw[0]),
        y=put(state.y, xw[1]),
        z=put(state.z, xw[2]),
        vx=put(state.vx, vw[0]),
        vy=put(state.vy, vw[1]),
        vz=put(state.vz, vw[2]),
        cell=put(state.cell, cell.astype(jnp.int32)),
    )
    state = state._replace(ftop=jnp.minimum(state.ftop + jnp.minimum(total, I), N).astype(jnp.int32))
    walk_slot = jnp.where(ok_slot & ~safe, slot, N)
    counts = {
        "ninsert": _isum(ok_slot),
        "emit_fail": _isum(fail),
        "overflow": _isum(valid & ~ok_slot) + jnp.maximum(total - I, 0),
        "nbound": _isum(jnp.where(ok_slot & safe, nb, 0)),
    }
    return state, (walk_slot, dtr), counts


def _beta_round(carry, s, r, q, key, R):
    """One round of R vectorised trials for the entries not yet accepted."""
    beta1, done, rnd = carry
    qt = _ctr(q[:, None], 1 << 16, rnd * R + jnp.arange(R, dtype=jnp.int32)[None, :])
    beta = -s[:, None] + (3.0 + s[:, None]) * rng.uniform(key(S_BETA), qt)
    f = (
        2.0
        * (beta + s[:, None])
        / (s + r)[:, None]
        * jnp.exp(0.5 + (0.5 * s * (s - r))[:, None] - beta * beta)
    )
    ok = f >= rng.uniform(key(S_ACCEPT), qt)
    first = jnp.argmax(ok, axis=1)
    b = jnp.take_along_axis(beta, first[:, None], axis=1)[:, 0]
    take = ~done & jnp.any(ok, axis=1)
    return jnp.where(take, b, beta1), done | take, rnd + 1


def _beta_cond(carry, max_rounds):
    _b, done, rnd = carry
    return ~jnp.all(done) & (rnd < max_rounds)


def _normal_beta(s, r, q, valid, key, cfg: Config):
    """SPARTA's outer accept-reject loop for the normal velocity, as rounds of ``trials``
    vectorised trials until every insert has accepted one (per-trial acceptance is ~0.18 at
    the sphere's xlo face, s = 7; 64 rounds of 32 is a failure probability of ~1e-170)."""
    R = cfg.trials
    carry = (jnp.zeros(q.shape), ~valid, jnp.int32(0))
    body = functools.partial(_beta_round, s=s, r=r, q=q, key=key, R=R)
    beta1, done, _ = jax.lax.while_loop(functools.partial(_beta_cond, max_rounds=64), body, carry)
    return beta1, ~done


# ---------------------------------------------------------------------------------------------
# 2c. the cell walk
# ---------------------------------------------------------------------------------------------


class Walk(NamedTuple):
    """Per-lane state of Update::move's while(1) loop.
    ``flags = status | stuck << 2 | (exclude + 1) << 8``; ``gid`` = global cell."""

    x: jax.Array
    y: jax.Array
    z: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array
    xn: jax.Array
    yn: jax.Array
    zn: jax.Array
    dtrem: jax.Array
    gid: jax.Array
    flags: jax.Array


def _flags(status, stuck, exclude):
    return status | (stuck << 2) | ((exclude + 1) << 8)


def _status(flags):
    return flags & 3


def _line_tri(s, e, tri):
    """Geometry::line_tri_intersect on broadcast blocks: s, e lists of 3 arrays (B, 1),
    tri (B, K, 12).  Returns (hit, param, side_outside)."""
    v0 = [tri[..., a] for a in range(3)]
    v1 = [tri[..., 3 + a] for a in range(3)]
    v2 = [tri[..., 6 + a] for a in range(3)]
    nrm = [tri[..., 9 + a] for a in range(3)]

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    dotstart = dot(nrm, sub(s, v0))
    dotstop = dot(nrm, sub(e, v0))
    miss = (
        ((dotstart < 0.0) & (dotstop < 0.0))
        | ((dotstart > 0.0) & (dotstop > 0.0))
        | ((dotstart == 0.0) & (dotstop == 0.0))
    )
    s2e = sub(e, s)
    den = dot(nrm, s2e)
    param = dot(nrm, sub(v0, s)) / jnp.where(den == 0.0, 1.0, den)
    param = jnp.clip(param, 0.0, 1.0)
    pt = [s[a] + param * s2e[a] for a in range(3)]
    for a_, b_ in ((v0, v1), (v1, v2), (v2, v0)):
        xp = cross(sub(b_, a_), sub(pt, a_))
        miss = miss | (dot(xp, nrm) < EPSSQNEG)
    return ~miss, param, dotstart > 0.0


def _cell_box(gid, cfg: Config):
    g = _unravel_gid(gid, cfg)
    lo = [cfg.lo[a] + g[a].astype(jnp.float64) * cfg.h[a] for a in range(3)]
    hi = [cfg.lo[a] + (g[a] + 1).astype(jnp.float64) * cfg.h[a] for a in range(3)]
    return g, lo, hi


def _face(w: Walk, lo, hi):
    """(outface, frac) of update.cpp: x, then y, then z with newfrac < frac."""
    p = (w.x, w.y, w.z)
    pn = (w.xn, w.yn, w.zn)
    outface = jnp.full(w.x.shape, INTERIOR, jnp.int32)
    frac = jnp.ones(w.x.shape)
    for a in range(3):
        f_lo = pn[a] < lo[a]
        f_hi = ~f_lo & (pn[a] >= hi[a])
        dm = pn[a] - p[a]
        nf = jnp.clip(
            jnp.where(dm != 0.0, (jnp.where(f_lo, lo[a], hi[a]) - p[a]) / jnp.where(dm != 0.0, dm, 1.0), 0.0),
            0.0,
            1.0,
        )
        if a == 0:
            take = f_lo | f_hi
        else:
            take = (f_lo | f_hi) & (nf < frac)
        frac = jnp.where(take, nf, frac)
        outface = jnp.where(take, jnp.where(f_lo, 2 * a, 2 * a + 1), outface)
    return outface, frac


def _snap(pt, outface, lo, hi):
    return [
        jnp.where(outface == 2 * a, lo[a], jnp.where(outface == 2 * a + 1, hi[a], pt[a]))
        for a in range(3)
    ]


def _argmin_reducer(a, b):
    """Smaller param wins, ties go to the lower slot (SPARTA keeps the first surf with
    param < minparam); the passengers ride with the winner."""
    pa, ka = a[0], a[1]
    pb, kb = b[0], b[1]
    take_a = (pa < pb) | ((pa == pb) & (ka <= kb))
    return tuple(jnp.where(take_a, x, y) for x, y in zip(a, b, strict=True))


def _surface_hit(w: Walk, outface, frac, lo, hi, consts: Consts, cfg: Config):
    """Earliest OUTSIDE hit among the cell's triangles on the in-cell part of the path."""
    K = cfg.K
    m = jnp.minimum(jnp.searchsorted(consts.cut_gid, w.gid).astype(jnp.int32), cfg.ncut - 1)
    is_cut = consts.cut_gid[m] == w.gid
    tri = consts.cut_tri[m]  # (B, K, 12)
    ids = jnp.where(is_cut[:, None], consts.cut_ids[m], -1)  # (B, K)
    nsurf = jnp.where(is_cut, consts.cut_nsurf[m], 0)
    crossing = outface != INTERIOR
    p = (w.x, w.y, w.z)
    pn = (w.xn, w.yn, w.zn)
    c = [jnp.where(crossing, p[a] + frac * (pn[a] - p[a]), pn[a]) for a in range(3)]
    c = _snap(c, outface, lo, hi)
    col = functools.partial(jnp.expand_dims, axis=1)
    hit, param, outside = _line_tri([col(v) for v in p], [col(v) for v in c], tri)
    exclude = (w.flags >> 8) - 1
    ok = (ids >= 0) & (ids != col(exclude)) & hit & outside
    param = jnp.where(ok, param, 2.0)
    kk = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), ids.shape)
    ops = (param, kk, tri[..., 9], tri[..., 10], tri[..., 11], ids)
    init = (jnp.float64(3.0), jnp.int32(K), jnp.float64(0), jnp.float64(0), jnp.float64(0), jnp.int32(-1))
    minparam, _k, nx_, ny_, nz_, msurf = jax.lax.reduce(ops, init, _argmin_reducer, (1,))
    minsurf = jnp.where(minparam < 2.0, msurf, -1)
    mx = [p[a] + minparam * (c[a] - p[a]) for a in range(3)]
    return minparam, minsurf, mx, (nx_, ny_, nz_), nsurf


def _diffuse(v, nrm, seed, istep, ctr, cfg: Config):
    """SurfCollideDiffuse::diffuse (3-D)."""
    vrm = math.sqrt(2.0 * G3.BOLTZ * cfg.tsurf / cfg.mass)
    u1 = rng.uniform(rng.step_key(seed, istep, S_DIFF1), ctr)
    u2 = rng.uniform(rng.step_key(seed, istep, S_DIFF2), ctr)
    u3 = rng.uniform(rng.step_key(seed, istep, S_DIFF3), ctr)
    vperp = vrm * jnp.sqrt(-jnp.log1p(-u1))
    theta = 2.0 * math.pi * u2
    vtan = vrm * jnp.sqrt(-jnp.log1p(-u3))
    vt1, vt2 = vtan * jnp.sin(theta), vtan * jnp.cos(theta)
    dot = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2]
    t1 = [v[a] - dot * nrm[a] for a in range(3)]
    ln = jnp.sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2])
    zero = ln == 0.0
    # normal incidence: SPARTA picks a random tangent; any unit tangent has the same law
    alt = [jnp.where(jnp.abs(nrm[0]) < 0.9, 0.0, -nrm[1]), jnp.where(jnp.abs(nrm[0]) < 0.9, -nrm[2], nrm[0]), jnp.where(jnp.abs(nrm[0]) < 0.9, nrm[1], 0.0)]
    t1 = [jnp.where(zero, alt[a], t1[a]) for a in range(3)]
    ln = jnp.sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2])
    t1 = [t1[a] / ln for a in range(3)]
    t2 = [
        nrm[1] * t1[2] - nrm[2] * t1[1],
        nrm[2] * t1[0] - nrm[0] * t1[2],
        nrm[0] * t1[1] - nrm[1] * t1[0],
    ]
    return [vperp * nrm[a] + vt1 * t1[a] + vt2 * t2[a] for a in range(3)]


def _surface_collide(w: Walk, nrm, slot, seed, istep, it, cfg: Config):
    """SurfCollideDiffuse::collide: one uniform draw; specular if U > acc, else diffuse."""
    v = [w.vx, w.vy, w.vz]
    vdot = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2]
    spec = [v[a] - 2.0 * vdot * nrm[a] for a in range(3)]
    if cfg.acc <= 0.0:
        return spec
    ctr = _ctr(slot, 4096, it)
    dif = _diffuse(v, nrm, seed, istep, ctr, cfg)
    if cfg.acc >= 1.0:
        return dif  # U > 1 never holds: SPARTA's draw is consumed, the result is always diffuse
    specular = rng.uniform(rng.step_key(seed, istep, S_SURF), ctr) > cfg.acc
    return [jnp.where(specular, spec[a], dif[a]) for a in range(3)]


def _walk_stage(w: Walk, slot, seed, istep, it, consts: Consts, cfg: Config):
    """One pass of Update::move's per-particle while(1) loop over every ACTIVE lane entry."""
    status, stuck = _status(w.flags), (w.flags >> 2) & 63
    act = status == ACTIVE
    g, lo, hi = _cell_box(w.gid, cfg)
    outface, frac = _face(w, lo, hi)
    if cfg.has_surf:
        minparam, minsurf, mx, nrm, nsurf = _surface_hit(w, outface, frac, lo, hi, consts, cfg)
        cflag = act & (minsurf >= 0)
        sv = _surface_collide(w, nrm, slot, seed, istep, it, cfg)
        dtrem_c = w.dtrem * (1.0 - minparam * frac)
        stuck_c = jnp.where(minparam <= 1.0e-14, stuck + 1, 0)
        discard_stuck = cflag & (stuck_c >= MAXSTUCK)
    else:
        cflag = jnp.zeros(w.x.shape, bool)
        nsurf = jnp.zeros(w.x.shape, jnp.int32)
        discard_stuck = cflag
        minsurf = jnp.full(w.x.shape, -1, jnp.int32)
        mx = [w.x, w.y, w.z]
        sv = [w.vx, w.vy, w.vz]
        dtrem_c = w.dtrem
        stuck_c = stuck
    crossing = outface != INTERIOR
    interior = act & ~cflag & ~crossing
    cross = act & ~cflag & crossing
    p = [w.x, w.y, w.z]
    pn = [w.xn, w.yn, w.zn]
    f = _snap([p[a] + frac * (pn[a] - p[a]) for a in range(3)], outface, lo, hi)
    axis = jnp.where(crossing, outface // 2, 0)
    upper = (outface % 2) == 1
    at_bound = jnp.zeros(w.x.shape, bool)
    for a in range(3):
        at_bound = at_bound | (
            (outface == 2 * a) & (g[a] == 0) | (outface == 2 * a + 1) & (g[a] == cfg.n[a] - 1)
        )
    if cfg.boundary == "o":
        exits = cross & at_bound
        reflect = jnp.zeros(w.x.shape, bool)
    else:
        exits = jnp.zeros(w.x.shape, bool)
        reflect = cross & at_bound
    newcell = cross & ~at_bound
    stride = jnp.where(axis == 0, 1, jnp.where(axis == 1, cfg.n[0], cfg.n[0] * cfg.n[1]))
    step = jnp.where(upper, stride, -stride)
    status = jnp.where(exits | discard_stuck, DEADW, jnp.where(interior, DONE, status))
    exclude = (w.flags >> 8) - 1
    flags = _flags(
        status,
        jnp.where(cflag, stuck_c, stuck),
        jnp.where(cflag, minsurf, jnp.where(cross, -1, exclude)),
    )
    newp, newpn, newv = [], [], []
    for a in range(3):
        mirror = reflect & (axis == a)
        bound = jnp.where(upper, hi[a], lo[a])
        newp.append(jnp.where(cflag, mx[a], jnp.where(interior, pn[a], jnp.where(cross, f[a], p[a]))))
        vv = (w.vx, w.vy, w.vz)[a]
        newv.append(jnp.where(cflag, sv[a], jnp.where(mirror, -vv, vv)))
        newpn.append(
            jnp.where(cflag, mx[a] + dtrem_c * sv[a], jnp.where(mirror, 2.0 * bound - pn[a], pn[a]))
        )
    w2 = Walk(
        x=newp[0],
        y=newp[1],
        z=newp[2],
        vx=newv[0],
        vy=newv[1],
        vz=newv[2],
        xn=newpn[0],
        yn=newpn[1],
        zn=newpn[2],
        dtrem=jnp.where(cflag, dtrem_c, jnp.where(cross, w.dtrem * (1.0 - frac), w.dtrem)),
        gid=jnp.where(newcell, w.gid + step, w.gid).astype(jnp.int32),
        flags=flags.astype(jnp.int32),
    )
    per = [cflag, jnp.where(act, nsurf, 0), exits, reflect, discard_stuck]
    cnt = jnp.sum(jnp.stack([a.astype(jnp.int32) for a in per], axis=1), axis=0, dtype=jnp.int32)
    return w2, cnt


def _compact(mask, size):
    """Indices of the first ``size`` set entries of mask (len(mask) where there are fewer)."""
    n = mask.shape[0]
    r = jnp.cumsum(mask.astype(jnp.int32)) - 1
    return (
        jnp.full(size, n, jnp.int32)
        .at[jnp.where(mask & (r < size), r, size)]
        .set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )


def _take(a, idx, fill):
    return jnp.take(a, idx, axis=0, mode="fill", fill_value=fill)


def _later_stage(carry, slots, seed, istep, size, consts: Consts, cfg: Config):
    w, it, cnt = carry
    idx = _compact(_status(w.flags) == ACTIVE, size)
    part = Walk(*(_take(a, idx, 0) for a in w))  # empty entries: flags 0 = DONE
    slot = _take(slots, idx, cfg.capacity)
    part, c = _walk_stage(part, slot, seed, istep, it, consts, cfg)
    w = Walk(*(a.at[idx].set(b, mode="drop") for a, b in zip(w, part, strict=True)))
    return w, it + 1, cnt + c


def _later_cond(carry, cfg: Config):
    w, it, _cnt = carry
    return jnp.any(_status(w.flags) == ACTIVE) & (it < cfg.max_walk)


def _lane_gather(state: State, c, rank, near, dt, inserts, consts: Consts, cfg: Config):
    """Slots and start state of the ``c``-th block of ``lane`` near particles.  Every move
    lasts dt except this step's unsafe inserts ``(slot, dtremain)``."""
    B, N, Cl = cfg.lane, cfg.capacity, cfg.ncell_local
    local = rank - c * B
    slots = (
        jnp.full(B, N, jnp.int32)
        .at[jnp.where(near & (local >= 0) & (local < B), local, B)]
        .set(jnp.arange(N, dtype=jnp.int32), mode="drop")
    )
    valid = slots < N
    gs = jnp.minimum(slots, N - 1)
    x, y, z = state.x[gs], state.y[gs], state.z[gs]
    vx, vy, vz = state.vx[gs], state.vy[gs], state.vz[gs]
    lcell = jnp.where(valid, state.cell[gs] - cfg.NEAR0, 0)
    sc = _unravel_local(lcell, cfg)
    gid = _gid([sc[a] + consts.offset[a] for a in range(3)], cfg)
    dtr = jnp.full(B, 1.0) * dt
    if inserts is not None:
        islot, idtr = inserts
        lpos = rank[jnp.minimum(islot, N - 1)] - c * B
        hit = (islot < N) & (lpos >= 0) & (lpos < B)
        dtr = dtr.at[jnp.where(hit, lpos, B)].set(idtr, mode="drop")
    w = Walk(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        xn=x + dtr * vx,
        yn=y + dtr * vy,
        zn=z + dtr * vz,
        dtrem=dtr,
        gid=gid.astype(jnp.int32),
        flags=jnp.where(valid, _flags(ACTIVE, 0, -1), _flags(DONE, 0, -1)).astype(jnp.int32),
    )
    return slots, w


def _lane_walk(carry, slots, w: Walk, seed, istep, consts: Consts, cfg: Config):
    """Walk one lane block to the end of its moves and write the results to the slots."""
    state, c, cnt, iters = carry
    w, cnt1 = _walk_stage(w, slots, seed, istep, jnp.int32(0), consts, cfg)
    stage = functools.partial(_later_stage, slots=slots, seed=seed, istep=istep, consts=consts, cfg=cfg)
    carry = stage((w, jnp.int32(1), cnt1), size=cfg.lane2)
    small = functools.partial(stage, size=cfg.lane3)
    w, it, cnt_w = jax.lax.while_loop(functools.partial(_later_cond, cfg=cfg), small, carry)
    put = functools.partial(_put, slots)
    g = _unravel_gid(w.gid, cfg)
    lc, inb = _local_index(g, consts, cfg)
    status = _status(w.flags)
    # a lane entry still ACTIVE after max_walk stages is discarded like a stuck particle
    dead = (status == DEADW) | (status == ACTIVE)
    state = state._replace(
        x=put(state.x, w.x),
        y=put(state.y, w.y),
        z=put(state.z, w.z),
        vx=put(state.vx, w.vx),
        vy=put(state.vy, w.vy),
        vz=put(state.vz, w.vz),
        cell=put(state.cell, jnp.where(dead, cfg.DEAD, jnp.where(inb, lc, cfg.LEFT0 + w.gid))),
    )
    return state, c + 1, cnt + cnt_w, iters + it


def _lane_chunk(carry, rank, near, dt, inserts, seed, istep, consts: Consts, cfg: Config):
    slots, w = _lane_gather(carry[0], carry[1], rank, near, dt, inserts, consts, cfg)
    return _lane_walk(carry, slots, w, seed, istep, consts, cfg)


def _lane_cond(carry, nchunk):
    return carry[1] < nchunk


def walk(state: State, dt, seed, istep, consts: Consts, cfg: Config, inserts=None):
    """The cell walk for every NEAR slot, in blocks of ``lane``."""
    near = (state.cell >= cfg.NEAR0) & (state.cell < cfg.LEFT0)
    rank = jnp.cumsum(near.astype(jnp.int32)) - 1
    nnear = rank[-1] + 1
    nchunk = (nnear + cfg.lane - 1) // cfg.lane
    carry = (state, jnp.int32(0), jnp.zeros(5, jnp.int32), jnp.int32(0))
    body = functools.partial(
        _lane_chunk, rank=rank, near=near, dt=dt, inserts=inserts, seed=seed, istep=istep, consts=consts, cfg=cfg
    )
    state, _c, cnt, iters = jax.lax.while_loop(functools.partial(_lane_cond, nchunk=nchunk), body, carry)
    counts = {
        "nscoll": cnt[0],
        "nscheck": cnt[1],
        "nexit_walk": cnt[2],
        "nbound": cnt[3],
        "nstuck": cnt[4],
        "nlane": nnear,
        "nwalk": iters,
    }
    return state, counts


# ---------------------------------------------------------------------------------------------
# 3. sort
# ---------------------------------------------------------------------------------------------


def sort_full(state: State, cfg: Config):
    """Particle::sort: slots ordered by local cell (dead last) and the bucket starts."""
    N, Cl = cfg.capacity, cfg.ncell_local
    key = jnp.minimum(state.cell, Cl)  # DEAD (and any stray code) last
    if cfg.sort == "pack":  # one int64 key (cell << 32 | slot): a single-operand radix sort
        packed = jnp.sort((key.astype(jnp.int64) << 32) | jnp.arange(N, dtype=jnp.int64))
        order = (packed & 0xFFFFFFFF).astype(jnp.int32)
        skey = (packed >> 32).astype(jnp.int32)
    else:  # "full": (key, slot) pairs sorted by key
        skey, order = jax.lax.sort((key, jnp.arange(N, dtype=jnp.int32)), num_keys=1)
    if cfg.starts == "search":  # bucket starts by binary search in the sorted keys
        start = jnp.searchsorted(skey, jnp.arange(Cl + 1, dtype=jnp.int32), side="left").astype(jnp.int32)
        start = jnp.concatenate([start, jnp.full(1, N, jnp.int32)])
    else:  # "scatter": histogram by scatter-add, then a prefix sum
        counts = jnp.zeros(Cl + 1, jnp.int32).at[key].add(1)
        start = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
    return state._replace(order=order, start=start, ftop=start[Cl])


# ---------------------------------------------------------------------------------------------
# 4. collisions: Collide::collisions_one + CollideVSS (argon: no internal energy)
# ---------------------------------------------------------------------------------------------


def _attempt_cells(vx, vy, vz, vremax, cells, act, k, state: State, npc, seed, istep, cfg: Config):
    """Attempt number k in each cell of ``cells`` where ``act`` (cells are independent).
    CollideVSS::test_collision + setup_collision + SCATTER_TwoBodyScattering (VSS branch;
    argon: no internal energy, so the relative speed is unchanged)."""
    Cl, N = cfg.ncell_local, cfg.capacity
    key = rng.step_key(seed, istep, S_COLL)
    base_ctr = _ctr(cells, 5, 0) + jnp.asarray(k, jnp.uint64) * np.uint64(5 * Cl)
    u = [rng.uniform(key, base_ctr + np.uint64(d)) for d in range(5)]
    n = npc[cells]
    npf = n.astype(jnp.float64)
    i = jnp.minimum((npf * u[0]).astype(jnp.int32), n - 1)
    j = jnp.minimum(((npf - 1.0) * u[1]).astype(jnp.int32), n - 2)
    j = j + (j >= i)
    base = state.start[cells]
    pi = state.order[jnp.clip(base + i, 0, N - 1)]
    pj = state.order[jnp.clip(base + j, 0, N - 1)]
    vix, viy, viz = vx[pi], vy[pi], vz[pi]
    vjx, vjy, vjz = vx[pj], vy[pj], vz[pj]
    du, dv, dw = vix - vjx, viy - vjy, viz - vjz
    vr2 = du * du + dv * dv + dw * dw
    vre = vr2 ** (1.0 - cfg.omega) * cfg.prefactor
    vm = vremax[cells]
    vm = jnp.where(act, jnp.maximum(vre, vm), vm)
    accept = act & (vm > 0.0) & (vre / jnp.where(vm > 0.0, vm, 1.0) >= u[2])
    vr = jnp.sqrt(vr2)
    ucm, vcm, wcm = 0.5 * (vix + vjx), 0.5 * (viy + vjy), 0.5 * (viz + vjz)
    eps = u[3] * 2.0 * math.pi
    cosx = 2.0 * u[4] ** (1.0 / cfg.alpha) - 1.0
    sinx = jnp.sqrt(1.0 - cosx * cosx)
    big = jnp.sqrt(dv * dv + dw * dw) > 1.0e-6
    d = jnp.where(big, jnp.sqrt(dv * dv + dw * dw), 1.0)
    se, ce = jnp.sin(eps), jnp.cos(eps)
    ua = jnp.where(big, cosx * du + sinx * d * se, cosx * du)
    vb = jnp.where(big, cosx * dv + sinx * (vr * dw * ce - du * dv * se) / d, sinx * du * ce)
    wc = jnp.where(big, cosx * dw - sinx * (vr * dv * ce + du * dw * se) / d, sinx * du * se)
    pi_w, pj_w = jnp.where(accept, pi, N), jnp.where(accept, pj, N)
    vx = vx.at[pi_w].set(ucm + 0.5 * ua, mode="drop").at[pj_w].set(ucm - 0.5 * ua, mode="drop")
    vy = vy.at[pi_w].set(vcm + 0.5 * vb, mode="drop").at[pj_w].set(vcm - 0.5 * vb, mode="drop")
    vz = vz.at[pi_w].set(wcm + 0.5 * wc, mode="drop").at[pj_w].set(wcm - 0.5 * wc, mode="drop")
    vremax = vremax.at[jnp.where(act, cells, Cl)].set(vm, mode="drop")
    return vx, vy, vz, vremax, _isum(accept)


def _collide_attempt(carry, state: State, npc, nattempt, seed, istep, cfg: Config):
    """("all") attempt k on every cell; inactive cells are masked."""
    vx, vy, vz, vremax, k, ncoll = carry
    cells = jnp.arange(cfg.ncell_local, dtype=jnp.int32)
    vx, vy, vz, vremax, nc = _attempt_cells(vx, vy, vz, vremax, cells, k < nattempt, k, state, npc, seed, istep, cfg)
    return vx, vy, vz, vremax, k + 1, ncoll + nc


def _prefix_sizes(Cl):
    """Static prefix lengths of the cell list: Cl, Cl/4, Cl/16, ... down to 256."""
    sizes = [Cl]
    while sizes[-1] > 1024:
        sizes.append(_round_up(sizes[-1] // 4, 256))
    return sizes


def _prefix_branch(vx, vy, vz, vremax, k, m, state, npc, nattempt, by_count, seed, istep, size, cfg):
    cells = by_count[:size]
    act = (jnp.arange(size, dtype=jnp.int32) < m) & (k < nattempt[cells])
    return _attempt_cells(vx, vy, vz, vremax, cells, act, k, state, npc, seed, istep, cfg)


def _collide_prefix(carry, state: State, npc, nattempt, by_count, m_of_k, seed, istep, cfg: Config):
    """("prefix") attempt k on the cells with more than k attempts: they are the first m_k
    entries of the cells ordered by attempt count (descending), so the attempt runs on the
    smallest static prefix that holds them."""
    vx, vy, vz, vremax, k, ncoll = carry
    m = m_of_k[jnp.minimum(k, m_of_k.shape[0] - 1)]
    sizes = _prefix_sizes(cfg.ncell_local)
    branch = jnp.sum((jnp.asarray(sizes[1:], jnp.int32) >= m).astype(jnp.int32))  # smallest size >= m
    fns = [
        functools.partial(
            _prefix_branch, state=state, npc=npc, nattempt=nattempt, by_count=by_count, seed=seed, istep=istep, size=sz, cfg=cfg
        )
        for sz in sizes
    ]
    vx, vy, vz, vremax, nc = jax.lax.switch(branch, fns, vx, vy, vz, vremax, k, m)
    return vx, vy, vz, vremax, k + 1, ncoll + nc


def _collide_cond(carry, kmax):
    return carry[4] < kmax


def collide(state: State, dt, seed, istep, consts: Consts, cfg: Config):
    """NTC with per-cell vremax and the fractional 'remain' carry (collide_modify default)."""
    Cl = cfg.ncell_local
    npc = state.start[1 : Cl + 1] - state.start[:Cl]
    npf = npc.astype(jnp.float64)
    live = npc > 1
    vol = jnp.where(consts.volume > 0.0, consts.volume, 1.0)
    attempt = 0.5 * npf * (npf - 1.0) * state.vremax * dt * cfg.fnum / vol + state.remain
    fl = jnp.floor(attempt)
    nattempt = jnp.where(live, fl, 0.0).astype(jnp.int32)
    remain = jnp.where(live, attempt - fl, state.remain)
    kmax = jnp.max(nattempt)
    carry = (state.vx, state.vy, state.vz, state.vremax, jnp.int32(0), jnp.int32(0))
    if cfg.collide_mode == "prefix":
        # cells by attempt count, descending (one packed int64 sort of Cl keys)
        packed = jnp.sort(((-nattempt).astype(jnp.int64) << 32) | jnp.arange(Cl, dtype=jnp.int64))
        by_count = (packed & 0xFFFFFFFF).astype(jnp.int32)
        # m_k = number of cells with more than k attempts, exact for k < 64; beyond, the k = 63
        # count (an over-estimate of the prefix; the per-cell k < nattempt test is exact)
        hist = jnp.zeros(65, jnp.int32).at[jnp.minimum(nattempt, 64)].add(1)
        m_of_k = Cl - jnp.cumsum(hist, dtype=jnp.int32)[:64]
        body = functools.partial(
            _collide_prefix,
            state=state,
            npc=npc,
            nattempt=nattempt,
            by_count=by_count,
            m_of_k=m_of_k,
            seed=seed,
            istep=istep,
            cfg=cfg,
        )
    else:
        body = functools.partial(
            _collide_attempt, state=state, npc=npc, nattempt=nattempt, seed=seed, istep=istep, cfg=cfg
        )
    for _ in range(cfg.collide_unroll):  # masked when k >= nattempt: exact either way
        carry = body(carry)
    vx, vy, vz, vremax, _k, ncoll = jax.lax.while_loop(
        functools.partial(_collide_cond, kmax=kmax), body, carry
    )
    state = state._replace(vx=vx, vy=vy, vz=vz, vremax=vremax, remain=remain)
    return state, _isum(nattempt), ncoll


def reset_vremax(state: State, cfg: Config):
    """Collide::reset_vremax: vremax to its initial value, remain to 0."""
    return state._replace(vremax=jnp.full_like(state.vremax, cfg.vremax0), remain=jnp.zeros_like(state.remain))


# ---------------------------------------------------------------------------------------------
# step / run
# ---------------------------------------------------------------------------------------------


def move(state: State, dt, seed, istep, consts: Consts, cfg: Config):
    """Update::move for all particles: dense move, emit (+ advance), cell walk."""
    with jax.named_scope("move_dense"):
        state, nbound_dense = dense_move(state, dt, consts, cfg)
    ec = {"ninsert": jnp.int32(0), "emit_fail": jnp.int32(0), "overflow": jnp.int32(0), "nbound": jnp.int32(0)}
    inserts = None
    if cfg.emit:
        with jax.named_scope("emit"):
            state, inserts, ec = emit(state, dt, seed, istep, consts, cfg)
    if cfg.has_surf:
        with jax.named_scope("move_walk"):
            state, wc = walk(state, dt, seed, istep, consts, cfg, inserts)
    else:
        z = jnp.int32(0)
        wc = {"nscoll": z, "nscheck": z, "nexit_walk": z, "nbound": z, "nstuck": z, "nlane": z, "nwalk": z}
    wc["nbound"] = wc["nbound"] + nbound_dense + ec["nbound"]
    return state, wc, ec


def step(state: State, istep, dt, seed, consts: Consts, cfg: Config, migrate=None):
    """One timestep (``istep``: SPARTA's 1-based ntimestep)."""
    Cl = cfg.ncell_local
    if cfg.collide and cfg.vre_every:
        reset = (istep % cfg.vre_every) == 0
        state = state._replace(
            vremax=jnp.where(reset, cfg.vremax0, state.vremax),
            remain=jnp.where(reset, 0.0, state.remain),
        )
    n_before = state.start[Cl]
    state, wc, ec = move(state, dt, seed, istep, consts, cfg)
    mig = {"overflow": jnp.int32(0), "sent": jnp.int32(0), "received": jnp.int32(0)}
    if migrate is not None:
        with jax.named_scope("migrate"):
            state, mig = migrate(state)
    nattempt = ncoll = jnp.int32(0)
    if cfg.collide:
        with jax.named_scope("sort"):
            state = sort_full(state, cfg)
        with jax.named_scope("collide"):
            state, nattempt, ncoll = collide(state, dt, seed, istep, consts, cfg)
    else:
        # no sort in SPARTA without collisions (the free slots are kept by ``ftop``); keep
        # ``start[Cl]`` (the alive count) current for the stats
        alive = _isum(state.cell < Cl)
        state = state._replace(start=state.start.at[Cl].set(alive))
    n_after = state.start[Cl]
    stats = Stats(
        np=n_after,
        nattempt=nattempt,
        ncoll=ncoll,
        nscoll=wc["nscoll"],
        nscheck=wc["nscheck"],
        # every other change of the count is an exit (or a stuck discard)
        nexit=n_before + ec["ninsert"] + mig["received"] - mig["sent"] - n_after - wc["nstuck"],
        nbound=wc["nbound"],
        ninsert=ec["ninsert"],
        nlane=wc["nlane"],
        nwalk=wc["nwalk"],
        nstuck=wc["nstuck"],
        emit_fail=ec["emit_fail"],
        overflow=ec["overflow"] + mig["overflow"],
    )
    return state, stats


def _scan_body(state, istep, dt, seed, consts, cfg):
    return step(state, istep, dt, seed, consts, cfg)


@functools.partial(jax.jit, static_argnames=("cfg", "nsteps"), donate_argnames=("state",))
def run(state: State, step0, dt, seed, consts: Consts, cfg: Config, nsteps: int):
    """``nsteps`` steps after SPARTA timestep ``step0`` in one lax.scan; returns (state, Stats).
    ``dt`` and ``step0`` are traced, so one executable serves every chunk of every run."""
    steps = jnp.asarray(step0, jnp.int32) + 1 + jnp.arange(nsteps, dtype=jnp.int32)
    body = functools.partial(
        _scan_body, dt=jnp.asarray(dt, jnp.float64), seed=jnp.asarray(seed, jnp.uint64), consts=consts, cfg=cfg
    )
    return jax.lax.scan(body, state, steps)


@functools.partial(jax.jit, static_argnames=("cfg",))
def diagnostics(state: State, cfg: Config):
    """The decks' stats columns: np, temp (compute temp: sum m v^2 / (3 k N)), max per cell."""
    Cl = cfg.ncell_local
    alive = state.cell < Cl
    v2 = jnp.where(alive, state.vx**2 + state.vy**2 + state.vz**2, 0.0)
    n = jnp.sum(alive.astype(jnp.int32))
    temp = cfg.mass * jnp.sum(v2) / (3.0 * G3.BOLTZ * jnp.maximum(n, 1))
    npc = state.start[1 : Cl + 1] - state.start[:Cl]
    return {"np": n, "temp": temp, "maxpercell": jnp.max(npc)}


def moves(stats: Stats):
    """SPARTA's 'Particle moves': particles present at each move (survivors + inserts)."""
    n = np.asarray(stats.np, np.int64)
    prev = np.concatenate([[n[0] + np.asarray(stats.nexit, np.int64)[0] - np.asarray(stats.ninsert, np.int64)[0]], n[:-1]])
    return int(np.sum(prev + np.asarray(stats.ninsert, np.int64)))
