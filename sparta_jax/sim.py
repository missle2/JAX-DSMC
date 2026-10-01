"""SPARTA's per-step algorithm for the 2-D circle deck, in JAX (forward only, not differentiable).

One step, in SPARTA's order (``Update::run``):

1. ``emit``      -- ``FixEmitFace::perform_task_twopass`` (start_of_step fix).
2. ``move``      -- ``Update::move<2,1,0>``: straight move, global boundaries, cut-cell surface
                    collisions (``Geometry::line_line_intersect`` + ``SurfCollideDiffuse``).
3. ``sort``      -- ``Particle::sort``: per-cell particle lists (here a cell-sorted slot list).
4. ``collide``   -- ``Collide::collisions_one`` with ``CollideVSS`` (NTC, vremax, remain).
5. tallies       -- ``compute surf`` + ``fix ave/surf``, ``compute grid`` + ``fix ave/grid``
                    (optional, ``Config.tally``).

Layout (SoA, fixed capacity ``N``): a slot is alive iff ``cell < C`` (``DEAD = C``).  Slots never
move; ``plist`` is the slot list sorted by cell with dead slots last -- SPARTA's per-cell
``first``/``next`` lists in array form -- and its dead tail is the free list the emission writes
into (SPARTA appends at ``nlocal``).  The move is split into a dense straight move for particles
whose segment cannot touch a cut cell (SPARTA's own ``optmove`` logic) and SPARTA's cell-by-cell
walk for the others, run on a compacted lane (first stage on every lane entry, later stages on
the few entries still moving, re-compacted each stage).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as G
from . import rng

EPSSQ = 1.0e-16
MAXSTUCK = 20
# SPARTA face ids
INTERIOR, XLO, XHI, YLO, YHI = -1, 0, 1, 2, 3
# walk status
DONE, ACTIVE, DEAD = 0, 1, 2
# RNG streams (rng.step_key(seed, step, stream))
S_NINSERT, S_Y, S_BETA, S_ACCEPT, S_THETA, S_VR, S_DT = range(7)
S_SURF, S_DIFF1, S_DIFF2, S_DIFF3 = range(10, 14)
S_COLL = 20


@dataclasses.dataclass(frozen=True)
class Config:
    """Static configuration (hashable; jit static argument)."""

    # deck numbers (python floats, baked into the executable)
    nx: int
    ny: int
    xlo: float
    ylo: float
    xhi: float
    yhi: float
    dx: float
    dy: float
    dt: float
    fnum: float
    mass: float
    vscale: float
    vstream: tuple
    acc: float
    tsurf: float
    omega: float
    alpha: float
    prefactor: float  # CollideVSS prefactor: cxs (2 k Tref / mr)^(omega - 1/2) / Gamma(5/2 - omega)
    vremax0: float  # CollideVSS::vremax_init
    ntask: int
    nlines: int
    K: int  # max lines per cell
    # capacities / algorithm choices
    capacity: int = 53248
    lane: int = 5376  # lane of the cell walk, first stage (chunked if more particles are near)
    lane2: int = 1536  # lane of the second walk stage
    lane3: int = 128  # lane of the later walk stages (re-compacted every stage)
    per_task: int = 8  # max insertions per task per step: floor(ntarget) + 1
    trials: int = 64  # vectorised accept-reject trials of the emit/face normal-velocity sampler
    mover_sizes: tuple = (1024, 2048, 4096, 5120, 6144, 8192)  # partition sort buffers
    sort: str = "partition"  # "full" (sort all keys) | "partition" (O(N) + small sort of movers)
    fixed_stages: int = 0  # lane3 stages run unconditionally before the while_loop (GPU)
    max_walk: int = 1000  # safety cap on walk stages (SPARTA: unbounded, MAXSTUCK only)
    tally: bool = False  # surface + grid tallies (compute surf/grid + fix ave/*)

    @property
    def ncell(self):
        return self.nx * self.ny


class Consts(NamedTuple):
    cell_lines: jax.Array  # (C, 6 K) f64: p1x p1y p2x p2y nx ny for each line of the cell
    cell_ids: jax.Array  # (C, K) int32 line ids, -1 padded
    nsurf: jax.Array  # (C,) int32
    safe: jax.Array  # (C + 1,) int32: cut-free cell range ilo | ihi | jlo | jhi (8 bits each)
    cell_ij: jax.Array  # (C,) int32: i | j << 16 (column and row of the cell)
    volume: jax.Array  # (C,)
    task_cell: jax.Array  # (T,)
    task_lo: jax.Array
    task_hi: jax.Array
    task_ntarget: jax.Array


class Tally(NamedTuple):
    """compute surf (n, fx, fy) and compute grid (n, sum v, sum v^2) sums over the window."""

    surf: jax.Array  # (L, 3): count, momentum given to the surface -(m dv) in x and y
    grid: jax.Array  # (C, 5): count, sum vx, vy, vz, sum |v|^2
    nsample: jax.Array


class State(NamedTuple):
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array
    cell: jax.Array  # int32, DEAD = C
    pkey: jax.Array  # int32 cell << 16 | slot, sorted: SPARTA's per-cell lists (plist, skey)
    scode: jax.Array  # int32 cut-free cell range of the slot's cell (safe_table), kept by sort
    start: jax.Array  # (C + 2,) int32 bucket starts, bucket C = dead; start[C] = alive count
    vremax: jax.Array  # (C,)
    remain: jax.Array  # (C,)
    tally: Tally


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
    nwalk: jax.Array  # walk stages this step
    nstuck: jax.Array
    emit_fail: jax.Array  # emit trials exhausted (probability ~1e-14 per particle)
    overflow: jax.Array  # capacity exceeded (particles lost) -- must stay 0


def make(deck: G.Deck | None = None, **overrides):
    """Build (Config, Consts, Geometry) for the deck."""
    geo = G.build(deck or G.Deck())
    d = geo.deck
    sp = d.species
    mr = sp.mass * sp.mass / (sp.mass + sp.mass)
    cxs = sp.diam * sp.diam * math.pi
    prefactor = (
        cxs * (2.0 * G.BOLTZ * sp.tref / mr) ** (sp.omega - 0.5) / math.gamma(2.5 - sp.omega)
    )
    vremax0 = 2.0 * cxs * geo.vscale
    per_task = math.floor(float(geo.task_ntarget.max())) + 1
    cfg = Config(
        nx=d.nx,
        ny=d.ny,
        xlo=d.lo[0],
        ylo=d.lo[1],
        xhi=d.hi[0],
        yhi=d.hi[1],
        dx=geo.dx,
        dy=geo.dy,
        dt=d.dt,
        fnum=d.fnum,
        mass=sp.mass,
        vscale=geo.vscale,
        vstream=tuple(d.vstream),
        acc=d.acc,
        tsurf=d.tsurf,
        omega=sp.omega,
        alpha=sp.alpha,
        prefactor=prefactor,
        vremax0=vremax0,
        ntask=len(geo.task_cell),
        nlines=len(geo.p1),
        K=geo.csurfs.shape[1],
        per_task=per_task,
    )
    cfg = dataclasses.replace(cfg, **overrides)
    ids = geo.csurfs
    s = np.maximum(ids, 0)
    lines = np.stack(
        [geo.p1[s, 0], geo.p1[s, 1], geo.p2[s, 0], geo.p2[s, 1], geo.norm[s, 0], geo.norm[s, 1]],
        axis=2,
    )  # (C, K, 6)
    consts = Consts(
        cell_lines=jnp.asarray(lines.reshape(len(ids), -1)),
        cell_ids=jnp.asarray(ids, jnp.int32),
        nsurf=jnp.asarray(geo.nsurf, jnp.int32),
        safe=jnp.asarray(safe_table(geo), jnp.int32),
        cell_ij=jnp.asarray(
            np.arange(len(ids)) % geo.deck.nx + (np.arange(len(ids)) // geo.deck.nx << 16),
            jnp.int32,
        ),
        volume=jnp.asarray(geo.volume),
        task_cell=jnp.asarray(geo.task_cell, jnp.int32),
        task_lo=jnp.asarray(geo.task_lo),
        task_hi=jnp.asarray(geo.task_hi),
        task_ntarget=jnp.asarray(geo.task_ntarget),
    )
    return cfg, consts, geo


def ring_table(geo: G.Geometry):
    """Per cell: r such that every cell within Chebyshev distance r is free of surfaces
    (-1 for a cut cell).  A move whose start, end and mirrored-end cells lie within r of the
    start cell cannot meet a surface, so SPARTA's walk reduces to the optimised straight move."""
    nx, ny = geo.deck.nx, geo.deck.ny
    cut = np.argwhere((geo.cell_type == G.OVERLAP).reshape(ny, nx))
    r = np.full(nx * ny, max(nx, ny), np.int64)
    for c in range(nx * ny):
        i, j = c % nx, c // nx
        if len(cut):
            dist = np.max(np.abs(cut - np.array([j, i])), axis=1).min()
            r[c] = dist - 1
    return r


def safe_table(geo: G.Geometry):
    """Per cell the column and row range [ilo, ihi] x [jlo, jhi] of its cut-free Chebyshev ring
    (``ring_table``), clipped to the grid, packed as four 8-bit fields of an int32.  A straight
    move whose end cell (``floor`` indices clamped to the grid) lies in the range cannot meet a
    surface.  Cut cells get an empty range.  Row C (dead slots): everything."""
    nx, ny = geo.deck.nx, geo.deck.ny
    assert max(nx, ny) <= 127, "8-bit cell index fields"
    r = ring_table(geo)
    out = np.empty(nx * ny + 1, np.int64)
    for c in range(nx * ny):
        i, j = c % nx, c // nx
        if r[c] < 0:
            f = (1, 0, 1, 0)
        else:
            f = (max(i - r[c], 0), min(i + r[c], nx - 1), max(j - r[c], 0), min(j + r[c], ny - 1))
        out[c] = f[0] | f[1] << 8 | f[2] << 16 | f[3] << 24
    out[-1] = 0 | 127 << 8 | 0 << 16 | 127 << 24
    return out.astype(np.int32)


def empty_state(cfg: Config) -> State:
    """``create_box`` without ``create_particles``: the deck starts from an empty box."""
    N, C = cfg.capacity, cfg.ncell
    assert N <= 65536, "packed int32 sort keys need N <= 2^16"
    start = jnp.zeros(C + 2, jnp.int32).at[C + 1].set(N)
    tally = Tally(
        surf=jnp.zeros((cfg.nlines, 3)), grid=jnp.zeros((C, 5)), nsample=jnp.zeros((), jnp.int32)
    )
    return State(
        x=jnp.zeros(N),
        y=jnp.zeros(N),
        vx=jnp.zeros(N),
        vy=jnp.zeros(N),
        vz=jnp.zeros(N),
        cell=jnp.full(N, C, jnp.int32),
        pkey=C * 65536 + jnp.arange(N, dtype=jnp.int32),
        scode=jnp.zeros(N, jnp.int32),
        start=start,
        vremax=jnp.full(C, cfg.vremax0),
        remain=jnp.zeros(C),
        tally=tally,
    )


def _put(slot, arr, val):
    return arr.at[slot].set(val, mode="drop")


def _isum(a):
    return jnp.sum(a, dtype=jnp.int32)


def cumsum(m, block: int = 8):
    """Inclusive int32 prefix sum of a 1-D int32 array.

    XLA:CPU lowers jnp.cumsum to a tree of reduce-windows over the whole array (~3.5 ns per
    element at N = 53k).  Two levels -- prefix sums inside blocks of 8 (a reduce-window of
    width 8), the same recursively on the block totals, then one broadcast add -- cost about
    half (measured with the mask producer and the compaction scatter attached)."""
    n = m.shape[0]
    if n <= 4 * block or n % block:
        return jnp.cumsum(m, dtype=jnp.int32)
    inner = jnp.cumsum(m.reshape(-1, block), axis=1, dtype=jnp.int32)
    tot = inner[:, -1]
    off = materialise(cumsum(tot, block) - tot)  # else every consumer re-derives all levels
    return (inner + off[:, None]).reshape(n)


def materialise(t):
    """``t`` stored as its own array.  XLA fuses a small computed table into every consumer;
    an N-sized fusion that gathers from it then recomputes the table per element (e.g. a
    401-entry cumsum: 3 us -> 150 us).  A scatter is always a fusion root, and
    optimization_barrier is expanded away before fusion on XLA:CPU."""
    return jnp.zeros_like(t).at[jnp.arange(t.shape[0], dtype=jnp.int32)].set(t)


_MAGIC = 6755399441055744.0  # 1.5 * 2^52: adding it rounds a double to an integer in its low bits


def floor_int(v):
    """int32(floor(v)) for |v| < 2^31, exact (round to nearest by the 1.5 * 2^52 trick, then one
    step down where that rounded up).  XLA:CPU's float -> int conversion does not vectorise
    (~2.5 ns per element); this does."""
    k = jax.lax.bitcast_convert_type(v + _MAGIC, jnp.int64).astype(jnp.int32)
    # (int -> float converts vectorise; (v + M) - M would be folded to v by the compiler)
    return k - (k.astype(jnp.float64) > v).astype(jnp.int32)


def _per_cell(u, h):
    """u / h as SPARTA's optmove_cell computes it; a multiplication when 1/h is exact."""
    return u * (1.0 / h) if math.frexp(h)[0] == 0.5 else u / h


# ---------------------------------------------------------------------------------------------
# 1. fix emit/face (twopass)
# ---------------------------------------------------------------------------------------------


def emit(state: State, seed, istep, consts: Consts, cfg: Config):
    """FixEmitFace::perform_task_twopass for the xlo face (perspecies, one species).

    Pass 1: ninsert = int(ntarget + U) per task.  Pass 2, per particle: x on the face, y uniform
    on the task face, normal velocity by SPARTA's accept-reject sampler (Bird 1994 eq. 12.5),
    tangential velocities by Box-Muller in polar form, dtremain = dt * U.  The particle joins
    this step's move with that dtremain (PINSERT).
    """
    T, Mt, R = cfg.ntask, cfg.per_task, cfg.trials
    N = cfg.capacity
    key = functools.partial(rng.step_key, seed, istep)
    q = jnp.arange(T * Mt, dtype=jnp.int32)
    u_n = rng.uniform(key(S_NINSERT), jnp.arange(T, dtype=jnp.int32))
    ninsert = jnp.floor(consts.task_ntarget + u_n).astype(jnp.int32)
    valid = (jnp.arange(Mt, dtype=jnp.int32)[None, :] < ninsert[:, None]).reshape(-1)
    lo = jnp.repeat(consts.task_lo, Mt)
    hi = jnp.repeat(consts.task_hi, Mt)
    cell = jnp.repeat(consts.task_cell, Mt)
    y = lo + rng.uniform(key(S_Y), q) * (hi - lo)
    # normal velocity: the inner do-while (beta = 6U - 3 until beta + s >= 0) is exactly a
    # uniform draw on [-s, 3); the outer do-while accepts with probability f(beta) <= 1.
    s = cfg.vstream[0] / cfg.vscale
    r = math.sqrt(s * s + 2.0)
    qt = q[:, None] * R + jnp.arange(R, dtype=jnp.int32)[None, :]
    beta = -s + (3.0 + s) * rng.uniform(key(S_BETA), qt)
    f = 2.0 * (beta + s) / (s + r) * jnp.exp(0.5 + (0.5 * s) * (s - r) - beta * beta)
    ok = f >= rng.uniform(key(S_ACCEPT), qt)
    first = jnp.argmax(ok, axis=1)
    beta1 = jnp.take_along_axis(beta, first[:, None], axis=1)[:, 0]
    fail = valid & ~jnp.any(ok, axis=1)
    vx = beta1 * cfg.vscale + cfg.vstream[0]
    theta = 2.0 * math.pi * rng.uniform(key(S_THETA), q)
    vr = cfg.vscale * jnp.sqrt(-jnp.log1p(-rng.uniform(key(S_VR), q)))
    vy = vr * jnp.sin(theta) + cfg.vstream[1]
    vz = vr * jnp.cos(theta) + cfg.vstream[2]
    dtr = cfg.dt * rng.uniform(key(S_DT), q)
    # add_particle: the next free slots are the dead tail of the sorted slot list
    rank = jnp.cumsum(valid.astype(jnp.int32)) - 1
    pos = state.start[cfg.ncell] + rank
    ok_slot = valid & (pos < N)
    slot = jnp.where(ok_slot, state.pkey[jnp.minimum(pos, N - 1)] & 0xFFFF, N)
    put = functools.partial(_put, slot)
    state = state._replace(
        x=put(state.x, jnp.full(T * Mt, cfg.xlo)),
        y=put(state.y, y),
        vx=put(state.vx, vx),
        vy=put(state.vy, vy),
        vz=put(state.vz, vz),
        cell=put(state.cell, cell + (cfg.ncell + 1)),  # near: the walk moves inserts (dt * U)
    )
    return state, (slot, dtr), _isum(ok_slot), _isum(fail), _isum(valid & ~ok_slot)


# ---------------------------------------------------------------------------------------------
# 2. move
# ---------------------------------------------------------------------------------------------


def _floor_index(u, n):
    return jnp.clip(floor_int(u), 0, n - 1)


def _field(code, k):
    """8-bit field k of an int32 code."""
    return (code >> (8 * k)) & 0xFF


def dense_cell(state: State, cfg: Config):
    """End cell of the straight move of every slot that provably cannot meet a surface or a
    reflecting wall; the code ``C + 1 + cell`` ('near') for every other alive slot.

    A move whose end cell lies in the cut-free Chebyshev range of its start cell (``scode``, the
    slot's cached copy of ``safe_table``) and that does not leave through a y face reduces to
    SPARTA's Update::optmove: deletion through the x faces (OUTFLOW) and the cell holding xnew.
    Near slots go to the cell walk, which decodes the start cell from the code; this step's
    inserts arrive coded near from ``emit`` (their move time is not dt).  One in-place
    elementwise pass that streams the slot arrays: no table lookups and no reductions (exits of
    dense slots are counted by particle conservation in ``step``).
    """
    C, nx, ny, dt = cfg.ncell, cfg.nx, cfg.ny, cfg.dt
    xn = state.x + dt * state.vx
    yn = state.y + dt * state.vy
    exit_ = (xn < cfg.xlo) | (xn >= cfg.xhi)
    i1 = _floor_index(_per_cell(xn - cfg.xlo, cfg.dx), nx)
    j1 = _floor_index(_per_cell(yn - cfg.ylo, cfg.dy), ny)
    ilo, ihi, jlo, jhi = (_field(state.scode, k) for k in range(4))
    safe = (i1 >= ilo) & (i1 <= ihi) & (j1 >= jlo) & (j1 <= jhi) & (yn >= cfg.ylo) & (yn < cfg.yhi)
    cell = state.cell
    moved = jnp.where(safe, jnp.where(exit_, C, j1 * nx + i1), C + 1 + cell)
    return jnp.where(cell < C, moved, cell).astype(jnp.int32)  # dead (C), near (> C) unchanged


def dense_move(state: State, cfg: Config):
    """The dense part of the move: ``dense_cell``, then x and y of the dense slots advanced in
    place (they read the stored cell; near slots keep their start point for the walk)."""
    cell = dense_cell(state, cfg)
    keep = cell > cfg.ncell
    x = jnp.where(keep, state.x, state.x + cfg.dt * state.vx)
    y = jnp.where(keep, state.y, state.y + cfg.dt * state.vy)
    return state._replace(x=x, y=y, cell=cell)


class Walk(NamedTuple):
    """Per-lane state of Update::move's while(1) loop.  The integer state is packed into two
    int32 words (fewer arrays to gather and scatter per stage):
    ``ij = i | j << 16`` (cell column and row) and
    ``flags = status | stuck << 2 | (exclude + 1) << 8``."""

    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array | None  # None when acc == 0 (specular: vz never changes)
    xn: jax.Array  # xnew
    yn: jax.Array
    dtrem: jax.Array
    ij: jax.Array
    flags: jax.Array


def _flags(status, stuck, exclude):
    return status | (stuck << 2) | ((exclude + 1) << 8)


def _status(flags):
    return flags & 3


def _line_line_intersect(sx, sy, ex, ey, ax, ay, bx, by, nx_, ny_):
    """Geometry::line_line_intersect (2-D): hit, param, point, and side == OUTSIDE."""
    dotstart = nx_ * (sx - ax) + ny_ * (sy - ay)
    dotstop = nx_ * (ex - ax) + ny_ * (ey - ay)
    miss = (
        ((dotstart < 0.0) & (dotstop < 0.0))
        | ((dotstart > 0.0) & (dotstop > 0.0))
        | ((dotstart == 0.0) & (dotstop == 0.0))
    )
    den = nx_ * (ex - sx) + ny_ * (ey - sy)
    param = (nx_ * (ax - sx) + ny_ * (ay - sy)) / jnp.where(den == 0.0, 1.0, den)
    miss = miss | (param < 0.0) | (param > 1.0)
    px = sx + param * (ex - sx)
    py = sy + param * (ey - sy)
    edx, edy = bx - ax, by - ay
    miss = miss | (edx * (px - ax) + edy * (py - ay) < -EPSSQ)
    miss = miss | (edx * (px - bx) + edy * (py - by) > EPSSQ)
    return ~miss, param, px, py, dotstart > 0.0


def _face(w: Walk, cfg: Config):
    """Cell bounds and the (outface, frac) of update.cpp for the current cell."""
    i, j = w.ij & 0xFFFF, w.ij >> 16
    fi, fj = i.astype(jnp.float64), j.astype(jnp.float64)
    lox = cfg.xlo + fi * cfg.dx
    hix = cfg.xlo + (fi + 1.0) * cfg.dx
    loy = cfg.ylo + fj * cfg.dy
    hiy = cfg.ylo + (fj + 1.0) * cfg.dy
    fx_lo = w.xn < lox
    fx_hi = ~fx_lo & (w.xn >= hix)
    dxm = w.xn - w.x
    fracx = jnp.where(dxm != 0.0, (jnp.where(fx_lo, lox, hix) - w.x) / dxm, 0.0)
    frac = jnp.where(fx_lo | fx_hi, jnp.clip(fracx, 0.0, 1.0), 1.0)
    outface = jnp.where(fx_lo, XLO, jnp.where(fx_hi, XHI, INTERIOR)).astype(jnp.int32)
    fy_lo = w.yn < loy
    fy_hi = ~fy_lo & (w.yn >= hiy)
    dym = w.yn - w.y
    fracy = jnp.clip(jnp.where(dym != 0.0, (jnp.where(fy_lo, loy, hiy) - w.y) / dym, 0.0), 0.0, 1.0)
    takey = (fy_lo | fy_hi) & (fracy < frac)
    frac = jnp.where(takey, fracy, frac)
    outface = jnp.where(takey, jnp.where(fy_lo, YLO, YHI), outface).astype(jnp.int32)
    return (i, j, lox, hix, loy, hiy), outface, frac


def _snap(px, py, outface, box):
    _i, _j, lox, hix, loy, hiy = box
    px = jnp.where(outface == XLO, lox, jnp.where(outface == XHI, hix, px))
    py = jnp.where(outface == YLO, loy, jnp.where(outface == YHI, hiy, py))
    return px, py


def _argmin_reducer(a, b):
    """Variadic-reduce comparator: smaller param wins, ties go to the lower line slot (SPARTA
    keeps the first line with param < minparam); the passengers ride with the winner."""
    pa, ka = a[0], a[1]
    pb, kb = b[0], b[1]
    take_a = (pa < pb) | ((pa == pb) & (ka <= kb))
    return tuple(jnp.where(take_a, x, y) for x, y in zip(a, b, strict=True))


def _surface_hit(w: Walk, icell, box, outface, frac, consts: Consts, cfg: Config):
    """Earliest OUTSIDE hit among the cell's lines on the in-cell part of the path.

    The K line tests run on a (B, K) block and are reduced with one variadic reduce that also
    carries the face result (frac, outface), the clipped end point and the winning line's normal
    and id: a reduction is a fusion root on XLA:CPU, so everything downstream reads materialised
    values instead of every output fusion recomputing the line tests (optimization_barrier is
    expanded away before fusion), and no gather is needed to fetch the winner's normal.
    """
    crossing = outface != INTERIOR
    cx = jnp.where(crossing, w.x + frac * (w.xn - w.x), w.xn)
    cy = jnp.where(crossing, w.y + frac * (w.yn - w.y), w.yn)
    cx, cy = _snap(cx, cy, outface, box)
    K = cfg.K
    rows = consts.cell_lines[icell].reshape(-1, K, 6)  # (B, K, 6)
    ids = consts.cell_ids[icell]  # (B, K)
    col = functools.partial(jnp.expand_dims, axis=1)
    ax, ay, bx, by, lnx, lny = (rows[:, :, m] for m in range(6))
    hit, param, _px, _py, outside = _line_line_intersect(
        col(w.x), col(w.y), col(cx), col(cy), ax, ay, bx, by, lnx, lny
    )
    exclude = (w.flags >> 8) - 1
    ok = (ids >= 0) & (ids != col(exclude)) & hit & outside
    param = jnp.where(ok, param, 2.0)
    kk = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), ids.shape)
    bc = functools.partial(jnp.broadcast_to, shape=ids.shape)
    of = col(outface.astype(jnp.int32))
    ops = (param, kk, bc(col(frac)), bc(of), bc(col(cx)), bc(col(cy)), lnx, lny, ids)
    f0, i0 = jnp.float64(0.0), jnp.int32(0)
    init = (jnp.float64(3.0), jnp.int32(K), f0, i0, f0, f0, f0, f0, jnp.int32(-1))
    minparam, _k, frac, outface, cx, cy, nrx, nry, msurf = jax.lax.reduce(
        ops, init, _argmin_reducer, (1,)
    )
    minsurf = jnp.where(minparam < 2.0, msurf, -1)
    mx = w.x + minparam * (cx - w.x)
    my = w.y + minparam * (cy - w.y)
    return (minparam, minsurf, mx, my, nrx, nry), outface, frac


def _walk_stage(w: Walk, slot, seed, istep, it, surf, consts: Consts, cfg: Config):
    """One pass of Update::move's per-particle while(1) loop over every ACTIVE lane entry.

    Returns the new lane state, the stage's counts (nscoll, nscheck, nexit, nbound, nstuck) and
    the surface tally.  ``slot`` (the RNG counter of SurfCollideDiffuse) is only read when
    acc > 0: with acc = 0 the draw ``U > acc`` is always true."""
    nx, ny = cfg.nx, cfg.ny
    status, stuck = _status(w.flags), (w.flags >> 2) & 63
    act = status == ACTIVE
    box, outface, frac = _face(w, cfg)
    i, j, _lox, _hix, loy, hiy = box
    icell = i + j * nx
    hitres, outface, frac = _surface_hit(w, icell, box, outface, frac, consts, cfg)
    minparam, minsurf, mx, my, nrx, nry = hitres
    crossing = outface != INTERIOR
    cflag = act & (minsurf >= 0)

    svx, svy, svz = _surface_collide(w, nrx, nry, slot, seed, istep, it, cfg)
    dtrem_c = w.dtrem * (1.0 - minparam * frac)
    stuck_c = jnp.where(minparam <= 1.0e-14, stuck + 1, 0)
    discard_stuck = cflag & (stuck_c >= MAXSTUCK)
    if cfg.tally:
        m = cfg.mass
        rows = jnp.where(cflag, minsurf, cfg.nlines)
        upd = jnp.stack([jnp.ones_like(w.x), -m * (svx - w.vx), -m * (svy - w.vy)], axis=1)
        surf = surf.at[rows].add(upd, mode="drop")

    # --- no surface collision: finish inside the cell, or cross a face ---
    interior = act & ~cflag & ~crossing
    cross = act & ~cflag & crossing
    fx, fy = _snap(w.x + frac * (w.xn - w.x), w.y + frac * (w.yn - w.y), outface, box)
    at_x = ((outface == XLO) & (i == 0)) | ((outface == XHI) & (i == nx - 1))
    at_ylo = (outface == YLO) & (j == 0)
    at_y = at_ylo | ((outface == YHI) & (j == ny - 1))
    exits = cross & at_x  # OUTFLOW: Domain::collide -> PDISCARD
    reflect = cross & at_y  # REFLECT: Domain::collide mirrors xnew, flips v, same cell
    step_ij = jnp.where(
        outface == XLO,
        -1,
        jnp.where(outface == XHI, 1, jnp.where(outface == YLO, -(1 << 16), 1 << 16)),
    )
    newcell = cross & ~exits & ~reflect
    ynr = jnp.where(at_ylo, 2.0 * loy - w.yn, 2.0 * hiy - w.yn)
    status = jnp.where(exits | discard_stuck, DEAD, jnp.where(interior, DONE, status))
    exclude = (w.flags >> 8) - 1
    flags = _flags(
        status,
        jnp.where(cflag, stuck_c, stuck),
        jnp.where(cflag, minsurf, jnp.where(cross, -1, exclude)),
    )
    w2 = Walk(
        x=jnp.where(cflag, mx, jnp.where(interior, w.xn, jnp.where(cross, fx, w.x))),
        y=jnp.where(cflag, my, jnp.where(interior, w.yn, jnp.where(cross, fy, w.y))),
        vx=jnp.where(cflag, svx, w.vx),
        vy=jnp.where(cflag, svy, jnp.where(reflect, -w.vy, w.vy)),
        vz=jnp.where(cflag, svz, w.vz) if cfg.acc > 0.0 else w.vz,
        xn=jnp.where(cflag, mx + dtrem_c * svx, w.xn),
        yn=jnp.where(cflag, my + dtrem_c * svy, jnp.where(reflect, ynr, w.yn)),
        dtrem=jnp.where(cflag, dtrem_c, jnp.where(cross, w.dtrem * (1.0 - frac), w.dtrem)),
        ij=jnp.where(newcell, w.ij + step_ij, w.ij),
        flags=flags.astype(jnp.int32),
    )
    # nscoll, nscheck, nexit, nbound, nstuck in one (B, 5) reduction (on XLA:CPU each
    # reduction is a tree of reduce-windows costing a few us)
    per = [cflag, None, exits, reflect, discard_stuck]
    per = [
        jnp.where(act, consts.nsurf[icell], 0) if a is None else a.astype(jnp.int32) for a in per
    ]
    cnt = jnp.sum(jnp.stack(per, axis=1), axis=0, dtype=jnp.int32)
    return w2, cnt, surf


def _surface_collide(w: Walk, nrx, nry, slot, seed, istep, it, cfg: Config):
    """SurfCollideDiffuse::collide: one uniform draw; specular (reflect3) if U > acc, else
    diffuse.  With acc = 0 the draw is always > acc, so it is skipped."""
    vdot = w.vx * nrx + w.vy * nry
    svx, svy, svz = w.vx - 2.0 * vdot * nrx, w.vy - 2.0 * vdot * nry, w.vz
    if cfg.acc > 0.0:
        ctr = slot * 4096 + it
        specular = rng.uniform(rng.step_key(seed, istep, S_SURF), ctr) > cfg.acc
        dvx, dvy, dvz = _diffuse(w.vx, w.vy, w.vz, nrx, nry, seed, istep, ctr, cfg)
        svx = jnp.where(specular, svx, dvx)
        svy = jnp.where(specular, svy, dvy)
        svz = jnp.where(specular, svz, dvz)
    return svx, svy, svz


def _diffuse(vx, vy, vz, nrx, nry, seed, istep, ctr, cfg: Config):
    """SurfCollideDiffuse::diffuse, diffuse branch (not used by the deck: acc = 0)."""
    vrm = math.sqrt(2.0 * G.BOLTZ * cfg.tsurf / cfg.mass)
    u1 = rng.uniform(rng.step_key(seed, istep, S_DIFF1), ctr)
    u2 = rng.uniform(rng.step_key(seed, istep, S_DIFF2), ctr)
    u3 = rng.uniform(rng.step_key(seed, istep, S_DIFF3), ctr)
    vperp = vrm * jnp.sqrt(-jnp.log1p(-u1))
    theta = 2.0 * math.pi * u2
    vtan = vrm * jnp.sqrt(-jnp.log1p(-u3))
    vt1, vt2 = vtan * jnp.sin(theta), vtan * jnp.cos(theta)
    dot = vx * nrx + vy * nry
    t1x, t1y, t1z = vx - dot * nrx, vy - dot * nry, vz
    ln = jnp.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    zero = ln == 0.0  # normal incidence: SPARTA picks a random tangent; any tangent will do
    t1x, t1y = jnp.where(zero, -nry, t1x), jnp.where(zero, nrx, t1y)
    t1z = jnp.where(zero, 0.0, t1z)
    ln = jnp.where(zero, 1.0, ln)
    t1x, t1y, t1z = t1x / ln, t1y / ln, t1z / ln
    t2x, t2y, t2z = nry * t1z, -nrx * t1z, nrx * t1y - nry * t1x  # norm x tangent1
    return (
        vperp * nrx + vt1 * t1x + vt2 * t2x,
        vperp * nry + vt1 * t1y + vt2 * t2y,
        vt1 * t1z + vt2 * t2z,
    )


def _compact(mask, size):
    """Indices of the first ``size`` set entries of mask (len(mask) where there are fewer)."""
    n = mask.shape[0]
    r = cumsum(mask.astype(jnp.int32)) - 1
    return (
        jnp.full(size, n, jnp.int32)
        .at[jnp.where(mask & (r < size), r, size)]
        .set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )


def _take(a, idx, fill):
    return None if a is None else jnp.take(a, idx, axis=0, mode="fill", fill_value=fill)


def _later_stage(carry, slots, seed, istep, size, consts: Consts, cfg: Config):
    """Next walk stage on the first ``size`` ACTIVE entries of the lane ``w``."""
    w, it, cnt, surf = carry
    idx = _compact(_status(w.flags) == ACTIVE, size)
    part = Walk(*(_take(a, idx, 0) for a in w))  # empty entry: flags 0 = DONE
    slot = _take(slots, idx, cfg.capacity) if cfg.acc > 0.0 else None
    part, c, surf = _walk_stage(part, slot, seed, istep, it, surf, consts, cfg)
    w = Walk(
        *(
            None if a is None else a.at[idx].set(b, mode="drop")
            for a, b in zip(w, part, strict=True)
        )
    )
    return w, it + 1, cnt + c, surf


def _later_cond(carry, cfg: Config):
    w, it, _cnt, _surf = carry
    return jnp.any(_status(w.flags) == ACTIVE) & (it < cfg.max_walk)


def _lane_gather(state: State, c, rank, near, inserts, consts: Consts, cfg: Config):
    """Slots and start state of the ``c``-th block of ``lane`` near particles.  Every move
    lasts dt except this step's inserts ``(slot, dtremain)`` (no per-slot dt array)."""
    B, N, C = cfg.lane, cfg.capacity, cfg.ncell
    local = rank - c * B
    slots = (
        jnp.full(B, N, jnp.int32)
        .at[jnp.where(near & (local >= 0) & (local < B), local, B)]
        .set(jnp.arange(N, dtype=jnp.int32), mode="drop")
    )
    valid = slots < N
    g = jnp.minimum(slots, N - 1)
    x, y, vx, vy = state.x[g], state.y[g], state.vx[g], state.vy[g]
    cell = jnp.where(valid, state.cell[g] - (C + 1), 0)  # near code C + 1 + start cell
    dtr = jnp.full(B, cfg.dt)
    if inserts is not None:
        islot, idtr = inserts
        lpos = rank[jnp.minimum(islot, N - 1)] - c * B  # inserts are near: their lane rank
        hit = (islot < N) & (lpos >= 0) & (lpos < B)
        dtr = dtr.at[jnp.where(hit, lpos, B)].set(idtr, mode="drop")
    w = Walk(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        vz=state.vz[g] if cfg.acc > 0.0 else None,
        xn=x + dtr * vx,
        yn=y + dtr * vy,
        dtrem=dtr,
        ij=consts.cell_ij[cell],
        flags=jnp.where(valid, _flags(ACTIVE, 0, -1), _flags(DONE, 0, -1)).astype(jnp.int32),
    )
    return slots, w


def _lane_walk(carry, slots, w: Walk, seed, istep, consts: Consts, cfg: Config):
    """Walk one lane block to the end of its moves and write the results to the slots.

    Stage 1 runs on the whole lane; later stages on the first ``lane2`` (then ``lane3``) entries
    still ACTIVE, gathered from and scattered back to the lane."""
    state, c, cnt, iters, surf = carry
    w, cnt1, surf = _walk_stage(w, slots, seed, istep, jnp.int32(0), surf, consts, cfg)
    stage = functools.partial(
        _later_stage, slots=slots, seed=seed, istep=istep, consts=consts, cfg=cfg
    )
    carry = stage((w, jnp.int32(1), cnt1, surf), size=cfg.lane2)
    small = functools.partial(stage, size=cfg.lane3)
    for _ in range(cfg.fixed_stages):  # unconditional (masked) stages: no loop-predicate sync
        carry = small(carry)
    w, it, cnt_w, surf = jax.lax.while_loop(functools.partial(_later_cond, cfg=cfg), small, carry)
    put = functools.partial(_put, slots)
    icell = (w.ij & 0xFFFF) + (w.ij >> 16) * cfg.nx
    state = state._replace(
        x=put(state.x, w.x),
        y=put(state.y, w.y),
        vx=put(state.vx, w.vx),
        vy=put(state.vy, w.vy),
        vz=put(state.vz, w.vz) if cfg.acc > 0.0 else state.vz,
        cell=put(state.cell, jnp.where(_status(w.flags) == DEAD, cfg.ncell, icell)),
    )
    return state, c + 1, cnt + cnt_w, iters + it, surf


def _lane_chunk(carry, rank, near, inserts, seed, istep, consts: Consts, cfg: Config):
    """Walk the ``c``-th block of ``lane`` near particles (they still hold their start state)."""
    slots, w = _lane_gather(carry[0], carry[1], rank, near, inserts, consts, cfg)
    return _lane_walk(carry, slots, w, seed, istep, consts, cfg)


def _lane_cond(carry, nchunk):
    return carry[1] < nchunk


def move(state: State, seed, istep, consts: Consts, cfg: Config, inserts=None):
    """Update::move<2,1,0> for all particles (dense straight move + cell walk on the lane).
    ``inserts``: this step's emitted ``(slot, dtremain)`` (``emit``); other moves last dt."""
    with jax.named_scope("move_dense"):
        new = dense_move(state, cfg)
    with jax.named_scope("move_select"):
        near = new.cell > cfg.ncell
        rank = cumsum(near.astype(jnp.int32)) - 1
        nnear = rank[-1] + 1
        nchunk = (nnear + cfg.lane - 1) // cfg.lane
    with jax.named_scope("move_walk"):
        carry = (new, jnp.int32(0), jnp.zeros(5, jnp.int32), jnp.int32(0), state.tally.surf)
        body = functools.partial(
            _lane_chunk,
            rank=rank,
            near=near,
            inserts=inserts,
            seed=seed,
            istep=istep,
            consts=consts,
            cfg=cfg,
        )
        carry = body(carry)  # first block unconditionally; more blocks only on lane overflow
        new, _c, cnt, iters, surf = jax.lax.while_loop(
            functools.partial(_lane_cond, nchunk=nchunk), body, carry
        )
    counts = {
        "nscoll": cnt[0],
        "nscheck": cnt[1],
        "nexit_walk": cnt[2],  # dense exits: by conservation in ``step``
        "nbound": cnt[3],
        "nstuck": cnt[4],
        "nlane": nnear,
        "nwalk": iters,
    }
    return new, surf, counts


# ---------------------------------------------------------------------------------------------
# 3. sort (per-cell particle lists)
# ---------------------------------------------------------------------------------------------


def _starts(counts):
    return jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])


def plist(state: State):
    """Slots sorted by cell, dead slots last (SPARTA's per-cell lists in array form)."""
    return state.pkey & 0xFFFF


def skey(state: State):
    """Cell of ``plist(state)[p]``."""
    return state.pkey >> 16


def _full_sort(cell, cfg: Config):
    N, C = cfg.capacity, cfg.ncell
    pkey = jnp.sort(cell * 65536 + jnp.arange(N, dtype=jnp.int32))
    counts = jnp.zeros(C + 1, jnp.int32).at[cell].add(1)
    return pkey, _starts(counts)


def sort_full(state: State, consts: Consts, cfg: Config):
    """Particle::sort as one int32 sort of packed (cell, slot) keys; dead slots sort last."""
    pkey, start = _full_sort(state.cell, cfg)
    scode = consts.safe[state.cell]
    return state._replace(pkey=pkey, start=start, scode=scode)


def sort_partition(state: State, consts: Consts, cfg: Config):
    """Particle::sort in O(N): slots that kept their cell keep their order (they were sorted
    last step); the few that changed cell ("movers": crossings, exits, inserts) are sorted
    alone and placed after the stayers of their new cell.  The mover sort is sized to the
    smallest of ``mover_sizes`` that holds the movers (lax.switch); more than the largest size
    falls back to the full sort.  The movers' cached ``scode`` is refreshed here."""
    N = cfg.capacity
    slot = state.pkey & 0xFFFF
    knew = state.cell[slot]  # new key at each sorted position
    stay = knew == state.pkey >> 16
    stay_i = stay.astype(jnp.int32)
    cs = cumsum(stay_i) - stay_i  # exclusive
    nmov = N - (cs[-1] + stay_i[-1])
    sizes = jnp.asarray(cfg.mover_sizes, jnp.int32)
    branch = jnp.sum(nmov > sizes, dtype=jnp.int32)  # == len(sizes): full sort
    branches = [
        functools.partial(_partition, size=m, consts=consts, cfg=cfg) for m in cfg.mover_sizes
    ]
    branches.append(functools.partial(_slow_sort, consts=consts, cfg=cfg))
    pkey, start, scode = jax.lax.switch(branch, branches, knew, stay, cs, state)
    return state._replace(pkey=pkey, start=start, scode=scode)


def _slow_sort(knew, stay, cs, state: State, consts: Consts, cfg: Config):
    pkey, start = _full_sort(state.cell, cfg)
    return pkey, start, consts.safe[state.cell]


def _partition(knew, stay, cs, state: State, size, consts: Consts, cfg: Config):
    N, C, M = cfg.capacity, cfg.ncell, size
    pos = jnp.arange(N, dtype=jnp.int32)
    slot = state.pkey & 0xFFFF
    # movers: compacted (mover rank = position - stayers before it), as packed new keys
    mval = (
        jnp.full(M, (C + 1) * 65536, jnp.int32)
        .at[jnp.where(stay, M, pos - cs)]
        .set(knew * 65536 + slot, mode="drop")
    )
    # stayers per old bucket = stayers before its end - stayers before its start
    cs_end = jnp.concatenate([cs, (cs[-1] + stay[-1]).astype(jnp.int32)[None]])
    nstay = cs_end[state.start[1:]] - cs_end[state.start[: C + 1]]
    hist = jnp.zeros(C + 2, jnp.int32).at[mval >> 16].add(1)[: C + 1]  # movers per new bucket
    start = _starts(nstay + hist)
    # stayers keep their order: destination = new bucket start + rank among the bucket's
    # stayers.  (offs is built from materialised C-sized arrays, so the N-sized fusion
    # that gathers it does not recompute it per element.)
    offs = materialise(start[: C + 1] - _starts(nstay)[: C + 1])
    dest_stay = jnp.where(stay, offs[knew] + cs, N)
    pkey = jnp.zeros(N, jnp.int32).at[dest_stay].set(state.pkey, mode="drop")
    # movers: sorted by (new bucket, slot), placed after the stayers of their bucket
    ms = jnp.sort(mval)
    mk = ms >> 16
    ok = mk <= C
    mkc = jnp.minimum(mk, C)
    mfirst = _starts(hist)[: C + 1]  # first sorted mover of each bucket
    base = materialise(start[: C + 1] + nstay - mfirst)
    dest_mov = jnp.where(ok, base[mkc] + jnp.arange(M, dtype=jnp.int32), N)
    pkey = pkey.at[dest_mov].set(ms, mode="drop")
    # cached cut-free range of the movers' new cells
    mslot = jnp.where(ok, ms & 0xFFFF, N)
    scode = state.scode.at[mslot].set(consts.safe[mkc], mode="drop")
    return pkey, start, scode


# ---------------------------------------------------------------------------------------------
# 4. collisions: Collide::collisions_one + CollideVSS
# ---------------------------------------------------------------------------------------------


def _collide_attempt(carry, state: State, npc, nattempt, seed, istep, cfg: Config):
    """Attempt number k in every cell that has more than k attempts (cells are independent)."""
    vx, vy, vz, vremax, k, ncoll = carry
    C, N = cfg.ncell, cfg.capacity
    key = rng.step_key(seed, istep, S_COLL)
    ctr = (k * C + jnp.arange(C, dtype=jnp.int32)[None, :]) * 5 + jnp.arange(5)[:, None]
    u = rng.uniform(key, ctr)
    act = k < nattempt
    npf = npc.astype(jnp.float64)
    # i = np * U; j = np * U until j != i  (== uniform over the other np - 1)
    i = jnp.minimum((npf * u[0]).astype(jnp.int32), npc - 1)
    j = jnp.minimum(((npf - 1.0) * u[1]).astype(jnp.int32), npc - 2)
    j = j + (j >= i)
    base = state.start[:C]
    pi = state.pkey[jnp.clip(base + i, 0, N - 1)] & 0xFFFF
    pj = state.pkey[jnp.clip(base + j, 0, N - 1)] & 0xFFFF
    vix, viy, viz = vx[pi], vy[pi], vz[pi]
    vjx, vjy, vjz = vx[pj], vy[pj], vz[pj]
    du, dv, dw = vix - vjx, viy - vjy, viz - vjz
    vr2 = du * du + dv * dv + dw * dw
    # test_collision
    vre = vr2 ** (1.0 - cfg.omega) * cfg.prefactor
    vremax = jnp.where(act, jnp.maximum(vre, vremax), vremax)
    accept = act & (vremax > 0.0) & (vre / jnp.where(vremax > 0.0, vremax, 1.0) >= u[2])
    # SCATTER_TwoBodyScattering, VSS branch (1/alpha != 1); argon: no internal energy, so the
    # post-collision etrans equals the pre-collision one and scale = 1
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
    return vx, vy, vz, vremax, k + 1, ncoll + _isum(accept)


def _collide_cond(carry, kmax):
    return carry[4] < kmax


def collide(state: State, seed, istep, consts: Consts, cfg: Config):
    """NTC with per-cell vremax and the fractional 'remain' carry (collide_modify default)."""
    C = cfg.ncell
    npc = state.start[1 : C + 1] - state.start[:C]
    npf = npc.astype(jnp.float64)
    live = npc > 1
    vol = jnp.where(consts.volume > 0.0, consts.volume, 1.0)
    attempt = 0.5 * npf * (npf - 1.0) * state.vremax * cfg.dt * cfg.fnum / vol + state.remain
    fl = jnp.floor(attempt)
    nattempt = jnp.where(live, fl, 0.0).astype(jnp.int32)
    remain = jnp.where(live, attempt - fl, state.remain)
    kmax = jnp.max(nattempt)
    carry = (state.vx, state.vy, state.vz, state.vremax, jnp.int32(0), jnp.int32(0))
    body = functools.partial(
        _collide_attempt, state=state, npc=npc, nattempt=nattempt, seed=seed, istep=istep, cfg=cfg
    )
    vx, vy, vz, vremax, _k, ncoll = jax.lax.while_loop(
        functools.partial(_collide_cond, kmax=kmax), body, carry
    )
    state = state._replace(vx=vx, vy=vy, vz=vz, vremax=vremax, remain=remain)
    return state, _isum(nattempt), ncoll


# ---------------------------------------------------------------------------------------------
# step / run
# ---------------------------------------------------------------------------------------------


def _grid_tally(state: State, on, cfg: Config):
    C = cfg.ncell
    w = jnp.where(on & (state.cell < C), 1.0, 0.0)
    v2 = state.vx**2 + state.vy**2 + state.vz**2
    vals = jnp.stack([w, w * state.vx, w * state.vy, w * state.vz, w * v2], axis=1)
    return jnp.zeros((C + 1, 5)).at[state.cell].add(vals)[:C]


def step(state: State, xs, seed, consts: Consts, cfg: Config):
    istep, tally_on = xs
    with jax.named_scope("emit"):
        state, inserts, n_in, fail, overflow = emit(state, seed, istep, consts, cfg)
    moved, surf, mc = move(state, seed, istep, consts, cfg, inserts)
    with jax.named_scope("sort"):
        sort = sort_partition if cfg.sort == "partition" else sort_full
        moved = sort(moved, consts, cfg)
    with jax.named_scope("collide"):
        moved, nattempt, ncoll = collide(moved, seed, istep, consts, cfg)
    if cfg.tally:
        with jax.named_scope("tally"):
            t = state.tally
            surf = jnp.where(tally_on, surf, t.surf)
            grid = t.grid + _grid_tally(moved, tally_on, cfg)
            nsample = t.nsample + tally_on.astype(jnp.int32)
            moved = moved._replace(tally=Tally(surf=surf, grid=grid, nsample=nsample))
    stats = Stats(
        np=moved.start[cfg.ncell],
        nattempt=nattempt,
        ncoll=ncoll,
        nscoll=mc["nscoll"],
        nscheck=mc["nscheck"],
        # every death is an OUTFLOW exit or a MAXSTUCK discard
        nexit=state.start[cfg.ncell] + n_in - moved.start[cfg.ncell] - mc["nstuck"],
        nbound=mc["nbound"],
        ninsert=n_in,
        nlane=mc["nlane"],
        nwalk=mc["nwalk"],
        nstuck=mc["nstuck"],
        emit_fail=fail,
        overflow=overflow,
    )
    return moved, stats


@functools.partial(jax.jit, static_argnames=("cfg", "nsteps"), donate_argnames=("state",))
def run(state: State, seed, step0, consts: Consts, cfg: Config, nsteps: int, tally_from=None):
    """``nsteps`` steps from step ``step0`` (0-based) in one lax.scan; returns (state, Stats).
    Tallies (``cfg.tally``) accumulate on steps >= ``tally_from``."""
    steps = step0 + jnp.arange(nsteps, dtype=jnp.int32)
    tf = jnp.int32(2**30) if tally_from is None else tally_from
    body = functools.partial(step, seed=jnp.asarray(seed, jnp.uint64), consts=consts, cfg=cfg)
    return jax.lax.scan(body, state, (steps, steps >= tf))


def moves(stats: Stats):
    """SPARTA's 'Particle moves': particles present at each move (survivors + inserts)."""
    n = np.asarray(stats.np, np.int64)
    prev = np.concatenate([[0], n[:-1]])
    return int(np.sum(prev + np.asarray(stats.ninsert, np.int64)))


def as_numpy_stats(stats: Stats):
    return {k: np.asarray(v) for k, v in stats._asdict().items()}
