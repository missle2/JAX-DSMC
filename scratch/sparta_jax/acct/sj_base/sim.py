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
    lane: int = 5120  # lane of the cell walk, first stage (chunked if more particles are near)
    lane2: int = 768  # lane of the later walk stages (re-compacted every stage)
    per_task: int = 8  # max insertions per task per step: floor(ntarget) + 1
    trials: int = 64  # vectorised accept-reject trials of the emit/face normal-velocity sampler
    movers: int = 8192  # partition sort: slots that changed cell (full sort if more)
    sort: str = "partition"  # "full" (sort all keys) | "partition" (O(N) + small sort of movers)
    fixed_stages: int = 4  # later walk stages run unconditionally before the while_loop
    max_walk: int = 1000  # safety cap on walk stages (SPARTA: unbounded, MAXSTUCK only)
    tally: bool = False  # surface + grid tallies (compute surf/grid + fix ave/*)

    @property
    def ncell(self):
        return self.nx * self.ny


class Consts(NamedTuple):
    cell_lines: jax.Array  # (C, 6 K) f64: p1x p1y p2x p2y nx ny for each line of the cell
    cell_ids: jax.Array  # (C, K) int32 line ids, -1 padded
    nsurf: jax.Array  # (C,) int32
    ring: jax.Array  # (C,) int32 cells of cut-free Chebyshev ring around the cell (-1: cut)
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
    dtr: jax.Array  # move time of each slot this step (dt, or dt * U for a fresh insert)
    plist: jax.Array  # int32 slots sorted by cell, dead last
    skey: jax.Array  # int32 cell of plist[p] (sorted keys)
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
        ring=jnp.asarray(ring_table(geo), jnp.int32),
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
        dtr=jnp.full(N, cfg.dt),
        plist=jnp.arange(N, dtype=jnp.int32),
        skey=jnp.full(N, C, jnp.int32),
        start=start,
        vremax=jnp.full(C, cfg.vremax0),
        remain=jnp.zeros(C),
        tally=tally,
    )


def _put(slot, arr, val):
    return arr.at[slot].set(val, mode="drop")


def _isum(a):
    return jnp.sum(a, dtype=jnp.int32)


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
    slot = jnp.where(ok_slot, state.plist[jnp.minimum(pos, N - 1)], N)
    put = functools.partial(_put, slot)
    state = state._replace(
        x=put(state.x, jnp.full(T * Mt, cfg.xlo)),
        y=put(state.y, y),
        vx=put(state.vx, vx),
        vy=put(state.vy, vy),
        vz=put(state.vz, vz),
        cell=put(state.cell, cell),
        dtr=put(state.dtr, dtr),
    )
    return state, slot, _isum(ok_slot), _isum(fail), _isum(valid & ~ok_slot)


# ---------------------------------------------------------------------------------------------
# 2. move
# ---------------------------------------------------------------------------------------------


def _floor_index(u, n):
    return jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 1)


def dense_move(state: State, consts: Consts, cfg: Config):
    """Straight move of every slot + the 'near' test that routes a slot to the cell walk.

    For a particle whose move (and its mirror image) stays in cells without surfaces, SPARTA's
    walk reduces to Update::optmove: xnew, a mirror in y (REFLECT), deletion through the x
    faces (OUTFLOW), and the cell holding the end point.  Everything else is 'near'.
    """
    C, nx, ny = cfg.ncell, cfg.nx, cfg.ny
    alive = state.cell < C
    xn = state.x + state.dtr * state.vx
    yn = state.y + state.dtr * state.vy
    below, above = yn < cfg.ylo, yn >= cfg.yhi
    flip = below | above
    ym = jnp.where(below, 2.0 * cfg.ylo - yn, jnp.where(above, 2.0 * cfg.yhi - yn, yn))
    exit_ = (xn < cfg.xlo) | (xn >= cfg.xhi)
    c0 = jnp.minimum(state.cell, C - 1)
    i0, j0 = c0 % nx, c0 // nx
    i1 = _floor_index((xn - cfg.xlo) / cfg.dx, nx)
    j1 = _floor_index((yn - cfg.ylo) / cfg.dy, ny)
    j2 = _floor_index((ym - cfg.ylo) / cfg.dy, ny)
    reach = jnp.maximum(jnp.maximum(jnp.abs(i1 - i0), jnp.abs(j1 - j0)), jnp.abs(j2 - j0))
    near = alive & ((reach > consts.ring[c0]) | (flip & exit_))
    dense = alive & ~near
    cell_new = jnp.where(exit_, C, j2 * nx + i1)
    out = state._replace(
        x=jnp.where(dense, xn, state.x),
        y=jnp.where(dense, ym, state.y),
        vy=jnp.where(dense & flip, -state.vy, state.vy),
        cell=jnp.where(dense, cell_new, state.cell),
    )
    return out, near, _isum(dense & exit_), _isum(dense & flip & ~exit_)


class Walk(NamedTuple):
    """Per-lane state of Update::move's while(1) loop."""

    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array
    xn: jax.Array  # xnew
    yn: jax.Array
    dtrem: jax.Array
    icell: jax.Array
    exclude: jax.Array
    stuck: jax.Array
    status: jax.Array  # ACTIVE / DONE / DEAD
    slot: jax.Array  # global slot (RNG counter; N for an empty lane entry)


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
    i, j = w.icell % cfg.nx, w.icell // cfg.nx
    lox = cfg.xlo + i * cfg.dx
    hix = cfg.xlo + (i + 1) * cfg.dx
    loy = cfg.ylo + j * cfg.dy
    hiy = cfg.ylo + (j + 1) * cfg.dy
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


def _surface_hit(w: Walk, box, outface, frac, consts: Consts, cfg: Config):
    """Earliest OUTSIDE hit among the cell's lines on the in-cell part of the path.

    The K line tests run on a (B, K) block and are reduced with one variadic reduce that also
    carries the face result (frac, outface) and the clipped end point: a reduction is a fusion
    root on XLA:CPU, so everything downstream reads materialised values instead of every output
    fusion recomputing the line tests (optimization_barrier is expanded away before fusion).
    """
    crossing = outface != INTERIOR
    cx = jnp.where(crossing, w.x + frac * (w.xn - w.x), w.xn)
    cy = jnp.where(crossing, w.y + frac * (w.yn - w.y), w.yn)
    cx, cy = _snap(cx, cy, outface, box)
    K = cfg.K
    rows = consts.cell_lines[w.icell].reshape(-1, K, 6)  # (B, K, 6)
    ids = consts.cell_ids[w.icell]  # (B, K)
    col = functools.partial(jnp.expand_dims, axis=1)
    ax, ay, bx, by, lnx, lny = (rows[:, :, m] for m in range(6))
    hit, param, _px, _py, outside = _line_line_intersect(
        col(w.x), col(w.y), col(cx), col(cy), ax, ay, bx, by, lnx, lny
    )
    ok = (ids >= 0) & (ids != col(w.exclude)) & hit & outside
    param = jnp.where(ok, param, 2.0)
    kk = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), ids.shape)
    bc = functools.partial(jnp.broadcast_to, shape=ids.shape)
    of = col(outface.astype(jnp.int32))
    ops = (param, kk, bc(col(frac)), bc(of), bc(col(cx)), bc(col(cy)))
    init = (
        jnp.float64(3.0), jnp.int32(K), jnp.float64(0.0), jnp.int32(0),
        jnp.float64(0.0), jnp.float64(0.0),
    )  # fmt: skip
    minparam, kbest, frac, outface, cx, cy = jax.lax.reduce(ops, init, _argmin_reducer, (1,))
    hitk = minparam < 2.0
    kb = jnp.minimum(kbest, K - 1)
    line = jnp.take_along_axis(rows, kb[:, None, None], axis=1)[:, 0, :]  # (B, 6)
    minsurf = jnp.where(hitk, jnp.take_along_axis(ids, kb[:, None], axis=1)[:, 0], -1)
    mx = w.x + minparam * (cx - w.x)
    my = w.y + minparam * (cy - w.y)
    return (minparam, minsurf, mx, my, line[:, 4], line[:, 5]), outface, frac


def _walk_stage(w: Walk, seed, istep, it, surf, consts: Consts, cfg: Config):
    """One pass of Update::move's per-particle while(1) loop over every ACTIVE lane entry."""
    nx, ny = cfg.nx, cfg.ny
    act = w.status == ACTIVE
    box, outface, frac = _face(w, cfg)
    i, j, _lox, _hix, loy, hiy = box
    hitres, outface, frac = _surface_hit(w, box, outface, frac, consts, cfg)
    minparam, minsurf, mx, my, nrx, nry = hitres
    crossing = outface != INTERIOR
    cflag = act & (minsurf >= 0)

    # --- surface collision: SurfCollideDiffuse::collide (acc = 0: always specular) ---
    ctr = w.slot * 4096 + it
    specular = rng.uniform(rng.step_key(seed, istep, S_SURF), ctr) > cfg.acc
    vdot = w.vx * nrx + w.vy * nry
    svx, svy, svz = w.vx - 2.0 * vdot * nrx, w.vy - 2.0 * vdot * nry, w.vz
    if cfg.acc > 0.0:
        dvx, dvy, dvz = _diffuse(w.vx, w.vy, w.vz, nrx, nry, seed, istep, ctr, cfg)
        svx = jnp.where(specular, svx, dvx)
        svy = jnp.where(specular, svy, dvy)
        svz = jnp.where(specular, svz, dvz)
    dtrem_c = w.dtrem * (1.0 - minparam * frac)
    stuck_c = jnp.where(minparam <= 1.0e-14, w.stuck + 1, 0)
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
    step_cell = jnp.where(
        outface == XLO, -1, jnp.where(outface == XHI, 1, jnp.where(outface == YLO, -nx, nx))
    )
    newcell = cross & ~exits & ~reflect
    ynr = jnp.where(at_ylo, 2.0 * loy - w.yn, 2.0 * hiy - w.yn)
    status = jnp.where(exits | discard_stuck, DEAD, jnp.where(interior, DONE, w.status)).astype(
        jnp.int32
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
        icell=jnp.where(newcell, w.icell + step_cell, w.icell),
        exclude=jnp.where(cflag, minsurf, jnp.where(cross, -1, w.exclude)),
        stuck=jnp.where(cflag, stuck_c, w.stuck),
        status=status,
        slot=w.slot,
    )
    cnt = jnp.stack(
        [
            _isum(cflag),  # nscoll
            jnp.sum(jnp.where(act, consts.nsurf[w.icell], 0), dtype=jnp.int32),  # nscheck
            _isum(exits),  # nexit
            _isum(reflect),  # nbound
            _isum(discard_stuck),  # nstuck
        ]
    )
    return w2, cnt, surf


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


def _gather_walk(w: Walk, idx, fill_slot):
    """Lane entries idx of w (idx == len: empty entry, inactive)."""
    n = w.x.shape[0]
    valid = idx < n
    g = jnp.minimum(idx, n - 1)
    out = Walk(*(a[g] for a in w))
    return out._replace(
        status=jnp.where(valid, out.status, DONE),
        slot=jnp.where(valid, out.slot, fill_slot),
    )


def _scatter_walk(w: Walk, idx, part: Walk):
    return Walk(*(a.at[idx].set(b, mode="drop") for a, b in zip(w, part, strict=True)))


def _compact(mask, size):
    """Indices of the first ``size`` set entries of mask (len(mask) where there are fewer)."""
    n = mask.shape[0]
    r = jnp.cumsum(mask.astype(jnp.int32)) - 1
    return (
        jnp.full(size, n, jnp.int32)
        .at[jnp.where(mask & (r < size), r, size)]
        .set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )


def _later_stage(carry, seed, istep, consts: Consts, cfg: Config):
    w, it, cnt, surf = carry
    idx = _compact(w.status == ACTIVE, cfg.lane2)
    part = _gather_walk(w, idx, cfg.capacity)
    part, c, surf = _walk_stage(part, seed, istep, it, surf, consts, cfg)
    return _scatter_walk(w, idx, part), it + 1, cnt + c, surf


def _later_cond(carry, cfg: Config):
    w, it, _cnt, _surf = carry
    return jnp.any(w.status == ACTIVE) & (it < cfg.max_walk)


def _lane_chunk(carry, rank, near, seed, istep, consts: Consts, cfg: Config):
    """Walk the ``c``-th block of ``lane`` near particles to the end of their move."""
    state, c, cnt, iters, surf = carry
    B, N, C = cfg.lane, cfg.capacity, cfg.ncell
    local = rank - c * B
    slots = (
        jnp.full(B, N, jnp.int32)
        .at[jnp.where(near & (local >= 0) & (local < B), local, B)]
        .set(jnp.arange(N, dtype=jnp.int32), mode="drop")
    )
    valid = slots < N
    g = jnp.minimum(slots, N - 1)
    x, y, vx, vy, dtr = state.x[g], state.y[g], state.vx[g], state.vy[g], state.dtr[g]
    w = Walk(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        vz=state.vz[g],
        xn=x + dtr * vx,
        yn=y + dtr * vy,
        dtrem=dtr,
        icell=jnp.where(valid, state.cell[g], 0),
        exclude=jnp.full(B, -1, jnp.int32),
        stuck=jnp.zeros(B, jnp.int32),
        status=jnp.where(valid, ACTIVE, DONE).astype(jnp.int32),
        slot=slots,
    )
    w, c1, surf = _walk_stage(w, seed, istep, jnp.int32(0), surf, consts, cfg)
    body = functools.partial(_later_stage, seed=seed, istep=istep, consts=consts, cfg=cfg)
    carry = (w, jnp.int32(1), c1, surf)
    for _ in range(cfg.fixed_stages):  # unconditional (masked) stages: no loop-predicate sync
        carry = body(carry)
    w, it, c2, surf = jax.lax.while_loop(functools.partial(_later_cond, cfg=cfg), body, carry)
    put = functools.partial(_put, slots)
    state = state._replace(
        x=put(state.x, w.x),
        y=put(state.y, w.y),
        vx=put(state.vx, w.vx),
        vy=put(state.vy, w.vy),
        vz=put(state.vz, w.vz) if cfg.acc > 0.0 else state.vz,
        cell=put(state.cell, jnp.where(w.status == DEAD, C, w.icell)),
    )
    return state, c + 1, cnt + c2, iters + it, surf


def _lane_cond(carry, nchunk):
    return carry[1] < nchunk


def move(state: State, seed, istep, consts: Consts, cfg: Config):
    """Update::move<2,1,0> for all particles (dense straight move + cell walk on the lane)."""
    with jax.named_scope("move_dense"):
        new, near, nexit_d, nbound_d = dense_move(state, consts, cfg)
    with jax.named_scope("move_select"):
        near_i = near.astype(jnp.int32)
        rank = jnp.cumsum(near_i) - 1
        nnear = rank[-1] + 1
        nchunk = (nnear + cfg.lane - 1) // cfg.lane
    with jax.named_scope("move_walk"):
        # near slots were left untouched by dense_move, so the lane reads their pre-move values
        body = functools.partial(
            _lane_chunk, rank=rank, near=near, seed=seed, istep=istep, consts=consts, cfg=cfg
        )
        carry = (new, jnp.int32(0), jnp.zeros(5, jnp.int32), jnp.int32(0), state.tally.surf)
        carry = body(carry)  # first chunk unconditionally; more chunks only on lane overflow
        new, _c, cnt, iters, surf = jax.lax.while_loop(
            functools.partial(_lane_cond, nchunk=nchunk), body, carry
        )
    counts = {
        "nscoll": cnt[0],
        "nscheck": cnt[1],
        "nexit": cnt[2] + nexit_d,
        "nbound": cnt[3] + nbound_d,
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


def _full_sort(cell, cfg: Config):
    N, C = cfg.capacity, cfg.ncell
    s = jnp.sort(cell * 65536 + jnp.arange(N, dtype=jnp.int32))
    counts = jnp.zeros(C + 1, jnp.int32).at[cell].add(1)
    return s & 0xFFFF, s >> 16, _starts(counts)


def sort_full(state: State, cfg: Config):
    """Particle::sort as one int32 sort of packed (cell, slot) keys; dead slots sort last."""
    plist, skey, start = _full_sort(state.cell, cfg)
    return state._replace(plist=plist, skey=skey, start=start)


def sort_partition(state: State, cfg: Config):
    """Particle::sort in O(N): slots that kept their cell keep their order (they were sorted
    last step); the few that changed cell ("movers": crossings, exits, inserts) are sorted
    alone and placed after the stayers of their new cell.  Falls back to the full sort when the
    movers overflow their buffer."""
    N, M = cfg.capacity, cfg.movers
    knew = state.cell[state.plist]  # new key at each sorted position
    stay = knew == state.skey
    stay_i = stay.astype(jnp.int32)
    cs = jnp.cumsum(stay_i) - stay_i  # exclusive
    nmov = N - (cs[-1] + stay_i[-1])
    fast = functools.partial(_partition, knew, stay, stay_i, cs, state, cfg)
    slow = functools.partial(_slow_sort, state.cell, cfg)
    plist, skey, start = jax.lax.cond(nmov <= M, fast, slow)
    return state._replace(plist=plist, skey=skey, start=start)


def _slow_sort(cell, cfg: Config):
    return _full_sort(cell, cfg)


def _partition(knew, stay, stay_i, cs, state: State, cfg: Config):
    N, C, M = cfg.capacity, cfg.ncell, cfg.movers
    old_start = state.start[: C + 1]
    pos = jnp.arange(N, dtype=jnp.int32)
    # movers: compact their positions (mover rank = position - stayers before it)
    mpos = jnp.full(M, N, jnp.int32).at[jnp.where(stay, M, pos - cs)].set(pos, mode="drop")
    mvalid = mpos < N
    g = jnp.minimum(mpos, N - 1)
    kn, ko = knew[g], state.skey[g]
    # bucket counts from last step's counts and the movers alone
    one = mvalid.astype(jnp.int32)
    nstay = (state.start[1:] - old_start) - jnp.zeros(C + 1, jnp.int32).at[ko].add(one)
    counts = nstay + jnp.zeros(C + 1, jnp.int32).at[kn].add(one)
    start = _starts(counts)
    # stayers: rank among the stayers of their bucket (old bucket == new bucket)
    offs = start[: C + 1] - cs[jnp.minimum(old_start, N - 1)]
    dest_stay = jnp.where(stay, offs[knew] + cs, N)
    # movers: sort by (new bucket, position), rank inside their bucket, after its stayers
    sb = jnp.sort(jnp.where(mvalid, kn * 65536 + mpos, (C + 1) * 65536))
    mk, mp = sb >> 16, sb & 0xFFFF
    ok = mk <= C
    mkc = jnp.minimum(mk, C)
    ar = jnp.arange(M, dtype=jnp.int32)
    first = jnp.full(C + 1, M, jnp.int32).at[mkc].min(jnp.where(ok, ar, M))
    dest_mov = jnp.where(ok, start[mkc] + nstay[mkc] + ar - first[mkc], N)
    plist = jnp.zeros(N, jnp.int32).at[dest_stay].set(state.plist, mode="drop")
    plist = plist.at[dest_mov].set(state.plist[jnp.minimum(mp, N - 1)], mode="drop")
    skey = jnp.zeros(N, jnp.int32).at[dest_stay].set(knew, mode="drop")
    skey = skey.at[dest_mov].set(mkc, mode="drop")
    return plist, skey, start


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
    pi = state.plist[jnp.clip(base + i, 0, N - 1)]
    pj = state.plist[jnp.clip(base + j, 0, N - 1)]
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
        state, slot, n_in, fail, overflow = emit(state, seed, istep, consts, cfg)
    moved, surf, mc = move(state, seed, istep, consts, cfg)
    with jax.named_scope("emit"):
        # fresh inserts go back to a full dt for the next step
        moved = moved._replace(dtr=moved.dtr.at[slot].set(cfg.dt, mode="drop"))
    with jax.named_scope("sort"):
        moved = sort_partition(moved, cfg) if cfg.sort == "partition" else sort_full(moved, cfg)
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
        nexit=mc["nexit"],
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
