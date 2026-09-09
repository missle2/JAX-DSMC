"""Spatial domain decomposition across devices (spec section 20.2).

Each device owns a contiguous slab of cells along ``x`` and the particles inside
it.  Every stage is then **local**:

* ``move`` and the boundary fold -- element-wise, always local;
* ``sort`` -- over that device's own particles into its own cells, so the global
  ``argsort`` that defeated :mod:`jax_dsmc.parallel` becomes ``P`` independent
  sorts of ``N/P`` elements;
* ``collide`` -- per cell, and a cell lives on exactly one device;
* ``sample`` -- ``segment_sum`` into that device's own cells.

The only communication is the particles that cross a slab boundary during a step,
exchanged with the two neighbours by :func:`migrate`.  Because the timestep is
CFL-limited a particle moves a fraction of a cell per step, so the exchange is
nearest-neighbour and tiny -- of order ``npc * (crossing fraction)`` particles per
boundary, against ``N/P`` held.  This is SPARTA's ``comm->migrate_particles()``
specialised to a 1-D cut of a Cartesian grid.

TWO CAPACITIES, TWO SILENT FAILURES.  Static shapes mean both the per-device
particle array and the migration buffer are fixed sizes, and overflowing either
loses particles rather than raising:

``n_dev``   per-device particle capacity.  A compressible flow piles particles up,
            so a slab can hold well over its share.  :func:`migrate` returns
            ``overflow_recv`` -- particles that arrived with no free slot.
``m_max``   migration buffer.  :func:`migrate` returns ``overflow_send`` --
            particles that needed to leave and could not.

Both are reported every step and both are **errors, not warnings**: a lost
particle is lost mass, and nothing downstream will notice.  :func:`suggest_capacity`
sizes them, and the driver grows them the way ``simulate.run_adaptive`` grows
``A_max`` -- by measuring, not by guessing (section 8.5.1).

WHY THE CUT IS ALONG ``x`` ONLY.  A 1-D cut keeps the neighbour set at two and the
slab-assignment arithmetic exact (``floor((x - lo)/slab_width)``).  A 2-D or 3-D
cut has 8 or 26 neighbours and only pays off when ``P`` grows past the point where
slabs get thinner than a few cells.  At ``P = 4`` a slab is still hundreds of cells
wide, so the 1-D cut costs nothing and is far easier to show correct.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

AXIS = "dev"


def slab_of(x, grid, n_slabs: int):
    """Which slab each particle belongs to, ``i32[N]``.

    The cut is along ``x`` (axis 0) and the slabs are equal in *cell count*, so a
    slab boundary always falls on a cell boundary -- a cell is never split, which
    is what keeps every per-cell quantity owned by exactly one device.
    """
    nx = grid.n[0]
    if nx % n_slabs:
        raise ValueError(f"x cell count {nx} does not divide over {n_slabs} slabs")
    per = nx // n_slabs
    lo = grid.lo[0]
    dx = (grid.hi[0] - lo) / nx
    ix = jnp.floor((x[:, 0] - lo) / dx).astype(jnp.int32)
    ix = jnp.clip(ix, 0, nx - 1)
    return ix // per


def sub_grid(grid, rank: int, n_slabs: int):
    """The ``GridSpec`` for one slab.  Host-side: ``rank`` is a Python int.

    Only ``lo``/``hi`` on axis 0 change; ``n`` shrinks on axis 0.  Everything that
    reads the grid (``cell_of``, ``cell_volume``, ``cell_bounds``) then works
    unmodified on the local slab, which is the point of keeping the mesh interface
    narrow (section 4).
    """
    nx = grid.n[0]
    per = nx // n_slabs
    dx = (grid.hi[0] - grid.lo[0]) / nx
    lo = grid.lo.at[0].set(grid.lo[0] + rank * per * dx)
    hi = grid.hi.at[0].set(grid.lo[0] + (rank + 1) * per * dx)
    return dataclasses.replace(grid, lo=lo, hi=hi, n=(per, *tuple(grid.n[1:])))


def _pack(mask, m_max):
    """Indices of up to ``m_max`` particles where ``mask``, padded with ``-1``.

    ``argsort`` on the boolean brings the selected particles to the front in one
    stable, statically-shaped operation -- no ``where``-with-dynamic-size, which
    JAX cannot compile.
    """
    order = jnp.argsort(~mask, stable=True)
    take = order[:m_max]
    ok = jnp.arange(m_max) < jnp.sum(mask)
    return jnp.where(ok, take, -1), jnp.maximum(jnp.sum(mask) - m_max, 0)


def _gather(state, idx):
    """Gather particles at ``idx`` (``-1`` yields a dead placeholder)."""
    valid = idx >= 0
    safe = jnp.maximum(idx, 0)
    return dataclasses.replace(
        state,
        x=state.x[safe],
        v=state.v[safe],
        species=state.species[safe],
        alive=state.alive[safe] & valid,
        logh_acc=state.logh_acc[safe],
    )


def _drop_index(idx, valid, n):
    """Turn invalid entries into an OUT-OF-BOUNDS index, for ``mode="drop"``.

    Clamping them to ``0`` instead -- the obvious thing -- is a real bug and a
    quiet one: every padding entry then addresses slot 0, so a padded scatter
    fights with whatever genuinely targets slot 0 and one of them wins
    arbitrarily.  It duplicated six particles per step here before the
    conservation check caught it.  An out-of-range index with ``mode="drop"`` is
    discarded outright, which is what "no particle" should mean.
    """
    return jnp.where(valid, idx, n)


def _scatter(state, slots, incoming):
    """Write ``incoming`` into ``slots``; invalid entries are dropped, not clamped."""
    n = state.alive.shape[0]
    valid = (slots >= 0) & incoming.alive
    tgt = _drop_index(slots, valid, n)
    kw = {"mode": "drop"}
    return dataclasses.replace(
        state,
        x=state.x.at[tgt].set(incoming.x, **kw),
        v=state.v.at[tgt].set(incoming.v, **kw),
        species=state.species.at[tgt].set(incoming.species, **kw),
        alive=state.alive.at[tgt].set(True, **kw),
        # A slot being refilled must NOT inherit the score history of whoever held
        # it (section 9.3, gap U6) -- the incoming particle brings its own.
        logh_acc=state.logh_acc.at[tgt].set(incoming.logh_acc, **kw),
    )


def migrate(state, grid, n_slabs, m_max, periodic=True):
    """Exchange slab-crossing particles with the two neighbours.

    Must be called inside ``shard_map`` over :data:`AXIS`.  Returns
    ``(state, diagnostics)`` where the diagnostics are the two overflow counts and
    the number actually sent each way.

    The exchange is two ``ppermute`` calls -- one shifting left, one right -- each
    carrying a fixed ``m_max``-sized buffer.  Fixed size is what makes this
    compilable and what makes it a capacity that can overflow; see the module
    docstring.
    """
    me = lax.axis_index(AXIS)
    want = slab_of(state.x, grid, n_slabs)
    # Signed slab displacement, wrapped for a periodic domain.  With a CFL-limited
    # step this is always in {-1, 0, +1}; anything else means the timestep is too
    # large for the decomposition and is reported rather than silently folded.
    d = want - me
    if periodic:
        d = jnp.where(d > n_slabs // 2, d - n_slabs, d)
        d = jnp.where(d < -(n_slabs // 2), d + n_slabs, d)
    far = jnp.sum(state.alive & (jnp.abs(d) > 1))

    go_r = state.alive & (d == 1)
    go_l = state.alive & (d == -1)
    idx_r, over_r = _pack(go_r, m_max)
    idx_l, over_l = _pack(go_l, m_max)
    send_r = _gather(state, idx_r)
    send_l = _gather(state, idx_l)

    # The senders leave, freeing their slots before anything is written back.
    n_slot = state.alive.shape[0]
    leaving = jnp.zeros_like(state.alive)
    leaving = leaving.at[_drop_index(idx_r, idx_r >= 0, n_slot)].set(True, mode="drop")
    leaving = leaving.at[_drop_index(idx_l, idx_l >= 0, n_slot)].set(True, mode="drop")
    state = dataclasses.replace(state, alive=state.alive & ~leaving)

    perm_r = [(i, (i + 1) % n_slabs) for i in range(n_slabs)]
    perm_l = [(i, (i - 1) % n_slabs) for i in range(n_slabs)]
    recv_l = jax.tree.map(lambda a: lax.ppermute(a, AXIS, perm_r), send_r)  # from the left
    recv_r = jax.tree.map(lambda a: lax.ppermute(a, AXIS, perm_l), send_l)  # from the right

    incoming = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), recv_l, recv_r)
    # COMPACT the arrivals before placing them.  `incoming` is two buffers glued
    # together, so its live entries sit in two blocks -- at the front of each half
    # -- while `_free_slots` returns valid slots only in one run at ITS front.
    # Written straight out, the second block's arrivals address slots past the end
    # of that run and are dropped: particles vanish with `overflow_recv` still
    # reading zero, because the count fits comfortably.  It cost 328 particles in
    # the 2-D test and nothing in the 1-D one, where `2*m_max` happened to be
    # smaller than the free-slot count.
    ord_in = jnp.argsort(~incoming.alive, stable=True)
    incoming = jax.tree.map(lambda a: a[ord_in], incoming)
    n_in = jnp.sum(incoming.alive)
    free = _free_slots(state.alive, 2 * m_max)
    # `free` may be shorter than `incoming` -- see `_free_slots` -- so trim the
    # arrivals to match.  Safe only because they were just compacted: the live ones
    # are at the front, so a truncation drops padding, and anything genuinely too
    # numerous to place is counted in `overflow_recv` rather than dropped quietly.
    incoming = jax.tree.map(lambda a: a[: free.shape[0]], incoming)
    room = jnp.minimum(jnp.sum(~state.alive), free.shape[0])
    over_recv = jnp.maximum(n_in - room, 0)
    state = _scatter(state, free, incoming)
    return state, {
        "overflow_send": over_r + over_l,
        "overflow_recv": over_recv,
        # what actually left, not what wanted to: an overflowing send is reported
        # by `overflow_send`, and double-counting it here would hide the overflow
        # behind a healthy-looking migration rate.
        "sent": jnp.minimum(jnp.sum(go_r), m_max) + jnp.minimum(jnp.sum(go_l), m_max),
        "far_jumps": far,
    }


def _free_slots(alive, k):
    """Indices of up to ``k`` dead slots, ordered; ``-1`` where there are none.

    ``k`` is clamped to the array length.  It is not always smaller: ``k`` is
    ``2 * m_max``, and when a slab is only a couple of cells wide a large fraction
    of its particles can cross in one step, so the migration buffer legitimately
    approaches the device capacity.  Without the clamp ``order[:k]`` silently
    returns a shorter array than ``arange(k)`` and the two fail to broadcast.
    """
    n = alive.shape[0]
    k = min(int(k), n)
    order = jnp.argsort(alive, stable=True)  # dead (False) first
    take = order[:k]
    ok = jnp.arange(k) < jnp.sum(~alive)
    return jnp.where(ok, take, -1)


def suggest_capacity(n_total, n_slabs, headroom=1.6):
    """Per-device particle capacity.  ``headroom`` covers density non-uniformity.

    A factor of 1.6 accommodates a slab holding 60 % more than its equal share --
    comfortable for a shock (whose peak compression is the density ratio, at most
    4 for a monatomic gas, but spread over a slab far wider than the wave) and for
    Taylor-Green (whose density varies by a few per cent).  It is a starting point:
    the driver measures ``overflow_recv`` and grows it, exactly as ``A_max`` is
    grown (section 8.5.1).
    """
    per = -(-n_total // n_slabs)
    return int(per * headroom)


def suggest_migration_buffer(n_dev, cells_x_per_slab, cfl=0.2, safety=8.0):
    """Migration buffer size.

    Only particles within one cell of a slab face can cross in a step, and only the
    fraction ``~cfl`` of those actually do, so the expected count is about
    ``(n_dev / cells_x_per_slab) * cfl``.  The buffer is that times ``safety``,
    because the quantity is a fluctuating count whose *tail* is what overflows --
    sizing to the mean would overflow about half the time.
    """
    expect = n_dev / max(cells_x_per_slab, 1) * cfl
    return max(64, int(expect * safety))


def local_grid(grid, n_slabs: int):
    """The calling device's own slab, built from ``lax.axis_index``.

    Must be called inside ``shard_map``.  ``lo``/``hi`` are traced anyway, so a
    traced rank costs nothing; the local cell count is static because every slab
    holds the same number of cells.

    **Why the cell arrays shard for free.**  Cells are raveled row-major with ``x``
    on axis 0, so ``global = ix * prod(n[1:]) + rest``.  A slab is
    ``ix in [r*per, (r+1)*per)``, hence global indices
    ``[r*C/P, (r+1)*C/P)`` -- a *contiguous block*, in the same order as this
    subgrid's own ``cell_of``.  So ``PartitionSpec(AXIS)`` on any ``f64[C]`` field
    (``fnum``, ``vremax``, every moment) gives each device exactly its own cells
    with no permutation anywhere.
    """
    rank = lax.axis_index(AXIS)
    nx = grid.n[0]
    per = nx // n_slabs
    dx = (grid.hi[0] - grid.lo[0]) / nx
    lo = grid.lo.at[0].set(grid.lo[0] + rank * per * dx)
    hi = grid.lo.at[0].set(grid.lo[0] + (rank + 1) * per * dx)
    hi = jnp.where(jnp.arange(grid.lo.shape[0]) == 0, hi, grid.hi)
    return dataclasses.replace(grid, lo=lo, hi=hi, n=(per, *tuple(grid.n[1:])))


def decomp_step(carry, k, grid, sp, params, static, n_slabs, m_max, base_key, bc):
    """One timestep on one slab.  Called inside ``shard_map``.

    Order matters: the boundary condition is applied against the **global** domain
    (a periodic wrap spans the whole box, not a slab), and only then are the
    slab-crossers migrated.  Doing it the other way round would wrap particles at
    slab faces, which is a different -- and wrong -- problem.
    """
    from jax_dsmc import sample, sort  # noqa: PLC0415
    from jax_dsmc.collide.ntc import collide  # noqa: PLC0415
    from jax_dsmc.move import move  # noqa: PLC0415

    state, cstate = carry
    state = move(state, grid, params.dt, bc)  # GLOBAL grid: BC is a global rule
    state, mig = migrate(state, grid, n_slabs, m_max, periodic=(bc.name == "PERIODIC"))
    lg = local_grid(grid, n_slabs)
    table = sort.build_table(state, lg)
    m = sample.moments(state, table, lg, params, sp)
    state, cstate, diag = collide(state, table, cstate, lg, sp, params, static, m.temp, k, base_key)
    return (state, cstate), (m, diag, mig)


def run_decomposed(
    state,
    cstate,
    grid,
    sp,
    params,
    static,
    base_key,
    mesh,
    bc,
    n_steps,
    m_max,
    emit_moments=True,
):
    """Run ``n_steps`` with the domain split over ``mesh``.

    Placement, and it is the whole contract:

    ==================  ==========================  =============================
    array               spec                        why
    ==================  ==========================  =============================
    ``state.*``         ``PartitionSpec(AXIS)``     each device holds its slab's
                                                    particles in a fixed capacity
    ``cstate.vremax``   ``PartitionSpec(AXIS)``     per cell, and a cell has one
                                                    owner
    ``params.fnum``     ``PartitionSpec(AXIS)``     per cell
    ``params.dt``       ``PartitionSpec()``         scalar
    ``grid.lo/hi``      ``PartitionSpec()``         the GLOBAL box; each device
                                                    derives its slab from
                                                    :func:`local_grid`
    ==================  ==========================  =============================

    **Randomness is per-device.**  ``base_key`` is folded with the device index, so
    the slabs draw independent streams -- without that, cell ``j`` of every slab
    would receive identical uniforms and the collisions would be correlated across
    the whole domain.  The consequence is that a decomposed run is *statistically*
    equivalent to a single-device run but **not bitwise equal to it**.  The frozen-
    randomness contract (section 1) still holds exactly *at a fixed decomposition*,
    which is what gradient checking needs; it does not hold across a change in
    device count, so a finite-difference check must keep ``P`` fixed.
    """
    import functools  # noqa: PLC0415

    from jax.experimental.shard_map import shard_map  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415

    n_slabs = mesh.devices.size
    # `static.N` is a SHAPE, and inside `shard_map` every shape is the per-device
    # one.  Leaving the global N here makes `collide`'s `logh.at[:N]` slice past the
    # end of a shard -- caught as a broadcast error, but only because the capacity
    # happened not to divide; a capacity that did divide would have silently
    # written the wrong particles.
    static = dataclasses.replace(static, N=state.x.shape[0] // n_slabs)
    body = functools.partial(
        decomp_step,
        grid=grid,
        sp=sp,
        static=static,
        n_slabs=n_slabs,
        m_max=m_max,
        bc=bc,
    )

    def inner(state, cstate, params, base_key):
        key = jax.random.fold_in(base_key, lax.axis_index(AXIS))
        f = functools.partial(body, params=params, base_key=key)
        return lax.scan(f, (state, cstate), jnp.arange(n_steps))

    st_spec = jax.tree.map(lambda _: P(AXIS), state)
    cs_spec = jax.tree.map(lambda _: P(AXIS), cstate)
    pr_spec = dataclasses.replace(params, dt=P(), fnum=P(AXIS), epsilon=P())
    out_spec = ((st_spec, cs_spec), (P(None, AXIS), P(), P()))
    f = shard_map(
        inner,
        mesh=mesh,
        in_specs=(st_spec, cs_spec, pr_spec, P()),
        out_specs=out_spec,
        check_rep=False,
    )
    return jax.jit(f)(state, cstate, params, base_key)
