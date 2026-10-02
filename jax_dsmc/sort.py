"""Cell occupancy table (spec section 6).

SPARTA builds per-cell linked lists (``collide.cpp:508-512``).  That is a
CPU-appropriate structure; on GPU the right primitive is a sort.

Particle ``j`` of cell ``k`` is ``order[starts[k] + j]``, valid for
``j < counts[k]``.

Differentiability (section 6.1): ``order`` is an integer permutation — a frozen
decision (section 0.5).  The *gather* ``v[order]`` is linear, so its VJP is a
scatter and cotangents flow back to the correct particles automatically.
Nothing special is needed.

Scaling (section 6.2): the full sort (:func:`build_table`) is ``O(N log N)``.
``SimStatic.sort = "partition"`` (:func:`update_table`, SPARTA-JAX's partition sort) carries
the previous step's order and sorts only the slots that changed cell -- ``O(N)`` passes
plus a sort of ~10 % of the slots on the SPARTA circle -- with the same cells, counts and
members; only the order within a cell differs (stayers first).  A blocked counting sort
that reproduces the stable permutation measured ~2x the packed-key sort (2026-10-01).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jax_dsmc.compaction import prefix_count
from jax_dsmc.grid import collision_cell, num_cells
from jax_dsmc.state import CellTable, GridSpec, ParticleState, SortCarry

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def stable_cell_order(c: jax.Array, n_keys: int | None = None):
    """``argsort(c, stable=True)`` for integer keys ``0 <= c < n_keys``, and the sorted keys.

    The keys packed with the slot index (``c << b | i``, ``b`` the bits of ``N - 1``) are
    unique, so any sort of them is the stable order -- one integer operand instead of a
    stable (key, index) pair sort, ~4x faster on XLA:CPU (scratch/perf, 2026-09-28) and the
    same permutation, bit for bit.  The packed word is ``int32`` when ``n_keys << b`` fits
    (no 64-bit division to unpack: a shift and a mask), else ``int64``.  ``n_keys``
    defaults to the bound ``2^62 / N`` of the ``int64`` packing.

    Returns ``(order i32[N], sorted keys i32[N])``.
    """
    n = c.shape[0]
    bits = max(1, (n - 1).bit_length())
    wide = n_keys is None or (n_keys << bits) > 2**31 - 1
    dtype = jnp.int64 if wide else jnp.int32
    packed = (c.astype(dtype) << bits) | jnp.arange(n, dtype=dtype)
    packed = jax.lax.sort(packed, is_stable=False)
    return (packed & ((1 << bits) - 1)).astype(jnp.int32), (packed >> bits).astype(jnp.int32)


def build_table(state: ParticleState, grid: GridSpec) -> CellTable:
    """Sort particles by cell and build the occupancy table.

    Args:
        state: ``ParticleState``; ``x`` is ``f64[N, D]`` TRACED, ``alive`` is
            ``bool[N]``.
        grid: ``GridSpec``; ``lo``/``hi`` TRACED, ``n`` STATIC.

    Returns:
        ``CellTable`` with ``order i32[N]``, ``starts i32[C]``, ``counts i32[C]``
        and ``cell i32[N]``.  Dead particles are assigned the sentinel cell ``C``
        so they sort past ``sum(counts)`` and are never selected for a collision.

    ``cell`` is returned rather than discarded because it is computed here
    anyway and is needed by the **count score** of section 8.1 — a per-cell term
    applied to every particle in that cell.  Returning it avoids recomputing
    ``collision_cell`` inside ``collide``.
    """
    C = num_cells(grid)
    # the collision cell: a split cell's regions are separate cells (SPARTA sub-cells)
    c = collision_cell(state.x, grid)  # i32[N]
    c = jnp.where(state.alive, c, C).astype(jnp.int32)  # dead -> sentinel cell C
    order, keys = stable_cell_order(c, C + 1)  # i32[N]  FROZEN (int)
    # the bucket boundaries of the sorted keys: a binary search per cell instead of a
    # bincount scatter over every slot and its prefix sum (the same integers)
    bounds = jnp.searchsorted(
        keys, jnp.arange(C + 1, dtype=jnp.int32), side="left", method="scan_unrolled"
    ).astype(jnp.int32)  # i32[C+1]: starts, then the first dead slot
    return CellTable(order=order, starts=bounds[:C], counts=bounds[1:] - bounds[:C], cell=c)


def partition_active(static) -> bool:
    """Does this run use the carried partition sort?  ``SimStatic.sort = "partition"`` on
    XLA:CPU only: on GPU the full sort of every slot is one fast kernel, where the
    partition's branch and searches are several launches (as SPARTA-JAX measured)."""
    return static.sort == "partition" and jax.default_backend() == "cpu"


def table_cells(state: ParticleState, grid: GridSpec) -> jax.Array:
    """``i32[N]``: each slot's collision cell, the sentinel ``C`` for a dead slot."""
    c = collision_cell(state.x, grid)
    return jnp.where(state.alive, c, num_cells(grid)).astype(jnp.int32)


def initial_carry(state: ParticleState, grid: GridSpec) -> SortCarry:
    """The :class:`SortCarry` of ``state`` by the full sort (the first step of a run)."""
    c = table_cells(state, grid)
    order, key = stable_cell_order(c, num_cells(grid) + 1)
    return SortCarry(order=order, key=key)


MOVER_TIERS = (16, 8, 6, 4)
"""The mover buffers of :func:`update_table`, ``N // k`` slots each (the smallest that holds
the step's movers is used; more than ``N // 4`` falls back to the full sort).  Measured on
the SPARTA circle (60 k slots, 46.5 k alive): ~11 % of the slots change cell per step."""


def _starts(counts):
    """Exclusive prefix sum with the total appended: ``i32[K + 1]``."""
    return jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])


def _full_update(c, carry, stay, cs, n_keys):
    """:func:`update_table`'s fallback: the full stable sort."""
    del carry, stay, cs
    return stable_cell_order(c, n_keys)


def _partition_update(c, carry, stay, cs, n_keys, size):
    """:func:`update_table` with at most ``size`` movers: the stayers scattered to their new
    places in their old order, the movers compacted, sorted by (cell, slot) and placed after
    their cell's stayers.  Returns the new ``(order, key)``."""
    n = c.shape[0]
    order, key = carry.order, carry.key
    knew = c[order]  # each sorted position's new cell
    pos = jnp.arange(n, dtype=jnp.int32)
    # movers compacted in sorted-position order, packed (new cell, slot)
    bits = max(1, (n - 1).bit_length())
    wide = (n_keys << bits) > 2**31 - 1
    dtype = jnp.int64 if wide else jnp.int32
    packed = (knew.astype(dtype) << bits) | order.astype(dtype)
    sentinel = jnp.asarray(n_keys, dtype) << bits
    mval = (
        jnp.full(size, sentinel, dtype).at[jnp.where(stay, size, pos - cs)].set(packed, mode="drop")
    )
    # stayers per cell: the stayer count at the cell's old bucket bounds
    bounds = jnp.searchsorted(key, jnp.arange(n_keys + 1, dtype=jnp.int32), side="left")
    cs_end = jnp.concatenate([cs, (cs[-1] + stay[-1]).astype(jnp.int32)[None]])
    nstay = cs_end[bounds[1:]] - cs_end[bounds[:-1]]  # i32[n_keys]
    mcell = (mval >> bits).astype(jnp.int32)
    hist = jnp.zeros(n_keys + 1, jnp.int32).at[mcell].add(1)[:n_keys]
    start = _starts(nstay + hist)[:n_keys]
    # a stayer goes to its cell's start plus its rank among the cell's stayers
    offset = start - _starts(nstay)[:n_keys]
    dest = jnp.where(stay, offset[knew] + cs, n)
    new_order = jnp.zeros(n, jnp.int32).at[dest].set(order, mode="drop")
    new_key = jnp.zeros(n, jnp.int32).at[dest].set(knew, mode="drop")
    # movers, sorted, after their cell's stayers
    ms = jax.lax.sort(mval, is_stable=False)
    mk = (ms >> bits).astype(jnp.int32)
    ok = mk < n_keys
    mkc = jnp.minimum(mk, n_keys - 1)
    first = _starts(hist)[:n_keys]
    mdest = jnp.where(
        ok, start[mkc] + nstay[mkc] + jnp.arange(size, dtype=jnp.int32) - first[mkc], n
    )
    mslot = (ms & ((1 << bits) - 1)).astype(jnp.int32)
    new_order = new_order.at[mdest].set(mslot, mode="drop")
    new_key = new_key.at[mdest].set(mk, mode="drop")
    return new_order, new_key


def update_table(
    state: ParticleState, grid: GridSpec, carry: SortCarry
) -> tuple[CellTable, SortCarry]:
    """:func:`build_table` from the previous step's order (``SimStatic.sort = "partition"``,
    SPARTA-JAX's partition sort): ``O(N)`` passes and a sort of the slots that changed cell
    only.  The table has :func:`build_table`'s cells, starts and counts, and each cell the
    same members; within a cell the slots that kept it come first in their previous order,
    then the newcomers by slot -- an order NTC's uniform pair selection is indifferent to
    in distribution, though not draw for draw.  A step with more movers than the largest
    buffer (:data:`MOVER_TIERS`) takes the full sort.  Returns ``(table, carry)``."""
    C = num_cells(grid)
    n_keys = C + 1
    c = table_cells(state, grid)
    n = c.shape[0]
    stay = c[carry.order] == carry.key
    stay_i = stay.astype(jnp.int32)
    cs = prefix_count(stay) - stay_i  # exclusive
    n_movers = n - (cs[-1] + stay_i[-1])
    sizes = [max(1, n // k) for k in MOVER_TIERS]
    branch = jnp.sum(n_movers > jnp.asarray(sizes, jnp.int32), dtype=jnp.int32)
    branches = [
        functools.partial(_partition_update, n_keys=n_keys, size=size) for size in sizes
    ] + [functools.partial(_full_update, n_keys=n_keys)]
    order, key = jax.lax.switch(branch, branches, c, carry, stay, cs)
    # bounds[C] is the first dead slot
    bounds = jnp.searchsorted(key, jnp.arange(n_keys, dtype=jnp.int32), side="left")
    bounds = bounds.astype(jnp.int32)
    table = CellTable(order=order, starts=bounds[:C], counts=bounds[1:] - bounds[:C], cell=c)
    return table, SortCarry(order=order, key=key)
