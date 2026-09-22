"""Uniform orthogonal mesh — the narrow mesh interface (spec section 4).

Four functions constitute the entire mesh interface: :func:`cell_of`,
:func:`cell_volume`, :func:`cell_bounds` and :func:`cell_centers`.  Milestone 5
replaces the implementation behind these names with cut/split cells and general
particle tracking, so **no cell-index arithmetic may leak out of this module**:
if another module ever computes ``floor((x - lo) / dx)`` itself, the M5 rewrite
breaks it silently.

Traced/static split (section 0.4): ``grid.lo`` and ``grid.hi`` are traced
(differentiable) leaves; ``grid.n`` is a static tuple of Python ints, because
``C = prod(n)`` is the shape of every per-cell array.

Reference: SPARTA ``grid.cpp`` / ``Grid::id_find_child``; the clip-at-the-upper-
edge convention matches SPARTA's rule that a boundary particle is never lost.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def num_cells(grid: GridSpec) -> int:
    """Total number of cells ``C = prod(n)``.  STATIC (a Python int)."""
    c = 1
    for nd in grid.n:
        c *= int(nd)
    return c


def spacing(grid: GridSpec) -> jnp.ndarray:
    """Cell size per axis, ``f64[D]``.  TRACED through ``lo``/``hi``."""
    return (grid.hi - grid.lo) / jnp.asarray(grid.n, dtype=jnp.float64)


def _row_major_strides(n) -> tuple[int, ...]:
    """Row-major strides for shape ``n``.  All-static Python ints.

    ``strides[-1] == 1`` and ``strides[d] = prod(n[d+1:])``, i.e. the *last* axis
    varies fastest -- the same convention as ``jnp.unravel_index`` used by
    :func:`_cell_multi_index`, so the two are exact inverses.
    """
    strides = [1] * len(n)
    for d in range(len(n) - 2, -1, -1):
        strides[d] = strides[d + 1] * int(n[d + 1])
    return tuple(strides)


def cell_of(x: jnp.ndarray, grid: GridSpec) -> jnp.ndarray:
    """Cell index of each particle.

    Args:
        x: ``f64[N, D]`` positions (traced).
        grid: ``GridSpec``; ``lo``/``hi`` traced, ``n`` static.

    Returns:
        ``i32[N]`` row-major cell index in ``[0, C)``.

    Cell membership is a **frozen decision** (section 0.5): the result is an
    integer, so it carries no tangent, and the float index is wrapped in
    ``stop_gradient`` as an explicit guard against a future refactor opening a
    gradient path through the ``floor``.

    A particle exactly at ``lo`` lands in cell 0 and one exactly at ``hi`` lands
    in cell ``C - 1``: the ``clip`` absorbs the upper edge rather than losing the
    particle, matching SPARTA.
    """
    dx = spacing(grid)
    idx = jnp.floor(lax.stop_gradient((x - grid.lo) / dx))  # f64[N, D]
    n = jnp.asarray(grid.n, dtype=jnp.int32)
    idx = jnp.clip(idx.astype(jnp.int32), 0, n - 1)  # i32[N, D]
    # Row-major ravel by explicit static strides.  NOT `jnp.ravel_multi_index`:
    # its default `mode="raise"` needs concrete indices to bounds-check and so
    # raises ConcretizationTypeError under `jit`.  `mode="clip"` would trace, but
    # it would also silently absorb an out-of-range index -- exactly the class of
    # defect section 8.4 was built to avoid.  The `clip` above is the one and only
    # place a boundary particle is folded back in, and the strides below are pure
    # static arithmetic because `grid.n` is static.
    strides = _row_major_strides(grid.n)
    flat = idx[:, 0] * strides[0]
    for d in range(1, len(strides)):
        flat = flat + idx[:, d] * strides[d]
    return flat.astype(jnp.int32)


def neighbour_cell(cell: jnp.ndarray, axis: int, side: int, grid: GridSpec) -> tuple:
    """The cell across face ``(axis, side)`` of ``cell``: ``(i32[N] index, bool[N] exists)``.

    ``exists`` is False when that face of the cell is a face of the box; the index is
    then the cell itself.  This is the one piece of neighbour arithmetic the cell-walk
    tracker (:mod:`jax_dsmc.mesh.walk`) needs, kept here so a hierarchical grid changes
    only this module (section 4).
    """
    strides = _row_major_strides(grid.n)
    n_axis = int(grid.n[axis])
    idx = (cell // strides[axis]) % n_axis
    target = idx + (1 if side else -1)
    exists = (target >= 0) & (target < n_axis)
    return jnp.where(exists, cell + (1 if side else -1) * strides[axis], cell).astype(
        jnp.int32
    ), exists


def cell_volume(grid: GridSpec) -> jnp.ndarray:
    """Volume of every cell, ``f64[C]``.  TRACED through ``lo``/``hi``.

    Returned as a full ``(C,)`` array rather than a scalar deliberately (section
    4.1): it costs nothing on a uniform mesh and makes the non-uniform M5 mesh a
    drop-in replacement.  This is one of the two routes by which a gradient
    reaches the mesh geometry (the NTC attempt count of section 8.1; the other is
    moment normalisation in ``sample.py``).

    A cut-cell grid carries its per-cell flow volumes in ``grid.volume`` (set by
    ``mesh.cut.build_mesh``); those are returned as they are.
    """
    if grid.volume.shape[0] > 0:
        return grid.volume
    dx = spacing(grid)
    # `transverse` makes this a TRUE volume at D < 3: a D-dimensional run is a
    # 3-D gas uniform in the 3-D suppressed directions, so `prod(dx)` alone is a
    # length at D=1 and an area at D=2.  It is 1.0 at D=3.  See `GridSpec`.
    return jnp.full((num_cells(grid),), jnp.prod(dx) * grid.transverse)


def _cell_multi_index(grid: GridSpec) -> jnp.ndarray:
    """``i32[C, D]`` per-axis index of every cell, row-major."""
    C = num_cells(grid)
    axes = jnp.unravel_index(jnp.arange(C, dtype=jnp.int32), tuple(int(nd) for nd in grid.n))
    return jnp.stack(axes, axis=-1).astype(jnp.int32)


def cell_neighbourhood(grid: GridSpec, radius: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The ``(2 radius + 1)^D`` block of cells around every cell: ``(i32[C, B] index,
    bool[C, B] exists)``; a block position outside the box has ``exists`` False and the
    cell's own index.  Static shapes; no periodic wrap (the pooled fits of the ensemble
    adjoint only need the neighbours that share the cell's surface)."""
    multi = _cell_multi_index(grid)  # (C, D)
    d = len(grid.n)
    offsets = jnp.stack(
        jnp.meshgrid(*([jnp.arange(-radius, radius + 1)] * d), indexing="ij"), axis=-1
    ).reshape(-1, d)  # (B, D)
    target = multi[:, None, :] + offsets[None, :, :]
    n = jnp.asarray(grid.n)
    exists = jnp.all((target >= 0) & (target < n), axis=-1)
    strides = jnp.asarray(_row_major_strides(grid.n))
    index = jnp.sum(jnp.clip(target, 0, n - 1) * strides, axis=-1).astype(jnp.int32)
    own = jnp.arange(index.shape[0], dtype=jnp.int32)[:, None]
    return jnp.where(exists, index, own), exists


def cell_bounds(grid: GridSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lower and upper corner of every cell, ``(f64[C, D], f64[C, D])``.  TRACED."""
    dx = spacing(grid)
    idx = _cell_multi_index(grid).astype(jnp.float64)
    lower = grid.lo + idx * dx
    upper = lower + dx
    return lower, upper


def cell_centers(grid: GridSpec) -> jnp.ndarray:
    """Centre of every cell, ``f64[C, D]``.  TRACED."""
    lower, upper = cell_bounds(grid)
    return 0.5 * (lower + upper)
