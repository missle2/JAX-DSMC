"""Cell occupancy table (spec section 6).

SPARTA builds per-cell linked lists (``collide.cpp:508-512``).  That is a
CPU-appropriate structure; on GPU the right primitive is a sort.

Particle ``j`` of cell ``k`` is ``order[starts[k] + j]``, valid for
``j < counts[k]``.

Differentiability (section 6.1): ``order`` is an integer permutation — a frozen
decision (section 0.5).  The *gather* ``v[order]`` is linear, so its VJP is a
scatter and cotangents flow back to the correct particles automatically.
Nothing special is needed.

Scaling (section 6.2): ``argsort`` is ``O(N log N)`` and becomes the bottleneck
around ``N ~ 1e9``, where a counting sort (``bincount`` + prefix sum + scatter)
does it in ``O(N)``.  Deferred; the interface does not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_dsmc.grid import collision_cell, num_cells
from jax_dsmc.state import CellTable, GridSpec, ParticleState

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


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
    order = jnp.argsort(c).astype(jnp.int32)  # i32[N]  FROZEN (int)
    counts = jnp.bincount(c, length=C + 1).astype(jnp.int32)  # i32[C+1]
    starts = (jnp.cumsum(counts) - counts).astype(jnp.int32)  # exclusive scan
    return CellTable(order=order, starts=starts[:C], counts=counts[:C], cell=c)
