"""Advection (spec section 9).

On a uniform grid with no embedded surfaces, SPARTA's general cell-walking loop
(``update.cpp:937-1029``) collapses to its own ``OPT`` fast path
(``update.cpp:764-864``): advance the full step, apply the face BC, recompute the
cell by arithmetic.  No walk, no per-crossing face test, no neighbour lookup --
which is also what keeps the trip count static.

M5 replaces this with real particle tracking through cut cells; the interface
(``move``) is what that rewrite must preserve.
"""

from __future__ import annotations

import jax

from jax_dsmc import boundary
from jax_dsmc.boundary import BC
from jax_dsmc.state import GridSpec, ParticleState

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def move(state: ParticleState, grid: GridSpec, dt, bc: BC = BC.PERIODIC) -> ParticleState:
    """Advance positions by one timestep and apply the boundary rule.

    Args:
        state: ``ParticleState``.
        grid: ``GridSpec``.
        dt: TRACED scalar timestep.
        bc: STATIC boundary rule.

    Only the first ``D`` components of ``v`` advect -- velocities stay
    3-component whatever ``D`` is (section 0.2), because VSS scattering and
    Chapman-Enskog transport theory both need all three.

    Dead particles are advected too (their slots carry inert data) and then masked
    by the caller; branching on ``alive`` here would be a data-dependent code path
    for no benefit.
    """
    return move_with_decisions(state, grid, dt, bc)[0]


def move_with_decisions(state: ParticleState, grid: GridSpec, dt, bc: BC = BC.PERIODIC):
    """:func:`move`, also returning the frozen boundary decision (``i32[N, D]``).

    Split out rather than folded into :func:`move` so the ordinary call sites keep
    their signature.  The time loop uses this form so the boundary fold index
    reaches ``Diagnostics.decision_checksum``; see :func:`jax_dsmc.boundary.fold_index`
    for why leaving it out silently invalidates every finite-difference check that
    involves a wall.
    """
    D = state.x.shape[-1]
    x_new = state.x + state.v[:, :D] * dt
    x, v, alive = boundary.apply(x_new, state.v, state.alive, grid, bc)
    idx = boundary.fold_index(x_new, grid, bc)
    state = ParticleState(x=x, v=v, species=state.species, alive=alive, logh_acc=state.logh_acc)
    return state, idx
