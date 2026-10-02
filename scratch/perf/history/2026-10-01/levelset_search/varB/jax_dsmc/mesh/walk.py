"""SPARTA's cell walk (``update.cpp:879-1547``) as a bounded ``fori_loop``.

The robust tracker: the particle is walked cell by cell, so only the cell's own element
list (``mesh.csurfs``, SPARTA ``csurfs``) is ever searched and a move of *any* length is
tracked exactly, at the cost of one iteration per cell crossing.  Each iteration resolves
one event per particle:

* the earliest of the cell's elements hit *strictly* before the exit face (``t_el <
  t_exit``, SPARTA ``update.cpp:1180-1215``): reflect (shared element branch of
  :func:`jax_dsmc.wall_adjoint._apply_events`), set ``exclude``;
* else the exit face: if it is a face of the box, the box rule (same routine); else move
  to the face, snap the crossed coordinate onto it, step to the neighbour cell
  (:func:`jax_dsmc.grid.neighbour_cell`) and reset ``exclude`` (``update.cpp:1368-1430``);
* else the move ends inside the cell: done.

A particle still moving after ``max_walk`` iterations is SPARTA's stuck particle
(``MAXSTUCK``): reported as ``walk_exhausted`` and discarded by the caller.  Random draws
follow the pass structure of the stencil tracker (one draw set per iteration under
``fold_in(key, 1 + k)``), so the two trackers are statistically identical and bitwise
identical on specular surfaces.  No nested function definitions (HANDOFF section 3).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jax_dsmc.grid import cell_bounds, cell_of, neighbour_cell
from jax_dsmc.mesh import analytic
from jax_dsmc.mesh import geometry as mesh_geometry
from jax_dsmc.mesh import levelset as levelset_mod


def _exit_face(x, vd, lo_c, hi_c):
    """Time to leave the cell box ``[lo_c, hi_c]`` and the face crossed: ``(t, axis, side)``."""
    moving = vd != 0.0
    side = vd > 0.0
    bound = jnp.where(side, hi_c, lo_c)
    t = jnp.where(moving, jnp.maximum((bound - x) / jnp.where(moving, vd, 1.0), 0.0), jnp.inf)
    axis = jnp.argmin(t, axis=1)
    rows = jnp.arange(x.shape[0])
    return t[rows, axis], axis, side[rows, axis]


def _cell_elements(x, vd, mesh, cell, exclude, remaining):
    """Earliest element of the *cell's own* list along the ray: ``(t f64[N], e i32[N])``."""
    if mesh.kind == "surface":
        ids = mesh.csurfs[cell]
        t = mesh_geometry._element_times(x, vd, mesh, ids)
        t = jnp.where(ids == exclude[:, None], jnp.inf, t)
        t_el = jnp.min(t, axis=1)
        big = jnp.iinfo(jnp.int32).max
        e = jnp.min(jnp.where(t == t_el[:, None], ids, big), axis=1)
        return t_el, e
    if mesh.kind == "levelset":
        return levelset_mod.hit_body(x, vd, mesh.body, remaining), jnp.zeros_like(exclude)
    # an analytic body: the hit is wherever it is; the cell test below keeps it local
    t_el = analytic.hit_body(x, vd, mesh.body, exclude == 0)
    return t_el, jnp.zeros_like(exclude)


def _walk_step(k, carry, grid, faces, wall, mass, key, acc_delta, tallies, mesh):
    """One iteration of the walk for every particle.  ``fori_loop`` body."""
    from jax_dsmc.wall_adjoint import _apply_events, _skip_events  # noqa: PLC0415 -- cycle

    (
        (x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf),
        (
            cell,
            done,
        ),
    ) = carry
    dim = x.shape[1]
    vd = v[:, :dim]
    lower, upper = cell_bounds(grid)
    lo_c = lower[cell]
    hi_c = upper[cell]
    t_exit, axis, side = _exit_face(x, vd, lo_c, hi_c)
    remaining = budget - elapsed
    t_el, e = _cell_elements(x, vd, mesh, cell, exclude, remaining)
    active = alive & ~done
    element_hit = active & (t_el < t_exit) & (t_el <= remaining)
    crossing = active & ~element_hit & (t_exit <= remaining)
    finished = active & ~element_hit & ~crossing
    # which neighbour lies across the exit face (static per-axis arithmetic, selected)
    neighbour = cell
    exists = jnp.zeros_like(alive)
    for a in range(dim):
        for s in (0, 1):
            nb, ok = neighbour_cell(cell, a, s, grid)
            pick = (axis == a) & (side == s)
            neighbour = jnp.where(pick, nb, neighbour)
            exists = jnp.where(pick, ok, exists)
    box_hit = crossing & ~exists
    interior = crossing & exists
    hit = element_hit | box_hit
    surface = jnp.where(element_hit, 2 * dim + e, 2 * axis + side.astype(jnp.int32))
    used = jnp.where(element_hit, t_el, jnp.where(crossing, t_exit, 0.0))
    n_surf = n_surf + jnp.sum(element_hit, dtype=jnp.int32)
    operands = (
        x,
        v,
        alive,
        logp,
        tally,
        events,
        hit,
        surface.astype(jnp.int32),
        jnp.where(hit, used, 0.0),
        jax.random.fold_in(key, 1 + k),
        k,
        exclude,
        elapsed + used,
    )
    x, v, alive, logp, tally, events, exclude = jax.lax.cond(
        jnp.any(hit),
        functools.partial(
            _apply_events,
            grid=grid,
            faces=faces,
            wall=wall,
            mass=mass,
            acc_delta=acc_delta,
            tallies=tallies,
            mesh=mesh,
        ),
        _skip_events,
        operands,
    )
    # interior crossing: onto the face exactly, into the neighbour, exclusion lifted
    x_face = x + t_exit[:, None] * vd
    bound = jnp.where(side[:, None], hi_c, lo_c)
    x_face = jnp.where(jnp.arange(dim)[None, :] == axis[:, None], bound, x_face)
    x = jnp.where(interior[:, None], x_face, x)
    cell = jnp.where(interior, neighbour, cell)
    exclude = jnp.where(interior, -1, exclude)
    # a box event leaves the particle on the box face: its cell is the one `cell_of` picks
    cell = jnp.where(box_hit, cell_of(x, grid), cell)
    elapsed = elapsed + used
    had_hit = had_hit | hit
    done = done | finished
    inner = (x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf)
    return inner, (cell, done)


def walk(init, grid, faces, wall, mass, key, acc_delta, tallies, mesh, max_walk):
    """Run the cell walk on the tracker carry ``init`` (see ``wall_adjoint._advance_impl``).

    Returns ``(carry, stuck)``: the carry after ``max_walk`` iterations and the
    ``bool[N]`` of alive particles whose move is unfinished.
    """
    x = init[0]
    body = functools.partial(
        _walk_step,
        grid=grid,
        faces=faces,
        wall=wall,
        mass=mass,
        key=key,
        acc_delta=acc_delta,
        tallies=tallies,
        mesh=mesh,
    )
    cell0 = cell_of(x, grid)
    done0 = jnp.zeros(x.shape[0], bool)
    carry, (_cell, done) = jax.lax.fori_loop(0, max_walk, body, (init, (cell0, done0)))
    alive = carry[2]
    return carry, alive & ~done
