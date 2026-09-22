"""P3 random-time walls, with the conditional-time derivative made explicit.

A realized 1-D boundary history restricts the Gaussian travel time to an
interval. Reparameterizing the truncated Gaussian in that interval and scoring
the interval probability gives a consistent derivative of the regularized
transition, including position observables. Merely scoring the hit probability
while freezing an unconditional travel time omits this conditional derivative.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_dsmc import sampling
from jax_dsmc.geometry import first_hit
from jax_dsmc.grid import cell_of
from jax_dsmc.mesh import geometry as mesh_geometry
from jax_dsmc.safe import safe_log
from jax_dsmc.surface import SurfaceTallies, zero


def conditional_time(raw, z, lower, upper, dt, epsilon):
    """Value-preserving conditional Gaussian sample and interval log probability."""
    eps = jnp.where(epsilon > 0.0, epsilon, 1.0)
    finite_lo, finite_hi = jnp.isfinite(lower), jnp.isfinite(upper)
    zl = (jnp.where(finite_lo, lower, dt) - dt) / eps
    zu = (jnp.where(finite_hi, upper, dt) - dt) / eps
    fl = jnp.where(finite_lo, jax.scipy.special.ndtr(zl), 0.0)
    fu = jnp.where(finite_hi, jax.scipy.special.ndtr(zu), 1.0)
    probability = jnp.maximum(fu - fl, 1e-30)
    within = jax.lax.stop_gradient(
        jnp.clip((jax.scipy.special.ndtr(z) - fl) / probability, 0.0, 1.0)
    )
    quantile = jnp.clip(fl + within * probability, 1e-15, 1.0 - 1e-15)
    sampled = dt + eps * jax.scipy.special.ndtri(quantile)
    # Preserve the original random realization exactly, but use its conditional tangent.
    corrected = jax.lax.stop_gradient(raw) + (sampled - jax.lax.stop_gradient(sampled))
    time = jnp.maximum(jnp.where(epsilon > 0.0, corrected, dt), 0.0)
    logp = jnp.where(epsilon > 0.0, safe_log(probability), 0.0)
    return time, logp


def next_hit(x, v, lo, hi):
    vd = v[:, 0]
    moving = vd != 0.0
    side = vd > 0.0
    bound = jnp.where(side, hi, lo)
    distance = (bound - x[:, 0]) / jnp.where(moving, vd, 1.0)
    return jnp.where(moving, jnp.maximum(distance, 0.0), jnp.inf), side


class WallEvents(NamedTuple):
    """Per-slot, per-bounce record of the step's diffuse-face hits, for the ensemble adjoint.

    Shapes ``(K_bounce, N)`` (``(K_bounce, N, 3)`` for ``v_out``): pass ``k`` of the
    chronological tracker records the hit it resolved -- the DIFFUSE face (``-1`` if the
    pass resolved no diffuse hit for that slot), the laboratory velocity the particle left
    with, and whether the accommodation draw took the diffuse branch.  Every hit is scored
    separately, weighted by λ at the end of the move: the realized remainder of the flight
    is the unbiased sample of the value after the hit, so nothing is linearised in the
    number of hits per step.

    With an embedded surface (``mesh``), **every** element hit is recorded too, specular
    or diffuse, under the surface id ``2D + e`` (``2D`` for an analytic body): the shape
    adjoint replays the step's wall map from the element ids alone (plan section 0), so
    nothing else about the hit is stored in a forward record.

    The three trailing fields are ``None`` in a record and filled only by a *traced*
    replay (``advance(trace=True)``, :mod:`jax_dsmc.shape`): ``x_hit`` ``f64[K, N, D]``
    the hit point, ``v_in`` ``f64[K, N, 3]`` the incident velocity, ``normal``
    ``f64[K, N, 3]`` the unit normal used (zero-padded past ``D``) -- what per-hit
    rewards and the kernel's normal score need.
    """

    face: jax.Array
    v_out: jax.Array
    diffuse: jax.Array
    x_hit: jax.Array | None = None
    v_in: jax.Array | None = None
    normal: jax.Array | None = None
    time: jax.Array | None = None
    """``f64[K, N]`` time since the start of the step at which pass ``k``'s event happened
    (every event, box faces included; 0 where there was none) -- the hit-timing band
    term of the shape adjoint puts its cotangent here."""


class TrackerCounts(NamedTuple):
    """Per-step tracking diagnostics of the mesh path (all ``i32`` scalars).

    ``exhausted``: particles with an unresolved hit after ``max_bounces`` passes (stencil)
    or ``max_walk`` iterations (walk); ``surface_hits``: element / body hits resolved;
    ``stencil_overrun``: particles whose remaining move exceeded the stencil dilation;
    ``inside_body``: particles found inside the body after the move and killed.
    """

    exhausted: jax.Array
    surface_hits: jax.Array
    stencil_overrun: jax.Array
    inside_body: jax.Array


def no_wall_events(n, bounces, trace_dim=None):
    """Empty events; ``trace_dim=D`` allocates the traced-replay fields as well."""
    trace = (
        (
            jnp.zeros((bounces, n, trace_dim)),
            jnp.zeros((bounces, n, 3)),
            jnp.zeros((bounces, n, 3)),
            jnp.zeros((bounces, n)),
        )
        if trace_dim is not None
        else (None, None, None, None)
    )
    return WallEvents(
        jnp.full((bounces, n), -1, jnp.int32),
        jnp.zeros((bounces, n, 3)),
        jnp.zeros((bounces, n), bool),
        *trace,
    )


def _trace_hit(events, k, selected, x_hit, v_in, normal):
    """Fill pass ``k``'s traced fields for ``selected`` (no-op on an untraced record)."""
    if events.x_hit is None:
        return events
    sel = selected[:, None]
    return events._replace(
        x_hit=events.x_hit.at[k].set(jnp.where(sel, x_hit, events.x_hit[k])),
        v_in=events.v_in.at[k].set(jnp.where(sel, v_in, events.v_in[k])),
        normal=events.normal.at[k].set(jnp.where(sel, normal, events.normal[k])),
    )


def _face_tally(tally, f, selected, incident, outgoing, mass, velocity_wall):
    """Extensive momentum / energy / heat transferred to face ``f`` by ``selected``."""
    impulse = jnp.where(selected[:, None], mass * (incident - outgoing), 0.0).sum(axis=0)
    energy = jnp.where(
        selected,
        0.5 * mass * (jnp.sum(incident**2, axis=1) - jnp.sum(outgoing**2, axis=1)),
        0.0,
    ).sum()
    return SurfaceTallies(
        tally.hits.at[f].add(selected.sum(dtype=jnp.int32)),
        tally.momentum.at[f].add(impulse),
        tally.energy.at[f].add(energy),
        tally.heat.at[f].add(energy - jnp.dot(velocity_wall, impulse)),
    )


def _surface_tally(tally, surface, selected, incident, outgoing, mass, velocity_wall):
    """:func:`_face_tally` for a per-particle surface id (scatter-add over ``surface``)."""
    impulse = jnp.where(selected[:, None], mass * (incident - outgoing), 0.0)
    energy = jnp.where(
        selected,
        0.5 * mass * (jnp.sum(incident**2, axis=1) - jnp.sum(outgoing**2, axis=1)),
        0.0,
    )
    heat = energy - jnp.sum(velocity_wall * impulse, axis=1)
    return SurfaceTallies(
        tally.hits.at[surface].add(selected.astype(jnp.int32)),
        tally.momentum.at[surface].add(impulse),
        tally.energy.at[surface].add(energy),
        tally.heat.at[surface].add(heat),
    )


def _element_events(x_at, v, logp, tally, events, exclude, hit, surface, draws, k, operands):
    """The element branch of one pass: reflect off the embedded surface (SPARTA
    ``surf_collide_specular.cpp:75-150`` / ``surf_collide_diffuse.cpp:206-306``).

    Every element carries the ``diffuse T acc`` law of its model row of ``WallParams``
    (``2D + model``; ``acc = 0`` is specular).  The normal is the flat element normal
    (``mesh.normals[e]``) or the exact one of an analytic body, in the flow direction;
    the diffuse frame is :func:`sampling._tangent_frame`.  Element models use the
    isotropic wall temperature (``wall.temp[row]`` or its first component).
    """
    grid, wall, mass, acc_delta, tallies, mesh = operands
    dim = x_at.shape[1]
    selected = hit & (surface >= 2 * dim)
    e = jnp.clip(surface - 2 * dim, 0, mesh.n_elements - 1)
    model = mesh.body.model[e] if mesh.kind == "surface" else jnp.zeros_like(e)
    row = 2 * dim + model
    normal = (
        jnp.zeros((x_at.shape[0], 3))
        .at[:, :dim]
        .set(mesh_geometry.surface_normal(mesh, surface, x_at, grid))
    )
    vn = jnp.sum(v * normal, axis=1, keepdims=True)
    v_spec = v - 2.0 * vn * normal
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[row]
    acc = wall.acc[row]
    vwall = wall.vwall[row]
    z, r, u = draws
    fresh = sampling.diffuse_velocity_general(z, r, normal, temp[:, None], mass, vwall, v)
    # the diffuse branch is a KERNEL to the adjoint: its parameters (temperature, drift,
    # accommodation) and the normal enter the VALUE after the hit through the score of
    # `sampling.log_diffuse_density`, never pathwise through the sampled velocity (review
    # section 5.9: a field's fitted derivative is a poor source term); the post-hit state
    # and the events therefore carry the frozen sample.  The hit's OWN reward is exact in
    # `v_out`, so the surface tally -- the pathwise entry of a per-hit reward -- is formed
    # from the traced sample: the reparameterised `d/dn E[r(v_out)]` with bounded weights,
    # which the shape adjoint pairs with the score weighted by the value-to-go AFTER this
    # hit (review section 7.8, GP)
    fresh_traced = fresh
    fresh = jax.lax.stop_gradient(fresh)
    # `surf_collide_diffuse.cpp:211` -- `random->uniform() > acc` takes the specular branch
    took_diffuse = u <= acc
    safe_delta = jnp.where(acc_delta > 0.0, acc_delta, 1.0)
    w = jnp.where(
        acc_delta > 0.0,
        jax.nn.sigmoid((acc - u) / safe_delta),
        took_diffuse.astype(jnp.float64),
    )
    v_out = w[:, None] * fresh + (1.0 - w)[:, None] * v_spec
    log_p = jnp.where(took_diffuse, safe_log(acc), safe_log(1.0 - acc))
    log_p = jnp.where(acc_delta > 0.0, 0.0, log_p)
    v_new = jnp.where(selected[:, None], v_out, v)
    v_new_tally = jnp.where(
        selected[:, None], w[:, None] * fresh_traced + (1.0 - w)[:, None] * v_spec, v
    )
    logp = logp + jnp.where(selected, log_p, 0.0)
    events = events._replace(
        face=events.face.at[k].set(jnp.where(selected, surface, events.face[k])),
        v_out=events.v_out.at[k].set(jnp.where(selected[:, None], v_out, events.v_out[k])),
        diffuse=events.diffuse.at[k].set(jnp.where(selected, took_diffuse, events.diffuse[k])),
    )
    events = _trace_hit(events, k, selected, x_at, v, normal)
    # SPARTA `update.cpp`: the element just left is excluded from the next search
    exclude = jnp.where(selected, e, exclude)
    if tallies:
        tally = _surface_tally(tally, surface, selected, v, v_new_tally, mass, vwall)
    return v_new, logp, tally, events, exclude


def _apply_events(operands, grid, faces, wall, mass, acc_delta, tallies, mesh=None):
    """Resolve the events of one pass for the particles flagged ``hit``."""
    from jax_dsmc.boundary import (  # noqa: PLC0415 -- break the boundary dispatch import cycle
        BC,
        diffuse_from_draws,
    )

    x, v, alive, logp, tally, events, hit, surface, used, pass_key, k, exclude, at_time = operands
    dim = x.shape[1]
    n = x.shape[0]
    k_draw, k_acc = jax.random.split(pass_key)
    z, r = sampling.standard_diffuse_draws(k_draw, n)
    u_acc = jax.random.uniform(k_acc, (n,), dtype=jnp.float64)
    draws = (z, r, u_acc)
    x_at = x + used[:, None] * v[:, :dim]
    if events.time is not None:
        events = events._replace(time=events.time.at[k].set(jnp.where(hit, at_time, 0.0)))
    for axis in range(dim):
        for s in (0, 1):
            f = 2 * axis + s
            selected = hit & (surface == f)
            rule = faces.of(axis, s)
            incident = v
            if rule is BC.DIFFUSE:
                new_v, lp = diffuse_from_draws(v, wall, mass, axis, s, draws, acc_delta)
                v = jnp.where(selected[:, None], new_v, v)
                logp = logp + jnp.where(selected, lp, 0.0)
                events = events._replace(
                    face=events.face.at[k].set(jnp.where(selected, f, events.face[k])),
                    v_out=events.v_out.at[k].set(
                        jnp.where(selected[:, None], new_v, events.v_out[k])
                    ),
                    diffuse=events.diffuse.at[k].set(
                        jnp.where(selected, u_acc <= wall.acc[f], events.diffuse[k])
                    ),
                )
                n_face = jnp.zeros((n, 3)).at[:, axis].set(-1.0 if s else 1.0)
                events = _trace_hit(events, k, selected, x_at, incident, n_face)
            elif rule is BC.SPECULAR:
                v = v.at[:, axis].set(jnp.where(selected, -v[:, axis], v[:, axis]))
            elif rule is BC.PERIODIC:
                x_at = x_at.at[:, axis].set(
                    jnp.where(selected, grid.lo[axis] if s else grid.hi[axis], x_at[:, axis])
                )
            else:
                alive = alive & ~selected
            if tallies:
                outgoing = jnp.where(alive[:, None], v, 0.0)
                velocity_wall = wall.vwall[f] if rule is BC.DIFFUSE else jnp.zeros(3)
                tally = _face_tally(tally, f, selected, incident, outgoing, mass, velocity_wall)
    # a box-face event ends the exclusion (SPARTA resets `exclude` on leaving the cell)
    exclude = jnp.where(hit & (surface < 2 * dim), -1, exclude)
    if mesh is not None:
        v, logp, tally, events, exclude = _element_events(
            x_at,
            v,
            logp,
            tally,
            events,
            exclude,
            hit,
            surface,
            draws,
            k,
            (grid, wall, mass, acc_delta, tallies, mesh),
        )
    x = jnp.where(hit[:, None], x_at, x)
    return x, v, alive, logp, tally, events, exclude


def _skip_events(operands):
    x, v, alive, logp, tally, events, *_rest, exclude, _at_time = operands
    return x, v, alive, logp, tally, events, exclude


def _bounce(k, carry, grid, faces, wall, mass, key, acc_delta, tallies, mesh=None):
    """One chronological event pass over every particle.  ``fori_loop`` body.

    Module-level and bound with ``functools.partial`` (decision 14).  Two cost
    decisions, both measured (scratch/bench/b6_wall_cost.py):

    * the diffuse randomness -- a standard normal triple, a standard Rayleigh speed
      and the accommodation uniform -- is drawn **once per pass**, not once per
      face: a particle meets at most one face per pass, so every face's re-emission
      law is unchanged and the pass costs one ``(n, 3)`` draw instead of ``2D``;
    * a pass in which **no** particle reaches a face (the usual case for every pass
      after the first) skips the draws, the face loop and the tallies through a
      ``lax.cond``; only the ``first_hit`` query is paid.  Draws happen inside the
      taken branch, so a skipped pass consumes no random numbers.
    """
    x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf = carry
    distance, surface = _next_surface(x, v, grid, mesh, exclude, budget - elapsed)
    hit = alive & (elapsed + distance <= budget)
    used = jnp.where(hit, distance, 0.0)
    if mesh is not None:
        # the stencil's sufficient condition, tested on the whole remaining move
        ok = mesh_geometry.stencil_sufficient(v, budget - elapsed, grid, mesh, x)
        overrun = overrun | (alive & ~ok)
        n_surf = n_surf + jnp.sum(hit & (surface >= 2 * x.shape[1]), dtype=jnp.int32)
    operands = (
        x,
        v,
        alive,
        logp,
        tally,
        events,
        hit,
        surface,
        used,
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
    elapsed = elapsed + used
    had_hit = had_hit | hit
    return x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf


def _next_surface(x, v, grid, mesh, exclude, remaining):
    """Earliest surface along the ray: box faces only, or box faces and the stencil's
    elements (:func:`jax_dsmc.mesh.geometry.first_hit`) when a mesh is present."""
    if mesh is None:
        return first_hit(x, v, grid)
    return mesh_geometry.first_hit(x, v, grid, mesh, exclude, remaining)


def _advance_impl(
    x,
    v,
    alive,
    grid,
    faces,
    wall,
    mass,
    dt,
    key,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    return_tallies=False,
    return_events=False,
    mesh=None,
    tracker="stencil",
    max_walk=8,
    trace=False,
):
    """Chronological first-event tracking and the P3 wall gradient; returns the
    ``apply_faces`` tuple.

    Specular, diffuse, periodic and absorbing faces participate in the same
    chronological event search in any ``D``. ``recrossed`` reports exhausted event
    capacity.  All parameters use physical units, including the wall temperature in
    kelvin.  The conditional-time estimator is validated by one-particle quadrature
    in 1-D; the deterministic tracking agrees with the legacy fold path in D=1,2,3.

    With ``mesh`` (a :class:`jax_dsmc.mesh.types.Mesh`) the embedded surface joins the
    same event search.  ``tracker="stencil"`` keeps the pass structure above with the
    per-cell candidate list (``mesh.candidates``) and counts ``stencil_overrun``;
    ``tracker="walk"`` is SPARTA's cell walk (:mod:`jax_dsmc.mesh.walk`, ``max_walk``
    iterations of one cell crossing or one hit each), exact for any move length.  Both
    end with the inside-body check that kills and counts a particle a missed hit left in
    the body.  The fourth element of the result is then a :class:`TrackerCounts`
    instead of the bare ``exhausted`` count, and the tallies have ``2D + E`` rows.

    ``trace=True`` (with ``return_events``) also fills the hit point, incident velocity
    and normal of every recorded hit (:class:`WallEvents`); the shape adjoint's replay
    uses it, a forward record never does.
    """
    n = x.shape[0]
    dim = x.shape[1]
    z = jax.random.normal(jax.random.fold_in(key, 0), (n,), dtype=jnp.float64)
    raw = dt + epsilon * z
    budget = jnp.maximum(raw, 0.0)
    originally_alive = alive
    n_surfaces = 2 * dim + (mesh.n_elements if mesh is not None else 0)
    init = (
        x,
        v,
        alive,
        jnp.zeros(n),
        jnp.zeros(n),
        jnp.zeros(n, bool),
        zero(n_surfaces),
        no_wall_events(
            n, max_bounces if tracker == "stencil" else max_walk, dim if trace else None
        ),
        budget,
        jnp.full((n,), -1, jnp.int32),
        jnp.zeros(n, bool),
        jnp.int32(0),
    )
    if mesh is None or tracker == "stencil":
        body = functools.partial(
            _bounce,
            grid=grid,
            faces=faces,
            wall=wall,
            mass=mass,
            key=key,
            acc_delta=acc_delta,
            tallies=return_tallies,
            mesh=mesh,
        )
        carry = jax.lax.fori_loop(0, max_bounces, body, init)
        x, v, alive, elapsed, logp, had_hit, tally, events, _, exclude, overrun, n_surf = carry
        distance, _ = _next_surface(x, v, grid, mesh, exclude, None)
        exhausted = alive & (elapsed + distance <= budget)
    elif tracker == "walk":
        from jax_dsmc.mesh.walk import walk  # noqa: PLC0415 -- walk imports this module

        carry, stuck = walk(
            init, grid, faces, wall, mass, key, acc_delta, return_tallies, mesh, max_walk
        )
        x, v, alive, elapsed, logp, had_hit, tally, events, _, exclude, overrun, n_surf = carry
        # SPARTA `MAXSTUCK`: a particle the walk could not finish is discarded
        exhausted = stuck
        alive = alive & ~stuck
        # the conditional-time interval needs the next surface; the stencil query is exact
        # for it whenever a near-miss within the regularisation is at stake
        distance, _ = _next_surface(x, v, grid, mesh, exclude, None)
    else:
        raise ValueError(f"tracker must be 'stencil' or 'walk', got {tracker!r}")
    lower = jnp.where(had_hit, elapsed, -jnp.inf)
    upper = jnp.where(alive, elapsed + distance, jnp.inf)
    time, interval_score = conditional_time(raw, z, lower, upper, dt, epsilon)
    x = x + jnp.where(alive, time - elapsed, 0.0)[:, None] * v[:, :dim]
    logp = logp + interval_score
    x = jnp.clip(x, grid.lo, grid.hi)
    counts = jnp.sum(exhausted)
    if mesh is not None:
        # SPARTA's catch-all: a particle inside the body after the move was mis-tracked
        inside = alive & mesh_geometry.inside_polygon(x, cell_of(x, grid), mesh, grid)
        alive = alive & ~inside
        counts = TrackerCounts(
            jnp.sum(exhausted, dtype=jnp.int32),
            n_surf,
            jnp.sum(overrun, dtype=jnp.int32),
            jnp.sum(inside, dtype=jnp.int32),
        )
    result = (
        x,
        v,
        alive,
        counts,
        jnp.where(originally_alive, logp, 0.0),
    )

    if return_events:
        result = (*result, events)
    return (*result, tally) if return_tallies else result


_advance_jitted = jax.jit(
    _advance_impl,
    static_argnames=(
        "faces",
        "max_bounces",
        "return_tallies",
        "return_events",
        "tracker",
        "max_walk",
        "trace",
    ),
)


@functools.wraps(_advance_impl)
def advance(
    x,
    v,
    alive,
    grid,
    faces,
    wall,
    mass,
    dt,
    key,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    return_tallies=False,
    return_events=False,
    mesh=None,
    tracker="stencil",
    max_walk=8,
    trace=False,
):
    # Eager calls need a stable compilation boundary; an enclosing jit/grad already
    # supplies tracing. Bypass the extra boundary there to keep its AD graph small.
    leaves = jax.tree.leaves((x, v, alive, grid, wall, mass, dt, key, epsilon, acc_delta, mesh))
    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
    kernel = _advance_impl if traced else _advance_jitted
    return kernel(
        x,
        v,
        alive,
        grid,
        faces,
        wall,
        mass,
        dt,
        key,
        epsilon=epsilon,
        acc_delta=acc_delta,
        max_bounces=max_bounces,
        return_tallies=return_tallies,
        return_events=return_events,
        mesh=mesh,
        tracker=tracker,
        max_walk=max_walk,
        trace=trace,
    )
