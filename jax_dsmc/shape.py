"""Shape adjoint: the VJP of the replayed wall map with respect to the geometry.

The ensemble adjoint (:mod:`jax_dsmc.ensemble_device`) carries the value ``lambda`` on the
particles and never linearises the geometry.  The geometry enters the forward only
through the move (:func:`jax_dsmc.boundary.apply_faces` -> :func:`jax_dsmc.wall_adjoint.advance`):
hit times, hit points, specular reflections and the normals of diffuse re-emissions.
Under the RNG contract that move is a pure function of the pre-move state, the step's
key and the geometry, so it can be **replayed** from a captured
:class:`jax_dsmc.simulate.EventRecord` and differentiated by ``jax.vjp`` with the
cotangents the ensemble backward provides (plan section 2, item 4):

* ``grad_x lambda``, ``grad_v lambda`` at every survivor's own post-move state (from the
  per-cell fit, :func:`jax_dsmc.ensemble_device.node_gradients`) on the replayed
  post-move state -- the pathwise term of hit times and specular reflections;
* ``lambda + r`` (the value after the move plus the step's hit rewards) on the move's
  log-probability (the epsilon hit-timing and accommodation branches);
* the objective's own weights on the step's surface tallies -- the pathwise part of a
  per-hit reward such as drag;
* ``(lambda + r_after) d/dn log q(v_out | n)`` on the normal of every diffuse-branch
  element hit (``r_after``: the rewards of the step's later hits; the hit's own reward
  is exact in ``v_out`` and enters pathwise through the tally of the traced sample --
  the reparameterised form with bounded weights, review section 7.8 -- when a tally
  cotangent is given, and joins the score weight otherwise);
* optionally the **hit-timing band term** on the event times: a reward that jumps at a
  hit (a velocity-dependent per-step reward; a per-hit reward at the window's edges)
  makes the expected objective depend on which step a hit falls in, a dependence the
  pathwise derivative of a fixed hit set cannot see.  Its density is estimated from the
  hits within a band ``Delta dt`` of the step end (:func:`timing_cotangent`), the
  value-preserving analogue of the epsilon hit-timing score with no forward
  regularisation (gate G3: the pathwise term alone read ``+0.82 +- 0.23`` for the
  plane's windowed ``v_y^2`` against an FD of ``+0.05 +- 0.15``; the band term is
  ``-0.43 +- 0.02``). -- the kernel score of the re-emission law's dependence on the normal
  (:func:`jax_dsmc.sampling.log_diffuse_density`; the sampled velocity itself is
  ``stop_gradient`` in :func:`jax_dsmc.wall_adjoint._element_events`, review section 5.9).

The geometry is the body's traced pytree: ``AnalyticBody`` (centre, radius) or the
``points`` of a ``Surface``; :func:`with_geometry` rebuilds the metric tables (normals,
areas) from it.  Cut volumes are NOT re-derived (the cut-volume term of the collision
step is deferred to gate G5) and the cell tables are fixed by design (no membership
term).  What the VJP cannot see -- the element-assignment jumps of a flat-normal
polygon and the silhouette of a diffuse body -- is the edge-band score of plan
section 2, item 5, not part of this module.

Everything here is module-level and ``lax.scan``-driven (decision 14); the per-step
replay is validated against the record (``replay_error``, roundoff at most).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc import boundary, ensemble_device, sampling
from jax_dsmc.ensemble_device import node_gradients
from jax_dsmc.mesh import geometry as mesh_geometry
from jax_dsmc.mesh import levelset as levelset_mod
from jax_dsmc.mesh import surface as surface_mod
from jax_dsmc.mesh.types import Mesh
from jax_dsmc.rng import STAGE_MOVE, key_for
from jax_dsmc.sampling import log_diffuse_density

jax.config.update("jax_enable_x64", True)


class Replay(NamedTuple):
    hit_reward: jax.Array
    """``f64[M, K, N]`` per-hit rewards of the objective (zeros without ``reward``)."""
    tallies: object
    """Per-step :class:`SurfaceTallies` of the replayed moves."""
    replay_error: jax.Array
    """As :class:`ShapeTerms`."""
    surface_hits: jax.Array
    """``i32[M]``."""
    timing_reward: jax.Array
    """``f64[M, K, N]`` the hit-timing band term per hit (``timing_cotangent`` of the
    objective's ``timing_jump``): the change of the objective per unit delay of the
    particle's trajectory, what the along-flight identity's second backward carries."""


class ShapeTerms(NamedTuple):
    pathwise: object
    """Geometry-shaped: ``grad lambda`` on the replayed post-move state and the
    objective's weights on the tallies (the sum of the three parts below)."""
    pathwise_x: object
    """Geometry-shaped: the ``grad_x lambda`` part alone (hit times and hit points)."""
    pathwise_v: object
    """Geometry-shaped: the ``grad_v lambda`` part alone (reflected velocities)."""
    pathwise_tally: object
    """Geometry-shaped: the objective's tally weights alone (a per-hit reward's own
    dependence on the geometry through the reflected velocity)."""
    score: object
    """Geometry-shaped: ``lambda + r`` on the move's log-probability and the normal
    score of the diffuse-branch element hits."""
    timing: object
    """Geometry-shaped: the hit-timing band term (:func:`timing_cotangent`); zero unless
    ``gradient`` was given ``timing_jump``."""
    silhouette: object
    """Geometry-shaped: the silhouette band term (:func:`silhouette_cotangent`): the value
    difference between a hit and a miss for the particles whose undeflected flight passes
    within ``silhouette_band`` of the surface, times the rate at which the surface's
    motion converts one into the other.  Zero unless ``gradient`` was given a band."""
    replay_error: jax.Array
    """``f64`` largest relative ``|x_post - record|`` (per box extent) / ``|v_post - record|``
    (per the record's largest speed) over survivors, all steps.  The replay is the same
    arithmetic as the forward under the RNG contract but compiled in another context, so
    a few ulp (``< 1e-12``) is roundoff; anything larger is a broken replay."""
    surface_hits: jax.Array
    """``i32[M]`` element hits per replayed step."""


def geometry(mesh: Mesh | None):
    """The traced geometry of a mesh: the analytic body itself, the surface points or the
    level-set coefficients (``None`` without a mesh: the box-only replay has no geometry
    control)."""
    if mesh is None:
        return None
    if mesh.kind == "surface":
        return mesh.body.points
    if mesh.kind == "levelset":
        return mesh.body.coef
    return mesh.body


def with_geometry(mesh: Mesh, geom) -> Mesh:
    """``mesh`` with its body and metric tables (normals, areas) taken from ``geom``
    (cell tables and cut volumes unchanged -- see the module docstring)."""
    if mesh.kind == "surface":
        body = dataclasses.replace(mesh.body, points=geom)
        return dataclasses.replace(
            mesh,
            body=body,
            normals=surface_mod.normals(geom, body.elements),
            areas=surface_mod.areas(geom, body.elements),
        )
    if mesh.kind == "levelset":
        return dataclasses.replace(mesh, body=dataclasses.replace(mesh.body, coef=geom))
    return dataclasses.replace(mesh, body=geom)


def wall_map(
    geom,
    x,
    v,
    alive,
    key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    tracker="stencil",
    max_walk=8,
):
    """The step's move as a function of the geometry: ``(x_post, v_post, alive_post,
    logp, tallies, events)`` of :func:`jax_dsmc.boundary.apply_faces` with the traced
    events (hit points, incident velocities, normals) filled."""
    mesh_g = None if mesh is None else with_geometry(mesh, geom)
    x_post, v_post, alive_post, _, logp, events, tallies = boundary.apply_faces(
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
        return_events=True,
        mesh=mesh_g,
        tracker=tracker,
        max_walk=max_walk,
        return_tallies=True,
        trace=True,
    )
    return x_post, v_post, alive_post, logp, tallies, events


def _normals_step(carry, x, grid, mesh, diffuse_faces, within):
    del carry
    return None, mesh_geometry.nearest_normal(x, grid, mesh, diffuse_faces, within)


@functools.partial(jax.jit, static_argnames=("faces", "within"))
def local_normals(records, grid, mesh, faces, within=1.0):
    """``f64[M, N, 3]``: the nearest surface normal (elements and DIFFUSE box faces within
    ``within`` cell widths, else zero) at every slot's post-move position of every captured
    step -- the ``normals`` argument that makes the backward's fits one-sided."""
    diffuse = tuple(f for f in range(2 * len(grid.n)) if faces.bcs[f] is boundary.BC.DIFFUSE)
    body = functools.partial(
        _normals_step, grid=grid, mesh=mesh, diffuse_faces=diffuse, within=within
    )
    _, out = lax.scan(body, None, records.x)
    return out


def _zero_cotangent(primal):
    """Zeros of the right dtype for every leaf (``float0`` for integer / boolean leaves)."""
    return jax.tree.map(
        lambda p: (
            jnp.zeros(p.shape, jax.dtypes.float0)
            if not jnp.issubdtype(p.dtype, jnp.inexact)
            else jnp.zeros_like(p)
        ),
        primal,
    )


def _hit_rows(events, mesh, dim):
    """``WallParams`` row of every recorded hit (``K, N``): the face id for box faces,
    ``2D + model`` for elements; 0 where there is no hit."""
    face = events.face
    e = jnp.clip(face - 2 * dim, 0, mesh.n_elements - 1)
    model = mesh.body.model[e] if mesh.kind == "surface" else jnp.zeros_like(e)
    return jnp.where(face >= 2 * dim, 2 * dim + model, jnp.maximum(face, 0))


def _log_density_rows(v_out, normal, temp, vwall, floor, mass):
    return jnp.sum(log_diffuse_density(v_out, normal, temp, mass, vwall, floor))


def normal_score(events, weight, wall, mesh, mass, dim):
    """Cotangent on ``events.normal`` (``K, N, 3``): ``weight[k, i] * grad_n log q(v_out | n)``
    at every diffuse-branch ELEMENT hit (box-face normals carry no geometry).  The body's
    ``grazing`` cut-off ``eps`` (level set / analytic body; ``0`` for a polygon) floors the
    score's ``1 / (v_out . n)`` at ``eps |v_out|`` -- the emission-side twin of the
    hit-time regularisation (:func:`jax_dsmc.mesh.levelset.regularised_quotient`)."""
    rows = _hit_rows(events, mesh, dim)
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[rows]
    vwall = wall.vwall[rows]
    k, n = rows.shape
    floor = getattr(mesh.body, "grazing", 0.0) * jnp.linalg.norm(events.v_out, axis=-1)
    flat = jax.vmap(jax.grad(_log_density_rows, argnums=1), in_axes=(0, 0, 0, 0, 0, None))
    grad_n = flat(
        events.v_out.reshape(k * n, 1, 3),
        events.normal.reshape(k * n, 1, 3),
        temp.reshape(k * n, 1),
        vwall.reshape(k * n, 1, 3),
        floor.reshape(k * n, 1),
        mass,
    ).reshape(k, n, 3)
    scored = (events.face >= 2 * dim) & events.diffuse
    return jnp.where(scored[..., None], weight[..., None] * grad_n, 0.0)


def timing_cotangent(events, jump, dt, band):
    """Cotangent on ``events.time`` (``K, N``) of the hit-timing band term.

    ``jump`` (``K, N``) is the change of the objective when the hit of pass ``k`` moves from
    the end of this step to the start of the next (the per-step reward's ``phi(v_out) -
    phi(v_in)``; a per-hit reward's ``+r`` at the window's last step and ``-r`` on the step
    before the window; zero elsewhere).  Hits within ``band * dt`` of the step end estimate
    the density of hit times at the boundary: moving the surface so a hit crosses it
    loses ``jump``, hence ``-jump / (band dt)`` per band hit on that hit's time.
    """
    inside = (events.face >= 0) & (events.time > dt * (1.0 - band))
    return jnp.where(inside, -jump / (band * dt), 0.0)


def _last_normal(events):
    """Normal of each slot's last recorded hit of the step (``f64[N, 3]``; zero rows for
    slots without one)."""
    hit = (events.face >= 0)[..., None]
    out = jnp.zeros(events.normal.shape[1:])
    for k in range(events.face.shape[0]):
        out = jnp.where(hit[k], events.normal[k], out)
    return out


def _to_go(lam, rewards):
    """``(K + 1, N)``: row ``j`` is ``lam`` plus the rewards of the hits of passes ``j, j+1,
    ...`` -- the reward-to-go from pass ``j``'s draw onward (the hit's own reward is a
    function of it); row 0 is the value at the pre-move state, row ``K`` the post-move
    value."""
    suffix = jnp.cumsum(rewards[::-1], axis=0)[::-1]
    return jnp.concatenate((lam[None] + suffix, lam[None]))


def _replay_step(carry, xs, geom, reward, timing_jump, timing_band, setup):
    """Forward-only traced replay of one step: hit rewards, timing rewards, replay check."""
    x, v, alive, err = carry
    rec_k, step = xs
    key = key_for(setup["base_key"], step, STAGE_MOVE)
    x_post, v_post, _alive_post, _logp, tallies, events = wall_map(
        geom, x, v, alive, key, **setup["map"]
    )
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    err = jnp.maximum(err, _mismatch(x_post, v_post, rec_k, survivors, setup["map"]["grid"]))
    r_k = (
        jnp.zeros(events.face.shape)
        if reward is None
        else jnp.where(survivors[None, :], reward(events, tallies, rec_k, step), 0.0)
    )
    tau_k = jnp.zeros(events.face.shape)
    if timing_jump is not None:
        jump = jnp.where(survivors[None, :], timing_jump(events, rec_k, step), 0.0)
        tau_k = timing_cotangent(events, jump, setup["map"]["dt"], timing_band)
    n_hits = jnp.sum((events.face >= 2 * x.shape[1]) & survivors[None, :], dtype=jnp.int32)
    return (rec_k.x, rec_k.v_post, rec_k.alive, err), (r_k, tallies, n_hits, tau_k)


def _mismatch(x_post, v_post, rec_k, survivors, grid):
    """Largest relative replay error over survivors: positions per box extent, velocities
    per the record's largest speed component."""
    dx = jnp.max(
        jnp.where(survivors[:, None], jnp.abs(x_post - rec_k.x) / (grid.hi - grid.lo), 0.0)
    )
    v_scale = jnp.maximum(jnp.max(jnp.abs(jnp.where(survivors[:, None], rec_k.v, 0.0))), 1e-300)
    dv = jnp.max(jnp.where(survivors[:, None], jnp.abs(v_post - rec_k.v), 0.0)) / v_scale
    return jnp.maximum(dx, dv)


def along_flight(gx, v, lam_timing):
    """Replace the component of the fitted ``grad_x lambda`` along the flight by the
    along-flight identity: a particle at ``x - s v`` is the particle at ``x`` delayed by
    ``s``, so ``v . grad_x V = -T`` with ``T`` the expected change of the objective per
    unit delay of the trajectory (``lam_timing``: the value of the timing rewards, carried
    by a second backward).  ``gx`` (``N, D``), ``v`` (``N, 3``): the perpendicular part
    stays fitted."""
    d = gx.shape[1]
    vd = v[:, :d]
    v2 = jnp.maximum(jnp.sum(vd * vd, axis=1), 1e-300)
    parallel = jnp.sum(gx * vd, axis=1) / v2
    return gx - parallel[:, None] * vd - (lam_timing / v2)[:, None] * vd


def _virtual_hit(x_star, n_star, v, key, wall, mass, dim):
    """The re-emitted velocity of a virtual hit at ``x_star`` with normal ``n_star``
    (``N, D``) under the body's wall model (row ``2D``): specular, or diffuse with the
    accommodation probability, from fresh draws.  ``f64[N, 3]``."""
    n = x_star.shape[0]
    normal = jnp.zeros((n, 3)).at[:, :dim].set(n_star)
    vn = jnp.sum(v * normal, axis=1, keepdims=True)
    v_spec = v - 2.0 * vn * normal
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[2 * dim]
    acc = wall.acc[2 * dim]
    vwall = wall.vwall[2 * dim]
    k_draw, k_acc = jax.random.split(key)
    z, r = sampling.standard_diffuse_draws(k_draw, n)
    u = jax.random.uniform(k_acc, (n,), dtype=jnp.float64)
    fresh = sampling.diffuse_velocity_general(
        z, r, normal, jnp.full((n, 1), temp), mass, jnp.broadcast_to(vwall, (n, 3)), v
    )
    return jnp.where((u <= acc)[:, None], fresh, v_spec)


def silhouette_cotangent(
    ls, x, v, survivors, events, to_go, rec_k, lam_k, step, key, setup, fit, band, reward
):
    """The silhouette band term of one step as a cotangent ``c_i`` on ``phi`` at the
    closest-approach point ``x*_i`` of every survivor's *undeflected* flight, and the
    points: ``(c f64[N], x_star f64[N, D])``.

    A particle whose undeflected flight passes within ``band`` of the surface
    (``|phi_min| < band``; ``phi_min < 0`` = it hit on its first pass with that
    penetration depth, ``> 0`` = it missed by that distance) changes outcome when the
    surface moves by more than ``phi_min`` there: a miss becomes a hit, a hit a miss.
    ``phi_min`` is taken along the undeflected *line* within the search reach (a hit's
    fragility is its whole chord's depth) and every passing is counted once: a miss in
    the step whose move contains the closest-approach point (strictly inside -- a
    particle starting there has just left the surface), a hit in the step it hits.  The
    change of the objective is ``V_hit - V_miss``, the rate ``-d phi_min / d theta`` times
    the density of ``phi_min`` at zero, which the band's particles on *both* sides
    estimate (a window of width ``2 band``; the miss side alone describes an outward
    move, the hit side alone an inward one, and the derivative is the same from either
    side -- value-preserving, like :func:`timing_cotangent`); ``d phi_min / d coef_j =
    B_j(x*)`` by the envelope theorem, so the term is the VJP of ``phi(x*)`` with
    cotangent ``c = -(V_hit - V_miss) / (2 band)``.  ``V`` of the realised outcome is the particle's own
    reward-to-go; the virtual outcome's value is the fitted adjoint at the virtual
    post-move state (:func:`ensemble_device.value_query`) plus the virtual hit's reward
    (``reward``, the objective's per-hit reward).  For a specular wall ``V_hit -> V_miss``
    at grazing incidence and the term vanishes; for a diffuse wall it is the O(1) term of
    the silhouette (a grazing hit re-emits thermally).
    """
    dim = x.shape[1]
    grid, wall, mass, dt = setup["grid"], setup["wall"], setup["mass"], setup["dt"]
    n = x.shape[0]
    frozen = jax.lax.stop_gradient(ls)
    # the closest approach of the undeflected LINE within the search reach (not just this
    # step's segment): a hit's fragility is the depth of its whole chord
    phi_min, s_min = levelset_mod.closest_approach(
        jax.lax.stop_gradient(x), jax.lax.stop_gradient(v), frozen, None
    )
    vd = v[:, :dim]
    speed = jnp.maximum(jnp.linalg.norm(vd, axis=1), 1e-300)
    vhat = vd / speed[:, None]
    length = speed * dt
    x_star = jax.lax.stop_gradient(x + s_min[:, None] * vhat)
    n_star = levelset_mod.normal(frozen, x_star)
    hit_first = events.face[0] >= 2 * dim
    hit_later = jnp.any(events.face[1:] >= 2 * dim, axis=0)
    # one passing per trajectory: a miss counts in the step whose move contains the
    # closest-approach point (strictly inside: a particle that starts at its closest
    # approach is one that just left the surface, and one still approaching at the end
    # of the step is counted when it gets there); a hit counts in the step it hits
    interior = (s_min > 1e-3 * ls.reach) & (s_min < length)
    passing = jnp.where(hit_first, s_min > 1e-3 * ls.reach, interior)
    in_band = survivors & (jnp.abs(phi_min) < band) & ~hit_later & passing
    # the virtual outcome
    t_star = s_min / speed
    v_hit = _virtual_hit(x_star, n_star, v, key, wall, mass, dim)
    x_after_hit = x_star + v_hit[:, :dim] * jnp.maximum(dt - t_star, 0.0)[:, None]
    x_miss = x + vd * dt
    x_alt = jnp.where(hit_first[:, None], x_miss, x_after_hit)
    v_alt = jnp.where(hit_first[:, None], v, v_hit)
    # the virtual state's value from the nodes that have LEFT the surface (v . n > 0):
    # a re-emitted particle has, and a particle that grazes past the tip is from then on
    # one that left the surface there -- the approaching population carries the value of
    # the hit it is about to make, which is exactly what the virtual miss does not get
    side = jnp.zeros((n, 3)).at[:, :dim].set(n_star)
    lam_alt = ensemble_device.value_query(
        rec_k.x, rec_k.v, rec_k.alive, lam_k, grid, x_alt, v_alt, direction=side, **fit
    )
    r_alt = jnp.zeros(n)
    if reward is not None:
        k_b = events.face.shape[0]
        virtual = events._replace(
            face=jnp.zeros_like(events.face).at[0].set(2 * dim),
            v_out=jnp.zeros_like(events.v_out).at[0].set(v_hit),
            diffuse=jnp.zeros_like(events.diffuse),
            x_hit=jnp.zeros_like(events.x_hit).at[0].set(x_star),
            v_in=jnp.zeros_like(events.v_in).at[0].set(v),
            normal=jnp.zeros_like(events.normal).at[0].set(side),
            time=jnp.zeros_like(events.time).at[0].set(t_star),
        )
        del k_b
        r_alt = reward(virtual, None, rec_k, step)[0]
    v_actual = to_go[0]
    v_virtual = lam_alt + jnp.where(hit_first, 0.0, r_alt)
    diff = jnp.where(hit_first, v_actual - v_virtual, v_virtual - v_actual)  # V_hit - V_miss
    c = jnp.where(in_band, -diff / (2.0 * band), 0.0)
    return c, x_star


def _phi_at(coef, ls, x_star):
    return levelset_mod.value(dataclasses.replace(ls, coef=coef), x_star)


def _gradient_step(  # noqa: PLR0915 -- one step's cotangents, kept in one place
    carry, xs, geom, setup, fit, tally_cotangent, timing_jump, timing_band, silhouette_band, reward
):
    """One step of the shape VJP (module-level ``scan`` body)."""
    x, v, alive, g_x, g_v, g_tally, g_score, g_time, g_sil, err = carry
    rec_k, lam_k, step, r_k, tally_weight, lam_timing_k, klass_k = xs
    key = key_for(setup["base_key"], step, STAGE_MOVE)
    f = functools.partial(wall_map, x=x, v=v, alive=alive, key=key, **setup["map"])
    primal, vjp = jax.vjp(f, geom)
    x_post, v_post, _alive_post, _logp, _tallies, events = primal
    dim = x.shape[1]
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    err = jnp.maximum(err, _mismatch(x_post, v_post, rec_k, survivors, setup["map"]["grid"]))
    # value gradients at the post-move state (the record's), on survivors only, from the
    # one-sided fit over the nodes on the hit particle's side of its last surface, of the
    # particle's own class (a body hit ahead or not)
    gx, gv = node_gradients(
        rec_k.x,
        rec_k.v,
        rec_k.alive,
        lam_k,
        setup["map"]["grid"],
        direction=_last_normal(events),
        klass=klass_k,
        **fit,
    )
    if lam_timing_k is not None:
        gx = along_flight(gx, rec_k.v, lam_timing_k)
    gx = jnp.where(survivors[:, None], gx, 0.0)
    gv = jnp.where(survivors[:, None], gv, 0.0)
    to_go = _to_go(jnp.where(survivors, lam_k, 0.0), jnp.where(survivors[None], r_k, 0.0))
    # --- pathwise: post-move state and the objective's tally weights ---
    zeros = _zero_cotangent(primal)
    cot_tally = zeros[4]
    if tally_cotangent is not None:
        cot_tally = cot_tally._replace(
            momentum=tally_weight * tally_cotangent.momentum,
            energy=tally_weight * tally_cotangent.energy,
            heat=tally_weight * tally_cotangent.heat,
        )
    (d_x,) = vjp((gx, zeros[1], zeros[2], zeros[3], zeros[4], zeros[5]))
    (d_v,) = vjp((zeros[0], gv, zeros[2], zeros[3], zeros[4], zeros[5]))
    if tally_cotangent is not None:
        (d_tally,) = vjp((zeros[0], zeros[1], zeros[2], zeros[3], cot_tally, zeros[5]))
        g_tally = jax.tree.map(jnp.add, g_tally, d_tally)
    # --- score: the move's log-probability and the diffuse normals ---
    # with a tally cotangent the hit's own reward is pathwise through the traced sample
    # (`wall_adjoint._element_events`), so the normal score carries the value-to-go
    # AFTER the hit; without one the reward's whole dependence is in the score
    score_weight = to_go[:-1] if tally_cotangent is None else to_go[1:]
    cot_events = zeros[5]._replace(
        normal=normal_score(
            events,
            score_weight,
            setup["map"]["wall"],
            setup["map"]["mesh"],
            setup["map"]["mass"],
            dim,
        )
    )
    (d_score,) = vjp((zeros[0], zeros[1], zeros[2], to_go[0], zeros[4], cot_events))
    g_x = jax.tree.map(jnp.add, g_x, d_x)
    g_v = jax.tree.map(jnp.add, g_v, d_v)
    g_score = jax.tree.map(jnp.add, g_score, d_score)
    if timing_jump is not None:
        jump = jnp.where(survivors[None, :], timing_jump(events, rec_k, step), 0.0)
        cot_time = zeros[5]._replace(
            time=timing_cotangent(events, jump, setup["map"]["dt"], timing_band)
        )
        (d_time,) = vjp((zeros[0], zeros[1], zeros[2], zeros[3], zeros[4], cot_time))
        g_time = jax.tree.map(jnp.add, g_time, d_time)
    if silhouette_band > 0.0:
        mesh_g = with_geometry(setup["map"]["mesh"], geom)
        c_sil, x_star = silhouette_cotangent(
            mesh_g.body,
            x,
            v,
            survivors,
            events,
            to_go,
            rec_k,
            lam_k,
            step,
            jax.random.fold_in(key, 7919),
            setup["map"],
            fit,
            silhouette_band,
            reward,
        )
        _phi, vjp_phi = jax.vjp(functools.partial(_phi_at, ls=mesh_g.body, x_star=x_star), geom)
        (d_sil,) = vjp_phi(c_sil)
        g_sil = jax.tree.map(jnp.add, g_sil, d_sil)
    n_hits = jnp.sum((events.face >= 2 * dim) & survivors[None, :], dtype=jnp.int32)
    return (
        rec_k.x,
        rec_k.v_post,
        rec_k.alive,
        g_x,
        g_v,
        g_tally,
        g_score,
        g_time,
        g_sil,
        err,
    ), n_hits


def _setup(base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static):
    return {
        "base_key": base_key,
        "map": {
            "grid": grid,
            "faces": faces,
            "wall": wall,
            "mass": mass,
            "dt": dt,
            "mesh": mesh,
            "epsilon": epsilon,
            "acc_delta": acc_delta,
            "max_bounces": static.K_bounce,
            "tracker": static.tracker,
            "max_walk": static.K_walk,
        },
    }


@functools.partial(
    jax.jit, static_argnames=("faces", "static", "reward", "timing_jump", "timing_band")
)
def replay(
    records,
    steps,
    state0,
    base_key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon,
    acc_delta,
    static,
    reward=None,
    timing_jump=None,
    timing_band=0.2,
):
    """Traced replay of the captured steps: a :class:`Replay`.

    ``records`` is the stacked :class:`EventRecord`, ``steps`` ``i32[M]`` their absolute
    step indices (the key contract), ``state0`` the pre-move state of the first captured
    step.  ``reward(events, tallies, rec_k, step) -> f64[K, N]`` (module-level, constants
    bound with ``partial``) is the per-hit reward of the objective, already divided by
    its normalization; a survivor's rewards are kept, a newborn's are the source's.
    ``timing_jump`` (as in :func:`gradient`) adds the per-hit timing rewards.
    """
    setup = _setup(base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static)
    body = functools.partial(
        _replay_step,
        geom=geometry(mesh),
        reward=reward,
        timing_jump=timing_jump,
        timing_band=timing_band,
        setup=setup,
    )
    init = (state0.x, state0.v, state0.alive, jnp.float64(0.0))
    (_, _, _, err), (r, tallies, hits, tau) = lax.scan(body, init, (records, steps))
    return Replay(r, tallies, err, hits, tau)


@functools.partial(
    jax.jit,
    static_argnames=(
        "faces",
        "static",
        "npc_max",
        "k",
        "basis",
        "chunk",
        "timing_jump",
        "pool",
        "silhouette_band",
        "reward",
        "value_class",
    ),
)
def gradient(
    records,
    lam,
    steps,
    state0,
    base_key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon,
    acc_delta,
    static,
    scale,
    npc_max,
    hit_reward=None,
    tally_cotangent=None,
    tally_weight=None,
    k=32,
    basis="quadratic_full",
    chunk=4096,
    timing_jump=None,
    timing_band=0.2,
    pool=0,
    lam_timing=None,
    silhouette_band=0.0,
    reward=None,
    value_class="hit_ahead",
):
    """The shape gradient of the captured window: :class:`ShapeTerms` shaped like
    :func:`geometry` ``(mesh)``.  ``silhouette_band > 0`` (a length, level-set bodies
    only) switches on :func:`silhouette_cotangent` with ``reward`` the objective's per-hit
    reward function (as :func:`replay`) for the virtual hits.

    ``lam`` ``f64[M, N]`` is the ensemble backward's adjoint at the post-move states
    (``Backward.lam``), ``hit_reward`` the array the backward was given (``None`` = no hit
    rewards), ``tally_cotangent`` a :class:`SurfaceTallies`-shaped ``dJ/d tally`` (its
    ``hits`` ignored) applied with the per-step factor ``tally_weight`` ``f64[M]`` (the
    window indicator), ``scale``/``npc_max``/``k``/``basis``/``chunk`` the fit settings
    of the backward.  ``timing_jump(events, rec_k, step) -> f64[K, N]`` (module-level)
    switches on the hit-timing band term of width ``timing_band`` (a fraction of ``dt``),
    see :func:`timing_cotangent`.  ``pool`` widens the fit's node set to the ``(2 pool +
    1)^D`` block of cells; ``lam_timing`` ``f64[M, N]`` (the value of the timing rewards of
    :func:`replay`, from a second backward) switches the along-flight component of
    ``grad_x lambda`` to the identity of :func:`along_flight`.

    ``value_class`` stratifies the ``grad lambda`` fits (:func:`node_gradients`):
    ``"hit_ahead"`` (default) fits a particle's value from the particles of its own class,
    class = a body hit at a later captured step; ``"none"`` is the mixed pool of the
    phase-D artifacts.
    """
    if value_class not in ("hit_ahead", "none"):
        raise ValueError(value_class)
    setup = _setup(base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static)
    geom = geometry(mesh)
    fit = {"npc_max": npc_max, "scale": scale, "k": k, "basis": basis, "chunk": chunk, "pool": pool}
    body = functools.partial(
        _gradient_step,
        geom=geom,
        setup=setup,
        fit=fit,
        tally_cotangent=tally_cotangent,
        timing_jump=timing_jump,
        timing_band=timing_band,
        silhouette_band=silhouette_band if mesh is not None and mesh.kind == "levelset" else 0.0,
        reward=reward,
    )
    m = lam.shape[0]
    if hit_reward is None:
        hit_reward = jnp.zeros((m, records.walls.face.shape[1], lam.shape[1]))
    if tally_weight is None:
        tally_weight = jnp.ones(m)
    klass = hit_ahead(records, mesh) if value_class == "hit_ahead" else None
    zero_g = jax.tree.map(jnp.zeros_like, geom)
    init = (state0.x, state0.v, state0.alive, *([zero_g] * 6), jnp.float64(0.0))
    (_, _, _, g_x, g_v, g_tally, g_score, g_time, g_sil, err), hits = lax.scan(
        body, init, (records, lam, steps, hit_reward, tally_weight, lam_timing, klass)
    )
    total = jax.tree.map(lambda a, b, c: a + b + c, g_x, g_v, g_tally)
    return ShapeTerms(total, g_x, g_v, g_tally, g_score, g_time, g_sil, err, hits)


def hit_ahead(records, mesh):
    """``bool[M, N]``: does the slot hit the body (a surface id ``>= 2D``) at a captured
    step *after* this one -- the value class of the stratified fit (the reward of that
    hit is in the slot's value; a slot without one carries only what collisions and the
    window edges give it)."""
    dim = records.x.shape[-1]
    del mesh
    hit = jnp.any(records.walls.face >= 2 * dim, axis=1)  # (M, N)
    later = jnp.flip(jnp.cumsum(jnp.flip(hit, 0), axis=0), 0) - hit.astype(jnp.int32)
    return later > 0


VARIANTS = {
    # name: (pool, k, along-flight) -- the estimators of grad_x lambda compared in review
    # section 7.6 item 5; every one of them is one-sided (the last hit's normal)
    "one": (0, 32, False),
    "one_pool": (1, 96, False),
    "along": (0, 32, True),
    "along_pool": (1, 96, True),
}


def gradient_variants(lam_timing, names=None, **kwargs):
    """:func:`gradient` for every entry of :data:`VARIANTS` (the same records, cotangents
    and rewards; only the estimate of ``grad_x lambda`` differs): ``{name: ShapeTerms}``.
    ``lam_timing`` is the timing-reward value of the along-flight variants; ``names``
    restricts the set (an optimiser needs one replay per step, not four)."""
    out = {}
    for name, (pool, k, along) in VARIANTS.items():
        if names is not None and name not in names:
            continue
        out[name] = gradient(**kwargs, pool=pool, k=k, lam_timing=lam_timing if along else None)
    return out
