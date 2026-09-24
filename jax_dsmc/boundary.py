"""Boundary conditions (spec section 9.1-9.3).

Forward-only in the sense that matters: the periodic and specular adjoints are
*free* -- they are the VJPs of a translation and a mirror, and reverse-mode AD
produces [P3]'s ``C = diag([-1, 1, 1])`` with no special handling.  Diffuse and
thermal walls are section 9.4 and are deferred to M4 (forward) / M7 (adjoint),
because the *event* of hitting a wall is a discontinuous function of position.

Only the first ``D`` components of ``v`` participate: velocities are always
3-component whatever the spatial dimension (section 0.2).
"""

from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import sampling
from jax_dsmc.grid import cell_of
from jax_dsmc.safe import safe_div, safe_log
from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


class BC(enum.Enum):
    """Which boundary rule to apply.  STATIC -- it selects a code path."""

    PERIODIC = "periodic"
    SPECULAR = "specular"
    OUTFLOW = "outflow"
    DIFFUSE = "diffuse"
    """SPARTA-style thermal wall: absorb and re-emit from the wall half-Maxwellian.

    Discontinuous in position -- the *event* of hitting the wall is a step function --
    so it is forward-only until M7 randomises the timestep (section 9.4).
    """
    INFLOW = "inflow"
    """Particles enter through this face from a reservoir; see :mod:`jax_dsmc.inflow`."""


@dataclasses.dataclass(frozen=True)
class Faces:
    """Which rule applies on each of the ``2 D`` faces.  **STATIC**.

    Ordered ``(lo_0, hi_0, lo_1, hi_1, ...)``, so face index ``2 * axis + side`` with
    ``side == 0`` for the low face.  Static because each entry selects a code path;
    the *parameters* of a wall (temperature, velocity, accommodation) are traced and
    live in :class:`WallParams`.

    A single global ``BC`` was enough through M3, but Couette needs thermal walls on
    one axis with periodic on another, and the inflow/outflow shock needs a different
    rule at each end of the same axis.
    """

    bcs: tuple[BC, ...]

    def __post_init__(self):
        if len(self.bcs) % 2 or not self.bcs:
            raise ValueError(f"need 2*D face rules, got {len(self.bcs)}")

    @classmethod
    def uniform(cls, bc: BC, dim: int) -> Faces:
        """The pre-M4 behaviour: one rule everywhere."""
        return cls((bc,) * (2 * dim))

    @property
    def dim(self) -> int:
        return len(self.bcs) // 2

    def of(self, axis: int, side: int) -> BC:
        return self.bcs[2 * axis + side]

    def uniform_bc(self) -> BC | None:
        """The single rule if every face agrees, else ``None`` -- picks the fast path."""
        first = self.bcs[0]
        return first if all(b is first for b in self.bcs) else None


def apply_periodic(x_new, v, grid: GridSpec):
    """``domain.cpp:301-330``.  Adjoint: the identity, matching [P3] section 3.1.

    ``mod`` is differentiable a.e. with derivative 1, which is correct -- the wrap
    is a translation, so it must not change a cotangent.
    """
    L = grid.hi - grid.lo
    return grid.lo + jnp.mod(x_new - grid.lo, L), v


def apply_specular(x_new, v, grid: GridSpec):
    """Closed-form multi-bounce fold (section 9.2).

    ``domain.cpp:337-357`` mirrors once and assumes at most one crossing; [P3]
    section 3.2 makes the same assumption explicitly.  A triangle-wave fold handles
    **any** number of bounces in one branch-free expression, at no extra cost.

    Adjoint: ``dx/dx_new == dv/dv == sgn``, i.e. [P3] eq. ``C_ki`` applied to both
    the position and velocity adjoints -- which AD produces from the code below.
    """
    D = x_new.shape[-1]
    L = grid.hi - grid.lo
    y = (x_new - grid.lo) / L
    m = jnp.floor(y)  # fold index, FROZEN
    frac = y - m
    even = jnp.mod(m, 2.0) == 0.0
    pos = jnp.where(even, frac, 1.0 - frac)
    sgn = jnp.where(even, 1.0, -1.0)
    x = grid.lo + pos * L
    v = v.at[:, :D].multiply(sgn)
    return x, v


def apply_outflow(x_new, v, alive, grid: GridSpec):
    """``domain.cpp:294-295``.  Dead particles keep their slots (static shapes).

    Through M4 ``alive`` is monotone non-increasing, so a dead slot is never
    touched again.  **From M5 that changes**: inflow refills dead slots, and any
    path that sets ``alive[i] = True`` for a previously dead ``i`` must zero
    ``logh_acc[i]`` in the same update, or the new particle inherits the score
    history of whatever occupied that index (section 9.3, gap U6).
    """
    inside = jnp.all((x_new >= grid.lo) & (x_new <= grid.hi), axis=-1)
    return x_new, v, alive & inside


def fold_index(x_new, grid: GridSpec, bc: BC):
    """The **frozen integer** that says which side of a boundary discontinuity we sit on.

    ``i32[N, D]``.  For :data:`BC.SPECULAR` this is exactly the ``m`` of
    :func:`apply_specular` -- and its *parity* flips the sign of ``v``, so a particle
    that changes fold index between two evaluations has had its velocity negated.
    That is a discontinuity of the map every bit as sharp as an accept-bit flip, and
    it is **invisible in the cell index**: a particle just inside the wall and one
    just outside both fold into cell 0.

    This function exists because that hole was found the hard way.  A
    finite-difference check whose flip guard covered cells, attempt counts, pairs
    and accept bits but *not* this reported "no flips" while FD and AD disagreed by
    a factor of three (``370.5`` vs ``127.2``).

    For :data:`BC.PERIODIC` the same quantity is the wrap count.  A wrap is a
    translation on the torus and does not break continuity, but it is included
    anyway: it costs one ``floor`` and it can only make the guard stricter.
    """
    if bc is BC.OUTFLOW:
        inside = jnp.all((x_new >= grid.lo) & (x_new <= grid.hi), axis=-1)
        return jnp.broadcast_to(inside[:, None], x_new.shape).astype(jnp.int32)
    return jnp.floor((x_new - grid.lo) / (grid.hi - grid.lo)).astype(jnp.int32)


def apply(x_new, v, alive, grid: GridSpec, bc: BC):
    """Dispatch on ``bc``.  ``bc`` is STATIC -- a Python enum, never traced."""
    if bc is BC.PERIODIC:
        x, v = apply_periodic(x_new, v, grid)
        return x, v, alive
    if bc is BC.SPECULAR:
        x, v = apply_specular(x_new, v, grid)
        return x, v, alive
    if bc is BC.OUTFLOW:
        return apply_outflow(x_new, v, alive, grid)
    raise ValueError(f"unknown boundary condition: {bc!r}")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WallParams:
    """**Traced** per-face wall parameters.  All ``2 D``-long, indexed like :class:`Faces`.

    Separate from :class:`Faces` because these are exactly the leaves a gradient
    should reach -- ``dJ/dT_wall`` and ``dJ/dv_wall`` are the Couette design
    sensitivities -- whereas the rule itself selects a code path and must stay static.

    Fields:
        temp:  ``f64[2D]`` isotropic or ``f64[2D,3]`` component wall temperatures.
        vwall: ``f64[2D, 3]`` wall velocity; must be tangential to its own face.
        acc:   ``f64[2D]`` accommodation coefficient.  ``1`` is fully diffuse,
               ``0`` is specular (``surf_collide_diffuse.cpp:211``).
    """

    temp: jax.Array
    vwall: jax.Array
    acc: jax.Array

    @classmethod
    def uniform(cls, dim: int, temp: float, acc: float = 1.0, n_models: int = 0) -> WallParams:
        """Every face the same, at rest.  ``n_models`` extra rows for embedded surfaces.

        Rows ``0 .. 2D - 1`` are the box faces; row ``2D + m`` is surface model ``m`` --
        the ``diffuse T acc`` law of every element with ``Surface.model == m`` (an
        analytic body is model 0).  ``acc = 0`` makes a model specular, exactly as
        SPARTA's ``surf_collide ... diffuse T 0.0``.
        """
        n = 2 * dim + n_models
        return cls(
            temp=jnp.full((n,), temp, dtype=jnp.float64),
            vwall=jnp.zeros((n, 3), dtype=jnp.float64),
            acc=jnp.full((n,), acc, dtype=jnp.float64),
        )

    def with_model(self, dim: int, model: int, temp=None, acc=None, vwall=None) -> WallParams:
        """Set the ``diffuse T acc`` law (and tangential drift) of surface model ``model``."""
        row = 2 * dim + model
        if row >= self.acc.shape[0]:
            raise ValueError(f"WallParams has no row for surface model {model} (D = {dim})")
        out = self
        if temp is not None:
            out = dataclasses.replace(out, temp=out.temp.at[row].set(temp))
        if acc is not None:
            out = dataclasses.replace(out, acc=out.acc.at[row].set(acc))
        if vwall is not None:
            vwall = jnp.asarray(vwall, dtype=jnp.float64)
            out = dataclasses.replace(out, vwall=out.vwall.at[row].set(vwall))
        return out

    def with_wall_velocity(self, axis: int, side: int, vwall) -> WallParams:
        """Set one face sliding.  Checks the tangency restriction eagerly."""
        vwall = jnp.asarray(vwall, dtype=jnp.float64)
        sampling.assert_tangential_wall(vwall, axis)
        return dataclasses.replace(self, vwall=self.vwall.at[2 * axis + side].set(vwall))

    def with_temp(self, axis: int, side: int, temp) -> WallParams:
        return dataclasses.replace(self, temp=self.temp.at[2 * axis + side].set(temp))


# --- walls that re-emit, in closed form (section 9.4) -----------------------
#
# [P3] eq. `thermal_v_L`/`thermal_v_R` gives the whole bounce as one expression:
#
#     v_{k+1} = g,     x_{k+1} = L + (x' - L)/(P v') * (P g)
#
# `(x' - L)/(P v')` is the *overshoot time* -- how long the particle spent past the
# wall -- and it is re-advected with the new normal velocity.  So a thermal wall
# needs no hit-time iteration and no sub-move scan: the mesh stays static and every
# face rule stays a closed-form function of the end-of-step position, which is what
# keeps the fast path.  The flux bias is carried by the half-Maxwellian draw itself
# (`sample_flux_normal`), not by tracking geometry.
#
# [P3] notes the form "does not include the possibility of multiple boundary
# crossings"; neither does SPARTA's specular (`domain.cpp:337-357`).  At the CFL-like
# step this code already uses, a second crossing needs `|v| dt` to exceed a box
# length.  Rather than iterate for a case that should not arise, it is DETECTED and
# reported (`recrossed`) -- a silent one would be a momentum sink.


def crossing_probabilities(x, v, grid: GridSpec, axis: int, dt, epsilon):
    """[P3] appendix ``app:log``: ``(p_lo, p_stay, p_hi)`` under ``tau ~ N(dt, eps^2)``.

    With ``t^L = (lo - x)/v`` and ``t^R = (hi - x)/v``, a particle crosses a face iff
    ``tau`` exceeds the time needed to reach it, so the crossing probability is
    ``1 - F(t)`` for ``F`` the ``N(dt, eps^2)`` CDF.  Only one side is reachable at a
    time -- the sign of ``v`` decides which -- so one of ``p_lo``, ``p_hi`` is always 0.

    These are the probabilities whose **logs** the M7 adjoint accumulates into
    ``logh_acc``: [P3] eq. ``J_final_t1_x`` shows the boundary sensitivity is
    ``(d log p_l / dx) r_i``, which is a DiCE score with reward ``r_i`` -- structurally
    the same object as [P2]'s ``eta``, and carried by the same accumulator.

    At ``epsilon == 0`` this degenerates to the deterministic indicator (0 or 1), which
    is exactly the discontinuity the regularisation exists to remove.
    """
    vd = v[:, axis]
    safe = jnp.where(jnp.abs(vd) > 0.0, vd, 1.0)
    t_lo = (grid.lo[axis] - x[:, axis]) / safe
    t_hi = (grid.hi[axis] - x[:, axis]) / safe
    eps = jnp.asarray(epsilon, dtype=jnp.float64)
    safe_eps = jnp.where(eps > 0.0, eps, 1.0)

    def _sf(t):  # P(tau > t)
        z = (t - dt) / (safe_eps * jnp.sqrt(2.0))
        smooth = 0.5 * jax.scipy.special.erfc(z)
        return jnp.where(eps > 0.0, smooth, (t < dt).astype(jnp.float64))

    p_lo = jnp.where(vd < 0.0, _sf(t_lo), 0.0)
    p_hi = jnp.where(vd > 0.0, _sf(t_hi), 0.0)
    return p_lo, 1.0 - p_lo - p_hi, p_hi


def bound_of(grid: GridSpec, axis: int, side: int):
    """The coordinate of face ``(axis, side)``; ``side == 0`` is the low face."""
    return grid.lo[axis] if side == 0 else grid.hi[axis]


def _diffuse_velocity(v, wall, mass, axis, side, key, delta=0.0):
    """Post-wall velocity, and the **score** of the accommodation draw.

    Returns ``(v_out, log_p)`` with ``log_p = log(acc)`` on the diffuse branch and
    ``log(1 - acc)`` on the specular one.

    The re-emitted velocity is reparameterised in ``temp`` and ``vwall``
    (:func:`sampling.sample_diffuse_velocity`), so *those* two parameters are pathwise
    and need no score at all -- measured exact against a fixed-key finite difference to
    3e-12.  ``acc`` is different in kind: it only enters through the **threshold** of a
    Bernoulli draw, so its pathwise derivative is zero almost everywhere and the whole
    sensitivity lives in this score.  Without it ``dJ/d(acc)`` reads exactly 0.0, which
    is a plausible-looking wrong answer rather than an obvious failure.
    """
    n = v.shape[0]
    k_new, k_acc = jax.random.split(key)
    z, r = sampling.standard_diffuse_draws(k_new, n)
    u = jax.random.uniform(k_acc, (n,), dtype=jnp.float64)
    return diffuse_from_draws(v, wall, mass, axis, side, (z, r, u), delta)


def diffuse_from_draws(v, wall, mass, axis, side, draws, delta=0.0):
    """:func:`_diffuse_velocity` on pre-drawn ``(z, r, u)`` -- see
    :func:`sampling.standard_diffuse_draws` for why a bounce pass draws once."""
    f = 2 * axis + side
    inward = 1.0 if side == 0 else -1.0
    z, r, u = draws
    fresh = sampling.diffuse_velocity_from_draws(
        z, r, wall.temp[f], mass, axis, inward, wall.vwall[f]
    )
    # `surf_collide_diffuse.cpp:211` -- `random->uniform() > acc` takes the specular branch
    took_diffuse = u <= wall.acc[f]
    v_spec = v.at[:, axis].multiply(-1.0)
    # `delta > 0` relaxes the Bernoulli threshold into a smooth blend, which makes `acc`
    # PATHWISE and removes it from the score entirely -- no attribution, no variance
    # penalty, and validatable against a fixed-key finite difference.  This is the same
    # move [P3] section 4 makes for the crossing event with `tau ~ N(dt, eps^2)`: a hard
    # indicator is replaced by a smooth dependence, trading an O(delta) physics bias for a
    # well-defined derivative.  `delta == 0` reproduces the hard threshold **exactly**, so
    # the relaxation ships switched off.
    safe_delta = jnp.where(delta > 0.0, delta, 1.0)
    w = jnp.where(
        delta > 0.0,
        jax.nn.sigmoid((wall.acc[f] - u) / safe_delta),
        took_diffuse.astype(jnp.float64),
    )
    v_out = w[:, None] * fresh + (1.0 - w)[:, None] * v_spec
    # the realised branch is data, not a function of `acc`: only the probability of it is
    log_p = jnp.where(took_diffuse, safe_log(wall.acc[f]), safe_log(1.0 - wall.acc[f]))
    return v_out, jnp.where(delta > 0.0, 0.0, log_p)


def _apply_face_rule(x, v, alive, hit, axis, side, bc, grid, wall, mass, key):
    """One face's rule, applied to the particles masked by ``hit``.  ``bc`` is STATIC."""
    b = bound_of(grid, axis, side)
    if bc is BC.PERIODIC:
        # the exact wrap, valid for any number of crossings (section 9.1)
        L = grid.hi[axis] - grid.lo[axis]
        wrapped = grid.lo[axis] + jnp.mod(x[:, axis] - grid.lo[axis], L)
        return x.at[:, axis].set(jnp.where(hit, wrapped, x[:, axis])), v, alive
    if bc is BC.SPECULAR:
        # the exact triangle-wave fold, valid for any number of bounces (section 9.2)
        L = grid.hi[axis] - grid.lo[axis]
        y = (x[:, axis] - grid.lo[axis]) / L
        m = jnp.floor(y)
        frac = y - m
        even = jnp.mod(m, 2.0) == 0.0
        folded = grid.lo[axis] + jnp.where(even, frac, 1.0 - frac) * L
        sgn = jnp.where(even, 1.0, -1.0)
        x = x.at[:, axis].set(jnp.where(hit, folded, x[:, axis]))
        return x, v.at[:, axis].multiply(jnp.where(hit, sgn, 1.0)), alive
    if bc in (BC.OUTFLOW, BC.INFLOW):
        # an outgoing particle simply leaves.  For INFLOW the *incoming* half is a
        # separate injection stage, because it changes which slots are alive.
        return x, v, alive & ~hit
    if bc is BC.DIFFUSE:
        f = 2 * axis + side
        inward = 1.0 if side == 0 else -1.0
        n = x.shape[0]
        k_new, k_acc = jax.random.split(key)
        fresh = sampling.sample_diffuse_velocity(
            k_new, n, wall.temp[f], mass, axis, inward, wall.vwall[f]
        )
        # accommodation: diffuse with probability `acc`, else specular
        # (`surf_collide_diffuse.cpp:211` -- `random->uniform() > acc` takes specular)
        u = jax.random.uniform(k_acc, (n,), dtype=jnp.float64)
        v_spec = v.at[:, axis].multiply(-1.0)
        v_out = jnp.where((u <= wall.acc[f])[:, None], fresh, v_spec)

        # [P3] eq. thermal_v_L.  `overshoot` and the old normal velocity share the
        # outward sign, so their ratio is the positive time spent beyond the wall.
        t_over = safe_div(x[:, axis] - b, v[:, axis])
        x_wall = b + t_over * v_out[:, axis]
        x = x.at[:, axis].set(jnp.where(hit, x_wall, x[:, axis]))
        return x, jnp.where(hit[:, None], v_out, v), alive
    raise ValueError(f"unknown boundary condition: {bc!r}")


def diffuse_hit_time(x, v, grid: GridSpec, faces: Faces, budget):
    """Time of the **first** diffuse-face crossing, ``f64[N]``, and which face, ``i32[N]``.

    Only diffuse faces are considered: periodic and specular keep their exact
    closed-form folds, which are correct for any number of crossings, whereas a wall
    must know *which* face was reached first.

    Resolving that is what fixes the corner.  Applying each face's rule in turn gives
    a particle that leaves through two faces in one step **two** independent
    re-emission draws, and since each draw is flux-weighted (biased toward fast
    particles) the second one pumps in energy: a cold gas in a cubic box settled at
    350 K (D=2) and 386 K (D=3) against a 300 K wall before this was resolved.
    """
    D = x.shape[-1]
    big = jnp.inf
    times = []
    for axis in range(D):
        vd = v[:, axis]
        safe = jnp.where(jnp.abs(vd) > 0.0, vd, 1.0)
        for side in (0, 1):
            if faces.of(axis, side) is not BC.DIFFUSE:
                times.append(jnp.full(x.shape[0], big))
                continue
            t = (bound_of(grid, axis, side) - x[:, axis]) / safe
            approaching = (vd < 0.0) if side == 0 else (vd > 0.0)
            times.append(jnp.where(approaching & (t > 0.0) & (t <= budget), t, big))
    stacked = jnp.stack(times, axis=-1)
    return jnp.min(stacked, axis=-1), jnp.argmin(stacked, axis=-1)


def diffuse_axes(faces: Faces) -> tuple[int, ...]:
    """Which axes carry a thermal face.  STATIC -- it gates the second bounce pass."""
    return tuple(
        axis
        for axis in range(faces.dim)
        if any(faces.of(axis, side) is BC.DIFFUSE for side in (0, 1))
    )


def n_bounce_passes(faces: Faces) -> int:
    """How many thermal bounces one step can need.

    A particle can only reach two thermal faces in a step if they lie on **different
    axes** -- opposite faces of the same axis cannot both be hit, since after the
    first the particle is travelling away from the second.  So:

    * thermal on one axis only (Couette: walls on ``y``, periodic on ``x``) -> **1**;
    * thermal on two or more axes (a box thermal on every face) -> **2**.

    Gated statically, so a configuration that cannot corner does not pay for the
    second pass at all.  Two is enough for the same reason one is enough per axis:
    reaching a third thermal face needs three crossings in a step, which is the
    same-face multi-crossing regime that section 9.4 measures at zero.
    """
    return 2 if len(diffuse_axes(faces)) >= 2 else 1


def _bounce_once(x, v, alive, t_used, logh, grid, faces, wall, mass, tau, key, acc_delta=0.0):
    """Resolve the first thermal face reached in the time still remaining.

    Leaves a bounced particle sitting **at the wall** with its new velocity and
    ``t_used`` advanced to the hit time; a particle that reaches no wall is returned
    untouched.  One final advance by ``dt - t_used`` then completes the step for
    everybody, which is what makes the passes composable.
    """
    D = x.shape[-1]
    keys = jax.random.split(key, 2 * D)
    t_hit, f_hit = diffuse_hit_time(x, v, grid, faces, tau - t_used)
    crosses = jnp.isfinite(t_hit) & alive
    t_safe = jnp.where(crosses, t_hit, 0.0)
    x_wall = x + v[:, :D] * t_safe[:, None]
    for axis in range(D):
        for side in (0, 1):
            if faces.of(axis, side) is not BC.DIFFUSE:
                continue
            f = 2 * axis + side
            hit = crosses & (f_hit == f)
            v_out, log_p = _diffuse_velocity(v, wall, mass, axis, side, keys[f], acc_delta)
            x = jnp.where(hit[:, None], x_wall, x)
            v = jnp.where(hit[:, None], v_out, v)
            # only a particle that actually reached this face made the draw
            logh = logh + jnp.where(hit, log_p, 0.0)
    return x, v, t_used + t_safe, logh


def crossing_score(x_old, v, alive, grid: GridSpec, faces: Faces, dt, epsilon):
    """``sum_axis log p`` of the realised crossing branch -- the M7 boundary score.

    [P3] eq. ``J_final_t1_x``: the boundary sensitivity is ``(d log p_l / dx) r_i``, a
    DiCE score with reward ``r_i``.  Which branch the particle took is *data* (it
    followed from the realised ``tau``); only the **probability** of that branch is a
    function of the state, and differentiating it supplies the term the pathwise
    derivative cannot see -- the one that accounts for a decision *flipping* as a
    parameter moves.

    Returns exactly ``0.0`` at ``epsilon == 0``, and not by a special case: the branch
    that actually happened has probability 1 there, and ``log 1 == 0``.  That is the
    deterministic limit doing the right thing on its own, and it is why switching the
    regularisation off leaves every M1-M5 gradient bit-identical.

    Only axes carrying a wall or an open face are scored.  A **periodic** axis has no
    decision to score: the fold is a continuous, exactly-invertible map applied whatever
    ``tau`` turns out to be, so there is no branch whose probability could shift.
    """
    D = x_old.shape[-1]
    logh = jnp.zeros(x_old.shape[0], dtype=jnp.float64)
    for axis in range(D):
        if all(faces.of(axis, side) is BC.PERIODIC for side in (0, 1)):
            continue
        p_lo, p_stay, p_hi = crossing_probabilities(x_old, v, grid, axis, dt, epsilon)
        # the realised branch, read off the *actual* post-move position
        x_axis = x_old[:, axis] + v[:, axis] * dt
        went_lo = x_axis < grid.lo[axis]
        went_hi = x_axis > grid.hi[axis]
        p = jnp.where(went_lo, p_lo, jnp.where(went_hi, p_hi, p_stay))
        logh = logh + jnp.where(alive, safe_log(p), 0.0)
    return logh


def spread_score_over_cell(logh, x, alive, grid: GridSpec):
    """Re-attribute each particle's boundary score to **every particle in its cell**.

    The DiCE surrogate pairs a score with the particles whose ``L`` it appears in, so that
    attribution set has to cover everything the decision influenced.  A single particle is
    the narrowest possible choice and it measurably under-counts: a wall decision changes
    the bouncer's velocity, and the bouncer then collides, so the decision reaches its
    neighbours too.  Attributing to one particle alone measured a **2.8x** low bias
    against an ensemble finite difference, while the same run with collisions switched off
    was exact -- which is what identifies attribution, rather than the log-probability
    itself, as the cause.

    The cell is the natural wider set and is *already* what ``collide`` uses for the NTC
    attempt-count score (``ntc.py:368`` adds ``logP[cell]`` to every particle in the cell).
    That is also why the shipped collision score is only ~6 % biased where this one was
    180 % -- its attribution was already wide.  Choosing the cell here keeps the two
    consistent rather than inventing a third convention.

    Every particle in the cell receives the **full** score, not a share of it -- matching
    ``ntc.py:368`` exactly.  That is the DiCE attribution rule, not an averaging: ``L_i``
    must contain the log-probability of each decision that influenced particle ``i``, and
    a decision that influenced all of them appears in all of their ``L``.  Dividing by the
    occupancy would reintroduce the under-counting this exists to remove.
    """
    cell = cell_of(x, grid)
    n_cell = int(np.prod(np.asarray(grid.n)))
    total = jnp.zeros((n_cell,), jnp.float64).at[cell].add(jnp.where(alive, logh, 0.0))
    return jnp.where(alive, total[cell], 0.0)


def apply_faces(
    x_old,
    v,
    alive,
    grid: GridSpec,
    faces: Faces,
    wall,
    mass,
    dt,
    key,
    epsilon=0.0,
    score_spread="particle",
    acc_delta=0.0,
    max_bounces=4,
    return_events=False,
    mesh=None,
    tracker="stencil",
    max_walk=8,
    return_tallies=False,
    trace=False,
    lane_size=0,
):
    """Advance one step under per-face rules, closed form.

    Returns ``(x, v, alive, recrossed, logh)``.  ``logh`` is the summed log-probability
    of every discrete boundary decision the particle took this step -- the accommodation
    draw and, when ``epsilon > 0``, the crossing branch.  It is added to
    ``ParticleState.logh_acc`` by the caller and consumed by :func:`adjoint.surrogate`.

    Diffuse faces dispatch to chronological Cartesian random-time tracking.
    The conditional-time estimator is tested most extensively in 1D; general
    corner/grazing sensitivities remain unvalidated. Without diffuse faces the
    exact Cartesian fold path below remains available.

    Legacy stages, in this order:

    1. **The wall bounce**, for the first diffuse face reached (if any).  The particle
       is advanced to the wall, re-emitted, and the *remaining* time is advected with
       the new velocity -- ``x = x_wall + (dt - t_hit) * g``.  In one spatial
       dimension that is exactly [P3] eq. ``thermal_v_L``, since
       ``(x' - L)/(P v') == dt - t_hit``; the form here is its D-general version and
       so also carries the tangential displacement.
    2. **The exact folds** for periodic, specular and outflow faces, applied to the
       result.  These are closed forms valid for *any* number of crossings, so they
       are left alone rather than folded into the hit-time logic.

    ``recrossed`` counts particles still outside afterwards -- the diagnostic for the
    multi-crossing case, which section 9.4 measures at exactly zero out to ~7x the
    production timestep.
    """
    D = x_old.shape[-1]
    if BC.DIFFUSE in faces.bcs or mesh is not None:
        # Lazy import breaks the boundary dispatch import cycle.  An embedded surface
        # (`mesh`) always takes the chronological tracker; the fourth element is then a
        # `wall_adjoint.TrackerCounts`.
        from jax_dsmc.wall_adjoint import advance  # noqa: PLC0415

        result = advance(
            x_old,
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
            return_events=return_events,
            mesh=mesh,
            tracker=tracker,
            max_walk=max_walk,
            return_tallies=return_tallies,
            trace=trace,
            lane_size=lane_size,
        )
        if score_spread == "cell":
            result = (
                *result[:4],
                spread_score_over_cell(result[4], x_old, alive, grid),
                *result[5:],
            )
        elif score_spread != "particle":
            raise ValueError("score_spread must be particle or cell")
        return result
    if return_events or return_tallies:
        raise ValueError(
            "Wall events and surface tallies come from the chronological tracker "
            "(DIFFUSE faces or an embedded surface) only"
        )
    keys = jax.random.split(key, 2 * D + 3)
    # [P3] section 4: the regularised step.  `epsilon == 0` gives `tau == dt` exactly.
    tau = sampling.sample_tau(keys[2 * D + 2], x_old.shape[0], dt, epsilon)
    x = x_old + v[:, :D] * tau[:, None]

    # scored against the PRE-move state and the pre-bounce velocity, which is what the
    # crossing probability is conditioned on
    logh = crossing_score(x_old, v, alive, grid, faces, dt, epsilon)

    if BC.DIFFUSE in faces.bcs:
        # Start from the *pre-move* state: each pass walks the particle to its wall.
        x = x_old
        t_used = jnp.zeros(x.shape[0], dtype=jnp.float64)
        for p in range(n_bounce_passes(faces)):
            x, v, t_used, logh = _bounce_once(
                x,
                v,
                alive,
                t_used,
                logh,
                grid,
                faces,
                wall,
                mass,
                tau,
                keys[2 * D + p],
                acc_delta,
            )
        # particles that never hit a wall still owe the whole step
        x = x + v[:, :D] * (tau - t_used)[:, None]

    for axis in range(D):
        for side in (0, 1):
            bc = faces.of(axis, side)
            if bc is BC.DIFFUSE:
                continue
            b = bound_of(grid, axis, side)
            outside = ((x[:, axis] < b) if side == 0 else (x[:, axis] > b)) & alive
            x, v, alive = _apply_face_rule(
                x, v, alive, outside, axis, side, bc, grid, wall, mass, keys[2 * axis + side]
            )
    if score_spread == "cell":
        logh = spread_score_over_cell(logh, x, alive, grid)
    inside = jnp.all((x >= grid.lo) & (x <= grid.hi), axis=-1)
    recrossed = jnp.sum(alive & ~inside)
    # Safety net, not physics.  A particle that crossed two faces in one step is left
    # outside by the single-pass bounce, and `cell_of` would silently clip it into an
    # edge cell -- so clamp it explicitly and let `recrossed` say so.  Measured at
    # exactly zero out to ~7x the production timestep (section 9.4); if this ever
    # fires in anger the timestep is wrong, not the boundary rule.
    x = jnp.clip(x, grid.lo, grid.hi)
    # a slot that died this step carries no score forward: it has no reward to weight
    return x, v, alive, recrossed, jnp.where(alive, logh, 0.0)
