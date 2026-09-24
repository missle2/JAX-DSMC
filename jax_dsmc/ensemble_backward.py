"""Particle-carried ensemble adjoint of the production solver (host reference).

The optimize-then-discretize adjoint of P3 section 5, quadratured on the events the
forward actually realized (branch review section 5): the adjoint ``mu`` is carried on
every particle slot, accepted NTC pairs are the Monte Carlo sample of the collision
measure, births carry their own value, and the only interpolation is at the partners'
pre-collision "ghost" states.  Consumes the :class:`jax_dsmc.simulate.EventRecord`
stack that ``simulate.run(..., capture=True)`` returns; the forward is untouched.

Per step ``k`` (forward order: move/walls -> inject -> collide), backward:

1. collision pull-back on the post-move state ``S^m_k = (x, v)``::

       lam_i = mu_i + sum over accepted pairs (i, j) of [ Lam(x_j, v_j^post) - Lam(x_j, v_j^pre) ]

   where ``mu`` are the carried values at ``t_{k+1}`` and ``Lam`` is the field at
   ``t_{k+1}`` interpolated from the alive particles of the same cell.  ``Lam(x_j,
   v_j^post)`` is the carried ``mu_j`` exactly when that was ``j``'s last collision
   of the step (the common case); otherwise it is interpolated too.
2. sources: every slot born this step carries ``lam`` at its post-move state;
   the caller turns ``(face, sampled velocity, lam)`` into parameter gradients with
   the source's log-density score and its intensity derivative.
3. move pull-back: a slot alive at ``t_k`` that survived the move keeps ``lam``;
   one that was absorbed has ``0``.  (Specular/periodic faces are deterministic
   maps; diffuse walls -- not exercised here -- are kernels whose realized
   re-emission is the unbiased single sample.)

Three collision pull-backs share step 1 (:func:`configure` selects one for the process):
``sequential`` -- the accepted attempts in reverse order, each partner ghost read from
the field as chained so far (the exact discrete adjoint of the NTC step; its noise grows
with the collisions per particle per step); ``linearised`` -- all attempts at once from
the end-of-step field (biased O(dt)); ``virtual`` -- the substepped operator
``exp(dt L*)`` with ``K`` virtual partners per particle drawn from the cell and weighted
by the NTC acceptance, every value from one smooth per-cell fit per substep, the fit
nodes at the step's staged intermediate states (P3's ``N_col`` averaging; unbiased and
low-variance at any collision rate, review section 5.12).

Host NumPy/SciPy; :mod:`jax_dsmc.ensemble_device` is the JIT port of the sequential and
virtual forms (:func:`from_device` turns its output into the same :class:`Adjoint`).
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.collide.vss import collision_rate, scatter
from jax_dsmc.ensemble_inflow import LocalAffine


@dataclass
class Births:
    step: int
    slots: np.ndarray
    faces: np.ndarray
    velocities: np.ndarray
    """``f64[n, 3]`` velocities the source SAMPLED (the score's argument)."""
    values: np.ndarray
    """``f64[n]`` adjoint at each newborn's post-move state; escaped births are absent."""


@dataclass
class Hits:
    """One step's diffuse-wall hits: the wall kernel's score is weighted by ``values``."""

    step: int
    slots: np.ndarray
    faces: np.ndarray
    v_out: np.ndarray
    """``f64[n, 3]`` laboratory velocity after the (last) hit."""
    diffuse: np.ndarray
    """``bool[n]`` whether the accommodation draw took the diffuse branch."""
    values: np.ndarray
    """``f64[n]`` adjoint at the slot's post-move state: by the tower property the
    unbiased weight for THIS hit's score, however many hits followed it in the step."""


@dataclass
class Pairs:
    """One step's accepted pairs with their four-term adjoint difference.

    ``four_term = Lam(v_i') + Lam(v_j') - Lam(v_i) - Lam(v_j)`` at the ``t_{k+1}`` field;
    a kernel parameter's gradient is ``sum_pairs d_theta log q(g) * four_term``.
    """

    step: int
    i: np.ndarray
    j: np.ndarray
    vi: np.ndarray
    vj: np.ndarray
    four_term: np.ndarray
    weight: np.ndarray = None
    """``f64[p]`` quadrature weight of each pair (1 for realized pairs; the NTC expectation
    weight for virtual ones).  ``None`` means all ones."""


@dataclass
class Adjoint:
    lambda0: np.ndarray
    """``f64[N]`` adjoint at the initial state, on the initially alive slots (else 0)."""
    births: list
    normalization: float
    hits: list = None
    pairs: list = None


def _cells(x, grid):
    """Host twin of :func:`jax_dsmc.grid.collision_cell` (numpy)."""
    lo, hi, n = np.asarray(grid.lo), np.asarray(grid.hi), np.asarray(grid.n)
    rel = (x - lo) / (hi - lo) * n
    idx = np.clip(np.floor(rel).astype(int), 0, n - 1)
    cell = np.ravel_multi_index(tuple(idx.T), tuple(n))
    if int(getattr(grid, "n_sub", 0)) == 0:
        return cell
    table = np.asarray(grid.split_table)
    per_axis = round(table.shape[1] ** (1.0 / n.size))
    k = np.clip(np.floor((rel - idx) * per_axis).astype(int), 0, per_axis - 1)
    flat = np.ravel_multi_index(tuple(k.T), (per_axis,) * n.size)
    row = np.asarray(grid.split_of)[cell]
    return np.where(row >= 0, table[np.maximum(row, 0), flat], cell)


RIDGE_REL = 1e-8
"""Slope ridge of the ghost fit, relative to the normal matrix's mean diagonal."""

GHOST_FIELD = "current"
"""Nodes of the ghost fit in the sequential pull-back: ``"current"`` (the values and
velocities as chained so far -- the exact discrete adjoint; shipped) or ``"end_of_step"``
(one smooth fit per step; lower noise above ~0.5 collisions/particle/step but **biased**:
-23 % on the relaxation arbiter at 0.16, review section 5.10)."""

VIRTUAL = (4, 8)
"""``(substeps, partners)`` selects the virtual-partner substepped pull-back (the shipped
default: 4 substeps, 8 partners, review section 5.12 -- unbiased to < 1 % at 0.16
collisions/particle/step and 7-56x lower per-seed sd than the realized-pair forms);
``None`` selects the realized-pair forms below."""

VIRTUAL_NODES = "staged"
"""Node velocities of the virtual pull-back's fits: ``"staged"`` (shipped: the NTC step's
own intermediate states, replayed from the attempt record at the substep boundaries --
particles whose collision falls inside a substep move their value between velocities
through the fit at that boundary) or ``"post"`` (the end-of-step state at every
substep, so the field ``f`` inside the step is frozen: an O(dt) splitting error the
substep count cannot remove, -6 % on the relaxation arbiter)."""

SEQUENTIAL = True
"""With ``VIRTUAL = None``: pull the accepted attempts back in reverse order (the discrete
adjoint of the NTC step) rather than all at once with the end-of-step field (its
dt-linearisation).

The exact form chains realized single-collision corrections within a step, so its
variance grows with the collisions per particle per step -- the same terms whose
omission is the linearised form's bias.  Keep gradient runs at <= ~0.2 collisions per
particle per step (standard DSMC practice); review section 5.10 has the measurements."""

BASIS = "quadratic_full"
"""Local basis of the ghost interpolant: ``"affine"`` (P3's), ``"quadratic"`` (adds the
``|v|^2`` and ``x^2`` columns) or ``"quadratic_full"`` (every second-order velocity
monomial, so any quadratic field is exact).  The adjoint field of an energy-like reward is
quadratic in ``v``, and a cell-wide affine fit of it is biased by the curvature times
the node spread -- measured at -8 % on the 128-particle inlet shock (review section 5.4)."""


class LocalFit:
    """Weighted local polynomial fit centred at each query, cell-local nodes.

    Same neighbour weighting as :class:`ensemble_inflow.LocalAffine`; the design adds
    ``|v|^2 - |v_q|^2`` and ``(x - x_q)^2`` when ``basis == "quadratic"``.
    """

    def __init__(self, points, values, neighbors, scale, basis):
        self.scale = np.asarray(scale)
        self.points = np.asarray(points) / self.scale
        self.values = np.asarray(values)
        self.k = min(neighbors, len(self.points))
        self.basis = basis

    def _design(self, offsets, q, pts):
        cols = [np.ones((*offsets.shape[:2], 1)), offsets]
        if self.basis == "quadratic":
            v2 = np.sum(pts[..., 1:] ** 2, axis=-1) - np.sum(q[:, None, 1:] ** 2, axis=-1)
            cols.append(v2[..., None])
            cols.append(offsets[..., :1] ** 2)
        elif self.basis == "quadratic_full":
            # every second-order monomial of the velocity offset: a quadratic field of any
            # anisotropy is then represented exactly, and the intercept is its value at q
            dv = offsets[..., 1:]
            iu = np.triu_indices(3)
            cols.append((dv[..., :, None] * dv[..., None, :])[..., iu[0], iu[1]])
            cols.append(offsets[..., :1] ** 2)
        return np.concatenate(cols, axis=2)

    def __call__(self, query, gradient=False):
        from scipy.spatial import cKDTree  # noqa: PLC0415 -- optional heavy import

        query = np.asarray(query) / self.scale
        tree = cKDTree(self.points)
        dist, index = tree.query(query, k=self.k)
        dist, index = dist.reshape(len(query), self.k), index.reshape(len(query), self.k)
        pts = self.points[index]
        design = self._design(pts - query[:, None], query, pts)
        weights = 1 / (1 + (dist / np.maximum(dist[:, -1:], 1e-12)) ** 2)
        matrix = np.einsum("nki,nk,nkj->nij", design, weights, design)
        # ridge relative to the fit's own scale: an absolute 1e-12 leaves cells with fewer
        # nodes than columns numerically singular (solver-dependent results at the 0.3 %
        # level); 1e-8 of the mean diagonal is invisible where the fit is determined
        ridge = np.full(design.shape[2], RIDGE_REL)
        ridge[0] = 0.0
        diag_scale = np.einsum("nii->n", matrix) / design.shape[2]
        matrix += ridge[None] * diag_scale[:, None, None] * np.eye(design.shape[2])
        rhs = np.einsum("nki,nk,nk->ni", design, weights, self.values[index])
        beta = np.linalg.solve(matrix, rhs[..., None])[:, :, 0]
        if gradient:
            # slopes are per scaled coordinate: divide by the scale to get d/dx, d/dv
            return beta[:, 0], beta[:, 1 : 1 + self.points.shape[1]] / self.scale
        return beta[:, 0]


def _field(nodes_x, nodes_v, nodes_cell, values, alive, q_x, q_v, q_cell, neighbors, scale):
    """Cell-local weighted local-polynomial interpolation of ``values`` at the query states."""
    out = np.zeros(len(q_x))
    if len(q_x) == 0:
        return out
    for c in np.unique(q_cell):
        sel = alive & (nodes_cell == c)
        qs = q_cell == c
        if sel.sum() == 0:
            continue
        pts = np.column_stack((nodes_x[sel], nodes_v[sel]))
        if BASIS == "affine":
            fit = LocalAffine(pts, values[sel], neighbors, scale)
        else:
            fit = LocalFit(pts, values[sel], neighbors, scale, BASIS)
        out[qs] = fit(np.column_stack((q_x[qs], q_v[qs])))
    return out


def _exact_terminal_field(
    nodes_x, nodes_v, nodes_cell, values, alive, q_x, q_v, q_cell, k, s, phi, normalization
):
    del nodes_x, nodes_v, nodes_cell, values, alive, q_cell, k, s
    return np.asarray(phi(jnp.asarray(q_x), jnp.asarray(q_v))) / normalization


def _collision_pullback_sequential(pairs, k, x, v_post, alive, cell, mu, neighbors, scale):
    """The discrete adjoint of the sequential NTC step: attempts pulled back in reverse.

    Walking the accepted attempts of a cell backward with the *current* ``mu`` and the
    *current* velocities makes ``mu_i``, ``mu_j`` the value function right after the
    attempt, and the ghost nodes the population as it stood then -- no O(dt) linearisation
    of the step.  Cost is one cell-local interpolation per accepted attempt, in sequence.
    """
    acc = np.asarray(pairs.accept[k])  # (A, C)
    a_idx, c_idx = np.nonzero(acc)
    pi = np.asarray(pairs.pi[k])[a_idx, c_idx]
    pj = np.asarray(pairs.pj[k])[a_idx, c_idx]
    lam = mu.copy()
    if len(pi) == 0:
        return lam, Pairs(k, pi, pj, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    vi = np.asarray(pairs.vi[k])[a_idx, c_idx]
    vj = np.asarray(pairs.vj[k])[a_idx, c_idx]
    v_cur = v_post.copy()
    four = np.zeros(len(pi))
    # reverse attempt order within each cell; cells are independent
    for c in np.unique(c_idx):
        rows = np.flatnonzero(c_idx == c)[::-1]
        members = alive & (cell == c)
        if GHOST_FIELD == "end_of_step":
            # one smooth field per cell from the end-of-step nodes (diagnostic; biased)
            fixed = LocalFit(
                np.column_stack((x[members], v_post[members])), mu[members], neighbors, scale, BASIS
            )
        for r in rows:
            i, j = pi[r], pj[r]
            if GHOST_FIELD == "end_of_step":
                fit = fixed
            else:
                pts = np.column_stack((x[members], v_cur[members]))
                fit = LocalFit(pts, lam[members], neighbors, scale, BASIS)
            ghosts = fit(np.column_stack((x[[j, i]], np.stack((vj[r], vi[r])))))
            new_i = lam[i] + lam[j] - ghosts[0]
            new_j = lam[j] + lam[i] - ghosts[1]
            four[r] = lam[i] + lam[j] - ghosts[0] - ghosts[1]
            lam[i], lam[j] = new_i, new_j
            v_cur[i], v_cur[j] = vi[r], vj[r]
    return lam, Pairs(k, pi, pj, vi, vj, four)


_virtual_kernels = {}


def _virtual_kernel(sp):
    """Jitted VSS rate + scatter over batches of virtual pairs, cached per species table."""
    key = id(sp)
    if key not in _virtual_kernels:

        def kernel(vi, vj, u_chi, u_eps):
            q = collision_rate(vi, vj, 0, 0, sp)
            vi_n, vj_n = scatter(vi, vj, 0, 0, sp, u_chi, u_eps)
            return q, vi_n, vj_n

        _virtual_kernels[key] = jax.jit(jax.vmap(kernel))
    return _virtual_kernels[key]


def _staged_velocities(pairs, k, v_pre, v_post, n, substeps):
    """Velocities of every particle at the ``substeps + 1`` slot boundaries of step ``k``.

    Boundary ``b`` of cell ``c`` sits after ``n_att,c * b / substeps`` attempts; the state
    there is ``v_pre`` overwritten by the post-collision velocity of each particle's last
    accepted slot before the boundary (the NTC step processes slots in index order).
    Returns ``(substeps + 1, N, 3)`` with ``[0] == v_pre`` and ``[-1] == v_post``.
    """
    pi, pj = np.asarray(pairs.pi[k]), np.asarray(pairs.pj[k])
    accept = np.asarray(pairs.accept[k]) & (pi < n)
    vi_post, vj_post = np.asarray(pairs.vi_post[k]), np.asarray(pairs.vj_post[k])
    n_att = (pi < n).sum(axis=0)  # (C,)
    slot = np.broadcast_to(np.arange(pi.shape[0])[:, None], pi.shape)
    frac = np.where(n_att > 0, slot / np.maximum(n_att, 1)[None, :], 1.0)  # slot fraction
    stages = np.empty((substeps + 1, *v_pre.shape))
    stages[0] = v_pre
    stages[-1] = v_post
    for b in range(1, substeps):
        sel = accept & (frac < b / substeps)
        a, c = np.nonzero(sel)
        part = np.concatenate((pi[a, c], pj[a, c]))
        vel = np.concatenate((vi_post[a, c], vj_post[a, c]))
        order = np.lexsort((np.concatenate((a, a)),))  # ascending slot
        part, vel = part[order], vel[order]
        # the last accepted slot of each participant wins
        last = len(part) - 1 - np.unique(part[::-1], return_index=True)[1]
        stage = v_pre.copy()
        stage[part[last]] = vel[last]
        stages[b] = stage
    return stages


def _collision_pullback_virtual(  # noqa: PLR0915 -- one substep loop, kept whole
    pairs,
    k,
    x,
    v_pre,
    v_post,
    alive,
    cell,
    mu,
    neighbors,
    scale,
    sp,
    sigma,
    substeps,
    partners,
    rng,
):
    """Backward evolution ``exp(dt L*)`` of the field, substepped, with virtual partners.

    Each of ``substeps`` applies ``(dt / substeps) L*`` to every alive particle:
    ``K = partners`` virtual partners drawn from the cell, VSS angles drawn fresh,
    weight ``q(g)/Sigma_c`` (the NTC acceptance) times the cell's expected attempts per
    particle ``2 n_att,c / N_c``, all four field values from ONE smooth fit per cell per
    substep (so their errors cancel).  No realized-pair information enters except the
    attempt count per cell; the linearisation error is ``O(dt rate / substeps)``.  This is
    P3's Monte Carlo collision correction (eq. MC_collision_adjoint_inflow) with the exact
    NTC weights, cell-local fits and a substepped operator.  Collided particles finally
    move their value from the post- to the pre-collision state through the fit.
    """
    n = len(mu)
    c_count = np.bincount(cell[alive], minlength=len(sigma))
    active = np.asarray(pairs.pi[k]) < n  # (A, C) attempted slots
    n_att = active.sum(axis=0)  # attempts per cell
    per_particle = np.where(c_count > 0, 2.0 * n_att / np.maximum(c_count, 1), 0.0)
    lam = mu.copy()
    members = {c: np.flatnonzero(alive & (cell == c)) for c in np.unique(cell[alive])}
    kernel = _virtual_kernel(sp)
    kept = []
    if VIRTUAL_NODES == "staged":
        stages = _staged_velocities(pairs, k, v_pre, v_post, n, substeps)
    else:
        stages = np.broadcast_to(v_post, (substeps + 1, *v_post.shape)).copy()
        stages[0] = v_pre
    for s_back in reversed(range(substeps)):
        v_nodes = stages[s_back + 1]
        fits = {
            c: LocalFit(np.column_stack((x[m], v_nodes[m])), lam[m], neighbors, scale, BASIS)
            for c, m in members.items()
            if len(m) >= 2
        }
        i_all, j_all = [], []
        for m in members.values():
            if len(m) < 2:
                continue
            i_rep = np.repeat(m, partners)
            # draw among the OTHER members: index shift avoids self-pairs exactly
            own = np.repeat(np.arange(len(m)), partners)
            j_pick = m[(own + 1 + rng.integers(len(m) - 1, size=len(i_rep))) % len(m)]
            i_all.append(i_rep)
            j_all.append(j_pick)
        if not i_all:
            break
        i_all, j_all = np.concatenate(i_all), np.concatenate(j_all)
        u_chi = rng.random(len(i_all))
        u_eps = rng.random(len(i_all))
        q, vi_n, vj_n = kernel(
            jnp.asarray(v_nodes[i_all]),
            jnp.asarray(v_nodes[j_all]),
            jnp.asarray(u_chi),
            jnp.asarray(u_eps),
        )
        q, vi_n, vj_n = np.asarray(q), np.asarray(vi_n), np.asarray(vj_n)
        c_i = cell[i_all]
        w = q / sigma[c_i] * per_particle[c_i] / (partners * substeps)
        four = np.zeros(len(i_all))
        for c, fit in fits.items():
            sel = c_i == c
            if not sel.any():
                continue
            ii, jj = i_all[sel], j_all[sel]
            qx = np.concatenate((x[ii], x[jj], x[ii], x[jj]))
            qv = np.concatenate((vi_n[sel], vj_n[sel], v_nodes[ii], v_nodes[jj]))
            vals = fit(np.column_stack((qx, qv)))
            m4 = sel.sum()
            four[sel] = vals[:m4] + vals[m4 : 2 * m4] - vals[2 * m4 : 3 * m4] - vals[3 * m4 :]
        np.add.at(lam, i_all, w * four)
        kept.append((i_all, j_all, v_nodes[i_all], v_nodes[j_all], four, w))
        # particles whose state differs across this boundary move their value through
        # the field as it stands (under "post" nodes this is the one end-of-step move)
        v_prev = stages[s_back]
        moved = np.flatnonzero(alive & np.any(v_prev != v_nodes, axis=1))
        if len(moved) == 0:
            continue
        fits = {
            c: LocalFit(np.column_stack((x[m], v_nodes[m])), lam[m], neighbors, scale, BASIS)
            for c, m in members.items()
            if len(m) >= 2
        }
        for c, fit in fits.items():
            sel = moved[cell[moved] == c]
            if len(sel) == 0:
                continue
            vals = fit(
                np.column_stack(
                    (np.concatenate((x[sel], x[sel])), np.concatenate((v_prev[sel], v_nodes[sel])))
                )
            )
            lam[sel] += vals[: len(sel)] - vals[len(sel) :]
    if not kept:
        empty = np.zeros(0)
        return lam, Pairs(
            k,
            empty.astype(int),
            empty.astype(int),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            empty,
            empty,
        )
    # kernel-parameter scores: every unordered pair is generated from both of its members
    # (the per-particle sums cover the pair measure twice), hence the 1/2
    cols = [np.concatenate(c) for c in zip(*kept, strict=True)]
    return lam, Pairs(k, cols[0], cols[1], cols[2], cols[3], cols[4], 0.5 * cols[5])


def _collision_pullback(pairs, k, x, v_post, alive, cell, mu, neighbors, scale, field=None):
    """``lam_i = mu_i + sum_pairs [Lam(partner post) - Lam(partner pre)]`` on one step.

    Returns ``(lam, Pairs)``; the latter carries each accepted pair's four-term value.
    """
    field = _field if field is None else field
    acc = np.asarray(pairs.accept[k]).reshape(-1)
    pi = np.asarray(pairs.pi[k]).reshape(-1)[acc]
    pj = np.asarray(pairs.pj[k]).reshape(-1)[acc]
    lam = mu.copy()
    if len(pi) == 0:
        return lam, Pairs(k, pi, pj, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    vi = np.asarray(pairs.vi[k]).reshape(-1, 3)[acc]
    vj = np.asarray(pairs.vj[k]).reshape(-1, 3)[acc]
    vi_n = np.asarray(pairs.vi_post[k]).reshape(-1, 3)[acc]
    vj_n = np.asarray(pairs.vj_post[k]).reshape(-1, 3)[acc]
    # each member of each pair receives the OTHER member's field difference
    slots = np.concatenate((pj, pi))
    owner = np.concatenate((pi, pj))
    v_pre_p = np.concatenate((vj, vi))
    v_post_p = np.concatenate((vj_n, vi_n))
    # the carried value is exact where this was the partner's last collision of the step
    last = np.all(np.isclose(v_post_p, v_post[slots]), axis=1)
    q_x = np.concatenate((x[slots], x[slots[~last]]))
    q_v = np.concatenate((v_pre_p, v_post_p[~last]))
    q_cell = np.concatenate((cell[slots], cell[slots[~last]]))
    vals = field(x, v_post, cell, mu, alive, q_x, q_v, q_cell, neighbors, scale)
    lam_pre = vals[: len(slots)]
    lam_post = np.where(last, mu[slots], 0.0)
    lam_post[~last] = vals[len(slots) :]
    np.add.at(lam, owner, lam_post - lam_pre)
    # four-term per pair: [post_i + post_j] - [pre_i + pre_j]; the first half of the
    # concatenated arrays is "j seen from i", the second "i seen from j"
    m = len(pi)
    four = (lam_post[:m] + lam_post[m:]) - (lam_pre[:m] + lam_pre[m:])
    return lam, Pairs(k, pi, pj, vi, vj, four)


def backward(  # noqa: PLR0915 -- the sweep reads best as one unit
    records,
    initial_state,
    final_state,
    phi,
    normalization,
    grid,
    v_scale,
    neighbors=32,
    exact_last=False,
    collision_term=True,
    sp=None,
    seed=0,
):
    """Carry the adjoint from ``t_M`` to ``t_0``; returns :class:`Adjoint`.

    ``sp`` (species parameters) and ``seed`` are needed by the virtual-partner
    pull-back (``VIRTUAL = (substeps, partners)``), which draws its own partners.

    ``exact_last`` evaluates the last step's ghosts with the terminal reward (known in
    closed form) instead of the interpolant -- a diagnostic that isolates the
    interpolation error of that step.  ``collision_term=False`` drops the partner
    terms altogether (pure bookkeeping + scores): a diagnostic for their size.

    ``phi(x, v) -> f64[N]`` is the terminal per-particle reward; ``J = sum_alive phi /
    normalization`` with a FIXED normalization.  ``v_scale`` (a thermal speed) and the
    cell width set the phase-space metric of the interpolant.
    """
    rec = records
    steps = int(rec.x.shape[0])
    dim = int(rec.x.shape[2])
    n_part = int(rec.x.shape[1])
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    scale = np.concatenate((dx, np.full(3, v_scale)))
    x_final = np.asarray(final_state.x)
    v_final = np.asarray(final_state.v)
    alive_final = np.asarray(final_state.alive)
    mu = np.where(alive_final, np.asarray(phi(x_final, v_final)) / normalization, 0.0)
    births, hits, pair_records = [], [], []
    rng = np.random.default_rng(seed)
    if VIRTUAL is not None and sp is None:
        raise ValueError("the virtual-partner pull-back needs the species parameters")
    for k in reversed(range(steps)):
        x = np.asarray(rec.x[k]).reshape(n_part, dim)
        v_post = np.asarray(rec.v_post[k])
        alive = np.asarray(rec.alive[k])
        born_face = np.asarray(rec.born_face[k])
        cell = _cells(x, grid)
        # --- 1. collision pull-back ---------------------------------------------
        field = _field
        if exact_last and k == steps - 1:
            field = functools.partial(_exact_terminal_field, phi=phi, normalization=normalization)
        if collision_term and VIRTUAL is not None and not (exact_last and k == steps - 1):
            substeps, partners = VIRTUAL
            lam, step_pairs = _collision_pullback_virtual(
                rec.pairs,
                k,
                x,
                np.asarray(rec.v[k]),
                v_post,
                alive,
                cell,
                mu,
                neighbors,
                scale,
                sp,
                np.asarray(rec.sigma[k]),
                substeps,
                partners,
                rng,
            )
            pair_records.append(step_pairs)
        elif collision_term and SEQUENTIAL and not (exact_last and k == steps - 1):
            lam, step_pairs = _collision_pullback_sequential(
                rec.pairs, k, x, v_post, alive, cell, mu, neighbors, scale
            )
            pair_records.append(step_pairs)
        elif collision_term:
            lam, step_pairs = _collision_pullback(
                rec.pairs, k, x, v_post, alive, cell, mu, neighbors, scale, field
            )
            pair_records.append(step_pairs)
        else:
            lam = mu.copy()
        lam = np.where(alive, lam, 0.0)
        # --- 2. sources ---------------------------------------------------------
        born = born_face >= 0
        if born.any():
            births.append(
                Births(
                    k,
                    np.flatnonzero(born),
                    born_face[born],
                    np.asarray(rec.born_v[k])[born],
                    lam[born],
                )
            )
        # --- 2b. wall hits: the realized re-emission is the kernel's unbiased sample, so the
        #        value passes through unchanged; the wall parameters' scores need lam there ---
        walls = rec.walls
        if walls is not None:
            face, v_out, diffuse = _step_walls(walls, k)  # (K_bounce, N), dense or lane
            for bounce in range(face.shape[0]):
                hit = (face[bounce] >= 0) & alive
                if hit.any():
                    hits.append(
                        Hits(
                            k,
                            np.flatnonzero(hit),
                            face[bounce][hit],
                            v_out[bounce][hit],
                            diffuse[bounce][hit],
                            lam[hit],
                        )
                    )
        # --- 3. move pull-back: survivors keep lam, absorbed and newborn slots reset ---
        mu = np.where(alive & ~born, lam, 0.0)
    alive0 = np.asarray(initial_state.alive)
    return Adjoint(np.where(alive0, mu, 0.0), births, normalization, hits, pair_records)


def _step_walls(walls, k):
    """Step ``k`` of a record's wall events as dense numpy rows ``(face, v_out, diffuse)``
    over every slot: a dense record as stored; a lane record (``SplitEvents``: pass 0 in
    ``head``, the later passes of the lane entries in ``tail`` with their ``slot``) with
    the tail rows scattered to their slots (one step at a time -- the dense form of a
    whole record is what the lane avoids storing)."""
    if not hasattr(walls, "head"):
        return np.asarray(walls.face[k]), np.asarray(walls.v_out[k]), np.asarray(walls.diffuse[k])
    if hasattr(walls, "pass_"):
        return _step_record_walls(walls, k)
    head_f = np.asarray(walls.head.face[k])
    n = head_f.shape[1]
    slot = np.asarray(walls.slot[k])
    use = slot >= 0
    tail_f = np.asarray(walls.tail.face[k])
    rows = 1 + tail_f.shape[0]
    face = np.full((rows, n), -1, dtype=head_f.dtype)
    v_out = np.zeros((rows, n, 3))
    diffuse = np.zeros((rows, n), dtype=bool)
    face[0], v_out[0], diffuse[0] = (
        head_f[0],
        np.asarray(walls.head.v_out[k])[0],
        np.asarray(walls.head.diffuse[k])[0],
    )
    face[1:, slot[use]] = tail_f[:, use]
    v_out[1:, slot[use]] = np.asarray(walls.tail.v_out[k])[:, use]
    diffuse[1:, slot[use]] = np.asarray(walls.tail.diffuse[k])[:, use]
    return face, v_out, diffuse


def _pad_rows(walls_k, rows):
    """Dense step events ``(face, v_out, diffuse)`` padded with empty rows to ``rows``."""
    face, v_out, diffuse = walls_k
    extra = rows - face.shape[0]
    if extra <= 0:
        return walls_k
    n = face.shape[1]
    return (
        np.concatenate((face, np.full((extra, n), -1, dtype=face.dtype))),
        np.concatenate((v_out, np.zeros((extra, n, 3)))),
        np.concatenate((diffuse, np.zeros((extra, n), dtype=bool))),
    )


def _step_record_walls(walls, k):
    """:func:`_step_walls` of a compact lane record (``wall_adjoint.RecordEvents``): pass 0
    from the head, every listed later-pass event at row ``pass_`` of its slot."""
    head_f = np.asarray(walls.head.face[k])
    n = head_f.shape[1]
    slot = np.asarray(walls.slot[k])
    pas = np.asarray(walls.pass_[k])
    use = slot >= 0
    rows = 1 + (int(pas[use].max()) if use.any() else 0)
    face = np.full((rows, n), -1, dtype=head_f.dtype)
    v_out = np.zeros((rows, n, 3))
    diffuse = np.zeros((rows, n), dtype=bool)
    face[0] = head_f[0]
    v_out[0] = np.asarray(walls.head.v_out[k])[0]
    diffuse[0] = np.asarray(walls.head.diffuse[k])[0]
    face[pas[use], slot[use]] = np.asarray(walls.face[k])[use]
    v_out[pas[use], slot[use]] = np.asarray(walls.v_out[k])[use]
    diffuse[pas[use], slot[use]] = np.asarray(walls.diffuse[k])[use]
    return face, v_out, diffuse


def _step_rewards(rewards, k, n):
    """Step ``k`` of the backward's hit rewards as dense rows over every slot: a dense
    ``f64[M, K, N]`` array as stored, or a lane replay's ``shape.HitRewards`` (per-event
    values with their lane slots) scattered like :func:`_step_walls`."""
    if not hasattr(rewards, "values"):
        return np.asarray(rewards[k])
    head = np.asarray(rewards.values.head[k])
    tail = np.asarray(rewards.values.tail[k])
    slot = np.asarray(rewards.slot[k])
    use = slot >= 0
    out = np.zeros((1 + tail.shape[0], n))
    out[0] = head[0]
    out[1:, slot[use]] = tail[:, use]
    return out


def from_device(records, dev, normalization, hit_reward=None):
    """The :class:`Adjoint` of a device backward (:mod:`jax_dsmc.ensemble_device`): the
    same per-step bookkeeping of births and wall hits as :func:`backward`, read off the
    device's ``lam[M, N]``.  ``pairs`` is ``None`` -- kernel-parameter scores come from the
    device's own ``four_term`` (sequential) or ``kernel`` (virtual) outputs.

    With value-form hit rewards (``hit_reward`` ``f64[M, K, N]``, the array the backward
    was given) the weight of a hit's score is the reward-to-go from the re-emission draw
    onward: ``lam`` at the post-move state plus the rewards of this hit (it is a function
    of the drawn velocity) and of the step's later hits of the same slot; a newborn's
    source score is weighted by ``lam`` plus all its birth-step hit rewards.
    """
    rec = records
    lam_all = np.asarray(dev.lam)
    rewards = None if hit_reward is None else hit_reward
    if rewards is not None and not hasattr(rewards, "values"):
        rewards = np.asarray(rewards)
    births, hits = [], []
    for k in range(lam_all.shape[0]):
        lam = lam_all[k]
        alive = np.asarray(rec.alive[k])
        born_face = np.asarray(rec.born_face[k])
        born = born_face >= 0
        # to_go[j] = lam + sum of the rewards of hits j, j+1, ... of the step (row K = lam)
        walls_k = None if rec.walls is None else _step_walls(rec.walls, k)
        r_k = None
        if rewards is not None and walls_k is not None:
            r_k = _step_rewards(rewards, k, lam.shape[0])
            # a compact record has only the rows it uses, a replay's rewards all K: the
            # common height (rows without events carry zero rewards / no events)
            walls_k = _pad_rows(walls_k, r_k.shape[0])
        n_pass = 0 if walls_k is None else walls_k[0].shape[0]
        to_go = np.broadcast_to(lam, (n_pass + 1, lam.shape[0])).copy()
        if r_k is not None and n_pass:
            full = np.zeros((n_pass, lam.shape[0]))
            full[: r_k.shape[0]] = r_k[:n_pass]
            to_go[:-1] += np.cumsum(full[::-1], axis=0)[::-1]
        if born.any():
            births.append(
                Births(
                    k,
                    np.flatnonzero(born),
                    born_face[born],
                    np.asarray(rec.born_v[k])[born],
                    to_go[0][born],
                )
            )
        if walls_k is not None:
            face, v_out, diffuse = walls_k
            for bounce in range(face.shape[0]):
                hit = (face[bounce] >= 0) & alive
                if hit.any():
                    hits.append(
                        Hits(
                            k,
                            np.flatnonzero(hit),
                            face[bounce][hit],
                            v_out[bounce][hit],
                            diffuse[bounce][hit],
                            to_go[bounce][hit],
                        )
                    )
    return Adjoint(np.asarray(dev.lambda0), births, normalization, hits, None)


def max_occupancy(records, grid):
    """Largest per-cell particle count over the captured steps (the device tables' ``npc_max``
    must be at least this, or nodes are silently dropped)."""
    n_cells = int(np.prod(grid.n))
    worst = 0
    for k in range(records.x.shape[0]):
        x = np.asarray(records.x[k])
        alive = np.asarray(records.alive[k])
        cell = _cells(x.reshape(len(x), -1), grid)
        worst = max(worst, int(np.bincount(cell[alive], minlength=n_cells).max()))
    return worst


def configure(pullback="virtual", substeps=4, partners=8, nodes="staged"):
    """Select the collision pull-back for the process: ``virtual`` (the shipped default:
    substepped operator with virtual partners, fit nodes ``"staged"`` or ``"post"``),
    ``sequential`` (exact, realized pairs) or ``linearised`` (batched realized pairs).
    Called without arguments it restores the defaults.  Returns a short tag for artifact
    names (the sequential form's is empty: it named the artifacts before section 5.12)."""
    global SEQUENTIAL, VIRTUAL, VIRTUAL_NODES  # noqa: PLW0603 -- process-wide selection
    if nodes not in ("post", "staged"):
        raise ValueError("nodes must be post or staged")
    VIRTUAL_NODES = nodes
    if pullback == "sequential":
        SEQUENTIAL, VIRTUAL = True, None
        return ""
    if pullback == "linearised":
        SEQUENTIAL, VIRTUAL = False, None
        return "_lin"
    if pullback == "virtual":
        SEQUENTIAL, VIRTUAL = False, (int(substeps), int(partners))
        return f"_virt{int(substeps)}x{int(partners)}{'st' if nodes == 'staged' else ''}"
    raise ValueError("pullback must be sequential, linearised or virtual")
