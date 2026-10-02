"""Particle-carried ensemble adjoint for P3's inflow model (route "b").

Same continuum adjoint as :mod:`ensemble_inflow` (route "a", the paper's retained
frames + local-affine reconstruction), quadratured differently:

* the adjoint ``mu_i`` is carried on every forward particle at its own state;
* the collision term uses the **realized** forward pairs as its Monte Carlo sample
  of the collision measure.  For a pair ``(i, j) -> (i', j')`` the four-term sum is
  ``lam(v_i') + lam(v_j') - lam(v_i) - lam(v_j)``; the first two are the carried values
  ``mu_i``, ``mu_j`` and ``lam(v_i)`` cancels against the transport term, leaving

      mu_i^{k} = mu_i + [i collided] * (mu_j - lam~(x_j, v_j^pre))

  so the only interpolated quantity is the *partner's ghost*: the field at where the
  partner would have been had it not collided.  Uncollided particles need nothing.

Sources contribute ``sum_born mu_born * score(v_born)`` over **all** births (escaped
births carry ``mu = 0``), with no subsampling.  Interpolation happens at one ghost
per collided particle instead of at every retained state every step, which is why
this route is expected to carry less of the Jensen bias measured in
``scratch/bench/probe_recon_bias.py``.

The forward consumes exactly the random streams of :func:`ensemble_inflow.forward`, in
the same order, so the two routes see bit-identical trajectories (checked by
``tests/test_ensemble_particle.py``).  Host NumPy/SciPy, like the reference.
"""

from dataclasses import dataclass

import numpy as np

from jax_dsmc import ensemble_backward as eb
from jax_dsmc import ensemble_inflow as ei


@dataclass
class Step:
    """Everything the backward pass needs about one forward step ``k -> k+1``."""

    pre: np.ndarray
    """``f64[n, 4]`` states at ``t_k`` (before collision), in slot order."""
    pairs: np.ndarray
    """``i32[p, 2]`` slots of the collided pairs (disjoint)."""
    survived: np.ndarray
    """``bool[n]`` whether the slot is inside the domain after transport."""
    next_slot: np.ndarray
    """``i32[n]`` slot at ``t_{k+1}`` for survivors (compaction map), ``-1`` otherwise."""
    born: list
    """per side: ``f64[N_s, 4]`` all births (escaped ones included)."""
    born_survived: list
    """per side: ``bool[N_s]``."""
    born_slot: list
    """per side: ``i32[N_s]`` slot at ``t_{k+1}`` for surviving births."""


@dataclass
class Trace:
    steps: list
    final: np.ndarray
    temperatures: np.ndarray
    injection_counts: tuple
    normalization: int
    objective: float
    config: ei.Config
    terminal_reward: object


def forward(  # noqa: PLR0915 -- mirrors the reference forward statement for statement
    cfg, temperatures, seed=0, injection_counts=None, terminal_reward=ei.reward, initial_seed=0
):
    """:func:`ensemble_inflow.forward` with the same draws, recording pairs and births.

    Retained-frame subsampling is a separate stream in the reference, so skipping it
    here leaves the physical trajectory unchanged.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.shape != (2, 3) or np.any(temperatures <= 0):
        raise ValueError("Expected positive (2, 3) boundary temperature variances")
    counts = (
        ei.nominal_counts(cfg, temperatures)
        if injection_counts is None
        else tuple(injection_counts)
    )
    counts = tuple(map(int, counts))
    rng = np.random.default_rng(np.random.SeedSequence([initial_seed, 0]))
    points = np.column_stack(
        (rng.random(cfg.n_initial), rng.normal(size=(cfg.n_initial, 3)) * cfg.initial_std)
    )
    steps = []
    for k in range(cfg.steps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, k]))
        pre = points.copy()
        cell = ei._cells(points, cfg)
        pairs = []
        for j in range(cfg.cells):
            ids = rng.permutation(np.flatnonzero(cell == j))
            n = len(ids)
            rate = n * cfg.cells / cfg.n_initial * cfg.q0 * cfg.dt
            if rate > 1:
                raise ValueError("dt * rho * Q0 exceeds one; reduce dt for disjoint collisions")
            number = int(np.floor(n * rate / 2 + rng.random()))
            if 2 * number > n:
                raise ValueError(
                    "Rounded collision count exceeds disjoint-pair capacity; reduce dt"
                )
            a, b = ids[: 2 * number : 2], ids[1 : 2 * number : 2]
            points[a, 1:], points[b, 1:] = ei.scatter_maxwell(points[a, 1:], points[b, 1:], rng)
            pairs.append(np.column_stack((a, b)))
        pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), int)
        points[:, 0] += cfg.dt * points[:, 1]
        survived = (points[:, 0] >= 0) & (points[:, 0] <= 1)
        next_slot = np.full(len(points), -1)
        next_slot[survived] = np.arange(survived.sum())
        points = points[survived]
        born, born_survived, born_slot = [], [], []
        for side, number in enumerate(counts):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3, k, side]))
            v = ei.sample_flux(rng, number, temperatures[side], side)
            b = np.column_stack((side + cfg.dt * rng.random(number) * v[:, 0], v))
            inside = (b[:, 0] >= 0) & (b[:, 0] <= 1)
            slot = np.full(number, -1)
            slot[inside] = len(points) + np.arange(inside.sum())
            born.append(b)
            born_survived.append(inside)
            born_slot.append(slot)
            points = np.concatenate((points, b[inside]))
        steps.append(Step(pre, pairs, survived, next_slot, born, born_survived, born_slot))
    normalization = cfg.n_initial + cfg.steps * sum(counts)
    return Trace(
        steps,
        points,
        temperatures.copy(),
        counts,
        normalization,
        float(np.sum(terminal_reward(points)) / normalization),
        cfg,
        terminal_reward,
    )


def ghost_field(states, values, queries, neighbors, scale):
    """Local-affine interpolation of the carried adjoint at ghost states.

    The particles at ``t_{k+1}`` are the interpolation nodes; the same weighted
    affine fit as :class:`ensemble_inflow.LocalAffine`, so route (b) differs from
    (a) only in *where* and *how often* it interpolates.
    """
    if len(queries) == 0:
        return np.zeros(0)
    return ei.LocalAffine(states, values, neighbors, scale)(queries)


def _source_gradient(st, mu_next, states_next, trace, source_term, neighbors, scale):
    """One step's contribution of both sources, by the score or the pathwise route."""
    gradient = np.zeros((2, 3))
    for side in range(2):
        alive_b = st.born_survived[side]
        slots_b = st.born_slot[side][alive_b]
        temp = trace.temperatures[side]
        if source_term == "score":
            mu_born = np.zeros(len(st.born[side]))
            mu_born[alive_b] = mu_next[slots_b]
            gradient[side] += np.sum(
                mu_born[:, None] * ei.temperature_score(st.born[side][:, 1:], temp), axis=0
            )
        elif alive_b.any():
            # slope of the t_{k+1} field at the surviving newborns (escaped: lam = 0)
            born_pts = states_next[slots_b]
            _, grad = eb.LocalFit(states_next, mu_next, neighbors, scale, eb.BASIS)(
                born_pts, gradient=True
            )
            v = born_pts[:, 1:]
            dx = born_pts[:, 0] - side  # xi * v_1
            dv = v / (2 * temp)  # d v_l / d T_l
            gradient[side, 0] += np.sum(grad[:, 0] * dx / (2 * temp[0]) + grad[:, 1] * dv[:, 0])
            gradient[side, 1] += np.sum(grad[:, 2] * dv[:, 1])
            gradient[side, 2] += np.sum(grad[:, 3] * dv[:, 2])
    return gradient


def backward(
    trace, cfg=None, neighbors=None, terminal_reward=None, exact_last=False, source_term="score"
):
    """Six source-temperature derivatives from one particle-carried backward sweep.

    ``exact_last`` evaluates the last step's ghosts with the terminal reward itself
    (the field at ``t_M`` is known in closed form), which isolates the interpolation
    error of that step; it is a diagnostic, not a production option.

    ``source_term``: ``"score"`` is P3's ``sum_births lam * d log b/dT``;
    ``"pathwise"`` reparameterises the birth, ``v = sqrt(T) (r, z2, z3)``,
    ``x = L + xi v_1``, so ``d/dT E[lam(x_b, v_b)] = E[(grad_x lam (x_b - L) +
    grad_v lam . v_b) / (2T)]`` with the field slope at the newborn's own node
    (:class:`ensemble_backward.LocalFit` with ``gradient=True``).  Both are
    unbiased; the pathwise form trades the score's variance for the slope's.
    """
    if source_term not in ("score", "pathwise"):
        raise ValueError("source_term must be score or pathwise")
    cfg = trace.config if cfg is None else cfg
    terminal_reward = trace.terminal_reward if terminal_reward is None else terminal_reward
    neighbors = cfg.neighbors if neighbors is None else neighbors
    scale = np.array([1 / cfg.cells, cfg.initial_std, cfg.initial_std, cfg.initial_std])
    # mu at t_M on the final particles
    mu_next = terminal_reward(trace.final) / trace.normalization
    states_next = trace.final
    gradient = np.zeros((2, 3))
    for k in reversed(range(cfg.steps)):
        st = trace.steps[k]
        # 1. sources: every birth carries its own adjoint (escaped births: 0)
        gradient += _source_gradient(st, mu_next, states_next, trace, source_term, neighbors, scale)
        # 2. transport pull-back: lam~ at the post-collision state = mu at t_{k+1}, 0 if absorbed
        n = len(st.pre)
        mu = np.zeros(n)
        mu[st.survived] = mu_next[st.next_slot[st.survived]]
        # 3. collisions: mu_i += mu_j - lam~(ghost_j); the ghost is the partner's
        #    pre-collision state advected as if it had not collided
        if len(st.pairs):
            a, b = st.pairs[:, 0], st.pairs[:, 1]
            ghost_a, ghost_b = st.pre[a].copy(), st.pre[b].copy()
            ghost_a[:, 0] += cfg.dt * ghost_a[:, 1]
            ghost_b[:, 0] += cfg.dt * ghost_b[:, 1]
            ghosts = np.concatenate((ghost_a, ghost_b))
            inside = (ghosts[:, 0] >= 0) & (ghosts[:, 0] <= 1)
            lam = np.zeros(len(ghosts))
            if exact_last and k == cfg.steps - 1:
                lam[inside] = terminal_reward(ghosts[inside]) / trace.normalization
            else:
                lam[inside] = ghost_field(states_next, mu_next, ghosts[inside], neighbors, scale)
            lam_a, lam_b = lam[: len(a)], lam[len(a) :]
            mu_pre = mu.copy()
            mu[a] = mu_pre[a] + mu_pre[b] - lam_b
            mu[b] = mu_pre[b] + mu_pre[a] - lam_a
        mu_next, states_next = mu, st.pre
    return gradient
