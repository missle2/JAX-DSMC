"""Step rules of the level-set optimiser (:mod:`examples.mesh.optimize`) beyond GD / Adam,
2026-10-19: what the drag problem's landscape rewards.

Every rule works on the area-tangent gradient ``g`` (``dJ/dcoef`` without its component along
``grad A``) and its smoothed, band-restricted form ``S g`` (the H1 metric of
:func:`optimize.smoothed_direction`), and returns a coefficient step whose largest entry --
the largest normal displacement, ``|grad phi| = 1`` -- the rule chooses.

* ``tr``: steepest descent in the smoothed metric inside a **trust region** of radius
  ``Delta`` (the largest normal displacement).  Each evaluated trial is compared with its
  base point: ``rho = (J_trial - J_base) / predicted``.  The objective is evaluated on the
  same seeds every iterate (common random numbers), so this difference is far less noisy
  than either value -- ``rho`` is the quantity a stochastic objective lets us measure well.
  ``rho > 0.75`` (and the step at the radius) grows ``Delta`` x1.5, ``rho < 0.25`` shrinks
  it x0.5, ``rho < 0`` also **rejects** the trial: the next step starts again from the base
  point (its gradient is kept).  A prediction smaller than half the objective's seed
  standard error is inconclusive: accepted, ``Delta`` kept.
* ``lbfgs``: the same trust region around a limited-memory BFGS direction -- with one
  (exactly restored) area equality this is the quasi-Newton SQP step on the constant-area
  set.  The initial inverse Hessian is ``gamma S`` (the smoothing as preconditioner, ``gamma
  = s.y / y.S y``); pairs ``(s, y)`` come from every evaluated trial against its base
  (rejected ones included: their curvature is real) and are skipped unless ``s . y > 0``.
  The step is the full quasi-Newton step when it fits in the radius, else scaled to it.
* ``mma``: Svanberg's method of moving asymptotes (the 2007 form of ``mmasub``) on the
  band's coefficients with the smoothed FULL gradient (not the area-tangent one) and the area
  as the single inequality ``A >= A0`` -- the drag falls as the body shrinks, so it is active,
  the canonical MMA setting of a volume-constrained problem (an equality as two opposed
  inequalities gives both convexified constraints curvature and stalls the step) -- with the
  move limit ``Delta = step`` around the current point; its
  own per-coefficient asymptotes shrink where successive moves change sign (its oscillation
  control) and widen where they do not.  The 1-D dual is maximised with L-BFGS-B.  No
  acceptance test (plain MMA, not GCMMA).

State lives in ``<run>/opt_state.npz`` (:func:`load_state` / :func:`save_state`), so a
resumed run continues the rule where it stopped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RULES = ("tr", "lbfgs", "mma")
GROW, SHRINK = 1.5, 0.5
RHO_GOOD, RHO_POOR = 0.75, 0.25
LBFGS_MEMORY = 5


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def save_state(path: Path, state: dict) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in state.items() if v is not None})


# --- the trust region (tr, lbfgs) -----------------------------------------------------------


def radius_bounds(step: float) -> tuple[float, float]:
    """``(Delta_min, Delta_max)``: a quarter of the nominal step to four of them (0.4 DSMC
    cells at the default 0.05 m)."""
    return step / 4.0, step * 4.0


def judge(state, objective, objective_se, step):
    """Compare the trial just evaluated (``objective``) with its base point.  Returns
    ``(accepted, rho, radius)``; the first evaluation is its own base."""
    radius = float(state.get("radius", step))
    if "base_objective" not in state:
        return True, float("nan"), radius
    lo, hi = radius_bounds(step)
    predicted = float(state["predicted"])
    actual = objective - float(state["base_objective"])
    if abs(predicted) < 0.5 * objective_se:
        return True, float("nan"), radius  # inconclusive: below the seed noise
    rho = actual / predicted
    at_edge = bool(state.get("at_edge", True))
    if rho < RHO_POOR:
        radius = max(lo, radius * SHRINK)
    elif rho > RHO_GOOD and at_edge:
        radius = min(hi, radius * GROW)
    return rho >= 0.0, rho, radius


def fit_to_radius(direction, radius):
    """The step ``direction`` clipped to the trust region (largest entry ``radius``) and
    whether it was clipped (the step reaches the edge)."""
    big = float(np.abs(direction).max())
    if big <= radius:
        return direction, False
    return direction * (radius / big), True


def lbfgs_direction(g, pairs, smooth):
    """``-H g`` by the two-loop recursion with ``H0 = gamma S`` (``smooth`` applies ``S``);
    ``pairs`` a list of ``(s, y)``, oldest first.  Without pairs: ``-S g`` (unscaled)."""
    if not pairs:
        return -smooth(g), False
    q = g.copy()
    alphas = []
    for s, y in reversed(pairs):
        rho = 1.0 / float(np.sum(s * y))
        a = rho * float(np.sum(s * q))
        alphas.append((a, rho, s, y))
        q = q - a * y
    s, y = pairs[-1]
    sy = smooth(y)
    gamma = float(np.sum(s * y)) / max(float(np.sum(y * sy)), 1e-300)
    r = gamma * smooth(q)
    for a, rho, s, y in reversed(alphas):
        b = rho * float(np.sum(y * r))
        r = r + s * (a - b)
    return -r, True


def add_pair(pairs, s, y):
    """Keep ``(s, y)`` if it has positive curvature; the newest ``LBFGS_MEMORY`` pairs."""
    if float(np.sum(s * y)) <= 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)):
        return pairs, False
    return [*pairs, (s, y)][-LBFGS_MEMORY:], True


def pairs_of(state):
    if "lbfgs_s" not in state:
        return []
    return list(zip(state["lbfgs_s"], state["lbfgs_y"], strict=True))


def pairs_state(pairs):
    if not pairs:
        return {}
    return {"lbfgs_s": np.stack([p[0] for p in pairs]), "lbfgs_y": np.stack([p[1] for p in pairs])}


# --- MMA ------------------------------------------------------------------------------------


def _mma_terms(grad, x, low, upp, span):
    """Svanberg's ``(p, q)`` of a gradient (2007: the 1.001 / 0.001 split and the
    ``1e-5 / span`` regularisation make every variable's subproblem strictly convex)."""
    pos, neg = np.maximum(grad, 0.0), np.maximum(-grad, 0.0)
    reg = 1e-5 / span
    p = (upp - x) ** 2 * (1.001 * pos + 0.001 * neg + reg)
    q = (x - low) ** 2 * (0.001 * pos + 1.001 * neg + reg)
    return p, q


def mma_asymptotes(state, x, move, k):
    """The asymptotes ``(low, upp)`` of iterate ``k`` (Svanberg's update: x0.7 where the last
    two moves changed sign, x1.2 where they did not, clamped to ``[0.01, 10]`` spans)."""
    span = 2.0 * move
    if k < 2 or "mma_x1" not in state or "mma_x2" not in state:
        return x - 0.5 * span, x + 0.5 * span
    x1, x2 = state["mma_x1"], state["mma_x2"]
    sign = (x - x1) * (x1 - x2)
    gamma = np.where(sign < 0.0, 0.7, np.where(sign > 0.0, 1.2, 1.0))
    low = x - gamma * (x1 - state["mma_low"])
    upp = x + gamma * (state["mma_upp"] - x1)
    low = np.clip(low, x - 10.0 * span, x - 0.01 * span)
    upp = np.clip(upp, x + 0.01 * span, x + 10.0 * span)
    return low, upp


def mma_step(x, g0, area, g_area, area0, low, upp, move):
    """One MMA subproblem on the variables ``x`` (1-D arrays: the band's coefficients):
    objective gradient ``g0``, the constraint ``(area0 - area) / area0 <= 0``.  Returns the
    new ``x`` and the multiplier (``f64[1]``)."""
    span = 2.0 * move
    alpha = np.maximum.reduce([x - move, low + 0.1 * (x - low), x - 0.5 * span])
    beta = np.minimum.reduce([x + move, upp - 0.1 * (upp - x), x + 0.5 * span])
    p0, q0 = _mma_terms(g0, x, low, upp, span)
    grads = (-g_area / area0,)
    values = ((area0 - area) / area0,)
    pq = [_mma_terms(gi, x, low, upp, span) for gi in grads]
    b = np.array([np.sum(p / (upp - x) + q / (x - low)) - f
                  for (p, q), f in zip(pq, values, strict=True)])  # fmt: skip
    c_art, d_art = 1000.0, 1.0  # Svanberg's artificial variables: a0 = 1, a = 0

    def primal(lam):
        pl = p0 + sum(li * p for li, (p, _q) in zip(lam, pq, strict=True))
        ql = q0 + sum(li * q for li, (_p, q) in zip(lam, pq, strict=True))
        sp, sq = np.sqrt(pl), np.sqrt(ql)
        xs = np.clip((sp * low + sq * upp) / np.maximum(sp + sq, 1e-300), alpha, beta)
        return xs, pl, ql

    def negative_dual(lam):
        xs, pl, ql = primal(lam)
        y = np.maximum(0.0, (lam - c_art) / d_art)
        w = np.sum(pl / (upp - xs) + ql / (xs - low))
        w += float(np.sum(c_art * y + 0.5 * d_art * y**2 - lam * y)) - float(lam @ b)
        grad = np.array([np.sum(p / (upp - xs) + q / (xs - low)) for p, q in pq]) - b - y
        return -w, -grad

    res = minimize(negative_dual, np.ones(len(grads)), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * len(grads), options={"maxiter": 500, "gtol": 1e-14})  # fmt: skip
    xs, _pl, _ql = primal(res.x)
    return xs, res.x
