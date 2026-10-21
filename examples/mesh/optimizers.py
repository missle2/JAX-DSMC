"""Step rules of the level-set optimiser (:mod:`examples.mesh.optimize`) beyond GD / Adam,
2026-10-19: what the drag problem's landscape rewards.

Every rule works on the area-tangent gradient ``g`` (``dJ/dcoef`` without its component along
``grad A``) and its smoothed, band-restricted form ``S g`` (the H1 metric of
:func:`optimize.smoothed_direction`), and returns a coefficient step whose largest entry --
the largest normal displacement, ``|grad phi| = 1`` -- the rule chooses.

* ``tr``: steepest descent in the smoothed metric inside a **trust region** of radius
  ``Delta`` (the largest normal displacement).  Each evaluated trial is compared with its
  base point on the same seeds (common random numbers): per seed ``d_s = J_s(trial) -
  J_s(base)``, the change ``mean(d)`` with standard error ``sd(d) / sqrt(S)`` -- the paired
  difference is far less noisy than either value (:func:`judge`, 2026-10-20 form):

  - **pooled rho**: ``rho = sum(actual) / sum(predicted)`` over the last ``RHO_WINDOW`` (5)
    trials at the current radius, its standard error from the trials' paired errors (with
    one seed per iterate, from the scatter of the window's changes about ``rho`` times the
    prediction; no decision before 3 trials).  2026-10-20 optrules: a single trial's rho was
    noise once the predicted change fell to the seed noise, and noise alone walked the radius
    to its floor.
  - **symmetric radius update**: ``x2`` when ``rho - se > 0.75`` (and the step at the edge),
    ``x0.5`` when ``rho + se < 0.25``, else kept: a rho centred on 0.5 (steepest descent at
    the right length) leaves the radius where it is (the earlier ``x1.5 / x0.5`` drifted it
    down ~6-10 % per iterate).  The window restarts at every change (rho depends on the
    radius).
  - **rejection**: the trial is rejected only when it is worse than its base beyond the
    noise (``mean(d) > se``); the radius halves and the next step starts again from the base
    point (its gradient kept).
  - **at the floor**: a rejection there cannot shrink, so retrying would repeat the identical
    trial on the identical seeds forever (the 2026-10-19 runs' "convergence").  Instead the
    base point is **re-evaluated on new seeds** (``seed_shift`` += S; zero step), a fresh
    objective and gradient, and the run goes on from it.
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
GROW, SHRINK = 2.0, 0.5
RHO_GOOD, RHO_POOR = 0.75, 0.25
RHO_WINDOW = 5  # trials pooled into rho (at the current radius)
RHO_MIN_ONE_SEED = 3  # one seed per iterate: trials before the scatter gives an error
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


def _window(state):
    """The pooled trials ``(actual, predicted, variance)`` at the current radius."""
    if "win_actual" not in state:
        return [], [], []
    return [list(np.atleast_1d(np.asarray(state[k], float))) for k in
            ("win_actual", "win_pred", "win_var")]  # fmt: skip


def window_state(window):
    actual, pred, var = window
    return {"win_actual": np.asarray(actual, float), "win_pred": np.asarray(pred, float),
            "win_var": np.asarray(var, float)}  # fmt: skip


def pooled_rho(actual, pred, var):
    """``(rho, se of rho, se of the newest trial's change)`` of the window.  ``var``: each
    trial's paired-difference variance of the mean (NaN with one seed: then the window's
    scatter about ``rho * predicted`` stands in, from ``RHO_MIN_ONE_SEED`` trials on)."""
    a, p, v = (np.asarray(x, float) for x in (actual, pred, var))
    rho = float(a.sum() / p.sum())
    if np.isnan(v).any():
        n = a.size
        if n < RHO_MIN_ONE_SEED:
            return rho, float("nan"), float("nan")
        v = np.full(n, float(np.sum((a - rho * p) ** 2)) / (n - 1))
    return rho, float(np.sqrt(v.sum()) / abs(p.sum())), float(np.sqrt(v[-1]))


def judge(state, objectives, step):
    """Judge the trial just evaluated (per-seed ``objectives``) against its base point (module
    docstring, ``tr``).  Returns a dict: ``accepted``, ``rebase`` (rejected at the floor: the
    base point is evaluated again on new seeds), ``radius`` (for the next step), the trial's
    own ``rho``, the window's ``rho_pooled`` / ``rho_se`` / ``window`` (lists, at the
    returned radius).  The first evaluation, and a base re-evaluated on new seeds, is its
    own base (accepted, no judgement)."""
    radius = float(state.get("radius", step))
    window = _window(state)
    out = {"accepted": True, "rebase": False, "radius": radius, "rho": float("nan"),
           "rho_pooled": float("nan"), "rho_se": float("nan"), "window": window}  # fmt: skip
    if "base_objectives" not in state or bool(state.get("rebase", False)):
        return out
    lo, hi = radius_bounds(step)
    d = np.asarray(objectives, float) - np.asarray(state["base_objectives"], float)
    actual, predicted = float(d.mean()), float(state["predicted"])
    var = float(d.var(ddof=1) / d.size) if d.size > 1 else float("nan")
    actual_w, pred_w, var_w = (x[-(RHO_WINDOW - 1) :] for x in window)
    window = [[*actual_w, actual], [*pred_w, predicted], [*var_w, var]]
    rho_pooled, rho_se, se_now = pooled_rho(*window)
    # worse than the base beyond the noise (NaN: no error yet -> not rejected)
    accepted = not actual > se_now
    if not accepted:
        new = max(lo, radius * SHRINK)
        out["rebase"] = radius <= lo
    elif rho_pooled + rho_se < RHO_POOR:
        new = max(lo, radius * SHRINK)
    elif rho_pooled - rho_se > RHO_GOOD and bool(state.get("at_edge", True)):
        new = min(hi, radius * GROW)
    else:
        new = radius
    if new != radius:
        window = [[], [], []]  # rho depends on the radius: pool again from here
    out.update(accepted=accepted, radius=new, rho=actual / predicted, rho_pooled=rho_pooled,
               rho_se=rho_se, window=window)  # fmt: skip
    return out


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
