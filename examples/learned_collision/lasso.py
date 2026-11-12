"""The linear correction's coefficients by a Gauss-Newton Lasso (2026-11-11): the whole L1
path, its cross-validation and the stability of every term, instead of one training run per
penalty.

Around the current coefficients ``A0`` (``f64[2, 9]``: ``(dlog r_s, dlog r_q)`` x the
standardised features, :mod:`.features`) each window's projected flux deviation is linear to
first order, ``d(A) ~ d(A0) + J (A - A0)`` with ``J = dd/dA`` by forward-mode AD (18 tangents
through the window).  Averaged over windows per case (a wave's deviations in units of its own
decaying flux, one row per mode; a shock's per window time and coarse cell), the squared bias
is a quadratic in ``A``, and

    min_A  1/2 sum_cases mean_rows w (dbar + Jbar (A - A0))^2 + alpha |A|_1

is a Lasso (coordinate descent on the 18 x 18 Gram matrix; the path over ``alpha`` warm
started).  ``--outer`` Gauss-Newton iterations re-linearise at the chosen fit.

**Cross-validation** of ``alpha`` (the predicted squared bias of the held-out part):
leave-one-case-out (generalisation across flows) and split-seed (the windows of half the
reference seeds against the other half: the noise the fit absorbs).  The chosen ``alpha`` is
the largest within one standard error of the minimum.  **Stability selection**: the fit at the
chosen ``alpha`` on ``--boot`` bootstrap resamples of the windows -- the fraction of
resamples keeping each term.  ``--signs``: the cell-size and time-step terms of the
``dlog r_s`` row held at ``>= 0`` (both errors raise the transport, so the stress must relax
faster) -- a bound that binds is a finding, not a fit.

    PYTHONPATH=. python3 -m examples.learned_collision.lasso [--windows 96] [--outer 3]

Writes ``examples/learned_collision/results/lasso_<tag>.json`` (paths, CV curves,
stabilities per outer iteration) and ``train_linear_lasso_<tag>.json`` (the final fit, as a
:mod:`.train` result for :mod:`.evaluate` / :mod:`.circle`).
"""

from __future__ import annotations

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import operators
from examples.learned_collision import windows2d as w2
from examples.learned_collision.shear_wave import BLOCK, RESULTS
from examples.learned_collision.train import TRAIN_CASES, CaseRunner, model_options, parser
from jax_dsmc.collide import relax

POSITIVE_RATE = ("dx2", "dx4", "dt2", "dt1", "dx2_inv_n")
"""Terms whose ``dlog r_s`` coefficient ``--signs`` holds at ``>= 0``."""


def _deviation(a, theta, states, cstates, keys, target, case, collide_fn, blocks, step0):
    """The two replicas' mean projected deviation ``[T, K, F]`` as a function of ``A``."""
    th = dict(theta, A=a)
    run = functools.partial(w2.rollout, th, case=case, collide_fn=collide_fn, blocks=blocks,
                            step0=step0)  # fmt: skip
    _, _, ms = jax.vmap(run)(states, cstates, keys)
    ref = w2.flux(target, case.kind)
    d = [w2.project(w2.flux(ms[r], case.kind) - ref, case) for r in range(2)]
    return 0.5 * (d[0] + d[1])


def _dev_and_jac(a, theta, states, cstates, keys, target, case, collide_fn, blocks, step0):
    """One window's deviation and its forward-mode Jacobian in ``A``."""
    f = functools.partial(_deviation, theta=theta, states=states, cstates=cstates, keys=keys,
                          target=target, case=case, collide_fn=collide_fn, blocks=blocks,
                          step0=step0)  # fmt: skip
    return f(a), jax.jacfwd(f)(a)


class Linearised:
    """One case's windows at ``A0``: per window the deviation and its Jacobian, reduced to
    regression rows (``[n_windows, rows, F]`` and ``[n_windows, rows, F, 18]``)."""

    def __init__(self, runner, theta, n_windows, rng, batch=8):
        case = runner.case
        dev_jac = functools.partial(_dev_and_jac, case=case, collide_fn=runner.collide_fn,
                                    blocks=runner.blocks, step0=runner.burn * BLOCK)  # fmt: skip
        both = jax.jit(jax.vmap(dev_jac, in_axes=(None, None, 0, 0, 0, 0)))
        ys, xs, seeds = [], [], []
        while n_windows > 0:
            m = min(n_windows, batch)
            n_windows -= m
            states, cstates, keys, targets, rows, seed = runner.draw_batch(rng, m)
            if runner.burn:
                states, cstates = runner.burn_fn(theta, states, cstates, keys)
            y, x = both(theta["A"], theta, states, cstates, keys, targets)
            y = np.asarray(y)  # [m, T, K, F]
            x = np.asarray(x).reshape(*y.shape, -1)  # [m, T, K, F, 18]
            for i in range(m):
                yi, xi = y[i], x[i]
                if runner.scale is not None:  # a wave: in units of its flux, time-averaged
                    s = runner.scale[rows[i]]
                    yi, xi = np.mean(yi / s, axis=0), np.mean(xi / s[..., None], axis=0)
                ys.append(yi.reshape(-1, yi.shape[-1]))
                xs.append(xi.reshape(-1, *xi.shape[-2:]))
            seeds += seed
        self.name, self.y, self.x, self.seeds = runner.name, np.stack(ys), np.stack(xs), np.array(
            seeds)  # fmt: skip
        # a shock's rows are absolute (weighted by the target's mean square); a wave's are
        # already relative to its own flux
        self.weights = np.ones_like(runner.weights) if runner.scale is not None else np.asarray(
            runner.weights)  # fmt: skip
        self.a0 = np.asarray(theta["A"]).ravel()

    def quadratic(self, idx=None):
        """``(G, c, e0)`` of ``1/2 mean_rows w (ybar + Xbar (a - a0))^2 = 1/2 a G a - c a +
        e0`` over windows ``idx`` (default all)."""
        idx = slice(None) if idx is None else idx
        y = self.y[idx].mean(axis=0)  # [rows, F]
        x = self.x[idx].mean(axis=0)  # [rows, F, 18]
        w = self.weights[None, :]
        n = y.shape[0]
        t = y - np.einsum("rfp,p->rf", x, self.a0)  # residual at a = 0 (linearised)
        g = np.einsum("rf,rfp,rfq->pq", w, x, x) / n
        c = -np.einsum("rf,rf,rfp->p", w, t, x) / n
        e0 = 0.5 * np.sum(w * t**2) / n
        return g, c, e0


def lasso_cd(g, c, alpha, a=None, bounds=None, sweeps=500, tol=1e-10):
    """Coordinate descent on ``1/2 a G a - c a + alpha |a|_1`` (``bounds``: ``[2, p]``
    per-coefficient lower and upper bounds, ``-inf`` / ``inf`` for none)."""
    p = len(c)
    a = np.zeros(p) if a is None else a.copy()
    lower, upper = (np.full(p, -np.inf), np.full(p, np.inf)) if bounds is None else bounds
    for _ in range(sweeps):
        change = 0.0
        for j in range(p):
            if g[j, j] <= 0.0:
                continue
            rho = c[j] - g[j] @ a + g[j, j] * a[j]
            new = np.sign(rho) * max(abs(rho) - alpha, 0.0) / g[j, j]
            new = min(max(new, lower[j]), upper[j])
            change = max(change, abs(new - a[j]))
            a[j] = new
        if change < tol:
            break
    return a


def _combine(parts):
    g = sum(p[0] for p in parts) / len(parts)
    c = sum(p[1] for p in parts) / len(parts)
    e = sum(p[2] for p in parts) / len(parts)
    return g, c, e


def _objective(a, g, c, e0):
    return float(0.5 * a @ g @ a - c @ a + e0)


def path(g, c, alphas, bounds):
    out, a = [], None
    for alpha in alphas:
        a = lasso_cd(g, c, alpha, a, bounds)
        out.append(a.copy())
    return np.array(out)


def _cv(train_parts, test_parts, alphas, bounds):
    """Held-out predicted squared bias along the path fitted on ``train_parts``."""
    g, c, _ = _combine(train_parts)
    coefs = path(g, c, alphas, bounds)
    gt, ct, et = _combine(test_parts)
    return np.array([_objective(a, gt, ct, et) for a in coefs])


def cross_validate(cases, alphas, bounds):
    """``{"loco": [folds, alphas], "seed": [2, alphas]}`` held-out errors."""
    loco = [_cv([c.quadratic() for c in cases if c is not h], [h.quadratic()], alphas, bounds)
            for h in cases]  # fmt: skip
    seed = []
    for half in (0, 1):
        tr_idx = [np.nonzero((c.seeds % 2) != half)[0] for c in cases]
        te_idx = [np.nonzero((c.seeds % 2) == half)[0] for c in cases]
        if any(len(i) == 0 for i in tr_idx + te_idx):
            continue  # a case with every window on one side of the split
        tr = [c.quadratic(i) for c, i in zip(cases, tr_idx, strict=True)]
        te = [c.quadratic(i) for c, i in zip(cases, te_idx, strict=True)]
        seed.append(_cv(tr, te, alphas, bounds))
    return {"loco": np.array(loco), "seed": np.array(seed).reshape(-1, len(alphas))}


def one_se(errors, alphas):
    """The largest ``alpha`` within one standard error (over folds) of the minimum mean."""
    if errors.shape[0] < 2:
        raise ValueError("cross-validation needs two or more folds")
    mean = errors.mean(axis=0)
    se = errors.std(axis=0, ddof=1) / np.sqrt(errors.shape[0])
    best = int(np.argmin(mean))
    ok = np.nonzero(mean <= mean[best] + se[best])[0]
    return float(alphas[ok.min()]), float(alphas[best])  # alphas descending: min index = largest


def stability(cases, alpha, bounds, n_boot, rng):
    """Fraction of window-bootstrap refits keeping each coefficient at ``alpha``."""
    kept = np.zeros(len(cases[0].a0))
    for _ in range(n_boot):
        parts = [c.quadratic(rng.integers(0, len(c.y), len(c.y))) for c in cases]
        g, c, _ = _combine(parts)
        kept += np.abs(lasso_cd(g, c, alpha, bounds=bounds)) > 1e-12
    return kept / n_boot


def _bounds(signs):
    """``[2, p]``: lower and upper bounds of the flattened ``A`` (``--signs``)."""
    lower = np.full((2, len(relax.FEATURES)), -np.inf)
    if signs:
        for name in POSITIVE_RATE:
            lower[0, relax.FEATURES.index(name)] = 0.0
    return np.stack([lower.ravel(), np.full(lower.size, np.inf)])


def main(argv=None):
    ap = parser()
    ap.description = __doc__.split("\n\n")[0]
    ap.set_defaults(model="linear", cases=list(TRAIN_CASES))
    ap.add_argument("--windows", type=int, default=2048, help="windows per case per outer step")
    ap.add_argument("--jac-batch", type=int, default=8, help="windows per Jacobian call")
    ap.add_argument("--outer", type=int, default=3, help="Gauss-Newton re-linearisations")
    ap.add_argument("--alphas", type=int, default=40)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--signs", action="store_true")
    ap.add_argument("--cv", default="loco", choices=("loco", "seed"), help="picks alpha")
    args = ap.parse_args(argv)
    if args.warm_pool:
        raise SystemExit("lasso draws snapshot windows only (no --warm-pool chains)")
    _, theta, _ = operators.operator("linear", "shear", **model_options(args))
    runners = [CaseRunner(name, "linear", args) for name in args.cases]
    rng = np.random.default_rng(args.seed)
    bounds = _bounds(args.signs)
    record, t0 = [], time.time()
    for it in range(args.outer):
        cases = [Linearised(r, theta, args.windows, rng, args.jac_batch) for r in runners]
        g, c, e0 = _combine([k.quadratic() for k in cases])
        alpha_max = float(np.max(np.abs(c))) * 1.01
        alphas = alpha_max * np.logspace(0, -4, args.alphas)
        coefs = path(g, c, alphas, bounds)
        cv = cross_validate(cases, alphas, bounds)
        alpha, alpha_min = one_se(cv[args.cv], alphas)
        a = lasso_cd(g, c, alpha, bounds=bounds)
        stab = stability(cases, alpha, bounds, args.boot, rng)
        record.append({
            "outer": it, "a0": cases[0].a0.tolist(), "alphas": alphas.tolist(),
            "path": coefs.tolist(), "cv_loco": cv["loco"].tolist(), "cv_seed": cv["seed"].tolist(),
            "alpha": alpha, "alpha_min": alpha_min, "coef": a.tolist(), "stability": stab.tolist(),
            "bias2_at_a0": e0, "bias2_fit": _objective(a, g, c, e0),
            "bound_binding": (np.isfinite(bounds) & (np.abs(a - bounds) < 1e-12)).any(0).tolist(),
            "seconds": time.time() - t0,
        })  # fmt: skip
        names = [f"{row}:{f}" for row in ("rs", "rq") for f in relax.FEATURES]
        kept = [f"{n} {v:+.3f} ({s:.2f})" for n, v, s in zip(names, a, stab, strict=True)
                if abs(v) > 1e-12]  # fmt: skip
        print(f"outer {it}: bias2 {e0:.4e} -> {record[-1]['bias2_fit']:.4e}, alpha {alpha:.3e} "
              f"(min {alpha_min:.3e}), kept {len(kept)}: {', '.join(kept)}  "
              f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
        theta = dict(theta, A=jnp.asarray(a.reshape(2, -1)))
        (RESULTS / f"lasso_{args.tag}.json").write_text(
            json.dumps({"args": vars(args), "outer": record})
        )
        result = {
            "args": dict(vars(args), model="linear"),
            "theta": {k: np.asarray(v).tolist() for k, v in theta.items()},
            "log": [],
        }
        (RESULTS / f"train_linear_lasso_{args.tag}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
