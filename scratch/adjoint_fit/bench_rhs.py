"""Gate for one backward carrying R value fields: the fit of R columns at once (one k-nearest
search, one normal matrix and factorisation, R right-hand sides) against R separate fits,
ensemble_device.field_query's arithmetic at the backward's sizes.
    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/adjoint_fit/bench_rhs.py"""
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/adjoint_fit")
import bench_fit as BF  # noqa: E402  (data, fit_parts, normal, timeit)

jax.config.update("jax_enable_x64", True)


def fit_columns(args, values_r, fast=False):
    """field_query's fit with node values ``(Q, P, R)``: returns ``beta (Q, p, R)`` -- one
    k-nearest search, one design and normal matrix, R right-hand sides."""
    node_x, node_v, valid, _val, q_x, q_v, scale = args
    pts = jnp.concatenate((node_x, node_v), axis=-1) / scale
    q = jnp.concatenate((q_x, q_v), axis=-1) / scale
    d2 = jnp.where(valid, jnp.sum((pts - q[:, None, :]) ** 2, axis=-1), jnp.inf)
    neg, ids = jax.lax.top_k(-d2, BF.K)
    dist = jnp.sqrt(jnp.maximum(-neg, 0.0))
    good = jnp.isfinite(dist)
    dk = jnp.max(jnp.where(good, dist, 0.0), axis=1, keepdims=True)
    w = jnp.where(good, 1.0 / (1.0 + (dist / jnp.maximum(dk, 1e-12)) ** 2), 0.0)
    chosen = jnp.take_along_axis(pts, ids[..., None], axis=1)
    design = jnp.where(good[..., None], BF.ED._design(chosen - q[:, None, :], "quadratic_full"), 0.0)
    vals = jnp.where(good[..., None], jnp.take_along_axis(values_r, ids[..., None], axis=1), 0.0)
    m, _ = BF.normal(design, w, vals[..., 0])
    if not fast:
        rhs = jnp.einsum("qki,qk,qkr->qir", design, w, vals)
        return jnp.linalg.solve(m, rhs)
    rhs = jnp.sum((design * w[..., None])[..., None] * vals[:, :, None, :], axis=1)  # (Q, p, R)
    return cholesky_solve(m, rhs)


def cholesky_solve(m, rhs):
    """Unrolled Cholesky of the (Q, p, p) normal matrices, then both triangular solves for
    every column of ``rhs`` (Q, p, R): the factorisation is shared by the R fields."""
    p = m.shape[-1]
    L = [[None] * p for _ in range(p)]
    for j in range(p):
        L[j][j] = jnp.sqrt(m[:, j, j] - sum(L[j][t] * L[j][t] for t in range(j)))
        for i in range(j + 1, p):
            L[i][j] = (m[:, i, j] - sum(L[i][t] * L[j][t] for t in range(j))) / L[j][j]
    y = []
    for i in range(p):
        y.append((rhs[:, i] - sum(L[i][t][:, None] * y[t] for t in range(i))) / L[i][i][:, None])
    x = [None] * p
    for i in reversed(range(p)):
        x[i] = (y[i] - sum(L[t][i][:, None] * x[t] for t in range(i + 1, p))) / L[i][i][:, None]
    return jnp.stack(x, axis=1)


args = BF.data()
rng = np.random.default_rng(1)
ref = None
for fast in (False, True):
    for r in (1, 2):
        # smooth fields (a quadratic in v plus a little noise), as the adjoint's are
        base = np.asarray(args[3])
        vals = jnp.asarray(np.stack([base * (1 + 0.5 * c) + 1e-6 * rng.normal(size=base.shape) for c in range(r)], -1))
        f = jax.jit(functools.partial(fit_columns, fast=fast))
        t = BF.timeit(f, (args, vals))
        out = np.asarray(f(args, vals))[:, :6]  # value and slopes: what the backward reads
        if r == 2 and not fast:
            ref = out
        diff = np.max(np.abs(out - ref) / np.abs(ref).max(axis=0)) if (r == 2 and fast) else float("nan")
        name = "sum + shared Cholesky" if fast else "einsum + LU (current)"
        print(f"{name:22s} R = {r}: {t:.2f} ms per chunk ({t / r:.2f} per field)  diff vs LU {diff:.1e}")
