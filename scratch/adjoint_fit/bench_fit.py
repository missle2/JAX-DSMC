"""ensemble_device.field_query at the backward's sizes (Q = 4096 queries per chunk, P nodes
per cell, k = 32, D = 2): time per chunk by basis, and solver variants of the full
quadratic with their agreement to the current LU result.

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/adjoint_fit/bench_fit.py
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

from jax_dsmc import ensemble_device as ED  # noqa: E402

Q, P, K, D = 4096, 256, 32, 2


def data(seed=0):
    rng = np.random.default_rng(seed)
    nx = rng.random((Q, P, D)) * 0.5
    nv = rng.normal(size=(Q, P, 3)) * 300.0
    valid = rng.random((Q, P)) < 0.6
    val = 1e-3 * (nx[..., 0] + (nv[..., 0] / 300) ** 2 + 0.3 * nv[..., 1] / 300) + 1e-5 * rng.normal(size=(Q, P))
    qx = rng.random((Q, D)) * 0.5
    qv = rng.normal(size=(Q, 3)) * 300.0
    scale = np.array([0.5, 0.5, 300.0, 300.0, 300.0])
    return [jnp.asarray(a) for a in (nx, nv, valid, val, qx, qv, scale)]


def fit_parts(node_x, node_v, node_valid, node_val, q_x, q_v, scale, k, basis):
    """field_query's front half: (design, weights, values, good) of the k nearest."""
    pts = jnp.concatenate((node_x, node_v), axis=-1) / scale
    q = jnp.concatenate((q_x, q_v), axis=-1) / scale
    d2 = jnp.sum((pts - q[:, None, :]) ** 2, axis=-1)
    d2 = jnp.where(node_valid, d2, jnp.inf)
    neg, ids = lax.top_k(-d2, k)
    dist = jnp.sqrt(jnp.maximum(-neg, 0.0))
    good = jnp.isfinite(dist)
    dk = jnp.max(jnp.where(good, dist, 0.0), axis=1, keepdims=True)
    w = jnp.where(good, 1.0 / (1.0 + (dist / jnp.maximum(dk, 1e-12)) ** 2), 0.0)
    chosen = jnp.take_along_axis(pts, ids[..., None], axis=1)
    values = jnp.where(good, jnp.take_along_axis(node_val, ids, axis=1), 0.0)
    design = jnp.where(good[..., None], ED._design(chosen - q[:, None, :], basis), 0.0)
    return design, w, values


def normal(design, w, values, ridge_rel=1e-8):
    m = jnp.einsum("qki,qk,qkj->qij", design, w, design)
    p = design.shape[-1]
    ridge = jnp.full(p, ridge_rel).at[0].set(0.0)
    m = m + ridge[None] * (jnp.trace(m, axis1=1, axis2=2) / p)[:, None, None] * jnp.eye(p)
    rhs = jnp.einsum("qki,qk,qk->qi", design, w, values)
    return m, rhs


def solve_lu(m, rhs):
    return jnp.linalg.solve(m, rhs[..., None])[..., 0]


def solve_cho(m, rhs):
    c = jax.scipy.linalg.cho_factor(m, lower=True)
    return jax.scipy.linalg.cho_solve(c, rhs)


def solve_unrolled(m, rhs):
    """Cholesky + two triangular solves written out over the (static, small) dimension:
    every operation is elementwise over the Q queries."""
    p = m.shape[-1]
    L = [[None] * p for _ in range(p)]
    for j in range(p):
        s = m[:, j, j] - sum(L[j][t] * L[j][t] for t in range(j))
        L[j][j] = jnp.sqrt(s)
        inv = 1.0 / L[j][j]
        for i in range(j + 1, p):
            L[i][j] = (m[:, i, j] - sum(L[i][t] * L[j][t] for t in range(j))) * inv
    y = []
    for i in range(p):
        y.append((rhs[:, i] - sum(L[i][t] * y[t] for t in range(i))) / L[i][i])
    x = [None] * p
    for i in reversed(range(p)):
        x[i] = (y[i] - sum(L[t][i] * x[t] for t in range(i + 1, p))) / L[i][i]
    return jnp.stack(x, axis=1)


def normal_sqrtw(design, w, values, ridge_rel=1e-8):
    """The same normal equations from the sqrt(w)-weighted design: a two-operand batched
    matmul."""
    a = design * jnp.sqrt(w)[..., None]
    m = jnp.einsum("qki,qkj->qij", a, a)
    p = design.shape[-1]
    ridge = jnp.full(p, ridge_rel).at[0].set(0.0)
    m = m + ridge[None] * (jnp.trace(m, axis1=1, axis2=2) / p)[:, None, None] * jnp.eye(p)
    rhs = jnp.einsum("qki,qk->qi", a, values * jnp.sqrt(w))
    return m, rhs


def _ridge(m, ridge_rel=1e-8):
    p = m.shape[-1]
    ridge = jnp.full(p, ridge_rel).at[0].set(0.0)
    return m + ridge[None] * (jnp.trace(m, axis1=1, axis2=2) / p)[:, None, None] * jnp.eye(p)


def normal_reduce(design, w, values):
    """Gram matrix as one broadcast-multiply-and-sum over the k nodes (a reduction
    fusion, no batched matmul)."""
    wd = design * w[..., None]
    m = jnp.sum(wd[:, :, :, None] * design[:, :, None, :], axis=1)
    return _ridge(m), jnp.sum(wd * values[..., None], axis=1)


def normal_triu(design, w, values):
    """Only the upper triangle's p(p+1)/2 products, summed over k, then mirrored."""
    p = design.shape[-1]
    iu, ju = np.triu_indices(p)
    wd = design * w[..., None]
    upper = jnp.sum(wd[:, :, iu] * design[:, :, ju], axis=1)  # (Q, p(p+1)/2)
    m = jnp.zeros((design.shape[0], p, p)).at[:, iu, ju].set(upper).at[:, ju, iu].set(upper)
    return _ridge(m), jnp.sum(wd * values[..., None], axis=1)


def variant(args, basis, assemble, solver, part):
    design, w, values = fit_parts(*args[:7], K, basis)
    if part == "front":
        return design.sum() + w.sum() + values.sum()
    m, rhs = assemble(design, w, values)
    if part == "assemble":
        return m.sum() + rhs.sum()
    return solver(m, rhs)[:, :6]


def timeit(f, args, n=20, reps=5):
    """Best of ``reps`` means over ``n`` calls (the laptop GPU's clocks wander)."""
    jax.block_until_ready(f(*args))
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            r = f(*args)
        jax.block_until_ready(r)
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def main():
    args = data()
    ref = None
    print(f"device {jax.devices()[0].device_kind}; Q {Q}, P {P}, k {K}; ms per chunk")
    for basis in ("quadratic_full", "quadratic", "affine"):
        for name, assemble, solver in (
            ("LU (current)", normal, solve_lu),
            ("Cholesky", normal, solve_cho),
            ("unrolled Cholesky", normal, solve_unrolled),
            ("sqrt(w) matmul + unrolled", normal_sqrtw, solve_unrolled),
            ("reduce-sum + unrolled", normal_reduce, solve_unrolled),
            ("upper triangle + unrolled", normal_triu, solve_unrolled),
        ):
            if basis != "quadratic_full" and name not in ("LU (current)", "upper triangle + unrolled"):
                continue
            times = {}
            for part in ("front", "assemble", "full"):
                f = jax.jit(functools.partial(variant, basis=basis, assemble=assemble, solver=solver, part=part))
                times[part] = timeit(lambda *a: f(a), args)
            out = np.asarray(jax.jit(functools.partial(variant, basis=basis, assemble=assemble, solver=solver, part="full"))(args))
            if basis == "quadratic_full" and ref is None:
                ref = out
            agree = np.max(np.abs(out - ref) / (np.abs(ref) + 1e-30)) if basis == "quadratic_full" else float("nan")
            print(f"{basis:15s} {name:28s} front {times['front']:7.2f}  +assemble {times['assemble'] - times['front']:7.2f}  "
                  f"+solve {times['full'] - times['assemble']:7.2f}  total {times['full']:7.2f}   max rel diff vs LU {agree:.1e}")


if __name__ == "__main__":
    main()
