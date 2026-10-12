"""Offline fit study on squash snapshots (snapshot.py): 5-fold cross-validated prediction of
the carried value at held-out nodes, per fit method, on the real field.

Every method sees the same folds: the held-out fifth of each cell's nodes is masked out of
the node tables and predicted from the rest.  CV MSE = E (fit - truth)^2 + sigma^2 (the
node values' own noise, common to all methods), so DIFFERENCES between methods are their
differences in squared prediction error; they are reported paired (per node) against the
first method, with the per-node se.  Also reported: the mean held-out residual (a bias
indicator) over all nodes and over the nodes of cells next to the body (low-occupancy cells
and cut cells are where fits differ most), and the fit time.

    PYTHONPATH=. python3 scratch/cellfit/cv.py SNAP.npz OUT.json [column lam|lamt] [steps]
"""

import functools
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import cell_fit, ensemble_device

FOLDS = 5
METHODS = (
    "knn:32",
    "knn:64",
    "knn:all",
    "cell:mean",
    "cell:lin",
    "cell:quad",
    "cell:quad:adapt1.5",
    "cell:quadfull",
    "cell:quadfull:adapt1.5",
    "cell:cubicv",
    "cell:cubicv:adapt1.5",
    "cell:cubicv:loo",
    "cell:cubic:adapt2",
    "cell:cubic:loo",
    "cell:quad:pool1",
    "cell:quadfull:pool1",
    "cell:cubicv:pool1",
    "cell:cubic:pool1:adapt2",
    "cell:cubic:pool1:loo",
    "knn:32:sided",
    "knn:64:sided",
    "cell:quad:split",
    "cell:quad:adapt1.5:split",
    "cell:quad:align0.9:split",
    "cell:quad:own:split",
    "cell:quad:own:align0.9:split",
    "cell:quad:pool1:split",
)
"""``:split`` methods are the library's split fit (``cell_fit.fit_split`` with the nodes'
normals; ``align`` with the per-query one-sided fallback in complex cells), compared with
the per-query one-sided fit ``knn:<k>:sided``; every other method is two-sided."""


def tables_of(x, v, lam, cell, alive, n_cells):
    """Per-cell node tables (C, P, .) and each node's (cell, rank)."""
    c = np.where(alive, cell, n_cells)
    order = np.argsort(c, kind="stable")
    counts = np.bincount(c[alive], minlength=n_cells)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    p = int(-(-counts.max() // 64) * 64)  # the backward's NPC_BUCKET
    rank = np.empty(len(c), np.int64)
    rank[order] = np.arange(len(c)) - np.concatenate((starts, [counts.sum()]))[c[order]]
    ids = np.zeros((n_cells, p), np.int64)
    valid = np.zeros((n_cells, p), bool)
    a = np.flatnonzero(alive)
    ids[c[a], rank[a]] = a
    valid[c[a], rank[a]] = True
    return ids, valid, rank, counts, p


def widen(t, nbr, exists):
    out = t[nbr]  # (C, B, P, ...)
    return out.reshape(out.shape[0], -1, *out.shape[3:])


def knn(k, tab, train, q_c, q_x, q_v, scale, direction):
    node_x, node_v, node_val, _ = tab
    k = node_x.shape[1] if k == "all" else min(int(k), node_x.shape[1])
    return ensemble_device.cell_query(
        (node_x, node_v, train, node_val), q_c, q_x, q_v, scale, k, "quadratic_full", 4096,
        direction=direction,
    )


def predict(method, tab, train, q_c, q_x, q_v, scale, nbr, exists, q_normal):
    node_x, node_v, node_val, node_normal = tab
    direction = jnp.sign(jnp.sum(q_v * q_normal, axis=1))[:, None] * q_normal
    if method.startswith("knn:"):
        side = direction if method.endswith(":sided") else None
        return knn(method.split(":")[1], tab, train, q_c, q_x, q_v, scale, side)
    split = method.endswith(":split")
    method = method.removesuffix(":split")
    spec = cell_fit.parse(method)
    own = {}
    tables = (node_x, node_v, train, node_val) + ((node_normal,) if split else ())
    if spec.pool:
        valid = (train[nbr] & exists[:, :, None]).reshape(train.shape[0], -1)
        tables = tuple(widen(t, nbr, exists) for t in tables)
        tables = (tables[0], tables[1], valid, *tables[3:])
        own = {"own_x": node_x, "own_valid": train}
    if not split:
        field = cell_fit.fit(*tables, scale, method, **own)
        return cell_fit.evaluate(field, q_c, q_x, q_v, scale, method)
    normal, complex_ = cell_fit.cell_normals(
        jnp.where(train[..., None], node_normal, 0.0), train, spec.align
    )
    field = cell_fit.fit_split(*tables[:4], tables[4], normal, complex_, scale, method, **own)
    value = cell_fit.evaluate(field, q_c, q_x, q_v, scale, method, q_normal=q_normal)
    if spec.align > 0.0 and spec.fallback:
        fallback = knn(32, tab, train, q_c, q_x, q_v, scale, direction)
        value = jnp.where(complex_[q_c], fallback, value)
    return value


@functools.partial(jax.jit, static_argnames=("method",))
def fold_predict(method, tab, train, q_c, q_x, q_v, scale, nbr, exists, q_normal):
    return predict(method, tab, train, q_c, q_x, q_v, scale, nbr, exists, q_normal)


def main():
    snap = np.load(sys.argv[1])
    out_path = sys.argv[2]
    column = sys.argv[3] if len(sys.argv) > 3 else "lam"
    steps = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 120, 240]
    scale = jnp.asarray(snap["scale"])
    n_cells = int(snap["n_cells"])
    nbr, exists = jnp.asarray(snap["nbr1"]), jnp.asarray(snap["exists1"])
    rng = np.random.default_rng(0)
    res = {m: {"sq": [], "resid": [], "seconds": 0.0} for m in METHODS}
    small, near = [], []
    for k in steps:
        x, v, lam = snap[f"x{k}"], snap[f"v{k}"], snap[f"{column}{k}"]
        alive, cell = snap[f"alive{k}"], snap[f"cell{k}"]
        normal = snap[f"normal{k}"].astype(np.float64)
        ids, valid, _, occ, _ = tables_of(x, v, lam, cell, alive, n_cells)
        tab = tuple(jnp.asarray(t[ids]) for t in (x, v, lam, normal))
        fold = rng.integers(0, FOLDS, size=len(x))
        a = np.flatnonzero(alive)
        small.append(occ[cell[a]] < 20)
        near.append(np.any(normal[a] != 0.0, axis=1))
        print(f"step {k}: {len(a)} nodes, {np.sum(occ > 0)} occupied cells, "
              f"mean occupancy {occ[occ > 0].mean():.1f}, near-surface nodes "
              f"{near[-1].mean():.1%}", flush=True)
        for m in METHODS:
            pred = np.zeros(len(x))
            t0 = time.time()
            for f in range(FOLDS):
                train = jnp.asarray(valid & (fold[ids] != f))
                held = a[fold[a] == f]
                got = fold_predict(m, tab, train, jnp.asarray(cell[held]),
                                   jnp.asarray(x[held]), jnp.asarray(v[held]), scale, nbr,
                                   exists, jnp.asarray(normal[held]))
                pred[held] = np.asarray(got)
            res[m]["seconds"] += time.time() - t0
            r = lam[a] - pred[a]
            res[m]["sq"].append(r**2)
            res[m]["resid"].append(r)
            print(f"  {m:26s} cv mse {np.mean(r**2):.4e}  mean resid {np.mean(r):+.2e}  "
                  f"({time.time() - t0:.1f} s)", flush=True)
    small = np.concatenate(small)
    near = np.concatenate(near)
    base = np.concatenate(res[METHODS[0]]["sq"])
    summary = {}
    for m in METHODS:
        sq = np.concatenate(res[m]["sq"])
        rs = np.concatenate(res[m]["resid"])
        d = sq - base
        summary[m] = {
            "cv_mse": float(sq.mean()),
            "vs_first": float(d.mean()),
            "vs_first_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "mean_resid": float(rs.mean()),
            "mean_resid_se": float(rs.std(ddof=1) / np.sqrt(len(rs))),
            "cv_mse_small_cells": float(sq[small].mean()),
            "cv_mse_near_surface": float(sq[near].mean()),
            "near_vs_first": float(d[near].mean()),
            "near_vs_first_se": float(d[near].std(ddof=1) / np.sqrt(near.sum())),
            "seconds": res[m]["seconds"],
        }
    var = float(np.concatenate([snap[f"{column}{k}"][snap[f"alive{k}"]] for k in steps]).var())
    print(f"\nvalue variance {var:.4e}; nodes in cells < 20: {small.mean():.1%}; "
          f"near-surface nodes {near.mean():.1%}")
    print(f"{'method':26s} {'cv mse':>10s} {'- ' + METHODS[0]:>22s} {'mean resid':>18s} "
          f"{'mse small cells':>15s} {'near-surface - first':>24s} {'fit s':>6s}")
    for m, s in summary.items():
        print(f"{m:26s} {s['cv_mse']:10.4e} {s['vs_first']:+10.3e} +- {s['vs_first_se']:.1e} "
              f"{s['mean_resid']:+.2e} +- {s['mean_resid_se']:.0e} {s['cv_mse_small_cells']:15.4e}"
              f" {s['near_vs_first']:+10.3e} +- {s['near_vs_first_se']:.1e} {s['seconds']:6.1f}")
    with open(out_path, "w") as fh:
        json.dump({"column": column, "steps": steps, "variance": var, "summary": summary}, fh,
                  indent=1)


if __name__ == "__main__":
    main()
