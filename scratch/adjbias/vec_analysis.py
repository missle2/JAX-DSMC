"""Offline analysis of the adjm10 ``*_vec`` term vectors (2026-10-25).

On the M 10 Adam body the adjoint's translation number flips sign between two translation
discretisations (terms.py's interpolated ``-d_x phi`` vs optimize.translation_modes'
coefficient central difference).  Here, per term and seed:

1. projections on several translations: both discretisations, the exact first-order one
   (``(translate(ls, eps) - ls) / eps``, unmasked) and the 1 m resample the paired FD uses;
2. the split of the coefficient gradient ``g`` into the part a surface normal-velocity
   density can produce (``g_row = M g_n``, ``M = -B^T diag(w / |grad phi|)``, B the spline
   basis at surface quadrature points) and the rest (``g_null``: coefficient directions that
   do not move the zero set), and the unambiguous surface translation number
   ``int g_n n_x ds``;
3. where along the surface (by x) the translation components sit.

    PYTHONPATH=. python3 scratch/adjbias/vec_analysis.py [names...]
        (default adjm10_body_vec adjm10_body_fm_vec)
"""

import dataclasses
import json
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

sys.path.insert(0, "scratch/adjbias")
import terms  # noqa: E402, F401  (translation, checked against the saved one)

from examples import ensemble_mesh as EM  # noqa: E402

TERMS = ("pathwise_x", "pathwise_v", "pathwise_tally", "timing", "order", "volume")


def basis_matrix(ls, x):
    """Sparse ``B`` with ``phi(x) = B @ coef.ravel()`` (cubic B-spline, 2-D)."""
    shape = ls.coef.shape
    u = (np.asarray(x) - np.asarray(ls.lo)) / ls.h
    i = np.floor(u).astype(int)
    f = u - i
    g = 1.0 - f
    w = np.stack([g**3 / 6, (3 * f**3 - 6 * f**2 + 4) / 6, (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6,
                  f**3 / 6], axis=-1)  # fmt: skip  # [N, 2, 4]
    off = np.arange(-1, 3)
    ix = i[:, 0, None] + off  # [N, 4]
    iy = i[:, 1, None] + off
    rows = np.repeat(np.arange(len(x)), 16)
    cols = (ix[:, :, None] * shape[1] + iy[:, None, :]).reshape(-1)
    vals = (w[:, 0, :, None] * w[:, 1, None, :]).reshape(-1)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(x), shape[0] * shape[1]))


def surface_quadrature(ls, lo, hi, spacing):
    """Midpoints, lengths and unit normals of a fine polyline of the zero set."""
    n = int(max(hi - lo) / spacing)
    segs = O.surface_polyline(ls, lo, hi, n)
    mids, lens = [], []
    for s in segs:
        mids.append(0.5 * (s[1:] + s[:-1]))
        lens.append(np.linalg.norm(s[1:] - s[:-1], axis=1))
    x = np.asarray(O.L.project(ls, jnp.asarray(np.concatenate(mids))))
    _phi, grad = O.L.value_and_grad(ls, jnp.asarray(x))
    grad = np.asarray(grad)
    gn = np.linalg.norm(grad, axis=1)
    return x, np.concatenate(lens), grad / gn[:, None], gn


def main():
    names = sys.argv[1:] or ["adjm10_body_vec", "adjm10_body_fm_vec"]
    for name in names:
        d = np.load(f"scratch/adjbias/cluster/{name}_vectors.npz")
        # the job's own config: the knot spacing and grid origin come from it
        cfg = dict(
            EM.default_config(), **json.load(open(f"scratch/adjbias/cluster/{name}.json"))["config"]
        )
        _grid, ls0 = O.initial_levelset(cfg)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(d["coef"]))
        assert ls.h == float(d["h"]) and ls.coef.shape == ls0.coef.shape
        assert np.allclose(sys.modules["terms"].translation(ls), d["translation"])
        seeds = sorted(int(k[2:]) for k in d.files if k.startswith("J_"))
        coef = np.asarray(ls.coef)
        inside = coef < 0
        kx = np.asarray(O.L.knots(ls)).reshape(*coef.shape, 2)
        blo = kx[inside].min(0) - 0.5
        bhi = kx[inside].max(0) + 0.5
        xs, ws, nrm, gradn = surface_quadrature(ls, blo, bhi, ls.h / 8)
        b = basis_matrix(ls, xs)
        # translations (flattened, per metre)
        eps = 1e-3
        tx = {
            "terms_interp": d["translation"].ravel(),
            "modes_cdiff": np.asarray(O.translation_modes(ls, 1.0 + ls.h)[0]).ravel(),
            "exact_eps": (np.asarray(O.translate(ls, np.array([eps, 0.0])).coef) - coef).ravel()
            / eps,
            "fd_1m": (np.asarray(O.translate(ls, np.array([1.0, 0.0])).coef) - coef).ravel(),
        }
        print(f"\n=== {name}: {len(seeds)} seeds, {len(xs)} surface points, "
              f"perimeter {ws.sum():.2f} m, x in [{xs[:, 0].min():.2f}, {xs[:, 0].max():.2f}]")  # fmt: skip
        print(
            "surface motion of each translation, V_n = -B t / |grad phi| vs n_x "
            "(rms error / rms n_x):"
        )
        for k, t in tx.items():
            vn = -(b @ t) / gradn
            err = np.sqrt(np.sum(ws * (vn - nrm[:, 0]) ** 2) / np.sum(ws * nrm[:, 0] ** 2))
            print(f"  {k:13s} {err:.4f}")
        # M: surface density -> coefficient gradient; least squares for g_n
        m_op = -(b.T @ sp.diags(ws / gradn)).tocsr()
        rows = {}
        for i in seeds:
            j = float(d[f"J_{i}"])
            for t in (*TERMS, "total"):
                g = (sum(d[f"{s}_{i}"] for s in TERMS) if t == "total" else d[f"{t}_{i}"]).ravel()
                sol = spl.lsqr(m_op, g, damp=0.0, atol=1e-12, btol=1e-12, iter_lim=4000)
                g_n = sol[0]
                g_row = m_op @ g_n
                g_null = g - g_row
                r = {k: 100 * g @ v / j for k, v in tx.items()}
                r["null_frac"] = float(np.linalg.norm(g_null) / max(np.linalg.norm(g), 1e-300))
                r["surface"] = 100 * float(np.sum(ws * g_n * nrm[:, 0])) / j
                r["row_terms"] = 100 * g_row @ tx["terms_interp"] / j
                r["row_modes"] = 100 * g_row @ tx["modes_cdiff"] / j
                r["null_terms"] = 100 * g_null @ tx["terms_interp"] / j
                r["null_modes"] = 100 * g_null @ tx["modes_cdiff"] / j
                rows.setdefault(t, []).append(r)
                if t == "pathwise_tally" and i == seeds[0]:
                    first = (g, g_n, g_null)
        cols = ("terms_interp", "modes_cdiff", "exact_eps", "fd_1m", "surface", "null_frac",
                "row_terms", "row_modes", "null_terms", "null_modes")  # fmt: skip
        print("% J/m, mean ± se over seeds (null_frac = |g_null| / |g|)")
        print(f"  {'term':15s}" + "".join(f"{c:>16s}" for c in cols))
        for t, rs in rows.items():
            cells = []
            for c in cols:
                v = np.array([r[c] for r in rs])
                cells.append(f"{v.mean():9.2f}±{v.std(ddof=1) / np.sqrt(len(v)):<6.2f}")
            print(f"  {t:15s}" + "".join(f"{s:>16s}" for s in cells))
        tot = rows["total"]
        for c in ("terms_interp", "modes_cdiff", "exact_eps", "surface"):
            print(f"  total per seed {c:13s}", np.round([r[c] for r in tot], 1))
        # where along x: binned contributions of seed-mean tally gradient
        g_mean = np.mean([d[f"pathwise_tally_{i}"].ravel() for i in seeds], 0)
        j_mean = np.mean([float(d[f"J_{i}"]) for i in seeds])
        kxf = kx.reshape(-1, 2)
        phik = np.asarray(O.L.value(ls, jnp.asarray(kxf)))
        edges = np.linspace(xs[:, 0].min() - 0.25, xs[:, 0].max() + 0.25, 9)
        print(
            "seed-mean pathwise_tally by knot x (% J/m): bins, terms_interp, modes_cdiff, "
            "inside-body knots share of terms_interp"
        )
        for a, c in zip(edges[:-1], edges[1:], strict=True):
            sel = (kxf[:, 0] >= a) & (kxf[:, 0] < c)
            p1 = 100 * np.sum(g_mean[sel] * tx["terms_interp"][sel]) / j_mean
            p2 = 100 * np.sum(g_mean[sel] * tx["modes_cdiff"][sel]) / j_mean
            p3 = 100 * np.sum((g_mean * tx["terms_interp"])[sel & (phik < 0)]) / j_mean
            thick = np.ptp(xs[(xs[:, 0] >= a) & (xs[:, 0] < c), 1]) if np.any(
                (xs[:, 0] >= a) & (xs[:, 0] < c)) else 0.0  # fmt: skip
            print(f"  x [{a:6.2f},{c:6.2f})  thick {thick:5.2f}  {p1:8.2f} {p2:8.2f} {p3:8.2f}")
        sg = np.asarray(first[1])
        print("seed-0 tally surface density g_n * n_x by x bin (% J/m):")
        for a, c in zip(edges[:-1], edges[1:], strict=True):
            sel = (xs[:, 0] >= a) & (xs[:, 0] < c)
            jj = float(d[f"J_{seeds[0]}"])
            print(f"  x [{a:6.2f},{c:6.2f})  {100 * np.sum((ws * sg * nrm[:, 0])[sel]) / jj:8.2f}")
        np.savez_compressed(f"scratch/adjbias/cluster/{name}_analysis.npz", xs=xs, ws=ws, nrm=nrm,
                            g_n_tally0=sg, g_null_tally0=first[2], g_mean_tally=g_mean,
                            **{f"tx_{k}": v for k, v in tx.items()})  # fmt: skip


if __name__ == "__main__":
    main()
