"""Gate 2 (2026-11-05): ``jax_dsmc.wall_knn`` against the lab's kNN (scratch/fitlab/knn2.py,
the winning fit: all time-pooled samples, box weights, ridge 1e-4 + condition guard 1e3, the
absolute metric d / 3 cm, s / 10 cm, v / 0.5 V_REF) on one seed's dump: the same features and
metric, the k nearest found by scipy's KD tree and by the JAX tile search, then the local fits.
Reports the neighbour-set agreement, the share of queries whose k-th neighbour lies beyond one
tile (where the tile search may be inexact), the per-hit gradient difference, the pathwise
sums per control against knn2's, and the JAX timings.

    PYTHONPATH=. python3 scratch/fitlab/jaxknn_check.py [--case drag] [--seed 0] [--tile 2]
"""

import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import knn2  # noqa: E402
import lab  # noqa: E402

from jax_dsmc import wall_knn  # noqa: E402

K = 64
SCALES = np.array([0.03, 0.1, 0.5, 0.5, 0.5])  # d, s (m); v_n, v_t, |v_z| (V_REF)
REF_FIT = "k2_d0.03_s0.1_v0.5_u0_f0_box_r0.0001_g1000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=float, default=2.0)
    ap.add_argument("--block-q", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=2048)
    args = ap.parse_args()
    path = dict(lab.seed_files(lab.CASES[args.case]))[args.seed]
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn.D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    feat_n = nc[keep][:, :5]
    del z, nc
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    feat_q = hc[:, :5]
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    jac = np.zeros((len(hc), 5, 2))
    jac[:, 0], jac[:, 1] = n_hat, t_hat
    zn, zq = feat_n / SCALES, feat_q / SCALES
    print(f"{args.case} seed {args.seed}: {len(zn)} nodes, {len(zq)} queries", flush=True)

    # the reference: scipy's KD tree, the own (step, slot) out
    t0 = time.time()
    dist, idx = cKDTree(zn).query(zq, k=K + 1, workers=-1)
    own = nkey[idx] == hkey[:, None]
    o = np.argsort(own, axis=1, kind="stable")
    idx_ref = np.take_along_axis(idx, o, axis=1)[:, :K]
    d_ref = np.take_along_axis(dist, o, axis=1)[:, :K]
    print(f"  scipy kNN {time.time() - t0:.1f}s; k-th distance quantiles 50/99/100 "
          f"{np.percentile(d_ref[:, -1], [50, 99, 100]).round(2)} (tile {args.tile})", flush=True)  # fmt: skip

    lo, n = wall_knn.tile_grid(zn[:, :2].min(0), zn[:, :2].max(0), args.tile)
    t0 = time.time()
    i_jax, d2_jax = wall_knn.knn(jnp.asarray(zn), jnp.asarray(nkey), jnp.asarray(zq),
                                 jnp.asarray(hkey), jnp.ones(len(zq), bool), jnp.asarray(lo), n,
                                 args.tile, K, args.block_q, args.block_n, 2)  # fmt: skip
    i_jax = np.asarray(jax.block_until_ready(i_jax))
    t_knn = time.time() - t0
    same = np.mean(np.sort(i_jax, axis=1) == np.sort(idx_ref, axis=1))
    beyond = np.mean(d_ref[:, -1] > args.tile)
    print(f"  JAX kNN {t_knn:.1f}s (incl. compile): neighbour sets agree on {100 * same:.3f} % "
          f"of entries; k-th neighbour beyond one tile at {100 * beyond:.3f} % of queries",
          flush=True)  # fmt: skip

    t0 = time.time()
    g_jax, bad = wall_knn.local_gradients(jnp.asarray(i_jax), jnp.asarray(feat_n),
                                          jnp.asarray(lam), jnp.asarray(feat_q), jnp.asarray(jac),
                                          2, 1e-4, 1e3)  # fmt: skip
    g_jax = np.asarray(g_jax)
    t_fit = time.time() - t0
    # the lab's fit on the scipy neighbours (knn2.fit, ridge 1e-4 + guard 1e3)
    g_lab, _guarded = knn2.fit(feat_n, lam, feat_q, idx_ref, d_ref, np.full(len(zq), K), "box",
                               1e-4, 1e3)  # fmt: skip
    g_lab = g_lab[:, :1] * n_hat + g_lab[:, 1:2] * t_hat
    rel = np.linalg.norm(g_jax - g_lab, axis=1) / np.maximum(np.linalg.norm(g_lab, axis=1), 1e-30)
    print(f"  JAX fit {t_fit:.1f}s; guarded {100 * np.mean(np.asarray(bad)):.3f} %; per-hit "
          f"|g_jax - g_lab| / |g_lab| median {np.median(rel):.2e} p99 "
          f"{np.percentile(rel, 99):.2e}", flush=True)  # fmt: skip
    ref = [r for r in json.loads((lab.OUT / f"knn2_{args.case}.json").read_text())["rows"]
           if r["seed"] == args.seed]  # fmt: skip
    for i, c in enumerate(meta["controls"]):
        pj = float(np.sum(g_jax * h["hit_dx"][i]))
        pl = float(np.sum(g_lab * h["hit_dx"][i]))
        pr = ref[0]["pathwise_x"][c][REF_FIT] if ref else float("nan")
        print(f"  {c:9s} pathwise_x  JAX {pj:+.6g}  lab fit {pl:+.6g}  knn2 run {pr:+.6g}")


if __name__ == "__main__":
    main()
