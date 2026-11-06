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
import knn3  # noqa: E402
import lab  # noqa: E402

from jax_dsmc import wall_knn  # noqa: E402

K = 64
S_SCALE = 0.033 * lab.RADIUS  # arc length (m), as knn3's metrics
REF_FIT = "k3_C1_0.25_0.033_0.3_o0"  # knn3's winner: d / min(mfp, cell / 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=float, default=2.0)
    ap.add_argument("--block-q", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=2048)
    ap.add_argument("--n-tile", type=int, default=2, help="tiled metric coordinates (d, s[, v_n])")
    ap.add_argument("--fp64", action="store_true", help="rank in float64")
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
    # the capped mean-free-path metric (knn3's C1_0.25_0.033_0.3): lambda_wall as knn3.wall_mfp
    lam_wall, cell = knn3.wall_mfp(args.case, feat_n, len(np.unique(h["hit_step"])), hth)
    d_sc = float(wall_knn.wall_distance_scale(lam_wall, cell))
    v_sc = float(
        wall_knn.velocity_scale(
            jnp.asarray(feat_n[:, 0]), jnp.asarray(feat_n[:, 2:5]), jnp.asarray(feat_q[:, 0])
        )
    )
    scales = np.array([d_sc, S_SCALE, v_sc, v_sc, v_sc])
    zn, zq = feat_n / scales, feat_q / scales
    print(
        f"  metric scales (lambda_wall {1000 * lam_wall:.1f} mm, cell {cell:g} m): d {1000 * d_sc:.1f} mm, v {v_sc:.3f} V_REF",
        flush=True,
    )
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

    for attempt in ("cold", "warm"):
        t0 = time.time()
        i_jax, d2_jax, tile_q = wall_knn.knn_exact(zn, jnp.asarray(nkey), zq, jnp.asarray(hkey),
                                                   args.tile, K, args.block_q, args.block_n,
                                                   args.n_tile, low_precision=not args.fp64)  # fmt: skip
        i_jax = np.asarray(jax.block_until_ready(i_jax))
        print(f"  JAX exact kNN ({attempt}) {time.time() - t0:.1f}s; queries re-searched with a "
              f"larger tile {100 * np.mean(tile_q > args.tile):.2f} %", flush=True)  # fmt: skip
    same = np.mean(np.sort(i_jax, axis=1) == np.sort(idx_ref, axis=1))
    print(f"  neighbour sets agree on {100 * same:.4f} % of entries", flush=True)

    for attempt in ("cold", "warm"):
        t0 = time.time()
        g_jax, bad = wall_knn.local_gradients(jnp.asarray(i_jax), jnp.asarray(feat_n),
                                              jnp.asarray(lam), jnp.asarray(feat_q),
                                              jnp.asarray(jac), 2, 1e-4, 1e3)  # fmt: skip
        g_jax = jax.block_until_ready(g_jax)
        print(f"  JAX fit ({attempt}) {time.time() - t0:.1f}s", flush=True)
    g_jax = np.asarray(g_jax)
    # the lab's fit on the scipy neighbours (knn2.fit, ridge 1e-4 + guard 1e3)
    g_lab, _guarded = knn2.fit(feat_n, lam, feat_q, idx_ref, d_ref, np.full(len(zq), K), "box",
                               1e-4, 1e3)  # fmt: skip
    g_lab = g_lab[:, :1] * n_hat + g_lab[:, 1:2] * t_hat
    rel = np.linalg.norm(g_jax - g_lab, axis=1) / np.maximum(np.linalg.norm(g_lab, axis=1), 1e-30)
    print(f"  guarded {100 * np.mean(np.asarray(bad)):.3f} %; per-hit "
          f"|g_jax - g_lab| / |g_lab| median {np.median(rel):.2e} p99 "
          f"{np.percentile(rel, 99):.2e}", flush=True)  # fmt: skip
    ref = [r for r in json.loads((lab.OUT / f"knn3_{args.case}_C.json").read_text())["rows"]
           if r["seed"] == args.seed]  # fmt: skip
    for i, c in enumerate(meta["controls"]):
        pj = float(np.sum(g_jax * h["hit_dx"][i]))
        pl = float(np.sum(g_lab * h["hit_dx"][i]))
        pr = ref[0]["pathwise_x"][c][REF_FIT] if ref else float("nan")
        print(f"  {c:9s} pathwise_x  JAX {pj:+.6g}  lab fit {pl:+.6g}  knn3 run {pr:+.6g}")


if __name__ == "__main__":
    main()
