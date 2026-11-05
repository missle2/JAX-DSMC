"""k-nearest-neighbour wall gradients (2026-11-05): an adaptive-scale grad_x lambda at each hit.

Per seed (production-feasible: the seed's own nodes only), the outgoing nodes (v_n > 0) are
put in a scaled space -- wall distance d / l_d, arc length R theta / l_s, the wall-frame
velocity / l_v (V_REF units) and optionally the step / l_t -- and each hit's k nearest
neighbours (the hit's own node left out) carry a local linear regression of the per-step
demeaned lambda on (d, s, v_n, v_t, |v_z|, [step]) about the hit; grad_x = g_d n + g_s t.
The neighbourhood shrinks where the nodes are dense (the cold stagnation layer) and grows
where they are sparse: the scale adapts, the anisotropy (l_d : l_s : l_v : l_t) is the knob.
Neighbour lists are queried once at the largest k; the smaller k are their prefixes.

pathwise_x per control into the lab's closure format (``out/knn_<case>_<tag>.json``; fits
``knn_k<k>_<metric>``), scored by ``summary.py``.

    PYTHONPATH=. python3 scratch/fitlab/knn.py --case drag [--seeds 0 8] [--ks 64 256 1024]
        [--metrics d0.01_s0.1_v0.5 ...]
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

D_MAX = 0.6
CHUNK = 2048
RIDGE = 1e-9


def parse_metric(text):
    """``d0.01_s0.1_v0.5[_t50]`` -> {"d": 0.01, "s": 0.1, "v": 0.5, "t": 50 or inf}."""
    out = {"t": np.inf}
    for part in text.split("_"):
        out[part[0]] = float(part[1:])
    return out


def coordinates(x, v, step):
    d, th, vn, vt, vz = lab.frame(x, v)
    s = lab.RADIUS * np.angle(np.exp(1j * (th - np.pi)))  # arc length from stagnation
    return np.stack((d, s, vn, vt, np.abs(vz), step.astype(float)), axis=1), th


def scaled(c, m):
    cols = [c[:, 0] / m["d"], c[:, 1] / m["s"], c[:, 2] / m["v"], c[:, 3] / m["v"],
            c[:, 4] / m["v"]]  # fmt: skip
    if np.isfinite(m["t"]):
        cols.append(c[:, 5] / m["t"])
    return np.stack(cols, axis=1)


def local_gradients(nc, lam, hc, idx, ks, use_t):
    """(g_d, g_s) per hit and k from the neighbour lists ``idx`` (H, k_max)."""
    out = {k: np.zeros((hc.shape[0], 2)) for k in ks}
    nf = 6 + (1 if use_t else 0)
    cols = [0, 1, 2, 3, 4] + ([5] if use_t else [])
    for a in range(0, hc.shape[0], CHUNK):
        b = min(a + CHUNK, hc.shape[0])
        nb = idx[a:b]
        rel = nc[nb][:, :, cols] - hc[a:b, None, :][:, :, cols]  # (q, k, f-1)
        scale = np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-12)
        feat = np.concatenate((np.ones(rel.shape[:2] + (1,)), rel / scale), axis=2)
        y = lam[nb]
        for k in ks:
            f, yy = feat[:, :k], y[:, :k]
            ata = np.einsum("qki,qkj->qij", f, f) + RIDGE * k * np.eye(nf)[None]
            aty = np.einsum("qki,qk->qi", f, yy)
            beta = np.linalg.solve(ata, aty[..., None])[..., 0]
            out[k][a:b] = beta[:, 1:3] / scale[:, 0, :2]
    return out


def run_seed(case, seed, path, ks, metrics):
    t0 = time.time()
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    nc = nc[keep]
    del z
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    kmax = max(ks)
    for mtext in metrics:
        m = parse_metric(mtext)
        tree = cKDTree(scaled(nc, m))
        _dist, idx = tree.query(scaled(hc, m), k=kmax + 1, workers=-1)
        # leave the hit's own node out (its step and slot), else drop the farthest
        own = nkey[idx] == hkey[:, None]
        order = np.argsort(own, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)[:, :kmax]
        grads = local_gradients(nc, lam, hc, idx, ks, np.isfinite(m["t"]))
        for k, g in grads.items():
            fits[f"knn_k{k}_{mtext}"] = g[:, :1] * n_hat + g[:, 1:2] * t_hat
        reach = np.median(np.abs(nc[idx[:, kmax - 1], 0] - hc[:, 0]))
        print(f"  {case} seed {seed} {mtext}: median |d - d_hit| of the {kmax}th neighbour "
              f"{100 * reach:.2f} cm  ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    px = {c: {name: float(np.sum(g * h["hit_dx"][i])) for name, g in fits.items()}
          for i, c in enumerate(meta["controls"])}  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        totals[c] = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
    return {"seed": seed, "pathwise_x": px, "base_total": totals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--metrics", nargs="+", default=["d0.01_s0.1_v0.5"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    for seed, path in lab.seed_files(lab.CASES[args.case]):
        if args.seeds[0] <= seed < args.seeds[1]:
            rows.append(run_seed(args.case, seed, path, args.ks, args.metrics))
    out = lab.OUT / f"knn_{args.case}{args.tag}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
