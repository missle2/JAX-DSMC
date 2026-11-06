"""kNN wall gradients with a wall-distance spread floor, kernel weights and a speed-relative
metric (2026-11-05, gate 1 of the GPU kNN plan).

kNN-64 (scratch/fitlab/knn.py) closed drag and heat and was the only fit near specular's
~zero wall slope, at 1.7-2.75x production's spread (6-11x specular): in the dense layer its 64
neighbours span < 1 mm of wall distance and the slope is a difference over almost nothing.
Here, per hit (each seed's own nodes, every captured step, the own node left out):

* neighbours in a metric that is either the absolute one (``d0.03_s0.1_v0.5``) or speed
  relative (``R<cd>_<cs>_<rv>``: wall distance / (cd x the hits' median flight length v_n dt),
  arc length / (cs R), log speed and the unit velocity's v_t, |v_z| over rv -- no V_REF);
* k grown from ``K_MIN`` until the neighbours' wall-distance range reaches ``c`` flight lengths
  of the HIT's own v_n dt (``c`` = 0: plain kNN), capped at ``K_MAX``;
* weights box or tricube in the metric distance (scaled to the k-th neighbour);
* the local linear fit in (d, s, v_n, v_t, |v_z|) -> grad_x = g_d n + g_s t.

The floor's cost is measured, not assumed (the specular spike concern): per fit, the share of
hits whose k the floor raised, the median wall-distance span, and the mean g_d of the nose
hits within 2 mm of the wall against the same metric's plain kNN.  Closure rows to
``out/knn2_<case><tag>.json`` (fits ``k2_<metric>_f<c>_<box|tri>``), the diagnostics in each
row's ``diag``.

    PYTHONPATH=. python3 scratch/fitlab/knn2.py --case drag [--metrics ...] [--floors 0 0.5]
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import lab  # noqa: E402

D_MAX = 0.6
K_MIN = 64
K_MAX = 512
CHUNK = 1024
RIDGE = 1e-9
SPEED_FLOOR = 0.05  # V_REF: the log-speed coordinate's floor
NOSE = 20.0  # deg: the near-wall slope diagnostic's region
NEAR = 0.002  # m
# (relative ridge, condition guard) of the local fit: the old absolute ridge, relative ridges,
# and the guard to the (1, d, s) fit
REGS = ((0.0, np.inf), (1e-4, 1e3))
# 0: every sample but the own (step, slot); 1: the nearest sample of each other particle;
# 2: every sample, each particle's weight split over its samples (one particle, one vote)
UNIQUE = (0, 1, 2)


def metric_coords(c, metric, flight):
    """Scaled coordinates for the KD tree: absolute (``d.._s.._v..``) or speed-relative."""
    if not metric.startswith("R"):
        return knn.scaled(c, knn.parse_metric(metric))
    cd, cs, rv = (float(x) for x in metric[1:].split("_"))
    speed = np.sqrt(np.sum(c[:, 2:5] ** 2, axis=1))
    return np.stack((c[:, 0] / (cd * flight), c[:, 1] / (cs * lab.RADIUS),
                     np.log(np.maximum(speed, SPEED_FLOOR)) / rv,
                     c[:, 3] / np.maximum(speed, 1e-12) / rv,
                     c[:, 4] / np.maximum(speed, 1e-12) / rv), axis=1)  # fmt: skip


def distinct_mask(slots, hit_slot):
    """``bool[H, K]``: neighbour j is another particle's nearest sample (rows in distance
    order): not the hit's own slot, and the first occurrence of its slot in the row."""
    order = np.argsort(slots, axis=1, kind="stable")  # stable: distance order within a slot
    srt = np.take_along_axis(slots, order, axis=1)
    first = np.ones_like(srt, dtype=bool)
    first[:, 1:] = srt[:, 1:] != srt[:, :-1]
    out = np.zeros_like(first)
    np.put_along_axis(out, order, first, axis=1)
    return out & (slots != hit_slot[:, None])


def chosen_k(d_nb, floor):
    """Per hit, the smallest k >= K_MIN whose first k neighbours span ``floor`` in d."""
    span = np.maximum.accumulate(d_nb, axis=1) - np.minimum.accumulate(d_nb, axis=1)
    ok = span >= floor[:, None]
    ok[:, : K_MIN - 1] = False
    ok[:, -1] = True
    return np.argmax(ok, axis=1) + 1


def multiplicity(slots, inside):
    """``[H, K]``: how many of the row's in-window neighbours share each one's particle."""
    sentinel = -1 - np.arange(slots.shape[1])[None, :]  # out-of-window entries unique
    s = np.where(inside, slots, sentinel)
    order = np.argsort(s, axis=1, kind="stable")
    srt = np.take_along_axis(s, order, axis=1)
    new = np.ones_like(srt, dtype=bool)
    new[:, 1:] = srt[:, 1:] != srt[:, :-1]
    run = np.cumsum(new, axis=1)  # run id per sorted entry
    counts = np.zeros_like(srt)
    rows = np.arange(s.shape[0])[:, None]
    np.add.at(counts, (np.broadcast_to(rows, run.shape), run - 1), 1)
    out = np.empty_like(srt)
    np.put_along_axis(out, order, np.take_along_axis(counts, run - 1, axis=1), axis=1)
    return out


def fit(nc, lam, hc, idx, dist, k_sel, kernel, ridge=0.0, guard=np.inf, slots=None):
    """(g_d, g_s) per hit from its first ``k_sel`` neighbours, box or tricube weights.
    ``ridge``: relative to the design's mean diagonal (0: the absolute 1e-9 k); ``guard``: a
    design whose condition number exceeds it takes the reduced fit in (1, d, s) instead (the
    velocity terms dropped at that hit).  Returns ``(g, guarded)``."""
    out = np.zeros((hc.shape[0], 2))
    guarded = np.zeros(hc.shape[0], bool)
    kmax = int(k_sel.max())  # the masked columns past every hit's k carry nothing
    idx, dist = idx[:, :kmax], dist[:, :kmax]
    j = np.arange(idx.shape[1])
    for a in range(0, hc.shape[0], CHUNK):
        b = min(a + CHUNK, hc.shape[0])
        nb = idx[a:b]
        inside = j[None, :] < k_sel[a:b, None]
        if kernel == "tri":
            r_k = np.take_along_axis(dist[a:b], (k_sel[a:b] - 1)[:, None], axis=1)
            u = np.clip(dist[a:b] / np.maximum(1.0001 * r_k, 1e-12), 0.0, 1.0)
            w = np.where(inside, (1.0 - u**3) ** 3, 0.0)
        else:
            w = inside.astype(float)
        if slots is not None:  # one particle, one vote
            w = w / multiplicity(slots[nb], inside)
        rel = nc[nb][:, :, :5] - hc[a:b, None, :5]
        scale = np.maximum(np.max(np.abs(rel) * inside[..., None], axis=1, keepdims=True), 1e-12)
        feat = np.concatenate((np.ones(rel.shape[:2] + (1,)), rel / scale), axis=2)
        fw = feat * w[..., None]
        ata = np.einsum("qki,qkj->qij", fw, feat)
        aty = np.einsum("qki,qk->qi", fw, lam[nb])
        if ridge > 0:
            diag = np.trace(ata, axis1=1, axis2=2) / 6.0
            ata = ata + ridge * diag[:, None, None] * np.eye(6)[None]
        else:
            ata = ata + RIDGE * np.eye(6)[None] * k_sel[a:b, None, None]
        beta = np.linalg.solve(ata, aty[..., None])[..., 0]
        g = beta[:, 1:3] / scale[:, 0, :2]
        if np.isfinite(guard):
            bad = np.linalg.cond(ata) > guard
            if bad.any():
                a3 = ata[bad][:, :3, :3]
                b3 = np.linalg.solve(a3 + 1e-9 * np.eye(3)[None], aty[bad][:, :3, None])[..., 0]
                g[bad] = b3[:, 1:3] / scale[bad][:, 0, :2]
                guarded[a:b] = bad
        out[a:b] = g
    return out, guarded


def run_seed(case, seed, path, metrics, floors, kernels):
    t0 = time.time()
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    nslot = z["node_slot"][keep]
    nc = nc[keep]
    del z
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    hit_flight = np.maximum(hc[:, 2], 0.0) * lab.V_REF * meta["dt"]  # m, the hit's v_n dt
    flight = float(np.median(hit_flight))
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (hth - np.pi)))))
    near = (ang < NOSE) & (hc[:, 0] < NEAR)
    fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    diag = {"flight_median_m": flight, "near_hits": int(near.sum())}
    for metric in metrics:
        tree = cKDTree(metric_coords(nc, metric, flight))
        dist, idx = tree.query(metric_coords(hc, metric, flight), k=K_MAX + 1, workers=-1)
        idx_all, dist_all = idx, dist
        for unique in UNIQUE:
            if unique == 1:
                # distinct particles: the hit's own particle out at every step, and one sample
                # (the nearest) per other particle -- time pooling otherwise fills a slow
                # particle's neighbourhood with its own trajectory (collinear in d, v_n)
                valid = distinct_mask(nslot[idx_all], h["hit_slot"])
            else:
                valid = nkey[idx_all] != hkey[:, None]  # the own (step, slot) only
            order = np.argsort(~valid, axis=1, kind="stable")
            idx = np.take_along_axis(idx_all, order, axis=1)[:, :K_MAX]
            dist = np.take_along_axis(dist_all, order, axis=1)[:, :K_MAX]
            n_valid = np.minimum(valid.sum(axis=1), K_MAX)
            d_nb = nc[idx, 0]
            for c in floors:
                k_sel = chosen_k(d_nb, c * hit_flight) if c > 0 else np.full(hc.shape[0], K_MIN)
                k_sel = np.maximum(np.minimum(k_sel, n_valid), 1)
                span = np.take_along_axis(np.maximum.accumulate(d_nb, axis=1)
                                          - np.minimum.accumulate(d_nb, axis=1),
                                          (k_sel - 1)[:, None], axis=1)[:, 0]  # fmt: skip
                for kernel in kernels:
                    for ridge, guard in REGS:
                        g, guarded = fit(nc, lam, hc, idx, dist, k_sel, kernel, ridge, guard,
                                         nslot if unique == 2 else None)  # fmt: skip
                        name = f"k2_{metric}_u{unique}_f{c:g}_{kernel}_r{ridge:g}_g{guard:g}"
                        fits[name] = g[:, :1] * n_hat + g[:, 1:2] * t_hat
                        diag[name] = {
                            "guarded": float(np.mean(guarded)),
                            "short": float(np.mean(n_valid < K_MIN)),
                            "k_median": float(np.median(k_sel)),
                            "span_median_mm": float(1000 * np.median(span)),
                            "near_gd": float(np.mean(g[near, 0])) if near.any() else None,
                        }
        print(f"  {case} seed {seed} {metric}: flight {1000 * flight:.1f} mm  "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    px = {c: {name: float(np.sum(g * h["hit_dx"][i])) for name, g in fits.items()}
          for i, c in enumerate(meta["controls"])}  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        totals[c] = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
    return {"seed": seed, "pathwise_x": px, "base_total": totals, "diag": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--metrics", nargs="+", default=["d0.03_s0.1_v0.5", "R3_0.033_0.3"])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0])
    ap.add_argument("--kernels", nargs="+", default=["box", "tri"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    for seed, path in lab.seed_files(lab.CASES[args.case]):
        if args.seeds[0] <= seed < args.seeds[1]:
            rows.append(run_seed(args.case, seed, path, args.metrics, args.floors, args.kernels))
    out = lab.OUT / f"knn2_{args.case}{args.tag}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
