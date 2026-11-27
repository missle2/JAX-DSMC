"""The wall-normal costate slope from the transport identity (2026-11-08, option 2b): the tracer
construction at zero exit distance.

Along a collisionless characteristic the backward obeys, exactly on the data (15.9 M real
free-flight node pairs, to 2e-15),

    lambda^m_k(z) = c_k(z) + lambda^m_{k+1}(z + dt v),     c = dt (L* lambda) (Backward.collision),

so to first order  0 = c + d_k lambda + dt v . grad_x lambda, and at the wall

    d lambda / d d = -[ (c + d_k lambda) / dt + v_t d lambda / d s ] / v_n,

the wall-normal slope -- the unresolved one -- from the tangential slope (the wall is long), the
per-step change (pooled over the window) and the collision source (a value, not a slope).
d_k lambda = d_k mean_k + d_k V: the per-step mean's drift (smoothed) plus the demeaned field's.

At each real hit query (x_post, g, step), local linear kNN fits (knn3's chosen metric plus a
time coordinate) of the demeaned lambda (slopes in d, s, step) and of c (value); variants:

* ``fit``: the fit's own d-slope (the chosen kNN, time-linear) -- the reference;
* ``id``: the identity's d-slope; ``id_noc``: without c (the collisionless along-flight form);
  ``id_not``: without d_k V;
* ``id_vX``: the identity only where v_n > X V_REF, the fit below (1 / v_n amplifies the
  identity's fit errors for grazing emissions).

    PYTHONPATH=. python3 scratch/fitlab/transport_id.py --case drag [--seeds 0 8] [--max-hits N]
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import knn3  # noqa: E402
import lab  # noqa: E402
import tracers as TR  # noqa: E402
import virtual_emit as VE  # noqa: E402

V_SPLITS = (0.3, 0.6, 1.0)
MEAN_SMOOTH = 11  # steps: the running mean of the per-step mean's drift


def local_coefs(feat_n, y, feat_q, idx):
    """``f64[H, P]``: the local linear fit's coefficients (intercept, then one slope per
    feature column, in the columns' units), guard dropping the velocity columns 3-5."""
    n_p = feat_n.shape[1] + 1
    out = np.zeros((len(feat_q), n_p))
    for a in range(0, len(feat_q), VE.CHUNK):
        b = min(a + VE.CHUNK, len(feat_q))
        nb = idx[a:b]
        rel = feat_n[nb] - feat_q[a:b, None, :]
        sc = np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-30)
        x = np.concatenate((np.ones((*rel.shape[:2], 1)), rel / sc), axis=2)
        ata = np.einsum("qki,qkj->qij", x, x)
        diag = np.trace(ata, axis1=1, axis2=2) / n_p
        ata = ata + VE.RIDGE * diag[:, None, None] * np.eye(n_p)[None]
        bad = np.linalg.cond(ata) > VE.GUARD
        keep = np.ones((b - a, n_p), bool)
        keep[bad, 3:6] = False
        mask = keep[:, :, None] & keep[:, None, :]
        a_eff = np.where(mask, ata, np.eye(n_p)[None])
        aty = np.einsum("qki,qk->qi", x, y[nb]) * keep
        beta = np.einsum("qij,qj->qi", np.linalg.inv(a_eff) * mask, aty)
        out[a:b, 0] = beta[:, 0]
        out[a:b, 1:] = beta[:, 1:] / sc[:, 0, :]
    return out


def run_seed(case, seed, path, max_hits, rng):
    t0 = time.time()
    meta = json.loads(path.with_suffix(".json").read_text())
    dt = float(meta["dt"])
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam", "node_coll"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn3.D_MAX)
    step_n = z["node_step"][keep].astype(float)
    feat6 = np.column_stack((nc[keep][:, :5], step_n))
    lam_raw = z["node_lam"][keep]
    coll = z["node_coll"][keep].astype(float)
    key_n = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    del z, nc
    st = step_n.astype(int)
    n_steps = int(st.max()) + 1
    mean_k = np.bincount(st, weights=lam_raw, minlength=n_steps) / np.maximum(
        np.bincount(st, minlength=n_steps), 1
    )
    lam_dm = lam_raw - mean_k[st]
    del lam_raw
    kern = np.ones(MEAN_SMOOTH) / MEAN_SMOOTH
    drift = np.gradient(np.convolve(mean_k, kern, mode="same"))  # per step
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_v_pre", "hit_dx"))
    sel = np.arange(len(h["hit_step"]))
    if max_hits and max_hits < len(sel):
        sel = np.sort(rng.choice(sel, size=max_hits, replace=False))
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    d_hit = float(np.median(hc[:, 0]))
    near = feat6[:, 0] < 3 * d_hit
    sigma_v = float(np.sqrt(np.mean(np.var(feat6[near, 2:5], axis=0))))
    mfp, cell = knn3.wall_mfp(case, feat6[:, :5], len(np.unique(h["hit_step"])), hth)
    sc6 = np.append(knn3.scales("C1_0.25_0.033_0.3", d_hit, sigma_v, mfp, cell), TR.T_SCALE)
    tree = cKDTree(feat6 / sc6)
    q6 = np.column_stack((hc[sel, :5], h["hit_step"][sel].astype(float)))
    own = h["hit_step"][sel].astype(np.int64) * 1_000_000 + h["hit_slot"][sel]
    coef_l = np.zeros((len(sel), 7))
    c_val = np.zeros(len(sel))
    for a in range(0, len(sel), TR.QCHUNK):
        b = min(a + TR.QCHUNK, len(sel))
        # the hit particle's own node (distance 0) dropped, as knn3's o0
        idx = VE.neighbours(tree, q6[a:b] / sc6, own[a:b], key_n)
        coef_l[a:b] = local_coefs(feat6, lam_dm, q6[a:b], idx)
        c_val[a:b] = local_coefs(feat6, coll, q6[a:b], idx)[:, 0]
    g_d_fit, g_s, g_k = coef_l[:, 1], coef_l[:, 2], coef_l[:, 6]
    g = h["hit_v"][sel]
    th = hth[sel]
    n_hat = np.stack((np.cos(th), np.sin(th)), axis=1)
    t_hat = np.stack((-np.sin(th), np.cos(th)), axis=1)
    vn = np.sum(g[:, :2] * n_hat, axis=1)
    vt = np.sum(g[:, :2] * t_hat, axis=1)
    dk = drift[h["hit_step"][sel]]
    ident = {
        "id": -((c_val + dk + g_k) / dt + vt * g_s) / vn,
        "id_noc": -((dk + g_k) / dt + vt * g_s) / vn,
        "id_not": -((c_val + dk) / dt + vt * g_s) / vn,
    }
    g_d = {"fit": g_d_fit, **ident}
    for x in V_SPLITS:
        g_d[f"id_v{x:g}"] = np.where(vn / lab.V_REF > x, ident["id"], g_d_fit)
    w = (h["hit_v_pre"][sel] - g)[:, :2]
    dth = np.einsum("chi,hi->ch", h["hit_dx"][:, sel], w) / np.sum(w * w, axis=1)[None]
    scale = len(h["hit_step"]) / len(sel)  # a subsample's sum, scaled to all hits
    px = {}
    for name, gd in g_d.items():
        gx = gd[:, None] * n_hat + g_s[:, None] * t_hat
        px[name] = scale * np.einsum("hi,hi,ch->c", gx, w, dth)
    rows = {}
    for i, c in enumerate(meta["controls"]):
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        total = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
        rows[c] = {"base_total": total, "prod_px": t["pathwise_x"],
                   **{k: float(v[i]) for k, v in px.items()}}  # fmt: skip
    ratio = np.median(np.abs(ident["id"]) / np.maximum(np.abs(g_d_fit), 1e-30))
    print(f"  {case} seed {seed}: {len(sel)} hits; median |id| / |fit| d-slope {ratio:.2f}; "
          f"radius px fit {rows['radius']['fit']:+.4f} id {rows['radius']['id']:+.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    return {"seed": seed, "controls": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--max-hits", type=int, default=0, help="subsample the hits (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    out = lab.OUT / f"transport_id_{args.case}{args.tag}.json"
    if not args.report:
        rng = np.random.default_rng(11)
        rows = []
        for seed, path in lab.seed_files(lab.CASES[args.case]):
            if args.seeds[0] <= seed < args.seeds[1]:
                rows.append(run_seed(args.case, seed, path, args.max_hits, rng))
                out.write_text(json.dumps({"case": args.case, "rows": rows}))
        print(f"wrote {out}")
    rows = json.loads(out.read_text())["rows"]
    if len(rows) > 1:
        TR.report(args.case, rows)


if __name__ == "__main__":
    main()
