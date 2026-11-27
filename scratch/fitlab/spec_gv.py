"""The specular pathwise_v gate (2026-11-27, paper plan phase B): is the production velocity
gradient of the costate at the hits what carries the specular excess?

spec_budget.py: across dt / 2, dt, 2 dt (FD flat) every specular term is fixed but
pathwise_v = sum_hits grad_v lambda . d v_post / d theta, which shrinks toward 0 as dt falls
(radius -0.032 / -0.052 / -0.072 against ~-0.10 needed to close; cos 2 theta +0.06 / +0.11 /
+0.15 against ~+0.17) -- the production grad_v (per-cell polynomial in v after the along-flight
identity) is dt-dependent.  Here grad_v is refitted at every hit from its k nearest outgoing
nodes (knn3's metric, the hit's own particle dropped) by the local linear fit in (d, s, v_n,
v_t, |v_z|) -- its velocity slopes, rotated to the lab frame -- and pathwise_v recomputed; the
condition guard (velocity columns dropped) is counted.  ``gx`` likewise (pathwise_x of the same
fit).  Totals replace production's pathwise_x and pathwise_v.

    PYTHONPATH=. python3 scratch/fitlab/spec_gv.py --cases spec spec_dt2 spec_2dt [--metric M]
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
import virtual_emit as VE  # noqa: E402

CHUNK = 2048


def fit_slopes(feat_n, y, feat_q, idx):
    """``(beta [H, 6], guarded [H])``: knn3's local linear fit (range-scaled features, relative
    ridge, condition guard dropping the velocity columns) -- intercept and the 5 slopes."""
    n_p = feat_n.shape[1] + 1
    out = np.zeros((len(feat_q), n_p))
    guarded = np.zeros(len(feat_q), bool)
    for a in range(0, len(feat_q), CHUNK):
        b = min(a + CHUNK, len(feat_q))
        nb = idx[a:b]
        rel = feat_n[nb] - feat_q[a:b, None, :]
        sc = np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-30)
        x = np.concatenate((np.ones((*rel.shape[:2], 1)), rel / sc), axis=2)
        ata = np.einsum("qki,qkj->qij", x, x)
        diag = np.trace(ata, axis1=1, axis2=2) / n_p
        ata = ata + knn3.RIDGE * diag[:, None, None] * np.eye(n_p)[None]
        bad = np.linalg.cond(ata) > knn3.GUARD
        keep = np.ones((b - a, n_p), bool)
        keep[bad, 3:6] = False
        mask = keep[:, :, None] & keep[:, None, :]
        a_eff = np.where(mask, ata, np.eye(n_p)[None])
        aty = np.einsum("qki,qk->qi", x, y[nb]) * keep
        beta = np.einsum("qij,qj->qi", np.linalg.inv(a_eff) * mask, aty)
        out[a:b, 0] = beta[:, 0]
        out[a:b, 1:] = beta[:, 1:] / sc[:, 0, :]
        guarded[a:b] = bad
    return out, guarded


def run_seed(case, seed, path, metric):
    t0 = time.time()
    meta = json.loads(path.with_suffix(".json").read_text())
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _ = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn3.D_MAX)
    feat_n = nc[keep][:, :5]
    key_n = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    del z, nc
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_dv", "hit_gv"))
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    feat_q = hc[:, :5]
    own = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    d_hit = float(np.median(hc[:, 0]))
    near = feat_n[:, 0] < 3 * d_hit
    sigma_v = float(np.sqrt(np.mean(np.var(feat_n[near, 2:5], axis=0))))
    mfp, cell = knn3.wall_mfp(case, feat_n, len(np.unique(h["hit_step"])), hth)
    sc = knn3.scales(metric, d_hit, sigma_v, mfp, cell)
    tree = cKDTree(feat_n / sc)
    beta = np.zeros((len(feat_q), 6))
    guarded = np.zeros(len(feat_q), bool)
    for a in range(0, len(feat_q), 8 * CHUNK):
        b = min(a + 8 * CHUNK, len(feat_q))
        idx = VE.neighbours(tree, feat_q[a:b] / sc, own[a:b], key_n)
        beta[a:b], guarded[a:b] = fit_slopes(feat_n, lam, feat_q[a:b], idx)
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    gx = beta[:, 1:2] * n_hat + beta[:, 2:3] * t_hat
    gv = np.zeros((len(feat_q), 3))
    gv[:, :2] = (beta[:, 3:4] * n_hat + beta[:, 4:5] * t_hat) / lab.V_REF
    gv[:, 2] = beta[:, 5] * np.sign(h["hit_v"][:, 2]) / lab.V_REF
    prod = meta["variants"].index(lab.BASE_VARIANT)
    rows = {}
    for i, c in enumerate(meta["controls"]):
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        total = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
        px = float(np.sum(gx * h["hit_dx"][i]))
        pv = float(np.sum(gv * h["hit_dv"][i]))
        pv_prod = float(np.sum(h["hit_gv"][prod] * h["hit_dv"][i]))
        rows[c] = {"base_total": total, "prod_px": t["pathwise_x"], "prod_pv": t["pathwise_v"],
                   "prod_pv_check": pv_prod, "knn_px": px, "knn_pv": pv}  # fmt: skip
    print(f"  {case} seed {seed}: {len(feat_q)} hits, guarded {guarded.mean():.1%}; radius pv "
          f"prod {rows['radius']['prod_pv']:+.4f} knn {rows['radius']['knn_pv']:+.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    return {"seed": seed, "guarded": float(guarded.mean()), "controls": rows}


def report(case, rows):
    fds = lab.finite_differences(case)
    print(f"\n{case}: {len(rows)} seeds")
    for c in ("radius", "mode2"):
        if c not in fds:
            continue
        fdv = np.array(list(fds[c].values()))
        fd_m, fd_se = fdv.mean(), fdv.std(ddof=1) / len(fdv) ** 0.5
        r = [x["controls"][c] for x in rows]
        base = np.array([x["base_total"] for x in r])
        out = {}
        for name, dpx, dpv in (
            ("production", 0, 0),
            ("knn gx, prod gv", "knn_px", 0),
            ("prod gx, knn gv", 0, "knn_pv"),
            ("knn gx + gv", "knn_px", "knn_pv"),
        ):
            tot = base.copy()
            if dpx:
                tot += np.array([x[dpx] - x["prod_px"] for x in r])
            if dpv:
                tot += np.array([x[dpv] - x["prod_pv"] for x in r])
            se = np.hypot(tot.std(ddof=1) / len(tot) ** 0.5, fd_se)
            out[name] = tot.mean()
            print(f"  {c:7s} {name:18s} total {tot.mean():+.5g}  ratio {tot.mean() / fd_m:.3f}  "
                  f"excess/se {(tot.mean() - fd_m) / se:+.1f}")  # fmt: skip
        pv_p = np.mean([x["prod_pv"] for x in r])
        pv_k = np.mean([x["knn_pv"] for x in r])
        print(f"  {c:7s} pathwise_v production {pv_p:+.4g}  knn {pv_k:+.4g}   FD {fd_m:+.5g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["spec", "spec_dt2", "spec_2dt"])
    ap.add_argument("--metric", default="C0.5_0.25_0.033_0.3")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    for case in args.cases:
        out = lab.OUT / f"spec_gv_{case}.json"
        if not args.report:
            rows = [run_seed(case, s, p, args.metric) for s, p in lab.seed_files(lab.CASES[case])]
            out.write_text(json.dumps({"case": case, "metric": args.metric, "rows": rows}))
        report(case, json.loads(out.read_text())["rows"])


if __name__ == "__main__":
    main()
