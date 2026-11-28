"""The specular radius gate (2026-11-28): the wall's normal shift through the jump of the value's
normal slope across the reflection.

For a specular plane wall moved out by delta, a hit happens tau = delta / |v_n| earlier and the
reflected particle ends the step 2 delta n further out, so (continuum, per hit)

    dJ = 2 delta d_n lam+(x, v_out) + tau [(T + L* lam)+ - (T + L* lam)-]
       = delta [d_n lam+(x, v_out) - d_n lam-(x, v_in)]

(the second form: the along-flight identity on both sides and the reflection identity
lam-(x_w, v) = lam+(x_w, R v) on the wall, whose gradient is then normal).  Production
evaluates the first form: pathwise_x's 2 delta n part with the along identity for the v_out
component (the realised collision increment for L* lam+), and the order term (one-sided fits
of the collision increment at the wall) for the exposure transfer.  Here the second form, fit-
only: the normal slopes of lam on the outgoing nodes (v_n > 0) at the hit's (x_post, v_out) and
on the incoming nodes (v_n < 0) at (x_post, v_in), by knn3's local linear fit (spec_gv.fit_slopes,
the tuned metric), each side from its own nodes.  The radius control's normal displacement is
1, so its per-hit total is d_n lam+ - d_n lam-; it replaces production's (pathwise_x along
the 2 n shift) + order, the rest (the flight part of pathwise_x, pathwise_v, tally, volume,
timing) kept.  Needs the slimmed dumps with the incoming nodes (slim.py --incoming).

    PYTHONPATH=. python3 scratch/fitlab/spec_jump.py --cases spec spec_h1000 spec_kn2 spec_kn10
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
import spec_gv  # noqa: E402
import virtual_emit as VE  # noqa: E402

D_MAX = 0.3  # the incoming slims' band


def side_slopes(feat_n, lam, key_n, feat_q, own, sc):
    """``beta [H, 6]`` of the local linear fit over one side's nodes at the queries."""
    tree = cKDTree(feat_n / sc)
    beta = np.zeros((len(feat_q), 6))
    step = 8 * spec_gv.CHUNK
    for a in range(0, len(feat_q), step):
        b = min(a + step, len(feat_q))
        idx = VE.neighbours(tree, feat_q[a:b] / sc, own[a:b], key_n)
        beta[a:b], _ = spec_gv.fit_slopes(feat_n, lam, feat_q[a:b], idx)
    return beta


def run_seed(case, seed, path, metric):
    t0 = time.time()
    meta = json.loads(path.with_suffix(".json").read_text())
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _ = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    near = nc[:, 0] < D_MAX
    lam_all = lab.demeaned({"node_lam": z["node_lam"][near], "node_step": z["node_step"][near]})
    key_all = z["node_step"][near].astype(np.int64) * 1_000_000 + z["node_slot"][near]
    nc = nc[near]
    del z
    out_side = nc[:, 2] > 0
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_v_pre", "hit_dx",
                        "hit_gx"))  # fmt: skip
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hc_in, _ = knn.coordinates(h["hit_x"], h["hit_v_pre"], h["hit_step"])
    own = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    d_hit = float(np.median(hc[:, 0]))
    sel = out_side & (nc[:, 0] < 3 * d_hit)
    sigma_v = float(np.sqrt(np.mean(np.var(nc[sel, 2:5], axis=0))))
    mfp, cell = knn3.wall_mfp(case, nc[out_side][:, :5], len(np.unique(h["hit_step"])), hth)
    sc = knn3.scales(metric, d_hit, sigma_v, mfp, cell)
    sides = {}
    for name, mask, q in (("out", out_side, hc), ("in", ~out_side, hc_in)):
        sides[name] = side_slopes(nc[mask][:, :5], lam_all[mask], key_all[mask], q[:, :5], own,
                                  sc)  # fmt: skip
    slope_out, slope_in = sides["out"][:, 1], sides["in"][:, 1]  # d/dd = d/dn (n outward)
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    prod = meta["variants"].index(lab.BASE_VARIANT)
    i = meta["controls"].index("radius")
    t = meta["production"]["radius"]["terms"][lab.BASE_VARIANT]
    total = sum(t[k] for k in lab.TERMS) + meta["production"]["radius"]["volume"]
    shift = 2.0 * n_hat  # the plane's 2 delta n part of d x_post / d R
    row = {
        "seed": seed,
        "base_total": total,
        "order": t["order"],
        "pathwise_x": t["pathwise_x"],
        # production's 2 n part of pathwise_x (its along identity included) and the rest
        "prod_shift": float(np.sum(h["hit_gx"][prod] * shift)),
        "prod_flight": float(np.sum(h["hit_gx"][prod] * (h["hit_dx"][i] - shift))),
        "knn_2out": float(np.sum(2.0 * slope_out)),
        "knn_in": float(np.sum(slope_in)),
        "jump": float(np.sum(slope_out - slope_in)),
        "dx_n_mean": float(np.mean(np.sum(h["hit_dx"][i] * n_hat, axis=1))),
    }
    print(f"  {case} seed {seed}: {len(hc)} hits; 2 n part prod {row['prod_shift']:+.4f} + order "
          f"{row['order']:+.4f} vs jump {row['jump']:+.4f} (2 out {row['knn_2out']:+.4f}, in "
          f"{row['knn_in']:+.4f}); dx.n {row['dx_n_mean']:.3f} ({time.time() - t0:.0f}s)",
          flush=True)  # fmt: skip
    return row


def report(case, rows):
    fd = np.array(list(lab.finite_differences(case)["radius"].values()))
    fd_m, fd_se = fd.mean(), fd.std(ddof=1) / len(fd) ** 0.5
    base = np.array([r["base_total"] for r in rows])
    alt = base - np.array([r["prod_shift"] + r["order"] for r in rows]) + np.array(
        [r["jump"] for r in rows])  # fmt: skip
    alt_out = base - np.array([r["prod_shift"] for r in rows]) + np.array(
        [r["knn_2out"] for r in rows])  # fmt: skip
    print(f"\n{case} radius ({len(rows)} seeds): FD {fd_m:+.4f} +- {fd_se:.4f}")
    for name, v in (("production", base), ("kNN 2 d_n lam+ (no identity) + prod order", alt_out),
                    ("jump d_n lam+ - d_n lam- (kNN both sides)", alt)):  # fmt: skip
        se = np.hypot(v.std(ddof=1) / len(v) ** 0.5, fd_se)
        print(f"  {name:44s} {v.mean():+.4f}  ratio {v.mean() / fd_m:.3f}  "
              f"excess/se {(v.mean() - fd_m) / se:+.1f}")  # fmt: skip
    for k in ("prod_shift", "order", "knn_2out", "knn_in", "jump", "prod_flight"):
        print(f"    {k:12s} {np.mean([r[k] for r in rows]):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["spec", "spec_h1000"])
    ap.add_argument("--metric", default="C0.5_0.25_0.033_0.3")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    for case in args.cases:
        out = lab.OUT / f"spec_jump_{case}.json"
        if not args.report:
            rows = [run_seed(case, s, p, args.metric) for s, p in lab.seed_files(lab.CASES[case])]
            out.write_text(json.dumps({"case": case, "metric": args.metric, "rows": rows}))
        report(case, json.loads(out.read_text())["rows"])


if __name__ == "__main__":
    main()
