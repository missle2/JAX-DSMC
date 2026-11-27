"""The tracer construction's slope gate (2026-11-08): does a tracer reproduce the real costate's
wall-normal slope at fixed velocity?

A replica tracer is launched from a real outgoing node's own state and step (any real node is
an unbiased sample of lambda at its state, so the tracer value must agree with it in
expectation everywhere, not only at the hit step).  Within cells of (v_n, |v_t|, wall region)
the least-squares slope in d (d < ``--d-band``) of the real demeaned value and of the tracer
value are compared: a correct construction has tracer / real slope 1 in every cell.  The
tracer value splits into the summed collision source (c-sum) and the exit value; their slopes
are reported separately.  Variants change the exit distance and the c and exit fits' scales.
(The characteristic identity itself is exact on the data: along 15.9 M real free-flight node
pairs lambda_k - lambda_{k+1} = c_k to 2e-15, 2026-11-08.)

    PYTHONPATH=. python3 scratch/fitlab/tracer_gate.py --case drag [--seed 0] [--n 200000]
        [--variants base dexit=0.05 cv=0.5 ex=0.5]
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import knn3  # noqa: E402
import lab  # noqa: E402
import tracers as TR  # noqa: E402

VN_BINS = (0.0, 0.5, 1.0, 1.5, 2.5, 99.0)
VT_BINS = (0.0, 1.0, 99.0)
NOSE = 0.75  # m of arc either side of the stagnation point


def cell_slopes(d, y, cell, n_cell):
    """Per cell: (slope, se, n) of the least-squares fit y = a + b d."""
    out = np.full((n_cell, 3), np.nan)
    for c in range(n_cell):
        m = cell == c
        if m.sum() < 200:
            continue
        dd = d[m] - d[m].mean()
        b = np.sum(dd * (y[m] - y[m].mean())) / np.sum(dd * dd)
        r = y[m] - y[m].mean() - b * dd
        out[c] = (b, np.sqrt(np.sum(r * r) / (m.sum() - 2) / np.sum(dd * dd)), m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200_000, help="real nodes replicated")
    ap.add_argument("--d-band", type=float, default=0.10)
    ap.add_argument("--d-exit", type=float, default=0.15)
    ap.add_argument("--variants", nargs="+", default=["base"],
                    help="base | key=val,... (see parse_variant)")  # fmt: skip
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    path = dict(lab.seed_files(lab.CASES[args.case]))[args.seed]
    meta = json.loads(path.with_suffix(".json").read_text())
    dt = float(meta["dt"])
    z = lab.load(path, ("node_step", "node_x", "node_v", "node_lam", "node_coll"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn3.D_MAX)
    step_n = z["node_step"][keep]
    feat_n = nc[keep][:, :5]
    x_n, v_n = z["node_x"][keep], z["node_v"][keep]
    lam_raw = z["node_lam"][keep]
    coll = z["node_coll"][keep].astype(float)
    del z, nc
    n_steps = int(step_n.max()) + 1
    mean_k = np.bincount(step_n, weights=lam_raw, minlength=n_steps) / np.maximum(
        np.bincount(step_n, minlength=n_steps), 1
    )
    lam_dm = lam_raw - mean_k[step_n]
    del lam_raw
    h = lab.load(path, ("hit_x", "hit_v", "hit_step"))
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    d_hit = float(np.median(hc[:, 0]))
    near = feat_n[:, 0] < 3 * d_hit
    sigma_v = float(np.sqrt(np.mean(np.var(feat_n[near, 2:5], axis=0))))
    mfp, cell = knn3.wall_mfp(args.case, feat_n, len(np.unique(h["hit_step"])), hth)
    sc = knn3.scales("C1_0.25_0.033_0.3", d_hit, sigma_v, mfp, cell)
    # the replicated real nodes: inside the band, at least one step before the record's end
    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero((feat_n[:, 0] < args.d_band) & (step_n < n_steps - 1))
    pick = rng.choice(pool, size=min(args.n, len(pool)), replace=False)
    f = feat_n[pick]
    cell_id = (np.digitize(f[:, 2], VN_BINS) - 1) * 4 + (np.digitize(np.abs(f[:, 3]), VT_BINS)
                                                       - 1) * 2 + (np.abs(f[:, 1]) > NOSE)  # fmt: skip
    n_cell = (len(VN_BINS) - 1) * 4
    real = cell_slopes(f[:, 0], lam_dm[pick], cell_id, n_cell)
    print(f"{args.case} seed {args.seed}: {len(pick)} replicated nodes, lambda_wall "
          f"{1000 * mfp:.0f} mm, C1 d scale {1000 * sc[0]:.0f} mm ({time.time() - t0:.0f}s)",
          flush=True)  # fmt: skip
    res = {"case": args.case, "seed": args.seed, "lambda_wall": mfp, "variants": {}}
    out = lab.OUT / f"tracer_gate_{args.case}_s{args.seed}{args.tag}.json"
    for spec in args.variants:
        opt = parse_variant(spec, args.d_exit)
        c_sc = sc.copy()
        c_sc[0] = opt["cd"] if opt["cd"] else c_sc[0]
        c_sc[1] *= opt["cs"]
        c_sc[2:] *= opt["cv"]
        e_sc = sc * opt["ex"]
        tl = bool(opt["tl"])
        c_field = TR.TimedField(feat_n, step_n.astype(float), coll, c_sc, tl, opt["ts"])
        exit_field = TR.TimedField(feat_n, step_n.astype(float), lam_dm, e_sc, tl, opt["ts"])
        _tid, _x, _k, _l, lam0, exit0 = TR.fly(x_n[pick], v_n[pick],
                                               step_n[pick].astype(np.int64), n_steps, dt,
                                               opt["dexit"], c_field, exit_field, mean_k)  # fmt: skip
        ok = np.isfinite(lam0)
        parts = {"tracer": lam0, "c_sum": lam0 - exit0, "exit": exit0}
        sl = {k: cell_slopes(f[ok, 0], v[ok], cell_id[ok], n_cell) for k, v in parts.items()}
        diff = cell_slopes(f[ok, 0], lam_dm[pick][ok] - lam0[ok], cell_id[ok], n_cell)
        good = np.isfinite(real[:, 0]) & np.isfinite(sl["tracer"][:, 0])
        w = real[good, 2]
        den = np.sum(w * real[good, 0] ** 2)
        ratio = {k: float(np.sum(w * v[good, 0] * real[good, 0]) / den) for k, v in sl.items()}
        chi = float(np.nansum((diff[:, 0] / diff[:, 1]) ** 2))
        print(f"  {spec}: tracer/real slope {ratio['tracer']:.3f} (c-sum part "
              f"{ratio['c_sum']:+.3f}, exit part {ratio['exit']:+.3f}); real-minus-tracer chi2 "
              f"{chi:.0f} over {int(np.isfinite(diff[:, 0]).sum())} cells ({time.time() - t0:.0f}s)",
              flush=True)  # fmt: skip
        for c in np.flatnonzero(good):
            vn = c // 4
            print(f"    v_n {VN_BINS[vn]}-{VN_BINS[vn + 1]} |v_t| {'<1' if c % 4 < 2 else '>1'} "
                  f"{'flank' if c % 2 else 'nose '}: real {real[c, 0]:+.2e} tracer "
                  f"{sl['tracer'][c, 0]:+.2e} (c {sl['c_sum'][c, 0]:+.2e} exit "
                  f"{sl['exit'][c, 0]:+.2e})  diff {diff[c, 0] / diff[c, 1]:+.1f} se")  # fmt: skip
        res["variants"][spec] = {"ratio": ratio, "chi2": chi, "real": real.tolist(),
                                 **{k: v.tolist() for k, v in sl.items()},
                                 "diff": diff.tolist()}  # fmt: skip
        out.write_text(json.dumps(res))


def parse_variant(spec, d_exit):
    """``"base"`` or ``"key=val,..."``: dexit (m), cd (c's d scale, m; 0 = the C1 one), cs / cv
    (c's arc / velocity scale factors), ex (the exit fit's scale factor), tl (1: the fits linear
    in time too), ts (their time scale, steps)."""
    opt = {"dexit": d_exit, "cd": 0.0, "cs": 1.0, "cv": 1.0, "ex": 1.0, "tl": 0.0,
           "ts": TR.T_SCALE}
    if spec != "base":
        for item in spec.split(","):
            k, v = item.split("=")
            opt[k] = float(v)
    return opt

if __name__ == "__main__":
    main()
