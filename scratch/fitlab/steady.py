"""Is the captured window statistically steady?  (2026-11-06: the pooled kNN fit pools every
captured step's near-wall nodes, unbiased only if the near-wall statistics are stationary.)

Per case, per seed, in B-step blocks of the captured window:

* hits per step (the wall's incoming number flux);
* the wall-normal impulse per step, sum |(v_pre - v) . n| (the pressure, so the drag);
* the outgoing near-wall nodes (v_n > 0, d < 5 cm) per step (the wall-layer density);
* the near-wall slope d lambda / d d of the per-step demeaned costate (outgoing nodes, d < 10
  cm, least squares lambda = a + b d in the block) -- the quantity the fits pool.

Each is divided by its whole-window value; the json holds the seeds' rows.

    PYTHONPATH=. python3 scratch/fitlab/steady.py [--cases drag spec ...] [--seeds 4] [--block 25]
"""

import argparse
import json

import numpy as np

import lab

D_NEAR = 0.05  # the wall-layer count's band (m)
D_SLOPE = 0.10  # the costate slope's band (m)


def _block_slope(d, lam, blk, n_blk):
    """Per-block least-squares slope of ``lam`` on ``d``; the whole-window slope too."""
    out = np.full(n_blk, np.nan)
    for b in range(n_blk):
        m = blk == b
        if np.sum(m) > 10:
            out[b] = np.polyfit(d[m], lam[m], 1)[0]
    return out, np.polyfit(d, lam, 1)[0]


def seed_rows(path, block):
    z = lab.load(path, ("node_step", "node_x", "node_v", "node_lam"))
    r = z["node_x"] - lab.CENTRE
    rn = np.linalg.norm(r, axis=1)
    d = rn - lab.RADIUS
    vn = np.sum(z["node_v"][:, :2] * r, axis=1) / rn
    keep = (vn > 0) & (d < lab.SETTINGS["d_max"])
    step = z["node_step"][keep]
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": step})
    d = d[keep]
    del z, r, rn, vn
    h = lab.load(path, ("hit_step", "hit_v", "hit_v_pre", "hit_normal"))
    m = int(max(step.max(), h["hit_step"].max())) + 1
    n_blk = m // block
    ok_n = step < n_blk * block
    blk_n = step // block
    ok_h = h["hit_step"] < n_blk * block
    blk_h = h["hit_step"][ok_h] // block
    hits = np.bincount(blk_h, minlength=n_blk) / block
    imp = np.abs(np.sum((h["hit_v_pre"] - h["hit_v"])[ok_h] * h["hit_normal"][ok_h], axis=1))
    impulse = np.bincount(blk_h, weights=imp, minlength=n_blk) / block
    near = ok_n & (d < D_NEAR)
    layer = np.bincount(blk_n[near], minlength=n_blk) / block
    sl = ok_n & (d < D_SLOPE)
    slope, slope_all = _block_slope(d[sl], lam[sl], blk_n[sl], n_blk)
    return {
        "steps": m, "block": block,
        "hits": (hits / hits.mean()).tolist(),
        "impulse": (impulse / impulse.mean()).tolist(),
        "layer": (layer / layer.mean()).tolist(),
        "slope": (slope / slope_all).tolist(),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["drag", "heat", "spec", "kn10", "hot",
                                                   "m25_diff", "m25_spec", "m03_diff",
                                                   "m03_spec", "drag_g2"])  # fmt: skip
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--block", type=int, default=25)
    args = ap.parse_args()
    out = lab.OUT / f"steady_b{args.block}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for case in args.cases:
        files = lab.seed_files(lab.CASES[case])[: args.seeds]
        rows = []
        for seed, path in files:
            rows.append({"seed": seed, **seed_rows(path, args.block)})
            r = rows[-1]
            print(f"{case} seed {seed}: {r['steps']} steps; block ranges hits "
                  f"{np.ptp(r['hits']):.3f} impulse {np.ptp(r['impulse']):.3f} layer "
                  f"{np.ptp(r['layer']):.3f} slope {np.nanmin(r['slope']):.2f}-"
                  f"{np.nanmax(r['slope']):.2f}", flush=True)  # fmt: skip
        res[case] = rows
        out.write_text(json.dumps(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
