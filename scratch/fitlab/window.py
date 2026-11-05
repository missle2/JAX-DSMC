"""Running-sum window wall gradients (2026-11-05): kNN's local linear fit without a neighbour
search, the production candidate.

The outgoing nodes (v_n > 0, d < D_MAX, every captured step of the seed) are grouped by an
arc-length sector of the wall (``--sector`` m) x a wall-frame velocity box (``--dv`` V_REF on
v_n, v_t, |v_z|) and sorted by wall distance d inside each group, once.  The regression's
sufficient statistics -- the outer products of the features f = (1, d, s, v_n, v_t, |v_z|)
(fixed scales) and f lambda -- are prefix-summed along that order.  A hit's window is the k
nodes of its group nearest in d (a fixed-count window, clipped at the group's ends: mm wide in
the dense nose layer, wide where the nodes are sparse); its normal equations are the
difference of two prefix sums (the hit's own node subtracted), one small solve gives
grad_x = g_d n + g_s t.  Cost: one sort, cumulative sums, two binary searches and a 6 x 6 solve
per hit -- static-shape work, the same order as the per-cell fit.

Closure rows in the lab's format (``out/win_<case><tag>.json``; fits
``win_w<width>_k<k>_s<sector>_v<dv>``; the window is the larger of the k nearest by d and
``[d - w/2, d + w/2]`` shifted off the wall), scored by ``summary.py``.

    PYTHONPATH=. python3 scratch/fitlab/window.py --case drag [--ks 32 64 256]
        [--sectors 0.1 0.2] [--dvs 0.5 1.0]
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import lab  # noqa: E402

D_MAX = 0.6
SCALES = np.array([1.0, 0.03, 0.1, 0.5, 0.5, 0.5])  # 1, d, s, v_n, v_t, |v_z|: kNN's metric
NF = 6
RIDGE = 1e-8
MIN_NODES = 12
BLOCK = 0  # time block of the groups, steps (0: pooled over the whole captured window)
SPEED_FLOOR = 0.05  # V_REF: the speed-relative boxes' lowest edge
FALLBACK = "cell_pool_plane"
IU = np.triu_indices(NF)


def group_ids(c, sector, dv):
    """:func:`_group_ids` with the time block (``BLOCK`` steps, 0: the whole window)."""
    gid = _group_ids(c, sector, dv)
    if BLOCK > 0:
        gid = gid * 1000 + (c[:, 5] // BLOCK).astype(np.int64)
    return gid


def _group_ids(c, sector, dv):
    """Group of each row of ``coordinates`` (sector x velocity box: ``dv > 0`` boxes of dv V_REF,
    ``dv = 0`` production's speed / angle bins, ``dv < 0`` speed-relative boxes of -dv), int64."""
    span = 2 * np.pi * lab.RADIUS
    i_s = np.floor((c[:, 1] + 0.5 * span) / sector).astype(np.int64)
    if dv == 0:
        # production's velocity bins (lab.SETTINGS): speed x v_n / |v_nt| cosine x sign v_t,
        # wider at speed -- the specular wall's fast re-emissions stay populated
        speed = np.sqrt(np.sum(c[:, 2:5] ** 2, axis=1))
        sb = np.searchsorted(lab.SETTINGS["speed"], speed, side="right") - 1
        cos = c[:, 2] / np.maximum(np.hypot(c[:, 2], c[:, 3]), 1e-12)
        cb = np.clip(np.searchsorted(lab.SETTINGS["cosine"], cos, side="right") - 1, 0, None)
        return ((i_s * 100 + sb) * 10 + cb) * 2 + (c[:, 3] > 0)
    if dv < 0:
        # speed-relative boxes, no velocity scale: log speed and the unit vector's v_t / |v|,
        # |v_z| / |v| in steps of r = -dv -- v_n varies by ~r of itself inside a box at any
        # Mach number or wall temperature (d = v_n tau: the d / v_n collinearity stays bounded)
        r = -dv
        speed = np.maximum(np.sqrt(np.sum(c[:, 2:5] ** 2, axis=1)), 1e-12)
        sb = np.floor(np.log(np.maximum(speed, SPEED_FLOOR) / SPEED_FLOOR) / np.log1p(r))
        ut = np.floor(c[:, 3] / speed / r) + 1000
        uz = np.floor(c[:, 4] / speed / r)
        return ((i_s * 1000 + sb.astype(np.int64)) * 2000 + ut.astype(np.int64)) * 1000 + uz.astype(
            np.int64
        )
    i_n = np.clip(np.floor(c[:, 2] / dv).astype(np.int64), 0, 999)
    i_t = np.clip(np.floor(c[:, 3] / dv).astype(np.int64) + 1000, 0, 1999)
    i_z = np.clip(np.floor(c[:, 4] / dv).astype(np.int64), 0, 999)
    return ((i_s * 1000 + i_n) * 2000 + i_t) * 1000 + i_z


def features(c):
    return np.concatenate((np.ones((c.shape[0], 1)), c[:, :5]), axis=1) / SCALES[None, :]


def moments(f, lam):
    """Per-row upper-triangle outer products and f lambda: (n, 21 + 6)."""
    outer = f[:, IU[0]] * f[:, IU[1]]
    return np.concatenate((outer, f * lam[:, None]), axis=1)


def solve(m):
    """Coefficients from summed moments (h, 27); the d and s slopes in physical units."""
    h = m.shape[0]
    a = np.zeros((h, NF, NF))
    a[:, IU[0], IU[1]] = m[:, : len(IU[0])]
    a[:, IU[1], IU[0]] = m[:, : len(IU[0])]
    b = m[:, len(IU[0]) :]
    a += RIDGE * np.maximum(a[:, :1, :1], 1.0) * np.eye(NF)[None]
    beta = np.linalg.solve(a, b[..., None])[..., 0]
    return beta[:, 1:3] / SCALES[None, 1:3]


def window_gradients(nc, lam, nkey, hc, hkey, ks, widths, sector, dv, dt=0.0):
    gid = group_ids(nc, sector, dv)
    order = np.lexsort((nc[:, 0], gid))
    gs, ds = gid[order], nc[order, 0]
    cum = np.zeros((order.size + 1, len(IU[0]) + NF))
    np.cumsum(moments(features(nc[order]), lam[order]), axis=0, out=cum[1:])
    # group bounds and the hit's rank by d inside its group
    hg = group_ids(hc, sector, dv)
    start = np.searchsorted(gs, hg, side="left")
    end = np.searchsorted(gs, hg, side="right")
    # rank by d within the group: one binary search on (dense group rank + d / D_MAX)
    ug, grank = np.unique(gs, return_inverse=True)
    comb = grank + 0.999 * ds / D_MAX
    hr = np.clip(np.searchsorted(ug, hg), 0, ug.size - 1)
    pos = np.clip(np.searchsorted(comb, hr + 0.999 * np.minimum(hc[:, 0], D_MAX) / D_MAX), start,
                  end)  # fmt: skip
    # the hit's own node (same step and slot), if it is among the nodes
    sk = np.argsort(nkey)
    j = np.clip(np.searchsorted(nkey[sk], hkey), 0, sk.size - 1)
    own_row = np.where(nkey[sk][j] == hkey, sk[j], -1)
    inv = np.empty(order.size, np.int64)
    inv[order] = np.arange(order.size)
    own_pos = np.where(own_row >= 0, inv[np.maximum(own_row, 0)], -1)
    out = {}
    for w in widths:
        # the width floor: [d - w/2, d + w/2] shifted off the wall to [0, w]; w < 0: -w of the
        # hit's own flight length per step, v_n dt (scale-free)
        w_hit = -w * hc[:, 2] * lab.V_REF * dt if w < 0 else np.full(hc.shape[0], w)
        a = np.maximum(hc[:, 0] - 0.5 * w_hit, 0.0)
        b = np.minimum(a + w_hit, D_MAX)
        lo_w = np.clip(np.searchsorted(comb, hr + 0.999 * a / D_MAX), start, end)
        hi_w = np.clip(np.searchsorted(comb, hr + 0.999 * b / D_MAX, side="right"), start, end)
        for k in ks:
            # the count floor: the k + 1 nearest by rank (the own node is taken back out)
            lo = np.maximum(start, pos - k // 2)
            hi = np.minimum(end, lo + k + 1)
            lo = np.maximum(start, hi - k - 1)
            if w != 0:
                lo, hi = np.minimum(lo, lo_w), np.maximum(hi, hi_w)
            m = cum[hi] - cum[lo]
            inside = (own_pos >= lo) & (own_pos < hi)
            m -= np.where(inside[:, None], cum[own_pos + 1] - cum[own_pos], 0.0)
            out[(k, w)] = (solve(m), hi - lo - inside)
    return out


def run_seed(case, seed, path, ks, widths, sectors, dvs):
    t0 = time.time()
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    nc = nc[keep]
    del z
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    for sector in sectors:
        for dv in dvs:
            t1 = time.time()
            res = window_gradients(nc, lam, nkey, hc, hkey, ks, widths, sector, dv, meta["dt"])
            for (k, w), (g, count) in res.items():
                grad = g[:, :1] * n_hat + g[:, 1:2] * t_hat
                # too few nodes for the 6-feature fit: production's cell_pool gradient
                grad = np.where((count >= MIN_NODES)[:, None], grad, fits[FALLBACK])
                fits[f"win_w{w:g}_k{k}_s{sector:g}_v{dv:g}"] = grad
            short = np.mean(res[(max(ks), widths[0])][1] < max(ks))
            print(f"  {case} seed {seed} sector {sector:g} dv {dv:g}: {time.time() - t1:.1f}s; "
                  f"hits whose group has < {max(ks)} nodes {100 * short:.0f} %", flush=True)  # fmt: skip
    px = {c: {name: float(np.sum(g * h["hit_dx"][i])) for name, g in fits.items()}
          for i, c in enumerate(meta["controls"])}  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        totals[c] = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
    print(f"  {case} seed {seed} done {time.time() - t0:.0f}s", flush=True)
    return {"seed": seed, "pathwise_x": px, "base_total": totals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 256])
    ap.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.03],
        help="window width floors in d, m (0: the fixed count alone)",
    )
    ap.add_argument("--sectors", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--dvs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    for seed, path in lab.seed_files(lab.CASES[args.case]):
        if args.seeds[0] <= seed < args.seeds[1]:
            rows.append(
                run_seed(args.case, seed, path, args.ks, args.widths, args.sectors, args.dvs)
            )
    out = lab.OUT / f"win_{args.case}{args.tag}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
