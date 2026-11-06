"""The near-singular kNN fits, visualised (2026-11-05): one seed's kNN-64 local fits computed
twice with the same neighbours and two equivalent input scalings -- each feature over its range
among the hit's 64 neighbours (A), and in fixed physical units (B: d / 1 cm, s / 10 cm,
v / V_REF).  Least squares is invariant to rescaling its columns; where A and B disagree the
data do not determine the slope and the tiny ridge does.

``examples/mesh/figures/fitlab/knn_singular.png``: (a) the per-hit wall-normal slope under A
against B, by the design's condition number; (b) where the disagreeing hits sit (angle from
stagnation, wall distance); (c) how concentrated the cos 2 theta pathwise sum is in a few
hits; (d) one ill-determined hit's neighbours in (d, v_n), coloured by lambda.

    PYTHONPATH=. python3 scratch/fitlab/singular.py [--case drag] [--seed 0]
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, "scratch/fitlab")
import figures as FG  # noqa: E402
import knn  # noqa: E402
import lab  # noqa: E402

K = 64
METRIC = "d0.03_s0.1_v0.5"
FIXED = np.array([0.01, 0.1, 1.0, 1.0, 1.0])  # scaling B: d, s in m; v in V_REF
RIDGE = 1e-9


def fits(nc, lam, hc, idx):
    """Per hit: g_d under scalings A and B, the condition number of A's design."""
    n = hc.shape[0]
    g_a, g_b, cond = np.zeros(n), np.zeros(n), np.zeros(n)
    for a in range(0, n, 2048):
        b = min(a + 2048, n)
        rel = nc[idx[a:b]][:, :, :5] - hc[a:b, None, :5]
        y = lam[idx[a:b]]
        for scale, out in ((np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-12), g_a),
                           (np.broadcast_to(FIXED, (b - a, 1, 5)), g_b)):  # fmt: skip
            f = np.concatenate((np.ones(rel.shape[:2] + (1,)), rel / scale), axis=2)
            ata = np.einsum("qki,qkj->qij", f, f)
            beta = np.linalg.solve(ata + RIDGE * K * np.eye(6)[None],
                                   np.einsum("qki,qk->qi", f, y)[..., None])[..., 0]  # fmt: skip
            out[a:b] = beta[:, 1] / scale[:, 0, 0]
            if out is g_a:
                cond[a:b] = np.linalg.cond(ata)
    return g_a, g_b, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = dict(lab.seed_files(lab.CASES[args.case]))[args.seed]
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn.D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    slot = z["node_slot"][keep]
    nc = nc[keep]
    del z
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    m = knn.parse_metric(METRIC)
    _d, idx = cKDTree(knn.scaled(nc, m)).query(knn.scaled(hc, m), k=K + 1, workers=-1)
    own = nkey[idx] == hkey[:, None]
    idx = np.take_along_axis(idx, np.argsort(own, axis=1, kind="stable"), axis=1)[:, :K]
    g_a, g_b, cond = fits(nc, lam, hc, idx)
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    w = np.sum(n_hat * h["hit_dx"][meta["controls"].index("mode2")], axis=1)
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (hth - np.pi)))))
    diff = np.abs(g_a - g_b)
    typical = np.median(np.abs(g_a))
    bad = diff > typical
    print(f"{args.case} seed {args.seed}: {len(hc)} hits; |A - B| > median |g_d| at "
          f"{bad.sum()} ({100 * bad.mean():.1f} %); cos2 normal-part sum A {np.sum(g_a * w):+.4g} "
          f"B {np.sum(g_b * w):+.4g}; cond quantiles 50/99/99.9 "
          f"{np.percentile(cond, [50, 99, 99.9]).round(0)}")  # fmt: skip

    unit = 1e-6
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=FG.SURF)
    ax = axes[0, 0]
    FG._style(ax)
    o = np.argsort(cond)
    sc = ax.scatter(g_b[o] / unit, g_a[o] / unit, c=np.log10(np.minimum(cond[o], 1e12)), s=4,
                    cmap="viridis", lw=0)  # fmt: skip
    lim = np.percentile(np.abs(np.concatenate((g_a, g_b))), 99.9) / unit
    ax.plot([-lim, lim], [-lim, lim], color=FG.INK2, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(
        r"$\partial\lambda/\partial d$, scaling B: fixed units ($10^{-6}$/m)", color=FG.INK
    )
    ax.set_ylabel(r"scaling A: each feature's range ($10^{-6}$/m)", color=FG.INK)
    ax.set_title("(a) the same fit, two equivalent scalings", color=FG.INK, fontsize=10, loc="left")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10 condition number (A)", color=FG.INK)

    ax = axes[0, 1]
    FG._style(ax)
    ax.scatter(ang[~bad], 1000 * hc[~bad, 0], s=1, color=FG.GRID, lw=0, label="agree")
    ax.scatter(ang[bad], 1000 * hc[bad, 0], s=6, color=FG.ORANGE, lw=0,
               label="|A - B| > median |slope|")  # fmt: skip
    ax.set_xlabel("angle from stagnation (deg)", color=FG.INK)
    ax.set_ylabel("hit wall distance (mm)", color=FG.INK)
    ax.set_ylim(0, 15)
    ax.set_title("(b) where the slope is not determined", color=FG.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=FG.INK, markerscale=3)

    ax = axes[1, 0]
    FG._style(ax)
    for g, colour, name in ((g_a, FG.BLUE, "A"), (g_b, FG.ORANGE, "B")):
        c = g * w
        share = np.cumsum(np.abs(c)[np.argsort(-np.abs(c))]) / np.sum(np.abs(c))
        frac = np.arange(1, c.size + 1) / c.size
        ax.plot(100 * frac, share, color=colour, lw=2,
                label=f"scaling {name}: sum {np.sum(c):+.3g}")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_xlabel("% of hits, largest |contribution| first", color=FG.INK)
    ax.set_ylabel("share of sum |g_d dx.n| (cos 2 theta)", color=FG.INK)
    ax.set_title("(c) a few hits carry the sum", color=FG.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=FG.INK)

    ax = axes[1, 1]
    FG._style(ax)
    worst = int(np.argmax(np.where(bad, np.abs((g_a - g_b) * w), 0.0)))
    nb = idx[worst]
    sc = ax.scatter(1000 * nc[nb, 0], nc[nb, 2], c=lam[nb] / unit, s=28, cmap="coolwarm",
                    edgecolors=FG.SURF, linewidths=0.6)  # fmt: skip
    ax.scatter([1000 * hc[worst, 0]], [hc[worst, 2]], marker="*", s=220, color=FG.INK,
               label="the hit")  # fmt: skip
    r = np.corrcoef(nc[nb, 0], nc[nb, 2])[0, 1]
    ax.set_xlabel("neighbour wall distance d (mm)", color=FG.INK)
    ax.set_ylabel(r"neighbour $v_n$ / V_REF", color=FG.INK)
    same = int(np.sum(slot[nb] == h["hit_slot"][worst]))
    ax.set_title(f"(d) one ill-determined hit ({ang[worst]:.0f} deg): {same} of its {K} neighbours "
                 f"are its own particle\nat other steps; corr(d, v_n) = {r:.2f}; slope A "
                 f"{g_a[worst] / unit:+.3g}, B {g_b[worst] / unit:+.3g}",
                 color=FG.INK, fontsize=9.5, loc="left")  # fmt: skip
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\lambda$ (demeaned, $10^{-6}$)", color=FG.INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=FG.INK, loc="lower right")
    fig.suptitle(f"{args.case} seed {args.seed}: kNN-64 local fits where the data do not fix the "
                 "wall slope", color=FG.INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    out = FG.FIG / "knn_singular.png"
    fig.savefig(out, dpi=150, facecolor=FG.SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
