"""kNN wall gradients that transfer across conditions (2026-11-05 night): the M 10-tuned kNN
(knn2: absolute metric d / 3 cm, s / 10 cm, v / 0.5 V_REF) fixed the M 10 Kn 0.5 diffuse bias
but broke M 0.32 diffuse (radius 0.30: pathwise_x -0.0020 +- 0.0022 per seed where production's
+0.0003 closes) and worsened M 25 specular.  Two paths, each swept:

(a) population-relative metric ``P<cd>_<cs>_<cv>``: wall distance over ``cd`` x the case's
    median post-hit wall distance, arc length over ``cs`` x R, the velocity over ``cv`` x the
    thermal spread of the outgoing near-wall gas (the rms of its velocity components about
    their mean, within 3 median hit distances of the wall) -- no M 10 scale anywhere;
(b) shrinkage toward production: per hit and component (wall-normal, tangential)
    ``g = g_prod + (g_knn - g_prod) D^2 / (D^2 + tau V)``, ``D = g_knn - g_prod``, ``V`` the
    kNN slope's variance from its own regression (residual variance x the inverse design):
    kNN where it differs from production by more than its noise, production elsewhere.

And the M 0.32 mechanism test: ``own`` 0 keeps the hit's own particle's other steps among the
neighbours (knn2's u0), 1 drops them (every other particle's repeated samples kept).

Fits ``k3_<metric>_o<own>`` (raw), ``..._sh<tau>`` (shrunk per hit) and ``..._rsh<tau>``
(shrunk per 10 degree sector: the sector's mean difference against its mean's variance); closure rows in the lab format
(``out/knn3_<case><tag>.json``), scored by varsum.py / the per-seed error script.

    PYTHONPATH=. python3 scratch/fitlab/knn3.py --case drag [--metrics ...] [--taus 1 4]
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scratch/fitlab")
sys.path.insert(0, "scratch/fitgate")
import knn  # noqa: E402
import lab  # noqa: E402
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

OMEGA = 0.81  # argon's VHS viscosity exponent: lambda ~ T^(omega - 1/2) / n
MFP_BAND = 0.02  # m: the wall layer whose density and temperature set lambda_wall
MFP_SECTORS = 36

D_MAX = 0.6
K = 64
CHUNK = 1024
RIDGE = 1e-4
GUARD = 1e3
PROD = "cell_along_pool_plane"
SECTORS = 36  # the regional shrinkage's wall sectors (10 degrees)
METRICS = ("d0.03_s0.1_v0.5", "P3_0.033_0.3", "P10_0.033_0.3", "P30_0.033_0.3",
           "P3_0.033_1", "P10_0.033_1", "P30_0.033_1")  # fmt: skip


def wall_mfp(case, feat_n, steps, hth):
    """The hit-weighted local mean free path at the wall (2026-11-06): per 10 degree sector,
    lambda_inf (n_inf / n_wall) (T_wall / T_inf)^(omega - 1/2) from the outgoing nodes within
    ``MFP_BAND`` of the wall (their count x 2 for both directions; T from their thermal rms in
    V_REF = sqrt(k T_inf / m) units), weighted by the sector's hits.  Independent of dt."""
    cfg = V.gate_config(lab.CASES[case])[0]
    k = int(cfg.get("grid_scale", 1))
    n_inf = EM.F.NRHO / (EM.F.CASES[cfg["case"]]["fnum"] * cfg["fnum_scale"]) * k**2
    lam_inf = EM.mean_free_path(EM.F.NRHO * EM.knudsen_density(cfg))
    sec_n = np.floor((feat_n[:, 1] / lab.RADIUS + np.pi) / (2 * np.pi) * MFP_SECTORS).astype(int)
    sec_n %= MFP_SECTORS
    sec_h = np.floor((np.angle(np.exp(1j * (hth - np.pi))) + np.pi) / (2 * np.pi)
                     * MFP_SECTORS).astype(int) % MFP_SECTORS  # fmt: skip
    band = feat_n[:, 0] < MFP_BAND
    lam = np.full(MFP_SECTORS, np.nan)
    for j in range(MFP_SECTORS):
        sel = band & (sec_n == j)
        if sel.sum() < 50:
            continue
        area = 2 * np.pi / MFP_SECTORS * (lab.RADIUS + MFP_BAND / 2) * MFP_BAND
        n_wall = 2 * sel.sum() / steps / area
        t_wall = float(np.mean(np.var(feat_n[sel, 2:5], axis=0)))
        lam[j] = lam_inf * (n_inf / n_wall) * max(t_wall, 1e-3) ** (OMEGA - 0.5)
    w = np.bincount(sec_h, minlength=MFP_SECTORS).astype(float)
    ok = np.isfinite(lam) & (w > 0)
    # the hit-weighted median sector value
    order = np.argsort(lam[ok])
    cum = np.cumsum(w[ok][order]) / w[ok].sum()
    cell = (
        float(
            np.min(
                (
                    np.asarray(EM.F.CASES[cfg["case"]]["hi"])
                    - np.asarray(EM.F.CASES[cfg["case"]]["lo"])
                )
                / np.asarray(EM.F.CASES[cfg["case"]]["n"])
            )
        )
        / k
    )
    return float(lam[ok][order][np.searchsorted(cum, 0.5)]), cell


def scales(metric, d_hit, sigma_v, mfp=None, cell=None):
    """``f64[5]``: the metric's scales of (d, s, v_n, v_t, |v_z|).  ``M<c>_<cs>_<cv>``: wall
    distance over ``c`` x the local wall mean free path, the rest as ``P``."""
    if metric.startswith("C"):
        # min(c x the wall mfp, c_cell x the cell): the Knudsen layer where it is thinner than
        # the grid, never smoother than a fraction of a cell where the gas is rarefied
        cm, cc, cs, cv = (float(x) for x in metric[1:].split("_"))
        d_sc = min(cm * mfp, cc * cell)
        return np.array([d_sc, cs * lab.RADIUS, cv * sigma_v, cv * sigma_v, cv * sigma_v])
    if metric.startswith("M"):
        cm, cs, cv = (float(x) for x in metric[1:].split("_"))
        return np.array([cm * mfp, cs * lab.RADIUS, cv * sigma_v, cv * sigma_v, cv * sigma_v])
    if metric.startswith("P"):
        cd, cs, cv = (float(x) for x in metric[1:].split("_"))
        return np.array([cd * d_hit, cs * lab.RADIUS, cv * sigma_v, cv * sigma_v, cv * sigma_v])
    m = knn.parse_metric(metric)
    return np.array([m["d"], m["s"], m["v"], m["v"], m["v"]])


def fit_with_variance(feat_n, lam, feat_q, idx):
    """``(g [H, 2], var [H, 2])``: the local linear fit's (g_d, g_s) and their variances
    (box weights, features scaled by their range, relative ridge, condition guard -> the
    (1, d, s) fit), as knn2's winning fit."""
    h = feat_q.shape[0]
    g = np.zeros((h, 2))
    var = np.zeros((h, 2))
    for a in range(0, h, CHUNK):
        b = min(a + CHUNK, h)
        nb = idx[a:b]
        rel = feat_n[nb] - feat_q[a:b, None, :]
        sc = np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-30)
        x = np.concatenate((np.ones((*rel.shape[:2], 1)), rel / sc), axis=2)
        y = lam[nb]
        ata = np.einsum("qki,qkj->qij", x, x)
        diag = np.trace(ata, axis1=1, axis2=2) / 6.0
        ata = ata + RIDGE * diag[:, None, None] * np.eye(6)[None]
        bad = np.linalg.cond(ata) > GUARD
        # the guarded rows: the (1, d, s) fit, the other columns' rows / columns made identity
        keep = np.ones((b - a, 6), bool)
        keep[bad, 3:] = False
        mask = keep[:, :, None] & keep[:, None, :]
        a_eff = np.where(mask, ata, np.eye(6)[None])
        aty = np.einsum("qki,qk->qi", x, y) * keep
        inv = np.linalg.inv(a_eff) * mask
        beta = np.einsum("qij,qj->qi", inv, aty)
        resid = y - np.einsum("qki,qi->qk", x * keep[:, None, :], beta)
        dof = np.maximum(K - keep.sum(axis=1), 1)
        s2 = np.sum(resid**2, axis=1) / dof
        g[a:b] = beta[:, 1:3] / sc[:, 0, :2]
        var[a:b] = s2[:, None] * inv[:, [1, 2], [1, 2]] / sc[:, 0, :2] ** 2
    return g, var


def run_seed(case, seed, path, metrics, owns, taus):
    t0 = time.time()
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    nslot = z["node_slot"][keep]
    feat_n = nc[keep][:, :5]
    del z, nc
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    feat_q = hc[:, :5]
    hkey = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    d_hit = float(np.median(hc[:, 0]))
    near = feat_n[:, 0] < 3 * d_hit
    sigma_v = float(np.sqrt(np.mean(np.var(feat_n[near, 2:5], axis=0))))
    mfp, cell = (
        wall_mfp(case, feat_n, len(np.unique(h["hit_step"])), hth)
        if any(m.startswith(("M", "C")) for m in metrics)
        else (None, None)
    )
    fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    gp = fits[PROD]
    gp_nt = np.stack((np.sum(gp * n_hat, axis=1), np.sum(gp * t_hat, axis=1)), axis=1)
    diag = {"d_hit_median_m": d_hit, "sigma_v_vref": sigma_v, "wall_mfp_m": mfp, "cell_m": cell}
    for metric in metrics:
        sc = scales(metric, d_hit, sigma_v, mfp, cell)
        _d, idx = cKDTree(feat_n / sc).query(feat_q / sc, k=K + 64, workers=-1)
        for own in owns:
            drop = nslot[idx] == h["hit_slot"][:, None] if own else nkey[idx] == hkey[:, None]
            order = np.argsort(drop, axis=1, kind="stable")
            ii = np.take_along_axis(idx, order, axis=1)[:, :K]
            short = np.mean(np.take_along_axis(drop, order, axis=1)[:, K - 1])
            g, var = fit_with_variance(feat_n, lam, feat_q, ii)
            base = f"k3_{metric}_o{own}"
            fits[base] = g[:, :1] * n_hat + g[:, 1:2] * t_hat
            # regional shrinkage: per 10 degree wall sector, the sector's mean difference
            # against the variance of that mean (per-hit noise averages out over the hits)
            sector = np.floor((hth + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
            cnt = np.bincount(sector, minlength=SECTORS).astype(float)
            dlt_all = g - gp_nt
            mean_d = np.stack([np.bincount(sector, weights=dlt_all[:, j], minlength=SECTORS)
                               for j in range(2)], axis=1) / np.maximum(cnt, 1)[:, None]  # fmt: skip
            mean_v = np.stack([np.bincount(sector, weights=var[:, j], minlength=SECTORS)
                               for j in range(2)], axis=1) / np.maximum(cnt, 1)[:, None] ** 2  # fmt: skip
            for tau in taus:
                w_r = mean_d**2 / np.maximum(mean_d**2 + tau * mean_v, 1e-300)
                gs = gp_nt + w_r[sector] * dlt_all
                fits[f"{base}_rsh{tau:g}"] = gs[:, :1] * n_hat + gs[:, 1:2] * t_hat
                diag[f"{base}_rsh{tau:g}"] = {"w_normal_hitmean": float(np.mean(w_r[sector, 0]))}
            for tau in taus:
                dlt = g - gp_nt
                w = dlt**2 / np.maximum(dlt**2 + tau * var, 1e-300)
                gs = gp_nt + w * dlt
                fits[f"{base}_sh{tau:g}"] = gs[:, :1] * n_hat + gs[:, 1:2] * t_hat
                diag[f"{base}_sh{tau:g}"] = {"w_normal_median": float(np.median(w[:, 0])),
                                             "w_normal_mean": float(np.mean(w[:, 0]))}  # fmt: skip
            span = np.ptp(feat_n[ii, 0], axis=1)
            diag[base] = {"span_median_mm": float(1000 * np.median(span)),
                          "own_short": float(short)}  # fmt: skip
        print(f"  {case} seed {seed} {metric}: d_hit {1000 * d_hit:.1f} mm sigma_v "
              f"{sigma_v:.2f} V_REF  ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    px = {c: {name: float(np.sum(gg * h["hit_dx"][i])) for name, gg in fits.items()}
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
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--owns", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--d-max",
        type=float,
        default=0.6,
        help="node band, m (0.15 for the refined grids' 4-16x particles)",
    )
    args = ap.parse_args()
    global D_MAX  # noqa: PLW0603 -- the scratch run's node band
    D_MAX = args.d_max
    rows = []
    for seed, path in lab.seed_files(lab.CASES[args.case]):
        if args.seeds[0] <= seed < args.seeds[1]:
            rows.append(run_seed(args.case, seed, path, args.metrics, args.owns, args.taus))
    out = lab.OUT / f"knn3_{args.case}{args.tag}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
