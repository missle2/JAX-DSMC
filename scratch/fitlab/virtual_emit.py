"""Virtual diffuse re-emissions (2026-11-08): the wall analogue of the virtual collision
pull-back, prototyped offline on the fit lab's dumps.

At a diffuse hit the re-emitted velocity g is a single draw of the wall kernel (the flux
Maxwellian at T_w about the hit normal).  Three wall terms evaluate a fitted field at that one
draw, so their per-hit noise includes the kernel sampling:

* pathwise_x: grad_x lam(x_out(g), g) . (v_in - g) dt_h/dtheta, with x_out(g) = x_h + t_rem g
  (the dumps' d x_post / d theta is exactly (v_in - g) dt_h/dtheta: one scalar per control);
* order: (c_in - c_out(g)) / dt . dt_h/dtheta, with c = dt L* lambda fitted at the hit point on
  the approaching side (v_in) and the leaving side (g).

The virtual form replaces each by its mean over K fresh draws of the kernel on the SAME fit
(the tower property: unbiased given the fit, as the virtual partners are given the field).  The
fit is knn3's chosen kNN (d / min(mfp, cell / 4), v / 0.3 thermal, ridge + guard, own (step,
slot) dropped) of the demeaned lambda (pathwise) and of the per-node increment c (order, the
value of a local linear fit at the hit point, nodes pooled over the window).

Per seed and control: pathwise_x and order for the realised draw and for K virtual draws, so the
script reports the per-seed sd each estimator leaves and the closure of the totals with the
production terms swapped out.

    PYTHONPATH=. python3 scratch/fitlab/virtual_emit.py --case drag [--seeds 0 8] [--k 8]
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
import knn3  # noqa: E402
import lab  # noqa: E402
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

METRIC = "C1_0.25_0.033_0.3"  # knn3's chosen metric
K_NB = knn3.K
EXTRA = 64  # neighbours queried beyond K_NB so the own particle's nodes can be dropped
CHUNK = 1024
RIDGE = knn3.RIDGE
GUARD = knn3.GUARD


def local_fit(feat_n, y, feat_q, idx):
    """``(value [H], g [H, 2])``: the local linear fit of ``y`` (box weights, features scaled
    by their range, relative ridge, condition guard -> the (1, d, s) fit) at each query, its
    intercept and (d, s) slopes -- knn3.fit_with_variance with the intercept kept."""
    h = feat_q.shape[0]
    val = np.zeros(h)
    g = np.zeros((h, 2))
    for a in range(0, h, CHUNK):
        b = min(a + CHUNK, h)
        nb = idx[a:b]
        rel = feat_n[nb] - feat_q[a:b, None, :]
        sc = np.maximum(np.abs(rel).max(axis=1, keepdims=True), 1e-30)
        x = np.concatenate((np.ones((*rel.shape[:2], 1)), rel / sc), axis=2)
        ata = np.einsum("qki,qkj->qij", x, x)
        diag = np.trace(ata, axis1=1, axis2=2) / 6.0
        ata = ata + RIDGE * diag[:, None, None] * np.eye(6)[None]
        bad = np.linalg.cond(ata) > GUARD
        keep = np.ones((b - a, 6), bool)
        keep[bad, 3:] = False
        mask = keep[:, :, None] & keep[:, None, :]
        a_eff = np.where(mask, ata, np.eye(6)[None])
        aty = np.einsum("qki,qk->qi", x, y[nb]) * keep
        beta = np.einsum("qij,qj->qi", np.linalg.inv(a_eff) * mask, aty)
        val[a:b] = beta[:, 0]
        g[a:b] = beta[:, 1:3] / sc[:, 0, :2]
    return val, g


def neighbours(tree, zq, own_key, node_key):
    """``K_NB`` nearest node indices per scaled query, the query's own particle-step dropped."""
    _d, idx = tree.query(zq, k=K_NB + EXTRA, workers=-1)
    drop = node_key[idx] == own_key[:, None]
    order = np.argsort(drop, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :K_NB]


def wall_draws(rng, n_hat, s_wall):
    """``f64[H, 3]`` velocities from the diffuse kernel (flux Maxwellian of spread ``s_wall``
    about the hit normals ``n_hat`` [H, 2], wall at rest), laboratory frame."""
    h = n_hat.shape[0]
    vn = s_wall * np.sqrt(-2.0 * np.log(1.0 - rng.random(h)))
    vt = s_wall * rng.standard_normal(h)
    vz = s_wall * rng.standard_normal(h)
    t_hat = np.stack((-n_hat[:, 1], n_hat[:, 0]), axis=1)
    g2 = vn[:, None] * n_hat + vt[:, None] * t_hat
    return np.column_stack((g2, vz))


def remaining_flight(x_post, g):
    """``t_rem`` with ``|x_post - t_rem g - C| = R`` (the flight since the hit; smaller root)."""
    r = x_post - lab.CENTRE
    gg = g[:, :2]
    a = np.sum(gg * gg, axis=1)
    b = -2.0 * np.sum(r * gg, axis=1)
    c = np.sum(r * r, axis=1) - lab.RADIUS**2
    return (-b - np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)


class Fields:
    """The seed's two kNN fields: lambda on the leaving nodes, c on both sides."""

    def __init__(self, case, path, n_steps, timed=0.0):
        z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam",
                            "node_coll"))  # fmt: skip
        nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
        near = nc[:, 0] < knn3.D_MAX
        out = near & (nc[:, 2] > 0)
        inc = near & (nc[:, 2] < 0)
        key = z["node_step"].astype(np.int64) * 1_000_000 + z["node_slot"]
        self.feat_out = nc[out][:, :5]
        self.feat_in = nc[inc][:, :5]
        self.key_out, self.key_in = key[out], key[inc]
        self.lam = lab.demeaned({"node_lam": z["node_lam"][out], "node_step": z["node_step"][out]})
        self.step_out = z["node_step"][out].astype(float)
        self.step_in = z["node_step"][inc].astype(float)
        self.c_out = z["node_coll"][out].astype(float)
        self.c_in = z["node_coll"][inc].astype(float)
        del z, nc
        h = lab.load(path, ("hit_x", "hit_v", "hit_step"))
        hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
        d_hit = float(np.median(hc[:, 0]))
        sel = self.feat_out[:, 0] < 3 * d_hit
        sigma_v = float(np.sqrt(np.mean(np.var(self.feat_out[sel, 2:5], axis=0))))
        mfp, cell = knn3.wall_mfp(case, self.feat_out, n_steps, hth)
        self.scales = knn3.scales(METRIC, d_hit, sigma_v, mfp, cell)
        self.tree_out = cKDTree(self.feat_out / self.scales)
        self.tree_in = cKDTree(self.feat_in / self.scales)
        # the c fits' neighbour search with a time coordinate (``timed`` steps; 0 = pooled)
        self.timed = timed
        if timed:
            sc6 = np.append(self.scales, timed)
            self.tree_out_t = cKDTree(np.column_stack((self.feat_out, self.step_out)) / sc6)
            self.tree_in_t = cKDTree(np.column_stack((self.feat_in, self.step_in)) / sc6)

    def query(self, x, v, step, own_key, side):
        """``(value, grad_x [H, 2])`` of lambda (side "lam") or c ("c_out", "c_in")."""
        qc, qth = knn.coordinates(x, v, step)
        fq = qc[:, :5]
        if side == "c_in":
            feat, tree, y, nkey = self.feat_in, self.tree_in, self.c_in, self.key_in
        else:
            feat, tree, y = self.feat_out, self.tree_out, self.lam if side == "lam" else self.c_out
            nkey = self.key_out
        zq = fq / self.scales
        if self.timed and side != "lam":
            tree = self.tree_in_t if side == "c_in" else self.tree_out_t
            zq = np.column_stack((fq, step.astype(float))) / np.append(self.scales, self.timed)
        idx = neighbours(tree, zq, own_key, nkey)
        val, g = local_fit(feat, y, fq, idx)
        n_hat = np.stack((np.cos(qth), np.sin(qth)), axis=1)
        t_hat = np.stack((-np.sin(qth), np.cos(qth)), axis=1)
        return val, g[:, :1] * n_hat + g[:, 1:2] * t_hat


def run_seed(case, seed, path, k_virtual, rng, timed=0.0):
    t0 = time.time()
    meta = json.loads(path.with_suffix(".json").read_text())
    dt = float(meta["dt"])
    cfg = V.gate_config(lab.CASES[case])[0]
    s_wall = lab.V_REF * np.sqrt(cfg["t_wall"] / EM.F.TEMP)
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_v_pre", "hit_normal",
                        "hit_dx"))  # fmt: skip
    n_steps = len(np.unique(h["hit_step"]))
    f = Fields(case, path, n_steps, timed)
    step, own = h["hit_step"], h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    g_real, v_in = h["hit_v"], h["hit_v_pre"]
    n_hat = h["hit_normal"][:, :2] / np.linalg.norm(h["hit_normal"][:, :2], axis=1)[:, None]
    t_rem = remaining_flight(h["hit_x"], g_real)
    x_h = h["hit_x"] - t_rem[:, None] * g_real[:, :2]
    # dt_h / dtheta per control: d x_post / d theta = (v_in - g) dt_h / dtheta exactly
    w = (v_in - g_real)[:, :2]
    dth = np.einsum("chi,hi->ch", h["hit_dx"], w) / np.sum(w * w, axis=1)[None]
    # realised draw
    _v, gx = f.query(h["hit_x"], g_real, step, own, "lam")
    px_real = np.einsum("hi,hi,ch->c", gx, w, dth)
    c_in, _g = f.query(x_h, v_in, step, own, "c_in")
    c_out_real, _g = f.query(x_h, g_real, step, own, "c_out")
    ord_real = np.einsum("h,ch->c", (c_in - c_out_real) / dt, dth)
    # virtual draws: the same fits at K fresh kernel samples per hit
    px_virt = np.zeros(dth.shape[0])
    c_out_virt = np.zeros(len(step))
    for _ in range(k_virtual):
        g = wall_draws(rng, n_hat, s_wall)
        x_out = x_h + t_rem[:, None] * g[:, :2]
        _v, gx = f.query(x_out, g, step, own, "lam")
        px_virt += np.einsum("hi,hi,ch->c", gx, (v_in - g)[:, :2], dth) / k_virtual
        c_val, _g = f.query(x_h, g, step, own, "c_out")
        c_out_virt += c_val / k_virtual
    ord_virt = np.einsum("h,ch->c", (c_in - c_out_virt) / dt, dth)
    # the kernel check: the realised draws' spread against the configured wall temperature
    vt_real = np.sum(g_real[:, :2] * np.stack((-n_hat[:, 1], n_hat[:, 0]), axis=1), axis=1)
    print(f"  {case} seed {seed}: {len(step)} hits, s_wall {s_wall:.0f} m/s (realised v_t rms "
          f"{np.sqrt(np.mean(vt_real**2)):.0f}), K {k_virtual} ({time.time() - t0:.0f}s)",
          flush=True)  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        totals[c] = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
    rows = {}
    for i, c in enumerate(meta["controls"]):
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        rows[c] = {"base_total": totals[c], "prod_px": t["pathwise_x"], "prod_order": t["order"],
                   "px_real": float(px_real[i]), "px_virt": float(px_virt[i]),
                   "order_real": float(ord_real[i]), "order_virt": float(ord_virt[i])}  # fmt: skip
    return {"seed": seed, "k": k_virtual, "controls": rows}


def report(case, rows):
    fds = lab.finite_differences(case)
    print(f"{case}: per-seed mean / sd over {len(rows)} seeds (closure = total / FD mean)")
    for c in ("radius", "mode2", "bump_nose"):
        if c not in fds:
            continue
        fd = np.array(list(fds[c].values()))
        r = [row["controls"][c] for row in rows]
        col = {k: np.array([x[k] for x in r]) for k in r[0]}
        base = col["base_total"]
        variants = {
            "production": base,
            "kNN px (realised)": base - col["prod_px"] + col["px_real"],
            "kNN px (virtual)": base - col["prod_px"] + col["px_virt"],
            "+ order pooled (realised)": (base - col["prod_px"] + col["px_real"]
                                          - col["prod_order"] + col["order_real"]),
            "+ order pooled (virtual)": (base - col["prod_px"] + col["px_virt"]
                                         - col["prod_order"] + col["order_virt"]),
        }  # fmt: skip
        print(f"  {c}: FD {fd.mean():+.4f} (sd {fd.std(ddof=1):.4f})")
        for k in ("prod_px", "px_real", "px_virt", "prod_order", "order_real", "order_virt"):
            print(f"    term {k:12s} {col[k].mean():+.4f} sd {col[k].std(ddof=1):.4f}")
        for k, v in variants.items():
            print(f"    total {k:26s} {v.mean():+.4f} sd {v.std(ddof=1):.4f} closure "
                  f"{v.mean() / fd.mean():.3f}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--k", type=int, default=8, help="virtual draws per hit")
    ap.add_argument("--report", action="store_true", help="only report the saved rows")
    ap.add_argument("--timed", type=float, default=0.0,
                    help="time scale (steps) of the c fits' neighbour search; 0 = pooled")  # fmt: skip
    args = ap.parse_args()
    tag = f"_t{args.timed:g}" if args.timed else ""
    out = lab.OUT / f"virtual_emit_{args.case}_k{args.k}{tag}.json"
    if not args.report:
        rng = np.random.default_rng(12345)
        rows = []
        for seed, path in lab.seed_files(lab.CASES[args.case]):
            if args.seeds[0] <= seed < args.seeds[1]:
                rows.append(run_seed(args.case, seed, path, args.k, rng, args.timed))
                out.write_text(json.dumps({"case": args.case, "rows": rows}))
        print(f"wrote {out}")
    report(args.case, json.loads(out.read_text())["rows"])


if __name__ == "__main__":
    main()
