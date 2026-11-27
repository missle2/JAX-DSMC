"""Post-hoc tracer re-emissions (2026-11-08, option 2 of the diffuse-wall resolvability idea):
extra costate samples in the wall layer without touching the forward.

A tracer is a zero-weight particle re-emitted at a real diffuse hit (the hit point and time of
the real one, a velocity g from the wall kernel) that flies straight: no realised collisions,
so it is a pure post-processing object.  Its post-move value follows from the adjoint's
characteristic (Duhamel) form of the backward recursion -- lambda^m_k = c_k + lambda^m_{k+1}
along the collisionless characteristic, c_k = dt (L* lambda)(z_k) the step's full four-term
collision source (own and partner parts), which the virtual pull-back already computes per
slot (``Backward.collision``, the dumps' ``node_coll``):

    lambda^m_k(z_k) = sum_{j = k}^{k_d - 1} c_j(z_j) + lambda^m_{k_d}(z_{k_d}),
    z_j = z_k + (j - k) dt g,

collisions entering in expectation through the field (the virtual pull-back's own
approximation), no hit rewards (an outgoing particle never re-hits the convex body).  The
tracer is deleted where it leaves the wall layer (d > ``--d-exit``) or at the record's end:
there its value is the fitted lambda (0 past the end).  c and the exit value are local
linear kNN fits (knn3's chosen metric plus a time coordinate, 25 steps) on the outgoing nodes.

Gate (``replica``): one tracer per real hit with that hit's own realised g -- its value at
the hit step must agree in mean with the real particle's carried value (binned by v_n).
Test: K fresh kernel draws per hit added as nodes (``real+tracer``), or alone (``tracer``),
then the chosen kNN's (and a tighter one's) grad lambda at the real hits -> pathwise_x per
control, against the FD with the production terms swapped out.

    PYTHONPATH=. python3 scratch/fitlab/tracers.py --case drag [--seeds 0 8] [--k 2]
        [--d-exit 0.15] [--hit-frac 1.0] [--time-linear]
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
import virtual_emit as VE  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

T_SCALE = 25.0  # steps: the c / exit fits' time coordinate
QCHUNK = 200_000  # queries per kNN batch (bounds the neighbour arrays)
FITS = {"C1": "C1_0.25_0.033_0.3", "C0.5": "C0.5_0.25_0.033_0.3"}  # the grad fits' metrics


class TimedField:
    """A local linear kNN fit of one per-node quantity on the outgoing nodes, neighbours
    searched in (d, s, v_n, v_t, |v_z|, step / t_scale); the fit linear in the first five, or
    (``time_linear``) in all six -- the costate is non-stationary over the window."""

    def __init__(self, feat, step, y, scales, time_linear=False, t_scale=T_SCALE):
        self.feat6 = np.column_stack((feat, step))
        self.y = y
        self.time_linear = time_linear
        self.sc6 = np.append(scales, t_scale)
        self.tree = cKDTree(self.feat6 / self.sc6)

    def value(self, feat_q, step_q):
        out = np.zeros(len(feat_q))
        for a in range(0, len(feat_q), QCHUNK):
            b = min(a + QCHUNK, len(feat_q))
            q6 = np.column_stack((feat_q[a:b], step_q[a:b]))
            _d, idx = self.tree.query(q6 / self.sc6, k=VE.K_NB, workers=-1)
            if self.time_linear:
                out[a:b] = local_value(self.feat6, self.y, q6, idx)
            else:
                out[a:b], _g = VE.local_fit(self.feat6[:, :5], self.y, feat_q[a:b], idx)
        return out


def local_value(feat_n, y, feat_q, idx):
    """The intercept of the local linear fit in every feature column (box weights, columns
    scaled by their range, relative ridge, condition guard dropping the velocity columns 3-5 of
    the design -- knn3's guard -- and keeping 1, d, s and any trailing columns)."""
    n_p = feat_n.shape[1] + 1
    val = np.zeros(len(feat_q))
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
        val[a:b] = np.einsum("qj,qj->q", (np.linalg.inv(a_eff) * mask)[:, 0, :], aty)
    return val


def fly(x0, g, k0, n_steps, dt, d_exit, c_field, exit_field, mean_k):
    """Tracers from post-move states ``x0`` [T, 2] at steps ``k0`` with velocities ``g``
    [T, 3]: per tracer-step node (x, step, demeaned value) inside the layer, and each tracer's
    value at its first step and the exit value's share of it.  Returns ``(tid, x, step, lam_dm,
    lam0_dm, exit0_dm)``."""
    t = len(x0)
    tid_l, x_l, k_l, c_l = [], [], [], []
    alive = np.ones(t, bool)
    exit_x = np.zeros((t, 2))
    exit_k = np.full(t, n_steps)
    age = 0
    while alive.any():
        ids = np.flatnonzero(alive)
        k = k0[ids] + age
        x = x0[ids] + (age * dt) * g[ids, :2]
        d = np.linalg.norm(x - lab.CENTRE, axis=1) - lab.RADIUS
        ended = k >= n_steps
        out = ~ended & (d > d_exit)
        exit_x[ids[out]] = x[out]
        exit_k[ids[out]] = k[out]
        alive[ids[ended | out]] = False
        live = ~(ended | out)
        if live.any():
            ii = ids[live]
            fq, _th = knn.coordinates(x[live], g[ii], k[live])
            tid_l.append(ii)
            x_l.append(x[live])
            k_l.append(k[live])
            c_l.append(c_field.value(fq[:, :5], k[live].astype(float)))
        age += 1
    # the deletion value: the fitted lambda^m at the exit (0 past the record's end)
    lam_exit = np.zeros(t)
    has = exit_k < n_steps
    if has.any():
        fq, _th = knn.coordinates(exit_x[has], g[has], exit_k[has])
        lam_exit[has] = exit_field.value(fq[:, :5], exit_k[has].astype(float)) + mean_k[exit_k[has]]
    tid = np.concatenate(tid_l)
    xs = np.concatenate(x_l)
    ks = np.concatenate(k_l)
    cs = np.concatenate(c_l)
    # lambda^m at each tracer-step: the suffix sum of c along the tracer plus the exit value
    order = np.lexsort((-ks, tid))  # per tracer, latest step first
    tid, xs, ks, cs = tid[order], xs[order], ks[order], cs[order]
    csum = np.cumsum(cs)
    first = np.r_[0, np.flatnonzero(np.diff(tid)) + 1]
    start = np.repeat(csum[first] - cs[first], np.diff(np.r_[first, len(tid)]))
    lam = csum - start + lam_exit[tid]
    lam_dm = lam - mean_k[ks]
    lam0 = np.full(t, np.nan)
    at0 = ks == k0[tid]
    lam0[tid[at0]] = lam_dm[at0]
    return tid, xs, ks, lam_dm, lam0, lam_exit - mean_k[np.minimum(k0, n_steps - 1)]


def grad_pathwise(feat_n, lam_n, scales, hits, own_key=None, node_key=None):
    """pathwise_x per control of the local linear kNN fit at the real hits (knn3's form)."""
    hc, hth = knn.coordinates(hits["x"], hits["g"], hits["step"])
    fq = hc[:, :5]
    tree = cKDTree(feat_n / scales)
    gx = np.zeros((len(fq), 2))
    for a in range(0, len(fq), QCHUNK):
        b = min(a + QCHUNK, len(fq))
        if own_key is None:
            _d, idx = tree.query(fq[a:b] / scales, k=VE.K_NB, workers=-1)
        else:
            idx = VE.neighbours(tree, fq[a:b] / scales, own_key[a:b], node_key)
        _v, g = VE.local_fit(feat_n, lam_n, fq[a:b], idx)
        n_hat = np.stack((np.cos(hth[a:b]), np.sin(hth[a:b])), axis=1)
        t_hat = np.stack((-np.sin(hth[a:b]), np.cos(hth[a:b])), axis=1)
        gx[a:b] = g[:, :1] * n_hat + g[:, 1:2] * t_hat
    return np.einsum("hi,hi,ch->c", gx, hits["w"], hits["dth"])


def run_seed(case, seed, path, args, rng):
    t0 = time.time()
    meta = json.loads(path.with_suffix(".json").read_text())
    dt = float(meta["dt"])
    cfg = V.gate_config(lab.CASES[case])[0]
    s_wall = lab.V_REF * np.sqrt(cfg["t_wall"] / EM.F.TEMP)
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam", "node_coll"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < knn3.D_MAX)
    step_n = z["node_step"][keep]
    feat_n = nc[keep][:, :5]
    key_n = step_n.astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    lam_raw = z["node_lam"][keep]
    coll = z["node_coll"][keep].astype(float)
    del z, nc
    n_steps = int(step_n.max()) + 1
    mean_k = np.bincount(step_n, weights=lam_raw, minlength=n_steps) / np.maximum(
        np.bincount(step_n, minlength=n_steps), 1
    )
    lam_dm = lam_raw - mean_k[step_n]
    del lam_raw
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_v_pre", "hit_normal",
                        "hit_lam", "hit_dx"))  # fmt: skip
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    d_hit = float(np.median(hc[:, 0]))
    near = feat_n[:, 0] < 3 * d_hit
    sigma_v = float(np.sqrt(np.mean(np.var(feat_n[near, 2:5], axis=0))))
    mfp, cell = knn3.wall_mfp(case, feat_n, len(np.unique(h["hit_step"])), hth)
    scales = {name: knn3.scales(m, d_hit, sigma_v, mfp, cell) for name, m in FITS.items()}
    tl = args.time_linear
    c_field = TimedField(feat_n, step_n.astype(float), coll, scales["C1"], tl)
    exit_field = TimedField(feat_n, step_n.astype(float), lam_dm, scales["C1"], tl)
    g_real, v_in = h["hit_v"], h["hit_v_pre"]
    n_hat = h["hit_normal"][:, :2] / np.linalg.norm(h["hit_normal"][:, :2], axis=1)[:, None]
    t_rem = VE.remaining_flight(h["hit_x"], g_real)
    x_h = h["hit_x"] - t_rem[:, None] * g_real[:, :2]
    w = (v_in - g_real)[:, :2]
    dth = np.einsum("chi,hi->ch", h["hit_dx"], w) / np.sum(w * w, axis=1)[None]
    hits = {"x": h["hit_x"], "g": g_real, "step": h["hit_step"], "w": w, "dth": dth}
    own = h["hit_step"].astype(np.int64) * 1_000_000 + h["hit_slot"]
    k0_all = h["hit_step"].astype(np.int64)
    print(f"  {case} seed {seed}: {len(k0_all)} hits, {len(feat_n)} nodes, lambda_wall "
          f"{1000 * mfp:.0f} mm ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip

    # gate: one replica tracer per (sampled) hit with the realised g
    sel = np.flatnonzero(rng.random(len(k0_all)) < args.hit_frac)
    _tid, _x, _k, _l, lam0, _e = fly(h["hit_x"][sel], g_real[sel], k0_all[sel], n_steps, dt,
                                 args.d_exit, c_field, exit_field, mean_k)  # fmt: skip
    real0 = h["hit_lam"][sel] - mean_k[k0_all[sel]]
    vn = hc[sel, 2]
    gate = {"bins": []}
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 99.0)):
        m = (vn >= lo) & (vn < hi) & np.isfinite(lam0)
        diff = real0[m] - lam0[m]
        gate["bins"].append({"vn": [lo, hi], "n": int(m.sum()), "real": float(real0[m].mean()),
                             "tracer": float(lam0[m].mean()), "diff": float(diff.mean()),
                             "se": float(diff.std(ddof=1) / np.sqrt(m.sum()))})  # fmt: skip
    ok = np.isfinite(lam0)
    gate["slope"] = float(np.polyfit(lam0[ok], real0[ok], 1)[0])
    gate["sd_real"], gate["sd_tracer"] = float(real0[ok].std()), float(lam0[ok].std())
    print(f"    replica gate ({time.time() - t0:.0f}s): regression of real on tracer value "
          f"{gate['slope']:.3f}; sd real {gate['sd_real']:.3g} tracer {gate['sd_tracer']:.3g}",
          flush=True)  # fmt: skip
    for b in gate["bins"]:
        print(f"      v_n {b['vn']}: n {b['n']} real {b['real']:+.3g} tracer {b['tracer']:+.3g} "
              f"diff {b['diff']:+.3g} +- {b['se']:.2g}", flush=True)  # fmt: skip

    # test: K fresh kernel draws per (sampled) hit as extra nodes
    tx, tk, tl, tg = [], [], [], []
    for _ in range(args.k):
        g = VE.wall_draws(rng, n_hat[sel], s_wall)
        x0 = x_h[sel] + t_rem[sel, None] * g[:, :2]
        tid, xs, ks, lam_t, _l0, _e = fly(x0, g, k0_all[sel], n_steps, dt, args.d_exit, c_field,
                                      exit_field, mean_k)  # fmt: skip
        tx.append(xs)
        tk.append(ks)
        tl.append(lam_t)
        tg.append(g[tid])
    tx, tk, tl, tg = (np.concatenate(a) for a in (tx, tk, tl, tg))
    feat_t, _th = knn.coordinates(tx, tg, tk)
    feat_t = feat_t[:, :5]
    print(f"    {len(feat_t)} tracer nodes ({time.time() - t0:.0f}s)", flush=True)
    px = {}
    for name, sc in scales.items():
        px[f"real_{name}"] = grad_pathwise(feat_n, lam_dm, sc, hits, own, key_n)
        px[f"real+tracer_{name}"] = grad_pathwise(np.vstack((feat_n, feat_t)),
                                                  np.concatenate((lam_dm, tl)), sc, hits)  # fmt: skip
        px[f"tracer_{name}"] = grad_pathwise(feat_t, tl, sc, hits)
    print(f"    fits done ({time.time() - t0:.0f}s)", flush=True)
    rows = {}
    for i, c in enumerate(meta["controls"]):
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        total = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
        rows[c] = {"base_total": total, "prod_px": t["pathwise_x"],
                   **{k: float(v[i]) for k, v in px.items()}}  # fmt: skip
    return {"seed": seed, "gate": gate, "n_tracer_nodes": len(feat_t), "controls": rows}


def report(case, rows):
    fds = lab.finite_differences(case)
    names = [k for k in rows[0]["controls"]["radius"] if k not in ("base_total", "prod_px")]
    print(f"{case}: {len(rows)} seeds; totals with production pathwise_x swapped for each fit")
    for c in ("radius", "mode2", "bump_nose"):
        if c not in fds:
            continue
        fd = np.array(list(fds[c].values()))
        col = {k: np.array([r["controls"][c][k] for r in rows]) for k in rows[0]["controls"][c]}
        print(f"  {c}: FD {fd.mean():+.4f}")
        tot = col["base_total"]
        print(f"    {'production':22s} total {tot.mean():+.4f} sd {tot.std(ddof=1):.4f} "
              f"closure {tot.mean() / fd.mean():.3f}")  # fmt: skip
        for k in names:
            v = col["base_total"] - col["prod_px"] + col[k]
            print(f"    {k:22s} px {col[k].mean():+.4f} sd {col[k].std(ddof=1):.4f}  total "
                  f"{v.mean():+.4f} sd {v.std(ddof=1):.4f} closure {v.mean() / fd.mean():.3f}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    ap.add_argument("--k", type=int, default=2, help="fresh tracers per sampled hit")
    ap.add_argument("--d-exit", type=float, default=0.15, help="deletion distance (m)")
    ap.add_argument("--hit-frac", type=float, default=1.0, help="share of hits that spawn")
    ap.add_argument("--time-linear", action="store_true", help="c / exit fits linear in time")
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    out = lab.OUT / f"tracers_{args.case}{args.tag}.json"
    if not args.report:
        rng = np.random.default_rng(2026)
        rows = []
        for seed, path in lab.seed_files(lab.CASES[args.case]):
            if args.seeds[0] <= seed < args.seeds[1]:
                rows.append(run_seed(args.case, seed, path, args, rng))
                out.write_text(json.dumps({"case": args.case, "settings": vars(args),
                                           "rows": rows}))  # fmt: skip
        print(f"wrote {out}")
    rows = json.loads(out.read_text())["rows"]
    if len(rows) > 1:
        report(args.case, rows)


if __name__ == "__main__":
    main()
