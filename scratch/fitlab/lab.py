"""The fit lab (2026-11-04): grad_x lambda at the body hits of the dumped seeds
(scratch/fitlab/dump.py) -- a pooled reference, per-step candidate fits, and the closure of
every one against the finite differences.

Reference: the steady window's value is lambda(x, v, t) ~ c(t) + V(x, v), so ``V`` is fitted
pooled over every seed and a block of steps (``lambda`` minus its per-(seed, step) mean over
the outgoing near-wall nodes), per angular sector, step block and velocity bin (speed x
direction cosine x sign of v_t, in the wall frame), on a cubic B-spline in the wall distance
(knots down to 5 cm: the hits sit within v dt ~ 1 cm of the wall, the cells are 50 cm)
plus the bin's linear velocity terms and the sector's tangential slope.  Its grad_x at a hit
is the model's Cartesian gradient at fixed v (central difference of the fitted model, the
wall frame rotating with x).

Closure: a fit's pathwise_x for control c and seed s is sum_hits g . dx_c (the dump checks
this reproduces production to 1e-13); its adjoint total is the production total with the
production pathwise_x swapped for it; compared to the FD (the gate rows, unpaired: their
forwards predate the dumps' code, so seed-mean against seed-mean).

    PYTHONPATH=. python3 scratch/fitlab/lab.py reference --case drag [--seeds 0 8]
    PYTHONPATH=. python3 scratch/fitlab/lab.py report
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

CACHE = Path("scratch/fitlab/cache")
SLIM = Path("scratch/fitlab/cache_slim")
OUT = Path("scratch/fitlab/out")
FITGATE = Path("examples/mesh/results/fitgate")
CENTRE = np.array([5.0, 5.0])
RADIUS = 3.0
V_REF = 238.49840532777446  # sqrt(k T_inf / m), the fits' velocity scale
CASES = {
    "drag": "c10_diff_m2",
    "heat": "c10_diff_heat_m2",
    "spec": "c10_spec",
    "kn10": "c10_diff_kn10_m2",
    # the transfer cases (2026-11-05, study group fitlab2): FDs in the dumps' json ("fd")
    "m25_diff": "c25_diff_m2",
    "m25_spec": "c25_spec_m2",
    "m03_diff": "c03_diff_m2",
    "m03_spec": "c03_spec_m2",
    "hot": "c10_hot_m2",
}
BASE_VARIANT = "cell_along_pool_plane"
# circle_long's cells (60 x 20 of 0.5 m from (-10, 0)): the angle of each centre about the body
N_CELLS = 1200
_I, _J = np.divmod(np.arange(N_CELLS), 20)
CELL_ANGLE = np.arctan2(0.25 + 0.5 * _J - CENTRE[1], -10.0 + 0.25 + 0.5 * _I - CENTRE[0])
TERMS = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette", "order")

# --- binning and features --------------------------------------------------------------------

SETTINGS = {
    "sectors": 36,
    "block": 50,
    "speed": (0.0, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 8.0, 11.0, 14.0, np.inf),
    "cosine": (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001),
    "knots": (0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5),
    "d_max": 2.4,
    "ridge": 1e-6,
}


def _spline(settings):
    inner = np.asarray(settings["knots"])
    t = np.concatenate(([0.0] * 4, inner, [settings["d_max"]] * 4))
    return t, len(t) - 4


def frame(x, v):
    """Wall-frame coordinates of positions ``x`` (n, 2) and velocities ``v`` (n, 3): wall
    distance d, angle th, v_n, v_t, v_z (in V_REF)."""
    rel = x - CENTRE
    r = np.hypot(rel[:, 0], rel[:, 1])
    n = rel / r[:, None]
    vn = (v[:, 0] * n[:, 0] + v[:, 1] * n[:, 1]) / V_REF
    vt = (-v[:, 0] * n[:, 1] + v[:, 1] * n[:, 0]) / V_REF
    return r - RADIUS, np.arctan2(rel[:, 1], rel[:, 0]), vn, vt, v[:, 2] / V_REF


def group_of(th, step, vn, vt, vz, settings, cell=None):
    """Group id (sector, block, speed bin, cosine bin, sign v_t); -1 for incoming (v_n <= 0)."""
    ns = settings["sectors"]
    if cell is not None and settings.get("units") == "cells":
        sector, ns = cell.astype(np.int64), N_CELLS
    elif settings.get("units") == "foot":
        # the cell of the wall point under the node (x - phi grad phi): a wall-normal column
        foot = CENTRE + RADIUS * np.stack((np.cos(th), np.sin(th)), axis=1)
        ix = np.floor((foot[:, 0] + 10.0) / 0.5).astype(np.int64)
        iy = np.floor(foot[:, 1] / 0.5).astype(np.int64)
        sector, ns = ix * 20 + iy, N_CELLS
    else:
        sector = np.floor((th + np.pi) / (2 * np.pi) * ns).astype(np.int64) % ns
    block = step // settings["block"]
    speed = np.sqrt(vn * vn + vt * vt + vz * vz)
    sb = np.searchsorted(settings["speed"], speed, side="right") - 1
    cos = vn / np.maximum(np.hypot(vn, vt), 1e-12)
    cb = np.clip(np.searchsorted(settings["cosine"], cos, side="right") - 1, 0, None)
    sign = (vt > 0).astype(np.int64)
    n_s, n_c = len(settings["speed"]) - 1, len(settings["cosine"]) - 1
    gid = (((block * ns + sector) * n_s + sb) * n_c + cb) * 2 + sign
    return np.where(vn > 0, gid, -1), sector


def features(d, th, vn, vt, vz, sector, settings):
    """``(n, p)``: the d spline, the bin's linear velocity terms, the sector's tangential
    slope (th about the sector centre) and its d / v_n couplings."""
    t, nb = _spline(settings)
    dd = np.clip(d, 0.0, settings["d_max"] - 1e-9)
    if settings.get("dbasis") == "linear":  # the near-wall slab's straight line in d
        b = np.stack((np.ones_like(dd), dd), axis=1)
    else:
        b = BSpline.design_matrix(dd, t, 3).toarray()
    if settings.get("units") in ("cells", "foot"):
        centre = CELL_ANGLE[sector]
    else:
        centre = -np.pi + (sector + 0.5) * 2 * np.pi / settings["sectors"]
    dth = np.angle(np.exp(1j * (th - centre)))
    extra = np.stack((vn, vt, vz * vz, dth, dth * d, d * vn, d * vt), axis=1)
    del nb
    # "couple": the spline times velocity terms, so the wall-distance profile may vary within
    # the bin (the near-wall population's velocity mix changes with d: only fresh re-emissions
    # within v_n dt of the wall)
    cols = [b, extra]
    v2 = vn * vn + vt * vt + vz * vz
    for name in settings.get("couple", ()):
        cols.append(
            b * {"vn": vn, "vt": vt, "v2": v2, "inv": 1.0 / (np.sqrt(v2) + 0.3)}[name][:, None]
        )
    return np.concatenate(cols, axis=1)


def n_features(settings):
    nb = 2 if settings.get("dbasis") == "linear" else _spline(settings)[1]
    return nb * (1 + len(settings.get("couple", ()))) + 7


# --- data -----------------------------------------------------------------------------------


def seed_files(gate):
    """``[(seed, npz path)]`` of a gate's dumps: the full cache, else the slimmed one
    (scratch/fitlab/slim.py, the transfer cases' pull)."""
    out = {}
    for cache in (SLIM, CACHE):  # the full dump wins
        for p in sorted(cache.glob(f"{gate}_s*.npz")):
            if "_smoke" not in p.name:
                out[int(p.stem.split("_s")[-1])] = p
    return sorted(out.items())


def load(path, keys):
    with np.load(path) as z:
        return {k: z[k] for k in keys}


def demeaned(nodes):
    """lambda minus its per-step mean over the outgoing nodes."""
    lam, step = nodes["node_lam"], nodes["node_step"]
    m = int(step.max()) + 1
    s = np.bincount(step, weights=lam, minlength=m)
    c = np.bincount(step, minlength=m)
    return lam - (s / np.maximum(c, 1))[step]


# --- the reference ---------------------------------------------------------------------------


def accumulate(gate, seeds, settings):
    """Normal equations per group, summed over the seeds' outgoing nodes."""
    p = n_features(settings)
    acc = {}
    for seed, path in seed_files(gate):
        if seed not in seeds:
            continue
        t0 = time.time()
        nodes = load(path, ("node_step", "node_x", "node_v", "node_lam", "node_cell"))
        d, th, vn, vt, vz = frame(nodes["node_x"], nodes["node_v"])
        gid, sector = group_of(th, nodes["node_step"], vn, vt, vz, settings, nodes["node_cell"])
        out = gid >= 0
        if settings.get("slab"):
            out &= d < settings["slab"]
        lam = demeaned({"node_lam": nodes["node_lam"][out], "node_step": nodes["node_step"][out]})
        gid, sector = gid[out], sector[out]
        d, th, vn, vt, vz = d[out], th[out], vn[out], vt[out], vz[out]
        del nodes
        order = np.argsort(gid, kind="stable")
        gs = gid[order]
        cuts = np.flatnonzero(np.diff(gs)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [gs.size]))
        for a, b in zip(starts, ends, strict=True):
            idx = order[a:b]
            x = features(d[idx], th[idx], vn[idx], vt[idx], vz[idx], sector[idx], settings)
            y = lam[idx]
            g = int(gs[a])
            m = acc.get(g)
            if m is None:
                m = acc[g] = [np.zeros((p, p)), np.zeros(p), 0.0, 0]
            m[0] += x.T @ x
            m[1] += x.T @ y
            m[2] += float(y @ y)
            m[3] += idx.size
        print(f"  {gate} seed {seed}: {gs.size} outgoing nodes, {len(starts)} groups, "
              f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip
    return acc


def solve(acc, settings, min_rows=64):
    coef = {}
    for g, (xtx, xty, _yy, n) in acc.items():
        if n < min_rows:
            continue
        scale = np.sqrt(np.maximum(np.diag(xtx), 1e-300))
        a = xtx / np.outer(scale, scale)
        a[np.diag_indices_from(a)] += settings["ridge"]
        coef[g] = np.linalg.solve(a, xty / scale) / scale
    return coef


def model_gradient(x, v, step, coef, settings, h=1e-4, cell=None, h_v=1.0):
    """grad_x of the fitted V at the hits (central differences at fixed v and group);
    NaN rows where the group has no fit; also the value and the group ids."""
    d, th, vn, vt, vz = frame(x, v)
    gid, sector = group_of(th, step, vn, vt, vz, settings, cell)
    grad = np.full((x.shape[0], 2), np.nan)
    value = np.full(x.shape[0], np.nan)
    vgrad = np.full((x.shape[0], 3), np.nan)
    order = np.argsort(gid, kind="stable")
    gs = gid[order]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    for a, b in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [gs.size])), strict=True):
        g = int(gs[a])
        if g < 0 or g not in coef:
            continue
        idx = order[a:b]
        beta = coef[g]
        sec = sector[idx]
        f0 = features(d[idx], th[idx], vn[idx], vt[idx], vz[idx], sec, settings)
        value[idx] = f0 @ beta
        for axis in range(2):
            vals = []
            for sgn in (1.0, -1.0):
                xs = x[idx].copy()
                xs[:, axis] += sgn * h
                fd = frame(xs, v[idx])
                vals.append(features(*fd, sec, settings) @ beta)
            grad[idx, axis] = (vals[0] - vals[1]) / (2 * h)
        for axis in range(3):
            vals = []
            for sgn in (1.0, -1.0):
                vs = v[idx].copy()
                vs[:, axis] += sgn * h_v
                fd = frame(x[idx], vs)
                vals.append(features(*fd, sec, settings) @ beta)
            vgrad[idx, axis] = (vals[0] - vals[1]) / (2 * h_v)
    return grad, value, gid, vgrad


# --- FDs ---------------------------------------------------------------------------------------


def _gate_fd(gate):
    rows = {}
    for path in sorted(FITGATE.glob(f"{gate}_base*_s*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("gate") == gate and r.get("basis", "base") == "base":
                rows[r["seed"]] = r["fd"][0]
    return rows


def _bump_fd(objective):
    rows = {}
    for path in sorted(FITGATE.glob(f"bump_{objective}_s*.jsonl")):
        if "_smoke" in path.name:
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            for par, c in r["controls"].items():
                rows.setdefault(par, {})[r["seed"]] = c["fd"]
    return rows


def finite_differences(case):
    """``{control: {seed: FD}}`` from the gate rows that share the dump's objective and body
    (mode2s drag/heat: 0 by symmetry)."""
    if case == "drag":
        out = {"radius": _gate_fd("c10_diff"), "mode2": _gate_fd("c10_diff_m2")}
        out.update(_bump_fd("drag"))
        out["mode2s"] = {"symmetry": 0.0}
    elif case == "heat":
        out = {"mode2": _gate_fd("c10_diff_heat_m2")}
        out.update(_bump_fd("heat"))
        out["mode2s"] = {"symmetry": 0.0}
    elif case == "spec":
        out = {"radius": _gate_fd("c10_spec"), "mode2": _gate_fd("c10_spec_m2"),
               "mode2s": {"symmetry": 0.0}}  # fmt: skip
    elif case == "kn10":
        out = {"radius": _gate_fd("c10_diff_kn10"), "mode2": _gate_fd("c10_diff_kn10_m2"),
               "mode2s": {"symmetry": 0.0}}  # fmt: skip
    else:
        # the transfer cases: each seed's dump carries its own central differences
        out = {"mode2s": {"symmetry": 0.0}}
        for seed, path in seed_files(CASES[case]):
            meta = json.loads(path.with_suffix(".json").read_text())
            for par, value in meta.get("fd", {}).items():
                out.setdefault(par, {})[seed] = value
    return out


# --- per-seed closure --------------------------------------------------------------------------


def hit_fits(path, coef, settings):
    """Per control, pathwise_x (and _v) of each fit at this seed's hits, and diagnostics."""
    keys = ("hit_step", "hit_x", "hit_v", "hit_normal", "hit_lt", "hit_coll", "hit_dx", "hit_dv",
            "hit_gx", "hit_gv", "hit_lam", "hit_cell")  # fmt: skip
    h = load(path, keys)
    meta = json.loads(path.with_suffix(".json").read_text())
    variants = meta["variants"]
    base = h["hit_gx"][variants.index(BASE_VARIANT)]
    g_ref, value, gid, gv_ref = model_gradient(
        h["hit_x"], h["hit_v"], h["hit_step"], coef, settings, cell=h["hit_cell"]
    )
    missing = np.isnan(g_ref[:, 0])
    g_ref = np.where(missing[:, None], base, g_ref)
    rel = h["hit_x"] - CENTRE
    n = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
    # the reference's normal derivative on the production fit's tangential part
    g_norm = base + np.sum((g_ref - base) * n, axis=1)[:, None] * n
    # the along-flight identity (T + L* V) on the reference's perpendicular part
    vd = h["hit_v"][:, :2]
    v2 = np.maximum(np.sum(vd * vd, axis=1), 1e-300)
    t_k = h["hit_lt"] + h["hit_coll"] / meta["dt"]
    par = np.sum(g_ref * vd, axis=1) / v2
    g_along = g_ref - par[:, None] * vd - (t_k / v2)[:, None] * vd
    fits = {name: h["hit_gx"][j] for j, name in enumerate(variants)}
    fits.update({"ref": g_ref, "ref_normal": g_norm, "ref_along": g_along})
    px = {
        c: {name: float(np.sum(g * h["hit_dx"][i])) for name, g in fits.items()}
        for i, c in enumerate(meta["controls"])
    }
    # the reference grad_v too (pathwise_v: nonzero on specular walls only), swapped into
    # the closure as a change of pathwise_x's slot: ref_v = ref's pathwise_x + its pathwise_v
    # less the production (base variant) pathwise_v
    gv_base = h["hit_gv"][variants.index(BASE_VARIANT)]
    gv_ref = np.where(np.isnan(gv_ref[:, :1]), gv_base, gv_ref)
    for i, c in enumerate(meta["controls"]):
        dpv = float(np.sum((gv_ref - gv_base) * h["hit_dv"][i]))
        px[c]["ref_v"] = px[c]["ref"] + dpv
        px[c]["pathwise_v_base"] = float(np.sum(gv_base * h["hit_dv"][i]))
        px[c]["pathwise_v_ref"] = float(np.sum(gv_ref * h["hit_dv"][i]))
    # the identity check: v . grad V (reference) against -(T + L* V), per speed bin
    lhs, rhs = np.sum(g_ref * vd, axis=1), -t_k
    speed = np.sqrt(v2 + h["hit_v"][:, 2] ** 2) / V_REF
    ident = {}
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)):
        sel = (speed >= lo) & (speed < hi) & ~missing
        if sel.sum() > 10:
            ident[f"{lo}-{hi}"] = [float(np.mean(lhs[sel])), float(np.mean(rhs[sel])),
                                   int(sel.sum())]  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][BASE_VARIANT]
        totals[c] = sum(t[k] for k in TERMS) + meta["production"][c]["volume"]
    return {
        "seed": meta["seed"], "objective": meta["objective"], "pathwise_x": px,
        "base_total": totals, "missing_hits": int(missing.sum()), "hits": int(missing.size),
        "identity": ident,
    }  # fmt: skip


def run_self(case, seeds, tag, settings):
    """Each seed's own pooled fit (its steps only) evaluated at its own hits."""
    gate = CASES[case]
    rows = []
    for seed, path in seed_files(gate):
        if seed not in seeds:
            continue
        coef = solve(accumulate(gate, {seed}, settings), settings)
        rows.append(hit_fits(path, coef, settings))
        print(f"  self seed {seed}: {len(coef)} groups, missing {rows[-1]['missing_hits']}",
              flush=True)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"self_{case}{tag}.json"
    out.write_text(json.dumps({"case": case, "settings": {k: list(v) if isinstance(v, tuple)
                               else v for k, v in settings.items()}, "rows": rows},
                              default=float))  # fmt: skip
    print(f"wrote {out}")


def run_reference(case, seeds, tag, settings):
    gate = CASES[case]
    t0 = time.time()
    acc = accumulate(gate, seeds, settings)
    coef = solve(acc, settings)
    print(f"{case}: {len(coef)} groups fitted ({len(acc)} with data), {time.time() - t0:.0f}s",
          flush=True)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(coef), dtype=np.int64)
    np.savez(OUT / f"coef_{case}{tag}.npz", ids=ids, coef=np.stack([coef[g] for g in ids]))
    rows = []
    for seed, path in seed_files(gate):
        if seed in seeds:
            rows.append(hit_fits(path, coef, settings))
            print(f"  hits seed {seed}: missing {rows[-1]['missing_hits']} of "
                  f"{rows[-1]['hits']}", flush=True)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"ref_{case}{tag}.json"
    out.write_text(json.dumps({"case": case, "settings": {k: list(v) if isinstance(v, tuple)
                               else v for k, v in settings.items()}, "rows": rows},
                              default=float))  # fmt: skip
    print(f"wrote {out}")


def report(path):
    data = json.loads(Path(path).read_text())
    case, rows = data["case"], data["rows"]
    fds = finite_differences(case)
    n = len(rows)
    print(f"\n{case} ({path}): {n} seeds")
    for c in rows[0]["pathwise_x"]:
        fd = fds.get(c)
        if not fd:
            continue
        fdv = np.array(list(fd.values()))
        fd_m = fdv.mean()
        fd_se = fdv.std(ddof=1) / len(fdv) ** 0.5 if len(fdv) > 1 else 0.0
        print(f"  {c:10s} FD {fd_m:+.4g} +- {2 * fd_se:.2g} ({len(fdv)} seeds)")
        for name in rows[0]["pathwise_x"][c]:
            tot = np.array([r["base_total"][c] - r["pathwise_x"][c][BASE_VARIANT]
                            + r["pathwise_x"][c][name] for r in rows])  # fmt: skip
            pxm = np.mean([r["pathwise_x"][c][name] for r in rows])
            se = np.hypot(tot.std(ddof=1) / n**0.5, fd_se)
            ratio = tot.mean() / fd_m if abs(fd_m) > 1e-12 else np.nan
            print(
                f"    {name:24s} pathwise_x {pxm:+.4g}  adjoint {tot.mean():+.4g}  "
                f"excess {tot.mean() - fd_m:+.4g} +- {2 * se:.2g}  ratio {ratio:.3f}"
            )
    print("  identity v.grad V vs -(T + L*V) at the hits (mean over seeds, by speed / V_REF):")
    for b in rows[0]["identity"]:
        lhs = np.mean([r["identity"][b][0] for r in rows if b in r["identity"]])
        rhs = np.mean([r["identity"][b][1] for r in rows if b in r["identity"]])
        print(f"    {b:6s} ref {lhs:+.4g}  identity {rhs:+.4g}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--case", choices=list(CASES), required=True)
    r.add_argument("--seeds", type=int, nargs=2, default=(0, 8))
    r.add_argument("--tag", default="")
    r.add_argument("--self", action="store_true", help="each seed's own fit (production-like)")
    r.add_argument("--set", nargs="*", default=[], help="key=value overrides of SETTINGS")
    p = sub.add_parser("report")
    p.add_argument("paths", nargs="+")
    args = ap.parse_args()
    if args.cmd == "reference":
        settings = dict(SETTINGS)
        for kv in args.set:
            k, v = kv.split("=", 1)
            settings[k] = json.loads(v)
            if isinstance(settings[k], list):
                settings[k] = tuple(settings[k])
        run = run_self if args.self else run_reference
        run(args.case, set(range(*args.seeds)), args.tag, settings)
    else:
        for path in args.paths:
            report(path)


if __name__ == "__main__":
    main()
