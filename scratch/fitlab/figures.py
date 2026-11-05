"""The fit lab's figures (2026-11-05): the wall-normal structure of lambda at the stagnation
nose, the scale mismatch with the per-cell fit, and the closure of every fit tried.

* ``profile_<case>.png``: lambda against the wall distance within 20 deg of stagnation
  (outgoing nodes, 8 seeds pooled): raw (per-step demeaned) and at fixed velocity (minus
  each narrow velocity cell x 50-step block's mean), with the hits' wall-distance
  distribution and the production / local slopes drawn as tangents.
* ``slopes.png``: the hit-weighted narrow-cell slope d lambda / dd against the regression
  window, beside every fit's mean normal derivative at the same hits (production, the
  running-sum windows, kNN) and the slope the nose closure needs.  Specular too (its post-hit
  shift is 2 delta n: pathwise_x is purely this slope).
* ``closure.png``: adjoint / FD (+- 2 s.e.) per fit family for the nose bump and cos 2 theta.

    PYTHONPATH=. python3 scratch/fitlab/figures.py [--recompute]

The profile / slope sums are cached in ``scratch/fitlab/cache/fig_<case>.npz``; figures go to
``examples/mesh/figures/fitlab/``.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import lab  # noqa: E402
import summary  # noqa: E402
import window  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from examples.mesh.figure_paths import figure_path  # noqa: E402

FIG = Path(figure_path("fitlab", ""))
ANGLE = 20.0
DV = 0.25
BLOCK = 50
D_MAX = 0.5
EDGES = np.concatenate((np.arange(0.0, 0.01, 0.001), np.arange(0.01, 0.06, 0.005),
                        np.arange(0.06, 0.2, 0.02), np.arange(0.2, 0.501, 0.05)))  # fmt: skip
WINDOWS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5)
# the needed nose closure factor on cell_pool_plane's normal slope (the 2026-11-05 fit lab)
NEEDED = {"drag": 0.47, "heat": 4.2, "spec": None}
# the local fits recomputed at the nose hits for the figures: (label, kind, settings)
LOCAL_FITS = (
    ("window 2 cm, 0.4 m, v box 0.5", "window", (64, 0.02, 0.4, 0.5)),
    ("window 2 cm, 0.4 m, speed bins", "window", (64, 0.02, 0.4, 0.0)),
    ("kNN k 64, d 3 cm", "knn", (64, "d0.03_s0.1_v0.5")),
)
# the reference palette (dataviz skill, light mode)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")  # fmt: skip
RED = "#e34948"
UNITS = {"drag": (1e-6, r"$10^{-6}$"), "heat": (1e-3, r"$10^{-3}$"), "spec": (1e-6, r"$10^{-6}$")}


def _key(vn, vt, vz, step):
    i = np.floor(vn / DV).astype(np.int64)
    j = np.floor(vt / DV).astype(np.int64) + 200
    k = np.floor(np.abs(vz) / DV).astype(np.int64)
    return ((i * 400 + j) * 100 + k) * 10 + step // BLOCK


def _reduce(keys, cols):
    uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
    tot = np.zeros((uk.size, cols[0].shape[1]))
    np.add.at(tot, inv, np.concatenate(cols))
    return uk, tot


def _local_slopes(path, h, hs):
    """d lambda / dn of :data:`LOCAL_FITS` at the hits ``hs`` of one seed (NaN: no fit)."""
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < window.D_MAX)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    nkey = z["node_step"][keep].astype(np.int64) * 1_000_000 + z["node_slot"][keep]
    nc = nc[keep]
    del z
    hit = lab.load(path, ("hit_slot",))["hit_slot"][hs]
    hc, _hth = knn.coordinates(h["hit_x"][hs], h["hit_v"][hs], h["hit_step"][hs])
    hkey = h["hit_step"][hs].astype(np.int64) * 1_000_000 + hit
    out = {}
    for label, kind, setting in LOCAL_FITS:
        if kind == "window":
            k, w, sector, dv = setting
            g, count = window.window_gradients(nc, lam, nkey, hc, hkey, [k], [w], sector, dv)[
                (k, w)
            ]
            out[label] = np.where(count >= window.MIN_NODES, g[:, 0], np.nan)
        else:
            k, metric = setting
            m = knn.parse_metric(metric)
            _dist, idx = cKDTree(knn.scaled(nc, m)).query(knn.scaled(hc, m), k=k + 1, workers=-1)
            own = nkey[idx] == hkey[:, None]
            idx = np.take_along_axis(idx, np.argsort(own, axis=1, kind="stable"), axis=1)[:, :k]
            out[label] = knn.local_gradients(nc, lam, hc, idx, [k], np.isfinite(m["t"]))[k][:, 0]
    return out


def compute(case):
    """Sums for the profiles and the window slopes, pooled over the seeds."""
    gate = lab.CASES[case]
    raw = np.zeros((EDGES.size - 1, 2))
    kb_keys, kb_cols, kw_keys, kw_cols, hit_keys = [], [], [], [], []
    hit_d, prod = [], {}
    nb = EDGES.size - 1
    for seed, path in lab.seed_files(gate):
        z = lab.load(path, ("node_step", "node_x", "node_v", "node_lam"))
        d, th, vn, vt, vz = lab.frame(z["node_x"], z["node_v"])
        out = vn > 0
        lam = lab.demeaned({"node_lam": z["node_lam"][out], "node_step": z["node_step"][out]})
        d, th, vn, vt, vz, step = d[out], th[out], vn[out], vt[out], vz[out], z["node_step"][out]
        del z
        ang = np.degrees(np.abs(np.angle(np.exp(1j * (th - np.pi)))))
        sel = (ang < ANGLE) & (d >= 0.0) & (d < D_MAX)
        d, lam, key = d[sel], lam[sel], _key(vn[sel], vt[sel], vz[sel], step[sel])
        b = np.clip(np.searchsorted(EDGES, d, side="right") - 1, 0, nb - 1)
        raw[:, 0] += np.bincount(b, minlength=nb)
        raw[:, 1] += np.bincount(b, weights=lam, minlength=nb)
        # per (velocity cell x block, d bin): n, sum lambda
        kb = key * 100 + b
        u, inv = np.unique(kb, return_inverse=True)
        kb_keys.append(u)
        kb_cols.append(np.stack((np.bincount(inv), np.bincount(inv, weights=lam)), axis=1))
        # per (window, key): n, sum d, sum d^2, sum lambda, sum d lambda
        for iw, w in enumerate(WINDOWS):
            m = d < w
            u, inv = np.unique(key[m], return_inverse=True)
            dm, lm = d[m], lam[m]
            kw_keys.append(u * 10 + iw)
            kw_cols.append(np.stack([np.bincount(inv, weights=x, minlength=u.size)
                                     for x in (np.ones(dm.size), dm, dm * dm, lm, dm * lm)],
                                    axis=1))  # fmt: skip
        h = lab.load(path, ("hit_step", "hit_x", "hit_v", "hit_gx"))
        meta = json.loads(path.with_suffix(".json").read_text())
        hd, hth, hvn, hvt, hvz = lab.frame(h["hit_x"], h["hit_v"])
        hs = np.degrees(np.abs(np.angle(np.exp(1j * (hth - np.pi))))) < ANGLE
        hit_d.append(hd[hs])
        hit_keys.append(_key(hvn[hs], hvt[hs], hvz[hs], h["hit_step"][hs]))
        rel = h["hit_x"][hs] - lab.CENTRE
        n = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
        for j, name in enumerate(meta["variants"]):
            prod.setdefault(name, []).append(np.sum(h["hit_gx"][j][hs] * n, axis=1))
        for label, slope in _local_slopes(path, h, hs).items():
            prod.setdefault(label, []).append(slope)
        print(f"  {case} seed {seed}", flush=True)
    # fixed-velocity profile: each key's offset (its mean over [0, D_MAX]) removed
    uk, tot = _reduce(kb_keys, kb_cols)
    key_of, bin_of = uk // 100, uk % 100
    ku, kinv = np.unique(key_of, return_inverse=True)
    kn = np.bincount(kinv, weights=tot[:, 0])
    ks = np.bincount(kinv, weights=tot[:, 1])
    resid = tot[:, 1] - tot[:, 0] * (ks / kn)[kinv]
    fixed = np.stack((np.bincount(bin_of, weights=tot[:, 0], minlength=nb),
                      np.bincount(bin_of, weights=resid, minlength=nb)), axis=1)  # fmt: skip
    # hit-weighted window slopes
    uk, tot = _reduce(kw_keys, kw_cols)
    hk, hc = np.unique(np.concatenate(hit_keys), return_counts=True)
    slopes = np.full((len(WINDOWS), 2), np.nan)
    for iw in range(len(WINDOWS)):
        sel = uk % 10 == iw
        keys, s = uk[sel] // 10, tot[sel]
        pos = np.clip(np.searchsorted(keys, hk), 0, max(keys.size - 1, 0))
        found = keys[pos] == hk
        nn, sd, sdd, sl, sdl = s[pos].T
        var = sdd / np.maximum(nn, 1) - (sd / np.maximum(nn, 1)) ** 2
        ok = found & (nn >= 30) & (var > 0)
        sl_k = (sdl / np.maximum(nn, 1) - sd * sl / np.maximum(nn, 1) ** 2) / np.where(ok, var, 1)
        w = hc[ok].astype(float)
        mean = np.sum(sl_k[ok] * w) / w.sum()
        se = np.sqrt(np.sum(w**2 * (sl_k[ok] - mean) ** 2)) / w.sum()
        slopes[iw] = mean, 2 * se
    names = list(prod)
    np.savez(lab.CACHE / f"fig_{case}.npz", raw=raw, fixed=fixed, slopes=slopes,
             hit_d=np.concatenate(hit_d), prod_names=np.array(names),
             prod=np.array([np.nanmean(np.concatenate(prod[k])) for k in names]))  # fmt: skip


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def _fit_lines(case, prod):
    """(slope, colour, style, label) of each fit's mean normal derivative at the nose hits."""
    lines = [(prod["cell_along_pool_plane"], MAGENTA, "-.", "production cell_along"),
             (prod["cell_pool_plane"], ORANGE, "--", "production cell_pool")]  # fmt: skip
    for (label, *_r), colour in zip(LOCAL_FITS, (GREEN, AQUA, RED), strict=True):
        if label in prod:
            lines.append((prod[label], colour, "-", label))
    if NEEDED.get(case) is not None:
        lines.append(
            (NEEDED[case] * prod["cell_pool_plane"], VIOLET, ":", "what the nose FD needs")
        )
    return lines


def _profile(case, z):
    unit, label = UNITS[case]
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    names = list(z["prod_names"])
    prod = dict(zip(names, z["prod"], strict=True))
    slopes = z["slopes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1.15]})  # fmt: skip
    for ax, xmax in zip(axes, (D_MAX, 0.06), strict=True):
        _style(ax)
        for arr, colour, name in ((z["fixed"], BLUE, "fixed velocity (narrow v-cell)"),
                                  (z["raw"], INK2, "raw (all velocities)")):  # fmt: skip
            ok = arr[:, 0] > 200
            prof = arr[ok, 1] / arr[ok, 0]
            sel = mid[ok] <= xmax
            # anchor both curves at the first bin, so the shapes compare
            prof = prof - prof[0]
            ax.plot(100 * mid[ok][sel], prof[sel] / unit, "-o", color=colour, lw=2, ms=3.5,
                    label=name)  # fmt: skip
        fixed = z["fixed"]
        ok = fixed[:, 0] > 200
        y0 = 0.0
        x = np.linspace(0.0, xmax, 50)
        for slope, colour, ls, name in _fit_lines(case, prod):
            ax.plot(100 * x, (y0 + slope * (x - mid[ok][0])) / unit, ls, color=colour, lw=1.6,
                    label=name)  # fmt: skip
        ax.set_xlim(0, 100 * xmax)
        ax.set_xlabel("wall distance d (cm)", color=INK)
    axes[0].axvspan(0, 50, color=YELLOW, alpha=0.08, lw=0)
    axes[0].text(25, axes[0].get_ylim()[1], "one 50 cm DSMC cell", ha="center", va="top",
                 color=INK2, fontsize=9)  # fmt: skip
    # the hits' post-move wall distances
    tw = axes[1].twinx()
    tw.hist(100 * z["hit_d"], bins=np.linspace(0, 6, 61), color=YELLOW, alpha=0.35,
            density=True, lw=0)  # fmt: skip
    tw.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        tw.spines[s].set_visible(False)
    tw.set_zorder(0)
    axes[1].set_zorder(1)
    axes[1].patch.set_visible(False)
    med = np.median(z["hit_d"])
    axes[1].text(1.6, axes[1].get_ylim()[0], f"post-hit particles, yellow\n(median "
                 f"{1000 * med:.0f} mm)", color="#8a6100", fontsize=9, va="bottom")  # fmt: skip
    axes[0].set_ylabel(rf"$\lambda$ - $\lambda$(first bin)  ({label})", color=INK)
    axes[0].set_title("full cell", color=INK, fontsize=10, loc="left")
    axes[1].set_title("first 6 cm: where the hits are", color=INK, fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5, loc="best", labelcolor=INK)
    fig.suptitle(f"{case}: re-emitted particles' value against wall distance, within "
                 f"{ANGLE:.0f} deg of stagnation (8 seeds)", color=INK, fontsize=11, x=0.01,
                 ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / f"profile_{case}.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def _slopes(zs):
    fig, axes = plt.subplots(1, len(zs), figsize=(5.2 * len(zs), 4.4), facecolor=SURF)
    for ax, (case, z) in zip(axes, zs.items(), strict=True):
        _style(ax)
        unit, label = UNITS[case]
        prod = dict(zip(list(z["prod_names"]), z["prod"], strict=True))
        w = 100 * np.array(WINDOWS)
        ax.set_xlim(0.8, 60)
        m, e = z["slopes"][:, 0] / unit, z["slopes"][:, 1] / unit
        # windows with too few nodes per narrow cell (specular near the wall) read off-scale:
        # dropped, so the fits' lines set the axis
        scale = max(abs(sl) for sl, *_r in _fit_lines(case, prod)) / unit
        keep = np.isfinite(m) & (np.abs(m) < 5 * scale) & (e < 5 * scale)
        w, m, e = w[keep], m[keep], e[keep]
        ax.errorbar(w, m, yerr=e, fmt="-o", color=BLUE, lw=2, ms=5, capsize=3,
                    label="narrow-cell local slope")  # fmt: skip
        for slope, colour, ls, name in _fit_lines(case, prod):
            ax.axhline(slope / unit, color=colour, ls=ls, lw=1.6, label=name)
        ax.set_xscale("log")
        ticks = 100 * np.array(WINDOWS)
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
        ax.set_xlabel("regression window [0, W] (cm)", color=INK)
        ax.set_ylabel(rf"$\partial\lambda/\partial d$ at the hits  ({label} per m)", color=INK)
        ax.set_title(case, color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("The wall slope depends on the scale it is measured over", color=INK,
                 fontsize=11, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "slopes.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def _closure_rows():
    """(family, label, case, file stem, fit name)."""
    rows = [("production", "cell_along (production)", "cell_along_pool_plane", ""),
            ("production", "cell_pool", "cell_pool_plane", "")]  # fmt: skip
    for tag, lab_ in (("_cp_k5", "first knot 10 cm"), ("", "first knot 5 cm"),
                      ("_k11", "first knot 2.5 cm")):  # fmt: skip
        rows.append(("pooled spline in d", lab_, "ref", f"ref_{{case}}{tag}"))
    for w in ("0.02", "0.03", "0.05", "0.1"):
        rows.append(("linear slab in d", f"{100 * float(w):g} cm slab", "ref",
                     f"ref_{{case}}_slab{w}"))  # fmt: skip
    for w in ("0.01", "0.02", "0.03", "0.05", "0.1"):
        rows.append(("narrow v-cell slope", f"{100 * float(w):g} cm window",
                     f"nar{w}_cell_pool", "nar"))  # fmt: skip
    for tag, lab_ in (("sec_b50", "sector x 50 steps"), ("foot_b50", "wall foot x 50 steps"),
                      ("cell_b50", "cells x 50 steps"), ("sec_b300", "sector x 300 steps")):  # fmt: skip
        rows.append(("single-seed fit", lab_, "ref", f"self_{{case}}_{tag}"))
    for metric, lab_ in (("k64_d0.03_s0.1_v0.5", "k 64, d 3 cm (best)"),
                         ("k256_d0.01_s0.1_v0.25", "k 256, v 0.25"),
                         ("k1024_d0.01_s0.3_v0.5", "k 1024, s 0.3"),
                         ("k256_d0.03_s0.3_v1", "k 256, d 3 cm, s 0.3, v 1"),
                         ("k64_d0.003_s0.1_v0.5", "k 64, d 3 mm")):  # fmt: skip
        rows.append(("kNN (single seed)", lab_, f"knn_{metric}", "knn_{case}"))
    return rows


def _closure_value(case, stem, name, fds, cache):
    if stem == "":
        stem = f"ref_{case}"
    if stem == "nar":
        stem = {"drag": "nar_drag_dv0.25", "heat": "nar_heat_dv0.5_s18_b100"}.get(case, "none")
    stem = stem.format(case=case)
    path = lab.OUT / f"{stem}.json"
    if not path.exists():
        return None
    if stem not in cache:
        cache[stem] = json.loads(path.read_text())["rows"]
    rows = cache[stem]
    out = {}
    for c in ("bump_nose", "mode2", "radius"):
        if name not in rows[0]["pathwise_x"].get(c, {}) or c not in fds:
            continue
        ratio, t = summary.closure(rows, fds, c, name)
        err = abs(ratio - 1.0) / abs(t) if t else 0.0
        out[c] = (ratio, 2 * err)
    return out


def _closure():
    rows = _closure_rows()
    panels = [
        ("drag", "bump_nose"),
        ("drag", "mode2"),
        ("heat", "bump_nose"),
        ("heat", "mode2"),
        ("spec", "radius"),
        ("kn10", "mode2"),
    ]
    fds = {case: lab.finite_differences(case) for case in ("drag", "heat", "spec", "kn10")}
    cache = {}
    fig, axes = plt.subplots(1, 6, figsize=(18, 7.4), sharey=True, facecolor=SURF)
    colours = {"production": ORANGE, "pooled spline in d": BLUE, "linear slab in d": AQUA,
               "narrow v-cell slope": VIOLET, "single-seed fit": GREEN,
               "kNN (single seed)": MAGENTA}  # fmt: skip
    ypos = []
    y, last = 0, None
    for fam, *_ in rows:
        if last is not None and fam != last:
            y += 0.7
        ypos.append(y)
        y += 1
        last = fam
    for ax, (case, control) in zip(axes, panels, strict=True):
        _style(ax)
        ax.axvline(1.0, color=INK, lw=1.2)
        ax.axvspan(0.9, 1.1, color=GREEN, alpha=0.07, lw=0)
        for yy, (fam, label, name, stem) in zip(ypos, rows, strict=True):
            v = _closure_value(case, stem, name, fds[case], cache)
            if not v or control not in v:
                continue
            r, e = v[control]
            clipped = min(max(r, -0.6), 3.6)
            ax.errorbar(clipped, yy, xerr=e if abs(clipped - r) < 1e-9 else None, fmt="o",
                        color=colours[fam], ms=6, capsize=2.5, lw=1.4)  # fmt: skip
            if clipped != r:
                ax.annotate(f"{r:.1f}", (clipped, yy), xytext=(4, 4), textcoords="offset points",
                            fontsize=7.5, color=INK2)  # fmt: skip
        ax.set_xlim(-0.7, 3.7)
        ax.set_title(f"{case}: {control.replace('bump_nose', 'nose bump')}"
                     .replace("mode2", "cos 2θ"), color=INK, fontsize=10, loc="left")  # fmt: skip
        ax.set_xlabel("adjoint / FD  (±2 s.e.)", color=INK)
    axes[0].set_yticks(ypos, [f"{label}" for _f, label, *_ in rows])
    axes[0].invert_yaxis()
    for tick, (fam, *_r) in zip(axes[0].get_yticklabels(), rows, strict=True):
        tick.set_color(INK)
    # family labels
    seen = {}
    for yy, (fam, *_r) in zip(ypos, rows, strict=True):
        seen.setdefault(fam, []).append(yy)
    for fam, ys in seen.items():
        axes[-1].text(3.75, np.mean(ys), fam, color=colours[fam], fontsize=9, va="center",
                      ha="left", fontweight="bold")  # fmt: skip
    fig.suptitle("Closure of every near-wall fit tried (1 = matches the FD; green band ±10%)",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(FIG / "closure.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    zs = {}
    for case in ("drag", "heat", "spec"):
        path = lab.CACHE / f"fig_{case}.npz"
        if args.recompute or not path.exists():
            compute(case)
        zs[case] = dict(np.load(path))
        _profile(case, zs[case])
    _slopes(zs)
    _closure()
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
