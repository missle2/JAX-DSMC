"""Tables of the 2026-10-12 ``gapgate`` cluster results (scratch/gapgate/cluster_job.sh).

    python3 scratch/gapgate/analysis.py
"""

import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402

CELL = "cell:quad:adapt1.5:align0.9:both"
CUBIC = "cell:cubicv:adapt1.5:align0.9:both"
FITS = {CELL: "shipped", CUBIC: "cubicv"}
SHAPES = ("c0", "a50", "a100", "a200", "s8a30", "g200")
GZ = ("0.05", "0.01", "0.2", "0")
TERMS = ("pathwise_tally", "pathwise_x", "pathwise_v", "timing", "order", "volume")
DIR = "scratch/gapgate/cluster"
# Okabe-Ito: distinguishable under the common colour-vision deficiencies
COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
SHAPE_NAMES = {"c0": "circle", "a50": "S1 it 50", "a100": "S1 it 100", "a200": "S1 it 200",
               "s8a30": "S8 it 30", "g200": "GD it 200"}


def se(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def rows_of(pattern):
    return [json.loads(line) for f in sorted(glob.glob(pattern)) for line in open(f)]


def assembled(run):
    t = run["variants"]["along_pool_plane"]
    return (run["tally"] + t["x"] + t["v"] + run["volume"] + run["order"] + run["score"]
            + t["sil"] + run["timing"])


def pm(v, scale=100, unit=" %"):
    return f"{np.mean(v) * scale:+.2f} +- {se(v) * scale:.2f}{unit}"


def sqh():
    out = {}
    for t in ("1000", "2000"):
        rows = rows_of(f"{DIR}/sqh_t{t}_seed*.jsonl")
        fd = np.array([r["fd"][0] for r in rows])
        obj = np.array([r["runs"][f"S4K8{CELL}_h300"]["objective"] for r in rows])
        print(f"sqh T = {t}: {len(rows)} seeds, FD {fd.mean():.5f} +- {se(fd):.5f}, "
              f"objective {obj.mean():.5f}")
        g = {}
        for b, lab in FITS.items():
            for h in ("300", "1000"):
                g[lab, h] = np.array([assembled(r["runs"][f"S4K8{b}_h{h}"]) for r in rows])
                rel = (g[lab, h] - fd) / fd.mean()
                out[t, lab, h] = rel
                print(f"  {lab:8s} h{h:4s} vs FD {pm(rel)}")
        for lab in FITS.values():
            d = (g[lab, "1000"] - g[lab, "300"]) / fd.mean()
            print(f"  {lab:8s} 1000 - 300, paired {pm(d)}")
        for h in ("300", "1000"):
            d = (g["cubicv", h] - g["shipped", h]) / fd.mean()
            print(f"  cubicv - shipped h{h}, paired {pm(d)}")
        # which term moves
        for lab_b, b in (("shipped", CELL), ("cubicv", CUBIC)):
            parts = {}
            for h in ("300", "1000"):
                rs = [r["runs"][f"S4K8{b}_h{h}"] for r in rows]
                parts[h] = {k: np.array([x["variants"]["along_pool_plane"][k] for x in rs])
                            for k in ("x", "v")}
                for k in ("volume", "order", "timing", "tally"):
                    parts[h][k] = np.array([x[k] for x in rs])
            print(f"  {lab_b} term shift 1000 - 300 (% FD): " + "  ".join(
                f"{k} {pm((parts['1000'][k] - parts['300'][k]) / fd.mean())}"
                for k in ("x", "v", "volume", "order")))
    return out


def directional(pattern, tags, fd_index=0):
    rows = rows_of(pattern)
    scale = abs(np.mean([r["fd"]["shift"][fd_index] for r in rows]))
    res = {}
    for d in ("shift", "mode2"):
        fd = np.array([r["fd"][d][fd_index] for r in rows])
        for tag in tags:
            g = np.array([r["runs"][tag]["directional"][d] for r in rows])
            res[tag, d] = ((g - fd) / scale, np.std(g - fd, ddof=1) / np.std(fd, ddof=1))
        res["fd", d] = fd / scale
    return rows, scale, res


def circh():
    out = {}
    for t in ("1000", "2000"):
        tags = [f"{b}_k32_h{h}" for b in FITS for h in ("300", "1000")]
        rows, scale, res = directional(f"{DIR}/circh_t{t}_b*.jsonl", tags)
        print(f"circh T = {t}: {len(rows)} seeds, FD shift {-scale:.5f}, FD mode2 / |shift| "
              f"{res['fd', 'mode2'].mean():+.3f}")
        for tag in tags:
            lab = f"{FITS[tag.split('_k32')[0]]} h{tag.rsplit('_h', 1)[1]}"
            out[t, lab] = {d: res[tag, d][0] for d in ("shift", "mode2")}
            print(f"  {lab:14s} shift {pm(res[tag, 'shift'][0], 1, '')}  mode2 "
                  f"{pm(res[tag, 'mode2'][0], 1, '')}")
        for b, lab in FITS.items():
            for d in ("shift",):
                g = {h: np.array([r["runs"][f"{b}_k32_h{h}"]["directional"][d] for r in rows])
                     for h in ("300", "1000")}
                print(f"  {lab} 1000 - 300 along {d}, paired: {pm((g['1000'] - g['300']) / scale, 1, '')}")
        for h in ("300",):
            terms = {}
            for b, lab in FITS.items():
                for k in TERMS:
                    terms[lab, k] = np.array([r["runs"][f"{b}_k32_h{h}"]["terms"][k]["shift"]
                                              for r in rows]) / scale
            print(f"  terms along shift h{h} (shipped): " + "  ".join(
                f"{k.replace('pathwise_', '')} {terms['shipped', k].mean():+.3f}" for k in TERMS))
    return out


def cub():
    out = {}
    for shape in SHAPES[1:]:
        tags = [f"{CELL}_k32", f"{CUBIC}_k32"]
        rows, scale, res = directional(f"{DIR}/cub_{shape}_b*.jsonl", tags)
        lam = {t: np.max([r["runs"][t]["lambda_rms"][:2] for r in rows]) for t in tags}
        line = f"cub {shape:5s} {len(rows)} seeds FD mode2/|shift| {res['fd', 'mode2'].mean():+.3f}"
        for tag in tags:
            lab = FITS[tag[:-4]]
            out[shape, lab] = {d: res[tag, d] for d in ("shift", "mode2")}
            line += (f" | {lab} shift {pm(res[tag, 'shift'][0], 1, '')} (sd {res[tag, 'shift'][1]:.1f})"
                     f" mode2 {pm(res[tag, 'mode2'][0], 1, '')} (sd {res[tag, 'mode2'][1]:.1f})"
                     f" lam {lam[tag]:.1e}")
        print(line)
    return out


def fmgz():
    out = {}
    for shape in SHAPES:
        rows = rows_of(f"{DIR}/fmgz_{shape}_b*.jsonl")
        fds = {d: np.array([r["fd"][d] for r in rows]) for d in ("shift", "mode2")}
        scale = abs(fds["shift"][:, 1].mean())
        lin = "  ".join(f"{d} h.025/.05/.1 " + "/".join(f"{fds[d][:, i].mean() / scale:+.3f}"
                                                       for i in range(3)) for d in fds)
        dl = (fds["mode2"][:, 0] - fds["mode2"][:, 2]) / scale
        print(f"fmgz {shape:5s} {len(rows)} seeds FD {lin}  (mode2 h.025-h.1 {pm(dl, 1, '')})")
        for g in GZ:
            tag = f"{CELL}_k32_gz{g}"
            line = f"   gz {g:4s}"
            for d in ("shift", "mode2"):
                gv = np.array([r["runs"][tag]["directional"][d] for r in rows])
                fd = fds[d][:, 1]  # h = 0.05, as fmsm and the cellgate optc runs
                rel = (gv - fd) / scale
                out[shape, g, d] = (rel, np.std(gv - fd, ddof=1) / np.std(fd, ddof=1))
                line += f" | {d} {pm(rel, 1, '')} (sd {out[shape, g, d][1]:.1f})"
            tt = {k: np.mean([r["runs"][tag]["terms"][k]["mode2"] for r in rows]) / scale
                  for k in ("pathwise_tally", "pathwise_v", "pathwise_x")}
            line += "  mode2 terms " + " ".join(f"{k[9:]} {v:+.3f}" for k, v in tt.items())
            print(line)
    return out


def fmsm():
    out = {}
    for shape in SHAPES[1:]:
        if shape == "s8a30":
            continue
        rows = rows_of(f"{DIR}/fmsm_{shape}_b*.jsonl")
        tag = f"{CELL}_k32"
        scale = abs(np.mean([r["fd"]["shift"][0] for r in rows]))
        line = f"fmsm {shape:5s} {len(rows)} seeds"
        for d in ("shift", "mode2"):
            fd = np.array([r["fd"][d][0] for r in rows])
            g = np.array([r["runs"][tag]["directional"][d] for r in rows])
            ratio = np.std(g - fd, ddof=1) / np.std(fd, ddof=1)
            out[shape, d] = ((g - fd) / scale, ratio)
            line += f" | {d} {pm((g - fd) / scale, 1, '')} sd ratio {ratio:.1f} (FD/|shift| {fd.mean() / scale:+.3f})"
        print(line)
    return out


def sqfm():
    out = {}
    for s in ("1", "0.35"):
        rows = rows_of(f"{DIR}/sqfm_s{s}_b*.jsonl")
        fd = np.array([r["fd"][0] for r in rows])
        print(f"sqfm s = {s}: {len(rows)} seeds FD {fd.mean():.4e} +- {se(fd):.1e}")
        for g in GZ:
            run = [r["runs"][f"S1K2{CELL}_gz{g}"] for r in rows]
            a = np.array([assembled(x) for x in run])
            tally = np.array([x["tally"] for x in run])
            v = np.array([x["variants"]["along_pool_plane"]["v"] for x in run])
            out[s, g] = (a - fd) / fd.mean()
            print(f"  gz {g:4s} vs FD {pm(out[s, g])}  tally {tally.mean() / fd.mean() * 100:.2f} %"
                  f"  v {v.mean() / fd.mean() * 100:+.2f} %")
        a0 = np.array([assembled(r["runs"][f"S1K2{CELL}_gz0"]) for r in rows])
        a5 = np.array([assembled(r["runs"][f"S1K2{CELL}_gz0.05"]) for r in rows])
        print(f"  exact - shipped, paired {pm((a0 - a5) / fd.mean())}")
    return out


def errbar(ax, x, v, i, label):
    ax.errorbar(x, np.mean(v), se(v), fmt=MARKERS[i], color=COLOURS[i], ms=7, capsize=3,
                lw=1.8, mec="white", label=label)


def plot_kn(sq, ci):
    settings = [("shipped", "300"), ("shipped", "1000"), ("cubicv", "300"), ("cubicv", "1000")]
    ticks = [f"{lab}\n{h} captured" for lab, h in settings]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, (t, name) in enumerate((("1000", "T = 1000 (production)"),
                                   ("2000", "T = 2000 (box filled)"))):
        for k, (lab, h) in enumerate(settings):
            x = k + (i - 0.5) * 0.25
            errbar(axes[0], x, sq[t, lab, h] * 100, i, name if k == 0 else None)
            for j, d in enumerate(("shift", "mode2")):
                errbar(axes[j + 1], x, ci[t, f"{lab} h{h}"][d] * 100, i, name if k == 0 else None)
    titles = ("squash s = 1: (adjoint − FD) / FD", "circle, uniform shift: (adjoint − FD) / |FD|",
              "circle, cos 2θ: (adjoint − FD) / |FD shift|")
    for ax, title in zip(axes, titles, strict=True):
        ax.axhline(0, color="0.4", lw=1)
        ax.set_xticks(range(4), ticks, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("%")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Kn 0.5, fnum ×1, 16 CRN seeds per point (mean ± se): backward fit × "
                 "captured horizon × run length", fontsize=11)
    fig.tight_layout()
    path = figure_path("", "gapgate_kn.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def plot_fm(gz, sm, sf):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    ax = axes[0]
    series = [(f"grazing cut-off {g}", g) for g in ("0", "0.05", "0.2")]
    for i, (lab, g) in enumerate(series):
        ax.plot(range(len(SHAPES)), [gz[s, g, "shift"][1] for s in SHAPES], MARKERS[i] + "-",
                color=COLOURS[i], lw=1.5, ms=7, mec="white", label=lab)
    xs = [SHAPES.index(s) for s in ("a50", "a100", "a200", "g200")]
    ax.plot(xs, [sm[s, "shift"][1] for s in ("a50", "a100", "a200", "g200")], MARKERS[3],
            color=COLOURS[3], ms=9, mec="white", label="smoothed shape (σ 12.5 cm), 0.05")
    ax.set_yscale("log")
    ax.axhline(1, color="0.4", lw=1)
    ax.set_ylabel("sd(adjoint − FD) / sd(FD) over seeds")
    ax.set_title("Seed scatter along the uniform shift", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for k, s in enumerate(SHAPES):
        f = np.array([r["fd"]["shift"] for r in rows_of(f"{DIR}/fmgz_{s}_b*.jsonl")])
        f = f / abs(f[:, 1].mean())
        for i, h in enumerate(("0.025", "0.05", "0.1")):
            errbar(ax, k + (i - 1) * 0.22, -f[:, i] * 100, i, f"FD h = {h}" if k == 0 else None)
    ax.set_ylabel("−FD / |FD (h 0.05)|  (%)")
    ax.set_title("FD along the shift: step-size dependence", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    cats = [("squash s = 1", sf["1"]), ("squash s = 0.35", sf["0.35"]),
            ("circle shift", "c0", "shift"), ("circle cos 2θ", "c0", "mode2"),
            ("S8 it 30 shift", "s8a30", "shift"), ("S8 it 30 cos 2θ", "s8a30", "mode2")]
    for k, cat in enumerate(cats):
        for i, g in enumerate(("0", "0.05", "0.2")):
            v = cat[1][g] if len(cat) == 2 else gz[cat[1], g, cat[2]][0]
            errbar(ax, k + (i - 1) * 0.22, v * 100, i, f"cut-off {g}" if k == 0 else None)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(len(cats)), [c[0].replace(" ", "\n", 1) for c in cats], fontsize=8)
    ax.set_ylabel("(adjoint − FD) / reference  (%)")
    ax.set_title("Biases on the FD-linear cases", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    for ax in axes[:2]:
        ax.set_xticks(range(len(SHAPES)), [SHAPE_NAMES[s].replace(" it", "\nit") for s in SHAPES],
                      fontsize=8)
    fig.suptitle("Free-molecular, fnum ×1, 16 CRN seeds (mean ± se); reference = |FD shift| "
                 "(optc) or FD (squash)", fontsize=11)
    fig.tight_layout()
    path = figure_path("", "gapgate_fm.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main():
    sq = sqh()
    ci = circh()
    cub()
    gz = fmgz()
    sm = fmsm()
    sf = sqfm()
    plot_kn(sq, ci)
    plot_fm(gz, sm, {s: {g: sf[s, g] for g in GZ} for s in ("1", "0.35")})


if __name__ == "__main__":
    main()
