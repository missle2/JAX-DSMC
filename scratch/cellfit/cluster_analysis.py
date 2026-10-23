"""The 2026-10-11/12 cluster gates of the shape adjoint against FD: ``cellgate`` (the Kn 0.5
squash at production density with every backward fit paired, and six optc shapes along the
uniform shift and the cos 2 theta mode, Kn 0.5 and free-molecular) and ``biasgate`` (the
64-seed Kn 0.5 FD, the Kn 0.5 and free-molecular whole-run horizon).  Prints the tables
and draws ``figures/gates/bias_gate_cluster.png`` and ``figures/gates/cellgate_optc.png``.

    python3 scratch/cellfit/cluster_analysis.py
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
SQ_FITS = {  # sweep.py setting -> label
    f"S4K8{CELL}": "shipped cell:quad:adapt1.5",
    "S4K8cell:cubicv:adapt1.5": "cell:cubicv:adapt1.5",
    "S4K8quadratic_full": "per-query k = 32",
    "S4K8cell:quad": "cell:quad (unguarded)",
}
OPTC_FITS = {  # geom_gate.py tag -> label (kn); fm has only the shipped fit
    f"{CELL}_k32": "shipped, split (one-sided)",
    f"{CELL}_k32_twosided": "shipped, two-sided",
    "quadratic_full_k32_twosided": "per-query k = 32, two-sided",
}
SHAPES = ("c0", "a50", "a100", "a200", "s8a30", "g200")
# Okabe-Ito: distinguishable under the common colour-vision deficiencies
COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")


def se(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def rows_of(pattern):
    return [json.loads(line) for f in sorted(glob.glob(pattern)) for line in open(f)]


def assembled(run, variant="along_pool_plane"):
    """sweep.py's per-setting record -> dD/ds (scratch/cellfit/pair.py's sum)."""
    t = run["variants"][variant]
    return (run["tally"] + t["x"] + t["v"] + run["volume"] + run["order"] + run["score"]
            + t["sil"] + run["timing"])


def fd_precise():
    rows = rows_of("scratch/bias/cluster/kn_fd*.jsonl")
    a = np.array([r["fd"] for r in rows])  # h = 0.1, 0.2
    d = a[:, 0] - a[:, 1]
    print(f"Kn 0.5 FD, {len(a)} seeds: h 0.1 {a[:, 0].mean():.5f} +- {se(a[:, 0]):.5f}, h 0.2 "
          f"{a[:, 1].mean():.5f} +- {se(a[:, 1]):.5f}; (h0.1 - h0.2) / FD "
          f"{d.mean() / a[:, 1].mean() * 100:+.2f} +- {se(d) / a[:, 1].mean() * 100:.2f} %")
    return a[:, 1].mean(), se(a[:, 1])


def squash():
    """Per s: {label: (per-seed (adjoint - FD) / FD, mean gradient)}."""
    out = {}
    for s in ("1", "0.6"):
        rows = rows_of(f"scratch/cellfit/cluster/sq_s{s}_seed*.jsonl")
        fd = np.array([r["fd"][0] for r in rows])
        print(f"squash s = {s}, {len(rows)} seeds, paired CRN FD (h 0.2) {fd.mean():.5f} "
              f"+- {se(fd):.5f}")
        out[s] = {}
        for key, label in SQ_FITS.items():
            g = np.array([assembled(r["runs"][key]) for r in rows])
            rel = (g - fd) / fd.mean()
            out[s][label] = rel
            secs = np.median([r["runs"][key]["seconds"] for r in rows])
            print(f"  {label:28s} {g.mean():.5f} +- {se(g):.5f}  vs FD {rel.mean() * 100:+.2f} +- "
                  f"{se(rel) * 100:.2f} %  worst seed {np.abs(rel).max() * 100:.0f} %  {secs:.0f} s")
        q = np.array([assembled(r["runs"][f"S4K8{CELL}"]) for r in rows])
        c = np.array([assembled(r["runs"]["S4K8cell:cubicv:adapt1.5"]) for r in rows])
        print(f"  cubicv - shipped, paired: {((c - q) / q).mean() * 100:+.2f} +- "
              f"{se((c - q) / q) * 100:.2f} %")
    return out


def horizons():
    """{case: (300 vs FD, 1000 vs FD, 1000 - 300), each per-seed relative}."""
    out = {}
    for name, pattern in (("Kn 0.5, s = 1", "kn_hor_seed*"), ("FM, s = 1", "fm_hor_s1.0"),
                          ("FM, s = 0.35", "fm_hor_s0.35")):
        rows = rows_of(f"scratch/bias/cluster/{pattern}.jsonl")
        fd = np.array([r["fd"][0] for r in rows])
        g = {m: np.array([r["horizons"][m]["variants"]["along_pool_plane"]
                          + r["horizons"][m]["timing"] for r in rows]) for m in ("300", "1000")}
        out[name] = ((g["300"] - fd) / fd, (g["1000"] - fd) / fd, (g["1000"] - g["300"]) / g["300"])
        print(f"horizon {name:13s} {len(rows)} seeds: " + "  ".join(
            f"{lab} {v.mean() * 100:+.2f} +- {se(v) * 100:.2f} %"
            for lab, v in zip(("300 vs FD", "1000 vs FD", "1000 - 300"), out[name], strict=True)))
    return out


def optc():
    """{(regime, direction): {label: {shape: per-seed (adjoint - FD) / |FD shift|}}}."""
    out = {}
    for reg in ("kn", "fm"):
        for shape in SHAPES:
            rows = rows_of(f"scratch/cellfit/cluster/optc_{shape}_{reg}_b*.jsonl")
            scale = abs(np.mean([r["fd"]["shift"][0] for r in rows]))
            fits = OPTC_FITS if reg == "kn" else {f"{CELL}_k32": "shipped (free-molecular)"}
            for d in ("shift", "mode2"):
                fd = np.array([r["fd"][d][0] for r in rows])
                line = f"optc {shape:5s} {reg} {d:5s} FD/|FD shift| {fd.mean() / scale:+.3f}"
                for key, label in fits.items():
                    g = np.array([r["runs"][key]["directional"][d] for r in rows])
                    rel = (g - fd) / scale
                    out.setdefault((reg, d), {}).setdefault(label, {})[shape] = rel
                    line += (f" | {label[:12]} {rel.mean():+.3f} +- {se(rel):.3f}"
                             f" (sd ratio {np.std(g - fd) / np.std(fd):.1f})")
                print(line)
    return out


def plot_squash(sq, hor, fd):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = list(SQ_FITS.values())
    for j, s in enumerate(("1", "0.6")):
        for i, label in enumerate(labels):
            rel = sq[s][label] * 100
            x = i + (j - 0.5) * 0.3
            ax.plot(np.full(len(rel), x), rel, MARKERS[j], color=COLOURS[i], alpha=0.25, ms=4)
            ax.errorbar(x, rel.mean(), se(rel), fmt=MARKERS[j], color=COLOURS[i], ms=8,
                        capsize=4, lw=2, mec="white",
                        label=f"s = {s} ({len(rel)} seeds)" if i == 0 else None)
    ax.axhline(0, color="0.4", lw=1)
    ax.axhspan(-fd[1] / fd[0] * 100, fd[1] / fd[0] * 100, color="0.85", zorder=0,
               label="64-seed FD, 1 se")
    ax.set_xticks(range(len(labels)), [lab.replace(" (", "\n(").replace("cell:", "cell:\n")
                                       for lab in labels], fontsize=8)
    ax.set_ylabel("adjoint − paired FD  (% of FD)")
    ax.set_title("Kn 0.5 squash, fnum ×1, 4 × 8: backward fits (mean ± se; dots: seeds)",
                 fontsize=10)
    ax.set_ylim(-12, 10)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    names = list(hor)
    for k, (lab, mk) in enumerate((("captured 300 vs FD", "o"), ("whole run (1000) vs FD", "s"),
                                    ("1000 − 300, paired", "D"))):
        for i, name in enumerate(names):
            v = hor[name][k] * 100
            ax.errorbar(i + (k - 1) * 0.22, v.mean(), se(v), fmt=mk, color=COLOURS[k], ms=7,
                        capsize=4, lw=2, mec="white", label=lab if i == 0 else None)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(len(names)), [f"{n}\n({len(hor[n][0])} seeds)" for n in names],
                  fontsize=8)
    ax.set_ylabel("% of FD (shipped fit)")
    ax.set_title("Captured horizon: the last 300 steps vs the whole run", fontsize=10)
    ax.set_ylim(-4, 4.5)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = figure_path("gates", "bias_gate_cluster.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def plot_optc(oc):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for col, d in enumerate(("shift", "mode2")):
        for row, reg in enumerate(("kn", "fm")):
            ax = axes[row, col]
            fits = oc[(reg, d)]
            n = len(fits)
            for i, (label, per) in enumerate(fits.items()):
                c = COLOURS[i] if reg == "kn" else COLOURS[3]
                for k, shape in enumerate(SHAPES):
                    v = per[shape]
                    x = k + (i - (n - 1) / 2) * 0.22
                    ax.errorbar(x, v.mean(), se(v), fmt=MARKERS[i], color=c, ms=7, capsize=3,
                                lw=1.8, mec="white", label=label if k == 0 else None)
            ax.axhline(0, color="0.4", lw=1)
            ax.set_ylim(-0.8, 0.8)
            ax.set_title(f"{'Kn 0.5' if reg == 'kn' else 'free-molecular'}: along "
                         f"{'the uniform shift' if d == 'shift' else 'cos 2θ'}", fontsize=10)
            if col == 0:
                ax.set_ylabel("(adjoint − FD) / |FD shift|")
            ax.legend(fontsize=7, loc="lower left")
    for ax in axes[1]:
        ax.set_xticks(range(len(SHAPES)), ["circle", "S1 it 50", "S1 it 100", "S1 it 200",
                                           "S8 it 30", "GD it 200"], fontsize=8)
    fig.suptitle("optc level-set iterates, 16 CRN seeds each, fnum ×1, FD h = 0.05 "
                 "(mean ± se)", fontsize=11)
    fig.tight_layout()
    path = figure_path("gates", "cellgate_optc.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main():
    fd = fd_precise()
    sq = squash()
    hor = horizons()
    oc = optc()
    plot_squash(sq, hor, fd)
    plot_optc(oc)


if __name__ == "__main__":
    main()
