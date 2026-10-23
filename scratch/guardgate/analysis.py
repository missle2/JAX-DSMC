"""Figures of the leverage-guard gates (scratch/guardgate/cluster_job.sh, 2026-10-18/19).

    PYTHONPATH=. python3 scratch/guardgate/analysis.py [scratch/guardgate/cluster]

Every fit is paired on one forward per seed against the seed's CRN central FD.  Panels:
(a) the blunt Kn 0.5 squash (where the cubic beat the quadratic), (b) the thin M 10 squash
against body thickness, (c) the optm Kn 0.5 M 10 iterates along the shift and the nose bump,
(d) the carried value's largest |lambda| (the blow-up's signature), (e) how often the guard
lowers the order, (f) the stored device gates, guard against the shipped cubic.  Writes
``examples/mesh/figures/optc/gates/guardgate.png`` and ``<dir>/analysis.json``.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/thingate")
from figure_paths import figure_path  # noqa: E402
from summarise import assembled, rows, se  # noqa: E402

FITS = {
    "cell:cubicv:adapt1.5:align0.9:both": "C",
    "cell:quad:adapt1.5:align0.9:both": "Q",
    "cell:cubicv:adapt1.5:guard8:align0.9:both": "G8",
    "cell:cubicv:adapt1.5:guard32:align0.9:both": "G32",
    "cell:cubicv:adapt1.5:guard128:align0.9:both": "G128",
}
LABELS = {
    "C": "C  cubic-in-v (shipped)",
    "Q": "Q  quadratic",
    "G8": "G8  guarded cubic, h 8",
    "G32": "G32  guarded cubic, h 32",
    "G128": "G128  guarded cubic, h 128",
    "G2": "G2  guarded cubic, h 2",
}
COLOURS = {"C": "#0072B2", "Q": "#E69F00", "G8": "#009E73", "G32": "#CC79A7", "G128": "#56B4E9",
           "G2": "#D55E00"}  # fmt: skip
ORDER = ("C", "Q", "G8", "G32", "G128")
THIN_S = ("0.193", "0.25", "0.3", "0.45")
ITERATES = (30, 58, 156, 246)
CLIP = 30.0  # % : beyond, a marker on the panel edge with its value


def name_of(tag):
    for basis, name in FITS.items():
        if basis in tag:
            return name
    if "guard2:" in tag:
        return "G2"
    return None


def squash(root, job_glob):
    """Per fit: per-seed (gradient - FD) / |mean FD| (%) and max |lambda|, over the jobs."""
    out, fds = {}, []
    for job in job_glob:
        path = f"{root}/{job}.jsonl"
        if not os.path.exists(path):
            continue
        for r in rows(path):
            fds.append(r["fd"][0])
            for tag, run in r["runs"].items():
                n = name_of(tag)
                rec = out.setdefault(n, {"diff": [], "lam": [], "levels": []})
                rec["diff"].append(assembled(run) - r["fd"][0])
                rec["lam"].append(run.get("lambda_absmax") or np.nan)
                if run.get("guard_levels"):
                    rec["levels"].append(run["guard_levels"])
    ref = abs(np.mean(fds))
    for rec in out.values():
        d = 100 * np.asarray(rec["diff"]) / ref
        rec.update(pct=float(d.mean()), pct_se=se(d), lam_med=float(np.nanmedian(rec["lam"])),
                   lowered=1 - float(np.mean([lv[-1] for lv in rec["levels"]]))
                   if rec["levels"] else None)  # fmt: skip
    return out, ref


def optm(root, it):
    """Per fit and direction: (g . d - FD) / |mean FD along the shift| (%), paired per seed."""
    rs = rows(f"{root}/om_it{it}_b0.jsonl")
    fd = {d: np.array([r["fd"][d][0] for r in rs]) for d in ("shift", "nose")}
    ref = abs(fd["shift"].mean())
    out = {}
    for tag in rs[0]["runs"]:
        n = name_of(tag)
        runs = [r["runs"][tag] for r in rs]
        rec = {"lam_med": float(np.median([x["lambda_absmax"] for x in runs]))}
        for d in ("shift", "nose"):
            diff = 100 * (np.array([x["directional"][d] for x in runs]) - fd[d]) / ref
            rec[d] = (float(diff.mean()), se(diff))
        lv = [x["guard_levels"] for x in runs if x.get("guard_levels")]
        rec["lowered"] = 1 - float(np.mean([v[-1] for v in lv])) if lv else None
        out[n] = rec
    return out, {d: float(v.mean()) for d, v in fd.items()}


def unit(root):
    """Per stored gate and guard: (guard - C) / |mean C| (%), paired per seed."""
    path = f"{root}/unit.jsonl"
    if not os.path.exists(path):
        return {}
    by = {}
    for r in rows(path):
        by.setdefault(r["artifact"], {})[name_of(r["basis"])] = np.asarray(r["grad"], float)
    out = {}
    for art, d in by.items():
        c = d.get("C")
        if c is None:
            continue
        out[art] = {n: (float(np.mean(100 * (g - c) / abs(c.mean()))),
                        se(100 * (g - c) / abs(c.mean())))
                    for n, g in d.items() if n != "C"}  # fmt: skip
    return out


def clipped(ax, x, y, e, colour, marker="o", label=None):
    """An error bar, or a marker pinned at +-CLIP with the value written beside it."""
    if abs(y) <= CLIP:
        ax.errorbar(x, y, 2 * e, fmt=marker, ms=6, capsize=2, color=colour, label=label,
                    mec="white", mew=0.6)  # fmt: skip
        return
    edge = np.sign(y) * CLIP
    ax.plot(x, edge, marker="^" if y > 0 else "v", ms=8, color=colour, ls="none", label=label)
    ax.annotate(f"{y:+.0f} %", (x, edge), xytext=(4, -10 if y > 0 else 4),
                textcoords="offset points", fontsize=6.5, color="0.25")  # fmt: skip


def style(ax, title, ylabel):
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def panel_blunt(ax, blunt):
    for i, n in enumerate(ORDER):
        r = blunt[n]
        clipped(ax, i, r["pct"], r["pct_se"], COLOURS[n])
    ax.set_xticks(range(len(ORDER)), ORDER)
    ax.set_ylim(-7, 3)
    style(ax, "(a) blunt Kn 0.5 squash, M 0.32, s 0.6 (16 seeds)", "adjoint - FD  (% of FD)")


def panel_thin(ax, thin):
    width = 0.14
    for j, n in enumerate(ORDER):
        for i, s in enumerate(THIN_S):
            r = thin[s][n]
            clipped(ax, i + (j - 2) * width, r["pct"], r["pct_se"], COLOURS[n],
                    label=LABELS[n] if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(THIN_S)),
                  [f"{3 * float(s):.2f} m\n({3 * float(s) / 0.5:.1f} cells)" for s in THIN_S])  # fmt: skip
    ax.set_ylim(-CLIP * 1.15, CLIP * 1.15)
    ax.set_xlabel("body thickness", fontsize=8.5)
    style(ax, "(b) thin Kn 0.5 squash, M 10 (8 seeds each)", "adjoint - FD  (% of FD)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def panel_optm(ax, om):
    width = 0.14
    for k, d in enumerate(("shift", "nose")):
        for j, n in enumerate(ORDER):
            for i, it in enumerate(ITERATES):
                y, e = om[it][0][n][d]
                x = 2 * i + k + (j - 2) * width
                clipped(ax, x, y, e, COLOURS[n], marker="o" if d == "shift" else "s")
    ticks = [2 * i + k for i in range(len(ITERATES)) for k in range(2)]
    ax.set_xticks(ticks, [f"it {it}\n{d}" for it in ITERATES for d in ("shift", "nose")],
                  fontsize=7.5)  # fmt: skip
    for i in range(1, len(ITERATES)):
        ax.axvline(2 * i - 0.5, color="0.8", lw=0.6)
    ax.set_ylim(-CLIP * 1.15, CLIP * 1.15)
    style(ax, "(c) optm Kn 0.5 M 10 iterates (4 seeds): g . d - FD",
          "% of |FD along the shift|")  # fmt: skip


def panel_lambda(ax, blunt, thin, om):
    cases = [("blunt\ns 0.6", blunt)] + [(f"M10\ns {s}", thin[s]) for s in THIN_S]
    cases += [(f"optm\nit {it}", om[it][0]) for it in ITERATES]
    for j, n in enumerate(ORDER):
        ys = [c[n]["lam_med"] for _, c in cases]
        ax.plot(range(len(cases)), ys, marker="o", ms=5, lw=1.2, color=COLOURS[n], label=n,
                mec="white", mew=0.6)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)), [c for c, _ in cases], fontsize=7)
    style(ax, "(d) largest |lambda| carried (median over seeds)", "|lambda| max")
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper left")


def panel_levels(ax, blunt, thin, om):
    cases = [("blunt\ns 0.6", blunt)] + [(f"M10\ns {s}", thin[s]) for s in THIN_S]
    cases += [(f"optm\nit {it}", om[it][0]) for it in ITERATES]
    width = 0.27
    for j, n in enumerate(("G8", "G32", "G128")):
        ys = [100 * (c[n]["lowered"] or 0.0) for _, c in cases]
        ax.bar(np.arange(len(cases)) + (j - 1) * width, ys, width * 0.92, color=COLOURS[n],
               label=LABELS[n])  # fmt: skip
    ax.set_xticks(range(len(cases)), [c for c, _ in cases], fontsize=7)
    style(ax, "(e) queries the guard answers below cubic order", "% of fit evaluations")
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def panel_unit(ax, un):
    arts = sorted(un)
    short = [a.replace("ensemble_", "").replace("_eps0.005", "").replace("_dev", "")[:30]
             for a in arts]  # fmt: skip
    width = 0.27
    for j, n in enumerate(("G2", "G8", "G32")):
        for i, a in enumerate(arts):
            if n in un[a]:
                y, e = un[a][n]
                clipped(ax, i + (j - 1) * width, y, e, COLOURS[n],
                        label=LABELS[n] if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(arts)), short, rotation=70, ha="right", fontsize=6)
    ax.set_ylim(-CLIP * 1.15, CLIP * 1.15)
    style(ax, "(f) stored device gates (4 seeds): guard - shipped cubic", "% of C")
    ax.legend(fontsize=7, frameon=False, loc="lower left")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scratch/guardgate/cluster"
    blunt, _ = squash(root, [f"sq_M032_s0.6_b{j}" for j in range(4)])
    thin = {s: squash(root, [f"sq_M10_s{s}_b{j}" for j in range(2)])[0] for s in THIN_S}
    om = {it: optm(root, it) for it in ITERATES}
    un = unit(root)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    panel_blunt(axes[0, 0], blunt)
    panel_thin(axes[0, 1], thin)
    panel_optm(axes[0, 2], om)
    panel_lambda(axes[1, 0], blunt, thin, om)
    panel_levels(axes[1, 1], blunt, thin, om)
    panel_unit(axes[1, 2], un)
    fig.suptitle("Leverage-guarded cubic backward fit (guardgate): every fit paired on one "
                 "forward per seed against the CRN FD; error bars +-2 s.e.", fontsize=11)  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc/gates", "guardgate.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)

    def strip(d):
        return {n: {k: v for k, v in r.items() if k not in ("diff", "lam", "levels")}
                for n, r in d.items()}  # fmt: skip

    summary = {"blunt": strip(blunt), "thin": {s: strip(v) for s, v in thin.items()},
               "optm": {it: {"fd": v[1], "fits": v[0]} for it, v in om.items()}, "unit": un}  # fmt: skip
    with open(os.path.join(root, "analysis.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for name, rec in (("blunt", blunt), *((f"thin {s}", thin[s]) for s in THIN_S)):
        print(name, "  ".join(f"{n} {rec[n]['pct']:+.2f}±{rec[n]['pct_se']:.2f}%" for n in ORDER))
    for it in ITERATES:
        print(f"optm {it}", "  ".join(
            f"{n} sh {om[it][0][n]['shift'][0]:+.2f}±{om[it][0][n]['shift'][1]:.2f} "
            f"no {om[it][0][n]['nose'][0]:+.2f}±{om[it][0][n]['nose'][1]:.2f}" for n in ORDER))  # fmt: skip


if __name__ == "__main__":
    main()
