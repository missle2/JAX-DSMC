"""Round-3 analysis (2026-11-14): lccv3 (rich 12-output MLP sweep, 11 inputs) against lccv
v2 (rate MLP, 9 inputs); lcextra (the lcrich models and the 9-input Lasso on all twelve
cases, rich 2x32 on the circle).

    PYTHONPATH=. python scratch/lc_round3.py

Figures: examples/learned_collision/figures/round3_*.png; tables to stdout.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc_round2 as r2

from examples.learned_collision import sweep_report as sr
from examples.learned_collision.plot_round1 import COLOR, INK, MUTED, _style
from jax_dsmc.collide import relax

R, FIG, TRAIN, ALL = r2.R, r2.FIG, r2.TRAIN, r2.ALL
TEST_K = ("K2", "K3", "K5", "K6", "K7")
V_COLOR = {"v2": "#1baf7a", "v3": sr.LEARNED}
V_LABEL = {"v2": "v2: rate outputs, 9 inputs", "v3": "v3: rich outputs, 11 inputs"}
LR_MARK = sr.LR_MARK
FACTORS = ("depth", "width", "weight_decay", "lr")
EVAL_MODELS = (  # (label, eval file, colour, linear?)
    ("linear q1 (rates)", "eval_test_linear.json", "#e87ba4", True),
    ("lasso 9-in (seed CV)", "eval_test_lasso9.json", "#b5487a", True),
    ("rich linear", "eval_test_rich_linear.json", "#f2a7c1", True),
    ("MLP p0 1x8 (v2 choice)", "eval_test_mlp.json", "#1baf7a", False),
    ("rich MLP 2x32", "eval_test_rich_2x32.json", sr.LEARNED, False),
    ("rich MLP 3x32", "eval_test_rich_p7.json", "#7a1f30", False),
)


def points(tag):
    plan = r2.load(f"sweep_plan_{tag}.json")
    pts = sr.collect(plan, R, sr.base_rows(R), "integral")
    return plan, pts


# --------------------------------------------------------------------------- sweeps


def sweep_table(sweeps):
    print("\n== design sweep, v2 vs v3 (error / base, geometric; held-out +- s.e. over folds) ==")
    print(f"{'point':40s} {'v2 held':>13s} {'v3 held':>13s} {'v2 trn':>7s} {'v3 trn':>7s} "
          f"{'v2 spr':>7s} {'v3 spr':>7s}")  # fmt: skip
    for a, b in zip(sweeps["v2"][1], sweeps["v3"][1], strict=True):
        print(f"{sr._label(a):40s} {math.exp(a['heldout']):6.3f}+-{a['heldout_se']:.2f} "
              f"{math.exp(b['heldout']):6.3f}+-{b['heldout_se']:.2f} {math.exp(a['train']):7.3f} "
              f"{math.exp(b['train']):7.3f} {a['spread']:7.3f} {b['spread']:7.3f}")  # fmt: skip
    for v, (_, pts) in sweeps.items():
        ok = [p for p in pts if p["point"]["weight_decay"] < 0.01 or p["point"]["lr"] > 0.003]
        x = np.log([p["params"] for p in ok])
        for key in ("train", "heldout"):
            y = np.array([p[key] for p in ok])
            k, _ = np.polyfit(x, y, 1)
            print(f"  {v} {key:8s} log-error vs log-params slope {k:+.3f} (learning points only,"
                  f" n={len(ok)})")  # fmt: skip
        held = np.array([math.exp(p["heldout"]) for p in pts])
        print(f"  {v} held-out median over points {np.median(held):.3f}, best {held.min():.3f}")


def main_effects(pts, key):
    """Mean log-ratio at each level of each factor (the L9's balanced main effects)."""
    out = {}
    for f in FACTORS:
        levels = sorted({p["point"][f] for p in pts})
        out[f] = (levels, [float(np.mean([p[key] for p in pts if p["point"][f] == lv]))
                           for lv in levels])  # fmt: skip
    return out


def point_label(p):
    q = p["point"]
    return (f"p{p['index']}\n{q['depth']}x{q['width']}\nwd {q['weight_decay']:g}\n"
            f"lr {q['lr']:g}")  # fmt: skip


def fig_sweep(sweeps, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    ax = axes[0, 0]
    n = len(sweeps["v2"][1])
    for j, v in enumerate(("v2", "v3")):
        pts = sweeps[v][1]
        x = np.arange(n) + (j - 0.5) * 0.28
        y = [math.exp(p["heldout"]) for p in pts]
        lo = [math.exp(p["heldout"]) * (1 - math.exp(-p["heldout_se"])) for p in pts]
        hi = [math.exp(p["heldout"]) * (math.exp(p["heldout_se"]) - 1) for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=V_COLOR[v], ms=6, mec="white",
                    label=V_LABEL[v])  # fmt: skip
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([point_label(p) for p in sweeps["v2"][1]], fontsize=7)
    ax.set_ylabel("held-out error / base (5 folds, geometric)", color=INK)
    ax.set_title("held-out case error per design point", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax, key, lab in ((axes[0, 1], "train", "training error / base (2 all-case seeds)"),
                         (axes[1, 0], "spread", "seed spread |log r0 - log r1|/sqrt2")):
        for v in ("v2", "v3"):
            for p in sweeps[v][1]:
                y = math.exp(p[key]) if key == "train" else p[key]
                ax.plot(p["params"], y, LR_MARK[p["point"]["lr"]], color=V_COLOR[v], ms=7,
                        mec="white", mew=0.6, alpha=0.4 if p["point"]["weight_decay"] == 0.01
                        and p["point"]["lr"] < 0.01 else 1.0)  # fmt: skip
                ax.annotate(f"p{p['index']}", (p["params"], y), fontsize=6, color=V_COLOR[v],
                            xytext=(3, 3), textcoords="offset points")  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("parameters", color=INK)
        ax.set_ylabel(lab, color=INK, fontsize=9)
        ax.set_title(lab.split(" (")[0] + " vs size (faded: wd 0.01, low lr: did not learn)",
                     color=INK, fontsize=9, loc="left")  # fmt: skip
    ax = axes[1, 1]
    for v in ("v2", "v3"):
        for p in sweeps[v][1]:
            ax.plot(math.exp(p["train"]), math.exp(p["heldout"]), LR_MARK[p["point"]["lr"]],
                    color=V_COLOR[v], ms=7, mec="white", mew=0.6)  # fmt: skip
    for a, b in zip(sweeps["v2"][1], sweeps["v3"][1], strict=True):
        ax.annotate("", (math.exp(b["train"]), math.exp(b["heldout"])),
                    (math.exp(a["train"]), math.exp(a["heldout"])),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.7})  # fmt: skip
        ax.annotate(f"p{b['index']}", (math.exp(b["train"]), math.exp(b["heldout"])),
                    fontsize=6, color=V_COLOR["v3"], xytext=(3, -8),
                    textcoords="offset points")  # fmt: skip
    lim = (0.35, 1.3)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("training error / base", color=INK)
    ax.set_ylabel("held-out error / base", color=INK)
    ax.set_title("each point v2 -> v3 (marker: lr  o 0.001  s 0.003  ^ 0.01)", color=INK,
                 fontsize=9, loc="left")  # fmt: skip
    for a in axes.flat:
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_effects(sweeps, path):
    keys = (("train", "training"), ("heldout", "held-out"), ("spread", "seed spread"))
    fig, axes = plt.subplots(len(keys), len(FACTORS), figsize=(12, 7.5), sharey="row")
    for i, (key, lab) in enumerate(keys):
        for v in ("v2", "v3"):
            eff = main_effects(sweeps[v][1], key)
            for j, f in enumerate(FACTORS):
                levels, means = eff[f]
                y = np.exp(means) if key != "spread" else means
                axes[i, j].plot(range(len(levels)), y, "o-", color=V_COLOR[v], lw=1.5,
                                label=V_LABEL[v])  # fmt: skip
                axes[i, j].set_xticks(range(len(levels)))
                axes[i, j].set_xticklabels([f"{x:g}" for x in levels])
        axes[i, 0].set_ylabel(lab + (" / base" if key != "spread" else ""), color=INK)
    for j, f in enumerate(FACTORS):
        axes[0, j].set_title(f, color=INK, fontsize=10, loc="left")
        axes[-1, j].set_xlabel(f, color=INK)
    for a in axes.flat:
        _style(a)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Main effects of the L9 factors (mean over the three points at each level)",
                 color=INK, fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_heldout_cases(sweeps, path):
    fig, axes = plt.subplots(1, len(TRAIN), figsize=(14, 3.8), sharey=True)
    for ax, case in zip(axes, TRAIN, strict=True):
        for j, v in enumerate(("v2", "v3")):
            pts = sweeps[v][1]
            x = np.arange(len(pts)) + (j - 0.5) * 0.3
            y = [math.exp(p["held"].get(case, math.nan)) for p in pts]
            ax.bar(x, y, 0.3, color=V_COLOR[v], label=V_LABEL[v])
        ax.axhline(1.0, color=INK, lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(sweeps["v2"][1])))
        ax.set_xticklabels([f"p{i}" for i in range(len(sweeps["v2"][1]))], fontsize=7)
        ax.set_title(f"{case} held out", color=INK, fontsize=10, loc="left")
        _style(ax)
    axes[0].set_ylabel("held-out error / base (log)", color=INK)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heldout_case_table(sweeps):
    print("\n== held-out case error / base: median over the 9 points (and the best point) ==")
    for case in TRAIN:
        parts = []
        for v in ("v2", "v3"):
            r = np.array([math.exp(p["held"][case]) for p in sweeps[v][1] if case in p["held"]])
            parts.append(f"{v} median {np.median(r):.3f} best {r.min():.3f} "
                         f"(<1 in {(r < 1).sum()}/9)")  # fmt: skip
        print(f"  {case}: " + "   ".join(parts))


def k8_heldout(sweeps):
    """K8 held out (train on M <= 3): the M10 extrapolation, per point and version."""
    print("\n== K8 held out: int txx / int qx (base 1.164 / 0.944) ==")
    for v, (plan, _) in sweeps.items():
        rows = []
        for r in plan["runs"]:
            if r["validate"] == ["K8"]:
                d = r2.load(f"train_mlp_{r['tag']}.json")["validation"]["K8"]
                rows.append(f"{d['int_txx_ratio']:.3f}/{d['int_qx_ratio']:.3f}")
        print(f"  {v}: " + "  ".join(rows))


def fig_tradeoff_v3(sweeps, path):
    """K1 / K4 / K8 integrals of every v2 and v3 run (held-out folds hollow)."""
    base, zero = sr.base_rows(R), r2.load("train_linear_lasso_in11.json")["validation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    print("\n== v3 runs against the v2 runs' K txx/qx trade line (residual mean, sd) ==")
    for ax, case in zip(axes, ("K1", "K4", "K8"), strict=True):
        cloud = {}
        for v, (plan, _) in sweeps.items():
            xs, ys = [], []
            for r in plan["runs"]:
                d = r2.load(f"train_mlp_{r['tag']}.json")["validation"][case]
                x, y = d["int_txx_ratio"], d["int_qx_ratio"]
                held = r["validate"] == [case]
                ax.plot(x, y, "o", ms=5 if held else 4, mfc="none" if held else V_COLOR[v],
                        mec=V_COLOR[v], alpha=1.0 if held else 0.6,
                        mew=1.0 if held else 0)  # fmt: skip
                if not held:
                    xs.append(x)
                    ys.append(y)
            cloud[v] = (np.array(xs), np.array(ys))
        k, c = r2._line(ax, *cloud["v2"])
        for v in ("v2", "v3"):
            res = cloud[v][1] - (k * cloud[v][0] + c)
            print(f"  {case} {v}: residual mean {res.mean():+.3f} sd {res.std():.3f}; txx "
                  f"median {np.median(cloud[v][0]):.3f}, qx median {np.median(cloud[v][1]):.3f}")
        for src, fam in ((base, "base"), (zero, "zero")):
            ax.plot(src[case]["int_txx_ratio"], src[case]["int_qx_ratio"], "*", ms=13,
                    color=r2.FAMILY_COLOR[fam], mec="white", mew=0.6)  # fmt: skip
        ax.plot(1.0, 1.0, "+", color=INK, ms=16, mew=2)
        ax.set_xlabel(r"$\int\tau_{xx}$ / fine", color=INK)
        ax.set_ylabel(r"$\int q_x$ / fine", color=INK)
        ax.set_title(case, color=INK, fontsize=10, loc="left")
        _style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=V_COLOR[v], label=V_LABEL[v] +
                          " (case trained)") for v in ("v2", "v3")]  # fmt: skip
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED,
                           label="hollow: case held out"),
                plt.Line2D([], [], marker="*", ls="", ms=11, color=r2.FAMILY_COLOR["base"],
                           label="base (eval)"),
                plt.Line2D([], [], marker="*", ls="", ms=11, color=r2.FAMILY_COLOR["zero"],
                           label="zero model (validate)"),
                plt.Line2D([], [], ls="--", color=V_COLOR["v2"], label="v2 trade line")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- generalisation


def coverage():
    """q95 of each case's features over the training cases' largest q95 (and the max)."""
    f = r2.load("features_v4.json")
    hi = np.max([np.array(f["cases"][c]["quantiles"][4]) for c in TRAIN], axis=0)
    cov = {c: np.array(f["cases"][c]["quantiles"][4]) / hi for c in ALL}
    return f["features"], cov


def fig_generalisation(models, path):
    names, cov = coverage()
    base = sr.base_rows(R)
    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_axes([0.05, 0.14, 0.36, 0.76])
    z = np.log2(np.stack([cov[c] for c in ALL]))
    im = ax.imshow(z.T, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i, j] > 0.1:
                ax.text(i, j, f"{2 ** z[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if z[i, j] > 2 else INK)  # fmt: skip
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([f"{c}{'*' if c in TRAIN else ''}" for c in ALL], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("feature q95 / training max q95 (log2 colour; * training)", color=INK,
                 fontsize=9, loc="left")  # fmt: skip
    fig.colorbar(im, ax=ax, shrink=0.8, label="log2")
    ax = fig.add_axes([0.55, 0.14, 0.43, 0.76])
    ext = {c: float(cov[c].max()) for c in ALL}
    held = sorted(TEST_K, key=ext.get)
    xs = np.arange(len(held) + 1)
    print("\n== extrapolation (max feature q95 / training max) vs error / base, shocks ==")
    for lab, rows, col, linear in models:
        tr = r2.geo([r2.error_se(rows[c])[0] / r2.error_se(base[c])[0]
                     for c in TRAIN if c[0] == "K"])  # fmt: skip
        ys = [tr] + [r2.error_se(rows[c])[0] / r2.error_se(base[c])[0] for c in held]
        ax.plot(xs, ys, "s--" if linear else "o-", color=col, ms=6, lw=1.3, label=lab,
                mec="white", mew=0.5)  # fmt: skip
        print(f"  {lab:24s} train-shocks {tr:.2f}  " + "  ".join(
            f"{c} {y:.2f}" for c, y in zip(held, ys[1:], strict=True)))  # fmt: skip
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["training\nshocks"] + [f"{c}\n{ext[c]:.1f}x" for c in held])
    ax.set_xlabel("held-out shocks, by largest feature q95 / training max", color=INK)
    ax.set_ylabel("shock error / base operator's", color=INK)
    ax.set_title("linear models fail where the features extrapolate; MLPs saturate", color=INK,
                 fontsize=9, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style(ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  extrapolation per case: " + ", ".join(f"{c} {ext[c]:.2f}" for c in ALL))


def fig_eval(models, path):
    base, ntc = sr.base_rows(R), sr.ntc_rows(R)
    entries = [("ES-FP + recon", r2.FAMILY_COLOR["base"], base)]
    entries += [(lab, col, rows) for lab, rows, col, _ in models]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    width = 0.84 / len(entries)
    cap = 0.2
    for i, c in enumerate(ALL):
        for j, (_, col, rows) in enumerate(entries):
            e, se = r2.error_se(rows[c])
            x = i - 0.42 + (j + 0.5) * width
            ax.bar(x, min(e, cap), width * 0.9, color=col, yerr=se if e < cap else None,
                   error_kw={"elinewidth": 0.7, "ecolor": MUTED})  # fmt: skip
            if e > cap:
                ax.text(x, cap - 0.004, f"{e:.2f}", rotation=90, fontsize=7, ha="center",
                        va="top", color="white")  # fmt: skip
        ax.plot(i, r2.error_se(ntc[c])[0], "_", color=r2.FAMILY_COLOR["ntc"], ms=30, mew=2.5)
        if c in TRAIN:
            ax.axvspan(i - 0.5, i + 0.5, color="#f3f2ee", zorder=0)
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([f"{c}\n{'train' if c in TRAIN else 'test'}" for c in ALL])
    ax.set_ylabel(r"error: $|\mu/\mu_{\rm ref}-1|$; shocks RMS of $\int$ ratios $-1$",
                  color=INK, fontsize=9)  # fmt: skip
    ax.set_ylim(0, cap)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for lab, col, _ in entries]
    handles.append(plt.Line2D([], [], marker="_", ls="", color=r2.FAMILY_COLOR["ntc"], ms=14,
                              mew=2.5, label="coarse NTC"))  # fmt: skip
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def eval_summary(models):
    base = sr.base_rows(R)
    print("\n== all twelve cases: geometric error / base ==")
    groups = {"train": TRAIN, "test shocks": TEST_K, "S2 (test)": ("S2",),
              "test shocks w/o K5 K7": ("K2", "K3", "K6")}  # fmt: skip
    print(f"{'model':26s}" + "".join(f"{g:>24s}" for g in groups) + "   profile L2 test shocks")
    for lab, rows, _, _ in models:
        cells = []
        for cases in groups.values():
            cells.append(r2.geo([r2.error_se(rows[c])[0] / r2.error_se(base[c])[0]
                                 for c in cases]))  # fmt: skip
        prof = r2.geo([sr.case_error(rows[c], "profile") / sr.case_error(base[c], "profile")
                       for c in TEST_K])  # fmt: skip
        print(f"{lab:26s}" + "".join(f"{x:24.3f}" for x in cells) + f"   {prof:.3f}")


# --------------------------------------------------------------------------- lasso


def lasso_report():
    print("\n== 9-input lasso (rate basis, warm chains) ==")
    names = [f"{o}:{f}" for o in ("rs", "rq") for f in relax.FEATURES[:9]]
    out = {}
    for tag in ("in9", "in9_seed"):
        d = r2.load(f"lasso_{tag}.json")
        out[tag] = d
        for o in d["outer"]:
            kept = [(names[j], o["coef"][j], o["stability"][j]) for j in range(len(names))
                    if abs(o["coef"][j]) > 1e-12]  # fmt: skip
            print(f"  {tag} outer {o['outer']}: bias2 at a0 {o['bias2_at_a0']:.4e} -> fit "
                  f"{o['bias2_fit']:.4e} (predicted); kept {len(kept)}: " +
                  ", ".join(f"{n} {c:+.3f} ({s:.2f})" for n, c, s in kept))  # fmt: skip
        starts = [o["bias2_at_a0"] for o in d["outer"]]
        fits = [o["bias2_fit"] for o in d["outer"]]
        print(f"  {tag}: predicted bias2 after each outer step {fits[:-1]} vs measured at the "
              f"next start {starts[1:]}")  # fmt: skip
    return names, out


def fig_lasso(names, out, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    cols = {"in9": "#1baf7a", "in9_seed": "#b5487a", "in11": "#4a3aa7"}
    for tag, lab in (("in9", "9 inputs"), ("in11", "11 inputs")):
        d = out.get(tag) or r2.load(f"lasso_{tag}.json")
        for o in d["outer"]:
            al = np.array(o["alphas"])
            for ax, k in zip(axes[:2], ("cv_loco", "cv_seed"), strict=True):
                m = np.array(o[k]).mean(0)
                ax.semilogx(al / al[0], m / m[0], color=cols[tag], lw=1.2,
                            label=lab if o["outer"] == 0 else None)  # fmt: skip
    for ax, t in zip(axes[:2], ("leave-one-case-out CV", "split-seed CV"), strict=True):
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.invert_xaxis()
        ax.set_ylim(0.3, 2.0)
        ax.set_xlabel(r"$\alpha/\alpha_{\max}$", color=INK)
        ax.set_ylabel("held-out bias$^2$ / zero model", color=INK, fontsize=8)
        ax.set_title(t + " (warm chains, 3 outer steps)", color=INK, fontsize=9, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[2]
    d = out["in9_seed"]
    width = 0.27
    for j, o in enumerate(d["outer"]):
        c = np.array(o["coef"])
        ax.bar(np.arange(len(names)) + (j - 1) * width, c, width,
               color=["#f2a7c1", "#d06a95", "#7a1f40"][j],
               label=f"outer {o['outer']} (alpha {o['alpha']:.1e})")  # fmt: skip
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.axvline(8.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, fontsize=7)
    ax.set_ylabel("coefficient (standardised features)", color=INK, fontsize=8)
    ax.set_title("split-seed lasso coefficients: they change sign between Gauss-Newton steps",
                 color=INK, fontsize=9, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def small_signal(theta, kind, rows=2):
    """``f64[rows, F]``: the correction's slope in the standardised features at zero (an
    equilibrium, resolved cell) -- a linear model's ``A``, an MLP's local linearisation."""
    t = {k: jnp.asarray(v, jnp.float64) for k, v in theta.items()}
    if kind == "linear":
        return np.asarray(t["A"])[:rows]
    x0 = jnp.zeros((1, t["W0"].shape[0]))
    jac = jax.jacfwd(functools.partial(relax._mlp, t))(x0)  # [1, out, 1, F]
    return np.asarray(jac[0, :rows, 0, :])


def fig_sensitivity(lasso_coef, path):
    """Rate rows: the lasso's coefficients beside every v2 all-case MLP's small-signal
    slopes (18 runs), and the v3 MLPs' (rich rows dlog_rs, dlog_rq: the same rows)."""
    names = relax.FEATURES[:9]
    groups = {}
    for v in ("v2", "v3"):
        plan = r2.load(f"sweep_plan_{v}.json")
        slopes = []
        for r in plan["runs"]:
            if r["validate"]:
                continue
            p = r["depth"], r["width"], r["weight_decay"], r["lr"]
            if r["weight_decay"] == 0.01 and r["lr"] < 0.01:
                continue  # did not learn
            d = r2.load(f"train_mlp_{r['tag']}.json")
            slopes.append(small_signal(d["theta"], "mlp")[:, :9])
            del p
        groups[v] = np.array(slopes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), sharey=True)
    print("\n== small-signal slopes (standardised features) at an equilibrium cell ==")
    for ax, row, lab in zip(axes, (0, 1), (r"$d\log r_s$ (stress rate)",
                                           r"$d\log r_q$ (heat-flux rate)"), strict=True):  # fmt: skip
        for j, v in enumerate(("v2", "v3")):
            s = groups[v][:, row, :]
            for k in range(len(names)):
                ax.plot(np.full(len(s), k + (j - 0.5) * 0.25), s[:, k], "o", ms=4,
                        color=V_COLOR[v], alpha=0.6, mew=0,
                        label=V_LABEL[v] + " MLPs" if k == 0 else None)  # fmt: skip
            med = np.median(s, axis=0)
            agree = (np.sign(s) == np.sign(med)).mean(axis=0)
            print(f"  {v} row {row}: median " + " ".join(f"{n} {m:+.3f}({a:.0%})" for n, m, a
                                                        in zip(names, med, agree, strict=True)))
        ax.bar(np.arange(len(names)), lasso_coef[row], 0.7, color="#b5487a", alpha=0.35,
               label="lasso 9-in (seed CV)", zorder=0)  # fmt: skip
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, fontsize=8)
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
        _style(ax)
    axes[0].set_ylabel("slope per standardised feature", color=INK)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Interpretation: each trained MLP linearised about an equilibrium cell, "
                 "against the lasso", color=INK, fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- circle


def fig_circle(path):
    rows = json.loads((R / "circle_report_v2.json").read_text())
    ops = ("ntc", "esfp_recon", "train_linear_lcfix_q1", "train_mlp_cv_p0_all_s0",
           "train_mlp_rich_2x32")  # fmt: skip
    labels = ("NTC", "ES-FP + recon", "linear q1", "MLP p0 (v2)", "rich MLP 2x32")
    colors = (r2.FAMILY_COLOR["ntc"], r2.FAMILY_COLOR["base"], "#e87ba4", "#1baf7a",
              sr.LEARNED)  # fmt: skip
    panels = (("diffuse", "drag_ratio"), ("diffuse", "heat_ratio"), ("specular", "drag_ratio"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    print("\n== circle (ratio to fine NTC) ==")
    for ax, (wall, key) in zip(axes, panels, strict=True):
        for j, op in enumerate(ops):
            for i, cells in enumerate(("c8", "c16")):
                r = [x for x in rows if x["wall"] == wall and x["cells"] == cells
                     and x["label"] == op]  # fmt: skip
                if not r:
                    continue
                ax.errorbar(i + (j - 2) * 0.15, r[0][key], yerr=r[0][key + "_se"], fmt="o",
                            color=colors[j], ms=6, mec="white", mew=0.6,
                            label=labels[j] if i == 0 else None)  # fmt: skip
                print(f"  {wall:8s} {cells:3s} {key:10s} {labels[j]:14s} {r[0][key]:.3f} +- "
                      f"{r[0][key + '_se']:.3f}")  # fmt: skip
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"8$\times$ coarser", r"16$\times$ coarser"], fontsize=8)
        ax.set_title(f"{wall} wall: {key.split('_')[0]} / fine NTC", color=INK, fontsize=10,
                     loc="left")  # fmt: skip
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    sweeps = {v: points(v) for v in ("v2", "v3")}
    sweep_table(sweeps)
    heldout_case_table(sweeps)
    k8_heldout(sweeps)
    fig_sweep(sweeps, FIG / "round3_sweep.png")
    fig_effects(sweeps, FIG / "round3_effects.png")
    fig_heldout_cases(sweeps, FIG / "round3_heldout_cases.png")
    fig_tradeoff_v3(sweeps, FIG / "round3_tradeoff.png")

    models = [(lab, r2.eval_rows(f), col, lin) for lab, f, col, lin in EVAL_MODELS]
    eval_summary(models)
    fig_eval(models, FIG / "round3_eval.png")
    fig_generalisation(models, FIG / "round3_generalisation.png")
    base, ntc = sr.base_rows(R), sr.ntc_rows(R)
    sr.fig_fields(r2.eval_rows("eval_test_rich_2x32.json"), base, ntc,
                  FIG / "round3_fields_rich2x32.png", "rich MLP 2x32")  # fmt: skip

    names, out = lasso_report()
    fig_lasso(names, out, FIG / "round3_lasso.png")
    coef = np.array(out["in9_seed"]["outer"][-1]["coef"]).reshape(2, 9)
    fig_sensitivity(coef, FIG / "round3_sensitivity.png")
    fig_circle(FIG / "round3_circle.png")
    print("\nfigures:", ", ".join(sorted(p.name for p in FIG.glob("round3_*.png"))))


if __name__ == "__main__":
    main()
