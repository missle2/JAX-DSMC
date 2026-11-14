"""Round-2 analysis (2026-11-14): lcin (11 inputs: lasso + rate MLPs), lcrich (12 rich
outputs), lceval (the v2 choice and lcfix_q1 linear on all twelve cases + the coarse circle).

    PYTHONPATH=. python scratch/lc_round2.py

Figures: examples/learned_collision/figures/round2_*.png; tables to stdout.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.learned_collision import sweep_report as sr
from examples.learned_collision.plot_round1 import COLOR, INK, MUTED, _style
from jax_dsmc.collide import relax

R = sr.RESULTS
FIG = sr.FIG
TRAIN = sr.TRAIN
ALL = ("S1", "S2", "S3", "S4", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8")
FAMILY_COLOR = {"base": COLOR["esfp_recon"], "zero": "#8a8986", "rates9": "#1baf7a",
                "rates11": "#4a3aa7", "rich": sr.LEARNED, "ntc": COLOR["ntc"],
                "linear": "#e87ba4"}  # fmt: skip
FAMILY_LABEL = {"base": "ES-FP + recon (eval)", "zero": "zero model (re-validated)",
                "rates9": "rate MLP, 9 inputs (lccv v2)", "rates11": "rate MLP, 11 inputs (lcin)",
                "rich": "rich 12-output (lcrich)"}  # fmt: skip
RICH_LS = {"rich linear": "-", "rich 2x32": "--", "rich 3x32 (p7)": "-."}
RUNS = (  # (label, family, train json)
    ("9-in p0 1x8 s0", "rates9", "train_mlp_cv_p0_all_s0.json"),
    ("9-in p0 1x8 s1", "rates9", "train_mlp_cv_p0_all_s1.json"),
    ("9-in p7 3x32 s0", "rates9", "train_mlp_cv_p7_all_s0.json"),
    ("9-in p7 3x32 s1", "rates9", "train_mlp_cv_p7_all_s1.json"),
    ("11-in p0 1x8 s0", "rates11", "train_mlp_in11_p0_s0.json"),
    ("11-in p0 1x8 s1", "rates11", "train_mlp_in11_p0_s1.json"),
    ("11-in p7 3x32 s0", "rates11", "train_mlp_in11_p7_s0.json"),
    ("11-in p7 3x32 s1", "rates11", "train_mlp_in11_p7_s1.json"),
    ("rich linear", "rich", "train_linear_rich_linear.json"),
    ("rich 2x32", "rich", "train_mlp_rich_2x32.json"),
    ("rich 3x32 (p7)", "rich", "train_mlp_rich_p7.json"),
)
METRICS = (("S3", "mu_ratio"), ("S4", "mu_ratio"), ("K1", "int_txx_ratio"),
           ("K1", "int_qx_ratio"), ("K4", "int_txx_ratio"), ("K4", "int_qx_ratio"),
           ("K8", "int_txx_ratio"), ("K8", "int_qx_ratio"))  # fmt: skip
METRIC_LABEL = {"mu_ratio": r"$\mu/\mu_{\rm ref}$", "int_txx_ratio": r"$\int\tau_{xx}$ ratio",
                "int_qx_ratio": r"$\int q_x$ ratio"}  # fmt: skip


def load(name):
    return json.loads((R / name).read_text())


def error_se(m):
    """``(case_error, its standard error)`` (delta method on the integral RMS)."""
    if "mu_ratio" in m:
        return abs(m["mu_ratio"] - 1.0), m["mu_ratio_se"]
    a, b = m["int_txx_ratio"] - 1.0, m["int_qx_ratio"] - 1.0
    e = math.hypot(a, b) / math.sqrt(2.0)
    se = math.hypot(a * m["int_txx_ratio_se"], b * m["int_qx_ratio_se"]) / (2.0 * max(e, 1e-9))
    return e, se


def geo(ratios):
    return float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-4)))))


def last_bias(log, frac=0.2):
    tail = log[-max(1, int(len(log) * frac)):]
    return {c: float(np.mean([x["bias2"][c] for x in tail])) for c in tail[0]["bias2"]}


# --------------------------------------------------------------------------- training cases


def training_table(base, zero, runs):
    print("\n== training cases (validate-train: 128 seeds / 16 starts) ==")
    head = "".join(f"{c} {k[4:7] if k != 'mu_ratio' else 'mu'}".rjust(15) for c, k in METRICS)
    print("model".ljust(20) + head + "   geo/base geo/zero  bias2 end/start")
    rows = [("base (eval)", base, None), ("zero (val)", zero, None)]
    rows += [(lab, d["validation"], d) for lab, _, d in runs]
    for lab, v, d in rows:
        cells = "".join(f"{v[c][k]:8.3f}+-{v[c][k + '_se']:.3f}" for c, k in METRICS)
        g_base = geo([error_se(v[c])[0] / error_se(base[c])[0] for c in TRAIN])
        g_zero = geo([error_se(v[c])[0] / error_se(zero[c])[0] for c in TRAIN])
        b2 = ""
        if d is not None:
            end, start = last_bias(d["log"]), d["log"][0]["bias2"]
            b2 = f"{sum(end.values()) / sum(start.values()):.3f}"
        print(f"{lab:20s}{cells}   {g_base:7.3f} {g_zero:8.3f}  {b2}")


def fig_training(base, zero, runs, path):
    rows = [("ES-FP + recon (eval, 32 starts)", "base", base),
            ("zero model (validate, 16 starts)", "zero", zero)]
    rows += [(lab, fam, d["validation"]) for lab, fam, d in runs]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.05 * len(METRICS), 5.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (case, key) in zip(axes, METRICS, strict=True):
        ax.axvline(1.0, color=INK, lw=1.0)
        ax.axvline(base[case][key], color=FAMILY_COLOR["base"], lw=1.0, ls="--")
        for yi, (_, fam, v) in zip(y, rows, strict=True):
            ax.errorbar(v[case][key], yi, xerr=v[case][key + "_se"], fmt="o", ms=5,
                        color=FAMILY_COLOR[fam], ecolor=FAMILY_COLOR[fam], elinewidth=1.2,
                        mec="white", mew=0.6)  # fmt: skip
        ax.set_title(f"{case}  {METRIC_LABEL[key]}", fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8)
    fig.suptitle("Training cases, forward-validated (1 = fine DSMC; dashed: base operator)",
                 color=INK, fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_curves(runs, path, smooth=10):
    cases = ("total", *TRAIN)
    fig, axes = plt.subplots(1, len(cases), figsize=(3.0 * len(cases), 3.3))
    for lab, fam, d in runs:
        log = d["log"]
        for ax, c in zip(axes, cases, strict=True):
            if c == "total":
                b = np.array([sum(x["bias2"].values()) for x in log])
            else:
                b = np.array([x["bias2"][c] for x in log])
            k = np.ones(smooth) / smooth
            s = np.convolve(b, k, mode="valid") / b[0]
            ls = RICH_LS.get(lab, "-" if lab.endswith("s0") else ":")
            ax.plot(np.arange(len(s)) + smooth - 1, s, ls, color=FAMILY_COLOR[fam], lw=1.3,
                    label=lab)  # fmt: skip
    for ax, c in zip(axes, cases, strict=True):
        ax.set_title(f"{c}: bias$^2$ / start", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("iteration", color=INK, fontsize=8)
        ax.axhline(1.0, color=MUTED, lw=0.8)
        _style(ax)
    axes[0].legend(fontsize=6, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tradeoff_points():
    """Every rate-MLP run's shock integrals (lccv v2: 63 runs; held-out folds flagged)."""
    plan = load("sweep_plan_v2.json")
    pts = []
    for r in plan["runs"]:
        path = R / f"train_mlp_{r['tag']}.json"
        if path.exists():
            v = json.loads(path.read_text()).get("validation") or {}
            pts.append((v, r["validate"][0] if r["validate"] else None))
    return pts


def _line(ax, xs, ys):
    """Least-squares trade-off line of the rate models, drawn across the panel."""
    k, c = np.polyfit(xs, ys, 1)
    x = np.linspace(min(xs) - 0.03, max(xs) + 0.03, 10)
    ax.plot(x, k * x + c, color=FAMILY_COLOR["rates9"], lw=0.8, ls="--")
    return k, c


def fig_tradeoff(base, zero, runs, path):
    pts = tradeoff_points()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    print("\n== shock integral trade-off: rich runs vs the rate runs' least-squares line ==")
    for ax, case in zip(axes, ("K1", "K4", "K8"), strict=True):
        xs, ys = [], []
        for v, held in pts:
            if case not in v:
                continue
            x, y = v[case]["int_txx_ratio"], v[case]["int_qx_ratio"]
            if held == case:
                ax.plot(x, y, "o", mfc="none", mec=FAMILY_COLOR["rates9"], ms=5, mew=1.0)
            else:
                ax.plot(x, y, "o", color=FAMILY_COLOR["rates9"], ms=4, alpha=0.6, mew=0)
                xs.append(x)
                ys.append(y)
        for lab, fam, d in runs:
            if fam == "rates9":
                continue
            v = d["validation"][case]
            ax.errorbar(v["int_txx_ratio"], v["int_qx_ratio"], xerr=v["int_txx_ratio_se"],
                        yerr=v["int_qx_ratio_se"], fmt="D" if fam == "rich" else "s",
                        color=FAMILY_COLOR[fam], ms=7 if fam == "rich" else 5, mec="white",
                        mew=0.6, elinewidth=0.8)  # fmt: skip
            if fam == "rates11":
                xs.append(v["int_txx_ratio"])
                ys.append(v["int_qx_ratio"])
        k, c = _line(ax, xs, ys)
        res = [d["validation"][case]["int_qx_ratio"] - (k * d["validation"][case]
               ["int_txx_ratio"] + c) for lab, fam, d in runs if fam == "rich"]  # fmt: skip
        rates_res = np.array(ys) - (k * np.array(xs) + c)
        print(f"  {case}: rate line slope {k:+.2f}; rich residuals "
              f"{', '.join(f'{x:+.3f}' for x in res)}; rate residual sd {rates_res.std():.3f}")
        for src, fam, mk in ((base, "base", "*"), (zero, "zero", "*")):
            ax.plot(src[case]["int_txx_ratio"], src[case]["int_qx_ratio"], mk, ms=13,
                    color=FAMILY_COLOR[fam], mec="white", mew=0.6)  # fmt: skip
        ax.plot(1.0, 1.0, "+", color=INK, ms=16, mew=2)
        ax.set_xlabel(r"$\int\tau_{xx}$ / fine", color=INK)
        ax.set_ylabel(r"$\int q_x$ / fine", color=INK)
        ax.set_title(f"{case}", color=INK, fontsize=10, loc="left")
        _style(ax)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR["rates9"], label="rate MLP, "
                   "9 inputs (lccv v2, case trained)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=FAMILY_COLOR["rates9"],
                   label="  same, case held out"),
        plt.Line2D([], [], marker="s", ls="", color=FAMILY_COLOR["rates11"],
                   label="rate MLP, 11 inputs (lcin)"),
        plt.Line2D([], [], marker="D", ls="", color=FAMILY_COLOR["rich"],
                   label="rich 12-output (lcrich)"),
        plt.Line2D([], [], marker="*", ls="", ms=11, color=FAMILY_COLOR["base"],
                   label="base (eval)"),
        plt.Line2D([], [], marker="*", ls="", ms=11, color=FAMILY_COLOR["zero"],
                   label="zero model (validate)"),
        plt.Line2D([], [], marker="+", ls="", ms=11, mew=2, color=INK, label="fine DSMC"),
        plt.Line2D([], [], ls="--", color=FAMILY_COLOR["rates9"], label="rate-model trade line"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- lasso


def lasso_summary():
    print("\n== lasso: CV error relative to the zero model (alpha_max) ==")
    out = {}
    for tag, lab in (("in11", "11 inputs, warm chains"), ("signs", "9 inputs, snapshots"),
                     ("free", "9 inputs, snapshots, free signs")):  # fmt: skip
        d = load(f"lasso_{tag}.json")
        out[tag] = (lab, d)
        for o in d["outer"]:
            parts = []
            for k in ("cv_loco", "cv_seed"):
                v = np.array(o[k])
                m = v.mean(0)
                i = int(m.argmin())
                se = v.std(0, ddof=1)[i] / math.sqrt(len(v))
                parts.append(f"{k[3:]} min/zero {m[i] / m[0]:.3f} (+-{se / m[0]:.3f})")
            kept = int((np.abs(o["coef"]) > 1e-12).sum())
            print(f"  {tag:6s} outer {o['outer']}: {'  '.join(parts)}  kept {kept}")
        v = np.array(d["outer"][-1]["cv_loco"])
        i = int(v.mean(0).argmin())
        print("     per held-out case (zero -> best alpha):",
              ", ".join(f"{c} {a:.4f}->{b:.4f}" for c, a, b in
                        zip(TRAIN, v[:, 0], v[:, i], strict=True)))  # fmt: skip
    return out


def fig_lasso(out, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"in11": FAMILY_COLOR["rates11"], "signs": FAMILY_COLOR["rates9"],
              "free": "#9fd8bf"}  # fmt: skip
    for tag, (lab, d) in out.items():
        for o in d["outer"]:
            al = np.array(o["alphas"])
            for ax, k in zip(axes[:2], ("cv_loco", "cv_seed"), strict=True):
                m = np.array(o[k]).mean(0)
                ax.semilogx(al / al[0], m / m[0], color=colors[tag], lw=1.2,
                            label=lab if o["outer"] == 0 else None)  # fmt: skip
    for ax, t in zip(axes[:2], ("leave-one-case-out", "split-seed (in-sample cases)"),
                     strict=True):  # fmt: skip
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$\alpha/\alpha_{\max}$", color=INK)
        ax.set_ylabel("predicted held-out bias$^2$ / zero model", color=INK, fontsize=8)
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.invert_xaxis()
        if t.startswith("leave"):
            ax.set_ylim(0.0, 3.0)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    # per held-out case, last outer iteration, in11 vs signs
    ax = axes[2]
    width = 0.38
    for j, tag in enumerate(("signs", "in11")):
        v = np.array(out[tag][1]["outer"][-1]["cv_loco"])
        i = int(v.mean(0).argmin())
        ax.bar(np.arange(5) + (j - 0.5) * width, v[:, i] / v[:, 0], width, color=colors[tag],
               label=out[tag][0])  # fmt: skip
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(TRAIN)
    ax.set_title("held-out case: best fit / zero", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- rich outputs


def case_medians(feats, q=2):
    return {c: np.array(v["quantiles"][q]) for c, v in feats["cases"].items()}


def rich_outputs(theta, kind, phi):
    t = {k: np.asarray(v) for k, v in theta.items()}
    out = relax.correction(t, np.asarray(phi)[None, :], kind)
    return np.asarray(out)[:, 0] / relax.RICH_BOUND


def fig_rich(feats, models, path):
    med = case_medians(feats)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 5.0), sharey=True)
    im = None
    for ax, (lab, kind, theta) in zip(axes, models, strict=True):
        z = np.stack([rich_outputs(theta, kind, med[c]) for c in ALL], axis=1)
        im = ax.imshow(z, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if abs(z[i, j]) >= 0.05:
                    ax.text(j, i, f"{z[i, j]:+.2f}"[0:5].replace("0.", "."), ha="center",
                            va="center", fontsize=5.5,
                            color="white" if abs(z[i, j]) > 0.6 else INK)  # fmt: skip
        ax.set_xticks(range(len(ALL)))
        ax.set_xticklabels([f"{c}{'*' if c in TRAIN else ''}" for c in ALL], fontsize=7)
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(range(len(relax.RICH_OUTPUTS)))
    axes[0].set_yticklabels(relax.RICH_OUTPUTS, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="output / bound at the case's median cell")
    fig.suptitle("Rich outputs at each case's median feature vector (* = training case)",
                 color=INK, fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def linear_row_rms(theta, feats):
    """Pooled RMS over all cells of each row's raw linear output (uncentred correlation)."""
    a = np.asarray(theta["A"])
    c = np.asarray(feats["correlation"])
    return np.sqrt(np.einsum("rf,fg,rg->r", a, c, a))


# --------------------------------------------------------------------------- lceval


def eval_rows(name):
    return {r["case"]: r for r in load(name)["rows"]}


def eval_table(base, ntc, models):
    print("\n== lceval: all twelve cases (512 seeds / 32 starts), error / base ==")
    print("case role   " + "".join(f"{m:>22s}" for m in ("base", "ntc", *models)))
    for c in ALL:
        role = "train" if c in TRAIN else "test "
        cells = []
        for rows in (base, ntc, *models.values()):
            m = rows[c]
            if "mu_ratio" in m:
                cells.append(f"mu {m['mu_ratio']:.3f}+-{m['mu_ratio_se']:.3f}")
            else:
                cells.append(f"{m['int_txx_ratio']:.3f}/{m['int_qx_ratio']:.3f}")
        print(f"{c:4s} {role}  " + "".join(f"{x:>22s}" for x in cells))
    for lab, rows in models.items():
        tr = geo([error_se(rows[c])[0] / error_se(base[c])[0] for c in TRAIN])
        te = geo([error_se(rows[c])[0] / error_se(base[c])[0] for c in ALL if c not in TRAIN])
        tes = geo([error_se(rows[c])[0] / error_se(base[c])[0] for c in ("S1", "S2")])
        tek = geo([error_se(rows[c])[0] / error_se(base[c])[0] for c in ALL
                   if c not in TRAIN and c[0] == "K"])  # fmt: skip
        prof = geo([sr.case_error(rows[c], "profile") / sr.case_error(base[c], "profile")
                    for c in ALL if c[0] == "K" and c not in TRAIN])  # fmt: skip
        print(f"  {lab:12s} geo error/base: train {tr:.3f}  test {te:.3f} (shear {tes:.3f}, "
              f"shock {tek:.3f}; shock profile L2 {prof:.3f})")


def fig_eval(base, ntc, models, path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    entries = [("ES-FP + recon", FAMILY_COLOR["base"], base)]
    palette = (FAMILY_COLOR["linear"], FAMILY_COLOR["rates9"], "#0f6b4a")
    entries += [(lab, col, rows) for (lab, rows), col in zip(models.items(), palette,
                                                             strict=True)]  # fmt: skip
    width = 0.8 / len(entries)
    for i, c in enumerate(ALL):
        for j, (_, col, rows) in enumerate(entries):
            e, se = error_se(rows[c])
            ax.bar(i - 0.4 + (j + 0.5) * width, e, width * 0.9, color=col, yerr=se,
                   error_kw={"elinewidth": 0.8, "ecolor": MUTED})  # fmt: skip
        e_ntc = error_se(ntc[c])[0]
        ax.plot(i, e_ntc, "_", color=FAMILY_COLOR["ntc"], ms=26, mew=2.5)
        if c in TRAIN:
            ax.axvspan(i - 0.5, i + 0.5, color="#f3f2ee", zorder=0)
    ax.set_xticks(range(len(ALL)))
    ax.set_xticklabels([f"{c}\n{'train' if c in TRAIN else 'test'}" for c in ALL])
    ax.set_ylabel(r"error: $|\mu/\mu_{\rm ref}-1|$; shocks RMS of $\int$ ratios $-1$",
                  color=INK, fontsize=9)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for lab, col, _ in entries]
    handles.append(plt.Line2D([], [], marker="_", ls="", color=FAMILY_COLOR["ntc"], ms=14,
                              mew=2.5, label="coarse NTC"))  # fmt: skip
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=5, loc="upper left")
    ax.set_ylim(0, 0.2)
    e7 = error_se(models["linear q1"]["K7"])[0] if "linear q1" in models else 0.0
    if e7 > 0.2:
        ax.annotate(f"{e7:.2f}", (ALL.index("K7") - 0.4 + 1.5 * width, 0.195), fontsize=8,
                    color=INK, ha="center", va="top", rotation=90)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_circle(path):
    rows = json.loads((R / "circle_report_v2.json").read_text())
    ops = ("ntc", "esbgk_recon", "esfp_recon", "train_linear_lcfix_q1", "train_mlp_cv_p0_all_s0")
    labels = ("NTC", "ES-BGK + recon", "ES-FP + recon", "linear (lcfix_q1)", "MLP p0 (v2)")
    colors = (FAMILY_COLOR["ntc"], COLOR["esbgk_recon"], FAMILY_COLOR["base"],
              FAMILY_COLOR["linear"], FAMILY_COLOR["rates9"])  # fmt: skip
    panels = (("diffuse", "drag_ratio"), ("diffuse", "heat_ratio"), ("specular", "drag_ratio"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    width = 0.16
    for ax, (wall, key) in zip(axes, panels, strict=True):
        for j, op in enumerate(ops):
            for i, cells in enumerate(("c8", "c16")):
                r = [x for x in rows if x["wall"] == wall and x["cells"] == cells
                     and x["label"] == op]  # fmt: skip
                if not r:
                    continue
                ax.errorbar(i + (j - 2) * width, r[0][key], yerr=r[0][key + "_se"], fmt="o",
                            color=colors[j], ms=6, mec="white", mew=0.6,
                            label=labels[j] if i == 0 else None)  # fmt: skip
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"8$\times$ coarser ($\Delta x = 2\lambda_\infty$)",
                            r"16$\times$ coarser"], fontsize=8)  # fmt: skip
        ax.set_title(f"{wall} wall: {key.split('_')[0]} / fine NTC", color=INK, fontsize=10,
                     loc="left")  # fmt: skip
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    base = sr.base_rows(R)
    ntc = sr.ntc_rows(R)
    zero = load("train_linear_lasso_in11.json")["validation"]
    runs = [(lab, fam, load(f)) for lab, fam, f in RUNS]
    training_table(base, zero, runs)
    fig_training(base, zero, runs, FIG / "round2_train_cases.png")
    fig_curves(runs, FIG / "round2_curves.png")
    fig_tradeoff(base, zero, runs, FIG / "round2_tradeoff.png")
    out = lasso_summary()
    fig_lasso(out, FIG / "round2_lasso_cv.png")

    feats = load("features_v4.json")
    rich = [(lab, d["args"]["model"], d["theta"]) for lab, fam, d in runs if fam == "rich"]
    fig_rich(feats, rich, FIG / "round2_rich_outputs.png")
    lin = load("train_linear_rich_linear.json")["theta"]
    rms = linear_row_rms(lin, feats)
    print("\n== rich linear: pooled RMS raw output / bound per row ==")
    for name, r, b in zip(relax.RICH_OUTPUTS, rms, relax.RICH_BOUND, strict=True):
        print(f"  {name:22s} {r:.4f}  ({r / b:.3f} of bound)")

    models = {"linear q1": eval_rows("eval_test_linear.json"),
              "MLP p0 s0": eval_rows("eval_test_mlp.json"),
              "MLP p0 s1": eval_rows("eval_test_mlp_s1.json")}  # fmt: skip
    eval_table(base, ntc, models)
    fig_eval(base, ntc, models, FIG / "round2_eval.png")
    sr.fig_fields(models["MLP p0 s0"], base, ntc, FIG / "round2_eval_fields.png", "MLP p0 (v2)")
    fig_circle(FIG / "round2_circle.png")
    print("\nfigures:", ", ".join(sorted(p.name for p in FIG.glob("round2_*.png"))))


if __name__ == "__main__":
    main()
