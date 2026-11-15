"""Report figures and tables of the class write-up (docs/learning_report/report.tex), from the
results in examples/learned_collision/results (rsync'd from the cluster) and the per-cell
features of scratch/lc_cellfeat.py.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python docs/learning_report/make_figures.py

Writes docs/learning_report/figures/fig_*.{pdf,png} and docs/learning_report/tables/*.tex.
The lcp3 evaluation of the cross-validated MLP (eval_test_v3p3_s0.json) is used when present.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratch"))
import lc_round2 as r2  # noqa: E402
import lc_round3 as r3  # noqa: E402
import lc_unsup as un  # noqa: E402

from examples.learned_collision import sweep_report as sr  # noqa: E402
from jax_dsmc.collide import relax  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
TABLES = HERE / "tables"
R, TRAIN, ALL = r2.R, r2.TRAIN, r2.ALL
TEST = tuple(c for c in ALL if c not in TRAIN)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 6.5  # inches: the text width

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})  # fmt: skip

# (label, source, colour, marker, kind): kind "base" (classical) or "linear" / "mlp" (learned);
# the base operator (the zero correction) is the fourth classical entry
BASELINES = (
    ("NTC", "ntc", "#9a9890", "v", "base"),
    ("NTC adjacent", "ntc_adjacent", "#eda100", "D", "base"),
    ("BGK + recon", "bgk_recon", "#008300", ">", "base"),
    ("ES-BGK + recon", "esbgk_recon", "#4a3aa7", "<", "base"),
    ("ES-FP + recon (base)", "esfp_recon", INK, "o", "base"),
)
LEARNED = (
    ("linear, rate", "eval_test_linear.json", "#e87ba4", "s", "linear"),
    ("lasso, rate (7-term diagnostic)", "eval_test_lasso9.json", "#b5487a", "s", "linear"),
    ("linear, rich", "eval_test_rich_linear.json", "#d6336c", "s", "linear"),
    ("MLP 2x8, rich (CV)", ("eval_test_v3p3_s0.json", "eval_test_v3p3_s1.json"), "#2a78d6", "o",
     "mlp"),
    ("MLP 2x32, rich (pre-sweep)", "eval_test_rich_2x32.json", "#c4314b", "o", "mlp"),
)
BASE_INDEX = 4
WIDTH_COLOR = {8: "#2a78d6", 32: "#eb6834", 128: "#1baf7a"}
FEATURE_TEX = {"dx2": r"$(\Delta x/\lambda)^2$", "dx4": r"$(\Delta x/\lambda)^4$",
               "dt2": r"$(\Delta t/\tau)^2$", "dt1": r"$\Delta t/\tau$", "inv_n": r"$1/N_c$",
               "dx2_inv_n": r"$(\Delta x/\lambda)^2/N_c$", "kn_u2": r"$\mathrm{Kn}_u^2$",
               "kn_t2": r"$\mathrm{Kn}_T^2$",
               "dx2_kn2": r"$(\Delta x/\lambda)^2\mathrm{Kn}^2$",
               "div_u": r"$\lambda\nabla\!\cdot u$", "t_strain_t": r"$g\cdot S g$"}  # fmt: skip
FEATURE_COLOR = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                 "#e34948", INK)  # fmt: skip


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------- the data


def classical_rows():
    ops = {}
    for name in sr.BASE_FILES:
        path = R / name
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                ops.setdefault(r["operator"], {})[r["case"]] = r
    return ops


def all_models():
    """``[(label, rows, colour, marker, kind)]``: the classical operators, then the learned
    models whose evaluation exists."""
    ops = classical_rows()
    out = [(lab, ops[key], col, mk, kind) for lab, key, col, mk, kind in BASELINES]
    for lab, f, col, mk, kind in LEARNED:
        files = f if isinstance(f, tuple) else (f,)
        if all((R / x).exists() for x in files):
            out.append((lab, Seeds([r2.eval_rows(x) for x in files]), col, mk, kind))
    return out


class Seeds:
    """A model's evaluation rows over one or more training seeds: ``seeds[case]`` is the list of
    the seeds' rows (:func:`case_error` combines them)."""

    def __init__(self, runs):
        self.runs = runs

    def __getitem__(self, case):
        return [r[case] for r in self.runs]


def case_error(rows, case):
    """``(error, s.e.)`` of one case: a row dict, or a :class:`Seeds` (geometric mean over the
    seeds; the s.e. from their seed noise)."""
    got = rows[case]
    if isinstance(got, dict):
        return r2.error_se(got)
    es = [r2.error_se(g) for g in got]
    logs = [math.log(max(e, 1e-4)) for e, _ in es]
    e = math.exp(float(np.mean(logs)))
    rel = math.sqrt(sum((se / max(x, 1e-4)) ** 2 for x, se in es)) / len(es)
    return e, e * rel


def log_ratio(rows, case, base):
    """``(log(error / base error), its s.e.)`` on ``case`` (seed noise of both, delta
    method)."""
    e, se = case_error(rows, case)
    b, sb = r2.error_se(base[case])
    e = max(e, 1e-4)
    return math.log(e / b), math.hypot(se / e, sb / b)


def group_ratio(rows, base, cases):
    """Geometric mean of error / base over ``cases`` and its 90 % interval (seed noise)."""
    lr = [log_ratio(rows, c, base) for c in cases]
    m = float(np.mean([x for x, _ in lr]))
    se = math.sqrt(sum(s * s for _, s in lr)) / len(lr)
    return math.exp(m), math.exp(m - 1.645 * se), math.exp(m + 1.645 * se)


# ------------------------------------------------------------------------- figure: lasso


def fig_lasso():
    seed = r2.load("lasso_in9_seed.json")["outer"][0]
    loco = r2.load("lasso_in9.json")["outer"][0]
    al = np.array(seed["alphas"])
    x = al / al[0]
    path = np.array(seed["path"])
    names = relax.FEATURES[:9]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.0))
    for ax, row, title in zip(axes[:2], (0, 1), (r"(a) stress rate $\delta_s$",
                                                  r"(b) heat-flux rate $\delta_q$"), strict=True):  # fmt: skip
        for k, n in enumerate(names):
            ax.plot(x, path[:, row * 9 + k], color=FEATURE_COLOR[k], lw=1.1,
                    label=FEATURE_TEX[n])  # fmt: skip
        ax.axhline(0.0, color=INK, lw=0.6)
        for a, ls in ((loco["alpha"], "-"), (seed["alpha"], "--")):
            ax.axvline(a / al[0], color=MUTED, lw=0.8, ls=ls)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"$\alpha/\alpha_{\max}$")
        style(ax)
    axes[0].set_ylabel("coefficient")
    ax = axes[2]
    for src, key, lab, col in ((loco, "cv_loco", "leave-one-case-out", "#1baf7a"),
                               (seed, "cv_seed", "split-seed", "#b5487a")):  # fmt: skip
        cv = np.array(src[key])
        m = cv.mean(0) / cv.mean(0)[0]
        se = cv.std(0, ddof=1) / math.sqrt(cv.shape[0]) / cv.mean(0)[0]
        ax.plot(x, m, color=col, lw=1.3, label=lab)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.2, lw=0)
        i = int(np.argmin(np.abs(al - src["alpha"])))
        ax.plot([x[i]], [m[i]], "o", color=col, ms=5, mec="white", mew=0.6)
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0.0, 1.6)
    ax.set_title("(c) CV bias$^2$ / zero model", loc="left")
    ax.set_xlabel(r"$\alpha/\alpha_{\max}$")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [matplotlib.lines.Line2D([], [], color=MUTED, lw=0.8, ls="-"),
                matplotlib.lines.Line2D([], [], color=MUTED, lw=0.8, ls="--")]  # fmt: skip
    labels += [r"$\alpha$ chosen by LOCO (0 terms)", r"$\alpha$ chosen by split-seed CV (12 terms)"]
    fig.legend(handles, labels, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), fontsize=6, handlelength=1.6, columnspacing=0.8)  # fmt: skip
    fig.subplots_adjust(wspace=0.32, bottom=0.27)
    save(fig, "fig_lasso")


# --------------------------------------------------------------------------- figure: MLP


def by_depth(p):
    return p["point"]["depth"]


LABEL_OFFSET = {0: (-14, 6), 1: (-16, -10), 2: (6, 5), 3: (-4, -13), 4: (-12, 9), 5: (-6, 6),
                6: (11, 2), 7: (11, 7), 8: (6, -9)}  # fmt: skip
"""Offset (points) of each design point's ID label in Figure 2, placed by hand off the lines."""


def fig_mlp():
    """The rich-output sweep: training error (bias) and held-out-flow error against size, each
    design point labelled by its ID (P1--P9) and its settings listed in a key beside the plot;
    the legend is in columns by width, training above held-out."""
    fig = plt.figure(figsize=(WIDTH, 2.35))
    ax = fig.add_axes([0.08, 0.36, 0.6, 0.58])
    _, pts = r3.points("v3")
    handles = []
    for w, col in WIDTH_COLOR.items():
        ps = sorted([p for p in pts if p["point"]["width"] == w], key=by_depth)
        n = np.array([p["params"] for p in ps])
        tr = np.exp([p["train"] for p in ps])
        he = np.exp([p["heldout"] for p in ps])
        se = np.array([p["heldout_se"] for p in ps])
        (h_tr,) = ax.plot(n, tr, "-", color=col, lw=1.2, marker="o", ms=3.5,
                          label=f"width {w}: training flows (bias)")  # fmt: skip
        h_he = ax.errorbar(n, he, yerr=[he * (1 - np.exp(-se)), he * (np.exp(se) - 1)],
                           fmt="--", color=col, lw=1.0, marker="s", ms=3.5, mfc="white",
                           capsize=1.5, elinewidth=0.6,
                           label=f"width {w}: held-out flow (bias + variance)")  # fmt: skip
        handles += [h_tr, h_he]
        for p, xx, yy in zip(ps, n, he, strict=True):
            ax.annotate(f"P{p['index'] + 1}", (xx, yy), xytext=LABEL_OFFSET[p["index"]],
                        textcoords="offset points", fontsize=6.5, color=col, weight="bold",
                        ha="center", va="center",
                        arrowprops={"arrowstyle": "-", "color": col, "lw": 0.5,
                                    "shrinkA": 0, "shrinkB": 2})  # fmt: skip
    ax.axhline(1.0, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.35, 1.6)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("error / base")
    style(ax)
    key = fig.add_axes([0.71, 0.36, 0.29, 0.58])
    key.axis("off")
    key.text(0.0, 1.0, "     size    wd      lr", fontsize=6, family="monospace", color=MUTED,
             va="top", transform=key.transAxes)  # fmt: skip
    for k, p in enumerate(sorted(pts, key=point_index)):
        q = p["point"]
        frozen = q["weight_decay"] == 0.01 and q["lr"] < 0.01
        note = "  no learning" if frozen else ("  CV choice" if p["index"] == 3 else "")
        key.text(0.0, 0.88 - 0.098 * k,
                 f"P{p['index'] + 1}  {q['depth']}x{q['width']:<3d}  {q['weight_decay']:<6g}  "
                 f"{q['lr']:<5g}{note}", fontsize=6, family="monospace",
                 color=WIDTH_COLOR[q["width"]], weight="bold" if p["index"] == 3 else "normal",
                 va="top", transform=key.transAxes)  # fmt: skip
    fig.legend(handles, [h.get_label() for h in handles], ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=6.3,
               columnspacing=1.2)  # fmt: skip
    save(fig, "fig_mlp")


def point_index(p):
    return p["index"]


# ------------------------------------------------------------------- figure: performance


def fig_performance(models):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw={"width_ratios": [1, 2]},
                             sharey=True)  # fmt: skip
    groups = (("S1", "S2", "S3", "S4"), ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"))
    n = len(models)
    for ax, cases, title in zip(axes, groups, (r"(a) shear waves: $|\mu/\mu_{\rm ref}-1|$",
                                               "(b) shocks: integral stress/heat-flux error"),
                                strict=True):  # fmt: skip
        for j, (lab, rows, col, mk, kind) in enumerate(models):
            xs = np.arange(len(cases)) + (j - (n - 1) / 2) * 0.075
            e = np.array([max(case_error(rows, c)[0], 1e-3) for c in cases])
            s = np.array([case_error(rows, c)[1] for c in cases])
            lo = np.maximum(e - s, 1e-3)
            ax.errorbar(xs, e, yerr=[e - lo, s], fmt=mk, color=col, ms=3.4, lw=0.6,
                        mfc="white" if kind == "base" else col, mew=0.9, capsize=0,
                        label=lab)  # fmt: skip
        for k, c in enumerate(cases):
            if c in TRAIN:
                ax.axvspan(k - 0.45, k + 0.45, color="#f3f2ee", zorder=-2)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([c + ("*" if c in TRAIN else "") for c in cases])
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 0.6)
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].set_ylabel("error against the fine reference")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.1), fontsize=6, columnspacing=0.8)  # fmt: skip
    fig.subplots_adjust(wspace=0.04, bottom=0.25)
    save(fig, "fig_performance")


# ---------------------------------------------------------------------------- figure: PCA


def fig_pca(share, v, proj, lasso):
    names = relax.FEATURES[:9]
    fig = plt.figure(figsize=(WIDTH, 2.0))
    ax = fig.add_axes([0.06, 0.17, 0.19, 0.7])
    k = np.arange(1, len(share) + 1)
    ax.bar(k, share, 0.7, color="#2a78d6")
    ax.plot(k, np.cumsum(share), "o-", color=INK, ms=2.5, lw=1.0)
    ax.axhline(0.9, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("component")
    ax.set_ylabel("variance share")
    ax.set_title(f"(a) spectrum (cond. {share[0] / share[-1]:.0f})", loc="left")
    ax.set_xticks(k)
    style(ax)
    ax = fig.add_axes([0.37, 0.17, 0.19, 0.7])
    ax.imshow(v[:, :4], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(9):
        for j in range(4):
            ax.text(j, i, f"{v[i, j]:+.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v[i, j]) > 0.6 else INK)  # fmt: skip
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"PC{j + 1}\n{share[j]:.0%}" for j in range(4)])
    ax.set_yticks(range(9))
    ax.set_yticklabels([FEATURE_TEX[n] for n in names], fontsize=6)
    ax.set_title("(b) loadings", loc="left")
    ax = fig.add_axes([0.71, 0.17, 0.29, 0.7])
    npc = 6
    for row, dx, lab, col in ((0, -0.18, r"$\delta_s$", "#1baf7a"),
                              (1, 0.18, r"$\delta_q$", "#2a78d6")):  # fmt: skip
        p = proj["v3"][:, row, :npc]
        ax.bar(np.arange(npc) + dx, np.clip(lasso[row, :npc], -0.07, 0.07), 0.34, color=col,
               alpha=0.25, label=f"lasso {lab}")  # fmt: skip
        for kk in range(npc):
            ax.plot(np.full(len(p), kk + dx), p[:, kk], "o", ms=2.0, color=col, alpha=0.7,
                    mew=0, label=f"{len(p)} MLPs {lab}" if kk == 0 else None)  # fmt: skip
            if abs(lasso[row, kk]) > 0.07:
                ax.text(kk + dx, 0.06 * np.sign(lasso[row, kk]), f"{lasso[row, kk]:+.2f}",
                        ha="center", va="center", fontsize=4.8, color=INK, rotation=90)  # fmt: skip
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.axvspan(3.5, npc - 0.5, color=GRID, alpha=0.6, zorder=-2)
    ax.set_ylim(-0.075, 0.075)
    ax.set_xticks(range(npc))
    ax.set_xticklabels([f"PC{kk + 1}" for kk in range(npc)])
    ax.set_ylabel("slope at equilibrium")
    ax.set_title("(c) slopes along the components", loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=5, loc="lower left", handletextpad=0.2,
              columnspacing=0.6)  # fmt: skip
    style(ax)
    save(fig, "fig_pca")


# --------------------------------------------------------------------------- figure: gate


def gate_bars(ax, ood, xs, labels):
    m = np.array([max(ood[c][0], 1e-4) for c in ALL])
    lo = np.array([max(ood[c][1][0], 1e-4) for c in ALL])
    hi = np.array([ood[c][1][1] for c in ALL])
    ax.bar(xs, m, 0.7, color=["#9a9890" if c in TRAIN else "#2a78d6" for c in ALL])
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.7, capsize=1.5)
    ax.axhline(un.TAIL, color=MUTED, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_ylabel("fraction out of range")
    ax.set_title("(a) 5-NN coverage gate", loc="left")
    style(ax)


def gate_regimes(ax, comp, n_regimes, xs, labels):
    bottom = np.zeros(len(ALL))
    for k in range(n_regimes):
        h = np.array([comp[c][k] for c in ALL])
        ax.bar(xs, h, 0.75, bottom=bottom, color=un.CLUSTER_COLORS[k], edgecolor="white",
               linewidth=0.5, label=f"G{k + 1}")  # fmt: skip
        bottom += h
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_ylabel("share of cells")
    ax.set_title("(b) mixture regimes", loc="left")
    ax.legend(frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5),
              handlelength=0.8)  # fmt: skip
    style(ax)


def gate_errors(ax, ood, models):
    base = models[BASE_INDEX][1]
    shocks = [c for c in ALL if c[0] == "K"]
    x = np.array([max(ood[c][0], 1e-4) for c in shocks])
    order = np.argsort(x)
    for lab, rows, col, mk, kind in models[BASE_INDEX + 1:]:
        y = np.array([math.exp(log_ratio(rows, c, base)[0]) for c in shocks])
        ax.plot(x[order], y[order], mk + ("--" if kind == "linear" else "-"), color=col,
                ms=3.5, lw=0.9, label=lab, mec="white", mew=0.4)  # fmt: skip
    for c in shocks:
        ax.text(max(ood[c][0], 1e-4), 1.02, c + ("*" if c in TRAIN else ""), fontsize=4.5,
                color=MUTED, ha="center", transform=ax.get_xaxis_transform())  # fmt: skip
    ax.axhline(1.0, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fraction out of range")
    ax.set_ylabel("error / base")
    ax.set_title("(c) shock error vs coverage", loc="left", pad=9)
    ax.legend(frameon=False, fontsize=4.8, loc="upper left", handlelength=1.6)
    style(ax)


def fig_gate(ood, comp, names_k, models):
    fig = plt.figure(figsize=(WIDTH, 1.85))
    xs = np.arange(len(ALL))
    labels = [c + ("*" if c in TRAIN else "") for c in ALL]
    gate_bars(fig.add_axes([0.06, 0.2, 0.27, 0.68]), ood, xs, labels)
    gate_regimes(fig.add_axes([0.40, 0.2, 0.17, 0.68]), comp, len(names_k), xs, labels)
    gate_errors(fig.add_axes([0.70, 0.2, 0.29, 0.68]), ood, models)
    save(fig, "fig_gate")


# --------------------------------------------------------------------------------- tables


def fmt_ci(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def mark(cells, values, best=True):
    """Wrap the best (lowest) and worst (highest) of ``values`` in the report's \\best /
    \\worst macros."""
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    out = list(cells)
    if best:
        out[lo] = r"\best{" + out[lo] + "}"
    out[hi] = r"\worst{" + out[hi] + "}"
    return out


def table_headline(models, ood):
    """Rows: every operator (the base = 1); columns: the evaluation sets; per column the best
    and worst operator highlighted."""
    base = models[BASE_INDEX][1]
    flagged = tuple(c for c in TEST if ood[c][0] > 0.1)
    in_range = tuple(c for c in TEST if c[0] == "K" and c not in flagged)
    cols = (("training flows", TRAIN), ("held-out shear", ("S2",)),
            ("held-out shocks, in range", in_range),
            ("held-out shocks, out of range", flagged))  # fmt: skip
    vals = [[group_ratio(rows, base, cases) for _, cases in cols] for _, rows, *_ in models]
    cells = [[fmt_ci(v) for v in row] for row in vals]
    for j in range(len(cols)):
        col = mark([c[j] for c in cells], [v[j][0] for v in vals])
        for i in range(len(models)):
            cells[i][j] = col[i]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
             "operator & " + " & ".join(c for c, _ in cols) + r" \\",
             " & " + " & ".join(", ".join(cs) if len(cs) < 4 else f"({len(cs)} flows)"
                                for _, cs in cols) + r" \\", r"\midrule"]  # fmt: skip
    print("\n== headline table: geometric mean error / base [90 % seed interval] ==")
    print(f"  out-of-range (gate-flagged) held-out flows: {flagged}; in range: {in_range}")
    for i, (lab, *_rest) in enumerate(models):
        print(f"  {lab:28s} " + "  ".join(fmt_ci(v) for v in vals[i]))
        lines.append(lab + " & " + " & ".join(cells[i]) + r" \\")
        if i == BASE_INDEX:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "headline.tex").write_text("\n".join(lines) + "\n")


def table_slices(models):
    """The held-out shocks sliced by the two design variables and their intersection; per
    model the worst slice highlighted."""
    base = models[BASE_INDEX][1]
    slices = ((r"Mach 1.5--3", ("K2", "K3")), (r"Mach 6--10", ("K5", "K6", "K7")),
              (r"$\Delta x=4\lambda_1$", ("K2", "K6")), (r"$\Delta x=\lambda_1$", ("K3", "K5", "K7")),
              (r"Mach 6--10 and $\Delta x=\lambda_1$", ("K5", "K7")))  # fmt: skip
    pick = models[BASE_INDEX + 1:]
    vals = [[group_ratio(m[1], base, cases) for m in pick] for _, cases in slices]
    cells = [[fmt_ci(v) for v in row] for row in vals]
    for j in range(len(pick)):
        col = mark([c[j] for c in cells], [v[j][0] for v in vals], best=False)
        for i in range(len(slices)):
            cells[i][j] = col[i]
    lines = [r"\begin{tabular}{@{}llc" + "c" * len(pick) + "@{}}", r"\toprule",
             "slice & flows & runs & " + " & ".join(m[0] for m in pick) + r" \\",
             r"\midrule"]  # fmt: skip
    print("\n== slices of the held-out shocks (error / base [90 %]) ==")
    for (lab, cases), row, v in zip(slices, cells, vals, strict=True):
        print(f"  {lab:40s} " + "  ".join(fmt_ci(x) for x in v))
        lines.append(f"{lab} & {', '.join(cases)} & {32 * len(cases)} & " + " & ".join(row)
                     + r" \\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "slices.tex").write_text("\n".join(lines) + "\n")


def table_circle():
    rows = json.loads((R / "circle_report_v2.json").read_text())
    names = {"ntc": "NTC", "esbgk_recon": "ES-BGK + recon", "esfp_recon": "ES-FP + recon (base)",
             "train_linear_lcfix_q1": "linear, rate",
             "train_mlp_rich_2x32": "MLP 2x32, rich (pre-sweep)",
             "train_mlp_cv_v3_p3_all_s0": "MLP 2x8, rich (CV), seed 0",
             "train_mlp_cv_v3_p3_all_s1": "MLP 2x8, rich (CV), seed 1"}  # fmt: skip
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"operator & drag, $2\lambda$ & drag, $4\lambda$ & heat, $2\lambda$ & "
             r"heat, $4\lambda$ \\", r"\midrule"]  # fmt: skip
    for key, lab in names.items():
        get = {(r["cells"], r["wall"]): r for r in rows if r["label"] == key}
        if ("c8", "diffuse") not in get or ("c16", "diffuse") not in get:
            continue  # not run (lcp3circ pending)
        c8, c16 = get[("c8", "diffuse")], get[("c16", "diffuse")]
        cells = [f"{r[k]:.3f}$\\pm${r[k + '_se']:.3f}" for k in ("drag_ratio", "heat_ratio")
                 for r in (c8, c16)]  # fmt: skip
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "circle.tex").write_text("\n".join(lines) + "\n")


def table_sweep():
    lines = [r"\begin{tabular}{@{}lrcc@{}}", r"\toprule",
             r"design point (depth$\times$width, weight decay, learning rate) & parameters & "
             r"training & held-out \\", r"\midrule"]  # fmt: skip
    for b in r3.points("v3")[1]:
        q = b["point"]
        lines.append(f"{q['depth']}$\\times${q['width']}, {q['weight_decay']:g}, {q['lr']:g} & "
                     f"{b['params']} & {math.exp(b['train']):.2f} & "
                     f"{math.exp(b['heldout']):.2f}$\\pm${b['heldout_se']:.2f} \\\\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "sweep.tex").write_text("\n".join(lines) + "\n")


def main():
    FIGS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    models = all_models()
    print("models:", [m[0] for m in models])
    fig_lasso()
    fig_mlp()
    fig_performance(models)
    z = un.load_cells()
    share, _, v, proj, lasso = un.pca_report(z, rng)
    fig_pca(share, v, proj, lasso)
    model, nn, _, _, white, thresh = un.fit_gmm(z, rng)
    comp, ood, names_k = un.gate_report(z, model, nn, white, thresh, rng)
    fig_gate(ood, comp, names_k, models)
    table_headline(models, ood)
    table_slices(models)
    table_circle()
    table_sweep()


if __name__ == "__main__":
    main()
