"""The MLP design sweep's analysis (``lccv``, 2026-11-13): cross-validated error per design
point, the bias / variance trade-off, the factors' main effects, the chosen design, and the
chosen model's shock fields against the fine DSMC.

**Error of a case**: the model's error over the untrained base operator's (ES-FP +
reconstruction, ``eval_base_*.json``) -- < 1 is an improvement, cases of different kinds
become comparable:

* shear wave: ``|mu / mu_ref - 1|``;
* shock: the RMS of ``|int tau_xx / ref - 1|`` and ``|int q_x / ref - 1|`` (``--metric
  integral``) or of the profile errors ``l2_txx``, ``l2_qx`` (``--metric profile``).

Errors are combined over cases by the geometric mean of the ratios.

**Per design point** (sweep.py: 5 leave-one-case-out folds + 2 all-case seeds):

* *held-out* -- each fold's error on its held-out case (generalisation across flows);
* *training* -- the all-case runs' error on the five training cases (the model's systematic
  misfit, the "bias");
* *seed spread* -- the two all-case seeds' difference per case (``|log r_0 - log r_1| /
  sqrt 2``, the training variance);
* *gap* -- held-out over the folds' own training-case error (over-fitting).

**Choice**: the lowest held-out error; among the points within one standard error of it
(the folds' spread), the fewest parameters.  Written to ``sweep_choice_<tag>.json`` (read by
the ``lctrain`` group).

**Figures** (``figures/sweep_<tag>_*.png``): ``tradeoff`` (training, held-out and seed spread
against the parameter count, and held-out against training error), ``effects`` (the L9 main
effects), ``heldout`` (per-case held-out ratios per design point), ``curves`` (the chosen
design's squared bias per case over the iterations), and with ``--eval eval_<tag>.json``
(evaluate.py rows of the trained model, which keep the profiles) ``fields``: density,
temperature, normal stress and heat flux of the shocks -- fine DSMC, coarse NTC, coarse
ES-FP + recon and the learned model -- and ``summary``: every case's ratio, base and model.

    python3 -m examples.learned_collision.sweep_report [--tag v2] [--eval eval_test_mlp.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.learned_collision.plot_round1 import COLOR, GRID, INK, MUTED, _style

RESULTS = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"
BASE_OP = "esfp_recon"
BASE_FILES = ("eval_base_shear.json", "eval_base_shear_hi.json", "eval_base_shockA.json",
              "eval_base_shockB.json")  # fmt: skip
TRAIN = ("S3", "S4", "K1", "K4", "K8")
LEARNED = "#c4314b"
"""The learned model's colour (not used by the round-1 operators shown beside it)."""
LR_MARK = {0.001: "o", 0.003: "s", 0.01: "^"}


def n_params(depth, width, inputs=9, outputs=2):
    """Weights and biases of an MLP with ``depth`` hidden layers of ``width``."""
    return (
        inputs * width + width + (depth - 1) * (width * width + width) + width * outputs + outputs
    )


def case_error(m, metric):
    """A case's scalar error from its forward metrics (module docstring)."""
    if "mu_ratio" in m:
        return abs(m["mu_ratio"] - 1.0)
    if metric == "profile":
        return math.hypot(m["l2_txx"], m["l2_qx"]) / math.sqrt(2.0)
    return math.hypot(m["int_txx_ratio"] - 1.0, m["int_qx_ratio"] - 1.0) / math.sqrt(2.0)


def base_rows(results):
    """``{case: row}`` of the base operator, the latest file winning (shear_hi: 512 seeds)."""
    out = {}
    for name in BASE_FILES:
        path = results / name
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                if r["operator"] == BASE_OP:
                    out[r["case"]] = r
    return out


def ntc_rows(results):
    out = {}
    for name in BASE_FILES:
        path = results / name
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                if r["operator"] == "ntc":
                    out[r["case"]] = r
    return out


def log_ratio(metrics, base, metric):
    """``log(error / base error)`` of one case (floored: an exact model is not infinite)."""
    e, e0 = case_error(metrics, metric), case_error(base, metric)
    return math.log(max(e, 1e-4) / max(e0, 1e-4))


def collect(plan, results, base, metric):
    """Per design point: held-out and training log-ratios, seed spread, gap, per-case
    held-out log-ratios; runs without results are skipped (and counted)."""
    points = []
    for p_i, point in enumerate(plan["design"]):
        params = n_params(point["depth"], point["width"], plan.get("inputs", 9),
                          plan.get("outputs", 2))  # fmt: skip
        rec = {"index": p_i, "point": point, "params": params,
               "held": {}, "fold_train": [], "all": [], "missing": 0, "logs": []}  # fmt: skip
        for r in plan["runs"]:
            if not r["tag"].startswith(f"{plan.get('prefix', 'cv')}_p{p_i}_"):
                continue
            path = results / f"train_mlp_{r['tag']}.json"
            if not path.exists():
                rec["missing"] += 1
                continue
            run = json.loads(path.read_text())
            val = run.get("validation")
            if not val:
                rec["missing"] += 1
                continue
            if r["validate"]:
                held = r["validate"][0]
                rec["held"][held] = log_ratio(val[held], base[held], metric)
                rec["fold_train"].append(np.mean([log_ratio(val[c], base[c], metric)
                                                  for c in r["cases"]]))  # fmt: skip
            else:
                rec["all"].append({c: log_ratio(val[c], base[c], metric) for c in TRAIN})
                rec["logs"].append(run["log"])
        held = np.array(list(rec["held"].values()))
        rec["heldout"] = float(held.mean()) if held.size else math.nan
        rec["heldout_se"] = float(held.std(ddof=1) / math.sqrt(held.size)) if held.size > 1 else 0.0
        rec["train"] = (float(np.mean([np.mean(list(a.values())) for a in rec["all"]]))
                        if rec["all"] else math.nan)  # fmt: skip
        rec["spread"] = (float(np.mean([abs(rec["all"][0][c] - rec["all"][1][c]) for c in TRAIN])
                               / math.sqrt(2.0)) if len(rec["all"]) == 2 else math.nan)  # fmt: skip
        rec["gap"] = (rec["heldout"] - float(np.mean(rec["fold_train"]))
                      if rec["fold_train"] else math.nan)  # fmt: skip
        points.append(rec)
    return points


def choose(points):
    """The lowest held-out error, then the fewest parameters within one standard error."""
    ok = [p for p in points if not math.isnan(p["heldout"])]
    if not ok:
        return None
    best = min(ok, key=_heldout)
    near = [p for p in ok if p["heldout"] <= best["heldout"] + best["heldout_se"]]
    return min(near, key=_size)


def _heldout(p):
    return p["heldout"]


def _size(p):
    return (p["params"], p["heldout"])


def _finite(x):
    return x if np.isfinite(x) else 0.0


def _label(p):
    q = p["point"]
    return f"p{p['index']}: {q['depth']}x{q['width']}, wd {q['weight_decay']:g}, lr {q['lr']:g}"


def fig_tradeoff(points, chosen, path):
    points = [p for p in points if np.isfinite(p["train"]) and np.isfinite(p["heldout"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for p in points:
        x, m = p["params"], LR_MARK[p["point"]["lr"]]
        ax.scatter(x, math.exp(p["train"]), marker=m, s=60, color=COLOR["ntc"], zorder=3,
                   edgecolor="white", linewidth=0.8)  # fmt: skip
        ax.errorbar(x, math.exp(p["heldout"]), yerr=[[math.exp(p["heldout"])
                    - math.exp(p["heldout"] - p["heldout_se"])],
                    [math.exp(p["heldout"] + p["heldout_se"]) - math.exp(p["heldout"])]],
                    fmt=m, ms=8, color=LEARNED, mec="white", capsize=3, zorder=4)  # fmt: skip
        ax.annotate(f"p{p['index']}", (x, math.exp(p["heldout"])), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=MUTED)  # fmt: skip
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(min(p["params"] for p in points), 0.985, "untrained ES-FP + recon", color=MUTED,
            fontsize=8, va="top")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters", color=INK)
    ax.set_ylabel("error / base error (geometric mean)", color=INK)
    ax.set_title("training cases (blue) and held-out case (red, +- s.e. over folds)", loc="left",
                 color=INK, fontsize=10)  # fmt: skip
    handles = [plt.Line2D([], [], marker=m, ls="", color=MUTED, label=f"lr {lr:g}")
               for lr, m in LR_MARK.items()]  # fmt: skip
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    ax = axes[1]
    spread = [p["spread"] for p in points if np.isfinite(p["spread"])]
    vmax = max(spread or [1.0])
    for p in points:
        ax.scatter(math.exp(p["train"]), math.exp(p["heldout"]), s=60 + 600 * _finite(p["spread"]) / vmax,
                   marker=LR_MARK[p["point"]["lr"]], color=LEARNED if p is chosen else COLOR["ntc"],
                   alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)  # fmt: skip
        ax.annotate(f"p{p['index']}", (math.exp(p["train"]), math.exp(p["heldout"])),
                    textcoords="offset points", xytext=(7, 4), fontsize=8, color=MUTED)  # fmt: skip
    lo = min(min(math.exp(p["train"]), math.exp(p["heldout"])) for p in points) * 0.95
    hi = max(max(math.exp(p["train"]), math.exp(p["heldout"])) for p in points) * 1.05
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1.5, zorder=1)
    ax.set_xlabel("training-case error / base  (bias)", color=INK)
    ax.set_ylabel("held-out error / base", color=INK)
    ax.set_title("marker area: seed-to-seed spread (variance); red: chosen", loc="left",
                 color=INK, fontsize=10)  # fmt: skip
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_effects(points, factors, path):
    fig, axes = plt.subplots(1, len(factors), figsize=(13, 3.8), sharey=True)
    for ax, (name, levels) in zip(axes, factors.items(), strict=True):
        means = []
        for lv in levels:
            vals = [p["heldout"] for p in points
                    if p["point"][name] == lv and not math.isnan(p["heldout"])]  # fmt: skip
            means.append(math.exp(np.mean(vals)) if vals else math.nan)
        ax.plot(range(len(levels)), means, "o-", color=LEARNED, lw=2, ms=7, mec="white")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{lv:g}" for lv in levels])
        ax.set_xlabel(name.replace("_", " "), color=INK)
        _style(ax)
    axes[0].set_ylabel("held-out error / base\n(mean over the 3 points per level)", color=INK)
    fig.suptitle("L9 main effects on the held-out error", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_heldout(points, path):
    mat = np.array([[p["held"].get(c, math.nan) for c in TRAIN] for p in points])
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(points) + 1.8))
    lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                dark = abs(mat[i, j]) > 0.6 * lim
                ax.text(j, i, f"{math.exp(mat[i, j]):.2f}", ha="center", va="center",
                        fontsize=8, color="white" if dark else INK)  # fmt: skip
    ax.set_xticks(range(len(TRAIN)))
    ax.set_xticklabels([f"{c} held out" for c in TRAIN])
    ax.set_yticks(range(len(points)))
    ax.set_yticklabels([_label(p) for p in points], fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("log(error / base);  blue = better", color=INK)
    ax.set_title("leave-one-case-out: held-out error / base error", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_curves(chosen, path):
    fig, axes = plt.subplots(1, len(TRAIN), figsize=(14, 3.2))
    for ax, case in zip(axes, TRAIN, strict=True):
        for s, log in enumerate(chosen["logs"]):
            b = [r["bias2"][case] for r in log]
            ax.plot(b, color=(LEARNED, COLOR["ntc"])[s % 2], lw=1.4, label=f"seed {s}")
        ax.set_yscale("log")
        ax.set_title(case, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("iteration", color=INK)
        _style(ax)
    axes[0].set_ylabel("window squared bias", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"chosen design {_label(chosen)}: training on all five cases", x=0.01,
                 ha="left", color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


FIELDS = ((0, r"$n/n_1$"), (1, r"$T/T_{\rm ref}$"), (2, r"$\tau_{xx}$"), (3, r"$q_x$"))


def _profile_x(row):
    prof = np.array(row["profile"])
    return (np.arange(prof.shape[0]) + 0.5) * row["dx_c"] - row["x_shock"], prof


def fig_fields(model_rows, base, ntc, path, label):
    cases = [c for c in ("K3", "K4", "K7", "K8") if c in model_rows]
    cases = cases or sorted(c for c in model_rows if c.startswith("K"))[:4]
    if not cases:
        return False
    fig, axes = plt.subplots(len(FIELDS), len(cases), figsize=(3.6 * len(cases), 9.5),
                             sharex="col", squeeze=False)  # fmt: skip
    for col, case in enumerate(cases):
        row = model_rows[case]
        ref = np.array(row["reference"])
        xr = (np.arange(ref.shape[0]) + 0.5) * row["dx_c"] - row["x_shock"]
        for k, (field, name) in enumerate(FIELDS):
            ax = axes[k, col]
            ax.step(xr, ref[:, field], where="mid", color=INK, lw=2.0, label="fine DSMC (NTC)")
            for src, color, lab in ((ntc.get(case), COLOR["ntc"], "coarse NTC"),
                                    (base.get(case), COLOR["esfp_recon"], "coarse ES-FP + recon"),
                                    (row, LEARNED, f"coarse {label}")):  # fmt: skip
                if src is None or "profile" not in src:
                    continue
                x, prof = _profile_x(src)
                ax.plot(x, prof[:, field], "o-", ms=4, lw=1.3, color=color, label=lab,
                        mec="white", mew=0.6)  # fmt: skip
            ax.set_xlim(-20, 25)
            if col == 0:
                ax.set_ylabel(name, color=INK)
            _style(ax)
        role = "training" if case in TRAIN else "held out"
        axes[0, col].set_title(rf"{case} ({role}), $\Delta x = {row['dx_c']:g}\,\lambda_1$",
                               loc="left", color=INK, fontsize=10)  # fmt: skip
        axes[-1, col].set_xlabel(r"$x - x_{\rm shock}$  [$\lambda_1$]", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_summary(model_rows, base, ntc, path, metric, label):
    cases = sorted(model_rows, key=_case_order)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.38
    for i, case in enumerate(cases):
        e_model = case_error(model_rows[case], metric)
        e_base = case_error(base[case], metric) if case in base else math.nan
        e_ntc = case_error(ntc[case], metric) if case in ntc else math.nan
        ax.bar(i - width / 2, e_base, width, color=COLOR["esfp_recon"])
        ax.bar(i + width / 2, e_model, width, color=LEARNED)
        if np.isfinite(e_ntc):
            ax.plot(i, e_ntc, "_", color=COLOR["ntc"], ms=22, mew=2.5)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"{c}\n({'train' if c in TRAIN else 'test'})" for c in cases])
    ax.set_ylabel("error (|mu/mu_ref - 1|; shocks: integral RMS)", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["esfp_recon"], label="ES-FP + recon"),
               plt.Rectangle((0, 0), 1, 1, color=LEARNED, label=label),
               plt.Line2D([], [], marker="_", ls="", color=COLOR["ntc"], ms=14, mew=2.5,
                          label="coarse NTC")]  # fmt: skip
    ax.legend(handles=handles, frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _case_order(c):
    return (c[0] != "S", int(c[1:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--metric", default="integral", choices=("integral", "profile"))
    ap.add_argument("--eval", default="", help="evaluate.py rows of the trained model")
    ap.add_argument("--eval-label", default="learned MLP")
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args(argv)
    results = Path(args.results)
    FIG.mkdir(exist_ok=True)
    plan = json.loads((results / f"sweep_plan_{args.tag}.json").read_text())
    base = base_rows(results)
    ntc = ntc_rows(results)
    points = collect(plan, results, base, args.metric)
    chosen = choose(points)
    print(f"{'point':44s} {'params':>7s} {'held-out':>9s} {'train':>7s} {'spread':>7s} {'gap':>7s}"
          f"  missing")  # fmt: skip
    for p in points:
        print(
            f"{_label(p):44s} {p['params']:7d} {math.exp(p['heldout']):9.3f} "
            f"{math.exp(p['train']):7.3f} {p['spread']:7.3f} {p['gap']:+7.3f}  {p['missing']}"
        )
    if chosen is None:
        raise SystemExit("no finished runs yet")
    print(f"chosen: {_label(chosen)}  ({chosen['params']} parameters)")
    stem = FIG / f"sweep_{args.tag}"
    fig_tradeoff(points, chosen, f"{stem}_tradeoff.png")
    fig_effects(points, plan.get("factors") or _factors(plan), f"{stem}_effects.png")
    fig_heldout(points, f"{stem}_heldout.png")
    if chosen["logs"]:
        fig_curves(chosen, f"{stem}_curves.png")
    choice = {"tag": args.tag, "metric": args.metric, "point": chosen["point"],
              "index": chosen["index"], "params": chosen["params"], "iters": plan["iters"],
              "train_args": plan.get("train_args", ""),
              "heldout_ratio": math.exp(chosen["heldout"]),
              "points": [{k: v for k, v in p.items() if k != "logs"} for p in points]}  # fmt: skip
    (results / f"sweep_choice_{args.tag}.json").write_text(json.dumps(choice, indent=1))
    if args.eval:
        rows = json.loads((results / args.eval).read_text())["rows"]
        model_rows = {r["case"]: r for r in rows if r["operator"] not in (BASE_OP, "ntc")}
        if fig_fields(model_rows, base, ntc, f"{stem}_fields.png", args.eval_label):
            print(f"fields: {stem}_fields.png")
        fig_summary(model_rows, base, ntc, f"{stem}_summary.png", args.metric, args.eval_label)
    print(f"figures: {stem}_*.png, choice: sweep_choice_{args.tag}.json")


def _factors(plan):
    """The design's factors and their levels, from the design points (sorted)."""
    names = list(plan["design"][0])
    return {n: sorted({p[n] for p in plan["design"]}) for n in names}


if __name__ == "__main__":
    main()
