"""Figures of the optimiser step-rule comparison (``optrules``, 2026-10-19/20 cluster runs).

    PYTHONPATH=. python3 scratch/optrules/analysis.py

Reads ``examples/mesh/results/optimize/circle_long_<regime>_S4_<arm>_tan_sm8_free_g32/`` (and
the earlier Adam run without the guard, ``..._adam_tan_sm8_free``, as the baseline) and writes
three figures under ``examples/mesh/figures/optc/`` plus ``scratch/optrules/analysis.json``:

* ``optrules_convergence.png``: drag J / J0 and the step (largest normal displacement) per
  iterate, each arm, both regimes; every arm costs the same seconds per iterate.
* ``optrules_shapes.png``: every arm's best shape (and its last), against the initial circle.
* ``optrules_mechanisms.png``: the evidence of each failure -- the trust region's rho against
  its noise and the radius floor's repeat loop, L-BFGS's curvature pairs and the step's
  support, MMA's step against its move limit (the regularisation-swamped Kn 0.5 gradient).
"""

import dataclasses
import json
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402
from figure_paths import figure_path  # noqa: E402

ROOT = "examples/mesh/results/optimize"
REGIMES = {"kn0.5": "Kn 0.5, M 0.3 (deck)", "kn0.5_M10": "Kn 0.5, M 10"}
ARMS = ("adam", "tr", "trper", "lbfgs", "mma")
LABELS = {"adam": "Adam", "tr": "trust region (TR)", "trper": "TR + perimeter 0.05",
          "lbfgs": "L-BFGS in TR", "mma": "MMA", "base": "Adam, unguarded backward (earlier)"}  # fmt: skip
COLOURS = {"adam": "#0072B2", "tr": "#D55E00", "trper": "#CC79A7", "lbfgs": "#009E73",
           "mma": "#E69F00", "base": "#8a8a86"}  # fmt: skip
STEP, MOVE_FLOOR = 0.05, 0.05 / 4


def run_dir(regime, arm):
    if arm == "base":
        return f"{ROOT}/circle_long_{regime}_S4_adam_tan_sm8_free"
    return f"{ROOT}/circle_long_{regime}_S4_{arm}_tan_sm8_free_g32"


def load(regime, arm):
    with open(f"{run_dir(regime, arm)}/history.json") as fh:
        hist = json.load(fh)
    rows = hist["rows"]
    j = np.array([r["objective"] for r in rows])
    out = {
        "hist": hist,
        "rows": rows,
        "J": j,
        "se": np.array([r["objective_se"] for r in rows]),
        "step": np.array([r.get("step_max", r["direction_max"] * r["eta"]) for r in rows]),
        "pred": np.array([r["predicted_decrease"] for r in rows]),
        "seconds": np.array([r["seconds"] for r in rows]),
    }
    for key in ("rho", "radius", "accepted", "area_shift"):
        out[key] = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
    # the trust region's repeat loop: a rejected trial identical to the one before it (the
    # same coefficients and seeds; equal to GPU summation-order rounding)
    same = np.r_[False, np.abs(np.diff(j)) < 1e-9 * np.abs(j[1:])]
    out["loop_from"] = int(np.argmax(same)) - 1 if same.any() else None
    return out


def summary(d):
    j, j0 = d["J"], d["J"][0]
    n = len(j)
    stop = d["loop_from"] if d["loop_from"] is not None else n
    at = {k: float(j[k] / j0) for k in (10, 25, 50, 100, 150, 200) if k < stop}
    return {
        "iterates": n,
        "productive_iterates": stop,
        "best": float(j.min() / j0),
        "best_at": int(j.argmin()),
        "last5": float(j[-5:].mean() / j0),
        "at": at,
        "seconds_per_iterate": float(np.median(d["seconds"])),
        "converged_flag": bool(d["hist"]["converged"]),
        "rejected": int(np.nansum(d["accepted"] == 0.0)),
    }


def shapes(regime, arms, data):
    """``{arm: (initial, best, last)}`` zero-set polylines."""
    cfg = dict(O.default_config(), **data[arms[0]]["hist"]["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)

    def line(arm, k):
        c = np.load(f"{run_dir(regime, arm)}/coef_{k}.npy")
        return O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi)

    out = {"_init": O.surface_polyline(ls0, lo, hi), "_box": (lo, hi)}
    for arm in arms:
        d = data[arm]
        stop = d["loop_from"] if d["loop_from"] is not None else len(d["J"])
        out[arm] = (line(arm, int(d["J"].argmin())), line(arm, stop))
    pts = np.concatenate([seg for arm in arms for pair in out[arm] for seg in pair]
                         + list(out["_init"]))  # fmt: skip
    centre = 0.5 * (pts.min(0) + pts.max(0))
    half = 0.55 * float((pts.max(0) - pts.min(0)).max())
    out["_view"] = [(c - half, c + half) for c in centre]
    return out


def fig_convergence(data, path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex="col")
    for col, regime in enumerate(REGIMES):
        ax, bx = axes[0, col], axes[1, col]
        for arm in ("base", *ARMS):
            d = data[regime].get(arm)
            if d is None:
                continue
            it = np.arange(len(d["J"]))
            j0 = d["J"][0]
            kw = {"color": COLOURS[arm], "lw": 1.3 if arm != "base" else 1.0}
            if arm == "base":
                kw["ls"] = "--"
            stop = d["loop_from"]
            end = len(it) if stop is None else stop + 1
            ax.plot(it[:end], d["J"][:end] / j0, label=LABELS[arm], **kw)
            ax.fill_between(it[:end], (d["J"][:end] - d["se"][:end]) / j0,
                            (d["J"][:end] + d["se"][:end]) / j0, color=COLOURS[arm], alpha=0.15,
                            lw=0)  # fmt: skip
            if stop is not None:
                # the repeated (identical) trial: the run's "convergence"
                ax.plot(it[stop:], d["J"][stop:] / j0, color=COLOURS[arm], lw=3.0, alpha=0.35)
                ax.plot([stop], [d["J"][stop] / j0], "x", color=COLOURS[arm], ms=9, mew=2)
            k = int(np.argmin(d["J"]))
            ax.plot([k], [d["J"][k] / j0], "o", color=COLOURS[arm], ms=5, mfc="white")
            if arm != "base":
                bx.semilogy(it, d["step"], color=COLOURS[arm], lw=1.0, label=LABELS[arm])
        ax.set_yscale("log")
        ax.set_ylabel("drag J / J0  (± seed se)")
        ax.set_title(f"{REGIMES[regime]}: drag per iterate (o best; x = TR's repeat loop starts)",
                     fontsize=9)  # fmt: skip
        ax.grid(alpha=0.3, which="both")
        bx.axhline(STEP, color="#52514e", lw=0.8, ls=":")
        bx.axhline(MOVE_FLOOR, color="#52514e", lw=0.8, ls=":")
        bx.text(1, STEP * 1.08, "nominal step 0.05 (Adam, MMA move limit)", fontsize=7)
        bx.text(1, MOVE_FLOOR * 1.08, "TR radius floor 0.0125", fontsize=7)
        bx.set_ylabel("largest normal displacement per iterate")
        bx.set_xlabel("iterate  (every arm ~same seconds per iterate)")
        bx.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_shapes(data, lines, path):
    arms = ("base", *ARMS)
    fig, axes = plt.subplots(2, len(arms), figsize=(2.6 * len(arms), 5.6))
    for row, regime in enumerate(REGIMES):
        lo, hi = lines[regime]["_box"]
        for col, arm in enumerate(arms):
            ax = axes[row, col]
            d = data[regime][arm]
            for seg in lines[regime]["_init"]:
                ax.plot(seg[:, 0], seg[:, 1], color="#8a8a86", lw=0.8, ls="--")
            best, last = lines[regime][arm]
            for seg in last:
                ax.plot(seg[:, 0], seg[:, 1], color=COLOURS[arm], lw=0.8, alpha=0.5)
            for seg in best:
                ax.fill(seg[:, 0], seg[:, 1], color=COLOURS[arm], alpha=0.25, lw=0)
                ax.plot(seg[:, 0], seg[:, 1], color=COLOURS[arm], lw=1.4)
            s = summary(d)
            ax.set_title(f"{LABELS[arm] if arm != 'base' else 'Adam (earlier)'}\n"
                         f"best J/J0 {s['best']:.3f} @ {s['best_at']}", fontsize=8)  # fmt: skip
            ax.set_aspect("equal")
            # the body's neighbourhood (flow from the left)
            ax.set_xlim(*lines[regime]["_view"][0])
            ax.set_ylim(*lines[regime]["_view"][1])
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(REGIMES[regime], fontsize=9)
    fig.suptitle("best shape (filled), last productive shape (thin), initial circle (dashed); "
                 "flow from the left, area held at the circle's", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def lbfgs_pairs(regime, arm):
    z = np.load(f"{run_dir(regime, arm)}/opt_state.npz")
    g = z["base_g"].ravel()
    out = []
    for s, y in zip(z["lbfgs_s"], z["lbfgs_y"], strict=True):
        s, y = s.ravel(), y.ravel()
        out.append({
            "cos_sy": float(s @ y / np.linalg.norm(s) / np.linalg.norm(y)),
            "y_over_g": float(np.linalg.norm(y) / np.linalg.norm(g)),
            # (|s|_2 / |s|_inf)^2: how many coefficients the step really moves
            "support": float((np.linalg.norm(s) / np.abs(s).max()) ** 2),
        })  # fmt: skip
    return out


def fig_mechanisms(data, pairs, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # (a) TR rho against its seed-noise band
    ax = axes[0, 0]
    for regime, mk in zip(REGIMES, ("o", "s"), strict=True):
        d = data[regime]["tr"]
        it = np.arange(len(d["J"]))
        # sd of rho: the CRN difference's se ~ the iterate's seed se (2026-10-19: per-seed
        # correlation of adjacent iterates ~0.5, so sd(J_b - J_a) ~ sd(J))
        noise = d["se"] / np.maximum(np.abs(d["pred"]), 1e-300)
        prev_noise = np.r_[np.nan, noise[:-1]]
        ok = np.isfinite(d["rho"])
        acc = d["accepted"] == 1.0
        ax.scatter(it[ok & acc], np.clip(d["rho"][ok & acc], -3, 4), marker=mk, s=16,
                   color=COLOURS["tr"], alpha=0.8, label=f"{REGIMES[regime]} accepted")  # fmt: skip
        ax.scatter(it[ok & ~acc], np.clip(d["rho"][ok & ~acc], -3, 4), marker=mk, s=22,
                   facecolor="none", edgecolor="k", label=f"{REGIMES[regime]} rejected")  # fmt: skip
        ax.plot(it, np.clip(0.5 + prev_noise, -3, 4), color=COLOURS["tr"], lw=0.8,
                ls="-" if regime == "kn0.5" else "--", alpha=0.7)  # fmt: skip
    ax.axhspan(0.25, 0.75, color="#d9d8d4", alpha=0.5, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-3.2, 4.2)
    ax.set_xlabel("iterate")
    ax.set_ylabel("rho = actual / predicted change")
    ax.set_title("(a) TR: rho (grey: keep radius); lines 0.5 + sd(rho) from seed noise\n"
                 "(solid Kn 0.5 M 0.3, dashed M 10)", fontsize=9)  # fmt: skip
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(alpha=0.3)
    # (b) TR radius
    ax = axes[0, 1]
    for regime, ls in zip(REGIMES, ("-", "--"), strict=True):
        for arm in ("tr", "trper", "lbfgs"):
            d = data[regime][arm]
            ax.semilogy(d["radius"], color=COLOURS[arm], ls=ls, lw=1.2,
                        label=f"{LABELS[arm]}, {REGIMES[regime]}")  # fmt: skip
            if d["loop_from"] is not None:
                ax.plot([d["loop_from"]], [d["radius"][d["loop_from"]]], "x",
                        color=COLOURS[arm], ms=9, mew=2)  # fmt: skip
    ax.axhline(MOVE_FLOOR, color="k", lw=0.6, ls=":")
    ax.set_xlim(0, 150)
    ax.set_xlabel("iterate")
    ax.set_ylabel("trust radius")
    ax.set_title("(b) radius walks down to the floor (x: the identical rejected trial\n"
                 "is retried forever = the 'converged' flag)", fontsize=9)  # fmt: skip
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")
    # (c) signal to noise of the predicted decrease
    ax = axes[0, 2]
    for regime, ls in zip(REGIMES, ("-", "--"), strict=True):
        for arm in ("adam", "tr", "lbfgs", "mma"):
            d = data[regime][arm]
            stop = d["loop_from"] if d["loop_from"] is not None else len(d["J"])
            snr = np.abs(d["pred"][:stop]) / d["se"][:stop]
            ax.semilogy(snr, color=COLOURS[arm], ls=ls, lw=0.9,
                        label=f"{LABELS[arm]}, {REGIMES[regime]}")  # fmt: skip
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.text(2, 0.55, "TR 'inconclusive' cut (|pred| < se/2: accepted, no rho)", fontsize=7)
    ax.set_xlabel("iterate")
    ax.set_ylabel("|predicted change| / seed se of J")
    ax.set_title("(c) the step's predicted change against the objective noise", fontsize=9)
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    # (d) L-BFGS curvature pairs
    ax = axes[1, 0]
    xs, labels = [], []
    for i, (key, rec) in enumerate(pairs.items()):
        for j, p in enumerate(rec):
            x = i + 0.12 * (j - 2)
            ax.plot(x, p["y_over_g"], "o", color=COLOURS["lbfgs"])
            ax.plot(x, max(abs(p["cos_sy"]), 1e-4), "^", color="#52514e")
        xs.append(i)
        labels.append(key)
    ax.set_yscale("log")
    ax.set_xticks(xs, labels, fontsize=7)
    ax.plot([], [], "o", color=COLOURS["lbfgs"], label="|y| / |g|  (gradient change / gradient)")
    ax.plot([], [], "^", color="#52514e", label="|cos(s, y)|  (curvature signal)")
    ax.legend(fontsize=7)
    ax.set_title("(d) L-BFGS's stored pairs (last 5 of each run): y is noise,\n"
                 "s . y ~ 0, so H's scale gamma = s.y / y.Sy collapses", fontsize=9)  # fmt: skip
    ax.grid(alpha=0.3, which="both")
    # (e) step support
    ax = axes[1, 1]
    for i, (key, rec) in enumerate(pairs.items()):
        ax.bar(i, np.median([p["support"] for p in rec]), color=COLOURS[key.split()[0]])
    ax.set_yscale("log")
    ax.set_xticks(range(len(pairs)), list(pairs), fontsize=7)
    ax.set_ylabel("(|s|_2 / |s|_inf)^2  ~ coefficients moved")
    ax.set_title("(e) step support: L-BFGS moves a handful of the 82641 coefficients;\n"
                 "the max-norm radius then scales the whole step down", fontsize=9)  # fmt: skip
    ax.grid(alpha=0.3, which="both", axis="y")
    # (f) MMA step against the move limit
    ax = axes[1, 2]
    for regime, ls in zip(REGIMES, ("-", "--"), strict=True):
        d = data[regime]["mma"]
        ax.semilogy(d["step"], color=COLOURS["mma"], ls=ls, lw=1.2,
                    label=f"MMA step, {REGIMES[regime]}")  # fmt: skip
        ax.semilogy(np.abs(d["area_shift"]), color="#52514e", ls=ls, lw=0.8,
                    label=f"area-restoring shift, {REGIMES[regime]}")  # fmt: skip
    ax.axhline(STEP, color="k", lw=0.6, ls=":")
    ax.set_xlabel("iterate")
    ax.set_ylabel("largest normal displacement")
    ax.set_title("(f) MMA: Kn 0.5 M 0.3 moves ~1/10 of its limit (|g| ~1e-3 << Svanberg's\n"
                 "1e-5/span regulariser); M10 at the limit, no acceptance test", fontsize=9)  # fmt: skip
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    data = {r: {a: load(r, a) for a in ("base", *ARMS)} for r in REGIMES}
    pairs = {f"lbfgs {REGIMES[r]}": lbfgs_pairs(r, "lbfgs") for r in REGIMES}
    pairs[f"tr {REGIMES['kn0.5']}"] = lbfgs_pairs("kn0.5", "tr")
    out = {r: {a: summary(d) for a, d in data[r].items()} for r in REGIMES}
    out["lbfgs_pairs"] = pairs
    for r in REGIMES:
        d = data[r]["tr"]
        ok = np.isfinite(d["rho"])
        rho = d["rho"][ok]
        out[r]["tr"]["rho_fractions"] = {
            "below_0.25": float(np.mean(rho < 0.25)),
            "above_0.75": float(np.mean(rho > 0.75)),
            "median": float(np.median(rho)),
        }
    lines = {r: shapes(r, ("base", *ARMS), data[r]) for r in REGIMES}
    fig_convergence(data, figure_path("optc", "optrules_convergence.png"))
    fig_shapes(data, lines, figure_path("optc", "optrules_shapes.png"))
    fig_mechanisms(data, pairs, figure_path("optc", "optrules_mechanisms.png"))
    with open("scratch/optrules/analysis.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for r in REGIMES:
        for a in ("base", *ARMS):
            s = out[r][a]
            print(f"{r:10s} {a:6s} best {s['best']:.3f}@{s['best_at']:3d}  last5 {s['last5']:.3f}"
                  f"  productive {s['productive_iterates']:3d}/{s['iterates']:3d}  "
                  f"rej {s['rejected']:2d}  at {s['at']}")  # fmt: skip
        print(r, "tr rho", out[r]["tr"]["rho_fractions"])


if __name__ == "__main__":
    main()
