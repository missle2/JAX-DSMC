"""Figures of the 2-D lsfilter runs (2026-10-27): band Gaussian (bd0) vs Helmholtz (hz0), both
without reinitialisation, beside the old method's runs (band, reinitialised every iterate:
optrc for M 10 drag, optax / optrules for Kn 0.5 drag, ldopt _ldfl_ for L / D).

* ``lsfilter_convergence.png``: per problem (columns), the objective against iterate for Adam
  (top) and MMA (middle) -- drag J / J0 or L / D, 25-iterate running mean -- and the median
  ``|grad phi|`` on the surface (bottom: 1 for a signed distance; without reinitialisation it
  drifts, the Gaussian band's most);
* ``lsfilter_shapes.png``: the last body of each run (filled), the old method's body at the
  same iterate (grey outline), the initial circle dashed.

A run whose objective went non-finite is cut at its last finite iterate.

    PYTHONPATH=. python3 scratch/lsfilter/figures.py
"""

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

ROOT = Path(O.RESULTS)
C, W = "circle_long_kn0.5", "free_g32_w500"
BAND, HELM, OLD = "#2a78d6", "#e34948", "#8a8a8a"
RUNNING = 25
# problem -> (title, regime, L / D?, {rule: old tag})
PROBLEMS = {
    "m10_drag": ("drag at fixed area, Kn 0.5 M 10", "_M10", False,
                 {"ox_adam": f"{C}_M10_S1_ox_adam_rc_tan_sm8_{W}",
                  "mma_s": f"{C}_M10_S1_mma_s_rc_tan_sm8_{W}"}),
    "kn_drag": ("drag at fixed area, Kn 0.5 M 0.3", "", False,
                {"ox_adam": f"{C}_S1_ox_adam_fix_tan_sm8_{W}",
                 "mma_s": f"{C}_S1_mma_s_tan_sm8_{W}"}),
    "kn_ld": ("L / D, area >= 0.5 A0, Kn 0.5 M 0.3", "", True,
              {"ox_adam": f"{C}_S1_ox_adam_ldfl_floor_sm8_{W}",
               "mma_s": f"{C}_S1_mma_s_ldfl_floor_sm8_{W}"}),
    "m10_ld": ("L / D, area >= 0.5 A0, Kn 0.5 M 10", "_M10", True,
               {"ox_adam": f"{C}_M10_S1_ox_adam_ldfl_floor_sm8_{W}",
                "mma_s": f"{C}_M10_S1_mma_s_ldfl_floor_sm8_{W}"}),
}  # fmt: skip
RULES = {"ox_adam": "Adam", "mma_s": "MMA"}


def tag(problem, rule, mark):
    _t, regime, lift, _old = PROBLEMS[problem]
    kind = f"ld{mark}_floor" if lift else f"{mark}_tan"
    return f"{C}{regime}_S1_{rule}_{kind}_sm8_{W}"


def running(x, w=RUNNING):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


class Run:
    """One run's history, cut at its last finite objective."""

    def __init__(self, tag_):
        self.out = ROOT / tag_
        self.ok = (self.out / "history.json").exists()
        if not self.ok:
            return
        hist = json.loads((self.out / "history.json").read_text())
        rows = hist["rows"]
        j = np.array([r["objective"] for r in rows])
        n = int(np.argmin(np.isfinite(j))) if not np.isfinite(j).all() else len(j)
        self.nan_at = None if n == len(j) else n
        self.rows, j = rows[:n], j[:n]
        self.cfg = dict(O.default_config(), **hist["config"])
        self.lift = self.cfg["objective"] == "lift_drag"
        self.value = running(-j) if self.lift else running(j) / j[0]
        self.grad = np.array([r.get("surface_grad_norm", np.nan) for r in self.rows])
        self.last = len(self.rows) if self.nan_at is None else self.nan_at
        while not (self.out / f"coef_{self.last}.npy").exists():
            self.last -= 1

    def body(self, k, ls0, lo, hi):
        k = min(k, self.last)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(self.out / f"coef_{k}.npy")))
        return [seg for seg in O.surface_polyline(ls, lo, hi) if len(seg)]


def _style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label(name, run):
    v = run.value[-1]
    measure = f"L/D {v:.2f}" if run.lift else f"J/J0 {v:.3f}"
    nan = f", nan at {run.nan_at}" if run.nan_at is not None else ""
    return f"{name} ({len(run.value)} it.{nan}): {measure}"


def convergence(runs):
    fig, axes = plt.subplots(3, len(PROBLEMS), figsize=(5.0 * len(PROBLEMS), 11.0))
    for col, (problem, (title, _r, lift, _old)) in enumerate(PROBLEMS.items()):
        for row, rule in enumerate(RULES):
            ax = axes[row, col]
            for name, key, colour, ls in (("band, reinit. (old)", "old", OLD, "--"),
                                          ("band", "bd0", BAND, "-"),
                                          ("Helmholtz", "hz0", HELM, "-")):  # fmt: skip
                run = runs[problem, rule, key]
                if not run.ok:
                    continue
                ax.plot(run.value, color=colour, ls=ls, lw=2.0 if key != "old" else 1.4,
                        label=_label(name, run))  # fmt: skip
                if key != "old":
                    axes[2, col].plot(run.grad, color=colour, lw=1.6,
                                      ls="-" if rule == "ox_adam" else ":",
                                      label=f"{name}, {RULES[rule]}")  # fmt: skip
            if problem == "m10_drag" and rule == "ox_adam":
                gd = runs["m10_drag_gd"]
                ax.plot(gd.value, color=HELM, ls=":", lw=1.6, label=_label("Helmholtz GD", gd))
            if not lift:
                ax.set_yscale("log")
            ax.set_ylabel(f"{'L / D' if lift else 'drag J / J0'}  ({RUNNING}-it. mean)")
            ax.set_title(f"{title}\n{RULES[rule]}", loc="left", fontsize=10, color=INK)
            ax.legend(fontsize=7.5, frameon=False)
            ax.set_xlabel("iterate")
            _style(ax)
        ax = axes[2, col]
        ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
        ax.set_ylabel("median |grad phi| on the surface")
        ax.set_title("level-set slope (1 = signed distance)", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_xlabel("iterate")
        _style(ax)
    fig.suptitle("lsfilter (2-D): band vs Helmholtz filter, both without reinitialisation; "
                 "dashed grey: the old method (band, reinitialised every iterate)",
                 fontsize=12, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc/lsfilter", "lsfilter_convergence.png")
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def shapes(runs, ls0, lo, hi):
    fig, axes = plt.subplots(2, len(PROBLEMS), figsize=(5.0 * len(PROBLEMS), 6.4))
    init = [seg for seg in O.surface_polyline(ls0, lo, hi) if len(seg)]
    for col, (problem, (title, _r, lift, _old)) in enumerate(PROBLEMS.items()):
        for row, rule in enumerate(RULES):
            ax = axes[row, col]
            for seg in init:
                ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=0.9, ls="--")
            shown = [runs[problem, rule, key] for key in ("bd0", "hz0")]
            k_ref = max((r.last for r in shown if r.ok), default=0)
            xs = [lo[0] + 3.0, lo[0] + 13.0]
            text = []
            old = runs[problem, rule, "old"]
            if old.ok:
                for seg in old.body(k_ref, ls0, lo, hi):
                    ax.plot(seg[:, 0], seg[:, 1], color=OLD, lw=1.2)
                    xs += [seg[:, 0].min(), seg[:, 0].max()]
            for name, key, colour in (("band", "bd0", BAND), ("Helmholtz", "hz0", HELM)):
                run = runs[problem, rule, key]
                if not run.ok:
                    text.append(f"{name}: not run yet")
                    continue
                for seg in run.body(run.last, ls0, lo, hi):
                    ax.fill(seg[:, 0], seg[:, 1], color=colour, alpha=0.35, lw=0)
                    ax.plot(seg[:, 0], seg[:, 1], color=colour, lw=1.4)
                    xs += [seg[:, 0].min(), seg[:, 0].max()]
                v = run.value[-1]
                text.append(f"{name} @ {run.last}: " + (f"L/D {v:.2f}" if lift else
                                                        f"J/J0 {v:.3f}"))  # fmt: skip
            ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
            ax.set_ylim(1.5, 8.5)
            ax.set_aspect("equal")
            ax.set_title(f"{title}, {RULES[rule]}\n" + ";  ".join(text), loc="left",
                         fontsize=9, color=INK)  # fmt: skip
            ax.tick_params(labelsize=8, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle(f"lsfilter (2-D) last bodies: band (blue) and Helmholtz (red) without "
                 "reinitialisation; grey: the old method at the same iterate; dashed: start; "
                 "flow ->", fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc/lsfilter", "lsfilter_shapes.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    runs = {}
    for problem, (_t, _r, _l, old) in PROBLEMS.items():
        for rule in RULES:
            runs[problem, rule, "old"] = Run(old[rule])
            for mark in ("bd0", "hz0"):
                runs[problem, rule, mark] = Run(tag(problem, rule, mark))
    runs["m10_drag_gd"] = Run(f"{C}_M10_S1_ox_gd_hz0_tan_sm8_{W}")
    any_run = next(r for r in runs.values() if r.ok)
    grid, ls0 = O.initial_levelset(any_run.cfg)
    convergence(runs)
    shapes(runs, ls0, np.asarray(grid.lo), np.asarray(grid.hi))


if __name__ == "__main__":
    main()
