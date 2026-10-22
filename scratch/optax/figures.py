"""Static figures of the optimiser comparison (2026-10-23), from scratch/optax/analysis.json
(analysis.py, drift.py) -- every one-seed arm per iterate (never per seed evaluation):

* ``optax_convergence.png``: drag J / J0 (15-iterate running mean over the raw one-seed
  values), the body's centroid x and its roughness, per iterate, both regimes;
* ``optax_shapes_<regime>.png``: each arm's best (running-mean) shape and its last;
* ``optax_seeds.png``: one seed against four per iterate for Adam / TR / L-BFGS / MMA.

    PYTHONPATH=. python3 scratch/optax/figures.py
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
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from analysis import ROOT, running, tag  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from style import COLOURS, DASHED, GRID, INK, LABELS, MUTED, ORDER, REGIME_TITLES  # noqa: E402

BOX_X = (-10.0, 20.0)
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip


def _style(arm):
    return {"color": COLOURS[arm], "ls": "--" if arm in DASHED else "-", "lw": 2.0}


def _series(d, key):
    pts = sorted((int(k), v) for k, v in d[key].items())
    return np.array([k for k, _ in pts]), np.array([v for _, v in pts])


def convergence(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.2), constrained_layout=True)
    for row, (regime, arms) in enumerate(data.items()):
        aj, ax_c, ax_r = axes[row]
        for arm in ORDER:
            d = arms.get(arm)
            if d is None:
                continue
            j, m = np.asarray(d["J"]), np.asarray(d["J_mean"])
            it = np.arange(len(j))
            aj.plot(it, j, color=COLOURS[arm], lw=0.5, alpha=0.18)
            aj.plot(it, m, label=LABELS[arm], **_style(arm))
            k, pos = _series(d, "position")
            ax_c.plot(k, pos[:, 0], **_style(arm))
            k, r = _series(d, "roughness")
            ax_r.plot(k, r[:, 0], **_style(arm))
        aj.set_yscale("log")
        aj.set_ylabel("drag J / J0  (running mean; raw faint)")
        aj.set_title(f"{REGIME_TITLES[regime]}: drag", loc="left")
        ax_c.axhline(BOX_X[1], color=MUTED, lw=1.0, ls=":")
        ax_c.text(2, BOX_X[1] - 0.4, "outflow face", color=MUTED, va="top", fontsize=9)
        ax_c.axhline(5.0, color=MUTED, lw=0.8)
        ax_c.set_ylabel("body centroid x (m)")
        ax_c.set_title(f"{REGIME_TITLES[regime]}: where the body sits (start x = 5)", loc="left")
        ax_r.set_yscale("log")
        ax_r.set_ylabel("total turning / 2 pi  (1 = convex)")
        ax_r.set_title(f"{REGIME_TITLES[regime]}: roughness", loc="left")
        for ax in axes[row]:
            ax.set_xlabel("iterate")
    axes[0, 0].legend(fontsize=9, ncol=2, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _line(ls0, lo, hi, regime, arm, k):
    c = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{k}.npy")
    return O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi)


def shapes(regime, arms, path):
    cfg = dict(O.default_config(),
               **json.load(open(f"{ROOT}/{tag(regime, 'S4_adam')}/history.json"))["config"])  # fmt: skip
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    init = O.surface_polyline(ls0, lo, hi)
    fig, axes = plt.subplots(3, 3, figsize=(17, 7.6), constrained_layout=True)
    for ax, arm in zip(axes.ravel(), ORDER, strict=True):
        d = arms[arm]
        best, last = d["best_mean_at"], max(int(k) for k in d["position"])
        for seg in init:
            ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=0.8, ls="--")
        for seg in _line(ls0, lo, hi, regime, arm, best):
            ax.fill(seg[:, 0], seg[:, 1], color=COLOURS[arm], alpha=0.3, lw=0)
            ax.plot(seg[:, 0], seg[:, 1], color=COLOURS[arm], lw=1.6)
        for seg in _line(ls0, lo, hi, regime, arm, last):
            ax.plot(seg[:, 0], seg[:, 1], color=INK, lw=1.0)
        ax.set_xlim(*BOX_X)
        ax.set_ylim(1.0, 9.0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([-10, 0, 10, 20])
        ax.set_yticks([])
        ax.set_title(f"{LABELS[arm]}: best {d['best_mean']:.3f} J0 @ {best}  |  "
                     f"last {d['final_mean']:.3f} @ {last}", fontsize=10, loc="left")  # fmt: skip
    fig.suptitle(f"{REGIME_TITLES[regime]}: best shape (filled) and last shape (black), "
                 "initial circle dashed, flow from the left; box x in [-10, 20]",
                 fontsize=11, color=INK)  # fmt: skip
    fig.savefig(path, dpi=110)
    plt.close(fig)


SEED_ARMS = (("ox_adam", "S4_adam", "Adam"), ("tr", "S4_tr2", "Trust region"),
             ("lbfgs", "S4_lbfgs", "L-BFGS"), ("mma", "S4_mma", "MMA"))  # fmt: skip


def seeds(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.2), constrained_layout=True)
    for row, (regime, arms) in enumerate(data.items()):
        aj, acos, apos = axes[row]
        for one, four, label in SEED_ARMS:
            colour = COLOURS[one]
            for arm, ls, n in ((one, "--", 1), (four, "-", 4)):
                d = arms.get(arm)
                if d is None:
                    continue
                m = np.asarray(d["J_mean"])
                aj.plot(np.arange(len(m)), m, color=colour, ls=ls, lw=2.0,
                        label=f"{label}, {n} seed{'s' if n > 1 else ''}")  # fmt: skip
                if one == "ox_adam":
                    c = np.array([np.nan if v is None else v for v in d["step_cos"]])
                    cm = running(np.where(np.isfinite(c), c, np.nanmean(c)))
                    acos.plot(np.arange(len(cm)) + 1, cm, color=colour, ls=ls, lw=2.0,
                              label=f"Adam, {n} seed{'s' if n > 1 else ''}")  # fmt: skip
                    k, pos = _series(d, "position")
                    apos.plot(k, pos[:, 0], color=colour, ls=ls, lw=2.0,
                              label=f"Adam, {n} seed{'s' if n > 1 else ''}")  # fmt: skip
        aj.set_yscale("log")
        aj.set_ylabel("drag J / J0  (running mean)")
        aj.set_title(f"{REGIME_TITLES[regime]}: drag per iterate", loc="left")
        acos.set_ylabel("cosine between successive steps")
        acos.set_title(f"{REGIME_TITLES[regime]}: step consistency (Adam)", loc="left")
        apos.axhline(BOX_X[1], color=MUTED, lw=1.0, ls=":")
        apos.set_ylabel("body centroid x (m)")
        apos.set_title(f"{REGIME_TITLES[regime]}: drift (Adam)", loc="left")
        for ax in axes[row]:
            ax.set_xlabel("iterate")
    axes[0, 0].legend(fontsize=9, ncol=2, frameon=False)
    axes[0, 1].legend(fontsize=9, frameon=False)
    fig.suptitle(
        "one seed against four per iterate (four-seed Adam / L-BFGS / MMA started cold, "
        "the one-seed runs and the four-seed TR warm-started)",
        fontsize=11,
        color=INK,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    data = json.load(open("scratch/optax/analysis.json"))
    convergence(data, figure_path("optc", "optax_convergence.png"))
    for regime, arms in data.items():
        shapes(regime, arms, figure_path("optc", f"optax_shapes_{regime}.png"))
    seeds(data, figure_path("optc", "optax_seeds.png"))
    print("wrote the figures")


if __name__ == "__main__":
    main()
