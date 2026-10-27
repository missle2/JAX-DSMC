"""Summary figures of the 3-D sphere optimisations (``sph3d``, 2026-10-26), in the manner of
the 2-D optimisation figures (scratch/optax/figures.py):

* ``sph3d_shapes.png``: per run, the best body (filled: the lowest 25-iterate running mean of
  J, the highest of L / D) and the last (black outline) as silhouettes from the side (lift,
  +y, up) and in planform (from +y), the initial sphere dashed, and the best body rendered
  isometrically (scratch/sph3d/shape3d.py);
* ``sph3d_convergence.png``: drag (top) and L / D (bottom) against iterate at both conditions
  -- the objective (running mean, raw faint), the volume / V0, the body's thinnest feature
  (the median half-thickness on its medial surface, in knots: below ~2-3 the per-iterate
  reinitialisation grows lobes, scratch/sph3d/reinit_change*.py) and its piece count.

Geometry every ``--every`` iterates, cached in ``scratch/sph3d/stats_<run>.json``.
``--set lsfilter3d`` draws the Helmholtz-filtered runs without reinitialisation (``_hz0_``,
2026-10-27) instead, the sph3d runs dashed in the convergence panels for reference
(``optc/lsfilter/lsfilter3d_{shapes,convergence}.png``).

    PYTHONPATH=. python3 scratch/sph3d/figures.py [--every 5] [--set sph3d|lsfilter3d]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, "scratch/sph3d")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "examples/mesh")
import shape3d as S  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from shape3d import O  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

ROOT = Path(O.RESULTS)
WINDOW = [(-4.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)]
RUNNING = 25
KN, M10 = "#2a78d6", "#eb6834"  # the palette's first two slots: the two conditions
# name -> (tag, title, colour, problem)
RUNS = {
    "drag": ("sphere_long_kn0.5_S1_ox_adam_drag_free_w300", "drag, Kn 0.5 M 0.3", KN, "drag"),
    "drag_m10": ("sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300", "drag, Kn 0.5 M 10", M10,
                 "drag"),
    "ld": ("sphere_long_kn0.5_S1_ox_adam_ldfl_free_w300", "L / D, Kn 0.5 M 0.3", KN, "ld"),
    "ld_m10": ("sphere_long_kn0.5_M10_S1_ox_adam_ldfl_free_w300", "L / D, Kn 0.5 M 10", M10,
               "ld"),
}  # fmt: skip
# the lsfilter3d runs: the same problems, Helmholtz filter, no reinitialisation
RUNS.update({
    f"{name}_hz0": (tag.replace("_free_w300", "_hz0_free_w300"), title, colour, problem)
    for name, (tag, title, colour, problem) in list(RUNS.items())
})
SETS = {
    "sph3d": {"names": ("drag", "drag_m10", "ld", "ld_m10"), "dir": "optc/sph3d",
              "stem": "sph3d", "what": "sph3d, band filter, reinitialised every iterate"},
    "lsfilter3d": {"names": ("drag_hz0", "drag_m10_hz0", "ld_hz0", "ld_m10_hz0"),
                   "dir": "optc/lsfilter", "stem": "lsfilter3d",
                   "what": "lsfilter3d, Helmholtz filter, no reinitialisation"},
}  # fmt: skip
Y_UP = (0.0, 1.0, 0.0)
ISO = (-0.506, 0.469, 0.723)


def running(x, w=RUNNING):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


def half_thickness(phi, h):
    """The median of ``-phi`` on the body's medial surface (its local maxima inside)."""
    inner = -phi
    ridge = (inner > 0) & (inner == ndimage.maximum_filter(inner, size=3))
    return float(np.median(inner[ridge])) if ridge.any() else float("nan")


class Run:
    def __init__(self, name, every):
        self.tag, self.title, self.colour, self.problem = RUNS[name]
        out = ROOT / self.tag
        hist = json.loads((out / "history.json").read_text())
        self.rows = hist["rows"]
        cfg = dict(O.default_config(), **hist["config"])
        _g, ls0 = O.initial_levelset(cfg)
        self.knot = float(ls0.h)
        self.grid = S.PhiGrid(ls0, WINDOW)
        self.out = out
        self.lift = cfg["objective"] == "lift_drag"
        j = np.array([r["objective"] for r in self.rows])
        self.raw = -j if self.lift else j / j[0]
        self.value = running(self.raw)
        self.best = int(np.argmax(self.value) if self.lift else np.argmin(self.value))
        self.last = len(self.rows)
        while not (out / f"coef_{self.last}.npy").exists():
            self.last -= 1
        self.volume = np.array([r["area"] for r in self.rows]) / self.rows[0]["area"]
        self.stats = self.geometry(name, every)

    def phi(self, k):
        return self.grid.phi(np.load(self.out / f"coef_{k}.npy"))

    def geometry(self, name, every):
        path = Path(f"scratch/sph3d/stats_{name}.json")
        cached = json.loads(path.read_text()) if path.exists() else {}
        for k in range(0, self.last + 1, every):
            if str(k) in cached:
                continue
            phi = self.phi(k)
            cached[str(k)] = {"pieces": int(ndimage.label(phi < 0)[1]),
                              "half": half_thickness(phi, self.grid.h) / self.knot}  # fmt: skip
        path.write_text(json.dumps(cached))
        ks = sorted(int(k) for k in cached)
        return {"k": np.array(ks), "pieces": np.array([cached[str(k)]["pieces"] for k in ks]),
                "half": np.array([cached[str(k)]["half"] for k in ks])}  # fmt: skip


def _silhouette(ax, inside, axis, g, **kw):
    """The projection of ``inside`` along ``axis`` as a filled region or an outline."""
    proj = inside.any(axis=axis).astype(float)
    keep = [a for a in range(3) if a != axis]
    xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in keep]
    if kw.pop("fill", False):
        ax.contourf(xs[0], xs[1], proj.T, levels=[0.5, 1.5], **kw)
    else:
        ax.contour(xs[0], xs[1], proj.T, levels=[0.5], **kw)


def shapes(runs, spec):
    fig, axes = plt.subplots(len(runs), 3, figsize=(16.5, 3.0 * len(runs)),
                             gridspec_kw={"width_ratios": (1.0, 1.0, 1.0)})  # fmt: skip
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for row, run in zip(axes, runs, strict=True):
        g = run.grid
        best, last = run.phi(run.best) < 0, run.phi(run.last) < 0
        v_best = run.value[min(run.best, len(run.value) - 1)]
        v_last = run.value[min(run.last, len(run.value)) - 1]
        measure = "L/D" if run.lift else "J/J0"
        for ax, (axis, label, ylab) in zip(
            row[:2], ((2, "side (lift +y up)", "y [m]"), (1, "planform (from +y)", "z [m]")),
            strict=True,
        ):
            _silhouette(ax, best, axis, g, fill=True, colors=[run.colour], alpha=0.45)
            _silhouette(ax, best, axis, g, colors=[run.colour], linewidths=1.4)
            _silhouette(ax, last, axis, g, colors=[INK], linewidths=1.0)
            ax.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=0.9, ls="--")
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-1.6, 1.6)
            if axis == 1:
                ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]  (flow ->)", fontsize=9, color=MUTED)
            ax.set_ylabel(ylab, fontsize=9, color=MUTED)
            ax.tick_params(labelsize=8, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            head = (f"{run.title}: best {measure} {v_best:.3g} @ {run.best} | last "
                    f"{v_last:.3g} @ {run.last};  " if axis == 2 else "")  # fmt: skip
            ax.set_title(f"{head}{label}", fontsize=10, loc="left", color=INK)
        ax = row[2]
        img = S.render(run.phi(run.best), g.lo, g.h, np.zeros(3), (3.6, 1.6), ISO, Y_UP)
        cmap = LinearSegmentedColormap.from_list("b", ["#0d2f5c", run.colour, "#f4f1ea"])
        ax.imshow(img, extent=(-3.6, 3.6, -1.6, 1.6), cmap=cmap, vmin=0.0, vmax=1.0,
                  interpolation="bilinear")  # fmt: skip
        ax.set_title(f"best body (iterate {run.best}), isometric", fontsize=10,
                     loc="left", color=INK)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(f"3-D sphere optimisations ({spec['what']}): best body filled, last body "
                 "black, initial sphere dashed", fontsize=12, color=INK)  # fmt: skip
    fig.tight_layout()
    path = figure_path(spec["dir"], f"{spec['stem']}_shapes.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def _reference(axes, run):
    """``run`` (an sph3d run beside its lsfilter3d one) dashed, in its condition's colour."""
    it = np.arange(len(run.value))
    kw = {"color": run.colour, "lw": 1.3, "ls": "--", "alpha": 0.6}
    axes[0].plot(it, run.value, label=f"{run.title}, sph3d (band, reinitialised)", **kw)
    axes[1].plot(it, run.volume, **kw)
    st = run.stats
    axes[2].plot(st["k"], st["half"], **kw)
    axes[3].step(st["k"], st["pieces"], where="post", **kw)


def convergence(runs, spec, references=()):
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.0))
    for row, problem in enumerate(("drag", "ld")):
        mine = [r for r in runs if r.problem == problem]
        for ref in (r for r in references if r.problem == problem):
            _reference(axes[row], ref)
        for run in mine:
            it = np.arange(len(run.value))
            ax = axes[row, 0]
            ax.plot(it, run.raw, color=run.colour, lw=0.6, alpha=0.25)
            ax.plot(it, run.value, color=run.colour, lw=2.0, label=run.title)
            ax.plot([run.best], [run.value[run.best]], "o", color=run.colour, ms=7,
                    markeredgecolor="white", markeredgewidth=1.2)  # fmt: skip
            axes[row, 1].plot(it, run.volume, color=run.colour, lw=2.0, label=run.title)
            st = run.stats
            axes[row, 2].plot(st["k"], st["half"], color=run.colour, lw=2.0, label=run.title)
            axes[row, 3].step(st["k"], st["pieces"], where="post", color=run.colour, lw=2.0,
                              label=run.title)  # fmt: skip
        lift = problem == "ld"
        axes[row, 0].set_ylabel(("L / D" if lift else "drag J / J0")
                                + f"  ({RUNNING}-iterate mean; raw faint)")  # fmt: skip
        axes[row, 0].set_title(("maximum L / D, volume >= 0.5 V0" if lift
                                else "minimum drag at fixed volume"), loc="left",
                               fontsize=11, color=INK)  # fmt: skip
        if not lift:
            axes[row, 0].set_ylim(0.0, 1.05)
        axes[row, 1].set_ylabel("volume / V0")
        if not lift:  # held by the tangent projection to rounding
            axes[row, 1].set_ylim(0.9, 1.1)
        axes[row, 1].set_title("volume", loc="left", fontsize=11, color=INK)
        if lift:
            axes[row, 1].axhline(0.5, color=MUTED, lw=0.9, ls="--")
            axes[row, 1].text(2, 0.52, "floor", color=MUTED, fontsize=9)
        axes[row, 2].set_ylabel("median half-thickness [knots]")
        axes[row, 2].set_title("thinnest feature (medial surface)", loc="left", fontsize=11,
                               color=INK)  # fmt: skip
        axes[row, 2].axhspan(0.0, 2.5, color="#e4e3df", alpha=0.6, lw=0)
        axes[row, 2].text(2, 0.3, "reinitialisation unstable below ~2-3 knots", color=MUTED,
                          fontsize=8)  # fmt: skip
        axes[row, 3].set_ylabel("pieces")
        axes[row, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[row, 3].set_title("connected pieces", loc="left", fontsize=11, color=INK)
        for ax in axes[row]:
            ax.set_xlabel("iterate")
            ax.grid(True, color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[row, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"3-D sphere optimisations ({spec['what']}): objective, volume and the body's "
                 "thinnest feature", fontsize=12, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    path = figure_path(spec["dir"], f"{spec['stem']}_convergence.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--set", choices=list(SETS), default="sph3d")
    args = ap.parse_args()
    spec = SETS[args.set]
    runs = [Run(name, args.every) for name in spec["names"]]
    references = ()
    if args.set != "sph3d":
        references = [Run(name, args.every) for name in SETS["sph3d"]["names"]]
    convergence(runs, spec, references)
    shapes(runs, spec)


if __name__ == "__main__":
    main()
