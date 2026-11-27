"""Square slide versions of two optimisation movies (2026-11-15, a "recent results" deck):

* ``sph3d_drag_m10_hz0_square.mp4``: the M 10 sphere drag minimisation (Helmholtz filter),
  four panels: side view, isometric view, speed on z = 0, drag history;
* ``dlhp_f4_square.mp4``: the constrained 3-D drag minimisation (lift floor, heat cap),
  heat-coloured side / planform / isometric views, temperature on z = 0, the drag history
  and the two constraints (each over its bound).

Drawn with scratch/sph3d/movie3d.py's and scratch/dlh3d/figures.py's renderers and their
playback; output under ``examples/mesh/figures/movie/slides``.

    PYTHONPATH=. python3 scratch/slides/square_movies.py [sph3d dlhp] [--frame K]
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

sys.path.insert(0, "scratch/sph3d")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "examples/mesh")
import movie3d as M  # noqa: E402
import shape3d as S  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

# scratch/dlh3d/figures.py by path (examples/mesh has its own figures module)
_spec = importlib.util.spec_from_file_location("dlh3d_figures", "scratch/dlh3d/figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

SPH = "drag_m10_hz0"
DLHP_TAG, DLHP_NAME = "dlhp_a20_pen_f4", "dlhp_f4"
SIZE, DPI = 10.0, 108  # inches, dots per inch: 1080 x 1080 px
DURATION, HOLD = 15.0, 2.0  # s: the movie, the last iterate's hold
FIELD_Y = 2.0  # half-height of the flow-field window [m]


def _history_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.set_xlabel("iterate", fontsize=10, color=MUTED)


def _field(fig, ax, run_lo, run_hi, field, phi, g, cmap, norm, label, body, xlim=None):
    im = ax.imshow(field.T, origin="lower", extent=(run_lo[0], run_hi[0], run_lo[1], run_hi[1]),
                   cmap=cmap, norm=norm, interpolation="bilinear")  # fmt: skip
    sl = np.take(phi, phi.shape[2] // 2, axis=2)
    xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in (0, 1)]
    ax.contourf(xs[0], xs[1], sl.T, levels=[-1e30, 0.0], colors=[body])
    ax.contour(xs[0], xs[1], sl.T, levels=[0.0], colors=[INK], linewidths=0.8)
    ax.set_xlim(*(xlim or (run_lo[0], run_hi[0])))
    ax.set_ylim(max(run_lo[1], -FIELD_Y), min(run_hi[1], FIELD_Y))
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=12, loc="left", color=INK, pad=3)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.set_xlabel("x [m]  (flow ->)", fontsize=10, color=MUTED)
    ax.set_ylabel("y [m]", fontsize=10, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015).ax.tick_params(labelsize=9)


def sph_frame(run, k, path):
    kk = min(k, run.last)
    g = run.grid
    phi = g.phi(np.load(run.out / f"coef_{kk}.npy"))
    fig = plt.figure(figsize=(SIZE, SIZE))
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.97, top=0.91, bottom=0.06, wspace=0.12,
                          hspace=0.3, height_ratios=(1.0, 1.5, 0.8))  # fmt: skip
    half_x = 0.5 * (M.VIEW_X[1] - M.VIEW_X[0])
    centre = np.array([0.5 * (M.VIEW_X[0] + M.VIEW_X[1]), 0.0, 0.0])
    ax = fig.add_subplot(gs[0, 0])
    img = S.render(phi, g.lo, g.h, centre, (half_x, 1.6), (0, 0, 1), M.Y_UP)
    M._shape_panel(ax, img, (*M.VIEW_X, -1.6, 1.6), "side view")
    M._outline(ax, run.phi0, (0, 1), g, colors=[MUTED], linewidths=0.9, linestyles="--")
    ax.set_xlim(*M.VIEW_X)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (flow ->)", fontsize=10, color=MUTED)
    ax.set_ylabel("y [m]", fontsize=10, color=MUTED)
    ax = fig.add_subplot(gs[0, 1])
    img = S.render(phi, g.lo, g.h, np.zeros(3), (3.0, 1.6), M.ISO, M.Y_UP)
    M._shape_panel(ax, img, (-3.0, 3.0, -1.6, 1.6), "isometric view")
    M._triad(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    _nrho, speed = run.mid_plane(k)
    ax = fig.add_subplot(gs[1, :])
    _field(fig, ax, run.lo, run.hi, speed, phi, g, "viridis", Normalize(0.0, 1.2),
           "speed |u| / U_inf, z = 0 plane", M.ADAM)  # fmt: skip
    ax = fig.add_subplot(gs[2, :])
    it = np.arange(len(run.value))
    ax.plot(it, run.raw, ".", color=M.ADAM, ms=2.5, alpha=0.25)
    ax.plot(it, run.value, color=M.ADAM, lw=1.0, alpha=0.3)
    upto = min(k + 1, len(run.value))
    ax.plot(it[:upto], run.value[:upto], color=M.ADAM, lw=2.4)
    ax.plot([upto - 1], [run.value[upto - 1]], "o", color=M.ADAM, ms=8,
            markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.axvline(min(k, len(run.value) - 1), color=MUTED, lw=0.8)
    ax.set_xlim(0, len(run.value))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"drag D / D0  ({M.RUNNING}-iterate mean; dots: each iterate)", fontsize=12,
                 loc="left", color=INK, pad=3)  # fmt: skip
    _history_axes(ax)
    v = run.value[min(kk, len(run.value) - 1)]
    vol = run.volume[min(kk, len(run.volume) - 1)]
    fig.suptitle(f"3-D minimum drag at fixed volume, M 10, Kn 0.5\niterate {k}:  D / D0 "
                 f"{v:.3f},  V / V0 {vol:.2f}", fontsize=15, color=INK, x=0.07,
                 ha="left", y=0.985)  # fmt: skip
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def dlhp_frame(run, k, path, vmax):
    kk = min(k, run.last)
    g = run.grid
    phi = run.phi(kk)
    hk = min(k, len(run.rows) - 1)
    fig = plt.figure(figsize=(SIZE, SIZE))
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.95, top=0.9, bottom=0.06, wspace=0.32,
                          hspace=0.32, height_ratios=(1.0, 1.0, 0.95))  # fmt: skip
    half_x = 0.5 * (F.VIEW_X[1] - F.VIEW_X[0])
    centre = np.array([0.5 * (F.VIEW_X[0] + F.VIEW_X[1]), 0.0, 0.0])
    views = (("side view (lift up)", (0, 0, 1), F.Y_UP, (0, 1), 1.0, "y"),
             ("planform (from above)", (0, 1, 0), (0, 0, -1), (0, 2), -1.0, "z"))  # fmt: skip
    for i, (label, eye, up, axes, sign, ylab) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        img = F.heat_image(run, hk, phi, centre, (half_x, 1.6), eye, up, vmax)
        F._image_panel(ax, img, (*F.VIEW_X, -1.6 * sign, 1.6 * sign), label)
        ax.title.set_fontsize(12)
        F._outline(ax, run.phi0, axes, g, colors=[MUTED], linewidths=0.9, linestyles="--")
        ax.set_xlim(*F.VIEW_X)
        ax.set_ylim(-1.6 * sign, 1.6 * sign)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]  (flow ->)", fontsize=10, color=MUTED)
        ax.set_ylabel(f"{ylab} [m]", fontsize=10, color=MUTED)
    ax = fig.add_subplot(gs[1, 0])
    img = F.heat_image(run, hk, phi, np.zeros(3), (3.0, 1.6), F.ISO, F.Y_UP, vmax)
    F._image_panel(ax, img, (-3.0, 3.0, -1.6, 1.6), "isometric, surface heat flux")
    ax.title.set_fontsize(12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax), cmap=F.HEAT_CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$\dot q / \dot q_{stag,FM}$", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    nrho, _speed, _temp = run.plane(k)
    ax = fig.add_subplot(gs[1, 1])
    _field(fig, ax, run.lo, run.hi, np.maximum(nrho, 0.1), phi, g, "Greys",
           LogNorm(0.1, run.n_top), "density n / n_inf, z = 0 (log)", F.BLUE,
           xlim=(-4.0, 4.0))  # fmt: skip
    c = run.col
    it = np.arange(len(run.rows))
    m = min(k + 1, len(it))

    def trace(ax, y, colour, label):
        ax.plot(it, y, color=colour, lw=1.0, alpha=0.3)
        ax.plot(it[:m], y[:m], color=colour, lw=2.2, label=label)
        ax.plot([it[m - 1]], [y[m - 1]], "o", color=colour, ms=7, markeredgecolor="white",
                markeredgewidth=1.4)  # fmt: skip

    ax = fig.add_subplot(gs[2, 0])
    trace(ax, c["drag"] / run.d0, F.BLUE, None)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("objective: drag D / D0", fontsize=12, loc="left", color=INK, pad=3)
    ax = fig.add_subplot(gs[2, 1])
    cap = float(run.cfg["heat_cap"])
    trace(ax, c["lift"] / run.d0 / run.lift_min, F.ORANGE, "lift / floor  (keep >= 1)")
    trace(ax, c["heat_measure"] / run.g0 / cap, F.GREEN, "peak heat / cap  (keep <= 1)")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.set_title("constraints (each over its bound)", fontsize=12, loc="left", color=INK, pad=3)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    for ax in fig.axes[-2:]:
        _history_axes(ax)
        ax.set_xlim(0, max(len(it) - 1, 1))
        ax.axvline(m - 1, color=MUTED, lw=0.7)
    r = run.rows[hk]
    fig.suptitle(
        f"3-D minimum drag with lift floor and heat-flux cap, M 10, Kn 0.5\niterate {k}:  "
        f"D/D0 {r['drag'] / run.d0:.3f},  L/D {r['lift'] / r['drag']:.2f},  "
        f"V/V0 {r['area'] / run.v0:.2f}",
        fontsize=15, color=INK, x=0.07, ha="left", y=0.985,
    )  # fmt: skip
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def even_plan(last):
    """Every iterate 0..last, frames spread evenly so the movie runs ``DURATION`` s, the
    last iterate held ``HOLD`` s."""
    budget = round((DURATION - HOLD) * M.FPS)
    edges = np.round(np.linspace(0, budget, last + 1)).astype(int)
    plan = [(k, int(edges[k + 1] - edges[k])) for k in range(last)]
    plan = [(k, r) for k, r in plan if r > 0]  # more iterates than frames: some skipped
    return plan + [(last, round(HOLD * M.FPS))]


def encode(frames, plan, draw, name):
    """Draw each planned iterate once, hard-link it ``repeat`` times, encode at M.FPS."""
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    drawn = frames / "drawn"
    drawn.mkdir()
    i = 0
    for n, (k, repeat) in enumerate(plan):
        draw(k, drawn / f"d_{n:04d}.png")
        for _ in range(repeat):
            (frames / f"f_{i:05d}.png").hardlink_to(drawn / f"d_{n:04d}.png")
            i += 1
        if n % 20 == 0:
            print(f"{name}: iterate {k}", flush=True)
    mp4 = figure_path("movie/slides", f"{name}_square.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{M.FPS}",
         "-i", str(frames / "f_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True,
    )  # fmt: skip
    shutil.rmtree(frames)
    print("wrote", mp4, flush=True)


class SphDraw:
    def __init__(self):
        self.run = M.Run(SPH)

    def __call__(self, k, path):
        sph_frame(self.run, k, path)


class DlhpDraw:
    def __init__(self):
        F.TAG, F.NAME = DLHP_TAG, DLHP_NAME
        self.run = F.Run()
        self.vmax = F.heat_vmax(self.run)

    def __call__(self, k, path):
        dlhp_frame(self.run, k, path, self.vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["sph3d", "dlhp"])
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    for which in args.which:
        draw = SphDraw() if which == "sph3d" else DlhpDraw()
        name = f"sph3d_{SPH}" if which == "sph3d" else DLHP_NAME
        if args.frame is not None:
            draw(args.frame, f"scratch/slides/frame_{name}_{args.frame}.png")
            continue
        last = draw.run.last if which == "sph3d" else len(draw.run.rows) - 1
        plan = even_plan(last)
        encode(Path(figure_path("movie/slides", f"{name}_frames")), plan, draw, name)


if __name__ == "__main__":
    main()
