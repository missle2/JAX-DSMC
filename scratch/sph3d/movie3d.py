"""MP4s of the 3-D sphere optimisations (``sph3d``, 2026-10-26), one per run: the body from
the side (lift, +y, up), in planform (from +y) and in isometric view (ray-marched,
scratch/sph3d/shape3d.py; dashed: the initial sphere's outline), the gas density and speed on
the z = 0 side plane (the two central cell planes averaged: ~16 particles a cell, one
snapshot) with the body's section, and the objective's history (J / J0 or L / D, running mean)
with the iterate marked.  Playback as scratch/ldopt/movies.py.

``<name>_hz0`` (2026-10-27): the lsfilter3d run (Helmholtz filter, no reinitialisation), its
sph3d run's history dashed in the history panel.

    PYTHONPATH=. python3 scratch/sph3d/movie3d.py [drag|drag_m10|ld|ld_m10][_hz0 ...] [--frame K]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, "scratch/sph3d")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "examples/mesh")
import shape3d as S  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from shape3d import O  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

ROOT = Path(O.RESULTS)
ADAM = "#2a78d6"
SLOW, FPS, SLOW_REPEAT, FAST_STRIDE, FAST_REPEAT = 50, 30, 3, 2, 2
RUNNING = 15
WINDOW = [(-4.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)]  # the phi grid
VIEW_X = (-3.4, 3.4)
Y_UP = (0.0, 1.0, 0.0)  # lift is force component 1 (+y, as in 2-D): y is up on screen
ISO = (-0.506, 0.469, 0.723)  # the isometric camera: upstream, above (+y), from +z
# name -> (tag, title)
RUNS = {
    "drag": ("sphere_long_kn0.5_S1_ox_adam_drag_free_w300",
             "Minimum drag at fixed volume, Kn 0.5 M 0.3 (Adam)"),
    "drag_m10": ("sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300",
                 "Minimum drag at fixed volume, Kn 0.5 M 10 (Adam)"),
    "ld": ("sphere_long_kn0.5_S1_ox_adam_ldfl_free_w300",
           "Maximum L / D, volume >= 0.5 V0, Kn 0.5 M 0.3 (Adam)"),
    "ld_m10": ("sphere_long_kn0.5_M10_S1_ox_adam_ldfl_free_w300",
               "Maximum L / D, volume >= 0.5 V0, Kn 0.5 M 10 (Adam)"),
}  # fmt: skip
RUNS.update({
    f"{name}_hz0": (tag.replace("_free_w300", "_hz0_free_w300"),
                    title.replace(" (Adam)", " (Adam, Helmholtz, not reinitialised)"))
    for name, (tag, title) in list(RUNS.items())
})  # fmt: skip
OLD = "#8a8a8a"
BODY_CMAP = LinearSegmentedColormap.from_list("body", ["#0d2f5c", ADAM, "#dbe9fa"])


def running(x, w=RUNNING):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


def shown_iterates(cap):
    slow = [(k, SLOW_REPEAT) for k in range(min(SLOW, cap + 1))]
    return slow + [(k, FAST_REPEAT) for k in range(SLOW, cap + 1, FAST_STRIDE)]


class Run:
    def __init__(self, name):
        self.tag, self.title = RUNS[name]
        out = ROOT / self.tag
        hist = json.loads((out / "history.json").read_text())
        self.cfg = dict(O.default_config(), **hist["config"])
        self.rows = hist["rows"]
        self.lift = self.cfg["objective"] == "lift_drag"
        self.cut = 2  # the side plane z = 0 (lift along y)
        grid, ls0 = O.initial_levelset(self.cfg)
        self.lo, self.hi, self.n = np.asarray(grid.lo), np.asarray(grid.hi), tuple(grid.n)
        self.grid = S.PhiGrid(ls0, WINDOW)
        self.out = out
        self.last = len(self.rows)
        while not (out / f"coef_{self.last}.npy").exists():
            self.last -= 1
        self.n_frames = len(self.rows)
        while not (out / f"frames_{self.n_frames - 1}.npz").exists():
            self.n_frames -= 1
        j = np.array([r["objective"] for r in self.rows])
        self.value = running(-j) if self.lift else running(j) / j[0]
        self.raw = -j if self.lift else j / j[0]
        self.volume = np.array([r["area"] for r in self.rows]) / self.rows[0]["area"]
        f0 = np.load(out / "frames_0.npz")
        self.n_inf = float(f0["nrho"][-1].reshape(self.n)[0].mean())
        u0 = f0["u"][-1].reshape(*self.n, 3)[0]
        self.u_inf = float(np.linalg.norm(u0, axis=-1).mean())
        self.phi0 = self.grid.phi(np.load(out / "coef_0.npy"))
        self.reference = None
        if name.endswith("_hz0"):
            rows = json.loads((ROOT / RUNS[name[:-4]][0] / "history.json").read_text())["rows"]
            j = np.array([r["objective"] for r in rows])
            self.reference = running(-j) if self.lift else running(j) / j[0]

    def mid_plane(self, k):
        """Density / n_inf and speed / U_inf on the mid-plane normal to ``self.cut`` (z: the
        side plane, lift along y): the two central cell planes' mean."""
        f = np.load(self.out / f"frames_{min(k, self.n_frames - 1)}.npz")
        a = self.cut
        mid = np.arange(self.n[a] // 2 - 1, self.n[a] // 2 + 1)
        nrho = f["nrho"][-1].reshape(self.n).take(mid, axis=a).mean(a) / self.n_inf
        u = np.linalg.norm(f["u"][-1].reshape(*self.n, 3), axis=-1)
        return nrho, u.take(mid, axis=a).mean(a) / self.u_inf


def _outline(ax, phi, axes, g, **kw):
    """The zero contour of ``phi`` sliced on its mid-plane normal to the dropped axis."""
    drop = ({0, 1, 2} - set(axes)).pop()
    sl = np.take(phi, phi.shape[drop] // 2, axis=drop)
    xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in axes]
    ax.contour(xs[0], xs[1], sl.T, levels=[0.0], **kw)


def _triad(ax):
    """The x, y, z axes' screen directions in the isometric view, bottom left."""
    right, up, _view = S.view_basis(ISO, Y_UP)
    base = np.array([-2.6, -1.25])
    for e, name in zip(np.eye(3), "xyz", strict=True):
        d = 0.45 * np.array([e @ right, e @ up])
        ax.annotate("", base + d, base, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
        ax.text(*(base + 1.25 * d), name, color=MUTED, fontsize=8, ha="center", va="center")


def _shape_panel(ax, img, extent, title):
    ax.imshow(img, extent=extent, cmap=BODY_CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=3)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ax.spines.values():
        side.set_color(GRID)


def frame(run, k, path):
    kk = min(k, run.last)
    g = run.grid
    phi = g.phi(np.load(run.out / f"coef_{kk}.npy"))
    st = S.body_stats(phi, g.lo, g.h)
    half_x = 0.5 * (VIEW_X[1] - VIEW_X[0])
    centre = np.array([0.5 * (VIEW_X[0] + VIEW_X[1]), 0.0, 0.0])
    fig = plt.figure(figsize=(17.0, 7.4))
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.985, top=0.9, bottom=0.08, wspace=0.28,
                          hspace=0.28)  # fmt: skip
    # (title, eye, screen up, the plane's axes, screen-up sign of the second axis)
    views = (
        ("side (lift +y up; from +z)", (0, 0, 1), Y_UP, (0, 1), 1.0),
        ("planform (from above, +y)", (0, 1, 0), (0, 0, -1), (0, 2), -1.0),
    )
    for i, (label, eye, up, axes, sign) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        img = S.render(phi, g.lo, g.h, centre, (half_x, 1.6), eye, up)
        extent = (VIEW_X[0], VIEW_X[1], -1.6 * sign, 1.6 * sign)
        _shape_panel(ax, img, extent, label)
        _outline(ax, run.phi0, axes, g, colors=[MUTED], linewidths=0.9, linestyles="--")
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(-1.6 * sign, 1.6 * sign)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]  (flow ->)", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{'yz'[i]} [m]", fontsize=9, color=MUTED)
    ax = fig.add_subplot(gs[0, 2])
    img = S.render(phi, g.lo, g.h, centre * 0, (3.0, 1.6), ISO, Y_UP)
    _shape_panel(ax, img, (-3.0, 3.0, -1.6, 1.6), "isometric, from upstream (flow ->)")
    _triad(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    nrho, speed = run.mid_plane(k)
    plane, other = "yz"[run.cut - 1], "zy"[run.cut - 1]
    for col, (field, label, vmax, cmap) in enumerate(
        ((nrho, f"density n / n_inf, {plane} = 0", 4.0 if "M10" in run.tag else 2.0, "Greys"),
         (speed, f"speed |u| / U_inf, {plane} = 0", 1.2, "viridis"))
    ):  # fmt: skip
        ax = fig.add_subplot(gs[1, col])
        b = 3 - run.cut  # the in-plane cross-stream axis
        im = ax.imshow(field.T, origin="lower", extent=(run.lo[0], run.hi[0], run.lo[b],
                       run.hi[b]), cmap=cmap, vmin=0.0, vmax=vmax,
                       interpolation="bilinear")  # fmt: skip
        sl = np.take(phi, phi.shape[run.cut] // 2, axis=run.cut)
        xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in (0, b)]
        ax.contourf(xs[0], xs[1], sl.T, levels=[-1e30, 0.0], colors=[ADAM])
        ax.contour(xs[0], xs[1], sl.T, levels=[0.0], colors=[INK], linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, loc="left", color=INK, pad=3)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.set_xlabel("x [m]", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{other} [m]", fontsize=9, color=MUTED)
    ax = fig.add_subplot(gs[1, 2])
    it = np.arange(len(run.value))
    if run.reference is not None:
        ax.plot(run.reference, color=OLD, lw=1.4, ls="--", label="sph3d: band, reinitialised")
        ax.legend(fontsize=9, frameon=False)
    ax.plot(it, run.raw, ".", color=ADAM, ms=2.5, alpha=0.25)
    ax.plot(it, run.value, color=ADAM, lw=1.0, alpha=0.3)
    upto = min(k + 1, len(run.value))
    ax.plot(it[:upto], run.value[:upto], color=ADAM, lw=2.4)
    ax.plot([upto - 1], [run.value[upto - 1]], "o", color=ADAM, ms=8, markeredgecolor="white",
            markeredgewidth=1.5)  # fmt: skip
    ax.axvline(k, color=MUTED, lw=0.8)
    ax.set_xlim(0, len(run.value) if run.reference is None else max(len(run.value),
                                                                     len(run.reference)))
    if run.lift:
        ax.set_ylabel(f"L / D  ({RUNNING}-it. mean; dots: each)", fontsize=10)
    else:
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(f"drag J / J0  ({RUNNING}-it. mean; dots: each)", fontsize=10)
    ax.set_xlabel("iterate", fontsize=10)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    v = run.value[min(kk, len(run.value) - 1)]
    measure = f"L/D {v:.2f}" if run.lift else f"J/J0 {v:.3f}"
    if st.get("pieces", 0):
        size = (f"length {st['hi'][0] - st['lo'][0]:.2f} m, span {st['hi'][1] - st['lo'][1]:.2f}"
                f" m, height {st['hi'][2] - st['lo'][2]:.2f} m, {st['pieces']} piece"
                f"{'s' if st['pieces'] > 1 else ''}")  # fmt: skip
    else:
        size = "no body"
    ended = "  (ended)" if k > run.last else ""
    fig.suptitle(f"{run.title}  --  iterate {k}{ended}   {measure}   volume "
                 f"{run.volume[min(kk, len(run.volume) - 1)]:.2f} V0   ({size})", fontsize=14,
                 color=INK, x=0.04, ha="left")  # fmt: skip
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def movie(name):
    run = Run(name)
    frames = Path(figure_path("movie/sph3d", f"sph3d_{name}"))
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    drawn = frames / "drawn"
    drawn.mkdir()
    plan = shown_iterates(run.last)
    plan[-1] = (plan[-1][0], 2 * FPS)
    i = 0
    for n, (k, repeat) in enumerate(plan):
        frame(run, k, drawn / f"d_{n:04d}.png")
        for _ in range(repeat):
            (frames / f"f_{i:05d}.png").hardlink_to(drawn / f"d_{n:04d}.png")
            i += 1
        if n % 20 == 0:
            print(f"{name}: iterate {k}", flush=True)
    mp4 = figure_path("movie/sph3d", f"sph3d_{name}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{FPS}",
         "-i", str(frames / "f_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True,
    )  # fmt: skip
    shutil.rmtree(frames)
    print("wrote", mp4, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(RUNS))
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    for name in args.names:
        if args.frame is not None:
            frame(Run(name), args.frame, f"scratch/sph3d/frame_{name}.png")
        else:
            movie(name)


if __name__ == "__main__":
    main()
