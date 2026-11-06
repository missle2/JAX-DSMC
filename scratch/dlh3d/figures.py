"""Figures and the MP4 of the 3-D constrained drag minimisation (``dlh3d``, 2026-11-02): the M 10,
Kn 0.5 sphere_long body minimising drag subject to lift L >= lift_min D_0, the area-weighted
p-norm heat flux G <= G_0 (heat_cap 1, relative), volume >= 0.8 V_0 and the thickness penalty
(optimize.py's dual ascent on each multiplier).

* ``dlh3d_convergence.png``: drag D / D_0 (dashed: the lsfilter3d M 10 drag-only run at fixed
  volume), lift L / D_0 against its floor, L / D (dashed: the lsfilter3d M 10 max-L/D run),
  heat G / G_0 against its cap, volume against its floor, the three multipliers, the body's
  incidence (principal axis in the x-y plane) and the shape-gradient norm;
* ``dlh3d_shapes.png``: the initial and latest bodies from the side and in planform, and the
  surface heat flux (``max(Q_n, 0) / A_n``, the constraint's node flux in units of the
  free-molecular stagnation value) rendered on both bodies from upstream and isometrically;
* ``dlh3d_heat.png``: the node heat flux against x (lower / upper surface split by y) at the
  first and latest iterate, the p-norm G marked;
* ``movie/dlh3d/dlh3d.mp4``: per iterate the heat-coloured body (side, planform, isometric),
  density / speed / temperature on the z = 0 plane and the five histories.

The node areas (``levelset.surface_on_nodes``, as optimize.heat_terms) are cached in
``scratch/dlh3d/cache/``; the recomputed G is checked against the history's ``heat_measure``.

    PYTHONPATH=. python3 scratch/dlh3d/figures.py [--movie] [--frame K] [--tag T --name N]
"""

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from scipy import ndimage  # noqa: E402

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scratch/sph3d")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "examples/mesh")
import shape3d as S  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from shape3d import O  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

ROOT = Path(O.RESULTS)
TAG = "sphere_long_kn0.5_M10_S1_ox_adam_dlh_hz0_f2_w300"
DRAG_REF = "sphere_long_kn0.5_M10_S1_ox_adam_drag_hz0_free_w300"
LD_REF = "sphere_long_kn0.5_M10_S1_ox_adam_ldfl_hz0_free_w300"
CACHE = Path("scratch/dlh3d/cache")
NAME = "dlh3d"  # output name (--tag / --name: another run of the same problem, 2026-11-05)
WINDOW = [(-4.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)]
VIEW_X = (-3.4, 3.4)
Y_UP = (0.0, 1.0, 0.0)
ISO = (-0.506, 0.469, 0.723)
FRONT = (-1.0, 0.0, 0.0)  # from upstream
# the reference palette's categorical order: drag, lift, heat, volume / thickness
BLUE, ORANGE, GREEN, PINK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6"
OLD = "#8a8a8a"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat", ["#fbf1e6", "#f6b98a", ORANGE, "#a63b10", "#4a1604"]
)
BODY_CMAP = LinearSegmentedColormap.from_list("body", ["#0d2f5c", BLUE, "#dbe9fa"])
FPS, REPEAT, HOLD = 30, 12, 60


def _spines(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)


def incidence(phi, lo, h):
    """The angle [deg] of the body's principal axis in the x-y plane above +x (nose-up
    positive for a body whose nose is upstream, -x): the voxel second moments."""
    idx = np.argwhere(phi < 0)
    if len(idx) < 3:
        return float("nan")
    x = lo + h * idx
    d = x - x.mean(0)
    c = d[:, :2].T @ d[:, :2]
    w, v = np.linalg.eigh(c)
    ax = v[:, np.argmax(w)]
    ax = ax if ax[0] > 0 else -ax
    return float(np.degrees(np.arctan2(ax[1], ax[0])))


class Run:
    def __init__(self):
        out = ROOT / TAG
        hist = json.loads((out / "history.json").read_text())
        self.hist = hist
        self.cfg = dict(O.default_config(), **hist["config"])
        self.rows = rows = hist["rows"]
        grid, self.ls0 = O.initial_levelset(self.cfg)
        self.lo, self.hi, self.n = np.asarray(grid.lo), np.asarray(grid.hi), tuple(grid.n)
        self.grid = S.PhiGrid(self.ls0, WINDOW)
        self.out = out
        self.last = len(rows)
        while not (out / f"coef_{self.last}.npy").exists():
            self.last -= 1
        self.n_frames = len(rows)
        while not (out / f"frames_{self.n_frames - 1}.npz").exists():
            self.n_frames -= 1
        col = {key: np.array([r[key] for r in rows], float)
               for key in ("drag", "lift", "heat_measure", "area", "lift_multiplier",
                           "heat_multiplier", "thickness_multiplier", "grad_norm",
                           "thinness", "penalised_objective")}  # fmt: skip
        self.col = col
        self.d0 = float(hist["lift_ref"])
        self.g0 = float(hist["heat_g0"])
        self.lift_min = float(self.cfg["lift_min"])
        self.area_floor = float(self.cfg["area_floor"])
        self.v0 = float(rows[0]["area"])
        f0 = np.load(out / "frames_0.npz")
        self.n_inf = float(f0["nrho"][-1].reshape(self.n)[0].mean())
        self.u_inf = float(np.linalg.norm(f0["u"][-1].reshape(*self.n, 3)[0], axis=-1).mean())
        self.t_inf = float(f0["temp"][-1].reshape(self.n)[0].mean())
        self.phi0 = self.phi(0)
        # the density colour scale's top: the peak n / n_inf over a spread of iterates
        sample = sorted({0, self.n_frames // 3, 2 * self.n_frames // 3, self.n_frames - 1})
        self.n_top = float(max(np.nanmax(self.plane(k)[0]) for k in sample))
        self.heat_ref = float(O.heat_reference(self.cfg))
        self.lattice = O.EM.heat_lattice(self.cfg, self.lo, self.hi)
        # no reference runs on the panels (2026-11-05: the lsfilter3d drag-only and max-L/D
        # runs had other wall conditions -- not comparable)
        self.refs = {}

    def phi(self, k):
        return self.grid.phi(np.load(self.out / f"coef_{min(k, self.last)}.npy"))

    def ls(self, k):
        coef = jnp.asarray(np.load(self.out / f"coef_{min(k, self.last)}.npy"))
        return dataclasses.replace(self.ls0, coef=coef)

    def node_heat(self, k):
        """``(Q, A)`` on the heat lattice at iterate ``k``: the window's node heat rates
        (``heat_<k>.npy``) and the body's node areas (cached)."""
        k = min(k, len(self.rows) - 1)
        q = np.load(self.out / f"heat_{k}.npy")
        cache = CACHE if NAME == "dlh3d" else CACHE / NAME
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"area_{k}.npy"
        if not path.exists():
            area = O.L.surface_on_nodes(self.ls(k), tuple(self.lo.tolist()),
                                        tuple(self.hi.tolist()), 2, self.lattice)  # fmt: skip
            np.save(path, np.asarray(area))
        return q, np.load(path)

    def heat_measure(self, k):
        """G recomputed from the cached areas (optimize._heat_measure in numpy)."""
        q, area = self.node_heat(k)
        keep = area > O.HEAT_AREA_MIN * area.max()
        flux = np.maximum(q[keep], 0.0) / area[keep] / self.heat_ref
        p = float(self.cfg["heat_p"])
        return float((np.sum(area[keep] * flux**p) / area[keep].sum()) ** (1.0 / p))

    def node_flux(self, k):
        """``(x, flux)`` of the nodes with surface: positions and ``max(Q, 0) / A / ref``."""
        q, area = self.node_heat(k)
        keep = area > O.HEAT_AREA_MIN * area.max()
        lo, h, n = self.lattice
        idx = np.stack(np.unravel_index(np.flatnonzero(keep), n), -1)
        x = np.asarray(lo) + h * idx
        return x, np.maximum(q[keep], 0.0) / area[keep] / self.heat_ref

    def flux_at(self, k, pts):
        """The surface heat flux at ``pts`` (``[..., 3]``): ``sum w Q / sum w A`` over each
        point's lattice cell (the multilinear weights the tallies use)."""
        q, area = self.node_heat(k)
        lo, h, n = self.lattice
        c = ((pts.reshape(-1, 3) - np.asarray(lo)) / h).T
        qq = ndimage.map_coordinates(np.maximum(q, 0.0).reshape(n), c, order=1, mode="nearest")
        aa = ndimage.map_coordinates(area.reshape(n), c, order=1, mode="nearest")
        return (qq / np.maximum(aa, 1e-12) / self.heat_ref).reshape(pts.shape[:-1])

    def plane(self, k):
        """Density / n_inf, speed / U_inf and temperature / T_inf on z = 0 (the two central
        cell planes' mean, the last snapshot of iterate ``k``)."""
        f = np.load(self.out / f"frames_{min(k, self.n_frames - 1)}.npz")
        mid = np.arange(self.n[2] // 2 - 1, self.n[2] // 2 + 1)
        nrho = f["nrho"][-1].reshape(self.n).take(mid, axis=2).mean(2) / self.n_inf
        u = np.linalg.norm(f["u"][-1].reshape(*self.n, 3), axis=-1).take(mid, axis=2).mean(2)
        temp = f["temp"][-1].reshape(self.n).take(mid, axis=2).mean(2) / self.t_inf
        return nrho, u / self.u_inf, temp


def heat_image(run, k, phi, centre, half, eye, up, vmax):
    """RGB image of the body from ``eye`` coloured by the surface heat flux, shaded."""
    shade, surf, hit = S.render_surface(phi, run.grid.lo, run.grid.h, centre, half, eye, up)
    flux = run.flux_at(k, surf)
    rgb = HEAT_CMAP(Normalize(0.0, vmax)(flux))[..., :3]
    rgb = rgb * (0.45 + 0.55 * np.nan_to_num(shade))[..., None]
    return np.where(hit[..., None], rgb, 1.0)


def _image_panel(ax, img, extent, title):
    ax.imshow(img, extent=extent, interpolation="bilinear")
    ax.set_title(title, fontsize=10, loc="left", color=INK, pad=3)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ax.spines.values():
        side.set_color(GRID)


def _outline(ax, phi, axes, g, **kw):
    drop = ({0, 1, 2} - set(axes)).pop()
    sl = np.take(phi, phi.shape[drop] // 2, axis=drop)
    xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in axes]
    ax.contour(xs[0], xs[1], sl.T, levels=[0.0], **kw)


def _silhouette(ax, inside, axis, g, **kw):
    proj = inside.any(axis=axis).astype(float)
    keep = [a for a in range(3) if a != axis]
    xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in keep]
    if kw.pop("fill", False):
        ax.contourf(xs[0], xs[1], proj.T, levels=[0.5, 1.5], **kw)
    else:
        ax.contour(xs[0], xs[1], proj.T, levels=[0.5], **kw)


def heat_vmax(run):
    """The colour limit: the largest node flux over the run's iterates (98th percentile)."""
    return float(max(np.percentile(run.node_flux(k)[1], 98) for k in range(len(run.rows))))


def history_panels(run, axes, upto=None):
    """Drag, lift, heat, volume and the multipliers against iterate on ``axes`` (5)."""
    c = run.col
    it = np.arange(len(run.rows))
    m = len(it) if upto is None else min(upto + 1, len(it))

    def draw(ax, y, colour, label=None):
        ax.plot(it, y, color=colour, lw=1.0, alpha=0.3)
        ax.plot(it[:m], y[:m], color=colour, lw=2.2, label=label)
        if upto is not None:
            ax.plot([it[m - 1]], [y[m - 1]], "o", color=colour, ms=7, markeredgecolor="white",
                    markeredgewidth=1.4)  # fmt: skip

    ax = axes[0]
    if "drag" in run.refs:
        d = np.array([r["objective"] for r in run.refs["drag"]])
        ax.plot(d / d[0], color=OLD, lw=1.2, ls="--", label="drag only, V = V0")
    draw(ax, c["drag"] / run.d0, BLUE)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("drag D / D0", loc="left", fontsize=10, color=INK)
    if run.refs:
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = axes[1]
    draw(ax, c["lift"] / run.d0, ORANGE)
    ax.axhline(run.lift_min, color=MUTED, lw=0.9, ls="--")
    ax.text(0.5, run.lift_min + 0.005, f"floor {run.lift_min:g} D0", color=MUTED, fontsize=8,
            va="bottom")  # fmt: skip
    ax.set_ylim(
        min(-0.02, (c["lift"] / run.d0).min() - 0.01),
        max(run.lift_min * 1.25, (c["lift"] / run.d0).max() * 1.08),
    )
    ax.set_title("lift L / D0", loc="left", fontsize=10, color=INK)
    ax = axes[2]
    draw(ax, c["heat_measure"] / run.g0, GREEN)
    ax.axhline(float(run.cfg["heat_cap"]), color=MUTED, lw=0.9, ls="--")
    ax.text(0.5, float(run.cfg["heat_cap"]) + 0.005, "cap", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(min(0.85, (c["heat_measure"] / run.g0).min() - 0.03), 1.1)
    ax.set_title(r"peak heat flux $\dot q_{pk} / \dot q_{pk,0}$ (p = 8 norm)", loc="left",
                 fontsize=10, color=INK)  # fmt: skip
    ax = axes[3]
    draw(ax, c["area"] / run.v0, PINK)
    ax.axhline(run.area_floor, color=MUTED, lw=0.9, ls="--")
    ax.text(0.5, run.area_floor + 0.005, "floor", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("volume V / V0", loc="left", fontsize=10, color=INK)
    ax = axes[4]
    draw(ax, c["lift_multiplier"], ORANGE, "lift nu")
    draw(ax, c["heat_multiplier"], GREEN, "heat mu")
    draw(ax, c["thickness_multiplier"], VIOLET, "thickness lambda")
    ax.set_title("multipliers (dual ascent)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes:
        _spines(ax)
        ax.set_xlim(0, max(len(it) - 1, 1))
        ax.set_xlabel("iterate", fontsize=9, color=MUTED)
        if upto is not None:
            ax.axvline(upto, color=MUTED, lw=0.7)


def frame(run, k, path, vmax):
    kk = min(k, run.last)
    g = run.grid
    phi = run.phi(kk)
    hk = min(k, len(run.rows) - 1)  # the heat of the iterate's own window
    fig = plt.figure(figsize=(19.0, 11.0))
    gs = fig.add_gridspec(3, 15, left=0.035, right=0.985, top=0.92, bottom=0.05, wspace=2.2,
                          hspace=0.34)  # fmt: skip
    centre = np.array([0.5 * (VIEW_X[0] + VIEW_X[1]), 0.0, 0.0])
    half_x = 0.5 * (VIEW_X[1] - VIEW_X[0])
    views = (
        ("side, heat flux (lift +y up; from +z)", (0, 0, 1), Y_UP, (0, 1), 1.0),
        ("planform, heat flux (from above, +y)", (0, 1, 0), (0, 0, -1), (0, 2), -1.0),
    )
    for i, (label, eye, up, axes, sign) in enumerate(views):
        ax = fig.add_subplot(gs[0, 5 * i : 5 * i + 5])
        img = heat_image(run, hk, phi, centre, (half_x, 1.6), eye, up, vmax)
        _image_panel(ax, img, (VIEW_X[0], VIEW_X[1], -1.6 * sign, 1.6 * sign), label)
        _outline(ax, run.phi0, axes, g, colors=[MUTED], linewidths=0.9, linestyles="--")
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(-1.6 * sign, 1.6 * sign)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]  (flow ->)", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{'yz'[i]} [m]", fontsize=9, color=MUTED)
    ax = fig.add_subplot(gs[0, 10:13])
    img = heat_image(run, hk, phi, np.zeros(3), (3.0, 1.6), ISO, Y_UP, vmax)
    _image_panel(ax, img, (-3.0, 3.0, -1.6, 1.6), "isometric, heat flux")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # the front view (2026-11-05): the body as the stream sees it, its frontal area
    ax = fig.add_subplot(gs[0, 13:15])
    img = heat_image(run, hk, phi, np.zeros(3), (1.6, 1.6), FRONT, Y_UP, vmax)
    _image_panel(ax, img, (-1.6, 1.6, -1.6, 1.6), "front (from upstream)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax), cmap=HEAT_CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\dot q / \dot q_{stag,FM}$", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    nrho, speed, temp = run.plane(k)
    fields = (
        (nrho, "density n / n_inf, z = 0 (log)", run.n_top, "Greys"),
        (speed, "speed |u| / U_inf, z = 0", 1.1, "viridis"),
        (temp, "temperature T / T_inf, z = 0", float(np.nanmax(run.plane(0)[2])), "magma"),
    )
    for col, (field, label, top, cmap) in enumerate(fields):  # fmt: skip
        ax = fig.add_subplot(gs[1, 5 * col : 5 * col + 5])
        # density on a log scale up to the movie's peak (a linear 0-6 scale saturated the
        # compressed layer at the nose, 2026-11-05)
        norm = LogNorm(0.1, top) if col == 0 else Normalize(0.0, top)
        im = ax.imshow(np.maximum(field.T, 0.1) if col == 0 else field.T, origin="lower",
                       extent=(run.lo[0], run.hi[0], run.lo[1], run.hi[1]), cmap=cmap,
                       norm=norm, interpolation="bilinear")  # fmt: skip
        sl = np.take(phi, phi.shape[2] // 2, axis=2)
        xs = [g.lo[a] + g.h * np.arange(g.n[a]) for a in (0, 1)]
        ax.contourf(xs[0], xs[1], sl.T, levels=[-1e30, 0.0], colors=[BLUE])
        ax.contour(xs[0], xs[1], sl.T, levels=[0.0], colors=[INK], linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10, loc="left", color=INK, pad=3)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.set_xlabel("x [m]", fontsize=9, color=MUTED)
        ax.set_ylabel("y [m]", fontsize=9, color=MUTED)
    axes = [fig.add_subplot(gs[2, 3 * i : 3 * i + 3]) for i in range(5)]
    history_panels(run, axes, upto=min(k, len(run.rows) - 1))
    r = run.rows[hk]
    ended = "  (latest)" if k >= len(run.rows) - 1 else ""
    fig.suptitle(
        f"Minimum drag, L >= {run.lift_min:g} D0, $\\dot q_{{pk}} \\leq \\dot q_{{pk,0}}$, "
        f"V >= {run.area_floor:g} V0;  M 10, "
        f"Kn 0.5, diffuse 3-D ({run.cfg['optimizer']})  --  iterate {k}{ended}:  D/D0 {r['drag'] / run.d0:.3f}  "
        f"L/D0 {r['lift'] / run.d0:.3f}  L/D {r['lift'] / r['drag']:.3f}  $\\dot q_{{pk}}/\\dot q_{{pk,0}}$ "
        f"{r['heat_measure'] / run.g0:.3f}  V/V0 {r['area'] / run.v0:.3f}",
        fontsize=13, color=INK, x=0.035, ha="left",
    )  # fmt: skip
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def movie(run, vmax):
    frames = Path(figure_path(f"movie/{NAME}", f"{NAME}_frames"))
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    i = 0
    for k in range(len(run.rows)):
        drawn = frames / f"d_{k:04d}.png"
        frame(run, k, drawn, vmax)
        for _ in range(HOLD if k == len(run.rows) - 1 else REPEAT):
            (frames / f"f_{i:05d}.png").hardlink_to(drawn)
            i += 1
        print(f"iterate {k}", flush=True)
    mp4 = figure_path(f"movie/{NAME}", f"{NAME}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{FPS}",
         "-i", str(frames / "f_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True,
    )  # fmt: skip
    shutil.rmtree(frames)
    print("wrote", mp4, flush=True)


def convergence(run, phis):
    c = run.col
    it = np.arange(len(run.rows))
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.0))
    history_panels(run, [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0]])
    ax = axes[1, 1]
    if "ld" in run.refs:
        j = np.array([-r["objective"] for r in run.refs["ld"]])
        ax.plot(j, color=OLD, lw=1.2, ls="--", label="max L/D, V >= 0.5 V0")
    ax.plot(it, c["lift"] / c["drag"], color=ORANGE, lw=2.2, label="this run")
    ax.set_title("L / D", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(0, len(it) + 5)
    ax = axes[1, 2]
    ks = sorted(phis)
    ax.plot(ks, [incidence(phis[k], run.grid.lo, run.grid.h) for k in ks], color=BLUE, lw=2.2)
    ax.set_title("body incidence (principal axis, x-y) [deg]", loc="left", fontsize=10,
                 color=INK)  # fmt: skip
    ax = axes[1, 3]
    ax.plot(it, c["grad_norm"], color=BLUE, lw=2.2)
    ax.set_ylim(0.0, None)
    ax.set_title("shape-gradient norm |g| (Lagrangian)", loc="left", fontsize=10, color=INK)
    for ax in axes[1, 1:]:
        _spines(ax)
        ax.set_xlabel("iterate", fontsize=9, color=MUTED)
    fig.suptitle(
        f"{NAME} (M 10, Kn 0.5, diffuse, {run.cfg.get('variant', '')} fit, fnum x "
        f"{run.cfg.get('fnum_scale', '')}): {len(run.rows)} iterates; D0 = the sphere's drag; "
        "dashed: lsfilter3d references (other problems)",
        fontsize=12, color=INK, x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout()
    path = figure_path(f"optc/{NAME}", f"{NAME}_convergence.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def shapes(run, vmax):
    g = run.grid
    k1 = hk1 = len(run.rows) - 1  # the latest iterate with its own heat tallies
    phi0, phi1 = run.phi(0), run.phi(k1)
    fig = plt.figure(figsize=(18.0, 10.0))
    gs = fig.add_gridspec(2, 6, left=0.04, right=0.97, top=0.92, bottom=0.05, wspace=0.35,
                          hspace=0.22)  # fmt: skip
    for i, (axis, label, ylab) in enumerate(((2, "side (lift +y up)", "y [m]"),
                                             (1, "planform (from +y)", "z [m]"))):  # fmt: skip
        ax = fig.add_subplot(gs[0, 3 * i : 3 * i + 3])
        _silhouette(ax, phi1 < 0, axis, g, fill=True, colors=[BLUE], alpha=0.45)
        _silhouette(ax, phi1 < 0, axis, g, colors=[BLUE], linewidths=1.4)
        _silhouette(ax, phi0 < 0, axis, g, colors=[MUTED], linewidths=1.0, linestyles="--")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-1.6, 1.6)
        if axis == 1:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        _spines(ax)
        ax.set_xlabel("x [m]  (flow ->)", fontsize=9, color=MUTED)
        ax.set_ylabel(ylab, fontsize=9, color=MUTED)
        ax.set_title(f"{label}: iterate {k1} filled, initial body dashed", loc="left",
                     fontsize=10, color=INK)  # fmt: skip
    for j, (k, phi, hk) in enumerate(((0, phi0, 0), (k1, phi1, hk1))):
        ax = fig.add_subplot(gs[1, 3 * j])
        img = heat_image(run, hk, phi, np.zeros(3), (1.6, 1.6), FRONT, Y_UP, vmax)
        _image_panel(ax, img, (-1.6, 1.6, -1.6, 1.6), f"iterate {k}: from upstream")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 3 * j + 1 : 3 * j + 3])
        img = heat_image(run, hk, phi, np.zeros(3), (3.0, 1.6), ISO, Y_UP, vmax)
        r = run.rows[hk]
        _image_panel(
            ax,
            img,
            (-3.0, 3.0, -1.6, 1.6),
            f"iterate {k}: isometric  ($\\dot q_{{pk}}/\\dot q_{{pk,0}}$ {r['heat_measure'] / run.g0:.3f})",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax), cmap=HEAT_CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"surface heat flux $\dot q / \dot q_{stag,FM}$", fontsize=9, color=MUTED)
    fig.suptitle(f"{NAME} bodies: the latest iterate against the initial body, and the surface "
                 "heat flux (node Q / A, the heat constraint's own flux)", fontsize=12,
                 color=INK, x=0.04, ha="left")  # fmt: skip
    path = figure_path(f"optc/{NAME}", f"{NAME}_shapes.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def heat_distribution(run):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), sharey=True)
    p = float(run.cfg["heat_p"])
    for ax, k in zip(axes, (0, len(run.rows) - 1), strict=True):
        x, flux = run.node_flux(k)
        q, area = run.node_heat(k)
        lower = x[:, 1] < 0
        for mask, colour, label in ((lower, ORANGE, "lower (y < 0, windward when lifting)"),
                                    (~lower, BLUE, "upper (y >= 0)")):  # fmt: skip
            ax.scatter(x[mask, 0], flux[mask], s=6, color=colour, alpha=0.5, lw=0, label=label)
        gk = run.rows[k]["heat_measure"]
        ax.axhline(gk, color=GREEN, lw=1.2, ls="--")
        ax.text(x[:, 0].max(), gk, rf" $\dot q_{{pk}}$ = {gk:.3f} (p = {p:g})", color=GREEN, fontsize=9,
                va="bottom", ha="right")  # fmt: skip
        ax.set_title(f"iterate {k}: node heat flux along the body (max {flux.max():.3f})",
                     loc="left", fontsize=10, color=INK)  # fmt: skip
        ax.set_xlabel("x [m]  (flow ->)", fontsize=9, color=MUTED)
        _spines(ax)
    axes[0].set_ylabel(r"$\dot q / \dot q_{stag,FM}$", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, markerscale=2.5)
    fig.tight_layout()
    path = figure_path(f"optc/{NAME}", f"{NAME}_heat.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--tag", default=None, help="the run (results/optimize/<tag>)")
    ap.add_argument("--name", default="dlh3d", help="the figures' and movie's name")
    args = ap.parse_args()
    global TAG, NAME  # noqa: PLW0603 -- a scratch script's run selection
    TAG, NAME = args.tag or TAG, args.name
    run = Run()
    for k in range(len(run.rows)):
        got, want = run.heat_measure(k), run.rows[k]["heat_measure"]
        if abs(got / want - 1.0) > 1e-3:
            print(f"iterate {k}: recomputed G {got:.5f} != history {want:.5f}", flush=True)
    print("G recomputed at every iterate", flush=True)
    vmax = heat_vmax(run)
    if args.frame is not None:
        frame(run, args.frame, f"scratch/dlh3d/frame_{NAME}_{args.frame}.png", vmax)
        return
    if args.movie:
        movie(run, vmax)
        return
    phis = {k: run.phi(k) for k in range(0, run.last + 1)}
    convergence(run, phis)
    shapes(run, vmax)
    heat_distribution(run)


if __name__ == "__main__":
    main()
