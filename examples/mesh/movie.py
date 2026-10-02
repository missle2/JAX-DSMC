"""A GIF of a level-set optimisation: the flow filling in around every iterate's shape,
then the shape stepping to the next iterate, from the optimiser's artifacts.

    python3 -m examples.mesh.movie --tag circle_fine_S4_gd [--fps 6 --morph 8 --hold 4 --stride 1]

Reads ``results/optimize/<tag>/{history.json, coef_<k>.npy, frames_<k>.npz}``
(``examples.mesh.optimize`` writes ``frames_<k>.npz`` for the first seed of iterate ``k``:
per-cell ``nrho``, ``u``, ``temp`` every ``frames_every`` warm-up steps from the empty box
and at the end of the captured window).  Per iterate the movie shows the frames of the
fill (panels ``nrho`` and ``|u|`` with the body carved out, colour limits fixed over the
whole movie), holds the steady state ``--hold`` frames, then ``--morph`` frames of the
body's zero contour blending linearly from ``coef_k`` to ``coef_{k+1}`` over the held
steady field; the third panel is the drag history with the current iterate marked.
``--stride s`` shows every ``s``-th iterate (the morph then spans ``s`` shape steps and
the final iterate is always shown), which keeps a 250-iterate run's GIF to a few hundred
frames.
PNG frames go to ``figures/movie/<tag>/`` and the GIF to ``figures/optimize_<tag>.gif``
(``--mp4``: an H.264 ``figures/optimize_<tag>.mp4`` through the ``ffmpeg`` executable
instead -- a tenth of the GIF's size, so longer runs can keep more frames).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimize as O  # the optimiser module: its artifacts are read here

from jax_dsmc import viz
from jax_dsmc.mesh import levelset as L

BODY = "#3b3a37"
SPEED_MAX = 400.0


def load_run(out: Path):
    """``(cfg, rows, coefs, frames)``: every iterate ``k`` with both ``frames_k.npz`` and
    ``coef_{k+1}.npy`` present (the movie needs the shape it steps to)."""
    hist = json.loads((out / "history.json").read_text())
    cfg, rows = hist["config"], hist["rows"]
    coefs, frames = [], []
    for k in range(len(rows) + 1):
        c = out / f"coef_{k}.npy"
        if not c.exists():
            break
        coefs.append(np.load(c))
        f = out / f"frames_{k}.npz"
        if f.exists():
            z = np.load(f)
            frames.append({key: z[key] for key in z.files})
    return cfg, rows, coefs, frames


def phi_on(ls, xx, yy):
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    return np.asarray(L.value(ls, pts)).reshape(xx.shape)


class Renderer:
    """Draws one frame: the two field panels with the body of ``coef`` carved out and its
    zero contour, and the drag history with iterate ``k`` marked."""

    def __init__(self, cfg, rows, grid, ls0, nrho_max, fine=300):
        self.cfg, self.rows, self.grid, self.ls0 = cfg, rows, grid, ls0
        self.lo, self.hi = np.asarray(grid.lo), np.asarray(grid.hi)
        self.limits = {"nrho": (0.0, nrho_max), "|u|": (0.0, SPEED_MAX)}
        xs = np.linspace(self.lo[0], self.hi[0], fine)
        ys = np.linspace(self.lo[1], self.hi[1], fine)
        self.xx, self.yy = np.meshgrid(xs, ys, indexing="ij")
        self.j = np.array([r["objective"] for r in rows])
        self.se = np.array([r["objective_se"] for r in rows])

    def phi(self, coef):
        return phi_on(dataclasses.replace(self.ls0, coef=jnp.asarray(coef)), self.xx, self.yy)

    def draw(self, path, fields, phi, k, title):
        fig, axes = viz.make_figure(1, 3, figsize=(13.2, 3.9))
        for ax, name in zip(axes[:2], ("nrho", "|u|"), strict=True):
            vmin, vmax = self.limits[name]
            im = viz.draw_field(
                ax, fields[name], self.grid.n, self.lo, self.hi, label=name, vmin=vmin, vmax=vmax
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.contourf(self.xx, self.yy, phi, levels=[-1e30, 0.0], colors=[BODY])
            ax.contour(self.xx, self.yy, phi, levels=[0.0], colors=[viz.SURFACE], linewidths=1.0)
            ax.set_aspect("equal")
        ax = axes[2]
        it = np.arange(len(self.j))
        ax.errorbar(it, self.j, yerr=self.se, color=viz.REFERENCE, lw=1.0, capsize=2, zorder=1)
        ax.plot(
            it[: k + 1], self.j[: k + 1], color=viz.SERIES_ANALYTIC, lw=viz.LINE_WIDTH, zorder=2
        )
        ax.plot([k], [self.j[k]], "o", color=viz.SERIES_MEASURED, ms=7, zorder=3)
        ax.set_xlabel("iterate")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("windowed drag [N/m]")
        ax.set_title(f"J = {self.j[k]:.4g}", fontsize=10)
        fig.suptitle(title, color=viz.INK, fontsize=11)
        viz.savefig(fig, path, dpi=80)


def fields_of(frame, i):
    u = frame["u"][i]
    return {"nrho": frame["nrho"][i], "|u|": np.sqrt(np.sum(u**2, axis=1))}


def render(out: Path, fig_dir: Path, morph: int, hold: int, every: int, stride: int = 1):
    """Every PNG frame of the movie, in order; returns their paths."""
    cfg, rows, coefs, frames = load_run(out)
    if not frames:
        raise SystemExit(f"no frames_<k>.npz under {out} (frames_every was 0?)")
    grid, ls0 = O.initial_levelset(cfg)
    steady = np.concatenate([f["nrho"][-1] for f in frames])
    nrho_max = float(np.percentile(steady, 99)) * 1.1
    r = Renderer(cfg, rows, grid, ls0, nrho_max)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n_iter = min(len(frames), len(coefs) - 1)
    shown = sorted({*range(0, n_iter, stride), n_iter - 1})
    for k in shown:
        f = frames[k]
        k_next = min(k + stride, n_iter)
        phi_k = r.phi(coefs[k])
        steps = f["step"]
        # every n-th fill frame, the last (steady) one always included
        for i in sorted({*range(0, len(steps), every), len(steps) - 1}):
            paths.append(fig_dir / f"f_{len(paths):04d}.png")
            r.draw(paths[-1], fields_of(f, i), phi_k, k, f"iterate {k}   step {int(steps[i])}")
        last = fields_of(f, len(steps) - 1)
        for _ in range(hold):
            paths.append(fig_dir / f"f_{len(paths):04d}.png")
            r.draw(paths[-1], last, phi_k, k, f"iterate {k}   steady (window mean of the drag)")
        for m in range(1, morph + 1):
            a = m / morph
            phi_m = r.phi((1.0 - a) * coefs[k] + a * coefs[k_next])
            paths.append(fig_dir / f"f_{len(paths):04d}.png")
            r.draw(paths[-1], last, phi_m, k, f"iterate {k} -> {k_next}   shape step")
        print(f"iterate {k}: {len(paths)} frames so far", flush=True)
    return paths


def assemble(paths, gif: Path, fps: float):
    from PIL import Image  # noqa: PLC0415 -- matplotlib's dependency, no new requirement

    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128) for p in paths]
    imgs[0].save(
        gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True
    )
    return gif


def assemble_mp4(fig_dir: Path, n_frames: int, mp4: Path, fps: float):
    """The frames ``f_0000.png ..`` of ``fig_dir`` (the first ``n_frames``: a previous,
    longer render may have left more) -> an H.264 MP4 by the ``ffmpeg`` executable."""
    import subprocess  # noqa: PLC0415

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}",
        "-i", str(fig_dir / "f_%04d.png"), "-frames:v", str(n_frames),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "23", str(mp4),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return mp4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="circle")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--morph", type=int, default=8, help="frames of the shape step")
    ap.add_argument("--hold", type=int, default=4, help="frames holding the steady state")
    ap.add_argument("--every", type=int, default=1, help="use every n-th fill frame")
    ap.add_argument("--stride", type=int, default=1, help="show every n-th iterate")
    ap.add_argument("--mp4", action="store_true", help="write an MP4 (ffmpeg) instead of a GIF")
    args = ap.parse_args(argv)
    out = Path(O.RESULTS) / args.tag
    fig_dir = Path(O.FIGURES) / "movie" / args.tag
    paths = render(out, fig_dir, args.morph, args.hold, args.every, args.stride)
    if args.mp4:
        movie = assemble_mp4(
            fig_dir, len(paths), Path(O.FIGURES) / f"optimize_{args.tag}.mp4", args.fps
        )
    else:
        movie = assemble(paths, Path(O.FIGURES) / f"optimize_{args.tag}.gif", args.fps)
    print(f"wrote {movie} ({len(paths)} frames, {movie.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
