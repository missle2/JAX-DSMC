"""Side-by-side MP4 of the optrules arms on one regime: every arm's shape at the same iterate
(left; an arm whose run ended -- or whose trust region entered the repeat loop -- holds its
last productive shape, marked "stopped") and the drag histories J / J0 with the iterate marked
(right).  Each arm costs the same seconds per iterate, so the clock is shared.

    PYTHONPATH=. python3 scratch/optrules/compare_movie.py REGIME [--stride 2 --fps 10]

REGIME ``kn0.5`` or ``kn0.5_M10``; writes ``examples/mesh/figures/movie/optrules_<REGIME>.mp4``.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import dataclasses  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import optimize as O  # noqa: E402
from analysis import ARMS, COLOURS, LABELS, REGIMES, load, run_dir  # noqa: E402
from figure_paths import figure_path  # noqa: E402

SHOWN = ("base", *ARMS)


def polylines(regime, arm, d, cfg, ls0, lo, hi, iterates):
    stop = d["loop_from"] if d["loop_from"] is not None else len(d["J"])
    out = {}
    for k in iterates:
        kk = min(k, stop)
        if kk not in out:
            c = np.load(f"{run_dir(regime, arm)}/coef_{kk}.npy")
            out[kk] = O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi)
    return stop, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regime", choices=list(REGIMES))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    regime = args.regime
    data = {a: load(regime, a) for a in SHOWN}
    cfg = dict(O.default_config(), **data["adam"]["hist"]["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n = max(len(d["J"]) for d in data.values())
    iterates = list(range(0, n + 1, args.stride))
    lines, stops = {}, {}
    for arm in SHOWN:
        stops[arm], lines[arm] = polylines(regime, arm, data[arm], cfg, ls0, lo, hi, iterates)
        print(arm, "shapes", len(lines[arm]), flush=True)
    init = O.surface_polyline(ls0, lo, hi)
    pts = np.concatenate([s for arm in SHOWN for segs in lines[arm].values() for s in segs])
    centre = 0.5 * (pts.min(0) + pts.max(0))
    half = 0.52 * (pts.max(0) - pts.min(0))
    frames = Path(figure_path("movie", f"optrules_{regime}"))
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    for i, k in enumerate(iterates):
        fig = plt.figure(figsize=(16, 6.0))
        gs = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 0.15, 2.2))
        for col, arm in enumerate(SHOWN):
            ax = fig.add_subplot(gs[col // 3, col % 3])
            kk = min(k, stops[arm])
            for seg in init:
                ax.plot(seg[:, 0], seg[:, 1], color="#8a8a86", lw=0.7, ls="--")
            for seg in lines[arm][kk]:
                ax.fill(seg[:, 0], seg[:, 1], color=COLOURS[arm], alpha=0.3, lw=0)
                ax.plot(seg[:, 0], seg[:, 1], color=COLOURS[arm], lw=1.3)
            d = data[arm]
            jk = d["J"][min(kk, len(d["J"]) - 1)] / d["J"][0]
            state = "" if k <= stops[arm] else "  (stopped)"
            name = "Adam (earlier)" if arm == "base" else LABELS[arm]
            ax.set_title(f"{name}\nJ/J0 {jk:.3f}{state}", fontsize=8)
            ax.set_xlim(centre[0] - half.max(), centre[0] + half.max())
            ax.set_ylim(centre[1] - half[1] * 1.3, centre[1] + half[1] * 1.3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[:, 4])
        for arm in SHOWN:
            d = data[arm]
            end = min(stops[arm] + 1, len(d["J"]))
            j = d["J"][:end] / d["J"][0]
            ax.plot(np.arange(end), j, color=COLOURS[arm], lw=0.8, alpha=0.35,
                    ls="--" if arm == "base" else "-")  # fmt: skip
            m = min(k + 1, end)
            ax.plot(np.arange(m), j[:m], color=COLOURS[arm], lw=1.6,
                    ls="--" if arm == "base" else "-", label=LABELS[arm])  # fmt: skip
            ax.plot([m - 1], [j[m - 1]], "o", color=COLOURS[arm], ms=5)
        ax.axvline(k, color="#52514e", lw=0.6)
        ax.set_yscale("log")
        ax.set_xlim(0, n)
        ax.set_xlabel("iterate (same seconds per iterate for every arm)")
        ax.set_ylabel("drag J / J0")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"optimiser step rules, {REGIMES[regime]} -- iterate {k}  (flow from the "
                     "left; dashed: initial circle)", fontsize=10)  # fmt: skip
        fig.savefig(frames / f"f_{i:04d}.png", dpi=80)
        plt.close(fig)
    mp4 = figure_path("movie", f"optrules_{regime}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps}",
         "-i", str(frames / "f_%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4],
        check=True,
    )  # fmt: skip
    print("wrote", mp4, len(iterates), "frames")


if __name__ == "__main__":
    main()
