"""Side-by-side MP4s of the optimiser comparison (2026-10-23), per regime:

* ``optax_all_<regime>.mp4``: every one-seed arm's body over its steady density field (3 x 4,
  left: the eight free arms, then the four centroid-fixed ones) and every arm's drag history
  to the shown iterate (right);
* ``optax_present_<regime>.mp4``: the presentation cut -- four arms (``PRESENT``) 2 x 2 on the
  left, the drag histories of ALL arms on the right.

An arm whose run ended holds its last body (marked "ended").  Playback is skewed: every
iterate of the first ``SLOW`` at 10 iterates / s, then every third at 45 / s (2026-10-23: 3x
slower and 1.5x faster than the earlier uniform 30 / s).  The bodies (zero contours) and fields
of the shown iterates are cached in ``scratch/optax/cache_<regime>.pkl``.

    PYTHONPATH=. python3 scratch/optax/compare_movies.py REGIME
        [--only all|present] [--frame K]   (--frame: one PNG of iterate K, no movie)
"""

import argparse
import dataclasses
import json
import pickle
import shutil
import subprocess
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
from analysis import ROOT, tag  # noqa: E402
from figure_paths import figure_path  # noqa: E402
from style import COLOURS, GRID, INK, LABELS, MUTED, ORDER, REGIME_TITLES, linestyle  # noqa: E402

PRESENT = ("ox_adam", "ox_adamcos", "lbfgs", "mma")
CAP = {"kn0.5": 450, "kn0.5_M10": 400}  # iterates shown (the longest arms are flat beyond)
VIEW = {"kn0.5": (-10.0, 12.0), "kn0.5_M10": (-2.0, 20.0)}  # x range of the body panels
DENSITY_MAX = {"kn0.5": 3.0, "kn0.5_M10": 3.0}  # n / n_inf colour limit: the free stream light grey
SLOW = 50  # iterates shown one by one, at SLOW_REPEAT video frames each
FPS, SLOW_REPEAT, FAST_STRIDE, FAST_REPEAT = 30, 3, 3, 2  # 10 iterates / s, then 45 / s


def shown_iterates(regime):
    """The iterates drawn and how many video frames (at ``FPS``) each is held."""
    slow = [(k, SLOW_REPEAT) for k in range(SLOW)]
    fast = [(k, FAST_REPEAT) for k in range(SLOW, CAP[regime] + 1, FAST_STRIDE)]
    return slow + fast


def cache(regime):
    path = Path(f"scratch/optax/cache_{regime}.pkl")
    shown = [k for k, _ in shown_iterates(regime)]
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved["shown"] == shown and set(saved["arms"]) == set(ORDER):
            return saved
    data = json.load(open("scratch/optax/analysis.json"))[regime]
    cfg = dict(O.default_config(),
               **json.load(open(f"{ROOT}/{tag(regime, 'S4_adam')}/history.json"))["config"])  # fmt: skip
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    out = {"shown": shown, "lo": lo, "hi": hi, "n": tuple(grid.n),
           "init": O.surface_polyline(ls0, lo, hi), "arms": {}}  # fmt: skip
    for arm in ORDER:
        d = data[arm]
        last = max(int(k) for k in d["position"])  # the last saved body
        n_frames = len(d["J"])  # one past the last saved field (a stopped run may lack it)
        while not Path(f"{ROOT}/{tag(regime, arm)}/frames_{n_frames - 1}.npz").exists():
            n_frames -= 1
        lines, fields = {}, {}
        for k in shown:
            kk = min(k, last)
            if kk not in lines:
                c = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{kk}.npy")
                lines[kk] = O.surface_polyline(
                    dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi
                )
            kf = min(k, n_frames - 1)
            if kf not in fields:
                fields[kf] = np.load(f"{ROOT}/{tag(regime, arm)}/frames_{kf}.npz")["nrho"][-1]
        out["arms"][arm] = {"last": last, "frames": n_frames, "lines": lines, "fields": fields,
                            "J": np.asarray(d["J"]), "J_mean": np.asarray(d["J_mean"])}  # fmt: skip
        print(f"{regime} {arm}: {len(lines)} bodies", flush=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _grid(field, n):
    """Per-cell ``(C,)`` -> ``n`` with x leading (row-major, as ``viz.grid_field``; split
    cells' sub-cells after the Cartesian ones are dropped)."""
    return np.asarray(field)[: int(np.prod(n))].reshape(n)


def _body_panel(ax, c, arm, k, regime, n_inf):
    a = c["arms"][arm]
    kk, kf = min(k, a["last"]), min(k, a["frames"] - 1)
    lo, hi = c["lo"], c["hi"]
    field = _grid(a["fields"][kf], c["n"]) / n_inf
    ax.imshow(field.T, origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), cmap="Greys",
              vmin=0.0, vmax=DENSITY_MAX[regime], interpolation="bilinear")  # fmt: skip
    for seg in c["init"]:
        ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=0.9, ls="--")
    for seg in a["lines"][kk]:
        ax.fill(seg[:, 0], seg[:, 1], color=COLOURS[arm], alpha=0.85, lw=0)
        ax.plot(seg[:, 0], seg[:, 1], color=INK, lw=0.8)
    ax.set_xlim(*VIEW[regime])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(GRID)
    jm = a["J_mean"][min(k, len(a["J_mean"]) - 1)]
    ended = "  (ended)" if k > a["last"] else ""
    ax.set_title(f"{LABELS[arm]}   J/J0 {jm:.3f}{ended}", fontsize=12, loc="left", color=INK, pad=3)


def _traces(ax, c, k, regime):
    for arm in ORDER:
        a = c["arms"][arm]
        m = a["J_mean"]
        it = np.arange(len(m))
        ls = linestyle(arm)
        ax.plot(it, m, color=COLOURS[arm], lw=1.0, alpha=0.25, ls=ls)
        upto = min(k + 1, len(m))
        ax.plot(it[:upto], m[:upto], color=COLOURS[arm], lw=2.4, ls=ls, label=LABELS[arm])
        ax.plot([upto - 1], [m[upto - 1]], "o", color=COLOURS[arm], ms=8,
                markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.axvline(k, color=MUTED, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlim(0, CAP[regime])
    ax.set_ylim(0.09 if regime == "kn0.5_M10" else 0.17, 1.05)
    ax.set_xlabel("iterate", fontsize=13)
    ax.set_ylabel("drag J / J0  (15-iterate running mean)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=10, loc="upper right", frameon=False, ncol=2, handlelength=2.6)


def frame(c, regime, k, arms, path):
    rows, cols = (2, 2) if len(arms) == 4 else (3, 4)
    # frame height fitted to the body panels (aspect ~2.2 at full box height): no empty bands;
    # the 3 x 4 frame is wider, its panels and the traces the size of the earlier 3 x 3 one's
    width, ratio = (19.2, 1.75) if rows == 2 else (23.4, 2.35)
    fig = plt.figure(figsize=(width, 7.2 if rows == 2 else 7.6))
    outer = fig.add_gridspec(1, 2, width_ratios=(ratio, 1.0), left=0.008, right=0.99, top=0.9,
                             bottom=0.08, wspace=0.1)  # fmt: skip
    left = outer[0].subgridspec(rows, cols, wspace=0.025, hspace=0.14)
    n_inf = float(np.median(_grid(c["arms"][arms[0]]["fields"][0], c["n"])[0]))  # inflow column
    for i, arm in enumerate(arms):
        _body_panel(fig.add_subplot(left[i // cols, i % cols]), c, arm, k, regime, n_inf)
    _traces(fig.add_subplot(outer[1]), c, k, regime)
    fig.suptitle(f"Drag minimisation at fixed area, {REGIME_TITLES[regime]}  --  iterate {k}"
                 "        (flow from the left; grey: gas density; dashed: initial circle)",
                 fontsize=16, color=INK, x=0.008, ha="left")  # fmt: skip
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def movie(c, regime, arms, name):
    frames = Path(figure_path("movie/optax", name))
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    drawn = frames / "drawn"
    drawn.mkdir()
    i = 0
    plan = shown_iterates(regime)
    # the last frame held for two seconds
    plan[-1] = (plan[-1][0], 2 * FPS)
    for n, (k, repeat) in enumerate(plan):
        frame(c, regime, k, arms, drawn / f"d_{n:04d}.png")
        for _ in range(repeat):
            (frames / f"f_{i:05d}.png").hardlink_to(drawn / f"d_{n:04d}.png")
            i += 1
    mp4 = figure_path("movie/optax", f"{name}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{FPS}",
         "-i", str(frames / "f_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True,
    )  # fmt: skip
    print("wrote", mp4, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regime", choices=list(CAP))
    ap.add_argument("--only", choices=("all", "present"), default=None)
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    c = cache(args.regime)
    layouts = {"all": tuple(ORDER), "present": PRESENT}
    for name, arms in layouts.items():
        if args.only not in (None, name):
            continue
        if args.frame is not None:
            frame(c, args.regime, args.frame, arms, f"scratch/optax/frame_{name}.png")
            continue
        movie(c, args.regime, arms, f"optax_{name}_{args.regime}")


if __name__ == "__main__":
    main()
