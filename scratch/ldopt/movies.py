"""Side-by-side MP4s of the 2026-10-26 runs (scratch/optax/compare_movies.py's layout):

* ``ldopt_ldrc.mp4``: the four unconstrained maximum-L / D runs (``ldopt`` ``_ldrc_none_``: Adam
  and MMA at Kn 0.5 / M 0.3 and Kn 0.5 / M 10), bodies over their steady density 2 x 2, the
  L / D histories on the right;
* ``optrc_m10.mp4``: the M 10 drag runs with the slide removed (``optrc``, ``--recentre 1``)
  beside the free runs they repeat (``ox_adam_fix``, ``mma_s``), J / J0 on the right.

A run that ended holds its last body (marked "ended").  Playback as compare_movies: every
iterate of the first ``SLOW`` at 10 iterates / s, then every third at 45 / s.

    PYTHONPATH=. python3 scratch/ldopt/movies.py [ld|rc] [--frame K]
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
from figure_paths import figure_path  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

ROOT = "examples/mesh/results/optimize"
ADAM, MMA = "#2a78d6", "#e34948"  # style.COLOURS' Adam and MMA
SLOW = 50
FPS, SLOW_REPEAT, FAST_STRIDE, FAST_REPEAT = 30, 3, 3, 2
RUNNING = 15  # iterates in the traces' running mean
DENSITY_MAX = 3.0  # n / n_inf colour limit

# name -> (tag, panel title, colour, trace line style)
MOVIES = {
    "ld": {
        "runs": {
            "kn0.5 Adam": ("circle_long_kn0.5_S1_ox_adam_ldrc_none_sm8_free_g32_w500",
                           "Adam, Kn 0.5 M 0.3", ADAM, "-"),
            "kn0.5 MMA": ("circle_long_kn0.5_S1_mma_s_ldrc_none_sm8_free_g32_w500",
                          "MMA, Kn 0.5 M 0.3", MMA, "-"),
            "M10 Adam": ("circle_long_kn0.5_M10_S1_ox_adam_ldrc_none_sm8_free_g32_w500",
                         "Adam, Kn 0.5 M 10", ADAM, "--"),
            "M10 MMA": ("circle_long_kn0.5_M10_S1_mma_s_ldrc_none_sm8_free_g32_w500",
                        "MMA, Kn 0.5 M 10", MMA, "--"),
        },  # fmt: skip
        "view": (-2.0, 12.0),
        "title": "Maximum L / D, no area constraint (recentred)",
        "lift": True,
    },
    "rc": {
        "runs": {
            "Adam free": ("circle_long_kn0.5_M10_S1_ox_adam_fix_tan_sm8_free_g32_w500",
                          "Adam, free (slides)", ADAM, "--"),
            "Adam rc": ("circle_long_kn0.5_M10_S1_ox_adam_rc_tan_sm8_free_g32_w500",
                        "Adam, recentred", ADAM, "-"),
            "MMA free": ("circle_long_kn0.5_M10_S1_mma_s_tan_sm8_free_g32_w500",
                         "MMA, free (slides)", MMA, "--"),
            "MMA rc": ("circle_long_kn0.5_M10_S1_mma_s_rc_tan_sm8_free_g32_w500",
                       "MMA, recentred", MMA, "-"),
        },  # fmt: skip
        "view": (-10.0, 20.0),
        "title": "Drag minimisation at fixed area, Kn 0.5 M 10: free vs recentred",
        "lift": False,
    },
}


def _running(x, w=RUNNING):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


def shown_iterates(cap):
    slow = [(k, SLOW_REPEAT) for k in range(min(SLOW, cap + 1))]
    fast = [(k, FAST_REPEAT) for k in range(SLOW, cap + 1, FAST_STRIDE)]
    return slow + fast


def cache(name):
    spec = MOVIES[name]
    path = Path(f"scratch/ldopt/cache_{name}.pkl")
    runs = {}
    for key, (tag, *_rest) in spec["runs"].items():
        rows = json.load(open(f"{ROOT}/{tag}/history.json"))["rows"]
        runs[key] = len(rows)
    cap = max(runs.values())
    shown = [k for k, _ in shown_iterates(cap)]
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved["shown"] == shown and saved["lengths"] == runs:
            return saved
    first = spec["runs"][next(iter(spec["runs"]))][0]
    cfg = dict(O.default_config(), **json.load(open(f"{ROOT}/{first}/history.json"))["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    out = {"shown": shown, "lengths": runs, "cap": cap, "lo": lo, "hi": hi,
           "n": tuple(grid.n), "init": O.surface_polyline(ls0, lo, hi), "runs": {}}  # fmt: skip
    for key, (tag, *_rest) in spec["runs"].items():
        rows = json.load(open(f"{ROOT}/{tag}/history.json"))["rows"]
        j = np.array([r["objective"] for r in rows])
        last = len(rows) - 1
        while not Path(f"{ROOT}/{tag}/coef_{last}.npy").exists():
            last -= 1
        n_frames = len(rows)
        while not Path(f"{ROOT}/{tag}/frames_{n_frames - 1}.npz").exists():
            n_frames -= 1
        lines, fields = {}, {}
        for k in shown:
            kk = min(k, last)
            if kk not in lines:
                c = np.load(f"{ROOT}/{tag}/coef_{kk}.npy")
                lines[kk] = O.surface_polyline(
                    dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi
                )
            kf = min(k, n_frames - 1)
            if kf not in fields:
                fields[kf] = np.load(f"{ROOT}/{tag}/frames_{kf}.npz")["nrho"][-1]
        area = np.array([r["area"] for r in rows])
        out["runs"][key] = {"last": last, "frames": n_frames, "lines": lines, "fields": fields,
                            "J": j, "area": area / area[0]}  # fmt: skip
        print(f"{name} {key}: {len(lines)} bodies", flush=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _grid(field, n):
    return np.asarray(field)[: int(np.prod(n))].reshape(n)


def _value(spec, a):
    """The traced quantity: L / D (running mean) or J / J0."""
    if spec["lift"]:
        return _running(-a["J"])
    return _running(a["J"]) / a["J"][0]


def _body_panel(ax, c, spec, key, k):
    a = c["runs"][key]
    _tag, title, colour, _ls = spec["runs"][key]
    kk, kf = min(k, a["last"]), min(k, a["frames"] - 1)
    lo, hi = c["lo"], c["hi"]
    field = _grid(a["fields"][kf], c["n"])
    n_inf = float(np.median(_grid(a["fields"][0], c["n"])[0]))  # the inflow column
    ax.imshow((field / n_inf).T, origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]),
              cmap="Greys", vmin=0.0, vmax=DENSITY_MAX, interpolation="bilinear")  # fmt: skip
    for seg in c["init"]:
        ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=0.9, ls="--")
    for seg in a["lines"][kk]:
        ax.fill(seg[:, 0], seg[:, 1], color=colour, alpha=0.85, lw=0)
        ax.plot(seg[:, 0], seg[:, 1], color=INK, lw=0.8)
    ax.set_xlim(*spec["view"])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)
    v = _value(spec, a)[kk]
    measure = f"L/D {v:.2f}" if spec["lift"] else f"J/J0 {v:.3f}"
    ended = "  (ended)" if k > a["last"] else ""
    ax.set_title(f"{title}   {measure}   area {a['area'][kk]:.2f} A0{ended}", fontsize=12,
                 loc="left", color=INK, pad=3)  # fmt: skip


def _traces(ax, c, spec, k):
    for key, (_tag, title, colour, ls) in spec["runs"].items():
        a = c["runs"][key]
        m = _value(spec, a)
        it = np.arange(len(m))
        ax.plot(it, m, color=colour, lw=1.0, alpha=0.25, ls=ls)
        upto = min(k + 1, len(m))
        ax.plot(it[:upto], m[:upto], color=colour, lw=2.4, ls=ls, label=title)
        ax.plot([upto - 1], [m[upto - 1]], "o", color=colour, ms=8,
                markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    ax.axvline(k, color=MUTED, lw=0.8)
    ax.set_xlim(0, c["cap"])
    if spec["lift"]:
        ax.set_ylabel(f"L / D  ({RUNNING}-iterate running mean)", fontsize=13)
    else:
        ax.set_yscale("log")
        ax.set_ylim(0.06, 1.05)
        ax.set_ylabel(f"drag J / J0  ({RUNNING}-iterate running mean)", fontsize=13)
    ax.set_xlabel("iterate", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=10, loc="best", frameon=False, handlelength=2.6)


def frame(c, name, k, path):
    spec = MOVIES[name]
    span = spec["view"][1] - spec["view"][0]
    ratio = 1.75 if span > 20 else 1.2
    fig = plt.figure(figsize=(19.2, 6.0) if span > 20 else (16.0, 7.2))
    outer = fig.add_gridspec(1, 2, width_ratios=(ratio, 1.0), left=0.008, right=0.99, top=0.9,
                             bottom=0.08, wspace=0.1)  # fmt: skip
    left = outer[0].subgridspec(2, 2, wspace=0.025, hspace=0.14)
    for i, key in enumerate(spec["runs"]):
        _body_panel(fig.add_subplot(left[i // 2, i % 2]), c, spec, key, k)
    _traces(fig.add_subplot(outer[1]), c, spec, k)
    fig.suptitle(f"{spec['title']}  --  iterate {k}        (flow from the left; grey: gas "
                 "density; dashed: initial circle)", fontsize=15, color=INK, x=0.008,
                 ha="left")  # fmt: skip
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def movie(c, name):
    out = {"ld": "ldopt_ldrc", "rc": "optrc_m10"}[name]
    frames = Path(figure_path("movie/ldopt", out))
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    drawn = frames / "drawn"
    drawn.mkdir()
    plan = shown_iterates(c["cap"])
    plan[-1] = (plan[-1][0], 2 * FPS)  # the last frame held for two seconds
    i = 0
    for n, (k, repeat) in enumerate(plan):
        frame(c, name, k, drawn / f"d_{n:04d}.png")
        for _ in range(repeat):
            (frames / f"f_{i:05d}.png").hardlink_to(drawn / f"d_{n:04d}.png")
            i += 1
    mp4 = figure_path("movie/ldopt", f"{out}.mp4")
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
    ap.add_argument("name", choices=list(MOVIES))
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    c = cache(args.name)
    if args.frame is not None:
        frame(c, args.name, args.frame, f"scratch/ldopt/frame_{args.name}.png")
        return
    movie(c, args.name)


if __name__ == "__main__":
    main()
