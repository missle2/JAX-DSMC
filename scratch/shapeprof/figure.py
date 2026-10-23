"""Per-phase seconds of one seed of the optimiser's gradient on the H200 MIG 2g.35gb slice
(scratch/shapeprof/profile.py, optm Kn 0.5 M 10 iterates 30 and 156): the 2026-10-19 profile
(before the guard's per-cell leverages, ``git show c80d533``) against the 2026-10-20 one.

    PYTHONPATH=. python3 scratch/shapeprof/figure.py

The 2026-10-20 ``warm500`` call compiled the warm path inside its timed call (the cold path
came from the persistent cache), so its warm-up is drawn hatched; the estimate beside it puts
the 500 settle steps at the captured window's per-step rate.  Writes
``examples/mesh/figures/optc/gates/shapeprof.png``.
"""

import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402

BEFORE = "c80d533"
PHASES = ("build", "warmup", "capture", "replay", "backward", "shape")
COLOURS = {"build": "#8a8a86", "warmup": "#0072B2", "capture": "#56B4E9", "replay": "#009E73",
           "backward": "#D55E00", "shape": "#E69F00"}  # fmt: skip
SETTINGS = ("unguarded", "plain_k32", "shipped")


def load(it, rev=None):
    path = f"scratch/shapeprof/cluster/optm_it{it}.json"
    if rev is None:
        with open(path) as fh:
            return json.load(fh)
    return json.loads(subprocess.check_output(["git", "show", f"{rev}:{path}"]))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, it in zip(axes, (30, 156), strict=True):
        old, new = load(it, BEFORE)["runs"], load(it)["runs"]
        bars = []
        for name in SETTINGS:
            bars.append((f"{name}\n10-19", old[name]["second"], None))
            bars.append((f"{name}\n10-20", new[name]["second"], None))
        warm = new["warm500"]["second"]
        bars.append(("warm500\n(+compile)", warm, "warmup"))
        steps = new["shipped"]["config"]["steps"] if "config" in new["shipped"] else 300
        est = dict(new["shipped"]["second"])
        est["warmup"] = 500 * new["shipped"]["second"]["capture"] / steps
        bars.append(("warm500\nestimate", est, "est"))
        for i, (label, rec, flag) in enumerate(bars):
            bottom = 0.0
            for p in PHASES:
                v = rec[p]
                hatch = "//" if (flag == "warmup" and p == "warmup") else None
                alpha = 0.5 if (flag == "est" and p == "warmup") else 1.0
                ax.bar(i, v, bottom=bottom, color=COLOURS[p], hatch=hatch, alpha=alpha,
                       edgecolor="white", lw=0.6, label=p if i == 0 else None)  # fmt: skip
                bottom += v
            ax.text(i, bottom + 2, f"{bottom:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7)
        ax.set_title(f"optm Kn 0.5 M 10 iterate {it}: one seed, seconds per phase (timed call)",
                     fontsize=9)  # fmt: skip
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("seconds (H200 MIG 2g.35gb)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("guarded backward 40-42 s -> 33-34 s (per-cell leverage tables); warm start: "
                 "the timed call paid the warm path's compile (hatched)", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(figure_path("optc/gates", "shapeprof.png"), dpi=140)


if __name__ == "__main__":
    main()
