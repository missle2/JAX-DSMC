"""Figure of the optrules2 / optrules1seed runs (2026-10-21, partial): the reworked trust region
and every rule with ONE seed per iterate, against the S = 4 references of optrules.

    PYTHONPATH=. python3 scratch/optrules/noise.py

Panels per regime: J / J0 (J0 the S = 4 initial drag; one-seed curves as a 10-iterate running
mean over the raw values) against the iterate and against the seed evaluations (iterates x S:
the hardware-neutral cost -- the runs sit on H200 2g.35gb and RTX 1g.24gb slices, ~1.8x apart),
and the trust radius of the reworked TR (S = 4 and S = 1).  A crashed run ends in an x.
Writes ``examples/mesh/figures/optc/optrules/optrules_noise.png``.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402

ROOT = "examples/mesh/results/optimize"
LOGS = "examples/mesh/cluster/logs"
REGIMES = {"kn0.5": "Kn 0.5, M 0.3 (deck)", "kn0.5_M10": "Kn 0.5, M 10"}
# label, tag (regime substituted), seeds, colour, line style
RUNS = (
    ("S4 Adam", "circle_long_{r}_S4_adam_tan_sm8_free_g32", 4, "#0072B2", "-"),
    ("S4 TR (10-19)", "circle_long_{r}_S4_tr_tan_sm8_free_g32", 4, "#D55E00", ":"),
    ("S4 TR reworked", "circle_long_{r}_S4_tr2_tan_sm8_free_g32_w500", 4, "#D55E00", "-"),
    ("S1 Adam", "circle_long_{r}_S1_adam_tan_sm8_free_g32_w500", 1, "#0072B2", "--"),
    ("S1 TR", "circle_long_{r}_S1_tr_tan_sm8_free_g32_w500", 1, "#D55E00", "--"),
    ("S1 L-BFGS", "circle_long_{r}_S1_lbfgs_tan_sm8_free_g32_w500", 1, "#009E73", "--"),
    ("S1 MMA", "circle_long_{r}_S1_mma_tan_sm8_free_g32_w500", 1, "#E69F00", "--"),
)


def rows(tag):
    with open(f"{ROOT}/{tag}/history.json") as fh:
        return json.load(fh)["rows"]


def crashed(tag):
    """Whether the run's newest log ends in a Python traceback."""
    import glob  # noqa: PLC0415

    name = tag.replace("circle_long_", "").replace("_tan_sm8_free_g32", "").replace("_w500", "")
    # the optrules2 job names drop the "S4_" of the tag
    logs = sorted(glob.glob(f"{LOGS}/optrules*_{name}_*.out")
                  + glob.glob(f"{LOGS}/optrules*_{name.replace('S4_', '')}_*.out"))  # fmt: skip
    if not logs:
        return False
    with open(logs[-1]) as fh:
        text = fh.read()
    return "Traceback" in text[-20000:]


def smooth(j, n=10):
    if len(j) < n:
        return np.arange(len(j)), j
    return np.arange(n - 1, len(j)), np.convolve(j, np.ones(n) / n, mode="valid")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for row, regime in enumerate(REGIMES):
        j0 = rows(RUNS[0][1].format(r=regime))[0]["objective"]
        for label, tag, seeds, colour, ls in RUNS:
            tag = tag.format(r=regime)
            try:
                h = rows(tag)
            except FileNotFoundError:
                continue
            j = np.array([r["objective"] for r in h]) / j0
            x, js = smooth(j) if seeds == 1 else (np.arange(len(j)), j)
            for col, scale in ((0, 1), (1, seeds)):
                ax = axes[row, col]
                if seeds == 1:
                    ax.plot(np.arange(len(j)) * scale, j, color=colour, lw=0.4, alpha=0.25)
                ax.plot(x * scale, js, color=colour, ls=ls, lw=1.4, label=label)
                if crashed(tag):
                    ax.plot([x[-1] * scale], [js[-1]], "x", color=colour, ms=9, mew=2)
            if "rebase" in h[-1]:
                ax = axes[row, 2]
                rad = [r["radius"] for r in h]
                ax.semilogy(rad, color=colour, ls=ls, lw=1.0, label=f"{label} radius")
                reb = [r["iterate"] for r in h if r.get("rebase")]
                ax.plot(reb, [0.0125 * (0.85 if seeds == 1 else 0.75)] * len(reb), "|",
                        color=colour, ms=6, label=f"{label}: floor rejection -> new seeds")  # fmt: skip
        for col, xlabel in ((0, "iterate"), (1, "seed evaluations (iterates x S)")):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("J / J0 (S = 4 initial drag)")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{REGIMES[regime]}: S = 1 as a 10-iterate mean (raw faint); x = crashed",
                         fontsize=9)  # fmt: skip
        axes[row, 2].axhline(0.0125, color="k", lw=0.6, ls=":")
        axes[row, 2].set_xlabel("iterate")
        axes[row, 2].set_ylabel("trust radius")
        axes[row, 2].set_title(f"{REGIMES[regime]}: reworked TR radius", fontsize=9)
        axes[row, 2].grid(alpha=0.3, which="both")
        axes[row, 2].legend(fontsize=6)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path("optc/optrules", "optrules_noise.png"), dpi=130)


if __name__ == "__main__":
    main()
