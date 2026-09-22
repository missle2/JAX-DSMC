"""Figure: route (b) particle-carried ensemble adjoint vs route (a) P3 retained frames vs
fixed-count CRN finite differences.  Reads the JSON artifacts written by
``scratch/bench/arbiter_two_step.py`` (two steps, 240 seeds) and
``scratch/bench/compare_ab.py`` (20 steps, 16 seeds).

    python3 -m examples.p3_replication.route_b_report
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
BENCH = os.path.join(HERE, "results", "route_b")
FIGURES = os.path.join(HERE, "figures")
# fixed-order categorical palette (validated: dataviz skill, light surface)
C_B, C_A, C_FD, C_PAPER = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
PAPER = (0.15171883, 0.15157001)


def _se(x):
    x = np.asarray(x)
    return x.std(axis=0, ddof=1) / np.sqrt(len(x))


def load_two_step():
    with open(os.path.join(BENCH, "arbiter_two_step.json"), encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    b = np.array([r["b_exact"] for r in rows])
    fd = {h: np.array([r["fd"][h] for r in rows]) for h in ("0.2", "0.1", "0.05")}
    a_rows = [r for r in rows if "a" in r]
    a = np.array([r["a"] for r in a_rows])
    b_for_a = np.array([r["b_exact"] for r in a_rows])
    return b, fd, a, b_for_a


def load_twenty_step(name="compare_ab_N2000_R512.json"):
    with open(os.path.join(BENCH, name), encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    return (np.array([r[k] for r in rows]) for k in ("a", "b", "fd"))


def _rows(name):
    with open(os.path.join(BENCH, name), encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def _p3_result(name):
    path = os.path.join(HERE, "results", name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e6e5e1", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(path):  # noqa: PLR0915 -- three panels, one figure
    b2, fd2, a2, b2a = load_two_step()
    a20, b20, fd20 = load_twenty_step()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    fig.patch.set_facecolor("#fcfcfb")

    # (1) two steps: paired differences from (b), per estimator, both sides
    ax = axes[0]
    _style(ax)
    labels, vals, errs, cols = [], [], [], []
    for h in ("0.2", "0.1", "0.05"):
        d = fd2[h] - b2
        labels.append(f"FD h={h}")
        vals.append(d.mean(0) / b2.mean(0) * 100)
        errs.append(1.96 * _se(d) / b2.mean(0) * 100)
        cols.append(C_FD)
    d = a2 - b2a
    labels.append(f"route (a)\n{len(a2)} seeds")
    vals.append(d.mean(0) / b2a.mean(0) * 100)
    errs.append(1.96 * _se(d) / b2a.mean(0) * 100)
    cols.append(C_A)
    x = np.arange(len(labels))
    for i, (v, e, c) in enumerate(zip(vals, errs, cols, strict=True)):
        for side, marker, off in ((0, "o", -0.12), (1, "s", 0.12)):
            ax.errorbar(
                i + off, v[side], yerr=e[side], fmt=marker, ms=5, lw=1.5, capsize=3, color=c
            )
    ax.axhline(0, color=C_B, lw=1.2)
    ax.text(
        0.02,
        0.96,
        "0 = route (b), interpolation-free",
        color=C_B,
        fontsize=8,
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("(estimate - route b) / route b   [%]")
    ax.set_title(
        f"two steps, strong collisions, {len(b2)} seeds\n(○ G_L, □ G_R; bars 95 % t-interval, paired)",
        fontsize=9,
    )

    # (2) twenty steps: route (a) against route (b), seed by seed, on identical trajectories
    ax = axes[1]
    _style(ax)
    n = len(b20)
    for side, marker in ((0, "o"), (1, "s")):
        ax.scatter(b20[:, side], a20[:, side], marker=marker, s=30, color=C_A)
    lo, hi = min(a20.min(), b20.min()) - 0.005, max(a20.max(), b20.max()) + 0.005
    ax.plot([lo, hi], [lo, hi], color=C_PAPER, lw=0.8, ls="--")
    d = a20 - b20
    ax.text(
        0.03,
        0.96,
        f"(a) - (b) paired: {d[:, 0].mean():+.4f} ± {1.96 * _se(d[:, 0]):.4f} (G_L)\n"
        f"                  {d[:, 1].mean():+.4f} ± {1.96 * _se(d[:, 1]):.4f} (G_R)",
        fontsize=8,
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel("route (b), particle-carried")
    ax.set_ylabel("route (a), P3 retained frames, R = 512")
    ax.set_title(
        f"20 steps, N₀ = 2000, {n} seeds: same trajectory, seed by seed\n(○ G_L, □ G_R; dashed: equality)",
        fontsize=9,
    )

    # (3) twenty steps: means with SE, against the paper
    ax = axes[2]
    _style(ax)
    for i, (_name, arr, c) in enumerate(
        (("route (b)", b20, C_B), ("route (a)", a20, C_A), ("FD", fd20, C_FD))
    ):
        for side, marker, off in ((0, "o", -0.1), (1, "s", 0.1)):
            ax.errorbar(
                i + off,
                arr[:, side].mean(),
                yerr=1.96 * _se(arr[:, side]),
                fmt=marker,
                ms=6,
                lw=1.5,
                capsize=3,
                color=c,
            )
    ax.axhline(PAPER[0], color=C_PAPER, lw=0.8, ls="--")
    ax.text(2.45, PAPER[0], "paper\n(N₀=2x10⁶)", fontsize=7, color=C_PAPER, va="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["route (b)", "route (a)\nR=512", "FD h=0.1"], fontsize=8)
    ax.set_ylabel("∂J₃/∂T_normal")
    ax.set_title(f"20 steps: means ± 95 % ({n} seeds)", fontsize=9)
    # (4) the paper's setting, N0 = 16 000: route (b) vs a 1600-seed FD vs the paper
    ax = axes[3]
    _style(ax)
    b16 = np.array([r["b"] for f in ("route_b_N16k.json", "route_b_N16k_b.json") for r in _rows(f)])
    fd16 = np.array([r["fd"] for r in _rows("fd_N16k.json")])
    b64 = np.array([r["b"] for r in _rows("route_b_N64k.json")])
    a2048 = _p3_result("p1_ballistic_R2048.json")
    entries = [
        (f"(b) N₀=16k\n{len(b16)} seeds", b16.mean(0), 1.96 * _se(b16), C_B, "o"),
        (f"(b) N₀=64k\n{len(b64)} seeds", b64.mean(0), 1.96 * _se(b64), C_B, "o"),
        (f"FD N₀=16k\n{len(fd16)} seeds", fd16.mean(0), 1.96 * _se(fd16), C_FD, "o"),
    ]
    if a2048 is not None:
        entries.append(
            (
                "(a) R=2048\n2 seeds",
                np.array(a2048["adjoint_normal_temperature"]["mean"]),
                1.96 * np.array(a2048["adjoint_normal_temperature"]["se"]),
                C_A,
                "o",
            )
        )
    for i, (_label, mean, err, color, _marker) in enumerate(entries):
        for side, mk, off in ((0, "o", -0.12), (1, "s", 0.12)):
            ax.errorbar(
                i + off, mean[side], yerr=err[side], fmt=mk, ms=6, lw=1.5, capsize=3, color=color
            )
    ax.axhline(PAPER[0], color=C_PAPER, lw=0.8, ls="--")
    ax.text(
        len(entries) - 0.5, PAPER[0] + 0.0008, "paper 0.1517", fontsize=7, color=C_PAPER, ha="right"
    )
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], fontsize=7)
    ax.set_ylabel("∂J₃/∂T_normal")
    ax.set_title("the paper's setting (20 steps): means ± 95 %\n(○ G_L, □ G_R)", fontsize=9)

    fig.suptitle(
        "Particle-carried ensemble adjoint (route b) vs P3 retained frames (route a) vs finite differences — same trajectories",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGURES, exist_ok=True)
    print("wrote", figure(os.path.join(FIGURES, "route_b_agreement.png")))
    b2, fd2, a2, b2a = load_two_step()
    print(
        f"two-step: b {b2.mean(0).round(4)} ± {_se(b2).round(4)}; "
        f"FD0.2-b {(fd2['0.2'] - b2).mean(0).round(4)} ± {_se(fd2['0.2'] - b2).round(4)}; "
        f"a-b {(a2 - b2a).mean(0).round(4)} ± {_se(a2 - b2a).round(4)}"
    )
    a20, b20, fd20 = load_twenty_step()
    print(
        f"20-step ({len(b20)} seeds): b {b20.mean(0).round(4)} ± {_se(b20).round(4)}; a {a20.mean(0).round(4)} ± {_se(a20).round(4)}; "
        f"FD {fd20.mean(0).round(4)} ± {_se(fd20).round(4)}; b-FD paired {(b20 - fd20).mean(0).round(4)} ± {_se(b20 - fd20).round(4)}; "
        f"a-FD paired {(a20 - fd20).mean(0).round(4)} ± {_se(a20 - fd20).round(4)}"
    )


if __name__ == "__main__":
    main()
