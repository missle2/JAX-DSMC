"""Round-1 figures (2026-11-11): the coarse baselines (``lcbase``), the shock profiles behind
them, the window-gradient gate (``lcsnr``) and the 8x specular circle.

    PYTHONPATH=. python3 -m examples.learned_collision.plot_round1

Writes ``examples/learned_collision/figures/round1_*.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.learned_collision.shear_wave import RESULTS

FIG = Path(__file__).resolve().parent / "figures"
OPS = ("ntc", "ntc_adjacent", "esfp", "esfp_recon", "bgk", "esbgk", "bgk_recon", "esbgk_recon")
LABEL = {
    "ntc": "NTC",
    "ntc_adjacent": "NTC adjacent",
    "esfp": "ES-FP",
    "esfp_recon": "ES-FP + recon",
    "bgk": "BGK",
    "esbgk": "ES-BGK",
    "bgk_recon": "BGK + recon",
    "esbgk_recon": "ES-BGK + recon",
}
COLOR = dict(
    zip(
        OPS,
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
        strict=True,
    )
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _rows():
    rows = []
    for tag in ("base_shear", "base_shockA", "base_shockB"):
        rows += json.loads((RESULTS / f"eval_{tag}.json").read_text())["rows"]
    return rows


def _dots(ax, rows, cases, key, se_key, ylabel, ref=None):
    width = 0.8 / len(OPS)
    for j, op in enumerate(OPS):
        xs, ys, es = [], [], []
        for i, case in enumerate(cases):
            r = next(r for r in rows if r["case"] == case and r["operator"] == op)
            xs.append(i - 0.4 + (j + 0.5) * width)
            ys.append(r[key])
            es.append(r[se_key] if se_key else 0.0)
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            fmt="o",
            ms=5,
            color=COLOR[op],
            ecolor=COLOR[op],
            elinewidth=1.2,
            capsize=0,
            label=LABEL[op],
            mec="white",
            mew=0.8,
        )
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1.0)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"{c}\n({_role(rows, c)})" for c in cases])
    ax.set_ylabel(ylabel, color=INK)
    _style(ax)


def _role(rows, case):
    return next(r["role"] for r in rows if r["case"] == case)


def baselines(rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    shear = ("S1", "S2", "S3", "S4")
    k1 = ("K1", "K3", "K5", "K7")
    k4 = ("K2", "K4", "K6", "K8")
    _dots(axes[0, 0], rows, shear, "mu_ratio", "mu_ratio_se", r"$\mu_{\rm eff}/\mu_{\rm ref}$", 1.0)
    axes[0, 0].set_title("Shear waves, dx = 2λ (8x): effective viscosity", loc="left", color=INK)
    axes[0, 0].set_ylim(0.6, 2.2)
    _dots(axes[0, 1], rows, k1 + k4, "l2_txx", "l2_txx_se", r"RMS $\tau_{xx}$ error / peak")
    axes[0, 1].axvline(3.5, color=MUTED, lw=0.8, ls=":")
    axes[0, 1].set_title(
        "Shocks: normal-stress profile error (left dx = λ₁, right 4λ₁)", loc="left", color=INK
    )
    _dots(
        axes[1, 0],
        rows,
        k1 + k4,
        "int_txx_ratio",
        "int_txx_ratio_se",
        r"$\int\tau_{xx}dx$ / reference",
        1.0,
    )
    axes[1, 0].axvline(3.5, color=MUTED, lw=0.8, ls=":")
    axes[1, 0].set_title(
        "Shocks: integrated normal stress (shift-invariant)", loc="left", color=INK
    )
    _dots(
        axes[1, 1],
        rows,
        k1 + k4,
        "int_qx_ratio",
        "int_qx_ratio_se",
        r"$\int q_x dx$ / reference",
        1.0,
    )
    axes[1, 1].axvline(3.5, color=MUTED, lw=0.8, ls=":")
    axes[1, 1].set_title(
        "Shocks: integrated heat flux (BGK's Pr = 1 -> -35 %)", loc="left", color=INK
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "round1_baselines.png", dpi=160)
    plt.close(fig)


def profiles(rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col")
    show = ("ntc", "ntc_adjacent", "esfp_recon")
    for col, case in enumerate(("K8", "K2")):
        ref_row = next(r for r in rows if r["case"] == case and r["operator"] == "ntc")
        tgt = np.array(ref_row["reference"])
        dx = ref_row["dx_c"]
        x = (np.arange(tgt.shape[0]) + 0.5) * dx - ref_row["x_shock"]
        for row, (field, name) in enumerate(
            ((2, r"$\tau_{xx}/(\rho v_{th}^2)$ per mass"), (3, r"$q_x$ per mass / $v_{th}^3$"))
        ):
            ax = axes[row, col]
            ax.step(x, tgt[:, field], where="mid", color=INK, lw=2.0, label="fine NTC (binned)")
            for op in show:
                r = next(r for r in rows if r["case"] == case and r["operator"] == op)
                prof = np.array(r["profile"])
                xo = (np.arange(prof.shape[0]) + 0.5) * dx - r["x_shock"]
                ax.plot(
                    xo,
                    prof[:, field],
                    "o-",
                    ms=5,
                    lw=1.4,
                    color=COLOR[op],
                    label=LABEL[op],
                    mec="white",
                    mew=0.8,
                )
            ax.set_xlim(-20, 25)
            ax.set_ylabel(name, color=INK)
            _style(ax)
            if row == 0:
                mach = "10" if case == "K8" else "1.5"
                ax.set_title(
                    f"{case}: M {mach}, dx = 4λ₁ (training case)"
                    if case == "K8"
                    else f"{case}: M {mach}, dx = 4λ₁ (held out)",
                    loc="left",
                    color=INK,
                )
        axes[1, col].set_xlabel(r"$(x - x_{\rm shock})/\lambda_1$", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "round1_shock_profiles.png", dpi=160)
    plt.close(fig)


def snr():
    d = json.loads((RESULTS / "grad_snr2d_v1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    est_color = {"pair": "#2a78d6", "bias": "#eb6834"}
    for ax, leaf, name in (
        (axes[0], "log_tau", r"$\partial L/\partial \log\tau$"),
        (axes[1], "log_pr", r"$\partial L/\partial \log Pr$"),
    ):
        for case, ls, mk in (("S2", "-", "o"), ("K1", "--", "s")):
            rows = [r for r in d["rows"] if r["case"] == case]
            blocks = [r["blocks"] for r in rows]
            for est in ("pair", "bias"):
                ax.plot(
                    blocks,
                    [r[f"{est}_{leaf}_snr"][0] for r in rows],
                    linestyle=ls,
                    marker=mk,
                    ms=7,
                    lw=2,
                    color=est_color[est],
                    label=f"{case} ({'shear' if case == 'S2' else 'shock'}) {est}",
                    mec="white",
                    mew=0.8,
                )
        ax.axhline(2.0, color=MUTED, lw=1, ls=":")
        ax.text(10.1, 2.05, r"$2\sigma$", color=MUTED, fontsize=9)
        ax.set_xlabel("window length [sample blocks of 20 steps]", color=INK)
        ax.set_ylabel("|mean| / s.e. over 64 windows", color=INK)
        ax.set_title(f"{name} at θ = 0", loc="left", color=INK)
        ax.set_xticks([2, 5, 10])
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, ncol=2, handlelength=3.5)
    fig.tight_layout()
    fig.savefig(FIG / "round1_grad_snr.png", dpi=160)
    plt.close(fig)


def circle():
    fine = json.loads((RESULTS / "circle_ntc_s8_v1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
    x = np.array(fine["x"])
    for ax, key, name in (
        (axes[0], "stagnation_n", r"$n/n_\infty$"),
        (axes[1], "stagnation_T", r"$T/T_\infty$"),
    ):
        ax.step(x, fine[key], where="mid", color=INK, lw=2, label="fine NTC (on 0.5 m cells)")
        for op in ("ntc", "esfp_recon", "esbgk_recon"):
            c = json.loads((RESULTS / f"circle_{op}_s1_v1.json").read_text())
            ax.plot(
                c["x"],
                c[key],
                "o-",
                ms=5,
                lw=1.4,
                color=COLOR[op],
                label=f"{LABEL[op]} (8x, drag {100 * (c['drag'] / fine['drag'] - 1):+.1f} %)",
                mec="white",
                mew=0.8,
            )
        ax.set_xlim(-6, 2.5)
        ax.set_ylabel(name, color=INK)
        ax.set_xlabel("x [m] along the stagnation line (nose at x = 2 m)", color=INK)
        _style(ax)
    axes[0].set_title("M 10 circle, Kn_D 0.042, specular wall", loc="left", color=INK)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "round1_circle_v1.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rows = _rows()
    baselines(rows)
    profiles(rows)
    snr()
    circle()
    print("wrote", sorted(p.name for p in FIG.glob("round1_*.png")))


if __name__ == "__main__":
    main()
