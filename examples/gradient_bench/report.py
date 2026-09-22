"""Tables and figures from the ``gradient_bench`` artifacts.  Reads JSON only.

    python3 -m examples.gradient_bench.report            # every table that has data
    python3 -m examples.gradient_bench.report --plot-only

Every number printed here is what the review document quotes; the document is checked
against this output rather than the other way round.
"""

import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.gradient_bench import FIGURES, RESULTS, studies

# fixed-order categorical palette (validated: dataviz skill, light surface)
SERIES = {
    "paper_b0": "#2a78d6",
    "paper_bpilot": "#eb6834",
    "traj_b0": "#1baf7a",
    "traj_bpilot": "#eda100",
}
ARM_LABEL = {
    "paper_b0": "terminal-label, b=0",
    "paper_bpilot": "terminal-label, b=pilot",
    "traj_b0": "trajectory, b=0",
    "traj_bpilot": "trajectory, b=pilot",
}


def load(name):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stats(art, fd_index=0):
    """Mean/SE of AD and FD (at one step), paired difference, t-interval, SE ratio."""
    rows = art["rows"]
    g = np.array([r["gradient"] for r in rows])
    f = np.array([r["fd"][fd_index] for r in rows])
    n = len(g)
    d = g - f
    se = lambda a: a.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan  # noqa: E731
    crit = student_t.ppf(0.975, n - 1) if n > 1 else np.nan
    out = {
        "n": n,
        "complete": art.get("complete", False),
        "ad": g.mean(),
        "ad_se": se(g),
        "fd": f.mean(),
        "fd_se": se(f),
        "diff": d.mean(),
        "diff_se": se(d),
        "diff_ci": crit * se(d),
        "z": d.mean() / se(d) if n > 1 and se(d) > 0 else np.nan,
        "se_ratio": se(g) / se(f) if n > 1 and se(f) > 0 else np.nan,
        "objective": np.mean([r["objective"] for r in rows]),
        "max_attempts": np.mean([r.get("max_attempts", np.nan) for r in rows]),
    }
    timed = [r for r in rows[1:] if "seconds_gradient" in r]  # seed 0 carries compilation
    if timed:
        out["s_grad"] = np.median([r["seconds_gradient"] for r in timed])
        out["s_fd"] = np.median([r["seconds_fd_pair"] for r in timed])
    return out


def fd_steps(art):
    return art["fd_steps"]


def _fmt(x, digits=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}g}"


def table_arms(prefix, arms=("paper_b0", "paper_bpilot", "traj_b0", "traj_bpilot")):
    """One study's estimator arms against the same-seed CRN finite difference."""
    lines = [
        "| arm | seeds | AD mean ± SE | FD(h=0.1) mean ± SE | AD-FD ± 95% CI | z | SE(AD)/SE(FD) | s/grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    any_data = False
    for arm in arms:
        art = load(f"{prefix}_{arm}")
        if art is None:
            continue
        any_data = True
        s = stats(art)
        lines.append(
            f"| {ARM_LABEL[arm]} | {s['n']}{'' if s['complete'] else '*'} | {_fmt(s['ad'])} ± {_fmt(s['ad_se'], 2)} "
            f"| {_fmt(s['fd'])} ± {_fmt(s['fd_se'], 2)} | {_fmt(s['diff'], 3)} ± {_fmt(s['diff_ci'], 2)} "
            f"| {_fmt(s['z'], 2)} | {_fmt(s['se_ratio'], 3)} | {_fmt(s.get('s_grad'), 2)} |"
        )
    return "\n".join(lines) if any_data else None


def table_fd_plateau(prefix, arm="traj_bpilot"):
    art = load(f"{prefix}_{arm}")
    if art is None:
        return None
    lines = ["| h | FD mean ± SE | AD-FD ± 95% CI | z |", "|---:|---:|---:|---:|"]
    for i, h in enumerate(fd_steps(art)):
        s = stats(art, i)
        lines.append(
            f"| {h} | {_fmt(s['fd'])} ± {_fmt(s['fd_se'], 2)} | {_fmt(s['diff'], 3)} ± {_fmt(s['diff_ci'], 2)} | {_fmt(s['z'], 2)} |"
        )
    return "\n".join(lines)


def scaling_rows():
    rows = []
    for axis, values in (("M", (5, 10, 20, 40)), ("N", (128, 256, 512, 1024))):
        for value in values:
            name = (
                "triad_speed_traj_bpilot"
                if (axis == "N" and value == 256) or (axis == "M" and value == 10)
                else f"scaling_{axis}{value}_traj_bpilot"
            )
            art = load(name)
            if art is None:
                continue
            s = stats(art)
            rows.append({"axis": axis, "value": value, **s})
    return rows


def _slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def table_scaling():
    rows = scaling_rows()
    if not rows:
        return None
    lines = [
        "| axis | value | seeds | SE(AD) | SE(FD) | ratio | AD-FD ± 95% CI | z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['axis']} | {r['value']} | {r['n']} | {_fmt(r['ad_se'], 3)} | {_fmt(r['fd_se'], 3)} | {_fmt(r['se_ratio'], 3)} | {_fmt(r['diff'], 3)} ± {_fmt(r['diff_ci'], 2)} | {_fmt(r['z'], 2)} |"
        )
    for axis in ("M", "N"):
        sub = [r for r in rows if r["axis"] == axis]
        if len(sub) >= 2:
            lines.append(
                f"\nfitted exponents vs {axis}: SE(AD) ∝ {axis}^{_slope([r['value'] for r in sub], [r['ad_se'] for r in sub]):+.2f}, "
                f"SE(FD) ∝ {axis}^{_slope([r['value'] for r in sub], [r['fd_se'] for r in sub]):+.2f}"
            )
    return "\n".join(lines)


def table_shock_baseline(parameter):
    """Reconstruct the (biased) same-trajectory mean baseline from the b=0 and b=1 arms.

    surrogate(b) has gradient g(0) - b * S with S = sum_i w_i dL_i, so per seed
    S = g(0) - g(1) and the old ``baseline="mean"`` estimator is g(0) - J * S.
    """
    b0, b1 = load(f"shockbl_{parameter}_paper_b0"), load(f"shockbl_{parameter}_paper_b1")
    if b0 is None or b1 is None:
        return None
    n = min(len(b0["rows"]), len(b1["rows"]))
    r0, r1 = b0["rows"][:n], b1["rows"][:n]
    assert all(
        a["seed"] == b["seed"] and a["objective"] == b["objective"]
        for a, b in zip(r0, r1, strict=True)
    )
    g0 = np.array([r["gradient"] for r in r0])
    g1 = np.array([r["gradient"] for r in r1])
    j = np.array([r["objective"] for r in r0])
    f = np.array([r["fd"][0] for r in r0])
    s_score = g0 - g1
    g_mean = g0 - j * s_score
    se = lambda a: a.std(ddof=1) / np.sqrt(n)  # noqa: E731
    lines = [
        f"| estimator ({parameter}, {n} seeds) | mean ± SE | vs FD(h=0.1) {f.mean():.4g} ± {se(f):.2g} | z |",
        "|---|---:|---:|---:|",
    ]
    for label, g in (
        ("terminal-label, b=0", g0),
        ("terminal-label, b=mean(φ) [main repo]", g_mean),
    ):
        d = g - f
        lines.append(
            f"| {label} | {g.mean():.4g} ± {se(g):.2g} | {d.mean():+.3g} ± {1.96 * se(d):.2g} | {d.mean() / se(d):.2f} |"
        )
    for arm in ("paper_bpilot", "traj_bpilot"):
        art = load(f"shockbl_{parameter}_{arm}")
        if art is not None:
            s = stats(art)
            lines.append(
                f"| {ARM_LABEL[arm]} | {_fmt(s['ad'])} ± {_fmt(s['ad_se'], 2)} | {_fmt(s['diff'], 3)} ± {_fmt(s['diff_ci'], 2)} | {_fmt(s['z'], 2)} |"
            )
    lines.append(
        f"\nscore-term share removed by the mean baseline: E[J·S]/E[g0-FD-free] — mean(J·S) = {np.mean(j * s_score):.4g}, mean(S) = {s_score.mean():.4g} ± {se(s_score):.2g}"
    )
    return "\n".join(lines)


def table_ensemble():
    """Route (b) ensemble adjoint of the inlet-shock case against its own CRN FD."""
    lines = [
        "| control | seeds | ensemble mean ± SE | FD(h=0.1) mean ± SE | diff ± 95% CI | z | SE ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    any_data = False
    for name in sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "ensemble_*.json"))
    ):
        art = load(name)
        any_data = True
        s = stats(art)
        lines.append(
            f"| {name[9:]} | {s['n']}{'' if s['complete'] else '*'} | {_fmt(s['ad'])} ± {_fmt(s['ad_se'], 2)} "
            f"| {_fmt(s['fd'])} ± {_fmt(s['fd_se'], 2)} | {_fmt(s['diff'], 3)} ± {_fmt(s['diff_ci'], 2)} "
            f"| {_fmt(s['z'], 2)} | {_fmt(s['se_ratio'], 3)} |"
        )
        for i, h in enumerate(fd_steps(art)[1:], 1):
            si = stats(art, i)
            lines.append(
                f"|   vs FD h={h} | | | {_fmt(si['fd'])} ± {_fmt(si['fd_se'], 2)} | {_fmt(si['diff'], 3)} ± {_fmt(si['diff_ci'], 2)} | {_fmt(si['z'], 2)} | |"
            )
    return "\n".join(lines) if any_data else None


def table_majorant():
    lines = [
        "| parameter | majorant | seeds | AD mean ± SE | FD mean ± SE | z | SE ratio | max attempts | s/grad |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    any_data = False
    for parameter in ("speed", "diameter"):
        for majorant in ("thermal", "certified"):
            art = load(f"majorant_{parameter}_{majorant}")
            if art is None:
                continue
            any_data = True
            s = stats(art)
            lines.append(
                f"| {parameter} | {majorant} | {s['n']} | {_fmt(s['ad'])} ± {_fmt(s['ad_se'], 2)} | {_fmt(s['fd'])} ± {_fmt(s['fd_se'], 2)} | {_fmt(s['z'], 2)} | {_fmt(s['se_ratio'], 3)} | {_fmt(s['max_attempts'], 3)} | {_fmt(s.get('s_grad'), 2)} |"
            )
    return "\n".join(lines) if any_data else None


# --- figures ---------------------------------------------------------------


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "text.color": "#0b0b0b",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "legend.frameon": False,
        }
    )
    return plt


def figure_arms(prefixes, path, title):
    """AD-FD with 95 % t-intervals per estimator arm, one panel per case.

    The y-range is clipped to +-0.6 |FD|: the unbaselined trajectory arm's interval is
    typically an order of magnitude wider and is reported by its SE ratio instead.
    """
    panels = [p for p in prefixes if any(load(f"{p}_{a}") for a in ARM_LABEL)]
    if not panels:
        return None
    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(7.0, 3.4 * len(panels) + 1.2), 3.9), squeeze=False
    )
    for ax, prefix in zip(axes[0], panels, strict=True):
        for i, arm in enumerate(ARM_LABEL):
            art = load(f"{prefix}_{arm}")
            if art is None:
                continue
            s = stats(art)
            scale = abs(s["fd"]) if s["fd"] else 1.0
            y, e = s["diff"] / scale, s["diff_ci"] / scale
            ax.errorbar(
                i,
                y,
                yerr=e,
                fmt="o",
                ms=5,
                lw=1.5,
                color=SERIES[arm],
                capsize=3,
                label=ARM_LABEL[arm],
            )
            ax.annotate(
                f"x{s['se_ratio']:.2g}",
                (i, min(max(y, -0.55), 0.55)),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=7,
                color="#52514e",
            )
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.axhspan(-0.05, 0.05, color="#e6e5e1", alpha=0.6, lw=0)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlim(-0.5, len(ARM_LABEL) - 0.5)
        ax.set_xticks(range(len(ARM_LABEL)))
        ax.set_xticklabels(["TL\nb=0", "TL\nb=pilot", "traj\nb=0", "traj\nb=pilot"], fontsize=8)
        ax.set_title(prefix.replace("_", " "), fontsize=9)
    axes[0][0].set_ylabel("(AD - FD) / |FD|")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"{title}: paired mean and 95 % t-interval; xk = SE(AD)/SE(FD); band = +-5 %", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_scaling(path):
    plt = _style()
    rows = scaling_rows()
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, axis, label in zip(
        axes, ("M", "N"), ("timesteps M (N=256)", "particles N (M=10)"), strict=True
    ):
        sub = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["value"])
        if not sub:
            continue
        x = [r["value"] for r in sub]
        for key, name, color in (
            ("ad_se", "trajectory AD, b=pilot", SERIES["traj_bpilot"]),
            ("fd_se", "CRN finite difference", SERIES["paper_b0"]),
        ):
            y = [r[key] for r in sub]
            ax.loglog(
                x, y, "o-", color=color, lw=2, ms=6, label=f"{name} (∝ {axis}^{_slope(x, y):+.2f})"
            )
        ax.set_xlabel(label)
        ax.set_ylabel("standard error of the estimate")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Estimator noise vs horizon and population — Couette ∂J/∂u_wall, 256 seeds", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


ARCHIVES = (
    ("affine_basis", "affine ghost basis (P3's)"),
    ("linearised_step", "dt-linearised collision step"),
)


VARIANT = re.compile(r"_((?:virt\d+x\d+(?:st)?|lin)(?:_dev)?|dev)$")
"""Pull-back variants of the shipped (sequential) estimator, named by ``configure``'s tag."""


CANDIDATE = "virt2x8st_dev"
"""The production candidate: virtual partners, 2 substeps, K = 8, staged nodes, device."""


def _variant_label(tag):
    device = tag.endswith("_dev") or tag == "dev"
    core = tag.replace("_dev", "").replace("dev", "")
    if core.startswith("virt"):
        core = "virtual partners " + core[4:].replace("st", ", staged nodes").replace(
            "x", " substeps, K="
        )
    elif core == "lin":
        core = "dt-linearised"
    else:
        core = "sequential"
    return core + (" (device)" if device else "")


def _ensemble_cases():
    """Every base ``ensemble_*.json`` artifact, with its archived 'before' versions and its
    pull-back variants (``_virt{S}x{K}``, ``_lin``) if any."""
    names = sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RESULTS, "ensemble_*.json"))
    )
    variants = {}
    for name in names:
        m = VARIANT.search(name)
        if m:
            variants.setdefault(name[: m.start()], []).append((m.group(1), load(name)))
    cases = []
    for name in names:
        if VARIANT.search(name):
            continue
        label = (
            name[9:]
            .replace("temperature_left", "T_left")
            .replace("accommodation", "acc")
            .replace("_N", "\nN=")
            .replace("_M", " M=")
            .replace("_eps0.005", "")
            .replace("_eps0", "\nε=0")
            .replace("couette_", "couette\n")
            .replace("relax_", "relax\n")
        )
        before = [(tag, load(f"{folder}/{name}")) for folder, tag in ARCHIVES]
        cases.append(
            (
                label,
                load(name),
                [(t, a) for t, a in before if a is not None],
                variants.get(name, []),
            )
        )
    return cases


ARBITERS = {
    "M10": ("fd_arbiter/relax_diameter_N2000_M10", "FD only, 1800 seeds"),
    "M20": ("fd_arbiter/relax_diameter_N2000_M20", "FD only, 1200 seeds"),
    "M40": ("linearised_step/ensemble_relax_diameter_N2000_M40", "paired FD, 192 seeds"),
    "omega": ("fd_arbiter/relax_omega_N2000_M10", "FD only, 1600 seeds"),
}
"""Independent arbiters of the relaxation ``diameter`` gradient per horizon (all at h = 0.1)."""

SCAN = [
    ("relax_diameter_N2000_M10", "M10", "sequential"),
    ("linearised_step/relax_diameter_N2000_M10", "M10", "Δt-linearised"),
    ("relax_diameter_N2000_M10_virt1x8", "M10", "virtual, post nodes, S=1"),
    ("relax_diameter_N2000_M10_virt2x8", "M10", "virtual, post nodes, S=2"),
    ("relax_diameter_N2000_M10_virt2x8st", "M10", "staged, S=2 (host)"),
    ("relax_diameter_N2000_M10_virt2x8st_dev", "M10", "staged, S=2"),
    ("relax_diameter_N2000_M10_virt4x8st_dev", "M10", "staged, S=4"),
    ("relax_diameter_N2000_M10_virt8x8st_dev", "M10", "staged, S=8"),
    ("relax_diameter_N2000_M10_virt2x2st_dev", "M10", "staged, S=2, K=2"),
    ("relax_diameter_N2000_M20", "M20", "sequential, Δt/2"),
    ("relax_diameter_N2000_M20_virt2x8st_dev", "M20", "staged, S=2, Δt/2"),
    ("relax_diameter_N2000_M40", "M40", "sequential, Δt/4"),
    ("relax_diameter_N2000_M40_virt2x8st_dev", "M40", "staged, S=2, Δt/4"),
    ("relax_omega_N2000_M10", "omega", "omega: sequential"),
    ("linearised_step/relax_omega_N2000_M10", "omega", "omega: Δt-linearised"),
    ("relax_omega_N2000_M10_virt2x8st_dev", "omega", "omega: staged, S=2"),
]


def _arbiter(key):
    name, label = ARBITERS[key]
    art = load(name)
    if art is None:
        return None, None, label
    f = np.array([r["fd"][0] for r in art["rows"]])
    return f.mean(), f.std(ddof=1) / np.sqrt(len(f)), label


def table_pullback_scan():
    lines = [
        "| pull-back (relaxation `diameter`) | seeds | estimate ± SE | per-seed sd | arbiter | bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    any_data = False
    for name, key, label in SCAN:
        art = load(f"ensemble_{name}") if "/" not in name else load(name.replace("/", "/ensemble_"))
        if art is None:
            continue
        any_data = True
        g = np.array([r["gradient"] for r in art["rows"]])
        ref, ref_se, ref_label = _arbiter(key)
        se = g.std(ddof=1) / np.sqrt(len(g))
        bias = 100 * (g.mean() - ref) / abs(ref)
        bias_se = 100 * np.hypot(se, ref_se) / abs(ref)
        lines.append(
            f"| {label} | {len(g)} | {g.mean():.4f} ± {se:.4f} | {g.std(ddof=1):.3f} "
            f"| {ref:.4f} ± {ref_se:.4f} ({ref_label}) | {bias:+.1f} % ({abs(bias) / bias_se:.1f}s) |"
        )
    return "\n".join(lines) if any_data else None


def figure_pullback_scan(path):
    """Every collision pull-back on the relaxation arbiter: bias against an independent FD."""
    rows = []
    for name, key, label in SCAN:
        art = load(f"ensemble_{name}") if "/" not in name else load(name.replace("/", "/ensemble_"))
        if art is None:
            continue
        g = np.array([r["gradient"] for r in art["rows"]])
        ref, ref_se, _ = _arbiter(key)
        se = g.std(ddof=1) / np.sqrt(len(g))
        rows.append(
            (
                label,
                100 * (g.mean() - ref) / abs(ref),
                100 * np.hypot(se, ref_se) / abs(ref),
                g.std(ddof=1),
                key,
            )
        )
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    colors = {
        "M10": SERIES["paper_b0"],
        "M20": SERIES["traj_b0"],
        "M40": SERIES["traj_bpilot"],
        "omega": SERIES["paper_bpilot"],
    }
    x = np.arange(len(rows))
    for i, (_label, bias, err, _sd, key) in enumerate(rows):
        ax.errorbar(i, bias, yerr=1.96 * err, fmt="o", ms=6, capsize=3, lw=1.4, color=colors[key])
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axhspan(-2, 2, color="#e6e5e1", alpha=0.6, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("(estimate - FD arbiter) / |arbiter|   [%]  (95 %)")
    ax.set_title(
        "relaxation ∂⟨v_x²⟩/∂θ vs independent FD (1800 / 1200 / 192 seeds by Δt; omega 1600)",
        fontsize=9,
    )
    handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in colors.values()]
    ax.legend(
        handles,
        ["Δt = 0.2", "Δt = 0.1", "Δt = 0.05", "omega, Δt = 0.2"],
        fontsize=8,
        loc="lower left",
    )
    ax = axes[1]
    ax.bar(x, [r[3] for r in rows], color=[colors[r[4]] for r in rows], width=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("per-seed sd of the estimate")
    ax.set_title("variance", fontsize=9)
    fig.suptitle("Collision pull-backs of the ensemble adjoint: bias and variance", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


RH_CASE = "ensemble_mach_N3200_M300_virt2x8st_dev_rho"
RH_SLOPE = 48.0 / 49.0 * 2.0
"""d(rho_2/rho_1)/dM at M = 2 (gamma = 5/3), per unit of the control multiplying M = 2:
24 M / (M^2 + 3)^2 * 2."""


def table_rh():
    art = load(RH_CASE)
    if art is None:
        return None
    s = stats(art)
    lines = [
        "| quantity | value |",
        "|---|---:|",
        f"| seeds | {s['n']}{'' if s['complete'] else '*'} |",
        f"| J = rho_2/rho_1 at 300 steps (RH stationary 2.286) | {s['objective']:.3f} |",
        f"| ensemble adjoint, virtual staged S=2 K=8 (device) | {s['ad']:.3f} ± {s['ad_se']:.3f} (per-seed sd {np.std([r['gradient'] for r in art['rows']], ddof=1):.2f}) |",
        f"| CRN FD, h = 0.1 | {s['fd']:.3f} ± {s['fd_se']:.3f} |",
        f"| paired difference | {s['diff']:.3f} ± {s['diff_ci']:.3f} (95 %), z = {s['z']:.2f} |",
        f"| Rankine-Hugoniot stationary slope (reference only) | {RH_SLOPE:.3f} |",
    ]
    for i, h in enumerate(fd_steps(art)[1:], 1):
        si = stats(art, i)
        lines.append(
            f"| CRN FD, h = {h} | {si['fd']:.3f} ± {si['fd_se']:.3f} (z = {si['z']:.2f}) |"
        )
    return "\n".join(lines)


def figure_rh(path):
    """The Rankine-Hugoniot acceptance case: per-seed ensemble adjoint vs CRN FD."""
    art = load(RH_CASE)
    if art is None:
        return None
    g = np.array([r["gradient"] for r in art["rows"]])
    f = np.array([r["fd"][0] for r in art["rows"]])
    s = stats(art)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(f, g, s=28, color=SERIES["paper_b0"], label="one seed (paired)")
    lim = [min(f.min(), g.min(), 0) - 0.3, max(f.max(), g.max()) + 0.3]
    ax.plot(lim, lim, color="#52514e", lw=0.8, ls="--")
    ax.errorbar(
        s["fd"],
        s["ad"],
        xerr=1.96 * s["fd_se"],
        yerr=1.96 * s["ad_se"],
        fmt="s",
        ms=9,
        color=SERIES["paper_bpilot"],
        capsize=4,
        lw=1.6,
        label=f"ensemble means ± 95 % ({s['n']} seeds), z = {s['z']:.1f}",
    )
    ax.axhline(RH_SLOPE, color=SERIES["traj_b0"], lw=1.0, ls=":", label="RH stationary slope")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("CRN finite difference, h = 0.1   [d(ρ₂/ρ₁)/dc]")
    ax.set_ylabel("ensemble adjoint   [d(ρ₂/ρ₁)/dc]")
    ax.set_title(
        "Inflow shock, d(ρ₂/ρ₁)/d(upstream Mach): 80 cells x 40, 300 steps\n"
        "(the case whose pathwise derivative had the wrong sign)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_k_stability(path):
    """rms of the carried adjoint over the backward sweep, per partner count K."""
    art = load("k_stability")
    if art is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    cmap = plt.get_cmap("viridis")
    runs = art["runs"]
    for i, run in enumerate(runs):
        color = cmap(i / max(len(runs) - 1, 1))
        rms = np.array(run["rms"])
        steps = np.arange(len(rms))
        axes[0].plot(steps, rms / rms[-1], color=color, lw=1.4, label=f"K = {run['partners']}")
        axes[1].plot(
            steps,
            np.array(run["max"]) / rms[-1],
            color=color,
            lw=1.4,
            label=f"K = {run['partners']}",
        )
    for ax, what in zip(axes, ("rms", "max"), strict=True):
        ax.set_yscale("log")
        ax.axhline(1, color="#52514e", lw=0.8, ls="--")
        ax.invert_xaxis()
        ax.set_xlabel("step k (backward sweep runs right to left)")
        ax.set_ylabel(f"{what} |λ_k| / rms |λ_M|")
        ax.legend(fontsize=8)
    cfg = art["config"]
    fig.suptitle(
        f"Virtual-partner backward stability vs K — inflow shock, {cfg['cells']} cells, "
        f"{art['mean_alive']:.0f} particles (max {art['max_occupancy']}/cell), {cfg['steps']} steps, "
        f"S = {cfg['substeps']}, seed {cfg['seed']}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_ensemble(path):  # noqa: PLR0912, PLR0915 -- three panels, one figure
    """Production ensemble adjoint (route b) vs CRN FD, and its noise against the score routes."""
    cases = [c for c in _ensemble_cases() if c[1] is not None]
    if not cases:
        return None
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6 + 0.9 * len(cases), 4.4),
        gridspec_kw={"width_ratios": [max(3, 0.6 * len(cases)), 2.2, 2]},
    )
    ax = axes[0]
    seen = set()
    for i, (_label, art, before, variants) in enumerate(cases):
        s = stats(art)
        ax.errorbar(
            i,
            100 * s["diff"] / abs(s["fd"]),
            yerr=100 * s["diff_ci"] / abs(s["fd"]),
            fmt="o",
            ms=6,
            lw=1.6,
            capsize=3,
            color=SERIES["paper_b0"],
            label="shipped estimator" if i == 0 else None,
        )
        for j, (tag, art_b) in enumerate(before):
            sb = stats(art_b)
            color = SERIES["paper_bpilot"] if tag.startswith("affine") else SERIES["traj_bpilot"]
            ax.errorbar(
                i + 0.18 + 0.12 * j,
                100 * sb["diff"] / abs(sb["fd"]),
                yerr=100 * sb["diff_ci"] / abs(sb["fd"]),
                fmt="o",
                ms=6,
                lw=1.2,
                capsize=3,
                mfc="none",
                color=color,
                label=f"before: {tag}" if tag not in seen else None,
            )
            seen.add(tag)
        for tag, art_v in variants:
            if tag != CANDIDATE:
                continue  # the scan over the other variants is figure_pullback_scan
            sv = stats(art_v)
            ax.errorbar(
                i - 0.2,
                100 * sv["diff"] / abs(sv["fd"]),
                yerr=100 * sv["diff_ci"] / abs(sv["fd"]),
                fmt="s",
                ms=5,
                lw=1.2,
                capsize=3,
                color=SERIES["paper_bpilot"],
                label=_variant_label(tag) if tag not in seen else None,
            )
            seen.add(tag)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axhspan(-2, 2, color="#e6e5e1", alpha=0.6, lw=0)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[0] for c in cases], fontsize=7)
    ax.set_xlim(-0.6, len(cases) - 0.3)
    ax.set_ylabel("(ensemble adjoint - FD) / |FD|   [%]")
    ax.set_title(
        "production solver, every validated control\n(bars 95 % t-interval, paired with the same seeds' CRN FD)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    # per-seed sd of the sequential form over the candidate's, per case
    ax = axes[1]
    xs, ratios, labels = [], [], []
    for label, art, _before, variants in cases:
        cand = dict(variants).get(CANDIDATE)
        if cand is None:
            continue
        sd = lambda a: np.std([r["gradient"] for r in a["rows"]], ddof=1)  # noqa: E731
        xs.append(len(xs))
        ratios.append(sd(art) / sd(cand))
        labels.append(label.replace("\n", " "))
    ax.bar(xs, ratios, color=SERIES["paper_bpilot"], width=0.8)
    for xi, r in zip(xs, ratios, strict=True):
        ax.text(xi, r * 1.1, f"{r:.2g}", ha="center", fontsize=7)
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("per-seed sd: sequential / virtual (staged, S=2, K=8)")
    ax.set_title("noise reduction of the virtual pull-back", fontsize=9)
    # noise: SE ratio to the FD for each estimator on the two N=128, M=4 inlet controls
    ax = axes[2]
    arms = [
        ("terminal-label", "paper_bpilot", SERIES["paper_b0"]),
        ("trajectory, b=pilot", "traj_bpilot", SERIES["traj_bpilot"]),
    ]
    for j, (control, prefix, ens) in enumerate(
        (
            ("temp", "inlet_temp", "ensemble_temp_N128_M4"),
            ("nrho", "inlet_nrho", "ensemble_nrho_N128_M4"),
        )
    ):
        ratios, colors = [], []
        for _name, arm, color in arms:
            art = load(f"{prefix}_{arm}")
            if art is not None:
                ratios.append(stats(art)["se_ratio"])
                colors.append(color)
        art = load(ens)
        if art is not None:
            ratios.append(stats(art)["se_ratio"])
            colors.append(SERIES["traj_b0"])
        x = j * (len(ratios) + 1) + np.arange(len(ratios))
        ax.bar(x, ratios, color=colors, width=0.8)
        for xi, r in zip(x, ratios, strict=True):
            ax.text(xi, r * 1.15, f"{r:.2g}", ha="center", fontsize=8)
        ax.text(x.mean(), 0.5, control, ha="center", fontsize=9, color="#52514e")
    ax.set_yscale("log")
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_xticks([])
    ax.set_ylabel("SE(estimate) / SE(CRN finite difference)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in arms] + [
        plt.Rectangle((0, 0), 1, 1, color=SERIES["traj_b0"])
    ]
    ax.legend(
        handles, [a[0] for a in arms] + ["ensemble adjoint (route b)"], fontsize=8, loc="upper left"
    )
    fig.suptitle(
        "Ensemble adjoint of the production solver (route b): reservoir, wall and kernel parameters",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_shape_oracle(path):
    """The planar shape oracle: three estimators of dJ/dgap against the analytic value."""
    art = load("shape_planar_oracle")
    if art is None:
        return None
    eps = sorted(art["rows"], key=float, reverse=True)
    x = np.arange(len(eps))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    target = abs(art["analytic_dJ_dgap"])
    ax.axhline(target, color="#52514e", lw=1.0, ls="--", label=f"analytic |dJ/dgap| = {target:.4f}")
    for key, label, color, off in (
        ("frozen", "pathwise, frozen hits (ε = 0)", SERIES["paper_bpilot"], -0.2),
        ("scored", "value-weighted ε-hit score (ensemble)", SERIES["paper_b0"], 0.0),
        ("fd", "ensemble finite difference", SERIES["traj_b0"], 0.2),
    ):
        m = [abs(art["rows"][e][key][0]) for e in eps]
        se = [1.96 * art["rows"][e][key][1] for e in eps]
        ax.errorbar(x + off, m, yerr=se, fmt="o", ms=6, lw=1.5, capsize=3, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {e}" for e in eps])
    ax.set_ylabel("|dJ / dgap|")
    ax.set_ylim(-0.03, target * 1.15)
    ax.set_title(
        f"collisionless planar impulse, {art['seeds']} keys x 20 000 particles\n(bars 95 %)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="center right")
    ax = axes[1]
    j_target = art["analytic_J"]
    ax.axhline(j_target, color="#52514e", lw=1.0, ls="--", label=f"analytic J = {j_target:.4f}")
    ax.errorbar(
        x,
        [abs(art["rows"][e]["J"][0]) for e in eps],
        yerr=[1.96 * art["rows"][e]["J"][1] for e in eps],
        fmt="o",
        ms=6,
        lw=1.5,
        capsize=3,
        color=SERIES["paper_b0"],
        label="J_ε (regularised forward)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {e}" for e in eps])
    ax.set_ylabel("|J|")
    ax.set_title("the regularised objective itself", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Shape derivative: the hit-set term is invisible to frozen-hit pathwise AD and recovered by the value-weighted ε-hit score",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


PATHWISE_CASES = [
    # (artifact suffix, column label, family) -- reparameterised controls first, then the
    # controls that reach J only through discrete decisions
    ("couette_speed_N256_M10_eps0.005", "wall speed\nCouette", "reparameterised"),
    ("couette_temperature_left_N256_M10_eps0.005", "wall T_left\nCouette", "reparameterised"),
    ("temp_N128_M4", "reservoir T\ninlet shock", "reparameterised"),
    ("couette_accommodation_N256_M10_eps0.005", "accommodation\nCouette", "decision"),
    ("nrho_N128_M4", "reservoir n\ninlet shock", "decision"),
    ("relax_diameter_N2000_M10", "σ (VSS diameter)\nrelaxation", "decision"),  # noqa: RUF001
    ("relax_omega_N2000_M10", "ω (VSS exponent)\nrelaxation", "decision"),
]
"""Controls with both a route-(b) ensemble artifact and a ``pathwise_*`` artifact on the
same seeds (``examples.gradient_bench.pathwise``)."""


def _pathwise_join(suffix):
    """Per-seed (CRN FD at h = fd_steps[0], ensemble gradient, pathwise gradient) arrays of
    one control, joined on the seed; ``None`` if either artifact is missing."""
    ens, pw = load(f"ensemble_{suffix}"), load(f"pathwise_{suffix}")
    if ens is None or pw is None:
        return None
    by_seed = {r["seed"]: r for r in pw["rows"]}
    rows = [(r, by_seed[r["seed"]]) for r in ens["rows"] if r["seed"] in by_seed]
    if not rows:
        return None
    fd = np.array([r["fd"][0] for r, _ in rows])
    g = np.array([r["gradient"] for r, _ in rows])
    p = np.array([q["gradient"] for _, q in rows])
    return fd, g, p


def _mean_ci(a):
    n = len(a)
    se = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return a.mean(), se, (student_t.ppf(0.975, n - 1) * se if n > 1 else 0.0)


def table_pathwise():
    lines = [
        "| control | seeds | CRN FD | ensemble adjoint | pathwise (fixed-key AD) | (ens−FD)/FD | (pw−FD)/FD |",  # noqa: RUF001
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suffix, label, _family in PATHWISE_CASES:
        j = _pathwise_join(suffix)
        if j is None:
            continue
        fd, g, p = j
        ref = fd.mean()
        f = lambda a: f"{_mean_ci(a)[0]:.4g} ± {_mean_ci(a)[1]:.2g}"  # noqa: E731
        d = lambda a: (  # noqa: E731
            f"{100 * (a - fd).mean() / abs(ref):+.1f} ± {100 * _mean_ci(a - fd)[2] / abs(ref):.1f} %"
        )
        lines.append(
            f"| {label.replace(chr(10), ', ')} | {len(fd)} | {f(fd)} | {f(g)} | {f(p)} | {d(g)} | {d(p)} |"
        )
    art = load("shape_planar_oracle")
    if art is not None:
        eps = max(art["rows"], key=float)
        row, ref = art["rows"][eps], abs(art["analytic_dJ_dgap"])
        lines.append(
            f"| wall position, planar oracle (ε = {eps}; reference analytic {ref:.4f}) | {art['seeds']} | "
            f"{abs(row['fd'][0]):.4g} ± {row['fd'][1]:.2g} | {abs(row['scored'][0]):.4g} ± {row['scored'][1]:.2g} | "
            f"{abs(row['frozen'][0]):.4g} ± {row['frozen'][1]:.2g} | "
            f"{100 * (abs(row['scored'][0]) - ref) / ref:+.1f} ± {196 * row['scored'][1] / ref:.1f} % | "
            f"{100 * (abs(row['frozen'][0]) - ref) / ref:+.1f} ± {196 * row['frozen'][1] / ref:.1f} % |"
        )
    return "\n".join(lines) if len(lines) > 2 else None


ESTIMATORS = (
    ("fd", "CRN finite difference (same seeds)", SERIES["traj_b0"], "D", -0.22),
    ("ens", "ensemble adjoint (route b)", SERIES["paper_b0"], "o", 0.0),
    ("pw", "pathwise: fixed-key jax.grad", SERIES["paper_bpilot"], "s", 0.22),
)


def figure_pathwise_vs_ensemble(path):  # noqa: PLR0912, PLR0915 -- two panels, one figure
    """Pathwise (fixed-key AD) against the ensemble adjoint and the same seeds' CRN FD on
    every benchmark control, plus the planar shape oracle: the two adjoints agree where the
    control has a reparameterised route, and the pathwise gradient is identically zero
    where the control enters only through discrete decisions."""
    columns = []
    for suffix, label, family in PATHWISE_CASES:
        j = _pathwise_join(suffix)
        if j is not None:
            columns.append((label, family, j))
    if not columns:
        return None
    n_prod = len(columns)
    oracle = load("shape_planar_oracle")
    if oracle is not None:
        eps = max(oracle["rows"], key=float)
        row, ref = oracle["rows"][eps], abs(oracle["analytic_dJ_dgap"])
        columns.append((f"wall position\nplanar oracle, ε={eps}", "decision", (row, ref)))
    fig, axes = plt.subplots(
        1, 2, figsize=(4.2 + 1.05 * len(columns), 4.6), gridspec_kw={"width_ratios": [3.2, 1.3]}
    )
    ax = axes[0]
    for i, (_label, _family, data) in enumerate(columns):
        if i < n_prod:
            fd, g, p = data
            ref = fd.mean()
            series = {"fd": fd, "ens": g, "pw": p}
            for key, label, color, marker, off in ESTIMATORS:
                m, _se, ci = _mean_ci(series[key])
                ax.errorbar(
                    i + off,
                    m / ref,
                    yerr=ci / abs(ref),
                    fmt=marker,
                    ms=6,
                    lw=1.5,
                    capsize=3,
                    color=color,
                    mec="#fcfcfb",
                    mew=0.6,
                    label=label if i == 0 else None,
                )
            if p.std(ddof=1) == 0:
                ax.annotate(
                    "≡ 0" if p.mean() == 0 else f"{p.mean() / ref:.2f}",
                    (i + 0.22, p.mean() / ref),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#52514e",
                )
        else:
            row, ref = data
            for key, (_name, _lab, color, marker, off) in zip(
                ("fd", "scored", "frozen"), ESTIMATORS, strict=True
            ):
                ax.errorbar(
                    i + off,
                    abs(row[key][0]) / ref,
                    yerr=1.96 * row[key][1] / ref,
                    fmt=marker,
                    ms=6,
                    lw=1.5,
                    capsize=3,
                    color=color,
                    mec="#fcfcfb",
                    mew=0.6,
                )
            ax.annotate(
                "≡ 0",
                (i + 0.22, 0.0),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#52514e",
            )
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.axhline(0, color="#52514e", lw=0.8)
    n_rep = sum(1 for c in columns if c[1] == "reparameterised")
    ax.axvspan(n_rep - 0.5, len(columns) - 0.5, color="#e6e5e1", alpha=0.5, lw=0)
    ax.text(
        (n_rep - 1) / 2,
        1.42,
        "control enters through a\nreparameterised draw",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#52514e",
    )
    ax.text(
        (n_rep + len(columns) - 1) / 2,
        1.42,
        "control enters only through discrete decisions\n(accept bit, emitted count, Bernoulli branch, hit set)",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#52514e",
    )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], fontsize=7)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(-0.2, 1.45)
    ax.set_ylabel("dJ/dθ  /  reference")
    ax.set_title(
        "three estimators of one sensitivity on identical realisations (bars 95 %)\n"
        "reference: the same seeds' CRN FD; analytic dJ/dgap for the oracle",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="lower left")
    # noise where both adjoints are valid: SE relative to the CRN FD on the same seeds
    ax = axes[1]
    xs, labels = [], []
    for i, (label, _family, data) in enumerate(columns[:n_prod]):
        fd, g, p = data
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a))  # noqa: E731
        ax.bar(i - 0.2, se(g) / se(fd), width=0.38, color=SERIES["paper_b0"])
        if p.std(ddof=1) > 0:
            ax.bar(i + 0.2, se(p) / se(fd), width=0.38, color=SERIES["paper_bpilot"])
        else:
            ax.text(i + 0.2, 0.06, "≡0", ha="center", fontsize=6.5, color="#52514e")
        xs.append(i)
        labels.append(label.split("\n")[0])
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=60, ha="right")
    ax.set_ylabel("SE(estimator) / SE(CRN FD)")
    ax.set_title("noise (a zero-variance zero\nis not a measurement)", fontsize=8.5)
    fig.suptitle(
        "Pathwise vs ensemble adjoint: identical where the control has a pathwise route, "
        "pathwise ≡ 0 for σ, ω, n, α and the wall position",  # noqa: RUF001
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):  # noqa: PLR0912 -- one table per study
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.gradient_bench.report", description=__doc__
    )
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(FIGURES, exist_ok=True)
    made = [
        figure_arms(
            ["triad_speed", "triad_temperature_left"],
            os.path.join(FIGURES, "estimator_triad.png"),
            "Couette wall controls",
        ),
        figure_arms(
            ["inlet_temp", "inlet_nrho", "inlet2_temp"],
            os.path.join(FIGURES, "estimator_inlet.png"),
            "Inlet-shock reservoir controls",
        ),
        figure_scaling(os.path.join(FIGURES, "trajectory_variance_scaling.png")),
        figure_ensemble(os.path.join(FIGURES, "ensemble_inlet_shock.png")),
        figure_pullback_scan(os.path.join(FIGURES, "pullback_scan.png")),
        figure_rh(os.path.join(FIGURES, "rankine_hugoniot.png")),
        figure_k_stability(os.path.join(FIGURES, "k_stability.png")),
        figure_shape_oracle(os.path.join(FIGURES, "shape_planar_oracle.png")),
        figure_pathwise_vs_ensemble(os.path.join(FIGURES, "pathwise_vs_ensemble.png")),
    ]
    for m in made:
        if m:
            print(f"wrote {m}")
    if args.plot_only:
        return
    for name in studies():
        if name.startswith(("triad", "inlet")):
            t = table_arms(name)
            if t:
                print(f"\n### {name}\n\n{t}")
                plateau = table_fd_plateau(name)
                if plateau:
                    print(f"\nFD step plateau (trajectory, b=pilot):\n\n{plateau}")
    t = table_scaling()
    if t:
        print(f"\n### scaling\n\n{t}")
    for parameter in ("diam", "omega"):
        t = table_shock_baseline(parameter)
        if t:
            print(f"\n### shock baseline bias ({parameter})\n\n{t}")
    t = table_majorant()
    if t:
        print(f"\n### majorant\n\n{t}")
    t = table_ensemble()
    if t:
        print(f"\n### ensemble adjoint (route b) — production solver\n\n{t}")
    t = table_pullback_scan()
    if t:
        print(f"\n### collision pull-back scan (relaxation diameter vs FD arbiters)\n\n{t}")
    t = table_rh()
    if t:
        print(f"\n### Rankine-Hugoniot acceptance case\n\n{t}")
    t = table_pathwise()
    if t:
        print(f"\n### pathwise vs ensemble adjoint (same seeds)\n\n{t}")


if __name__ == "__main__":
    main()
