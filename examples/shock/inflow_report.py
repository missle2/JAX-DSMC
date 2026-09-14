"""Run the two-sided-inflow shock and report it (spec section 9.3).

    python3 -m examples.shock.inflow_report            # run, log, plot
    python3 -m examples.shock.inflow_report --plot-only # re-plot from the saved log

Writes ``results/inflow_driven.log`` (a parseable table plus the profile arrays) and
``figures/inflow_driven.png``.  As elsewhere in this repo the figure is built by parsing
the log, so it cannot drift from the run that produced it.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from examples.shock.inflow_driven import Case, profile
from jax_dsmc.viz import (
    INK_SECONDARY,
    REFERENCE,
    SERIES_ANALYTIC,
    SERIES_MEASURED,
    make_figure,
    savefig,
)

RESULTS = "examples/shock/results"
FIGURES = "examples/shock/figures"
MOMENTS = ("nrho1", "temp1", "ux1", "nrho2", "temp2", "ux2")
PRETTY = {
    "nrho1": "nrho upstream",
    "temp1": "T upstream",
    "ux1": "ux upstream",
    "nrho2": "nrho downstream",
    "temp2": "T downstream",
    "ux2": "ux downstream",
}


def run(case: Case, warm, sample, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    r = profile(case, n_warm=warm, n_sample=sample)
    np.savez(
        os.path.join(out_dir, "inflow_driven.npz"),
        x=r["x"],
        nrho=r["nrho"],
        ux=r["ux"],
        temp=r["temp"],
        mfp=r["mfp"],
        length=r["length"],
        **{f"m_{k}": r["measured"][k] for k in MOMENTS},
        **{f"t_{k}": r["target"][k] for k in MOMENTS},
    )
    lines = [
        f"# mach={case.mach} n_cells={case.n_cells} npc={case.npc} "
        f"gap_mfp={case.gap_mfp} warm={warm} sample={sample}",
        f"# shock_frac={r['shock_frac']:.4f} downstream_mfp={r['downstream_mfp']:.2f} "
        f"flux_ratio={r['flux_ratio']:.5f} shortfall={r['shortfall']:.3f} "
        f"clipped={r['clipped']}",
        "# moment measured target err_pct",
    ]
    for k in MOMENTS:
        meas, targ = r["measured"][k], r["target"][k]
        lines.append(f"{k} {meas:.6e} {targ:.6e} {100.0 * (meas - targ) / targ:+.4f}")
    path = os.path.join(out_dir, "inflow_driven.log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse(out_dir=RESULTS):
    path = os.path.join(out_dir, "inflow_driven.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python3 -m examples.shock.inflow_report`")
    meta, rows = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                continue
            name, meas, targ, err = line.split()
            rows[name] = {"measured": float(meas), "target": float(targ), "err": float(err)}
    if len(rows) != len(MOMENTS):
        raise ValueError(f"{path}: expected {len(MOMENTS)} moments, got {len(rows)}")
    return meta, rows


def table(meta, rows):
    print("\n=== two-sided-inflow normal shock vs Rankine-Hugoniot ==================")
    print("  Upstream rows test the INFLOW BOUNDARY CONDITION -- does it deliver the")
    print("  state it was asked for.  Downstream rows test the INTERIOR PHYSICS --")
    print("  does the solver reproduce Rankine-Hugoniot from that inlet.\n")
    print(f"  {'moment':>18} {'measured':>13} {'target':>13} {'err':>9}")
    worst = 0.0
    for k in MOMENTS:
        r = rows[k]
        worst = max(worst, abs(r["err"]))
        print(f"  {PRETTY[k]:>18} {r['measured']:>13.4e} {r['target']:>13.4e} {r['err']:>8.2f}%")
    print(f"\n  Worst moment: {worst:.2f} %.")
    print(
        f"  Mass flux nrho*ux conserved end to end to "
        f"{abs(100.0 * (float(meta['flux_ratio']) - 1.0)):.2f} %."
    )
    print(
        f"  Shock at x/L = {float(meta['shock_frac']):.3f}, leaving "
        f"{float(meta['downstream_mfp']):.1f} mfp of downstream run."
    )
    print(
        f"  inflow shortfall {meta['shortfall']}, attempts clipped {meta['clipped']} "
        "(both must be zero)."
    )


def figure(meta, rows, path, out_dir=RESULTS):
    npz = os.path.join(out_dir, "inflow_driven.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(npz)
    d = np.load(npz)
    x, mfp = d["x"], float(d["mfp"])
    xn = (x - x[int(float(meta["shock_frac"]) * len(x))]) / mfp

    fig, ax = make_figure(1, 4, figsize=(19.0, 4.3))
    panels = (
        (0, d["nrho"], float(d["t_nrho1"]), float(d["t_nrho2"]), r"$n$  [m$^{-3}$]", "density"),
        (1, d["ux"], float(d["t_ux1"]), float(d["t_ux2"]), r"$u_x$  [m/s]", "velocity"),
        (2, d["temp"], float(d["t_temp1"]), float(d["t_temp2"]), r"$T$  [K]", "temperature"),
    )
    for i, field, up, dn, ylabel, title in panels:
        ax[i].plot(xn, field, "-", color=SERIES_MEASURED, linewidth=1.8, label="DSMC")
        ax[i].axhline(up, color=REFERENCE, linestyle="--", linewidth=1.5, label="Rankine-Hugoniot")
        ax[i].axhline(dn, color=REFERENCE, linestyle="--", linewidth=1.5)
        ax[i].axvline(0.0, color=INK_SECONDARY, linewidth=0.8)
        ax[i].set(xlabel=r"$(x - x_{\rm shock})/\lambda_1$", ylabel=ylabel)
        ax[i].set_title(title, fontsize=11)
        if i == 0:
            ax[i].legend(frameon=False, fontsize=8.5)

    # Mass flux is the strongest internal check: `n u_x` must be constant across a
    # steady shock.  Per-cell scatter is ~3 % at this particle count, which would hide
    # the claim rather than show it, so the running mean goes over the raw trace -- and
    # the quoted number is a plateau-to-plateau comparison, not a per-cell one.
    flux = d["nrho"] * d["ux"]
    flux = flux / flux[: len(flux) // 5].mean()
    win = max(3, len(flux) // 20)
    smooth = np.convolve(flux, np.ones(win) / win, mode="same")
    edge = win // 2 + 1
    ax[3].plot(xn, flux, "-", color=SERIES_ANALYTIC, linewidth=0.7, alpha=0.35, label="per cell")
    ax[3].plot(
        xn[edge:-edge],
        smooth[edge:-edge],
        "-",
        color=SERIES_MEASURED,
        linewidth=2.0,
        label=f"running mean ({win} cells)",
    )
    ax[3].axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.5)
    ax[3].axvline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax[3].set_ylim(0.94, 1.06)
    ax[3].set(xlabel=r"$(x - x_{\rm shock})/\lambda_1$", ylabel=r"$n u_x$  (normalised)")
    ax[3].set_title(
        f"mass flux: plateau-to-plateau {abs(100.0 * (float(meta['flux_ratio']) - 1.0)):.2f}%",
        fontsize=11,
    )
    ax[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return savefig(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.shock.inflow_report", description=__doc__
    )
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--warm", type=int, default=20_000)
    ap.add_argument("--sample", type=int, default=10_000)
    ap.add_argument("--figures", default=FIGURES)
    args = ap.parse_args(argv)

    if not args.plot_only:
        print(run(Case(), args.warm, args.sample))
    meta, rows = parse()
    table(meta, rows)
    os.makedirs(args.figures, exist_ok=True)
    print(f"\n   {figure(meta, rows, os.path.join(args.figures, 'inflow_driven.png'))}")


if __name__ == "__main__":
    main()
