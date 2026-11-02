"""Fitgate trends (2026-11-01): the M 10 circle's adjoint bias against Knudsen number, time
step, cell size and the derivative-only settings (paired), with the in-situ collision-volume
gate and the wall-probe gate.  Writes examples/mesh/figures/fitgate/trends.png, and
fields.png: the volume gate with the collision field swapped (volf jobs, FD shared per seed).

    PYTHONPATH=. python3 scratch/fitgate/trends.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "scratch/fitgate")
import analyze as A  # noqa: E402

OUT = A.OUT
VARIANTS = {
    "along_pool_plane": "#2a78d6",
    "cell_along_pool_plane": "#eb6834",
    "cellxv_along_pool_plane": "#1baf7a",
    "one_pool_plane": "#6250d6",
}
FD_COLOR = "0.25"
PAIRED = (
    ("c10_spec_h1000", "horizon 1000"),
    ("c10_spec_gz001", "grazing 0.001"),
    ("c10_spec_two", "two-sided fits"),
    ("c10_spec_v816", "virtual 8x16"),
    ("c10_spec_mem", "+ membership"),
)


def mean_se(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def ratio_se(a, fd):
    """adjoint mean / FD mean and its s.e. (independent errors)."""
    ma, sa = mean_se(a)
    mf, sf = mean_se(fd)
    r = ma / mf
    return r, abs(r) * np.hypot(sa / ma, sf / mf)


def series(data, gates, variant, membership=False):
    out = []
    for g in gates:
        rows = data[(g, "base")]
        a = [A.total(r, variant) + (r["terms"][variant].get("membership", 0.0) if membership
             else 0.0) for r in rows]  # fmt: skip
        out.append(
            (
                mean_se(a),
                mean_se([r["fd"][0] for r in rows]),
                ratio_se(a, [r["fd"][0] for r in rows]),
            )
        )
    return out


def sweep_panel(ax, data, gates, xs, xlabel, title, absolute):
    for j, (v, c) in enumerate(VARIANTS.items()):
        s = series(data, gates, v)
        xv = np.array(xs) * (1 + (j - 1.5) * 0.03)
        if absolute:
            ax.errorbar(xv, [p[0][0] for p in s], yerr=[2 * p[0][1] for p in s], fmt="o-", ms=5,
                        lw=1.2, color=c, label=v)  # fmt: skip
        else:
            ax.errorbar(xv, [p[2][0] for p in s], yerr=[2 * p[2][1] for p in s], fmt="o-", ms=5,
                        lw=1.2, color=c, label=v)  # fmt: skip
    if absolute:
        s = series(data, gates, "along_pool_plane")
        ax.errorbar(xs, [p[1][0] for p in s], yerr=[2 * p[1][1] for p in s], fmt="s--", ms=6,
                    lw=1.4, color=FD_COLOR, label="central difference", zorder=4)  # fmt: skip
        ax.set_ylabel("dD/dR (absolute)", fontsize=8)
    else:
        ax.axhline(1.0, color="0.45", lw=1)
        ax.set_ylabel("adjoint / central difference", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(color="0.93", lw=0.6)
    ax.tick_params(labelsize=7)


def paired_panel(ax, data):
    """(setting - baseline) adjoint per shared seed, over the baseline FD."""
    base = {r["seed"]: r for r in data[("c10_spec", "base")]}
    fd = np.mean([r["fd"][0] for r in data[("c10_spec", "base")]])
    y = np.arange(len(PAIRED))
    for j, (v, c) in enumerate(VARIANTS.items()):
        m, e = [], []
        for g, _ in PAIRED:
            rows = data[(g, "base")]
            d = [A.total(r, v) + r["terms"][v].get("membership", 0.0) - A.total(base[r["seed"]], v)
                 for r in rows]  # fmt: skip
            mm, ss = mean_se(np.array(d) / fd)
            m.append(mm)
            e.append(2 * ss)
        ax.errorbar(m, y + (j - 1.5) * 0.15, xerr=e, fmt="o", ms=5, lw=1.2, color=c)
    ax.axvline(0.0, color="0.45", lw=1)
    base_bias = (
        np.mean([A.total(r, "along_pool_plane") for r in data[("c10_spec", "base")]]) / fd - 1
    )
    ax.axvline(-base_bias, color="#e87ba4", lw=1.4, ls="--")
    ax.annotate("shift needed to remove\nalong_pool_plane's bias", (-base_bias, len(PAIRED) - 0.6),
                fontsize=7, color="0.3", ha="left", xytext=(4, 0), textcoords="offset points")  # fmt: skip
    ax.set_yticks(y, [t for _, t in PAIRED], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired change of the adjoint / FD (same seeds, same FD)", fontsize=8)
    ax.set_title("circle M10 Kn0.5 specular: derivative-only settings (8 seeds)", fontsize=9)
    ax.grid(axis="x", color="0.93", lw=0.6)
    ax.tick_params(labelsize=7)


def volume_rows(gate):
    paths = sorted(A.RESULTS.glob(f"volume_{gate}_s*.jsonl"))
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    return [r for r in rows if not r.get("smoke")]


def volume_panel(ax):
    labels, k = [], 0
    for gate, name in (
        ("c10_spec", "M10 spec"),
        ("c10_diff", "M10 diff"),
        ("sq_spec", "squash spec"),
    ):
        rows = volume_rows(gate)
        if not rows:
            continue
        for cells in ("all", "cut"):
            pred = [r[f"pred_{cells}"] for r in rows]
            for i, (eps, mk) in enumerate(zip(rows[0]["eps"], "os", strict=True)):
                r, se = ratio_se(pred, [row[f"fd_{cells}"][i] for row in rows])
                ax.errorbar(np.clip(r, 0.0, 2.5), k + 0.15 * (i - 0.5), xerr=2 * se, fmt=mk,
                            ms=5, color="#2a78d6" if i == 0 else "#eb6834", lw=1.2,
                            label=f"eps = {eps}" if k == 0 else None)  # fmt: skip
            labels.append(f"{name}, {cells} cells (n = {len(rows)})")
            k += 1
    ax.axvline(1.0, color="0.45", lw=1)
    ax.set_xlim(0.0, 2.5)
    ax.set_yticks(range(k), labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("predicted / FD of the collision-volume scaling (±2 s.e.)", fontsize=8)
    ax.set_title("collision pull-back in situ (volume gate)", fontsize=9)
    ax.grid(axis="x", color="0.93", lw=0.6)
    ax.tick_params(labelsize=7)


FIELDS = ("base", "noguard", "guard8", "cubic", "quad", "pool1", "knn")


def field_rows(gate, field):
    tag = gate if field == "base" else f"{gate}_{field}"
    return {r["seed"]: r for r in volume_rows(tag) if r.get("field", "base") == field}


def cell_part(r, key, cells):
    """A row's pred or FD (FD: mean of its eps) on all cells or all minus the cut cells."""
    out = np.mean(r[f"{key}_all"])
    return out if cells == "all" else out - np.mean(r[f"{key}_cut"])


def field_panel(ax, gate, title):
    """pred / FD per collision field, on the seeds each field ran (the FD's seeds' mean)."""
    base = field_rows(gate, "base")
    labels = []
    for k, field in enumerate(FIELDS):
        rows = field_rows(gate, field)
        seeds = sorted(set(rows) & set(base))
        for i, (cells, mk, c) in enumerate((("all", "o", "#2a78d6"), ("noncut", "s", "#eb6834"))):
            if len(seeds) < 2:
                continue
            r, se = ratio_se([cell_part(rows[s], "pred", cells) for s in seeds],
                             [cell_part(base[s], "fd", cells) for s in seeds])  # fmt: skip
            ax.errorbar(r, k + 0.15 * (i - 0.5), xerr=2 * se, fmt=mk, ms=5, color=c, lw=1.2,
                        label=f"{cells} cells" if k == 0 else None)  # fmt: skip
        labels.append(f"{field} (n = {len(seeds)})")
    ax.axvline(1.0, color="0.45", lw=1)
    ax.set_xlim(0.6, 2.0)
    ax.set_yticks(range(len(FIELDS)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("predicted / FD of the collision-volume scaling (±2 s.e.; FD of both eps)",
                  fontsize=8)  # fmt: skip
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", color="0.93", lw=0.6)
    ax.tick_params(labelsize=7)


def fields_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    field_panel(axes[0], "c10_spec", "M10 Kn0.5 specular: collision field swapped")
    field_panel(axes[1], "c10_diff", "M10 Kn0.5 diffuse: collision field swapped")
    fig.tight_layout()
    fig.savefig(OUT / "fields.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/fields.png")


def probe_rows(tag):
    paths = sorted(A.RESULTS.glob(f"probe_{tag}_s*.jsonl"))
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    return [r for r in rows if not r.get("smoke")]


def probe_panel(ax):
    labels, k = [], 0
    for tag, name in (("c10_spec", "Kn0.5"), ("c10_spec_kn10", "Kn10")):
        rows = probe_rows(tag)
        if not rows:
            continue
        for probe in ("xn", "vs", "vr"):
            fd = np.array([r["fd"][probe] for r in rows])
            for i in range(fd.shape[1]):
                m, s = mean_se(fd[:, i])
                ax.errorbar(m, k - 0.2 + 0.1 * i, xerr=2 * s, fmt="s", ms=4, color=FD_COLOR, lw=1,
                            label="FD (each estimate)" if k == 0 and i == 0 else None)  # fmt: skip
            for j, (v, c) in enumerate(VARIANTS.items()):
                m, s = mean_se([r["pred"][v][probe] for r in rows])
                ax.errorbar(m, k + 0.12 + 0.06 * j, xerr=2 * s, fmt="o", ms=4, color=c, lw=1)
            labels.append(f"{name}: {probe} (n = {len(rows)})")
            k += 1
    ax.axvline(0.0, color="0.45", lw=1)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yticks(range(k), labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("dJ/deps (symlog)", fontsize=8)
    ax.set_title("wall-probe gate: hit particles' post-move state perturbed", fontsize=9)
    ax.grid(axis="x", color="0.93", lw=0.6)
    ax.tick_params(labelsize=7)


def main():
    data = A.load()
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    sweep_panel(axes[0, 0], data, ("c10_spec_kn10", "c10_spec_kn2", "c10_spec"), [0.1, 0.5, 2.0],
                "1 / Kn", "circle M10 specular: bias against collisionality (16-32 seeds)", False)  # fmt: skip
    sweep_panel(axes[0, 1], data, ("c10_spec_dt2", "c10_spec", "c10_spec_2dt"), [0.5, 1.0, 2.0],
                "dt / dt_0", "circle M10 Kn0.5 specular: time step", True)  # fmt: skip
    sweep_panel(axes[0, 2], data, ("c10_spec", "c10_spec_g2n", "c10_spec_g4n"), [1.0, 0.5, 0.25],
                "cell size / dx_0 (same N per cell)", "circle M10 Kn0.5 specular: cell size", True)  # fmt: skip
    axes[0, 2].invert_xaxis()
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 1].legend(fontsize=7, loc="lower left")
    paired_panel(axes[1, 0], data)
    volume_panel(axes[1, 1])
    probe_panel(axes[1, 2])
    fig.suptitle("M 10 circle shape-adjoint bias: what it does and does not depend on "
                 "(seed means ±2 s.e.)", fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "trends.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/trends.png")
    fields_figure()


if __name__ == "__main__":
    main()
