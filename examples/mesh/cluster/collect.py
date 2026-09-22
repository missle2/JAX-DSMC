"""Collect the cluster study's artifacts into the convergence figures and a table.

    python3 -m examples.mesh.cluster.collect

* ``figures/optimize_convergence.png``: drag vs iterate and vs cost (``S x iterate``
  seeds evaluated) for every ``(S, optimiser)`` run, area check.
* ``figures/gradient_seed_convergence.png``: the ``cos 2 theta`` gradient's standard error
  vs number of seeds (bootstrap over the 32-seed artifact) against the CRN FD.
* the sphere gate row (ensemble vs FD, z) if its artifact exists.
The per-seed files of the seed jobs (``..._dev_seed{i}.json``) are merged into the usual
artifacts first.  Everything comes from the artifacts; nothing is typed in.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
RESULTS_OPT = os.path.join(MESH, "results", "optimize")
FIGURES = os.path.join(MESH, "figures")
GRADIENT_RESULTS = os.path.join(os.path.dirname(MESH), "gradient_bench", "results")
SEED_START = 32


def merge_seed_artifacts() -> dict[str, int]:
    """The per-seed files of the ``sphere`` / ``mode2`` groups (``<artifact>_seed{i}.json``,
    one row each with ``seed = 0``) -> the usual artifact ``<artifact>.json`` with the rows
    renumbered by their file index; rewritten from the per-seed files every time.  Returns
    ``{artifact: seeds merged}``."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(GRADIENT_RESULTS, "ensemble_mesh_*_dev_seed*.json"))):
        stem, _, idx = os.path.basename(path)[: -len(".json")].rpartition("_seed")
        target = os.path.join(GRADIENT_RESULTS, stem + ".json")
        with open(path) as fh:
            d = json.load(fh)
        if not d["rows"]:
            continue
        row = dict(d["rows"][0], seed=int(idx))
        if target in merged:
            with open(target) as fh:
                combined = json.load(fh)
        else:
            combined = dict(d, rows=[])
            combined["config"]["seed_start"] = SEED_START
        combined["rows"] = [r for r in combined["rows"] if r["seed"] != row["seed"]] + [row]
        combined["rows"].sort(key=lambda r: r["seed"])
        with open(target, "w") as fh:
            json.dump(combined, fh, indent=1)
        merged[target] = len(combined["rows"])
    return {os.path.basename(k): v for k, v in merged.items()}


def load_histories():
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_OPT, "circle_fine_S*_*", "history.json"))):
        with open(path) as fh:
            h = json.load(fh)
        tag = os.path.basename(os.path.dirname(path))
        out[tag] = h
    return out


def figure_optimize(hist, path_fig):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for h in hist.values():
        rows = h["rows"]
        s = h["config"]["seeds_per_iterate"]
        opt = h["config"]["optimizer"]
        it = np.array([r["iterate"] for r in rows])
        j = np.array([r["objective"] for r in rows])
        se = np.array(
            [r["objective_se"] if r["objective_se"] == r["objective_se"] else 0.0 for r in rows]
        )
        ls = "-" if opt == "gd" else "--"
        axes[0].errorbar(it, j, yerr=se, fmt="o" + ls, ms=3, capsize=2, label=f"S={s} {opt}")
        axes[1].plot((it + 1) * s, j, "o" + ls, ms=3, label=f"S={s} {opt}")
        axes[2].plot(
            it,
            np.array([r["area"] for r in rows]) / h["area_target"] - 1.0,
            ls,
            label=f"S={s} {opt}",
        )
    axes[0].set_xlabel("iterate")
    axes[0].set_ylabel("windowed drag J (N/m)")
    axes[0].set_title("per iterate (CRN seeds; error bar = seed spread / sqrt S)", fontsize=8)
    axes[1].set_xlabel("seeds evaluated (S x iterates)")
    axes[1].set_ylabel("J")
    axes[1].set_title("per unit cost", fontsize=8)
    axes[2].set_xlabel("iterate")
    axes[2].set_ylabel("area / target - 1")
    axes[2].set_title("constraint", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)


def gradient_seed_convergence(path_fig):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    paths = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_circle_levelset*_mode2_*grid2*_dev.json")
        )
    )
    if not paths:
        return None
    with open(paths[0]) as fh:
        d = json.load(fh)
    g = np.array([r["gradient"] for r in d["rows"]])
    fd = np.array([r["fd"][0] for r in d["rows"]])
    n = len(g)
    rng = np.random.default_rng(0)
    counts = [c for c in (1, 2, 4, 8, 16, 32) if c <= n]
    se = []
    for c in counts:
        means = [g[rng.choice(n, c, replace=True)].mean() for _ in range(2000)]
        se.append(float(np.std(means)))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(counts, se, "o-", label="SE of the ensemble mean (bootstrap)")
    ax.loglog(counts, se[0] / np.sqrt(np.array(counts)), color="0.5", lw=1, label="1 / sqrt(S)")
    ax.axhline(abs(fd.mean()) * 0.01, color="tab:red", ls="--", lw=1, label="1 % of |FD|")
    ax.set_xlabel("seeds S")
    ax.set_ylabel("standard error of dJ/dθ")
    ax.set_title(
        f"cos 2θ mode, {n} seeds: ensemble {g.mean():.3e} ± {g.std(ddof=1) / math.sqrt(n):.1e}, FD {fd.mean():.3e}",
        fontsize=7,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    return {
        "seeds": n,
        "counts": counts,
        "se": se,
        "ensemble": float(g.mean()),
        "fd": float(fd.mean()),
        "artifact": os.path.basename(paths[0]),
    }


def sphere_gate():
    paths = sorted(
        glob.glob(
            os.path.join(GRADIENT_RESULTS, "ensemble_mesh_sphere_levelset*_radius_*_dev.json")
        )
    )
    if not paths:
        return None
    with open(paths[0]) as fh:
        d = json.load(fh)
    g = np.array([r["gradient"] for r in d["rows"]])
    fd = np.array([r["fd"][0] for r in d["rows"]])
    n = len(g)
    z = (
        (g.mean() - fd.mean()) / math.sqrt((g.var(ddof=1) + fd.var(ddof=1)) / n)
        if n > 1
        else float("nan")
    )
    return {
        "seeds": n,
        "ensemble": float(g.mean()),
        "ensemble_se": float(g.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "fd": float(fd.mean()),
        "fd_se": float(fd.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "z": float(z),
        "artifact": os.path.basename(paths[0]),
    }


def main():
    os.makedirs(FIGURES, exist_ok=True)
    report = {}
    merged = merge_seed_artifacts()
    for name, n in merged.items():
        print(f"merged {n} seed files -> {name}")
    report["merged"] = merged
    hist = load_histories()
    if hist:
        figure_optimize(hist, os.path.join(FIGURES, "optimize_convergence.png"))
        report["optimize"] = {
            tag: {
                "seeds_per_iterate": h["config"]["seeds_per_iterate"],
                "optimizer": h["config"]["optimizer"],
                "iterates": len(h["rows"]),
                "J_first": h["rows"][0]["objective"],
                "J_last": h["rows"][-1]["objective"],
                "seconds": sum(r["seconds"] for r in h["rows"]),
            }
            for tag, h in hist.items()
        }
        print("| run | S | optimiser | iterates | J first | J last | change | GPU s |")
        print("|---|---|---|---|---|---|---|---|")
        for tag, r in report["optimize"].items():
            print(
                f"| {tag} | {r['seeds_per_iterate']} | {r['optimizer']} | {r['iterates']} | {r['J_first']:.4g} | {r['J_last']:.4g} | {(r['J_last'] / r['J_first'] - 1) * 100:+.2f} % | {r['seconds']:.0f} |"
            )
    gs = gradient_seed_convergence(os.path.join(FIGURES, "gradient_seed_convergence.png"))
    if gs:
        report["gradient_seed_convergence"] = gs
        print(
            "gradient SE vs seeds:",
            dict(zip(gs["counts"], [f"{v:.2e}" for v in gs["se"]], strict=True)),
        )
    sg = sphere_gate()
    if sg:
        report["sphere"] = sg
        print(
            f"sphere radius: ensemble {sg['ensemble']:.4g} ± {sg['ensemble_se']:.2g} vs FD {sg['fd']:.4g} ± {sg['fd_se']:.2g}, z = {sg['z']:+.2f} ({sg['seeds']} seeds)"
        )
    with open(os.path.join(HERE, "study_results.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(HERE, "study_results.json"))


if __name__ == "__main__":
    main()
