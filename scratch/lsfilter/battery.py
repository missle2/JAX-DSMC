"""The lsfilter battery's summary (2026-10-26): band Gaussian vs Helmholtz-filtered runs, pair
by pair, from whatever has been pulled.

Per run: iterates done, the best 25-iterate running mean of the objective (L / D for the L / D
runs) and where, the last 25 iterates' mean, whether the patience test stopped it, the share of
checked iterates whose body is in more than one piece (every ``--every``-th iterate: tip
droplets, break-up), the mean largest coefficient step and the seconds per iterate.  Writes
``scratch/lsfilter/battery.png`` (the objective histories, one panel per comparison) and
``battery.json``.

    PYTHONPATH=. python3 scratch/lsfilter/battery.py [--every 10] [--no-pieces]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

ROOT = Path(O.RESULTS)
C = "circle_long_kn0.5"
W = "free_g32_w500"
BAND, HELM = "#2a78d6", "#e34948"
# comparison -> [(label, tag, colour, line style)]
COMPARISONS = {}
OLD = "#9a9a9a"  # the reinitialised band runs (optrc, ldopt _ldfl_): the old method
for regime, label in (("_M10", "M10"), ("", "Kn0.5")):
    for arm, rule in (("ox_adam", "Adam"), ("mma_s", "MMA")):
        runs = [("band", f"{C}{regime}_S1_{arm}_bd0_tan_sm8_{W}", BAND, "-"),
                ("Helmholtz", f"{C}{regime}_S1_{arm}_hz0_tan_sm8_{W}", HELM, "-")]
        if regime:
            runs.append(("band, reinitialised (optrc)", f"{C}{regime}_S1_{arm}_rc_tan_sm8_{W}",
                         OLD, ":"))
        COMPARISONS[f"{label} drag, {rule}"] = runs
        COMPARISONS[f"{label} L/D floor, {rule}"] = [
            ("band", f"{C}{regime}_S1_{arm}_ldbd0_floor_sm8_{W}", BAND, "-"),
            ("Helmholtz", f"{C}{regime}_S1_{arm}_ldhz0_floor_sm8_{W}", HELM, "-"),
            ("band, reinitialised (ldopt)", f"{C}{regime}_S1_{arm}_ldfl_floor_sm8_{W}", OLD, ":"),
        ]
COMPARISONS["M10 drag, Helmholtz GD"] = [
    ("Helmholtz GD", f"{C}_M10_S1_ox_gd_hz0_tan_sm8_{W}", HELM, "--")]
COMPARISONS["M10 drag from a diamond"] = [
    ("band", f"{C}_M10_S1_ox_adam_dmbd0_tan_sm8_{W}", BAND, "-"),
    ("Helmholtz", f"{C}_M10_S1_ox_adam_dmhz0_tan_sm8_{W}", HELM, "-")]
COMPARISONS["M10 L/D from a plate"] = [
    ("band", f"{C}_M10_S1_ox_adam_plbd0_floor_sm8_{W}", BAND, "-"),
    ("Helmholtz", f"{C}_M10_S1_ox_adam_plhz0_floor_sm8_{W}", HELM, "-")]
for cond, label in (("kn0.5", "Kn0.5"), ("kn0.5_M10", "M10")):
    COMPARISONS[f"3-D sphere drag, {label}"] = [
        ("band, reinitialised (sph3d)", f"sphere_long_{cond}_S1_ox_adam_drag_free_w300", OLD, ":"),
        ("Helmholtz", f"sphere_long_{cond}_S1_ox_adam_drag_hz0_free_w300", HELM, "-")]
    COMPARISONS[f"3-D sphere L/D floor, {label}"] = [
        ("band, reinitialised (sph3d)", f"sphere_long_{cond}_S1_ox_adam_ldfl_free_w300", OLD, ":"),
        ("Helmholtz", f"sphere_long_{cond}_S1_ox_adam_ldfl_hz0_free_w300", HELM, "-")]

for problem, title in (("drag", "Newtonian drag"), ("ld", "Newtonian L/D floor"),
                       ("dm", "Newtonian drag from a diamond"),
                       ("pl", "Newtonian L/D from a plate")):  # fmt: skip
    arms = ("ox_adam", "mma_s", "ox_gd") if problem in ("drag", "ld") else ("ox_adam",)
    runs = []
    for arm in arms:
        if arm != "ox_gd":
            runs.append((f"band {arm}", f"lsf_newton_{problem}_band_{arm}", BAND,
                         "-" if arm == "ox_adam" else ":"))  # fmt: skip
        runs.append((f"Helmholtz {arm}", f"lsf_newton_{problem}_helmholtz_{arm}", HELM,
                     {"ox_adam": "-", "mma_s": ":", "ox_gd": "--"}[arm]))  # fmt: skip
    if problem == "drag":
        runs.append(("Helmholtz Adam + perimeter", "lsf_newton_drag_helmholtz_ox_adam_per",
                     "#1baf7a", "-"))  # fmt: skip
    COMPARISONS[title] = runs


def running(x, w=25):
    c = np.cumsum(np.r_[0.0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(i + 1, w) for i in range(len(x))])


def pieces(out, cfg, rows, every):
    """Fraction of the checked iterates whose body has more than one connected piece."""
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    if ls0.coef.ndim != 2:
        return float("nan")
    xs = np.linspace(lo[0], hi[0], 1200)
    ys = np.linspace(lo[1], hi[1], 400)
    pts = jnp.asarray(np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2))
    multi = checked = 0
    for k in range(0, len(rows), every):
        p = out / f"coef_{k}.npy"
        if not p.exists():
            continue
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(p)))
        inside = np.asarray(O.L.value(ls, pts)).reshape(len(xs), len(ys)) < 0
        multi += ndimage.label(inside)[1] > 1
        checked += 1
    return multi / max(checked, 1)


def summarise(tag, every, count_pieces):
    out = ROOT / tag
    if not (out / "history.json").exists():
        return None
    hist = json.loads((out / "history.json").read_text())
    rows, cfg = hist["rows"], dict(O.default_config(), **hist["config"])
    j = np.array([r["objective"] for r in rows])
    lift = cfg.get("objective") == "lift_drag"
    value = -j if lift else j
    run = running(value)
    best = int(np.argmax(run) if lift else np.argmin(run))
    return {
        "iterates": len(rows),
        "best": float(run[best]),
        "best_at": best,
        "last25": float(value[-25:].mean()),
        "stopped": bool(hist.get("converged")),
        "multi_piece": pieces(out, cfg, rows, every) if count_pieces else float("nan"),
        "step_max": float(np.mean([r.get("step_max", r.get("direction_max", np.nan)) for r in rows])),
        "seconds": float(np.mean([r["seconds"] for r in rows])),
        "first": float(value[0]),
        "grad_norm": [float(r.get("surface_grad_norm", np.nan)) for r in rows[-1:]]
        + [float(np.nanmax([r.get("surface_grad_norm", np.nan) for r in rows]))],
        "series": value.tolist(),
        "measure": "L/D" if lift else "J",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--no-pieces", action="store_true")
    args = ap.parse_args()
    table = {}
    for title, runs in COMPARISONS.items():
        table[title] = {}
        for label, tag, _c, _ls in runs:
            s = summarise(tag, args.every, not args.no_pieces)
            if s is None:
                continue
            table[title][label] = s
            print(f"{title:32s} {label:28s} n {s['iterates']:3d}  best {s['measure']} "
                  f"{s['best']:.4g} @ {s['best_at']:3d}  last25 {s['last25']:.4g}  "
                  f"stopped {s['stopped']!s:5s}  multi {s['multi_piece']:.2f}  "
                  f"step {s['step_max']:.3f}  {s['seconds']:.0f} s/it  "
                  f"ratio {s['best'] / s['first']:.3f}  |grad| last/max "
                  f"{s['grad_norm'][0]:.2f}/{s['grad_norm'][1]:.2f}", flush=True)  # fmt: skip
    shown = [t for t in COMPARISONS if table[t]]
    if not shown:
        return
    cols = 3
    rows_n = -(-len(shown) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.2 * rows_n), squeeze=False)
    for ax, title in zip(axes.ravel(), shown, strict=False):
        for label, tag, colour, style in COMPARISONS[title]:
            s = table[title].get(label)
            if s is None:
                continue
            ax.plot(running(np.array(s["series"]), 10), color=colour, ls=style, lw=1.5,
                    label=label)  # fmt: skip
            ax.set_ylabel(f"{s['measure']} (10-iterate mean)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("iterate")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, color="#e4e3df", lw=0.6)
    for ax in axes.ravel()[len(shown):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("scratch/lsfilter/battery.png", dpi=90)
    slim = {t: {k: {x: y for x, y in v.items() if x != "series"} for k, v in d.items()}
            for t, d in table.items()}  # fmt: skip
    Path("scratch/lsfilter/battery.json").write_text(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
