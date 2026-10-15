"""The 2026-10-13 ``optab`` A/B runs (examples/mesh/cluster/study.py --group optab): step
smoothing 2 / 4 / 8 knots against a coarser level set, the optc2 S = 4 Adam run in the long box
with the augmented-Lagrangian area.  Also the projection-mode long-box runs for reference.

    run_locked.sh cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/optab/analysis.py

Per run, from ``history.json`` and ``coef_<k>.npy``:

* the drag against the circle of the same area, ``eta = D / (D_0 sqrt(A / A_0))`` (free-
  molecular specular drag of a circle goes as its diameter): the area did not hold in these
  runs, so the raw drag mostly measures the area;
* the area error and the multiplier (the Lagrangian's control loop);
* the shape's roughness from its zero contour on a 1 cm grid: total turning / 2 pi per
  component (1 = convex), the smallest radius of curvature (curvature smoothed over 5 cm),
  the fraction of the perimeter that is concave, the number of components;
* the adjoint's apparent error along the trajectory: the observed drag change of a step
  (common random numbers: the same seeds every iterate) against the adjoint's first-order
  prediction.  Not testable on the optab runs: their ``g . dc`` is dominated by the wound-up
  pull on the area, whose second-order area change alone is ~30x a step's drag change (only
  1-4 steps per run have a pull below the initial price).  On the projection-mode runs
  (circle_coarse_*, circle_long_S{4,8}_adam) instead: ``dD_k = a g . dc_k + b shift_k`` (the
  projection's uniform shift moves the drag too, by an unknown ``dD/dshift``).  The step is
  taken along the noisy gradient itself, so ``g . dc`` carries the noise's own power and
  ``E[a] = S / (S + rho)`` with ``rho`` the per-seed noise-to-signal power of the step's
  projection: a slope below 1 that rises with the seed count is noise, not bias.

Writes ``figures/optc/optab_*.png`` and ``scratch/optab/analysis.json``.
"""

import dataclasses
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402
from figure_paths import figure_path  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

RES = "examples/mesh/results/optimize"
AB = ("ctl", "smooth4", "smooth8", "lsh0.125", "lsh0.25")
LABEL = {
    "ctl": "control: smooth 2 knots (12.5 cm), h 6.25 cm",
    "smooth4": "smooth 4 knots (25 cm)",
    "smooth8": "smooth 8 knots (50 cm)",
    "lsh0.125": "h 12.5 cm, smooth 2 knots (25 cm)",
    "lsh0.25": "h 25 cm, smooth 2 knots (50 cm)",
}
PROJ = {
    "circle_long_S4_adam": "projection S4 (reference)",
    "circle_long_S8_adam": "projection S8 (reference)",
}
# Okabe-Ito
COLOURS = {
    "ctl": "#000000",
    "smooth4": "#0072B2",
    "smooth8": "#56B4E9",
    "lsh0.125": "#D55E00",
    "lsh0.25": "#E69F00",
    "circle_long_S4_adam": "#009E73",
    "circle_long_S8_adam": "#CC79A7",
}
STYLE = {"smooth4": "-", "smooth8": "-", "lsh0.125": "--", "lsh0.25": "--"}
EVERY = 5
DX = 0.01  # contour grid (m)


def load(tag):
    with open(f"{RES}/{tag}/history.json") as fh:
        return json.load(fh)


def contour_metrics(ls, lo, hi):
    """Roughness of the zero contour (module doc)."""
    xs = np.linspace(lo[0], hi[0], 601)
    ys = np.linspace(lo[1], hi[1], 201)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    coarse = np.asarray(L.value(ls, jnp.asarray(np.stack([xx.ravel(), yy.ravel()], 1))))
    inside = coarse.reshape(xx.shape) < 0
    bx, by = xx[inside], yy[inside]
    a = np.array([bx.min(), by.min()]) - 0.3
    b = np.array([bx.max(), by.max()]) + 0.3
    a, b = np.maximum(a, lo), np.minimum(b, hi)
    gx = np.arange(a[0], b[0], DX)
    gy = np.arange(a[1], b[1], DX)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    phi = np.asarray(L.value(ls, jnp.asarray(np.stack([xx.ravel(), yy.ravel()], 1))))
    fig, ax = plt.subplots()
    cs = ax.contour(xx, yy, phi.reshape(xx.shape), levels=[0.0])
    segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 8]
    plt.close(fig)
    turning, r_min, concave, perim = [], [], 0.0, 0.0
    for seg in segs:
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        if d[-1] < 0.1:
            continue
        s = np.arange(0.0, d[-1], DX)
        p = np.stack([np.interp(s, d, seg[:, 0]), np.interp(s, d, seg[:, 1])], 1)
        t = np.diff(p, axis=0, append=p[:1])
        theta = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        dtheta = np.diff(theta, append=theta[0] + np.sign(theta[-1] - theta[0]) * 2 * np.pi)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        # orientation: counter-clockwise turns by +2 pi, so convex parts have dtheta > 0
        sign = np.sign(dtheta.sum()) or 1.0
        kappa = gaussian_filter1d(sign * dtheta / DX, 5.0 / (DX * 100), mode="wrap")
        turning.append(np.abs(dtheta).sum() / (2 * np.pi))
        r_min.append(1.0 / max(np.abs(kappa).max(), 1e-9))
        concave += DX * np.sum(kappa < -0.05)
        perim += d[-1]
    return {
        "components": len(turning),
        "turning": float(max(turning)) if turning else float("nan"),
        "r_min": float(min(r_min)) if r_min else float("nan"),
        "concave": float(concave / perim) if perim else float("nan"),
        "length": float(b[0] - a[0] - 0.6),
        "height": float(b[1] - a[1] - 0.6),
    }


def shapes(tag, h):
    grid, ls = O.initial_levelset(h["config"])
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n = len(h["rows"])
    out = {}
    for k in sorted(set(range(0, n + 1, EVERY)) | {n}):
        path = f"{RES}/{tag}/coef_{k}.npy"
        if os.path.exists(path):
            out[k] = contour_metrics(
                dataclasses.replace(ls, coef=jnp.asarray(np.load(path))), lo, hi
            )
    return out


def _fit(obs, x):
    sc = x.std(0)
    xs = x / sc
    b = np.linalg.lstsq(xs, obs, rcond=None)[0]
    r = obs - xs @ b
    cov = (r @ r / (len(obs) - x.shape[1])) * np.linalg.inv(xs.T @ xs)
    return float(b[0] / sc[0]), float(np.sqrt(cov[0, 0]) / sc[0])


def apparent_projection(h):
    """``a`` of ``dD = a g . dc + b shift`` (module doc), whole run and first half."""
    r = h["rows"]
    obs = np.array([r[k + 1]["objective"] - r[k]["objective"] for k in range(len(r) - 1)])
    x = np.array([[r[k]["predicted_decrease"], r[k]["area_shift"]] for k in range(len(r) - 1)])
    half = np.arange(len(obs)) < len(obs) // 2
    a, e = _fit(obs, x)
    a1, e1 = _fit(obs[half], x[half])
    return {
        "S": h["config"]["seeds_per_iterate"],
        "optimizer": h["config"]["optimizer"],
        "a": a,
        "a_se": e,
        "a_first": a1,
        "a_first_se": e1,
        "n": len(obs),
    }


def dilution_rho(recs):
    """``rho`` of ``a = S / (S + rho)`` by weighted least squares over the runs' first halves."""
    grid = np.linspace(0.5, 30.0, 600)
    cost = [
        sum(((r["a_first"] - r["S"] / (r["S"] + q)) / r["a_first_se"]) ** 2 for r in recs)
        for q in grid
    ]
    return float(grid[int(np.argmin(cost))])


def eta(h):
    r = h["rows"]
    d0, a0 = r[0]["objective"], r[0]["area"]
    return np.array([x["objective"] / (d0 * np.sqrt(x["area"] / a0)) for x in r])


def main():
    tags = {f"circle_long_S4_adam_al_ab_{t}": t for t in AB}
    tags.update({t: t for t in PROJ})
    hist = {t: load(t) for t in tags}
    out = {}
    for tag, key in tags.items():
        h = hist[tag]
        e = eta(h)
        rec = {
            "iterates": len(h["rows"]),
            "converged": bool(h.get("converged")),
            "eta_last20": float(e[-20:].mean()),
            "eta_min": float(e.min()),
            "area_rel_range": [
                float(min(x["area"] for x in h["rows"]) / h["rows"][0]["area"]),
                float(max(x["area"] for x in h["rows"]) / h["rows"][0]["area"]),
            ],
            "shapes": {str(k): v for k, v in shapes(tag, h).items()},
            "seconds_per_iterate": float(np.median([x["seconds"] for x in h["rows"]])),
        }
        out[key] = rec
        sh = rec["shapes"]
        last = sh[max(sh, key=int)]
        print(
            f"{key:22s} n {rec['iterates']:3d} eta last20 {rec['eta_last20']:.3f} min "
            f"{rec['eta_min']:.3f} area x[{rec['area_rel_range'][0]:.2f}, "
            f"{rec['area_rel_range'][1]:.2f}]  last shape: turning "
            f"{last['turning']:.2f} r_min {last['r_min'] * 100:.1f} cm concave "
            f"{last['concave']:.2f} comps {last['components']}",
            flush=True,
        )
    proj = {}
    for path in sorted(os.listdir(RES)):
        if path.startswith(("circle_coarse_S", "circle_long_S")) and path.endswith(
            ("_adam", "_gd")
        ):
            proj[path] = apparent_projection(load(path))
            q = proj[path]
            print(
                f"{path:24s} S {q['S']} {q['optimizer']:4s} slope {q['a']:.2f} +- {q['a_se']:.2f}"
                f"  first half {q['a_first']:.2f} +- {q['a_first_se']:.2f}  ({q['n']} steps)"
            )
    rho = dilution_rho(list(proj.values()))
    print(f"noise dilution fit: rho = {rho:.1f} (a = S / (S + rho))")
    out["projection_apparent"] = {"runs": proj, "rho": rho}
    with open("scratch/optab/analysis.json", "w") as fh:
        json.dump(out, fh)
    plot(hist, tags, out)


def _line(ax, x, y, key, **kw):
    ax.plot(
        x,
        y,
        STYLE.get(key, "-"),
        color=COLOURS[key],
        lw=1.4,
        label=LABEL.get(key, PROJ.get(key)),
        **kw,
    )


def plot(hist, tags, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    for tag, key in tags.items():
        h, rec = hist[tag], out[key]
        it = np.arange(len(h["rows"]))
        _line(axes[0, 0], it, [x["area"] / h["rows"][0]["area"] for x in h["rows"]], key)
        _line(axes[0, 1], it, [x["objective"] / h["rows"][0]["objective"] for x in h["rows"]], key)
        _line(axes[0, 2], it, eta(h), key)
        sh = rec["shapes"]
        ks = sorted(int(k) for k in sh)
        _line(axes[1, 0], ks, [sh[str(k)]["turning"] for k in ks], key)
        _line(axes[1, 1], ks, [sh[str(k)]["r_min"] * 100 for k in ks], key)
    pa = out["projection_apparent"]
    for opt, marker, colour in (("adam", "o", "#0072B2"), ("gd", "s", "#D55E00")):
        for j, (tag, q) in enumerate(t for t in pa["runs"].items() if t[1]["optimizer"] == opt):
            dx = 0.06 if "long" in tag else -0.06
            axes[1, 2].errorbar(
                q["S"] * np.exp(dx),
                q["a_first"],
                q["a_first_se"],
                fmt=marker,
                color=colour,
                mfc="white" if "long" in tag else colour,
                ms=6,
                capsize=2,
                label=f"{opt.upper()} (open: long box)" if j == 0 else None,
            )
    s = np.geomspace(0.8, 10, 100)
    axes[1, 2].plot(
        s,
        s / (s + pa["rho"]),
        color="0.3",
        lw=1.2,
        label=f"noise dilution S / (S + ρ), ρ = {pa['rho']:.1f}",
    )
    axes[1, 2].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axes[1, 2].set_ylim(-0.1, 1.15)
    titles = (
        ("(a) area / initial area", "iterate"),
        ("(b) drag / initial drag", "iterate"),
        ("(c) drag / drag of the circle of the same area", "iterate"),
        ("(d) shape: total turning / 2π (1 = convex)", "iterate"),
        ("(e) shape: smallest radius of curvature [cm]", "iterate"),
        ("(f) projection runs: observed ΔD / adjoint g·Δc (first half)", "seeds per iterate S"),
    )
    for ax, (t, xl) in zip(axes.ravel(), titles, strict=True):
        ax.set_title(t, fontsize=10, loc="left")
        ax.set_xlabel(xl, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1, 2].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(
        "optab (2026-10-13): optc2 S = 4 Adam, free-molecular, long box, augmented-"
        "Lagrangian area -- step smoothing vs level-set spacing (rtx, 300 iterates each)",
        fontsize=11,
    )
    fig.tight_layout()
    path = figure_path("optc", "optab_overview.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    main()
