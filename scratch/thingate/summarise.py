"""Summary of the thin-body adjoint gates (scratch/thingate/cluster_job.sh, 2026-10-17).

    run_locked.sh cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/thingate/summarise.py \
        [scratch/thingate/cluster]

Squash jobs (``sq_*``, ``sqg2_*``; one row per seed, the fits paired on its forward): per
(job kind, s, fit) the assembled ``dD/ds`` (the shipped ``along_pool_plane`` variant: tally +
pathwise x + v + volume + order + score + silhouette + timing), its paired difference from the
seed's CRN FD (mean, s.e., z), the pathwise v part (mean, median and max |.|) and the carried
value's largest |lambda|.  optm jobs (``om_*``, ``omg2_*``): per (iterate, fit) ``g . d``
against the FD along each direction, each term's projection, the seed-mean gradient's norm
against the optimiser's ``grad_norm`` at that iterate (the reproduction check: the shipped
fit on the same seeds), and where the per-knot pathwise v map peaks (from the npz maps).
Writes ``<dir>/summary.json`` and ``examples/mesh/figures/optc/thingate.png``.
"""

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402

FIT_NAMES = {
    "cell:quad:adapt1.5:align0.9:both": "Q quadratic",
    "cell:cubicv:adapt1.5:align0.9:both": "C cubic-in-v (shipped)",
    "cell:cubicv:adapt1.5:ridge1e-3:align0.9:both": "CR cubic, ridge 1e-3",
    "cell:cubicv:adapt4:align0.9:both": "CA cubic, adapt 4",
    "cell:cubicv:adapt1.5:loo:align0.9:both": "CL cubic, leave-one-out order",
    "cell:cubicv:adapt1.5:guard2:align0.9:both": "G2 cubic, leverage guard 2",
    "cell:cubicv:adapt1.5:guard8:align0.9:both": "G8 cubic, leverage guard 8",
    "cell:cubicv:adapt1.5:guard32:align0.9:both": "G32 cubic, leverage guard 32",
}
ORDER = list(FIT_NAMES.values())
COLOURS = dict(
    zip(ORDER, ("#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000",
                "#999999"), strict=True)
)  # fmt: skip
OPTM = "examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_adam_tan_sm8_free"


def se(a):
    a = np.asarray(a, float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def fit_name(tag):
    """``S4K8cell:..`` (sweep.py) or ``cell:.._k32`` (geom_gate.py) -> a FIT_NAMES label."""
    for basis, name in FIT_NAMES.items():
        if basis in tag:
            return name
    return tag


def assembled(run):
    t = run["variants"]["along_pool_plane"]
    return (run["tally"] + t["x"] + t["v"] + run["volume"] + run["order"] + run["score"]
            + t["sil"] + run["timing"])  # fmt: skip


def guard_levels(runs):
    """The mean fraction of queries per order (mean, lin, quad, cubic) of a guard fit's runs."""
    levels = [x["guard_levels"] for x in runs if x.get("guard_levels")]
    return np.mean(levels, axis=0).tolist() if levels else None


def rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def squash(root):
    groups = {}
    for path in sorted(glob.glob(f"{root}/sq*_s*_b*.jsonl")):
        kind = re.sub(r"_s[0-9.]+_b\d+\.jsonl$", "", os.path.basename(path))
        for r in rows(path):
            groups.setdefault((kind, r["s"]), []).append(r)
    out = {}
    for (kind, s), rs in sorted(groups.items()):
        fd = np.array([r["fd"][0] if r["fd"] else np.nan for r in rs])
        rec = {"kind": kind, "s": s, "seeds": len(rs), "fd": float(np.mean(fd)), "fd_se": se(fd),
               "fits": {}}  # fmt: skip
        for tag in rs[0]["runs"]:
            runs = [r["runs"][tag] for r in rs]
            g = np.array([assembled(x) for x in runs])
            v = np.array([x["variants"]["along_pool_plane"]["v"] for x in runs])
            d = g - fd
            lam = [x.get("lambda_absmax") for x in runs if x.get("lambda_absmax") is not None]
            rec["fits"][fit_name(tag)] = {
                "gradient": float(g.mean()), "gradient_se": se(g),
                "minus_fd": float(d.mean()), "minus_fd_se": se(d),
                "z": float(d.mean() / se(d)) if len(d) > 1 and se(d) > 0 else float("nan"),
                "pathwise_v": float(v.mean()), "pathwise_v_median_abs": float(np.median(np.abs(v))),
                "pathwise_v_max_abs": float(np.abs(v).max()),
                "tally": float(np.mean([x["tally"] for x in runs])),
                "lambda_absmax_median": float(np.median(lam)) if lam else None,
                "per_seed": g.tolist(),
                "guard_levels": guard_levels(runs),
            }  # fmt: skip
        out[f"{kind} s={s:.4g}"] = rec
    return out


def knot_xy(shape_):
    """Knot coordinates of the optm level set (its grid from the optimiser's config)."""
    import jax  # noqa: PLC0415

    jax.config.update("jax_platforms", "cpu")
    import optimize as O  # noqa: PLC0415

    from jax_dsmc.mesh import levelset as L  # noqa: PLC0415

    with open(f"{OPTM}/history.json") as fh:
        cfg = json.load(fh)["config"]
    _grid, ls = O.initial_levelset(cfg)
    x = np.asarray(L.knots(ls))
    assert ls.coef.shape == tuple(shape_), (ls.coef.shape, shape_)
    return x


def optm(root):
    out = {}
    hist = None
    if os.path.exists(f"{OPTM}/history.json"):
        with open(f"{OPTM}/history.json") as fh:
            hist = json.load(fh)["rows"]
    xy = None
    for path in sorted(glob.glob(f"{root}/om*_it*_b*.jsonl")):
        job = os.path.basename(path)[: -len(".jsonl")]
        rs = rows(path)
        if not rs:
            continue
        it = rs[0]["iterate"]
        rec = {"iterate": it, "seeds": len(rs), "fd": {}, "fits": {}}
        if hist and it < len(hist):
            rec["optimiser_grad_norm"] = hist[it]["grad_norm"]
        for n, fds in rs[0]["fd"].items():
            f = np.array([r["fd"][n][0] for r in rs])
            rec["fd"][n] = {"mean": float(f.mean()), "se": se(f), "per_seed": f.tolist()}
        for tag in rs[0]["runs"]:
            name = fit_name(tag)
            runs = [r["runs"][tag] for r in rs]
            fr = {"directional": {}, "terms": {}}
            for n in rs[0]["fd"]:
                g = np.array([x["directional"][n] for x in runs])
                f = np.array(rec["fd"][n]["per_seed"])
                d = g - f
                fr["directional"][n] = {
                    "g_dot_d": float(g.mean()), "se": se(g), "minus_fd": float(d.mean()),
                    "minus_fd_se": se(d), "per_seed": g.tolist(),
                }  # fmt: skip
                for term in runs[0]["terms"]:
                    fr["terms"].setdefault(term, {})[n] = float(
                        np.mean([x["terms"][term][n] for x in runs])
                    )
            fr["lambda_absmax"] = [x.get("lambda_absmax") for x in runs]
            fr["guard_levels"] = guard_levels(runs)
            maps = sorted(glob.glob(f"{root}/maps/{job}/it{it}_seed*_{tag.replace(':', '-')}.npz"))
            if maps:
                grads = [np.load(m)["gradient"].astype(float) for m in maps]
                fr["seed_mean_grad_norm"] = float(np.linalg.norm(np.mean(grads, axis=0)))
                pv = np.mean([np.load(m)["pathwise_v"].astype(float) for m in maps], axis=0)
                if xy is None:
                    xy = knot_xy(pv.shape)
                top = np.argsort(np.abs(pv).ravel())[::-1][:5]
                fr["pathwise_v_peak"] = [
                    {"x": float(xy[i, 0]), "y": float(xy[i, 1]), "value": float(pv.ravel()[i])}
                    for i in top
                ]
                fr["pathwise_v_norm"] = float(np.linalg.norm(pv))
            rec["fits"][name] = fr
        out[job] = rec
    return out


def plot(sq, om):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    ax = axes[0]
    keys = list(sq)
    width = 0.8 / max(len(ORDER), 1)
    for j, name in enumerate(ORDER):
        xs, ys, es = [], [], []
        for i, k in enumerate(keys):
            f = sq[k]["fits"].get(name)
            if f is None:
                continue
            xs.append(i + (j - (len(ORDER) - 1) / 2) * width)
            ys.append(f["minus_fd"])
            es.append(2 * f["minus_fd_se"] if np.isfinite(f["minus_fd_se"]) else 0.0)
        if xs:
            ax.errorbar(xs, ys, es, fmt="o", ms=5, capsize=2, color=COLOURS[name], label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xticks(range(len(keys)), [k.replace(" ", "\n") for k in keys], fontsize=7.5)
    ax.set_title("(a) squash: adjoint dD/ds minus the CRN FD, paired per seed (±2 s.e.)",
                 fontsize=10, loc="left")  # fmt: skip
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    jobs = list(om)
    for j, name in enumerate(ORDER):
        xs, ys, es = [], [], []
        for i, k in enumerate(jobs):
            f = om[k]["fits"].get(name)
            if f is None or "nose" not in f["directional"]:
                continue
            d = f["directional"]["nose"]
            xs.append(i + (j - (len(ORDER) - 1) / 2) * width)
            ys.append(d["minus_fd"])
            es.append(2 * d["minus_fd_se"] if np.isfinite(d["minus_fd_se"]) else 0.0)
        if xs:
            ax.errorbar(xs, ys, es, fmt="o", ms=5, capsize=2, color=COLOURS[name], label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(range(len(jobs)), jobs, fontsize=7.5)
    ax.set_title("(b) optm Kn 0.5 M 10 iterates: g · (nose bump) minus the CRN FD (±2 s.e.)",
                 fontsize=10, loc="left")  # fmt: skip
    ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Thin-body adjoint gates (thingate): the backward fits paired on one forward "
                 "per seed", fontsize=11)  # fmt: skip
    fig.tight_layout()
    path = figure_path("optc", "thingate.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def levels_text(f):
    lv = f.get("guard_levels")
    return "" if not lv else "  orders " + "/".join(f"{x:.3f}" for x in lv)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scratch/thingate/cluster"
    sq, om = squash(root), optm(root)
    for k, rec in sq.items():
        print(f"{k}  ({rec['seeds']} seeds)  FD {rec['fd']:+.4g} ± {rec['fd_se']:.2g}")
        for name, f in rec["fits"].items():
            lam = f["lambda_absmax_median"]
            print(f"    {name:32s} dD/ds {f['gradient']:+.4g} ± {f['gradient_se']:.2g}  "
                  f"- FD {f['minus_fd']:+.4g} ± {f['minus_fd_se']:.2g} (z {f['z']:+.1f})  "
                  f"v {f['pathwise_v']:+.4g} (median |v| {f['pathwise_v_median_abs']:.3g}, "
                  f"max {f['pathwise_v_max_abs']:.3g})  max|lambda| "
                  f"{'-' if lam is None else f'{lam:.3g}'}{levels_text(f)}")  # fmt: skip
    for job, rec in om.items():
        print(
            f"{job}  iterate {rec['iterate']} ({rec['seeds']} seeds)  optimiser |g| "
            f"{rec.get('optimiser_grad_norm', float('nan')):.4g}  FD "
            + "  ".join(f"{n} {v['mean']:+.4g} ± {v['se']:.2g}" for n, v in rec["fd"].items())
        )
        for name, f in rec["fits"].items():
            dirs = "  ".join(f"g.{n} {d['g_dot_d']:+.4g} (- FD {d['minus_fd']:+.3g} ± "
                             f"{d['minus_fd_se']:.2g})" for n, d in f["directional"].items())  # fmt: skip
            peak = f.get("pathwise_v_peak", [{}])[0]
            print(
                f"    {name:32s} {dirs}  |mean g| {f.get('seed_mean_grad_norm', float('nan')):.4g}"
                f"  v peak at ({peak.get('x', float('nan')):.2f}, {peak.get('y', float('nan')):.2f})"
                f"  max|lambda| {max((v for v in f['lambda_absmax'] if v is not None), default=0):.3g}"
                + levels_text(f)
            )
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump({"squash": sq, "optm": om}, fh, indent=1)
    if sq or om:
        plot(sq, om)


if __name__ == "__main__":
    main()
