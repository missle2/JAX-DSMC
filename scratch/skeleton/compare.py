"""Probe thinness vs spread skeleton on the dlhp bodies (2026-11-27, paper plan phase C gate):
does the skeleton push the nose outward less than the probe measure (levelset.body_thinness),
which blunts tips to ~its depth?

For iterates of results/optimize/<tag> (default dlhp_a20_pen_f4, the bulged-nose run): each
measure's value and its steepest-descent surface motion.  A coefficient gradient ``g`` moves
the surface along the outward normal by ``u = sum_j B_j g_j / |grad phi|`` per unit descent
step (``dphi = -sum B_j g_j``; positive u = the body grows there).  Reported per region (nose:
the front ``--nose`` m of the body; rest), each measure's push normalised by its own gradient's
largest surface value; the figure (examples/mesh/figures/skeleton/compare_<tag>.png) shows the
side view (x-y) of the surface points coloured by u for both measures, and S_c on z = 0.

    PYTHONPATH=. python3 scratch/skeleton/compare.py [--tag T] [--iterates 30 60 90 119]
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

sys.path.insert(0, "scratch/sph3d")
sys.path.insert(0, "scratch/optax")
sys.path.insert(0, "examples/mesh")
from figure_paths import figure_path  # noqa: E402
from shape3d import O  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402
from jax_dsmc.mesh import skeleton as K  # noqa: E402

ROOT = Path(O.RESULTS)


def push(ls, grad, pts):
    """Outward surface motion per unit steepest-descent step of the gradient ``grad``."""
    num = L.value(dataclasses.replace(ls, coef=grad), pts)
    _phi, g = L.value_and_grad(ls, pts)
    return np.asarray(num / L.safe_norm(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dlhp_a20_pen_f4")
    ap.add_argument("--iterates", type=int, nargs="+", default=[30, 60, 90, 119])
    ap.add_argument("--nose", type=float, default=0.6, help="m of body length from the front")
    args = ap.parse_args()
    out = ROOT / args.tag
    hist = json.loads((out / "history.json").read_text())
    cfg = dict(O.default_config(), **hist["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    depth = O.thickness_min(cfg, ls0)
    s = K.SkeletonSettings(r_s=0.5 * depth)
    fig, axes = plt.subplots(len(args.iterates), 3, figsize=(18, 3.2 * len(args.iterates)),
                             squeeze=False)  # fmt: skip
    rows = []
    for i, k in enumerate(args.iterates):
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
        t_probe, g_probe = L.thinness_and_gradient(ls, lo, hi, depth, 4)
        p_skel, g_skel = K.penalty_and_gradient(ls, lo, hi, 1.0, s)
        f = K.surface_fields(ls, lo, hi, s)
        surf = np.abs(f["phi"]) < 0.5 * f["hv"]
        idx = np.argwhere(surf)
        pts = f["origin"] + f["hv"] * (idx + 0.5)
        u_probe = push(ls, g_probe, jnp.asarray(pts))
        u_skel = push(ls, g_skel, jnp.asarray(pts))
        sc = f["s_c"][surf]
        x_front = pts[:, 0].min()
        nose = pts[:, 0] < x_front + args.nose
        row = {"iterate": k, "probe": float(t_probe), "skeleton": float(p_skel)}
        for name, u in (("probe", u_probe), ("skeleton", u_skel)):
            scale = max(np.abs(u).max(), 1e-30)
            row[f"{name}_nose_out"] = float(np.mean(np.maximum(u[nose], 0)) / scale)
            row[f"{name}_rest_out"] = float(np.mean(np.maximum(u[~nose], 0)) / scale)
            row[f"{name}_nose_share"] = float(np.abs(u[nose]).sum() / np.abs(u).sum())
        row["skeleton_sc_nose"] = float(sc[nose].mean())
        row["skeleton_sc_rest"] = float(sc[~nose].mean())
        rows.append(row)
        print(json.dumps(row), flush=True)
        for j, (name, u) in enumerate((("probe thinness", u_probe), ("skeleton", u_skel))):
            ax = axes[i, j]
            lim = max(np.abs(u).max(), 1e-30)
            order = np.argsort(pts[:, 2])  # far side first
            sca = ax.scatter(pts[order, 0], pts[order, 1], c=u[order], s=3, cmap="RdBu_r",
                             vmin=-lim, vmax=lim)  # fmt: skip
            fig.colorbar(sca, ax=ax, fraction=0.03)
            ax.set_aspect("equal")
            ax.set_title(f"iterate {k}: {name} descent push (red = grows)", fontsize=10,
                         loc="left")  # fmt: skip
        ax = axes[i, 2]
        mid = f["counts"][2] // 2
        ext = (f["origin"][0], f["origin"][0] + f["hv"] * f["counts"][0], f["origin"][1],
               f["origin"][1] + f["hv"] * f["counts"][1])  # fmt: skip
        im = ax.imshow(f["s_c"][:, :, mid].T, origin="lower", extent=ext, cmap="magma",
                       vmin=0, vmax=1)  # fmt: skip
        ax.contour(np.linspace(ext[0], ext[1], f["counts"][0]),
                   np.linspace(ext[2], ext[3], f["counts"][1]), f["phi"][:, :, mid].T,
                   levels=[0.0], colors="white", linewidths=0.8)  # fmt: skip
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_aspect("equal")
        ax.set_title(f"S_c on z = 0 (P {p_skel:.4f}; probe T {t_probe:.4f})", fontsize=10,
                     loc="left")  # fmt: skip
    fig.tight_layout()
    path = figure_path("skeleton", f"compare_{args.tag}.png")
    fig.savefig(path, dpi=90)
    Path(str(path).replace(".png", ".json")).write_text(json.dumps(rows, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
