"""Band-Gaussian vs Helmholtz-filtered steps from the same gradients (2026-10-26).

Along a run's saved iterates (``coef_<k>.npy``, ``grad_<k>.npy``: ``--save-gradient 1``), the
area-tangent gradient of each iterate is fed to both regularisations of one rule, each with
its own state threaded through the sequence (the rule's memory sees the same gradients):

* ``gd``: the bare smoothing, ``S S g`` against ``F F^T g`` (the steepest-descent step);
* ``ox_adam`` and ``mma_s``: the rule's step (optax chain / ``optimize.rule_step``).

Steps are compared as normal displacements on the iterate's surface, ``V = -dphi / |grad
phi|`` at the zero-contour points: the cosine between the two displacement profiles and the
ratio of their RMS.  Prints a per-rule summary and writes ``scratch/lsfilter/stepcompare_<tag>
.json`` and a figure of the profiles at a few iterates.

    PYTHONPATH=. python3 scratch/lsfilter/stepcompare.py TAG [--every 5]
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

sys.path.insert(0, "examples/mesh")
import optax_rules as OX  # noqa: E402
import optimize as O  # noqa: E402

RULES = ("gd", "ox_adam", "mma_s")


def surface_points(ls, lo, hi):
    """The zero contour's points (the longest piece) and their arc length."""
    pieces = O.surface_polyline(ls, lo, hi, n=800)
    pts = max(pieces, key=len)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, arc


def displacement(ls, delta, pts):
    """``-dphi / |grad phi|`` of the coefficient step ``delta`` at the points ``pts``."""
    x = jnp.asarray(pts)
    dphi = O.L.value(dataclasses.replace(ls, coef=jnp.asarray(delta)), x)
    grad = O.L.value_and_grad(ls, x)[1]
    return -np.asarray(dphi) / np.linalg.norm(np.asarray(grad), axis=1)


def compare(v1, v2):
    cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-300))
    ratio = float(np.sqrt(np.mean(v2**2)) / max(np.sqrt(np.mean(v1**2)), 1e-300))
    return cos, ratio


class Replay:
    """One rule under one filter, its state threaded along the iterates."""

    def __init__(self, rule, cfg, ls):
        self.rule, self.cfg = rule, cfg
        self.state = {"area0": None}
        if rule == "ox_adam":
            self.ctx = OX.Context()
            self.tx = OX.build("ox_adam", self.ctx, cfg["step"])
            self.ox_state = self.tx.init(jnp.asarray(ls.coef))

    def step(self, ls, g, g_full, g_area, k, lo, hi, area0):
        cfg = self.cfg
        helm = cfg["filter"] == "helmholtz"
        # as optimize.run: the Helmholtz rules project the constraints in the design space
        rows = O.constraint_rows(g_area, ls.coef.ndim) if helm else None
        smoothing = O.smoothing(cfg, ls, lo, hi, rows)
        g_tangent = None if helm else g_area
        if self.rule == "gd":
            d = smoothing.out(smoothing.into(g))
            if not helm:
                d, _ = O.tangent_component(d, g_area)
            return -cfg["step"] * d / max(np.abs(d).max(), 1e-300)
        if self.rule == "ox_adam":
            d, self.ox_state = OX.step(self.tx, self.ox_state, self.ctx, g, ls, g_tangent, cfg,
                                       smoothing)  # fmt: skip
            return d
        self.state["area0"] = area0
        _base, d, _p, self.state, _x = O.rule_step(
            dict(cfg, optimizer="mma_s"), ls, g, g_full, g_area, np.zeros(1), self.state, k, lo, hi
        )
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()
    out = Path(O.RESULTS) / args.tag
    hist = json.loads((out / "history.json").read_text())
    cfg = dict(O.default_config(), **hist["config"])
    grid, ls = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    area0 = hist["area_target"]
    ks = sorted(int(p.stem.split("_")[1]) for p in out.glob("grad_*.npy"))
    band = {r: Replay(r, dict(cfg, filter="band"), ls) for r in RULES}
    helm = {r: Replay(r, dict(cfg, filter="helmholtz"), ls) for r in RULES}
    result = {r: [] for r in RULES}
    profiles = {}
    for k in ks:
        ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
        g_full = np.load(out / f"grad_{k}.npy").astype(np.float64)
        g_area = O.area_and_gradient(ls, lo, hi)[1]
        g, _ = O.constrained_gradient(g_full, g_area)
        show = k % args.every == 0
        pts, arc = surface_points(ls, lo, hi) if show else (None, None)
        for r in RULES:
            db = band[r].step(ls, g, g_full, g_area, k, lo, hi, area0)
            dh = helm[r].step(ls, g, g_full, g_area, k, lo, hi, area0)
            if not show:
                continue
            vb, vh = displacement(ls, db, pts), displacement(ls, dh, pts)
            cos, ratio = compare(vb, vh)
            result[r].append({"k": k, "cos": cos, "rms_ratio": ratio,
                              "pred_band": float(np.sum(g_full * db)),
                              "pred_helm": float(np.sum(g_full * dh))})  # fmt: skip
            if k in (ks[0], ks[len(ks) // 2], ks[-1]):
                profiles[(r, k)] = (arc, vb, vh)
        if show:
            print(f"k {k}: " + "  ".join(f"{r} cos {result[r][-1]['cos']:+.3f} rms "
                                          f"{result[r][-1]['rms_ratio']:.2f}" for r in RULES),
                  flush=True)  # fmt: skip
    for r in RULES:
        c = np.array([x["cos"] for x in result[r]])
        q = np.array([x["rms_ratio"] for x in result[r]])
        print(f"{r:8s} cos median {np.median(c):+.3f} min {c.min():+.3f}  rms ratio median "
              f"{np.median(q):.2f}")  # fmt: skip
    Path(f"scratch/lsfilter/stepcompare_{args.tag}.json").write_text(json.dumps(result, indent=1))
    fig, axes = plt.subplots(len(RULES), 3, figsize=(13, 2.6 * len(RULES)), squeeze=False)
    for i, r in enumerate(RULES):
        for j, k in enumerate((ks[0], ks[len(ks) // 2], ks[-1])):
            if (r, k) not in profiles:
                continue
            arc, vb, vh = profiles[(r, k)]
            ax = axes[i, j]
            ax.plot(arc, vb, color="#2a78d6", lw=1.4, label="band Gaussian")
            ax.plot(arc, vh, color="#e34948", lw=1.4, ls="--", label="Helmholtz")
            ax.set_title(f"{r}, iterate {k}", fontsize=10, loc="left")
            ax.set_xlabel("arc length [m]")
            ax.set_ylabel("normal step [m]")
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"scratch/lsfilter/stepcompare_{args.tag}.png", dpi=90)


if __name__ == "__main__":
    main()
