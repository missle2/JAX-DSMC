"""Separate objectives through one backward (2026-11-03, ``ensemble_mesh.shape_terms(
objectives=...)``) against the single-objective backwards on the same seed: drag (the
``drag`` objective), lift (``drag`` minus ``drag_lift`` at lift weight 1: D - (D - L)) and
the total heat rate (the ``heat`` objective = the heat functional at unit node weights).
Every term of every variant must agree to the backward's rounding (2 vs 6 value columns:
~1e-13 relative); also the wall-clock of the one 6-column sweep against the three 2-column
ones.  The diffuse M 10 circle (c10_diff's gate) with the heat lattice of dlh3d.

    PYTHONPATH=. python3 scratch/multiobj/check_multi.py [--smoke] [--seed 0]

``--smoke``: fnum x 16, 150 + 40 steps (a laptop CPU run of the plumbing, minutes).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

PARTS = ("pathwise", "score", "silhouette", "timing", "order")
VARIANTS = ("cell_along_pool_plane",)


def total(terms, volume):
    g = sum(np.asarray(getattr(terms, k)) for k in PARTS if getattr(terms, k) is not None)
    return g if volume is None else g + np.asarray(volume)


def single(cfg, key, objective, weight=0.0):
    c = {**cfg, "objective": objective, "lift_weight": weight}
    t0 = time.time()
    terms, info, s = EM.shape_terms(c, key, None, "radius", 32, cfg["timing_band"], VARIANTS)
    return total(terms[VARIANTS[0]], s["volume"]), info, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {**V.gate_config("c10_diff")[0], "heat_patch": 0.2}
    if args.smoke:
        cfg = {**cfg, "fnum_scale": 16.0 * cfg["fnum_scale"], "warmup": 150, "steps": 40,
               "window": 20}  # fmt: skip
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    key = jax.random.key(32 + args.seed)
    t0 = time.time()
    terms, info, s = EM.shape_terms(
        cfg, key, None, "radius", 32, cfg["timing_band"], VARIANTS,
        objectives=("drag", "lift", "heat"),
    )  # fmt: skip
    t_multi = time.time() - t0
    multi = {n: total(terms[n][VARIANTS[0]], s["volume"][n]) for n in terms}
    g_drag, i_drag, t_drag = single(cfg, key, "drag")
    g_dl, _i, t_dl = single(cfg, key, "drag_lift", 1.0)
    g_heat, i_heat, t_heat = single(cfg, key, "heat")
    ref = {"drag": g_drag, "lift": g_drag - g_dl, "heat": g_heat}
    rows = {}
    for name in ("drag", "lift", "heat"):
        a, b = multi[name], ref[name]
        rel = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))
        dot = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        rows[name] = {"max_rel_diff": rel, "cosine": dot, "norm_multi": float(np.linalg.norm(a)),
                      "norm_single": float(np.linalg.norm(b))}  # fmt: skip
        print(f"{name:5s} max |multi - single| / max |single| {rel:.2e}  cosine {dot:.12f}  "
              f"|g| {np.linalg.norm(a):.6g} vs {np.linalg.norm(b):.6g}", flush=True)  # fmt: skip
    print(f"values: multi {info['objective_values']}  single drag {i_drag['objective']:.6g} "
          f"heat {i_heat['objective']:.6g}", flush=True)  # fmt: skip
    print(f"seconds: one 6-column seed {t_multi:.0f} (backward {info['seconds_backward']:.0f}, "
          f"shape {info['seconds_shape']:.0f})  three 2-column seeds {t_drag:.0f} + {t_dl:.0f} + "
          f"{t_heat:.0f}", flush=True)  # fmt: skip
    out = Path("examples/mesh/results/multiobj")
    out.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"s{args.seed}"
    (out / f"check_multi_{tag}.json").write_text(json.dumps({
        "rows": rows, "values": info["objective_values"], "seconds_multi": t_multi,
        "seconds_backward_multi": info["seconds_backward"], "seconds_single": [t_drag, t_dl, t_heat],
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
