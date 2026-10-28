"""Single-iterate tests of the thickness penalty and the topology guard on the DSMC runs'
saved bodies and shape gradients (2026-10-27).  Per (run, iterate): the thin share T of the
surface (depth 2 knots), the DSMC gradient g and T's gradient through the run's Helmholtz
filter (``F F^T``): their sizes (``w_eq``: the weight at which the penalty's filtered step is
as large as the objective's) and their cosine; then one filtered steepest-descent step of the
run's step size along ``g + w dT`` per weight ``w`` (in units of J, -L / D for the L / D runs;
drag J0 for drag): the first-order objective change ``g . delta``, T after, and the body's
pieces after on the knots and on a half-knot sampling, unguarded and guarded (with the knots
the guard held).  Steepest descent stands in for Adam (its elementwise scaling would change
the step's shape, not the balance at a given weight).

    PYTHONPATH=. python3 scratch/topo/single.py
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from jax_dsmc.mesh import topology as T  # noqa: E402

ROOT = Path(O.RESULTS)
C2, W2 = "circle_long_kn0.5", "sm8_free_g32_w500"
CASES = [
    (f"{C2}_S1_ox_adam_ldhz0_floor_{W2}", 249),
    (f"{C2}_M10_S1_ox_adam_ldhz0_floor_{W2}", 160),
    (f"{C2}_M10_S1_ox_adam_hz0_tan_{W2}", 249),
    ("sphere_long_kn0.5_S1_ox_adam_ldfl_hz0_free_w300", 200),
    ("sphere_long_kn0.5_S1_ox_adam_ldfl_hz0_free_w300", 249),
    ("sphere_long_kn0.5_M10_S1_ox_adam_ldfl_hz0_free_w300", 149),
    ("sphere_long_kn0.5_M10_S1_ox_adam_drag_hz0_free_w300", 149),
]
WEIGHTS = (0.0, 0.5, 1.0, 2.0, 5.0)
SAMPLE_CHUNK = 1 << 16


@jax.jit
def _values(ls, x):
    return O.L.value(ls, x)


def fine_pieces(ls, lo, hi):
    """Face-connected pieces of ``phi < 0`` sampled every half knot over the box."""
    spacing = 0.5 * float(ls.h)
    axes = [np.arange(lo[a] + 0.5 * spacing, hi[a], spacing) for a in range(len(lo))]
    x = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(lo))
    pad = -len(x) % SAMPLE_CHUNK
    xp = np.concatenate((x, np.repeat(x[-1:], pad, 0)))
    phi = np.concatenate([np.asarray(_values(ls, jnp.asarray(xp[i : i + SAMPLE_CHUNK])))
                          for i in range(0, len(xp), SAMPLE_CHUNK)])[: len(x)]  # fmt: skip
    return int(ndimage.label(phi.reshape([len(a) for a in axes]) < 0)[1])


def case(tag, k):
    out = ROOT / tag
    hist = json.loads((out / "history.json").read_text())
    cfg = dict(O.default_config(), **hist["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
    g = np.load(out / f"grad_{k}.npy").astype(np.float64)
    lift = cfg["objective"] == "lift_drag"
    scale = 1.0 if lift else float(hist["rows"][0]["objective"])
    depth = 2.0 * float(ls.h)
    t0, g_t = O.thinness_and_gradient(ls, lo, hi, depth)
    sm = O.smoothing(cfg, ls, lo, hi)
    d_g = np.asarray(sm.out(sm.into(g)))
    d_t = np.asarray(sm.out(sm.into(scale * g_t)))
    w_eq = np.abs(d_g).max() / max(np.abs(d_t).max(), 1e-300)
    cos = float(np.sum(d_g * d_t) / (np.linalg.norm(d_g) * np.linalg.norm(d_t) + 1e-300))
    v0 = O.knot_values(ls)
    print(f"{tag} @ {k}: {'L/D' if lift else 'J'} {hist['rows'][k]['objective']:+.4g}, T {t0:.3f}"
          f" (depth {depth:g} m), pieces knots {T.pieces(v0)} / fine {fine_pieces(ls, lo, hi)}; "
          f"w_eq {w_eq:.3g}, cos(F F^T g, F F^T dT) {cos:+.3f}", flush=True)  # fmt: skip
    rows = []
    for w in WEIGHTS:
        d = d_g + w * d_t
        delta = -cfg["step"] * d / np.abs(d).max()
        stepped = dataclasses.replace(ls, coef=ls.coef + jnp.asarray(delta))
        guarded, counts = O.topology_guard(ls, stepped)
        t1 = O.thinness_and_gradient(stepped, lo, hi, depth)[0]
        row = {"w": w, "dJ": float(np.sum(g * delta)), "T": t1,
               "pieces": [T.pieces(O.knot_values(stepped)), fine_pieces(stepped, lo, hi)],
               "guarded": [T.pieces(O.knot_values(guarded)), fine_pieces(guarded, lo, hi)],
               "held": [counts["held_inside"], counts["held_outside"]]}  # fmt: skip
        rows.append(row)
        print(f"   w {w:4.1f}: dJ {row['dJ']:+.4f}  T {t0:.3f} -> {t1:.3f}  pieces knots/fine "
              f"{row['pieces']} -> guarded {row['guarded']} (held in/out {row['held']})",
              flush=True)  # fmt: skip
    return {"tag": tag, "k": k, "T": t0, "w_eq": w_eq, "cos": cos, "steps": rows}


def main():
    jax.config.update("jax_enable_x64", True)
    results = []
    for tag, k in CASES:
        if not (ROOT / tag / f"grad_{k}.npy").exists():
            print(f"{tag} @ {k}: no saved gradient")
            continue
        results.append(case(tag, k))
    Path("scratch/topo/single.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
