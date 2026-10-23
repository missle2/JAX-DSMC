"""Every optimiser run on the long circle at Kn 0.5 (M 0.3 deck and M 10), 2026-10-23: the
one-seed arms (optax1seed, optrules1seed) and the four-seed ones (optrules, optrules2), per
iterate -- drag on a running mean (one seed's drag is noisy), its noise, the shape's roughness,
and how consistent successive steps are.  Writes scratch/optax/analysis.json, read by the
figure and movie scripts.

    PYTHONPATH=. python3 scratch/optax/analysis.py
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optimize as O  # noqa: E402
from roughness import turning  # noqa: E402

ROOT = "examples/mesh/results/optimize"
REGIMES = {"kn0.5": "Kn 0.5, M 0.3", "kn0.5_M10": "Kn 0.5, M 10"}
S1 = ("adam", "ox_adam", "ox_adamcos", "ox_ademamix", "ox_prodigy", "ox_sfadam", "tr", "lbfgs",
      "mma", "ox_adam_cm", "ox_adamcos_cm", "ox_prodigy_cm", "ox_prodigycos_cm")  # fmt: skip
# four seeds per iterate: optrules (cold start, 2026-10-19) and the reworked TR (optrules2)
S4 = {"S4_adam": "S4_adam_tan_sm8_free_g32", "S4_tr": "S4_tr_tan_sm8_free_g32",
      "S4_lbfgs": "S4_lbfgs_tan_sm8_free_g32", "S4_mma": "S4_mma_tan_sm8_free_g32",
      "S4_tr2": "S4_tr2_tan_sm8_free_g32_w500"}  # fmt: skip
WINDOW = 15  # running-mean width (iterates)
ROUGH_EVERY = 10
OUT = "scratch/optax/analysis.json"


def tag(regime, arm):
    if arm in S4:
        return f"circle_long_{regime}_{S4[arm]}"
    return f"circle_long_{regime}_S1_{arm}_tan_sm8_free_g32_w500"


def running(x, w=WINDOW):
    """Centred running mean, shrinking at the ends."""
    out = np.empty_like(x)
    h = w // 2
    for i in range(len(x)):
        out[i] = x[max(0, i - h) : i + h + 1].mean()
    return out


def load(regime, arm):
    d = Path(ROOT) / tag(regime, arm)
    hist = json.load(open(d / "history.json"))
    rows = hist["rows"]
    j = np.array([r["objective"] for r in rows])
    n = len(j)
    while n > 0 and not (d / f"coef_{n}.npy").exists():
        n -= 1
    coefs = [np.load(d / f"coef_{k}.npy") for k in range(n + 1)]
    return hist, j, coefs


def step_cosines(coefs):
    steps = np.diff(np.stack([c.ravel() for c in coefs]), axis=0)
    a, b = steps[1:], steps[:-1]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def main():
    # an arm already in the json at its current length is kept (the roughness is the slow part)
    old = json.load(open(OUT)) if Path(OUT).exists() else {}
    out = {}
    for regime in REGIMES:
        out[regime] = {}
        hist0 = json.load(open(Path(ROOT) / tag(regime, "S4_adam") / "history.json"))
        j0 = hist0["rows"][0]["objective"]  # the 4-seed initial drag: every run starts here
        cfg = dict(O.default_config(), **hist0["config"])
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        for arm in (*S1, *S4):
            try:
                hist, j, coefs = load(regime, arm)
            except FileNotFoundError:
                continue
            kept = old.get(regime, {}).get(arm)
            if kept is not None and len(kept["J"]) == len(j):
                out[regime][arm] = kept
                continue
            seeds = 4 if arm in S4 else 1
            r = j / j0
            m = running(r)
            resid = r - m
            rough = {}
            for k in sorted({*range(0, len(coefs), ROUGH_EVERY), len(coefs) - 1}):
                lsk = dataclasses.replace(ls0, coef=jnp.asarray(coefs[k]))
                segs = O.surface_polyline(lsk, lo, hi, n=1500)
                rough[k] = [*turning(max(segs, key=len)), len(segs)]
            cos = step_cosines(coefs)
            n = len(r)
            out[regime][arm] = {
                "seeds": seeds,
                "J": r.tolist(),
                "J_mean": m.tolist(),
                "noise_sd": float(np.std(resid[WINDOW:])),
                "best_mean": float(m.min()),
                "best_mean_at": int(m.argmin()),
                "final_mean": float(r[-WINDOW:].mean()),
                "at": {k: float(m[k]) for k in (25, 50, 100, 150, 200, 300) if k < n},
                "hits": [r_["hits"] for r_ in hist["rows"]],
                "roughness": {str(k): v for k, v in rough.items()},
                "step_cos": [None if not np.isfinite(c) else float(c) for c in cos],
                "converged": bool(hist.get("converged")),
            }
            s = out[regime][arm]
            print(f"{regime:10s} {arm:12s} n={n:4d} best~{s['best_mean']:.3f}@{s['best_mean_at']:4d} "
                  f"final~{s['final_mean']:.3f} sd={s['noise_sd']:.3f} "
                  f"rough0/best/last={rough[0][0]:.2f}/"
                  f"{rough[min(rough, key=lambda k: abs(k - s['best_mean_at']))][0]:.2f}/"
                  f"{rough[max(rough)][0]:.2f} cos={np.nanmean(cos[:50]):.2f}/"
                  f"{np.nanmean(cos[-100:]):.2f}", flush=True)  # fmt: skip
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main()
