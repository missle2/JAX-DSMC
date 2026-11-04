"""The compact pair record's bitwise check on one optimiser iterate (2026-11-03, study group
``bitchk``): ``bitchk_dmhz0_compact`` (``--record lean``) against ``bitchk_dmhz0_dense``
(``--record lean_dense``) -- the iterate-0 gradient, the next shape and the history's
objective / forces must be bit for bit equal.  The 10-27 lsfilter run's stored iterate 0
(the same command on an RTX 1g.24gb slice) is printed as a looser reference: code has moved
since (identical there too would mean nothing else in its path changed).

    PYTHONPATH=. python3 scratch/bitchk/compare.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("examples/mesh/results/optimize")
RUNS = {
    "compact": "bitchk_dmhz0_compact",
    "dense": "bitchk_dmhz0_dense",
    "10-27": "circle_long_kn0.5_M10_S1_ox_adam_dmhz0_tan_sm8_free_g32_w500",
}
KEYS = ("objective", "grad_norm", "hits", "area_after_step", "predicted_decrease")


def load(tag):
    out = ROOT / tag
    row = json.loads((out / "history.json").read_text())["rows"][0]
    return {
        "grad": np.load(out / "grad_0.npy"),
        "coef": np.load(out / "coef_1.npy") if (out / "coef_1.npy").exists() else None,
        "row": row,
    }


def compare(a, b, name_a, name_b):
    print(f"\n{name_a} vs {name_b}:")
    for key in ("grad", "coef"):
        x, y = a[key], b[key]
        if x is None or y is None:
            print(f"  {key}: missing")
            continue
        same = np.array_equal(x, y)
        diff = float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(y))), 1e-300)
        print(f"  {key:5s} bitwise {'EQUAL' if same else 'DIFFERENT'}  max |diff| / max |ref| "
              f"{diff:.3e}")  # fmt: skip
    for key in KEYS:
        va, vb = a["row"].get(key), b["row"].get(key)
        print(f"  {key:20s} {va!r:>24}  {vb!r:>24}  {'EQUAL' if va == vb else 'DIFFERENT'}")


def main():
    runs = {name: load(tag) for name, tag in RUNS.items() if (ROOT / tag).exists()}
    if "compact" in runs and "dense" in runs:
        compare(runs["compact"], runs["dense"], "compact", "dense")
    if "compact" in runs and "10-27" in runs:
        compare(runs["compact"], runs["10-27"], "compact", "10-27 run")


if __name__ == "__main__":
    main()
