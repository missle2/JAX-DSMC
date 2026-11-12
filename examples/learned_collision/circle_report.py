"""The circle test's comparison table (:mod:`.circle`): every coarse run against the fine
NTC run of the same wall, on the coarse run's own cells.

Per run: drag and (diffuse wall) heat load over the fine run's; the shock stand-off (where
the stagnation-line density first reaches twice the free stream's, linearly interpolated
on each run's own cells) minus the fine run's; the stagnation-line peak temperature against
the fine run's restricted onto the same cells; and RMS field errors (``n``, ``T``, ``|u|``,
over the fine field's range in the region) in the **shock layer** (fine ``n > 1.5``
upstream of the body centre) and the **wake** (behind the body), both without the cells
within one coarse cell of the surface (the cut cells, whose fluid volume differs).

    PYTHONPATH=. python3 -m examples.learned_collision.circle_report [--tag v2]

Writes ``examples/learned_collision/results/circle_report_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from examples.learned_collision.shear_wave import RESULTS
from examples.mesh import forward as F

CENTRE = np.array(F.CASES["circle_xlong"]["centre"])
RADIUS = F.CASES["circle_xlong"]["radius"]


def standoff(x, n):
    """Nose minus the first upstream crossing of ``n = 2`` [m] (NaN if none)."""
    x, n = np.asarray(x), np.asarray(n)
    nose = CENTRE[0] - RADIUS
    for i in range(1, len(x)):
        if x[i] >= nose:
            break
        if n[i - 1] < 2.0 <= n[i]:
            xs = x[i - 1] + (2.0 - n[i - 1]) / (n[i] - n[i - 1]) * (x[i] - x[i - 1])
            return float(nose - xs)
    return float("nan")


def _ratio(a, a_se, b, b_se):
    r = a / b
    return r, abs(r) * math.sqrt((a_se / a) ** 2 + (b_se / b) ** 2)


def compare(run, fine):
    """The metrics (module docstring) of one coarse run against the fine run."""
    cells, h = run["cells"], run["cell"]
    ref = {k: np.array(v) for k, v in fine[f"fields_{cells}"].items()}
    got = {k: np.array(v) for k, v in run[f"fields_{cells}"].items()}
    nx, ny = ref["nrho"].shape
    xc = F.CASES["circle_xlong"]["lo"][0] + (np.arange(nx) + 0.5) * h
    yc = (np.arange(ny) + 0.5) * h
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    clear = np.hypot(xx - CENTRE[0], yy - CENTRE[1]) > RADIUS + h * math.sqrt(2.0)
    regions = {
        "shock": clear & (ref["nrho"] > 1.5) & (xx < CENTRE[0]),
        "wake": clear & (xx > CENTRE[0] + RADIUS) & (np.abs(yy - CENTRE[1]) < RADIUS + 1),
    }
    out = {"label": run["label"], "cells": cells, "wall": run["wall"]}  # fmt: skip
    out["drag_ratio"], out["drag_ratio_se"] = _ratio(run["drag"], run["drag_se"], fine["drag"],
                                                     fine["drag_se"])  # fmt: skip
    if run["wall"] == "diffuse":
        out["heat_ratio"], out["heat_ratio_se"] = _ratio(run["heat"], run["heat_se"],
                                                         fine["heat"], fine["heat_se"])  # fmt: skip
    out["standoff"] = standoff(run["x"], run["stagnation"]["nrho"])
    out["standoff_fine"] = standoff(fine["x"], fine["stagnation"]["nrho"])
    rows = slice(ny // 2 - 1, ny // 2 + 1)
    out["peak_T"] = float(got["temp"][:, rows].mean(axis=1).max())
    out["peak_T_fine"] = float(ref["temp"][:, rows].mean(axis=1).max())
    for region, mask in regions.items():
        for f in ("nrho", "temp", "speed"):
            span = float(np.ptp(ref[f][mask])) if mask.any() else float("nan")
            err = float(np.sqrt(np.mean((got[f][mask] - ref[f][mask]) ** 2)))
            out[f"{region}_{f}"] = err / span if span > 0 else float("nan")
    return out


def from_v1(run):
    """A first-round (v1, 2026-11-11) coarse specular run in the v2 layout: its fields were
    already on its own 0.5 m cells (``scale 1`` = ``c8``); the stagnation line is re-taken as
    the mean of the two rows either side of ``y = 5 m``, as v2 does."""
    fields = {k: np.array(run[k]) for k in ("nrho", "speed", "temp")}
    ny = fields["nrho"].shape[1]
    rows = slice(ny // 2 - 1, ny // 2 + 1)
    return {
        "label": run["operator"],
        "cells": "c8",
        "wall": "specular",
        "cell": 0.5,
        "drag": run["drag"],
        "drag_se": run["drag_se"],
        "heat": 0.0,
        "heat_se": 0.0,
        "fields_c8": {k: v.tolist() for k, v in fields.items()},
        "x": run["x"],
        "stagnation": {k: v[:, rows].mean(axis=1).tolist() for k, v in fields.items()},
    }


def load(tag):
    """The runs of ``tag``, plus the v1 coarse specular runs no v2 run replaces."""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"circle_*_{tag}.json"))
            if not p.name.startswith("circle_report")]  # fmt: skip
    have = {(r["label"], r["cells"], r["wall"]) for r in runs}
    for p in sorted(RESULTS.glob("circle_*_s1_v1.json")):
        run = from_v1(json.loads(p.read_text()))
        if (run["label"], "c8", "specular") not in have:
            runs.append(run)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args(argv)
    runs = load(args.tag)
    fine = {r["wall"]: r for r in runs if r["cells"] == "fine"}
    rows = []
    for run in runs:
        if run["cells"] == "fine" or run["wall"] not in fine:
            continue
        rows.append(compare(run, fine[run["wall"]]))
    rows.sort(key=lambda r: (r["wall"], r["cells"], r["label"]))
    for r in rows:
        heat = (f"heat {r['heat_ratio']:.3f}+-{r['heat_ratio_se']:.3f}  "
                if "heat_ratio" in r else "")  # fmt: skip
        print(f"{r['wall']:8s} {r['cells']:3s} {r['label']:22s} drag {r['drag_ratio']:.3f}+-"
              f"{r['drag_ratio_se']:.3f}  {heat}stand-off {r['standoff']:.2f} (fine "
              f"{r['standoff_fine']:.2f}) m  peak T {r['peak_T']:.1f} ({r['peak_T_fine']:.1f})  "
              f"shock n/T/u {r['shock_nrho']:.3f}/{r['shock_temp']:.3f}/{r['shock_speed']:.3f}"
              f"  wake {r['wake_nrho']:.3f}/{r['wake_temp']:.3f}/{r['wake_speed']:.3f}")  # fmt: skip
    (RESULTS / f"circle_report_{args.tag}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
