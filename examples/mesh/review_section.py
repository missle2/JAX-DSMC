"""Emit the markdown of review section 7 (mesh forward gates) from the artifacts.

python3 examples/mesh/review_section.py >> docs/ADJOINT_BRANCH_REVIEW.md
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main() -> int:
    with open(os.path.join(RESULTS, "geometry_checks.json")) as fh:
        geo = json.load(fh)
    with open(os.path.join(RESULTS, "record_budget.json")) as fh:
        budget = json.load(fh)
    lines = []
    lines.append(
        "### 7.1 Geometry (`examples/mesh/geometry_checks.py`, `results/geometry_checks.json`)\n"
    )
    lines.append(
        "| case | cells outside / inside / cut (SPARTA) | flow volume ours / SPARTA dump / exact | max cell error | touched full cells | brute-force candidate mismatches | watertight | build s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in ("circle", "sphere"):
        g = geo[case]
        lines.append(
            f"| {case} (sub {g['sub']}) | {tuple(g['counts'])} ({tuple(g['counts_sparta'])}) | "
            f"{g['flow_volume_ours']:.5f} / {g['flow_volume_sparta_dump']:.5f} / {g['flow_volume_exact']:.5f} | "
            f"{g['max_cell_error']:.2e} | {g['n_touched_full']} | {g['brute_force_mismatched_cells']} | "
            f"{g['watertight']['even_crossings']} ({g['watertight']['segments']} segments) | {g['build_seconds']:.1f} |"
        )
    lines.append("")
    conv = geo["convergence"]["circle"]
    lines.append(
        "Volume-fraction convergence (circle, max cell error in cell-volume units): "
        + ", ".join(f"sub {r['sub']}: {r['max_cell_error'] / 0.25:.1e}" for r in conv)
        + ".\n"
    )
    lines.append(
        "### 7.2 Forward gates (`examples/mesh/report.py`, `results/forward_gates.json`)\n"
    )
    with open(os.path.join(RESULTS, "logs", "report.log")) as fh:
        for line in fh:
            if line.startswith(("###", "|")):
                lines.append(line.rstrip().replace("### ", "#### "))
        lines.append("")
    lines.append("### 7.3 Free-molecular drag (FG5, `examples/mesh/free_molecular.py`)\n")
    lines.append(
        "| case | model | C_D at 2r/L = 0.075 (0.05 for the sphere) | extrapolated to zero blockage | half-range reference | z | full-Maxwellian closed form |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for case in ("circle", "sphere"):
        path = os.path.join(RESULTS, f"free_molecular_{case}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            fm = json.load(fh)
        for acc, label in ((0.0, "specular"), (1.0, "diffuse 300 K")):
            rows = [r for r in fm["rows"] if r["acc"] == acc]
            small = min(rows, key=lambda r: r["radius"])
            s = fm["summary"][f"acc{acc:g}"]
            lines.append(
                f"| {case} | {label} | {small['cd']:.3f} ± {small['cd_se']:.3f} | {s['cd_extrapolated']:.3f} ± {s['cd_se']:.3f} | "
                f"{s['cd_reference']:.3f} | {s['z']:+.2f} | {small.get('cd_reference_full_maxwellian', float('nan')):.3f} |"
            )
    lines.append("")
    lines.append("### 7.4 Record budget (G9, `examples/mesh/record_budget.py`)\n")
    for r in budget["rows"]:
        lines.append(
            f"* {r['slots']} slots x {r['steps']} captured steps, `K_bounce = {r['k_bounce']}`: {r['gib']:.2f} GiB ({r['bytes_per_slot_step']:.0f} B per slot-step); `walls.v_out` is the largest leaf and is replayable from the element ids."
        )
    lines.append("")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
