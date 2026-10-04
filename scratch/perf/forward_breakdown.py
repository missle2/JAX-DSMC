"""Per-step GPU time of the polygon forward (scratch/perf/profile/forward_polygon, from
profile_forward.py) by what the work is, each kernel attributed through the optimised HLO
dump's source metadata (XLA_FLAGS=--xla_dump_to=/tmp/hlo_poly, scratch/perf/dump_run.py),
against SPARTA's whole step on one core -> examples/mesh/figures/performance/forward_step_breakdown.png
and scratch/perf/profile/forward_breakdown.json.

    python3 scratch/perf/forward_breakdown.py /tmp/hlo_poly
"""

import ast
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
# Categories by the source function of a kernel's root (its file:line, mapped to the enclosing
# function with `ast`, so edits that move lines do not break the attribution); first match wins
WALL_EVENTS = "wall events: draws, re-emission, tallies"
TRACKER = "tracker: passes, lanes, compaction, epilogue"
SEARCH = "surface search + inside check"
FUNCTIONS = [
    ("wall_adjoint.py", ("conditional_time", "_fixed_time", "_time_draws", "_no_time_draws"),
     "epsilon conditional time (unused at epsilon = 0)"),
    ("wall_adjoint.py", ("_reemission", "_dense_reemission", "_compact_reemission",
                         "_element_events", "_pass_draws", "_slot_uniforms", "_apply_events",
                         "_face_tally", "_face_tallies", "_compact_face_tallies",
                         "_box_tallies", "_surface_tally", "_trace_hit"), WALL_EVENTS),
    ("sampling.py", None, WALL_EVENTS),
    ("inflow.py", ("flux_quantile", "_flux_quantile_jvp", "_flux_quantile_start",
                   "_flux_model_step", "_flux_halley_step", "flux_cdf"),
     "inflow velocity (Halley inverse CDF)"),
    ("inflow.py", None, "inflow: positions, slots, entrant tracking"),
    ("geometry.py", None, SEARCH),
    ("levelset.py", None, SEARCH),
    ("analytic.py", None, SEARCH),
    ("ntc.py", None, "NTC collisions (0 accepted at this Kn)"),
    ("vss.py", None, "NTC collisions (0 accepted at this Kn)"),
    ("safe.py", None, "NTC collisions (0 accepted at this Kn)"),
    ("sort.py", None, "cell sort"),
    ("sample.py", None, "cell temperature"),
    ("wall_adjoint.py", None, TRACKER),
    ("compaction.py", None, TRACKER),
]
PATHS = [
    ("inflow_velocities", "inflow velocity (Halley inverse CDF)"),
    ("collide", "NTC collisions (0 accepted at this Kn)"),
    ("inflow", "inflow: positions, slots, entrant tracking"),
    ("surface_search", SEARCH),
    ("move", TRACKER),
]
SOURCES = {
    name: os.path.join(ROOT, "jax_dsmc", sub, name)
    for sub, name in [("", "wall_adjoint.py"), ("", "sampling.py"), ("", "inflow.py"),
                      ("mesh", "geometry.py"), ("mesh", "levelset.py"), ("mesh", "analytic.py"),
                      ("collide", "ntc.py"), ("collide", "vss.py"), ("", "safe.py"),
                      ("", "sort.py"), ("", "sample.py"), ("", "compaction.py"),
                      ("", "geometry.py")]
}  # fmt: skip
_FUNCS = {}


def _function_of(name, line):
    """The innermost function of ``jax_dsmc``'s file ``name`` containing ``line``."""
    if name not in _FUNCS:
        spans = []
        path = SOURCES.get(name)
        if path and os.path.exists(path):
            for node in ast.walk(ast.parse(open(path).read())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append((node.lineno, node.end_lineno, node.name))
        _FUNCS[name] = spans
    inside = [(hi - lo, f) for lo, hi, f in _FUNCS[name] if lo <= line <= hi]
    return min(inside)[1] if inside else None


def classify(location, path):
    if location:
        name, line = location.rsplit(":", 1)
        func = _function_of(name, int(line))
        for f, funcs, cat in FUNCTIONS:
            if name == f and (funcs is None or func in funcs):
                return cat
    return next((c for key, c in PATHS if key in path), "other")


def fusion_sources(dump):
    """{fusion name: 'file:line' of its root's metadata} from the optimised HLO text."""
    path = glob.glob(os.path.join(dump, "*jit__run*after_optimizations.txt"))[0]
    out = {}
    for line in open(path):
        m = re.match(
            r"\s*(?:ROOT )?%?([\w.\-]+) = .*?source_file=\"([^\"]+)\" source_line=(\d+)", line
        )
        if m:
            out.setdefault(m.group(1), f"{os.path.basename(m.group(2))}:{m.group(3)}")
    return out


def main(dump):
    src = fusion_sources(dump)
    base = os.path.join(HERE, "profile", "forward_polygon")
    ev = np.load(os.path.join(base, "events.npz"))["events"]
    tables = json.load(open(os.path.join(base, "events.json")))
    ops, kern = tables["op"], tables["kernel"]
    steps = 200
    cats = {}
    for o, k, d in zip(ev[:, 3].astype(int), ev[:, 4].astype(int), ev[:, 1], strict=True):
        name = re.sub(r"_(\d+)$", r".\1", kern[k])
        cat = classify(src.get(name, src.get(kern[k], "")), ops[o])
        cats[cat] = cats.get(cat, 0.0) + d / 1e6 / steps
    scopes = json.load(open(os.path.join(HERE, "profile", "forward_scopes.json")))["polygon"]
    wall_ms = scopes["wall_seconds_200_steps"] / steps * 1e3
    busy_ms = sum(cats.values())
    cats["GPU idle: launches, syncs"] = wall_ms - busy_ms
    sparta = []
    for line in open(os.path.join(HERE, "bench_sparta.txt")).readlines()[1:]:
        b, r, _rep, secs, *_ = line.split()
        if b == "serial":
            sparta.append(float(secs))
    sparta_ms = min(sparta)  # per 1000 steps -> ms per step
    json.dump({"ms_per_step": cats, "wall_ms": wall_ms, "sparta_serial_ms": sparta_ms},
              open(os.path.join(HERE, "profile", "forward_breakdown.json"), "w"), indent=1)  # fmt: skip
    order = sorted(cats.items(), key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(order))[::-1]
    colours = ["#c9c9c4" if "idle" in c else "#2a78d6" for c, _v in order]
    ax.barh(y, [v for _c, v in order], color=colours)
    for yi, (_c, v) in zip(y, order, strict=True):
        ax.text(v, yi, f" {v:.2f} ms", va="center", fontsize=7.5)
    ax.axvline(sparta_ms, color="#eb6834", lw=1.5)
    ax.text(sparta_ms, y[0] + 0.6, f" SPARTA's whole step, 1 core: {sparta_ms:.2f} ms", color="#eb6834",
            fontsize=7.5)  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels([c for c, _v in order], fontsize=7.5)
    ax.set_xlabel("ms per step")
    ax.set_title(
        f"polygon circle forward, steady state (~46k particles, 60k slots), RTX 3050 Ti float64: "
        f"{wall_ms:.1f} ms/step wall",
        fontsize=8.5,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(ROOT, "examples", "mesh", "figures", "performance", "forward_step_breakdown.png"), dpi=150
    )
    for c, v in order:
        print(f"{v:6.3f} ms  {c}")
    print("wall", wall_ms, "sparta", sparta_ms)


if __name__ == "__main__":
    main(sys.argv[1])
