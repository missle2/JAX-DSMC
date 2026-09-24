"""The lane tracker (SPARTA's move loop) against the dense stencil tracker: speed, memory,
and agreement of the results -- artifact and figure.

    python3 -m examples.mesh.tracker_bench --import-logs   # probe logs -> results/tracker_bench.json
    python3 -m examples.mesh.tracker_bench --plot          # the JSON -> figures/tracker_lane.png
    python3 -m examples.mesh.tracker_bench --table         # the review's markdown table

The measurements are the GPU probes of 2026-09-24 (RTX 3050 Ti laptop, float64), all on the
roughest optc shape (the level-set circle after 8 GD iterates of ``circle_coarse_S1_gd``,
concave windward notches) at seed 32:

* ``scratch/mesh/probe_lane_speed.py`` -- the uncaptured forward (700 + 300 steps, 60 k
  slots): compiled wall time, particles deleted as stuck, the windowed drag tally, the
  alive count; and the lane occupancy (slots with a pass-0 event) over 20 steady moves.
  Logs ``probe_lane_speed.log`` (the stencil row and the occupancy) and
  ``probe_lane_speed3.log`` (the lane rows after entrants moved to the lane as well --
  the earlier lane timings still paid a dense K-pass loop over the inflow's entrants
  every step and are not used).
* ``scratch/mesh/probe_memory.py`` -- allocator peaks of the captured forward and of the
  full ``shape_terms`` chain (replay, two backwards, shape VJP) at 4 k / 7.8 k / 15 k
  slots: ``probe_memory.log`` (stencil, 4 passes), ``probe_memory_lane.log`` (lane, 32,
  the record's later passes as ``K - 1`` lane rows) and ``probe_memory_lane_compact.log``
  (lane, 32, the compact record -- the later passes as a list of the recorded events --
  and the slim replay: per-slot totals formed on demand, the timing rewards kept as
  totals only, no host bookkeeping in ``shape_terms``).
* ``scratch/mesh/probe_lane_gradient.py`` -- one ``shape_terms`` seed at ``fnum x 4`` with
  the stencil at 4 passes and the lane at 4 and 32: objective, hits, replay error and the
  full coefficient gradient (``lane_gradient.npz``).
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGS = os.path.join(ROOT, "scratch", "mesh")
ARTIFACT = os.path.join(HERE, "results", "tracker_bench.json")
FIGURE = os.path.join(HERE, "figures", "tracker_lane.png")
GRADIENTS = os.path.join(HERE, "results", "tracker_bench_gradient.npz")

_RUN = re.compile(
    r"^(?P<tracker>\w+) K=(?P<k>\d+) run (?P<rep>\d+): (?P<sec>\d+)s\s+stuck (?P<stuck>\d+)\s+"
    r"lane_overflow (?P<over>\d+)\s+alive (?P<alive>\d+)\s+drag-sum (?P<drag>\S+)"
)
_OCC = re.compile(r"lane occupancy per step mean (?P<mean>\d+) max (?P<max>\d+)")
_SLOTS = re.compile(r"^slots (?P<slots>\d+)")
MEMORY = (
    # name, probe log, label, colour
    ("stencil_4", "probe_memory.log", "stencil, 4 passes", "0.45"),
    ("lane_32", "probe_memory_lane.log", "lane, 32 passes", "tab:blue"),
    (
        "lane_32_compact",
        "probe_memory_lane_compact.log",
        "lane, 32, compact record + slim replay",
        "tab:green",
    ),
)


def _forward_rows(path, keep):
    """Compiled (second) runs of the ``keep`` configurations in a speed log."""
    rows = []
    with open(path) as fh:
        for line in fh:
            m = _RUN.match(line.strip())
            if m and int(m["rep"]) == 1 and (m["tracker"], int(m["k"])) in keep:
                rows.append(
                    {
                        "tracker": m["tracker"],
                        "k_bounce": int(m["k"]),
                        "seconds": float(m["sec"]),
                        "stuck": int(m["stuck"]),
                        "lane_overflow": int(m["over"]),
                        "alive": int(m["alive"]),
                        "drag_sum": float(m["drag"]),
                    }
                )
    return rows


def _memory_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.startswith("{")]


def _slope(rows, key):
    n = np.array([r["N_slots"] for r in rows], dtype=float)
    y = np.array([r[key] for r in rows], dtype=float)
    slope, intercept = np.polyfit(n, y, 1)
    return float(slope), float(intercept)


def import_logs():
    forward = _forward_rows(os.path.join(LOGS, "probe_lane_speed.log"), {("stencil", 4)})
    forward += _forward_rows(
        os.path.join(LOGS, "probe_lane_speed3.log"), {("lane", 4), ("lane", 32)}
    )
    occupancy = {}
    with open(os.path.join(LOGS, "probe_lane_speed.log")) as fh:
        for line in fh:
            if (m := _OCC.search(line)) and (s := _SLOTS.match(line)):
                occupancy = {"slots": int(s["slots"]), "mean": int(m["mean"]), "max": int(m["max"])}
    memory = {}
    for name, log, _label, _color in MEMORY:
        rows = _memory_rows(os.path.join(LOGS, log))
        memory[name] = {
            "rows": rows,
            "per_slot_bytes": {
                key: _slope(rows, key)[0]
                for key in ("records_bytes", "peak_forward_captured", "peak_shape_terms")
            },
        }
    z = np.load(os.path.join(LOGS, "lane_gradient.npz"))
    np.savez(GRADIENTS, **{k: z[k] for k in z.files})  # the figure's panel (c)
    ref = z["stencil4"]
    gradient = {
        "configs": ["stencil_4", "lane_4", "lane_32"],
        "objective": z["objective"].tolist(),
        "hits": z["hits"].tolist(),
        "max_rel_diff_lane4": float(np.abs(z["lane4"] - ref).max() / np.abs(ref).max()),
        "max_rel_diff_lane32": float(np.abs(z["lane32"] - ref).max() / np.abs(ref).max()),
        "norm": [float(np.linalg.norm(z[k])) for k in ("stencil4", "lane4", "lane32")],
    }
    out = {
        "shape": "circle_coarse_S1_gd coef_8 (roughest optc shape), seed 32",
        "device": "RTX 3050 Ti laptop, float64",
        "forward": forward,
        "lane_occupancy": occupancy,
        "memory": memory,
        "gradient": gradient,
    }
    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    with open(ARTIFACT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", ARTIFACT)
    return out


def plot():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with open(ARTIFACT) as fh:
        a = json.load(fh)
    z = np.load(GRADIENTS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _panel_forward(axes[0], a)
    _panel_memory(axes[1], a)
    _panel_gradient(axes[2], a, z)
    fig.suptitle(
        f"SPARTA-style lane tracker vs the dense stencil tracker ({a['shape']}; {a['device']})",
        fontsize=9,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=130)
    plt.close(fig)
    print("wrote", FIGURE)


def _panel_forward(ax, a):
    """(a) forward time, with the discrete outcome of every run."""
    fw = a["forward"]
    labels = [f"{r['tracker']}\n{r['k_bounce']} passes" for r in fw]
    colors = ["0.55" if r["tracker"] == "stencil" else "tab:blue" for r in fw]
    bars = ax.bar(labels, [r["seconds"] for r in fw], color=colors)
    for bar, r in zip(bars, fw, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{r['seconds']:.0f} s\nstuck {r['stuck']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    drags = {f"{r['drag_sum']:.7e}" for r in fw}
    alive = {r["alive"] for r in fw}
    ax.set_ylabel("uncaptured forward, 1000 steps (s)")
    ax.set_ylim(0, 1.3 * max(r["seconds"] for r in fw))
    occ = a["lane_occupancy"]
    ax.set_title(
        f"(a) forward: drag tally {'identical' if len(drags) == 1 else 'DIFFERS'} "
        f"({next(iter(drags)) if len(drags) == 1 else ''}),\n"
        f"alive {'identical' if len(alive) == 1 else 'DIFFERS'}; lane holds "
        f"{occ['mean']}-{occ['max']} of {occ['slots']:,} slots per step",
        fontsize=8,
    )


def _panel_memory(ax, a):
    """(b) memory per slot: captured forward and the full adjoint chain."""
    for name, _log, label, color in MEMORY:
        rows = a["memory"][name]["rows"]
        n = np.array([r["N_slots"] for r in rows])
        for key, marker, what in (
            ("peak_shape_terms", "o-", "full solve"),
            ("peak_forward_captured", "s--", "captured forward"),
        ):
            per = a["memory"][name]["per_slot_bytes"][key] / 1e3
            ax.plot(
                n / 1e3,
                np.array([r[key] for r in rows]) / 1e9,
                marker,
                color=color,
                ms=4,
                label=f"{label}: {what} ({per:.0f} kB/slot)",
            )
    ax.set_xlabel("particle slots (thousands)")
    ax.set_ylabel("GPU allocator peak (GB)")
    ax.set_title("(b) memory of the adjoint chain (300 captured steps)", fontsize=8)
    ax.legend(fontsize=6.5)


def _panel_gradient(ax, a, z):
    """(c) the shape gradient, coefficient by coefficient."""
    ref, lane = z["stencil4"].ravel(), z["lane32"].ravel()
    nz = ref != 0.0
    ax.plot(ref[nz], lane[nz], ".", ms=2, color="tab:blue")
    lim = np.abs(ref[nz]).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    g = a["gradient"]
    ax.set_xlabel("dJ/dcoef, stencil, 4 passes")
    ax.set_ylabel("dJ/dcoef, lane, 32 passes")
    ax.set_title(
        f"(c) shape gradient ({nz.sum()} coefficients): max rel. diff "
        f"{g['max_rel_diff_lane32']:.1e};\nJ {g['objective'][0]:.6e} / {g['objective'][2]:.6e}, "
        f"hits {g['hits'][0]} / {g['hits'][2]}",
        fontsize=8,
    )


def table():
    """The review's markdown table, from the artifact."""
    with open(ARTIFACT) as fh:
        a = json.load(fh)
    print("| tracker | passes | forward (s) | stuck | alive | drag tally |")
    print("|---|---|---|---|---|---|")
    for r in a["forward"]:
        print(
            f"| {r['tracker']} | {r['k_bounce']} | {r['seconds']:.0f} | {r['stuck']} | "
            f"{r['alive']} | {r['drag_sum']:.7e} |"
        )
    print()
    print(
        "| per particle slot | stencil, 4 passes | lane, 32 passes | lane, compact record + slim replay | change |"
    )
    print("|---|---|---|---|---|")
    names = {
        "records_bytes": "event record (300 captured steps)",
        "peak_forward_captured": "peak, captured forward",
        "peak_shape_terms": "peak, full adjoint chain",
    }
    for key, label in names.items():
        s4 = a["memory"]["stencil_4"]["per_slot_bytes"][key] / 1e3
        l32 = a["memory"]["lane_32"]["per_slot_bytes"][key] / 1e3
        lc = a["memory"]["lane_32_compact"]["per_slot_bytes"][key] / 1e3
        print(
            f"| {label} | {s4:.1f} kB | {l32:.1f} kB | {lc:.1f} kB | {100 * (lc / s4 - 1):+.0f} % |"
        )
    t = [a["memory"][k]["rows"][-1] for k in ("stencil_4", "lane_32", "lane_32_compact")]
    print(
        f"| adjoint chain time at {t[0]['N_slots']} slots | {t[0]['seconds_shape_terms']:.0f} s | "
        f"{t[1]['seconds_shape_terms']:.0f} s | {t[2]['seconds_shape_terms']:.0f} s | "
        f"{100 * (t[2]['seconds_shape_terms'] / t[0]['seconds_shape_terms'] - 1):+.0f} % |"
    )
    g = a["gradient"]
    print()
    print(
        f"Gradient (fnum x4, seed 32): J {g['objective']}, hits {g['hits']}; max |dg|/|g| "
        f"lane 4 {g['max_rel_diff_lane4']:.1e}, lane 32 {g['max_rel_diff_lane32']:.1e}; "
        f"lane occupancy {a['lane_occupancy']['mean']} mean / {a['lane_occupancy']['max']} max "
        f"of {a['lane_occupancy']['slots']} slots."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--import-logs", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the review's markdown table")
    args = ap.parse_args(argv)
    if args.import_logs:
        import_logs()
    if args.table:
        table()
    if args.plot or not (args.import_logs or args.table):
        plot()


if __name__ == "__main__":
    main()
