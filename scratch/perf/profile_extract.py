"""Flatten a profile (scratch/perf/profile/<tag>/) into scratch/perf/profile/<tag>/events.npz:
every GPU kernel / memcpy (start ns, duration ns, hlo_module id, op-path id, kernel-name
id) and the host stage annotations (``stage:*``) and jitted dispatches (``PjitFunction``)
with their windows.  String tables alongside (json).

    python3 scratch/perf/profile_extract.py gpu_after
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xspace  # noqa: E402


def intern(table, index, s):
    if s not in index:
        index[s] = len(table)
        table.append(s)
    return index[s]


def main(tag):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile", tag)
    for stage in sorted(os.listdir(root)):
        sub = os.path.join(root, stage)
        if os.path.isdir(os.path.join(sub, "plugins")):
            extract(sub)


def extract(root):
    # the newest trace: a folder re-profiled keeps the old runs beside it
    traces = glob.glob(os.path.join(root, "plugins", "profile", "*", "*.xplane.pb"))
    path = max(traces, key=os.path.getmtime)
    buf, planes = xspace.planes(path)
    tables = {"module": ([], {}), "op": ([], {}), "kernel": ([], {})}
    rows = []
    gpu = next(s for n, s in planes if n.startswith("/device:GPU"))
    for _line, name, start, dur, stats in xspace.plane_events(buf, gpu, event_stats=True):
        module = stats.get("hlo_module", "?")
        op = stats.get("name") or stats.get("hlo_op") or name
        rows.append(
            (
                start,
                dur,
                intern(*tables["module"], module),
                intern(*tables["op"], op),
                intern(*tables["kernel"], name),
            )
        )
    arr = np.array(rows, dtype=np.float64)
    host = next(s for n, s in planes if n == "/host:CPU")
    marks = []
    for line, name, start, dur, _s in xspace.plane_events(buf, host):
        if name.startswith("stage:") or name.startswith("PjitFunction"):
            marks.append({"line": line, "name": name, "start": start, "dur": dur})
    np.savez(os.path.join(root, "events.npz"), events=arr)
    with open(os.path.join(root, "events.json"), "w") as fh:
        json.dump({k: v[0] for k, v in tables.items()} | {"host": marks}, fh)
    print(root, len(rows), "device events,", len(marks), "host marks", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
