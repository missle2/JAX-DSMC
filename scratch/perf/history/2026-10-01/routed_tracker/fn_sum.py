"""Per-function times of a profile_forward_cpu.py JSON (move files only)."""

import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jax_dsmc")
PATHS = {
    "wall_adjoint.py": "wall_adjoint.py",
    "geometry.py": "mesh/geometry.py",
    "levelset.py": "mesh/levelset.py",
    "sampling.py": "sampling.py",
    "inflow.py": "inflow.py",
    "grid.py": "grid.py",
}
src = {}


def fn_of(path, line):
    if path not in src:
        src[path] = open(os.path.join(ROOT, path)).read().split("\n")
    for i in range(line - 1, -1, -1):
        m = re.match(r"def (\w+)", src[path][i])
        if m:
            return m.group(1)
    return "?"


d = json.load(open(sys.argv[1]))
agg = {}
for loc, ms in d["lines"].items():
    f = loc.split(":")[0]
    key = f if f not in PATHS else f + ":" + fn_of(PATHS[f], int(loc.split(":")[1]))
    agg[key] = agg.get(key, 0.0) + ms
for k, v in sorted(agg.items(), key=lambda t: -t[1])[: int(sys.argv[2]) if len(sys.argv) > 2 else 30]:
    print(f"{v:.3f} {k}")
