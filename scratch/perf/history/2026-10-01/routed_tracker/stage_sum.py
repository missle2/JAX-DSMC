"""Group a profile_forward_cpu.py JSON's per-line times into move vs the rest."""

import json
import sys

MOVE = ("wall_adjoint.py", "geometry.py", "levelset.py", "sampling.py", "analytic.py", "walk.py")
for path in sys.argv[1:]:
    d = json.load(open(path))
    groups = {}
    for loc, ms in d["lines"].items():
        f = loc.split(":")[0]
        g = "move" if f in MOVE else ("unattributed" if f == "?" else f)
        groups[g] = groups.get(g, 0.0) + ms
    print(path.split("/")[-1], f"wall {d['wall_ms_per_step']:.2f}",
          {k: round(v, 2) for k, v in sorted(groups.items(), key=lambda t: -t[1])})
