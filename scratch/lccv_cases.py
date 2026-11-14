"""Per-case metrics of every lccv run present: shear mu ratio, shock int txx / int qx."""
import json
import sys
from pathlib import Path

import numpy as np

from examples.learned_collision import sweep_report as sr

R = Path(sys.argv[1])
plan = json.loads((R / "sweep_plan_partial.json").read_text())
base = sr.base_rows(R)


def fmt(m):
    if m is None:
        return "      --     "
    if "mu_ratio" in m:
        return f"  {m['mu_ratio']:.3f}      "
    return f"{m['int_txx_ratio']:.3f}/{m['int_qx_ratio']:.3f} "


cases = ["S3", "S4", "K1", "K4", "K8"]
print("run".ljust(16) + "".join(c.center(13) for c in cases) + "  last-fifth bias2   held")
print("base".ljust(16) + "".join(fmt(base[c]) for c in cases))
for r in plan["runs"]:
    p = R / f"train_mlp_{r['tag']}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    v = d.get("validation") or {}
    log = d["log"]
    tail = log[-max(1, len(log) // 5):]
    b2 = np.mean([sum(x["bias2"].values()) for x in tail])
    b0 = np.mean([sum(x["bias2"].values()) for x in log[:10]])
    row = "".join(fmt(v.get(c)) for c in cases)
    held = r["validate"][0] if r["validate"] else "all"
    print(r["tag"][3:].ljust(16) + row + f"  {b0:.2e}->{b2:.2e}  {held}")
