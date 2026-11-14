"""Synthetic sweep results (from one real validation) to smoke sweep_report.py."""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from examples.learned_collision import sweep

RES = Path("examples/learned_collision/results")
out = Path(sys.argv[1])
template = json.loads((RES / "train_mlp_lcfix_mlp_q1.json").read_text())
plan = {"iters": 215, "design": sweep.design(), "train_args": "--warm-pool v1",
        "runs": [{"tag": t, "cases": c, "validate": v, "seed": s, **p}
                 for t, c, v, s, p in sweep.runs(sweep.design())]}
(out / "sweep_plan_fake.json").write_text(json.dumps(plan))
for name in ("eval_base_shear.json", "eval_base_shear_hi.json", "eval_base_shockA.json",
             "eval_base_shockB.json"):
    shutil.copy(RES / name, out / name)
rng = np.random.default_rng(0)
for k, r in enumerate(plan["runs"]):
    if k == 5:
        continue  # a missing run
    run = copy.deepcopy(template)
    noise = 0.02 * (1 + r["width"] / 64) * (r["validate"] != [])
    for c, m in run["validation"].items():
        for key in ("mu_ratio", "int_txx_ratio", "int_qx_ratio"):
            if key in m:
                m[key] += rng.normal(0, 0.01 + noise)
    if r["validate"]:
        run["validation"][r["validate"][0]] = copy.deepcopy(template["validation"][r["validate"][0]])
    (out / f"train_mlp_{r['tag']}.json").write_text(json.dumps(run))
rows = json.loads((RES / "eval_base_shockA.json").read_text())["rows"]
rows += json.loads((RES / "eval_base_shockB.json").read_text())["rows"]
rows += json.loads((RES / "eval_base_shear.json").read_text())["rows"]
fake = [dict(r, operator="train_mlp_v1") for r in rows if r["operator"] == "esbgk_recon"]
(out / "eval_fake.json").write_text(json.dumps({"rows": fake}))
