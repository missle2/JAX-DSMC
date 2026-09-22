"""CRN finite differences of the plate's width gradient vs the FD step.

The GP gate's two steps (0.1 / 0.05) read 1.336 +- 0.060 and 1.169 +- 0.096 e-20 over the
same four seeds -- a 1.5 sigma drift.  This runs the same forwards at more steps on the
same seeds (RNG contract: seed i = ``jax.random.key(32 + i)``) so the h -> 0 limit of the
FD can be read; output ``probe_plate_fd_steps.json`` (copied to
``examples/mesh/results/plate_fd_steps.json`` for ``report.py``).

    python3 scratch/mesh/probe_plate_fd_steps.py --seeds 4 --steps 0.2 0.025 0.0125
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import jax  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from examples import ensemble_mesh as EM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=32)
    ap.add_argument("--steps", type=float, nargs="+", default=[0.2, 0.025, 0.0125])
    args = ap.parse_args()
    cfg = EM.default_config()
    cfg.update(
        {
            "case": "plate",
            "body": "levelset",
            "acc": 1.0,
            "levelset_h": 0.0625,
            "levelset_samples": 16,
            "grazing": 0.05,
            "silhouette_band": 0.03,
        }
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_plate_fd_steps.json")
    rows = json.load(open(path))["rows"] if os.path.exists(path) else []
    done = {(r["seed"], h) for r in rows for h in r["fd"]}
    for i in range(args.seeds):
        key = jax.random.key(args.seed_start + i)
        row = next((r for r in rows if r["seed"] == i), None)
        if row is None:
            row = {"seed": i, "fd": {}}
            rows.append(row)
        for h in args.steps:
            if (i, str(h)) in done:
                continue
            t0 = time.time()
            (fd,) = EM.finite_differences(cfg, "width", key, [h])
            row["fd"][str(h)] = float(fd)
            print(f"seed {i} h {h}: FD {fd:.5g}  ({time.time() - t0:.0f}s)", flush=True)
            with open(path, "w") as fh:
                json.dump({"config": cfg, "rows": rows}, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
