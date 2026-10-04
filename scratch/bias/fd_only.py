"""CRN central differences only (no adjoint) of the windowed drag in s, at several h, for
many seeds: a precise FD mean where the paired gate's per-seed FD noise is too large.

    PYTHONPATH=. python3 scratch/bias/fd_only.py OUT.jsonl S SEED0 SEED1 '{overrides}'
"""

import json
import sys
import time

import jax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    hs = over.pop("fd_steps", [0.1, 0.2])
    cfg = dict(OS.default_config(), squash=s, **over)
    for i in range(int(sys.argv[3]), int(sys.argv[4])):
        t0 = time.time()
        fd = EM.finite_differences(cfg, "squash_y", jax.random.key(cfg["seed_start"] + i), hs)
        row = {"s": s, "seed": i, "config": over, "fd_steps": hs, "fd": fd,
               "seconds": time.time() - t0}
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
