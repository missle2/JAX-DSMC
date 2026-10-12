"""The stored virtual-partner device gates (examples/gradient_bench/results) re-run with a
per-cell backward fit (jax_dsmc.cell_fit): per artifact and basis, the gradient over the
first NSEEDS seeds, paired per seed with the stored per-query (k-nearest) gradient --
the forward and FD are deterministic per seed (reproduce.py: ~1e-15) -- and against the
stored FD (the smallest step).

    PYTHONPATH=. python3 scratch/cellfit/gates.py OUT.jsonl NSEEDS 'basis1,basis2' [regex]
"""

import glob
import json
import os
import re
import sys
import tempfile
import time

import jax
import numpy as np

sys.path.insert(0, "scratch/collconv")
from reproduce import argv_of  # noqa: E402

from examples.gradient_bench import RESULTS  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

SKIP = ("mesh", "mach", "_seed")


def se(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a))


def main():
    path, seeds, bases = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    pattern = re.compile(sys.argv[4] if len(sys.argv) > 4 else "_dev")
    files = sorted(glob.glob(os.path.join(RESULTS, "ensemble_*_virt*_dev*.json")))
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = os.path.basename(f)[:-5]
            if any(s in name for s in SKIP) or not pattern.search(name):
                continue
            for basis in bases:
                out = os.path.join(tmp, f"{name}_{basis}.json")
                module, argv, stored, name = argv_of(f, seeds, out)
                argv[argv.index("--backward-basis") + 1] = basis
                t0 = time.time()
                module.main(argv)
                ensemble_backward.configure()
                # the examples jit a fresh closure per seed (ensemble_couette's wall score):
                # the executables pile up over seeds x bases until the OOM killer fires
                jax.clear_caches()
                fresh = json.load(open(out))["rows"]
                old = {r["seed"]: r for r in stored["rows"]}
                rows = [r for r in fresh if r["seed"] in old]
                g = np.array([r["gradient"] for r in rows], float)
                g0 = np.array([old[r["seed"]]["gradient"] for r in rows], float)
                fd = np.array([np.atleast_1d(old[r["seed"]]["fd"])[-1] for r in rows], float)
                obj = max(abs(r["objective"] - old[r["seed"]]["objective"]) for r in rows)
                rec = {
                    "artifact": name, "basis": basis, "n": len(rows),
                    "grad": g.tolist(), "grad_knn": g0.tolist(), "fd": fd.tolist(),
                    "objective_mismatch": obj, "seconds": time.time() - t0,
                }  # fmt: skip
                d = g - g0
                ref = abs(g0.mean())
                print(f"{name[:70]:70s} {basis:22s} n={len(rows)} cell {g.mean():+.4g} knn "
                      f"{g0.mean():+.4g} paired {d.mean()/ref*100:+.1f}% +- {se(d)/ref*100:.1f}%"
                      f"  FD {fd.mean():+.4g} +- {se(fd):.2g} cell-FD "
                      f"{(g - fd).mean()/abs(fd.mean())*100:+.1f}% +- "
                      f"{se(g - fd)/abs(fd.mean())*100:.1f}%  knn-FD "
                      f"{(g0 - fd).mean()/abs(fd.mean())*100:+.1f}% +- "
                      f"{se(g0 - fd)/abs(fd.mean())*100:.1f}%  obj {obj:.0e} "
                      f"({rec['seconds']:.0f} s)", flush=True)
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
