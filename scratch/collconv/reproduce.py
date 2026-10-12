"""Path 1a: have the validated collisional adjoints changed?  Re-run the first seeds of every
virtual-partner gate artifact (examples/gradient_bench/results, produced 2026-09-21 or
earlier) with the code as it stands and compare objective, FD and gradient per seed.

The forward, the partner draws (keyed) and the fits are deterministic, so an unchanged
implementation reproduces the stored gradient to the solve's rounding (~1e-12; the LU
solve became an unrolled Cholesky after the artifacts).

    PYTHONPATH=. python3 scratch/collconv/reproduce.py OUT.jsonl NSEEDS [name-regex]
"""

import glob
import json
import os
import re
import sys
import tempfile
import time

from examples import ensemble_couette, ensemble_inlet_shock, ensemble_relaxation
from examples.gradient_bench import RESULTS

SKIP = ("mesh", "mach")  # mesh: collisionless planes; mach: 800 s/seed, run separately


def argv_of(path, seeds, out):
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    name = os.path.basename(path)[:-5]
    virt = re.search(r"_virt(\d+)x(\d+)(st)?(_dev)?", name)
    argv = []
    for key, value in stored["config"].items():
        if key in ("pullback", "substeps", "partners", "nodes", "device"):
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            argv += [flag] if value else []
        else:
            argv += [flag, str(value)]
    argv += ["--pullback", "virtual", "--substeps", virt.group(1), "--partners", virt.group(2)]
    argv += ["--nodes", "staged" if virt.group(3) else "post"]
    if virt.group(4):
        argv.append("--device")
    # the stored artifacts are the per-query fit's (the default until 2026-10-11)
    argv += ["--backward-basis", "quadratic_full"]
    argv += ["--seeds", str(seeds), "--output", out]
    if name.startswith("ensemble_couette_"):
        module = ensemble_couette
    elif name.startswith("ensemble_relax_"):
        module = ensemble_relaxation
    else:
        module = ensemble_inlet_shock
    return module, argv, stored, name


def main():
    path, seeds = sys.argv[1], int(sys.argv[2])
    pattern = re.compile(sys.argv[3] if len(sys.argv) > 3 else ".")
    files = sorted(glob.glob(os.path.join(RESULTS, "ensemble_*_virt*.json")))
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = os.path.basename(f)[:-5]
            if any(s in name for s in SKIP) or not pattern.search(name):
                continue
            out = os.path.join(tmp, f"{name}.json")
            module, argv, stored, name = argv_of(f, seeds, out)
            t0 = time.time()
            module.main(argv)
            with open(out, encoding="utf-8") as fh:
                fresh = json.load(fh)["rows"]
            rows = []
            for new in fresh:
                old = next(r for r in stored["rows"] if r["seed"] == new["seed"])
                rel = {
                    k: max(abs(a - b) / max(abs(b), 1e-300) for a, b in
                           zip(*(x if isinstance(x, list) else [x] for x in (new[k], old[k]))))
                    for k in ("objective", "gradient", "fd")
                }
                rows.append({"seed": new["seed"], "rel": rel,
                             "gradient": [new["gradient"], old["gradient"]]})
            rec = {"artifact": name, "argv": argv, "rows": rows, "seconds": time.time() - t0}
            worst = {k: max(r["rel"][k] for r in rows) for k in ("objective", "fd", "gradient")}
            print(f"{name:72s} " + " ".join(f"{k} {v:.1e}" for k, v in worst.items()), flush=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
