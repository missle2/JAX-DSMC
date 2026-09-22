"""Check that the stored benchmark artifacts were produced by the current implementation.

    JAX_PLATFORMS=cpu python3 -m examples.gradient_bench.provenance

Seed 0 of every pilot (the paper estimator on each case) is re-evaluated with the code as
it stands and compared with the stored row.  The objective checks the forward random
stream and physics; the gradient checks the estimator.  Agreement to round-off means the
figures built from those artifacts describe the current code; a mismatch names the study
that has to be re-run (or labelled with its provenance).
"""

import glob
import json
import os
import re
import sys
import tempfile

from examples import (
    ensemble_couette,
    ensemble_inlet_shock,
    ensemble_relaxation,
    gradient_convergence,
)
from examples.gradient_bench import RESULTS, studies

TOL = 1e-8
"""Objective: the forward must reproduce to round-off."""
GRAD_TOL = 1e-2
"""Gradient: the host backward's ridge became relative (1e-8 of the normal-matrix
diagonal) after the earlier ensemble artifacts were produced; that moves their gradients
by 1e-7 (Couette) to 3e-3 (inlet, through ~0.5 % of ghost queries whose scaled normal
matrix has condition number ~1e6 -- `scratch/bench/probe_ridge_sensitivity.py`), below the
statistical resolution of any study (>= 1 %)."""


def check(job, tmpdir):
    with open(job.path, encoding="utf-8") as fh:
        stored = json.load(fh)
    row = stored["rows"][0]
    out = os.path.join(tmpdir, f"{job.name}.json")
    argv = job.argv(0.0)
    argv[argv.index("--seeds") + 1] = "1"
    argv[argv.index("--output") + 1] = out
    if "--time" in argv:
        argv.remove("--time")
    gradient_convergence.main(argv)
    with open(out, encoding="utf-8") as fh:
        fresh = json.load(fh)["rows"][0]
    rel = {k: abs(fresh[k] - row[k]) / max(abs(row[k]), 1e-300) for k in ("objective", "gradient")}
    ok = rel["objective"] < TOL and rel["gradient"] < GRAD_TOL
    return ok, rel, stored.get("backend"), row, fresh


def _ensemble_argv(path, tmpdir):
    """Rebuild the command line of an ensemble artifact from its stored config and name."""
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    name = os.path.basename(path)[:-5]
    argv = []
    for key, value in stored["config"].items():
        if key in ("pullback", "substeps", "partners"):
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key.replace('_', '-')}")
        else:
            argv += [f"--{key.replace('_', '-')}", str(value)]
    if name.endswith("_dev"):
        argv.append("--device")
        name = name[:-4]
    virt = re.search(r"_virt(\d+)x(\d+)(st)?$", name)
    if virt:
        argv += ["--pullback", "virtual", "--substeps", virt.group(1), "--partners", virt.group(2)]
        argv += ["--nodes", "staged" if virt.group(3) else "post"]
    elif name.endswith("_lin"):
        argv += ["--pullback", "linearised"]
    else:
        argv += ["--pullback", "sequential"]  # untagged artifacts predate section 5.12
    argv += ["--seeds", "1", "--output", os.path.join(tmpdir, f"{name}.json")]
    if name.startswith("ensemble_couette_"):
        module = ensemble_couette
    elif name.startswith("ensemble_relax_"):
        module = ensemble_relaxation
    else:
        module = ensemble_inlet_shock
    return module, argv, stored


def check_ensemble(path, tmpdir):
    module, argv, stored = _ensemble_argv(path, tmpdir)
    module.main(argv)
    with open(argv[-1], encoding="utf-8") as fh:
        fresh = json.load(fh)["rows"][0]
    row = stored["rows"][0]
    rel = {k: abs(fresh[k] - row[k]) / max(abs(row[k]), 1e-300) for k in ("objective", "gradient")}
    return rel["objective"] < TOL and rel["gradient"] < GRAD_TOL, rel, row, fresh


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for path in sorted(glob.glob(os.path.join(RESULTS, "ensemble_*.json"))):
            name = os.path.basename(path)[:-5]
            ok, rel, row, fresh = check_ensemble(path, tmpdir)
            tag = "ok" if ok else "CHANGED"
            print(
                f"{'ensemble':24s} {name:44s} {tag:8s} "
                f"objective rel {rel['objective']:.1e} gradient rel {rel['gradient']:.1e}"
            )
            if not ok:
                failures.append((name, row, fresh))
        for study, jobs in studies().items():
            pilots = [j for j in jobs if j.seed_start == 0]
            for job in pilots:
                if not os.path.exists(job.path):
                    print(f"{study:24s} {job.name:36s} missing artifact")
                    continue
                ok, rel, backend, row, fresh = check(job, tmpdir)
                tag = "ok" if ok else "CHANGED"
                print(
                    f"{study:24s} {job.name:36s} {tag:8s} stored on {backend}: "
                    f"objective rel {rel['objective']:.1e} gradient rel {rel['gradient']:.1e}"
                )
                if not ok:
                    failures.append((job.name, row, fresh))
    for name, row, fresh in failures:
        print(f"  {name}: stored J={row['objective']!r} g={row['gradient']!r}")
        print(f"  {' ' * len(name)}  fresh  J={fresh['objective']!r} g={fresh['gradient']!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
