#!/usr/bin/env python3
"""Profile one timestep's worth of work of both codes under Nsight Systems, on one GPU.

    python3 bench/profile_plan.py --platform alp-h200 [--particles 4M] [--submit]

For collide and sphere (and free), SPARTA (headline build + GPU protocol) and sparta_jax (the
platform's tuned per-deck settings) run 200 timed steps under `nsys profile -t cuda,nvtx`, and
`nsys stats` turns each report into per-kernel and per-memcpy CSV summaries:
    ${RESULTS_DIR}/profile/<code>_<deck>_<size>.nsys-rep
    ${RESULTS_DIR}/profile/<code>_<deck>_<size>_{cuda_gpu_kern_sum,cuda_gpu_mem_time_sum}.csv
The time per step of each kernel family is then compared against SPARTA's own timer sections
(Move/Coll/Sort/Output) and sparta_jax's stages, and against the memory traffic each needs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix  # noqa: E402

NSYS = "bash ${PWD}/bench/env/nsys.sh"  # finds nsys; absolute (SPARTA runs from bench/decks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", required=True)
    p.add_argument("--particles", default="4M")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--decks", nargs="+", default=["free", "collide", "sphere"])
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    pf = matrix.PLATFORMS[a.platform]
    if not pf["gpu"] or pf.get("vendor", "nvidia") == "amd" or "mi100" in a.platform:
        sys.exit("NVIDIA GPU platforms only (nsys)")
    n = int(float(a.particles[:-1]) * {"K": 1e3, "M": 1e6}[a.particles[-1]])
    cfg = matrix.best_config(a.platform)
    out = '"${RESULTS_DIR}/profile"'
    lines = [f"mkdir -p {out}; {NSYS} --version"]
    for deck in a.decks:
        for code in ("sparta", "jax"):
            r = matrix.mk_run(a.platform, code, deck, n, 1, 1, cfg)
            r.reps, r.steps, r.est_s = 1, a.steps, 600.0
            tag = f"{code}_{deck}_{a.particles}"
            rep = f"${{RESULTS_DIR}}/profile/{tag}"
            prof = f"{NSYS} profile -t cuda,nvtx --force-overwrite true -o {rep}"
            line = matrix.plan_line(r, a.platform)
            line = line.replace(f"--variant {shlex.quote(r.variant)}", f"--variant prof_{r.variant}")
            line = line.replace('--out "${RESULTS_DIR}"', f"--out {out}")
            if code == "sparta":  # the profiler goes between srun and spa
                line = line.replace('--launch "$(pf_launch 1 1 bind)"', f"--launch \"$(pf_launch 1 1 bind) {prof}\"")
            else:
                line = line.replace("$(pf_launch 1 1 bind) python3", f"$(pf_launch 1 1 bind) {prof} python3")
            lines.append(line)
            lines.append(f"{NSYS} stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "
                         f"--format csv --output {rep} {rep}.nsys-rep > /dev/null && ls {rep}_*.csv")
    plan_dir = os.path.join(matrix.ROOT, "bench", "plans", a.platform)
    os.makedirs(plan_dir, exist_ok=True)
    plan = os.path.join(plan_dir, f"profile-{a.platform}.plan")
    with open(plan, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"plan {plan}: {len(a.decks) * 2} profiled runs")
    if a.submit:
        res = matrix.sbatch_args(a.platform, 1, 1)
        os.makedirs(os.path.join(matrix.ROOT, "joblogs"), exist_ok=True)
        cmd = ["sbatch", *shlex.split(res), "--time=00:45:00", f"--job-name=profile-{a.platform}",
               f"--output={os.path.join(matrix.ROOT, 'joblogs', '%x_%j.out')}",
               f"--export=ALL,PLATFORM={a.platform},PLAN={plan}", os.path.join(HERE, "jobs", "bench.sbatch")]
        print("+", " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
