#!/bin/bash
# One benchmark job: runs the lines of a plan file one after another on this allocation.
# Resources, time and job name come from bench/matrix.py on the sbatch command line; the plan
# from the environment:  sbatch <resources> --export=ALL,PLATFORM=pm-gpu,PLAN=<file> bench.sbatch
#
# Plan lines are shell commands (run_bench.py invocations, with any per-run environment
# assignments in front).  A failing line is reported and the job goes on to the next: one
# out-of-memory size must not cost the rest of the job.  `# ...` lines are comments.
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${PLAN:?PLAN (plan file) not set}"
: "${PLATFORM:?PLATFORM not set}"
cd "${SLURM_SUBMIT_DIR}"
# A job submitted from inside another allocation (e.g. an acompile shell) inherits that
# allocation's memory setting (sbatch --export=ALL); with the job's own one srun then refuses
# every step: "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually
# exclusive" (Alpine jobs 36502982-94, 2026-11-27).  Steps take the job's memory (srun --mem=0
# in pf_launch), so none of these is needed.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
# shellcheck disable=SC1091
source bench/env/runtime.sh
echo "[$(date -Iseconds)] job ${SLURM_JOB_ID} ${SLURM_JOB_NAME}: ${SLURM_JOB_NUM_NODES} node(s) $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -3 | paste -sd, -)..."
echo "plan ${PLAN}: $(grep -cv '^\s*\(#\|$\)' "${PLAN}") runs"
n=0; fail=0
while IFS= read -r line || [ -n "${line}" ]; do
  case "${line}" in ''|'#'*) continue ;; esac
  n=$((n + 1))
  echo "[$(date -Iseconds)] run ${n}: ${line}"
  t0=$(date +%s)
  if ! bash -c "${line}"; then
    fail=$((fail + 1)); echo "[$(date -Iseconds)] run ${n} FAILED"
  fi
  echo "[$(date -Iseconds)] run ${n} done in $(( $(date +%s) - t0 )) s"
done < "${PLAN}"
echo "[$(date -Iseconds)] ${n} runs, ${fail} failed"
