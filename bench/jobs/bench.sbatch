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
