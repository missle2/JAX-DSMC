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
# Likewise whatever runtime.sh exported in the submitting shell for ANOTHER platform: a pm-cpu
# job submitted after `source runtime.sh` for pm-gpu kept JAX_PLATFORMS=cuda, and a CPU-only
# jaxlib then has no backend at all ("assert _default_backend is not None", job 62131098,
# 2026-11-27).  runtime.sh below sets each of these again for ${PLATFORM}.
unset JAX_PLATFORMS JAX_VENV XLA_FLAGS XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION \
  JAX_COMPILATION_CACHE_DIR MACHINE SITE GPUS_PER_NODE CORES_PER_NODE SMT GRES_COUNT RESULTS_DIR \
  SPARTA_LIBPATH OMP_NUM_THREADS OMP_PLACES OMP_PROC_BIND MPICH_GPU_SUPPORT_ENABLED \
  NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC FI_PROVIDER FI_MR_CACHE_MONITOR \
  FI_CXI_DISABLE_HOST_REGISTER KOKKOS_ARCH JAX_FLAVOR MODULES QOS CONSTRAINT
if [ -n "${VIRTUAL_ENV:-}" ]; then  # the submitting shell's venv off PATH
  PATH="$(echo "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"; unset VIRTUAL_ENV
fi
# no core files: an aborting run (MI100 SPARTA sphere, job 36514785) wrote one per rep into the
# submit directory, GBs each
ulimit -c 0
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
  # </dev/null: srun (inside the line) forwards the job's stdin to its tasks and so reads the
  # rest of the plan file -- every job stopped after run 1 with "1 runs, 0 failed" (Alpine
  # jobs 36503311-60, 2026-11-27)
  if ! bash -c "${line}" < /dev/null; then
    fail=$((fail + 1)); echo "[$(date -Iseconds)] run ${n} FAILED"
  fi
  echo "[$(date -Iseconds)] run ${n} done in $(( $(date +%s) - t0 )) s"
done < "${PLAN}"
echo "[$(date -Iseconds)] ${n} runs, ${fail} failed"
# a check job is the gate of the jobs submitted --after it (afterok): any failed line fails it
case "${SLURM_JOB_NAME}" in check-*) [ "${fail}" = 0 ] || exit 1 ;; esac
