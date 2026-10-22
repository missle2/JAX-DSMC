#!/bin/bash
# Environment of every cluster job (CURC Alpine).  Sourced by job.sbatch and check_env.sbatch;
# edit here, not in the job scripts.
#
# GPU_KIND=a100 (default): the A100 nodes (partition aa100), 40 GB, the venv below.
# GPU_KIND=h200:           NVIDIA H200 nodes (partition ah200; CUDA > 12.0 only, which
#                          the cuda12 wheels are) -- a venv of `pip install "jax[cuda12]"`:
#                          the wheels bring their own CUDA libraries and sm_90 kernels,
#                          only the node's driver matters.  Set up once on a login node
#                          (python >= 3.10; `module spider python` for the name):
#                            module load python
#                            python3 -m venv /projects/$USER/jax-env
#                            source /projects/$USER/jax-env/bin/activate
#                            pip install --upgrade pip
#                            pip install "jax[cuda12]" numpy scipy matplotlib optax
#                          (verified 2026-09-21: jax 0.6.2, CudaDevice, check_env ok;
#                          JAX_VENV overrides the path).
# GPU_KIND=mi100:          AMD MI100 nodes, `module load rocm/5.3.0; module load jax`.
#                          NOT usable as of 2026-09-21: that jax module is 0.4.7 and the
#                          code needs >= 0.4.36 (jax.tree.map, jax.random.key,
#                          register_dataclass(data_fields, meta_fields)); kept for the
#                          day CURC ships a ROCm >= 6.0 jax (or an Apptainer rocm/jax image).
export GPU_KIND="${GPU_KIND:-a100}"  # submit_curc.sh passes its own (default a100)
# A job submitted from a shell with the venv active (`source .../activate`) inherits
# VIRTUAL_ENV and the loaded python module (sbatch --export=ALL) but not the venv's shell
# function `deactivate`, which `module purge` then calls: the optax1seed jobs of 2026-10-22
# printed "deactivate: command not found" ~600 MB deep and never got past this file.  Drop
# the inherited venv and give purge a no-op `deactivate`; the `activate` below defines the
# real one.  (submit_curc.sh strips the venv too.)
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(echo "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONHOME
fi
declare -F deactivate > /dev/null || deactivate() { :; }
export -f deactivate
module purge
if [ "${GPU_KIND}" = "mi100" ]; then
  module load rocm/5.3.0
  module load jax
else
  module load python 2>/dev/null || true   # the venv's interpreter is what runs
  # shellcheck disable=SC1091
  source "${JAX_VENV:-/projects/${USER}/jax-env}/bin/activate"
  export -f deactivate   # so child shells that call it (module commands) find it too
fi
export JAX_ENABLE_X64=1
# Pin the backend.  Without this JAX falls back to the CPU when a node's driver refuses
# cuInit and the job runs to completion at a quarter of the speed (2026-09-22: every
# ah200 job landed on one broken node, `cuInit(0) failed: CUDA_ERROR_UNKNOWN`, and burned
# 2 h per seed on the CPU).  job.sbatch's preflight turns that into an immediate exit.
if [ "${GPU_KIND}" = "mi100" ]; then export JAX_PLATFORMS=rocm; else export JAX_PLATFORMS=cuda; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several seed workers may share a GPU
# XLA's command buffers (CUDA graphs: a run of kernels launched as one) are ON here, XLA's
# default set.  Until 2026-10-22 this file switched them off (`--xla_gpu_enable_command_buffer=`),
# a copy of the laptop's WSL crash guard (run_locked.sh) that the cluster never needed; the
# shape step and the backward are hundreds of thousands of microsecond kernels, launch-bound
# (scratch/shapefast; the speedprof group measures both settings and the WHILE/CONDITIONAL
# kinds).  The flags are in the compilation cache's key: the first jobs after the switch
# compile afresh.  Off again for one submission:
#   XLA_FLAGS=--xla_gpu_enable_command_buffer= bash examples/mesh/cluster/submit_curc.sh ...
export XLA_FLAGS="${XLA_FLAGS:-}"
export PYTHONUNBUFFERED=1
# Persistent compilation cache, shared by every job on the shared /projects filesystem: a
# job whose programs were compiled before (same shapes, flags, jax version and device kind --
# all in the cache key, so H200 / A100 / RTX entries never mix) loads them instead of
# compiling (2026-10-11: ~25 min to the first output on rtx, much of it compiling).  Seeds of
# one setting, resubmitted optimisers and later queue waves hit; jobs starting together all
# compile.  No size limit, so no lock file (a limit makes JAX lock every access, and a job
# killed holding the lock blocks the cache); a torn entry from two concurrent writers fails
# to decompress, which JAX reports as a warning and compiles instead.  Clear it after a jax
# upgrade: rm -r /projects/$USER/jax-cache.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/projects/${USER}/jax-cache}"
# The repository root on the import path.  `python3 -m examples...` puts the working
# directory there itself, but a plain `python3 scratch/<dir>/<script>.py` puts only the
# script's own directory: every biasgate job of 2026-10-04 died on `import jax_dsmc`
# (ModuleNotFoundError) for that reason.  job.sbatch runs from the repository root.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-${PWD}}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
