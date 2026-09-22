#!/bin/bash
# Environment of every cluster job (CURC Alpine).  Sourced by job.sbatch and check_env.sbatch;
# edit here, not in the job scripts.
#
# GPU_KIND=h200 (default): NVIDIA H200 nodes (partition ah200; CUDA > 12.0 only, which
#                          the cuda12 wheels are) -- a venv of `pip install "jax[cuda12]"`:
#                          the wheels bring their own CUDA libraries and sm_90 kernels,
#                          only the node's driver matters.  Set up once on a login node
#                          (python >= 3.10; `module spider python` for the name):
#                            module load python
#                            python3 -m venv /projects/$USER/jax-env
#                            source /projects/$USER/jax-env/bin/activate
#                            pip install --upgrade pip
#                            pip install "jax[cuda12]" numpy scipy matplotlib
#                          (verified 2026-09-21: jax 0.6.2, CudaDevice, check_env ok;
#                          JAX_VENV overrides the path).
# GPU_KIND=a100:           the A100 nodes (partition aa100), same venv.
# GPU_KIND=mi100:          AMD MI100 nodes, `module load rocm/5.3.0; module load jax`.
#                          NOT usable as of 2026-09-21: that jax module is 0.4.7 and the
#                          code needs >= 0.4.36 (jax.tree.map, jax.random.key,
#                          register_dataclass(data_fields, meta_fields)); kept for the
#                          day CURC ships a ROCm >= 6.0 jax (or an Apptainer rocm/jax image).
export GPU_KIND="${GPU_KIND:-h200}"
module purge
if [ "${GPU_KIND}" = "mi100" ]; then
  module load rocm/5.3.0
  module load jax
else
  module load python 2>/dev/null || true   # the venv's interpreter is what runs
  # shellcheck disable=SC1091
  source "${JAX_VENV:-/projects/${USER}/jax-env}/bin/activate"
fi
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several seed workers may share a GPU
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"  # the WSL crash guard; harmless here
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
