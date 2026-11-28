#!/bin/bash
# nsys from $NSYS, the CUDA toolkit SPARTA/JAX use (CUDA_ROOT; Alpine: /projects/$USER/cuda-12.8),
# a cudatoolkit/nsight module, or PATH -- then exec it with the arguments (bench/profile_plan.py).
for c in "${NSYS:-}" "${CUDA_ROOT:-/nonexistent}"/nsight-systems-*/bin/nsys "${CUDA_ROOT:-/nonexistent}"/bin/nsys \
         "${CUDA_HOME:-/nonexistent}"/bin/nsys "$(command -v nsys 2>/dev/null)"; do
  [ -n "${c}" ] && [ -x "${c}" ] && exec "${c}" "$@"
done
echo "nsys not found (set NSYS=/path/to/nsys; Alpine: the CUDA 12.8 runfile's nsight-systems)" >&2
exit 127
