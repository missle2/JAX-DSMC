#!/bin/bash
# nsys (Nsight Systems CLI), newest found, then exec it with the arguments (bench/profile_plan.py).
# Looks in: $NSYS; the CUDA toolkit SPARTA uses (CUDA_ROOT; Alpine /projects/$USER/cuda-12.8 --
# the runfile puts it in <root>/nsight-systems-<ver>/bin); Alpine's cuda module installs; NERSC's
# cudatoolkit (CUDA_HOME / nsys on PATH); /opt/nvidia.  (2026-11-28: CUDA_ROOT was not exported
# by platforms.sh, so the first profile jobs found nothing.)
cands=("${NSYS:-}")
for root in "${CUDA_ROOT:-}" "/projects/${USER}/cuda-12.8" "${CUDA_HOME:-}" /curc/sw/install/cuda/* /opt/nvidia; do
  [ -n "${root}" ] && [ -d "${root}" ] || continue
  cands+=("${root}"/nsight-systems-*/bin/nsys "${root}"/nsight-systems/*/bin/nsys "${root}"/bin/nsys)
done
cands+=("$(command -v nsys 2>/dev/null)")
best=""
for c in "${cands[@]}"; do
  [ -n "${c}" ] && [ -x "${c}" ] || continue
  best="${c}"; [ -n "${NSYS:-}" ] && break  # an explicit NSYS wins
done
# newest by path version (nsight-systems-2024.6.2 > 2023.1.2)
if [ -z "${NSYS:-}" ]; then
  best="$(printf '%s\n' "${cands[@]}" | while read -r c; do [ -n "${c}" ] && [ -x "${c}" ] && echo "${c}"; done | sort -V | tail -n 1)"
fi
if [ -z "${best}" ]; then
  echo "nsys not found under: ${CUDA_ROOT:-?} /projects/${USER}/cuda-12.8 /curc/sw/install/cuda/* (set NSYS=/path/to/nsys)" >&2
  exit 127
fi
[ "${1:-}" = "--version" ] && echo "nsys: ${best}"
exec "${best}" "$@"
