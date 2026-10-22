#!/bin/bash
# One job of the 2026-10-22 cluster speed A/B (examples/mesh/cluster/study.py --group
# speedprof), from the repository root:  bash scratch/shapefast/cluster_job.sh <job>
# Output scratch/shapefast/cluster/<job>_<setting>.json (+ _grad.npy) per setting, one
# process each (XLA reads its flags at start-up); a setting = XLA command buffers x the shape
# VJP's batched steps (SHAPE_BATCH, ensemble_mesh's shape_batch):
#   off_b1      --xla_gpu_enable_command_buffer= (the old env.sh), steps one at a time
#   off_b4      the same flags, 4 steps at a time (the shipped shape_batch)
#   default_b4  XLA's own command buffers (env.sh since 2026-10-22), 4 at a time
#   default_b12 the same, 12 at a time (memory: peak_gb in the json)
#   while_b4    the default kinds plus WHILE and CONDITIONAL (whole loops as CUDA graphs)
# A setting that fails (an XLA error, a crash, out of memory) is recorded and the next runs.
# SETTINGS (space-separated, default all five) picks a subset, e.g. the quick check of the
# new defaults against the old: SETTINGS="off_b1 default_b4" (passed through by sbatch).
#   m10_it156  the optm Kn 0.5 M 10 run's iterate 156 (its own config)
#   kn05_it200 the optrules1seed Kn 0.5 Adam run's iterate 200
set -uo pipefail
job="$1"
out=scratch/shapefast/cluster
case "${job}" in
  m10_it156) run=examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_adam_tan_sm8_free; it=156 ;;
  kn05_it200) run=examples/mesh/results/optimize/circle_long_kn0.5_S1_adam_tan_sm8_free_g32_w500; it=200 ;;
  *) echo "unknown job ${job}"; exit 1 ;;
esac
[ -f "${run}/coef_${it}.npy" ] || { echo "missing ${run}/coef_${it}.npy"; exit 1; }
[ "${CHECK:-0}" = 1 ] && exit 0
mkdir -p "${out}"
base="FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN"
rc=0
for setting in ${SETTINGS:-off_b1 off_b4 default_b4 default_b12 while_b4}; do
  case "${setting%_b*}" in
    off) xla="--xla_gpu_enable_command_buffer=" ;;
    default) xla="" ;;
    while) xla="--xla_gpu_enable_command_buffer=${base},CONDITIONAL,WHILE" ;;
  esac
  batch="${setting##*_b}"
  echo "[$(date)] ${job} ${setting}: XLA_FLAGS='${xla}' SHAPE_BATCH=${batch}"
  XLA_FLAGS="${xla}" SHAPE_BATCH="${batch}" python3 scratch/shapefast/iterate_time.py \
    "${out}/${job}_${setting}.json" "${run}" "${it}" || { echo "${setting} failed"; rc=1; }
done
exit ${rc}
