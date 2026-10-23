#!/bin/bash
# The optimiser comparison's outputs after the optcm runs (2026-10-23), serially in one CPU
# slot: the combined movies first, then the figures, then the per-run movies of the
# centroid-fixed arms.
set -euo pipefail
PY=/home/aakash/envs/jax-env/bin/python
for regime in kn0.5_M10 kn0.5; do
  "${PY}" scratch/optax/compare_movies.py "${regime}" --only all
done
for regime in kn0.5_M10 kn0.5; do
  "${PY}" scratch/optax/compare_movies.py "${regime}" --only present
done
"${PY}" scratch/optax/figures.py
for regime in kn0.5 kn0.5_M10; do
  for arm in ox_adam_cm ox_adamcos_cm ox_prodigy_cm ox_prodigycos_cm; do
    tag="circle_long_${regime}_S1_${arm}_tan_sm8_free_g32_w500"
    n=$(ls "examples/mesh/results/optimize/${tag}" | grep -c '^frames_')
    stride=$(( (n + 59) / 60 ))
    echo "[$(date +%T)] ${tag}: ${n} iterates, stride ${stride}"
    "${PY}" -m examples.mesh.movie --tag "${tag}" --mp4 --every 100 --hold 2 \
      --morph 3 --stride "${stride}" --fps 8 --dpi 130 | tail -1
  done
done
echo done
