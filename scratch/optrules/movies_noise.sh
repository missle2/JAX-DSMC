#!/bin/bash
# Per-run MP4s of the optrules2 / optrules1seed runs (2026-10-21; examples/mesh/movie.py): the
# steady flow of every shown iterate, then the shape step; ~60 iterates shown per run.
set -euo pipefail
PY=/home/aakash/envs/jax-env/bin/python
for regime in kn0.5 kn0.5_M10; do
  for arm in S4_tr2 S1_adam S1_tr S1_lbfgs S1_mma; do
    tag="circle_long_${regime}_${arm}_tan_sm8_free_g32_w500"
    n=$(ls "examples/mesh/results/optimize/${tag}" | grep -c '^frames_')
    stride=$(( (n + 59) / 60 ))
    echo "[$(date +%T)] ${tag}: ${n} iterates, stride ${stride}"
    PYTHONPATH=. "${PY}" -m examples.mesh.movie --tag "${tag}" --mp4 --every 100 --hold 2 \
      --morph 3 --stride "${stride}" --fps 8 | tail -1
  done
done
