#!/bin/bash
# Per-run MP4s of every one-seed optimiser run (2026-10-23: the optax1seed arms and the
# optrules1seed ones, re-rendered at their final length; examples/mesh/movie.py at dpi 130):
# the steady flow of every shown iterate, then the shape step; ~60 iterates shown per run.
set -euo pipefail
PY=/home/aakash/envs/jax-env/bin/python
for regime in kn0.5 kn0.5_M10; do
  for arm in ox_adam ox_adamcos ox_ademamix ox_prodigy ox_sfadam adam tr lbfgs mma; do
    tag="circle_long_${regime}_S1_${arm}_tan_sm8_free_g32_w500"
    n=$(ls "examples/mesh/results/optimize/${tag}" | grep -c '^frames_')
    stride=$(( (n + 59) / 60 ))
    echo "[$(date +%T)] ${tag}: ${n} iterates, stride ${stride}"
    PYTHONPATH=. "${PY}" -m examples.mesh.movie --tag "${tag}" --mp4 --every 100 --hold 2 \
      --morph 3 --stride "${stride}" --fps 8 --dpi 130 | tail -1
  done
done
