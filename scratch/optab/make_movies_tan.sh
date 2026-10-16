#!/bin/bash
# The 2026-10-15 movies: the tangent-mode optab A/B runs (first ~110-129 iterates, every 3rd shown,
# every 8th fill frame), one CPU process at a time.  Run from the repository root.
set -u
L=/home/aakash/jax_dsmc/run_locked.sh
for t in ctl smooth4 smooth8 lsh0.125 lsh0.25; do
  echo "[$(date +%T)] optab tan ${t}"
  $L cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m examples.mesh.movie \
    --tag "circle_long_S4_adam_tan_ab_${t}" --stride 3 --every 8 --fps 8 --mp4 || echo "FAILED ${t}"
done
echo "[$(date +%T)] movies done"
