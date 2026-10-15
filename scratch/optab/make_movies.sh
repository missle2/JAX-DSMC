#!/bin/bash
# The 2026-10-14 movies: the optab A/B runs (300 iterates, every 6th shown, every 8th fill frame) and the cubic-fit
# squashkn2 runs, one CPU process at a time.  Run from the repository root.
set -u
L=/home/aakash/jax_dsmc/run_locked.sh
for t in ctl smooth4 smooth8 lsh0.125 lsh0.25; do
  echo "[$(date +%T)] optab ${t}"
  $L cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m examples.mesh.movie \
    --tag "circle_long_S4_adam_al_ab_${t}" --stride 6 --every 8 --fps 8 --mp4 || echo "FAILED ${t}"
done
for t in kn0.5_cell-cubicv-adapt1.5-align0.9-both_S4 kn0.5_cell-cubicv-adapt1.5-align0.9-both_S8 \
         kn0.5_M10_cell-cubicv-adapt1.5-align0.9-both_S2 kn0.5_M10_cell-cubicv-adapt1.5-align0.9-both_S4; do
  echo "[$(date +%T)] squash ${t}"
  $L cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m examples.mesh.movie --squash \
    --tag "ellipse_${t}_st0.35_adam" --stride 4 --every 2 --fps 8 --mp4 || echo "FAILED ${t}"
done
echo "[$(date +%T)] movies done"
