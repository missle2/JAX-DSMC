#!/bin/bash
# The movies of the optm runs (free boundary, smooth 8, S4 Adam), one CPU process at a time.
# 2026-10-16: kn0.5 (236 iterates) and M10 (300) every 3rd iterate, kn0.5_M10 (68) every iterate.
# 2026-10-17: kn0.5 (300) every 3rd, kn0.5_M10 (247) every 2nd; M10 unchanged (done at 300).
# Arguments: "<run>_ <stride>" pairs (default: all three as on 10-16).  Run from the repository root.
set -u
L=/home/aakash/jax_dsmc/run_locked.sh
[ $# -gt 0 ] || set -- "kn0.5_ 3" "M10_ 3" "kn0.5_M10_ 1"
for spec in "$@"; do
  set -- $spec
  t="circle_long_${1}S4_adam_tan_sm8_free"
  echo "[$(date +%T)] ${t}"
  $L cpu env JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m examples.mesh.movie \
    --tag "$t" --stride "$2" --every 8 --fps 8 --mp4 || echo "FAILED ${t}"
done
echo "[$(date +%T)] movies done"
