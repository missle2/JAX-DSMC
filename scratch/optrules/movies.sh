#!/bin/bash
# The per-run MP4s of the optrules arms (examples/mesh/movie.py): the steady flow of every
# shown iterate, then the shape step; ~60 iterates shown per run.  From the repository root.
set -euo pipefail
PY=/home/aakash/envs/jax-env/bin/python
for regime in kn0.5 kn0.5_M10; do
  for arm in adam tr trper lbfgs mma; do
    tag="circle_long_${regime}_S4_${arm}_tan_sm8_free_g32"
    n=$(ls "examples/mesh/results/optimize/${tag}" | grep -c '^frames_')
    stride=$(( (n + 59) / 60 ))
    echo "[$(date +%T)] ${tag}: ${n} iterates, stride ${stride}"
    PYTHONPATH=. "${PY}" -m examples.mesh.movie --tag "${tag}" --mp4 --every 100 --hold 2 \
      --morph 3 --stride "${stride}" --fps 8 | tail -1
  done
done
