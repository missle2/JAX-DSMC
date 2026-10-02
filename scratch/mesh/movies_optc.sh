#!/bin/bash
# render the optc movies of the given tags in turn; stride = ceil(iterates / 85)
for tag in "$@"; do
  n=$(python3 -c "import json;print(len(json.load(open('examples/mesh/results/optimize/$tag/history.json'))['rows']))")
  stride=$(( (n + 84) / 85 ))
  echo "[$(date +%T)] $tag: $n iterates, stride $stride"
  JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m examples.mesh.movie --tag "$tag" --every 4 --hold 3 --morph 5 --fps 12 --stride "$stride" --mp4 2>&1 | grep -v "frames so far"
done
