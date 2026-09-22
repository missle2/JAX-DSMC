#!/bin/bash
# The forward duplicates behind report.py: 4 seeds each of the circle and the sphere, the
# cell-walk tracker and the analytic body on seed 0.  GPU: ~1 min per circle, ~4 min per sphere.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for s in 0 1 2 3; do python3 examples/mesh/forward.py --case circle --seed $s; done
python3 examples/mesh/forward.py --case circle --seed 0 --tracker walk
python3 examples/mesh/forward.py --case circle --seed 0 --body analytic
for s in 0 1 2 3; do python3 examples/mesh/forward.py --case sphere --seed $s --snapshots; done
python3 examples/mesh/forward.py --case sphere --seed 0 --body analytic --snapshots
python3 examples/mesh/forward.py --case sphere --seed 0 --tracker walk --snapshots
