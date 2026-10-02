#!/usr/bin/env bash
# SPARTA's circle_ar (examples/mesh/sparta/in.circle_ar, 1000 steps, ~37 M particle moves)
# on this machine's CPUs: the serial build and the MPI build (no Kokkos) at 1..16 ranks,
# 3 repeats each; loop time and particle moves from the log -> scratch/perf/bench_sparta.txt
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../examples/mesh/sparta
serial=$HOME/sparta/sparta/build/src/spa_serial
mpi=$HOME/sparta/sparta/build_mpi/src/spa_
work=$(mktemp -d)
cp "$src"/* "$work"/
cd "$work"
out=$here/bench_sparta.txt
echo "build ranks repeat loop_seconds moves particles" > "$out"
for r in 1 2 3; do
  "$serial" -var seed 12345 -in in.circle_ar -log log.s -screen none
  echo "serial 1 $r $(awk '/^Loop time/{print $4}' log.s) $(awk '/^Particle moves/{print $4}' log.s) $(awk '/^Loop time/{print $(NF-1)}' log.s)" >> "$out"
  for np in 1 2 4 8 16; do
    mpirun --oversubscribe -np $np "$mpi" -var seed 12345 -in in.circle_ar -log log.m -screen none
    echo "mpi $np $r $(awk '/^Loop time/{print $4}' log.m) $(awk '/^Particle moves/{print $4}' log.m) $(awk '/^Loop time/{print $(NF-1)}' log.m)" >> "$out"
  done
done
rm -rf "$work"
cat "$out"
