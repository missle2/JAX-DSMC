#!/usr/bin/env bash
# Interleaved A/B of jax_dsmc/<file>: alternates the two versions ROUNDS times, one-core
# polygon forward (bench_forward.py, best of 2 repeats each).  Restores the B version.
#   bash scratch/xla_ops/ab.sh jax_dsmc/compaction.py old.py new.py 3
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC
target=$1; a=$2; b=$3; rounds=${4:-3}; body=${BODY:-polygon}
for r in $(seq "$rounds"); do
  for v in "$a" "$b"; do
    cp "$v" "$target"
    s=$(JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false PYTHONPATH=. taskset -c 3 \
        python3 scratch/perf/bench_forward.py --bodies "$body" --repeats 2 2>/dev/null \
        | tail -1 | python3 -c "import json,sys; print(round(json.loads(sys.stdin.read())['seconds'],3))")
    echo "round $r $(basename "$v") $s"
  done
done
cp "$b" "$target"
