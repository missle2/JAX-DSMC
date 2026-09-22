#!/bin/bash
# P1-P3: P3 replication at scale on the JAX backend (GPU).  One log per job.
set -u
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
OUT=examples/p3_replication/results
LOG=scratch/bench/p3
mkdir -p $LOG
run() { # name, args...
  name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "skip $name"; return; fi
  echo "=== $name $(date +%T)"; python3 "$@" --output "$OUT/$name.json" > "$LOG/$name.log" 2>&1; echo "    exit=$? $(date +%T)"
}
SEEDS=${SEEDS:-16}
# P1 inlet Test 3: one-variable-at-a-time around N0=4000, retained 512, partners 8, neighbors 32
for rec in ballistic affine; do
  run p1_${rec}_N4k     -m examples.p3_replication --backend jax --reconstruction $rec --particles 4000  --retained 512  --source-samples 1995  --partners 8  --neighbors 32 --seeds $SEEDS --fd-step 0.1
done
run p1_ballistic_N16k   -m examples.p3_replication --backend jax --reconstruction ballistic --particles 16000 --retained 512  --source-samples 7979  --partners 8  --neighbors 32 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_R128   -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 128  --source-samples 1995  --partners 8  --neighbors 32 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_R2048  -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 2048 --source-samples 1995  --partners 8  --neighbors 32 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_P4     -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 512  --source-samples 1995  --partners 4  --neighbors 32 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_P16    -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 512  --source-samples 1995  --partners 16 --neighbors 32 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_K16    -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 512  --source-samples 1995  --partners 8  --neighbors 16 --seeds $SEEDS --fd-step 0.1
run p1_ballistic_K64    -m examples.p3_replication --backend jax --reconstruction ballistic --particles 4000  --retained 512  --source-samples 1995  --partners 8  --neighbors 64 --seeds $SEEDS --fd-step 0.1
: skip p1_ballistic_N64k   -m examples.p3_replication --backend jax --reconstruction ballistic --particles 64000 --retained 2048 --source-samples 31915 --partners 8  --neighbors 32 --seeds 8 --fd-step 0.1
# P2 Couette Test 4 and P3 heat Test 1 walls
for n in 2000 8000 32000; do
  run p2_couette_N$n -m examples.p3_replication.walls --case couette --localize --particles $n --seeds 32
done
run p3_heat_N8k -m examples.p3_replication.walls --case heat --localize --particles 8000 --seeds 32
