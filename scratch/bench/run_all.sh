#!/bin/bash
# Overnight chain: forward gate F2, wall-cost B6 (GPU, both worktrees), gradient bench, P3 scale-up.
# One log per stage under scratch/bench; a failing stage does not stop the chain.
set -u
B=/home/aakash/jax_dsmc/JAX-DSMC-adjoint
M=/home/aakash/jax_dsmc/JAX-DSMC
cd $B
stage() { echo "=== $1  $(date '+%F %T')"; }
stage F2; python3 -m examples.shock.inflow_report > scratch/bench/F2_inflow_shock.log 2>&1; echo "    exit=$?"
stage B6-gpu-branch; PYTHONPATH=$B python3 scratch/bench/b6_wall_cost.py --n 40000 --steps 100 --dims 1,2,3 --bounces 3,8 --label branch-gpu --out scratch/bench/B6_branch_gpu.json > scratch/bench/B6_branch_gpu.log 2>&1; echo "    exit=$?"
stage B6-gpu-main; (cd $M && PYTHONPATH=$M python3 $B/scratch/bench/b6_wall_cost.py --n 40000 --steps 100 --dims 1,2,3 --bounces 3 --label main-gpu --out $B/scratch/bench/B6_main_gpu.json > $B/scratch/bench/B6_main_gpu.log 2>&1); echo "    exit=$?"
for study in triad_speed triad_temperature_left inlet_temp inlet_nrho scaling majorant shockbl_diam shockbl_omega inlet2_temp; do
  stage bench-$study; python3 -m examples.gradient_bench --study $study --seeds-scale 0.75 > scratch/bench/bench_$study.log 2>&1; echo "    exit=$?"
done
stage P3; SEEDS=16 bash scratch/bench/run_p3.sh > scratch/bench/P3_chain.log 2>&1; echo "    exit=$?"
stage report; python3 -m examples.gradient_bench.report > scratch/bench/bench_report.log 2>&1; echo "    exit=$?"
stage DONE
