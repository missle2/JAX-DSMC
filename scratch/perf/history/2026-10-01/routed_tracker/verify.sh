#!/bin/bash
# full verification of wt_track2: tests, gradient gate, regress, A/B timings vs wt_tally
T=/home/aakash/jax_dsmc/wt_track2
B=/home/aakash/jax_dsmc/wt_tally
M=/home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=$T/scratch/track2
cd $T
: > $O/v_tests.log
for g in "test_mesh_tracking.py test_mesh_geometry.py" "test_mesh_forward.py test_mesh_cut.py test_mesh_levelset.py" "test_mesh_adjoint.py" "test_walls.py test_paper_walls.py test_rng.py" "test_ensemble_backward.py test_ensemble_device.py test_ensemble_particle.py test_ensemble_inflow.py test_ensemble_inflow_jax.py" "test_forward_corrections.py test_inflow.py test_move.py test_geometry_corrections.py"; do
  args=""; for f in $g; do args="$args tests/$f"; done
  echo "== $g" >> $O/v_tests.log
  PYTHONPATH=. $L cpu python3 -m pytest -q $args 2>&1 | grep -E "passed|failed|error|FAILED|ERROR" >> $O/v_tests.log
done
echo TESTS_DONE >> $O/v_tests.log
cd $M
$L gpu python3 scratch/perf/noise_gate.py gradient --code $T --save track2 > $O/v_grad_run.log 2>&1
python3 scratch/perf/noise_gate.py gradient --compare base track2 > $O/v_grad.txt 2>&1
cd $T
PYTHONPATH=. $L gpu python3 scratch/perf/regress.py --compare batch_fwd --save track2 > $O/v_regress.log 2>&1
for tree in $B $T; do
  cd $tree
  PYTHONPATH=. $L gpu python3 scratch/perf/bench_forward.py > $O/v_bench_gpu_$(basename $tree).log 2>&1
  PYTHONPATH=. XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $L cpu taskset -c 5 python3 scratch/perf/bench_forward.py > $O/v_bench_cpu_$(basename $tree).log 2>&1
done
echo ALL_DONE > $O/v_done
