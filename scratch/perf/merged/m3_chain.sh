#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/perf/merged
( $L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/JAX-DSMC --save merged3 > $O/m3_fwd_run.log 2>&1
  python3 scratch/perf/noise_gate.py forward --compare base merged3 > $O/m3_fwd.txt 2>&1
  $L gpu python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/JAX-DSMC --save merged3 > $O/m3_grad_run.log 2>&1
  python3 scratch/perf/noise_gate.py gradient --compare base merged3 > $O/m3_grad.txt 2>&1
  echo DONE > $O/m3_gpu_done ) &
G=(
"test_mesh_tracking.py test_mesh_geometry.py test_ntc.py"
"test_mesh_levelset.py test_mesh_cut.py test_mesh_forward.py test_ntc_reference.py"
"test_mesh_adjoint.py"
"test_walls.py test_paper_walls.py test_forward_corrections.py test_inflow.py"
"test_ensemble_backward.py test_ensemble_device.py test_ensemble_particle.py"
"test_ensemble_inflow.py test_ensemble_inflow_jax.py test_gaps.py"
"test_geometry_corrections.py test_move.py test_state.py test_rng.py test_turbulence_gates.py"
"test_adjoint.py test_adjoint_oracle.py test_bkw.py test_checkpoint_blocks.py test_couette.py test_decomp.py"
"test_gradient_validation.py test_initialize.py test_p2_replication.py test_perf.py test_perf_budget.py"
"test_safe.py test_sample.py test_stochastic_reference.py test_trajectory_scores.py test_vss.py"
)
: > $O/m3_tests.log
for g in "${G[@]}"; do
  args=""; for f in $g; do args="$args tests/$f"; done
  echo "== $g" >> $O/m3_tests.log
  PYTHONPATH=. $L cpu python3 -m pytest -q $args 2>&1 | grep -E "passed|failed|error|FAILED|ERROR" >> $O/m3_tests.log
done
echo TESTS_DONE >> $O/m3_tests.log
wait
