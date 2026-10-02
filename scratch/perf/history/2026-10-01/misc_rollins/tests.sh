#!/bin/bash
cd /home/aakash/jax_dsmc/wt_misc
L=/home/aakash/jax_dsmc/run_locked.sh
O=scratch/misc/tests.log
: > $O
for g in "test_ntc.py test_ntc_reference.py test_vss.py test_sample.py test_rng.py test_state.py" "test_ensemble_backward.py test_ensemble_device.py test_ensemble_particle.py test_ensemble_inflow.py test_ensemble_inflow_jax.py" "test_mesh_forward.py test_mesh_tracking.py" "test_mesh_adjoint.py" "test_forward_corrections.py test_inflow.py test_trajectory_scores.py test_perf_budget.py test_perf.py" "test_adjoint.py test_adjoint_oracle.py test_bkw.py test_checkpoint_blocks.py test_couette.py test_decomp.py" "test_gaps.py test_geometry_corrections.py test_move.py test_turbulence_gates.py test_gradient_validation.py test_initialize.py test_p2_replication.py test_safe.py test_stochastic_reference.py test_walls.py test_paper_walls.py"; do
  args=""; for f in $g; do args="$args tests/$f"; done
  echo "== $g" >> $O
  PYTHONPATH=. $L cpu python3 -m pytest -q $args 2>&1 | grep -E "passed|failed|error|FAILED|ERROR" >> $O
done
echo TESTS_DONE >> $O
