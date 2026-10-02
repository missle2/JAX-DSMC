#!/bin/bash
# the tracker gate's test modules, one pytest process per module, serialised
cd /home/aakash/jax_dsmc/wt_track2
out=scratch/track2/tests_$1.log
: > $out
for m in tests/test_mesh_geometry.py tests/test_mesh_forward.py tests/test_mesh_cut.py tests/test_mesh_levelset.py tests/test_walls.py tests/test_paper_walls.py tests/test_ensemble_backward.py tests/test_ensemble_device.py tests/test_ensemble_particle.py tests/test_ensemble_inflow.py tests/test_ensemble_inflow_jax.py tests/test_rng.py tests/test_mesh_adjoint.py; do
  echo "== $m" >> $out
  PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh cpu python3 -m pytest -q $m 2>&1 | grep -E "passed|failed|error|FAILED|ERROR" >> $out
done
echo ALLDONE >> $out
