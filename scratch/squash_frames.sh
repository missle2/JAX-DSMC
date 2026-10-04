#!/bin/bash
# re-run the first-seed forwards of the finished squash runs for their movies
cd /home/aakash/jax_dsmc/JAX-DSMC
export XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.
L=/home/aakash/jax_dsmc/run_locked.sh
$L gpu python3 -m examples.mesh.optimize_squash --seeds-per-iterate 1 --frames
$L gpu python3 -m examples.mesh.optimize_squash --seeds-per-iterate 4 --frames
$L gpu python3 -m examples.mesh.optimize_squash --seeds-per-iterate 1 --knudsen 0.5 --frames
echo ALLDONE
