#!/bin/bash
# all GPU work of sparta_jax in one lock hold
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"
python3 examples/sparta_jax/bench.py --tag gpu --repeats 3 --set sort=full
python3 examples/sparta_jax/bench.py --tag gpu_fullsort --repeats 3 --set sort=full
python3 examples/sparta_jax/bench.py --tag gpu_tally --repeats 2 --set tally=1
rm -rf /tmp/sj_hlo_gpu
XLA_FLAGS="$XLA_FLAGS --xla_dump_to=/tmp/sj_hlo_gpu --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*" python3 examples/sparta_jax/profile.py --tag gpu --dump /tmp/sj_hlo_gpu
python3 examples/sparta_jax/validate.py --seeds 8
