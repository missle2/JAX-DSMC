#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
python3 examples/sparta_jax/bench.py --tag gpu --repeats 5 --set sort=full
python3 examples/sparta_jax/bench.py --tag gpu_partition --repeats 3
python3 examples/sparta_jax/bench.py --tag gpu_tally --repeats 3 --set sort=full tally=1
rm -rf /tmp/sj_hlo_gpu
XLA_FLAGS="$XLA_FLAGS --xla_dump_to=/tmp/sj_hlo_gpu --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*" python3 examples/sparta_jax/profile.py --tag gpu --dump /tmp/sj_hlo_gpu --set sort=full
