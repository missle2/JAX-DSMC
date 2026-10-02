#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/wt_levelset --save levelset_search --bodies levelset
python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/wt_levelset --save levelset_search
cd /home/aakash/jax_dsmc/wt_levelset && PYTHONPATH=. python3 scratch/perf/regress.py --compare batch_fwd --save levelset_search
echo GATES DONE
