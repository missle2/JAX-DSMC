#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC
L=/home/aakash/jax_dsmc/run_locked.sh
O=/home/aakash/jax_dsmc/wt_misc/scratch/misc
$L gpu python3 scratch/perf/noise_gate.py forward --code /home/aakash/jax_dsmc/wt_misc --save misc > $O/gate_fwd_run.log 2>&1
python3 scratch/perf/noise_gate.py forward --compare base misc > $O/gate_fwd.txt 2>&1
python3 scratch/perf/noise_gate.py forward --compare merged_tally misc > $O/gate_fwd_vs_tally.txt 2>&1
$L gpu python3 scratch/perf/noise_gate.py gradient --code /home/aakash/jax_dsmc/wt_misc --save misc > $O/gate_grad_run.log 2>&1
python3 scratch/perf/noise_gate.py gradient --compare base misc > $O/gate_grad.txt 2>&1
python3 scratch/perf/noise_gate.py gradient --compare merged misc > $O/gate_grad_vs_merged.txt 2>&1
echo DONE > $O/gate_done
