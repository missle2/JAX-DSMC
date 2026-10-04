#!/bin/bash
# is the FM -2 % systematic? (1) grazing cut-off: grazing 0 on the s = 1 tb10 seeds (the forward
# is identical; the cut-off only changes the hit-time derivative) -> paired with
# fm_gate_f2_tb10.jsonl; (2) horizon 300 vs 600 captured steps, FM, s = 1 and 0.35
cd /home/aakash/jax_dsmc/JAX-DSMC
run() { PYTHONPATH=. /home/aakash/jax_dsmc/run_locked.sh gpu python3 "$@"; }
run scratch/bias/kn_gate.py scratch/bias/fm_gate_f2_tb10_gz0.jsonl 1.0 0 8 \
  '{"fnum_scale": 2, "virtual": [1, 2], "fd_steps": [0.05], "timing_band": 10, "grazing": 0.0}'
for s in 1.0 0.35; do
  run scratch/bias/horizon.py scratch/bias/horizon_fm_f4.jsonl $s 0 8 \
    '{"fnum_scale": 4, "virtual": [1, 2], "fd_steps": [0.05], "horizons": [300, 600]}'
done
