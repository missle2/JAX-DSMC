#!/bin/bash
# Device (GPU) evaluation of the virtual pull-back with staged nodes on every validated control.
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
stage() { echo "=== $1 $(date '+%F %T')"; }
V="--pullback virtual --nodes staged --device"
R="python3 -m examples.ensemble_relaxation --seeds 48 $V"
I="python3 -m examples.ensemble_inlet_shock $V --substeps 2 --partners 8"
C="python3 -m examples.ensemble_couette --seeds 48 $V --substeps 2 --partners 8"
stage relax_S4;   $R --parameter diameter --substeps 4 --partners 8 > scratch/bench/gpu_relax_S4st.log 2>&1
stage relax_S8;   $R --parameter diameter --substeps 8 --partners 8 > scratch/bench/gpu_relax_S8st.log 2>&1
stage relax_K2;   $R --parameter diameter --substeps 2 --partners 2 > scratch/bench/gpu_relax_K2st.log 2>&1
stage relax_K32;  $R --parameter diameter --substeps 2 --partners 32 > scratch/bench/gpu_relax_K32st.log 2>&1
stage relax_M20;  $R --parameter diameter --substeps 2 --partners 8 --dt 0.1 --steps 20 > scratch/bench/gpu_relax_M20st.log 2>&1
stage relax_M40;  $R --parameter diameter --substeps 2 --partners 8 --dt 0.05 --steps 40 > scratch/bench/gpu_relax_M40st.log 2>&1
stage relax_omega; $R --parameter omega --substeps 2 --partners 8 > scratch/bench/gpu_relax_omega_st.log 2>&1
stage inlet_temp128; $I --parameter temp --seeds 64 > scratch/bench/gpu_inlet_temp128.log 2>&1
stage inlet_nrho128; $I --parameter nrho --seeds 64 > scratch/bench/gpu_inlet_nrho128.log 2>&1
stage inlet_temp256; $I --parameter temp --seeds 48 --particles 256 --steps 8 > scratch/bench/gpu_inlet_temp256.log 2>&1
stage inlet_nrho256; $I --parameter nrho --seeds 48 --particles 256 --steps 8 > scratch/bench/gpu_inlet_nrho256.log 2>&1
stage inlet_temp2048; $I --parameter temp --seeds 32 --particles 2048 --dt 0.5 --attempts 2048 --steps 4 > scratch/bench/gpu_inlet_temp2048.log 2>&1
stage inlet_nrho2048; $I --parameter nrho --seeds 32 --particles 2048 --dt 0.5 --attempts 2048 --steps 4 > scratch/bench/gpu_inlet_nrho2048.log 2>&1
stage couette_speed; $C --parameter speed > scratch/bench/gpu_couette_speed.log 2>&1
stage couette_speed_eps0; $C --parameter speed --epsilon 0 > scratch/bench/gpu_couette_speed_eps0.log 2>&1
stage couette_tleft; $C --parameter temperature_left > scratch/bench/gpu_couette_tleft.log 2>&1
stage couette_acc; $C --parameter accommodation > scratch/bench/gpu_couette_acc.log 2>&1
echo done
