#!/bin/bash
# Virtual-partner evaluation queue (continues run_virtual.sh after its S=2 run; K=32 dropped as 4x the S=4 cost).
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
while kill -0 565239 2>/dev/null; do sleep 20; done
stage() { echo "=== $1 $(date '+%F %T')"; }
R="python3 -m examples.ensemble_relaxation --parameter diameter --seeds 48 --pullback virtual"
stage inlet2048;  python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 32 --particles 2048 --dt 0.5 --attempts 2048 --steps 4 --pullback virtual --substeps 4 --partners 8 > scratch/bench/virt_inlet_2048.log 2>&1
stage S4;   $R --substeps 4 --partners 8 > scratch/bench/virt_relax_S4.log 2>&1
stage M20;  $R --substeps 2 --partners 8 --dt 0.1 --steps 20 > scratch/bench/virt_relax_M20.log 2>&1
stage M40;  $R --substeps 2 --partners 8 --dt 0.05 --steps 40 > scratch/bench/virt_relax_M40.log 2>&1
stage K2;   $R --substeps 2 --partners 2 > scratch/bench/virt_relax_K2.log 2>&1
stage omega; python3 -m examples.ensemble_relaxation --parameter omega --seeds 48 --pullback virtual --substeps 2 --partners 8 > scratch/bench/virt_relax_omega.log 2>&1
stage inlet128; python3 -m examples.ensemble_inlet_shock --parameter temp --seeds 64 --pullback virtual --substeps 2 --partners 8 > scratch/bench/virt_inlet_128.log 2>&1
stage S8;   $R --substeps 8 --partners 8 > scratch/bench/virt_relax_S8.log 2>&1
echo done
