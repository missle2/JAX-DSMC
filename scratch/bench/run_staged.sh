#!/bin/bash
cd /home/aakash/jax_dsmc/JAX-DSMC-adjoint
export JAX_PLATFORMS=cpu
R="python3 -m examples.ensemble_relaxation --parameter diameter --seeds 48 --pullback virtual --partners 8 --nodes staged"
echo "=== S2st $(date '+%F %T')"; $R --substeps 2 > scratch/bench/virt_relax_S2st.log 2>&1
echo "=== S4st $(date '+%F %T')"; $R --substeps 4 > scratch/bench/virt_relax_S4st.log 2>&1
echo "=== S8st $(date '+%F %T')"; $R --substeps 8 > scratch/bench/virt_relax_S8st.log 2>&1
echo done
