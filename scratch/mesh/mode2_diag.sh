#!/bin/bash
# after the held-out drag job: the old coarse mode2 configuration on the current code,
# with and without the hit-ahead value class (4 CRN seeds each)
until ! ps -o pid= -p 693080 > /dev/null; do sleep 20; done
export XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.
for vc in hit_ahead none; do
  echo "[$(date +%T)] value_class $vc"
  python3 -m examples.ensemble_mesh --body levelset --parameter mode2 --fnum-scale 4 \
    --levelset-h 0.125 --levelset-samples 8 --substeps 1 --partners 2 --seeds 4 \
    --value-class $vc --output scratch/mesh/mode2_coarse_now_$vc.json 2>&1 | grep -E "^seed=|Error"
done
echo "[$(date +%T)] done"
