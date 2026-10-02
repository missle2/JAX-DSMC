#!/bin/bash
export XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.
python3 scratch/perf/regress.py --record full --npc-bucket 1 --compare base --save bucket1 > scratch/perf/regress_bucket1.log 2>&1
python3 scratch/perf/regress.py --record lean --compare fix3 --save lean > scratch/perf/regress_lean.log 2>&1
