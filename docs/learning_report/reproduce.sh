#!/usr/bin/env bash
# Reproduce every simulation of the learned-collision report on a SLURM GPU cluster.
# From the repository root, on a login node, with the venv active:
#   bash docs/learning_report/reproduce.sh <stage>      # DRY=1: print the sbatch lines only
# Run the stages in order; start the next when `squeue -u $USER` shows the previous done.
#   refs      fine references: 8 slab flows (lcref) and 2 fine cylinders
#   prep      shock warm pool, input statistics, classical baselines, coarse cylinders
#   train     linear (rate), lasso (LOCO and split-seed), rich linear/MLPs, the L9 sweep
#   eval      the learned models on all twelve flows and on the cylinder
#   analysis  CPU only: cylinder report, per-cell inputs, every figure and table, the PDF
# Results land in examples/learned_collision/results/ (git-ignored). Every job's command
# comes from examples/mesh/cluster/study.py, the definitions used for the report's runs.
set -euo pipefail
SBATCH_SCRIPT=docs/learning_report/run_lc.sbatch
LC="python3 -m examples.learned_collision"

submit() {  # submit <job name> <command>
  # shellcheck disable=SC2086  # SBATCH_EXTRA is a list of sbatch options
  if [ -n "${DRY:-}" ]; then echo "sbatch ${SBATCH_EXTRA:-} --job-name=$1 CMD=$2"; return; fi
  sbatch ${SBATCH_EXTRA:-} --job-name="$1" --export=ALL,CMD="$2" "${SBATCH_SCRIPT}"
}

submit_group() {  # submit_group <study group> [name regex]: one job per line of the group
  python3 - "$1" "${2:-.}" <<'PY' |
import re
import sys

from examples.mesh.cluster import study

for name, cmd in study.NEW_GROUPS[sys.argv[1]](1).items():
    if re.search(sys.argv[2], name):
        print(f"{name}\t{cmd}")
PY
  while IFS=$'\t' read -r name cmd; do submit "${name}" "${cmd}"; done
}

case "${1:-}" in
  refs)
    submit_group lcref
    submit_group lccirc '^lccirc_fine'      # needs a >= 48 GB GPU (SBATCH_EXTRA)
    ;;
  prep)
    submit lcwarm "${LC}.warm_pool --blocks 160 --tag v1"
    submit lcfeat "${LC}.features --tag v2 && ${LC}.features --tag v4"
    submit_group lcbase
    submit_group lccirc '^lccirc_coarse'
    ;;
  train)
    submit_group lcfix '^lcfix_q1$'          # linear, rate outputs
    submit_group lcextra '^lclasso'          # lasso: LOCO, and split-seed + its evaluation
    submit_group lcrich                      # rich linear, rich 3x32 and 2x32 (pre-sweep)
    # The L9 sweep: sweep_plan_v3.json (results archive) fixes its 63 runs, 177 iterations
    # each, in 7 jobs (~7.8 GPU-h each). To rebuild the plan from the timing runs instead:
    #   ${LC}.sweep --tag v3 --hours 55 --jobs 7 \
    #     --train-args "--outputs rich --features features_v4.json --warm-pool v1"
    submit_group lccv3
    ;;
  eval)
    submit_group lceval '^lceval_(linear|circle_linear)$'
    submit_group lcextra '^lceval_'          # rich linear, 3x32, 2x32; 2x32 on the cylinder
    submit_group lcp3                        # the cross-validated MLP, both seeds
    submit_group lcp3circ                    # ... and on the cylinder
    ;;
  analysis)
    export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=.
    ${LC}.circle_report --tag v2
    python3 scratch/lc_cellfeat.py
    python3 docs/learning_report/make_figures.py
    (cd docs/learning_report && latexmk -pdf report.tex)
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
