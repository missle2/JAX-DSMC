#!/bin/bash
# Local preflight of a cluster group before submission, the way job.sbatch runs a job: from
# the repository root, PYTHONPATH exactly as env.sh sets it (nothing else), CPU only.
# The 2026-10-04 biasgate jobs all died on `import jax_dsmc` after queueing.
#
#   bash examples/mesh/cluster/preflight.sh <group> [name-regex]     (from the repo root)
#
# 1. the import check of job.sbatch (seconds);
# 2. every job of scratch/*/cluster_job.sh with CHECK=1: its name parses and its input files
#    exist (seconds per job);
# 3. ONE toy run per code path -- the job name with its shape, seed / block / S and other
#    numbers stripped (sq_s1_seed3, sq_s0.6_seed0 -> sq_s_seed; optc_a100_kn_b2 -> optc_kn_b;
#    kn0.5_S8 -> kn_S): script jobs with SMOKE=1 (fnum x16, 30 + 30 steps, one seed, into
#    *_smoke/), optimisers with toy flags, one iterate, a /tmp/preflight tag.  Shapes and
#    seeds run the same code, so one of each kind proves the plumbing; what depends on
#    density or shape (NTC attempt budget, inflow shortfall, memory) a toy cannot show --
#    check that at production size once (scratch/cellfit/attempts_check.py).
# Exit status: the number of failures.
set -uo pipefail
group="$1"
filter="${2:-.}"
ROOT="$(pwd)"
[ -f "${ROOT}/jax_dsmc/__init__.py" ] || { echo "run from the repository root"; exit 2; }
LOCK=/home/aakash/jax_dsmc/run_locked.sh
[ -x "${LOCK}" ] || LOCK=""
TOY_OPT="--iterations 1 --fnum-scale 16 --warmup 30 --steps 30 --window 20 --frames-every 0"
TOY_OPT="${TOY_OPT} --seeds-per-iterate 2"
mkdir -p /tmp/preflight
failed=0

clean() {  # run "$1" in a clean environment as job.sbatch would; extra env as "$2"
  env -u PYTHONPATH JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 \
    SLURM_SUBMIT_DIR="${ROOT}" ${2:-} bash -c '
      export PYTHONPATH="${SLURM_SUBMIT_DIR:-${PWD}}${PYTHONPATH:+:${PYTHONPATH}}"
      eval "$0"' "$1"
}

if clean '(cd /tmp && python3 -c "import jax_dsmc, examples.ensemble_mesh")' > /tmp/preflight/import.log 2>&1; then
  echo "import check: ok"
else
  echo "import check FAILED"; cat /tmp/preflight/import.log; exit 1
fi
mapfile -t NAMES < <(python3 -m examples.mesh.cluster.study --group "${group}" --names --gpus-per-node 1)
mapfile -t CMDS < <(python3 -m examples.mesh.cluster.study --group "${group}" --list --gpus-per-node 1)
declare -A SEEN=()
checked=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  grep -Eq "${filter}" <<< "${name}" || continue
  cmd="${CMDS[$i]}"
  if [[ "${cmd}" == bash\ scratch/* ]]; then
    if ! clean "${cmd}" CHECK=1 > "/tmp/preflight/${group}_${name}.check" 2>&1; then
      echo "check FAILED ${name}: $(tail -1 "/tmp/preflight/${group}_${name}.check")"
      failed=$((failed + 1))
    fi
    checked=$((checked + 1))
  fi
  kind="$(sed -E 's/^optc_[a-z0-9]+_/optc_/; s/[0-9.]+//g' <<< "${name}")"
  [ -n "${SEEN[${kind}]:-}" ] && continue
  SEEN[${kind}]=1
  if [[ "${cmd}" == python3\ -m\ examples.mesh.optimize* ]]; then
    # a fresh toy run: the optimisers resume from their tag directory, so a left-over one
    # would "pass" by skipping the finished iterate
    rm -rf "/tmp/preflight/${group}_${name}"
    cmd="${cmd} ${TOY_OPT} --tag /tmp/preflight/${group}_${name}"
    # optimize.py's worker pool (optimize_squash has none)
    [[ "${cmd}" == python3\ -m\ examples.mesh.optimize\ * ]] && cmd="${cmd} --workers 1"
  fi
  log="/tmp/preflight/${group}_${name}.log"
  echo "[$(date +%H:%M:%S)] toy ${group}/${name} (kind ${kind})"
  clean "${LOCK:+${LOCK} cpu }${cmd}" SMOKE=1 > "${log}" 2>&1
  rc=$?
  if [ ${rc} -eq 0 ] && ! grep -q "Traceback" "${log}"; then
    echo "    ok"
  else
    echo "    FAILED (exit ${rc}): ${log}"; tail -5 "${log}" | sed 's/^/      /'
    failed=$((failed + 1))
  fi
done
echo "preflight ${group}: ${checked} jobs checked, ${#SEEN[@]} toy runs, ${failed} failed"
exit "${failed}"
