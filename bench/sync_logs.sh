#!/bin/bash
# Publish this cluster's finished job logs and benchmark results through git (instead of
# committing .out files by hand), so they can be pulled anywhere with a plain `git pull`.
#
#   bash bench/sync_logs.sh              # copy -> commit -> pull --rebase -> push
#   NO_PUSH=1 bash bench/sync_logs.sh    # copy + commit only
#
# Copies, into cluster_logs/<site>/ (tracked):
#   jobs/      Slurm output of FINISHED jobs from joblogs/ (running jobs are skipped: their file
#              is still growing; the next sync picks it up)
#   results/   ${BENCH_ROOT}/results: every platform's rows.jsonl and the SPARTA-format run logs
#   misc/      untracked *.log / *.out of the repository root (build logs, hand-submitted jobs)
#   builds/    each SPARTA build's BUILD_INFO.txt
# Jobs only ever write to joblogs/ (git-ignored) and ${BENCH_ROOT}, never to cluster_logs/, so a
# pull can no longer cut a running job's log short (tune-pm-cpu-n1_62133655.out, 2026-11-28).
#
# One passphrase per sync: without a running ssh-agent the script starts one for its pull+push.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
if [ -n "${NERSC_HOST:-}" ]; then
  SITE=perlmutter; BENCH_ROOT="${BENCH_ROOT:-${PSCRATCH}/sparta-scaling}"
elif [ -d /curc ]; then
  SITE=alpine; BENCH_ROOT="${BENCH_ROOT:-/projects/${USER}/sparta-scaling}"
else
  echo "unknown site (neither NERSC nor CURC)"; exit 1
fi
DEST="cluster_logs/${SITE}"
mkdir -p "${DEST}/jobs" "${DEST}/results"

# 1. finished jobs' Slurm output
running=" $(squeue --me -h -o %A 2>/dev/null | tr '\n' ' ') "
nj=0; nskip=0
for f in joblogs/*.out; do
  [ -e "${f}" ] || continue
  id="${f##*_}"; id="${id%.out}"
  if [[ "${running}" == *" ${id} "* ]]; then nskip=$((nskip + 1)); continue; fi
  if ! cmp -s "${f}" "${DEST}/jobs/${f##*/}"; then cp -p "${f}" "${DEST}/jobs/"; nj=$((nj + 1)); fi
done
echo "[sync_logs] ${SITE}: ${nj} job log(s) new/updated, ${nskip} running job(s) skipped"

# 1b. untracked *.log / *.out in the repository root: build logs (build_k47.log, ...) and jobs
#     sbatch'ed by hand (their output lands in the submit directory, not joblogs/)
nm=0
for f in *.log *.out; do
  [ -e "${f}" ] || continue
  git ls-files --error-unmatch "${f}" > /dev/null 2>&1 && continue  # already tracked (old logs)
  id="${f##*_}"; id="${id%.out}"
  [[ "${f}" == *.out && "${running}" == *" ${id} "* ]] && continue
  mkdir -p "${DEST}/misc"
  if ! cmp -s "${f}" "${DEST}/misc/${f}"; then cp -p "${f}" "${DEST}/misc/"; nm=$((nm + 1)); fi
done
# build records of every SPARTA build (flags, compilers, Kokkos) -- small
for b in "${BENCH_ROOT}"/builds/*/BUILD_INFO.txt; do
  [ -e "${b}" ] || continue
  d="${b%/BUILD_INFO.txt}"; mkdir -p "${DEST}/builds"; cp -p "${b}" "${DEST}/builds/${d##*/}.txt"
done
echo "[sync_logs] ${nm} root log(s) new/updated"

# 2. results (rows + run logs), additive
if [ -d "${BENCH_ROOT}/results" ]; then
  rsync -a "${BENCH_ROOT}/results/" "${DEST}/results/"
  echo "[sync_logs] results: $(find "${DEST}/results" -name rows.jsonl | wc -l) rows.jsonl, $(du -sh "${DEST}/results" | cut -f1)"
fi

# 3. commit just cluster_logs/<site>, then publish
git add -A "${DEST}"
if git diff --cached --quiet -- "${DEST}"; then
  echo "[sync_logs] nothing new"
else
  git commit -q -m "logs: ${SITE} $(date +%Y-%m-%dT%H:%M)" -- "${DEST}"
  echo "[sync_logs] committed $(git log -1 --format=%h)"
fi
[ "${NO_PUSH:-0}" = 1 ] && exit 0
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  eval "$(ssh-agent -s)" > /dev/null
  trap 'ssh-agent -k > /dev/null' EXIT
  ssh-add
fi
# the other cluster pushes to the same branch: rebase onto it (disjoint directories, no conflicts)
for try in 1 2 3; do
  git pull -q --rebase --autostash && git push -q && { echo "[sync_logs] pushed"; exit 0; }
  echo "[sync_logs] push raced (try ${try}); retrying"
done
exit 1
