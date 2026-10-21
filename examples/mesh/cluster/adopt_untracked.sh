#!/bin/bash
# ONE-TIME, on the CURC checkout (2026-10-21): take the laptop's commit that stops tracking the
# run data (results/optimize*, cluster/logs, bad_nodes.txt, scratch/*/cluster -- now in
# .gitignore, moved by pull_results.sh only) WITHOUT git deleting the cluster's copies or
# refusing because running jobs keep rewriting them.
#
# The script arrives with the commit it adopts, so run it from the fetched commit, from inside
# the checkout:
#   cd /projects/$USER/JAX-DSMC && git fetch origin
#   git show origin/main:examples/mesh/cluster/adopt_untracked.sh > /tmp/adopt_$USER.sh
#   bash /tmp/adopt_$USER.sh
#
# 1. hard-link backup of every data path (instant, no extra space) under
#    /projects/$USER/dsmc-data-backup-<time>/ -- a safety net only;
# 2. the same paths untracked here (`git rm --cached`: the files stay) and committed locally;
# 3. `git pull`: both sides deleted the same paths, so the merge is clean and leaves the files;
# 4. HEAD reset (soft) to origin/main: the tree is identical, so no local commit remains and
#    later pulls fast-forward.  Running jobs are not disturbed (git never touches their files).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"  # the checkout the shell is in
PATHS=(examples/mesh/results/optimize examples/mesh/results/optimize_squash
       examples/mesh/cluster/logs examples/mesh/cluster/bad_nodes.txt)
mapfile -t SCRATCH < <(git ls-files | grep -E '^scratch/[^/]+/cluster/' | cut -d/ -f1-3 | sort -u)
backup="/projects/${USER}/dsmc-data-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${backup}"
for p in "${PATHS[@]}" "${SCRATCH[@]}"; do
  if [ -e "${p}" ]; then cp -al --parents "${p}" "${backup}/"; fi
done
echo "backup (hard links): ${backup}"
# -f: some of these may be staged already (an earlier `git add` on the cluster) with content
# the running jobs have since rewritten; --cached never touches the files, -f only lets git
# drop those stale staged copies
git rm -r -q --cached -f --ignore-unmatch "${PATHS[@]}" "${SCRATCH[@]}"
if ! git diff --cached --quiet; then
  git commit -q -m "untrack run data (cluster side, transitional)"
fi
git pull -q --no-rebase --no-edit
git reset -q --soft origin/main
if git diff --cached --quiet origin/main; then
  echo "done: HEAD = origin/main, the data stays on disk untracked"
else
  echo "WARNING: the index differs from origin/main (git status); the data is untouched"
fi
git status --short | head -20
