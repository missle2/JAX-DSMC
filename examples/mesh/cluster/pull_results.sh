#!/usr/bin/env bash
# Pull the run data from the CURC checkout into this checkout.  Since 2026-10-21 this is the
# ONLY way run data moves: git carries code, docs, figures and analyses, and everything the
# jobs write (results/optimize*, cluster/logs, scratch/*/cluster) is in .gitignore -- commit
# after a pull without picking up data, and the cluster's `git pull` never meets the files its
# jobs keep writing.
#
#   bash examples/mesh/cluster/pull_results.sh            # results (mesh + paper), logs, figures, scratch/*/cluster
#   DRY=1 bash examples/mesh/cluster/pull_results.sh      # list what would change, copy nothing
#   WARM=1 bash examples/mesh/cluster/pull_results.sh     # also the optimisers' warm-start gas
#   bash examples/mesh/cluster/pull_results.sh examples/mesh/results/optimize   # just these paths
#
# The warm starts (<run>/warm/seed<i>.npz, ~0.2 GB per seed, rewritten every iterate) only
# resume a run on the cluster; they are skipped unless WARM=1.  scratch/<probe>/cluster/ (the
# probes' job outputs) is pulled with the default paths or by naming "scratch".
#
# Password + Duo authentication, no keys: one ssh connection is opened (you authenticate once)
# and every rsync goes through it; it is closed on exit.  Pull only: nothing on the cluster
# is touched, and nothing here is deleted.  A local file newer than the cluster's copy is kept
# (--update).  Overrides: REMOTE_USER, HOST (e.g. dtn.rc.colorado.edu for big pulls), REMOTE_DIR.
set -euo pipefail

REMOTE_USER=${REMOTE_USER:-aaja8389}
HOST=${HOST:-login.rc.colorado.edu}
REMOTE_DIR=${REMOTE_DIR:-/projects/aaja8389/JAX-DSMC}
LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

if [ $# -gt 0 ]; then
    PATHS=("$@")
else
    PATHS=(examples/mesh/results examples/paper/results examples/mesh/cluster/logs examples/mesh/figures
           scratch)
fi

CTL="${TMPDIR:-/tmp}/jaxdsmc-pull-$$.sock"
SSH=(ssh -o ControlPath="$CTL" -o PubkeyAuthentication=no)
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=yes -fN "$REMOTE_USER@$HOST"
trap '"${SSH[@]}" -O exit "$REMOTE_USER@$HOST" 2>/dev/null || true' EXIT

RSYNC=(rsync -aP --update --relative --info=stats1,progress2 --exclude .git/
       --exclude __pycache__/ -e "${SSH[*]}")
[ -n "${DRY:-}" ] && RSYNC+=(--dry-run --itemize-changes)
[ -z "${WARM:-}" ] && RSYNC+=(--exclude 'warm/seed*.npz')
# scratch: only the probes' cluster outputs (the scripts come by git)
SCRATCH_ONLY=(--include '/scratch/' --include '/scratch/*/' --include '/scratch/*/cluster/***'
              --exclude '*' --prune-empty-dirs)

failed=()
for p in "${PATHS[@]}"; do
    p=${p%/}
    echo "== $p"
    # "/./" marks where --relative starts, so remote $REMOTE_DIR/$p lands at $LOCAL_DIR/$p
    extra=()
    [ "$p" = scratch ] && extra=("${SCRATCH_ONLY[@]}")
    "${RSYNC[@]}" "${extra[@]}" "$REMOTE_USER@$HOST:$REMOTE_DIR/./$p" "$LOCAL_DIR/" || failed+=("$p")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
