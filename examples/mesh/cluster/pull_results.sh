#!/usr/bin/env bash
# Pull the full run data from the CURC checkout -- including what .gitignore keeps out of git
# (level-set coefficients, flow-movie frames, Adam state, VTK dumps) -- into this checkout.
#
#   bash examples/mesh/cluster/pull_results.sh            # results, logs, figures
#   DRY=1 bash examples/mesh/cluster/pull_results.sh      # list what would change, copy nothing
#   bash examples/mesh/cluster/pull_results.sh examples/mesh/results/optimize   # just these paths
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
    PATHS=(examples/mesh/results examples/mesh/cluster/logs examples/mesh/figures)
fi

CTL="${TMPDIR:-/tmp}/jaxdsmc-pull-$$.sock"
SSH=(ssh -o ControlPath="$CTL" -o PubkeyAuthentication=no)
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=yes -fN "$REMOTE_USER@$HOST"
trap '"${SSH[@]}" -O exit "$REMOTE_USER@$HOST" 2>/dev/null || true' EXIT

RSYNC=(rsync -aP --update --relative --info=stats1,progress2 --exclude .git/
       --exclude __pycache__/ -e "${SSH[*]}")
[ -n "${DRY:-}" ] && RSYNC+=(--dry-run --itemize-changes)

failed=()
for p in "${PATHS[@]}"; do
    p=${p%/}
    echo "== $p"
    # "/./" marks where --relative starts, so remote $REMOTE_DIR/$p lands at $LOCAL_DIR/$p
    "${RSYNC[@]}" "$REMOTE_USER@$HOST:$REMOTE_DIR/./$p" "$LOCAL_DIR/" || failed+=("$p")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
