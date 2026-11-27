#!/bin/bash
# Run a command with extra modules loaded on top of the job environment, e.g. a SPARTA binary
# built with the Cray compiler:   bash bench/env/with_modules.sh PrgEnv-cray -- <command...>
# srun propagates the environment to the ranks, so this wraps run_bench.py (which calls srun).
set -euo pipefail
MODS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do MODS+=("$1"); shift; done
shift
if ! type module > /dev/null 2>&1; then
  for f in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [ -f "${f}" ] && { source "${f}"; break; }
  done
fi
for m in "${MODS[@]}"; do module load "${m}"; done
exec "$@"
