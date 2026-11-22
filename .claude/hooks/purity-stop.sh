#!/usr/bin/env bash
# Stop: the purity lints over every scope this host builds, before a session says it is done.
# Exit 2 sends the findings back to the session. On the second stop in a row (stop_hook_active)
# it reports them and lets the session end, so a session can't loop; CI still blocks the PR.
set -uo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
active=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))')
if "${root}/scripts/purity.sh" --scope all; then
    exit 0
fi
[[ "${active}" == True ]] && exit 0
exit 2
