#!/usr/bin/env bash
# PostToolUse on Edit and Write: the purity lints over the package that owns the edited file.
# Exit 2 puts the findings in front of the session, so it fixes them before going on.
set -uo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')
case "${path}" in
    *.rs|*Cargo.toml) ;;
    *) exit 0 ;;
esac
"${root}/scripts/purity.sh" --file "${path}" || exit 2
