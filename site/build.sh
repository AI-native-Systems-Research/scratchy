#!/usr/bin/env bash
# Assemble the GitHub Pages site. See build.py for what it actually does —
# markdown rendering is client-side (zero-md), so there is no mdBook step.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/build.py"
