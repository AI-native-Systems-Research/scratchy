#!/usr/bin/env bash
# Render the site, serve it, and re-render (open pages reload) on every change.
# Extra arguments pass through: site/open.sh --port 8001 --no-open
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec cargo run --release --quiet --manifest-path "$here/Cargo.toml" -- serve "$@"
