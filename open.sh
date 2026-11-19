#!/usr/bin/env bash
# Render the site, serve it, and re-render (open pages reload) on every change.
# Run it from a worktree of the site branch at site/ in a main checkout
# (git worktree add site origin/site, from that checkout's root): the pages
# read main's docs and DSL files from the folder above this one.
# Extra arguments pass through: site/open.sh --port 8001 --no-open
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec cargo run --release --quiet --manifest-path "$here/Cargo.toml" -- serve "$@"
