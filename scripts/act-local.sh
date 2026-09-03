#!/usr/bin/env bash
#
# Run the GitHub Actions CI locally with `act`, WITHOUT clobbering your global
# Rust toolchain.
#
# Why this wrapper exists
# -----------------------
# The `macos-arm64 (metal)` CI job must run natively on the Apple-Silicon host
# (Metal has no Linux container), so `act` needs `-P macos-26=-self-hosted`.
# But then `dtolnay/rust-toolchain@stable` runs on your host and executes
# `rustup toolchain install stable` + `rustup default stable` against your real
# `~/.rustup` — silently changing your global default toolchain (this is the
# "act fubars my local cargo install" gotcha). Pointing RUSTUP_HOME at a
# throwaway (but persisted) sandbox sends those writes there instead, leaving
# your `~/.rustup` untouched. The sandbox is verified populated after a run;
# your global default is never rewritten.
#
# The linux jobs (`cuda`, `fmt`) run inside Docker containers, so
# they can't touch your host toolchain regardless — only the self-hosted macOS
# job needed protecting.
#
# Usage
# -----
#   scripts/act-local.sh                 # full `pull_request` CI (all jobs)
#   scripts/act-local.sh -j macos-metal  # just the metal job (skip slow emulated CUDA)
#   scripts/act-local.sh push            # a different event
# Any extra args are forwarded to `act`.
#
# Requires Docker running for the linux/fmt jobs (the macOS job runs natively).
set -euo pipefail

# Persisted across runs so the sandbox toolchain is downloaded once, not every
# run. Isolated from ~/.rustup, so the job's `rustup default` never touches your
# global default. Override the location with SCRATCHY_ACT_RUSTUP if you like.
SANDBOX_RUSTUP="${SCRATCHY_ACT_RUSTUP:-${XDG_CACHE_HOME:-$HOME/.cache}/scratchy/act-rustup}"
mkdir -p "$SANDBOX_RUSTUP"

# Default to the pull_request event unless the first arg is a concrete event
# name (i.e. not an `act` flag like -j / -P).
EVENT="pull_request"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  EVENT="$1"
  shift
fi

echo "act: RUSTUP_HOME sandbox = $SANDBOX_RUSTUP (your ~/.rustup stays untouched)"
exec env RUSTUP_HOME="$SANDBOX_RUSTUP" act "$EVENT" \
  -P macos-26=-self-hosted \
  --container-architecture linux/amd64 \
  --env RUSTUP_HOME="$SANDBOX_RUSTUP" \
  "$@"
