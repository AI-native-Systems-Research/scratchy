#!/usr/bin/env bash
# Formally verify the SuperDSC (SDSC) Rust IR invariants with Verus.
#
# The SDSC IR's witnesses (`superdsc_opspec.rs`: MaterializedStick / WorkPlan ≤32
# cores / StickExtent %64 / TimeTile fits-LX) are RUNTIME-enforced in the typed
# constructors; this proves the arithmetic they rely on is SMT-valid for ALL
# inputs, so the on-card DeepTools DtExceptions are formally impossible (not just
# example-tested). Verus runs on its OWN pinned toolchain (rust 1.96 + z3),
# SEPARATE from the stable `cargo build` — this is a verification step, not part
# of the normal build.
#
# Toolchain (one-time): the release is unpacked at $VERUS_DIR; it needs
#   rustup toolchain install 1.96.0-aarch64-apple-darwin
set -euo pipefail
VERUS_DIR="${VERUS_DIR:-$HOME/verus-toolchain/verus-arm64-macos}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$VERUS_DIR/verus" "$HERE/sdsc_invariants.rs"
