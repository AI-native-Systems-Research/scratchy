#!/usr/bin/env bash
# ⭐ THE CI-ONLY SIBLING: a minimal senlib-rs checkout, synthesized where the real one
# cannot reach. `backends/spyre-hw` resolves `flex-rs` by path into a sibling senlib-rs
# checkout, and cargo loads every workspace member's dependencies' manifests — optional
# ones included — so even the emulator-only `spyre` build and a plain `cargo metadata`
# refuse to resolve without that sibling. The real checkout lives on IBM-internal git
# (a card host fact); this script writes the smallest tree that satisfies resolution:
# the flex-rs manifest with the one feature this workspace names (`snt1p0`), over an
# EMPTY lib.
#
# That emptiness is the guard, not a gap: every CI step that only RESOLVES the card
# workspace (the emulator clippy and tests, the `spyre` binary's targets, `cargo
# metadata`, `cargo fmt`) runs against the stub, and any step that would COMPILE
# flex-rs fails loudly on its unresolved imports instead of passing against a hollow
# crate. The steps that must compile flex-rs (the card-surface clippy, the worker's
# card path, the card crate's compile_fail doctests) are scripts/spyre-hw-card-gate.sh,
# run on a card host beside the real checkout — never here.
#
# Idempotent: rewrites the same bytes, so a job may call it before any step.
#
# ⛔ AND ONLY ITS OWN BYTES: the sibling path this writes is exactly where the layout
# contract puts the REAL checkout on a card host, so a real tree there is a refusal,
# not an overwrite. A checkout is real if it carries git metadata, or if its flex-rs
# has source (the stub's lib.rs is empty by construction — a non-empty one is code
# this script did not write).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
stub="$root/../senlib-rs"

if [[ -e "$stub/.git" ]]; then
  echo "refusing: $stub is a git checkout — the real senlib-rs, not a stub's home." >&2
  echo "this script synthesizes the CI-only sibling; it never writes over a real tree." >&2
  exit 1
fi
if [[ -s "$stub/crates/flex-rs/src/lib.rs" ]]; then
  echo "refusing: $stub/crates/flex-rs/src/lib.rs has source — a real senlib-rs, not a stub's home." >&2
  echo "this script synthesizes the CI-only sibling; it never writes over a real tree." >&2
  exit 1
fi

mkdir -p "$stub/crates/flex-rs/src"

cat >"$stub/Cargo.toml" <<'EOF'
# CI stub (scripts/ci-senlib-stub.sh) — not the real senlib-rs workspace. The real
# checkout is IBM-internal; this exists so the sibling path the card workspace's
# manifests resolve against is present on a runner. Its flex-rs has no source.
[workspace]
resolver = "2"
members = ["crates/flex-rs"]
EOF

cat >"$stub/crates/flex-rs/Cargo.toml" <<'EOF'
# CI stub (scripts/ci-senlib-stub.sh): the real flex-rs manifest's name, version and
# the features backends/spyre-hw names — and nothing else. No dependencies, so the
# stub resolves standalone; no source, so nothing can compile against it and pass.
# `alloc-census` is named by the seat's census wiring (the card crate's feature of
# the same name forwards to it); the stub carries the NAME only — hollow, like the
# rest of this tree, and a build that would use it still fails on unresolved imports.
[package]
name = "flex-rs"
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
description = "CI-only stub of senlib-rs's flex-rs: manifest for resolution, no source."

[features]
snt1p0 = []
alloc-census = []
EOF

: >"$stub/crates/flex-rs/src/lib.rs"

echo "senlib-rs stub at $stub"
