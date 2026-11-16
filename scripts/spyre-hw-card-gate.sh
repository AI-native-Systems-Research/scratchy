#!/usr/bin/env bash
# ⭐ THE CARD-HOST GATE: every check CI used to run on backends/spyre-hw that must
# COMPILE senlib-rs's flex-rs. The workspace resolves flex-rs by path into a sibling
# senlib-rs checkout (IBM-internal git — a card host fact), so no GitHub runner can
# run these; CI's counterpart (scripts/ci-senlib-stub.sh) covers the steps that only
# resolve. Run this on a card host, from anywhere in the scratchy tree:
#
#   scripts/spyre-hw-card-gate.sh
#
# It fails loudly (never skips) when the sibling checkout is absent. Nothing here
# links the card runtime — clippy and compile-only doctests — so the export-controlled
# stub senlib-rs names is enough; no `--config patches/oco.toml` (that route is for
# the builds that link: the card binary, the card leg).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
hw="$root/backends/spyre-hw"

# The seam's own layout contract: an absent checkout fails by path — never a fallback.
if [ ! -f "$hw/../../../senlib-rs/crates/flex-rs/Cargo.toml" ]; then
    echo "scripts/spyre-hw-card-gate.sh: no senlib-rs checkout at the sibling path" >&2
    echo "this gate compiles flex-rs, which lives in senlib-rs (IBM-internal) beside the tree" >&2
    exit 1
fi

cd "$hw"

# The workspace clippy CI ran before the seam: every member — card and flex-rs included.
# The build script's FFI link directives are skipped (nothing here links); a model
# feature must be named (zero-model is a build-time panic).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy --workspace \
    --features model/granite-3.1-8b-instruct -- -D warnings

# The card surface alone, as the triton job stated it: the card runtime and flex-rs,
# linted at -D warnings (it rotted between card runs once — #553).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-target-spyre-card -p flex-rs -- -D warnings

# The worker's CARD path, against the card's lowering (`spyre-lower`): same code with
# or without device images, so no device compiler is needed.
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-target-spyre-worker -p scratchy-target-spyre-cli \
    --features scratchy-target-spyre-worker/spyre-hw,model/granite-3.1-2b-instruct \
    --all-targets -- -D warnings

# The card crate's compile_fail doctests (the nightly job's leg).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo +nightly test --doc -p scratchy-target-spyre-card
