#!/usr/bin/env bash
# ⭐ THE CARD-HOST GATE: every check CI used to run on backends/spyre-hw that must
# COMPILE senlib-rs's flex-rs, plus the card runs that prove the transport live. The
# workspace resolves flex-rs by path into a sibling
# senlib-rs checkout (IBM-internal git — a card host fact), so no GitHub runner can
# run these; CI's counterpart (scripts/ci-senlib-stub.sh) covers the steps that only
# resolve. Run this on a card host, from anywhere in the scratchy tree:
#
#   scripts/spyre-hw-card-gate.sh
#
# It fails loudly (never skips) when the sibling checkout is absent. Everything here
# but the last leg is clippy and compile-only doctests — the export-controlled stub
# senlib-rs names is enough for those; the last leg LINKS and RUNS on the cards
# (`cards_all_gather alone2`, #1429's count law), so it takes the oco route
# (`--config patches/oco.toml`) the card binary's builds take.
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
# feature must be named (zero-model is a build-time panic), and so must the chip
# (`snt1p0`, Spyre generation 1 — ktir-core's compile_error! refuses a build that
# names none).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy --workspace \
    --features model/granite-3.1-8b-instruct,snt1p0 -- -D warnings

# The card surface alone, as the triton job stated it: the card runtime and flex-rs,
# linted at -D warnings (it rotted between card runs once — #553).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-target-spyre-card -p flex-rs --features snt1p0 -- -D warnings

# The worker's CARD path, against the card's lowering (`spyre-lower`): same code with
# or without device images, so no device compiler is needed.
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-target-spyre-worker -p scratchy-target-spyre-cli \
    --features scratchy-target-spyre-worker/spyre-hw,model/granite-3.1-2b-instruct,snt1p0 \
    --all-targets -- -D warnings

# The card BINARY's real shape: the CLI crate at its own `spyre-hw` feature — the only
# configuration where `main.rs`'s `Command` carries both variants (`Shared` beside the
# cfg'd `Bundle`), i.e. the enum every card build actually compiles. No other leg
# reaches it: the workspace clippy builds the member at its defaults (`spyre-hw` off,
# `Command` single-variant), and the worker leg above names the WORKER's `spyre-hw`,
# which does not flow upstream to the CLI. That hole is where `Command::Shared`'s
# `large_enum_variant` (≥360 B inline beside a ~72 B variant) lived unseen until
# 2026-11-17; the box fix landed first so this leg is green at birth, and it stays so
# that nothing of the class crawls through again. Same model as the worker leg, so
# the bake it rides is already paid for there.
SCRATCHY_SKIP_SENDNN_CXX=1 cargo clippy -p scratchy-target-spyre-cli \
    --features spyre-hw,model/granite-3.1-2b-instruct,snt1p0 -- -D warnings

# The card crate's compile_fail doctests (the nightly job's leg).
SCRATCHY_SKIP_SENDNN_CXX=1 cargo +nightly test --doc -p scratchy-target-spyre-card --features snt1p0

# ⭐ THE COUNT LAW, LIVE ON THE CARDS (#1429 — the law is stated once, on
# ktir-superdsc's `Message`): one wdone bump per run, on the last message, so every
# message of a multi-message send must be its own complete send (`RunPlace::ALONE`).
# `alone2` is the 8-MiB two-message all-gather through the production CB path
# (per-message F-sends, one wdone wait per slot) and must pass byte-exact; the
# run-flags shape it replaces wedges the card — zero completions, no error — which is
# the defect this leg exists to keep out. The host's whole card table — no N passed —
# so any card host runs it. The one leg here that links and runs: it takes the oco
# route, like every link-bearing build.
SCRATCHY_SKIP_SENDNN_CXX=1 cargo --config patches/oco.toml build --release \
    --example cards_all_gather -p scratchy-target-spyre-card --features snt1p0
DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share \
LD_LIBRARY_PATH="/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/spyre-comms/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    target/release/examples/cards_all_gather alone2
