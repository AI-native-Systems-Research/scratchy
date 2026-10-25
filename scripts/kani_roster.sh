#!/usr/bin/env bash
# One Kani proof per crate that carries `#[kani::proof]` harnesses, so a toolchain gap or a harness
# that stops building fails loudly instead of leaving every proof silently unrun (#618: Kani 0.67's
# nightly was rustc 1.93, below ktir-core's `rust-version = 1.94`, and no proof built for a month).
#
# Needs `cargo kani` from Kani 0.68.0 or later — its bundled nightly (2026-08-21) meets the 1.94 pin
# that ktir-core's NEON f16 codec needs (`stdarch_neon_f16`, stable since 1.94):
#   cargo install --locked kani-verifier --version 0.68.0 && cargo kani setup
# CI runs this in the `kani` job. Each proof below is the cheapest in its crate; run a crate's
# others with `cargo kani -p <crate>`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# scratchy-target-spyre's build reaches flex-rs's C++ shim, which needs the Spyre SDK; a proof
# links none of it.
export SCRATCHY_SKIP_SENDNN_CXX=1
# Fails closed: a name that matches no harness must not pass as a proof, whatever Kani's exit code,
# so each run must report exactly one harness verified.
proof() {
    echo "== $1: $2"
    local out
    out=$(cargo kani -p "$1" --harness "$2" --exact 2>&1 | tee /dev/stderr)
    if ! grep -q "Complete - 1 successfully verified harnesses, 0 failures, 1 total" <<<"$out"; then
        echo "kani_roster: $1 $2 did not verify exactly one harness" >&2
        exit 1
    fi
}
proof ktir-superdsc work::kani_proofs::the_one_row_fill_partitions_both_axes
proof scratchy-target-spyre fold_plan::cursor_proofs::every_row_of_a_batch_reads_its_own_cursor
proof flex-rs scheduler::completion::kani_proofs::a_timed_out_cb_is_never_a_clean_completion
proof scratchy-core-common kv::step_reach_proofs::the_allocation_covers_every_slot_the_launch_writes
