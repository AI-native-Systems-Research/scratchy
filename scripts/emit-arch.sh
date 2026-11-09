#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Emit one arch's forward to stdout, for ANY arch, on ANY host.
#
# `scratchy-models`' build script gates each arch on a backend (SPYRE_CAPABLE /
# METAL_ONLY in scratchy-forwards.rs), so a host without macOS can only
# `cargo check` the spyre-capable arches — every other arch's codegen is
# unverifiable there. The pipeline itself (parse → classify → shape → CFG →
# unroll → schedule → codegen) is target-neutral, so this drives it directly via
# the `emit_arch` example, with no backend gate.
#
# Verified to reproduce the build script's own emission byte-for-byte (modulo
# `reroot_crate`, which only the build script applies — see --rerooted).
#
# Usage:
#   scripts/emit-arch.sh <arch> <stem> [presets] [--rerooted]
#
#   scripts/emit-arch.sh llama llama-3.2-1b > /tmp/llama.rs
#   scripts/emit-arch.sh qwen3 qwen3-0.6b  > /tmp/qwen3.rs
#
# Typical use — prove a refactor changed no emitted code:
#   git stash && scripts/emit-arch.sh qwen3 qwen3-0.6b > /tmp/before.rs
#   git stash pop && scripts/emit-arch.sh qwen3 qwen3-0.6b > /tmp/after.rs
#   diff /tmp/before.rs /tmp/after.rs && echo "no semantic change"
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '3,27p' "$0" >&2
  exit 2
fi
arch="$1"
stem="$2"
presets="${3:-}"
reroot="${4:-}"

# The scope passes as the example's ARGUMENTS (values, not env vars — the
# example states its scope purely from argv): this script passes its own
# [presets] positional, empty by default — dense-only, matching a plain
# `-Fmodel/<stem>` build with no quant features, rather than silently
# including every quant preset — and the one [stem] it means. Name presets to
# scope them:
#   scripts/emit-arch.sh qwen3 qwen3-0.6b fp8-dynamic-per-channel
# The macro crate `compile_error!`s without a backend feature, so one is always
# passed. `spyre` is the default because it's the only backend that doesn't need
# macOS — the emit up to codegen is target-neutral, so it is also the cheapest
# way to exercise any arch. Override for a target-specific check:
#   SCRATCHY_EMIT_FEATURES=metal scripts/emit-arch.sh qwen3 qwen3-0.6b
emit() {
  cargo run -q -p scratchy-forward-compiler-macro --example emit_arch \
    --features "${SCRATCHY_EMIT_FEATURES:-spyre}" \
    -- "$arch" "$presets" "$stem"
}

if [ "$reroot" = "--rerooted" ]; then
  # What the build script writes to $OUT_DIR/<mod>.rs: paths re-rooted under
  # `pub mod <mod>`. Only needed to diff against a captured OUT_DIR file.
  emit | sed "s/crate ::/crate :: ${arch//-/_} ::/g"
else
  emit
fi
