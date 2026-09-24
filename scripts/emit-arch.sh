#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Emit one arch's forward to stdout, for ANY arch, on ANY host.
#
# `scratchy-models`' build script gates each arch on a backend (SPYRE_CAPABLE /
# CUDA_ONLY in scratchy-forwards.rs), so a host with neither a CUDA toolchain nor
# macOS can only `cargo check` llama and granite — the other 23 arches' codegen
# is unverifiable there. The pipeline itself (parse → classify → shape → CFG →
# unroll → schedule → codegen) is target-neutral, so this drives it directly via
# the `emit_arch` example, with no backend gate.
#
# Verified to reproduce the build script's own emission byte-for-byte (modulo
# `reroot_crate`, which only the build script applies — see --rerooted).
#
# Usage:
#   scripts/emit-arch.sh <arch> <stem> [--rerooted]
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
reroot="${3:-}"

# Mirror Cargo's feature-name → env-var transform exactly: uppercase, `-` → `_`,
# every other byte verbatim (INCLUDING `.` — `llama-3.2-1b` becomes
# CARGO_FEATURE_LLAMA_3.2_1B, which needs `env` since bash can't export a name
# containing a dot). This is the same gate config.rs's `model_feature_enabled`
# reads, so only the named stem is expanded.
var="CARGO_FEATURE_$(printf '%s' "$stem" | tr '[:lower:]-' '[:upper:]_')"

# SCRATCHY_QUANTS unset means "every preset"; the real build script always sets
# it (to the union of enabled presets, empty when none). Set it empty so a
# default run matches a plain `-Fmodel/<stem>` build rather than silently
# including every quant preset.
# The macro crate `compile_error!`s without a backend feature, so one is always
# passed. `spyre` is the default because it's the only backend that needs neither
# nvcc nor macOS — the emit up to codegen is target-neutral, so it is also the
# cheapest way to exercise any arch. Override for a target-specific check:
#   SCRATCHY_EMIT_FEATURES=cuda scripts/emit-arch.sh gemma3-mm gemma-3-4b-it
emit() {
  env "$var=1" SCRATCHY_QUANTS="${SCRATCHY_QUANTS-}" \
    cargo run -q -p scratchy-forward-compiler-macro --example emit_arch \
    --features "${SCRATCHY_EMIT_FEATURES:-spyre}" \
    -- "$arch"
}

if [ "$reroot" = "--rerooted" ]; then
  # What the build script writes to $OUT_DIR/<mod>.rs: paths re-rooted under
  # `pub mod <mod>`. Only needed to diff against a captured OUT_DIR file.
  emit | sed "s/crate ::/crate :: ${arch//-/_} ::/g"
else
  emit
fi
