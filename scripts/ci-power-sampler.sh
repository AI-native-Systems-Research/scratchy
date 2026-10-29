#!/usr/bin/env bash
# CI check of the Metal matrix's energy sampler on this runner's macOS: its unit tests (each
# reader's channels to joules), then a real run. A VM's IOReport has no energy channels, so
# there it must fail with that error rather than write zeros; on a real Mac it must sample.
set -euo pipefail
cargo test -p scratchy-power-sampler
cargo build --release -p scratchy-power-sampler
if [[ "$(sysctl -n kern.hv_vmm_present)" == 1 ]]; then
    if out=$(target/release/scratchy-power-sampler --samples 3 2>&1); then
        echo "${out}"
        echo "sampled in a VM, which has no energy channels" >&2
        exit 1
    fi
    echo "${out}"
    grep -q "no energy channels" <<<"${out}"
else
    target/release/scratchy-power-sampler --samples 3
fi
