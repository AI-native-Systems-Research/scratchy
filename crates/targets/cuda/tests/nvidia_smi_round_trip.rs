// SPDX-License-Identifier: Apache-2.0
//! End-to-end check that `detect_via_nvidia_smi` returns a name
//! resolvable by `for_name`. Skipped on machines without
//! `nvidia-smi` (e.g. CI without a GPU).
//!
//! Lives in `tests/` (integration) rather than the `#[cfg(test)]`
//! mod inside lib.rs because the unit-test mod's `detect`-flow tests
//! would either flake on env-var leak or duplicate this without
//! covering the subprocess path.

use std::process::Command;

fn nvidia_smi_available() -> bool {
    Command::new("nvidia-smi")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn detect_via_nvidia_smi_returns_a_known_profile() {
    if !nvidia_smi_available() {
        eprintln!("skipping: nvidia-smi not available on this machine");
        return;
    }

    let name = scratchy_target_cuda::targets::detect_via_nvidia_smi()
        .expect("nvidia-smi succeeded but detect_via_nvidia_smi failed");
    assert!(!name.is_empty(), "detected name was empty");

    let profile = scratchy_target_cuda::targets::for_name(&name).unwrap_or_else(|| {
        let known: Vec<&str> = scratchy_target_cuda::targets::ALL
            .iter()
            .map(|p| p.name)
            .collect();
        panic!(
            "nvidia-smi returned `{name}` which doesn't match any known \
             profile {known:?}. If this is a new GPU we support, add a \
             ProfileDef to scratchy-target-cuda; if it's a name-format \
             mismatch, fix `normalize_gpu_name`."
        );
    });

    assert_eq!(profile.name, name, "profile lookup name mismatch");
}
