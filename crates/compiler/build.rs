// SPDX-License-Identifier: Apache-2.0
// Detects whether scratchy-builder-cuda will build the FA3 (sm_90+) library
// for this host and emits `cfg(fa3_built)` to gate the FA3 FFI module.
// Mirrors the arch-detection logic in
// `scratchy-builder-cuda/build.rs::detect_cuda_arch`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rustc-check-cfg=cfg(fa3_built)");
    if cuda_arch_ge_90() {
        println!("cargo:rustc-cfg=fa3_built");
    }
}

fn cuda_arch_ge_90() -> bool {
    if let Ok(arch) = std::env::var("CUDA_ARCH") {
        return arch.parse::<u32>().unwrap_or(0) >= 90;
    }
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        && let Ok(s) = std::str::from_utf8(&out.stdout)
        && let Some(line) = s.lines().next()
    {
        let digits: String = line.trim().chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(arch) = digits.parse::<u32>() {
            return arch >= 90;
        }
    }
    false
}
