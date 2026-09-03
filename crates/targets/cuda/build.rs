// SPDX-License-Identifier: Apache-2.0
// Two jobs:
//  1. Detect whether scratchy-builder-cuda will build the FA3 (sm_90+) library
//     for this host and emit `cfg(fa3_built)` to gate the FA3 FFI module.
//     Mirrors the arch-detection logic in
//     `scratchy-builder-cuda/build.rs::detect_cuda_arch`.
//  2. Emit the linker directives for the CUDA kernel `.a` files compiled by
//     scratchy-builder-cuda's build.rs (the single kernel-linking nexus after
//     the `scratchy-serving-cuda` dissolution). Kernel *compilation* still
//     lives in that builder crate; this build.rs only names the libraries.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rustc-check-cfg=cfg(fa3_built)");
    if cuda_arch_ge_90() {
        println!("cargo:rustc-cfg=fa3_built");
    }

    #[cfg(feature = "cuda")]
    cuda_link();
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

#[cfg(feature = "cuda")]
fn cuda_link() {
    // Locate the shared cudaforge cache populated by scratchy-builder-cuda's
    // build.rs. The path is canonical — keep it identical to the builder's
    // output dir so its `.a` files are found without a rebuild.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("scratchy-serving-cuda");
    let cache_str = cache_dir.to_string_lossy().to_string();

    println!("cargo:rustc-link-search={}", cache_str);

    println!("cargo:rustc-link-lib=static=vllm_kernels");
    println!("cargo:rustc-link-lib=static=ggml_kernels");
    println!("cargo:rustc-link-lib=static=marlin_kernels");
    println!("cargo:rustc-link-lib=static=marlin_moe_kernels");
    println!("cargo:rustc-link-lib=static=cutlass_scaled_mm");
    println!("cargo:rustc-link-lib=static=cutlass_standalone_gemm");
    println!("cargo:rustc-link-lib=static=cutlass_moe_grouped");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_silu_mul");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_bias");
    println!("cargo:rustc-link-lib=static=vllm_flash_attn");
    // FA3 is Hopper-only and may not have been built (build.rs in
    // scratchy-builder-cuda skips it on pre-sm_90 hosts). Link only if the
    // .a is present so non-Hopper builds keep working.
    let fa3_lib = std::path::Path::new(&cache_str).join("libvllm_flash_attn_3.a");
    if fa3_lib.exists() {
        println!("cargo:rustc-link-lib=static=vllm_flash_attn_3");
    }
    println!("cargo:rustc-link-lib=static=flashinfer_attn");

    // cudart_static requires rt + dl; cublas/cublasLt remain dynamic.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
