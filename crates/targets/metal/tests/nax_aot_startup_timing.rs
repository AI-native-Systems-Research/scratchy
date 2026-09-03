// SPDX-License-Identifier: Apache-2.0
//! Quantify the TTFT lever of the NAX AOT switch: time the runtime
//! `newLibraryWithSource` JIT compile of `quantized_qmm_nax` against the
//! AOT `newLibraryWithData` load of the build.rs-embedded metallib.
#![cfg(target_os = "macos")]

use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::shader_cache::{
    compile_nax_library_from_source, load_library_from_bytes,
};

#[test]
#[ignore = "startup timing probe — run with --ignored --nocapture"]
fn nax_jit_vs_aot_library_build_time() {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    // AOT: newLibraryWithData of the build.rs-embedded metallib.
    let bytes: &'static [u8] = scratchy_target_metal::embedded_nax_metallib();
    // warm
    let _ = load_library_from_bytes(&device, bytes).expect("aot");
    let mut aot = Vec::new();
    for _ in 0..10 {
        let t = std::time::Instant::now();
        let _lib = load_library_from_bytes(&device, bytes).expect("aot");
        aot.push(t.elapsed().as_secs_f64() * 1e3);
    }
    aot.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // JIT: newLibraryWithSource compile of the same source.
    let _ = compile_nax_library_from_source(&device).expect("jit warm");
    let mut jit = Vec::new();
    for _ in 0..10 {
        let t = std::time::Instant::now();
        let _lib = compile_nax_library_from_source(&device).expect("jit");
        jit.push(t.elapsed().as_secs_f64() * 1e3);
    }
    jit.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!(
        "NAX library build: AOT(newLibraryWithData) median={:.2} ms  JIT(newLibraryWithSource) median={:.2} ms  saving={:.2} ms",
        aot[aot.len() / 2],
        jit[jit.len() / 2],
        jit[jit.len() / 2] - aot[aot.len() / 2],
    );
}
