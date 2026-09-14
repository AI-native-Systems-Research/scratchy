// SPDX-License-Identifier: Apache-2.0
//! Under `-Fsendnn`: link the IBM Spyre SDK, so the Rust SuperDSC executor
//! ([`scratchy_target_spyre::superdsc_exec`]) can run a dxp-compiled bundle on the AIU.
//! No-op for every other build (emit-only / KTIR emulator), so a Mac or `runner` build pulls in no SDK.
//!
//! All fxa_* runtime/stream/alloc/DMA/compute symbols are provided natively by `fxa_rust_abi.rs`
//! over flex-rs. flex-rs's own `cxx-shim/senlib_ffi.cpp` is the only C++ TU in this path, built by
//! `flex-rs/build.rs`, not this file.
//!
//! The SDK has no pkg-config; paths come from env with on-pod defaults.

fn main() {
    // 🛑 No `rerun-if-changed` on `csrc/` — that C++ TU is gone (see above), and a
    // watch on a path that does not exist makes cargo re-run this script on EVERY
    // build, which re-fingerprints and rebuilds `scratchy-target-spyre` (measured:
    // 4.3s of rustc on a no-op build). Watch only paths that exist.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SENDNN_INCLUDE");
    println!("cargo:rerun-if-env-changed=SENDNN_LIB");

    if std::env::var_os("CARGO_FEATURE_SENDNN").is_none() {
        return;
    }

    let lib = std::env::var("SENDNN_LIB").unwrap_or_else(|_| "/opt/ibm/spyre/runtime/lib".into());

    // The executor runs the dxp device program DIRECTLY on the AIU via the flex runtime
    // (<flex/flex.hpp> + deeptools' sen_data_convert / processSpyreCodeArtifacts), so the adapter TU
    // needs the flex/senlib/comms/deeptools include roots. These live under $SPYRE_ROOT.
    println!("cargo:rerun-if-env-changed=SPYRE_ROOT");
    let spyre_root = std::env::var("SPYRE_ROOT").unwrap_or_else(|_| "/opt/ibm/spyre".into());

    // CHECK-ONLY ESCAPE HATCH (`SCRATCHY_SKIP_SENDNN_CXX=1`), mirroring `SCRATCHY_PLAN_ONLY_BAKE`
    // and `SCRATCHY_SKIP_CUDA_KERNELS`: skip compiling the adapter so the sendnn-gated RUST can be
    // typechecked on a machine with no SDK. Without it, `cargo check --features sendnn` dies HERE,
    // before any Rust typechecking — which means the whole `#[cfg(feature = "sendnn")]` worker could
    // only ever be compiled on the pod, and compile errors would reach it instead of being caught.
    // Link flags are still emitted, so a real build with this set fails loudly at link time rather
    // than producing a binary with no adapter.
    emit_link_flags(&lib, &spyre_root);
}

/// Linker flags the adapter needs, emitted whether or not we compiled it here.
fn emit_link_flags(lib: &str, _spyre_root: &str) {
    println!("cargo:rustc-link-search=native={lib}");
    let boost_lib = std::env::var("BOOST_LIB").unwrap_or_else(|_| "/usr/lib64".into());
    println!("cargo:rustc-link-search=native={boost_lib}");
    for l in [
        // The SDK shared libs are themselves linked against boost (verified via `ldd`: libflex.so →
        // libboost_log/thread/filesystem/…); the adapter TU includes NO boost headers. Linking
        // these resolves the SDK libs' transitive boost symbols at link time.
        "boost_log",
        "boost_thread",
        "boost_system",
        "boost_filesystem",
    ] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
}
