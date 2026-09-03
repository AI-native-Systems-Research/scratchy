//! Compiles and links `cxx-shim/senlib_ffi.cpp` — the real C++ senlib FFI
//! binding (see its own file header). Unconditional by default — flex-rs is
//! the only fxa_* path, so its shim must build whenever anything in this
//! workspace actually links against real senlib, with no separate opt-in.
//!
//! This links ONLY against `senlib` (+ its own transitive deps: flightlog,
//! boost). It does NOT link `libflex.so` or any `sendnn_*` library — this
//! crate has no `flex::` dependency at all, by design.
//!
//! `SCRATCHY_SKIP_SENDNN_CXX=1` skips the compile (cargo check/clippy only —
//! a link will fail): the SAME escape hatch `scratchy-target-spyre`'s own
//! `build.rs` already uses for `spyre_sdk_abi.cpp`, so a laptop with no
//! Spyre SDK headers/libs stays checkable without a second, flex-rs-specific
//! toggle to remember.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=SCRATCHY_SKIP_SENDNN_CXX");
    println!("cargo:rerun-if-env-changed=FLEX_RS_SPYRE_ROOT");
    println!("cargo:rerun-if-changed=cxx-shim/senlib_ffi.cpp");

    let spyre_root: PathBuf = env::var("FLEX_RS_SPYRE_ROOT")
        .unwrap_or_else(|_| "/opt/ibm/spyre".to_string())
        .into();
    let senlib_lib = spyre_root.join("senlib/lib");
    println!("cargo:rustc-link-search=native={}", senlib_lib.display());
    let boost_lib = env::var("BOOST_LIB").unwrap_or_else(|_| "/usr/lib64".to_string());
    println!("cargo:rustc-link-search=native={boost_lib}");
    // Link only senlib itself + its own real transitive deps (verified via
    // `ldd libsenlib.so` on the pod: flightlog + this boost set). No `flex`,
    // no `sendnn_*` — this crate has zero dependence on libflex.so. Emitted
    // whether or not we compile below, matching spyre_sdk_abi.cpp's own
    // `emit_link_flags` split (a real build with the shim skipped fails
    // loudly at link time instead of silently missing symbols).
    for l in [
        "senlib",
        "flightlog",
        "boost_log",
        "boost_log_setup",
        "boost_thread",
        "boost_filesystem",
        "boost_atomic",
        "boost_regex",
        "boost_json",
        "boost_container",
        "boost_locale",
        "boost_chrono",
    ] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }

    if env::var_os("SCRATCHY_SKIP_SENDNN_CXX").is_some() {
        println!(
            "cargo:warning=SCRATCHY_SKIP_SENDNN_CXX set — senlib_ffi.cpp NOT compiled (cargo check \
             only; a link will fail)"
        );
        return;
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        // senlib headers (parse_init_buffer.hpp) use std::span (C++20).
        .std("c++20")
        .file("cxx-shim/senlib_ffi.cpp")
        .warnings(true);
    // senlib's own include root, plus `runtime/include` for the `common/`
    // headers senlib's own public headers (device_identity.hpp) need to
    // parse `common/self_doc_enum.hpp`. This crate's own .cpp code never
    // includes anything under `flex/`, and only senlib itself is linked
    // above — no libflex.so, no sendnn_*.
    for sub in [
        "senlib/include",
        "senlib/flightlog/include",
        "runtime/include",
    ] {
        build.include(spyre_root.join(sub));
    }
    build.compile("senlib_ffi");
}
