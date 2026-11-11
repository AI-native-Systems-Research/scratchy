//! Links the crate's C-ABI implementation: two STATIC archives, never a
//! shared object and never a runtime loader.
//!
//! The declarations live in `src/senlib_ffi.rs` (names, signatures, and the
//! per-door contract prose). Their implementation is built on card hosts
//! from the IBM-internal trees and staged as archives:
//!
//!   - `libflex_shim.a`  — the FFI export layer over the native port of the
//!     senlib doors (the mirror of this crate's declarations);
//!   - `libsenlib_oco.a` — the export-controlled OCO crate the FFI layer's
//!     own undefined symbols resolve against.
//!
//! Both are found in the one directory named by `SENLIB_STATIC_LIB_DIR` at
//! build time; there is no search, no fallback directory, and no default —
//! a build that names no directory, or whose directory is missing either
//! archive, REFUSES here, naming exactly what is missing. What executes on
//! a card host is provable from the linked binary itself: the archives are
//! embedded, not loaded.
//!
//! `SCRATCHY_SKIP_SENDNN_CXX=1` skips emitting the link directives — the
//! check/clippy escape hatch (cargo check and clippy run build scripts but
//! perform no final link). A real link under the escape hatch fails loudly
//! at `ld`. This is the same contract the escape hatch always carried.
//!
//! This is the only build script in the spyre tree.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=SCRATCHY_SKIP_SENDNN_CXX");
    println!("cargo:rerun-if-env-changed=SENLIB_STATIC_LIB_DIR");

    if env::var_os("SCRATCHY_SKIP_SENDNN_CXX").is_some_and(|v| v == "1") {
        // Check/clippy only: no link directives are emitted, so no archives
        // are needed. A link will fail at `ld` with undefined references to
        // every door — loud, by design.
        return;
    }

    let missing_dir_reason = |dir_name: &str| -> String {
        format!(
            "flex-rs links its C-ABI implementation from STATIC archives, and the build \
             cannot proceed without them. Set SENLIB_STATIC_LIB_DIR to the directory \
             holding libflex_shim.a and libsenlib_oco.a (both built on card hosts from \
             the IBM-internal trees; the archives embed into the binary, so the \
             resulting executable is a card-host artifact). {dir_name}"
        )
    };

    let dir: PathBuf = env::var_os("SENLIB_STATIC_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "{}",
                missing_dir_reason("SENLIB_STATIC_LIB_DIR is not set at all.")
            )
        });

    for archive in ["libflex_shim.a", "libsenlib_oco.a"] {
        if !dir.join(archive).is_file() {
            panic!(
                "{}",
                missing_dir_reason(&format!(
                    "The named directory ({}) does not contain {archive}.",
                    dir.display()
                ))
            );
        }
    }

    // The FFI archive's own undefined OCO symbols resolve against the OCO
    // archive at this link: dependent first, provider after (static-archive
    // resolution is left-to-right).
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=flex_shim");
    println!("cargo:rustc-link-lib=static=senlib_oco");
}
