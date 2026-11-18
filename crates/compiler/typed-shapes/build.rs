// SPDX-License-Identifier: Apache-2.0
//! Writes `$OUT_DIR/signature_bounds.rs`, the markers of the bounds the op signatures name, which
//! `src/lib.rs` includes. They come from the front end's `typed_emit::SIGNATURE_BOUNDS`, the one
//! list. The front end is a build-dependency, so a change to that list rebuilds this script and
//! reruns it; the script reads no other file.

use scratchy_forward_compiler_frontend::typed_emit;

fn main() {
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"))
        .join("signature_bounds.rs");
    std::fs::write(&out, typed_emit::emit_signature_markers())
        .unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}
