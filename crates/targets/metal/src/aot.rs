// SPDX-License-Identifier: Apache-2.0
//! AOT-compile MSL source to a `.metallib` blob via `xcrun metal -c`
//! and `xcrun metallib`. Shared by `scratchy-forward-compiler-macro` (proc-macro
//! expansion baking) and `scratchy-cost-sweep-metal` (benchmark of
//! the synthesized kernels).
//!
//! Same flow as `scratchy-target-metal/build.rs`.
//!
//! Panics on `xcrun` failure — synth compile errors are build errors
//! that need to surface, not runtime soft-fail.

use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-call sequence so concurrent compiles of the SAME synth symbol get
/// distinct temp dirs. Many arches/models share synth kernels, and the
/// consolidated `scratchy-models` build script runs this from a rayon fan-out,
/// so `std::process::id()` is identical across the racing threads — without a
/// unique suffix they clobber each other's `.metal`/`.air` files (and one
/// thread's cleanup deletes another's dir mid-compile).
static SYNTH_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn aot_compile_metallib(symbol: &str, source: &str) -> Vec<u8> {
    // Skip on non-macOS hosts (no `xcrun`). The synth metallibs are
    // only ever consumed by the metal backend.
    let host_os =
        std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| std::env::consts::OS.to_string());
    if host_os != "macos" {
        return Vec::new();
    }

    let tmp_dir = std::env::temp_dir().join(format!(
        "scratchy-synth-{}-{}-{}",
        symbol,
        std::process::id(),
        SYNTH_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&tmp_dir)
        .unwrap_or_else(|e| panic!("synth: create tmp dir {tmp_dir:?}: {e}"));

    let metal_path = tmp_dir.join(format!("{symbol}.metal"));
    // Only the final artifact needs an absolute path (it is read back
    // below). The compiler steps run with `tmp_dir` as their working
    // directory and take BARE filenames, so no build-specific path can
    // reach the emitted bytes — see the flag comment below.
    let metallib_path = tmp_dir.join(format!("{symbol}.metallib"));

    let mut f = std::fs::File::create(&metal_path)
        .unwrap_or_else(|e| panic!("synth: create {metal_path:?}: {e}"));
    f.write_all(source.as_bytes())
        .unwrap_or_else(|e| panic!("synth: write {metal_path:?}: {e}"));
    drop(f);

    // Run the compiler INSIDE `tmp_dir` and hand it BARE filenames.
    // `-frecord-sources` bakes the source path it was given into the
    // blob's DEPF section, so an absolute path here puts this build's
    // PID and sequence number inside the emitted `&[u8]` — two
    // identical builds then differ by five lines and metal emission is
    // not byte-reproducible, which voids any bit-equality gate over the
    // generated code. Relative names record as `<symbol>.metal`.
    // The directory stays PID+seq unique: that is what keeps concurrent
    // compiles of the same symbol from clobbering each other, and it no
    // longer leaks into the output.
    let status = Command::new("xcrun")
        .current_dir(&tmp_dir)
        .args([
            "-sdk", "macosx", "metal", "-O3",
            // NO `-frecord-sources`. It attaches a source archive whose
            // SARC section records the compiler's WORKING DIRECTORY —
            // this build's PID-and-sequence temp dir — directly inside
            // the emitted `&[u8]`. Two identical builds then produce
            // different generated code, which voids any bit-equality
            // gate over metal emission. Dropping it
            // also shrinks each synth blob ~4x (13,976 -> 3,479 bytes
            // measured on a trivial kernel).
            //
            "-c",
        ])
        .arg(format!("{symbol}.metal"))
        .arg("-o")
        .arg(format!("{symbol}.air"))
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metal: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metal` failed for `{symbol}` (source at {metal_path:?})");
    }

    let status = Command::new("xcrun")
        .current_dir(&tmp_dir)
        .args(["-sdk", "macosx", "metallib"])
        .arg(format!("{symbol}.air"))
        .arg("-o")
        .arg(format!("{symbol}.metallib"))
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metallib: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metallib` failed for `{symbol}`");
    }

    let bytes = std::fs::read(&metallib_path)
        .unwrap_or_else(|e| panic!("synth: read {metallib_path:?}: {e}"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
    bytes
}
