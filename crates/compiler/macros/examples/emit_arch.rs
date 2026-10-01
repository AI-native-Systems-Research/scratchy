// SPDX-License-Identifier: Apache-2.0
//! Emit ONE arch's forward, for any arch, on any host — a verification tool.
//!
//! `scratchy-models`' build script gates each arch on a backend
//! (`SPYRE_CAPABLE` / `CUDA_ONLY` in `scratchy-forwards.rs`, mirroring the
//! `pub mod` cfgs in its `src/lib.rs`), so on a machine with neither a CUDA
//! toolchain nor macOS only `llama` and `granite` can be built at all. That
//! makes the other 23 arches' codegen unverifiable via `cargo check` there,
//! even though parse → classify → shape → CFG → unroll → schedule → codegen
//! is entirely target-neutral up to the final emit.
//!
//! This driver runs that same pipeline directly — the identical
//! `parse_python_file` + `compile_carrier` + `render_tokens` sequence
//! `emit_arch()` uses — with no backend gate, and prints the rendered
//! emit to stdout. Diffing its output across a refactor proves the
//! change was semantics-preserving for arches the host cannot compile.
//!
//! ```text
//! cargo run -p scratchy-forward-compiler-macro --example emit_arch -- qwen3
//! ```
//!
//! Writes nothing; redirect stdout to capture. Stderr carries progress.

use scratchy_forward_compiler_macro::{compile_carrier, parse_python_file, render_tokens};
use std::path::PathBuf;

fn main() {
    let arch = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: emit_arch <arch>   (e.g. qwen3, gemma3-mm, deepseek-v2)");
        std::process::exit(2);
    });

    // This example lives in crates/compiler/macros/, so the repo root is
    // three levels up. Resolved from CARGO_MANIFEST_DIR so the tool works
    // from any cwd.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("resolve repo root");
    let models = root.join("crates/models/arch");
    let dsl_path = models.join(format!("dsl/{arch}.py"));
    let configs_dir = models.join(format!("configs/{arch}"));
    assert!(dsl_path.is_file(), "no DSL at {}", dsl_path.display());
    assert!(
        configs_dir.is_dir(),
        "no configs at {}",
        configs_dir.display()
    );

    eprintln!("emit_arch: {arch}");
    let text = std::fs::read_to_string(&dsl_path).expect("read DSL");
    let carrier =
        parse_python_file(&text).unwrap_or_else(|e| panic!("carrier {}: {e}", dsl_path.display()));
    let tokens = compile_carrier(carrier, &configs_dir)
        .unwrap_or_else(|e| panic!("forward pipeline ({arch}): {e}"));

    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    eprintln!("emit_arch: {arch} ok, {} bytes", rendered.len());
    print!("{rendered}");
}
