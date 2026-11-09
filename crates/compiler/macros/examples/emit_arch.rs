// SPDX-License-Identifier: Apache-2.0
//! Emit ONE arch's forward, for any arch, on any host — a verification tool.
//!
//! `scratchy-models`' build script gates each arch on a backend
//! (`scratchy_forward_compiler_macro::arch_compiled`, mirroring the
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
//! cargo run -p scratchy-forward-compiler-macro --example emit_arch -- qwen3 "" qwen3-0.6b
//! ```
//!
//! Writes nothing; redirect stdout to capture. Stderr carries progress.

use scratchy_forward_compiler_macro::{Scope, compile_carrier, parse_python_file, render_tokens};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let arch = args.next().unwrap_or_else(|| {
        eprintln!(
            "usage: emit_arch <arch> [presets] [stems...]   (e.g. qwen3, gemma3-mm, deepseek-v2)"
        );
        std::process::exit(2);
    });
    // The quant presets the emission synthesizes variants for — a value, stated
    // here in one place with all three cases:
    //   - `emit_arch qwen3 fp8-dynamic-per-channel,ggml qwen3-0.6b` — a comma
    //     list: only those presets synthesize (what a build with those
    //     `quant/<preset>` features emits).
    //   - `emit_arch qwen3 "" qwen3-0.6b` — the empty string: dense-only,
    //     matching a plain `-Fmodel/<stem>` build with no quant features (the
    //     form `scripts/emit-arch.sh` passes by default).
    //   - `emit_arch qwen3` — no argument at all: every preset and every stem
    //     in the arch, the legacy bare-invocation behaviour.
    let quants = args.next().map(|presets| {
        presets
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect::<std::collections::HashSet<_>>()
    });
    // The models to emit, as explicit stems of this arch — a value too: the
    // scope is never read out of the environment here. Zero stems = every
    // config in the arch (the bare verification run); `scripts/emit-arch.sh`
    // passes the one stem it means as this argument.
    let stems: Vec<String> = args.collect();
    let mut scope = if stems.is_empty() {
        Scope::all()
    } else {
        let pairs: Vec<(&str, &str)> = stems.iter().map(|s| (arch.as_str(), s.as_str())).collect();
        Scope::stems(&pairs)
    };
    if let Some(presets) = quants {
        scope = scope.with_quants(presets);
    }

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
    let tokens = compile_carrier(carrier, &configs_dir, None, &scope)
        .unwrap_or_else(|e| panic!("forward pipeline ({arch}): {e}"));

    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    eprintln!("emit_arch: {arch} ok, {} bytes", rendered.len());
    print!("{rendered}");
}
