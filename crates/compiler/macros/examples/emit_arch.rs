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
//! `parse_carrier` + `compile_in_dir` + `render_tokens` sequence
//! `emit_arch()` uses — with no backend gate, and prints the rendered
//! emit to stdout. Diffing its output across a refactor proves the
//! change was semantics-preserving for arches the host cannot compile.
//!
//! ```text
//! cargo run -p scratchy-forward-compiler-macro --example emit_arch -- qwen3
//! ```
//!
//! Writes nothing; redirect stdout to capture. Stderr carries progress.

use scratchy_forward_compiler_macro::{
    CompileMode, DEFAULT_DECODER_WORKLOADS, ForwardArgs, compile_in_dir, parse_carrier,
    render_tokens,
};
use std::path::PathBuf;

/// Mirrors `scratchy-forwards.rs`'s `args_from_attr`: the carrier's
/// attribute args, defaulted to the standard decoder workload set when
/// the DSL doesn't name its own.
fn args_from_attr(attr: &syn::Attribute) -> ForwardArgs {
    let mut args: ForwardArgs = match &attr.meta {
        syn::Meta::List(list) => syn::parse2(list.tokens.clone()).expect("parse #[forward] args"),
        _ => syn::parse_str("").expect("empty #[forward] args"),
    };
    if args.workloads.is_empty() {
        args.workloads = DEFAULT_DECODER_WORKLOADS.to_vec();
    }
    args
}

fn carrier_attr(attrs: &[syn::Attribute]) -> Option<(CompileMode, usize)> {
    attrs.iter().enumerate().find_map(|(i, a)| {
        match a
            .path()
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .as_deref()
        {
            Some("forward") => Some((CompileMode::DECODER, i)),
            Some("vision_forward") => Some((CompileMode::VISION, i)),
            _ => None,
        }
    })
}

fn item_attrs_mut(item: &mut syn::Item) -> Option<&mut Vec<syn::Attribute>> {
    match item {
        syn::Item::Fn(f) => Some(&mut f.attrs),
        syn::Item::Mod(m) => Some(&mut m.attrs),
        _ => None,
    }
}

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
    let dsl_path = models.join(format!("dsl/{arch}.rs.in"));
    let configs_dir = models.join(format!("configs/{arch}"));
    assert!(dsl_path.is_file(), "no DSL at {}", dsl_path.display());
    assert!(
        configs_dir.is_dir(),
        "no configs at {}",
        configs_dir.display()
    );

    eprintln!("emit_arch: {arch}");
    let text = std::fs::read_to_string(&dsl_path).expect("read DSL");
    let file = syn::parse_file(&text).expect("parse DSL file");

    let mut carrier_item = None;
    let mut mode = CompileMode::DECODER;
    let mut args = None;
    for mut item in file.items {
        let found = item_attrs_mut(&mut item).and_then(|attrs| carrier_attr(attrs));
        if let Some((m, i)) = found {
            let attr = item_attrs_mut(&mut item).unwrap().remove(i);
            mode = m;
            args = Some(args_from_attr(&attr));
            carrier_item = Some(item);
            break;
        }
    }
    let carrier_item = carrier_item.expect("no #[forward]/#[vision_forward] carrier");

    let carrier = parse_carrier(carrier_item).expect("parse carrier");
    let tokens = compile_in_dir(&args.unwrap(), &carrier, mode, &configs_dir)
        .unwrap_or_else(|e| panic!("forward pipeline ({arch}): {e}"));

    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    eprintln!("emit_arch: {arch} ok, {} bytes", rendered.len());
    print!("{rendered}");
}
