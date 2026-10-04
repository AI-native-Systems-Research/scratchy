//! What the DEPENDENCY-FREE build is held to.
//!
//! This file has no `#![cfg(feature = "ruff")]`, so it runs in both configurations. It
//! exercises the parts of the crate that must keep working with zero dependencies on the
//! offline pod: the TTIR value type, the golden reader, the printer, the structural diff, and
//! the target policy. Only the step that turns Python TEXT into an AST needs ruff.
//!
//! `cargo test --no-default-features` runs exactly these.

use triton_frontend::diff;
use triton_frontend::target::Target;
use triton_frontend::ttir::{self, Attr, Type};

fn golden(name: &str) -> String {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{name}.ttir_raw.mlir"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// EVERY raw golden must parse, including the ones no fixture case can compile yet.
///
/// This is the load-bearing test for the golden reader: a subset parser that silently drops
/// what it does not understand would make every future diff pass while comparing less and
/// less. `ttir::parse` returns an `Err` naming the line instead, and this asserts it never
/// needs to.
#[test]
fn every_raw_golden_parses() {
    let names = [
        "vector_add",
        "mul",
        "swiglu_mlp",
        "swiglu_mlp_granite",
        "swiglu_mlp_tiledk",
        "attention_flash_noncausal",
        "attention_flash_causal",
    ];
    let mut failures = Vec::new();
    for n in names {
        match ttir::parse::parse_module(&golden(n)) {
            Ok(m) => {
                if m.funcs.is_empty() {
                    failures.push(format!("{n}: parsed but produced ZERO functions"));
                }
                // Every function must have at least one op, or the parse consumed nothing.
                for f in &m.funcs {
                    let ops = f.body.blocks.first().map(|b| b.ops.len()).unwrap_or(0);
                    if ops == 0 {
                        failures.push(format!("{n}: function `{}` parsed with no ops", f.name));
                    }
                }
            }
            Err(e) => failures.push(format!("{n}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} golden(s) did not parse cleanly:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The goldens the fixtures cannot reach yet still carry the ops the remaining work needs.
/// Asserted so a future change to the parser cannot quietly stop seeing them.
#[test]
fn the_unreached_goldens_contain_the_constructs_still_to_do() {
    let m = ttir::parse::parse_module(&golden("swiglu_mlp")).expect("swiglu golden parses");
    let mut seen = std::collections::BTreeSet::new();
    fn walk(r: &ttir::Region, seen: &mut std::collections::BTreeSet<String>) {
        for b in &r.blocks {
            for op in &b.ops {
                seen.insert(op.name.clone());
                for sub in &op.regions {
                    walk(sub, seen);
                }
            }
        }
    }
    for f in &m.funcs {
        walk(&f.body, &mut seen);
    }
    for needed in ["tt.call", "scf.for", "scf.yield", "tt.dot", "ub.poison"] {
        assert!(
            seen.contains(needed),
            "the swiglu golden should contain `{needed}` -- it is on the remaining-work list. \
             Seen: {seen:?}"
        );
    }
    // Multiple functions: the kernel plus the generated `standard.zeros` helpers.
    assert!(
        m.funcs.len() >= 3,
        "swiglu's golden should have the kernel plus generated private helpers, got {}",
        m.funcs.len()
    );
    assert!(
        m.funcs
            .iter()
            .any(|f| f.name.starts_with("triton.language.standard.zeros__")),
        "the generated `zeros` helper should be present with its mangled name, got {:?}",
        m.funcs.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
}

/// A module printed and read back must compare clean against itself. That keeps the two test
/// instruments honest about each other: if the printer emits something the parser cannot read,
/// or reads back differently, the diff would be measuring the instruments rather than the
/// front end.
#[test]
fn print_then_parse_round_trips_structurally() {
    for name in ["vector_add", "embedding", "decoder_layer"] {
        let original = ttir::parse::parse_module(&golden(name)).expect("golden parses");
        let text = ttir::print::print_module(&original);
        let reparsed = match ttir::parse::parse_module(&text) {
            Ok(m) => m,
            Err(e) => panic!("our own printed output did not parse:\n{e}\n---\n{text}"),
        };
        let r = diff::compare(&original, &reparsed);
        assert!(
            r.ok(),
            "print -> parse is not structurally idempotent:\n{}\n--- printed ---\n{}",
            r.render(),
            text
        );
        assert!(r.matched_ops > 0, "round trip matched zero ops");
    }
}

/// The type spellings must round-trip, since every result-type comparison in the diff is a
/// string comparison of these.
#[test]
fn type_spellings_round_trip() {
    let cases = [
        "i1",
        "i32",
        "i64",
        "f16",
        "f32",
        "bf16",
        "!tt.ptr<f16>",
        "tensor<64xf16>",
        "tensor<64x128xf16>",
        "tensor<4096x64xf16>",
        "!tt.tensordesc<64x128xf16>",
        "tensor<64xi32>",
        // A descriptor's BLOCK element carries Triton's signedness where nothing else does;
        // `embedding`'s golden is the only one that has it. See `ttir::desc_block_elem`.
        "!tt.tensordesc<64xsi32>",
        "!tt.tensordesc<64xui32>",
        "!tt.tensordesc<1x128xf16>",
    ];
    for c in cases {
        let t = ttir::parse::parse_type(c).unwrap_or_else(|| panic!("`{c}` did not parse"));
        assert_eq!(t.to_string(), c, "`{c}` did not round-trip");
    }
}

/// The dependency-free build still has the target policy, and the two presets must differ --
/// otherwise the divergence is not actually wired to anything.
#[test]
fn the_target_presets_differ_on_the_divergence() {
    assert!(
        !Target::spyre().div_promotes_narrow_floats,
        "Spyre must NOT promote f16 division to f32"
    );
    assert!(
        Target::upstream_gpu().div_promotes_narrow_floats,
        "the upstream preset must keep Triton's promotion, or it is not a control"
    );
    assert_eq!(
        Target::default(),
        Target::spyre(),
        "the default target must be Spyre"
    );
}

/// The diff must not be fooled by a same-shape, same-op module with one attribute changed --
/// checked here too, on a golden-vs-itself basis, so the dependency-free build carries at
/// least one planted-difference control of its own.
#[test]
fn control_planted_difference_is_caught_without_the_parser_feature() {
    let expected = ttir::parse::parse_module(&golden("vector_add")).expect("golden parses");
    let mut mutated = expected.clone();
    let mut hit = false;
    for op in &mut mutated.funcs[0].body.blocks[0].ops {
        if op.name == "arith.constant" {
            if let Some(Attr::Int(v, t)) = op.attrs.get("value").cloned() {
                op.attrs.insert("value".to_string(), Attr::Int(v + 1, t));
                hit = true;
                break;
            }
        }
    }
    assert!(hit, "control found no integer constant to mutate");
    let r = diff::compare(&expected, &mutated);
    assert!(
        r.findings.iter().any(|f| f.contains(".attr[value]:")),
        "a planted attribute change must be caught by name. Findings: {:?}",
        r.findings
    );

    // And a shape change, which a type-only comparison would miss.
    let mut mutated = expected.clone();
    let target = mutated.funcs[0].body.blocks[0]
        .ops
        .iter()
        .find(|o| o.name == "tt.descriptor_load")
        .and_then(|o| o.results.first().copied())
        .expect("no tt.descriptor_load");
    mutated.values[target.0 as usize].ty = Type::tensor(&[32], Type::f16());
    let r = diff::compare(&expected, &mutated);
    assert!(
        r.findings
            .iter()
            .any(|f| f.contains(".result_types:") && f.contains("tensor<32xf16>")),
        "a planted shape change must be caught by name. Findings: {:?}",
        r.findings
    );
}
