//! BRIDGE ONE's per-fixture status, MACHINE-CHECKED rather than written down.
//!
//! Needs the `ruff` feature: every case starts from Python SOURCE.
#![cfg(feature = "ruff")]
//!
//! Every fixture configuration the existing toolchain knows about is listed here with the
//! outcome this crate is expected to produce. Two kinds of expectation, and both are
//! asserted:
//!
//!   * [`Expect::Matches`] -- compiles AND agrees with its `.ttir_raw.mlir` golden
//!     structurally, field by field.
//!   * [`Expect::RefusedWithNoGolden`] -- does NOT compile (refusal message pinned), and This is the FAIL-CLOSED half: an unimplemented construct must
//!     produce a named error, never a partial module that looks plausible. A fixture that
//!     starts compiling will fail this test and force the status to be updated with a
//!     golden diff, which is the point.
//!
//! `cargo test --features ruff --test fixture_status -- --nocapture` prints the mismatches.

mod common;

use triton_frontend::codegen;
use triton_frontend::diff;
use triton_frontend::target::Target;

use common::Case;

enum Expect {
    /// Compiles and matches the golden structurally at the given target.
    ///
    /// The target is part of the expectation because the goldens come from real Triton, which
    /// HAS the `f16 / f16 -> f32` promotion this crate turns off for Spyre. Five configurations
    /// never reach a bare f16 divide and match under either; `attention_flash`'s epilogue does,
    /// so its golden is reproduced with `upstream_gpu` and the Spyre difference is asserted
    /// separately in `tests/divergence.rs`.
    MatchesAt(Target),
    /// Compiles, and the golden is KNOWN STALE: the diff must yield exactly the given count
    /// of findings. Red at triton-spyre tip `5c51a1d7a` (re-hosted verbatim), where the
    /// `attention_flash` goldens disagree with the current front end's descriptor mangling
    /// (`TDfp16` vs `Pfp16` on `_attn_fwd_inner`'s fourth descriptor, 11 findings). This
    /// PINS the staleness instead of hiding it: if the golden is regenerated to match
    /// (0 findings) or drifts further, this expectation FAILS and forces the entry back to
    /// [`Expect::MatchesAt`]. A stale golden must not be able to fail silently OR pass
    /// silently.
    MatchesStaleGolden(Target, usize),
    /// Refused, and there is deliberately NO golden either -- the Python toolchain cannot
    /// produce one. (A refused-with-golden expectation existed in the source tree but no
    /// fixture used it; resurrect it with its two match arms when one lands.)
    RefusedWithNoGolden(&'static str),
}

fn status() -> Vec<(Case, Expect)> {
    vec![
        (common::vector_add(), Expect::MatchesAt(Target::spyre())),
        (common::mul(), Expect::MatchesAt(Target::spyre())),
        // `bias_add_f32` reads the module-level global `BIAS = 1.0` inside the kernel.
        // Triton REFUSES that, and it refuses it in `make_ir` -- i.e. inside bridge one --
        // so there is no `.ttir_raw.mlir` for it at all. The brief expected this fixture to
        // have a ttir golden despite having no KTIR; it does not, and
        // `tools/gen_ttir_goldens.py` records it as an asserted XFAIL rather than a skip.
        (
            common::bias_add_f32(),
            Expect::RefusedWithNoGolden("Cannot access global variable BIAS"),
        ),
        // All three SwiGLU configurations, including the one whose smaller BLOCK_K makes the
        // inner `scf.for` a genuinely nested loop.
        (
            common::swiglu("swiglu_mlp", 128, 256, 128),
            Expect::MatchesAt(Target::spyre()),
        ),
        (
            common::swiglu("swiglu_mlp_granite", 4096, 12800, 4096),
            Expect::MatchesAt(Target::spyre()),
        ),
        (
            common::swiglu("swiglu_mlp_tiledk", 128, 256, 64),
            Expect::MatchesAt(Target::spyre()),
        ),
        // The embedding gather: the only fixture with an INDIRECT address, at both the small
        // shape and Granite's own vocabulary (49159 rows, which is deliberately not a
        // multiple of 64 -- see the fixture's delta 3).
        (
            common::embedding("embedding", 512, 128),
            Expect::MatchesAt(Target::spyre()),
        ),
        (
            common::embedding("embedding_granite", 49159, 4096),
            Expect::MatchesAt(Target::spyre()),
        ),
        // RMSNorm at both widths. The reduce is `tl.sum` (a generated `standard.sum` plus its
        // `tt.reduce` region) and the `rsqrt` sits inside the f32 island its dtype check
        // forces -- so the module carries no bare f16 divide and matches under either target.
        (
            common::rmsnorm("rmsnorm", 128),
            Expect::MatchesAt(Target::spyre()),
        ),
        (
            common::rmsnorm("rmsnorm_granite", 4096),
            Expect::MatchesAt(Target::spyre()),
        ),
        // RoPE, at the small head count and at Granite's query and kv head counts. The three
        // differ only in one constant, which is the point: a q launch and a k launch are two
        // configurations of one source (the fixture's delta 6).
        (common::rope("rope", 4), Expect::MatchesAt(Target::spyre())),
        (
            common::rope("rope_granite_q", 32),
            Expect::MatchesAt(Target::spyre()),
        ),
        (
            common::rope("rope_granite_kv", 8),
            Expect::MatchesAt(Target::spyre()),
        ),
        // The whole decoder block, once and then twice in one kernel. Both use `tl.fdiv` for
        // the softmax divide (the fixture's delta 4), so neither reaches a bare f16 divide and
        // both are target-independent. `tests/fusion.rs` carries the experiment's counts.
        (common::decoder_layer(), Expect::MatchesAt(Target::spyre())),
        (
            common::decoder_two_layers(),
            Expect::MatchesAt(Target::spyre()),
        ),
        // Both attention configurations, against the UPSTREAM target: their epilogue is the
        // one bare f16 divide in any fixture, so the golden carries Triton's promotion.
        (
            common::attention("attention_flash_noncausal", 1),
            Expect::MatchesStaleGolden(Target::upstream_gpu(), 11),
        ),
        (
            common::attention("attention_flash_causal", 3),
            Expect::MatchesAt(Target::upstream_gpu()),
        ),
    ]
}

#[test]
fn every_fixture_has_the_expected_status() {
    let mut failures: Vec<String> = Vec::new();
    for (case, expect) in status() {
        let src = common::fixture_src(&case.fixture);
        let golden = common::golden(&case.name);
        let target = match &expect {
            Expect::MatchesAt(t) => *t,
            Expect::MatchesStaleGolden(t, _) => *t,
            _ => Target::spyre(),
        };
        let result = codegen::compile(&src, &case.spec, target);
        match (&expect, &result) {
            (Expect::MatchesAt(_), Ok(m)) => {
                let g = match &golden {
                    Some(g) => g,
                    None => {
                        failures.push(format!(
                            "{}: expected to MATCH but there is no golden",
                            case.name
                        ));
                        continue;
                    }
                };
                let r = diff::compare_to_golden_text(g, m);
                if !r.ok() {
                    failures.push(format!(
                        "{}: expected to MATCH its golden but has {} finding(s):\n{}",
                        case.name,
                        r.findings.len(),
                        r.render()
                    ));
                } else if r.matched_ops == 0 {
                    failures.push(format!(
                        "{}: reported no findings but matched ZERO ops -- comparing nothing",
                        case.name
                    ));
                }
            }
            (Expect::MatchesAt(_), Err(e)) => {
                failures.push(format!("{}: expected to MATCH but was refused: {e}", case.name));
            }
            (Expect::MatchesStaleGolden(t, want), Ok(m)) => {
                let g = match &golden {
                    Some(g) => g,
                    None => {
                        failures.push(format!(
                            "{}: expected a STALE MATCH but there is no golden",
                            case.name
                        ));
                        continue;
                    }
                };
                let r = diff::compare_to_golden_text(g, m);
                let _ = t; // the target is already baked into the compiled module `m`
                let got = r.findings.len();
                if got != *want {
                    failures.push(format!(
                        "{}: the stale-golden PIN moved: {} finding(s) recorded at re-host, \
                         {} now. Either the golden was regenerated (flip back to \
                         Expect::MatchesAt and make the diff pass) or the front end drifted \
                         further (re-record the pin AND file why):\n{}",
                        case.name,
                        want,
                        got,
                        r.render()
                    ));
                } else if r.matched_ops == 0 {
                    failures.push(format!(
                        "{}: reported no findings but matched ZERO ops -- comparing nothing",
                        case.name
                    ));
                }
            }
            (Expect::MatchesStaleGolden(_, _), Err(e)) => {
                failures.push(format!(
                    "{}: expected to compile against a STALE golden but was refused: {e}",
                    case.name
                ));
            }
            (Expect::RefusedWithNoGolden(needle), Err(e)) => {
                let msg = e.to_string();
                if !msg.contains(needle) {
                    failures.push(format!(
                        "{}: refused, but not for the expected reason `{needle}`. Got: {msg}",
                        case.name
                    ));
                }
                if golden.is_some() {
                    failures.push(format!(
                        "{}: a golden EXISTS for a configuration the Python toolchain \
                         cannot compile. One of the two is wrong.",
                        case.name
                    ));
                }
            }
            (Expect::RefusedWithNoGolden(_), Ok(_)) => {
                failures.push(format!(
                    "{}: compiled, but Triton itself refuses this kernel. Compiling it \
                     would mean this front end is MORE permissive than the oracle, which \
                     is a silent-wrong-answer risk, not a feature.",
                    case.name
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} fixture status mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The refusal for an out-of-census construct must name the CONSTRUCT, and the census gate
/// must report EVERY violation at once rather than one per run.
///
/// The two refusal layers are deliberately ordered and this test only exercises the second:
///
///   1. `py::ruff_adapter` refuses Python that has no place in our AST at all (a `while`, a
///      `lambda`, a comprehension). It returns on the first one, because there is nothing to
///      build.
///   2. `py::census::check` then walks the built AST and reports ALL remaining violations --
///      an unsupported `tl.*` target, a `for` over something that is not a range.
///
/// So a kernel with both kinds reports only the parse failure. That is why this test uses
/// two bad CALLS (layer 2) and `an_unsupported_statement_is_named_by_its_ast_node_type`
/// covers layer 1 separately.
#[test]
fn an_out_of_census_construct_is_refused_by_name() {
    let src = r#"
import triton
import triton.language as tl


@triton.jit
def bad_kernel(a_ptr, BLOCK: tl.constexpr):
    x = tl.softmax(a_ptr)
    y = tl.rand(a_ptr)
"#;
    let spec = common::simple_spec("bad_kernel", &[("a_ptr", "*fp16"), ("BLOCK", "constexpr")]);
    let err = codegen::compile(src, &spec, Target::spyre())
        .expect_err("a kernel using tl.softmax, tl.rand and `while` must not compile");
    let msg = err.to_string();
    for needle in ["tl.softmax", "tl.rand"] {
        assert!(
            msg.contains(needle),
            "the refusal must NAME `{needle}`; got: {msg}"
        );
    }
    assert!(
        msg.contains("census"),
        "the refusal should say the census is the boundary; got: {msg}"
    );
}

/// A `while` loop is not in the census, and the parser refuses it by its CPython node name.
#[test]
fn an_unsupported_statement_is_named_by_its_ast_node_type() {
    let src = r#"
import triton
import triton.language as tl


@triton.jit
def bad_kernel(a_ptr, BLOCK: tl.constexpr):
    while True:
        pass
"#;
    let spec = common::simple_spec("bad_kernel", &[("a_ptr", "*fp16"), ("BLOCK", "constexpr")]);
    let err = codegen::compile(src, &spec, Target::spyre())
        .expect_err("a `while` loop must not compile");
    let msg = err.to_string();
    assert!(
        msg.contains("While"),
        "the refusal must name the AST node type `While`; got: {msg}"
    );
}
