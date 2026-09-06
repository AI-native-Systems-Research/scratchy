// SPDX-License-Identifier: Apache-2.0
//! THE BRIDGE 2 ORACLE — our `DataflowIR -> SentientIR` lowering against the reference's own.
//!
//! # 🛑 WHY THIS SHAPE AND NOT THE LIT TESTS
//!
//! `dcc/test/` holds 825 `.mlir` tests, **668 carrying `CHECK-SENT-IR`** expectations — DataflowIR
//! input beside its expected SentientIR, written by the vendor. That is the obvious oracle and it is
//! the wrong one to start with, for a mechanical reason: **their input is TEXT.** Our bridge takes a
//! typed [`islands::dataflow_ir::Run`], and this crate deliberately has no MLIR parser — an island is
//! emit-only and strings are barred from reaching one. Using a lit test therefore needs either a
//! DataflowIR parser (a whole second front) or hand-transcription per test, which is what
//! `examples/golden_xrfbmm.rs` does and costs real effort per case.
//!
//! ⭐⭐ THE DIFFERENTIAL NEEDS NEITHER, BECAUSE BOTH SIDES START FROM **OUR OWN** EMITTED DATAFLOWIR.
//! For each program in the corpus:
//!
//! ```text
//!   our DataflowIR  ──our bridge──►  our SentientIR
//!         │                                │  compare
//!         └────────dcc_standalone──►  reference SentientIR   (the committed golden)
//! ```
//!
//! No parser, no transcription, and the input is a real program rather than a lit test's fixture.
//!
//! # The corpus
//!
//! 18 input/golden pairs in `tests/sentient_corpus/`, three smallest of each of the six program kinds
//! the granite bake produces: `mul`, `matmul`, `add`, `rsqrt`, `mean`, `batchmatmul`.
//!
//! ⛔ THE CORPUS IS A SAMPLE OF 417, NOT THE WHOLE SET. Every one of the 417 programs the bake staged
//! reaches `DataflowToSentient` in the reference with zero parse failures, and goldens for all of them
//! regenerate with `tests/sentient_corpus/REGENERATE.md`. Only 18 are committed because 417 is 6.3 MB.
//! A green run here is evidence over six program shapes, not over the corpus.
//!
//! ⛔⛔ AND IT IS ONE MODEL. `granite-3.1-2b-instruct` at one quant preset. `Model` is a const-generic
//! trait precisely because there are 164 configs across 25 architectures, and a lowering that matches
//! here can still be wrong for a different head dim or a decode rung. Widen the corpus before trusting
//! a green run as coverage — this file cannot tell you it is narrow.
//!
//! # What the goldens actually demand
//!
//! Measured across all 417: the reference's SentientIR uses **seven** of the dialect's 29 ops —
//! `scalar_constant` (3,094), `load_and_store` (943), `load_and_send` (943), `vector_binary` (417),
//! `receive_and_store` (417), `vector_mac` (386), `for` (45). So the port's required output surface is
//! seven ops, not twenty-nine.
//!
//! ⭐ AND `dataflow.get_unit` (2,502) AND `dataflow.program_unit` (1,668) SURVIVE INTO THE GOLDENS,
//! which is the mixed-rung claim confirmed on real output rather than inferred: SentientIR is not the
//! DataflowIR ops replaced.

use std::path::{Path, PathBuf};

/// Where the corpus lives.
fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sentient_corpus")
}

/// Every input/golden pair, by program name.
fn pairs() -> Vec<(String, PathBuf, PathBuf)> {
    let dir = corpus();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("corpus directory") {
        let path = entry.expect("corpus entry").path();
        let name = path.file_name().expect("file name").to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".dfir.mlir") else {
            continue;
        };
        let golden = dir.join(format!("{stem}.sentient.mlir"));
        out.push((stem.to_owned(), path, golden));
    }
    out.sort();
    out
}

/// 🎯 EVERY INPUT HAS ITS GOLDEN, AND BOTH ARE NON-EMPTY.
///
/// ⛔ THIS IS THE ONE ASSERTION THAT CAN FAIL TODAY, and it is worth having on its own: the corpus was
/// assembled by copying pairs, and a `while read` over a file with no trailing newline silently dropped
/// the last golden. An oracle missing half a pair reports success by skipping.
#[test]
fn the_corpus_is_completely_paired() {
    let pairs = pairs();
    assert!(
        pairs.len() >= 18,
        "expected at least the 18 committed pairs, found {}",
        pairs.len()
    );
    for (name, input, golden) in &pairs {
        assert!(golden.is_file(), "{name}: input has no golden at {golden:?}");
        let (i, g) = (
            std::fs::metadata(input).expect("input metadata").len(),
            std::fs::metadata(golden).expect("golden metadata").len(),
        );
        assert!(i > 0 && g > 0, "{name}: empty member ({i} in, {g} golden)");
    }
}

/// 🎯 THE GOLDENS ARE REFERENCE OUTPUT, NOT SOMETHING WE WROTE.
///
/// ⭐ A golden begins with the reference's own dump banner. If a future edit ever regenerates these
/// from our side, this fails — which is the difference between an oracle and a mirror.
#[test]
fn the_goldens_came_from_the_reference() {
    for (name, _, golden) in pairs() {
        let text = std::fs::read_to_string(&golden).expect("golden");
        // ⛔ THE BANNER, NOT THE COMMENT MARKER. `mkgolden.py` slices the reference's output from
        // `index("IR Dump After DataflowToSentient")`, so the leading `// -----// ` is cut. Asserting
        // the marker instead of the banner failed on every golden — and that failure is the reason
        // this test is worth keeping: the artifact, not my memory of it, is the authority.
        assert!(
            text.starts_with("IR Dump After DataflowToSentientLoweringPass (dcc-dataflow-to-sentient)"),
            "{name}: golden does not open with the reference's own dump banner — it may not be \
             reference output"
        );
    }
}

/// 🎯 OUR BRIDGE REPRODUCES THE REFERENCE'S SENTIENTIR.
///
/// ⛔⛔ IGNORED UNTIL THE BRIDGE EXISTS, AND DELIBERATELY PRESENT ANYWAY. The gate is written before
/// the port so that "does it work?" has an answer that predates the work — the same discipline the ISA
/// table's snapshot test used. Remove the `#[ignore]` with the first ported level; do not remove the
/// test to make a run green.
///
/// The comparison should be the OPERATION SEQUENCE plus the addressing, not raw bytes: SSA numbering
/// and attribute-dictionary whitespace are the printer's, and `examples/golden_xrfbmm.rs` already
/// learned that an op-sequence match alone is blind to a wrong subscript — it reported "56 against 56,
/// the sequences agree" for a nest reading a sixty-fourth of its weight tile.
#[test]
#[ignore = "no dataflow_ir_to_sentient bridge yet; this is the gate it must pass"]
fn our_sentient_matches_the_reference() {
    unimplemented!(
        "wire this to bridges::dataflow_ir_to_sentient once level 0 lands: for each pair, lower the \
         input and compare the op sequence and addressing against the golden"
    )
}
