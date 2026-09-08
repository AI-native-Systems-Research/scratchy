// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `Splat.cpp` — 2 of bridge 2's 384 functions (dependency level(s) [2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e235_createSentientConstants` | 235/384 | 31 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34` |
//! | `e279_createSplatOperation` | 279/384 | 110 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70` |

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::{ScalarTy, Vector};
use crate::islands::sentient::dialects::{self as sen, Val, sentient, vectorchain};

/// HOW WIDE THE BITSTREAM A SPLAT'S CONSTANT IS PACKED INTO IS — the literal `128` of
/// `Splat.cpp:55`, and what fixes the emitted vector's length independently of how many values the
/// `constant_bitstream` carries.
const BITSTREAM_BITS: u32 = 128;

/// THE CONSTANT A SPLAT LOWERS TO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentientConstant {
    /// The op that replaces the `vectorchain.constant_bitstream`.
    pub op: sen::Op,
    /// Its result — `sentient_const_op.getResult()`.
    pub value: Val,
}

/// Replaces: e235_createSentientConstants
///
/// One value in the `constant_bitstream` becomes a `sentient.scalar_constant` carrying its
/// `is_symbol`; more than one becomes a `sentient.vector_constant` whose elements are the shuffle's
/// indices read off those values.
///
/// ⛔ TRAP: the vector's length is `128 / bitwidth` — NOT `value.len()` — and the reference
/// `DT_CHECK`s the shuffle has exactly that many indices. That check is the `None` here, which its
/// caller (entry 279) already returns a `LogicalResult` for.
#[must_use]
pub fn create_sentient_constants(
    const_bit_op: &vectorchain::Op,
    shuffle_op: &vectorchain::Op,
    values: &mut Values,
) -> Option<SentientConstant> {
    let vectorchain::Op::ConstantBitstream {
        value: vals,
        ty,
        is_symbol,
        ..
    } = const_bit_op
    else {
        return None;
    };
    if let [val] = vals.as_slice() {
        let value = values.mint();
        return Some(SentientConstant {
            op: sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: *val,
                result: value,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::of_elem(ty.elem),
                is_symbol: *is_symbol,
            }),
            value,
        });
    }
    let vectorchain::Op::Shuffle { indices, .. } = shuffle_op else {
        return None;
    };
    let total_elements = BITSTREAM_BITS / ty.elem.bits();
    if indices.len() != total_elements as usize {
        return None;
    }
    let mut extended_vals = Vec::with_capacity(indices.len());
    for idx in indices {
        extended_vals.push(*vals.get(usize::try_from(*idx).ok()?)?);
    }
    let value = values.mint();
    Some(SentientConstant {
        op: sen::Op::Sentient(sentient::Op::VectorConstant {
            value: extended_vals,
            result: value,
            ty: Vector {
                len: u64::from(total_elements),
                elem: ty.elem,
            },
        }),
        value,
    })
}
// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e279_createSplatOperation

#[cfg(test)]
mod unit_tests {
    use super::{SentientConstant, create_sentient_constants};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::{ElemType, ScalarTy, Vector};
    use crate::islands::sentient::dialects::{self as sen, Val, sentient, vectorchain};

    /// `dcc/test/.../splat_const_bit.mlir:20` takes the scalar arm and
    /// `shuffle_splat_pattern.mlir:30` the vector one, from the same two-value bitstream.
    #[test]
    fn one_value_is_a_scalar_constant_and_two_are_a_shuffled_vector_of_eight() {
        let f16 = |len| Vector {
            len,
            elem: ElemType::F16,
        };
        let mut values = Values::default();
        let shuffle = vectorchain::Op::Shuffle {
            result: Val(3),
            input: Val(2),
            variable: Vec::new(),
            indices: vec![0, 1, 1, 1, 1, 1, 1, 1],
            repetition: 8,
            input_ty: f16(2),
            ty: f16(8),
        };

        let one = vectorchain::Op::ConstantBitstream {
            result: Val(2),
            value: vec![0xff],
            ty: f16(1),
            is_symbol: false,
        };
        let scalar = create_sentient_constants(&one, &shuffle, &mut values);
        assert_eq!(
            scalar,
            Some(SentientConstant {
                op: sen::Op::Sentient(sentient::Op::ScalarConstant {
                    value: 0xff,
                    result: Val(0),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Elem(ElemType::F16),
                    is_symbol: false,
                }),
                value: Val(0),
            }),
            "`%0 = sentient.scalar_constant {{value = 0xff : si64}} : f16`"
        );

        let two = vectorchain::Op::ConstantBitstream {
            result: Val(2),
            value: vec![0xffff, 0x0],
            ty: f16(2),
            is_symbol: false,
        };
        let vector = create_sentient_constants(&two, &shuffle, &mut values);
        assert_eq!(
            vector,
            Some(SentientConstant {
                op: sen::Op::Sentient(sentient::Op::VectorConstant {
                    value: vec![0xffff, 0, 0, 0, 0, 0, 0, 0],
                    result: Val(1),
                    ty: f16(8),
                }),
                value: Val(1),
            }),
            "`sentient.vector_constant {{value = [0xffff, 0x0, …]}} : vector<8xf16>`"
        );
    }
}
