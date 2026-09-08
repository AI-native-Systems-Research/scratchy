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

use super::vc_vector_chain_helper::{get_mask_value_for_non_pt, input_precision_from_operand};
use super::vc_vector_operands::{
    ComputeComp, VectorOperand, defining_position, expanded_shuffle_indices, op_at,
};
use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::Op as DfirOp;
use crate::islands::dataflow_ir::ty::{ScalarTy, Vector};
use crate::islands::sentient::dialects::{
    self as sen, Val, agen, arith, dataflow, sentient, uniform, vectorchain,
};

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
// ══════════════════════════════════════════════════════════════════════════════════════════════
// 279/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `isFirstElemSplat` (`dialect_utils/VectorChain/Utils.cpp:80-88`) — every expanded index is 0.
///
/// ⛔ NOT ONE OF THE 384, and neither is its sibling below: `dialect_utils` sits outside bridge 2's
/// list, and entry 279 IS its three branch tests, so a port without them decides nothing.
fn is_first_elem_splat(indices: &[i32], repetition: u32, variables: usize, pads: usize) -> bool {
    expanded_shuffle_indices(indices, repetition, variables, pads)
        .iter()
        .all(|index| *index == 0)
}

/// `isPadLeftFor8FirstElemSplat` (`dialect_utils/VectorChain/Utils.cpp:109-122`) — the input's first
/// element in every eighth lane and the first pad operand in the other seven.
///
/// ⛔ NO PAD SEGMENT IS FALSE, AND THAT IS THE `std::optional` COMPARISON: `indices_expanded[i] !=
/// getFirstPadIndex()` is an `int` against an optional, which C++ answers `!=` for whenever the
/// optional is empty, so lane 1 returns at once. ⛔ AND THE FIRST PAD INDEX IS
/// `-(variable.len() + 1)`, `-1` only where there are no variables (`VectorChain.td:487-492`).
fn is_pad_left_for_8_first_elem_splat(
    indices: &[i32],
    repetition: u32,
    variables: usize,
    pads: usize,
) -> bool {
    if pads == 0 {
        return false;
    }
    let pad_index = -i32::try_from(variables).unwrap_or(i32::MAX) - 1;
    expanded_shuffle_indices(indices, repetition, variables, pads)
        .iter()
        .enumerate()
        .all(|(lane, index)| {
            if lane % 8 == 0 {
                *index == 0
            } else {
                *index == pad_index
            }
        })
}

/// `sentient.scalar_constant {value = 0 : si64} : index` — the splat's default mask operand.
fn index_zero(result: Val) -> sen::Op {
    sen::Op::Sentient(sentient::Op::ScalarConstant {
        value: 0,
        result,
        reg_locale: sentient::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    })
}

/// `sentient.logical_port {portName = ..} : index` — `symbolizeSentientComputePort(name).value()`.
fn logical_port(port_name: sentient::Port, result: Val) -> sen::Op {
    sen::Op::Sentient(sentient::Op::LogicalPort { port_name, result })
}

/// Replaces: e279_createSplatOperation
///
/// The ops one `vectorchain.shuffle` — or one `arith.constant` splat feeding an immediate copy —
/// lowers to: the default `index` mask constant, the value broadcast, the destination
/// `sentient.logical_port`, and the `sentient.splat`.
///
/// ⛔ TRAP: ONLY THE MIDDLE PAD ARM IS DISTINCT. `isFirstElemSplat` and the register-init
/// fall-through both emit `SentientSplatPad::none` (`Splat.cpp:160-181`); only
/// `isPadLeftFor8FirstElemSplat` differs, and it needs [`vectorchain::Op::Shuffle`]'s `pad` segment.
/// ⛔ TRAP: an `op` that is neither a shuffle nor an `arith.constant`-fed copy emits ONLY the mask
/// constant and still succeeds — the reference's two arms are `if`s it may match neither of.
/// ⛔ TRAP: an overridden mask does NOT retract the default constant, which the reference has already
/// inserted; the dead `scalar_constant` is emitted here too.
/// ⛔ DIVERGENCE (3): the reference leaves `mlir::Value result` NULL when `from.op_` is none of its
/// four, `cast`s a scalar `arith.constant`'s attribute to `SplatElementsAttr`, and calls
/// `new_map_values.front()` on a possibly-empty vector. All three answer `None` here.
#[must_use]
pub fn create_splat_operation<A: Arch>(
    op: &DfirOp,
    from: &VectorOperand,
    to: &VectorOperand,
    comp: ComputeComp,
    scope: &[DfirOp],
    values: &mut Values,
) -> Option<Vec<sen::Op>> {
    let mut mask_value = values.mint();
    let mut emitted = vec![index_zero(mask_value)];

    // `if (auto const_op = llvm::dyn_cast<mlir::arith::ConstantOp>(from.op_))`
    if let Some(DfirOp::Arith(arith::Op::DenseConstant { splat, ty, .. })) = op_at(&from.op, scope) {
        // `if (mlir::isa<IntegerAttr>(splat_value))` — a float splat is the reference's
        // `emitError("Unable to create the splat operation")` and its `failure()`.
        if ty.elem.is_float() {
            return None;
        }
        let input = values.mint();
        emitted.push(sen::Op::Sentient(sentient::Op::ScalarConstant {
            value: *splat,
            result: input,
            reg_locale: sentient::RegType::Imm,
            // `rewriter.getI64Type()`, not the vector's element type.
            ty: ScalarTy::Int(64),
            is_symbol: false,
        }));
        let output = values.mint();
        emitted.push(logical_port(to.name()?, output));
        emitted.push(sen::Op::Sentient(sentient::Op::Splat {
            input,
            output,
            mask: mask_value,
            pad: sentient::SplatPad::None,
            precision: input_precision_from_operand(from)?,
            program_header: false,
            unroll_factor: sentient::UnrollFactor::X1,
            unroll_incr_result: false,
            // `dataflow::getDbgNameAttr(op)` — this island's shuffle carries no `dbgName`.
            dbg_name: None,
        }));
        return Some(emitted);
    }

    // `} else if (auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op))`
    let DfirOp::VectorChain(vectorchain::Op::Shuffle {
        variable,
        pad,
        mask: lane_mask,
        indices,
        repetition,
        ..
    }) = op
    else {
        return Some(emitted);
    };

    // `if (mask_operand) { comp != PT ? getMaskValueForNonPT(..) : std::nullopt }`
    if comp != ComputeComp::Pt
        && let Some(lane_mask) = lane_mask
        && let Some(position) = defining_position(lane_mask.val(), scope)
        && let Some(DfirOp::VectorChain(defining)) = op_at(&position, scope)
        && let Some(non_pt) = get_mask_value_for_non_pt::<A>(defining, values)
    {
        emitted.push(non_pt.op);
        mask_value = non_pt.value;
    }

    // `mlir::Value result;` and the three arms that may write it.
    let result = match op_at(&from.op, scope)? {
        DfirOp::VectorChain(const_bit @ vectorchain::Op::ConstantBitstream { .. }) => {
            let constant = create_sentient_constants(const_bit, op_as_vectorchain(op)?, values)?;
            emitted.push(constant.op);
            constant.value
        }
        // `uniform.query_map` — rebuild the mapping with each constant bitstream lowered, then
        // re-query it with the original key.
        DfirOp::Uniform(uniform::Op::QueryMap { map, key, .. }) => {
            let position = defining_position(*map, scope)?;
            let Some(DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) =
                op_at(&position, scope)
            else {
                return None;
            };
            let mut new_pairs = Vec::with_capacity(pairs.len());
            for (mapped_key, mapped_value) in pairs {
                if let Some(position) = defining_position(*mapped_value, scope)
                    && let Some(DfirOp::VectorChain(
                        const_bit @ vectorchain::Op::ConstantBitstream { .. },
                    )) = op_at(&position, scope)
                {
                    let constant =
                        create_sentient_constants(const_bit, op_as_vectorchain(op)?, values)?;
                    emitted.push(constant.op);
                    new_pairs.push((*mapped_key, constant.value));
                }
            }
            // `new_map_values.front().getType()` — the reference reads a type off the first entry
            // without checking there is one.
            if new_pairs.is_empty() {
                return None;
            }
            let new_map = values.mint();
            emitted.push(sen::Op::Uniform(uniform::Op::DefImmutableMapping {
                result: new_map,
                pairs: new_pairs,
            }));
            let queried = values.mint();
            emitted.push(sen::Op::Uniform(uniform::Op::QueryMap {
                result: queried,
                map: new_map,
                key: *key,
            }));
            queried
        }
        // `isa<dataflow::ReceiveOp, agen::VectorLoadOp>(from.op_)` — the value is already in a port.
        DfirOp::Dataflow(dataflow::Op::Receive { .. })
        | DfirOp::Agen(agen::Op::VectorLoad { .. }) => {
            let source = values.mint();
            emitted.push(logical_port(from.name()?, source));
            source
        }
        _ => return None,
    };

    let output = values.mint();
    emitted.push(logical_port(to.name()?, output));

    let pad_mode = if is_first_elem_splat(indices, *repetition, variable.len(), pad.len()) {
        sentient::SplatPad::None
    } else if is_pad_left_for_8_first_elem_splat(indices, *repetition, variable.len(), pad.len()) {
        sentient::SplatPad::Left
    } else {
        // "For Register Init" — the same op as the first arm.
        sentient::SplatPad::None
    };
    emitted.push(sen::Op::Sentient(sentient::Op::Splat {
        input: result,
        output,
        mask: mask_value,
        pad: pad_mode,
        precision: input_precision_from_operand(from)?,
        program_header: false,
        unroll_factor: sentient::UnrollFactor::X1,
        unroll_incr_result: false,
        dbg_name: None,
    }));
    Some(emitted)
}

/// The `vectorchain` op behind a [`DfirOp`], for handing this unit's own shuffle to entry 235.
fn op_as_vectorchain(op: &DfirOp) -> Option<&vectorchain::Op> {
    match op {
        DfirOp::VectorChain(op) => Some(op),
        _ => None,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::vc_vector_operands::{OpId, OperandValue, VectorOperandType};
    use super::{
        ComputeComp, DfirOp, SentientConstant, VectorOperand, create_sentient_constants,
        create_splat_operation,
    };
    use crate::arch::Target;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::arith;
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
            pad: Vec::new(),
            mask: None,
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
    /// ⭐⭐ IBM'S OWN PAD-LEFT CASE — `splat_const_bit.mlir:82-91` in, `:14-25` out. `pad(%c0)` with
    /// `indices = [0, -1 × 7]` puts the bitstream's first element in every eighth lane, which is the
    /// one arm of the three that does not emit `none`.
    #[test]
    fn the_vendor_pad_left_shuffle_emits_a_left_padded_splat_over_its_own_constant() {
        let f16 = |len| Vector {
            len,
            elem: ElemType::F16,
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            }),
            DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                result: Val(1),
                value: vec![0xff],
                ty: f16(1),
                is_symbol: false,
            }),
            DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(2),
                input: Val(1),
                variable: Vec::new(),
                pad: vec![Val(0)],
                mask: None,
                indices: vec![0, -1, -1, -1, -1, -1, -1, -1],
                repetition: 8,
                input_ty: f16(1),
                ty: f16(64),
            }),
        ];
        let operand = |kind, op: &[u32], value| VectorOperand {
            kind,
            op: OpId::at(op),
            values: vec![value],
            orig_precision: Some(sentient::Precision::Fp16),
            on_the_fly_conv_precision: Some(sentient::Precision::Fp16),
        };

        let mut values = Values::default();
        let emitted = create_splat_operation::<Target>(
            &scope[2],
            &operand(
                VectorOperandType::ConstantBitstream,
                &[1],
                OperandValue::Literal(0xff),
            ),
            &operand(
                VectorOperandType::Link,
                &[0],
                OperandValue::Port(sentient::Port::Pe),
            ),
            ComputeComp::Sfp,
            &scope,
            &mut values,
        );

        assert_eq!(
            emitted,
            Some(vec![
                // `%3 = sentient.scalar_constant {value = 0 : si64} : index`
                sen::Op::Sentient(sentient::Op::ScalarConstant {
                    value: 0,
                    result: Val(0),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
                // `%5 = sentient.scalar_constant {value = 0xff : si64} : f16`
                sen::Op::Sentient(sentient::Op::ScalarConstant {
                    value: 0xff,
                    result: Val(1),
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Elem(ElemType::F16),
                    is_symbol: false,
                }),
                // `%6 = sentient.logical_port {portName = #sentient<compute_port pe>} : index`
                sen::Op::Sentient(sentient::Op::LogicalPort {
                    port_name: sentient::Port::Pe,
                    result: Val(2),
                }),
                // `sentient.splat input(%5 : f16) output(%6) mask(%4)
                //  {pad = #sentient<splat_pad left>, precision = #sentient<precision fp16>, ..}`
                sen::Op::Sentient(sentient::Op::Splat {
                    input: Val(1),
                    output: Val(2),
                    mask: Val(0),
                    pad: sentient::SplatPad::Left,
                    precision: sentient::Precision::Fp16,
                    program_header: false,
                    unroll_factor: sentient::UnrollFactor::X1,
                    unroll_incr_result: false,
                    dbg_name: None,
                }),
            ])
        );
    }
}
