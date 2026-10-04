// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⛔⛔⛔ THE WHOLE-FUNCTION DOOR DROPS A `linalg.matmul`'s THIRD OPERAND, UNREAD AND UNDIAGNOSED —
//! AND FLASH ATTENTION'S OUTPUT ACCUMULATOR IS THAT OPERAND.
//!
//! # WHAT THIS MEASURES
//!
//! `linalg.matmul(A, B, C)` computes `C + A·B`: the third operand is the DPS init, and for a
//! zero-init it carries no data. `emit::whole_function::lower_function` encodes that reading as a
//! CONSTANT:
//!
//! ```text
//!   whole_function.rs:1376   let n_in = match program { ... Lowering::Node(Program::Matmul) => 2, ... }
//!   whole_function.rs:1410   (None, None) => op.operands.iter().copied().take(n_in).collect(),
//! ```
//!
//! and operand 2 is then read by NOTHING: `grep -n 'operands.get(2)\|operands\[2\]'` over
//! `whole_function.rs` is empty, and `matmul_b_orientation` reads only the `indexing_maps`
//! attribute. So a matmul whose `outs` is a LIVE VALUE lowers to the same descriptor as one whose
//! `outs` is zero — the running sum is discarded with no refusal.
//!
//! ⭐ THAT IS EXACTLY FLASH ATTENTION'S `P·V` LEG. From `KTIR_DUMP=1` on
//! `attention_flash_causal` (`KTIR_WHOLE=1 bake_py`), inside the KV loop:
//!
//! ```text
//!   %682 = ArithMulf(%650, %681)              acc * alpha[:, None]      [64, 128]
//!   %686 = LinalgMatmul(%673, %684, %682)     tl.dot(p, v, acc)         [64, 128]
//!   ScfYield(%686, %688, %665, %699, %710)    acc, l_i, m_i, k_off, v_off
//! ```
//!
//! `%682` is the rescaled running output — the flash recurrence's whole state. Dropping it makes the
//! emitted program compute `p·v` for ONE trip and discard every earlier trip's contribution.
//!
//! # WHY NO EXISTING GUARD CATCHES IT
//!
//! `triton-ktir`'s `dot_to_linalg` DOES guard the accumulator — "matmul accumulator is not
//! zero-initialized (a bias/fused init would be dropped)", `dot_to_linalg.rs:391` — but only from
//! `verify_canonical_matmul_kernel`, which `run()` calls under `if single_dot_kernel`
//! (`dot_to_linalg.rs:146`), i.e. only when the module holds EXACTLY ONE `tt.dot`. Attention holds
//! two per `_attn_fwd_inner` call (the score matmul and `P·V`), so `dots.len() != 1`, `canonical`
//! is `false`, and those guards never run. Nothing else in either crate reads a matmul's operand 2.
//!
//! ⭐ AND THE IR HAS A CHANNEL FOR THE FACT. `ktir_core::attrkey::AttrKey` carries `NIns` and
//! `OutsVar` — a producer-stated input arity and outs variable, which `triton-ktir`'s
//! `to_ktir_emit.rs:416-418` already translates from source text. `grep -rn 'NIns\|OutsVar'` over
//! `ktir-superdsc/src` is EMPTY: the door hardcodes 2 instead of reading what the producer states.
//! So this is an UNREAD ATTRIBUTE, not a missing IR capability.
//!
//! # ⛔⛔⛔ AND IT IS **NOT** LATENT — A CONFIGURATION RECORDED AS WORKING IS ALREADY WRONG
//!
//! The first draft of this module said the drop was latent behind the `regions` refusal, so "no
//! baked descriptor is wrong TODAY". **THAT WAS FALSE, and this measurement is what falsified it.**
//! Attention indeed cannot reach this door (`regions` refuses its in-loop `desc_k` windows first) —
//! but attention is not the only kernel with a real accumulation.
//!
//! `decoder_block.py:274-275` splits the score matmul over the head dimension in two pieces, in
//! STRAIGHT-LINE code with no loop at all:
//!
//! ```python
//!   qk = tl.dot(q1r, k1r.T, out_dtype=tl.float16)
//!   qk = tl.dot(q2r, k2r.T, qk)          # accumulates onto the first half
//! ```
//!
//! MEASURED on `decoder_layer_one_flat` (`KTIR_WHOLE=1 bake_py`, which BAKES: `ops=52 files=53`,
//! and is recorded passing `dxp_standalone --bundle -b sentient` with RC=0):
//!
//! ```text
//!   KTIR      %447 = LinalgMatmul(%434, %444, %403)    q1r·k1rᵀ, outs = a zero splat
//!             %450 = LinalgMatmul(%437, %448, %447)    q2r·k2rᵀ, outs = %447   ⛔
//!             %454 = ArithMulf(%450, %453)             qk * QK_SCALE
//!   %447's USES: exactly one — operand 2 of %450. Nothing else.
//!
//!   EMITTED   23_matmul_o32     writes t32   (q1r·k1rᵀ)
//!             25_matmul_o34     writes t34   (q2r·k2rᵀ)
//!             26_scalarmul_o35  reads  t34
//!   and NO op among the 52 combines t32 with t34.
//! ```
//!
//! So `t32` is written and never read: the baked bundle computes its attention scores as
//! `q2r·k2rᵀ` ALONE, discarding half the head dimension's contribution — while baking clean and
//! compiling through dxp. `decoder_two_layers_flat` (`ops=104`) carries the same defect twice.
//!
//! ⚖️ WHAT IS **NOT** ESTABLISHED. No card number was taken for the decoder against its own
//! `test/numeric/decoder_layer_one_flat/ref_out.bin` (a torch reference exists, generated
//! 2026-09-18; `meta.json` records no run). The argument above is an ARTIFACT-level proof — a
//! written-and-never-read buffer plus a measured operand drop — not a scored divergence. A pod run
//! against that reference is the confirming measurement and has not been done.
//!
//! ⛔ NOTHING HERE LOOSENS A GUARD. The probe only ASSERTS what the door does; it changes no
//! vendored code. The fix belongs upstream — see the report accompanying this file.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};

/// How the `P·V` matmul's third operand is spelled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Acc {
    /// THE CONTROL — a fresh `tensor.empty`, which is what the door's `n_in = 2` assumes. Dropping
    /// this really does lose nothing.
    Zero,
    /// ⛔ ATTENTION'S OWN SHAPE — a LOADED, data-carrying value standing in for `acc * alpha`. This
    /// is the flash recurrence's state, and `%682` above is this operand.
    Live,
}

/// One KTIR program shaped like flash attention's `P·V` leg: `out = matmul(p, v, acc)`.
///
/// Built from the op kinds `regions` and `lower_function` actually read — `construct_memory_view`,
/// `construct_access_tile`, `arith.constant`, `ktdp.load`, `linalg.matmul`, `ktdp.store` — so the
/// door sees a real program and not a stub. The ONLY difference between the two cases is which value
/// sits in operand 2 of the matmul; every view, tile, load and extent is identical.
///
/// `[64, 64] · [64, 128] -> [64, 128]` is attention's own geometry at `BLOCK_M = BLOCK_N = 64`,
/// `HEAD_DIM = 128`.
fn pv_program(acc: Acc) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // %0 = P address, %1 = V address, %2 = output address, %3 = the index constant 0.
    let (p_p, p_v, p_out, zero) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    let (v_p, t_p, val_p) = (Ssa(4), Ssa(5), Ssa(6));
    let (v_v, t_v, val_v) = (Ssa(7), Ssa(8), Ssa(9));
    let (v_o, t_o) = (Ssa(10), Ssa(11));
    let (acc_v, prod) = (Ssa(12), Ssa(13));

    let mut ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // P: [64, 64]
        shape(
            Operation::new(a, Some(v_p), OpKind::KtdpConstructMemoryView, &[p_p]),
            64,
            64,
        ),
        shape(
            Operation::new(a, Some(t_p), OpKind::KtdpConstructAccessTile, &[v_p, zero, zero]),
            64,
            64,
        ),
        shape(Operation::new(a, Some(val_p), OpKind::KtdpLoad, &[t_p]), 64, 64),
        // V: [64, 128]
        shape(
            Operation::new(a, Some(v_v), OpKind::KtdpConstructMemoryView, &[p_v]),
            64,
            128,
        ),
        shape(
            Operation::new(a, Some(t_v), OpKind::KtdpConstructAccessTile, &[v_v, zero, zero]),
            64,
            128,
        ),
        shape(Operation::new(a, Some(val_v), OpKind::KtdpLoad, &[t_v]), 64, 128),
        // OUT: [64, 128]
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            64,
            128,
        ),
        shape(
            Operation::new(a, Some(t_o), OpKind::KtdpConstructAccessTile, &[v_o, zero, zero]),
            64,
            128,
        ),
    ];

    // THE ONE VARIABLE. `Zero` mints the fresh `tensor.empty` the door assumes; `Live` reuses the
    // LOADED V tile, a value that demonstrably carries data (it is an operand of the matmul itself).
    let acc_operand = match acc {
        Acc::Zero => {
            ops.push(shape(
                Operation::new(a, Some(acc_v), OpKind::TensorEmpty, &[]),
                64,
                128,
            ));
            acc_v
        }
        Acc::Live => val_v,
    };

    ops.push(shape(
        Operation::new(a, Some(prod), OpKind::LinalgMatmul, &[val_p, val_v, acc_operand]),
        64,
        128,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[prod, t_o]));

    KtirNode {
        func: IRFunction {
            name: "pv_s0",
            arguments: a.args(vec![
                (p_p, IrType::Index),
                (p_v, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Matmul,
        bindings: vec![BufferId::new(101), BufferId::new(102), BufferId::new(103)],
        mask: None,
        node_out_tid: None,
    }
}

/// Every emitted descriptor, as JSON — the artifact `dxp_standalone` reads, so two programs with
/// equal JSON are the same program on the card.
fn descriptors_of(k: &KtirNode) -> Result<String, String> {
    let mut sid = 0i64;
    let ops = lower_function(k, None, &mut sid).map_err(|e| e.message)?;
    Ok(ops
        .iter()
        .map(|o| {
            let body = o
                .op
                .as_ref()
                .map(|d| serde_json::to_string(d).expect("a descriptor serializes"))
                .unwrap_or_else(|| "<no descriptor>".to_string());
            format!("{}\n{}", o.op_name, body)
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn descriptors(acc: Acc) -> Result<String, String> {
    descriptors_of(&pv_program(acc))
}

/// ⛔⛔⛔ THE MEASUREMENT: A LIVE ACCUMULATOR AND A ZERO ONE EMIT THE **SAME DESCRIPTOR**.
///
/// Two programs that compute different things — `v + p·v` and `0 + p·v` — produce byte-identical
/// SuperDSC. That is the drop, stated as an artifact rather than as a reading of `take(2)`.
///
/// ⭐ AND IT IS NOT A REFUSAL. If the door refused the live accumulator this test fails with the
/// refusal text, which would be the GOOD outcome (fail-closed) and is what the upstream fix should
/// make happen.
#[test]
fn a_live_accumulator_and_a_zero_one_emit_the_same_descriptor() {
    let zero = match descriptors(Acc::Zero) {
        Ok(j) => j,
        Err(e) => panic!(
            "the CONTROL (zero-init `outs`) must lower — it is the shape the door is built for. \
             If this refuses, the probe is measuring something else: {e}"
        ),
    };
    let live = match descriptors(Acc::Live) {
        // ⭐ THE FAIL-CLOSED OUTCOME. A refusal here means the hole is shut; re-read this module and
        // retire it rather than weakening the assertion.
        Err(e) => panic!(
            "GOOD NEWS, AND THIS TEST IS NOW WRONG: the door REFUSED a live `outs` operand instead \
             of dropping it. That is the fix this probe argues for. Refusal: {e}"
        ),
        Ok(j) => j,
    };

    println!("ZERO-INIT `outs`  -> {} bytes of descriptor JSON", zero.len());
    println!("LIVE      `outs`  -> {} bytes of descriptor JSON", live.len());

    assert_eq!(
        live, zero,
        "the two differ, so operand 2 DOES reach the descriptor — re-read the module docs: this \
         probe's whole claim is that it does not"
    );
}

/// ⭐ THE POSITIVE CONTROL, without which the assertion above could pass vacuously.
///
/// If `descriptors()` returned something insensitive to the program — a constant, or an empty list —
/// the equality above would hold for a reason that has nothing to do with the accumulator. So
/// perturb an operand the door DEMONSTRABLY reads (V's column extent, which sets the contraction's
/// `n`) and require the outcome to CHANGE. One variable, and it is one the door is known to consult.
#[test]
fn the_comparison_is_sensitive_to_an_operand_the_door_does_read() {
    let base = descriptors(Acc::Zero).expect("the control lowers");
    assert!(
        !base.is_empty() && base.contains('{'),
        "the probe produced no descriptor JSON at all ({} bytes), so the equality test above \
         proves nothing",
        base.len()
    );

    // The same program with every `[64, 128]` extent narrowed to `[64, 64]`. `n` is an extent the
    // matmul door reads directly, so this MUST move the bytes (or refuse).
    let a: &'static Arena = Arena::global();
    let mut k = pv_program(Acc::Zero);
    let ops: Vec<Operation<'static>> = k
        .func
        .operations
        .iter()
        .map(|o| {
            let wide = o.attributes.iter().any(|(key, v)| {
                matches!((key, v), (AttrKey::Shape, Attr::IntList(s)) if *s == [64, 128])
            });
            if wide {
                (*o)
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![64, 64])))
            } else {
                *o
            }
        })
        .collect();
    k.func.operations = a.ops(ops);

    match descriptors_of(&k) {
        Ok(j) => assert_ne!(
            j, base,
            "narrowing the contraction's `n` from 128 to 64 did not change the descriptor, so this \
             probe cannot tell two programs apart and the equality assertion above is vacuous"
        ),
        // A refusal is also a CHANGE in behaviour, which is all this control needs to establish.
        Err(e) => println!("narrowed program refused (still a distinguishable outcome): {e}"),
    }
}
