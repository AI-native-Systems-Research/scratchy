// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔⛔ THE FUSED MoE KERNEL AT THE WHOLE-FUNCTION DOOR — RUNG ZERO OF THE RECOGNISER LADDER.
//!
//! # WHAT THIS DRIVES
//!
//! `kernels/moe.py` — gemma-4's whole sparse block (router + experts + combine) as ONE
//! Triton program — through the full ladder and into `emit_whole`
//! (`emit::whole_function::lower_function`): the door the recognisers live behind. The
//! kernel is `include_str!`'d from the spyre target — the file the splice will splice,
//! not a test-local copy — and the geometry is the ladder's own golden geometry
//! (`triton-numeric`'s `moe.rs`, `triton-frontend`'s `common::moe`): one token, 64
//! experts, top-8, `H=128`, `I=64`.
//!
//! # WHY A DOOR TEST AND NOT A FROZEN GOLDEN
//!
//! The door is where the fused kernel's op CHAINS get recognised as the per-piece
//! programs the crate already names (`RouteTopK`, `ExpertGatedAct`, `ExpertMatmul`,
//! `ExpertCombine`, …). A frozen dump cannot round-trip on main (the printer drops
//! `linalg.generic` bodies), and a printer arm would be a second copy of the format.
//! Driving the kernel itself means every rung of the recogniser ladder is measured
//! against the live artifact, not a transcription of one.
//!
//! # RUNG ZERO, AND THE LADDER LAW
//!
//! Today the door recognises NOTHING of the fused MoE: an op with no 1:1 `Program` and
//! no chain proof is refused BY NAME, and the whole-function door refuses the WHOLE
//! function rather than emitting part of it (a bundle short a program computes
//! something else). This test pins the FIRST refusal — the op the walk names when it
//! stops. Each recogniser that lands (W8A8 slab gemm → router → gelu → combine) moves
//! that first refusal FORWARD through program order, and this test's assertion moves
//! with it, ON PURPOSE: the pinned refusal is the recogniser campaign's work queue
//! made executable. When the last recogniser lands, the same drive must BAKE, and the
//! refusal-pinning arms are replaced by the op-list assertion — the matmul_fp8
//! contract's positive control is the shape of that final state.
//!
//! The node's `program` label is INERT under this door — `lower_function` derives every
//! op's kind from its `OpKind` and never reads `node.program` — so it is minted with
//! `Program::ExpertMatmul`, the variant the W8A8 recogniser (the ladder's first rung)
//! emits its legs under. The splice fold will name the fused kind when it lands; the
//! label changes with it, one word.

use ktir_superdsc::ktir_node::Program;
use std::collections::HashMap;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = include_str!("../../../targets/spyre/kernels/moe.py");

/// The golden geometry: one token, 64 experts, top-8 (`M=1` is the door's own law — the
/// fused kernel is a decode-step kernel, and `M>1` is refused by the recognisers).
const M: usize = 1;
const H: usize = 128;
const I: usize = 64;
const E: usize = 64;
const K: usize = 8;

/// The kernel's 15 pointers, in signature order — `triton-numeric`'s `moe.rs` table.
const PTRS: &[(&str, &str)] = &[
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_wd", "*fp8e4nv"),
    ("desc_sd", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_ids", "*i32"),
    ("desc_rs", "*fp16"),
    ("desc_act", "*fp16"),
];

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in PTRS {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let h = H as f64;
    let ces: [(&str, Val); 10] = [
        ("M", Val::Int(M as i128)),
        ("H", Val::Int(H as i128)),
        ("I", Val::Int(I as i128)),
        ("E", Val::Int(E as i128)),
        ("K", Val::Int(K as i128)),
        ("EPS", Val::Float(1e-6)),
        ("INV_H", Val::Float(1.0 / h)),
        ("TEMP", Val::Float(h.powf(-0.5))),
        ("INV_QMAX", Val::Float(1.0 / 448.0)),
        ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
    ];
    let mut constexprs = HashMap::new();
    for (k, v) in ces {
        signature.insert(k.to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
        constexprs.insert(k.to_string(), v);
    }
    KernelSpec {
        kernel: "moe_fwd".to_string(),
        signature,
        constexprs,
        file: "moe.py".to_string(),
    }
}

/// The whole ladder, value-level: `.py` → TTIR → KTIR → the node the door takes. `Err`
/// carries the first stage's refusal text, so a ladder-stage break and a door refusal
/// are distinguishable in the assertion message.
fn drive() -> Result<Vec<String>, String> {
    let mut tt = codegen::compile(SRC, &spec(), Target::spyre())
        .map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&m, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))?;
    let ops = triton_ktir_superdsc::emit_whole(&node, None)
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// RUNG ZERO: the fused MoE kernel is REFUSED at the door, naming the op the walk
/// stopped on — pinned so every recogniser that lands must move it FORWARD, and a
/// recogniser that leaves it where it was bought nothing.
#[test]
fn the_fused_moe_kernel_is_refused_at_the_door_naming_the_first_unrecognised_op() {
    let err = match drive() {
        Ok(names) => panic!(
            "BAKED ({} ops) — the fused MoE kernel lowered through the whole-function door \
             with NO MoE recogniser landed; something is recognising chains it should not",
            names.len()
        ),
        Err(msg) => msg,
    };
    // The refusal must come from the door itself (a ladder-stage break is a different
    // defect and must not be pinned here as the door's answer).
    assert!(
        err.starts_with("door:"),
        "the refusal came from a ladder stage, not the door: {err:?}"
    );
    // ⚠️ PINNED OBSERVATION (rung zero, 2026-10-18, moe-door @ 6244a069f): the door's
    // FIRST refusal is the SILU-CHAIN pre-read refusing the kernel's GELU-TANH — moe.py
    // spells gelu through the sigmoid identity (`g / (1 + exp(-1.596·(g + 0.0447·g³)))`,
    // the cubic in an f32 island), which shares the silu longhand's five op kinds with a
    // DIFFERENT numerator (the divf divides `g`, the exp's chain head is the cubic `v`),
    // and the chains reader refuses the whole function rather than misread one as the
    // other. That refusal is CORRECT BY DESIGN — gelu is not silu, and a near-miss silu
    // is exactly the misrecognition the reader's fail-closed discriminator exists for —
    // and it is the recogniser campaign's queue head made executable: the pre-reads gate
    // the per-op walk, so NO recogniser is reachable while this stands. The gelu step of
    // the landing order (or a silu-reader refinement that distinguishes the two
    // longhands) is what moves it. The needles are SUBSTRINGs so wording survives but
    // meaning does not.
    assert!(
        err.contains("is not the value the `arith.negf` consumed")
            && err.contains("refused rather than lowered as `OpFunc::Silu`"),
        "rung zero: the door's first refusal moved or changed shape — update this pin \
         DELIBERATELY (it is the recogniser campaign's work queue), do not delete it: {err:?}"
    );
}
