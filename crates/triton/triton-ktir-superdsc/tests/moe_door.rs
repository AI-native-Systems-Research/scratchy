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
//! # RUNG TWO, AND THE LADDER LAW
//!
//! The door recognises the kernel's EIGHT gelu-tanh longhands (the tanh form through
//! the sigmoid identity, claimed by `program_gelu_mul_chains` before the silu reader
//! it fronts for) and its gate/up W8A8 projections (`program_w8a8_gemm_chains`: the
//! per-token quantize, the per-pair gathered fp8 slab gemms, the two-multiply
//! dequant, claimed only when proven and only at `M=1`), and NOTHING ELSE of the
//! fused MoE: an op with no 1:1 `Program` and no chain proof is refused BY NAME, and
//! the whole-function door refuses the WHOLE function rather than emitting part of
//! it (a bundle short a program computes something else). This test pins the FIRST
//! refusal — the stage the door stops at. Each recogniser that lands (gelu ✅ → W8A8
//! gate/up ✅ → down slab gemm ✅ → router → combine) moves that first refusal FORWARD
//! through the pre-reads and program order, and this test's assertion moves with it,
//! ON PURPOSE: the pinned refusal is the recogniser campaign's work queue made
//! executable. When the last recogniser lands, the same drive must BAKE, and the
//! refusal-pinning arms are replaced by the op-list assertion — the W8A8 half's
//! contract (`w8a8_door.rs`) is the shape of that final state.
//!
//! The node's `program` label is INERT under this door — `lower_function` derives every
//! op's kind from its `OpKind` and never reads `node.program` — so it is minted with
//! `Program::ExpertMatmul`, the variant the W8A8 recogniser emits its legs under. The
//! splice fold will name the fused kind when it lands; the label changes with it, one
//! word.

use ktir_core::ir::Ssa;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{
    gather_facts_at_tile, gathers_of, gathers_of_excluding,
};
use ktir_superdsc::ktir_node::{KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID};
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
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr"),
        );
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
fn drive_node() -> Result<KtirNode, String> {
    let mut tt =
        codegen::compile(SRC, &spec(), Target::spyre()).map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    triton_ktir_superdsc::node_for(&m, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))
}

/// The node through the door: the op names a full recognition would emit.
fn drive() -> Result<Vec<String>, String> {
    let node = drive_node()?;
    let ops = triton_ktir_superdsc::emit_whole(&node, None)
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// A REAL layout for the full kernel: all 15 parameters at their true byte
/// sizes (an fp16-sized reservation is fine — only the byte extent matters to
/// the footprint checks; the ids' 4-byte i32s and the banks' 1-byte fp8s are
/// sized directly in bytes), the three E4M3 clamp consts, the router norm's
/// epsilon registry slot, and nothing else — the walk's own mints reserve their
/// intermediates into it as it descends. This is the drive the router rung
/// measures with: at `layout: None` the walk stops minting the CLAIMED router
/// norm chain's output intermediate (the fused RmsNorm emission's own mint),
/// which measures the unit-test arm's limit, not the door's.
fn full_layout(node: &KtirNode) -> BundleLayout {
    // True bytes, in signature order: x_r [M,H]·2, rg [H]·2, rw [E,H]·2,
    // pes [E]·2, xe [M,H]·2, wg [E,I,H]·1, sg [E,I]·2, wu [E,I,H]·1,
    // su [E,I]·2, wd [E,H,I]·1, sd [E,H]·2, o [M,H]·2, ids [M,K]·4,
    // rs [M,K]·2, act [M,K·I]·2.
    let bytes: [u64; 15] = [
        (M * H * 2) as u64,
        (H * 2) as u64,
        (E * H * 2) as u64,
        (E * 2) as u64,
        (M * H * 2) as u64,
        (E * I * H) as u64,
        (E * I * 2) as u64,
        (E * I * H) as u64,
        (E * I * 2) as u64,
        (E * H * I) as u64,
        (E * H * 2) as u64,
        (M * H * 2) as u64,
        (M * K * 4) as u64,
        (M * K * 2) as u64,
        (M * K * I * 2) as u64,
    ];
    let mut l = BundleLayout::default();
    for (i, b) in node.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Distinct 2 MiB slots — the widest parameter (a 512 KiB fp8 bank)
        // fits with room to spare.
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: i as u64 * 0x20_0000,
                size: bytes[i],
            },
        );
    }
    for (i, tid) in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID]
        .into_iter()
        .enumerate()
    {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: 15 * 0x20_0000 + i as u64 * 0x1000,
                size: 128,
            },
        );
    }
    // The router norm's epsilon — the `scalarmul_scales` registry law the fused
    // RmsNorm emission resolves its `[1,1]` const by. The program states EPS as
    // its f16 IMAGE (`0.000001013279`, the legalization's rounding of `1e-6`),
    // which is why the registry holds the f32 constant and `scale_slot`'s f16
    // fallback is what reconciles them — the exact case its doc names.
    l.scalarmul_scales = vec![1e-6];
    let eps_tid = ktir_superdsc::reserved_tids::scalarmul_scale_tid(0);
    l.ids
        .borrow_mut()
        .insert(act_name(eps_tid), PlaceId::Act(eps_tid));
    l.placements.insert(
        eps_tid,
        TensorPlacement {
            tid: eps_tid,
            role: SegRole::Activation,
            segment: SegRole::Activation.segment(),
            bank: 0,
            offset: 15 * 0x20_0000 + 3 * 0x1000,
            size: 2,
        },
    );
    l
}

/// The full kernel through the door WITH a layout — the router rung's own
/// measurement drive. `Err` carries the first stage's refusal text.
fn drive_laid_out() -> Result<Vec<String>, String> {
    let node = drive_node()?;
    let layout = full_layout(&node);
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// RUNG ONE: the fused MoE kernel is REFUSED at the door, naming the stage the
/// pre-reads stop at — pinned so every recogniser that lands must move it FORWARD, and
/// a recogniser that leaves it where it was bought nothing.
#[test]
fn the_fused_moe_kernel_is_refused_at_the_door_naming_the_first_unrecognised_op() {
    let err = match drive() {
        Ok(names) => panic!(
            "BAKED ({} ops) — the fused MoE kernel lowered through the whole-function door \
             with the W8A8/router/combine recognisers unlanded; something is recognising \
             chains it should not",
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
    // ⚠️ PINNED OBSERVATION (rung three, 2026-10-19, moe-door @ the DOWN recogniser):
    // `program_w8a8_down_chains` claims the kernel's DOWN projection — the last 16
    // tiles (2 banks × 8 pairs) — so EVERY pre-read now passes (gelu, silu, rmsnorm,
    // bcast, W8A8 gate/up, W8A8 down) and the walk descends into the body for the
    // FIRST time. The first refusal is no longer a pre-read's: the ROUTER NORM is
    // CLAIMED (`program_rmsnorm_chains` fires at the chain's TERMINAL gain multiply,
    // `xn = xr·r·rg`), and it is that claimed emission's OUTPUT-intermediate mint
    // that refuses — an `ArithMulf` whose result needs a `BundleLayout` the
    // unit-test arm (`layout: None`) cannot mint. The layout-carrying twin below is
    // the drive that descends past it (first stop: the router rung's own head,
    // `TensorGenerate`), and at this file's golden `I=64` the down door's own
    // staged-bank law (`in=64 pads to 128`, pinned in `down_door.rs`) is the
    // worker-manifest pad the splice fold owns.
    // The needles are SUBSTRINGs so wording survives but meaning does not.
    assert!(
        err.starts_with("door:")
            && err.contains("`ArithMulf`'s result is an intermediate")
            && err.contains("`layout: None` is the unit-test arm"),
        "rung three: the door's first refusal moved or changed shape — update this pin \
         DELIBERATELY (it is the recogniser campaign's work queue), do not delete it: {err:?}"
    );
}

/// RUNG THREE, THE LAYOUT-CARRYING HALF — the router rung's measurement. With
/// a REAL layout the claimed router norm's fused emission mints (the epsilon
/// registry above is its one demand) and the walk descends into the router's
/// body, past the norm and the f16 logits dot; this pin names the first op the
/// 1:1 walk cannot express, which is the router rung's own head.
#[test]
fn the_full_kernel_with_a_layout_refuses_at_the_first_unexpressible_op() {
    let err = match drive_laid_out() {
        Ok(names) => panic!(
            "BAKED ({} ops) — the full fused MoE kernel lowered through the door with a \
             layout; the router and combine rungs are both unlanded, so something is \
             recognising chains it should not",
            names.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:"),
        "the refusal came from a ladder stage, not the door: {err:?}"
    );
    assert!(
        err.contains("has no 1:1 `Program`"),
        "the walk refused, but not at an unexpressible op — measure again before pinning: \
         {err:?}"
    );
    // ⚠️ PINNED OBSERVATION (rung four's head, 2026-10-19, moe-door @ the DOWN
    // recogniser, router unlanded): with the epsilon in `scalarmul_scales` (and
    // its reserved scale-slot placed — `full_layout`'s registry block) the fused
    // RmsNorm emission FIRES on the router norm, the walk emits the router-weight
    // load `[64,128]` and the f16 logits dot 1:1 (`LinalgMatmul [1,64]`, op 107 —
    // the dot is EXPRESSIBLE unclaimed), and the first op the 1:1 walk cannot
    // express is `TensorGenerate` — op 108 of 1111, shape `[64]`:
    // `offs = tl.arange(0, E).to(tl.float16)`, the round-form top-K's
    // lane-offset vector (`moe.py:148`). The router section the rung must claim
    // runs ops 108–465: eight rounds of (max-reduce → lane select → idx-reduce →
    // one-hot accumulate into `vals`/`idsf`/`psel` → mask select), the ids store
    // (`ArithFptosi` + `KtdpStore`, op 444), the softmax (`·TEMP` → max → sub →
    // exp → sum → div → `·psel`, ops 445–463), and the rs store (op 465). The
    // experts (claimed: gelu/silu/W8A8 gate/up/down) begin at op 466; the COMBINE
    // rung's head is what the router rung's landing exposes next.
    assert!(
        err.contains("`TensorGenerate` has no 1:1 `Program`"),
        "rung four: the door's first layout-carrying refusal moved or changed shape — update \
         this pin DELIBERATELY (it is the router rung's work queue), do not delete it: {err:?}"
    );
}

/// ⛔ THE EXCLUSION CONTRACT — rung one's reader called DIRECTLY, pinning the seam the W8A8
/// pre-read will claim its gate/up tiles through before that recogniser exists:
///
/// * the UNCLAIMED read refuses the fused kernel's 48 tiles (rung one's door pin, stated at
///   the reader itself);
/// * claiming ONE tile moves the count, never the verdict — a claim cannot merge two banks
///   into one pair;
/// * the per-tile FACTS reader carries no set judgement and no stick law: every one of the
///   48 tiles reads clean, and every expert pair's tiles anchor at their own index WORD
///   (0..7), all inside `desc_ids`'s first `SenUint32` stick — the raw-anchor fact the
///   W8A8 pre-read verifies per tile and re-stages at emission (the m=1 body's minted,
///   replicated, stick-aligned index);
/// * claiming ONE BANK (one descriptor's 8 tiles) still refuses — five pairs remain, and
///   only the count moves with the claim; claiming ALL BANKS BUT ONE leaves ONE pair, the
///   discriminator goes silent, and the refusal moves to the law that was next in line
///   all along: the stick law, on the non-stick anchors of the pairs beyond the first.
///   That is the refusal the recogniser takes over by claiming its tiles — the exclusion
///   ends where the recogniser's own verification begins.
///
/// A claimed tile is CONSUMED, not forgiven. If this seam ever lets a mixed-pair set through
/// without a whole bank claimed, the W8A8 step's exclusion is a discriminator relaxation
/// wearing a claim's clothes.
#[test]
fn claimed_tiles_are_consumed_not_forgiven_by_the_gather_discriminator() {
    let node = drive_node().expect("the ladder drive is rung one's own green path");
    let unclaimed =
        gathers_of(&node).expect_err("48 tiles over 2 parameter pairs is the rung-one refusal");
    assert!(
        unclaimed.message.contains("48")
            && unclaimed
                .message
                .contains("different index/value parameter pairs"),
        "the unclaimed read changed shape: {}",
        unclaimed.message
    );

    // The census: 6 descriptors x 8 expert pairs. Claims are tile RESULT ssas — the
    // identity `Gather::tile` carries, never a position in this enumeration.
    let tiles: Vec<&_> = node
        .func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .collect();
    assert_eq!(
        tiles.len(),
        48,
        "the census the rung-one pin was measured on"
    );

    // The FACTS: every tile reads through the per-tile walk alone — no stick law, no set
    // judgement — the reading the W8A8 pre-read composes. Pin the raw anchors: every
    // expert pair's tiles start at their own index word, all inside the first stick.
    let facts: Vec<_> = tiles
        .iter()
        .map(|t| {
            gather_facts_at_tile(&node, t)
                .expect("every tile of the fused kernel reads clean through the per-tile walk")
        })
        .collect();
    let mut anchors: Vec<i64> = facts.iter().map(|g| g.first_entry).collect();
    anchors.sort_unstable();
    anchors.dedup();
    assert_eq!(
        anchors,
        (0..8).collect::<Vec<i64>>(),
        "every expert pair's tiles anchor at its own index word in desc_ids"
    );

    // ONE claim: the count moves (48 -> 47 in the refusal), the verdict does not.
    let first = facts[0];
    let one = gathers_of_excluding(&node, &[first.tile])
        .expect_err("one claimed tile cannot merge two banks into one pair");
    assert!(
        one.message.contains("47")
            && one
                .message
                .contains("different index/value parameter pairs"),
        "a single claim must move the count without softening the discriminator: {}",
        one.message
    );

    // ONE BANK: one descriptor's tiles — 8 of them, one per expert pair. The census is
    // 6 descriptors x 8 pairs (the rung-one message's "2 pairs" is pairs seen at the
    // refusal, not the bank count), so a bank is one descriptor, not a weights/scales
    // half, and claiming one leaves five pairs for the discriminator to judge.
    let bank: Vec<Ssa> = facts
        .iter()
        .filter(|g| g.value_tid == first.value_tid)
        .map(|g| g.tile)
        .collect();
    assert_eq!(bank.len(), 8, "one descriptor's tiles: one per expert pair");
    let one_bank = gathers_of_excluding(&node, &bank)
        .expect_err("one bank claimed leaves five banks over five pairs");
    assert!(
        one_bank.message.contains("40")
            && one_bank
                .message
                .contains("different index/value parameter pairs"),
        "one bank claimed must leave the discriminator judging the five remaining banks: {}",
        one_bank.message
    );

    // ALL BANKS BUT ONE: five descriptors claimed, ONE pair left — the discriminator is
    // silent, and the refusal moves to the stick law on the non-stick anchors (every
    // pair but the first anchors at words 1..7). This is the refusal the W8A8 pre-read
    // takes over: the recogniser verifies those raw anchors itself and re-stages them.
    let other_banks: Vec<Ssa> = facts
        .iter()
        .filter(|g| g.value_tid != first.value_tid)
        .map(|g| g.tile)
        .collect();
    assert_eq!(other_banks.len(), 40, "the other five descriptors' tiles");
    let rest = gathers_of_excluding(&node, &other_banks)
        .expect_err("one pair left must meet the stick law on its non-stick anchors");
    assert!(
        rest.message.contains("not a whole 32-entry"),
        "with the discriminator silent, the stick law must fire on the non-stick pairs: {}",
        rest.message
    );
}
