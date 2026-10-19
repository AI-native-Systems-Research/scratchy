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
use ktir_superdsc::reserved_tids::{
    FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID, router_identity_tid, router_sm_mask_tid,
    router_topk_iota_tid, router_topk_pad_tid, scalarmul_scale_tid,
};
use std::collections::HashMap;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = include_str!("../../../targets/spyre/kernels/moe.py");

/// The door's bake geometry: one token, 64 experts, top-8 (`M=1` is the door's own law — the
/// fused kernel is a decode-step kernel, and `M>1` is refused by the recognisers).
/// `H=128`, `I=128`: the ALIGNED arm — `fp8_k_pad(128) = 128`, the padded layout IS the
/// logical one, so the down emission's ten legs are complete (`down_door.rs`'s own bake
/// geometry). The ladder's numeric golden runs `I=64`, a HALF-STICK width the down emission
/// refuses at the staged-bank law (the worker-manifest bank no whole-function program
/// states — `down_door.rs`'s half-stick and 26b-width arms pin it, the splice fold owns
/// lifting it); the full-kernel BAKE therefore runs at the aligned width, like every door
/// suite arm that bakes.
const M: usize = 1;
const H: usize = 128;
const I: usize = 128;
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

fn spec_with(k: usize) -> KernelSpec {
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
        ("K", Val::Int(k as i128)),
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

/// The whole ladder, value-level, at a source and a top-k width: `.py` → TTIR → KTIR → the
/// node the door takes. `Err` carries the first stage's refusal text, so a ladder-stage
/// break and a door refusal are distinguishable in the assertion message.
fn drive_node_src(src: &str, k: usize) -> Result<KtirNode, String> {
    let mut tt = codegen::compile(src, &spec_with(k), Target::spyre())
        .map_err(|e| format!("codegen: {e}"))?;
    opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &[1]).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &[1]).map_err(|e| format!("to_ktir: {e}"))?;
    triton_ktir_superdsc::node_for(&m, Program::ExpertMatmul)
        .map_err(|e| format!("handoff: {}", e.message))
}

fn drive_node() -> Result<KtirNode, String> {
    drive_node_src(SRC, K)
}

/// The node through the door: the op names a full recognition would emit.
fn drive() -> Result<Vec<String>, String> {
    let node = drive_node()?;
    let ops = triton_ktir_superdsc::emit_whole(&node, None)
        .map_err(|e| format!("door: {}", e.message))?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// The softmax temperature — `H^-0.5` at the golden geometry, the CANONICAL value the
/// registry holds (an unregistered-temperature mutant moves the kernel's splat, never this
/// registry — `router_door`'s own law).
const TEMP: f64 = 0.08838834764831845;

/// A REAL layout for the full kernel: all 15 parameters at their true byte
/// sizes (an fp16-sized reservation is fine — only the byte extent matters to
/// the footprint checks; the ids' 4-byte i32s and the banks' 1-byte fp8s are
/// sized directly in bytes), the four router consts and the three E4M3 clamp
/// consts the emissions read by reserved tid, and BOTH registry slots
/// (`scalarmul_scales` holds the router norm's epsilon and the softmax's
/// temperature) — and nothing else: the walk's own mints reserve their
/// intermediates into it as it descends. This is the drive the combine rung
/// measures with: the census passes (the combine claims the route-weights
/// read), the walk runs the whole kernel, and every fire arm's mint lands
/// here.
fn full_layout_for(node: &KtirNode, k: usize) -> BundleLayout {
    // True bytes, in signature order: x_r [M,H]·2, rg [H]·2, rw [E,H]·2,
    // pes [E]·2, xe [M,H]·2, wg [E,I,H]·1, sg [E,I]·2, wu [E,I,H]·1,
    // su [E,I]·2, wd [E,H,I]·1, sd [E,H]·2, o [M,H]·2, ids [M,k]·4,
    // rs [M,k]·2, act [M,k·I]·2.
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
        (M * k * 4) as u64,
        (M * k * 2) as u64,
        (M * k * I * 2) as u64,
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
    // The reserved consts, past the fifteen parameters: the router's four (the sanitize
    // mask, the iota row, the pad row, the identity table — the router emission reads them
    // by reserved tid) and the three E4M3 clamp consts the quantize chains read.
    let base = 15u64 * 0x20_0000;
    for (i, (tid, size)) in [
        (router_sm_mask_tid(), 128u64),
        (router_topk_iota_tid(), 128),
        (router_topk_pad_tid(), 128),
        (router_identity_tid(), 8192),
        (FP8_POS448_TID, 128),
        (FP8_NEG448_TID, 128),
        (FP8_INV448_TID, 128),
    ]
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
                offset: base + i as u64 * 0x4000,
                size,
            },
        );
    }
    // The registry slots: the norm's epsilon and the softmax's temperature, one `[1,1]`
    // f16 each. The program states both as their f16 IMAGES (`0.000001013279`, the
    // legalization's rounding of `1e-6`; `0.088378906`, the rounding of `H^-0.5`), which
    // is why the registry holds the f32 constants and `scale_slot`'s f16 fallback is what
    // reconciles them — the exact case its doc names.
    l.scalarmul_scales = vec![1e-6, TEMP as f32];
    for i in 0..2usize {
        let tid = scalarmul_scale_tid(i);
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: base + 7 * 0x4000 + i as u64 * 0x1000,
                size: 2,
            },
        );
    }
    l
}

/// A source and top-k width through the whole ladder and the door WITH a layout — the
/// top-1 arm's and the mutants' drive. Returns the emitted ops themselves, so the pins can
/// read arg bindings, not just names.
fn drive_ops_src(src: &str, k: usize) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let node = drive_node_src(src, k)?;
    let layout = full_layout_for(&node, k);
    triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("door: {}", e.message))
}

/// A mutant source: `SRC` with each anchor replaced. Every anchor is asserted present, so
/// a moe.py edit that silently orphaned a mutant fails the mutant's own drive, not the
/// door's answer.
fn src_with(repls: &[(&str, &str)]) -> String {
    let mut s = SRC.to_string();
    for (from, to) in repls {
        assert!(s.contains(from), "a mutant anchor drifted: {from:?}");
        s = s.replace(from, to);
    }
    s
}

/// RUNG ONE: the fused MoE kernel is REFUSED at the door, naming the stage the
/// pre-reads stop at — pinned so every recogniser that lands must move it FORWARD, and
/// a recogniser that leaves it where it was bought nothing.
#[test]
fn the_fused_moe_kernel_is_refused_at_the_door_naming_the_first_unrecognised_op() {
    let err = match drive() {
        Ok(names) => panic!(
            "BAKED ({} ops) — the fused MoE kernel lowered through the whole-function door \
             with `layout: None`; the unit-test arm cannot place intermediates, so something \
             is minting without a layout",
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
    // ⚠️ PINNED OBSERVATION (rung six — the COMBINE, 2026-10-19, moe-door): the combine
    // rung claims the route-weights read, so the SCRATCH CENSUS passes and the walk RUNS —
    // the layout-less arm's first refusal is no longer a pre-read's at all. It is the
    // walk's FIRST intermediate mint: the router norm chain's output (the fused RmsNorm
    // emission's own mint, an `ArithMulf`'s result), which `layout: None` cannot place.
    // That measures the unit-test arm's limit, not the door's — the door's answer for the
    // full kernel is the LAYOUT-CARRYING BAKE the next test pins (the kernel BAKES now;
    // every recogniser in the ladder has landed).
    // The needles are SUBSTRINGs so wording survives but meaning does not.
    assert!(
        err.starts_with("door:")
            && err.contains("`ArithMulf`'s result is an intermediate")
            && err.contains("cannot place one"),
        "rung six: the door's first layout-less refusal moved or changed shape — update this \
         pin DELIBERATELY (it is the recogniser campaign's work queue), do not delete it: \
         {err:?}"
    );
}

/// ⚠️ RUNG SIX — THE BAKE (the layout-carrying arm's refusal pin, gone green-to-gone):
/// with the combine rung landed, the FULL fused MoE kernel lowers end-to-end through the
/// whole-function door with a real layout. This is the op-list contract in
/// `w8a8_door.rs`'s shape, pinning the COMBINE's own slice of the bake:
///
/// * **The `2k−1` legs, in order**: pair 0's product seeds the accumulator, every later
///   pair a product leg then an add leg, the final add last — the LEFT-DEEP SLOT-ORDER
///   sum the TTIR states (moe.py's zero seed FOLDED by `make_ttir` — measured), kept
///   operand-for-operand so the f16 op sequence is the reference semantics
///   torch-spyre is judged against.
/// * **THE PER-POSITION LAW**: each pair's product leg fires IMMEDIATELY after that
///   pair's down legs (`emdo_p{j}` — the kernel's own `gelu_j → down_j → combine_j`
///   interleaving), never batched at the first pair; a single fire at pair 0 would read
///   act windows whose gelu writers have not run.
/// * **THE FOLDED STORE**: the FINAL add writes the `desc_o` PARAMETER itself (its
///   binding, not a synth), and NOTHING else writes it — the o store folds into the
///   producing add, the same law the act store and the rs/ids stores ride.
#[test]
fn the_full_kernel_bakes_the_combine_interleaved_per_pair() {
    let ops = drive_ops_src(SRC, K).expect(
        "the full fused MoE kernel bakes through the whole-function door — the combine rung \
         is the last recogniser the ladder owed",
    );
    let names: Vec<&str> = ops.iter().map(|o| o.op_name.as_str()).collect();
    // The o parameter's own tid, read off the node's bindings (desc_o is the twelfth
    // pointer of the signature; the bindings follow it in order).
    let o_ptr = PTRS
        .iter()
        .position(|(p, _)| *p == "desc_o")
        .expect("desc_o is in the signature");
    let node = drive_node().expect("the ladder drive is green");
    let o_tid = node.bindings[o_ptr].get();
    // The whole ladder's emission: the norm head (5), the logits dot (1), the router (64),
    // the gate and up projections with their ONE shared quantize (2·30 + 5), and per pair
    // the gated act (2), the ten down legs, and the combine's legs (1 at pair 0, 2 after).
    assert_eq!(names.len(), 238, "the full bake's op count");
    // THE COMBINE'S OWN LEGS, as a subsequence: exactly the 2k−1 names, in order.
    let ecomb: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("ecomb"))
        .collect();
    let expect: Vec<String> = std::iter::once(format!("ecombmul_j0_o{o_tid}"))
        .chain((1..K).flat_map(|j| {
            [
                format!("ecombmul_j{j}_o{o_tid}"),
                format!("ecombadd_j{j}_o{o_tid}"),
            ]
        }))
        .collect();
    assert_eq!(
        ecomb, expect,
        "the combine's 2k−1 legs: slot 0's product seeds the accumulator, every later pair \
         a product leg then an add leg, the final add last — the left-deep slot-order sum"
    );
    // THE PER-POSITION LAW: pair j's product leg stands IMMEDIATELY after pair j's down
    // legs, and (j>0) its add leg immediately after the product — the kernel's own
    // interleaving, preserved op for op.
    for j in 0..K {
        let mul = names
            .iter()
            .position(|n| *n == format!("ecombmul_j{j}_o{o_tid}"))
            .unwrap_or_else(|| panic!("pair {j}'s product leg is in the bake"));
        assert!(
            names[mul - 1].starts_with(&format!("emdo_p{j}_")),
            "pair {j}'s product leg fires immediately after its own down legs"
        );
        if j > 0 {
            assert_eq!(
                names[mul + 1],
                format!("ecombadd_j{j}_o{o_tid}"),
                "pair {j}'s add leg fires immediately after its product leg"
            );
        }
    }
    // THE FOLDED STORE: the final add writes the o PARAMETER itself, and it is the ONE op
    // that does — the store folds into the producing add and nothing else touches the
    // output.
    let last = ops.last().expect("the bake is non-empty");
    assert_eq!(
        last.op_name,
        format!("ecombadd_j{}_o{o_tid}", K - 1),
        "the final op is the last pair's add"
    );
    let writers: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| {
            o.arg_bindings
                .iter()
                .any(|b| !b.is_input && b.buffer == act_name(o_tid))
        })
        .collect();
    assert_eq!(
        writers.len(),
        1,
        "exactly one op writes the o parameter — the folded store's law"
    );
    let out_binding = last
        .arg_bindings
        .iter()
        .find(|b| !b.is_input)
        .expect("the final add states an output");
    assert_eq!(
        out_binding.buffer,
        act_name(o_tid),
        "the final add writes the o parameter itself, not a synth"
    );
}

/// ⭐ TOP-1: at `K=1` the combine is ONE product leg and NO add — the product IS the
/// whole sum, so `ecombmul_j0` writes the o parameter directly (the same k=1 law the
/// three one-hot doors fixed at `300e41fb1`). The whole kernel bakes at top-1: the
/// router's top-1 arm is `router_door.rs`'s own green drive, and the projections hold a
/// single pair each.
#[test]
fn the_full_kernel_bakes_at_top_1_with_a_single_product_leg() {
    let ops =
        drive_ops_src(SRC, 1).expect("the full fused MoE kernel bakes at K=1 (top-1 routing)");
    let names: Vec<&str> = ops.iter().map(|o| o.op_name.as_str()).collect();
    let node = drive_node_src(SRC, 1).expect("the ladder drive is green at K=1");
    let o_ptr = PTRS
        .iter()
        .position(|(p, _)| *p == "desc_o")
        .expect("desc_o is in the signature");
    let o_tid = node.bindings[o_ptr].get();
    let ecomb: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.starts_with("ecomb"))
        .collect();
    assert_eq!(
        ecomb,
        vec![format!("ecombmul_j0_o{o_tid}").as_str()],
        "at top-1 the combine is ONE product leg — no accumulating add follows"
    );
    // The product leg writes the o parameter ITSELF (the seed-is-the-result law), and it
    // is the one op that does.
    let mul = ops
        .iter()
        .find(|o| o.op_name == format!("ecombmul_j0_o{o_tid}"))
        .expect("the single product leg");
    let out_binding = mul
        .arg_bindings
        .iter()
        .find(|b| !b.is_input)
        .expect("the product leg states an output");
    assert_eq!(
        out_binding.buffer,
        act_name(o_tid),
        "the top-1 product writes the o parameter itself, not a synth"
    );
}

/// ⛔ THE UNCLAIMED READER — a product over a non-terminal. moe.py's product line gains a
/// `·2.0` on the down side, so the product multiply's expert side is a plain `mulf`, not a
/// down projection's terminal. The combine weights the pair's OWN down output; a product
/// over anything else is an op this door does not recognise, refused BY NAME at the
/// per-pair agreement — never silently lowered as if it were the terminal.
#[test]
fn a_product_over_a_non_terminal_is_refused_by_name() {
    let src = src_with(&[(
        "acc = acc + down * w[:, None]",
        "acc = acc + down * 2.0 * w[:, None]",
    )]);
    let err = match drive_ops_src(&src, K) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — the unclaimed-reader mutant lowered; the per-pair agreement did \
             not fire",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:") && err.contains("is not a down projection's terminal"),
        "refused, but not for the reason this test pins: {err:?}"
    );
}

/// ⛔ THE WRONG SLOT — the one-hot's lane disagrees with the terminal's pair. moe.py's
/// compare reads lane `K−1−j` instead of `j`, so every pair's weight is another pair's: a
/// mismatch is two pairs wearing one chain (pair `p`'s down output weighted by slot `q`'s
/// score), and the per-pair agreement refuses it BY NAME rather than emitting a combine
/// that weights the wrong expert.
#[test]
fn a_wrong_slot_weight_is_refused_by_name() {
    let src = src_with(&[("koffs[None, :] == j", "koffs[None, :] == (K - 1 - j)")]);
    let err = match drive_ops_src(&src, K) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — the wrong-slot mutant lowered; the per-pair agreement did not \
             fire",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:")
            && err.contains("weights pair")
            && err.contains("the one-hot's lane and the terminal's pair must agree"),
        "refused, but not for the reason this test pins: {err:?}"
    );
}

/// ⛔ THE REORDERED SUM — the pair loop runs backward, so the accumulation is right-deep
/// seeded at pair `K−1` instead of left-deep seeded at pair 0. f16 is not associative, so
/// a different accumulation ORDER is a different function than the reference semantics
/// (`make_ttir` folds moe.py's zero seed into a LEFT-DEEP SLOT-ORDER chain — measured);
/// the proof refuses the reorder BY NAME rather than silently re-associating the sum.
#[test]
fn a_reordered_sum_is_refused_by_name() {
    let src = src_with(&[(
        "for j in tl.static_range(0, K):\n            e = tl.load(desc_ids + (m * K + j))",
        "for j in tl.static_range(K - 1, -1, -1):\n            e = tl.load(desc_ids + (m * K + j))",
    )]);
    let err = match drive_ops_src(&src, K) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — the reordered-sum mutant lowered; the slot-order law did not fire",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:") && err.contains("left-deep in slot order"),
        "refused, but not for the reason this test pins: {err:?}"
    );
}

/// ⛔ THE DANGLING RESULT — the o store reads `acc + down` instead of the weighted sum
/// itself, so the final add's sole consumer is another `addf`, not the store. The combine's
/// result is the function's output and nothing else's input; the folded-store law refuses
/// the dangle BY NAME rather than emitting a combine whose output quietly feeds an op the
/// door never accounted for.
#[test]
fn a_dangling_result_is_refused_by_name() {
    let src = src_with(&[(
        "o_desc.store([m, 0], acc)",
        "o_desc.store([m, 0], acc + down)",
    )]);
    let err = match drive_ops_src(&src, K) {
        Ok(ops) => panic!(
            "BAKED ({} ops) — the dangling-result mutant lowered; the folded-store law did not \
             fire",
            ops.len()
        ),
        Err(msg) => msg,
    };
    assert!(
        err.starts_with("door:") && err.contains("not the o store"),
        "refused, but not for the reason this test pins: {err:?}"
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
