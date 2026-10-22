// SPDX-License-Identifier: Apache-2.0
//! THE DEVICE ARGMAX'S ADMISSION, pinned from both sides — the mint-site gate
//! [`lm_head_admits_the_argmax`] over `SubtileIR`s minted by `lower_region`,
//! the same front door a model's forward comes through.
//!
//! ⭐ THE UNTIED-REFUSAL PIN, INVERTED. #698's composite device argmax gathered
//! the chosen token's embedding row ON CARD, so its admission carried a TIE
//! conjunct — untied models (llama) placed no composite and kept the host token
//! step, pinned by test. The spliced kernel gathers NOTHING: it reads the logits
//! row and answers an index, so the tie conjunct is DROPPED, and the admission
//! is keyed on the tail's own staging alone. The inversion pinned here: an
//! UNTIED-shaped tail — a dense lm-head reading its own weight source, no
//! embedding bank anywhere in the graph — ADMITS.
//!
//! The boundary that remains is the STAGING, never the preset's presence or a
//! model's name: an fp8-dynamic tail (arity-3, the per-channel scale as the
//! matmul's third operand) does not mint, and a tail with no recognizable
//! lm-head matmul under its single-input post-op chain does not mint either —
//! quietly, never a build refusal, because the kernel demands nothing from the
//! lm-head. The post-op chain itself stays TRANSPARENT: granite's logits-scaling
//! ScalarMul and gemma's TanhSoftCap are walked through, exactly where they sit
//! in the real tails.
//!
//! ⭐ THE TWO RANGE LAWS (#813 review), pinned from both sides at real vocabs:
//! the staged bramp/binc carry BLOCK NUMBERS as fp16, exact only to 2048 — a
//! vocab past that (gemma-4/nemotron's 262144) would collide adjacent block
//! numbers and answer the LATER block, a silent wrong token; and a POWER-OF-TWO
//! vocab (262144, mistral-nemo's 131072) has no pad lane, so the all-NaN
//! sentinel has nowhere to live. Both refuse here and the model keeps the host
//! argmax; the boundary case (NB exactly 2048, one pad lane) still admits.
//!
//! No device, no emulator session: these are pure facts of the graph, so the
//! file runs at default features.

use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SourceShape, SubOp, lower_region};
use scratchy_target_spyre::lower_subtile_tape_to_ktir::lm_head_admits_the_argmax;

/// The one-request decode tail over `ops`, K=4 vocab 5 — the smallest shape
/// every case below differs on.
fn tail(ops: Vec<OpDesc>, result: usize) -> bool {
    let input = LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: 4 },
            SourceShape { rows: 4, cols: 5 },
            // The fp8 case's per-channel scale, `[1, n]`.
            SourceShape { rows: 1, cols: 5 },
        ],
        ops,
        result,
    };
    let g = lower_region(&input, std::num::NonZeroU32::new(64).unwrap());
    lm_head_admits_the_argmax(&g)
}

fn matmul(weight: GemmWeight, n_inputs: usize) -> OpDesc {
    let inputs = (0..n_inputs)
        .map(|i| match i {
            2 => InputRef::Ext(2),
            j => InputRef::Ext(j),
        })
        .collect();
    OpDesc {
        op: SubOp::MatmulTile { n: 5, weight },
        m: 1,
        inputs,
    }
}

#[test]
fn an_untied_dense_lm_head_admits_the_device_argmax() {
    // THE INVERTED PIN: the weight is the graph's own second source — no
    // embedding bank, no tie fact anywhere — and the tail admits. #698 refused
    // exactly this shape; the spliced kernel has no gather link to demand a tie.
    assert!(
        tail(vec![matmul(GemmWeight::Dense, 2)], 0,),
        "an untied dense lm-head admits the device argmax — the tie conjunct is gone"
    );
}

#[test]
fn an_fp8_dynamic_lm_head_does_not_admit() {
    assert!(
        !tail(vec![matmul(GemmWeight::Fp8Dynamic, 3)], 0,),
        "an fp8-dynamic tail stages its scale as the matmul's third operand — no mint"
    );
}

#[test]
fn the_logits_epilogues_stay_transparent_to_the_walk() {
    // granite's tail: the vocab matmul under its logits-scaling ScalarMul.
    assert!(
        tail(
            vec![
                matmul(GemmWeight::Dense, 2),
                OpDesc {
                    op: SubOp::ScalarMul { scale: 8.0 },
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            1,
        ),
        "the argmax reads THROUGH granite's logits-scaling ScalarMul"
    );
    // gemma's tail: the vocab matmul under its TanhSoftCap.
    assert!(
        tail(
            vec![
                matmul(GemmWeight::Dense, 2),
                OpDesc {
                    op: SubOp::TanhSoftCap { cap: 30.0 },
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            1,
        ),
        "the argmax reads THROUGH gemma's TanhSoftCap"
    );
}

#[test]
fn a_tail_without_an_lm_head_matmul_does_not_admit_quietly() {
    // The result ends at a norm, not a matmul: no recognized tail, no mint —
    // and no build refusal, the quiet boundary.
    assert!(
        !tail(
            vec![
                matmul(GemmWeight::Dense, 2),
                OpDesc {
                    op: SubOp::RmsNormUnit { eps: 1e-6 },
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            1,
        ),
        "a non-matmul result tail does not mint — the model keeps the host argmax"
    );
}

/// The one-request dense tail at a given VOCAB — the range laws read the result
/// row's own width (the emitted vocab, the same number the mint and the staged
/// tables derive from), so this number alone moves the two new conjuncts.
fn tail_at(vocab: u32) -> bool {
    let input = LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: 4 },
            SourceShape { rows: 4, cols: vocab },
        ],
        ops: vec![OpDesc {
            op: SubOp::MatmulTile {
                n: vocab,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    };
    let g = lower_region(&input, std::num::NonZeroU32::new(64).unwrap());
    lm_head_admits_the_argmax(&g)
}

#[test]
fn a_vocab_past_the_fp16_block_range_does_not_admit() {
    // 200000 rounds to PAD_V 262144 with NB 4096 > 2048 while PAD_V != V, so ONLY
    // the block-range law refuses here — fp16's exact-integer range ends at 2048
    // (spacing 2 above it), and the staged bramp/binc would collide adjacent
    // block numbers into a silently later-block answer. gemma-4/nemotron's
    // 262144 violates this AND the pad-lane law; this case isolates the range law.
    assert!(
        !tail_at(200_000),
        "a vocab baking NB > 2048 does not mint — the staged block numbers leave fp16's \
         exact-integer range and the model keeps the host argmax"
    );
}

#[test]
fn a_power_of_two_vocab_does_not_admit() {
    // mistral-nemo's 131072: NB is exactly 2048 (in range) but PAD_V == V — there
    // is no pad lane, padoff is all zeros, and an all-NaN row answers lane 0 while
    // the host reference answers the sentinel. The pad-lane law refuses alone here.
    assert!(
        !tail_at(131_072),
        "a power-of-two vocab has no pad lane, so the all-NaN sentinel law has nowhere to \
         live — the model keeps the host argmax"
    );
}

#[test]
fn the_widest_in_range_vocab_still_admits() {
    // 131071 rounds to PAD_V 131072: NB is exactly 2048 and one pad lane exists —
    // the boundary case on the ADMITTING side, so neither law refuses a hair early.
    assert!(
        tail_at(131_071),
        "NB exactly 2048 with a pad lane still admits — the boundary is the law, not a \
         margin"
    );
}
