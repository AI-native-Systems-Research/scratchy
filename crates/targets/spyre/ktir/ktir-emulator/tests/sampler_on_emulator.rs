// SPDX-License-Identifier: Apache-2.0
//! THE TOKEN DRAW ON THE EMULATOR'S OWN SESSION. The minted greedy/Gumbel sampler (pinned in
//! `fixtures/sampler_granite/minted.rs`, granite's 49155-wide row padded to 65536) runs through
//! [`Session`] the way the spyre target's resident session runs it, on a logits row whose
//! maximum is planted: a greedy draw must answer exactly that index.
//!
//! ⭐ UNDER THE CARD'S WIDEN MODEL. The program states a pin window (`Some`), so the resident runs
//! it through `comm_sched`'s pinned entry, which arms `ConvertModel::Card`: every f16 to f32
//! `arith.extf` writes the card's `dl16tofp32` lane stagger, and the tables the bundle binds are
//! staged in that arrangement. On macOS the row's staggered widens run as fused Metal map
//! windows, the path that drew argmax + 8 when it widened without the stagger.
#![cfg(feature = "optimizer")]

use std::collections::{HashMap, HashSet};

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use ktir_emulator::program::Session;
use ktir_superdsc::element_arrangement::dl16tofp32_stagger;

#[rustfmt::skip]
#[path = "fixtures/sampler_granite/minted.rs"]
mod minted;

/// The sampler's lane block, the blocks per group of its two-level ramp, and its group count.
const BLOCK: usize = 64;
const GROUP_BLOCKS: usize = 64;
const GROUPS: usize = minted::PAD_VOCAB / (BLOCK * GROUP_BLOCKS);

/// One of the sampler's twenty operand tables in LOGICAL lane order: the host contract
/// `crates/targets/spyre/kernels/sample_fwd.py` states, as the spyre target's
/// `sample_staged_tables` builds it. Whole-row tables give lane `i < VOCAB` its value and a pad
/// lane its pad value; the `[rows, 1]` tables hold one word.
fn logical_table(operand: u8) -> Vec<u8> {
    let (v, pad) = (minted::VOCAB, minted::PAD_VOCAB);
    let row = |real: &dyn Fn(usize) -> usize, pad_lane: usize| -> Vec<f32> {
        (0..pad)
            .map(|i| if i < v { real(i) } else { pad_lane } as f32)
            .collect()
    };
    let f32s = |w: Vec<f32>| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    match operand {
        // bramp, lramp, binc, linc: the block and lane ramps and indices.
        1 => f32s(row(&|i| GROUP_BLOCKS - (i / BLOCK) % GROUP_BLOCKS, 0)),
        2 => f32s(row(&|i| BLOCK - i % BLOCK, 0)),
        3 => f32s(row(&|i| (i / BLOCK) % GROUP_BLOCKS, GROUP_BLOCKS)),
        4 => f32s(row(&|i| i % BLOCK, BLOCK)),
        // ones, sixtyfour: `[1, BLOCK]`.
        5 => f32s(vec![1.0; BLOCK]),
        6 => f32s(vec![64.0; BLOCK]),
        // padoff: 0 on the real lanes, f32::MIN on the pads.
        7 => f32s(
            (0..pad)
                .map(|i| if i < v { 0.0 } else { f32::MIN })
                .collect(),
        ),
        // gramp, ginc: the group ramp and index.
        8 => f32s(row(&|i| GROUPS - i / (BLOCK * GROUP_BLOCKS), 0)),
        9 => f32s(row(&|i| i / (BLOCK * GROUP_BLOCKS), GROUPS)),
        // hc1, hc2, hc3, c256: the counter hash's i32 words.
        10 => (-1640531535i32).to_le_bytes().to_vec(),
        11 => (-2048144777i32).to_le_bytes().to_vec(),
        12 => (-1028477379i32).to_le_bytes().to_vec(),
        13 => 256i32.to_le_bytes().to_vec(),
        // lane: every lane its own index, pads included.
        14 => f32s((0..pad).map(|i| i as f32).collect()),
        // fm8, ftop, fzero, fhalf, fm22: the fold's f32 constants.
        15 => f32s(vec![0.00390625]),
        16 => f32s(vec![8388607.0]),
        17 => f32s(vec![0.0]),
        18 => f32s(vec![0.5]),
        19 => f32s(vec![2f32.powi(-22)]),
        // cmin: the fp16 clamp 2^-14.
        20 => ktir_emulator::codec::encode(&[2f32.powi(-14)], DType::F16),
        other => panic!("the sampler has twenty operand tables, not {other}"),
    }
}

/// A table's bytes as the bundle stages them: the logical words, permuted into the
/// `Dl16ToFp32` arrangement by ktir-superdsc's own staging when the bake marked it so.
fn staged_table(operand: u8, dtype: DType, shape: [usize; 2], staggered: bool) -> Vec<u8> {
    let bytes = logical_table(operand);
    assert_eq!(
        bytes.len(),
        shape[0] * shape[1] * dtype.bytes_per_elem(),
        "operand {operand}: the table's bytes must fill its [rows, cols] = {shape:?}"
    );
    if !staggered {
        return bytes;
    }
    assert_eq!(
        dtype,
        DType::F32,
        "operand {operand}: the stagger is an f32-word fact"
    );
    let words: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    dl16tofp32_stagger(&words)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// One resident session over the sampler, every load-time table and the greedy seat staged.
fn session() -> Session {
    let bindings: Vec<Binding> = minted::BINDINGS
        .iter()
        .map(|&(arg, tensor, is_output)| Binding {
            arg,
            tensor,
            is_output,
        })
        .collect();
    let sources: HashSet<u64> = bindings
        .iter()
        .filter(|b| !b.is_output)
        .map(|b| b.tensor)
        .collect();
    let spec = ProgramSpec {
        nodes: vec![NodeSpec {
            func: minted::SAMPLER.name.to_string(),
            program: minted::PROGRAM,
            bindings,
        }],
        sources,
        results: [minted::IDX_TID].into_iter().collect(),
    };
    let tensor = |data: Vec<u8>, shape: [usize; 2], dtype: DType| Arg::TensorBytes {
        data,
        shape: shape.to_vec(),
        dtype,
    };
    let mut weights: Vec<(u64, Arg)> = minted::TABLES
        .iter()
        .map(|&(tid, operand, dtype, shape, staggered)| {
            let data = staged_table(operand, dtype, shape, staggered);
            (tid, tensor(data, shape, dtype))
        })
        .collect();
    // The greedy seat: the seed quadruple's words at 0 (the gate word's 0 IS the greedy law),
    // and the temperature's reciprocal at 1.0.
    for tid in minted::SEED_TIDS {
        weights.push((tid, tensor(0i32.to_le_bytes().to_vec(), [1, 1], DType::I32)));
    }
    weights.push((
        minted::INVT_TID,
        tensor(1.0f32.to_le_bytes().to_vec(), [1, 1], DType::F32),
    ));
    let funcs = std::slice::from_ref(&minted::SAMPLER);
    // `Some` pin window: the pinned entry, so the card's widen model.
    Session::new_multi_floored(
        vec![(funcs, &spec)],
        weights,
        &HashMap::new(),
        vec![Some(0)],
    )
    .expect("the sampler's session builds")
}

/// Draw greedily over the row with its unique maximum planted at `argmax`.
fn greedy_draw(argmax: usize) -> u32 {
    let mut session = session();
    // Distinct small values everywhere, the planted maximum well above them, zeros on the pads
    // (the kernel's own pad law drives the pads out of the contest).
    let logits: Vec<f32> = (0..minted::PAD_VOCAB)
        .map(|i| {
            if i == argmax {
                0.5
            } else if i < minted::VOCAB {
                ((i * 7919) % 1009) as f32 * 1e-4
            } else {
                0.0
            }
        })
        .collect();
    let row = Arg::Tensor {
        data: logits,
        shape: vec![1, minted::PAD_VOCAB],
        dtype: DType::F16,
    };
    session
        .set_sources(&[(minted::ROW_TID, row)])
        .expect("the row stages");
    let out = session
        .run_program(0, &[(minted::IDX_TID, 0)])
        .expect("the draw runs");
    out[&minted::IDX_TID].data[0] as u32
}

/// The fused map windows a draw runs that read through the card's stagger table. The sampler
/// widens f16 to f32 twice on its padded row (its log islands' own widens reach the emulator
/// already collapsed to bare f16 logs), and each widen lands in a window of its own:
///   1. `x32 = x.to(tl.float32)`, the logits row's widen, alone in its window;
///   2. `noise = g16.to(tl.float32)`, the Gumbel noise's widen, fused with the second log and its
///      negation (`lt = tl.log(c2)`, `g16 = -lt`); the `tl.maximum` before them is not a fusable
///      map op, so the window starts at the log.
///
/// PINNED, not counted from the plan: a widen whose window fails to emit drops out of the plan
/// too, so a count read from the plan would fall with it and hide exactly that fallback.
#[cfg(metal)]
const STAGGERED_WINDOWS_PER_DRAW: usize = 2;

/// How many fused map windows that read through the stagger table have run on Metal so far,
/// where the emulator has its Metal backend; `None` elsewhere.
#[cfg(metal)]
fn staggered_windows_on_metal() -> Option<usize> {
    Some(
        ktir_emulator::metal::MAP_REGION_STAGGERED_GPU_COUNT
            .load(std::sync::atomic::Ordering::Relaxed),
    )
}

#[cfg(not(metal))]
fn staggered_windows_on_metal() -> Option<usize> {
    None
}

#[test]
fn a_greedy_draw_answers_the_planted_maximum() {
    let mut missed = Vec::new();
    let mut on_metal = Vec::new();
    for argmax in [0usize, 7, 8, 9, 15, 16, 39752, 39751, 39744, 49154] {
        let before = staggered_windows_on_metal();
        let got = greedy_draw(argmax);
        let windows = before.zip(staggered_windows_on_metal()).map(|(b, a)| a - b);
        println!(
            "  planted {argmax:>6} -> drew {got:>6} (diff {:+}){}",
            got as i64 - argmax as i64,
            windows.map_or(String::new(), |n| format!(
                ", {n} staggered windows on Metal"
            ))
        );
        if got as usize != argmax {
            missed.push(format!("{argmax} -> {got}"));
        }
        if let Some(n) = windows {
            on_metal.push((argmax, n));
        }
    }
    assert!(
        missed.is_empty(),
        "the greedy draw missed the planted maximum: {missed:?}"
    );
    // ⭐ THE OFFLOAD PROOF. The interpreter alone already draws right, so a staggered window
    // that fell back to it would pass the check above unseen: EACH draw must run BOTH on Metal.
    #[cfg(metal)]
    for (argmax, n) in on_metal {
        assert_eq!(
            n, STAGGERED_WINDOWS_PER_DRAW,
            "planted {argmax}: the draw must run each staggered widen's window as a fused \
             Metal kernel"
        );
    }
}
