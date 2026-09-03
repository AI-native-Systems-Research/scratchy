// SPDX-License-Identifier: Apache-2.0
//! Correctness test for the constrained-decoding `grammar_mask` kernel:
//! forces every token NOT in a request's grammar allow-set to -inf, in
//! place, leaving allowed tokens (and unmasked rows) untouched — so the
//! downstream greedy argmax can only pick a grammar-valid token.
//!
//! Dispatches on the production MTL4 path (see
//! `common::dispatch_threadgroups`).

mod common;

use half::{bf16, f16};
use objc2_metal::MTLSize;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::grammar_mask::{
    GRAMMAR_MASK_TG_SIZE, GrammarMaskKernels, push_allow_bitset_row, words_per_row,
};

/// argmax of a masked logits row (smaller index wins ties), matching the
/// `argmax` kernel convention. Treats -inf as "never chosen".
fn argmax_row(row: &[f32]) -> usize {
    let mut best = f32::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

/// grammar_mask kernel binding contract: `buffer(0)=logits` (in/out),
/// `buffer(1)=allow_bits`, `buffer(2)=rows`, `buffer(3)=vocab` (u32),
/// `buffer(4)=words_per_row` (u32); grid is one threadgroup per masked
/// row, `GRAMMAR_MASK_TG_SIZE` threads striding the vocab axis.
fn run_grammar_mask_mtl4(
    use_bf16: bool,
    logits: &common::Buffer,
    allow_buf: &common::Buffer,
    rows_buf: &common::Buffer,
    num_rows: u32,
    vocab: u32,
    words_per_row: u32,
) -> bool {
    let device = detect_device()
        .expect("Metal 4 GPU present (caller pre-guards)")
        .device;
    let kernels = GrammarMaskKernels::new(&device).expect("grammar_mask kernels");
    let pso = if use_bf16 {
        &kernels.bf16
    } else {
        &kernels.f16
    };
    let vocab_buf = common::shared_u32(&device, vocab);
    let wpr_buf = common::shared_u32(&device, words_per_row);
    common::dispatch_threadgroups(
        &device,
        pso,
        &[logits, allow_buf, rows_buf, &vocab_buf, &wpr_buf],
        MTLSize {
            width: num_rows as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GRAMMAR_MASK_TG_SIZE,
            height: 1,
            depth: 1,
        },
    )
}

#[test]
fn grammar_mask_bf16_mixed_batch() {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let batch = 3usize;
    let vocab = 100u32;
    // logits[r][v] = v  → global max is v=99 (disallowed) for any allow-set
    // that excludes it, so masking must shift argmax to the best ALLOWED id.
    let host_f32: Vec<f32> = (0..batch)
        .flat_map(|_r| (0..vocab).map(|v| v as f32))
        .collect();
    let host_bf16: Vec<bf16> = host_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let logits = common::shared_slice(&device, &host_bf16);

    // Mask logits row 0 (allow {3,50,75}) and row 2 (allow {10}); leave
    // row 1 untouched (not present in `rows`).
    let wpr = words_per_row(vocab);
    let mut allow_bits: Vec<u32> = Vec::new();
    push_allow_bitset_row(&mut allow_bits, &[3, 50, 75], vocab);
    push_allow_bitset_row(&mut allow_bits, &[10], vocab);
    let rows: Vec<u32> = vec![0, 2];
    let allow_buf = common::shared_slice(&device, &allow_bits);
    let rows_buf = common::shared_slice(&device, &rows);

    if !run_grammar_mask_mtl4(
        true,
        &logits,
        &allow_buf,
        &rows_buf,
        rows.len() as u32,
        vocab,
        wpr,
    ) {
        return;
    }

    let got: Vec<bf16> = common::read_slice(&logits, batch * vocab as usize);
    let row = |r: usize| -> Vec<f32> {
        got[r * vocab as usize..(r + 1) * vocab as usize]
            .iter()
            .map(|x| x.to_f32())
            .collect()
    };

    // Row 0: allowed {3,50,75} keep value v; everything else -inf.
    let r0 = row(0);
    for (v, &val) in r0.iter().enumerate() {
        if [3usize, 50, 75].contains(&v) {
            assert_eq!(val, v as f32, "row0 allowed token {v} changed");
        } else {
            assert!(val.is_infinite() && val < 0.0, "row0 token {v} not -inf");
        }
    }
    assert_eq!(argmax_row(&r0), 75, "row0 argmax must be best allowed (75)");

    // Row 1: untouched.
    let r1 = row(1);
    for (v, &val) in r1.iter().enumerate() {
        assert_eq!(val, v as f32, "row1 (unmasked) token {v} changed");
    }
    assert_eq!(argmax_row(&r1), 99, "row1 argmax is the unmasked max (99)");

    // Row 2: only token 10 survives.
    let r2 = row(2);
    for (v, &val) in r2.iter().enumerate() {
        if v == 10 {
            assert_eq!(val, 10.0, "row2 allowed token 10 changed");
        } else {
            assert!(val.is_infinite() && val < 0.0, "row2 token {v} not -inf");
        }
    }
    assert_eq!(
        argmax_row(&r2),
        10,
        "row2 argmax must be the only allowed (10)"
    );
}

#[test]
fn grammar_mask_f16_large_vocab_multiword() {
    // Large vocab exercises the multi-word bitset (words_per_row > 1) and
    // the threadgroup stride (256 threads over thousands of tokens).
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let vocab = 5000u32;
    // logits[v] = (v % 1000) as f32 so the allowed set's max is well-defined
    // and several disallowed tokens out-rank it pre-mask.
    let host_f16: Vec<f16> = (0..vocab)
        .map(|v| f16::from_f32((v % 1000) as f32))
        .collect();
    let logits = common::shared_slice(&device, &host_f16);

    let allowed: Vec<u32> = vec![0, 31, 32, 63, 64, 999, 4096, 4999];
    let wpr = words_per_row(vocab);
    let mut allow_bits: Vec<u32> = Vec::new();
    push_allow_bitset_row(&mut allow_bits, &allowed, vocab);
    let rows: Vec<u32> = vec![0];
    let allow_buf = common::shared_slice(&device, &allow_bits);
    let rows_buf = common::shared_slice(&device, &rows);

    if !run_grammar_mask_mtl4(false, &logits, &allow_buf, &rows_buf, 1, vocab, wpr) {
        return;
    }

    let got: Vec<f16> = common::read_slice(&logits, vocab as usize);
    for (v, gv) in got.iter().enumerate() {
        let g = gv.to_f32();
        if allowed.contains(&(v as u32)) {
            assert_eq!(g, (v as u32 % 1000) as f32, "allowed token {v} changed");
        } else {
            assert!(g.is_infinite() && g < 0.0, "token {v} not -inf");
        }
    }
    // Best allowed value: token 999 has value 999 (the max among allowed).
    let masked: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();
    assert_eq!(
        argmax_row(&masked),
        999,
        "argmax must be best allowed (999)"
    );
}

#[test]
fn grammar_mask_bf16_padded_vocab_tail_masked() {
    // Simulates an lm_head whose logits width (vocab) exceeds the bitset
    // capacity (words_per_row*32) — i.e. padding past the tokenizer vocab.
    // Every slot beyond the bitset must be forced to -inf (never sampled),
    // and the bits[] read must stay in bounds.
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let vocab = 100u32; // runtime logits width
    let words = 2u32; // bitset covers only [0, 64); [64, 100) is "padding"
    assert!(words * 32 < vocab, "test must exercise the tail path");

    let host: Vec<bf16> = (0..vocab).map(|v| bf16::from_f32(v as f32)).collect();
    let logits = common::shared_slice(&device, &host);

    // Allow tokens 3 and 50 (both inside the bitset). Token 99 (the unmasked
    // global max) lives in the padded tail and MUST be masked.
    let mut allow_bits = vec![0u32; words as usize];
    for &t in &[3u32, 50] {
        allow_bits[(t >> 5) as usize] |= 1u32 << (t & 31);
    }
    let allow_buf = common::shared_slice(&device, &allow_bits);
    let rows_buf = common::shared_slice(&device, &[0u32]);

    if !run_grammar_mask_mtl4(true, &logits, &allow_buf, &rows_buf, 1, vocab, words) {
        return;
    }

    let got: Vec<f32> = common::read_slice::<bf16>(&logits, vocab as usize)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    for (v, &val) in got.iter().enumerate() {
        if v == 3 || v == 50 {
            assert_eq!(val, v as f32, "allowed token {v} changed");
        } else {
            assert!(
                val.is_infinite() && val < 0.0,
                "token {v} (incl padded tail >=64) must be -inf, got {val}"
            );
        }
    }
    assert_eq!(
        argmax_row(&got),
        50,
        "argmax must be best allowed (50), not padded max 99"
    );
}
