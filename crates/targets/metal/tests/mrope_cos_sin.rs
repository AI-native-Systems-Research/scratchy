// SPDX-License-Identifier: Apache-2.0
//! Unit tests for the MRoPE per-token cos/sin override builder
//! (`build_mrope_cos_sin_override`, option (b) of the Qwen3.5-VL metal
//! mrope wiring). Verifies the band-split position selection, the
//! text-token broadcast, and the row layout / dtype — all without a GPU.

use scratchy_target_metal::interpreter::metal::{MetalDtype, build_mrope_cos_sin_override};

// Qwen3.5-VL text-decoder rope params (partial rotary 0.25 * head_dim 256).
const ROT_DIM: usize = 64;
const HALF: usize = ROT_DIM / 2; // 32 rotary pairs
const THETA: f64 = 100_000.0;
const SECTION: [u32; 3] = [11, 11, 10]; // T/H/W pair split (sums to HALF)

fn build(positions: &[u32], n: usize) -> Vec<u8> {
    build_mrope_cos_sin_override(positions, n, ROT_DIM, THETA, SECTION, MetalDtype::Bf16)
}

/// Each rotary pair `i` of a `[3, n]` build must equal pair `i` of a 1D
/// build at the band-owning position: T for `i < 11`, H for `i < 22`, W
/// otherwise. Proves both the band selection AND that every band runs the
/// identical angle math (function-vs-function, no external reference).
#[test]
fn band_split_picks_the_section_correct_position() {
    let row_t = build(&[10], 1); // every pair uses pos 10
    let row_h = build(&[100], 1); // every pair uses pos 100
    let row_w = build(&[1000], 1); // every pair uses pos 1000
    let row_3d = build(&[10, 100, 1000], 1); // T=10, H=100, W=1000

    let pair = |buf: &[u8], i: usize| -> ([u8; 2], [u8; 2]) {
        let cos = [buf[2 * i], buf[2 * i + 1]];
        let sin = [buf[2 * (HALF + i)], buf[2 * (HALF + i) + 1]];
        (cos, sin)
    };
    for i in 0..HALF {
        let (band, src) = if i < 11 {
            ("T", &row_t)
        } else if i < 22 {
            ("H", &row_h)
        } else {
            ("W", &row_w)
        };
        assert_eq!(
            pair(&row_3d, i),
            pair(src, i),
            "pair {i} must come from the {band} band"
        );
    }
}

/// A text token inside a `[3, n]` batch has T == H == W, so its row must be
/// byte-identical to a 1D build at that position (the broadcast case the
/// macro forward hits for non-image tokens / decode).
#[test]
fn text_token_broadcast_equals_1d() {
    assert_eq!(build(&[7, 7, 7], 1), build(&[7], 1));
    // Multi-token broadcast: [3, 2] with all rows equal == 1D over both.
    assert_eq!(build(&[3, 9, 3, 9, 3, 9], 2), build(&[3, 9], 2));
}

/// Row layout / dtype / size: `[n, ROT_DIM]` bf16, `[cos(0..HALF) |
/// sin(0..HALF)]` per row. At pos 0 every angle is 0, so cos == 1.0
/// (bf16 `0x3F80`) and sin == 0.0 (bf16 `0x0000`) for all pairs — a
/// hardcoded golden that pins the layout and element type with no
/// floating-point dependency.
#[test]
fn one_d_layout_dtype_and_size() {
    let positions = [0u32, 1, 5, 42, 4096];
    let out = build(&positions, positions.len());
    assert_eq!(out.len(), positions.len() * ROT_DIM * 2);

    let row0 = &out[0..ROT_DIM * 2];
    let one_bf16 = 0x3F80u16.to_ne_bytes(); // bf16(1.0)
    let zero_bf16 = 0x0000u16.to_ne_bytes(); // bf16(0.0)
    for i in 0..HALF {
        assert_eq!(
            [row0[2 * i], row0[2 * i + 1]],
            one_bf16,
            "cos pair {i} at pos 0 must be 1.0"
        );
        assert_eq!(
            [row0[2 * (HALF + i)], row0[2 * (HALF + i) + 1]],
            zero_bf16,
            "sin pair {i} at pos 0 must be 0.0"
        );
    }
}
