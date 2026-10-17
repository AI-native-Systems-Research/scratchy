// SPDX-License-Identifier: Apache-2.0
//! Exactness test: `concat_rows` (`out[t] = a[t] ++ b[t]`: an MTP head's input fusion, or a
//! forward's exported hidden states side by side) against the rows laid side by side on the host.
//! A copy, so the output bits must equal the inputs' bits — checked on raw 16-bit patterns for
//! both dtype instantiations. Shapes cover the MTP decode row (hidden 2048), a verify step's rows,
//! an odd row count, and operands of different widths (gpt-oss's two exported residual streams
//! so far beside its third); the grid is over-launched past `CONCAT_ROWS_N` so the bounds guard
//! is exercised. Dispatches on the production MTL4 path (`common::dispatch_threadgroups`).

mod common;

use objc2_metal::{MTLBuffer, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::ids::ElementCount;
use scratchy_target_metal::tape::kernel_constants::ConcatRowsConstants;
use scratchy_target_metal::tape::lowered::{ActivationWidth, ConcatWidths};

/// Distinct 16-bit patterns per (operand, element), so a swapped half or a shifted row shows.
fn pattern(n: usize, tag: u16) -> Vec<u16> {
    (0..n)
        .map(|i| (i as u16).wrapping_mul(7).wrapping_add(tag))
        .collect()
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run(rows: usize, (wa, wb): (usize, usize), symbol: &str) {
    let Some(dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = dev.device;
    let (a, b) = (pattern(rows * wa, 1), pattern(rows * wb, 0x8001));
    let width = wa + wb;
    let n = rows * width;
    // A guard row past the output: the kernel must leave it alone.
    let out_buf = common::shared_zeroed(&device, (n + width) * 2);
    let (a_buf, b_buf) = (
        common::shared_bytes(&device, &bytes(&a)),
        common::shared_bytes(&device, &bytes(&b)),
    );

    let constants = ConcatRowsConstants {
        elements: ElementCount(n as u32),
        widths: ConcatWidths {
            a: ActivationWidth::of_cols(wa as u32),
            b: ActivationWidth::of_cols(wb as u32),
        },
    };
    let pipeline = baked_pipeline(&device, "elementwise", symbol, constants.into())
        .expect("concat_rows pipeline");
    let tg = 256;
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &a_buf, &b_buf],
        MTLSize {
            width: n.div_ceil(tg) + 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let want: Vec<u16> = a
        .chunks(wa)
        .zip(b.chunks(wb))
        .flat_map(|(ra, rb)| ra.iter().chain(rb).copied())
        .chain(std::iter::repeat_n(0, width))
        .collect();
    let got: Vec<u16> = unsafe {
        let p = out_buf.contents().as_ptr() as *const u16;
        (0..want.len()).map(|i| *p.add(i)).collect()
    };
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g,
            w,
            "{symbol} {rows}x({wa}+{wb}): element {i} (row {})",
            i / width
        );
    }
}

#[test]
fn concat_rows_bf16_decode_row() {
    run(1, (2048, 2048), "concat_rows_bf16");
}

#[test]
fn concat_rows_bf16_verify_rows() {
    run(4, (2048, 2048), "concat_rows_bf16");
}

#[test]
fn concat_rows_f16_odd_rows() {
    run(37, (24, 24), "concat_rows_f16");
}

#[test]
fn concat_rows_bf16_unequal_operands() {
    run(5, (2 * 2880, 2880), "concat_rows_bf16");
}
