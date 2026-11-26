// SPDX-License-Identifier: Apache-2.0
//! `mtp_chain` for a parallel-drafting head (P-EAGLE): before its one pass the chain lays out the
//! rows by what the target accepted, after it the drafts. Two sequences at two drafts a step: `a`
//! verified two drafts and the target kept one; `b` decoded one row. Before the pass `a`'s four
//! rows read the target's tokens and hidden rows through its newest (row 1), the mask token and
//! `mask_hidden` past it; `b`'s two rows its newest and one placeholder. After it each sequence's
//! drafts are the argmaxes from its newest on. Dispatches on the production MTL4 path
//! (`common::dispatch_threadgroups`).

mod common;

use objc2_metal::{MTLBuffer, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::ids::{HiddenSize, NumDrafts, RotDim};
use scratchy_target_metal::tape::kernel_constants::{DraftsPastFirst, MtpChainConstants};

const HIDDEN: usize = 300;
const DRAFTS: u32 = 2;
const MASK: u32 = 201_020;

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

fn halves(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|h| h.to_ne_bytes()).collect()
}

/// A target hidden row's elements, distinct per row.
fn row(r: u16) -> Vec<u16> {
    (0..HIDDEN as u16)
        .map(|i| i.wrapping_mul(3).wrapping_add(r << 10))
        .collect()
}

fn read<T: Copy>(buf: &common::Buffer, n: usize) -> Vec<T> {
    // SAFETY: the buffer holds at least `n` elements of `T`, and the command buffer is done.
    unsafe {
        let p = buf.contents().as_ptr() as *const T;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

#[test]
fn a_parallel_head_lays_out_its_pass_and_picks_its_drafts() {
    let Some(dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = dev.device;
    let constants = MtpChainConstants {
        hidden: HiddenSize(HIDDEN as u32),
        drafts: NumDrafts(DRAFTS),
        rope: RotDim(0),
        past_first: DraftsPastFirst::Parallel,
    };
    let pipeline = baked_pipeline(&device, "mtp_chain", "mtp_chain", constants.into())
        .expect("mtp_chain pipeline");
    // The step's rows: `a` at 0..3 (its token and two drafts), `b` at 3. The target kept `a`'s
    // first draft (its token at row 0 is the draft row 1 verifies), not its second.
    let target = [11, 12, 13, 40];
    let drafted = [11, 99, 0, 0];
    // Per sequence: first step row, token rows, first row in the pass.
    let seqs = [0, 3, 0, 3, 1, 4];
    let pass_rows = 6;
    let hidden: Vec<u16> = (0..4).flat_map(row).collect();
    let mask_hidden = row(9);
    let buf = |bytes: &[u8]| common::shared_bytes(&device, bytes);
    let zeroed = |len: usize| common::shared_zeroed(&device, len);
    let (target, drafted, seqs) = (
        buf(&words(&target)),
        buf(&words(&drafted)),
        buf(&words(&seqs)),
    );
    let (hidden, mask_hidden) = (buf(&halves(&hidden)), buf(&halves(&mask_hidden)));
    let accepted = zeroed(2 * 4);
    // The host lays the mask token under every row; the chain overwrites through each newest.
    let ids = buf(&words(&[MASK; 6]));
    let laid_hidden = zeroed(pass_rows * HIDDEN * 2);
    let drafts = zeroed(2 * DRAFTS as usize * 4);
    let unused = zeroed(64);
    let (window_start, rows) = (
        common::shared_u32(&device, 0),
        common::shared_u32(&device, 1),
    );
    // The pass's argmax, one a row.
    let picked = buf(&words(&[70, 71, 72, 73, 74, 75]));
    let run = |depth: u32| {
        let depth = common::shared_u32(&device, depth);
        let bindings = [
            &picked,
            &hidden,
            &target,
            &drafted,
            &seqs,
            &accepted,
            &unused,
            &ids,
            &unused,
            &unused,
            &unused,
            &laid_hidden,
            &drafts,
            &depth,
            &window_start,
            &unused,
            &unused,
            &rows,
            &mask_hidden,
        ];
        common::dispatch_threadgroups(
            &device,
            &pipeline,
            &bindings,
            MTLSize {
                width: 2,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    };
    if !run(0) {
        return;
    }
    assert_eq!(read::<u32>(&accepted, 2), [1, 0]);
    assert_eq!(read::<u32>(&ids, 6), [11, 12, MASK, MASK, 40, MASK]);
    let want: Vec<u16> = [row(0), row(1), row(9), row(9), row(3), row(9)].concat();
    assert_eq!(read::<u16>(&laid_hidden, want.len()), want);
    assert!(run(1));
    // `a`'s drafts from its newest (row 1), `b`'s from its (row 4).
    assert_eq!(read::<u32>(&drafts, 4), [71, 72, 74, 75]);
}
