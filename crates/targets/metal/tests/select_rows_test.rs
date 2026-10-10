// SPDX-License-Identifier: Apache-2.0
//! `select_rows`: each op copies the variant its selector names — variants a stride apart,
//! overlapping where the stride is shorter than the copy — from a staged table or a device buffer,
//! to its destination at its offset; ops of one dispatch pick independently. GPU test — run with
//! `--test-threads=1` (standing rule).

mod common;

use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{read_slice, shared_slice, shared_zeroed};
use scratchy_target_metal::select_rows::{SelectFrom, SelectInto, SelectOp, Selection};
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};

#[test]
fn select_rows_copies_the_variant_each_selector_names() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let pso = baked_build(
        &cache,
        &PipelineKey::new("select_rows", "select_rows", Vec::new()),
    )
    .expect("select_rows");

    // Selectors, as an earlier command buffer leaves them: sequence 0 kept 2, sequence 1 kept 1.
    let selectors = shared_slice(&device, &[2u32, 1]);
    // An earlier step's outputs: its rows' tokens, and its drafts.
    let tokens = shared_slice(&device, &[300u32, 301, 302, 303]);
    let drafts = shared_slice(&device, &[400u32, 401]);
    let dsts: Vec<_> = (0..4).map(|_| shared_zeroed(&device, 64)).collect();
    let selection = |source, stride, len, selector, to: usize, at| Selection {
        source,
        stride,
        len,
        selector: (selectors.clone(), selector),
        to: SelectInto::Buffer(dsts[to].clone()),
        at,
    };
    let selections = [
        // Three variants of four words.
        selection(SelectFrom::Table((100..112).collect()), 4, 4, 0, 0, 0),
        // A window over five two-word rows, three rows a copy, 8 bytes into its destination.
        selection(SelectFrom::Table((200..210).collect()), 2, 6, 4, 1, 8),
        // The token at the row sequence 0 kept up to, from row 1 on.
        selection(SelectFrom::Device(tokens.clone(), 4), 1, 1, 0, 2, 0),
        // The drafts whole: stride 0.
        selection(SelectFrom::Device(drafts.clone(), 0), 0, 2, 4, 3, 4),
    ];
    // The staged region: the ops, then each table.
    let ops_bytes = selections.len() * size_of::<SelectOp>();
    let mut region = vec![0u8; ops_bytes];
    let mut ops = Vec::new();
    for s in &selections {
        let SelectInto::Buffer(to) = &s.to else {
            unreachable!("the test's selections land in buffers");
        };
        let src_at = region.len();
        if let SelectFrom::Table(table) = &s.source {
            region.extend(table.iter().flat_map(|w| w.to_le_bytes()));
        }
        ops.push(SelectOp::new(src_at, s, to.gpuAddress() + s.at as u64));
    }
    for (i, op) in ops.iter().enumerate() {
        let at = i * size_of::<SelectOp>();
        region[at..at + size_of::<SelectOp>()].copy_from_slice(&op.bytes());
    }
    let region = shared_slice(&device, &region);
    let size = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    // Bound past the kernel's two so the buffers its ops address are resident.
    let mut bufs = vec![&region, &region, &selectors, &tokens, &drafts];
    bufs.extend(dsts.iter());
    if !common::dispatch_threadgroups(&device, &pso, &bufs, size(ops.len()), size(64)) {
        return;
    }
    let read = |i: usize, n| read_slice::<u32>(&dsts[i], n);
    assert_eq!(read(0, 4), [108, 109, 110, 111], "variant 2 of 3");
    assert_eq!(
        read(1, 8),
        [0, 0, 202, 203, 204, 205, 206, 207],
        "rows 1-3 of the window, past the op's offset"
    );
    assert_eq!(read(2, 1), [303], "row 1 + 2 of the earlier step's tokens");
    assert_eq!(
        read(3, 3),
        [0, 400, 401],
        "the drafts, past the op's offset"
    );
}
