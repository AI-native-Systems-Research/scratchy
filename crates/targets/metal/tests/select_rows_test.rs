// SPDX-License-Identifier: Apache-2.0
//! `select_rows`: each op copies the variant its selector names — variants a stride apart,
//! overlapping where the stride is shorter than the copy — to its destination, at its offset; ops of
//! one dispatch pick independently. GPU test — run with `--test-threads=1` (standing rule).

mod common;

use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{read_slice, shared_slice, shared_zeroed};
use scratchy_target_metal::select_rows::{SelectInto, SelectOp, Selection};
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
    let (dst_a, dst_b) = (shared_zeroed(&device, 64), shared_zeroed(&device, 64));
    // Three variants of four words; then a window table of five two-word rows, three rows a copy.
    let variants: Vec<u32> = (0..12).map(|i| 100 + i).collect();
    let window: Vec<u32> = (0..10).map(|i| 200 + i).collect();
    let selections = [
        Selection {
            table: variants,
            stride: 4,
            len: 4,
            selector: (selectors.clone(), 0),
            to: SelectInto::Buffer(dst_a.clone()),
            at: 0,
        },
        Selection {
            table: window,
            stride: 2,
            len: 6,
            selector: (selectors.clone(), 4),
            to: SelectInto::Buffer(dst_b.clone()),
            at: 8,
        },
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
        region.extend(s.table.iter().flat_map(|w| w.to_le_bytes()));
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
    let bufs = [&region, &region, &selectors, &dst_a, &dst_b];
    if !common::dispatch_threadgroups(&device, &pso, &bufs, size(ops.len()), size(64)) {
        return;
    }
    assert_eq!(
        read_slice::<u32>(&dst_a, 4),
        [108, 109, 110, 111],
        "variant 2 of 3"
    );
    let b: Vec<u32> = read_slice::<u32>(&dst_b, 8);
    assert_eq!(b[..2], [0, 0], "nothing before the op's offset");
    assert_eq!(
        b[2..],
        [202, 203, 204, 205, 206, 207],
        "rows 1-3 of the window"
    );
}
