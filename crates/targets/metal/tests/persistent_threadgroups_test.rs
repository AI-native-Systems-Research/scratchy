//! What the decode megakernel requires of the GPU: one 1024-thread threadgroup per core
//! (`MK_P` = the GPU's cores), all running at once — its grid barriers wait for every one.
//!
//! One launch of `P` such threadgroups, each holding ~32 KB of threadgroup memory as the
//! megakernel does (so a core holds one): each checks in, then waits (a bounded spin) until all
//! `P` have. `--nocapture` prints, per round, how many checked in and how long the first to
//! arrive waited for the last — a threadgroup that arrives LATE (another context held its core
//! a while) shows as a long wait; one that NEVER runs alongside the others leaves the count short.

use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary, MTLSize};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::device::gpu_cores;
use scratchy_target_metal::mtl4_dispatch::{dispatch_threadgroups, read_slice, shared_slice};

const PROBE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void probe(device atomic_uint* count [[buffer(0)]],
                  device uint* waited [[buffer(1)]],
                  device uint* order [[buffer(2)]],
                  constant uint* params [[buffer(3)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint held[8000];
  held[t] = t;
  if (t == 0) {
    order[tg] = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    uint s = 0;
    while (s < params[1] && atomic_load_explicit(count, memory_order_relaxed) < params[0]) {
      ++s;
    }
    waited[tg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (held[(t + 1u) % 1024u] == 0xffffffffu) {
    waited[tg] = 0u;
  }
}
"#;

/// Polls a threadgroup may spend waiting for the others (seconds on any GPU).
const LIMIT: u32 = 1 << 28;
const ROUNDS: usize = 5;

#[test]
fn every_persistent_threadgroup_runs_at_once() {
    let Some(dev) = detect_device().filter(|_| scratchy_target_metal::metal4_available()) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = dev.device;
    let Some(cores) = gpu_cores(&device) else {
        eprintln!("skipping: the GPU's core count is unknown");
        return;
    };
    let p = cores.get();
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(PROBE), Some(&opts))
        .expect("compile the probe");
    let func = library
        .newFunctionWithName(&NSString::from_str("probe"))
        .expect("probe");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline");
    let mut short = Vec::new();
    for round in 0..ROUNDS {
        let count = shared_slice(&device, &[0u32]);
        let waited = shared_slice(&device, &vec![0u32; p as usize]);
        let order = shared_slice(&device, &vec![0u32; p as usize]);
        let params = shared_slice(&device, &[p, LIMIT]);
        let started = std::time::Instant::now();
        assert!(dispatch_threadgroups(
            &device,
            &pipeline,
            &[&count, &waited, &order, &params],
            MTLSize {
                width: p as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        ));
        let elapsed = started.elapsed();
        let arrived = read_slice::<u32>(&count, 1)[0];
        let waited = read_slice::<u32>(&waited, p as usize);
        let order = read_slice::<u32>(&order, p as usize);
        let first = order.iter().position(|&o| o == 0).unwrap_or(0);
        println!(
            "round {round}: {arrived} of {p} threadgroups checked in; the first waited {} polls \
             for the last (limit {LIMIT}); launch took {elapsed:?}",
            waited[first],
        );
        if arrived < p {
            short.push((round, arrived));
        }
    }
    assert!(
        short.is_empty(),
        "{} rounds of {ROUNDS} ran fewer than all {p} threadgroups at once: {short:?}",
        short.len()
    );
}
