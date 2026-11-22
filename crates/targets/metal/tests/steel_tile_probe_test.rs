// SPDX-License-Identifier: Apache-2.0
//! The classic simdgroup steel paged attention's key tile (`SteelTile`'s BK), timed: a 2048-token
//! prompt's prefill at each head dim the kernel takes, 16 keys a step against 32, interleaved
//! rounds. A probe, not a gate: `--ignored --nocapture`.
//!
//! GPU test — run with `--test-threads=1` (standing rule).

mod common;

use half::f16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

fn buf<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(16);
    let b = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            b.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        )
    };
    b
}

fn halves(n: usize, seed: u32) -> Vec<f16> {
    (0..n)
        .map(|i| {
            f16::from_f32(
                (((i as u32).wrapping_mul(2654435761) ^ seed) % 2001) as f32 / 1000.0 - 1.0,
            )
        })
        .collect()
}

#[test]
#[ignore = "a timing probe: --ignored --nocapture"]
fn steel_key_tile_timings() {
    let Some(di) = detect_device() else {
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (nq, nkv, bs, m) = (16usize, 2usize, 16usize, 2048usize);
    let blocks = m / bs;
    for hd in [64usize, 96, 128, 256] {
        let elems = blocks * nkv * bs * hd;
        let (k, v) = (
            buf(&device, &halves(elems, 1)),
            buf(&device, &halves(elems, 2)),
        );
        let (kt, vt) = (
            buf(&device, &[k.gpuAddress()]),
            buf(&device, &[v.gpuAddress()]),
        );
        let bt = buf(&device, &(0..blocks as u32).collect::<Vec<_>>());
        let q = buf(&device, &halves(m * nq * hd, 3));
        let out = buf(&device, &vec![f16::ZERO; m * nq * hd]);
        let cu = buf(&device, &[0u32, m as u32]);
        let used = buf(&device, &[m as u32]);
        let mut times = std::collections::BTreeMap::<u32, Vec<f64>>::new();
        let pipes: Vec<_> = [16u32, 32]
            .into_iter()
            .filter_map(|bk| {
                let consts = vec![
                    ConstantValue::uint(0, hd as u32),
                    ConstantValue::uint(1, nq as u32),
                    ConstantValue::uint(2, nkv as u32),
                    ConstantValue::float(3, 1.0 / (hd as f32).sqrt()),
                    ConstantValue::uint(4, bs as u32),
                    ConstantValue::uint(5, blocks as u32),
                    ConstantValue::uint(6, 4096),
                    ConstantValue::int(7, 0),
                    ConstantValue::uint(99, 0),
                    ConstantValue::uint(30, 32),
                    ConstantValue::uint(31, bk),
                    ConstantValue::uint(32, 4),
                    ConstantValue::uint(33, 1),
                ];
                let key =
                    PipelineKey::new("attention_steel_paged", "attention_steel_paged_f16", consts);
                match baked_build(&cache, &key) {
                    Ok(p) => Some((bk, p)),
                    Err(e) => {
                        eprintln!("hd {hd} bk {bk}: no pipeline ({e})");
                        None
                    }
                }
            })
            .collect();
        let bufs = [&out, &q, &cu, &used, &bt, &kt, &vt, &k, &v];
        let grid = MTLSize {
            width: m.div_ceil(32),
            height: nq,
            depth: 1,
        };
        let threads = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        for round in 0..6 {
            for (bk, p) in &pipes {
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    common::dispatch_threadgroups(&device, p, &bufs, grid, threads);
                }
                if round > 0 {
                    times
                        .entry(*bk)
                        .or_default()
                        .push(t.elapsed().as_secs_f64() / 5.0);
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2] * 1e3
        };
        let line: Vec<String> = (times.iter_mut())
            .map(|(bk, v)| format!("bk{bk} {:.3} ms", median(v)))
            .collect();
        eprintln!("steel paged hd{hd} m{m}: {}", line.join(", "));
    }
}
