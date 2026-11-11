// SPDX-License-Identifier: Apache-2.0
//! FP8 S8 W2c's probe (/tmp/d-w2c.md §3), on the H100: is the fp8 decode GEMV bound by the rate
//! of its TMA operations? `kernels/tma_ring_probe.cu` streams the 27B's gate+up (34,816 × 5,120)
//! and down (5,120 × 17,408) e4m3 weights through the GEMV's grid, warps and rings, with no dot
//! product, at five sizes of TMA operation: 256-byte vectors at 3 and at 4 rows a warp (today's),
//! 512-byte 2-row tiles, 4 KB 16-row tiles and 16 KB 16 × 1,024 tiles per 4 warps.
//!
//! Each variant is timed by CUDA events, cold (L2 cleared before each launch) and hot, in two
//! rounds, and printed as `PROBE <variant> <shape> cold|hot <TB/s>` of the weight bytes. Recorded,
//! not asserted, but for a check that every variant reads the weight and not the TMA's zero fill:
//! every byte is 1.0 (e4m3 0x38), and lane `l` of each row's warp folds byte `l` of each 256
//! columns, so the folds of a variant add up to exactly `n · (k / 256) · 32`.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.

pub mod common {
    pub mod watchdog;
}

use std::ffi::c_void;

use scratchy_target_cuda::kernels::tma_ring_probe;
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// `cudaEvent_t`, declared here for the timing only (the runtime declares no event call).
type CudaEvent = *mut c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
    fn cudaMemsetAsync(dst: *mut c_void, value: i32, bytes: usize, stream: *mut c_void) -> i32;
}

/// The bytes cleared before a cold run: twice the H100's 50 MiB L2 (`tests/gemv_fp8.rs`'s).
const SCRUB_BYTES: usize = 128 << 20;

/// The variants, by the launcher's number.
const VARIANTS: &[(u32, &str)] = &[
    (0, "vec256_r3"),
    (1, "vec256"),
    (2, "pair512"),
    (3, "tile4k"),
    (4, "page16k"),
];

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA
/// events on the default stream; `scrub` (when given) is cleared before each
/// (`tests/gemv_fp8.rs`'s `event_times`).
fn event_times(scrub: Option<&DeviceBuffer>, launch: impl FnMut()) -> (f64, f64, f64) {
    let mut i = 0;
    timed(
        || {
            if let Some(s) = scrub {
                // SAFETY: `s` is a live allocation of `SCRUB_BYTES`; the default stream orders it
                // before the timed launch.
                let set = unsafe { cudaMemsetAsync(s.ptr(), i, SCRUB_BYTES, std::ptr::null_mut()) };
                assert_eq!(set, 0, "cudaMemsetAsync");
            }
            i += 1;
        },
        launch,
    )
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA
/// events on the default stream, `before` run (untimed, on the same stream) before each.
fn timed(mut before: impl FnMut(), mut launch: impl FnMut()) -> (f64, f64, f64) {
    let (mut start, mut end): (CudaEvent, CudaEvent) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: plain runtime calls on the default stream with out-pointers to locals; each event is
    // destroyed below.
    unsafe {
        assert_eq!(cudaEventCreate(&mut start), 0, "cudaEventCreate");
        assert_eq!(cudaEventCreate(&mut end), 0, "cudaEventCreate");
    }
    let mut times = Vec::new();
    for i in 0..23 {
        let mut ms = 0.0f32;
        before();
        // SAFETY: as above.
        unsafe {
            assert_eq!(cudaEventRecord(start, std::ptr::null_mut()), 0, "record");
            launch();
            assert_eq!(cudaEventRecord(end, std::ptr::null_mut()), 0, "record");
            assert_eq!(cudaEventSynchronize(end), 0, "synchronize");
            assert_eq!(cudaEventElapsedTime(&mut ms, start, end), 0, "elapsed");
        }
        if i >= 3 {
            times.push(f64::from(ms) * 1e3);
        }
    }
    // SAFETY: both events were created above and are no longer used.
    unsafe {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// The sum of the f32 folds in `out`, in f64.
fn fold_sum(out: &DeviceBuffer) -> f64 {
    let bytes: Vec<u8> = (out.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect();
    (bytes.as_chunks::<4>().0.iter())
        .map(|&c| f64::from(f32::from_le_bytes(c)))
        .sum()
}

/// The probe at the 27B's gate+up and down: every variant cold and hot, two rounds, one
/// `PROBE` line each; then each shape's variants must have folded the same bytes.
#[test]
fn the_tma_ring_probe_bandwidth_is_recorded() {
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    let mut same = true;
    for (n, k, shape) in [
        (34816u32, 5120u32, "gate+up_34816x5120"),
        (5120, 17408, "down_5120x17408"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        // e4m3 0x38 is 1.0.
        let b = DeviceBuffer::upload_bytes(&device, &vec![0x38u8; nu * ku]).expect("b");
        let want = (nu * (ku / 256) * 32) as f64;
        let mut sums = Vec::new();
        for round in 1..=2 {
            for &(v, name) in VARIANTS {
                let out = DeviceBuffer::upload_bytes(&device, &vec![0u8; (16 * nu + 256) * 4])
                    .expect("out");
                for (temp, sc) in [("cold", Some(&scrub)), ("hot", None)] {
                    let (med, lo, hi) = event_times(sc, || {
                        // SAFETY: `b` holds n·k bytes and `out` 16·n + 256 f32, two cudaMalloc'd
                        // allocations; null is the default stream, which the events wait on.
                        unsafe {
                            tma_ring_probe(
                                (b.ptr().cast_const(), out.ptr()),
                                (n, k),
                                v,
                                std::ptr::null_mut(),
                            )
                        }
                        .expect("tma_ring_probe launch");
                    });
                    eprintln!(
                        "PROBE {name} {shape} {temp} {:.2} (round {round}: median={med:.1}us \
                         min={lo:.1}us max={hi:.1}us)",
                        (nu * ku) as f64 / med / 1e6
                    );
                }
                sums.push((name, fold_sum(&out)));
            }
        }
        let ok = sums.iter().all(|&(_, s)| s == want);
        eprintln!(
            "PROBE_FOLDS {shape}: want={want} {sums:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        same &= ok;
    }
    assert!(
        same,
        "a ring probe variant did not read every byte of its weight"
    );
}

// ── FP8 S8 W2d: the consume probe and the read-flush cold ──────────

/// The consume variants, by the launcher's number (/tmp/d-w2d.md §1's (a) to (e)).
const CONSUME: &[(u32, &str)] = &[
    (0, "a_wait"),
    (1, "b_bytes"),
    (2, "c_widen"),
    (3, "d_dot"),
    (4, "e_full"),
];

/// The read flush's buffer: 100 MiB, twice the H100's L2, as `[20480, 5120]` e4m3 for the ring
/// probe's `vec256` to stream.
const FLUSH_N: u32 = 20480;
const FLUSH_K: u32 = 5120;

/// A buffer of `len` copies of `byte`.
fn filled(device: &CudaDevice, len: usize, byte: u8) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![byte; len]).expect("upload")
}

/// The wrapping sum of the i32 folds in `out`, and their sum as i64.
fn int_folds(out: &DeviceBuffer) -> (u32, i64) {
    let bytes: Vec<u8> = (out.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let v: Vec<i32> = (bytes.as_chunks::<4>().0.iter())
        .map(|&c| i32::from_le_bytes(c))
        .collect();
    (
        v.iter().fold(0u32, |a, &x| a.wrapping_add(x as u32)),
        v.iter().map(|&x| i64::from(x)).sum(),
    )
}

/// The three ways a run starts: L2 cleared by a 128 MiB memset (`cold`, the earlier tests'), by
/// a 100 MiB read (`readcold`: L2 left holding clean lines, as in the decode tape), or not at all
/// (`hot`). Each returns `(median, min, max)` µs of `launch`.
fn temps(
    scrub: &DeviceBuffer,
    flush: &(DeviceBuffer, DeviceBuffer),
    mut launch: impl FnMut(),
) -> [(&'static str, (f64, f64, f64)); 3] {
    let cold = event_times(Some(scrub), &mut launch);
    let readcold = timed(
        || {
            // SAFETY: `flush.0` holds FLUSH_N · FLUSH_K bytes and `flush.1` 16 · FLUSH_N + 256
            // f32, two allocations; the default stream orders it before the timed launch.
            unsafe {
                tma_ring_probe(
                    (flush.0.ptr().cast_const(), flush.1.ptr()),
                    (FLUSH_N, FLUSH_K),
                    1,
                    std::ptr::null_mut(),
                )
            }
            .expect("read flush");
        },
        &mut launch,
    );
    let hot = event_times(None, &mut launch);
    [("cold", cold), ("readcold", readcold), ("hot", hot)]
}

/// FP8 S8 W2d's discriminating probe (/tmp/d-w2d.md §1, §2.1): today's load (256-byte vector
/// boxes, a 4-stage ring per warp) with the consumer's work added one cost at a time, (a) the
/// handshake, (b) every byte read, (c) the widen, (d) the activation row and the FMA, (e) the
/// quantizing prologue, at 3 and 4 rows a warp and exactly 2 and 3 CTAs per SM, cold, read-flush
/// cold and hot, two rounds, as `PROBE2 <variant> <ctas> <shape> cold|readcold|hot <TB/s>`; the
/// real fp8 GEMV (one launch at 34,816 × 5,120, the fused 2 × 17,408 gate+up, the down) the
/// same way. Shapes: the 27B gate+up (34,816 × 5,120), its GDN qkv+z group's bytes as one stream
/// (16,384 × 5,120) and its down (5,120 × 17,408), the last at 2 CTAs only (the 3-CTA form holds a
/// 32-chunk row, `k` at most 8,192). Recorded, not asserted, but for the folds: with every weight
/// byte 1.0 (e4m3 0x38) and the activation 1.0, each variant's folds add up to an exact count
/// that proves it read every byte.
#[test]
fn the_tma_consume_probe_bandwidth_is_recorded() {
    use scratchy_target_cuda::kernels::{
        gate_up_gemv_fp8, gemv_fp8_abt, tma_consume_probe, tma_probe_occupancy,
    };
    use scratchy_target_cuda::step::{gemv_scratch_bytes, gemv_splits};
    let device = h100();
    let scrub = filled(&device, SCRUB_BYTES, 0);
    let flush_n = FLUSH_N as usize;
    let flush = (
        filled(&device, flush_n * FLUSH_K as usize, 0x38),
        filled(&device, (16 * flush_n + 256) * 4, 0),
    );
    for &(c, name) in CONSUME {
        for rows in [3u32, 4] {
            for ctas in [2u32, 3] {
                let occ = tma_probe_occupancy(0, Some((c, rows, ctas))).expect("occupancy");
                eprintln!(
                    "PROBE2_OCC {name}_r{rows} {ctas}: {occ} CTAs per SM {}",
                    if occ == ctas as i32 { "PASS" } else { "FAIL" }
                );
            }
        }
    }
    // The ring probe pads its shared memory to the GEMV's 68,608 bytes (static memory aside), so its
    // occupancy is the GEMV's by shared memory: recorded, as `__launch_bounds__(256, 2)` only sets a
    // floor, and three such CTAs fit an SM's 228 KB.
    for &(v, name) in VARIANTS {
        let occ = tma_probe_occupancy(v, None).expect("occupancy");
        eprintln!("PROBE2_OCC ring {name} (68,608 B like the GEMV): {occ} CTAs per SM");
    }
    let mut pass = true;
    for (n, k, shape) in [
        (34816u32, 5120u32, "gate+up_34816x5120"),
        (16384, 5120, "gdn_group_16384x5120"),
        (5120, 17408, "down_5120x17408"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        let b = filled(&device, nu * ku, 0x38);
        // bf16 1.0, every element.
        let a = DeviceBuffer::upload(&device, &vec![0x3F80u16; ku]).expect("a");
        for round in 1..=2 {
            for &(c, name) in CONSUME {
                for rows in [3u32, 4] {
                    for ctas in [2u32, 3] {
                        if ctas == 3 && k > 8192 {
                            if round == 1 && rows == 3 {
                                eprintln!(
                                    "PROBE2_SKIP {name} 3 {shape}: not run, the 3-CTA form holds k <= 8192"
                                );
                            }
                            continue;
                        }
                        let out = filled(&device, (11 * nu + 256) * 4, 0);
                        let launch = || {
                            // SAFETY: `b` holds n·k bytes, `a` k bf16 and `out` 11·n + 256 i32,
                            // three allocations; null is the default stream.
                            unsafe {
                                tma_consume_probe(
                                    (b.ptr().cast_const(), a.ptr().cast_const(), out.ptr()),
                                    (n, k),
                                    (c, rows, ctas),
                                    std::ptr::null_mut(),
                                )
                            }
                            .expect("tma_consume_probe launch");
                        };
                        for (temp, (med, lo, hi)) in temps(&scrub, &flush, launch) {
                            eprintln!(
                                "PROBE2 {name}_r{rows} {ctas} {shape} {temp} {:.2} (round {round}: \
                                 median={med:.1}us min={lo:.1}us max={hi:.1}us)",
                                (nu * ku) as f64 / med / 1e6
                            );
                        }
                        let (wrapped, total) = int_folds(&out);
                        let elems = (nu * ku) as i64;
                        let ok = match c {
                            0 => total == elems / 256 * 32,
                            1 => wrapped == ((elems / 4) as u64).wrapping_mul(0x3838_3838) as u32,
                            4 => total == 32 * elems * 448,
                            _ => total == 32 * elems,
                        };
                        if !ok {
                            eprintln!(
                                "PROBE2_FOLDS {name}_r{rows} {ctas} {shape}: total={total} \
                                 wrapped={wrapped:#x} FAIL"
                            );
                        }
                        pass &= ok;
                    }
                }
            }
        }
        eprintln!(
            "PROBE2_FOLDS {shape}: {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }
    // Today's real fp8 GEMV, the same three ways.
    for (n, k, what) in [
        (34816u32, 5120u32, "gate+up_34816x5120"),
        (5120, 17408, "down_5120x17408"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        let x = DeviceBuffer::upload(&device, &vec![0x3F80u16; ku]).expect("x");
        let w = filled(&device, nu * ku, 0x38);
        let s = DeviceBuffer::upload_bytes(&device, &vec![0u8; (nu * 4).next_multiple_of(256)])
            .expect("scales");
        let c = DeviceBuffer::upload(&device, &vec![0u16; nu]).expect("c");
        let splits = gemv_splits(n, k, 1);
        let p = filled(
            &device,
            (gemv_scratch_bytes(n, 1, splits).expect("bytes") as usize).max(256),
            0,
        );
        for round in 1..=2 {
            let launch = || {
                // SAFETY: as `tests/gemv_fp8.rs`'s `run_gemv`: `x` k bf16, `w` n·k bytes, `s` n f32,
                // `c` n bf16, `p` the split's partials, all their own allocations.
                unsafe {
                    gemv_fp8_abt(
                        (
                            x.ptr().cast_const(),
                            (w.ptr().cast_const(), s.ptr().cast_const()),
                            c.ptr(),
                        ),
                        p.ptr(),
                        (n, k, splits),
                        std::ptr::null_mut(),
                    )
                }
                .expect("gemv_fp8_abt launch");
            };
            for (temp, (med, lo, hi)) in temps(&scrub, &flush, launch) {
                eprintln!(
                    "PROBE2 gemv_fp8 2 {what} {temp} {:.2} (round {round}: median={med:.1}us min={lo:.1}us \
                     max={hi:.1}us)",
                    (nu * ku) as f64 / med / 1e6
                );
            }
        }
    }
    let (n, k) = (17408u32, 5120u32);
    let (nu, ku) = (n as usize, k as usize);
    let x = DeviceBuffer::upload(&device, &vec![0x3F80u16; ku]).expect("x");
    let (gw, uw) = (
        filled(&device, nu * ku, 0x38),
        filled(&device, nu * ku, 0x38),
    );
    let s = DeviceBuffer::upload_bytes(&device, &vec![0u8; (nu * 4).next_multiple_of(256)])
        .expect("scales");
    let c = DeviceBuffer::upload(&device, &vec![0u16; nu]).expect("c");
    let splits = gemv_splits(n, k, 1);
    let p = filled(
        &device,
        (2 * gemv_scratch_bytes(n, 1, splits).expect("bytes") as usize).max(256),
        0,
    );
    for round in 1..=2 {
        let launch = || {
            // SAFETY: as `tests/gemv_fp8.rs`'s gate/up test, for both weights and their partials.
            unsafe {
                gate_up_gemv_fp8(
                    (
                        x.ptr().cast_const(),
                        (gw.ptr().cast_const(), s.ptr().cast_const()),
                        (uw.ptr().cast_const(), s.ptr().cast_const()),
                    ),
                    (c.ptr(), p.ptr()),
                    (n, k, splits),
                    0,
                    std::ptr::null_mut(),
                )
            }
            .expect("gate_up_gemv_fp8 launch");
        };
        for (temp, (med, lo, hi)) in temps(&scrub, &flush, launch) {
            eprintln!(
                "PROBE2 gemv_fp8 2 gate+up_fused_2x17408x5120 {temp} {:.2} (round {round}: \
                 median={med:.1}us min={lo:.1}us max={hi:.1}us)",
                (2 * nu * ku) as f64 / med / 1e6
            );
        }
    }
    assert!(
        pass,
        "a consume probe variant did not read every byte of its weight"
    );
}
