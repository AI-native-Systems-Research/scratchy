// SPDX-License-Identifier: Apache-2.0
//! W7 on the H100: the greedy argmax kernel (`scratchy-target-cuda`'s `kernels/argmax_bf16.cu`, the
//! one a one-row tape's graph runs after its last step) against the host's `plan::greedy_token`,
//! bit for bit through `plan::greedy_code`: random rows at Gemma's and the Qwens' vocab widths,
//! rows with ties (the lowest id wins), −0 against +0, ±inf, and rows with NaNs (the lowest NaN id
//! refused). And the launcher's refusals. Built only under `nvcc`; no GPU fails rather than skips.

use std::ffi::c_void;

use scratchy_target_cuda::kernels::{KernelError, argmax_bf16};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::ARGMAX_CHUNK;
use scratchy_target_cuda_worker::plan::{greedy_code, greedy_token};

/// Gemma 4's vocab and the Qwens'.
const VOCABS: [usize; 2] = [262_144, 248_320];

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// A splitmix64 stream, fixed by `seed`.
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `n` random logits in [−16, 16) as bf16.
fn random_row(n: usize, seed: u64) -> Vec<u16> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| bf16(((next() >> 40) as f32 / 16_777_216.0 - 0.5) * 32.0))
        .collect()
}

/// The device argmax's code for `row`.
fn device_code(device: &CudaDevice, iota: &DeviceBuffer, row: &[u16]) -> i32 {
    let r = DeviceBuffer::upload(device, row).expect("upload the row");
    let out = DeviceBuffer::upload_bytes(device, &[0xA5u8; 256]).expect("the output");
    // SAFETY: `r` holds `row.len()` bf16, `iota` ARGMAX_CHUNK f32, `out` 256 bytes, each its own
    // allocation; null is the default stream.
    unsafe {
        argmax_bf16(
            (r.ptr().cast_const(), row.len() as u32),
            iota.ptr().cast_const(),
            out.ptr(),
            std::ptr::null_mut(),
        )
    }
    .expect("the argmax");
    let bits = out.download().expect("download the output");
    i32::from_le_bytes([
        bits[0].to_le_bytes()[0],
        bits[0].to_le_bytes()[1],
        bits[1].to_le_bytes()[0],
        bits[1].to_le_bytes()[1],
    ])
}

fn iota(device: &CudaDevice) -> DeviceBuffer {
    let bytes: Vec<u8> = (0..ARGMAX_CHUNK)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    DeviceBuffer::upload_bytes(device, &bytes).expect("the iota")
}

/// Each case's device token against `greedy_token`, bit for bit: random rows, then planted ties,
/// signed zeros, infinities and NaNs at both vocab widths.
#[test]
fn the_device_argmax_is_greedy_token_bit_for_bit() {
    let device = CudaDevice::open().expect("an H100");
    let iota = iota(&device);
    let mut cases: Vec<(String, Vec<u16>)> = Vec::new();
    for (v, &n) in VOCABS.iter().enumerate() {
        for seed in 0..8u64 {
            cases.push((
                format!("random n={n} seed={seed}"),
                random_row(n, seed ^ (v as u64) << 32),
            ));
        }
        let mut next = stream(0x77 ^ n as u64);
        // A planted maximum at several ids, the lowest first or last.
        for k in [2usize, 3, 7] {
            let mut row = random_row(n, 100 + k as u64);
            let mut ids: Vec<usize> = (0..k).map(|_| (next() % n as u64) as usize).collect();
            ids.push(n - 1);
            for &id in &ids {
                row[id] = bf16(64.0);
            }
            cases.push((format!("{k}+1 ties n={n} at {ids:?}"), row));
        }
        cases.push((format!("all equal n={n}"), vec![bf16(1.5); n]));
        cases.push((format!("all -inf n={n}"), vec![bf16(f32::NEG_INFINITY); n]));
        let mut zeros = vec![bf16(-1.0); n];
        zeros[n / 3] = 0x8000; // −0
        zeros[n / 2] = 0x0000; // +0, equal to it: the lower id wins
        cases.push((format!("-0 before +0 n={n}"), zeros));
        let mut inf = random_row(n, 300);
        inf[n - 5] = bf16(f32::INFINITY);
        inf[n - 2] = bf16(f32::INFINITY);
        cases.push((format!("+inf twice n={n}"), inf));
        for (what, pattern) in [
            ("quiet", 0x7FC0u16),
            ("negative", 0xFFC1),
            ("signalling", 0x7F81),
        ] {
            let mut nan = random_row(n, 400);
            nan[n / 7] = bf16(1000.0);
            let first = (n / 5) | 1;
            nan[first] = pattern;
            nan[n - 1] = 0x7FC0;
            cases.push((format!("{what} NaN at {first} n={n}"), nan));
        }
    }
    let mut pass = true;
    for (what, row) in &cases {
        let code = device_code(&device, &iota, row);
        let (got, want) = (greedy_code(code), greedy_token(row));
        let ok = got == want;
        eprintln!(
            "W7 argmax {what}: device {got:?} host {want:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the device argmax differs from greedy_token: see the lines above"
    );
}

/// The launcher refuses a vocab of 0, off the chunk, or above 2^24, and a null pointer, writing
/// nothing.
#[test]
fn the_device_argmax_refuses_each_bad_row() {
    let device = CudaDevice::open().expect("an H100");
    let iota = iota(&device);
    let row = DeviceBuffer::upload(&device, &vec![0u16; 512]).expect("a row");
    let out = DeviceBuffer::upload_bytes(&device, &[0xA5u8; 256]).expect("the output");
    let null: *const c_void = std::ptr::null();
    let mut pass = true;
    for (what, vocab, row_ptr) in [
        ("vocab 0", 0, row.ptr().cast_const()),
        ("vocab 300", 300, row.ptr().cast_const()),
        ("vocab 2^24 + 256", (1 << 24) + 256, row.ptr().cast_const()),
        ("a null row", 512, null),
    ] {
        // SAFETY: refused before any pointer is touched.
        let r = unsafe {
            argmax_bf16(
                (row_ptr, vocab),
                iota.ptr().cast_const(),
                out.ptr(),
                std::ptr::null_mut(),
            )
        };
        let untouched = out
            .download()
            .expect("download")
            .iter()
            .all(|&b| b == 0xA5A5);
        let ok = matches!(r, Err(KernelError::RowShape { rows: 1, .. })) && untouched;
        eprintln!(
            "W7 argmax refusal {what}: {r:?} untouched={untouched} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(pass, "a bad row was not refused");
}
