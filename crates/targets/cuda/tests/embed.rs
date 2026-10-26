// SPDX-License-Identifier: Apache-2.0
//! The token-embedding gather (`kernels/embed_bf16.cu`) against an exact reference, on the H100,
//! and its launcher's refusals.
//!
//! A gather is a copy, so the reference is the table's own rows and the check is bit for bit.
//! Every output row starts as a bf16 NaN, so a row the kernel skips (a padding row included)
//! fails; ids include 0, `vocab − 1` and repeats. Built only under `nvcc` (`required-features`).
//! No GPU fails rather than skips. Every case prints its line before a test fails.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{bf16_inputs, mismatched_bits};
use scratchy_target_cuda::kernels::{KernelError, embed_bf16};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// Every output element before a launch: a bf16 NaN, so an element the kernel never writes fails.
const SENTINEL: u16 = 0xFFFF;
/// Extra output elements past `rows·hidden`, which the kernel must leave untouched.
const TAIL: usize = 4096;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// The ids as the bytes `DeviceBuffer::upload_bytes` takes.
fn id_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

/// Gather `ids` from a `[vocab, hidden]` table over a NaN-poisoned output: the output's
/// `rows·hidden` elements and whether the tail past them held.
fn run(
    device: &CudaDevice,
    table: &[u16],
    ids: &[u32],
    hidden: usize,
    vocab: u32,
) -> (Vec<u16>, bool) {
    let rows = ids.len();
    let t = DeviceBuffer::upload(device, table).expect("upload the table");
    let i = DeviceBuffer::upload_bytes(device, &id_bytes(ids)).expect("upload the ids");
    let out = DeviceBuffer::upload(device, &vec![SENTINEL; rows * hidden + TAIL]).expect("out");
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `ids` holds `rows` u32, `table`
    // `vocab·hidden` bf16 and `out` `rows·hidden` (and the tail); `out` is its own allocation;
    // null is the default stream.
    unsafe {
        embed_bf16(
            i.ptr(),
            t.ptr(),
            out.ptr(),
            (rows as u32, hidden as u32, vocab),
            std::ptr::null_mut(),
        )
    }
    .expect("embed_bf16 launch");
    let mut bits = out.download().expect("download the output");
    let tail = bits[rows * hidden..].iter().all(|&b| b == SENTINEL);
    bits.truncate(rows * hidden);
    (bits, tail)
}

/// The gather on the host: row `r` is the table's row `ids[r]`.
fn reference(table: &[u16], ids: &[u32], hidden: usize) -> Vec<u16> {
    (ids.iter())
        .flat_map(|&id| table[id as usize * hidden..][..hidden].iter().copied())
        .collect()
}

#[test]
fn the_gather_is_bit_exact_on_every_row() {
    let device = h100();
    let mut pass = true;
    // (vocab, hidden, rows): qwen3.5-0.8b's table and qwen3.5-9b's (2 GB) at a 512-row bucket, a
    // narrow table, one row.
    for (vocab, hidden, rows) in [
        (248_320u32, 1024usize, 512usize),
        (248_320, 4096, 512),
        (37, 8, 300),
        (5, 64, 1),
    ] {
        let table = bf16_inputs(vocab as usize * hidden, 0x0816_0000 ^ u64::from(vocab));
        // 0, vocab − 1, a repeated id and a spread of the rest; the padding rows id 0, as the
        // caller pads.
        let mut ids: Vec<u32> = (0..rows as u32)
            .map(|r| r.wrapping_mul(2_654_435_761) % vocab)
            .collect();
        ids[0] = 0;
        if rows > 2 {
            ids[1] = vocab - 1;
            ids[2] = ids[1];
        }
        let (gpu, tail) = run(&device, &table, &ids, hidden, vocab);
        let want = reference(&table, &ids, hidden);
        let mismatched = mismatched_bits(&gpu, &want);
        let unwritten = (gpu.iter().zip(&want))
            .filter(|&(&g, &w)| g == SENTINEL && w != SENTINEL)
            .count();
        let ok = mismatched == 0 && tail;
        eprintln!(
            "embed_bf16 vocab={vocab} hidden={hidden} rows={rows}: mismatched_bits={mismatched} \
             unwritten_poison={unwritten} tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an embed gather is not bit-exact");
}

/// An id at or above `vocab` writes a `+0.0` row and reads nothing; the rows around it are
/// gathered as usual.
#[test]
fn an_id_outside_the_vocabulary_writes_a_zero_row() {
    let device = h100();
    let (vocab, hidden) = (16u32, 64usize);
    let table = bf16_inputs(vocab as usize * hidden, 0x0816_00FF);
    let ids = [3, vocab, 7, u32::MAX];
    let (gpu, tail) = run(&device, &table, &ids, hidden, vocab);
    let mut want = Vec::new();
    for &id in &ids {
        match id < vocab {
            true => want.extend_from_slice(&table[id as usize * hidden..][..hidden]),
            false => want.extend(std::iter::repeat_n(0u16, hidden)),
        }
    }
    let mismatched = mismatched_bits(&gpu, &want);
    let ok = mismatched == 0 && tail;
    eprintln!(
        "embed_bf16 ids {ids:?} over vocab={vocab}: mismatched_bits={mismatched} \
         tail_intact={tail} {}",
        verdict(ok)
    );
    assert!(ok, "an out-of-vocabulary id did not write a +0.0 row");
}

/// The shapes the launcher must refuse with `EmbedShape`, writing nothing: a zero dimension, a
/// `hidden` off the 8-wide vector, a dimension above `INT_MAX`, and `rows · hidden` above
/// `INT_MAX`. Refused before any pointer is touched, so the buffers are small.
#[test]
fn the_launcher_refuses_a_shape_it_cannot_run() {
    let device = h100();
    let refused: [(u32, u32, u32); 8] = [
        (0, 64, 16),
        (4, 0, 16),
        (4, 64, 0),
        (4, 60, 16),
        (1 << 31, 8, 16),
        (4, 1 << 31, 16),
        (4, 64, 1 << 31),
        (65_536, 65_536, 16),
    ];
    let table = DeviceBuffer::upload(&device, &vec![0; 4096]).expect("upload a table");
    let ids = DeviceBuffer::upload_bytes(&device, &[0; 64]).expect("upload ids");
    let mut pass = true;
    for (rows, hidden, vocab) in refused {
        let out = DeviceBuffer::upload(&device, &vec![SENTINEL; 4096]).expect("upload out");
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            embed_bf16(
                ids.ptr(),
                table.ptr(),
                out.ptr(),
                (rows, hidden, vocab),
                std::ptr::null_mut(),
            )
        };
        let unchanged = (out.download().expect("download out"))
            .iter()
            .all(|&b| b == SENTINEL);
        let ok =
            got == Err(KernelError::EmbedShape {
                rows,
                hidden,
                vocab,
            }) && unchanged;
        eprintln!(
            "embed_bf16 rows={rows} hidden={hidden} vocab={vocab}: {got:?} \
             out_unchanged={unchanged} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the embed launcher did not refuse a shape it cannot run"
    );
}

/// The same gather twice gives the same bits, over qwen3.5-9b's table width at a 512-row bucket:
/// the bulk copies are bytes, so nothing about the launch can change them.
#[test]
fn the_gather_repeats_bit_for_bit() {
    let device = h100();
    let (vocab, hidden, rows) = (4096u32, 4096usize, 512usize);
    let table = bf16_inputs(vocab as usize * hidden, 0x0816_2000);
    let ids: Vec<u32> = (0..rows as u32)
        .map(|r| r.wrapping_mul(2_654_435_761) % vocab)
        .collect();
    let (a, a_tail) = run(&device, &table, &ids, hidden, vocab);
    let (b, b_tail) = run(&device, &table, &ids, hidden, vocab);
    let mismatched = mismatched_bits(&a, &b);
    let ok = mismatched == 0 && a_tail && b_tail;
    eprintln!(
        "embed_bf16 vocab={vocab} hidden={hidden} rows={rows} repeat \
         mismatched_bits={mismatched} tail_intact={} {}",
        a_tail && b_tail,
        verdict(ok)
    );
    assert!(ok, "the embed gather did not repeat bit for bit");
}
