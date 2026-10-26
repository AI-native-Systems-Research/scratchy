// SPDX-License-Identifier: Apache-2.0
//! The prefill attention's whole output, per case, dumped for a bit-identity comparison across
//! two commits (AI-native-Systems-Research/scratchy#845): the same file is built and run at the
//! base and at the change, and the two dumps are compared byte for byte outside this test.
//!
//! Each case writes its full `[rows, q_heads · 256]` bf16 output, padding rows included, as raw
//! little-endian bytes to `<CARGO_TARGET_TMPDIR>/attn_prefill_dump/<case>.bin`, and prints `case=
//! elems= nonfinite= fnv1a=`. The only assertion is that every launch returned 0: what the output
//! should be is `attention.rs`'s business, unchanged.
//!
//! The caches are built on the host and uploaded, not written by `kv_append_bf16`, so a dump
//! depends on the prefill kernel alone: every written slot holds its key or value, placed through
//! `PagedKvLayout` and `slot_of` over a permuted block table, and every other element of the pool
//! is the bf16 NaN `0x7FC1`. The cases, with fixed seeds: `attention.rs`'s six random cases; its
//! four chunk-against-one-forward sequences, each as the one forward and as the chunk; its GQA
//! probe at 8/2 and 16/4 heads (K = 0, kv head h's V the constant h + 1); and one row over a
//! partly filled first block.
//!
//! Uses only what `c8d80fda` already has (`kernels::attn_prefill_bf16`, `runtime`,
//! `PagedKvLayout`, `common::numerics`), so it builds unchanged at that base. Built only under
//! `nvcc` (`required-features`), and needs an sm_90 GPU: no GPU fails rather than skips.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use std::fs;
use std::path::PathBuf;

use common::numerics::{bf16_inputs, slot_of};
use half::bf16;
use scratchy_subtile::paged_kv_layout::PagedKvLayout;
use scratchy_target_cuda::kernels::attn_prefill_bf16;
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// qwen3.5's head dim, and the KV block size.
const HD: usize = 256;
const BS: usize = 64;
/// The pool's blocks.
const BLOCKS: usize = 16;
const SCALE: f32 = 0.0625;
/// A permuted block table over the pool: logical block `j` is physical block `TABLE[j]`.
const TABLE: [u32; 8] = [11, 3, 14, 0, 7, 9, 2, 5];
/// An output element no launch wrote: a bf16 NaN.
const SENTINEL: u16 = 0xFFFF;
/// A cache element no key or value was written to: a bf16 NaN.
const POISON: u16 = 0x7FC1;

/// `attention.rs`'s random cases, `(q_heads, kv_heads, rows, P, L)`, with its seeds.
const CASES: [(usize, usize, usize, usize, usize); 6] = [
    (8, 2, 128, 0, 128),
    (8, 2, 256, 0, 256),
    (8, 2, 128, 300, 128),
    (8, 2, 128, 300, 1),
    (8, 2, 128, 37, 100),
    (16, 4, 256, 64, 256),
];
/// `attention.rs`'s chunk-against-one-forward sequences, `(tokens, P)`, at 8/2 heads.
const CHUNKS: [(usize, usize); 4] = [(512, 300), (512, 64), (256, 100), (448, 383)];

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn up(device: &CudaDevice, host: &[u16]) -> DeviceBuffer {
    DeviceBuffer::upload(device, host).expect("upload")
}

/// `u32`s as the `u16` pairs `DeviceBuffer` uploads (little-endian, as the H100 reads them).
fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let halves: Vec<u16> = (host.iter())
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    up(device, &halves)
}

fn layout(kv_heads: usize) -> PagedKvLayout {
    PagedKvLayout {
        num_blocks: BLOCKS as u32,
        num_kv_heads: kv_heads as u32,
        block_size: BS as u32,
        head_dim: HD as u32,
    }
}

/// One layer's K and V pools, built on the host: the `tokens` rows of `k` and `v` (`[tokens,
/// kv_heads · HD]`) at their slots over [`TABLE`], every other element [`POISON`]; uploaded.
fn host_caches(
    device: &CudaDevice,
    (k, v): (&[u16], &[u16]),
    tokens: usize,
    kvh: usize,
) -> (DeviceBuffer, DeviceBuffer) {
    let l = layout(kvh);
    let (mut kp, mut vp) = (
        vec![POISON; l.buffer_elems()],
        vec![POISON; l.buffer_elems()],
    );
    for t in 0..tokens {
        let slot = slot_of(&TABLE, t, BS) as u32;
        for h in 0..kvh {
            let (at, row) = (
                l.elem_offset_for_global_slot(slot, h as u32),
                (t * kvh + h) * HD,
            );
            kp[at..][..HD].copy_from_slice(&k[row..][..HD]);
            vp[at..][..HD].copy_from_slice(&v[row..][..HD]);
        }
    }
    (up(device, &kp), up(device, &vp))
}

/// One launch of the prefill attention of `q` (`[rows, q_heads · HD]`) over the caches at `[P,
/// L]`; its whole output, downloaded.
fn attend(
    device: &CudaDevice,
    q: &[u16],
    caches: (&DeviceBuffer, &DeviceBuffer),
    (prefix_len, q_len): (usize, usize),
    (rows, q_heads, kv_heads): (usize, usize, usize),
) -> Vec<u16> {
    let qb = up(device, q);
    let out = up(device, &vec![SENTINEL; rows * q_heads * HD]);
    let tb = up_u32(device, &TABLE);
    let sb = up_u32(device, &[prefix_len as u32, q_len as u32]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `q` and `out` hold rows·q_heads·HD bf16,
    // `out` its own allocation; the caches BLOCKS·kv_heads·BS·HD each; `TABLE` covers the
    // ceil((P+L)/64) ≤ 8 blocks the keys need, each below BLOCKS; null is the default stream,
    // which the download below waits for.
    let launched = unsafe {
        attn_prefill_bf16(
            (qb.ptr(), out.ptr()),
            (caches.0.ptr(), caches.1.ptr()),
            (tb.ptr(), sb.ptr()),
            (rows as u32, q_heads as u32, kv_heads as u32, HD as u32),
            (BS as u32, BLOCKS as u32),
            SCALE,
            std::ptr::null_mut(),
        )
    };
    let shape = (q_heads, kv_heads, rows, prefix_len, q_len);
    launched.unwrap_or_else(|e| panic!("attn_prefill_bf16 {shape:?} did not return 0: {e:?}"));
    out.download().expect("download the output")
}

/// FNV-1a (64-bit) of `bytes`: a short name for a dump, printed beside it.
fn fnv1a(bytes: &[u8]) -> u64 {
    (bytes.iter()).fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// Write `out` as `<case>.bin` under the dump directory, and print its line.
fn dump(case: &str, out: &[u16]) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("attn_prefill_dump");
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let bytes: Vec<u8> = out.iter().flat_map(|e| e.to_le_bytes()).collect();
    let path = dir.join(format!("{case}.bin"));
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    let nonfinite = (out.iter())
        .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
        .count();
    eprintln!(
        "case={case} elems={} nonfinite={nonfinite} fnv1a={:016x}",
        out.len(),
        fnv1a(&bytes)
    );
}

/// `attention.rs`'s six random cases, with its seeds, over host-built caches.
#[test]
fn dump_the_random_cases() {
    let device = h100();
    for (case, &(qh, kvh, rows, p, len)) in CASES.iter().enumerate() {
        let tokens = p + len;
        let seed = 0x0A77_E000 + case as u64;
        let q = bf16_inputs(rows * qh * HD, seed);
        let k = bf16_inputs(tokens * kvh * HD, seed ^ 0x100);
        let v = bf16_inputs(tokens * kvh * HD, seed ^ 0x200);
        let (kc, vc) = host_caches(&device, (&k, &v), tokens, kvh);
        let out = attend(&device, &q, (&kc, &vc), (p, len), (rows, qh, kvh));
        dump(
            &format!("random_{case}_h{qh}x{kvh}_r{rows}_p{p}_l{len}"),
            &out,
        );
    }
}

/// `attention.rs`'s four chunk sequences, each as one forward over the whole sequence and as the
/// chunk at `P`, with its seeds.
#[test]
fn dump_the_chunk_and_one_forward_launches() {
    let device = h100();
    let (qh, kvh) = (8usize, 2usize);
    for (tokens, p) in CHUNKS {
        let len = tokens - p;
        let (rows_one, rows_chunk) = (tokens.next_multiple_of(128), len.next_multiple_of(128));
        let k = bf16_inputs(tokens * kvh * HD, 0x0D08_0001 ^ tokens as u64);
        let v = bf16_inputs(tokens * kvh * HD, 0x0D08_0002 ^ tokens as u64);
        let q = bf16_inputs(tokens * qh * HD, 0x0D08_0003 ^ tokens as u64);
        let (kc, vc) = host_caches(&device, (&k, &v), tokens, kvh);
        let width = qh * HD;
        let mut q_one = vec![0u16; rows_one * width];
        q_one[..tokens * width].copy_from_slice(&q);
        let mut q_chunk = vec![0u16; rows_chunk * width];
        q_chunk[..len * width].copy_from_slice(&q[p * width..]);
        let one = attend(
            &device,
            &q_one,
            (&kc, &vc),
            (0, tokens),
            (rows_one, qh, kvh),
        );
        dump(&format!("chunk_t{tokens}_p{p}_one"), &one);
        let chunk = attend(
            &device,
            &q_chunk,
            (&kc, &vc),
            (p, len),
            (rows_chunk, qh, kvh),
        );
        dump(&format!("chunk_t{tokens}_p{p}_chunk"), &chunk);
    }
}

/// `attention.rs`'s GQA probe at 8/2 and 16/4 heads: K = 0, kv head h's V the constant h + 1.
#[test]
fn dump_the_gqa_probe() {
    let device = h100();
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        let (rows, p, len) = (128usize, 0usize, 128usize);
        let tokens = p + len;
        let k = vec![0u16; tokens * kvh * HD];
        let v: Vec<u16> = (0..tokens * kvh * HD)
            .map(|e| bf16::from_f32(((e / HD) % kvh + 1) as f32).to_bits())
            .collect();
        let (kc, vc) = host_caches(&device, (&k, &v), tokens, kvh);
        let q = bf16_inputs(rows * qh * HD, 0x0A77_6000 + qh as u64);
        let out = attend(&device, &q, (&kc, &vc), (p, len), (rows, qh, kvh));
        dump(&format!("gqa_h{qh}x{kvh}"), &out);
    }
}

/// One real row over a first block of one written slot and 63 NaN ones: the partly filled last
/// tile at its smallest.
#[test]
fn dump_one_row_over_a_partly_filled_block() {
    let device = h100();
    let (qh, kvh, rows, p, len) = (8usize, 2usize, 128usize, 0usize, 1usize);
    let tokens = p + len;
    let k = bf16_inputs(tokens * kvh * HD, 0x0D09_0001);
    let v = bf16_inputs(tokens * kvh * HD, 0x0D09_0002);
    let q = bf16_inputs(rows * qh * HD, 0x0D09_0003);
    let (kc, vc) = host_caches(&device, (&k, &v), tokens, kvh);
    let out = attend(&device, &q, (&kc, &vc), (p, len), (rows, qh, kvh));
    dump(&format!("partial_h{qh}x{kvh}_r{rows}_p{p}_l{len}"), &out);
}
