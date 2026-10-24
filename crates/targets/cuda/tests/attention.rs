// SPDX-License-Identifier: Apache-2.0
//! The paged KV cache's kernels on the H100, against an f64 CPU reference and exact probes: the
//! cache write (`kernels/kv_append_bf16.cu`), the causal GQA prefill attention over the cache
//! (`kernels/attn_prefill_bf16.cu`), the one-row decode attention over it
//! (`kernels/attn_decode_bf16.cu`), and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. Each test owns its device buffers and calls the launchers directly, with no
//! worker. Every case prints its line before a test fails, so one run reports all the numbers.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{
    ATTN_SCORE_MAX, AttnCase, attention_bound, attention_reference, bf16_inputs, bits,
    compare_attention, gather_paged, mismatched_bits, slot_of,
};
use half::bf16;
use scratchy_subtile::paged_kv_layout::PagedKvLayout;
use scratchy_target_cuda::kernels::{
    KernelError, attn_decode_bf16, attn_prefill_bf16, kv_append_bf16,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// qwen3.5's head dim, and the KV block size.
const HD: usize = 256;
const BS: usize = 64;
/// The pool's blocks in every test.
const BLOCKS: usize = 16;
const SCALE: f32 = 0.0625;
/// A permuted block table over the pool: logical block `j` is physical block `TABLE[j]`.
const TABLE: [u32; 8] = [11, 3, 14, 0, 7, 9, 2, 5];
/// An output element no launch wrote: a bf16 NaN.
const SENTINEL: u16 = 0xFFFF;
/// A cache element no append wrote: a bf16 NaN, so a write over it, or a read of it into an
/// output, shows.
const POISON: u16 = 0x7FC1;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
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

/// One layer's pool: `BLOCKS` blocks of `kv_heads` heads.
fn layout(kv_heads: usize) -> PagedKvLayout {
    PagedKvLayout {
        num_blocks: BLOCKS as u32,
        num_kv_heads: kv_heads as u32,
        block_size: BS as u32,
        head_dim: HD as u32,
    }
}

/// `rows` slots of a sequence at `prefix_len` over `table`, `u32::MAX` past `q_len`.
fn slots(table: &[u32], prefix_len: usize, q_len: usize, rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|i| {
            if i < q_len {
                slot_of(table, prefix_len + i, BS) as u32
            } else {
                u32::MAX
            }
        })
        .collect()
}

/// Append `k` and `v` (`[rows, kv_heads · HD]`) at `slots` into the caches, synchronously.
fn append(
    caches: (&DeviceBuffer, &DeviceBuffer),
    (k, v): (&DeviceBuffer, &DeviceBuffer),
    slots: &DeviceBuffer,
    rows: usize,
    kv_heads: usize,
) -> Result<(), KernelError> {
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `k` and `v` hold rows·kv_heads·HD bf16,
    // `slots` rows u32, the caches BLOCKS·kv_heads·BS·HD bf16 each, every slot in a block below
    // BLOCKS or u32::MAX; null is the default stream, which the download below waits for.
    let r = unsafe {
        kv_append_bf16(
            (k.ptr(), v.ptr()),
            slots.ptr(),
            (caches.0.ptr(), caches.1.ptr()),
            (rows as u32, kv_heads as u32, HD as u32),
            (BS as u32, BLOCKS as u32),
            std::ptr::null_mut(),
        )
    };
    // The download waits for the default stream, and with it the launch.
    r.map(|()| {
        slots.download().expect("wait for the append");
    })
}

/// Run the attention of `q` (`[rows, q_heads · HD]`) over the caches, `[P, L]` and `table`; the
/// output, downloaded.
fn attend(
    device: &CudaDevice,
    q: &[u16],
    caches: (&DeviceBuffer, &DeviceBuffer),
    table: &[u32],
    (prefix_len, q_len): (usize, usize),
    (rows, q_heads, kv_heads): (usize, usize, usize),
) -> Result<Vec<u16>, KernelError> {
    let qb = up(device, q);
    let out = up(device, &vec![SENTINEL; rows * q_heads * HD]);
    let tb = up_u32(device, table);
    let sb = up_u32(device, &[prefix_len as u32, q_len as u32]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `q` and `out` hold rows·q_heads·HD bf16,
    // `out` its own allocation; the caches BLOCKS·kv_heads·BS·HD each; `table` covers the
    // ceil((P+L)/64) blocks the keys need, each below BLOCKS; null is the default stream.
    unsafe {
        attn_prefill_bf16(
            (qb.ptr(), out.ptr()),
            (caches.0.ptr(), caches.1.ptr()),
            (tb.ptr(), sb.ptr()),
            (rows as u32, q_heads as u32, kv_heads as u32, HD as u32),
            (BS as u32, BLOCKS as u32),
            SCALE,
            std::ptr::null_mut(),
        )
    }?;
    Ok(out.download().expect("download the output"))
}

/// One layer's two caches, every element `fill`.
fn caches(device: &CudaDevice, kv_heads: usize, fill: &[u16]) -> (DeviceBuffer, DeviceBuffer) {
    assert_eq!(fill.len(), layout(kv_heads).buffer_elems());
    (up(device, fill), up(device, fill))
}

/// One bit-exact append of test 1: its device buffers (the caches, K, V and the slots) and the
/// caches as the append left them, for the refusals that follow.
struct Appended {
    caches: (DeviceBuffer, DeviceBuffer),
    kv: (DeviceBuffer, DeviceBuffer),
    slots: DeviceBuffer,
    after: (Vec<u16>, Vec<u16>),
    pass: bool,
}

/// 256 rows of `kvh` kv heads appended over a permuted table, rows 200..256 padding: the cache at
/// each real row's slot is the row bit for bit, and every other element keeps its poison.
fn append_bit_exact(device: &CudaDevice, kvh: usize) -> Appended {
    let (rows, real) = (256usize, 200usize);
    let l = layout(kvh);
    let k = bf16_inputs(rows * kvh * HD, 0x0CA0_0001);
    let v = bf16_inputs(rows * kvh * HD, 0x0CA0_0002);
    let (kb, vb) = (up(device, &k), up(device, &v));
    let poison = vec![POISON; l.buffer_elems()];
    let (kc, vc) = caches(device, kvh, &poison);
    let slot_map = slots(&TABLE[..4], 0, real, rows);
    let sb = up_u32(device, &slot_map);
    append((&kc, &vc), (&kb, &vb), &sb, rows, kvh).expect("kv_append_bf16");
    let (kh, vh) = (kc.download().expect("K"), vc.download().expect("V"));
    let (mut want_k, mut want_v) = (poison.clone(), poison.clone());
    for (t, &slot) in slot_map.iter().enumerate().take(real) {
        for h in 0..kvh {
            let at = l.elem_offset_for_global_slot(slot, h as u32);
            let row = (t * kvh + h) * HD;
            want_k[at..][..HD].copy_from_slice(&k[row..][..HD]);
            want_v[at..][..HD].copy_from_slice(&v[row..][..HD]);
        }
    }
    let mismatched = mismatched_bits(&kh, &want_k) + mismatched_bits(&vh, &want_v);
    eprintln!(
        "kv_append rows={rows} real={real} kv_heads={kvh} table={:?}: mismatched_bits={mismatched} \
         {}",
        &TABLE[..4],
        verdict(mismatched == 0)
    );
    Appended {
        caches: (kc, vc),
        kv: (kb, vb),
        slots: sb,
        after: (kh, vh),
        pass: mismatched == 0,
    }
}

/// Test 1: 256 rows appended over a permuted table of 16 blocks, rows 200..256 padding, at
/// qwen3.5-0.8b's 2 kv heads and qwen3.5-9b's 4: the cache at each real row's slot is the row
/// bit for bit, every other element keeps its poison; and each launcher refusal leaves the
/// 2-head cache untouched.
#[test]
fn kv_append_writes_the_mapped_slots_bit_exactly() {
    let device = h100();
    let nine_b = append_bit_exact(&device, 4);
    let Appended {
        caches: (kc, vc),
        kv: (kb, vb),
        slots: sb,
        after: (kh, vh),
        pass,
    } = append_bit_exact(&device, 2);
    let mut pass = pass && nine_b.pass;

    let refusals: [((u32, u32, u32), u32, &str); 6] = [
        ((0, 2, 256), 64, "rows 0"),
        ((256, 0, 256), 64, "kv_heads 0"),
        ((256, 2, 0), 64, "head_dim 0"),
        ((256, 2, 252), 64, "head_dim 252"),
        ((256, 2, 256), 16, "block_size 16"),
        ((1 << 20, 8, 256), 64, "rows·cols above INT_MAX"),
    ];
    for ((r, h, d), block_size, what) in refusals {
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            kv_append_bf16(
                (kb.ptr(), vb.ptr()),
                sb.ptr(),
                (kc.ptr(), vc.ptr()),
                (r, h, d),
                (block_size, BLOCKS as u32),
                std::ptr::null_mut(),
            )
        };
        let want = Err(KernelError::KvAppendShape {
            rows: r,
            kv_heads: h,
            head_dim: d,
            block_size,
        });
        let untouched = kc.download().expect("K") == kh && vc.download().expect("V") == vh;
        eprintln!(
            "kv_append refusal ({what}): {got:?} cache_untouched={untouched} {}",
            verdict(got == want && untouched)
        );
        pass &= got == want && untouched;
    }
    assert!(pass, "the KV append is not bit-exact, or a refusal wrote");
}

/// The random-input attention cases: `(q_heads, kv_heads, rows, P, L)`.
const CASES: [(usize, usize, usize, usize, usize); 6] = [
    (8, 2, 128, 0, 128),
    (8, 2, 256, 0, 256),
    (8, 2, 128, 300, 128),
    (8, 2, 128, 300, 1),
    (8, 2, 128, 37, 100),
    (16, 4, 256, 64, 256),
];

/// The keys and values of a sequence of `tokens`, `[tokens, kv_heads · HD]`, written into fresh
/// caches over [`TABLE`] with the append kernel; every other element of the pool is `fill`.
fn filled_caches(
    device: &CudaDevice,
    (k, v): (&[u16], &[u16]),
    tokens: usize,
    kvh: usize,
    fill: u16,
) -> (DeviceBuffer, DeviceBuffer) {
    let (kc, vc) = caches(device, kvh, &vec![fill; layout(kvh).buffer_elems()]);
    let sb = up_u32(device, &slots(&TABLE, 0, tokens, tokens));
    append(
        (&kc, &vc),
        (&up(device, k), &up(device, v)),
        &sb,
        tokens,
        kvh,
    )
    .expect("kv_append_bf16 fills the cache");
    (kc, vc)
}

/// Test 2: Every random case within the f64 bound, its score maximum at most 64. K and V are read
/// back from the cache, so the reference uses exactly the bits the kernel read. Every slot the
/// appends did not write is a NaN, so a read past `kv_len` or through a wrong block poisons the
/// row.
#[test]
fn the_attention_is_within_the_f64_bound() {
    let device = h100();
    let mut pass = true;
    for (case, &(qh, kvh, rows, p, len)) in CASES.iter().enumerate() {
        let tokens = p + len;
        let seed = 0x0A77_E000 + case as u64;
        let q = bf16_inputs(rows * qh * HD, seed);
        let k = bf16_inputs(tokens * kvh * HD, seed ^ 0x100);
        let v = bf16_inputs(tokens * kvh * HD, seed ^ 0x200);
        let (kc, vc) = filled_caches(&device, (&k, &v), tokens, kvh, POISON);
        let (kr, vr) = (
            gather_paged(&kc.download().expect("K"), &TABLE, tokens, (kvh, BS, HD)),
            gather_paged(&vc.download().expect("V"), &TABLE, tokens, (kvh, BS, HD)),
        );
        let out = attend(&device, &q, (&kc, &vc), &TABLE, (p, len), (rows, qh, kvh))
            .expect("attn_prefill_bf16");
        let (refs, score_max) = attention_reference(&AttnCase {
            q: &q,
            k: &kr,
            v: &vr,
            q_heads: qh,
            kv_heads: kvh,
            head_dim: HD,
            prefix_len: p,
            q_len: len,
            scale: f64::from(SCALE),
        });
        let cmp = compare_attention(&out, &refs, qh * HD);
        let padding_finite =
            (out[len * qh * HD..].iter()).all(|&b| bf16::from_bits(b).to_f32().is_finite());
        let ok = cmp.pass() && score_max <= ATTN_SCORE_MAX && padding_finite;
        eprintln!(
            "attention heads={qh}/{kvh} rows={rows} P={p} L={len}: {} score_max={score_max:.4} \
             (≤ {ATTN_SCORE_MAX}) padding_finite={padding_finite} {}",
            cmp.line(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an attention case is out of bound");
}

/// How the probe poisons the V of every slot the appends did not write.
#[derive(Clone, Copy, Debug)]
enum VPoison {
    /// One-hot at lane `slot % 256`, value 64: a finite value a read of the slot moves into a lane
    /// that must be zero.
    OneHot,
    /// Every element a bf16 NaN: the value an unwritten slot of the never-cleared pool may hold.
    /// A masked slot's p is 0, and 0 · NaN = NaN, so this catches a kernel that multiplies the
    /// partial last tile's unwritten V rows instead of zeroing them.
    Nan,
}

/// Test 3: With K = 0 every score is 0 and every p is 1, so `o_i = Σ_{j∈A_i} v_j / |A_i|`. V of key j
/// is one-hot at lane j (both kv heads); every slot the appends did not write holds the poison
/// ([`VPoison`]). So row i's output is `1/(P+i+1)` in lanes `0 ..= P+i`, and exactly `+0.0`
/// elsewhere (no NaN anywhere), unless the mask, the block lookup or the partial last tile's V is
/// wrong. kv_len = 137 and 100 leave the last block partial; 128 and 256 fill it.
#[test]
fn the_mask_and_block_table_probe_is_exact() {
    let device = h100();
    let (qh, kvh) = (8usize, 2usize);
    let l = layout(kvh);
    let mut pass = true;
    for poison in [VPoison::OneHot, VPoison::Nan] {
        for (rows, p, len) in [
            (128usize, 0usize, 128usize),
            (128, 37, 100),
            (256, 0, 256),
            (128, 0, 100),
        ] {
            let tokens = p + len;
            assert!(tokens <= HD, "one-hot V needs kv_len ≤ 256");
            // The poison in every slot's V; K zero.
            let mut vpool = match poison {
                VPoison::OneHot => vec![0u16; l.buffer_elems()],
                VPoison::Nan => vec![POISON; l.buffer_elems()],
            };
            if let VPoison::OneHot = poison {
                for slot in 0..BLOCKS * BS {
                    for h in 0..kvh {
                        vpool[l.elem_offset_for_global_slot(slot as u32, h as u32) + slot % HD] =
                            bits(64.0);
                    }
                }
            }
            let kc = up(&device, &vec![0u16; l.buffer_elems()]);
            let vc = up(&device, &vpool);
            let k = vec![0u16; tokens * kvh * HD];
            let mut v = vec![0u16; tokens * kvh * HD];
            for j in 0..tokens {
                for h in 0..kvh {
                    v[(j * kvh + h) * HD + j] = bits(1.0);
                }
            }
            let sb = up_u32(&device, &slots(&TABLE, 0, tokens, tokens));
            append(
                (&kc, &vc),
                (&up(&device, &k), &up(&device, &v)),
                &sb,
                tokens,
                kvh,
            )
            .expect("kv_append_bf16");
            let q = bf16_inputs(rows * qh * HD, 0x0A77_F000 + rows as u64 + p as u64);
            let out = attend(&device, &q, (&kc, &vc), &TABLE, (p, len), (rows, qh, kvh))
                .expect("attn_prefill_bf16");
            let (mut nonzero, mut off) = (0usize, 0usize);
            let nonfinite = (out[..len * qh * HD].iter())
                .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
                .count();
            for i in 0..len {
                let n = p + i + 1;
                let want = bits(1.0 / n as f32);
                for g in 0..qh {
                    for d in 0..HD {
                        let got = out[(i * qh + g) * HD + d];
                        if d < n {
                            // Within one bf16 rounding of 1/n: its bf16, or a neighbour.
                            off += usize::from(got.abs_diff(want) > 1);
                        } else {
                            nonzero += usize::from(got != 0);
                        }
                    }
                }
            }
            let ok = nonzero == 0 && off == 0 && nonfinite == 0;
            eprintln!(
                "mask/table probe poison={poison:?} rows={rows} P={p} L={len}: \
             nonzero_in_zero_lanes={nonzero} one_over_n_lanes_off={off} \
             nonfinite_outputs={nonfinite} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the mask, the block table or the partial last tile's V is wrong"
    );
}

/// Test 4: With K = 0 and kv head h's V the constant h + 1, query head g's output is exactly g / hr + 1.
#[test]
fn the_gqa_probe_is_exact() {
    let device = h100();
    let mut pass = true;
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        let (rows, p, len) = (128usize, 0usize, 128usize);
        let tokens = p + len;
        let k = vec![0u16; tokens * kvh * HD];
        let v: Vec<u16> = (0..tokens * kvh * HD)
            .map(|e| bits(((e / HD) % kvh + 1) as f32))
            .collect();
        let (kc, vc) = filled_caches(&device, (&k, &v), tokens, kvh, 0);
        let q = bf16_inputs(rows * qh * HD, 0x0A77_6000 + qh as u64);
        let out = attend(&device, &q, (&kc, &vc), &TABLE, (p, len), (rows, qh, kvh))
            .expect("attn_prefill_bf16");
        let hr = qh / kvh;
        let want: Vec<u16> = (0..len * qh * HD)
            .map(|e| bits(((e / HD) % qh / hr + 1) as f32))
            .collect();
        let mismatched = mismatched_bits(&out[..len * qh * HD], &want);
        eprintln!(
            "GQA probe heads={qh}/{kvh}: mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    assert!(pass, "a query head read the wrong kv head");
}

/// Test 5: Two launches on the same input are bit-identical.
#[test]
fn the_attention_is_deterministic() {
    let device = h100();
    let (qh, kvh, rows, p, len) = CASES[4];
    let tokens = p + len;
    let k = bf16_inputs(tokens * kvh * HD, 0x0D07_0001);
    let v = bf16_inputs(tokens * kvh * HD, 0x0D07_0002);
    let q = bf16_inputs(rows * qh * HD, 0x0D07_0003);
    let (kc, vc) = filled_caches(&device, (&k, &v), tokens, kvh, 0);
    let a = attend(&device, &q, (&kc, &vc), &TABLE, (p, len), (rows, qh, kvh)).expect("first");
    let b = attend(&device, &q, (&kc, &vc), &TABLE, (p, len), (rows, qh, kvh)).expect("second");
    let mismatched = mismatched_bits(&a, &b);
    eprintln!(
        "attention repeat: mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    assert_eq!(mismatched, 0, "two launches on one input differ");
}

/// A refused attention launch: `(rows, q_heads, kv_heads, head_dim)` and `(block_size,
/// num_blocks)`.
type Refusal = ((u32, u32, u32, u32), (u32, u32));

/// Test 6: Each launcher refusal returns the shape code and leaves `out` untouched.
#[test]
fn the_attention_launcher_refuses_each_bad_shape() {
    let device = h100();
    let (kc, vc) = caches(&device, 2, &vec![0u16; layout(2).buffer_elems()]);
    let q = up(&device, &vec![0u16; 256 * 8 * HD]);
    let out = up(&device, &vec![SENTINEL; 256 * 8 * HD]);
    let tb = up_u32(&device, &TABLE);
    let sb = up_u32(&device, &[0, 128]);
    let mut pass = true;
    let cases: [(Refusal, f32, &str); 6] = [
        (((128, 8, 2, 128), (64, 16)), SCALE, "head_dim 128"),
        (((192, 8, 2, 256), (64, 16)), SCALE, "rows 192"),
        (
            ((128, 8, 3, 256), (64, 16)),
            SCALE,
            "q_heads % kv_heads ≠ 0",
        ),
        (((128, 8, 2, 256), (16, 16)), SCALE, "block_size 16"),
        (((128, 8, 2, 256), (64, 0)), SCALE, "num_blocks 0"),
        (((128, 8, 2, 256), (64, 16)), f32::NAN, "scale NaN"),
    ];
    for (((rows, qh, kvh, hd), (block_size, num_blocks)), scale, what) in cases {
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            attn_prefill_bf16(
                (q.ptr(), out.ptr()),
                (kc.ptr(), vc.ptr()),
                (tb.ptr(), sb.ptr()),
                (rows, qh, kvh, hd),
                (block_size, num_blocks),
                scale,
                std::ptr::null_mut(),
            )
        };
        let want = Err(KernelError::AttnShape {
            rows,
            q_heads: qh,
            kv_heads: kvh,
            head_dim: hd,
            block_size,
            num_blocks,
        });
        let untouched = (out.download().expect("out"))
            .iter()
            .all(|&b| b == SENTINEL);
        eprintln!(
            "attention refusal ({what}): {got:?} out_untouched={untouched} {}",
            verdict(got == want && untouched)
        );
        pass &= got == want && untouched;
    }
    assert!(pass, "a refusal launched or wrote");
}

/// Test 7: the rows of a chunk are bit-identical to the same rows of one forward over the whole
/// sequence, over the same cache. A row's query and keys are the same either way; what changes is
/// where it sits in the bucket, and so how many KV tiles its CTA reads. In one 512-row forward
/// (P = 0, L = 512) sequence rows 256..383 are one CTA reading 6 tiles; as a chunk (P = 300,
/// L = 212) sequence rows 300..427 are one CTA reading 7. A tile past a row's last key is masked
/// out for it, so it must change that row's output by nothing, to the bit: the online softmax's
/// rescale must be exactly 1 there. qwen3.5-0.8b's geometry (8 q heads, 2 kv heads) and the
/// sequence of `cuda_prefill`'s (f).
#[test]
fn a_chunk_is_bit_identical_to_the_same_rows_of_one_forward() {
    let device = h100();
    let (qh, kvh) = (8usize, 2usize);
    let mut pass = true;
    for (tokens, p) in [(512usize, 300usize), (512, 64), (256, 100), (448, 383)] {
        let len = tokens - p;
        let rows_one = tokens.next_multiple_of(128);
        let rows_chunk = len.next_multiple_of(128);
        let k = bf16_inputs(tokens * kvh * HD, 0x0D08_0001 ^ tokens as u64);
        let v = bf16_inputs(tokens * kvh * HD, 0x0D08_0002 ^ tokens as u64);
        let q = bf16_inputs(tokens * qh * HD, 0x0D08_0003 ^ tokens as u64);
        let (kc, vc) = filled_caches(&device, (&k, &v), tokens, kvh, POISON);
        let width = qh * HD;
        let mut q_one = vec![0u16; rows_one * width];
        q_one[..tokens * width].copy_from_slice(&q);
        let mut q_chunk = vec![0u16; rows_chunk * width];
        q_chunk[..len * width].copy_from_slice(&q[p * width..]);
        let one = attend(
            &device,
            &q_one,
            (&kc, &vc),
            &TABLE,
            (0, tokens),
            (rows_one, qh, kvh),
        )
        .expect("one forward");
        let chunk = attend(
            &device,
            &q_chunk,
            (&kc, &vc),
            &TABLE,
            (p, len),
            (rows_chunk, qh, kvh),
        )
        .expect("the chunk");
        let (a, b) = (&chunk[..len * width], &one[p * width..tokens * width]);
        let mismatched = mismatched_bits(a, b);
        let worst_row = (0..len)
            .max_by_key(|&r| mismatched_bits(&a[r * width..][..width], &b[r * width..][..width]))
            .filter(|_| mismatched > 0);
        let ok = mismatched == 0;
        eprintln!(
            "attention chunk (P={p}, L={len}) vs rows {p}..{tokens} of one {tokens}-row forward: \
             mismatched_bits={mismatched}/{} worst_row={worst_row:?} {}",
            a.len(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "a chunk's attention differs from one forward's: see the lines above"
    );
}

// ── The decode attention (one query row) ───────────────────────────

/// The decode tests' pool: room for 8192 keys (128 blocks) and spare blocks a shuffled table can
/// pick from.
const DEC_BLOCKS: usize = 160;
/// Every decode `kv_len` the random cases run: one key, one warp's worth and less, each side of a
/// block boundary, the parity lengths, and the codegen capacity.
const DEC_KV_LENS: [usize; 12] = [1, 2, 7, 8, 9, 63, 64, 65, 300, 1000, 4096, 8192];

/// The decode pool's layout: `DEC_BLOCKS` blocks of `kv_heads` heads.
fn dec_layout(kv_heads: usize) -> PagedKvLayout {
    PagedKvLayout {
        num_blocks: DEC_BLOCKS as u32,
        ..layout(kv_heads)
    }
}

/// A permutation of the pool's blocks (splitmix64 Fisher-Yates, fixed by `seed`): a block table
/// whose logical order is not the physical one.
fn shuffled_blocks(seed: u64) -> Vec<u32> {
    let mut table: Vec<u32> = (0..DEC_BLOCKS as u32).collect();
    let mut state = seed;
    for i in (1..table.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        table.swap(i, (z % (i as u64 + 1)) as usize);
    }
    table
}

/// Fresh decode caches, every element `fill`, then the `tokens` keys and values written over
/// `table` with the append kernel.
fn dec_caches(
    device: &CudaDevice,
    (k, v): (&[u16], &[u16]),
    (tokens, kvh): (usize, usize),
    table: &[u32],
    fill: u16,
) -> (DeviceBuffer, DeviceBuffer) {
    let elems = dec_layout(kvh).buffer_elems();
    let (kc, vc) = (
        up(device, &vec![fill; elems]),
        up(device, &vec![fill; elems]),
    );
    let slot_map = slots(table, 0, tokens, tokens);
    let sb = up_u32(device, &slot_map);
    let (kb, vb) = (up(device, k), up(device, v));
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `k` and `v` hold tokens·kvh·HD bf16,
    // `slots` tokens u32, the caches DEC_BLOCKS·kvh·BS·HD bf16 each, every slot in a block below
    // DEC_BLOCKS; null is the default stream, which the download below waits for.
    unsafe {
        kv_append_bf16(
            (kb.ptr(), vb.ptr()),
            sb.ptr(),
            (kc.ptr(), vc.ptr()),
            (tokens as u32, kvh as u32, HD as u32),
            (BS as u32, DEC_BLOCKS as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("kv_append_bf16 fills the decode cache");
    sb.download().expect("wait for the append");
    (kc, vc)
}

/// The decode attention of one query row `q` (`[1, q_heads · HD]`) over the decode caches, the
/// sequence `[P, L]` and `table`; the output, downloaded.
fn attend_decode(
    device: &CudaDevice,
    q: &[u16],
    caches: (&DeviceBuffer, &DeviceBuffer),
    table: &[u32],
    (prefix_len, q_len): (usize, usize),
    (q_heads, kv_heads): (usize, usize),
) -> Result<Vec<u16>, KernelError> {
    let qb = up(device, q);
    let out = up(device, &vec![SENTINEL; q_heads * HD]);
    let tb = up_u32(device, table);
    let sb = up_u32(device, &[prefix_len as u32, q_len as u32]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `q` and `out` hold q_heads·HD bf16, `out`
    // its own allocation; the caches DEC_BLOCKS·kv_heads·BS·HD each; `table` covers the
    // ceil((P+L)/64) blocks the keys need, each below DEC_BLOCKS; null is the default stream.
    unsafe {
        attn_decode_bf16(
            (qb.ptr(), out.ptr()),
            (caches.0.ptr(), caches.1.ptr()),
            (tb.ptr(), sb.ptr()),
            (q_heads as u32, kv_heads as u32, HD as u32),
            (BS as u32, DEC_BLOCKS as u32),
            SCALE,
            std::ptr::null_mut(),
        )
    }?;
    Ok(out.download().expect("download the output"))
}

/// One decode case's inputs: the query row, and the sequence's keys and values.
fn decode_inputs(kv_len: usize, (qh, kvh): (usize, usize), seed: u64) -> [Vec<u16>; 3] {
    [
        bf16_inputs(qh * HD, seed),
        bf16_inputs(kv_len * kvh * HD, seed ^ 0x100),
        bf16_inputs(kv_len * kvh * HD, seed ^ 0x200),
    ]
}

/// Test 8: every decode `kv_len` within #791's f64 bound at L = 1 (the score-max condition
/// printed), no output non-finite, over a shuffled block table whose unused slots are NaN. The
/// same output, bit for bit, over the identity table with a clean (zero) pool, and on a repeat.
/// K and V are read back from the cache, so the reference uses exactly the bits the kernel read.
/// At qwen3.5-0.8b's heads (8 / 2) and qwen3.5-9b's (16 / 4).
#[test]
fn the_decode_attention_is_within_the_f64_bound() {
    let device = h100();
    let mut pass = true;
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        for (case, &kv_len) in DEC_KV_LENS.iter().enumerate() {
            let seed = 0x0DEC_0000 + case as u64;
            let [q, k, v] = decode_inputs(kv_len, (qh, kvh), seed);
            let shuffled = shuffled_blocks(seed);
            let need = kv_len.div_ceil(BS);
            let (kc, vc) = dec_caches(&device, (&k, &v), (kv_len, kvh), &shuffled, POISON);
            let (kr, vr) = (
                gather_paged(&kc.download().expect("K"), &shuffled, kv_len, (kvh, BS, HD)),
                gather_paged(&vc.download().expect("V"), &shuffled, kv_len, (kvh, BS, HD)),
            );
            let seq = (kv_len - 1, 1);
            let out = attend_decode(&device, &q, (&kc, &vc), &shuffled[..need], seq, (qh, kvh))
                .expect("attn_decode_bf16");
            let repeat = attend_decode(&device, &q, (&kc, &vc), &shuffled[..need], seq, (qh, kvh))
                .expect("attn_decode_bf16, again");
            let identity: Vec<u32> = (0..need as u32).collect();
            let (kc0, vc0) = dec_caches(&device, (&k, &v), (kv_len, kvh), &identity, 0);
            let clean = attend_decode(&device, &q, (&kc0, &vc0), &identity, seq, (qh, kvh))
                .expect("attn_decode_bf16, identity table, clean pool");
            let (refs, score_max) = attention_reference(&AttnCase {
                q: &q,
                k: &kr,
                v: &vr,
                q_heads: qh,
                kv_heads: kvh,
                head_dim: HD,
                prefix_len: kv_len - 1,
                q_len: 1,
                scale: f64::from(SCALE),
            });
            let cmp = compare_attention(&out, &refs, qh * HD);
            let nonfinite = (out.iter())
                .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
                .count();
            let (re, cl) = (
                mismatched_bits(&out, &repeat),
                mismatched_bits(&out, &clean),
            );
            let ok =
                cmp.pass() && nonfinite == 0 && score_max <= ATTN_SCORE_MAX && re == 0 && cl == 0;
            eprintln!(
                "decode attention heads={qh}/{kvh} kv_len={kv_len}: {} nonfinite={nonfinite} \
                 score_max={score_max:.4} (≤ {ATTN_SCORE_MAX}) repeat_mismatched_bits={re} \
                 shuffled_poisoned_vs_identity_clean_mismatched_bits={cl} {}",
                cmp.line(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a decode attention case failed: see the lines above");
}

/// Test 9: #791's probes at L = 1. With K = 0 every score is 0 and every p is 1. (a) GQA: kv head
/// h's V the constant h + 1, so query head g's output is exactly g / hr + 1. (b) Mask and table:
/// V of key j one-hot at lane j, every slot the appends did not write poisoned (one-hot 64 at lane
/// `slot % 256`, or NaN), so the output is `1/kv_len` in lanes `0 .. kv_len` and exactly `+0.0`
/// elsewhere, unless the kernel reads a key at or past kv_len or through a wrong block.
#[test]
fn the_decode_attention_probes_are_exact() {
    let device = h100();
    let mut pass = true;
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        let kv_len = 300usize;
        let k = vec![0u16; kv_len * kvh * HD];
        let v: Vec<u16> = (0..kv_len * kvh * HD)
            .map(|e| bits(((e / HD) % kvh + 1) as f32))
            .collect();
        let table = shuffled_blocks(0x0DEC_6000 + qh as u64);
        let (kc, vc) = dec_caches(&device, (&k, &v), (kv_len, kvh), &table, POISON);
        let q = bf16_inputs(qh * HD, 0x0DEC_6100 + qh as u64);
        let need = kv_len.div_ceil(BS);
        let out = attend_decode(
            &device,
            &q,
            (&kc, &vc),
            &table[..need],
            (kv_len - 1, 1),
            (qh, kvh),
        )
        .expect("attn_decode_bf16");
        let hr = qh / kvh;
        let want: Vec<u16> = (0..qh * HD)
            .map(|e| bits((e / HD / hr + 1) as f32))
            .collect();
        let mismatched = mismatched_bits(&out, &want);
        eprintln!(
            "decode GQA probe heads={qh}/{kvh} kv_len={kv_len}: mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    let (qh, kvh) = (8usize, 2usize);
    let l = dec_layout(kvh);
    for poison in [VPoison::OneHot, VPoison::Nan] {
        for kv_len in [1usize, 7, 64, 65, 137, 256] {
            let table = shuffled_blocks(0x0DEC_7000 + kv_len as u64);
            let mut vpool = match poison {
                VPoison::OneHot => vec![0u16; l.buffer_elems()],
                VPoison::Nan => vec![POISON; l.buffer_elems()],
            };
            if let VPoison::OneHot = poison {
                for slot in 0..DEC_BLOCKS * BS {
                    for h in 0..kvh {
                        vpool[l.elem_offset_for_global_slot(slot as u32, h as u32) + slot % HD] =
                            bits(64.0);
                    }
                }
            }
            let kc = up(&device, &vec![0u16; l.buffer_elems()]);
            let vc = up(&device, &vpool);
            let k = vec![0u16; kv_len * kvh * HD];
            let mut v = vec![0u16; kv_len * kvh * HD];
            for j in 0..kv_len {
                for h in 0..kvh {
                    v[(j * kvh + h) * HD + j] = bits(1.0);
                }
            }
            let sb = up_u32(&device, &slots(&table, 0, kv_len, kv_len));
            let (kb, vb) = (up(&device, &k), up(&device, &v));
            // SAFETY: as `dec_caches`'.
            unsafe {
                kv_append_bf16(
                    (kb.ptr(), vb.ptr()),
                    sb.ptr(),
                    (kc.ptr(), vc.ptr()),
                    (kv_len as u32, kvh as u32, HD as u32),
                    (BS as u32, DEC_BLOCKS as u32),
                    std::ptr::null_mut(),
                )
            }
            .expect("kv_append_bf16");
            sb.download().expect("wait for the append");
            let q = bf16_inputs(qh * HD, 0x0DEC_7100 + kv_len as u64);
            let need = kv_len.div_ceil(BS);
            let out = attend_decode(
                &device,
                &q,
                (&kc, &vc),
                &table[..need],
                (kv_len - 1, 1),
                (qh, kvh),
            )
            .expect("attn_decode_bf16");
            let want = bits(1.0 / kv_len as f32);
            let (mut nonzero, mut off) = (0usize, 0usize);
            let nonfinite = (out.iter())
                .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
                .count();
            for g in 0..qh {
                for d in 0..HD {
                    let got = out[g * HD + d];
                    if d < kv_len {
                        off += usize::from(got.abs_diff(want) > 1);
                    } else {
                        nonzero += usize::from(got != 0);
                    }
                }
            }
            let ok = nonzero == 0 && off == 0 && nonfinite == 0;
            eprintln!(
                "decode mask/table probe poison={poison:?} kv_len={kv_len}: \
                 nonzero_in_zero_lanes={nonzero} one_over_n_lanes_off={off} \
                 nonfinite_outputs={nonfinite} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a decode probe is not exact: see the lines above");
}

/// Test 10: the decode attention against the prefill attention's row for the same query at the
/// same position over the same cache: the prefill launched with 128 rows, `L = min(kv_len, 128)`
/// and `P = kv_len − L`, so its row `L − 1` is the query at position `kv_len − 1`. Two different
/// kernels (the prefill rounds `p` to bf16 before P·V, the decode stays in f32), so no bit
/// requirement: their difference is within the sum of their two #791 bounds.
#[test]
fn the_decode_attention_is_within_two_bounds_of_the_prefill_row() {
    let device = h100();
    let (qh, kvh, rows) = (8usize, 2usize, 128usize);
    let mut pass = true;
    for kv_len in [65usize, 300, 1000] {
        let seed = 0x0DEC_8000 + kv_len as u64;
        let len = kv_len.min(rows);
        let p = kv_len - len;
        let q_rows = bf16_inputs(rows * qh * HD, seed);
        let q = &q_rows[(len - 1) * qh * HD..][..qh * HD];
        let k = bf16_inputs(kv_len * kvh * HD, seed ^ 0x100);
        let v = bf16_inputs(kv_len * kvh * HD, seed ^ 0x200);
        let table = shuffled_blocks(seed);
        let need = kv_len.div_ceil(BS);
        let (kc, vc) = dec_caches(&device, (&k, &v), (kv_len, kvh), &table, POISON);
        let decode = attend_decode(
            &device,
            q,
            (&kc, &vc),
            &table[..need],
            (kv_len - 1, 1),
            (qh, kvh),
        )
        .expect("attn_decode_bf16");
        let qb = up(&device, &q_rows);
        let out = up(&device, &vec![SENTINEL; rows * qh * HD]);
        let tb = up_u32(&device, &table[..need]);
        let sb = up_u32(&device, &[p as u32, len as u32]);
        // SAFETY: as `attend`'s, over the decode pool of DEC_BLOCKS blocks.
        unsafe {
            attn_prefill_bf16(
                (qb.ptr(), out.ptr()),
                (kc.ptr(), vc.ptr()),
                (tb.ptr(), sb.ptr()),
                (rows as u32, qh as u32, kvh as u32, HD as u32),
                (BS as u32, DEC_BLOCKS as u32),
                SCALE,
                std::ptr::null_mut(),
            )
        }
        .expect("attn_prefill_bf16");
        let prefill = out.download().expect("download the prefill output");
        let prefill = &prefill[(len - 1) * qh * HD..][..qh * HD];
        let (kr, vr) = (
            gather_paged(&kc.download().expect("K"), &table, kv_len, (kvh, BS, HD)),
            gather_paged(&vc.download().expect("V"), &table, kv_len, (kvh, BS, HD)),
        );
        let (refs, score_max) = attention_reference(&AttnCase {
            q,
            k: &kr,
            v: &vr,
            q_heads: qh,
            kv_heads: kvh,
            head_dim: HD,
            prefix_len: kv_len - 1,
            q_len: 1,
            scale: f64::from(SCALE),
        });
        let (mut bad, mut worst, mut ulps) = (0usize, 0.0f64, 0u16);
        for ((&d, &f), &(r, pv)) in decode.iter().zip(prefill).zip(&refs) {
            let (df, ff) = (
                f64::from(bf16::from_bits(d).to_f32()),
                f64::from(bf16::from_bits(f).to_f32()),
            );
            let bound = 2.0 * attention_bound(r, pv);
            let diff = (df - ff).abs();
            bad += usize::from(!(df.is_finite() && ff.is_finite() && diff <= bound));
            worst = worst.max(diff / bound);
            ulps = ulps.max(d.abs_diff(f));
        }
        let ok = bad == 0 && score_max <= ATTN_SCORE_MAX;
        eprintln!(
            "decode vs prefill row {} (P={p}, L={len}) kv_len={kv_len}: out_of_bound={bad} \
             worst_ratio={worst:.4} max_ulps={ulps} score_max={score_max:.4} {}",
            len - 1,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the decode attention is outside two bounds of the prefill row"
    );
}

/// A refused decode launch: `(q_heads, kv_heads, head_dim)` and `(block_size, num_blocks)`.
type DecodeRefusal = ((u32, u32, u32), (u32, u32));

/// Test 11: each decode launcher refusal returns the shape code and leaves `out` untouched.
#[test]
fn the_decode_attention_launcher_refuses_each_bad_shape() {
    let device = h100();
    let elems = dec_layout(2).buffer_elems();
    let (kc, vc) = (
        up(&device, &vec![0u16; elems]),
        up(&device, &vec![0u16; elems]),
    );
    let q = up(&device, &vec![0u16; 16 * HD]);
    let out = up(&device, &vec![SENTINEL; 16 * HD]);
    let tb = up_u32(&device, &[0, 1]);
    let sb = up_u32(&device, &[99, 1]);
    let mut pass = true;
    let cases: [(DecodeRefusal, f32, &str); 7] = [
        (((8, 2, 128), (64, 16)), SCALE, "head_dim 128"),
        (((8, 3, 256), (64, 16)), SCALE, "q_heads % kv_heads ≠ 0"),
        (((8, 0, 256), (64, 16)), SCALE, "kv_heads 0"),
        (((8, 2, 256), (16, 16)), SCALE, "block_size 16"),
        (((8, 2, 256), (64, 0)), SCALE, "num_blocks 0"),
        (((8, 2, 256), (64, 16)), f32::NAN, "scale NaN"),
        (((8, 2, 256), (64, 16)), 0.0, "scale 0"),
    ];
    for (((qh, kvh, hd), (block_size, num_blocks)), scale, what) in cases {
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            attn_decode_bf16(
                (q.ptr(), out.ptr()),
                (kc.ptr(), vc.ptr()),
                (tb.ptr(), sb.ptr()),
                (qh, kvh, hd),
                (block_size, num_blocks),
                scale,
                std::ptr::null_mut(),
            )
        };
        let want = Err(KernelError::AttnDecodeShape {
            q_heads: qh,
            kv_heads: kvh,
            head_dim: hd,
            block_size,
            num_blocks,
        });
        let untouched = (out.download().expect("out"))
            .iter()
            .all(|&b| b == SENTINEL);
        eprintln!(
            "decode attention refusal ({what}): {got:?} out_untouched={untouched} {}",
            verdict(got == want && untouched)
        );
        pass &= got == want && untouched;
    }
    assert!(pass, "a decode refusal launched or wrote");
}

/// The KV append is instantiated for head_dim 256 only (#853), so head_dim 128 (a multiple of 8)
/// is refused and leaves the cache untouched; and the same append twice, over the same poisoned
/// caches, gives the same bits.
#[test]
fn kv_append_refuses_head_dim_128_and_repeats() {
    let device = h100();
    let Appended {
        caches: (kc, vc),
        kv: (kb, vb),
        slots: sb,
        after: (kh, vh),
        pass,
    } = append_bit_exact(&device, 2);
    let (r, h, d, block_size) = (256u32, 2u32, 128u32, BS as u32);
    // SAFETY: refused before any pointer is touched.
    let got = unsafe {
        kv_append_bf16(
            (kb.ptr(), vb.ptr()),
            sb.ptr(),
            (kc.ptr(), vc.ptr()),
            (r, h, d),
            (block_size, BLOCKS as u32),
            std::ptr::null_mut(),
        )
    };
    let want = Err(KernelError::KvAppendShape {
        rows: r,
        kv_heads: h,
        head_dim: d,
        block_size,
    });
    let untouched = kc.download().expect("K") == kh && vc.download().expect("V") == vh;
    let refused = got == want && untouched;
    eprintln!(
        "kv_append refusal (head_dim 128): {got:?} cache_untouched={untouched} {}",
        verdict(refused)
    );
    let again = append_bit_exact(&device, 2);
    let mismatched = mismatched_bits(&again.after.0, &kh) + mismatched_bits(&again.after.1, &vh);
    eprintln!(
        "kv_append rows=256 kv_heads=2 repeat mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    assert!(
        pass && again.pass && refused && mismatched == 0,
        "the KV append did not refuse head_dim 128, or did not repeat bit for bit"
    );
}

/// The decode attention merges and stores the 16 rows of each consumer warpgroup's warp 0, so a GQA
/// group above 16 (#854) is refused: 34 query heads over 2 kv heads (`hr` 17), every other
/// dimension one the kernel takes, leaves `out` untouched.
#[test]
fn the_decode_attention_launcher_refuses_a_group_above_16() {
    let device = h100();
    let elems = dec_layout(2).buffer_elems();
    let (kc, vc) = (
        up(&device, &vec![0u16; elems]),
        up(&device, &vec![0u16; elems]),
    );
    let (qh, kvh, hd, block_size, num_blocks) = (34u32, 2u32, HD as u32, BS as u32, 16u32);
    let q = up(&device, &vec![0u16; qh as usize * HD]);
    let out = up(&device, &vec![SENTINEL; qh as usize * HD]);
    let tb = up_u32(&device, &[0, 1]);
    let sb = up_u32(&device, &[99, 1]);
    // SAFETY: refused before any pointer is touched.
    let got = unsafe {
        attn_decode_bf16(
            (q.ptr(), out.ptr()),
            (kc.ptr(), vc.ptr()),
            (tb.ptr(), sb.ptr()),
            (qh, kvh, hd),
            (block_size, num_blocks),
            SCALE,
            std::ptr::null_mut(),
        )
    };
    let want = Err(KernelError::AttnDecodeShape {
        q_heads: qh,
        kv_heads: kvh,
        head_dim: hd,
        block_size,
        num_blocks,
    });
    let untouched = (out.download().expect("out"))
        .iter()
        .all(|&b| b == SENTINEL);
    let ok = got == want && untouched;
    eprintln!(
        "decode attention refusal (q_heads / kv_heads = 17): {got:?} out_untouched={untouched} {}",
        verdict(ok)
    );
    assert!(ok, "the decode attention did not refuse a group of 17");
}

/// The decode attention's time per launch at kv_len 300 and 8192, at qwen3.5-0.8b's heads (8 / 2)
/// and qwen3.5-9b's (16 / 4): 50 launches on the default stream after one warm-up, timed to the
/// download that waits for them. Recorded for the base-against-change comparison, not asserted.
#[test]
fn the_decode_attention_time_is_recorded() {
    const LAUNCHES: u32 = 50;
    let device = h100();
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        for kv_len in [300usize, 8192] {
            let seed = 0x0DEC_9000 + kv_len as u64;
            let [q, k, v] = decode_inputs(kv_len, (qh, kvh), seed);
            let table = shuffled_blocks(seed);
            let need = kv_len.div_ceil(BS);
            let (kc, vc) = dec_caches(&device, (&k, &v), (kv_len, kvh), &table, POISON);
            let qb = up(&device, &q);
            let out = up(&device, &vec![SENTINEL; qh * HD]);
            let tb = up_u32(&device, &table[..need]);
            let sb = up_u32(&device, &[kv_len as u32 - 1, 1]);
            let launch = || {
                // SAFETY: as `attend_decode`'s.
                unsafe {
                    attn_decode_bf16(
                        (qb.ptr(), out.ptr()),
                        (kc.ptr(), vc.ptr()),
                        (tb.ptr(), sb.ptr()),
                        (qh as u32, kvh as u32, HD as u32),
                        (BS as u32, DEC_BLOCKS as u32),
                        SCALE,
                        std::ptr::null_mut(),
                    )
                }
                .expect("attn_decode_bf16");
            };
            launch();
            out.download().expect("wait for the warm-up");
            let start = std::time::Instant::now();
            for _ in 0..LAUNCHES {
                launch();
            }
            out.download().expect("wait for the launches");
            let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES);
            eprintln!("decode attention time heads={qh}/{kvh} kv_len={kv_len}: {us:.1} us/launch");
        }
    }
}

/// For the base-against-change comparison (#854): the decode output at every `DEC_KV_LENS` case
/// of `the_decode_attention_is_within_the_f64_bound` (its inputs, its shuffled table, its poisoned
/// pool), at both head counts, written as raw little-endian bf16 to
/// `<CARGO_TARGET_TMPDIR>/attn_decode_dump/<case>.bin`, with `case= elems= fnv1a=` printed. The
/// same file run at two commits gives the `base_vs_pr mismatched_bits=` per case; this test
/// asserts only that each launch ran.
#[test]
fn the_decode_attention_outputs_are_dumped_for_a_base_comparison() {
    let device = h100();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("attn_decode_dump");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    for (qh, kvh) in [(8usize, 2usize), (16, 4)] {
        for (case, &kv_len) in DEC_KV_LENS.iter().enumerate() {
            let seed = 0x0DEC_0000 + case as u64;
            let [q, k, v] = decode_inputs(kv_len, (qh, kvh), seed);
            let shuffled = shuffled_blocks(seed);
            let need = kv_len.div_ceil(BS);
            let (kc, vc) = dec_caches(&device, (&k, &v), (kv_len, kvh), &shuffled, POISON);
            let out = attend_decode(
                &device,
                &q,
                (&kc, &vc),
                &shuffled[..need],
                (kv_len - 1, 1),
                (qh, kvh),
            )
            .expect("attn_decode_bf16");
            let bytes: Vec<u8> = out.iter().flat_map(|e| e.to_le_bytes()).collect();
            let name = format!("attn_decode_h{qh}_{kvh}_kv{kv_len}");
            std::fs::write(dir.join(format!("{name}.bin")), &bytes).expect("write a dump");
            let fnv = (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
            });
            eprintln!("case={name} elems={} fnv1a={fnv:016x}", out.len());
        }
    }
}
