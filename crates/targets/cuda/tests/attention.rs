// SPDX-License-Identifier: Apache-2.0
//! The paged KV cache's kernels on the H100, against an f64 CPU reference and exact probes: the
//! cache write (`kernels/kv_append_bf16.cu`) and the causal GQA prefill attention over the cache
//! (`kernels/attn_prefill_bf16.cu`), and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. Each test owns its device buffers and calls the launchers directly, with no
//! worker. Every case prints its line before a test fails, so one run reports all the numbers.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{
    ATTN_SCORE_MAX, AttnCase, attention_reference, bf16_inputs, bits, compare_attention,
    gather_paged, mismatched_bits, slot_of,
};
use half::bf16;
use scratchy_subtile::paged_kv_layout::PagedKvLayout;
use scratchy_target_cuda::kernels::{KernelError, attn_prefill_bf16, kv_append_bf16};
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

/// Test 1: 256 rows of 2 kv heads appended over a permuted table of 16 blocks, rows 200..256 padding:
/// the cache at each real row's slot is the row bit for bit, every other element keeps its
/// poison, and each launcher refusal leaves the cache untouched.
#[test]
fn kv_append_writes_the_mapped_slots_bit_exactly() {
    let device = h100();
    let (rows, kvh, real) = (256usize, 2usize, 200usize);
    let l = layout(kvh);
    let k = bf16_inputs(rows * kvh * HD, 0x0CA0_0001);
    let v = bf16_inputs(rows * kvh * HD, 0x0CA0_0002);
    let (kb, vb) = (up(&device, &k), up(&device, &v));
    let poison = vec![POISON; l.buffer_elems()];
    let (kc, vc) = caches(&device, kvh, &poison);
    let slot_map = slots(&TABLE[..4], 0, real, rows);
    let sb = up_u32(&device, &slot_map);
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
    let mut pass = mismatched == 0;

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
