// SPDX-License-Identifier: Apache-2.0
//! FP8 S5 on the H100: the e4m3 KV cache's kernels. The append (`kernels/kv_append_fp8.cu`) bit for
//! bit against the host's `e4m3_satfinite`, over every bf16 bit pattern; each attention kernel
//! over the e4m3 cache (`kernels/attn_*_fp8kv.cu`) bit for bit against its bf16 kernel over the
//! same cache widened on the host; a poisoned pool; determinism; the launchers' refusals; and the
//! decode times of both, recorded.
//!
//! No new f64 gate: each fp8kv kernel must equal, bit for bit, a bf16 kernel that
//! `tests/attention.rs` already holds to the f64 bound over ANY bf16 cache, and the widened cache is
//! one (every e4m3 value is a bf16 value).
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips. A binary of its
//! own, so a fault here cannot leave `tests/attention.rs` without a verdict: a CUDA fault poisons
//! the context for every later test in its process. Every case prints its line before a test fails.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
    pub mod watchdog;
}

use std::ffi::c_void;

use common::fp8::{e4m3_satfinite, e4m3_to_bf16_bits};
use common::numerics::{bf16_inputs, bits, centred_gains, mismatched_bits, slot_of};
use half::bf16;
use scratchy_subtile::paged_kv_layout::PagedKvLayout;
use scratchy_target_cuda::kernels::{
    KernelError, attn_decode_d512_bf16, attn_decode_d512_fp8kv, attn_decode_windowed_bf16,
    attn_decode_windowed_fp8kv, attn_prefill_d512_bf16, attn_prefill_d512_fp8kv,
    attn_prefill_windowed_bf16, attn_prefill_windowed_fp8kv, kv_append_fp8,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// The head dims, and the KV block size.
const HD: usize = 256;
const D512: usize = 512;
const BS: usize = 64;
/// The append tests' pool, and its permuted block table (`tests/attention.rs`'s).
const BLOCKS: usize = 16;
const TABLE: [u32; 8] = [11, 3, 14, 0, 7, 9, 2, 5];
/// The attention tests' pool: 8192 tokens' blocks and more.
const DEC_BLOCKS: usize = 160;
/// The head-dim-256 tests' scale with `bf16_inputs`-sized scores (`tests/attention.rs`'s).
const SCALE: f32 = 0.0625;
/// Gemma 4's sliding window.
const WINDOW: u32 = 1024;
/// An output element no launch wrote: a bf16 NaN.
const SENTINEL: u16 = 0xFFFF;
/// A cache byte no append wrote: an e4m3 NaN, which widens to a bf16 NaN.
const POISON: u8 = 0x7F;

/// `tests/attention.rs`'s cases, reused verbatim. The prefill's `(q_heads, kv_heads, rows, P, L)`.
const CASES: [(usize, usize, usize, usize, usize); 7] = [
    (8, 2, 128, 0, 128),
    (8, 2, 256, 0, 256),
    (8, 2, 128, 300, 128),
    (8, 2, 128, 300, 1),
    (8, 2, 128, 37, 100),
    (16, 4, 256, 64, 256),
    (16, 2, 256, 64, 256),
];
/// The decode `kv_len`s.
const DEC_KV_LENS: [usize; 12] = [1, 2, 7, 8, 9, 63, 64, 65, 300, 1000, 4096, 8192];
/// The windowed prefill's `(P, L, rows)` and decode `kv_len`s, at 16 query heads over 8.
const WINDOW_PREFILLS: [(usize, usize, usize); 6] = [
    (0, 2048, 2048),
    (0, 4096, 4096),
    (1000, 512, 512),
    (1500, 300, 384),
    (3000, 1024, 1024),
    (0, 100, 2048),
];
const WINDOW_DECODES: [usize; 8] = [1, 64, 1024, 1025, 1088, 2049, 4096, 8192];
/// The head-dim-512 prefill's `(P, L)` and decode `kv_len`s, at 16 query heads over 2.
const D512_PREFILLS: [(usize, usize); 4] = [(0, 512), (0, 4096), (300, 512), (4000, 96)];
const D512_DECODES: [usize; 7] = [1, 64, 65, 320, 321, 4096, 8192];

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn up(device: &CudaDevice, host: &[u16]) -> DeviceBuffer {
    DeviceBuffer::upload(device, host).expect("upload")
}

fn up_bytes(device: &CudaDevice, host: &[u8]) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, host).expect("upload")
}

/// `u32`s as the `u16` pairs `DeviceBuffer` uploads (little-endian, as the H100 reads them).
fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let halves: Vec<u16> = (host.iter())
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    up(device, &halves)
}

fn down_bytes(b: &DeviceBuffer) -> Vec<u8> {
    (b.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect()
}

/// A permutation of the attention pool's blocks (splitmix64 Fisher-Yates, fixed by `seed`).
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

/// `rows` slots of a sequence from position 0 over `table`, `u32::MAX` past `real`.
fn slots(table: &[u32], real: usize, rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|i| match i < real {
            true => slot_of(table, i, BS) as u32,
            false => u32::MAX,
        })
        .collect()
}

/// Append `k` and `v` (`[rows, kv_heads · head_dim]` bf16) at `slots` into the e4m3 caches of
/// `blocks` blocks, synchronously.
fn append(
    caches: (&DeviceBuffer, &DeviceBuffer),
    (k, v): (&DeviceBuffer, &DeviceBuffer),
    slots: &DeviceBuffer,
    (rows, kv_heads, head_dim): (usize, usize, usize),
    blocks: usize,
) -> Result<(), KernelError> {
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `k` and `v` hold rows·kv_heads·head_dim
    // bf16, `slots` rows u32, the caches blocks·kv_heads·64·head_dim bytes each, every slot in a
    // block below `blocks` or u32::MAX; null is the default stream, which the download waits for.
    let r = unsafe {
        kv_append_fp8(
            (k.ptr(), v.ptr()),
            slots.ptr(),
            (caches.0.ptr(), caches.1.ptr()),
            (rows as u32, kv_heads as u32, head_dim as u32),
            (BS as u32, blocks as u32),
            std::ptr::null_mut(),
        )
    };
    r.map(|()| {
        slots.download().expect("wait for the append");
    })
}

// ── kv_append_fp8 ───────────────────────────────────────────────────

/// `rows` rows of `kvh` heads of `hd` appended at `slots` into a pool of `BLOCKS` blocks, every
/// byte `POISON` before: each real row's slot holds `e4m3_satfinite` of the row, every other byte
/// is still `POISON`. Returns the case's mismatches as `(nan, saturated, subnormal, other)` by the
/// input's class, and the device buffers for the refusals that follow.
fn append_exact(
    device: &CudaDevice,
    (k, v): (&[u16], &[u16]),
    (rows, real, kvh, hd): (usize, usize, usize, usize),
) -> ([usize; 4], [DeviceBuffer; 5]) {
    let l = PagedKvLayout {
        num_blocks: BLOCKS as u32,
        num_kv_heads: kvh as u32,
        block_size: BS as u32,
        head_dim: hd as u32,
    };
    let poison = vec![POISON; l.buffer_elems()];
    let (kc, vc) = (up_bytes(device, &poison), up_bytes(device, &poison));
    let (kb, vb) = (up(device, k), up(device, v));
    let slot_map = slots(&TABLE[..4], real, rows);
    let sb = up_u32(device, &slot_map);
    append((&kc, &vc), (&kb, &vb), &sb, (rows, kvh, hd), BLOCKS).expect("kv_append_fp8");
    let (mut want_k, mut want_v) = (poison.clone(), poison.clone());
    for (t, &slot) in slot_map.iter().enumerate().take(real) {
        for h in 0..kvh {
            let at = l.elem_offset_for_global_slot(slot, h as u32);
            let row = (t * kvh + h) * hd;
            for e in 0..hd {
                want_k[at + e] = e4m3_satfinite(bf16::from_bits(k[row + e]).to_f32());
                want_v[at + e] = e4m3_satfinite(bf16::from_bits(v[row + e]).to_f32());
            }
        }
    }
    // Each mismatch by what the input was: unwritten bytes and written ones alike are counted.
    let mut by_class = [0usize; 4];
    let inputs = |at: usize, of: &[u16]| -> Option<f32> {
        let (block, rest) = (at / l.kv_blk_stride(), at % l.kv_blk_stride());
        let (h, off, e) = (rest / (BS * hd), rest / hd % BS, rest % hd);
        let slot = block * BS + off;
        let t = slot_map[..real].iter().position(|&s| s as usize == slot)?;
        Some(bf16::from_bits(of[(t * kvh + h) * hd + e]).to_f32())
    };
    for (got, want, of) in [(down_bytes(&kc), &want_k, k), (down_bytes(&vc), &want_v, v)] {
        for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                let class = match inputs(at, of) {
                    Some(x) if x.is_nan() => 0,
                    Some(x) if x.abs() >= 448.0 => 1,
                    Some(x) if x != 0.0 && x.abs() < 2f32.powi(-6) => 2,
                    _ => 3,
                };
                by_class[class] += 1;
            }
        }
    }
    (by_class, [kc, vc, kb, vb, sb])
}

/// T1: the append writes `e4m3_satfinite` of each element at each real row's slot and nothing
/// else: at head dim 256 over 2, 4 and 8 kv heads (Qwen3.6-35B's, Qwen3.8-27B's, Gemma's sliding
/// layers') and at 512 over 2 (Gemma's global layers), 256 rows over a permuted table, rows
/// 200..256 padding. Then every bf16 bit pattern (65,536 as 256 rows of one head of 256: NaN to
/// 0x7F; ±Inf and |x| ≥ 448 to ±448; the subnormals). Then the 6 launcher refusals, each leaving
/// the cache untouched.
#[test]
fn kv_append_fp8_writes_the_mapped_slots_bit_exactly() {
    let device = h100();
    let mut pass = true;
    for (kvh, hd) in [(2usize, HD), (4, HD), (8, HD), (2, D512)] {
        let k = bf16_inputs(256 * kvh * hd, 0x0F8A_0001 ^ (kvh * hd) as u64);
        let v = bf16_inputs(256 * kvh * hd, 0x0F8A_0002 ^ (kvh * hd) as u64);
        let (bad, _) = append_exact(&device, (&k, &v), (256, 200, kvh, hd));
        let ok = bad == [0; 4];
        eprintln!(
            "kv_append_fp8 rows=256 real=200 kv_heads={kvh} head_dim={hd}: mismatched bytes \
             (nan, saturated, subnormal, other)={bad:?} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    let every: Vec<u16> = (0..=u16::MAX).collect();
    let reversed: Vec<u16> = every.iter().rev().copied().collect();
    let (bad, [kc, vc, kb, vb, sb]) = append_exact(&device, (&every, &reversed), (256, 256, 1, HD));
    let ok = bad == [0; 4];
    eprintln!(
        "kv_append_fp8 every bf16 bit pattern: mismatched bytes (nan, saturated, subnormal, \
         other)={bad:?} {}",
        verdict(ok)
    );
    pass &= ok;

    let (kh, vh) = (down_bytes(&kc), down_bytes(&vc));
    let refusals: [((u32, u32, u32), u32, &str); 6] = [
        ((0, 1, 256), 64, "rows 0"),
        ((256, 0, 256), 64, "kv_heads 0"),
        ((256, 1, 0), 64, "head_dim 0"),
        ((256, 1, 252), 64, "head_dim 252"),
        ((256, 1, 256), 16, "block_size 16"),
        ((1 << 20, 8, 256), 64, "rows·cols above INT_MAX"),
    ];
    for ((r, h, d), block_size, what) in refusals {
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            kv_append_fp8(
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
        let untouched = down_bytes(&kc) == kh && down_bytes(&vc) == vh;
        eprintln!(
            "kv_append_fp8 refusal ({what}): {got:?} cache_untouched={untouched} {}",
            verdict(got == want && untouched)
        );
        pass &= got == want && untouched;
    }
    assert!(
        pass,
        "the e4m3 KV append is not bit-exact, or a refusal wrote"
    );
}

// ── The attention kernels over the e4m3 cache ──────────────────────

/// The prefill launchers' one signature (`kernels::attn_prefill_windowed_bf16`'s).
type PrefillLaunch = unsafe fn(
    (*const c_void, *mut c_void),
    (*const c_void, *const c_void),
    (*const c_void, *const c_void),
    (u32, u32, u32, u32),
    (u32, u32),
    (f32, u32),
    *mut c_void,
) -> Result<(), KernelError>;
/// The decode launchers' one signature (`kernels::attn_decode_windowed_bf16`'s).
type DecodeLaunch = unsafe fn(
    (*const c_void, *mut c_void),
    (*const c_void, *const c_void),
    (*const c_void, *const c_void),
    (u32, u32, u32),
    (u32, u32),
    (f32, u32),
    *mut c_void,
) -> Result<(), KernelError>;

/// One of the four attention kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Prefill,
    Decode,
    PrefillD512,
    DecodeD512,
}

impl Kernel {
    fn head_dim(self) -> usize {
        match self {
            Self::Prefill | Self::Decode => HD,
            Self::PrefillD512 | Self::DecodeD512 => D512,
        }
    }
}

/// One launch: the kernel, `(q_heads, kv_heads)`, `(P, L)`, the bucket's rows (1 for a decode),
/// the window (0: none) and the scale.
#[derive(Clone, Copy, Debug)]
struct Case {
    kernel: Kernel,
    heads: (usize, usize),
    seq: (usize, usize),
    rows: usize,
    window: u32,
    scale: f32,
}

impl Case {
    fn tokens(&self) -> usize {
        self.seq.0 + self.seq.1
    }
    fn decode(
        kernel: Kernel,
        heads: (usize, usize),
        kv_len: usize,
        window: u32,
        scale: f32,
    ) -> Self {
        Self {
            kernel,
            heads,
            seq: (kv_len - 1, 1),
            rows: 1,
            window,
            scale,
        }
    }
}

/// One case's inputs: the query rows in [−1/4, 1/4] at head dim 256 and [−1/8, 1/8] at 512 (so
/// `scale · Σ|qk| ≤ 64` at scale 1), the sequence's keys and values in [−1, 1].
fn inputs(c: &Case, seed: u64) -> [Vec<u16>; 3] {
    let (qh, kvh) = c.heads;
    let hd = c.kernel.head_dim();
    let shrink = (HD as f32) / (hd as f32);
    let q = (centred_gains(c.rows * qh * hd, seed).into_iter())
        .map(|b| bits(bf16::from_bits(b).to_f32() * shrink))
        .collect();
    [
        q,
        bf16_inputs(c.tokens() * kvh * hd, seed ^ 0x100),
        bf16_inputs(c.tokens() * kvh * hd, seed ^ 0x200),
    ]
}

/// A fresh e4m3 pool of `DEC_BLOCKS` blocks for `c`, every byte `fill`, then the sequence's keys
/// and values appended over `table`.
fn fp8_pool(
    device: &CudaDevice,
    c: &Case,
    (k, v): (&[u16], &[u16]),
    table: &[u32],
    fill: u8,
) -> (DeviceBuffer, DeviceBuffer) {
    let (kvh, hd) = (c.heads.1, c.kernel.head_dim());
    let bytes = vec![fill; DEC_BLOCKS * kvh * BS * hd];
    let (kc, vc) = (up_bytes(device, &bytes), up_bytes(device, &bytes));
    let sb = up_u32(device, &slots(table, c.tokens(), c.tokens()));
    let (kb, vb) = (up(device, k), up(device, v));
    append(
        (&kc, &vc),
        (&kb, &vb),
        &sb,
        (c.tokens(), kvh, hd),
        DEC_BLOCKS,
    )
    .expect("kv_append_fp8 fills the cache");
    (kc, vc)
}

/// The e4m3 cache `c`, widened to bf16 on the host (exact) and uploaded: the bf16 kernels' cache.
fn widened(device: &CudaDevice, c: &DeviceBuffer) -> DeviceBuffer {
    up(device, &e4m3_to_bf16_bits(&down_bytes(c)))
}

/// One launch's device inputs and output, uploaded once.
struct Launch {
    q: DeviceBuffer,
    out: DeviceBuffer,
    table: DeviceBuffer,
    sequence: DeviceBuffer,
}

impl Launch {
    fn new(device: &CudaDevice, c: &Case, q: &[u16], table: &[u32]) -> Self {
        let (qh, hd) = (c.heads.0, c.kernel.head_dim());
        Self {
            q: up(device, q),
            out: up(device, &vec![SENTINEL; c.rows * qh * hd]),
            table: up_u32(device, table),
            sequence: up_u32(device, &[c.seq.0 as u32, c.seq.1 as u32]),
        }
    }

    /// Enqueue `c`'s kernel, the fp8kv one or its bf16 reference, over `caches`.
    fn fire(
        &self,
        c: &Case,
        fp8: bool,
        (kc, vc): (&DeviceBuffer, &DeviceBuffer),
    ) -> Result<(), KernelError> {
        let ((qh, kvh), hd) = (c.heads, c.kernel.head_dim() as u32);
        let io = (self.q.ptr().cast_const(), self.out.ptr());
        let caches = (kc.ptr().cast_const(), vc.ptr().cast_const());
        let index = (
            self.table.ptr().cast_const(),
            self.sequence.ptr().cast_const(),
        );
        let pool = (BS as u32, DEC_BLOCKS as u32);
        let sw = (c.scale, c.window);
        let null = std::ptr::null_mut();
        let prefill: Option<PrefillLaunch> = match (c.kernel, fp8) {
            (Kernel::Prefill, true) => Some(attn_prefill_windowed_fp8kv),
            (Kernel::Prefill, false) => Some(attn_prefill_windowed_bf16),
            (Kernel::PrefillD512, true) => Some(attn_prefill_d512_fp8kv),
            (Kernel::PrefillD512, false) => Some(attn_prefill_d512_bf16),
            _ => None,
        };
        let decode: DecodeLaunch = match (c.kernel, fp8) {
            (Kernel::DecodeD512, true) => attn_decode_d512_fp8kv,
            (Kernel::DecodeD512, false) => attn_decode_d512_bf16,
            (_, true) => attn_decode_windowed_fp8kv,
            (_, false) => attn_decode_windowed_bf16,
        };
        // SAFETY: cudaMalloc pointers are 256-byte aligned; `q` and `out` hold rows·q_heads·hd
        // bf16, `out` its own allocation; the caches DEC_BLOCKS·kv_heads·64·hd elements each (e4m3
        // for the fp8kv kernel, bf16 for its reference); `table` covers the ceil((P+L)/64) blocks
        // the keys need, each below DEC_BLOCKS; `L ≥ 1`; null is the default stream.
        unsafe {
            match prefill {
                Some(f) => f(
                    io,
                    caches,
                    index,
                    (c.rows as u32, qh as u32, kvh as u32, hd),
                    pool,
                    sw,
                    null,
                ),
                None => decode(
                    io,
                    caches,
                    index,
                    (qh as u32, kvh as u32, hd),
                    pool,
                    sw,
                    null,
                ),
            }
        }
    }
}

/// `c`'s kernel (fp8kv or its bf16 reference) over `caches`; the output, downloaded.
fn launch(
    device: &CudaDevice,
    c: &Case,
    fp8: bool,
    q: &[u16],
    caches: (&DeviceBuffer, &DeviceBuffer),
    table: &[u32],
) -> Result<Vec<u16>, KernelError> {
    let l = Launch::new(device, c, q, table);
    l.fire(c, fp8, caches)?;
    Ok(l.out.download().expect("download the output"))
}

/// Each case's fp8kv output against its bf16 kernel's over the same cache widened on the host, bit
/// for bit. The pool is poisoned with `POISON` (an e4m3 NaN) and filled over a shuffled table, so
/// every kv_len off a multiple of 64 runs the partial tile's zeroing on NaN bytes.
fn bit_identical(device: &CudaDevice, what: &str, cases: &[Case]) -> bool {
    let mut pass = true;
    for (i, c) in cases.iter().enumerate() {
        let seed = 0x0F8C_5000 ^ ((i as u64) << 24) ^ c.tokens() as u64;
        let [q, k, v] = inputs(c, seed);
        let table = shuffled_blocks(seed);
        let need = &table[..c.tokens().div_ceil(BS)];
        let (k8, v8) = fp8_pool(device, c, (&k, &v), &table, POISON);
        let (kb, vb) = (widened(device, &k8), widened(device, &v8));
        let got = launch(device, c, true, &q, (&k8, &v8), need).expect("the fp8kv launch");
        let want = launch(device, c, false, &q, (&kb, &vb), need).expect("the bf16 launch");
        let mismatched = mismatched_bits(&got, &want);
        eprintln!(
            "{what} {c:?}: fp8kv vs bf16 on the widened cache mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    pass
}

/// T2, the head-dim-256 prefill: `tests/attention.rs`'s `CASES` (window 0) and `WINDOW_PREFILLS`
/// (16 over 8, window 1024), each bit-identical to `attn_prefill_bf16.cu` on the widened cache.
#[test]
fn the_fp8kv_prefill_is_bit_identical_to_the_bf16_prefill_on_the_widened_cache() {
    let device = h100();
    let causal = CASES.map(|(qh, kvh, rows, p, len)| Case {
        kernel: Kernel::Prefill,
        heads: (qh, kvh),
        seq: (p, len),
        rows,
        window: 0,
        scale: SCALE,
    });
    let windowed = WINDOW_PREFILLS.map(|(p, len, rows)| Case {
        kernel: Kernel::Prefill,
        heads: (16, 8),
        seq: (p, len),
        rows,
        window: WINDOW,
        scale: 1.0,
    });
    let cases: Vec<Case> = causal.into_iter().chain(windowed).collect();
    assert!(
        bit_identical(&device, "prefill d256", &cases),
        "an fp8kv prefill differs from the bf16 prefill on the widened cache"
    );
}

/// T2, the head-dim-256 decode: every `DEC_KV_LENS` at 8 over 2, 16 over 2 (Qwen3.6-35B), 24 over
/// 4 (Qwen3.8-27B) and 16 over 8, and `WINDOW_DECODES` (16 over 8, window 1024), each
/// bit-identical to `attn_decode_bf16.cu` on the widened cache.
#[test]
fn the_fp8kv_decode_is_bit_identical_to_the_bf16_decode_on_the_widened_cache() {
    let device = h100();
    let mut cases = Vec::new();
    for heads in [(8, 2), (16, 2), (24, 4), (16, 8)] {
        cases.extend(DEC_KV_LENS.map(|n| Case::decode(Kernel::Decode, heads, n, 0, SCALE)));
    }
    cases.extend(WINDOW_DECODES.map(|n| Case::decode(Kernel::Decode, (16, 8), n, WINDOW, 1.0)));
    assert!(
        bit_identical(&device, "decode d256", &cases),
        "an fp8kv decode differs from the bf16 decode on the widened cache"
    );
}

/// T2, the head-dim-512 prefill: `D512_PREFILLS` (16 over 2), each bit-identical to
/// `attn_prefill_d512_bf16.cu` on the widened cache.
#[test]
fn the_fp8kv_d512_prefill_is_bit_identical_to_the_bf16_one_on_the_widened_cache() {
    let device = h100();
    let cases = D512_PREFILLS.map(|(p, len)| Case {
        kernel: Kernel::PrefillD512,
        heads: (16, 2),
        seq: (p, len),
        rows: len.div_ceil(128) * 128,
        window: 0,
        scale: 1.0,
    });
    assert!(
        bit_identical(&device, "prefill d512", &cases),
        "an fp8kv D = 512 prefill differs from the bf16 one on the widened cache"
    );
}

/// T2, the head-dim-512 decode: `D512_DECODES` (16 over 2), each bit-identical to
/// `attn_decode_d512_bf16.cu` on the widened cache.
#[test]
fn the_fp8kv_d512_decode_is_bit_identical_to_the_bf16_one_on_the_widened_cache() {
    let device = h100();
    let cases = D512_DECODES.map(|n| Case::decode(Kernel::DecodeD512, (16, 2), n, 0, 1.0));
    assert!(
        bit_identical(&device, "decode d512", &cases),
        "an fp8kv D = 512 decode differs from the bf16 one on the widened cache"
    );
}

/// One partial-last-tile case of each kernel at `kv_len`: a prefill of the whole sequence in one
/// bucket, a decode of its last key.
fn partial_cases(kv_len: usize) -> [Case; 4] {
    let prefill = |kernel, heads, scale| Case {
        kernel,
        heads,
        seq: (0, kv_len),
        rows: kv_len.div_ceil(128) * 128,
        window: 0,
        scale,
    };
    [
        prefill(Kernel::Prefill, (8, 2), SCALE),
        Case::decode(Kernel::Decode, (8, 2), kv_len, 0, SCALE),
        prefill(Kernel::PrefillD512, (16, 2), 1.0),
        Case::decode(Kernel::DecodeD512, (16, 2), kv_len, 0, 1.0),
    ]
}

/// T3: each kernel at a partial last tile (kv_len 65, 321, 1025), its pool filled with 0x7F, then
/// 0xFF (both e4m3 NaN), then 0x00 before the append: every output finite, and the three
/// bit-identical. The kernels never use a byte no append wrote.
#[test]
fn a_poisoned_pool_gives_finite_bits_identical_to_a_zeroed_pool() {
    let device = h100();
    let mut pass = true;
    for kv_len in [65usize, 321, 1025] {
        for c in partial_cases(kv_len) {
            let seed = 0x0F8C_3000 ^ kv_len as u64;
            let [q, k, v] = inputs(&c, seed);
            let table = shuffled_blocks(seed);
            let need = &table[..kv_len.div_ceil(BS)];
            let outs: Vec<Vec<u16>> = [0x7Fu8, 0xFF, 0x00]
                .into_iter()
                .map(|fill| {
                    let (kc, vc) = fp8_pool(&device, &c, (&k, &v), &table, fill);
                    launch(&device, &c, true, &q, (&kc, &vc), need).expect("the fp8kv launch")
                })
                .collect();
            let finite = (outs.iter().flatten()).all(|&b| bf16::from_bits(b).to_f32().is_finite());
            let mismatched =
                mismatched_bits(&outs[1], &outs[0]) + mismatched_bits(&outs[2], &outs[0]);
            let ok = finite && mismatched == 0;
            eprintln!(
                "poisoned pool {c:?}: finite={finite} 0xFF,0x00 vs 0x7F mismatched_bits={mismatched} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "an fp8kv kernel read a byte no append wrote");
}

/// T4: each kernel three times over one pool, then once over a second pool written through another
/// shuffled table: all four outputs bit-identical. No atomics, and the order of additions is a
/// function of the shape alone, not of which blocks the table names.
#[test]
fn the_fp8kv_kernels_are_deterministic() {
    let device = h100();
    let mut pass = true;
    let cases = [
        Case {
            kernel: Kernel::Prefill,
            heads: (16, 2),
            seq: (300, 128),
            rows: 128,
            window: 0,
            scale: SCALE,
        },
        Case::decode(Kernel::Decode, (16, 2), 1000, 0, SCALE),
        Case {
            kernel: Kernel::PrefillD512,
            heads: (16, 2),
            seq: (300, 512),
            rows: 512,
            window: 0,
            scale: 1.0,
        },
        Case::decode(Kernel::DecodeD512, (16, 2), 321, 0, 1.0),
    ];
    for c in cases {
        let seed = 0x0F8C_4000 ^ c.tokens() as u64;
        let [q, k, v] = inputs(&c, seed);
        let need = c.tokens().div_ceil(BS);
        let mut outs = Vec::new();
        for (table_seed, repeats) in [(seed, 3), (seed ^ 1, 1)] {
            let table = shuffled_blocks(table_seed);
            let (kc, vc) = fp8_pool(&device, &c, (&k, &v), &table, POISON);
            for _ in 0..repeats {
                let out = launch(&device, &c, true, &q, (&kc, &vc), &table[..need]);
                outs.push(out.expect("the fp8kv launch"));
            }
        }
        let mismatched: usize = (outs[1..].iter())
            .map(|o| mismatched_bits(o, &outs[0]))
            .sum();
        eprintln!(
            "determinism {c:?}: 2 repeats and a second table mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    assert!(pass, "an fp8kv kernel is not deterministic");
}

/// T5: `tests/attention.rs`'s refusal lists (the prefill's, the decode's, a group of 17, a window
/// off the block, the D = 512 launchers'), pointed at the fp8kv launchers: each returns its shape
/// code and leaves `out` untouched.
#[test]
fn the_fp8kv_launchers_refuse_each_bad_shape() {
    let device = h100();
    let dummy = up(&device, &vec![0u16; 64 * 1024]);
    let table = up_u32(&device, &[0; 8]);
    let seq = up_u32(&device, &[0, 128]);
    let mut pass = true;
    // (rows, q_heads, kv_heads, head_dim, block_size, num_blocks, scale, window, what)
    type PrefillRefusal = ((u32, u32, u32, u32), (u32, u32), (f32, u32), &'static str);
    let prefill: [(PrefillLaunch, &str, PrefillRefusal); 13] = [
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 8, 2, 128), (64, 16), (SCALE, 0), "head_dim 128"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((192, 8, 2, 256), (64, 16), (SCALE, 0), "rows 192"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            (
                (128, 8, 3, 256),
                (64, 16),
                (SCALE, 0),
                "q_heads % kv_heads ≠ 0",
            ),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 8, 2, 256), (16, 16), (SCALE, 0), "block_size 16"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 8, 2, 256), (64, 0), (SCALE, 0), "num_blocks 0"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 8, 2, 256), (64, 16), (f32::NAN, 0), "scale NaN"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 16, 8, 256), (64, 160), (1.0, 1000), "window 1000"),
        ),
        (
            attn_prefill_windowed_fp8kv,
            "prefill",
            ((128, 16, 8, 256), (64, 160), (1.0, 32), "window 32"),
        ),
        (
            attn_prefill_d512_fp8kv,
            "prefill d512",
            (
                (128, 16, 2, 256),
                (64, 160),
                (1.0, 0),
                "head_dim 256 to D = 512",
            ),
        ),
        (
            attn_prefill_d512_fp8kv,
            "prefill d512",
            ((128, 16, 2, 512), (16, 160), (1.0, 0), "block_size 16"),
        ),
        (
            attn_prefill_d512_fp8kv,
            "prefill d512",
            ((64, 16, 2, 512), (64, 160), (1.0, 0), "rows 64"),
        ),
        (
            attn_prefill_d512_fp8kv,
            "prefill d512",
            ((128, 16, 2, 512), (64, 160), (f32::NAN, 0), "scale NaN"),
        ),
        (
            attn_prefill_d512_fp8kv,
            "prefill d512",
            ((128, 15, 2, 512), (64, 160), (1.0, 0), "q_heads 15 over 2"),
        ),
    ];
    for (f, kernel, ((rows, qh, kvh, hd), (block_size, num_blocks), sw, what)) in prefill {
        let out = up(&device, &[SENTINEL; 64]);
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            f(
                (dummy.ptr(), out.ptr()),
                (dummy.ptr(), dummy.ptr()),
                (table.ptr(), seq.ptr()),
                (rows, qh, kvh, hd),
                (block_size, num_blocks),
                sw,
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
        let kept = out.download().expect("out").iter().all(|&b| b == SENTINEL);
        eprintln!(
            "{kernel} fp8kv refusal ({what}): {got:?} out_untouched={kept} {}",
            verdict(got == want && kept)
        );
        pass &= got == want && kept;
    }
    type DecodeRefusal = ((u32, u32, u32), (u32, u32), (f32, u32), &'static str);
    let decode: [(DecodeLaunch, &str, DecodeRefusal); 14] = [
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 2, 128), (64, 16), (SCALE, 0), "head_dim 128"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 3, 256), (64, 16), (SCALE, 0), "q_heads % kv_heads ≠ 0"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 0, 256), (64, 16), (SCALE, 0), "kv_heads 0"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 2, 256), (16, 16), (SCALE, 0), "block_size 16"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 2, 256), (64, 0), (SCALE, 0), "num_blocks 0"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 2, 256), (64, 16), (f32::NAN, 0), "scale NaN"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((8, 2, 256), (64, 16), (0.0, 0), "scale 0"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((34, 2, 256), (64, 16), (SCALE, 0), "a group of 17"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((16, 8, 256), (64, 160), (1.0, 1000), "window 1000"),
        ),
        (
            attn_decode_windowed_fp8kv,
            "decode",
            ((16, 8, 256), (64, 160), (1.0, 32), "window 32"),
        ),
        (
            attn_decode_d512_fp8kv,
            "decode d512",
            ((16, 2, 256), (64, 160), (1.0, 0), "head_dim 256 to D = 512"),
        ),
        (
            attn_decode_d512_fp8kv,
            "decode d512",
            ((34, 2, 512), (64, 160), (1.0, 0), "a group of 17"),
        ),
        (
            attn_decode_d512_fp8kv,
            "decode d512",
            ((16, 2, 512), (16, 160), (1.0, 0), "block_size 16"),
        ),
        (
            attn_decode_d512_fp8kv,
            "decode d512",
            ((16, 2, 512), (64, 160), (f32::NAN, 0), "scale NaN"),
        ),
    ];
    for (f, kernel, ((qh, kvh, hd), (block_size, num_blocks), sw, what)) in decode {
        let out = up(&device, &[SENTINEL; 64]);
        // SAFETY: refused before any pointer is touched.
        let got = unsafe {
            f(
                (dummy.ptr(), out.ptr()),
                (dummy.ptr(), dummy.ptr()),
                (table.ptr(), seq.ptr()),
                (qh, kvh, hd),
                (block_size, num_blocks),
                sw,
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
        let kept = out.download().expect("out").iter().all(|&b| b == SENTINEL);
        eprintln!(
            "{kernel} fp8kv refusal ({what}): {got:?} out_untouched={kept} {}",
            verdict(got == want && kept)
        );
        pass &= got == want && kept;
    }
    assert!(pass, "an fp8kv refusal launched or wrote");
}

/// T6, recorded, not gated: the decode time of each fp8kv kernel and of its bf16 reference, at
/// kv_len 1 to 8192, for Qwen3.6-35B's 16 over 2, Qwen3.8-27B's 24 over 4, Gemma's sliding 16 over
/// 8 (window 1024) and its global 16 over 2 at head dim 512. 50 launches after a warm-up each.
#[test]
fn the_fp8kv_decode_time_is_recorded() {
    const LAUNCHES: u32 = 50;
    let device = h100();
    let shapes = [
        (Kernel::Decode, (16, 2), 0, SCALE),
        (Kernel::Decode, (24, 4), 0, SCALE),
        (Kernel::Decode, (16, 8), WINDOW, 1.0),
        (Kernel::DecodeD512, (16, 2), 0, 1.0),
    ];
    for (kernel, heads, window, scale) in shapes {
        for kv_len in [1usize, 64, 256, 1024, 2176, 4096, 8192] {
            let c = Case::decode(kernel, heads, kv_len, window, scale);
            let seed = 0x0F8C_6000 ^ kv_len as u64;
            let [q, k, v] = inputs(&c, seed);
            let table = shuffled_blocks(seed);
            let (k8, v8) = fp8_pool(&device, &c, (&k, &v), &table, POISON);
            let (kb, vb) = (widened(&device, &k8), widened(&device, &v8));
            let l = Launch::new(&device, &c, &q, &table[..kv_len.div_ceil(BS)]);
            let mut us = [0.0f64; 2];
            for (i, (fp8, caches)) in [(true, (&k8, &v8)), (false, (&kb, &vb))]
                .into_iter()
                .enumerate()
            {
                l.fire(&c, fp8, caches).expect("the warm-up");
                l.out.download().expect("wait for the warm-up");
                let start = std::time::Instant::now();
                for _ in 0..LAUNCHES {
                    l.fire(&c, fp8, caches).expect("a timed launch");
                }
                l.out.download().expect("wait for the launches");
                us[i] = start.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES);
            }
            eprintln!(
                "decode time {kernel:?} heads={}/{} window={window} kv_len={kv_len}: fp8kv \
                 {:.1} us bf16 {:.1} us ({:+.1}%)",
                heads.0,
                heads.1,
                us[0],
                us[1],
                (us[0] / us[1] - 1.0) * 100.0
            );
        }
    }
}
