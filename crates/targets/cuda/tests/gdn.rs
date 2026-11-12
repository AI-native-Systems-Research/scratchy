// SPDX-License-Identifier: Apache-2.0
//! The Gated-DeltaNet kernels on the H100, against f64 CPU references and exact probes: the causal
//! conv1d (`kernels/gdn_conv1d.cu`), the gating (`kernels/gdn_gating.cu`), the recurrent delta-rule
//! scan (`kernels/gdn_scan.cu`) and the gated norm (`kernels/gdn_rms_norm_gated.cu`), launched in
//! the order a `GatedDeltaNet` step launches them, over a GDN state pool laid out by
//! `GdnStateLayout`; and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. Each test owns its device buffers and calls the launchers directly, with no
//! worker. Every case prints its line before a test fails, so one run reports all the numbers.
//!
//! Every reference reads the values the GPU read: bf16 inputs widened exactly, the f32 bundle, and
//! the GPU's own f32 scratch for a stage's inputs, so each check measures one kernel's error
//! (`common::numerics`' bounds). Every state slot a test does not start from is NaN-poisoned, and
//! every padding row of every input is NaN: a kernel that loads a fresh slot, or reads a padding
//! row, puts a NaN where a check sees it.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
    pub mod watchdog;
}

use common::numerics::{
    F32Comparison, GDN_CHUNK_REL, GDN_CHUNK_STEP, GDN_SCAN_STEP, GdnDims, ScanInputs,
    ScanReference, ScanVariant, bf16_in, compare_f32, compare_rows, f32_from_halves, f32_inputs,
    f32_to_halves, gdn_chunk_bound, gdn_conv_bound, gdn_conv_reference, gdn_gating_bound,
    gdn_gating_reference, gdn_norm_reference, gdn_scan_bound, gdn_scan_reference, widen,
};
use scratchy_subtile::gdn_state_layout::GdnStateLayout;
use scratchy_target_cuda::kernels::{
    KernelError, gdn_chunk_scan, gdn_chunk_scan_part, gdn_conv1d, gdn_gating, gdn_rms_norm_gated,
    gdn_scan,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::gdn_scale;

/// qwen3.5-0.8b's linear attention: 16 key heads, 16 value heads.
const G08: GdnDims = GdnDims {
    nk: 16,
    nv: 16,
    head_k: 128,
    head_v: 128,
    conv_kernel: 4,
};
/// qwen3.5-9b's: 32 value heads over 16 key heads, value head `h` reading key head `h / 2`.
const G9: GdnDims = GdnDims { nv: 32, ..G08 };
/// Qwen3.8-27B's: 48 value heads over 16 key heads, value head `h` reading key head `h / 3`.
const G27: GdnDims = GdnDims { nv: 48, ..G08 };
/// The chunk lengths every stage is checked at.
const LENGTHS: [usize; 8] = [1, 2, 63, 64, 65, 300, 1024, 2048];
/// The lengths only the 9b geometry is also checked at: qwen3.5-9b's largest prefill bucket.
const LENGTHS_9B: [usize; 1] = [4096];
/// Padding rows past every chunk's real rows: NaN in every input.
const PAD: usize = 3;
/// The pool's slots in every test.
const SLOTS: u32 = 4;
/// The gated norm's eps, qwen3.5's `rms_norm_eps`.
const EPS: f32 = 1e-6;
/// An f32 no kernel wrote, or a poisoned state entry: a NaN.
const NAN_F32: u32 = 0x7FC0_0001;
/// A bf16 padding-row input, or an output no kernel wrote: a NaN.
const NAN_BF16: u16 = 0x7FC1;

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

fn up_f32(device: &CudaDevice, host: &[f32]) -> DeviceBuffer {
    up(device, &f32_to_halves(host))
}

/// `len` f32 of the bit pattern `bits`.
fn up_f32_bits(device: &CudaDevice, len: usize, bits: u32) -> DeviceBuffer {
    up_f32(device, &vec![f32::from_bits(bits); len])
}

fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let halves: Vec<u16> = (host.iter())
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    up(device, &halves)
}

fn down_f32(b: &DeviceBuffer) -> Vec<f32> {
    f32_from_halves(&b.download().expect("download"))
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// The bit patterns of `v`.
fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The GDN state pool of one layer: `SLOTS` slots of conv ring and ssm state, laid out by
/// `GdnStateLayout`, every entry `fill` at first.
struct Pool {
    layout: GdnStateLayout,
    conv: DeviceBuffer,
    ssm: DeviceBuffer,
}

impl Pool {
    fn new(device: &CudaDevice, d: GdnDims, fill: u32) -> Self {
        let layout = GdnStateLayout::new(
            SLOTS,
            d.nk as u32,
            d.nv as u32,
            d.head_k as u32,
            d.head_v as u32,
            d.conv_kernel as u32,
        );
        Self {
            layout,
            conv: up_f32_bits(device, layout.conv_buffer_elems(), fill),
            ssm: up_f32_bits(device, layout.ssm_buffer_elems(), fill),
        }
    }

    fn conv_slot(&self, slot: u32) -> Vec<f32> {
        let all = down_f32(&self.conv);
        all[self.layout.conv_slot_offset(slot)..][..self.layout.conv_elems_per_slot()].to_vec()
    }

    fn ssm_slot(&self, slot: u32) -> Vec<f32> {
        let all = down_f32(&self.ssm);
        all[self.layout.ssm_slot_offset(slot)..][..self.layout.ssm_elems_per_slot()].to_vec()
    }

    /// Every slot's conv ring and ssm state, as bits.
    fn snapshot(&self) -> (Vec<u32>, Vec<u32>) {
        (
            bits_of(&down_f32(&self.conv)),
            bits_of(&down_f32(&self.ssm)),
        )
    }

    /// The bits of every slot but `slot` that differ between two snapshots.
    fn others_changed(&self, before: &(Vec<u32>, Vec<u32>), slot: u32) -> usize {
        let after = self.snapshot();
        let l = self.layout;
        let outside = |i: usize, per: usize, at: usize| i < at || i >= at + per;
        let conv = (before.0.iter().zip(&after.0).enumerate())
            .filter(|&(i, (x, y))| {
                outside(i, l.conv_elems_per_slot(), l.conv_slot_offset(slot)) && x != y
            })
            .count();
        let ssm = (before.1.iter().zip(&after.1).enumerate())
            .filter(|&(i, (x, y))| {
                outside(i, l.ssm_elems_per_slot(), l.ssm_slot_offset(slot)) && x != y
            })
            .count();
        conv + ssm
    }

    /// Slot `slot` set to `conv` and `ssm`, the rest kept: the pool re-uploaded.
    fn write_slot(&mut self, device: &CudaDevice, slot: u32, conv: &[f32], ssm: &[f32]) {
        let l = self.layout;
        let mut c = down_f32(&self.conv);
        c[l.conv_slot_offset(slot)..][..l.conv_elems_per_slot()].copy_from_slice(conv);
        let mut s = down_f32(&self.ssm);
        s[l.ssm_slot_offset(slot)..][..l.ssm_elems_per_slot()].copy_from_slice(ssm);
        self.conv = up_f32(device, &c);
        self.ssm = up_f32(device, &s);
    }
}

/// A sequence's bf16 inputs, token-major: `qkv [T, conv_dim]`, `z [T, value_dim]`, `a` and `b
/// [T, nv]`, in the issue's ranges.
struct Inputs {
    qkv: Vec<u16>,
    z: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
}

fn inputs(d: GdnDims, tokens: usize, seed: u64) -> Inputs {
    Inputs {
        qkv: bf16_in(tokens * d.conv_dim(), seed ^ 0x01, (-1.0, 1.0)),
        z: bf16_in(tokens * d.value_dim(), seed ^ 0x02, (-1.0, 1.0)),
        a: bf16_in(tokens * d.nv, seed ^ 0x03, (-2.0, 2.0)),
        b: bf16_in(tokens * d.nv, seed ^ 0x04, (-2.0, 2.0)),
    }
}

/// A layer's f32 bundle: the conv taps `[conv_dim, K]`, `a_log`, `dt_bias` `[nv]`, the norm's gain
/// `[head_v]`.
struct Bundle {
    w: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

fn bundle(d: GdnDims, seed: u64) -> Bundle {
    Bundle {
        w: f32_inputs(d.conv_dim() * d.conv_kernel, seed ^ 0x05, (-0.5, 0.5)),
        a_log: f32_inputs(d.nv, seed ^ 0x06, (-1.0, 0.5)),
        dt_bias: f32_inputs(d.nv, seed ^ 0x07, (-0.5, 0.5)),
        norm: f32_inputs(d.head_v, seed ^ 0x08, (0.5, 1.5)),
    }
}

struct BundleBufs {
    w: DeviceBuffer,
    a_log: DeviceBuffer,
    dt_bias: DeviceBuffer,
    norm: DeviceBuffer,
}

fn upload_bundle(device: &CudaDevice, b: &Bundle) -> BundleBufs {
    BundleBufs {
        w: up_f32(device, &b.w),
        a_log: up_f32(device, &b.a_log),
        dt_bias: up_f32(device, &b.dt_bias),
        norm: up_f32(device, &b.norm),
    }
}

/// One chunk: tokens `from .. from + q_len` of a sequence's inputs, in a bucket of `rows`, every
/// padding row NaN.
struct Chunk {
    q_len: usize,
    rows: usize,
    qkv: Vec<u16>,
    z: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
}

fn chunk(x: &Inputs, d: GdnDims, from: usize, q_len: usize, rows: usize) -> Chunk {
    let take = |v: &[u16], cols: usize| {
        let mut out = v[from * cols..(from + q_len) * cols].to_vec();
        out.resize(rows * cols, NAN_BF16);
        out
    };
    Chunk {
        q_len,
        rows,
        qkv: take(&x.qkv, d.conv_dim()),
        z: take(&x.z, d.value_dim()),
        a: take(&x.a, d.nv),
        b: take(&x.b, d.nv),
    }
}

/// One step's outputs, each `rows` rows: the f32 scratch the four launches pass along, and the
/// bf16 output.
struct StepOut {
    conv_out: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    o: Vec<f32>,
    out: Vec<u16>,
}

/// The four launches of one GDN step, on the default stream, in order: conv1d, gating, scan, gated
/// norm, over slot `slot` of `pool` (from zero, the slot not read, when `fresh`). Every scratch
/// element and output starts NaN.
fn run_step(
    device: &CudaDevice,
    pool: &Pool,
    bb: &BundleBufs,
    d: GdnDims,
    c: &Chunk,
    slot: (u32, bool),
) -> Result<StepOut, KernelError> {
    run_step_with(device, pool, bb, d, c, slot, Scan::Recurrent)
}

/// Which scan a step's third launch is: today's recurrent `gdn_scan`, or the chunked
/// `gdn_chunk_scan` (FP8 S8 F6a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scan {
    Recurrent,
    Chunked,
}

/// [`run_step`] with `scan` as its third launch.
fn run_step_with(
    device: &CudaDevice,
    pool: &Pool,
    bb: &BundleBufs,
    d: GdnDims,
    c: &Chunk,
    (slot, fresh): (u32, bool),
    scan: Scan,
) -> Result<StepOut, KernelError> {
    let rows = c.rows;
    let seq = up_u32(device, &[slot, c.q_len as u32, u32::from(fresh), 0]);
    let (qkv, z, a, b) = (
        up(device, &c.qkv),
        up(device, &c.z),
        up(device, &c.a),
        up(device, &c.b),
    );
    let conv_out = up_f32_bits(device, rows * d.conv_dim(), NAN_F32);
    let g = up_f32_bits(device, rows * d.nv, NAN_F32);
    let beta = up_f32_bits(device, rows * d.nv, NAN_F32);
    let o = up_f32_bits(device, rows * d.value_dim(), NAN_F32);
    let out = up(device, &vec![NAN_BF16; rows * d.value_dim()]);
    let l = pool.layout;
    let (r, cd, nv, hv) = (
        rows as u32,
        d.conv_dim() as u32,
        d.nv as u32,
        d.head_v as u32,
    );
    let null = std::ptr::null_mut();
    // SAFETY: cudaMalloc pointers are 256-byte aligned; every buffer holds the elements its
    // launcher's contract names for `rows` rows of this geometry; the pool holds `SLOTS` slots of
    // its layout and `slot < SLOTS`; `1 ≤ q_len ≤ rows`; no output overlaps another buffer; null
    // is the default stream, which runs the launches in order and the downloads below wait for.
    unsafe {
        gdn_conv1d(
            (qkv.ptr(), bb.w.ptr()),
            (pool.conv.ptr(), l.conv_elems_per_slot() as u64),
            seq.ptr(),
            conv_out.ptr(),
            (r, cd, d.conv_kernel as u32),
            null,
        )?;
        gdn_gating(
            (a.ptr(), b.ptr()),
            (bb.a_log.ptr(), bb.dt_bias.ptr()),
            seq.ptr(),
            (g.ptr(), beta.ptr()),
            (r, nv),
            null,
        )?;
        let launch = match scan {
            Scan::Recurrent => gdn_scan,
            Scan::Chunked => gdn_chunk_scan,
        };
        launch(
            (conv_out.ptr(), g.ptr(), beta.ptr()),
            (pool.ssm.ptr(), l.ssm_elems_per_slot() as u64),
            seq.ptr(),
            o.ptr(),
            (r, d.nk as u32, nv),
            (d.head_k as u32, hv, gdn_scale(d.head_k as u32)),
            null,
        )?;
        gdn_rms_norm_gated(
            (o.ptr(), z.ptr(), bb.norm.ptr()),
            seq.ptr(),
            out.ptr(),
            (r, nv, hv),
            EPS,
            null,
        )?;
    }
    Ok(StepOut {
        conv_out: down_f32(&conv_out),
        g: down_f32(&g),
        beta: down_f32(&beta),
        o: down_f32(&o),
        out: out.download().expect("download out"),
    })
}

/// How many of `v`'s elements from `from` on are not the NaN no kernel writes.
fn written_past(v: &[f32], from: usize) -> usize {
    v[from..].iter().filter(|x| x.to_bits() != NAN_F32).count()
}

/// The conv's checks for one step: `conv_out`'s real rows within bound of the reference after
/// `ring` (zeros on a fresh start), its padding rows unwritten, and the slot's new ring bit-exact.
fn check_conv(
    what: &str,
    d: GdnDims,
    c: &Chunk,
    w: &[f32],
    ring: &[f64],
    s: &StepOut,
    after: &[f32],
) -> bool {
    let cd = d.conv_dim();
    let x = widen(&c.qkv[..c.q_len * cd]);
    let (refs, new_ring) =
        gdn_conv_reference(&x, &to_f64(w), ring, (c.q_len, cd, d.conv_kernel), false);
    let r: Vec<f64> = refs.iter().map(|p| p.0).collect();
    let cmp = compare_f32(
        &s.conv_out,
        &r,
        |i| gdn_conv_bound(refs[i].0, refs[i].1),
        |_| true,
    );
    let padding = written_past(&s.conv_out, c.q_len * cd);
    let ring_bits = (after.iter().zip(&new_ring))
        .filter(|(g, r)| g.to_bits() != (**r as f32).to_bits())
        .count();
    let ok = cmp.pass() && padding == 0 && ring_bits == 0;
    eprintln!(
        "{what} conv1d L={}: {} padding_written={padding} ring_mismatched_bits={ring_bits} {}",
        c.q_len,
        cmp.line(),
        verdict(ok)
    );
    ok
}

/// The gating's checks for one step: `g` and `β` within bound, their padding rows unwritten.
fn check_gating(what: &str, d: GdnDims, c: &Chunk, bu: &Bundle, s: &StepOut) -> bool {
    let n = c.q_len * d.nv;
    let (g, beta) = gdn_gating_reference(
        &widen(&c.a[..n]),
        &widen(&c.b[..n]),
        &to_f64(&bu.a_log),
        &to_f64(&bu.dt_bias),
        d.nv,
        true,
    );
    let gc = compare_f32(&s.g, &g, |i| gdn_gating_bound(g[i]), |_| true);
    let bc = compare_f32(&s.beta, &beta, |i| gdn_gating_bound(beta[i]), |_| true);
    let padding = written_past(&s.g, n) + written_past(&s.beta, n);
    let ok = gc.pass() && bc.pass() && padding == 0;
    eprintln!(
        "{what} gating L={}: g {} beta {} padding_written={padding} {}",
        c.q_len,
        gc.line(),
        bc.line(),
        verdict(ok)
    );
    ok
}

/// The scan's reference for one step from the GPU's own conv_out, g and β.
fn scan_reference(
    d: GdnDims,
    s: &StepOut,
    q_len: usize,
    start: (&[f64], Option<&[f64]>, usize),
    variant: ScanVariant,
) -> ScanReference {
    let (cd, nv) = (d.conv_dim(), d.nv);
    let conv_out = to_f64(&s.conv_out[..q_len * cd]);
    let (g, beta) = (to_f64(&s.g[..q_len * nv]), to_f64(&s.beta[..q_len * nv]));
    gdn_scan_reference(
        &ScanInputs {
            conv_out: &conv_out,
            g: &g,
            beta: &beta,
            state: start.0,
            smax: start.1,
            t0: start.2,
            q_len,
        },
        d,
        variant,
    )
}

/// `o` against a scan reference under the long-recurrence bound, over the elements `keep` admits.
fn compare_o(s: &StepOut, r: &ScanReference, keep: impl Fn(usize) -> bool) -> F32Comparison {
    let refs: Vec<f64> = r.o.iter().map(|p| p.0).collect();
    compare_f32(&s.o, &refs, |i| gdn_scan_bound(r.o[i].0, r.o[i].1), keep)
}

/// The scan's checks for one step: `o` within the long-recurrence bound, its padding rows
/// unwritten, and the slot's state within `2⁻¹⁷·(t_last+1)·Smax`. Prints the worst ratio and
/// the linear term at the worst element. Returns the reference, for a continuation's.
fn check_scan(
    what: &str,
    d: GdnDims,
    s: &StepOut,
    q_len: usize,
    start: (&[f64], Option<&[f64]>, usize),
    after: &[f32],
) -> (bool, ScanReference) {
    let r = scan_reference(d, s, q_len, start, ScanVariant::Real);
    let cmp = compare_o(s, &r, |_| true);
    let linear = GDN_SCAN_STEP * r.o.get(cmp.worst).map_or(0.0, |p| p.1);
    let padding = written_past(&s.o, q_len * d.value_dim());
    let hk = d.head_k;
    let st = compare_f32(
        after,
        &r.state,
        |i| GDN_SCAN_STEP * r.state_scale[i / hk],
        |_| true,
    );
    let ok = cmp.pass() && st.pass() && padding == 0;
    eprintln!(
        "{what} scan L={q_len} t0={}: o {} linear_term_at_worst={linear:.4e} state {} \
         padding_written={padding} {}",
        start.2,
        cmp.line(),
        st.line(),
        verdict(ok)
    );
    (ok, r)
}

/// The gated norm's checks for one step: real rows within the row bound of the reference from the
/// GPU's `o`, padding rows exactly `+0.0`.
fn check_norm(what: &str, d: GdnDims, c: &Chunk, bu: &Bundle, s: &StepOut) -> bool {
    let vd = d.value_dim();
    let n = c.q_len * vd;
    let refs = gdn_norm_reference(
        &to_f64(&s.o[..n]),
        &widen(&c.z[..n]),
        &to_f64(&bu.norm),
        (c.q_len, d.nv, d.head_v),
        f64::from(EPS),
        true,
    );
    let cmp = compare_rows(&s.out[..n], &refs, vd);
    let not_zero = s.out[n..].iter().filter(|&&b| b != 0).count();
    let ok = cmp.pass() && not_zero == 0;
    eprintln!(
        "{what} gated_norm L={}: {} padding_rows_not_plus_zero={not_zero} {}",
        c.q_len,
        cmp.line(),
        verdict(ok)
    );
    ok
}

/// Where a step's sequence starts: the conv ring and the ssm state before it (zeros on a fresh
/// start), the running `Smax` of a continuation, and its first token within the sequence.
struct Start<'a> {
    ring: &'a [f64],
    state: &'a [f64],
    smax: Option<&'a [f64]>,
    t0: usize,
}

impl Start<'_> {
    fn scan(&self) -> (&[f64], Option<&[f64]>, usize) {
        (self.state, self.smax, self.t0)
    }
}

/// Every stage of one step from its own GPU inputs (conv, gating, scan, norm), the step having
/// run over `slot` of `pool`.
fn check_step(
    what: &str,
    (d, bu): (GdnDims, &Bundle),
    (c, s): (&Chunk, &StepOut),
    start: &Start<'_>,
    (pool, slot): (&Pool, u32),
) -> (bool, ScanReference) {
    let mut ok = check_conv(what, d, c, &bu.w, start.ring, s, &pool.conv_slot(slot));
    ok &= check_gating(what, d, c, bu, s);
    let (scan_ok, r) = check_scan(what, d, s, c.q_len, start.scan(), &pool.ssm_slot(slot));
    ok &= scan_ok;
    ok &= check_norm(what, d, c, bu, s);
    (ok, r)
}

/// A fresh start of geometry `d`: zero ring and state.
fn zeros(d: GdnDims) -> (Vec<f64>, Vec<f64>) {
    (
        vec![0.0; d.conv_dim() * (d.conv_kernel - 1)],
        vec![0.0; d.value_dim() * d.head_k],
    )
}

fn name(d: GdnDims) -> &'static str {
    match d.nv / d.nk {
        1 => "0.8b",
        2 => "9b",
        _ => "27b",
    }
}

// ── Every stage at every length, both geometries, fresh ───────────

/// Fresh in slot 1 of a pool NaN-poisoned everywhere: every stage within bound at every length
/// (the 9b geometry also at [`LENGTHS_9B`]), the padding rows unread and unwritten, and the other
/// slots bit-unchanged.
#[test]
fn every_gdn_stage_matches_its_reference_at_every_length() {
    let device = h100();
    let mut pass = true;
    for (d, extra) in [(G08, &[][..]), (G9, &LENGTHS_9B[..]), (G27, &[][..])] {
        let bu = bundle(d, 0x6D00);
        let bb = upload_bundle(&device, &bu);
        // A longer sequence of the same seed: its first 2048 tokens are the shorter one's.
        let tokens = LENGTHS
            .iter()
            .chain(extra)
            .copied()
            .max()
            .expect("a length");
        let x = inputs(d, tokens, 0x6D01);
        let (ring, state) = zeros(d);
        for &l in LENGTHS.iter().chain(extra) {
            let pool = Pool::new(&device, d, NAN_F32);
            let before = pool.snapshot();
            let c = chunk(&x, d, 0, l, l + PAD);
            let s = run_step(&device, &pool, &bb, d, &c, (1, true)).expect("the step launches");
            let start = Start {
                ring: &ring,
                state: &state,
                smax: None,
                t0: 0,
            };
            let what = format!("{} fresh", name(d));
            let (ok, _) = check_step(&what, (d, &bu), (&c, &s), &start, (&pool, 1));
            let others = pool.others_changed(&before, 1);
            eprintln!(
                "{what} L={l}: other_slots_mismatched_bits={others} {}",
                verdict(others == 0)
            );
            pass &= ok && others == 0;
        }
    }
    assert!(pass, "a GDN stage is out of bound");
}

// ── Continuation from a known state ───────────────────────────────

/// A known, finite start state in slot 2 (the others NaN): every stage within bound of the
/// reference started from it, at `q_len` 1, 65 and 300; the other slots bit-unchanged.
#[test]
fn a_continued_step_starts_from_its_slot() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0x6C00);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 300, 0x6C01);
        let l = GdnStateLayout::new(4, 16, d.nv as u32, 128, 128, 4);
        let ring = f32_inputs(l.conv_elems_per_slot(), 0x6C02, (-1.0, 1.0));
        let state = f32_inputs(l.ssm_elems_per_slot(), 0x6C03, (-0.5, 0.5));
        for q_len in [1, 65, 300] {
            let mut pool = Pool::new(&device, d, NAN_F32);
            pool.write_slot(&device, 2, &ring, &state);
            let before = pool.snapshot();
            let c = chunk(&x, d, 0, q_len, q_len + PAD);
            let s = run_step(&device, &pool, &bb, d, &c, (2, false)).expect("the step launches");
            let start = Start {
                ring: &to_f64(&ring),
                state: &to_f64(&state),
                smax: None,
                t0: 0,
            };
            let what = format!("{} continued", name(d));
            let (ok, _) = check_step(&what, (d, &bu), (&c, &s), &start, (&pool, 2));
            let others = pool.others_changed(&before, 2);
            eprintln!(
                "{what} L={q_len}: other_slots_mismatched_bits={others} {}",
                verdict(others == 0)
            );
            pass &= ok && others == 0;
        }
    }
    assert!(pass, "a continued GDN step is out of bound");
}

// ── Self-checks: a wrong kernel cannot pass ───────────────────────

/// The fraction of `o`'s elements with `t ≥ 1` a mutated scan reference puts out of bound.
fn scan_misses(d: GdnDims, s: &StepOut, q_len: usize, variant: ScanVariant) -> F32Comparison {
    let (_, state) = zeros(d);
    let r = scan_reference(d, s, q_len, (&state, None, 0), variant);
    let vd = d.value_dim();
    compare_o(s, &r, |i| i >= vd && i < q_len * vd)
}

/// A missing correction, the decay after the update, and `β = b`, each a mutated reference
/// compared with the GPU's output: at least half the `t ≥ 1` elements out of bound, at `q_len`
/// 64 and 300, except the missing correction at 300, which needs a quarter (issue #807, amended
/// after the H100 run: 29.5% / 33.1% there, 67% / 72% at 64; any non-trivial fraction fails the
/// real check, which allows none). In 9b, the modulo head mapping: at least 28 of the 32 value heads out of bound
/// somewhere. A sigmoid gate in the norm, and reversed conv taps: at least half out of bound.
#[test]
fn every_self_check_misses_by_its_margin() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0x5C00);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 300, 0x5C01);
        for q_len in [64, 300] {
            let pool = Pool::new(&device, d, NAN_F32);
            let c = chunk(&x, d, 0, q_len, q_len);
            let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
            let no_correction_need = if q_len == 300 { 0.25 } else { 0.5 };
            for (what, variant, need) in [
                (
                    "no delta-rule correction (u = beta*v)",
                    ScanVariant::NoCorrection,
                    no_correction_need,
                ),
                ("decay after the update", ScanVariant::DecayLast, 0.5),
            ] {
                let m = scan_misses(d, &s, q_len, variant);
                let ok = m.fraction() >= need;
                eprintln!(
                    "self-check {} L={q_len} {what}: out_of_bound={}/{} ({:.4}, need >= {need}) {}",
                    name(d),
                    m.out_of_bound,
                    m.n,
                    m.fraction(),
                    verdict(ok)
                );
                pass &= ok;
            }
            // β = b: the scan's reference from the GPU's conv_out and g, with b itself as β.
            let nv = d.nv;
            let (_, b_raw) = gdn_gating_reference(
                &widen(&c.a[..q_len * nv]),
                &widen(&c.b[..q_len * nv]),
                &to_f64(&bu.a_log),
                &to_f64(&bu.dt_bias),
                nv,
                false,
            );
            let (_, state) = zeros(d);
            let conv_out = to_f64(&s.conv_out[..q_len * d.conv_dim()]);
            let gpu_g = to_f64(&s.g[..q_len * nv]);
            let r = gdn_scan_reference(
                &ScanInputs {
                    conv_out: &conv_out,
                    g: &gpu_g,
                    beta: &b_raw,
                    state: &state,
                    smax: None,
                    t0: 0,
                    q_len,
                },
                d,
                ScanVariant::Real,
            );
            let m = compare_o(&s, &r, |_| true);
            let ok = m.fraction() >= 0.5;
            eprintln!(
                "self-check {} L={q_len} beta = b (no sigmoid): out_of_bound={}/{} ({:.4}, need \
                 >= 0.5) {}",
                name(d),
                m.out_of_bound,
                m.n,
                m.fraction(),
                verdict(ok)
            );
            pass &= ok;
            if d.nv == G9.nv {
                let (_, state) = zeros(d);
                let r = scan_reference(d, &s, q_len, (&state, None, 0), ScanVariant::ModuloHeads);
                let refs: Vec<f64> = r.o.iter().map(|p| p.0).collect();
                let (vd, hv) = (d.value_dim(), d.head_v);
                let heads_missed = (0..d.nv)
                    .filter(|&h| {
                        (0..q_len * vd).any(|i| {
                            (i % vd) / hv == h && {
                                let bound = gdn_scan_bound(r.o[i].0, r.o[i].1);
                                {
                                    let g = f64::from(s.o[i]);
                                    !g.is_finite() || (g - refs[i]).abs() > bound
                                }
                            }
                        })
                    })
                    .count();
                let ok = heads_missed >= 28;
                eprintln!(
                    "self-check 9b L={q_len} key head h % nk: value_heads_out_of_bound=\
                     {heads_missed}/32 (need >= 28) {}",
                    verdict(ok)
                );
                pass &= ok;
            }
            // The norm with a sigmoid gate.
            let n = q_len * d.value_dim();
            let wrong = gdn_norm_reference(
                &to_f64(&s.o[..n]),
                &widen(&c.z[..n]),
                &to_f64(&bu.norm),
                (q_len, d.nv, d.head_v),
                f64::from(EPS),
                false,
            );
            let cmp = compare_rows(&s.out[..n], &wrong, d.value_dim());
            let frac = cmp.out_of_bound as f64 / n as f64;
            let ok = frac >= 0.5;
            eprintln!(
                "self-check {} L={q_len} norm gate sigmoid instead of SiLU: out_of_bound={}/{n} \
                 ({frac:.4}, need >= 0.5) {}",
                name(d),
                cmp.out_of_bound,
                verdict(ok)
            );
            pass &= ok;
            // The conv with its taps reversed.
            let cd = d.conv_dim();
            let (zr, _) = zeros(d);
            let (refs, _) = gdn_conv_reference(
                &widen(&c.qkv[..q_len * cd]),
                &to_f64(&bu.w),
                &zr,
                (q_len, cd, d.conv_kernel),
                true,
            );
            let r: Vec<f64> = refs.iter().map(|p| p.0).collect();
            let m = compare_f32(
                &s.conv_out,
                &r,
                |i| gdn_conv_bound(refs[i].0, refs[i].1),
                |_| true,
            );
            let ok = m.fraction() >= 0.5;
            eprintln!(
                "self-check {} L={q_len} conv taps reversed: out_of_bound={}/{} ({:.4}, need >= \
                 0.5) {}",
                name(d),
                m.out_of_bound,
                m.n,
                m.fraction(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a self-check's mutated reference passed");
}

// ── Exact probes ───────────────────────────────────────────────────

/// 9b: NaN in the `q` and `k` columns of key head `j`, every row: exactly value heads `2j` and
/// `2j + 1` have a non-finite `o`; every other head is finite.
#[test]
fn a_key_heads_nan_reaches_exactly_its_two_value_heads() {
    let device = h100();
    let d = G9;
    let bu = bundle(d, 0x4B00);
    let bb = upload_bundle(&device, &bu);
    let mut pass = true;
    for j in [0usize, 5, 15] {
        let mut x = inputs(d, 64, 0x4B01);
        let (cd, kd, hk) = (d.conv_dim(), d.key_dim(), d.head_k);
        for t in 0..64 {
            for c in 0..hk {
                x.qkv[t * cd + j * hk + c] = NAN_BF16;
                x.qkv[t * cd + kd + j * hk + c] = NAN_BF16;
            }
        }
        let pool = Pool::new(&device, d, 0);
        let c = chunk(&x, d, 0, 64, 64);
        let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
        let (vd, hv) = (d.value_dim(), d.head_v);
        let non_finite: Vec<usize> = (0..d.nv)
            .filter(|&h| (0..64).any(|t| (0..hv).any(|i| !s.o[t * vd + h * hv + i].is_finite())))
            .collect();
        let want = vec![2 * j, 2 * j + 1];
        let ok = non_finite == want;
        eprintln!(
            "key-head probe j={j}: value heads with non-finite o = {non_finite:?} (want \
             {want:?}) {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a key head's NaN reached another value head");
}

/// `v = 0` for every value head but one, fresh: every other head's `o` and final state are
/// exactly zero.
#[test]
fn a_zero_value_head_stays_exactly_zero() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0x2E00);
        let bb = upload_bundle(&device, &bu);
        let mut x = inputs(d, 300, 0x2E01);
        let (cd, kd, hv, vd) = (d.conv_dim(), d.key_dim(), d.head_v, d.value_dim());
        let live = 3usize;
        for t in 0..300 {
            for h in (0..d.nv).filter(|&h| h != live) {
                for i in 0..hv {
                    x.qkv[t * cd + 2 * kd + h * hv + i] = 0;
                }
            }
        }
        // The conv of an all-zero channel is SiLU(0) = 0 exactly, so `v` reaches the scan as 0.
        let pool = Pool::new(&device, d, NAN_F32);
        let c = chunk(&x, d, 0, 300, 300);
        let s = run_step(&device, &pool, &bb, d, &c, (1, true)).expect("the step launches");
        let o_nonzero = (0..300 * vd)
            .filter(|&i| (i % vd) / hv != live && s.o[i] != 0.0)
            .count();
        let state = pool.ssm_slot(1);
        let per_head = hv * d.head_k;
        let s_nonzero = (0..state.len())
            .filter(|&i| i / per_head != live && state[i] != 0.0)
            .count();
        let live_nonzero = (0..300 * vd)
            .filter(|&i| (i % vd) / hv == live && s.o[i] != 0.0)
            .count();
        let ok = o_nonzero == 0 && s_nonzero == 0 && live_nonzero > 0;
        eprintln!(
            "value-isolation probe {}: other heads' o not zero = {o_nonzero}, state not zero = \
             {s_nonzero}, live head's o non-zero = {live_nonzero} {}",
            name(d),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a zero value head's o or state is not exactly zero");
}

/// NaN in the padding rows of `z`, `a`, `b` and `qkv` (every chunk in these tests pads with NaN):
/// the padding rows of the output are `+0.0`, the real rows within bound, and the state equal,
/// bit for bit, to that of the same `q_len` tokens in a bucket with no padding.
#[test]
fn padding_rows_reach_nothing() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0x9A00);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 100, 0x9A01);
        let padded = chunk(&x, d, 0, 100, 128);
        let exact = chunk(&x, d, 0, 100, 100);
        let pool_p = Pool::new(&device, d, NAN_F32);
        let sp = run_step(&device, &pool_p, &bb, d, &padded, (0, true)).expect("padded");
        let pool_e = Pool::new(&device, d, NAN_F32);
        let se = run_step(&device, &pool_e, &bb, d, &exact, (0, true)).expect("exact");
        let n = 100 * d.value_dim();
        let zeros_ok = sp.out[n..].iter().all(|&b| b == 0);
        let same_out = sp.out[..n] == se.out[..];
        let state_bits = (bits_of(&pool_p.ssm_slot(0))
            .iter()
            .zip(bits_of(&pool_e.ssm_slot(0))))
        .filter(|(a, b)| **a != *b)
        .count()
            + (bits_of(&pool_p.conv_slot(0))
                .iter()
                .zip(bits_of(&pool_e.conv_slot(0))))
            .filter(|(a, b)| **a != *b)
            .count();
        let (ring, state) = zeros(d);
        let start = Start {
            ring: &ring,
            state: &state,
            smax: None,
            t0: 0,
        };
        let (ok, _) = check_step(
            &format!("{} padded", name(d)),
            (d, &bu),
            (&padded, &sp),
            &start,
            (&pool_p, 0),
        );
        let all = ok && zeros_ok && same_out && state_bits == 0;
        eprintln!(
            "padding {}: padding_rows_plus_zero={zeros_ok} real_rows_bit_identical={same_out} \
             state_mismatched_bits={state_bits} {}",
            name(d),
            verdict(all)
        );
        pass &= all;
    }
    assert!(pass, "a padding row reached an output or the state");
}

// ── Continuity, stale state, repeats ───────────────────────────────

/// One forward of `L1 + L2` tokens, and the same tokens as two chunks from the state the first
/// left: the second chunk's outputs and the final state bit-identical to the one forward's.
#[test]
fn two_chunks_are_bit_identical_to_one_forward() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xC0C0);
        let bb = upload_bundle(&device, &bu);
        for (l1, l2) in [
            (1, 1),
            (2, 1),
            (3, 5),
            (64, 64),
            (100, 200),
            (300, 1),
            (300, 212),
        ] {
            let x = inputs(d, l1 + l2, 0xC0C1 ^ (l1 as u64) << 16 ^ l2 as u64);
            let pool_one = Pool::new(&device, d, NAN_F32);
            let one = chunk(&x, d, 0, l1 + l2, l1 + l2);
            let so = run_step(&device, &pool_one, &bb, d, &one, (3, true)).expect("one forward");
            let pool_two = Pool::new(&device, d, NAN_F32);
            let c1 = chunk(&x, d, 0, l1, l1 + PAD);
            run_step(&device, &pool_two, &bb, d, &c1, (3, true)).expect("chunk 1");
            let c2 = chunk(&x, d, l1, l2, l2 + PAD);
            let s2 = run_step(&device, &pool_two, &bb, d, &c2, (3, false)).expect("chunk 2");
            let vd = d.value_dim();
            let out = (so.out[l1 * vd..].iter().zip(&s2.out[..l2 * vd]))
                .filter(|(a, b)| a != b)
                .count();
            let o = (so.o[l1 * vd..].iter().zip(&s2.o[..l2 * vd]))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state = (bits_of(&pool_one.ssm_slot(3))
                .iter()
                .zip(bits_of(&pool_two.ssm_slot(3))))
            .filter(|(a, b)| **a != *b)
            .count()
                + (bits_of(&pool_one.conv_slot(3))
                    .iter()
                    .zip(bits_of(&pool_two.conv_slot(3))))
                .filter(|(a, b)| **a != *b)
                .count();
            let ok = out + o + state == 0;
            eprintln!(
                "continuity {} ({l1}, {l2}): out_mismatched={out} o_mismatched={o} \
                 state_mismatched_bits={state} {}",
                name(d),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "two chunks differ from one forward: stop and report");
}

/// A pool of 4 slots. (a) Fresh in slot 2 with every slot NaN: finite, within bound of the
/// zero-start reference, slot 2 finite after, slots 0, 1, 3 bit-unchanged. (b) Fresh in slot 2
/// holding a previous sequence's large finite state: bit-identical to (a). (c) Continued in slot
/// 2 from a known state, the others NaN: within bound, the others bit-unchanged.
#[test]
fn stale_state_cannot_leak() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0x57A1);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 300, 0x57A2);
        let c = chunk(&x, d, 0, 300, 300 + PAD);
        let (ring, state) = zeros(d);
        let fresh_start = Start {
            ring: &ring,
            state: &state,
            smax: None,
            t0: 0,
        };
        // (a)
        let pool_a = Pool::new(&device, d, NAN_F32);
        let before = pool_a.snapshot();
        let sa = run_step(&device, &pool_a, &bb, d, &c, (2, true)).expect("(a)");
        let (ok_a, _) = check_step(
            &format!("{} stale (a)", name(d)),
            (d, &bu),
            (&c, &sa),
            &fresh_start,
            (&pool_a, 2),
        );
        let finite = pool_a
            .ssm_slot(2)
            .iter()
            .chain(&pool_a.conv_slot(2))
            .all(|v| v.is_finite());
        let others_a = pool_a.others_changed(&before, 2);
        // (b)
        let mut pool_b = Pool::new(&device, d, NAN_F32);
        let l = pool_b.layout;
        let big_ring = f32_inputs(l.conv_elems_per_slot(), 0x57A3, (-1e4, 1e4));
        let big_state = f32_inputs(l.ssm_elems_per_slot(), 0x57A4, (-1e4, 1e4));
        pool_b.write_slot(&device, 2, &big_ring, &big_state);
        let sb = run_step(&device, &pool_b, &bb, d, &c, (2, true)).expect("(b)");
        let b_vs_a = (sa.out.iter().zip(&sb.out)).filter(|(x, y)| x != y).count()
            + (bits_of(&pool_a.ssm_slot(2))
                .iter()
                .zip(bits_of(&pool_b.ssm_slot(2))))
            .filter(|(x, y)| **x != *y)
            .count()
            + (bits_of(&pool_a.conv_slot(2))
                .iter()
                .zip(bits_of(&pool_b.conv_slot(2))))
            .filter(|(x, y)| **x != *y)
            .count();
        // (c)
        let mut pool_c = Pool::new(&device, d, NAN_F32);
        let known_ring = f32_inputs(l.conv_elems_per_slot(), 0x57A5, (-1.0, 1.0));
        let known_state = f32_inputs(l.ssm_elems_per_slot(), 0x57A6, (-0.5, 0.5));
        pool_c.write_slot(&device, 2, &known_ring, &known_state);
        let before_c = pool_c.snapshot();
        let sc = run_step(&device, &pool_c, &bb, d, &c, (2, false)).expect("(c)");
        let (kr, ks) = (to_f64(&known_ring), to_f64(&known_state));
        let start_c = Start {
            ring: &kr,
            state: &ks,
            smax: None,
            t0: 0,
        };
        let (ok_c, _) = check_step(
            &format!("{} stale (c)", name(d)),
            (d, &bu),
            (&c, &sc),
            &start_c,
            (&pool_c, 2),
        );
        let others_c = pool_c.others_changed(&before_c, 2);
        let ok = ok_a && finite && others_a == 0 && b_vs_a == 0 && ok_c && others_c == 0;
        eprintln!(
            "stale state {}: (a) slot_2_finite={finite} other_slots_mismatched_bits={others_a}; \
             (b) bit_mismatches_vs_(a)={b_vs_a}; (c) other_slots_mismatched_bits={others_c} {}",
            name(d),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "stale state leaked");
}

/// Two identical runs give identical bits: outputs and state.
#[test]
fn repeats_are_bit_identical() {
    let device = h100();
    let d = G9;
    let bu = bundle(d, 0xEE00);
    let bb = upload_bundle(&device, &bu);
    let x = inputs(d, 1024, 0xEE01);
    let c = chunk(&x, d, 0, 1024, 1024 + PAD);
    let run = || {
        let pool = Pool::new(&device, d, NAN_F32);
        let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
        (s.out, bits_of(&s.o), pool.snapshot())
    };
    let (a, b) = (run(), run());
    let differ = usize::from(a.0 != b.0) + usize::from(a.1 != b.1) + usize::from(a.2 != b.2);
    eprintln!(
        "repeats 9b L=1024: differing_buffers={differ} {}",
        verdict(differ == 0)
    );
    assert_eq!(differ, 0, "two identical runs differ");
}

// ── Launcher refusals ──────────────────────────────────────────────

/// Each launcher's rules, one at a time: the launch refused before it touches a pointer, and the
/// output left as it was.
#[test]
fn every_launcher_refuses_what_it_does_not_take() {
    let device = h100();
    let small = |n: usize| up_f32_bits(&device, n, NAN_F32);
    let (x, w, st, seq, out) = (
        small(1 << 12),
        small(1 << 12),
        small(1 << 12),
        small(4),
        small(1 << 12),
    );
    let unchanged = |b: &DeviceBuffer| down_f32(b).iter().all(|v| v.to_bits() == NAN_F32);
    let null = std::ptr::null_mut();
    let mut pass = true;
    let mut report =
        |what: &str, got: Result<(), KernelError>, want: KernelError, buf: &DeviceBuffer| {
            let ok = got == Err(want) && unchanged(buf);
            eprintln!(
                "refused {what}: {got:?} output_unchanged={} {}",
                unchanged(buf),
                verdict(ok)
            );
            pass &= ok;
        };
    // The conv: kernel outside 2..=8, a state stride that is not `conv_dim · (K − 1)`, no rows.
    for (rows, cd, k, se) in [
        (4u32, 288u32, 1u32, 0u64),
        (4, 288, 9, 288 * 8),
        (4, 288, 4, 288 * 4),
        (0, 288, 4, 288 * 3),
    ] {
        // SAFETY: the launcher refuses each shape before it touches a pointer.
        let got = unsafe {
            gdn_conv1d(
                (x.ptr(), w.ptr()),
                (st.ptr(), se),
                seq.ptr(),
                out.ptr(),
                (rows, cd, k),
                null,
            )
        };
        report(
            &format!("conv1d rows={rows} conv_dim={cd} kernel={k} slot_elems={se}"),
            got,
            KernelError::GdnConvShape {
                rows,
                conv_dim: cd,
                kernel: k,
            },
            &out,
        );
    }
    // The gating: no rows, no heads.
    for (rows, nv) in [(0u32, 16u32), (4, 0)] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_gating(
                (x.ptr(), x.ptr()),
                (w.ptr(), w.ptr()),
                seq.ptr(),
                (out.ptr(), out.ptr()),
                (rows, nv),
                null,
            )
        };
        report(
            &format!("gating rows={rows} nv={nv}"),
            got,
            KernelError::GdnGatingShape { rows, nv },
            &out,
        );
    }
    // The scan: head_k not 128, head_v off 32 or above 256, nv not a multiple of nk, a wrong
    // state stride, a zero scale.
    let scale = gdn_scale(128);
    for (nk, nv, hk, hv, se, sc) in [
        (1u32, 1u32, 64u32, 32u32, 32u64 * 128, scale),
        (1, 1, 128, 40, 40 * 128, scale),
        (1, 1, 128, 512, 512 * 128, scale),
        (2, 3, 128, 32, 3 * 32 * 128, scale),
        (1, 1, 128, 32, 32 * 64, scale),
        (1, 1, 128, 32, 32 * 128, 0.0),
    ] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_scan(
                (x.ptr(), x.ptr(), x.ptr()),
                (st.ptr(), se),
                seq.ptr(),
                out.ptr(),
                (4, nk, nv),
                (hk, hv, sc),
                null,
            )
        };
        report(
            &format!("scan nk={nk} nv={nv} head_k={hk} head_v={hv} slot_elems={se} scale={sc}"),
            got,
            KernelError::GdnScanShape {
                rows: 4,
                nk,
                nv,
                head_k: hk,
                head_v: hv,
            },
            &out,
        );
    }
    // The gated norm: head_v off 32, above 256, a zero eps.
    for (hv, eps) in [(40u32, EPS), (512, EPS), (32, 0.0)] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_rms_norm_gated(
                (x.ptr(), x.ptr(), w.ptr()),
                seq.ptr(),
                out.ptr(),
                (4, 1, hv),
                eps,
                null,
            )
        };
        report(
            &format!("gated norm head_v={hv} eps={eps}"),
            got,
            KernelError::GdnNormShape {
                rows: 4,
                nv: 1,
                head_v: hv,
            },
            &out,
        );
    }
    assert!(
        pass,
        "a launcher took a shape it must refuse, or wrote its output"
    );
}

/// Each launcher at every zero dimension it takes, and at a count above `INT_MAX` (the 9b's
/// geometry at a `rows` that overflows it): the launch refused before it touches a pointer, its
/// output and the state buffer left as they were (#807).
#[test]
fn every_launcher_refuses_zero_dimensions_and_counts_above_int_max() {
    let device = h100();
    let small = |n: usize| up_f32_bits(&device, n, NAN_F32);
    let (x, w, st, seq, out) = (
        small(1 << 12),
        small(1 << 12),
        small(1 << 12),
        small(4),
        small(1 << 12),
    );
    let unchanged = |b: &DeviceBuffer| down_f32(b).iter().all(|v| v.to_bits() == NAN_F32);
    let null = std::ptr::null_mut();
    let mut pass = true;
    let mut report = |what: &str, got: Result<(), KernelError>, want: KernelError| {
        let (o, s) = (unchanged(&out), unchanged(&st));
        let ok = got == Err(want) && o && s;
        eprintln!(
            "refused {what}: {got:?} output_unchanged={o} state_unchanged={s} {}",
            verdict(ok)
        );
        pass &= ok;
    };
    let (cd9, nv9) = (G9.conv_dim() as u32, G9.nv as u32);
    // The conv: no rows, no channels, and `rows · conv_dim` = 2¹⁸ · 8192 = 2³¹.
    for (rows, cd) in [(0u32, cd9), (4, 0), (1 << 18, cd9)] {
        let se = u64::from(cd) * 3;
        // SAFETY: the launcher refuses each shape before it touches a pointer.
        let got = unsafe {
            gdn_conv1d(
                (x.ptr(), w.ptr()),
                (st.ptr(), se),
                seq.ptr(),
                out.ptr(),
                (rows, cd, 4),
                null,
            )
        };
        report(
            &format!("conv1d rows={rows} conv_dim={cd} kernel=4 slot_elems={se}"),
            got,
            KernelError::GdnConvShape {
                rows,
                conv_dim: cd,
                kernel: 4,
            },
        );
    }
    // The gating: no rows, no heads, and `rows · nv` = 2²⁶ · 32 = 2³¹.
    for (rows, nv) in [(0u32, nv9), (4, 0), (1 << 26, nv9)] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_gating(
                (x.ptr(), x.ptr()),
                (w.ptr(), w.ptr()),
                seq.ptr(),
                (out.ptr(), out.ptr()),
                (rows, nv),
                null,
            )
        };
        report(
            &format!("gating rows={rows} nv={nv}"),
            got,
            KernelError::GdnGatingShape { rows, nv },
        );
    }
    // The scan: no rows, no key heads, no value heads, a zero head_v, and `rows · conv_dim` =
    // 2¹⁸ · 8192 = 2³¹ in the 9b's geometry.
    let scale = gdn_scale(128);
    for (rows, nk, nv, hv) in [
        (0u32, 16u32, nv9, 128u32),
        (4, 0, nv9, 128),
        (4, 16, 0, 128),
        (4, 16, nv9, 0),
        (1 << 18, 16, nv9, 128),
    ] {
        let se = u64::from(nv) * u64::from(hv) * 128;
        // SAFETY: as above.
        let got = unsafe {
            gdn_scan(
                (x.ptr(), x.ptr(), x.ptr()),
                (st.ptr(), se),
                seq.ptr(),
                out.ptr(),
                (rows, nk, nv),
                (128, hv, scale),
                null,
            )
        };
        report(
            &format!("scan rows={rows} nk={nk} nv={nv} head_v={hv} slot_elems={se}"),
            got,
            KernelError::GdnScanShape {
                rows,
                nk,
                nv,
                head_k: 128,
                head_v: hv,
            },
        );
    }
    // The gated norm: no rows, no heads, a zero head_v, and `rows · nv · head_v` = 2¹⁹ · 32 · 128
    // = 2³¹.
    for (rows, nv, hv) in [
        (0u32, nv9, 128u32),
        (4, 0, 128),
        (4, nv9, 0),
        (1 << 19, nv9, 128),
    ] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_rms_norm_gated(
                (x.ptr(), x.ptr(), w.ptr()),
                seq.ptr(),
                out.ptr(),
                (rows, nv, hv),
                EPS,
                null,
            )
        };
        report(
            &format!("gated norm rows={rows} nv={nv} head_v={hv}"),
            got,
            KernelError::GdnNormShape {
                rows,
                nv,
                head_v: hv,
            },
        );
    }
    assert!(
        pass,
        "a launcher took a zero dimension or a count above INT_MAX, or wrote its output"
    );
}

/// A decode's GDN step: the same four launches at `rows = 1` exactly, with no padding row (every
/// other GDN case runs `rows = q_len + PAD`). One forward of `L1 + 1` tokens, then the same tokens
/// as a first chunk of `L1` and a one-token chunk launched with `rows = 1`, from the state the
/// first left: the one token's outputs and the final state bit-identical to the one forward's,
/// in the 0.8b and 9b geometries. `(1, 1)` launches both chunks at `rows = 1`.
#[test]
fn a_one_row_step_with_no_padding_is_bit_identical_to_one_forward() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xDEC1);
        let bb = upload_bundle(&device, &bu);
        for l1 in [300usize, 1] {
            let x = inputs(d, l1 + 1, 0xDEC2 ^ (l1 as u64) << 16);
            let pool_one = Pool::new(&device, d, NAN_F32);
            let one = chunk(&x, d, 0, l1 + 1, l1 + 1);
            let so = run_step(&device, &pool_one, &bb, d, &one, (3, true)).expect("one forward");
            let pool_two = Pool::new(&device, d, NAN_F32);
            let c1 = chunk(&x, d, 0, l1, l1);
            let first = run_step(&device, &pool_two, &bb, d, &c1, (3, true));
            let c2 = chunk(&x, d, l1, 1, 1);
            let second = first.and_then(|_| run_step(&device, &pool_two, &bb, d, &c2, (3, false)));
            let s2 = match second {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "one-row step {} ({l1}, 1) at rows = 1: a launcher refused: {e} FAIL",
                        name(d)
                    );
                    pass = false;
                    continue;
                }
            };
            let vd = d.value_dim();
            let out = mismatched(&so.out[l1 * vd..][..vd], &s2.out);
            let o = (so.o[l1 * vd..][..vd].iter().zip(&s2.o))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state = (bits_of(&pool_one.ssm_slot(3)).iter())
                .zip(bits_of(&pool_two.ssm_slot(3)))
                .filter(|(a, b)| **a != *b)
                .count()
                + (bits_of(&pool_one.conv_slot(3)).iter())
                    .zip(bits_of(&pool_two.conv_slot(3)))
                    .filter(|(a, b)| **a != *b)
                    .count();
            let ok = out + o + state == 0;
            eprintln!(
                "one-row step {} ({l1}, 1) at rows = 1, no padding: out_mismatched={out} \
                 o_mismatched={o} state_mismatched_bits={state} {}",
                name(d),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a one-row GDN step differs from one forward, or was refused: stop and report"
    );
}

/// How many bf16 elements of `a` and `b` differ in any bit.
fn mismatched(a: &[u16], b: &[u16]) -> usize {
    assert_eq!(a.len(), b.len(), "the two outputs differ in length");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

// ── #855: R8, the base-comparison dump and the timing line ─────────

/// R8 (#855's first STOP gate): the scan's rank-1 update, `warp::row_map<base_ops::fma_AxCtB>(S,
/// T, S, u)` on its `rt_fl<16, 128>` state tile, is `T · u + S` with `u` one value per row (the
/// probe kernel in `kernels/gdn_scan.cu`). Two cases, TK-0's probe's: small integers and halves,
/// where every result is exact in f32, and random values in [-2, 2] within `2⁻²²·(|T·u| + |S|)`.
/// Each line also counts the elements a wrong reading would match: `S·u + T`, `T·S + u`, and `u`
/// taken by column.
#[test]
fn the_rank_one_update_is_t_times_u_plus_s() {
    use scratchy_target_cuda::kernels::gdn_rank_one_probe;
    const R: usize = 16;
    const C: usize = 128;
    let device = h100();
    let mut pass = true;
    for exact in [true, false] {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32
        };
        let (mut s, mut t, mut u) = (vec![0f32; R * C], vec![0f32; R * C], vec![0f32; R]);
        for r in 0..R {
            u[r] = if exact {
                (r as f32 + 1.0) * 0.5 * if r % 2 == 1 { -1.0 } else { 1.0 }
            } else {
                rnd()
            };
            for c in 0..C {
                s[r * C + c] = if exact {
                    ((r * C + c) % 61) as f32 - 30.0
                } else {
                    rnd()
                };
                t[r * C + c] = if exact {
                    ((r * 7 + c * 3) % 17) as f32 - 8.0
                } else {
                    rnd()
                };
            }
        }
        let (sb, tb, ub) = (
            up_f32(&device, &s),
            up_f32(&device, &t),
            up_f32(&device, &u),
        );
        let out = up_f32_bits(&device, R * C, NAN_F32);
        // SAFETY: cudaMalloc pointers are 256-byte aligned; `s`, `t` and `out` hold 16·128 f32
        // and `u` 16; `out` is its own allocation; null is the default stream, which the
        // download below waits for.
        unsafe {
            gdn_rank_one_probe(
                (sb.ptr(), tb.ptr(), ub.ptr()),
                out.ptr(),
                std::ptr::null_mut(),
            )
        }
        .expect("the rank-one probe launches");
        let got = down_f32(&out);
        let (mut bad, mut s_u_t, mut t_s_u, mut by_col, mut worst) = (0, 0, 0, 0, 0.0f64);
        for r in 0..R {
            for c in 0..C {
                let i = r * C + c;
                let (si, ti, ur) = (f64::from(s[i]), f64::from(t[i]), f64::from(u[r]));
                let want = ti * ur + si;
                let bound = if exact {
                    0.0
                } else {
                    ((ti * ur).abs() + si.abs()) * 2f64.powi(-22)
                };
                let g = f64::from(got[i]);
                let err = (g - want).abs();
                // A NaN `got` counts as out of reference.
                bad += usize::from(err.is_nan() || err > bound);
                if bound > 0.0 {
                    worst = worst.max(err / bound);
                }
                let near = |alt: f64| (g - alt).abs() <= bound.max(1e-30);
                s_u_t += usize::from(near(si * ur + ti));
                t_s_u += usize::from(near(ti * si + ur));
                by_col += usize::from(near(ti * f64::from(u[c % R]) + si));
            }
        }
        let what = if exact {
            "small integers and halves (every result exact in f32)"
        } else {
            "random values in [-2, 2]"
        };
        let ok = bad == 0;
        eprintln!(
            "R8 {what}: out_of_reference={bad} of {} worst_err/bound={worst:.6} (matches S*u+T at \
             {s_u_t}, T*S+u at {t_s_u}, u by column at {by_col}) {}",
            R * C,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "R8: the rank-one update is not T·u + S: stop before the scan tests"
    );
}

/// For the base-against-change comparison (#855): the scan's `o` (the real rows) and the slot's
/// final state of a fresh step at the 0.8b and 9b geometries, L ∈ {1, 64, 300, 2048}, written as
/// raw little-endian f32 to `<CARGO_TARGET_TMPDIR>/gdn_scan_dump/<case>_{o,state}.bin`, with
/// `case= o_elems= state_elems= fnv1a=` printed. The same file run at two commits gives the
/// `base_vs_pr mismatched_bits=` per case; this test asserts only that each step ran.
#[test]
fn the_scan_outputs_are_dumped_for_a_base_comparison() {
    let device = h100();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gdn_scan_dump");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    let fnv = |bytes: &[u8]| {
        (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        })
    };
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xD0D0);
        let bb = upload_bundle(&device, &bu);
        for q_len in [1usize, 64, 300, 2048] {
            let x = inputs(d, q_len, 0xD0D1 ^ q_len as u64);
            let c = chunk(&x, d, 0, q_len, q_len + PAD);
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run_step(&device, &pool, &bb, d, &c, (1, true)).expect("the step launches");
            let o: Vec<u8> = (s.o[..q_len * d.value_dim()].iter())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let state: Vec<u8> = (pool.ssm_slot(1).iter())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let case = format!("gdn_scan_{}_L{q_len}", name(d));
            std::fs::write(dir.join(format!("{case}_o.bin")), &o).expect("write o");
            std::fs::write(dir.join(format!("{case}_state.bin")), &state).expect("write the state");
            eprintln!(
                "case={case} o_elems={} state_elems={} fnv1a_o={:016x} fnv1a_state={:016x}",
                o.len() / 4,
                state.len() / 4,
                fnv(&o),
                fnv(&state)
            );
        }
    }
}

/// The scan's time per launch at L = 300 and 4096, at the 0.8b and 9b geometries: 20 launches of
/// the scan alone on the default stream, over one step's conv_out, g and β, after one warm-up,
/// timed to the download that waits for them. Recorded for the base-against-change comparison,
/// not asserted.
#[test]
fn the_scan_time_is_recorded() {
    const LAUNCHES: u32 = 20;
    let device = h100();
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xD1D0);
        let bb = upload_bundle(&device, &bu);
        for q_len in [300usize, 4096] {
            let x = inputs(d, q_len, 0xD1D1 ^ q_len as u64);
            let c = chunk(&x, d, 0, q_len, q_len);
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
            let (cv, g, beta) = (
                up_f32(&device, &s.conv_out),
                up_f32(&device, &s.g),
                up_f32(&device, &s.beta),
            );
            let o = up_f32_bits(&device, q_len * d.value_dim(), NAN_F32);
            let seq = up_u32(&device, &[0, q_len as u32, 1, 0]);
            let launch = || {
                // SAFETY: as `run_step`'s, for the scan alone over this step's own scratch.
                unsafe {
                    gdn_scan(
                        (cv.ptr(), g.ptr(), beta.ptr()),
                        (pool.ssm.ptr(), pool.layout.ssm_elems_per_slot() as u64),
                        seq.ptr(),
                        o.ptr(),
                        (q_len as u32, d.nk as u32, d.nv as u32),
                        (d.head_k as u32, d.head_v as u32, gdn_scale(d.head_k as u32)),
                        std::ptr::null_mut(),
                    )
                }
                .expect("gdn_scan");
            };
            launch();
            seq.download().expect("wait for the warm-up");
            let start = std::time::Instant::now();
            for _ in 0..LAUNCHES {
                launch();
            }
            seq.download().expect("wait for the launches");
            let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES);
            eprintln!("gdn scan time {} L={q_len}: {us:.1} us/launch", name(d));
        }
    }
}

// ── #856: the two tightenings, the base-comparison dump and the timing lines ──

/// The two launcher tightenings (#856), each refused before a pointer is touched with its output
/// unchanged: the conv1d at `conv_dim` 192 (a valid kernel and state stride, but not a multiple of
/// 128, one warp's channels) and the gating at `nv` 24 (not a multiple of 16).
#[test]
fn the_tightened_gdn_launchers_refuse_a_width_off_their_instantiation() {
    let device = h100();
    let small = |n: usize| up_f32_bits(&device, n, NAN_F32);
    let (x, w, st, seq, out) = (
        small(1 << 12),
        small(1 << 12),
        small(1 << 12),
        small(4),
        small(1 << 12),
    );
    let unchanged = |b: &DeviceBuffer| down_f32(b).iter().all(|v| v.to_bits() == NAN_F32);
    let null = std::ptr::null_mut();
    let (rows, cd, k) = (4u32, 192u32, 4u32);
    // SAFETY: the launcher refuses this shape before it touches a pointer.
    let got = unsafe {
        gdn_conv1d(
            (x.ptr(), w.ptr()),
            (st.ptr(), u64::from(cd) * 3),
            seq.ptr(),
            out.ptr(),
            (rows, cd, k),
            null,
        )
    };
    let want = KernelError::GdnConvShape {
        rows,
        conv_dim: cd,
        kernel: k,
    };
    let kept = unchanged(&out) && unchanged(&st);
    let conv_ok = got == Err(want) && kept;
    eprintln!(
        "refused conv1d rows={rows} conv_dim={cd} kernel={k}: {got:?} output_unchanged={kept} {}",
        verdict(conv_ok)
    );
    let (rows, nv) = (4u32, 24u32);
    // SAFETY: as above.
    let got = unsafe {
        gdn_gating(
            (x.ptr(), x.ptr()),
            (w.ptr(), w.ptr()),
            seq.ptr(),
            (out.ptr(), out.ptr()),
            (rows, nv),
            null,
        )
    };
    let kept = unchanged(&out);
    let gating_ok = got == Err(KernelError::GdnGatingShape { rows, nv }) && kept;
    eprintln!(
        "refused gating rows={rows} nv={nv}: {got:?} output_unchanged={kept} {}",
        verdict(gating_ok)
    );
    assert!(
        conv_ok && gating_ok,
        "a tightened GDN launcher did not refuse a width off its instantiation"
    );
}

/// For the base-against-change comparison (#856): one fresh step at the 0.8b and 9b geometries,
/// L ∈ {1, 64, 300, 2048}: the conv1d's `conv_out` (real rows) and the slot's conv ring after it,
/// the gating's `g` and `β` (real rows), and the gated norm's `out` (every row), written as raw
/// little-endian bytes (f32, bf16 for `out`) to `<CARGO_TARGET_TMPDIR>/gdn_dump/<case>_<what>.bin`,
/// with `case= what= elems= fnv1a=` printed. The same file run at two commits gives the
/// `base_vs_pr mismatched_bits=` per case and buffer; this test asserts only that each step ran.
#[test]
fn the_gdn_stage_outputs_are_dumped_for_a_base_comparison() {
    let device = h100();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gdn_dump");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    let fnv = |bytes: &[u8]| {
        (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        })
    };
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xD2D0);
        let bb = upload_bundle(&device, &bu);
        for q_len in [1usize, 64, 300, 2048] {
            let x = inputs(d, q_len, 0xD2D1 ^ q_len as u64);
            let c = chunk(&x, d, 0, q_len, q_len + PAD);
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run_step(&device, &pool, &bb, d, &c, (1, true)).expect("the step launches");
            let case = format!("gdn_{}_L{q_len}", name(d));
            let out: Vec<u8> = s.out.iter().flat_map(|b| b.to_le_bytes()).collect();
            for (what, bytes, elem) in [
                (
                    "conv_out",
                    f32_bytes(&s.conv_out[..q_len * d.conv_dim()]),
                    4,
                ),
                ("ring", f32_bytes(&pool.conv_slot(1)), 4),
                ("g", f32_bytes(&s.g[..q_len * d.nv]), 4),
                ("beta", f32_bytes(&s.beta[..q_len * d.nv]), 4),
                ("out", out, 2),
            ] {
                std::fs::write(dir.join(format!("{case}_{what}.bin")), &bytes)
                    .expect("write a dump");
                eprintln!(
                    "case={case} what={what} elems={} fnv1a={:016x}",
                    bytes.len() / elem,
                    fnv(&bytes)
                );
            }
        }
    }
}

/// Each of the three kernels' time per launch at L = 300 and 4096, at the 0.8b and 9b geometries:
/// 20 launches of one kernel alone on the default stream, over one step's own inputs and scratch,
/// after one warm-up, timed to the download that waits for them. Recorded for the
/// base-against-change comparison, not asserted.
#[test]
fn the_gdn_conv_gating_and_norm_times_are_recorded() {
    const LAUNCHES: u32 = 20;
    let device = h100();
    let null = std::ptr::null_mut();
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xD3D0);
        let bb = upload_bundle(&device, &bu);
        for q_len in [300usize, 4096] {
            let x = inputs(d, q_len, 0xD3D1 ^ q_len as u64);
            let c = chunk(&x, d, 0, q_len, q_len);
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
            let (r, cd, nv, hv) = (
                q_len as u32,
                d.conv_dim() as u32,
                d.nv as u32,
                d.head_v as u32,
            );
            let (qkv, z, a, b) = (
                up(&device, &c.qkv),
                up(&device, &c.z),
                up(&device, &c.a),
                up(&device, &c.b),
            );
            let conv_out = up_f32_bits(&device, q_len * d.conv_dim(), NAN_F32);
            let (g, beta) = (
                up_f32_bits(&device, q_len * d.nv, NAN_F32),
                up_f32_bits(&device, q_len * d.nv, NAN_F32),
            );
            let o = up_f32(&device, &s.o);
            let out = up(&device, &vec![NAN_BF16; q_len * d.value_dim()]);
            // `start` 0: the conv reads the ring the step above wrote, never a fresh NaN slot.
            let seq = up_u32(&device, &[0, q_len as u32, 0, 0]);
            let l = pool.layout;
            let time = |what: &str, launch: &dyn Fn()| {
                launch();
                seq.download().expect("wait for the warm-up");
                let start = std::time::Instant::now();
                for _ in 0..LAUNCHES {
                    launch();
                }
                seq.download().expect("wait for the launches");
                let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES);
                eprintln!("gdn {what} time {} L={q_len}: {us:.1} us/launch", name(d));
            };
            // SAFETY (each closure): as `run_step`'s, each kernel alone over this step's buffers.
            time("conv1d", &|| {
                unsafe {
                    gdn_conv1d(
                        (qkv.ptr(), bb.w.ptr()),
                        (pool.conv.ptr(), l.conv_elems_per_slot() as u64),
                        seq.ptr(),
                        conv_out.ptr(),
                        (r, cd, d.conv_kernel as u32),
                        null,
                    )
                }
                .expect("gdn_conv1d");
            });
            time("gating", &|| {
                unsafe {
                    gdn_gating(
                        (a.ptr(), b.ptr()),
                        (bb.a_log.ptr(), bb.dt_bias.ptr()),
                        seq.ptr(),
                        (g.ptr(), beta.ptr()),
                        (r, nv),
                        null,
                    )
                }
                .expect("gdn_gating");
            });
            time("gated_norm", &|| {
                unsafe {
                    gdn_rms_norm_gated(
                        (o.ptr(), z.ptr(), bb.norm.ptr()),
                        seq.ptr(),
                        out.ptr(),
                        (r, nv, hv),
                        EPS,
                        null,
                    )
                }
                .expect("gdn_rms_norm_gated");
            });
        }
    }
}

// ── #931: the gating at every multiple of 16 ──────────────────────────

/// The gating alone (no conv, scan or norm) at `nv` 16 and 32 (qwen3.5's 0.8b and 9b), 48
/// (qwen3.6-27b), 64 (qwen3.5-122b-a10b) and 128, each at every length with `PAD` NaN padding
/// rows: `g` and `β` within bound of the reference, no padding element written, and a repeat
/// bit-identical.
#[test]
fn the_gating_takes_every_multiple_of_16() {
    let device = h100();
    let null = std::ptr::null_mut();
    let mut pass = true;
    for nv in [16usize, 32, 48, 64, 128] {
        let a_log = f32_inputs(nv, 0x931A ^ nv as u64, (-1.0, 0.5));
        let dt_bias = f32_inputs(nv, 0x931B ^ nv as u64, (-0.5, 0.5));
        let (lg, dg) = (up_f32(&device, &a_log), up_f32(&device, &dt_bias));
        for q_len in [1usize, 2, 63, 64, 65, 300, 2048] {
            let (rows, n) = (q_len + PAD, q_len * nv);
            let seed = 0x931C ^ ((nv as u64) << 16) ^ q_len as u64;
            let padded = |mut v: Vec<u16>| {
                v.resize(rows * nv, NAN_BF16);
                v
            };
            let a = padded(bf16_in(n, seed ^ 0x03, (-2.0, 2.0)));
            let b = padded(bf16_in(n, seed ^ 0x04, (-2.0, 2.0)));
            let (ab, bb) = (up(&device, &a), up(&device, &b));
            let seq = up_u32(&device, &[0, q_len as u32, 1, 0]);
            let run = || {
                let g = up_f32_bits(&device, rows * nv, NAN_F32);
                let beta = up_f32_bits(&device, rows * nv, NAN_F32);
                // SAFETY: cudaMalloc pointers are 256-byte aligned; `a`, `b`, `g`, `beta` hold
                // `rows·nv` elements, `a_log` and `dt_bias` `nv`, `sequence` 4 u32 with
                // `1 ≤ q_len ≤ rows`; `g` and `beta` overlap nothing; the download waits for the
                // default stream.
                unsafe {
                    gdn_gating(
                        (ab.ptr(), bb.ptr()),
                        (lg.ptr(), dg.ptr()),
                        seq.ptr(),
                        (g.ptr(), beta.ptr()),
                        (rows as u32, nv as u32),
                        null,
                    )
                }
                .expect("gdn_gating launches");
                (down_f32(&g), down_f32(&beta))
            };
            let (g, beta) = run();
            let (rg, rbeta) = gdn_gating_reference(
                &widen(&a[..n]),
                &widen(&b[..n]),
                &to_f64(&a_log),
                &to_f64(&dt_bias),
                nv,
                true,
            );
            let gc = compare_f32(&g, &rg, |i| gdn_gating_bound(rg[i]), |_| true);
            let bc = compare_f32(&beta, &rbeta, |i| gdn_gating_bound(rbeta[i]), |_| true);
            let padding = written_past(&g, n) + written_past(&beta, n);
            let (g2, beta2) = run();
            let repeat = bits_of(&g) == bits_of(&g2) && bits_of(&beta) == bits_of(&beta2);
            let ok = gc.pass() && bc.pass() && padding == 0 && repeat;
            eprintln!(
                "gating nv={nv} L={q_len}: g {} beta {} padding_written={padding} \
                 repeat_identical={repeat} {}",
                gc.line(),
                bc.line(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "the gating failed at a multiple of 16");
}

/// The gating at `nv` 40, a multiple of 8 but not of 16: refused before a pointer is touched,
/// its output unchanged.
#[test]
fn the_gating_refuses_a_width_off_16() {
    let device = h100();
    let small = |n: usize| up_f32_bits(&device, n, NAN_F32);
    let (x, w, seq, out) = (small(1 << 12), small(1 << 12), small(4), small(1 << 12));
    let unchanged = |b: &DeviceBuffer| down_f32(b).iter().all(|v| v.to_bits() == NAN_F32);
    let (rows, nv) = (4u32, 40u32);
    // SAFETY: the launcher refuses this shape before it touches a pointer.
    let got = unsafe {
        gdn_gating(
            (x.ptr(), x.ptr()),
            (w.ptr(), w.ptr()),
            seq.ptr(),
            (out.ptr(), out.ptr()),
            (rows, nv),
            std::ptr::null_mut(),
        )
    };
    let kept = unchanged(&out);
    let ok = got == Err(KernelError::GdnGatingShape { rows, nv }) && kept;
    eprintln!(
        "refused gating rows={rows} nv={nv}: {got:?} output_unchanged={kept} {}",
        verdict(ok)
    );
    assert!(ok, "the gating took a width that is not a multiple of 16");
}

/// #1049's one-launch GDN decode is the four launches bit for bit at one row: the bf16 output, the
/// conv ring and the state of the slot, for the three geometries (one, two and three value heads
/// per key head: the 0.8B, the 35B, the 27B), fresh (the pool NaN-poisoned, so a read of the slot
/// would show) and continued from a slot the four launches filled with 300 tokens; every other
/// slot bit-unchanged.
#[test]
fn the_gdn_decode_fold_is_its_four_launches_bit_for_bit() {
    use scratchy_target_cuda::kernels::{GdnDecodeOperands, gdn_decode};
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xDEC5);
        let bb = upload_bundle(&device, &bu);
        for prior in [0usize, 300] {
            let x = inputs(d, prior + 1, 0xDEC6 ^ (prior as u64) << 16);
            let (chain, fused) = (
                Pool::new(&device, d, NAN_F32),
                Pool::new(&device, d, NAN_F32),
            );
            if prior > 0 {
                let c = chunk(&x, d, 0, prior, prior);
                for p in [&chain, &fused] {
                    run_step(&device, p, &bb, d, &c, (3, true)).expect("the prefix");
                }
            }
            let fresh = prior == 0;
            let tok = chunk(&x, d, prior, 1, 1);
            let before = fused.snapshot();
            let want = run_step(&device, &chain, &bb, d, &tok, (3, fresh)).expect("the four");
            let seq = up_u32(&device, &[3, 1, u32::from(fresh), 0]);
            let (qkv, a, b, z) = (
                up(&device, &tok.qkv),
                up(&device, &tok.a),
                up(&device, &tok.b),
                up(&device, &tok.z),
            );
            let out = up(&device, &vec![NAN_BF16; d.value_dim()]);
            let l = fused.layout;
            let operands = GdnDecodeOperands {
                qkv: qkv.ptr(),
                a: a.ptr(),
                b: b.ptr(),
                z: z.ptr(),
                w: bb.w.ptr(),
                a_log: bb.a_log.ptr(),
                dt_bias: bb.dt_bias.ptr(),
                norm: bb.norm.ptr(),
                out: out.ptr(),
            };
            let dims = (
                d.nk as u32,
                d.nv as u32,
                d.head_k as u32,
                d.head_v as u32,
                d.conv_kernel as u32,
            );
            // SAFETY: as `run_step`'s at one row: every buffer holds what the launcher's contract
            // names, the pool `SLOTS` slots of its layout and slot 3 below it, `q_len` 1; null is
            // the default stream, which the downloads below wait for.
            let launched = unsafe {
                gdn_decode(
                    &operands,
                    (fused.conv.ptr(), l.conv_elems_per_slot() as u64),
                    (fused.ssm.ptr(), l.ssm_elems_per_slot() as u64),
                    seq.ptr(),
                    dims,
                    (gdn_scale(d.head_k as u32), EPS),
                    std::ptr::null_mut(),
                )
            };
            if let Err(e) = launched {
                eprintln!("gdn_decode {} prior={prior}: refused: {e} FAIL", name(d));
                pass = false;
                continue;
            }
            let got = out.download().expect("download out");
            let out_bad = mismatched(&got, &want.out);
            let differ = |x: Vec<f32>, y: Vec<f32>| {
                (bits_of(&x).iter().zip(bits_of(&y)))
                    .filter(|(p, q)| **p != *q)
                    .count()
            };
            let ring = differ(fused.conv_slot(3), chain.conv_slot(3));
            let state = differ(fused.ssm_slot(3), chain.ssm_slot(3));
            let others = fused.others_changed(&before, 3);
            let ok = out_bad + ring + state + others == 0;
            eprintln!(
                "gdn_decode {} prior={prior}: out_mismatched={out_bad} ring_mismatched_bits={ring} \
                 state_mismatched_bits={state} other_slots_changed={others} {}",
                name(d),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the one-launch GDN decode differs from its four launches"
    );
}

// ── FP8 S8 F6a: the chunked scan (`kernels/gdn_chunk_scan.cu`) ─────────────────

/// The chunk the chunked scan works in.
const CHUNK: usize = 64;
/// The chunked scan's lengths (/tmp/a-f6-plan.md §5, F6-1).
const CHUNK_LENGTHS: [usize; 8] = [1, 2, 63, 64, 65, 128, 2048, 4096];

/// `o` against a scan reference under the chunked bound, over the elements `keep` admits.
fn compare_o_chunked(
    s: &StepOut,
    r: &ScanReference,
    keep: impl Fn(usize) -> bool,
) -> F32Comparison {
    let refs: Vec<f64> = r.o.iter().map(|p| p.0).collect();
    compare_f32(&s.o, &refs, |i| gdn_chunk_bound(r.o[i].0, r.o[i].1), keep)
}

/// The chunked scan's state check on element `i` of a reference's final state (`hk` key lanes a
/// row): `2⁻⁷·|ref| + 2⁻¹⁰·(t_last+1)·Smax`, the output check's form (user decision 2026-11-12:
/// the first rank-1 update's bf16 operands put up to ~2⁻⁷ of an element on it, past a term in
/// Smax alone).
fn chunk_state_bound(r: &ScanReference, i: usize, hk: usize) -> f64 {
    GDN_CHUNK_REL * r.state[i].abs() + GDN_CHUNK_STEP * r.state_scale[i / hk]
}

/// [`check_scan`] under the chunked bound: `o` within it, its padding rows unwritten, the slot's
/// state within [`chunk_state_bound`].
fn check_chunk_scan(
    what: &str,
    d: GdnDims,
    s: &StepOut,
    q_len: usize,
    start: (&[f64], Option<&[f64]>, usize),
    after: &[f32],
) -> (bool, ScanReference) {
    let r = scan_reference(d, s, q_len, start, ScanVariant::Real);
    let cmp = compare_o_chunked(s, &r, |_| true);
    let linear = GDN_CHUNK_STEP * r.o.get(cmp.worst).map_or(0.0, |p| p.1);
    let padding = written_past(&s.o, q_len * d.value_dim());
    let hk = d.head_k;
    let st = compare_f32(after, &r.state, |i| chunk_state_bound(&r, i, hk), |_| true);
    let ok = cmp.pass() && st.pass() && padding == 0;
    eprintln!(
        "{what} chunked scan L={q_len} t0={}: o {} linear_term_at_worst={linear:.4e} state {} \
         padding_written={padding} {}",
        start.2,
        cmp.line(),
        st.line(),
        verdict(ok)
    );
    (ok, r)
}

/// [`check_step`] with the chunked scan's check.
fn check_chunk_step(
    what: &str,
    (d, bu): (GdnDims, &Bundle),
    (c, s): (&Chunk, &StepOut),
    start: &Start<'_>,
    (pool, slot): (&Pool, u32),
) -> bool {
    let mut ok = check_conv(what, d, c, &bu.w, start.ring, s, &pool.conv_slot(slot));
    ok &= check_gating(what, d, c, bu, s);
    let (scan_ok, _) = check_chunk_scan(what, d, s, c.q_len, start.scan(), &pool.ssm_slot(slot));
    ok &= scan_ok;
    ok & check_norm(what, d, c, bu, s)
}

// ── FP8 S8 D1: the one-value-head GDN decode (`gdn_decode_head_ker`) ───────────

/// The model a GDN geometry is: the 35B's is the 9b's (16 key heads, 32 value heads, 128 × 128).
fn model(d: GdnDims) -> &'static str {
    match d.nv / d.nk {
        2 => "35B",
        3 => "27B",
        _ => name(d),
    }
}

/// How many bits of two f32 slices differ.
fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    (bits_of(a).iter().zip(bits_of(b)))
        .filter(|(x, y)| **x != *y)
        .count()
}

/// Fresh in slot 1 of a pool NaN-poisoned everywhere, the chunked scan as the third launch, at the
/// 35B's and the 27B's geometries: every stage within its bound (the scan within the chunked one)
/// at every length of [`CHUNK_LENGTHS`], the padding rows unread and unwritten, the other slots
/// bit-unchanged.
#[test]
fn the_chunked_scan_matches_its_reference_at_every_length() {
    let device = h100();
    let mut pass = true;
    for d in [G9, G27] {
        let bu = bundle(d, 0xF600);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, CHUNK_LENGTHS[CHUNK_LENGTHS.len() - 1], 0xF601);
        let (ring, state) = zeros(d);
        for &l in &CHUNK_LENGTHS {
            let pool = Pool::new(&device, d, NAN_F32);
            let before = pool.snapshot();
            let c = chunk(&x, d, 0, l, l + PAD);
            let s = run_step_with(&device, &pool, &bb, d, &c, (1, true), Scan::Chunked)
                .expect("the step launches");
            let start = Start {
                ring: &ring,
                state: &state,
                smax: None,
                t0: 0,
            };
            let what = format!("{} fresh", model(d));
            let ok = check_chunk_step(&what, (d, &bu), (&c, &s), &start, (&pool, 1));
            let others = pool.others_changed(&before, 1);
            eprintln!(
                "{what} chunked L={l}: other_slots_mismatched_bits={others} {}",
                verdict(others == 0)
            );
            pass &= ok && others == 0;
        }
    }
    assert!(pass, "the chunked scan is out of its bound");
}

/// A known, finite start state in slot 2 (the others NaN): the chunked step within bound of the
/// reference started from it, at `q_len` 1, 65 and 300; the other slots bit-unchanged.
#[test]
fn a_continued_chunked_scan_starts_from_its_slot() {
    let device = h100();
    let mut pass = true;
    for d in [G9, G27] {
        let bu = bundle(d, 0xF610);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 300, 0xF611);
        let l = GdnStateLayout::new(4, 16, d.nv as u32, 128, 128, 4);
        let ring = f32_inputs(l.conv_elems_per_slot(), 0xF612, (-1.0, 1.0));
        let state = f32_inputs(l.ssm_elems_per_slot(), 0xF613, (-0.5, 0.5));
        for q_len in [1, 65, 300] {
            let mut pool = Pool::new(&device, d, NAN_F32);
            pool.write_slot(&device, 2, &ring, &state);
            let before = pool.snapshot();
            let c = chunk(&x, d, 0, q_len, q_len + PAD);
            let s = run_step_with(&device, &pool, &bb, d, &c, (2, false), Scan::Chunked)
                .expect("the step launches");
            let start = Start {
                ring: &to_f64(&ring),
                state: &to_f64(&state),
                smax: None,
                t0: 0,
            };
            let what = format!("{} continued", model(d));
            let ok = check_chunk_step(&what, (d, &bu), (&c, &s), &start, (&pool, 2));
            let others = pool.others_changed(&before, 2);
            eprintln!(
                "{what} chunked L={q_len}: other_slots_mismatched_bits={others} {}",
                verdict(others == 0)
            );
            pass &= ok && others == 0;
        }
    }
    assert!(pass, "a continued chunked step is out of its bound");
}

/// The chunked self-checks' inputs (`tokens` rows of geometry `d`): q̂ = k̂, each key head's `q`
/// columns the same values as its `k` columns and its `q` channels the `k` channels' conv taps,
/// the same row at every token, so `o` is the state's own row and the bound's linear term small
/// beside it; `b` drawn in `b_range`, `a_log` the same for every value head.
fn aligned(
    d: GdnDims,
    tokens: usize,
    seed: u64,
    a_log: f32,
    b_range: (f64, f64),
) -> (Inputs, Bundle) {
    let mut x = inputs(d, tokens, seed);
    let (cd, kd, k) = (d.conv_dim(), d.key_dim(), d.conv_kernel);
    for t in 0..tokens {
        for c in 0..kd {
            let v = x.qkv[kd + c];
            x.qkv[t * cd + c] = v;
            x.qkv[t * cd + kd + c] = v;
        }
    }
    x.b = bf16_in(tokens * d.nv, seed ^ 0x04, b_range);
    let mut bu = bundle(d, seed);
    for c in 0..kd {
        for j in 0..k {
            bu.w[c * k + j] = bu.w[(kd + c) * k + j];
        }
    }
    bu.a_log = vec![a_log; d.nv];
    (x, bu)
}

/// The chunked bound must still bite (/tmp/a-f6-plan.md §2): a kernel computing a mutation would
/// fail the real check. Each mutated f64 reference against the chunked scan's `o`, under the REAL
/// reference's bound (the bound the real check applies; a mutation that grows the state would
/// grow its own), puts at least half the judged elements out of it, at `q_len` 128 (two chunks),
/// on [`aligned`] inputs where each mistake is large:
/// - fast decay (`a_log` 1.5), small β (`b` in [-5, -3]): the decay after the update (judged on
///   `t ≥ 1`), chunk 1's decay missing (tokens 64..128, judged on 65..128), `β = b` (`t ≥ 1`);
/// - slow decay (`a_log` -5), β as drawn: no delta-rule correction, the intra-chunk term dropped
///   (T = I; both `t ≥ 1`).
///
/// The kernel passes the real check on the same inputs.
#[test]
fn every_chunked_self_check_misses_by_its_margin() {
    const NEED: f64 = 0.5;
    let device = h100();
    let mut pass = true;
    let q_len = 2 * CHUNK;
    for d in [G9, G27] {
        for (regime, a_log, b_range) in [
            ("fast decay, small beta", 1.5f32, (-5.0, -3.0)),
            ("slow decay", -5.0, (-2.0, 2.0)),
        ] {
            let (x, bu) = aligned(d, q_len, 0xF620, a_log, b_range);
            let bb = upload_bundle(&device, &bu);
            let pool = Pool::new(&device, d, NAN_F32);
            let c = chunk(&x, d, 0, q_len, q_len);
            let s = run_step_with(&device, &pool, &bb, d, &c, (0, true), Scan::Chunked)
                .expect("the step launches");
            let (vd, nv) = (d.value_dim(), d.nv);
            let (_, state) = zeros(d);
            let real = scan_reference(d, &s, q_len, (&state, None, 0), ScanVariant::Real);
            let rc = compare_o_chunked(&s, &real, |_| true);
            eprintln!(
                "chunked self-check {} L={q_len} {regime}: the real check o {} {}",
                model(d),
                rc.line(),
                verdict(rc.pass())
            );
            pass &= rc.pass();
            let mutated = |variant| scan_reference(d, &s, q_len, (&state, None, 0), variant);
            let cases: Vec<(&str, ScanReference, usize)> = if a_log > 0.0 {
                // β = b: the reference from the GPU's conv_out and g, with b itself as β.
                let (_, b_raw) = gdn_gating_reference(
                    &widen(&c.a[..q_len * nv]),
                    &widen(&c.b[..q_len * nv]),
                    &to_f64(&bu.a_log),
                    &to_f64(&bu.dt_bias),
                    nv,
                    false,
                );
                let conv_out = to_f64(&s.conv_out[..q_len * d.conv_dim()]);
                let gpu_g = to_f64(&s.g[..q_len * nv]);
                let beta_b = gdn_scan_reference(
                    &ScanInputs {
                        conv_out: &conv_out,
                        g: &gpu_g,
                        beta: &b_raw,
                        state: &state,
                        smax: None,
                        t0: 0,
                        q_len,
                    },
                    d,
                    ScanVariant::Real,
                );
                vec![
                    ("decay after the update", mutated(ScanVariant::DecayLast), 1),
                    (
                        "chunk 1's decay missing",
                        mutated(ScanVariant::NoDecayOver {
                            from: CHUNK,
                            to: 2 * CHUNK,
                        }),
                        CHUNK + 1,
                    ),
                    ("beta = b (no sigmoid)", beta_b, 1),
                ]
            } else {
                vec![
                    (
                        "no delta-rule correction (u = beta*v)",
                        mutated(ScanVariant::NoCorrection),
                        1,
                    ),
                    (
                        "the intra-chunk term dropped (T = I)",
                        mutated(ScanVariant::StaleInChunks { len: CHUNK }),
                        1,
                    ),
                ]
            };
            for (what, m, from) in cases {
                let refs: Vec<f64> = m.o.iter().map(|p| p.0).collect();
                let miss = compare_f32(
                    &s.o,
                    &refs,
                    |i| gdn_chunk_bound(real.o[i].0, real.o[i].1),
                    |i| (from * vd..q_len * vd).contains(&i),
                );
                let ok = miss.fraction() >= NEED;
                eprintln!(
                    "chunked self-check {} L={q_len} {regime}: {what}: out_of_bound={}/{} \
                     ({:.4}, need >= {NEED}) {}",
                    model(d),
                    miss.out_of_bound,
                    miss.n,
                    miss.fraction(),
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "a chunked self-check's mutated reference passed");
}

/// The scan's exact properties, on the chunked scan: (a) 35B, NaN in key head `j`'s `q` and `k`
/// reaches exactly value heads `2j` and `2j + 1`; (b) a zero value head's `o` and final state stay
/// exactly zero; (c) padding rows (NaN in every input) reach nothing, the real rows and the state
/// bit-identical to the same tokens with no padding; (d) a fresh start in a slot holding NaN, or
/// a previous sequence's large state, gives the same bits, the other slots untouched; (e) two
/// identical runs give identical bits.
#[test]
fn the_chunked_scan_keeps_the_scans_exact_properties() {
    let device = h100();
    let mut pass = true;
    let run = |pool: &Pool, bb: &BundleBufs, d: GdnDims, c: &Chunk, slot: u32| {
        run_step_with(&device, pool, bb, d, c, (slot, true), Scan::Chunked)
            .expect("the step launches")
    };
    // (a)
    {
        let d = G9;
        let bb = upload_bundle(&device, &bundle(d, 0xF630));
        for j in [0usize, 5, 15] {
            let mut x = inputs(d, 100, 0xF631);
            let (cd, kd, hk) = (d.conv_dim(), d.key_dim(), d.head_k);
            for t in 0..100 {
                for c in 0..hk {
                    x.qkv[t * cd + j * hk + c] = NAN_BF16;
                    x.qkv[t * cd + kd + j * hk + c] = NAN_BF16;
                }
            }
            let s = run(
                &Pool::new(&device, d, 0),
                &bb,
                d,
                &chunk(&x, d, 0, 100, 100),
                0,
            );
            let (vd, hv) = (d.value_dim(), d.head_v);
            let non_finite: Vec<usize> = (0..d.nv)
                .filter(|&h| {
                    (0..100).any(|t| (0..hv).any(|i| !s.o[t * vd + h * hv + i].is_finite()))
                })
                .collect();
            let want = vec![2 * j, 2 * j + 1];
            let ok = non_finite == want;
            eprintln!(
                "chunked key-head probe j={j}: value heads with non-finite o = {non_finite:?} \
                 (want {want:?}) {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    for d in [G9, G27] {
        let bb = upload_bundle(&device, &bundle(d, 0xF640));
        let (cd, kd, hv, vd) = (d.conv_dim(), d.key_dim(), d.head_v, d.value_dim());
        // (b)
        let mut x = inputs(d, 300, 0xF641);
        let live = 3usize;
        for t in 0..300 {
            for h in (0..d.nv).filter(|&h| h != live) {
                for i in 0..hv {
                    x.qkv[t * cd + 2 * kd + h * hv + i] = 0;
                }
            }
        }
        let pool = Pool::new(&device, d, NAN_F32);
        let s = run(&pool, &bb, d, &chunk(&x, d, 0, 300, 300), 1);
        let o_nonzero = (0..300 * vd)
            .filter(|&i| (i % vd) / hv != live && s.o[i] != 0.0)
            .count();
        let per_head = hv * d.head_k;
        let state = pool.ssm_slot(1);
        let s_nonzero = (0..state.len())
            .filter(|&i| i / per_head != live && state[i] != 0.0)
            .count();
        let live_nonzero = (0..300 * vd)
            .filter(|&i| (i % vd) / hv == live && s.o[i] != 0.0)
            .count();
        let ok = o_nonzero == 0 && s_nonzero == 0 && live_nonzero > 0;
        eprintln!(
            "chunked value-isolation probe {}: other heads' o not zero = {o_nonzero}, state not \
             zero = {s_nonzero}, live head's o non-zero = {live_nonzero} {}",
            model(d),
            verdict(ok)
        );
        pass &= ok;
        // (c)
        let x = inputs(d, 100, 0xF642);
        let (pool_p, pool_e) = (
            Pool::new(&device, d, NAN_F32),
            Pool::new(&device, d, NAN_F32),
        );
        let sp = run(&pool_p, &bb, d, &chunk(&x, d, 0, 100, 128), 0);
        let se = run(&pool_e, &bb, d, &chunk(&x, d, 0, 100, 100), 0);
        let n = 100 * vd;
        let zeros_ok = sp.out[n..].iter().all(|&b| b == 0);
        let o_padding = written_past(&sp.o, n);
        let same = differing_bits(&sp.o[..n], &se.o) + mismatched(&sp.out[..n], &se.out);
        let state_bits = differing_bits(&pool_p.ssm_slot(0), &pool_e.ssm_slot(0));
        let ok = zeros_ok && o_padding == 0 && same == 0 && state_bits == 0;
        eprintln!(
            "chunked padding {}: padding_rows_plus_zero={zeros_ok} o_padding_written={o_padding} \
             real_rows_mismatched={same} state_mismatched_bits={state_bits} {}",
            model(d),
            verdict(ok)
        );
        pass &= ok;
        // (d)
        let c = chunk(&x, d, 0, 100, 100 + PAD);
        let pool_a = Pool::new(&device, d, NAN_F32);
        let before = pool_a.snapshot();
        let sa = run(&pool_a, &bb, d, &c, 2);
        let finite = pool_a.ssm_slot(2).iter().all(|v| v.is_finite());
        let others = pool_a.others_changed(&before, 2);
        let mut pool_b = Pool::new(&device, d, NAN_F32);
        let l = pool_b.layout;
        let big_ring = f32_inputs(l.conv_elems_per_slot(), 0xF643, (-1e4, 1e4));
        let big_state = f32_inputs(l.ssm_elems_per_slot(), 0xF644, (-1e4, 1e4));
        pool_b.write_slot(&device, 2, &big_ring, &big_state);
        let sb = run(&pool_b, &bb, d, &c, 2);
        let b_vs_a =
            differing_bits(&sa.o, &sb.o) + differing_bits(&pool_a.ssm_slot(2), &pool_b.ssm_slot(2));
        let ok = finite && others == 0 && b_vs_a == 0;
        eprintln!(
            "chunked stale state {}: slot_2_finite={finite} other_slots_mismatched_bits={others} \
             stale_vs_nan_mismatched_bits={b_vs_a} {}",
            model(d),
            verdict(ok)
        );
        pass &= ok;
        // (e)
        let c = chunk(&inputs(d, 1024, 0xF645), d, 0, 1024, 1024 + PAD);
        let twice = || {
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run(&pool, &bb, d, &c, 0);
            (bits_of(&s.o), pool.snapshot())
        };
        let (a, b) = (twice(), twice());
        let differ = usize::from(a.0 != b.0) + usize::from(a.1 != b.1);
        eprintln!(
            "chunked repeats {} L=1024: differing_buffers={differ} {}",
            model(d),
            verdict(differ == 0)
        );
        pass &= differ == 0;
    }
    assert!(
        pass,
        "the chunked scan lost one of the scan's exact properties"
    );
}

/// One forward of `L1 + L2` tokens against the same tokens as two steps, the second from the
/// state the first left, both chunked. With `L1` a multiple of 64 the second step's chunks are the
/// one forward's: its outputs and the final state bit-identical. Off a multiple of 64 the chunks
/// differ: the second step's `o` and the final state within the chunked bound of the whole
/// sequence's reference.
#[test]
fn chunk_aligned_pieces_are_bit_identical_to_one_forward() {
    let device = h100();
    let mut pass = true;
    for d in [G9, G27] {
        let bu = bundle(d, 0xF650);
        let bb = upload_bundle(&device, &bu);
        for (l1, l2) in [
            (64, 64),
            (64, 1),
            (128, 300),
            (64, 212),
            (100, 200),
            (3, 61),
        ] {
            let x = inputs(d, l1 + l2, 0xF651 ^ (l1 as u64) << 16 ^ l2 as u64);
            let pool_one = Pool::new(&device, d, NAN_F32);
            let one = chunk(&x, d, 0, l1 + l2, l1 + l2);
            let so = run_step_with(&device, &pool_one, &bb, d, &one, (3, true), Scan::Chunked)
                .expect("one forward");
            let pool_two = Pool::new(&device, d, NAN_F32);
            let c1 = chunk(&x, d, 0, l1, l1 + PAD);
            run_step_with(&device, &pool_two, &bb, d, &c1, (3, true), Scan::Chunked)
                .expect("piece 1");
            let c2 = chunk(&x, d, l1, l2, l2 + PAD);
            let s2 = run_step_with(&device, &pool_two, &bb, d, &c2, (3, false), Scan::Chunked)
                .expect("piece 2");
            let vd = d.value_dim();
            let (after_one, after_two) = (pool_one.ssm_slot(3), pool_two.ssm_slot(3));
            if l1 % CHUNK == 0 {
                let o = differing_bits(&so.o[l1 * vd..][..l2 * vd], &s2.o[..l2 * vd]);
                let out = mismatched(&so.out[l1 * vd..][..l2 * vd], &s2.out[..l2 * vd]);
                let state = differing_bits(&after_one, &after_two);
                let ok = o + out + state == 0;
                eprintln!(
                    "chunked continuity {} ({l1}, {l2}), aligned: o_mismatched={o} \
                     out_mismatched={out} state_mismatched_bits={state} {}",
                    model(d),
                    verdict(ok)
                );
                pass &= ok;
            } else {
                // The whole sequence's reference, from the one forward's own conv_out, g and β
                // (piece 2's equal them: the conv and gating are bit-continuous).
                let (_, state) = zeros(d);
                let r = scan_reference(d, &so, l1 + l2, (&state, None, 0), ScanVariant::Real);
                let refs: Vec<f64> = r.o.iter().map(|p| p.0).collect();
                let cmp = compare_f32(
                    &s2.o[..l2 * vd],
                    &refs[l1 * vd..],
                    |i| gdn_chunk_bound(r.o[l1 * vd + i].0, r.o[l1 * vd + i].1),
                    |_| true,
                );
                let hk = d.head_k;
                let st = compare_f32(
                    &after_two,
                    &r.state,
                    |i| chunk_state_bound(&r, i, hk),
                    |_| true,
                );
                let ok = cmp.pass() && st.pass();
                eprintln!(
                    "chunked continuity {} ({l1}, {l2}), off the chunk: o {} state {} {}",
                    model(d),
                    cmp.line(),
                    st.line(),
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(
        pass,
        "chunked pieces differ from one forward: stop and report"
    );
}

/// Decode after a chunked prefill: a prefill of `L` tokens by the chunked scan, then 8 one-row
/// decode steps (`rows = 1`, today's recurrent scan, the four launches the one-launch decode is
/// bit for bit), each from the state the last left. Every decode token's `o` and the final state
/// within the chunked bound of the f64 recurrence of the whole sequence (the state they start from
/// carries the chunked prefill's error), at the 35B's and 27B's geometries, `L` 300 and 2048.
#[test]
fn a_decode_after_a_chunked_prefill_follows_the_recurrence() {
    const DECODES: usize = 8;
    let device = h100();
    let mut pass = true;
    for d in [G9, G27] {
        let bu = bundle(d, 0xF660);
        let bb = upload_bundle(&device, &bu);
        for l in [300usize, 2048] {
            let x = inputs(d, l + DECODES, 0xF661 ^ l as u64);
            let pool = Pool::new(&device, d, NAN_F32);
            let pre = run_step_with(
                &device,
                &pool,
                &bb,
                d,
                &chunk(&x, d, 0, l, l + PAD),
                (1, true),
                Scan::Chunked,
            )
            .expect("the prefill");
            let (cd, nv, vd) = (d.conv_dim(), d.nv, d.value_dim());
            let (mut conv_out, mut g, mut beta) = (
                pre.conv_out[..l * cd].to_vec(),
                pre.g[..l * nv].to_vec(),
                pre.beta[..l * nv].to_vec(),
            );
            let mut o = Vec::with_capacity(DECODES * vd);
            for t in l..l + DECODES {
                let s = run_step(&device, &pool, &bb, d, &chunk(&x, d, t, 1, 1), (1, false))
                    .expect("a decode step");
                conv_out.extend_from_slice(&s.conv_out);
                g.extend_from_slice(&s.g);
                beta.extend_from_slice(&s.beta);
                o.extend_from_slice(&s.o);
            }
            let (_, state) = zeros(d);
            let (conv_out, g, beta) = (to_f64(&conv_out), to_f64(&g), to_f64(&beta));
            let r = gdn_scan_reference(
                &ScanInputs {
                    conv_out: &conv_out,
                    g: &g,
                    beta: &beta,
                    state: &state,
                    smax: None,
                    t0: 0,
                    q_len: l + DECODES,
                },
                d,
                ScanVariant::Real,
            );
            let refs: Vec<f64> = r.o.iter().map(|p| p.0).collect();
            let cmp = compare_f32(
                &o,
                &refs[l * vd..],
                |i| gdn_chunk_bound(r.o[l * vd + i].0, r.o[l * vd + i].1),
                |_| true,
            );
            let hk = d.head_k;
            let st = compare_f32(
                &pool.ssm_slot(1),
                &r.state,
                |i| chunk_state_bound(&r, i, hk),
                |_| true,
            );
            let ok = cmp.pass() && st.pass();
            eprintln!(
                "decode after a chunked prefill {} L={l} + {DECODES}: o {} state {} {}",
                model(d),
                cmp.line(),
                st.line(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a decode after a chunked prefill left the recurrence's bound"
    );
}

/// The chunked scan refuses exactly what the scan refuses (both refusal tests' scan cases), before
/// it touches a pointer, its output and the state left as they were; its timing split refuses the
/// same and a `part` above 2.
#[test]
fn the_chunked_scan_refuses_what_the_scan_refuses() {
    let device = h100();
    let small = |n: usize| up_f32_bits(&device, n, NAN_F32);
    let (x, st, seq, out) = (small(1 << 12), small(1 << 12), small(4), small(1 << 12));
    let unchanged = |b: &DeviceBuffer| down_f32(b).iter().all(|v| v.to_bits() == NAN_F32);
    let null = std::ptr::null_mut();
    let scale = gdn_scale(128);
    let nv9 = G9.nv as u32;
    let mut pass = true;
    // `(rows, nk, nv, head_k, head_v, slot_elems, scale)`.
    let cases: Vec<(u32, u32, u32, u32, u32, u64, f32)> = [
        (4u32, 1u32, 1u32, 64u32, 32u32, 32u64 * 128, scale),
        (4, 1, 1, 128, 40, 40 * 128, scale),
        (4, 1, 1, 128, 512, 512 * 128, scale),
        (4, 2, 3, 128, 32, 3 * 32 * 128, scale),
        (4, 1, 1, 128, 32, 32 * 64, scale),
        (4, 1, 1, 128, 32, 32 * 128, 0.0),
    ]
    .into_iter()
    .chain(
        [
            (0u32, 16u32, nv9, 128u32),
            (4, 0, nv9, 128),
            (4, 16, 0, 128),
            (4, 16, nv9, 0),
            (1 << 18, 16, nv9, 128),
        ]
        .map(|(rows, nk, nv, hv)| {
            let se = u64::from(nv) * u64::from(hv) * 128;
            (rows, nk, nv, 128, hv, se, scale)
        }),
    )
    .collect();
    for (rows, nk, nv, hk, hv, se, sc) in cases {
        let want = KernelError::GdnScanShape {
            rows,
            nk,
            nv,
            head_k: hk,
            head_v: hv,
        };
        for part in [None, Some(0u32), Some(1), Some(2)] {
            // SAFETY: refused before any pointer is touched; every buffer is a live allocation.
            let got = unsafe {
                match part {
                    None => gdn_chunk_scan(
                        (x.ptr(), x.ptr(), x.ptr()),
                        (st.ptr(), se),
                        seq.ptr(),
                        out.ptr(),
                        (rows, nk, nv),
                        (hk, hv, sc),
                        null,
                    ),
                    Some(p) => gdn_chunk_scan_part(
                        (x.ptr(), x.ptr(), x.ptr()),
                        (st.ptr(), se),
                        seq.ptr(),
                        out.ptr(),
                        (rows, nk, nv),
                        (hk, hv, sc),
                        (p, null, null),
                    ),
                }
            };
            let (o, s) = (unchanged(&out), unchanged(&st));
            let ok = got == Err(want) && o && s;
            eprintln!(
                "chunked refused part={part:?} rows={rows} nk={nk} nv={nv} head_k={hk} \
                 head_v={hv} slot_elems={se} scale={sc}: {got:?} output_unchanged={o} \
                 state_unchanged={s} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    // A shape it takes, at a part it does not (4), and the traced part with no trace (3).
    let se = u64::from(nv9) * 128 * 128;
    let want = KernelError::GdnScanShape {
        rows: 4,
        nk: 16,
        nv: nv9,
        head_k: 128,
        head_v: 128,
    };
    for part in [4u32, 3] {
        // SAFETY: as above.
        let got = unsafe {
            gdn_chunk_scan_part(
                (x.ptr(), x.ptr(), x.ptr()),
                (st.ptr(), se),
                seq.ptr(),
                out.ptr(),
                (4, 16, nv9),
                (128, 128, scale),
                (part, null, null),
            )
        };
        let ok = got == Err(want) && unchanged(&out) && unchanged(&st);
        eprintln!(
            "chunked refused part={part} (no trace): {got:?} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the chunked scan took a shape the scan refuses, or wrote its output"
    );
}

/// The chunked scan's time per launch (one GDN layer's scan) beside today's recurrent scan, at the
/// 27B's (nv 48) and the 35B's (nv 32) geometries and `q_len` 128, 2048 and 4096, fresh, over one
/// step's own conv_out, g and β; and its split: the launch with the solve skipped (T = I) and the
/// launch stopped after the precompute. Each 20 launches after a warm-up, timed to the download
/// that waits for them, in the order A, B, A, B (A today's, B the chunked and its parts). Printed
/// as `F6_TIME` lines; then one traced launch's phase split (each CTA's thread 0's cycles per
/// phase and chunk, averaged over the CTAs, and each phase's share) as an `F6_PHASE` line.
/// Recorded, not asserted.
#[test]
fn the_chunked_scan_time_is_recorded() {
    const LAUNCHES: u32 = 20;
    let device = h100();
    for d in [G27, G9] {
        let bu = bundle(d, 0xF670);
        let bb = upload_bundle(&device, &bu);
        for q_len in [128usize, 2048, 4096] {
            let x = inputs(d, q_len, 0xF671 ^ q_len as u64);
            let c = chunk(&x, d, 0, q_len, q_len);
            let pool = Pool::new(&device, d, NAN_F32);
            let s = run_step(&device, &pool, &bb, d, &c, (0, true)).expect("the step launches");
            let (cv, g, beta) = (
                up_f32(&device, &s.conv_out),
                up_f32(&device, &s.g),
                up_f32(&device, &s.beta),
            );
            let o = up_f32_bits(&device, q_len * d.value_dim(), NAN_F32);
            let seq = up_u32(&device, &[0, q_len as u32, 1, 0]);
            // `None` today's scan, `Some(part)` the chunked scan's.
            let launch = |part: Option<u32>| {
                let (rows, slot) = (
                    (q_len as u32, d.nk as u32, d.nv as u32),
                    (pool.ssm.ptr(), pool.layout.ssm_elems_per_slot() as u64),
                );
                let heads = (d.head_k as u32, d.head_v as u32, gdn_scale(d.head_k as u32));
                let null = std::ptr::null_mut();
                // SAFETY: as `run_step`'s, for the scan alone over this step's own scratch.
                unsafe {
                    match part {
                        None => gdn_scan(
                            (cv.ptr(), g.ptr(), beta.ptr()),
                            slot,
                            seq.ptr(),
                            o.ptr(),
                            rows,
                            heads,
                            null,
                        ),
                        Some(p) => gdn_chunk_scan_part(
                            (cv.ptr(), g.ptr(), beta.ptr()),
                            slot,
                            seq.ptr(),
                            o.ptr(),
                            rows,
                            heads,
                            (p, null, null),
                        ),
                    }
                }
                .expect("the scan launches");
            };
            let time = |part: Option<u32>| {
                launch(part);
                seq.download().expect("wait for the warm-up");
                let start = std::time::Instant::now();
                for _ in 0..LAUNCHES {
                    launch(part);
                }
                seq.download().expect("wait for the launches");
                start.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES)
            };
            let parts = [None, Some(0), Some(1), Some(2)];
            let mut us = [0.0f64; 4];
            for _ in 0..2 {
                for (k, &p) in parts.iter().enumerate() {
                    us[k] += time(p) / 2.0;
                }
            }
            let [today, full, no_solve, precompute] = us;
            eprintln!(
                "F6_TIME {} L={q_len}: today={today:.1} us chunked={full:.1} us ({:.2}x) \
                 no_solve={no_solve:.1} us precompute_only={precompute:.1} us solve_share={:.3} \
                 sequential_share={:.3}",
                model(d),
                today / full,
                (full - no_solve) / full,
                (full - precompute) / full,
            );
            // The phase split: one traced launch, each CTA's thread 0's cycles per phase.
            let ctas = (d.head_v / 32) * d.nv;
            let trace = up_f32_bits(&device, ctas * (PHASES.len() + 1), NAN_F32);
            // SAFETY: as `launch`'s; `trace` holds `ctas · 13` f32.
            unsafe {
                gdn_chunk_scan_part(
                    (cv.ptr(), g.ptr(), beta.ptr()),
                    (pool.ssm.ptr(), pool.layout.ssm_elems_per_slot() as u64),
                    seq.ptr(),
                    o.ptr(),
                    (q_len as u32, d.nk as u32, d.nv as u32),
                    (d.head_k as u32, d.head_v as u32, gdn_scale(d.head_k as u32)),
                    (3, trace.ptr(), std::ptr::null_mut()),
                )
            }
            .expect("the traced scan launches");
            let t = down_f32(&trace);
            let cols = PHASES.len() + 1;
            let chunks = f64::from(t[cols - 1]);
            let mean =
                |p: usize| (0..ctas).map(|c| f64::from(t[c * cols + p])).sum::<f64>() / ctas as f64;
            let total: f64 = (0..PHASES.len()).map(mean).sum();
            let split: Vec<String> = (PHASES.iter().enumerate())
                .map(|(p, name)| {
                    let m = mean(p);
                    format!("{name}={:.0} ({:.3})", m / chunks, m / total)
                })
                .collect();
            eprintln!(
                "F6_PHASE {} L={q_len}: chunks={chunks} cycles_per_chunk={:.0} {}",
                model(d),
                total / chunks,
                split.join(" ")
            );
        }
    }
}

/// The chunked scan's traced phases (`kernels/gdn_chunk_scan.cu`'s `PHASES`), in order; its trace's
/// last column is the chunk count.
const PHASES: [&str; 12] = [
    "state_load",
    "rows",
    "g_beta_G",
    "kk_qk_qs",
    "a_n_p",
    "diag_solve",
    "t_products",
    "u_w",
    "v_new",
    "o",
    "state_update",
    "state_store",
];

/// Whether the one-launch decode (`gdn_decode`) takes geometry `d`: the decode tapes' rule
/// (`steps_from_tape`'s `gdn_decode_takes`), else a one-row bucket's GDN step is the four launches
/// with the recurrent scan (`step::gdn_scan_is_recurrent`): the 27B's three value heads a key head.
fn gdn_decode_takes(d: GdnDims) -> bool {
    let heads = d.nv / d.nk;
    (d.head_v == 128 || d.head_v == 256)
        && d.nv.is_multiple_of(d.nk)
        && (1..=4).contains(&heads)
        && 16usize.is_multiple_of(heads)
        && 2 + heads * d.head_v / 128 <= 8
}

/// FP8 S8 F6-2, decode after a chunked prefill at the real tapes' shapes: the 35B's and 27B's GDN
/// geometries, a prefill bucket of 512 and of 2,048 rows holding `rows − 37` real tokens (the
/// rest padding) by the chunked scan, then 8 decodes as the decode tapes run them, each from the
/// slot the last left: the one-launch decode (`gdn_decode`) where it takes the geometry (the
/// 35B), else the four launches at one row with the recurrent scan (the 27B). The slot's state
/// after the decodes within the chunked state bound of the f64 recurrence over the whole sequence
/// (the prefill's own conv_out, g and β, then the decode tokens' from the CPU conv and gating, from
/// the ring the prefill left), and every decode's output finite.
#[test]
fn a_decode_after_a_chunked_prefill_bucket_follows_the_recurrence() {
    use scratchy_target_cuda::kernels::{GdnDecodeOperands, gdn_decode};
    const DECODES: usize = 8;
    let device = h100();
    let mut pass = true;
    for d in [G9, G27] {
        let bu = bundle(d, 0xF680);
        let bb = upload_bundle(&device, &bu);
        for rows in [512usize, 2048] {
            let q_len = rows - 37;
            let x = inputs(d, q_len + DECODES, 0xF681 ^ rows as u64);
            let pool = Pool::new(&device, d, NAN_F32);
            let pre = run_step_with(
                &device,
                &pool,
                &bb,
                d,
                &chunk(&x, d, 0, q_len, rows),
                (1, true),
                Scan::Chunked,
            )
            .expect("the prefill");
            let ring = to_f64(&pool.conv_slot(1));
            let l = pool.layout;
            let dims = (
                d.nk as u32,
                d.nv as u32,
                d.head_k as u32,
                d.head_v as u32,
                d.conv_kernel as u32,
            );
            let mut finite = true;
            let one_launch = gdn_decode_takes(d);
            for t in q_len..q_len + DECODES {
                let tok = chunk(&x, d, t, 1, 1);
                if !one_launch {
                    let s = run_step(&device, &pool, &bb, d, &tok, (1, false)).expect("a decode");
                    finite &= s.out.iter().all(|&h| (h & 0x7F80) != 0x7F80);
                    continue;
                }
                let seq = up_u32(&device, &[1, 1, 0, 0]);
                let (qkv, a, b, z) = (
                    up(&device, &tok.qkv),
                    up(&device, &tok.a),
                    up(&device, &tok.b),
                    up(&device, &tok.z),
                );
                let out = up(&device, &vec![NAN_BF16; d.value_dim()]);
                let operands = GdnDecodeOperands {
                    qkv: qkv.ptr(),
                    a: a.ptr(),
                    b: b.ptr(),
                    z: z.ptr(),
                    w: bb.w.ptr(),
                    a_log: bb.a_log.ptr(),
                    dt_bias: bb.dt_bias.ptr(),
                    norm: bb.norm.ptr(),
                    out: out.ptr(),
                };
                // SAFETY: as `run_step`'s at one row: every buffer holds what the launcher's
                // contract names, the pool `SLOTS` slots of its layout and slot 1 below it, `q_len`
                // 1; null is the default stream, which the downloads below wait for.
                unsafe {
                    gdn_decode(
                        &operands,
                        (pool.conv.ptr(), l.conv_elems_per_slot() as u64),
                        (pool.ssm.ptr(), l.ssm_elems_per_slot() as u64),
                        seq.ptr(),
                        dims,
                        (gdn_scale(d.head_k as u32), EPS),
                        std::ptr::null_mut(),
                    )
                }
                .expect("a decode launches");
                let got = out.download().expect("download out");
                finite &= got.iter().all(|&h| (h & 0x7F80) != 0x7F80);
            }
            let (cd, nv) = (d.conv_dim(), d.nv);
            let span = |v: &[u16], cols: usize| widen(&v[q_len * cols..(q_len + DECODES) * cols]);
            let (conv_ref, _) = gdn_conv_reference(
                &span(&x.qkv, cd),
                &to_f64(&bu.w),
                &ring,
                (DECODES, cd, d.conv_kernel),
                false,
            );
            let (g_ref, beta_ref) = gdn_gating_reference(
                &span(&x.a, nv),
                &span(&x.b, nv),
                &to_f64(&bu.a_log),
                &to_f64(&bu.dt_bias),
                nv,
                true,
            );
            let mut conv_out = to_f64(&pre.conv_out[..q_len * cd]);
            conv_out.extend(conv_ref.iter().map(|p| p.0));
            let mut g = to_f64(&pre.g[..q_len * nv]);
            g.extend(g_ref);
            let mut beta = to_f64(&pre.beta[..q_len * nv]);
            beta.extend(beta_ref);
            let (_, state) = zeros(d);
            let r = gdn_scan_reference(
                &ScanInputs {
                    conv_out: &conv_out,
                    g: &g,
                    beta: &beta,
                    state: &state,
                    smax: None,
                    t0: 0,
                    q_len: q_len + DECODES,
                },
                d,
                ScanVariant::Real,
            );
            let hk = d.head_k;
            let st = compare_f32(
                &pool.ssm_slot(1),
                &r.state,
                |i| chunk_state_bound(&r, i, hk),
                |_| true,
            );
            let ok = st.pass() && finite;
            let by = if one_launch {
                "gdn_decode"
            } else {
                "the four launches (recurrent scan)"
            };
            eprintln!(
                "decode by {by} after a chunked prefill bucket {} rows={rows} q_len={q_len} + \
                 {DECODES}: state {} outputs_finite={finite} {}",
                model(d),
                st.line(),
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a decode after a chunked prefill bucket left the recurrence's bound"
    );
}

/// Whether the one-key-head decode (variant 0, #1049's) takes geometry `d`: at most 4 value heads
/// a key head, dividing 16, their v channels within its 8 warps (the 27B's 3 do not divide 16). The
/// one-value-head form (variant 1, the launcher's) takes every `heads` up to 8.
fn one_key_head_takes(d: GdnDims) -> bool {
    let heads = d.nv / d.nk;
    heads <= 4 && 16usize.is_multiple_of(heads) && 2 + heads * d.head_v / 128 <= 8
}

/// A one-row step's four launches (conv1d, gating, the recurrent scan, the gated norm) over
/// buffers allocated once, so a graph can capture them: the 27B's decode today.
struct FourLaunches {
    qkv: DeviceBuffer,
    z: DeviceBuffer,
    a: DeviceBuffer,
    b: DeviceBuffer,
    seq: DeviceBuffer,
    conv_out: DeviceBuffer,
    g: DeviceBuffer,
    beta: DeviceBuffer,
    o: DeviceBuffer,
    out: DeviceBuffer,
}

impl FourLaunches {
    fn new(device: &CudaDevice, d: GdnDims, tok: &Chunk) -> Self {
        Self {
            qkv: up(device, &tok.qkv),
            z: up(device, &tok.z),
            a: up(device, &tok.a),
            b: up(device, &tok.b),
            seq: up_u32(device, &[3, 1, 0, 0]),
            conv_out: up_f32_bits(device, d.conv_dim(), NAN_F32),
            g: up_f32_bits(device, d.nv, NAN_F32),
            beta: up_f32_bits(device, d.nv, NAN_F32),
            o: up_f32_bits(device, d.value_dim(), NAN_F32),
            out: up(device, &vec![NAN_BF16; d.value_dim()]),
        }
    }

    /// The four launches over `pool`'s slot 3, continued, on `stream`.
    fn launch(&self, (d, bb): (GdnDims, &BundleBufs), pool: &Pool, stream: *mut std::ffi::c_void) {
        let l = pool.layout;
        let (cd, nv, hv) = (d.conv_dim() as u32, d.nv as u32, d.head_v as u32);
        // SAFETY: as `run_step`'s at one row, on `stream` (null or live).
        unsafe {
            gdn_conv1d(
                (self.qkv.ptr(), bb.w.ptr()),
                (pool.conv.ptr(), l.conv_elems_per_slot() as u64),
                self.seq.ptr(),
                self.conv_out.ptr(),
                (1, cd, d.conv_kernel as u32),
                stream,
            )
            .expect("conv1d");
            gdn_gating(
                (self.a.ptr(), self.b.ptr()),
                (bb.a_log.ptr(), bb.dt_bias.ptr()),
                self.seq.ptr(),
                (self.g.ptr(), self.beta.ptr()),
                (1, nv),
                stream,
            )
            .expect("gating");
            gdn_scan(
                (self.conv_out.ptr(), self.g.ptr(), self.beta.ptr()),
                (pool.ssm.ptr(), l.ssm_elems_per_slot() as u64),
                self.seq.ptr(),
                self.o.ptr(),
                (1, d.nk as u32, nv),
                (d.head_k as u32, hv, gdn_scale(d.head_k as u32)),
                stream,
            )
            .expect("scan");
            gdn_rms_norm_gated(
                (self.o.ptr(), self.z.ptr(), bb.norm.ptr()),
                self.seq.ptr(),
                self.out.ptr(),
                (1, nv, hv),
                EPS,
                stream,
            )
            .expect("gated norm");
        }
    }
}

/// One decode token's device operands at geometry `d`, over `pool`'s slot 3.
struct DecodeCase {
    qkv: DeviceBuffer,
    a: DeviceBuffer,
    b: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    seq: DeviceBuffer,
}

impl DecodeCase {
    fn new(device: &CudaDevice, d: GdnDims, tok: &Chunk, fresh: bool) -> Self {
        Self {
            qkv: up(device, &tok.qkv),
            a: up(device, &tok.a),
            b: up(device, &tok.b),
            z: up(device, &tok.z),
            out: up(device, &vec![NAN_BF16; d.value_dim()]),
            seq: up_u32(device, &[3, 1, u32::from(fresh), 0]),
        }
    }

    /// The decode at `variant` over `pool`, `trace` its phases when not null, on `stream`.
    fn launch(
        &self,
        (d, bb): (GdnDims, &BundleBufs),
        pool: &Pool,
        (variant, trace, stream): (u32, *mut std::ffi::c_void, *mut std::ffi::c_void),
    ) -> Result<(), KernelError> {
        use scratchy_target_cuda::kernels::{GdnDecodeOperands, gdn_decode_variant};
        let operands = GdnDecodeOperands {
            qkv: self.qkv.ptr(),
            a: self.a.ptr(),
            b: self.b.ptr(),
            z: self.z.ptr(),
            w: bb.w.ptr(),
            a_log: bb.a_log.ptr(),
            dt_bias: bb.dt_bias.ptr(),
            norm: bb.norm.ptr(),
            out: self.out.ptr(),
        };
        let l = pool.layout;
        // SAFETY: as `run_step`'s at one row: every buffer holds what the launcher's contract
        // names, the pool `SLOTS` slots of its layout and slot 3 below it, `q_len` 1; `trace`, when
        // given, holds `nv · 8` f32; `stream` null or live.
        unsafe {
            gdn_decode_variant(
                &operands,
                (pool.conv.ptr(), l.conv_elems_per_slot() as u64),
                (pool.ssm.ptr(), l.ssm_elems_per_slot() as u64),
                self.seq.ptr(),
                (
                    d.nk as u32,
                    d.nv as u32,
                    d.head_k as u32,
                    d.head_v as u32,
                    d.conv_kernel as u32,
                ),
                (gdn_scale(d.head_k as u32), EPS),
                (variant, trace, stream),
            )
        }
    }
}

/// FP8 S8 D1: the one-value-head decode (variant 1, the launcher's) is the four launches, bit for
/// bit, at the 0.8B's, the 35B's (9b's) and the 27B's geometries (the 27B's three value heads a key
/// head, which only this form takes), and so is the one-key-head decode (variant 0) where it takes
/// the geometry; fresh (the pool NaN-poisoned) and continued from a slot the four launches filled
/// with 300 tokens: the bf16 output, the conv ring and the state of the slot, and every other slot
/// bit-unchanged.
#[test]
fn the_one_value_head_decode_is_the_one_key_head_decode_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9, G27] {
        let bu = bundle(d, 0xD1A0);
        let bb = upload_bundle(&device, &bu);
        for prior in [0usize, 300] {
            let x = inputs(d, prior + 1, 0xD1A1 ^ (prior as u64) << 16);
            let pools: Vec<Pool> = (0..3).map(|_| Pool::new(&device, d, NAN_F32)).collect();
            if prior > 0 {
                let c = chunk(&x, d, 0, prior, prior);
                for p in &pools {
                    run_step(&device, p, &bb, d, &c, (3, true)).expect("the prefix");
                }
            }
            let fresh = prior == 0;
            let tok = chunk(&x, d, prior, 1, 1);
            let befores: Vec<_> = pools.iter().map(Pool::snapshot).collect();
            let four = run_step(&device, &pools[0], &bb, d, &tok, (3, fresh)).expect("the four");
            let mut outs = Vec::new();
            for variant in [0u32, 1] {
                if variant == 0 && !one_key_head_takes(d) {
                    continue;
                }
                let c = DecodeCase::new(&device, d, &tok, fresh);
                let p = &pools[1 + variant as usize];
                c.launch(
                    (d, &bb),
                    p,
                    (variant, std::ptr::null_mut(), std::ptr::null_mut()),
                )
                .expect("a decode launches");
                outs.push((
                    variant,
                    c.out.download().expect("out"),
                    p.conv_slot(3),
                    p.ssm_slot(3),
                    p.others_changed(&befores[1 + variant as usize], 3),
                ));
            }
            let (want_out, want_ring, want_state) = (
                four.out.clone(),
                pools[0].conv_slot(3),
                pools[0].ssm_slot(3),
            );
            for (variant, out, ring, state, others) in &outs {
                let o = mismatched(out, &want_out);
                let r = differing_bits(ring, &want_ring);
                let s = differing_bits(state, &want_state);
                let ok = o + r + s + others == 0;
                eprintln!(
                    "D1_BITS {} prior={prior} variant={variant} vs the four launches: \
                     out_mismatched={o} ring_mismatched_bits={r} state_mismatched_bits={s} \
                     other_slots_changed={others} {}",
                    model(d),
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "a GDN decode form differs from the four launches");
}

type CudaEvent = *mut std::ffi::c_void;
const STREAM_BLOCKING: u32 = 0;
const CAPTURE_THREAD_LOCAL: i32 = 1;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn cudaGraphInstantiateWithFlags(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        flags: u64,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
}

/// `launch(stream)` ten times captured into one graph on a blocking stream, replayed 20 times after
/// 3 warm-ups between CUDA events on the default stream (which orders the blocking stream): the
/// median replay's µs per node.
fn graph_per_node(launch: impl Fn(*mut std::ffi::c_void)) -> f64 {
    let (mut stream, mut graph, mut exec) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    let (mut start, mut end): (CudaEvent, CudaEvent) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: out-pointers to locals; nothing else is captured on this thread; every handle is
    // destroyed below.
    unsafe {
        assert_eq!(
            cudaStreamCreateWithFlags(&mut stream, STREAM_BLOCKING),
            0,
            "stream"
        );
        assert_eq!(
            cudaStreamBeginCapture(stream, CAPTURE_THREAD_LOCAL),
            0,
            "begin capture"
        );
        for _ in 0..10 {
            launch(stream);
        }
        assert_eq!(cudaStreamEndCapture(stream, &mut graph), 0, "end capture");
        assert_eq!(
            cudaGraphInstantiateWithFlags(&mut exec, graph, 0),
            0,
            "instantiate"
        );
        assert_eq!(cudaEventCreate(&mut start), 0, "event");
        assert_eq!(cudaEventCreate(&mut end), 0, "event");
    }
    let mut times = Vec::new();
    for i in 0..23 {
        let mut ms = 0.0f32;
        // SAFETY: `exec` is the instantiated graph, `stream` and the events live.
        unsafe {
            assert_eq!(cudaEventRecord(start, std::ptr::null_mut()), 0, "record");
            assert_eq!(cudaGraphLaunch(exec, stream), 0, "graph launch");
            assert_eq!(cudaEventRecord(end, std::ptr::null_mut()), 0, "record");
            assert_eq!(cudaEventSynchronize(end), 0, "synchronize");
            assert_eq!(cudaEventElapsedTime(&mut ms, start, end), 0, "elapsed");
        }
        if i >= 3 {
            times.push(f64::from(ms) * 1e3 / 10.0);
        }
    }
    // SAFETY: the handles were created above and are no longer used.
    unsafe {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Recorded, not asserted (D1's gate): the decode's µs per launch inside a captured graph of 10
/// back-to-back nodes, today's decode against the one-value-head form (`one_value_head`),
/// continued from a slot of 300 tokens, at the 35B's and the 27B's geometries, two rounds (A, B,
/// A, B), as `D1_TIME` lines. Today's is the one-key-head form (`one_key_head`, #1049's) where it
/// takes the geometry (the 35B), else the four launches at one row (`four_launches`, the 27B's
/// decode tape today). Then one traced launch of each form that takes the geometry, its phases'
/// cycles (the mean and the largest over the CTAs, thread 0's from its start at each phase's end),
/// as `D1_PHASE` lines.
#[test]
fn the_gdn_decode_graph_times_and_phases_are_recorded() {
    let device = h100();
    for d in [G9, G27] {
        let bu = bundle(d, 0xD1B0);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 301, 0xD1B1);
        let pool = Pool::new(&device, d, NAN_F32);
        run_step(
            &device,
            &pool,
            &bb,
            d,
            &chunk(&x, d, 0, 300, 300),
            (3, true),
        )
        .expect("prefix");
        let tok = chunk(&x, d, 300, 1, 1);
        let c = DecodeCase::new(&device, d, &tok, false);
        let run = |variant: u32| {
            graph_per_node(|s| {
                c.launch((d, &bb), &pool, (variant, std::ptr::null_mut(), s))
                    .expect("a decode launches")
            })
        };
        let today = one_key_head_takes(d);
        let fl = FourLaunches::new(&device, d, &tok);
        let four = || graph_per_node(|s| fl.launch((d, &bb), &pool, s));
        for round in 1..=2 {
            let (a, b) = (if today { run(0) } else { four() }, run(1));
            eprintln!(
                "D1_TIME {} round {round}: {}={a:.2} us one_value_head={b:.2} us per launch \
                 ({:.2}x)",
                model(d),
                if today {
                    "one_key_head"
                } else {
                    "four_launches"
                },
                a / b
            );
        }
        let names: [&[&str]; 2] = [
            &["conv", "gating", "barrier", "scan", "barrier", "norm"],
            &[
                "prefetch",
                "conv_gating",
                "cluster_barrier",
                "qk_ring",
                "scan",
                "barrier",
                "norm",
            ],
        ];
        for variant in [0u32, 1] {
            if variant == 0 && !today {
                continue;
            }
            let ctas = if variant == 0 { d.nk } else { d.nv };
            let trace = up_f32_bits(&device, ctas * 8, NAN_F32);
            c.launch(
                (d, &bb),
                &pool,
                (variant, trace.ptr(), std::ptr::null_mut()),
            )
            .expect("a traced decode launches");
            let t = down_f32(&trace);
            let phases: Vec<String> = (names[variant as usize].iter().enumerate())
                .map(|(p, name)| {
                    let col: Vec<f64> = (0..ctas).map(|i| f64::from(t[i * 8 + p])).collect();
                    let mean = col.iter().sum::<f64>() / ctas as f64;
                    let max = col.iter().copied().fold(0.0, f64::max);
                    format!("{name}@{mean:.0}/{max:.0}")
                })
                .collect();
            eprintln!(
                "D1_PHASE {} {} ({ctas} CTAs): cycles at each phase's end, mean/max: {}",
                model(d),
                ["one_key_head", "one_value_head"][variant as usize],
                phases.join(" ")
            );
        }
    }
}
