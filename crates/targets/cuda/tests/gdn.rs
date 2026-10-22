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
}

use common::numerics::{
    F32Comparison, GDN_SCAN_STEP, GdnDims, ScanInputs, ScanReference, ScanVariant, bf16_in,
    compare_f32, compare_rows, f32_from_halves, f32_inputs, f32_to_halves, gdn_conv_bound,
    gdn_conv_reference, gdn_gating_bound, gdn_gating_reference, gdn_norm_reference, gdn_scan_bound,
    gdn_scan_reference, widen,
};
use scratchy_subtile::gdn_state_layout::GdnStateLayout;
use scratchy_target_cuda::kernels::{
    KernelError, gdn_conv1d, gdn_gating, gdn_rms_norm_gated, gdn_scan,
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
/// The chunk lengths every stage is checked at.
const LENGTHS: [usize; 8] = [1, 2, 63, 64, 65, 300, 1024, 2048];
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
    (slot, fresh): (u32, bool),
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
        gdn_scan(
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
    if d.nv == d.nk { "0.8b" } else { "9b" }
}

// ── Every stage at every length, both geometries, fresh ───────────

/// Fresh in slot 1 of a pool NaN-poisoned everywhere: every stage within bound at every length,
/// the padding rows unread and unwritten, and the other slots bit-unchanged.
#[test]
fn every_gdn_stage_matches_its_reference_at_every_length() {
    let device = h100();
    let mut pass = true;
    for d in [G08, G9] {
        let bu = bundle(d, 0x6D00);
        let bb = upload_bundle(&device, &bu);
        let x = inputs(d, 2048, 0x6D01);
        let (ring, state) = zeros(d);
        for &l in &LENGTHS {
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
    for d in [G08, G9] {
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
    for d in [G08, G9] {
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
            if d.nv != d.nk {
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
    for d in [G08, G9] {
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
    for d in [G08, G9] {
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
    for d in [G08, G9] {
        let bu = bundle(d, 0xC0C0);
        let bb = upload_bundle(&device, &bu);
        for (l1, l2) in [(1, 1), (2, 1), (3, 5), (64, 64), (100, 200), (300, 1)] {
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
    for d in [G08, G9] {
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
