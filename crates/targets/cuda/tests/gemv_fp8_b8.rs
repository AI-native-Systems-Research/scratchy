// SPDX-License-Identifier: Apache-2.0
//! FP8 S8 T1 on the H100 (/tmp/a-tc-plan.md §1, §3.1, §5.2): B8, the tensor-core decode GEMV in its
//! production form (`kernels/gemv_fp8_tc.cu`'s `gemv_fp8_b8_ker`), and the one-row act + quantize
//! (`kernels/act_quant_fp8.cu`), at every row of §1's decode inventory of the three FP8 models. No
//! tape launches them yet.
//!
//! - S3's f64 bound (`|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹¹·S`, `tests/gemm_fp8.rs`'s) at every shape,
//!   the K tails (Gemma's k = 2,112 and 704) and the expert form included, against f64 from the
//!   host quantizer's bytes; the residual epilogue `with_residual`'s bits on the GEMV's own row; the
//!   f32 store the bf16 store's value before its rounding.
//! - A group member is its own launch, bit for bit; an expert pair is the dense B8 on its expert's
//!   slice and its activation row; a repeat differs in 0 bits.
//! - The act + quantize is `quant_rows_fp8` of its own bf16 activation row, bit for bit.
//! - TK-1's R1 on B8's 128-byte-swizzled fp8 boxes: every byte past `k`, past the activation's row
//!   0 and past the weight's rows comes back 0.
//! - The per-shape table, recorded: today's GEMV (its own prologue quantize) against B8, B8 plus its
//!   quantize share (Q1 for a normed input, the act + quantize for a down's), and B8 plus 5 µs,
//!   read-cold, A, B, A, B, as `T1_ROW` lines.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.

pub mod common {
    pub mod fp8;
    pub mod numerics;
    pub mod watchdog;
}

use std::ffi::c_void;

use common::fp8::{e4m3_value, quantize_rows};
use common::numerics::{bf16_inputs, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{
    B8Experts, B8Store, Fp8Operand, KernelError, act_quant_fp8, expert_gate_up_gemv_fp8,
    expert_gemv_fp8, gate_up_gemv_fp8, gemv_fp8_abt, gemv_fp8_b8, gemv_fp8_b8_box_probe,
    gemv_fp8_residual, gemv_group_fp8_abt, quant_row_fp8_one,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::{GatedOp, gemv_scratch_bytes, gemv_splits};

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// A splitmix64 stream, fixed by `seed`.
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// splitmix64 e4m3 weight bytes, every finite value (`tests/gemv_fp8.rs`'s).
fn e4m3_weights(len: usize, seed: u64) -> Vec<u8> {
    let mut next = stream(seed);
    (0..len)
        .map(|_| {
            let z = next();
            let b = (z % 0x7F) as u8;
            if z & (1 << 40) != 0 { b | 0x80 } else { b }
        })
        .collect()
}

/// Per-channel scales in `[2⁻¹⁰, 2⁻⁸)`, bf16-exact (`tests/gemv_fp8.rs`'s).
fn weight_scales(n: usize, seed: u64) -> Vec<f32> {
    bf16_inputs(n, seed)
        .iter()
        .map(|&b| {
            let v = bf16::from_bits(b).to_f32().abs();
            bf16::from_f32((1.0 + v * 3.0) / 1024.0).to_f32()
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A device buffer of `bytes` bytes (at least 256), every one `byte`.
fn filled(device: &CudaDevice, bytes: usize, byte: u8) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![byte; bytes.max(256)]).expect("upload")
}

fn up_bytes(device: &CudaDevice, bytes: &[u8]) -> DeviceBuffer {
    let mut b = bytes.to_vec();
    b.resize(b.len().max(256).next_multiple_of(256), 0);
    DeviceBuffer::upload_bytes(device, &b).expect("upload")
}

/// A buffer's bytes.
fn down_bytes(b: &DeviceBuffer) -> Vec<u8> {
    (b.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect()
}

/// A buffer's first `len` f32s.
fn down_f32(b: &DeviceBuffer, len: usize) -> Vec<f32> {
    let bytes = down_bytes(b);
    (bytes.as_chunks::<4>().0.iter().take(len))
        .map(|q| f32::from_le_bytes(*q))
        .collect()
}

/// Byte offset `at` into `b`.
fn at(b: &DeviceBuffer, at: usize) -> *mut c_void {
    b.ptr().cast::<u8>().wrapping_add(at).cast()
}

/// Which step of the decode tape a shape is, as B8 stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A plain GEMV or group: bf16 rows.
    Plain,
    /// A down or output projection with W4a's residual epilogue; `Some` when its input is an
    /// activation (the 27B's down: the act + quantize at that op makes its input).
    Residual(Option<GatedOp>),
    /// A gate/up pair (members gate then up): f32 rows, then the act + quantize at `GatedOp`.
    GateUp(GatedOp),
    /// A down projection: its input is the act + quantize's output (`GatedOp` the activation).
    Down(GatedOp),
}

/// The dense form, or the expert form: `pairs` pairs over a bank of `experts` blocks of
/// `expert_rows` rows, pair `p` reading activation row `p / row_div`.
#[derive(Clone, Copy, Debug)]
enum Form {
    Dense,
    Expert {
        experts: u32,
        expert_rows: u32,
        row_div: u32,
    },
}

/// One row of §1's inventory.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    ns: &'static [u32],
    k: u32,
    kind: Kind,
    form: Form,
    /// Launches per decode token (W0's tapes).
    launches: u32,
}

/// The routed experts' 8 picks a token; the tests' banks hold 8 experts, all picked.
const PAIRS: u32 = 8;
const BANK: u32 = 8;
/// The expert count today's expert GEMVs are launched with: as the tape launches them, a model's
/// (they take 128 or 256), over the `BANK`-expert bank. Every pick is below `BANK`, so only the
/// bank's experts are read.
const TODAY_EXPERTS: u32 = 128;

const SILU: GatedOp = GatedOp::Silu;
const GELU: GatedOp = GatedOp::Gelu;

/// §1's inventory: every one-row fp8 GEMV of the three models' decode tapes that B8 takes by
/// shape (`n`, `k` multiples of 64), with its launches per token. The 27B's GDN a/b group (`n` 48)
/// is not in it.
const INVENTORY: &[Shape] = &[
    Shape {
        name: "27B_gate+up",
        ns: &[17408, 17408],
        k: 5120,
        kind: Kind::GateUp(SILU),
        form: Form::Dense,
        launches: 64,
    },
    Shape {
        name: "27B_down",
        ns: &[5120],
        k: 17408,
        kind: Kind::Residual(Some(SILU)),
        form: Form::Dense,
        launches: 64,
    },
    Shape {
        name: "27B_gdn_out+attn_o",
        ns: &[5120],
        k: 6144,
        kind: Kind::Residual(None),
        form: Form::Dense,
        launches: 64,
    },
    Shape {
        name: "27B_gdn_qkv+z",
        ns: &[10240, 6144],
        k: 5120,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 48,
    },
    Shape {
        name: "27B_attn_qkv",
        ns: &[12288, 1024, 1024],
        k: 5120,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 16,
    },
    Shape {
        name: "35B_expert_gate+up",
        ns: &[512, 512],
        k: 2048,
        kind: Kind::GateUp(SILU),
        form: Form::Expert {
            experts: BANK,
            expert_rows: 1024,
            row_div: PAIRS,
        },
        launches: 40,
    },
    Shape {
        name: "35B_expert_down",
        ns: &[2048],
        k: 512,
        kind: Kind::Down(SILU),
        form: Form::Expert {
            experts: BANK,
            expert_rows: 2048,
            row_div: 1,
        },
        launches: 40,
    },
    Shape {
        name: "35B_shared_gate+up",
        ns: &[512, 512],
        k: 2048,
        kind: Kind::GateUp(SILU),
        form: Form::Dense,
        launches: 40,
    },
    Shape {
        name: "35B_shared_down",
        ns: &[2048],
        k: 512,
        kind: Kind::Down(SILU),
        form: Form::Dense,
        launches: 40,
    },
    Shape {
        name: "35B_attn_o",
        ns: &[2048],
        k: 4096,
        kind: Kind::Residual(None),
        form: Form::Dense,
        launches: 10,
    },
    Shape {
        name: "35B_attn_qkv",
        ns: &[8192, 512, 512],
        k: 2048,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 10,
    },
    Shape {
        name: "Gemma_expert_gate+up",
        ns: &[704, 704],
        k: 2816,
        kind: Kind::GateUp(GELU),
        form: Form::Expert {
            experts: BANK,
            expert_rows: 1408,
            row_div: PAIRS,
        },
        launches: 30,
    },
    Shape {
        name: "Gemma_expert_down",
        ns: &[2816],
        k: 704,
        kind: Kind::Down(GELU),
        form: Form::Expert {
            experts: BANK,
            expert_rows: 2816,
            row_div: 1,
        },
        launches: 30,
    },
    Shape {
        name: "Gemma_gate+up",
        ns: &[2112, 2112],
        k: 2816,
        kind: Kind::GateUp(GELU),
        form: Form::Dense,
        launches: 30,
    },
    Shape {
        name: "Gemma_down",
        ns: &[2816],
        k: 2112,
        kind: Kind::Down(GELU),
        form: Form::Dense,
        launches: 30,
    },
    Shape {
        name: "Gemma_attn_o_sliding",
        ns: &[2816],
        k: 4096,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 25,
    },
    Shape {
        name: "Gemma_attn_o_global",
        ns: &[2816],
        k: 8192,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 5,
    },
    Shape {
        name: "Gemma_attn_qkv_sliding",
        ns: &[4096, 2048, 2048],
        k: 2816,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 25,
    },
    Shape {
        name: "Gemma_attn_qkv_global",
        ns: &[8192, 1024],
        k: 2816,
        kind: Kind::Plain,
        form: Form::Dense,
        launches: 5,
    },
];

impl Shape {
    fn pairs(&self) -> u32 {
        match self.form {
            Form::Dense => 1,
            Form::Expert { .. } => PAIRS,
        }
    }

    /// The activation rows: the expert down's own 8, else 1 (the gate/up's broadcast row).
    fn a_rows(&self) -> u32 {
        match self.form {
            Form::Expert { row_div: 1, .. } => PAIRS,
            _ => 1,
        }
    }

    fn row_div(&self) -> u32 {
        match self.form {
            Form::Dense => 1,
            Form::Expert { row_div, .. } => row_div,
        }
    }

    /// The weight bytes one launch reads (the expert pairs' slices: the whole bank).
    fn weight_bytes(&self) -> usize {
        let rows: u32 = self.ns.iter().sum();
        rows as usize * self.k as usize * self.pairs() as usize
    }
}

/// One shape's inputs on the device, and the host's copies its references read.
struct Case {
    s: Shape,
    /// The activation rows `[a_rows, k]` bf16 (today's GEMVs read them), their host quantization
    /// `[a_rows, k]` e4m3 and `[a_rows]` scales (B8 reads those).
    x: DeviceBuffer,
    q: DeviceBuffer,
    a_scale: DeviceBuffer,
    q_host: Vec<u8>,
    a_scale_host: Vec<f32>,
    /// Per member: its weight (dense `[n, k]`, or a view into the bank) and scales.
    weights: Vec<(DeviceBuffer, DeviceBuffer)>,
    w_host: Vec<Vec<u8>>,
    s_host: Vec<Vec<f32>>,
    /// The expert form's bank (the members view into it) and its picks.
    bank: Option<(DeviceBuffer, DeviceBuffer)>,
    topk: DeviceBuffer,
    topk_host: Vec<u32>,
    /// Per member: the residual `[pairs, n]` bf16 (`Kind::Residual`).
    residual: Vec<DeviceBuffer>,
    residual_host: Vec<Vec<u16>>,
}

impl Case {
    fn new(device: &CudaDevice, s: Shape, seed: u64) -> Self {
        let (k, a_rows) = (s.k as usize, s.a_rows() as usize);
        let x_host: Vec<u16> = (0..a_rows)
            .flat_map(|r| bf16_inputs(k, seed ^ (r as u64) << 40))
            .collect();
        let (q_host, a_scale_host) = quantize_rows(&x_host, a_rows, k);
        let mut next = stream(seed ^ 0x70C);
        // A permutation of the bank: every pair a distinct expert.
        let mut topk_host: Vec<u32> = (0..BANK).collect();
        for i in (1..topk_host.len()).rev() {
            topk_host.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let (mut weights, mut w_host, mut s_host, mut bank) =
            (Vec::new(), Vec::new(), Vec::new(), None);
        match s.form {
            Form::Dense => {
                for (i, &n) in s.ns.iter().enumerate() {
                    let w = e4m3_weights(n as usize * k, seed ^ 0x100 ^ i as u64);
                    let sc = weight_scales(n as usize, seed ^ 0x200 ^ i as u64);
                    weights.push((up_bytes(device, &w), up_bytes(device, &f32_bytes(&sc))));
                    w_host.push(w);
                    s_host.push(sc);
                }
            }
            Form::Expert {
                experts,
                expert_rows,
                ..
            } => {
                let rows = experts as usize * expert_rows as usize;
                let w = e4m3_weights(rows * k, seed ^ 0x300);
                let sc = weight_scales(rows, seed ^ 0x400);
                bank = Some((up_bytes(device, &w), up_bytes(device, &f32_bytes(&sc))));
                w_host.push(w);
                s_host.push(sc);
            }
        }
        let residual_host: Vec<Vec<u16>> = match s.kind {
            Kind::Residual(_) => (s.ns.iter().enumerate())
                .map(|(i, &n)| {
                    bf16_inputs(s.pairs() as usize * n as usize, seed ^ 0x500 ^ i as u64)
                })
                .collect(),
            _ => Vec::new(),
        };
        Self {
            s,
            x: DeviceBuffer::upload(device, &x_host).expect("x"),
            q: up_bytes(device, &q_host),
            a_scale: up_bytes(device, &f32_bytes(&a_scale_host)),
            q_host,
            a_scale_host,
            weights,
            w_host,
            s_host,
            bank,
            topk: up_bytes(
                device,
                &(topk_host.iter())
                    .flat_map(|t| t.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            topk_host,
            residual: (residual_host.iter())
                .map(|r| DeviceBuffer::upload(device, r).expect("r"))
                .collect(),
            residual_host,
        }
    }

    /// Member `i`'s weight operand: its own, or (expert) its rows' view into the bank: member 1 of
    /// a gate/up pair is the block's second `n` rows.
    fn weight(&self, i: usize) -> Fp8Operand {
        match (&self.bank, self.s.form) {
            (Some((w, sc)), Form::Expert { .. }) => {
                let skip: usize = self.s.ns[..i].iter().map(|&n| n as usize).sum();
                (
                    at(w, skip * self.s.k as usize).cast_const(),
                    at(sc, skip * 4).cast_const(),
                )
            }
            _ => (
                self.weights[i].0.ptr().cast_const(),
                self.weights[i].1.ptr().cast_const(),
            ),
        }
    }

    fn experts(&self) -> Option<B8Experts> {
        match self.s.form {
            Form::Dense => None,
            Form::Expert {
                experts,
                expert_rows,
                ..
            } => Some(B8Experts {
                topk_idx: self.topk.ptr().cast_const(),
                experts,
                expert_rows,
            }),
        }
    }

    /// Output buffers for every member: `[pairs, n]` bf16 or f32, poisoned.
    fn outputs(&self, device: &CudaDevice, f32: bool) -> Vec<DeviceBuffer> {
        let each = if f32 { 4 } else { 2 };
        (self.s.ns.iter())
            .map(|&n| filled(device, self.s.pairs() as usize * n as usize * each, 0xFF))
            .collect()
    }

    /// B8 over the members `members` (indices) into `outs`, at `store`.
    fn b8(
        &self,
        members: &[usize],
        outs: &[&DeviceBuffer],
        store: B8Store<'_>,
    ) -> Result<(), KernelError> {
        let ops: Vec<(Fp8Operand, *mut c_void, u32)> = (members.iter().zip(outs))
            .map(|(&i, o)| (self.weight(i), o.ptr(), self.s.ns[i]))
            .collect();
        // SAFETY: every operand is its own allocation (or a view inside the bank) of its size;
        // the activation holds `a_rows · k` bytes and `a_rows` scales; null is the default stream.
        unsafe {
            gemv_fp8_b8(
                (
                    self.q.ptr().cast_const(),
                    self.a_scale.ptr().cast_const(),
                    self.s.a_rows(),
                ),
                &ops,
                self.s.k,
                (self.s.pairs(), self.s.row_div(), self.experts()),
                store,
                std::ptr::null_mut(),
            )
        }
    }

    /// The f64 reference of member `i` (`(ref, S)` per `[pairs, n]` element): the host quantizer's
    /// bytes against the member's (or its expert's) rows.
    fn reference(&self, i: usize) -> Vec<(f64, f64)> {
        let (k, n) = (self.s.k as usize, self.s.ns[i] as usize);
        let lut: Vec<f64> = (0..=255u8).map(e4m3_value).collect();
        let mut out = Vec::with_capacity(self.s.pairs() as usize * n);
        for p in 0..self.s.pairs() as usize {
            let xr = p / self.s.row_div() as usize;
            let qa = &self.q_host[xr * k..][..k];
            let (w, sc, row0) = match self.s.form {
                Form::Dense => (&self.w_host[i], &self.s_host[i], 0),
                Form::Expert { expert_rows, .. } => {
                    let skip: usize = self.s.ns[..i].iter().map(|&n| n as usize).sum();
                    let e = self.topk_host[p] as usize;
                    (
                        &self.w_host[0],
                        &self.s_host[0],
                        e * expert_rows as usize + skip,
                    )
                }
            };
            for j in 0..n {
                let (mut dot, mut abs) = (0.0, 0.0);
                for (&a, &b) in qa.iter().zip(&w[(row0 + j) * k..][..k]) {
                    let v = lut[a as usize] * lut[b as usize];
                    dot += v;
                    abs += v.abs();
                }
                let s = f64::from(self.a_scale_host[xr]) * f64::from(sc[row0 + j]);
                out.push((dot * s, abs * s));
            }
        }
        out
    }
}

/// `gpu` values against `refs` under S3's bound: the out-of-bound count and the worst ratio.
fn compare(gpu: &[f64], refs: &[(f64, f64)]) -> (usize, f64) {
    let (mut bad, mut worst) = (0usize, 0.0f64);
    for (&g, &(r, s)) in gpu.iter().zip(refs) {
        let bound = r.abs() / 128.0 + s / 2048.0;
        let ratio = if g.is_finite() {
            (g - r).abs() / bound.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        bad += usize::from(ratio.is_nan() || ratio > 1.0);
        worst = worst.max(ratio);
    }
    (bad, worst)
}

fn bf16_values(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|&b| f64::from(bf16::from_bits(b).to_f32()))
        .collect()
}

/// `with_residual` on the host: `bf16(bf16(v) + r)`, `v` already the stored bf16.
fn with_residual(v: u16, r: u16) -> u16 {
    let sum = bf16::from_bits(v).to_f32() + bf16::from_bits(r).to_f32();
    bf16::from_f32(sum).to_bits()
}

/// T1's bound: B8 at every inventory shape inside S3's f64 bound (each member's worst ratio
/// printed as `B8_BOUND`). A residual shape: its bf16 rows without the residual inside the bound,
/// and with it `with_residual` of those rows, bit for bit. A gate/up shape: its f32 rows inside the
/// bound, and its bf16 rows (one more launch) those rows rounded, bit for bit.
#[test]
fn b8_is_inside_s3_s_bound_at_every_inventory_shape() {
    let device = h100();
    let mut pass = true;
    for (case, &s) in INVENTORY.iter().enumerate() {
        let c = Case::new(&device, s, 0x7180 ^ (case as u64) << 32);
        let all: Vec<usize> = (0..s.ns.len()).collect();
        let bf = c.outputs(&device, false);
        c.b8(
            &all,
            &bf.iter().collect::<Vec<_>>(),
            B8Store::Bf16 { residual: None },
        )
        .expect("B8 bf16");
        for (i, o) in bf.iter().enumerate() {
            let len = (s.pairs() * s.ns[i]) as usize;
            let bits = &o.download().expect("download")[..len];
            let (bad, worst) = compare(&bf16_values(bits), &c.reference(i));
            eprintln!(
                "B8_BOUND {} member {i} (n={} k={}): out_of_bound={bad} worst_ratio={worst:.3} {}",
                s.name,
                s.ns[i],
                s.k,
                verdict(bad == 0)
            );
            pass &= bad == 0;
        }
        match s.kind {
            Kind::Residual(_) => {
                let with = c.outputs(&device, false);
                let rs: Vec<*const c_void> =
                    c.residual.iter().map(|r| r.ptr().cast_const()).collect();
                c.b8(
                    &all,
                    &with.iter().collect::<Vec<_>>(),
                    B8Store::Bf16 {
                        residual: Some(&rs),
                    },
                )
                .expect("B8 residual");
                for i in 0..s.ns.len() {
                    let len = (s.pairs() * s.ns[i]) as usize;
                    let plain = &bf[i].download().expect("plain")[..len];
                    let got = &with[i].download().expect("with")[..len];
                    let want: Vec<u16> = (plain.iter().zip(&c.residual_host[i]))
                        .map(|(&v, &r)| with_residual(v, r))
                        .collect();
                    let bad = mismatched_bits(got, &want);
                    eprintln!(
                        "B8_RESIDUAL {} member {i}: with_residual of its own rows mismatched_bits={bad} {}",
                        s.name,
                        verdict(bad == 0)
                    );
                    pass &= bad == 0;
                }
            }
            Kind::GateUp(_) => {
                let f = c.outputs(&device, true);
                c.b8(&all, &f.iter().collect::<Vec<_>>(), B8Store::F32)
                    .expect("B8 f32");
                for (i, o) in f.iter().enumerate() {
                    let len = (s.pairs() * s.ns[i]) as usize;
                    let v = down_f32(o, len);
                    let (bad, worst) = compare(
                        &v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
                        &c.reference(i),
                    );
                    let rounded: Vec<u16> =
                        v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
                    let same = mismatched_bits(&rounded, &bf[i].download().expect("bf")[..len]);
                    eprintln!(
                        "B8_F32 {} member {i}: out_of_bound={bad} worst_ratio={worst:.3}, rounded vs the \
                         bf16 store mismatched_bits={same} {}",
                        s.name,
                        verdict(bad == 0 && same == 0)
                    );
                    pass &= bad == 0 && same == 0;
                }
            }
            Kind::Plain | Kind::Down(_) => {}
        }
    }
    assert!(pass, "a B8 shape is out of S3's bound: see the lines above");
}

/// Every group's (and gate/up pair's) member against its own one-member launch, and a repeat of
/// the whole group, bit for bit.
#[test]
fn a_b8_member_is_its_own_launch_and_a_repeat_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (case, &s) in INVENTORY.iter().enumerate() {
        let c = Case::new(&device, s, 0x7190 ^ (case as u64) << 32);
        let all: Vec<usize> = (0..s.ns.len()).collect();
        let (one, two) = (c.outputs(&device, false), c.outputs(&device, false));
        let store = || B8Store::Bf16 { residual: None };
        c.b8(&all, &one.iter().collect::<Vec<_>>(), store())
            .expect("group");
        c.b8(&all, &two.iter().collect::<Vec<_>>(), store())
            .expect("repeat");
        for i in 0..s.ns.len() {
            let len = (s.pairs() * s.ns[i]) as usize;
            let alone = c.outputs(&device, false);
            c.b8(&[i], &[&alone[i]], store()).expect("alone");
            let g = &one[i].download().expect("group")[..len];
            let (r, a) = (
                mismatched_bits(g, &two[i].download().expect("repeat")[..len]),
                mismatched_bits(g, &alone[i].download().expect("alone")[..len]),
            );
            eprintln!(
                "B8_BITS {} member {i}: repeat mismatched_bits={r}, own launch mismatched_bits={a} {}",
                s.name,
                verdict(r == 0 && a == 0)
            );
            pass &= r == 0 && a == 0;
        }
    }
    assert!(pass, "a B8 member differs from its own launch or a repeat");
}

/// Every expert shape's pairs against the dense B8 on that pair's expert slice (the bank's block
/// rows of the member) and its activation row, bit for bit.
#[test]
fn a_b8_expert_pair_is_the_dense_b8_on_its_slice() {
    let device = h100();
    let mut pass = true;
    for (case, &s) in INVENTORY.iter().enumerate() {
        let Form::Expert { expert_rows, .. } = s.form else {
            continue;
        };
        let c = Case::new(&device, s, 0x71A0 ^ (case as u64) << 32);
        let all: Vec<usize> = (0..s.ns.len()).collect();
        let outs = c.outputs(&device, false);
        c.b8(
            &all,
            &outs.iter().collect::<Vec<_>>(),
            B8Store::Bf16 { residual: None },
        )
        .expect("expert");
        let (w, sc) = c.bank.as_ref().expect("a bank");
        let k = s.k as usize;
        let mut bad = 0usize;
        for (i, out) in outs.iter().enumerate() {
            let n = s.ns[i] as usize;
            let skip: usize = s.ns[..i].iter().map(|&n| n as usize).sum();
            let expert = &out.download().expect("expert")[..s.pairs() as usize * n];
            for p in 0..s.pairs() as usize {
                let e = c.topk_host[p] as usize;
                let row0 = e * expert_rows as usize + skip;
                let xr = p / s.row_div() as usize;
                let dense = filled(&device, n * 2, 0xFF);
                let op = (at(w, row0 * k).cast_const(), at(sc, row0 * 4).cast_const());
                // SAFETY: the slice's `n · k` bytes and `n` scales lie inside the bank, the row's
                // `k` bytes and its scale inside the activation; `dense` holds `n` bf16.
                unsafe {
                    gemv_fp8_b8(
                        (
                            at(&c.q, xr * k).cast_const(),
                            at(&c.a_scale, xr * 4).cast_const(),
                            1,
                        ),
                        &[(op, dense.ptr(), n as u32)],
                        s.k,
                        (1, 1, None),
                        B8Store::Bf16 { residual: None },
                        std::ptr::null_mut(),
                    )
                }
                .expect("the dense slice");
                bad += mismatched_bits(
                    &expert[p * n..][..n],
                    &dense.download().expect("dense")[..n],
                );
            }
        }
        eprintln!(
            "B8_EXPERT {}: every pair against the dense B8 on its slice mismatched_bits={bad} {}",
            s.name,
            verdict(bad == 0)
        );
        pass &= bad == 0;
    }
    assert!(
        pass,
        "an expert pair differs from the dense B8 on its slice"
    );
}

/// The act + quantize at every inventory activation (the gate/up shapes' `n`, their pairs' rows,
/// both ops), over f32 gate and up rows as B8 stores them: its bytes and scales are the host
/// `quant_rows_fp8` (`common::fp8::quantize_rows`, that kernel's bit-for-bit reference) of its own
/// bf16 activation rows, bit for bit; every activation finite.
#[test]
fn the_act_quantize_is_quant_rows_fp8_of_its_bf16_rows() {
    let device = h100();
    let mut pass = true;
    for (case, &s) in INVENTORY.iter().enumerate() {
        let Kind::GateUp(op) = s.kind else { continue };
        let (rows, n) = (s.pairs() as usize, s.ns[0] as usize);
        let mut next = stream(0x71B0 ^ case as u64);
        // Gate and up in [−8, 8): B8's scaled sums' range at the inventory's inputs.
        let mut f = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| ((next() >> 40) as f32 / 16_777_216.0 - 0.5) * 16.0)
                .collect()
        };
        let (g, u) = (f(rows * n), f(rows * n));
        let gb = up_bytes(&device, &f32_bytes(&g));
        let ub = up_bytes(&device, &f32_bytes(&u));
        let q = filled(&device, rows * n, 0xA5);
        let scale = filled(&device, rows * 4, 0xA5);
        let act = filled(&device, rows * n * 2, 0xFF);
        // SAFETY: each buffer its own allocation of its size; null is the default stream.
        unsafe {
            act_quant_fp8(
                (gb.ptr().cast_const(), ub.ptr().cast_const()),
                (q.ptr(), scale.ptr(), act.ptr()),
                (rows as u32, n as u32),
                op.code(),
                std::ptr::null_mut(),
            )
        }
        .expect("the act + quantize");
        let a = &act.download().expect("act")[..rows * n];
        let (want_q, want_s) = quantize_rows(a, rows, n);
        let got_q = &down_bytes(&q)[..rows * n];
        let got_s = down_f32(&scale, rows);
        let bytes_bad = (got_q.iter().zip(&want_q)).filter(|(a, b)| a != b).count();
        let scales_bad = (got_s.iter().zip(&want_s))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let nonfinite = a
            .iter()
            .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
            .count();
        let ok = bytes_bad == 0 && scales_bad == 0 && nonfinite == 0;
        eprintln!(
            "ACT_QUANT {} ({op:?}, rows={rows} n={n}): bytes mismatched={bytes_bad} scales \
             mismatched={scales_bad} nonfinite_act={nonfinite} {}",
            s.name,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the act + quantize differs from quant_rows_fp8 of its row"
    );
}

/// TK-1's R1 on B8's 128-byte-swizzled fp8 boxes: the activation's `[32, 256]` box (row `row` of a
/// 3-row `[rows, 1, k]` view) and the weight's `[64, 256]` box (tile 1 of a 96-row weight), each
/// first filled with NaN, at every K step of `k` 704 and 2,112 (the inventory's K tails) and 512:
/// each in-bounds byte its source's, every byte past `k`, past row 0, or past the weight's 96 rows
/// 0.
#[test]
fn b8_boxes_past_their_edges_read_zeros() {
    let device = h100();
    let (rows, n) = (3usize, 96usize);
    let poison_x = filled(&device, 32 * 256, 0x7F);
    let poison_w = filled(&device, 64 * 256, 0x7F);
    let mut pass = true;
    for k in [512usize, 704, 2112] {
        // Nonzero, non-NaN bytes everywhere, so a zero is the fill's.
        let nonzero = |len: usize, seed: u64| -> Vec<u8> {
            e4m3_weights(len, seed)
                .iter()
                .map(|&b| if b & 0x7F == 0 { 0x38 } else { b })
                .collect()
        };
        let (x, w) = (nonzero(rows * k, k as u64), nonzero(n * k, k as u64 ^ 1));
        let (xb, wb) = (up_bytes(&device, &x), up_bytes(&device, &w));
        for j in 0..k.div_ceil(256) {
            let (dx, dw) = (
                filled(&device, 32 * 256, 0x11),
                filled(&device, 64 * 256, 0x11),
            );
            // SAFETY: every buffer its own allocation of its size; null is the default stream.
            unsafe {
                gemv_fp8_b8_box_probe(
                    (xb.ptr().cast_const(), rows as u32, 1),
                    (wb.ptr().cast_const(), n as u32, 1),
                    (k as u32, j as u32),
                    (poison_x.ptr().cast_const(), poison_w.ptr().cast_const()),
                    (dx.ptr(), dw.ptr()),
                    std::ptr::null_mut(),
                )
            }
            .expect("the probe");
            let (gx, gw) = (down_bytes(&dx), down_bytes(&dw));
            let mut bad = 0usize;
            for r in 0..32 {
                for col in 0..256 {
                    let kc = j * 256 + col;
                    let want = if r == 0 && kc < k { x[k + kc] } else { 0 };
                    bad += usize::from(gx[r * 256 + col] != want);
                }
            }
            for r in 0..64 {
                for col in 0..256 {
                    let (row, kc) = (64 + r, j * 256 + col);
                    let want = if row < n && kc < k {
                        w[row * k + kc]
                    } else {
                        0
                    };
                    bad += usize::from(gw[r * 256 + col] != want);
                }
            }
            eprintln!(
                "B8_BOX k={k} step={j}: wrong bytes={bad} {}",
                verdict(bad == 0)
            );
            pass &= bad == 0;
        }
    }
    assert!(pass, "a B8 box past its edge did not read zeros");
}

/// B8's and the act + quantize's refusals, each returning its shape code and writing nothing.
#[test]
fn b8_and_the_act_quantize_refuse_each_bad_shape() {
    let device = h100();
    let c = Case::new(&device, INVENTORY[9], 7);
    let out = filled(&device, 2048 * 2, 0xFF);
    let null = std::ptr::null_mut();
    let mut pass = true;
    let b8 = |ns: &[u32], k: u32, pairs: (u32, u32), store: B8Store<'_>| {
        let ops: Vec<(Fp8Operand, *mut c_void, u32)> =
            ns.iter().map(|&n| (c.weight(0), out.ptr(), n)).collect();
        // SAFETY: refused before any pointer is touched.
        unsafe {
            gemv_fp8_b8(
                (c.q.ptr().cast_const(), c.a_scale.ptr().cast_const(), 1),
                &ops,
                k,
                (pairs.0, pairs.1, None),
                store,
                null,
            )
        }
    };
    let rs = [c.residual[0].ptr().cast_const()];
    let cases: Vec<(&str, Result<(), KernelError>)> = vec![
        ("no member", b8(&[], 4096, (1, 1), B8Store::F32)),
        ("4 members", b8(&[64; 4], 4096, (1, 1), B8Store::F32)),
        ("n 0", b8(&[0], 4096, (1, 1), B8Store::F32)),
        ("n 2000", b8(&[2000], 4096, (1, 1), B8Store::F32)),
        ("k 4000", b8(&[2048], 4000, (1, 1), B8Store::F32)),
        ("pairs 0", b8(&[2048], 4096, (0, 1), B8Store::F32)),
        ("row_div 0", b8(&[2048], 4096, (1, 0), B8Store::F32)),
        (
            "a pair past the rows",
            b8(&[2048], 4096, (2, 1), B8Store::F32),
        ),
        (
            "a residual for 1 of 2",
            b8(
                &[1024, 1024],
                4096,
                (1, 1),
                B8Store::Bf16 {
                    residual: Some(&rs),
                },
            ),
        ),
    ];
    for (what, r) in cases {
        let untouched = out.download().expect("out").iter().all(|&b| b == 0xFFFF);
        let ok = matches!(r, Err(KernelError::GroupShape { .. })) && untouched;
        eprintln!(
            "B8_REFUSAL {what}: {r:?} untouched={untouched} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    let g = filled(&device, 4096, 0);
    for (what, rows, n, op) in [
        ("rows 0", 0, 512, 0),
        ("n 500", 1, 500, 0),
        ("n 17,472", 1, 17472, 0),
        ("op 7", 1, 512, 7),
    ] {
        // SAFETY: refused before any pointer is touched.
        let r = unsafe {
            act_quant_fp8(
                (g.ptr().cast_const(), g.ptr().cast_const()),
                (out.ptr(), out.ptr(), null),
                (rows, n),
                op,
                null,
            )
        };
        let untouched = out.download().expect("out").iter().all(|&b| b == 0xFFFF);
        let ok = matches!(r, Err(KernelError::RowShape { .. })) && untouched;
        eprintln!(
            "ACT_QUANT_REFUSAL {what}: {r:?} untouched={untouched} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a bad shape was not refused");
}

// ── The table ──────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

type CudaEvent = *mut c_void;

/// The median µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA events on the
/// default stream, `before` run (untimed, on the same stream) before each (`tests/gemv_fp8_tc.rs`'s).
fn timed(mut before: impl FnMut(), mut launch: impl FnMut()) -> f64 {
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
    times[times.len() / 2]
}

/// The read flush: today's GEMV over a 100 MiB `[20480, 5120]` weight (twice the H100's L2), so L2
/// holds clean lines of other data before each timed launch, as in the decode tape.
struct Flush {
    x: DeviceBuffer,
    w: DeviceBuffer,
    s: DeviceBuffer,
    c: DeviceBuffer,
    p: DeviceBuffer,
}

const FLUSH_N: u32 = 20480;
const FLUSH_K: u32 = 5120;

impl Flush {
    fn new(device: &CudaDevice) -> Self {
        let splits = gemv_splits(FLUSH_N, FLUSH_K, 1);
        Self {
            x: DeviceBuffer::upload(device, &vec![0x3F80u16; FLUSH_K as usize]).expect("x"),
            w: filled(device, (FLUSH_N * FLUSH_K) as usize, 0x38),
            s: filled(device, FLUSH_N as usize * 4, 0),
            c: filled(device, FLUSH_N as usize * 2, 0),
            p: filled(
                device,
                gemv_scratch_bytes(FLUSH_N, 1, splits).expect("bytes") as usize,
                0,
            ),
        }
    }

    fn run(&self) {
        let splits = gemv_splits(FLUSH_N, FLUSH_K, 1);
        // SAFETY: as `Case`'s buffers; null is the default stream.
        unsafe {
            gemv_fp8_abt(
                (
                    self.x.ptr().cast_const(),
                    (self.w.ptr().cast_const(), self.s.ptr().cast_const()),
                    self.c.ptr(),
                ),
                self.p.ptr(),
                (FLUSH_N, FLUSH_K, splits),
                std::ptr::null_mut(),
            )
        }
        .expect("read flush");
    }
}

/// Today's launch of `c`'s step: the CUDA-core GEMV the tape plays for it, its bf16 activation
/// quantized in its own prologue, into `out` (scratch `p`).
fn today(c: &Case, out: &[DeviceBuffer], p: &DeviceBuffer) {
    let s = c.s;
    let null = std::ptr::null_mut();
    let x = c.x.ptr().cast_const();
    let n0 = s.ns[0];
    // SAFETY: every operand its own allocation of its size, `p` at least every split's partials;
    // null is the default stream.
    let r = unsafe {
        match (s.kind, s.form) {
            (Kind::GateUp(op), Form::Dense) => gate_up_gemv_fp8(
                (x, c.weight(0), c.weight(1)),
                (out[0].ptr(), p.ptr()),
                (n0, s.k, gemv_splits(n0, s.k, 1)),
                op.code(),
                null,
            ),
            (Kind::GateUp(op), Form::Expert { expert_rows, .. }) => expert_gate_up_gemv_fp8(
                (x, c.weight(0), c.weight(1), c.topk.ptr().cast_const()),
                (out[0].ptr(), p.ptr()),
                (1, PAIRS, n0, s.k),
                (TODAY_EXPERTS, expert_rows, gemv_splits(n0, s.k, PAIRS)),
                op.code(),
                null,
            ),
            (_, Form::Expert { expert_rows, .. }) => expert_gemv_fp8(
                (x, c.weight(0), c.topk.ptr().cast_const()),
                (out[0].ptr(), p.ptr()),
                (1, PAIRS, n0, s.k),
                (TODAY_EXPERTS, expert_rows),
                (true, gemv_splits(n0, s.k, PAIRS)),
                null,
            ),
            (Kind::Residual(_), Form::Dense) if s.ns.len() == 1 => gemv_fp8_residual(
                (x, c.weight(0), out[0].ptr()),
                p.ptr(),
                (n0, s.k, gemv_splits(n0, s.k, 1)),
                c.residual[0].ptr().cast_const(),
                null,
            ),
            (_, Form::Dense) if s.ns.len() == 1 => gemv_fp8_abt(
                (x, c.weight(0), out[0].ptr()),
                p.ptr(),
                (n0, s.k, gemv_splits(n0, s.k, 1)),
                null,
            ),
            (_, Form::Dense) => {
                let members: Vec<(Fp8Operand, *mut c_void, u32)> = (0..s.ns.len())
                    .map(|i| (c.weight(i), out[i].ptr(), s.ns[i]))
                    .collect();
                let splits: Vec<u32> = s.ns.iter().map(|&n| gemv_splits(n, s.k, 1)).collect();
                // A third of the scratch a member, each third 256-byte aligned.
                let third = (p.bytes() as usize / 3) / 256 * 256;
                let parts: Vec<*mut c_void> = (0..s.ns.len()).map(|i| at(p, i * third)).collect();
                gemv_group_fp8_abt(x, &members, &parts, (s.k, &splits), null)
            }
        }
    };
    r.expect("today's GEMV");
}

/// The quantize share of `c`'s B8 launch, enqueued: the act + quantize of its input for a down,
/// else Q1 of its normed row.
fn share(c: &Case, (q, sc, act): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer)) {
    let s = c.s;
    let null = std::ptr::null_mut();
    // SAFETY: each buffer its own allocation of its size; null is the default stream.
    let r = unsafe {
        match s.kind {
            Kind::Down(op) | Kind::Residual(Some(op)) => act_quant_fp8(
                (act.ptr().cast_const(), act.ptr().cast_const()),
                (q.ptr(), sc.ptr(), null),
                (s.a_rows(), s.k),
                op.code(),
                null,
            ),
            _ => quant_row_fp8_one(c.x.ptr().cast_const(), (q.ptr(), sc.ptr()), s.k, null),
        }
    };
    r.expect("the quantize share");
}

/// T1's pod gate table (/tmp/a-tc-plan.md §5.2, recorded): every inventory row's today, B8, B8
/// plus its measured quantize share (Q1, or the act + quantize for a down's input) and B8 plus 5
/// µs, read-cold, A, B, A, B; the gain per token at both, as `T1_ROW` lines. A gate/up row's B8 is
/// its two f32 members (the act + quantize after it is the down's share).
#[test]
fn the_b8_inventory_table_is_recorded() {
    let device = h100();
    let flush = Flush::new(&device);
    for (case, &s) in INVENTORY.iter().enumerate() {
        let c = Case::new(&device, s, 0x71C0 ^ (case as u64) << 32);
        let all: Vec<usize> = (0..s.ns.len()).collect();
        let (bf, f) = (c.outputs(&device, false), c.outputs(&device, true));
        let max_n = *s.ns.iter().max().expect("a member") as usize;
        let p = filled(
            &device,
            3 * 2 * gemv_scratch_bytes(max_n as u32, PAIRS, 16).expect("bytes") as usize,
            0,
        );
        let rows = s.a_rows() as usize * s.k as usize;
        let (q, sc, act) = (
            filled(&device, rows, 0),
            filled(&device, 64, 0),
            filled(&device, rows * 4, 0),
        );
        let store = || match s.kind {
            Kind::GateUp(_) => B8Store::F32,
            _ => B8Store::Bf16 { residual: None },
        };
        let outs = |gu: bool| {
            if gu {
                f.iter().collect::<Vec<_>>()
            } else {
                bf.iter().collect::<Vec<_>>()
            }
        };
        let gu = matches!(s.kind, Kind::GateUp(_));
        for round in 1..=2 {
            let t_today = timed(|| flush.run(), || today(&c, &bf, &p));
            let t_b8 = timed(
                || flush.run(),
                || c.b8(&all, &outs(gu), store()).expect("B8"),
            );
            let t_share = timed(|| flush.run(), || share(&c, (&q, &sc, &act)));
            let launches = f64::from(s.launches);
            let measured = t_b8 + t_share;
            let at5 = t_b8 + 5.0;
            eprintln!(
                "T1_ROW {} launches={} weight_MB={:.2} round={round}: readcold today {t_today:.1} us | \
                 b8 {t_b8:.1} us | b8+share {measured:.1} us (share {t_share:.1} us, {}) | b8+5us \
                 {at5:.1} us | gain_per_token {:.3} ms at the measured share, {:.3} ms at 5 us",
                s.name,
                s.launches,
                s.weight_bytes() as f64 / 1e6,
                if matches!(s.kind, Kind::Down(_) | Kind::Residual(Some(_))) {
                    "act+quantize"
                } else {
                    "Q1"
                },
                (t_today - measured) * launches / 1e3,
                (t_today - at5) * launches / 1e3
            );
        }
    }
}
