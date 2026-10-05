// SPDX-License-Identifier: Apache-2.0
//! The dense gated matvec (`affine_qmv_gated[_fast]`, `KernelId::AffineQmvGated`) against the
//! commands it replaces in a one-row decode step: the gate and up `affine_qmv[_fast]` matvecs,
//! then `silu_mul` / `gelu_mul`. The output must be the same bits.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{HiddenSize, KDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatedQmvConstants, AffineQmvConstants, SiluMulConstants,
};
use scratchy_target_metal::tape::step::GatedAct;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

#[derive(Clone, Copy)]
enum Dtype {
    F16,
    Bf16,
}

impl Dtype {
    fn bits(self, x: f32) -> u16 {
        match self {
            Dtype::F16 => f16::from_f32(x).to_bits(),
            Dtype::Bf16 => bf16::from_f32(x).to_bits(),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }
}

/// One gated MLP's shape: `k` in, `n` out (intermediate), its quantization and activation.
#[derive(Clone, Copy)]
struct Case {
    act: Dtype,
    scale: Dtype,
    group_size: usize,
    bits: usize,
    k: usize,
    n: usize,
    gated: GatedAct,
}

impl Case {
    /// `affine_qmv_fast`'s shape rule, as the lowering picks it.
    fn fast(&self) -> bool {
        self.n.is_multiple_of(8) && self.k.is_multiple_of(512)
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 24) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn shared<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    // SAFETY: `buf` holds at least `bytes` bytes and does not overlap `data`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        );
    }
    buf
}

fn read_u16(buf: &Buffer, n: usize) -> Vec<u16> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` u16s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, n).to_vec() }
}

fn tg(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// An affine projection's packed codes, scales and biases.
struct Weights {
    w: Buffer,
    scales: Buffer,
    biases: Buffer,
}

impl Weights {
    fn new(device: &Device, c: &Case, rng: &mut Lcg) -> Self {
        let words: Vec<u32> = (0..c.n * c.k * c.bits / 32)
            .map(|_| rng.next() as u32)
            .collect();
        // Centred weights of ~0.01: codes span `2^bits` steps of the scale, the bias takes off half.
        let groups = c.n * c.k / c.group_size;
        let steps = ((1u32 << c.bits) - 1) as f32;
        let (scales, biases): (Vec<u16>, Vec<u16>) = (0..groups)
            .map(|_| {
                let scale = (0.5 + rng.unit().abs()) * 0.02 / steps;
                let bias = -scale * steps / 2.0 * (1.0 + 0.1 * rng.unit());
                (c.scale.bits(scale), c.scale.bits(bias))
            })
            .unzip();
        Self {
            w: shared(device, &words),
            scales: shared(device, &scales),
            biases: shared(device, &biases),
        }
    }
}

/// `act(gate · x) * (up · x)` of case `c`: the fused kernel's bits, and the unfused chain's.
fn run(c: Case) -> Option<(Vec<u16>, Vec<u16>)> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let qmv = || AffineQmvConstants {
        k: KDimI32(c.k as i32),
        n: NDimI32(c.n as i32),
        codes: AffineCodes::AsWritten,
    };
    let (a, s, gs, b) = (c.act.tag(), c.scale.tag(), c.group_size, c.bits);
    let fast = if c.fast() { "_fast" } else { "" };
    let pso = |lib: &'static str, name: String, constants: Vec<ConstantValue>| {
        let name: &'static str = Box::leak(name.into_boxed_str());
        baked_build(&cache, &PipelineKey::new(lib, name, constants)).expect(name)
    };
    let matvec = pso(
        "quantized_qmv",
        format!("affine_qmv{fast}_{a}_s_{s}_gs_{gs}_b_{b}_batch_0"),
        qmv().into(),
    );
    let gated = AffineGatedQmvConstants {
        qmv: qmv(),
        act: c.gated,
    };
    let fused = pso(
        "quantized_qmv",
        format!("affine_qmv_gated{fast}_{a}_s_{s}_gs_{gs}_b_{b}"),
        gated.into(),
    );
    let act_name = match c.gated {
        GatedAct::Silu => "silu_mul",
        GatedAct::Gelu => "gelu_mul",
    };
    let n = HiddenSize(c.n as u32);
    let act = pso(
        "silu_mul",
        format!("{act_name}_{a}"),
        SiluMulConstants { n }.into(),
    );

    let mut rng = Lcg(c.k as u64 * 31 + c.n as u64);
    let (gate, up) = (
        Weights::new(&device, &c, &mut rng),
        Weights::new(&device, &c, &mut rng),
    );
    let x: Vec<u16> = (0..c.k).map(|_| c.act.bits(rng.unit())).collect();
    let x = shared(&device, &x);
    let row = vec![0u16; c.n];
    let (gate_y, up_y, unfused, out) = (
        shared(&device, &row),
        shared(&device, &row),
        shared(&device, &row),
        shared(&device, &row),
    );

    let mut batch = Mtl4DispatchBatch::begin(&device)?;
    for (wt, y) in [(&gate, &gate_y), (&up, &up_y)] {
        let binds = [
            (&wt.w, 0),
            (&wt.scales, 1),
            (&wt.biases, 2),
            (&x, 3),
            (y, 4),
        ];
        batch.encode(&matvec, &binds, &[], &[], &[], tg(1, c.n / 8), tg(32, 2));
    }
    batch.barrier();
    let binds = [(&unfused, 0), (&gate_y, 1), (&up_y, 2)];
    batch.encode(
        &act,
        &binds,
        &[],
        &[],
        &[],
        tg(c.n.div_ceil(256), 1),
        tg(256, 1),
    );
    let binds = [
        (&gate.w, 0),
        (&gate.scales, 1),
        (&gate.biases, 2),
        (&x, 3),
        (&out, 4),
        (&up.w, 5),
        (&up.scales, 6),
        (&up.biases, 7),
    ];
    batch.encode(&fused, &binds, &[], &[], &[], tg(1, c.n / 8), tg(32, 4));
    batch.commit(true);
    Some((read_u16(&out, c.n), read_u16(&unfused, c.n)))
}

fn check(c: Case) {
    let Some((fused, unfused)) = run(c) else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let zeros = unfused.iter().filter(|&&v| v == 0).count();
    assert!(
        zeros < c.n / 4,
        "the chain computed nothing: {zeros} of {} zero",
        c.n
    );
    let diff = fused.iter().zip(&unfused).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "{diff} of {} rows differ", c.n);
}

/// Llama 3.2 3B's MLP: 4-bit g64, `qmv_fast`'s shape.
const LLAMA: Case = Case {
    act: Dtype::Bf16,
    scale: Dtype::F16,
    group_size: 64,
    bits: 4,
    k: 3072,
    n: 8192,
    gated: GatedAct::Silu,
};

#[test]
fn llama_3b_silu_fast() {
    check(LLAMA);
}

#[test]
fn bf16_scales_and_group_sizes() {
    for group_size in [32, 64, 128] {
        check(Case {
            scale: Dtype::Bf16,
            group_size,
            ..LLAMA
        });
    }
}

#[test]
fn f16_activations() {
    check(Case {
        act: Dtype::F16,
        ..LLAMA
    });
}

/// Gemma 4's dense MLP: 8-bit g64 over a hidden of 2816 — not a multiple of 512, so the generic
/// matvec — and GELU.
#[test]
fn gemma4_8bit_gelu_generic() {
    check(Case {
        scale: Dtype::Bf16,
        bits: 8,
        k: 2816,
        n: 2112,
        gated: GatedAct::Gelu,
        ..LLAMA
    });
}

#[test]
fn four_bit_generic_and_eight_bit_fast() {
    check(Case {
        scale: Dtype::Bf16,
        k: 2816,
        n: 1024,
        ..LLAMA
    });
    check(Case {
        act: Dtype::F16,
        bits: 8,
        k: 1024,
        n: 512,
        ..LLAMA
    });
}
