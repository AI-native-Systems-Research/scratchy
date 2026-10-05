// SPDX-License-Identifier: Apache-2.0
//! A row program (`MetalFusion::RowProgram`, `row_program.metal`) against the commands it
//! replaces: each of its steps run as its own kernel — `rmsnorm_*`, `residual_add_*`,
//! `scalar_weight_mul_*`, `scalar_mul_*` — over buffers between them.
//!
//! The case is a Gemma 4 layer's end: two branch norms summed, normed, added into the residual,
//! scaled by the layer scalar — and a scale after, three rows stored, over two rows. The program
//! holds each value in the activation type, but the compiler may skip a rounding the kernels'
//! stores take, so its rows need not be their bits: each stored row must be as close to the exact
//! chain as the kernels' — within the bound their roundings carry, which the kernels' rows meet —
//! and a chain with its norms' gains swapped, which the same bound must reject, shows the bound
//! sees a wrong operand.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ComputePipelineState, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{BucketM, ElementCount, LayerId, QSize, RmsNormEps};
use scratchy_target_metal::tape::kernel_constants::{
    NORM_THREADS, RmsNormConstants, ScalarMulConstants,
};
use scratchy_target_metal::tape::step::{
    Eps, GainOffset, HiddenSize, ROW_PROGRAM_INSTRS, RowInstr, RowProgram, Scale,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

#[derive(Clone, Copy, Debug)]
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

    fn value(self, b: u16) -> f64 {
        f64::from(match self {
            Dtype::F16 => f16::from_bits(b).to_f32(),
            Dtype::Bf16 => bf16::from_bits(b).to_f32(),
        })
    }

    /// The unit roundoff: the largest relative error of rounding to the type.
    fn roundoff(self) -> f64 {
        match self {
            Dtype::Bf16 => 2f64.powi(-8),
            Dtype::F16 => 2f64.powi(-11),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }
}

/// `rows` rows of `width`; activations in `act`, the norms' gains in `gain` offset by `offset`.
#[derive(Clone, Copy, Debug)]
struct Case {
    act: Dtype,
    gain: Dtype,
    width: usize,
    rows: usize,
    offset: f32,
}

const EPS: f32 = 1e-6;
const SCALE: f32 = 0.625;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn shared(device: &Device, data: &[u16]) -> Buffer {
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

fn read(buf: &Buffer, n: usize) -> Vec<u16> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` u16s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, n).to_vec() }
}

fn size((x, y, z): (usize, usize, usize)) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}

fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

/// The program: `out0 = scale · s0 · (res + norm(norm(x0, g0) + norm(x1, g1), g2))`, with
/// `out1` the residual sum before its scales and `out2` the branch sum before its norm.
fn program(c: &Case) -> RowProgram {
    let norm = |dst, a, gain| RowInstr::Norm {
        dst,
        a,
        gain,
        eps: Eps(EPS),
        offset: GainOffset(c.offset),
    };
    let steps = [
        RowInstr::Load { dst: 0, input: 0 },
        RowInstr::Load { dst: 1, input: 1 },
        norm(2, 0, 0),
        norm(3, 1, 1),
        RowInstr::Add { dst: 4, a: 2, b: 3 },
        norm(5, 4, 2),
        RowInstr::Load { dst: 6, input: 2 },
        RowInstr::Add { dst: 7, a: 6, b: 5 },
        RowInstr::ScaleWeight {
            dst: 8,
            a: 7,
            scalar: 0,
        },
        RowInstr::Scale {
            dst: 9,
            a: 8,
            scale: Scale(SCALE),
        },
        RowInstr::Store { output: 0, a: 9 },
        RowInstr::Store { output: 1, a: 7 },
        RowInstr::Store { output: 2, a: 4 },
    ];
    let mut instrs = [None; ROW_PROGRAM_INSTRS];
    for (i, s) in instrs.iter_mut().zip(steps) {
        *i = Some(s);
    }
    let slot = |s| Some(scratchy_target_metal::tape::step::ArenaSlotIdx(s));
    RowProgram {
        instrs,
        inputs: [slot(0), slot(1), slot(2), None],
        outputs: [slot(3), slot(4), slot(5)],
        gains: [Some(LayerId(0)), Some(LayerId(0)), Some(LayerId(0)), None, None, None, None, None],
        scalars: [Some(LayerId(0)), None],
        width: HiddenSize(c.width as u32),
    }
}

struct Rig {
    device: Device,
    _cache: SpecializedPipelineCache,
    program: ComputePipelineState,
    norm: ComputePipelineState,
    add: ComputePipelineState,
    scale_weight: ComputePipelineState,
    scale: ComputePipelineState,
}

fn rig(c: &Case) -> Option<Rig> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (act, gain) = (c.act.tag(), c.gain.tag());
    let build = |library, name: &'static str, constants| {
        baked_build(&cache, &PipelineKey::new(library, name, constants)).expect(name)
    };
    let program = build(
        "row_program",
        leak(format!("row_program_{act}_s_{gain}")),
        (&program(c)).into(),
    );
    let norm = RmsNormConstants {
        bucket_m: BucketM(c.rows as u32),
        q_size: QSize(c.width as u32),
        rms_norm_eps: RmsNormEps(EPS),
        weight_offset: c.offset,
    };
    let norm = build(
        "rmsnorm",
        leak(format!("rmsnorm_{act}_s_{gain}_specialized")),
        norm.into(),
    );
    let add = build(
        "elementwise",
        leak(format!("residual_add_{act}_specialized")),
        Vec::new(),
    );
    let scale_weight = build(
        "elementwise",
        leak(format!("scalar_weight_mul_{act}_specialized")),
        Vec::new(),
    );
    let scale = ScalarMulConstants {
        scale: SCALE,
        elements: ElementCount((c.rows * c.width) as u32),
    };
    let scale = build(
        "elementwise",
        leak(format!("scalar_mul_{act}_specialized")),
        scale.into(),
    );
    Some(Rig {
        device,
        _cache: cache,
        program,
        norm,
        add,
        scale_weight,
        scale,
    })
}

/// The program's rows: the two branches and the residual, the three gains and the scalar weight.
struct Inputs {
    x: [Vec<u16>; 3],
    gains: [Vec<u16>; 3],
    scalar: Vec<u16>,
}

fn inputs(c: &Case, rng: &mut Lcg) -> Inputs {
    let n = c.rows * c.width;
    let mut row = |amp: f32| (0..n).map(|_| c.act.bits(amp * rng.unit())).collect::<Vec<_>>();
    let x = [row(3.0), row(40.0), row(8.0)];
    let mut gain = || {
        (0..c.width)
            .map(|_| c.gain.bits(1.0 - c.offset + 0.5 * rng.unit()))
            .collect::<Vec<_>>()
    };
    let gains = [gain(), gain(), gain()];
    let scalar = vec![c.act.bits(0.37)];
    Inputs { x, gains, scalar }
}

/// The three stored rows: by the program, or by the kernels it replaces.
fn run(r: &Rig, c: &Case, i: &Inputs, fused: bool) -> [Vec<u16>; 3] {
    let n = c.rows * c.width;
    let buf = |d: &[u16]| shared(&r.device, d);
    let x = [buf(&i.x[0]), buf(&i.x[1]), buf(&i.x[2])];
    let g = [buf(&i.gains[0]), buf(&i.gains[1]), buf(&i.gains[2])];
    let s = buf(&i.scalar);
    let zeros = vec![0u16; n];
    let mut batch = Mtl4DispatchBatch::begin(&r.device).expect("mtl4");
    let rows = size((c.rows, 1, 1));
    let threads = size((NORM_THREADS as usize, 1, 1));
    let out = if fused {
        let out = [buf(&zeros), buf(&zeros), buf(&zeros)];
        let binds = [
            (&x[0], 0),
            (&x[1], 1),
            (&x[2], 2),
            (&out[0], 4),
            (&out[1], 5),
            (&out[2], 6),
            (&g[0], 7),
            (&g[1], 8),
            (&g[2], 9),
            (&s, 15),
        ];
        batch.encode(&r.program, &binds, &[], &[], &[], rows, threads);
        out
    } else {
        // The residual adds into its own row, as the unfused tape's does.
        let (y, res, sum) = (buf(&zeros), buf(&i.x[2]), buf(&zeros));
        let (normed, weighted) = (buf(&zeros), buf(&zeros));
        let mut step = |pso, binds: &[(&Buffer, usize)], grid, t| {
            batch.encode(pso, binds, &[], &[], &[], grid, t);
            batch.barrier();
        };
        let (each, per) = (size((n / 256, 1, 1)), size((256, 1, 1)));
        step(&r.norm, &[(&sum, 0), (&x[0], 1), (&g[0], 2)], rows, threads);
        step(&r.norm, &[(&normed, 0), (&x[1], 1), (&g[1], 2)], rows, threads);
        step(&r.add, &[(&sum, 0), (&normed, 1)], each, per);
        step(&r.norm, &[(&normed, 0), (&sum, 1), (&g[2], 2)], rows, threads);
        step(&r.add, &[(&res, 0), (&normed, 1)], each, per);
        step(&r.scale_weight, &[(&weighted, 0), (&res, 1), (&s, 2)], each, per);
        step(&r.scale, &[(&y, 0), (&weighted, 1)], each, per);
        [y, res, sum]
    };
    batch.commit(true);
    [read(&out[0], n), read(&out[1], n), read(&out[2], n)]
}

fn cases() -> Vec<Case> {
    let gemma4 = Case {
        act: Dtype::Bf16,
        gain: Dtype::Bf16,
        width: 2816,
        rows: 2,
        offset: 0.0,
    };
    vec![
        gemma4,
        Case {
            offset: 1.0,
            width: 1152,
            ..gemma4
        },
        Case {
            act: Dtype::F16,
            gain: Dtype::F16,
            ..gemma4
        },
        Case {
            gain: Dtype::F16,
            offset: 1.0,
            ..gemma4
        },
    ]
}

/// Each stored row's exact value, with the bound on its error the kernels' roundings carry.
type Exact = [Vec<(f64, f64)>; 3];

/// The chain over `i` in f64, its norms' gains `gains` (an order of `i.gains`), and each row's
/// error bound: rounding each value the kernels store (`u` relative each), carried through the
/// steps after — a norm moves its row by its input's error over the row's rms, and its rms by
/// that error's rms.
fn exact(c: &Case, i: &Inputs, gains: [usize; 3]) -> Exact {
    let u = c.act.roundoff();
    let (w, o) = (c.width, f64::from(c.offset));
    let val = |v: &[u16], d: Dtype| v.iter().map(|&b| d.value(b)).collect::<Vec<_>>();
    let x: Vec<Vec<f64>> = i.x.iter().map(|v| val(v, c.act)).collect();
    let g: Vec<Vec<f64>> = gains.iter().map(|&k| val(&i.gains[k], c.gain)).collect();
    let scalar = c.act.value(i.scalar[0]);
    let scale = f64::from(SCALE);
    let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / w as f64 + f64::from(EPS)).sqrt();
    let norm = |v: &[f64], g: &[f64]| {
        let r = rms(v);
        v.iter().zip(g).map(|(e, g)| e / r * (g + o)).collect::<Vec<_>>()
    };
    let mut rows: Exact = Default::default();
    for row in 0..c.rows {
        let at = |v: &[f64]| v[row * w..(row + 1) * w].to_vec();
        let (n0, n1) = (norm(&at(&x[0]), &g[0]), norm(&at(&x[1]), &g[1]));
        let x2 = at(&x[2]);
        let sum: Vec<f64> = n0.iter().zip(&n1).map(|(a, b)| a + b).collect();
        // The sum's error: its terms' roundings and its own.
        let e_sum: Vec<f64> = (n0.iter().zip(&n1))
            .map(|(a, b)| u * (a.abs() + b.abs()) + u * (a + b).abs())
            .collect();
        let normed = norm(&sum, &g[2]);
        let (r, er) = (rms(&sum), rms(&e_sum));
        for j in 0..w {
            let gain = (g[2][j] + o).abs();
            let e_normed = gain * e_sum[j] / r + normed[j].abs() * er / r + u * normed[j].abs();
            let res = x2[j] + normed[j];
            let e_res = e_normed + u * (res.abs() + e_normed);
            let weighted = res * scalar;
            let e_weighted = scalar.abs() * e_res + u * weighted.abs();
            let y = weighted * scale;
            let e_y = scale * e_weighted + u * y.abs();
            rows[0].push((y, e_y));
            rows[1].push((res, e_res));
            rows[2].push((sum[j], e_sum[j]));
        }
    }
    rows
}

/// Whether every stored row is within its bound of the exact chain's; the first that is not.
fn within(c: &Case, got: &[Vec<u16>; 3], exact: &Exact) -> Result<(), String> {
    for (k, (got, exact)) in got.iter().zip(exact).enumerate() {
        for (j, (&b, &(e, bound))) in got.iter().zip(exact).enumerate() {
            // Float arithmetic's own error, far below the stored types'.
            let bound = bound * 1.01 + 1e-30;
            let v = c.act.value(b);
            if (v - e).abs() > bound {
                return Err(format!("output {k}[{j}]: {v} vs exact {e} (bound {bound})"));
            }
        }
    }
    Ok(())
}

#[test]
fn a_row_program_is_as_close_to_its_chain_as_its_steps_kernels() {
    for c in cases() {
        let Some(r) = rig(&c) else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let i = inputs(&c, &mut Lcg(0x5eed ^ c.width as u64));
        let (fused, steps) = (run(&r, &c, &i, true), run(&r, &c, &i, false));
        let chain = exact(&c, &i, [0, 1, 2]);
        within(&c, &steps, &chain).unwrap_or_else(|e| panic!("{c:?}: the kernels: {e}"));
        within(&c, &fused, &chain).unwrap_or_else(|e| panic!("{c:?}: the program: {e}"));
        let swapped = exact(&c, &i, [1, 0, 2]);
        assert!(
            within(&c, &fused, &swapped).is_err(),
            "{c:?}: the bound accepts the chain with its gains swapped"
        );
    }
}
