// SPDX-License-Identifier: Apache-2.0
//! Nemotron-H's Mamba-2 mixer on Metal — the GDN kernels in their Mamba-2 forms (`gdn_conv1d_varlen`
//! over the in-projection's `xBC` columns with its bias, `gdn_gating` + `gdn_scan_varlen` or
//! `gdn_scan_simd` running the selective scan with the `D` skip, `gdn_rms_norm_gated` gating
//! before a per-group norm) — against a CPU reference of mlx-lm's `NemotronHMamba2Mixer` (minus
//! its projections) carrying each sequence's conv ring and ssm state across steps.
//!
//! Each step is a varlen batch over the one-slot-per-sequence state pool, dispatched with the
//! constants and bindings `lower_one(MetalStep::Mamba2)` bakes (pinned on the lowering side by
//! `mamba2_lowers_the_gdn_kernels_in_their_mamba2_forms`): a multi-sequence prefill, its
//! continuation across a chunk boundary (chunks shorter than the conv window included), decode
//! steps of one token a sequence, and a slot re-taken fresh mid-run. Both scan mappings run — the
//! per-thread scan (state 16, the parity-tiny geometry) and the simd scan (state 128, the real
//! checkpoints'). GPU tests — run with `--test-threads=1` (standing rule).

use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

const EPS: f32 = 1e-5;
/// The lowering's 1-D / channel threadgroup cap (`THREADS_PER_GROUP`).
const THREADS: usize = 256;

/// The model dtype a kernel instantiation reads its activations and `T` weights in.
#[derive(Clone, Copy, Debug)]
enum Dtype {
    F32,
    Bf16,
}

impl Dtype {
    fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }

    /// `x` as this dtype holds it.
    fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::Bf16 => bf16::from_f32(x).to_f32(),
        }
    }

    fn bytes(self, xs: &[f32]) -> Vec<u8> {
        match self {
            Self::F32 => xs.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Bf16 => xs
                .iter()
                .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
                .collect(),
        }
    }

    fn decode(self, buf: &Buffer, n: usize) -> Vec<f32> {
        match self {
            Self::F32 => read::<f32>(buf, n),
            Self::Bf16 => read::<u16>(buf, n)
                .into_iter()
                .map(|b| bf16::from_bits(b).to_f32())
                .collect(),
        }
    }

    /// The output's tolerance: its rounding to this dtype, plus f32 reassociation.
    fn tol(self, reference: f32) -> f32 {
        match self {
            Self::F32 => 2e-4 * (1.0 + reference.abs()),
            Self::Bf16 => 1e-2 * reference.abs() + 2e-3,
        }
    }

    fn symbol(self, kernel: &str) -> &'static str {
        Box::leak(format!("{kernel}_{}", self.tag()).into_boxed_str())
    }
}

/// A Mamba-2 mixer's geometry, in the GDN state pool's terms: groups as key heads, heads as
/// value heads, `[head_dim, state]` a head.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    heads: usize,
    head_dim: usize,
    groups: usize,
    state: usize,
    kernel: usize,
}

impl Geometry {
    fn inner(self) -> usize {
        self.heads * self.head_dim
    }
    fn conv_dim(self) -> usize {
        self.inner() + 2 * self.groups * self.state
    }
    /// The in-projection row: z | xBC | dt.
    fn row(self) -> usize {
        self.inner() + self.conv_dim() + self.heads
    }
    fn simd(self) -> bool {
        self.state.is_multiple_of(32) && self.head_dim.is_multiple_of(4)
    }
}

/// Deterministic pseudo-random fill in `[-scale, scale]`.
fn fill(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
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

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` `T`s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n).to_vec() }
}

fn size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

/// A mixer layer's weights as the bundle holds them: the conv taps/bias and `dt_bias` in the
/// model dtype, `A_log`, `D` and the norm weight f32 (`gated_delta_net_load`).
struct Weights {
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
}

impl Weights {
    fn new(geo: Geometry, dtype: Dtype) -> Self {
        let round = |v: Vec<f32>| v.into_iter().map(|x| dtype.round(x)).collect();
        Self {
            conv_w: round(fill(geo.conv_dim() * geo.kernel, 1, 0.5)),
            conv_b: round(fill(geo.conv_dim(), 2, 0.5)),
            a_log: (1..=geo.heads).map(|h| (h as f32).ln()).collect(),
            dt_bias: round(fill(geo.heads, 3, 0.5)),
            d: fill(geo.heads, 4, 1.0),
            norm: fill(geo.inner(), 5, 0.1).iter().map(|x| 1.0 + x).collect(),
        }
    }
}

/// One sequence's persistent state: the last `kernel - 1` raw `xBC` inputs a channel (oldest
/// first, `[conv_dim, kernel - 1]`) and the `[heads, head_dim, state]` ssm state — the pool's
/// per-slot layouts.
#[derive(Clone)]
struct SeqState {
    conv: Vec<f32>,
    ssm: Vec<f32>,
}

impl SeqState {
    fn fresh(geo: Geometry) -> Self {
        Self {
            conv: vec![0.0; (geo.kernel - 1) * geo.conv_dim()],
            ssm: vec![0.0; geo.heads * geo.head_dim * geo.state],
        }
    }
}

/// mlx-lm `NemotronHMamba2Mixer` minus its projections, over `tokens` rows of `zx` continuing
/// `st`: the causal conv (+bias, SiLU) over `[ring ++ xBC]`, the selective scan per token (mlx-lm
/// `ssm_update_kernel`), `MambaRMSNormGated`. Returns the `[tokens, inner]` output.
fn reference(geo: Geometry, w: &Weights, zx: &[f32], tokens: usize, st: &mut SeqState) -> Vec<f32> {
    let (inner, conv_dim, row) = (geo.inner(), geo.conv_dim(), geo.row());
    let (k, gs) = (geo.kernel, geo.groups * geo.state);
    let per_group = geo.heads / geo.groups;
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let mut out = vec![0.0; tokens * inner];
    for t in 0..tokens {
        let r = &zx[t * row..][..row];
        let xbc = &r[inner..inner + conv_dim];
        let mut conv = vec![0.0f32; conv_dim];
        for (c, v) in conv.iter_mut().enumerate() {
            let ring = &mut st.conv[c * (k - 1)..][..k - 1];
            let mut acc = w.conv_b[c] + xbc[c] * w.conv_w[c * k + k - 1];
            for (j, &past) in ring.iter().enumerate() {
                acc += past * w.conv_w[c * k + j];
            }
            *v = silu(acc);
            ring.rotate_left(1);
            ring[k - 2] = xbc[c];
        }

        let (x, b, cm) = (
            &conv[..inner],
            &conv[inner..inner + gs],
            &conv[inner + gs..],
        );
        let mut y = vec![0.0f32; inner];
        for h in 0..geo.heads {
            let g = h / per_group;
            let av = r[inner + conv_dim + h] + w.dt_bias[h];
            let dt = if av <= 20.0 { av.exp().ln_1p() } else { av };
            let decay = (-(w.a_log[h].exp()) * dt).exp();
            for p in 0..geo.head_dim {
                let xv = x[h * geo.head_dim + p];
                let s = &mut st.ssm[(h * geo.head_dim + p) * geo.state..][..geo.state];
                let mut acc = 0.0;
                for (n, sv) in s.iter_mut().enumerate() {
                    *sv = decay * *sv + xv * dt * b[g * geo.state + n];
                    acc += *sv * cm[g * geo.state + n];
                }
                y[h * geo.head_dim + p] = acc + w.d[h] * xv;
            }
        }
        let width = inner / geo.groups;
        for g in 0..geo.groups {
            let gated: Vec<f32> = (0..width)
                .map(|i| y[g * width + i] * silu(r[g * width + i]))
                .collect();
            let inv = 1.0 / (gated.iter().map(|v| v * v).sum::<f32>() / width as f32 + EPS).sqrt();
            for (i, v) in gated.iter().enumerate() {
                out[t * inner + g * width + i] = v * inv * w.norm[g * width + i];
            }
        }
    }
    out
}

/// One varlen step: `(slot, tokens, fresh)` a sequence, in batch order.
type Step = Vec<(usize, usize, bool)>;

/// The GPU side: the pool's conv/ssm state and the weights, resident across steps.
struct Gpu {
    device: Device,
    cache: SpecializedPipelineCache,
    geo: Geometry,
    dtype: Dtype,
    conv_state: Buffer,
    ssm_state: Buffer,
    conv_w: Buffer,
    conv_b: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    d: Buffer,
    norm: Buffer,
}

impl Gpu {
    fn key(
        &self,
        library: &'static str,
        symbol: &'static str,
        c: Vec<ConstantValue>,
    ) -> PipelineKey {
        PipelineKey::new(library, symbol, c)
    }

    /// Run one step's commands, as the lowering emits them a barrier apart; the `[tokens, inner]`
    /// output, or `None` with no MTL4 queue.
    fn step(&self, zx: &[f32], cu: &[i32], slots: &[i32], fresh: &[u32]) -> Option<Vec<f32>> {
        let (geo, dtype) = (self.geo, self.dtype);
        let (inner, conv_dim, row) = (geo.inner(), geo.conv_dim(), geo.row());
        let (heads, groups) = (geo.heads as u32, geo.groups as u32);
        let tokens = *cu.last().unwrap() as usize;
        let num_seqs = slots.len();
        let dev = &self.device;
        let zx_buf = shared(dev, &dtype.bytes(zx));
        let (cu_b, si_b, fr_b) = (shared(dev, cu), shared(dev, slots), shared(dev, fresh));
        let conv_out = shared(dev, &vec![0f32; tokens * conv_dim]);
        let (g_buf, beta_buf) = (
            shared(dev, &vec![0f32; tokens * geo.heads]),
            shared(dev, &vec![0f32; tokens * geo.heads]),
        );
        let y = shared(dev, &vec![0f32; tokens * inner]);
        let out = shared(dev, &vec![0u8; tokens * inner * 4]);
        let dt_slice = |slot: u16| {
            [
                ConstantValue::uint(slot, row as u32),
                ConstantValue::uint(slot + 1, (inner + conv_dim) as u32),
            ]
        };
        let state_bindings = |first: usize| {
            [
                (&self.ssm_state, first),
                (&cu_b, first + 1),
                (&si_b, first + 2),
                (&fr_b, first + 3),
            ]
        };
        let mut batch = Mtl4DispatchBatch::begin(dev)?;

        let conv = self.key(
            "gdn_conv1d_varlen",
            dtype.symbol("gdn_conv1d_varlen"),
            vec![
                ConstantValue::uint(0, conv_dim as u32),
                ConstantValue::uint(1, geo.kernel as u32),
                ConstantValue::uint(2, row as u32),
                ConstantValue::uint(3, inner as u32),
                ConstantValue::boolean(4, true),
                ConstantValue::uint(5, 0),
            ],
        );
        let tg_y = conv_dim.min(THREADS);
        batch.encode(
            &baked_build(&self.cache, &conv).expect("conv pipeline"),
            &[
                (&conv_out, 0),
                (&zx_buf, 1),
                (&self.conv_w, 2),
                (&self.conv_state, 3),
                (&cu_b, 4),
                (&si_b, 5),
                (&fr_b, 6),
                (&self.conv_b, 7),
            ],
            &[],
            &[],
            &[],
            size(num_seqs, conv_dim.div_ceil(tg_y), 1),
            size(1, tg_y, 1),
        );
        batch.barrier();
        let scan_constants = || {
            vec![
                ConstantValue::uint(0, groups),
                ConstantValue::uint(1, heads),
                ConstantValue::uint(2, geo.state as u32),
                ConstantValue::uint(3, geo.head_dim as u32),
                ConstantValue::float(4, (geo.state as f32).powf(-0.5)),
                ConstantValue::boolean(5, true),
                ConstantValue::uint(8, 0),
            ]
        };
        if geo.simd() {
            let mut c = scan_constants();
            c.extend(dt_slice(6));
            let scan = self.key("gdn_scan_varlen", dtype.symbol("gdn_scan_simd"), c);
            let mut b = vec![(&y, 0), (&conv_out, 1), (&zx_buf, 2), (&zx_buf, 3)];
            b.extend(state_bindings(4));
            b.extend([(&self.a_log, 8), (&self.dt_bias, 9), (&self.d, 10)]);
            batch.encode(
                &baked_build(&self.cache, &scan).expect("simd scan pipeline"),
                &b,
                &[],
                &[],
                &[],
                size(1, geo.heads * geo.head_dim / 4, num_seqs),
                size(32, 4, 1),
            );
        } else {
            let mut c = vec![
                ConstantValue::uint(0, (tokens * geo.heads) as u32),
                ConstantValue::uint(1, heads),
                ConstantValue::boolean(2, true),
            ];
            c.extend(dt_slice(3));
            let gating = self.key("gdn_gating", dtype.symbol("gdn_gating"), c);
            batch.encode(
                &baked_build(&self.cache, &gating).expect("gating pipeline"),
                &[
                    (&g_buf, 0),
                    (&beta_buf, 1),
                    (&zx_buf, 2),
                    (&zx_buf, 3),
                    (&self.a_log, 4),
                    (&self.dt_bias, 5),
                ],
                &[],
                &[],
                &[],
                size((tokens * geo.heads).div_ceil(THREADS), 1, 1),
                size(THREADS, 1, 1),
            );
            batch.barrier();
            let scan = self.key("gdn_scan_varlen", "gdn_scan_varlen_f32", scan_constants());
            let mut b = vec![(&y, 0), (&conv_out, 1), (&g_buf, 2), (&beta_buf, 3)];
            b.extend(state_bindings(4));
            b.push((&self.d, 8));
            let tgx = geo.head_dim.min(THREADS);
            batch.encode(
                &baked_build(&self.cache, &scan).expect("scan pipeline"),
                &b,
                &[],
                &[],
                &[],
                size(geo.head_dim.div_ceil(tgx), geo.heads, num_seqs),
                size(tgx, 1, 1),
            );
        }
        batch.barrier();
        let width = inner / geo.groups;
        let norm = self.key(
            "gdn_rms_norm_gated",
            dtype.symbol("gdn_rms_norm_gated"),
            vec![
                ConstantValue::uint(0, width as u32),
                ConstantValue::uint(1, (tokens * geo.groups) as u32),
                ConstantValue::float(2, EPS),
                ConstantValue::uint(3, row as u32),
                ConstantValue::uint(4, groups),
                ConstantValue::boolean(5, true),
            ],
        );
        batch.encode(
            &baked_build(&self.cache, &norm).expect("norm pipeline"),
            &[(&out, 0), (&y, 1), (&zx_buf, 2), (&self.norm, 3)],
            &[],
            &[],
            &[],
            size(tokens * geo.groups, 1, 1),
            size(width.next_power_of_two().min(THREADS), 1, 1),
        );
        batch.commit(true);
        Some(dtype.decode(&out, tokens * inner))
    }
}

/// Run `steps` over a pool of `slots` slots on the GPU and on the reference, comparing every
/// step's output and, after the last, every slot's conv ring and ssm state.
fn run(geo: Geometry, dtype: Dtype, slots: usize, steps: &[Step]) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let w = Weights::new(geo, dtype);
    let ring = (geo.kernel - 1) * geo.conv_dim();
    let ssm = geo.heads * geo.head_dim * geo.state;
    // Garbage in every slot: a fresh sequence must not read it.
    let gpu = Gpu {
        conv_state: shared(&device, &fill(slots * ring, 11, 9.0)),
        ssm_state: shared(&device, &fill(slots * ssm, 12, 9.0)),
        conv_w: shared(&device, &dtype.bytes(&w.conv_w)),
        conv_b: shared(&device, &dtype.bytes(&w.conv_b)),
        a_log: shared(&device, &w.a_log),
        dt_bias: shared(&device, &dtype.bytes(&w.dt_bias)),
        d: shared(&device, &w.d),
        norm: shared(&device, &w.norm),
        device,
        cache,
        geo,
        dtype,
    };
    let mut states = vec![SeqState::fresh(geo); slots];
    for (i, step) in steps.iter().enumerate() {
        let tokens: usize = step.iter().map(|s| s.1).sum();
        let zx: Vec<f32> = fill(tokens * geo.row(), 100 + i as u32, 1.5)
            .into_iter()
            .map(|x| dtype.round(x))
            .collect();
        let mut cu = vec![0i32];
        let mut want = Vec::with_capacity(tokens * geo.inner());
        for &(slot, n, fresh) in step {
            let at = *cu.last().unwrap() as usize;
            cu.push((at + n) as i32);
            if fresh {
                states[slot] = SeqState::fresh(geo);
            }
            let rows = &zx[at * geo.row()..(at + n) * geo.row()];
            want.extend(reference(geo, &w, rows, n, &mut states[slot]));
        }
        let slot_ids: Vec<i32> = step.iter().map(|s| s.0 as i32).collect();
        let fresh: Vec<u32> = step.iter().map(|s| u32::from(s.2)).collect();
        let Some(got) = gpu.step(&zx, &cu, &slot_ids, &fresh) else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };
        for (j, (&g, &r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= dtype.tol(r),
                "{geo:?} {dtype:?} step {i}: out[{j}] (token {}, col {}) = {g}, reference {r}",
                j / geo.inner(),
                j % geo.inner(),
            );
        }
    }
    let conv = read::<f32>(&gpu.conv_state, slots * ring);
    let ssm_got = read::<f32>(&gpu.ssm_state, slots * ssm);
    for slot in steps.iter().flatten().map(|s| s.0) {
        for (j, (&g, &r)) in conv[slot * ring..][..ring]
            .iter()
            .zip(&states[slot].conv)
            .enumerate()
        {
            // The ring holds the raw inputs: exact.
            assert_eq!(g, r, "{geo:?} {dtype:?} slot {slot}: conv ring [{j}]");
        }
        for (j, (&g, &r)) in ssm_got[slot * ssm..][..ssm]
            .iter()
            .zip(&states[slot].ssm)
            .enumerate()
        {
            assert!(
                (g - r).abs() <= 1e-4 * (1.0 + r.abs()),
                "{geo:?} {dtype:?} slot {slot}: ssm state [{j}] = {g}, reference {r}"
            );
        }
    }
}

/// Prefill → its continuation (chunks shorter than the conv window too) → decode steps → a slot
/// re-taken fresh, the batch order and the slot ids shuffled between steps.
fn schedule() -> Vec<Step> {
    vec![
        // A varlen prefill of three sequences into slots 2, 0, 3.
        vec![(2, 5, true), (0, 1, true), (3, 9, true)],
        // Each continues across the chunk boundary: 2 and 1 tokens are under the ring's 3.
        vec![(0, 7, false), (3, 2, false), (2, 1, false)],
        // Decode steps: one token a sequence.
        vec![(3, 1, false), (2, 1, false), (0, 1, false)],
        vec![(2, 1, false), (0, 1, false), (3, 1, false)],
        // Slot 0 re-taken by a new sequence (fresh) while the others decode.
        vec![(0, 4, true), (3, 1, false), (2, 1, false)],
        vec![(3, 1, false), (0, 1, false), (2, 1, false)],
    ]
}

/// The parity-tiny geometry (state 16): the gating, then the per-thread scan.
#[test]
fn mamba2_per_thread_scan_matches_reference() {
    let geo = Geometry {
        heads: 8,
        head_dim: 16,
        groups: 2,
        state: 16,
        kernel: 4,
    };
    assert!(!geo.simd());
    for dtype in [Dtype::F32, Dtype::Bf16] {
        run(geo, dtype, 4, &schedule());
    }
}

/// The real checkpoints' state (128, Nano/Super): the simd scan, lanes splitting the state, its
/// `dt` read straight off the in-projection row.
#[test]
fn mamba2_simd_scan_matches_reference() {
    let geo = Geometry {
        heads: 8,
        head_dim: 64,
        groups: 2,
        state: 128,
        kernel: 4,
    };
    assert!(geo.simd());
    for dtype in [Dtype::F32, Dtype::Bf16] {
        run(geo, dtype, 4, &schedule());
    }
}

/// The gate-before-norm grouped RMSNorm alone (`gdn_rms_norm_gated`, Mamba-2 form) at a group
/// wider than the threadgroup (Nano's 512 over 256 threads) and one narrower than a power of two:
/// `rmsnorm_g(silu(z)·y)·w`, `z` a column slice of a wider row, the weight spanning all groups.
#[test]
fn mamba2_gate_before_norm_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    for (groups, width, tokens) in [(8usize, 512usize, 3usize), (3, 48, 5)] {
        let inner = groups * width;
        let row = inner + 40;
        for dtype in [Dtype::F32, Dtype::Bf16] {
            let y = fill(tokens * inner, 21, 2.0);
            let zx: Vec<f32> = fill(tokens * row, 22, 3.0)
                .into_iter()
                .map(|x| dtype.round(x))
                .collect();
            let weight: Vec<f32> = fill(inner, 23, 0.5).iter().map(|x| 1.0 + x).collect();
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let mut want = vec![0f32; tokens * inner];
            for t in 0..tokens {
                for g in 0..groups {
                    let v: Vec<f32> = (0..width)
                        .map(|i| y[t * inner + g * width + i] * silu(zx[t * row + g * width + i]))
                        .collect();
                    let inv =
                        1.0 / (v.iter().map(|a| a * a).sum::<f32>() / width as f32 + EPS).sqrt();
                    for i in 0..width {
                        want[t * inner + g * width + i] = v[i] * inv * weight[g * width + i];
                    }
                }
            }
            let key = PipelineKey::new(
                "gdn_rms_norm_gated",
                dtype.symbol("gdn_rms_norm_gated"),
                vec![
                    ConstantValue::uint(0, width as u32),
                    ConstantValue::uint(1, (tokens * groups) as u32),
                    ConstantValue::float(2, EPS),
                    ConstantValue::uint(3, row as u32),
                    ConstantValue::uint(4, groups as u32),
                    ConstantValue::boolean(5, true),
                ],
            );
            let pipeline = baked_build(&cache, &key).expect("norm pipeline");
            let (y_b, zx_b, w_b) = (
                shared(&device, &y),
                shared(&device, &dtype.bytes(&zx)),
                shared(&device, &weight),
            );
            let out = shared(&device, &vec![0u8; tokens * inner * 4]);
            let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
                eprintln!("skipping: no MTL4 queue");
                return;
            };
            batch.encode(
                &pipeline,
                &[(&out, 0), (&y_b, 1), (&zx_b, 2), (&w_b, 3)],
                &[],
                &[],
                &[],
                size(tokens * groups, 1, 1),
                size(width.next_power_of_two().min(THREADS), 1, 1),
            );
            batch.commit(true);
            let got = dtype.decode(&out, tokens * inner);
            for (j, (&g, &r)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - r).abs() <= dtype.tol(r),
                    "groups {groups} width {width} {dtype:?}: out[{j}] = {g}, reference {r}"
                );
            }
        }
    }
}
