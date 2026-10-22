// SPDX-License-Identifier: Apache-2.0
//! A draft head's pass past its vocabulary prefix ([`SuffixDraft`]) against the whole
//! vocabulary's: over a random MLX-affine lm_head `W [V, H]` and a pass's rows `x [m, H]`, the
//! matvec of the prefix's rows, then the suffix's — read at the prefix's row offsets into the
//! codes, scales and biases — and `argmax_split` pick what one matvec over every row and the
//! argmax over the whole row pick, for every row count a pass runs and both ways a device reads
//! codes.

#![cfg(target_os = "macos")]

use half::bf16;
use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{
    Buffer, Mtl4DispatchBatch, read_slice, shared_slice, shared_zeroed,
};
use scratchy_target_metal::off_tape::VocabSuffix;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{KDimI32, LogitsWidth, MDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineQmvConstants, AffineQmvWideConstants, ArgmaxConstants,
};
use scratchy_target_metal::tape::lowered::MetalDtype;
use scratchy_target_metal::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, pick_qmv_kernel_wide, qmv_dispatch_shape,
    qmv_kernel_static_name,
};
use scratchy_target_metal::vocab_suffix::{LmHead, LmHeadPart, PassPick, SuffixDraft};

/// The lm_head: `V` rows, `H` wide, 4-bit codes in groups of 64; its first `P` rows the prefix.
const V: u32 = 3000;
const H: u32 = 256;
const P: u32 = 1024;
const BITS: u32 = 4;
const GROUP: u32 = 64;
/// The most rows a pass runs.
const VERIFY_ROWS: u32 = 8;

fn size((width, height, depth): (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

/// `n` bf16 values of a deterministic pattern, `salt` apart.
fn bf16s(n: usize, salt: u32, scale: f32) -> Vec<u16> {
    (0..n as u32)
        .map(|i| {
            let h = i
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            bf16::from_f32(scale * ((h >> 16) as f32 / 65536.0 - 0.5)).to_bits()
        })
        .collect()
}

/// The matvec key over `n` rows of the lm_head for `m` input rows, its codes read as `codes`.
fn qmv_key(n: u32, m: u32, codes: AffineCodes) -> (PipelineKey, QmvKernel) {
    let kernel = match m {
        1 => pick_qmv_kernel(n, H, BITS),
        _ => pick_qmv_kernel_wide(n, H, BITS, m, true),
    };
    let name = qmv_kernel_static_name(kernel, DequantDtype::Bf16, ScaleDtype::Bf16, BITS, GROUP);
    let (k, n_dim) = (KDimI32(H as i32), NDimI32(n as i32));
    let constants: Vec<ConstantValue> = match kernel {
        QmvKernel::Wide { .. } => AffineQmvWideConstants {
            k,
            n: n_dim,
            m: MDimI32(m as i32),
            codes,
        }
        .into(),
        _ => AffineQmvConstants { k, n: n_dim, codes }.into(),
    };
    (PipelineKey::new("quantized_qmv", name, constants), kernel)
}

#[test]
fn a_pass_past_the_prefix_picks_what_the_whole_vocabulary_picks() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let build = |key: &PipelineKey| baked_build(&cache, key).expect("pipeline");
    let suffix = VocabSuffix {
        prefix: P,
        rows: V - P,
        hidden: H,
        bits: BITS,
        group: GROUP,
        scale: ScaleDtype::Bf16,
        verify_rows: VERIFY_ROWS,
        window: 4,
    };
    let keys = suffix.kernel_keys(MetalDtype::Bf16);
    let words: Vec<u32> = (0..V * H * BITS / 32)
        .map(|i| i.wrapping_mul(2246822519).rotate_left(13) ^ 0x5bd1_e995)
        .collect();
    let groups = (V * H / GROUP) as usize;
    let (scales, biases) = (
        shared_slice(&device, &bf16s(groups, 1, 0.02)),
        shared_slice(&device, &bf16s(groups, 2, 0.01)),
    );
    // A pass's rows sit at row 3 of its hidden buffer, as a pass-1 window past its first rows; the
    // whole-vocabulary matvec reads the same rows from a buffer of their own.
    let first = 3usize;
    let rows = bf16s((first + VERIFY_ROWS as usize) * H as usize, 3, 1.0);
    let x_all = shared_slice(&device, &rows);
    let x = shared_slice(&device, &rows[first * H as usize..]);
    let hidden = x_all.gpuAddress() + (first * H as usize * 2) as u64;
    // Every id picked: the prefix's and the suffix's both, or the comparison proves nothing.
    let mut picked = Vec::new();
    for codes_read in [AffineCodes::AsWritten, AffineCodes::Offset8] {
        // Stored the way the kernels read them: offset-8 codes are XOR 0x88 a byte.
        let stored: Vec<u32> = match codes_read {
            AffineCodes::AsWritten => words.clone(),
            AffineCodes::Offset8 => words.iter().map(|w| w ^ 0x8888_8888).collect(),
        };
        let codes = shared_slice(&device, &stored);
        let qmv_pipelines: Vec<Box<_>> = (keys.qmv.get(codes_read).iter())
            .map(|k| Box::new(build(k)))
            .collect();
        for m in 1..=VERIFY_ROWS {
            let qmv = |batch: &mut Mtl4DispatchBatch, n: u32, y: &Buffer| {
                let (key, kernel) = qmv_key(n, m, codes_read);
                let (grid, threads) = qmv_dispatch_shape(kernel, m, n, 1);
                let pso = build(&key);
                let binds = [(&codes, 0), (&scales, 1), (&biases, 2), (&x, 3), (y, 4)];
                batch.encode(&pso, &binds, &[], &[], &[], size(grid), size(threads));
                batch.barrier();
            };
            // The whole vocabulary: one matvec over every row, the argmax over each whole row.
            let full = shared_zeroed(&device, (m * V) as usize * 2);
            let want = shared_zeroed(&device, m as usize * 4);
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
            qmv(&mut batch, V, &full);
            let argmax = build(&PipelineKey::new(
                "argmax",
                "argmax_bf16",
                ArgmaxConstants {
                    vocab: LogitsWidth(V),
                }
                .into(),
            ));
            let one = |w: usize| MTLSize {
                width: w,
                height: 1,
                depth: 1,
            };
            batch.encode(
                &argmax,
                &[(&full, 0), (&want, 1)],
                &[(m, 2)],
                &[],
                &[],
                one(m as usize),
                one(1024),
            );
            batch.commit(true);
            // The prefix's matvec, then the pass past it.
            let prefix_logits = shared_zeroed(&device, (m * P) as usize * 2);
            let got = shared_zeroed(&device, m as usize * 4);
            let batch_const = shared_slice(&device, &[m]);
            let mut batch = Mtl4DispatchBatch::begin(&device).expect("batch");
            qmv(&mut batch, P, &prefix_logits);
            let res = batch.residency();
            let logits = res.pin(shared_zeroed(&device, (VERIFY_ROWS * (V - P)) as usize * 2));
            let pins = [&got, &batch_const, &x_all].map(|b| res.pin(b.clone()));
            let part = |buffer: &Buffer| LmHeadPart {
                buffer: buffer.clone(),
                offset: 0,
            };
            let lm_head = LmHead {
                codes: part(&codes),
                scales: part(&scales),
                biases: part(&biases),
            };
            let draft = SuffixDraft::new(
                suffix,
                lm_head,
                qmv_pipelines.clone(),
                Box::new(build(&keys.argmax)),
                logits,
            )
            .expect("suffix draft");
            let pick = PassPick {
                hidden,
                prefix_logits: prefix_logits.gpuAddress(),
                picked: got.gpuAddress(),
                batch_const: batch_const.gpuAddress(),
                batch: m as usize,
            };
            draft
                .encode(batch.encoder(), &device, pick)
                .expect("encode");
            batch.commit(true);
            drop(pins);
            let (want, got): (Vec<u32>, Vec<u32>) =
                (read_slice(&want, m as usize), read_slice(&got, m as usize));
            assert_eq!(got, want, "{m} rows, codes {codes_read:?}");
            picked.extend(want);
        }
    }
    assert!(
        picked.iter().any(|&id| id < P) && picked.iter().any(|&id| id >= P),
        "the rows' argmaxes fall on both sides of the prefix: {picked:?}"
    );
}
