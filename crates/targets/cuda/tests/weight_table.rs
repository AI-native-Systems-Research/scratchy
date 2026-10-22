// SPDX-License-Identifier: Apache-2.0
//! The weight table's host half (`scratchy_target_cuda::weights`) against a tiny synthetic
//! checkpoint, with no toolchain.
//!
//! The checkpoint is written with `safetensors::serialize`, once in each name ordering a VL
//! checkpoint ships (`model.language_model.*`, the official one, and `language_model.model.*`, the
//! repacked one); the table always names the second, which the shared `GpuWeights` reader aliases.
//! Checks: every `LoadError` arm, found before anything is staged; a bf16 tensor staged byte for
//! byte; a GDN bundle staged region by region from bf16 and f32 pieces, bf16 widened to f32
//! exactly and every gap between regions `+0.0`; and the stub allocator refusing an upload.

use std::path::Path;

use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use scratchy_target_cuda::step::gdn_bundle;
use scratchy_target_cuda::tape::{ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, TapeInput};
use scratchy_target_cuda::weights::{
    CudaWeight, HostStaging, LoadError, check, open_checkpoint, stage,
};
use scratchy_tensors::{DType, DeviceAllocator};

/// The GDN geometry of the synthetic layer: 8 conv channels of 4 taps, 2 value heads of 32.
const CONV_DIM: u32 = 8;
const CONV_KERNEL: u32 = 4;
const NV: u32 = 2;
const HEAD_V: u32 = 32;

/// The tensors under the decoder root, as (name after the root, dtype, shape, bytes).
fn tensors() -> Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)> {
    let bf16s = |n: usize, seed: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| bf16::from_f32(seed + i as f32 / 64.0).to_le_bytes())
            .collect()
    };
    let f32s = |n: usize, seed: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed - i as f32 / 8.0).to_le_bytes())
            .collect()
    };
    let (c, k, nv, hv) = (
        CONV_DIM as usize,
        CONV_KERNEL as usize,
        NV as usize,
        HEAD_V as usize,
    );
    vec![
        (
            "embed_tokens.weight",
            Dtype::BF16,
            vec![16, 8],
            bf16s(128, 0.5),
        ),
        ("norm.weight", Dtype::BF16, vec![8], bf16s(8, -1.0)),
        (
            "layers.0.mlp.up_proj.weight",
            Dtype::F32,
            vec![4, 8],
            f32s(32, 1.0),
        ),
        (
            "layers.0.linear_attn.conv1d.weight",
            Dtype::BF16,
            vec![c, 1, k],
            bf16s(c * k, -0.25),
        ),
        (
            "layers.0.linear_attn.A_log",
            Dtype::F32,
            vec![nv],
            f32s(nv, 0.75),
        ),
        (
            "layers.0.linear_attn.dt_bias",
            Dtype::BF16,
            vec![nv],
            bf16s(nv, 2.0),
        ),
        (
            "layers.0.linear_attn.norm.weight",
            Dtype::F32,
            vec![hv],
            f32s(hv, 3.0),
        ),
    ]
}

/// The synthetic checkpoint in `dir` as one `model.safetensors`, every tensor under `root`.
fn write_checkpoint(dir: &Path, root: &str) {
    let all = tensors();
    let views: Vec<(String, TensorView<'_>)> = (all.iter())
        .map(|(name, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes).expect("a tensor view");
            (format!("{root}.{name}"), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize the checkpoint");
    std::fs::write(dir.join("model.safetensors"), bytes).expect("write the checkpoint");
}

/// A fresh directory under the target dir for one test's checkpoint.
fn checkpoint_dir(name: &str, root: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weight_table_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    write_checkpoint(&dir, root);
    dir
}

/// The table names the repacked ordering; both checkpoints must load through the alias.
const ROOT: &str = "language_model.model";
const ORDERINGS: [&str; 2] = ["model.language_model", "language_model.model"];

const EMBED: CudaWeight = CudaWeight::Bf16 {
    tensor: "language_model.model.embed_tokens.weight",
    shape: &[16, 8],
};
const NORM: CudaWeight = CudaWeight::Bf16 {
    tensor: "language_model.model.norm.weight",
    shape: &[8],
};
const BUNDLE: CudaWeight = CudaWeight::GdnBundle {
    conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
    a_log: "language_model.model.layers.0.linear_attn.A_log",
    dt_bias: "language_model.model.layers.0.linear_attn.dt_bias",
    norm: "language_model.model.layers.0.linear_attn.norm.weight",
    conv_dim: CONV_DIM,
    conv_kernel: CONV_KERNEL,
    nv: NV,
    head_v: HEAD_V,
};

/// A tape that declares `weight_bytes` and nothing else the table checks read.
fn tape(weight_bytes: &'static [u64]) -> CudaTape {
    let span = ArenaSpan {
        offset: ArenaOffset(0),
        bytes: 0,
    };
    CudaTape {
        bucket_m: 1,
        arena: ArenaLayout {
            colour_bytes: &[],
            colour_offsets: &[],
            total_bytes: 0,
        },
        input: TapeInput::Hidden(span),
        result: span,
        weight_bytes,
        steps: &[],
        rotary: None,
        kv: None,
        gdn: None,
    }
}

fn bundle_bytes() -> u64 {
    gdn_bundle(CONV_DIM, CONV_KERNEL, NV, HEAD_V)
        .expect("the bundle fits")
        .bytes
}

/// The table and the tapes of a well-formed model: the embedding table at 0, a source with no
/// weight at 1 (`None`, 0 bytes), the norm at 2, the GDN bundle at 3.
fn good() -> (Vec<Option<CudaWeight>>, Vec<CudaTape>) {
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes()]));
    (
        vec![Some(EMBED), None, Some(NORM), Some(BUNDLE)],
        vec![tape(bytes), tape(bytes)],
    )
}

#[test]
fn a_well_formed_table_checks_in_both_orderings() {
    let (table, tapes) = good();
    for root in ORDERINGS {
        let w = open_checkpoint(&checkpoint_dir(&format!("good_{root}"), root)).expect("open");
        check(&table, &tapes, &w).unwrap_or_else(|e| panic!("{root}: {e}"));
    }
}

#[test]
fn a_bf16_tensor_is_staged_byte_for_byte() {
    for root in ORDERINGS {
        let mut w = open_checkpoint(&checkpoint_dir(&format!("bf16_{root}"), root)).expect("open");
        let staged = stage(0, &EMBED, &mut w).expect("stage the table");
        let want = &tensors()[0].3;
        assert_eq!(&staged, want, "{root}: the embedding table's bytes");
    }
}

/// The bundle's four regions at `gdn_bundle`'s offsets, each piece widened to f32 exactly (bf16
/// to f32 loses nothing, f32 is copied), the conv as `[conv_dim, K]` in on-disk order, and every
/// byte between regions `+0.0`.
#[test]
fn a_gdn_bundle_is_packed_exactly() {
    let layout = gdn_bundle(CONV_DIM, CONV_KERNEL, NV, HEAD_V).expect("the bundle fits");
    assert_eq!(
        (
            layout.conv1d,
            layout.a_log,
            layout.dt_bias,
            layout.norm,
            layout.bytes
        ),
        (0, 256, 512, 768, 1024),
        "the synthetic bundle's regions"
    );
    let all = tensors();
    let widen = |bytes: &[u8], dtype: Dtype| -> Vec<f32> {
        match dtype {
            Dtype::BF16 => (bytes.as_chunks::<2>().0.iter())
                .map(|&p| bf16::from_le_bytes(p).to_f32())
                .collect(),
            _ => (bytes.as_chunks::<4>().0.iter())
                .map(|&q| f32::from_le_bytes(q))
                .collect(),
        }
    };
    for root in ORDERINGS {
        let mut w =
            open_checkpoint(&checkpoint_dir(&format!("bundle_{root}"), root)).expect("open");
        let staged = stage(3, &BUNDLE, &mut w).expect("stage the bundle");
        assert_eq!(
            staged.len() as u64,
            layout.bytes,
            "{root}: the bundle's bytes"
        );
        let f: Vec<f32> = (staged.as_chunks::<4>().0.iter())
            .map(|&q| f32::from_le_bytes(q))
            .collect();
        let mut written = vec![false; f.len()];
        for (piece, at) in [
            (3, layout.conv1d),
            (4, layout.a_log),
            (5, layout.dt_bias),
            (6, layout.norm),
        ] {
            let (name, dtype, _, bytes) = &all[piece];
            let want = widen(bytes, *dtype);
            let at = at as usize / 4;
            assert_eq!(
                &f[at..at + want.len()],
                want.as_slice(),
                "{root}: {name} at {at}"
            );
            written[at..at + want.len()].fill(true);
        }
        for (i, v) in f.iter().enumerate().filter(|(i, _)| !written[*i]) {
            assert_eq!(v.to_bits(), 0, "{root}: gap element {i} is {v}, not +0.0");
        }
        assert!(
            written.iter().any(|w| !w),
            "the synthetic bundle has gaps to check"
        );
    }
}

/// `table` against the good tapes (or `tapes`), with the checkpoint at the repacked root.
fn refusal(name: &str, table: &[Option<CudaWeight>], tapes: &[CudaTape]) -> LoadError {
    let w = open_checkpoint(&checkpoint_dir(name, ROOT)).expect("open");
    let e = check(table, tapes, &w).expect_err("the table is refused");
    eprintln!("{name}: {e}");
    e
}

#[test]
fn a_missing_checkpoint_is_refused() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("weight_table_none");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the dir");
    let Err(e) = open_checkpoint(&dir) else {
        panic!("an empty dir opened as a checkpoint");
    };
    assert!(matches!(e, LoadError::NoCheckpoint { .. }), "{e}");
}

#[test]
fn a_table_of_another_length_is_refused() {
    let (table, tapes) = good();
    let e = refusal("length", &table[..3], &tapes);
    assert!(
        matches!(
            e,
            LoadError::TableLength {
                table: 3,
                tape: 4,
                bucket: 0
            }
        ),
        "{e}"
    );
}

#[test]
fn a_missing_tensor_is_refused_by_name_and_index() {
    let (mut table, tapes) = good();
    table[2] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.final_norm.weight",
        shape: &[8],
    });
    let e = refusal("missing", &table, &tapes);
    assert!(e.to_string().contains("final_norm.weight"), "{e}");
    assert!(matches!(e, LoadError::Missing { ix: 2, .. }), "{e}");
}

/// A dense tensor stored as f32 is refused, never converted.
#[test]
fn a_dense_tensor_of_another_dtype_is_refused() {
    let (mut table, _) = good();
    table[2] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.mlp.up_proj.weight",
        shape: &[4, 8],
    });
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 64, bundle_bytes()]));
    let e = refusal("dtype", &table, &[tape(bytes)]);
    assert!(
        matches!(
            e,
            LoadError::Dtype {
                ix: 2,
                got: DType::F32,
                ..
            }
        ),
        "{e}"
    );
}

#[test]
fn a_tensor_of_another_shape_is_refused() {
    let (mut table, tapes) = good();
    table[0] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.embed_tokens.weight",
        shape: &[8, 16],
    });
    let e = refusal("shape", &table, &tapes);
    assert!(e.to_string().contains("[16, 8]"), "{e}");
    let LoadError::Shape {
        ix: 0, got, want, ..
    } = e
    else {
        panic!("{e}");
    };
    assert_eq!((got, want), (vec![16, 8], vec![8, 16]));
    // A GDN piece of another shape: a bundle of 3 value heads reads `[3]` A_log.
    let (mut table, _) = good();
    table[3] = Some(bundle_of(3));
    let e = refusal("gdn_shape", &table, &tapes);
    assert!(
        matches!(e, LoadError::Shape { ix: 3, tensor, .. } if tensor.ends_with("A_log")),
        "{e}"
    );
}

/// [`BUNDLE`] with `nv` value heads.
fn bundle_of(nv: u32) -> CudaWeight {
    let CudaWeight::GdnBundle {
        conv1d,
        a_log,
        dt_bias,
        norm,
        conv_dim,
        conv_kernel,
        head_v,
        ..
    } = BUNDLE
    else {
        unreachable!("BUNDLE is a bundle")
    };
    CudaWeight::GdnBundle {
        conv1d,
        a_log,
        dt_bias,
        norm,
        conv_dim,
        conv_kernel,
        nv,
        head_v,
    }
}

/// A packed size other than the tapes declare, and a `None` index that declares bytes.
#[test]
fn bytes_other_than_declared_are_refused() {
    let (table, _) = good();
    let short: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes() - 256]));
    let e = refusal("bytes", &table, &[tape(short)]);
    assert!(
        matches!(e, LoadError::Bytes { ix: 3, packed, declared } if packed == bundle_bytes() && declared == bundle_bytes() - 256),
        "{e}"
    );
    let unread: &'static [u64] = Box::leak(Box::new([256, 2, 16, bundle_bytes()]));
    let e = refusal("unread", &table, &[tape(unread)]);
    assert!(
        matches!(
            e,
            LoadError::Bytes {
                ix: 1,
                packed: 0,
                declared: 2
            }
        ),
        "{e}"
    );
}

/// The packed size is the largest any tape declares: a bucket declaring less does not hide one
/// declaring more.
#[test]
fn the_largest_declared_size_is_the_one_checked() {
    let (table, _) = good();
    let a: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes()]));
    let b: &'static [u64] = Box::leak(Box::new([512, 0, 16, bundle_bytes()]));
    let e = refusal("largest", &table, &[tape(a), tape(b)]);
    assert!(
        matches!(
            e,
            LoadError::Bytes {
                ix: 0,
                packed: 256,
                declared: 512
            }
        ),
        "{e}"
    );
}

/// The stub allocator makes no device memory: an upload through it is refused.
#[test]
fn host_staging_refuses_an_upload() {
    let bytes = [0u8; 16];
    let mut a = HostStaging;
    // SAFETY: `bytes` is 16 readable bytes for the call.
    let got = unsafe { a.alloc_and_copy_host(bytes.as_ptr(), bytes.len()) };
    let e = got.expect_err("HostStaging allocated device memory");
    assert!(e.to_string().contains("DeviceBuffer"), "{e}");
    // SAFETY: as above.
    let got = unsafe { a.alloc_and_copy_host_aligned(bytes.as_ptr(), bytes.len(), 2) };
    assert!(got.is_err(), "the aligned upload went through");
}

/// The tie, as the compiler emits it: the lm_head's index has no entry and declares 0 bytes, and
/// the one tensor it shares with the embedding table is staged once, at the table's index. The
/// reader hands a tensor out once, so a second entry naming it would be refused as missing.
#[test]
fn a_tied_head_stages_its_tensor_once() {
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes(), 0]));
    let table = [Some(EMBED), None, Some(NORM), Some(BUNDLE), None];
    let mut w = open_checkpoint(&checkpoint_dir("tied", ROOT)).expect("open");
    check(&table, &[tape(bytes)], &w).expect("the tied table checks");
    let staged: Vec<usize> = (table.iter().enumerate())
        .filter_map(|(ix, e)| e.as_ref().map(|e| (ix, e)))
        .map(|(ix, e)| stage(ix, e, &mut w).map(|b| b.len()))
        .collect::<Result<_, _>>()
        .expect("stage every entry");
    assert_eq!(staged, [256, 16, bundle_bytes() as usize]);
    let e = stage(4, &EMBED, &mut w).expect_err("the table's tensor was handed out twice");
    assert!(matches!(e, LoadError::Missing { ix: 4, .. }), "{e}");
}
