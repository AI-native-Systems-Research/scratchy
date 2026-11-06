// SPDX-License-Identifier: Apache-2.0
//! ONE MODEL'S WEIGHT TABLE, AND THE HOST HALF OF ITS LOADER.
//!
//! `scratchy-target-cuda-compiler` emits, beside a model's tapes, `CUDA_WEIGHTS: &[Option<
//! CudaWeight>]`: per weight index a tape reads (`WeightIx`), which checkpoint tensor fills it, the
//! shape that tensor must have, and how its bytes become the device buffer's. `None` is an index no
//! step reads (an activation source, or a tensor another index already holds).
//!
//! The loader plays that table against a checkpoint read through the shared
//! `scratchy_layers::weights::GpuWeights` (which aliases the `language_model.model.*` and
//! `model.language_model.*` orderings of a VL checkpoint). This half needs no toolchain:
//! [`check`] checks every entry against the checkpoint's header before anything is allocated, and
//! [`stage`] makes one entry's host bytes. The upload is `runtime::load_weights`, under `nvcc`.
//!
//! Every conversion is exact: a bf16 or e4m3 tensor is uploaded byte for byte, and a GDN bundle's
//! pieces and an fp8 weight's per-channel scales are widened to f32, which loses nothing from bf16
//! or f32.

use std::path::{Path, PathBuf};

use scratchy_layers::weights::GpuWeights;
use scratchy_tensors::{DType, DeviceAllocator, PoolMemory};
use serde::Serialize;

use scratchy_subtile::handoff::{BundleTensor, PER_EXPERT_TENSOR_LEAVES};

use crate::step::{expert_bank, fp8_moe_layout, fp8_weight, gdn_bundle, moe_bundle};
use crate::tape::CudaTape;

/// How one weight index is filled from the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CudaWeight {
    /// One bf16 tensor uploaded byte for byte. `shape` is what the step reading it needs: a GEMM's
    /// `[n, k]`, a norm's `[cols]`, the embed table's `[vocab, hidden]`.
    Bf16 {
        tensor: &'static str,
        shape: &'static [u64],
    },
    /// One GDN layer's packed f32 bundle (`step::gdn_bundle`), from four tensors: `conv1d`
    /// `[conv_dim, 1, conv_kernel]`, `a_log` and `dt_bias` `[nv]`, `norm` `[head_v]`, each bf16 or
    /// f32.
    GdnBundle {
        conv1d: &'static str,
        a_log: &'static str,
        dt_bias: &'static str,
        norm: &'static str,
        conv_dim: u32,
        conv_kernel: u32,
        nv: u32,
        head_v: u32,
    },
    /// One MoE layer's dense bf16 bundle (`step::moe_bundle`), its three checkpoint tensors back to
    /// back, each uploaded byte for byte: `router` `[experts, hidden]` (`{prefix}.gate.weight`),
    /// `gate_up` `[experts, 2·inter, hidden]` (`{prefix}.experts.gate_up_proj`; per expert the
    /// first `inter` rows gate, the next up) and `down` `[experts, hidden, inter]`
    /// (`{prefix}.experts.down_proj`). Only the fused 3-D form: a checkpoint storing each expert's
    /// projections as tensors of their own is refused, naming the fused tensor it lacks.
    MoeBundle {
        router: &'static str,
        gate_up: &'static str,
        down: &'static str,
        experts: u32,
        hidden: u32,
        inter: u32,
    },
    /// Gemma's dense bf16 expert bank (`step::expert_bank`): [`Self::MoeBundle`] with no router,
    /// its two checkpoint tensors back to back, each uploaded byte for byte: `gate_up` `[experts,
    /// 2·inter, hidden]` (`{prefix}.gate_up_proj`; per expert the first `inter` rows gate, the next
    /// up) and `down` `[experts, hidden, inter]` (`{prefix}.down_proj`). The router's tensors are
    /// [`Self::Bf16`] entries of their own.
    ExpertBank {
        gate_up: &'static str,
        down: &'static str,
        experts: u32,
        hidden: u32,
        inter: u32,
    },
    /// One per-channel fp8 projection (`step::fp8_weight`): `weight` `[n, k]` e4m3 uploaded byte
    /// for byte, then `scale` (`[n, 1]` bf16, `{prefix}.weight_scale`) widened to `[n]` f32.
    Fp8 {
        weight: &'static str,
        scale: &'static str,
        /// `[n, k]`.
        shape: &'static [u64],
    },
    /// Qwen3.6's fp8 MoE layer (`step::fp8_moe_layout` with the router): `router` `[experts,
    /// hidden]` bf16 (`{prefix}.gate.weight`, which the checkpoint leaves unquantized), then each
    /// expert's e4m3 `gate_proj` and `up_proj` `[inter, hidden]` and `down_proj` `[hidden,
    /// inter]`, gathered from `{experts_at}.{e}.<leaf>` (`handoff::PER_EXPERT_TENSOR_LEAVES`),
    /// then their `[out, 1]` bf16 scales widened to f32.
    Fp8MoeBundle {
        router: &'static str,
        experts_at: &'static str,
        experts: u32,
        hidden: u32,
        inter: u32,
    },
    /// Gemma's fp8 expert bank: [`Self::Fp8MoeBundle`] with no router (its `router.proj` is a
    /// [`Self::Bf16`] entry of its own, unquantized), the experts at `{experts_at}.{e}.<leaf>`.
    Fp8ExpertBank {
        experts_at: &'static str,
        experts: u32,
        hidden: u32,
        inter: u32,
    },
}

/// The dtype an fp8 weight has on disk: e4m3, never converted.
const FP8_DTYPES: &[DType] = &[DType::Fp8E4m3];

/// The dtypes a dense tensor may have on disk: bf16 only, never converted.
const DENSE_DTYPES: &[DType] = &[DType::BF16];
/// The dtypes a GDN bundle's piece may have on disk: both widen to f32 exactly.
const GDN_DTYPES: &[DType] = &[DType::BF16, DType::F32];

/// One checkpoint tensor an entry reads, the dtypes it may have, and the shape it must have.
struct Read {
    tensor: String,
    accepted: &'static [DType],
    shape: Vec<u64>,
}

impl CudaWeight {
    /// The tensor that names this entry: a bf16 entry's one tensor, a bundle's conv1d.
    pub fn lead(&self) -> &'static str {
        match *self {
            Self::Bf16 { tensor, .. } => tensor,
            Self::GdnBundle { conv1d, .. } => conv1d,
            Self::MoeBundle { router, .. } => router,
            Self::ExpertBank { gate_up, .. } => gate_up,
            Self::Fp8 { weight, .. } => weight,
            Self::Fp8MoeBundle { router, .. } => router,
            Self::Fp8ExpertBank { experts_at, .. } => experts_at,
        }
    }

    /// The tensors this entry reads, each with its dtypes and shape.
    fn reads(&self) -> Vec<Read> {
        match *self {
            Self::Bf16 { tensor, shape } => vec![Read {
                tensor: tensor.to_string(),
                accepted: DENSE_DTYPES,
                shape: shape.to_vec(),
            }],
            Self::Fp8 {
                weight,
                scale,
                shape,
            } => vec![
                Read {
                    tensor: weight.to_string(),
                    accepted: FP8_DTYPES,
                    shape: shape.to_vec(),
                },
                Read {
                    tensor: scale.to_string(),
                    accepted: DENSE_DTYPES,
                    shape: vec![shape.first().copied().unwrap_or(0), 1],
                },
            ],
            Self::Fp8MoeBundle {
                router,
                experts_at,
                experts,
                hidden,
                inter,
            } => {
                let mut reads = vec![Read {
                    tensor: router.to_string(),
                    accepted: DENSE_DTYPES,
                    shape: vec![u64::from(experts), u64::from(hidden)],
                }];
                reads.extend(expert_reads(experts_at, experts, hidden, inter));
                reads
            }
            Self::Fp8ExpertBank {
                experts_at,
                experts,
                hidden,
                inter,
            } => expert_reads(experts_at, experts, hidden, inter),
            Self::GdnBundle {
                conv1d,
                a_log,
                dt_bias,
                norm,
                conv_dim,
                conv_kernel,
                nv,
                head_v,
            } => {
                let read = |tensor: &str, shape: &[u32]| Read {
                    tensor: tensor.to_string(),
                    accepted: GDN_DTYPES,
                    shape: shape.iter().map(|&d| u64::from(d)).collect(),
                };
                vec![
                    read(conv1d, &[conv_dim, 1, conv_kernel]),
                    read(a_log, &[nv]),
                    read(dt_bias, &[nv]),
                    read(norm, &[head_v]),
                ]
            }
            Self::MoeBundle {
                router,
                gate_up,
                down,
                experts,
                hidden,
                inter,
            } => {
                let read = |tensor: &str, shape: &[u32]| Read {
                    tensor: tensor.to_string(),
                    accepted: DENSE_DTYPES,
                    shape: shape.iter().map(|&d| u64::from(d)).collect(),
                };
                vec![
                    read(router, &[experts, hidden]),
                    read(gate_up, &[experts, 2 * inter, hidden]),
                    read(down, &[experts, hidden, inter]),
                ]
            }
            Self::ExpertBank {
                gate_up,
                down,
                experts,
                hidden,
                inter,
            } => {
                let read = |tensor: &str, shape: &[u32]| Read {
                    tensor: tensor.to_string(),
                    accepted: DENSE_DTYPES,
                    shape: shape.iter().map(|&d| u64::from(d)).collect(),
                };
                vec![
                    read(gate_up, &[experts, 2 * inter, hidden]),
                    read(down, &[experts, hidden, inter]),
                ]
            }
        }
    }

    /// The device buffer's bytes, or `None` when the count overflows `u64`.
    pub fn packed_bytes(&self) -> Option<u64> {
        match *self {
            Self::Bf16 { shape, .. } => (shape.iter()).try_fold(2u64, |acc, &d| acc.checked_mul(d)),
            Self::GdnBundle {
                conv_dim,
                conv_kernel,
                nv,
                head_v,
                ..
            } => gdn_bundle(conv_dim, conv_kernel, nv, head_v).map(|b| b.bytes),
            Self::MoeBundle {
                experts,
                hidden,
                inter,
                ..
            } => moe_bundle(experts, hidden, inter).map(|b| b.bytes),
            Self::ExpertBank {
                experts,
                hidden,
                inter,
                ..
            } => expert_bank(experts, hidden, inter).map(|b| b.bytes),
            Self::Fp8 { shape, .. } => fp8_layout(shape).map(|w| w.bytes),
            Self::Fp8MoeBundle {
                experts,
                hidden,
                inter,
                ..
            } => fp8_moe_layout(experts, hidden, inter, true).map(|b| b.experts.bytes),
            Self::Fp8ExpertBank {
                experts,
                hidden,
                inter,
                ..
            } => fp8_moe_layout(experts, hidden, inter, false).map(|b| b.experts.bytes),
        }
    }
}

/// The buffer of an fp8 projection of `shape`, `[n, k]` (`step::fp8_weight`); `None` for another
/// rank or on overflow.
fn fp8_layout(shape: &[u64]) -> Option<crate::step::Fp8Weight> {
    let &[n, k] = shape else {
        return None;
    };
    fp8_weight(u32::try_from(n).ok()?, u32::try_from(k).ok()?)
}

/// Expert `e`'s checkpoint tensor of `leaf` (a [`PER_EXPERT_TENSOR_LEAVES`] row's) under
/// `experts_at`: `{experts_at}.{e}.{leaf}`.
fn expert_tensor(experts_at: &str, e: u32, which: BundleTensor) -> String {
    let leaf = (PER_EXPERT_TENSOR_LEAVES.iter())
        .find_map(|&(t, l)| (t == which).then_some(l))
        .expect("every fp8 expert tensor has a per-expert leaf");
    format!("{experts_at}.{e}.{leaf}")
}

/// Every expert's fp8 tensors, in expert order: its gate, up and down weights (e4m3 `[inter,
/// hidden]`, `[inter, hidden]`, `[hidden, inter]`) and their `[out, 1]` bf16 scales.
fn expert_reads(experts_at: &str, experts: u32, hidden: u32, inter: u32) -> Vec<Read> {
    let (h, i) = (u64::from(hidden), u64::from(inter));
    let mut reads = Vec::new();
    for e in 0..experts {
        for (w, sc, out, inp) in [
            (BundleTensor::ExpertGateW, BundleTensor::ExpertGateS, i, h),
            (BundleTensor::ExpertUpW, BundleTensor::ExpertUpS, i, h),
            (BundleTensor::ExpertDownW, BundleTensor::ExpertDownS, h, i),
        ] {
            reads.push(Read {
                tensor: expert_tensor(experts_at, e, w),
                accepted: FP8_DTYPES,
                shape: vec![out, inp],
            });
            reads.push(Read {
                tensor: expert_tensor(experts_at, e, sc),
                accepted: DENSE_DTYPES,
                shape: vec![out, 1],
            });
        }
    }
    reads
}

/// Why a checkpoint cannot fill a weight table. Every arm but [`LoadError::Upload`] is found
/// before any device allocation, and each names the weight index and, where there is one, the
/// tensor.
#[derive(Debug)]
pub enum LoadError {
    /// The checkpoint directory could not be opened as safetensors.
    NoCheckpoint { dir: PathBuf, why: String },
    /// The table has `table` entries; the tape of `bucket` declares `tape` weights.
    TableLength {
        table: usize,
        tape: usize,
        bucket: usize,
    },
    /// Weight `ix` reads `tensor`, which the checkpoint does not hold.
    Missing { ix: usize, tensor: &'static str },
    /// Weight `ix`, a MoE bundle, reads the fused 3-D `tensor` (`experts.gate_up_proj` or
    /// `experts.down_proj`), which the checkpoint does not hold: it stores each expert's
    /// projections as tensors of their own (`experts.{e}.gate_proj.weight`), a form this loader does
    /// not take.
    PerExpertTensors { ix: usize, tensor: &'static str },
    /// Weight `ix` reads `tensor`, which is `got` on disk; it may be one of `accepted`.
    Dtype {
        ix: usize,
        tensor: &'static str,
        got: DType,
        accepted: &'static [DType],
    },
    /// Weight `ix` reads `tensor`, which is `got` on disk; the step reading it needs `want`.
    Shape {
        ix: usize,
        tensor: &'static str,
        got: Vec<u64>,
        want: Vec<u64>,
    },
    /// Weight `ix`, which reads `tensor` (a bundle's first, its conv1d; `None` for an index with
    /// no entry, which packs to 0), packs to `packed` bytes; the largest any tape declares is
    /// `declared`.
    Bytes {
        ix: usize,
        tensor: Option<&'static str>,
        packed: u64,
        declared: u64,
    },
    /// Uploading weight `ix` failed; the buffers made before it were freed.
    #[cfg(feature = "nvcc")]
    Upload {
        ix: usize,
        error: crate::runtime::CudaError,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCheckpoint { dir, why } => {
                write!(f, "cannot open the checkpoint at {}: {why}", dir.display())
            }
            Self::TableLength {
                table,
                tape,
                bucket,
            } => write!(
                f,
                "the weight table has {table} entries; the tape of bucket {bucket} declares {tape} \
                 weights"
            ),
            Self::Missing { ix, tensor } => {
                write!(f, "weight {ix}: the checkpoint has no tensor {tensor}")
            }
            Self::PerExpertTensors { ix, tensor } => write!(
                f,
                "weight {ix}: the checkpoint has no fused expert tensor {tensor}; it stores each \
                 expert's projections as tensors of their own, which the CUDA loader does not take"
            ),
            Self::Dtype {
                ix,
                tensor,
                got,
                accepted,
            } => {
                let accepted: Vec<String> = accepted.iter().map(|d| d.to_string()).collect();
                write!(
                    f,
                    "weight {ix}: {tensor} is {got} on disk; it may be {}",
                    accepted.join(" or ")
                )
            }
            Self::Shape {
                ix,
                tensor,
                got,
                want,
            } => write!(
                f,
                "weight {ix}: {tensor} is {got:?} on disk; the step reading it needs {want:?}"
            ),
            Self::Bytes {
                ix,
                tensor: Some(tensor),
                packed,
                declared,
            } => write!(
                f,
                "weight {ix} ({tensor}) packs to {packed} bytes; the tapes declare {declared}"
            ),
            Self::Bytes {
                ix,
                tensor: None,
                packed,
                declared,
            } => write!(
                f,
                "weight {ix} (no entry, no tensor) packs to {packed} bytes; the tapes declare \
                 {declared}"
            ),
            #[cfg(feature = "nvcc")]
            Self::Upload { ix, error } => write!(f, "weight {ix}: the upload failed: {error}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// `GpuWeights`' allocator for a CUDA load: it holds no device memory and makes none. CUDA uploads
/// through `runtime::DeviceBuffer`, one allocation per weight, so every hook that would put bytes
/// on a device refuses; `GpuWeights` is used for its host-side reads only (the header, the
/// aliases, the mmap'd bytes).
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStaging;

/// The device memory [`HostStaging`] tracks: none. Uninhabited, so no value of it exists.
pub enum NoDeviceMemory {}

impl PoolMemory for NoDeviceMemory {
    fn ptr(&self) -> *mut u8 {
        match *self {}
    }
}

impl DeviceAllocator for HostStaging {
    type Stream = ();
    type Mem = NoDeviceMemory;

    fn stream(&self) {}

    unsafe fn alloc_and_copy_host(
        &mut self,
        _src_host: *const u8,
        bytes: usize,
    ) -> anyhow::Result<*mut u8> {
        anyhow::bail!(
            "HostStaging allocates no device memory ({bytes} bytes asked): CUDA uploads through \
             DeviceBuffer"
        )
    }
}

/// Weight `ix`'s tensors, by `weights`' header: each present, of an accepted dtype, of exactly
/// the shape the entry needs.
fn check_tensors<A: DeviceAllocator>(
    ix: usize,
    entry: &CudaWeight,
    weights: &GpuWeights<A>,
) -> Result<(), LoadError> {
    for Read {
        tensor,
        accepted,
        shape,
    } in entry.reads()
    {
        // A per-expert name is built here, not baked: a refusal keeps the one it names.
        let named = |t: &str| -> &'static str { Box::leak(t.to_owned().into_boxed_str()) };
        let missing = || match *entry {
            CudaWeight::MoeBundle {
                router,
                gate_up,
                down,
                ..
            } if tensor != router => {
                let fused = if tensor == gate_up { gate_up } else { down };
                let base = fused.rsplit_once(".experts.").map_or(fused, |(b, _)| b);
                let per_expert = format!("{base}.experts.0.gate_proj.weight");
                match weights.tensor_info(&per_expert) {
                    Some(_) => LoadError::PerExpertTensors { ix, tensor: fused },
                    None => LoadError::Missing {
                        ix,
                        tensor: named(&tensor),
                    },
                }
            }
            _ => LoadError::Missing {
                ix,
                tensor: named(&tensor),
            },
        };
        let (got, dtype) = (weights.tensor_info(&tensor)).ok_or_else(missing)?;
        if !accepted.contains(&dtype) {
            return Err(LoadError::Dtype {
                ix,
                tensor: named(&tensor),
                got: dtype,
                accepted,
            });
        }
        let got: Vec<u64> = got.iter().map(|&d| d as u64).collect();
        if got != shape {
            return Err(LoadError::Shape {
                ix,
                tensor: named(&tensor),
                got,
                want: shape,
            });
        }
    }
    Ok(())
}

/// Little-endian `bytes` of `dtype` (bf16 or f32, which [`check_tensors`] accepted) as f32, read
/// byte by byte: an mmap'd tensor need not be aligned to its element.
fn widen(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => {
            let (pairs, _) = bytes.as_chunks::<2>();
            (pairs.iter())
                .map(|&p| half::bf16::from_le_bytes(p).to_f32())
                .collect()
        }
        _ => {
            let (quads, _) = bytes.as_chunks::<4>();
            quads.iter().map(|&q| f32::from_le_bytes(q)).collect()
        }
    }
}

/// The checkpoint at `dir`, opened for its host-side reads: `GpuWeights::from_dir` (a sharded
/// `model.safetensors.index.json` or one `model.safetensors`, mmap'd, its header parsed, the VL
/// name orderings aliased). No tensor's bytes are read, and nothing is allocated on a device.
pub fn open_checkpoint(dir: &Path) -> Result<GpuWeights<HostStaging>, LoadError> {
    GpuWeights::from_dir(dir, HostStaging).map_err(|e| LoadError::NoCheckpoint {
        dir: dir.to_path_buf(),
        why: e.to_string(),
    })
}

/// The largest `weight_bytes[ix]` any of `tapes` declares.
fn declared(tapes: &[CudaTape], ix: usize) -> u64 {
    (tapes.iter())
        .map(|t| t.weight_bytes.get(ix).copied().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// Check every entry of `table` against `weights`' header and `tapes`' declared sizes, reading no
/// tensor's bytes: the table is as long as every tape's `weight_bytes`; each entry's tensors are
/// present, of an accepted dtype and of exactly the shape it needs; its packed size is the largest
/// any tape declares for its index; and an index with no entry declares 0.
pub fn check<A: DeviceAllocator>(
    table: &[Option<CudaWeight>],
    tapes: &[CudaTape],
    weights: &GpuWeights<A>,
) -> Result<(), LoadError> {
    for (bucket, tape) in tapes.iter().enumerate() {
        if tape.weight_bytes.len() != table.len() {
            return Err(LoadError::TableLength {
                table: table.len(),
                tape: tape.weight_bytes.len(),
                bucket,
            });
        }
    }
    for (ix, entry) in table.iter().enumerate() {
        let declared = declared(tapes, ix);
        let Some(entry) = entry else {
            if declared != 0 {
                return Err(LoadError::Bytes {
                    ix,
                    tensor: None,
                    packed: 0,
                    declared,
                });
            }
            continue;
        };
        check_tensors(ix, entry, weights)?;
        let packed = entry.packed_bytes().unwrap_or(u64::MAX);
        if packed != declared {
            return Err(LoadError::Bytes {
                ix,
                tensor: Some(entry.lead()),
                packed,
                declared,
            });
        }
    }
    Ok(())
}

/// The host bytes of weight `ix`, its entry `entry`, taken out of `weights`: a bf16 tensor's bytes
/// as they are on disk, or a GDN bundle packed as `step::gdn_bundle` lays it out, every piece
/// widened to f32 and every gap between regions `+0.0`. [`check`] has passed for the entry; the
/// checks are repeated, so a direct caller gets the same refusals.
pub fn stage<A: DeviceAllocator>(
    ix: usize,
    entry: &CudaWeight,
    weights: &mut GpuWeights<A>,
) -> Result<Vec<u8>, LoadError> {
    check_tensors(ix, entry, weights)?;
    let missing = |tensor: &str| LoadError::Missing {
        ix,
        tensor: Box::leak(tensor.to_owned().into_boxed_str()),
    };
    let overflow = |tensor| LoadError::Bytes {
        ix,
        tensor: Some(tensor),
        packed: u64::MAX,
        declared: 0,
    };
    // A tensor's bytes, taken out of the reader (it hands each out once).
    let mut take = |tensor: &str| {
        (weights.take_to_cpu_bytes(tensor))
            .map(|(bytes, dtype, _)| (bytes, dtype))
            .map_err(|_| missing(tensor))
    };
    match *entry {
        CudaWeight::Bf16 { tensor, .. } => Ok(take(tensor)?.0),
        CudaWeight::GdnBundle {
            conv1d,
            a_log,
            dt_bias,
            norm,
            conv_dim,
            conv_kernel,
            nv,
            head_v,
        } => {
            let layout =
                gdn_bundle(conv_dim, conv_kernel, nv, head_v).ok_or_else(|| overflow(conv1d))?;
            let mut packed = vec![0f32; layout.bytes as usize / 4];
            for (tensor, at) in [
                (conv1d, layout.conv1d),
                (a_log, layout.a_log),
                (dt_bias, layout.dt_bias),
                (norm, layout.norm),
            ] {
                // `[conv_dim, 1, K]` flattened is row `c` channel `c`'s taps, oldest first: the
                // bundle's `[conv_dim, K]`.
                let (bytes, dtype) = take(tensor)?;
                let values = widen(&bytes, dtype);
                packed[at as usize / 4..][..values.len()].copy_from_slice(&values);
            }
            Ok(packed.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        CudaWeight::MoeBundle {
            router,
            gate_up,
            down,
            ..
        } => {
            // `check_tensors` checked each piece is bf16 of its shape, so the three are the
            // bundle's regions in order, back to back with no gap (`step::moe_bundle`).
            let mut packed = Vec::new();
            for tensor in [router, gate_up, down] {
                packed.extend_from_slice(&take(tensor)?.0);
            }
            Ok(packed)
        }
        CudaWeight::ExpertBank { gate_up, down, .. } => {
            // As the bundle's, with no router: `step::expert_bank`'s two regions back to back.
            let mut packed = Vec::new();
            for tensor in [gate_up, down] {
                packed.extend_from_slice(&take(tensor)?.0);
            }
            Ok(packed)
        }
        CudaWeight::Fp8 {
            weight,
            scale,
            shape,
        } => {
            let layout = fp8_layout(shape).ok_or_else(|| overflow(weight))?;
            let mut packed = take(weight)?.0;
            packed.resize(layout.scale as usize, 0);
            let (bytes, dtype) = take(scale)?;
            packed.extend(widen(&bytes, dtype).iter().flat_map(|v| v.to_le_bytes()));
            Ok(packed)
        }
        CudaWeight::Fp8MoeBundle {
            router,
            experts_at,
            experts,
            hidden,
            inter,
        } => {
            let layout =
                fp8_moe_layout(experts, hidden, inter, true).ok_or_else(|| overflow(router))?;
            let mut packed = vec![0u8; layout.experts.bytes as usize];
            let bytes = take(router)?.0;
            packed[..bytes.len()].copy_from_slice(&bytes);
            gather_experts(
                &mut packed,
                layout,
                experts_at,
                (experts, hidden, inter),
                take,
            )?;
            Ok(packed)
        }
        CudaWeight::Fp8ExpertBank {
            experts_at,
            experts,
            hidden,
            inter,
        } => {
            let layout = (fp8_moe_layout(experts, hidden, inter, false))
                .ok_or_else(|| overflow(experts_at))?;
            let mut packed = vec![0u8; layout.experts.bytes as usize];
            gather_experts(
                &mut packed,
                layout,
                experts_at,
                (experts, hidden, inter),
                take,
            )?;
            Ok(packed)
        }
    }
}

/// Every expert's fp8 tensors of `experts_at`, taken by `take`, into `packed` at `layout`'s
/// regions: per expert its gate rows then its up rows (`[2·inter, hidden]`), its down rows, the
/// gate's and up's scales (`[2·inter]` f32) and the down's (`[hidden]` f32), the bytes verbatim and
/// the scales widened from bf16.
fn gather_experts(
    packed: &mut [u8],
    layout: crate::step::Fp8MoeBundle,
    experts_at: &str,
    (experts, hidden, inter): (u32, u32, u32),
    mut take: impl FnMut(&str) -> Result<(Vec<u8>, DType), LoadError>,
) -> Result<(), LoadError> {
    let (h, i) = (hidden as usize, inter as usize);
    let at = |base: u64, e: u32, per: usize| base as usize + e as usize * per;
    for e in 0..experts {
        let parts = [
            (
                BundleTensor::ExpertGateW,
                layout.experts.gate_up,
                2 * i * h,
                0,
            ),
            (
                BundleTensor::ExpertUpW,
                layout.experts.gate_up,
                2 * i * h,
                i * h,
            ),
            (BundleTensor::ExpertDownW, layout.experts.down, h * i, 0),
            (
                BundleTensor::ExpertGateS,
                layout.gate_up_scale,
                2 * i * 4,
                0,
            ),
            (
                BundleTensor::ExpertUpS,
                layout.gate_up_scale,
                2 * i * 4,
                i * 4,
            ),
            (BundleTensor::ExpertDownS, layout.down_scale, h * 4, 0),
        ];
        for (which, base, per, within) in parts {
            let (bytes, dtype) = take(&expert_tensor(experts_at, e, which))?;
            let bytes = match dtype {
                DType::Fp8E4m3 => bytes,
                _ => widen(&bytes, dtype)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            };
            let start = at(base, e, per) + within;
            packed[start..start + bytes.len()].copy_from_slice(&bytes);
        }
    }
    Ok(())
}
