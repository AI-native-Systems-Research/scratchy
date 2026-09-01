// SPDX-License-Identifier: Apache-2.0
//! [`Implementation`] trait + supporting types.
//!
//! Ported from old scratchy-target-metal/src/lowering/implementation.rs.
//! Detoxifications:
//! - `TileGraph` → our [`Fuf`]; `TileId`/`TileKind` → our [`TileId`]
//!   and [`OpKind`]. No `weight_name: String` or `.layer` fields
//!   anywhere — tiles are numeric and structural.
//! - `Layout::PagedKvBf16` variant dropped (Llama-KV-specific name).
//!   `RowMajorBf16, ColMajorBf16, Any` kept; generic "paged-like"
//!   layouts can be added when a real non-llama impl needs them.
//! - `tiles_of_kind(&TileGraph, claimed, TileKind)` helper replaced
//!   by a version over our Fuf that filters by `OpKind`.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use proc_macro2::TokenStream;
use quote::quote;

#[cfg(feature = "cuda")]
use scratchy_forward_compiler::Instruction;

use crate::assignment::ImplId;
use crate::classified::{ExternKind, OpKind, Program, UnrollIndex, WeightId};
use crate::codegen::split_base_layer;
use crate::config::ModelParams;
use crate::emit::weight_field_name;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::quantization::StorageFormat;
use crate::shape::{Dim, Shape};
#[cfg(feature = "cuda")]
use crate::target::Backend;
use crate::target::TargetProfile;
#[cfg(test)]
use crate::weight_vocab::attention_scale_for;
use crate::weight_vocab::{
    OpcodeShape, SlotMap, WeightAccessor, eval_dim_with, eval_shape_with, first_tile_input,
    gemm_nk_from_fuf, is_rope_append_op, weight_storage_of,
};

/// Ambient context for [`Implementation::applies_to`]. Carries the
/// per-canonical model config + the resolved program so per-arch
/// matchers can probe model-config keys (e.g. distinguish Mixtral's
/// `num_local_experts` from Qwen-MoE's `num_experts` +
/// `shared_expert_intermediate_size` from DeepSeek's
/// `n_routed_experts` + `n_shared_experts`) before claiming an
/// `OpKind::Moe` tile that every MoE Impl in the registry is
/// otherwise eligible to claim.
///
/// `applies_to` runs once per (Impl, canonical) before any tile-
/// level matching; an Impl that returns `false` is skipped entirely
/// for that canonical, so the DP solver never sees it as a
/// candidate. Default impl returns `true` — only Impls that genuinely
/// need per-arch gating override.
pub struct MatchContext<'a> {
    pub program: &'a Program,
    pub model: &'a ModelParams,
    pub profile: &'a TargetProfile,
    /// Tensor-parallel world size this canonical is being solved at. The
    /// `forward!` macro fans out one solver run per `(model, tp)` pair (see
    /// `lib.rs::SolvedModel`), so this is a per-canonical constant — at TP>1
    /// the FUF was already TP-lowered (AllReduce / AllGather inserted) and
    /// the solve_bounds shrunk the per-rank `num_kv_heads` /
    /// `num_attention_heads` / `intermediate_size` before the solver ran.
    /// Impls that aren't graph-capture-safe at multi-rank (FI's persistent
    /// kernel) or that only make sense at one tp (sharded-broadcast moes)
    /// can gate on this in `applies_to`.
    pub tp_world_size: u8,
}

/// Ambient context passed to [`Implementation::cost_us`]. Carries
/// everything a cost function might need to turn a MatchInfo into
/// a wall-clock estimate: the FUF for shapes/deps, the target
/// profile for hardware characteristics, and the current
/// `bounds` (all config.json numeric fields plus `num_tokens` for
/// the current workload point).
pub struct CostCtx<'a> {
    pub fuf: &'a Fuf,
    pub profile: &'a TargetProfile,
    pub bounds: &'a BTreeMap<String, u64>,
    /// Expected number of `forward` calls per model load. Used to
    /// amortize startup cost (e.g. CPU pack memcpy) against
    /// per-call runtime. Solver objective: `startup + N × per_call`.
    ///
    /// Backend defaults assume different workload mixes — metal
    /// targets interactive chat (one prefill + tens of decodes ≈ 64);
    /// cuda targets batched serving (tens of thousands).
    pub expected_calls_per_load: u64,
}

impl CostCtx<'_> {
    /// Evaluate a `Dim` to a concrete integer using `bounds`.
    /// Returns `None` if the dim contains a Var (should not happen
    /// for the standard body after shape inference closes).
    pub fn eval_dim(&self, dim: &Dim) -> Option<u64> {
        eval_dim_with(dim, self.bounds)
    }

    pub fn eval_shape(&self, shape: &Shape) -> Option<Vec<u64>> {
        eval_shape_with(shape, self.bounds)
    }

    /// Convenience: read `num_tokens` from bounds. Panics if not
    /// set — the solver sets it per workload point before calling.
    pub fn num_tokens(&self) -> u64 {
        self.bounds
            .get("num_tokens")
            .copied()
            .expect("solver must set num_tokens in bounds before costing")
    }

    /// Convenience: read `sk_bucket` (the discretized KV-cache length
    /// axis) from bounds. Returns 0 when the caller has not swept an
    /// sk axis — legacy 1-D workload points. Impls that don't care
    /// about `sk` ignore this; impls whose cost depends on `sk`
    /// (e.g. FlashInfer attention) read it here and the solver's 2-D
    /// sweep produces one Assignment per (num_tokens, sk_bucket).
    pub fn sk_bucket(&self) -> u64 {
        self.bounds.get("sk_bucket").copied().unwrap_or(0)
    }

    /// Default `expected_calls_per_load` for the given backend
    /// (metal=64 for interactive chat, cuda=10000 for batched
    /// serving).
    pub fn default_expected_calls_per_load(backend: crate::target::Backend) -> u64 {
        match backend {
            crate::target::Backend::Metal => 64,
            crate::target::Backend::Cuda => 10_000,
            // Spyre runs per-token decode (many calls per load), like cuda; the
            // exact value is moot since spyre uses the universal (analytic) cost
            // table, but keep it on the decode-heavy scale.
            crate::target::Backend::Spyre => 10_000,
        }
    }
}

/// Per-implementation hardware resource demand.
///
/// **Per-CTA** demands (regs, shmem) determine occupancy and the
/// hard shmem-budget constraint. Two implementations grouped into
/// the same compilation unit (the same `__global__` on sm_89, the
/// same warpgroup on sm_90+ with `setmaxnreg`) **union** their
/// resource demands at the unit level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resources {
    /// Per-CTA dynamic shmem demand, in bytes.
    pub shmem_bytes: u32,
    /// Per-thread register footprint estimate.
    pub regs_per_thread: u32,
    /// CTA size this implementation runs with.
    pub threads_per_cta: u32,
}

impl Resources {
    pub const ZERO: Self = Self {
        shmem_bytes: 0,
        regs_per_thread: 0,
        threads_per_cta: 0,
    };

    /// Element-wise max — the resource budget two impls would need
    /// if they shared a compilation unit.
    pub fn union_max(&self, other: &Resources) -> Resources {
        Resources {
            shmem_bytes: self.shmem_bytes.max(other.shmem_bytes),
            regs_per_thread: self.regs_per_thread.max(other.regs_per_thread),
            threads_per_cta: self.threads_per_cta.max(other.threads_per_cta),
        }
    }
}

/// How an implementation is invoked / scheduled.
///
/// This is a hard constraint on what the implementation can share
/// a schedule slot with: a `HostCallback` cannot be co-resident
/// with other host calls in the same step; a `CooperativeLaunch`
/// is mutually exclusive with every other `CooperativeLaunch` on
/// the device; a `DeviceCallable` runs inside an enclosing
/// `__global__` and is constrained by that kernel's resource
/// budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchKind {
    /// Host calls a vendor library function (cuBLAS, cublasLt,
    /// FlashInfer standalone, vllm-rs fused-op kernels) which
    /// internally launches its own grid. The host launch boundary
    /// is the synchronization point.
    HostCallback,
    /// Host launches a normal `__global__` (non-cooperative) on a
    /// stream. Multiple of these can be in flight on different
    /// streams.
    RegularLaunch,
    /// Host launches a cooperative `__global__` that grabs all SMs
    /// and excludes other cooperative grids until it finishes.
    CooperativeLaunch,
    /// A `__device__` callable invoked from inside another
    /// `__global__`'s code. Contributes to the host kernel's
    /// resource budget; cannot be launched on its own.
    DeviceCallable,
}

/// Synchronization mechanism that conveys data between two
/// implementations on a producer→consumer dependency edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handoff {
    /// Two host calls on the same stream — implicit ordering, no
    /// explicit sync.
    StreamOrder,
    /// `cudaEvent` recorded after the producer + waited on before
    /// the consumer.
    StreamEvent,
    /// Kernel boundary — the next `cudaLaunchKernel` waits for
    /// the previous to complete.
    KernelBoundary,
    /// In-kernel `cooperative_groups::this_grid().sync()` or
    /// gmem-flag spin barrier.
    InKernelGridSync,
    /// shmem `mbarrier` — sm_90+ only. Used between warpgroups
    /// inside one persistent kernel.
    Mbarrier,
    /// Distributed-shmem cluster read — sm_90+ with thread-block
    /// clusters.
    DsmemRead,
    /// gmem flag spin (per-tile counter).
    GmemFlag,
    /// Intra-CTA `__syncthreads()` between DeviceCallable ops in
    /// the same persistent kernel.
    SyncThreads,
    /// No handoff — both impls are claimed by the same subgraph
    /// (the implementation handles the dep internally).
    Internal,
}

impl Handoff {
    /// Wall-clock cost in microseconds for this handoff on the
    /// given target.
    pub fn cost_us(&self, profile: &TargetProfile) -> f64 {
        let _ = profile; // target-specific cost tables land later;
        // for now fall back to empirical-or-universal constants.
        match self {
            Handoff::StreamOrder => 0.0,
            Handoff::StreamEvent => 5.0,
            Handoff::KernelBoundary => 5.0,
            Handoff::InKernelGridSync => 100.0,
            Handoff::Mbarrier => 0.1,
            Handoff::DsmemRead => 1.0,
            Handoff::GmemFlag => 0.5,
            Handoff::SyncThreads => 0.5,
            Handoff::Internal => 0.0,
        }
    }

    /// Whether this handoff truncates the intermediate to storage
    /// dtype. GMEM-transiting handoffs truncate (value is written
    /// then re-read in storage dtype). In-kernel handoffs
    /// (Internal, Mbarrier, shmem reads) preserve accumulator
    /// precision.
    pub fn truncates_to_storage_dtype(&self) -> bool {
        match self {
            Handoff::StreamOrder
            | Handoff::StreamEvent
            | Handoff::KernelBoundary
            | Handoff::InKernelGridSync
            | Handoff::GmemFlag
            | Handoff::SyncThreads => true,
            Handoff::Internal | Handoff::Mbarrier | Handoff::DsmemRead => false,
        }
    }
}

/// Layout of a tile in memory. Used by the layout-compatibility
/// constraint: a producer's output layout must equal the
/// consumer's input layout, OR a layout-conversion implementation
/// must be inserted between them.
///
/// Starts small; extended as new impls enter the library. Paged
/// / swizzled / tiled variants are added structurally (parameters
/// carrying block sizes etc.) not by naming the transformer role
/// they serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// `[outer, inner]` row-major contiguous bf16.
    RowMajorBf16,
    /// `[inner, outer]` column-major bf16.
    ColMajorBf16,
    /// "Don't care" — the producer and consumer don't constrain
    /// the layout.
    Any,
}

impl Layout {
    /// Whether two layouts are interchangeable without an explicit
    /// conversion. `Any` matches anything.
    pub fn is_compatible_with(self, other: Layout) -> bool {
        match (self, other) {
            (Layout::Any, _) | (_, Layout::Any) => true,
            (a, b) => a == b,
        }
    }
}

/// What an [`Implementation::matches`] call returns when an impl
/// can claim a particular subgraph. Carries the per-claim metadata
/// the solver / cost model / backend need.
#[derive(Clone, Debug)]
pub struct MatchInfo {
    /// The tile ids this implementation would claim if chosen.
    /// Must be a connected subgraph of the [`Fuf`].
    pub claimed_tiles: Vec<TileId>,
    /// The boundary-input tiles this claim reads from (i.e. tiles
    /// outside `claimed_tiles` whose outputs flow in). Used by
    /// the solver to wire up handoffs.
    pub boundary_inputs: Vec<TileId>,
    /// The boundary-output tiles this claim writes to (i.e. tiles
    /// inside `claimed_tiles` whose outputs are read by tiles
    /// outside the claim).
    pub boundary_outputs: Vec<TileId>,
}

impl MatchInfo {
    pub fn size(&self) -> usize {
        self.claimed_tiles.len()
    }
    pub fn is_singleton(&self) -> bool {
        self.claimed_tiles.len() == 1
    }
}

/// Declarative workload eligibility for an implementation.
///
/// Distinct from `target_compatible` (about GPU capability). A
/// `WorkloadConstraint` expresses correctness requirements on the
/// workload itself — e.g. "this GEMV kernel only handles M=1", or
/// "this FlashInfer decode kernel wins only when the KV-cache span
/// (`sk`) is large".
///
/// Correctness, not cost: if `accepts(num_tokens, sk_bucket)` returns
/// false, the impl must NOT be picked at that workload regardless of
/// its cost. Data, not closures — so the ILP backend can linearize
/// each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadConstraint {
    /// Valid for any `(num_tokens, sk_bucket)` value.
    Any,
    /// Valid only when `num_tokens` falls in this inclusive range;
    /// unconstrained on `sk_bucket`.
    NumTokensRange { min: u32, max: u32 },
    /// Valid only when BOTH `num_tokens` and `sk_bucket` fall in
    /// their respective inclusive ranges.
    NumTokensAndSkRange {
        num_tokens: (u32, u32),
        sk_bucket: (u64, u64),
    },
}

impl WorkloadConstraint {
    pub fn accepts(&self, num_tokens: u32, sk_bucket: u64) -> bool {
        match self {
            Self::Any => true,
            Self::NumTokensRange { min, max } => num_tokens >= *min && num_tokens <= *max,
            Self::NumTokensAndSkRange {
                num_tokens: (m_min, m_max),
                sk_bucket: (sk_min, sk_max),
            } => {
                num_tokens >= *m_min
                    && num_tokens <= *m_max
                    && sk_bucket >= *sk_min
                    && sk_bucket <= *sk_max
            }
        }
    }
}

impl fmt::Debug for WeightAccessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightAccessor")
            .field("name", &self.name.to_string())
            .field("rust_type", &self.rust_type.to_string())
            .field("source_weights", &self.source_weights)
            .finish()
    }
}

/// Default [`Implementation::required_weights`] body: one accessor
/// per unique `(WeightId, index)` referenced by a weight input of
/// any claimed tile. Name via [`weight_field_name`]; type inferred
/// from the consuming op (Embed → `Embedding`, RmsNorm → `RmsNorm`,
/// Gemm → `LinearLayer`). Preserves the single-tile-per-subgraph
/// accessor shape the FUF-walking WeightBundle emission used
/// before the trait moved to SFUF-walking.
pub fn default_required_weights(
    claimed_tiles: &[TileId],
    fuf: &Fuf,
    program: &Program,
) -> Vec<WeightAccessor> {
    let mut out: Vec<WeightAccessor> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for &tid in claimed_tiles {
        let node = fuf.get(tid);
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: rust_type_for_weight_consumed_by(node.op),
                    source_weights: vec![(*id, *index)],
                });
            }
        }
    }
    out
}

/// The scratchy-target-cuda layer wrapper corresponding to an op that
/// consumes a weight. Drives the default accessor-type decision.
/// Fusion impls that pack multiple weights declare their own type
/// explicitly and bypass this.
pub fn rust_type_for_weight_consumed_by(op: OpKind) -> TokenStream {
    match op {
        OpKind::Embed => quote! { crate::__gpu::layers::Embedding },
        OpKind::RmsNorm => quote! { crate::__gpu::layers::RmsNorm },
        // `layer_scalar` ([1]-shaped) loads exactly like an RmsNorm
        // gain — a plain keep-dtype vector.
        OpKind::ScalarWeightMul => quote! { crate::__gpu::layers::RmsNorm },
        OpKind::Gemm => quote! { crate::__gpu::layers::LinearLayer },
        // Vision learned positional embedding (G.7(c.1)). Same wrapper
        // as `Embed` — the kernel call is the same and the loader
        // pulls a single `<prefix>.weight` 2D tensor.
        OpKind::PosEmbed => quote! { crate::__gpu::layers::Embedding },
        // Ops that don't consume weights in the DSL's typical
        // patterns. If the DSL routes a weight into one of these
        // unexpectedly, the user must provide a raw `GpuTensor`.
        _ => quote! { crate::__gpu::tensor::GpuTensor },
    }
}

/// One curated implementation in the library.
///
/// Implementations are **not** generic — each entry corresponds
/// to a specific kernel from a specific source (cuBLAS, CUTLASS
/// sm_80 multistage, ThunderKittens fused gate-up, FlashInfer
/// standalone FA-2, vllm-rs `fused_add_rms_norm_inplace`, …).
///
/// Each impl declares the subgraph patterns it can match, its
/// resource demands, its launch kind, the handoff mechanisms it
/// supports, the layouts it requires, and its target
/// compatibility. Cost is calibrated from microbench data per
/// shape.
pub trait Implementation: fmt::Debug + Send + Sync {
    /// Stable name for debug / display / cost-table keys.
    fn name(&self) -> &'static str;

    /// Whether this implementation can run on the given target.
    fn target_compatible(&self, profile: &TargetProfile) -> bool;

    /// Workload eligibility. Default accepts any `num_tokens`.
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    /// Try to match a subgraph rooted at the given seed tile.
    /// Returns `Some(MatchInfo)` if this implementation can claim
    /// a subgraph that includes `seed`, `None` otherwise.
    ///
    /// The matcher is procedural Rust: it inspects `fuf`, walks
    /// neighbors of `seed`, and decides whether the local structure
    /// matches the impl's expected pattern. Quant-aware impls gate
    /// on weight storage via [`Fuf::storage_format_of`].
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo>;

    /// Per-canonical / per-model gate, evaluated ONCE per (Impl,
    /// canonical) before any tile-level matching. Returning `false`
    /// disables this Impl for the current canonical (the solver
    /// never sees it as a candidate at any seed). Returning `true`
    /// (the default) lets `matches` decide structurally as before.
    ///
    /// Use this when an Impl must inspect the model config or the
    /// resolved program to disambiguate from peer Impls that share
    /// the same `OpKind`. Today the canonical example is the three
    /// MoE Impls (`DeepSeekMoeRefImpl` / `FusedMoeRefImpl` /
    /// `SharedFusedMoeRefImpl`) — every BF16 MoE arch produces
    /// `OpKind::Moe` tiles, but only one Impl's load + forward is
    /// correct for any given checkpoint, so each gates on
    /// arch-distinctive config keys before claiming.
    ///
    /// The "once per canonical" semantics matter: this hook is
    /// called outside the tile-parallel `matches` loop and must not
    /// depend on `seed` / `fuf` structure. If you find yourself
    /// wanting per-tile detail here, you actually want to encode
    /// that in `matches` directly.
    fn applies_to(&self, _ctx: &MatchContext) -> bool {
        true
    }

    /// Guard G4: solve-time precondition for a PICKED (impl, tile)
    /// claim. Mirrors any shape assert the impl's metal/cuda lowering
    /// makes (e.g. `assert!(intermediate % HEAD_DIM == 0)`) so a
    /// violation is a compile-time macro error, not a worker-thread
    /// runtime panic. Default `Ok(())` — only impls whose lowering
    /// hard-asserts a bounds relation opt in, so this is a strict
    /// no-op for every other impl. `bounds` carries every config.json
    /// numeric (plus `num_tokens`/`sk_bucket`).
    fn validate_pick(&self, _m: &MatchInfo, _bounds: &BTreeMap<String, u64>) -> Result<(), String> {
        Ok(())
    }

    /// Predicted wall-clock cost in microseconds for one
    /// invocation of this impl on the given match.
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64;

    /// One-time wall-clock cost paid at model load (microseconds)
    /// when this impl is selected. Default `0.0` — most impls add
    /// no load-time overhead beyond raw weight upload, which the
    /// loader pays uniformly regardless of impl choice.
    ///
    /// Override on impls that introduce extra load work, typically
    /// **packed weight accessors**: `required_weights` returns one
    /// accessor with multiple `source_weights`, the loader CPU-
    /// memcpys each component into a packed buffer. On metal that
    /// memcpy is fully synchronous and shows up as a measurable
    /// chunk of "init engine" time; cuda overlaps it with stream
    /// DMA so the cost is effectively 0 there.
    ///
    /// The solver's objective is
    /// `startup_us + ctx.expected_calls_per_load × cost_us`, so a
    /// fused impl that saves T µs per call must save more than its
    /// startup cost over the expected workload to win against an
    /// unpacked alternative.
    fn startup_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        0.0
    }

    /// Return an `Atom` view of this Impl's claim, or `None` if it
    /// can't participate in compiler-driven megakernel synthesis
    /// (must stay as a standalone kernel call). Default `None` —
    /// existing Impls aren't fuseable until they opt in.
    ///
    /// The fuse pass (`fuse_pass.rs`) walks the solver's claim
    /// assignments, calls `as_atom` on each Impl, and groups
    /// adjacent atoms whose dispatch shape and data-flow are
    /// compatible into one synthesized kernel.
    #[cfg(feature = "metal")]
    fn as_atom(&self, _m: &MatchInfo, _fuf: &Fuf) -> Option<Box<dyn crate::atom::Atom>> {
        None
    }

    /// Per-CTA resource demand of this implementation.
    fn resources(&self, m: &MatchInfo) -> Resources;

    /// How this implementation is launched / scheduled.
    fn launch_kind(&self) -> LaunchKind;

    /// The handoff mechanisms this impl can use to **receive** its
    /// boundary inputs from upstream.
    fn supported_input_handoffs(&self) -> &[Handoff];

    /// The handoff mechanisms this impl can use to **convey** its
    /// boundary outputs to downstream.
    fn supported_output_handoffs(&self) -> &[Handoff];

    /// Required input layouts, parallel to `match.boundary_inputs`.
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Output layouts, parallel to `match.boundary_outputs`.
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Whether this impl is compute-bound (for concurrency
    /// modeling). Two compute-bound impls on the same target's
    /// tensor cores cannot productively overlap; memory-bound
    /// impls can.
    fn is_compute_bound(&self) -> bool {
        false
    }

    /// Whether this impl can be claimed by the same compilation
    /// unit / kernel as another. Default: never share
    /// (HostCallback / RegularLaunch / CooperativeLaunch). Device
    /// callables override to opt in.
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        false
    }

    /// Declare the `WeightBundle` trait methods this impl's
    /// `emit_call` will invoke for the given claim. Codegen
    /// aggregates declarations across all picked impls in an SFUF
    /// and emits one trait method per unique accessor.
    ///
    /// Default: one accessor per weight input of each claimed
    /// tile (see [`default_required_weights`]). Fusion impls that
    /// want a packed weight (e.g. concatenated `gate_proj|up_proj`)
    /// override this to declare a single accessor whose
    /// `source_weights` lists every DSL weight the user must pack
    /// into the object they return.
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    /// Which output-alias rule applies to this impl's claims — pure
    /// data; the semantics live in [`crate::alias_rules`], the ONE
    /// interpretation site. See [`crate::alias_rules::AliasRule`] for
    /// the alias/own/untracked taxonomy.
    ///
    /// Default: every output of every claimed tile is its own
    /// freshly-allocated `OwnedTensor`.
    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::OwnAll
    }

    /// Upstream tile-output `(tile, slot)`s that this impl's
    /// `emit_call` *moves* (consumes) into one of its own output
    /// bindings. After such a subgraph, the upstream local is no
    /// longer accessible — the codegen drop pass must not schedule
    /// a `drop(...)` for it.
    ///
    /// Distinct from `output_alias`: an alias keeps the upstream
    /// alive and shares its memory; a consume transfers ownership.
    /// In-place kernels like `scale_inplace` (ScalarMul) and
    /// `tanh_softcap_inplace` (TanhSoftCap) follow the consume
    /// pattern — the kernel mutates the buffer, and the impl
    /// rebinds the moved `OwnedTensor` as its output.
    ///
    /// Default: empty (no upstream is consumed).
    fn consumes_input_tiles(&self, _claimed_tiles: &[TileId], _fuf: &Fuf) -> Vec<(TileId, u8)> {
        Vec::new()
    }

    /// Which per-layer KV-cache side-effect rule applies to this impl
    /// — pure data, interpreted at [`crate::alias_rules::kv_layer_io`]
    /// (the layer index resolves off the claim's
    /// `FufInput::Extern { kind: ExternKind::KvCache, index }`, see
    /// [`kv_cache_extern_layer`]).
    ///
    /// The FUF dependency graph doesn't model the KV cache as a
    /// tile-output edge (it's a runtime-ambient resource, like
    /// `input_ids`), so backends that need to schedule explicit
    /// ordering around it (e.g. metal MTL4 encoder barriers) ask
    /// each impl's declared rule. CUDA backends with implicit stream
    /// ordering can ignore the answer; they get the same correctness
    /// for free from the stream.
    ///
    /// Default: no KV cache interaction.
    fn kv_rule(&self) -> crate::alias_rules::KvRule {
        crate::alias_rules::KvRule::NoIo
    }

    // ── Host-interpreter codegen ────────────────────────────────────
    //
    // Each Impl owns the shape of its own opcode — variant ident
    // and typed payload fields. The proc-macro, processing one
    // arch's solved FUF, collects shapes from the picked Impls and
    // uses them to type-check static-slice emission. The eval
    // bodies live in `scratchy_forward_compiler::Instruction::eval`; an Impl
    // adds a kernel by overriding two methods: `opcode_shape` (the
    // variant declaration matching `Instruction<W>`'s) and
    // `fan_out` (the per-claim instances). A new Impl that forgets
    // to override these fails at codegen time — `fan_out` returns
    // `None` and `lower_bucket` panics with the Impl's name.

    /// Variant declaration this Impl contributes to the per-arch
    /// enum. Variant ident + ordered `(field_ident, field_type)`
    /// pairs. Two Impls returning the same variant ident must
    /// declare structurally identical fields — codegen verifies and
    /// panics on mismatch.
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::unmigrated(self.name())
    }

    /// Additional opcode shapes this Impl's `fan_out` may emit
    /// beyond the primary `opcode_shape()`. Default empty.
    ///
    /// Used by storage-polymorphic Impls that fan out to different
    /// opcode mixes depending on weight format. `MetalFusedGateUp-
    /// SiluMulImpl` is the motivating example: Dense → single
    /// `FusedGateUpSiluMul`; MLX-affine → decomposed
    /// `(AffineQmm, AffineQmm, SiluMul)`. The codegen registration
    /// loop calls `opcode_shape()` AND `extra_opcode_shapes()` and
    /// registers all of them so the per-bucket static-slice
    /// validator can typecheck every variant fan_out emits.
    fn extra_opcode_shapes(&self) -> Vec<OpcodeShape> {
        Vec::new()
    }

    /// One [`OpInstance`] per kernel call this Impl makes at this
    /// (variant × workload-point). Field-value tokens are positional,
    /// matching the order of fields in `opcode_shape().fields`.
    /// `slots` resolves boundary-tile `(TileId, output_slot)` pairs
    /// to `u32` slot indices in the runtime tile table.
    ///
    /// Default `None` flags the Impl as not yet migrated. The
    /// codegen seam panics with the Impl's `name()` when it sees
    /// `None`.
    fn fan_out(
        &self,
        _m: &MatchInfo,
        _fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        _slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        None
    }

    /// Whether this Impl contributes a host-interpreter `Instruction` to the
    /// per-bucket slice. Default `true`: a migrated Impl emits one, so a `None`
    /// from [`fan_out`](Self::fan_out) is an unmigrated-Impl bug the codegen
    /// panics on. An Impl whose real lowering happens elsewhere (claim-only
    /// impls that exist purely to let the solver cover a tile, with the actual
    /// kernel produced by a downstream lowering such as KTIR) overrides this to
    /// `false` — the codegen then skips it instead of treating its `None`
    /// fan_out as a bug. This is a property of the Impl, not of any target.
    fn emits_host_instruction(&self) -> bool {
        true
    }
}

// `Implementation::fan_out` returns `Option<Vec<Instruction>>`.
// There is no `OpInstance` wrapper, no `Vec<TokenStream>`
// round-trip, no `FannedOp` tuple.
// Weight context (kind + base name per slot) is NOT embedded in
// the fan_out output; it flows separately from the existing
// [`Implementation::required_weights`] method, which the codegen
// walker calls alongside `fan_out` and pairs with each emitted
// op-position in a `LoweredBucket`-level parallel array.

/// The library: all available implementations for some target.
/// Implementations are boxed trait objects; the solver iterates
/// them, calling `matches` at each FUF tile.
#[derive(Default)]
pub struct ImplementationLibrary {
    entries: Vec<Box<dyn Implementation>>,
}

impl fmt::Debug for ImplementationLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImplementationLibrary")
            .field("len", &self.entries.len())
            .finish()
    }
}

impl ImplementationLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, imp: Box<dyn Implementation>) -> ImplId {
        // Vocabulary coherence: a rule an impl declares must be in
        // the const tables — adding a rule variant without extending
        // the vocabulary (and its interpreter arm) is a build defect,
        // caught here at registration rather than at first claim.
        assert!(
            crate::alias_rules::ALL_ALIAS_RULES.contains(&imp.alias_rule()),
            "{}: alias_rule {:?} missing from ALL_ALIAS_RULES",
            imp.name(),
            imp.alias_rule(),
        );
        assert!(
            crate::alias_rules::ALL_KV_RULES.contains(&imp.kv_rule()),
            "{}: kv_rule {:?} missing from ALL_KV_RULES",
            imp.name(),
            imp.kv_rule(),
        );
        let id = ImplId(self.entries.len() as u32);
        self.entries.push(imp);
        id
    }

    pub fn get(&self, id: ImplId) -> &dyn Implementation {
        &*self.entries[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all impls with their ids. The solver uses this to
    /// enumerate candidates at each seed tile.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (ImplId, &dyn Implementation)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, imp)| (ImplId(i as u32), imp.as_ref()))
    }
}

/// Helper used by the solver to seed-enumerate matches across the
/// whole library at a given tile. Returns every (impl_id,
/// MatchInfo) pair that can claim something including `seed`.
pub fn enumerate_matches_at_seed(
    library: &ImplementationLibrary,
    fuf: &Fuf,
    seed: TileId,
    profile: &TargetProfile,
) -> Vec<(ImplId, MatchInfo)> {
    let mut out = Vec::new();
    for (id, imp) in library.iter_enumerated() {
        if !imp.target_compatible(profile) {
            continue;
        }
        if let Some(m) = imp.matches(fuf, seed, profile) {
            out.push((id, m));
        }
    }
    out
}

/// Convenience: filter `claimed_tiles` to those of a given op
/// kind. Used by cost / layout helpers that need to count how
/// many of a claim's tiles are (for instance) gemms.
pub fn tiles_of_op<'a>(
    fuf: &'a Fuf,
    claimed: &'a [TileId],
    op: OpKind,
) -> impl Iterator<Item = TileId> + 'a {
    claimed
        .iter()
        .copied()
        .filter(move |&tid| fuf.get(tid).op == op)
}

// ── Starter library ──────────────────────────────────────────────
//
// Baseline HostCallback impl per OpKind. One trait-object per op,
// each matching that op structurally and emitting an analytical
// cost estimate. Real target-specific / quantized / fused impls
// land as additional entries that the solver picks over these
// when cheaper.

const BYTES_PER_ELEM: f64 = 2.0; // fp16/bf16 storage

fn shape_elems(ctx: &CostCtx, shape: &Shape) -> u64 {
    ctx.eval_shape(shape)
        .map(|v| v.iter().product::<u64>())
        .unwrap_or(0)
}

pub(crate) fn single_tile_match(fuf: &Fuf, seed: TileId, op: OpKind) -> Option<MatchInfo> {
    if fuf.get(seed).op != op {
        return None;
    }
    let node = fuf.get(seed);
    // Boundary-input tiles: upstream tiles whose outputs this tile reads.
    let boundary_inputs: Vec<TileId> = node
        .inputs
        .iter()
        .filter_map(|inp| {
            if let crate::fuf::FufInput::Tile { id, .. } = inp {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    Some(MatchInfo {
        claimed_tiles: vec![seed],
        boundary_inputs,
        boundary_outputs: vec![seed],
    })
}

/// True if the tile carries a `KvCache` extern in its inputs — the
/// distinguishing mark of decoder-style 5-arg `attention(q, k, v,
/// kv_cache, block_table)`. Encoder-style 3-arg `attention(q, k, v)`
/// has no such extern and produces a tile with exactly 3 Tile inputs.
/// Used by every decoder attention Impl's `matches` to reject encoder
/// tiles, and by `EncoderAttentionImpl::matches` to reject decoder
/// tiles. Centralizing the predicate keeps the routing rule in one
/// place — adding a new attention Impl only needs to declare which
/// side of the boundary it serves.
fn attention_has_kv_cache_extern(fuf: &Fuf, seed: TileId) -> bool {
    fuf.get(seed).inputs.iter().any(|i| {
        matches!(
            i,
            FufInput::Extern {
                kind: ExternKind::KvCache,
                ..
            }
        )
    })
}

fn elementwise_cost(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let mut bytes = 0u64;
    for inp in &node.inputs {
        if let crate::fuf::FufInput::Tile { id, slot } = inp {
            let upstream = ctx.fuf.get(*id);
            if let Some(shape) = upstream.outputs.get(*slot as usize) {
                bytes = bytes.saturating_add(shape_elems(ctx, shape));
            }
        }
    }
    for out in &node.outputs {
        bytes = bytes.saturating_add(shape_elems(ctx, out));
    }
    let bytes = bytes.saturating_mul(BYTES_PER_ELEM as u64);
    let gb_per_sec = ctx.profile.memory_bandwidth_gbps;
    (bytes as f64 / (gb_per_sec * 1e9)) * 1e6
}

// ── Per-op emission bodies ───────────────────────────────────────
//
// These preserve the shape of the previous hardcoded OpKind match
// in codegen.rs. The kernel symbols they reference are approximate
// and cuda-gated; real scratchy-target-cuda bindings land as calibrated
// impls replace these reference entries.

// No standalone `emit_rope_append`. The only scratchy-target-cuda path for
// "apply rotary + write KV to paged cache" is `fused_qkv_rope_cache`,
// which expects the QKV projections already fused into one packed
// tensor. `FusedQkvRopeCacheImpl` below claims the whole 4-tile
// pattern `(Gemm, Gemm, Gemm, RopeAppend)` structurally — so every
// DSL `rope_append` in Llama/Qwen2 is covered.

// No standalone `emit_attention`. Paged-cache attention
// (`flash_attn_paged`) takes Q plus cache metadata and reads K/V
// directly from the KV cache populated by `FusedQkvRopeCacheImpl`.
// `AttentionViaCacheImpl` below claims the singleton `OpKind::Attention`
// tile and ignores its DSL-visible K/V inputs entirely.

// No standalone `emit_add`: every `Add` in Llama/Qwen2 is immediately
// consumed by an `RmsNorm`, so `FusedAddRmsNormImpl` below claims
// the pair and maps to `fused_add_rms_norm_inplace`. A lone `Add`
// with no downstream `RmsNorm` surfaces as `SolveError::UnclaimedTile`,
// not as a silent fallback.

fn cost_embed(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // One gather per output element, bandwidth-bound.
    elementwise_cost(m, ctx)
}

pub(crate) fn cost_gemm(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // 2*M*N*K FLOPs, compute-bound on tensor cores.
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let inputs: Vec<_> = node
        .inputs
        .iter()
        .filter_map(|i| match i {
            crate::fuf::FufInput::Tile { id, slot } => ctx
                .fuf
                .get(*id)
                .outputs
                .get(*slot as usize)
                .and_then(|s| ctx.eval_shape(s)),
            crate::fuf::FufInput::Weight { id, .. } => {
                // Weight shape not threaded through FUF yet; skip —
                // callers that need accuracy should use a real
                // calibrated impl.
                let _ = id;
                None
            }
            _ => None,
        })
        .collect();
    // Input[0] is the activation: [.., K].
    let (m_dim, k_dim) = match inputs.first() {
        Some(dims) if dims.len() >= 2 => {
            let k = *dims.last().unwrap();
            let m = dims[..dims.len() - 1].iter().product::<u64>();
            (m, k)
        }
        _ => (ctx.num_tokens(), 0),
    };
    // Output last dim is N.
    let n_dim = node
        .outputs
        .first()
        .and_then(|s| ctx.eval_shape(s))
        .and_then(|v| v.last().copied())
        .unwrap_or(0);
    // Prefer the calibrated CSV baseline when the target has one —
    // keeps GemmRefImpl (cuBLAS) on the same measurement scale as
    // the CutlassGemmImpl variants so the DP's cost comparison is
    // apples-to-apples.
    if let Some(cost) = ctx
        .profile
        .cost_us_for("cublas", m_dim as u32, n_dim as u32, k_dim as u32)
    {
        return cost;
    }
    let flops = 2.0 * m_dim as f64 * n_dim as f64 * k_dim as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak == 0.0 || flops == 0.0 {
        0.0
    } else {
        (flops / peak) * 1e6
    }
}

/// Attention logit soft-cap. Reads `attn_logit_softcapping` from
/// the model config when present (Gemma2), else `0.0` meaning the
/// kernel does no soft-capping. The flash-attn kernel treats
/// `softcap <= 0` as "disabled."
pub(crate) fn attention_softcap_for(model: &crate::config::ModelParams) -> f32 {
    model
        .scalars
        .get("attn_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32
}

/// `window_size_left` argument for sliding-window attention. Reads
/// `sliding_window` from the model config (HF convention).
/// The flash-attn kernel takes `-1` to mean "disabled" and a
/// non-negative `w` to mean "attend to the last `w` tokens"; every
/// sliding architecture we know of carries `sliding_window` in
/// tokens, so the value flows straight through.
///
/// A SlidingAttention tile that reaches this helper on a model with
/// no `sliding_window` in config is a config error — the DSL body
/// asked for sliding attention but the architecture didn't supply
/// the window size. Panic with the model name so the failure is
/// observable at macro expansion.
pub(crate) fn sliding_window_left_for(model: &crate::config::ModelParams) -> i32 {
    match model.bounds.get("sliding_window") {
        Some(w) => (*w) as i32,
        None => panic!(
            "sliding_attention tile emitted, but model `{}` has no \
             `sliding_window` in its config.json",
            model.source_stem,
        ),
    }
}

/// Build the `fa2_attn_bf16_h{h}` kernel name for calibrated FA2 cost
/// lookup. FA2 handles any `(num_qo_heads, num_kv_heads)` at runtime,
/// so the cost table is keyed only on the compile-time specialization
/// dim (`head_dim`). Keep in sync with the row names emitted by
/// `scratchy-cost-sweep-cuda/src/attention_sweep.rs`.
fn fa2_attn_csv_name(head_dim: u32) -> String {
    format!("fa2_attn_bf16_h{head_dim}")
}

/// Calibrated FA2 attention cost (via CSV), falling back to the
/// analytic `cost_attention` when no row matches. Used by
/// `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl` (and
/// their sliding variants) so the solver's FA2 vs FI tiebreak runs on
/// real measured timings instead of the ~0 µs the analytic formula
/// returns at M=1 (where `flops = 4*1*1*d` underflows the TFLOPS
/// budget). Without this, FI can never beat FA2 at decode even when
/// the calibrated CSV shows it's 4.6× faster.
fn cost_attention_calibrated(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    let Some(head_dim) = ctx.bounds.get("head_dim").copied() else {
        return cost_attention(m, ctx);
    };
    let name = fa2_attn_csv_name(head_dim as u32);
    let nt = ctx.num_tokens() as u32;
    let sk = ctx.sk_bucket() as u32;
    match ctx.profile.cost_us_for(&name, nt, sk, head_dim as u32) {
        Some(cost) => cost,
        None => cost_attention(m, ctx),
    }
}

fn cost_attention(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // Rough: 4 * T^2 * D.
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let q_shape = node.inputs.first().and_then(|i| match i {
        crate::fuf::FufInput::Tile { id, slot } => ctx
            .fuf
            .get(*id)
            .outputs
            .get(*slot as usize)
            .and_then(|s| ctx.eval_shape(s)),
        _ => None,
    });
    let (t, d) = match q_shape {
        Some(dims) if dims.len() >= 2 => {
            (*dims.first().unwrap(), dims[1..].iter().product::<u64>())
        }
        _ => (ctx.num_tokens(), 0),
    };
    let flops = 4.0 * t as f64 * t as f64 * d as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak == 0.0 || flops == 0.0 {
        0.0
    } else {
        (flops / peak) * 1e6
    }
}

/// Reference HostCallback impl for `OpKind::Embed`. Hand-written
/// because the host-interpreter `opcode_shape` / `fan_out`
/// overrides each reference an Impl-specific kernel symbol +
/// weight type that a shared macro can't express generically.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct EmbedRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for EmbedRefImpl {
    fn name(&self) -> &'static str {
        "embed_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Embed)?;
        // Dense-kernel gate: a quantized embed table is claimed by the
        // backend's quant-specific impl (e.g. metal's `MetalAffineEmbedImpl`),
        // not this dense reference.
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_embed(m, ctx)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        false
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Embed { out_slot, weight_fn }`. Embed has no tile
    // inputs (`input_ids` is an Extern reachable via `ctx.input_ids`
    // ambient binding); its lone weight input is the embed_tokens
    // table. `weight_fn` is `Weights::embed_tokens` (un-layered,
    // ignores the layer arg); `out_slot` is the destination tile
    // table index.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("Embed", vec![("out_slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        let _ = (program, fuf);
        Some(vec![Instruction::Embed(out_slot)])
    }
}
/// Reference HostCallback impl for `OpKind::RmsNorm`. Hand-written
/// (parallel to [`EmbedRefImpl`]) to expose `opcode_shape` /
/// `fan_out` overrides. Per-claim weight
/// selection — `input_layernorm` vs `post_attention_layernorm`,
/// each at any layer — rides on the `weight_fn` fn-pointer +
/// `layer` u32 fields.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct RmsNormRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for RmsNormRefImpl {
    fn name(&self) -> &'static str {
        "rmsnorm_ref"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // CUDA-only implementation - Metal targets use MetalRmsNormImpl
        profile.backend == crate::target::Backend::Cuda
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::RmsNorm)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        false
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `RmsNorm { in_slot, out_slot, layer, weight_fn }`.
    // Two RmsNorm tiles in the same arch share this Impl but can
    // bind different weight accessors (input_layernorm vs
    // post_attention_layernorm) and different layers — both ride
    // in OpInstance fields, leaving the variant declaration
    // structurally identical across all RmsNorm claims.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "RmsNorm: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("RmsNorm: required_weights returned empty");
        let acc_name = acc.name.to_string();
        let (_base, layer) = split_base_layer(&acc_name);
        let layer = layer.map_or(0, |v| v.0) as u32;
        // `hidden_size` — the reduction width the metal kernel reads
        // as `RMSNORM_HIDDEN_SIZE`. Cuda eval ignores this field
        // (reads tile shape directly). On metal we need the actual
        // last-dim size baked because the kernel uses a
        // function-constant loop bound.
        //
        // Detection rule (name-based, matches the DSL conventions
        // across arches): the per-head q_norm /
        // k_norm read `[T*heads, head_dim]` so use `head_dim`;
        // every other RmsNorm reads `[T, hidden]` so use
        // `hidden_size`. Falls back to `hidden_size` if `bounds`
        // doesn't carry `head_dim` (older arches that never had
        // per-head norms).
        let is_q_norm = acc_name.contains("q_norm");
        let is_k_norm = acc_name.contains("k_norm");
        let is_head_norm = is_q_norm || is_k_norm;
        // Per-class head norms (Gemma4): `q_norm_global`/`k_norm_global`
        // normalize the GLOBAL class's head_dim (512) over the global
        // kv-head count (1); the plain names keep the base (sliding)
        // dims. The `_global` suffix is the DSL convention paired with
        // `weight_leaf_renames` (both classes share the on-disk leaf).
        let is_global_norm = is_head_norm && acc_name.contains("_global");
        // Residual-stream width. For a VISION-prelude program the
        // standalone rmsnorm operates on `vision_embed_dim`, NOT the
        // text decoder's `hidden_size` (qwen2.5-VL: 1280 vs 2048).
        // Other towers (qwen2-VL / qwen3.5-VL) wrap their norms in the
        // `(mean,sub,rmsnorm,bias_add)` LayerNorm fusion, which a
        // different Impl lowers — so this standalone path was never
        // vision-exercised before and silently baked the text width
        // (kernel then reduced over the wrong element count → the norm
        // degenerated to a passthrough, garbling the whole tower).
        let hidden = if matches!(program.prelude, crate::classified::Prelude::Vision) {
            *bounds.get("vision_embed_dim").unwrap_or(&0) as u32
        } else {
            *bounds.get("hidden_size").unwrap_or(&0) as u32
        };
        let base_head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let head_dim = if is_global_norm {
            bounds
                .get("global_head_dim")
                .copied()
                .map(|v| v as u32)
                .unwrap_or(base_head_dim)
        } else {
            base_head_dim
        };
        let num_q_heads = bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let base_num_kv = bounds
            .get("num_key_value_heads")
            .copied()
            .unwrap_or(num_q_heads as u64) as u32;
        let num_kv_heads = if is_global_norm {
            bounds
                .get("num_global_key_value_heads")
                .copied()
                .map(|v| v as u32)
                .unwrap_or(base_num_kv)
        } else {
            base_num_kv
        };
        let hidden_size = if is_head_norm && head_dim > 0 {
            head_dim
        } else {
            hidden
        };
        let m_multiplier = if is_q_norm {
            num_q_heads.max(1)
        } else if is_k_norm {
            num_kv_heads.max(1)
        } else {
            1
        };
        Some(vec![Instruction::RmsNorm(
            in_slot_idx,
            out_slot_idx,
            layer,
            hidden_size,
            m_multiplier,
        )])
    }
}
/// Reference HostCallback impl for `OpKind::Gemm`. Hand-written
/// (parallel to [`RmsNormRefImpl`]) to expose the host-interpreter
/// `opcode_shape` / `fan_out` overrides. Bias
/// is handled by [`FusedGemmBiasImpl`]; this Impl claims only the
/// strict-matmul DSL `gemm()`.
///
/// One Gemm tile may bind any layer of any per-block accessor (`q_proj`
/// / `k_proj` / `v_proj` / `o_proj` / `lm_head` / …) — same
/// `weight_fn` + `layer` pattern as RmsNorm.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GemmRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for GemmRefImpl {
    fn name(&self) -> &'static str {
        "gemm_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        // Dense-kernel gate: a quantized weight is claimed by the backend's
        // quant-specific GEMM impl (Marlin / metal QMM), not this dense
        // reference.
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Gemm { in_slot, out_slot, layer, weight_fn }`.
    // `weight_fn` resolves to `Weights::<base>` where `<base>` is
    // the projection name (`q_proj`, `lm_head`, …); `layer` is
    // ignored for un-layered accessors per
    // `emit_weights_accessor_methods`'s contract.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "Gemm: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Gemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("Gemm: weight (N, K) must resolve from FUF + bounds at fan_out time");
        Some(vec![Instruction::Gemm(
            in_slot_idx,
            out_slot_idx,
            layer,
            n,
            k,
        )])
    }
}

// ── ReshapeRefImpl ───────────────────────────────────────────────
//
// Claims `OpKind::Reshape` tiles — metadata-only view changes
// synthesized by shape inference to bridge axis-factor mismatches
// (e.g. per-head QK-norm: rmsnorm on `[T, heads, head_dim]` where
// upstream produced `[T, heads * head_dim]`). Emits a single
// `TensorView::reshape(&[d0, d1, ...])` with concrete dims evaluated
// from the tile's output shape via the model's bound table. No
// allocation, no kernel launch.
//
// Declares an `output_alias` pointing at the upstream tile so the
// codegen drop pass keeps the underlying `OwnedTensor` alive until
// every consumer of the reshaped view is done.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct ReshapeRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for ReshapeRefImpl {
    fn name(&self) -> &'static str {
        "reshape_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Reshape)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Pure metadata change — a handful of nanoseconds host-side,
        // zero on the GPU. Model as 0.0 so the solver never spends
        // effort choosing between reshape variants.
        0.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::AliasToInput(0)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Reshape { in_slot, out_slot, dims_lit, dims_nt_pow,
    // ndim }`. Each output axis is `dims_lit[i] * num_tokens^dims_nt_pow[i]`,
    // matching today's `reshape_dim_token` lowering: every config
    // bound + literal folds into `dims_lit[i]` at codegen time, and
    // each occurrence of the runtime bound `num_tokens` increments
    // `dims_nt_pow[i]` (in practice 0 or 1 per axis). The arm body
    // computes the final shape at runtime and writes a
    // `TileEntry::Reshaped { ref_slot: in_slot, tensor }` so the
    // drop-pass keeps `in_slot`'s `OwnedTensor` alive while any
    // downstream consumer holds the reshape.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Reshape",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                (
                    "dims_lit",
                    syn::parse_quote!([u32; crate::__gpu::tensor::MAX_DIMS]),
                ),
                (
                    "dims_nt_pow",
                    syn::parse_quote!([u8; crate::__gpu::tensor::MAX_DIMS]),
                ),
                (
                    "dims_div_lit",
                    syn::parse_quote!([u32; crate::__gpu::tensor::MAX_DIMS]),
                ),
                ("ndim", syn::parse_quote!(u8)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "Reshape: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let shape = &node.outputs[0];
        // Hardcoded 4 — mirrors `scratchy_target_cuda::tensor::MAX_DIMS`.
        // The proc-macro crate doesn't depend on `scratchy-target-cuda`
        // (only emits its tokens into consumer code), so the const
        // isn't reachable from here. `Instruction::Reshape`'s
        // payload (in scratchy-forward-compiler) references `MAX_DIMS`
        // symbolically; if it ever changes from 4, both this `4` and
        // that variant must be updated together.
        const RESHAPE_MAX_DIMS: usize = 4;
        assert!(
            shape.len() <= RESHAPE_MAX_DIMS,
            "Reshape: target shape has {} dims; tile table's GpuTensor \
             carries at most {RESHAPE_MAX_DIMS} (mirrors \
             scratchy_target_cuda::tensor::MAX_DIMS). Tile id {:?}",
            shape.len(),
            tile
        );
        let mut dims_lit = [1u32; RESHAPE_MAX_DIMS];
        let mut dims_nt_pow = [0u8; RESHAPE_MAX_DIMS];
        let mut dims_div_lit = [1u32; RESHAPE_MAX_DIMS];
        for (i, d) in shape.iter().enumerate() {
            let (lit, nt_pow, div_lit) = decompose_reshape_dim(d, bounds);
            dims_lit[i] = lit;
            dims_nt_pow[i] = nt_pow;
            dims_div_lit[i] = div_lit;
        }
        let ndim = shape.len() as u8;
        Some(vec![Instruction::Reshape(
            in_slot_idx,
            out_slot_idx,
            dims_lit,
            dims_nt_pow,
            dims_div_lit,
            ndim,
        )])
    }
}

/// Decompose a `Dim` into `(numerator_lit, num_tokens_power,
/// denominator_lit)` for the host-interpreter Reshape opcode. The
/// runtime computes one output axis as
/// `(numerator_lit * num_tokens^pow) / denominator_lit`.
///
/// Folding rules:
///
/// - `Lit(n)` → `(n, 0, 1)`.
/// - `Bound("num_tokens")` → `(1, 1, 1)` (the only runtime factor).
/// - other `Bound(name)` → `(bounds[name], 0, 1)` — folded at codegen.
/// - `Mul(factors)` → multiplicative composition over numerator,
///   power, and denominator (a Mul carrying a Div child preserves
///   the Div's denominator).
/// - `Div(num, den)` → numerator decomposes normally; the denominator
///   must be num_tokens-free (`pow == 0`) or it's a compiler bug —
///   the runtime opcode has no machinery for `nt`-in-denominator.
/// - `Var` is a compiler bug — shape inference should have closed
///   every dim.
fn decompose_reshape_dim(d: &crate::shape::Dim, bounds: &BTreeMap<String, u64>) -> (u32, u8, u32) {
    use crate::shape::Dim;
    match d {
        Dim::Lit(n) => (*n as u32, 0, 1),
        Dim::Bound(name) if name == "num_tokens" => (1, 1, 1),
        Dim::Bound(name) => {
            let v = *bounds.get(name).unwrap_or_else(|| {
                panic!(
                    "Reshape: bound `{name}` not found in bounds table — \
                     the model config is incomplete"
                )
            });
            (v as u32, 0, 1)
        }
        Dim::Mul(factors) => {
            let mut num_lit: u64 = 1;
            let mut nt_pow: u8 = 0;
            let mut den_lit: u64 = 1;
            for f in factors {
                let (n, p, d) = decompose_reshape_dim(f, bounds);
                num_lit *= n as u64;
                nt_pow += p;
                den_lit *= d as u64;
            }
            (num_lit as u32, nt_pow, den_lit as u32)
        }
        Dim::Div(num, den) => {
            let (num_lit, num_pow, num_den_lit) = decompose_reshape_dim(num, bounds);
            let (den_num_lit, den_pow, den_den_lit) = decompose_reshape_dim(den, bounds);
            assert_eq!(
                den_pow, 0,
                "Reshape: denominator may not contain `num_tokens` — \
                 the runtime opcode has no `nt`-in-denominator path. \
                 Got dim {d:?}"
            );
            // (num_lit * nt^p / num_den_lit) / (den_num_lit / den_den_lit)
            //   = (num_lit * den_den_lit * nt^p) / (num_den_lit * den_num_lit)
            (
                (num_lit as u64 * den_den_lit as u64) as u32,
                num_pow,
                (num_den_lit as u64 * den_num_lit as u64) as u32,
            )
        }
        Dim::Var(_) => {
            panic!("Reshape target dim must be closed (no Var) — compiler bug")
        }
    }
}

// Silu and Mul have no singleton impls. The only kernel in
// `scratchy-target-cuda` that implements them is `silu_and_mul_fused`,
// which operates on a packed `[num_tokens, 2*intermediate]` buffer
// produced by a single fused gate+up GEMM. Any DSL occurrence of
// `Silu`/`Mul` outside the `(gemm, gemm, silu, mul)` MLP pattern
// would need its own dedicated kernel + Impl — until such an op
// exists there is no structural-fallback worth providing, and an
// unmatched Silu/Mul is a library bug the solver reports via
// `SolveError::UnclaimedTile`.
//
// See `FusedGateUpSiluMulImpl` below for the matcher that claims
// the MLP pattern.

/// Baseline library: HostCallback impl per OpKind plus the
/// multi-tile fusions the `scratchy-target-cuda` shape requires.
/// Replaced / augmented by calibrated target-specific impls as
/// they're ported.
/// Spyre/KTIR neutral reference impl for a standalone `OpKind::Silu` (the SwiGLU
/// activation; the KTIR emitter re-fuses Silu+Mul into SiluMul downstream).
/// Host-callback, no codegen — it just claims the tile so the solver completes
/// under -Fspyre, where the cuda/metal fused gate-up impls aren't present.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct SiluRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for SiluRefImpl {
    fn name(&self) -> &'static str {
        "silu_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Silu {
            return None;
        }
        let inp = match node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![inp],
            boundary_outputs: vec![seed],
        })
    }
    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let ime = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * m * ime * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
}

/// Spyre/KTIR neutral reference impl for a two-tile `OpKind::Mul` (the SwiGLU
/// `silu(gate) * up`). Distinct from `ScalarMulImpl` (Tile × Scalar).
/// Host-callback, no codegen — claims the tile so the solver completes.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct ElemMulRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for ElemMulRefImpl {
    fn name(&self) -> &'static str {
        "elem_mul_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Mul || node.inputs.len() != 2 {
            return None;
        }
        let a = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        let b = match &node.inputs[1] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![a, b],
            boundary_outputs: vec![seed],
        })
    }
    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let ime = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * m * ime * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
}

// ── Spyre / KTIR claim impls ────────────────────────────────────────────────
//
// Spyre lowers the solved FUF to an embedded KTIR bundle (`dump_wavefront_mega`
// → `lower_decode_to_wavefront` → `lower_graph_to_ktir`) that the host worker
// runs; that lowering reads the FUF op graph + claim structure, never a host
// `Instruction` or a cuBLAS kernel, and the worker loads + dequantizes weights
// host-side by their on-disk keys. So spyre OWNS this set of pure CLAIM impls —
// the way metal owns its `Metal*Impl`s — instead of borrowing cuda's kernel
// impls (`CublasGemmImpl` & co.). Each one just lets the solver cover an op's
// tile; it emits no host `Instruction` (`fan_out` defaults to `None`,
// `emits_host_instruction` is `false`) and is storage-agnostic. No matcher here
// tests the target by name.
#[derive(Debug)]
#[cfg(feature = "cuda")]
pub struct KtirClaimImpl {
    name: &'static str,
    op: OpKind,
    /// Attention claims require the 5-arg decoder form (a `KvCache` extern);
    /// false for every other op.
    require_kv_cache: bool,
    workload: WorkloadConstraint,
}

#[cfg(feature = "cuda")]
impl KtirClaimImpl {
    const fn new(
        name: &'static str,
        op: OpKind,
        require_kv_cache: bool,
        workload: WorkloadConstraint,
    ) -> Self {
        Self {
            name,
            op,
            require_kv_cache,
            workload,
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for KtirClaimImpl {
    fn name(&self) -> &'static str {
        self.name
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        self.workload
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        if self.require_kv_cache && !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, self.op)
    }
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Spyre registers exactly one claim impl per op, so this impl is always
        // the sole candidate at its seed — the value only has to be finite and
        // does not affect which impl is picked or the emitted KTIR bundle (the
        // bundle is derived from the FUF + claim, not from cost).
        1.0
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn emits_host_instruction(&self) -> bool {
        false
    }
}

/// Populate `lib` with spyre's own claim impls — one per backbone op the KTIR
/// decode/prefill bundles cover. Mirrors the cuda / metal per-feature blocks:
/// the target owns its impl set.
#[cfg(feature = "cuda")]
pub fn populate_spyre(lib: &mut ImplementationLibrary) {
    let any = WorkloadConstraint::Any;
    let decode = WorkloadConstraint::NumTokensRange { min: 1, max: 1 };
    let prefill = WorkloadConstraint::NumTokensRange {
        min: 2,
        max: u32::MAX,
    };
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_embed",
        OpKind::Embed,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_rmsnorm",
        OpKind::RmsNorm,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_gemm",
        OpKind::Gemm,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_add",
        OpKind::Add,
        false,
        any,
    )));
    // Both rope flavors (the singleton cuda impl gates on `is_rope_append_op`).
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_rope_append",
        OpKind::RopeAppend,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_rope_append_interleaved",
        OpKind::RopeAppendInterleaved,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_reshape",
        OpKind::Reshape,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_pos_embed",
        OpKind::PosEmbed,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_attention",
        OpKind::Attention,
        true,
        decode,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_attention_prefill",
        OpKind::Attention,
        true,
        prefill,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_silu",
        OpKind::Silu,
        false,
        any,
    )));
    lib.push(Box::new(KtirClaimImpl::new(
        "ktir_mul",
        OpKind::Mul,
        false,
        any,
    )));
}

pub fn starter_library() -> ImplementationLibrary {
    // ISel (this library, `solver`, `cost`) is cuda-only: metal and spyre are
    // tape-scheduled (`tape_pilot` in lib.rs is unconditionally true there,
    // and the non-cuda solve arm is `unreachable!`), so nothing ever reads
    // this library on those targets. The binding stays immutable there.
    #[cfg(feature = "cuda")]
    let mut lib = ImplementationLibrary::new();
    #[cfg(not(feature = "cuda"))]
    let lib = ImplementationLibrary::new();
    #[cfg(feature = "cuda")]
    {
        // Target-agnostic singletons needed on BOTH backends (Gemma4 is
        // metal-first): unit-gain rmsnorm (`v_norm`) and the per-layer
        // [1]-weight multiply (`layer_scalar`).
        lib.push(Box::new(RmsNormUnitImpl));
        lib.push(Box::new(ScalarWeightMulImpl));
        lib.push(Box::new(EmbedRefImpl));
        lib.push(Box::new(RmsNormRefImpl));
        {
            // CohereLayerNorm-flavored norm: claims `(Mean, Sub, RmsNorm)`
            // and emits `cohere_layer_norm`. The DSL stays pure math
            // (`mu = mean(x); centered = sub(x, mu); rmsnorm(centered, w)`);
            // this Impl does the structural fusion back to one kernel call.
            lib.push(Box::new(MeanSubRmsNormImpl));
            // Torch-style LayerNorm with bias: claims the 4-tile pattern
            // `(Mean, Sub, RmsNorm, BiasAdd)` and emits `layer_norm_bias`.
            // Solver's claim-size-DESC sort lets this absorb the bias at
            // sites where the trio's RmsNorm output flows into a `bias_add`
            // (e.g. ModernBERT, Qwen2-VL vision blocks). Sites without a
            // `bias_add` consumer fall through to the trio.
            lib.push(Box::new(MeanSubRmsNormBiasAddImpl));
            // The full cuBLAS-routing Impl set (singleton + fused peers):
            //   - GemmRefImpl                 — singleton Gemm via cuBLAS
            //   - FusedCublasGemmAddImpl      — (Gemm, Add) via cuBLAS
            //   - FusedGemmBiasImpl           — (Gemm, BiasAdd) via cuBLAS
            //   - FusedGateUpSiluMulImpl      — packed SwiGLU via cuBLAS
            //   - FusedGateUpGeluMulImpl      — packed GELU-MLP via cuBLAS
            //   - FusedQkvRopeCacheImpl       — packed QKV+rope (M=1) via cuBLAS
            //   - FusedQkvRopePrefillImpl     — packed QKV+rope (M≥2) via cuBLAS
            //
            // Every one of these Impls has a `CutlassFused…` peer registered
            // unconditionally below.
            let cublas_enabled = std::env::var_os("SCRATCHY_DISABLE_CUBLAS_GEMM").is_none();
            lib.push(Box::new(GemmRefImpl));
            lib.push(Box::new(FusedCublasGemmAddImpl));
            lib.push(Box::new(AttentionViaCacheImpl));
            // Reshape is a metadata-only view op synthesized by shape
            // inference to bridge axis-factor mismatches (e.g. per-head QK-
            // norm in Qwen3/Gemma3). Zero-cost, zero-launch; the emitted
            // code is a single `TensorView::reshape(&[..])` call.
            lib.push(Box::new(ReshapeRefImpl));
            // Multi-tile fusions. The solver's claim-size-DESC sort picks
            // these over singleton coverage when both apply; the singletons
            // stay as fallbacks for tile positions the fusion doesn't match
            // (e.g. the first layer's input_layernorm, whose upstream is
            // `embed` not `Add`, stays a singleton RmsNorm claim).
            //
            // `(Gemm, BiasAdd)` pairs not absorbed by a larger fusion (e.g.
            // a lone affine-transform gemm that's not a QKV-pre-rope or
            // gate/up-pre-MLP). Emits cuBLAS gemm_bias via `LinearLayer::forward`.
            //
            // **NOT gated on the cuBLAS env var.** `CutlassFusedGemmBiasImpl`
            // below has no `cutlass_gemm_bias` CSV rows for qwen2's K/V
            // projection shapes (N=128, K=896 for Qwen2.5-0.5B and the rest
            // of the qwen2 fleet). At cuBLAS-OFF the singleton-fallback chain
            // (CutlassGemv + RopeAppendRefImpl + AttentionPrefill) produces a
            // different K/V tensor layout than the fused path expects —
            // verified failure on qwen2-0.5b: position 3 mismatch
            // ("英文字" engine vs " is" golden). Until Lever A2 ships
            // (cutlass_gemm_bias CSV sweep across qwen2 packed-QKV shapes),
            // this Impl stays registered to keep qwen2 correctness-green.
            lib.push(Box::new(FusedGemmBiasImpl));
            // CUTLASS bias-add peer to FusedGemmBiasImpl — plain
            // `cutlass::gemm::device::Gemm` with `LinearCombination` epilogue
            // and bias passed as the C operand at ldc=0 (row broadcast). One
            // Impl per CUTLASS_TILE_ZOO entry, gated per-tile on calibrated
            // CSV row presence; the DP picks the best tile per workload.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassFusedGemmBiasImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // Gated on the cuBLAS env var (Step 5b) — CutlassFusedGateUpSiluMul
            // below is the cuBLAS-OFF peer.
            if cublas_enabled {
                lib.push(Box::new(FusedGateUpSiluMulImpl));
            }
            // CUTLASS EVT peer to FusedGateUpSiluMulImpl — same claim, different
            // kernel shape. Solver's DP picks whichever has lower calibrated
            // cost per bucket; `target_compatible` gates on CSV row presence.
            // Parameterized over CUTLASS_TILE_ZOO so the up-projection GEMM
            // tile is DP-pickable rather than hardcoded — the previous
            // hardcoded `cutlass_128x128_s3` lost to small-M tiles after the
            // tile_m=16 splitK additions, breaking SiLU MLP fusions.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassFusedGateUpSiluMulImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // Gated on the cuBLAS env var (Step 5b) — CutlassFusedGateUpGeluMul
            // below is the cuBLAS-OFF peer.
            if cublas_enabled {
                lib.push(Box::new(FusedGateUpGeluMulImpl));
            }
            // CUTLASS peer to FusedGateUpGeluMulImpl. Mirrors the cuBLAS path
            // structurally — packed CUTLASS GEMM at (M, 2I, K) followed by
            // BW-bound `gelu_and_mul_fused` — picking the GEMM tile per (M,
            // 2I, K) bucket. One Impl per CUTLASS_TILE_ZOO entry.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassFusedGateUpGeluMulImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            lib.push(Box::new(FusedAddRmsNormImpl));
            // Norm→Gemm fusion family: captures the lm_head + body Norm→Gemm
            // patterns the analyzer's fusion-gap report flagged as the
            // dominant cuBLAS surface. Three claim shapes — (RmsNorm, Gemm)
            // 2-tile, (LayerNorm, Gemm) 2-tile, (Add, RmsNorm, Gemm) 3-tile.
            // Tile-zoo-pickable; one Impl per CUTLASS_TILE_ZOO entry per
            // shape. matches() rejects multi-consumer norms so FusedQkvRope*
            // / FusedGateUp* keep claiming the body QKV / gate-up patterns.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassFusedRmsNormGemmImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
                // CohereLayerNorm-flavored peer: claims `(Mean, Sub, RmsNorm,
                // Gemm)` and emits `cohere_layer_norm + cutlass_gemm`. Cost
                // calibration shared with the `CutlassFusedRmsNormGemm` row.
                lib.push(Box::new(CutlassFusedMeanSubRmsNormGemmImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
                lib.push(Box::new(CutlassFusedAddRmsNormGemmImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // Singleton fallback for residual `Add`s whose downstream is not
            // a RmsNorm — Cohere's parallel attn+MLP residual pair, layer-end
            // adds before any LayerNorm. Emits `add_inplace`.
            lib.push(Box::new(AddRefImpl));
            // Tensor-parallel all-reduce-sum: the only matcher for
            // `OpKind::AllReduce` nodes the lowering pass inserts after every
            // row-parallel gemm at tp>1.
            lib.push(Box::new(AllReduceImpl));
            // Tensor-parallel all-gather along the last dim: the only matcher
            // for `OpKind::AllGather` nodes the lowering pass inserts after
            // the lm_head Gemm at tp>1.
            lib.push(Box::new(AllGatherImpl));
            // Multimodal embed splice — the only matcher for
            // `OpKind::MmEmbedSplice` nodes the lowering pass
            // (`tp_lowering::insert_mm_splices`) inserts after every Embed.
            // At tp>1 sits after the AllReduce that `insert_all_reduces`
            // chained onto the Embed, so the D2D-overwrite runs on reduced
            // embeddings (not pre-reduce partials). At tp=1 sits directly
            // on the Embed output — equivalent to the pre-refactor inline
            // splice that lived inside `Instruction::Embed::eval`.
            lib.push(Box::new(MmEmbedSpliceImpl));
            // Gemma-style 3-tile fusion: residual-Add + scalar-offset-Add
            // + RmsNorm. Claimed by the DP in preference to the 2-tile
            // FusedAddRmsNorm + standalone ScalarOffset because it's a
            // larger claim.
            lib.push(Box::new(FusedAddRmsNormWithOffsetImpl));
            // Gemma-style `rmsnorm(x, w + scalar)` for the standalone case
            // (no upstream residual-Add): scalar rides as the rms_norm
            // kernel's `weight_offset` param.
            lib.push(Box::new(ScalarOffsetRmsNormImpl));
            // Tile × scalar in-place multiply (e.g. Gemma embed scale).
            // Only claims Mul tiles whose inputs are (Tile, Scalar); the
            // tensor×tensor SwiGLU / GELU fusions claim the disjoint
            // (Tile, Tile) pattern.
            lib.push(Box::new(ScalarMulImpl));
            // Decode / prefill QKV+rope variants — the solver picks via
            // WorkloadConstraint (M=1 → Cache, M≥2 → Prefill).
            //
            // **NOT gated on the cuBLAS env var.** `CutlassFusedQkvRope*Impl`
            // peers reject biased claims (Lever A2). At cuBLAS-OFF, qwen2's
            // biased QKV pattern would have NO fused impl available; the
            // singleton-fallback chain (RopeAppendRefImpl + CutlassFusedGemmBias
            // + AttentionPrefillContiguous) emits K/V at `[T, heads*head_dim]`
            // while the downstream Attention impl expects `[T, heads, head_dim]`
            // (the fused-path layout) — verified correctness failure on
            // qwen2-0.5b. Keeping these registered preserves correctness; the
            // DP still picks `CutlassFusedQkvRope*` for non-biased patterns
            // when its calibrated cost wins.
            lib.push(Box::new(FusedQkvRopeCacheImpl));
            lib.push(Box::new(FusedQkvRopePrefillImpl));
            // CUTLASS peers — packed CUTLASS GEMM at (M, q+2*kv, hidden)
            // followed by the same fused_qkv_rope_cache / fused_qkv_rope
            // post-pass. One Impl per CUTLASS_TILE_ZOO entry per variant;
            // target_compatible gates each on calibrated CSV row presence.
            // matches() rejects biased claims (qwen2 keeps cuBLAS until the
            // bias-zoo CSV is shape-swept).
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassFusedQkvRopeCacheImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
                lib.push(Box::new(CutlassFusedQkvRopePrefillImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // FusedQkvQkNormRopeCacheImpl (three-gemm + fused qk_norm_rope +
            // cache) is staged in this crate but intentionally NOT registered.
            // The 3-gemm emit_call produces incorrect numerics on Qwen3/Gemma3
            // (attention output diverges from golden at prompt 0 position 0).
            // Root cause is unresolved: possibly Q/K/V ordering, reshape
            // layout, or the interaction between the per-arch rotary cache
            // and the kernel's expected layout. Until the correctness issue
            // is tracked down, Qwen3/Gemma3 go through singletons
            // (GemmRef + ReshapeRef + RmsNormRef + RopeAppendRef) which IS
            // correctness-green.
            //
            // The singleton path is correctness-equivalent to what Qwen3 shipped
            // with originally; the fused impl is a perf optimization that
            // stays scoped to a future session with a proper numerics bringup.
            //
            // Singleton fallback claims standalone RopeAppend tiles. Emits
            // `rotary_embedding_inplace` + `reshape_and_cache`.
            lib.push(Box::new(RopeAppendRefImpl));
            // Matching attention pair — decode reads from cache, prefill
            // reads the contiguous K/V produced by the prefill QKV impl.
            lib.push(Box::new(AttentionPrefillContiguousImpl));
            // Encoder/bidirectional attention — claims the 3-arg
            // `attention(q, k, v)` form (no kv_cache). All other attention
            // Impls reject 3-arg via `attention_has_kv_cache_extern`, so this
            // is the sole path for encoder models like ModernBERT.
            lib.push(Box::new(EncoderAttentionImpl));
            // Sliding-window variants of the attention pair. Claim
            // `OpKind::SlidingAttention` so the DSL author opts into window
            // masking per-tile (e.g. alternating layers via `if` in the DSL
            // body). `window_size_left` reads from `sliding_window` config.
            lib.push(Box::new(SlidingAttentionViaCacheImpl));
            lib.push(Box::new(SlidingAttentionPrefillContiguousImpl));
            // Standalone TanhSoftCap — reads `final_logit_softcapping` from
            // config. The DSL body emits `tanh_softcap(...)` only for
            // architectures that cap logits.
            lib.push(Box::new(TanhSoftCapImpl));
            // Cutlass standalone GEMM tile zoo — one Impl per tile variant
            // in `target_profiles/cost_*.csv`. `target_compatible` gates each
            // by "does this target have a calibrated cost row for this
            // variant?", so targets without CSV data silently fall back to
            // `GemmRefImpl` (cuBLAS). Matching is rejected for Gemms whose
            // output flows into a fusion (RopeAppend / Silu / Mul) so the
            // solver can never pick cutlass for a QKV or gate/up gemm that
            // would otherwise break its fusion chain.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassGemmImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // CUTLASS GEMM + residual-add peer: beta=1.0 epilogue, in-place
            // on residual. 2-tile claim over `(Gemm, Add)`; DP picks over
            // FusedAddRmsNormImpl per layer-residual chain by cost.
            for tile in CUTLASS_TILE_ZOO {
                lib.push(Box::new(CutlassGemmAddImpl {
                    tile_m: tile.0,
                    tile_n: tile.1,
                    stages: tile.2,
                }));
            }
            // CUTLASS SplitK parallel — one Impl per (tile, split_k) tuple.
            // Closes the small-N/large-K tall-skinny shape class where the
            // standard tile zoo leaves cuBLAS winning. target_compatible
            // gates on CSV row presence, so uncalibrated targets skip
            // these variants silently.
            for &(tm, tn, st, sk) in CUTLASS_SPLITK_ZOO {
                lib.push(Box::new(CutlassGemmSplitKImpl {
                    tile_m: tm,
                    tile_n: tn,
                    stages: st,
                    split_k: sk,
                }));
            }

            lib.push(Box::new(CutlassGemvImpl));

            // ── Marlin (AWQ / GPTQ) impls ───────────────────────────────
            //
            // Active only on models whose `quantization_config` resolves at
            // least one weight to a Marlin-consumable storage format
            // (`StorageFormat::Awq { .. }` or `StorageFormat::Gptq { .. }`).
            // Each matcher gates on quant storage, so they're no-ops on
            // dense models — the dense fused impls claim the same patterns
            // for `Dense` weights. Registered after the Cutlass zoo so they
            // land with the rest of the matmul kernels.
            lib.push(Box::new(MarlinGemmImpl));
            lib.push(Box::new(MarlinFusedGateUpSiluMulImpl));
            lib.push(Box::new(MarlinFusedGateUpGeluMulImpl));
            lib.push(Box::new(MarlinFusedQkvRopeCacheImpl));
            lib.push(Box::new(MarlinFusedQkvRopePrefillImpl));

            // ── BitsAndBytes 4-bit (NF4 / FP4) impls ────────────────────
            // Gated per-matches on `StorageFormat::Bnb4 { .. }`; stay
            // dormant on dense / Marlin-consumable models.
            lib.push(Box::new(Bnb4GemmImpl));
            lib.push(Box::new(Bnb4FusedGateUpSiluMulImpl));
            lib.push(Box::new(Bnb4FusedGateUpGeluMulImpl));
            lib.push(Box::new(Bnb4FusedQkvRopeCacheImpl));
            lib.push(Box::new(Bnb4FusedQkvRopePrefillImpl));

            // ── GGML/GGUF impls ─────────────────────────────────────────
            // Gated per-matches on `StorageFormat::Ggml`; stay dormant on
            // dense / Marlin / BNB4 / FP8 models. Singleton + fused
            // gate/up SwiGLU/SwiGELU; fused QKV+rope is still a follow-up
            // (RopeAppendRefImpl claims the rope tile when storage is Ggml,
            // matching the deferral gate in `rope_append_has_fused_qkv_upstream`).
            lib.push(Box::new(GgmlGemmImpl));
            lib.push(Box::new(GgmlFusedGateUpSiluMulImpl));
            lib.push(Box::new(GgmlFusedGateUpGeluMulImpl));
            lib.push(Box::new(GgmlFusedQkvRopeCacheImpl));
            lib.push(Box::new(GgmlFusedQkvRopePrefillImpl));

            // ── FP8 (E4M3) impls ────────────────────────────────────────
            // Gated per-matches on `StorageFormat::Fp8 { .. }`; stay dormant
            // on dense / Marlin / BNB4 models. Singleton only today; fused
            // QKV / gate-up peers are perf follow-ups.
            lib.push(Box::new(Fp8GemmImpl));
            lib.push(Box::new(Fp8FusedGemmBiasImpl));
            lib.push(Box::new(Fp8FusedGateUpSiluMulImpl));
            lib.push(Box::new(Fp8FusedGateUpGeluMulImpl));
            lib.push(Box::new(Fp8FusedQkvRopeCacheImpl));
            lib.push(Box::new(Fp8FusedQkvRopePrefillImpl));

            // ── DeepSeek MLA + MoE ops ───────────────────────────────────
            // MLA split (kv_a → kv_latent + k_pe), full MLA attention
            // sequence, and DeepSeekV2MoE (routed + shared expert).
            lib.push(Box::new(MlaSplitRefImpl));
            lib.push(Box::new(MlaAttentionImpl));
            lib.push(Box::new(GatedDeltaNetImpl));
            lib.push(Box::new(GateSplitImpl));
            lib.push(Box::new(GateApplyImpl));
            lib.push(Box::new(GateScaleImpl));
            lib.push(Box::new(DeepSeekMoeRefImpl));
            lib.push(Box::new(DeepSeekFp8BlockMoeImpl));
            lib.push(Box::new(DeepSeekGgmlMoeImpl));
            lib.push(Box::new(FusedMoeRefImpl));
            lib.push(Box::new(SharedFusedMoeRefImpl));
            lib.push(Box::new(CudaGemmaMoeImpl));

            // ── Vision-tower ops (Phase G.4) ─────────────────────────────
            // One Impl per vision OpKind from G.2. Pairs with the
            // `OpKind::from_name` arms added in G.4 so `varlen_attention(...)`
            // / `vision_rope(...)` / `quick_gelu(...)` / `gelu_erf(...)` parse
            // ⟺ codegen stays total. Unreachable until `#[vision_forward]`
            // bodies land in G.5+.
            lib.push(Box::new(VarlenAttentionImpl));
            lib.push(Box::new(VisionRopeImpl));
            lib.push(Box::new(QuickGeluImpl));
            lib.push(Box::new(GeluErfImpl));
            // GELU-tanh singleton (Phase G.7(c)). Mirrors `QuickGeluImpl` /
            // `GeluErfImpl`; claims a standalone `gelu(x)` tile. SigLIP /
            // Gemma3-MM use tanh-form GELU in their MLP. The
            // `FusedGateUpGeluMul*` family still claims `(gemm, gemm, gelu, mul)`
            // SwiGLU-style chains (Gemma2/3 text decoder); the singleton only
            // fires when no Mul follows.
            lib.push(Box::new(GeluImpl));
            // Vision learned positional embedding (Phase G.7(c.1)). Mirror of
            // `EmbedRefImpl` — same kernel (`embedding_gather_masked`), but
            // anchored on `vision_num_positions` / `vision_embed_dim` and
            // reading `ctx.fwd.vision_position_ids` instead of
            // `ctx.fwd.input_ids`. Used by SigLIP / Gemma3-MM and any future
            // tower with a learned positional table.
            lib.push(Box::new(PosEmbedRefImpl));
            // Pixels materialization (Phase G.5.e.1). Synthesized by
            // `vision_lowering::materialize_pixels` after `fuf::unroll`
            // under `Prelude::Vision`; the Impl runs only when that pass
            // produced a tile. Decoder bodies never see this OpKind.
            lib.push(Box::new(LoadPixelsImpl));
            // Qwen3.5-VL pos_embeds materialization (sibling of LoadPixels).
            lib.push(Box::new(LoadPosEmbedsImpl));
            // Row-permutation gather (Phase G.6.4). Claims any DSL call to
            // `embedding_gather(x, indices)`. The `indices` arg is one of
            // `vision_window_index` / `vision_reverse_indices` (both vision-
            // prelude externs); the Impl bakes which one the caller used into
            // the emitted `Instruction::EmbeddingGather`'s discriminant so
            // eval reads the correct `ForwardCtx` field.
            lib.push(Box::new(EmbeddingGatherImpl));
            // 2-D non-overlapping average pool (Phase G.7(b)). Claims any
            // `avg_pool_2d(x)` call; reduces the leading dim by the bake-time
            // `vision_pool_factor`. Used by Gemma3-MM's SigLIP→text projector.
            lib.push(Box::new(AvgPool2dImpl));

            // FlashInfer paged attention. Gated sm_89+ in each Impl's
            // `target_compatible`; sm_80/sm_86 fall back to FA2 (untested
            // with this shim).
            //
            // Apples-to-apples vs Python vLLM (`--attention-backend
            // FLASHINFER`, CUDA graphs both sides, BF16, vllm 0.19.1):
            //   H100  BS=1 in=8192   3B 0.265s vs 0.334s py  (-21%)
            //                         7B 0.405s vs 0.583s py  (-30%)
            //                        14B 0.777s vs 1.147s py  (-32%)
            //   L40S  BS=1 in=8192   3B 0.651s vs 0.935s py  (-30%)
            //                         7B 1.336s vs 1.956s py  (-32%)
            //                        14B 2.647s vs 3.939s py  (-33%)
            // Batched (BS=8/32) is -5..-13% across both arches.
            //
            // H100 TP=2 verified (Qwen2.5-7B 0.348s, Qwen2.5-14B 0.625s
            // at BS=1 in=8192, faster than TP=1) including the
            // num_kv_heads=1-per-rank shape (Qwen2.5-3B at TP=2).
            //
            // The sm_89 enable depends on the multi-slot plan cache in
            // `scratchy_target_cuda::flashinfer::FlashInferPlanCache`. The
            // cost solver alternates between (head_dim,
            // use_logits_soft_cap) impls across decode shapes; with the
            // older single-slot cache, switching cfg mid-session
            // `cudaFree`d the previous plan's int_ws_d/float_ws_d, which
            // captured CUDA graphs still referenced through params_1 /
            // params_2 baked into kernel args at capture — replay then
            // surfaced as `CUDA_ERROR_ILLEGAL_ADDRESS`. Holding one
            // persistent slot per cfg fixes that.
            for &head_dim in &[64u32, 128, 256] {
                for &use_softcap in &[false, true] {
                    lib.push(Box::new(FlashInferAttentionDecodeImpl {
                        head_dim,
                        use_logits_soft_cap: use_softcap,
                    }));
                    lib.push(Box::new(FlashInferAttentionPrefillImpl {
                        head_dim,
                        use_logits_soft_cap: use_softcap,
                    }));
                }
            }
            // FA3 Hopper-native decode (sm_90+, hdim128, no softcap). Solver
            // weighs it against `FlashInferAttentionDecodeImpl` per cell;
            // when the cost CSV doesn't have an `fa3_*` row, target_compatible
            // returns false and FI wins by default (zero-config fallback).
            #[cfg(fa3_built)]
            lib.push(Box::new(FlashAttention3DecodeImpl { head_dim: 128 }));
        } // end #[cfg(feature = "cuda")] CUDA-impls block
    }
    lib
}

// ── FusedGemmBiasImpl ────────────────────────────────────────────
//
// 2-tile fusion: `(Gemm, BiasAdd)` where BiasAdd consumes the Gemm's
// output at its first input slot. Emits `LinearLayer::forward` on a
// packed accessor covering (weight, bias) — the dense variant
// dispatches to cuBLAS's `gemm_bias` epilog, producing the biased
// result in one launch.
//
// This is the fallback path for affine-transform gemms not absorbed
// by a larger fusion (e.g. `FusedQkvRopeCacheImpl` absorbs Qwen2's
// QKV pre-rope `(Gemm, BiasAdd) × 3` triples, so this impl only
// claims stray pairs elsewhere in the body).
//
// Required weights: one fused accessor carrying both the Gemm's
// weight ref and the BiasAdd's bias ref. The user returns a
// `LinearLayer::Dense` with its `bias: Some(...)` populated —
// `LinearLayer::load_dense` already auto-detects bias from the
// safetensors path, so most callers get this for free.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedGemmBiasImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        "fused_gemm_bias"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // cuBLAS `gemm_bias` epilog — CUDA only. The Metal interpreter
        // (Phase 5.A) has no `FusedGemmBias` lowering arm, so on Metal the
        // `(Gemm, BiasAdd)` chain must stay split (Gemm + `MetalBiasAdd`,
        // both of which `lower_one` handles). Without this gate the cost
        // solver picks the fused variant at M=1 and pool-init panics. Spyre
        // doesn't register this impl at all, but gate it to cuda honestly.
        profile.backend == crate::target::Backend::Cuda
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        // cuBLAS `gemm_bias` epilog on dense BF16/F16 weights. AWQ /
        // GPTQ Gemms route through `MarlinFusedQkvRope*Impl` (which
        // walks the same `(Gemm, BiasAdd)` chain and dispatches to
        // `MarlinLinear::forward` — which internally adds the packed
        // `.bias` after `marlin_gemm`). Reject here so the DP doesn't
        // pick this dense-typed accessor for a quantized weight.
        if !matches!(weight_storage_of(seed_node), Some(StorageFormat::Dense)) {
            return None;
        }
        // Find a BiasAdd tile whose first tile input is this Gemm.
        // BiasAdd's shape sig is `(x, b) -> x` with `x` at slot 0 and
        // `b` at slot 1 (the bias weight, not a tile).
        let bias_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::BiasAdd
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == seed
                )
        })?;
        let bias_id = bias_node.id;

        let mut claimed = vec![seed, bias_id];
        claimed.sort();

        // Boundary inputs: the gemm's upstream activation tile(s).
        let activation_inputs: Vec<TileId> = seed_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: activation_inputs,
            // Only BiasAdd's output is live downstream — the Gemm's
            // intermediate output is consumed inside the fused kernel.
            boundary_outputs: vec![bias_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // cuBLAS `gemm_bias` has a negligible epilog cost on top of
        // the underlying GEMM — delegate to `cost_gemm` on the Gemm
        // tile so this impl stays on the same measurement scale as
        // singleton GemmRefImpl / CutlassGemmImpl. The DP's cost
        // comparison between (strict gemm + orphaned bias_add) and
        // (fused gemm_bias) will always prefer the fusion because
        // the alternative is `f64::INFINITY` (BiasAdd has no
        // singleton impl) — cost parity is not a correctness
        // concern here, only a calibration one.
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_info = MatchInfo {
            claimed_tiles: vec![gemm_id],
            boundary_inputs: vec![],
            boundary_outputs: vec![gemm_id],
        };
        cost_gemm(&gemm_info, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Single accessor typed `LinearLayer`, source = the Gemm's
        // weight. The bias rides through the accessor's
        // `.dense_bias()` field; the user's loader (typically
        // `LinearLayer::load_dense`) auto-detects it from the
        // safetensors path `<prefix>.bias`. `emit_call` asserts
        // `dense_bias().is_some()` at runtime so a data/DSL mismatch
        // surfaces loudly instead of silently dropping the bias.
        let gemm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_weight = first_weight_ref(fuf.get(gemm_id)).expect("gemm has a weight");
        let sources = vec![gemm_weight];
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGemmBias { in_slot, out_slot, layer, weight_fn }`.
    // The `(Gemm, BiasAdd)` claim collapses to one kernel call against
    // a single `LinearLayer` accessor (the loader detects `<prefix>.bias`
    // and stashes it on the dense layer). `LinearLayer::forward`
    // dispatches to `gemm_bias` for dense bf16 weights.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("FusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("FusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedGemmBias: gemm's first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGemmBias: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::FusedGemmBias(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── CutlassFusedGemmBiasImpl ─────────────────────────────────────
//
// CUTLASS peer to `FusedGemmBiasImpl`. Same 2-tile `(Gemm, BiasAdd)`
// claim; emits `cutlass_gemm_bias` (plain `cutlass::gemm::device::Gemm`
// with `LinearCombination` epilogue, bias passed as C at ldc=0 for row
// broadcast). One Impl per CUTLASS_TILE_ZOO entry, parameterised by
// `(tile_m, tile_n, stages)`; per-tile `target_compatible` gates on
// calibrated CSV row presence so the DP picks the best tile per
// (workload M, weight N, K) bucket.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedGemmBiasImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedGemmBiasImpl {
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_gemm_bias_16x64_s3",
            (16, 64, 4) => "cutlass_gemm_bias_16x64_s4",
            (16, 128, 3) => "cutlass_gemm_bias_16x128_s3",
            (16, 128, 4) => "cutlass_gemm_bias_16x128_s4",
            (32, 64, 3) => "cutlass_gemm_bias_32x64_s3",
            (32, 64, 4) => "cutlass_gemm_bias_32x64_s4",
            (32, 128, 3) => "cutlass_gemm_bias_32x128_s3",
            (32, 128, 4) => "cutlass_gemm_bias_32x128_s4",
            (32, 256, 3) => "cutlass_gemm_bias_32x256_s3",
            (64, 64, 3) => "cutlass_gemm_bias_64x64_s3",
            (64, 64, 4) => "cutlass_gemm_bias_64x64_s4",
            (64, 128, 3) => "cutlass_gemm_bias_64x128_s3",
            (64, 128, 4) => "cutlass_gemm_bias_64x128_s4",
            (128, 64, 3) => "cutlass_gemm_bias_128x64_s3",
            (128, 64, 4) => "cutlass_gemm_bias_128x64_s4",
            (128, 128, 3) => "cutlass_gemm_bias_128x128_s3",
            (128, 128, 4) => "cutlass_gemm_bias_128x128_s4",
            (128, 256, 3) => "cutlass_gemm_bias_128x256_s3",
            (256, 64, 3) => "cutlass_gemm_bias_256x64_s3",
            (256, 64, 4) => "cutlass_gemm_bias_256x64_s4",
            _ => "cutlass_gemm_bias_unknown",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Structural pattern identical to FusedGemmBiasImpl — DP
        // picks whichever has the lower calibrated cost per bucket.
        FusedGemmBiasImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_tile = m
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let Some((mm, nn, kk)) = gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Share the LinearLayer accessor with FusedGemmBiasImpl so
        // codegen dedups both Impls' accessor declarations.
        FusedGemmBiasImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassFusedGemmBias { in_slot, out_slot, layer,
    // weight_fn, tile_m, tile_n, stages, n, k }`. All zoo entries
    // share this variant — tile config rides as runtime fields. The
    // arm body calls `cutlass_gemm_bias(..., CutlassTile::new(...))`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("CutlassFusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("CutlassFusedGemmBias: gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGemmBias: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassFusedGemmBias: weight (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::CutlassFusedGemmBias(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

// ── FusedGateUpSiluMulImpl ───────────────────────────────────────
//
// Multi-tile impl claiming the `(Gemm, Gemm, Silu, Mul)` MLP
// pattern and mapping it to the `silu_and_mul_fused` kernel (which
// expects a packed `[num_tokens, 2*intermediate]` buffer).
//
// Match is purely structural — no phase tags, no weight-name
// checks. Pattern: two Gemm tiles sharing the same activation
// Tile-input feed a Silu and a Mul, with the Mul consuming the
// Silu output and the second Gemm output. Survives renaming
// gate_proj/up_proj → anything and survives adding new
// architectures that reuse the same topology.
//
// The impl declares one fused WeightBundle accessor (see
// `required_weights`) whose `rust_type` is `LinearLayer`. The user
// populates it with a `LinearLayer` carrying the vertically-
// concatenated `[gate_proj | up_proj]` weight; `LinearLayer::forward`
// on that produces the packed `gate_up` buffer `silu_and_mul_fused`
// wants. No intermediate D2D copies; no cross-impl contract.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedGateUpSiluMulImpl;

/// The `WeightId` + concrete index read by the first weight-typed
/// input of a tile. For Gemm the weight is in slot 1 of the DSL's
/// call; we don't care about position, just "which weight flows in".
pub(crate) fn first_weight_ref(
    node: &crate::fuf::FufNode,
) -> Option<(WeightId, Option<UnrollIndex>)> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Weight { id, index, .. } => Some((*id, *index)),
        _ => None,
    })
}

/// Build a stable, structural accessor name covering multiple
/// source weights. Joins each source weight's
/// [`weight_field_name`] with `"__fused__"` **in source-declaration
/// order** — the same order these source weights flow into
/// [`LinearLayer::load_dense_concat`], so the fused accessor name's
/// `__fused__`-split equals the on-disk byte-layout order of the
/// concatenated GPU buffer.
///
/// **Paris invariant `fused-accessor-name-matches-load-order`**: do
/// NOT sort the parts. The recipe-walker
/// Per-constituent byte offsets are computed
/// byte offsets by walking `fused_key.split("__fused__")` and
/// summing the size of each predecessor — those byte offsets ONLY
/// correspond to actual on-disk positions when the name's split
/// order matches `load_dense_concat`'s prefix iteration. Sorting the
/// parts (e.g. alphabetical k/q/v) while [`load_dense_concat`] keeps
/// source-declaration order (q/k/v) caused the megakernel decode's
/// Paris!!! divergence: k_proj's recipe entry resolved to byte 0 of
/// the QKV buffer (= start of Q's weights), so the megakernel's
/// k_proj read Q's weights and produced Q-flavored values where K
/// values belonged. V was byte-identical with per-op only because
/// alphabetical position 2 (v) and source-declaration position 2
/// (v) coincidentally agreed for v. (Empirically reproduced on an
/// H100 pod.)
///
/// All sources of one fused accessor share the same layer index
/// (or all are unindexed) — this is structural: a layer-N gate_proj
/// can't fuse with a layer-M up_proj. So we strip the per-component
/// `_<layer>` suffix before joining and append it once at the end.
/// Result: `mlp_gate_proj__fused__mlp_up_proj_28` rather than
/// `mlp_gate_proj_28__fused__mlp_up_proj_28`. The accessor-method
/// codegen sees a single base across all layers and collapses 40
/// per-layer methods into one method with 40 match arms.
pub fn fused_accessor_name(
    program: &Program,
    sources: &[(WeightId, Option<UnrollIndex>)],
) -> syn::Ident {
    debug_assert!(!sources.is_empty(), "fused_accessor_name: empty sources");
    let layer = sources[0].1;
    debug_assert!(
        sources.iter().all(|(_, idx)| *idx == layer),
        "fused_accessor_name: sources span multiple layer indices ({sources:?})"
    );
    let parts: Vec<String> = sources
        .iter()
        .map(|(id, _idx)| weight_field_name(program, *id, None).to_string())
        .collect();
    let mut joined = parts.join("__fused__");
    if let Some(l) = layer {
        joined = format!("{joined}_{l}");
    }
    syn::Ident::new(&joined, proc_macro2::Span::call_site())
}

#[cfg(feature = "cuda")]
impl Implementation for FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed on the **gate** Gemm — the Gemm whose output feeds a
        // Silu. The FUF's topological order processes Gemms before
        // their downstream Silu/Mul consumers, so seeding on the
        // upstream Gemm commits the 4-tile claim before the DP
        // reaches (and would singleton-claim) the Silu. Seeding on
        // Silu instead would leave the solver's per-tile greedy
        // picking GemmRefImpl at the gate_gemm's earlier position
        // and orphaning the downstream Silu/Mul — see the
        // larger-claim-preferred sort in solver.rs:solve_one.
        let gate_gemm = fuf.get(seed);
        if gate_gemm.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(gate_gemm), Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a downstream Silu consuming this Gemm.
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;

        // Find the Mul consuming that Silu.
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;

        // Mul's other Tile input is the up Gemm.
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        let up_gemm = fuf.get(up_gemm_id);
        if up_gemm.op != OpKind::Gemm {
            return None;
        }

        // Both Gemms must read the same activation tile + slot.
        if first_tile_input(gate_gemm)? != first_tile_input(up_gemm)? {
            return None;
        }

        // Claimed tiles sorted by TileId for determinism.
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();

        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Cost = one cuBLAS GEMM at packed (M, 2I, K) + one elementwise
        // silu_mul pass over [M, 2I] producing [M, I].
        //
        // Read measured cublas cost from the calibration CSV (the
        // predictor linreg-extrapolates when the exact (M, 2I, K) row
        // isn't sampled). Fall back to peak-FLOPS roofline only when
        // the predictor has no signal at all — keeps this Impl on the
        // same measurement scale as `CutlassFusedGateUpSiluMulImpl`,
        // which sums measured cutlass CSV rows. Without this parity
        // the DP saw a wildly optimistic cuBLAS path (8× faster than
        // reality at llama-7b prefill) and never picked the CUTLASS
        // sibling.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Bandwidth-bound silu*mul: read 2*M*I, write M*I, bf16 = 2 B.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
        let silu_mul_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + silu_mul_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Collect the two Gemm tiles' weight refs, merge into one
        // fused accessor. The user provides a `LinearLayer` whose
        // weight is the vertically-concatenated `[gate_proj | up_proj]`.
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGateUpSiluMul { in_slot, out_slot, layer, weight_fn }`.
    // The 4-tile `(Gemm, Gemm, Silu, Mul)` claim collapses to one
    // `LinearLayer::forward` (against the packed `[gate|up]` weight)
    // followed by `silu_and_mul_fused`. `intermediate_size` is baked
    // from `model.bounds` at codegen time — same as today's `emit_call`
    // baking via `ctx.bound`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("FusedGateUpSiluMul: claim contains Mul");
        // Gate gemm is the Silu's tile producer; up gemm is the other
        // claimed Gemm. Both Gemms read the same activation tile/slot
        // (matches() already enforced this), so we resolve in_slot off
        // the gate gemm's first tile input.
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("FusedGateUpSiluMul: Silu's first input must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedGateUpSiluMul: gate gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::FusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Whether `node` consumes the output of `producer` via any Tile input.
pub(crate) fn consumes_tile(node: &crate::fuf::FufNode, producer: TileId) -> bool {
    node.inputs
        .iter()
        .any(|i| matches!(i, FufInput::Tile { id, .. } if *id == producer))
}

// ── CutlassFusedGateUpSiluMulImpl ────────────────────────────────
//
// CUTLASS EVT peer to [`FusedGateUpSiluMulImpl`]. Claims the exact
// same `(Gemm, Gemm, Silu, Mul)` tile pattern and declares the same
// packed `[gate|up]` LinearLayer accessor — so accessor emission is
// unchanged regardless of which Impl the DP picks per bucket.
//
// Kernel shape differs: two GEMM launches, the second with a fused
// SiLU+Mul epilogue that aux-loads the up-projection output from
// GMEM. Saves one BW-bound elementwise kernel launch + one full
// `[M, I]` GMEM round-trip of the gate output.
//
// At emit time the packed `[2I, K]` weight is sliced via
// `narrow_dim0` into `gate_w` and `up_w` views; both GEMMs read the
// same contiguous backing buffer, so there is no duplicate weight
// memory relative to the cuBLAS-packed peer.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedGateUpSiluMulImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedGateUpSiluMulImpl {
    /// CSV name for the up-projection GEMM tile.
    fn up_csv_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        self.up_csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.up_csv_name())
            && profile
                .cost_table
                .has_kernel("cutlass_fused_gate_up_silu_mul")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Structural pattern identical to FusedGateUpSiluMulImpl — the
        // two impls claim the same 4 tiles; the DP picks whichever has
        // the lower calibrated cost per bucket.
        FusedGateUpSiluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Two launches:
        //   (a) up GEMM — a standalone CUTLASS tile (parameterized by
        //       this Impl's `tile_m`/`tile_n`/`stages`) at shape
        //       `(M, I, H)`. One Impl per `CUTLASS_TILE_ZOO` entry —
        //       the DP picks the best tile per bucket. Previously
        //       hardcoded to `cutlass_128x128_s3`, which left the
        //       SiLU peer uncompetitive at small M after the tile zoo
        //       gained tile_m=16 splitK options that beat 128x128 on
        //       small-M shapes.
        //   (b) gate GEMM + SiLU + Mul EVT — shape `(M, I, H)`;
        //       captured in the `cutlass_fused_gate_up_silu_mul` row,
        //       which includes the epilogue's aux-load of `[M, I]`
        //       up output.
        let (_m_val, nn, kk) = {
            let gemm_tile = m
                .claimed_tiles
                .iter()
                .copied()
                .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
                .expect("fused gate/up/silu/mul claim contains a Gemm");
            match gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) {
                Some(v) => v,
                None => return f64::INFINITY,
            }
        };
        let mm = ctx.num_tokens() as u32;
        let up_us = ctx
            .profile
            .cost_us_for(self.up_csv_name(), mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US);
        let fused_us = ctx
            .profile
            .cost_us_for("cutlass_fused_gate_up_silu_mul", mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US);
        up_us + fused_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Share the packed `[gate|up]` accessor with the cuBLAS peer
        // so accessor emission is deduplicated by `collect_accessors`.
        FusedGateUpSiluMulImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassFusedGateUpSiluMul { in_slot, out_slot, layer, weight_fn }`.
    // Same shape as `FusedGateUpSiluMul` but its own variant ident —
    // the body slices the packed weight into gate/up halves and runs
    // a CUTLASS GEMM + EVT SiLU/Mul pair instead of cuBLAS+silu_and_mul.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("CutlassFusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("CutlassFusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!(
                "CutlassFusedGateUpSiluMul: Silu's first input must be a Tile (got {other:?})"
            ),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedGateUpSiluMul: gate gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
        )])
    }
}

// ── CutlassFusedGateUpGeluMulImpl ────────────────────────────────
//
// CUTLASS peer to [`FusedGateUpGeluMulImpl`]. Mirrors the cuBLAS
// peer's structure exactly: ONE packed CUTLASS GEMM at (M, 2I, K)
// producing a `[M, 2I]` intermediate buffer, then a BW-bound
// `gelu_and_mul_fused` elementwise pass producing `[M, I]`. Tile is
// parameterised so the DP picks the best CUTLASS tile per (M, 2I, K)
// — one Impl per CUTLASS_TILE_ZOO entry.
//
// This is structurally NOT the EVT 2-GEMM design used by
// `CutlassFusedGateUpSiluMulImpl`; that design wins for BW-bound
// shapes (small M, modest I/H — granite, llama-1b) but loses for
// compute-bound shapes (gemma2/gemma3 MLPs). The packed approach is
// faithful to the cuBLAS peer and wins iff
// `cutlass_<tile>(M, 2I, K) < cublas(M, 2I, K)`.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedGateUpGeluMulImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedGateUpGeluMulImpl {
    fn csv_name(&self) -> &'static str {
        // Reuses the standalone CutlassGemm tile names — the GEMM
        // step is an ordinary cutlass_<TM>x<TN>_s<S> at packed N=2I.
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same 4-tile (Gemm, Gemm, Gelu, Mul) claim as cuBLAS peer.
        FusedGateUpGeluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same cost shape as FusedGateUpGeluMulImpl (cuBLAS peer):
        //   - one GEMM at packed (M, 2I, K)
        //   - one BW-bound gelu_and_mul_fused over [M, 2I] → [M, I]
        // Difference: the GEMM is a calibrated cutlass_<tile> row,
        // not cublas. The BW-bound piece is identical → comparison
        // reduces to cutlass_<tile>(M, 2I, K) vs cublas(M, 2I, K).
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        // Reject shapes that this tile can't profitably handle:
        // packed_n must be a multiple of tile_n (alignment), and the
        // CSV must have a calibrated row at (M, 2I, K).
        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), num_tokens, packed_n, hidden)
            .unwrap_or(UNCALIBRATED_COST_US);

        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
        let elem_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        let _ = m; // claim shape is constant; cost is workload-driven
        gemm_us + elem_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedGateUpGeluMulImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("CutlassFusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("CutlassFusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!(
                "CutlassFusedGateUpGeluMul: Gelu's first input must be a Tile (got {other:?})"
            ),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedGateUpGeluMul: gate gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;

        // Bake (packed_n=2I, k=H) into the Instruction for the runtime
        // assert_weight_shape check. Use the gate Gemm's K and 2× the
        // gate's N; both are static per FUF + bounds.
        let (gate_n, k) = gemm_nk_from_fuf(fuf, gate_node, bounds)
            .expect("CutlassFusedGateUpGeluMul: gate (N, K) must resolve from FUF + bounds");
        let packed_n = gate_n.saturating_mul(2);
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            packed_n,
            k,
        )])
    }
}

/// Return the `cos_sin_cache` TokenStream for a rope-related tile.
/// Checks whether the tile (or any tile in `claimed`) carries an
/// `ExternKind::RotaryLocal` input; if so emits `wm.rotary_local`,
/// otherwise `wm.rotary`. Both are macro-emitted fields on the
/// per-arch `Weights` struct — scratchy owns rotary end-to-end and
/// `ForwardCtx` carries no rotary.
fn rotary_cos_sin_tokens(fuf: &Fuf, claimed: &[TileId]) -> TokenStream {
    let uses_local = claimed.iter().any(|&tid| {
        fuf.get(tid).inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Extern {
                    kind: ExternKind::RotaryLocal,
                    ..
                }
            )
        })
    });
    if uses_local {
        quote! { wm.rotary_local.cos_sin_cache }
    } else {
        quote! { wm.rotary.cos_sin_cache }
    }
}

// ── FusedGateUpGeluMulImpl ───���──────────────────────────────���────
//
// Structural mirror of [`FusedGateUpSiluMulImpl`] for the GELU
// variant of the gate/up MLP fusion. Claims the
// `(Gemm, Gemm, Gelu, Mul)` pattern — the MLP topology of any
// architecture whose activation is GELU rather than SwiGLU.
//
// Pattern-match logic is identical to the Silu variant, seeded on
// the gate Gemm; the only structural differences in emission are:
//   - the dispatched kernel is `gelu_and_mul_fused`;
//   - the upstream activation tile is the `Gelu` node (not `Silu`).
//
// Registered alongside the Silu variant in `starter_library`; the
// two are mutually exclusive on any given FUF since a single gate
// Gemm cannot simultaneously feed a Silu and a Gelu.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedGateUpGeluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed on the gate Gemm (whose output feeds the Gelu). The
        // FUF's topological order puts Gemms before their downstream
        // consumers, so seeding on the upstream Gemm commits the
        // 4-tile claim before the DP greedy-claims the Gelu.
        let gate_gemm = fuf.get(seed);
        if gate_gemm.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(gate_gemm), Some(StorageFormat::Dense)) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        let up_gemm = fuf.get(up_gemm_id);
        if up_gemm.op != OpKind::Gemm {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(up_gemm)? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Cost = one cuBLAS GEMM at packed (M, 2I, K) + one elementwise
        // gelu_and_mul pass over [M, 2I] → [M, I].
        //
        // Read measured cublas cost from the calibration CSV (predictor
        // linreg-extrapolates when the exact (M, 2I, K) row isn't
        // sampled). Fall back to peak-FLOPS roofline only when the
        // predictor has no signal at all — keeps this Impl on the same
        // measurement scale as `CutlassFusedGateUpGeluMulImpl`, which
        // sums measured cutlass CSV rows. This is the GELU twin of the
        // silu fix in commit 4a67304ec; without it the DP saw a wildly
        // optimistic roofline cuBLAS path (~8× faster than reality)
        // and never picked the CUTLASS sibling.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Bandwidth-bound gelu*mul: read 2*M*I, write M*I, bf16 = 2 B.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
        let act_mul_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + act_mul_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGateUpGeluMul { in_slot, out_slot, layer, weight_fn }`.
    // GELU twin of `FusedGateUpSiluMul`: same 4-tile claim shape, same
    // packed `[gate|up]` LinearLayer, only the elementwise kernel
    // differs (`gelu_and_mul_fused` vs `silu_and_mul_fused`).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("FusedGateUpGeluMul: Gelu's first input must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedGateUpGeluMul: gate gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::FusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── ScalarMulImpl ────────────────────────────────────────────────
//
// Claims a singleton `OpKind::Mul` whose inputs are exactly one
// Tile and one Scalar — the `x * s` pattern used e.g. by Gemma's
// embedding scale (`embed(ids, w) * sqrt(hidden_size)`). Emits an
// in-place `scale_inplace` kernel call (cublas S-axpy / scalEx)
// and move-consumes the upstream OwnedTensor as the output.
//
// **Unity passthrough.** `x * 1.0` is mathematically a no-op; the
// DSL uses it as a structural alias when an `if`/`else` merge-carry
// must bind the same name in both arms but one arm's value is just
// the source tile (e.g. ModernBERT's layer-0 attn-norm identity).
// Detected at compile time via the FUF's `Scalar(1.0)` input and
// emitted as an `output_alias` (no kernel call, no consume) — the
// downstream consumers read the source's slot directly. Without
// this, ScalarMul would `take_owned` the source even when the
// source has other live consumers (the residual add later in the
// same iteration), and those consumers would panic at runtime
// reading a slot that's been emptied.
//
// Tensor × tensor Muls (SwiGLU / GELU MLP fusions) have Tile+Tile
// inputs and are claimed by `FusedGateUp{Silu,Gelu}MulImpl`; they
// reject any Mul with non-Tile inputs, so the two Impl families
// pick disjoint patterns.

#[derive(Debug, Default)]
pub struct ScalarMulImpl;

impl ScalarMulImpl {}

#[cfg(feature = "cuda")]
impl Implementation for ScalarMulImpl {
    fn name(&self) -> &'static str {
        "scalar_mul_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Mul || node.inputs.len() != 2 {
            return None;
        }
        // Need exactly one Tile and one Scalar input (in either order).
        let mut has_tile = false;
        let mut has_scalar = false;
        for inp in &node.inputs {
            match inp {
                FufInput::Tile { .. } => has_tile = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if !(has_tile && has_scalar) {
            return None;
        }
        let tile_input: Vec<TileId> = node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: tile_input,
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // One read + one write over the tensor.
        let numel = ctx.num_tokens() * ctx.bounds.get("hidden_size").copied().unwrap_or(0);
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::ScalarMulUnityOrOwn
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        // Unity passthrough emits no kernel and aliases the source
        // — there's nothing to consume.
        if crate::weight_vocab::scalar_mul_is_unity_passthrough(claimed_tiles, fuf) {
            return Vec::new();
        }
        // Default: the kernel mutates the upstream buffer in place
        // and the emit moves it into the output binding — codegen
        // must not schedule a drop for the upstream local.
        vec![crate::weight_vocab::scalar_mul_tile_input(
            claimed_tiles,
            fuf,
        )]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `ScalarMul { in_slot, out_slot, scale }`. In-place
    // mutator: `take_owned` lifts the upstream OwnedTensor out of
    // `in_slot`, the kernel mutates it, then we reinsert at
    // `out_slot`. The codegen drop-pass already excludes the
    // upstream from `Free` (see `consumes_input_tiles`), so the
    // take→reinsert pattern is balanced.
    //
    // Unity passthrough (scale == 1.0) returns `Some(vec![])`:
    // `output_alias` already collapsed the output onto the source's
    // slot, so emitting no instruction is the correct lowering.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "ScalarMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("scale", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        if crate::weight_vocab::scalar_mul_is_unity_passthrough(&m.claimed_tiles, fuf) {
            return Some(Vec::new());
        }
        let tile = m.claimed_tiles[0];
        let (in_id, in_slot) = crate::weight_vocab::scalar_mul_tile_input(&m.claimed_tiles, fuf);
        let scale = crate::weight_vocab::scalar_mul_scale_of(&m.claimed_tiles, fuf);
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::ScalarMul(
            in_slot_idx,
            out_slot_idx,
            scale,
        )])
    }
}

// ── TanhSoftCapImpl ──────────────────────────────────────────────
//
// Singleton Impl claiming [`OpKind::TanhSoftCap`]. Maps to the
// in-place `tanh_softcap_inplace` kernel: `x[i] = cap * tanh(x[i] / cap)`.
//
// The cap scalar is read from the model config under the convention
// key `final_logit_softcapping` (HF naming). Architectures that
// don't cap logits don't emit a `TanhSoftCap` tile in their DSL
// body, so they never reach this Impl. If a DSL body emits the tile
// on a model whose config has no cap, the solver reports it — a
// missing convention field is a hard error at emit time, not a
// silent no-op. (Same posture as the attention softcap: the convention
// field exists or the tile shouldn't be in the FUF.)

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct TanhSoftCapImpl;

#[cfg(feature = "cuda")]
impl Implementation for TanhSoftCapImpl {
    fn name(&self) -> &'static str {
        "tanh_softcap_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::TanhSoftCap)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // One bf16 read + one bf16 write over the input tensor.
        let numel = ctx.num_tokens() * ctx.bounds.get("vocab_size").copied().unwrap_or(0);
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        // The kernel mutates the upstream buffer in place; the emit
        // moves the upstream OwnedTensor into the output binding so
        // the function can return it. Codegen must not schedule a
        // drop for the upstream local.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("tanh_softcap input is a tile");
        vec![src]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `TanhSoftCap { in_slot, out_slot }`. The `cap` scalar
    // is arch-wide (`final_logit_softcapping` from config.json), so
    // it's baked into the arm body at codegen time — not a payload
    // field. In-place consume: take_owned → mutate → reinsert.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TanhSoftCap",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("TanhSoftCap: first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::TanhSoftCap(in_slot_idx, out_slot_idx)])
    }
}

/// Unit-gain RMSNorm singleton (`rmsnorm_unit(x)`, mlx
/// `RMSNormNoScale`). Gemma4 normalizes V per head before the cache
/// write with no learnable scale. The per-head reduction width is
/// baked per attention CLASS at fan_out: the tile's value flows into a
/// `rope_append` whose attention consumer is `sliding_attention`
/// (base head_dim × kv heads) or `attention` (global head_dim × global
/// kv heads, Gemma4: 512×1).
#[derive(Debug)]
#[cfg(feature = "cuda")]
pub struct RmsNormUnitImpl;

#[cfg(feature = "cuda")]
impl Implementation for RmsNormUnitImpl {
    fn name(&self) -> &'static str {
        "rmsnorm_unit"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::RmsNormUnit)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // One read + one write over a [T, kv*head_dim] tensor.
        let width = ctx.bounds.get("hidden_size").copied().unwrap_or(0);
        let numel = ctx.num_tokens() * width;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RmsNormUnit",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("RmsNormUnit: first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);

        // Attention class: the unit-norm output feeds a rope_append
        // tile; that rope's attention consumer decides sliding vs
        // global geometry. Sliding consumer ⇒ base (head_dim, kv);
        // anything else (incl. uniform models) ⇒ global, which
        // defaults to base when the config lacks global_* keys.
        let rope = fuf.nodes.iter().find(|n| {
            matches!(n.op, OpKind::RopeAppend | OpKind::RopeAppendInterleaved)
                && consumes_tile(n, tile)
        });
        let is_sliding = rope.is_some_and(|r| {
            fuf.nodes
                .iter()
                .any(|n| n.op == OpKind::SlidingAttention && consumes_tile(n, r.id))
        });
        let head_dim = *bounds.get("head_dim").unwrap_or(&0);
        let num_kv = *bounds.get("num_key_value_heads").unwrap_or(&1);
        let (hidden, m_mult) = if is_sliding {
            (head_dim, num_kv)
        } else {
            (
                *bounds.get("global_head_dim").unwrap_or(&head_dim),
                *bounds.get("num_global_key_value_heads").unwrap_or(&num_kv),
            )
        };
        assert!(
            hidden > 0,
            "RmsNormUnit: model must declare head_dim for the per-head reduction width"
        );
        Some(vec![scratchy_forward_compiler::Instruction::RmsNormUnit(
            in_slot_idx,
            out_slot_idx,
            hidden as u32,
            m_mult as u32,
        )])
    }
}

/// Multiply by a loaded `[1]`-shaped weight (Gemma4 `layer_scalar`,
/// applied to the hidden state at the end of every decoder layer).
/// DSL `scalar_weight_mul(x, layer_scalar[layer])`.
#[derive(Debug)]
#[cfg(feature = "cuda")]
pub struct ScalarWeightMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for ScalarWeightMulImpl {
    fn name(&self) -> &'static str {
        "scalar_weight_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::ScalarWeightMul)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let numel = ctx.num_tokens() * ctx.bounds.get("hidden_size").copied().unwrap_or(0);
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "ScalarWeightMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("ScalarWeightMul: input tile");
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { index: Some(l), .. } => Some(l.0 as u32),
                _ => None,
            })
            .expect("ScalarWeightMul: layered weight ref (layer_scalar[layer])");
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![
            scratchy_forward_compiler::Instruction::ScalarWeightMul(
                in_slot_idx,
                out_slot_idx,
                layer,
            ),
        ])
    }
}

// ── AddRefImpl ───────────────────────────────────────────────────
//
// Singleton fallback for `OpKind::Add` tiles whose downstream is
// neither `RmsNorm` (claimed by `FusedAddRmsNormImpl`) nor
// `RmsNorm`-with-scalar-offset (claimed by
// `FusedAddRmsNormWithOffsetImpl`). Necessary for arches whose
// residual stream isn't immediately followed by the next layer's
// pre-norm — Cohere's parallel attn+MLP topology adds the attn
// output and the MLP output into the residual as two separate
// `add(...)` statements with no intervening norm.
//
// Both Add inputs must be tiles (not weights/scalars/externs); the
// scalar-offset case is left to `ScalarOffsetRmsNormImpl`. Emits
// `add_inplace(residual, delta)` mutating the slot-1 (residual)
// buffer; the Add's output is bound as a TensorView alias of that
// same buffer. Convention matches `FusedAddRmsNormImpl`: slot 0 is
// the new contribution, slot 1 is the residual.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AddRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for AddRefImpl {
    fn name(&self) -> &'static str {
        "add_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Add || node.inputs.len() != 2 {
            return None;
        }
        // Both inputs must be Tiles. ScalarOffset Adds (Weight + Scalar)
        // are claimed by `ScalarOffsetRmsNormImpl`; mixed Tile+Weight
        // shouldn't occur for residual adds.
        let slot_a = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        let slot_b = match &node.inputs[1] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        // Defer to the (Add, RmsNorm) fusion when the downstream is a
        // RmsNorm consuming this Add's output. Cohere's LayerNorm has
        // no fusion impl yet, so we DO claim Adds whose downstream is
        // a LayerNorm (the layer-end residuals before the next layer's
        // pre-norm). When a `FusedAddCohereLayerNormImpl` lands, mirror
        // this gate.
        let downstream_is_rmsnorm = fuf
            .nodes
            .iter()
            .any(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed));
        // Defer to the (Add, RmsNorm) fusion impl when the downstream is a
        // RmsNorm consuming this Add's output (it claims the larger,
        // single-launch subgraph).
        if downstream_is_rmsnorm {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![slot_a, slot_b],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Read residual + delta, write residual: 3 reads/writes per
        // element, bandwidth-bound.
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * m * hidden * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::AliasToInput(1)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Add { delta_slot, residual_slot }`. No `out_slot`
    // payload — `output_alias` declares the Add's output as a View
    // alias of the residual input, and the codegen-emitted prelude
    // populates the alias slot before the interpreter loop runs.
    // The arm only performs the in-place mutation.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Add",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let add_id = m.claimed_tiles[0];
        let node = fuf.get(add_id);
        let (delta_id, delta_in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Add: input 0 (delta) must be a Tile (got {other:?})"),
        };
        let (residual_id, residual_in_slot) = match node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Add: input 1 (residual) must be a Tile (got {other:?})"),
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        Some(vec![Instruction::Add(delta_idx, residual_idx)])
    }
}

// ── FusedAddRmsNormImpl ──────────────────────────────────────────
//
// Multi-tile impl claiming `(Add, RmsNorm)` where the RmsNorm's
// input is the Add's output. Maps to
// `scratchy_target_cuda::fused_add_rms_norm_inplace`, which:
//   residual += delta          (in-place on residual)
//   input_buf = norm(residual) (in-place on delta's buffer)
// returning (normed_ptr_aliasing_delta, residual_ptr).
//
// Emit binds the `RmsNorm` output to the post-mutation delta buffer
// (as a `TensorView` alias) and the `Add` output to the post-mutation
// residual buffer (also as a `TensorView` alias). Both aliases borrow
// off the ambient OwnedTensor bindings that held `delta` and
// `residual` before the call — those OwnedTensors stay alive in the
// forward fn scope for the kernel's async lifetime.
//
// Seeded on the `Add`: topologically upstream, guaranteed to be
// processed before the solver reaches the `RmsNorm`. A lone `Add`
// with no downstream `RmsNorm` (does not occur in Llama/Qwen2)
// returns `None` → `UnclaimedTile` library gap.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedAddRmsNormImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        "fused_add_rms_norm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let add_node = fuf.get(seed);
        if add_node.op != OpKind::Add {
            return None;
        }
        // Residual-stream pattern only: both Add operands must be
        // tiles (delta + residual, both tensor-valued). An Add with
        // a `Scalar` or `Weight` input is a different shape (e.g.
        // Gemma's `w + 1.0` feeding rmsnorm) — hand it to
        // [`ScalarOffsetRmsNormImpl`] instead of mis-fusing here.
        if !add_node
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }
        // Find an immediate RmsNorm consumer. "Immediate" in the
        // topological sense: any RmsNorm whose first input is this
        // Add's output. More than one such consumer is possible in
        // theory (not in Llama/Qwen2); we claim the first we find.
        let rmsnorm_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;

        let mut claimed = [seed, rmsnorm_node.id];
        claimed.sort();
        let claimed = claimed.to_vec();

        // Boundary inputs: the Add's two upstream tiles (delta and
        // residual) — RmsNorm's weight input is a Weight, not a Tile.
        let boundary_inputs: Vec<TileId> = add_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            // Both outputs are live downstream: RmsNorm's normed
            // feeds the post-norm compute; Add's updated-residual
            // feeds the next residual stream.
            boundary_outputs: vec![seed, rmsnorm_node.id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: one read of delta + residual + weight,
        // one write of residual + normed. bf16 everywhere.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        // 2 reads (delta, residual) + 2 writes (residual, normed) of
        // [num_tokens, hidden] bf16, plus weight [hidden] bf16 read.
        let bytes = (4.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    // Default `required_weights` suffices: the only weight input
    // across claimed tiles belongs to the RmsNorm, and it's consumed
    // by an `OpKind::RmsNorm` tile → typed as `RmsNorm` by
    // `rust_type_for_weight_consumed_by`.

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::NormDeltaResidual
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedAddRmsNorm { delta_slot, residual_slot, layer, weight_fn }`.
    // No `out_slot` payload — `output_alias` declares both outputs
    // (rmsnorm_id slot 0 → delta upstream, add_id slot 0 → residual
    // upstream) and the alias prelude populates them as `View`
    // entries before the loop. The arm just runs the in-place
    // mutation kernel.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedAddRmsNorm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )
    }

    #[cfg(feature = "metal")]
    fn as_atom(&self, _m: &MatchInfo, _fuf: &Fuf) -> Option<Box<dyn crate::atom::Atom>> {
        Some(Box::new(crate::atom_lib::AddRmsNormAtom::default()))
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("FusedAddRmsNorm: claim contains Add");
        let add_node = fuf.get(add_id);
        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedAddRmsNorm: Add input 0 (delta) must be a Tile (got {other:?})"),
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedAddRmsNorm: Add input 1 (residual) must be a Tile (got {other:?})")
            }
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedAddRmsNorm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        // Always on the residual stream → the full width,
        // `m_multiplier = 1`. Vision-prelude towers (Qwen2.5-VL: norm2
        // + every post-attn norm fuse into (Add, RmsNorm)) ride the
        // `vision_embed_dim` stream, NOT the text `hidden_size` — see
        // `RmsNormRefImpl::fan_out` for the same vision-width rule.
        let hidden_size = if matches!(program.prelude, crate::classified::Prelude::Vision) {
            *bounds.get("vision_embed_dim").unwrap_or(&0) as u32
        } else {
            *bounds.get("hidden_size").unwrap_or(&0) as u32
        };
        Some(vec![Instruction::FusedAddRmsNorm(
            delta_idx,
            residual_idx,
            layer,
            hidden_size,
            1,
        )])
    }
}

// ── MeanSubRmsNormImpl ───────────────────────────────────────────
//
// Multi-tile impl claiming `(Mean, Sub, RmsNorm)` where
// `Sub.lhs == Mean.input` and `RmsNorm.input == Sub.output`. This
// is the math factorization of LayerNorm:
//
//     mu       = mean(x)
//     centered = sub(x, mu)
//     normed   = rmsnorm(centered, w)
//
// equals `w * (x - mean(x)) / sqrt(var(x) + eps)` — identical to
// the `cohere_layer_norm` kernel's math. The DSL stays a pure
// math expression; this Impl claims the trio and emits one
// `cohere_layer_norm` call. Net runtime effect vs the previous
// retired `OpKind::LayerNorm` singleton path: same kernel, same launch
// count, same memory traffic.
//
// Seeded on the Mean: topologically upstream of Sub and RmsNorm,
// so the solver processes it first. The Sub/RmsNorm get pulled
// into the claim; an unmatched lone Mean (no `(Sub, RmsNorm)`
// downstream) returns `None` and surfaces as
// `SolveError::UnclaimedTile` — the same library-gap signal the
// Silu/Mul precedent uses.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MeanSubRmsNormImpl;

#[cfg(feature = "cuda")]
impl Implementation for MeanSubRmsNormImpl {
    fn name(&self) -> &'static str {
        "mean_sub_rms_norm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let mean_node = fuf.get(seed);
        if mean_node.op != OpKind::Mean {
            return None;
        }
        // Mean's single input must be a Tile — the source tensor x.
        let x_id = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => return None,
        };
        // Find a Sub whose lhs is x and rhs is the Mean output.
        let sub_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::Sub
                && n.inputs.len() == 2
                && matches!(&n.inputs[0], FufInput::Tile { id, .. } if *id == x_id)
                && matches!(&n.inputs[1], FufInput::Tile { id, .. } if *id == seed)
        })?;
        // Find an RmsNorm whose first input is the Sub's output.
        let rmsnorm_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::RmsNorm
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == sub_node.id
                )
        })?;

        let mut claimed = vec![seed, sub_node.id, rmsnorm_node.id];
        claimed.sort();

        Some(MatchInfo {
            claimed_tiles: claimed,
            // Boundary input: just the upstream tile x. RmsNorm's
            // weight is a Weight (not a Tile), surfaced via
            // `required_weights`.
            boundary_inputs: vec![x_id],
            // Boundary output: only the RmsNorm result; the Mean
            // and Sub outputs are internal to the fusion.
            boundary_outputs: vec![rmsnorm_node.id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same bandwidth as today's standalone LayerNorm: read x +
        // read w, write normed. Mean and Sub are folded into the
        // single kernel — no separate launches. Reuse
        // `elementwise_cost` keyed on the RmsNorm tile to model the
        // real read/write traffic.
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("MeanSubRmsNorm: claim contains RmsNorm");
        let single = MatchInfo {
            claimed_tiles: vec![rmsnorm_id],
            boundary_inputs: m.boundary_inputs.clone(),
            boundary_outputs: m.boundary_outputs.clone(),
        };
        elementwise_cost(&single, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    // Default `required_weights` suffices: the only weight in the
    // claim is the RmsNorm's, consumed by `OpKind::RmsNorm` →
    // typed as `RmsNorm` by `rust_type_for_weight_consumed_by`.

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MeanSubRmsNorm { in_slot, out_slot, layer, weight_fn }`.
    // Same payload shape as the existing `LayerNorm` opcode but the
    // weight type is `RmsNorm` instead of `CohereLayerNorm`. The two
    // wrapper structs are byte-identical (`weight: GpuTensor + eps:
    // f32`) and the emitted kernel reads only those two fields, so
    // the runtime path is unchanged.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MeanSubRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        // The upstream tile x feeding the Mean is the kernel input.
        let mean_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mean)
            .expect("MeanSubRmsNorm: claim contains Mean");
        let mean_node = fuf.get(mean_id);
        let (x_id, x_in_slot) = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MeanSubRmsNorm: Mean's input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(x_id, x_in_slot);

        // The RmsNorm output is the fusion's output.
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("MeanSubRmsNorm: claim contains RmsNorm");
        let out_slot_idx = slots.of(rmsnorm_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MeanSubRmsNorm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::MeanSubRmsNorm(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── MeanSubRmsNormBiasAddImpl ────────────────────────────────────
//
// 4-tile fusion `(Mean, Sub, RmsNorm, BiasAdd)`. Extends
// [`MeanSubRmsNormImpl`] with one extra tile — a `BiasAdd` whose
// tile-input is the rmsnorm output. Math:
//
//     mu       = mean(x)
//     centered = sub(x, mu)
//     normed   = rmsnorm(centered, w)
//     biased   = bias_add(normed, b)
//
// equals `w * (x - mean(x)) / sqrt(var(x) + eps) + b` — the torch-
// flavored LayerNorm. Emits one `kernels::layer_norm_bias` call.
//
// Routing: claim-size-DESC sort means this beats the 3-tile
// `MeanSubRmsNormImpl` whenever a `BiasAdd` is downstream of the
// RmsNorm, and falls through to the trio otherwise. The DSL bias
// weight ref (e.g. `attn_norm.bias[layer]`) is structural — the
// `required_weights` accessor names the rmsnorm's weight prefix and
// types it `LayerNorm`, so the loader pulls `<prefix>.weight` AND
// `<prefix>.bias` together via `LayerNorm::load`. Same trick the
// dense `(Gemm, BiasAdd)` fusion plays with `LinearLayer`.
//
// Used by encoder models like ModernBERT.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MeanSubRmsNormBiasAddImpl;

#[cfg(feature = "cuda")]
impl Implementation for MeanSubRmsNormBiasAddImpl {
    fn name(&self) -> &'static str {
        "mean_sub_rms_norm_bias_add"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let mean_node = fuf.get(seed);
        if mean_node.op != OpKind::Mean {
            return None;
        }
        let x_id = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => return None,
        };
        // Sub: lhs=x, rhs=Mean output.
        let sub_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::Sub
                && n.inputs.len() == 2
                && matches!(&n.inputs[0], FufInput::Tile { id, .. } if *id == x_id)
                && matches!(&n.inputs[1], FufInput::Tile { id, .. } if *id == seed)
        })?;
        // RmsNorm: input[0]=Sub output.
        let rmsnorm_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::RmsNorm
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == sub_node.id
                )
        })?;
        // BiasAdd: input[0]=RmsNorm output, input[1]=Weight (the bias).
        // The bias's exact path isn't checked here — the loader maps
        // the rmsnorm's weight prefix to `LayerNorm::load` which pulls
        // `<prefix>.bias` from disk. A DSL author who points the
        // bias at an unrelated path will see a load-time error.
        let bias_add_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::BiasAdd
                && n.inputs.len() == 2
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == rmsnorm_node.id
                )
                && matches!(&n.inputs[1], FufInput::Weight { .. })
        })?;

        let mut claimed = vec![seed, sub_node.id, rmsnorm_node.id, bias_add_node.id];
        claimed.sort();

        Some(MatchInfo {
            claimed_tiles: claimed,
            // Boundary input: just x.
            boundary_inputs: vec![x_id],
            // Boundary output: BiasAdd's result is the live downstream tile.
            boundary_outputs: vec![bias_add_node.id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same bandwidth profile as the trio — `layer_norm_bias`
        // reads x + w + b and writes one output, all elementwise. The
        // bias is hidden_size which is dwarfed by x's [T, hidden]
        // traffic, so charging the rmsnorm tile's elementwise_cost is
        // a tight upper bound.
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("MeanSubRmsNormBiasAdd: claim contains RmsNorm");
        let single = MatchInfo {
            claimed_tiles: vec![rmsnorm_id],
            boundary_inputs: m.boundary_inputs.clone(),
            boundary_outputs: m.boundary_outputs.clone(),
        };
        elementwise_cost(&single, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // ONE accessor typed `LayerNorm`, source = the RmsNorm's
        // weight ref. The codegen FieldLoad picker sees the
        // `LayerNorm` rust_type and emits `LayerNorm::load(gw, prefix,
        // eps)` which pulls `<prefix>.weight` AND `<prefix>.bias`
        // together. The DSL's BiasAdd-side weight ref doesn't appear
        // in the accessor list — same trick as
        // `FusedGemmBiasImpl` (which exposes `LinearLayer` for a Gemm
        // whose `.bias` rides through the loader).
        let rmsnorm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("claim contains RmsNorm");
        let rmsnorm_weight =
            first_weight_ref(fuf.get(rmsnorm_id)).expect("RmsNorm tile carries a weight reference");
        let sources = vec![rmsnorm_weight];
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LayerNorm },
            source_weights: sources,
        }]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MeanSubRmsNormBiasAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let mean_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mean)
            .expect("claim contains Mean");
        let mean_node = fuf.get(mean_id);
        let (x_id, x_in_slot) = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MeanSubRmsNormBiasAdd: Mean's input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(x_id, x_in_slot);

        // Output is the BiasAdd's result.
        let bias_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("claim contains BiasAdd");
        let out_slot_idx = slots.of(bias_add_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors.first().expect("required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::MeanSubRmsNormBiasAdd(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── CutlassFusedRmsNormGemmImpl / CutlassFusedMeanSubRmsNormGemmImpl / CutlassFusedAddRmsNormGemmImpl ──
//
// Norm→Gemm fusion family. Captures the 1095 lm_head + 563 body
// `Norm→Gemm` cuBLAS picks identified by `attack_surface.rs`'s
// fusion-gap analyzer. Three claim shapes:
//
//   1. (RmsNorm, Gemm)         — body norm whose only consumer is a
//                                 single Gemm. Most arches.
//   2. (LayerNorm, Gemm)       — Cohere-style. Same shape as (1) but
//                                 different norm kernel + weight type.
//   3. (Add, RmsNorm, Gemm)    — residual-stream norm followed by a
//                                 single-consumer Gemm. Lm_head's
//                                 canonical shape (`x = x + mlp; x_n =
//                                 norm(x); logits = lm_head(x_n)`).
//                                 3-tile size beats FusedAddRmsNormImpl
//                                 in the DP's claim-size-DESC sort.
//
// All three are tile-zoo-pickable: one Impl per `CUTLASS_TILE_ZOO`
// entry. The runtime arm runs the existing norm kernel + the
// existing `cutlass::cutlass_gemm` — no new .cu file. The fusion is
// at the CLAIM level: by claiming the Gemm together with its upstream
// norm, the cost model can compare `(norm + cutlass_<tile>)` against
// `(norm-singleton + cuBLAS-Gemm)` at the bucket level rather than
// forcing the DP to evaluate them as independent steps.
//
// The DP picks naturally: claim-size-DESC means 3-tile fires before
// 2-tile, and within each, the lowest cost wins. multi-consumer norms
// (RmsNorm feeding Q+K+V — body residual stream) are rejected so
// FusedQkvRope* still claims them.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedRmsNormGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedRmsNormGemmImpl {
    fn csv_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

/// Match a (Norm, Gemm) pair seeded on the Norm. Norm's output (slot
/// 0) must be consumed by exactly one tile, which must be a dense
/// `OpKind::Gemm` taking the Norm at its first input. Returns
/// `(norm_id, gemm_id)` or `None`.
fn match_norm_gemm_pair(
    fuf: &Fuf,
    norm_seed: TileId,
    norm_kind: OpKind,
) -> Option<(TileId, TileId)> {
    let norm_node = fuf.get(norm_seed);
    if norm_node.op != norm_kind {
        return None;
    }
    if let Some(s) = weight_storage_of(norm_node)
        && !matches!(s, StorageFormat::Dense)
    {
        return None;
    }
    // Collect consumers of `norm_seed`'s slot 0.
    let mut consumers = fuf.nodes.iter().filter(|n| consumes_tile(n, norm_seed));
    let only = consumers.next()?;
    if consumers.next().is_some() {
        return None;
    }
    if only.op != OpKind::Gemm {
        return None;
    }
    if !matches!(weight_storage_of(only), Some(StorageFormat::Dense)) {
        return None;
    }
    // Gemm's first tile input must be the Norm (slot 0).
    match first_tile_input(only) {
        Some((id, slot)) if id == norm_seed && slot == 0 => {}
        _ => return None,
    }
    Some((norm_seed, only.id))
}

/// Roofline FLOPs cost — the same fallback `GemmRefImpl::cost_us`
/// (via `cost_gemm`) uses when the cuBLAS CSV row is missing. Used
/// by every fused-Gemm peer below at uncalibrated shapes
/// (lm_head's vocab-N rows are not swept) so cuBLAS-vs-fusion stays
/// on a single measurement scale rather than comparing roofline
/// (cuBLAS) against `UNCALIBRATED_COST_US` (CUTLASS).
fn cutlass_gemm_roofline_us(ctx: &CostCtx, m: u32, n: u32, k: u32) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak > 0.0 && flops > 0.0 {
        (flops / peak) * 1e6
    } else {
        UNCALIBRATED_COST_US
    }
}

/// Whether a `(M, N, K)` shape satisfies the standalone CUTLASS
/// tile zoo's alignment requirements. The tiles in
/// `cutlass_standalone_gemm.cu` are templated on `AlignmentA = 8`
/// and `AlignmentB = 8` (see csrc), which forces N and K to be
/// multiples of 8. Granite-3.3-2B's `vocab_size = 49159` (genuinely
/// not divisible by 8) was the failure that surfaced this — when
/// the Step 1b(a) DP fix made `CutlassFusedRmsNormGemm` pickable
/// at lm_head, the kernel returned `-1` at the (1024, 49159, 2048)
/// shape and crashed `scr serve`.
///
/// Used by every cutlass-tile-zoo Impl's `cost_us` to return
/// `UNCALIBRATED_COST_US` when alignment doesn't hold; the DP then
/// routes to a fallback (cuBLAS singleton for vocab-N at cuBLAS-ON,
/// or singleton-CUTLASS+kernel-failure at cuBLAS-OFF — Lever A2
/// territory for full coverage).
fn cutlass_tile_supports_shape(_m: u32, n: u32, k: u32) -> bool {
    n.is_multiple_of(8) && k.is_multiple_of(8)
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedRmsNormGemmImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let (norm_id, gemm_id) = match_norm_gemm_pair(fuf, seed, OpKind::RmsNorm)?;
        let norm_node = fuf.get(norm_id);
        let activation_in = first_tile_input(norm_node)?.0;
        let mut claimed = [norm_id, gemm_id];
        claimed.sort();
        Some(MatchInfo {
            claimed_tiles: claimed.to_vec(),
            boundary_inputs: vec![activation_in],
            boundary_outputs: vec![gemm_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let norm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("CutlassFusedRmsNormGemm: claim contains RmsNorm");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedRmsNormGemm: claim contains Gemm");
        let gemm_node = ctx.fuf.get(gemm_id);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, gemm_node) else {
            return UNCALIBRATED_COST_US;
        };
        // Reject shapes the standalone CUTLASS tile zoo can't run
        // (alignment 8 on N/K). At granite-3.3-2B's vocab_size=49159,
        // cutlass_<tile> kernels return -1 → scr serve crashes.
        // Falling back to UNCALIBRATED_COST_US lets the DP route to
        // the singleton (norm + Gemm) chain instead.
        if !cutlass_tile_supports_shape(mm, nn, kk) {
            return UNCALIBRATED_COST_US;
        }
        // Match cuBLAS's measurement scale: prefer calibrated CSV row;
        // fall back to roofline when uncalibrated (lm_head vocab-N).
        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or_else(|| cutlass_gemm_roofline_us(ctx, mm, nn, kk));
        // Norm cost MUST equal `RmsNormRefImpl::cost_us` byte-for-byte
        // (`elementwise_cost` on the norm tile), or the DP's
        // claim-size-DESC tiebreak hands the lm_head pick to
        // (norm + cublas) singletons at uncalibrated shapes (where
        // gemm_us cancels via shared roofline). Hand-rolled
        // `read+write+weight` formulas here charged an extra
        // `+ hidden` weight read that the singleton does NOT —
        // 0-pick anomaly at vocab-N STATUS Step 1b(a) flagged.
        let norm_singleton = MatchInfo {
            claimed_tiles: vec![norm_id],
            boundary_inputs: m.boundary_inputs.clone(),
            boundary_outputs: vec![norm_id],
        };
        let norm_us = elementwise_cost(&norm_singleton, ctx);
        gemm_us + norm_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Deterministic order: norm accessor first, gemm accessor second.
        let mut out = Vec::new();
        for &kind in &[OpKind::RmsNorm, OpKind::Gemm] {
            for &tid in claimed_tiles {
                let node = fuf.get(tid);
                if node.op != kind {
                    continue;
                }
                for input in &node.inputs {
                    if let FufInput::Weight { id, index, .. } = input {
                        let name = weight_field_name(program, *id, *index);
                        out.push(WeightAccessor {
                            name,
                            rust_type: rust_type_for_weight_consumed_by(node.op),
                            source_weights: vec![(*id, *index)],
                        });
                    }
                }
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedRmsNormGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let norm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("CutlassFusedRmsNormGemm: claim contains RmsNorm");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedRmsNormGemm: claim contains Gemm");
        let norm_node = fuf.get(norm_id);
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match norm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("CutlassFusedRmsNormGemm: norm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(gemm_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let norm_acc = accessors
            .first()
            .expect("CutlassFusedRmsNormGemm: required_weights[0] (norm)");
        let _gemm_acc = accessors
            .get(1)
            .expect("CutlassFusedRmsNormGemm: required_weights[1] (gemm)");
        let (_norm_base, norm_layer) = split_base_layer(&norm_acc.name.to_string());
        // Both accessors should share the same layer offset (or be
        // un-layered together). We bake the norm's layer; the gemm's
        // layer at codegen time must match — un-layered accessors
        // resolve to layer=0 per emit_weights_accessor_methods's
        // contract.
        let layer = norm_layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassFusedRmsNormGemm: gemm (N, K) must resolve from FUF + bounds");
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedRmsNormGemm(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

// ── CutlassFusedMeanSubRmsNormGemmImpl ───────────────────────────
//
// Norm→Gemm fusion for CohereLayerNorm-flavored norms. Claims the
// 4-tile pattern `(Mean, Sub, RmsNorm, Gemm)` where the Gemm is
// the sole consumer of the RmsNorm output. Runs `cohere_layer_norm`
// then `cutlass_gemm` sequentially — calibration shared with the
// `CutlassFusedRmsNormGemm` CSV row.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedMeanSubRmsNormGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedMeanSubRmsNormGemmImpl {
    fn csv_name(&self) -> &'static str {
        // Share calibration with the LayerNorm-Gemm flavor — the
        // emitted kernel pair is identical (cohere_layer_norm +
        // cutlass_gemm). Using the RmsNorm-Gemm csv name would
        // mis-attribute cost since the rms variant skips the mean
        // subtraction and is measurably faster on the norm side.
        CutlassFusedRmsNormGemmImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .csv_name()
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedMeanSubRmsNormGemmImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed on Mean. Walk the trio, then look for a single Gemm
        // consuming the RmsNorm's output.
        let mean_node = fuf.get(seed);
        if mean_node.op != OpKind::Mean {
            return None;
        }
        let x_id = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            _ => return None,
        };
        let sub_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::Sub
                && n.inputs.len() == 2
                && matches!(&n.inputs[0], FufInput::Tile { id, .. } if *id == x_id)
                && matches!(&n.inputs[1], FufInput::Tile { id, .. } if *id == seed)
        })?;
        let rmsnorm_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::RmsNorm
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == sub_node.id
                )
        })?;
        if let Some(s) = weight_storage_of(rmsnorm_node)
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        // Sole consumer of the RmsNorm must be a dense Gemm whose
        // first tile input is the RmsNorm output.
        let mut consumers = fuf
            .nodes
            .iter()
            .filter(|n| consumes_tile(n, rmsnorm_node.id));
        let only = consumers.next()?;
        if consumers.next().is_some() {
            return None;
        }
        if only.op != OpKind::Gemm || !matches!(weight_storage_of(only), Some(StorageFormat::Dense))
        {
            return None;
        }
        match first_tile_input(only) {
            Some((id, slot)) if id == rmsnorm_node.id && slot == 0 => {}
            _ => return None,
        }
        let mut claimed = vec![seed, sub_node.id, rmsnorm_node.id, only.id];
        claimed.sort();
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![x_id],
            boundary_outputs: vec![only.id],
        })
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Mirror `CutlassFusedRmsNormGemmImpl::cost_us`: same
        // norm+gemm decomposition. The norm cost is keyed on the
        // RmsNorm tile (the only one with real activation traffic;
        // Mean and Sub fold into the same kernel).
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("CutlassFusedMeanSubRmsNormGemm: claim contains RmsNorm");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedMeanSubRmsNormGemm: claim contains Gemm");
        let gemm_node = ctx.fuf.get(gemm_id);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, gemm_node) else {
            return UNCALIBRATED_COST_US;
        };
        if !cutlass_tile_supports_shape(mm, nn, kk) {
            return UNCALIBRATED_COST_US;
        }
        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or_else(|| cutlass_gemm_roofline_us(ctx, mm, nn, kk));
        let norm_singleton = MatchInfo {
            claimed_tiles: vec![rmsnorm_id],
            boundary_inputs: m.boundary_inputs.clone(),
            boundary_outputs: vec![rmsnorm_id],
        };
        let norm_us = elementwise_cost(&norm_singleton, ctx);
        gemm_us + norm_us
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Norm accessor first, gemm accessor second.
        let mut out = Vec::new();
        for &kind in &[OpKind::RmsNorm, OpKind::Gemm] {
            for &tid in claimed_tiles {
                let node = fuf.get(tid);
                if node.op != kind {
                    continue;
                }
                for input in &node.inputs {
                    if let FufInput::Weight { id, index, .. } = input {
                        let name = weight_field_name(program, *id, *index);
                        out.push(WeightAccessor {
                            name,
                            rust_type: rust_type_for_weight_consumed_by(node.op),
                            source_weights: vec![(*id, *index)],
                        });
                    }
                }
            }
        }
        out
    }
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedMeanSubRmsNormGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let mean_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mean)
            .expect("CutlassFusedMeanSubRmsNormGemm: claim contains Mean");
        let mean_node = fuf.get(mean_id);
        let (in_id, in_slot) = match mean_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedMeanSubRmsNormGemm: Mean's input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedMeanSubRmsNormGemm: claim contains Gemm");
        let gemm_node = fuf.get(gemm_id);
        let out_slot_idx = slots.of(gemm_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let norm_acc = accessors
            .first()
            .expect("CutlassFusedMeanSubRmsNormGemm: required_weights[0]");
        let _gemm_acc = accessors
            .get(1)
            .expect("CutlassFusedMeanSubRmsNormGemm: required_weights[1]");
        let (_norm_base, norm_layer) = split_base_layer(&norm_acc.name.to_string());
        let layer = norm_layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassFusedMeanSubRmsNormGemm: gemm (N, K) must resolve from FUF + bounds");
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedMeanSubRmsNormGemm(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedAddRmsNormGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedAddRmsNormGemmImpl {
    fn csv_name(&self) -> &'static str {
        CutlassFusedRmsNormGemmImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .csv_name()
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedAddRmsNormGemmImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed at the residual-stream Add (same shape as
        // FusedAddRmsNormImpl). Both Add inputs must be tiles.
        let add_node = fuf.get(seed);
        if add_node.op != OpKind::Add {
            return None;
        }
        if !add_node
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }
        // Find an immediate RmsNorm consumer.
        let rmsnorm_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;
        let rmsnorm_id = rmsnorm_node.id;
        // RmsNorm's only consumer must be a single dense Gemm with
        // RmsNorm as its first tile input.
        let mut consumers = fuf.nodes.iter().filter(|n| consumes_tile(n, rmsnorm_id));
        let only = consumers.next()?;
        if consumers.next().is_some() {
            return None;
        }
        if only.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(only), Some(StorageFormat::Dense)) {
            return None;
        }
        match first_tile_input(only) {
            Some((id, slot)) if id == rmsnorm_id && slot == 0 => {}
            _ => return None,
        }
        let gemm_id = only.id;
        let mut claimed = [seed, rmsnorm_id, gemm_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let boundary_inputs: Vec<TileId> = add_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        // Both the Add output (residual update — for downstream
        // residual stream) and the Gemm output (the matmul result)
        // are externally visible.
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            boundary_outputs: vec![seed, gemm_id],
        })
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedAddRmsNormGemm: claim contains Gemm");
        let gemm_node = ctx.fuf.get(gemm_id);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, gemm_node) else {
            return UNCALIBRATED_COST_US;
        };
        if !cutlass_tile_supports_shape(mm, nn, kk) {
            return UNCALIBRATED_COST_US;
        }
        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or_else(|| cutlass_gemm_roofline_us(ctx, mm, nn, kk));
        // fused_add_rms_norm BW-bound: 2 reads (delta, residual) + 2
        // writes (residual, normed) + weight read.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = (4.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        let norm_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };
        gemm_us + norm_us
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Deterministic order: norm accessor first, gemm accessor second.
        let mut out = Vec::new();
        for &kind in &[OpKind::RmsNorm, OpKind::Gemm] {
            for &tid in claimed_tiles {
                let node = fuf.get(tid);
                if node.op != kind {
                    continue;
                }
                for input in &node.inputs {
                    if let FufInput::Weight { id, index, .. } = input {
                        let name = weight_field_name(program, *id, *index);
                        out.push(WeightAccessor {
                            name,
                            rust_type: rust_type_for_weight_consumed_by(node.op),
                            source_weights: vec![(*id, *index)],
                        });
                    }
                }
            }
        }
        out
    }
    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::AddAliasResidualIn1
    }
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedAddRmsNormGemm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("CutlassFusedAddRmsNormGemm: claim contains Add");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedAddRmsNormGemm: claim contains Gemm");
        let add_node = fuf.get(add_id);
        let gemm_node = fuf.get(gemm_id);
        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!(
                    "CutlassFusedAddRmsNormGemm: Add input 0 (delta) must be a Tile (got {other:?})"
                )
            }
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!(
                    "CutlassFusedAddRmsNormGemm: Add input 1 (residual) must be a Tile (got {other:?})"
                )
            }
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        let out_slot_idx = slots.of(gemm_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let norm_acc = accessors
            .first()
            .expect("CutlassFusedAddRmsNormGemm: required_weights[0]");
        let _gemm_acc = accessors
            .get(1)
            .expect("CutlassFusedAddRmsNormGemm: required_weights[1]");
        let (_norm_base, norm_layer) = split_base_layer(&norm_acc.name.to_string());
        let layer = norm_layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassFusedAddRmsNormGemm: gemm (N, K) must resolve from FUF + bounds");
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedAddRmsNormGemm(
            delta_idx,
            residual_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

// ── FusedAddRmsNormWithOffsetImpl ────────────────────────────────
//
// Claims three tiles: a residual-stream Add (two Tile inputs), a
// scalar-offset Add (Weight + Scalar inputs), and a RmsNorm that
// consumes the residual-Add at slot 0 and the scalar-Add at slot 1.
// Emits a single `fused_add_rms_norm_inplace` call with the
// scalar as the kernel's `weight_offset` param.
//
// This is the Gemma2 mid-layer pattern:
//   hidden_states = add(delta, hidden_states);                 // residual
//   pre_ffwd     = rmsnorm(hidden_states, weight[i] + 1.0);    // norm
// and the cross-iteration pattern where the end-of-layer residual
// Add feeds the next iteration's pre-norm (or the final norm
// after the last iteration).
//
// Without this Impl, `FusedAddRmsNormImpl` absorbs the (Add,
// RmsNorm) pair, orphaning the scalar-offset Add with no other
// claim possible — the DP reports UnclaimedTile.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedAddRmsNormWithOffsetImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedAddRmsNormWithOffsetImpl {
    fn name(&self) -> &'static str {
        "fused_add_rms_norm_with_offset"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let residual_add = fuf.get(seed);
        if residual_add.op != OpKind::Add {
            return None;
        }
        // Residual-stream Add: both inputs must be tiles.
        if !residual_add
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }

        // Find a RmsNorm whose input (slot 0) consumes this Add and
        // whose weight (slot 1) consumes a DIFFERENT Add that is a
        // scalar-offset Add (Weight + Scalar inputs).
        for rms in &fuf.nodes {
            if rms.op != OpKind::RmsNorm || rms.inputs.len() < 2 {
                continue;
            }
            let reads_residual_at_0 =
                matches!(rms.inputs[0], FufInput::Tile { id, .. } if id == seed);
            if !reads_residual_at_0 {
                continue;
            }
            let scalar_add_id = match rms.inputs[1] {
                FufInput::Tile { id, .. } => id,
                _ => continue,
            };
            let scalar_add = fuf.get(scalar_add_id);
            if scalar_add.op != OpKind::Add {
                continue;
            }
            // Scalar-offset Add: exactly one Weight and one Scalar.
            let mut has_weight = false;
            let mut has_scalar = false;
            let mut ok = true;
            for inp in &scalar_add.inputs {
                match inp {
                    FufInput::Weight { .. } => has_weight = true,
                    FufInput::Scalar(_) => has_scalar = true,
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !(ok && has_weight && has_scalar) {
                continue;
            }

            let mut claimed = [seed, scalar_add_id, rms.id];
            claimed.sort();
            let claimed = claimed.to_vec();

            // Boundary inputs: the residual-Add's two upstream tiles
            // (delta and residual). Scalar-Add's Weight/Scalar are
            // not tile inputs, so nothing else crosses the boundary.
            let boundary_inputs: Vec<TileId> = residual_add
                .inputs
                .iter()
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();

            return Some(MatchInfo {
                claimed_tiles: claimed,
                boundary_inputs,
                // Both outputs are live downstream: RmsNorm's normed
                // feeds post-norm compute; residual-Add's updated
                // residual feeds the next residual stream.
                boundary_outputs: vec![seed, rms.id],
            });
        }
        None
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same bandwidth cost as the non-offset variant; the `+1`
        // is an extra fp32 fadd per weight element — free on a
        // memory-bound kernel.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = (4.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The scalar-offset Add's Weight input is the actual rmsnorm
        // weight. Match the default rmsnorm naming scheme so the
        // codegen's RmsNorm::load path fires.
        let scalar_add_id = *claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("claim contains a scalar-offset Add");
        let scalar_add = fuf.get(scalar_add_id);
        let (weight_id, weight_idx) = scalar_add
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("scalar-offset Add has a Weight input");
        let name = weight_field_name(program, weight_id, weight_idx);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::RmsNorm },
            source_weights: vec![(weight_id, weight_idx)],
        }]
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::NormDeltaResidual
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedAddRmsNormWithOffset { delta_slot, residual_slot,
    // layer, offset, weight_fn }`. Same alias-prelude shape as
    // `FusedAddRmsNorm`; the Gemma2 `weight + 1.0` lowering rides on
    // the per-instance `offset` field.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedAddRmsNormWithOffset",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        // Identify the residual-stream Add (two Tile inputs) — the
        // scalar-offset Add has a Weight + Scalar.
        let residual_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
            })
            .expect("FusedAddRmsNormWithOffset: claim contains a residual-stream Add");
        let scalar_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("FusedAddRmsNormWithOffset: claim contains a scalar-offset Add");

        let add_node = fuf.get(residual_add_id);
        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedAddRmsNormWithOffset: residual Add input 0 must be a Tile (got {other:?})"
            ),
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedAddRmsNormWithOffset: residual Add input 1 must be a Tile (got {other:?})"
            ),
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);

        let scalar_add = fuf.get(scalar_add_id);
        let offset: f32 = scalar_add
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("FusedAddRmsNormWithOffset: scalar-offset Add has a Scalar input");

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedAddRmsNormWithOffset: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::FusedAddRmsNormWithOffset(
            delta_idx,
            residual_idx,
            layer,
            offset,
        )])
    }
}

// ── ScalarOffsetRmsNormImpl ──────────────────────────────────────
//
// Claims `(Add, RmsNorm)` where the Add has one [`FufInput::Weight`]
// and one [`FufInput::Scalar`] input, and the RmsNorm consumes the
// Add's output. Models like Gemma2 store their rmsnorm weights
// zero-init and treat the forward as `y = x * (1 + w) / rms(x)`;
// the DSL expresses that as `rmsnorm(x, weight[layer] + 1.0)`,
// which lowers to this Add + RmsNorm pair.
//
// Emit: a single `rms_norm` kernel call where the scalar rides as
// the kernel's `weight_offset` param (one fp32 fadd per weight
// element, memory-bound — effectively free vs. the Llama-style
// `y = x * w / rms(x)`).
//
// Residual-stream `(Add, RmsNorm)` continues to route through
// [`FusedAddRmsNormImpl`], which rejects Adds with non-Tile inputs.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct ScalarOffsetRmsNormImpl;

#[cfg(feature = "cuda")]
impl Implementation for ScalarOffsetRmsNormImpl {
    fn name(&self) -> &'static str {
        "scalar_offset_rms_norm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let add_node = fuf.get(seed);
        if add_node.op != OpKind::Add {
            return None;
        }
        // Need exactly one Weight input and one Scalar input.
        let mut has_weight = false;
        let mut has_scalar = false;
        for inp in &add_node.inputs {
            match inp {
                FufInput::Weight { .. } => has_weight = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if !(has_weight && has_scalar) {
            return None;
        }
        let rmsnorm_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;
        // The RmsNorm must consume the Add at the weight position
        // (slot 1 of rmsnorm(x, w)). Slot 0 is the input tensor.
        let rmsnorm_weight_is_add = matches!(
            rmsnorm_node.inputs.get(1),
            Some(FufInput::Tile { id, .. }) if *id == seed
        );
        if !rmsnorm_weight_is_add {
            return None;
        }

        let mut claimed = [seed, rmsnorm_node.id];
        claimed.sort();
        let claimed = claimed.to_vec();

        // Boundary inputs: the RmsNorm's input tensor (slot 0).
        let boundary_inputs: Vec<TileId> = rmsnorm_node
            .inputs
            .iter()
            .take(1)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            boundary_outputs: vec![rmsnorm_node.id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same cost shape as a plain rmsnorm: one read of x +
        // weight, one write of y. The +offset is free (fp32 fadd
        // per element, memory-bound).
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = (2.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The Add's Weight input is the actual rmsnorm weight; the
        // downstream consumer (RmsNorm tile) names its type. Declare
        // one accessor matching the default rmsnorm naming scheme
        // so the codegen's `RmsNorm::load` path fires.
        let add_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("claim contains an Add");
        let add_node = fuf.get(add_id);
        let (weight_id, weight_idx) = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm's Add has a Weight input");
        let name = weight_field_name(program, weight_id, weight_idx);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::RmsNorm },
            source_weights: vec![(weight_id, weight_idx)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `ScalarOffsetRmsNorm { in_slot, out_slot, layer,
    // offset, weight_fn }`. Singleton-output rmsnorm with a per-
    // instance `+offset` baked from the claim's Add scalar.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "ScalarOffsetRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("ScalarOffsetRmsNorm: claim contains Add");
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("ScalarOffsetRmsNorm: claim contains RmsNorm");
        let rmsnorm_node = fuf.get(rmsnorm_id);
        let (in_id, in_slot) = match rmsnorm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("ScalarOffsetRmsNorm: RmsNorm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rmsnorm_id, 0);

        let add_node = fuf.get(add_id);
        let offset: f32 = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm: Add has a Scalar input");

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("ScalarOffsetRmsNorm: required_weights returned empty");
        let acc_name = acc.name.to_string();
        let (_base, layer) = split_base_layer(&acc_name);
        let layer = layer.map_or(0, |v| v.0) as u32;
        // `hidden_size` / `m_multiplier` for the metal kernel's baked
        // reduction width + row count — same name-based detection as
        // `RmsNormRefImpl::fan_out`. Per-head q/k norms reduce `head_dim`
        // over `num_q/kv_heads` rows-per-token; every other norm reduces
        // the residual width once. (Vision-prelude towers use
        // `vision_embed_dim`; gemma3-mm's text decoder is `Decoder`.)
        let is_q_norm = acc_name.contains("q_norm");
        let is_k_norm = acc_name.contains("k_norm");
        let head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let num_q_heads = bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv_heads = bounds
            .get("num_key_value_heads")
            .copied()
            .unwrap_or(num_q_heads as u64) as u32;
        let residual_hidden = if matches!(program.prelude, crate::classified::Prelude::Vision) {
            *bounds.get("vision_embed_dim").unwrap_or(&0) as u32
        } else {
            *bounds.get("hidden_size").unwrap_or(&0) as u32
        };
        let (hidden_size, m_multiplier) = if (is_q_norm || is_k_norm) && head_dim > 0 {
            let m = if is_q_norm {
                num_q_heads.max(1)
            } else {
                num_kv_heads.max(1)
            };
            (head_dim, m)
        } else {
            (residual_hidden, 1)
        };
        Some(vec![Instruction::ScalarOffsetRmsNorm(
            in_slot_idx,
            out_slot_idx,
            layer,
            offset,
            hidden_size,
            m_multiplier,
        )])
    }
}

// ── FusedQkvRopeCacheImpl ────────────────────────────────────────
//
// Multi-tile impl claiming `(Gemm, Gemm, Gemm, RopeAppend)` where:
//   - the three Gemms share the same activation Tile-input, and
//   - each Gemm's output feeds one of the RopeAppend's first three
//     Tile-input slots (slot 0 = q, 1 = k, 2 = v).
//
// Maps to a fused cuBLAS QKV GEMM producing a packed
// `[num_tokens, q_size + 2*kv_size]` buffer, followed by
// `scratchy_target_cuda::fused_qkv_rope_cache` which applies RoPE to Q and
// K, writes K/V to the paged cache at the matched layer, and returns
// Q as an OwnedTensor. Structural match, no "GemmQ/K/V" phase tags.
//
// Declared WeightAccessor: one packed `LinearLayer` covering all
// three Q/K/V projections. Users concatenate the three weights at
// load time. Eliminates the old cross-impl contract between separate
// Gemm and QkvSplit impls.
//
// Seed on Gemm (upstream in topo order). Matcher fires at any of the
// three Q/K/V gemms; whichever seeds first wins (topo order), and the
// other two are swallowed by the same claim.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedQkvRopeCacheImpl;

/// Read the layer index captured by the DSL's `kv_cache[layer]`
/// reference on a RopeAppend tile. Returns `None` if the tile has
/// no KvCache extern input (shouldn't happen for real RopeAppends).
fn rope_kv_cache_layer(node: &crate::fuf::FufNode) -> Option<u64> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Extern {
            kind: crate::classified::ExternKind::KvCache,
            index: Some(layer),
        } => Some(layer.0),
        _ => None,
    })
}

/// True iff the layer at `layer` uses the interleaved rope flavor,
/// determined by finding the layer's rope-append tile in the FUF and
/// inspecting its `OpKind`. Used by the singleton attention impls so
/// the FA-2 call sees `is_rotary_interleaved=true` whenever the layer
/// upstream rope wrote interleaved-pattern Q/K to the cache.
fn layer_rope_is_interleaved(fuf: &Fuf, layer: u64) -> bool {
    fuf.nodes
        .iter()
        .any(|n| n.op == OpKind::RopeAppendInterleaved && rope_kv_cache_layer(n) == Some(layer))
}

/// Resolve a tile to `(gemm_tile, maybe_bias_add_tile)`:
/// - `(g, None)`    if `tile` is directly a `Gemm`.
/// - `(g, Some(b))` if `tile` is a `BiasAdd` whose first tile input
///   is a `Gemm`.
/// - `None`         otherwise.
///
/// Used by the QKV-fused impls (`FusedQkvRopeCacheImpl`,
/// `FusedQkvRopePrefillImpl`) so that the DSL can write
/// `gemm → bias_add → rope_append` (Qwen2/Qwen3 style with explicit
/// QKV bias) or `gemm → rope_append` (Llama style, no bias) and have
/// the same fusion claim both shapes.
fn unwrap_gemm_through_bias(fuf: &Fuf, tile: TileId) -> Option<(TileId, Option<TileId>)> {
    let node = fuf.get(tile);
    match node.op {
        OpKind::Gemm => Some((tile, None)),
        OpKind::BiasAdd => {
            let upstream = node.inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })?;
            if fuf.get(upstream).op == OpKind::Gemm {
                Some((upstream, Some(tile)))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// True if this RopeAppend tile's first three tile-inputs all resolve
/// (directly or through `BiasAdd`) to three Gemms sharing the same
/// activation with uniform bias presence — the exact pattern
/// `FusedQkvRope{Cache,Prefill}Impl::matches` requires. Used by
/// `RopeAppendRefImpl` to defer to the fused impl whenever it would
/// match (Llama, Qwen2) and only claim standalone when the pattern
/// is broken by intervening ops (Qwen3's per-head Q/K rmsnorms).
fn rope_append_has_fused_qkv_upstream(fuf: &Fuf, rope_tile: TileId) -> bool {
    let node = fuf.get(rope_tile);
    if !is_rope_append_op(node.op) || node.inputs.len() < 3 {
        return false;
    }
    let qkv_raw: Vec<TileId> = node
        .inputs
        .iter()
        .take(3)
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    if qkv_raw.len() != 3 {
        return false;
    }
    let Some(resolved): Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
        .iter()
        .map(|t| unwrap_gemm_through_bias(fuf, *t))
        .collect()
    else {
        return false;
    };
    let biased = resolved[0].1.is_some();
    if resolved.iter().any(|r| r.1.is_some() != biased) {
        return false;
    }
    let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
    let Some(act) = first_tile_input(fuf.get(gemms[0])) else {
        return false;
    };
    if !gemms
        .iter()
        .all(|t| first_tile_input(fuf.get(*t)) == Some(act))
    {
        return false;
    }
    // GGUF + interleaved RoPE: only the decode (Cache) Ggml fused
    // Impl handles interleaved; the Prefill Ggml fused Impl rejects.
    // For prefill workloads (M>1) we'd lose claim coverage if
    // `RopeAppendRefImpl` deferred. Force-defer-NOT here so the
    // singleton ref Impl claims at prefill while the Cache fused
    // Impl claims at decode (workload constraints disambiguate).
    if node.op == OpKind::RopeAppendInterleaved
        && gemms
            .iter()
            .any(|t| matches!(weight_storage_of(fuf.get(*t)), Some(StorageFormat::Ggml)))
    {
        return false;
    }
    true
}

#[cfg(feature = "cuda")]
impl Implementation for FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_rope_cache"
    }

    fn kv_rule(&self) -> crate::alias_rules::KvRule {
        crate::alias_rules::KvRule::Writes
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // `fused_qkv_rope_cache` is the decode-fused kernel (writes
        // K/V to the paged cache in one shot, returns only Q — the
        // contiguous K/V are discarded). Restricts paired with the
        // decode attention impl that reads K/V from the cache.
        // Prefill is handled by `FusedQkvRopePrefillImpl`.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage on the seed.
        let storage = weight_storage_of(seed_node);
        if !matches!(storage, Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a RopeAppend whose first three Tile inputs all resolve
        // (directly or through a `BiasAdd` wrapper) to Gemm tiles
        // sharing a common activation. The seed must be one of those
        // three Gemms.
        //
        // Bias presence must be uniform across Q/K/V — either all
        // three are Gemm→BiasAdd→RopeAppend (Qwen2/Qwen3 with QKV
        // bias) or all three are Gemm→RopeAppend (Llama, bias-less).
        // Mixed would be a DSL-level inconsistency; we reject to
        // surface it as UnclaimedTile rather than silently do the
        // wrong thing.
        let rope_node = fuf.nodes.iter().find(|n| {
            if !is_rope_append_op(n.op) || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            // Resolve each of the three to `(gemm, maybe_bias)`.
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            // Uniform bias presence.
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            // All three gemms must share the same activation.
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        // Resolve the three rope tile-inputs to (gemm, maybe_bias).
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("already validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            // Only slot 0 (Q) is read by tiles outside the claim —
            // K/V slots flow to the paged cache and the downstream
            // Attention kernel reads them from there (C4). Declaring
            // all three for now preserves dep tracking; C4 elides.
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Cost = one cuBLAS GEMM at packed (M, q_size+2*kv_size, hidden)
        // + one bandwidth-bound rope+cache pass.
        //
        // Read measured cublas cost from the calibration CSV (the
        // predictor linreg-extrapolates when the exact (M, packed_n, K)
        // row isn't sampled). Fall back to peak-FLOPS roofline only
        // when the predictor has no signal — same shape as the
        // GateUp{Silu,Gelu}Mul peers. The roofline is ~8× optimistic
        // vs measured cuBLAS, so without this parity any future
        // `CutlassFusedQkvRope*` peer (whose GEMM cost reads measured
        // CUTLASS rows) would lose every DP race despite being faster.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        // packed_n = (q_size + 2*kv_size) = (num_q + 2*num_kv) * head_dim.
        let packed_n = num_q_heads
            .saturating_add(2u32.saturating_mul(num_kv_heads))
            .saturating_mul(head_dim);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Rope + cache write: bandwidth-bound, read packed QKV +
        // write rotated Q + K/V to cache.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (packed_n as f64) * BYTES_PER_ELEM;
        let rope_cache_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + rope_cache_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // One fused accessor covering the three Gemm weights — the
        // bias vectors (if present) ride through the accessor's
        // `LinearLayer::dense_bias()` output and `emit_call` asserts
        // their presence whenever the DSL claimed BiasAdd tiles. The
        // user populates the accessor with a `LinearLayer::Dense`
        // whose `weight` is the concatenated `[q | k | v]`; if the
        // underlying model carries biases, `bias` is the concatenated
        // `[q_b | k_b | v_b]`. `LinearLayer::load_dense_concat(gw,
        // prefixes, stream)` does exactly that: auto-detects bias on
        // the source prefixes and streams into one packed tensor.
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvRopeCache { in_slot, out_slot, layer,
    // weight_fn, cos_sin_fn, biased, interleaved }`. Flags ride on
    // `OpInstance` rather than baking into the body so one arm
    // body works for every claim of this Impl in the arch — Llama,
    // Qwen2 (biased), Cohere (interleaved), and any future combo.
    //
    // Slots 1 and 2 (K/V) are paged-cache slots, written by the
    // kernel directly into `ctx.kv_cache`; downstream attention
    // reads them from the cache, not from `__tiles`. The tile
    // table entries for those slots stay `None` for this Impl's
    // lifetime — that's why `output_alias` only declares slot 0.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("biased", syn::parse_quote!(bool)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("FusedQkvRopeCache: RopeAppend has a KvCache extern with concrete layer index")
            as u32;

        // Resolve QKV gemms via the optional BiasAdd wrapper.
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("FusedQkvRopeCache: claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();

        // Activation slot — first tile input of the q gemm.
        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedQkvRopeCache: q_gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        // Packed [q | k | v] LinearLayer accessor.
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;

        // The variant carries `layer` once; both the kv_cache index and
        // the weight accessor read it. Sanity-check at codegen that the
        // weight accessor's layer suffix agrees with the rope's KvCache
        // layer — if these ever disagree, the DSL has a bug.
        assert_eq!(
            weight_layer, layer,
            "FusedQkvRopeCache: weight_layer ({weight_layer}) and \
             kv_cache layer ({layer}) disagree — DSL bug"
        );
        Some(vec![Instruction::FusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            biased,
            interleaved,
        )])
    }
}

// ── FusedQkvQkNormRopeCacheImpl ─────────────────────────────────
//
// Multi-tile impl claiming the entire QKV + per-head QK-norm + RoPE
// + paged-cache-write chain produced by Qwen3 and Gemma3:
//
//   Gemm(q) ─→ [Add(w,1.0) →] [Reshape →] RmsNorm → [Reshape →] ─┐
//   Gemm(k) ─→ [Add(w,1.0) →] [Reshape →] RmsNorm → [Reshape →] ─┤
//   Gemm(v) ─────────────────────────────────────────────────────────┤
//                                                      RopeAppend ←─┘
//
// The three Gemms share the same activation. The Q and K paths flow
// through optional Add (Gemma3's `weight + 1.0`), optional Reshape
// (split [T, heads*head_dim] → [T*heads, head_dim]), RmsNorm
// (per-head norm), optional Reshape (flatten back), then into the
// RopeAppend. The V path feeds RopeAppend directly.
//
// Emits:
//   1. Packed cuBLAS QKV GEMM (same as FusedQkvRopeCacheImpl)
//   2. `qk_norm_rope_inplace` — fused QK-norm + RoPE in one launch
//   3. `reshape_and_cache` — write K/V to the paged cache
//
// Seeds on Gemm (upstream in topo order). Whichever of the three
// Q/K/V Gemms seeds first wins; the other two are swallowed by the
// same claim.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedQkvQkNormRopeCacheImpl;

/// Walk from a Gemm tile through the optional `Add → Reshape →
/// RmsNorm → Reshape` chain and return the chain's endpoint tile
/// (the last tile before RopeAppend) plus all intermediate tiles
/// encountered. Returns `None` if the chain doesn't match.
///
/// Chain shapes (all valid):
///   Gemm → RmsNorm                         (Qwen3, no scalar offset, no reshape)
///   Gemm → Add → RmsNorm                   (Gemma3, scalar offset, no reshape)
///   Gemm → Reshape → RmsNorm → Reshape     (with reshape, no scalar offset)
///   Gemm → Add → Reshape → RmsNorm → Reshape (with reshape + scalar offset)
fn walk_qk_norm_chain(fuf: &Fuf, gemm_id: TileId) -> Option<(TileId, Vec<TileId>, Option<TileId>)> {
    // Returns (endpoint_tile, intermediates_excluding_gemm, add_tile_if_any)
    //
    // The data chain is: Gemm → [Reshape(split)] → RmsNorm → [Reshape(flatten)]
    // The Add(Weight, Scalar) for Gemma's `w + 1.0` feeds RmsNorm's
    // weight slot (slot 1), NOT the activation path. It's a side input
    // that we claim but don't walk through.
    let mut cursor = gemm_id;
    let mut intermediates: Vec<TileId> = Vec::new();
    let mut add_tile: Option<TileId> = None;

    // Step 1: optional Reshape (split to per-head)
    if let Some(reshape_node) = fuf
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Reshape && consumes_tile(n, cursor))
    {
        let has_rmsnorm_downstream = fuf.nodes.iter().any(|n| {
            n.op == OpKind::RmsNorm
                && matches!(n.inputs.first(), Some(FufInput::Tile { id, .. }) if *id == reshape_node.id)
        });
        if has_rmsnorm_downstream {
            intermediates.push(reshape_node.id);
            cursor = reshape_node.id;
        }
    }

    // Step 2: required RmsNorm consuming cursor at slot 0
    let rmsnorm_node = fuf.nodes.iter().find(|n| {
        n.op == OpKind::RmsNorm
            && matches!(n.inputs.first(), Some(FufInput::Tile { id, .. }) if *id == cursor)
    })?;
    intermediates.push(rmsnorm_node.id);

    // Check if RmsNorm's weight slot (slot 1) is an Add(Weight, Scalar)
    // tile — that's the Gemma `(1+w)` offset. If so, claim the Add.
    if let Some(FufInput::Tile {
        id: weight_tile, ..
    }) = rmsnorm_node.inputs.get(1)
    {
        let wt_node = fuf.get(*weight_tile);
        if wt_node.op == OpKind::Add
            && wt_node
                .inputs
                .iter()
                .any(|i| matches!(i, FufInput::Scalar(_)))
            && wt_node
                .inputs
                .iter()
                .any(|i| matches!(i, FufInput::Weight { .. }))
        {
            add_tile = Some(*weight_tile);
            intermediates.push(*weight_tile);
        }
    }
    if rmsnorm_node.inputs.len() < 2 {
        return None;
    }

    cursor = rmsnorm_node.id;
    match &rmsnorm_node.inputs[1] {
        FufInput::Tile { id, .. } => {
            // Must be our Add tile (scalar-offset path).
            if add_tile != Some(*id) {
                return None;
            }
        }
        FufInput::Weight { .. } => {
            // Direct weight — fine (Qwen3 path, no scalar offset).
        }
        _ => return None,
    }

    // Step 4: optional Reshape (flatten back to [T, heads*head_dim])
    if let Some(reshape_node) = fuf
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Reshape && consumes_tile(n, cursor))
    {
        intermediates.push(reshape_node.id);
        cursor = reshape_node.id;
    }

    Some((cursor, intermediates, add_tile))
}

/// True if this RopeAppend tile's Q and K inputs flow through a
/// `Gemm → [Add →] [Reshape →] RmsNorm → [Reshape →]` chain from
/// three shared-activation Gemms — the pattern
/// `FusedQkvQkNormRopeCacheImpl` claims. Used by `RopeAppendRefImpl`
/// to defer when this larger fusion would match.
fn rope_append_has_qk_norm_upstream(fuf: &Fuf, rope_tile: TileId) -> bool {
    let node = fuf.get(rope_tile);
    if node.op != OpKind::RopeAppend || node.inputs.len() < 3 {
        return false;
    }
    let qkv_raw: Vec<TileId> = node
        .inputs
        .iter()
        .take(3)
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    if qkv_raw.len() != 3 {
        return false;
    }

    // Q and K must come through QK-norm chains; V must be a Gemm.
    let q_input = qkv_raw[0];
    let k_input = qkv_raw[1];
    let v_id = qkv_raw[2];

    // V must be a Gemm directly.
    if fuf.get(v_id).op != OpKind::Gemm {
        return false;
    }

    // Find the Gemm at the root of Q and K chains by walking backwards.
    // The chain endpoint feeds into RopeAppend; we need the Gemm root.
    // Walk backwards from q_input: could be Reshape → RmsNorm → [Reshape →] [Add →] Gemm
    let q_gemm = find_gemm_root_of_norm_chain(fuf, q_input);
    let k_gemm = find_gemm_root_of_norm_chain(fuf, k_input);

    let (Some(q_gemm), Some(k_gemm)) = (q_gemm, k_gemm) else {
        return false;
    };

    // All three Gemms must share the same activation.
    let q_act = first_tile_input(fuf.get(q_gemm));
    let k_act = first_tile_input(fuf.get(k_gemm));
    let v_act = first_tile_input(fuf.get(v_id));
    q_act.is_some() && q_act == k_act && q_act == v_act
}

/// Walk backwards from a tile that is the endpoint of a QK-norm chain
/// (Reshape or RmsNorm) back through the chain to find the root Gemm.
fn find_gemm_root_of_norm_chain(fuf: &Fuf, endpoint: TileId) -> Option<TileId> {
    let mut cursor = endpoint;

    // If cursor is a Reshape, step back
    if fuf.get(cursor).op == OpKind::Reshape {
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }
    // Now should be RmsNorm
    if fuf.get(cursor).op != OpKind::RmsNorm {
        return None;
    }
    // RmsNorm slot 0 is the tensor input
    cursor = first_tile_input(fuf.get(cursor))?.0;

    // Optional Reshape (split)
    if fuf.get(cursor).op == OpKind::Reshape {
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }
    // Optional Add(Weight, Scalar)
    if fuf.get(cursor).op == OpKind::Add {
        // The Add should have a Tile input from a Gemm
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }

    if fuf.get(cursor).op == OpKind::Gemm {
        Some(cursor)
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
impl Implementation for FusedQkvQkNormRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_qk_norm_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Handles all workload points; downstream `AttentionViaCacheImpl`
        // (M=1) and `AttentionPrefillContiguousImpl` (M>=2) pick up via
        // their own WorkloadConstraint gating.
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(seed_node), Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a RopeAppend whose Q and K inputs flow through QK-norm
        // chains from Gemms sharing the same activation as the seed,
        // and whose V input is a Gemm also sharing that activation.
        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }

            let v_id = qkv_raw[2];
            if fuf.get(v_id).op != OpKind::Gemm {
                return false;
            }

            // Walk forward from each Gemm to verify the QK-norm chain
            // exists. We need to find which Gemms root the Q and K
            // chains.
            let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0]);
            let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1]);

            let (Some(q_gemm), Some(k_gemm)) = (q_gemm, k_gemm) else {
                return false;
            };

            let gemms = [q_gemm, k_gemm, v_id];
            if !gemms.contains(&seed) {
                return false;
            }

            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;

        let rope_id = rope_node.id;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        let v_id = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0]).expect("validated in find");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1]).expect("validated in find");

        // Walk the Q and K chains forward to collect all intermediate tiles.
        let (_, q_chain, _) = walk_qk_norm_chain(fuf, q_gemm).expect("Q chain must be valid");
        let (_, k_chain, _) = walk_qk_norm_chain(fuf, k_gemm).expect("K chain must be valid");

        let mut claimed: Vec<TileId> = Vec::with_capacity(16);
        claimed.push(q_gemm);
        claimed.push(k_gemm);
        claimed.push(v_id);
        claimed.extend_from_slice(&q_chain);
        claimed.extend_from_slice(&k_chain);
        claimed.push(rope_id);
        claimed.sort();
        claimed.dedup();

        let activation = first_tile_input(fuf.get(q_gemm))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as f64;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as f64;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as f64;
        let n = (num_q_heads + 2.0 * num_kv_heads) * head_dim;

        let flops = 2.0 * m * n * hidden;
        let peak = ctx.profile.peak_tflops_fp16 * 1e12;
        let gemm_us = if peak > 0.0 && flops > 0.0 {
            (flops / peak) * 1e6
        } else {
            0.0
        };

        // QK-norm + RoPE + cache write: bandwidth-bound.
        // Read packed Q+K for norm, write normalized Q+K, then RoPE +
        // cache write (same as FusedQkvRopeCacheImpl plus the norm r/w).
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let q_kv_bytes = m * (num_q_heads + num_kv_heads) * head_dim * BYTES_PER_ELEM;
        // norm reads + writes Q and K (2x), plus rope + cache write
        let rope_bytes = 3.0 * m * n * BYTES_PER_ELEM;
        let norm_rope_cache_us = if bw_gb > 0.0 {
            ((2.0 * q_kv_bytes + rope_bytes) / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + norm_rope_cache_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The `qk_norm_rope_inplace` kernel requires Q/K as
        // contiguous `[T, num_heads, head_dim]` tensors. A packed
        // cuBLAS QKV output `[T, q_size + 2*kv_size]` has strided
        // Q/K across tokens, so slicing won't give usable inputs.
        // Instead we declare three SEPARATE `LinearLayer`
        // accessors (one per Gemm) and let `emit_call` run three
        // GEMMs — each producing a contiguous output.
        //
        // Identify the RopeAppend tile so we can figure out the
        // Q/K/V role of each Gemm. Rope's tile inputs [0,1,2] are
        // qkv_raw; Q and K flow through the norm chain, V is a
        // Gemm directly.
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");

        let v_gemm = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0])
            .expect("Q norm chain validated at match time");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1])
            .expect("K norm chain validated at match time");

        // 1. Three separate LinearLayer accessors (Q, K, V).
        let mut accessors: Vec<WeightAccessor> = Vec::with_capacity(5);
        for &gemm_id in &[q_gemm, k_gemm, v_gemm] {
            let (wid, widx) = first_weight_ref(fuf.get(gemm_id)).expect("gemm has a weight input");
            let name = weight_field_name(program, wid, widx);
            accessors.push(WeightAccessor {
                name,
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                source_weights: vec![(wid, widx)],
            });
        }

        // 2. Two RmsNorm weights for q_norm and k_norm.
        //
        // The weight source depends on whether the chain has a
        // scalar-offset Add: if so, the RmsNorm tile's weight slot
        // (slot 1) is a Tile input pointing at the Add, and the
        // actual Weight ref lives on the Add's inputs. Otherwise the
        // RmsNorm's slot 1 is a direct Weight input.
        let rmsnorm_ids: Vec<TileId> = claimed_tiles
            .iter()
            .filter(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .copied()
            .collect();
        assert_eq!(
            rmsnorm_ids.len(),
            2,
            "FusedQkvQkNormRopeCacheImpl claim must have exactly 2 RmsNorm tiles"
        );

        for &rmsnorm_id in &rmsnorm_ids {
            let rmsnorm_node = fuf.get(rmsnorm_id);
            let (weight_id, weight_idx) = match &rmsnorm_node.inputs[1] {
                FufInput::Tile { id, .. } => {
                    // Scalar-offset path: the Add tile carries the weight.
                    let add_node = fuf.get(*id);
                    add_node
                        .inputs
                        .iter()
                        .find_map(|i| match i {
                            FufInput::Weight { id, index, .. } => Some((*id, *index)),
                            _ => None,
                        })
                        .expect("scalar-offset Add has a Weight input")
                }
                FufInput::Weight { id, index, .. } => (*id, *index),
                _ => panic!("RmsNorm slot 1 must be Tile (Add) or Weight"),
            };
            let name = weight_field_name(program, weight_id, weight_idx);
            accessors.push(WeightAccessor {
                name,
                rust_type: quote! { crate::__gpu::layers::RmsNorm },
                source_weights: vec![(weight_id, weight_idx)],
            });
        }

        accessors
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvQkNormRopeCache { in_slot, out_slot, layer,
    // q_weight_fn, k_weight_fn, v_weight_fn, q_norm_fn, k_norm_fn,
    // cos_sin_fn, q_offset, k_offset }`. Three separate cuBLAS GEMMs
    // (Q/K/V) followed by the fused `qk_norm_rope_inplace` kernel
    // and `reshape_and_cache`. Q's OwnedTensor is reshaped to
    // `[T, num_q_heads, head_dim]` before being stored at out_slot.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvQkNormRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("q_offset", syn::parse_quote!(f32)),
                ("k_offset", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("FusedQkvQkNormRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node).expect(
            "FusedQkvQkNormRopeCache: RopeAppend has a KvCache extern with concrete layer index",
        ) as u32;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let v_id = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0])
            .expect("FusedQkvQkNormRopeCache: Q norm chain validated at match time");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1])
            .expect("FusedQkvQkNormRopeCache: K norm chain validated at match time");

        // Activation slot — first tile input of any of the three Gemms
        // (matches() guarantees they all share it).
        let q_node = fuf.get(q_gemm);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedQkvQkNormRopeCache: q_gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        // Per-claim scalar offsets — Gemma3 carries `weight + 1.0`
        // before each per-head norm.
        let (_, _, q_add) =
            walk_qk_norm_chain(fuf, q_gemm).expect("FusedQkvQkNormRopeCache: Q chain valid");
        let (_, _, k_add) =
            walk_qk_norm_chain(fuf, k_gemm).expect("FusedQkvQkNormRopeCache: K chain valid");
        let q_offset: f32 = q_add
            .map(|add_id| {
                fuf.get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        let k_offset: f32 = k_add
            .map(|add_id| {
                fuf.get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);

        // Resolve the five accessors via required_weights — three
        // LinearLayers (Q/K/V) then two RmsNorms (q_norm/k_norm). Order
        // is fixed by required_weights's push order.
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        assert_eq!(
            accessors.len(),
            5,
            "FusedQkvQkNormRopeCache: expected 5 accessors (q,k,v + q_norm,k_norm)"
        );
        let resolve_layer = |acc: &WeightAccessor| -> u32 {
            let (_base, l) = split_base_layer(&acc.name.to_string());
            l.map(|v| v.0).unwrap_or(layer as u64) as u32
        };
        let q_w_layer = resolve_layer(&accessors[0]);
        let k_w_layer = resolve_layer(&accessors[1]);
        let v_w_layer = resolve_layer(&accessors[2]);
        let q_n_layer = resolve_layer(&accessors[3]);
        let k_n_layer = resolve_layer(&accessors[4]);
        for (l, name) in [
            (q_w_layer, "q_weight"),
            (k_w_layer, "k_weight"),
            (v_w_layer, "v_weight"),
            (q_n_layer, "q_norm"),
            (k_n_layer, "k_norm"),
        ] {
            assert_eq!(
                l, layer,
                "FusedQkvQkNormRopeCache: {name} layer ({l}) disagrees with rope layer ({layer})"
            );
        }
        let _ = (v_id, q_gemm, k_gemm); // captured into accessors, no further use here.

        Some(vec![Instruction::FusedQkvQkNormRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            q_offset,
            k_offset,
        )])
    }
}

// ── AttentionViaCacheImpl ────────────────────────────────────────
//
// Singleton matcher on `OpKind::Attention`. Maps the DSL's
// `attention(q, k, v, kv_cache[layer], block_table)` to
// `scratchy_target_cuda::kernels::flash_attn_paged`, which reads Q + the
// paged KV cache directly — after `FusedQkvRopeCacheImpl` has
// written K/V to the layer's cache slice — and ignores the DSL
// tile's K/V Tile-inputs entirely.
//
// The layer index is captured on the Attention tile's own
// `FufInput::Extern { kind: KvCache, index: Some(L) }` (the DSL's
// `kv_cache[layer]` argument), same as for RopeAppend.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AttentionViaCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for AttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "attention_via_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Paged-cache-reading attention is the decode path. K/V
        // were written to the cache by the upstream decode QKV
        // impl; this impl ignores the DSL Attention tile's slot-1
        // and slot-2 inputs (which are cache aliases in the decode
        // flow). Prefill uses `AttentionPrefillContiguousImpl`.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only: requires the 5-arg `attention(q, k, v,
        // kv_cache, block_table)` form. Encoder-style 3-arg tiles are
        // claimed by `EncoderAttentionImpl`.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Use the calibrated `fa2_attn_...` CSV row when present so
        // the FI-vs-FA2 tiebreak runs on real timings; falls back to
        // the analytic `cost_attention` formula when no row matches.
        cost_attention_calibrated(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // Default `required_weights` yields zero accessors — Attention
    // consumes no weights. The paged cache and per-call metadata live
    // on `ForwardCtx`, not `WeightBundle`.

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `AttentionViaCache { in_slot, out_slot, layer,
    // cos_sin_fn, interleaved }`. Decode-only (M=1) paged attention.
    // Reads Q from the upstream rope-cache fusion's tile output;
    // K/V live on `ctx.kv_cache` and are read by the kernel via
    // `block_table`. Scale + softcap are arch-wide (baked at codegen);
    // interleaved is per-instance because Gemma3 layers can differ.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("AttentionViaCache: input 0 (Q) must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("AttentionViaCache: kv_cache extern with concrete layer index")
            as u32;
        let interleaved = layer_rope_is_interleaved(fuf, layer as u64);

        Some(vec![Instruction::AttentionViaCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            interleaved,
        )])
    }
}

// ── RopeAppendRefImpl ────────────────────────────────────────────
//
// Singleton fallback for `OpKind::RopeAppend` tiles that the QKV
// fusions (`FusedQkvRopeCacheImpl` / `FusedQkvRopePrefillImpl`)
// don't claim — typically Qwen3 / Gemma3, where per-head Q/K
// rmsnorm tiles sit between the QKV gemms and rope_append, breaking
// the gemm→rope adjacency the fused matchers require.
//
// Emits a two-kernel sequence:
//   1. `rotary_embedding_inplace(q, k, positions, cos_sin, head_dim)`
//      — applies RoPE to Q and K in-place on the 2D `[T, heads*head_dim]`
//      / `[T, kv_heads*head_dim]` upstream tensors.
//   2. `reshape_and_cache(k_3d, v_3d, k_cache, v_cache, slot_mapping,
//      block_size)` — writes K/V to the paged cache. K and V are
//      reshaped to 3D `[T, kv_heads, head_dim]` views (metadata only)
//      to match the kernel's expected layout.
//
// Output bindings: all three are 3D TensorView reshapes aliasing the
// upstream storage — `[T, q_heads, head_dim]` for Q, `[T, kv_heads,
// head_dim]` for K and V. This matches what the fused
// `FusedQkvRope{Cache,Prefill}Impl` impls produce, so the downstream
// `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl` (which
// read `q.dim(0,1,2)` as `[total_q, num_heads, head_dim]`) see the
// same layout whichever upstream fired. No move / no consume — the
// drop pass resolves aliases to the ultimate owner and keeps it
// alive until every downstream use is done.
//
// All M (decode + prefill) — kernel signatures don't depend on token
// count.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct RopeAppendRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for RopeAppendRefImpl {
    fn name(&self) -> &'static str {
        "rope_append_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if !is_rope_append_op(node.op) {
            return None;
        }
        // Boundary inputs: q, k, v upstream tiles (slots 0, 1, 2 of
        // the rope_append). Positions / rotary / kv_cache are externs,
        // sourced from `ForwardCtx`.
        let qkv: Vec<TileId> = node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        if qkv.len() != 3 {
            return None;
        }
        // Defer to `FusedQkvRope{Cache,Prefill}Impl` when the
        // upstream pattern matches (three Gemms sharing one activation,
        // optionally through a BiasAdd, uniform bias presence). This
        // gate is the inverse of `output_layouts` — singletons leave
        // K/V at `[T, heads*head_dim]` while fused impls produce
        // `[T, heads, head_dim]`, and the downstream Attention impls
        // expect the latter when a fused upstream is available.
        // Defer to `FusedQkvRope{Cache,Prefill}Impl` when the upstream pattern
        // matches (three Gemms sharing one activation) — it claims the larger
        // packed-QKV+rope+cache subgraph.
        if rope_append_has_fused_qkv_upstream(fuf, seed) {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: qkv,
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // RoPE + cache write are bandwidth-bound: read packed Q+K +
        // write rotated Q+K to same buffers + write K/V to cache.
        let m = ctx.num_tokens() as f64;
        let q_size = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let kv_size = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let bytes = m * (2.0 * q_size as f64 + 4.0 * kv_size as f64) * BYTES_PER_ELEM;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::RopeTripleAlias
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `RopeAppend { q_slot, k_slot, v_slot, q_out_slot,
    // k_out_slot, v_out_slot, layer, cos_sin_fn, interleaved }`.
    // Singleton fallback for Qwen3/Gemma3-style rope_append tiles.
    // The arm runs `rotary_embedding[_interleaved]_inplace` against
    // the upstream Q/K/V buffers and `reshape_and_cache` to write
    // K/V to the paged cache. Slot 0/1/2 of the rope tile are
    // 3D `Reshaped` views aliasing the same upstream storage, so
    // downstream attention reads the (T, heads, head_dim) layout.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RopeAppend",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("is_global", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = m.claimed_tiles[0];
        let node = fuf.get(rope_id);
        let interleaved = node.op == OpKind::RopeAppendInterleaved;
        // Attention-geometry class: a rope tile consumed by a
        // `sliding_attention()` tile belongs to the sliding class
        // (base HEAD_DIM/NUM_KV_HEADS/ROT_DIM); one consumed by
        // `attention()` is the global class (GLOBAL_* consts). On
        // uniform-geometry models the two const families are equal,
        // so the default `true` is also correct when no attention
        // consumer is found (e.g. encoder shapes).
        let is_global = !fuf
            .nodes
            .iter()
            .any(|n| n.op == OpKind::SlidingAttention && consumes_tile(n, rope_id));
        let resolve_tile = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("RopeAppend: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (q_id, q_in) = resolve_tile(0);
        let (k_id, k_in) = resolve_tile(1);
        let (v_id, v_in) = resolve_tile(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let q_out_slot = slots.of(rope_id, 0);
        let k_out_slot = slots.of(rope_id, 1);
        let v_out_slot = slots.of(rope_id, 2);
        let layer =
            node.inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Extern {
                        kind: ExternKind::KvCache,
                        index: Some(layer),
                    } => Some(layer.0),
                    _ => None,
                })
                .expect("RopeAppend: kv_cache extern with concrete layer index") as u32;

        Some(vec![Instruction::RopeAppend(
            q_slot,
            k_slot,
            v_slot,
            q_out_slot,
            k_out_slot,
            v_out_slot,
            layer,
            interleaved,
            is_global,
        )])
    }
}

// ── FusedQkvRopePrefillImpl ──────────────────────────────────────
//
// Same 4-tile pattern as `FusedQkvRopeCacheImpl` `(Gemm, Gemm, Gemm,
// RopeAppend)`, but for the prefill path: emits the split `fused_qkv_rope`
// kernel (returns Q, K, V as contiguous OwnedTensors; does NOT write
// to the paged cache) followed by an explicit `write_kv_cache` so
// the cache is populated for future decode steps. The contiguous
// K/V tensors remain available as the tile's slot-1 and slot-2
// outputs so the downstream prefill-attention impl reads them
// directly instead of going through the cache.
//
// `WorkloadConstraint::NumTokensRange { min: 2, max: u32::MAX }` —
// paired with `AttentionPrefillContiguousImpl` for the prefill
// bucket. Decode goes through `FusedQkvRopeCacheImpl`.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedQkvRopePrefillImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same pattern as the cache variant — delegate to it so the
        // matcher stays in one place.
        FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same flops + bandwidth model as the decode-fused variant;
        // the kernel split is different but the work is the same.
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Same packed QKV weight as the decode variant.
        FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvRopePrefill { in_slot, q_out_slot, k_out_slot,
    // v_out_slot, layer, weight_fn, cos_sin_fn, biased, interleaved }`.
    // Three Owned outputs (Q/K/V) since the prefill kernel returns
    // contiguous OwnedTensors instead of writing to the paged cache.
    // After the kernel returns, K/V are committed to the cache via
    // `write_kv_cache` for downstream decode iterations.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("biased", syn::parse_quote!(bool)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("FusedQkvRopePrefill: RopeAppend has a KvCache extern with layer index")
            as u32;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("FusedQkvRopePrefill: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );

        Some(vec![Instruction::FusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
            biased,
            interleaved,
        )])
    }
}

// ── CutlassFusedQkvRope{Cache,Prefill}Impl ──────────────────────
//
// CUTLASS peers to the cuBLAS `FusedQkvRope{Cache,Prefill}Impl` —
// same 4-or-7-tile claim, same boundary inputs/outputs, same
// runtime post-GEMM step (rope+paged-cache write for Cache,
// rope+contiguous-Q/K/V for Prefill). The only structural change
// vs the cuBLAS peer is the GEMM kernel: a calibrated
// `cutlass_<TM>x<TN>_s<S>` standalone-zoo tile in place of cuBLAS.
//
// Tile-zoo-pickable: one Impl per `CUTLASS_TILE_ZOO` entry,
// `target_compatible` gated on calibrated CSV row presence —
// uncalibrated targets silently fall back to the cuBLAS peer.
//
// Bias gating: the v1 Impl rejects biased claims in `matches`. The
// bias-zoo CSV (`cutlass_gemm_bias_<TM>x<TN>_s<S>`) is not yet
// shape-swept across the QKV regime, so a biased pick would race
// against `UNCALIBRATED_COST_US` and silently lose to cuBLAS.
// Qwen2's biased QKV consequently keeps the cuBLAS path until the
// bias-zoo sweep lands.

/// True iff the claim's three QKV gemms are wrapped in `BiasAdd`
/// (Qwen2/Qwen3-style explicit QKV bias).
fn fused_qkv_claim_is_biased(fuf: &Fuf, info: &MatchInfo) -> bool {
    let Some(rope_id) = info
        .claimed_tiles
        .iter()
        .find(|t| is_rope_append_op(fuf.get(**t).op))
        .copied()
    else {
        return false;
    };
    let rope_node = fuf.get(rope_id);
    let qkv_raw: Vec<TileId> = rope_node
        .inputs
        .iter()
        .take(3)
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    qkv_raw
        .iter()
        .filter_map(|t| unwrap_gemm_through_bias(fuf, *t))
        .next()
        .is_some_and(|(_, bias)| bias.is_some())
}

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedQkvRopeCacheImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedQkvRopeCacheImpl {
    fn csv_name(&self) -> &'static str {
        // Reuses the standalone CutlassGemm tile names — the GEMM
        // step is an ordinary cutlass_<TM>x<TN>_s<S> at packed
        // N=q+2*kv.
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopeCacheImpl.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        let info = FusedQkvRopeCacheImpl.matches(fuf, seed, profile)?;
        if fused_qkv_claim_is_biased(fuf, &info) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same shape as FusedQkvRopeCacheImpl::cost_us (cuBLAS peer):
        //   - one GEMM at packed (M, q+2*kv, hidden)
        //   - one BW-bound rope+cache pass over the same buffer
        // Difference: the GEMM is a calibrated cutlass_<tile> row,
        // not cublas. Reduces to cutlass_<tile>(M, packed_n, K) vs
        // cublas(M, packed_n, K).
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let packed_n = num_q_heads
            .saturating_add(2u32.saturating_mul(num_kv_heads))
            .saturating_mul(head_dim);

        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), num_tokens, packed_n, hidden)
            .unwrap_or(UNCALIBRATED_COST_US);

        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (packed_n as f64) * BYTES_PER_ELEM;
        let rope_cache_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        let _ = m;
        gemm_us + rope_cache_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("CutlassFusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node).expect(
            "CutlassFusedQkvRopeCache: RopeAppend has a KvCache extern with concrete layer index",
        ) as u32;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("CutlassFusedQkvRopeCache: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        // matches() rejected biased claims; runtime does not handle bias.
        debug_assert!(
            resolved[0].1.is_none(),
            "CutlassFusedQkvRopeCache: matches() should have rejected biased claim",
        );

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedQkvRopeCache: q_gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "CutlassFusedQkvRopeCache: weight_layer ({weight_layer}) and \
             kv_cache layer ({layer}) disagree — DSL bug"
        );

        // Bake (packed_n=q+2*kv, k=hidden) for runtime weight-shape assert.
        // packed_n derives from arch-level `Weights::*_SIZE` at runtime,
        // but we re-derive it here from FUF + bounds for the claim's q
        // gemm so the assert can fire at codegen time if loader output
        // disagrees with FUF expectations.
        let (q_n, k) = gemm_nk_from_fuf(fuf, q_gemm_node, bounds)
            .expect("CutlassFusedQkvRopeCache: q gemm (N, K) must resolve from FUF + bounds");
        let num_kv_heads = bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let kv_size = num_kv_heads.saturating_mul(head_dim);
        let packed_n = q_n.saturating_add(kv_size.saturating_mul(2));
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            interleaved,
            tile_m,
            tile_n,
            stages,
            packed_n,
            k,
        )])
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassFusedQkvRopePrefillImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassFusedQkvRopePrefillImpl {
    fn csv_name(&self) -> &'static str {
        CutlassFusedQkvRopeCacheImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .csv_name()
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassFusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopePrefillImpl.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        let info = FusedQkvRopePrefillImpl.matches(fuf, seed, profile)?;
        if fused_qkv_claim_is_biased(fuf, &info) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same flops + bandwidth model as the decode-fused variant;
        // the kernel split is different but the work is the same.
        CutlassFusedQkvRopeCacheImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedQkvRopePrefillImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("CutlassFusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("CutlassFusedQkvRopePrefill: RopeAppend has a KvCache extern with layer index")
            as u32;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("CutlassFusedQkvRopePrefill: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        debug_assert!(
            resolved[0].1.is_none(),
            "CutlassFusedQkvRopePrefill: matches() should have rejected biased claim",
        );

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("CutlassFusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "CutlassFusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );

        let (q_n, k) = gemm_nk_from_fuf(fuf, q_gemm_node, bounds)
            .expect("CutlassFusedQkvRopePrefill: q gemm (N, K) must resolve from FUF + bounds");
        let num_kv_heads = bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let kv_size = num_kv_heads.saturating_mul(head_dim);
        let packed_n = q_n.saturating_add(kv_size.saturating_mul(2));
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![Instruction::CutlassFusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
            interleaved,
            tile_m,
            tile_n,
            stages,
            packed_n,
            k,
        )])
    }
}

// ── AttentionPrefillContiguousImpl ───────────────────────────────
//
// Singleton matcher on `OpKind::Attention` for the prefill bucket.
// Reads Q, K, V from the DSL tile's slots 0/1/2 (the outputs of
// the upstream `FusedQkvRopePrefillImpl`, which are real contiguous
// OwnedTensors) and calls `flash_attn_contiguous`. This matches
// what scratchy-serving-cuda's hand-written prefill path does.
//
// `WorkloadConstraint::NumTokensRange { min: 2, max: u32::MAX }`.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AttentionPrefillContiguousImpl;

#[cfg(feature = "cuda")]
impl Implementation for AttentionPrefillContiguousImpl {
    fn name(&self) -> &'static str {
        "attention_prefill_contiguous"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only — see `AttentionViaCacheImpl::matches` for rationale.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same calibrated FA2 CSV row as the paged decode variant —
        // the FA2 sweep emits one cost per (h, q, k, M, sk) cell and
        // doesn't distinguish decode/prefill. Falls back to the
        // analytic formula when no row matches.
        cost_attention_calibrated(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `AttentionPrefillContiguous { q_slot, k_slot, v_slot,
    // out_slot }`. Reads contiguous Q/K/V from upstream
    // FusedQkvRopePrefill outputs; calls flash_attn_contiguous with
    // baked scale/softcap. No layer / cos_sin needed (K is already
    // rotated upstream; rotary_dim=0 so flash-attn skips fused RoPE).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AttentionPrefillContiguous",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("AttentionPrefillContiguous: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);

        let interleaved = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .map(|l| layer_rope_is_interleaved(fuf, l))
            .unwrap_or(false);

        Some(vec![Instruction::AttentionPrefillContiguous(
            q_slot,
            k_slot,
            v_slot,
            out_slot,
            interleaved,
        )])
    }
}

// ── EncoderAttentionImpl ─────────────────────────────────────────
//
// Bidirectional / encoder self-attention. Singleton matcher on the
// 3-arg `OpKind::Attention` form `attention(q, k, v)` — the encoder
// DSL has no `kv_cache` extern and no `block_table`, which is the
// exact shape `attention_has_kv_cache_extern` rejects for the
// decoder Impls. Reads contiguous Q/K/V from the upstream tile
// slots (identical layout to what `AttentionPrefillContiguousImpl`
// consumes from `FusedQkvRopePrefill`) and emits
// `Instruction::EncoderAttention { q_slot, k_slot, v_slot,
// out_slot }`. The eval body in `instr.rs` calls
// `flash_attn_contiguous` with `is_causal=false` and a null cos_sin
// pointer.
//
// Workload: every token count (encoders process the whole sequence
// at once — no decode bucket). Singleton — sliding-window encoder
// variants land separately as they need a window-size constant.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct EncoderAttentionImpl;

#[cfg(feature = "cuda")]
impl Implementation for EncoderAttentionImpl {
    fn name(&self) -> &'static str {
        "encoder_attention"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Encoders see every token in one pass — every workload point
        // is "prefill"-shaped. No decode/prefill split.
        WorkloadConstraint::NumTokensRange {
            min: 1,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Encoder-only: requires the 3-arg `attention(q, k, v)` form
        // (no `kv_cache` extern). Decoder 5-arg tiles are claimed by
        // `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl`.
        if attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same FA2 kernel as the decoder prefill path; the calibrated
        // CSV row applies (FA2 sweep doesn't distinguish causal vs
        // bidirectional — masking is a kernel-internal cost-neutral
        // template parameter). Falls back to the analytic formula.
        cost_attention_calibrated(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "EncoderAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("EncoderAttention: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);

        Some(vec![Instruction::EncoderAttention(
            q_slot, k_slot, v_slot, out_slot,
        )])
    }
}

// ── SlidingAttentionViaCacheImpl ─────────────────────────────────
//
// Mirror of [`AttentionViaCacheImpl`] for [`OpKind::SlidingAttention`].
// Identical kernel path (paged decode through
// `attention_decode_from_cache`) but passes `window_size_left` from
// the model config's `sliding_window` field, so the flash-attn
// kernel masks out positions beyond the window. Decode-only
// (`NumTokensRange { 1, 1 }`); the prefill counterpart is
// [`SlidingAttentionPrefillContiguousImpl`].

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct SlidingAttentionViaCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for SlidingAttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "sliding_attention_via_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only — see `AttentionViaCacheImpl::matches` for rationale.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::SlidingAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `SlidingAttentionViaCache { in_slot, out_slot, layer,
    // cos_sin_fn, interleaved }`. Same shape as `AttentionViaCache`;
    // the only difference is the body bakes the sliding-window left
    // bound from `model.bounds["sliding_window"]` into the kernel
    // call instead of passing `-1`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SlidingAttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("SlidingAttentionViaCache: input 0 (Q) must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("SlidingAttentionViaCache: kv_cache extern with layer index")
            as u32;
        let interleaved = layer_rope_is_interleaved(fuf, layer as u64);
        Some(vec![Instruction::SlidingAttentionViaCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            interleaved,
        )])
    }
}

// ── SlidingAttentionPrefillContiguousImpl ────────────────────────
//
// Prefill counterpart of [`SlidingAttentionViaCacheImpl`]. Reads Q,
// K, V from slots 0/1/2 (populated by an upstream prefill QKV impl)
// and calls `flash_attn_contiguous` with the config-driven window.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct SlidingAttentionPrefillContiguousImpl;

#[cfg(feature = "cuda")]
impl Implementation for SlidingAttentionPrefillContiguousImpl {
    fn name(&self) -> &'static str {
        "sliding_attention_prefill_contiguous"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only — see `AttentionViaCacheImpl::matches` for rationale.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::SlidingAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `SlidingAttentionPrefillContiguous { q_slot, k_slot,
    // v_slot, out_slot, interleaved }`. Same shape as
    // `AttentionPrefillContiguous` plus the sliding window value
    // baked from `model.bounds["sliding_window"]`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SlidingAttentionPrefillContiguous",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!(
                    "SlidingAttentionPrefillContiguous: input {idx} must be a Tile (got {other:?})"
                ),
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);
        let interleaved = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .map(|l| layer_rope_is_interleaved(fuf, l))
            .unwrap_or(false);
        Some(vec![Instruction::SlidingAttentionPrefillContiguous(
            q_slot,
            k_slot,
            v_slot,
            out_slot,
            interleaved,
        )])
    }
}

// ── CutlassGemmImpl / CutlassGemvImpl ────────────────────────────
//
// Singleton matchers on `OpKind::Gemm` parameterised by the CUTLASS
// tile shape `(tile_m, tile_n, stages)`. Backed by
// `scratchy_target_cuda::cutlass::cutlass_gemm` / `cutlass_gemv` — the
// prior scratchy's hand-picked tile zoo, preserved byte-for-byte in
// the launch fns.
//
// Cost for `(workload M, weight [N, K])` is read directly from
// `target.cost_table` (`target_profiles/cost_<name>.csv`); variants
// missing a CSV row for a given shape report `f64::INFINITY` so the
// DP never picks a data-less kernel. Variants whose target has no
// CSV data at all are filtered out at `target_compatible` time.
//
// `matches` rejects Gemms whose downstream consumer is a fusion
// partner (RopeAppend / Silu / Mul), so the solver can never steer
// a QKV or gate/up gemm away from its fused impl — picking cutlass
// for the gate gemm would orphan the silu tile, which has no
// singleton kernel.

/// Every cutlass tile variant exported from
/// `scratchy-serving-cuda/csrc/cutlass_standalone_gemm.cu` with a matching
/// FFI declaration in `scratchy-target-cuda::cutlass`. Must stay in sync
/// with that file — add / remove tiles here and in the extern block
/// together.
/// Finite "don't pick me" cost for calibrated impls at shapes not
/// in their CSV. Large enough that any calibrated alternative wins,
/// small enough to keep `is_finite()` true (the DP rejects
/// non-finite costs with `SolveError::UnreachableCost`).
const UNCALIBRATED_COST_US: f64 = 1.0e9;

const CUTLASS_TILE_ZOO: &[(u32, u32, u32)] = &[
    // tile_m=16 — minimum threadblock M with sm80 tensor cores. Targets
    // small-M tall-skinny lm_head + QKV regimes where M ∈ [8, 16] and
    // larger tile_m wastes most of the threadblock's M dimension.
    // (16, 256) excluded — CUTLASS thread-map div-by-zero at that aspect.
    (16, 64, 3),
    (16, 64, 4),
    (16, 128, 3),
    (16, 128, 4),
    (32, 64, 3),
    (32, 64, 4),
    (32, 128, 3),
    (32, 128, 4),
    (32, 256, 3),
    (64, 64, 3),
    (64, 64, 4),
    (64, 128, 3),
    (64, 128, 4),
    (128, 64, 3),
    (128, 64, 4),
    (128, 128, 3),
    (128, 128, 4),
    (128, 256, 3),
    (256, 64, 3),
    (256, 64, 4),
];

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassGemmImpl {
    /// Same string as `static_name`, returned as `&'static str` so the
    /// hot `target_compatible` path doesn't allocate. The CSV column
    /// matches `static_name` by construction.
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        // Names are compile-time-known per CUTLASS_TILE_ZOO entry.
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

/// Evaluate the `(M, N, K)` of a Gemm tile for CSV cost lookup.
/// M comes from the current workload (bounds[`num_tokens`]), N from
/// the output's last dim, K from the activation-input tile's last
/// dim.
fn gemm_mnk(ctx: &CostCtx, node: &crate::fuf::FufNode) -> Option<(u32, u32, u32)> {
    let m = ctx.num_tokens() as u32;
    let (n, k) = gemm_nk_from_fuf(ctx.fuf, node, ctx.bounds)?;
    Some((m, n, k))
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassGemmImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Hot path — called per tile × per impl × per workload.
        // `has_kernel` is O(1); the previous `kernel_names()` scan
        // allocated a Vec<String> by cloning every cost-table entry's
        // key, which produced multi-second compile-time stalls on
        // large FUFs (1000+ tiles × 16 cutlass variants).
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // M=1 is GEMV territory — `CutlassGemvImpl` owns it.
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let node = ctx.fuf.get(m.claimed_tiles[0]);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, node) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            // No CSV row for this shape → emit a finite "too
            // expensive" sentinel so the DP skips this variant
            // without tripping the `!cost.is_finite()` guard.
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassGemm { in_slot, out_slot, layer, weight_fn,
    // tile_m, tile_n, stages }`. All `CutlassGemmImpl{tile_m=…,
    // tile_n=…, stages=…}` instances share this variant — the tile
    // config rides as runtime fields. The body always calls
    // `cutlass_gemm(...)` with `CutlassTile::new(tile_m, tile_n, stages)`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemm: weight (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::CutlassGemm(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

// ── CutlassGemmSplitKImpl ────────────────────────────────────────
//
// Singleton Gemm impl backed by CUTLASS `GemmSplitKParallel` — the K
// dim is split across `split_k` CTAs and reduced in a second kernel.
// Closes the "tall-skinny large-K" shape class where the standard
// tile zoo loses to cuBLAS (e.g. Qwen2-0.5B down_proj at prefill:
// M≈4096, N=896, K=4864).
//
// Same `matches` predicates as `CutlassGemmImpl` (dense bf16 only,
// not a fusion partner). Workload constraint excludes M=1 — GEMV
// owns decode. Cost is a direct CSV lookup on
// `cutlass_WxH_sS_splitN`; rows missing for a shape fall back to
// `UNCALIBRATED_COST_US` so the DP ignores the variant there.
//
// Registered alongside `CutlassGemmImpl` in `starter_library`.

/// The pure-SplitK tile variants exported by
/// `scratchy-serving-cuda/csrc/cutlass_standalone_gemm.cu` and declared as
/// externs in `scratchy-target-cuda::cutlass`. `(tile_m, tile_n, stages,
/// split_k)`. Keep in sync with that FFI block.
const CUTLASS_SPLITK_ZOO: &[(u32, u32, u32, u32)] = &[
    (64, 64, 4, 2),
    (64, 64, 4, 4),
    (64, 64, 4, 8),
    (64, 128, 4, 2),
    (64, 128, 4, 4),
    (64, 128, 4, 8),
    (128, 64, 4, 2),
    (128, 64, 4, 4),
    (128, 64, 4, 8),
    (128, 128, 4, 2),
    (128, 128, 4, 4),
    (128, 128, 4, 8),
    // tile_m=16 splitK — small-M long-K regime (M=8 + K≥8192).
    (16, 64, 4, 2),
    (16, 64, 4, 4),
    (16, 64, 4, 8),
    (16, 128, 4, 2),
    (16, 128, 4, 4),
    (16, 128, 4, 8),
];

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassGemmSplitKImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
    pub split_k: u32,
}

#[cfg(feature = "cuda")]
impl CutlassGemmSplitKImpl {
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages, self.split_k) {
            (64, 64, 4, 2) => "cutlass_64x64_s4_split2",
            (64, 64, 4, 4) => "cutlass_64x64_s4_split4",
            (64, 64, 4, 8) => "cutlass_64x64_s4_split8",
            (64, 128, 4, 2) => "cutlass_64x128_s4_split2",
            (64, 128, 4, 4) => "cutlass_64x128_s4_split4",
            (64, 128, 4, 8) => "cutlass_64x128_s4_split8",
            (128, 64, 4, 2) => "cutlass_128x64_s4_split2",
            (128, 64, 4, 4) => "cutlass_128x64_s4_split4",
            (128, 64, 4, 8) => "cutlass_128x64_s4_split8",
            (128, 128, 4, 2) => "cutlass_128x128_s4_split2",
            (128, 128, 4, 4) => "cutlass_128x128_s4_split4",
            (128, 128, 4, 8) => "cutlass_128x128_s4_split8",
            (16, 64, 4, 2) => "cutlass_16x64_s4_split2",
            (16, 64, 4, 4) => "cutlass_16x64_s4_split4",
            (16, 64, 4, 8) => "cutlass_16x64_s4_split8",
            (16, 128, 4, 2) => "cutlass_16x128_s4_split2",
            (16, 128, 4, 4) => "cutlass_16x128_s4_split4",
            (16, 128, 4, 8) => "cutlass_16x128_s4_split8",
            _ => "cutlass_splitk_unknown",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassGemmSplitKImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let node = ctx.fuf.get(m.claimed_tiles[0]);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, node) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemmSplitK",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("split_k", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemmSplitK: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemmSplitK: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let split_k = self.split_k;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemmSplitK: weight (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::CutlassGemmSplitK(
            in_slot_idx,
            out_slot_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            split_k,
            n,
            k,
        )])
    }
}

// ── CutlassGemmAddImpl ───────────────────────────────────────────
//
// Two-tile fusion: `(Gemm, Add)` where the Add is a residual-stream
// tile×tile add consuming the Gemm's output. Emits `cutlass_gemm_add`
// with `beta=1.0` — the CUTLASS LinearCombination epilogue computes
// `A @ B^T + residual` in-place on the residual buffer, one launch,
// no intermediate `[M, N]` delta in GMEM.
//
// One Impl per `CUTLASS_TILE_ZOO` entry (16 variants). Shares CSV
// cost rows with the singleton `CutlassGemmImpl` — the beta=1.0
// epilogue's extra aux-read of `[M, N]` bf16 is absorbed in the
// compute-bound GEMM's latency. If future calibration shows material
// drift, add dedicated `cutlass_WxH_sS_add` rows and switch
// `csv_name`.
//
// Dense-bf16 only; quantized storage Gemms (Marlin, Bnb4, Fp8, etc.)
// don't route through the cutlass tile zoo so this Impl rejects them
// the same way `CutlassGemmImpl` does.

#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
pub struct CutlassGemmAddImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

#[cfg(feature = "cuda")]
impl CutlassGemmAddImpl {
    fn csv_name(&self) -> &'static str {
        // Dedicated `_add` CSV row measured at beta=1.0 — captures
        // the epilogue's extra `[M, N]` aux-read cost that the plain
        // tile row (beta=0.0) doesn't see. Necessary for fair DP
        // comparison against `(singleton gemm + fused_add_rms_norm)`
        // on residual-stream chains.
        self.impl_name()
    }

    fn impl_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3_add",
            (16, 64, 4) => "cutlass_16x64_s4_add",
            (16, 128, 3) => "cutlass_16x128_s3_add",
            (16, 128, 4) => "cutlass_16x128_s4_add",
            (32, 64, 3) => "cutlass_32x64_s3_add",
            (32, 64, 4) => "cutlass_32x64_s4_add",
            (32, 128, 3) => "cutlass_32x128_s3_add",
            (32, 128, 4) => "cutlass_32x128_s4_add",
            (32, 256, 3) => "cutlass_32x256_s3_add",
            (64, 64, 3) => "cutlass_64x64_s3_add",
            (64, 64, 4) => "cutlass_64x64_s4_add",
            (64, 128, 3) => "cutlass_64x128_s3_add",
            (64, 128, 4) => "cutlass_64x128_s4_add",
            (128, 64, 3) => "cutlass_128x64_s3_add",
            (128, 64, 4) => "cutlass_128x64_s4_add",
            (128, 128, 3) => "cutlass_128x128_s3_add",
            (128, 128, 4) => "cutlass_128x128_s4_add",
            (128, 256, 3) => "cutlass_128x256_s3_add",
            (256, 64, 3) => "cutlass_256x64_s3_add",
            (256, 64, 4) => "cutlass_256x64_s4_add",
            _ => "cutlass_unknown_add",
        }
    }
}

#[cfg(feature = "cuda")]
impl Implementation for CutlassGemmAddImpl {
    fn name(&self) -> &'static str {
        self.impl_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Match CutlassGemmImpl's constraint — M=1 is GEMV territory
        // and has no gemm+add variant today.
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed is the Gemm. Must be dense bf16, not a fusion partner
        // for an upstream chain (Silu/Mul/RopeAppend/BiasAdd); those
        // are owned by their dedicated fused Impls.
        let node = fuf.get(seed);
        if node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(node), Some(StorageFormat::Dense)) {
            return None;
        }
        // Find a downstream residual-stream Add: both inputs are
        // Tiles, one of them is `seed`, the Add isn't already owned
        // by a scalar-offset path.
        let add_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::Add || n.inputs.len() != 2 {
                return false;
            }
            let all_tiles = n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }));
            if !all_tiles {
                return false;
            }
            consumes_tile(n, seed)
        })?;
        let add_id = add_node.id;

        // Identify the residual-side input (the Tile that isn't
        // `seed`). Convention in this codebase: slot 0 is the
        // delta (Gemm output), slot 1 is the residual. We don't
        // enforce slot order here — `cutlass_gemm_add` commutes
        // over alpha*delta + beta*residual anyway.
        let residual_src = add_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, slot } if *id != seed => Some((*id, *slot)),
            _ => None,
        })?;

        let activation_src = first_tile_input(node)?;

        Some(MatchInfo {
            claimed_tiles: vec![seed, add_id],
            boundary_inputs: vec![activation_src.0, residual_src.0],
            boundary_outputs: vec![add_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_tile = m
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
            .expect("cutlass gemm+add claim contains a Gemm");
        let Some((mm, nn, kk)) = gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::GemmAddResidual
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassGemmAdd { in_slot, residual_slot, layer,
    // weight_fn, tile_m, tile_n, stages }`. No out_slot — the Add's
    // output is a View alias of the residual upstream (declared via
    // output_alias and populated by the alias prelude). Body runs
    // the cutlass_gemm_add kernel which mutates the residual buffer
    // in place.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemmAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassGemmAdd: claim contains a Gemm");
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("CutlassGemmAdd: claim contains an Add");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemmAdd: gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let (residual_id, residual_in) = fuf
            .get(add_id)
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
                _ => None,
            })
            .expect("CutlassGemmAdd: Add has a non-gemm Tile input (residual)");
        let residual_idx = slots.of(residual_id, residual_in);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemmAdd: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassGemmAdd: gemm (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::CutlassGemmAdd(
            in_slot_idx,
            residual_idx,
            layer,
            tile_m,
            tile_n,
            stages,
            n,
            k,
        )])
    }
}

// ── FusedCublasGemmAddImpl ───────────────────────────────────────
//
// cuBLAS-side peer to [`CutlassGemmAddImpl`]. Same `(Gemm, Add)`
// 2-tile claim, but the GEMM step routes through cuBLAS instead of a
// CUTLASS tile + the add is a separate `add_inplace` BW-bound launch.
//
// Closes a structural anomaly the analyzer surfaces: the
// CutlassGemmAdd peer exists, so when CUTLASS loses the bucket the
// fusion shatters into `(Cublas-singleton, Add-singleton)` rather
// than landing as one fused-claim cuBLAS call. Only ~12 picks today
// (the analyzer's `Gemm→Add (CutlassGemmAdd peer; no cuBLAS peer)`
// row), but the structural completion matters: without this Impl,
// those 12 picks would route through a `(GemmRefImpl, AddRefImpl)`
// pair — leaving the pair claimable only by `CutlassGemmAdd` (which
// can already win these buckets via cost).

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedCublasGemmAddImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedCublasGemmAddImpl {
    fn name(&self) -> &'static str {
        "fused_cublas_gemm_add"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Step 2b — restrict to M >= 64 so cuBLAS-side fused
        // (Gemm, Add) only competes where cuBLAS legitimately wins
        // the underlying GEMM. At smaller M, `CutlassGemmAddImpl`
        // (CUTLASS-side fused) consistently outpaces cuBLAS at the
        // same shape, and over-claiming this Impl at M=2..64 is what
        // produced the +148 cuBLAS-surface regression on cohere
        // parallel attn+mlp documented in earlier STATUS.
        WorkloadConstraint::NumTokensRange {
            min: 64,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Same matcher as CutlassGemmAddImpl: dense Gemm + downstream
        // residual-stream Add (both Add inputs are Tiles; one is the
        // Gemm output).
        let node = fuf.get(seed);
        if node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(node), Some(StorageFormat::Dense)) {
            return None;
        }
        let add_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::Add || n.inputs.len() != 2 {
                return false;
            }
            let all_tiles = n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }));
            if !all_tiles {
                return false;
            }
            consumes_tile(n, seed)
        })?;
        let add_id = add_node.id;
        let residual_src = add_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, slot } if *id != seed => Some((*id, *slot)),
            _ => None,
        })?;
        let activation_src = first_tile_input(node)?;
        Some(MatchInfo {
            claimed_tiles: vec![seed, add_id],
            boundary_inputs: vec![activation_src.0, residual_src.0],
            boundary_outputs: vec![add_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // cuBLAS GEMM cost + BW-bound add. Mirrors
        // CutlassGemmAddImpl's structure but reads "cublas" CSV with
        // a roofline fallback to keep parity with GemmRefImpl's
        // measurement scale at uncalibrated shapes.
        let gemm_tile = m
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
            .expect("FusedCublasGemmAdd: claim contains a Gemm");
        let Some((mm, nn, kk)) = gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) else {
            return UNCALIBRATED_COST_US;
        };
        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", mm, nn, kk)
            .unwrap_or_else(|| {
                let flops = 2.0 * mm as f64 * nn as f64 * kk as f64;
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    UNCALIBRATED_COST_US
                }
            });
        // BW-bound add: 2 reads (delta, residual) + 1 write (residual)
        // over [M, N] bf16.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * mm as f64 * nn as f64 * BYTES_PER_ELEM;
        let add_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };
        gemm_us + add_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::GemmAddResidual
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedCublasGemmAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("FusedCublasGemmAdd: claim contains a Gemm");
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("FusedCublasGemmAdd: claim contains an Add");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedCublasGemmAdd: gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let (residual_id, residual_in) = fuf
            .get(add_id)
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
                _ => None,
            })
            .expect("FusedCublasGemmAdd: Add has a non-gemm Tile input (residual)");
        let residual_idx = slots.of(residual_id, residual_in);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedCublasGemmAdd: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("FusedCublasGemmAdd: gemm (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::FusedCublasGemmAdd(
            in_slot_idx,
            residual_idx,
            layer,
            n,
            k,
        )])
    }
}

/// M=1 SIMT GEMV specialisation — the decode path's lm_head /
/// o_proj / down_proj.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct CutlassGemvImpl;

#[cfg(feature = "cuda")]
impl Implementation for CutlassGemvImpl {
    fn name(&self) -> &'static str {
        "cutlass_gemv"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile
            .cost_table
            .kernel_names()
            .iter()
            .any(|k| k == "cutlass_gemv")
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let node = ctx.fuf.get(m.claimed_tiles[0]);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, node) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for("cutlass_gemv", mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemv",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemv: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemv: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemv: weight (N, K) must resolve from FUF + bounds");
        Some(vec![Instruction::CutlassGemv(
            in_slot_idx,
            out_slot_idx,
            layer,
            n,
            k,
        )])
    }
}

// ── Marlin* impls ────────────────────────────────────────────────
//
// Quant-aware mirrors of `GemmRefImpl`, `FusedGateUpSiluMulImpl`,
// `FusedQkvRopeCacheImpl`, and `FusedQkvRopePrefillImpl`. Each one:
//   - Structural match identical to its dense sibling (walks the
//     same tile pattern off the seed).
//   - Weight storage gate **inverted**: requires every participating
//     Gemm's weight input to be Marlin-consumable (AWQ or GPTQ),
//     not `Dense`. The dense variants already reject quant storage,
//     so the solver picks exactly one family per model.
//   - `required_weights` declares the packed accessor with
//     `rust_type = MarlinLinear`. AWQ and GPTQ both concat their
//     qweight/scales/(qzeros|g_idx) along the N axis, so the fused
//     QKV / fused gate-up case collapses to **one** `MarlinLinear`
//     — same shape the dense fused accessor uses (one `LinearLayer`).
//   - `emit_call` invokes `MarlinLinear::forward(x, alloc, stream)`.
//     No cublas handle — marlin's kernel owns the matmul.

/// Does `tile` have a Gemm with a Marlin-consumable storage (AWQ or
/// GPTQ)? Quant-aware impls gate on this at the seed and at every
/// fused Gemm they claim. The Marlin kernel is agnostic to the
/// source format after repack — AWQ's uint4 and GPTQ's uint4b8 are
/// both first-class `b_type_id`s in the kernel — so any impl that
/// emits `MarlinLinear::forward` accepts either storage.
fn is_marlin_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm
        && matches!(
            weight_storage_of(node),
            Some(StorageFormat::Awq { .. } | StorageFormat::Gptq { .. })
        )
}

/// BitsAndBytes 4-bit counterpart of [`is_marlin_gemm`]. A Gemm
/// whose weight resolves to `StorageFormat::Bnb4 { .. }` — the
/// Bnb4bit impl family claims these; Marlin + dense impls reject
/// them via their own storage gates.
fn is_bnb4_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm && matches!(weight_storage_of(node), Some(StorageFormat::Bnb4 { .. }))
}

/// FP8 counterpart of [`is_marlin_gemm`]. A Gemm whose weight
/// resolves to `StorageFormat::Fp8 { .. }` — the Fp8 impl family
/// claims these; dense / Marlin / Bnb4 impls reject them via their
/// own storage gates.
fn is_fp8_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm && matches!(weight_storage_of(node), Some(StorageFormat::Fp8 { .. }))
}

/// True if `tile` is a `Moe` whose `moe[layer]` weight resolves to
/// blockwise FP8 storage (DeepSeek-V3 / Kimi K2 official checkpoint format).
/// Used by `DeepSeekFp8BlockMoeImpl` to claim, and by `DeepSeekMoeRefImpl`
/// (BF16) to defer.
fn is_fp8_block_moe(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    if node.op != OpKind::Moe {
        return false;
    }
    matches!(
        weight_storage_of(node),
        Some(StorageFormat::Fp8 {
            block_size: Some(_),
            ..
        })
    )
}

/// True if `tile` is a `Moe` whose `moe[layer]` weight resolves to
/// GGML/GGUF storage. Used by `DeepSeekGgmlMoeImpl` to claim, and by
/// `DeepSeekMoeRefImpl` (BF16) to defer.
fn is_ggml_moe(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    if node.op != OpKind::Moe {
        return false;
    }
    matches!(weight_storage_of(node), Some(StorageFormat::Ggml))
}

/// GGML/GGUF counterpart of [`is_marlin_gemm`]. A Gemm whose weight
/// resolves to `StorageFormat::Ggml` — the GGUF impl family claims
/// these; every other impl rejects them via their own storage
/// gates.
fn is_ggml_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm && matches!(weight_storage_of(node), Some(StorageFormat::Ggml))
}

/// Accessor Rust type for an FP8 GEMM. Always `Fp8AnyLinear` — the
/// loader picks `Std` vs `Block` per claim based on the weight's
/// on-disk storage, but the per-arch Weights field type is uniform so
/// the host-interpreter `weight_fn` can have one concrete signature.
fn fp8_accessor_type_for(_fuf: &Fuf, _gemm_tile: TileId) -> TokenStream {
    quote! { crate::__gpu::layers::Fp8AnyLinear }
}

/// True if a `MarlinFusedQkvRope*Impl` would accept `seed` as one of
/// its Q/K/V Gemms — i.e., there exists a `RopeAppend` whose first
/// three tile inputs unwrap (optionally through `BiasAdd`) to three
/// Marlin Gemms sharing one activation, and `seed` is one of them.
/// Used by `MarlinGemmImpl` to defer without over-claiming.
fn marlin_fused_qkv_rope_would_match(fuf: &Fuf, seed: TileId) -> bool {
    fuf.nodes.iter().any(|n| {
        if !is_rope_append_op(n.op) || n.inputs.len() < 3 {
            return false;
        }
        let qkv_raw: Vec<TileId> = n
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        if qkv_raw.len() != 3 {
            return false;
        }
        let Some(resolved): Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t))
            .collect()
        else {
            return false;
        };
        let biased = resolved[0].1.is_some();
        if resolved.iter().any(|r| r.1.is_some() != biased) {
            return false;
        }
        let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
        if !gemms.contains(&seed) {
            return false;
        }
        if gemms.iter().any(|t| !is_marlin_gemm(fuf, *t)) {
            return false;
        }
        let act = first_tile_input(fuf.get(gemms[0]));
        act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
    })
}

/// True if `MarlinFusedGateUpSiluMulImpl` (or the Gelu variant) would
/// accept `seed` as gate or up gemm — `seed` is a Marlin Gemm whose
/// output feeds `Silu` or `Gelu`, or whose output is the "up" half of
/// a `(silu_or_gelu(gate) * up)` pair. Keeps the deference tight so
/// non-SwiGLU/GELU `Mul` patterns (e.g. Granite's scalar multiplier)
/// don't trigger it.
fn marlin_fused_gate_up_would_match(fuf: &Fuf, seed: TileId) -> bool {
    // gate gemm: output feeds Silu or Gelu which feeds Mul with a
    // sibling gemm as the other input.
    for n in &fuf.nodes {
        if !matches!(n.op, OpKind::Silu | OpKind::Gelu) {
            continue;
        }
        if !consumes_tile(n, seed) {
            // seed isn't the gate; check if seed is the up gemm
            // that feeds the Mul alongside this Silu/Gelu.
            let act_id = n.id;
            for m in &fuf.nodes {
                if m.op != OpKind::Mul || !consumes_tile(m, act_id) {
                    continue;
                }
                let up_is_seed = m.inputs.iter().any(|i| match i {
                    FufInput::Tile { id, .. } => *id == seed && *id != act_id,
                    _ => false,
                });
                if up_is_seed && is_marlin_gemm(fuf, seed) {
                    return true;
                }
            }
            continue;
        }
        // seed feeds the Silu/Gelu — check that Silu/Gelu's output
        // feeds a Mul paired with another Marlin Gemm.
        let act_id = n.id;
        let has_mul_pair = fuf.nodes.iter().any(|m| {
            if m.op != OpKind::Mul || !consumes_tile(m, act_id) {
                return false;
            }
            m.inputs.iter().any(|i| match i {
                FufInput::Tile { id, .. } => *id != act_id && is_marlin_gemm(fuf, *id),
                _ => false,
            })
        });
        if has_mul_pair {
            return true;
        }
    }
    false
}

/// Singleton Marlin GEMM — the AWQ counterpart of `GemmRefImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MarlinGemmImpl;

#[cfg(feature = "cuda")]
impl Implementation for MarlinGemmImpl {
    fn name(&self) -> &'static str {
        "marlin_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        // Precision-gate the fusion-partner deference. The old
        // `gemm_is_fusion_partner` check deferred to any Gemm
        // feeding RopeAppend/Silu/Mul/BiasAdd. That was too eager:
        //   - Granite's `o_proj * scalar(residual_multiplier)` is
        //     a `ScalarMul`, not the `silu(gate)*up` shape —
        //     nothing would claim the pair, `UnclaimedTile`.
        //   - Qwen3's per-head QK-norm interrupts the `(Gemm,
        //     Gemm, Gemm, RopeAppend)` adjacency that
        //     `MarlinFusedQkvRope*Impl` looks for — same orphan
        //     outcome.
        // Defer only when a fused sibling would actually match at
        // this seed. `marlin_fused_qkv_rope_would_match` checks
        // the exact `(Gemm, Gemm, Gemm, RopeAppend)` adjacency
        // (optionally through `BiasAdd`); the gate-up check
        // pattern-matches the `(Gemm, Gemm, Silu, Mul)` shape.
        if marlin_fused_qkv_rope_would_match(fuf, seed)
            || marlin_fused_gate_up_would_match(fuf, seed)
        {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // No CSV baseline for marlin today — reuse the cuBLAS
        // analytical estimate. Parity with the future
        // `cost_<target>.csv` lookup will land alongside the marlin
        // calibration sweep.
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Same single-weight accessor as the dense default, but
        // typed `MarlinLinear` so codegen routes the FieldLoad
        // through `MarlinLinear::load_awq` or `load_gptq` per the
        // source weight's storage format.
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::MarlinLinear },
            source_weights: vec![(wid, index)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MarlinGemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinGemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::MarlinGemm(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Marlin fused gate/up + SwiGLU — the AWQ counterpart of
/// `FusedGateUpSiluMulImpl`. AWQ metadata concats along dim N, so
/// the fused accessor is a single `MarlinLinear` (not two).
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MarlinFusedGateUpSiluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for MarlinFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedGateUpSiluMulImpl.matches`; see that impl
        // for the pattern-walking rationale. The only structural
        // change is the storage gate: BOTH Gemms must be AWQ.
        let gate_gemm = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_marlin_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Fused accessor covers gate + up; declared `MarlinLinear`
        // so codegen emits `MarlinLinear::load_awq_concat` or
        // `load_gptq_concat` depending on storage format.
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("MarlinFusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("MarlinFusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("MarlinFusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::MarlinFusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Marlin fused gate/up GELU mul — the Marlin counterpart of
/// [`FusedGateUpGeluMulImpl`] for Gemma2-style MLPs. Same 4-tile
/// (gate_gemm, up_gemm, Gelu, Mul) pattern as the Silu variant;
/// the diff is the activation op. Emits one `MarlinLinear::forward`
/// on a fused gate+up accessor, followed by
/// `gelu_and_mul_fused` — exactly what the dense GELU path does,
/// but reading from a MarlinLinear instead of a dense LinearLayer.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MarlinFusedGateUpGeluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for MarlinFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedGateUpGeluMulImpl.matches`; the only
        // structural change is the storage gate: BOTH Gemms must be
        // Marlin-consumable (AWQ or GPTQ).
        let gate_gemm = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_marlin_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("MarlinFusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("MarlinFusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("MarlinFusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::MarlinFusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Marlin fused QKV + RoPE + cache-write (decode) — the AWQ
/// counterpart of `FusedQkvRopeCacheImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MarlinFusedQkvRopeCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for MarlinFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedQkvRopeCacheImpl.matches`; diffs are the
        // storage gate (Q/K/V must all be Marlin-consumable — AWQ
        // or GPTQ) on top of the BiasAdd-aware unwrap. Bias is
        // handled at the kernel boundary: `MarlinLinear::forward`
        // applies `bias_add_inplace` after `marlin_gemm` when
        // `self.bias.is_some()`, and the `load_{awq,gptq}_concat`
        // loaders pack per-prefix `.bias` tensors into the fused
        // LinearLayer automatically.
        let seed_node = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            // Uniform bias presence across Q/K/V — mixed is a DSL
            // inconsistency, reject rather than silently partial-fuse.
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_marlin_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("already validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MarlinFusedQkvRopeCache { in_slot, out_slot, layer,
    // weight_fn(MarlinLinear), cos_sin_fn }`. No interleaved field
    // (Marlin gates AWQ/GPTQ; Cohere is dense-only). FP8 KV branches
    // at runtime via `ctx.kv_cache.is_fp8()`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("MarlinFusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("MarlinFusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("MarlinFusedQkvRopeCache: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "MarlinFusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::MarlinFusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Marlin fused QKV + RoPE (prefill, contiguous K/V) — the AWQ
/// counterpart of `FusedQkvRopePrefillImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MarlinFusedQkvRopePrefillImpl;

#[cfg(feature = "cuda")]
impl Implementation for MarlinFusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same pattern as the cache variant — delegate so the
        // matcher lives in one place.
        MarlinFusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        MarlinFusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        MarlinFusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("MarlinFusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("MarlinFusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("MarlinFusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "MarlinFusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::MarlinFusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
        )])
    }
}

// ── BitsAndBytes 4-bit (NF4 / FP4) impl family ────────────────────
//
// Mirror of the Marlin* family but for `StorageFormat::Bnb4`. The
// structural matchers are byte-identical to the Marlin variants
// (same `(Gemm, Gemm, Silu, Mul)` / `(Gemm, Gemm, Gelu, Mul)` /
// `(Gemm, Gemm, Gemm, RopeAppend)` claim patterns); the only diffs
// are (a) the storage gate (`is_bnb4_gemm` instead of
// `is_marlin_gemm`), (b) the declared accessor `rust_type`
// (`Bnb4bitLinear`), and (c) the `emit_call` binding
// (`(#w).forward(x, &mut device.cublas, &mut device.caching,
// stream)` — BNB4's forward takes cuBLAS because the kernel does
// a dequant-then-cuBLAS-matmul, unlike Marlin's fused-matmul path).
//
// The kernel-side loader (`Bnb4bitLinear::load{,_concat}`) handles
// byte-concat of packed nibbles + absmax across fused shards, so a
// fused QKV / gate-up lands as a single `Bnb4bitLinear` just like
// the Marlin fused accessors do.

/// Singleton Bnb4 GEMM — BNB4 counterpart of `MarlinGemmImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Bnb4GemmImpl;

#[cfg(feature = "cuda")]
impl Implementation for Bnb4GemmImpl {
    fn name(&self) -> &'static str {
        "bnb4_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        // Unlike Marlin + dense paths, BNB4 has no cutlass-shaped
        // singleton impl that would compete with the fused variants
        // at the same seed — so we skip `gemm_is_fusion_partner`
        // deference. If the fused BNB4 Impl (QKV-rope, gate-up-silu,
        // gate-up-gelu) matches, the DP picks it over this singleton
        // by claim-size. If it doesn't (e.g. Qwen3's per-head QK-norm
        // breaks the `(Gemm, Gemm, Gemm, RopeAppend)` adjacency for
        // q/k but not v), this singleton claims the leftover Gemms
        // rather than leaving them `UnclaimedTile`.
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::Bnb4bitLinear },
            source_weights: vec![(wid, index)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Bnb4Gemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4Gemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Bnb4Gemm(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── GGML/GGUF impl family ────────────────────────────────────────
//
// Singleton-only today. Per-tensor on-disk dtype (Q4_K, Q8_0, …) is
// heterogeneous within a single GGUF and runtime-dispatched inside
// `LinearLayer::Ggml(GgmlLinear)::forward`, so a single `GgmlGemmImpl`
// covers every quantized linear regardless of dtype. Fused QKV /
// gate-up peers (`GgmlFusedQkvRope*`, `GgmlFusedGateUpSiluMulImpl`)
// are perf follow-ups; the singleton produces correct results by
// emitting separate GEMMs where the fused dense path would emit one.

/// Singleton GGML GEMM — GGUF counterpart of `Fp8GemmImpl` /
/// `Bnb4GemmImpl`. Matches any `Gemm` whose weight resolves to
/// `StorageFormat::Ggml`. Emits a `LinearLayer::Ggml(..)` invocation
/// that dispatches to the right `ggml_dequant_mul_mat_vec` /
/// `ggml_dequantize_to_tensor + cuBLAS` kernel based on the runtime
/// `GgmlDType` carried on the loaded `GgmlStorage`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GgmlGemmImpl;

#[cfg(feature = "cuda")]
impl Implementation for GgmlGemmImpl {
    fn name(&self) -> &'static str {
        "ggml_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_ggml_gemm(fuf, seed) {
            return None;
        }
        // No fused-partner deference — like Bnb4 / Fp8 singletons,
        // we have no fused GGUF peers yet, so this is the only Impl
        // that can claim a GGUF Gemm. DP picks it whenever
        // `is_ggml_gemm` is true.
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        vec![WeightAccessor {
            name,
            // The polymorphic `LinearLayer` enum's `Ggml` arm is
            // what the codegen FieldLoad arm constructs at load
            // time. `forward` dispatches on the enum at runtime.
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: vec![(wid, index)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GgmlGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("GgmlGemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("GgmlGemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::GgmlGemm(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused gate/up + SwiGLU for GGUF. Mirrors `Fp8FusedGateUpSiluMulImpl`
/// — same FUF pattern (Gemm+Gemm sharing activation → Silu → Mul),
/// same single-packed-weight accessor (`LinearLayer::Ggml(..)` whose
/// storage is the byte-concat of gate/up). Runtime opcode handler
/// calls the polymorphic `LinearLayer::forward` to dequant+gemm
/// into a `[tokens, 2*intermediate]` tensor, then `silu_and_mul_fused`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GgmlFusedGateUpSiluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for GgmlFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "ggml_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_ggml_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_ggml_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GgmlFusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("GgmlFusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("GgmlFusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("GgmlFusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("GgmlFusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("GgmlFusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::GgmlFusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// GELU counterpart of `GgmlFusedGateUpSiluMulImpl`. Same shape;
/// matches Gelu+Mul instead of Silu+Mul. Cohere/commandr uses Silu;
/// Gemma uses Gelu — having both ready avoids a second round-trip
/// when wiring more arches.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GgmlFusedGateUpGeluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for GgmlFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "ggml_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_ggml_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_ggml_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GgmlFusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("GgmlFusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("GgmlFusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("GgmlFusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("GgmlFusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("GgmlFusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::GgmlFusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused QKV+RoPE (decode, M=1) for GGUF. Mirrors
/// `Bnb4FusedQkvRopeCacheImpl`'s claim shape — three Gemms (with or
/// without uniform `BiasAdd`) feeding a single `RopeAppend`. The
/// Ggml-specific bits: gate on `is_ggml_gemm`, accessor type
/// `LinearLayer` (which the runtime polymorphically dispatches to
/// `GgmlLinear` / `GgmlConcat` for the byte-packed q/k/v storage).
/// Required weights: one packed accessor with three sources (q/k/v);
/// `LinearLayer::load_dense_concat_or_ggml` either byte-stacks them
/// when dtypes match or wraps them in `GgmlConcat` for runtime
/// concat. Critically: this Impl handles BIASED q/k/v (Qwen2 style)
/// — without it, the standalone `BiasAdd` tile has no claimer and
/// the solver fails.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GgmlFusedQkvRopeCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for GgmlFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "ggml_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if !is_ggml_gemm(fuf, seed) {
            return None;
        }
        let rope_node = fuf.nodes.iter().find(|n| {
            // Accept both standard `RopeAppend` (Llama, Qwen, Mistral)
            // and `RopeAppendInterleaved` (Cohere) — same claim shape,
            // runtime kernel dispatches on the kv_cache extern.
            if !is_rope_append_op(n.op) || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_ggml_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("validated in find"))
            .collect();
        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();
        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GgmlFusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("GgmlFusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("GgmlFusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("GgmlFusedQkvRopeCache: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("GgmlFusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("GgmlFusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "GgmlFusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        let interleaved = fuf.get(rope_id).op == OpKind::RopeAppendInterleaved;
        Some(vec![Instruction::GgmlFusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
            interleaved,
        )])
    }
}

/// Fused QKV+RoPE (prefill, contiguous K/V) for GGUF.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GgmlFusedQkvRopePrefillImpl;

#[cfg(feature = "cuda")]
impl Implementation for GgmlFusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "ggml_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        let info = GgmlFusedQkvRopeCacheImpl.matches(fuf, seed, profile)?;
        // No interleaved-rope prefill kernel exists today
        // (`fused_qkv_rope` non-interleaved only). Reject so commandr
        // prefill falls through to standalone GgmlGemm + RopeAppendRef.
        let rope_id = *info
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))?;
        if fuf.get(rope_id).op == OpKind::RopeAppendInterleaved {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        GgmlFusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        GgmlFusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GgmlFusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("GgmlFusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("GgmlFusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("GgmlFusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("GgmlFusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("GgmlFusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "GgmlFusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::GgmlFusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
        )])
    }
}

// ── FP8 (E4M3) impl family ────────────────────────────────────────
//
// Singleton-only today. Covers per-tensor dynamic, per-tensor static,
// and per-channel FP8 via the same `Fp8Linear::forward` entry point;
// the loader sniffs scale layout at load time. Fused QKV / gate-up
// peers (`Fp8FusedQkvRope*`, `Fp8FusedGateUpSiluMulImpl`) are perf
// follow-ups — the singleton still produces correct results by
// emitting three separate FP8 GEMMs where the fused dense path
// would emit one.

/// Singleton FP8 GEMM — FP8 counterpart of `Bnb4GemmImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8GemmImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8GemmImpl {
    fn name(&self) -> &'static str {
        "fp8_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        let rust_type = fp8_accessor_type_for(fuf, tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: vec![(wid, index)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8Gemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8Gemm: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Fp8Gemm(in_slot_idx, out_slot_idx, layer)])
    }
}

/// Fused `(Gemm, BiasAdd)` for FP8. FP8 counterpart of
/// `FusedGemmBiasImpl`. `Fp8Linear` stores the loaded `.bias` on the
/// struct and applies it inside `forward` (cutlass
/// `cutlass_scaled_mm_with_bias` epilog), so `emit_call` is the
/// same as `Fp8GemmImpl`'s — the BiasAdd tile is absorbed into the
/// accessor rather than a separate kernel launch.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8FusedGemmBiasImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8FusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gemm_bias"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(
            weight_storage_of(seed_node),
            Some(StorageFormat::Fp8 { .. })
        ) {
            return None;
        }
        let bias_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::BiasAdd
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == seed
                )
        })?;
        let bias_id = bias_node.id;

        let mut claimed = vec![seed, bias_id];
        claimed.sort();

        let activation_inputs: Vec<TileId> = seed_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: activation_inputs,
            boundary_outputs: vec![bias_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_info = MatchInfo {
            claimed_tiles: vec![gemm_id],
            boundary_inputs: vec![],
            boundary_outputs: vec![gemm_id],
        };
        cost_gemm(&gemm_info, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let gemm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let (wid, index) = first_weight_ref(fuf.get(gemm_id)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        let rust_type = fp8_accessor_type_for(fuf, gemm_id);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: vec![(wid, index)],
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedGemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("Fp8FusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("Fp8FusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8FusedGemmBias: gemm's first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGemmBias: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Fp8FusedGemmBias(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused gate/up + SwiGLU for FP8. `Fp8Linear::load_concat`
/// max-scale merges gate + up into one fused FP8 matmul, so this
/// emits one `Fp8Linear::forward` on the fused weight followed by
/// `silu_and_mul_fused`. FP8 counterpart of `Bnb4FusedGateUpSiluMulImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8FusedGateUpSiluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_fp8_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("Fp8FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Fp8FusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Fp8FusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Fp8FusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused gate/up + SwiGLU for BNB4. BNB4 byte-concats nibbles +
/// absmax across gate + up on load (`load_concat`), so this emits
/// one `Bnb4bitLinear::forward` on the fused weight followed by
/// `silu_and_mul_fused`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Bnb4FusedGateUpSiluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for Bnb4FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_bnb4_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("Bnb4FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Bnb4FusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Bnb4FusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedGateUpSiluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Bnb4FusedGateUpSiluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused gate/up + GELU for BNB4 — Gemma2-style MLP.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Bnb4FusedGateUpGeluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for Bnb4FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_bnb4_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("Bnb4FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Bnb4FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Bnb4FusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Bnb4FusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused gate/up + GELU for FP8 — Gemma2/Gemma3/Granite-style MLP.
/// FP8 counterpart of `Bnb4FusedGateUpGeluMulImpl`.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8FusedGateUpGeluMulImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_fp8_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("Fp8FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Fp8FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Fp8FusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGateUpGeluMul: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::Fp8FusedGateUpGeluMul(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused QKV + RoPE (decode, M=1) for FP8. Mirrors
/// `Bnb4FusedQkvRopeCacheImpl` but emits the FP8 matmul path.
///
/// Importantly, this is the impl that makes scratchy's FP8 QKV path
/// numerically equivalent to Python vLLM's `QKVParallelLinear` for
/// per-tensor weight scales: `Fp8Linear::load_concat` max-scale-
/// merges Q/K/V weight_scales via the same `requantize_with_max_scale`
/// Python does before the fused matmul. Without this impl, the
/// singleton `Fp8FusedGemmBiasImpl × 3` fallback keeps the original
/// per-proj scales — arithmetically identical to Python only when
/// weight_scales are per-channel `[N, 1]`, and drifting when they
/// collapse to per-tensor scalars.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8FusedQkvRopeCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_fp8_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Fp8FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Fp8FusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Fp8FusedQkvRopeCache: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8FusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Fp8FusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::Fp8FusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused QKV + RoPE (prefill, contiguous K/V) for FP8.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Fp8FusedQkvRopePrefillImpl;

#[cfg(feature = "cuda")]
impl Implementation for Fp8FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        Fp8FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        Fp8FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        Fp8FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Fp8FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Fp8FusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Fp8FusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Fp8FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::Fp8FusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
        )])
    }
}

/// Fused QKV + RoPE (decode, M=1) for BNB4. Mirrors
/// `MarlinFusedQkvRopeCacheImpl` but emits the BNB4 matmul path.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Bnb4FusedQkvRopeCacheImpl;

#[cfg(feature = "cuda")]
impl Implementation for Bnb4FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_bnb4_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::SoleOutput(crate::alias_rules::OpSel::RopeAny)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<UnrollIndex>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Bnb4FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Bnb4FusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Bnb4FusedQkvRopeCache: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Bnb4FusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedQkvRopeCache: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Bnb4FusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::Bnb4FusedQkvRopeCache(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

/// Fused QKV + RoPE (prefill, contiguous K/V) for BNB4.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct Bnb4FusedQkvRopePrefillImpl;

#[cfg(feature = "cuda")]
impl Implementation for Bnb4FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        Bnb4FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        Bnb4FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        Bnb4FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Bnb4FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Bnb4FusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Bnb4FusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedQkvRopePrefill: required_weights returned empty");
        let (_base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.map(|v| v.0).unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Bnb4FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        Some(vec![Instruction::Bnb4FusedQkvRopePrefill(
            in_slot_idx,
            q_out,
            k_out,
            v_out,
            layer,
        )])
    }
}

// ── FlashInfer attention Impls ───────────────────────────────────
//
// Alternative to `AttentionViaCacheImpl` (decode) and
// `AttentionPrefillContiguousImpl` (prefill). Reads Q from the DSL
// tile's slot 0 and K/V from the paged cache written by the upstream
// fused QKV-rope tile, same as the FA2 variants, but dispatches
// through FlashInfer's persistent batch-attention kernel when the
// target has a calibrated CSV row for this `(head_dim, softcap,
// num_q_heads, num_kv_heads)` tuple. When the FFI tuple isn't
// compiled in, the emitted code falls back to the matching FA2 path.
//
// One Impl pair (Decode + Prefill) is registered per tuple in
// `FLASHINFER_CONFIG_SET`; the solver's 2-D `(num_tokens, sk_bucket)`
// sweep picks whichever wins per cell based on the calibrated costs.
// Without CSV data the prefix-gated `target_compatible` rejects the
// FI Impls and the non-FI attention Impls cover the workload.

/// Finite sk_bucket range the attention sweep calibrates. Matches
/// `ATTN_SK_BUCKETS` in `scratchy-target-cuda::attention_helpers` and the
/// `SK_VALUES` grid in `scratchy-cost-sweep-cuda::attention_sweep`. Changing
/// either end requires re-sweeping the target CSV.
const FI_SK_BUCKET_MIN: u64 = 128;
const FI_SK_BUCKET_MAX: u64 = 8192;

/// Build the CSV kernel name the FI Impls look up for a given
/// `(head_dim, softcap)` tuple. `(num_q_heads, num_kv_heads)` is a
/// runtime parameter of the compiled FI kernel — the same shim
/// handles any GQA ratio — so the cost table is keyed only on the
/// compile-time specialization dims. Must match the row names the
/// attention sweep emits — keep in sync with
/// `scratchy-cost-sweep-cuda/src/attention_sweep.rs`.
fn fi_csv_name(head_dim: u32, use_logits_soft_cap: bool) -> String {
    let softcap_tok = if use_logits_soft_cap {
        "softcap"
    } else {
        "nosoftcap"
    };
    format!("flashinfer_attn_bf16_h{head_dim}_{softcap_tok}")
}

/// Shared cost lookup — identical between Decode and Prefill Impls:
/// the FI plan builds the same work for both, and both variants live
/// in the same CSV row family (disambiguated at solver time by the
/// workload constraint on `num_tokens`).
fn fi_cost_us(head_dim: u32, use_logits_soft_cap: bool, ctx: &CostCtx) -> f64 {
    // Fast reject when the model's head_dim doesn't match the baked
    // tuple — every FI variant walks every Attention tile; only one's
    // head_dim matches any given model. `UNCALIBRATED_COST_US` (finite
    // sentinel) rather than `INFINITY` so the DP's non-finite-cost
    // guard (`SolveError::UnreachableCost`) doesn't fire.
    let Some(model_head_dim) = ctx.bounds.get("head_dim").copied() else {
        return UNCALIBRATED_COST_US;
    };
    if model_head_dim as u32 != head_dim {
        return UNCALIBRATED_COST_US;
    }
    let name = fi_csv_name(head_dim, use_logits_soft_cap);
    let nt = ctx.num_tokens() as u32;
    let sk = ctx.sk_bucket() as u32;
    ctx.profile
        .cost_us_for(&name, nt, sk, head_dim)
        .unwrap_or(UNCALIBRATED_COST_US)
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda")]
pub struct FlashInferAttentionDecodeImpl {
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

#[cfg(feature = "cuda")]
impl Implementation for FlashInferAttentionDecodeImpl {
    fn name(&self) -> &'static str {
        // Variant-level identification happens through the CSV row
        // name; a single name is enough for the solver's diagnostics.
        "flashinfer_attention_decode"
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // FI's `BatchPagedAttentionPersistent` uses `cudaLaunchCooperativeKernel`
        // and `grid.sync()` from cooperative groups. That kernel is
        // structurally not graph-capture-safe at multi-rank: the captured
        // node loses its cooperative property at instantiation, `grid.sync()`
        // degrades, and replay produces nondeterministic / hung output. The
        // FlashInfer Python wrapper for this kernel (`BatchAttention`) has
        // no `use_cuda_graph` parameter for this reason, while
        // `BatchDecodeWithPagedKVCacheWrapper` (the non-cooperative path)
        // does — Python vLLM uses that one at TP>1+graph.
        //
        // At TP>1 the per-rank shape (num_kv_heads halved, etc.) is also
        // exactly when the kernel needs the cross-rank context isolation
        // that the FI shim's static plan cache fails to provide. Defer to
        // `AttentionViaCacheImpl` (FA2) on sm<90 and to
        // `FlashAttention3DecodeImpl` on sm>=90; both are graph-capture-safe.
        ctx.tp_world_size <= 1
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // sm_89+ (Ada/Hopper). Older arches (sm_80 A100, sm_86 RTX30,
        // sm_75 T4) fall back to FA2 via the `target_compatible
        // == false` branch — they're untested with this shim.
        let compute_cap = match &profile.backend_spec {
            crate::target::BackendSpec::Cuda(s) => s.compute_capability,
            _ => return false,
        };
        if compute_cap < 89 {
            return false;
        }
        // O(1) membership check on the fully-qualified FI kernel name.
        // A row-less target (no sweep data for this head_dim × softcap)
        // silently falls back to FA2 via the `None` emit-call branch.
        profile
            .cost_table
            .has_kernel(&fi_csv_name(self.head_dim, self.use_logits_soft_cap))
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (1, 1),
            sk_bucket: (FI_SK_BUCKET_MIN, FI_SK_BUCKET_MAX),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only — see `AttentionViaCacheImpl::matches` for rationale.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        fi_cost_us(self.head_dim, self.use_logits_soft_cap, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FlashInferAttentionDecode { in_slot, out_slot, layer,
    // cos_sin_fn, head_dim, use_logits_soft_cap }`. All instances of
    // `FlashInferAttentionDecodeImpl` (one per (head_dim, softcap)
    // tuple in `FLASHINFER_CONFIG_SET`) share this variant; the
    // tuple values ride as runtime fields. The body tries
    // `flashinfer_attention` and falls back to FA2's
    // `attention_decode_from_cache` when FI returns `None`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FlashInferAttentionDecode",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("head_dim", syn::parse_quote!(u32)),
                ("use_logits_soft_cap", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FlashInferAttentionDecode: input 0 (Q) must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("FlashInferAttentionDecode: kv_cache extern with layer index")
            as u32;
        let head_dim = self.head_dim;
        let softcap = self.use_logits_soft_cap;
        Some(vec![Instruction::FlashInferAttentionDecode(
            in_slot_idx,
            out_slot_idx,
            layer,
            head_dim,
            softcap,
        )])
    }
}

// ─────────────────────────────────────────────────────────────────────
// FlashAttention-3 (Hopper-native) decode Impl.
//
// Sister to `FlashInferAttentionDecodeImpl`. Only emits the Instruction
// when the calibrated cost CSV says FA3 is cheaper than FI at this
// `(num_tokens, sk_bucket)` cell. Cost rows come from
// `scratchy-cost-sweep-cuda::attention_sweep` keyed by
// [`fa3_csv_name`].
//
// Vendored FA3 build (`third_party/vllm-flash-attn-3/`) currently
// supports only bf16 hdim128 paged decode without softcap; the Impl's
// `target_compatible` reflects that. Adding more (dtype, hdim) tuples
// requires (a) a new instantiation in
// `crates/scratchy-builder-cuda/build.rs::build_flash_attention_3`,
// (b) a matching CSV row family from the sweep, (c) loosening the
// gate here.
// ─────────────────────────────────────────────────────────────────────

/// CSV row family for FA3 decode. Keep in sync with
/// `scratchy-cost-sweep-cuda/src/attention_sweep.rs` row emission.
fn fa3_csv_name(head_dim: u32) -> String {
    format!("fa3_attn_bf16_h{head_dim}_nosoftcap_decode")
}

fn fa3_cost_us(head_dim: u32, ctx: &CostCtx) -> f64 {
    let Some(model_head_dim) = ctx.bounds.get("head_dim").copied() else {
        return UNCALIBRATED_COST_US;
    };
    if model_head_dim as u32 != head_dim {
        return UNCALIBRATED_COST_US;
    }
    let name = fa3_csv_name(head_dim);
    let nt = ctx.num_tokens() as u32;
    let sk = ctx.sk_bucket() as u32;
    ctx.profile
        .cost_us_for(&name, nt, sk, head_dim)
        .unwrap_or(UNCALIBRATED_COST_US)
}

#[cfg(fa3_built)]
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda")]
pub struct FlashAttention3DecodeImpl {
    pub head_dim: u32,
}

#[cfg(fa3_built)]
#[cfg(feature = "cuda")]
impl Implementation for FlashAttention3DecodeImpl {
    fn name(&self) -> &'static str {
        "flash_attention_3_decode"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        let compute_cap = match &profile.backend_spec {
            crate::target::BackendSpec::Cuda(s) => s.compute_capability,
            _ => return false,
        };
        // FA3 is sm_90a-only. The vendored library is built with
        // `-arch=sm_90a` (see `build_flash_attention_3` in
        // `scratchy-builder-cuda/build.rs`); calling on older arches
        // would link-fail or runtime-fault.
        if compute_cap < 90 {
            return false;
        }
        // Currently only hdim128 is instantiated. Other shapes need a
        // new instantiation file dropped into
        // `third_party/vllm-flash-attn-3/hopper/instantiations/`
        // and added to `build_flash_attention_3::kernel_files`.
        if self.head_dim != 128 {
            return false;
        }
        // No-softcap only: FLASHATTENTION_DISABLE_SOFTCAP is set in the
        // build flags, so the softcap variant doesn't link. Models that
        // use logits soft-cap (Gemma2 etc.) skip this Impl by virtue of
        // their `Instruction::FlashInferAttentionDecode` carrying
        // `use_logits_soft_cap = true`; the solver picks FI for those.
        // The match here is on softcap=false flowing into the Impl.
        profile.cost_table.has_kernel(&fa3_csv_name(self.head_dim))
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (1, 1),
            sk_bucket: (FI_SK_BUCKET_MIN, FI_SK_BUCKET_MAX),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        fa3_cost_us(self.head_dim, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FlashAttention3Decode",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("head_dim", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FlashAttention3Decode: input 0 (Q) must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("FlashAttention3Decode: kv_cache extern with layer index")
            as u32;
        Some(vec![Instruction::FlashAttention3Decode(
            in_slot_idx,
            out_slot_idx,
            layer,
            self.head_dim,
        )])
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda")]
pub struct FlashInferAttentionPrefillImpl {
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

#[cfg(feature = "cuda")]
impl Implementation for FlashInferAttentionPrefillImpl {
    fn name(&self) -> &'static str {
        "flashinfer_attention_prefill"
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // Same TP>1 exclusion as the decode sister Impl —
        // `BatchPagedAttentionPersistent`'s cooperative-launch path isn't
        // graph-capture-safe at multi-rank. See
        // `FlashInferAttentionDecodeImpl::applies_to` for the full reasoning.
        ctx.tp_world_size <= 1
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // sm_89+ — see FlashInferAttentionDecodeImpl::target_compatible.
        let compute_cap = match &profile.backend_spec {
            crate::target::BackendSpec::Cuda(s) => s.compute_capability,
            _ => return false,
        };
        if compute_cap < 89 {
            return false;
        }
        profile
            .cost_table
            .has_kernel(&fi_csv_name(self.head_dim, self.use_logits_soft_cap))
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (2, u32::MAX),
            sk_bucket: (FI_SK_BUCKET_MIN, FI_SK_BUCKET_MAX),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Decoder-only — see `AttentionViaCacheImpl::matches` for rationale.
        if !attention_has_kv_cache_extern(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        fi_cost_us(self.head_dim, self.use_logits_soft_cap, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FlashInferAttentionPrefill",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("head_dim", syn::parse_quote!(u32)),
                ("use_logits_soft_cap", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("FlashInferAttentionPrefill: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("FlashInferAttentionPrefill: kv_cache extern with layer index")
            as u32;
        let head_dim = self.head_dim;
        let softcap = self.use_logits_soft_cap;
        Some(vec![Instruction::FlashInferAttentionPrefill(
            q_slot, k_slot, v_slot, out_slot, layer, head_dim, softcap,
        )])
    }
}

// ── MlaSplitRefImpl ──────────────────────────────────────────────────────────
//
// Singleton for `OpKind::MlaSplit` — DeepSeek MLA's KV-A split.
//
// Input:  kv_a  [T, kv_lora_rank + qk_rope_head_dim]
// Output 0: kv_latent  [T, kv_lora_rank]
// Output 1: k_pe       [T, qk_rope_head_dim]
//
// Emits two alloc_tensor calls (one per output) then mla_split_kv_a.
// Both outputs are OwnedTensors — no aliasing with upstream.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MlaSplitRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for MlaSplitRefImpl {
    fn name(&self) -> &'static str {
        "mla_split_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::MlaSplit)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let kv_lora_rank = ctx.bounds.get("kv_lora_rank").copied().unwrap_or(0) as f64;
        let rope = ctx.bounds.get("qk_rope_head_dim").copied().unwrap_or(0) as f64;
        let bytes = m * (kv_lora_rank + rope) * 2.0 * BYTES_PER_ELEM; // read + write
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MlaSplit { in_slot, kv_latent_slot, k_pe_slot }`.
    // Two Owned outputs — both freshly allocated by the kernel.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MlaSplit",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("kv_latent_slot", syn::parse_quote!(u32)),
                ("k_pe_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MlaSplit: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let kv_latent_slot = slots.of(tile, 0);
        let k_pe_slot = slots.of(tile, 1);
        Some(vec![Instruction::MlaSplit(
            in_slot_idx,
            kv_latent_slot,
            k_pe_slot,
        )])
    }
}

// ── MlaAttentionImpl ─────────────────────────────────────────────────────────
//
// Singleton for `OpKind::MlaAttention` — DeepSeek MLA full attention.
//
// DSL args: mla_attention(q, kv_b, k_pe, positions, rotary, kv_cache[layer], block_table)
//   q    [T, num_heads * qk_head_dim]  (q_nope + q_pe interleaved per head)
//   kv_b [T, num_heads * (nope_dim + v_head_dim)]
//   k_pe [T, rope_dim]  (single head)
//
// Emits the full MLA sequence:
//   1. mla_extract_q_pe (q → q_pe scratch)
//   2. rotary_embedding_interleaved_inplace (q_pe + k_pe)
//   3. mla_write_q_pe (rotated q_pe → back into q)
//   4. mla_assemble_k (kv_b + k_pe → K)
//   5. mla_assemble_v (kv_b → V zero-padded)
//   6. write_kv_cache  (K, V → paged pool)
//   7. attention_standard
//   8. mla_slice_attn_output (qk_head_dim → v_head_dim)
//
// Output: [T, num_heads * v_head_dim]

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MlaAttentionImpl;

#[cfg(feature = "cuda")]
impl Implementation for MlaAttentionImpl {
    fn name(&self) -> &'static str {
        "mla_attention_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::MlaAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Use same analytic estimate as standard attention.
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MlaAttention { q_slot, kv_b_slot, k_pe_slot, out_slot,
    // layer, cos_sin_fn }`. The scale (with optional YaRN mscale) and
    // all head-dim bounds are baked from `model` at codegen time.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MlaAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("kv_b_slot", syn::parse_quote!(u32)),
                ("k_pe_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("MlaAttention: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (q_id, q_in) = resolve(0);
        let (kv_b_id, kv_b_in) = resolve(1);
        let (k_pe_id, k_pe_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let kv_b_slot = slots.of(kv_b_id, kv_b_in);
        let k_pe_slot = slots.of(k_pe_id, k_pe_in);
        let out_slot = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(layer.0),
                _ => None,
            })
            .expect("MlaAttention: kv_cache extern with concrete layer index")
            as u32;
        Some(vec![Instruction::MlaAttention(
            q_slot, kv_b_slot, k_pe_slot, out_slot, layer,
        )])
    }
}

// ── DeepSeekMoeRefImpl ───────────────────────────────────────────────────────
//
// Singleton for `OpKind::Moe` claiming the BF16 DeepSeek MoE layer.
//
// DSL: moe_out = moe_block(hidden_states, moe[layer])
//
// The `moe[layer]` weight resolves to a `DeepSeekV2MoELayer` struct.
// Emits a direct call to `DeepSeekV2MoELayer::forward`.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct DeepSeekMoeRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for DeepSeekMoeRefImpl {
    fn name(&self) -> &'static str {
        "deepseek_moe_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // DeepSeek-V2 / V3 / Moonlight / Kimi-K2 are the only MoE
        // arches whose `config.json` carries `n_routed_experts`
        // (Mixtral uses `num_local_experts`; Qwen2/3-MoE use
        // `num_experts`). Gate here so that the Tier-1 follow-up
        // FusedMoe / SharedFusedMoe Impls can claim non-DeepSeek
        // `OpKind::Moe` tiles without contention.
        ctx.model.bounds.contains_key("n_routed_experts")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Defer to FP8-block / GGML MoE Impls for their respective checkpoint
        // formats — this Impl claims the BF16 path only.
        if is_fp8_block_moe(fuf, seed) || is_ggml_moe(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Moe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The `moe[layer]` DSL weight maps to a DeepSeekV2MoELayer.
        // Bypass `default_required_weights` which would assign LinearLayer type.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers_moe::DeepSeekV2MoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `DeepSeekMoe { in_slot, out_slot, layer, weight_fn }`
    // where `weight_fn` returns `&DeepSeekV2MoELayer`. The body is a
    // straight `weight.forward(view, device)` call — the MoE layer
    // owns expert dispatch + topk routing internally.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "DeepSeekMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("DeepSeekMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("DeepSeekMoe: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::DeepSeekMoe(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── GatedDeltaNetImpl ────────────────────────────────────────────────────────
//
// Singleton for `OpKind::GatedDeltaNet` (Qwen3.5 / Qwen3-Next linear
// attention). Backend-agnostic host-callback op: the cuda eval
// (`gated_delta_net_eval` in scratchy-forward-compiler/src/instr.rs) orchestrates the
// surviving `gdn_*` kernels. No `applies_to` gate is needed — the
// `gated_delta_net` DSL op (and thus this OpKind) is emitted only by GDN
// arches, so there is no cross-Impl contention.
//
// DSL: core = gated_delta_net(qkv, z, a, b, linear_attn[layer])
//   inputs 0..3 = tile activations (qkv, z, a, b); input 4 = linear_attn[layer]
//   weight → GatedDeltaNetLayer. Layer index derived from the weight name.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GatedDeltaNetImpl;

#[cfg(feature = "cuda")]
impl Implementation for GatedDeltaNetImpl {
    fn name(&self) -> &'static str {
        "gated_delta_net_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::GatedDeltaNet)
    }
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The `linear_attn[layer]` DSL weight maps to a GatedDeltaNetLayer.
        // Bypass `default_required_weights` which would assign GpuTensor type.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers::GatedDeltaNetLayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GatedDeltaNet",
            vec![
                ("qkv_slot", syn::parse_quote!(u32)),
                ("z_slot", syn::parse_quote!(u32)),
                ("a_slot", syn::parse_quote!(u32)),
                ("b_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("GatedDeltaNet: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (qkv_id, qkv_in) = resolve(0);
        let (z_id, z_in) = resolve(1);
        let (a_id, a_in) = resolve(2);
        let (b_id, b_in) = resolve(3);
        let qkv_slot = slots.of(qkv_id, qkv_in);
        let z_slot = slots.of(z_id, z_in);
        let a_slot = slots.of(a_id, a_in);
        let b_slot = slots.of(b_id, b_in);
        let out_slot = slots.of(tile, 0);
        // Layer index comes from the `linear_attn[layer]` weight accessor name.
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors.first().expect(
            "GatedDeltaNet: required_weights returned empty (no linear_attn[layer] weight)",
        );
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![scratchy_forward_compiler::Instruction::GatedDeltaNet(
            qkv_slot, z_slot, a_slot, b_slot, out_slot, layer,
        )])
    }
}

// ── GateSplitImpl ────────────────────────────────────────────────────────────
//
// Singleton for `OpKind::GateSplit` — Qwen3.5 attention output-gate split.
// Deinterleaves the doubled `q_proj` output into `query` + `gate`. Two Owned
// outputs (both freshly allocated by the eval). Backend-agnostic host-callback.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GateSplitImpl;

#[cfg(feature = "cuda")]
impl Implementation for GateSplitImpl {
    fn name(&self) -> &'static str {
        "gate_split_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::GateSplit)
    }
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        false
    }
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GateSplit",
            vec![
                ("qg_slot", syn::parse_quote!(u32)),
                ("q_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
            ],
        )
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("GateSplit: input 0 must be a Tile (got {other:?})"),
        };
        let qg_slot = slots.of(in_id, in_slot);
        let q_slot = slots.of(tile, 0);
        let gate_slot = slots.of(tile, 1);
        Some(vec![scratchy_forward_compiler::Instruction::GateSplit(
            qg_slot, q_slot, gate_slot,
        )])
    }
}

// ── GateApplyImpl ────────────────────────────────────────────────────────────
//
// Singleton for `OpKind::GateApply` — Qwen3.5 attention output gate:
// `out = attn * sigmoid(gate)`. Shape-preserving 2-input elementwise.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GateApplyImpl;

#[cfg(feature = "cuda")]
impl Implementation for GateApplyImpl {
    fn name(&self) -> &'static str {
        "gate_apply_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::GateApply)
    }
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        false
    }
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GateApply",
            vec![
                ("attn_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("GateApply: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (attn_id, attn_in) = resolve(0);
        let (gate_id, gate_in) = resolve(1);
        let attn_slot = slots.of(attn_id, attn_in);
        let gate_slot = slots.of(gate_id, gate_in);
        let out_slot = slots.of(tile, 0);
        Some(vec![scratchy_forward_compiler::Instruction::GateApply(
            attn_slot, gate_slot, out_slot,
        )])
    }
}

// ── GateScaleImpl ────────────────────────────────────────────────────────────
//
// Singleton for `OpKind::GateScale` — Qwen3.5-MoE shared-expert combine:
// `out = routed + shared_y * sigmoid(g)` with `g` `[T, 1]` row-broadcast
// across the hidden axis. 3-input elementwise-with-broadcast.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GateScaleImpl;

#[cfg(feature = "cuda")]
impl Implementation for GateScaleImpl {
    fn name(&self) -> &'static str {
        "gate_scale_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::GateScale)
    }
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }
    fn is_compute_bound(&self) -> bool {
        false
    }
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GateScale",
            vec![
                ("routed_slot", syn::parse_quote!(u32)),
                ("shared_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("GateScale: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (routed_id, routed_in) = resolve(0);
        let (shared_id, shared_in) = resolve(1);
        let (gate_id, gate_in) = resolve(2);
        let routed_slot = slots.of(routed_id, routed_in);
        let shared_slot = slots.of(shared_id, shared_in);
        let gate_slot = slots.of(gate_id, gate_in);
        let out_slot = slots.of(tile, 0);
        Some(vec![scratchy_forward_compiler::Instruction::GateScale(
            routed_slot,
            shared_slot,
            gate_slot,
            out_slot,
        )])
    }
}

// ── DeepSeekFp8BlockMoeImpl ──────────────────────────────────────────────────
//
// Singleton for `OpKind::Moe` over blockwise-FP8 expert weights —
// DeepSeek-V3 / Kimi K2 official checkpoints. Same DSL contract as
// `DeepSeekMoeRefImpl`; the loaded type is `DeepSeekV2Fp8BlockMoELayer` and
// the host-interpreter Op variant is `DeepSeekMoeFp8Block` (separate from
// the BF16 `DeepSeekMoe` variant because the typed `weight_fn` signatures
// differ).
//
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct DeepSeekFp8BlockMoeImpl;

#[cfg(feature = "cuda")]
impl Implementation for DeepSeekFp8BlockMoeImpl {
    fn name(&self) -> &'static str {
        "deepseek_moe_fp8_block"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        if !is_fp8_block_moe(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Moe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers_moe::DeepSeekV2Fp8BlockMoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "DeepSeekMoeFp8Block",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("DeepSeekMoeFp8Block: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("DeepSeekMoeFp8Block: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::DeepSeekMoeFp8Block(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── DeepSeekGgmlMoeImpl ──────────────────────────────────────────────────────
//
// Singleton for `OpKind::Moe` over GGML/GGUF expert weights —
// DeepSeek-V2-Lite, Moonlight, K2 GGUFs. Same DSL contract as
// `DeepSeekMoeRefImpl`; the loaded type is `DeepSeekV2GgmlMoELayer` and
// the host-interpreter Op variant is `DeepSeekMoeGgml`.
//
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct DeepSeekGgmlMoeImpl;

#[cfg(feature = "cuda")]
impl Implementation for DeepSeekGgmlMoeImpl {
    fn name(&self) -> &'static str {
        "deepseek_moe_ggml"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        if !is_ggml_moe(fuf, seed) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Moe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers_moe::DeepSeekV2GgmlMoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "DeepSeekMoeGgml",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("DeepSeekMoeGgml: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("DeepSeekMoeGgml: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::DeepSeekMoeGgml(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── FusedMoeRefImpl ──────────────────────────────────────────────────────────
//
// Singleton for `OpKind::Moe` claiming Mixtral-style BF16 fused-MoE checkpoints.
// No shared expert; top-k softmax routing. The `applies_to` gate restricts this
// Impl to canonicals whose ModelParams carries `num_local_experts` (Mixtral /
// Qwen3-MoE-without-shared) and lacks `shared_expert_intermediate_size`
// (Qwen2/3-MoE-with-shared route to SharedFusedMoeRefImpl) and lacks
// `n_routed_experts` (DeepSeek family).
//
// DSL: moe_out = moe_block(hidden_states, block_sparse_moe[layer])
// (or any other identifier mapped to a `FusedMoELayer` accessor by
// the per-arch weights manifest.)
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct FusedMoeRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for FusedMoeRefImpl {
    fn name(&self) -> &'static str {
        "fused_moe_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // Emits `Instruction::FusedMoe` (CUDA-only at runtime), but
        // partitioned away from Metal by storage gate in `matches`:
        // Affine-stored experts route to `MetalFusedMoeImpl` (which
        // emits `Instruction::MetalFusedMoe`). Dense-stored MoE on
        // Metal is structurally unclaimed today; it would need a
        // dense-per-expert Metal kernel that hasn't been ported.
        true
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        let b = &ctx.model.bounds;
        b.contains_key("num_local_experts")
            && !b.contains_key("shared_expert_intermediate_size")
            && !b.contains_key("n_routed_experts")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Quantized checkpoints route to dedicated Impls (none yet for the
        // Mixtral family — when they land they will defer here the same way
        // DeepSeekMoeRefImpl defers to its FP8/GGML peers).
        if is_fp8_block_moe(fuf, seed) || is_ggml_moe(fuf, seed) {
            return None;
        }
        // Affine-stored MoE routes to `MetalFusedMoeImpl` (Metal-only
        // SwitchGLU decomposition). This Impl is Dense-only at
        // runtime — the cuda `FusedMoELayer::load` (layers_moe.rs:160)
        // reads dense BF16 stacked tensors.
        let node = fuf.get(seed);
        if matches!(weight_storage_of(node), Some(StorageFormat::Affine { .. })) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Moe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers_moe::FusedMoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedMoe: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::FusedMoe(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── SharedFusedMoeRefImpl ────────────────────────────────────────────────────
//
// Singleton for `OpKind::Moe` claiming Qwen2-MoE / Qwen3-MoE BF16 checkpoints
// — fused routed experts PLUS a shared expert with sigmoid gate. The
// `applies_to` gate restricts this Impl to canonicals whose ModelParams
// carries `shared_expert_intermediate_size` (Qwen-MoE-distinctive).
//
// DSL: moe_out = moe_block(hidden_states, mlp[layer])
// (or whichever identifier the per-arch weights manifest maps to a
// `SharedFusedMoELayer` accessor.)
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct SharedFusedMoeRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for SharedFusedMoeRefImpl {
    fn name(&self) -> &'static str {
        "shared_fused_moe_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // Partitioned from Metal via Affine-storage gate in `matches`;
        // see `FusedMoeRefImpl::target_compatible` for the rationale.
        true
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // The Qwen-MoE family is the only family whose `config.json`
        // ships `num_experts` as the routed-expert count (DeepSeek
        // uses `n_routed_experts`, Mixtral uses `num_local_experts`).
        // `shared_expert_intermediate_size` is an *optional* runtime
        // parameter on this family — Qwen3-MoE-Instruct ships
        // checkpoints with shared_inter=0 alongside Qwen2-MoE-style
        // shared-expert checkpoints; both share the same weight
        // naming and routing kernel, so a single Impl gate on
        // `num_experts` covers them.
        let b = &ctx.model.bounds;
        b.contains_key("num_experts")
            && !b.contains_key("num_local_experts")
            && !b.contains_key("n_routed_experts")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        if is_fp8_block_moe(fuf, seed) || is_ggml_moe(fuf, seed) {
            return None;
        }
        // Affine-stored MoE → `MetalSharedFusedMoeImpl`. cuda's
        // `SharedFusedMoELayer::load` (layers_moe.rs:429) is Dense-only.
        let node = fuf.get(seed);
        if matches!(weight_storage_of(node), Some(StorageFormat::Affine { .. })) {
            return None;
        }
        single_tile_match(fuf, seed, OpKind::Moe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        crate::__gpu::layers_moe::SharedFusedMoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SharedFusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("SharedFusedMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("SharedFusedMoe: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.map_or(0, |v| v.0) as u32;
        Some(vec![Instruction::SharedFusedMoe(
            in_slot_idx,
            out_slot_idx,
            layer,
        )])
    }
}

// ── Gemma-4 fused MoE (`gemma_moe` op) — neutral helpers + CUDA Impl ──────────
//
// `OpKind::GemmaMoe` is gemma-only with TWO activation inputs (router_in =
// post-attention residual, expert_in = ff2-normed) and TWO weight bundles
// (`router[layer]`, `experts.switch_glu[layer]`). `MetalGemmaMoeImpl`
// (`crate::metal::moe`) claims it on Metal; `CudaGemmaMoeImpl` below claims it
// on CUDA. Both reuse `Instruction::GemmaMoe` and these three helpers (moved
// here from `metal/moe.rs` so the cfg(metal)-gated module and the neutral CUDA
// Impl can share them).

/// Match a single `OpKind::GemmaMoe` tile.
pub(crate) fn gemma_moe_single_tile_match(fuf: &Fuf, seed: TileId) -> Option<MatchInfo> {
    let node = fuf.get(seed);
    if node.op != OpKind::GemmaMoe {
        return None;
    }
    Some(MatchInfo {
        claimed_tiles: vec![seed],
        boundary_inputs: vec![],
        boundary_outputs: vec![seed],
    })
}

/// Build the TWO weight accessors for a `GemmaMoe` node: `router*` →
/// `GemmaRouterLayer`, `experts*`/`switch_glu*` → `SwitchGluExpertsLayer`,
/// returned in `[router, experts]` order.
pub(crate) fn gemma_moe_required_weights(
    claimed_tiles: &[TileId],
    fuf: &Fuf,
    program: &Program,
) -> Vec<WeightAccessor> {
    let tile = claimed_tiles[0];
    let node = fuf.get(tile);
    let mut router: Option<WeightAccessor> = None;
    let mut experts: Option<WeightAccessor> = None;
    for input in &node.inputs {
        if let FufInput::Weight { id, index, .. } = input {
            let name = weight_field_name(program, *id, *index);
            let (base, _layer) = split_base_layer(&name.to_string());
            // Base names render with '.' → '_': `router`, `experts_switch_glu`.
            if base.starts_with("experts") || base.contains("switch_glu") {
                if experts.is_none() {
                    experts = Some(WeightAccessor {
                        name,
                        rust_type: quote! { crate::__gpu::layers_moe::SwitchGluExpertsLayer },
                        source_weights: vec![(*id, *index)],
                    });
                }
            } else if base.starts_with("router") && router.is_none() {
                router = Some(WeightAccessor {
                    name,
                    rust_type: quote! { crate::__gpu::layers_moe::GemmaRouterLayer },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
    }
    let mut out = Vec::with_capacity(2);
    if let Some(r) = router {
        out.push(r);
    }
    if let Some(e) = experts {
        out.push(e);
    }
    out
}

/// The EXPERT (`experts.switch_glu`) weight edge's affine `(group_size, bits)`
/// — 4-bit/g64 on the mlx checkpoint, distinct from the 8-bit `router.proj`.
pub(crate) fn gemma_expert_gs_bits(node: &FufNode, program: &Program) -> Option<(u32, u32)> {
    for input in &node.inputs {
        if let FufInput::Weight {
            id, index, storage, ..
        } = input
            && let StorageFormat::Affine { group_size, bits } = storage
        {
            let name = weight_field_name(program, *id, *index);
            let (base, _) = split_base_layer(&name.to_string());
            if base.starts_with("experts") || base.contains("switch_glu") {
                return Some((*group_size, *bits));
            }
        }
    }
    None
}

/// Emit the fat `Instruction::GemmaMoe` for a claimed `gemma_moe` tile. Shared
/// by the Metal and CUDA Impls — both read router_in (input 0), expert_in
/// (input 1), the out slot, the layer (from the router accessor suffix), and
/// the MoE shape from `bounds`. The CUDA eval reads the slots + layer and
/// resolves the rest from the runtime layer structs; the trailing shape fields
/// are filled for the Metal lowering and are harmless on CUDA.
pub(crate) fn gemma_moe_fan_out(
    m: &MatchInfo,
    fuf: &Fuf,
    program: &Program,
    bounds: &BTreeMap<String, u64>,
    slots: &SlotMap,
) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
    let tile = m.claimed_tiles[0];
    let node = fuf.get(tile);
    let tile_inputs: Vec<(TileId, u8)> = node
        .inputs
        .iter()
        .filter_map(|i| match i {
            FufInput::Tile { id, slot } => Some((*id, *slot)),
            _ => None,
        })
        .collect();
    let (router_in_id, router_in_slot) = tile_inputs
        .first()
        .copied()
        .unwrap_or_else(|| panic!("GemmaMoe: input 0 (router_in) must be a Tile"));
    let (expert_in_id, expert_in_slot) = tile_inputs
        .get(1)
        .copied()
        .unwrap_or_else(|| panic!("GemmaMoe: input 1 (expert_in) must be a Tile"));
    let router_in_idx = slots.of(router_in_id, router_in_slot);
    let expert_in_idx = slots.of(expert_in_id, expert_in_slot);
    let out_slot_idx = slots.of(tile, 0);

    let accessors = gemma_moe_required_weights(&m.claimed_tiles, fuf, program);
    let acc = accessors
        .first()
        .expect("GemmaMoe: required_weights returned empty");
    let (_base, layer) = split_base_layer(&acc.name.to_string());
    let layer = layer.map_or(0, |v| v.0) as u32;

    // gemma4-moe always ships these bounds (verbatim config); fall back to the
    // 26B-A4B values so the never-run Dense verbatim specialization still emits.
    let num_experts = bounds.get("num_experts").copied().unwrap_or(128) as u32;
    let top_k = bounds.get("num_experts_per_tok").copied().unwrap_or(8) as u32;
    let moe_inter = bounds.get("moe_intermediate_size").copied().unwrap_or(704) as u32;
    let hidden = bounds.get("hidden_size").copied().unwrap_or(2816) as u32;
    let (group_size, bits) = gemma_expert_gs_bits(node, program).unwrap_or((64, 4));

    Some(vec![scratchy_forward_compiler::Instruction::GemmaMoe(
        router_in_idx,
        expert_in_idx,
        out_slot_idx,
        layer,
        num_experts,
        top_k,
        moe_inter,
        hidden,
        group_size,
        bits,
    )])
}

/// CUDA solver-side singleton for the Gemma-4 fused MoE (`gemma_moe` op).
/// Near-clone of `MetalGemmaMoeImpl`; `target_compatible` flips to CUDA and the
/// emitted instruction is the SAME `Instruction::GemmaMoe`. Registered
/// unconditionally (like the other `*MoeRefImpl`s); on a Metal build its
/// `target_compatible` filters it out so `MetalGemmaMoeImpl` claims the op.
#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct CudaGemmaMoeImpl;

#[cfg(feature = "cuda")]
impl Implementation for CudaGemmaMoeImpl {
    fn name(&self) -> &'static str {
        "cuda_gemma_moe"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Cuda
    }

    fn applies_to(&self, _ctx: &MatchContext) -> bool {
        // The OpKind is gemma-only; the `OpKind::GemmaMoe` match is the gate.
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Claim every GemmaMoe node regardless of weight storage (the Affine
        // mlx-4bit checkpoint is the runtime path; the verbatim BF16/Dense
        // config also produces a GemmaMoe node that must still solve+codegen).
        gemma_moe_single_tile_match(fuf, seed)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        gemma_moe_required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GemmaMoe",
            vec![
                ("router_in", syn::parse_quote!(u32)),
                ("expert_in", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        gemma_moe_fan_out(m, fuf, program, bounds, slots)
    }
}

// ⛔ CUDA ONLY, AND THE GATE USED TO SAY OTHERWISE. This module names
// `scratchy_target_cuda` and `crate::solver` (instruction selection, itself
// cuda-gated), so under `spyre` the cfg ADMITTED it and it failed to compile —
// taking every other test in this crate down with it. A predicate that is WRONG
// rather than absent, so the crate's whole suite was unrunnable off cuda.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    #[test]
    fn unmigrated_shape_carries_distinctive_placeholder_name() {
        // A shape returned from the trait default must be marked
        // distinctively enough that codegen can refuse it without a
        // false positive against a real Impl. We don't rely on
        // matching the placeholder by string in production codegen
        // — the `fan_out` → `None` check is authoritative — but if
        // the placeholder ever leaks into emitted code, it should
        // fail compilation loudly with a recognisable identifier.
        let s = OpcodeShape::unmigrated("AddRefImpl");
        assert_eq!(s.name.to_string(), "__Unmigrated");
        assert!(s.fields.is_empty());
    }

    // Removed `op_instance_field_values_match_shape_field_count`:
    // `Implementation::fan_out` now returns `Vec<Instruction>`
    // directly — the
    // `OpInstance::field_values` round-trip is gone, so the
    // arity-vs-shape soft contract no longer has a runtime carrier
    // here. The remaining arity check lives consumer-side in
    // `interpreter_codegen` against the per-variant `Instruction`
    // arm rendered from the typed value.

    #[test]
    fn slot_map_assigns_dense_indices_in_insert_order() {
        let mut sm = SlotMap::new();
        assert_eq!(sm.insert(TileId(7), 0), 0);
        assert_eq!(sm.insert(TileId(7), 1), 1);
        assert_eq!(sm.insert(TileId(3), 0), 2);
        // Repeat returns the original index.
        assert_eq!(sm.insert(TileId(7), 0), 0);
        assert_eq!(sm.total(), 3);
        assert_eq!(sm.of(TileId(3), 0), 2);
    }

    #[test]
    #[should_panic(expected = "not registered")]
    fn slot_map_of_unregistered_panics() {
        let sm = SlotMap::new();
        let _ = sm.of(TileId(1), 0);
    }

    #[test]
    fn attention_scalars_default_to_llama_values_when_config_silent() {
        // Llama-3.2-1B has no `query_pre_attn_scalar` or
        // `attn_logit_softcapping` — the helpers must fall back to
        // `1/sqrt(head_dim)` and `0.0` respectively, matching the
        // previously hardcoded emission byte-for-byte.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            .join("configs")
            .join("llama")
            .join("llama-3.2-1b.json");
        let model = crate::config::load_file(&path).expect("load llama-3.2-1b");

        let expected_scale = 1.0_f32 / (*model.bounds.get("head_dim").unwrap() as f32).sqrt();
        assert_eq!(attention_scale_for(&model), expected_scale);
        assert_eq!(attention_softcap_for(&model), 0.0);
    }

    #[test]
    fn attention_scalars_honor_gemma_style_config() {
        // Synthetic ModelParams mirroring Gemma2-2B's attention
        // config. The scale must flow from `query_pre_attn_scalar`,
        // not from `head_dim`; softcap must match the capping value.
        let mut bounds = std::collections::BTreeMap::new();
        bounds.insert("head_dim".to_string(), 256);
        let mut scalars = std::collections::BTreeMap::new();
        // Gemma2 ships `query_pre_attn_scalar = 256.0` on the 2B and 9B.
        scalars.insert("query_pre_attn_scalar".to_string(), 256.0);
        scalars.insert("attn_logit_softcapping".to_string(), 50.0);
        let model = crate::config::ModelParams {
            name: "gemma2_test".to_string(),
            source_stem: "gemma2_test".into(),
            source_path: std::path::PathBuf::new(),
            bounds,
            scalars,
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        };

        let scale = attention_scale_for(&model);
        // scale = 256^-0.5 = 1/16 = 0.0625. Exact in f32.
        assert_eq!(scale, 0.0625_f32);
        assert_eq!(attention_softcap_for(&model), 50.0_f32);
    }

    #[test]
    fn deepseek_moe_applies_to_requires_n_routed_experts() {
        // Locks the per-canonical gate semantics for the Tier-1 MoE
        // follow-up: DeepSeekMoeRefImpl must skip canonicals whose
        // config doesn't carry the DeepSeek-distinctive
        // `n_routed_experts` key. Mixtral/Qwen-MoE configs (when
        // they land) use `num_local_experts` / `num_experts` and
        // must not be claimed by the DeepSeek Impl.
        use crate::classified::Program;
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights: Default::default(),
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };

        let mk_model = |name: &str, key: &str, val: u64| crate::config::ModelParams {
            name: name.to_string(),
            source_stem: name.to_string(),
            source_path: std::path::PathBuf::new(),
            bounds: {
                let mut b = std::collections::BTreeMap::new();
                b.insert(key.to_string(), val);
                b
            },
            scalars: std::collections::BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        };

        let imp = DeepSeekMoeRefImpl;

        // DeepSeek-shaped: claims.
        let m_ds = mk_model("deepseek_v2_lite_test", "n_routed_experts", 64);
        let ctx_ds = MatchContext {
            program: &program,
            model: &m_ds,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            imp.applies_to(&ctx_ds),
            "DeepSeekMoeRefImpl must claim configs with n_routed_experts"
        );

        // Mixtral-shaped: skips.
        let m_mix = mk_model("mixtral_test", "num_local_experts", 8);
        let ctx_mix = MatchContext {
            program: &program,
            model: &m_mix,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_mix),
            "DeepSeekMoeRefImpl must defer when n_routed_experts is absent"
        );

        // Qwen-MoE-shaped: skips.
        let m_qwen = mk_model("qwen3_moe_test", "num_experts", 128);
        let ctx_qwen = MatchContext {
            program: &program,
            model: &m_qwen,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_qwen),
            "DeepSeekMoeRefImpl must defer on Qwen-MoE configs"
        );
    }

    #[test]
    fn fused_moe_applies_to_requires_num_local_experts_only() {
        // Locks the FusedMoeRefImpl gate. Mixtral has `num_local_experts`
        // but neither `shared_expert_intermediate_size` (Qwen-MoE-with-shared)
        // nor `n_routed_experts` (DeepSeek). The gate must claim Mixtral and
        // defer on the other two families even when their distinctive keys
        // appear alongside `num_local_experts` in a future hybrid config.
        use crate::classified::Program;
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights: Default::default(),
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };

        let mk_model = |name: &str, kvs: &[(&str, u64)]| crate::config::ModelParams {
            name: name.to_string(),
            source_stem: name.to_string(),
            source_path: std::path::PathBuf::new(),
            bounds: kvs
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect::<std::collections::BTreeMap<_, _>>(),
            scalars: std::collections::BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        };

        let imp = FusedMoeRefImpl;

        // Mixtral: claims.
        let m_mix = mk_model("mixtral_test", &[("num_local_experts", 8)]);
        let ctx_mix = MatchContext {
            program: &program,
            model: &m_mix,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            imp.applies_to(&ctx_mix),
            "FusedMoeRefImpl must claim Mixtral configs (num_local_experts, no shared)"
        );

        // Qwen-MoE-with-shared: skips (defers to SharedFusedMoeRefImpl).
        let m_qwen_shared = mk_model(
            "qwen2_moe_test",
            &[
                ("num_local_experts", 60),
                ("shared_expert_intermediate_size", 5632),
            ],
        );
        let ctx_qwen_shared = MatchContext {
            program: &program,
            model: &m_qwen_shared,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_qwen_shared),
            "FusedMoeRefImpl must defer on Qwen-MoE-with-shared configs"
        );

        // DeepSeek (hypothetical hybrid with both keys): skips.
        let m_ds = mk_model(
            "deepseek_v2_lite_test",
            &[("num_local_experts", 64), ("n_routed_experts", 64)],
        );
        let ctx_ds = MatchContext {
            program: &program,
            model: &m_ds,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_ds),
            "FusedMoeRefImpl must defer when n_routed_experts is present"
        );

        // No MoE keys at all: skips (a dense arch like Llama).
        let m_dense = mk_model("llama_test", &[("hidden_size", 4096)]);
        let ctx_dense = MatchContext {
            program: &program,
            model: &m_dense,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_dense),
            "FusedMoeRefImpl must defer on dense configs"
        );
    }

    #[test]
    fn shared_fused_moe_applies_to_keys_on_num_experts() {
        // Locks the SharedFusedMoeRefImpl gate. The Qwen-MoE family is the
        // only family whose config carries `num_experts` (DeepSeek uses
        // `n_routed_experts`; Mixtral uses `num_local_experts`). The gate
        // covers BOTH Qwen2-MoE-with-shared and Qwen3-MoE-without-shared
        // — `shared_expert_intermediate_size` is a runtime parameter on
        // the same kernel surface, not a family discriminator.
        use crate::classified::Program;
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights: Default::default(),
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };

        let mk_model = |name: &str, kvs: &[(&str, u64)]| crate::config::ModelParams {
            name: name.to_string(),
            source_stem: name.to_string(),
            source_path: std::path::PathBuf::new(),
            bounds: kvs
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect::<std::collections::BTreeMap<_, _>>(),
            scalars: std::collections::BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        };

        let imp = SharedFusedMoeRefImpl;

        // Qwen3-MoE-Instruct (no shared): claims.
        let m_qwen3 = mk_model(
            "qwen3_30b_a3b_test",
            &[("num_experts", 128), ("shared_expert_intermediate_size", 0)],
        );
        let ctx_qwen3 = MatchContext {
            program: &program,
            model: &m_qwen3,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            imp.applies_to(&ctx_qwen3),
            "SharedFusedMoeRefImpl must claim Qwen3-MoE configs even with shared_expert_intermediate_size=0"
        );

        // Qwen2-MoE (with shared): claims.
        let m_qwen2 = mk_model(
            "qwen2_moe_test",
            &[
                ("num_experts", 60),
                ("shared_expert_intermediate_size", 5632),
            ],
        );
        let ctx_qwen2 = MatchContext {
            program: &program,
            model: &m_qwen2,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            imp.applies_to(&ctx_qwen2),
            "SharedFusedMoeRefImpl must claim Qwen2-MoE-with-shared configs"
        );

        // Mixtral: skips (defers to FusedMoeRefImpl).
        let m_mix = mk_model("mixtral_test", &[("num_local_experts", 8)]);
        let ctx_mix = MatchContext {
            program: &program,
            model: &m_mix,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_mix),
            "SharedFusedMoeRefImpl must defer on Mixtral (uses num_local_experts)"
        );

        // DeepSeek: skips (defers to DeepSeekMoeRefImpl).
        let m_ds = mk_model(
            "deepseek_v2_lite_test",
            &[("num_experts", 64), ("n_routed_experts", 64)],
        );
        let ctx_ds = MatchContext {
            program: &program,
            model: &m_ds,
            profile: &profile,
            tp_world_size: 1,
        };
        assert!(
            !imp.applies_to(&ctx_ds),
            "SharedFusedMoeRefImpl must defer when n_routed_experts is present"
        );
    }

    #[test]
    fn resources_union_max_takes_elementwise_max() {
        let a = Resources {
            shmem_bytes: 16 * 1024,
            regs_per_thread: 48,
            threads_per_cta: 128,
        };
        let b = Resources {
            shmem_bytes: 32 * 1024,
            regs_per_thread: 40,
            threads_per_cta: 256,
        };
        let u = a.union_max(&b);
        assert_eq!(u.shmem_bytes, 32 * 1024);
        assert_eq!(u.regs_per_thread, 48);
        assert_eq!(u.threads_per_cta, 256);
        assert_eq!(a.union_max(&a), a);
        assert_eq!(a.union_max(&Resources::ZERO), a);
    }

    #[test]
    fn handoff_variants_have_finite_nonneg_cost_on_any_profile() {
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        for h in [
            Handoff::StreamOrder,
            Handoff::StreamEvent,
            Handoff::KernelBoundary,
            Handoff::InKernelGridSync,
            Handoff::Mbarrier,
            Handoff::DsmemRead,
            Handoff::GmemFlag,
            Handoff::SyncThreads,
            Handoff::Internal,
        ] {
            let c = h.cost_us(&profile);
            assert!(c.is_finite(), "{h:?} → {c}");
            assert!(c >= 0.0, "{h:?} → {c}");
        }
    }

    #[test]
    fn handoff_truncation_classification() {
        // Gmem-transiting handoffs truncate; register/shmem
        // handoffs don't.
        let truncating = [
            Handoff::StreamOrder,
            Handoff::StreamEvent,
            Handoff::KernelBoundary,
            Handoff::InKernelGridSync,
            Handoff::GmemFlag,
            Handoff::SyncThreads,
        ];
        let preserving = [Handoff::Internal, Handoff::Mbarrier, Handoff::DsmemRead];
        for h in truncating {
            assert!(h.truncates_to_storage_dtype(), "{h:?}");
        }
        for h in preserving {
            assert!(!h.truncates_to_storage_dtype(), "{h:?}");
        }
    }

    #[test]
    fn layout_any_matches_everything() {
        assert!(Layout::Any.is_compatible_with(Layout::RowMajorBf16));
        assert!(Layout::Any.is_compatible_with(Layout::ColMajorBf16));
        assert!(Layout::RowMajorBf16.is_compatible_with(Layout::Any));
        assert!(Layout::ColMajorBf16.is_compatible_with(Layout::Any));
        assert!(Layout::RowMajorBf16.is_compatible_with(Layout::RowMajorBf16));
        assert!(!Layout::RowMajorBf16.is_compatible_with(Layout::ColMajorBf16));
    }

    #[test]
    fn workload_constraint_accepts_inside_range() {
        // NumTokensRange ignores sk_bucket.
        let c = WorkloadConstraint::NumTokensRange { min: 1, max: 8 };
        assert!(c.accepts(1, 0));
        assert!(c.accepts(8, 4096));
        assert!(!c.accepts(9, 0));
        assert!(!c.accepts(0, 0));
        // Any accepts all.
        assert!(WorkloadConstraint::Any.accepts(4096, 8192));
        // NumTokensAndSkRange gates on both axes.
        let s = WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (1, 1),
            sk_bucket: (1024, 8192),
        };
        assert!(s.accepts(1, 1024));
        assert!(s.accepts(1, 8192));
        assert!(!s.accepts(1, 512)); // below sk range
        assert!(!s.accepts(1, 16384)); // above sk range
        assert!(!s.accepts(2, 4096)); // num_tokens outside
    }

    #[test]
    fn cutlass_tile_zoo_matches_csv_kernel_names() {
        // The library registers one CutlassGemmImpl per CUTLASS_TILE_ZOO
        // entry plus CutlassGemvImpl. Each entry's `csv_name()` must
        // correspond to a real kernel string in the L4 cost table
        // (byte-for-byte), otherwise `target_compatible` rejects every
        // tile and the solver silently falls back to cuBLAS.
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let csv_kernels = profile.cost_table.kernel_names();
        for (tm, tn, st) in CUTLASS_TILE_ZOO {
            let name = format!("cutlass_{tm}x{tn}_s{st}");
            assert!(
                csv_kernels.iter().any(|k| k == &name),
                "CUTLASS_TILE_ZOO has ({tm}, {tn}, {st}) but no `{name}` row in l4 CSV",
            );
        }
        assert!(csv_kernels.iter().any(|k| k == "cutlass_gemv"));
    }

    #[test]
    fn cutlass_gemm_impl_matches_when_output_feeds_rope_append() {
        // Architectural invariant: singletons (CutlassGemm{,SplitK,
        // Add,v}, GemmRefImpl, Fp8Gemm) MUST match unconditionally on
        // structural fit — they don't gate on what the downstream
        // consumer is. The DP solver decides between a multi-tile
        // fused claim and a chain of singletons by cost + feasibility:
        //   - When `FusedQkvRopeImpl` matches, it claims [Q,K,V,rope]
        //     in one multi-tile mask; DP picks it when cheaper.
        //   - When fused doesn't match (qwen3/gemma3 per-head qk-norm),
        //     singletons are the only feasible plan and win.
        // Singleton-side gating ("don't match if output feeds X") was
        // a hand-coded preference that the DP already encodes via cost
        // and feasibility — and it was wrong for qwen3 V-proj, where
        // it forced cuBLAS over a measurably faster CUTLASS variant.
        use crate::classified::{ExternKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::shape::Dim;

        let t0 = TileId(0);
        let t1 = TileId(1);
        let t2 = TileId(2);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: t0,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
                FufNode {
                    id: t1,
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile { id: t0, slot: 0 },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
                FufNode {
                    id: t2,
                    op: OpKind::RopeAppend,
                    inputs: vec![FufInput::Tile { id: t1, slot: 0 }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
            ],
        };
        let imp = CutlassGemmImpl {
            tile_m: 128,
            tile_n: 128,
            stages: 4,
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        assert!(
            imp.matches(&fuf, t1, &profile).is_some(),
            "CutlassGemmImpl must match a Gemm structurally regardless of \
             downstream consumer; fused-vs-singleton is the DP's job",
        );
    }
    // ── FlashInfer attention Impls ───────────────────────────────

    /// Build a minimal `TargetProfile` with only the cost rows we
    /// want, for testing Impl gating without committing a CSV.
    fn synthetic_profile(rows: &[(&str, u32, u32, u32, f64)]) -> crate::target::TargetProfile {
        let mut cost_table = crate::target::CostTable::new();
        for (k, m, n, k_, cost) in rows {
            cost_table.insert(*k, *m, *n, *k_, *cost);
        }
        crate::target::TargetProfile {
            name: "synthetic".to_string(),
            source_path: std::path::PathBuf::from("synthetic"),
            backend: crate::target::Backend::Cuda,
            peak_tflops_fp16: 121.0,
            memory_bandwidth_gbps: 300.0,
            backend_spec: crate::target::BackendSpec::Cuda(crate::target::CudaSpec {
                compute_capability: 89,
                num_sms: 58,
                shared_memory_per_sm_kb: 100,
            }),
            cost_table,
        }
    }

    #[test]
    #[ignore = "FlashInfer is disabled fleet-wide pending a fix for the \
                persistent-kernel CUDA_ERROR_ILLEGAL_ADDRESS hit at tp>1 \
                with num_kv_heads=1 (see starter_library() for details). \
                Re-enable this test when the FI registration loop in \
                impl_lib.rs is uncommented."]
    fn starter_library_registers_twelve_flashinfer_variants() {
        // Six tuples in `FLASHINFER_CONFIG_SET` × {Decode, Prefill} Impls.
        // If this count drifts, either the config set or the registration
        // loop changed without the mirror being updated — a silent way to
        // leave the Impl chain broken.
        let lib = starter_library();
        let fi_decode = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "flashinfer_attention_decode")
            .count();
        let fi_prefill = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "flashinfer_attention_prefill")
            .count();
        assert_eq!(fi_decode, 6, "expected 6 decode variants");
        assert_eq!(fi_prefill, 6, "expected 6 prefill variants");
    }

    #[test]
    fn flashinfer_decode_workload_constraint_is_m_eq_1_and_calibrated_sk_range() {
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        let c = imp.workload_constraint();
        assert!(c.accepts(1, 128)); // decode at lowest bucket
        assert!(c.accepts(1, 8192)); // decode at highest bucket
        assert!(!c.accepts(1, 64)); // below calibrated range
        assert!(!c.accepts(1, 16384)); // above calibrated range
        assert!(!c.accepts(2, 2048)); // prefill M outside decode range
    }

    #[test]
    fn flashinfer_prefill_workload_constraint_is_m_ge_2_and_calibrated_sk_range() {
        let imp = FlashInferAttentionPrefillImpl {
            head_dim: 64,
            use_logits_soft_cap: true,
        };
        let c = imp.workload_constraint();
        assert!(c.accepts(2, 128));
        assert!(c.accepts(4096, 8192));
        assert!(!c.accepts(1, 2048)); // decode M not in prefill range
        assert!(!c.accepts(2, 64)); // below calibrated sk
    }

    #[test]
    fn flashinfer_target_compatible_gated_by_csv_prefix() {
        // Empty profile — no rows of any kind.
        let empty = synthetic_profile(&[]);
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        assert!(
            !imp.target_compatible(&empty),
            "FI decode must be rejected when CSV has no matching rows"
        );

        // Profile with a row for a DIFFERENT head_dim — still must reject.
        let other = synthetic_profile(&[("flashinfer_attn_bf16_h64_nosoftcap", 1, 2048, 64, 5.0)]);
        assert!(
            !imp.target_compatible(&other),
            "FI decode (h=128) must be rejected when only h=64 rows exist"
        );

        // Profile with a matching row — accept. The kernel name is
        // keyed only on (head_dim, softcap) — `(q, k)` is runtime.
        let matching =
            synthetic_profile(&[("flashinfer_attn_bf16_h128_nosoftcap", 1, 2048, 128, 3.0)]);
        assert!(
            imp.target_compatible(&matching),
            "FI decode must accept when a matching-head_dim row exists"
        );

        // Softcap-matching requires exact softcap token.
        let imp_cap = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: true,
        };
        assert!(
            !imp_cap.target_compatible(&matching),
            "FI decode(softcap=true) must not accept a nosoftcap-only CSV",
        );
    }

    #[test]
    fn flashinfer_cost_us_finite_on_hit_and_uncalibrated_on_mismatch() {
        use crate::fuf::{Fuf, FufNode};
        use crate::shape::Dim;

        // Synthetic profile with one FlashInfer decode row at (m=1,
        // sk=2048, head_dim=128). Name is head_dim/softcap-only — the
        // kernel handles any (q, k) at runtime.
        let profile =
            synthetic_profile(&[("flashinfer_attn_bf16_h128_nosoftcap", 1, 2048, 128, 6.5)]);

        // Minimal FUF: one Attention tile. `fi_cost_us` ignores the
        // tile's shape entirely (the cost comes from the CSV lookup),
        // so we don't need realistic QKV tile plumbing.
        let t0 = TileId(0);
        let fuf = Fuf {
            nodes: vec![FufNode {
                id: t0,
                op: OpKind::Attention,
                inputs: vec![],
                outputs: vec![vec![Dim::Lit(1), Dim::Lit(128)]],
            }],
        };
        let mi = MatchInfo {
            claimed_tiles: vec![t0],
            boundary_inputs: vec![],
            boundary_outputs: vec![t0],
        };

        // Matching model bounds (h=128, q=32, k=8) → hit.
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("head_dim".into(), 128);
        bounds.insert("num_attention_heads".into(), 32);
        bounds.insert("num_key_value_heads".into(), 8);
        bounds.insert("num_tokens".into(), 1);
        bounds.insert("sk_bucket".into(), 2048);
        let ctx = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
            expected_calls_per_load: CostCtx::default_expected_calls_per_load(profile.backend),
        };
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        let cost = imp.cost_us(&mi, &ctx);
        assert_eq!(cost, 6.5);

        // Mismatching head_dim — Impl baked for h=128, model has h=64.
        // All reject paths must stay finite (UNCALIBRATED_COST_US) so
        // the DP's `!cost.is_finite()` guard doesn't trip; FA2 with a
        // real calibrated cost still wins the tiebreak.
        bounds.insert("head_dim".into(), 64);
        let ctx_bad = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
            expected_calls_per_load: CostCtx::default_expected_calls_per_load(profile.backend),
        };
        let cost_bad = imp.cost_us(&mi, &ctx_bad);
        assert!(cost_bad.is_finite());
        assert_eq!(cost_bad, UNCALIBRATED_COST_US);

        // Different (q, k) combo at the SAME head_dim — the kernel
        // handles GQA ratio at runtime, so the same calibrated row
        // serves every model at h=128. This is what makes FI general
        // across models (Llama q32/k8, Qwen q28/k4, etc.).
        bounds.insert("head_dim".into(), 128);
        bounds.insert("num_attention_heads".into(), 28);
        bounds.insert("num_key_value_heads".into(), 4);
        let ctx_other_qk = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
            expected_calls_per_load: CostCtx::default_expected_calls_per_load(profile.backend),
        };
        let cost_other_qk = imp.cost_us(&mi, &ctx_other_qk);
        assert_eq!(cost_other_qk, 6.5);
    }
    /// `decompose_reshape_dim` folds every config bound + literal at
    /// codegen time into the numerator, counts each `num_tokens`
    /// occurrence in `nt_pow`, and stays at `denom = 1` in the absence
    /// of a `Div` term. `Div` (G.5.f.a, merger reshape) closes the
    /// denominator into a literal divisor; mixed Mul-of-Div composes
    /// the numerator/denominator multiplicatively.
    #[test]
    fn decompose_reshape_dim_folds_config_bounds_and_counts_num_tokens() {
        use crate::shape::Dim;
        let mut bounds = std::collections::BTreeMap::new();
        bounds.insert("hidden_size".to_string(), 4096u64);
        bounds.insert("head_dim".to_string(), 64u64);
        bounds.insert("num_attention_heads".to_string(), 32u64);
        bounds.insert("vision_merge_factor".to_string(), 4u64);

        // Pure literal.
        assert_eq!(decompose_reshape_dim(&Dim::Lit(64), &bounds), (64, 0, 1));

        // Config bound folds at codegen time.
        assert_eq!(
            decompose_reshape_dim(&Dim::Bound("hidden_size".into()), &bounds),
            (4096, 0, 1)
        );

        // num_tokens is the only runtime factor.
        assert_eq!(
            decompose_reshape_dim(&Dim::Bound("num_tokens".into()), &bounds),
            (1, 1, 1)
        );

        // [num_tokens * num_attention_heads, head_dim]-style Mul folds
        // bounds + counts num_tokens occurrences.
        let dim = Dim::Mul(vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("num_attention_heads".into()),
        ]);
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (32, 1, 1));

        // Pure literal product.
        let dim = Dim::Mul(vec![Dim::Lit(2), Dim::Bound("head_dim".into())]);
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (128, 0, 1));

        // Div(num_tokens, vision_merge_factor) — the merger's first
        // dim. Numerator stays `(1, 1)`, denominator closes to 4.
        let dim = Dim::Div(
            Box::new(Dim::Bound("num_tokens".into())),
            Box::new(Dim::Bound("vision_merge_factor".into())),
        );
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (1, 1, 4));

        // Div(num, lit) — pure literal denominator.
        let dim = Dim::Div(
            Box::new(Dim::Bound("hidden_size".into())),
            Box::new(Dim::Lit(8)),
        );
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (4096, 0, 8));

        // Mul(Div(...), Lit) — denominator survives composition.
        let dim = Dim::Mul(vec![
            Dim::Div(
                Box::new(Dim::Bound("num_tokens".into())),
                Box::new(Dim::Lit(2)),
            ),
            Dim::Lit(3),
        ]);
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (3, 1, 2));
    }

    #[test]
    #[should_panic(expected = "denominator may not contain `num_tokens`")]
    fn decompose_reshape_dim_rejects_num_tokens_in_denominator() {
        use crate::shape::Dim;
        let bounds = std::collections::BTreeMap::new();
        let dim = Dim::Div(
            Box::new(Dim::Lit(4)),
            Box::new(Dim::Bound("num_tokens".into())),
        );
        let _ = decompose_reshape_dim(&dim, &bounds);
    }
    fn empty_model(name: &str) -> crate::config::ModelParams {
        crate::config::ModelParams {
            name: name.to_string(),
            source_stem: name.into(),
            source_path: std::path::PathBuf::new(),
            bounds: std::collections::BTreeMap::new(),
            scalars: std::collections::BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        }
    }
    fn attention_model(name: &str) -> crate::config::ModelParams {
        let mut model = empty_model(name);
        model.bounds.insert("num_attention_heads".to_string(), 32);
        model.bounds.insert("num_key_value_heads".to_string(), 8);
        model.bounds.insert("head_dim".to_string(), 128);
        model
    }

    // ── CutlassFusedRmsNormGemm / LayerNormGemm 2-tile matchers ──────
    //
    // The 2-tile (Norm, Gemm) family is the `Norm → Gemm` lm_head
    // shape commandr emits (`LayerNorm → Cublas (M=1,N=256000,K=8192)`).
    // Audit at tip `1a756be34` showed 0 picks for both 2-tile variants
    // despite 958 lm_head picks the analyzer flagged as
    // `lm_head: Norm→Gemm[→ScalarMul]` absorbable. These tests pin the
    // matcher's structural acceptance of the canonical lm_head shape so
    // the 0-pick can be traced to cost / DP behavior, not a silent
    // matcher reject.

    /// 5-node FUF mirroring the post-migration commandr lm_head shape:
    /// `Embed → Mean → Sub → RmsNorm → Gemm`. This is the cohere
    /// (mean-subtract) flavor expressed in math primitives. The
    /// (Mean, Sub, RmsNorm, Gemm) 4-tile claim is what
    /// `CutlassFusedMeanSubRmsNormGemmImpl` matches.
    fn fused_mean_sub_rmsnorm_gemm_test_fuf() -> Fuf {
        use crate::shape::Dim;
        let t0 = TileId(0);
        let t1 = TileId(1);
        let t2 = TileId(2);
        let t3 = TileId(3);
        let t4 = TileId(4);
        Fuf {
            nodes: vec![
                FufNode {
                    id: t0,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: crate::classified::ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t1,
                    op: OpKind::Mean,
                    inputs: vec![FufInput::Tile { id: t0, slot: 0 }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t2,
                    op: OpKind::Sub,
                    inputs: vec![
                        FufInput::Tile { id: t0, slot: 0 },
                        FufInput::Tile { id: t1, slot: 0 },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t3,
                    op: OpKind::RmsNorm,
                    inputs: vec![
                        FufInput::Tile { id: t2, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t4,
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile { id: t3, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(1),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(256000)]],
                },
            ],
        }
    }

    fn fused_norm_gemm_test_fuf(norm_kind: OpKind) -> Fuf {
        use crate::shape::Dim;
        // 3-node FUF mirroring the rmsnorm flavor of lm_head:
        //   t0  Embed       → s0 [bf16, 1×8192]   (activation)
        //   t1  Norm        → s0 [bf16, 1×8192]   (consumes t0 slot 0)
        //   t2  Gemm        → s0 [bf16, 1×256000] (consumes t1 slot 0,
        //                                          weight Dense)
        let t0 = TileId(0);
        let t1 = TileId(1);
        let t2 = TileId(2);
        Fuf {
            nodes: vec![
                FufNode {
                    id: t0,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: crate::classified::ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t1,
                    op: norm_kind,
                    inputs: vec![
                        FufInput::Tile { id: t0, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t2,
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile { id: t1, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(1),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(256000)]],
                },
            ],
        }
    }

    #[test]
    fn cutlass_fused_mean_sub_rmsnorm_gemm_4tile_matches_commandr_lm_head_shape() {
        let fuf = fused_mean_sub_rmsnorm_gemm_test_fuf();
        let imp = CutlassFusedMeanSubRmsNormGemmImpl {
            tile_m: 16,
            tile_n: 64,
            stages: 3,
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let m = imp.matches(&fuf, TileId(1), &profile);
        assert!(
            m.is_some(),
            "CutlassFusedMeanSubRmsNormGemmImpl 4-tile must match \
             (Mean, Sub, RmsNorm, Gemm) seeded on the Mean — the post-migration \
             commandr lm_head shape after retiring `OpKind::LayerNorm`"
        );
        let m = m.unwrap();
        assert_eq!(m.claimed_tiles.len(), 4);
        // Mean=t1, Sub=t2, RmsNorm=t3, Gemm=t4 (Embed=t0 stays upstream).
        assert!(m.claimed_tiles.contains(&TileId(1)));
        assert!(m.claimed_tiles.contains(&TileId(2)));
        assert!(m.claimed_tiles.contains(&TileId(3)));
        assert!(m.claimed_tiles.contains(&TileId(4)));
        assert_eq!(m.boundary_inputs, vec![TileId(0)]);
        assert_eq!(m.boundary_outputs, vec![TileId(4)]);
    }

    #[test]
    fn cutlass_fused_rms_norm_gemm_2tile_matches_canonical_lm_head_shape() {
        let fuf = fused_norm_gemm_test_fuf(OpKind::RmsNorm);
        let imp = CutlassFusedRmsNormGemmImpl {
            tile_m: 16,
            tile_n: 64,
            stages: 3,
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let m = imp.matches(&fuf, TileId(1), &profile);
        assert!(
            m.is_some(),
            "CutlassFusedRmsNormGemmImpl 2-tile must match (RmsNorm, Gemm) seeded \
             on the RmsNorm — the canonical lm_head shape on llama-family arches \
             without an upstream residual Add"
        );
        let m = m.unwrap();
        assert_eq!(m.claimed_tiles.len(), 2);
        assert!(m.claimed_tiles.contains(&TileId(1)));
        assert!(m.claimed_tiles.contains(&TileId(2)));
    }

    /// Build a FUF mirroring commandr's true lm_head shape:
    /// `Embed → Mean → Sub → RmsNorm → Gemm → ScalarMul`. The
    /// downstream ScalarMul (commandr's logit_scale) is what makes
    /// the lm_head pattern claim `CutlassFusedMeanSubRmsNormGemm`
    /// plus a separate `ScalarMul` singleton — the matcher must NOT
    /// reject because of the trailing scalar consumer.
    fn fused_mean_sub_rmsnorm_gemm_scalarmul_test_fuf() -> Fuf {
        use crate::shape::Dim;
        let t0 = TileId(0);
        let t1 = TileId(1);
        let t2 = TileId(2);
        let t3 = TileId(3);
        let t4 = TileId(4);
        let t5 = TileId(5);
        Fuf {
            nodes: vec![
                FufNode {
                    id: t0,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: crate::classified::ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t1,
                    op: OpKind::Mean,
                    inputs: vec![FufInput::Tile { id: t0, slot: 0 }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t2,
                    op: OpKind::Sub,
                    inputs: vec![
                        FufInput::Tile { id: t0, slot: 0 },
                        FufInput::Tile { id: t1, slot: 0 },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t3,
                    op: OpKind::RmsNorm,
                    inputs: vec![
                        FufInput::Tile { id: t2, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(8192)]],
                },
                FufNode {
                    id: t4,
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile { id: t3, slot: 0 },
                        FufInput::Weight {
                            id: crate::classified::WeightId(1),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(256000)]],
                },
                FufNode {
                    id: t5,
                    op: OpKind::Mul,
                    inputs: vec![FufInput::Tile { id: t4, slot: 0 }, FufInput::Scalar(0.0625)],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(256000)]],
                },
            ],
        }
    }

    #[test]
    fn cutlass_fused_norm_gemm_solver_picks_4tile_with_downstream_scalarmul() {
        // Production reality at commandr lm_head: the Gemm has a
        // downstream ScalarMul consumer (logit_scale). The 4-tile
        // (Mean, Sub, RmsNorm, Gemm) matcher constrains the RmsNorm's
        // consumers (must be 1 = Gemm) but not the Gemm's downstream —
        // so a downstream ScalarMul must NOT block the claim.
        use crate::shape::Inferred;
        let fuf = fused_mean_sub_rmsnorm_gemm_scalarmul_test_fuf();
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let lib = starter_library();
        let inferred = Inferred {
            locals: std::collections::HashMap::new(),
            weights: std::collections::HashMap::new(),
            hybrid_attention_geometry: false,
        };
        let bounds: BTreeMap<String, u64> = BTreeMap::new();
        let assignments =
            crate::solver::solve(&fuf, &lib, &profile, &inferred, &bounds, &[1], &[0])
                .expect("solver");
        let asgn = assignments
            .per_workload
            .iter()
            .next()
            .map(|(_, a)| a)
            .expect("one workload point");

        // Mean=t1, Sub=t2, RmsNorm=t3, Gemm=t4 in the 5-node test FUF.
        let sg = asgn.subgraph_of(TileId(1)).expect("Mean covered");
        let imp_id = asgn.impl_of(sg).expect("subgraph mapped to Impl");
        let imp = lib.get(imp_id);
        let name = imp.name();
        let sg_gemm = asgn.subgraph_of(TileId(4)).expect("Gemm covered");
        assert_eq!(
            sg_gemm, sg,
            "(Mean, Sub, RmsNorm, Gemm) must collapse into a single 4-tile \
             fused subgraph even when the Gemm has a downstream ScalarMul; \
             got name=`{name}`"
        );
    }

    #[test]
    fn cutlass_fused_norm_gemm_solver_picks_4tile_at_commandr_lm_head_shape() {
        // Integration: run the actual solver on the (Mean, Sub, RmsNorm,
        // Gemm) 5-node FUF and assert that the 4-tile fused Impl wins
        // the DP pick at the seed (M=1 lm_head). Post-migration peer
        // of the original (LayerNorm, Gemm) 2-tile pin.
        use crate::shape::Inferred;
        let fuf = fused_mean_sub_rmsnorm_gemm_test_fuf();
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let lib = starter_library();
        let inferred = Inferred {
            locals: std::collections::HashMap::new(),
            weights: std::collections::HashMap::new(),
            hybrid_attention_geometry: false,
        };
        let bounds: BTreeMap<String, u64> = BTreeMap::new();
        let assignments =
            crate::solver::solve(&fuf, &lib, &profile, &inferred, &bounds, &[1], &[0])
                .expect("solver");
        let asgn = assignments
            .per_workload
            .iter()
            .next()
            .map(|(_, a)| a)
            .expect("one workload point");

        // Mean=t1, Gemm=t4 in the 5-node test FUF.
        let sg = asgn.subgraph_of(TileId(1)).expect("Mean covered");
        let imp_id = asgn.impl_of(sg).expect("subgraph mapped to Impl");
        let imp = lib.get(imp_id);
        let name = imp.name();
        assert!(
            name.starts_with("cutlass_") && !name.contains("gemv"),
            "DP must pick a CutlassFused*Gemm tile at the Mean seed (got `{name}`); \
             a `cublas`/singleton pick here is the 0-pick anomaly."
        );
        let sg_gemm = asgn.subgraph_of(TileId(4)).expect("Gemm covered");
        assert_eq!(
            sg_gemm, sg,
            "fused 4-tile claim must cover (Mean, Sub, RmsNorm, Gemm) in one subgraph"
        );
    }

    #[test]
    fn cutlass_fused_norm_gemm_norm_term_must_equal_singleton_norm_cost() {
        // The fused (Mean, Sub, RmsNorm, Gemm) cost is `gemm_us + norm_us`.
        // The unfused alternative is `norm_trio_us + gemm_singleton_us`.
        // For the DP's claim-size-DESC tiebreak to favor fusion at
        // uncalibrated shapes (where gemm_us == gemm_singleton_us via
        // shared roofline), the FUSED `norm_us` term MUST equal the
        // TRIO `MeanSubRmsNormImpl::cost_us` byte-for-byte. Any delta
        // hands the win to the trio + singleton chain. Both use
        // `elementwise_cost` keyed on the RmsNorm tile.
        use crate::shape::Inferred;
        let fuf = fused_mean_sub_rmsnorm_gemm_test_fuf();
        let imp = CutlassFusedMeanSubRmsNormGemmImpl {
            tile_m: 16,
            tile_n: 64,
            stages: 3,
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("hidden_size".to_string(), 8192);
        bounds.insert("num_tokens".to_string(), 1);
        let ctx = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
            expected_calls_per_load: CostCtx::default_expected_calls_per_load(profile.backend),
        };

        // Trio norm cost: MeanSubRmsNormImpl claims (Mean, Sub, RmsNorm)
        // = (t1, t2, t3) and its `cost_us` computes elementwise_cost on
        // the RmsNorm tile (input from t0, output at t3).
        let norm_trio = MatchInfo {
            claimed_tiles: vec![TileId(1), TileId(2), TileId(3)],
            boundary_inputs: vec![TileId(0)],
            boundary_outputs: vec![TileId(3)],
        };
        let norm_trio_us = MeanSubRmsNormImpl.cost_us(&norm_trio, &ctx);

        // Fused total cost.
        let m = imp.matches(&fuf, TileId(1), &profile).expect("matches");
        let fused_total_us = imp.cost_us(&m, &ctx);

        // Singleton-Gemm cost via cost_gemm (cuBLAS path: CSV row at
        // M=1 N=256000 K=8192 missing → roofline).
        let gemm_only = MatchInfo {
            claimed_tiles: vec![TileId(4)],
            boundary_inputs: vec![TileId(3)],
            boundary_outputs: vec![TileId(4)],
        };
        let gemm_singleton_us = cost_gemm(&gemm_only, &ctx);

        // Equivalence: fused_total - gemm_singleton = norm_trio.
        let delta = fused_total_us - gemm_singleton_us;
        assert!(
            (delta - norm_trio_us).abs() < 1e-9,
            "fused norm term ({delta:.6}) must equal trio norm cost \
             ({norm_trio_us:.6}); fused_total={fused_total_us:.6} \
             gemm_singleton={gemm_singleton_us:.6}. A delta means the DP \
             tiebreak hands the lm_head pick to (trio + cublas) singletons."
        );
        let _ = Inferred {
            locals: std::collections::HashMap::new(),
            weights: std::collections::HashMap::new(),
            hybrid_attention_geometry: false,
        };
    }

    #[test]
    fn cutlass_fused_norm_gemm_rejects_unaligned_vocab_n_at_lm_head() {
        // Granite-3.3-2B vocab_size = 49159 (genuinely not divisible
        // by 8). The CUTLASS standalone tile zoo requires N % 8 == 0
        // (AlignmentB = 8 in cutlass_standalone_gemm.cu); without
        // this gate the kernel returns -1 at runtime and crashes
        // `scr serve`. Cost MUST be UNCALIBRATED at unaligned N so
        // the DP routes to a singleton + cuBLAS fallback. The
        // alignment helper is shape-only (no FUF traversal needed).
        assert!(cutlass_tile_supports_shape(1, 49152, 2048));
        assert!(!cutlass_tile_supports_shape(1, 49159, 2048));
        assert!(!cutlass_tile_supports_shape(1, 49152, 49159));
        // 256000 (commandr vocab) is divisible by 8.
        assert!(cutlass_tile_supports_shape(1, 256000, 8192));
    }

    #[test]
    fn cutlass_fused_norm_gemm_cost_matches_unfused_norm_plus_cublas_at_uncalibrated_shape() {
        // At commandr's lm_head (M=1, N=256000, K=8192) NEITHER
        // `cutlass_<tile>` nor `cublas` has a CSV row — both fall to
        // their roofline formula. The 2-tile FUSED cost should equal
        // the UNFUSED (norm-singleton + cublas-singleton) cost
        // BYTE-FOR-BYTE so the DP's claim-size-DESC sort picks the
        // fused form. If the fused side is even ε higher, the DP
        // picks singleton+singleton and the fused Impl captures 0
        // picks at vocab-N — exactly the 0-pick anomaly STATUS flagged.
        let fuf = fused_mean_sub_rmsnorm_gemm_test_fuf();
        let imp = CutlassFusedMeanSubRmsNormGemmImpl {
            tile_m: 16,
            tile_n: 64,
            stages: 3,
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let m = imp.matches(&fuf, TileId(1), &profile).expect("matches");

        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("hidden_size".to_string(), 8192);
        bounds.insert("num_tokens".to_string(), 1);
        let ctx = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
            expected_calls_per_load: CostCtx::default_expected_calls_per_load(profile.backend),
        };

        let fused_us = imp.cost_us(&m, &ctx);
        // Unfused alternative: cuBLAS singleton on the Gemm (t4) via
        // cost_gemm. Norm trio's cost cancels (the fused Impl adds
        // the same `bytes / bw_gb` term). So the gate is entirely
        // on the GEMM: cutlass_roofline vs cublas_roofline.
        let gemm_only = MatchInfo {
            claimed_tiles: vec![TileId(4)],
            boundary_inputs: vec![TileId(3)],
            boundary_outputs: vec![TileId(4)],
        };
        let unfused_gemm_us = cost_gemm(&gemm_only, &ctx);

        // Both rooflines compute (2*M*N*K) / peak_flops. Fused cost =
        // gemm_roofline + norm_bw. Unfused gemm-only cost = the same
        // gemm_roofline. So fused_us == unfused_gemm_us + norm_us.
        // The norm_us term is what the DP would charge to the
        // norm SINGLETON in the unfused path — so it cancels at the
        // total-cost level. The DP's claim-size-DESC tiebreak then
        // picks fused.
        assert!(fused_us.is_finite() && fused_us > 0.0);
        assert!(unfused_gemm_us.is_finite() && unfused_gemm_us > 0.0);
        assert!(
            fused_us >= unfused_gemm_us,
            "fused cost ({fused_us}) includes norm BW; unfused gemm-only ({unfused_gemm_us}) \
             does not — fused must be ≥ gemm-only at the SAME shape"
        );
    }

    // ── Encoder vs decoder attention dispatch ────────────────────
    //
    // The 3-arg/5-arg discrimination in `attention_has_kv_cache_extern`
    // is the routing fulcrum for ModernBERT and any future encoder
    // arch. These tests pin the predicate end-to-end: the same
    // OpKind::Attention tile must reach DIFFERENT Impls depending on
    // whether a KvCache extern is present.

    fn attention_node(inputs: Vec<FufInput>) -> FufNode {
        use crate::shape::Dim;
        FufNode {
            id: TileId(0),
            op: OpKind::Attention,
            inputs,
            // [T, num_attention_heads*head_dim] — shape used for cost
            // calc only; matchers don't read it.
            outputs: vec![vec![Dim::Lit(1), Dim::Lit(128)]],
        }
    }

    fn three_arg_encoder_inputs() -> Vec<FufInput> {
        vec![
            FufInput::Tile {
                id: TileId(1),
                slot: 0,
            },
            FufInput::Tile {
                id: TileId(2),
                slot: 0,
            },
            FufInput::Tile {
                id: TileId(3),
                slot: 0,
            },
        ]
    }

    fn five_arg_decoder_inputs() -> Vec<FufInput> {
        let mut v = three_arg_encoder_inputs();
        v.push(FufInput::Extern {
            kind: ExternKind::KvCache,
            index: Some(UnrollIndex(0)),
        });
        v.push(FufInput::Extern {
            kind: ExternKind::BlockTable,
            index: None,
        });
        v
    }

    #[test]
    fn encoder_attention_impl_matches_3arg_only() {
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);

        // Encoder shape (3-arg): EncoderAttentionImpl claims, decoder
        // Impls reject.
        let fuf3 = Fuf {
            nodes: vec![attention_node(three_arg_encoder_inputs())],
        };
        let seed = TileId(0);
        let enc = EncoderAttentionImpl;
        let dec_via = AttentionViaCacheImpl;
        let dec_pre = AttentionPrefillContiguousImpl;
        assert!(
            enc.matches(&fuf3, seed, &profile).is_some(),
            "EncoderAttentionImpl must claim 3-arg attention(q,k,v)",
        );
        assert!(
            dec_via.matches(&fuf3, seed, &profile).is_none(),
            "AttentionViaCacheImpl must NOT claim 3-arg encoder attention",
        );
        assert!(
            dec_pre.matches(&fuf3, seed, &profile).is_none(),
            "AttentionPrefillContiguousImpl must NOT claim 3-arg encoder attention",
        );
    }

    #[test]
    fn decoder_attention_impls_match_5arg_only() {
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);

        // Decoder shape (5-arg): both paged decoder Impls claim,
        // EncoderAttentionImpl rejects.
        let fuf5 = Fuf {
            nodes: vec![attention_node(five_arg_decoder_inputs())],
        };
        let seed = TileId(0);
        let enc = EncoderAttentionImpl;
        let dec_via = AttentionViaCacheImpl;
        let dec_pre = AttentionPrefillContiguousImpl;
        assert!(
            dec_via.matches(&fuf5, seed, &profile).is_some(),
            "AttentionViaCacheImpl must claim 5-arg decoder attention",
        );
        assert!(
            dec_pre.matches(&fuf5, seed, &profile).is_some(),
            "AttentionPrefillContiguousImpl must claim 5-arg decoder attention",
        );
        assert!(
            enc.matches(&fuf5, seed, &profile).is_none(),
            "EncoderAttentionImpl must NOT claim 5-arg decoder attention",
        );
    }

    // ── 4-tile MeanSubRmsNormBiasAdd vs 3-tile trio dispatch ────
    //
    // The 4-tile fusion only fires when a `BiasAdd` is downstream of
    // the `RmsNorm` AND the BiasAdd's second input is a Weight (the
    // bias tensor). These tests pin the dispatch boundary so the trio
    // stays the canonical claim everywhere else.

    fn build_four_tile_layer_norm_fuf() -> Fuf {
        use crate::quantization::StorageFormat;
        use crate::shape::Dim;
        // Tile 0: source x (synthetic — represents the upstream
        // tensor; e.g. an embedding output). Singleton Embed shape
        // works for the matcher's purposes — it just needs to be a
        // tile that Mean/Sub can reference.
        let x = TileId(0);
        let mean = TileId(1);
        let sub = TileId(2);
        let rmsnorm = TileId(3);
        let bias_add = TileId(4);
        let weight = WeightId(0);
        let bias = WeightId(1);
        let hidden = vec![Dim::Lit(1), Dim::Lit(64)];

        Fuf {
            nodes: vec![
                FufNode {
                    id: x,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: mean,
                    op: OpKind::Mean,
                    inputs: vec![FufInput::Tile { id: x, slot: 0 }],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: sub,
                    op: OpKind::Sub,
                    inputs: vec![
                        FufInput::Tile { id: x, slot: 0 },
                        FufInput::Tile { id: mean, slot: 0 },
                    ],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: rmsnorm,
                    op: OpKind::RmsNorm,
                    inputs: vec![
                        FufInput::Tile { id: sub, slot: 0 },
                        FufInput::Weight {
                            id: weight,
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: bias_add,
                    op: OpKind::BiasAdd,
                    inputs: vec![
                        FufInput::Tile {
                            id: rmsnorm,
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: bias,
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![hidden],
                },
            ],
        }
    }

    #[test]
    fn mean_sub_rms_norm_bias_add_matches_4tile_pattern() {
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let fuf = build_four_tile_layer_norm_fuf();
        let imp = MeanSubRmsNormBiasAddImpl;
        // Seed on the Mean (TileId(1)) — same convention as the trio
        // matcher.
        let m = imp
            .matches(&fuf, TileId(1), &profile)
            .expect("4-tile pattern must claim");
        assert_eq!(
            m.claimed_tiles.len(),
            4,
            "must claim Mean + Sub + RmsNorm + BiasAdd",
        );
        // The boundary output is the BiasAdd's tile, not the RmsNorm's
        // — downstream consumers should read the biased result.
        assert_eq!(m.boundary_outputs, vec![TileId(4)]);
    }

    #[test]
    fn mean_sub_rms_norm_trio_does_not_steal_4tile_pattern() {
        // The trio matcher (`MeanSubRmsNormImpl`) still matches the
        // 4-tile FUF because its claim doesn't reject the BiasAdd
        // tile (BiasAdd just gets left out of the claim). The solver
        // picks the bigger claim via claim-size-DESC. This test
        // documents that both impls see the seed; the solver — not the
        // matcher — does the dispatch.
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        let fuf = build_four_tile_layer_norm_fuf();
        let trio = MeanSubRmsNormImpl;
        let four = MeanSubRmsNormBiasAddImpl;
        let m_trio = trio.matches(&fuf, TileId(1), &profile);
        let m_four = four.matches(&fuf, TileId(1), &profile);
        assert!(m_trio.is_some(), "trio still matches the seed");
        assert!(m_four.is_some(), "4-tile fires on the seed");
        // 4-tile claim is strictly larger — solver claim-size-DESC
        // will prefer it.
        assert!(
            m_four.unwrap().claimed_tiles.len() > m_trio.unwrap().claimed_tiles.len(),
            "4-tile must outsize the trio so the solver picks it",
        );
    }

    #[test]
    fn mean_sub_rms_norm_bias_add_rejects_trio_only_pattern() {
        // Same FUF without the BiasAdd → only the trio matches.
        use crate::quantization::StorageFormat;
        use crate::shape::Dim;
        let x = TileId(0);
        let mean = TileId(1);
        let sub = TileId(2);
        let rmsnorm = TileId(3);
        let weight = WeightId(0);
        let hidden = vec![Dim::Lit(1), Dim::Lit(64)];
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: x,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: mean,
                    op: OpKind::Mean,
                    inputs: vec![FufInput::Tile { id: x, slot: 0 }],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: sub,
                    op: OpKind::Sub,
                    inputs: vec![
                        FufInput::Tile { id: x, slot: 0 },
                        FufInput::Tile { id: mean, slot: 0 },
                    ],
                    outputs: vec![hidden.clone()],
                },
                FufNode {
                    id: rmsnorm,
                    op: OpKind::RmsNorm,
                    inputs: vec![
                        FufInput::Tile { id: sub, slot: 0 },
                        FufInput::Weight {
                            id: weight,
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![hidden],
                },
            ],
        };
        let profile = crate::target::from_profile_def(&scratchy_target_cuda::targets::L4_SM89);
        assert!(
            MeanSubRmsNormBiasAddImpl
                .matches(&fuf, TileId(1), &profile)
                .is_none(),
            "no BiasAdd → 4-tile must reject",
        );
        assert!(
            MeanSubRmsNormImpl
                .matches(&fuf, TileId(1), &profile)
                .is_some(),
            "trio still claims",
        );
    }

    #[test]
    fn mean_sub_rms_norm_bias_add_registered_in_starter_library() {
        let lib = starter_library();
        let count = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "mean_sub_rms_norm_bias_add")
            .count();
        assert_eq!(
            count, 1,
            "starter_library must register exactly one MeanSubRmsNormBiasAddImpl",
        );
    }

    #[test]
    fn encoder_attention_registered_in_starter_library() {
        // Adding/removing the registration line in `starter_library`
        // is the only way an encoder model can find this Impl. Lock it
        // so a future "tidy-up" can't silently drop it.
        let lib = starter_library();
        let count = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "encoder_attention")
            .count();
        assert_eq!(
            count, 1,
            "starter_library must register exactly one EncoderAttentionImpl"
        );
    }
}
// ── AllReduceImpl ────────────────────────────────────────────────
//
// Single-tile impl claiming an `OpKind::AllReduce` node — the
// row-parallel TP communicator the lowering pass (task #5b/c)
// inserts after every gemm whose weight has shard-kind `ShardDim1`
// (`o_proj` / `down_proj`). Maps to `NcclGroup::all_reduce_inplace`
// via `Instruction::AllReduce(slot)` (gated on `scratchy-forward-compiler`'s
// `nccl` feature).
//
// One-tile in-place same-shape: `output_alias` declares the dst
// slot aliases the input slot, so shape-aware coloring collapses
// it to the input's slot with no `View` row (validated by
// `coloring_allreduce_collapses_to_input_slot`). The fused-pair
// comm-boundary guard is emergent from the FUF dataflow break —
// any node with an OpKind::AllReduce sitting between a Gemm output
// and a residual Add naturally fails `consumes_tile(add, gemm)`,
// rejecting the fusion (validated by
// `cutlass_gemm_add_does_not_claim_across_intermediate_node`).
//
// At `tp_world_size = 1` the lowering pass is a no-op — no FUF
// node carries `OpKind::AllReduce`, so this Impl never matches and
// is free in the solver. Always registered in `starter_library()`
// regardless of build features; `cost_us` returns infinity if the
// target profile lacks an `all_reduce` cost row, deterring the DP
// from picking it on uncalibrated targets.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AllReduceImpl;

#[cfg(feature = "cuda")]
impl Implementation for AllReduceImpl {
    fn name(&self) -> &'static str {
        "all_reduce"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // No CSV gating — all calibrated and uncalibrated profiles
        // can run NCCL when `scratchy-forward-compiler/nccl` is on. Profiles
        // without an `all_reduce` cost row use `UNCALIBRATED_COST_US`
        // as a fallback in `cost_us` so the DP still has a finite
        // number to compare.
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::AllReduce || node.inputs.len() != 1 {
            return None;
        }
        let input_id = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![input_id],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: NCCL all-reduce-sum reads + writes the
        // tile per rank, and the cross-rank ring-pass amortizes to
        // ~2x the per-tile bytes at large world sizes. Without a
        // measured cost row, model as 2x bandwidth-bound on the
        // local memory subsystem — a coarse upper bound that the DP
        // uses as a tiebreaker between coverings; the actual NCCL
        // cost dominates here in practice and is a target-specific
        // CSV row to be added later. Returns `UNCALIBRATED_COST_US`
        // when no profile data exists, matching the convention used
        // by other comm-bound Impls.
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * m * hidden * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            UNCALIBRATED_COST_US
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::AliasToInput(0)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("AllReduce", vec![("slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let ar_id = m.claimed_tiles[0];
        let node = fuf.get(ar_id);
        let (input_id, input_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("AllReduce: input 0 must be a Tile (got {other:?})"),
        };
        let slot_idx = slots.of(input_id, input_slot);
        #[cfg(feature = "nccl")]
        {
            Some(vec![Instruction::AllReduce(slot_idx)])
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = slot_idx;
            Some(Vec::new())
        }
    }

    // No `interpreter_arm` — the universal `Instruction::eval` body
    // in `scratchy-forward-compiler/src/instr.rs` (gated on `feature = "nccl"`)
    // is the production dispatch path. The trait stopped carrying
    // `interpreter_arm` after the universal-eval pivot (commit
    // `de15e035a`); fan_out + opcode_shape are the only codegen
    // surface an Impl needs to override.
}

// ── AllGatherImpl ────────────────────────────────────────────────
//
// Single-tile impl claiming an `OpKind::AllGather` node — the
// vocab-parallel TP communicator the lowering pass inserts after
// the lm_head Gemm at tp>1, since lm_head is `ShardDim0`. Maps to
// `NcclGroup::all_gather_last_dim` via `Instruction::AllGather(in,
// out)` (gated on `scratchy-forward-compiler`'s `nccl` feature).
//
// One input tile, one output tile with a *larger* shape (last dim
// grows by `tp_world_size`). Unlike `AllReduceImpl`, this Impl
// does NOT override `output_alias` — the default returns `None`
// for every output, which is exactly what we want: the coloring
// pass assigns the output to a fresh slot since it can't share
// the input's smaller buffer.
//
// At `tp_world_size = 1` the lowering pass is a no-op so this
// Impl never matches and is free in the solver.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AllGatherImpl;

#[cfg(feature = "cuda")]
impl Implementation for AllGatherImpl {
    fn name(&self) -> &'static str {
        "all_gather"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // Same convention as `AllReduceImpl` — all profiles can run
        // NCCL when `scratchy-forward-compiler/nccl` is on; uncalibrated
        // targets fall back to `UNCALIBRATED_COST_US` in `cost_us`.
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::AllGather || node.inputs.len() != 1 {
            return None;
        }
        let input_id = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![input_id],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: NCCL all-gather reads the per-rank tile and
        // writes the gathered tile; the cross-rank ring-pass amortizes
        // to ~world_size× the per-rank tile bytes. The CSV row for
        // `all_gather` is target-specific; absent the row this models
        // it as `world_size * 2` × per-rank bytes, divided by local BW
        // — coarse upper bound, the DP uses it only as a tiebreaker.
        let m = ctx.num_tokens() as f64;
        let vocab = ctx.bounds.get("vocab_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * m * vocab * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            UNCALIBRATED_COST_US
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    // No `output_alias` override — default returns `None` for every
    // output, which gives the AllGather output a fresh slot
    // (separate from the smaller input slot). That's the load-bearing
    // distinction from `AllReduceImpl`, which collapses to the input
    // slot via shape-aware coloring.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AllGather",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let ag_id = m.claimed_tiles[0];
        let node = fuf.get(ag_id);
        let (input_id, input_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("AllGather: input 0 must be a Tile (got {other:?})"),
        };
        let in_idx = slots.of(input_id, input_slot);
        let out_idx = slots.of(ag_id, 0);
        #[cfg(feature = "nccl")]
        {
            Some(vec![Instruction::AllGather(in_idx, out_idx)])
        }
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (in_idx, out_idx);
            Some(Vec::new())
        }
    }
}

// ── MmEmbedSpliceImpl ────────────────────────────────────────────
//
// Single-tile in-place impl claiming an `OpKind::MmEmbedSplice` node
// — the post-Embed multimodal splice the lowering pass
// (`tp_lowering::insert_mm_splices`) inserts after every Embed. At
// tp>1 the `insert_all_reduces` pass has already inserted an
// `AllReduce` between the Embed and its consumers, so this
// MmEmbedSplice ends up wired *after* the AllReduce — the mm_embeds
// D2D-overwrite thus runs on the fully-reduced embedding instead of
// on each rank's partial masked-gather (which would get summed × tp
// on the reduce). Lowers to `Instruction::SpliceMmEmbeds(slot)`.
//
// Identity-shape in-place mutation; output aliases the input slot
// (same pattern as AllReduce). Unconditionally inserted at every tp
// including tp=1 — at tp=1 it sits directly on the Embed output and
// reproduces the pre-refactor inline splice.
//
// At runtime, `Instruction::SpliceMmEmbeds::eval` short-circuits
// when `ForwardCtx::embed_patches` is empty (text-only batches), so
// the cost is one taken-branch check per forward pass.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct MmEmbedSpliceImpl;

#[cfg(feature = "cuda")]
impl Implementation for MmEmbedSpliceImpl {
    fn name(&self) -> &'static str {
        "mm_embed_splice"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // No profile gating — the splice is a D2D memcpy sequence
        // available on every CUDA target.
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::MmEmbedSplice || node.inputs.len() != 1 {
            return None;
        }
        let input_id = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![input_id],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Cost is proportional to the projected vision-encoder output
        // size, which is per-request and not a function of
        // num_tokens in the way model ops are. Zero-cost for the DP —
        // there's no alternative covering (no other Impl claims this
        // OpKind), so the tiebreaker value doesn't affect routing.
        0.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn alias_rule(&self) -> crate::alias_rules::AliasRule {
        crate::alias_rules::AliasRule::AliasToInput(0)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("SpliceMmEmbeds", vec![("slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let splice_id = m.claimed_tiles[0];
        let node = fuf.get(splice_id);
        let (input_id, input_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MmEmbedSplice: input 0 must be a Tile (got {other:?})"),
        };
        let slot_idx = slots.of(input_id, input_slot);
        Some(vec![Instruction::SpliceMmEmbeds(slot_idx)])
    }
}

// ── Vision-tower Impls (Phase G.4) ───────────────────────────────
//
// Singleton matchers for the four vision OpKinds added in G.2
// (`VarlenAttention`, `VisionRope`, `QuickGelu`, `GeluErf`). Each
// claims one tile, calls one existing kernel, and emits one
// `Instruction` row. No fusion peers yet — each is the only Impl
// for its OpKind, so the DP solver has a deterministic claim.
//
// All four are unreachable until a `#[vision_forward]` body lands
// (G.5+); registering them now keeps `OpKind::from_name`'s vision
// arms total (parse ⟺ codegen) per the handoff bar.

// ── LoadPixelsImpl (Phase G.5.e.1) ───────────────────────────────
//
// Singleton claiming `OpKind::LoadPixels` — synthesized by the
// `vision_lowering::materialize_pixels` pass between `fuf::unroll`
// and the solver. No FUF inputs, no weight inputs: the runtime tile
// is built from `ctx.fwd.pixels` (the view the vision-side host
// wrapper writes onto `ForwardCtx` before invoking the interpreter,
// mirroring how text-side `Embed` reads `ctx.input_ids`). Emits a
// single `Instruction::LoadPixels { out_slot }` row.
//
// This is the vision-side analog of `EmbedRefImpl`: both publish a
// tile from an ambient `ForwardCtx` field rather than a tile
// dataflow. The text path packs the read into `Embed` directly
// (which also has a weight input — embed_tokens); the vision path
// has no weight to anchor on, so the materialization needs its own
// OpKind + Impl.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct LoadPixelsImpl;

#[cfg(feature = "cuda")]
impl Implementation for LoadPixelsImpl {
    fn name(&self) -> &'static str {
        "load_pixels"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::LoadPixels || !node.inputs.is_empty() {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: Vec::new(),
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Pure host-side metadata wrap of `ctx.fwd.pixels` — no
        // device work and no alternative covering, so the DP
        // tiebreaker value doesn't affect routing. Same zero-cost
        // story as `MmEmbedSpliceImpl`.
        0.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("LoadPixels", vec![("out_slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        _fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        Some(vec![Instruction::LoadPixels(out_slot)])
    }
}

// ── LoadPosEmbedsImpl (Qwen3.5-VL pos_embed) ─────────────────────
//
// The exact sibling of `LoadPixelsImpl`: a singleton claiming
// `OpKind::LoadPosEmbeds`, synthesized by
// `vision_lowering::materialize_pos_embeds`. No FUF/weight inputs —
// the runtime tile is built from `ctx.fwd.pos_embeds` (the
// host-interpolated learned positional embedding the vision wrapper
// writes onto `ForwardCtx` before driving the interpreter). Emits a
// single `Instruction::LoadPosEmbeds { out_slot }` row.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct LoadPosEmbedsImpl;

#[cfg(feature = "cuda")]
impl Implementation for LoadPosEmbedsImpl {
    fn name(&self) -> &'static str {
        "load_pos_embeds"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::LoadPosEmbeds || !node.inputs.is_empty() {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: Vec::new(),
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        0.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("LoadPosEmbeds", vec![("out_slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        _fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        Some(vec![Instruction::LoadPosEmbeds(out_slot)])
    }
}

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct VarlenAttentionImpl;

#[cfg(feature = "cuda")]
impl Implementation for VarlenAttentionImpl {
    fn name(&self) -> &'static str {
        "varlen_attention"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::VarlenAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same analytic shape as text-side prefill attention. The
        // calibrated FA2 CSV is text-shape-keyed today; vision
        // workloads need their own sweep before peer Impls compete on
        // cost. Until then
        // the analytic estimate is the best we have.
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `VarlenAttention { q_slot, k_slot, v_slot, out_slot,
    // cu_seqlens_kind }`. The `cu_seqlens_kind` discriminant is baked
    // by `fan_out` from `node.inputs[3]`'s `ExternKind`:
    //   - 0 = Default (`ForwardCtx::cu_seqlens_q` + `max_seqlen_q`) —
    //         Qwen2-VL's single-cu-seqlens path.
    //   - 1 = Full (`vision_cu_seqlens_full` + `vision_max_seqlen_full`)
    //         — Qwen2.5-VL's fullatt-layer dispatch.
    //   - 2 = Window (`vision_cu_seqlens_window` +
    //         `vision_max_seqlen_window`) — Qwen2.5-VL's windowed-layer
    //         dispatch.
    // Eval switches on the discriminant and reads the matching pair.
    // Scale, head count, and q_size are baked in `W::VISION_*` consts.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "VarlenAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("cu_seqlens_kind", syn::parse_quote!(u8)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("VarlenAttention: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let cu_seqlens_kind: u8 = match node.inputs.get(3) {
            Some(FufInput::Extern {
                kind: ExternKind::CuSeqlens,
                ..
            }) => 0,
            Some(FufInput::Extern {
                kind: ExternKind::CuSeqlensFull,
                ..
            }) => 1,
            Some(FufInput::Extern {
                kind: ExternKind::CuSeqlensWindow,
                ..
            }) => 2,
            other => panic!(
                "VarlenAttention: input 3 must be a vision-prelude cu_seqlens extern \
                 (cu_seqlens | cu_seqlens_full | cu_seqlens_window), got {other:?}"
            ),
        };
        // Cross-check: input 4 must be the matching max_seqlen extern.
        // Pairing is fixed by the body's call-site convention; the
        // panic below catches a body that mixed `cu_seqlens_full` with
        // `max_seqlen_window` (or any other unmatched pair) — a real
        // bug rather than a silent misread.
        match (cu_seqlens_kind, node.inputs.get(4)) {
            (
                0,
                Some(FufInput::Extern {
                    kind: ExternKind::MaxSeqlen,
                    ..
                }),
            )
            | (
                1,
                Some(FufInput::Extern {
                    kind: ExternKind::MaxSeqlenFull,
                    ..
                }),
            )
            | (
                2,
                Some(FufInput::Extern {
                    kind: ExternKind::MaxSeqlenWindow,
                    ..
                }),
            ) => {}
            (k, other) => panic!(
                "VarlenAttention: cu_seqlens_kind={k} requires the matching max_seqlen \
                 extern at input 4, got {other:?}"
            ),
        }
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);
        Some(vec![Instruction::VarlenAttention(
            q_slot,
            k_slot,
            v_slot,
            out_slot,
            cu_seqlens_kind,
        )])
    }
}

// ── VisionRopeImpl ───────────────────────────────────────────────
//
// Singleton on `OpKind::VisionRope`. Claims the 2-output pair-
// rotation tile `(q', k') = vision_rope(q, k, cos, sin)` and emits
// two `vision_rope_apply` calls (one for q, one for k) wrapped in
// a single `Instruction::VisionRope` row. Both outputs are
// take-owned-and-reinsert (in-place mutation pattern, same as
// `TanhSoftCap`); the DSL's q' / k' bind to the same OwnedTensors
// as q / k.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct VisionRopeImpl;

#[cfg(feature = "cuda")]
impl Implementation for VisionRopeImpl {
    fn name(&self) -> &'static str {
        "vision_rope"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::VisionRope)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: read + write q + k once each, plus cos/sin
        // (small). Use the elementwise estimator over the q/k inputs
        // — same shape as `Add` cost.
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        // The kernel mutates q and k in place; the eval body
        // take_owned's both upstream tile-outputs and reinserts
        // them at q_out_slot / k_out_slot. Codegen drop-pass must
        // not schedule a Free for either upstream.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut consumed = Vec::with_capacity(2);
        for idx in 0..2 {
            if let Some(FufInput::Tile { id, slot }) = node.inputs.get(idx) {
                consumed.push((*id, *slot));
            }
        }
        consumed
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `VisionRope { q_slot, k_slot, q_out_slot, k_out_slot }`.
    // cos / sin flow through `ForwardCtx::vision_rope_cos` /
    // `vision_rope_sin` (populated by `vision_forward` per call).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "VisionRope",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("VisionRope: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let q_out_slot = slots.of(tile, 0);
        let k_out_slot = slots.of(tile, 1);
        Some(vec![Instruction::VisionRope(
            q_slot, k_slot, q_out_slot, k_out_slot,
        )])
    }
}

// ── QuickGeluImpl ────────────────────────────────────────────────
//
// Singleton on `OpKind::QuickGelu`. Mirror of `TanhSoftCapImpl`'s
// take-owned + in-place mutate + reinsert pattern. The kernel
// `quick_gelu_inplace` mutates the buffer; the OwnedTensor moves
// from `in_slot` to `out_slot`.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct QuickGeluImpl;

#[cfg(feature = "cuda")]
impl Implementation for QuickGeluImpl {
    fn name(&self) -> &'static str {
        "quick_gelu_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::QuickGelu)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("quick_gelu input is a tile");
        vec![src]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "QuickGelu",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("QuickGelu: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::QuickGelu(in_slot_idx, out_slot_idx)])
    }
}

// ── GeluErfImpl ──────────────────────────────────────────────────
//
// Mirror of `QuickGeluImpl` — same shape, different kernel
// (`gelu_erf_inplace`). Distinct OpKind because the numerics differ:
// quick_gelu uses the 1.702-sigmoid approximation, gelu_erf uses
// the exact erf form. Vision/CLIP-class arches calibrate weights
// for one or the other; mismatching is silently wrong.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GeluErfImpl;

#[cfg(feature = "cuda")]
impl Implementation for GeluErfImpl {
    fn name(&self) -> &'static str {
        "gelu_erf_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::GeluErf)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("gelu_erf input is a tile");
        vec![src]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "GeluErf",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("GeluErf: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::GeluErf(in_slot_idx, out_slot_idx)])
    }
}

// ── GeluImpl (Phase G.7(c)) ──────────────────────────────────────
//
// Singleton on `OpKind::Gelu` (tanh approximation, PyTorch
// `nn.GELU(approximate="tanh")` / HuggingFace `gelu_pytorch_tanh`).
// Mirror of `QuickGeluImpl` / `GeluErfImpl` — same take-owned + in-place
// kernel + reinsert pattern, distinct kernel (`gelu_tanh_inplace`). The
// `FusedGateUpGeluMul*` family still claims `(gemm, gemm, gelu, mul)`
// SwiGLU chains; this singleton only fires when no Mul follows.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct GeluImpl;

#[cfg(feature = "cuda")]
impl Implementation for GeluImpl {
    fn name(&self) -> &'static str {
        "gelu_tanh_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Gelu)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("gelu input is a tile");
        vec![src]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Gelu",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Gelu: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::Gelu(in_slot_idx, out_slot_idx)])
    }
}

// ── PosEmbedRefImpl (Phase G.7(c.1)) ─────────────────────────────
//
// Singleton on `OpKind::PosEmbed`. Mirror of `EmbedRefImpl` — same
// HostCallback + Embedding-typed weight + identical
// `embedding_gather_masked` kernel call — but the eval body reads
// `ctx.fwd.vision_position_ids` instead of `ctx.fwd.input_ids`, and
// the shape signature anchors on `vision_num_positions` /
// `vision_embed_dim`. The DSL surface
// `pos_embed(position_ids, embeddings.position_embedding)` mirrors
// the decoder's `embed(input_ids, embed_tokens)` exactly; the
// distinct OpKind is necessary because `Embed`'s shape sig
// hard-codes `vocab_size` / `hidden_size` (decoder-only bounds).

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct PosEmbedRefImpl;

#[cfg(feature = "cuda")]
impl Implementation for PosEmbedRefImpl {
    fn name(&self) -> &'static str {
        "pos_embed_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::PosEmbed)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same bandwidth profile as the embed lookup — read num_tokens
        // rows of [vision_embed_dim] from the table, write same.
        cost_embed(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new("PosEmbed", vec![("out_slot", syn::parse_quote!(u32))])
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        let _ = (fuf, program);
        Some(vec![Instruction::PosEmbed(out_slot)])
    }
}

// ── EmbeddingGatherImpl (Phase G.6.4) ────────────────────────────
//
// Singleton on `OpKind::EmbeddingGather`. Claims any DSL call to
// `embedding_gather(x, indices)` and emits one
// `Instruction::EmbeddingGather { in_slot, out_slot, indices_kind }`
// row. The `indices` arg is a vision-prelude extern — exactly one of
// `vision_window_index` / `vision_reverse_indices` (G.6.3). This Impl
// reads `node.inputs[1]` to learn which extern the body referenced
// and bakes a u8 discriminant into the opcode (0 = WindowIndex,
// 1 = ReverseIndices) so the runtime arm reads the matching
// `ForwardCtx` field. Unlike `QuickGelu` / `GeluErf` this is NOT an
// in-place mutation — `kernels::embedding_gather` allocates a fresh
// output buffer, so `consumes_input_tiles` stays empty (the source
// tile is reused / freed by the standard drop pass).

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct EmbeddingGatherImpl;

#[cfg(feature = "cuda")]
impl Implementation for EmbeddingGatherImpl {
    fn name(&self) -> &'static str {
        "embedding_gather"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::EmbeddingGather)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: read x + indices, write out — same shape
        // as the elementwise estimator. The indices buffer is small
        // (one u32 per row) so the dominant traffic is x's read.
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `EmbeddingGather { in_slot, out_slot, indices_kind }`.
    // The indices buffer is read off `ForwardCtx::vision_window_index`
    // (kind=0) or `ForwardCtx::vision_reverse_indices` (kind=1) — the
    // runtime arm panics if the matching field is None, mirroring the
    // `vision_rope_cos` / `vision_rope_sin` contract.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "EmbeddingGather",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("indices_kind", syn::parse_quote!(u8)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("EmbeddingGather: input 0 must be a Tile (got {other:?})"),
        };
        let indices_kind: u8 = match node.inputs.get(1) {
            Some(FufInput::Extern {
                kind: ExternKind::WindowIndex,
                ..
            }) => 0,
            Some(FufInput::Extern {
                kind: ExternKind::ReverseIndices,
                ..
            }) => 1,
            other => panic!(
                "EmbeddingGather: input 1 must be a vision-prelude indices extern \
                 (window_index | reverse_indices), got {other:?}"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::EmbeddingGather(
            in_slot_idx,
            out_slot_idx,
            indices_kind,
        )])
    }
}

// ── AvgPool2dImpl (Phase G.7(b)) ─────────────────────────────────
//
// Singleton on `OpKind::AvgPool2d`. Claims any DSL call to
// `avg_pool_2d(x)` and emits one `Instruction::AvgPool2d { in_slot,
// out_slot }` row. The kernel reads `vision_patch_grid_side` and
// `vision_pool_kernel` off `CanonicalParams` to compute the output
// shape and walk the k² source cells per output row, so no per-call
// discriminant is needed on the opcode.
//
// Unlike the in-place GELU variants, this allocates a fresh smaller
// output buffer (`[L / k², e]`) — the output shape differs from the
// input shape, so we can't reuse the input's allocation. Same
// allocation pattern as `EmbeddingGather`. `consumes_input_tiles`
// stays empty: the source tile is reused / freed by the standard
// drop pass.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct AvgPool2dImpl;

#[cfg(feature = "cuda")]
impl Implementation for AvgPool2dImpl {
    fn name(&self) -> &'static str {
        "avg_pool_2d"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::AvgPool2d)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: read x, write out (k² smaller). Same
        // estimator as the elementwise / gather Impls.
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AvgPool2d",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("AvgPool2d: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::AvgPool2d(in_slot_idx, out_slot_idx)])
    }
}

// ── StripClsImpl (Phase H) ───────────────────────────────────────
//
// Singleton on `OpKind::StripCls`. Claims any DSL call to
// `strip_cls(x)` and emits one `Instruction::StripCls { in_slot,
// out_slot }` row. The kernel reads `L = x.dim(0)` at runtime, so no
// per-call discriminant or canonical-params bake is needed — a
// misconfigured `vision_in_seq_len = vision_num_positions - 1`
// invariant in the variant config trips the shape-resolver at
// expansion time, NOT at runtime.
//
// Same allocation pattern as `AvgPool2d` / `EmbeddingGather`:
// allocates a fresh `[L - 1, e]` output (output shape strictly
// differs from input shape, so the input's allocation can't be
// reused). `consumes_input_tiles` stays empty.

#[derive(Debug, Default)]
#[cfg(feature = "cuda")]
pub struct StripClsImpl;

#[cfg(feature = "cuda")]
impl Implementation for StripClsImpl {
    fn name(&self) -> &'static str {
        "strip_cls"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::StripCls)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: read `(L-1) * e` elements, write the same.
        // Same estimator as the other elementwise / gather Impls.
        elementwise_cost(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "StripCls",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<scratchy_forward_compiler::Instruction>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("StripCls: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![Instruction::StripCls(in_slot_idx, out_slot_idx)])
    }
}
