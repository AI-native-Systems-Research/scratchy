// SPDX-License-Identifier: Apache-2.0
//! THE DECODE MEGAKERNEL, COMPILED — per bucket-1 tape, at expansion.
//!
//! The decode forward becomes generated kernels. Every command the planned context
//! ([`GateCtx::decode_one`]) admits is fed to the SHARED plan ([`plan`]) with the dataflow of the
//! step row it came from ([`RowAccess`], the barrier walk's hazard signature named): all commands
//! of a row carry the row's whole access set, so a multi-command row is ordered by its own
//! writes. Every admitted command must be playable — an adapter ([`MkAdapter::of`]) and every
//! load-time scalar modeled ([`MkLoadConstant`]) — or the bake fails, naming each command that
//! stopped it.
//!
//! The shared pass offers every RUN one launch may play ([`runs`]: units in which every wait is
//! met by one threadgroup), and each run is COMPILED here into MSL: one `[[kernel]]` whose body is
//! the run's steps in order as straight-line adapter calls — every constant a literal of the
//! step's policy struct, every geometry a literal — and nothing else: nothing in it waits on
//! another threadgroup. Its work split over the launch's `MK_P` threadgroups — each spread step's
//! virtual threadgroups per item and its items' cursor, each lane group's threadgroup — is
//! function constants. Which runs play the forward, beside each command's own dispatch kernel,
//! and how each splits, is solved at load from the device's measured facts (the runtime's
//! `split`). The tape's layer loops stay ROLLED exactly as the tape keeps them: a REGION — the
//! units from a loop iteration's start to the next boundary every plan keeps — offers the same
//! runs in every iteration, which the bake checks, so one kernel plays a run in every iteration.
//! Only binding ADDRESSES and the load's scalars are runtime data: a launch reads its steps'
//! addresses from the block of the address table its region instance binds, at positions fixed
//! here.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;

use scratchy_subtile::megakernel_plan::{
    FusionPlan, Items, Placement, Run, StepFlow, UnitIx, plan, required_launches, runs,
};
use scratchy_target_metal::interpreter::metal::megakernel::MK_BODIES;
use scratchy_target_metal::msl_offline::{AIR_TO_METALLIB, MSL_TO_AIR};
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantType, ConstantValue};
use scratchy_target_metal::tape::ids::{
    ArenaSlotIdx, HeadDim, LayerId, NumKvHeads, NumQHeads, NumTokens, TqDecodeHeads,
};
use scratchy_target_metal::tape::kernel_constants::{
    AttentionViaCacheConstants, AttentionViaCacheTqConstants,
};
use scratchy_target_metal::tape::lowered::{
    Binding, BindingMask, CapPatch, CommandOrigin, GateCtx, GatedCommand, KernelId, LoweredCommand,
    LoweredMetalTape, MK_FC_CAL, MK_FC_HEADS, MK_FC_LOAD, MK_FC_SPLIT, MK_TG_MEMORY, MK_THREADS,
    MScaleAxis, MegakernelError, MegakernelTape, MkAdapter, MkCalibration, MkConst, MkGeometry,
    MkHeads, MkKernelStep, MkLoadConstant, MkLoadSource, MkPlace, MkRegion, MkRun,
    MkStreamBindings, MkStreamWork, MkUnit, MkWork, PatchTarget, RuntimeBindingKind, TapeLoop,
    baked, mk_geometry,
};
use scratchy_target_metal::tape::step::{MetalLoc, MetalStep, MetalStepTape, RowAccess, StepRow};

use crate::static_tape::BakeDefect;

/// K or V: the halves of a KV cache, of the codec's fp16 scratch and of a coded layer's store.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Half {
    K,
    V,
}

/// What the plan orders: a row's [`MetalLoc`], refined by what each command BINDS — a named
/// region of the op scratch (its layout gives every region its own offset, and a binding addresses
/// exactly its region), a K or V half. Commands of one row, and rows touching different regions or
/// halves, then wait only on what they actually share.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MkLoc {
    Arena(ArenaSlotIdx),
    OpScratch(u32),
    Kv(LayerId, Half),
    TqScratch(Half),
    TqStore(LayerId, Half),
    /// A row location no binding resolves to (the codec's staging buffer): every command of the
    /// row touches it.
    Row(MetalLoc),
    /// Memory outside the rows' vocabulary a command may write (the split-K, roped-K and unfused
    /// attention scratches, a Gated-DeltaNet state): read and written by every command binding it.
    Private(u8, u32),
}

impl MkLoc {
    /// The row location this refines (`None`: private).
    fn row(self) -> Option<MetalLoc> {
        Some(match self {
            Self::Arena(s) => MetalLoc::Arena(s),
            Self::OpScratch(_) => MetalLoc::OpScratch,
            Self::Kv(l, _) => MetalLoc::Kv(l),
            Self::TqScratch(_) => MetalLoc::TqScratch,
            Self::TqStore(l, _) => MetalLoc::TqStore(l),
            Self::Row(l) => l,
            Self::Private(..) => return None,
        })
    }
}

/// Where KV physically lives. With the model's codec TurboQuant, the
/// runtime binds every CODED layer's `KvCacheK/V` to ONE fp16 scratch (the model's runtime
/// factory hands out `scratch_{k,v}_table` for them) and keeps the layer's own KV in its packed
/// store — so a row's `Kv(l)` of a coded layer is two locations, the shared [`MetalLoc::TqScratch`]
/// and the layer's [`MetalLoc::TqStore`]. The coded layers are the ones the planned commands bind
/// a packed store for; with the codec off there are none and `Kv(l)` is the layer's own pool.
struct KvAliasing {
    coded: HashSet<LayerId>,
}

impl KvAliasing {
    fn of(commands: &[GatedCommand], admitted: &[usize]) -> Self {
        let coded = admitted
            .iter()
            .flat_map(|&i| commands[i].command.bindings)
            .filter_map(|b| match *b {
                Binding::Runtime {
                    kind:
                        RuntimeBindingKind::TqPackedK { layer }
                        | RuntimeBindingKind::TqPackedV { layer }
                        | RuntimeBindingKind::TqNormsK { layer }
                        | RuntimeBindingKind::TqNormsV { layer },
                    ..
                } => Some(layer),
                _ => None,
            })
            .collect();
        Self { coded }
    }

    /// The physical locations a row's `l` is.
    fn row(&self, l: MetalLoc) -> Vec<MetalLoc> {
        match l {
            MetalLoc::Kv(layer) if self.coded.contains(&layer) => {
                vec![MetalLoc::TqScratch, MetalLoc::TqStore(layer)]
            }
            other => vec![other],
        }
    }

    /// The location a binding points at, when a step can write it: an arena slot, a region of the
    /// op scratch, a KV half of a layer (or, coded, of the shared scratch), a coded layer's packed
    /// store, a private scratch or state. Weights, runtime inputs and inline scalars are never
    /// written by a step.
    fn binding(&self, b: &Binding) -> Option<MkLoc> {
        use RuntimeBindingKind as K;
        let kv = |layer: LayerId, h: Half| match self.coded.contains(&layer) {
            true => MkLoc::TqScratch(h),
            false => MkLoc::Kv(layer, h),
        };
        Some(match *b {
            Binding::ArenaSlot { slot, .. } => MkLoc::Arena(ArenaSlotIdx(slot)),
            Binding::MoeScratch { byte_offset, .. } => MkLoc::OpScratch(byte_offset),
            Binding::Scratch { .. } => MkLoc::Private(0, 0),
            Binding::RopedKScratch { .. } => MkLoc::Private(1, 0),
            Binding::AttnUnfusedScratch { .. } => MkLoc::Private(2, 0),
            Binding::Runtime { kind, .. } => match kind {
                K::KvCacheK { layer } => kv(layer, Half::K),
                K::KvCacheV { layer } => kv(layer, Half::V),
                K::TqPackedK { layer } | K::TqNormsK { layer } => MkLoc::TqStore(layer, Half::K),
                K::TqPackedV { layer } | K::TqNormsV { layer } => MkLoc::TqStore(layer, Half::V),
                K::GdnConvState { layer } => MkLoc::Private(3, layer.0),
                K::GdnSsmState { layer } => MkLoc::Private(4, layer.0),
                _ => return None,
            },
            _ => return None,
        })
    }

    /// Command `cmd`'s dataflow in its row's physical `reads` / `writes`: each row location as the
    /// command's bindings refine it — a location it binds nothing of it does not touch, a binding
    /// outside the adapter's `written` mask it does not write — plus what it binds privately; a
    /// row location no binding resolves to stays whole. `Err`: a binding outside its row's
    /// dataflow.
    fn flow(
        &self,
        cmd: &LoweredCommand,
        written: BindingMask,
        reads: &[MetalLoc],
        writes: &[MetalLoc],
    ) -> Result<(Vec<MkLoc>, Vec<MkLoc>), MegakernelError> {
        let mut bound = Vec::new();
        for b in cmd.bindings {
            let Some(f) = self.binding(b) else { continue };
            if f.row()
                .is_some_and(|l| !reads.contains(&l) && !writes.contains(&l))
            {
                return Err(MegakernelError::UnstatedLocation {
                    symbol: cmd.function,
                    binding: b.binding_index(),
                });
            }
            bound.push((f, written.contains(b.binding_index())));
        }
        let pick = |row: &[MetalLoc], write: bool| {
            let mut v: Vec<MkLoc> = Vec::new();
            let refined = bound
                .iter()
                .filter(|(f, w)| (*w || !write) && f.row().is_none_or(|l| row.contains(&l)));
            let whole = row.iter().filter(|l| matches!(l, MetalLoc::CodecStaging));
            for f in refined
                .map(|(f, _)| *f)
                .chain(whole.map(|l| MkLoc::Row(*l)))
            {
                if !v.contains(&f) {
                    v.push(f);
                }
            }
            v
        };
        Ok((pick(reads, false), pick(writes, true)))
    }
}

/// A decode attention (fp16 KV) whose query heads may group per virtual threadgroup — the
/// device's pick for TurboQuant decode at the launch's threadgroups ([`TqDecodeHeads::for_group`]),
/// made at load ([`MkHeads`]) — by its geometry `(head_dim, q heads, kv heads)`. `None`: not a
/// decode attention, or one head on every GPU. The body computes each head exactly as it does one
/// head per threadgroup.
fn heads_of(cmd: &LoweredCommand) -> Option<(u32, u32, u32)> {
    if cmd.kernel != KernelId::AttentionViaCache {
        return None;
    }
    let at = |slot: ConstSlot| {
        let c = cmd.constants.iter().find(|v| v.index == slot.get());
        c.map(|v| v.bits)
    };
    use AttentionViaCacheConstants as A;
    let (hd, q, kv) = (at(A::HEAD_DIM)?, at(A::NUM_Q_HEADS)?, at(A::NUM_KV_HEADS)?);
    TqDecodeHeads::candidates(HeadDim(hd), NumQHeads(q), NumKvHeads(kv))
        .any(|h| h.get() > 1)
        .then_some((hd, q, kv))
}

/// A command as a generated kernel plays it: its constants and grid, one head per virtual
/// threadgroup (a decode attention's grouping, [`heads_of`], divides the grid where the step is
/// spelled).
fn played(cmd: &LoweredCommand) -> (Vec<ConstantValue>, (u32, u32, u32)) {
    (
        cmd.constants.to_vec(),
        cmd.dispatch.threadgroups_at(NumTokens(1), 1),
    )
}

/// A step's geometry in items of `vtgs_per_item` virtual threadgroups, or (`None`) of as many as
/// one threadgroup's threads and memory hold — its widest, what the plan sees.
fn step_geometry(
    cmd: &LoweredCommand,
    adapter: &MkAdapter,
    vtgs_per_item: Option<u32>,
) -> Result<MkGeometry, MegakernelError> {
    let (_, grid) = played(cmd);
    let tpg = cmd.dispatch.threads_per_threadgroup;
    let widest = mk_geometry(grid, tpg, adapter.tg_bytes, MK_THREADS)?;
    match vtgs_per_item {
        Some(k) => mk_geometry(grid, tpg, adapter.tg_bytes, k * widest.vtg_stride),
        None => Ok(widest),
    }
}

fn defect(what: &str, e: MegakernelError) -> BakeDefect {
    BakeDefect(format!("megakernel: {what}: {e}"))
}

/// The tape's command positions as its loops nest them — the SAME walk as
/// [`LoweredMetalTape::expanded_origins`] (outermost loop first, nesting by containment).
enum Node {
    Cmd(usize),
    Loop { iters: u32, body: Vec<Node> },
}

fn nodes(loops: &[TapeLoop], range: std::ops::Range<usize>) -> Vec<Node> {
    let mut out = Vec::new();
    let mut pos = range.start;
    while pos < range.end {
        let entered = loops
            .iter()
            .enumerate()
            .find(|(_, l)| l.start as usize == pos && (l.start + l.period) as usize <= range.end);
        match entered {
            Some((i, l)) => {
                let body = pos..pos + l.period as usize;
                let inner = nodes(&loops[i + 1..], body.clone());
                out.push(Node::Loop {
                    iters: l.iters,
                    body: inner,
                });
                pos = body.end;
            }
            None => {
                out.push(Node::Cmd(pos));
                pos += 1;
            }
        }
    }
    out
}

/// The expanded commands' baked positions, each flagged when a loop iteration starts at it or a
/// loop ended right before it.
fn expand(nodes: &[Node], out: &mut Vec<(usize, bool)>, pending: &mut bool) {
    for n in nodes {
        match n {
            Node::Cmd(b) => out.push((*b, std::mem::take(pending))),
            Node::Loop { iters, body } => {
                for _ in 0..*iters {
                    *pending = true;
                    expand(body, out, pending);
                }
                *pending = true;
            }
        }
    }
}

/// The load-time scalars of the baked command at `b`: its capacity patches and what the device
/// serves; `Err` names a patch the kernel cannot take as a function constant.
fn load_sources(
    tape: &LoweredMetalTape,
    patches: &[CapPatch],
    b: usize,
) -> Result<Vec<MkLoadSource>, MegakernelError> {
    let cmd = &tape.commands[b].command;
    let mut out = cmd.device_served().to_vec();
    for p in patches.iter().filter(|p| p.cmd_idx as usize == b) {
        let source = match p.target {
            PatchTarget::Constant(pos) => {
                MkLoadSource::Constant(ConstSlot(cmd.constants[pos as usize].index))
            }
            PatchTarget::Threadgroups(ax) => match MScaleAxis::of_index(ax) {
                Some(axis) => MkLoadSource::Threadgroups(axis),
                None => {
                    return Err(MegakernelError::LoadPatch {
                        symbol: cmd.function,
                        target: p.target,
                    });
                }
            },
            target => {
                return Err(MegakernelError::LoadPatch {
                    symbol: cmd.function,
                    target,
                });
            }
        };
        if !out.contains(&source) {
            out.push(source);
        }
    }
    Ok(out)
}

/// Why a kernel never runs in a bucket-1 decode forward.
#[derive(Clone, Copy, Debug)]
enum NotAtDecode {
    /// A prompt's forward: a prefill attention / matmul / MoE grouping / TurboQuant staging, the
    /// lm-head row slice.
    Prefill,
    /// A decode BATCH (`SMALL_M_TOKENS` steps).
    DecodeBatch,
    /// The vision tower.
    Vision,
    /// Metadata: no command is dispatched.
    Metadata,
}

impl NotAtDecode {
    fn name(self) -> &'static str {
        match self {
            Self::Prefill => "prefill-only",
            Self::DecodeBatch => "decode-batch-only",
            Self::Vision => "vision-only",
            Self::Metadata => "metadata",
        }
    }
}

/// Whether the decode's segment kernels play `kernel` — EXHAUSTIVE over [`KernelId`], so a new
/// kernel is classed before any tape can emit it. The bake holds every admitted bucket-1 decode
/// command to it: a played kernel's symbol must have an adapter ([`MegakernelError::NoAdapter`]),
/// and a decode command whose kernel is classed [`NotAtDecode`] fails the bake
/// ([`MegakernelError::NotAtDecode`]: the class is wrong) — never a dispatch beside the segments.
fn at_decode(kernel: KernelId) -> Result<(), NotAtDecode> {
    use KernelId as K;
    match kernel {
        K::Embed
        | K::AffineEmbed
        | K::MmEmbedSplice
        | K::RmsNorm
        | K::RmsNormUnit
        | K::FusedAddRmsNorm
        | K::NormAddScalarMul
        | K::VisionLayerNorm
        | K::ScalarWeightMul
        | K::ScalarMul
        | K::TanhSoftCap
        | K::Add
        | K::BiasAdd
        | K::Gemm
        | K::AffineQmvQuad
        | K::AffineQmvFast
        | K::AffineQmv
        | K::FusedGateUpSiluMul
        | K::SiluMul
        | K::GeluMul
        | K::RopeAppend
        | K::RopeAppendNormed
        | K::TqQuantizeToPacked
        | K::AttentionViaCache
        | K::AttentionViaCacheTq
        | K::GateSplit
        | K::GateApply
        | K::GateScale
        | K::GatedDeltaNet
        | K::Softmax
        | K::ArgPartitionTopK
        | K::SliceTrailingColsU32
        | K::TakeAlongAxis
        | K::MoePerExpertScale
        | K::AffineGatherQmvFast
        | K::AffineGatherQmv
        | K::MoeWeightedSum => Ok(()),
        K::AttentionPrefillSdpaPaged
        | K::RopeOnceNax
        | K::RopeOnceSteel
        | K::RopeOnceGqaShared
        | K::AttnGatherKRope
        | K::AttnGatherVCopyT
        | K::AttnQConvert
        | K::AttnOConvert
        | K::AttnCausalSoftmax
        | K::AttnGemmQk
        | K::AttnGemmPv
        | K::AffineQmmT
        | K::AffineGatherQmmT
        | K::AffineGatherQmmTNax
        | K::AffineQmmTSplitK
        | K::AffineQmmTNax
        | K::AffineW4a8Quant
        | K::AffineQmmW4a8
        | K::AffineGatherW4a8Quant
        | K::AffineGatherQmmW4a8
        | K::SplitKReduceSum
        | K::GatherLastToken
        | K::ScatterFirstToLastRow
        | K::MoeGroupOffsets
        | K::MoeGroupInit
        | K::MoeGroupScatter
        | K::MoeGroupScatterQ8
        | K::MoeGroupGather
        | K::TqStageRotated
        | K::TqRotateRows => Err(NotAtDecode::Prefill),
        K::AffineQmmSmallM => Err(NotAtDecode::DecodeBatch),
        K::VisionRope
        | K::VisionVarlenAttn
        | K::EmbeddingGather
        | K::VisionGelu
        | K::VisionLoadPixels => Err(NotAtDecode::Vision),
        K::Reshape => Err(NotAtDecode::Metadata),
    }
}

/// Whether a step touches every location item by item ([`StepFlow::local`]) — EXHAUSTIVE over
/// [`MetalStep`], so a new step is classed before any tape can emit it: an elementwise step, a
/// per-head or per-head-group one (rope, attention, the KV codec), the experts' per-row combine.
/// A matmul (every output reads the whole input), a norm (a reduction over the row), and the
/// router's ranking are not.
fn item_local(step: &MetalStep) -> bool {
    use MetalStep as S;
    use scratchy_target_metal::tape::step::MoeStep as M;
    match step {
        S::SpliceMmEmbeds(..)
        | S::ScalarWeightMul(..)
        | S::ScalarMul(..)
        | S::MetalBiasAdd(..)
        | S::SiluMul(..)
        | S::GeluMul(..)
        | S::Gelu(..)
        | S::GeluErf(..)
        | S::QuickGelu(..)
        | S::TanhSoftCap(..)
        | S::Add(..)
        | S::RopeAppend(..)
        | S::RopeAppendNormed(..)
        | S::AttentionViaCache(..)
        | S::SlidingAttentionViaCache(..)
        | S::GateSplit(..)
        | S::GateApply(..)
        | S::GateScale(..)
        | S::GatedDeltaNet(..)
        | S::KvEncode(..)
        | S::KvStage(..)
        | S::RotateRows(..)
        | S::AttnPackedKv(..)
        | S::Moe(_, M::GatedAct(..) | M::Combine(..)) => true,
        S::Embed(..)
        | S::AffineEmbed(..)
        | S::Reshape(..)
        | S::RmsNorm(..)
        | S::ScalarOffsetRmsNorm(..)
        | S::RmsNormUnit(..)
        | S::MeanSubRmsNorm(..)
        | S::MeanSubRmsNormBiasAdd(..)
        | S::FusedAddRmsNorm(..)
        | S::FusedAddRmsNormWithOffset(..)
        | S::NormAddScalarMul(..)
        | S::Gemm(..)
        | S::AffineQmm(..)
        | S::FusedGateUpSiluMul(..)
        | S::FusedGateUpGeluMul(..)
        | S::AttentionPrefillPaged(..)
        | S::SlidingAttentionPrefillPaged(..)
        | S::EncoderAttention(..)
        | S::VarlenAttention(..)
        | S::VisionRope(..)
        | S::LoadPixels(..)
        | S::LoadPosEmbeds(..)
        | S::EmbeddingGather(..)
        | S::SampleRows(..)
        | S::Moe(
            _,
            M::RouterNorm(..)
            | M::RouterLogits(..)
            | M::Softmax(..)
            | M::Argsort
            | M::TopK
            | M::GatherScores
            | M::Scale(..)
            | M::Renorm
            | M::ExpertScale(..)
            | M::Sort(..)
            | M::ExpertMatmul(..)
            | M::Unsort,
        ) => false,
    }
}

/// The planned decode forward: the admitted commands (expanded indices, in order), their plan,
/// every run a launch may play, and each admitted step's dataflow and work.
struct Planned {
    admitted: Vec<usize>,
    plan: FusionPlan,
    runs: Vec<Run>,
    flows: Vec<StepFlow<MkLoc>>,
    work: Vec<MkWork>,
    calibrations: Vec<Calibration>,
    /// The units that open a launch on any placement ([`required_launches`]).
    required: Vec<UnitIx>,
}

/// A matvec body the load measures the device with ([`MkCalibration`]): `call` (the streaming
/// adapter's calibration adapter) at row length `k`, its constant policy spelled from `constants`
/// (`k` at slot `k_slot`, the rows at `n_slot`), its virtual threadgroups `tpg` (`stride` threads)
/// of `rows` rows, at most `widest` to an item.
#[derive(Clone, PartialEq, Debug)]
struct Calibration {
    call: &'static str,
    k: u32,
    rows: u32,
    bits: u32,
    group: u32,
    scale_bytes: u32,
    bindings: MkStreamBindings,
    tpg: (u32, u32, u32),
    /// The axis of the step's grid its rows run along (its extent: rows ÷ `rows`); any other
    /// axis repeats the stream (an expert, a batch row), each a stream of the same rows.
    rows_axis: usize,
    stride: u32,
    widest: u32,
    tg_bytes: u32,
    tg_memory: u32,
    constants: &'static [MkConst],
    k_slot: ConstSlot,
    n_slot: ConstSlot,
    /// The step's other constants, as its policy spells them.
    fixed: Vec<ConstantValue>,
}

/// Whether a step tape decodes: it attends over the KV cache earlier forwards filled, or scans a
/// recurrent state they left. A refused canonical's empty tape does not, nor does an encoder's
/// (its attention reads only the step's own K/V, whatever it appends to a cache).
fn decodes(steps: &MetalStepTape) -> bool {
    steps.backbone.iter().any(|r| {
        matches!(
            r,
            StepRow::Step(
                MetalStep::AttentionViaCache(..)
                    | MetalStep::SlidingAttentionViaCache(..)
                    | MetalStep::AttnPackedKv(..)
                    | MetalStep::GatedDeltaNet(..),
                _,
            )
        )
    })
}

/// The megakernel of a bucket-1 decode tape: one [`MegakernelTape`], its generated kernels able
/// to play every command a decode step admits — a command they cannot play fails the bake. `tape`
/// is the capacity-0 bake, `patches` its load patches, `row_commands` the commands each step row
/// became, `steps` the rows' dataflow. A tape without a decode step ([`decodes`]) has none.
pub fn bake_megakernel(
    tape: &LoweredMetalTape,
    patches: &[CapPatch],
    row_commands: &[u32],
    steps: &MetalStepTape,
) -> Result<Vec<MegakernelTape>, BakeDefect> {
    let access: Vec<&RowAccess> = steps
        .backbone_access
        .iter()
        .chain(&steps.lm_head_access)
        .collect();
    let rows: Vec<&StepRow> = steps.backbone.iter().chain(&steps.lm_head).collect();
    if !decodes(steps) {
        return Ok(Vec::new());
    }
    if access.len() != row_commands.len() {
        return Err(BakeDefect(format!(
            "megakernel: {} row dataflows for {} lowered rows",
            access.len(),
            row_commands.len()
        )));
    }
    let row_of: Vec<usize> = (0..row_commands.len())
        .flat_map(|r| std::iter::repeat_n(r, row_commands[r] as usize))
        .collect();
    if row_of.len() != tape.commands.len() {
        return Err(BakeDefect(format!(
            "megakernel: rows account for {} commands, the tape has {}",
            row_of.len(),
            tape.commands.len()
        )));
    }
    let origins = tape.expanded_origins();
    let commands = tape.commands_expanded();
    let tree = nodes(tape.loops, 0..tape.commands.len());
    let mut walk = Vec::with_capacity(commands.len());
    expand(&tree, &mut walk, &mut false);
    if !walk.iter().map(|w| w.0).eq(origins.iter().map(|o| o.baked)) {
        return Err(BakeDefect(
            "megakernel: the loop tree does not expand as the tape".into(),
        ));
    }
    let [ctx, span] = [false, true].map(GateCtx::decode_one);
    let admitted: Vec<usize> = (0..commands.len())
        .filter(|&i| commands[i].gate.is_none_or(|g| g.admits(ctx)))
        .collect();
    // Every admitted command inside a generated kernel: a command they cannot play fails the
    // bake (each distinct cause named). The worker plays the kernels whether or not the sequence
    // holds an unrotated span block, so no gate may read that here.
    let mut stops: Vec<MegakernelError> = commands
        .iter()
        .filter(|c| c.gate.is_some_and(|g| g.admits(ctx) != g.admits(span)))
        .map(|c| MegakernelError::SpanDependentGate {
            function: c.command.function,
        })
        .collect();
    stops.dedup();
    for &i in &admitted {
        let cmd = &commands[i].command;
        let stop = match (
            at_decode(cmd.kernel),
            MkAdapter::of(cmd.library, cmd.function),
        ) {
            (Err(class), _) => Err(MegakernelError::NotAtDecode {
                kernel: cmd.kernel,
                function: cmd.function,
                class: class.name(),
            }),
            (Ok(()), None) => Err(MegakernelError::NoAdapter {
                library: cmd.library,
                function: cmd.function,
            }),
            (Ok(()), Some(_)) => load_sources(tape, patches, origins[i].baked).map(drop),
        };
        if let Err(e) = stop
            && !stops.contains(&e)
        {
            stops.push(e);
        }
    }
    if !stops.is_empty() {
        let stops: Vec<String> = stops.iter().map(ToString::to_string).collect();
        return Err(BakeDefect(format!(
            "the bucket-1 decode tape is not playable by generated kernels: {}",
            stops.join("; ")
        )));
    }
    let kv = KvAliasing::of(&commands, &admitted);
    let mut flows = Vec::with_capacity(admitted.len());
    let mut work = Vec::with_capacity(admitted.len());
    let mut heads_slots: HashMap<usize, ConstSlot> = HashMap::new();
    let mut calibrations: Vec<Calibration> = Vec::new();
    let mut pending = false;
    let mut next = admitted.iter().peekable();
    for (e, &(_, brk)) in walk.iter().enumerate() {
        pending |= brk;
        if next.next_if_eq(&&e).is_none() {
            continue;
        }
        let cmd = &commands[e].command;
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("checked above");
        let origin = origins[e];
        let sources = load_sources(tape, patches, origin.baked).expect("checked above");
        // A grid the load sizes is spread whatever its baked size: a pinned unit plays one
        // item.
        let load = sources
            .iter()
            .any(|s| matches!(s, MkLoadSource::Threadgroups(_)));
        let g = step_geometry(cmd, adapter, None).map_err(|e| defect(cmd.function, e))?;
        let (constants, grid) = played(cmd);
        let heads = heads_of(cmd).map(|(head_dim, q, kv)| {
            let next = ConstSlot(MK_FC_HEADS.0 + heads_slots.len() as u16);
            let slot = *heads_slots.entry(origin.baked).or_insert(next);
            MkHeads {
                slot,
                head_dim,
                q,
                kv,
            }
        });
        // A streaming step streams at the rates its body's calibration measures, at its row
        // length.
        let stream = adapter.stream.and_then(|st| {
            let bytes = st.bytes_per_vtg(&constants)?;
            let k = constants.iter().find(|c| c.index == st.k.get())?.bits;
            let tpg = cmd.dispatch.threads_per_threadgroup;
            let n = constants.iter().find(|c| c.index == st.n.get())?.bits;
            let rows_axis = [grid.0, grid.1, grid.2]
                .iter()
                .position(|&e| e == n.div_ceil(st.rows))?;
            let fixed = (constants.iter())
                .filter(|c| c.index != st.k.get() && c.index != st.n.get())
                .copied()
                .collect();
            let cal = Calibration {
                call: st.calibrate,
                k,
                rows: st.rows,
                bits: st.bits,
                group: st.group,
                scale_bytes: st.scale_bytes,
                bindings: st.bindings,
                tpg,
                rows_axis,
                stride: g.vtg_stride,
                widest: g.vtgs_per_item,
                tg_bytes: adapter.tg_bytes.0,
                tg_memory: g.tg_memory,
                constants: adapter.constants,
                k_slot: st.k,
                n_slot: st.n,
                fixed,
            };
            let at = calibrations
                .iter()
                .position(|c| *c == cal)
                .unwrap_or_else(|| {
                    calibrations.push(cal);
                    calibrations.len() - 1
                });
            Some(MkStreamWork {
                bytes,
                calibration: at as u32,
            })
        });
        // The plan sees one head per virtual threadgroup: the most items the step can have.
        // A streaming step's items are its virtual threadgroups — the work split packs them —
        // so it is never folded into one threadgroup's chain because they fit one item.
        let vtgs = NonZeroU32::new(grid.0 * grid.1 * grid.2);
        let items = match (load, stream.and(vtgs)) {
            (true, _) => Items(g.items.0.max(NonZeroU32::MIN.saturating_add(1))),
            (false, Some(vtgs)) => Items(vtgs),
            (false, None) => g.items,
        };
        work.push(MkWork {
            vtg_threads: g.vtg_stride,
            widest: g.vtgs_per_item,
            stream,
            grid,
            load,
            heads,
        });
        let row = access[row_of[origin.baked]];
        let at = |l: &MetalLoc| kv.row(l.advanced(origin.layers));
        let reads: Vec<MetalLoc> = row.reads.iter().flat_map(at).collect();
        let writes: Vec<MetalLoc> = row.writes.iter().flat_map(at).collect();
        // The plan orders what the ROW states, as the command's bindings refine it; a command
        // binding anything else would race.
        let (reads, writes) = kv
            .flow(cmd, adapter.writes, &reads, &writes)
            .map_err(|e| defect(cmd.function, e))?;
        let row_step = match rows[row_of[origin.baked]] {
            StepRow::Step(step, _) => Some(step),
            StepRow::Loop { .. } => None,
        };
        let local = match row_step.is_some_and(item_local) {
            true => reads.iter().chain(&writes).copied().collect(),
            false => Vec::new(),
        };
        flows.push(StepFlow {
            reads,
            writes,
            items,
            segment_break: std::mem::take(&mut pending),
            whole: adapter.tail && !load,
            local,
        });
    }
    let plan = plan(&flows);
    // The shared scratch is ONE buffer every coded layer reuses: every access to it, across
    // layers, must be ordered.
    let scratch = [Half::K, Half::V].map(MkLoc::TqScratch);
    if let Some(pair) = scratch.iter().find_map(|l| plan.unordered_on(&flows, *l)) {
        let symbol = |u: UnitIx| {
            let first = plan.units[u.0 as usize].steps.start as usize;
            commands[admitted[first]].command.function
        };
        let e = MegakernelError::UnorderedScratch {
            first: symbol(pair.0),
            second: symbol(pair.1),
        };
        return Err(defect("the codec's shared scratch", e));
    }
    let runs = runs(&plan, &flows);
    let required = required_launches(&plan, &flows);
    let planned = Planned {
        admitted,
        plan,
        runs,
        flows,
        work,
        calibrations,
        required,
    };
    let generator = Gen::new(tape, patches, &commands, &origins, &planned)?;
    let generator_longest = generator.longest_launches();
    let mk = generator.kernels()?;
    eprintln!(
        "[m2-megakernel] {}: {} regions, {} runs a launch may play, all {} admitted commands \
         ({} baked steps), {} load constants, {} bytes of MSL; launches per forward the dataflow \
         requires: {}, the longest runs take: {}",
        mk.library,
        mk.regions.len(),
        mk.regions.iter().map(|r| r.runs.len()).sum::<usize>(),
        planned.admitted.len(),
        mk.steps.len(),
        mk.load_constants.len(),
        mk.source.len(),
        planned.required.len(),
        generator_longest,
    );
    Ok(vec![mk])
}

/// A run as its region holds it, relative to the region's first unit: its units, where each
/// runs, and each wait inside — what must be alike in every instance of the region.
#[derive(Clone, PartialEq, Eq, Debug)]
struct RunShape {
    units: Range<u32>,
    places: Vec<Placement>,
    waits: Vec<(u32, u32)>,
}

/// A region of the rolled text: the plan's units of its first instance, how many instances a
/// forward plays, its runs (indices into [`Planned::runs`], those of the first instance), and the
/// address table's blocks its instances bind.
struct Region {
    units: Range<u32>,
    instances: u32,
    runs: Vec<usize>,
    table_at: u32,
    block_len: u32,
}

/// The kernel generator of the decode tape.
struct Gen<'a> {
    commands: &'a [GatedCommand],
    planned: &'a Planned,
    /// Per admitted step: its baked position, its unit.
    baked: Vec<usize>,
    unit_of: Vec<usize>,
    /// Per baked position: its address row (its region, the row's offset in the region's block,
    /// the row's length).
    rows: Vec<Option<(usize, u32, u32)>>,
    /// The regions, in the order the rolled text first reaches them.
    regions: Vec<Region>,
    loads: Vec<MkLoadConstant>,
    load_types: Vec<ConstantType>,
    policies: Vec<String>,
    /// Threadgroup memory the widest co-issued group of the kernel being spelled needs
    /// ([`Gen::coissued`]).
    coissue_tg: u32,
}

impl<'a> Gen<'a> {
    fn new(
        tape: &LoweredMetalTape,
        patches: &[CapPatch],
        commands: &'a [GatedCommand],
        origins: &[CommandOrigin],
        planned: &'a Planned,
    ) -> Result<Self, BakeDefect> {
        let Planned {
            admitted,
            plan,
            runs,
            flows,
            ..
        } = planned;
        let n = tape.commands.len();
        let mut unit_of = vec![0usize; admitted.len()];
        for (u, unit) in plan.units.iter().enumerate() {
            for s in unit.steps.clone() {
                unit_of[s as usize] = u;
            }
        }
        let baked: Vec<usize> = admitted.iter().map(|&e| origins[e].baked).collect();
        let mut first = vec![None; n];
        for (a, &b) in baked.iter().enumerate() {
            first[b].get_or_insert(a);
        }
        // The region instances: from the plan's first unit and every unit a segment break opens.
        let units = plan.units.len() as u32;
        let starts: Vec<u32> = (0..units)
            .filter(|&u| u == 0 || flows[plan.units[u as usize].steps.start as usize].segment_break)
            .collect();
        let ends = starts.iter().skip(1).copied().chain([units]);
        let steps_of = |r: &Range<u32>| {
            let s = plan.units[r.start as usize].steps.start;
            let e = plan.units[r.end as usize - 1].steps.end;
            s..e
        };
        // What every instance of a region must share: its baked steps, its units (relative to
        // its first step) and its runs (relative to its first unit).
        let shape = |r: &Range<u32>| {
            let steps = steps_of(r);
            let held: Vec<usize> = steps.clone().map(|a| baked[a as usize]).collect();
            let unit_steps: Vec<(u32, u32)> = plan.units[r.start as usize..r.end as usize]
                .iter()
                .map(|u| (u.steps.start - steps.start, u.steps.end - steps.start))
                .collect();
            let inside: Vec<usize> = (0..runs.len())
                .filter(|&k| runs[k].units().start >= r.start && runs[k].units().end <= r.end)
                .collect();
            let shapes: Vec<RunShape> = inside
                .iter()
                .map(|&k| {
                    let run = &runs[k];
                    let waits = run
                        .units()
                        .flat_map(|u| {
                            run.local_waits(UnitIx(u))
                                .iter()
                                .map(move |(w, _)| (u, w.0))
                        })
                        .map(|(u, w)| (u - r.start, w - r.start))
                        .collect();
                    RunShape {
                        units: run.units().start - r.start..run.units().end - r.start,
                        places: run.placements().to_vec(),
                        waits,
                    }
                })
                .collect();
            (held, unit_steps, shapes, inside)
        };
        let mut regions: Vec<Region> = Vec::new();
        let mut region_of: HashMap<usize, usize> = HashMap::new();
        let mut rows: Vec<Option<(usize, u32, u32)>> = vec![None; n];
        for (s, e) in starts.iter().copied().zip(ends) {
            let r = s..e;
            let (held, unit_steps, shapes, inside) = shape(&r);
            let key = held[0];
            if let Some(&k) = region_of.get(&key) {
                let (h0, u0, s0, _) = shape(&regions[k].units);
                if (h0, u0, s0) != (held, unit_steps, shapes) {
                    return Err(BakeDefect(format!(
                        "megakernel: the region baked {key} opens plays differently in a later \
                         loop iteration"
                    )));
                }
                regions[k].instances += 1;
                continue;
            }
            let mut block_len = 0u32;
            for &b in &held {
                let cmd = &commands[admitted[first[b].expect("admitted")]].command;
                let row_len = cmd
                    .bindings
                    .iter()
                    .map(|x| u32::from(x.binding_index()) + 1);
                let row_len = row_len.max().unwrap_or(0);
                if rows[b]
                    .replace((regions.len(), block_len, row_len))
                    .is_some()
                {
                    return Err(BakeDefect(format!(
                        "megakernel: `{}` (baked {b}) is played by two regions",
                        cmd.function
                    )));
                }
                block_len += row_len;
            }
            region_of.insert(held[0], regions.len());
            regions.push(Region {
                units: r,
                instances: 1,
                runs: inside,
                table_at: 0,
                block_len,
            });
        }
        let mut table_len = 0u32;
        for region in &mut regions {
            region.table_at = table_len;
            table_len += region.instances * region.block_len;
        }
        let mut loads: Vec<MkLoadConstant> = Vec::new();
        let mut load_types = Vec::new();
        for (a, &b) in baked
            .iter()
            .enumerate()
            .filter(|(a, b)| first[**b] == Some(*a))
        {
            let cmd = &commands[admitted[a]].command;
            let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
            let sources = load_sources(tape, patches, b).map_err(|e| defect(cmd.function, e))?;
            for source in sources {
                let ty = match source {
                    MkLoadSource::Threadgroups(_) => ConstantType::UInt,
                    MkLoadSource::Constant(slot) => {
                        let declared = adapter.constants.iter().find(|c| c.slot == slot);
                        declared.map_or(ConstantType::UInt, |c| c.ty)
                    }
                };
                loads.push(MkLoadConstant {
                    index: ConstSlot(MK_FC_LOAD.0 + loads.len() as u16),
                    baked: b as u32,
                    source,
                });
                load_types.push(ty);
            }
        }
        Ok(Self {
            commands,
            planned,
            baked,
            unit_of,
            rows,
            regions,
            loads,
            load_types,
            policies: Vec::new(),
            coissue_tg: 0,
        })
    }

    /// Launches per forward the regions' longest runs take.
    fn longest_launches(&self) -> u32 {
        let runs = &self.planned.runs;
        (self.regions.iter())
            .map(|r| {
                let (mut at, mut n) = (r.units.start, 0);
                while at < r.units.end {
                    let longest = r.runs.iter().map(|&k| runs[k].units());
                    at = longest
                        .filter(|u| u.start == at)
                        .map(|u| u.end)
                        .max()
                        .unwrap_or(at + 1);
                    n += 1;
                }
                n * r.instances
            })
            .sum()
    }

    fn command(&self, a: usize) -> &'a LoweredCommand {
        &self.commands[self.planned.admitted[a]].command
    }

    /// The load constant reading `source` of the baked command at `b`, by name.
    fn load(&self, b: usize, source: MkLoadSource) -> Option<String> {
        let at = self
            .loads
            .iter()
            .position(|l| l.baked as usize == b && l.source == source);
        at.map(|k| format!("MK_LOAD_{k}"))
    }

    /// The function constant holding admitted step `a`'s heads per virtual threadgroup, by name.
    fn heads(&self, a: usize) -> Option<String> {
        let h = self.planned.work[a].heads?;
        Some(format!("MK_H{}", h.slot.get() - MK_FC_HEADS.get()))
    }

    /// The policy struct spelling the constants of admitted step `a` for its adapter (defined
    /// once per text): each a literal, or the load constant a slot is served by
    /// ([`MkLoadConstant`]); `has_…` = set.
    fn policy(&mut self, a: usize) -> Result<String, BakeDefect> {
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        let mut body = String::new();
        let (constants, _) = played(cmd);
        let heads = self.heads(a);
        for c in adapter.constants {
            if let Some(h) = heads
                .as_ref()
                .filter(|_| c.slot == AttentionViaCacheTqConstants::HEADS)
            {
                let _ = writeln!(
                    body,
                    "  static METAL_FUNC uint {}() {{ return {h}; }}\n  \
                     static METAL_FUNC bool has_{}() {{ return true; }}",
                    c.name, c.name,
                );
                continue;
            }
            let given = constants.iter().find(|v| v.index == c.slot.get());
            if let Some(v) = given.filter(|v| v.ty != c.ty) {
                let e = MegakernelError::ConstantType {
                    symbol: adapter.function,
                    index: v.index,
                    declared: c.ty,
                    given: v.ty,
                };
                return Err(defect(adapter.function, e));
            }
            // An unset constant reads 0, as the dispatch kernels' optional constants default.
            let value = self
                .load(self.baked[a], MkLoadSource::Constant(c.slot))
                .unwrap_or_else(|| literal(c.ty, given.map_or(0, |v: &ConstantValue| v.bits)));
            let set = given.is_some();
            let _ = writeln!(
                body,
                "  static METAL_FUNC {} {}() {{ return {value}; }}\n  \
                 static METAL_FUNC bool has_{}() {{ return {set}; }}",
                msl_type(c.ty),
                c.name,
                c.name,
            );
        }
        let n = match self.policies.iter().position(|p| *p == body) {
            Some(n) => n,
            None => {
                self.policies.push(body);
                self.policies.len() - 1
            }
        };
        Ok(format!("MkC{n}"))
    }

    /// Admitted step `a`'s geometry at its widest: as many virtual threadgroups per item as one
    /// threadgroup's threads and memory hold.
    fn geometry(&self, a: usize) -> Result<MkGeometry, BakeDefect> {
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        step_geometry(cmd, adapter, None).map_err(|e| defect(cmd.function, e))
    }

    /// Every run's kernel, one library: each the units of the run in order, their rows at their
    /// offsets in the block of the region instance the launch binds.
    fn kernels(mut self) -> Result<MegakernelTape, BakeDefect> {
        let planned = self.planned;
        let plan = &planned.plan;
        let mut kernels = String::new();
        let mut splits = String::new();
        let mut split_at = MK_FC_SPLIT.get();
        let mut regions = Vec::with_capacity(self.regions.len());
        let mut steps: Vec<MkKernelStep> = Vec::new();
        let mut kernel_count = 0usize;
        for ri in 0..self.regions.len() {
            let (units, run_ids) = {
                let r = &self.regions[ri];
                (r.units.clone(), r.runs.clone())
            };
            let first_step = plan.units[units.start as usize].steps.start;
            let step_at = steps.len() as u32;
            let mut mk_runs = Vec::with_capacity(run_ids.len());
            for k in run_ids {
                let run = &planned.runs[k];
                let n = kernel_count;
                kernel_count += 1;
                let spread = (run.placements().iter())
                    .filter(|p| matches!(p, Placement::Spread { .. }))
                    .count() as u16;
                for i in 0..spread {
                    let _ = writeln!(
                        splits,
                        "constant uint MK_R{n}_K{i} [[function_constant({})]];\n\
                         constant uint MK_R{n}_C{i} [[function_constant({})]];",
                        split_at + 2 * i,
                        split_at + 2 * i + 1
                    );
                }
                for g in 0..run.groups() as u16 {
                    let _ = writeln!(
                        splits,
                        "constant uint MK_R{n}_L{g} [[function_constant({})]];",
                        split_at + 2 * spread + g
                    );
                }
                self.coissue_tg = 0;
                let mut body = String::new();
                let mut tg_memory = 16;
                let mut ord = 0;
                for (u, placement) in run.units().zip(run.placements()) {
                    let place = match *placement {
                        Placement::Spread { .. } => {
                            ord += 1;
                            Place::Spread {
                                run: n,
                                ord: ord - 1,
                            }
                        }
                        Placement::Pinned(l) => Place::Pinned { run: n, group: l.0 },
                        Placement::Everywhere => Place::Everywhere,
                    };
                    let follows = matches!(place, Place::Pinned { .. })
                        && !run.local_waits(UnitIx(u)).is_empty();
                    let unit_steps = plan.units[u as usize].steps.clone();
                    self.unit(unit_steps.start as usize, place, follows, &mut body)?;
                    for a in unit_steps {
                        tg_memory = tg_memory.max(self.geometry(a as usize)?.tg_memory);
                    }
                }
                let tg_memory = tg_memory.max(self.coissue_tg).next_multiple_of(16);
                let _ = write!(
                    kernels,
                    "[[kernel, max_total_threads_per_threadgroup(1024)]] void mk_r{n}(\n    \
                     constant ulong* mk_a [[buffer(0)]],\n    \
                     uint mk_t [[thread_index_in_threadgroup]],\n    \
                     uint mk_idx [[threadgroup_position_in_grid]]) {{\n  \
                     threadgroup uchar mk_tgm[{tg_memory}] __attribute__((aligned(16)));\n  \
                     const uint mk_p = MK_P;\n{body}}}\n"
                );
                let places: Vec<MkPlace> = (run.placements().iter())
                    .map(|p| match p {
                        Placement::Spread { .. } => MkPlace::Spread,
                        Placement::Pinned(l) => MkPlace::Lane(l.0),
                        Placement::Everywhere => MkPlace::Everywhere,
                    })
                    .collect();
                let waits: Vec<(u32, u32)> = run
                    .units()
                    .flat_map(|u| {
                        run.local_waits(UnitIx(u))
                            .iter()
                            .map(move |(w, _)| (u, w.0))
                    })
                    .map(|(u, w)| (u - units.start, w - units.start))
                    .collect();
                mk_runs.push(MkRun {
                    kernel: String::leak(format!("mk_r{n}")),
                    first: run.units().start - units.start,
                    end: run.units().end - units.start,
                    places: baked(places),
                    waits: baked(waits),
                    split_at: ConstSlot(split_at),
                });
                split_at += 2 * spread + run.groups() as u16;
            }
            let region = &self.regions[ri];
            let held = plan.units[units.start as usize].steps.start
                ..plan.units[units.end as usize - 1].steps.end;
            for a in held {
                let b = self.baked[a as usize];
                let (_, row_at, row_len) = self.rows[b].expect("a played command has a row");
                steps.push(MkKernelStep {
                    baked: b as u32,
                    function: self.command(a as usize).function,
                    region: ri as u32,
                    row_at,
                    row_len,
                    work: planned.work[a as usize],
                });
            }
            let mk_units: Vec<MkUnit> = plan.units[units.start as usize..units.end as usize]
                .iter()
                .map(|u| MkUnit {
                    first: step_at + u.steps.start - first_step,
                    end: step_at + u.steps.end - first_step,
                })
                .collect();
            let required = (planned.required.iter())
                .filter(|u| units.contains(&u.0))
                .count() as u32;
            regions.push(MkRegion {
                opens: self.baked[first_step as usize] as u32,
                required,
                table_at: region.table_at,
                block_len: region.block_len,
                units: baked(mk_units),
                runs: baked(mk_runs),
            });
        }
        let mut s = String::from(
            "// Generated by `megakernel_bake.rs`: every launch the decode forward of one tape may \
             play, one kernel each.\n",
        );
        for (k, (l, ty)) in self.loads.iter().zip(&self.load_types).enumerate() {
            let _ = writeln!(
                s,
                "constant {} MK_LOAD_{k} [[function_constant({})]];",
                msl_type(*ty),
                l.index.get()
            );
        }
        let mut heads: Vec<ConstSlot> = (planned.work.iter())
            .filter_map(|w| w.heads.map(|h| h.slot))
            .collect();
        heads.sort();
        heads.dedup();
        for slot in heads {
            let _ = writeln!(
                s,
                "constant uint MK_H{} [[function_constant({})]];",
                slot.get() - MK_FC_HEADS.get(),
                slot.get()
            );
        }
        s.push_str(&splits);
        let calibrations = self.calibrations(&mut kernels, &mut s);
        for (n, p) in self.policies.iter().enumerate() {
            let _ = write!(s, "struct MkC{n} {{\n{p}}};\n");
        }
        s.push_str(&kernels);
        let library = String::leak(format!("megakernel_{:016x}", fnv1a(s.as_bytes())));
        let table_len = (self.regions.iter())
            .map(|r| r.instances * r.block_len)
            .sum();
        Ok(MegakernelTape {
            library,
            source: String::leak(s),
            // Compiled by `compile_metallib`; the emission includes its bytes.
            metallib: &[],
            regions: baked(regions),
            steps: baked(steps),
            calibrations: baked(calibrations),
            table_len,
            load_constants: baked(self.loads),
        })
    }

    /// Every calibration's kernels ([`MkCalibration`]): its policy and the function constants it
    /// reads into `decls`, its `lane` and `native` kernels into `kernels`.
    fn calibrations(&self, kernels: &mut String, decls: &mut String) -> Vec<MkCalibration> {
        let calibrations = &self.planned.calibrations;
        if !calibrations.is_empty() {
            let _ = writeln!(
                decls,
                "constant uint MK_CAL_N [[function_constant({})]];",
                MK_FC_CAL.get()
            );
        }
        let mut out = Vec::with_capacity(calibrations.len());
        for (j, c) in calibrations.iter().enumerate() {
            let _ = writeln!(decls, "struct MkCal{j} {{");
            for k in c.constants {
                let given = c.fixed.iter().find(|v| v.index == k.slot.get());
                let value = match k.slot {
                    slot if slot == c.k_slot => literal(k.ty, c.k),
                    slot if slot == c.n_slot => format!("{}(MK_CAL_N)", msl_type(k.ty)),
                    _ => literal(k.ty, given.map_or(0, |v| v.bits)),
                };
                let set = k.slot == c.k_slot || k.slot == c.n_slot || given.is_some();
                let _ = writeln!(
                    decls,
                    "  static METAL_FUNC {} {}() {{ return {value}; }}\n  \
                     static METAL_FUNC bool has_{}() {{ return {set}; }}",
                    msl_type(k.ty),
                    k.name,
                    k.name,
                );
            }
            let _ = writeln!(decls, "}};");
            let call = replace_word(c.call, "MK_C", &format!("MkCal{j}"));
            let (tx, ty, tz) = c.tpg;
            // One stream: the calibration's rows along the step's rows axis.
            let mut grid = ["1u".to_string(), "1u".to_string(), "1u".to_string()];
            grid[c.rows_axis] = format!("MK_CAL_N / {}u", c.rows);
            let [gx, gy, gz] = grid;
            let step = |k: &str| {
                format!(
                    "const MkStep s = {{mk_a, uint3({gx}, {gy}, {gz}), uint3({tx}u, {ty}u, \
                     {tz}u), {k}, {}u, {}u}};",
                    c.stride, c.tg_bytes
                )
            };
            let after = match c.tg_bytes {
                0 => "",
                _ => "\n    threadgroup_barrier(mem_flags::mem_threadgroup);",
            };
            let tg_memory = c.tg_memory.max(16).next_multiple_of(16);
            let head = |name: String| {
                format!(
                    "[[kernel, max_total_threads_per_threadgroup(1024)]] void {name}(\n    \
                     constant ulong* mk_a [[buffer(0)]],\n    \
                     constant uint& mk_cal_k [[buffer(1)]],\n    \
                     uint mk_t [[thread_index_in_threadgroup]],\n    \
                     uint mk_idx [[threadgroup_position_in_grid]],\n    \
                     uint mk_p [[threadgroups_per_grid]]) {{\n  \
                     threadgroup uchar mk_tgm[{tg_memory}] __attribute__((aligned(16)));\n"
                )
            };
            let (lane, native) = (format!("mk_cal{j}"), format!("mk_cal{j}_native"));
            let _ = write!(
                kernels,
                "{}  {}\n  const uint items = (MK_CAL_N / {}u + mk_cal_k - 1u) / mk_cal_k;\n  \
                 for (uint it = mk_idx; it < items; it += mk_p) {{\n    \
                 {call}(s, mk_lane(s, it, mk_t), mk_tgm);{after}\n  }}\n}}\n\
                 {}  {}\n  {call}(s, mk_lane(s, mk_idx, mk_t), mk_tgm);\n}}\n",
                head(lane.clone()),
                step("mk_cal_k"),
                c.rows,
                head(native.clone()),
                step("1u"),
            );
            out.push(MkCalibration {
                lane: String::leak(lane),
                native: String::leak(native),
                rows: c.rows,
                k: c.k,
                bits: c.bits,
                group: c.group,
                scale_bytes: c.scale_bytes,
                bindings: c.bindings,
                act_bytes: 4,
                tpg: c.tpg,
                widest: c.widest,
            });
        }
        out
    }

    /// The unit opened by admitted step `a`, as its run's kernel plays it at `placement`
    /// (`follows`: a pinned unit after a unit of its run it waits on, on its lane).
    fn unit(
        &mut self,
        a: usize,
        placement: Place,
        follows: bool,
        out: &mut String,
    ) -> Result<(), BakeDefect> {
        let ind = "  ";
        let planned = self.planned;
        let unit = &planned.plan.units[self.unit_of[a]];
        let members = unit.steps.start as usize..unit.steps.end as usize;
        match placement {
            Place::Spread { run, ord } => {
                let s = self.step_decl(a, PerItem::Split { run, ord })?;
                let _ = writeln!(
                    out,
                    "{ind}{{ // {}\n{ind}  {}\n{ind}  for (uint it = mk_first(mk_idx, mk_p, \
                     MK_R{run}_C{ord}); it < {}; it += mk_p) {{\n{ind}    {}(s, \
                     mk_lane(s, it, mk_t), mk_tgm);{}\n{ind}  }}\n{ind}}}",
                    s.what,
                    s.decl,
                    s.items,
                    s.call,
                    if s.tg_bytes > 0 {
                        format!("\n{ind}    threadgroup_barrier(mem_flags::mem_threadgroup);")
                    } else {
                        String::new()
                    },
                );
            }
            // One lane plays a pinned unit; a copied one runs on every lane, each lane reading
            // its own copy's writes after it.
            Place::Pinned { .. } | Place::Everywhere => {
                let _ = match placement {
                    Place::Pinned { run, group } => {
                        writeln!(out, "{ind}if (mk_idx == MK_R{run}_L{group}) {{")
                    }
                    _ => writeln!(out, "{ind}{{ // on every threadgroup"),
                };
                let mut last_tg = 0;
                let members: Vec<usize> = members.collect();
                let mut j = 0;
                while j < members.len() {
                    if j > 0 || follows {
                        let _ = writeln!(
                            out,
                            "{ind}  threadgroup_barrier(mem_flags::mem_device | \
                             mem_flags::mem_threadgroup);"
                        );
                    }
                    let group = self.coissued(&members[j..]);
                    let m = members[j];
                    j += group.len();
                    if group.len() > 1 {
                        last_tg = self.coissue(&group, ind, out)?;
                        continue;
                    }
                    let s = self.step_decl(m, PerItem::Widest)?;
                    if s.items == "1u" {
                        let _ = writeln!(
                            out,
                            "{ind}  {{ // {}\n{ind}    {}\n{ind}    {}(s, mk_lane(s, 0u, mk_t), mk_tgm);\n\
                             {ind}  }}",
                            s.what, s.decl, s.call,
                        );
                        last_tg = s.tg_bytes;
                        continue;
                    }
                    // A whole step: this threadgroup plays every item.
                    let _ = writeln!(
                        out,
                        "{ind}  {{ // {}\n{ind}    {}\n{ind}    for (uint it = 0u; it < {}; ++it) {{\n\
                         {ind}      {}(s, mk_lane(s, it, mk_t), mk_tgm);{}\n{ind}    }}\n{ind}  }}",
                        s.what,
                        s.decl,
                        s.items,
                        s.call,
                        if s.tg_bytes > 0 {
                            format!("\n{ind}      threadgroup_barrier(mem_flags::mem_threadgroup);")
                        } else {
                            String::new()
                        },
                    );
                    last_tg = 0;
                }
                if placement == Place::Everywhere {
                    let _ = writeln!(
                        out,
                        "{ind}  threadgroup_barrier(mem_flags::mem_device | \
                         mem_flags::mem_threadgroup);"
                    );
                } else if last_tg > 0 {
                    let _ = writeln!(
                        out,
                        "{ind}  threadgroup_barrier(mem_flags::mem_threadgroup);"
                    );
                }
                let _ = writeln!(out, "{ind}}}");
            }
        }
        Ok(())
    }

    /// The members of a pinned unit, from `rest[0]` on, that play side by side in ONE pass of the
    /// threadgroup: consecutive steps of one adapter with the same constants and grid (none a load
    /// constant), each whole in one item, none touching what an earlier one writes nor writing what
    /// it reads — member `j` on threads `[j·W, (j+1)·W)` of the threadgroup (`W` = a member's
    /// threads), its own share of threadgroup memory. The body is ONE call, so every member runs
    /// every barrier of it. A TurboQuant K and V compress of a one-KV-head layer play at once.
    fn coissued(&self, rest: &[usize]) -> Vec<usize> {
        let first = self.command(rest[0]);
        // A grid the device's heads size plays alone.
        if self.planned.work[rest[0]].heads.is_some() {
            return vec![rest[0]];
        }
        let (constants, grid) = played(first);
        let tpg = first.dispatch.threads_per_threadgroup;
        let vtgs = grid.0 * grid.1 * grid.2;
        let Ok(g) = self.geometry(rest[0]) else {
            return vec![rest[0]];
        };
        let adapter = MkAdapter::of(first.library, first.function).expect("playable");
        let (stride, tg_bytes) = (g.vtg_stride, adapter.tg_bytes.0);
        let loaded = |m: usize| self.loads.iter().any(|l| l.baked as usize == self.baked[m]);
        let flows = &self.planned.flows;
        let mut group = vec![rest[0]];
        if loaded(rest[0]) {
            return group;
        }
        for &m in &rest[1..] {
            let cmd = self.command(m);
            let k = group.len() as u32 + 1;
            let alike = (cmd.library, cmd.function) == (first.library, first.function)
                && played(cmd) == (constants.clone(), grid)
                && cmd.dispatch.threads_per_threadgroup == tpg;
            let fits = k * vtgs * stride <= MK_THREADS && (k * vtgs + 1) * tg_bytes <= MK_TG_MEMORY;
            let apart = group.iter().all(|&g| {
                let (e, l) = (&flows[g], &flows[m]);
                !l.reads
                    .iter()
                    .chain(&l.writes)
                    .any(|x| e.writes.contains(x))
                    && !l.writes.iter().any(|x| e.reads.contains(x))
            });
            if !(alike && fits && apart) || loaded(m) {
                break;
            }
            group.push(m);
        }
        group
    }

    /// A co-issued group ([`Gen::coissued`]) as the kernel spells it; returns its threadgroup
    /// bytes per virtual threadgroup.
    fn coissue(&mut self, group: &[usize], ind: &str, out: &mut String) -> Result<u32, BakeDefect> {
        let (_, grid) = played(self.command(group[0]));
        let vtgs = grid.0 * grid.1 * grid.2;
        let mut texts = Vec::with_capacity(group.len());
        for &m in group {
            texts.push(self.step_decl(m, PerItem::Coissued(vtgs))?);
        }
        let s0 = &texts[0];
        let width = vtgs * s0.stride;
        let share = vtgs * s0.tg_bytes;
        let k = group.len() as u32;
        self.coissue_tg = self.coissue_tg.max((k * vtgs + 1) * s0.tg_bytes);
        let what: Vec<&str> = texts.iter().map(|t| t.what.as_str()).collect();
        let _ = writeln!(out, "{ind}  {{ // {}", what.join(" || "));
        for (j, t) in texts.iter().enumerate() {
            let decl = t
                .decl
                .replacen("const MkStep s =", &format!("const MkStep s{j} ="), 1);
            let _ = writeln!(out, "{ind}    {decl}");
        }
        let pick = (1..k).rev().fold(format!("s{}", k - 1), |e, j| {
            format!("mk_j == {}u ? s{} : {e}", j - 1, j - 1)
        });
        let _ = writeln!(
            out,
            "{ind}    const uint mk_j = min(mk_t / {width}u, {}u);\n\
             {ind}    const MkStep s = {pick};\n\
             {ind}    {}(s, mk_lane(s, 0u, mk_t - mk_j * {width}u), mk_tgm + mk_j * {share}u);\n\
             {ind}  }}",
            k - 1,
            s0.call,
        );
        Ok(s0.tg_bytes)
    }

    /// Admitted step `a` as its run's kernel spells it — its address row at its offset in the
    /// block the launch binds, whatever the iteration — its items as `per_item` says.
    fn step_decl(&mut self, a: usize, per_item: PerItem) -> Result<StepText, BakeDefect> {
        let b = self.baked[a];
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        let g = match per_item {
            PerItem::Coissued(k) => {
                step_geometry(cmd, adapter, Some(k)).map_err(|e| defect(cmd.function, e))?
            }
            PerItem::Widest | PerItem::Split { .. } => self.geometry(a)?,
        };
        let (_, row_at, _) = self.rows[b].expect("a played command has rows");
        let row = format!("mk_a + {row_at}u");
        let (_, grid) = played(cmd);
        let axis = |ax: MScaleAxis, v: u32| {
            self.load(b, MkLoadSource::Threadgroups(ax))
                .unwrap_or_else(|| format!("{v}u"))
        };
        let (gx, gy, gz) = (
            axis(MScaleAxis::X, grid.0),
            axis(MScaleAxis::Y, grid.1),
            axis(MScaleAxis::Z, grid.2),
        );
        let load_sized = [&gx, &gy, &gz].iter().any(|v| v.starts_with("MK_LOAD_"));
        // A decode attention's heads grouped per virtual threadgroup, as the load picks them.
        let heads = self.heads(a);
        let gy = match &heads {
            Some(h) => format!("({gy} / {h})"),
            None => gy,
        };
        // A spread step's virtual threadgroups per item are its run's split (a function constant);
        // its items follow from them, from its grid where the load sizes it.
        let (k, items) = match per_item {
            PerItem::Split { run, ord } => {
                let k = format!("MK_R{run}_K{ord}");
                let items = match load_sized || heads.is_some() {
                    true => format!("mk_items(s.grid, {k})"),
                    false => format!("({}u + {k} - 1u) / {k}", grid.0 * grid.1 * grid.2),
                };
                (k, items)
            }
            _ if load_sized || heads.is_some() => (
                format!("{}u", g.vtgs_per_item),
                format!("mk_items(s.grid, {}u)", g.vtgs_per_item),
            ),
            _ => (
                format!("{}u", g.vtgs_per_item),
                format!("{}u", g.items.0.get()),
            ),
        };
        let t = cmd.dispatch.threads_per_threadgroup;
        let decl = format!(
            "const MkStep s = {{{row}, uint3({gx}, {gy}, {gz}), uint3({}u, {}u, {}u), {k}, {}u, \
             {}u}};",
            t.0, t.1, t.2, g.vtg_stride, adapter.tg_bytes.0
        );
        let policy = self.policy(a)?;
        Ok(StepText {
            what: format!("{} (baked {b})", cmd.function),
            decl,
            items,
            call: replace_word(adapter.call, "MK_C", &policy),
            tg_bytes: adapter.tg_bytes.0,
            stride: g.vtg_stride,
        })
    }
}

/// Where a unit runs, as its run's kernel spells it: spread unit `ord` of run `run` (its split
/// the run's function constants), the threadgroup of the run's lane group `group`, or every
/// threadgroup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Place {
    Spread { run: usize, ord: usize },
    Pinned { run: usize, group: u32 },
    Everywhere,
}

/// How many virtual threadgroups a step's items play, as its spelling says.
#[derive(Clone, Copy)]
enum PerItem {
    /// Its widest (a pinned unit's member).
    Widest,
    /// A co-issued member's whole grid in one item.
    Coissued(u32),
    /// Spread unit `ord` of run `run`: the run's split.
    Split { run: usize, ord: usize },
}

/// One step's spelling.
struct StepText {
    what: String,
    decl: String,
    items: String,
    call: String,
    tg_bytes: u32,
    stride: u32,
}

fn msl_type(ty: ConstantType) -> &'static str {
    match ty {
        ConstantType::UInt => "uint",
        ConstantType::Int => "int",
        ConstantType::Float => "float",
        ConstantType::Bool => "bool",
    }
}

fn literal(ty: ConstantType, bits: u32) -> String {
    match ty {
        ConstantType::UInt => format!("{bits}u"),
        ConstantType::Int => format!("as_type<int>({bits:#010x}u)"),
        ConstantType::Float => format!("as_type<float>({bits:#010x}u)"),
        ConstantType::Bool => (bits != 0).to_string(),
    }
}

/// `text` with every whole-word `word` replaced by `by`.
fn replace_word(text: &str, word: &str, by: &str) -> String {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(word) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + word.len()..].chars().next();
        out.push_str(&rest[..at]);
        if before.is_some_and(ident) || after.is_some_and(ident) {
            out.push_str(word);
        } else {
            out.push_str(by);
        }
        rest = &rest[at + word.len()..];
    }
    out.push_str(rest);
    out
}

/// Compiles `mk`'s library at build time into `dir` as every shader is compiled — the build
/// machine's `xcrun metal` with [`MSL_TO_AIR`] then [`AIR_TO_METALLIB`], its toolchain's language
/// standard — so its bodies compile exactly as their dispatch kernels; returns the metallib's
/// path for the emission to include. Named by a hash of everything the compile reads — the
/// toolchain's version, the flags, the adapters' bodies and the generated kernel — so a change to
/// any of them compiles afresh; written through files of this compile's own, so models expanding
/// in parallel never read a half-written one.
pub fn compile_metallib(mk: &MegakernelTape, dir: &Path) -> Result<PathBuf, BakeDefect> {
    let fail = |what: String| BakeDefect(format!("megakernel `{}`: {what}", mk.library));
    std::fs::create_dir_all(dir).map_err(|e| fail(format!("create {}: {e}", dir.display())))?;
    let inputs = format!(
        "{}\n{}\n{}\n{MK_BODIES}\n{}",
        toolchain()?,
        MSL_TO_AIR.join(" "),
        AIR_TO_METALLIB.join(" "),
        mk.source
    );
    let metallib = dir.join(format!(
        "{}-{:016x}.metallib",
        mk.library,
        fnv1a(inputs.as_bytes())
    ));
    if metallib.exists() {
        return Ok(metallib);
    }
    static COMPILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let compile = COMPILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let own = |ext: &str| {
        dir.join(format!(
            "{}.{}.{compile}.{ext}",
            mk.library,
            std::process::id()
        ))
    };
    let (source, air, lib) = (own("metal"), own("air"), own("metallib"));
    std::fs::write(&source, format!("{MK_BODIES}\n{}", mk.source))
        .map_err(|e| fail(format!("write {}: {e}", source.display())))?;
    let run = |args: &[&str], from: &Path, to: &Path, extra: &[&str]| {
        let out = Command::new("xcrun")
            .args(args)
            .args(extra)
            .arg(from)
            .arg("-o")
            .arg(to)
            .output()
            .map_err(|e| fail(format!("spawn xcrun: {e}")))?;
        match out.status.success() {
            true => Ok(()),
            false => Err(fail(format!(
                "`xcrun {}` failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr)
            ))),
        }
    };
    run(MSL_TO_AIR, &source, &air, &["-c"])?;
    run(AIR_TO_METALLIB, &air, &lib, &[])?;
    std::fs::rename(&lib, &metallib)
        .map_err(|e| fail(format!("rename to {}: {e}", metallib.display())))?;
    for f in [&source, &air] {
        std::fs::remove_file(f).map_err(|e| fail(format!("remove {}: {e}", f.display())))?;
    }
    Ok(metallib)
}

/// The build machine's Metal toolchain, as `xcrun metal --version` names it (asked once).
fn toolchain() -> Result<&'static str, BakeDefect> {
    static VERSION: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    let version = VERSION.get_or_init(|| {
        let out = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "--version"])
            .output()
            .map_err(|e| format!("spawn xcrun: {e}"))?;
        match out.status.success() {
            true => Ok(String::from_utf8_lossy(&out.stdout).into_owned()),
            false => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
        }
    });
    version
        .as_deref()
        .map_err(|e| BakeDefect(format!("megakernel: the Metal toolchain's version: {e}")))
}

/// FNV-1a, 64-bit: a deterministic content name (the bake must be reproducible).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
