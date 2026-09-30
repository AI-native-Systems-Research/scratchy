// SPDX-License-Identifier: Apache-2.0
//! THE DECODE MEGAKERNEL, COMPILED — per bucket-1 tape, at expansion.
//!
//! The WHOLE decode forward becomes ONE kernel. Every command the planned context
//! ([`GateCtx::decode_one`]) admits is fed to the SHARED plan ([`plan`]) with the dataflow of the
//! step row it came from ([`RowAccess`], the barrier walk's hazard signature named): all commands
//! of a row carry the row's whole access set, so a multi-command row is ordered by its own
//! writes. Every admitted command must be playable — an adapter ([`MkAdapter::of`]) and every
//! load-time scalar modeled ([`MkLoadConstant`]) — or the bake fails, naming each command that
//! stopped it; a kernel is never split.
//!
//! The plan is SCHEDULED by the shared pass ([`schedule`]) and COMPILED here into MSL: one
//! `[[kernel]]` whose body is the tape's steps in order as straight-line adapter calls — every
//! constant a literal of the step's policy struct, every geometry a literal, the work split a
//! compile-time round-robin over the `MK_P` persistent threadgroups, a grid barrier between
//! phases and nothing else — with the tape's layer loops kept ROLLED exactly as the tape keeps
//! them (a phase break at every iteration start makes every iteration schedule alike, which the
//! bake checks). Only binding ADDRESSES and the load's scalars are runtime data: a step reads its
//! addresses from the address table at a position fixed here (its row, plus the iteration times
//! the row length), the scalars are function constants.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use std::num::NonZeroU32;

use scratchy_subtile::megakernel_plan::{
    Items, MegakernelPlan, Placement, Schedule, StepFlow, UnitIx, plan, schedule,
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
    Binding, BindingMask, CapPatch, CommandOrigin, CommandSpan, GateCtx, GatedCommand, KernelId,
    LoweredCommand, LoweredMetalTape, MK_FC_LOAD, MK_TG_MEMORY, MK_THREADS, MScaleAxis,
    MegakernelError, MegakernelTape, MkAdapter, MkGeometry, MkKernelStep, MkLoadConstant,
    MkLoadSource, PatchTarget, RuntimeBindingKind, TapeLoop, baked, mk_geometry,
};
use scratchy_target_metal::tape::step::{MetalLoc, MetalStep, MetalStepTape, RowAccess, StepRow};

use crate::static_tape::BakeDefect;

/// An elementwise step ([`MkAdapter::tail`]: a residual add, a scalar scale) of at most this many
/// items may be played whole by the threadgroup that ran what it waits on, instead of spreading
/// after a grid barrier: a few passes of one threadgroup cost less than the barrier (~3 us in the
/// forward). A heavier step (a matvec, a gemm) is spread, however few its items.
const MK_WHOLE_ITEMS: u32 = 4;

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

/// A decode attention's (fp16 KV) query heads per virtual threadgroup, as the device's pick for
/// TurboQuant decode makes it ([`TqDecodeHeads::for_group`]) at the launch's persistent
/// threadgroups — the GPU's cores, `MK_P`, known at load — so its items fill them instead of
/// spilling into another round. The rule's candidates, most heads first: `(q / h, h)`, `h` taken
/// when `q / h ≥ ⌊4·MK_P / 5⌋`, else one head. `None`: not a decode attention, or one head on
/// every GPU. The body computes each head exactly as it does one head per threadgroup.
fn heads_rule(cmd: &LoweredCommand) -> Option<Vec<(u32, u32)>> {
    if cmd.kernel != KernelId::AttentionViaCache {
        return None;
    }
    let at = |slot: ConstSlot| {
        let c = cmd.constants.iter().find(|v| v.index == slot.get());
        c.map(|v| v.bits)
    };
    use AttentionViaCacheConstants as A;
    let q = at(A::NUM_Q_HEADS)?;
    let candidates = TqDecodeHeads::candidates(
        HeadDim(at(A::HEAD_DIM)?),
        NumQHeads(q),
        NumKvHeads(at(A::NUM_KV_HEADS)?),
    );
    let mut rule: Vec<(u32, u32)> = candidates
        .map(|h| (q / h.get(), h.get()))
        .filter(|&(_, h)| h > 1)
        .collect();
    rule.reverse();
    (!rule.is_empty()).then_some(rule)
}

/// [`heads_rule`] as MSL: a function of `MK_P` the pipeline's specialization folds.
fn heads_msl(rule: &[(u32, u32)]) -> String {
    rule.iter().rev().fold("1u".into(), |e, (threadgroups, h)| {
        format!("({threadgroups}u >= MK_P * 4u / 5u ? {h}u : {e})")
    })
}

/// A command as the megakernel plays it: its constants and grid, one head per virtual threadgroup
/// (a decode attention's grouping, [`heads_rule`], is spelled in `MK_P` where the step is).
fn played(cmd: &LoweredCommand) -> (Vec<ConstantValue>, (u32, u32, u32)) {
    (
        cmd.constants.to_vec(),
        cmd.dispatch.threadgroups_at(NumTokens(1), 1),
    )
}

/// A step's geometry: its adapter's widest packing — what the plan and schedule see — or items of
/// `vtgs_per_item` virtual threadgroups.
fn step_geometry(
    cmd: &LoweredCommand,
    adapter: &MkAdapter,
    vtgs_per_item: Option<u32>,
) -> Result<MkGeometry, MegakernelError> {
    let (constants, grid) = played(cmd);
    let tpg = cmd.dispatch.threads_per_threadgroup;
    let item_threads = adapter.item_threads_for(&constants);
    let widest = mk_geometry(grid, tpg, adapter.tg_bytes, item_threads)?;
    match vtgs_per_item {
        Some(k) => mk_geometry(grid, tpg, adapter.tg_bytes, k * widest.vtg_stride),
        None => Ok(widest),
    }
}

/// A spread step as its phase's packing reads it: its virtual threadgroups (MSL: a literal, or
/// with its decode attention's heads grouped, [`heads_rule`]), its declared items (the plan's),
/// and — `fit`: `None` for a grid the load sizes, played at its widest — its items' virtual
/// threadgroups at its adapter's width and across the whole persistent threadgroup.
#[derive(Clone, PartialEq, Eq, Debug)]
struct SpreadIn {
    vtgs: String,
    declared: u32,
    fit: Option<(u32, u32)>,
}

/// One phase's work split, as every phase of its shape spells it: its spread steps in order, its
/// pinned units that wait on nothing in the phase (`free`: each keeps a slot of the last round),
/// and its pinned lane groups (a unit and the in-phase units it waits on share a lane).
#[derive(Clone, PartialEq, Eq, Debug)]
struct PhaseShape {
    spread: Vec<SpreadIn>,
    free: u32,
    groups: u32,
}

/// Where a unit runs, as the kernel spells it: spread step `ord` of phase shape `shape`, the
/// lane of pinned group `group` of it, or on every threadgroup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Place {
    Spread { shape: usize, ord: usize },
    Pinned { shape: usize, group: u32 },
    Everywhere,
}

/// A phase whose items, at their adapters' widths, fill at most this many rounds is bound by its
/// rounds, not by its streams: its items widen up to the whole persistent threadgroup, so every
/// lane plays one even share (Qwen3-0.6B's 128-tile down and output projections: -0.12 ms/token on
/// a base M5; phases of up to 8 rounds measured the same there).
const MK_ROUND_BOUND_PHASE: u32 = 4;

/// Places each phase's units ([`Place`]) and collects the phase shapes they spell. The split
/// itself is the shape's [`PhaseShape::msl`]: a function of the launch's `MK_P`.
fn pack_phases(
    sched: &Schedule,
    plan: &MegakernelPlan<MkLoc>,
    spread_in: &[SpreadIn],
) -> (Vec<Place>, Vec<PhaseShape>) {
    let mut places = vec![Place::Everywhere; plan.units.len()];
    let mut shapes: Vec<PhaseShape> = Vec::new();
    for phase in &sched.phases {
        let units: Vec<usize> = phase.clone().map(|u| u as usize).collect();
        let in_phase = |w: &UnitIx| phase.contains(&w.0);
        let first = |u: usize| plan.units[u].steps.start as usize;
        let spread: Vec<usize> = units
            .iter()
            .copied()
            .filter(|&u| matches!(sched.placements[u], Placement::Spread { .. }))
            .collect();
        let pinned: Vec<usize> = units
            .iter()
            .copied()
            .filter(|&u| matches!(sched.placements[u], Placement::Pinned(_)))
            .collect();
        let free = pinned
            .iter()
            .filter(|&&u| !plan.units[u].waits.iter().any(in_phase))
            .count() as u32;
        // The pinned units in lane groups: a unit and the in-phase units it waits on share a lane.
        let mut group: Vec<usize> = (0..pinned.len()).collect();
        let root = |g: &[usize], mut i: usize| {
            while g[i] != i {
                i = g[i];
            }
            i
        };
        for (i, &u) in pinned.iter().enumerate() {
            for w in plan.units[u].waits.iter().filter(|w| in_phase(w)) {
                if let Some(j) = pinned.iter().position(|&p| p == w.0 as usize) {
                    let (a, b) = (root(&group, i), root(&group, j));
                    group[a.max(b)] = a.min(b);
                }
            }
        }
        let mut ordinal: Vec<Option<u32>> = vec![None; pinned.len()];
        let mut groups = 0u32;
        let group_of: Vec<u32> = (0..pinned.len())
            .map(|i| {
                let r = root(&group, i);
                *ordinal[r].get_or_insert_with(|| {
                    groups += 1;
                    groups - 1
                })
            })
            .collect();
        let shape = PhaseShape {
            spread: spread
                .iter()
                .map(|&u| spread_in[first(u)].clone())
                .collect(),
            free,
            groups,
        };
        let shape = match shapes.iter().position(|s| *s == shape) {
            Some(s) => s,
            None => {
                shapes.push(shape);
                shapes.len() - 1
            }
        };
        for (ord, &u) in spread.iter().enumerate() {
            places[u] = Place::Spread { shape, ord };
        }
        for (i, &u) in pinned.iter().enumerate() {
            places[u] = Place::Pinned {
                shape,
                group: group_of[i],
            };
        }
    }
    (places, shapes)
}

impl PhaseShape {
    /// The split of every phase of shape `s` among the launch's `mk_p` participants — the
    /// threadgroups that checked in — as functions of `mk_p` the phase inlines. The phase keeps the
    /// fewest rounds its items allow — at their adapters' widths, or across the whole threadgroup
    /// for a phase of few rounds ([`MK_ROUND_BOUND_PHASE`]) — a slot of the last round left to each
    /// free pinned unit; the slots those rounds hold beyond the items go to the steps in proportion
    /// to their virtual threadgroups, each step's items then even (`MK_S{s}_K{i}` virtual
    /// threadgroups per item, `MK_S{s}_N{i}` items, starting at cursor `MK_S{s}_C{i}`); pinned group
    /// `g` plays on the participant after the phase's last spread item (`MK_S{s}_L{g}`), the least
    /// loaded.
    fn msl(&self, s: usize, out: &mut String) {
        let p = |name: &str| format!("MK_S{s}_{name}(mk_p)");
        let mut def = |ty: &str, name: &str, expr: &str| {
            let _ = writeln!(
                out,
                "METAL_FUNC {ty} MK_S{s}_{name}(uint mk_p) {{ return {expr}; }}"
            );
        };
        let (min, max) = (
            |a: &str, b: &str| format!("({a} < {b} ? {a} : {b})"),
            |a: &str, b: &str| format!("({a} > {b} ? {a} : {b})"),
        );
        let declared: u32 = self.spread.iter().map(|x| x.declared).sum();
        def(
            "bool",
            "RB",
            &format!("({declared}u + mk_p - 1u) / mk_p <= {MK_ROUND_BOUND_PHASE}u"),
        );
        let mut n0 = Vec::with_capacity(self.spread.len());
        for (i, x) in self.spread.iter().enumerate() {
            def("uint", &format!("V{i}"), &x.vtgs);
            match x.fit {
                Some((widest, whole)) => {
                    def(
                        "uint",
                        &format!("KA{i}"),
                        &format!("{} ? {whole}u : {widest}u", p("RB")),
                    );
                    let (v, ka) = (p(&format!("V{i}")), p(&format!("KA{i}")));
                    def(
                        "uint",
                        &format!("NA{i}"),
                        &format!("({v} + {ka} - 1u) / {ka}"),
                    );
                    n0.push(p(&format!("NA{i}")));
                }
                None => n0.push(format!("{}u", x.declared)),
            }
        }
        let sum = |v: &[String]| match v {
            [] => "0u".to_string(),
            _ => v.join(" + "),
        };
        let fit_v: Vec<String> = (0..self.spread.len())
            .filter(|&i| self.spread[i].fit.is_some())
            .map(|i| p(&format!("V{i}")))
            .collect();
        def("uint", "NT", &sum(&n0));
        def("uint", "RES", &min(&format!("{}u", self.free), "mk_p - 1u"));
        def(
            "uint",
            "R",
            &format!("({} + {} + mk_p - 1u) / mk_p", p("NT"), p("RES")),
        );
        def(
            "uint",
            "SLOTS",
            &max(&format!("{} * mk_p - {}", p("R"), p("RES")), &p("NT")),
        );
        def("uint", "VF", &sum(&fit_v));
        def("uint", "C0", "0u");
        for (i, x) in self.spread.iter().enumerate() {
            let (v, k) = (p(&format!("V{i}")), p(&format!("K{i}")));
            match x.fit {
                Some(_) => {
                    let share = format!(
                        "{} + ({} - {}) * {v} / {}",
                        p(&format!("NA{i}")),
                        p("SLOTS"),
                        p("NT"),
                        p("VF")
                    );
                    let items = min(&v, &share);
                    def(
                        "uint",
                        &format!("K{i}"),
                        &format!("({v} + {items} - 1u) / {items}"),
                    );
                    def("uint", &format!("N{i}"), &format!("({v} + {k} - 1u) / {k}"));
                }
                None => def("uint", &format!("N{i}"), &format!("{}u", x.declared)),
            }
            def(
                "uint",
                &format!("C{}", i + 1),
                &format!("{} + {}", p(&format!("C{i}")), p(&format!("N{i}"))),
            );
        }
        for g in 0..self.groups {
            def(
                "uint",
                &format!("L{g}"),
                &format!("({} + {g}u) % mk_p", p(&format!("C{}", self.spread.len()))),
            );
        }
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

/// Whether the ONE decode kernel plays `kernel` — EXHAUSTIVE over [`KernelId`], so a new kernel is
/// classed before any tape can emit it. The bake holds every admitted bucket-1 decode command to
/// it: a played kernel's symbol must have an adapter ([`MegakernelError::NoAdapter`]), and a decode
/// command whose kernel is classed [`NotAtDecode`] fails the bake
/// ([`MegakernelError::NotAtDecode`]: the class is wrong) — never a dispatch beside the kernel.
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

/// The planned decode forward: the admitted commands (expanded indices, in order), their
/// plan and its schedule.
struct Planned {
    admitted: Vec<usize>,
    plan: MegakernelPlan<MkLoc>,
    sched: Schedule,
    /// Per unit: where it runs ([`pack_phases`]).
    places: Vec<Place>,
    /// The phase shapes the places name.
    shapes: Vec<PhaseShape>,
    /// Per admitted step: what it reads and writes, as the plan ordered it.
    flows: Vec<StepFlow<MkLoc>>,
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

/// The megakernel of a bucket-1 decode tape: one [`MegakernelTape`] playing every command a
/// decode step admits — a command it cannot play fails the bake. `tape` is the capacity-0
/// bake, `patches` its load patches, `row_commands` the commands each step row became, `steps`
/// the rows' dataflow. A tape without a decode step ([`decodes`]) has none.
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
    // Every admitted command inside the ONE kernel: a command the kernel cannot play fails
    // the bake (each distinct cause named). The worker plays the kernel whether or not the
    // sequence holds an unrotated span block, so no gate may read that here.
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
            "the bucket-1 decode tape is not playable inside one \
             kernel: {}",
            stops.join("; ")
        )));
    }
    let kv = KvAliasing::of(&commands, &admitted);
    let mut flows = Vec::with_capacity(admitted.len());
    let mut spread_in: Vec<SpreadIn> = Vec::with_capacity(admitted.len());
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
        let load_sized = sources
            .iter()
            .any(|s| matches!(s, MkLoadSource::Threadgroups(_)));
        let geometry = |k| step_geometry(cmd, adapter, k).map_err(|e| defect(cmd.function, e));
        let g = geometry(None)?;
        let (_, grid) = played(cmd);
        let whole = geometry(Some(MK_THREADS / g.vtg_stride))?;
        let items = match load_sized {
            true => Items(g.items.0.max(NonZeroU32::MIN.saturating_add(1))),
            false => g.items,
        };
        // The plan sees one head per virtual threadgroup: the most items the step can have.
        let vtgs = match heads_rule(cmd) {
            Some(rule) => format!(
                "({}u * ({}u / {}) * {}u)",
                grid.0,
                grid.1,
                heads_msl(&rule),
                grid.2
            ),
            None => format!("{}u", grid.0 * grid.1 * grid.2),
        };
        spread_in.push(SpreadIn {
            vtgs,
            declared: items.0.get(),
            fit: (!load_sized).then_some((g.vtgs_per_item, whole.vtgs_per_item)),
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
        flows.push(StepFlow {
            reads,
            writes,
            items,
            phase_break: std::mem::take(&mut pending),
            whole: adapter.tail && !load_sized && items.0.get() <= MK_WHOLE_ITEMS,
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
    // The phase structure assumes no core count: every pinned unit is placed as if on a lane
    // of its own. The split within each phase is a function of the launch's `MK_P`.
    let sched = schedule(&plan, &flows, plan.units.len() as u32);
    let (places, shapes) = pack_phases(&sched, &plan, &spread_in);
    let planned = Planned {
        admitted,
        plan,
        sched,
        places,
        shapes,
        flows,
    };
    let generator = Gen::new(tape, patches, &commands, &origins, &planned, &kv)?;
    let mk = generator.kernel(&tree)?;
    eprintln!(
        "[m2-megakernel] {}: ONE kernel, all {} admitted commands \
         ({} baked steps), {} grid barriers per forward, {} load constants, {} bytes of MSL",
        mk.library,
        planned.admitted.len(),
        mk.steps.len(),
        mk.grid_barriers,
        mk.load_constants.len(),
        mk.source.len(),
    );
    Ok(vec![mk])
}

/// How a step runs, as the generated code spells it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StepKey {
    /// The step opens its unit, which runs there.
    unit: Option<Place>,
    /// The step opens a phase: a grid barrier stands before it (unless nothing ran before).
    phase: bool,
    /// A pinned unit whose waits include a unit of its own phase (on its lane).
    follows: bool,
}

/// The kernel generator of the decode tape.
struct Gen<'a> {
    commands: &'a [GatedCommand],
    planned: &'a Planned,
    kv: &'a KvAliasing,
    /// Per admitted step: its baked position, its schedule key, its unit.
    baked: Vec<usize>,
    key: Vec<StepKey>,
    unit_of: Vec<usize>,
    /// Per baked position: the admitted index of its first instance, and its address rows
    /// (`table_at`, `row_len`).
    first: Vec<Option<usize>>,
    rows: Vec<Option<(u32, u32)>>,
    table_len: u32,
    loads: Vec<MkLoadConstant>,
    load_types: Vec<ConstantType>,
    policies: Vec<String>,
    /// Threadgroup memory the widest co-issued group needs ([`Gen::coissued`]).
    coissue_tg: u32,
}

impl<'a> Gen<'a> {
    fn new(
        tape: &LoweredMetalTape,
        patches: &[CapPatch],
        commands: &'a [GatedCommand],
        origins: &[CommandOrigin],
        planned: &'a Planned,
        kv: &'a KvAliasing,
    ) -> Result<Self, BakeDefect> {
        let Planned {
            admitted,
            plan,
            sched,
            places,
            ..
        } = planned;
        let n = tape.commands.len();
        let mut unit_of = vec![0usize; admitted.len()];
        for (u, unit) in plan.units.iter().enumerate() {
            for s in unit.steps.clone() {
                unit_of[s as usize] = u;
            }
        }
        let phase_start: HashSet<u32> = sched.phases.iter().map(|p| p.start).collect();
        let phase_of = |u: usize| sched.phases.iter().position(|p| p.contains(&(u as u32)));
        let mut key = Vec::with_capacity(admitted.len());
        for (a, &u) in unit_of.iter().enumerate() {
            let unit = &plan.units[u];
            let opens = unit.steps.start as usize == a;
            let follows = matches!(sched.placements[u], Placement::Pinned(_))
                && unit
                    .waits
                    .iter()
                    .any(|w| phase_of(w.0 as usize) == phase_of(u));
            key.push(StepKey {
                unit: opens.then_some(places[u]),
                phase: opens && phase_start.contains(&(u as u32)),
                follows,
            });
        }
        let baked: Vec<usize> = admitted.iter().map(|&e| origins[e].baked).collect();
        let mut instances = vec![0u32; n];
        for &b in &baked {
            instances[b] += 1;
        }
        let mut first = vec![None; n];
        let mut rows: Vec<Option<(u32, u32)>> = vec![None; n];
        let mut table_len = 0u32;
        let mut loads: Vec<MkLoadConstant> = Vec::new();
        let mut load_types = Vec::new();
        for (a, &b) in baked.iter().enumerate() {
            if first[b].is_some() {
                continue;
            }
            first[b] = Some(a);
            let cmd = &commands[admitted[a]].command;
            let row_len = cmd
                .bindings
                .iter()
                .map(|x| u32::from(x.binding_index()) + 1);
            let row_len = row_len.max().unwrap_or(0);
            rows[b] = Some((table_len, row_len));
            table_len += instances[b] * row_len;
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
        // Every instance of a baked command runs as its first: the kernel spells the first once
        // inside the rolled loops.
        for (a, &b) in baked.iter().enumerate() {
            let f = first[b].expect("an admitted command has a first instance");
            if key[a] != key[f] {
                let symbol = commands[admitted[a]].command.function;
                return Err(BakeDefect(format!(
                    "megakernel: `{symbol}` (baked {b}) schedules differently in a later loop \
                     iteration"
                )));
            }
        }
        Ok(Self {
            commands,
            planned,
            kv,
            baked,
            key,
            unit_of,
            first,
            rows,
            table_len,
            loads,
            load_types,
            policies: Vec::new(),
            coissue_tg: 0,
        })
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

    /// The policy struct spelling the constants of admitted step `a` for its adapter (defined
    /// once per text): each a literal, or the load constant a slot is served by
    /// ([`MkLoadConstant`]); `has_…` = set.
    fn policy(&mut self, a: usize) -> Result<String, BakeDefect> {
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        let mut body = String::new();
        let (constants, _) = played(cmd);
        let heads = heads_rule(cmd).map(|rule| heads_msl(&rule));
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

    /// Admitted step `a`'s geometry at its adapter's width, or (`whole`) across the whole
    /// persistent threadgroup — the widest items a phase's split may give it.
    fn geometry(&self, a: usize, whole: bool) -> Result<MkGeometry, BakeDefect> {
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        let at = |k| step_geometry(cmd, adapter, k).map_err(|e| defect(cmd.function, e));
        let widest = at(None)?;
        match whole {
            true => at(Some(MK_THREADS / widest.vtg_stride)),
            false => Ok(widest),
        }
    }

    /// The whole kernel: the tape's nodes in order, loops rolled.
    fn kernel(mut self, tree: &[Node]) -> Result<MegakernelTape, BakeDefect> {
        let mut body = String::new();
        let mut at = Emit {
            loops: Vec::new(),
            prior: Prior::Nothing,
            sites: 0,
        };
        self.emit(tree, &mut at, &mut body)?;
        let mut tg_memory = self.coissue_tg.max(16);
        for a in 0..self.baked.len() {
            tg_memory = tg_memory.max(self.geometry(a, true)?.tg_memory);
        }
        let tg_memory = tg_memory.next_multiple_of(16);
        let mut s = String::from(
            "// Generated by `megakernel_bake.rs`: the decode forward of one tape as one kernel.\n",
        );
        for (k, (l, ty)) in self.loads.iter().zip(&self.load_types).enumerate() {
            let _ = writeln!(
                s,
                "constant {} MK_LOAD_{k} [[function_constant({})]];",
                msl_type(*ty),
                l.index.get()
            );
        }
        for (n, shape) in self.planned.shapes.iter().enumerate() {
            shape.msl(n, &mut s);
        }
        for (n, p) in self.policies.iter().enumerate() {
            let _ = write!(s, "struct MkC{n} {{\n{p}}};\n");
        }
        let _ = write!(
            s,
            "[[kernel, max_total_threads_per_threadgroup(1024)]] void mk_forward(\n    \
             constant ulong* mk_a [[buffer(0)]],\n    device MkSync* mk_sync [[buffer(1)]],\n    \
             uint mk_t [[thread_index_in_threadgroup]]) {{\n  \
             threadgroup uchar mk_tgm[{tg_memory}] __attribute__((aligned(16)));\n  \
             threadgroup uint mk_ok[3];\n  uint mk_idx, mk_p;\n  \
             if (!mk_check_in(mk_sync, mk_t, mk_ok, mk_idx, mk_p)) return;\n  \
             uint mk_gen = 0u;\n{body}}}\n"
        );
        let library = String::leak(format!("megakernel_{:016x}", fnv1a(s.as_bytes())));
        let mut steps = Vec::new();
        for (b, f) in self.first.iter().enumerate() {
            let (Some(a), Some((table_at, row_len))) = (*f, self.rows[b]) else {
                continue;
            };
            steps.push(MkKernelStep {
                baked: b as u32,
                function: self.command(a).function,
                table_at,
                row_len,
            });
        }
        let admitted = &self.planned.admitted;
        let grid_barriers = self.key.iter().filter(|k| k.phase).count() as u32 - 1;
        Ok(MegakernelTape {
            library,
            source: String::leak(s),
            // Compiled by `compile_metallib`; the emission includes its bytes.
            metallib: &[],
            kernel: "mk_forward",
            commands: CommandSpan {
                start: admitted[0] as u32,
                end: admitted[admitted.len() - 1] as u32 + 1,
            },
            steps: baked(steps),
            table_len: self.table_len,
            grid_barriers,
            load_constants: baked(self.loads),
        })
    }

    fn emit(&mut self, nodes: &[Node], at: &mut Emit, out: &mut String) -> Result<(), BakeDefect> {
        for node in nodes {
            let ind = "  ".repeat(at.loops.len() + 1);
            match node {
                Node::Cmd(b) => {
                    // Gated off at decode, or emitted with the unit it belongs to.
                    let Some(a) = self.first[*b] else { continue };
                    let Some(placement) = self.key[a].unit else {
                        continue;
                    };
                    if self.key[a].phase {
                        at.barrier(&ind, out);
                    }
                    self.unit(a, placement, &at.loops, &ind, out)?;
                    at.prior = Prior::Always;
                }
                Node::Loop { iters, body } => {
                    let counter = format!("mk_l{}", at.loops.len());
                    let _ = writeln!(
                        out,
                        "{ind}for (uint {counter} = 0u; {counter} < {iters}u; ++{counter}) {{"
                    );
                    let outer = std::mem::replace(&mut at.prior, Prior::Nothing);
                    at.prior = match outer.clone() {
                        Prior::Always => Prior::Always,
                        Prior::Nothing => Prior::Unless(vec![counter.clone()]),
                        Prior::Unless(mut cs) => {
                            cs.push(counter.clone());
                            Prior::Unless(cs)
                        }
                    };
                    at.loops.push((counter, *iters));
                    self.emit(body, at, out)?;
                    at.loops.pop();
                    let _ = writeln!(out, "{ind}}}");
                    // After a loop that ran a step, something always ran.
                    let ran = matches!(at.prior, Prior::Always);
                    at.prior = if ran { Prior::Always } else { outer };
                }
            }
        }
        Ok(())
    }

    /// The unit opened by admitted step `a`, at the loop nesting `loops`.
    fn unit(
        &mut self,
        a: usize,
        placement: Place,
        loops: &[(String, u32)],
        ind: &str,
        out: &mut String,
    ) -> Result<(), BakeDefect> {
        let planned = self.planned;
        let unit = &planned.plan.units[self.unit_of[a]];
        let members = unit.steps.start as usize..unit.steps.end as usize;
        // The iteration the steps run in: their instance among the baked command's.
        let inst = loops.split_first().map(|((c0, _), rest)| {
            rest.iter().fold(c0.clone(), |e, (c, iters)| {
                format!("({e}) * {iters}u + {c}")
            })
        });
        match placement {
            Place::Spread { shape, ord } => {
                let s = self.step_decl(a, inst.as_deref(), PerItem::Split { shape, ord })?;
                let _ = writeln!(
                    out,
                    "{ind}{{ // {}\n{ind}  {}\n{ind}  for (uint it = mk_first(mk_idx, mk_p, \
                     MK_S{shape}_C{ord}(mk_p)); it < {}; it += mk_p) {{\n{ind}    {}(s, \
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
                    Place::Pinned { shape, group } => {
                        writeln!(out, "{ind}if (mk_idx == MK_S{shape}_L{group}(mk_p)) {{")
                    }
                    _ => writeln!(out, "{ind}{{ // on every participant"),
                };
                let mut last_tg = 0;
                let members: Vec<usize> = members.collect();
                let mut j = 0;
                while j < members.len() {
                    if j > 0 || self.key[a].follows {
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
                        last_tg = self.coissue(&group, inst.as_deref(), ind, out)?;
                        continue;
                    }
                    let s = self.step_decl(m, inst.as_deref(), PerItem::Widest)?;
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
    /// every barrier of it. A TurboQuant K and V compress of a one-KV-head layer play at once
    /// (Gemma-3-1B: -0.22 ms/token on the base M5).
    fn coissued(&self, rest: &[usize]) -> Vec<usize> {
        let first = self.command(rest[0]);
        // A grid the launch's `MK_P` sizes plays alone.
        if heads_rule(first).is_some() {
            return vec![rest[0]];
        }
        let (constants, grid) = played(first);
        let tpg = first.dispatch.threads_per_threadgroup;
        let vtgs = grid.0 * grid.1 * grid.2;
        let Ok(g) = self.geometry(rest[0], false) else {
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
    fn coissue(
        &mut self,
        group: &[usize],
        inst: Option<&str>,
        ind: &str,
        out: &mut String,
    ) -> Result<u32, BakeDefect> {
        let (_, grid) = played(self.command(group[0]));
        let vtgs = grid.0 * grid.1 * grid.2;
        let mut texts = Vec::with_capacity(group.len());
        for &m in group {
            texts.push(self.step_decl(m, inst, PerItem::Coissued(vtgs))?);
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

    /// Admitted step `a` as the kernel spells it, in the iteration `inst` (`None`: outside every
    /// loop), its items as `per_item` says.
    fn step_decl(
        &mut self,
        a: usize,
        inst: Option<&str>,
        per_item: PerItem,
    ) -> Result<StepText, BakeDefect> {
        let b = self.baked[a];
        let cmd = self.command(a);
        let adapter = MkAdapter::of(cmd.library, cmd.function).expect("playable");
        let g = match per_item {
            PerItem::Coissued(k) => {
                step_geometry(cmd, adapter, Some(k)).map_err(|e| defect(cmd.function, e))?
            }
            PerItem::Widest | PerItem::Split { .. } => self.geometry(a, false)?,
        };
        let (table_at, row_len) = self.rows[b].expect("a played command has rows");
        let row = match inst {
            Some(i) if row_len > 0 => format!("mk_a + {table_at}u + ({i}) * {row_len}u"),
            _ => format!("mk_a + {table_at}u"),
        };
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
        // A decode attention's heads grouped per virtual threadgroup, in `MK_P`.
        let heads = heads_rule(cmd).map(|rule| heads_msl(&rule));
        let gy = match &heads {
            Some(h) => format!("({gy} / {h})"),
            None => gy,
        };
        let split = match per_item {
            PerItem::Split { shape, ord } => self.planned.shapes[shape].spread[ord]
                .fit
                .map(|_| (shape, ord)),
            PerItem::Widest | PerItem::Coissued(_) => None,
        };
        let (k, items) = match (load_sized, split) {
            (true, _) => (
                format!("{}u", g.vtgs_per_item),
                format!("mk_items(s.grid, {}u)", g.vtgs_per_item),
            ),
            (false, Some((shape, ord))) => (
                format!("MK_S{shape}_K{ord}(mk_p)"),
                format!("MK_S{shape}_N{ord}(mk_p)"),
            ),
            (false, None) if heads.is_some() => (
                format!("{}u", g.vtgs_per_item),
                format!("mk_items(s.grid, {}u)", g.vtgs_per_item),
            ),
            (false, None) => (
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
        // What the plan shares must be read and written device-coherently.
        for x in cmd.bindings {
            let shared = self
                .kv
                .binding(x)
                .is_some_and(|l| self.planned.plan.shared.contains(&l));
            if shared && !adapter.coherent.contains(x.binding_index()) {
                let e = MegakernelError::IncoherentShared {
                    symbol: cmd.function,
                    binding: x.binding_index(),
                };
                return Err(defect(cmd.function, e));
            }
        }
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

/// How many virtual threadgroups a step's items play, as its spelling says.
#[derive(Clone, Copy)]
enum PerItem {
    /// Its adapter's width (a pinned unit's member).
    Widest,
    /// A co-issued member's whole grid in one item.
    Coissued(u32),
    /// Spread step `ord` of phase shape `shape`: the shape's split, in `MK_P`.
    Split { shape: usize, ord: usize },
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

/// Where the emission stands: the loops around it (counter, iterations), what ran before, and
/// the grid-barrier sites so far.
struct Emit {
    loops: Vec<(String, u32)>,
    prior: Prior,
    sites: u32,
}

impl Emit {
    /// The grid barrier opening a phase here: none before the kernel's first phase, guarded by
    /// the counters of a loop the kernel starts in.
    fn barrier(&mut self, ind: &str, out: &mut String) {
        let site = self.sites;
        let call = format!("mk_grid_sync(mk_sync, mk_idx, mk_p, mk_t, mk_gen, mk_ok, {site}u);");
        match &self.prior {
            Prior::Nothing => return,
            Prior::Always => {
                let _ = writeln!(out, "{ind}{call}");
            }
            Prior::Unless(cs) => {
                let _ = writeln!(out, "{ind}if (({}) != 0u) {call}", cs.join(" | "));
            }
        }
        self.sites += 1;
    }
}

/// What ran before a point of the kernel text.
#[derive(Clone, Debug)]
enum Prior {
    /// Nothing: the kernel's first phase needs no barrier.
    Nothing,
    /// Something, unless every one of these loop counters is 0 (loops the kernel starts in).
    Unless(Vec<String>),
    /// Something, always.
    Always,
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

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_target_metal::tape::ids::GpuCores;
    use std::collections::HashMap;

    /// Evaluates the generated program-scope constants at one `MK_P`: the arithmetic the
    /// pipeline's specialization folds (unsigned `+ - * / %`, comparisons, `?:`).
    /// Evaluates the generated program-scope constants and split functions with `p` threadgroups
    /// launched and all `p` checked in (`MK_P` = `mk_p` = `p`): the arithmetic the kernel runs
    /// (unsigned `+ - * / %`, comparisons, `?:`), each function of `mk_p` a name.
    fn eval_constants(text: &str, p: u32) -> HashMap<String, u64> {
        let mut env = HashMap::from([
            ("MK_P".to_string(), u64::from(p)),
            ("mk_p".to_string(), u64::from(p)),
        ]);
        for line in text.lines() {
            let line = line.trim();
            let (name, expr) = if let Some(rest) = line.strip_prefix("constant ") {
                let (_, rest) = rest.split_once(' ').expect("a typed constant");
                rest.split_once(" = ").expect("an initialized constant")
            } else if let Some(rest) = line.strip_prefix("METAL_FUNC ") {
                let (_, rest) = rest.split_once(' ').expect("a typed function");
                let (name, rest) = rest
                    .split_once("(uint mk_p) { return ")
                    .expect("a split function");
                (name, rest.trim_end_matches(" }"))
            } else {
                continue;
            };
            let v = eval(expr.trim_end_matches(';'), &env);
            env.insert(name.to_string(), v);
        }
        env
    }

    fn tokens(expr: &str) -> Vec<String> {
        let chars: Vec<char> = expr.chars().collect();
        let mut out = Vec::new();
        let mut word = String::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_ascii_alphanumeric() || c == '_' {
                word.push(c);
                i += 1;
                continue;
            }
            if !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
            if (c == '<' || c == '>') && chars.get(i + 1) == Some(&'=') {
                out.push(format!("{c}="));
                i += 2;
                continue;
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
            i += 1;
        }
        if !word.is_empty() {
            out.push(word);
        }
        out
    }

    fn eval(expr: &str, env: &HashMap<String, u64>) -> u64 {
        let t = tokens(expr);
        let mut at = 0;
        let v = ternary(&t, &mut at, env);
        assert_eq!(at, t.len(), "trailing tokens in {expr}");
        v
    }

    fn ternary(t: &[String], at: &mut usize, env: &HashMap<String, u64>) -> u64 {
        let c = compare(t, at, env);
        if t.get(*at).is_some_and(|x| x == "?") {
            *at += 1;
            let a = ternary(t, at, env);
            assert_eq!(t[*at], ":");
            *at += 1;
            let b = ternary(t, at, env);
            return if c != 0 { a } else { b };
        }
        c
    }

    fn compare(t: &[String], at: &mut usize, env: &HashMap<String, u64>) -> u64 {
        let a = additive(t, at, env);
        let op = t.get(*at).cloned().unwrap_or_default();
        if !["<", ">", "<=", ">="].contains(&op.as_str()) {
            return a;
        }
        *at += 1;
        let b = additive(t, at, env);
        u64::from(match op.as_str() {
            "<" => a < b,
            ">" => a > b,
            "<=" => a <= b,
            _ => a >= b,
        })
    }

    fn additive(t: &[String], at: &mut usize, env: &HashMap<String, u64>) -> u64 {
        let mut v = multiplicative(t, at, env);
        while let Some(op) = t.get(*at).filter(|x| *x == "+" || *x == "-").cloned() {
            *at += 1;
            let b = multiplicative(t, at, env);
            v = match op.as_str() {
                "+" => v + b,
                _ => v.checked_sub(b).expect("no unsigned underflow"),
            };
        }
        v
    }

    fn multiplicative(t: &[String], at: &mut usize, env: &HashMap<String, u64>) -> u64 {
        let mut v = primary(t, at, env);
        while let Some(op) = t
            .get(*at)
            .filter(|x| ["*", "/", "%"].contains(&x.as_str()))
            .cloned()
        {
            *at += 1;
            let b = primary(t, at, env);
            v = match op.as_str() {
                "*" => v * b,
                "/" => v / b,
                _ => v % b,
            };
        }
        v
    }

    fn primary(t: &[String], at: &mut usize, env: &HashMap<String, u64>) -> u64 {
        let x = t[*at].clone();
        *at += 1;
        if x == "(" {
            let v = ternary(t, at, env);
            assert_eq!(t[*at], ")");
            *at += 1;
            return v;
        }
        if t.get(*at).is_some_and(|y| y == "(") {
            // A split function, called with the participants.
            assert_eq!(t[*at + 1..*at + 3], ["mk_p", ")"], "{x}(mk_p)");
            *at += 3;
        }
        match x.strip_suffix('u').and_then(|n| n.parse().ok()) {
            Some(n) => n,
            None => *env.get(&x).unwrap_or_else(|| panic!("unbound {x}")),
        }
    }

    /// The megakernel's heads grouping, spelled in `MK_P`, is the device's pick for TurboQuant
    /// decode ([`TqDecodeHeads::for_group`]) at every core count.
    #[test]
    fn heads_are_the_devices_pick_at_every_core_count() {
        let geometries = [
            (64, 32, 8),
            (128, 24, 8),
            (128, 32, 8),
            (256, 16, 8),
            (512, 16, 2),
            (128, 28, 4),
            (64, 9, 3),
        ];
        for (hd, q, kv) in geometries {
            let geometry = (HeadDim(hd), NumQHeads(q), NumKvHeads(kv));
            let mut rule: Vec<(u32, u32)> =
                TqDecodeHeads::candidates(geometry.0, geometry.1, geometry.2)
                    .map(|h| (q / h.get(), h.get()))
                    .filter(|&(_, h)| h > 1)
                    .collect();
            rule.reverse();
            let text = format!("constant uint H = {};", heads_msl(&rule));
            for p in 1..=128 {
                let pick =
                    TqDecodeHeads::for_group(geometry.0, geometry.1, geometry.2, GpuCores(p));
                assert_eq!(
                    eval_constants(&text, p)["H"],
                    u64::from(pick.get()),
                    "hd {hd}, {q} q / {kv} kv heads, {p} cores"
                );
            }
        }
    }

    /// Every phase split is valid and even at every core count: each item plays at least one
    /// virtual threadgroup and no more than the whole threadgroup holds, a step's items cover its
    /// grid with none empty, the cursors follow the items, no lane plays more rounds than the
    /// split's, and every pinned lane is a threadgroup of the launch.
    #[test]
    fn a_phase_splits_evenly_at_every_core_count() {
        let fit = |vtgs: u32, widest: u32, whole: u32| SpreadIn {
            vtgs: format!("{vtgs}u"),
            declared: vtgs.div_ceil(widest),
            fit: Some((widest, whole)),
        };
        let shapes = [
            PhaseShape {
                spread: vec![fit(256, 1, 16)],
                free: 0,
                groups: 0,
            },
            PhaseShape {
                spread: vec![fit(128, 4, 32), fit(8, 1, 4)],
                free: 2,
                groups: 2,
            },
            PhaseShape {
                spread: vec![fit(3, 1, 1)],
                free: 1,
                groups: 1,
            },
            PhaseShape {
                spread: vec![fit(4096, 8, 32), fit(4096, 8, 32), fit(17, 2, 8)],
                free: 0,
                groups: 3,
            },
            PhaseShape {
                spread: vec![
                    fit(640, 16, 16),
                    SpreadIn {
                        vtgs: "12u".into(),
                        declared: 6,
                        fit: None,
                    },
                ],
                free: 5,
                groups: 5,
            },
        ];
        for (s, shape) in shapes.iter().enumerate() {
            let mut text = String::new();
            shape.msl(s, &mut text);
            for p in 1..=128u32 {
                let env = eval_constants(&text, p);
                let get = |n: String| env[&format!("MK_S{s}_{n}")];
                let mut total = 0;
                for (i, x) in shape.spread.iter().enumerate() {
                    assert_eq!(
                        get(format!("C{i}")),
                        total,
                        "shape {s}, {p} cores: cursor {i}"
                    );
                    let n = get(format!("N{i}"));
                    total += n;
                    if let Some((_, whole)) = x.fit {
                        let (v, k) = (get(format!("V{i}")), get(format!("K{i}")));
                        assert!(
                            (1..=u64::from(whole)).contains(&k),
                            "shape {s}, {p} cores: step {i} plays {k} per item"
                        );
                        assert!(
                            n * k >= v && (n - 1) * k < v,
                            "shape {s}, {p} cores: step {i}'s {n} items of {k} and its grid of {v}"
                        );
                    }
                }
                let rounds = get("R".into());
                assert!(
                    total.div_ceil(u64::from(p)) <= rounds,
                    "shape {s}, {p} cores: {total} items over {rounds} rounds"
                );
                for g in 0..shape.groups {
                    assert!(
                        get(format!("L{g}")) < u64::from(p),
                        "shape {s}, {p} cores: lane of group {g}"
                    );
                }
            }
        }
    }
}
