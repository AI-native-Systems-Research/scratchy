//! HOW MANY BYTES A BUFFER HOLDS — `DesignSpaceConfig::getBufferCapacityForNode`
//! (`dsc/dsc2.cpp:3977`) and the closure beneath it.
//!
//! This is the one `dsc/` call the L3 scheduler stops on: `try_alloc_l3`
//! ([`super::dl_ops`]) asks `L3Placement::buffer_capacity_even_sticks` for the capacity of the
//! allocation it is about to commit, and every program reaches that question. ⛔ **A FABRICATED
//! CAPACITY COMMITS A FABRICATED PLACEMENT** — it compiles, it bakes, and the card reads memory
//! nothing filled. A stop here is faithful; a plausible number is not.
//!
//! ⭐ THE CLOSURE LIVES IN ITS OWN FILE SO NONE OF ITS THREE CALLERS OWNS IT. The same reference
//! call stands behind three carriers with three different signatures —
//! `L3Placement::buffer_capacity_even_sticks` (`stages/carriers.rs`),
//! `v1::Placement::buffer_capacity` (`stages/ddc_reads.rs`) and `conv::DdlSizes::buffer_capacity`
//! (`stages/ddc_sites.rs`) — and answering it inside any one of them would leave the other two to
//! reinvent it. `DesignSpaceConfig::lx_chunk_capacity` ([`super::dsc`]) is this same call already
//! made for the CHUNK stage at one fixed site: reusing it answers a different question with the
//! same number.
//!
//! ⛔ THE HOME IS NOT `impl DesignSpaceConfig`. That type carries no allocate nodes and no schedule
//! tree; ours live in a separate `v1::AllocArena` (`BTreeMap<AllocId, AllocateNode>`) handed to
//! `try_alloc_l3` as its own parameter. The units here take the node, the labelled DS and the
//! ancestor loop chain as arguments — the loop chain in particular, because the reference's
//! `getOwnerLoop()`/`getPrev()` walk is over a chain every caller already holds.
//!
//! ⛔ THE AUTHORITY IS `/Users/nickm/git/deeptools-src/<file>:<line>`, NEVER A COMMENT IN THIS
//! CRATE ABOUT IT. Eight defects in this stack were found by reading IBM's source instead of our
//! own prose, and every one was self-documented as deliberate; two were citations off by 410 and 48
//! lines. Note one drift you will meet: `stages/carriers.rs` cites `dsc/dsc2.cpp:3755` for
//! `getBufferCapacityForNodePerDimCustomLocation`, whose definition begins at **3754**.
//!
//! Five units land here, in dependency order — see `crustify-capacity/UNITS.tsv` for each one's
//! `rust_home` and callees, and `crustify-capacity/AGENT-BRIEF.md` §5 for the arm-by-arm fixture
//! measurement that says which arms to port and which to defer with a citation:
//!
//! | unit | authority | L |
//! |---|---|---|
//! | `e015_getSizeDataStageForNode` | `dsc/dsc2.cpp:3616-3752` | 137 |
//! | `e016_getSizeDataStageForNode_2arg` | `dsc/dsc2.cpp:3611-3614` | 4 |
//! | `e017_getBufferCapacityForNodePerDimCustomLocation` | `dsc/dsc2.cpp:3754-3964` | 211 |
//! | `e018_getBufferCapacityForNodePerDim` | `dsc/dsc2.cpp:3966-3975` | 10 |
//! | `e019_getBufferCapacityForNode` | `dsc/dsc2.cpp:3977-4005` | 29 |
//!
//! ⛔ INTENTIONALLY EMPTY. No signature shells and no `todo!` bodies: `todo!` is capped in this
//! crate and ratchets DOWN only, and a signature is something a porter derives from the C++ — not
//! something to inherit from whoever created the file.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickPart, cumulative_stick_sizes,
};
use crate::schedule::ddc::fold::PadType;
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::Scale;
use crate::schedule::ddc::transformation_util::{PaddingForm, StageName};
use crate::schedule::ddc::v1::{DimSample, LdsSticks, sampled_as, stick_divisor};
use crate::schedule::dsc2::{AllocateNode, Coordinate, Dsc, LdsIdx, LoopNode, Padding};
use crate::units::{Corelet, Row};

use super::dsc::{
    DataStage, DataStages, FilledDims, IndirectAlloc, LabeledDs, NamedDims, PadElems, SenComponent,
    StageDims, StatedVolumes, Symbolic, SymbolicDimInfo, UnneededPad, VolumeLimit,
};

/// WHICH NODE IS BEING SIZED — the `nodeType_ == dsc2::ScheduleNode::ALLOCATE` test
/// (`dsc/dsc2.cpp:3625`) together with the two allocate fields the HBM arm reads.
///
/// ⛔ AN ARGUMENT AND NOT `AllocateNode`: `nonUnifiedAllocInHBM_` (`dsc/dsc2.h:1004`) has no field
/// here, and every struct-literal site of [`crate::schedule::dsc2::AllocateNode`] lives in a fenced
/// file — so this follows `AllocateNode::page_sizes`, which takes its indirect alloc the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizedNode {
    /// `dsc2::ScheduleNode::ALLOCATE`.
    Allocate {
        /// `component_`.
        component: SenComponent,
        /// `nonUnifiedAllocInHBM_`, measured `false` on all 1899 allocate nodes of g0.
        non_unified_in_hbm: bool,
    },
    /// Every other `nodeType_` — a transfer, a compute, a sync or a condition.
    Other,
}

/// THE LOOPS ABOVE A NODE — `getOwnerLoop()` applied until `getPrev()` is null (`dsc/dsc2.cpp:3645`,
/// `:3648`, `:3676`), innermost first and WITHOUT the tree head, plus the head's own `denId_`.
///
/// ⛔ THE CHAIN IS AN ARGUMENT AND NOT A WALK: `ScheduleTree::head_` is itself a `LoopNode` seeded
/// with `denId_ = 0` (`dsc/dsc2.h:623`, `:629`), so the loop that ends the reference's walk is the
/// head — and `schedule/stages/tree.rs`, which owns descending to it, is fenced to another agent.
#[derive(Debug, Clone)]
pub struct AncestorLoops<'a> {
    nested: Vec<&'a LoopNode>,
    root_den: Option<DatastageId>,
}

impl<'a> AncestorLoops<'a> {
    /// The enclosing loops innermost first, then `ScheduleTree::head_den` for the head they hang off.
    #[must_use]
    pub const fn of(nested: Vec<&'a LoopNode>, root_den: Option<DatastageId>) -> Self {
        Self { nested, root_den }
    }

    /// `node->getOwnerLoop() == nullptr || node->getOwnerLoop()->prev_ == nullptr` — the node hangs
    /// straight off the tree head, which is what the HBM arm demands (`dsc/dsc2.cpp:3628-3630`).
    #[must_use]
    pub fn at_root(&self) -> bool {
        self.nested.is_empty()
    }
}

/// WHAT SIZING A NODE READS OFF THE DSC — the three `DesignSpaceConfig` members
/// `getSizeDataStageForNode` touches beyond the ones [`LdsSticks`] and [`Dsc`] already carry.
pub trait SizeDsc: LdsSticks + Dsc {
    /// `N_` (`dsc/designSpaceConfig.h`) as a whole stage, [`None`] for a DSC that states no dim of
    /// it — an all-`-1` `DataStructDims` is `empty()` by the reference's own test
    /// (`dsc/dims.cpp:112`). Only its `paddingSizes_` landed before this unit, as `full_padding`.
    fn whole_data_structure(&self) -> Option<NamedDims>;

    /// `getLayoutDimSet(ldsIdx)` — `primaryDsInfo_.at(labeledDs_.at(ldsIdx).dsType_)
    /// .layoutDimOrder_` as a set, whose [`None`] is that pair of `.at`s.
    ///
    /// ⛔ NOT [`Dsc::layout_dims`], WHICH IS `getLayoutDims`: that is the allocate node's own order
    /// and this is the labelled DS's. The two orders stay two.
    fn layout_dim_set(&self, lds: LdsIdx) -> Option<BTreeSet<PrimaryDim>>;

    /// `dataStageParam_`.
    fn data_stages(&self) -> &DataStages;
}

/// Replaces: e015_getSizeDataStageForNode
///
/// HOW BIG ONE BUFFER'S DATA STAGE IS. An HBM allocation is sized by the WHOLE data structure `N_`;
/// anything else gets, per layout dim, either the stride of the parametric loop walking it or the
/// extent of the datastage the nearest enclosing loop divides by — carrying that stage's splits,
/// symbolic bounds and padding across, pruning the volume limits against the core, and recompounding.
///
/// ⛔ [`None`] IS A STOP AND NEVER A SIZE: the HBM `DT_CHECK_MSG` (`dsc/dsc2.cpp:3628`), an unseeded
/// `denId_` reaching `dataStageParam_.at(-1)` (`:3672`), a `paddingSizes_` entry the den stage does
/// not state (`:3727`), and every stop of the four callees. A fabricated extent here sizes a buffer
/// the card then reads past.
#[must_use]
pub fn size_data_stage_for_node(
    node: SizedNode,
    lds: LdsIdx,
    padding: &Padding,
    ancestors: &AncestorLoops<'_>,
    dsc: &(impl SizeDsc + ?Sized),
) -> Option<DataStage> {
    // `an->component_ == SenComponents::HBM && !an->nonUnifiedAllocInHBM_` (`:3626`), the arm every
    // HBM allocation of g0 takes.
    if let SizedNode::Allocate {
        component: SenComponent::Hbm,
        non_unified_in_hbm: false,
    } = node
    {
        if !ancestors.at_root() {
            return None;
        }
        // `newDstg.ss_ = newDstg.el_ = N_` (`:3631`) — the name comes across with the dims.
        let whole = dsc.whole_data_structure()?;
        return Some(DataStage {
            ss: whole.clone(),
            el: whole,
        });
    }

    // `getCumulativeStickSizes(labeledDs_.at(ldsIdx).dsType_)` (`:3634`), read unconditionally so
    // its stop lands where the reference's does even though only the non-allocate arm uses it.
    let stick_sizes = cumulative_stick_sizes(&dsc.stick_dims(lds), StickPart::Whole)?;
    let mut remaining = dsc.layout_dim_set(lds)?;
    // `dataStageParam_.at(0).ss_` with `DT_CHECK(coreDs.name_ == "core")` (`:3637-3638`), discharged
    // by [`DataStages`] holding the core stage as a field rather than at an index.
    let core = dsc.data_stages().core().ss.dims.dims();

    // `for (auto& [dim, padInfo] : coreDs.paddingSizes_)` (`:3639-3643`). ⛔ THE SET IS WIDENED IN
    // PLACE: the reference inserts into the container it is testing, so a window dim added for an
    // earlier key can satisfy a later key's own `targetDims.count(dim)` and a collect-then-extend
    // would lose that cascade.
    for (&dim, pad_info) in &core.padding {
        if let Some(window) = pad_info.window_dim
            && remaining.contains(&dim)
            && padding.get(dim) != PadType::NoPad
        {
            remaining.insert(window);
        }
    }

    let mut ss = StageDims::default();
    let mut el = StageDims::default();
    let mut den_for_dim: BTreeMap<PrimaryDim, DatastageId> = BTreeMap::new();
    let mut above = ancestors.nested.iter();
    // `while (!targetDims.empty())` (`:3646-3669`).
    while !remaining.is_empty() {
        let Some(enclosing) = above.next() else {
            // `myParentLoop->getPrev() == nullptr` (`:3648`): the head takes every dim still left,
            // and neither its own `dims_` nor its parametric flag is ever read.
            let den = ancestors.root_den?;
            for dim in std::mem::take(&mut remaining) {
                den_for_dim.insert(dim, den);
            }
            break;
        };
        if enclosing.parametric_lds.is_some() {
            // `myParentLoop->dims_[0].dim_` (`:3653`).
            let Some(loop_dim) = enclosing.dims.first().map(|entry| entry.dim) else {
                continue;
            };
            if remaining.contains(&loop_dim) {
                let stride = Extent(i64::try_from(enclosing.parametric_stride(dsc)?.0).ok()?);
                ss.extents.insert(loop_dim, stride);
                el.extents.insert(loop_dim, stride);
                // `newDstg.ss_.paddingSizes_[loopDim];` (`:3659-3660`) — a bare `operator[]`, whose
                // whole effect is the ZERO entry it default-inserts.
                if core.padding.contains_key(&loop_dim) {
                    ss.padding.entry(loop_dim).or_default();
                    el.padding.entry(loop_dim).or_default();
                }
                remaining.remove(&loop_dim);
            }
        } else {
            for entry in &enclosing.dims {
                if remaining.remove(&entry.dim) {
                    // `denDsForDim[ldim] = myParentLoop->denId_` (`:3667`), whose `-1` is the
                    // `dataStageParam_.at(dsIdx)` throw below brought forward.
                    den_for_dim.insert(entry.dim, enclosing.den?);
                }
            }
        }
    }

    // `for (auto& [dim, dsIdx] : denDsForDim)` (`:3673-3706`).
    let mut ss_info: BTreeMap<PrimaryDim, SymbolicDimInfo> = BTreeMap::new();
    let mut el_info: BTreeMap<PrimaryDim, SymbolicDimInfo> = BTreeMap::new();
    let mut stated: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit> = BTreeMap::new();
    for (&dim, &den) in &den_for_dim {
        let den_stage = dsc.data_stages().at(den)?;
        let den_ss = den_stage.ss.dims.dims();
        let den_el = den_stage.el.dims.dims();
        // `primaryDimToValHandler_st(dim) = myDenDstg.<half>.primaryDimToVal_st(dim)` (`:3675-3677`).
        // ⭐ `Extent(-1)` IS THE REFERENCE'S OWN ANSWER, NOT A FABRICATION: with every padding type
        // `NOPAD`, `calculate_padded` reaches no `DT_ERROR`, so the only [`None`] the whole-extent
        // reading has is the `-1` of an unstated slot (`dsc/dims.cpp:563-566`).
        ss.extents
            .insert(dim, den_ss.whole_extent(dim).unwrap_or(Extent(-1)));
        el.extents
            .insert(dim, den_el.whole_extent(dim).unwrap_or(Extent(-1)));
        // ⛔ THE GUARD IS ON THE SS SIDE AND THE EL COPY IS AN `.at` (`:3679-3697`): a stage whose
        // two halves disagree about a split is a real throw, so it is a stop and not an absence.
        if let Some(shares) = den_ss.corelet_split.get(&dim) {
            ss.corelet_split.insert(dim, shares.clone());
            el.corelet_split
                .insert(dim, den_el.corelet_split.get(&dim)?.clone());
        }
        if let Some(rows) = den_ss.row_split.get(&dim) {
            ss.row_split.insert(dim, rows.clone());
            el.row_split
                .insert(dim, den_el.row_split.get(&dim)?.clone());
        }
        if let Some(shares) = den_ss.pe_sfp_split.get(&dim) {
            ss.pe_sfp_split.insert(dim, shares.clone());
            el.pe_sfp_split
                .insert(dim, den_el.pe_sfp_split.get(&dim)?.clone());
        }
        if let Some(&info) = den_ss.symbolic.info().get(&dim) {
            ss_info.insert(dim, info);
            el_info.insert(dim, *den_el.symbolic.info().get(&dim)?);
        }
        // `for (const auto& [symDims, volumeLimit] : myDenDstg.ss_.maxSymbolicVolume_)` (`:3699`) —
        // only keys naming `dim`, and the SMALLER of two limits two den stages state for one key.
        for (sym_dims, &limit) in den_ss.symbolic.volumes() {
            if !sym_dims.contains(&dim) {
                continue;
            }
            let held = stated.entry(sym_dims.clone()).or_insert(limit);
            *held = (*held).min(limit);
        }
    }

    // `newDstg.ss_.pruneMaxSymbolicVolumes(coreDs)` (`:3709`) — the UNFUSED prune, which is what
    // [`Symbolic::prune_volumes`]'s own doc cites this line for.
    //
    // ⛔ NOT `Symbolic::new(ss_info, stated)` DIRECTLY: that filter DROPS a limit keyed on a dim no
    // den stage made symbolic here, where the reference keeps it and lets the prune re-key it. Every
    // key the prune returns is a subset of the info it pruned against, so the filter is then inert.
    let volumes = StatedVolumes::new(stated)
        .pruned_against(&Symbolic::new(ss_info.clone(), BTreeMap::new()), &core.symbolic);
    ss.symbolic = Symbolic::new(ss_info, volumes.clone());
    // `newDstg.el_.maxSymbolicVolume_ = newDstg.ss_.maxSymbolicVolume_` (`:3710`).
    el.symbolic = Symbolic::new(el_info, volumes);

    // ⭐ A SECOND PASS, `// run in a separate loop so that all symbolic, unpadded, window dims are
    // set` (`:3711-3735`).
    for (&dim, &den) in &den_for_dim {
        if padding.get(dim) == PadType::NoPad {
            continue;
        }
        let den_ss = dsc.data_stages().at(den)?.ss.dims.dims();
        // `DT_ERROR("Dim with padding is missing padding info")` (`:3716-3718`), raised BEFORE the
        // emplace and so regardless of what `newDstg` already states.
        let from_den = *den_ss.padding.get(&dim)?;
        // `emplace` DOES NOT OVERWRITE. Only the parametric arm could have put an entry here, and it
        // erased its dim from `targetDims` before `denDsForDim` could name it, so the two are
        // disjoint — but the non-overwriting spelling is the reference's and stays.
        ss.padding.entry(dim).or_insert(from_den);
        if !matches!(node, SizedNode::Allocate { .. })
            && let Some(pad) = ss.padding.get_mut(&dim)
        {
            // The three counts are zeroed FIRST, because the span is then read off the very half
            // being built (`:3721-3726`).
            pad.unneeded = UnneededPad::NONE;
            let total = full_span_with_unneeded(&ss, dim)?;
            if let Some(&(_, stick)) = stick_sizes.iter().find(|&&(sized, _)| sized == dim) {
                let stick = i64::try_from(stick.0).ok()?;
                if total.0 < stick
                    && let Some(pad) = ss.padding.get_mut(&dim)
                {
                    pad.unneeded.total = PadElems(u32::try_from(stick - total.0).ok()?);
                }
            }
        }
        // `newDstg.el_.paddingSizes_.emplace(dim, padInfo)` (`:3733`) — the FINAL ss_ entry, after
        // the unneeded counts were rewritten in it.
        let Some(&settled) = ss.padding.get(&dim) else {
            continue;
        };
        el.padding.entry(dim).or_insert(settled);
    }

    // `newDstg.ss_.compound(); newDstg.el_.compound();` (`:3737-3738`). ⛔ THE RAW
    // [`StageDims::compound`] AND NOT [`FilledDims::compound`], whose restore-if-emptied guard is a
    // documented divergence this body must not inherit.
    ss.compound();
    el.compound();
    // A default-constructed `dsc2::DataStage` carries the EMPTY `name_`; the layout order is
    // non-empty, so both halves state a dim and neither [`FilledDims::of`] refuses.
    Some(DataStage {
        ss: NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(ss)?,
        },
        el: NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(el)?,
        },
    })
}

/// `newDstg.ss_.primaryDimToVal_st(dim, NO_COMPONENT, -1, -1, {dim, PADDED_FULLSPAN_WUNNEEDED})`
/// (`dsc/dsc2.cpp:3723-3725`) with the reference's `-1` TOLD APART FROM ITS THROWS.
///
/// ⛔ [`StageDims::padded_extent`] answers [`None`] both for `if (val < 0) return -1`
/// (`dsc/dims.cpp:564`) and for the four `DT_ERROR`s under it, and this one caller needs the first as
/// a VALUE: a dim whose den stage stated nothing was just written `-1`, and the reference goes on to
/// compute `stickSize + 1` of unneeded pad from it. A blanket `?` would stop there instead.
fn full_span_with_unneeded(ss: &StageDims, dim: PrimaryDim) -> Option<Extent> {
    if !ss.symbolic.info().contains_key(&dim) && ss.extent(dim).is_none_or(|slot| slot.0 < 0) {
        return Some(Extent(-1));
    }
    ss.padded_extent(dim, PadType::PaddedFullSpanWUnneeded)
}

#[cfg(test)]
mod tests_e015 {
    //! ⭐⭐ TWO SIZINGS THE REFERENCE ITSELF EXPORTED — `/Users/nickm/tmp/bridge1-fixtures/g0/debug/
    //! sdsc_14/sdsc.json`, DSC `14_t729_fq_afp8_op`, the ONE g0 program of 187 that carries padding
    //! and a parametric loop.
    //!
    //! It states `primaryDsInfo_["OUTPUT"] = {layoutDimOrder_: ["mb","out","y"], stickDimOrder_:
    //! ["out"], stickSize_: [128]}`, `N_ = {out_: 2048, mb_: 1, y_: 1}` with `paddingSizes_` on `out`
    //! and `mb`, `dataStageParam_["0"].ss_` (`name_: "core"`) `= {out_: 128, mb_: 1, y_: 1}`,
    //! `["2"] = {out_: 64, mb_: 1, y_: 1}`, and `scheduleTreeHeadDenId_ = 0`.
    //!
    //! * `allocate-Tensor1_hbm` (`ldsIdx_: 1`, `component_: "hbm"`, `nonUnifiedAllocInHBM_: 0`) hangs
    //!   straight off `root_level_operations`, so it is sized `N_`: `out` is **2048**.
    //! * `transfer_lds1_src:sfp_dst:lxsu` sits under `parametric_loop_out(padded)__2`
    //!   (`parametricLdsIdx_: 1`), `parametric_loop_mb(padded)` (likewise) and `loop_ds1_ds2_y`
    //!   (`denId_: 2`), so `out` takes the parametric stride the export's own `loopEleOffsets_:
    //!   {"out": 128}` records — **128** — and `y` comes from datastage 2.
    //!
    //! ⭐ THE THREE ANSWERS FOR `out` DISCRIMINATE: 2048 whole, 128 through the parametric arm and 64
    //! through the ordinary den arm. Neither arm can reach the other's number.

    use std::collections::{BTreeMap, BTreeSet};

    use crate::arch::Elements;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::dsc2::{BlockNode, LayoutDims, LoopDim, NodeName};

    use super::super::dsc::{
        DataStage, DataStages, DimPadding, FilledDims, NamedDims, PadSizes, SenComponent, StageDims,
    };
    use super::{
        AncestorLoops, Dsc, LdsIdx, LdsSticks, LoopNode, Padding, SizeDsc, SizedNode,
        size_data_stage_for_node,
    };

    const MB: PrimaryDim = PrimaryDim::Mb;
    const OUT: PrimaryDim = PrimaryDim::Out;
    const Y: PrimaryDim = PrimaryDim::Y;

    /// `paddingSizes_` as the export prints it — `padFront_`/`padBack_` and every other count zero.
    fn padding_of(edges: &[(PrimaryDim, PadSizes)]) -> BTreeMap<PrimaryDim, DimPadding> {
        edges
            .iter()
            .map(|&(dim, sizes)| {
                (
                    dim,
                    DimPadding {
                        sizes,
                        ..DimPadding::default()
                    },
                )
            })
            .collect()
    }

    /// One half of a stage: its `out_`/`mb_`/`y_` slots and its `paddingSizes_`.
    fn half(
        name: &str,
        extents: &[(PrimaryDim, i64)],
        padding: &BTreeMap<PrimaryDim, DimPadding>,
    ) -> NamedDims {
        let mut dims = StageDims::default();
        for &(dim, extent) in extents {
            dims.extents.insert(dim, Extent(extent));
        }
        dims.padding = padding.clone();
        NamedDims {
            name: StageName(name.to_owned()),
            dims: FilledDims::of(dims).expect("a stage stating three dims"),
        }
    }

    /// A stage whose two halves the export prints with the same numbers.
    fn stage(
        name: &str,
        extents: &[(PrimaryDim, i64)],
        padding: &BTreeMap<PrimaryDim, DimPadding>,
    ) -> DataStage {
        DataStage {
            ss: half(name, extents, padding),
            el: half(name, extents, padding),
        }
    }

    /// `parametric_loop_<dim>(padded)` — `parametricLdsIdx_: 1`, no `numId_` and no `denId_`.
    pub(super) fn parametric(name: &str, dim: PrimaryDim) -> LoopNode {
        LoopNode {
            block: BlockNode {
                name: NodeName(name.to_owned()),
                children: Vec::new(),
            },
            dims: vec![LoopDim {
                dim,
                kind: MetaDimKind::Padded,
            }],
            num: None,
            den: None,
            parametric_lds: Some(LdsIdx(1)),
        }
    }

    /// `loop_ds1_ds<den>_<dim>` — an ordinary loop dividing `dim` by datastage `den`.
    pub(super) fn dividing(name: &str, dim: PrimaryDim, den: DatastageId) -> LoopNode {
        LoopNode {
            block: BlockNode {
                name: NodeName(name.to_owned()),
                children: Vec::new(),
            },
            dims: vec![LoopDim {
                dim,
                kind: MetaDimKind::Unpadded,
            }],
            num: Some(DatastageId(1)),
            den: Some(den),
            parametric_lds: None,
        }
    }

    /// `sdsc_14`'s DSC through the four seams this unit reads it by.
    pub(super) struct Sdsc14 {
        stages: DataStages,
    }

    impl Sdsc14 {
        /// `dataStageParam_` as the export prints it: `"0"` is `core`, `"1"` is `chunk`, `"2"` is the
        /// stage `loop_ds1_ds2_y` divides by — whose `paddingSizes_` the export voids.
        pub(super) fn of() -> Self {
            let unpadded = padding_of(&[(OUT, PadSizes::Unpadded), (MB, PadSizes::Unpadded)]);
            let voided = padding_of(&[(OUT, PadSizes::Voided), (MB, PadSizes::Voided)]);
            let core = &[(OUT, 128), (MB, 1), (Y, 1)];
            let mut stages = DataStages::new(
                stage("core", core, &unpadded),
                stage("chunk", core, &unpadded),
            );
            stages.set(
                DatastageId(2),
                DataStage {
                    ss: half("2", &[(OUT, 64), (MB, 1), (Y, 1)], &voided),
                    el: half("2el", &[(OUT, 64), (MB, 1), (Y, 1)], &voided),
                },
            );
            Self { stages }
        }
    }

    impl LdsSticks for Sdsc14 {
        /// `primaryDsInfo_["OUTPUT"]` — `stickDimOrder_: ["out"]`, `stickSize_: [128]`.
        fn stick_dims(&self, _lds: LdsIdx) -> StickDims {
            StickDims(vec![(OUT, Elements(128))])
        }
    }

    impl Dsc for Sdsc14 {
        /// `getLayoutDims(1)` — the order `allocate_lds1_lx` carries.
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            LayoutDims::new(MB, vec![OUT, Y])
        }
    }

    impl SizeDsc for Sdsc14 {
        /// `N_ = {"name_": "n", "out_": 2048, "mb_": 1, "y_": 1}`, padded on `out` and `mb`.
        fn whole_data_structure(&self) -> Option<NamedDims> {
            Some(half(
                "n",
                &[(OUT, 2048), (MB, 1), (Y, 1)],
                &padding_of(&[(OUT, PadSizes::Unpadded), (MB, PadSizes::Unpadded)]),
            ))
        }

        /// `getLayoutDimSet(1)` — `primaryDsInfo_["OUTPUT"].layoutDimOrder_` as a set.
        fn layout_dim_set(&self, _lds: LdsIdx) -> Option<BTreeSet<PrimaryDim>> {
            Some(BTreeSet::from([MB, OUT, Y]))
        }

        fn data_stages(&self) -> &DataStages {
            &self.stages
        }
    }

    /// e015 — the two sizings `sdsc_14`'s own export pins.
    #[test]
    fn an_hbm_allocation_is_sized_whole_and_a_transfer_by_the_loops_above_it() {
        let dsc = Sdsc14::of();

        let hbm = size_data_stage_for_node(
            SizedNode::Allocate {
                component: SenComponent::Hbm,
                non_unified_in_hbm: false,
            },
            LdsIdx(1),
            &Padding::default(),
            &AncestorLoops::of(Vec::new(), Some(DatastageId(0))),
            &dsc,
        )
        .expect("`allocate-Tensor1_hbm`, which hangs off the tree head");
        // `N_`, not the core stage's 128 and not datastage 2's 64.
        assert_eq!(hbm.ss.dims.dims().extent(OUT), Some(Extent(2048)));
        assert_eq!(hbm.el.dims.dims().extent(OUT), Some(Extent(2048)));
        assert_eq!(hbm.ss.dims.dims().extent(MB), Some(Extent(1)));

        let out_loop = parametric("parametric_loop_out(padded)__2", OUT);
        let mb_loop = parametric("parametric_loop_mb(padded)", MB);
        let y_loop = dividing("loop_ds1_ds2_y", Y, DatastageId(2));
        let sized = size_data_stage_for_node(
            SizedNode::Other,
            LdsIdx(1),
            &Padding::default(),
            &AncestorLoops::of(
                vec![&out_loop, &mb_loop, &y_loop],
                Some(DatastageId(0)),
            ),
            &dsc,
        )
        .expect("`transfer_lds1_src:sfp_dst:lxsu`, whose three dims the chain above it all divides");
        // `out` is the parametric stride the export's `loopEleOffsets_` records, NOT stage 2's 64.
        assert_eq!(sized.ss.dims.dims().extent(OUT), Some(Extent(128)));
        assert_eq!(sized.el.dims.dims().extent(OUT), Some(Extent(128)));
        // `mb` is the `1` its place in the layout order earns it, `y` comes from datastage 2.
        assert_eq!(sized.ss.dims.dims().extent(MB), Some(Extent(1)));
        assert_eq!(sized.ss.dims.dims().extent(Y), Some(Extent(1)));
        // The parametric arm's ZERO entries, for the two dims the core stage states padding for.
        assert_eq!(
            sized.ss.dims.dims().padding.get(&OUT),
            Some(&DimPadding::default())
        );
        assert_eq!(sized.ss.dims.dims().padding.get(&Y), None);
    }

    /// e015 — the two stops, and neither of them is a size.
    #[test]
    fn an_hbm_allocation_below_a_loop_and_an_unseeded_head_are_both_stops() {
        let dsc = Sdsc14::of();
        let y_loop = dividing("loop_ds1_ds2_y", Y, DatastageId(2));

        // `DT_CHECK_MSG(.., "HBM allocation should be at root of schedule tree")`
        // (`dsc/dsc2.cpp:3628-3630`) — the same node one loop deeper.
        assert_eq!(
            size_data_stage_for_node(
                SizedNode::Allocate {
                    component: SenComponent::Hbm,
                    non_unified_in_hbm: false,
                },
                LdsIdx(1),
                &Padding::default(),
                &AncestorLoops::of(vec![&y_loop], Some(DatastageId(0))),
                &dsc,
            ),
            None
        );

        // `dataStageParam_.at(denDsForDim[dim])` for the head's own `denId_` (`:3672`): `mb` and `out`
        // reach the head, and a tree the scheduler never seeded states no `denId_` there.
        assert_eq!(
            size_data_stage_for_node(
                SizedNode::Other,
                LdsIdx(1),
                &Padding::default(),
                &AncestorLoops::of(vec![&y_loop], None),
                &dsc,
            ),
            None
        );
    }
}

/// Replaces: e016_getSizeDataStageForNode_2arg
///
/// [`size_data_stage_for_node`] ASKED WITH THE ALLOCATION ITSELF, which is where the `ldsIdx_` and the
/// `padding_` come from (`dsc/dsc2.cpp:3611-3614`).
///
/// ⛔ THE TWO ARGUMENTS ARE NOT THE SAME NODE, AND THAT IS THE WHOLE CONTENT OF THE FOUR LINES: `node`
/// says WHERE the sizing happens — the loops above it, plus its own `nodeType_`/`component_`/
/// `nonUnifiedAllocInHBM_` for the HBM arm (`:3625-3627`) — while `alloc` supplies ONLY the labelled
/// DS and the padding form. `ddc/ddc_transformation.cpp:1695` hands over a TRANSFER beside a reference
/// tensor's allocate node, and `dsc/dsc2.cpp:3541` and `:3765` a `nodeForLocation` different again.
/// Taking the component off `alloc` would size a transfer under three loops as whole HBM.
///
/// ⛔ [`None`] FOR AN ALLOCATION WITH NO LABELLED DS IS `DT_ERROR("Cannot get datastage for node
/// without an allocation for lds")` (`:3620`), which [`AllocateNode::lds`]'s [`Option`] brings forward
/// into the forwarder — 85 of g0's 1899 allocate nodes are constant allocations stating `ldsIdx_: -1`.
/// Plus every stop of [`size_data_stage_for_node`].
#[must_use]
pub fn size_data_stage_of_alloc_at_node(
    node: SizedNode,
    alloc: &AllocateNode,
    ancestors: &AncestorLoops<'_>,
    dsc: &(impl SizeDsc + ?Sized),
) -> Option<DataStage> {
    size_data_stage_for_node(node, alloc.lds?, &alloc.placement.padding, ancestors, dsc)
}

#[cfg(test)]
mod tests_e016 {
    //! ⭐⭐ THE THREE FORWARDINGS `sdsc_14`'s OWN EXPORT PINS — `/Users/nickm/tmp/bridge1-fixtures/g0/
    //! debug/sdsc_14/sdsc.json`, DSC `14_t729_fq_afp8_op`, over [`super::tests_e015::Sdsc14`]'s four
    //! seams and the same three loops that module documents.
    //!
    //! * `getSizeDataStageForNode(allocate-Tensor1_hbm, allocate-Tensor1_hbm)` — the
    //!   `ddc/ddcv1.cpp:1924` and `L3DlOpsScheduler.cpp:4848` spelling, where the two arguments ARE one
    //!   node: `out` is **2048**, `N_`.
    //! * `getSizeDataStageForNode(transfer_lds1_src:sfp_dst:lxsu, allocate-Tensor1_hbm)` — the
    //!   `ddc/ddc_transformation.cpp:1695` spelling, a transfer beside a tensor's allocate node: `out`
    //!   is **128**, the parametric stride the export's own `loopEleOffsets_` records. ⭐ THIS IS THE
    //!   DISCRIMINATOR FOR THE TRAP: the alloc handed over is `component_: "hbm"` with
    //!   `nonUnifiedAllocInHBM_: 0`, so a forwarder reading the component off `alloc` rather than
    //!   `node` would answer 2048 — and, one loop down, stop outright on the root check.
    //! * `allocate_const0_pelrf` (`constIdx_: 0`, `ldsIdx_: -1`) — the `dsc/dsc2.cpp:3620` `DT_ERROR`,
    //!   a stop and not a size.
    //!
    //! ⚠️ MEASURED, AND IT SAYS THIS PATH CARRIES NO PADDING TODAY: `padding_` is EMPTY on all 1899
    //! allocate nodes of all 187 g0 reference exports, so every dim forwards `NOPAD` and e015's second
    //! padding pass (`dsc/dsc2.cpp:3711-3735`) is skipped for every one of them. `ldsIdx_ == -1` on 85
    //! of those 1899. Both counts are a 187-program sample of 134 bundles, not a proof.

    use std::collections::BTreeMap;

    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::schedule::ddc::fold::ConstIdx;
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, MaxDimSize, NodeName, NumBuffers, StartAddress,
    };

    use super::super::dsc::SenComponent;
    use super::tests_e015::{Sdsc14, dividing, parametric};
    use super::{
        AllocateNode, AncestorLoops, LdsIdx, Padding, SizedNode, size_data_stage_of_alloc_at_node,
    };

    const MB: PrimaryDim = PrimaryDim::Mb;
    const OUT: PrimaryDim = PrimaryDim::Out;
    const Y: PrimaryDim = PrimaryDim::Y;

    /// `allocate-Tensor1_hbm` as the export prints it: `ldsIdx_: 1`, `component_: "hbm"`, `padding_:
    /// {}`, `numBuffers_: 1`, `layoutDimOrder_: ["mb","out","y"]` with `maxDimSizes_: [-1,-1,-1]`.
    fn tensor1_hbm() -> AllocateNode {
        AllocateNode {
            name: NodeName("allocate-Tensor1_hbm".to_owned()),
            component: SenComponent::Hbm,
            lds: Some(LdsIdx(1)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new(
                (MB, MaxDimSize::Unset),
                vec![(OUT, MaxDimSize::Unset), (Y, MaxDimSize::Unset)],
            ),
            start_address: StartAddress::default(),
            placement: AllocPlacement {
                num_buffers: NumBuffers::Single,
                padding: Padding::default(),
                buffer_offset: BTreeMap::new(),
                is_start_addr_symbolic: false,
            },
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    /// `allocate_const0_pelrf` as the export prints it — `constIdx_: 0` and `ldsIdx_: -1`.
    ///
    /// ⚠️ ITS `layoutDimOrder_` IS `[]` IN THE EXPORT AND CANNOT BE SPELLED HERE: [`AllocLayout`] is
    /// non-empty by construction, discharging a DIFFERENT reference abort —
    /// `gapStickSpread_[layoutDimOrder_.at(0)]` (`ddc/ddcv1.cpp:1704`). e016 reads no layout at all, so
    /// the dims this fixture borrows from `allocate-Tensor1_hbm` are unread by the call under test, as
    /// are the `startAddressCoreCorelet_` and `allocUsers_` it borrows with them.
    fn const0_pelrf() -> AllocateNode {
        AllocateNode {
            name: NodeName("allocate_const0_pelrf".to_owned()),
            component: SenComponent::Pelrf,
            lds: None,
            const_idx: Some(ConstIdx(0)),
            ..tensor1_hbm()
        }
    }

    /// e016 — the allocation's own `ldsIdx_`/`padding_` reaching the sizing, and the node still
    /// deciding which arm sizes it.
    #[test]
    fn the_alloc_states_the_lds_and_the_node_states_the_arm() {
        let dsc = Sdsc14::of();
        let head = Some(DatastageId(0));
        let alloc = tensor1_hbm();

        // `getSizeDataStageForNode(allocNode, allocNode)` on an HBM allocation hanging off the tree
        // head: `N_`, and neither the core stage's 128 nor datastage 2's 64.
        let whole = size_data_stage_of_alloc_at_node(
            SizedNode::Allocate {
                component: SenComponent::Hbm,
                non_unified_in_hbm: false,
            },
            &alloc,
            &AncestorLoops::of(Vec::new(), head),
            &dsc,
        )
        .expect("`allocate-Tensor1_hbm`, whose `ldsIdx_` is 1 and not -1");
        assert_eq!(whole.ss.dims.dims().extent(OUT), Some(Extent(2048)));
        assert_eq!(whole.el.dims.dims().extent(OUT), Some(Extent(2048)));

        // `getSizeDataStageForNode(transferNode, refTensorAllocNode)`: the SAME hbm allocate node,
        // three loops down, sizing a TRANSFER. The HBM arm keys off the node, so `out` is the
        // parametric stride 128 — a component read off `alloc` would say 2048 here.
        let out_loop = parametric("parametric_loop_out(padded)__2", OUT);
        let mb_loop = parametric("parametric_loop_mb(padded)", MB);
        let y_loop = dividing("loop_ds1_ds2_y", Y, DatastageId(2));
        let sized = size_data_stage_of_alloc_at_node(
            SizedNode::Other,
            &alloc,
            &AncestorLoops::of(vec![&out_loop, &mb_loop, &y_loop], head),
            &dsc,
        )
        .expect("`transfer_lds1_src:sfp_dst:lxsu`, sized by the loops above it");
        assert_eq!(sized.ss.dims.dims().extent(OUT), Some(Extent(128)));
        assert_eq!(sized.ss.dims.dims().extent(Y), Some(Extent(1)));

        // `DT_ERROR("Cannot get datastage for node without an allocation for lds")` (`:3620`) — the
        // constant allocation, at the very same position the 2048 came from.
        assert_eq!(
            size_data_stage_of_alloc_at_node(
                SizedNode::Allocate {
                    component: SenComponent::Pelrf,
                    non_unified_in_hbm: false,
                },
                &const0_pelrf(),
                &AncestorLoops::of(Vec::new(), head),
                &dsc,
            ),
            None
        );
    }
}

/// WHICH BUFFER, SAMPLED WHERE — `getBufferCapacityForNodePerDimCustomLocation`'s `ldsIdx`, `comp`,
/// `corelet` and `row` (`dsc/dsc2.cpp:3757`), with `labeledDs_.at(ldsIdx)` already resolved beside
/// the index and the component's forcing of the other two APPLIED BY THE CONSTRUCTOR.
///
/// ⛔ THE FORCING CANNOT BE A STEP INSIDE THE BODY, because the `row` it produces then SELECTS the
/// coordinate (`:3774-3778`): a sample assembled any other way would read `sliceViewCoordinates_`
/// where the reference reads `allocateCoordinates_`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledBuffer<'a> {
    /// `ldsIdx`.
    lds: LdsIdx,
    /// `labeledDs_.at(ldsIdx)` — PAIRED WITH THE INDEX, which is what makes that `.at` unspellable,
    /// exactly as [`LabeledDs`]'s own zip of `layoutDimOrder_` with `scale_` is.
    info: &'a LabeledDs,
    /// `comp`.
    comp: SenComponent,
    /// `corelet`, once forced.
    corelet: Option<Corelet>,
    /// `row`, once forced.
    row: Option<Row>,
}

impl<'a> SampledBuffer<'a> {
    /// `if (!is_any_of(comp, PTARF, PTIRF, PTXRF)) { row = -1; if (!is_any_of(comp, SFPLRF, PELRF,
    /// L0, L0_SCALE, SFPSTATE, PESTATE)) corelet = -1; }` (`dsc/dsc2.cpp:3768-3772`).
    #[must_use]
    pub const fn of(
        lds: LdsIdx,
        info: &'a LabeledDs,
        comp: SenComponent,
        corelet: Option<Corelet>,
        row: Option<Row>,
    ) -> Self {
        let (corelet, row) = match comp {
            SenComponent::Ptarf | SenComponent::Ptirf | SenComponent::Ptxrf => (corelet, row),
            SenComponent::Sfplrf
            | SenComponent::Pelrf
            | SenComponent::L0
            | SenComponent::L0Scale
            | SenComponent::Sfpstate
            | SenComponent::Pestate => (corelet, None),
            _ => (None, None),
        };
        Self {
            lds,
            info,
            comp,
            corelet,
            row,
        }
    }
}

/// WHAT SIZING AN ALLOCATION READS OFF ITS NODE THAT [`AllocateNode`] HAS NO SLOT FOR — five
/// `dsc2::AllocateNode` members the capacity walk touches (`dsc/dsc2.h:990-1006`).
///
/// ⛔ PARAMETERS AND NOT FIELDS, for the reason [`AllocateNode::page_sizes`] takes its indirection
/// and e015 takes `nonUnifiedAllocInHBM_`: [`AllocateNode`] derives no [`Default`] and every one of
/// its construction sites is an exhaustive struct literal, three of them in `schedule/ddl/
/// conversion.rs`, which another agent owns this wave.
#[derive(Debug, Clone, Copy)]
pub struct AllocSizing<'a> {
    /// `allocateCoordinates_`.
    pub allocate_coordinates: &'a Coordinate,
    /// `sliceViewCoordinates_` — ⛔ NEVER SERIALISED (`dsc/dsc2.cpp:1827` is a bare `// TO DO:
    /// sliceview coordinate`), so [`None`] is what all 187 g0 reference exports state.
    pub slice_view_coordinates: Option<&'a Coordinate>,
    /// `ignoreSymbolicVolumeLimits_`.
    pub ignore_symbolic_volume_limits: bool,
    /// `indirectAllocType_`, as `getPageSize()` is asked with it.
    pub indirect: Option<IndirectAlloc>,
    /// `backGapCore_`'s KEYS ALONE. ⚠️ NARROWED TO WHAT THE DEFERRED GAP ARM NEEDS TO FIRE: the
    /// per-core gap values (`dsc/dsc2.cpp:3941-3954`) land with that arm.
    pub back_gap_dims: &'a BTreeSet<PrimaryDim>,
}

/// HOW THE CAPACITY IS ASKED FOR — the three trailing default arguments (`dsc/dsc2.cpp:3759-3761`).
///
/// ⛔ NO [`Default`] DERIVE: `includeGaps` DEFAULTS TRUE and `bool::default()` is false, so a derive
/// would silently drop every back gap. [`Self::DEFAULTS`] is the signature's own three values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityForm {
    /// `doNotRound` — leave each dim unrounded to full sticks.
    pub do_not_round: bool,
    /// `includeGaps` — add `backGapCore_`'s back gap to each dim.
    pub include_gaps: bool,
    /// `allowSymbolicVolumeLimit` — whether a symbolic volume limit over a layout dim is admitted
    /// rather than aborted (`:3789-3791`).
    pub allow_symbolic_volume_limit: bool,
}

impl CapacityForm {
    /// The declaration's own three defaults, verbatim.
    pub const DEFAULTS: Self = Self {
        do_not_round: false,
        include_gaps: true,
        allow_symbolic_volume_limit: false,
    };
}

/// Replaces: e017_getBufferCapacityForNodePerDimCustomLocation
///
/// HOW MANY ELEMENTS ONE BUFFER SPANS ALONG EACH LAYOUT DIM, sized at a location that need not be
/// the allocation's own: a `-1` scale spans one element, a `-2` scale a whole stick, and every other
/// dim the larger of the sizing stage's two halves read at this component, row and corelet — then MX
/// block scaled, rounded up to full sticks, and spread over its stick gap.
///
/// ⛔ `Extent(-1)` IS A SIZE AND NOT AN ABSENCE (`dsc/dsc2.cpp:3818`): e019 folds each dim in as
/// `max(size, 1)` (`:3987`), so a dim the sizing stage states nothing for contributes ONE. [`None`]
/// is a stop — every `.at`, and the narrowing named on [`sampled_or_absent`].
/// ⛔ THE COORDINATE SELECTOR IS NOT `v1::CoordinateOffsets::alloc_coordinate`, a DIFFERENT rule
/// over the same two fields (`:3774-3778` against `ddc/v1.rs:2281`).
/// ⛔ THE MX DIVIDE AND THE ROUNDING SIT INSIDE THE `scale >= 0` ARM (`:3830-3935`, brace-checked
/// against the authority), so a `-1` or `-2` scale is neither divided nor rounded.
#[must_use]
pub fn buffer_capacity_per_dim_at(
    node: SizedNode,
    alloc: &AllocateNode,
    sizing: AllocSizing<'_>,
    at: SampledBuffer<'_>,
    form: CapacityForm,
    ancestors: &AncestorLoops<'_>,
    dsc: &(impl SizeDsc + ?Sized),
) -> Option<Vec<(PrimaryDim, Extent)>> {
    let ldims = dsc.layout_dims(at.lds);
    // ⛔ THE SIZING GOES THROUGH THE ALLOCATION'S OWN `ldsIdx_`, NOT `at.lds` (`:3765`): e016 forwards
    // `myAllocNode->ldsIdx_`, and the two are a different lds wherever `node` is not the allocation.
    let dstg = size_data_stage_of_alloc_at_node(node, alloc, ancestors, dsc)?;

    // `(row != -1 && sliceViewCoordinates_.foldConstructed()) ? sliceView : allocate` (`:3774-3778`).
    let effective = match (at.row, sizing.slice_view_coordinates) {
        (Some(_), Some(slice_view)) if slice_view.fold_constructed() => slice_view,
        _ => sizing.allocate_coordinates,
    };
    let has_coordinate = effective.fold_constructed();

    // `if (!myAllocNode->ignoreSymbolicVolumeLimits_)` (`:3780-3802`). ⛔ THE STATE MACHINE UNDER IT
    // (`:3809-3817`) IS INERT RATHER THAN SKIPPED, and that is why it needs no code: the only writer
    // of `currentVolumeLimit` is this loop, so on every path that leaves it the limit is EMPTY,
    // `dimInSymVolume`/`symVolumeInProgress`/`lastInSymVolume` are all false and the
    // `DT_CHECK_MSG(!(symVolumeInProgress && !dimInSymVolume), ..)` cannot fire.
    if !sizing.ignore_symbolic_volume_limits {
        for keyed in dstg.ss.dims.dims().symbolic.volumes().keys() {
            if ldims.iter().any(|entry| keyed.contains(&entry)) {
                todo!(
                    "capacity::buffer_capacity_per_dim_at: dsc/dsc2.cpp:3780-3802 and :3809-3817 — \
                     {keyed:?} is a symbolic volume limit naming this tensor's layout dims \
                     (allowSymbolicVolumeLimit = {allowed}), so the limit has to be carried DOWN \
                     the layout order — every dim it names sized -1 until the last, which takes the \
                     limit itself — and the two partial-match aborts raised on the way",
                    allowed = form.allow_symbolic_volume_limit,
                );
            }
        }
    }

    // `getCumulativeStickSizes(myLds.dsType_)` (`:3804`).
    let stick_sizes = cumulative_stick_sizes(&dsc.stick_dims(at.lds), StickPart::Whole)?;
    // `myAllocNode->getPageSize()` (`:3806`) — EMPTY for the `NO_INDIRECTION` every program states.
    let page_size = alloc.page_sizes(sizing.indirect);
    let padded = padding_form(&alloc.placement.padding);
    let sample = DimSample {
        comp: sampled_as(at.comp),
        row: at.row,
        corelet: at.corelet,
    };

    let mut size_per_dim: Vec<(PrimaryDim, Extent)> = Vec::new();
    // `for (auto& entry : ldims)` (`:3808-3956`).
    for entry in ldims.iter() {
        let mut dim_size;
        // `myLds.scale_.at(getDimIndexInLayoutOrder(myLds.dsType_, entry))` (`:3820-3822`) AS ONE
        // LOOKUP, whose [`None`] is *"Invalid layoutDimOrder_ index."*.
        match at.info.scale(entry)? {
            // `scale == -1` (`:3824-3826`).
            Scale::UnitStick => {
                dim_size = Extent(1);
                if page_size.contains_key(&entry) {
                    return None;
                }
            }
            // `scale == -2` (`:3827-3829`) — `stickSizePerDim.at(entry)`, and that `.at` throws for a
            // dim the stick order does not name, where the rounding arm below reads 1 instead.
            Scale::StickDim => {
                let &(_, stick) = stick_sizes.iter().find(|&&(sized, _)| sized == entry)?;
                dim_size = Extent(i64::try_from(stick.0).ok()?);
                if page_size.contains_key(&entry) {
                    return None;
                }
            }
            Scale::Sized(_) => {
                // `hasCoordinate && !effectiveCoord.coreIdToWkSlice_.empty()` (`:3833-3835`), the
                // reference's own `FIXME` about using the coordinate only for a custom work slice.
                if has_coordinate && effective.wk_slices().next().is_some() {
                    todo!(
                        "capacity::buffer_capacity_per_dim_at: dsc/dsc2.cpp:3833-3877 — {entry:?} \
                         has a custom coreIdToWkSlice_, so its size is the PRODUCT of the \
                         cardinalities of every ELEM_ARR_COORD fold, plus the corelet and dummy \
                         rowsplit folds in LX and the core and dummy rowsplit folds in HBM; ⛔ \
                         dsc2::FoldDim::cardinality_at CANNOT reach those positions — its \
                         FoldPosition is the closed three, and this walks folds().enumerate()"
                    );
                }
                // `std::max(ssVal, elVal)` over `primaryDimToVal_st(entry, comp, row, corelet,
                // myAllocNode->padding_)` on both halves (`:3878-3891`).
                let ss = sampled_or_absent(dstg.ss.dims.dims(), entry, sample, &padded)?;
                let el = sampled_or_absent(dstg.el.dims.dims(), entry, sample, &padded)?;
                dim_size = Extent(ss.0.max(el.0));
                if page_size.contains_key(&entry) {
                    todo!(
                        "capacity::buffer_capacity_per_dim_at: dsc/dsc2.cpp:3893-3922 — {entry:?} \
                         is paged: an INDEX_TENSOR counts the pages it addresses, a VALUE_TENSOR \
                         takes one page (min for a fixed dim, the page itself for a symbolic one, \
                         whose granularity then divides the volume limit down)"
                    );
                }
                // `myLds.scaledLdsCategory_ == SCALE_TENSOR && entry == myLds.mxInfo_.dim`
                // (`:3924-3929`), which [`LabeledDs::scale_tensor`] already reads as the ONE pair.
                //
                // ⭐ PORTED, NOT DEFERRED, though `mxInfo_.blkSize` is 0 on all 807 g0 labelled DSs:
                // [`ScaleBlock`](crate::schedule::ddc::fold::ScaleBlock) is non-zero by construction,
                // so the divide is total and the arm costs a `todo!` a bundle outside g0 could walk
                // into — the reading [`StageDims::sampled_extent`] took for `peSfpSplit_`.
                if let Some(mx) = at.info.scale_tensor()
                    && mx.dim == entry
                {
                    dim_size = Extent(dim_size.0 / i64::try_from(mx.blk_size.count().0).ok()?);
                    // `DT_CHECK(dimSize != 0)` (`:3928`) — a block wider than the dim is a stop.
                    if dim_size.0 == 0 {
                        return None;
                    }
                }
                // `if (!doNotRound) if (auto it = stickSizePerDim.find(entry); ..)` (`:3930-3935`) —
                // and a dim the sticks do not name divides by 1, which is the identity.
                if !form.do_not_round {
                    dim_size = rounded_to_sticks(dim_size, stick_divisor(&stick_sizes, entry)?)?;
                }
            }
        }
        // `if (includeGaps && myAllocNode->backGapCore_.count(entry))` (`:3937-3955`).
        if form.include_gaps && sizing.back_gap_dims.contains(&entry) {
            todo!(
                "capacity::buffer_capacity_per_dim_at: dsc/dsc2.cpp:3937-3955 — {entry:?} carries a \
                 backGapCore_ gap to add to its size: HBM reads the `-1` key and every other \
                 component the first core's, which every core of coreIdsUsed_ must then agree with"
            );
        }
        // `sizePerDim.emplace_back(entry, dimSize + gap)` (`:3956`), whose `gap` is 0 until that arm
        // lands.
        size_per_dim.push((entry, dim_size));
    }

    // `for (auto gapStickSpread : myAllocNode->gapStickSpread_) for (auto& sizeDim : sizePerDim)`
    // (`:3958-3961`) — a TRUNCATING integer divide, as the reference's `int /=` is.
    for (&dim, &spread) in &alloc.gap_stick_spread {
        let spread = i64::try_from(NonZeroU64::new(spread.0)?.get()).ok()?;
        for sized in &mut size_per_dim {
            if sized.0 == dim {
                sized.1 = Extent(sized.1.0 / spread);
            }
        }
    }
    Some(size_per_dim)
}

/// `myAllocNode->padding_` AS `primaryDimToVal_st`'s `padding` ARGUMENT — one `PaddingFormType`
/// reached through the two Rust wrappers of it, [`Padding`] on the allocate node and [`PaddingForm`]
/// on the dims being read. Both answer `NOPAD` for a dim they do not name, so the copy is total.
fn padding_form(padding: &Padding) -> PaddingForm {
    let mut form = PaddingForm::default();
    for dim in padding.dims() {
        form.set_padding(dim, padding.get(dim));
    }
    form
}

/// `primaryDimToVal_st(entry, comp, row, corelet, myAllocNode->padding_)` (`dsc/dsc2.cpp:3885-3890`)
/// with the reference's `-1` TOLD APART FROM ITS THROWS, as [`full_span_with_unneeded`] does for e015.
///
/// ⛔ `-1` IS THE ANSWER THIS CALLER NEEDS AS A VALUE. Where no split and no symbol names `dim`,
/// [`StageDims::sampled_extent`] provably reduces to the raw slot through `calculate_padded`'s `if
/// (val < 0) return -1` (`dsc/dims.cpp:566-567`), so its [`None`] is unambiguously that `-1` — and
/// e015 writes exactly that slot for a dim its den stage stated nothing for.
///
/// ⚠️ A NARROWING, IN THE SAFE DIRECTION: where a split DOES name `dim` the [`None`] stays a stop,
/// because the `.at`s under it throw and the reference's `-1` is then unreachable anyway.
fn sampled_or_absent(
    dims: &StageDims,
    dim: PrimaryDim,
    at: DimSample,
    padded: &PaddingForm,
) -> Option<Extent> {
    if !dims.row_split.contains_key(&dim)
        && !dims.pe_sfp_split.contains_key(&dim)
        && !dims.corelet_split.contains_key(&dim)
        && !dims.symbolic.info().contains_key(&dim)
        && dims.extent(dim).is_none_or(|slot| slot.0 < 0)
    {
        return Some(Extent(-1));
    }
    dims.sampled_extent(dim, at, padded, None, false)
}

/// `std::ceil(float(dimSize) / it->second) * it->second` (`dsc/dsc2.cpp:3933`) IN INTEGERS.
///
/// ⭐ IT ROUNDS `-1` UP TO ZERO for any stick wider than one element, exactly as the reference's
/// `std::ceil(-0.015625) * 64` does, and e019 then reads both as `max(size, 1)`.
///
/// ⚠️ DIVERGENCE, AND IT IS THE EXACT DIRECTION: the reference rounds through a 32-bit `float`, which
/// cannot hold a stick count past 2^24 — this is exact for every size.
fn rounded_to_sticks(size: Extent, sticks: NonZeroU64) -> Option<Extent> {
    let sticks = i64::try_from(sticks.get()).ok()?;
    let up = size.0.saturating_neg().div_euclid(sticks).saturating_neg();
    Some(Extent(up.saturating_mul(sticks)))
}

#[cfg(test)]
mod tests_e017 {
    //! ⭐⭐ TWO CAPACITIES THE REFERENCE ITSELF RECORDED — `/Users/nickm/tmp/bridge1-fixtures/g0/
    //! debug/sdsc_1/sdsc.json`, DSC `rmmean_o728`, exported after ITS OWN L3/ddc/dcg ran.
    //!
    //! ⭐ WHERE THE REFERENCE WROTE ITS ANSWER DOWN, so nothing here is hand-transcribed:
    //! `bufferOffsetCoreCorelet_[core][cl] = kv.second / numBuffers_` where `kv.second == numBuffers_
    //! * getBufferCapacityForNode(allocNode, ldsIdx, component_, corelet, row)`
    //! (`ddc/ddcv1.cpp:244`, `:352-357`; `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5556-5561`,
    //! `:5658`). For an allocate node with `ldsIdx_ >= 0` and `numBuffers_ != -1` the EXPORTED OFFSET
    //! IS THE CAPACITY. (`numBuffers_ == -1` is excluded because `ddcv1.cpp:230` raises its size to
    //! the whole memory first; both nodes below state 2.)
    //!
    //! The export states `primaryDsInfo_["OUTPUT"] = {layoutDimOrder_: ["mb","out","y"],
    //! stickDimOrder_: ["out"], stickSize_: [64]}`, `dataStageParam_["1"]` (`name_: "chunk"`) `=
    //! {out_: 2048, mb_: 1, y_: 1}` with every split, symbol, volume and padding map EMPTY,
    //! `scheduleTreeHeadDenId_: 0`, and the tree `loop_ds0_ds1_y` (ROOT, `prev_: ""`, `denId_: 1`) →
    //! `loop_ds0_ds1_mb` (`denId_: 1`) → {`allocate_lds1_lx`, `loop_ds0_ds1_out` (`denId_: 1`) →
    //! `allocate_lds0_lx`}. So all three layout dims are sized by datastage 1.
    //!
    //! * `allocate_lds1_lx` — `ldsIdx_: 2`, `scale_: [1,-2,1]`, `wordLength: 2`, `numBuffers_: 2`,
    //!   `bufferOffsetCoreCorelet_: {"0": {"0": 256}}`. Per dim `[(mb,1),(out,64),(y,1)]`: `out` takes
    //!   the STICK, 64, and is NOT rounded. e019 then reads `64 * 2 = 128` bytes — an ODD number of
    //!   128-byte sticks, so L3's `forceEvenNumSticks` adds one (`dsc2.cpp:3996-4003`): **256**.
    //! * `allocate_lds0_lx` — `ldsIdx_: 0`, `scale_: [1,1,1]`, `wordLength: 2`, `numBuffers_: 2`,
    //!   `bufferOffsetCoreCorelet_: {"0": {"0": 4096}}`. Per dim `[(mb,1),(out,2048),(y,1)]`: `out`
    //!   comes from the CHUNK stage, `max(2048, 2048)`. e019 reads `2048 * 2 = 4096` — an even 32
    //!   sticks, so no bump: **4096**.
    //!
    //! ⭐ THE TWO DISCRIMINATE THE TWO LIVE SCALE ARMS AGAINST EACH OTHER: 64 is unreachable from the
    //! data-stage path and 2048 unreachable from the stick path, on the SAME stage and the SAME stick
    //! order — only `scale_` differs.
    //!
    //! ⚠️ WHAT THIS DOES NOT PIN: the rounding runs on all three `Sized` dims here and is the IDENTITY
    //! on each (1, 2048 and 1 are whole stick counts), so its ceiling is unwitnessed by g0 — no
    //! program of the 187 rounds a capacity dim UP.

    use std::collections::{BTreeMap, BTreeSet};

    use crate::arch::Elements;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, Coordinate, LayoutDims, MaxDimSize, NodeName, NumBuffers,
        StartAddress,
    };

    use super::super::dsc::{
        DataStage, DataStages, FilledDims, LabeledDs, NamedDims, Pinning, SenComponent, StageDims,
    };
    use super::tests_e015::dividing;
    use super::{
        AllocSizing, AllocateNode, AncestorLoops, CapacityForm, Dsc, LdsIdx, LdsSticks, Padding,
        SampledBuffer, SizeDsc, SizedNode, buffer_capacity_per_dim_at,
    };

    const MB: PrimaryDim = PrimaryDim::Mb;
    const OUT: PrimaryDim = PrimaryDim::Out;
    const Y: PrimaryDim = PrimaryDim::Y;

    /// One half of a stage — sdsc_1 states nothing but the three slots.
    fn half(name: &str, out: i64) -> NamedDims {
        let mut dims = StageDims::default();
        dims.extents.insert(OUT, Extent(out));
        dims.extents.insert(MB, Extent(1));
        dims.extents.insert(Y, Extent(1));
        NamedDims {
            name: StageName(name.to_owned()),
            dims: FilledDims::of(dims).expect("a stage stating three dims"),
        }
    }

    /// `sdsc_1`'s DSC through the five seams the capacity walk reads it by.
    struct Sdsc1 {
        stages: DataStages,
        tensor0: LabeledDs,
        tensor1: LabeledDs,
    }

    impl Sdsc1 {
        /// `dataStageParam_["0"]` (`core`) and `["1"]` (`chunk`) both `{out_: 2048, mb_: 1, y_: 1}`,
        /// plus `labeledDs_[0]` (`scale_: [1,1,1]`) and `labeledDs_[2]` (`scale_: [1,-2,1]`), each
        /// zipped onto `layoutDimOrder_: ["mb","out","y"]`.
        fn of() -> Self {
            let stage = |name: &str| DataStage {
                ss: half(name, 2048),
                el: half(name, 2048),
            };
            Self {
                stages: DataStages::new(stage("core"), stage("chunk")),
                tensor0: LabeledDs::new(
                    DsType::Output,
                    vec![
                        (MB, Scale::Sized(1.0)),
                        (OUT, Scale::Sized(1.0)),
                        (Y, Scale::Sized(1.0)),
                    ],
                    LdsIdx(0),
                    Pinning::default(),
                ),
                tensor1: LabeledDs::new(
                    DsType::Output,
                    vec![
                        (MB, Scale::Sized(1.0)),
                        (OUT, Scale::StickDim),
                        (Y, Scale::Sized(1.0)),
                    ],
                    LdsIdx(2),
                    Pinning::default(),
                ),
            }
        }
    }

    impl LdsSticks for Sdsc1 {
        /// `primaryDsInfo_["OUTPUT"]` — `stickDimOrder_: ["out"]`, `stickSize_: [64]`, no slice flags.
        fn stick_dims(&self, _lds: LdsIdx) -> StickDims {
            StickDims(vec![(OUT, Elements(64))])
        }
    }

    impl Dsc for Sdsc1 {
        /// `getLayoutDims(..)` — `["mb","out","y"]`, which both allocate nodes carry.
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            LayoutDims::new(MB, vec![OUT, Y])
        }
    }

    impl SizeDsc for Sdsc1 {
        /// `N_ = {"name_": "n", "out_": 2048, "mb_": 1, "y_": 1}` — unread here, since neither node
        /// is the HBM arm.
        fn whole_data_structure(&self) -> Option<NamedDims> {
            Some(half("n", 2048))
        }

        fn layout_dim_set(&self, _lds: LdsIdx) -> Option<BTreeSet<PrimaryDim>> {
            Some(BTreeSet::from([MB, OUT, Y]))
        }

        fn data_stages(&self) -> &DataStages {
            &self.stages
        }
    }

    /// `allocate_lds<n>_lx` as the export prints it: `component_: "lx"`, `numBuffers_: 2`,
    /// `padding_: {}`, `gapStickSpread_: {}`, `maxDimSizes_: [-1,-1,-1]`.
    fn lds_lx(name: &str, lds: LdsIdx) -> AllocateNode {
        AllocateNode {
            name: NodeName(name.to_owned()),
            component: SenComponent::Lx,
            lds: Some(lds),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new(
                (MB, MaxDimSize::Unset),
                vec![(OUT, MaxDimSize::Unset), (Y, MaxDimSize::Unset)],
            ),
            start_address: StartAddress::default(),
            placement: AllocPlacement {
                num_buffers: NumBuffers::Double,
                padding: Padding::default(),
                buffer_offset: BTreeMap::new(),
                is_start_addr_symbolic: false,
            },
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    /// e017 — the two per-dim sizings behind `sdsc_1`'s own two exported buffer offsets.
    #[test]
    fn the_stick_scaled_dim_takes_the_stick_and_the_plain_one_takes_the_datastage() {
        let dsc = Sdsc1::of();
        // `allocateCoordinates_` with `foldConstructed_: 0` and `coreIdToWkSlice_: {}`, which is what
        // all 1745 serialised coordinate blocks of g0 state.
        let coordinates = Coordinate::default();
        let no_gaps = BTreeSet::new();
        let sizing = AllocSizing {
            allocate_coordinates: &coordinates,
            slice_view_coordinates: None,
            ignore_symbolic_volume_limits: false,
            indirect: None,
            back_gap_dims: &no_gaps,
        };
        let head = Some(DatastageId(1));
        let mb_loop = dividing("loop_ds0_ds1_mb", MB, DatastageId(1));
        let out_loop = dividing("loop_ds0_ds1_out", OUT, DatastageId(1));

        // `allocate_lds1_lx`, whose `scale_.at(1)` is `-2`: `out` is the STICK.
        let lds1 = lds_lx("allocate_lds1_lx", LdsIdx(2));
        assert_eq!(
            buffer_capacity_per_dim_at(
                SizedNode::Allocate {
                    component: SenComponent::Lx,
                    non_unified_in_hbm: false,
                },
                &lds1,
                sizing,
                SampledBuffer::of(LdsIdx(2), &dsc.tensor1, SenComponent::Lx, None, None),
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(vec![&mb_loop], head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(64)), (Y, Extent(1))])
        );

        // `allocate_lds0_lx`, whose `scale_` is all ones: `out` is the CHUNK stage's 2048.
        let lds0 = lds_lx("allocate_lds0_lx", LdsIdx(0));
        assert_eq!(
            buffer_capacity_per_dim_at(
                SizedNode::Allocate {
                    component: SenComponent::Lx,
                    non_unified_in_hbm: false,
                },
                &lds0,
                sizing,
                SampledBuffer::of(LdsIdx(0), &dsc.tensor0, SenComponent::Lx, None, None),
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(vec![&out_loop, &mb_loop], head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(2048)), (Y, Extent(1))])
        );
    }
}

/// WHICH ALLOCATION ONE NODE SIZES WHEN IT IS BOTH THE LOCATION AND THE SUBJECT — `node->nodeType_ ==
/// ALLOCATE ? node : myLds.memOrg_.at(comp).allocateNode_` (`dsc/dsc2.cpp:3762-3764`) reached with
/// `nodeForLocation == node` (`:3972-3974`), which is what collapses e017's two nodes into one.
///
/// ⛔ THE `.at(comp)` AND A NULL `allocateNode_` ARE THE CALLER'S STOP: [`MemOrg`] hands back no
/// `dsc2::AllocateNode` for an arbitrary component, so the caller resolves that cell and NAMES which
/// of the two it resolved by picking a variant — the same reason [`AllocSizing`] is a parameter.
///
/// [`MemOrg`]: super::dsc::MemOrg
#[derive(Debug, Clone, Copy)]
pub enum CapacityNode<'a> {
    /// `nodeType_ == ALLOCATE`, where `static_cast<const dsc2::AllocateNode*>(node)` IS the node
    /// stating the location, so the sizing arm reads this very allocation's own `component_`.
    Allocation {
        /// The allocation, which is also the node.
        alloc: &'a AllocateNode,
        /// The five members [`AllocSizing`] carries for it.
        sizing: AllocSizing<'a>,
        /// `nonUnifiedAllocInHBM_`, measured `false` on all 1899 allocate nodes of g0.
        non_unified_in_hbm: bool,
    },
    /// Every other `nodeType_` — a transfer, compute, loop, sync or condition states the location and
    /// `myLds.memOrg_.at(comp).allocateNode_` is the allocation sized there.
    MemOrgOf {
        /// `myLds.memOrg_.at(comp).allocateNode_`, A DIFFERENT NODE from the one stating the location,
        /// which is why its `component_` is never the sizing arm.
        alloc: &'a AllocateNode,
        /// The five members [`AllocSizing`] carries for it.
        sizing: AllocSizing<'a>,
    },
}

/// Replaces: e018_getBufferCapacityForNodePerDim
///
/// [`buffer_capacity_per_dim_at`] SIZED AT THE NODE'S OWN LOCATION — the ordinary spelling, where the
/// single node handed over both says where the sizing happens and which allocation is sized.
///
/// ⛔ THE SIZING ARM IS DERIVED AND NOT SUPPLIED, AND THAT IS THE WHOLE CONTENT OF THE TEN LINES
/// (`dsc/dsc2.cpp:3966-3975`): only an ALLOCATE node reaches the HBM arm, off its OWN `component_`
/// (`:3626`), so an HBM allocation at the tree root is sized whole as `N_` while every other node kind
/// sizes its memOrg allocation by the loops above ITSELF. Handing that memOrg node's component over as
/// the arm would size a transfer at the root as whole HBM, and one loop deeper would stop outright on
/// *"HBM allocation should be at root of schedule tree"* (`:3628`).
/// ⛔ [`None`] IS EVERY STOP OF [`buffer_capacity_per_dim_at`], and `Extent(-1)` is still a size.
#[must_use]
pub fn buffer_capacity_per_dim(
    node: CapacityNode<'_>,
    at: SampledBuffer<'_>,
    form: CapacityForm,
    ancestors: &AncestorLoops<'_>,
    dsc: &(impl SizeDsc + ?Sized),
) -> Option<Vec<(PrimaryDim, Extent)>> {
    let (sized, alloc, sizing) = match node {
        CapacityNode::Allocation {
            alloc,
            sizing,
            non_unified_in_hbm,
        } => (
            SizedNode::Allocate {
                component: alloc.component,
                non_unified_in_hbm,
            },
            alloc,
            sizing,
        ),
        CapacityNode::MemOrgOf { alloc, sizing } => (SizedNode::Other, alloc, sizing),
    };
    buffer_capacity_per_dim_at(sized, alloc, sizing, at, form, ancestors, dsc)
}

#[cfg(test)]
mod tests_e018 {
    //! ⭐⭐ THE CAPACITY `sdsc_14`'s OWN EXPORT RECORDED, ASKED THE ORDINARY WAY —
    //! `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_14/sdsc.json`, DSC `14_t729_fq_afp8_op`, over
    //! [`super::tests_e015::Sdsc14`]'s four seams and its own `scheduleTreeHeadDenId_: 0`. All four
    //! cases are ONE labelled DS, `labeledDs_[1]` — `dsType_: "OUTPUT"`, `scale_: [1,1,1]`,
    //! `wordLength: 1`, `primaryDsInfo_["OUTPUT"].stickSize_: [128]` — so nothing but the node moves.
    //!
    //! * `allocate_lds1_lx` (`component_: "lx"`, `numBuffers_: 2`, hanging off `loop_ds0_ds1_mb` →
    //!   `loop_ds0_ds1_out` → `loop_ds0_ds1_y`, each `denId_: 1`) is `[(mb,1),(out,128),(y,1)]`: `out`
    //!   is the CHUNK stage, already a whole stick. ⭐ WHERE THE REFERENCE WROTE THAT DOWN: e019 reads
    //!   `128 * wordLength 1 = 128` bytes, an ODD number of 128-byte sticks, so L3's
    //!   `forceEvenNumSticks` adds one (`dsc/dsc2.cpp:3996-4003`) — **256**, which is exactly the
    //!   `bufferOffsetCoreCorelet_` the export prints on that node for all 16 cores.
    //! * `allocate-Tensor1_hbm` (`component_: "hbm"`, `prev_: ""`) is `[(mb,1),(out,2048),(y,1)]`:
    //!   `out` is `N_`, and `N_.out_` is 2048 in that same export.
    //! * that same HBM node sampled at `SenComponent::Lx` — the `L3DlOpsScheduler.cpp:5886` spelling,
    //!   where the sampled component is NOT the one the allocation sits in — is STILL
    //!   `[(mb,1),(out,2048),(y,1)]`, because the sizing arm is the allocation's OWN `component_`.
    //!
    //! ⭐⭐ THE NEGATIVE CONTROL IS THE DERIVATION ITSELF, not a second fixture: the last case hands
    //! `allocate-Tensor1_hbm` over as a memOrg allocation at the SAME root position with the SAME
    //! sample, and `out` drops from 2048 to the CORE stage's 128 — `dataStageParam_["0"].ss_.out_`,
    //! which the head's own `denId_: 0` selects. Only the variant differs, and neither number is
    //! reachable from the other arm.

    use std::collections::{BTreeMap, BTreeSet};

    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, Coordinate, MaxDimSize, NodeName, NumBuffers, StartAddress,
    };

    use super::super::dsc::{LabeledDs, Pinning, SenComponent};
    use super::tests_e015::{Sdsc14, dividing};
    use super::{
        AllocSizing, AllocateNode, AncestorLoops, CapacityForm, CapacityNode, LdsIdx, Padding,
        SampledBuffer, buffer_capacity_per_dim,
    };

    const MB: PrimaryDim = PrimaryDim::Mb;
    const OUT: PrimaryDim = PrimaryDim::Out;
    const Y: PrimaryDim = PrimaryDim::Y;

    /// One of `sdsc_14`'s `labeledDs_[1]` allocate nodes as the export prints it: `layoutDimOrder_:
    /// ["mb","out","y"]` with `maxDimSizes_: [-1,-1,-1]`, `padding_: {}` and `gapStickSpread_: {}`.
    fn tensor1(name: &str, component: SenComponent, num_buffers: NumBuffers) -> AllocateNode {
        AllocateNode {
            name: NodeName(name.to_owned()),
            component,
            lds: Some(LdsIdx(1)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new(
                (MB, MaxDimSize::Unset),
                vec![(OUT, MaxDimSize::Unset), (Y, MaxDimSize::Unset)],
            ),
            start_address: StartAddress::default(),
            placement: AllocPlacement {
                num_buffers,
                padding: Padding::default(),
                buffer_offset: BTreeMap::new(),
                is_start_addr_symbolic: false,
            },
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    /// e018 — the two positions `sdsc_14` states for one labelled DS, and the arm the node picks.
    #[test]
    fn one_node_states_both_the_location_and_the_allocation_it_sizes() {
        let dsc = Sdsc14::of();
        // `labeledDs_[1]`: `scale_: [1,1,1]` zipped onto `layoutDimOrder_: ["mb","out","y"]`.
        let tensor1_lds = LabeledDs::new(
            DsType::Output,
            vec![
                (MB, Scale::Sized(1.0)),
                (OUT, Scale::Sized(1.0)),
                (Y, Scale::Sized(1.0)),
            ],
            LdsIdx(1),
            Pinning::default(),
        );
        // `allocateCoordinates_` with `foldConstructed_: 0`, `ignoreSymbolicVolumeLimits_: 0`,
        // `indirectAllocType_: "no_indirection"` and `backGapCore_: {}` — all four nodes state these.
        let coordinates = Coordinate::default();
        let no_gaps = BTreeSet::new();
        let sizing = AllocSizing {
            allocate_coordinates: &coordinates,
            slice_view_coordinates: None,
            ignore_symbolic_volume_limits: false,
            indirect: None,
            back_gap_dims: &no_gaps,
        };
        let head = Some(DatastageId(0));

        // `getBufferCapacityForNodePerDim(allocate_lds1_lx, 1, LX, -1, -1)`, three `denId_: 1` loops
        // above it: the CHUNK stage's 128, which the exported offset 256 is `forceEvenNumSticks` of.
        let mb_loop = dividing("loop_ds0_ds1_mb", MB, DatastageId(1));
        let out_loop = dividing("loop_ds0_ds1_out", OUT, DatastageId(1));
        let y_loop = dividing("loop_ds0_ds1_y", Y, DatastageId(1));
        let lx = tensor1("allocate_lds1_lx", SenComponent::Lx, NumBuffers::Double);
        assert_eq!(
            buffer_capacity_per_dim(
                CapacityNode::Allocation {
                    alloc: &lx,
                    sizing,
                    non_unified_in_hbm: false,
                },
                SampledBuffer::of(LdsIdx(1), &tensor1_lds, SenComponent::Lx, None, None),
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(vec![&mb_loop, &out_loop, &y_loop], head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(128)), (Y, Extent(1))])
        );

        // `getBufferCapacityForNodePerDim(allocate-Tensor1_hbm, 1, HBM, -1, -1)` at the tree root:
        // `N_`, because THIS node is the allocation and its own `component_` is `hbm`.
        let hbm = tensor1(
            "allocate-Tensor1_hbm",
            SenComponent::Hbm,
            NumBuffers::Single,
        );
        let at_hbm = SampledBuffer::of(LdsIdx(1), &tensor1_lds, SenComponent::Hbm, None, None);
        assert_eq!(
            buffer_capacity_per_dim(
                CapacityNode::Allocation {
                    alloc: &hbm,
                    sizing,
                    non_unified_in_hbm: false,
                },
                at_hbm,
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(Vec::new(), head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(2048)), (Y, Extent(1))])
        );

        // ⭐ THE SECOND CONTROL, AND A REAL SPELLING: `getBufferCapacityForNodePerDim(indAllocation,
        // indexLdsIdx, indirectLoc->storage_, -1, -1, true)` (`L3DlOpsScheduler.cpp:5886`) samples at a
        // component the allocation itself is NOT in. The arm comes off `component_` (`dsc2.cpp:3626`),
        // so the SAME hbm node sampled at LX is still `N_` — reading the arm off the sample says 128.
        assert_eq!(
            buffer_capacity_per_dim(
                CapacityNode::Allocation {
                    alloc: &hbm,
                    sizing,
                    non_unified_in_hbm: false,
                },
                SampledBuffer::of(LdsIdx(1), &tensor1_lds, SenComponent::Lx, None, None),
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(Vec::new(), head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(2048)), (Y, Extent(1))])
        );

        // THE CONTROL: the SAME allocation, the SAME position, the SAME sample — reached as
        // `myLds.memOrg_.at(HBM).allocateNode_` beside a node that is NOT an allocate. The HBM arm is
        // out of reach, so `out` is the head's own datastage 0, the CORE stage: 128 and not 2048.
        assert_eq!(
            buffer_capacity_per_dim(
                CapacityNode::MemOrgOf {
                    alloc: &hbm,
                    sizing,
                },
                at_hbm,
                CapacityForm::DEFAULTS,
                &AncestorLoops::of(Vec::new(), head),
                &dsc,
            ),
            Some(vec![(MB, Extent(1)), (OUT, Extent(128)), (Y, Extent(1))])
        );
    }
}
