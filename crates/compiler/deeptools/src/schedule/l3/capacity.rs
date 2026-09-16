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

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickPart, cumulative_stick_sizes,
};
use crate::schedule::ddc::fold::PadType;
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation_util::StageName;
use crate::schedule::ddc::v1::LdsSticks;
use crate::schedule::dsc2::{Dsc, LdsIdx, LoopNode, Padding};

use super::dsc::{
    DataStage, DataStages, FilledDims, NamedDims, PadElems, SenComponent, StageDims, StatedVolumes,
    Symbolic, SymbolicDimInfo, UnneededPad, VolumeLimit,
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
    fn parametric(name: &str, dim: PrimaryDim) -> LoopNode {
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
    fn dividing(name: &str, dim: PrimaryDim, den: DatastageId) -> LoopNode {
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
    struct Sdsc14 {
        stages: DataStages,
    }

    impl Sdsc14 {
        /// `dataStageParam_` as the export prints it: `"0"` is `core`, `"1"` is `chunk`, `"2"` is the
        /// stage `loop_ds1_ds2_y` divides by — whose `paddingSizes_` the export voids.
        fn of() -> Self {
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
