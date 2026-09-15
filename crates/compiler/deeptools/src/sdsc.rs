// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ BRIDGE 1'S SCHEDULING SEAM — the ONE surface a caller OUTSIDE this crate names, and the only
//! one it is allowed to.
//!
//! ```text
//! <a target's own SuperDSC wire form>
//!     ──the target's conversion, speaking ONLY the names below──► SuperDsc + the three stage arguments
//!     ──run_stages_2a_2b───────────────────────────────────────► Scheduling  (the SCHEDULED tree)
//! ```
//!
//! # ⛔⛔ WHY THIS MODULE EXISTS: `scratchy knows nothing about l3`
//!
//! The root `CLAUDE.md` is explicit — *"Everything common is shared. One implementation of every
//! target-neutral pass"* and *"Per-target surface = opcode lowering only."* The scheduler is
//! target-neutral by definition, so a target crate reaching into
//! [`crate::schedule::l3::dsc`], [`crate::schedule::dsc2`], [`crate::schedule::ddc::fold`],
//! [`crate::schedule::ddc::v1`], [`crate::schedule::ddc::transformation`] and
//! [`crate::schedule::stages`] — nine distinct internal paths, which is what
//! `crates/targets/spyre/src/superdsc_to_l3_sdsc.rs` did — is the boundary being absent rather than
//! declared. **This module IS the declaration.**
//!
//! ⭐ AND IT IS A BOUNDARY, NOT A SECOND IMPLEMENTATION. Every item below is a `pub use`; nothing is
//! defined here and no fact is restated. That is the point: the scheduler's internals can be
//! rearranged — `l3::dsc` merged, `dsc2` renamed, `stages` split — and no target crate changes,
//! because a target names only what this file re-exports. Adding logic here would make it a second
//! place the vocabulary lives, which is the drift this crate already paid for once
//! (`Isa::InstOperand`, 89 enumerators kept in step by a hand-written assertion).
//!
//! ⛔⛔ WHAT IT IS *NOT*: it is NOT the file relocation. The wire→[`SuperDsc`] conversion still lives
//! in the spyre crate and still speaks both vocabularies; moving it in here needs the SuperDSC wire
//! DTOs to move too, and that is blocked twice — they are `serde`/`serde_json::Value` types and this
//! crate has NO serde dependency (its `Cargo.toml` lists only `sys-arch-spec`), and two of their
//! fields are typed by `scratchy_subtile` (`SliceIndex` on `coreIdToWkSlice_`, `DeviceWalk` on
//! `AllocNode::maxDimSizes_`), which this crate may never depend on. The relocation is owed; this
//! module is what makes it a MOVE rather than a rewrite of nine import paths.
//!
//! ⛔ NOTHING HERE WIDENS `pub`. Every re-export is already `pub` on its own module, so this adds no
//! access a target did not have — it CONCENTRATES it, so that `grep deeptools::schedule
//! crates/targets/` is the audit and its answer is zero.
//!
//! # ⭐ THE THREE STAGE ARGUMENTS, AND WHY A CALLER STATES THEM
//!
//! [`run_stages_2a_2b`] takes the super-DSC plus three things that are NOT fields of
//! [`DesignSpaceConfig`] or [`SuperDsc`], so no carrier can read them off the value in hand:
//!
//! * [`OpFuncs`] — `computeOp_[i].opFuncName`, which the min-param units specialise on.
//! * `&[Vec<DscComputeOp>]` — `computeOp_` itself, POSITIONALLY beside `sdsc.dscs()`. ⛔ AN EMPTY
//!   LIST IS A FALSE GREEN: [`crate::schedule::ddc::v1::PrepDsc::compute_ops`] is the FIRST provider
//!   call `run_v1` makes (`ddc/v1.rs:6437`) and an empty answer makes it `continue` past the DSC, so
//!   the stage answers [`DscFilled::Yes`] having placed no address and minted no node.
//! * `&[StorageName]` — `dsc.name_`, which only the two verbose lines read.

/// ⭐ THE DIM VOCABULARY A SuperDSC IS WRITTEN IN — `PrimaryDimTypes` (`dsc/dims.h:34`) and the two
/// extent newtypes beside it. ⛔ RE-EXPORTED HERE BECAUSE THEY LIVE INSIDE A BRIDGE'S OWN MODULE
/// (`bridges::superdsc_to_dataflow_ir::shape_constraints`) AND NOT ON A TOP-LEVEL SURFACE: a target
/// needing them was reaching past a bridge's boundary as well as the scheduler's.
pub use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
pub use crate::schedule::ddc::fold::{ConstIdx, NodeKind, ScaledLds};
pub use crate::schedule::ddc::transformation::{DsType, Scale};
pub use crate::schedule::ddc::transformation_util::StageName;
pub use crate::schedule::ddc::v1::{DscComputeOp, DscFilled, L0Tethered, OpFuncs, StorageName};
/// ⭐ THE OP-FUNC ITSELF COMES FROM THE ARCH SPEC, NOT FROM `schedule` — `sys-arch-spec` is the crate
/// BOTH the compiler and the senulator include, and [`OpFunc::from_spelling`] is
/// `EnumsConversion::stringToOpFuncs`, the reference's own parse boundary. Re-exported here so a
/// caller reads `opFuncName` through the same seam it builds everything else through.
pub use sys_arch_spec::arch_enums::OpFunc;
pub use crate::schedule::dsc2::{LayoutDims, LdsIdx, WordLength};
pub use crate::schedule::l3::dsc::{
    ConstantInfo, CoreIdsUsed, CoreletShare, CoreletsUsed, DATA_STAGE_CORE, DataStage, DataStages,
    DdcFacts, DesignSpaceConfig, DimPadding, DscIdx, DscList, DscScheduleStep, FilledDims, LabeledDs,
    LabeledDsList, LdsRecord, NamedDims, Pinning, PrimaryDsInfo, SenComponent, StageDims, SuperDsc,
    WkSlice, WkSliceCount, WkSliceId,
};
pub use crate::schedule::stages::{Scheduling, StagesRan, run_stages_2a_2b};

// ⭐⭐ THE SCHEDULE TREE ITSELF, so a LOWERING can walk the typed tree instead of the wire's
// `nodeType_` STRING. [`Scheduling::state`] hands out a [`DscState`]; [`DscState::dscs`] hands out one
// [`DscTree`] per DSC, and that is the object carrying the nodes stages 2a and 2b minted.
//
// ⛔⛔ THREE MORE ROWS ARE OWED AND ARE DELIBERATELY ABSENT: `TreeData`, `Kind` and `Cond`
// (`schedule/stages/tree.rs`). A lowering that walks the tree must match on `Kind` —
// `ScheduleNode::NodeType` as the closed set it is — or the only thing it can read is the wire's
// `nodeType_` STRING, which is the reach-in this seam exists to remove. They need TWO edits in files
// this module does not own: `pub(super)` -> `pub` in `stages/tree.rs`, and a
// `pub use tree::{Cond, Kind, TreeData};` in `stages.rs` (whose `mod tree;` is private).
//
// ⛔ AND THEY ARE ABSENT BECAUSE THOSE EDITS ARE NOT COMMITTED YET, NOT BECAUSE THEY ARE UNWANTED. The
// row below is exactly what to add once they are, and adding it FIRST would red the whole crate on a
// clean checkout while passing in a working tree that happens to hold the other agent's changes —
// which is a green caused by somebody else's build.
//
//     pub use crate::schedule::stages::{Cond, Kind, TreeData};
pub use crate::schedule::stages::{DscState, DscTree};
