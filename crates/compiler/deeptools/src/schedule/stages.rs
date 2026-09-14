// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE FOUR SCHEDULER STAGES, COMPOSED AND CALLABLE — `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-57`.
//!
//! Campaign 6 ported 382 units of these stages and NOTHING CALLED ANY OF THEM. Every entry point
//! takes its state through trait parameters — `L3RunInputs<F, P>` / `L3RunSurgery<E, M, K, S>` for
//! stage 2a, `Dsc2Sites` for stage 2b — and no concrete type implemented them, so `F` and `E` were
//! uninhabited and the stages were dead code. THIS MODULE IS THOSE CONCRETE TYPES. Writing them is
//! the integration, not a second port: the bodies exist, this is what reaches them.
//!
//! # THE ORDER, AND WHY EACH STAGE IS OR IS NOT HERE
//!
//! 1. `sbf::doCoreletSplitSdsc` — ⛔ EXCLUDED, and not by us: `SchedulerStages.cpp:25` returns early
//!    unless `numCoreletsPerCore == 2`, and scratchy emits `numCoreletsUsed_ = 1` in all 313 sampled
//!    SuperDSCs. Recorded in `crustify-ddc/EXCLUSIONS.tsv`.
//! 2. `L3DlOpsScheduler::run` — [`l3::dl_ops::run`], `e382_run`.
//! 3. `ddc::Ddc::run_v1` — [`ddc::v1::run_v1`], `e379_run_v1`. THE STAGE THAT PLACES ADDRESSES.
//! 4. `DcgManager::runDcgForDlOpsStandalone` — ⛔ NOT WIRED, ON PURPOSE. Its body delegates to
//!    `dcg_fe/pcfg_gen/` and `dcg_be/`, both declared out of campaign 6's scope, and it fills
//!    `pcfg_[core][L3LU]/[L3SU]` — the PCFG path this project has ruled off, because our DL path
//!    emits those units directly (`bridges/superdsc_to_dataflow_ir/transfer.rs` carries 61 L3LU/L3SU
//!    references). Porting `pcfg_gen` to reach it is forbidden, so it stays a `todo!` inside
//!    `dcg::manager` and this module does not call it.
//!
//! # ⛔⛔ THE ONE STRUCTURAL FACT A CALLER MUST WORK AROUND — REVIEW 382'S OWN NOTE
//!
//! `L3RunInputs::reads` is `&'a F` and `L3RunSurgery::env` is `&'a mut E`. A shared and an exclusive
//! borrow cannot view ONE object, so the reference's single `this` is SEVERED by the port's own
//! signature — and its review says so at `l3/dl_ops.rs:20556-20560`: the placed address (`:5687`
//! writes, `:4971`/`:5034` rewrite, `:5816` reads back) and the minted tree (290/353/368 write
//! `env`; 289/295/332 read `reads`) are both chains that cross the cut.
//!
//! ⭐ SO THE TWO CARRIERS SHARE ONE STATE THROUGH A CELL, WHICH IS THE HONEST REPAIR. [`Reads`] and
//! [`Env`] are two objects, as the signature demands, each holding `&'s DscState` — and the state's
//! mutable interior is what restores the reference's aliasing. It is NOT a workaround that changes
//! behaviour: it makes the two views name the same tree and the same placed address, which is what
//! the reference does and what any faithful caller must do. Unifying the carriers in the port itself
//! is the cross-entry work review 382 names over entries 050/219/220/222/292/333; until that lands,
//! the cell is what a caller has.
//!
//! # ⛔ NO FABRICATION, ANYWHERE
//!
//! Every provider method answers from the SuperDSC it was given. `dxp_standalone` and `dbo-opt` run
//! THESE SAME FOUR STAGES over scratchy's SuperDSC today and produce a working `init_binary`
//! (`RunSchedulerOnSdsc.cpp:144-170`), so every fact these methods are asked for is already present
//! in what scratchy writes — there is no data gap, by construction. A method that cannot yet be
//! answered is a `todo!` NAMING THE TRAIT AND METHOD, never a plausible constant: a fabricated
//! placement is the failure this crate ranks worse than a stop.

mod carriers;
mod env;
mod state;
mod tree;

pub use carriers::{Placement, Sink, Symbols, Trackers};
pub use env::Env;
pub use state::{DscState, DscTree};

use super::l3;

/// ⭐ WHAT STAGE 2A LEFT — the answer the caller measures, and the reason this returns a value rather
/// than printing one.
///
/// ⛔ [`None`] IS A CALLEE'S OWN STOP, propagated, not an error we invent: `l3::dl_ops::run` answers
/// `Option<()>` and every `None` in it is a ported unit's refusal or *"Memory allocation must be
/// valid to commit."*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StagesRan {
    /// Whether stage 2a completed.
    pub l3: bool,
    /// Whether stage 2b completed.
    pub ddc: bool,
}

/// ⛔ PLACEHOLDER FOR THE COMPOSED ENTRY POINT — the providers below are what it needs, and they are
/// being written trait by trait. This exists so the module has its shape from the first commit and so
/// `schedule::stages` is the ONE name a caller ever learns.
#[must_use]
pub fn run_stages(_sdsc: &mut l3::dsc::SuperDsc) -> StagesRan {
    // ⛔ NOT YET COMPOSED. `l3::dl_ops::run`'s `F`/`P`/`E`/`M`/`K`/`S` and `ddc::v1::run_v1`'s
    // `Dsc2Sites` are 61 and ~41-supertrait surfaces respectively; each lands with its own commit and
    // its own census reading. Answering `StagesRan::default()` is the truthful report that neither
    // has run, and it keeps this function total — a `todo!` here would stop the bake for every group
    // before dbo-opt, which is the one thing this crate forbids.
    StagesRan::default()
}
