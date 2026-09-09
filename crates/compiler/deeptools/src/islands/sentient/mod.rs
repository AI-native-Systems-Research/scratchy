//! THE SENTIENTIR ISLAND — the rung where the datapath is explicit but the registers are not yet
//! numbered.
//!
//! ```text
//! DataflowIR ──D1–D28──► SentientIR ──D29–D75 (in place)──► ──D76──► ProgIR
//!                          (here)
//! ```
//!
//! ⭐ WHERE THIS SITS, MEASURED. `dbo/docs/pass_pipeline.md` on the pod names every pass D1-D76;
//! D1-D28 is `buildDSCToSentientIRPipeline`, D29-D75 rewrite SentientIR in place (the prologue,
//! `O2Pipeline` run as three rounds, and `registerManagementPasses`), and D76 `SentientToProgIR`
//! leaves MLIR entirely.
//!
//! ⛔⛔ AND THE RUNG IS **MIXED**, NOT A CLEAN DIALECT SWAP — see [`dialects`] for the dump that shows
//! `dataflow.*` and `agen.*` ops alive alongside `sentient.*` after the conversion named for them.

pub mod dialects;
pub mod print;
pub mod ty;

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::dataflow;
use crate::islands::dataflow_ir::{KernelName, ProgramName, Units};
use crate::model::Model;
use crate::units::DfirUnit;
use crate::workload::Workload;
use dialects::Op;

/// ONE `dataflow.program_unit` AT THIS RUNG — the units it runs on, and what they run.
///
/// # 🛑 THE PER-UNIT STRUCTURE SURVIVES THE WHOLE RUNG, AND THAT IS MEASURED
///
/// ⛔⛔ THIS ISLAND'S `Program` HELD A FLAT `Vec<Op>` AND THAT WAS WRONG. `dataflow.program_unit`
/// appears in **657 of the 668** `CHECK-SENT-IR` expectations in `dcc/test/` — IBM's own statement of
/// what SentientIR looks like — including the smallest complete program they ship
/// (`dcc/test/PT/xrfbmm_int8_fwd.mlir:18`:
/// `dataflow.program_unit iter_arg : %5 -> (%2) {precision = "int8"} : {`). So a Sentient program is
/// still a set of per-unit programs, exactly as ProgIR is
/// ([`crate::islands::progir::Program::per_unit`]) and as the rung below is.
///
/// ⛔⛔ AND A FLAT LIST COULD NOT STATE WHICH UNIT AN OP RUNS ON, so it could not refuse a placement
/// the backend refuses. `Helper.cpp:2177-2179` admits an `agen.composite_load_and_store` on an L3 half
/// ONLY, with a bare `LogicalResult::failure()` and no message — dbo-opt then printed just the
/// caller's wrapper, *"Unable to generate loops and sentient statements for the composite vector
/// operations"*, and that was read as being about the transfer's shape rather than its unit. An
/// island with no unit binding cannot catch that; this one can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramUnit<A: Arch> {
    /// The units this runs on — ⭐ [`Units`] IS THE RUNG BELOW'S, reused: it already refuses a list
    /// mixing kinds and cannot be empty, and those facts do not change on the way down.
    pub on: Units,
    /// `precision =`, present only where the unit computes.
    pub precision: Option<dataflow::Precision>,
    /// What it runs.
    pub body: Vec<Op>,
    /// The arch it was lowered for.
    pub arch: core::marker::PhantomData<A>,
}

impl<A: Arch> ProgramUnit<A> {
    /// WHETHER THIS UNIT MAY RUN A COMPOSITE MEMORY-TO-MEMORY TRANSFER.
    ///
    /// ⛔ ONLY THE L3 HALVES (`Helper.cpp:2177-2179`), and the refusal is SILENT — see the type's
    /// note. Exhaustive, with no wildcard: a new unit kind must say whether it is an L3 half rather
    /// than inherit `false`.
    #[must_use]
    pub fn moves_memory(&self) -> bool {
        matches!(self.on.kind(), DfirUnit::L3lu | DfirUnit::L3su)
    }
}

/// A PROGRAM'S UNITS — NON-EMPTY BY CONSTRUCTION.
///
/// ⭐ THE HEAD IS A FIELD, NOT AN INDEX, for the same reason as the rung below: `units.is_empty()` is
/// not a question that can be asked, so "found no program to compile" is unreachable from our side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramUnits<A: Arch> {
    head: ProgramUnit<A>,
    rest: Vec<ProgramUnit<A>>,
}

impl<A: Arch> ProgramUnits<A> {
    /// A program's units, the first being what makes it a program at all.
    #[must_use]
    pub fn of(head: ProgramUnit<A>, rest: Vec<ProgramUnit<A>>) -> ProgramUnits<A> {
        ProgramUnits { head, rest }
    }

    /// Every unit, head first.
    pub fn iter(&self) -> impl Iterator<Item = &ProgramUnit<A>> {
        core::iter::once(&self.head).chain(self.rest.iter())
    }

    /// Every unit, head first, FOR A REWRITE.
    ///
    /// ⛔ THE IN-PLACE PASSES NEED THIS AND [`Self::iter`] CANNOT SERVE THEM. `LexicalOrdering::runOn`
    /// (`dcc/src/Transform/Sentient/LexicalOrdering.cpp:113`) hoists constants OUT of unit bodies and
    /// rewires their readers, and a shared iterator can do neither.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ProgramUnit<A>> {
        core::iter::once(&mut self.head).chain(self.rest.iter_mut())
    }
}

/// ONE PROGRAM AT THE SENTIENT RUNG.
///
/// # 🛑 THREE CONST-GENERIC TRAITS, AND EACH ONE IS A GUARD
///
/// ⛔⛔ NOT DECORATION, AND THE RUNG BELOW LEARNED THIS THE EXPENSIVE WAY. Its `Program` carries
/// `A: Arch` because `Dd2` and `Sen1p5` both exist in every build — only `Target` is
/// feature-selected — so without it a program lowered against DD2's eight PT rows and one lowered
/// against SEN1P5's four are the SAME TYPE and can be put in one [`Run`].
///
/// ⭐ `M: Model` AND `W: Workload` MATTER MORE HERE THAN BELOW, because this is the rung where their
/// consequences become instructions. `Exploit<A, M, W>`'s flags decide which ops EXIST — `IS_DECODE`
/// removes the row nest, `FITS_LX` the tiling loop, `STICK_ALIGNED` the mask and every
/// `element_wise_selection` that consumes it — so a program that forgot which rung it was baked for
/// would be an instruction sequence nothing could place.
///
/// ⚠️ AND THEY ARE CURRENTLY CARRIED, NOT READ. An island is a representation and should not branch on
/// a workload; the BRIDGE is what reads those flags and emits different ops. Until it does, these
/// parameters are expressed and not exploited — which this crate's rules name as the thing three
/// versions were thrown away for, so it is worth saying rather than leaving to be discovered.
///
/// ⚠️ **TODO(quant):** a fourth parameter `Q: Quant` belongs here. There is no `Quant` trait in this
/// crate yet, and the preset it would encode is real — `quant/fp8-dynamic-per-channel` is one of the
/// acceptance build's own features, and its JSON states weight `num_bits`, `type`, `strategy`
/// (`channel`) and `symmetric`, plus a separate `input_activations` block. Those are what a compute's
/// `opAPrecision`/`ComputePrecision` should be read from rather than defaulted to fp16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<A: Arch, M: Model, W: Workload> {
    /// The module's symbol — ⭐ THE RUNG BELOW'S TYPE, because a program keeps its name as it is
    /// lowered. Minting a second name would make the two rungs' artifacts uncorrelatable.
    pub name: ProgramName,
    /// The preamble: the units and views the program declares, before any unit runs.
    pub preamble: Vec<Op>,
    /// THE PROGRAM UNITS — AT LEAST ONE, AND `Vec<Op>` CANNOT SPELL THAT. See [`ProgramUnit`].
    pub units: ProgramUnits<A>,
    /// The arch, model and rung this was lowered for.
    pub bound: core::marker::PhantomData<(A, M, W)>,
}

/// A WHOLE RUN AT THE SENTIENT RUNG — ⛔ ALL PROGRAMS ON ONE ARCH, MODEL AND RUNG, by the type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run<A: Arch, M: Model, W: Workload> {
    /// The kernel's name.
    pub kernel: KernelName,
    /// The programs, in the order they run.
    pub programs: Vec<Program<A, M, W>>,
}
