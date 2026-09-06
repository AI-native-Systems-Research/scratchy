//! THE SENTIENTIR ISLAND — the rung where the datapath is explicit but the registers are not yet
//! numbered.
//!
//! ```text
//! DataflowIR ──D1–D28──► SentientIR ──D29–D75 (in place)──► ──D76──► ProgIR
//!                          (here)
//! ```
//!
//! ⭐ WHERE THIS SITS, MEASURED. `dbo/docs/pass_pipeline.md` names every pass D1-D76; D1-D28 is
//! `buildDSCToSentientIRPipeline`, D29-D75 rewrite SentientIR in place (the prologue, `O2Pipeline`
//! run as three rounds, and `registerManagementPasses`), and D76 `SentientToProgIR` leaves MLIR
//! entirely. Running one real granite program through all of it: 128 pass invocations, of which 34
//! change the IR.
//!
//! ⛔⛔ AND THE RUNG IS **MIXED**, NOT A CLEAN DIALECT SWAP — see
//! [`dialects`] for the dump that shows `dataflow.*` and `agen.*` ops alive
//! alongside `sentient.*` after the conversion named for them.

pub mod dialects;
pub mod print;
pub mod ty;

use crate::arch::Arch;
use crate::islands::dataflow_ir::{KernelName, ProgramName};
use crate::model::Model;
use crate::workload::Workload;
use dialects::Op;

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
/// `element_wise_selection` that consumes it, `NO_CACHE_WALK` the cache walk — and a program that
/// forgot which rung it was baked for would be an instruction sequence nothing could place.
///
/// ⚠️ **TODO(quant):** a fourth parameter `Q: Quant` belongs here. There is no `Quant` trait in this
/// crate yet, and the preset it would encode is real — `quant/fp8-dynamic-per-channel` is one of the
/// acceptance build's own features, and its JSON states weight `num_bits`, `type`, `strategy`
/// (`channel`) and `symmetric`, plus a separate `input_activations` block. Those are exactly the
/// constants a compute's `opAPrecision`/`ComputePrecision` should be read from rather than defaulted
/// to fp16. Left as a TODO deliberately: inventing the trait's constants before a lowering reads one
/// is how a parameter becomes decoration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<A: Arch, M: Model, W: Workload> {
    /// The module's symbol — ⭐ THE RUNG BELOW'S TYPE, because a program keeps its name as it is
    /// lowered. Minting a second name would make the two rungs' artifacts uncorrelatable.
    pub name: ProgramName,
    /// The body.
    pub body: Vec<Op>,
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
