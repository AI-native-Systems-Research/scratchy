//! THE D76 PASS ITSELF — `runOnOperation`, `GenerateProgIR`, `GenerateProgIRForProgramUnit`,
//! the register initialisation, the program-length equalisation and the `init.smc` collapse.
//!
//! ⭐ ONE `ProgramUnitOp` SPANS THE SET: handles = cores x corelets x num_folds.
//! ⛔ THE PROGRAM BOUNDS ARE READ, NOT RESTATED — `max_ibuff_entries(unit)` per component and
//! per arch, never `kMaxCompIBuff` as if it were any unit's limit.
//!
//! 7 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e015_initializeUtilizedRegisters` | 0 | 39 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:429` |
//! | `e016_replaceProgramBodyWithSmcOp` | 0 | 49 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:560` |
//! | `e095_equalizeProgramLength` | 2 | 90 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:469` |
//! | `e127_GenerateProgIR` | 5 | 65 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:53` |
//! | `e128_GenerateProgIR` | 5 | 111 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:119` |
//! | `e129_GenerateProgIRForProgramUnit` | 6 | 186 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:234` |
//! | `e130_runOnOperation` | 7 | 135 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:610` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::Arch;
use crate::bridges::sentient_to_progir::construct::compute::{
    BinaryInstrOp, BinaryRefusal, ComputeComp, FmaRefusal, Mac, TernaryInstrOp, TernaryRefusal,
    UnaryInstrOp, UnaryRefusal,
};
use crate::bridges::sentient_to_progir::construct::instr_tag;
use crate::bridges::sentient_to_progir::construct::mask_and_splat::{
    ActiveMaskValue, SplatPadRefusal, construct_samv_reset_instruction,
};
use crate::bridges::sentient_to_progir::construct::opaque::{OpaqueInvocation, OpaqueRefusal};
use crate::bridges::sentient_to_progir::construct::scalar::{
    CmpOperands, ElseRegion, Sync, construct_jcmp_instr, construct_jmp_instr, construct_nop_instr,
};
use crate::bridges::sentient_to_progir::construct::transfer::{
    ExtractScalarLoad, L3LoadAndStore, LoadCompute, LrfCopy, Store,
};
use crate::bridges::sentient_to_progir::lower::compute::{
    AddRefusal, MacXrfState, ScalarAdd, ScalarSub, Splatted, SubRefusal, Successor,
    lower_add_operation, lower_binary_operation, lower_common_operations,
    lower_incr_mask_operation, lower_mac_operation, lower_opaque_operation, lower_samv_operation,
    lower_set_mask_operation, lower_splat_operation, lower_sub_operation, lower_ternary_operation,
    lower_unary_operation,
};
use crate::bridges::sentient_to_progir::lower::control::{
    CodeGraph, ForLoop, Nops, SendDestRefusal, SendDestination, UniformRegionSite, UniformizedOp,
    YieldParent, YieldedValue, lower_for_operation, lower_nop_operation, lower_return_operation,
    lower_set_send_destination_operation, lower_sync_operation, lower_uniform_operations,
    lower_uniform_yield_operation, lower_yield_operation, nops_of,
};
use crate::bridges::sentient_to_progir::lower::labels_and_regs::{
    LabelToJumps, LoweredOp, add_to_labels_map, update_label_and_add_to_code_graph,
};
use crate::bridges::sentient_to_progir::lower::transfer::{
    LoadOf, LowerTo, ScalarCopy, SelfConsumer, TransferRefusal, lower_copy_operation,
    lower_load_and_extract_scalar_operation, lower_load_and_send_operation,
    lower_load_and_store_operation, lower_load_compute_and_send_operation,
    lower_receive_and_extract_scalar_operation, lower_receive_and_store_operation,
};
use crate::bridges::sentient_to_progir::reg_def_tracker::{
    OpRegDefs, REG_DEF_CHECKING, RegSet, record_op_reg_defs,
};
use crate::bridges::sentient_to_progir::state::{
    CopyOps, LabelCounter, Labels, OpSite, RegGraphs, RegsToInit, UnitKey,
};
use crate::bridges::sentient_to_progir::uniform::block::{InstrIndex, UniformInstrBlocks};
use crate::bridges::sentient_to_progir::uniform::instr::{
    OperandMapRefusal, UniformInstrInfo, UniformLabel,
};
use crate::bridges::sentient_to_progir::utils::ComputeUnit;
use crate::formats::Bits;
use crate::islands::progir::dialects::{Op as ProgIrOp, init};
use crate::islands::progir::ty::{FoldId, Operand, OperandValue, RegType};
use crate::islands::progir::{
    Block, Instruction, OperandField, Program, RegInit, UnitProgram, UnitRegState, print,
};
use crate::islands::sentient;
use crate::islands::sentient::dialects::Op as SenOp;
use crate::islands::sentient::dialects::sentient::{CmpPredicate, Reg};
use crate::model::Model;
use crate::units::{Core, DfirUnit};
use crate::workload::Workload;
use std::collections::BTreeMap;
use sys_arch_spec::regfile::{Component, max_ibuff_entries};

/// Replaces: e015_initializeUtilizedRegisters
///
/// Give every register a unit uses a zero start, except those the header already initialises.
/// ⭐ THE TWO BRANCHES COLLAPSE: a component with no register state is an empty snapshot, which is
/// exactly what the reference's else-branch does with one.
/// ⛔ THE SNAPSHOT CANNOT GO STALE OBSERVABLY. `auto reg_init = ...getSimpleRegInit()` copies the
/// header (`progir.h:497` returns a reference; the `auto` drops it), and each `(file, index)` occurs
/// once in a `UtilizedRegisters`, so nothing this loop adds could have suppressed a later add.
/// The reference's `addRegInit` ASSIGNS (`progir.h:491-496`) where our island's state is a `Vec`.
pub fn initialize_utilized_registers<A: Arch, M: Model, W: Workload>(
    regs_to_init: &RegsToInit,
    progstateinfo: &mut Vec<(Core, Program<A, M, W>)>,
) {
    for (unit, reg_map) in regs_to_init {
        // ⛔ A UNIT WITH NO PROGRAM COMPONENT IS THE REFERENCE'S `DT_ERROR` (see
        // [`UnitKey::program_key`]), and `regs_to_init_` only ever holds the nine compute units.
        let Some((core, comp)) = unit.program_key() else {
            continue;
        };
        // ⭐ `progstateinfo_[core]` — `operator[]` on a `std::map`, so a core nothing has emitted for
        // gets an empty program rather than nothing at all.
        if !progstateinfo.iter().any(|(id, _)| *id == core) {
            progstateinfo.push((core, Program::default()));
        }
        for (_, program) in progstateinfo.iter_mut().filter(|(id, _)| *id == core) {
            let already: Vec<RegInit> = program
                .reg_state
                .iter()
                .find(|(unit, _)| *unit == comp)
                .map(|(_, state)| state.clone())
                .unwrap_or_default();
            if !program.reg_state.iter().any(|(unit, _)| *unit == comp) {
                program.reg_state.push((comp, UnitRegState::new()));
            }
            for (_, state) in program
                .reg_state
                .iter_mut()
                .filter(|(unit, _)| *unit == comp)
            {
                for (file, regs) in reg_map {
                    for index in regs {
                        if already
                            .iter()
                            .any(|init| init.file == *file && init.index == *index)
                        {
                            continue;
                        }
                        state.push(RegInit {
                            file: *file,
                            index: *index,
                            value: Operand::every(OperandValue::Int(0)),
                        });
                    }
                }
            }
        }
    }
}

/// Replaces: e016_replaceProgramBodyWithSmcOp
///
/// The module this pass leaves behind: the program's own function, its body dropped, holding
/// `init.smc` and a return.
/// ⛔ BOTH `signalPassFailure` PATHS ARE TYPE GUARDS HERE. One [`sentient::Program`] is one function,
/// so *"expects exactly one function in the module"* is unviolatable, and [`print::module`] emits
/// `func.func @name()` with no results, so there is never anything left to return.
/// ⛔ AND THE TWO NAMES ARE DIFFERENT FACTS: the function keeps the program's own name, while the op
/// carries `dccExtContext().prog_name_`.
#[must_use]
pub fn replace_program_body_with_smc_op<A: Arch, M: Model, W: Workload>(
    program: &sentient::Program<A, M, W>,
    prog_name: &str,
) -> String {
    print::module(
        &program.name.to_string(),
        &[ProgIrOp::Init(init::Op::Smc {
            name: prog_name.to_owned(),
        })],
    )
}
/// HOW MANY UNIFORMIZATION PADDING LABELS HAVE BEEN MINTED — `uniformization_padding_counter_`, a
/// member of the pass rather than of any one program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PaddingCounter(pub u32);

impl PaddingCounter {
    /// `uniformization_padding_counter_++` — the number this padding takes, then the step.
    pub fn bump(&mut self) -> u32 {
        let now = self.0;
        self.0 += 1;
        now
    }
}

/// WHY A PROGRAM COULD NOT BE PADDED TO THE LONGEST ONE — the reference's two failure paths, returned
/// rather than asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramLengthOffender {
    /// *"The program contains no instruction."* (`:502`) — there is no RETURN to pad in front of.
    Empty {
        /// Which core.
        core: Core,
        /// Which unit of it.
        unit: Component,
    },
    /// *"curr_prog_length can't be larger than max_prog_length."* (`:557-558`) — this core's program
    /// is longer than the one declared longest.
    Longer {
        /// Which core.
        core: Core,
        /// Which unit of it.
        unit: Component,
        /// What it holds.
        instructions: usize,
        /// What the declared longest holds.
        longest: usize,
    },
}

/// `static_cast<ProgIrCodeBlock*>(senCompProgram_.at(unit).blocks.front())->instrVector` — one unit's
/// straight-line program. ⛔ THE REFERENCE'S CAST IS UNCHECKED: a first block that is not CODE would
/// be read as one, where this answers `None`.
fn first_code_block<A: Arch, M: Model, W: Workload>(
    program: &Program<A, M, W>,
    unit: Component,
) -> Option<&Vec<Instruction>> {
    match program
        .per_unit
        .iter()
        .find(|(at, _)| *at == unit)?
        .1
        .blocks
        .first()?
    {
        Block::Code(instrs) => Some(instrs),
        _ => None,
    }
}

/// The same, mutably.
fn first_code_block_mut<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    unit: Component,
) -> Option<&mut Vec<Instruction>> {
    match program
        .per_unit
        .iter_mut()
        .find(|(at, _)| *at == unit)?
        .1
        .blocks
        .first_mut()?
    {
        Block::Code(instrs) => Some(instrs),
        _ => None,
    }
}

/// Replaces: e095_equalizeProgramLength
///
/// Pads every core's program for one unit out to the longest core's: one short, a NOP before the
/// RETURN; further short, a tagged jump over a dead-code copy of the longest program's tail.
///
/// ⛔ THE REFERENCE'S OWN `curr > max` PATH IS DEAD CODE — both lengths are `size_t`, so the
/// difference WRAPS and `> 1` takes it, padding a program already too long. Returned here instead.
/// ⛔ AND AN EMPTY PROGRAM IS NOT PADDED, where the reference's `back()` on one is undefined.
pub fn equalize_program_length<A: Arch, M: Model, W: Workload>(
    max_length_unit_core_map: &[(Component, Core)],
    progstateinfo: &mut [(Core, Program<A, M, W>)],
    padding_counter: &mut PaddingCounter,
) -> Vec<ProgramLengthOffender> {
    let mut offenders = Vec::new();
    // iterate through all unit types
    for (sen_comp, max_core_id) in max_length_unit_core_map.iter().copied() {
        let Some(max_instr_list) = progstateinfo
            .iter()
            .find(|(core, _)| *core == max_core_id)
            .and_then(|(_, program)| first_code_block(program, sen_comp))
            .cloned()
        else {
            continue;
        };
        let max_prog_length = max_instr_list.len();
        // iterate through psInfo for all cores
        for (core, program) in progstateinfo.iter_mut() {
            let core = *core;
            // skip padding if current core doesn't has this unit type
            let Some(curr_instr_vector) = first_code_block_mut(program, sen_comp) else {
                continue;
            };
            let curr_prog_length = curr_instr_vector.len();
            if curr_prog_length == 0 {
                offenders.push(ProgramLengthOffender::Empty {
                    core,
                    unit: sen_comp,
                });
                continue;
            }
            if curr_prog_length > max_prog_length {
                offenders.push(ProgramLengthOffender::Longer {
                    core,
                    unit: sen_comp,
                    instructions: curr_prog_length,
                    longest: max_prog_length,
                });
                continue;
            }
            // only pad instructions when the length of current program is shorter.
            let before_return = curr_prog_length.saturating_sub(1);
            match max_prog_length - curr_prog_length {
                0 => continue,
                // if diff is one, insert NOP before RETURN
                1 => {
                    if let Some(nop) = construct_nop_instr(Some("NOP for uniformization")).regular()
                    {
                        curr_instr_vector.insert(before_return, nop.instr());
                    }
                }
                _ => {
                    let padding_counter = padding_counter.bump();
                    // use RETURN's tag if existing.
                    let jcmp_tag = match &curr_instr_vector[before_return].tag {
                        Some(tag) => tag.clone(),
                        None => format!("uniformization_padding_{padding_counter}"),
                    };
                    curr_instr_vector[before_return].tag = Some(jcmp_tag.clone());
                    let jcmp = construct_jmp_instr(&jcmp_tag)
                        .with_common_comment("jump for uniformization");
                    if let Some(jcmp) = jcmp.regular() {
                        curr_instr_vector.insert(before_return, jcmp.instr());
                    }
                    let mut pc_targets_in_dead_code: Vec<String> = Vec::new();
                    for donor in max_instr_list
                        .iter()
                        .take(max_prog_length.saturating_sub(1))
                        .skip(curr_prog_length)
                    {
                        let mut instr = donor.clone();
                        instr.comment = Some("padding for uniformization".to_owned());
                        instr.dead = true;
                        // sanitize pc_targets: modify them to avoid collision with the valid code
                        if let Some((_, operand)) = instr
                            .fields
                            .iter_mut()
                            .find(|(field, _)| *field == OperandField::PcTarget)
                            && let Some(target) = operand.as_string(None).map(str::to_owned)
                        {
                            pc_targets_in_dead_code.push(target.clone());
                            *operand = instr_tag(&format!("{target}_uniformization"));
                        }
                        // sanitize tags: keep one only where the dead code still jumps to it.
                        if let Some(tag) = instr.tag.take() {
                            instr.tag = pc_targets_in_dead_code
                                .contains(&tag)
                                .then(|| format!("{tag}_uniformization"));
                        }
                        let end = curr_instr_vector.len().saturating_sub(1);
                        curr_instr_vector.insert(end, instr);
                    }
                }
            }
        }
    }
    offenders
}
/// `checkProgIR` (`Passes.td:38-39`), default `true` — a run-fixed pass option, so a const the way
/// [`REG_DEF_CHECKING`] is.
pub const CHECK_PROG_IR: bool = true;

/// `ibuff_usage_percent_` — WHOLE PERCENT, because `instr_count * 100 / max_instr_count` divides in
/// `int` and only then widens into the `float` field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct IbuffUsage(pub u32);

/// `struct CodeQualityStats` (`SentientToProgIR.hpp:69-76`) — what `CollectCodeQualityStats`
/// accumulates, and its absence is the option being off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeQualityStats {
    /// `num_conditional_jcmps_`.
    pub conditional_jcmps: u32,
    /// `ibuff_usage_percent_`.
    pub ibuff_usage: IbuffUsage,
    /// `num_copy_ops_`.
    pub copy_ops: CopyOps,
    /// `num_nops_`.
    pub nops: Nops,
    /// `comp_to_num_regs_`.
    pub regs_per_comp: Vec<(Component, Vec<(RegType, u32)>)>,
}

/// `mlir::WalkResult` — what a lowered op tells the walk that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Walk {
    /// `WalkResult::advance()`.
    Advance,
    /// `WalkResult::skip()` — a uniformized op's regions were lowered by the recursive call.
    Skip,
    /// `WalkResult::interrupt()`, which stops only the walk that saw it.
    Interrupt,
}

/// EVERY WAY LOWERING ONE UNIT CAN REFUSE — `signalPassFailure()` and the operand-mapping refusals,
/// returned rather than raised.
#[derive(Debug, Clone, PartialEq)]
pub enum LoweringRefusal {
    /// A `sentient.for`'s bound or step.
    For(OperandMapRefusal),
    /// A `sentient.mac`.
    Mac(FmaRefusal),
    /// A `sentient.binary`.
    Binary(BinaryRefusal),
    /// A `sentient.unary`.
    Unary(UnaryRefusal),
    /// A `sentient.ternary`.
    Ternary(TernaryRefusal),
    /// A `sentient.yield` or `uniform.yield`.
    Yield(OperandMapRefusal),
    /// A `sentient.add`.
    Add(AddRefusal),
    /// A `sentient.sub`.
    Sub(SubRefusal),
    /// One of the six load/store lowerings.
    Transfer(TransferRefusal),
    /// A `sentient.set_send_destination`.
    SendDestination(SendDestRefusal),
    /// A `sentient.splat`'s padding.
    Splat(SplatPadRefusal),
    /// A `sentient.opaque`.
    Opaque(OpaqueRefusal),
    /// The op needs a compute component and this unit is not one — the reference passes its `comp`
    /// straight through and writes the instruction anyway.
    NotAComputeUnit {
        /// Where the op sits.
        at: OpSite,
        /// The unit that reached it.
        comp: Component,
    },
    /// `op->emitError("unable to lower the op into prog. IR")`.
    Unlowerable(OpSite),
    /// `"Require larger IBUFF"` — the unit's estimate exceeds its component's capacity.
    Ibuff {
        /// `getEstimatedInstructionCount(...)`.
        estimated: u32,
        /// `getMaxIBuffEntries(unit_op)`.
        max: u16,
    },
}

/// WHAT ONE `GenerateProgIR` CALL WROTE AND WHERE THE WALK GOES NEXT.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    /// The `WalkResult` the reference returns.
    pub walk: Walk,
    /// Everything the op refused, in the order it refused.
    pub refused: Vec<LoweringRefusal>,
}

/// THE PASS STATE ONE OP'S LOWERING READS AND WRITES — the reference's threaded arguments plus the
/// `SentientToProgIRPass` members the lowerings touch.
pub struct Lowering<'a> {
    /// `std::map<Operation*, std::string>& labels`, a local of [`generate_unit_prog_ir`].
    pub labels: &'a mut Labels,
    /// `UniformInstrBlocks& uniform_instr_region`.
    pub blocks: &'a mut UniformInstrBlocks,
    /// `ProgIrGraphMap& reg_graph`.
    pub reg_graph: &'a mut RegGraphs,
    /// `regs_to_init_`.
    pub regs_to_init: &'a mut RegsToInit,
    /// `std::map<std::string, std::set<InstrIndex>>& label_to_jumps`.
    pub label_to_jumps: &'a mut LabelToJumps,
    /// `std::vector<InstrIndex>& nop_for_labels`.
    pub nop_for_labels: &'a mut Vec<InstrIndex>,
    /// `int& labels_ctr`, a local of [`generate_unit_prog_ir`].
    pub labels_ctr: &'a mut LabelCounter,
    /// `regDefTracker_`'s current set — `None` is `enabled()` answering false.
    pub reg_defs: Option<&'a mut RegSet>,
    /// `std::optional<CodeQualityStats>& cq_stats`.
    pub cq_stats: &'a mut Option<CodeQualityStats>,
    /// `has_samv_`.
    pub has_samv: &'a mut Option<ActiveMaskValue>,
    /// The units a scalar op's register initialisation is charged to.
    pub units: &'a [UnitKey],
    /// `fullRegInit.getValue()`.
    pub full_reg_init: bool,
}

impl Lowering<'_> {
    /// The same state with its own statistics sink — one region of a uniformized op collects into
    /// `cq_stats_region`, never into the program's.
    #[must_use]
    pub fn with_stats<'s>(
        &'s mut self,
        cq_stats: &'s mut Option<CodeQualityStats>,
    ) -> Lowering<'s> {
        Lowering {
            labels: &mut *self.labels,
            blocks: &mut *self.blocks,
            reg_graph: &mut *self.reg_graph,
            regs_to_init: &mut *self.regs_to_init,
            label_to_jumps: &mut *self.label_to_jumps,
            nop_for_labels: &mut *self.nop_for_labels,
            labels_ctr: &mut *self.labels_ctr,
            reg_defs: self.reg_defs.as_deref_mut(),
            cq_stats,
            has_samv: &mut *self.has_samv,
            units: self.units,
            full_reg_init: self.full_reg_init,
        }
    }

    /// The subset the six transfer lowerings take.
    fn transfer_to(&mut self) -> LowerTo<'_> {
        LowerTo {
            labels: &*self.labels,
            blocks: &mut *self.blocks,
            regs_to_init: &mut *self.regs_to_init,
            units: self.units,
            full_reg_init: self.full_reg_init,
        }
    }
}

/// ONE OP AS ITS LOWERING NEEDS IT — the `dyn_cast` chain's answer, already decomposed.
pub struct OpToLower<'a> {
    /// Which op this is, for the label map and the jump targets.
    pub at: OpSite,
    /// `regDefTracker_.recordOpRegDefs(op)`'s input.
    pub reg_defs: OpRegDefs,
    /// The arm the reference's chain selects.
    pub kind: OpKind<'a>,
}

/// THE 27 ARMS OF `GenerateProgIR`'s `dyn_cast` CHAIN, each carrying its lowering's own input.
pub enum OpKind<'a> {
    /// `sentient::ForOp`.
    For(ForLoop),
    /// `sentient::MacOp`.
    Mac {
        /// The FMA itself.
        mac: Mac,
        /// What the XRF analyzer contributes to it.
        xrf: MacXrfState,
    },
    /// `sentient::BinaryOp`.
    Binary(BinaryInstrOp<'a>),
    /// `sentient::UnaryOp`.
    Unary(UnaryInstrOp<'a>),
    /// `sentient::TernaryOp`.
    Ternary(TernaryInstrOp<'a>),
    /// `sentient::IfOp`, decomposed into `ConstructJCMPInstr`'s arguments.
    If {
        /// `if_op.getPredicate()`.
        predicate: CmpPredicate,
        /// The two compared operands.
        operands: CmpOperands,
        /// The `else` body, when there is one.
        else_region: ElseRegion,
        /// The `if`'s own results.
        results: Vec<Reg>,
        /// The op the `if` ends before.
        next: OpSite,
    },
    /// `sentient::YieldOp`.
    Yield {
        /// The yielded values.
        yielded: Vec<YieldedValue>,
        /// The region the yield closes.
        parent: YieldParent<'a>,
    },
    /// `sentient::AddOp`.
    Add(ScalarAdd),
    /// `sentient::SubOp`.
    Sub(ScalarSub),
    /// `sentient::LoadAndSendOp`.
    LoadAndSend {
        /// The load.
        load: LoadOf<'a>,
        /// `getElementSizeInBits(...)`.
        element_size: Bits,
    },
    /// `sentient::ReceiveAndStoreOp`.
    ReceiveAndStore {
        /// The store.
        store: Store<'a>,
        /// `getElementSizeInBits(...)`.
        element_size: Bits,
    },
    /// `sentient::LoadAndStoreOp`.
    LoadAndStore {
        /// The L3 load-and-store.
        ls: L3LoadAndStore,
        /// `getElementSizeInBits(...)`.
        element_size: Bits,
    },
    /// `sentient::LoadAndExtractScalarOp`.
    LoadAndExtractScalar(ExtractScalarLoad),
    /// `sentient::ReceiveAndExtractScalarOp`.
    ReceiveAndExtractScalar(LrfCopy),
    /// `sentient::LoadComputeAndSendOp`.
    LoadComputeAndSend {
        /// The load and the compute it feeds.
        load: LoadCompute<'a>,
        /// The SOURCE element size.
        src_element_size: Bits,
    },
    /// `sentient::SyncOp`.
    Sync {
        /// The sync.
        sync: Sync,
        /// Whether an L0 unit is tethered to this one.
        l0_tethered: bool,
        /// `getDbgName(op)`.
        dbg_name: Option<String>,
        /// The op after the sync.
        next: Option<&'a SenOp>,
    },
    /// `dataflow::ReturnOp`.
    Return,
    /// `sentient::NOPOp`.
    Nop {
        /// `getDbgName(op)`.
        dbg_name: Option<String>,
    },
    /// `sentient::CopyOp`.
    Copy {
        /// The scalar copy.
        copy: ScalarCopy<'a>,
        /// What a label on this op moves to.
        next: Option<Successor>,
    },
    /// The 12 ops that emit nothing but may carry a label.
    Common {
        /// What the label moves to.
        next: Option<Successor>,
    },
    /// `sentient::SetSendDestinationOp`.
    SetSendDestination(SendDestination<'a>),
    /// `sentient::SplatOp`.
    Splat(Splatted<'a>),
    /// `sentient::OpaqueOp`.
    Opaque(OpaqueInvocation<'a>),
    /// `sentient::SetActiveMaskValueOp`.
    Samv {
        /// The mask value.
        samv: ActiveMaskValue,
        /// `getDbgName(op)`.
        dbg_name: Option<String>,
        /// `getProgStitch() || forceSAMVReset` — whether the program's returns reset the mask.
        reset_at_return: bool,
    },
    /// `uniform::UniformizeRegionsOp` or `uniform::EqualizePatternOp`.
    Uniformized {
        /// The op and its regions.
        uniform: UniformizedOp<'a>,
        /// What a label on it moves to when it lowers no block.
        next: Option<Successor>,
    },
    /// `uniform::YieldOp`.
    UniformYield {
        /// The yielded values.
        yielded: Vec<YieldedValue>,
        /// Which region of the set this closes.
        site: UniformRegionSite,
        /// The op the region's jump lands on.
        next: Option<Successor>,
    },
    /// `sentient::SetMaskOp`.
    SetMask {
        /// `op.getMaskValue()`.
        mask_value: i64,
        /// `getDbgName(op)`.
        dbg_name: Option<String>,
    },
    /// `sentient::IncrMaskOp`.
    IncrMask {
        /// `getDbgName(op)`.
        dbg_name: Option<String>,
    },
    /// No arm claimed the op — the reference's `else`.
    Unlowerable,
}

/// ONE `dataflow::ProgramUnitOp` AS ITS WALK READS IT.
pub struct UnitProgramToLower<'a> {
    /// `getEstimatedInstructionCount(dccExtContext(), unit_op)`.
    pub estimated_instructions: u32,
    /// The unit's ops in pre-order, a uniformized op's subtree folded into its own arm because that
    /// is the subtree the walk skips.
    pub body: Vec<OpToLower<'a>>,
}

/// `cq_stats.value().num_copy_ops_ += …; num_nops_ += …` under one `CollectCodeQualityStats` gate.
fn charge(cq_stats: &mut Option<CodeQualityStats>, copy_ops: CopyOps, nops: Nops) {
    if let Some(stats) = cq_stats.as_mut() {
        stats.copy_ops.0 += copy_ops.0;
        stats.nops.0 += nops.0;
    }
}

/// Replaces: e128_GenerateProgIR
///
/// One op to its lowering, in the reference's own branch order, and the `WalkResult` that implies.
///
/// ⛔ A UNIFORMIZED OP ANSWERS `Skip` — its regions were just lowered by the recursive call.
/// ⛔ AND A SAMV ONLY EVER *SETS* `has_samv_` (`LowerSentientHelper.cpp:963-964`): a later
/// non-resetting SAMV must not clear a reset the program still owes.
#[must_use]
pub fn generate_prog_ir<A: Arch>(
    op: OpToLower<'_>,
    comp: Component,
    lowering: &mut Lowering<'_>,
) -> Generated {
    let OpToLower { at, reg_defs, kind } = op;
    if REG_DEF_CHECKING && let Some(defs) = lowering.reg_defs.as_deref_mut() {
        record_op_reg_defs(defs, &reg_defs);
    }
    let mut walk = Walk::Advance;
    let mut refused = Vec::new();
    // The compute narrowings, taken once so an arm needing one can refuse instead of writing an
    // instruction for a unit that has none.
    let compute_comp = ComputeComp::of_component(comp);
    let compute_unit = ComputeUnit::of_component(comp);
    let not_a_compute_unit = || LoweringRefusal::NotAComputeUnit { at, comp };
    match kind {
        OpKind::For(for_op) => {
            let lowered = lower_for_operation::<A>(
                comp,
                &for_op,
                lowering.labels_ctr,
                lowering.reg_graph,
                lowering.label_to_jumps,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::For));
        }
        OpKind::Mac { mac, xrf } => match compute_comp {
            Some(unit) => refused.extend(
                lower_mac_operation::<A>(
                    unit,
                    &mac,
                    xrf,
                    CodeGraph {
                        labels: &mut *lowering.labels,
                        region: &mut *lowering.blocks,
                        at,
                    },
                )
                .into_iter()
                .map(LoweringRefusal::Mac),
            ),
            None => refused.push(not_a_compute_unit()),
        },
        OpKind::Binary(binary) => match compute_comp {
            Some(unit) => refused.extend(
                lower_binary_operation::<A>(
                    unit,
                    &binary,
                    CodeGraph {
                        labels: &mut *lowering.labels,
                        region: &mut *lowering.blocks,
                        at,
                    },
                )
                .into_iter()
                .map(LoweringRefusal::Binary),
            ),
            None => refused.push(not_a_compute_unit()),
        },
        OpKind::Unary(unary) => match compute_comp {
            Some(unit) => refused.extend(
                lower_unary_operation::<A>(
                    unit,
                    &unary,
                    CodeGraph {
                        labels: &mut *lowering.labels,
                        region: &mut *lowering.blocks,
                        at,
                    },
                )
                .into_iter()
                .map(LoweringRefusal::Unary),
            ),
            None => refused.push(not_a_compute_unit()),
        },
        OpKind::Ternary(ternary) => match compute_unit {
            Some(unit) => refused.extend(
                lower_ternary_operation::<A>(
                    unit,
                    &ternary,
                    CodeGraph {
                        labels: &mut *lowering.labels,
                        region: &mut *lowering.blocks,
                        at,
                    },
                )
                .into_iter()
                .map(LoweringRefusal::Ternary),
            ),
            None => refused.push(not_a_compute_unit()),
        },
        OpKind::If {
            predicate,
            operands,
            else_region,
            results,
            next,
        } => {
            if let Some(stats) = lowering.cq_stats.as_mut() {
                stats.conditional_jcmps += 1;
            }
            let super_instr = construct_jcmp_instr(
                predicate,
                operands,
                &else_region,
                &results,
                next,
                lowering.labels,
                lowering.labels_ctr,
            );
            add_to_labels_map(lowering.blocks, lowering.label_to_jumps, &super_instr);
            update_label_and_add_to_code_graph(
                lowering.labels,
                lowering.blocks,
                at,
                LoweredOp::Other,
                super_instr,
                false,
            );
        }
        OpKind::Yield { yielded, parent } => {
            let lowered = lower_yield_operation::<A>(
                comp,
                &yielded,
                &parent,
                lowering.label_to_jumps,
                lowering.nop_for_labels,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, lowered.copy_ops, lowered.nops);
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Yield));
        }
        OpKind::Add(add) => {
            let lowered = lower_add_operation::<A>(
                comp,
                &add,
                lowering.units,
                lowering.full_reg_init,
                lowering.regs_to_init,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Add));
        }
        OpKind::Sub(sub) => {
            let lowered = lower_sub_operation::<A>(
                comp,
                &sub,
                lowering.units,
                lowering.full_reg_init,
                lowering.regs_to_init,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Sub));
        }
        OpKind::LoadAndSend {
            mut load,
            element_size,
        } => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_load_and_send_operation::<A>(comp, &mut load, element_size, at, &mut to)
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::ReceiveAndStore {
            mut store,
            element_size,
        } => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_receive_and_store_operation::<A>(comp, &mut store, element_size, at, &mut to)
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::LoadAndStore {
            mut ls,
            element_size,
        } => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_load_and_store_operation::<A>(comp, &mut ls, element_size, at, &mut to)
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::LoadAndExtractScalar(load) => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_load_and_extract_scalar_operation::<A>(&load, SelfConsumer, at, &mut to)
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::ReceiveAndExtractScalar(copy) => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_receive_and_extract_scalar_operation(&copy, at, &mut to)
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::LoadComputeAndSend {
            mut load,
            src_element_size,
        } => {
            let lowered = {
                let mut to = lowering.transfer_to();
                lower_load_compute_and_send_operation::<A>(
                    comp,
                    &mut load,
                    src_element_size,
                    at,
                    &mut to,
                )
            };
            charge(lowering.cq_stats, lowered.copy_ops, Nops(0));
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Transfer));
        }
        OpKind::Sync {
            sync,
            l0_tethered,
            dbg_name,
            next,
        } => {
            let nops = lower_sync_operation::<A>(
                &sync,
                l0_tethered,
                dbg_name.as_deref(),
                next,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, CopyOps(0), nops);
        }
        OpKind::Return => lower_return_operation(CodeGraph {
            labels: &mut *lowering.labels,
            region: &mut *lowering.blocks,
            at,
        }),
        OpKind::Nop { dbg_name } => {
            lower_nop_operation(
                dbg_name.as_deref(),
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, CopyOps(0), Nops(1));
        }
        OpKind::Copy { copy, next } => {
            let copied = lower_copy_operation::<A>(
                comp,
                &copy,
                next,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
                lowering.reg_graph,
                lowering.nop_for_labels,
            );
            charge(
                lowering.cq_stats,
                copied.lowered.copy_ops,
                copied.label.map_or(Nops(0), nops_of),
            );
            refused.extend(
                copied
                    .lowered
                    .refused
                    .into_iter()
                    .map(LoweringRefusal::Transfer),
            );
        }
        OpKind::Common { next } => {
            let placement = lower_common_operations(
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
                next,
                lowering.nop_for_labels,
            );
            charge(lowering.cq_stats, CopyOps(0), nops_of(placement));
        }
        OpKind::SetSendDestination(dest) => refused.extend(
            lower_set_send_destination_operation(
                dest,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            )
            .into_iter()
            .map(LoweringRefusal::SendDestination),
        ),
        OpKind::Splat(splatted) => match compute_unit {
            Some(unit) => {
                let mut copies = 0;
                refused.extend(
                    lower_splat_operation::<A>(
                        unit,
                        splatted,
                        lowering.units,
                        lowering.reg_graph,
                        CodeGraph {
                            labels: &mut *lowering.labels,
                            region: &mut *lowering.blocks,
                            at,
                        },
                        lowering.cq_stats.is_some().then_some(&mut copies),
                    )
                    .into_iter()
                    .map(LoweringRefusal::Splat),
                );
                charge(lowering.cq_stats, CopyOps(copies), Nops(0));
            }
            None => refused.push(not_a_compute_unit()),
        },
        OpKind::Opaque(opaque) => refused.extend(
            lower_opaque_operation(
                &opaque,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            )
            .into_iter()
            .map(LoweringRefusal::Opaque),
        ),
        OpKind::Samv {
            samv,
            dbg_name,
            reset_at_return,
        } => {
            if let Some(owed) = lower_samv_operation(
                samv,
                dbg_name.as_deref(),
                reset_at_return,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            ) {
                *lowering.has_samv = Some(owed);
            }
        }
        OpKind::Uniformized { uniform, next } => {
            refused.extend(lower_uniform_operations::<A>(
                comp, uniform, at, next, lowering,
            ));
            walk = Walk::Skip;
        }
        OpKind::UniformYield {
            yielded,
            site,
            next,
        } => {
            let lowered = lower_uniform_yield_operation::<A>(
                comp,
                &yielded,
                &site,
                next,
                lowering.nop_for_labels,
                CodeGraph {
                    labels: &mut *lowering.labels,
                    region: &mut *lowering.blocks,
                    at,
                },
            );
            charge(lowering.cq_stats, lowered.copy_ops, lowered.nops);
            refused.extend(lowered.refused.into_iter().map(LoweringRefusal::Yield));
        }
        OpKind::SetMask {
            mask_value,
            dbg_name,
        } => lower_set_mask_operation(
            mask_value,
            dbg_name.as_deref(),
            CodeGraph {
                labels: &mut *lowering.labels,
                region: &mut *lowering.blocks,
                at,
            },
        ),
        OpKind::IncrMask { dbg_name } => lower_incr_mask_operation(
            dbg_name.as_deref(),
            CodeGraph {
                labels: &mut *lowering.labels,
                region: &mut *lowering.blocks,
                at,
            },
        ),
        OpKind::Unlowerable => {
            refused.push(LoweringRefusal::Unlowerable(at));
            walk = Walk::Interrupt;
        }
    }
    Generated { walk, refused }
}

/// Replaces: e127_GenerateProgIR
///
/// One unit's whole program: the IBUFF bound, then every op in pre-order, each `dataflow.return`
/// preceded by the mask reset a SAMV owes.
///
/// ⛔ THE BOUND IS PER COMPONENT (`DccExtContext.cpp:267-281`), never `kMaxCompIBuff`, and exceeding
/// it ABANDONS THE UNIT — `signalPassFailure(); return;` lowers nothing.
/// ⭐ THE XRF ANALYZER IS DROPPED — `ConstructFMAInstr` never reads it (`…Helper.cpp:1425`).
#[must_use]
pub fn generate_unit_prog_ir<A: Arch>(
    unit: UnitProgramToLower<'_>,
    comp: Component,
    lowering: &mut Lowering<'_>,
) -> Vec<LoweringRefusal> {
    *lowering.labels = Labels::default();
    *lowering.labels_ctr = LabelCounter(0);
    // `DT_CHECK(!unit_op.getUnits().empty())` has no input left: `Units` is non-empty by type, and
    // `getMaxIBuffEntries` resolves that front unit to exactly this component.
    let max_instr_count = max_ibuff_entries(comp);
    if let Some(stats) = lowering.cq_stats.as_mut() {
        stats.ibuff_usage =
            IbuffUsage(unit.estimated_instructions * 100 / u32::from(max_instr_count));
    }
    if CHECK_PROG_IR && unit.estimated_instructions > u32::from(max_instr_count) {
        return vec![LoweringRefusal::Ibuff {
            estimated: unit.estimated_instructions,
            max: max_instr_count,
        }];
    }
    *lowering.has_samv = None;
    let mut refused = Vec::new();
    for op in unit.body {
        if let (Some(owed), OpKind::Return) = (*lowering.has_samv, &op.kind) {
            lowering
                .blocks
                .add_instruction_to_last_block(construct_samv_reset_instruction(
                    owed.mask_value,
                    owed.precision,
                ));
            *lowering.has_samv = None;
        }
        let generated = generate_prog_ir::<A>(op, comp, lowering);
        refused.extend(generated.refused);
        if generated.walk == Walk::Interrupt {
            break;
        }
    }
    refused
}
/// `CollectCodeQualityStats` (`SentientToProgIR.cpp:36-40`), a `cl::opt<bool>` defaulting to false —
/// a const the way [`REG_DEF_CHECKING`] is.
pub const COLLECT_CODE_QUALITY_STATS: bool = false;

/// WHICH RESULT OF ITS `dataflow.get_unit` ONE HANDLE IS — `getResultNum(unit.getDefiningOp(), unit)`
/// (`cpp:340`).
///
/// ⭐ A `get_unit`'s RESULT NUMBER *IS* THE FOLD — `Dataflow.td:48,56-58`, the reading
/// [`crate::bridges::dataflow_ir_to_sentient::tf_unit_filtering::FoldId`] already states. It is still
/// not the fold *id*: `foldIds_` maps this index to what the SDSC calls that fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FoldIndex(pub u32);

/// ONE HANDLE OF THE SET — one unit at one fold, which is one `dataflow.get_unit` result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitHandle {
    /// The unit `getUnitName` names.
    pub key: UnitKey,
    /// Which of its folds.
    pub fold: FoldIndex,
}

/// `unit_op.getUnits()` — ⛔ NON-EMPTY BY CONSTRUCTION, which is what
/// `DT_CHECK(unit_op.getUnits().size() >= 1)` (`cpp:236`) had to say at run time, and what
/// [`crate::islands::sentient::ProgramUnits`] says for the rung above.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitHandles {
    head: UnitHandle,
    rest: Vec<UnitHandle>,
}

impl UnitHandles {
    /// The handles, the first being the one whose type decides the program unit's component.
    #[must_use]
    pub fn of(head: UnitHandle, rest: Vec<UnitHandle>) -> UnitHandles {
        UnitHandles { head, rest }
    }

    /// `getUnits()[0]` — the handle `getType()` is read off.
    #[must_use]
    pub const fn first(&self) -> UnitHandle {
        self.head
    }

    /// Every handle, head first.
    pub fn iter(&self) -> impl Iterator<Item = UnitHandle> + '_ {
        core::iter::once(self.head).chain(self.rest.iter().copied())
    }
}

/// ONE `dataflow::ProgramUnitOp` AS [`generate_prog_ir_for_program_unit`] READS IT.
pub struct ProgramUnitToLower<'a> {
    /// `getUnits()` — cores x corelets x folds of ONE unit type.
    pub units: UnitHandles,
    /// The body, lowered ONCE and filed under every fold-0 handle.
    pub program: UnitProgramToLower<'a>,
}

/// EVERY WAY ONE PROGRAM UNIT CAN REFUSE.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramUnitRefusal {
    /// `DT_CHECK_MSG(record != …end(), "unexpected unit")` (`cpp:239-241`) — the program unit's own
    /// type has no component, so nothing of it is lowered at all.
    UnexpectedUnit(DfirUnit),
    /// `getSenComponentForProgramStateInfo`'s `DT_ERROR` (`DccExtContext.cpp:210-238`) — this handle
    /// has no program component, so its program and register state are not filed.
    UnkeyableUnit(UnitKey),
    /// What lowering the body refused.
    Lowering(LoweringRefusal),
}

/// THE PASS MEMBERS ONE PROGRAM UNIT'S LOWERING OUTLIVES — everything on `SentientToProgIRPass` that
/// is neither a local of this call nor part of its output.
pub struct PassState<'a> {
    /// `regs_to_init_`.
    pub regs_to_init: &'a mut RegsToInit,
    /// `unit_foldid_map_` — one entry per handle, in handle order, which is what a uniform operand
    /// map reads a key unit's fold out of.
    pub unit_folds: &'a mut Vec<(UnitKey, Option<FoldId>)>,
    /// `comp_to_code_quality_stats_`.
    pub code_quality: &'a mut Vec<(Component, CodeQualityStats)>,
    /// `regDefTracker_`'s current set — `None` is `enabled()` answering false.
    pub reg_defs: Option<&'a mut RegSet>,
    /// `dcc_ext_ctx_.artifacts_->foldIds_` — ⛔ `None` IS NO SDSC AT ALL, which is one fold.
    pub fold_ids: Option<&'a [FoldId]>,
    /// `fullRegInit.getValue()`.
    pub full_reg_init: bool,
    /// The next padding-branch name — ⭐ A PASS COUNTER, not a local: the reference draws each with
    /// `rand()` (see [`UniformLabel`]) and two handles' padding must not collide.
    pub padding_label: &'a mut UniformLabel,
}

/// Replaces: e129_GenerateProgIRForProgramUnit
///
/// One `dataflow.program_unit`: lower its body once, fold away every NOP a label alone earned, then
/// file that one program and its register state under each fold-0 handle's core.
///
/// ⛔ ONLY FOLD 0 IS FILED — a later fold's variations live in the operand map, not in a program.
/// ⛔ A TAG IS NEVER `"be"` (`cpp:349`): that guard compares one against the `be` FIELD's value.
#[must_use]
pub fn generate_prog_ir_for_program_unit<A: Arch, M: Model, W: Workload>(
    unit_op: ProgramUnitToLower<'_>,
    psinfo_map: &mut Vec<(Core, Program<A, M, W>)>,
    max_length_unit_core_map: &mut Vec<(Component, Core)>,
    pass: &mut PassState<'_>,
) -> Vec<ProgramUnitRefusal> {
    let ProgramUnitToLower { units, program } = unit_op;
    // `unit_foldid_map_[unit] = …` for every handle. The subscript ASSIGNS, so a second call for the
    // same program unit replaces its entries rather than doubling them.
    pass.unit_folds
        .retain(|(held, _)| !units.iter().any(|handle| handle.key == *held));
    for handle in units.iter() {
        let fold = match pass.fold_ids {
            Some(ids) => ids.get(handle.fold.0 as usize).copied(),
            None => Some(FoldId(0)),
        };
        pass.unit_folds.push((handle.key, fold));
    }
    let Some((_, comp)) = units.first().key.program_key() else {
        return vec![ProgramUnitRefusal::UnexpectedUnit(units.first().key.unit)];
    };
    let mut blocks = UniformInstrBlocks::default();
    let mut reg_graph = RegGraphs::default();
    let mut label_to_jumps = LabelToJumps::new();
    let mut nop_for_labels: Vec<InstrIndex> = Vec::new();
    let mut cq_stats = COLLECT_CODE_QUALITY_STATS.then(CodeQualityStats::default);
    // `labels`, `labels_ctr` and `has_samv_` are per-unit and [`generate_unit_prog_ir`] resets each.
    let mut labels = Labels::default();
    let mut labels_ctr = LabelCounter(0);
    let mut has_samv = None;
    let keys: Vec<UnitKey> = units.iter().map(|handle| handle.key).collect();
    let mut refused: Vec<ProgramUnitRefusal> = generate_unit_prog_ir::<A>(
        program,
        comp,
        &mut Lowering {
            labels: &mut labels,
            blocks: &mut blocks,
            reg_graph: &mut reg_graph,
            regs_to_init: pass.regs_to_init,
            label_to_jumps: &mut label_to_jumps,
            nop_for_labels: &mut nop_for_labels,
            labels_ctr: &mut labels_ctr,
            reg_defs: pass.reg_defs.as_deref_mut(),
            cq_stats: &mut cq_stats,
            has_samv: &mut has_samv,
            units: &keys,
            full_reg_init: pass.full_reg_init,
        },
    )
    .into_iter()
    .map(ProgramUnitRefusal::Lowering)
    .collect();

    if let Some(mut stats) = cq_stats {
        // The `llvm::dbgs()` dump is every number this value already holds, so it is the port of it.
        // ⭐ AND THE COUNTS ARE FILED WITH THE STATS: the reference writes them into its dying local
        // one line too late (`cpp:934`), and nothing in the tree reads `comp_to_code_quality_stats_`,
        // so neither ordering is observable and this one is the value the dump names.
        let mut reg_num_per_type: BTreeMap<RegType, u32> = BTreeMap::new();
        for (unit, reg_map) in pass.regs_to_init.iter() {
            if unit.program_key().map(|(_, gen_comp)| gen_comp) != Some(comp) {
                continue;
            }
            for (reg_type, regs) in reg_map {
                let held = reg_num_per_type.entry(*reg_type).or_default();
                *held = (*held).max(u32::try_from(regs.len()).unwrap_or(u32::MAX));
            }
        }
        stats.regs_per_comp = vec![(comp, reg_num_per_type.into_iter().collect())];
        match pass.code_quality.iter().position(|(at, _)| *at == comp) {
            Some(at) => pass.code_quality[at].1 = stats,
            None => pass.code_quality.push((comp, stats)),
        }
    }

    // ⛔ ONLY FOLD 0: `if (fold_idx > 0) continue;` — the later folds' variations are already in the
    // operand map, so they file no program of their own.
    for handle in units.iter().filter(|handle| handle.fold == FoldIndex(0)) {
        let Some((core, my_comp)) = handle.key.program_key() else {
            refused.push(ProgramUnitRefusal::UnkeyableUnit(handle.key));
            continue;
        };
        // `psinfo_map[core]` — the subscript default-constructs, so a core nothing has emitted for
        // still gets a program.
        if !psinfo_map.iter().any(|(at, _)| *at == core) {
            psinfo_map.push((core, Program::default()));
        }

        // Every NOP a label alone earned hands that label to the instruction after it, and every
        // jump aimed at the NOP is re-aimed there.
        let mut nop_to_be_deleted: Vec<UniformInstrInfo> = Vec::new();
        if !blocks.blocks.is_empty() {
            for index in nop_for_labels.clone() {
                let Some(curr_label) = blocks
                    .uniform_instr_mut(index)
                    .map(|nop| nop.tag.clone().unwrap_or_default())
                else {
                    continue;
                };
                let next_index = InstrIndex {
                    instr: index.instr + 1,
                    ..index
                };
                if !blocks.does_instr_exist(next_index) {
                    continue;
                }
                let next_label = blocks
                    .uniform_instr_mut(next_index)
                    .and_then(|next| next.tag.clone())
                    .unwrap_or_default();
                if curr_label == "be" || next_label == "be" {
                    continue;
                }
                // A non-empty target is what `pc_target` needs, so an untagged successor takes the
                // NOP's own label first.
                let target = if next_label.is_empty() {
                    if let Some(next) = blocks.uniform_instr_mut(next_index) {
                        next.tag = (!curr_label.is_empty()).then(|| curr_label.clone());
                    }
                    curr_label.clone()
                } else {
                    next_label
                };
                for jump_index in label_to_jumps.get(&curr_label).cloned().unwrap_or_default() {
                    if let Some(jump) = blocks.uniform_instr_mut(jump_index) {
                        jump.set_common_field(
                            OperandField::PcTarget,
                            Operand::every(OperandValue::InstrTag(target.clone())),
                        );
                    }
                    label_to_jumps
                        .entry(target.clone())
                        .or_default()
                        .insert(jump_index);
                }
                if let Some(nop) = blocks.uniform_instr_mut(index) {
                    nop_to_be_deleted.push(nop.clone());
                }
            }
        }

        // `psinfo.senCompProgram_[my_comp].destroyGraph()` — the subscript creates and the call
        // empties, so a unit that emitted nothing still files an EMPTY program.
        // ⛔ `DT_CHECK(blocks.size() == 1)` IS STRUCTURAL HERE: `addInstruction` opens exactly one
        // lazy CODE block (`progir.h:459-466`), which is the one block below.
        let mut unit_program = UnitProgram::default();
        let mut filed_size = None;
        if !blocks.blocks.is_empty() {
            let instrs =
                blocks.uniformized_unit_uniform_instr_list(handle.key, *pass.padding_label);
            for _ in &blocks.blocks {
                *pass.padding_label = pass.padding_label.bump();
            }
            let kept: Vec<Instruction> = instrs
                .iter()
                .filter(|instr| !nop_to_be_deleted.contains(instr))
                .map(|instr| instr.uniformized_instr(handle.key))
                .collect();
            filed_size = Some(kept.len());
            unit_program.blocks.push(Block::Code(kept));
        }
        if let Some((_, held)) = psinfo_map.iter_mut().find(|(at, _)| *at == core) {
            match held.per_unit.iter().position(|(at, _)| *at == my_comp) {
                Some(at) => held.per_unit[at].1 = unit_program,
                None => held.per_unit.push((my_comp, unit_program)),
            }
        }

        // Which core holds the longest program for this unit — read AFTER the write above, because
        // the core already recorded may be this one.
        if let Some(instr_size) = filed_size {
            let recorded = max_length_unit_core_map
                .iter()
                .find(|(at, _)| *at == my_comp)
                .map(|(_, at)| *at);
            let max_instr_size = recorded
                .and_then(|at| psinfo_map.iter().find(|(core, _)| *core == at))
                .and_then(|(_, held)| first_code_block(held, my_comp))
                .map_or(0, Vec::len);
            // ⭐ THE TIE GOES TO THE LOWER CORE, and only when the incumbent is non-empty.
            let takes_it = match recorded {
                None => true,
                Some(at) => {
                    max_instr_size < instr_size
                        || (max_instr_size > 0 && max_instr_size == instr_size && at > core)
                }
            };
            if takes_it {
                match max_length_unit_core_map
                    .iter()
                    .position(|(at, _)| *at == my_comp)
                {
                    Some(at) => max_length_unit_core_map[at].1 = core,
                    None => max_length_unit_core_map.push((my_comp, core)),
                }
            }
        }

        // `psinfo.regState_[my_comp].destroyGraph()`, then this unit's initialisers — and `head` being
        // null is `blocks.empty()`, the same graph being empty, so an empty one is ERASED.
        let state = reg_graph
            .get(handle.key)
            .filter(|state| !state.is_empty())
            .map(|state| {
                let mut sorted = state.clone();
                // `regInfo` is `map<RegType, map<unsigned, …>>`, so the file orders and the index
                // orders within it.
                sorted.sort_by_key(|init| (init.file, init.index));
                sorted
            });
        if let Some((_, held)) = psinfo_map.iter_mut().find(|(at, _)| *at == core) {
            let at = held.reg_state.iter().position(|(at, _)| *at == my_comp);
            match (state, at) {
                (Some(state), Some(at)) => held.reg_state[at].1 = state,
                (Some(state), None) => held.reg_state.push((my_comp, state)),
                (None, Some(at)) => {
                    held.reg_state.remove(at);
                }
                (None, None) => {}
            }
        }
    }
    refused
}
// crustify:todo: e130_runOnOperation

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::sentient_to_progir::construct::descriptive;
    use crate::bridges::sentient_to_progir::construct::int;
    use crate::bridges::sentient_to_progir::construct::scalar::CmpImm;
    use crate::bridges::sentient_to_progir::lower::control::RegionIndex;
    use crate::bridges::sentient_to_progir::state::{UnitKey, UtilizedRegisters};
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::progir::ty::RegType;
    use crate::islands::progir::{OpCode, UnitProgram};
    use crate::islands::sentient::dialects::sentient::RegIndex;
    use crate::islands::sentient::dialects::sentient::{
        RawPrecision, SliceId, ValidEntries, WslLen,
    };
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::{DfirUnit, Residency};
    use std::collections::BTreeSet;
    use sys_arch_spec::regfile::Component;

    struct M;
    impl Model for M {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    struct W;
    impl Workload for W {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 128;
    }

    #[test]
    fn only_a_register_the_header_does_not_already_initialise_is_zeroed() {
        let core = Core::checked(0).expect("core 0");
        let key = |unit| UnitKey::of(unit, Residency::CoreWide { core }).expect("core 0 keys");
        let mut progstateinfo: Vec<(Core, Program<Target, M, W>)> = Vec::new();
        // The header already starts `lrf1`; `lrf1` and `lar2` are both used.
        progstateinfo.push((core, Program::default()));
        progstateinfo[0].1.reg_state.push((
            Component::Pe,
            vec![RegInit {
                file: RegType::Lrf,
                index: RegIndex::at::<1>(),
                value: Operand::every(OperandValue::Int(7)),
            }],
        ));
        let used = |file, index| UtilizedRegisters::from([(file, BTreeSet::from([index]))]);
        let mut regs_to_init = RegsToInit::new();
        regs_to_init.insert(
            key(DfirUnit::Pe),
            UtilizedRegisters::from([
                (RegType::Lrf, BTreeSet::from([RegIndex::at::<1>()])),
                (RegType::Lar, BTreeSet::from([RegIndex::at::<2>()])),
            ]),
        );
        initialize_utilized_registers(&regs_to_init, &mut progstateinfo);
        assert_eq!(
            progstateinfo[0].1.reg_state,
            vec![(
                Component::Pe,
                vec![
                    RegInit {
                        file: RegType::Lrf,
                        index: RegIndex::at::<1>(),
                        value: Operand::every(OperandValue::Int(7)),
                    },
                    RegInit {
                        file: RegType::Lar,
                        index: RegIndex::at::<2>(),
                        value: Operand::every(OperandValue::Int(0)),
                    },
                ],
            )]
        );
        // ⛔ A core with no program at all still gets one, and everything it uses is zeroed.
        let other = Core::checked(1).expect("core 1");
        let elsewhere =
            UnitKey::of(DfirUnit::Sfp, Residency::CoreWide { core: other }).expect("core 1 keys");
        initialize_utilized_registers(
            &RegsToInit::from([(elsewhere, used(RegType::Lrf, RegIndex::at::<0>()))]),
            &mut progstateinfo,
        );
        assert_eq!(
            progstateinfo[1].1.reg_state,
            vec![(
                Component::Sfp,
                vec![RegInit {
                    file: RegType::Lrf,
                    index: RegIndex::at::<0>(),
                    value: Operand::every(OperandValue::Int(0)),
                }],
            )]
        );
    }

    #[test]
    fn the_emptied_function_keeps_its_own_name_and_holds_only_the_smc_reference() {
        let program: sentient::Program<Target, M, W> = sentient::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Pe, Val(0)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        assert_eq!(
            replace_program_body_with_smc_op(&program, "default_prog_name"),
            concat!(
                "module {\n",
                "  func.func @g0_0_add() attributes {grid = [1]} {\n",
                "    init.smc {name = \"default_prog_name\"}\n",
                "    return\n",
                "  }\n",
                "}\n"
            )
        );
    }
    /// e095: three cores of one unit padded to the longest — a NOP where one short, and a tagged jump
    /// over the donor's sanitized tail where further short. A longer program is an offender.
    #[test]
    fn a_shorter_program_is_padded_to_the_longest_core() {
        let at = |core: u32| Core::checked(core).expect("every arch has 32 cores");
        let ret = Instruction {
            opcode: OpCode::RETURN,
            symbolic_opcode: None,
            fields: Vec::new(),
            dead: false,
            tag: None,
            comment: None,
        };
        let mut jump = ret.clone();
        jump.opcode = OpCode::JCMP;
        jump.fields = vec![(OperandField::PcTarget, instr_tag("loop"))];
        let mut labelled = ret.clone();
        labelled.opcode = OpCode::NOP;
        labelled.tag = Some("loop".to_owned());
        let core = |core: u32, instrs: Vec<Instruction>| {
            (
                at(core),
                Program::<Target, M, W> {
                    per_unit: vec![(
                        Component::Pt,
                        UnitProgram {
                            blocks: vec![Block::Code(instrs)],
                        },
                    )],
                    ..Program::default()
                },
            )
        };
        let donor = vec![labelled.clone(), jump, labelled, ret.clone()];
        let mut progstateinfo = vec![
            core(0, donor.clone()),
            core(1, vec![ret.clone()]),
            core(2, vec![ret.clone(), ret.clone(), ret.clone()]),
            core(
                3,
                vec![ret.clone(), ret.clone(), ret.clone(), ret.clone(), ret],
            ),
        ];
        let mut counter = PaddingCounter::default();
        let offenders =
            equalize_program_length(&[(Component::Pt, at(0))], &mut progstateinfo, &mut counter);
        assert_eq!(
            offenders,
            vec![ProgramLengthOffender::Longer {
                core: at(3),
                unit: Component::Pt,
                instructions: 5,
                longest: 4,
            }]
        );
        assert_eq!(counter, PaddingCounter(1), "only core 1 minted a label");

        let padded =
            first_code_block(&progstateinfo[1].1, Component::Pt).expect("core 1's program");
        assert_eq!(padded.len(), 4);
        assert_eq!(
            padded.iter().map(|instr| instr.opcode).collect::<Vec<_>>(),
            vec![OpCode::JCMP, OpCode::JCMP, OpCode::NOP, OpCode::RETURN]
        );
        assert_eq!(
            padded[0].comment.as_deref(),
            Some("jump for uniformization")
        );
        assert_eq!(
            padded[0].fields,
            vec![
                (OperandField::Mode, descriptive("always")),
                (
                    OperandField::PcTarget,
                    instr_tag("uniformization_padding_0")
                ),
            ]
        );
        assert_eq!(
            padded[3].tag.as_deref(),
            Some("uniformization_padding_0"),
            "the RETURN is what the jump lands on"
        );
        // The donor's own tail: a pc_target and a tag both get the suffix, and both are dead.
        assert!(padded[1].dead && padded[2].dead);
        assert_eq!(
            padded[1].fields,
            vec![(OperandField::PcTarget, instr_tag("loop_uniformization"))]
        );
        assert_eq!(padded[2].tag.as_deref(), Some("loop_uniformization"));

        // One short: a NOP before the RETURN, and nothing else touched.
        let nopped =
            first_code_block(&progstateinfo[2].1, Component::Pt).expect("core 2's program");
        assert_eq!(
            nopped.iter().map(|instr| instr.opcode).collect::<Vec<_>>(),
            vec![OpCode::RETURN, OpCode::RETURN, OpCode::NOP, OpCode::RETURN]
        );
        assert_eq!(nopped[2].comment.as_deref(), Some("NOP for uniformization"));
        assert_eq!(
            first_code_block(&progstateinfo[0].1, Component::Pt),
            Some(&donor),
            "the longest core is left alone"
        );
    }
    /// The pass state one lowering call reads, owned by the test so each call takes a fresh borrow.
    #[derive(Default)]
    struct State {
        labels: Labels,
        blocks: UniformInstrBlocks,
        reg_graph: RegGraphs,
        regs_to_init: RegsToInit,
        label_to_jumps: LabelToJumps,
        nop_for_labels: Vec<InstrIndex>,
        labels_ctr: LabelCounter,
        cq_stats: Option<CodeQualityStats>,
        has_samv: Option<ActiveMaskValue>,
    }

    impl State {
        fn lowering(&mut self) -> Lowering<'_> {
            Lowering {
                labels: &mut self.labels,
                blocks: &mut self.blocks,
                reg_graph: &mut self.reg_graph,
                regs_to_init: &mut self.regs_to_init,
                label_to_jumps: &mut self.label_to_jumps,
                nop_for_labels: &mut self.nop_for_labels,
                labels_ctr: &mut self.labels_ctr,
                reg_defs: None,
                cq_stats: &mut self.cq_stats,
                has_samv: &mut self.has_samv,
                units: &[],
                full_reg_init: false,
            }
        }
    }

    fn op(at: u32, kind: OpKind<'static>) -> OpToLower<'static> {
        OpToLower {
            at: OpSite(at),
            reg_defs: OpRegDefs::None,
            kind,
        }
    }

    /// e127: the SAMV a stitched program owes is reset in front of the FIRST return only, and a unit
    /// whose estimate exceeds its component's IBUFF is an offender that lowers nothing at all.
    #[test]
    fn the_owed_samv_reset_precedes_the_first_return_and_an_oversized_unit_lowers_nothing() {
        let mask = ActiveMaskValue {
            maskall: false,
            slice_id_xsl: SliceId(5),
            valid_entries: ValidEntries(12),
            mask_value: RegIndex::at::<0>(),
            precision: RawPrecision(8),
            xslinner: true,
            wsl_len: WslLen(8),
        };
        let mut state = State {
            labels_ctr: LabelCounter(7),
            cq_stats: Some(CodeQualityStats::default()),
            ..State::default()
        };
        let refused = generate_unit_prog_ir::<Target>(
            UnitProgramToLower {
                estimated_instructions: 4,
                body: vec![
                    op(
                        0,
                        OpKind::Samv {
                            samv: mask,
                            dbg_name: None,
                            reset_at_return: true,
                        },
                    ),
                    op(1, OpKind::Return),
                    op(2, OpKind::Return),
                ],
            },
            Component::Pt,
            &mut state.lowering(),
        );
        assert_eq!(refused, Vec::new());
        assert_eq!(
            state.labels_ctr,
            LabelCounter(0),
            "the counter is a per-unit local"
        );
        assert_eq!(state.has_samv, None, "the reset was paid");
        assert_eq!(
            state.blocks.blocks[0].instr_lists()[0]
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![OpCode::SAMV, OpCode::SAMV, OpCode::RETURN, OpCode::RETURN]
        );
        assert_eq!(
            state.blocks.blocks[0].instr_lists()[0][1].common_field(OperandField::Sliceidxsl),
            Some(&int(7)),
            "the second SAMV is the reset"
        );

        let over = generate_unit_prog_ir::<Target>(
            UnitProgramToLower {
                estimated_instructions: 200,
                body: vec![op(3, OpKind::Nop { dbg_name: None })],
            },
            Component::Pt,
            &mut state.lowering(),
        );
        assert_eq!(
            over,
            vec![LoweringRefusal::Ibuff {
                estimated: 200,
                max: 128,
            }]
        );
        assert_eq!(
            state.blocks.blocks[0].instr_lists()[0].len(),
            4,
            "nothing was lowered"
        );
        assert_eq!(
            state.cq_stats.expect("collecting").ibuff_usage,
            IbuffUsage(156),
            "200 * 100 / 128, in integer division"
        );
    }

    /// e128: a `sentient.if` mints its JCMP and is counted as a conditional jump; an op no arm claims
    /// is an offender that INTERRUPTS the walk.
    #[test]
    fn an_if_is_counted_and_an_unclaimed_op_interrupts_the_walk() {
        let mut state = State {
            cq_stats: Some(CodeQualityStats::default()),
            ..State::default()
        };
        state.blocks.append_regular_block();
        let generated = generate_prog_ir::<Target>(
            op(
                1,
                OpKind::If {
                    predicate: CmpPredicate::Eq,
                    operands: CmpOperands::LccrVsImm {
                        lccr: RegIndex::at::<1>(),
                        imm: CmpImm::Constant(2),
                    },
                    else_region: ElseRegion::None,
                    results: Vec::new(),
                    next: OpSite(9),
                },
            ),
            Component::L3su,
            &mut state.lowering(),
        );
        assert_eq!(
            generated,
            Generated {
                walk: Walk::Advance,
                refused: Vec::new(),
            }
        );
        assert_eq!(
            state.blocks.blocks[0].instr_lists()[0]
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![OpCode::JCMP]
        );
        assert_eq!(state.labels.get(OpSite(9)), Some("if-label-0-end"));
        assert_eq!(
            state.label_to_jumps.get("if-label-0-end"),
            Some(&BTreeSet::from([InstrIndex {
                block: 0,
                region: RegionIndex(0),
                instr: 0,
            }]))
        );

        let bad = generate_prog_ir::<Target>(
            op(2, OpKind::Unlowerable),
            Component::L3su,
            &mut state.lowering(),
        );
        assert_eq!(
            bad,
            Generated {
                walk: Walk::Interrupt,
                refused: vec![LoweringRefusal::Unlowerable(OpSite(2))],
            }
        );
        assert_eq!(state.cq_stats.expect("collecting").conditional_jcmps, 1);
    }

    /// e129: the NOP a label alone earned hands its tag to the RETURN and vanishes; both cores of a
    /// two-core set file the same program, only fold 0 files at all, and the length tie goes to the
    /// lower core even though the higher one recorded it first.
    #[test]
    fn a_label_only_nop_is_folded_away_and_only_fold_zero_files_a_program() {
        let at = |core: u32| Core::checked(core).expect("every arch has 32 cores");
        let key = |core| UnitKey::of(DfirUnit::Pe, Residency::CoreWide { core }).expect("keys");
        let handle = |core, fold| UnitHandle {
            key: key(at(core)),
            fold: FoldIndex(fold),
        };
        let mut unit_folds = Vec::new();
        let mut code_quality = Vec::new();
        let mut padding_label = UniformLabel(0);
        let mut regs_to_init = RegsToInit::new();
        let mut pass = PassState {
            regs_to_init: &mut regs_to_init,
            unit_folds: &mut unit_folds,
            code_quality: &mut code_quality,
            reg_defs: None,
            fold_ids: Some(&[FoldId(0), FoldId(3)]),
            full_reg_init: false,
            padding_label: &mut padding_label,
        };
        let mut psinfo_map: Vec<(Core, Program<Target, M, W>)> = Vec::new();
        let mut max_length_unit_core_map = Vec::new();
        let refused = generate_prog_ir_for_program_unit::<Target, M, W>(
            ProgramUnitToLower {
                // ⛔ CORE 1 FIRST, so the tie-break has an incumbent to displace.
                units: UnitHandles::of(handle(1, 0), vec![handle(0, 0), handle(1, 1)]),
                program: UnitProgramToLower {
                    estimated_instructions: 4,
                    body: vec![
                        op(0, OpKind::Nop { dbg_name: None }),
                        op(
                            1,
                            OpKind::If {
                                predicate: CmpPredicate::Eq,
                                operands: CmpOperands::LccrVsImm {
                                    lccr: RegIndex::at::<1>(),
                                    imm: CmpImm::Constant(2),
                                },
                                else_region: ElseRegion::None,
                                results: Vec::new(),
                                next: OpSite(9),
                            },
                        ),
                        // Labelled by the JCMP above and emitting nothing, so it earns a NOP.
                        op(9, OpKind::Common { next: None }),
                        op(10, OpKind::Return),
                    ],
                },
            },
            &mut psinfo_map,
            &mut max_length_unit_core_map,
            &mut pass,
        );
        assert_eq!(refused, Vec::new());
        assert_eq!(
            unit_folds,
            vec![
                (key(at(1)), Some(FoldId(0))),
                (key(at(0)), Some(FoldId(0))),
                (key(at(1)), Some(FoldId(3))),
            ],
            "every handle is cached, and fold 1 is the SDSC's third fold"
        );
        assert_eq!(
            code_quality,
            Vec::new(),
            "`CollectCodeQualityStats` is off by default"
        );
        // ⛔ THE NOP IS GONE AND THE RETURN CARRIES ITS LABEL.
        let filed = |core: u32| {
            first_code_block(
                &psinfo_map
                    .iter()
                    .find(|(held, _)| *held == at(core))
                    .expect("the core files a program")
                    .1,
                Component::Pe,
            )
            .expect("one CODE block")
            .iter()
            .map(|instr| (instr.opcode, instr.tag.clone()))
            .collect::<Vec<_>>()
        };
        assert_eq!(
            filed(1),
            vec![
                (OpCode::NOP, None),
                (OpCode::JCMP, None),
                (OpCode::RETURN, Some("if-label-0-end".to_owned())),
            ]
        );
        assert_eq!(filed(0), filed(1), "one body, filed under both cores");
        assert_eq!(
            psinfo_map.len(),
            2,
            "fold 1 is core 1 again and files nothing new"
        );
        assert_eq!(
            psinfo_map[0].1.reg_state,
            Vec::new(),
            "an empty register graph is erased, not filed empty"
        );
        assert_eq!(
            max_length_unit_core_map,
            vec![(Component::Pe, at(0))],
            "the tie went to the lower core"
        );
    }
}
