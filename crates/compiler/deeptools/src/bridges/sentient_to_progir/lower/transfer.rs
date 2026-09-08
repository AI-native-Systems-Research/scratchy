//! THE TRANSFER OPERATION HANDLERS — load_and_send, receive_and_store, load_and_store, the
//! scalar extracts, load_compute_and_send and copy.
//!
//! ⭐ THESE ARE FIVE OF THE SEVEN SENTIENT OPS THE REFERENCE ACTUALLY EMITS across all 417
//! staged programs, so this file is on the hot path for every program.
//!
//! 7 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e099_LowerLoadAndSendOperation` | 3 | 29 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:144` |
//! | `e100_LowerReceiveAndStoreOperation` | 3 | 49 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:174` |
//! | `e101_LowerLoadAndStoreOperation` | 3 | 55 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:224` |
//! | `e102_LowerLoadAndExtractScalarOperation` | 3 | 24 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:280` |
//! | `e103_LowerReceiveAndExtractScalarOperation` | 3 | 13 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:305` |
//! | `e104_LowerLoadComputeAndSendOperation` | 3 | 33 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:319` |
//! | `e122_LowerCopyOperation` | 4 | 95 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:406` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::scalar::{
    AddrFile, Assign, AssignKind, construct_assign_instr,
};
use crate::bridges::sentient_to_progir::construct::transfer::{
    ExtractScalarLoad, ExtractScalarRefusal, L3Indirection, L3Load, L3LoadAndStore, L3Store,
    L3StoreSource, LdzConst, Load, LoadCompute, LoadRefusal, LoadUnit, LrfCopy, LrfCopyRefusal,
    Store, StoreRefusal, construct_extract_scalar_load_instr, construct_l3_load_and_store_instr,
    construct_l3_load_instr, construct_load_compute_instr, construct_load_instr,
    construct_lrf_copy_instr, construct_store_instr, construct_zr_assign_instr,
};
use crate::bridges::sentient_to_progir::lower::compute::{
    LabelPlacement, Successor, lower_common_operations,
};
use crate::bridges::sentient_to_progir::lower::control::CodeGraph;
use crate::bridges::sentient_to_progir::lower::labels_and_regs::{
    AddressScale, LoweredOp, RegImm, RegImmSource, add_to_reg_init, address_scale,
    get_reg_imm_vals, update_label_and_add_to_code_graph,
};
use crate::bridges::sentient_to_progir::state::{
    CopyOps, Labels, OpSite, RegGraphs, RegsToInit, UnitKey,
};
use crate::bridges::sentient_to_progir::uniform::block::{
    InstrIndex, UniformInstrBlock, UniformInstrBlocks,
};
use crate::bridges::sentient_to_progir::uniform::instr::{OperandMapRefusal, UniformInstrInfo};
use crate::formats::Bits;
use crate::islands::progir::ty::FoldId;
use crate::islands::sentient::dialects::sentient::{Reg, RegType as SenRegType};
use sys_arch_spec::regfile::Component;

/// WHAT EVERY TRANSFER LOWERING WRITES INTO — the reference's by-reference parameters and the two
/// register-initialisation members of the pass, as one place.
///
/// ⭐ ONE STRUCT BECAUSE ALL SIX LOWERINGS TAKE THE SAME FIVE, and threading them positionally is how
/// `labels` and `regs_to_init` end up swapped at one callsite out of six.
pub struct LowerTo<'a> {
    /// `labels` — which op carries which branch label.
    pub labels: &'a Labels,
    /// `uniform_instr_region` — the blocks the instructions land in.
    pub blocks: &'a mut UniformInstrBlocks,
    /// `regs_to_init_` — which registers each unit must initialise.
    pub regs_to_init: &'a mut RegsToInit,
    /// `getUnitsFromOperation(op)` — the units an instruction's operand map answers for.
    pub units: &'a [UnitKey],
    /// `full_reg_init_` — whether this run records registers to initialise at all.
    pub full_reg_init: bool,
}

/// WHAT ONE TRANSFER LOWERING COULD NOT LOWER — every offender its constructors hand back, plus its
/// own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferRefusal {
    /// `emitError("Unable to create assign instruction")` + `signalPassFailure` (`:302-305`): the
    /// address and the result are not in two registers of one file.
    Uncopyable {
        /// `getMutableAddr()`'s register.
        src: Reg,
        /// `getResult()`'s register.
        tgt: Reg,
    },
    /// What the copy's per-unit operand map refused.
    Copy(OperandMapRefusal),
    /// What the load refused.
    Load(LoadRefusal),
    /// What the store refused.
    Store(StoreRefusal),
    /// What the scalar-extract load refused.
    ExtractScalar(ExtractScalarRefusal),
    /// What the LRF copy refused.
    LrfCopy(LrfCopyRefusal),
    /// What the load-compute refused.
    LoadCompute(OperandMapRefusal),
    /// `DT_CHECK(is_constant || is_symbol || is_multicast || is_get_unit)` (`:466`): a program
    /// header's input is a register, or a query map of mixed kinds.
    NotAnImmediate,
    /// The sen1p5 EBR's `DT_CHECK` that its input is a `sentient.constant` or a `uniform.query_map`
    /// (`:431-433`) — nothing else has a value to shift.
    EbrInputNotEvaluable,
    /// *"sen1p5 EBR requires element size to initialize reg"* (`:436-437`).
    EbrElementSize,
    /// *"EBR is not divisible by an even number of sticks"* (`:440-441`), and by what it was not.
    EbrNotEvenSticks(i64),
}

/// WHAT ONE TRANSFER LOWERING COST AND WHAT IT COULD NOT DO.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lowered {
    /// `++num_copy_ops_` for the copies this lowering materialised.
    pub copy_ops: CopyOps,
    /// The offenders, in the order they arose.
    pub refused: Vec<TransferRefusal>,
}

/// THE LOAD `ConstructLoadInstr`'s COMPONENT SWITCH PICKS — ⭐ ONE ENUM BECAUSE OUR PORT SPLIT THAT
/// FUNCTION'S L3 HEAD OFF, mirroring [`Store`]'s two arms so the two halves of a transfer read alike.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadOf<'a> {
    /// `comp == L3LU` — an `L3_LD*`.
    L3(L3Load),
    /// Any other load unit.
    Unit {
        /// Which unit, which decides the field names.
        unit: LoadUnit,
        /// The load itself.
        load: Load<'a>,
    },
}

impl LoadOf<'_> {
    /// `load_op.getMutableAddr()` — ⭐ THE SAME FIELD IN BOTH SHAPES.
    const fn mutable_addr(&mut self) -> &mut Reg {
        match self {
            LoadOf::L3(l3) => &mut l3.mutable_addr,
            LoadOf::Unit { load, .. } => &mut load.mutable_addr,
        }
    }

    /// `load_op.getResult()`.
    const fn result(&self) -> Reg {
        match self {
            LoadOf::L3(l3) => l3.result,
            LoadOf::Unit { load, .. } => load.result,
        }
    }
}

/// `store_op.getMutableAddr()` — ⛔ A FREE FUNCTION, NOT AN ACCESSOR ON [`Store`]: the two store
/// shapes are the constructor's, and this seam is the only reader that needs the field mutably.
const fn store_mutable_addr<'s>(store: &'s mut Store<'_>) -> &'s mut Reg {
    match store {
        Store::L3(l3) => &mut l3.mutable_addr,
        Store::Unit(unit) => &mut unit.mutable_addr,
    }
}

/// `store_op.getResult()`.
const fn store_result(store: &Store<'_>) -> Reg {
    match store {
        Store::L3(l3) => l3.result,
        Store::Unit(unit) => unit.result,
    }
}

/// THE COPY ONE TRANSFER MATERIALISES BEFORE ITSELF — what `ConstructAssignInstr` is called with
/// (`:151-152`).
struct Copy {
    /// The unit the copy runs on.
    comp: Component,
    /// `getMutableAddr()`'s register.
    src: Reg,
    /// `getResult()`'s register.
    tgt: Reg,
    /// `getElementSize()` — ⭐ DEAD ON EVERY REGISTER-TO-REGISTER ARM (only the immediate arms scale
    /// by it), and passed anyway because the op carries it and the reference hands it over.
    element_size: Bits,
}

/// WHICH TWO FILES A LOWERING'S IMPLICIT COPY IS BETWEEN — the four register-to-register arms of
/// `ConstructAssignInstr` (`:213-260`).
///
/// ⛔ `None` IS THAT FUNCTION'S FINAL `else`, an `emitError` + `signalPassFailure`, and never a
/// silently dropped copy. ⛔ AND NEVER BETWEEN LAR AND EAR, which is why the pair is matched whole.
const fn reg_to_reg(src: Reg, tgt: Reg) -> Option<AssignKind> {
    match (src.locale, tgt.locale) {
        (SenRegType::Jcr, SenRegType::Jcr) => Some(AssignKind::JcrFromJcr),
        (SenRegType::Lrf, SenRegType::Lrf) => Some(AssignKind::LrfFromLrf),
        (SenRegType::Lar, SenRegType::Lar) => Some(AssignKind::AddrFromAddr(AddrFile::Lar)),
        (SenRegType::Ear, SenRegType::Ear) => Some(AssignKind::AddrFromAddr(AddrFile::Ear)),
        _ => None,
    }
}

/// `ConstructAssignInstr` + `updateLabelAndAddToCodeGraph` (`:151-156`), which four lowerings run
/// identically, and whose answer is whether the op's label landed on the COPY
/// (`getLastInstr().getTag()`, `:242-246`).
///
/// ⛔ `program_header` IS FALSE: none of the four passes one, so the immediate arms never divert.
fn emit_copy<A: Arch>(
    copy: &Copy,
    at: OpSite,
    force_not_add_label: bool,
    to: &mut LowerTo<'_>,
    lowered: &mut Lowered,
) -> bool {
    let Some(kind) = reg_to_reg(copy.src, copy.tgt) else {
        lowered.refused.push(TransferRefusal::Uncopyable {
            src: copy.src,
            tgt: copy.tgt,
        });
        return false;
    };
    let assigned = construct_assign_instr::<A>(
        &Assign {
            src_index: copy.src.index,
            tgt_index: copy.tgt.index,
            kind,
        },
        copy.comp,
        copy.element_size,
        false,
    );
    lowered.copy_ops.0 += assigned.copy_ops.0;
    lowered
        .refused
        .extend(assigned.refused.into_iter().map(TransferRefusal::Copy));
    let Some(instr) = assigned.instr else {
        return false;
    };
    update_label_and_add_to_code_graph(
        to.labels,
        to.blocks,
        at,
        LoweredOp::Other,
        instr,
        force_not_add_label,
    );
    to.blocks
        .blocks
        .last_mut()
        .and_then(UniformInstrBlock::last_instr_mut)
        .is_some_and(|last| last.tag.is_some())
}

/// `ConstructLoadInstr`'s two halves, rejoined.
fn build_load<A: Arch>(
    load: &LoadOf<'_>,
    to: &mut LowerTo<'_>,
    lowered: &mut Lowered,
) -> UniformInstrInfo {
    match load {
        LoadOf::L3(l3) => {
            construct_l3_load_instr::<A>(l3, to.units, to.full_reg_init, to.regs_to_init)
        }
        LoadOf::Unit { unit, load } => {
            let (instr, refused) =
                construct_load_instr::<A>(*unit, load, to.units, to.full_reg_init, to.regs_to_init);
            lowered
                .refused
                .extend(refused.into_iter().map(TransferRefusal::Load));
            instr
        }
    }
}

/// Replaces: e099_LowerLoadAndSendOperation
///
/// Lower a `sentient.load_and_send`: the load, preceded by a copy of its address register into the
/// register its own result names when the two differ.
///
/// ⛔ THE COPY TAKES THE OP'S LABEL AND THE LOAD IS THEN FORCED NOT TO (`:157-158,167-168`), so a
/// jump to this op lands on the copy.
/// ⛔ AND THE SUBSTITUTED ADDRESS IS RESTORED (`:159-166`): the copy exists only in ProgIR, so the
/// description this seam was handed comes back unchanged.
pub fn lower_load_and_send_operation<A: Arch>(
    comp: Component,
    load: &mut LoadOf<'_>,
    element_size: Bits,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let src = *load.mutable_addr();
    let tgt = load.result();
    let copied = src.index != tgt.index;
    if copied {
        emit_copy::<A>(
            &Copy {
                comp,
                src,
                tgt,
                element_size,
            },
            at,
            false,
            to,
            &mut lowered,
        );
        load.mutable_addr().index = tgt.index;
    }
    let instr = build_load::<A>(load, to, &mut lowered);
    if copied {
        load.mutable_addr().index = src.index;
    }
    update_label_and_add_to_code_graph(to.labels, to.blocks, at, LoweredOp::Other, instr, copied);
    lowered
}

/// Replaces: e100_LowerReceiveAndStoreOperation
///
/// Lower a `sentient.receive_and_store`: the store, its address copy, and — for an L3LU storing a
/// non-zero 16-bit constant — the `L3_LDZimm16` that puts that value in the ZR first.
///
/// ⛔ THE ZR ASSIGN TAKES THE LABEL WHEN IT EXISTS, and everything after it is forced not to
/// (`:184-190`), which is what `is_ldz` carries into both later calls.
/// ⭐ THE COMPONENT TEST IS THE `Store::L3` ARM'S OWN: only that shape takes a constant at all.
pub fn lower_receive_and_store_operation<A: Arch>(
    comp: Component,
    store: &mut Store<'_>,
    element_size: Bits,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let is_ldz = match store {
        Store::L3(L3Store {
            source: L3StoreSource::Constant(LdzConst::Fp16(imm)),
            ..
        }) if *imm != 0 => {
            let zr = construct_zr_assign_instr(*imm);
            update_label_and_add_to_code_graph(
                to.labels,
                to.blocks,
                at,
                LoweredOp::Other,
                zr,
                false,
            );
            true
        }
        _ => false,
    };
    let src = *store_mutable_addr(store);
    let tgt = store_result(store);
    let copied = src.index != tgt.index;
    if copied {
        emit_copy::<A>(
            &Copy {
                comp,
                src,
                tgt,
                element_size,
            },
            at,
            is_ldz,
            to,
            &mut lowered,
        );
        store_mutable_addr(store).index = tgt.index;
    }
    let (instr, refused) =
        construct_store_instr::<A>(store, to.units, to.full_reg_init, to.regs_to_init);
    lowered
        .refused
        .extend(refused.into_iter().map(TransferRefusal::Store));
    if copied {
        store_mutable_addr(store).index = src.index;
    }
    update_label_and_add_to_code_graph(
        to.labels,
        to.blocks,
        at,
        LoweredOp::Other,
        instr,
        copied || is_ldz,
    );
    lowered
}

/// Replaces: e101_LowerLoadAndStoreOperation
///
/// Lower a `sentient.load_and_store`: the L3 transfer, preceded by up to two address copies — one per
/// half — and whichever of the three instructions ends up carrying the op's label.
///
/// ⛔ AN IBR WRITE HAS NO DESTINATION COPY (`:250`), because it has no destination address register.
/// ⛔ AND THE LABEL IS CLAIMED ONCE ACROSS ALL THREE: `label_added_already` is re-read from the block
/// after each copy, because a copy the constructor declined to emit did not take the label.
pub fn lower_load_and_store_operation<A: Arch>(
    comp: Component,
    ls: &mut L3LoadAndStore,
    element_size: Bits,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let (src0, src1) = (ls.src_mutable_addr, ls.dst_mutable_addr);
    let (tgt0, tgt1) = (ls.results[0], ls.results[1]);
    let copy_src = src0.index != tgt0.index;
    let copy_dst = !matches!(ls.indirection, L3Indirection::IbrWrite) && src1.index != tgt1.index;
    let mut label_added_already = false;
    if copy_src {
        label_added_already |= emit_copy::<A>(
            &Copy {
                comp,
                src: src0,
                tgt: tgt0,
                element_size,
            },
            at,
            false,
            to,
            &mut lowered,
        );
        ls.src_mutable_addr.index = tgt0.index;
    }
    if copy_dst {
        label_added_already |= emit_copy::<A>(
            &Copy {
                comp,
                src: src1,
                tgt: tgt1,
                element_size,
            },
            at,
            label_added_already,
            to,
            &mut lowered,
        );
        ls.dst_mutable_addr.index = tgt1.index;
    }
    let instr =
        construct_l3_load_and_store_instr::<A>(ls, to.units, to.full_reg_init, to.regs_to_init);
    if copy_src {
        ls.src_mutable_addr.index = src0.index;
    }
    if copy_dst {
        ls.dst_mutable_addr.index = src1.index;
    }
    update_label_and_add_to_code_graph(
        to.labels,
        to.blocks,
        at,
        LoweredOp::Other,
        instr,
        label_added_already,
    );
    lowered
}

/// WHETHER A `sentient.load_and_extract_scalar`'s CONSUMER IS THE UNIT RUNNING IT — the witness for
/// *"expecting load_and_extract_scalar op consumer to be self"* (`:296-297`).
///
/// ⛔ A WITNESS, NOT AN ASSERT INSIDE THE LOWERING: the check has nothing to do with the instruction,
/// so it belongs where the consumer is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfConsumer;

impl SelfConsumer {
    /// `unit_op.getRegion().getArguments().empty() ? getUnits()[0] : getArguments()[0]`, compared to
    /// `getConsumer()` (`:290-297`).
    ///
    /// ⚠️ THE REFERENCE READS `getUnits()[0]` UNCHECKED, so a program unit naming no unit reads off
    /// the end there; `None` is that case here as well as a consumer that is someone else.
    #[must_use]
    pub fn of(
        consumer: UnitKey,
        region_args: &[UnitKey],
        units: &[UnitKey],
    ) -> Option<SelfConsumer> {
        let self_unit = region_args.first().or_else(|| units.first())?;
        (consumer == *self_unit).then_some(SelfConsumer)
    }
}

/// Replaces: e102_LowerLoadAndExtractScalarOperation
///
/// Lower a `sentient.load_and_extract_scalar`: one LXLU load, and nothing else.
///
/// ⛔ NO COMPONENT PARAMETER — LXLU is the only one (`:287-288`), and the constructor hardcodes it.
/// ⭐ THE REFERENCE READS THE ADDRESS REGISTER INTO A LOCAL AND NEVER USES IT (`:290`), so there is
/// no address copy here: this is the one transfer whose result is not an updated address.
pub fn lower_load_and_extract_scalar_operation<A: Arch>(
    load: &ExtractScalarLoad,
    _consumer: SelfConsumer,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let (instr, refused) =
        construct_extract_scalar_load_instr::<A>(load, to.units, to.full_reg_init, to.regs_to_init);
    lowered
        .refused
        .extend(refused.into_iter().map(TransferRefusal::ExtractScalar));
    update_label_and_add_to_code_graph(to.labels, to.blocks, at, LoweredOp::Other, instr, false);
    lowered
}

/// Replaces: e103_LowerReceiveAndExtractScalarOperation
///
/// Lower a `sentient.receive_and_extract_scalar`: one LXSU `LRFCOPY`, and nothing else.
///
/// ⛔ NO COMPONENT PARAMETER — LXSU is the only one (`:311-312`).
pub fn lower_receive_and_extract_scalar_operation(
    copy: &LrfCopy,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let (instr, refused) =
        construct_lrf_copy_instr(copy, to.units, to.full_reg_init, to.regs_to_init);
    lowered
        .refused
        .extend(refused.into_iter().map(TransferRefusal::LrfCopy));
    update_label_and_add_to_code_graph(to.labels, to.blocks, at, LoweredOp::Other, instr, false);
    lowered
}

/// Replaces: e104_LowerLoadComputeAndSendOperation
///
/// Lower a `sentient.load_compute_and_send`: the load-compute, preceded by the same address copy the
/// plain load takes.
///
/// ⛔ THE COPY IS SCALED BY THE **SOURCE** ELEMENT SIZE (`:326-328`), whose own comment says the
/// address calculation follows the destination — the reference passes `getSrcElementSize()` regardless.
pub fn lower_load_compute_and_send_operation<A: Arch>(
    comp: Component,
    load: &mut LoadCompute<'_>,
    src_element_size: Bits,
    at: OpSite,
    to: &mut LowerTo<'_>,
) -> Lowered {
    let mut lowered = Lowered::default();
    let (src, tgt) = (load.mutable_addr, load.result);
    let copied = src.index != tgt.index;
    if copied {
        emit_copy::<A>(
            &Copy {
                comp,
                src,
                tgt,
                element_size: src_element_size,
            },
            at,
            false,
            to,
            &mut lowered,
        );
        load.mutable_addr.index = tgt.index;
    }
    let (instr, refused) =
        construct_load_compute_instr::<A>(load, to.units, to.full_reg_init, to.regs_to_init);
    lowered
        .refused
        .extend(refused.into_iter().map(TransferRefusal::LoadCompute));
    if copied {
        load.mutable_addr.index = src.index;
    }
    update_label_and_add_to_code_graph(to.labels, to.blocks, at, LoweredOp::Other, instr, copied);
    lowered
}

/// WHAT A `sentient.scalar_copy`'s INPUT IS — the three shapes its two routes turn on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CopyInput<'a> {
    /// A constant, a symbol, a multicast, a `dataflow.get_unit`, or a `uniform.query_map` over them.
    Imm(&'a RegImmSource),
    /// A register, which is none of those — ⛔ A PROGRAM HEADER REFUSES IT (`:466`).
    Reg,
    /// `isa<BlockArgument>` — ⛔ IT TAKES THE ASSIGN ROUTE EVEN IN A PROGRAM HEADER (`:463`).
    BlockArg,
}

/// THE `sentient.scalar_copy` FIELDS ITS LOWERING READS.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarCopy<'a> {
    /// `getInp()`.
    pub input: CopyInput<'a>,
    /// `ConstructAssignInstr`'s view of `getInp() -> getOut()`, for the route that emits one.
    pub assign: Assign,
    /// `getRegLocale()`/`getRegIndex()` — the register `getOut()` names.
    pub reg: Reg,
    /// The `element_size` attribute.
    pub element_size: Option<Bits>,
    /// `getProgramHeader()`.
    pub program_header: bool,
    /// `getQueryKeyAndUnitsFromParentRegion`'s units, each with the fold it stands for.
    pub units: &'a [(UnitKey, Option<FoldId>)],
}

/// WHAT LOWERING ONE `sentient.scalar_copy` COST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Copied {
    /// The copies materialised, and the offenders.
    pub lowered: Lowered,
    /// ⛔ `None` IS THE ASSIGN ROUTE, which places the label itself rather than moving it on.
    pub label: Option<LabelPlacement>,
}

/// WHICH OF THE FOUR DEFINING OPS ONE INPUT IS — `isConstant` (`Utils/Utils.cpp:423-443`), `isSymbol`,
/// `isMulticast` and `isGetUnit` (`Transform/Sentient/Analyses/Utils.cpp:141,221,240`).
///
/// ⛔ ALL-OF FOR THREE OF THEM AND ANY-OF FOR THE SYMBOL, which is what makes a query map mixing
/// symbols with constants a symbol rather than nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputKinds {
    /// Every value is a `dataflow.create_multicast_group`.
    is_multicast: bool,
    /// Any value is a `symbol.create_symbol`.
    is_symbol: bool,
    /// The input is one of the four kinds at all.
    is_one_kind: bool,
}

impl InputKinds {
    /// The four predicates over one input's values.
    fn of(source: &RegImmSource) -> InputKinds {
        let held: Vec<RegImm> = match source {
            RegImmSource::Common(imm) => vec![*imm],
            // ⛔ A UNIT THE MAPPING MISSES IS NOT A VALUE: the reference reads the mapping's own
            // values, not one per unit, so an uncovered unit cannot vote on the kind.
            RegImmSource::Mapped(entries) => {
                entries.iter().filter_map(|(_, _, held)| *held).collect()
            }
        };
        let all = |kind: fn(&RegImm) -> bool| !held.is_empty() && held.iter().all(kind);
        InputKinds {
            is_multicast: all(|imm| matches!(imm, RegImm::Multicast(_))),
            is_symbol: held.iter().any(|imm| matches!(imm, RegImm::Symbol(_))),
            is_one_kind: all(|imm| matches!(imm, RegImm::Constant(_)))
                || all(|imm| matches!(imm, RegImm::Multicast(_)))
                || all(|imm| matches!(imm, RegImm::Unit(_)))
                || held.iter().any(|imm| matches!(imm, RegImm::Symbol(_))),
        }
    }
}

/// `evaluateShift(ev, 1, /*shift_right*/ true)` and the two `DT_CHECK`s around it (`:427-452`).
///
/// ⭐ ONLY A CONSTANT SHIFTS — a symbol id or a multicast id is not an address the evaluator halves,
/// and `None` back means nothing shifted because one of the checks named an offender.
fn halve_for_sen1p5_ebr<A: Arch>(
    source: Option<&RegImmSource>,
    element_size: Option<Bits>,
    refused: &mut Vec<TransferRefusal>,
) -> Option<RegImmSource> {
    let evaluable = matches!(
        source,
        Some(RegImmSource::Common(RegImm::Constant(_)) | RegImmSource::Mapped(_))
    );
    if !evaluable {
        refused.push(TransferRefusal::EbrInputNotEvaluable);
        return None;
    }
    let Some(width) = element_size
        .map(|bits| u64::from(bits.0))
        .filter(|width| *width > 0)
    else {
        refused.push(TransferRefusal::EbrElementSize);
        return None;
    };
    // ⛔ TRUNCATE THEN DOUBLE, in that order: `2 * num_elems_in_stick` over a stick's element
    // count (`:438-441`), which is not `2 * bits / width` when the width does not divide the stick.
    let two_sticks = 2 * (A::BYTES_PER_STICK.get() * 8 / width);
    let mut halve = |value: i64| {
        // ⛔ AND A WIDTH WIDER THAN TWO STICKS DIVIDES BY NOTHING, which is that check failing.
        if value.unsigned_abs().checked_rem(two_sticks) != Some(0) {
            refused.push(TransferRefusal::EbrNotEvenSticks(value));
        }
        value >> 1
    };
    Some(match source? {
        RegImmSource::Common(RegImm::Constant(value)) => {
            RegImmSource::Common(RegImm::Constant(halve(*value)))
        }
        RegImmSource::Mapped(entries) => RegImmSource::Mapped(
            entries
                .iter()
                .map(|(unit, fold, held)| {
                    let held = match *held {
                        Some(RegImm::Constant(value)) => Some(RegImm::Constant(halve(value))),
                        held => held,
                    };
                    (*unit, *fold, held)
                })
                .collect(),
        ),
        common => common.clone(),
    })
}

/// Replaces: e122_LowerCopyOperation
///
/// One `sentient.scalar_copy`: a program header's immediate initialises the register, and an assign
/// instruction carries everything else.
///
/// ⛔ `element_size` `None` IS THE ABSENT ATTRIBUTE **AND** THE `-1` (`:418-422,456-460`), worth
/// `8 * scale` — the identity scaling — since no width is negative, and that is where the check went.
/// ⛔ A REFUSED ASSIGN EMITS NOTHING, where the reference's unchecked `.value()` (`:498`) aborts; and
/// the parent-region walk's two checks are the caller's, because the units arrive as a parameter.
#[must_use]
pub fn lower_copy_operation<A: Arch>(
    comp: Component,
    copy: &ScalarCopy<'_>,
    next: Option<Successor>,
    graph: CodeGraph<'_>,
    reg_graph: &mut RegGraphs,
    nop_for_labels: &mut Vec<InstrIndex>,
) -> Copied {
    let mut lowered = Lowered::default();
    let source = match copy.input {
        CopyInput::Imm(source) => Some(source),
        CopyInput::Reg | CopyInput::BlockArg => None,
    };
    let kinds = source.map(InputKinds::of);
    let scale = if kinds.is_some_and(|kinds| kinds.is_multicast) {
        AddressScale::unscaled()
    } else {
        address_scale::<A>(comp)
    };
    let element_size = copy.element_size.unwrap_or(Bits(8 * scale.get()));
    let shifted = (A::GEN >= IsaGen::Sen1p5 && copy.reg.locale == SenRegType::Ebr)
        .then(|| halve_for_sen1p5_ebr::<A>(source, copy.element_size, &mut lowered.refused))
        .flatten();
    let source = shifted.as_ref().or(source);
    if copy.program_header && !matches!(copy.input, CopyInput::BlockArg) {
        let (Some(source), Some(kinds)) = (source, kinds) else {
            lowered.refused.push(TransferRefusal::NotAnImmediate);
            return Copied {
                lowered,
                label: None,
            };
        };
        if !kinds.is_one_kind {
            lowered.refused.push(TransferRefusal::NotAnImmediate);
            return Copied {
                lowered,
                label: None,
            };
        }
        let imm_vals = get_reg_imm_vals::<A>(
            source,
            copy.units,
            comp,
            element_size,
            scale,
            copy.reg.locale == SenRegType::Mvr,
        );
        add_to_reg_init(copy.reg, &imm_vals, reg_graph, kinds.is_symbol);
        Copied {
            lowered,
            label: Some(lower_common_operations(graph, next, nop_for_labels)),
        }
    } else {
        let assigned =
            construct_assign_instr::<A>(&copy.assign, comp, element_size, copy.program_header);
        lowered.copy_ops.0 += assigned.copy_ops.0;
        lowered
            .refused
            .extend(assigned.refused.into_iter().map(TransferRefusal::Copy));
        if let Some(instr) = assigned.instr {
            update_label_and_add_to_code_graph(
                graph.labels,
                graph.region,
                graph.at,
                LoweredOp::Other,
                instr,
                false,
            );
        }
        Copied {
            lowered,
            label: None,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Sen1p5;
    use crate::arch::{Bounded, Dd2, Elements};
    use crate::bridges::sentient_to_progir::construct::transfer::{
        L3Half, LdzConst, LoadComputeConsumer, LoadComputeShuffle, LoadTarget, Peer,
    };
    use crate::bridges::sentient_to_progir::uniform::block::UniformInstrBlock;
    use crate::bridges::sentient_to_progir::utils::LoadConsumer;
    use crate::islands::progir::ty::{Operand, OperandValue, RegType};
    use crate::islands::progir::{OpCode, OperandField, RegInit};
    use crate::islands::sentient::dialects::sentient::RegIndex;
    use crate::units::Core;
    use crate::units::{Corelet, DfirUnit, Residency};

    fn at(index: usize) -> Option<RegIndex> {
        Some(RegIndex::ALL[index])
    }

    fn reg(locale: SenRegType, index: usize) -> Reg {
        Reg {
            locale,
            index: at(index),
        }
    }

    /// The instructions one block holds.
    fn emitted(blocks: &UniformInstrBlocks) -> Vec<UniformInstrInfo> {
        match &blocks.blocks[0] {
            UniformInstrBlock::Regular(instrs) => instrs.clone(),
            UniformInstrBlock::Uniform(_) => panic!("a regular block"),
        }
    }

    /// One labelled op, and somewhere for its instructions to go.
    fn site() -> (Labels, UniformInstrBlocks, RegsToInit) {
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "L0".to_owned());
        (labels, UniformInstrBlocks::default(), RegsToInit::new())
    }

    #[test]
    fn a_send_whose_address_and_result_differ_copies_the_address_first_and_gives_it_the_label() {
        let (labels, mut blocks, mut regs) = site();
        let mut load = LoadOf::L3(L3Load {
            mutable_addr: reg(SenRegType::Lar, 0),
            immutable_addr: reg(SenRegType::Lbr, 0),
            result: reg(SenRegType::Lar, 1),
            target: LoadTarget::Peer {
                peer: Peer::Unnamed,
                dir: None,
            },
            burst: Elements(1),
            update: false,
            dbg_name: None,
        });
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered = lower_load_and_send_operation::<Dd2>(
            Component::L3lu,
            &mut load,
            Bits(8),
            OpSite(0),
            &mut to,
        );
        let instrs = emitted(&blocks);
        // `uniform_scalar_sub.mlir:10` — `L3_LARREGCOPY :: src0:1 src1:0  // LAR <- LAR`, then the
        // transfer reading the register that copy wrote.
        assert_eq!(instrs[0].opcode, OpCode::LARREGCOPY);
        assert_eq!(
            instrs[0].common_field(OperandField::Src0),
            Some(&Operand::every(OperandValue::Int(1)))
        );
        assert_eq!(
            instrs[0].common_field(OperandField::Src1),
            Some(&Operand::every(OperandValue::Int(0)))
        );
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(instrs[1].opcode, OpCode::ST);
        assert_eq!(
            instrs[1].common_field(OperandField::Src0),
            Some(&Operand::every(OperandValue::Int(1))),
            "the transfer reads the substituted register"
        );
        assert_eq!(instrs[1].tag, None, "the label was already claimed");
        // ⛔ AND THE DESCRIPTION CAME BACK UNCHANGED.
        assert_eq!(*load.mutable_addr(), reg(SenRegType::Lar, 0));
        assert_eq!(
            lowered,
            Lowered {
                copy_ops: CopyOps(1),
                refused: Vec::new(),
            }
        );
    }

    #[test]
    fn an_l3_store_of_a_non_zero_fp16_constant_loads_the_zr_first_and_that_takes_the_label() {
        let (labels, mut blocks, mut regs) = site();
        let l3_store = |value: u16| {
            Store::L3(L3Store {
                mutable_addr: reg(SenRegType::Lar, 0),
                immutable_addr: reg(SenRegType::Lbr, 0),
                result: reg(SenRegType::Lar, 0),
                source: L3StoreSource::Constant(LdzConst::Fp16(value)),
                burst: Elements(1),
                update: false,
                dbg_name: None,
            })
        };
        let mut store = l3_store(23129);
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered = lower_receive_and_store_operation::<Dd2>(
            Component::L3lu,
            &mut store,
            Bits(16),
            OpSite(0),
            &mut to,
        );
        let instrs = emitted(&blocks);
        assert_eq!(instrs[0].opcode, OpCode::LDZimm16);
        assert_eq!(
            instrs[0].common_field(OperandField::Imm),
            Some(&Operand::every(OperandValue::Int(23129)))
        );
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(instrs[1].tag, None);
        assert_eq!(lowered.copy_ops, CopyOps(0), "one address, no copy");
        assert!(lowered.refused.is_empty());
        // ⛔ A ZERO CONSTANT IS ALREADY IN THE ZR — no `LDZimm16`, and the store keeps the label.
        let mut zero = l3_store(0);
        let (labels, mut blocks, mut regs) = site();
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        lower_receive_and_store_operation::<Dd2>(
            Component::L3lu,
            &mut zero,
            Bits(16),
            OpSite(0),
            &mut to,
        );
        let instrs = emitted(&blocks);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
    }

    #[test]
    fn a_load_and_store_copies_both_halves_addresses_and_only_the_first_copy_takes_the_label() {
        let (labels, mut blocks, mut regs) = site();
        let mut ls = L3LoadAndStore {
            half: L3Half::Load { multicast: None },
            indirection: L3Indirection::Direct,
            src_mutable_addr: reg(SenRegType::Lar, 0),
            src_immutable_addr: reg(SenRegType::Lbr, 0),
            dst_mutable_addr: reg(SenRegType::Ear, 1),
            dst_immutable_addr: reg(SenRegType::Ebr, 0),
            results: [reg(SenRegType::Lar, 1), reg(SenRegType::Ear, 0)],
            src_is_lx: false,
            dst_is_lx: false,
            dir: None,
            burst: Elements(1),
            update: false,
            dbg_name: None,
        };
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered = lower_load_and_store_operation::<Dd2>(
            Component::L3lu,
            &mut ls,
            Bits(8),
            OpSite(0),
            &mut to,
        );
        let instrs = emitted(&blocks);
        // `symbolic_ebr.mlir:46,54` — `L3_LARREGCOPY :: src0:1 src1:0` then
        // `L3_EARREGCOPY :: src0:0 src1:1`, both before the transfer.
        assert_eq!(instrs[0].opcode, OpCode::LARREGCOPY);
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(instrs[1].opcode, OpCode::EARREGCOPY);
        assert_eq!(
            instrs[1].common_field(OperandField::Src0),
            Some(&Operand::every(OperandValue::Int(0)))
        );
        assert_eq!(
            instrs[1].common_field(OperandField::Src1),
            Some(&Operand::every(OperandValue::Int(1)))
        );
        assert_eq!(
            instrs[1].tag, None,
            "the label is claimed once across all three"
        );
        assert_eq!(instrs[2].tag, None);
        assert_eq!(instrs.len(), 3);
        assert_eq!(ls.src_mutable_addr, reg(SenRegType::Lar, 0));
        assert_eq!(ls.dst_mutable_addr, reg(SenRegType::Ear, 1));
        assert_eq!(lowered.copy_ops, CopyOps(2));
        assert!(lowered.refused.is_empty());
    }

    #[test]
    fn an_extract_scalar_load_emits_one_self_consuming_load_and_no_address_copy() {
        let (labels, mut blocks, mut regs) = site();
        let core = Core::checked(0).expect("core 0");
        let key = |unit| UnitKey::of(unit, Residency::CoreWide { core }).expect("core 0 keys");
        let consumer = SelfConsumer::of(key(DfirUnit::Lxlu), &[], &[key(DfirUnit::Lxlu)])
            .expect("the consumer is this unit");
        // ⛔ SOMEONE ELSE'S CONSUMER IS THE REFERENCE'S DT_CHECK.
        assert_eq!(
            SelfConsumer::of(key(DfirUnit::Lxsu), &[], &[key(DfirUnit::Lxlu)]),
            None
        );
        let load = ExtractScalarLoad {
            mutable_addr: reg(SenRegType::Lar, 0),
            immutable_addr: reg(SenRegType::Lbr, 0),
            addr_result: reg(SenRegType::Lar, 1),
            immutable: None,
            element_size: Bits(8),
            total_elements: Elements(128),
            update: false,
            dbg_name: None,
        };
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered =
            lower_load_and_extract_scalar_operation::<Dd2>(&load, consumer, OpSite(0), &mut to);
        let instrs = emitted(&blocks);
        assert_eq!(
            instrs.len(),
            1,
            "the address register is read and never copied"
        );
        assert_eq!(instrs[0].opcode, OpCode::LDST);
        assert_eq!(
            instrs[0].common_field(OperandField::Consumertag),
            Some(&Operand::every(OperandValue::Descriptive(
                "self".to_owned()
            )))
        );
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(lowered, Lowered::default());
    }

    #[test]
    fn a_received_scalar_becomes_one_lrf_copy_from_its_producer() {
        let (labels, mut blocks, mut regs) = site();
        let copy = LrfCopy {
            result: reg(SenRegType::Lrf, 0),
            producer: DfirUnit::Sfp,
            position: 0,
            dbg_name: None,
        };
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered = lower_receive_and_extract_scalar_operation(&copy, OpSite(0), &mut to);
        let instrs = emitted(&blocks);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].opcode, OpCode::LRFCOPY);
        assert_eq!(
            instrs[0].common_field(OperandField::Producertag),
            Some(&Operand::every(OperandValue::Descriptive("sfp".to_owned())))
        );
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(lowered, Lowered::default());
    }

    #[test]
    fn a_load_compute_copies_its_lrf_address_into_the_register_its_result_names() {
        let (labels, mut blocks, mut regs) = site();
        let mut load = LoadCompute {
            mutable_addr: reg(SenRegType::Lrf, 0),
            immutable_addr: Reg {
                locale: SenRegType::Imm,
                index: None,
            },
            result: reg(SenRegType::Lrf, 1),
            immutable_value: 0,
            element_index: Bounded::at::<0>(),
            scale_index: Bounded::at::<0>(),
            shuffle: LoadComputeShuffle::NoShuffle,
            consumer: LoadComputeConsumer::Unit(LoadConsumer::CrossPtnLink),
            update: false,
            dbg_name: None,
        };
        let mut to = LowerTo {
            labels: &labels,
            blocks: &mut blocks,
            regs_to_init: &mut regs,
            units: &[],
            full_reg_init: false,
        };
        let lowered = lower_load_compute_and_send_operation::<Dd2>(
            Component::Lxlu,
            &mut load,
            Bits(8),
            OpSite(0),
            &mut to,
        );
        let instrs = emitted(&blocks);
        assert_eq!(instrs[0].opcode, OpCode::LRFREGCOPY);
        assert_eq!(instrs[0].tag.as_deref(), Some("L0"));
        assert_eq!(instrs[1].opcode, OpCode::LDCVTI);
        assert_eq!(instrs[1].tag, None);
        assert_eq!(load.mutable_addr, reg(SenRegType::Lrf, 0));
        assert_eq!(lowered.copy_ops, CopyOps(1));
        assert!(lowered.refused.is_empty());
    }

    /// e122 — ⭐ IBM'S OWN HEADER COPY: `if_else_label.mlir` copies 131072 into `lbr0` with
    /// `element_size = 16` on the L3 store unit, and its `LBR : 0 : 2048` is 131072 × 16 / 8 / 128
    /// with no instruction emitted at all.
    #[test]
    fn a_program_headers_constant_scales_into_a_register_initialiser_and_emits_nothing() {
        let l3su = UnitKey {
            unit: DfirUnit::L3su,
            core: Core::checked(1).expect("every arch has core 1"),
            corelet: Corelet::checked(0),
        };
        let units = [(l3su, None)];
        fn header<'a>(
            source: &'a RegImmSource,
            locale: SenRegType,
            units: &'a [(UnitKey, Option<FoldId>)],
        ) -> ScalarCopy<'a> {
            ScalarCopy {
                input: CopyInput::Imm(source),
                assign: Assign {
                    src_index: None,
                    tgt_index: at(0),
                    kind: AssignKind::LrfFromLrf,
                },
                reg: reg(locale, 0),
                element_size: Some(Bits(16)),
                program_header: true,
                units,
            }
        }
        let mut labels = Labels::default();
        let mut blocks = UniformInstrBlocks::default();
        let mut reg_graph = RegGraphs::default();
        let mut nop_for_labels = Vec::new();
        let lbr = RegImmSource::Common(RegImm::Constant(131_072));
        let copied = lower_copy_operation::<Dd2>(
            Component::L3su,
            &header(&lbr, SenRegType::Lbr, &units),
            None,
            CodeGraph {
                labels: &mut labels,
                region: &mut blocks,
                at: OpSite(0),
            },
            &mut reg_graph,
            &mut nop_for_labels,
        );
        assert_eq!(copied.lowered, Lowered::default());
        assert_eq!(copied.label, Some(LabelPlacement::Unlabelled));
        assert!(
            blocks.blocks.is_empty(),
            "a register initialiser is not an instruction"
        );
        assert_eq!(
            reg_graph.get(l3su),
            Some(&vec![RegInit {
                file: RegType::Lbr,
                index: RegIndex::ALL[0],
                value: Operand::every(OperandValue::Int(2048)),
            }])
        );
        // ⛔ AND A SEN1P5 EBR IS HALVED FIRST, but only where it is an even number of sticks:
        // 128 B × 8 / 16 b is 64 elements a stick, so 256 divides by two of them and 64 does not.
        let mut halved = RegGraphs::default();
        let aligned = RegImmSource::Common(RegImm::Constant(256));
        let copied = lower_copy_operation::<Sen1p5>(
            Component::L3su,
            &header(&aligned, SenRegType::Ebr, &units),
            None,
            CodeGraph {
                labels: &mut labels,
                region: &mut blocks,
                at: OpSite(0),
            },
            &mut halved,
            &mut nop_for_labels,
        );
        assert_eq!(copied.lowered, Lowered::default());
        assert_eq!(
            halved.get(l3su),
            Some(&vec![RegInit {
                file: RegType::Ebr,
                index: RegIndex::ALL[0],
                value: Operand::every(OperandValue::Int(2)),
            }]),
            "128 × 16 / 8 / 128"
        );
        let odd = RegImmSource::Common(RegImm::Constant(64));
        let copied = lower_copy_operation::<Sen1p5>(
            Component::L3su,
            &header(&odd, SenRegType::Ebr, &units),
            None,
            CodeGraph {
                labels: &mut labels,
                region: &mut blocks,
                at: OpSite(0),
            },
            &mut halved,
            &mut nop_for_labels,
        );
        assert_eq!(
            copied.lowered.refused,
            vec![TransferRefusal::EbrNotEvenSticks(64)]
        );
        // ⛔ AND THE DIVISOR TRUNCATES BEFORE IT DOUBLES: 1024 b a stick over a 96 b element is 10
        // elements, so 60 clears two sticks' 20 where a `2 * 1024 / 96` of 21 would have refused it.
        let mut refused = Vec::new();
        assert_eq!(
            halve_for_sen1p5_ebr::<Sen1p5>(
                Some(&RegImmSource::Common(RegImm::Constant(60))),
                Some(Bits(96)),
                &mut refused,
            ),
            Some(RegImmSource::Common(RegImm::Constant(30)))
        );
        assert!(refused.is_empty(), "60 is twenty elements times three");
    }
}
