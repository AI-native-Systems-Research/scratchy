// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `VectorRegisterInitialization.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e254_runOn` | 254 | 0 | 6 | `dcc/src/Transform/Sentient/VectorRegisterInitialization.cpp:59` |
//! | `e255_isCandidate` | 255 | 0 | 32 | `dcc/src/Transform/Sentient/VectorRegisterInitialization.cpp:125` |
//! | `e256_markForRegisterInit` | 256 | 0 | 6 | `dcc/src/Transform/Sentient/VectorRegisterInitialization.cpp:159` |
//! | `e397_runOnOperation` | 397 | 1 | 6 | `dcc/src/Transform/Sentient/VectorRegisterInitialization.cpp:66` |
//! | `e398_runOn` | 398 | 1 | 44 | `dcc/src/Transform/Sentient/VectorRegisterInitialization.cpp:80` |


// ⛔ THE PASS IS COMPLETE BUT STILL NOT WIRED INTO THE PIPELINE: nothing calls
// [`VectorRegisterInitialization::run_on_operation`] yet, so every item here is reachable only from
// this file's own tests. ⭐ REMOVE THIS WITH THE FIRST PIPELINE CALLER.
#![allow(dead_code)]

use std::collections::BTreeSet;

use super::op_rerolling::unroll_operands::lrf_at;
use crate::arch::Arch;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, dataflow, regions_ref, sentient, symbol, uniform,
};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::units::DfirUnit;
use crate::workload::Workload;

/// THE PORTS ALREADY READ OR WRITTEN AT ONE POINT IN PROGRAM ORDER — `PortListTy =
/// std::set<SentientComputePort>` (`dcc/src/Dialect/Sentient/Utils.hpp:27`), whose element is this
/// island's [`sentient::Port`].
pub(crate) type PortList = BTreeSet<sentient::Port>;

/// `Statistic<"vector_register_init_count", "num-vector-register-inits", "Number of vector register
/// initializations moved to the program header">` (`Transform/Sentient/Passes.td:162`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VectorRegisterInitCount(pub(crate) u32);

/// `VectorRegisterInitializationPass`'s OWN STATE (`:51-78`) — the statistic these units share.
///
/// ⭐ `opts_` IS DELIBERATELY ABSENT. Its one reader is the `opts_.OptLevel == 0` test inside the
/// `#if 0` at `:88-92` that the FIXME at `:84-87` explains, so the C++ carries the option and
/// nothing reads it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct VectorRegisterInitialization {
    /// How many splats this pass has moved into the program header.
    pub(crate) vector_register_init_count: VectorRegisterInitCount,
}

impl VectorRegisterInitialization {
    /// Replaces: e254_runOn
    ///
    /// Runs the pass on the PE and SFP program units of the module, and on no others.
    ///
    /// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)` and `runOn(dataflow::ProgramUnitOp)` (e398) are
    /// one C++ overload set and cannot both be `run_on` here.
    /// ⭐ `getUnitType(unit.getUnits()[0].getDefiningOp())` IS `Units::kind()` — the list is
    /// constructed BY kind and has a head, so its first element's type is the list's.
    pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        for unit in program.units.iter_mut() {
            // `if (is_any_of(type, PE, SFP)) runOn(unit);` — the LRF this pass initialises is the
            // vector register file, which only the two vector compute units have.
            if matches!(unit.on.kind(), DfirUnit::Pe | DfirUnit::Sfp) {
                self.run_on_unit(unit);
            }
        }
    }

    /// Replaces: e398_runOn
    ///
    /// One program unit in program order: mark each candidate splat, then add the ports the op just
    /// examined accesses, so a later splat into the same port is refused (`:80-123`).
    ///
    /// ⛔ THE INSTRUCTION ESTIMATOR IS NEVER ASKED, and that is the reference and not a gap: its only
    /// reader is the `opts_.OptLevel == 0` test inside the `#if 0` at `:88-92`.
    /// ⭐ TWO PHASES, ONE WALK — `programHeader` is read by neither [`Self::is_candidate`] nor
    /// [`collect_accessed_lrf_ports`], so marking after the scan is the reference's single pass.
    fn run_on_unit<A: Arch>(&mut self, unit: &mut ProgramUnit<A>) {
        let mut accessed = PortList::new();
        let mut to_mark: Vec<usize> = Vec::new();
        {
            let regions: [&[Op]; 1] = [&unit.body];
            let defs = Definitions::from_innermost(&regions);
            // `for (auto &block : unit.getRegion()) for (auto &op : block)` — the OUTERMOST block, which
            // the reference's own comment states is deliberate: *"for now only optimize sentient.splat
            // operations in the outermost block of the program unit"* (`:104-107`).
            for (index, op) in unit.body.iter().enumerate() {
                if self.is_candidate(op, &accessed, defs) {
                    to_mark.push(index);
                }
                collect_accessed_lrf_ports(op, defs, &mut accessed);
            }
        }
        for index in to_mark {
            self.mark_for_register_init(&mut unit.body[index]);
        }
    }

    /// Replaces: e255_isCandidate
    ///
    /// Whether one op is a `sentient.splat` of constants into a port nothing has accessed yet.
    ///
    /// ⛔ THE PORT RULE IS DELIBERATELY CONSERVATIVE (`:151-155`): the splat must be the FIRST access
    /// through its output port, so a port already in `accessed_ports` disqualifies it outright.
    /// ⛔ A `uniform.query_map` INPUT NEEDS **EVERY** ENTRY of the mapping behind it constant, not
    /// just the entry one unit reads: the marked splat becomes a register init for every unit of the
    /// list, each reading its own mapped value (`ConstructProgIRHelper.cpp:4377-4383`).
    pub(crate) fn is_candidate(
        &self,
        op: &Op,
        accessed_ports: &PortList,
        defs: Definitions<'_>,
    ) -> bool {
        // `if (!isa<sentient::SplatOp>(op)) return false;`
        let Op::Sentient(sentient::Op::Splat { input, output, .. }) = op else {
            return false;
        };
        // The three `getDefiningOp<T>()`s, then `if (!const_input && !vector_const && !query_map)
        // return false;` and the query map's own walk.
        match defs.of(*input) {
            Some(Op::Sentient(
                sentient::Op::ScalarConstant { .. } | sentient::Op::VectorConstant { .. },
            )) => {}
            Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
                if !all_mapped_values_are_constants(*map, defs) {
                    return false;
                }
            }
            _ => return false,
        }
        // `auto output_port = splat_op.getOutput().getDefiningOp<sentient::LogicalPortOp>();`
        let Some(Op::Sentient(sentient::Op::LogicalPort { port_name, .. })) = defs.of(*output)
        else {
            todo!(
                "isCandidate: DT_CHECK_MSG(output_port, \"expected output port to be specified \
                 through logical_port\") on {output:?} (:149-150)"
            )
        };
        // `if (set_of_accessed_ports.count(output_port.getPortName())) return false; return true;`
        !accessed_ports.contains(port_name)
    }

    /// Replaces: e256_markForRegisterInit
    ///
    /// Sets `programHeader = true` on a candidate `sentient.splat` and counts it — THE PASS'S WHOLE
    /// EFFECT on the IR; `ConstructProgIRHelper` then reads the attribute instead of emitting an
    /// IMMCOPY, and only when `pad` is `none` (`ConstructProgIRHelper.cpp:3834`).
    ///
    /// ⛔ THE `DT_CHECK_MSG(isa<sentient::SplatOp>(op), ..)` IS NOT UNREPRESENTABLE HERE: the argument
    /// is `Operation &op` in the reference and any [`Op`] here, so it stays a named stop.
    pub(crate) fn mark_for_register_init(&mut self, op: &mut Op) {
        let Op::Sentient(sentient::Op::Splat { program_header, .. }) = op else {
            todo!(
                "markForRegisterInit: DT_CHECK_MSG(isa<sentient::SplatOp>(op), \"expected a splat \
                 operation\") on {op:?} (:160)"
            )
        };
        *program_header = true;
        self.vector_register_init_count.0 += 1;
    }

    /// Replaces: e397_runOnOperation
    ///
    /// The pass entry: unless the flag disables it, run [`Self::run_on_program`] over the module
    /// (`:66-72`).
    ///
    /// ⛔ `markAnalysesPreserved<Liveness>()` IS PASS-MANAGER BOOKKEEPING WITH NO IR EFFECT — it tells
    /// MLIR's pass manager not to invalidate a cached `Liveness`, and there is no pass manager here. It
    /// is documented rather than fabricated, and `Liveness` is out of campaign scope anyway.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        if DISABLE_THIS_PASS {
            return;
        }
        self.run_on_program(program);
    }
}

/// `-dcc-vector-register-init-disable`, `cl::init(false)` (`:47-49`) — a `dcc-opt` command-line flag,
/// not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `dyn_cast<mlir::uniform::DefImmutableMappingOp>(query_map.getMap().getDefiningOp())` and the
/// `isa<sentient::ConstantOp, sentient::VectorConstantOp>` walk over its values (`:136-145`).
///
/// ⛔ AN ENTRY WITH NO DEFINING OP ANSWERS `false`, WHICH IS THE REFERENCE'S INTENT AND NOT ITS
/// BEHAVIOUR: `isa<>` on the null `Operation *` of a region argument aborts there. An empty mapping is
/// vacuously constant, exactly as the reference's loop is.
fn all_mapped_values_are_constants(map: Val, defs: Definitions<'_>) -> bool {
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(map) else {
        todo!(
            "isCandidate: a uniform.query_map's $map is not a uniform.def_immutable_mapping, which \
             `dyn_cast` + `target_map.getValues()` dereferences unchecked (:137-139)"
        )
    };
    pairs.iter().all(|(_key, value)| {
        matches!(
            defs.of(*value),
            Some(Op::Sentient(
                sentient::Op::ScalarConstant { .. } | sentient::Op::VectorConstant { .. }
            ))
        )
    })
}

/// `collectAccessedPorts(op, "lrf", output)` (`dcc/src/Dialect/Sentient/Utils.cpp:55-169`) — every LRF
/// port one op reads or writes, added to `accessed` in program order.
///
/// ⛔ PORTED INLINE BECAUSE IT DECIDES THIS PASS'S WHOLE EFFECT: without it `accessed_ports` stays
/// empty and [`VectorRegisterInitialization::is_candidate`] marks EVERY splat. `Dialect/Sentient/Utils.cpp`
/// helpers are already ported inline beside their consumer (`doesMacOpUseRegister`, `localeToRegisterClass`).
/// ⛔ NO `port_kind` PARAMETER: [`sentient::Port::Lrf`] is the enum's only spelling containing `"lrf"`,
/// so the reference's `.contains(port_kind)` string test is [`insert_if_lrf`] — and this crate forbids a
/// string standing for a closed set.
fn collect_accessed_lrf_ports(op: &Op, defs: Definitions<'_>, accessed: &mut PortList) {
    let regions = regions_ref(op);
    if !regions.is_empty() {
        // `DT_CHECK((isa<ForOp, IfOp, UniformizeRegionsOp, EqualizePatternOp>(op)) && "unknown op with
        // regions")` (`:58-61`), the recursion, then `if (!op.getRegions().empty()) return;` (`:68`).
        if !matches!(
            op,
            Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
                | Op::Uniform(
                    uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. }
                )
                | Op::UniformRegions(_)
        ) {
            panic!(
                "DT_CHECK(isa<ForOp, IfOp, UniformizeRegionsOp, EqualizePatternOp>(op)) — \"unknown \
                 op with regions\" on {op:?} (`Dialect/Sentient/Utils.cpp:58-61`)"
            )
        }
        for region in regions {
            for inner in region {
                collect_accessed_lrf_ports(inner, defs, accessed);
            }
        }
        return;
    }
    match op {
        // `MacOp` (`:70-90`) and `TernaryOp` (`:105-122`) — the three operands, then the concat of all
        // four forwarding lists. ⭐ A SET, so inserting each operand's own forwarding beside its port is
        // the same membership, and the two arms are character-for-character the same read.
        Op::Sentient(sentient::Op::VectorMac {
            op_a,
            op_b,
            op_c,
            result,
            ..
        })
        | Op::Sentient(sentient::Op::VectorTernary {
            op_a,
            op_b,
            op_c,
            result,
            ..
        }) => {
            for operand in [op_a, op_b, op_c] {
                insert_operand_ports(accessed, operand);
            }
            insert_forwarded(accessed, &result.forwarding);
        }
        // `BinaryOp` (`:91-105`).
        Op::Sentient(sentient::Op::VectorBinary {
            op_a, op_b, result, ..
        }) => {
            for operand in [op_a, op_b] {
                insert_operand_ports(accessed, operand);
            }
            insert_forwarded(accessed, &result.forwarding);
        }
        // `UnaryOp` (`:123-135`).
        Op::Sentient(sentient::Op::VectorUnary { op_a, result, .. }) => {
            insert_operand_ports(accessed, op_a);
            insert_forwarded(accessed, &result.forwarding);
        }
        // `SplatOp` (`:136-148`) — ⛔ the INPUT port only if the input comes from a `logical_port`, the
        // OUTPUT port under a `DT_CHECK_MSG` that it does.
        Op::Sentient(sentient::Op::Splat { input, output, .. }) => {
            if let Some(Op::Sentient(sentient::Op::LogicalPort { port_name, .. })) = defs.of(*input)
            {
                insert_if_lrf(accessed, *port_name);
            }
            let Some(Op::Sentient(sentient::Op::LogicalPort { port_name, .. })) = defs.of(*output)
            else {
                panic!(
                    "DT_CHECK_MSG(output_port, \"expected output ssa value to come from \
                     logical_port\") on {output:?} (`Dialect/Sentient/Utils.cpp:143-144`)"
                )
            };
            insert_if_lrf(accessed, *port_name);
        }
        // `OpaqueOp` (`:149-164`) — `port_str = "lrf" + port_str.erase(0, 1)` over the printed `R<n>`,
        // and ⛔ the insert is UNCONDITIONAL: this arm has no `contains(port_kind)` test at all.
        Op::Sentient(sentient::Op::Opaque {
            read_write,
            read_only,
            ..
        }) => {
            for (_name, addr) in read_write.iter().chain(read_only) {
                let Some(index) = lrf_at(addr.0) else {
                    panic!(
                        "DT_CHECK_MSG(compute_port.has_value(), \"Invalid port\") for R{} \
                         (`Dialect/Sentient/Utils.cpp:161`)",
                        addr.0
                    )
                };
                accessed.insert(sentient::Port::Lrf(index));
            }
        }
        // `else if (mayAccessPort(op)) llvm_unreachable("unknown operation");` (`:165-168`).
        other => {
            if may_access_port(other) {
                panic!(
                    "llvm_unreachable(\"unknown operation\") on {other:?} \
                     (`Dialect/Sentient/Utils.cpp:166-167`)"
                )
            }
        }
    }
}

/// One operand's own port and every port it forwards to.
fn insert_operand_ports(accessed: &mut PortList, operand: &sentient::Operand) {
    insert_if_lrf(accessed, operand.port);
    insert_forwarded(accessed, &operand.forwarding);
}

/// One `*Forwarding` array.
fn insert_forwarded(accessed: &mut PortList, forwarding: &[sentient::Port]) {
    for &port in forwarding {
        insert_if_lrf(accessed, port);
    }
}

/// `if (stringifySentientComputePort(port).contains("lrf")) output.insert(port);`
fn insert_if_lrf(accessed: &mut PortList, port: sentient::Port) {
    if matches!(port, sentient::Port::Lrf(_)) {
        accessed.insert(port);
    }
}

/// `mayAccessPort(op)` (`dcc/src/Dialect/Sentient/Utils.cpp:36-53`) — the negated `isa<>` list of the
/// ops that provably touch no compute port.
///
/// ⛔ `mlir::dataflow::ReturnOp` HAS NO ISLAND VARIANT, so it is absent from the list rather than
/// missing from it.
/// ⛔ `sentient.load` AND `sentient.vector_constant` ARE IN NEITHER the arms above NOR this list, so
/// they reach the reference's own `llvm_unreachable` — that is its behaviour, not an omission here.
fn may_access_port(op: &Op) -> bool {
    !matches!(
        op,
        Op::Dataflow(
            dataflow::Op::GetUnit { .. }
                | dataflow::Op::GetLocalUnit { .. }
                | dataflow::Op::CreateMulticastGroup { .. }
                | dataflow::Op::CreateGroup { .. }
        ) | Op::Sentient(
            sentient::Op::Yield { .. }
                | sentient::Op::LoadAndSend { .. }
                | sentient::Op::ReceiveAndStore { .. }
                | sentient::Op::LoadAndStore { .. }
                | sentient::Op::LoadComputeAndSend { .. }
                | sentient::Op::LoadAndExtractScalar { .. }
                | sentient::Op::ReceiveAndExtractScalar { .. }
                | sentient::Op::ScalarAdd { .. }
                | sentient::Op::ScalarSub { .. }
                | sentient::Op::ScalarMul { .. }
                | sentient::Op::ScalarCopy { .. }
                | sentient::Op::ScalarConstant { .. }
                | sentient::Op::Sync { .. }
                | sentient::Op::Nop { .. }
                | sentient::Op::SetSendDst { .. }
                | sentient::Op::LogicalPort { .. }
                | sentient::Op::Samv { .. }
                | sentient::Op::SetMask { .. }
                | sentient::Op::IncrMask { .. }
        ) | Op::Uniform(
            uniform::Op::DefImmutableMapping { .. }
                | uniform::Op::Yield { .. }
                | uniform::Op::QueryMap { .. }
        ) | Op::Symbol(
            symbol::Op::CreateSymbol { .. }
                | symbol::Op::ImmutableMapping { .. }
                | symbol::Op::QueryMap { .. }
        )
    )
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::sentient::{
        LrfIndex, Port, Precision, RegType, SplatPad, UnrollFactor,
    };

    /// A model, so a program is typed; nothing here reads it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// `%r = sentient.scalar_constant {value = N : si64} : index`.
    fn scalar_const(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%p = sentient.logical_port {portName = port}`.
    fn logical_port(result: u32, port_name: Port) -> Op {
        Op::Sentient(sentient::Op::LogicalPort {
            port_name,
            result: Val(result),
        })
    }

    /// `sentient.splat %in, %out, %mask {..}`.
    fn splat(input: u32, output: u32) -> Op {
        Op::Sentient(sentient::Op::Splat {
            input: Val(input),
            output: Val(output),
            mask: Val(99),
            pad: SplatPad::None,
            precision: Precision::Fp16,
            program_header: false,
            unroll_factor: UnrollFactor::X1,
            unroll_incr_result: false,
            dbg_name: None,
        })
    }

    /// One program with one unit on `kind`, holding `body`.
    fn program_on(kind: DfirUnit, body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(kind, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e254 — a unit that is neither PE nor SFP is not visited at all, which is the whole content of
    /// this overload: reaching one would panic on the unported e398.
    #[test]
    fn run_on_program_skips_a_unit_that_is_not_pe_or_sfp() {
        let mut program = program_on(DfirUnit::L3lu, vec![scalar_const(1, 7)]);
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_program(&mut program);
        assert_eq!(pass, VectorRegisterInitialization::default());
    }

    /// e254 — and a PE unit IS visited: its candidate splat comes back marked.
    #[test]
    fn run_on_program_delegates_a_pe_unit_to_e398() {
        let mut program = program_on(
            DfirUnit::Pe,
            vec![
                scalar_const(1, 7),
                logical_port(2, Port::Lrf(LrfIndex::L0)),
                splat(1, 2),
            ],
        );
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_program(&mut program);
        assert_eq!(
            (pass.vector_register_init_count, marked(&program)),
            (VectorRegisterInitCount(1), vec![true])
        );
    }

    /// Which splats of the one unit came back with `programHeader` set, in program order.
    fn marked(program: &Program<Dd2, AnyModel, AnyRung>) -> Vec<bool> {
        program
            .units
            .iter()
            .flat_map(|unit| unit.body.iter())
            .filter_map(|op| match op {
                Op::Sentient(sentient::Op::Splat { program_header, .. }) => Some(*program_header),
                _ => None,
            })
            .collect()
    }

    /// e398 — THE CONSERVATIVE RULE IS THE WALK'S ORDER: the first splat into `lrf0` is marked, and the
    /// second is refused because collecting the first one's OUTPUT port put `lrf0` in the accessed set.
    #[test]
    fn e398_run_on_marks_only_the_first_splat_into_a_port() {
        let mut program = program_on(
            DfirUnit::Pe,
            vec![
                scalar_const(1, 7),
                logical_port(2, Port::Lrf(LrfIndex::L0)),
                splat(1, 2),
                logical_port(3, Port::Lrf(LrfIndex::L0)),
                splat(1, 3),
                logical_port(4, Port::Lrf(LrfIndex::L1)),
                splat(1, 4),
            ],
        );
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_program(&mut program);
        assert_eq!(
            (pass.vector_register_init_count, marked(&program)),
            (VectorRegisterInitCount(2), vec![true, false, true])
        );
    }

    /// e397 — the entry runs the pass, the flag being off; `markAnalysesPreserved` has no IR effect.
    #[test]
    fn e397_run_on_operation_runs_the_pass() {
        let mut program = program_on(
            DfirUnit::Sfp,
            vec![
                scalar_const(1, 7),
                logical_port(2, Port::Lrf(LrfIndex::L0)),
                splat(1, 2),
            ],
        );
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_operation(&mut program);
        assert_eq!(
            (pass.vector_register_init_count, marked(&program)),
            (VectorRegisterInitCount(1), vec![true])
        );
    }

    /// e255 — the vendor's own shape: a splat of a `sentient.scalar_constant` into a port nothing has
    /// touched is a candidate.
    #[test]
    fn is_candidate_accepts_a_constant_splat_into_an_untouched_port() {
        let body = vec![
            scalar_const(1, 7),
            logical_port(2, Port::Lrf(LrfIndex::L0)),
            splat(1, 2),
        ];
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let pass = VectorRegisterInitialization::default();
        assert!(pass.is_candidate(&body[2], &PortList::new(), defs));
    }

    /// e255 — and the conservative rule: the same splat is refused once its output port has been
    /// accessed by an earlier op.
    #[test]
    fn is_candidate_refuses_a_splat_into_an_already_accessed_port() {
        let body = vec![
            scalar_const(1, 7),
            logical_port(2, Port::Lrf(LrfIndex::L0)),
            splat(1, 2),
        ];
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let pass = VectorRegisterInitialization::default();
        let accessed = PortList::from([Port::Lrf(LrfIndex::L0)]);
        assert!(!pass.is_candidate(&body[2], &accessed, defs));
    }

    /// e256 — the effect: `programHeader` goes true on the splat and the statistic counts it.
    #[test]
    fn mark_for_register_init_sets_the_program_header_and_counts() {
        let mut op = splat(1, 2);
        let mut pass = VectorRegisterInitialization::default();
        pass.mark_for_register_init(&mut op);
        assert_eq!(
            (op, pass.vector_register_init_count),
            (
                Op::Sentient(sentient::Op::Splat {
                    input: Val(1),
                    output: Val(2),
                    mask: Val(99),
                    pad: SplatPad::None,
                    precision: Precision::Fp16,
                    program_header: true,
                    unroll_factor: UnrollFactor::X1,
                    unroll_incr_result: false,
                    dbg_name: None,
                }),
                VectorRegisterInitCount(1)
            )
        );
    }
}
