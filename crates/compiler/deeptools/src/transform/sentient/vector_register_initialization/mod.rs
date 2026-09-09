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


#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — [`VectorRegisterInitialization::run_on_operation`]
// (e397) is now the entry, and nothing outside this file's own tests constructs the pass to call it.
// CI runs clippy with `-D warnings`. ⭐ REMOVE THIS WHEN THE PASS IS REGISTERED IN THE PIPELINE.

use std::collections::BTreeSet;

use crate::arch::Arch;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, dataflow, defining_op, sentient, uniform,
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

    /// Replaces: e397_runOnOperation
    ///
    /// The pass entry: the whole module, unless `-dcc-vector-register-init-disable` was given
    /// (`:66-72`).
    ///
    /// ⭐ `markAnalysesPreserved<Liveness>()` IS PASS-MANAGER BOOKKEEPING WITH NO IR EFFECT — it keeps
    /// a cached `Liveness` from being invalidated, and this pass writes one attribute. ⛔ NOT an
    /// out-of-scope analysis CONSUMPTION: nothing here reads a `Liveness`, so it needs no `todo!`.
    /// ⭐ `getOperation()` IS THE ARGUMENT, as [`super::scalar_op_reordering`]'s e367 records.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        if DISABLE_THIS_PASS {
            return;
        }
        self.run_on_program(program);
    }

    /// Replaces: e398_runOn
    ///
    /// Walks the unit's outermost block in program order, marking every `sentient.splat` that is still
    /// the first access through its port and adding each op's port accesses as it passes (`:80-123`).
    ///
    /// ⛔ CANDIDATES COME FROM THE OUTERMOST BLOCK ONLY (the file's own header, `:16-18`), but
    /// [`collect_accessed_ports`] still descends: a port touched inside a `sentient.for` clobbers every
    /// later candidate outside it.
    /// ⭐ THE `InstructionEstimator` IS TAKEN AND NEVER READ — its one reader is inside the `#if 0` at
    /// `:88-92`, so this consumes NO out-of-scope analysis; the same fact as `opts_`'s absence.
    /// ⭐ `curr_unit` IS READ FOR `LLVM_DEBUG` ALONE (`:93-96`).
    fn run_on_unit<A: Arch>(&mut self, unit: &mut ProgramUnit<A>) {
        let mut accessed_ports = PortList::new();
        for index in 0..unit.body.len() {
            let candidate = {
                let regions: [&[Op]; 1] = [&unit.body];
                let defs = Definitions::from_innermost(&regions);
                self.is_candidate(&unit.body[index], &accessed_ports, defs)
            };
            if candidate {
                self.mark_for_register_init(&mut unit.body[index]);
            }
            collect_accessed_ports(
                &unit.body[index],
                &unit.body,
                PortKind::Lrf,
                &mut accessed_ports,
            );
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
}

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

/// `-dcc-vector-register-init-disable`, `cl::init(false)` (`:47-49`).
const DISABLE_THIS_PASS: bool = false;

/// WHICH REGISTER FILE `collectAccessedPorts` IS ASKED ABOUT — its `const std::string port_kind`
/// parameter, which every insert but the opaque op's is gated on.
///
/// ⛔ AN ENUM, NOT A STRING: the set is closed and this crate forbids a string for one. The whole
/// reference tree has exactly ONE caller and it passes `"lrf"` (`:120`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PortKind {
    /// `"lrf"` — the local register file this pass initialises.
    Lrf,
}

impl PortKind {
    /// `stringifySentientComputePort(port).contains(port_kind)` — ⭐ A SUBSTRING TEST, so `lrf0`
    /// through `lrf31` all match and no other port spelling does.
    fn matches(self, port: sentient::Port) -> bool {
        match self {
            PortKind::Lrf => port.spelling().contains("lrf"),
        }
    }
}

/// ⛔ NOT AN ANCHORED UNIT: `dcc/src/Dialect/Sentient/Utils.cpp` is outside this campaign's file list,
/// and e398's effect — WHICH splats are still first through their port, in program order — cannot be
/// stated without it. The `imm_size_valid`/`enclosing_region_units` in-scope-because-it-is-needed case.
///
/// `collectAccessedPorts(op, port_kind, output)` (`dcc/src/Dialect/Sentient/Utils.cpp:56-170`): every
/// port of `port_kind` this op reads or writes, at any depth.
///
/// ⛔ `mayAccessPort`'s NEGATIVE LIST (`:36-53`) IS THE SILENT ARMS OF THIS MATCH, spelled variant by
/// variant so a new island op reaches the same `llvm_unreachable("unknown operation")` (`:167-169`)
/// the reference gives it rather than being quietly skipped — the miscompile that file warns of (`:32`).
/// ⚠️ `sentient.load` AND `sentient.vector_constant` REACH THAT ABORT TODAY — neither is in the
/// negative list — even though [`VectorRegisterInitialization::is_candidate`] accepts a splat of a
/// `sentient.vector_constant`.
/// ⚠️ A LOWER-RUNG `uniform.uniformize_regions` IS THE ISLAND GAP e241 RECORDED: the reference's
/// `DT_CHECK` admits it and recurses, and its regions hold ops of the rung below.
fn collect_accessed_ports(op: &Op, unit_body: &[Op], port_kind: PortKind, output: &mut PortList) {
    // `for (auto &region : op.getRegions())` with its `DT_CHECK(isa<ForOp, IfOp,
    // UniformizeRegionsOp, EqualizePatternOp>(op))`, then `if (!op.getRegions().empty()) return;`.
    match op {
        Op::Sentient(sentient::Op::For { body, .. }) => {
            for inner in body {
                collect_accessed_ports(inner, unit_body, port_kind, output);
            }
            return;
        }
        Op::Sentient(sentient::Op::If {
            then_body,
            else_body,
            ..
        }) => {
            for inner in then_body.iter().chain(else_body) {
                collect_accessed_ports(inner, unit_body, port_kind, output);
            }
            return;
        }
        Op::UniformRegions(regions) => {
            for region in regions.regions() {
                for inner in &region.body {
                    collect_accessed_ports(inner, unit_body, port_kind, output);
                }
            }
            return;
        }
        Op::AffineFor(_) => panic!(
            "DT_CHECK(isa<ForOp, IfOp, UniformizeRegionsOp, EqualizePatternOp>(op)) && \"unknown op \
             with regions\" (`Dialect/Sentient/Utils.cpp:57-60`): {op:?}"
        ),
        Op::Uniform(
            uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. },
        ) => {
            todo!(
                "collectAccessedPorts recursing into a lower-rung uniform.uniformize_regions \
                 (`Dialect/Sentient/Utils.cpp:56-67`) — the island extension e241 already records"
            )
        }
        _ => {}
    }
    match op {
        // The four compute arms: each read port, then every forwarding port of every operand and of
        // the result, `symbolizeSentientComputePort`d back from its own spelling.
        Op::Sentient(sentient::Op::VectorMac {
            op_a,
            op_b,
            op_c,
            result,
            ..
        }) => {
            insert_operand_ports(&[op_a, op_b, op_c], result, port_kind, output);
        }
        Op::Sentient(sentient::Op::VectorBinary {
            op_a, op_b, result, ..
        }) => {
            insert_operand_ports(&[op_a, op_b], result, port_kind, output);
        }
        Op::Sentient(sentient::Op::VectorTernary {
            op_a,
            op_b,
            op_c,
            result,
            ..
        }) => {
            insert_operand_ports(&[op_a, op_b, op_c], result, port_kind, output);
        }
        Op::Sentient(sentient::Op::VectorUnary { op_a, result, .. }) => {
            insert_operand_ports(&[op_a], result, port_kind, output);
        }
        // ⭐ THE SPLAT'S INPUT PORT COUNTS ONLY WHERE THE INPUT COMES FROM A `sentient.logical_port`
        // (`:141-144`); its OUTPUT must, and a `DT_CHECK_MSG` says so.
        Op::Sentient(sentient::Op::Splat {
            input, output: out, ..
        }) => {
            if let Some(Op::Sentient(sentient::Op::LogicalPort { port_name, .. })) =
                defining_op(*input, unit_body)
                && port_kind.matches(*port_name)
            {
                output.insert(*port_name);
            }
            let Some(Op::Sentient(sentient::Op::LogicalPort { port_name, .. })) =
                defining_op(*out, unit_body)
            else {
                panic!(
                    "DT_CHECK_MSG(output_port, \"expected output ssa value to come from \
                     logical_port\") (`Dialect/Sentient/Utils.cpp:148-149`): {out:?}"
                )
            };
            if port_kind.matches(*port_name) {
                output.insert(*port_name);
            }
        }
        // ⛔ THE OPAQUE OP'S REGISTERS GO IN WITH NO `port_kind` GATE AT ALL (`:154-166`) — every
        // entry of both dictionaries is an `lrf` by construction, `"R<n>"` respelled as `lrf<n>`.
        // ⭐ `llvm_unreachable("Register dictionary should have values.")` (`:165`) IS GONE THE WAY
        // e252_size's `DT_CHECK` IS: a `RegAddr` is not a string that might not be one.
        Op::Sentient(sentient::Op::Opaque {
            read_write,
            read_only,
            ..
        }) => {
            for (_name, addr) in read_write.iter().chain(read_only) {
                let Some(index) = sentient::LrfIndex::at(addr.0) else {
                    panic!(
                        "DT_CHECK_MSG(compute_port.has_value(), \"Invalid port\") \
                         (`Dialect/Sentient/Utils.cpp:163`): lrf{}",
                        addr.0
                    )
                };
                output.insert(sentient::Port::Lrf(index));
            }
        }
        // `mayAccessPort` IS FALSE FOR THESE, so the reference reaches neither an insert nor its abort.
        Op::Sentient(
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
            | sentient::Op::IncrMask { .. },
        )
        | Op::Dataflow(
            dataflow::Op::GetUnit { .. }
            | dataflow::Op::GetLocalUnit { .. }
            | dataflow::Op::CreateGroup { .. }
            | dataflow::Op::CreateMulticastGroup { .. },
        )
        | Op::Uniform(
            uniform::Op::Yield { .. }
            | uniform::Op::DefImmutableMapping { .. }
            | uniform::Op::QueryMap { .. },
        )
        | Op::Symbol(_) => {}
        // `else if (mayAccessPort(op)) llvm_unreachable("unknown operation");`
        other => panic!(
            "llvm_unreachable(\"unknown operation\") (`Dialect/Sentient/Utils.cpp:167-169`): \
             {other:?} is neither handled nor in mayAccessPort's list"
        ),
    }
}

/// The `stringifySentientComputePort(op.getOpX()).contains(port_kind)` reads and the
/// `llvm::concat<const Attribute>(..Forwarding, getResultForwarding())` walk the four compute arms
/// repeat verbatim (`:70-134`) — ⭐ THE READ PORT IS GATED TOO, and every op's list ends with the
/// result's forwarding.
fn insert_operand_ports(
    operands: &[&sentient::Operand],
    result: &sentient::ResultPorts,
    port_kind: PortKind,
    output: &mut PortList,
) {
    for operand in operands {
        if port_kind.matches(operand.port) {
            output.insert(operand.port);
        }
    }
    let forwarded = operands
        .iter()
        .flat_map(|operand| operand.forwarding.iter())
        .chain(result.forwarding.iter());
    for port in forwarded {
        if port_kind.matches(*port) {
            output.insert(*port);
        }
    }
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
                    iter_arg: None,
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
    /// this overload: a candidate in its body is left unmarked.
    #[test]
    fn run_on_program_skips_a_unit_that_is_not_pe_or_sfp() {
        let mut program = program_on(DfirUnit::L3lu, two_splats_into_one_port());
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_program(&mut program);
        assert_eq!(pass, VectorRegisterInitialization::default());
        assert_eq!(
            program.units.iter().next().expect("the head unit").body,
            two_splats_into_one_port()
        );
    }

    /// `%1 = scalar_constant 7`, `%2 = logical_port lrf0`, then TWO splats of `%1` through `%2`.
    fn two_splats_into_one_port() -> Vec<Op> {
        vec![
            scalar_const(1, 7),
            logical_port(2, Port::Lrf(LrfIndex::L0)),
            splat(1, 2),
            splat(1, 2),
        ]
    }

    /// Whether each op of `body` carries `programHeader = true`.
    fn marked(body: &[Op]) -> Vec<bool> {
        body.iter()
            .map(|op| {
                matches!(
                    op,
                    Op::Sentient(sentient::Op::Splat {
                        program_header: true,
                        ..
                    })
                )
            })
            .collect()
    }

    /// e398 — THE ORDER IS THE EFFECT: the first splat into `lrf0` is marked, and the second is not,
    /// because collecting the first one's accessed ports has already claimed that port.
    #[test]
    fn run_on_marks_only_the_first_splat_through_a_port() {
        let mut program = program_on(DfirUnit::Pe, two_splats_into_one_port());
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_program(&mut program);

        assert_eq!(
            marked(&program.units.iter().next().expect("the head unit").body),
            vec![false, false, true, false]
        );
        assert_eq!(pass.vector_register_init_count, VectorRegisterInitCount(1));
    }

    /// e397 — the pass entry reaches the unit walk, `DisableThisPass` being off by default (`:47-49`).
    #[test]
    fn run_on_operation_is_the_entry_to_the_unit_walk() {
        let mut program = program_on(DfirUnit::Pe, two_splats_into_one_port());
        let mut pass = VectorRegisterInitialization::default();
        pass.run_on_operation(&mut program);

        assert_eq!(
            marked(&program.units.iter().next().expect("the head unit").body),
            vec![false, false, true, false]
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
