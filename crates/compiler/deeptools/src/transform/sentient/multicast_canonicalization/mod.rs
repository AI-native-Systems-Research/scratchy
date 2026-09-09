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

//! `MulticastCanonicalization.cpp` — 8 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e097_runOn` | 097 | 0 | 6 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:83` |
//! | `e098_processDirectMulticast` | 098 | 0 | 14 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:121` |
//! | `e099_createNewProducerQMap` | 099 | 0 | 26 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:299` |
//! | `e323_runOnOperation` | 323 | 1 | 5 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:90` |
//! | `e324_processQMapOfDirectMulticast` | 324 | 1 | 16 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:138` |
//! | `e516_processConditional` | 516 | 3 | 129 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:168` |
//! | `e569_runOn` | 569 | 4 | 43 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:327` |
//! | `e605_runOn` | 605 | 5 | 24 | `dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:371` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{dataflow, uniform};
use crate::islands::dataflow_ir::link::RecvEnd;
use crate::islands::sentient::dialects::sentient::StoreSource;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

/// Replaces: e097_runOn
///
/// Runs the pass over every program unit of one module.
///
/// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)`, `runOn(ProgramUnitOp)` (e605) and
/// `runOn(ReceiveAndStoreOp)` (e569) are one C++ overload set and cannot all be `run_on` here.
///
/// ⛔ TRAP: THE WALK IS COLLECTED BEFORE IT IS ITERATED (`:84-87`) because `runOn` rewrites the unit
/// it is handed. A program's units are a fixed list here, so the snapshot is the droppable mechanism.
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    for unit in program.units.iter_mut() {
        run_on_unit(unit);
    }
}

/// `runOn(dataflow::ProgramUnitOp)` — entry 605, level 5, not yet ported.
fn run_on_unit<A: Arch>(unit: &mut ProgramUnit<A>) -> ! {
    let _ = unit;
    todo!(
        "e605_runOn(dataflow::ProgramUnitOp) — the L3LU/L3SU gate and the unique-producer store walk          (MulticastCanonicalization.cpp:371), which schedules e569_runOn per store"
    )
}

/// ONE `dataflow.create_multicast_group` READ AS A STORE'S `$producer` — the group handle and the
/// unit that produces the data (`Dataflow.td:174-181`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectMulticast {
    /// What the op binds, which is what the store currently names.
    group: Val,
    /// The group's own `$producer`.
    producer: Val,
}

impl DirectMulticast {
    /// `dyn_cast<dataflow::CreateMulticastGroupOp>(ras.getProducer().getDefiningOp())` (`:333-334`) —
    /// `None` where the producer is anything else, which is the `else if` chain moving on.
    #[must_use]
    pub fn of(group: Val, scope: &[Op]) -> Option<DirectMulticast> {
        match dialects::defining_op(group, scope) {
            Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { producer, .. })) => {
                Some(DirectMulticast {
                    group,
                    producer: *producer,
                })
            }
            _ => None,
        }
    }
}

/// Replaces: e098_processDirectMulticast
///
/// Moves the group off every store's `$producer` into its `$multicast_info` and puts the group's own
/// `$producer` — the unit the data comes from — in its place.
///
/// ⛔ TRAP: THE `ras` PARAMETER IS UNUSED (`:121-135`). The function re-collects every store reading
/// the group and rewrites all of them, so the store it was handed is rewritten only because it is one
/// of those; passing a store that does not read the group changes nothing.
///
/// ⛔ TRAP: THE GROUP OP ITSELF IS NOT ERASED — `$multicast_info` keeps reading it.
///
/// ⛔ DELIBERATE DIVERGENCE: THE FILTER IS THE `$producer` POSITION, NOT ANY USE. The reference takes
/// every `multicast->getUses()` owned by a store (`:126-130`), which also catches one already holding
/// the group in `$multicast_info` and overwrites ITS unrelated `$producer`; its own comment says what
/// it meant — *"List of ReceiveAndStoreOps with multicast as the producer"* (`:124`).
pub fn process_direct_multicast(multicast: DirectMulticast, body: &mut [Op]) {
    for op in body.iter_mut() {
        if let Op::Sentient(sentient::Op::ReceiveAndStore {
            producer,
            multicast_info,
            ..
        }) = op
            && *producer == StoreSource::Multicast(multicast.group)
        {
            *producer = StoreSource::Wire(RecvEnd::from_multicast_group(multicast.producer));
            *multicast_info = Some(multicast.group);
        }
        for region in dialects::regions_mut(op) {
            process_direct_multicast(multicast, region);
        }
    }
}

/// THE TWO OPS A `query_map` OF MULTICAST GROUPS BECOMES, in the reference's creation order — both
/// built at the original `uniform.def_immutable_mapping`'s position (`OpBuilder tmp_builder(immutable_map)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerQMap {
    /// `uniform.def_immutable_mapping` over the same keys, values replaced by the producers.
    pub map: Op,
    /// `uniform.query_map` reading it with the original's `$key`.
    pub qmap: Op,
    /// What the new `query_map` binds — the `$producer` a store then reads.
    pub result: Val,
}

/// Replaces: e099_createNewProducerQMap
///
/// Rebuilds a `uniform.query_map` whose every value is a `dataflow.create_multicast_group` into one
/// whose values are those groups' `$producer`s — `None` where any value is not a group, which is the
/// reference's `return nullptr`, *"not a type of q_map we know how to process"*.
///
/// ⭐ `keys` IS `getListOfKeyOpsFromUniformMapping(orig_qmap)` (`:301`) — the caller resolves it,
/// because that helper reads the key's binding region's unit list (`dcc/src/Dialect/Uniform/Utils.cpp:151`).
///
/// ⛔ DELIBERATE DIVERGENCE WHERE A KEY IS ABSENT FROM THE MAPPING: `getNonNullValuesFromKeys` pushes
/// only the keys its lookup holds
/// (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Uniform/Uniform.cpp:548-556`),
/// so a short `new_values` is then paired POSITIONALLY with the full `keys` (`:318-320`) and every
/// later key takes an earlier key's producer. Re-zipping each key as it is looked up keeps the pairs
/// aligned; on a mapping whose keys all resolve — the only shape the callers build — both agree.
#[must_use]
pub fn create_new_producer_qmap(
    orig_qmap: &uniform::Op,
    keys: &[Val],
    scope: &[Op],
    values: &mut Values,
) -> Option<ProducerQMap> {
    let uniform::Op::QueryMap { map, key, .. } = orig_qmap else {
        return None;
    };
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) =
        dialects::defining_op(*map, scope)
    else {
        return None;
    };
    let mut new_pairs: Vec<(Val, Val)> = Vec::new();
    for &wanted in keys {
        let Some((_, value)) = pairs.iter().find(|(each, _)| *each == wanted) else {
            continue;
        };
        match dialects::defining_op(*value, scope) {
            Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { producer, .. })) => {
                new_pairs.push((wanted, *producer));
            }
            _ => return None,
        }
    }
    let new_map = values.mint();
    let result = values.mint();
    Some(ProducerQMap {
        map: Op::Uniform(uniform::Op::DefImmutableMapping {
            result: new_map,
            pairs: new_pairs,
        }),
        qmap: Op::Uniform(uniform::Op::QueryMap {
            result,
            map: new_map,
            key: *key,
        }),
        result,
    })
}

#[cfg(test)]
mod unit_tests {
    use super::{
        DirectMulticast, ProducerQMap, create_new_producer_qmap, process_direct_multicast,
        process_qmap_of_direct_multicast, run_on_operation,
    };
    use crate::arch::{Dd2, Elements};
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::dataflow::{
        ConsumerCount, MulticastGroupId, OutstandingRequests,
    };
    use crate::islands::dataflow_ir::dialects::{dataflow, uniform};
    use crate::islands::dataflow_ir::link::RecvEnd;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units, Values};
    use crate::islands::sentient::dialects::sentient::StoreSource;
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

    /// A model, so the program is typed; nothing here reads it.
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

    /// `%g = dataflow.create_multicast_group(%p -> ())`.
    fn group(result: Val, producer: Val) -> Op {
        Op::Dataflow(dataflow::Op::CreateMulticastGroup {
            result,
            producer,
            consumers: Vec::new(),
            num_consumers: ConsumerCount(1),
            group_id: MulticastGroupId(3),
            count: OutstandingRequests(0),
            init_packet_opt_en: false,
        })
    }

    /// A `sentient.receive_and_store` whose `$producer` names `source`.
    fn store(producer: StoreSource, result: Val) -> Op {
        Op::Sentient(sentient::Op::ReceiveAndStore {
            mutable_addr: Val(90),
            immutable_addr: Val(91),
            increment: Val(92),
            producer,
            result,
            dst: None,
            drop_first: None,
            multicast_info: None,
            extent: sentient::Extent::of(Elements(64), Bits(16)),
            interleaved_group: Elements(0),
            coalesce: false,
            subword_length: 0,
            stride: 0,
            permute: false,
            shuffle_mode: None,
            reg: sentient::Reg {
                locale: sentient::RegType::Imm,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// Both stores reading the group are rewritten, not just one, and the store reading something
    /// else is left alone.
    #[test]
    fn process_direct_multicast_rewrites_every_store_on_the_group() {
        let (unit, handle, other) = (Val(0), Val(1), Val(2));
        let mut body = vec![
            group(handle, unit),
            store(StoreSource::Multicast(handle), Val(10)),
            store(StoreSource::Constant(other), Val(11)),
            Op::Sentient(sentient::Op::For {
                iv: Val(20),
                bound: Val(21),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![store(StoreSource::Multicast(handle), Val(12))],
            }),
        ];
        let multicast = DirectMulticast::of(handle, &body).expect("the group defines the handle");
        process_direct_multicast(multicast, &mut body);
        let rewritten = StoreSource::Wire(RecvEnd::from_multicast_group(unit));
        assert!(matches!(
            body[1],
            Op::Sentient(sentient::Op::ReceiveAndStore { producer, multicast_info, .. })
                if producer == rewritten && multicast_info == Some(handle)
        ));
        assert!(matches!(
            body[2],
            Op::Sentient(sentient::Op::ReceiveAndStore { producer, multicast_info, .. })
                if producer == StoreSource::Constant(other) && multicast_info.is_none()
        ));
        let Op::Sentient(sentient::Op::For { body: inner, .. }) = &body[3] else {
            panic!("the loop survives")
        };
        assert!(matches!(
            inner[0],
            Op::Sentient(sentient::Op::ReceiveAndStore { producer, multicast_info, .. })
                if producer == rewritten && multicast_info == Some(handle)
        ));
    }

    /// A mapping of two groups becomes a mapping of their producers under the same keys; one value
    /// that is not a group refuses the whole map.
    #[test]
    fn create_new_producer_qmap_replaces_values_with_producers() {
        let (k0, k1) = (Val(0), Val(1));
        let (g0, g1) = (Val(2), Val(3));
        let (p0, p1) = (Val(4), Val(5));
        let (mapping, key) = (Val(6), Val(7));
        let orig = uniform::Op::QueryMap {
            result: Val(8),
            map: mapping,
            key,
        };
        let mut scope = vec![
            group(g0, p0),
            group(g1, p1),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: mapping,
                pairs: vec![(k0, g0), (k1, g1)],
            }),
        ];
        let mut values = Values::default();
        for _ in 0..9 {
            let _ = values.mint();
        }
        let built = create_new_producer_qmap(&orig, &[k0, k1], &scope, &mut values)
            .expect("both values are multicast groups");
        assert_eq!(
            built,
            ProducerQMap {
                map: Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(9),
                    pairs: vec![(k0, p0), (k1, p1)],
                }),
                qmap: Op::Uniform(uniform::Op::QueryMap {
                    result: Val(10),
                    map: Val(9),
                    key,
                }),
                result: Val(10),
            }
        );
        // `else { return nullptr; }` — a value that is not a `create_multicast_group`.
        scope[1] = Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result: g1,
            reg_locale: sentient::RegType::Imm,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            is_symbol: false,
        });
        assert_eq!(
            create_new_producer_qmap(&orig, &[k0, k1], &scope, &mut values),
            None
        );
    }

    /// e323 — the flag is off, so the entry walks the module's units and stops where e605 is not
    /// ported. ⭐ REACHING THE SEAM IS WHAT IS TESTABLE: it proves the entry runs the pass at all.
    #[test]
    #[should_panic(expected = "e605_runOn")]
    fn e323_runs_the_pass_over_the_module() {
        let mut program = program_on(DfirUnit::L3lu, Vec::new());
        run_on_operation(&mut program);
    }

    /// e324 — a `query_map` of two groups gains a second `query_map` of their producers, both built at
    /// the ORIGINAL mapping's position, and the store on the old map reads the new one with the old in
    /// `$multicast_info`.
    #[test]
    fn e324_moves_the_store_onto_a_query_map_of_the_producers() {
        let (k0, k1) = (Val(0), Val(1));
        let (g0, g1) = (Val(2), Val(3));
        let (p0, p1) = (Val(4), Val(5));
        let (mapping, key, qmap) = (Val(6), Val(7), Val(8));
        let mut body = vec![
            group(g0, p0),
            group(g1, p1),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: mapping,
                pairs: vec![(k0, g0), (k1, g1)],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: qmap,
                map: mapping,
                key,
            }),
            store(StoreSource::QueryMap(qmap), Val(9)),
        ];
        let mut values = Values::default();
        for _ in 0..10 {
            let _ = values.mint();
        }
        process_qmap_of_direct_multicast(qmap, &[k0, k1], &mut body, &mut values);
        assert_eq!(body.len(), 7);
        assert_eq!(
            body[2],
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(10),
                pairs: vec![(k0, p0), (k1, p1)],
            })
        );
        assert_eq!(
            body[3],
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(11),
                map: Val(10),
                key,
            })
        );
        assert!(matches!(
            body[6],
            Op::Sentient(sentient::Op::ReceiveAndStore { producer, multicast_info, .. })
                if producer == StoreSource::QueryMap(Val(11)) && multicast_info == Some(qmap)
        ));
    }
}

/// `-dcc-multicast-canonicalization-disable`, `cl::init(false)` (`:72-75`) — a `dcc-opt` command-line
/// flag, not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// Replaces: e323_runOnOperation
///
/// The pass entry: every program unit of the module, unless the flag turned the pass off.
///
/// ⭐ `getOperation()` IS THE PROGRAM HERE — the pass is declared on a `ModuleOp`, which this island
/// spells as the [`Program`] whose units [`run_on_program`] walks.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    if DISABLE_THIS_PASS {
        return;
    }
    run_on_program(program);
}

/// Replaces: e324_processQMapOfDirectMulticast
///
/// Builds the producers' `uniform.query_map` beside the original mapping ([`create_new_producer_qmap`])
/// and moves every store off the old map: the new map becomes `$producer`, the old one
/// `$multicast_info`.
///
/// ⭐ `keys` IS `getListOfKeyOpsFromUniformMapping(query_map)` (`:301`), which e099 already asks its
/// caller for; and the `ras` parameter is unused for the same reason as in e098 — every store on the
/// map is rewritten, so the one handed over is rewritten only because it is one of them.
/// ⛔ DIVERGENCE: `DT_CHECK_MSG(new_query_map, "unable to create a query map for the producers")`
/// (`:141`) aborts; a map this cannot rebuild leaves the IR untouched instead.
pub fn process_qmap_of_direct_multicast(
    query_map: Val,
    keys: &[Val],
    body: &mut Vec<Op>,
    values: &mut Values,
) {
    let Some(Op::Uniform(orig_qmap)) = dialects::defining_op(query_map, body) else {
        return;
    };
    let orig_qmap = orig_qmap.clone();
    let uniform::Op::QueryMap { map, .. } = &orig_qmap else {
        return;
    };
    let map = *map;
    let Some(built) = create_new_producer_qmap(&orig_qmap, keys, body, values) else {
        return;
    };
    // `OpBuilder tmp_builder(immutable_map)` — BOTH new ops go where the ORIGINAL mapping is, not
    // where the store is (`:322-330`).
    insert_before_definition(body, map, &[built.map, built.qmap]);
    assign_producer_qmap(body, query_map, built.result);
}

/// The ops immediately before whatever defines `val`, in the block that defines it — `OpBuilder`
/// constructed on an op. Nothing is inserted where no block defines it.
fn insert_before_definition(block: &mut Vec<Op>, val: Val, ops: &[Op]) -> bool {
    if let Some(at) = block
        .iter()
        .position(|op| dialects::results(op).contains(&val))
    {
        let tail = block.split_off(at);
        block.extend_from_slice(ops);
        block.extend(tail);
        return true;
    }
    for op in block.iter_mut() {
        for region in dialects::regions_mut(op) {
            if insert_before_definition(region, val, ops) {
                return true;
            }
        }
    }
    false
}

/// `getProducerMutable().assign(new_query_map.getResult())` and
/// `getMulticastInfoMutable().assign(query_map)` over every store reading `query_map` as its
/// `$producer` — the same `$producer`-position filter e098's own note argues for.
fn assign_producer_qmap(block: &mut [Op], query_map: Val, new_query_map: Val) {
    for op in block.iter_mut() {
        if let Op::Sentient(sentient::Op::ReceiveAndStore {
            producer,
            multicast_info,
            ..
        }) = op
            && *producer == StoreSource::QueryMap(query_map)
        {
            *producer = StoreSource::QueryMap(new_query_map);
            *multicast_info = Some(query_map);
        }
        for region in dialects::regions_mut(op) {
            assign_producer_qmap(region, query_map, new_query_map);
        }
    }
}

// crustify:todo: e516_processConditional
//   authority : dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:168  (129 body lines, level 3)
//   original  : void MulticastCanonicalizationPass::processConditional( sentient::ReceiveAndStoreOp ras, sentient::IfOp top_if_op)
//   calls     : e099_createNewProducerQMap, e252_size, e422_insert

// crustify:todo: e569_runOn
//   authority : dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:327  (43 body lines, level 4)
//   original  : void MulticastCanonicalizationPass::runOn(sentient::ReceiveAndStoreOp ras)
//   calls     : e098_processDirectMulticast, e324_processQMapOfDirectMulticast, e516_processConditional

// crustify:todo: e605_runOn
//   authority : dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:371  (24 body lines, level 5)
//   original  : void MulticastCanonicalizationPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e097_runOn, e569_runOn
