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

use std::collections::BTreeMap;

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
        Conditional, DirectMulticast, InBlock, ProducerQMap, create_new_producer_qmap,
        process_conditional, process_direct_multicast, process_qmap_of_direct_multicast,
        run_on_operation,
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
    /// e516 — THE VENDOR'S OWN CASE (`:24-43`): the group each region yields gains that group's
    /// `$producer` beside it, the `if` gains the matching result, and the store reads the new result
    /// with the group in `$multicast_info`. ⭐ The register the `if` result carried is gone with the
    /// clone, which is what `cloneIfOp`'s empty attribute arrays mean.
    #[test]
    fn e516_yields_the_producer_beside_the_conditionally_chosen_group() {
        let (g1, g2) = (Val(1), Val(2));
        let (u3, u4) = (Val(3), Val(4));
        let chosen = Val(5);
        let mut body = vec![
            group(g1, u3),
            group(g2, u4),
            Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Eq,
                lhs: Val(6),
                rhs: Val(0),
                yielded: vec![sentient::Yielded {
                    result: chosen,
                    reg: sentient::Reg {
                        locale: sentient::RegType::Lrf,
                        index: Some(sentient::RegIndex::at::<2>()),
                    },
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![Op::Sentient(sentient::Op::Yield { results: vec![g1] })],
                else_body: vec![Op::Sentient(sentient::Op::Yield { results: vec![g2] })],
            }),
            store(StoreSource::Conditional(chosen), Val(7)),
        ];
        let mut values = Values::default();
        for _ in 0..8 {
            let _ = values.mint();
        }
        assert_eq!(
            process_conditional(chosen, InBlock(2), &mut body, &mut values),
            Conditional::Split
        );
        let unassigned = sentient::Reg {
            locale: sentient::RegType::Unknown,
            index: None,
        };
        let Op::Sentient(sentient::Op::If {
            yielded,
            then_body,
            else_body,
            ..
        }) = &body[2]
        else {
            panic!("the conditional survives")
        };
        assert_eq!(
            *yielded,
            vec![
                sentient::Yielded {
                    result: chosen,
                    reg: unassigned,
                    element_size: None,
                },
                sentient::Yielded {
                    result: Val(8),
                    reg: unassigned,
                    element_size: None,
                },
            ]
        );
        assert_eq!(
            *then_body,
            vec![Op::Sentient(sentient::Op::Yield {
                results: vec![g1, u3]
            })]
        );
        assert_eq!(
            *else_body,
            vec![Op::Sentient(sentient::Op::Yield {
                results: vec![g2, u4]
            })]
        );
        assert!(matches!(
            body[3],
            Op::Sentient(sentient::Op::ReceiveAndStore { producer, multicast_info, .. })
                if producer == StoreSource::Wire(RecvEnd::from_multicast_group(Val(8)))
                    && multicast_info == Some(chosen)
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

/// WHERE THE TOP `sentient.if` SITS IN THE BLOCK THAT HOLDS IT — `top_if_op` as a POSITION, because
/// the walk rewrites the block it was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InBlock(pub usize);

/// WHETHER THE CONDITIONAL TURNED OUT TO BE A MULTICAST ONE — the reference's two exits (`:263-272`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conditional {
    /// The `if` gained a `$producer` result and every store on it was moved onto it.
    Split,
    /// No region yielded a group, so nothing was rewritten — *"did not identify as a multicast
    /// scenario"* (`:264-266`).
    NotAMulticast,
}

/// WHAT THE WALK REMEMBERS ABOUT EVERY `sentient.if` IT HAS PASSED — the reference's
/// `std::map<sentient::IfOp, std::map<int, int>>` plus what an in-place growth needs beside it.
///
/// ⭐ AN `if` IS KEYED BY ITS FIRST RESULT. Op identity in a tree being rewritten cannot be a borrow,
/// and [`Values`] mints a result value unique to the op that binds it, so that value IS the op.
#[derive(Debug, Default)]
struct IndexMaps {
    /// Per `if`: old result index -> new result index.
    old_to_new: BTreeMap<Val, BTreeMap<usize, usize>>,
    /// Every result of every visited `if`, back to that `if`'s key.
    key_of_result: BTreeMap<Val, Val>,
    /// Per `if`: its result values AFTER it grew — `if_op.getResult(i)` for the new indices.
    results: BTreeMap<Val, Vec<Val>>,
    /// [`create_new_producer_qmap`]'s two ops, each with the mapping they are built in front of.
    inserted: Vec<(Val, ProducerQMap)>,
}

/// Replaces: e516_processConditional
///
/// Splits a conditionally chosen multicast group in two: every region of the `sentient.if` yields the
/// group's `$producer` beside the group itself, and every store on that `if` then reads the new result
/// as `$producer` with the old one as `$multicast_info`.
///
/// ⛔ TRAP: `cloneIfOp` LEAVES `regLocales`/`regIndices` EMPTY (`Dialect/Sentient/Utils.hpp:56-57`),
/// so EVERY result of a grown `if` comes back unassigned — not only the ones it gained.
/// ⛔ TRAP: A NESTED `if`'S FORWARDED RESULT IS LOOKED UP BY ITS POSITION IN **THIS** YIELD (`:225`),
/// not by its own result index among the nested `if`'s results.
/// ⭐ THE CLONE IS AN IN-PLACE GROWTH HERE, so *"the producer still defines from `top_if_op`"* (`:263`)
/// — the exit that changes nothing — is "the top `if` did not grow".
pub fn process_conditional(
    producer: Val,
    top_if: InBlock,
    body: &mut Vec<Op>,
    values: &mut Values,
) -> Conditional {
    // The tree as it was before the rewrite — `operand.getDefiningOp()` for a yield operand reaches
    // both outwards and inwards, and the ops it finds are the ones the walk does not touch.
    let snapshot = body.clone();
    let mut maps = IndexMaps::default();
    let Some(Op::Sentient(top)) = body.get_mut(top_if.0) else {
        return Conditional::NotAMulticast;
    };
    let Some(top_key) = grow_if(top, &snapshot, values, &mut maps) else {
        return Conditional::NotAMulticast;
    };
    let results = maps.results.remove(&top_key).unwrap_or_default();
    // `DT_CHECK_MSG(multicast_index >= 0, "expected a non-negative index")` (`:283`) — the caller took
    // this `if` from the producer's own defining op, so a producer that is not one of its results is
    // an abort, not an answer.
    let Some(multicast_index) = results.iter().position(|&result| result == producer) else {
        panic!("expected a non-negative index")
    };
    let Some(&producer_index) = maps
        .old_to_new
        .get(&top_key)
        .and_then(|indices| indices.get(&multicast_index))
    else {
        panic!("no mapping exists for the result index")
    };
    let new_producer = results[producer_index];
    for (map, built) in maps.inserted {
        insert_before_definition(body, map, &[built.map, built.qmap]);
    }
    assign_conditional_producer(body, producer, new_producer);
    Conditional::Split
}

/// ONE `sentient.if`, POST-ORDER: the `if`s nested under it, then each region's `sentient.yield`, then
/// the `if` itself — `visitIf` (`:238-252`). `Some(key)` exactly where the reference clones it.
fn grow_if(
    if_op: &mut sentient::Op,
    snapshot: &[Op],
    values: &mut Values,
    maps: &mut IndexMaps,
) -> Option<Val> {
    let sentient::Op::If {
        yielded,
        then_body,
        else_body,
        ..
    } = if_op
    else {
        return None;
    };
    let key = yielded.first().map(|entry| entry.result);
    if let Some(key) = key {
        for entry in yielded.iter() {
            maps.key_of_result.insert(entry.result, key);
        }
    }
    for region in [&mut *then_body, &mut *else_body] {
        grow_nested_ifs(region, snapshot, values, maps);
        // A yield with operands inside a result-less `if` is not a shape the reference can index.
        if let Some(key) = key {
            extend_yield(region, key, snapshot, values, maps);
        }
    }
    // `result_types` IS THE THEN-TERMINATOR'S OPERAND LIST (`:239-240`), and an `if` whose yield
    // needed no replacement is not replaced either (`:243-244`).
    let (Some(key), Some(then_count)) = (key, yield_operand_count(then_body)) else {
        return None;
    };
    if then_count <= yielded.len() {
        return None;
    }
    // ⛔ THE CLONE'S REGISTER ARRAYS ARE EMPTY, so every entry — old ones included — is unassigned.
    let unassigned = sentient::Reg {
        locale: sentient::RegType::Unknown,
        index: None,
    };
    for entry in yielded.iter_mut() {
        entry.reg = unassigned;
    }
    for _ in yielded.len()..then_count {
        yielded.push(sentient::Yielded {
            result: values.mint(),
            reg: unassigned,
            element_size: None,
        });
    }
    maps.results
        .insert(key, yielded.iter().map(|entry| entry.result).collect());
    Some(key)
}

/// EVERY `sentient.if` UNDER ONE REGION, AT ANY DEPTH, INNERMOST FIRST — the post-order walk
/// (`:255-261`), which descends through whatever else carries a region on the way.
fn grow_nested_ifs(
    block: &mut [Op],
    snapshot: &[Op],
    values: &mut Values,
    maps: &mut IndexMaps,
) {
    for op in block.iter_mut() {
        if let Op::Sentient(inner) = op
            && matches!(inner, sentient::Op::If { .. })
        {
            grow_if(inner, snapshot, values, maps);
            continue;
        }
        for region in dialects::regions_mut(op) {
            grow_nested_ifs(region, snapshot, values, maps);
        }
    }
}

/// `visitYield` (`:174-236`) — the operands one region's terminator gains, and the old-to-new result
/// index each of them records against the `if` that owns the yield.
///
/// ⛔ A YIELD INSIDE A `sentient.for` IS NOT ONE OF THESE: `dyn_cast<IfOp>(old_yield->getParentOp())`
/// returns early for it (`:176`), which is why only a region's own terminator is read.
fn extend_yield(
    block: &mut [Op],
    key: Val,
    snapshot: &[Op],
    values: &mut Values,
    maps: &mut IndexMaps,
) {
    let Some(Op::Sentient(sentient::Op::Yield { results })) = block.last_mut() else {
        return;
    };
    let mut new_operands: Vec<Val> = Vec::new();
    for (old_result_index, &operand) in results.iter().enumerate() {
        let new_result_index = results.len() + new_operands.len();
        // `isa<BlockArgument>(operand)` — kept as an operand, never expanded (`:182`).
        let Some(def) = dialects::defining_op(operand, snapshot) else {
            continue;
        };
        match def {
            Op::Dataflow(dataflow::Op::CreateMulticastGroup { producer, .. }) => {
                new_operands.push(*producer);
                maps.old_to_new
                    .entry(key)
                    .or_default()
                    .insert(old_result_index, new_result_index);
            }
            Op::Uniform(
                orig @ uniform::Op::QueryMap {
                    map,
                    key: query_key,
                    ..
                },
            ) => {
                let scope = [snapshot];
                let defs = dialects::Definitions::from_innermost(&scope);
                // `llvm::none_of(getListOfValueOpsFromUniformMapping(query_map), …)` (`:200-208`) —
                // a mapping that answers with no group at all is left as it is.
                let mapped = dialects::uniform_mapping_values(*map, *query_key, defs);
                if !mapped.iter().any(|&value| {
                    matches!(
                        dialects::defining_op(value, snapshot),
                        Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }))
                    )
                }) {
                    continue;
                }
                let keys = dialects::uniform_mapping_keys(*query_key, defs);
                // ⛔ DIVERGENCE: `DT_CHECK_MSG(new_query, "unable to create a producer map…")`
                // (`:209-211`) aborts; a mapping this cannot rebuild leaves the yield alone, which is
                // the divergence e324 already records.
                let Some(built) = create_new_producer_qmap(orig, &keys, snapshot, values) else {
                    continue;
                };
                new_operands.push(built.result);
                maps.inserted.push((*map, built));
                maps.old_to_new
                    .entry(key)
                    .or_default()
                    .insert(old_result_index, new_result_index);
            }
            // `if_op.getResult(results_index_map[if_op][old_result_index])` (`:222-226`) — a result
            // this walk has already expanded, forwarded one level out.
            Op::Sentient(sentient::Op::If { .. }) => {
                let Some(&nested) = maps.key_of_result.get(&operand) else {
                    continue;
                };
                let Some(&forwarded_index) = maps
                    .old_to_new
                    .get(&nested)
                    .and_then(|indices| indices.get(&old_result_index))
                else {
                    continue;
                };
                let Some(&forwarded) = maps
                    .results
                    .get(&nested)
                    .and_then(|results| results.get(forwarded_index))
                else {
                    continue;
                };
                new_operands.push(forwarded);
            }
            _ => {}
        }
    }
    // `YieldOp::create(builder, …, operands)` then `old_yield->erase()` (`:229-235`) — one yield in
    // place of the other, which in place is the operands it gained.
    results.extend(new_operands);
}

/// `old_if_op.getThenRegion().front().getTerminator()->getOperandTypes()` (`:239-240`) — how many
/// results the `if` must have, and `None` where the region has no terminator to ask.
fn yield_operand_count(block: &[Op]) -> Option<usize> {
    match block.last() {
        Some(Op::Sentient(sentient::Op::Yield { results })) => Some(results.len()),
        _ => None,
    }
}

/// `getProducerMutable().assign(new_if_op.getResult(producer_index))` and
/// `getMulticastInfoMutable().assign(tmp)` over every store reading the old `if` result as its
/// `$producer` (`:288-296`) — the same `$producer`-position filter e098's own note argues for.
fn assign_conditional_producer(block: &mut [Op], old_result: Val, new_producer: Val) {
    for op in block.iter_mut() {
        if let Op::Sentient(sentient::Op::ReceiveAndStore {
            producer,
            multicast_info,
            ..
        }) = op
            && producer.val() == old_result
        {
            *producer = StoreSource::Wire(RecvEnd::from_multicast_group(new_producer));
            *multicast_info = Some(old_result);
        }
        for region in dialects::regions_mut(op) {
            assign_conditional_producer(region, old_result, new_producer);
        }
    }
}

// crustify:todo: e569_runOn
//   authority : dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:327  (43 body lines, level 4)
//   original  : void MulticastCanonicalizationPass::runOn(sentient::ReceiveAndStoreOp ras)
//   calls     : e098_processDirectMulticast, e324_processQMapOfDirectMulticast, e516_processConditional

// crustify:todo: e605_runOn
//   authority : dcc/src/Transform/Sentient/MulticastCanonicalization.cpp:371  (24 body lines, level 5)
//   original  : void MulticastCanonicalizationPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e097_runOn, e569_runOn
