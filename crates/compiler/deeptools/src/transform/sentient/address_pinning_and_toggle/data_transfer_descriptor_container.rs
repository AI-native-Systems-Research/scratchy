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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e422_insert` | 422 | 2 | 9 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2169` |
//! | `e423_lookup` | 423 | 2 | 11 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2179` |
//! | `e491_computeChainingInfo` | 491 | 3 | 120 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2192` |

use std::collections::BTreeMap;

use super::looping_chain_mutable_addr_descriptor::mutable_addr_of;
use super::{DataTransferDescriptor, TransferEnd, op_at, unit_type_of};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, Val, results, sentient, use_count,
};
use crate::units::DfirUnit;

/// WHICH DESCRIPTOR — a position in [`DataTransferDescriptorContainer::descriptors`].
///
/// ⛔ THE REFERENCE KEYS ON THE ADDRESS OF THE DESCRIPTOR (`std::map<const DataTransferDescriptor *,
/// unsigned char> chaining_info_`, `:957`). A pointer key is not portable to Rust while the
/// container owns the same objects, and the index is stable because the container only ever grows by
/// `insert` and is cleared wholesale by `clear_all` (`:886`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescriptorId(pub u32);

/// ONE CHAINING BIT — `enum Flags : unsigned char` (`:939-943`), the argument
/// [`DataTransferDescriptorContainer::set_chaining_info`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChainFlag {
    /// `kPartOfChain = 0x01`.
    PartOfChain,
    /// `kHeadOfChain = 0x02`.
    HeadOfChain,
    /// `kHeadOfLoopingChain = 0x04`.
    HeadOfLoopingChain,
}

/// WHAT `chaining_info_` HOLDS FOR ONE DESCRIPTOR — the three `Flags` bits, named.
///
/// ⛔ AN ABSENT ENTRY IS NOT `Default`: `setChainingInfo(desc, flag, false)` on a descriptor with no
/// entry inserts NOTHING (`:945-955`), so [`DataTransferDescriptorContainer::chaining_info`] must
/// keep "no entry" distinct from "all three clear".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChainFlags {
    /// `kPartOfChain` — this transfer is part of some SSA chain.
    pub part_of_chain: bool,
    /// `kHeadOfChain` — and it is that chain's head.
    pub head_of_chain: bool,
    /// `kHeadOfLoopingChain` — and the chain's last link reaches back to it.
    pub head_of_looping_chain: bool,
}

impl ChainFlags {
    /// Reads one bit — `it->second & flag`.
    #[must_use]
    pub fn get(self, flag: ChainFlag) -> bool {
        match flag {
            ChainFlag::PartOfChain => self.part_of_chain,
            ChainFlag::HeadOfChain => self.head_of_chain,
            ChainFlag::HeadOfLoopingChain => self.head_of_looping_chain,
        }
    }

    /// Writes one bit — `|= flag` for `true`, `&= ~flag` for `false`.
    pub fn set(&mut self, flag: ChainFlag, val: bool) {
        match flag {
            ChainFlag::PartOfChain => self.part_of_chain = val,
            ChainFlag::HeadOfChain => self.head_of_chain = val,
            ChainFlag::HeadOfLoopingChain => self.head_of_looping_chain = val,
        }
    }
}

/// THE PASS'S DESCRIPTORS FOR ONE ADDRESS ROLE — `class DataTransferDescriptorContainer`
/// (`AddressPinningAndToggle.cpp:872-958`): stable syntactic-order iteration plus the chaining
/// relationships between its elements.
///
/// ⭐ IT OWNS THEM. The reference is a `std::vector<DataTransferDescriptor *>` whose `clear_all`
/// does not delete, because the pass's `cleanup()` (`:1656-1664`) deletes every element and then
/// clears — so each descriptor is reached, and freed, through exactly one container. Owning `Vec` is
/// that, minus the delete loop.
///
/// ⛔ NO `sorted_list_`, AND ITS OWN COMMENT ("only read when doing lookups", `:935-937`) UNDERSTATES
/// IT: `getSortedList()` (`:893`) has a second reader, `computeAddressInfoList` (e614, `:1685-1686`),
/// which ZIPS this container against the other one. That pairing survives the drop — both containers
/// are inserted into in lockstep from the same op (`:1410-1426`, sizes checked equal at `:1367`), so
/// descriptor `i` here describes the same transfer as descriptor `i` there under insertion order
/// exactly as it did under address order. ⭐ e614 PAIRS BY INDEX; only the visit ORDER is lost.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataTransferDescriptorContainer {
    /// The descriptors, in the syntactic order `insert` saw them.
    pub descriptors: Vec<DataTransferDescriptor>,
    /// `chaining_info_` — only the descriptors some chain has touched. See [`ChainFlags`].
    pub chaining_info: BTreeMap<DescriptorId, ChainFlags>,
}

impl DataTransferDescriptorContainer {
    /// Replaces: e422_insert
    ///
    /// Takes ownership of one descriptor, appended in the syntactic order the walk found it, and
    /// answers WHICH one it now is (`:2169-2177`).
    ///
    /// ⛔ THE `std::sort` IS NOT REPRODUCIBLE, AND `lookup` IS NOT ITS ONLY READER: the comparator is
    /// the raw ADDRESS of the op (`:2173-2176`), which Rust does not hand out, and besides
    /// [`Self::lookup`]'s `lower_bound` that order reaches `computeAddressInfoList`'s zip
    /// (`:1685-1686`) — [`DataTransferDescriptorContainer`] says why an index pairs that zip.
    /// ⭐ IT RETURNS AN ID THE REFERENCE'S CALLER DOES NOT KEEP: the `new`'d pointer is discarded at
    /// every callsite (`:1410-1431`) and descriptors are reached again by iterating (`:1443`, `:1452`)
    /// or by `lookup` (`:2225`) — but this `insert` MOVES the descriptor in, so the [`DescriptorId`] is
    /// the only handle those two readers have left.
    pub fn insert(&mut self, desc: DataTransferDescriptor) -> DescriptorId {
        self.validate();
        let id = DescriptorId(self.descriptors.len() as u32);
        self.descriptors.push(desc);
        id
    }

    /// Replaces: e423_lookup
    ///
    /// WHICH descriptor describes the op at `op`, and `None` for an op no descriptor in this container
    /// describes (`:2179-2189`).
    ///
    /// ⛔ THE `lower_bound` IS THE MEMOISATION, NOT THE ANSWER: the reference binary-searches
    /// `sorted_list_` by op address and then re-tests `&(*iter)->getOperation() == op` (`:2188`),
    /// because address order only narrows the candidate — that final equality is the whole predicate,
    /// and a scan of `descriptors` decides it identically.
    /// ⛔ `DT_CHECK(op)` (`:2183`) HAS NO ARM HERE: an [`OpId`] is not nullable, so the reference's
    /// null `Operation *` is not expressible.
    #[must_use]
    pub fn lookup(&self, op: &OpId) -> Option<DescriptorId> {
        self.validate();
        self.descriptors
            .iter()
            .position(|desc| desc.op == *op)
            .map(|index| DescriptorId(index as u32))
    }
}

impl DataTransferDescriptorContainer {
    /// Replaces: e491_computeChainingInfo
    ///
    /// WRITES EVERY CHAINING BIT THIS CONTAINER HOLDS (`:2192-2311`): pass one decides part-of/head-of
    /// chain from what feeds each transfer's mutable address, pass two walks each chain forward to see
    /// whether it closes through its own loop's yield and re-heads whatever consumes the loop result.
    ///
    /// ⛔ ALL THREE `LLVM_DEBUG`/`DEBUG_WITH_TYPE` BLOCKS ARE TRACING, and the trailing one only reads
    /// e007/e273/e417 back.
    /// ⛔ THE `while (!found)` LOOP ADVANCES ONLY THROUGH TRANSFERS: every other user breaks it, so a
    /// chain that reaches anything else is not looping.
    pub fn compute_chaining_info(&mut self, unit_body: &[Op]) {
        let regions: [&[Op]; 1] = [unit_body];
        let defs = Definitions::from_innermost(&regions);

        for index in 0..self.descriptors.len() {
            let curr = DescriptorId(index as u32);
            let desc = &self.descriptors[index];
            if !desc.is_valid() {
                continue;
            }
            let unit = desc.memory_unit.dfir_unit();
            let op_id = desc.op.clone();
            let Some(op) = op_at(&op_id, unit_body) else {
                todo!("computeChainingInfo: DT_CHECK(at(i)) — no op at {op_id:?} (:2196)")
            };
            let mutable_addr = mutable_addr_for_unit(op, unit, defs);
            // `!isa<BlockArgument>(mutable_addr)` IS "it has a defining op", and its position is the
            // key `lookup` answers about.
            let defining = defining_op_id(unit_body, mutable_addr);
            let fed_by_transfer = defining
                .as_ref()
                .and_then(|id| op_at(id, unit_body))
                .is_some_and(is_transfer);

            if fed_by_transfer {
                self.set_chaining_info(curr, ChainFlag::PartOfChain, true);
                self.set_chaining_info(curr, ChainFlag::HeadOfChain, false);
                // The transfer feeding this address is the chain's head when ITS own mutable address
                // comes from outside the chain (`:2220-2234`).
                let head = defining.as_ref().and_then(|id| self.lookup(id));
                if let Some(head) = head {
                    let head_desc = &self.descriptors[head.0 as usize];
                    let head_unit = head_desc.memory_unit.dfir_unit();
                    let head_op_id = head_desc.op.clone();
                    let Some(head_op) = op_at(&head_op_id, unit_body) else {
                        todo!(
                            "computeChainingInfo: no op at {head_op_id:?} for the potential head \
                             (:2226-2229)"
                        )
                    };
                    let head_addr = mutable_addr_for_unit(head_op, head_unit, defs);
                    let head_fed_by_transfer = defining_op_id(unit_body, head_addr)
                        .and_then(|id| op_at(&id, unit_body).cloned())
                        .is_some_and(|op| is_transfer(&op));
                    if !head_fed_by_transfer {
                        self.set_chaining_info(head, ChainFlag::PartOfChain, true);
                        self.set_chaining_info(head, ChainFlag::HeadOfChain, true);
                    }
                }
            } else if results(op)
                .iter()
                .all(|val| use_count(*val, unit_body) == 0)
            {
                // `curr_desc.getOperation().use_empty()` — nothing reads this transfer at all.
                self.set_chaining_info(curr, ChainFlag::PartOfChain, false);
                self.set_chaining_info(curr, ChainFlag::HeadOfChain, false);
            } else if defining.is_none()
                && use_count(
                    mutable_addr_result_of(op, mutable_addr_end(op, unit, defs)),
                    unit_body,
                ) == 1
            {
                // The loop-carried case: the address arrives as an iter arg and the result it produces
                // is read once (`:2242-2252`).
                self.set_chaining_info(curr, ChainFlag::PartOfChain, true);
                self.set_chaining_info(curr, ChainFlag::HeadOfChain, true);
            }
        }

        for index in 0..self.descriptors.len() {
            let curr = DescriptorId(index as u32);
            let desc = &self.descriptors[index];
            if !desc.is_valid() {
                continue;
            }
            let unit = desc.memory_unit.dfir_unit();
            let op_id = desc.op.clone();
            let Some(op) = op_at(&op_id, unit_body) else {
                todo!("computeChainingInfo: no op at {op_id:?} in the looping-chain pass (:2258)")
            };
            let mut curr_val = mutable_addr_result_of(op, mutable_addr_end(op, unit, defs));
            let parent = parent_for_of(&op_id, unit_body);
            let mut reached_a_yield = false;
            loop {
                if use_count(curr_val, unit_body) != 1 {
                    break;
                }
                let Some(user) =
                    using_op_id(unit_body, curr_val).and_then(|id| op_at(&id, unit_body))
                else {
                    break;
                };
                if is_transfer(user) {
                    // ⭐ THE UNIT STAYS `curr_desc`'s, not the user's — the reference passes
                    // `curr_desc.getMemoryUnit()` all the way down (`:2266-2267`).
                    curr_val = mutable_addr_result_of(user, mutable_addr_end(user, unit, defs));
                    continue;
                }
                reached_a_yield = matches!(user, Op::Sentient(sentient::Op::Yield { .. }));
                break;
            }
            // `yield_op->getParentOp()` being `curr_desc.getOperation().getParentOp()` and yielding
            // `curr_val` is one question here: `curr_val` has a single use, so a parent loop that
            // yields it IS that use (`:2268-2275`).
            let found =
                reached_a_yield && parent.is_some_and(|(yielded, _)| yielded.contains(&curr_val));
            self.set_chaining_info(curr, ChainFlag::HeadOfLoopingChain, found);

            // The chain can continue AFTER the loop, through the loop result carrying `curr_val`
            // (`:2280-2303`).
            if found && let Some((yielded, carried)) = parent {
                let for_result = yielded
                    .iter()
                    .position(|val| *val == curr_val)
                    .and_then(|res_idx| carried.get(res_idx))
                    .map(|entry| entry.result);
                let Some(for_result) = for_result else {
                    todo!(
                        "computeChainingInfo: DT_CHECK(res_idx >= 0 && res_idx < \
                         inner_for_op.getNumResults()) — {curr_val:?} is yielded at a position the \
                         loop carries nothing for (:2290-2292)"
                    )
                };
                if use_count(for_result, unit_body) == 1
                    && let Some(user_id) = using_op_id(unit_body, for_result)
                    && op_at(&user_id, unit_body).is_some_and(is_transfer)
                    && let Some(consumer) = self.lookup(&user_id)
                {
                    self.set_chaining_info(consumer, ChainFlag::HeadOfChain, false);
                    self.set_chaining_info(consumer, ChainFlag::PartOfChain, true);
                }
            }
        }
    }
}

/// THE THREE OPS THE CHAINING WALK FOLLOWS — `isa<LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>`
/// (`:2216-2218`, `:2263-2264`, `:2296-2297`).
///
/// ⛔ `LoadAndExtractScalarOp` AND `LoadComputeAndSendOp` ARE DELIBERATELY ABSENT — "they are not
/// supported outside LX" (`:2213-2214`), and they are absent from all three of those `isa` lists.
fn is_transfer(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(
            sentient::Op::LoadAndSend { .. }
                | sentient::Op::ReceiveAndStore { .. }
                | sentient::Op::LoadAndStore { .. }
        )
    )
}

/// `dcc::utils::getMutableAddrResultIndex(op, mem_unit)` (`Analyses/Utils.cpp:536-555`) as WHICH END
/// of the transfer carries `unit`'s address.
///
/// ⛔ THE `-1` IS NOT EXPRESSIBLE AS AN END, AND EVERY CALLER HERE DEREFERENCES IT: e491 feeds this
/// index straight into `getResult(..)` (`:2245-2248`, `:2260-2261`) and into
/// `getMutableAndImmutableAddr`, whose `{nullptr, nullptr}` then meets an `isa<BlockArgument>`
/// (`:2216`). Both are the reference's own stop, so the two non-answers stay named stops here.
fn mutable_addr_end(op: &Op, unit: DfirUnit, defs: Definitions<'_>) -> TransferEnd {
    match op {
        // `return 0` for the three single-address ops, and `getAddrResultIdx()` for
        // `LoadAndExtractScalarOp`, whose ONE address is its [`TransferEnd::Src`] here.
        Op::Sentient(
            sentient::Op::LoadAndSend { .. }
            | sentient::Op::ReceiveAndStore { .. }
            | sentient::Op::LoadComputeAndSend { .. }
            | sentient::Op::LoadAndExtractScalar { .. },
        ) => TransferEnd::Src,
        Op::Sentient(sentient::Op::LoadAndStore {
            src,
            dst,
            is_ibr_write,
            ..
        }) => {
            if unit_type_of(*src, defs) == Some(unit) {
                TransferEnd::Src
            } else if unit_type_of(*dst, defs) == Some(unit) {
                TransferEnd::Dst
            } else if *is_ibr_write {
                todo!(
                    "getMutableAddrResultIndex: llvm_unreachable(\"cannot obtain mem_unit addr \
                     from this operation\") — an IBR-write sentient.load_and_store with neither end \
                     on {unit:?} (Analyses/Utils.cpp:550-554)"
                )
            } else {
                todo!(
                    "computeChainingInfo: getMutableAddrResultIndex answered -1 for the {unit:?} \
                     address of a sentient.load_and_store, which the reference then hands to \
                     getResult/isa<BlockArgument> (:2216, Analyses/Utils.cpp:550-551)"
                )
            }
        }
        _ => todo!(
            "getMutableAddrResultIndex: llvm_unreachable(\"cannot obtain mem_unit addr from this \
             operation\") on {op:?} (Analyses/Utils.cpp:554)"
        ),
    }
}

/// `op.getResult(getMutableAddrResultIndex(op, mem_unit))` — the SSA value the next link of a chain
/// reads as its own mutable address.
fn mutable_addr_result_of(op: &Op, end: TransferEnd) -> Val {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { result, .. }
            | sentient::Op::ReceiveAndStore { result, .. }
            | sentient::Op::LoadComputeAndSend { result, .. },
        ) => *result,
        // `getAddrResultIdx()` names the ADDRESS result, never the extracted datum.
        Op::Sentient(sentient::Op::LoadAndExtractScalar { addr_result, .. }) => *addr_result,
        Op::Sentient(sentient::Op::LoadAndStore { results, .. }) => match end {
            TransferEnd::Src => results.0,
            TransferEnd::Dst => results.1,
        },
        _ => todo!(
            "computeChainingInfo: getResult(getMutableAddrResultIndex(..)) on {op:?}, which binds no \
             address result (Analyses/Utils.cpp:554)"
        ),
    }
}

/// `dcc::utils::getMutableAndImmutableAddr(op, mem_unit).first` (`Analyses/Utils.cpp:556-567`).
fn mutable_addr_for_unit(op: &Op, unit: DfirUnit, defs: Definitions<'_>) -> Val {
    mutable_addr_of(op, mutable_addr_end(op, unit, defs))
}

/// `val.getDefiningOp()` AS A POSITION — `None` for the reference's `isa<BlockArgument>(val)`, which
/// is exactly the value no op in this scope defines.
fn defining_op_id(unit_body: &[Op], val: Val) -> Option<OpId> {
    find_op_id(
        unit_body,
        0,
        &|op| results(op).contains(&val),
        &mut Vec::new(),
    )
}

/// `*val.user_begin()` AS A POSITION — only meaningful where `val` has exactly one use, which both
/// callers check first.
fn using_op_id(unit_body: &[Op], val: Val) -> Option<OpId> {
    find_op_id(
        unit_body,
        0,
        &|op| dialects::operands(op).contains(&val),
        &mut Vec::new(),
    )
}

/// THE POSITION OF THE FIRST OP SATISFYING `pred`, in the regions-concatenated numbering [`op_at`]
/// reads back — `base` is where the region being walked starts in its owner's flattened child list.
fn find_op_id(
    scope: &[Op],
    base: u32,
    pred: &dyn Fn(&Op) -> bool,
    prefix: &mut Vec<u32>,
) -> Option<OpId> {
    for (index, op) in scope.iter().enumerate() {
        prefix.push(base + index as u32);
        if pred(op) {
            let id = OpId::at(prefix);
            prefix.pop();
            return Some(id);
        }
        let mut offset = 0u32;
        for region in dialects::regions_ref(op) {
            if let Some(found) = find_op_id(region, offset, pred, prefix) {
                prefix.pop();
                return Some(found);
            }
            offset += region.len() as u32;
        }
        prefix.pop();
    }
    None
}

/// THE `sentient.for` HOLDING THE OP AT `id`, as its terminator's yielded operands and its carried
/// entries — `getParentOp()` off a position is its block prefix (see [`OpId::block`]).
fn parent_for_of<'a>(
    id: &OpId,
    unit_body: &'a [Op],
) -> Option<(&'a [Val], &'a [sentient::Carried])> {
    let Op::Sentient(sentient::Op::For { carried, body, .. }) =
        op_at(&OpId::at(id.block()), unit_body)?
    else {
        return None;
    };
    let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
        return None;
    };
    Some((results, carried))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{
        Carried, Extent, Reg, RegType, ShuffleMode,
    };
    use crate::transform::sentient::address_pinning_and_toggle::DescriptorMemoryUnit;
    use crate::transform::sentient::analyses::{EvaluatedValue, RegionSite};

    /// One transfer describing the op at `path` — `op_` is the only field either unit reads.
    fn transfer(path: &[u32]) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(path),
            pattern_desc: None,
            base_addrs: Vec::new(),
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
        }
    }

    /// 422/656 — insertion order is the identity, so the ids come back 0, 1, 2 and the container holds
    /// the descriptors in the order the walk saw them.
    #[test]
    fn e422_insert_appends_and_names_the_position_it_appended_at() {
        let mut container = DataTransferDescriptorContainer::default();
        let first = container.insert(transfer(&[0]));
        let second = container.insert(transfer(&[3, 1]));
        let third = container.insert(transfer(&[3, 2]));
        assert_eq!(
            [first, second, third],
            [DescriptorId(0), DescriptorId(1), DescriptorId(2)]
        );
        assert_eq!(
            container
                .descriptors
                .iter()
                .map(|desc| desc.op.path().to_vec())
                .collect::<Vec<_>>(),
            vec![vec![0], vec![3, 1], vec![3, 2]]
        );
    }

    /// 423/656 — the final `&(*iter)->getOperation() == op` both ways: a nested op that IS described
    /// is found at its own position, and one at a path no descriptor holds is the `nullptr`.
    ///
    /// ⛔ `[3, 1]` AND `[3]` ARE DIFFERENT OPS, not the same one at two depths — a prefix must miss,
    /// which is what a `lower_bound` over addresses would have no way to get wrong and a sloppy scan
    /// would.
    #[test]
    fn e423_lookup_finds_the_descriptor_whose_op_is_exactly_this_one() {
        let mut container = DataTransferDescriptorContainer::default();
        container.insert(transfer(&[0]));
        container.insert(transfer(&[3, 1]));
        assert_eq!(container.lookup(&OpId::at(&[3, 1])), Some(DescriptorId(1)));
        assert_eq!(container.lookup(&OpId::at(&[0])), Some(DescriptorId(0)));
        assert_eq!(container.lookup(&OpId::at(&[3])), None);
        assert_eq!(container.lookup(&OpId::at(&[1])), None);
    }

    /// One `sentient.load_and_send` reading `mutable_addr` and binding `result` — the two fields the
    /// chaining walk follows.
    fn load_and_send(mutable_addr: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr,
            immutable_addr: Val(90),
            increment: Val(91),
            consumer: SendEnd::to_self(Val(92)),
            result,
            extent: Extent {
                total_elements: Elements(64),
                element_size: Bits(16),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// A VALID transfer at `path` — one base address is what [`DataTransferDescriptor::is_valid`]
    /// needs with no pattern, and `computeChainingInfo` skips anything invalid.
    fn chained(path: &[u32]) -> DataTransferDescriptor {
        DataTransferDescriptor {
            base_addrs: vec![EvaluatedValue(0)],
            ..transfer(path)
        }
    }

    /// The vendor's own two cases in one nest (`:2203-2212`): a loop-carried address chained into a
    /// second transfer whose yielded result feeds a third after the loop.
    ///
    /// ⭐ THE THIRD DESCRIPTOR PROVES e008's ASYMMETRY: pass one clears both its bits, inserting NO
    /// entry, and pass two's post-loop arm is what puts it in the chain.
    #[test]
    fn e491_heads_the_carried_transfer_chains_the_second_and_re_heads_the_loops_consumer() {
        let unit_body = vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(0),
                reg_locale: RegType::Imm,
                ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(1),
                bound: Val(0),
                bound_reg: None,
                carried: vec![Carried {
                    init: Val(0),
                    arg: Val(2),
                    result: Val(3),
                    reg: Reg {
                        locale: RegType::Unknown,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    load_and_send(Val(2), Val(4)),
                    load_and_send(Val(4), Val(5)),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(5)],
                    }),
                ],
            }),
            load_and_send(Val(3), Val(6)),
        ];
        let mut container = DataTransferDescriptorContainer::default();
        let carried = container.insert(chained(&[1, 0]));
        let inner = container.insert(chained(&[1, 1]));
        let consumer = container.insert(chained(&[2]));

        container.compute_chaining_info(&unit_body);

        // The carried transfer is the head of a chain that closes through its own loop's yield.
        assert_eq!(
            container.chaining_info.get(&carried),
            Some(&ChainFlags {
                part_of_chain: true,
                head_of_chain: true,
                head_of_looping_chain: true,
            })
        );
        // The second link is in the chain and is not its head — and its own walk reaches the same
        // yield, so it is a looping head too.
        assert_eq!(
            container.chaining_info.get(&inner),
            Some(&ChainFlags {
                part_of_chain: true,
                head_of_chain: false,
                head_of_looping_chain: true,
            })
        );
        // ⭐ REACHED ONLY BY PASS TWO: pass one saw an unused transfer fed by the loop, not by a
        // transfer, and cleared two bits into no entry at all.
        assert_eq!(
            container.chaining_info.get(&consumer),
            Some(&ChainFlags {
                part_of_chain: true,
                head_of_chain: false,
                head_of_looping_chain: false,
            })
        );
    }
}
