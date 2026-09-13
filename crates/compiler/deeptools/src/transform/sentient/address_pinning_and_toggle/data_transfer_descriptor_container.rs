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

// crustify:todo: e491_computeChainingInfo
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2192  (120 body lines, level 3)
//   original  : void DataTransferDescriptorContainer::computeChainingInfo()
//   calls     : e007_isPartOfSomeChain, e008_setChainingInfo, e252_size, e273_isHeadOfChain, e278_isValid, e417_isHeadOfLoopingChain, e423_lookup

use std::collections::BTreeMap;

use super::DataTransferDescriptor;
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;

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
/// ⛔ NO `sorted_list_`: it is a lookup memoisation (`:937`), which the campaign lets a port drop —
/// `lookup` (`e423`) is scheduled separately and owns that decision.
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
    /// ⛔ THE `std::sort` GOES WITH `sorted_list_`, AND ORDERS NOTHING OBSERVABLE: its comparator is
    /// `&a->getOperation() < &b->getOperation()` (`:2173-2176`), the raw ADDRESS of the op, and the
    /// only reader of that order is [`Self::lookup`]'s `lower_bound`. Reproducing it would need
    /// allocation addresses Rust does not hand out, and it would still answer the same question.
    /// ⭐ IT RETURNS THE [`DescriptorId`] THE REFERENCE'S CALLER ALREADY HELD: the reference is handed
    /// a pointer it keeps using (`:1444`, `:2226`), so a `()` return would make the descriptor it just
    /// moved in unreachable.
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::transform::sentient::analyses::RegionSite;

    /// One transfer describing the op at `path` — `op_` is the only field either unit reads.
    fn transfer(path: &[u32]) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(path),
            pattern_desc: None,
            base_addrs: Vec::new(),
            region: RegionSite::default(),
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
}
