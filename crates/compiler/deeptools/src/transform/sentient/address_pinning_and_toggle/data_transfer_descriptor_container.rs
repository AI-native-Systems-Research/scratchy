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


// crustify:todo: e422_insert
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2169  (9 body lines, level 2)
//   original  : void DataTransferDescriptorContainer::insert(DataTransferDescriptor *desc)
//   calls     : e274_validate

// crustify:todo: e423_lookup
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2179  (11 body lines, level 2)
//   original  : DataTransferDescriptor *DataTransferDescriptorContainer::lookup( const Operation *op) const
//   calls     : e274_validate

// crustify:todo: e491_computeChainingInfo
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2192  (120 body lines, level 3)
//   original  : void DataTransferDescriptorContainer::computeChainingInfo()
//   calls     : e007_isPartOfSomeChain, e008_setChainingInfo, e252_size, e273_isHeadOfChain, e278_isValid, e417_isHeadOfLoopingChain, e423_lookup


use std::collections::BTreeMap;

use super::DataTransferDescriptor;

/// WHICH DESCRIPTOR — a position in [`DataTransferDescriptorContainer::descriptors`].
///
/// ⛔ THE REFERENCE KEYS ON THE ADDRESS OF THE DESCRIPTOR (`std::map<const DataTransferDescriptor *,
/// unsigned char> chaining_info_`, `:955`). A pointer key is not portable to Rust while the
/// container owns the same objects, and the index is stable because the container only ever grows by
/// `insert` and is cleared wholesale by `clear_all` (`:888`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescriptorId(pub u32);

/// ONE CHAINING BIT — `enum Flags : unsigned char` (`:944-948`), the argument
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
/// entry inserts NOTHING (`:949-958`), so [`DataTransferDescriptorContainer::chaining_info`] must
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
/// (`AddressPinningAndToggle.cpp:872-956`): stable syntactic-order iteration plus the chaining
/// relationships between its elements.
///
/// ⭐ IT OWNS THEM. The reference is a `std::vector<DataTransferDescriptor *>` whose `clear_all`
/// does not delete, because the pass's `cleanup()` (`:1657-1663`) deletes every element and then
/// clears — so each descriptor is reached, and freed, through exactly one container. Owning `Vec` is
/// that, minus the delete loop.
///
/// ⛔ NO `sorted_list_`: it is a lookup memoisation (`:934`), which the campaign lets a port drop —
/// `lookup` (`e423`) is scheduled separately and owns that decision.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataTransferDescriptorContainer {
    /// The descriptors, in the syntactic order `insert` saw them.
    pub descriptors: Vec<DataTransferDescriptor>,
    /// `chaining_info_` — only the descriptors some chain has touched. See [`ChainFlags`].
    pub chaining_info: BTreeMap<DescriptorId, ChainFlags>,
}
