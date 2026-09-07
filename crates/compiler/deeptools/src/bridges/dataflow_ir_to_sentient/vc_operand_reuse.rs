// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `OperandReuse.cpp` — 7 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e055_getId` | 055/384 | 6 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65` |
//! | `e056_getAbsorbtionFlag` | 056/384 | 6 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73` |
//! | `e057_setReuseFlag` | 057/384 | 2 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91` |
//! | `e058_dominates` | 058/384 | 2 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38` |
//! | `e162_insertIfNotExists` | 162/384 | 8 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81` |
//! | `e227_OperandReuse` | 227/384 | 0 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30` |
//! | `e276_setReuseInformation` | 276/384 | 44 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17` |
//!
//! Original files homed here: `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp`, `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp`

use std::collections::HashMap;

use crate::islands::dataflow_ir::dialects::Val;

/// WHICH DATA ORIGIN AN OPERAND READS, AS THE EMITTED OP SPELLS IT.
///
/// ⛔⛔ `Unassigned` IS THE REFERENCE'S **MINUS ONE**, AND IT IS PRINTED. `OperandTag::id_` is
/// declared `int id_ = -1` (`OperandReuse.hpp:44`) and `getId` answers `-1` for an op it has never
/// seen; the caller writes that straight into the emitted compute's `op<X>DataID`, where -1 is also
/// the `.td` default — but *explicitly set*, so it prints. `opADataID = -1 : si32` stands in a
/// `CHECK-SENT-IR` line of `dcc/test/PE/test1.mlir:18`, and 12,350 explicit -1 data ids across 144
/// of the authority's test files say the same.
///
/// ⭐ SO THIS IS A TWO-CASE TYPE AND NOT AN `Option` THAT VANISHES: an absent attribute and an
/// explicit -1 print differently, and
/// [`Operand::data_id`](crate::islands::sentient::dialects::sentient::Operand) already keeps that
/// difference — `Some(-1)` prints, `None` does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataId {
    /// The origin's position in the order [`OperandReuse`] first saw it.
    Assigned(DataOriginId),
    /// ⛔ NOT "NO ANSWER" — the answer -1, which the emitted op carries and prints.
    #[default]
    Unassigned,
}

impl DataId {
    /// THE `si32` THE ATTRIBUTE CARRIES — the id, or -1.
    ///
    /// ⭐ THE SENTINEL LIVES HERE AND NOWHERE ELSE, so no other unit has to remember that -1 is what
    /// an unseen origin answers.
    #[must_use]
    pub const fn attribute(self) -> i32 {
        match self {
            // ⭐ THE ID IS A MAP SIZE (`insertIfNotExists`, entry 162), so it cannot reach the
            // sentinel from below: 0 is a real data id.
            DataId::Assigned(id) => id.0 as i32,
            DataId::Unassigned => -1,
        }
    }
}

/// ONE DATA ORIGIN'S POSITION IN THE ORDER IT WAS FIRST SEEN.
///
/// ⛔ A NEWTYPE BECAUSE IT IS NOT A COUNT AND NOT A PORT. `insertIfNotExists` mints it as
/// `data_origins_.size()` (`OperandReuse.cpp:83`), so it is dense and starts at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataOriginId(pub u32);

/// WHAT [`OperandReuse`] REMEMBERS ABOUT ONE DATA ORIGIN — `OperandReuse.hpp:43-46`.
///
/// ```cpp
/// struct OperandTag {
///   int id_ = -1;
///   bool absorbed_ = false;
/// };
/// ```
///
/// ⭐ THE DEFAULTS ARE THE C++ MEMBER INITIALISERS, and both survive as this type's [`Default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OperandTag {
    /// `id_`.
    pub id: DataId,
    /// `absorbed_` — whether a chain has already consumed this origin's value.
    pub absorbed: bool,
}

/// THE DATA-ORIGIN TABLE ONE PROGRAM UNIT'S VECTOR CHAINS SHARE — `OperandReuse.hpp:24`.
///
/// ⛔ KEYED BY THE VALUE THE ORIGIN PRODUCES. The reference keys on `Operation *`, a pointer
/// identity; a data origin is always an op with a result (it is reached through a
/// `VectorOperand`'s `op_`), so its result [`Val`] identifies it exactly and this crate has no
/// address to key on.
///
/// ⛔ THE CLASS DECLARATION IS ENTRY 227'S AND `dominance_info_` COMES WITH IT, together with the
/// `dominates` predicate that reads it (entry 058, `OperandReuse.hpp:38`). This is the minimum the
/// two getters below need, and it grows when those units land — [`Default`] stands in for the
/// constructor's `data_origins_.clear()` (`OperandReuse.hpp:26-28`) until then.
#[derive(Debug, Default)]
pub struct OperandReuse {
    /// `data_origins_`.
    data_origins: HashMap<Val, OperandTag>,
}

impl OperandReuse {
    /// Replaces: e055_getId
    ///
    /// **055/384** `OperandReuse::getId` —
    /// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65` (6L).
    ///
    /// ```cpp
    /// std::optional<int> OperandReuse::getId(Operation *op) {
    ///   if (data_origins_.count(op) != 0) {
    ///     return data_origins_[op].id_;
    ///   } else {
    ///     return -1;
    ///   }
    /// }
    /// ```
    ///
    /// # ⛔⛔ THE `optional` IS NEVER EMPTY, SO IT IS A SENTINEL AND NOT AN ABSENCE
    ///
    /// Both branches return an engaged `std::optional<int>`: the id on a hit, and **-1** — not
    /// `std::nullopt` — on a miss. Its callers rely on that; every one of them is a bare
    /// `.value()` with nothing testing `has_value()` first — all 62 of them, across
    /// `VectorChainToSentientPESFP.cpp:123`, `:363`, `:560-562` and
    /// `VectorChainToSentientPT.cpp:144`, `:411-412`, `:931` — and each writes the result straight
    /// into an emitted compute's `op<X>DataID`. Port the optional as an `Option` and every one of
    /// those call sites gains an arm the reference cannot take.
    ///
    /// ⭐ SO THE RETURN TYPE IS [`DataId`], WHICH HAS THE TWO ANSWERS AND NO THIRD ONE. That -1
    /// reaches the output is the reason this matters rather than being a tidying: see [`DataId`].
    ///
    /// ⛔ AND ITS SISTER [`OperandReuse::absorbtion_flag`] IS THE OTHER CASE — same shape, same six
    /// lines, and there the empty optional is real. The pair is only worth porting together
    /// because they differ in exactly that.
    #[must_use]
    pub fn id(&self, op: Val) -> DataId {
        // ⭐ THE MISS AND THE `OperandTag` DEFAULT ARE THE SAME -1, which is what lets the two
        // branches read as one lookup without either of them going missing.
        self.data_origins
            .get(&op)
            .map_or(DataId::Unassigned, |tag| tag.id)
    }

    /// Replaces: e056_getAbsorbtionFlag
    ///
    /// **056/384** `OperandReuse::getAbsorbtionFlag` —
    /// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73` (6L).
    ///
    /// ```cpp
    /// std::optional<bool> OperandReuse::getAbsorbtionFlag(Operation *op) {
    ///   if (data_origins_.count(op) != 0) {
    ///     return data_origins_[op].absorbed_;
    ///   } else {
    ///     return std::nullopt;
    ///   }
    /// }
    /// ```
    ///
    /// # ⛔⛔ HERE THE EMPTY OPTIONAL IS REAL, AND BOTH CALLERS TEST FOR IT
    ///
    /// Unlike [`OperandReuse::id`], the miss branch returns `std::nullopt`, and the two callers that
    /// read the flag distinguish **three** answers, not two:
    ///
    /// ```cpp
    /// auto absorption_flag = reuse_info.getAbsorbtionFlag(from.op_);
    /// if (absorption_flag.has_value() && !absorption_flag.value()) {
    ///   // it has been used but not absorbed by some other op that is already lowered
    /// ```
    ///
    /// (`VectorChainToSentientPESFP.cpp:1289-1290`, and verbatim at
    /// `VectorChainToSentientPT.cpp:898-899`.) Only `Some(false)` — registered as a data origin and
    /// *not* absorbed — makes them emit the mask constant and the consuming compute that follow;
    /// `Some(true)` and `None` are both silence, for opposite reasons. So the [`Option`] survives
    /// the port: it is the reference's own three-way answer, and `false` is not `None`.
    ///
    /// ⛔ THE ONE UNGUARDED `.value()` CANNOT BE THE EMPTY CASE, AND THAT IS WHY IT IS SAFE:
    /// `setReuseInformation` (entry 276) writes `} else if (this->getAbsorbtionFlag(operand_i.op_)
    /// .value())`, but the `if` it chains off is `!this->insertIfNotExists(operand_i.op_)` on the
    /// same op (`OperandReuse.cpp:27-29`) — the short circuit means the entry was just inserted or
    /// was already there. ⭐ THE PORT KEEPS THAT AS THE SAME ORDERING, not as an unwrap.
    #[must_use]
    pub fn absorbtion_flag(&self, op: Val) -> Option<bool> {
        self.data_origins.get(&op).map(|tag| tag.absorbed)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// A TABLE HOLDING THE GIVEN ORIGINS.
    ///
    /// ⛔ WRITTEN STRAIGHT, NOT THROUGH `insertIfNotExists` — that is entry 162 and it does not
    /// exist yet. The ids here are what it will mint: the position of first insertion.
    fn table(origins: &[(Val, OperandTag)]) -> OperandReuse {
        OperandReuse {
            data_origins: origins.iter().copied().collect(),
        }
    }

    /// 🎯 055/384 — A REGISTERED ORIGIN ANSWERS WITH ITS ID.
    #[test]
    fn a_registered_origin_answers_with_its_id() {
        let reuse = table(&[(
            Val(4),
            OperandTag {
                id: DataId::Assigned(DataOriginId(0)),
                absorbed: false,
            },
        )]);
        assert_eq!(reuse.id(Val(4)), DataId::Assigned(DataOriginId(0)));
        assert_eq!(reuse.id(Val(4)).attribute(), 0, "zero is a real data id");
    }

    /// 🎯 055/384 — ⛔ AND AN UNREGISTERED ONE ANSWERS **MINUS ONE**, NOT "NO ANSWER".
    ///
    /// The reference's miss branch is `return -1;` on an engaged optional, and its callers
    /// `.value()` it without a guard, so the -1 lands in the emitted compute's `op<X>DataID` and
    /// prints there — `opADataID = -1 : si32`, which 88 of the authority's test cases show.
    #[test]
    fn an_unregistered_origin_answers_minus_one() {
        let reuse = table(&[]);
        assert_eq!(reuse.id(Val(4)), DataId::Unassigned);
        assert_eq!(reuse.id(Val(4)).attribute(), -1);
    }

    /// 🎯 055/384 — AND THE ANSWER IS PER-ORIGIN, NOT PER-TABLE.
    #[test]
    fn each_origin_keeps_its_own_id() {
        let reuse = table(&[
            (
                Val(4),
                OperandTag {
                    id: DataId::Assigned(DataOriginId(0)),
                    absorbed: false,
                },
            ),
            (
                Val(9),
                OperandTag {
                    id: DataId::Assigned(DataOriginId(1)),
                    absorbed: true,
                },
            ),
        ]);
        assert_eq!(reuse.id(Val(4)).attribute(), 0);
        assert_eq!(reuse.id(Val(9)).attribute(), 1);
        assert_eq!(reuse.id(Val(5)).attribute(), -1);
    }

    /// 🎯 056/384 — AN ORIGIN'S ABSORPTION FLAG IS ITS OWN, and `false` is an answer.
    #[test]
    fn an_absorption_flag_is_read_back_as_written() {
        let reuse = table(&[
            (
                Val(4),
                OperandTag {
                    id: DataId::Assigned(DataOriginId(0)),
                    absorbed: false,
                },
            ),
            (
                Val(9),
                OperandTag {
                    id: DataId::Assigned(DataOriginId(1)),
                    absorbed: true,
                },
            ),
        ]);
        assert_eq!(reuse.absorbtion_flag(Val(4)), Some(false));
        assert_eq!(reuse.absorbtion_flag(Val(9)), Some(true));
    }

    /// 🎯 056/384 — ⛔ AND AN UNREGISTERED OP HAS **NO** FLAG, WHICH IS NOT `false`.
    ///
    /// This is the one place the two getters part company: `getId` invents -1 while
    /// `getAbsorbtionFlag` returns `std::nullopt`, and both callers gate on `has_value()` before
    /// they read the flag (`VectorChainToSentientPESFP.cpp:1290`, `VectorChainToSentientPT.cpp:899`),
    /// so an op that is not a data origin at all takes neither branch.
    #[test]
    fn an_unregistered_op_has_no_absorption_flag_at_all() {
        let reuse = table(&[(
            Val(4),
            OperandTag {
                id: DataId::Assigned(DataOriginId(0)),
                absorbed: false,
            },
        )]);
        assert_eq!(reuse.absorbtion_flag(Val(7)), None);
        assert_ne!(
            reuse.absorbtion_flag(Val(7)),
            Some(false),
            "not registered and registered-but-unabsorbed are different answers"
        );
    }

    /// 🎯 055,056/384 — AND THE `OperandTag` DEFAULT IS THE C++ MEMBER INITIALISER'S: -1, false.
    #[test]
    fn an_untouched_tag_is_minus_one_and_unabsorbed() {
        let tag = OperandTag::default();
        assert_eq!(tag.id.attribute(), -1);
        assert!(!tag.absorbed);
    }
}

