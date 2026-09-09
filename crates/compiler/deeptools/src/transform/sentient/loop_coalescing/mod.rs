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

//! `LoopCoalescing.cpp` — 6 of the campaign's 656 units (dependency level(s) [0, 1, 2, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e071_getLargestDivisor` | 071 | 0 | 27 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:379` |
//! | `e072_isAllLessThanMax` | 072 | 0 | 6 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:408` |
//! | `e311_getCandidateLoops` | 311 | 1 | 56 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:76` |
//! | `e312_splitBounds` | 312 | 1 | 10 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:417` |
//! | `e445_coalesceLoops` | 445 | 2 | 165 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:135` |
//! | `e563_runOnOperation` | 563 | 4 | 74 | `dcc/src/Transform/Sentient/LoopCoalescing.cpp:302` |

// ⛔ NOTHING CALLS THESE TWO UNTIL `e312_splitBounds` / `e445_coalesceLoops` LAND, and CI runs clippy
// with `-D warnings`. ⭐ REMOVE THIS WITH e445.
#![allow(dead_code)]

/// A loop's trip count — `sentient.for`'s `$bound`, and what coalescing multiplies together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TripCount(pub i64);

/// THE WIDEST TRIP COUNT A LOOP-CONTROL REGISTER CAN HOLD — `limit`, and why coalescing splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TripLimit(pub i64);

impl TripLimit {
    /// `MAX_LCCR_VALUE` (`SentientOps.hpp:34`) — one LCCR, as `splitBounds` (`:421`) asks it.
    pub(crate) const LCCR: Self = Self(65535);

    /// `(int64_t)(MAX_LCCR_VALUE) * MAX_LCCR_VALUE` (`:204`) — what two LCCRs together can hold.
    pub(crate) const LCCR_SQUARED: Self = Self(Self::LCCR.0 * Self::LCCR.0);
}

/// A prime small enough to be worth trial-dividing by — `prime_pool`'s element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SmallPrime(i64);

impl SmallPrime {
    /// `std::vector<int> prime_pool = {11, 7, 5, 3, 2};` (`:386`), in the reference's order — largest
    /// first, so each step takes the biggest available bite.
    const POOL: [Self; 5] = [Self(11), Self(7), Self(5), Self(3), Self(2)];
}

/// What [`get_largest_divisor`] found.
///
/// ⛔ `DT_ERROR("No valid prime factor but input still too large!")` (`:399`) IS THE SECOND ARM: the
/// reference aborts, so the arm must be impossible to mistake for a usable trip count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LargestDivisor {
    /// A divisor of the input that fits the limit.
    Fits(TripCount),
    /// The input has no factor in [`SmallPrime::POOL`] and is still over the limit — the cofactor
    /// reached when the reference gives up. ⚠️ NOT a divisor that fits.
    NoSmallPrimeFactor(TripCount),
}

/// Replaces: e071_getLargestDivisor
///
/// Divide `input` by primes from [`SmallPrime::POOL`] until it fits `limit`, and answer with what is
/// left — always a divisor of the original, which is what makes `new_bounds[i] / new_bounds0[i]`
/// (`:422`) exact.
///
/// ⛔ TRAP: THE NAME PROMISES THE *LARGEST* DIVISOR AND THE BODY IS GREEDY, NOT MAXIMAL — `29 * 32`
/// with limit 100 reduces by 2 five times to 29, though 32 also fits. The reference's own comment
/// (`:384-385`) states the assumption: the loop is not a large prime.
pub(crate) fn get_largest_divisor(mut input: TripCount, limit: TripLimit) -> LargestDivisor {
    // `if (input <= limit) return input;` (`:380-382`).
    while input.0 > limit.0 {
        // `for (auto &p : prime_pool) if (input % p == 0) { input /= p; found = true; break; }`
        // (`:388-395`).
        let Some(p) = SmallPrime::POOL.iter().find(|p| input.0 % p.0 == 0) else {
            return LargestDivisor::NoSmallPrimeFactor(input);
        };
        input = TripCount(input.0 / p.0);
    }
    LargestDivisor::Fits(input)
}

/// Replaces: e072_isAllLessThanMax
///
/// Whether every coalesced bound still fits `max`.
///
/// ⛔ TRAP: THE NAME SAYS "LESS THAN" AND THE BODY TESTS `bound > max` (`:411`) — a bound EQUAL to the
/// max passes. That is the right reading: `max` is a representable value, not an exclusive bound.
pub(crate) fn is_all_less_than_max(new_bounds: &[TripCount], max: TripLimit) -> bool {
    new_bounds.iter().all(|bound| bound.0 <= max.0)
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// `splitBounds`' own use (`:421-422`): a bound over one LCCR reduces to a divisor that fits, and
    /// the cofactor is exact.
    #[test]
    fn largest_divisor_is_a_divisor_that_fits() {
        let bound = TripCount(65536 * 3);
        let LargestDivisor::Fits(divisor) = get_largest_divisor(bound, TripLimit::LCCR) else {
            panic!("65536 * 3 is a power of two times three");
        };
        assert!(divisor.0 <= TripLimit::LCCR.0);
        assert_eq!(bound.0 % divisor.0, 0);
    }

    /// The `DT_ERROR` arm: a prime beyond the pool cannot be reduced at all.
    #[test]
    fn a_large_prime_has_no_small_prime_factor() {
        assert_eq!(
            get_largest_divisor(TripCount(65537), TripLimit::LCCR),
            LargestDivisor::NoSmallPrimeFactor(TripCount(65537))
        );
    }

    /// e072's boundary: equal to the max passes, one over does not.
    #[test]
    fn all_less_than_max_admits_the_max_itself() {
        assert!(is_all_less_than_max(
            &[TripCount(1), TripCount(65535)],
            TripLimit::LCCR
        ));
        assert!(!is_all_less_than_max(&[TripCount(65536)], TripLimit::LCCR));
    }
}

// crustify:todo: e311_getCandidateLoops
//   authority : dcc/src/Transform/Sentient/LoopCoalescing.cpp:76  (56 body lines, level 1)
//   original  : void getCandidateLoops( SmallVectorImpl<sentient::ForOp> &band_ForOps, sentient::ForOp root_ForOp, unsigned max_loops = std::numeric_limits<unsigned>::max())
//   calls     : e252_size

// crustify:todo: e312_splitBounds
//   authority : dcc/src/Transform/Sentient/LoopCoalescing.cpp:417  (10 body lines, level 1)
//   original  : void LoopCoalescingPass::splitBounds(std::vector<int64_t> &new_bounds, std::vector<int64_t> &new_bounds0, std::vector<int64_t> &new_bounds1)
//   calls     : e071_getLargestDivisor, e252_size

// crustify:todo: e445_coalesceLoops
//   authority : dcc/src/Transform/Sentient/LoopCoalescing.cpp:135  (165 body lines, level 2)
//   original  : void coalesceLoops(SmallVector<sentient::ForOp, 4> loops, OpBuilder &const_builder)
//   calls     : e071_getLargestDivisor, e072_isAllLessThanMax, e252_size, e312_splitBounds

// crustify:todo: e563_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopCoalescing.cpp:302  (74 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e252_size, e311_getCandidateLoops, e445_coalesceLoops, e498_updateForOperation, e546_findAndReplaceRedundantIterArgsUsedInConditions
