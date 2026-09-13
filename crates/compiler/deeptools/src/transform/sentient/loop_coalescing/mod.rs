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

// ⛔ NOTHING IN THIS FILE HAS A CALLER YET and CI runs clippy with `-D warnings`. e445 consumes
// `split_bounds` and `is_all_less_than_max`, but `get_candidate_loops`, [`MaxLoops`] and
// [`ProcessedLoops::mark`]/[`ProcessedLoops::holds`] belong to `e563_runOnOperation`, which is what
// walks the unit and calls the two of them. ⭐ REMOVE THIS WITH e563: that is the unit that wires the
// pass up, and only then is an unused item here a real defect again.
#![allow(dead_code)]

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, Val, sentient, uniform, use_count,
};
use crate::transform::sentient::ForRef;
use crate::transform::sentient::utils::{ConstKind, InBlock, is_constant};

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
    /// `std::vector<int> prime_pool = {11, 7, 5, 3, 2};` (`:387`), in the reference's order — largest
    /// first, so each step takes the biggest available bite.
    const POOL: [Self; 5] = [Self(11), Self(7), Self(5), Self(3), Self(2)];
}

/// What [`get_largest_divisor`] found.
///
/// ⛔ `DT_ERROR("No valid prime factor but input still too large!")` (`:400`) IS THE SECOND ARM: the
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

    /// `sentient.scalar_constant` naming `result`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.for` carrying exactly one value.
    fn sentient_for(iv: Val, bound: Val, carried: sentient::Carried, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: vec![carried],
            dbg_name: None,
            body,
        })
    }

    /// One carried value, unassigned.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// A two-deep perfect nest is a band of two, innermost last; a non-constant outer bound is a band
    /// of NONE, because the dynamic-loop refusal (`:89-91`) precedes the push.
    #[test]
    fn e311_collects_the_perfect_nest_and_refuses_a_dynamic_outer_bound() {
        let inner = sentient_for(
            Val(20),
            Val(2),
            carried(Val(11), Val(21), Val(22)),
            vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(21)],
            })],
        );
        let outer = sentient_for(
            Val(10),
            Val(1),
            carried(Val(0), Val(11), Val(12)),
            vec![
                inner,
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(22)],
                }),
            ],
        );
        let scope = vec![constant(Val(1), 8), constant(Val(2), 4), outer];
        let root = &scope[2];
        assert_eq!(
            get_candidate_loops(root, &scope, MaxLoops::UNLIMITED),
            [ForRef(Val(10)), ForRef(Val(20))]
        );
        assert_eq!(get_candidate_loops(root, &scope, MaxLoops(1)), [ForRef(Val(10))]);

        let dynamic = vec![constant(Val(2), 4), scope[2].clone()];
        assert!(get_candidate_loops(&dynamic[1], &dynamic, MaxLoops::UNLIMITED).is_empty());
    }

    /// `coalesceLoops`' own vectors (`:246-251`): a bound over one LCCR splits into an exact pair, one
    /// under it keeps the pre-filled cofactor of `1`.
    #[test]
    fn e312_splits_only_the_bounds_over_one_lccr() {
        let split = split_bounds(&[TripCount(65536 * 3), TripCount(7)]);
        assert_eq!(split[1], SplitBound {
            fitting: TripCount(7),
            cofactor: TripCount(1)
        });
        assert!(split[0].fitting.0 <= TripLimit::LCCR.0);
        assert_eq!(split[0].fitting.0 * split[0].cofactor.0, 65536 * 3);
    }

    /// e445 — the reference's own two-loop case: the coalesced bound is the product of the band's,
    /// minted at the head of the preamble, and the innermost body moves up in place of the loop below.
    #[test]
    fn e445_fuses_a_band_of_two_into_its_outermost_loop() {
        let inner = sentient_for(
            Val(20),
            Val(2),
            carried(Val(11), Val(21), Val(22)),
            vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(21)],
            })],
        );
        let outer = sentient_for(
            Val(10),
            Val(1),
            carried(Val(0), Val(11), Val(12)),
            vec![
                inner,
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(22)],
                }),
            ],
        );
        let mut block = vec![constant(Val(1), 8), constant(Val(2), 4), outer];
        let mut preamble = Vec::new();
        let mut vals = Values::default();
        for _ in 0..30 {
            let _ = vals.mint();
        }
        let mut processed = ProcessedLoops::default();
        processed.mark(ForRef(Val(10)));
        processed.mark(ForRef(Val(20)));

        coalesce_loops(
            &[ForRef(Val(10)), ForRef(Val(20))],
            &mut Builders {
                preamble: &mut preamble,
                const_at: InBlock(0),
                block: &mut block,
                at: InBlock(2),
                vals: &mut vals,
            },
            &mut processed,
        );

        // The mark comes off every loop of the band (`:154`).
        assert!(!processed.holds(ForRef(Val(10))));
        assert!(!processed.holds(ForRef(Val(20))));
        // `8 * 4`, and the const builder writes to the preamble, not beside the band.
        assert_eq!(preamble, vec![constant(Val(30), 32)]);
        // The inner loop is gone, its yield operand answers the outer one, and the iter arg it read
        // is now the outermost's.
        assert_eq!(
            block[2],
            sentient_for(
                Val(10),
                Val(30),
                carried(Val(0), Val(11), Val(12)),
                vec![Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(11)],
                })]
            )
        );
    }

    /// e445 — a product one LCCR cannot hold splits across the outer TWO loops and returns there, so a
    /// band of exactly two keeps both, and the outermost takes the COFACTOR.
    #[test]
    fn e445_splits_a_product_too_wide_for_one_lccr_and_leaves_the_band_standing() {
        let body = vec![Op::Sentient(sentient::Op::Yield {
            results: vec![Val(21)],
        })];
        let inner = sentient_for(Val(20), Val(2), carried(Val(11), Val(21), Val(22)), body);
        let outer = sentient_for(
            Val(10),
            Val(1),
            carried(Val(0), Val(11), Val(12)),
            vec![
                inner.clone(),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(22)],
                }),
            ],
        );
        let mut block = vec![constant(Val(1), 65536), constant(Val(2), 3), outer];
        let mut preamble = Vec::new();
        let mut vals = Values::default();
        for _ in 0..30 {
            let _ = vals.mint();
        }

        coalesce_loops(
            &[ForRef(Val(10)), ForRef(Val(20))],
            &mut Builders {
                preamble: &mut preamble,
                const_at: InBlock(0),
                block: &mut block,
                at: InBlock(2),
                vals: &mut vals,
            },
            &mut ProcessedLoops::default(),
        );

        // `65536 * 3` is `6 * 32768`: the remainder is minted FIRST and goes to the outermost loop.
        assert_eq!(
            preamble,
            vec![constant(Val(30), 6), constant(Val(31), 32768)]
        );
        let Op::Sentient(sentient::Op::For { bound, body, .. }) = &block[2] else {
            panic!("the band's outermost loop is still a sentient.for")
        };
        assert_eq!(*bound, Val(30));
        // ⭐ AND THE NEST IS STILL TWO DEEP — the split consumed the second loop, so there was none
        // left to fuse into.
        assert_eq!(
            body[0],
            sentient_for(
                Val(20),
                Val(31),
                carried(Val(11), Val(21), Val(22)),
                vec![Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(21)],
                })]
            )
        );
    }
}

/// HOW MANY LOOPS OF A BAND TO COLLECT — the `max_loops` parameter (`:78`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaxLoops(pub usize);

impl MaxLoops {
    /// `std::numeric_limits<unsigned>::max()` — the default, and what the one in-scope caller
    /// (`:317`) takes: collect the whole band.
    pub(crate) const UNLIMITED: Self = Self(usize::MAX);
}

/// The fields of one `sentient.for` the band walk reads — the parent pointers this island does not
/// keep, supplied per step.
struct ForView<'a> {
    iv: Val,
    bound: Val,
    carried: &'a [sentient::Carried],
    body: &'a [Op],
    /// `dataflow::getDbgNameAttr(op)` — what `coalesceLoops` builds the band's merged name from
    /// (`:275-281`), and `None` for a loop carrying no `dbgName`.
    dbg_name: &'a Option<String>,
}

/// `dyn_cast<sentient::ForOp>(op)`.
fn for_view(op: &Op) -> Option<ForView<'_>> {
    let Op::Sentient(sentient::Op::For {
        iv,
        bound,
        carried,
        body,
        dbg_name,
        ..
    }) = op
    else {
        return None;
    };
    Some(ForView {
        iv: *iv,
        bound: *bound,
        carried,
        body,
        dbg_name,
    })
}

/// Replaces: e311_getCandidateLoops
///
/// The perfectly nested band at `root`, outermost first: each loop's bound is constant, its induction
/// variable unread, and its body one `sentient.for` plus the yield that returns that loop's results.
///
/// ⛔ TRAP: THE PUSH COMES BEFORE THE BODY CHECKS (`:99-102`), so the band's LAST entry is the loop
/// that FAILED them — an unnested loop still yields a band of one, and it is the caller's
/// `loops.size() > 1` (`:325`) that rejects it.
///
/// ⛔ TRAP: `hasOneUse() || use_empty()` on a region iter arg (`:126-128`) is `use_count(..) <= 1`,
/// ONE PER USE — the single permitted use is the inner loop's matching iter operand, proved just above.
pub(crate) fn get_candidate_loops(root: &Op, scope: &[Op], max_loops: MaxLoops) -> Vec<ForRef> {
    let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
    let mut band = Vec::new();
    let Some(mut root) = for_view(root) else {
        return band;
    };
    // `int n_loop_carried_args = root_ForOp.getNumRegionIterArgs();` (`:79`).
    let n_loop_carried_args = root.carried.len();
    for _ in 0..max_loops.0 {
        // The region-argument count `1 + n` and `getNumRegionIterArgs() != n` (`:83-87`) are ONE
        // question at this rung: the body's arguments ARE `[iv, carried…]`.
        if use_count(root.iv, root.body) != 0 || root.carried.len() != n_loop_carried_args {
            return band;
        }
        if !is_constant(root.bound, ConstKind::ScalarConstant, defs) {
            return band;
        }
        band.push(ForRef(root.iv));
        if root.body.len() != 2 {
            return band;
        }
        let Some(Op::Sentient(sentient::Op::Yield { results: yielded })) = root.body.last() else {
            return band;
        };
        let Some(inner) = root.body.first().and_then(for_view) else {
            return band;
        };
        if inner.carried.len() != n_loop_carried_args {
            return band;
        }
        for j in 0..n_loop_carried_args {
            if inner.carried[j].init != root.carried[j].arg
                || yielded.get(j) != Some(&inner.carried[j].result)
                || use_count(root.carried[j].arg, root.body) > 1
            {
                return band;
            }
        }
        root = inner;
    }
    band
}

/// ONE BOUND'S SPLIT — `new_bounds0[i]` and `new_bounds1[i]`, the two loops `coalesceLoops` makes of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitBound {
    /// `new_bounds0[i]` — a divisor of the bound that fits one LCCR.
    pub(crate) fitting: TripCount,
    /// `new_bounds1[i]` — the exact cofactor.
    pub(crate) cofactor: TripCount,
}

/// Replaces: e312_splitBounds
///
/// Splits every bound wider than one LCCR into a divisor that fits and its exact cofactor.
///
/// ⛔ TRAP: THE `else` BRANCH NEVER WRITES `new_bounds1[i]` (`:423`), so the cofactor there is the
/// `1` the caller pre-filled the vector with (`:246-251`) — an unsplit bound coalesces as `n * 1`,
/// not as `n * 0`.
pub(crate) fn split_bounds(new_bounds: &[TripCount]) -> Vec<SplitBound> {
    new_bounds
        .iter()
        .map(|bound| {
            if bound.0 <= TripLimit::LCCR.0 {
                return SplitBound {
                    fitting: *bound,
                    cofactor: TripCount(1),
                };
            }
            match get_largest_divisor(*bound, TripLimit::LCCR) {
                LargestDivisor::Fits(fitting) => SplitBound {
                    fitting,
                    cofactor: TripCount(bound.0 / fitting.0),
                },
                LargestDivisor::NoSmallPrimeFactor(_) => {
                    todo!("DT_ERROR(\"No valid prime factor but input still too large!\") (`:400`)")
                }
            }
        })
        .collect()
}

/// THE `"processed"` MARK — `#define PROCESSED "processed"` (`:29`), which `runOnOperation` stamps on
/// every loop of a candidate band (`:326-327`) and `coalesceLoops` takes off again (`:154`).
///
/// ⛔ NOT AN ISLAND FIELD: the attribute is written and erased inside this one pass, no other pass and
/// no printer reads it, so it is this pass's own bookkeeping and stays in the pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProcessedLoops(Vec<ForRef>);

impl ProcessedLoops {
    /// `for_op->setAttr(PROCESSED, ..)` (`:327`).
    pub(crate) fn mark(&mut self, for_op: ForRef) {
        if !self.holds(for_op) {
            self.0.push(for_op);
        }
    }

    /// `op->hasAttr(PROCESSED)` (`:323`).
    #[must_use]
    pub(crate) fn holds(&self, for_op: ForRef) -> bool {
        self.0.contains(&for_op)
    }

    /// `loop->removeAttr(PROCESSED)` (`:154`).
    pub(crate) fn remove(&mut self, for_op: ForRef) {
        self.0.retain(|held| *held != for_op);
    }
}

/// THE TWO INSERTION POINTS `coalesceLoops` BUILDS WITH — `const_builder`, at the head of the block
/// that holds the `dataflow.program_unit` (`:316-317`), and `builder(outermost)`, immediately before
/// the band's outermost loop (`:145`).
///
/// ⛔ BOTH ARE CURSORS AND NOT FIXED INDICES. MLIR's insertion point sits after the op it just made,
/// so consecutive `create`s land in call order and `const_builder` keeps its place ACROSS bands — one
/// builder serves the whole unit (`:316`). And an insert before the outermost loop MOVES that loop,
/// which is why [`Builders::at`] is maintained rather than passed again.
pub(crate) struct Builders<'a> {
    /// The block holding the `dataflow.program_unit` — the program preamble.
    pub(crate) preamble: &'a mut Vec<Op>,
    /// `const_builder`'s point in it.
    pub(crate) const_at: InBlock,
    /// The block holding the band.
    pub(crate) block: &'a mut Vec<Op>,
    /// `builder`'s point in it — which is the outermost loop's own index.
    pub(crate) at: InBlock,
    /// Fresh value names for everything minted here.
    pub(crate) vals: &'a mut Values,
}

impl Builders<'_> {
    /// `sentient::ConstantOp::create(const_builder, loc, ty, value)`.
    fn constant(&mut self, value: TripCount, ty: ScalarTy) -> Val {
        let result = self.vals.mint();
        self.preamble.insert(
            self.const_at.0,
            Op::Sentient(sentient::Op::ScalarConstant {
                value: value.0,
                result,
                reg_locale: sentient::RegType::Imm,
                ty,
                is_symbol: false,
            }),
        );
        self.const_at.0 += 1;
        result
    }

    /// `dcc::uniform::utils::createQueryMapFromConstants` (`Dialect/Uniform/Utils.cpp:367-380`) — one
    /// constant per bound at the const cursor, then the mapping and the query of it before the band.
    ///
    /// ⛔ INDEX-TYPED, NOT THE BAND'S `const_type`: both the constants and the map are
    /// `builder.getIndexType()` (`:373`, `Utils.cpp:355-362`), so a query-map bound comes out an index
    /// however the constant bounds were typed.
    fn query_map_from_constants(
        &mut self,
        query_map_key: Val,
        keys: &[Val],
        const_vals: &[TripCount],
    ) -> Val {
        if keys.len() != const_vals.len() {
            panic!(
                "DT_CHECK(keys.size() == const_vals.size()) \
                 (dcc/src/Dialect/Uniform/Utils.cpp:371)"
            )
        }
        let pairs = keys
            .iter()
            .zip(const_vals)
            .map(|(key, value)| (*key, self.constant(*value, ScalarTy::Index)))
            .collect();
        let map = self.vals.mint();
        let result = self.vals.mint();
        self.block.insert(
            self.at.0,
            Op::Uniform(uniform::Op::DefImmutableMapping { result: map, pairs }),
        );
        self.block.insert(
            self.at.0 + 1,
            Op::Uniform(uniform::Op::QueryMap {
                result,
                map,
                key: query_map_key,
            }),
        );
        // ⭐ THE OUTERMOST LOOP HAS MOVED DOWN BY THE TWO OPS JUST PUT IN FRONT OF IT.
        self.at.0 += 2;
        result
    }

    /// The band's loop `depth` levels in, mutably — the perfect nest e311 proved, descended.
    fn loop_at_mut(&mut self, depth: usize) -> &mut Op {
        let Some(outermost) = self.block.get_mut(self.at.0) else {
            panic!("`builder(outermost)`'s insertion point is the outermost loop's own index (`:145`)")
        };
        nested_loop_mut(outermost, depth)
    }

    /// The same loop, read-only.
    fn loop_at(&self, depth: usize) -> &Op {
        let Some(outermost) = self.block.get(self.at.0) else {
            panic!("`builder(outermost)`'s insertion point is the outermost loop's own index (`:145`)")
        };
        let mut op = outermost;
        for _ in 0..depth {
            let Some(view) = for_view(op) else {
                panic!("the band is a perfect nest of `sentient.for` ops (`:99-131`)")
            };
            let Some(inner) = view.body.first() else {
                panic!("a band loop's body opens with the loop below it (`:104-108`)")
            };
            op = inner;
        }
        op
    }
}

/// [`Builders::loop_at_mut`]'s descent, recursive so the reborrow chain is one per level.
fn nested_loop_mut(op: &mut Op, depth: usize) -> &mut Op {
    if depth == 0 {
        return op;
    }
    let Op::Sentient(sentient::Op::For { body, .. }) = op else {
        panic!("the band is a perfect nest of `sentient.for` ops (`:99-131`)")
    };
    let Some(inner) = body.first_mut() else {
        panic!("a band loop's body opens with the loop below it (`:104-108`)")
    };
    nested_loop_mut(inner, depth - 1)
}

/// `outermost.setBound(new_bound_op)`.
fn set_bound(op: &mut Op, bound: Val) {
    let Op::Sentient(sentient::Op::For { bound: at, .. }) = op else {
        panic!("only a `sentient.for` has a `$bound` to set (`:200`)")
    };
    *at = bound;
}

/// `outermost.setDbgName(new_dbg_name_attr)` (`:281`).
fn set_dbg_name(op: &mut Op, name: String) {
    let Op::Sentient(sentient::Op::For { dbg_name, .. }) = op else {
        panic!("only a `sentient.for` has a `dbgName` to set (`:281`)")
    };
    *dbg_name = Some(name);
}

/// WHAT THE BAND'S BOUNDS MULTIPLY OUT TO — the locals `coalesceLoops`' first loop fills (`:147-194`).
///
/// ⭐ ONE READ PASS BEFORE ANY MUTATION, because [`Definitions`] borrows the very blocks the rest of
/// the unit rewrites.
struct BandBounds {
    /// `new_bound` — the product of every `sentient.scalar_constant` bound.
    new_bound: TripCount,
    /// `const_type` — the type of the LAST constant bound seen (`:163`). `None` is the reference's
    /// default-constructed, null `mlir::Type` (`:148`), which only a band with no constant bound
    /// leaves behind.
    const_type: Option<ScalarTy>,
    /// `new_bounds` — one product per unit, EMPTY when no bound was a `uniform.query_map`.
    new_bounds: Vec<TripCount>,
    /// `bound_keys` — the unit list of the LAST query-map bound seen.
    bound_keys: Vec<Val>,
    /// `query_map_key` — that query map's `$key`.
    query_map_key: Option<Val>,
}

/// The band's loops, outermost first — `loops` descended through the nest e311 proved.
fn band_views<'a>(loops: &[ForRef], outermost: &'a Op) -> Vec<ForView<'a>> {
    let mut views = Vec::with_capacity(loops.len());
    let mut next = Some(outermost);
    for for_op in loops {
        let Some(view) = next.and_then(for_view) else {
            panic!("the band is a perfect nest of `sentient.for` ops (`:99-131`)")
        };
        if view.iv != for_op.0 {
            panic!("this band is not the nest rooted at the loop `builder` points before (`:145`)")
        }
        next = view.body.first();
        views.push(view);
    }
    views
}

/// `coalesceLoops`' first loop (`:153-194`) — everything it reads, before anything is written.
fn band_bounds(loops: &[ForRef], outermost: &Op, defs: Definitions<'_>) -> BandBounds {
    let mut found = BandBounds {
        new_bound: TripCount(1),
        const_type: None,
        new_bounds: Vec::new(),
        bound_keys: Vec::new(),
        query_map_key: None,
    };
    // `int num_of_const_bound = 1;` (`:150`) — 1 until a query map says how many units there are.
    let mut num_of_const_bound = 1;
    for view in band_views(loops, outermost) {
        if !is_constant(view.bound, ConstKind::ScalarConstant, defs) {
            panic!(
                "We do not yet support coalescing of loops with non-constant bound \
                 (`:158-162`, signalPassFailure)"
            )
        }
        match defs.of(view.bound) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, ty, .. })) => {
                found.new_bound.0 *= value;
                found.const_type = Some(*ty);
            }
            Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
                found.query_map_key = Some(*key);
                found.bound_keys = dialects::uniform_mapping_keys(*key, defs);
                let bound_values = dialects::uniform_mapping_values(*map, *key, defs);
                // The three `DT_CHECK`s (`:173-177`): the map answers for every key it was asked
                // about, it answers at all, and every query-map bound of the band has the same width.
                if found.bound_keys.len() != bound_values.len() || bound_values.is_empty() {
                    panic!("a bound query map answers for every one of its keys (`:173-175`)")
                }
                if num_of_const_bound != 1 && num_of_const_bound != bound_values.len() {
                    panic!("every query-map bound of a band covers the same units (`:176-177`)")
                }
                num_of_const_bound = bound_values.len();
                if found.new_bounds.is_empty() {
                    found.new_bounds = vec![TripCount(1); num_of_const_bound];
                }
                for (product, value) in found.new_bounds.iter_mut().zip(&bound_values) {
                    let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) =
                        defs.of(*value)
                    else {
                        panic!("All values of query map has to be constants (`:184-187`)")
                    };
                    product.0 *= value;
                }
            }
            // ⛔ NO `else` IN THE REFERENCE'S CHAIN (`:163-190`), and the refusal just above is why it
            // needs none: a bound `isConstant` answered for is one of these two ops.
            _ => {}
        }
    }
    // `if (new_bounds.size() != 0) for (auto &bound : new_bounds) bound *= new_bound;` (`:192-194`).
    for bound in &mut found.new_bounds {
        bound.0 *= found.new_bound.0;
    }
    found
}

/// Replaces: e445_coalesceLoops
///
/// Fuses a perfectly nested band into its outermost loop: that loop's bound becomes the product of the
/// band's, split across the outer TWO loops when one loop-control register cannot hold it, and the
/// innermost body moves up in place of the loop below.
///
/// ⛔ THE SPLIT PAIR IS ORDERED THE OTHER WAY IN THE TWO BRANCHES — a constant bound gives the outermost
/// the COFACTOR and the second the fitting divisor (`:211-217`), a query-map bound gives the outermost
/// the FITTING one (`:257-259`). ⭐ AND A SPLIT MAKES `loops[1]` THE NEW OUTERMOST (`:219-225`), so a
/// band of exactly two is already coalesced and returns there.
/// ⛔ AN UNNAMED LOOP ANYWHERE IN THE BAND LEAVES THE OUTERMOST'S OWN `dbgName` ALONE (`:279-281`).
pub(crate) fn coalesce_loops(
    loops: &[ForRef],
    builders: &mut Builders<'_>,
    processed: &mut ProcessedLoops,
) {
    // `DT_CHECK(loops.size() > 1)` (`:139`) — the caller returns on `loops.size() < 2` (`:325`).
    if loops.len() < 2 {
        panic!("coalesceLoops is given a band of at least two loops (`:139`, `:325`)")
    }
    for for_op in loops {
        processed.remove(*for_op);
    }

    let bounds = {
        let scopes: [&[Op]; 2] = [&*builders.block, &*builders.preamble];
        let defs = Definitions::from_innermost(&scopes);
        band_bounds(loops, builders.loop_at(0), defs)
    };

    let innermost = loops.len() - 1;
    let mut outermost = 0;
    let mut second = 1;

    if bounds.new_bounds.is_empty() {
        let Some(const_type) = bounds.const_type else {
            panic!("a band with no query-map bound has a constant one, so `const_type` is set (`:163`)")
        };
        // 2. Assign the newly calculated bound to the outermost loop (`:196-202`).
        if bounds.new_bound.0 <= TripLimit::LCCR.0 {
            let new_bound = builders.constant(bounds.new_bound, const_type);
            set_bound(builders.loop_at_mut(outermost), new_bound);
        } else {
            if bounds.new_bound.0 >= TripLimit::LCCR_SQUARED.0 {
                panic!("Currently we only support splitting into 2 loops (`:203-208`)")
            }
            let LargestDivisor::Fits(outer_bound) =
                get_largest_divisor(bounds.new_bound, TripLimit::LCCR)
            else {
                todo!("DT_ERROR(\"No valid prime factor but input still too large!\") (`:400`)")
            };
            let remainder = TripCount(bounds.new_bound.0 / outer_bound.0);
            let new_bound_outer = builders.constant(remainder, const_type);
            let new_bound_second = builders.constant(outer_bound, const_type);
            set_bound(builders.loop_at_mut(outermost), new_bound_outer);
            set_bound(builders.loop_at_mut(second), new_bound_second);
            outermost = second;
            if loops.len() > 2 {
                second = 2;
            } else {
                return;
            }
        }
    } else {
        let Some(query_map_key) = bounds.query_map_key else {
            panic!("a non-empty `new_bounds` was filled by a query-map bound, which has a key (`:167`)")
        };
        if is_all_less_than_max(&bounds.new_bounds, TripLimit::LCCR) {
            let new_bound = builders.query_map_from_constants(
                query_map_key,
                &bounds.bound_keys,
                &bounds.new_bounds,
            );
            set_bound(builders.loop_at_mut(outermost), new_bound);
        } else {
            if !is_all_less_than_max(&bounds.new_bounds, TripLimit::LCCR_SQUARED) {
                panic!("Currently we only support splitting into 2 loops (`:239-244`)")
            }
            let split = split_bounds(&bounds.new_bounds);
            let fitting: Vec<TripCount> = split.iter().map(|bound| bound.fitting).collect();
            let cofactors: Vec<TripCount> = split.iter().map(|bound| bound.cofactor).collect();
            let new_bound0 =
                builders.query_map_from_constants(query_map_key, &bounds.bound_keys, &fitting);
            let new_bound1 =
                builders.query_map_from_constants(query_map_key, &bounds.bound_keys, &cofactors);
            set_bound(builders.loop_at_mut(outermost), new_bound0);
            set_bound(builders.loop_at_mut(second), new_bound1);
            outermost = second;
            if loops.len() > 2 {
                second = 2;
            } else {
                return;
            }
        }
    }

    if outermost == innermost {
        return;
    }

    // `getNewDbgNameFromList("LC(", loop_op_list)` over the WHOLE original band (`:275-281`).
    let dbg_names: Vec<Option<String>> = band_views(loops, builders.loop_at(0))
        .iter()
        .map(|view| view.dbg_name.clone())
        .collect();
    let borrowed: Vec<Option<&str>> = dbg_names.iter().map(Option::as_deref).collect();
    if let Some((first, rest)) = borrowed.split_first()
        && let Some(name) = new_dbg_name_from_list(DbgNamePrefix::Lc, *first, rest)
    {
        set_dbg_name(builders.loop_at_mut(outermost), name);
    }

    // 3. Move the innermost's operations up, drop its terminator and the second-outermost loop
    // (`:283-299`).
    let outer_args: Vec<Val> = carried_of(builders.loop_at(outermost))
        .iter()
        .map(|carried| carried.arg)
        .collect();
    let inner_args: Vec<Val> = carried_of(builders.loop_at(innermost))
        .iter()
        .map(|carried| carried.arg)
        .collect();
    let second_results: Vec<Val> = carried_of(builders.loop_at(second))
        .iter()
        .map(|carried| carried.result)
        .collect();

    let (moved, ret_values) = {
        let Op::Sentient(sentient::Op::For { body, .. }) = builders.loop_at_mut(innermost) else {
            panic!("the band is a perfect nest of `sentient.for` ops (`:99-131`)")
        };
        for (of, with) in inner_args.iter().zip(&outer_args) {
            dialects::replace_all_uses_with(body, *of, *with);
        }
        let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
            panic!("a `sentient.for` body ends in the `sentient.yield` this reads (`:290`)")
        };
        let ret_values = results.clone();
        body.pop();
        (core::mem::take(body), ret_values)
    };

    let Op::Sentient(sentient::Op::For { body, .. }) = builders.loop_at_mut(outermost) else {
        panic!("the band is a perfect nest of `sentient.for` ops (`:99-131`)")
    };
    for (of, with) in second_results.iter().zip(&ret_values) {
        dialects::replace_all_uses_with(body, *of, *with);
    }
    // ⭐ THE SPLICE AND THE `second.erase()` IN ONE (`:294-299`): `second` is the head of this body —
    // that is what makes the nest perfect — so the innermost's ops go exactly where it was.
    body.splice(0..1, moved);
}

/// `for_op.getRegionIterArgs()` / `.getResults()`, whichever the caller reads.
fn carried_of(op: &Op) -> &[sentient::Carried] {
    let Op::Sentient(sentient::Op::For { carried, .. }) = op else {
        panic!("only a `sentient.for` carries iteration arguments (`:285-287`)")
    };
    carried
}

// crustify:todo: e563_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopCoalescing.cpp:302  (74 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e252_size, e311_getCandidateLoops, e445_coalesceLoops, e498_updateForOperation, e546_findAndReplaceRedundantIterArgsUsedInConditions
