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

//! `Helper.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [0]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e089_getMaskValueForPT` | 089/384 | 196 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17` |

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::MaskedColumns;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::{
    AffineExpr, BoundType, GenericComp, IntegerSet, ScalarTy, Vector,
};
use crate::islands::sentient::dialects::{
    self as sen, Definitions, Val, arith, sentient, vectorchain,
};

/// THE COMPONENT THIS FILE'S LOWERING IS FOR — a witness that the unit is a PT.
///
/// ```cpp
/// DT_CHECK_MSG(comp == PT, "function only checks mask value for PT unit");
/// ```
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:20`)
///
/// # ⛔⛔ AN ABORT ON ITS FIRST LINE IS A PARAMETER THAT SHOULD NOT EXIST
///
/// `getMaskValueForPT` takes a `SenComponents comp` and then refuses to run for any value but one.
/// Everything after that line — the slice arithmetic, the two mask branches, the accepted affine
/// set — is stated for the PT row and nothing else. So the check is not a validation the function
/// performs; it is a precondition on being called at all, and this type is that precondition written
/// down. ⭐ A caller classifies its component ONCE, here, and the port below has no component branch
/// to take.
///
/// ⛔ ZERO-SIZED ON PURPOSE. It carries no component field: there is one value it could hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtUnit;

impl PtUnit {
    /// `comp == PT` — `None` for every other component.
    ///
    /// ⭐ THE `_` ARM IS THIS TYPE'S WHOLE CONTENT, not a gap in a walk: fourteen components exist
    /// ([`GenericComp`]) and exactly one of them is this one.
    #[must_use]
    pub const fn of(comp: GenericComp) -> Option<PtUnit> {
        match comp {
            GenericComp::Pt => Some(PtUnit),
            _ => None,
        }
    }
}

/// HOW MANY LANES A PT ROW HOLDS, AND HOW MANY OF THEM ONE SLICE OWNS.
///
/// ```cpp
/// // The number of lanes should be equally divisible into slices.
/// DT_CHECK_MSG(op_num_elems % sys_def.numSlicesPerStick == 0,
///              "expecting equal elements per slice from op result");
/// int num_lanes_in_slice = op_num_elems / sys_def.numSlicesPerStick;
/// ```
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:53-56`)
///
/// # ⛔⛔ THE DIVISIBILITY IS A **BUILD-TIME GUARD**, NOT AN ASSERT
///
/// The reference aborts on an indivisible lane count and then divides. Both facts belong to the same
/// value, so this type holds the quotient it computed: a `PtLanes` cannot exist for a vector whose
/// element count does not divide into slices, and [`PtLanes::lanes_in_slice`] therefore needs no
/// second division and no second check. ⭐ THAT IS THE CRATE'S RULE — the invariant is the type, and
/// there is no run-time refusal left to write.
///
/// ⛔ AND IT IS WHY `num_lanes_in_slice` CANNOT BE CONFUSED WITH THE ROW WIDTH. Both appear in one
/// expression of the accepted affine set — `d0 + s0 * num_lanes_in_slice - op_num_elems` (`:199`) —
/// and swapping them accepts a set no fixture writes and rejects the one every fixture does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtLanes {
    /// `op_num_elems` — the lanes in the row.
    elems: u32,
    /// `num_lanes_in_slice` — already divided.
    lanes_in_slice: u32,
}

impl PtLanes {
    /// THE LANES OF ONE MASKED RESULT — `None` when they do not divide into slices.
    ///
    /// ⭐ `sys_def.numSlicesPerStick` IS [`Arch::SLICES_PER_STICK`], 8 (`sysdef.cpp:229`) — a
    /// constant of the arch being compiled for, not a value read out of a system-definition object.
    #[must_use]
    pub fn of<A: Arch>(ty: Vector) -> Option<PtLanes> {
        // `unsigned op_num_elems = op_result_type.getNumElements();` — a row this crate can address.
        let elems = u32::try_from(ty.len).ok()?;
        if elems % A::SLICES_PER_STICK != 0 {
            return None;
        }
        Some(PtLanes {
            elems,
            lanes_in_slice: elems / A::SLICES_PER_STICK,
        })
    }

    /// `op_num_elems`.
    #[must_use]
    pub const fn elems(self) -> u32 {
        self.elems
    }

    /// `num_lanes_in_slice`.
    #[must_use]
    pub const fn lanes_in_slice(self) -> u32 {
        self.lanes_in_slice
    }
}

/// AN ENCLOSING `sentient.for`, AS THE MASK CHECK NEEDS TO SEE IT.
///
/// ⭐ THREE FACTS AND NO MORE: the bound the mask's subtraction must read, the induction variable it
/// must subtract, and the arguments the loop binds — which is what makes "a block argument of a loop
/// that is not its induction variable" a distinguishable case (see [`MaskParameterOwner`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnclosingLoop<'a> {
    /// `rhs_parent.getBound()`.
    pub bound: Val,
    /// `rhs_parent.getInductionVar()`.
    pub iv: Val,
    /// `getRegionIterArgs()` — the carried arguments, which are region arguments too.
    carried: &'a [sentient::Carried],
}

impl<'a> EnclosingLoop<'a> {
    /// A LOOP — `None` for any other op, which is `dyn_cast<sentient::ForOp>` failing.
    #[must_use]
    pub fn of(op: &'a sen::Op) -> Option<EnclosingLoop<'a>> {
        match op {
            sen::Op::Sentient(sentient::Op::For {
                iv,
                bound,
                carried,
                ..
            }) => Some(EnclosingLoop {
                bound: *bound,
                iv: *iv,
                carried,
            }),
            // ⭐ A WITNESS CONSTRUCTOR, not a walk: see [`PtUnit::of`].
            _ => None,
        }
    }

    /// Whether this loop's region binds a value as an argument — `arg.getOwner()->getParentOp() == this`.
    #[must_use]
    pub fn binds(&self, val: Val) -> bool {
        val == self.iv || self.carried.iter().any(|carried| carried.arg == val)
    }
}

/// WHAT DEFINES THE RIGHT-HAND SIDE OF A DYNAMIC MASK'S SUBTRACTION — the three outcomes the
/// reference distinguishes, and it treats them differently.
///
/// # ⛔⛔ TWO REFUSALS AND ONE ABORT, WHICH A `bool` WOULD FLATTEN
///
/// ```cpp
/// auto rhs_block_arg = dyn_cast<BlockArgument>(rhs);
/// if (!rhs_block_arg) {
///   operand->emitOpError("non-constant PT mask value should be loop iterator");
///   return std::nullopt;
/// }
/// auto rhs_parent =
///     dyn_cast<sentient::ForOp>(rhs_block_arg.getOwner()->getParentOp());
/// DT_CHECK_MSG(rhs_parent, "Parent operation of RHS of operation making up mask should be a "
///                          "sentient::ForOp");
/// if (lhs != rhs_parent.getBound() || rhs != rhs_parent.getInductionVar()) { … return std::nullopt; }
/// ```
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:145-163`)
///
/// A value produced by an op is a REFUSAL, a region argument of something that is not a loop is an
/// ABORT, and a region argument of a loop is the accepted shape whose two operands are then compared.
/// Collapsing the first two into one answer would either turn a stop into a diagnostic or a
/// diagnostic into a stop.
///
/// # ⭐ THE OWNER IS SUPPLIED, BECAUSE THIS ISLAND'S OPS HAVE NO PARENT POINTERS
///
/// `rhs_block_arg.getOwner()->getParentOp()` walks UP; the island's ops are a tree of `Vec`s. That is
/// the *mechanism for reaching an operand* the campaign brief allows a port to be given instead — so
/// [`MaskParameterOwner::of`] takes the enclosing loops the caller is already inside, innermost
/// first, exactly as [`Definitions`] takes the enclosing regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskParameterOwner<'a> {
    /// `dyn_cast<BlockArgument>(rhs)` FAILED — some op defines it. A refusal.
    Defined,
    /// A region argument, but its region's parent is not a `sentient.for`. The `DT_CHECK_MSG` abort.
    ForeignRegion,
    /// A region argument of a `sentient.for` — the shape the two operand comparisons then judge.
    Loop(EnclosingLoop<'a>),
}

impl<'a> MaskParameterOwner<'a> {
    /// WHICH OF THE THREE `rhs` IS, given where the caller is standing.
    ///
    /// ⛔ AN UNLISTED REGION READS AS [`MaskParameterOwner::ForeignRegion`], WHICH IS THE HONEST
    /// ANSWER: a value that no listed region defines and no enclosing loop binds is not a loop
    /// iterator of this nest, and that is precisely what the reference's `DT_CHECK_MSG` stops on.
    #[must_use]
    pub fn of(
        rhs: Val,
        definitions: Definitions<'_>,
        loops: &'a [EnclosingLoop<'a>],
    ) -> MaskParameterOwner<'a> {
        if definitions.of(rhs).is_some() {
            return MaskParameterOwner::Defined;
        }
        match loops.iter().find(|enclosing| enclosing.binds(rhs)) {
            Some(enclosing) => MaskParameterOwner::Loop(*enclosing),
            None => MaskParameterOwner::ForeignRegion,
        }
    }
}

/// THE MASK VALUE A PT COMPUTE IS GIVEN — the `Value` the reference returns, with the op that had to
/// exist for it.
///
/// # ⛔⛔ THE STATIC ARM **EMITS**, AND A BARE `Val` WOULD LOSE THE OP
///
/// ```cpp
/// // Create a constant op for the mask value. This is a dummy Value.
/// OpBuilder builder(op);
/// return sentient::ConstantOp::create(builder, op->getLoc(),
///                                     builder.getIndexType(), masked_columns);
/// ```
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:112-116`) — an `OpBuilder`
/// constructed at the masked op INSERTS the constant into the module before that op, and the returned
/// `Value` is its result. A port that returned only the value would describe a program in which
/// nothing defines it, which is the campaign brief's *"a predicate is not a port"* exactly.
///
/// ⭐ AND THE COLUMNS ARE KEPT BESIDE IT, in the type the mask tree already uses
/// ([`MaskedColumns`]) — `updateLoopMaskTreeForConstantMask` needs the count, and recovering it by
/// reading the literal back out of the emitted op would be a second derivation of one number.
///
/// ⭐ AND THE TWO VARIANTS ARE DELIBERATELY LOPSIDED. The static arm carries a whole op because it
/// EMITS one; the dynamic arm carries a value because it emits nothing. Boxing the op to even out the
/// two sizes would put an allocation on the path every constant PT mask takes, to describe a
/// difference that IS the function's answer.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskValue {
    /// A CONSTANT MASK — the *"dummy Value"* and the op that binds it.
    Constant {
        /// The `sentient.scalar_constant` to insert before the masked op.
        op: sen::Op,
        /// What it binds — the returned `Value`.
        value: Val,
        /// `masked_columns`, which the mask tree takes as its `start_val`.
        columns: MaskedColumns,
    },
    /// A DYNAMIC MASK — `return rhs`, the enclosing loop's induction variable. ⭐ NOTHING IS EMITTED:
    /// the op that defines it is the loop that was already there.
    LoopIterator(Val),
}

/// ONE ACCEPTED MASK CONSTRAINT AS ITS LINEAR FORM — `c_d * d0 + c_s * s0 + k`.
///
/// ⛔⛔ MLIR COMPARES CANONICAL FORMS AND THIS ISLAND DOES NOT CANONICALISE.
/// `checkAffineSetConstraints` tests `constraint == constraintA` on `AffineExpr`s
/// (`Helper.cpp:184-186`), and in MLIR that is a uniqued-canonical-form comparison: `d0 + s0*8 - 64`
/// and `-64 + s0*8 + d0` are ONE object there. [`AffineExpr`] here is a structural tree whose own
/// note says the builders are the only place it normalises, so a structural `==` would reject sets
/// the reference accepts purely on the order the terms were written in.
///
/// ⭐ SO THE COMPARISON IS ON COEFFICIENTS, which is what the two accepted constraints are: both are
/// linear in `d0` and `s0` with an integer offset, and a set is accepted only if its two rows match
/// this pair. ⛔ A `mod`, a `floordiv`, a product of two variables or any dimension past `d0`
/// yields no linear form at all and is therefore *"unexpected constraint detected in PT mask_set"* —
/// which is stricter than a term-blind comparison and never looser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LinearForm {
    /// The coefficient of `d0`.
    d0: i64,
    /// The coefficient of `s0`.
    s0: i64,
    /// The constant term.
    constant: i64,
}

impl LinearForm {
    /// `c_d * d0 + c_s * s0 + k`, or `None` for an expression that is not of that shape.
    fn of(expr: &AffineExpr) -> Option<LinearForm> {
        match expr {
            AffineExpr::Dim(0) => Some(LinearForm {
                d0: 1,
                s0: 0,
                constant: 0,
            }),
            AffineExpr::Sym(0) => Some(LinearForm {
                d0: 0,
                s0: 1,
                constant: 0,
            }),
            AffineExpr::Const(k) => Some(LinearForm {
                d0: 0,
                s0: 0,
                constant: *k,
            }),
            AffineExpr::Add(a, b) => {
                let (a, b) = (LinearForm::of(a)?, LinearForm::of(b)?);
                Some(LinearForm {
                    d0: a.d0 + b.d0,
                    s0: a.s0 + b.s0,
                    constant: a.constant + b.constant,
                })
            }
            AffineExpr::Mul(a, b) => {
                let (a, b) = (LinearForm::of(a)?, LinearForm::of(b)?);
                // ⛔ LINEAR MEANS ONE SIDE IS A CONSTANT. `d0 * s0` has no linear form.
                match (a.d0 == 0 && a.s0 == 0, b.d0 == 0 && b.s0 == 0) {
                    (true, _) => Some(LinearForm {
                        d0: b.d0 * a.constant,
                        s0: b.s0 * a.constant,
                        constant: b.constant * a.constant,
                    }),
                    (false, true) => Some(LinearForm {
                        d0: a.d0 * b.constant,
                        s0: a.s0 * b.constant,
                        constant: a.constant * b.constant,
                    }),
                    (false, false) => None,
                }
            }
            // A dimension or symbol past position 0, a `mod` or a `floordiv`: not this shape.
            AffineExpr::Dim(_)
            | AffineExpr::Sym(_)
            | AffineExpr::Mod(_, _)
            | AffineExpr::FloorDiv(_, _) => None,
        }
    }
}

/// THE ONE MASK SET A DYNAMIC PT MASK MAY HAVE — `checkAffineSetConstraints`.
///
/// ```cpp
/// auto checkAffineSetConstraints =
///     [](IntegerSet& mask_set, AffineExpr& constraintA, AffineExpr& constraintB,
///        Operation* mask_operand) -> LogicalResult {
///   auto num_constraints = mask_set.getNumConstraints();
///   if (num_constraints != 2)
///     return mask_operand->emitOpError("PT mask set has incorrect number of constraints");
///   bool found_constraintA = false, found_constraintB = false;
///   for (int i = 0; i < num_constraints; ++i) {
///     auto constraint = mask_set.getConstraint(i);
///     if (constraint == constraintA)        found_constraintA = true;
///     else if (constraint == constraintB)   found_constraintB = true;
///     else return mask_operand->emitOpError("unexpected constraint detected in PT mask_set");
///   }
///   if (!found_constraintA || !found_constraintB)
///     return mask_operand->emitOpError("expected constraints for PT mask_set not found");
///   return LogicalResult::success();
/// };
/// …
/// AffineExpr symbol_constraint = d0 + s0 * num_lanes_in_slice - op_num_elems;
/// AffineExpr non_symbol_constraint = -d0 + (op_num_elems - 1);
/// ```
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:174-205`)
///
/// # ⭐⭐ IT IS ORDER-INSENSITIVE, AND THE FIXTURE WRITES THEM IN THE `A, B` ORDER ANYWAY
///
/// `#set = affine_set<(d0)[s0] : (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>`
/// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:5`) over a `vector<64xf16>`:
/// 64 lanes, 8 per slice, so `d0 + s0*8 - 64` and `-d0 + 63` — both rows matched, in that order. The
/// two `found_*` flags exist for the transposed spelling, and accepting it costs nothing.
///
/// ⛔ AN EQUALITY ROW IS NOT ONE OF THESE TWO. MLIR's `getConstraint(i)` returns the expression and
/// `isEq(i)` the flag separately, and the reference compares expressions only — but a set of two
/// rows where one is `d0 + s0*8 - 64 == 0` describes a single lane, not a span, and this island keeps
/// the flag on the row ([`crate::islands::dataflow_ir::ty::Constraint`]) so it is checked here
/// rather than dropped.
fn check_affine_set_constraints(mask_set: &IntegerSet, lanes: PtLanes) -> bool {
    // `if (num_constraints != 2)`
    if mask_set.constraints.len() != 2 {
        return false;
    }

    let elems = i64::from(lanes.elems());
    // `d0 + s0 * num_lanes_in_slice - op_num_elems`
    let constraint_a = LinearForm {
        d0: 1,
        s0: i64::from(lanes.lanes_in_slice()),
        constant: -elems,
    };
    // `-d0 + (op_num_elems - 1)`
    let constraint_b = LinearForm {
        d0: -1,
        s0: 0,
        constant: elems - 1,
    };

    let mut found_a = false;
    let mut found_b = false;
    for constraint in &mask_set.constraints {
        // See this function's note: an `== 0` row is neither of the two.
        if constraint.is_equality {
            return false;
        }
        match LinearForm::of(&constraint.expr) {
            Some(form) if form == constraint_a => found_a = true,
            Some(form) if form == constraint_b => found_b = true,
            // `else return … "unexpected constraint detected in PT mask_set"`
            _ => return false,
        }
    }

    // `if (!found_constraintA || !found_constraintB)` — the two rows cannot be the same row twice.
    found_a && found_b
}

/// Replaces: e089_getMaskValueForPT
///
/// # THE MASK VALUE A PT COMPUTE TAKES, READ OFF THE `create_affine_mask` THAT STATES IT
///
/// ```cpp
/// std::optional<Value> vectorchain::getMaskValueForPT(
///     Operation* op, Operation* operand, const SenComponents comp,
///     const SenSystemDef& sys_def) {
///   DT_CHECK_MSG(comp == PT, "function only checks mask value for PT unit");
///   // For PT, masking can only be specified with a create_affine_mask operation.
///   auto cam_op = dyn_cast<vectorchain::CreateAffineMaskOp>(operand);
///   if (!cam_op) { operand->emitOpError("PT mask should come from a CreateAffineMaskOp"); … }
///   auto mask_set = cast<IntegerSetAttr>(operand->getAttr("mask_set")).getValue();
///   if (mask_set.getNumDims() != 1) { … "Mask affine set has to have one dimension." … }
///   … // result-type checks, then:
///   int num_lanes_in_slice = op_num_elems / sys_def.numSlicesPerStick;
///   auto mask_parameter = cam_op.getMaskParameter();
///   bool static_mask = !mask_parameter;
///   if (!static_mask) {
///     if (auto const_op = llvm::dyn_cast<arith::ConstantOp>(mask_parameter.getDefiningOp())) {
///       static_mask = true;
///       auto affine_const = getAffineConstantExpr(
///           cast<IntegerAttr>(const_op.getValue()).getInt(), const_op.getContext());
///       mask_set = mask_set.replaceDimsAndSymbols({}, {affine_const},
///                                                 mask_set.getNumDims(), 0);
///     }
///   }
///   if (static_mask) { … } else { … }
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17-215`; the two
/// branches are quoted at [`MaskValue`], [`MaskParameterOwner`] and
/// [`check_affine_set_constraints`], and the bound reads below.)
///
/// # ⭐⭐ THREE KINDS OF MASK, NOT TWO, AND THE THIRD IS THE INTERESTING ONE
///
/// * NO `mask_parameter` — `static_mask = !mask_parameter`, a set that already says which lanes are
///   off;
/// * a parameter that is an `arith.constant` — ⛔ **PROMOTED TO STATIC**, by folding the literal into
///   the set's symbol (`replaceDimsAndSymbols`, `:69-77`). Without that fold the promoted set's `d0`
///   rows still hold `s0` and [`IntegerSet::constant_bound`] answers `None` for every one of them,
///   so `static_mask = true` would be followed by two `.value()` calls on absent bounds;
/// * a parameter that is an `arith.subi` of a loop's bound and its induction variable — DYNAMIC, and
///   the returned value is that induction variable.
///
/// The vendor writes all three in one file: `create_affine_mask {mask_set = #set2}` with no
/// parameter (`dynamic_pt_masking.mlir:296`), and `create_affine_mask %14 {mask_set = #set}` with
/// `%14 = arith.subi %13, %arg4` inside `sentient.for %arg4 = %13` (`:250-252`).
///
/// # THE STATIC ARM'S ARITHMETIC, PINNED BY A `CHECK-SENT-IR` LINE
///
/// `#set2 = affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` over `vector<64xf16>`: `Ub` is 63,
/// which is `op_num_elems - 1` ✓; `Lb` is 48, so `num_masked_elems = 64 - 48 = 16` and
/// `masked_columns = 16 / 8 = 2` — and the golden is
/// `sentient.scalar_constant {value = 2 : si64}` (`dynamic_pt_masking.mlir:86`, `:211`). The
/// all-lanes-off set `(d0 - 64 >= 0, -d0 + 63 >= 0)` gives `64 - 64 = 0` columns and its golden is
/// `{value = 0 : si64}` (`:85`) — ⭐ an EMPTY set with `Lb > Ub`, which is why
/// [`IntegerSet::constant_bound`] must not test for emptiness.
///
/// # `OpBuilder builder(op)` — ⛔ THE INSERTION POINT IS RETURNED, NOT TAKEN
///
/// The builder is constructed AT the masked op, so the constant lands immediately before it. This
/// port hands the op back in [`MaskValue::Constant`] for the caller to place, which is this crate's
/// standing shape for an emission whose position is the caller's business.
///
/// # THE STOPS, EACH A CITED ABORT OR UB
///
/// * `getConstantBound(UB/LB, 0).value()` on an absent bound — `std::optional::value()` throws
///   (`:91-93`, `:101-103`). Two `todo!`s, because the two sides fail for different sets.
/// * `dyn_cast<arith::ConstantOp>(mask_parameter.getDefiningOp())` with a null defining op — LLVM's
///   `dyn_cast` asserts on null (`:70-71`), and the reference's own `DT_CHECK_MSG(mask_def_op, …)`
///   twenty lines later (`:124-125`) says the case was meant to be impossible.
/// * A region argument owned by something other than a `sentient.for` — see [`MaskParameterOwner`].
///
/// # THE DT_CHECKS THAT ARE **DISCHARGED BY CONSTRUCTION**
///
/// `comp == PT` is [`PtUnit`]; `op->getNumResults() == 1` and `dyn_cast<VectorType>` of both results
/// are the [`Vector`] parameters, which cannot be absent or of another kind; `op_num_elems %
/// numSlicesPerStick == 0` is [`PtLanes`].
#[must_use]
pub fn get_mask_value_for_pt<A: Arch>(
    _unit: PtUnit,
    masked: Vector,
    mask: &vectorchain::Op,
    definitions: Definitions<'_>,
    loops: &[EnclosingLoop<'_>],
    values: &mut Values,
) -> Option<MaskValue> {
    // ⛔ THE TWO ISLAND VARIANTS ARE ONE C++ OP. `vectorchain.create_affine_mask` carries a required
    // `mask_set` attribute whichever form it is printed in; this island splits the op in two so that
    // the prefix form can defend its own type agreement ([`vectorchain::Op::CreateAffineMaskSet`]),
    // and the prefix form's set is the one it elides — `lanes` live lanes of `len` is
    // `(d0 - lanes >= 0, -d0 + (len - 1) >= 0)`, one dimension and no symbol. ⭐ CHECKED AGAINST THE
    // GOLDENS: 48 live of 64 is `#set2` and lowers to `{value = 2}`, 64 live of 64 is the all-off set
    // and lowers to `{value = 0}` (`dynamic_pt_masking.mlir:85-86`).
    let (mask_set, mask_parameter, mask_ty) = match mask {
        vectorchain::Op::CreateAffineMaskSet {
            mask_set,
            mask_parameter,
            ty,
            ..
        } => (mask_set.clone(), *mask_parameter, *ty),
        vectorchain::Op::CreateAffineMask { mask, .. } => {
            let ty = mask.ty();
            let live = i64::try_from(mask.live()).ok()?;
            let len = i64::try_from(ty.len).ok()?;
            let set = IntegerSet {
                dims: 1,
                symbols: 0,
                constraints: vec![
                    crate::islands::dataflow_ir::ty::Constraint {
                        expr: AffineExpr::dim(0).plus(AffineExpr::Const(-live)),
                        is_equality: false,
                    },
                    crate::islands::dataflow_ir::ty::Constraint {
                        expr: AffineExpr::dim(0)
                            .times(-1)
                            .plus(AffineExpr::Const(len - 1)),
                        is_equality: false,
                    },
                ],
            };
            (set, None, ty)
        }
        // `if (!cam_op) { operand->emitOpError("PT mask should come from a CreateAffineMaskOp"); }`
        _ => return None,
    };

    // `if (mask_set.getNumDims() != 1)` — "Mask affine set has to have one dimension."
    if mask_set.dims != 1 {
        return None;
    }

    // `if (mask_result_type.getNumElements() != op_num_elems)` — the mask and the value it masks
    // describe the same PT row. ⭐ BEFORE THE SLICE GUARD, which is the reference's own order
    // (`:47-50` then `:53-55`): a mask of the wrong width is a diagnostic whatever the row's width is.
    if mask_ty.len != masked.len {
        return None;
    }

    // The masked result's lanes, with the slice division already discharged ([`PtLanes`]).
    let lanes = PtLanes::of::<A>(masked)?;

    // `bool static_mask = !mask_parameter;` and the promotion of a constant parameter.
    let (mask_set, static_mask) = match mask_parameter {
        None => (mask_set, true),
        Some(parameter) => match definitions.of(parameter) {
            // `dyn_cast<arith::ConstantOp>` — either integer-typed constant is one.
            Some(sen::Op::Arith(arith::Op::Constant { value, .. })) => (
                // `replaceDimsAndSymbols({}, {affine_const}, getNumDims(), 0)`
                mask_set.replace_symbols(&[AffineExpr::Const(*value)], 0),
                true,
            ),
            // ⭐ INCLUDING AN `i1`: `cast<IntegerAttr>(const_op.getValue()).getInt()` (`:73-74`)
            // accepts a `BoolAttr`, which IS an `IntegerAttr` of width one in MLIR, so `true` folds
            // to 1 rather than making the mask dynamic.
            Some(sen::Op::Arith(arith::Op::ConstantInt { value, .. })) => {
                let literal = match value {
                    arith::IntConst::Bool(bit) => i64::from(*bit),
                    arith::IntConst::Int { value, .. } => *value,
                };
                (mask_set.replace_symbols(&[AffineExpr::Const(literal)], 0), true)
            }
            // Anything else stays dynamic; the `else` branch judges what it is.
            Some(_) => (mask_set, false),
            None => todo!(
                "getMaskValueForPT: a mask parameter with no defining op — the reference hands \
                 `dyn_cast<arith::ConstantOp>` a null pointer (Helper.cpp:70-71) and its own \
                 DT_CHECK_MSG at :124 says this cannot happen"
            ),
        },
    };

    if static_mask {
        // ── CONSTANT MASK ────────────────────────────────────────────────────────────────────────
        // `if (mask_set.getNumSymbols() != 0)` — "Mask affine set should not have any symbols".
        if mask_set.symbols != 0 {
            return None;
        }

        // `getConstantBound(UB, 0).value()` — the upper bound must be the last lane.
        let Some(upper_bound) = mask_set.constant_bound(BoundType::Ub, 0) else {
            todo!(
                "getMaskValueForPT: a constant mask set with no constant upper bound on d0 — \
                 the reference calls `std::optional::value()` on it (Helper.cpp:91-93)"
            )
        };
        // `if (upper_bound != op_num_elems - 1)`
        if upper_bound != i64::from(lanes.elems()) - 1 {
            return None;
        }

        // `getConstantBound(LB, 0).value()` — the first masked lane.
        let Some(lower_bound) = mask_set.constant_bound(BoundType::Lb, 0) else {
            todo!(
                "getMaskValueForPT: a constant mask set with no constant lower bound on d0 — \
                 the reference calls `std::optional::value()` on it (Helper.cpp:101-103)"
            )
        };
        let num_masked_elems = i64::from(lanes.elems()) - lower_bound;
        let slices = i64::from(A::SLICES_PER_STICK);
        // `if (num_masked_elems < 0 || num_masked_elems % sys_def.numSlicesPerStick != 0)`
        if num_masked_elems < 0 || num_masked_elems % slices != 0 {
            return None;
        }
        // ⭐ IN RANGE BY THE TWO CHECKS ABOVE: `0 <= num_masked_elems <= op_num_elems`, and the row
        // width is a `u32`.
        let masked_columns = u32::try_from(num_masked_elems / slices)
            .expect("a masked-column count of a PT row fits a u32");

        // `OpBuilder builder(op); return sentient::ConstantOp::create(builder, op->getLoc(),
        //                                                            builder.getIndexType(),
        //                                                            masked_columns);`
        // ⭐ THE LOCALE IS `imm`, the four-argument `create`'s declared default
        // (`SentientOps.td:848-852`) — the same value
        // [`super::std_standard_to_sentient::lower_constant_index_to_sentient`] writes.
        let value = values.mint();
        Some(MaskValue::Constant {
            op: sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: i64::from(masked_columns),
                result: value,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            value,
            columns: MaskedColumns(masked_columns),
        })
    } else {
        // ── DYNAMIC MASK ─────────────────────────────────────────────────────────────────────────
        // `if (mask_set.getNumSymbols() != 1)` — "Mask affine set should have 1 symbol". ⭐ THE
        // REFERENCE ASKS TWICE (`:118-121` and `:130-133`, with two different messages); one answer.
        if mask_set.symbols != 1 {
            return None;
        }

        // `auto non_const_mask = dyn_cast<arith::SubIOp>(mask_def_op);`
        //
        // ⭐ THE PARAMETER IS PRESENT AND DEFINED: the `None` arm above is the reference's own null
        // deref and the `static_mask` arms have returned.
        let parameter = mask_parameter?;
        let sub = match definitions.of(parameter) {
            Some(sen::Op::Arith(arith::Op::SubI(sub))) => *sub,
            // "Non-constant PT mask parameter is an unexpected operation"
            _ => return None,
        };

        // "Must be a sub of the <loop upper bound> - <loop iterator>." ⭐ AND THE REFERENCE EXPLAINS
        // WHY THE SUB EXISTS AT ALL: *"Sentient loops count down, not up, so to retain this
        // difference in behaviour during lowering, loop iterators are converted to subs before use
        // in loops. At vectorchain lowering, these subs exist."* (`Helper.cpp:139-143`)
        match MaskParameterOwner::of(sub.rhs, definitions, loops) {
            // "non-constant PT mask value should be loop iterator"
            MaskParameterOwner::Defined => return None,
            MaskParameterOwner::ForeignRegion => todo!(
                "getMaskValueForPT: the rhs of a dynamic mask's `arith.subi` is a region argument \
                 of something other than a `sentient.for` — DT_CHECK_MSG (Helper.cpp:152-156)"
            ),
            // `if (lhs != rhs_parent.getBound() || rhs != rhs_parent.getInductionVar())`
            MaskParameterOwner::Loop(enclosing) => {
                if sub.lhs != enclosing.bound || sub.rhs != enclosing.iv {
                    return None;
                }
            }
        }

        // `if (checkAffineSetConstraints(...).failed()) return std::nullopt;`
        if !check_affine_set_constraints(&mask_set, lanes) {
            return None;
        }

        // `// Return the loop iterator used for the mask.` / `return rhs;`
        Some(MaskValue::LoopIterator(sub.rhs))
    }
}


#[cfg(test)]
mod unit_tests {
    use super::{
        EnclosingLoop, MaskValue, PtLanes, PtUnit, get_mask_value_for_pt,
    };
    use crate::arch::Dd2;
    use crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::MaskedColumns;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, Constraint, ElemType, GenericComp, IntegerSet, ScalarTy, Vector,
    };
    use crate::islands::sentient::dialects::{
        self as sen, Definitions, Val, arith, sentient, vectorchain,
    };

    /// The masked result — `vector<64xf16>`, the width every PT fixture uses.
    const ROW: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };
    /// The mask's own type — `vector<64xi1>`.
    const MASK_TY: Vector = Vector {
        len: 64,
        elem: ElemType::Int(1),
    };

    /// An inequality row.
    fn ineq(expr: AffineExpr) -> Constraint {
        Constraint {
            expr,
            is_equality: false,
        }
    }

    /// `affine_set<(d0) : (d0 - first >= 0, -d0 + (len - 1) >= 0)>` — a constant PT mask.
    fn constant_set(first: i64, len: i64) -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                ineq(AffineExpr::dim(0).plus(AffineExpr::Const(-first))),
                ineq(AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(len - 1))),
            ],
        }
    }

    /// `#set = affine_set<(d0)[s0] : (d0 + s0 * lanes_in_slice - len >= 0, -d0 + (len - 1) >= 0)>` —
    /// the ONE set a dynamic PT mask may have
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:5`).
    fn dynamic_set(lanes_in_slice: i64, len: i64) -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 1,
            constraints: vec![
                ineq(
                    AffineExpr::dim(0)
                        .plus(AffineExpr::sym(0).times(lanes_in_slice))
                        .plus(AffineExpr::Const(-len)),
                ),
                ineq(AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(len - 1))),
            ],
        }
    }

    /// `vectorchain.create_affine_mask [%parameter] {mask_set} : vector<64xi1>`.
    fn mask_op(mask_set: IntegerSet, mask_parameter: Option<Val>) -> vectorchain::Op {
        vectorchain::Op::CreateAffineMaskSet {
            result: Val(100),
            mask_set,
            mask_parameter,
            ty: MASK_TY,
        }
    }

    /// 🎯 089/384 — A CONSTANT MASK'S COLUMNS COME FROM ITS LOWER BOUND, AND IT **EMITS**.
    ///
    /// `#set2 = affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` over a `vector<64xf16>`:
    /// `num_masked_elems = 64 - 48 = 16`, `masked_columns = 16 / 8 = 2`, and the golden is
    /// `sentient.scalar_constant {value = 2 : si64}`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:86`, `:211`).
    #[test]
    fn a_constant_masks_columns_come_from_its_lower_bound() {
        let mut values = Values::default();
        let mask = mask_op(constant_set(48, 64), None);

        let value = get_mask_value_for_pt::<Dd2>(
            PtUnit,
            ROW,
            &mask,
            Definitions::from_innermost(&[]),
            &[],
            &mut values,
        );

        let Some(MaskValue::Constant { op, value, columns }) = value else {
            unreachable!("a constant mask with no parameter")
        };
        assert_eq!(columns, MaskedColumns(2));
        assert_eq!(
            op,
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: 2,
                result: value,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            "the `sentient.scalar_constant` the reference's OpBuilder inserts at the masked op"
        );
    }

    /// 🎯 089/384 — AN ALL-LANES-OFF SET IS ZERO COLUMNS, AND IT IS AN **EMPTY** SET.
    ///
    /// `affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` admits no integer at all — `Lb` 64 is
    /// greater than `Ub` 63 — yet both bounds are present and the golden is
    /// `sentient.scalar_constant {value = 0 : si64}` (`dynamic_pt_masking.mlir:85`). An emptiness test
    /// anywhere on this path would turn it into a refusal.
    #[test]
    fn an_all_lanes_off_set_is_zero_columns() {
        let mut values = Values::default();
        let mask = mask_op(constant_set(64, 64), None);

        let Some(MaskValue::Constant { columns, .. }) = get_mask_value_for_pt::<Dd2>(
            PtUnit,
            ROW,
            &mask,
            Definitions::from_innermost(&[]),
            &[],
            &mut values,
        ) else {
            unreachable!("the empty set is a legal mask")
        };
        assert_eq!(columns, MaskedColumns(0));
    }

    /// 🎯 089/384 — THE ISLAND'S PREFIX FORM IS THE SAME OP AND STATES THE SAME SET.
    ///
    /// `vectorchain.create_affine_mask` carries a `mask_set` whichever form this island prints it in;
    /// 48 live lanes of 64 is `#set2`, so it must answer with `#set2`'s own two columns.
    #[test]
    fn the_prefix_form_states_the_same_set() {
        let mut values = Values::default();
        let mask = vectorchain::Op::CreateAffineMask {
            result: Val(100),
            mask: vectorchain::LaneMask::prefix_of(48, MASK_TY),
        };

        let Some(MaskValue::Constant { columns, .. }) = get_mask_value_for_pt::<Dd2>(
            PtUnit,
            ROW,
            &mask,
            Definitions::from_innermost(&[]),
            &[],
            &mut values,
        ) else {
            unreachable!("a prefix mask is a constant mask")
        };
        assert_eq!(columns, MaskedColumns(2));
    }

    /// 🎯 089/384 — A CONSTANT MASK PARAMETER IS **PROMOTED**, AND THE FOLD IS WHAT MAKES IT READABLE.
    ///
    /// ⭐⭐ THE TWO VENDOR SETS ARE THE SAME SET: `#set` with `s0 = 2` is
    /// `d0 + 2*8 - 64 = d0 - 48`, which is `#set2` — and `#set2`'s golden is
    /// `{value = 2 : si64}` (`dynamic_pt_masking.mlir:5`, `:86`). So a dynamic set whose parameter
    /// turns out to be `arith.constant 2` must produce exactly two columns, and without
    /// `replaceDimsAndSymbols` its rows still hold `s0` and have no constant bound at all.
    #[test]
    fn a_constant_mask_parameter_is_promoted_and_folded() {
        let mut values = Values::default();
        let scope = vec![sen::Op::Arith(arith::Op::Constant {
            result: Val(0),
            value: 2,
        })];
        let mask = mask_op(dynamic_set(8, 64), Some(Val(0)));

        let Some(MaskValue::Constant { columns, .. }) = get_mask_value_for_pt::<Dd2>(
            PtUnit,
            ROW,
            &mask,
            Definitions::from_innermost(&[&scope]),
            &[],
            &mut values,
        ) else {
            unreachable!("a constant parameter promotes the mask to static")
        };
        assert_eq!(columns, MaskedColumns(2));
    }

    /// 🎯 089/384 — A DYNAMIC MASK RETURNS THE LOOP ITERATOR, AND EMITS NOTHING.
    ///
    /// ```text
    /// sentient.for %arg4 = %13 { ..
    ///   %14 = arith.subi %13, %arg4 : index
    ///   %260 = vectorchain.create_affine_mask %14 {mask_set = #set} : vector<64xi1>
    /// ```
    /// (`dynamic_pt_masking.mlir:250-252`) — `lhs` is the loop's bound and `rhs` its induction
    /// variable, so the mask value is `%arg4` itself.
    #[test]
    fn a_dynamic_mask_returns_the_loop_iterator() {
        let mut values = Values::default();
        let scope = vec![sen::Op::Arith(arith::Op::SubI(arith::IntBinary {
            result: Val(14),
            lhs: Val(13),
            rhs: Val(4),
            ty: ScalarTy::Index,
        }))];
        let for_op = sen::Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(13),
            carried: Vec::new(),
            dbg_name: None,
            body: Vec::new(),
        });
        let loops = [EnclosingLoop::of(&for_op).expect("a sentient.for")];
        let mask = mask_op(dynamic_set(8, 64), Some(Val(14)));

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[&scope]),
                &loops,
                &mut values,
            ),
            Some(MaskValue::LoopIterator(Val(4)))
        );
        assert_eq!(values.issued(), 0, "the dynamic arm mints nothing");
    }

    /// 🎯 089/384 — A SUBTRACTION THAT IS NOT `<loop bound> - <loop iterator>` IS REFUSED.
    ///
    /// `if (lhs != rhs_parent.getBound() || rhs != rhs_parent.getInductionVar())` (`Helper.cpp:160`).
    #[test]
    fn a_subtraction_of_something_other_than_the_loops_bound_is_refused() {
        let mut values = Values::default();
        let scope = vec![sen::Op::Arith(arith::Op::SubI(arith::IntBinary {
            result: Val(14),
            // ⛔ NOT the loop's bound.
            lhs: Val(99),
            rhs: Val(4),
            ty: ScalarTy::Index,
        }))];
        let for_op = sen::Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(13),
            carried: Vec::new(),
            dbg_name: None,
            body: Vec::new(),
        });
        let loops = [EnclosingLoop::of(&for_op).expect("a sentient.for")];
        let mask = mask_op(dynamic_set(8, 64), Some(Val(14)));

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[&scope]),
                &loops,
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — A MASK PARAMETER THAT IS NOT AN `arith.subi` IS REFUSED.
    ///
    /// *"Non-constant PT mask parameter is an unexpected operation"* (`Helper.cpp:127-131`).
    #[test]
    fn a_mask_parameter_that_is_not_a_subtraction_is_refused() {
        let mut values = Values::default();
        let scope = vec![sen::Op::Arith(arith::Op::AddI(arith::IntBinary {
            result: Val(14),
            lhs: Val(13),
            rhs: Val(4),
            ty: ScalarTy::Index,
        }))];
        let mask = mask_op(dynamic_set(8, 64), Some(Val(14)));

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[&scope]),
                &[],
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — A SET WHOSE SYMBOL COEFFICIENT IS NOT THE SLICE WIDTH IS REFUSED.
    ///
    /// *"unexpected constraint detected in PT mask_set"* (`Helper.cpp:187-190`) — `s0 * 4` where the
    /// row's slice holds 8 lanes describes a mask that advances at half the rate.
    #[test]
    fn a_set_with_the_wrong_slice_width_is_refused() {
        let mut values = Values::default();
        let scope = vec![sen::Op::Arith(arith::Op::SubI(arith::IntBinary {
            result: Val(14),
            lhs: Val(13),
            rhs: Val(4),
            ty: ScalarTy::Index,
        }))];
        let for_op = sen::Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(13),
            carried: Vec::new(),
            dbg_name: None,
            body: Vec::new(),
        });
        let loops = [EnclosingLoop::of(&for_op).expect("a sentient.for")];
        let mask = mask_op(dynamic_set(4, 64), Some(Val(14)));

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[&scope]),
                &loops,
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — AN UPPER BOUND THAT IS NOT THE LAST LANE IS REFUSED.
    ///
    /// *"Constant mask upper bound does not match expected value"* (`Helper.cpp:94-98`): the mask and
    /// the row it masks have to be the same width.
    #[test]
    fn an_upper_bound_short_of_the_last_lane_is_refused() {
        let mut values = Values::default();
        let mask = mask_op(constant_set(48, 32), None);

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[]),
                &[],
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — A MASK OVER A DIFFERENT NUMBER OF LANES THAN THE OP IS REFUSED.
    ///
    /// *"number of PT mask elements should match number of op elements"* (`Helper.cpp:47-50`).
    #[test]
    fn a_mask_of_another_width_is_refused() {
        let mut values = Values::default();
        let mask = vectorchain::Op::CreateAffineMaskSet {
            result: Val(100),
            mask_set: constant_set(48, 64),
            mask_parameter: None,
            ty: Vector {
                len: 32,
                elem: ElemType::Int(1),
            },
        };

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[]),
                &[],
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — A MASK THAT IS NOT A `create_affine_mask` IS REFUSED.
    ///
    /// *"PT mask should come from a CreateAffineMaskOp"* (`Helper.cpp:25-28`).
    #[test]
    fn a_mask_from_another_op_is_refused() {
        let mut values = Values::default();
        let mask = vectorchain::Op::Cast {
            result: Val(100),
            input: Val(99),
            input_ty: ROW,
            ty: MASK_TY,
        };

        assert_eq!(
            get_mask_value_for_pt::<Dd2>(
                PtUnit,
                ROW,
                &mask,
                Definitions::from_innermost(&[]),
                &[],
                &mut values,
            ),
            None
        );
    }

    /// 🎯 089/384 — THE PT IS THE ONLY COMPONENT THIS LOWERING IS FOR.
    ///
    /// `DT_CHECK_MSG(comp == PT, …)` (`Helper.cpp:20`), as a witness rather than an abort.
    #[test]
    fn only_a_pt_has_this_lowering() {
        assert_eq!(PtUnit::of(GenericComp::Pt), Some(PtUnit));
        assert_eq!(PtUnit::of(GenericComp::Pe), None);
        assert_eq!(PtUnit::of(GenericComp::Sfp), None);
    }

    /// 🎯 089/384 — A ROW THAT DOES NOT DIVIDE INTO SLICES HAS NO LANES.
    ///
    /// `DT_CHECK_MSG(op_num_elems % sys_def.numSlicesPerStick == 0, …)` (`Helper.cpp:53-55`) as a
    /// build-time guard: the quotient exists only where the reference would not have aborted.
    #[test]
    fn a_row_that_does_not_divide_into_slices_has_no_lanes() {
        let lanes = PtLanes::of::<Dd2>(ROW).expect("64 lanes divide into eight slices");
        assert_eq!((lanes.elems(), lanes.lanes_in_slice()), (64, 8));

        assert!(
            PtLanes::of::<Dd2>(Vector {
                len: 60,
                elem: ElemType::F16,
            })
            .is_none()
        );
    }
}
