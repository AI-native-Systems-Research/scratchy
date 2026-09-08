//! THE DATAFLOWIR ISLAND — what scratchy lowers to, and what the backend compiler consumes.
//!
//! ```text
//! subtile tape ──► DataflowIR ──► dbo-opt ──► spyreCodeDir/{init_binary.bin, spyrecode.json}
//!                  (here)         (C++)
//! ```
//!
//! An emitted run is a top module holding one UNNAMED declaration module, which calls the programs
//! in the order they run, and one NAMED module per program holding its DataflowIR. That shape is
//! fixed by `dbo/test/adapt-scheduler-dfir-multi.mlir`, the golden input of the pass that consumes
//! it, and `dbo/src/Pipeline/RunProgramPipelines.cpp:199-211` is what the stage below looks for: the
//! inner module, and a `dataflow::ProgramUnitOp` inside it.

pub mod dialects;
pub mod link;
pub mod print;
pub mod ty;

use crate::arch::Arch;
use crate::generated::OpFunc;
use crate::units;
use dialects::{Op, Val};

/// MINTS SSA VALUES, so a program's numbering is the builder's and never a caller's.
///
/// ⛔ THE COUNTER IS THE ONLY WAY TO GET A [`Val`]. Two ops binding the same value is not a diagnosis
/// anyone would enjoy making from MLIR's own error, so the identity is issued rather than written.
#[derive(Debug, Default)]
pub struct Values {
    next: u32,
}

impl Values {
    /// A fresh value.
    pub fn mint(&mut self) -> Val {
        let val = Val(self.next);
        self.next += 1;
        val
    }

    /// How many have been issued — the width a reader needs to align the printed names.
    #[must_use]
    pub fn issued(&self) -> u32 {
        self.next
    }
}

/// ONE VALUE STANDING FOR ANOTHER — MLIR's `IRMapping`, restricted to values.
///
/// ⭐ WHAT A CLONE READS ITS OPERANDS THROUGH. `transformSCFToAffineLoop` builds one of these before
/// it copies a loop body, mapping the `scf.for`'s induction variable and region arguments to the
/// `affine.for`'s (`TransformLoopToLegalizeForSentientLowering.cpp:113-119`), and then every operand
/// of every cloned op is looked up in it.
///
/// ⛔ A MISS IS NOT A REFUSAL — it is a value defined OUTSIDE what is being cloned, and it comes
/// through unchanged. That is `IRMapping::lookupOrDefault`, and it is why a body reading a constant
/// hoisted above the loop still reads that same constant after the clone.
#[derive(Debug, Default, Clone)]
pub struct ValueMapping {
    pairs: Vec<(Val, Val)>,
}

impl ValueMapping {
    /// An empty mapping: everything stands for itself.
    #[must_use]
    pub fn new() -> ValueMapping {
        ValueMapping { pairs: Vec::new() }
    }

    /// `from` now stands for `to`.
    ///
    /// ⭐ A LATER ENTRY SHADOWS AN EARLIER ONE, as assigning to a `DenseMap` slot does. See
    /// [`ValueMapping::lookup_or_default`].
    pub fn map(&mut self, from: Val, to: Val) {
        self.pairs.push((from, to));
    }

    /// What `val` stands for, or [`None`] where nothing was mapped — `IRMapping::lookupOrNull`.
    ///
    /// ⛔ A DIFFERENT ANSWER FROM [`Self::lookup_or_default`]'s, AND CALLERS BRANCH ON IT.
    /// `updateTPMVInfo` reassigns an index only when the lookup hit, because *"indices may contain
    /// iterators that weren't re-cloned"* (`TransformPagedMemViewImpl.cpp:385-387`) — with the
    /// defaulting form that test could never fail.
    #[must_use]
    pub fn lookup(&self, val: Val) -> Option<Val> {
        self.pairs
            .iter()
            .rev()
            .find(|(from, _)| *from == val)
            .map(|(_, to)| *to)
    }

    /// What `val` stands for — ITSELF where nothing was mapped.
    ///
    /// ⛔ THE SEARCH IS FROM THE BACK, so the newest entry for a value wins. Cloning one body twice
    /// maps the same source value twice, and the second clone must not read the first clone's names.
    #[must_use]
    pub fn lookup_or_default(&self, val: Val) -> Val {
        self.pairs
            .iter()
            .rev()
            .find(|(from, _)| *from == val)
            .map_or(val, |(_, to)| *to)
    }
}

impl Values {
    /// CLONES `ops`, MINTING A FRESH VALUE FOR EVERYTHING THEY DEFINE — `OpBuilder::clone`.
    ///
    /// # ⭐⭐ THE ONE MECHANISM `transformSCFToAffineLoop` IS BUILT ON
    ///
    /// `builder.clone(op, bv_map)` over a loop body's non-terminator ops
    /// (`TransformLoopToLegalizeForSentientLowering.cpp:120-127`) does three things at once, and all
    /// three are here:
    ///
    /// 1. every OPERAND is read through `mapping` — so the copy reads the new loop's induction
    ///    variable where the original read the old one, and reads values defined outside the body
    ///    unchanged;
    /// 2. every value the op DEFINES — its results and whatever its regions bind — gets a FRESH
    ///    name from this counter, recorded in `mapping` so later ops of the same body pick it up;
    /// 3. regions are cloned the same way, recursively, so a nested loop inside the body comes out
    ///    with its own induction variable rather than sharing the original's.
    ///
    /// ⛔⛔ FRESH NAMES ARE THE WHOLE POINT, NOT A DETAIL. The caller clones ONE `scf.for` body
    /// TWICE — once into each arm of the `scf.if` that selects between the two static trip counts
    /// (`scf_loop_with_result.mlir:52-77`) — and two copies binding the same SSA names is not a
    /// program. MLIR would say *"redefinition of value"*; a typed island has nothing to say at all,
    /// it just prints a module whose second definition silently wins.
    ///
    /// ⛔ AND THE ORDER WITHIN ONE OP IS MLIR'S: OPERANDS, RESULTS, BLOCK ARGUMENTS, REGIONS. Two
    /// separate facts hold it in place:
    ///
    /// * operands come FIRST because an op cannot read what it defines — registering its results
    ///   before rewriting its operands would let a self-referential name through;
    /// * results come before the region's arguments, and those before the region's body, because
    ///   that is the order MLIR's own printer NUMBERS values in, and the numbering is the output.
    ///   `%23 = affine.for %24 = 0 to 16 iter_args(%25 = %20)` followed by `%26 = affine.for %27`
    ///   (`scf_loop_with_result.mlir:56-57`) is result, induction variable, carried argument, then
    ///   the nested loop — so a clone that minted its body's names before its own result would emit
    ///   a correct program that no vendored expectation matches.
    ///
    /// ⭐ THE TERMINATOR IS THE CALLER'S BUSINESS. `without_terminator()` is at the call site
    /// (`:126`) because the yield an `affine.for` needs is an `affine.yield`, built from the
    /// `scf.yield`'s operands looked up through the same mapping (`:129-136`) — not a clone of it.
    pub fn clone_ops(&mut self, ops: &[Op], mapping: &mut ValueMapping) -> Vec<Op> {
        let mut cloned = Vec::with_capacity(ops.len());
        for op in ops {
            let mut copy = op.clone();
            let parts = dialects::parts_mut(&mut copy);
            for operand in parts.operands {
                *operand = mapping.lookup_or_default(*operand);
            }
            for result in parts.results {
                let fresh = self.mint();
                mapping.map(*result, fresh);
                *result = fresh;
            }
            for arg in parts.block_args {
                let fresh = self.mint();
                mapping.map(*arg, fresh);
                *arg = fresh;
            }
            for region in parts.regions {
                // ⭐ TAKEN, NOT COPIED AGAIN. `op.clone()` above already deep-copied the body; this
                // renumbers that copy in place rather than making a third one.
                let source = core::mem::take(region);
                *region = self.clone_ops(&source, mapping);
            }
            cloned.push(copy);
        }
        cloned
    }

    /// ONE OP CLONED WITH ITS REGIONS LEFT EMPTY — `Operation::cloneWithoutRegions(IRMapping &)`.
    ///
    /// # ⭐⭐ THE THREE THINGS IT DOES, AND THE ONE IT DELIBERATELY DOES NOT
    ///
    /// It is [`Self::clone_ops`] for a single op minus the region recursion: operands are read through
    /// `mapping`, every result is freshly minted and recorded in `mapping` so later clones of the same
    /// body pick the new name up, and the regions come out EMPTY for the caller to fill.
    ///
    /// ⛔⛔ ITS ONE CALLER FILLS THEM WITH A DIFFERENT BODY, WHICH IS WHY IT EXISTS.
    /// `cloneOpsForRegions` (`FlatteningLocalRegions.cpp:257-271`) clones a region-carrying op and then
    /// fills region `rn` of the copy with only those of the ORIGINAL's children whose
    /// `is_in_region_num` is `rn` **and** which belong to the unit class being built — so a deep clone
    /// would copy the ops it is about to filter out, and copy them with the wrong names.
    ///
    /// # ⚠️ BLOCK ARGUMENTS ARE **NOT** REMINTED, AND THAT IS THE REFERENCE'S OWN GAP
    ///
    /// MLIR's `cloneWithoutRegions` creates the regions with NO BLOCKS AT ALL, so it maps no block
    /// arguments; the caller then calls `emplaceBlock()` (`:265`), which creates a block with no
    /// arguments either. A cloned op that BINDS one — an `affine.for`'s induction variable — therefore
    /// has nothing declaring it in the reference, and every operand of the cloned body that read it
    /// comes through `arg_map.lookupOrDefault` **unchanged**, naming a value the new op does not
    /// define. This island holds a binder as a field on the op rather than as a block argument, so the
    /// literal transcription — leave the field alone — is also the well-formed one: the copy declares
    /// the same name its copied body reads. ⭐ No fixture reaches it. The only region-carrying ops the
    /// pass actually clones are `scf.if`s, which bind nothing
    /// (`dcc/test/Transform/FlatteningLocalRegions/flatten_local_region4.mlir:346-357`).
    #[must_use]
    pub fn clone_without_regions(&mut self, op: &Op, mapping: &mut ValueMapping) -> Op {
        let mut copy = op.clone();
        let parts = dialects::parts_mut(&mut copy);
        // Operands first, for the same reason as in [`Self::clone_ops`]: an op cannot read what it
        // defines.
        for operand in parts.operands {
            *operand = mapping.lookup_or_default(*operand);
        }
        // `mapper.map(getResult(i), newOp->getResult(i))` — MLIR's own clone records this, and
        // `cloneOpsForRegions` depends on it: the ops it clones next read the copy's names.
        for result in parts.results {
            let fresh = self.mint();
            mapping.map(*result, fresh);
            *result = fresh;
        }
        // `op.clone()` above deep-copied the bodies; the copy's regions are emptied rather than never
        // filled, because there is no way to build a `DfirOp` without its region fields.
        for region in parts.regions {
            region.clear();
        }
        copy
    }
}

/// ONE `dataflow.program_unit` — the units it runs on, and what they run.
///
/// ⛔⛔ ONE NODE IS THREE OF THESE, because the datapath has three ends. A compute has NO READ PORT
/// TO THE SCRATCHPAD — `VectorOperands.cpp:187-206` resolves a compute operand's view to a register
/// file and an `lx` view is `Unknown memory type` — so the loader reads memory and sends, the
/// compute drains the wire and sends on, and the store unit receives and writes.
///
/// ⛔ WHICH IS WHY WRAPPING THE WHOLE BODY IN ONE UNIT WOULD NOT BE THE FIX. It satisfies the pass
/// and describes a compute reading the LX directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramUnit<A: Arch> {
    /// The units this runs on, in the order the schedule names them — ALL OF ONE KIND.
    pub on: Units,
    /// `precision =`, present only where the unit computes.
    pub precision: Option<dialects::dataflow::Precision>,
    /// What it runs.
    pub body: Vec<Op>,
    /// The arch it was lowered for.
    pub arch: core::marker::PhantomData<A>,
}

/// THE UNITS ONE `dataflow.program_unit` RUNS ON — ALL OF ONE KIND.
///
/// # 🛑 THE KIND IS NOT A LABEL, IT IS WHAT THE LOWERING READS
///
/// ⛔⛔ `Helper.cpp:2173-2176` takes the unit kind from `getUnits()[0]` alone, under the comment
/// *"It is guaranteed from the upstream passes that all units of a unit operation will have same
/// types."* A list mixing an `lxlu` with an `lxsu` makes that comment false, and every downstream
/// question about "the unit" is then answered from whichever happened to be first. R126
/// `Src unit types has to be the same.` and R163 `Unit type is inconsistent.` are the same fact
/// checked elsewhere.
///
/// ⛔ THIS WAS A `Vec<Val>`, and the emitter built it by pushing `Lxlu` and `Lxsu` into one bag —
/// the kind was known at the binding and thrown away one line later.
/// ⛔⛔ NON-EMPTY, AND THAT IS A CRASH NOT A DIAGNOSTIC. `ProgramUnitsReduction.cpp:175` is
/// `dcc::getUnitType(unit.getUnits()[0].getDefiningOp())` — an unguarded `[0]` on a `ValueRange`.
/// `Dataflow.td:107` declares `Variadic<Index>:$units`, so zero units PARSES and VERIFIES; the pass
/// then aborts the whole compiler with
/// *"Assertion failed: (Index < size() && \"invalid index for value range\")"* and no diagnostic at
/// all. `head` is a field so `vals().is_empty()` is not a question that can be asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Units {
    kind: units::DfirUnit,
    head: dialects::Val,
    rest: Vec<dialects::Val>,
}

impl Units {
    /// Every bound unit OF THIS KIND, in the order the schedule named them — or `None` where the
    /// schedule names none.
    ///
    /// ⛔ THE ONLY CONSTRUCTOR, AND THE KIND IS THE FILTER. A caller cannot hand over a list that
    /// mixes kinds, because it does not hand over a list at all — it names a kind and gets the
    /// units that are of it.
    ///
    /// ⛔⛔ `None` IS NOT A REFUSAL, IT IS AN ABSENCE. A schedule that names no unit of this kind
    /// has no program unit of this kind — the same shape as `IS_DECODE` removing the row nest. The
    /// caller emits one fewer unit; it does not emit an empty one and it does not stop.
    #[must_use]
    pub fn of(kind: units::DfirUnit, bound: &[(units::DfirUnit, dialects::Val)]) -> Option<Units> {
        let mut vals = bound.iter().filter(|(k, _)| *k == kind).map(|(_, v)| *v);
        let head = vals.next()?;
        Some(Units {
            kind,
            head,
            rest: vals.collect(),
        })
    }

    /// ONE bound unit of a kind — total, and no `Option`.
    ///
    /// ⭐ FOR A UNIT THE ARCH PROVIDES rather than the schedule names. [`Self::of`] filters a list
    /// the template wrote and so may find none; a unit bound unconditionally from the machine's own
    /// topology is already known to exist, and saying so here is what keeps the caller from having
    /// an `expect` for a case that cannot arise.
    #[must_use]
    pub fn one(kind: units::DfirUnit, val: dialects::Val) -> Units {
        Units {
            kind,
            head: val,
            rest: Vec::new(),
        }
    }

    /// Which kind these are.
    #[must_use]
    pub fn kind(&self) -> units::DfirUnit {
        self.kind
    }

    /// The unit the backend reads the kind from — `getUnits()[0]`, which always exists.
    #[must_use]
    pub fn first(&self) -> dialects::Val {
        self.head
    }

    /// The bound units, in schedule order.
    #[must_use]
    pub fn vals(&self) -> Vec<dialects::Val> {
        core::iter::once(self.head)
            .chain(self.rest.iter().copied())
            .collect()
    }

    /// WHETHER AN `agen.composite_load_and_store` MAY RUN ON THIS KIND.
    ///
    /// ⛔⛔ ONLY THE L3 HALVES, AND THE REFUSAL IS SILENT. `Helper.cpp:2177-2179` is
    /// `if (!is_any_of(comp, L3LU, L3SU)) return LogicalResult::failure();` — a bare failure with no
    /// message, so all dbo-opt prints is the caller's wrapper, *"Unable to generate loops and
    /// sentient statements for the composite vector operations"* (`:2965-2967`). The emitter put the
    /// HBM→LX transfer on an `lxlu` and read that text as being about the transfer's shape.
    ///
    /// ⭐ IBM AGREES: their `l3lu` unit holds the `composite_load_and_store`s
    /// (`/tmp/ktir_ref/export/debug/dfir.mlir:64` on `%0,%1`, transfers at `:82,:90`) while their
    /// `lxlu` unit holds `agen.vector_load` + `dataflow.send` and no transfer at all (`:104-129`).
    ///
    /// ⭐ AND A SECOND PASS ASKS THE SAME QUESTION IN THE SAME WORDS. `CanonicalizeToggle.cpp:59-61`
    /// is the same `dcc::getUnitType(unit_op.getUnits()[0].getDefiningOp<GetUnitOp>())` followed by
    /// the same `is_any_of(comp, L3LU, L3SU)`, selecting which units get their toggles canonicalized
    /// (entry 178, see
    /// [`crate::bridges::dataflow_ir_to_sentient::tf_canonicalize_toggle::run_on_operation`]) — a
    /// toggle is a data transfer's address, so the units that hold one are the units that move
    /// memory. ⛔ ONE PREDICATE, ASKED HERE: two copies of it would be two answers to whether an
    /// `lxlu` moves memory.
    ///
    /// ⛔ EXHAUSTIVE, NO WILDCARD. A new unit kind must say whether it is an L3 half rather than
    /// silently inherit `false`.
    #[must_use]
    pub const fn moves_memory(&self) -> bool {
        use units::DfirUnit;
        match self.kind {
            DfirUnit::L3lu | DfirUnit::L3su => true,
            DfirUnit::Sfp
            | DfirUnit::Pe
            | DfirUnit::PtRow(_)
            | DfirUnit::Lxlu
            | DfirUnit::Lxsu
            | DfirUnit::Lx
            | DfirUnit::Hbm
            | DfirUnit::L0lu
            | DfirUnit::L0su
            | DfirUnit::L0
            | DfirUnit::Constant
            | DfirUnit::SfpState
            | DfirUnit::PeState
            | DfirUnit::SfpRing
            // ⛔ A VIEWED MEMORY, NOT A MOVER. The virtual IBR is what an indirect access INDEXES
            // THROUGH (`Helper.cpp:388-431`); the L3 halves still do the moving.
            | DfirUnit::LxVirtualIbr
            | DfirUnit::L3Ibr
            // ⛔ AND THE CROSS-PARTITION LINK MOVES NOTHING OF ITS OWN either: it is a send
            // destination the LXLU routes to, not a memory a transfer reads or writes.
            | DfirUnit::CrossPtnLink => false,
        }
    }
}

/// A PROGRAM'S UNITS — NON-EMPTY BY CONSTRUCTION.
///
/// ⭐ THE HEAD IS A FIELD, NOT AN INDEX. `units.is_empty()` is not a question that can be asked,
/// which is what makes "found no program to compile" unreachable from our side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramUnits<A: Arch> {
    head: ProgramUnit<A>,
    rest: Vec<ProgramUnit<A>>,
}

impl<A: Arch> ProgramUnits<A> {
    /// A program's units, the first one being what makes it a program at all.
    #[must_use]
    pub fn of(head: ProgramUnit<A>, rest: Vec<ProgramUnit<A>>) -> Self {
        Self { head, rest }
    }

    /// Every unit, head first.
    pub fn iter(&self) -> impl Iterator<Item = &ProgramUnit<A>> {
        core::iter::once(&self.head).chain(self.rest.iter())
    }

    /// Every unit, head first, FOR REWRITING IN PLACE.
    ///
    /// ⭐ FOR A MODULE-WIDE PASS. `redefineConstantVectors(module_op)`
    /// (`VectorChainToSentientPT.cpp:983`) walks the whole module before any unit is lowered, and the
    /// module here IS this list plus [`Program::preamble`] — so a pass that rewrites a use needs to
    /// reach every body. ⛔ IT CANNOT ADD OR REMOVE A UNIT, so the non-emptiness this type exists for
    /// still holds.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ProgramUnit<A>> {
        core::iter::once(&mut self.head).chain(self.rest.iter_mut())
    }
}

/// ONE PROGRAM: a named module holding the DataflowIR one schedule runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<A: Arch> {
    /// The module's symbol, which is also its function's name.
    pub name: ProgramName,
    /// `attributes {grid = [N]}` on the function — the grid the schedule was split for.
    pub grid: Grid,
    /// The preamble: the units and views the program declares, before any unit runs.
    pub preamble: Vec<Op>,
    /// THE PROGRAM UNITS — AT LEAST ONE, AND `Vec<Op>` CANNOT SPELL THAT.
    ///
    /// ⛔⛔ dbo-opt: *"dbo-adapt-scheduler-dfir found no program to compile"*. The predicate is
    /// `AdaptSchedulerDfir.cpp:63-78` — it walks each child module for a `func.func` containing a
    /// `dataflow::ProgramUnitOp` and fails when none has one. `body: Vec<Op>` accepted any op
    /// sequence, so a program with no `program_unit` at all was constructible — and `Op::ProgramUnit`
    /// had exactly ONE mention in the whole crate, the printer arm that would have rendered it.
    /// A variant that compiles, prints, and is never built.
    ///
    /// ⭐ A NON-EMPTY LIST, so "a program with no units" is not a state that exists. The units are
    /// the STRUCTURE of a program, not ops among ops, which is why they are their own field rather
    /// than something the emitter may or may not push.
    pub units: ProgramUnits<A>,
    /// The arch this program was lowered for.
    ///
    /// ⭐⭐ NOT DECORATION. `Dd2` and `Sen1p5` both exist in every build — only [`crate::arch::Target`]
    /// is feature-selected — so without this, a program lowered against DD2's eight PT rows and one
    /// lowered against SEN1P5's four are the same type and can be put in one [`Run`]. The units
    /// inside them are bound from the arch's own topology, so that mixture emits a program naming
    /// rows the target does not have.
    pub arch: core::marker::PhantomData<A>,
}

/// A WHOLE RUN: the programs, and the order they are called in.
///
/// ⭐ THE DECLARATION MODULE IS DERIVED, NOT STORED. It is exactly "call each program once, in
/// order" — so holding it as data would be holding a second copy of [`Run::programs`] that could
/// disagree with it.
///
/// # 🛑 A RUN CANNOT MIX TWO ARCHES
///
/// ⛔ `Dd2` AND `Sen1p5` BOTH EXIST IN EVERY BUILD — only [`crate::arch::Target`] is
/// feature-selected — so before [`Program`] carried its arch, one lowered against DD2's eight PT
/// rows and one lowered against SEN1P5's four were the SAME TYPE and could go in one run. The units
/// inside are bound from the arch's own topology, so that mixture emits a program naming rows the
/// target does not have. VERIFIED RED:
///
/// ```compile_fail
/// use deeptools::arch::{Arch, Dd2, Sen1p5};
/// use deeptools::generated::OpFunc;
/// use deeptools::units::DfirUnit;
/// use deeptools::islands::dataflow_ir::dialects::Val;
/// use deeptools::islands::dataflow_ir::{
///     Grid, GroupId, KernelName, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Run,
///     Units,
/// };
/// fn one<A: Arch>(index: u32) -> Program<A> {
///     Program {
///         name: ProgramName { group: GroupId(0), index: OpIndex(index), func: OpFunc::Add },
///         grid: Grid::single(),
///         preamble: Vec::new(),
///         units: ProgramUnits::of(
///             ProgramUnit {
///                 on: Units::one(DfirUnit::Sfp, Val(0)),
///                 precision: None,
///                 body: Vec::new(),
///                 arch: core::marker::PhantomData,
///             },
///             Vec::new(),
///         ),
///         arch: core::marker::PhantomData,
///     }
/// }
/// let dd2: Program<Dd2> = one(0);
/// let sen: Program<Sen1p5> = one(1);
/// let _ = Run { kernel: KernelName(GroupId(0)), programs: vec![dd2, sen] };
/// ```
///
/// while one arch alone is fine:
///
/// ```
/// use deeptools::arch::{Arch, Dd2};
/// use deeptools::generated::OpFunc;
/// use deeptools::units::DfirUnit;
/// use deeptools::islands::dataflow_ir::dialects::Val;
/// use deeptools::islands::dataflow_ir::{
///     Grid, GroupId, KernelName, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Run,
///     Units,
/// };
/// fn one<A: Arch>(index: u32) -> Program<A> {
///     Program {
///         name: ProgramName { group: GroupId(0), index: OpIndex(index), func: OpFunc::Add },
///         grid: Grid::single(),
///         preamble: Vec::new(),
///         units: ProgramUnits::of(
///             ProgramUnit {
///                 on: Units::one(DfirUnit::Sfp, Val(0)),
///                 precision: None,
///                 body: Vec::new(),
///                 arch: core::marker::PhantomData,
///             },
///             Vec::new(),
///         ),
///         arch: core::marker::PhantomData,
///     }
/// }
/// let dd2: Program<Dd2> = one(0);
/// let _ = Run { kernel: KernelName(GroupId(0)), programs: vec![dd2] };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run<A: Arch> {
    /// The kernel's name, which the declaration module's function takes.
    pub kernel: KernelName,
    /// The programs, in the order they run.
    ///
    /// ⛔ ALL ON ONE ARCH, by the type. See [`Program::arch`].
    pub programs: Vec<Program<A>>,
}

/// WHICH LAUNCH GROUP — the unit a bundle is compiled and cached as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

/// WHERE AN OP SITS IN ITS GROUP, in the order the group's json lists it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpIndex(pub u32);

/// THE SYMBOL OF ONE PROGRAM'S MODULE.
///
/// ⭐⭐ THE PARTS, NOT THE TEXT. This was a `String` built at the call site with `format!`, which
/// made the symbol's shape a convention rather than a fact: nothing stopped two programs taking the
/// same name, and nothing could read the group back out of one. Rendering happens once, in
/// [`core::fmt::Display`].
///
/// ⛔ NO `&'static str` ARM. There was one, carrying a reference file's own symbol so a byte-exact
/// comparison could reproduce it. Nothing this crate lowers has a name it did not compute, so the
/// only thing that arm bought was a way back to stringly-typed symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramName {
    /// Which group it belongs to.
    pub group: GroupId,
    /// Where it sits in that group.
    pub index: OpIndex,
    /// Which op-func it lowers — carried so the symbol says what it is.
    pub func: OpFunc,
}

impl core::fmt::Display for ProgramName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "g{}_{}_{}",
            self.group.0,
            self.index.0,
            self.func.spelling()
        )
    }
}

/// THE SYMBOL OF A RUN'S KERNEL — the group it compiles. See [`ProgramName`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelName(pub GroupId);

impl core::fmt::Display for KernelName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "group_{}", self.0.0)
    }
}

/// THE GRID A SCHEDULE WAS SPLIT FOR — `attributes {grid = [N]}`.
///
/// ⛔ NOT A BARE `Vec<u32>`. The extents are per-axis counts drawn from the arch's own topology, and
/// an empty grid is not a grid — [`Grid::single`] is the un-split case, spelled rather than left to
/// a caller to remember as `vec![1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid(Vec<u32>);

impl Grid {
    /// The un-split grid: one instance.
    #[must_use]
    pub fn single() -> Grid {
        Grid(vec![1])
    }

    /// The extents, outermost first.
    #[must_use]
    pub fn extents(&self) -> &[u32] {
        &self.0
    }
}
