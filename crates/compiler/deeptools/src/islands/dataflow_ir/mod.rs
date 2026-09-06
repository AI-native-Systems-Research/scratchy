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

pub mod op;
pub mod print;
pub mod ty;

use crate::arch::Arch;
use crate::generated::OpFunc;
use op::{Op, Val};

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
    /// The units this runs on, in the order the schedule names them.
    pub on: Vec<op::Val>,
    /// `precision =`, present only where the unit computes.
    pub precision: Option<op::Precision>,
    /// What it runs.
    pub body: Vec<Op>,
    /// The arch it was lowered for.
    pub arch: core::marker::PhantomData<A>,
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
/// use deeptools::islands::dataflow_ir::{
///     Grid, GroupId, KernelName, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Run,
/// };
/// fn one<A: Arch>(index: u32) -> Program<A> {
///     Program {
///         name: ProgramName { group: GroupId(0), index: OpIndex(index), func: OpFunc::Add },
///         grid: Grid::single(),
///         preamble: Vec::new(),
///         units: ProgramUnits::of(
///             ProgramUnit {
///                 on: Vec::new(),
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
/// use deeptools::islands::dataflow_ir::{
///     Grid, GroupId, KernelName, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Run,
/// };
/// fn one<A: Arch>(index: u32) -> Program<A> {
///     Program {
///         name: ProgramName { group: GroupId(0), index: OpIndex(index), func: OpFunc::Add },
///         grid: Grid::single(),
///         preamble: Vec::new(),
///         units: ProgramUnits::of(
///             ProgramUnit {
///                 on: Vec::new(),
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
