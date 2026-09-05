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

/// ONE PROGRAM: a named module holding the DataflowIR one schedule runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<A: Arch> {
    /// The module's symbol, which is also its function's name.
    pub name: ProgramName,
    /// `attributes {grid = [N]}` on the function — the grid the schedule was split for.
    pub grid: Grid,
    /// The body: the units it declares and the programs it runs on them.
    pub body: Vec<Op>,
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
/// use deeptools::arch::{Dd2, Sen1p5};
/// use deeptools::islands::dataflow_ir::{Grid, GroupId, KernelName, Program, ProgramName, Run};
/// let dd2: Program<Dd2> = Program {
///     name: ProgramName::Golden("a"),
///     grid: Grid::single(),
///     body: Vec::new(),
///     arch: core::marker::PhantomData,
/// };
/// let sen: Program<Sen1p5> = Program {
///     name: ProgramName::Golden("b"),
///     grid: Grid::single(),
///     body: Vec::new(),
///     arch: core::marker::PhantomData,
/// };
/// let _ = Run { kernel: KernelName::Group(GroupId(0)), programs: vec![dd2, sen] };
/// ```
///
/// while one arch alone is fine:
///
/// ```
/// use deeptools::arch::Dd2;
/// use deeptools::islands::dataflow_ir::{Grid, GroupId, KernelName, Program, ProgramName, Run};
/// let dd2: Program<Dd2> = Program {
///     name: ProgramName::Golden("a"),
///     grid: Grid::single(),
///     body: Vec::new(),
///     arch: core::marker::PhantomData,
/// };
/// let _ = Run { kernel: KernelName::Group(GroupId(0)), programs: vec![dd2] };
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
/// same name, and nothing could read the group back out of one. Rendering is
/// [`ProgramName::spelling`]'s job and happens once, in the printer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProgramName {
    /// One this crate emits: which group, where in it, and which op-func.
    Emitted {
        /// Which group it belongs to.
        group: GroupId,
        /// Where it sits in that group.
        index: OpIndex,
        /// Which op-func it lowers — carried so the symbol says what it is.
        func: OpFunc,
    },
    /// 🛑 A REFERENCE FILE'S OWN SYMBOL, and the only reason this is an enum.
    ///
    /// IBM's golden DataflowIR names its modules whatever it names them (`dataflowProgram`), and the
    /// golden test compares BYTES — so reproducing their text means carrying their symbol.
    ///
    /// ⛔ `&'static str`, NOT `String`. A golden's name is a literal in a test; it cannot be built at
    /// runtime from anything a caller computed, which is what keeps this from being a way back to
    /// stringly-typed names. No lowering constructs it.
    Golden(&'static str),
}

impl core::fmt::Display for ProgramName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProgramName::Emitted { group, index, func } => {
                write!(f, "g{}_{}_{}", group.0, index.0, func.spelling())
            }
            ProgramName::Golden(symbol) => f.write_str(symbol),
        }
    }
}

/// THE SYMBOL OF A RUN'S KERNEL. See [`ProgramName`] for why the golden arm exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelName {
    /// The group this run compiles.
    Group(GroupId),
    /// A reference file's own kernel symbol.
    Golden(&'static str),
}

impl core::fmt::Display for KernelName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KernelName::Group(group) => write!(f, "group_{}", group.0),
            KernelName::Golden(symbol) => f.write_str(symbol),
        }
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
