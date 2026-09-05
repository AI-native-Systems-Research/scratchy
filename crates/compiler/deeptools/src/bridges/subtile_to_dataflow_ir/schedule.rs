//! THE TEMPLATE'S SCHEDULE, WALKED.
//!
//! ⭐⭐ THE DDL TEMPLATE IS THE SCHEDULE AND THE NODE IS THE SHAPE. This file is the first half:
//! which units take part, what moves between them, in what loop nest, and where the computes sit —
//! all of it READ from the vendored template that `build.rs` walked, none of it invented. The
//! node's extents fill it in; neither side can be derived from the other.
//!
//! ⛔⛔ AN EMITTER THAT DID NOT READ `Program::stmts` WOULD BE INVENTING THE SCHEDULE, and would
//! look perfectly reasonable doing it. The first version of the tape walk emitted a hand-written
//! body shape and never called `OpFunc::program` at all.
//!
//! ⭐ THE CONDITIONS ARE ALREADY RESOLVED. `build.rs` flattens the branches it can decide for a
//! given op-bind — 995 exclusive branches across the vendored set — so `If` and `Dataflow` do not
//! appear among the statement kinds the reachable schedules use at all. What survives is
//! [`Enclosing::Arm`], the branches whose predicate the walk genuinely could not settle.

use crate::generated::{Attrs, Enclosing, NameId, Program, Role, Stmt, StmtKind};

/// WHERE A STATEMENT SITS IN THE NEST, as a comparable key.
///
/// ⭐ THE PATH SUPERSEDES THE DEPTH. `Stmt::depth` counts enclosing `ddl.loop`s, which is what a
/// predicate resolves a loop label against; `Stmt::path` says WHICH loops, in order, and that is
/// what decides whether two statements share a body or merely sit at the same level.
type Path = &'static [Enclosing];

/// ONE LEVEL OF THE WALKED NEST.
#[derive(Debug)]
pub enum Level {
    /// A statement that emits at this level.
    Stmt(&'static Stmt),
    /// A `ddl.loop`, and everything inside it.
    Loop {
        /// Which statement in the program is the loop itself.
        stmt: usize,
        /// Its body.
        body: Vec<Level>,
    },
    /// One arm of a branch the walk could not decide.
    Arm {
        /// `true` for the `then` arm.
        then_arm: bool,
        /// Its body.
        body: Vec<Level>,
    },
}

/// REBUILD THE NEST FROM THE STATEMENTS' PATHS.
///
/// The statements arrive flat, each carrying every construct that encloses it, outermost first. Two
/// statements belong to the same body exactly when their paths agree up to that depth.
#[must_use]
pub fn nest(program: &'static Program) -> Vec<Level> {
    build(program.stmts, 0)
}

fn build(stmts: &'static [Stmt], depth: usize) -> Vec<Level> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < stmts.len() {
        let path: Path = stmts[at].path;
        if path.len() <= depth {
            out.push(Level::Stmt(&stmts[at]));
            at += 1;
            continue;
        }
        // A run of statements sharing this enclosing construct.
        let head = &path[depth];
        let mut end = at;
        while end < stmts.len()
            && stmts[end].path.len() > depth
            && same(&stmts[end].path[depth], head)
        {
            end += 1;
        }
        let body = build(&stmts[at..end], depth + 1);
        out.push(match *head {
            Enclosing::Loop { stmt } => Level::Loop { stmt, body },
            Enclosing::Arm { then_arm, .. } => Level::Arm { then_arm, body },
        });
        at = end;
    }
    out
}

/// Whether two enclosing constructs are the SAME one, not merely the same shape.
///
/// ⛔ A LOOP IS IDENTIFIED BY ITS OWN STATEMENT, not by its position in the path. Two sibling loops
/// at the same depth would otherwise merge into one, and every statement of the second would be
/// emitted inside the first.
const fn same(a: &Enclosing, b: &Enclosing) -> bool {
    match (a, b) {
        (Enclosing::Loop { stmt: x }, Enclosing::Loop { stmt: y }) => *x == *y,
        (
            Enclosing::Arm { then_arm: x, .. },
            Enclosing::Arm { then_arm: y, .. },
        ) => *x == *y,
        _ => false,
    }
}

/// WHICH OF THE OP'S OPERANDS A TEMPLATE NAME IS, if any.
///
/// ⭐ THE JOIN, AND IT IS DECLARED DATA. `Program::roles` is built by `build.rs` from the bind's own
/// `ddl.operation_bind`, so "which SSA name receives input 1" is read rather than guessed from a
/// naming convention. The nuked attempt recovered this from string dimension names.
#[must_use]
pub fn role_of(program: &'static Program, name: NameId) -> Option<Role> {
    program
        .roles
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, role)| *role)
}

/// EVERY STATEMENT THAT EMITS SOMETHING, and what it is.
///
/// ⛔ THE DECLARATIONS ARE NOT NOTHING — they bind the names everything else refers to — but they
/// produce no DataflowIR of their own. Separating the two is what stops the walker from silently
/// skipping a kind it merely has no arm for.
#[must_use]
pub const fn emits(kind: StmtKind) -> bool {
    matches!(
        kind,
        StmtKind::Unit
            | StmtKind::Allocate
            | StmtKind::GetExternalDataTransferAllocation
            | StmtKind::DataTransfer
            | StmtKind::Compute
            | StmtKind::Loop
            | StmtKind::Sync
            | StmtKind::ImplicitSync
            | StmtKind::Opaque
            | StmtKind::CoreToCoreCommunication
    )
}

/// THE UNITS THIS SCHEDULE BINDS, in the order it binds them.
///
/// ⭐ READ FROM THE `ddl.unit` STATEMENTS, which is the template saying which parts of the machine
/// take part. A program that bound a different set would be scheduled against a machine the
/// template did not describe.
#[must_use]
pub fn units_of(program: &'static Program) -> Vec<crate::units::DfirUnit> {
    let mut out: Vec<crate::units::DfirUnit> = Vec::new();
    for stmt in program.stmts {
        if let Attrs::Unit { unit, .. } = stmt.attrs {
            for dfir in crate::units::dfir_units(unit) {
                if !out.contains(&dfir) {
                    out.push(dfir);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Level, emits, nest, units_of};
    use crate::arch::{Arch, Target};
    use crate::generated::{DataType, OpFunc, StmtKind};

    fn count(levels: &[Level], want: fn(&Level) -> bool, n: &mut usize) {
        for level in levels {
            if want(level) {
                *n += 1;
            }
            match level {
                Level::Loop { body, .. } | Level::Arm { body, .. } => count(body, want, n),
                Level::Stmt(_) => {}
            }
        }
    }

    /// ⭐⭐ THE NEST IS REBUILT, AND IT IS NOT FLAT.
    ///
    /// `broadcast_ops.ddl` puts its work inside two enclosing `ddl.loop`s and a third around the
    /// streamed transfer (`:143-161`), so a walk that returned a flat list would have lost the nest
    /// the template describes — and every transfer inside it would lose the induction variable it
    /// is strided by.
    #[test]
    fn the_loop_nest_survives_the_walk() {
        let program = OpFunc::Add.program(Target::GEN, DataType::Sen169Fp16);
        let levels = nest(program);

        let mut loops = 0;
        count(&levels, |l| matches!(l, Level::Loop { .. }), &mut loops);
        assert!(
            loops >= 2,
            "broadcast_ops nests its work at least two loops deep; the walk found {loops}"
        );

        // ⛔ AND NOT EVERYTHING IS AT THE TOP. A flat walk would put every statement at level 0.
        let top_stmts = levels
            .iter()
            .filter(|l| matches!(l, Level::Stmt(_)))
            .count();
        assert!(
            top_stmts < program.stmts.len(),
            "a flat walk is not a nest: {top_stmts} of {} statements sat at the top",
            program.stmts.len()
        );
    }

    /// ⭐⭐ EVERY STATEMENT APPEARS EXACTLY ONCE — the nest re-parents, it does not drop.
    ///
    /// ⛔ EQUALITY, NOT A BOUND. "the walk saw at least one statement" passes on a walk that lost
    /// nine tenths of the schedule. A `ddl.loop`'s own statement is a `Level::Stmt` like any other
    /// — its path does not enclose itself — so the count comes out exact or the walk is wrong.
    #[test]
    fn the_walk_loses_no_statement() {
        for op_func in [
            OpFunc::Add,
            OpFunc::Matmul,
            OpFunc::Silu,
            OpFunc::Mean,
            OpFunc::Restickifyophbm,
        ] {
            let program = op_func.program(Target::GEN, DataType::Sen169Fp16);
            let levels = nest(program);
            let mut seen = 0;
            count(&levels, |l| matches!(l, Level::Stmt(_)), &mut seen);
            assert_eq!(
                seen,
                program.stmts.len(),
                "{op_func:?}: the nest holds {seen} statements but the schedule has {}",
                program.stmts.len()
            );
        }
    }

    /// ⭐ THE SCHEDULE NAMES REAL UNITS, and a row span expands.
    #[test]
    fn the_schedule_binds_the_units_the_template_names() {
        use crate::units::DfirUnit;

        let add = units_of(OpFunc::Add.program(Target::GEN, DataType::Sen169Fp16));
        assert!(
            add.contains(&DfirUnit::Sfp) && add.contains(&DfirUnit::Lxlu),
            "an elementwise add streams LX -> SFP, so both must be bound: got {add:?}"
        );

        // ⛔ CARRY THE VALUE. `bmm.ddl` names `ptrow1-7`, which on this arch's {} rows is a SPAN, and
        // the walk must expand it into that many distinct PT rows rather than one unit.
        let matmul = units_of(OpFunc::Matmul.program(Target::GEN, DataType::Sen169Fp16));
        let rows = matmul.iter().filter(|u| u.is_pt_row()).count();
        assert_eq!(
            rows,
            Target::PT_ROWS as usize,
            "a matmul's schedule names every PT row of this arch"
        );
    }

    /// ⛔ THE DECLARATIONS EMIT NOTHING, and saying so is what keeps a missing arm visible.
    #[test]
    fn declarations_are_not_emissions() {
        assert!(!emits(StmtKind::Dimension));
        assert!(!emits(StmtKind::Layout));
        assert!(!emits(StmtKind::Type));
        assert!(!emits(StmtKind::Tensor));
        assert!(emits(StmtKind::DataTransfer));
        assert!(emits(StmtKind::Compute));
        assert!(emits(StmtKind::Unit));
    }
}
