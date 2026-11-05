//! The capture a drafter reads of its target: every target arch's DSL gets it from the compiler, never
//! from the DSL.
//!
//! A DFlash2 drafter (Splash's, ~/git/splash runtime/model/QwenTarget.cpp `addCapture`) reads the
//! output of each target layer in its `target_layer_ids`, side by side. Which layers is the drafter's
//! fact, which the target's config carries as its `draft_capture_layers` layer set when a drafter is
//! compiled beside it (the macro crate's `config` head facts; empty otherwise). The rows it reads are the
//! layer loop's residual stream at the end of each of those layers: what the loop carries into its
//! next iteration and hands the code after it.
//!
//! [`insert`] appends, to the end of an arch's decoder layer loop (`for layer in
//! range(num_hidden_layers)`), the capture of the variable that loop carries out:
//!
//! ```python
//! if layer in draft_capture_layers:
//!     __draft_capture = capture_hidden(<carried>)
//! else:
//!     __draft_capture = <carried>
//! ```
//!
//! With an empty layer set the `else` arm runs at every layer: an alias, no tile. So a target with no
//! drafter compiles to the tapes it did before. A loop that carries no single such variable gets no
//! capture; a target of it with a drafter beside it then has no capture site, which
//! `to_wavefront` refuses.

use std::collections::BTreeSet;

use proc_macro2::{Ident, Span};

use crate::ast::{Ast, BoolExpr, BoundExpr, Expr, InSet, Stmt};

/// The layer set of a target's layers whose outputs a head compiled beside it reads, which the
/// target captures (`capture_hidden`): the head's `target_capture_layers`, written into the
/// target's config before it parses. Every model has the set, empty unless a head reads some.
pub const DRAFT_CAPTURE_LAYERS: &str = "draft_capture_layers";

/// The loop bound a decoder layer loop runs to.
const LAYER_COUNT: &str = "num_hidden_layers";
/// The op a capture is, and the local its result binds (read by nothing).
const CAPTURE_OP: &str = "capture_hidden";
const CAPTURE_LOCAL: &str = "__draft_capture";

/// `ast` with its decoder layer loop's carried variable captured at every layer in
/// `draft_capture_layers` (module docs). The last top-level loop over `num_hidden_layers` that
/// carries exactly one variable out — read before it, written in it, read after it — gets it.
pub fn insert(ast: &Ast) -> Ast {
    let mut out = ast.clone();
    let statements = &mut out.statements;
    let site = (0..statements.len()).rev().find_map(|i| {
        let Stmt::For {
            ivar, end, body, ..
        } = &statements[i]
        else {
            return None;
        };
        if !matches!(end, BoundExpr::Ident(e) if e == LAYER_COUNT) {
            return None;
        }
        let before = assigned(&statements[..i]);
        let after = read(&statements[i + 1..]);
        let carried: Vec<String> = (assigned(body).into_iter())
            .filter(|name| before.contains(name) && after.contains(name))
            .collect();
        match carried.as_slice() {
            [name] => Some((i, ivar.clone(), name.clone())),
            _ => None,
        }
    });
    if let Some((i, ivar, carried)) = site
        && let Stmt::For { body, .. } = &mut statements[i]
    {
        body.push(capture(&ivar, &carried));
    }
    out
}

fn ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

/// The capture of `carried` at the layers of `draft_capture_layers` (module docs).
fn capture(ivar: &Ident, carried: &str) -> Stmt {
    let carried = Expr::Var(ident(carried));
    Stmt::If {
        cond: BoolExpr::In {
            ivar: ivar.clone(),
            members: InSet::Named(ident(DRAFT_CAPTURE_LAYERS)),
        },
        then_body: vec![Stmt::Assign {
            target: ident(CAPTURE_LOCAL),
            value: Expr::Call {
                op: ident(CAPTURE_OP),
                args: vec![carried.clone()],
            },
            line: None,
        }],
        else_body: vec![Stmt::Assign {
            target: ident(CAPTURE_LOCAL),
            value: carried,
            line: None,
        }],
    }
}

/// Every name `stmts` assign, in any arm or loop body.
fn assigned(stmts: &[Stmt]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for s in stmts {
        match s {
            Stmt::Assign { target, .. } => {
                names.insert(target.to_string());
            }
            Stmt::AssignTuple { targets, .. } => names.extend(targets.iter().map(Ident::to_string)),
            Stmt::For { body, .. } => names.extend(assigned(body)),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                names.extend(assigned(then_body));
                names.extend(assigned(else_body));
            }
        }
    }
    names
}

/// Every name `stmts` read, in any arm or loop body.
fn read(stmts: &[Stmt]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for s in stmts {
        match s {
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => {
                reads(value, &mut names)
            }
            Stmt::For { body, .. } => names.extend(read(body)),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                names.extend(read(then_body));
                names.extend(read(else_body));
            }
        }
    }
    names
}

fn reads(e: &Expr, names: &mut BTreeSet<String>) {
    match e {
        Expr::Var(v) => {
            names.insert(v.to_string());
        }
        Expr::Index { target, .. } => reads(target, names),
        Expr::Call { args, .. } => args.iter().for_each(|a| reads(a, names)),
        Expr::Mul { lhs, rhs } | Expr::Add { lhs, rhs } => {
            reads(lhs, names);
            reads(rhs, names);
        }
        Expr::Reshape { source, .. } => reads(source, names),
        Expr::Path(_) | Expr::ScalarLit(_) | Expr::SqrtBound(_) | Expr::ConfigScalar { .. } => {}
    }
}
