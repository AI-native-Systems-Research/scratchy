//! The capture a drafter reads of its target: every target arch's DSL gets it from the compiler, never
//! from the DSL.
//!
//! A DFlash2 drafter (Splash's, ~/git/splash runtime/model/QwenTarget.cpp `addCapture`) reads the
//! output of each target layer in its `target_layer_ids`, side by side. Which layers is the drafter's
//! fact, which the target's config carries as its `draft_capture_layers` layer set when a drafter is
//! compiled beside it ([`crate::config`]'s head facts; empty otherwise). The rows it reads are the
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
use crate::config::DRAFT_CAPTURE_LAYERS;

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
        let Stmt::For { ivar, end, body, .. } = &statements[i] else {
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
        }],
        else_body: vec![Stmt::Assign {
            target: ident(CAPTURE_LOCAL),
            value: carried,
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
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => reads(value, &mut names),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classified::OpKind;
    use crate::classify::classify;
    use crate::config::{self, LayerSet, ModelParams};
    use crate::fuf::{Fuf, FufInput, unroll};
    use crate::parse_python::parse_body;
    use crate::shape::infer;
    use std::path::PathBuf;

    /// A decoder: an embedding, a layer loop carrying `hidden_states`, the final norm and head.
    const DECODER: &str = "
        hidden_states = embed(input_ids, embed_tokens)
        for layer in range(num_hidden_layers):
            normed = rmsnorm(hidden_states, input_layernorm[layer])
            hidden_states = add(normed, hidden_states)
        normed = rmsnorm(hidden_states, norm)
        logits = gemm(normed, lm_head)
        ";

    fn llama(captured: &[u64]) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch/configs/llama/llama-3.2-1b.json");
        let mut params = config::load_file(&path).expect("load config");
        let set = LayerSet::of(captured.iter().copied());
        params.layer_sets.insert(DRAFT_CAPTURE_LAYERS.to_string(), set);
        params
    }

    fn unroll_ast(ast: &Ast, params: &ModelParams) -> Fuf {
        let program = classify(ast).expect("classify");
        let manifest = crate::weights_manifest::WeightsManifest::llama_test_conventions();
        let inferred = infer(&program, &manifest, &Default::default()).expect("infer");
        let cfg = build_cfg(&program, params).expect("build cfg");
        unroll(&cfg, &inferred).expect("unroll")
    }

    fn ops(fuf: &Fuf) -> Vec<OpKind> {
        fuf.nodes.iter().map(|n| n.op).collect()
    }

    /// A target with no drafter beside it (an empty `draft_capture_layers`) unrolls to exactly the
    /// tiles it did before the capture was inserted: every layer takes the alias arm.
    #[test]
    fn a_target_with_no_drafter_unrolls_as_before() {
        let ast = parse_body(DECODER).expect("parse");
        let params = llama(&[]);
        let (before, after) = (unroll_ast(&ast, &params), unroll_ast(&insert(&ast), &params));
        assert_eq!(ops(&before), ops(&after));
    }

    /// With a drafter reading layers 1 and 3, the target captures the residual stream each of them
    /// carries out — the add that ends the layer — and nothing else.
    #[test]
    fn a_target_captures_the_layers_its_drafter_reads() {
        let ast = parse_body(DECODER).expect("parse");
        let fuf = unroll_ast(&insert(&ast), &llama(&[1, 3]));
        let adds: Vec<u32> = (fuf.nodes.iter())
            .filter(|n| n.op == OpKind::Add)
            .map(|n| n.id.0)
            .collect();
        let captured: Vec<u32> = (fuf.nodes.iter())
            .filter(|n| n.op == OpKind::CaptureHidden)
            .map(|n| match n.inputs[0] {
                FufInput::Tile { id, .. } => id.0,
                ref other => panic!("a capture of {other:?}"),
            })
            .collect();
        assert_eq!(captured, [adds[1], adds[3]]);
    }

    /// A layer loop whose writes nothing after it reads carries nothing out: no capture.
    #[test]
    fn a_loop_carrying_nothing_out_gets_no_capture() {
        let src = "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                scratch = rmsnorm(hidden_states, input_layernorm[layer])
            logits = gemm(hidden_states, lm_head)
            ";
        let ast = parse_body(src).expect("parse");
        let Stmt::For { body, .. } = &insert(&ast).statements[1] else {
            panic!("the loop moved");
        };
        assert_eq!(body.len(), 1);
    }
}
