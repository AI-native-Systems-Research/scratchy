// SPDX-License-Identifier: Apache-2.0
//! Phase 2: classify free variables.
//!
//! Walks the raw AST and produces a [`classified::Program`] in
//! which every variable reference is tagged as a local, extern
//! param, or weight ref. Unknown reads (a bare name that isn't any
//! of those at a given site) are parse-time errors, not papered
//! over.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use syn::Ident;

use crate::ast::{self, BoundExpr};
use crate::classified::{
    BoolPred, Bound, Expr, ExternKind, LocalId, LocalTable, OpKind, Prelude, Program, Stmt,
    WeightTable,
};
use crate::shape::{Dim, canonical_mul};

pub type ClassifyResult<T> = Result<T, syn::Error>;

/// Convert a parser-produced [`ast::DimSpec`] into a [`Dim`]. `Lit` /
/// `Bound` map directly; `Mul` flattens into [`Dim::Mul`] via
/// `canonical_mul` (matching the form `apply_reshape_hints` produces);
/// `Div` becomes [`Dim::Div`] — kept symbolic until codegen folds the
/// numerator + closes the denominator against the per-model `bounds`.
fn dimspec_to_dim(s: &ast::DimSpec) -> Dim {
    match s {
        ast::DimSpec::Lit(n) => Dim::Lit(*n),
        ast::DimSpec::Bound(ident) => Dim::Bound(ident.to_string()),
        ast::DimSpec::Mul(a, b) => canonical_mul(vec![dimspec_to_dim(a), dimspec_to_dim(b)]),
        ast::DimSpec::Div(num, den) => {
            Dim::Div(Box::new(dimspec_to_dim(num)), Box::new(dimspec_to_dim(den)))
        }
    }
}

/// Classify the AST against the decoder prelude. Convenience entry
/// point used by unit tests in adjacent modules; production code in
/// `compile_common` calls [`classify_with`] with an explicit prelude.
#[allow(dead_code)]
pub fn classify(ast: &ast::Ast) -> ClassifyResult<Program> {
    classify_with(ast, Prelude::Decoder)
}

/// Classify the AST against a specific prelude. The prelude selects
/// which extern names are recognized — decoder bodies use
/// `input_ids`/`positions`/etc., vision bodies use
/// `pixels`/`cu_seqlens`/etc. — and is otherwise inert: every
/// downstream pass operates on the same `Program` shape.
pub fn classify_with(ast: &ast::Ast, prelude: Prelude) -> ClassifyResult<Program> {
    let mut cx = Ctx {
        prelude,
        ..Ctx::default()
    };
    let statements = cx.classify_stmts(&ast.statements)?;
    Ok(Program {
        statements,
        locals: cx.locals,
        weights: cx.weights,
        reshape_targets: cx.reshape_targets,
        prelude,
        decoder_safetensors_prefix: None,
        weight_leaf_renames: Vec::new(),
    })
}

#[derive(Default)]
struct Ctx {
    locals: LocalTable,
    weights: WeightTable,
    /// Name → most-recent LocalId for that name. Reflects straight-
    /// line SSA: a new assignment shadows the previous binding.
    /// For-loops push their induction variable and pop it on exit.
    scope: HashMap<String, Vec<LocalId>>,
    prelude: Prelude,
    /// DSL-authored reshape targets accumulated during classify.
    /// Identical surface to `shape::apply_reshape_hints`'s synthesized
    /// reshapes — both flow into `Program::reshape_targets` keyed by
    /// the producing `LocalId`. See [`Ctx::classify_stmt`] for the
    /// `ast::Expr::Reshape` Assign-arm intercept.
    reshape_targets: HashMap<LocalId, Vec<Dim>>,
}

impl Ctx {
    fn classify_stmts(&mut self, stmts: &[ast::Stmt]) -> ClassifyResult<Vec<Stmt>> {
        stmts.iter().map(|s| self.classify_stmt(s)).collect()
    }

    fn classify_stmt(&mut self, stmt: &ast::Stmt) -> ClassifyResult<Stmt> {
        match stmt {
            ast::Stmt::Assign { target, value } => {
                // `out = reshape(source, [d0, d1, ...])` is the only
                // place an `ast::Expr::Reshape` is admitted. We
                // intercept here (before `classify_expr`) so the
                // target shape can be threaded into
                // `program.reshape_targets` keyed by the freshly
                // bound `LocalId`. The classified Stmt drops the
                // explicit shape and becomes a Reshape OpKind call —
                // identical surface to the synthesized form from
                // `shape::apply_reshape_hints`, so every downstream
                // pass (CFG / FUF / shape::infer / solver / codegen)
                // already knows how to handle it.
                if let ast::Expr::Reshape {
                    source,
                    target_shape,
                } = value
                {
                    let source = self.classify_expr(source)?;
                    let id = self.bind(target.clone());
                    let dims: Vec<Dim> = target_shape.iter().map(dimspec_to_dim).collect();
                    self.reshape_targets.insert(id, dims);
                    return Ok(Stmt::Assign {
                        target: id,
                        value: Expr::Call {
                            op: OpKind::Reshape,
                            args: vec![source],
                        },
                    });
                }
                // Classify RHS first so that `x = f(x)` reads the
                // previous binding of `x`, not the new one.
                let value = self.classify_expr(value)?;
                let id = self.bind(target.clone());
                Ok(Stmt::Assign { target: id, value })
            }
            ast::Stmt::AssignTuple { targets, value } => {
                let value = self.classify_expr(value)?;
                let ids = targets.iter().map(|t| self.bind(t.clone())).collect();
                Ok(Stmt::AssignTuple {
                    targets: ids,
                    value,
                })
            }
            ast::Stmt::For {
                ivar,
                start,
                end,
                body,
            } => {
                let start = self.classify_bound(start);
                let end = self.classify_bound(end);
                // Snapshot the pre-loop top of stack for every
                // name currently in scope. After classifying the
                // body we diff against this snapshot: any name
                // whose top-of-stack changed was shadowed by a
                // body write, and needs a loop-carry entry.
                let pre: HashMap<String, LocalId> = self
                    .scope
                    .iter()
                    .filter_map(|(name, stack)| stack.last().map(|id| (name.clone(), *id)))
                    .collect();
                // Push the loop var as a fresh local scoped to the body.
                let iv_id = self.push_scope(ivar.clone());
                let body = self.classify_stmts(body)?;
                self.pop_scope(&ivar.to_string());

                // Compute loop-carry pairs.
                let mut loop_carry = Vec::new();
                for (name, outer_id) in &pre {
                    if let Some(stack) = self.scope.get(name)
                        && let Some(top) = stack.last()
                        && *top != *outer_id
                    {
                        loop_carry.push((*outer_id, *top));
                    }
                }
                // Stable ordering for determinism.
                loop_carry.sort_by_key(|(o, _)| o.0);

                Ok(Stmt::For {
                    ivar: iv_id,
                    start,
                    end,
                    body,
                    loop_carry,
                })
            }
            ast::Stmt::If {
                cond,
                then_body,
                else_body,
            } => self.classify_if(cond, then_body, else_body),
        }
    }

    /// Classify an `if { then } else { else }` statement. Both arms
    /// must bind the same set of names — violations are errors.
    /// For each such name, a fresh merge `LocalId` is introduced
    /// that subsequent reads resolve to; unroll-time dispatch
    /// populates the merge binding from whichever arm ran.
    fn classify_if(
        &mut self,
        cond: &ast::BoolExpr,
        then_body: &[ast::Stmt],
        else_body: &[ast::Stmt],
    ) -> ClassifyResult<Stmt> {
        let cond = self.classify_bool_expr(cond)?;

        // Snapshot the full scope stack so we can restore between
        // arms. `locals` and `weights` tables accrete across arms —
        // unused LocalIds are harmless.
        let pre_scope = self.scope.clone();
        let pre_tops: BTreeMap<String, Option<LocalId>> = self
            .scope
            .iter()
            .map(|(n, s)| (n.clone(), s.last().copied()))
            .collect();

        let then_body = self.classify_stmts(then_body)?;
        let post_then: BTreeMap<String, Option<LocalId>> = self
            .scope
            .iter()
            .map(|(n, s)| (n.clone(), s.last().copied()))
            .collect();

        // Restore for else arm.
        self.scope = pre_scope.clone();

        let else_body = self.classify_stmts(else_body)?;
        let post_else: BTreeMap<String, Option<LocalId>> = self
            .scope
            .iter()
            .map(|(n, s)| (n.clone(), s.last().copied()))
            .collect();

        // Restore for post-if; we install merge bindings next.
        self.scope = pre_scope;

        // A name is "changed" in an arm if its top-of-stack LocalId
        // differs from the pre-If top-of-stack.
        let then_changed = diff_tops(&pre_tops, &post_then);
        let else_changed = diff_tops(&pre_tops, &post_else);

        let then_keys: BTreeSet<&String> = then_changed.keys().collect();
        let else_keys: BTreeSet<&String> = else_changed.keys().collect();

        // Names changed in only ONE arm are allowed when they are *new*
        // arm-local scratch (not bound before the `if`): e.g. a hybrid
        // layer whose `then` arm is GDN (binds `qkv`/`z`/…) and `else` arm
        // is attention (binds `q`/`gate`/…). Such names are dead after the
        // branch — only the merged (intersection) names escape. But a name
        // that existed *before* the `if` and is reassigned in only one arm
        // is the dangerous silent-divergence case (a read after the `if`
        // would see the pre-If value when the other arm ran), so that
        // stays an error. Both arms reassigning the same pre-If name (e.g.
        // the loop-carried output) is fine — it lands in the intersection.
        let one_arm_only: Vec<&String> = then_keys
            .symmetric_difference(&else_keys)
            .copied()
            .collect();
        let dangerous: Vec<&str> = one_arm_only
            .iter()
            .filter(|n| pre_tops.get(**n).is_some_and(|b| b.is_some()))
            .map(|s| s.as_str())
            .collect();
        if !dangerous.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "`if`/`else` arms reassign name(s) bound before the `if` in only one arm: \
                     {dangerous:?}; both arms must reassign such names (a read after the `if` \
                     would otherwise see the pre-`if` value when the other arm ran). \
                     Arm-local scratch names (new in one arm, dead after) are allowed.",
                ),
            ));
        }

        // Merge over the INTERSECTION — names changed in BOTH arms escape
        // with a value from whichever arm ran. Arm-unique new names stay
        // arm-local (not re-bound after the `if`; a read after resolves
        // to the pre-If scope, or errors as out-of-scope if never bound).
        let mut merge_carry = Vec::new();
        for name in then_keys.intersection(&else_keys) {
            let then_final = then_changed[*name];
            let else_final = else_changed[*name];
            // Synthesize a merge binding that shadows any pre-If
            // binding of this name. Use `bind` so reads after the
            // If resolve to this merge id.
            let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let merge_id = self.bind(ident);
            merge_carry.push((merge_id, then_final, else_final));
        }

        Ok(Stmt::If {
            cond,
            then_body,
            else_body,
            merge_carry,
        })
    }

    fn classify_bool_expr(&self, expr: &ast::BoolExpr) -> ClassifyResult<BoolPred> {
        match expr {
            ast::BoolExpr::Modulo {
                ivar,
                divisor,
                remainder,
            } => {
                let ivar_id = self.lookup_local(ivar).ok_or_else(|| {
                    syn::Error::new(
                        ivar.span(),
                        format!(
                            "`if` condition must reference an enclosing loop variable; \
                             `{ivar}` is not in scope",
                        ),
                    )
                })?;
                Ok(BoolPred::Modulo {
                    ivar: ivar_id,
                    divisor: self.classify_bound(divisor),
                    remainder: self.classify_bound(remainder),
                })
            }
            ast::BoolExpr::NotModulo {
                ivar,
                divisor,
                remainder,
            } => {
                let ivar_id = self.lookup_local(ivar).ok_or_else(|| {
                    syn::Error::new(
                        ivar.span(),
                        format!(
                            "`if` condition must reference an enclosing loop variable; \
                             `{ivar}` is not in scope",
                        ),
                    )
                })?;
                Ok(BoolPred::NotModulo {
                    ivar: ivar_id,
                    divisor: self.classify_bound(divisor),
                    remainder: self.classify_bound(remainder),
                })
            }
            ast::BoolExpr::Less { ivar, bound } => {
                let ivar_id = self.lookup_local(ivar).ok_or_else(|| {
                    syn::Error::new(
                        ivar.span(),
                        format!(
                            "`if` condition must reference an enclosing loop variable; \
                             `{ivar}` is not in scope",
                        ),
                    )
                })?;
                Ok(BoolPred::Less {
                    ivar: ivar_id,
                    bound: self.classify_bound(bound),
                })
            }
            ast::BoolExpr::In { ivar, members } => {
                let ivar_id = self.lookup_local(ivar).ok_or_else(|| {
                    syn::Error::new(
                        ivar.span(),
                        format!(
                            "`if` condition must reference an enclosing loop variable; \
                             `{ivar}` is not in scope",
                        ),
                    )
                })?;
                Ok(BoolPred::In {
                    ivar: ivar_id,
                    members: members.clone(),
                })
            }
        }
    }

    fn classify_bound(&self, b: &BoundExpr) -> Bound {
        match b {
            BoundExpr::Lit(n) => Bound::Lit(*n),
            BoundExpr::Ident(i) => Bound::Sym(i.clone()),
        }
    }

    fn classify_expr(&mut self, expr: &ast::Expr) -> ClassifyResult<Expr> {
        match expr {
            ast::Expr::Var(ident) => self.classify_read(ident, None),

            ast::Expr::Path(segments) => {
                // Dotted paths never resolve to locals or externs
                // (those are all single-ident). They're weight refs.
                let id = self.weights.intern(segments.clone());
                Ok(Expr::Weight { id, index: None })
            }

            ast::Expr::Index { target, index } => {
                // Resolve the index (must be a local — typically a
                // loop variable).
                let index_id = self.lookup_local(index).ok_or_else(|| {
                    syn::Error::new(
                        index.span(),
                        format!(
                            "index `{index}` must refer to a local binding \
                             (usually the enclosing loop variable)",
                        ),
                    )
                })?;
                // Classify the target with the index attached.
                let target_expr = self.classify_read_target(target)?;
                attach_index(target_expr, index_id)
            }

            ast::Expr::Call { op, args } => {
                let kind = OpKind::from_name(&op.to_string())
                    .ok_or_else(|| syn::Error::new(op.span(), format!("unknown op `{op}`")))?;
                let args = args
                    .iter()
                    .map(|a| self.classify_expr(a))
                    .collect::<ClassifyResult<Vec<_>>>()?;
                Ok(Expr::Call { op: kind, args })
            }

            ast::Expr::Mul { lhs, rhs } => {
                let lhs = self.classify_expr(lhs)?;
                let rhs = self.classify_expr(rhs)?;
                Ok(Expr::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
            ast::Expr::Add { lhs, rhs } => {
                // The DSL's `+` operator lowers to an `OpKind::Add`
                // call. The fused (Add, ...) solver patterns then
                // match structurally; a ScalarLit arg flags the
                // scalar-offset variant (e.g. `w + 1.0`), while two
                // tensor args remain the ordinary residual add.
                let lhs = self.classify_expr(lhs)?;
                let rhs = self.classify_expr(rhs)?;
                Ok(Expr::Call {
                    op: OpKind::Add,
                    args: vec![lhs, rhs],
                })
            }
            ast::Expr::ScalarLit(v) => Ok(Expr::ScalarLit(*v)),
            ast::Expr::SqrtBound(ident) => Ok(Expr::SqrtBound(ident.clone())),
            ast::Expr::ConfigScalar { name, recip } => Ok(Expr::ConfigScalar {
                name: name.clone(),
                recip: *recip,
            }),
            // `reshape(...)` is only admitted as the top-level RHS of
            // an `Assign`, where `classify_stmt` intercepts it to
            // record the target shape on `Program::reshape_targets`.
            // Reaching here means the user nested it inside another
            // expression (e.g. `y = relu(reshape(x, [..]))`) — that
            // would silently lose the target shape, so reject early.
            ast::Expr::Reshape { .. } => Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`reshape(source, [..])` may only appear as the top-level RHS of an assignment",
            )),
        }
    }

    /// Classify a read that appears as the target of an indexing
    /// operation. Like `classify_expr` but only admits Var and Path
    /// shapes (a call or mul can't be indexed in the DSL).
    fn classify_read_target(&mut self, expr: &ast::Expr) -> ClassifyResult<Expr> {
        match expr {
            ast::Expr::Var(ident) => self.classify_read(ident, None),
            ast::Expr::Path(segments) => {
                let id = self.weights.intern(segments.clone());
                Ok(Expr::Weight { id, index: None })
            }
            _ => Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "indexing target must be a variable or a dotted path",
            )),
        }
    }

    fn classify_read(&mut self, ident: &Ident, index: Option<LocalId>) -> ClassifyResult<Expr> {
        let name = ident.to_string();

        // Preference order: local binding, then extern param, then
        // weight ref. A local binding shadows anything — you can't
        // write a local that happens to be named `input_ids` and
        // still read the extern. (If that ever matters we'll add a
        // parse-time warn; the issue has never arisen in practice.)
        if let Some(id) = self.lookup_local(ident) {
            return Ok(Expr::Local(id));
        }
        if let Some(kind) = ExternKind::from_name_for(&name, self.prelude) {
            return Ok(Expr::Extern { kind, index });
        }
        // Weight ref by bare ident (e.g. `embed_tokens`, `lm_head`,
        // `norm`). Intern as a single-segment path.
        let id = self.weights.intern(vec![ident.clone()]);
        Ok(Expr::Weight { id, index })
    }

    fn lookup_local(&self, ident: &Ident) -> Option<LocalId> {
        self.scope
            .get(&ident.to_string())
            .and_then(|stack| stack.last().copied())
    }

    /// Introduce a fresh local binding shadowing any previous
    /// binding of the same name. Used by assignment statements —
    /// the new binding outlives the statement.
    fn bind(&mut self, ident: Ident) -> LocalId {
        let name = ident.to_string();
        let id = self.locals.push(ident);
        // Replace the top of the stack so shadowing is observable
        // to subsequent reads at this scope depth.
        let stack = self.scope.entry(name).or_default();
        if stack.is_empty() {
            stack.push(id);
        } else {
            *stack.last_mut().unwrap() = id;
        }
        id
    }

    /// Introduce a fresh local binding scoped to the enclosing
    /// construct (used for for-loop induction variables). The
    /// caller must pair this with `pop_scope` on exit.
    fn push_scope(&mut self, ident: Ident) -> LocalId {
        let name = ident.to_string();
        let id = self.locals.push(ident);
        self.scope.entry(name).or_default().push(id);
        id
    }

    fn pop_scope(&mut self, name: &str) {
        if let Some(stack) = self.scope.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.scope.remove(name);
            }
        }
    }
}

/// Compute the set of names whose top-of-stack `LocalId` changed
/// between the `pre` snapshot and the `post` snapshot. A name is
/// changed if (a) it wasn't in pre but is in post, or (b) it was in
/// both but the top-of-stack id differs.
fn diff_tops(
    pre: &BTreeMap<String, Option<LocalId>>,
    post: &BTreeMap<String, Option<LocalId>>,
) -> BTreeMap<String, LocalId> {
    let mut out = BTreeMap::new();
    for (name, post_top) in post {
        let pre_top = pre.get(name).copied().flatten();
        if let Some(post_id) = *post_top
            && pre_top != Some(post_id)
        {
            out.insert(name.clone(), post_id);
        }
    }
    out
}

/// Attach an index to a Weight or Extern expression; error on
/// anything else (you can't index a Call, a Mul, or a Local).
fn attach_index(expr: Expr, index: LocalId) -> ClassifyResult<Expr> {
    match expr {
        Expr::Weight { id, index: None } => Ok(Expr::Weight {
            id,
            index: Some(index),
        }),
        Expr::Extern { kind, index: None } => Ok(Expr::Extern {
            kind,
            index: Some(index),
        }),
        Expr::Local(_) => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cannot index a local binding",
        )),
        // Double-indexing or indexing a call/mul shouldn't reach here
        // because the parser's Index wraps one expression and the
        // target path produces at most one index at a time.
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("cannot index expression: {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::WeightId;
    use crate::parse::parse_block;

    fn classify_src(src: &str) -> Program {
        let file: syn::File =
            syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("syntactic parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("DSL parse");
        classify_with(&ast, Prelude::Decoder).expect("classification")
    }

    fn classify_err(src: &str) -> syn::Error {
        let file: syn::File = syn::parse_str(&format!("fn _carrier() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("DSL parse");
        classify_with(&ast, Prelude::Decoder).expect_err("expected classification error")
    }

    fn classify_vision_src(src: &str) -> Program {
        let file: syn::File =
            syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("syntactic parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("DSL parse");
        classify_with(&ast, Prelude::Vision).expect("classification")
    }

    #[test]
    fn simple_local_binding() {
        let p = classify_src("x = embed(input_ids, embed_tokens);");
        match &p.statements[0] {
            Stmt::Assign { target, value } => {
                assert_eq!(p.locals.name(*target).to_string(), "x");
                match value {
                    Expr::Call { op, args } => {
                        assert_eq!(*op, OpKind::Embed);
                        // arg 0: input_ids → ExternKind::InputIds
                        assert!(matches!(
                            args[0],
                            Expr::Extern {
                                kind: ExternKind::InputIds,
                                index: None
                            }
                        ));
                        // arg 1: embed_tokens → a single-segment weight
                        match &args[1] {
                            Expr::Weight { id, index: None } => {
                                let path = p.weights.path(*id);
                                assert_eq!(path.len(), 1);
                                assert_eq!(path[0], "embed_tokens");
                            }
                            other => panic!("expected weight, got {other:?}"),
                        }
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn local_shadowing() {
        // Second `x = ...` must introduce a NEW LocalId, and the
        // read inside the call must use the PREVIOUS id.
        let p = classify_src("x = embed(input_ids, embed_tokens); x = add(x, x);");
        let (t0, v0_call) = match &p.statements[0] {
            Stmt::Assign {
                target,
                value: Expr::Call { args, .. },
            } => (*target, args.clone()),
            _ => panic!(),
        };
        let (t1, v1) = match &p.statements[1] {
            Stmt::Assign { target, value } => (*target, value.clone()),
            _ => panic!(),
        };

        assert_ne!(t0, t1, "second x must be a fresh LocalId");

        match v1 {
            Expr::Call {
                op: OpKind::Add,
                args,
            } => {
                // Both args are reads of the PREVIOUS x (t0).
                match (&args[0], &args[1]) {
                    (Expr::Local(a), Expr::Local(b)) => {
                        assert_eq!(*a, t0, "first add arg should read old x");
                        assert_eq!(*b, t0, "second add arg should read old x");
                    }
                    _ => panic!("expected two local reads"),
                }
            }
            _ => panic!("expected add call"),
        }

        // First statement's args aren't locals; this just proves
        // the helper vector v0_call above is usable.
        let _ = v0_call;
    }

    #[test]
    fn undefined_read_is_error() {
        // `mystery_var` is not a local, not a fixed extern, and
        // would therefore classify as a bare-ident weight. So it's
        // NOT an error — everything unknown becomes a weight. This
        // test makes that behavior explicit.
        let p = classify_src("x = embed(mystery_var, embed_tokens);");
        match &p.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[0] {
                Expr::Weight { id, index: None } => {
                    let path = p.weights.path(*id);
                    assert_eq!(path[0], "mystery_var");
                }
                other => panic!("expected bare-ident weight, got {other:?}"),
            },
            _ => panic!(),
        }

        // Indexing with an undefined variable IS an error (the
        // index must be a local).
        let err = classify_err("x = embed(input_ids[mystery_idx], embed_tokens);");
        let msg = err.to_string();
        assert!(
            msg.contains("mystery_idx"),
            "error should mention bad index: {msg}"
        );
    }

    #[test]
    fn for_loop_scopes_induction_var() {
        let p = classify_src(
            "for layer in 0..num_hidden_layers { x = embed(kv_cache[layer], embed_tokens); }",
        );
        match &p.statements[0] {
            Stmt::For { ivar, body, .. } => {
                // `layer` inside the body must resolve to the ivar
                // LocalId.
                match &body[0] {
                    Stmt::Assign {
                        value: Expr::Call { args, .. },
                        ..
                    } => match &args[0] {
                        Expr::Extern {
                            kind: ExternKind::KvCache,
                            index: Some(idx),
                        } => {
                            assert_eq!(*idx, *ivar, "index must resolve to loop var");
                        }
                        other => panic!("expected indexed kv_cache, got {other:?}"),
                    },
                    _ => panic!(),
                }
            }
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn weight_interning_dedupes() {
        // Two reads of the same dotted-path weight should produce
        // the same WeightId.
        let p = classify_src(
            "for layer in 0..num_hidden_layers { \
                q = gemm(q, self_attn.q_proj[layer]); \
                q = gemm(q, self_attn.q_proj[layer]); \
            }",
        );
        match &p.statements[0] {
            Stmt::For { body, .. } => {
                let w0 = weight_of_second_arg(&body[0]);
                let w1 = weight_of_second_arg(&body[1]);
                assert_eq!(w0, w1, "same dotted path should intern to same id");
            }
            _ => panic!(),
        }
    }

    fn weight_of_second_arg(stmt: &Stmt) -> WeightId {
        match stmt {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[1] {
                Expr::Weight { id, .. } => *id,
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_op_is_error() {
        let err = classify_err("x = frobnicate(a, b);");
        assert!(err.to_string().contains("unknown op"));
        assert!(err.to_string().contains("frobnicate"));
    }

    #[test]
    fn if_merge_carry_is_emitted() {
        let p = classify_src(
            "for layer in 0..4 { \
                if layer % 2 == 0 { attn = attention(q, k, v, kv_cache, block_table); } \
                else { attn = sliding_attention(q, k, v, kv_cache, block_table); } \
                hidden_states = add(attn, attn); \
            }",
        );
        match &p.statements[0] {
            Stmt::For { body, .. } => match &body[0] {
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                    merge_carry,
                } => {
                    // Both arms bind exactly `attn`.
                    assert_eq!(merge_carry.len(), 1, "one merged name (attn)");
                    let (merge_id, then_final, else_final) = merge_carry[0];
                    assert_ne!(merge_id, then_final);
                    assert_ne!(merge_id, else_final);
                    assert_ne!(then_final, else_final);
                    assert_eq!(p.locals.name(merge_id).to_string(), "attn");
                    assert_eq!(p.locals.name(then_final).to_string(), "attn");
                    assert_eq!(p.locals.name(else_final).to_string(), "attn");
                    match cond {
                        BoolPred::Modulo {
                            divisor, remainder, ..
                        } => {
                            assert!(matches!(divisor, Bound::Lit(2)));
                            assert!(matches!(remainder, Bound::Lit(0)));
                        }
                        _ => panic!("expected Modulo"),
                    }
                    // Each arm has exactly one assignment.
                    assert_eq!(then_body.len(), 1);
                    assert_eq!(else_body.len(), 1);
                }
                _ => panic!("expected If"),
            },
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn if_in_literal_array_classifies_to_in_pred() {
        let p = classify_src(
            "for layer in 0..32 { \
                if [7, 15, 23, 31].contains(&layer) { \
                    attn = attention(q, k, v, kv_cache, block_table); \
                } else { \
                    attn = sliding_attention(q, k, v, kv_cache, block_table); \
                } \
                hidden_states = add(attn, attn); \
            }",
        );
        match &p.statements[0] {
            Stmt::For { body, .. } => match &body[0] {
                Stmt::If { cond, .. } => match cond {
                    BoolPred::In { ivar, members } => {
                        assert_eq!(p.locals.name(*ivar).to_string(), "layer");
                        assert_eq!(members, &vec![7u64, 15, 23, 31]);
                    }
                    other => panic!("expected In, got {other:?}"),
                },
                _ => panic!("expected If"),
            },
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn if_in_with_unbound_ivar_is_rejected() {
        let err = classify_err(
            "for layer in 0..32 { \
                if [7, 15].contains(&ghost) { \
                    attn = attention(q, k, v, kv_cache, block_table); \
                } else { \
                    attn = sliding_attention(q, k, v, kv_cache, block_table); \
                } \
            }",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("not in scope") || msg.contains("loop variable"),
            "error mentions scope/loop-var requirement: {msg}"
        );
    }

    #[test]
    fn if_arms_unique_new_scratch_names_allowed() {
        // Hybrid-layer pattern: the two arms are structurally different
        // token mixers that bind disjoint *new* scratch names (`a` only in
        // `then`, `b` only in `else`). Both are dead after the branch, so
        // this is allowed — only the intersection escapes (here empty).
        let p = classify_src(
            "for layer in 0..4 { \
                if layer % 2 == 0 { a = attention(q, k, v, kv_cache, block_table); } \
                else { b = sliding_attention(q, k, v, kv_cache, block_table); } \
            }",
        );
        match &p.statements[0] {
            Stmt::For { body, .. } => match &body[0] {
                Stmt::If { merge_carry, .. } => {
                    // Disjoint new names → empty intersection → no merge.
                    assert!(
                        merge_carry.is_empty(),
                        "arm-unique scratch names should not be merged"
                    );
                }
                _ => panic!("expected If"),
            },
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn if_arm_reassigns_pre_if_name_in_one_arm_is_rejected() {
        // `x` is bound *before* the `if` and reassigned in only the `then`
        // arm — the dangerous silent-divergence case (a read after the `if`
        // would see the pre-`if` value when the `else` arm ran). Rejected.
        let err = classify_err(
            "for layer in 0..4 { \
                x = attention(q, k, v, kv_cache, block_table); \
                if layer % 2 == 0 { x = sliding_attention(q, k, v, kv_cache, block_table); } \
                else { y = attention(q, k, v, kv_cache, block_table); } \
                hidden_states = add(x, x); \
            }",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("only one arm") || msg.contains("bound before"),
            "error flags a pre-`if` name reassigned in one arm only: {msg}"
        );
    }

    #[test]
    fn if_condition_with_unbound_ivar_is_rejected() {
        // `ghost` is not a local, not an extern. The classifier
        // would normally turn it into a weight ref on the RHS of an
        // assignment, but inside a predicate it must resolve to a
        // loop local — so this is rejected.
        let err = classify_err(
            "for layer in 0..4 { \
                if ghost % 2 == 0 { x = attention(q, k, v, kv_cache, block_table); } \
                else { x = sliding_attention(q, k, v, kv_cache, block_table); } \
            }",
        );
        let msg = err.to_string();
        assert!(msg.contains("ghost"), "error names unbound ivar: {msg}");
    }

    #[test]
    fn realistic_llama_body_classifies_fully() {
        let p = classify_src(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn = attention(q, k, v, kv_cache[layer], block_table);
                oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                up = gemm(normed2, mlp.up_proj[layer]);
                down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
            "#,
        );

        // Weights interned: input_layernorm, q_proj, k_proj, v_proj,
        // o_proj, post_attention_layernorm, gate_proj, up_proj,
        // down_proj, embed_tokens, norm, lm_head = 12 distinct.
        //   self_attn.{q,k,v,o}_proj = 4
        //   mlp.{gate,up,down}_proj = 3
        //   input_layernorm, post_attention_layernorm = 2
        //   embed_tokens, norm, lm_head = 3
        //   TOTAL = 12.
        assert_eq!(p.weights.len(), 12, "unique weights interned");
    }

    // ── Vision prelude routing (G.3) ─────────────────────────────

    #[test]
    fn vision_prelude_recognizes_pixels_extern() {
        // Under the vision prelude, `pixels` is the encoder's data
        // input — analogous to `input_ids` under the decoder prelude.
        // Classify must tag it as `ExternKind::Pixels`, not as a
        // weight ref.
        let p = classify_vision_src("x = layer_norm(pixels, ln_q);");
        match &p.statements[0] {
            Stmt::Assign { value, .. } => match value {
                Expr::Call { args, .. } => assert!(
                    matches!(
                        args[0],
                        Expr::Extern {
                            kind: ExternKind::Pixels,
                            index: None
                        }
                    ),
                    "pixels resolves to ExternKind::Pixels under vision prelude, got {:?}",
                    args[0],
                ),
                other => panic!("expected call, got {other:?}"),
            },
            other => panic!("expected assign, got {other:?}"),
        }
    }

    #[test]
    fn vision_prelude_recognizes_full_vision_extern_set() {
        // All six vision externs map: pixels, cu_seqlens, cos, sin,
        // grid_thw, max_seqlen. Pin the routing in one place so a
        // future shape-signature edit doesn't silently relabel any
        // of them as weights.
        let cases: &[(&str, ExternKind)] = &[
            ("pixels", ExternKind::Pixels),
            ("cu_seqlens", ExternKind::CuSeqlens),
            ("cos", ExternKind::Cos),
            ("sin", ExternKind::Sin),
            ("grid_thw", ExternKind::GridThw),
            ("max_seqlen", ExternKind::MaxSeqlen),
        ];
        for (name, expected) in cases {
            assert_eq!(
                ExternKind::from_name_for(name, Prelude::Vision),
                Some(*expected),
                "vision extern `{name}` routes",
            );
            assert_eq!(
                ExternKind::from_name_for(name, Prelude::Decoder),
                None,
                "vision extern `{name}` is invisible under decoder prelude",
            );
        }
    }

    #[test]
    fn decoder_externs_invisible_under_vision_prelude() {
        // Disjoint preludes: `input_ids` is unrecognized in a vision
        // body. With no extern match, classify falls through to the
        // weight-ref path and interns it as a single-segment weight
        // — observable as a non-zero weight count for a body that
        // only writes the would-be extern.
        let p = classify_vision_src("x = layer_norm(input_ids, ln);");
        match &p.statements[0] {
            Stmt::Assign { value, .. } => match value {
                Expr::Call { args, .. } => match &args[0] {
                    Expr::Weight { .. } => { /* expected: weight ref, not extern */ }
                    other => panic!(
                        "input_ids must NOT resolve to an extern under vision prelude, got {other:?}",
                    ),
                },
                other => panic!("expected call, got {other:?}"),
            },
            other => panic!("expected assign, got {other:?}"),
        }
    }

    #[test]
    fn dsl_authored_reshape_records_target_shape() {
        // `reshape(x, [num_tokens, vision_embed_dim])` — the G.5.c
        // "literal-or-bound" surface. The classified Stmt must match
        // the shape of `apply_reshape_hints`'s synthesized form: a
        // Reshape OpKind call with the source as its only arg, plus
        // an entry in `program.reshape_targets` keyed by the new
        // local. Bound names round-trip as `Dim::Bound(name)`,
        // literals as `Dim::Lit(n)`.
        let p = classify_vision_src("y = reshape(pixels, [num_tokens, vision_embed_dim]);");
        let (target, call_args) = match &p.statements[0] {
            Stmt::Assign {
                target,
                value: Expr::Call { op, args },
            } => {
                assert_eq!(*op, OpKind::Reshape, "must lower to OpKind::Reshape");
                (*target, args.clone())
            }
            other => panic!("expected Stmt::Assign with reshape call, got {other:?}"),
        };
        assert_eq!(p.locals.name(target).to_string(), "y");
        // Source arg preserved verbatim — `pixels` resolves to the
        // vision-prelude extern.
        assert_eq!(call_args.len(), 1);
        assert!(matches!(
            call_args[0],
            Expr::Extern {
                kind: ExternKind::Pixels,
                index: None,
            }
        ));
        // Target shape recorded.
        let dims = p
            .reshape_targets
            .get(&target)
            .expect("reshape_targets entry");
        assert_eq!(dims.len(), 2);
        match &dims[0] {
            Dim::Bound(name) => assert_eq!(name, "num_tokens"),
            other => panic!("expected Bound, got {other:?}"),
        }
        match &dims[1] {
            Dim::Bound(name) => assert_eq!(name, "vision_embed_dim"),
            other => panic!("expected Bound, got {other:?}"),
        }
    }

    #[test]
    fn dsl_authored_reshape_with_literal_dim() {
        // Literal dims survive as `Dim::Lit(n)`.
        let p = classify_vision_src("y = reshape(pixels, [num_tokens, 1280]);");
        let target = match &p.statements[0] {
            Stmt::Assign { target, .. } => *target,
            _ => panic!(),
        };
        let dims = p.reshape_targets.get(&target).unwrap();
        assert!(matches!(dims[0], Dim::Bound(ref n) if n == "num_tokens"));
        assert!(matches!(dims[1], Dim::Lit(1280)));
    }

    #[test]
    fn dsl_authored_reshape_records_arithmetic_dims() {
        // G.5.f.a — reshape DimSpec arithmetic. The merger reshape
        // wants `[num_tokens / vision_merge_factor, vision_merge_hidden]`;
        // a multiplicative form `[num_tokens, vision_embed_dim *
        // vision_merge_factor]` must round-trip too.
        let p = classify_vision_src(
            "y = reshape(pixels, [num_tokens / vision_merge_factor, vision_merge_hidden]);",
        );
        let target = match &p.statements[0] {
            Stmt::Assign { target, .. } => *target,
            _ => panic!(),
        };
        let dims = p.reshape_targets.get(&target).unwrap();
        match &dims[0] {
            Dim::Div(num, den) => {
                assert!(matches!(num.as_ref(), Dim::Bound(n) if n == "num_tokens"));
                assert!(matches!(den.as_ref(), Dim::Bound(n) if n == "vision_merge_factor"));
            }
            other => panic!("expected Dim::Div for first dim, got {other:?}"),
        }
        assert!(matches!(&dims[1], Dim::Bound(n) if n == "vision_merge_hidden"));

        let p = classify_vision_src(
            "y = reshape(pixels, [num_tokens, vision_embed_dim * vision_merge_factor]);",
        );
        let target = match &p.statements[0] {
            Stmt::Assign { target, .. } => *target,
            _ => panic!(),
        };
        let dims = p.reshape_targets.get(&target).unwrap();
        // Second dim becomes `Dim::Mul([vision_embed_dim,
        // vision_merge_factor])` after canonical_mul sorts by name.
        match &dims[1] {
            Dim::Mul(factors) => {
                let names: Vec<_> = factors
                    .iter()
                    .filter_map(|d| match d {
                        Dim::Bound(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(names, vec!["vision_embed_dim", "vision_merge_factor"]);
            }
            other => panic!("expected Dim::Mul for second dim, got {other:?}"),
        }
    }

    #[test]
    fn nested_reshape_is_rejected() {
        // Nesting would silently lose the target shape (no LocalId
        // to key reshape_targets on), so it must error at classify
        // time, not produce wrong code at codegen.
        let err =
            classify_err("y = layer_norm(reshape(hidden_states, [num_tokens, hidden_size]), ln);");
        assert!(
            err.to_string().contains("top-level RHS of an assignment"),
            "expected top-level-RHS error, got: {err}"
        );
    }

    #[test]
    fn vision_externs_invisible_under_decoder_prelude() {
        // Mirror of the above: `pixels` does not exist under the
        // decoder prelude — it lands as a weight ref, not an extern.
        let p = classify_src("x = layer_norm(pixels, ln);");
        match &p.statements[0] {
            Stmt::Assign { value, .. } => match value {
                Expr::Call { args, .. } => match &args[0] {
                    Expr::Weight { .. } => { /* expected */ }
                    other => panic!(
                        "pixels must NOT resolve to an extern under decoder prelude, got {other:?}",
                    ),
                },
                other => panic!("expected call, got {other:?}"),
            },
            other => panic!("expected assign, got {other:?}"),
        }
    }
}
