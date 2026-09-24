// SPDX-License-Identifier: Apache-2.0
//! `ruff_python_ast` → [`Ast`] — the Python costume for the DSL.
//!
//! Same contract as [`crate::parse`]: walk the carrier `def` body,
//! recognize the DSL's statement and expression shapes, reject
//! everything else. The output types are IDENTICAL to the syn-based
//! parser's, so [`crate::classify`] and everything downstream cannot
//! tell which costume a file wore. Identifiers are minted as
//! `syn::Ident`s at the call-site span (the arch name), because the
//! AST is typed in `syn::Ident` and downstream error reporting uses
//! their spans.
//!
//! Dialect (a strict subset of Python):
//!   - `@forward` / `@vision_forward(...)` decorator on the carrier
//!     `def` — the same metadata the `#[forward]` attribute carried:
//!     `workloads = [...]`, `sk_buckets = [...]`, `pixel_pack = path`,
//!     `processor = path`.
//!   - `name = expr`            → [`Stmt::Assign`]
//!   - `(a, b) = expr`          → [`Stmt::AssignTuple`]
//!   - `for ivar in range(b):`  → [`Stmt::For`]
//!   - `if <pred>:` / `else:`   → [`Stmt::If`]
//!   - exprs: calls, attribute chains, `[ivar]` indexing, `*`/`+`,
//!     numeric literals, list literals (only inside reshape / the
//!     `in` predicate).
//!
//! The `import` lines are permitted (and ignored) so the file is
//! executable Python for the CI oracle; the decorated `def` is the
//! carrier. Anything else — classes, comprehensions, walrus, star-
//! args, keyword args, `while`, `try` — is a loud error, the same
//! no-silent-typos property the Rust costume enforces.

use ruff_python_ast as py;
use ruff_python_parser::parse_module;

use crate::ast::{Ast, BoolExpr, BoundExpr, DimSpec, Expr, Stmt};

/// The carrier `def` plus its decorator metadata — the Python
/// costume's equivalent of the `#[forward]`-annotated `ItemFn` that
/// [`crate::parse_carrier`] returns for the Rust costume.
pub struct PythonCarrier {
    /// The parsed DSL body.
    pub ast: Ast,
    /// The arch name (the `def`'s name).
    pub arch_name: String,
    /// True for `@vision_forward`, false for `@forward`.
    pub vision: bool,
    /// `workloads = [...]` from the decorator (empty = default ladder).
    pub workloads: Vec<u64>,
    /// `sk_buckets = [...]` from the decorator (empty = default ladder).
    pub sk_buckets: Vec<u64>,
    /// `processor = <path>` from the decorator (vision carriers).
    pub processor: Option<String>,
    /// `pixel_pack = <path>` from the decorator (vision carriers).
    pub pixel_pack: Option<String>,
}

/// Parse a `.py` carrier file: the `@forward`/`@vision_forward`-decorated
/// top-level `def`, its decorator metadata, and its body as DSL statements.
pub fn parse_python_file(
    text: &str,
    name_span: proc_macro2::Span,
) -> Result<PythonCarrier, String> {
    let parsed = parse_module(text).map_err(|e| format!("python parse: {e}"))?;
    let mut carrier: Option<&py::StmtFunctionDef> = None;
    for stmt in parsed.syntax().body.iter() {
        match stmt {
            // Imports are the oracle's business, not ours.
            py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => {}
            py::Stmt::FunctionDef(f) => {
                if carrier.is_some() {
                    return Err("carrier file must contain exactly one `def`".to_string());
                }
                if f.is_async {
                    return Err("the carrier `def` cannot be `async`".to_string());
                }
                carrier = Some(f);
            }
            other => {
                return Err(format!(
                    "carrier file top level must be imports + one `def` — found {:?}",
                    py::AnyNodeRef::from(other).kind()
                ));
            }
        }
    }
    let f = carrier.ok_or("carrier file has no `def` (the forward body)")?;

    // Decorator metadata: exactly one of `@forward` / `@vision_forward`,
    // carrying the same kwargs the attribute costume did.
    let mut vision = false;
    let mut workloads = Vec::new();
    let mut sk_buckets = Vec::new();
    let mut processor = None;
    let mut pixel_pack = None;
    if f.decorator_list.is_empty() {
        return Err(
            "carrier `def` must be decorated with `@forward` or `@vision_forward(...)`".to_string(),
        );
    }
    if f.decorator_list.len() > 1 {
        return Err("carrier `def` carries multiple decorators".to_string());
    }
    let dec = &f.decorator_list[0];
    match &dec.expression {
        py::Expr::Name(n) if n.id == "forward" => {}
        py::Expr::Name(n) if n.id == "vision_forward" => vision = true,
        py::Expr::Call(c) => {
            let name = match c.func.as_ref() {
                py::Expr::Name(n) => n.id.as_str().to_string(),
                _ => {
                    return Err(
                        "decorator must be `@forward` or `@vision_forward(...)`".to_string()
                    );
                }
            };
            match name.as_str() {
                "forward" => {}
                "vision_forward" => vision = true,
                _ => {
                    return Err(format!(
                        "unknown decorator `@{name}` — expected `@forward` / `@vision_forward`"
                    ));
                }
            }
            if !c.arguments.args.is_empty() {
                return Err("decorator takes keyword arguments only".to_string());
            }
            for kw in &c.arguments.keywords {
                let key = kw
                    .arg
                    .as_ref()
                    .map(|a| a.as_str().to_string())
                    .ok_or("decorator kwargs must be named")?;
                let value_is_list = matches!(&kw.value, py::Expr::List(_));
                let value_is_path = matches!(&kw.value, py::Expr::Attribute(_));
                match (key.as_str(), value_is_list, value_is_path) {
                    ("workloads", true, _) => workloads = parse_int_list_kw(&key, &kw.value)?,
                    ("sk_buckets", true, _) => sk_buckets = parse_int_list_kw(&key, &kw.value)?,
                    ("processor", _, true) => processor = Some(parse_path_kw(&kw.value)?),
                    ("pixel_pack", _, true) => pixel_pack = Some(parse_path_kw(&kw.value)?),
                    _ => {
                        return Err(format!(
                            "unknown or ill-typed decorator argument `{key}` — expected \
                             `workloads = [..]`, `sk_buckets = [..]`, `processor = path`, \
                             or `pixel_pack = path`"
                        ));
                    }
                }
            }
        }
        _ => {
            return Err(
                "carrier `def` must be decorated with `@forward` or `@vision_forward(...)`"
                    .to_string(),
            );
        }
    }

    let mut statements = Vec::with_capacity(f.body.len());
    for (i, s) in f.body.iter().enumerate() {
        // `return <ident>` is the torch costume for "this value is
        // the output" — the Rust costume says it by simply being the
        // last statement. Admit it only as the terminal statement,
        // and only when it names the value the last assign bound.
        if let py::Stmt::Return(r) = s {
            if i + 1 != f.body.len() {
                return Err("`return` may only appear as the final statement".to_string());
            }
            let name = match r.value.as_deref() {
                Some(py::Expr::Name(n)) => n.id.as_str(),
                _ => return Err("`return <ident>` must name the forward's final value".to_string()),
            };
            match statements.last() {
                Some(Stmt::Assign { target, .. }) if target == name => {}
                _ => {
                    return Err(format!(
                        "`return {name}` must name the value the preceding statement assigned"
                    ));
                }
            }
            break;
        }
        // A bare string literal is a docstring — skipped, not math.
        if let py::Stmt::Expr(e) = s
            && matches!(&*e.value, py::Expr::StringLiteral(_))
        {
            continue;
        }
        statements.push(parse_stmt(s, name_span)?);
    }
    Ok(PythonCarrier {
        ast: Ast { statements },
        arch_name: f.name.id.as_str().to_string(),
        vision,
        workloads,
        sk_buckets,
        processor,
        pixel_pack,
    })
}

/// `workloads = [256, 1024, ...]` — a list of non-negative int literals.
fn parse_int_list_kw(key: &str, value: &py::Expr) -> Result<Vec<u64>, String> {
    let py::Expr::List(l) = value else {
        unreachable!("caller checked the value is a list literal")
    };
    let mut vals = Vec::with_capacity(l.elts.len());
    for el in &l.elts {
        let py::Expr::NumberLiteral(n) = el else {
            return Err(format!(
                "decorator `{key}` must be a list of integer literals"
            ));
        };
        let py::Number::Int(i) = &n.value else {
            return Err(format!(
                "decorator `{key}` must be a list of integer literals"
            ));
        };
        let v = i
            .as_u64()
            .ok_or_else(|| format!("decorator `{key}` values must be non-negative integers"))?;
        vals.push(v);
    }
    Ok(vals)
}

/// `processor = crate::PROCESSOR` — a plain identifier chain flattened
/// to a `::`-joined path string.
fn parse_path_kw(value: &py::Expr) -> Result<String, String> {
    let mut segs = Vec::new();
    collect_path_segments_ref(value, &mut segs)?;
    Ok(segs.join("::"))
}

fn collect_path_segments_ref(expr: &py::Expr, out: &mut Vec<String>) -> Result<(), String> {
    match expr {
        py::Expr::Name(n) => {
            out.push(n.id.as_str().to_string());
            Ok(())
        }
        py::Expr::Attribute(a) => {
            collect_path_segments_ref(&a.value, out)?;
            out.push(a.attr.id.as_str().to_string());
            Ok(())
        }
        _ => Err("decorator path must be a plain identifier chain".to_string()),
    }
}

fn ident(name: &str, at: proc_macro2::Span) -> syn::Ident {
    syn::Ident::new(name, at)
}

fn parse_stmt(stmt: &py::Stmt, at: proc_macro2::Span) -> Result<Stmt, String> {
    match stmt {
        py::Stmt::Assign(a) => {
            if a.targets.len() != 1 {
                return Err("chained assignment is not part of the DSL".to_string());
            }
            let value = parse_expr(&a.value, at)?;
            match &a.targets[0] {
                py::Expr::Name(n) => Ok(Stmt::Assign {
                    target: ident(&n.id, at),
                    value,
                }),
                py::Expr::Tuple(t) => {
                    let mut targets = Vec::with_capacity(t.elts.len());
                    for el in &t.elts {
                        match el {
                            py::Expr::Name(n) => targets.push(ident(&n.id, at)),
                            _ => {
                                return Err(
                                    "tuple destructuring targets must be plain identifiers"
                                        .to_string(),
                                );
                            }
                        }
                    }
                    Ok(Stmt::AssignTuple { targets, value })
                }
                _ => Err(
                    "assignment target must be an identifier or a tuple of identifiers".to_string(),
                ),
            }
        }
        py::Stmt::For(f) => {
            if f.is_async {
                return Err("`async for` is not part of the DSL".to_string());
            }
            let ivar = match f.target.as_ref() {
                py::Expr::Name(n) => ident(&n.id, at),
                _ => return Err("for-loop variable must be a plain identifier".to_string()),
            };
            // `range(start, end)` / `range(end)` (start defaults 0).
            let (start, end) = match f.iter.as_ref() {
                py::Expr::Call(c) => {
                    let is_range = matches!(
                        c.func.as_ref(),
                        py::Expr::Name(n) if n.id == "range"
                    );
                    if !is_range {
                        return Err("for-loop must iterate `range(<bound>)`".to_string());
                    }
                    match c.arguments.args.len() {
                        1 => (BoundExpr::Lit(0), parse_bound(&c.arguments.args[0], at)?),
                        2 => (
                            parse_bound(&c.arguments.args[0], at)?,
                            parse_bound(&c.arguments.args[1], at)?,
                        ),
                        n => return Err(format!("range(...) takes 1 or 2 bounds, got {n}")),
                    }
                }
                _ => return Err("for-loop must iterate `range(<bound>)`".to_string()),
            };
            let body = parse_stmts(&f.body, at)?;
            if !f.orelse.is_empty() {
                return Err("`for ... else` is not part of the DSL".to_string());
            }
            Ok(Stmt::For {
                ivar,
                start,
                end,
                body,
            })
        }
        py::Stmt::If(i) => {
            let cond = parse_bool_expr(i.test.as_ref(), at)?;
            let then_body = parse_stmts(&i.body, at)?;
            let else_body = if i.elif_else_clauses.iter().any(|c| c.test.is_some()) {
                return Err("`elif` is not supported; nest `if` inside `else:`".to_string());
            } else if let Some(els) = i.elif_else_clauses.first() {
                parse_stmts(&els.body, at)?
            } else {
                return Err(
                    "`if` must have an `else:` arm; both arms must bind the same names".to_string(),
                );
            };
            Ok(Stmt::If {
                cond,
                then_body,
                else_body,
            })
        }
        other => Err(format!(
            "statement `{:?}` is not part of the DSL dialect",
            py::AnyNodeRef::from(other).kind()
        )),
    }
}

fn parse_stmts(stmts: &[py::Stmt], at: proc_macro2::Span) -> Result<Vec<Stmt>, String> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        // A bare string literal is a docstring — the oracle's
        // documentation, not math. Everything else is DSL or error.
        if let py::Stmt::Expr(e) = s
            && matches!(&*e.value, py::Expr::StringLiteral(_))
        {
            continue;
        }
        out.push(parse_stmt(s, at)?);
    }
    Ok(out)
}

fn parse_bool_expr(expr: &py::Expr, at: proc_macro2::Span) -> Result<BoolExpr, String> {
    // `[7, 15, 23, 31].contains(layer)` — set membership. (Torch
    // idiom is `in`; the Rust costume spells `.contains`. Accept the
    // `in` operator form: `layer in [7, 15, ...]`.)
    if let py::Expr::Compare(c) = expr
        && let Some((left, op, right)) = c.as_single()
    {
        let left = left as &py::Expr;
        let right = right as &py::Expr;
        match *op {
            py::CmpOp::In => {
                let ivar = parse_ivar(left, at)?;
                let members = parse_int_list(right)?;
                return Ok(BoolExpr::In { ivar, members });
            }
            py::CmpOp::Eq | py::CmpOp::NotEq => {
                // `ivar % divisor == remainder` / `!=`.
                if let py::Expr::BinOp(inner) = left
                    && inner.op == py::Operator::Mod
                {
                    let ivar = parse_ivar(inner.left.as_ref(), at)?;
                    let divisor = parse_bound(inner.right.as_ref(), at)?;
                    let remainder = parse_bound(right, at)?;
                    return Ok(if *op == py::CmpOp::Eq {
                        BoolExpr::Modulo {
                            ivar,
                            divisor,
                            remainder,
                        }
                    } else {
                        BoolExpr::NotModulo {
                            ivar,
                            divisor,
                            remainder,
                        }
                    });
                }
            }
            py::CmpOp::Lt => {
                let ivar = parse_ivar(left, at)?;
                let bound = parse_bound(right, at)?;
                return Ok(BoolExpr::Less { ivar, bound });
            }
            _ => {}
        }
    }
    Err(
        "`if` condition must be `ivar % N == M`, `ivar % N != M`, `ivar < N`, \
        or `ivar in [lit, lit, ...]`"
            .to_string(),
    )
}

fn parse_ivar(expr: &py::Expr, at: proc_macro2::Span) -> Result<syn::Ident, String> {
    match expr {
        py::Expr::Name(n) => Ok(ident(&n.id, at)),
        _ => Err("expected a plain identifier (the enclosing loop induction variable)".to_string()),
    }
}

fn parse_int_list(expr: &py::Expr) -> Result<Vec<u64>, String> {
    match expr {
        py::Expr::List(l) => l
            .elts
            .iter()
            .map(|e| match e {
                py::Expr::NumberLiteral(n) => match &n.value {
                    py::Number::Int(i) => i.as_u64().ok_or_else(|| {
                        "`in` list elements must be non-negative integers".to_string()
                    }),
                    _ => Err("`in` list elements must be integer literals".to_string()),
                },
                _ => Err("`in` list elements must be integer literals".to_string()),
            })
            .collect(),
        _ => Err("`in` predicate requires a list literal `[lit, lit, ...]`".to_string()),
    }
}

fn parse_bound(expr: &py::Expr, at: proc_macro2::Span) -> Result<BoundExpr, String> {
    match expr {
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i
                .as_u64()
                .map(BoundExpr::Lit)
                .ok_or_else(|| "loop bound must be a non-negative integer".to_string()),
            _ => Err("loop bound must be an integer literal or a bare identifier".to_string()),
        },
        py::Expr::Name(n) => Ok(BoundExpr::Ident(ident(&n.id, at))),
        _ => Err("loop bound must be an integer literal or a bare identifier".to_string()),
    }
}

fn parse_expr(expr: &py::Expr, at: proc_macro2::Span) -> Result<Expr, String> {
    match expr {
        // A simple identifier — `hidden_states`, `input_ids`.
        py::Expr::Name(n) => Ok(Expr::Var(ident(&n.id, at))),

        // Attribute chains — `self_attn.q_proj`. A single-segment
        // chain is impossible in Python (the root is always a Name),
        // so this is always a Path of >= 2.
        py::Expr::Attribute(_) => {
            let segments = collect_path_segments(expr, at)?;
            Ok(Expr::Path(segments))
        }

        // Indexing: `self_attn.q_proj[layer]`.
        py::Expr::Subscript(s) => {
            let target = parse_expr(&s.value, at)?;
            let index = match s.slice.as_ref() {
                py::Expr::Name(n) => ident(&n.id, at),
                _ => {
                    return Err(
                        "index must be a single identifier (the enclosing loop variable)"
                            .to_string(),
                    );
                }
            };
            Ok(Expr::Index {
                target: Box::new(target),
                index,
            })
        }

        // Call: `gemm(x, w)`, `rmsnorm(x, w)`, …
        py::Expr::Call(c) => {
            if !c.arguments.keywords.is_empty() {
                return Err("keyword arguments are not part of the DSL".to_string());
            }
            let op = match c.func.as_ref() {
                py::Expr::Name(n) => ident(&n.id, at),
                _ => return Err("call target must be a plain op name".to_string()),
            };
            let op_str = op.to_string();
            if op_str == "sqrt" {
                if c.arguments.args.len() != 1 {
                    return Err("`sqrt(<bound_name>)` takes exactly one argument".to_string());
                }
                let name = match &c.arguments.args[0] {
                    py::Expr::Name(n) => ident(&n.id, at),
                    _ => return Err("`sqrt(...)` argument must be a bound identifier".to_string()),
                };
                return Ok(Expr::SqrtBound(name));
            }
            if op_str == "reshape" {
                if c.arguments.args.len() != 2 {
                    return Err(
                        "`reshape(source, [dim0, dim1, ...])` takes exactly two arguments"
                            .to_string(),
                    );
                }
                let source = parse_expr(&c.arguments.args[0], at)?;
                let target_shape = parse_dim_list(&c.arguments.args[1], at)?;
                return Ok(Expr::Reshape {
                    source: Box::new(source),
                    target_shape,
                });
            }
            if op_str == "scalar" || op_str == "recip_scalar" {
                if c.arguments.args.len() != 1 {
                    return Err(
                        "`scalar(<name>)` / `recip_scalar(<name>)` takes exactly one argument"
                            .to_string(),
                    );
                }
                let name = match &c.arguments.args[0] {
                    py::Expr::Name(n) => ident(&n.id, at),
                    _ => {
                        return Err(
                            "`scalar(...)` argument must be a config key identifier".to_string()
                        );
                    }
                };
                return Ok(Expr::ConfigScalar {
                    name,
                    recip: op_str == "recip_scalar",
                });
            }
            let args = c
                .arguments
                .args
                .iter()
                .map(|e| parse_expr(e, at))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Call { op, args })
        }

        // `gate * up` / `w + 1.0`.
        py::Expr::BinOp(b) => {
            let lhs = parse_expr(&b.left, at)?;
            let rhs = parse_expr(&b.right, at)?;
            match b.op {
                py::Operator::Mult => Ok(Expr::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                py::Operator::Add => Ok(Expr::Add {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                other => Err(format!(
                    "only `*` and `+` are admitted in DSL expressions (got {other:?})"
                )),
            }
        }

        // Numeric literal — int promoted to f64, matching the Rust costume.
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i
                .as_u64()
                .map(|v| Expr::ScalarLit(v as f64))
                .ok_or_else(|| "DSL int literals must be non-negative".to_string()),
            py::Number::Float(f) => Ok(Expr::ScalarLit(*f)),
            _ => Err("only numeric literals are admitted as DSL scalars".to_string()),
        },

        // Parenthesized — ruff already unwraps these.
        other => Err(format!(
            "expression `{:?}` is not part of the DSL dialect",
            py::AnyNodeRef::from(other).kind()
        )),
    }
}

/// Flatten `a.b.c` (Name root + attribute chain) into path segments.
fn collect_path_segments(
    expr: &py::Expr,
    at: proc_macro2::Span,
) -> Result<Vec<syn::Ident>, String> {
    let mut out = Vec::new();
    collect_into(expr, at, &mut out)?;
    Ok(out)
}

fn collect_into(
    expr: &py::Expr,
    at: proc_macro2::Span,
    out: &mut Vec<syn::Ident>,
) -> Result<(), String> {
    match expr {
        py::Expr::Name(n) => {
            out.push(ident(&n.id, at));
            Ok(())
        }
        py::Expr::Attribute(a) => {
            collect_into(&a.value, at, out)?;
            let name = a.attr.id.as_str();
            out.push(ident(name, at));
            Ok(())
        }
        _ => Err("path root must be a plain identifier".to_string()),
    }
}

/// `reshape`'s second arg: a list of dims, each a literal / bound /
/// `*` / `/` tree.
fn parse_dim_list(expr: &py::Expr, at: proc_macro2::Span) -> Result<Vec<DimSpec>, String> {
    let l = match expr {
        py::Expr::List(l) => l,
        _ => {
            return Err(
                "reshape target shape must be a list literal `[dim0, dim1, ...]`".to_string(),
            );
        }
    };
    if l.elts.is_empty() {
        return Err("reshape target shape must have at least one dim".to_string());
    }
    l.elts.iter().map(|e| parse_dim_spec(e, at)).collect()
}

fn parse_dim_spec(expr: &py::Expr, at: proc_macro2::Span) -> Result<DimSpec, String> {
    match expr {
        py::Expr::NumberLiteral(n) => match &n.value {
            py::Number::Int(i) => i
                .as_u64()
                .map(DimSpec::Lit)
                .ok_or_else(|| "reshape dim must be a non-negative integer".to_string()),
            _ => Err("reshape dim must be a non-negative integer".to_string()),
        },
        py::Expr::Name(nb) => Ok(DimSpec::Bound(ident(&nb.id, at))),
        py::Expr::BinOp(b) => {
            let lhs = parse_dim_spec(&b.left, at)?;
            let rhs = parse_dim_spec(&b.right, at)?;
            match b.op {
                py::Operator::Mult => Ok(DimSpec::Mul(Box::new(lhs), Box::new(rhs))),
                py::Operator::Div => Ok(DimSpec::Div(Box::new(lhs), Box::new(rhs))),
                // Python floor-division `//` — the DSL's integer `/`.
                py::Operator::FloorDiv => Ok(DimSpec::Div(Box::new(lhs), Box::new(rhs))),
                other => Err(format!(
                    "reshape dim arithmetic admits only `*`, `/`, and `//` (got {other:?})"
                )),
            }
        }
        _ => Err(
            "reshape dim must be an integer literal, a bound identifier, \
             or a `*` / `/` arithmetic over those"
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Ast {
        parse_python_file(src, proc_macro2::Span::call_site())
            .expect("DSL parse")
            .ast
    }

    #[test]
    fn assign_and_call() {
        let ast = parse("@forward\ndef f():\n    x = gemm(a, w.p[layer])\n    return x\n");
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            Stmt::Assign { target, value } => {
                assert_eq!(target.to_string(), "x");
                assert!(matches!(value, Expr::Call { .. }));
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    #[test]
    fn for_range_and_if_modulo() {
        let ast = parse(
            "@forward\ndef f():\n    for layer in range(num_hidden_layers):\n\
             \x20\x20\x20\x20\x20\x20\x20\x20if layer % 2 == 0:\n\
             \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20x = attention(q, k, v)\n\
             \x20\x20\x20\x20\x20\x20\x20\x20else:\n\
             \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20x = sliding(q, k, v)\n",
        );
        match &ast.statements[0] {
            Stmt::For { ivar, body, .. } => {
                assert_eq!(ivar.to_string(), "layer");
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected For, got {other:?}"),
        }
    }

    #[test]
    fn subset_is_enforced() {
        // comprehension — outside the dialect, must be a loud error.
        let err = parse_python_file(
            "@forward\ndef f():\n    x = [y for y in z]\n",
            proc_macro2::Span::call_site(),
        );
        assert!(err.is_err());
        // class at top level — rejected.
        let err = parse_python_file("class A:\n    pass\n", proc_macro2::Span::call_site());
        assert!(err.is_err());
        // no decorator — rejected.
        let err = parse_python_file("def f():\n    x = y\n", proc_macro2::Span::call_site());
        assert!(err.is_err());
    }

    #[test]
    fn decorator_metadata_is_parsed() {
        let carrier = parse_python_file(
            "@vision_forward(workloads=[256, 1024], processor=crate.PROCESSOR)\n\
             def vision():\n    x = gemm(a, b)\n    return x\n",
            proc_macro2::Span::call_site(),
        )
        .expect("decorator parse");
        assert_eq!(carrier.arch_name, "vision");
        assert!(carrier.vision);
        assert_eq!(carrier.workloads, vec![256, 1024]);
        assert!(carrier.sk_buckets.is_empty());
        assert_eq!(carrier.processor.as_deref(), Some("crate::PROCESSOR"));
        assert!(carrier.pixel_pack.is_none());

        let carrier = parse_python_file(
            "@forward\ndef plain():\n    x = gemm(a, b)\n    return x\n",
            proc_macro2::Span::call_site(),
        )
        .expect("decorator parse");
        assert_eq!(carrier.arch_name, "plain");
        assert!(!carrier.vision);
        assert!(carrier.workloads.is_empty());

        let err = parse_python_file(
            "@forward(bogus=1)\ndef f():\n    x = y\n    return x\n",
            proc_macro2::Span::call_site(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn qwen3_carrier_parses() {
        let text = include_str!("../../../models/arch/dsl/qwen3.py");
        let ast = parse(text);
        // embed + loop + final norm + logits: 4 top-level statements
        // (`return logits` is the torch costume's output marker; it
        // does not become a statement).
        assert_eq!(ast.statements.len(), 4);
    }

    /// The costume flip, as a test: `qwen3.py` (torch costume) drives
    /// the full classify → shape → CFG → unroll → schedule → codegen
    /// chain to a NON-EMPTY, complete emission. (The flip-time proof —
    /// byte-equality against the pinned `.rs.in` emission — served its
    /// one-time purpose and was deleted with the fixture; this keeps
    /// the end-to-end parse→emission coverage without pinning bytes.
    /// Numerical equivalence of the emission is covered by the
    /// scratchy-models `qwen3_py_parity` e2e gate.)
    #[test]
    fn python_costume_emits_the_qwen3_tape() {
        use crate::{CompileMode, DEFAULT_DECODER_WORKLOADS, ForwardArgs};

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let arch = "qwen3";
        let configs_dir = root.join("crates/models/arch/configs").join(arch);

        let mut args: ForwardArgs = syn::parse_str("").expect("empty #[forward] args");
        args.workloads = DEFAULT_DECODER_WORKLOADS.to_vec();

        // Python costume: the .py carrier, same configs, same mode.
        let py_text =
            std::fs::read_to_string(root.join("crates/models/arch/dsl/qwen3.py")).unwrap();
        let py_carrier =
            parse_python_file(&py_text, proc_macro2::Span::call_site()).expect("python parse");
        let py_tokens = crate::compile_ast(
            &args,
            &py_carrier.arch_name,
            proc_macro2::Span::call_site(),
            py_carrier.ast,
            CompileMode::DECODER,
            &configs_dir,
        )
        .expect("python-costume pipeline");

        let mut py_rendered = String::with_capacity(1 << 20);
        crate::render_tokens(py_tokens, &mut py_rendered);

        // Non-vacuity guard: model selection reads `CARGO_FEATURE_*`
        // ambient env vars (they belong to the scratchy-models crate,
        // so they are only set when the test is invoked as
        // `env 'CARGO_FEATURE_QWEN3_0.6B=1' cargo test ...`). Without
        // this check, an unscoped run emits an EMPTY token stream and
        // the test passes having verified nothing.
        assert!(
            py_rendered.len() > 100_000,
            "qwen3 emission is {} bytes — model selection filtered everything out. \
             Run with `env 'CARGO_FEATURE_QWEN3_0.6B=1' cargo test -p \
             scratchy-forward-compiler-macro --features metal --lib parse_python`",
            py_rendered.len(),
        );

        // The emission must be COMPLETE, not just big: the flip pinned
        // per-bucket tape symbols + the load/forward entry points. Check
        // the shape, not the bytes — byte-pinning every codegen change
        // would force regenerating a ~67k-line fixture with no extra
        // correctness signal over the numeric e2e gate.
        for marker in [
            "FORWARD_TABLE",
            "fn load",
            "fn forward",
            "q_proj",
            "rope",
            "lm_head",
        ] {
            assert!(
                py_rendered.contains(marker),
                "qwen3 emission lacks `{marker}` — the pipeline emitted an incomplete tape"
            );
        }
    }
}
