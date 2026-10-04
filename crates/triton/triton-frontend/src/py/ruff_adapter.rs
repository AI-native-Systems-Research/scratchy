//! Lower `ruff_python_ast` into [`crate::py::ast`].
//!
//! **THIS IS THE ONLY MODULE IN THE CRATE THAT MENTIONS A RUFF TYPE.** Everything
//! downstream sees only our own AST, so swapping the parser -- or adding a
//! dependency-free one for the offline pod build -- is a change confined to this file.
//!
//! The lowering is total in one direction only: every ruff node we accept becomes one of
//! ours, and every ruff node we do NOT accept becomes an [`Error`] naming the Python
//! construct and its line and column. That is the census gate's first line of defence; the
//! second is [`crate::py::census::check`], which catches things that ARE representable in
//! our AST but are still out of scope (an unsupported `tl.*` target, say).

use ruff_python_ast as ra;
use ruff_text_size::Ranged;

use crate::py::ast::*;
use crate::{Error, Result};

/// Byte-offset -> (line, col) using the same conventions as CPython's `ast`:
/// line is 1-based, col is a 0-based UTF-8 byte offset within the line.
struct Lines {
    starts: Vec<usize>,
}

impl Lines {
    fn new(src: &str) -> Lines {
        let mut starts = vec![0usize];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Lines { starts }
    }

    fn pos(&self, offset: usize) -> Pos {
        // The last line start <= offset.
        let idx = match self.starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        Pos {
            line: (idx + 1) as u32,
            col: (offset - self.starts[idx]) as u32,
        }
    }
}

struct Ctx {
    lines: Lines,
}

impl Ctx {
    fn pos_of(&self, r: &impl Ranged) -> Pos {
        self.lines.pos(r.range().start().to_usize())
    }
}

/// Parse Python source into our AST.
pub fn parse(src: &str) -> Result<PyModule> {
    let parsed = ruff_python_parser::parse_module(src).map_err(|e| {
        let lines = Lines::new(src);
        let p = lines.pos(e.location.start().to_usize());
        Error::new(format!("Python syntax error: {}", e.error), p.line, p.col)
    })?;
    let ctx = Ctx {
        lines: Lines::new(src),
    };
    let mut out = PyModule::default();
    for stmt in &parsed.syntax().body {
        match stmt {
            ra::Stmt::FunctionDef(f) => {
                out.functions.push(lower_function(&ctx, f)?);
            }
            // Module-level `NAME = <literal>`: captured so a kernel that reads one can be
            // refused BY NAME (see `PyModule::globals`).
            ra::Stmt::Assign(a) => {
                if let (1, Some(ra::Expr::Name(n))) = (a.targets.len(), a.targets.first()) {
                    if let Some(lit) = literal_of(&a.value) {
                        out.globals.push((n.id.to_string(), lit));
                    }
                }
            }
            // Imports, docstrings and host functions are not the front end's business.
            _ => {}
        }
    }
    Ok(out)
}

/// A module-level literal, including a negated one (`X = -1.0`).
fn literal_of(e: &ra::Expr) -> Option<Literal> {
    match e {
        ra::Expr::NumberLiteral(n) => match &n.value {
            ra::Number::Int(i) => i.as_i64().map(|v| Literal::Int(v as i128)),
            ra::Number::Float(f) => Some(Literal::Float(*f)),
            ra::Number::Complex { .. } => None,
        },
        ra::Expr::BooleanLiteral(b) => Some(Literal::Bool(b.value)),
        ra::Expr::StringLiteral(s) => Some(Literal::Str(s.value.to_str().to_string())),
        ra::Expr::NoneLiteral(_) => Some(Literal::None),
        ra::Expr::UnaryOp(u) if matches!(u.op, ra::UnaryOp::USub) => match literal_of(&u.operand) {
            Some(Literal::Int(v)) => Some(Literal::Int(-v)),
            Some(Literal::Float(v)) => Some(Literal::Float(-v)),
            _ => None,
        },
        _ => None,
    }
}

/// Is this decorator `@triton.jit` or `@jit` (possibly called, `@triton.jit(...)`)?
fn is_jit_decorator(d: &ra::Decorator) -> bool {
    let mut e = &d.expression;
    if let ra::Expr::Call(c) = e {
        e = &c.func;
    }
    // Walk the attribute chain collecting the parts, as ast_census.py does.
    let mut parts = Vec::new();
    loop {
        match e {
            ra::Expr::Attribute(a) => {
                parts.push(a.attr.to_string());
                e = &a.value;
            }
            ra::Expr::Name(n) => {
                parts.push(n.id.to_string());
                break;
            }
            _ => break,
        }
    }
    parts.iter().any(|p| p == "jit")
}

fn lower_function(ctx: &Ctx, f: &ra::StmtFunctionDef) -> Result<FunctionDef> {
    let pos = ctx.pos_of(f);
    let is_jit = f.decorator_list.iter().any(is_jit_decorator);
    let mut params = Vec::new();
    if !is_jit {
        // Host code: recorded by name only, never walked.
        return Ok(FunctionDef {
            name: f.name.to_string(),
            params,
            body: Vec::new(),
            is_jit: false,
            pos,
        });
    }
    if f.parameters.vararg.is_some() || f.parameters.kwarg.is_some() {
        return Err(Error::new(
            format!(
                "kernel `{}` uses *args or **kwargs, which is outside the supported \
                 parameter forms",
                f.name
            ),
            pos.line,
            pos.col,
        ));
    }
    for p in f
        .parameters
        .posonlyargs
        .iter()
        .chain(f.parameters.args.iter())
        .chain(f.parameters.kwonlyargs.iter())
    {
        let annotation = match &p.parameter.annotation {
            Some(a) => Some(dotted_of(a).ok_or_else(|| {
                let ap = ctx.pos_of(&**a);
                Error::new(
                    format!(
                        "parameter `{}` has an annotation that is not a dotted name; only \
                         `tl.constexpr` and dtype annotations are supported",
                        p.parameter.name
                    ),
                    ap.line,
                    ap.col,
                )
            })?),
            None => None,
        };
        params.push(Param {
            name: p.parameter.name.to_string(),
            annotation,
            pos: ctx.pos_of(&p.parameter),
        });
    }
    let body = lower_body(ctx, &f.body)?;
    Ok(FunctionDef {
        name: f.name.to_string(),
        params,
        body,
        is_jit: true,
        pos,
    })
}

fn dotted_of(e: &ra::Expr) -> Option<String> {
    match e {
        ra::Expr::Name(n) => Some(n.id.to_string()),
        ra::Expr::Attribute(a) => Some(format!("{}.{}", dotted_of(&a.value)?, a.attr)),
        _ => None,
    }
}

fn lower_body(ctx: &Ctx, body: &[ra::Stmt]) -> Result<Vec<Stmt>> {
    let mut out = Vec::new();
    for s in body {
        // A bare string expression is a docstring; Triton's walk reaches it as an Expr
        // whose value is a Constant and does nothing with it.
        if let ra::Stmt::Expr(e) = s {
            if matches!(&*e.value, ra::Expr::StringLiteral(_)) {
                continue;
            }
        }
        out.push(lower_stmt(ctx, s)?);
    }
    Ok(out)
}

fn lower_stmt(ctx: &Ctx, s: &ra::Stmt) -> Result<Stmt> {
    let pos = ctx.pos_of(s);
    match s {
        ra::Stmt::Assign(a) => {
            if a.targets.len() != 1 {
                return Err(Error::new(
                    "chained assignment (`a = b = ...`) is not supported",
                    pos.line,
                    pos.col,
                ));
            }
            Ok(Stmt::Assign {
                target: lower_target(ctx, &a.targets[0])?,
                value: lower_expr(ctx, &a.value)?,
                pos,
            })
        }
        ra::Stmt::AnnAssign(a) => Ok(Stmt::AnnAssign {
            target: lower_target(ctx, &a.target)?,
            annotation: lower_expr(ctx, &a.annotation)?,
            value: match &a.value {
                Some(v) => Some(lower_expr(ctx, v)?),
                None => None,
            },
            pos,
        }),
        ra::Stmt::AugAssign(a) => Ok(Stmt::AugAssign {
            target: lower_target(ctx, &a.target)?,
            op: lower_binop(ctx, a.op, pos)?,
            value: lower_expr(ctx, &a.value)?,
            pos,
        }),
        ra::Stmt::For(f) => {
            let target = match &*f.target {
                ra::Expr::Name(n) => n.id.to_string(),
                other => {
                    let p = ctx.pos_of(other);
                    return Err(Error::new(
                        "a `for` loop target must be a single name (tuple unpacking in a \
                         loop target is not supported)",
                        p.line,
                        p.col,
                    ));
                }
            };
            Ok(Stmt::For {
                target,
                iter: lower_expr(ctx, &f.iter)?,
                body: lower_body(ctx, &f.body)?,
                orelse: lower_body(ctx, &f.orelse)?,
                pos,
            })
        }
        ra::Stmt::If(i) => {
            // ruff models `elif` as an `elif_else_clauses` list; rebuild the nested
            // if/else shape CPython's `ast` gives, which is what Triton walks.
            let mut orelse: Vec<Stmt> = Vec::new();
            for clause in i.elif_else_clauses.iter().rev() {
                match &clause.test {
                    Some(test) => {
                        let cpos = ctx.pos_of(clause);
                        orelse = vec![Stmt::If {
                            test: lower_expr(ctx, test)?,
                            body: lower_body(ctx, &clause.body)?,
                            orelse,
                            pos: cpos,
                        }];
                    }
                    None => {
                        orelse = lower_body(ctx, &clause.body)?;
                    }
                }
            }
            Ok(Stmt::If {
                test: lower_expr(ctx, &i.test)?,
                body: lower_body(ctx, &i.body)?,
                orelse,
                pos,
            })
        }
        ra::Stmt::Expr(e) => Ok(Stmt::Expr {
            value: lower_expr(ctx, &e.value)?,
            pos,
        }),
        ra::Stmt::Return(r) => Ok(Stmt::Return {
            value: match &r.value {
                Some(v) => Some(lower_expr(ctx, v)?),
                None => None,
            },
            pos,
        }),
        ra::Stmt::Pass(_) => Ok(Stmt::Pass { pos }),
        other => Err(Error::new(
            format!(
                "Python statement `{}` is not supported inside a @triton.jit kernel",
                stmt_kind(other)
            ),
            pos.line,
            pos.col,
        )),
    }
}

/// The CPython `ast` node-type name, so a refusal NAMES the construct the way the census
/// does.
fn stmt_kind(s: &ra::Stmt) -> &'static str {
    match s {
        ra::Stmt::FunctionDef(_) => "FunctionDef",
        ra::Stmt::ClassDef(_) => "ClassDef",
        ra::Stmt::Return(_) => "Return",
        ra::Stmt::Delete(_) => "Delete",
        ra::Stmt::Assign(_) => "Assign",
        ra::Stmt::AugAssign(_) => "AugAssign",
        ra::Stmt::AnnAssign(_) => "AnnAssign",
        ra::Stmt::TypeAlias(_) => "TypeAlias",
        ra::Stmt::For(_) => "For",
        ra::Stmt::While(_) => "While",
        ra::Stmt::If(_) => "If",
        ra::Stmt::With(_) => "With",
        ra::Stmt::Match(_) => "Match",
        ra::Stmt::Raise(_) => "Raise",
        ra::Stmt::Try(_) => "Try",
        ra::Stmt::Assert(_) => "Assert",
        ra::Stmt::Import(_) => "Import",
        ra::Stmt::ImportFrom(_) => "ImportFrom",
        ra::Stmt::Global(_) => "Global",
        ra::Stmt::Nonlocal(_) => "Nonlocal",
        ra::Stmt::Expr(_) => "Expr",
        ra::Stmt::Pass(_) => "Pass",
        ra::Stmt::Break(_) => "Break",
        ra::Stmt::Continue(_) => "Continue",
        ra::Stmt::IpyEscapeCommand(_) => "IpyEscapeCommand",
    }
}

fn expr_kind(e: &ra::Expr) -> &'static str {
    match e {
        ra::Expr::BoolOp(_) => "BoolOp",
        ra::Expr::Named(_) => "NamedExpr",
        ra::Expr::BinOp(_) => "BinOp",
        ra::Expr::UnaryOp(_) => "UnaryOp",
        ra::Expr::Lambda(_) => "Lambda",
        ra::Expr::If(_) => "IfExp",
        ra::Expr::Dict(_) => "Dict",
        ra::Expr::Set(_) => "Set",
        ra::Expr::ListComp(_) => "ListComp",
        ra::Expr::SetComp(_) => "SetComp",
        ra::Expr::DictComp(_) => "DictComp",
        ra::Expr::Generator(_) => "GeneratorExp",
        ra::Expr::Await(_) => "Await",
        ra::Expr::Yield(_) => "Yield",
        ra::Expr::YieldFrom(_) => "YieldFrom",
        ra::Expr::Compare(_) => "Compare",
        ra::Expr::Call(_) => "Call",
        ra::Expr::FString(_) => "JoinedStr",
        ra::Expr::TString(_) => "TemplateStr",
        ra::Expr::StringLiteral(_) => "Constant(str)",
        ra::Expr::BytesLiteral(_) => "Constant(bytes)",
        ra::Expr::NumberLiteral(_) => "Constant(number)",
        ra::Expr::BooleanLiteral(_) => "Constant(bool)",
        ra::Expr::NoneLiteral(_) => "Constant(None)",
        ra::Expr::EllipsisLiteral(_) => "Constant(Ellipsis)",
        ra::Expr::Attribute(_) => "Attribute",
        ra::Expr::Subscript(_) => "Subscript",
        ra::Expr::Starred(_) => "Starred",
        ra::Expr::Name(_) => "Name",
        ra::Expr::List(_) => "List",
        ra::Expr::Tuple(_) => "Tuple",
        ra::Expr::Slice(_) => "Slice",
        ra::Expr::IpyEscapeCommand(_) => "IpyEscapeCommand",
    }
}

fn lower_target(ctx: &Ctx, e: &ra::Expr) -> Result<AssignTarget> {
    let pos = ctx.pos_of(e);
    match e {
        ra::Expr::Name(n) => Ok(AssignTarget::Name {
            id: n.id.to_string(),
            pos,
        }),
        ra::Expr::Tuple(t) => {
            let mut elts = Vec::new();
            for x in &t.elts {
                elts.push(lower_target(ctx, x)?);
            }
            Ok(AssignTarget::Tuple { elts, pos })
        }
        other => Err(Error::new(
            format!(
                "assignment to a {} is not supported (only a name or a tuple of names)",
                expr_kind(other)
            ),
            pos.line,
            pos.col,
        )),
    }
}

fn lower_binop(ctx: &Ctx, op: ra::Operator, pos: Pos) -> Result<BinOpKind> {
    let _ = (ctx, pos);
    Ok(match op {
        ra::Operator::Add => BinOpKind::Add,
        ra::Operator::Sub => BinOpKind::Sub,
        ra::Operator::Mult => BinOpKind::Mult,
        ra::Operator::Div => BinOpKind::Div,
        ra::Operator::FloorDiv => BinOpKind::FloorDiv,
        ra::Operator::Mod => BinOpKind::Mod,
        ra::Operator::BitAnd => BinOpKind::BitAnd,
        ra::Operator::BitOr => BinOpKind::BitOr,
        ra::Operator::BitXor => BinOpKind::BitXor,
        ra::Operator::LShift => BinOpKind::LShift,
        ra::Operator::RShift => BinOpKind::RShift,
        ra::Operator::Pow => BinOpKind::Pow,
        ra::Operator::MatMult => BinOpKind::MatMult,
    })
}

fn lower_expr(ctx: &Ctx, e: &ra::Expr) -> Result<Expr> {
    let pos = ctx.pos_of(e);
    match e {
        ra::Expr::Name(n) => Ok(Expr::Name {
            id: n.id.to_string(),
            pos,
        }),
        ra::Expr::NumberLiteral(n) => {
            let value = match &n.value {
                ra::Number::Int(i) => match i.as_i64() {
                    Some(v) => Literal::Int(v as i128),
                    None => {
                        return Err(Error::new(
                            "integer literal does not fit in 64 bits",
                            pos.line,
                            pos.col,
                        ))
                    }
                },
                ra::Number::Float(f) => Literal::Float(*f),
                ra::Number::Complex { .. } => {
                    return Err(Error::new(
                        "complex literals are not supported",
                        pos.line,
                        pos.col,
                    ))
                }
            };
            Ok(Expr::Constant { value, pos })
        }
        ra::Expr::BooleanLiteral(b) => Ok(Expr::Constant {
            value: Literal::Bool(b.value),
            pos,
        }),
        ra::Expr::StringLiteral(s) => Ok(Expr::Constant {
            value: Literal::Str(s.value.to_str().to_string()),
            pos,
        }),
        ra::Expr::NoneLiteral(_) => Ok(Expr::Constant {
            value: Literal::None,
            pos,
        }),
        ra::Expr::Attribute(a) => Ok(Expr::Attribute {
            value: Box::new(lower_expr(ctx, &a.value)?),
            attr: a.attr.to_string(),
            pos,
        }),
        ra::Expr::Call(c) => {
            let mut args = Vec::new();
            for a in c.arguments.args.iter() {
                if matches!(a, ra::Expr::Starred(_)) {
                    let p = ctx.pos_of(a);
                    return Err(Error::new(
                        "*args unpacking at a call site is not supported",
                        p.line,
                        p.col,
                    ));
                }
                args.push(lower_expr(ctx, a)?);
            }
            let mut keywords = Vec::new();
            for k in c.arguments.keywords.iter() {
                match &k.arg {
                    Some(name) => keywords.push((name.to_string(), lower_expr(ctx, &k.value)?)),
                    None => {
                        let p = ctx.pos_of(k);
                        return Err(Error::new(
                            "**kwargs unpacking at a call site is not supported",
                            p.line,
                            p.col,
                        ));
                    }
                }
            }
            Ok(Expr::Call {
                func: Box::new(lower_expr(ctx, &c.func)?),
                args,
                keywords,
                pos,
            })
        }
        ra::Expr::BinOp(b) => Ok(Expr::BinOp {
            left: Box::new(lower_expr(ctx, &b.left)?),
            op: lower_binop(ctx, b.op, pos)?,
            right: Box::new(lower_expr(ctx, &b.right)?),
            pos,
        }),
        ra::Expr::UnaryOp(u) => Ok(Expr::UnaryOp {
            op: match u.op {
                ra::UnaryOp::USub => UnaryOpKind::USub,
                ra::UnaryOp::UAdd => UnaryOpKind::UAdd,
                ra::UnaryOp::Not => UnaryOpKind::Not,
                ra::UnaryOp::Invert => UnaryOpKind::Invert,
            },
            operand: Box::new(lower_expr(ctx, &u.operand)?),
            pos,
        }),
        ra::Expr::Compare(c) => {
            if c.ops.len() != 1 || c.comparators.len() != 1 {
                return Err(Error::new(
                    "simultaneous multiple comparison is not supported",
                    pos.line,
                    pos.col,
                ));
            }
            let op = match c.ops[0] {
                ra::CmpOp::Eq => CmpKind::Eq,
                ra::CmpOp::NotEq => CmpKind::NotEq,
                ra::CmpOp::Lt => CmpKind::Lt,
                ra::CmpOp::LtE => CmpKind::LtE,
                ra::CmpOp::Gt => CmpKind::Gt,
                ra::CmpOp::GtE => CmpKind::GtE,
                other => {
                    return Err(Error::new(
                        format!("comparison operator `{other:?}` is not supported"),
                        pos.line,
                        pos.col,
                    ))
                }
            };
            Ok(Expr::Compare {
                left: Box::new(lower_expr(ctx, &c.left)?),
                op,
                right: Box::new(lower_expr(ctx, &c.comparators[0])?),
                pos,
            })
        }
        ra::Expr::Subscript(s) => Ok(Expr::Subscript {
            value: Box::new(lower_expr(ctx, &s.value)?),
            index: Box::new(lower_expr(ctx, &s.slice)?),
            pos,
        }),
        ra::Expr::Slice(s) => Ok(Expr::Slice {
            lower: match &s.lower {
                Some(x) => Some(Box::new(lower_expr(ctx, x)?)),
                None => None,
            },
            upper: match &s.upper {
                Some(x) => Some(Box::new(lower_expr(ctx, x)?)),
                None => None,
            },
            step: match &s.step {
                Some(x) => Some(Box::new(lower_expr(ctx, x)?)),
                None => None,
            },
            pos,
        }),
        ra::Expr::List(l) => {
            let mut elts = Vec::new();
            for x in &l.elts {
                elts.push(lower_expr(ctx, x)?);
            }
            Ok(Expr::List { elts, pos })
        }
        ra::Expr::Tuple(t) => {
            let mut elts = Vec::new();
            for x in &t.elts {
                elts.push(lower_expr(ctx, x)?);
            }
            Ok(Expr::Tuple { elts, pos })
        }
        other => Err(Error::new(
            format!(
                "Python expression `{}` is not supported inside a @triton.jit kernel",
                expr_kind(other)
            ),
            pos.line,
            pos.col,
        )),
    }
}
