// SPDX-License-Identifier: Apache-2.0
//! THE TYPED-RUST EMIT PASS: a carrier's classified [`Program`] as a Rust function over the op
//! signatures in `typed_shapes`, so rustc type-checks the carrier's shapes.
//!
//! One emit rule per classified [`Stmt`] / [`Expr`] kind and one signature per op kind, named as the
//! op is (`OpKind::as_str`). It never names an arch: everything that differs between arches is the
//! carrier's own program and its `weights.json`. Each local `l<id>` is the classifier's
//! [`LocalId`], and each binding goes through `Trace::bind`, so running the function records the
//! shape every binding's type states.
//!
//! - A weight is a field of the carrier's `Weights`, typed by its `weights.json` declaration.
//! - A loop's carried variable is `mut`, reassigned from its last binding at the body's end: the
//!   body's carry is the loop's own type. The body runs once (`Trace::loop_once`).
//! - An `if` runs both arms; each merge is `Trace::merge(then, else)`, whose one type parameter is
//!   the law that both arms bind it alike.
//! - Each binding's line ends in a comment naming its carrier file and line, so rustc's error at a
//!   line names the DSL line it came from.
//!
//! This module is compiled twice from this one file: into the crate, and into the crate's build
//! script (`build.rs`), which runs it over every carrier before the crate compiles.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;

use crate::classified::{Expr, LocalId, Program, Stmt};
use crate::dim::Dim;
use crate::weights_manifest::WeightsManifest;

/// The bounds the op signatures in `typed_shapes` name themselves, beside those a `weights.json`
/// declares. Config vocabulary, shared by every arch.
pub const SIGNATURE_BOUNDS: [&str; 4] = [
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "num_tokens",
];

/// The marker type a bound is spelled as: its name in UpperCamelCase (`head_dim` → `HeadDim`).
pub fn marker(bound: &str) -> String {
    bound
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// A dim as a type, and every bound it names into `bounds`. A product is written in the canonical
/// order its `Dim::Mul` already has, nested to the right.
pub fn dim_type(dim: &Dim, bounds: &mut BTreeSet<String>) -> Result<String, String> {
    match dim {
        Dim::Bound(name) => {
            bounds.insert(name.clone());
            Ok(marker(name))
        }
        Dim::Mul(factors) => {
            let types = (factors.iter())
                .map(|f| dim_type(f, bounds))
                .collect::<Result<Vec<_>, _>>()?;
            let (last, rest) = types.split_last().ok_or("an empty product")?;
            Ok(rest
                .iter()
                .rev()
                .fold(last.clone(), |acc, t| format!("Mul<{t}, {acc}>")))
        }
        Dim::Lit(_) | Dim::Div(..) | Dim::Var(_) => Err(format!(
            "the typed emit has no type for the dim `{dim:?}` yet"
        )),
    }
}

/// One carrier to emit.
pub struct Carrier<'a> {
    /// The module its function is emitted in: the carrier's file stem, `-` as `_`.
    pub module: &'a str,
    /// Its arch: the carrier's `def` name, which `typed_program` keys it by.
    pub arch: &'a str,
    /// Its file, as error comments name it.
    pub file: &'a str,
    /// Its text, whose lines the comments quote.
    pub source: &'a str,
    /// Its classified program.
    pub program: &'a Program,
    /// Its arch's `weights.json`.
    pub manifest: &'a WeightsManifest,
}

/// The carrier's module: its `Weights`, typed from its manifest, and `forward`, its program over
/// the signatures. Every bound a weight names goes into `bounds`.
pub fn emit(c: &Carrier<'_>, bounds: &mut BTreeSet<String>) -> Result<String, String> {
    let mut cx = Cx {
        c,
        reads: HashSet::new(),
        named: HashSet::new(),
        carried: HashSet::new(),
        fields: BTreeMap::new(),
        out: String::new(),
    };
    cx.scan(&c.program.statements);
    let mut weights = String::new();
    for id in c.program.weights.ids() {
        let path = c.program.weights.path(id);
        let field = path.join("_");
        let declared = c.manifest.lookup(path).ok_or_else(|| {
            format!(
                "{}: the weight `{}` has no `weights.json` declaration to type it by",
                c.file,
                path.join(".")
            )
        })?;
        let dims = (declared.iter())
            .map(|d| dim_type(d, bounds))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: weight `{}`: {e}", c.file, path.join(".")))?;
        let ty = match dims.as_slice() {
            [d] => format!("W1<{d}>"),
            [k, n] => format!("W2<{k}, {n}>"),
            _ => {
                return Err(format!(
                    "{}: weight `{}` is rank {}; the typed emit has rank 1 and 2",
                    c.file,
                    path.join("."),
                    dims.len()
                ));
            }
        };
        if let Some(other) = cx.fields.insert(field.clone(), path.join(".")) {
            return Err(format!(
                "{}: the weights `{other}` and `{}` both spell the field `{field}`",
                c.file,
                path.join(".")
            ));
        }
        let _ = writeln!(weights, "        pub {field}: {ty},");
    }
    let init: String = (cx.fields.keys())
        .map(|f| format!("            {f}: Declared::declared(),\n"))
        .collect();
    cx.stmts(&c.program.statements, 2)?;
    Ok(format!(
        "/// `{file}`, typed: its weights as `weights.json` declares them, and its program over the\n\
         /// op signatures.\n\
         pub mod {module} {{\n\
         \x20   use super::*;\n\n\
         \x20   /// The weights it reads, at their declared shapes.\n\
         \x20   pub struct Weights {{\n{weights}    }}\n\n\
         \x20   /// `{file}`'s program, statement for statement.\n\
         \x20   pub fn forward<T: DimType>(tr: &mut Trace) {{\n\
         \x20       let inputs = Inputs::<T>::declared();\n\
         \x20       let w = Weights {{\n{init}        }};\n{body}    }}\n\
         }}\n",
        file = c.file,
        module = c.module,
        body = cx.out,
    ))
}

/// The marker of every bound in `bounds` and in [`SIGNATURE_BOUNDS`].
pub fn emit_markers(bounds: &BTreeSet<String>) -> String {
    let all: BTreeSet<&str> = (bounds.iter().map(String::as_str))
        .chain(SIGNATURE_BOUNDS)
        .collect();
    let rows: String = (all.iter())
        .map(|b| format!("    {} => {b:?},\n", marker(b)))
        .collect();
    format!("bounds! {{\n{rows}}}\n")
}

/// `typed_program`: each emitted carrier's traced function, by its arch.
pub fn emit_dispatch(carriers: &[(String, String)]) -> String {
    let arms: String = (carriers.iter())
        .map(|(arch, module)| {
            format!("        {arch:?} => Some({module}::forward::<NumTokens>),\n")
        })
        .collect();
    format!(
        "/// The traced function of the carrier whose arch is `arch`, if the emit pass typed it.\n\
         pub fn typed_program(arch: &str) -> Option<fn(&mut Trace)> {{\n\
         \x20   match arch {{\n{arms}        _ => None,\n    }}\n}}\n"
    )
}

struct Cx<'a, 'c> {
    c: &'a Carrier<'c>,
    /// Every local some expression reads.
    reads: HashSet<LocalId>,
    /// Every local the emitted code names: those read, a loop's carried-out bindings, and an
    /// `if`'s arm results.
    named: HashSet<LocalId>,
    /// Every local a loop carries out of an iteration (its pre-loop binding).
    carried: HashSet<LocalId>,
    /// Each weight field, by the dotted path it spells.
    fields: BTreeMap<String, String>,
    out: String,
}

impl Cx<'_, '_> {
    fn scan(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match s {
                Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => {
                    self.scan_expr(value)
                }
                Stmt::For {
                    body, loop_carry, ..
                } => {
                    self.scan(body);
                    for (outer, top) in loop_carry {
                        self.carried.insert(*outer);
                        self.named.insert(*outer);
                        self.named.insert(*top);
                    }
                }
                Stmt::If {
                    then_body,
                    else_body,
                    merge_carry,
                    ..
                } => {
                    self.scan(then_body);
                    self.scan(else_body);
                    for (_, then_final, else_final) in merge_carry {
                        self.named.insert(*then_final);
                        self.named.insert(*else_final);
                    }
                }
            }
        }
    }

    fn scan_expr(&mut self, e: &Expr) {
        match e {
            Expr::Local(id) => {
                self.reads.insert(*id);
                self.named.insert(*id);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| self.scan_expr(a)),
            Expr::Mul { lhs, rhs } => {
                self.scan_expr(lhs);
                self.scan_expr(rhs);
            }
            _ => {}
        }
    }

    /// `let mut l<id>` for a carried local, `let l<id>` for a named one, `let _` for one nothing
    /// names (its shape is still recorded).
    fn binder(&self, id: LocalId) -> String {
        if self.carried.contains(&id) {
            format!("mut l{}", id.0)
        } else if self.named.contains(&id) {
            format!("l{}", id.0)
        } else {
            "_".to_string()
        }
    }

    /// The comment naming the carrier line that binds `id`.
    fn origin(&self, id: LocalId) -> String {
        let name = self.c.program.locals.name(id);
        match self.c.program.locals.line(id) {
            Some(line) => {
                let text = (self.c.source.lines().nth(line as usize - 1))
                    .map(str::trim)
                    .unwrap_or("");
                format!(" // {}:{line}: {text}", self.c.file)
            }
            None => format!(" // {}: `{name}`, inserted by the compiler", self.c.file),
        }
    }

    fn stmts(&mut self, stmts: &[Stmt], depth: usize) -> Result<(), String> {
        for s in stmts {
            self.stmt(s, depth)?;
        }
        Ok(())
    }

    fn line(&mut self, depth: usize, text: &str) {
        let _ = writeln!(self.out, "{}{text}", "    ".repeat(depth));
    }

    fn stmt(&mut self, s: &Stmt, depth: usize) -> Result<(), String> {
        match s {
            Stmt::Assign { target, value } => {
                let e = self.expr(value)?;
                let text = format!(
                    "let {} = tr.bind({}, {e});{}",
                    self.binder(*target),
                    target.0,
                    self.origin(*target)
                );
                self.line(depth, &text);
            }
            Stmt::AssignTuple { targets, value } => {
                let e = self.expr(value)?;
                let binders: Vec<String> = targets.iter().map(|t| self.binder(*t)).collect();
                let ids: Vec<String> = targets.iter().map(|t| t.0.to_string()).collect();
                let first = targets.first().ok_or("an empty tuple assignment")?;
                let text = format!(
                    "let ({}) = tr.bind_tuple([{}], {e});{}",
                    binders.join(", "),
                    ids.join(", "),
                    self.origin(*first)
                );
                self.line(depth, &text);
            }
            Stmt::For {
                body, loop_carry, ..
            } => {
                self.line(depth, "for _ in tr.loop_once() {");
                self.stmts(body, depth + 1)?;
                for (outer, top) in loop_carry {
                    self.line(depth + 1, &format!("l{} = l{};", outer.0, top.0));
                }
                self.line(depth, "}");
                // After the loop its carried-out bindings are the carried variables' last values.
                for (outer, top) in loop_carry {
                    if self.reads.contains(top) {
                        self.line(depth, &format!("let l{} = l{};", top.0, outer.0));
                    }
                }
            }
            Stmt::If {
                then_body,
                else_body,
                merge_carry,
                ..
            } => {
                let finals = |pick: fn(&(LocalId, LocalId, LocalId)) -> LocalId| {
                    let ids: Vec<String> = (merge_carry.iter())
                        .map(|m| format!("l{}", pick(m).0))
                        .collect();
                    format!("({},)", ids.join(", "))
                };
                let merges: Vec<String> = (merge_carry.iter())
                    .map(|(m, ..)| format!("m{}", m.0))
                    .collect();
                self.line(depth, &format!("let ({},) = merge(", merges.join(", ")));
                self.line(depth + 1, "{");
                self.stmts(then_body, depth + 2)?;
                self.line(depth + 2, &finals(|m| m.1));
                self.line(depth + 1, "},");
                self.line(depth + 1, "{");
                self.stmts(else_body, depth + 2)?;
                self.line(depth + 2, &finals(|m| m.2));
                self.line(depth + 1, "},");
                self.line(depth, ");");
                for (m, ..) in merge_carry {
                    let text = format!(
                        "let {} = tr.bind({}, m{});{}",
                        self.binder(*m),
                        m.0,
                        m.0,
                        self.origin(*m)
                    );
                    self.line(depth, &text);
                }
            }
        }
        Ok(())
    }

    fn expr(&self, e: &Expr) -> Result<String, String> {
        Ok(match e {
            Expr::Local(id) => format!("l{}", id.0),
            Expr::Extern { kind, .. } => format!("inputs.{}", kind.name()),
            Expr::Weight { id, .. } => format!("w.{}", self.c.program.weights.path(*id).join("_")),
            Expr::Call { op, args } => {
                let args = (args.iter())
                    .map(|a| self.expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("ops::{}({})", op.as_str(), args.join(", "))
            }
            Expr::Mul { lhs, rhs } => format!("ops::mul({}, {})", self.expr(lhs)?, self.expr(rhs)?),
            Expr::ScalarLit(_) | Expr::SqrtBound(_) | Expr::ConfigScalar { .. } => {
                return Err(format!(
                    "{}: the typed emit has no rule for the scalar `{e:?}` yet",
                    self.c.file
                ));
            }
        })
    }
}
