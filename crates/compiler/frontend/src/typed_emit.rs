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

use crate::bundles::Bundle;
use crate::classified::{Expr, LocalId, OpKind, Program, Stmt};
use crate::dim::{DERIVED_DIMS, DerivedRule, Dim, canonical_add, canonical_mul};
use crate::hf_config::ParamExpr;
use crate::weights_manifest::WeightsManifest;

/// The bounds the op signatures in `typed_shapes` name themselves, beside those a `weights.json`
/// declares. Config vocabulary, shared by every arch.
pub const SIGNATURE_BOUNDS: [&str; 11] = [
    "backbone_hidden_size",
    "global_head_dim",
    "head_dim",
    "hidden_size",
    "num_attention_heads",
    "num_global_key_value_heads",
    "num_key_value_heads",
    "num_tokens",
    "vision_embed_dim",
    "vision_in_features",
    "vision_rope_half_dim",
];

/// The input widths each carrier names in its own scope, in `Inputs`' parameter order (`Px`,
/// `Rh`, `Bh`, `Ve`): a carrier's arch.json may derive any of them.
const INPUT_BOUNDS: [&str; 4] = [
    "vision_in_features",
    "vision_rope_half_dim",
    "backbone_hidden_size",
    "vision_embed_dim",
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

/// A dim as a type, and every bound it names into `bounds`. A product or sum is written in the
/// canonical order its `Dim::Mul` or `Dim::Add` already has, nested to the right.
pub fn dim_type(dim: &Dim, bounds: &mut BTreeSet<String>) -> Result<String, String> {
    match dim {
        Dim::Bound(name) => {
            bounds.insert(name.clone());
            Ok(marker(name))
        }
        Dim::Mul(parts) | Dim::Add(parts) => {
            let (op, empty) = match dim {
                Dim::Mul(_) => ("Mul", "an empty product"),
                _ => ("Add", "an empty sum"),
            };
            let types = (parts.iter())
                .map(|f| dim_type(f, bounds))
                .collect::<Result<Vec<_>, _>>()?;
            let (last, rest) = types.split_last().ok_or(empty)?;
            Ok(rest
                .iter()
                .rev()
                .fold(last.clone(), |acc, t| format!("{op}<{t}, {acc}>")))
        }
        Dim::Lit(n) => Ok(format!("Lit<{n}>")),
        Dim::Div(n, d) => Ok(format!(
            "Div<{}, {}>",
            dim_type(n, bounds)?,
            dim_type(d, bounds)?
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
    /// What its configs and arch.json say about its derived dims.
    pub facts: &'a CarrierFacts,
}

/// What a carrier's configs and arch.json say about the derived dims its weights name: each gated
/// rule's flag, resolved over every one of its configs (`Err` naming the configs on each side when
/// they disagree), its arch.json `params` `expr` formulas, and the weight bundles its program
/// passes whole to an op, derived from its configs (`Err` when they cannot be, or disagree).
#[derive(Debug)]
pub struct CarrierFacts {
    pub gated: BTreeMap<String, Result<u64, String>>,
    pub exprs: BTreeMap<String, ParamExpr>,
    pub bundles: Result<BTreeMap<String, Bundle>, String>,
}

impl Default for CarrierFacts {
    /// No configs: no flags, formulas or bundles.
    fn default() -> Self {
        Self {
            gated: BTreeMap::new(),
            exprs: BTreeMap::new(),
            bundles: Ok(BTreeMap::new()),
        }
    }
}

/// The carrier's module: its `Weights`, typed from its manifest, and `forward`, its program over
/// the signatures. Every bound a weight names goes into `bounds`.
pub fn emit(c: &Carrier<'_>, base_bounds: &mut BTreeSet<String>) -> Result<String, String> {
    let mut named: BTreeSet<String> = BTreeSet::new();
    let bounds = &mut named;
    let bundles = (c.facts.bundles.as_ref()).map_err(|e| format!("{}: {e}", c.file))?;
    let mut cx = Cx {
        c,
        bundles,
        reads: HashSet::new(),
        named: HashSet::new(),
        carried: HashSet::new(),
        fields: BTreeMap::new(),
        bounds: BTreeSet::new(),
        out: String::new(),
    };
    cx.scan(&c.program.statements);
    let mut weights = String::new();
    for id in c.program.weights.ids() {
        let path = c.program.weights.path(id);
        let dotted = path.join(".");
        let field = field_name(path);
        // A bundle: one field per role, each at its derived shape.
        let typed: Vec<(String, &crate::dim::Shape)> = match bundles.get(&dotted) {
            Some(bundle) => (bundle.kind.roles().iter())
                .zip(&bundle.roles)
                .map(|(role, shape)| (format!("{field}_{}", role.name()), shape))
                .collect(),
            None => {
                let declared = c.manifest.lookup(path).ok_or_else(|| {
                    format!(
                        "{}: the weight `{dotted}` has no `weights.json` declaration to type it by",
                        c.file
                    )
                })?;
                vec![(field.clone(), declared)]
            }
        };
        for (field, declared) in typed {
            let ty = weight_type(declared, bounds)
                .map_err(|e| format!("{}: weight `{dotted}` (`{field}`): {e}", c.file))?;
            if let Some(other) = cx.fields.insert(field.clone(), dotted.clone()) {
                return Err(format!(
                    "{}: the weights `{other}` and `{dotted}` both spell the field `{field}`",
                    c.file
                ));
            }
            let _ = writeln!(weights, "        pub {field}: {ty},");
        }
    }
    let init: String = (cx.fields.keys())
        .map(|f| format!("            {f}: Declared::declared(),\n"))
        .collect();
    cx.stmts(&c.program.statements, 2)?;
    named.append(&mut cx.bounds);
    // The inputs' widths, which a carrier's arch.json may derive (`Inputs`' `Px`, `Rh`).
    named.extend(INPUT_BOUNDS.map(str::to_string));
    let aliases = carrier_aliases(c, named, base_bounds)?;
    Ok(format!(
        "/// `{file}`, typed: its weights as `weights.json` declares them (its bundles as its configs\n\
         /// derive them), and its program over the op signatures.\n\
         pub mod {module} {{\n\
         \x20   use super::*;\n\n{aliases}\
         \x20   /// The weights it reads, at their declared shapes.\n\
         \x20   pub struct Weights {{\n{weights}    }}\n\n\
         \x20   /// `{file}`'s program, statement for statement.\n\
         \x20   pub fn forward<T: DimType>(tr: &Trace) {{\n\
         \x20       tr.program(&[{locals}]);\n\
         \x20       let inputs = Inputs::<T, {input_widths}>::declared();\n\
         \x20       let w = Weights {{\n{init}        }};\n{body}    }}\n\
         }}\n",
        file = c.file,
        module = c.module,
        body = cx.out,
        input_widths = INPUT_BOUNDS.map(marker).join(", "),
        locals = (c.program.locals.ids())
            .map(|id| format!("{:?}", c.program.locals.name(id).to_string()))
            .collect::<Vec<_>>()
            .join(", "),
    ))
}

/// The marker of every base bound in `bounds` and in [`SIGNATURE_BOUNDS`]: every derived dim is a
/// type alias in the module of the carrier that names it ([`carrier_aliases`]).
pub fn emit_markers(bounds: &BTreeSet<String>) -> Result<String, String> {
    let all: BTreeSet<&str> = (bounds.iter().map(String::as_str))
        .chain(SIGNATURE_BOUNDS)
        .collect();
    let rows: String = (all.iter())
        .map(|b| format!("    {} => {b:?},\n", marker(b)))
        .collect();
    Ok(format!("bounds! {{\n{rows}}}\n"))
}

/// The carrier's derived dims among `named` (and every derived dim their rules read), each a type
/// alias for its rule (`type KvDim = Mul<HeadDim, NumKeyValueHeads>;`); every base bound they read
/// goes into `base`. A derived dim is one of [`DERIVED_DIMS`] (a gated one by the carrier's
/// resolved flag) or one of its arch.json `expr` params.
fn carrier_aliases(
    c: &Carrier<'_>,
    named: BTreeSet<String>,
    base: &mut BTreeSet<String>,
) -> Result<String, String> {
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut pending: Vec<String> = named.into_iter().collect();
    while let Some(name) = pending.pop() {
        if aliases.contains_key(&name) || base.contains(&name) {
            continue;
        }
        let dim = if let Some(d) = DERIVED_DIMS.iter().find(|d| d.name == name) {
            Some(rule_dim(c, &name, &d.rule)?)
        } else if let Some(e) = c.facts.exprs.get(&name) {
            Some(expr_dim(c, &name, e)?)
        } else {
            None
        };
        match dim {
            Some(dim) => {
                let mut reads = BTreeSet::new();
                let ty = dim_type(&dim, &mut reads)?;
                pending.extend(reads);
                aliases.insert(name, ty);
            }
            None => {
                base.insert(name);
            }
        }
    }
    let types: String = (aliases.iter())
        .map(|(name, ty)| {
            format!(
                "    /// `{name}`, a derived dim.\n    pub type {} = {ty};\n",
                marker(name)
            )
        })
        .collect();
    let rows: String = (aliases.keys())
        .map(|name| {
            format!(
                "            ({name:?}, <{} as DimType>::dim()),\n",
                marker(name)
            )
        })
        .collect();
    Ok(format!(
        "{types}\n    /// Each derived dim's type, as the dim it states (for the tests: every one must equal \
         the\n    /// config loader's value under each config's bounds).\n    #[cfg(test)]\n    \
         pub fn derived_dims() -> Vec<(&'static str, Dim)> {{\n        vec![\n{rows}        ]\n    }}\n\n"
    ))
}

/// A [`DERIVED_DIMS`] rule as a dim, a gated rule's arm chosen by the carrier's resolved flag.
fn rule_dim(c: &Carrier<'_>, name: &str, rule: &DerivedRule) -> Result<Dim, String> {
    Ok(match rule {
        DerivedRule::Bound(b) => Dim::Bound(b.to_string()),
        DerivedRule::Lit(n) => Dim::Lit(*n),
        DerivedRule::Mul(factors) => canonical_mul(
            (factors.iter())
                .map(|f| rule_dim(c, name, f))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        DerivedRule::Gated { flag, on, off } => {
            let value = (c.facts.gated.get(*flag))
                .ok_or_else(|| format!("{}: `{name}`'s flag `{flag}` was not resolved", c.file))?
                .clone()
                .map_err(|why| format!("{}: `{name}`: {why}", c.file))?;
            rule_dim(c, name, if value != 0 { on } else { off })?
        }
        DerivedRule::Add(terms) => canonical_add(
            (terms.iter())
                .map(|t| rule_dim(c, name, t))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

/// An arch.json `expr` param as a dim: sums, products, quotients and literals. A difference or a
/// root is refused by name: a dim the types cannot hold is never approximated.
fn expr_dim(c: &Carrier<'_>, name: &str, e: &ParamExpr) -> Result<Dim, String> {
    Ok(match e {
        ParamExpr::Bound(b) => Dim::Bound(b.clone()),
        ParamExpr::Lit(n) => Dim::Lit(*n),
        ParamExpr::Mul(a, b) => canonical_mul(vec![expr_dim(c, name, a)?, expr_dim(c, name, b)?]),
        ParamExpr::Div(a, b) => Dim::Div(
            Box::new(expr_dim(c, name, a)?),
            Box::new(expr_dim(c, name, b)?),
        ),
        ParamExpr::Add(a, b) => canonical_add(vec![expr_dim(c, name, a)?, expr_dim(c, name, b)?]),
        ParamExpr::Sub(..) | ParamExpr::Sqrt(_) => {
            return Err(format!(
                "{}: the arch.json param `{name}` (`{e:?}`) is a dim the types cannot hold \
                 (a difference or root)",
                c.file
            ));
        }
    })
}

/// `typed_program`: each emitted carrier's traced function, by its arch.
pub fn emit_dispatch(carriers: &[(String, String)]) -> String {
    let derived_arms: String = (carriers.iter())
        .map(|(arch, module)| format!("        {arch:?} => Some({module}::derived_dims()),\n"))
        .collect();
    let arms: String = (carriers.iter())
        .map(|(arch, module)| {
            format!("        {arch:?} => Some({module}::forward::<NumTokens>),\n")
        })
        .collect();
    format!(
        "/// The traced function of the carrier whose arch is `arch`, if the emit pass typed it.\n\
         pub fn typed_program(arch: &str) -> Option<fn(&Trace)> {{\n\
         \x20   match arch {{\n{arms}        _ => None,\n    }}\n}}\n\n\
         /// The derived dims of the carrier whose arch is `arch`, each as its type states it.\n\
         #[cfg(test)]\n\
         pub fn derived_dims_of(arch: &str) -> Option<Vec<(&'static str, Dim)>> {{\n\
         \x20   match arch {{\n{derived_arms}        _ => None,\n    }}\n}}\n"
    )
}

/// The bound a row count is spelled by in a DSL shape: the typed program's row type `T`.
const ROWS: &str = "num_tokens";

/// An op the DSL calls at more than one arity, each arity its own signature: `(op, arity, the
/// signature's name)`. Any other arity is the op's own name.
const ARITY_FORMS: [(OpKind, usize, &str); 3] = [
    // `attention(q, k, v)`: an encoder's, with no KV cache or block table.
    (OpKind::Attention, 3, "encoder_attention"),
    // `moe_block(router_in, expert_in, moe)`: a latent MoE's, its experts on their own input.
    (OpKind::Moe, 3, "latent_moe_block"),
    // `draft_conv(x, dynamic, base, residual)`: the finishing stage, onto a residual.
    (OpKind::DraftConv, 4, "draft_conv_residual"),
];

/// A weight's type at its declared shape: `W1<D>`, `W2<K, N>`, or an expert bank's `W3<E, K, N>`.
fn weight_type(
    declared: &crate::dim::Shape,
    bounds: &mut BTreeSet<String>,
) -> Result<String, String> {
    let dims = (declared.iter())
        .map(|d| dim_type(d, bounds))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match dims.as_slice() {
        [d] => format!("W1<{d}>"),
        [k, n] => format!("W2<{k}, {n}>"),
        [e, k, n] => format!("W3<{e}, {k}, {n}>"),
        _ => {
            return Err(format!(
                "rank {}; the typed emit has ranks 1 to 3",
                dims.len()
            ));
        }
    })
}

/// A weight's field in `Weights`: its dotted path, `_`-joined, in snake case (a checkpoint's
/// `attn.Wo` is `attn_wo`). [`emit`] refuses two weights that spell one field.
fn field_name<S: AsRef<str>>(path: &[S]) -> String {
    let parts: Vec<&str> = path.iter().map(AsRef::as_ref).collect();
    parts.join("_").to_ascii_lowercase()
}

/// Whether `e` is a scalar operand (it has no shape).
fn is_scalar(e: &Expr) -> bool {
    matches!(
        e,
        Expr::ScalarLit(_) | Expr::SqrtBound(_) | Expr::ConfigScalar { .. }
    )
}

struct Cx<'a, 'c> {
    c: &'a Carrier<'c>,
    /// The weight bundles the program passes whole to an op, derived from its configs.
    bundles: &'a BTreeMap<String, Bundle>,
    /// Every local some expression reads.
    reads: HashSet<LocalId>,
    /// Every local the emitted code names: those read, a loop's carried-out bindings, and an
    /// `if`'s arm results.
    named: HashSet<LocalId>,
    /// Every local a loop carries out of an iteration (its pre-loop binding).
    carried: HashSet<LocalId>,
    /// Each weight field, by the dotted path it spells.
    fields: BTreeMap<String, String>,
    /// Every bound a statement's annotated type names (a reshape target's, an `as_dim`'s width).
    bounds: BTreeSet<String>,
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
            Stmt::Assign {
                target,
                value: Expr::Call { op, args },
            } if *op == OpKind::Reshape => {
                let text = self.regroup(*target, args)?;
                self.line(depth, &text);
            }
            // `x = as_dim(y, <bound>)`: `y`'s rows, typed at the stated width.
            Stmt::Assign {
                target,
                value: Expr::Local(source),
            } if self.c.program.dim_casts.contains_key(target) => {
                let ty = dim_type(&self.c.program.dim_casts[target], &mut self.bounds)?;
                let text = format!(
                    "let {}: Act<T, {ty}> = tr.bind({}, ops::as_dim(l{}));{}",
                    self.binder(*target),
                    target.0,
                    source.0,
                    self.origin(*target)
                );
                self.line(depth, &text);
            }
            Stmt::Assign { target, value } => {
                let e = self.top(value, *target)?;
                let text = format!(
                    "let {} = tr.bind({}, {e});{}",
                    self.binder(*target),
                    target.0,
                    self.origin(*target)
                );
                self.line(depth, &text);
            }
            Stmt::AssignTuple { targets, value } => {
                let first = targets.first().ok_or("an empty tuple assignment")?;
                let e = self.top(value, *first)?;
                let binders: Vec<String> = targets.iter().map(|t| self.binder(*t)).collect();
                let ids: Vec<String> = targets.iter().map(|t| t.0.to_string()).collect();
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

    /// A DSL `reshape` as the structural regrouping its target names, annotated with the target's
    /// type so rustc checks it against the source's: splitting a product axis into the rows
    /// (`[num_tokens · H.., Hd]`, [`split_heads`]) or merging it back (`[num_tokens, Hd · H..]`,
    /// `merge_heads`); grouping `M` rows into one (`[num_tokens / M, X]`, `group_rows`) or back
    /// (`[num_tokens, E]`, `ungroup_rows`). Any other target is refused: a new kind of regrouping
    /// is a new structural signature, never a value check.
    fn regroup(&mut self, target: LocalId, args: &[Expr]) -> Result<String, String> {
        let [source] = args else {
            return Err(format!("{}: reshape takes one source", self.c.file));
        };
        let dims = (self.c.program.reshape_targets.get(&target))
            .ok_or_else(|| format!("{}: a reshape with no target shape", self.c.file))?;
        let rows = Dim::Bound(ROWS.to_string());
        let mut bounds = std::mem::take(&mut self.bounds);
        let refuse = || {
            format!(
                "{}: reshape to {dims:?} at{} is no structural regrouping the typed emit knows \
                 (split a product axis into the rows, or merge it back)",
                self.c.file,
                self.origin(target)
            )
        };
        let (op, ty) = match dims.as_slice() {
            // [rows · H.., Hd]: split.
            [Dim::Mul(prefix), hd] if prefix.contains(&rows) => {
                let heads: Vec<Dim> = prefix.iter().filter(|d| **d != rows).cloned().collect();
                let heads = dim_type(&canonical_mul(heads), &mut bounds)?;
                let hd = dim_type(hd, &mut bounds)?;
                ("split_heads", format!("Act<Mul<{heads}, T>, {hd}>"))
            }
            // [rows / M, X]: M consecutive rows become one.
            [Dim::Div(n, m), x] if **n == rows => {
                let m = dim_type(m, &mut bounds)?;
                let x = dim_type(x, &mut bounds)?;
                ("group_rows", format!("Act<Div<T, {m}>, {x}>"))
            }
            // [rows, Hd · H..]: merge back.
            [r, product @ Dim::Mul(_)] if *r == rows => (
                "merge_heads",
                format!("Act<T, {}>", dim_type(product, &mut bounds)?),
            ),
            // [rows, E]: grouped rows back to one per row (`group_rows`' inverse).
            [r, e] if *r == rows => (
                "ungroup_rows",
                format!("Act<T, {}>", dim_type(e, &mut bounds)?),
            ),
            _ => return Err(refuse()),
        };
        self.bounds = bounds;
        Ok(format!(
            "let {}: {ty} = tr.bind({}, ops::{op}({}));{}",
            self.binder(target),
            target.0,
            self.expr(source, &mut Nest::of(target))?,
            self.origin(target)
        ))
    }

    /// A statement's value, binding local `target` (a tuple's first): its operands each go through
    /// [`Self::expr`], so the trace records every nested value under `target`.
    fn top(&self, e: &Expr, target: LocalId) -> Result<String, String> {
        let mut nest = Nest::of(target);
        match e {
            Expr::Call { .. } | Expr::Mul { .. } => self.value(e, &mut nest),
            _ => self.expr(e, &mut nest),
        }
    }

    /// `e`, an operand inside a statement: a call or product is recorded as the statement's next
    /// nested value, after its own operands.
    fn expr(&self, e: &Expr, nest: &mut Nest) -> Result<String, String> {
        match e {
            Expr::Call { .. } | Expr::Mul { .. } => {
                let value = self.value(e, nest)?;
                let n = nest.next;
                nest.next += 1;
                Ok(format!("tr.nested({}, {n}, {value})", nest.target.0))
            }
            _ => self.value(e, nest),
        }
    }

    fn value(&self, e: &Expr, nest: &mut Nest) -> Result<String, String> {
        Ok(match e {
            Expr::Local(id) => format!("l{}", id.0),
            Expr::Extern { kind, .. } => format!("inputs.{}", kind.name()),
            Expr::Weight { id, .. } => {
                let path = self.c.program.weights.path(*id);
                let field = field_name(path);
                match self.bundles.get(&path.join(".")) {
                    // A bundle is its kind's constructor over its roles: its signature relates
                    // them. It is no weight of the program's (its roles are the checkpoint's).
                    Some(bundle) => {
                        use crate::bundles::{BundleArg, BundleRole, args, constructor};
                        let role = |r: &BundleRole| format!("w.{field}_{}", r.name());
                        let args: Vec<String> = (args(bundle.kind).iter())
                            .map(|arg| match arg {
                                BundleArg::Role(r) => role(r),
                                BundleArg::Part(ctor, parts) => {
                                    let parts: Vec<String> = parts.iter().map(role).collect();
                                    format!("ops::{ctor}({})", parts.join(", "))
                                }
                            })
                            .collect();
                        format!("ops::{}({})", constructor(bundle.kind), args.join(", "))
                    }
                    None => format!("tr.weight({}, w.{field})", id.0),
                }
            }
            // A sum with a scalar operand offsets the tensor (Gemma's `(1 + w)` norm weight).
            Expr::Call { op, args }
                if *op == OpKind::Add
                    && let [a, b] = args.as_slice()
                    && (is_scalar(a) || is_scalar(b)) =>
            {
                let (x, s) = if is_scalar(b) { (a, b) } else { (b, a) };
                format!(
                    "ops::offset({}, {})",
                    self.expr(x, nest)?,
                    self.expr(s, nest)?
                )
            }
            Expr::Call { op, args } => {
                let args = (args.iter())
                    .map(|a| self.expr(a, nest))
                    .collect::<Result<Vec<_>, _>>()?;
                let name = (ARITY_FORMS.iter())
                    .find(|(o, arity, _)| o == op && *arity == args.len())
                    .map_or(op.as_str(), |(_, _, name)| name);
                format!("ops::{name}({})", args.join(", "))
            }
            // A product with a scalar operand scales the tensor; two tensors multiply elementwise.
            Expr::Mul { lhs, rhs } => match (is_scalar(lhs), is_scalar(rhs)) {
                (false, true) => format!(
                    "ops::scale({}, {})",
                    self.expr(lhs, nest)?,
                    self.expr(rhs, nest)?
                ),
                (true, false) => format!(
                    "ops::scale({}, {})",
                    self.expr(rhs, nest)?,
                    self.expr(lhs, nest)?
                ),
                _ => format!(
                    "ops::mul({}, {})",
                    self.expr(lhs, nest)?,
                    self.expr(rhs, nest)?
                ),
            },
            Expr::ScalarLit(_) | Expr::SqrtBound(_) | Expr::ConfigScalar { .. } => {
                "Scalar".to_string()
            }
        })
    }
}

/// Where a statement's nested values are recorded: its target, and the next value's place.
struct Nest {
    target: LocalId,
    next: u32,
}

impl Nest {
    fn of(target: LocalId) -> Self {
        Self { target, next: 0 }
    }
}
