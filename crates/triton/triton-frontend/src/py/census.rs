//! THE SCOPE BOUNDARY, measured rather than guessed -- and enforced.
//!
//! `../triton-superdsc/tools/ast_census.py` walks the `@triton.jit` functions of the
//! fixtures transitively and reports what the front end must handle. Re-run:
//!
//! ```text
//! python3 ../triton-superdsc/tools/ast_census.py \
//!     ../../test/fixtures/{vector_add,mul,bias_add_f32,swiglu_mlp,attention_flash,\
//!                          embedding,rmsnorm,rope}.py
//!
//! 9 jit fn(s) across 8 files
//! AST NODE TYPES the front end must handle (33 distinct)
//! CALL TARGETS (the `tl.*` surface plus helpers), 39 distinct
//! ```
//!
//! ## The 39 call targets, CLASSIFIED.
//!
//! The census counts a call target by its dotted SPELLING, so a method on a local value
//! (`a_desc.load`) and a module function (`tl.dot`) are counted alike, and two receivers of
//! the same method are counted twice. Sorted into what the front end actually has to
//! implement, the 39 are:
//!
//! | class | count | targets |
//! |---|---|---|
//! | `tl.*` functions | 16 | `arange` `dot` `exp` `fdiv` `full` `make_tensor_descriptor` `math.exp2` `max` `maximum` `multiple_of` `program_id` `range` `rsqrt` `static_assert` `sum` `zeros` |
//! | Python builtins | 1 | `float` |
//! | user `@triton.jit` fns | 1 | `_attn_fwd_inner` |
//! | a method on an unnamed receiver | 1 | `to` (as in `tl.max(qk, 1).to(...)`) |
//! | receiver-qualified methods | 20 | `a_desc.load`, `b_desc.load`, `c.to`, `c_desc.store`, `cos_desc.load`, `desc_k.load`, `desc_mask.load`, `desc_o.store`, `desc_q.load`, `desc_v.load`, `ids_desc.load`, `o_desc.store`, `qk.to`, `sin_desc.load`, `table_desc.gather`, `w_desc.load`, `wd_desc.load`, `wg_desc.load`, `wu_desc.load`, `x_desc.load` |
//!
//! Those five rows sum to 39, which is the point of writing them out: the headline number is
//! 39 rather than 22 because `x_desc.load` and `a_desc.load` are ONE implementation reached
//! through different receivers. The distinct method implementations are FOUR --
//! [`VALUE_METHODS`] -- and `gather` is the newest of them, arriving with `embedding.py`.
//!
//! (This table previously said "value methods 3" and "receiver-qualified duplicates 13" while
//! listing 15 names under the latter, which summed to 32 rather than the 33 the census
//! reported. The rows above are the re-run, with the arithmetic checked.)
//!
//! ## Fail closed.
//!
//! [`check`] walks a parsed module and returns EVERY violation, each naming the construct
//! and its line and column. A construct outside the census is never lowered partially.

use crate::py::ast::{AssignTarget, Expr, FunctionDef, PyModule, Stmt};
use crate::Error;

/// The `tl.*` surface the fixtures reach, by dotted name with the `tl.` prefix stripped.
pub const TL_FUNCTIONS: &[&str] = &[
    "arange",
    "dot",
    "exp",
    "fdiv",
    "full",
    "make_tensor_descriptor",
    "math.exp2",
    "max",
    "maximum",
    "multiple_of",
    "program_id",
    "range",
    "rsqrt",
    "math.rsqrt",
    "static_assert",
    "sum",
    "zeros",
];

/// Methods called on a value rather than on a module.
///
/// `gather` is the one entry NOT in the original five-fixture census: it is
/// `embedding.py`'s indirect read (`tt.descriptor_gather`), added with that fixture.
pub const VALUE_METHODS: &[&str] = &["load", "store", "to", "gather"];

/// Python builtins the kernels call. `float(x)` is a constexpr cast in
/// `attention_flash.py`'s `sm_scale` handling.
pub const BUILTINS: &[&str] = &["float", "range"];

/// The dtype attributes reached as `tl.<name>`.
pub const TL_DTYPES: &[&str] = &[
    "float16", "float32", "bfloat16", "float64", "int1", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
];

/// Module aliases that mean "the triton language module".
pub const TL_ALIASES: &[&str] = &["tl", "triton.language"];

fn is_tl_chain(dotted: &str) -> Option<String> {
    for a in TL_ALIASES {
        if let Some(rest) = dotted.strip_prefix(&format!("{a}.")) {
            return Some(rest.to_string());
        }
    }
    None
}

/// Everything a call target may resolve to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallTarget {
    /// `tl.<name>` where `<name>` is in [`TL_FUNCTIONS`].
    TlFunction(String),
    /// A method on a value: the receiver expression's dotted spelling (if any) and the
    /// method name from [`VALUE_METHODS`].
    Method { receiver: Option<String>, method: String },
    Builtin(String),
    /// A `@triton.jit` function defined in the same module.
    UserJit(String),
}

/// Classify a call's callee, or say why it is out of scope.
pub fn classify_call(func: &Expr, module: &PyModule) -> std::result::Result<CallTarget, String> {
    let pos = func.pos();
    let _ = pos;
    match func.dotted() {
        Some(dotted) => {
            if let Some(rest) = is_tl_chain(&dotted) {
                if TL_FUNCTIONS.contains(&rest.as_str()) {
                    return Ok(CallTarget::TlFunction(rest));
                }
                return Err(format!(
                    "`tl.{rest}` is not in the supported `tl.*` surface. \
                     Supported: {}",
                    TL_FUNCTIONS.join(", ")
                ));
            }
            // A bare name: a user jit function, or a builtin.
            if !dotted.contains('.') {
                if let Some(f) = module.function(&dotted) {
                    if f.is_jit {
                        return Ok(CallTarget::UserJit(dotted));
                    }
                    return Err(format!(
                        "`{dotted}` is defined in this module but is not decorated \
                         @triton.jit, so it is host code and cannot be called from a kernel"
                    ));
                }
                if BUILTINS.contains(&dotted.as_str()) {
                    return Ok(CallTarget::Builtin(dotted));
                }
                return Err(format!("`{dotted}` is not a known function"));
            }
            // `<something>.<method>`
            let (recv, method) = dotted.rsplit_once('.').unwrap();
            if VALUE_METHODS.contains(&method) {
                return Ok(CallTarget::Method {
                    receiver: Some(recv.to_string()),
                    method: method.to_string(),
                });
            }
            Err(format!(
                "method `.{method}` (on `{recv}`) is not supported. \
                 Supported value methods: {}",
                VALUE_METHODS.join(", ")
            ))
        }
        None => {
            // The callee is an expression, e.g. `f(x).y(...)`. Only a method on a
            // non-trivial receiver is in scope.
            if let Expr::Attribute { attr, .. } = func {
                if VALUE_METHODS.contains(&attr.as_str()) {
                    return Ok(CallTarget::Method {
                        receiver: None,
                        method: attr.clone(),
                    });
                }
                return Err(format!("method `.{attr}` is not supported"));
            }
            Err(format!(
                "a call whose callee is a {} expression is not supported",
                func.kind_name()
            ))
        }
    }
}

/// Walk a module and collect every out-of-census construct.
///
/// Returns `Ok(())` only when the whole module is inside the census. Otherwise EVERY
/// violation is reported, so one run names all the work rather than one item at a time.
pub fn check(module: &PyModule) -> std::result::Result<(), Vec<Error>> {
    let mut errs = Vec::new();
    for f in module.jit_functions() {
        check_fn(f, module, &mut errs);
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

fn check_fn(f: &FunctionDef, module: &PyModule, errs: &mut Vec<Error>) {
    for s in &f.body {
        check_stmt(s, module, errs);
    }
}

fn check_stmt(s: &Stmt, module: &PyModule, errs: &mut Vec<Error>) {
    match s {
        Stmt::Assign { target, value, .. } => {
            check_target(target, errs);
            check_expr(value, module, errs);
        }
        Stmt::AnnAssign {
            target,
            annotation,
            value,
            ..
        } => {
            check_target(target, errs);
            check_expr(annotation, module, errs);
            if let Some(v) = value {
                check_expr(v, module, errs);
            }
        }
        Stmt::AugAssign { target, value, .. } => {
            check_target(target, errs);
            check_expr(value, module, errs);
        }
        Stmt::For {
            iter,
            body,
            orelse,
            pos,
            ..
        } => {
            // `visit_For` accepts only a call to `range` / `tl.range` / `tl.static_range`.
            let ok = match iter {
                Expr::Call { func, .. } => match func.dotted() {
                    Some(d) => {
                        d == "range"
                            || is_tl_chain(&d).map(|r| r == "range" || r == "static_range")
                                == Some(true)
                    }
                    None => false,
                },
                _ => false,
            };
            if !ok {
                errs.push(Error::new(
                    "a `for` loop must iterate `range(...)`, `tl.range(...)` or \
                     `tl.static_range(...)`; nothing else is supported",
                    pos.line,
                    pos.col,
                ));
            }
            check_expr(iter, module, errs);
            for s in body {
                check_stmt(s, module, errs);
            }
            if !orelse.is_empty() {
                errs.push(Error::new(
                    "`for ... else` is not supported",
                    pos.line,
                    pos.col,
                ));
            }
        }
        Stmt::If {
            test, body, orelse, ..
        } => {
            check_expr(test, module, errs);
            for s in body.iter().chain(orelse.iter()) {
                check_stmt(s, module, errs);
            }
        }
        Stmt::Expr { value, .. } => check_expr(value, module, errs),
        Stmt::Return { value, .. } => {
            if let Some(v) = value {
                check_expr(v, module, errs);
            }
        }
        Stmt::Pass { .. } => {}
    }
}

fn check_target(t: &AssignTarget, errs: &mut Vec<Error>) {
    match t {
        AssignTarget::Name { .. } => {}
        AssignTarget::Tuple { elts, .. } => {
            for e in elts {
                check_target(e, errs);
            }
        }
    }
}

fn check_expr(e: &Expr, module: &PyModule, errs: &mut Vec<Error>) {
    match e {
        Expr::Call {
            func,
            args,
            keywords,
            pos,
        } => {
            if let Err(why) = classify_call(func, module) {
                errs.push(Error::new(why, pos.line, pos.col));
            }
            for a in args {
                check_expr(a, module, errs);
            }
            for (_, v) in keywords {
                check_expr(v, module, errs);
            }
        }
        Expr::Attribute { value, .. } => check_expr(value, module, errs),
        Expr::BinOp { left, right, .. } => {
            check_expr(left, module, errs);
            check_expr(right, module, errs);
        }
        Expr::UnaryOp { operand, .. } => check_expr(operand, module, errs),
        Expr::Compare { left, right, .. } => {
            check_expr(left, module, errs);
            check_expr(right, module, errs);
        }
        Expr::Subscript { value, index, .. } => {
            check_expr(value, module, errs);
            check_expr(index, module, errs);
        }
        Expr::Slice {
            lower, upper, step, ..
        } => {
            for x in [lower, upper, step].into_iter().flatten() {
                check_expr(x, module, errs);
            }
        }
        Expr::List { elts, .. } | Expr::Tuple { elts, .. } => {
            for x in elts {
                check_expr(x, module, errs);
            }
        }
        Expr::Name { .. } | Expr::Constant { .. } => {}
    }
}
