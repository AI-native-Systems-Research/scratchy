//! A port of `python/triton/compiler/code_generator.py`: the AST walk that builds TTIR.
//!
//! # How locations work, measured against the oracle rather than assumed.
//!
//! `CodeGenerator.visit` (`code_generator.py:1581`) sets the builder's location from the
//! node it is about to visit and RESTORES it afterwards:
//!
//! ```text
//! here_loc = create_loc(file, begin_line + node.lineno, begin_col + node.col_offset)
//! ```
//!
//! with `begin_line = def_file_line_number - 1` and `begin_col = 1`
//! (`code_generator.py:290,305`). Since the parsed source starts at the `def`, that works
//! out to **the node's file line, and its 0-based column PLUS ONE** -- MLIR columns are
//! 1-based where CPython's `col_offset` is 0-based. Verified against
//! `tests/goldens/vector_add.ttir_raw.mlir`: `tl.program_id(0)` sits at 0-based column 10
//! of file line 47 and the golden says `47:11`.
//!
//! While visiting the right-hand side of an assignment to a plain name,
//! `_name_loc_prefix` (`code_generator.py:433`) installs that name, and every op created
//! during the visit gets a `NameLoc` carrying it. That is where `%offset`, `%a_desc` and
//! the rest of the golden's readable SSA names come from, and it is why a single Python
//! statement's twelve ops all share one name.

use std::collections::HashMap;

use crate::py::ast::{
    AssignTarget, BinOpKind, CmpKind, Expr, FunctionDef, Literal, Pos, PyModule, Stmt, UnaryOpKind,
};
use crate::py::census::{self, CallTarget};
use crate::semantic::{Bitwise, CmpOp, Semantic, Val};
use crate::target::Target;
use crate::ttir::{FloatKind, Func, Loc, Module, Signedness, Type, Visibility};
use crate::{Error, Result};

/// How one kernel parameter is passed.
#[derive(Clone, Debug, PartialEq)]
pub enum ArgSpec {
    /// A pointer argument, e.g. Triton's `"*fp16"`.
    Ptr(Type),
    /// A runtime scalar, e.g. `"i32"`.
    Scalar(Type),
    /// A compile-time constant, bound from [`KernelSpec::constexprs`].
    Constexpr,
}

impl ArgSpec {
    /// Parse a Triton signature spelling: `"*fp16"`, `"i32"`, `"constexpr"`.
    pub fn parse(s: &str) -> Result<ArgSpec> {
        if s == "constexpr" {
            return Ok(ArgSpec::Constexpr);
        }
        if let Some(elem) = s.strip_prefix('*') {
            return Ok(ArgSpec::Ptr(parse_dtype(elem)?));
        }
        Ok(ArgSpec::Scalar(parse_dtype(s)?))
    }
}

/// Triton's dtype spellings, as they appear in a signature dict.
pub fn parse_dtype(s: &str) -> Result<Type> {
    Ok(match s {
        "fp16" | "float16" => Type::f16(),
        "bf16" | "bfloat16" => Type::Float(FloatKind::BF16),
        "fp32" | "float32" => Type::f32(),
        "fp64" | "float64" => Type::Float(FloatKind::F64),
        "fp8e4nv" | "float8e4nv" => Type::Float(FloatKind::F8E4M3FN),
        "fp8e5" | "float8e5" => Type::Float(FloatKind::F8E5M2),
        "i1" | "int1" => Type::i1(),
        "i8" | "int8" => Type::Int(8, Signedness::Signed),
        "i16" | "int16" => Type::Int(16, Signedness::Signed),
        "i32" | "int32" => Type::i32(),
        "i64" | "int64" => Type::i64(),
        "u8" | "uint8" => Type::Int(8, Signedness::Unsigned),
        "u16" | "uint16" => Type::Int(16, Signedness::Unsigned),
        "u32" | "uint32" => Type::Int(32, Signedness::Unsigned),
        "u64" | "uint64" => Type::Int(64, Signedness::Unsigned),
        other => {
            return Err(Error::new(
                format!("unknown dtype spelling `{other}` in a kernel signature"),
                0,
                0,
            ))
        }
    })
}

/// What to compile.
#[derive(Clone, Debug)]
pub struct KernelSpec {
    /// The `@triton.jit` function to compile as the kernel entry.
    pub kernel: String,
    /// Per parameter NAME, how it is passed. Parameters not listed are an error.
    pub signature: HashMap<String, ArgSpec>,
    /// Bindings for every [`ArgSpec::Constexpr`] parameter.
    pub constexprs: HashMap<String, Val>,
    /// The path recorded in locations. Only used for the `Loc::File` payload, which the
    /// structural diff does not compare.
    pub file: String,
}

/// Compile Python source into a TTIR module.
///
/// Needs the `ruff` feature, which is what supplies the Python parser. Without it the rest
/// of the crate still builds and tests -- see [`compile_ast`], which takes an AST that has
/// already been built and is how the dependency-free configuration is exercised.
#[cfg(feature = "ruff")]
pub fn compile(src: &str, spec: &KernelSpec, target: Target) -> Result<Module> {
    let module = crate::py::ruff_adapter::parse(src)?;
    compile_ast(&module, spec, target)
}

/// The dependency-free build has no Python parser, so this says so rather than existing as
/// a function that cannot work.
#[cfg(not(feature = "ruff"))]
pub fn compile(_src: &str, _spec: &KernelSpec, _target: Target) -> Result<Module> {
    Err(Error::new(
        "no Python parser is compiled in: build with `--features ruff`. The parser is behind \
         a Cargo feature because this workspace's other crates are dependency-free and must \
         build offline on the pod; see the crate docs. `compile_ast` works either way.",
        0,
        0,
    ))
}

/// Compile an already-parsed module. Split out so tests can build an AST directly.
pub fn compile_ast(module: &PyModule, spec: &KernelSpec, target: Target) -> Result<Module> {
    // The census gate runs FIRST and reports everything, so a kernel with three
    // unsupported constructs names all three rather than one per run.
    if let Err(errs) = census::check(module) {
        let mut msg = format!(
            "{} construct(s) outside the supported census \
             (../triton-superdsc/tools/ast_census.py):",
            errs.len()
        );
        for e in &errs {
            msg.push_str(&format!("\n  at {}:{}: {}", e.line, e.col, e.message));
        }
        let first = errs.first().cloned().unwrap_or(Error::new("", 0, 0));
        return Err(Error::new(msg, first.line, first.col));
    }

    let f = module.function(&spec.kernel).ok_or_else(|| {
        Error::new(
            format!("no function named `{}` in this module", spec.kernel),
            0,
            0,
        )
    })?;
    if !f.is_jit {
        return Err(Error::new(
            format!(
                "`{}` is not decorated @triton.jit, so it is host code and cannot be \
                 compiled as a kernel",
                spec.kernel
            ),
            f.pos.line,
            f.pos.col,
        ));
    }

    let mut cg = CodeGen {
        sem: Semantic::new(&spec.file, target),
        py: module,
        spec,
        scope: Scope::default(),
        pending_annotation: HashMap::new(),
        generated: std::collections::HashSet::new(),
        return_sites: Vec::new(),
        fn_ret_types: HashMap::new(),
    };
    cg.visit_kernel(f)?;
    Ok(cg.sem.module)
}

/// An INSERTION-ORDERED name scope.
///
/// # The ordering is load-bearing, not tidiness.
///
/// `_find_carries` (`code_generator.py:465`) decides a loop's carried values by iterating
/// `liveins.items()` -- a **Python dict**, so insertion-ordered -- and the resulting order is
/// the order of `scf.for`'s `iter_args`, of its region's block arguments, of its results, and
/// of the `scf.yield` operands. A `HashMap` would produce a correct-looking loop with its
/// carries permuted, which the structural diff would catch as an operand mismatch but which
/// would be a genuine miscompile if it ever slipped through.
///
/// Measured: `swiglu_mlp`'s inner loop yields `iter_args(%g_45 = %g, %u_46 = %u)` in exactly
/// the order `g` and `u` were first bound in the enclosing scope.
///
/// Reassigning an existing name does NOT move it, matching Python dict semantics.
#[derive(Clone, Default, Debug)]
struct Scope {
    order: Vec<String>,
    map: HashMap<String, Val>,
}

impl Scope {
    fn insert(&mut self, name: &str, v: Val) {
        if !self.map.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.map.insert(name.to_string(), v);
    }

    fn get(&self, name: &str) -> Option<&Val> {
        self.map.get(name)
    }

    fn remove(&mut self, name: &str) {
        if self.map.remove(name).is_some() {
            self.order.retain(|n| n != name);
        }
    }

    /// Names in insertion order.
    fn names(&self) -> Vec<String> {
        self.order.clone()
    }
}

/// The path recorded in a generated `standard.py` helper's locations.
///
/// A literal, because this crate never reads `standard.py` -- the helpers whose bodies are
/// one line are reimplemented directly rather than parsed. If a helper ever needs its real
/// source, that is the moment to parse the file instead of extending this.
const STANDARD_PY: &str = "triton/language/standard.py";

struct CodeGen<'a> {
    sem: Semantic,
    py: &'a PyModule,
    spec: &'a KernelSpec,
    scope: Scope,
    /// Names annotated `: tl.constexpr` by an `AnnAssign` still awaiting their value.
    pending_annotation: HashMap<String, String>,
    /// Mangled symbols already emitted, so a helper called twice at the same constexpr
    /// arguments is generated once -- `if not self.module.has_function(fn_name)`
    /// (`code_generator.py:1372`).
    generated: std::collections::HashSet<String>,
    /// `return` statements seen in the function currently being built, awaiting
    /// `handle_returns`.
    return_sites: Vec<ReturnSite>,
    /// Result types of already-generated symbols, so a second call needs no regeneration --
    /// Triton's `self.function_ret_types`.
    fn_ret_types: HashMap<String, Vec<Type>>,
}

/// A `return` recorded by `visit_Return`, to be turned into a terminator by
/// `handle_returns`.
struct ReturnSite {
    /// Index of the block the `return` statement was in.
    block: usize,
    value: Val,
    loc: Loc,
}

impl<'a> CodeGen<'a> {
    fn loc_of(&self, pos: Pos) -> Loc {
        // MLIR columns are 1-based; `col_offset` is 0-based. See the module docs.
        self.sem.file_loc(pos.line, pos.col + 1)
    }

    /// Run `f` with the location set from `pos`, restoring it afterwards -- the
    /// save/restore `CodeGenerator.visit` does.
    fn at<T>(&mut self, pos: Pos, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let saved = self.sem.loc.clone();
        self.sem.loc = self.loc_of(pos);
        let r = f(self);
        self.sem.loc = saved;
        r
    }

    /// Attach `pos` to an error that has none, so every refusal points at source.
    fn located(e: Error, pos: Pos) -> Error {
        if e.line == 0 && e.col == 0 {
            Error::new(e.message, pos.line, pos.col)
        } else {
            e
        }
    }

    // ===------------------------------------------------------------------===//
    //                            visit_FunctionDef
    // ===------------------------------------------------------------------===//

    fn visit_kernel(&mut self, f: &FunctionDef) -> Result<()> {
        let func_loc = self.loc_of(f.pos);
        self.sem.loc = func_loc.clone();

        // Bind parameters: a pointer/scalar becomes a block argument, a constexpr becomes a
        // compile-time value.
        let mut arg_types = Vec::new();
        let mut block_args = Vec::new();
        for p in &f.params {
            let spec = self.spec.signature.get(&p.name).ok_or_else(|| {
                Error::new(
                    format!(
                        "parameter `{}` of kernel `{}` has no entry in the signature",
                        p.name, f.name
                    ),
                    p.pos.line,
                    p.pos.col,
                )
            })?;
            match spec {
                ArgSpec::Constexpr => {
                    let v = self.spec.constexprs.get(&p.name).ok_or_else(|| {
                        Error::new(
                            format!(
                                "parameter `{}` is declared constexpr but no value was bound \
                                 for it",
                                p.name
                            ),
                            p.pos.line,
                            p.pos.col,
                        )
                    })?;
                    self.scope.insert(&p.name, v.clone());
                }
                ArgSpec::Ptr(elem) => {
                    let ty = Type::ptr(elem.clone());
                    // A parameter's location is the FUNCTION's location wrapped in the
                    // parameter name -- `loc("a_ptr"(#loc))` in the goldens, where `#loc`
                    // is the `def` line. That is `visit_FunctionDef`'s
                    // `_maybe_set_loc_to_name(arg_value, arg_name)` applied to a block
                    // argument whose own loc is the function's.
                    let id = self
                        .sem
                        .new_block_arg(ty.clone(), func_loc.named(&p.name));
                    self.scope.insert(&p.name, Val::Ir(id));
                    arg_types.push(ty);
                    block_args.push(id);
                }
                ArgSpec::Scalar(t) => {
                    let id = self
                        .sem
                        .new_block_arg(t.clone(), func_loc.named(&p.name));
                    self.scope.insert(&p.name, Val::Ir(id));
                    arg_types.push(t.clone());
                    block_args.push(id);
                }
            }
        }

        let saved_returns = std::mem::take(&mut self.return_sites);
        self.sem.push_frame();
        self.visit_body(&f.body)?;
        self.sem.loc = func_loc.clone();
        self.sem.name_prefix = None;
        let ret_types = self.handle_returns(&func_loc)?;
        let mut blocks = self.sem.pop_frame();
        if let Some(b) = blocks.first_mut() {
            b.args = block_args;
        }
        self.return_sites = saved_returns;

        // The kernel goes FIRST, ahead of any private helper generated while walking it --
        // which is the order `module.push_back` produces in Triton, since the kernel's
        // `tt.func` is pushed before its body is visited.
        self.sem.module.funcs.insert(
            0,
            Func {
                name: f.name.clone(),
                visibility: Visibility::Public,
                noinline: false,
                arg_types,
                ret_types,
                body: crate::ttir::Region { blocks },
                loc: func_loc.clone(),
            },
        );
        self.sem.module.loc = func_loc;
        Ok(())
    }

    /// `handle_returns` (`code_generator.py:606`), which is where the terminators are
    /// actually written.
    ///
    /// For each recorded `return`, a `tt.return` goes back into the block that `return`
    /// statement was in. Then, into the LAST (unreachable) block, a `ub.poison` per result
    /// plus a second `tt.return`. A function with no `return` at all -- every kernel -- gets a
    /// single operand-less `tt.return` and no extra block.
    ///
    /// Returns the function's result types.
    fn handle_returns(&mut self, func_loc: &Loc) -> Result<Vec<Type>> {
        let sites = std::mem::take(&mut self.return_sites);
        if sites.is_empty() {
            // A kernel's `tt.return` carries the FUNCTION's location, not the last
            // statement's -- the goldens show `tt.return loc(#loc)` with `#loc` the `def`
            // line.
            let op = crate::ttir::Op::new("tt.return", func_loc.clone());
            self.sem.emit_raw(op);
            return Ok(Vec::new());
        }

        // Every fixture returns from exactly one place. Several `return`s would need
        // `decide_return_type`'s common-type reduction across them, so more than one is
        // refused rather than assumed compatible.
        if sites.len() > 1 {
            return Err(Error::new(
                "more than one `return` in a @triton.jit function is not yet lowered: Triton \
                 reduces the return types to a common type across all of them \
                 (`decide_return_type`, `code_generator.py:556`), and no fixture has two.",
                0,
                0,
            ));
        }
        let site = &sites[0];
        let handles = site.value.ir_handles();
        let ret_types: Vec<Type> = handles.iter().map(|h| self.sem.ty(*h)).collect();

        let mut ret = crate::ttir::Op::new("tt.return", site.loc.clone());
        ret.operands = handles;
        self.sem.emit_in_block(site.block, ret);

        // The unreachable block `visit_Return` already started: one poison per result, then
        // the second terminator. Both carry the FUNCTION's location.
        self.sem.loc = func_loc.clone();
        let mut poisons = Vec::with_capacity(ret_types.len());
        for t in &ret_types {
            poisons.push(self.sem.poison(t.clone()));
        }
        let mut dead = crate::ttir::Op::new("tt.return", func_loc.clone());
        dead.operands = poisons;
        self.sem.emit_raw(dead);
        Ok(ret_types)
    }

    // ===------------------------------------------------------------------===//
    //                              Statements
    // ===------------------------------------------------------------------===//

    fn visit_body(&mut self, body: &[Stmt]) -> Result<()> {
        for s in body {
            self.visit_stmt(s)?;
            if matches!(s, Stmt::Return { .. }) {
                // `visit_compound_statement` stops at a return: the rest is dead code.
                break;
            }
        }
        Ok(())
    }

    fn visit_stmt(&mut self, s: &Stmt) -> Result<()> {
        match s {
            Stmt::Assign { target, value, pos } => self.at(*pos, |me| {
                let v = me.visit_assign_rhs(target, value)?;
                me.assign_target(target, v)
            }),
            Stmt::AnnAssign {
                target,
                annotation,
                value,
                pos,
            } => self.at(*pos, |me| {
                if let (AssignTarget::Name { id, .. }, Some(a)) = (target, annotation.dotted()) {
                    me.pending_annotation.insert(id.clone(), a);
                }
                match value {
                    Some(v) => {
                        let val = me.visit_assign_rhs(target, v)?;
                        me.assign_target(target, val)
                    }
                    // A bare annotation binds nothing.
                    None => Ok(()),
                }
            }),
            // `visit_AugAssign` rewrites `x op= e` into `x = x op e` and re-visits.
            Stmt::AugAssign {
                target,
                op,
                value,
                pos,
            } => {
                let name = match target {
                    AssignTarget::Name { id, .. } => id.clone(),
                    other => {
                        return Err(Error::new(
                            "augmented assignment to a tuple target is not supported",
                            other.pos().line,
                            other.pos().col,
                        ))
                    }
                };
                let rewritten = Stmt::Assign {
                    target: target.clone(),
                    value: Expr::BinOp {
                        left: Box::new(Expr::Name {
                            id: name,
                            pos: *pos,
                        }),
                        op: *op,
                        right: Box::new(value.clone()),
                        pos: *pos,
                    },
                    pos: *pos,
                };
                self.visit_stmt(&rewritten)
            }
            Stmt::Expr { value, pos } => self.at(*pos, |me| {
                me.visit_expr(value)?;
                Ok(())
            }),
            // `visit_Return` (`code_generator.py:544`) does NOT emit the terminator: it
            // RECORDS the value and its insertion point, then starts a fresh block so
            // anything after the `return` is unreachable but still well formed.
            // `handle_returns` writes the real `tt.return` back into the recorded block
            // afterwards, once the common return type is known.
            Stmt::Return { value, pos } => self.at(*pos, |me| {
                let v = match value {
                    None => Val::None,
                    Some(e) => me.visit_expr(e)?,
                };
                let block = me.sem.current_block();
                me.return_sites.push(ReturnSite {
                    block,
                    value: v,
                    loc: me.sem.loc.clone(),
                });
                me.sem.start_block(Vec::new());
                Ok(())
            }),
            Stmt::Pass { .. } => Ok(()),
            Stmt::For {
                target,
                iter,
                body,
                orelse,
                pos,
            } => {
                if !orelse.is_empty() {
                    return Err(Error::new(
                        "`for ... else` is not supported",
                        pos.line,
                        pos.col,
                    ));
                }
                self.at(*pos, |me| me.visit_for(target, iter, body, *pos))
            }
            // `visit_If` (`code_generator.py:967`) splits on whether the condition is a
            // TENSOR or a compile-time value. A compile-time condition is evaluated here and
            // only the taken branch is visited -- no `scf.if` is emitted at all. Every `if`
            // in the fixtures is of that kind (`attention_flash.py`'s `if STAGE & 1:`, where
            // `STAGE` is a constexpr), which is why a runtime condition is refused below
            // rather than half-built.
            Stmt::If {
                test,
                body,
                orelse,
                pos,
            } => {
                let cond = self.at(*pos, |me| me.visit_expr(test))?;
                let taken = match &cond {
                    Val::Bool(b) => *b,
                    Val::Int(i) => *i != 0,
                    Val::Ir(_) => {
                        return Err(Error::new(
                            "an `if` on a RUNTIME value is not yet lowered: Triton emits an \
                             `scf.if` (or, when the branch contains a `return`, splits the \
                             enclosing function into blocks -- `visit_if_top_level`). No \
                             fixture has one; every `if` in them tests a constexpr and is \
                             resolved at compile time.",
                            pos.line,
                            pos.col,
                        ))
                    }
                    other => {
                        return Err(Error::new(
                            format!(
                                "`if` conditionals accept a bool or an int, not a {}",
                                other.kind_name()
                            ),
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.visit_body(if taken { body } else { orelse })
            }
        }
    }

    // ===------------------------------------------------------------------===//
    //                               visit_For
    // ===------------------------------------------------------------------===//

    /// `visit_For` (`code_generator.py:1204`) for the `range` / `tl.range` forms.
    ///
    /// The emission order is not obvious and is taken from the oracle rather than invented.
    /// For `for n in tl.range(0, D_FF, BLOCK_N)` the golden shows, in order:
    ///
    /// ```text
    /// %c0_i32   = arith.constant 0 : i32        <- to_tensor(start)
    /// %c256_i32 = arith.constant 256 : i32      <- to_tensor(end)
    /// %c64_i32  = arith.constant 64 : i32       <- to_tensor(step)
    /// %0 = arith.bitcast %c0_i32   : i32 to i32 <- create_int_cast, lower bound
    /// %1 = arith.bitcast %c256_i32 : i32 to i32 <- create_int_cast, upper bound
    /// %2 = arith.bitcast %c64_i32  : i32 to i32 <- create_int_cast, step
    /// %3 = ub.poison : i32                      <- the induction-variable placeholder
    /// %acc_31 = scf.for %n = %0 to %1 step %2 iter_args(%acc_33 = %acc) -> (...) : i32 {
    /// ```
    ///
    /// The three `arith.bitcast`s are `i32 to i32` -- no-ops -- because `visit_For` calls
    /// `create_int_cast` DIRECTLY rather than going through `cast`, and `create_int_cast`
    /// emits a bitcast when the widths already agree. They are faithfully reproduced.
    ///
    /// None of these carry the loop variable's name: they are built outside any assignment,
    /// so `arith.constant`'s own `%c<value>_<type>` naming shows through. The `scf.for`
    /// itself DOES get a name, from `_maybe_set_loc_to_name` applied to each carried result
    /// in turn -- which nests, so a two-carry loop ends up named after the second.
    fn visit_for(&mut self, target: &str, iter: &Expr, body: &[Stmt], pos: Pos) -> Result<()> {
        let (start, end, step, is_static) = self.loop_bounds(iter, pos)?;

        // `tl.static_range` is fully unrolled at compile time and emits no loop at all.
        if is_static {
            let (s, e, st) = match (start.as_int(), end.as_int(), step.as_int()) {
                (Some(s), Some(e), Some(st)) if st != 0 => (s, e, st),
                _ => {
                    return Err(Error::new(
                        "tl.static_range needs compile-time constant bounds and a non-zero \
                         step",
                        pos.line,
                        pos.col,
                    ))
                }
            };
            let mut i = s;
            while (st > 0 && i < e) || (st < 0 && i > e) {
                self.scope.insert(target, Val::Int(i));
                self.visit_body(body)?;
                i += st;
            }
            return Ok(());
        }

        // A negative constant step is flipped, since scf.for cannot express one
        // (`code_generator.py:1247`).
        let (start, end, step, negative_step) = match step.as_int() {
            Some(v) if v < 0 => (end, start, Val::Int(-v), true),
            _ => (start, end, step, false),
        };
        if negative_step {
            return Err(Error::new(
                "a negative loop step is not yet lowered: Triton flips the bounds and \
                 recomputes the induction variable inside the body \
                 (`code_generator.py:1319`), which no fixture exercises, so it is refused \
                 rather than guessed at",
                pos.line,
                pos.col,
            ));
        }

        let lb = self.sem.to_tensor(&start).map_err(|e| Self::located(e, pos))?;
        let ub = self.sem.to_tensor(&end).map_err(|e| Self::located(e, pos))?;
        let st = self.sem.to_tensor(&step).map_err(|e| Self::located(e, pos))?;
        for (v, what) in [(lb, "lower bound"), (ub, "upper bound"), (st, "step")] {
            let ty = self.sem.ty(v);
            if !ty.is_int() {
                return Err(Error::new(
                    format!("for loop {what} must be an integer, got {ty}"),
                    pos.line,
                    pos.col,
                ));
            }
            if ty.is_tensor() {
                return Err(Error::new(
                    format!("for loop {what} must be a scalar, got {ty}"),
                    pos.line,
                    pos.col,
                ));
            }
        }

        // The induction type is the promotion of all three bounds.
        let iv_ty = {
            let a = self.sem.ty(lb);
            let b = self.sem.ty(ub);
            let c = self.sem.ty(st);
            let t = self
                .sem
                .integer_promote(a.scalar(), b.scalar())
                .map_err(|e| Self::located(e, pos))?;
            self.sem
                .integer_promote(&t, c.scalar())
                .map_err(|e| Self::located(e, pos))?
        };
        let signed = iv_ty.signedness() != Signedness::Unsigned;
        let lb = self.sem.int_cast(lb, &iv_ty, signed);
        let ub = self.sem.int_cast(ub, &iv_ty, signed);
        let st = self.sem.int_cast(st, &iv_ty, signed);

        // The induction-variable placeholder. Triton creates a `ub.poison`, binds the loop
        // variable to it for the dry run, and later replaces its uses with the real
        // induction variable. Here the dry run's ops are discarded outright, so the
        // placeholder only ever needs to type-check -- but it IS emitted, because the
        // oracle emits it.
        let _placeholder = self.sem.poison(iv_ty.clone());

        // ---- carry discovery: `_find_carries`' dry run --------------------------
        let liveins = self.scope.clone();
        self.scope.insert(target, Val::Ir(_placeholder));
        self.sem.push_frame();
        let dry = self.visit_body(body);
        self.sem.discard_frame();
        dry?;
        // `_find_carries` compares the FLATTENED IR HANDLES of each livein before and after
        // (`code_generator.py:492`): `if live_handles != loop_handles`. Comparing `Val`s
        // structurally instead would be wrong for a descriptor, which is one language value
        // over several handles.
        let mut carries: Vec<String> = Vec::new();
        for name in liveins.names() {
            if name == target {
                continue;
            }
            let (before, after) = match (liveins.get(&name), self.scope.get(&name)) {
                (Some(b), Some(a)) => (b.clone(), a.clone()),
                _ => continue,
            };
            if !before.is_triton_value() {
                // A constexpr reassigned in the loop cannot be carried
                // (`_verify_loop_carried_variable`, `code_generator.py:1116`).
                if !same_const(&before, &after) {
                    return Err(Error::new(
                        format!(
                            "`{name}` is a compile-time value reassigned inside a loop, which \
                             cannot be loop-carried (it would need a different value per \
                             iteration at compile time)"
                        ),
                        pos.line,
                        pos.col,
                    ));
                }
                continue;
            }
            if before.ir_handles() == after.ir_handles() {
                continue;
            }
            // A carry that is not a single IR value would need its handles flattened into
            // `iter_args` and reassembled from the block arguments. No fixture carries a
            // descriptor or a tuple through a loop, so it is refused rather than guessed at.
            if !matches!(after, Val::Ir(_)) {
                return Err(Error::new(
                    format!(
                        "`{name}` is a {} that changes inside the loop, so it would have to be \
                         loop-carried as several flattened handles and reassembled from the \
                         block arguments. No fixture does this, so it is refused rather than \
                         half-implemented.",
                        after.kind_name()
                    ),
                    pos.line,
                    pos.col,
                ));
            }
            carries.push(name);
        }
        // Reset the scope, discarding everything the dry run bound.
        self.scope = liveins.clone();

        // ---- the real loop ------------------------------------------------------
        let init: Vec<crate::ttir::ValueId> = carries
            .iter()
            .map(|n| match self.scope.get(n) {
                Some(Val::Ir(id)) => *id,
                _ => unreachable!("a carry is an IR value by construction"),
            })
            .collect();
        let carry_types: Vec<Type> = init.iter().map(|v| self.sem.ty(*v)).collect();

        let for_loc = self.sem.loc.clone();
        self.sem.push_frame();
        // Region block arguments: the induction variable, then one per carry.
        let iv = self
            .sem
            .new_block_arg(iv_ty.clone(), for_loc.clone().named(target));
        let mut block_args = vec![iv];
        // The loop variable stays bound to the PLACEHOLDER while the body is visited; uses are
        // rewritten to `iv` afterwards (`code_generator.py:1324`). That ordering is observable:
        // `tl.multiple_of` on the induction variable stamps `tt.divisibility` on the poison,
        // which is exactly what the attention golden shows.
        self.scope.insert(target, Val::Ir(_placeholder));
        for (n, t) in carries.iter().zip(&carry_types) {
            let a = self.sem.new_block_arg(t.clone(), for_loc.clone().named(n));
            self.scope.insert(n, Val::Ir(a));
            block_args.push(a);
        }
        self.visit_body(body)?;

        // `scf.yield` carries the CURRENT value of each carried name, in carry order, and
        // takes the `for` statement's own (unnamed) location.
        let yields: Vec<crate::ttir::ValueId> = carries
            .iter()
            .map(|n| match self.scope.get(n) {
                Some(Val::Ir(id)) => Ok(*id),
                other => Err(Error::new(
                    format!(
                        "loop-carried `{n}` is a {} at the end of the body but was a tensor \
                         at the start; a carried value may not change kind",
                        other.map(|v| v.kind_name()).unwrap_or("nothing")
                    ),
                    pos.line,
                    pos.col,
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        if !yields.is_empty() {
            let mut y = crate::ttir::Op::new("scf.yield", for_loc.clone());
            y.operands = yields;
            self.sem.emit_raw(y);
        }
        let mut blocks = self.sem.pop_frame();
        if let Some(b) = blocks.first_mut() {
            b.args = block_args;
        }
        // `iv_placeholder.replace_all_uses_with(iv)`, now that the body is built.
        Semantic::replace_uses(&mut blocks, _placeholder, iv);

        // Restore the enclosing scope before binding the results.
        self.scope = liveins;

        let mut operands = vec![lb, ub, st];
        operands.extend_from_slice(&init);
        let results = self.sem.emit_with_region(
            "scf.for",
            &operands,
            &carry_types,
            &[],
            blocks,
            for_loc,
        );

        // Bind results, and name each in turn. `_maybe_set_loc_to_name` NESTS, and setting a
        // result's location sets the OP's, so a two-carry loop's `scf.for` ends up located
        // `loc("u"(loc("g"(<for stmt>))))` -- measured in the golden as
        // `#loc62 = loc("u"(#loc48))`, `#loc48 = loc("g"(#loc13))`.
        let block = self.sem.current_block();
        let op_index = self.sem.last_op_index();
        for (n, r) in carries.iter().zip(&results) {
            self.scope.insert(n, Val::Ir(*r));
            self.sem.name_op_results(block, op_index, n, true);
        }
        Ok(())
    }

    /// The `(start, end, step, is_static_range)` of a `for` statement's iterable.
    fn loop_bounds(&mut self, iter: &Expr, pos: Pos) -> Result<(Val, Val, Val, bool)> {
        let (func, args, keywords) = match iter {
            Expr::Call {
                func,
                args,
                keywords,
                ..
            } => (func, args, keywords),
            _ => {
                return Err(Error::new(
                    "a `for` loop must iterate `range(...)`, `tl.range(...)` or \
                     `tl.static_range(...)`",
                    pos.line,
                    pos.col,
                ))
            }
        };
        let dotted = func.dotted().unwrap_or_default();
        let is_static = dotted.ends_with("static_range");
        let is_tl_range = dotted.ends_with(".range");
        let is_builtin_range = dotted == "range";
        if !(is_static || is_tl_range || is_builtin_range) {
            return Err(Error::new(
                format!(
                    "a `for` loop must iterate `range(...)`, `tl.range(...)` or \
                     `tl.static_range(...)`, not `{dotted}(...)`"
                ),
                pos.line,
                pos.col,
            ));
        }
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.visit_expr(a)?);
        }
        // `tl.range`'s tuning keywords (num_stages, loop_unroll_factor, flatten, ...) become
        // attributes on `scf.for`. None of the fixtures pass one; a future kernel that does
        // must be refused rather than have the hint silently dropped, since dropping it
        // changes performance without changing correctness and would be invisible.
        for (k, _) in keywords {
            if k == "step" {
                continue;
            }
            return Err(Error::new(
                format!(
                    "`{dotted}(..., {k}=...)` is not lowered: it becomes a `tt.{k}` attribute \
                     on `scf.for`, and silently dropping a scheduling hint would change \
                     performance invisibly. No fixture uses one."
                ),
                pos.line,
                pos.col,
            ));
        }
        let step_kw = keywords
            .iter()
            .find(|(k, _)| k == "step")
            .map(|(_, v)| v.clone());
        let step = match step_kw {
            Some(e) => self.visit_expr(&e)?,
            None => vals.get(2).cloned().unwrap_or(Val::Int(1)),
        };
        // `range(a)` means `range(0, a)`; `range(a, b)` and `range(a, b, c)` are literal.
        let (start, end) = match vals.len() {
            0 => {
                return Err(Error::new(
                    format!("`{dotted}()` needs at least one bound"),
                    pos.line,
                    pos.col,
                ))
            }
            1 => (Val::Int(0), vals[0].clone()),
            _ => (vals[0].clone(), vals[1].clone()),
        };
        Ok((start, end, step, is_static))
    }

    /// Visit an assignment's right-hand side with the target's name installed as the loc
    /// prefix (`code_generator.py:756`).
    fn visit_assign_rhs(&mut self, target: &AssignTarget, value: &Expr) -> Result<Val> {
        match target {
            AssignTarget::Name { id, .. } => {
                let saved = self.sem.name_prefix.clone();
                self.sem.name_prefix = Some(id.clone());
                // `_sanitize_target_value` runs INSIDE the `_name_loc_prefix` block
                // (`code_generator.py:756-757`), so a bare number materialized here carries the
                // target's name. Measured: `y_dim = Z * H * N_CTX` folds to a constexpr and
                // then becomes `%y_dim = arith.constant 1024 : i32 loc("y_dim"(...))`. Doing
                // it after the prefix was restored left those constants unnamed.
                let r = self.visit_expr(value).and_then(|v| {
                    let annotated_constexpr = self
                        .pending_annotation
                        .get(id)
                        .map(|a| a.ends_with("constexpr"))
                        .unwrap_or(false);
                    if annotated_constexpr {
                        Ok(v)
                    } else {
                        self.sanitize_value(v, value.pos())
                    }
                });
                self.sem.name_prefix = saved;
                r
            }
            _ => {
                let v = self.visit_expr(value)?;
                self.sanitize_value(v, value.pos())
            }
        }
    }

    /// `_sanitize_value` (`code_generator.py:729`).
    ///
    /// # A BARE PYTHON NUMBER ASSIGNED TO A NAME BECOMES AN IR CONSTANT.
    ///
    /// This is the least obvious rule in the whole walk, and it is not cosmetic:
    ///
    /// ```text
    /// value = _unwrap_if_constexpr(value)
    /// if value is not None and not _is_triton_value(value) \
    ///        and not isinstance(value, (language.dtype, language.tuple)):
    ///     value = self.semantic.to_tensor(value)
    /// ```
    ///
    /// So `qk_scale = sm_scale`, where `sm_scale` is a constexpr `1.0`, does NOT bind a
    /// constexpr -- it unwraps to a plain Python float and is materialized as
    /// `arith.constant 1.000000e+00 : f32`. `attention_flash.py` depends on this, and the
    /// oracle proves it two ways: its raw TTIR contains `arith.constant 1.44269502 : f32`
    /// (the f32 rounding of the source's `1.44269504`) with an `arith.mulf` on it, and its
    /// `_attn_fwd_inner` symbol mangles that argument as **`fp32`**, a runtime type, rather
    /// than as `c1.44269504`.
    ///
    /// The escape hatch is an explicit `x: tl.constexpr = ...` annotation, which
    /// `_sanitize_target_value` honours by keeping the value compile-time.
    ///
    /// A `dtype`, a tuple, and `None` pass through untouched.
    fn sanitize_value(&mut self, v: Val, pos: Pos) -> Result<Val> {
        match &v {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) => {
                let id = self.sem.to_tensor(&v).map_err(|e| Self::located(e, pos))?;
                Ok(Val::Ir(id))
            }
            Val::Str(s) => Err(Error::new(
                format!("cannot convert the string {s:?} to a tensor"),
                pos.line,
                pos.col,
            )),
            // A tuple is sanitized ELEMENTWISE (`_apply_to_tuple_values`), so a tuple of
            // numbers becomes a tuple of constants.
            Val::Seq(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items.clone() {
                    out.push(self.sanitize_value(it, pos)?);
                }
                Ok(Val::Seq(out))
            }
            _ => Ok(v),
        }
    }

    fn assign_target(&mut self, target: &AssignTarget, v: Val) -> Result<()> {
        match target {
            AssignTarget::Name { id, .. } => {
                self.pending_annotation.remove(id);
                self.scope.insert(id, v);
                Ok(())
            }
            AssignTarget::Tuple { elts, pos } => match v {
                Val::Seq(vals) if vals.len() == elts.len() => {
                    for (t, x) in elts.iter().zip(vals) {
                        self.assign_target(t, x)?;
                    }
                    Ok(())
                }
                Val::Seq(vals) => Err(Error::new(
                    format!(
                        "cannot unpack {} value(s) into {} target(s)",
                        vals.len(),
                        elts.len()
                    ),
                    pos.line,
                    pos.col,
                )),
                other => Err(Error::new(
                    format!(
                        "cannot unpack a {} into {} targets",
                        other.kind_name(),
                        elts.len()
                    ),
                    pos.line,
                    pos.col,
                )),
            },
        }
    }

    // ===------------------------------------------------------------------===//
    //                             Expressions
    // ===------------------------------------------------------------------===//

    fn visit_expr(&mut self, e: &Expr) -> Result<Val> {
        let pos = e.pos();
        self.at(pos, |me| me.visit_expr_inner(e))
    }

    fn visit_expr_inner(&mut self, e: &Expr) -> Result<Val> {
        let pos = e.pos();
        match e {
            Expr::Constant { value, .. } => Ok(match value {
                Literal::Int(v) => Val::Int(*v),
                Literal::Float(v) => Val::Float(*v),
                Literal::Bool(v) => Val::Bool(*v),
                Literal::Str(v) => Val::Str(v.clone()),
                Literal::None => Val::None,
            }),
            Expr::Name { id, .. } => self.deref_name(id, pos),
            Expr::List { elts, .. } | Expr::Tuple { elts, .. } => {
                let mut out = Vec::with_capacity(elts.len());
                for x in elts {
                    out.push(self.visit_expr(x)?);
                }
                Ok(Val::Seq(out))
            }
            // A slice. ONLY THE UNBOUNDED `:` EXISTS, and this used to be a silent hole.
            //
            // Triton's `tensor.__getitem__` (`python/triton/language/core.py`) accepts exactly
            // two subscript items: `None`, which is an `expand_dims`, and a FULL `:`, which is
            // a no-op. Anything carrying a start, a stop or a step raises
            //
            //     ValueError: unsupported tensor index: <triton.language.core.slice object ...>
            //
            // MEASURED, on `x[:, :HALF]` -- the obvious transcription of Hugging Face's
            // `rotate_half`, which is why `rope.py` exists in the form it does.
            //
            // This arm returned `Val::Slice` for ANY slice, and `subscript` treats `Val::Slice`
            // as a no-op, so `x[:, :64]` COMPILED HERE and yielded the whole 128-wide tile:
            // a wrong answer with no diagnostic, from a kernel the oracle refuses outright.
            // Now the bounds are checked and the refusal names the construct.
            // `tests/slicing.rs` is the control.
            Expr::Slice {
                lower,
                upper,
                step,
                pos,
            } => {
                if lower.is_some() || upper.is_some() || step.is_some() {
                    return Err(Error::new(
                        "a BOUNDED tensor slice is not supported: Triton's own \
                         `tensor.__getitem__` accepts only `None` and a full `:`, and refuses \
                         anything with a start, stop or step (`unsupported tensor index`). \
                         Read the two halves as two descriptor loads at different offsets, \
                         which is what `rope.py` does.",
                        pos.line,
                        pos.col,
                    ));
                }
                Ok(Val::Slice)
            }
            Expr::Attribute { value, attr, .. } => {
                // A `tl.*` chain that names a dtype, or a module path.
                if let Some(dotted) = e.dotted() {
                    if let Some(v) = self.resolve_tl_attribute(&dotted) {
                        return Ok(v);
                    }
                }
                let recv = self.visit_expr(value)?;
                // Tensor PROPERTIES. These are attribute accesses, not calls, so they are not
                // among the census's 33 call targets -- but they are `Attribute` nodes, which
                // the AST census does count, so they are in scope. `attention_flash.py` uses
                // `q.T` and `k.T`.
                match (attr.as_str(), &recv) {
                    // `tensor.T` is `trans(self)` with the default permutation, which reverses
                    // the dimensions. Golden:
                    // `tt.trans %q {order = array<i32: 1, 0>} : tensor<64x128xf16> ->
                    //  tensor<128x64xf16>`
                    ("T", Val::Ir(id)) => {
                        let rank = self.sem.ty(*id).shape().len();
                        if rank < 2 {
                            return Err(Error::new(
                                format!("`.T` needs a rank-2 or higher tensor, got rank {rank}"),
                                pos.line,
                                pos.col,
                            ));
                        }
                        let order: Vec<i64> = (0..rank as i64).rev().collect();
                        return self
                            .sem
                            .trans(*id, &order)
                            .map(Val::Ir)
                            .map_err(|e| Self::located(e, pos));
                    }
                    ("dtype", Val::Ir(id)) => {
                        return Ok(Val::Dtype(self.sem.ty(*id).scalar().clone()))
                    }
                    ("shape", Val::Ir(id)) => {
                        let s = self.sem.ty(*id).shape().to_vec();
                        return Ok(Val::Seq(
                            s.into_iter().map(|d| Val::Int(d as i128)).collect(),
                        ));
                    }
                    _ => {}
                }
                Err(Error::new(
                    format!(
                        "attribute `.{attr}` on a {} is not supported",
                        recv.kind_name()
                    ),
                    pos.line,
                    pos.col,
                ))
            }
            Expr::BinOp {
                left, op, right, ..
            } => {
                let l = self.visit_expr(left)?;
                let r = self.visit_expr(right)?;
                self.binary(*op, l, r, pos)
            }
            Expr::UnaryOp { op, operand, .. } => {
                let v = self.visit_expr(operand)?;
                self.unary(*op, v, pos)
            }
            Expr::Compare {
                left, op, right, ..
            } => {
                let l = self.visit_expr(left)?;
                let r = self.visit_expr(right)?;
                self.comparison(*op, l, r, pos)
            }
            Expr::Subscript { value, index, .. } => {
                let base = self.visit_expr(value)?;
                let idx = self.visit_expr(index)?;
                self.subscript(base, idx, pos)
            }
            Expr::Call {
                func,
                args,
                keywords,
                ..
            } => self.visit_call(func, args, keywords, pos),
        }
    }

    fn deref_name(&mut self, id: &str, pos: Pos) -> Result<Val> {
        if let Some(v) = self.scope.get(id) {
            return Ok(v.clone());
        }
        // A module-level global. Triton REFUSES this, and the refusal names the variable --
        // reproduced here, quoting Triton's own wording, because `bias_add_f32.py` depends
        // on being refused for exactly this reason.
        if self.py.globals.iter().any(|(n, _)| n == id) {
            return Err(Error::new(
                format!(
                    "Cannot access global variable {id} from within @jit'ed function. \
                     Triton kernels can only access global variables that are instantiated \
                     as constexpr (`x = triton.language.constexpr(42)`)."
                ),
                pos.line,
                pos.col,
            ));
        }
        Err(Error::new(
            format!("`{id}` is not defined"),
            pos.line,
            pos.col,
        ))
    }

    /// `tl.float16` and friends. Returns `None` when the chain is not a dtype.
    fn resolve_tl_attribute(&self, dotted: &str) -> Option<Val> {
        for alias in census::TL_ALIASES {
            if let Some(rest) = dotted.strip_prefix(&format!("{alias}.")) {
                if census::TL_DTYPES.contains(&rest) {
                    return parse_dtype(rest).ok().map(Val::Dtype);
                }
            }
        }
        None
    }

    fn binary(&mut self, op: BinOpKind, l: Val, r: Val, pos: Pos) -> Result<Val> {
        // Constant folding: Triton's `constexpr` arithmetic happens in Python, so two
        // compile-time numbers never reach the builder.
        if l.is_number() && r.is_number() {
            if let Some(v) = fold_binary(op, &l, &r) {
                return Ok(v);
            }
        }
        let out = match op {
            BinOpKind::Add => self.sem.add(&l, &r, true),
            BinOpKind::Sub => self.sem.sub(&l, &r, true),
            BinOpKind::Mult => self.sem.mul(&l, &r, true),
            BinOpKind::Div => self.sem.truediv(&l, &r),
            BinOpKind::FloorDiv => self.sem.floordiv(&l, &r),
            BinOpKind::Mod => self.sem.modulo(&l, &r),
            BinOpKind::BitAnd => self.sem.bitwise(&l, &r, Bitwise::And),
            BinOpKind::BitOr => self.sem.bitwise(&l, &r, Bitwise::Or),
            BinOpKind::BitXor => self.sem.bitwise(&l, &r, Bitwise::Xor),
            BinOpKind::LShift => self.sem.bitwise(&l, &r, Bitwise::Shl),
            BinOpKind::RShift => self.sem.bitwise(&l, &r, Bitwise::Shr),
            BinOpKind::Pow | BinOpKind::MatMult => {
                return Err(Error::new(
                    format!(
                        "the `{}` operator ({}) is not supported",
                        op.py_symbol(),
                        op.dunder()
                    ),
                    pos.line,
                    pos.col,
                ))
            }
        };
        out.map(Val::Ir).map_err(|e| Self::located(e, pos))
    }

    fn unary(&mut self, op: UnaryOpKind, v: Val, pos: Pos) -> Result<Val> {
        match op {
            UnaryOpKind::USub => match &v {
                Val::Int(x) => Ok(Val::Int(-x)),
                Val::Float(x) => Ok(Val::Float(-x)),
                Val::Ir(id) => self
                    .sem
                    .minus_ir(*id)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos)),
                other => Err(Error::new(
                    format!("unary `-` is not defined on a {}", other.kind_name()),
                    pos.line,
                    pos.col,
                )),
            },
            UnaryOpKind::UAdd => Ok(v),
            UnaryOpKind::Not => match &v {
                Val::Bool(b) => Ok(Val::Bool(!b)),
                Val::Int(x) => Ok(Val::Bool(*x == 0)),
                other => Err(Error::new(
                    format!("`not` is not defined on a {}", other.kind_name()),
                    pos.line,
                    pos.col,
                )),
            },
            UnaryOpKind::Invert => Err(Error::new(
                "the `~` operator is not supported",
                pos.line,
                pos.col,
            )),
        }
    }

    fn comparison(&mut self, op: CmpKind, l: Val, r: Val, pos: Pos) -> Result<Val> {
        if l.is_const() && r.is_const() {
            if let Some(v) = fold_compare(op, &l, &r) {
                return Ok(v);
            }
        }
        let cop = match op {
            CmpKind::Eq => CmpOp::Eq,
            CmpKind::NotEq => CmpOp::Ne,
            CmpKind::Lt => CmpOp::Lt,
            CmpKind::LtE => CmpOp::Le,
            CmpKind::Gt => CmpOp::Gt,
            CmpKind::GtE => CmpOp::Ge,
        };
        self.sem
            .compare(&l, &r, cop)
            .map(Val::Ir)
            .map_err(|e| Self::located(e, pos))
    }

    fn subscript(&mut self, base: Val, idx: Val, pos: Pos) -> Result<Val> {
        match (&base, &idx) {
            // A sequence index: `SHAPE[0]`.
            (Val::Seq(items), Val::Int(i)) => {
                let n = items.len() as i128;
                let k = if *i < 0 { n + i } else { *i };
                items.get(k as usize).cloned().ok_or_else(|| {
                    Error::new(
                        format!("index {i} is out of range for a sequence of {n}"),
                        pos.line,
                        pos.col,
                    )
                })
            }
            // `x[:, None]` / `x[None, :]`: insert a size-1 axis where `None` appears.
            (Val::Ir(v), Val::Seq(items)) => {
                let mut cur = *v;
                for (axis, item) in items.iter().enumerate() {
                    match item {
                        Val::None => {
                            cur = self
                                .sem
                                .expand_dims(cur, axis as i64)
                                .map_err(|e| Self::located(e, pos))?;
                        }
                        Val::Slice => {}
                        other => {
                            return Err(Error::new(
                                format!(
                                    "a tensor subscript may only contain `:` and `None`, \
                                     got a {}",
                                    other.kind_name()
                                ),
                                pos.line,
                                pos.col,
                            ))
                        }
                    }
                }
                Ok(Val::Ir(cur))
            }
            _ => Err(Error::new(
                format!(
                    "subscripting a {} with a {} is not supported",
                    base.kind_name(),
                    idx.kind_name()
                ),
                pos.line,
                pos.col,
            )),
        }
    }

    // ===------------------------------------------------------------------===//
    //                                Calls
    // ===------------------------------------------------------------------===//

    fn visit_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[(String, Expr)],
        pos: Pos,
    ) -> Result<Val> {
        let target = census::classify_call(func, self.py)
            .map_err(|why| Error::new(why, pos.line, pos.col))?;

        // Method calls need the receiver, which is an expression to visit.
        if let CallTarget::Method { method, .. } = &target {
            let recv_expr = match func {
                Expr::Attribute { value, .. } => value.as_ref(),
                _ => {
                    return Err(Error::new(
                        "a method call must have a receiver",
                        pos.line,
                        pos.col,
                    ))
                }
            };
            let recv = self.visit_expr(recv_expr)?;
            let mut avals = Vec::with_capacity(args.len());
            for a in args {
                avals.push(self.visit_expr(a)?);
            }
            let kw = self.visit_keywords(keywords)?;
            return self.call_method(method, recv, &avals, &kw, pos);
        }

        let mut avals = Vec::with_capacity(args.len());
        for a in args {
            avals.push(self.visit_expr(a)?);
        }
        let kw = self.visit_keywords(keywords)?;

        match target {
            CallTarget::TlFunction(name) => self.call_tl(&name, &avals, &kw, pos),
            CallTarget::Builtin(name) => self.call_builtin(&name, &avals, pos),
            CallTarget::UserJit(name) => self.call_user_jit(&name, &avals, &kw, pos),
            CallTarget::Method { .. } => unreachable!("handled above"),
        }
    }

    fn visit_keywords(&mut self, keywords: &[(String, Expr)]) -> Result<Vec<(String, Val)>> {
        let mut out = Vec::with_capacity(keywords.len());
        for (k, v) in keywords {
            out.push((k.clone(), self.visit_expr(v)?));
        }
        Ok(out)
    }

    fn call_method(
        &mut self,
        method: &str,
        recv: Val,
        args: &[Val],
        _kw: &[(String, Val)],
        pos: Pos,
    ) -> Result<Val> {
        // `.load` / `.store` take only the descriptor HANDLE; the shape and stride values a
        // descriptor also carries (see `Val::Desc`) matter only when it crosses a function
        // boundary.
        let recv_id = match recv {
            Val::Ir(id) => id,
            Val::Desc { handle, .. } => handle,
            other => {
                return Err(Error::new(
                    format!("`.{method}` is not defined on a {}", other.kind_name()),
                    pos.line,
                    pos.col,
                ))
            }
        };
        match method {
            "load" => {
                let offsets = self.offsets_arg(args, 0, pos)?;
                self.sem
                    .descriptor_load(recv_id, &offsets)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            // NOTE THE ORDER. `semantic.descriptor_store` casts the VALUE first and only then
            // converts the offsets to IR values (`semantic.py:1086-1089`), so the implicit
            // `arith.truncf` comes BEFORE the offset constants. Building the offsets first put
            // one constant ahead of the truncate and diverged the op sequence by one position.
            "store" => {
                let value = args.get(1).ok_or_else(|| {
                    Error::new(
                        "`.store` needs an offsets list and a value",
                        pos.line,
                        pos.col,
                    )
                })?;
                let v = self
                    .sem
                    .to_tensor(value)
                    .map_err(|e| Self::located(e, pos))?;
                let elem = match self.sem.ty(recv_id) {
                    Type::TensorDesc(_, e) => (*e).clone(),
                    other => {
                        return Err(Error::new(
                            format!("`.store` needs a tensor descriptor, got {other}"),
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                let v = self.sem.cast(v, &elem).map_err(|e| Self::located(e, pos))?;
                let offsets = self.offsets_arg(args, 0, pos)?;
                self.sem
                    .descriptor_store(recv_id, &offsets, v)
                    .map_err(|e| Self::located(e, pos))?;
                Ok(Val::None)
            }
            // `desc.gather(x_offsets, y_offset)`. NOTE THE ORDER, which is Triton's: the
            // descriptor and the index vector are validated FIRST and `y_offset` is turned
            // into an IR value only after (`semantic.py:1165`), so the constant that a
            // literal `0` produces lands immediately before the gather and nowhere else.
            "gather" => {
                let x = match args.first() {
                    Some(v) => self.sem.to_tensor(v).map_err(|e| Self::located(e, pos))?,
                    None => {
                        return Err(Error::new(
                            "`.gather` needs an index vector and a column offset",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                let y = match args.get(1) {
                    Some(v) => self.make_scalar(v, Type::i32(), pos)?,
                    None => {
                        return Err(Error::new(
                            "`.gather` needs a column offset (its second argument)",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.sem
                    .descriptor_gather(recv_id, x, y)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            "to" => {
                let dt = match args.first() {
                    Some(Val::Dtype(t)) => t.clone(),
                    _ => {
                        return Err(Error::new(
                            "`.to(...)` needs a dtype, e.g. `tl.float32`",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.sem
                    .cast(recv_id, &dt)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            other => Err(Error::new(
                format!("method `.{other}` is not supported"),
                pos.line,
                pos.col,
            )),
        }
    }

    /// A descriptor `offsets` argument: a list of scalars, each made an i32 IR value.
    fn offsets_arg(
        &mut self,
        args: &[Val],
        which: usize,
        pos: Pos,
    ) -> Result<Vec<crate::ttir::ValueId>> {
        let seq = match args.get(which) {
            Some(Val::Seq(items)) => items.clone(),
            Some(other) => {
                return Err(Error::new(
                    format!(
                        "descriptor offsets must be a list, got a {}",
                        other.kind_name()
                    ),
                    pos.line,
                    pos.col,
                ))
            }
            None => {
                return Err(Error::new(
                    "descriptor offsets are missing",
                    pos.line,
                    pos.col,
                ))
            }
        };
        let mut out = Vec::with_capacity(seq.len());
        for v in &seq {
            out.push(self.make_scalar(v, Type::i32(), pos)?);
        }
        Ok(out)
    }

    /// A `shape` argument that must be compile-time constant, as `_shape_check_impl`
    /// requires.
    fn const_shape(&mut self, v: &Option<Val>, who: &str, pos: Pos) -> Result<Vec<i64>> {
        let items = match v {
            Some(Val::Seq(items)) => items.clone(),
            Some(other) => {
                return Err(Error::new(
                    format!(
                        "{who}'s shape must be a list or tuple, got a {}",
                        other.kind_name()
                    ),
                    pos.line,
                    pos.col,
                ))
            }
            None => {
                return Err(Error::new(
                    format!("{who} is missing its shape"),
                    pos.line,
                    pos.col,
                ))
            }
        };
        let mut out = Vec::with_capacity(items.len());
        for it in &items {
            match it.as_int() {
                Some(n) if n > 0 => out.push(n as i64),
                Some(n) => {
                    return Err(Error::new(
                        format!("{who}'s shape entries must be positive, got {n}"),
                        pos.line,
                        pos.col,
                    ))
                }
                None => {
                    return Err(Error::new(
                        format!(
                            "{who}'s shape must be compile-time constant, but one entry is a {}",
                            it.kind_name()
                        ),
                        pos.line,
                        pos.col,
                    ))
                }
            }
        }
        Ok(out)
    }

    /// `semantic.full` (`semantic.py:622`): `splat(make_scalar(value, dtype), shape)`.
    fn full(
        &mut self,
        shape: &[i64],
        value: &Val,
        dtype: Type,
        pos: Pos,
    ) -> Result<crate::ttir::ValueId> {
        let scalar = self.make_scalar(value, dtype, pos)?;
        self.sem
            .splat(scalar, shape)
            .map_err(|e| Self::located(e, pos))
    }

    /// `call_JitFunction` (`code_generator.py:1358`) for a `@triton.jit` function defined in
    /// the same module as the kernel.
    ///
    /// Triton does NOT inline it: the callee is generated as a `tt.func private` whose symbol
    /// encodes its fully-qualified Python name and every argument's mangled type, and the call
    /// site becomes a `tt.call`. Measured in the attention golden, whose symbol is
    /// `@attention_flash._attn_fwd_inner__fp16S64_128S_..._c256` -- note the module prefix is
    /// the FIXTURE's module name, since `get_full_name` is
    /// `f"{fn.__module__}.{fn.__qualname__}"` (`runtime/jit.py:461`).
    fn call_user_jit(
        &mut self,
        fn_name: &str,
        args: &[Val],
        kw: &[(String, Val)],
        pos: Pos,
    ) -> Result<Val> {
        let callee = self
            .py
            .function(fn_name)
            .ok_or_else(|| {
                Error::new(format!("no function `{fn_name}` in this module"), pos.line, pos.col)
            })?
            .clone();

        // `fn.signature.bind(*args, **kwargs)` then `apply_defaults()`, reordered into
        // parameter order.
        if args.len() > callee.params.len() {
            return Err(Error::new(
                format!(
                    "`{fn_name}` takes {} parameters but {} positional arguments were given",
                    callee.params.len(),
                    args.len()
                ),
                pos.line,
                pos.col,
            ));
        }
        let mut bound: Vec<Val> = Vec::with_capacity(callee.params.len());
        for (i, p) in callee.params.iter().enumerate() {
            if let Some(v) = args.get(i) {
                bound.push(v.clone());
                continue;
            }
            match kw.iter().find(|(k, _)| *k == p.name) {
                Some((_, v)) => bound.push(v.clone()),
                None => {
                    return Err(Error::new(
                        format!(
                            "`{fn_name}` parameter `{}` has no argument and no default",
                            p.name
                        ),
                        pos.line,
                        pos.col,
                    ))
                }
            }
        }

        // Mangle from each argument's LANGUAGE type -- a descriptor counts once here even
        // though it flattens to several IR values below.
        let mut mangles = Vec::with_capacity(bound.len());
        for v in &bound {
            let ty_of = |x: &Val| -> Option<Type> {
                match x {
                    Val::Ir(id) => Some(self.sem.ty(*id)),
                    Val::Desc { handle, .. } => Some(self.sem.ty(*handle)),
                    _ => None,
                }
            };
            let m = crate::mangle::mangle_arg(v, &ty_of).ok_or_else(|| {
                Error::new(
                    format!(
                        "an argument of `{fn_name}` is a {} and has no mangled spelling, so the \
                         callee's symbol name cannot be formed",
                        v.kind_name()
                    ),
                    pos.line,
                    pos.col,
                )
            })?;
            mangles.push(m);
        }
        let symbol = crate::mangle::mangle_fn(
            &format!("{}.{}", self.module_name(), fn_name),
            &mangles,
        );

        if !self.generated.contains(&symbol) {
            self.generate_user_jit(&symbol, &callee, &bound, pos)?;
        }
        let ret_types = self
            .fn_ret_types
            .get(&symbol)
            .cloned()
            .unwrap_or_default();

        let operands: Vec<crate::ttir::ValueId> =
            bound.iter().flat_map(|v| v.ir_handles()).collect();
        let results = self.sem.call(&symbol, &operands, &ret_types);
        Ok(match results.len() {
            0 => Val::None,
            1 => Val::Ir(results[0]),
            // A multi-result call reassembles into a tuple, which the caller then unpacks --
            // `acc, l_i, m_i = _attn_fwd_inner(...)`.
            _ => Val::Seq(results.into_iter().map(Val::Ir).collect()),
        })
    }

    /// The Python module name a fixture is imported under, which is its file stem. This is the
    /// prefix `get_full_name` puts on a generated symbol.
    fn module_name(&self) -> String {
        std::path::Path::new(&self.spec.file)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "kernel".to_string())
    }

    /// Build the private `tt.func` for a `@triton.jit` function in the kernel's module.
    fn generate_user_jit(
        &mut self,
        symbol: &str,
        callee: &FunctionDef,
        bound: &[Val],
        pos: Pos,
    ) -> Result<()> {
        // Generation happens mid-expression at the CALL SITE, so everything about the
        // caller's state has to be saved: the scope, the pending location and name prefix, and
        // the return sites.
        let saved_scope = std::mem::take(&mut self.scope);
        let saved_loc = self.sem.loc.clone();
        let saved_prefix = self.sem.name_prefix.take();
        let saved_returns = std::mem::take(&mut self.return_sites);

        let func_loc = self.loc_of(callee.pos);
        self.sem.loc = func_loc.clone();

        // Bind parameters. An IR-backed argument becomes block arguments (flattened, so a
        // descriptor becomes 1 + 2 * rank of them, named `p`, `p.shape.i`, `p.stride.i` as in
        // the golden); a compile-time argument is bound directly and contributes none.
        let mut arg_types = Vec::new();
        let mut block_args = Vec::new();
        for (p, v) in callee.params.iter().zip(bound) {
            match v {
                Val::Ir(id) => {
                    let ty = self.sem.ty(*id);
                    let a = self
                        .sem
                        .new_block_arg(ty.clone(), func_loc.named(&p.name));
                    self.scope.insert(&p.name, Val::Ir(a));
                    arg_types.push(ty);
                    block_args.push(a);
                }
                Val::Desc {
                    handle,
                    shape,
                    strides,
                } => {
                    let hty = self.sem.ty(*handle);
                    let h = self
                        .sem
                        .new_block_arg(hty.clone(), func_loc.named(&p.name));
                    arg_types.push(hty);
                    block_args.push(h);
                    let mut new_shape = Vec::with_capacity(shape.len());
                    for (i, s) in shape.iter().enumerate() {
                        let ty = self.sem.ty(*s);
                        let a = self.sem.new_block_arg(
                            ty.clone(),
                            func_loc.named(&format!("{}.shape.{i}", p.name)),
                        );
                        arg_types.push(ty);
                        block_args.push(a);
                        new_shape.push(a);
                    }
                    let mut new_strides = Vec::with_capacity(strides.len());
                    for (i, s) in strides.iter().enumerate() {
                        let ty = self.sem.ty(*s);
                        let a = self.sem.new_block_arg(
                            ty.clone(),
                            func_loc.named(&format!("{}.stride.{i}", p.name)),
                        );
                        arg_types.push(ty);
                        block_args.push(a);
                        new_strides.push(a);
                    }
                    self.scope.insert(
                        &p.name,
                        Val::Desc {
                            handle: h,
                            shape: new_shape,
                            strides: new_strides,
                        },
                    );
                }
                other => {
                    self.scope.insert(&p.name, other.clone());
                }
            }
        }

        self.sem.push_frame();
        let walked = self.visit_body(&callee.body);
        let ret_types = match walked {
            Ok(()) => {
                self.sem.loc = func_loc.clone();
                self.sem.name_prefix = None;
                self.handle_returns(&func_loc)
            }
            Err(e) => Err(e),
        };
        let ret_types = match ret_types {
            Ok(t) => t,
            Err(e) => {
                self.sem.discard_frame();
                self.scope = saved_scope;
                self.sem.loc = saved_loc;
                self.sem.name_prefix = saved_prefix;
                self.return_sites = saved_returns;
                // Wrap with the CALL SITE's position, as Triton does
                // (`code_generator.py:1385`: "Wrap the error in the callee with the location
                // of the call").
                return Err(Error::new(
                    format!("in `{}` called here: {e}", callee.name),
                    pos.line,
                    pos.col,
                ));
            }
        };
        let mut blocks = self.sem.pop_frame();
        if let Some(b) = blocks.first_mut() {
            b.args = block_args;
        }

        self.sem.module.funcs.push(Func {
            name: symbol.to_string(),
            visibility: Visibility::Private,
            noinline: false,
            arg_types,
            ret_types: ret_types.clone(),
            body: crate::ttir::Region { blocks },
            loc: func_loc,
        });
        self.generated.insert(symbol.to_string());
        self.fn_ret_types.insert(symbol.to_string(), ret_types);

        self.scope = saved_scope;
        self.sem.loc = saved_loc;
        self.sem.name_prefix = saved_prefix;
        self.return_sites = saved_returns;
        Ok(())
    }

    /// Emit a `tt.call` to the generated `triton.language.standard.zeros`, generating that
    /// function on first use.
    ///
    /// `standard.py:120`'s body is one line -- `return core.full(shape, 0, dtype)` -- so the
    /// generated function is two constants, a `tt.return`, and the unreachable block
    /// `handle_returns` appends.
    fn call_std_zeros(
        &mut self,
        shape: &[i64],
        dtype: Type,
        pos: Pos,
    ) -> Result<crate::ttir::ValueId> {
        let shape_val = Val::Seq(shape.iter().map(|d| Val::Int(*d as i128)).collect());
        let arg_mangles = vec![
            crate::mangle::mangle_arg(&shape_val, &|_| None).ok_or_else(|| {
                Error::new("tl.zeros shape is not manglable", pos.line, pos.col)
            })?,
            crate::mangle::mangle_arg(&Val::Dtype(dtype.clone()), &|_| None).ok_or_else(|| {
                Error::new("tl.zeros dtype is not manglable", pos.line, pos.col)
            })?,
        ];
        let symbol = crate::mangle::mangle_fn("triton.language.standard.zeros", &arg_mangles);
        let ret_ty = Type::Tensor(shape.to_vec(), std::rc::Rc::new(dtype.clone()));

        if !self.generated.contains(&symbol) {
            self.generate_std_zeros(&symbol, shape, dtype, &ret_ty)?;
        }
        let results = self.sem.call(&symbol, &[], std::slice::from_ref(&ret_ty));
        Ok(results[0])
    }

    /// `tl.max` / `tl.sum`: a `tt.call` to a generated reduction, generating it and its
    /// combiner on first use.
    ///
    /// # The two differ in ONE way that changes the result type.
    ///
    /// `standard.max` UPCASTS anything narrower than 32 bits before reducing
    /// (`standard.py:184`), so an f16 input yields an **f32** result. `standard.sum` does not:
    /// `_pick_sum_dtype` (`standard.py:270`) widens only integers, so an f16 input stays f16.
    /// Measured in the golden:
    ///
    /// ```text
    /// max__fp16S64_64S_c1_cFalse_cTrue_cFalse : (tensor<64x64xf16>) -> tensor<64xf32>
    /// sum__fp16S64_64S_c1_cFalse_cNone        : (tensor<64x64xf16>) -> tensor<64xf16>
    /// ```
    fn call_std_reduction(
        &mut self,
        which: &str,
        input: crate::ttir::ValueId,
        axis: Option<Val>,
        kw: &[(String, Val)],
        pos: Pos,
    ) -> Result<crate::ttir::ValueId> {
        let in_ty = self.sem.ty(input);
        let rank = in_ty.shape().len();
        let axis_i = match axis.as_ref().and_then(|v| v.as_int()) {
            Some(a) if a >= 0 && (a as usize) < rank => a as usize,
            Some(a) if a < 0 && ((rank as i128) + a) >= 0 => ((rank as i128) + a) as usize,
            _ => {
                return Err(Error::new(
                    format!(
                        "tl.{which} needs a compile-time `axis` in range for a rank-{rank} \
                         input; reducing every axis (axis=None) reshapes first \
                         (`semantic.py:1667`) and no fixture does it"
                    ),
                    pos.line,
                    pos.col,
                ))
            }
        };
        let kwv = |k: &str| kw.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        let keep_dims = kwv("keep_dims").unwrap_or(Val::Bool(false));
        if keep_dims.as_int() != Some(0) {
            return Err(Error::new(
                format!(
                    "tl.{which}(..., keep_dims=True) is not lowered: it re-expands the reduced \
                     axis with `tt.expand_dims` afterwards (`core.py:3072`), and no fixture \
                     uses it"
                ),
                pos.line,
                pos.col,
            ));
        }

        // The mangled argument list, in the callee's parameter order.
        let (mangle_args, def_line, combiner, widen_to): (Vec<Val>, u32, &str, Option<Type>) =
            match which {
                // max(input, axis, return_indices=False, return_indices_tie_break_left=True,
                //     keep_dims=False)
                "max" => {
                    let ri = kwv("return_indices").unwrap_or(Val::Bool(false));
                    if ri.as_int() != Some(0) {
                        return Err(Error::new(
                            "tl.max(..., return_indices=True) is not lowered: it uses \
                             `_reduce_with_indices` and a two-value combiner, and no fixture \
                             uses it",
                            pos.line,
                            pos.col,
                        ));
                    }
                    let tb = kwv("return_indices_tie_break_left").unwrap_or(Val::Bool(true));
                    // `standard.py:184`: anything narrower than 32 bits is upcast first.
                    let widen = match in_ty.scalar() {
                        Type::Float(k) if k.bitwidth() < 32 => Some(Type::f32()),
                        Type::Int(b, _) if *b < 32 => Some(Type::i32()),
                        _ => None,
                    };
                    (
                        vec![Val::Int(axis_i as i128), ri, tb, keep_dims.clone()],
                        177,
                        "_elementwise_max",
                        widen,
                    )
                }
                // sum(input, axis, keep_dims=False, dtype=None)
                "sum" => {
                    let dtype = kwv("dtype").unwrap_or(Val::None);
                    if !matches!(dtype, Val::None) {
                        return Err(Error::new(
                            "tl.sum(..., dtype=...) is not lowered; no fixture passes one",
                            pos.line,
                            pos.col,
                        ));
                    }
                    // `_pick_sum_dtype` widens INTEGERS only, so a float input is untouched.
                    let widen = match in_ty.scalar() {
                        Type::Int(b, Signedness::Unsigned) if *b < 32 => {
                            Some(Type::Int(32, Signedness::Unsigned))
                        }
                        Type::Int(b, _) if *b < 32 => Some(Type::i32()),
                        _ => None,
                    };
                    (
                        vec![Val::Int(axis_i as i128), keep_dims.clone(), Val::None],
                        287,
                        "_sum_combine",
                        widen,
                    )
                }
                other => {
                    return Err(Error::new(
                        format!("tl.{other} is not a supported reduction"),
                        pos.line,
                        pos.col,
                    ))
                }
            };

        let mut mangles = vec![crate::mangle::mangle_type(&in_ty)];
        for a in &mangle_args {
            mangles.push(
                crate::mangle::mangle_arg(a, &|_| None).ok_or_else(|| {
                    Error::new(
                        format!("tl.{which} has an argument with no mangled spelling"),
                        pos.line,
                        pos.col,
                    )
                })?,
            );
        }
        let symbol = crate::mangle::mangle_fn(
            &format!("triton.language.standard.{which}"),
            &mangles,
        );

        if !self.generated.contains(&symbol) {
            self.generate_std_reduction(
                &symbol, &in_ty, axis_i, def_line, combiner, widen_to, pos,
            )?;
        }
        let ret_types = self.fn_ret_types.get(&symbol).cloned().unwrap_or_default();
        let results = self.sem.call(&symbol, &[input], &ret_types);
        Ok(results[0])
    }

    /// Build the private `tt.func` for `standard.max` / `standard.sum`.
    #[allow(clippy::too_many_arguments)]
    fn generate_std_reduction(
        &mut self,
        symbol: &str,
        in_ty: &Type,
        axis: usize,
        def_line: u32,
        combiner: &str,
        widen_to: Option<Type>,
        pos: Pos,
    ) -> Result<()> {
        let std_py: std::rc::Rc<str> = std::rc::Rc::from(STANDARD_PY);
        // ONLY THE `def` LINE IS RECORDED for a generated `standard.*` helper, not each inner
        // statement's position. This crate does not parse `standard.py` -- the helpers are
        // reimplemented -- so inner positions would be guesswork. It costs nothing to the
        // oracle diff, which does not compare a location's file, line or column (the goldens
        // embed an absolute path from whichever machine generated them); it only makes a
        // printed module slightly less precise to read.
        let at = |line: u32| Loc::File {
            file: std::rc::Rc::clone(&std_py),
            line,
            col: 1,
        };
        let func_loc = at(def_line);

        let saved_scope = std::mem::take(&mut self.scope);
        let saved_loc = self.sem.loc.clone();
        let saved_prefix = self.sem.name_prefix.take();
        let saved_returns = std::mem::take(&mut self.return_sites);

        self.sem.loc = func_loc.clone();
        self.sem.push_frame();
        let param = self
            .sem
            .new_block_arg(in_ty.clone(), func_loc.named("input"));

        // The upcast, when the helper does one. In `standard.py` it is written
        // `input = input.to(core.float32)` -- an ASSIGNMENT to `input` -- so the resulting
        // `arith.extf` carries the name `input`, which the golden shows and which a bare cast
        // here did not produce.
        let reduced_input = match &widen_to {
            Some(t) => {
                self.sem.name_prefix = Some("input".to_string());
                let r = self.sem.cast(param, t).map_err(|e| Self::located(e, pos));
                self.sem.name_prefix = None;
                r?
            }
            None => param,
        };
        let elem = self.sem.ty(reduced_input).scalar().clone();

        // The combiner, generated first so the region can call it.
        let combiner_symbol = self.ensure_combiner(combiner, &elem, pos)?;

        // The region: two scalar arguments, a call to the combiner, `tt.reduce.return`.
        self.sem.push_frame();
        let a = self.sem.new_block_arg(elem.clone(), Loc::Unknown);
        let b = self.sem.new_block_arg(elem.clone(), Loc::Unknown);
        let combined = self
            .sem
            .call(&combiner_symbol, &[a, b], std::slice::from_ref(&elem));
        self.sem.reduce_return(combined[0], func_loc.clone());
        let mut region_blocks = self.sem.pop_frame();
        if let Some(blk) = region_blocks.first_mut() {
            blk.args = vec![a, b];
        }

        let reduced = self
            .sem
            .reduce_with_region(reduced_input, axis, region_blocks, func_loc.clone())
            .map_err(|e| Self::located(e, pos))?;

        self.return_sites.push(ReturnSite {
            block: self.sem.current_block(),
            value: Val::Ir(reduced),
            loc: func_loc.clone(),
        });
        self.sem.start_block(Vec::new());
        let ret_types = self.handle_returns(&func_loc)?;

        let mut blocks = self.sem.pop_frame();
        if let Some(blk) = blocks.first_mut() {
            blk.args = vec![param];
        }
        self.sem.module.funcs.push(Func {
            name: symbol.to_string(),
            visibility: Visibility::Private,
            noinline: false,
            arg_types: vec![in_ty.clone()],
            ret_types: ret_types.clone(),
            body: crate::ttir::Region { blocks },
            loc: func_loc,
        });
        self.generated.insert(symbol.to_string());
        self.fn_ret_types.insert(symbol.to_string(), ret_types);

        self.scope = saved_scope;
        self.sem.loc = saved_loc;
        self.sem.name_prefix = saved_prefix;
        self.return_sites = saved_returns;
        Ok(())
    }

    /// Generate `standard._elementwise_max` / `standard._sum_combine` for one element type,
    /// returning its mangled symbol.
    ///
    /// Both are one-line `@jit` helpers: `core.maximum(a, b)` (`standard.py:169`) and
    /// `a + b` (`standard.py:262`).
    fn ensure_combiner(&mut self, which: &str, elem: &Type, pos: Pos) -> Result<String> {
        let m = crate::mangle::mangle_type(elem);
        let symbol = crate::mangle::mangle_fn(
            &format!("triton.language.standard.{which}"),
            &[m.clone(), m],
        );
        if self.generated.contains(&symbol) {
            return Ok(symbol);
        }
        let def_line = if which == "_elementwise_max" { 169 } else { 262 };
        let std_py: std::rc::Rc<str> = std::rc::Rc::from(STANDARD_PY);
        let func_loc = Loc::File {
            file: std_py,
            line: def_line,
            col: 1,
        };

        let saved_scope = std::mem::take(&mut self.scope);
        let saved_loc = self.sem.loc.clone();
        let saved_prefix = self.sem.name_prefix.take();
        let saved_returns = std::mem::take(&mut self.return_sites);

        self.sem.loc = func_loc.clone();
        self.sem.push_frame();
        let a = self.sem.new_block_arg(elem.clone(), func_loc.named("a"));
        let b = self.sem.new_block_arg(elem.clone(), func_loc.named("b"));
        let out = match which {
            "_elementwise_max" => self
                .sem
                .max_min(&Val::Ir(a), &Val::Ir(b), true)
                .map_err(|e| Self::located(e, pos))?,
            _ => self
                .sem
                .add(&Val::Ir(a), &Val::Ir(b), true)
                .map_err(|e| Self::located(e, pos))?,
        };
        self.return_sites.push(ReturnSite {
            block: self.sem.current_block(),
            value: Val::Ir(out),
            loc: func_loc.clone(),
        });
        self.sem.start_block(Vec::new());
        let ret_types = self.handle_returns(&func_loc)?;
        let mut blocks = self.sem.pop_frame();
        if let Some(blk) = blocks.first_mut() {
            blk.args = vec![a, b];
        }
        self.sem.module.funcs.push(Func {
            name: symbol.clone(),
            visibility: Visibility::Private,
            noinline: false,
            arg_types: vec![elem.clone(), elem.clone()],
            ret_types: ret_types.clone(),
            body: crate::ttir::Region { blocks },
            loc: func_loc,
        });
        self.generated.insert(symbol.clone());
        self.fn_ret_types.insert(symbol.clone(), ret_types);

        self.scope = saved_scope;
        self.sem.loc = saved_loc;
        self.sem.name_prefix = saved_prefix;
        self.return_sites = saved_returns;
        Ok(symbol)
    }

    /// Build the private `tt.func` for `standard.zeros`.
    fn generate_std_zeros(
        &mut self,
        symbol: &str,
        shape: &[i64],
        dtype: Type,
        ret_ty: &Type,
    ) -> Result<()> {
        // Locations point into `standard.py`, exactly as Triton's do. Line and column are
        // not compared by the structural diff (the goldens carry an absolute path from
        // whichever machine generated them), but recording the real ones keeps a printed
        // module readable and matches the oracle if anyone ever does compare them:
        //   120:1   `def zeros(shape, dtype):`      -- the function, and its dead block
        //   129:12  `core.full(shape, 0, dtype)`    -- the returned expression
        //   129:5   `return ...`                    -- the return statement
        let std_py = std::rc::Rc::from(STANDARD_PY);
        let at = |line: u32, col: u32| Loc::File {
            file: std::rc::Rc::clone(&std_py),
            line,
            col,
        };

        // A generated callee is a fresh function: no pending assignment name, and its own
        // location. Save and restore the caller's, since generation happens mid-expression.
        let saved_loc = self.sem.loc.clone();
        let saved_prefix = self.sem.name_prefix.take();

        self.sem.push_frame();
        self.sem.loc = at(129, 12);
        let body_val = self.full(shape, &Val::Int(0), dtype, Pos { line: 129, col: 11 })?;

        // `visit_Return` records the return and then starts a DEAD block, into which
        // `handle_returns` puts a poison per result and a second terminator.
        let ret_block = self.sem.current_block();
        self.sem.loc = at(129, 5);
        let mut ret = crate::ttir::Op::new("tt.return", self.sem.loc.clone());
        ret.operands = vec![body_val];
        self.sem.emit_in_block(ret_block, ret);

        self.sem.start_block(Vec::new());
        self.sem.loc = at(120, 1);
        let poison = self.sem.poison(ret_ty.clone());
        let mut dead_ret = crate::ttir::Op::new("tt.return", self.sem.loc.clone());
        dead_ret.operands = vec![poison];
        self.sem.emit_raw(dead_ret);

        let blocks = self.sem.pop_frame();
        self.sem.module.funcs.push(Func {
            name: symbol.to_string(),
            visibility: Visibility::Private,
            noinline: false,
            arg_types: Vec::new(),
            ret_types: vec![ret_ty.clone()],
            body: crate::ttir::Region { blocks },
            loc: at(120, 1),
        });
        self.generated.insert(symbol.to_string());
        self.fn_ret_types
            .insert(symbol.to_string(), vec![ret_ty.clone()]);

        self.sem.loc = saved_loc;
        self.sem.name_prefix = saved_prefix;
        Ok(())
    }

    /// `make_scalar` (`semantic.py:615`): cast an existing value, or build a constant.
    fn make_scalar(&mut self, v: &Val, ty: Type, pos: Pos) -> Result<crate::ttir::ValueId> {
        match v {
            Val::Ir(id) => self.sem.cast(*id, &ty).map_err(|e| Self::located(e, pos)),
            other => self
                .sem
                .scalar_constant(other, ty)
                .map_err(|e| Self::located(e, pos)),
        }
    }

    fn call_builtin(&mut self, name: &str, args: &[Val], pos: Pos) -> Result<Val> {
        match name {
            // Python's `float()` PARSES A STRING, and the fixtures rely on it:
            // `attention_flash.py` seeds its running maximum with
            // `tl.zeros([BLOCK_M], dtype=tl.float16) - float("inf")`. Accepting only numbers
            // here refused a kernel the oracle compiles.
            "float" => match args.first() {
                Some(Val::Str(s)) => {
                    let t = s.trim();
                    let v = match t.to_ascii_lowercase().as_str() {
                        "inf" | "+inf" | "infinity" | "+infinity" => f64::INFINITY,
                        "-inf" | "-infinity" => f64::NEG_INFINITY,
                        "nan" | "+nan" | "-nan" => f64::NAN,
                        _ => t.parse::<f64>().map_err(|_| {
                            Error::new(
                                format!("could not convert string to float: {s:?}"),
                                pos.line,
                                pos.col,
                            )
                        })?,
                    };
                    Ok(Val::Float(v))
                }
                Some(v) => v.as_f64().map(Val::Float).ok_or_else(|| {
                    Error::new(
                        format!("float() needs a number or a string, got a {}", v.kind_name()),
                        pos.line,
                        pos.col,
                    )
                }),
                None => Err(Error::new("float() needs an argument", pos.line, pos.col)),
            },
            other => Err(Error::new(
                format!("builtin `{other}` is not supported in a kernel"),
                pos.line,
                pos.col,
            )),
        }
    }

    fn call_tl(
        &mut self,
        name: &str,
        args: &[Val],
        kw: &[(String, Val)],
        pos: Pos,
    ) -> Result<Val> {
        let get = |i: usize, key: &str| -> Option<Val> {
            kw.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .or_else(|| args.get(i).cloned())
        };
        match name {
            "program_id" => {
                let axis = get(0, "axis")
                    .and_then(|v| v.as_int())
                    .ok_or_else(|| Error::new("tl.program_id needs an axis", pos.line, pos.col))?;
                self.sem
                    .program_id(axis)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            "make_tensor_descriptor" => self.make_tensor_descriptor(args, kw, pos),
            // `tl.multiple_of(x, n)` emits NO OP: it stamps `tt.divisibility` on the op that
            // defined `x` and returns `x` unchanged. See `Semantic::set_divisibility`.
            "multiple_of" => {
                let x = get(0, "input").ok_or_else(|| {
                    Error::new("tl.multiple_of needs a value", pos.line, pos.col)
                })?;
                let values: Vec<i128> = match get(1, "values") {
                    Some(Val::Seq(items)) => items.iter().filter_map(|v| v.as_int()).collect(),
                    Some(v) => match v.as_int() {
                        Some(n) => vec![n],
                        None => {
                            return Err(Error::new(
                                "tl.multiple_of's divisor must be a compile-time integer",
                                pos.line,
                                pos.col,
                            ))
                        }
                    },
                    None => {
                        return Err(Error::new(
                            "tl.multiple_of needs a divisor",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                let rank = match &x {
                    Val::Ir(id) => self.sem.ty(*id).shape().len().max(1),
                    _ => 1,
                };
                if rank != values.len() {
                    return Err(Error::new(
                        format!(
                            "Shape of input to multiple_of does not match the length of values \
                             ({rank} against {})",
                            values.len()
                        ),
                        pos.line,
                        pos.col,
                    ));
                }
                if let Val::Ir(id) = &x {
                    self.sem.set_divisibility(*id, &values);
                }
                Ok(x)
            }
            "static_assert" => {
                let cond = get(0, "cond");
                let truthy = match &cond {
                    Some(Val::Bool(b)) => *b,
                    Some(Val::Int(i)) => *i != 0,
                    other => {
                        return Err(Error::new(
                            format!(
                                "tl.static_assert needs a compile-time condition, got a {}",
                                other.as_ref().map(|v| v.kind_name()).unwrap_or("nothing")
                            ),
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                if !truthy {
                    let msg = match get(1, "msg") {
                        Some(Val::Str(s)) => s,
                        _ => "static assertion failed".to_string(),
                    };
                    return Err(Error::new(
                        format!("tl.static_assert failed: {msg}"),
                        pos.line,
                        pos.col,
                    ));
                }
                Ok(Val::None)
            }
            "fdiv" => {
                let a = get(0, "x").ok_or_else(|| {
                    Error::new("tl.fdiv needs two arguments", pos.line, pos.col)
                })?;
                let b = get(1, "y").ok_or_else(|| {
                    Error::new("tl.fdiv needs two arguments", pos.line, pos.col)
                })?;
                self.sem
                    .fdiv(&a, &b)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            "maximum" => {
                let a = get(0, "x").ok_or_else(|| {
                    Error::new("tl.maximum needs two arguments", pos.line, pos.col)
                })?;
                let b = get(1, "y").ok_or_else(|| {
                    Error::new("tl.maximum needs two arguments", pos.line, pos.col)
                })?;
                self.sem
                    .max_min(&a, &b, true)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            "arange" => {
                let a = get(0, "start")
                    .and_then(|v| v.as_int())
                    .ok_or_else(|| Error::new("tl.arange needs a start", pos.line, pos.col))?;
                let b = get(1, "end")
                    .and_then(|v| v.as_int())
                    .ok_or_else(|| Error::new("tl.arange needs an end", pos.line, pos.col))?;
                self.sem
                    .arange(a, b)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            // `tl.dot(input, other, acc=None, ..., out_dtype=tl.float32)`. `acc` is optional:
            // `attention_flash.py`'s first dot passes none and gives `out_dtype=tl.float16`,
            // and Triton then splats its zero into an accumulator.
            "dot" => {
                let a = self.ir_arg(&get(0, "input"), "tl.dot", pos)?;
                let b = self.ir_arg(&get(1, "other"), "tl.dot", pos)?;
                let acc = match get(2, "acc") {
                    None | Some(Val::None) => None,
                    Some(v) => Some(
                        self.sem.to_tensor(&v).map_err(|e| Self::located(e, pos))?,
                    ),
                };
                let out_dtype = match get(6, "out_dtype") {
                    Some(Val::Dtype(t)) => Some(t),
                    None | Some(Val::None) => None,
                    Some(other) => {
                        return Err(Error::new(
                            format!(
                                "tl.dot's out_dtype must be a dtype, got a {}",
                                other.kind_name()
                            ),
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.sem
                    .dot(a, b, acc, out_dtype)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            // `tl.rsqrt` and `tl.math.rsqrt` are the SAME function: `math.py`'s `rsqrt` is
            // decorated `@core._tensor_member_fn` and re-exported into `tl` by
            // `language/__init__.py`, so both spellings reach `create_rsqrt`. Measured in
            // `rmsnorm.ttir_raw.mlir`, where `tl.rsqrt` emits `math.rsqrt`.
            "exp" | "math.exp2" | "rsqrt" | "math.rsqrt" => {
                let v = self.ir_arg(&get(0, "x"), name, pos)?;
                let op = match name {
                    "exp" => "math.exp",
                    "math.exp2" => "math.exp2",
                    _ => "math.rsqrt",
                };
                self.sem
                    .math_unary(op, v)
                    .map(Val::Ir)
                    .map_err(|e| Self::located(e, pos))
            }
            // `tl.full` is a BUILTIN (`core.py:2010`, `@builtin` taking `_semantic`), so it
            // lowers INLINE -- unlike `tl.zeros`, which is `@jit` and becomes a call.
            // Measured: the golden's `one` is `arith.constant 1.0 : f16` followed by
            // `arith.constant dense<1.0> : tensor<64x64xf16>`, both carrying the kernel
            // variable's name, with no `tt.call` anywhere near them.
            "full" => {
                let shape = self.const_shape(&get(0, "shape"), "tl.full", pos)?;
                let value = get(1, "value").ok_or_else(|| {
                    Error::new("tl.full needs a fill value", pos.line, pos.col)
                })?;
                let dtype = match get(2, "dtype") {
                    Some(Val::Dtype(t)) => t,
                    _ => {
                        return Err(Error::new(
                            "tl.full needs a dtype, e.g. `tl.float16`",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.full(&shape, &value, dtype, pos).map(Val::Ir)
            }
            // `tl.zeros` IS `@jit` (`standard.py:120`), so Triton generates a private
            // `tt.func` for it and emits a `tt.call`.
            "zeros" => {
                let shape = self.const_shape(&get(0, "shape"), "tl.zeros", pos)?;
                let dtype = match get(1, "dtype") {
                    Some(Val::Dtype(t)) => t,
                    _ => {
                        return Err(Error::new(
                            "tl.zeros needs a dtype, e.g. `tl.float16`",
                            pos.line,
                            pos.col,
                        ))
                    }
                };
                self.call_std_zeros(&shape, dtype, pos).map(Val::Ir)
            }
            // `tl.max` / `tl.sum` are `@jit` helpers in `standard.py`, so each becomes a call
            // to a generated function whose body is a `tt.reduce` with a combiner region that
            // itself calls another generated helper.
            "max" | "sum" => {
                let input = self.ir_arg(&get(0, "input"), &format!("tl.{name}"), pos)?;
                let axis = get(1, "axis");
                self.call_std_reduction(name, input, axis, kw, pos).map(Val::Ir)
            }
            "range" => Err(Error::new(
                "`tl.range(...)` is only supported as the iterable of a `for` statement, \
                 which is the only way the fixtures use it; as a value it has no lowering."
                    .to_string(),
                pos.line,
                pos.col,
            )),
            other => Err(Error::new(
                format!("`tl.{other}` is not in the supported surface"),
                pos.line,
                pos.col,
            )),
        }
    }

    fn ir_arg(&mut self, v: &Option<Val>, who: &str, pos: Pos) -> Result<crate::ttir::ValueId> {
        match v {
            Some(x) => self.sem.to_tensor(x).map_err(|e| Self::located(e, pos)),
            None => Err(Error::new(
                format!("{who} is missing an argument"),
                pos.line,
                pos.col,
            )),
        }
    }

    /// `tl.make_tensor_descriptor` (`semantic.py:1851`).
    ///
    /// Shape entries become i32 and stride entries i64 (`make_scalar(x, tl.int32)` /
    /// `make_scalar(..., tl.int64)`), which is why the goldens show
    /// `[%n], [%c1_i64]` -- and why the `1` in `strides=[1]` becomes an
    /// `arith.constant 1 : i64` carrying the ASSIGNMENT's name and location rather than
    /// the literal's own.
    fn make_tensor_descriptor(
        &mut self,
        args: &[Val],
        kw: &[(String, Val)],
        pos: Pos,
    ) -> Result<Val> {
        let get = |i: usize, key: &str| -> Option<Val> {
            kw.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .or_else(|| args.get(i).cloned())
        };
        let base = match get(0, "base") {
            Some(Val::Ir(id)) => id,
            _ => {
                return Err(Error::new(
                    "tl.make_tensor_descriptor needs a pointer as its first argument",
                    pos.line,
                    pos.col,
                ))
            }
        };
        let seq = |v: Option<Val>, what: &str| -> Result<Vec<Val>> {
            match v {
                Some(Val::Seq(items)) => Ok(items),
                Some(other) => Err(Error::new(
                    format!(
                        "tl.make_tensor_descriptor's `{what}` must be a list, got a {}",
                        other.kind_name()
                    ),
                    pos.line,
                    pos.col,
                )),
                None => Err(Error::new(
                    format!("tl.make_tensor_descriptor is missing `{what}`"),
                    pos.line,
                    pos.col,
                )),
            }
        };
        let shape = seq(get(1, "shape"), "shape")?;
        let strides = seq(get(2, "strides"), "strides")?;
        let block = seq(get(3, "block_shape"), "block_shape")?;

        let ndim = shape.len();
        if !(1..=5).contains(&ndim) {
            return Err(Error::new(
                format!("tl.make_tensor_descriptor expects 1 <= ndim <= 5 but got {ndim}"),
                pos.line,
                pos.col,
            ));
        }
        if strides.len() != ndim {
            return Err(Error::new(
                format!(
                    "tl.make_tensor_descriptor expects {ndim} strides but got {}",
                    strides.len()
                ),
                pos.line,
                pos.col,
            ));
        }
        if block.len() != ndim {
            return Err(Error::new(
                format!(
                    "tl.make_tensor_descriptor expects block_shape to have {ndim} \
                     dimensions but got {}",
                    block.len()
                ),
                pos.line,
                pos.col,
            ));
        }
        // The last stride must be 1 (`semantic.py:1868`).
        match strides.last().and_then(|v| v.as_int()) {
            Some(1) => {}
            Some(other) => {
                return Err(Error::new(
                    format!("Tensor descriptor last dim must be 1 but got {other}"),
                    pos.line,
                    pos.col,
                ))
            }
            None => {
                return Err(Error::new(
                    "Tensor descriptor last stride must be the compile-time constant 1",
                    pos.line,
                    pos.col,
                ))
            }
        }
        // The last block dimension must be at least 16 bytes (`semantic.py:1863`).
        let mut block_shape = Vec::with_capacity(ndim);
        for b in &block {
            match b.as_int() {
                Some(v) => block_shape.push(v as i64),
                None => {
                    return Err(Error::new(
                        "tl.make_tensor_descriptor's block_shape must be compile-time \
                         constants",
                        pos.line,
                        pos.col,
                    ))
                }
            }
        }
        let base_ty = self.sem.ty(base);
        let elem_bits = match base_ty.scalar() {
            Type::Ptr(e, _) => match &**e {
                Type::Float(k) => k.bitwidth(),
                Type::Int(b, _) => *b,
                other => {
                    return Err(Error::new(
                        format!("a descriptor over {other} is not supported"),
                        pos.line,
                        pos.col,
                    ))
                }
            },
            other => {
                return Err(Error::new(
                    format!(
                        "tl.make_tensor_descriptor needs a pointer, got {other}"
                    ),
                    pos.line,
                    pos.col,
                ))
            }
        };
        let elem_size = (elem_bits / 8).max(1) as i64;
        let contig = *block_shape.last().unwrap();
        if contig * elem_size < 16 {
            return Err(Error::new(
                format!(
                    "Descriptor block shape must have at least 16 bytes in the last \
                     dimension, but got {contig} * {elem_size} = {} bytes",
                    contig * elem_size
                ),
                pos.line,
                pos.col,
            ));
        }

        let mut shape_ids = Vec::with_capacity(ndim);
        for s in &shape {
            shape_ids.push(self.make_scalar(s, Type::i32(), pos)?);
        }
        let mut stride_ids = Vec::with_capacity(ndim);
        for s in &strides {
            stride_ids.push(self.make_scalar(s, Type::i64(), pos)?);
        }
        self.sem
            .make_tensor_descriptor(base, &shape_ids, &stride_ids, &block_shape)
            .map_err(|e| Self::located(e, pos))
    }
}

/// Compile-time arithmetic on two literals.
fn fold_binary(op: BinOpKind, l: &Val, r: &Val) -> Option<Val> {
    // Integer-only operators.
    if let (Some(a), Some(b)) = (l.as_int(), r.as_int()) {
        if matches!(l, Val::Int(_) | Val::Bool(_)) && matches!(r, Val::Int(_) | Val::Bool(_)) {
            return match op {
                BinOpKind::Add => Some(Val::Int(a + b)),
                BinOpKind::Sub => Some(Val::Int(a - b)),
                BinOpKind::Mult => Some(Val::Int(a * b)),
                BinOpKind::FloorDiv if b != 0 => Some(Val::Int(a.div_euclid(b))),
                BinOpKind::Mod if b != 0 => Some(Val::Int(a.rem_euclid(b))),
                BinOpKind::BitAnd => Some(Val::Int(a & b)),
                BinOpKind::BitOr => Some(Val::Int(a | b)),
                BinOpKind::BitXor => Some(Val::Int(a ^ b)),
                BinOpKind::LShift => Some(Val::Int(a << b)),
                BinOpKind::RShift => Some(Val::Int(a >> b)),
                // Python's `/` on two ints is a float.
                BinOpKind::Div if b != 0 => Some(Val::Float(a as f64 / b as f64)),
                BinOpKind::Pow if (0..64).contains(&b) => Some(Val::Int(a.pow(b as u32))),
                _ => None,
            };
        }
    }
    let (a, b) = (l.as_f64()?, r.as_f64()?);
    match op {
        BinOpKind::Add => Some(Val::Float(a + b)),
        BinOpKind::Sub => Some(Val::Float(a - b)),
        BinOpKind::Mult => Some(Val::Float(a * b)),
        BinOpKind::Div => Some(Val::Float(a / b)),
        BinOpKind::FloorDiv => Some(Val::Float((a / b).floor())),
        BinOpKind::Mod => Some(Val::Float(a.rem_euclid(b))),
        BinOpKind::Pow => Some(Val::Float(a.powf(b))),
        _ => None,
    }
}

fn fold_compare(op: CmpKind, l: &Val, r: &Val) -> Option<Val> {
    if let (Val::Str(a), Val::Str(b)) = (l, r) {
        return Some(Val::Bool(match op {
            CmpKind::Eq => a == b,
            CmpKind::NotEq => a != b,
            _ => return None,
        }));
    }
    if matches!(l, Val::None) || matches!(r, Val::None) {
        let both_none = matches!(l, Val::None) && matches!(r, Val::None);
        return Some(Val::Bool(match op {
            CmpKind::Eq => both_none,
            CmpKind::NotEq => !both_none,
            _ => return None,
        }));
    }
    let (a, b) = (l.as_f64()?, r.as_f64()?);
    Some(Val::Bool(match op {
        CmpKind::Eq => a == b,
        CmpKind::NotEq => a != b,
        CmpKind::Lt => a < b,
        CmpKind::LtE => a <= b,
        CmpKind::Gt => a > b,
        CmpKind::GtE => a >= b,
    }))
}

/// Whether two non-IR values are the same compile-time constant.
fn same_const(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Float(x), Val::Float(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::None, Val::None) => true,
        (Val::Dtype(x), Val::Dtype(y)) => x == y,
        (Val::Seq(x), Val::Seq(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_const(p, q))
        }
        _ => false,
    }
}
