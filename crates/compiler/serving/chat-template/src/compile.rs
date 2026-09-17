// SPDX-License-Identifier: Apache-2.0
//! Build-time half: parse a Jinja chat template and emit Rust source.
//!
//! ⛔ THE EMITTER IS MECHANICAL, AND THAT IS THE WHOLE POINT. It walks
//! minijinja's own AST and emits one Rust fragment per node. It never "reads"
//! the template and never reasons about what the template is trying to do.
//!
//! That is not stylistic caution. Translating TinyLlama's template by reading it
//! and writing the equivalent Rust by hand produced 0/6 byte-identical results —
//! four `EmitRaw` newline nodes were silently dropped, because they sit inside
//! `if` branches where a human does not expect them. Walking the AST produced
//! 8/8. Any shortcut that re-introduces human judgement re-introduces that bug.
//!
//! ## Why minijinja's parser and not our own
//!
//! Using the interpreter's own parser means our output and the interpreter's can
//! only diverge in *our codegen* — which the golden tests catch directly. A
//! hand-written parser would add a second, independent place for divergence to
//! originate, and divergence here is silent.
//!
//! ## Unsupported constructs are build errors
//!
//! There is no fallback inside the emitter. A construct it cannot lower is a
//! [`CompileError`] carrying file, line and column, so the build fails naming
//! exactly what to implement next. That is what makes adding a model a bounded
//! task rather than an open question.

use minijinja::machinery::ast::{self, Expr, Stmt};
use minijinja::machinery::WhitespaceConfig;
use minijinja::syntax::SyntaxConfig;

/// The whitespace configuration the serving runtime uses.
///
/// ⛔ MUST MATCH `crates/serving/api/src/chat_template.rs::build_env`, which sets
/// `trim_blocks` and `lstrip_blocks`. This is not a formatting preference: it
/// changes the AST. The same template parsed with defaults yields a leading
/// `"\n<|user|>\n"` where the runtime yields `"<|user|>\n"`, and that stray
/// newline is the documented cause of TinyLlama's `<` moving from token 529 to
/// 29966 and generating nonsense.
pub const RUNTIME_WHITESPACE: WhitespaceConfig = WhitespaceConfig {
    trim_blocks: true,
    lstrip_blocks: true,
    keep_trailing_newline: false,
};

/// A construct the emitter cannot lower yet, located in the source template.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
    pub template: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.template, self.line, self.col, self.message
        )
    }
}

impl std::error::Error for CompileError {}

/// Compile `source` into the body of a Rust function.
///
/// Returns Rust source declaring `pub fn <fn_name>(__ctx: &serde_json::Value)
/// -> Result<String, TemplateError>`. The caller decides what to do with it —
/// this crate deliberately does not know about registries or `inventory`.
pub fn compile(source: &str, template_name: &str, fn_name: &str) -> Result<String, CompileError> {
    let ast = minijinja::machinery::parse(
        source,
        template_name,
        SyntaxConfig,
        RUNTIME_WHITESPACE,
    )
    .map_err(|e| CompileError {
        message: format!("template failed to parse: {e}"),
        template: template_name.to_string(),
        line: e.line().unwrap_or(0),
        col: 0,
    })?;

    let mut em = Emitter {
        out: String::new(),
        scopes: Vec::new(),
        loop_depth: 0,
        seq: 0,
        template: template_name.to_string(),
    };
    em.line(0, "#[allow(unused_mut, unused_variables, clippy::all)]");
    em.line(
        0,
        &format!(
            "pub fn {fn_name}(__ctx: &::serde_json::Value) -> ::core::result::Result<::std::string::String, TemplateError> {{"
        ),
    );
    // Capacity: the literal budget is known here, so the allocation is sized
    // once instead of growing. Cheap, and it is information only the compiler has.
    let budget = source.len().next_power_of_two().max(64);
    em.line(1, &format!("let mut __out = ::std::string::String::with_capacity({budget});"));
    em.line(1, "let __root = Val::Ref(__ctx);");
    em.stmt(&ast, 1)?;
    em.line(1, "Ok(__out)");
    em.line(0, "}");

    // The drift gate: the exact bytes this renderer was generated from, plus a
    // descriptor the caller can register. Emitted here rather than assembled by
    // the caller so the source and the renderer cannot be paired up wrongly.
    em.line(0, "");
    em.line(
        0,
        &format!("pub const TEMPLATE_NAME: &str = {};", rust_str(template_name)),
    );
    em.line(
        0,
        &format!("pub const TEMPLATE_SOURCE: &str = {};", rust_str(source)),
    );
    // ⛔ GENERATED CODE MUST BE LINT-CLEAN. Consumers build with
    // `clippy -- -D warnings`, so a style lint fired by emitted code fails THEIR
    // build for something they cannot edit. Two defences: an explicit allow on
    // every emitted item, and not emitting the lint-triggering shape in the first
    // place (here, `render: render` -> the field shorthand). The allow alone is
    // not enough — future lints arrive without warning.
    em.line(0, "#[allow(clippy::all)]");
    em.line(0, "pub static COMPILED: CompiledTemplate = CompiledTemplate {");
    em.line(1, "name: TEMPLATE_NAME,");
    em.line(1, "source: TEMPLATE_SOURCE,");
    if fn_name == "render" {
        em.line(1, "render,");
    } else {
        em.line(1, &format!("render: {fn_name},"));
    }
    em.line(0, "};");
    Ok(em.out)
}

struct Emitter {
    out: String,
    /// Jinja name -> Rust expression yielding a `Val`. Innermost scope last.
    scopes: Vec<Vec<(String, String)>>,
    loop_depth: usize,
    seq: usize,
    template: String,
}

impl Emitter {
    fn line(&mut self, indent: usize, s: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn fresh(&mut self) -> usize {
        self.seq += 1;
        self.seq
    }

    fn lookup(&self, name: &str) -> Option<&str> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|s| s.iter().rev())
            .find(|(n, _)| n == name)
            .map(|(_, e)| e.as_str())
    }

    fn err<T>(&self, msg: impl Into<String>, span: minijinja::machinery::Span) -> Result<T, CompileError> {
        Err(CompileError {
            message: msg.into(),
            template: self.template.clone(),
            line: span.start_line as usize,
            col: span.start_col as usize,
        })
    }

    // -- statements ------------------------------------------------------
    fn stmt(&mut self, s: &Stmt<'_>, ind: usize) -> Result<(), CompileError> {
        match s {
            Stmt::Template(t) => {
                for c in &t.children {
                    self.stmt(c, ind)?;
                }
                Ok(())
            }
            // A literal span. Whitespace control is already applied by the lexer,
            // so this is emitted verbatim — no trimming, no normalisation.
            Stmt::EmitRaw(r) => {
                let lit = rust_str(r.raw);
                self.line(ind, &format!("__out.push_str({lit});"));
                Ok(())
            }
            Stmt::EmitExpr(e) => {
                // A `{{ "literal" }}` is indistinguishable from raw output once
                // the value is known to be a string, so emit the push directly
                // rather than routing a constant through the value layer.
                // ⛔ STRINGS ONLY. For a non-string constant the value layer's
                // rules differ from `push_str` — `true` renders `True`, `null`
                // renders `None` — so those must keep going through it.
                if let Expr::Const(c) = &e.expr
                    && let Some(lit) = c.value.as_str()
                {
                    self.line(ind, &format!("__out.push_str({});", rust_str(lit)));
                    return Ok(());
                }
                let v = self.expr(&e.expr)?;
                self.line(ind, &format!("render_into(&mut __out, &({v}));"));
                Ok(())
            }
            Stmt::IfCond(c) => {
                let cond = self.expr(&c.expr)?;
                self.line(ind, &format!("if truthy(&({cond})) {{"));
                self.scopes.push(Vec::new());
                for st in &c.true_body {
                    self.stmt(st, ind + 1)?;
                }
                self.scopes.pop();
                if !c.false_body.is_empty() {
                    self.line(ind, "} else {");
                    self.scopes.push(Vec::new());
                    for st in &c.false_body {
                        self.stmt(st, ind + 1)?;
                    }
                    self.scopes.pop();
                }
                self.line(ind, "}");
                Ok(())
            }
            Stmt::ForLoop(f) => self.for_loop(f, ind),
            // Everything else, including `set`, `with`, macros and the
            // multi-template statements. Reported with its location so the build
            // error names the next thing to implement.
            other => self.err(
                format!("unsupported statement: {}", stmt_name(other)),
                stmt_span(s),
            ),
        }
    }

    /// `{% for target in iter %}`.
    ///
    /// The sequence is materialised before the loop because `loop.last` needs
    /// the length up front. `to_seq` is fallible — iterating a number or a bool
    /// is an error in the interpreter, so it is an error here too.
    fn for_loop(&mut self, f: &ast::ForLoop<'_>, ind: usize) -> Result<(), CompileError> {
        if f.filter_expr.is_some() {
            return self.err("`for ... if ...` loop filters are not supported", f.iter.span());
        }
        if f.recursive {
            return self.err("recursive loops are not supported", f.iter.span());
        }
        let Expr::Var(target) = &f.target else {
            return self.err("only a single loop variable is supported", f.target.span());
        };

        let iter = self.expr(&f.iter)?;
        let d = self.loop_depth;
        let id = self.fresh();
        self.line(ind, &format!("let __seq{d} = to_seq(&({iter}))?; // #{id}"));
        self.line(ind, &format!("let __n{d} = __seq{d}.len();"));
        self.line(
            ind,
            &format!("for (__i{d}, __item{d}) in __seq{d}.iter().enumerate() {{"),
        );

        self.scopes
            .push(vec![(target.id.to_string(), format!("(*__item{d})"))]);
        self.loop_depth += 1;
        for st in &f.body {
            self.stmt(st, ind + 1)?;
        }
        self.loop_depth -= 1;
        self.scopes.pop();
        self.line(ind, "}");

        // `{% else %}` on a for loop runs when the sequence was empty.
        if !f.else_body.is_empty() {
            self.line(ind, &format!("if __n{d} == 0 {{"));
            self.scopes.push(Vec::new());
            for st in &f.else_body {
                self.stmt(st, ind + 1)?;
            }
            self.scopes.pop();
            self.line(ind, "}");
        }
        Ok(())
    }

    // -- expressions -----------------------------------------------------
    fn expr(&mut self, e: &Expr<'_>) -> Result<String, CompileError> {
        Ok(match e {
            Expr::Var(v) => match self.lookup(v.id) {
                Some(local) => format!("{local}.clone()"),
                None => format!("get_key(&__root, {})", rust_str(v.id)),
            },
            Expr::Const(c) => const_to_rust(&c.value),
            Expr::GetItem(g) => {
                let base = self.expr(&g.expr)?;
                // ⛔ SPECIALISE ON WHAT THE TEMPLATE ALREADY TOLD US. `message['role']`
                // and `messages[0]` have constant subscripts, so the key-kind
                // dispatch inside `get_item` is work the template already answered.
                // Emit the direct accessor and keep `get_item` for the genuinely
                // dynamic case (`messages[i]`).
                if let Expr::Const(c) = &g.subscript_expr {
                    if let Some(k) = c.value.as_str() {
                        return Ok(format!("get_key(&({base}), {})", rust_str(k)));
                    }
                    if let Ok(::serde_json::Value::Number(n)) = ::serde_json::to_value(&c.value)
                        && let Some(i) = n.as_i64()
                    {
                        return Ok(format!("get_index(&({base}), {i}i64)"));
                    }
                }
                let key = self.expr(&g.subscript_expr)?;
                format!("get_item(&({base}), &({key}))")
            }
            Expr::GetAttr(g) => {
                // `loop.first` / `loop.last` / `loop.index` / `loop.index0`
                if let Expr::Var(v) = &g.expr
                    && v.id == "loop"
                    && self.lookup("loop").is_none()
                {
                    return self.loop_attr(g.name, e.span());
                }
                let base = self.expr(&g.expr)?;
                format!("get_key(&({base}), {})", rust_str(g.name))
            }
            Expr::BinOp(b) => {
                let l = self.expr(&b.left)?;
                let r = self.expr(&b.right)?;
                let boolean = |c: String| {
                    format!("Val::Owned(::serde_json::Value::Bool({c}))")
                };
                match b.op {
                    ast::BinOpKind::Add => format!("add(&({l}), &({r}))?"),
                    // ⛔ COMPARISONS ARRIVE HERE, NOT IN `Compare`. `Compare` is
                    // only for CHAINED comparisons (`a < b < c`); a plain `a != b`
                    // is a `BinOp`. Getting this backwards cost a build cycle.
                    ast::BinOpKind::Eq => boolean(format!("eq(&({l}), &({r}))")),
                    ast::BinOpKind::Ne => boolean(format!("ne(&({l}), &({r}))")),
                    ast::BinOpKind::In => boolean(format!("contains(&({l}), &({r}))")),
                    // ⛔ minijinja's `and` returns `False` (a bool) when the left
                    // side is falsy — NOT the left operand as Python does.
                    // Verified: `{{ '' and 'B' }}` renders "False", not "".
                    ast::BinOpKind::ScAnd => format!(
                        "{{ let __l = {l}; if truthy(&__l) {{ {r} }} else {{ Val::Owned(::serde_json::Value::Bool(false)) }} }}"
                    ),
                    // `or` does return the operand on both sides.
                    ast::BinOpKind::ScOr => format!(
                        "{{ let __l = {l}; if truthy(&__l) {{ __l }} else {{ {r} }} }}"
                    ),
                    // ⛔ ORDERING COMPARISONS DELIBERATELY NOT IMPLEMENTED. Jinja's
                    // `<` / `>` coerce across kinds, and the coercion rules have
                    // not been verified against the interpreter yet. A build error
                    // naming the operator is correct; guessing the semantics would
                    // produce a subtly wrong prompt that tests would not catch.
                    ref other => {
                        return self.err(format!("unsupported operator: {other:?}"), e.span())
                    }
                }
            }
            Expr::UnaryOp(u) => {
                let v = self.expr(&u.expr)?;
                match u.op {
                    ast::UnaryOpKind::Not => format!(
                        "Val::Owned(::serde_json::Value::Bool(!truthy(&({v}))))"
                    ),
                    ref other => {
                        return self.err(format!("unsupported unary operator: {other:?}"), e.span())
                    }
                }
            }
            Expr::Compare(c) => {
                let l = self.expr(&c.expr)?;
                if c.ops.len() != 1 {
                    return self.err("chained comparisons are not supported", e.span());
                }
                let cmp = &c.ops[0];
                let r = self.expr(&cmp.expr)?;
                let call = match cmp.op {
                    ast::CompareOpKind::Eq => format!("eq(&({l}), &({r}))"),
                    ast::CompareOpKind::Ne => format!("ne(&({l}), &({r}))"),
                    ast::CompareOpKind::In => format!("contains(&({l}), &({r}))"),
                    ast::CompareOpKind::NotIn => format!("!contains(&({l}), &({r}))"),
                    ref other => {
                        return self.err(format!("unsupported comparison: {other:?}"), e.span())
                    }
                };
                format!("Val::Owned(::serde_json::Value::Bool({call}))")
            }
            other => {
                return self.err(
                    format!("unsupported expression: {}", other.description()),
                    e.span(),
                )
            }
        })
    }

    fn loop_attr(
        &self,
        name: &str,
        span: minijinja::machinery::Span,
    ) -> Result<String, CompileError> {
        if self.loop_depth == 0 {
            return self.err("`loop` used outside a for loop", span);
        }
        let d = self.loop_depth - 1;
        let b = |cond: String| format!("Val::Owned(::serde_json::Value::Bool({cond}))");
        let n = |num: String| format!("Val::Owned(::serde_json::Value::from({num}))");
        Ok(match name {
            "first" => b(format!("__i{d} == 0")),
            "last" => b(format!("__i{d} + 1 == __n{d}")),
            "index" => n(format!("(__i{d} + 1) as u64")),
            "index0" => n(format!("__i{d} as u64")),
            "length" => n(format!("__n{d} as u64")),
            "revindex" => n(format!("(__n{d} - __i{d}) as u64")),
            "revindex0" => n(format!("(__n{d} - __i{d} - 1) as u64")),
            other => return self.err(format!("unsupported `loop.{other}`"), span),
        })
    }
}

fn rust_str(s: &str) -> String {
    format!("{s:?}")
}

/// Lower a parsed constant to Rust. The parser const-folds, so `{{ 1 + 2 }}`
/// can arrive here already as `3`.
///
/// ⛔ EVERY ARM MUST BE A DIRECT CONSTRUCTOR. An earlier version emitted
/// `serde_json::from_str("0").unwrap()` for numbers — which re-parsed JSON text
/// at runtime, on every request, to recover an integer the template stated
/// literally. That is precisely the runtime work this project exists to delete,
/// and it is easy to reintroduce by reaching for `to_value` as a shortcut. If a
/// constant cannot be lowered to a literal here, that is a compile error, not a
/// reason to defer the work to run time.
fn const_to_rust(v: &minijinja::Value) -> String {
    if let Some(s) = v.as_str() {
        return format!("Val::Str(::std::borrow::Cow::Borrowed({}))", rust_str(s));
    }
    if v.is_none() || v.is_undefined() {
        // `none` is a value that renders "None"; undefined renders empty.
        return if v.is_undefined() {
            "Val::Undefined".to_string()
        } else {
            "Val::Owned(::serde_json::Value::Null)".to_string()
        };
    }
    match ::serde_json::to_value(v) {
        Ok(::serde_json::Value::Bool(b)) => {
            format!("Val::Owned(::serde_json::Value::Bool({b}))")
        }
        Ok(::serde_json::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                format!("Val::Owned(::serde_json::Value::Number({i}i64.into()))")
            } else if let Some(u) = n.as_u64() {
                format!("Val::Owned(::serde_json::Value::Number({u}u64.into()))")
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                format!(
                    "Val::Owned(::serde_json::Value::Number(\
                     ::serde_json::Number::from_f64({f}f64).expect(\"finite\")))"
                )
            }
        }
        // Composite literals (`[1,2]`, `{{'a':1}}`) are rare in chat templates and
        // would need a recursive literal builder. A build error naming the
        // construct beats emitting a runtime parse.
        _ => "__UNSUPPORTED_COMPOSITE_LITERAL__".to_string(),
    }
}

fn stmt_name(s: &Stmt<'_>) -> &'static str {
    match s {
        Stmt::Template(_) => "template",
        Stmt::EmitExpr(_) => "expression output",
        Stmt::EmitRaw(_) => "raw output",
        Stmt::ForLoop(_) => "for",
        Stmt::IfCond(_) => "if",
        Stmt::Set(_) => "set",
        Stmt::SetBlock(_) => "set block",
        Stmt::WithBlock(_) => "with",
        Stmt::AutoEscape(_) => "autoescape",
        Stmt::FilterBlock(_) => "filter block",
        Stmt::Do(_) => "do",
        _ => "statement",
    }
}

fn stmt_span(s: &Stmt<'_>) -> minijinja::machinery::Span {
    match s {
        Stmt::Template(x) => x.span(),
        Stmt::EmitExpr(x) => x.span(),
        Stmt::EmitRaw(x) => x.span(),
        Stmt::ForLoop(x) => x.span(),
        Stmt::IfCond(x) => x.span(),
        Stmt::Set(x) => x.span(),
        Stmt::SetBlock(x) => x.span(),
        Stmt::WithBlock(x) => x.span(),
        Stmt::AutoEscape(x) => x.span(),
        Stmt::FilterBlock(x) => x.span(),
        Stmt::Do(x) => x.span(),
        _ => minijinja::machinery::Span::default(),
    }
}
