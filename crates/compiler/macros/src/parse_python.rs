// SPDX-License-Identifier: Apache-2.0
//! The DSL front end: a lexer + recursive-descent parser for the
//! `dsl/<arch>.py` carrier dialect, lowered to [`Ast`]. Recognize the
//! DSL's statement and expression shapes, reject everything else.
//!
//! Hand-rolled, not a general Python parser: the compiler only has to
//! read the dialect below, and whether a carrier is valid *Python* is
//! checked by executing it under torch (the oracle). Expressions are
//! parsed with Python's grammar and precedence into [`PyExpr`], then
//! lowered with the DSL's restrictions. Identifiers are minted as
//! `syn::Ident`s at the caller's span; errors carry `line:col`.
//!
//! Dialect (a strict subset of Python):
//!   - `import` / `from` lines (ignored — they are the oracle's).
//!   - one `@forward` / `@vision_forward(...)`-decorated `def f():`,
//!     the decorator carrying the compile metadata: `workloads = [...]`,
//!     `sk_buckets = [...]`, `pixel_pack = path`, `processor = path`.
//!   - `name = expr`            → [`Stmt::Assign`]
//!   - `(a, b) = expr`          → [`Stmt::AssignTuple`]
//!   - `for ivar in range(b):`  → [`Stmt::For`]
//!   - `if <pred>:` / `else:`   → [`Stmt::If`]
//!   - a final `return <ident>` naming the last assigned value.
//!   - exprs: calls, attribute chains, `[ivar]` indexing, `*`/`+`,
//!     numeric literals, list literals (only inside reshape / the
//!     `in` predicate).
//!   - comments, docstrings, `;`-separated statements, and bracketed
//!     line continuation.
//!
//! Anything else — classes, comprehensions, walrus, star-args, keyword
//! args, `while`, `try`, one-line suites, tabs — is a loud error.

use crate::ast::{Ast, BoolExpr, BoundExpr, DimSpec, Expr, Stmt};
use proc_macro2::Span;
use std::fmt;

/// The carrier `def` plus its decorator metadata.
///
/// A carrier declares its arch's MATH only. The arch's other facts —
/// safetensors layout, rope style, decoder prefix, bound defaults, the
/// `Params` bound schema — are data about the checkpoints, so they
/// live with them, in `configs/<arch>/arch.json` (read by
/// [`crate::config::load_arch_json`]).
pub struct PythonCarrier {
    /// The parsed DSL body.
    pub ast: Ast,
    /// The arch name (the `def`'s name).
    pub arch_name: String,
    /// True for `@vision_forward`, false for `@forward`.
    pub vision: bool,
    /// `workloads = [...]` from the decorator (`None` = default ladder).
    pub workloads: Option<Vec<u64>>,
    /// `sk_buckets = [...]` from the decorator (`None` = default ladder).
    pub sk_buckets: Option<Vec<u64>>,
    /// `processor = <path>` from the decorator (vision carriers).
    pub processor: Option<syn::Path>,
    /// `pixel_pack = <path>` from the decorator (vision carriers).
    pub pixel_pack: Option<syn::Path>,
}

/// Parse a `.py` carrier file: the `@forward`/`@vision_forward`-decorated
/// top-level `def`, its decorator metadata, and its body as DSL statements.
pub fn parse_python_file(text: &str) -> Result<PythonCarrier, String> {
    let mut p = Parser {
        toks: lex(text)?,
        i: 0,
        span: Span::call_site(),
    };
    let mut carrier = None;
    loop {
        let at = p.at();
        match p.peek() {
            Tok::Eof => break,
            // Imports are the oracle's business, not ours.
            Tok::Name(kw) if kw == "import" || kw == "from" => p.skip_line(),
            Tok::Name(kw) if kw == "def" || kw == "async" => carrier = Some(p.carrier(carrier)?),
            Tok::Op("@") => carrier = Some(p.carrier(carrier)?),
            other => {
                return Err(err(
                    at,
                    format!(
                        "carrier file top level must be imports + one `def` — found {other}. \
                         The DSL declares only the arch's math; declarations (SAFETENSORS, \
                         SCALE_DTYPE, BOUND_DEFAULTS, …) belong in configs/<arch>/arch.json"
                    ),
                ));
            }
        }
    }
    carrier.ok_or_else(|| "carrier file has no `def` (the forward body)".to_string())
}

/// Test helper: parse a bare DSL body (any common indentation) as the
/// body of a `@forward` carrier.
#[cfg(test)]
pub(crate) fn parse_body(body: &str) -> Result<Ast, String> {
    let indent = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut src = String::from("@forward\ndef _carrier():\n");
    for line in body.lines() {
        src.push_str("    ");
        src.push_str(line.get(indent..).unwrap_or(""));
        src.push('\n');
    }
    parse_python_file(&src).map(|c| c.ast)
}

// ── Lexer ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pos {
    line: u32,
    col: u32,
}

fn err(at: Pos, msg: impl fmt::Display) -> String {
    format!("line {}:{}: {msg}", at.line, at.col)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Name(String),
    /// `None` when the literal does not fit a `u64`.
    Int(Option<u64>),
    Float(f64),
    Str,
    Op(&'static str),
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Name(n) => write!(f, "`{n}`"),
            Tok::Int(_) => f.write_str("an integer literal"),
            Tok::Float(_) => f.write_str("a float literal"),
            Tok::Str => f.write_str("a string literal"),
            Tok::Op(op) => write!(f, "`{op}`"),
            Tok::Newline => f.write_str("end of line"),
            Tok::Indent => f.write_str("an indented block"),
            Tok::Dedent => f.write_str("the end of the block"),
            Tok::Eof => f.write_str("end of file"),
        }
    }
}

/// Longest first, so `//` wins over `/`.
const OPS: &[&str] = &[
    "//", "==", "!=", "<=", ">=", "->", "**", "(", ")", "[", "]", "{", "}", ",", ".", ":", "=",
    "+", "-", "*", "/", "%", "@", "<", ">", ";",
];

const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

struct Cursor {
    src: Vec<char>,
    i: usize,
    pos: Pos,
}

impl Cursor {
    fn peek(&self, k: usize) -> Option<char> {
        self.src.get(self.i + k).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek(0)?;
        self.i += 1;
        if c == '\n' {
            self.pos.line += 1;
            self.pos.col = 1;
        } else {
            self.pos.col += 1;
        }
        Some(c)
    }

    fn skip_comment(&mut self) {
        while self.peek(0).is_some_and(|c| c != '\n') {
            self.bump();
        }
    }

    fn digits(&mut self, out: &mut String) {
        while let Some(c) = self.peek(0).filter(|c| c.is_ascii_digit() || *c == '_') {
            if c != '_' {
                out.push(c);
            }
            self.bump();
        }
    }
}

/// Python's logical-line tokenization: INDENT/DEDENT from leading
/// spaces, NEWLINE only outside brackets, blank and comment-only lines
/// invisible.
fn lex(text: &str) -> Result<Vec<(Tok, Pos)>, String> {
    let mut cur = Cursor {
        src: text.chars().collect(),
        i: 0,
        pos: Pos { line: 1, col: 1 },
    };
    let mut toks = Vec::new();
    let mut indents = vec![0u32];
    let mut depth = 0u32;
    let mut line_start = true;
    loop {
        if line_start {
            let mut width = 0;
            while let Some(c @ (' ' | '\t' | '\r')) = cur.peek(0) {
                if c == '\t' {
                    return Err(err(cur.pos, "tabs are not admitted — indent with spaces"));
                }
                if c == ' ' {
                    width += 1;
                }
                cur.bump();
            }
            match cur.peek(0) {
                None => break,
                Some('#') => {
                    cur.skip_comment();
                    continue;
                }
                Some('\n') => {
                    cur.bump();
                    continue;
                }
                Some(_) => {}
            }
            if width > indents[indents.len() - 1] {
                indents.push(width);
                toks.push((Tok::Indent, cur.pos));
            }
            while width < indents[indents.len() - 1] {
                indents.pop();
                toks.push((Tok::Dedent, cur.pos));
            }
            if width != indents[indents.len() - 1] {
                return Err(err(
                    cur.pos,
                    "dedent does not match any enclosing indentation level",
                ));
            }
            line_start = false;
        }
        let at = cur.pos;
        let Some(c) = cur.peek(0) else { break };
        match c {
            ' ' | '\t' | '\r' => {
                cur.bump();
            }
            '#' => cur.skip_comment(),
            '\\' if cur.peek(1) == Some('\n') => {
                cur.bump();
                cur.bump();
            }
            '\n' => {
                cur.bump();
                if depth == 0 {
                    toks.push((Tok::Newline, at));
                    line_start = true;
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut name = String::new();
                while let Some(c) = cur
                    .peek(0)
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                {
                    name.push(c);
                    cur.bump();
                }
                toks.push((Tok::Name(name), at));
            }
            c if c.is_ascii_digit()
                || (c == '.' && cur.peek(1).is_some_and(|d| d.is_ascii_digit())) =>
            {
                toks.push((lex_number(&mut cur)?, at));
            }
            '"' | '\'' => {
                lex_string(&mut cur)?;
                toks.push((Tok::Str, at));
            }
            _ => {
                let op = OPS
                    .iter()
                    .find(|op| {
                        op.chars()
                            .enumerate()
                            .all(|(k, oc)| cur.peek(k) == Some(oc))
                    })
                    .ok_or_else(|| err(at, format!("unexpected character `{c}`")))?;
                for _ in 0..op.len() {
                    cur.bump();
                }
                match *op {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => {
                        depth = depth
                            .checked_sub(1)
                            .ok_or_else(|| err(at, format!("unmatched `{op}`")))?;
                    }
                    _ => {}
                }
                toks.push((Tok::Op(op), at));
            }
        }
    }
    if depth != 0 {
        return Err(err(cur.pos, "unclosed bracket at end of file"));
    }
    if toks.last().is_some_and(|(t, _)| *t != Tok::Newline) {
        toks.push((Tok::Newline, cur.pos));
    }
    for _ in 1..indents.len() {
        toks.push((Tok::Dedent, cur.pos));
    }
    toks.push((Tok::Eof, cur.pos));
    Ok(toks)
}

/// Decimal int / float literal (`_` separators admitted).
fn lex_number(cur: &mut Cursor) -> Result<Tok, String> {
    let at = cur.pos;
    let mut text = String::new();
    let mut float = false;
    cur.digits(&mut text);
    if cur.peek(0) == Some('.') {
        float = true;
        text.push('.');
        cur.bump();
        cur.digits(&mut text);
    }
    if matches!(cur.peek(0), Some('e' | 'E')) {
        float = true;
        text.push('e');
        cur.bump();
        if let Some(sign @ ('+' | '-')) = cur.peek(0) {
            text.push(sign);
            cur.bump();
        }
        cur.digits(&mut text);
    }
    if cur
        .peek(0)
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(err(
            at,
            "numeric literals must be decimal (no hex/octal/binary, suffixes, or complex)",
        ));
    }
    if float {
        text.parse()
            .map(Tok::Float)
            .map_err(|_| err(at, format!("malformed float literal `{text}`")))
    } else {
        Ok(Tok::Int(text.parse().ok()))
    }
}

/// Skip a `'…'` / `"…"` / triple-quoted string. Only docstrings use
/// them; their contents are never read.
fn lex_string(cur: &mut Cursor) -> Result<(), String> {
    let at = cur.pos;
    let quote = cur.bump();
    let triple = cur.peek(0) == quote && cur.peek(1) == quote;
    if triple {
        cur.bump();
        cur.bump();
    }
    loop {
        match cur.bump() {
            None => return Err(err(at, "unterminated string literal")),
            Some('\\') => {
                cur.bump();
            }
            Some('\n') if !triple => return Err(err(at, "unterminated string literal")),
            c if c == quote => {
                if !triple {
                    return Ok(());
                }
                if cur.peek(0) == quote && cur.peek(1) == quote {
                    cur.bump();
                    cur.bump();
                    return Ok(());
                }
            }
            Some(_) => {}
        }
    }
}

// ── Parser: Python's grammar over the dialect's statements ──────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    MatMul,
    Pow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    In,
}

/// A Python expression as the parser reads it — Python's grammar and
/// precedence; the DSL's restrictions are applied when lowering.
#[derive(Debug)]
enum PyExpr {
    Name(String, Pos),
    Attr {
        value: Box<PyExpr>,
        attr: String,
        at: Pos,
    },
    Call {
        func: Box<PyExpr>,
        args: Vec<PyExpr>,
        kwargs: Vec<(String, PyExpr)>,
        at: Pos,
    },
    Subscript {
        value: Box<PyExpr>,
        index: Box<PyExpr>,
        at: Pos,
    },
    BinOp {
        op: BinOp,
        lhs: Box<PyExpr>,
        rhs: Box<PyExpr>,
        at: Pos,
    },
    Compare {
        op: CmpOp,
        lhs: Box<PyExpr>,
        rhs: Box<PyExpr>,
        at: Pos,
    },
    Unary(Pos),
    Int(Option<u64>, Pos),
    Float(f64, Pos),
    Str(Pos),
    List(Vec<PyExpr>, Pos),
    Tuple(Vec<PyExpr>, Pos),
}

impl PyExpr {
    fn at(&self) -> Pos {
        match self {
            PyExpr::Name(_, at)
            | PyExpr::Unary(at)
            | PyExpr::Int(_, at)
            | PyExpr::Float(_, at)
            | PyExpr::Str(at)
            | PyExpr::List(_, at)
            | PyExpr::Tuple(_, at)
            | PyExpr::Attr { at, .. }
            | PyExpr::Call { at, .. }
            | PyExpr::Subscript { at, .. }
            | PyExpr::BinOp { at, .. }
            | PyExpr::Compare { at, .. } => *at,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            PyExpr::Name(..) => "a name",
            PyExpr::Attr { .. } => "an attribute access",
            PyExpr::Call { .. } => "a call",
            PyExpr::Subscript { .. } => "a subscript",
            PyExpr::BinOp { .. } => "a binary operation",
            PyExpr::Compare { .. } => "a comparison",
            PyExpr::Unary(_) => "a unary operator",
            PyExpr::Int(Some(_), _) => "an integer literal",
            PyExpr::Int(None, _) => "an integer literal too large for u64",
            PyExpr::Float(..) => "a float literal",
            PyExpr::Str(_) => "a string literal",
            PyExpr::List(..) => "a list literal",
            PyExpr::Tuple(..) => "a tuple",
        }
    }
}

fn bin(op: BinOp, lhs: PyExpr, rhs: PyExpr) -> PyExpr {
    PyExpr::BinOp {
        op,
        at: lhs.at(),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// A call's positional args and `key=value` kwargs, in source order.
type CallArgs = (Vec<PyExpr>, Vec<(String, PyExpr)>);

struct Parser {
    /// Always ends in [`Tok::Eof`]; `bump` never moves past it.
    toks: Vec<(Tok, Pos)>,
    i: usize,
    span: Span,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].0
    }

    fn at(&self) -> Pos {
        self.toks[self.i].1
    }

    fn bump(&mut self) -> (Tok, Pos) {
        let tok = self.toks[self.i].clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        tok
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Tok::Op(o) if *o == op)
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == kw)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let hit = self.is_op(op);
        if hit {
            self.bump();
        }
        hit
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), String> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            Err(err(
                self.at(),
                format!("expected {what}, found {}", self.peek()),
            ))
        }
    }

    fn expect_op(&mut self, op: &'static str) -> Result<(), String> {
        self.expect(&Tok::Op(op), &format!("`{op}`"))
    }

    /// A non-keyword identifier.
    fn name(&mut self, what: &str) -> Result<String, String> {
        if let Tok::Name(n) = self.peek()
            && !KEYWORDS.contains(&n.as_str())
        {
            let n = n.clone();
            self.bump();
            return Ok(n);
        }
        Err(err(
            self.at(),
            format!("expected {what}, found {}", self.peek()),
        ))
    }

    fn skip_line(&mut self) {
        while !matches!(self.peek(), Tok::Newline | Tok::Eof) {
            self.bump();
        }
        self.bump();
    }

    // ── top level ──

    /// Decorators + `def name():` + body. `prior` is the carrier
    /// already seen, if any — a second `def` is an error.
    fn carrier(&mut self, prior: Option<PythonCarrier>) -> Result<PythonCarrier, String> {
        if prior.is_some() {
            return Err(err(
                self.at(),
                "carrier file must contain exactly one `def`",
            ));
        }
        let mut decorators = Vec::new();
        while self.eat_op("@") {
            decorators.push(self.expr()?);
            self.expect(&Tok::Newline, "end of line after the decorator")?;
        }
        let def_at = self.at();
        if self.is_kw("async") {
            return Err(err(def_at, "the carrier `def` cannot be `async`"));
        }
        if !self.is_kw("def") {
            return Err(err(
                def_at,
                format!("expected `def` after the decorator, found {}", self.peek()),
            ));
        }
        self.bump();
        let arch_name = self.name("the carrier's name")?;
        self.expect_op("(")?;
        if !self.eat_op(")") {
            return Err(err(
                self.at(),
                "the carrier `def` takes no parameters — its inputs are the prelude's free names",
            ));
        }
        let mut carrier = carrier_meta(arch_name, &decorators, def_at)?;
        self.expect_op(":")?;
        self.expect(&Tok::Newline, "a line break after `:`")?;
        self.expect(&Tok::Indent, "an indented block")?;
        carrier.ast.statements = self.block(true)?;
        Ok(carrier)
    }

    // ── statements ──

    /// `:` NEWLINE INDENT stmt+ DEDENT.
    fn suite(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect_op(":")?;
        self.expect(
            &Tok::Newline,
            "a line break after `:` (DSL blocks are indented on their own lines)",
        )?;
        self.expect(&Tok::Indent, "an indented block")?;
        self.block(false)
    }

    /// Statements up to and including the block's DEDENT. `top` is
    /// the carrier body, where a final `return <ident>` is admitted.
    fn block(&mut self, top: bool) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        loop {
            let at = self.at();
            let kw = match self.peek() {
                Tok::Dedent => {
                    self.bump();
                    return Ok(out);
                }
                Tok::Name(n) if KEYWORDS.contains(&n.as_str()) => n.clone(),
                _ => {
                    out.extend(self.simple_line()?);
                    continue;
                }
            };
            match kw.as_str() {
                "for" => out.push(self.for_stmt()?),
                "if" => out.push(self.if_stmt()?),
                "return" if top => self.return_stmt(&out)?,
                "return" => {
                    return Err(err(at, "`return` may only appear as the final statement"));
                }
                _ => {
                    return Err(err(
                        at,
                        format!("statement `{kw}` is not part of the DSL dialect"),
                    ));
                }
            }
        }
    }

    /// `target = value` NEWLINE, or a docstring (skipped → `None`).
    /// `simple (; simple)* [;]` NEWLINE — Python's simple-statement line.
    fn simple_line(&mut self) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        loop {
            out.extend(self.simple_stmt()?);
            if !self.eat_op(";") || self.peek() == &Tok::Newline {
                break;
            }
        }
        self.expect(&Tok::Newline, "end of line")?;
        Ok(out)
    }

    fn simple_stmt(&mut self) -> Result<Option<Stmt>, String> {
        let lhs = self.expr_or_tuple()?;
        if !self.eat_op("=") {
            if !matches!(self.peek(), Tok::Newline | Tok::Op(";")) {
                return Err(err(
                    self.at(),
                    format!("expected `=` or end of line, found {}", self.peek()),
                ));
            }
            return match lhs {
                PyExpr::Str(_) => Ok(None),
                other => Err(err(
                    other.at(),
                    format!(
                        "{} as a statement is not part of the DSL dialect",
                        other.kind()
                    ),
                )),
            };
        }
        let value = self.expr_or_tuple()?;
        if self.is_op("=") {
            return Err(err(self.at(), "chained assignment is not part of the DSL"));
        }
        let value = lower_expr(&value, self.span)?;
        match lhs {
            PyExpr::Name(n, _) => Ok(Some(Stmt::Assign {
                target: ident(&n, self.span),
                value,
            })),
            PyExpr::Tuple(elts, _) => {
                let targets = elts
                    .iter()
                    .map(|el| match el {
                        PyExpr::Name(n, _) => Ok(ident(n, self.span)),
                        other => Err(err(
                            other.at(),
                            "tuple destructuring targets must be plain identifiers",
                        )),
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Some(Stmt::AssignTuple { targets, value }))
            }
            other => Err(err(
                other.at(),
                "assignment target must be an identifier or a tuple of identifiers",
            )),
        }
    }

    fn for_stmt(&mut self) -> Result<Stmt, String> {
        self.bump();
        let ivar = self.name("the for-loop variable (a plain identifier)")?;
        if !self.is_kw("in") {
            return Err(err(
                self.at(),
                format!("expected `in`, found {}", self.peek()),
            ));
        }
        self.bump();
        let (start, end) = lower_range(&self.expr()?, self.span)?;
        let body = self.suite()?;
        if self.is_kw("else") {
            return Err(err(self.at(), "`for ... else` is not part of the DSL"));
        }
        Ok(Stmt::For {
            ivar: ident(&ivar, self.span),
            start,
            end,
            body,
        })
    }

    fn if_stmt(&mut self) -> Result<Stmt, String> {
        let at = self.bump().1;
        let cond = lower_bool(&self.expr()?, self.span)?;
        let then_body = self.suite()?;
        if self.is_kw("elif") {
            return Err(err(
                self.at(),
                "`elif` is not supported; nest `if` inside `else:`",
            ));
        }
        if !self.is_kw("else") {
            return Err(err(
                at,
                "`if` must have an `else:` arm; both arms must bind the same names",
            ));
        }
        self.bump();
        let else_body = self.suite()?;
        Ok(Stmt::If {
            cond,
            then_body,
            else_body,
        })
    }

    /// `return <ident>` — marks the forward's output, which is the
    /// last assigned value. Admitted only as the terminal statement,
    /// and only naming that value. Emits no statement.
    fn return_stmt(&mut self, prior: &[Stmt]) -> Result<(), String> {
        let at = self.bump().1;
        let name = match self.peek() {
            Tok::Newline => None,
            _ => match self.expr_or_tuple()? {
                PyExpr::Name(n, _) => Some(n),
                _ => None,
            },
        };
        let Some(name) = name else {
            return Err(err(
                at,
                "`return <ident>` must name the forward's final value",
            ));
        };
        self.expect(&Tok::Newline, "end of line")?;
        if !matches!(prior.last(), Some(Stmt::Assign { target, .. }) if target == name.as_str()) {
            return Err(err(
                at,
                format!("`return {name}` must name the value the preceding statement assigned"),
            ));
        }
        if self.peek() != &Tok::Dedent {
            return Err(err(
                self.at(),
                "`return` may only appear as the final statement",
            ));
        }
        Ok(())
    }

    // ── expressions (Python precedence) ──

    /// `expr (, expr)* [,]` — a bare tuple where Python admits one.
    fn expr_or_tuple(&mut self) -> Result<PyExpr, String> {
        let first = self.expr()?;
        if !self.is_op(",") {
            return Ok(first);
        }
        let at = first.at();
        let mut elts = vec![first];
        while self.eat_op(",") {
            if matches!(self.peek(), Tok::Newline | Tok::Op(")" | "]" | "=" | ":")) {
                break;
            }
            elts.push(self.expr()?);
        }
        Ok(PyExpr::Tuple(elts, at))
    }

    fn cmp_op(&self) -> Option<CmpOp> {
        Some(match self.peek() {
            Tok::Op("==") => CmpOp::Eq,
            Tok::Op("!=") => CmpOp::NotEq,
            Tok::Op("<") => CmpOp::Lt,
            Tok::Op("<=") => CmpOp::LtE,
            Tok::Op(">") => CmpOp::Gt,
            Tok::Op(">=") => CmpOp::GtE,
            Tok::Name(n) if n == "in" => CmpOp::In,
            _ => return None,
        })
    }

    fn expr(&mut self) -> Result<PyExpr, String> {
        let lhs = self.arith()?;
        let Some(op) = self.cmp_op() else {
            return Ok(lhs);
        };
        self.bump();
        let rhs = self.arith()?;
        if self.cmp_op().is_some() {
            return Err(err(
                self.at(),
                "chained comparisons are not part of the DSL",
            ));
        }
        Ok(PyExpr::Compare {
            op,
            at: lhs.at(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    }

    fn arith(&mut self) -> Result<PyExpr, String> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Tok::Op("+") => BinOp::Add,
                Tok::Op("-") => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            lhs = bin(op, lhs, self.term()?);
        }
    }

    fn term(&mut self) -> Result<PyExpr, String> {
        let mut lhs = self.factor()?;
        loop {
            let op = match self.peek() {
                Tok::Op("*") => BinOp::Mul,
                Tok::Op("/") => BinOp::Div,
                Tok::Op("//") => BinOp::FloorDiv,
                Tok::Op("%") => BinOp::Mod,
                Tok::Op("@") => BinOp::MatMul,
                _ => return Ok(lhs),
            };
            self.bump();
            lhs = bin(op, lhs, self.factor()?);
        }
    }

    fn factor(&mut self) -> Result<PyExpr, String> {
        if matches!(self.peek(), Tok::Op("-" | "+")) {
            let at = self.bump().1;
            self.factor()?;
            return Ok(PyExpr::Unary(at));
        }
        let base = self.primary()?;
        if self.eat_op("**") {
            let exp = self.factor()?;
            return Ok(bin(BinOp::Pow, base, exp));
        }
        Ok(base)
    }

    /// An atom followed by `.attr` / `(args)` / `[index]` trailers.
    fn primary(&mut self) -> Result<PyExpr, String> {
        let mut e = self.atom()?;
        loop {
            let at = e.at();
            if self.eat_op(".") {
                let attr = self.name("an attribute name")?;
                e = PyExpr::Attr {
                    value: Box::new(e),
                    attr,
                    at,
                };
            } else if self.eat_op("(") {
                let (args, kwargs) = self.call_args()?;
                e = PyExpr::Call {
                    func: Box::new(e),
                    args,
                    kwargs,
                    at,
                };
            } else if self.eat_op("[") {
                let index = self.expr_or_tuple()?;
                if self.is_op(":") {
                    return Err(err(self.at(), "slices are not part of the DSL"));
                }
                self.expect_op("]")?;
                e = PyExpr::Subscript {
                    value: Box::new(e),
                    index: Box::new(index),
                    at,
                };
            } else {
                return Ok(e);
            }
        }
    }

    /// Call arguments after the `(`, through the `)`.
    fn call_args(&mut self) -> Result<CallArgs, String> {
        let (mut args, mut kwargs) = (Vec::new(), Vec::new());
        while !self.eat_op(")") {
            if self.is_op("*") || self.is_op("**") {
                return Err(err(self.at(), "star-args are not part of the DSL"));
            }
            let arg = self.expr()?;
            if self.eat_op("=") {
                let PyExpr::Name(key, _) = arg else {
                    return Err(err(
                        arg.at(),
                        "keyword argument name must be a plain identifier",
                    ));
                };
                kwargs.push((key, self.expr()?));
            } else if kwargs.is_empty() {
                args.push(arg);
            } else {
                return Err(err(
                    arg.at(),
                    "positional argument follows keyword argument",
                ));
            }
            if !self.eat_op(",") {
                self.expect_op(")")?;
                break;
            }
        }
        Ok((args, kwargs))
    }

    fn atom(&mut self) -> Result<PyExpr, String> {
        let (tok, at) = self.bump();
        match tok {
            Tok::Name(n) if KEYWORDS.contains(&n.as_str()) => {
                Err(err(at, format!("`{n}` is not part of the DSL dialect")))
            }
            Tok::Name(n) => Ok(PyExpr::Name(n, at)),
            Tok::Int(v) => Ok(PyExpr::Int(v, at)),
            Tok::Float(f) => Ok(PyExpr::Float(f, at)),
            Tok::Str => {
                // Implicit concatenation: `"a" "b"` is one literal.
                while self.peek() == &Tok::Str {
                    self.bump();
                }
                Ok(PyExpr::Str(at))
            }
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Ok(PyExpr::Tuple(Vec::new(), at));
                }
                let inner = self.expr_or_tuple()?;
                self.expect_op(")")?;
                Ok(inner)
            }
            Tok::Op("[") => {
                let mut elts = Vec::new();
                while !self.eat_op("]") {
                    elts.push(self.expr()?);
                    if !self.eat_op(",") {
                        self.expect_op("]")?;
                        break;
                    }
                }
                Ok(PyExpr::List(elts, at))
            }
            other => Err(err(at, format!("expected an expression, found {other}"))),
        }
    }
}

// ── Lowering: PyExpr → the DSL's Ast, applying its restrictions ─────

fn ident(name: &str, at: Span) -> syn::Ident {
    syn::Ident::new(name, at)
}

fn int_lit(e: &PyExpr, what: &str) -> Result<u64, String> {
    match e {
        PyExpr::Int(Some(v), _) => Ok(*v),
        other => Err(err(
            other.at(),
            format!(
                "{what} must be a non-negative integer literal, found {}",
                other.kind()
            ),
        )),
    }
}

/// `a.b.c` → `["a", "b", "c"]`: a Name root plus attribute accesses.
fn dotted<'a>(e: &'a PyExpr, out: &mut Vec<&'a str>) -> Result<(), String> {
    match e {
        PyExpr::Name(n, _) => {
            out.push(n);
            Ok(())
        }
        PyExpr::Attr { value, attr, .. } => {
            dotted(value, out)?;
            out.push(attr);
            Ok(())
        }
        other => Err(err(
            other.at(),
            format!(
                "path root must be a plain identifier, found {}",
                other.kind()
            ),
        )),
    }
}

/// The decorator → the carrier's metadata (body left empty).
fn carrier_meta(
    arch_name: String,
    decorators: &[PyExpr],
    def_at: Pos,
) -> Result<PythonCarrier, String> {
    let mut carrier = PythonCarrier {
        ast: Ast {
            statements: Vec::new(),
        },
        arch_name,
        vision: false,
        workloads: None,
        sk_buckets: None,
        processor: None,
        pixel_pack: None,
    };
    let undecorated = "carrier `def` must be decorated with `@forward` or `@vision_forward(...)`";
    let dec = match decorators {
        [] => return Err(err(def_at, undecorated)),
        [dec] => dec,
        [_, second, ..] => {
            return Err(err(
                second.at(),
                "carrier `def` carries multiple decorators",
            ));
        }
    };
    let (name, kwargs) = match dec {
        PyExpr::Name(n, _) => (n, &[][..]),
        PyExpr::Call {
            func,
            args,
            kwargs,
            at,
        } => {
            let PyExpr::Name(n, _) = &**func else {
                return Err(err(*at, undecorated));
            };
            if !args.is_empty() {
                return Err(err(*at, "decorator takes keyword arguments only"));
            }
            (n, kwargs.as_slice())
        }
        other => return Err(err(other.at(), undecorated)),
    };
    carrier.vision = match name.as_str() {
        "forward" => false,
        "vision_forward" => true,
        _ => {
            return Err(err(
                dec.at(),
                format!("unknown decorator `@{name}` — expected `@forward` / `@vision_forward`"),
            ));
        }
    };
    for (key, value) in kwargs {
        match (key.as_str(), value) {
            ("workloads" | "sk_buckets", PyExpr::List(elts, _)) => {
                let what = format!("each `{key}` entry");
                let vals = elts
                    .iter()
                    .map(|v| int_lit(v, &what))
                    .collect::<Result<_, _>>()?;
                if key == "workloads" {
                    carrier.workloads = Some(vals);
                } else {
                    carrier.sk_buckets = Some(vals);
                }
            }
            ("processor" | "pixel_pack", PyExpr::Attr { .. }) => {
                let mut segs = Vec::new();
                dotted(value, &mut segs)?;
                let path = Some(
                    syn::parse_str::<syn::Path>(&segs.join("::"))
                        .map_err(|e| err(value.at(), format!("decorator `{key}`: {e}")))?,
                );
                if key == "processor" {
                    carrier.processor = path;
                } else {
                    carrier.pixel_pack = path;
                }
            }
            _ => {
                return Err(err(
                    value.at(),
                    format!(
                        "unknown or ill-typed decorator argument `{key}` — expected \
                         `workloads = [..]`, `sk_buckets = [..]`, `processor = path`, \
                         or `pixel_pack = path`"
                    ),
                ));
            }
        }
    }
    Ok(carrier)
}

/// `range(end)` / `range(start, end)` (start defaults 0).
fn lower_range(iter: &PyExpr, span: Span) -> Result<(BoundExpr, BoundExpr), String> {
    let not_range = || err(iter.at(), "for-loop must iterate `range(<bound>)`");
    let PyExpr::Call {
        func, args, kwargs, ..
    } = iter
    else {
        return Err(not_range());
    };
    if !matches!(&**func, PyExpr::Name(n, _) if n == "range") || !kwargs.is_empty() {
        return Err(not_range());
    }
    match args.as_slice() {
        [end] => Ok((BoundExpr::Lit(0), lower_bound(end, span)?)),
        [start, end] => Ok((lower_bound(start, span)?, lower_bound(end, span)?)),
        _ => Err(err(
            iter.at(),
            format!("range(...) takes 1 or 2 bounds, got {}", args.len()),
        )),
    }
}

fn lower_bound(e: &PyExpr, span: Span) -> Result<BoundExpr, String> {
    match e {
        PyExpr::Name(n, _) => Ok(BoundExpr::Ident(ident(n, span))),
        PyExpr::Int(..) => Ok(BoundExpr::Lit(int_lit(e, "a loop bound")?)),
        other => Err(err(
            other.at(),
            format!(
                "loop bound must be an integer literal or a bare identifier, found {}",
                other.kind()
            ),
        )),
    }
}

fn lower_bool(e: &PyExpr, span: Span) -> Result<BoolExpr, String> {
    let bad = || {
        err(
            e.at(),
            "`if` condition must be `ivar % N == M`, `ivar % N != M`, `ivar < N`, \
             or `ivar in [lit, lit, ...]`",
        )
    };
    let ivar_of = |x: &PyExpr| match x {
        PyExpr::Name(n, _) => Ok(ident(n, span)),
        other => Err(err(
            other.at(),
            "expected a plain identifier (the enclosing loop induction variable)",
        )),
    };
    let PyExpr::Compare { op, lhs, rhs, .. } = e else {
        return Err(bad());
    };
    match (op, &**lhs) {
        (CmpOp::In, _) => {
            let PyExpr::List(elts, _) = &**rhs else {
                return Err(err(
                    rhs.at(),
                    "`in` predicate requires a list literal `[lit, lit, ...]`",
                ));
            };
            if elts.is_empty() {
                return Err(err(rhs.at(), "`in` list must have at least one element"));
            }
            Ok(BoolExpr::In {
                ivar: ivar_of(lhs)?,
                members: elts
                    .iter()
                    .map(|m| int_lit(m, "an `in` list element"))
                    .collect::<Result<_, _>>()?,
            })
        }
        (
            CmpOp::Eq | CmpOp::NotEq,
            PyExpr::BinOp {
                op: BinOp::Mod,
                lhs: v,
                rhs: d,
                ..
            },
        ) => {
            let ivar = ivar_of(v)?;
            let divisor = lower_bound(d, span)?;
            let remainder = lower_bound(rhs, span)?;
            Ok(if *op == CmpOp::Eq {
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
            })
        }
        (CmpOp::Lt, _) => Ok(BoolExpr::Less {
            ivar: ivar_of(lhs)?,
            bound: lower_bound(rhs, span)?,
        }),
        _ => Err(bad()),
    }
}

fn lower_expr(e: &PyExpr, span: Span) -> Result<Expr, String> {
    match e {
        // A simple identifier — `hidden_states`, `input_ids`.
        PyExpr::Name(n, _) => Ok(Expr::Var(ident(n, span))),
        // Attribute chains — `self_attn.q_proj` (always >= 2 segments).
        PyExpr::Attr { .. } => {
            let mut segs = Vec::new();
            dotted(e, &mut segs)?;
            Ok(Expr::Path(segs.iter().map(|s| ident(s, span)).collect()))
        }
        // Indexing: `self_attn.q_proj[layer]`.
        PyExpr::Subscript { value, index, .. } => {
            let PyExpr::Name(i, _) = &**index else {
                return Err(err(
                    index.at(),
                    "index must be a single identifier (the enclosing loop variable)",
                ));
            };
            Ok(Expr::Index {
                target: Box::new(lower_expr(value, span)?),
                index: ident(i, span),
            })
        }
        PyExpr::Call {
            func,
            args,
            kwargs,
            at,
        } => lower_call(func, args, kwargs, *at, span),
        // `gate * up` / `w + 1.0`.
        PyExpr::BinOp {
            op: op @ (BinOp::Mul | BinOp::Add),
            lhs,
            rhs,
            ..
        } => {
            let lhs = Box::new(lower_expr(lhs, span)?);
            let rhs = Box::new(lower_expr(rhs, span)?);
            Ok(if *op == BinOp::Mul {
                Expr::Mul { lhs, rhs }
            } else {
                Expr::Add { lhs, rhs }
            })
        }
        PyExpr::BinOp { op, at, .. } => Err(err(
            *at,
            format!("only `*` and `+` are admitted in DSL expressions (got {op:?})"),
        )),
        // Numeric literal — int promoted to f64.
        PyExpr::Int(..) => Ok(Expr::ScalarLit(int_lit(e, "a DSL int literal")? as f64)),
        PyExpr::Float(f, _) => Ok(Expr::ScalarLit(*f)),
        other => Err(err(
            other.at(),
            format!("{} is not part of the DSL dialect", other.kind()),
        )),
    }
}

/// `gemm(x, w)`, `rmsnorm(x, w)`, … plus the compile-time forms
/// `sqrt(bound)`, `scalar(key)` / `recip_scalar(key)`, and
/// `reshape(source, [dims])`.
fn lower_call(
    func: &PyExpr,
    args: &[PyExpr],
    kwargs: &[(String, PyExpr)],
    at: Pos,
    span: Span,
) -> Result<Expr, String> {
    if !kwargs.is_empty() {
        return Err(err(at, "keyword arguments are not part of the DSL"));
    }
    let PyExpr::Name(op, _) = func else {
        return Err(err(func.at(), "call target must be a plain op name"));
    };
    let name_arg = |usage: &str| match args {
        [PyExpr::Name(n, _)] => Ok(ident(n, span)),
        _ => Err(err(at, usage)),
    };
    match (op.as_str(), args) {
        ("sqrt", _) => Ok(Expr::SqrtBound(name_arg(
            "`sqrt(<bound_name>)` takes exactly one bound identifier",
        )?)),
        ("scalar" | "recip_scalar", _) => Ok(Expr::ConfigScalar {
            name: name_arg(
                "`scalar(<name>)` / `recip_scalar(<name>)` takes exactly one config key identifier",
            )?,
            recip: op == "recip_scalar",
        }),
        ("reshape", [source, shape]) => Ok(Expr::Reshape {
            source: Box::new(lower_expr(source, span)?),
            target_shape: lower_dims(shape, span)?,
        }),
        ("reshape", _) => Err(err(
            at,
            "`reshape(source, [dim0, dim1, ...])` takes exactly two arguments",
        )),
        _ => Ok(Expr::Call {
            op: ident(op, span),
            args: args
                .iter()
                .map(|a| lower_expr(a, span))
                .collect::<Result<_, _>>()?,
        }),
    }
}

/// `reshape`'s second arg: a list of dims, each a literal / bound /
/// `*` / `/` tree.
fn lower_dims(e: &PyExpr, span: Span) -> Result<Vec<DimSpec>, String> {
    match e {
        PyExpr::List(elts, at) if elts.is_empty() => {
            Err(err(*at, "reshape target shape must have at least one dim"))
        }
        PyExpr::List(elts, _) => elts.iter().map(|d| lower_dim(d, span)).collect(),
        other => Err(err(
            other.at(),
            "reshape target shape must be a list literal `[dim0, dim1, ...]`",
        )),
    }
}

fn lower_dim(e: &PyExpr, span: Span) -> Result<DimSpec, String> {
    match e {
        PyExpr::Int(..) => Ok(DimSpec::Lit(int_lit(e, "a reshape dim")?)),
        PyExpr::Name(n, _) => Ok(DimSpec::Bound(ident(n, span))),
        // Python floor-division `//` is the DSL's integer `/` too.
        PyExpr::BinOp {
            op: op @ (BinOp::Mul | BinOp::Div | BinOp::FloorDiv),
            lhs,
            rhs,
            ..
        } => {
            let lhs = Box::new(lower_dim(lhs, span)?);
            let rhs = Box::new(lower_dim(rhs, span)?);
            Ok(if *op == BinOp::Mul {
                DimSpec::Mul(lhs, rhs)
            } else {
                DimSpec::Div(lhs, rhs)
            })
        }
        other => Err(err(
            other.at(),
            "reshape dim must be an integer literal, a bound identifier, \
             or a `*` / `/` / `//` arithmetic over those",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Ast {
        parse_python_file(src).expect("DSL parse").ast
    }

    fn parse_err(src: &str) -> String {
        match parse_python_file(src) {
            Ok(_) => panic!("expected a parse error for:\n{src}"),
            Err(e) => e,
        }
    }

    /// A bare carrier body → its `Ast`.
    fn body(src: &str) -> Ast {
        parse_body(src).expect("DSL parse")
    }

    /// A bare carrier body that must be rejected → the error.
    fn body_err(src: &str) -> String {
        match parse_body(src) {
            Ok(_) => panic!("expected a parse error for:\n{src}"),
            Err(e) => e,
        }
    }

    /// The first statement of a one-loop body's `for`.
    fn loop_body(ast: &Ast) -> &[Stmt] {
        match &ast.statements[0] {
            Stmt::For { body, .. } => body,
            other => panic!("expected for-loop, got {other:?}"),
        }
    }

    fn reshape_dims(ast: &Ast) -> &[DimSpec] {
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Reshape { target_shape, .. },
                ..
            } => target_shape,
            other => panic!("expected reshape Assign, got {other:?}"),
        }
    }

    #[test]
    fn assign_simple() {
        let ast = body("x = foo(a, b)");
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            Stmt::Assign {
                target,
                value: Expr::Call { op, args },
            } => {
                assert_eq!(target.to_string(), "x");
                assert_eq!(op.to_string(), "foo");
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected call assign, got {other:?}"),
        }
    }

    #[test]
    fn assign_tuple() {
        let ast = body("(q, k, v) = rope(q, k, v, pos)");
        match &ast.statements[0] {
            Stmt::AssignTuple { targets, value } => {
                let names: Vec<String> = targets.iter().map(|i| i.to_string()).collect();
                assert_eq!(names, vec!["q", "k", "v"]);
                assert!(matches!(value, Expr::Call { .. }));
            }
            other => panic!("expected tuple assign, got {other:?}"),
        }
    }

    #[test]
    fn for_loop_with_symbolic_bound() {
        let ast = body(
            "
            for layer in range(num_hidden_layers):
                x = y
            ",
        );
        match &ast.statements[0] {
            Stmt::For {
                ivar,
                start,
                end,
                body,
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert!(matches!(start, BoundExpr::Lit(0)));
                assert!(matches!(end, BoundExpr::Ident(i) if i == "num_hidden_layers"));
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected for-loop, got {other:?}"),
        }
    }

    #[test]
    fn dotted_path_expression() {
        let ast = body("x = self_attn.q_proj");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Path(segs),
                ..
            } => {
                let names: Vec<String> = segs.iter().map(|i| i.to_string()).collect();
                assert_eq!(names, vec!["self_attn", "q_proj"]);
            }
            other => panic!("expected path expression, got {other:?}"),
        }
    }

    #[test]
    fn indexed_weight_reference() {
        let ast = body("q = gemm(x, self_attn.q_proj[layer])");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[1] {
                Expr::Index { target, index } => {
                    assert!(matches!(&**target, Expr::Path(segs) if segs.len() == 2));
                    assert_eq!(index.to_string(), "layer");
                }
                other => panic!("expected indexed path, got {other:?}"),
            },
            other => panic!("expected call, got {other:?}"),
        }
    }

    #[test]
    fn mul_expression() {
        let ast = body("down = gemm(gate * up, mlp.down_proj[layer])");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[0] {
                Expr::Mul { lhs, rhs } => match (&**lhs, &**rhs) {
                    (Expr::Var(l), Expr::Var(r)) => {
                        assert_eq!(l.to_string(), "gate");
                        assert_eq!(r.to_string(), "up");
                    }
                    other => panic!("expected (gate, up) var operands, got {other:?}"),
                },
                other => panic!("expected Mul, got {other:?}"),
            },
            other => panic!("expected call, got {other:?}"),
        }
    }

    #[test]
    fn bare_var_expression() {
        let ast = body("x = add(a, b)");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => assert!(matches!(&args[1], Expr::Var(i) if i == "b")),
            other => panic!("expected call, got {other:?}"),
        }
    }

    #[test]
    fn keyword_statement_is_rejected() {
        let e = body_err("del x");
        assert!(e.contains("`del`"), "error should name the statement: {e}");
    }

    #[test]
    fn if_modulo_parses() {
        let ast = body(
            "
            for layer in range(4):
                if layer % 2 == 0:
                    x = attention(q, k, v, kv, b)
                else:
                    x = sliding_attention(q, k, v, kv, b)
            ",
        );
        match &loop_body(&ast)[0] {
            Stmt::If {
                cond:
                    BoolExpr::Modulo {
                        ivar,
                        divisor,
                        remainder,
                    },
                then_body,
                else_body,
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert!(matches!(divisor, BoundExpr::Lit(2)));
                assert!(matches!(remainder, BoundExpr::Lit(0)));
                assert_eq!(then_body.len(), 1);
                assert_eq!(else_body.len(), 1);
            }
            other => panic!("expected Modulo If, got {other:?}"),
        }
    }

    #[test]
    fn if_not_modulo_parses() {
        let ast = body(
            "
            for layer in range(4):
                if layer % full_attention_interval != 3:
                    x = gemm(a, b)
                else:
                    x = gemm(a, b)
            ",
        );
        assert!(matches!(
            &loop_body(&ast)[0],
            Stmt::If {
                cond: BoolExpr::NotModulo {
                    divisor: BoundExpr::Ident(_),
                    remainder: BoundExpr::Lit(3),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn if_less_with_symbolic_bound_parses() {
        let ast = body(
            "
            for layer in range(4):
                if layer < num_dense_layers:
                    x = gemm(a, b)
                else:
                    x = gemm(a, b)
            ",
        );
        match &loop_body(&ast)[0] {
            Stmt::If {
                cond: BoolExpr::Less { ivar, bound },
                ..
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert!(matches!(bound, BoundExpr::Ident(i) if i == "num_dense_layers"));
            }
            other => panic!("expected Less If, got {other:?}"),
        }
    }

    #[test]
    fn if_in_literal_array_parses() {
        let ast = body(
            "
            for layer in range(32):
                if layer in [7, 15, 23, 31]:
                    x = gemm(a, b)
                else:
                    x = gemm(a, b)
            ",
        );
        match &loop_body(&ast)[0] {
            Stmt::If {
                cond: BoolExpr::In { ivar, members },
                ..
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert_eq!(members, &vec![7u64, 15, 23, 31]);
            }
            other => panic!("expected In If, got {other:?}"),
        }
    }

    /// The `if` header of a loop body, with both arms filled in.
    fn if_src(cond: &str) -> String {
        format!(
            "for layer in range(32):\n    if {cond}:\n        x = gemm(a, b)\n    \
             else:\n        x = gemm(a, b)\n"
        )
    }

    #[test]
    fn if_in_with_symbolic_bound_in_array_is_rejected() {
        let e = body_err(&if_src("layer in [7, foo, 23]"));
        assert!(e.contains("integer literal"), "{e}");
    }

    #[test]
    fn if_in_empty_array_is_rejected() {
        let e = body_err(&if_src("layer in []"));
        assert!(e.contains("at least one element"), "{e}");
    }

    #[test]
    fn if_in_non_list_is_rejected() {
        let e = body_err(&if_src("layer in (7, 15)"));
        assert!(e.contains("list literal"), "{e}");
    }

    #[test]
    fn if_without_else_is_rejected() {
        let e = body_err(
            "
            for layer in range(4):
                if layer < 2:
                    x = gemm(a, b)
            ",
        );
        assert!(e.contains("else"), "{e}");
    }

    #[test]
    fn unsupported_condition_shape_is_rejected() {
        let e = body_err(&if_src("layer + 1 == 3"));
        assert!(e.contains("ivar % N"), "error names supported shapes: {e}");
    }

    #[test]
    fn realistic_llama_body_parses() {
        let ast = body(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                attn = attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)

                normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
                gate = silu(gemm(normed2, mlp.gate_proj[layer]))
                up = gemm(normed2, mlp.up_proj[layer])
                down = gemm(gate * up, mlp.down_proj[layer])
                hidden_states = add(down, hidden_states)
            normed = rmsnorm(hidden_states, norm)
            logits = gemm(normed, lm_head)
            ",
        );
        assert_eq!(
            ast.statements.len(),
            4,
            "expected 4 top-level statements (embed, for, norm, lm_head)",
        );
        assert!(matches!(ast.statements[0], Stmt::Assign { .. }));
        match &ast.statements[1] {
            Stmt::For {
                ivar, end, body, ..
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert!(matches!(end, BoundExpr::Ident(i) if i == "num_hidden_layers"));
                // normed, q, k, v, (q,k,v)=rope, attn, oproj, hidden=add,
                // normed2, gate, up, down, hidden=add
                assert_eq!(body.len(), 13, "13 body statements per iteration");
            }
            other => panic!("expected for-loop, got {other:?}"),
        }
    }

    #[test]
    fn reshape_with_bound_dims_parses() {
        let ast = body("y = reshape(x, [num_tokens, vision_embed_dim])");
        let dims = reshape_dims(&ast);
        assert_eq!(dims.len(), 2);
        assert!(matches!(&dims[0], DimSpec::Bound(i) if i == "num_tokens"));
        assert!(matches!(&dims[1], DimSpec::Bound(i) if i == "vision_embed_dim"));
    }

    #[test]
    fn reshape_with_literal_dims_parses() {
        let ast = body("y = reshape(x, [256, 1280])");
        let dims = reshape_dims(&ast);
        assert!(matches!(dims[0], DimSpec::Lit(256)));
        assert!(matches!(dims[1], DimSpec::Lit(1280)));
    }

    #[test]
    fn reshape_with_wrong_arity_is_rejected() {
        for src in ["y = reshape(x)", "y = reshape(x, [1], [2])"] {
            let e = body_err(src);
            assert!(e.contains("exactly two arguments"), "{src}: {e}");
        }
    }

    #[test]
    fn reshape_with_non_list_second_arg_is_rejected() {
        let e = body_err("y = reshape(x, hidden_size)");
        assert!(e.contains("list literal"), "{e}");
    }

    #[test]
    fn reshape_with_div_dim_parses() {
        for src in [
            "y = reshape(x, [num_tokens / vision_merge_factor, vision_merge_hidden])",
            "y = reshape(x, [num_tokens // vision_merge_factor, vision_merge_hidden])",
        ] {
            let ast = body(src);
            let dims = reshape_dims(&ast);
            assert_eq!(dims.len(), 2);
            match &dims[0] {
                DimSpec::Div(num, den) => {
                    assert!(matches!(num.as_ref(), DimSpec::Bound(i) if i == "num_tokens"));
                    assert!(
                        matches!(den.as_ref(), DimSpec::Bound(i) if i == "vision_merge_factor")
                    );
                }
                other => panic!("{src}: expected DimSpec::Div, got {other:?}"),
            }
            assert!(matches!(&dims[1], DimSpec::Bound(i) if i == "vision_merge_hidden"));
        }
    }

    #[test]
    fn reshape_with_mul_and_lit_arithmetic_parses() {
        let ast = body("y = reshape(x, [num_tokens, vision_embed_dim * 4])");
        match &reshape_dims(&ast)[1] {
            DimSpec::Mul(a, b) => {
                assert!(matches!(a.as_ref(), DimSpec::Bound(i) if i == "vision_embed_dim"));
                assert!(matches!(b.as_ref(), DimSpec::Lit(4)));
            }
            other => panic!("expected DimSpec::Mul, got {other:?}"),
        }
    }

    #[test]
    fn reshape_arithmetic_groups_with_parens() {
        let ast = body("y = reshape(x, [(num_tokens / 2) * 3])");
        match &reshape_dims(&ast)[0] {
            DimSpec::Mul(lhs, rhs) => {
                assert!(matches!(rhs.as_ref(), DimSpec::Lit(3)));
                match lhs.as_ref() {
                    DimSpec::Div(num, den) => {
                        assert!(matches!(num.as_ref(), DimSpec::Bound(i) if i == "num_tokens"));
                        assert!(matches!(den.as_ref(), DimSpec::Lit(2)));
                    }
                    other => panic!("expected DimSpec::Div nested, got {other:?}"),
                }
            }
            other => panic!("expected DimSpec::Mul, got {other:?}"),
        }
    }

    #[test]
    fn reshape_with_unsupported_binop_is_rejected() {
        let e = body_err("y = reshape(x, [num_tokens + 4])");
        assert!(e.contains("`*` / `/`"), "{e}");
    }

    /// `;`-separated simple statements on one line are Python, and
    /// each is its own statement.
    #[test]
    fn semicolon_separated_statements() {
        let ast = body("x = embed(input_ids, embed_tokens); x = add(x, x);");
        assert_eq!(ast.statements.len(), 2);
        body_err("x = a; for i in range(2):\n    y = b\n");
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

    /// Comments, docstrings, blank lines, and a call split across
    /// lines inside brackets are layout, not statements.
    #[test]
    fn layout_is_invisible() {
        let ast = parse(
            "import torch\n\n# header\n@forward\ndef f():\n    \"\"\"doc\n    string\"\"\"\n\n    \
             # a comment\n    x = reshape(\n        y,\n        [a // 2, b],  # trailing\n    )\n    \
             return x\n",
        );
        assert_eq!(ast.statements.len(), 1);
        assert!(matches!(
            &ast.statements[0],
            Stmt::Assign {
                value: Expr::Reshape { .. },
                ..
            }
        ));
    }

    #[test]
    fn subset_is_enforced() {
        // comprehension — outside the dialect, must be a loud error.
        parse_err("@forward\ndef f():\n    x = [y for y in z]\n");
        // class at top level — rejected.
        parse_err("class A:\n    pass\n");
        // no decorator — rejected.
        parse_err("def f():\n    x = y\n");
        // one-line suite, tabs, `elif`, keyword args, slices — rejected.
        parse_err("@forward\ndef f():\n    for i in range(n): x = y\n");
        parse_err("@forward\ndef f():\n\tx = y\n");
        parse_err("@forward\ndef f():\n    x = g(a, k=1)\n");
        parse_err("@forward\ndef f():\n    x = w[0:2]\n");
        // `return` anywhere but last, or naming the wrong value.
        parse_err("@forward\ndef f():\n    x = y\n    return x\n    z = x\n");
        parse_err("@forward\ndef f():\n    x = y\n    return y\n");
    }

    #[test]
    fn errors_carry_line_and_column() {
        let e = parse_err(
            "@forward\ndef f():\n    for layer in range(n):\n        if layer < 2:\n            \
             x = a\n        elif layer < 4:\n            x = b\n",
        );
        assert!(e.starts_with("line 6:9:"), "{e}");
        assert!(e.contains("`elif`"), "{e}");
    }

    #[test]
    fn decorator_metadata_is_parsed() {
        let carrier = parse_python_file(
            "@vision_forward(workloads=[256, 1024], processor=crate.PROCESSOR)\n\
             def vision():\n    x = gemm(a, b)\n    return x\n",
        )
        .expect("decorator parse");
        assert_eq!(carrier.arch_name, "vision");
        assert!(carrier.vision);
        assert_eq!(carrier.workloads, Some(vec![256, 1024]));
        assert!(carrier.sk_buckets.is_none());
        let processor = carrier.processor.expect("processor path");
        let segs: Vec<_> = processor
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        assert_eq!(segs, ["crate", "PROCESSOR"]);
        assert!(carrier.pixel_pack.is_none());

        let carrier = parse_python_file(
            "@forward(sk_buckets=[])\ndef plain():\n    x = gemm(a, b)\n    return x\n",
        )
        .expect("decorator parse");
        assert_eq!(carrier.arch_name, "plain");
        assert!(!carrier.vision);
        assert!(carrier.workloads.is_none());
        // An explicit empty list is kept distinct from an omitted one.
        assert_eq!(carrier.sk_buckets, Some(vec![]));

        parse_err("@forward(bogus=1)\ndef f():\n    x = y\n    return x\n");
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

    /// `qwen3.py` drives the full parse → classify → shape → CFG →
    /// unroll → schedule → codegen chain to a NON-EMPTY, complete
    /// emission, without pinning bytes. Numerical equivalence of the
    /// emission is covered by the scratchy-models `qwen3_py_parity`
    /// e2e gate.
    #[test]
    fn python_costume_emits_the_qwen3_tape() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let arch = "qwen3";
        let configs_dir = root.join("crates/models/arch/configs").join(arch);

        let py_text =
            std::fs::read_to_string(root.join("crates/models/arch/dsl/qwen3.py")).unwrap();
        let py_carrier = parse_python_file(&py_text).expect("python parse");
        let py_tokens =
            crate::compile_carrier(py_carrier, &configs_dir).expect("python-costume pipeline");

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
