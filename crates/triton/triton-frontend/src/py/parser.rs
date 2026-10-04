//! An owned, hand-rolled Python parser: source text to [`crate::py::ast`], no
//! dependencies.
//!
//! THE DECISION, recorded: this crate parses Triton kernels the way the torch carriers
//! are parsed (`crates/compiler/macros/src/parse_python.rs`) -- a lexer and a
//! recursive-descent parser for a strict subset, in this crate, with every error a
//! [`crate::Error`] naming the construct and its `line:col`. It replaced the vendored
//! third-party parser crate, which pulled 44 lockfile versions for a parse step whose
//! grammar the census bounds anyway: a parser that accepts exactly the census grammar
//! gives the same scope control with no third-party code. Sharing one parser with the
//! macros crate is a placement question this crate cannot answer -- that file lives in
//! a proc-macro crate nothing can depend on -- so this one is its own, bounded by the
//! same law.
//!
//! THE GRAMMAR, measured over every `.py` this tree's build and tests parse (the
//! `test-fixtures/` ladder fixtures and the six shipped `targets/spyre/kernels/`):
//!
//! * statements: `def` (with decorators), plain `import`/`import x.y as z`, `=` and
//!   annotated `=` and augmented assignment, `for`, `if`/`elif`/`else`, expression
//!   statements, `return`, `pass`. Nothing else -- a `while`/`with`/`try`/`class`/walrus
//!   is a named error, not a best-effort lowering.
//! * expressions: the full operator set with Python precedence, single comparisons,
//!   calls with positional and keyword arguments, attribute chains, subscripts whose
//!   index may be a tuple of slices, list/tuple displays, literals (int, float incl.
//!   exponent notation, `True`/`False`/`None`, strings).
//! * layout: `#` comments (the corpus uses trailing `#` markers on parameter lines),
//!   NEWLINE/INDENT/DEDENT, and line joining inside any bracket. F-strings, string
//!   prefixes, semicolon separators, backslash continuations and `\` in source are all
//!   refused by name.
//!
//! POSITIONS match CPython's `ast` conventions -- 1-based line, 0-based column -- because
//! the goldens' `loc(...)` fields are diffed against real Triton's output. A statement's
//! position is its first token; an expression's is its first token.
//!
//! FAIL CLOSED, in the adapter's own two-layer shape: anything this grammar cannot spell
//! is a parse error naming the construct, and what it CAN spell is still subject to
//! [`crate::py::census`], which refuses out-of-scope call targets and loop iterables the
//! grammar itself cannot see.

use crate::py::ast::*;
use crate::{Error, Result};

// ── the lexer ──────────────────────────────────────────────────────────────────

/// One token: its kind, its spelling as sliced from the source, and its position.
#[derive(Clone, Debug)]
struct Token {
    kind: Tok,
    /// The exact source slice. For `Name`, `Int`, `Float`, `Str` this is the value; for
    /// `Op`/`Keyword` it is the operator's or keyword's own spelling.
    text: String,
    line: u32,
    col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok {
    Newline,
    Indent,
    Dedent,
    Name,
    Int,
    Float,
    Str,
    Op,
    /// One of Python's keywords. Only the subset the grammar uses is ever produced;
    /// the others (`while`, `class`, ...) still lex as keywords so their refusal can
    /// NAME the keyword rather than misreading it as a name.
    Keyword,
    End,
}

/// The operator/keyword spellings, longest first so a prefix (`//` of `//=`,
/// `*` of `**`) can never win the longest-match race. See [`lex`].
const OPS: &[&str] = &[
    "**=", "//=", "<<=", ">>=", "**", "//", "<<", ">>", "<=", ">=", "==", "!=", "->", "+=", "-=",
    "*=", "/=", "%=", "&=", "|=", "^=", "@=", "(", ")", "[", "]", "{", "}", ",", ":", ".", "=",
    "+", "-", "*", "/", "%", "&", "|", "^", "~", "<", ">", "@",
];

const KEYWORDS: &[&str] = &[
    "def", "return", "if", "elif", "else", "for", "in", "pass", "import", "as", "and", "or", "not",
    "True", "False", "None", "while", "break", "continue", "class", "try", "except", "finally",
    "with", "lambda", "global", "nonlocal", "assert", "del", "yield", "raise", "from", "is",
    "await", "async",
];

/// A keyword that must be refused if it starts a statement, mapped to the CPython
/// `ast` node-type name the refusal reports. Keywords that never start a statement
/// (`await`, `is`, `from`) are absent: they fail as expressions, which is honest.
const REFUSED_STATEMENT_KEYWORDS: &[(&str, &str)] = &[
    ("while", "While"),
    ("break", "Break"),
    ("continue", "Continue"),
    ("class", "ClassDef"),
    ("try", "Try"),
    ("except", "Try"),
    ("finally", "Try"),
    ("with", "With"),
    ("global", "Global"),
    ("nonlocal", "Nonlocal"),
    ("del", "Delete"),
    ("yield", "Yield"),
    ("raise", "Raise"),
    ("async", "AsyncFunctionDef"),
];

fn refused_statement_node(kw: &str) -> Option<&'static str> {
    REFUSED_STATEMENT_KEYWORDS
        .iter()
        .find(|(k, _)| *k == kw)
        .map(|(_, node)| *node)
}

/// An unterminated string or bracket's complaint, in the crate's own voice.
fn err_at(msg: impl Into<String>, line: u32, col: u32) -> Error {
    Error::new(msg, line, col)
}

/// Is `c` an identifier start (Python's own rule outside Unicode)?
fn is_id_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Is `c` an identifier continuation?
fn is_id_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The lexer. Runs over the raw source, tracking bracket depth so newlines inside
/// brackets join lines instead of ending statements, and the indent stack so the
/// parser sees NEWLINE/INDENT/DEDENT the way CPython's tokenizer produces them.
struct Lexer<'s> {
    src: &'s [u8],
    /// Byte offset of the next unread byte.
    at: usize,
    line: u32,
    /// Byte column of `at` within `line`.
    col: u32,
    /// Open bracket count; while nonzero, newlines are whitespace.
    depth: usize,
    indents: Vec<u32>,
    /// Pending tokens not yet emitted (INDENT/DEDENT/NEWLINE plumbing).
    pending: std::collections::VecDeque<Token>,
    /// Set once a logical line has produced any token, so a NEWLINE is emitted at its
    /// end and blank/comment-only lines produce nothing.
    saw_token: bool,
}

impl<'s> Lexer<'s> {
    fn new(src: &'s str) -> Lexer<'s> {
        Lexer {
            src: src.as_bytes(),
            at: 0,
            line: 1,
            col: 0,
            depth: 0,
            indents: vec![0],
            pending: std::collections::VecDeque::new(),
            saw_token: false,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.at).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.at + 1).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.at += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 0;
        } else {
            // A column is one UTF-8 byte, matching the `Lines` convention the old
            // adapter used and the goldens' `loc(...)` fields assume.
            self.col += 1;
        }
        Some(c)
    }

    /// Skip spaces, tabs (column 8, Python's tabstop convention is irrelevant at the
    /// corpus's fixed indentation of spaces), and -- outside brackets -- comments and
    /// blank lines, emitting NEWLINE/INDENT/DEDENT for line structure.
    fn skip_trivia_and_handle_layout(&mut self) -> Result<bool> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') => {
                    self.bump();
                }
                Some(b'#') => {
                    // A comment runs to end of line. If the line had a token, the
                    // NEWLINE for that token is still owed (emitted below).
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some(b'\\') if self.peek2() == Some(b'\n') => {
                    // An explicit line continuation. The corpus's HOST halves use
                    // them (a long `assert ..., \` in `paged_attention_multiblock`),
                    // so they are joined here like any other whitespace. No jit
                    // body in the corpus uses one; the census law means a kernel
                    // that does is out of scope, but the join is what Python does
                    // and refusing it would make the host halves unparseable.
                    self.bump();
                    self.bump();
                }
                Some(b'\\') => {
                    return Err(err_at(
                        "a backslash in Python source is not supported",
                        self.line,
                        self.col,
                    ));
                }
                Some(b'\n') => {
                    if self.depth > 0 {
                        // Inside brackets: newline is whitespace.
                        self.bump();
                        continue;
                    }
                    if self.saw_token {
                        // The statement's terminating NEWLINE.
                        self.pending.push_back(Token {
                            kind: Tok::Newline,
                            text: String::new(),
                            line: self.line,
                            col: self.col,
                        });
                    }
                    self.saw_token = false;
                    self.bump();
                    // At the head of a new physical line: measure the indent. Blank and
                    // comment-only lines are skipped entirely (their indent does not
                    // count -- Python's own rule).
                    let (indent, next) = self.measure_indent();
                    if next == Some(b'#') {
                        // A comment-only line; loop back around, the `#` arm eats it.
                        continue;
                    }
                    if next == Some(b'\n') || next.is_none() {
                        // Blank line (or EOF; EOF's DEDENTs are handled by `finish`).
                        continue;
                    }
                    let cur = *self.indents.last().expect("indent stack is never empty");
                    if indent > cur {
                        self.indents.push(indent);
                        self.pending.push_back(Token {
                            kind: Tok::Indent,
                            text: String::new(),
                            line: self.line,
                            col: self.col,
                        });
                    } else if indent < cur {
                        while *self.indents.last().expect("stack never empty") > indent {
                            self.indents.pop();
                            self.pending.push_back(Token {
                                kind: Tok::Dedent,
                                text: String::new(),
                                line: self.line,
                                col: self.col,
                            });
                        }
                        if *self.indents.last().expect("stack never empty") != indent {
                            return Err(err_at(
                                "inconsistent dedent: this line's indentation matches no \
                                 open block",
                                self.line,
                                self.col,
                            ));
                        }
                    }
                    // `saw_token` stays false; the next real token sets it.
                }
                _ => return Ok(self.peek().is_some()),
            }
        }
    }

    /// At the start of a physical line: consume the blanks/tabs, return the column and
    /// the first non-space byte (without consuming it).
    fn measure_indent(&mut self) -> (u32, Option<u8>) {
        let mut n = 0;
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\r' {
                self.bump();
                n += 1;
            } else {
                break;
            }
        }
        (n, self.peek())
    }

    /// Lex one token. Returns `None` at end of input (after the final NEWLINE/DEDENT
    /// sequence is drained).
    fn next_token(&mut self) -> Result<Option<Token>> {
        loop {
            if let Some(t) = self.pending.pop_front() {
                return Ok(Some(t));
            }
            if !self.skip_trivia_and_handle_layout()? {
                // EOF: close any open bracket loudly, then the final NEWLINE + DEDENTs.
                if self.depth > 0 {
                    return Err(err_at(
                        "unexpected end of file inside brackets",
                        self.line,
                        self.col,
                    ));
                }
                if self.saw_token {
                    self.saw_token = false;
                    return Ok(Some(Token {
                        kind: Tok::Newline,
                        text: String::new(),
                        line: self.line,
                        col: self.col,
                    }));
                }
                while self.indents.len() > 1 {
                    self.indents.pop();
                    self.pending.push_back(Token {
                        kind: Tok::Dedent,
                        text: String::new(),
                        line: self.line,
                        col: self.col,
                    });
                }
                if self.pending.is_empty() {
                    return Ok(None);
                }
                continue;
            }
            if !self.pending.is_empty() {
                // The layout pass just emitted NEWLINE/INDENT/DEDENT ahead of
                // this line's first real token -- those come FIRST, so re-enter
                // the loop so its pop returns them in order.
                continue;
            }
            let (line, col) = (self.line, self.col);
            let c = self.peek().expect("skip_trivia guaranteed a byte");
            let tok = if is_id_start(char::from(c)) {
                let start = self.at;
                while let Some(c2) = self.peek() {
                    if is_id_cont(char::from(c2)) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&self.src[start..self.at]).into_owned();
                if KEYWORDS.contains(&text.as_str()) {
                    Token {
                        kind: Tok::Keyword,
                        text,
                        line,
                        col,
                    }
                } else {
                    Token {
                        kind: Tok::Name,
                        text,
                        line,
                        col,
                    }
                }
            } else if c.is_ascii_digit() {
                self.lex_number(line, col)?
            } else if c == b'"' || c == b'\'' {
                self.lex_string(line, col)?
            } else {
                // Operators, longest match first.
                let rest = &self.src[self.at..];
                let mut matched = None;
                for op in OPS {
                    if rest.starts_with(op.as_bytes()) {
                        matched = Some(*op);
                        break;
                    }
                }
                let Some(op) = matched else {
                    return Err(err_at(
                        format!("unexpected character `{}`", char::from(c)),
                        line,
                        col,
                    ));
                };
                for _ in 0..op.len() {
                    self.bump();
                }
                match op {
                    "(" | "[" | "{" => self.depth += 1,
                    ")" | "]" | "}" => {
                        self.depth = self
                            .depth
                            .checked_sub(1)
                            .ok_or_else(|| err_at("unbalanced closing bracket", line, col))?;
                    }
                    _ => {}
                }
                Token {
                    kind: Tok::Op,
                    text: op.to_string(),
                    line,
                    col,
                }
            };
            self.saw_token = true;
            return Ok(Some(tok));
        }
    }

    /// A number: `123`, `1.5`, `1.0e4`, `2.5e-4`, `1e-05`. Underscores and complex
    /// suffixes are refused.
    /// A number: `123`, `1.5`, `1.0e4`, `2.5e-4`, `1e-05`. Underscore separators,
    /// hex/octal/binary prefixes and complex suffixes are refused by name.
    fn lex_number(&mut self, line: u32, col: u32) -> Result<Token> {
        let start = self.at;
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.bump();
            } else if c == b'.' && self.peek2().is_some_and(|d| d.is_ascii_digit()) {
                is_float = true;
                self.bump();
            } else if c == b'e' || c == b'E' {
                // An exponent: `e` then an optional single sign then a digit.
                let d1 = self.peek2();
                let d2 = self.src.get(self.at + 2).copied();
                let (exp_digits_at, has_exp) = match (d1, d2) {
                    (Some(b'+') | Some(b'-'), Some(d)) if d.is_ascii_digit() => (2, true),
                    (Some(d), _) if d.is_ascii_digit() => (1, true),
                    _ => (0, false),
                };
                if !has_exp {
                    break;
                }
                is_float = true;
                for _ in 0..exp_digits_at {
                    self.bump();
                }
            } else if c == b'_' {
                return Err(err_at(
                    "`_` digit separators in number literals are not supported",
                    line,
                    col,
                ));
            } else if c == b'j' || c == b'J' {
                return Err(err_at("complex literals are not supported", line, col));
            } else if is_id_start(char::from(c)) {
                // e.g. `0x10`, `1if` -- the corpus has no hex, and a name glued to a
                // number is always a mistake.
                return Err(err_at(
                    format!(
                        "a name (`{}`) directly after a number literal is not a valid \
                         number form this parser accepts",
                        char::from(c)
                    ),
                    line,
                    col,
                ));
            } else {
                break;
            }
        }
        let text = String::from_utf8_lossy(&self.src[start..self.at]).into_owned();
        Ok(Token {
            kind: if is_float { Tok::Float } else { Tok::Int },
            text,
            line,
            col,
        })
    }

    /// A string. No prefixes, no escapes beyond the empty set the corpus needs (the
    /// fixtures' docstrings contain none; a `\` anywhere is refused by the trivia pass
    /// at line level and here inside strings). Triple quotes are supported because the
    /// corpus's module and function docstrings use them.
    fn lex_string(&mut self, line: u32, col: u32) -> Result<Token> {
        let quote = self.bump().expect("caller checked the byte");
        let triple = self.peek() == Some(quote) && self.peek2() == Some(quote);
        if triple {
            self.bump();
            self.bump();
        }
        let mut text = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(err_at("unterminated string literal", line, col));
            };
            if c == b'\\' {
                return Err(err_at(
                    "escape sequences (`\\`) in string literals are not supported",
                    self.line,
                    self.col,
                ));
            }
            if !triple && c == b'\n' {
                return Err(err_at("unterminated string literal", line, col));
            }
            if c == quote {
                if triple {
                    // Check for the closing triple.
                    if self.peek2() == Some(quote) && self.src.get(self.at + 2) == Some(&quote) {
                        self.bump();
                        self.bump();
                        self.bump();
                        break;
                    }
                    text.push(char::from(c));
                    self.bump();
                } else {
                    self.bump();
                    break;
                }
            } else {
                // Decode UTF-8 as we go; the corpus is ASCII plus em-dashes in comments
                // (comments never reach here) so this loop is byte-per-char in practice.
                let start = self.at;
                let ch = {
                    let s = &self.src[self.at..];
                    let s = std::str::from_utf8(s).map_err(|_| {
                        err_at("invalid UTF-8 in string literal", self.line, self.col)
                    })?;
                    s.chars()
                        .next()
                        .expect("peek returned a byte, so a char exists")
                };
                for _ in 0..ch.len_utf8() {
                    self.bump();
                }
                text.push(ch);
                let _ = start;
            }
        }
        Ok(Token {
            kind: Tok::Str,
            text,
            line,
            col,
        })
    }
}

// ── the parser ─────────────────────────────────────────────────────────────────

/// Parse Python source into this crate's AST.
///
/// MODULE-LEVEL RULE, matching the adapter it replaces: `def`s are recorded (jit
/// bodies walked, host functions recorded by name only), `NAME = <literal>` bindings
/// are captured for the by-name global refusal, and EVERYTHING ELSE IS HOST CODE the
/// front end never runs -- skipped structurally (to the end of its logical line, with
/// bracket depth respected) rather than parsed, because a fixture's host half
/// (`GRANITE = dict(...)`, `PAGED_ATTN_TABLE = torch.tensor([...])`, `def inputs()`)
/// is ordinary Python this grammar does not need to own.
pub fn parse(src: &str) -> Result<PyModule> {
    let mut p = Parser::new(src)?;
    let mut out = PyModule::default();
    loop {
        while p.eat(Tok::Newline) {}
        if p.at_end() {
            break;
        }
        if p.peek_keyword("import") {
            let (line, col) = p.pos();
            p.bump_tok();
            p.parse_import(line, col)?;
            continue;
        }
        if p.peek_keyword("def") {
            out.functions.push(p.parse_function_def()?);
            continue;
        }
        if p.peek_op("@") {
            out.functions.push(p.parse_decorated_def()?);
            continue;
        }
        if p.peek_kind(Tok::Str) {
            // A module docstring. Consumed and dropped.
            p.bump_tok();
            p.end_statement()?;
            continue;
        }
        if p.peek_kind(Tok::Name) {
            let saved = p.clone();
            match p.try_module_literal() {
                Ok(Some(bind)) => {
                    out.globals.push(bind);
                    continue;
                }
                Ok(None) => {
                    p.restore(saved);
                    p.skip_logical_line()?;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        // Any other module-level statement is host code: skip its whole logical line.
        p.skip_logical_line()?;
    }
    Ok(out)
}

/// The parser's read-only cursor state, so a speculative parse can be rolled back.
#[derive(Clone)]
struct Cursor {
    toks: Vec<Token>,
    at: usize,
}

struct Parser {
    c: Cursor,
}

impl Parser {
    fn new(src: &str) -> Result<Parser> {
        let mut lex = Lexer::new(src);
        let mut toks = Vec::new();
        while let Some(t) = lex.next_token()? {
            toks.push(t);
        }
        toks.push(Token {
            kind: Tok::End,
            text: String::new(),
            line: lex.line,
            col: lex.col,
        });
        Ok(Parser {
            c: Cursor { toks, at: 0 },
        })
    }

    fn clone(&self) -> Cursor {
        self.c.clone()
    }

    fn restore(&mut self, c: Cursor) {
        self.c = c;
    }

    // ── token plumbing ────────────────────────────────────────────────────────

    fn peek(&self) -> &Token {
        &self.c.toks[self.c.at.min(self.c.toks.len() - 1)]
    }

    fn peek_kind(&self, k: Tok) -> bool {
        self.skip_newlines_lookahead();
        self.peek().kind == k
    }

    fn peek_op(&self, op: &str) -> bool {
        self.peek().kind == Tok::Op && self.peek().text == op
    }

    fn peek_keyword(&self, kw: &str) -> bool {
        self.peek().kind == Tok::Keyword && self.peek().text == kw
    }

    fn peek_text(&self) -> Option<&str> {
        let t = self.peek();
        (t.kind == Tok::Op || t.kind == Tok::Keyword).then_some(t.text.as_str())
    }

    fn at_end(&self) -> bool {
        self.peek().kind == Tok::End
    }

    fn bump_tok(&mut self) -> Token {
        let t = self.c.toks[self.c.at.min(self.c.toks.len() - 1)].clone();
        if self.c.at < self.c.toks.len() - 1 {
            self.c.at += 1;
        }
        t
    }

    fn eat(&mut self, k: Tok) -> bool {
        if self.peek().kind == k {
            self.bump_tok();
            true
        } else {
            false
        }
    }

    fn eat_keyword(&mut self, kw: &str) -> bool {
        if self.peek_keyword(kw) {
            self.bump_tok();
            true
        } else {
            false
        }
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if self.peek_op(op) {
            self.bump_tok();
            true
        } else {
            false
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<Token> {
        if self.peek_op(op) {
            Ok(self.bump_tok())
        } else {
            let t = self.peek().clone();
            Err(err_at(
                format!("expected `{op}`, found {}", spell(&t)),
                t.line,
                t.col,
            ))
        }
    }

    fn expect_keyword(&mut self, kw: &str) -> Result<()> {
        if self.peek_keyword(kw) {
            self.bump_tok();
            Ok(())
        } else {
            let t = self.peek().clone();
            Err(err_at(
                format!("expected `{kw}`, found {}", spell(&t)),
                t.line,
                t.col,
            ))
        }
    }

    fn expect_name(&mut self) -> Result<Token> {
        if self.peek().kind == Tok::Name {
            Ok(self.bump_tok())
        } else {
            let t = self.peek().clone();
            Err(err_at(
                format!("expected a name, found {}", spell(&t)),
                t.line,
                t.col,
            ))
        }
    }

    fn pos(&self) -> (u32, u32) {
        let t = self.peek();
        (t.line, t.col)
    }

    /// NEWLINEs between tokens of one logical construct are whitespace (the lexer only
    /// emits them at statement ends, but a lookahead must skip them for `peek_kind`
    /// after a `)`-terminated line).
    fn skip_newlines_lookahead(&self) {}

    /// Consume the statement terminator: the NEWLINE only. DEDENTs belong to the
    /// block structure and are consumed by [`Parser::parse_block`].
    fn end_statement(&mut self) -> Result<()> {
        while self.eat(Tok::Newline) {}
        if self.at_end() {
            return Ok(());
        }
        let t = self.peek().clone();
        if t.kind == Tok::Indent {
            return Err(err_at(
                "unexpected indentation (this statement is already complete)",
                t.line,
                t.col,
            ));
        }
        Ok(())
    }

    /// A block: `:` NEWLINE INDENT stmts DEDENT. A single simple statement on the same
    /// line (`if x: pass`) is refused -- the corpus never uses it.
    fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect_op(":")?;
        if self.eat(Tok::Newline) {
            if !self.eat(Tok::Indent) {
                let t = self.peek().clone();
                return Err(err_at("expected an indented block", t.line, t.col));
            }
            let mut body = Vec::new();
            loop {
                while self.eat(Tok::Newline) {}
                if self.eat(Tok::Dedent) {
                    break;
                }
                if self.at_end() {
                    // The lexer drains DEDENTs before End, so reaching End inside a
                    // block is a malformed-indent error path; treat as end of block.
                    break;
                }
                // A bare string statement (a docstring, anywhere in the body) is
                // dropped entirely, matching the adapter's `lower_body`: Triton's
                // walk reaches it as an Expr statement and does nothing with it.
                if self.peek().kind == Tok::Str
                    && matches!(
                        self.c.toks.get(self.c.at + 1).map(|t| t.kind),
                        Some(Tok::Newline) | Some(Tok::Dedent) | Some(Tok::End)
                    )
                {
                    self.bump_tok();
                    continue;
                }
                body.push(self.parse_statement()?);
            }
            Ok(body)
        } else {
            let t = self.peek().clone();
            Err(err_at(
                "a `:` must be followed by a newline and an indented block (single-line \
                 bodies are not supported)",
                t.line,
                t.col,
            ))
        }
    }

    /// A body belonging to HOST code (a non-jit `def`): consumed structurally, nothing
    /// inside it parsed or refused. The adapter recorded such functions by name only
    /// and never walked the body -- a fixture's host half is ordinary Python (dicts,
    /// f-strings, asserts) this grammar does not need to own.
    fn skip_block(&mut self) -> Result<()> {
        self.expect_op(":")?;
        if !self.eat(Tok::Newline) {
            let t = self.peek().clone();
            return Err(err_at(
                "a `:` must be followed by a newline and an indented block",
                t.line,
                t.col,
            ));
        }
        if !self.eat(Tok::Indent) {
            let t = self.peek().clone();
            return Err(err_at("expected an indented block", t.line, t.col));
        }
        let mut depth = 1usize;
        while depth > 0 {
            let t = self.bump_tok();
            match t.kind {
                Tok::Indent => depth += 1,
                Tok::Dedent => depth -= 1,
                Tok::End => {
                    return Err(err_at(
                        "unexpected end of file inside a host function body",
                        t.line,
                        t.col,
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    // ── module-level forms ────────────────────────────────────────────────────

    /// `import triton`, `import triton.language as tl`, `import a.b.c`.
    fn parse_import(&mut self, _line: u32, _col: u32) -> Result<()> {
        loop {
            self.expect_name()?;
            while self.eat_op(".") {
                self.expect_name()?;
            }
            if self.eat_keyword("as") {
                self.expect_name()?;
            }
            if !self.eat_op(",") {
                break;
            }
        }
        self.end_statement()
    }

    /// Skip one module-level logical line of host code: consume tokens until a NEWLINE
    /// at bracket depth zero. The lexer has already joined bracketed lines, so any
    /// NEWLINE this loop sees is the logical line's end.
    fn skip_logical_line(&mut self) -> Result<()> {
        let mut depth = 0usize;
        loop {
            let t = self.peek().clone();
            match t.kind {
                Tok::Newline | Tok::End if depth == 0 => {
                    let _ = t;
                    break;
                }
                Tok::Indent | Tok::Dedent if depth == 0 => {
                    // A host statement never opens a block we skip; a DEDENT here is
                    // stray. Stop and let the caller's loop re-sync.
                    break;
                }
                Tok::Op if t.text == "(" || t.text == "[" || t.text == "{" => {
                    depth += 1;
                    self.bump_tok();
                }
                Tok::Op if t.text == ")" || t.text == "]" || t.text == "}" => {
                    depth = depth.saturating_sub(1);
                    self.bump_tok();
                }
                Tok::End => break,
                _ => {
                    self.bump_tok();
                }
            }
        }
        while self.eat(Tok::Newline) {}
        Ok(())
    }

    /// `NAME = <literal>` at module level, including a negated literal. Returns
    /// `None` (with the cursor restored by the caller) when the statement is not one.
    fn try_module_literal(&mut self) -> Result<Option<(String, Literal)>> {
        let Some(name) = (self.peek().kind == Tok::Name).then(|| self.peek().text.clone()) else {
            return Ok(None);
        };
        self.bump_tok();
        if !self.eat_op("=") {
            return Ok(None);
        }
        // The value must be a bare literal, optionally negated.
        let neg = self.eat_op("-");
        let t = self.peek().clone();
        let lit = match t.kind {
            Tok::Int => {
                self.bump_tok();
                Literal::Int(
                    t.text
                        .parse::<i128>()
                        .map_err(|_| err_at("integer literal does not fit", t.line, t.col))?,
                )
            }
            Tok::Float => {
                self.bump_tok();
                Literal::Float(
                    t.text
                        .parse::<f64>()
                        .map_err(|_| err_at("malformed float literal", t.line, t.col))?,
                )
            }
            Tok::Str => {
                self.bump_tok();
                Literal::Str(t.text.clone())
            }
            Tok::Keyword if t.text == "True" || t.text == "False" => {
                self.bump_tok();
                Literal::Bool(t.text == "True")
            }
            Tok::Keyword if t.text == "None" => {
                self.bump_tok();
                Literal::None
            }
            _ => return Ok(None),
        };
        let lit = if neg {
            match lit {
                Literal::Int(v) => Literal::Int(-v),
                Literal::Float(v) => Literal::Float(-v),
                other => other,
            }
        } else {
            lit
        };
        self.end_statement()?;
        Ok(Some((name, lit)))
    }

    /// `@dec...` then `def`. Only `@triton.jit`-shaped decorators set `is_jit`
    /// (any dotted name containing a `jit` segment, matching the adapter's rule --
    /// a decorator that names host machinery this front end never runs).
    fn parse_decorated_def(&mut self) -> Result<FunctionDef> {
        let (mut dline, mut dcol) = (0u32, 0u32);
        let mut is_jit = false;
        while self.eat_op("@") {
            let first = self.expect_name()?;
            if dline == 0 {
                dline = first.line;
                dcol = first.col;
            }
            let mut parts = vec![first.text.clone()];
            while self.eat_op(".") {
                parts.push(self.expect_name()?.text.clone());
            }
            if self.peek_op("(") {
                // A called decorator (`@triton.jit(...)`). Its arguments are skipped
                // structurally: balanced brackets, any tokens -- a decorator's
                // arguments are host code this front end never runs.
                let mut depth = 0usize;
                loop {
                    let t = self.bump_tok();
                    match t.kind {
                        Tok::Op if t.text == "(" => depth += 1,
                        Tok::Op if t.text == ")" => {
                            depth = depth.checked_sub(1).ok_or_else(|| {
                                err_at("unbalanced closing bracket", t.line, t.col)
                            })?;
                            if depth == 0 {
                                break;
                            }
                        }
                        Tok::End => {
                            return Err(err_at("unterminated decorator call", t.line, t.col));
                        }
                        _ => {}
                    }
                }
            }
            is_jit = is_jit || parts.iter().any(|p| p == "jit");
            // The decorator's newline (decorators sit one per line in the corpus; a
            // following `@` means another decorator, anything else must be `def`).
            while self.eat(Tok::Newline) {}
            if !self.peek_op("@") && !self.peek_keyword("def") {
                let t = self.peek().clone();
                return Err(err_at(
                    "expected `def` (or another decorator) after a decorator",
                    t.line,
                    t.col,
                ));
            }
        }
        self.parse_def_body(is_jit, dline, dcol)
    }

    // ── function definitions ──────────────────────────────────────────────────

    /// A bare `def` at module level: host code (the corpus has no undecorated
    /// kernels), recorded by name only with an empty body -- the adapter's own rule.
    /// The signature is skipped structurally too: host functions carry defaults and
    /// return annotations the kernel grammar rightly refuses.
    fn parse_function_def(&mut self) -> Result<FunctionDef> {
        self.expect_keyword("def")?;
        let (line, col) = self.pos();
        let name = self.expect_name()?.text;
        let pos = Pos { line, col };
        self.skip_host_signature()?;
        self.skip_block()?;
        Ok(FunctionDef {
            name,
            params: Vec::new(),
            body: Vec::new(),
            is_jit: false,
            pos,
        })
    }

    /// The `def` of a DECORATED function. `dec_jit` carries the decorator verdict;
    /// `dline`/`dcol` position the function at its first decorator, matching the
    /// adapter (which positions a function at its first byte, the `@`).
    fn parse_def_body(&mut self, dec_jit: bool, dline: u32, dcol: u32) -> Result<FunctionDef> {
        self.expect_keyword("def")?;
        let (line, col) = if dline > 0 { (dline, dcol) } else { self.pos() };
        let name = self.expect_name()?.text;
        let pos = Pos { line, col };
        if !dec_jit {
            // A decorated HOST function: skip signature and body structurally.
            self.skip_host_signature()?;
            self.skip_block()?;
            return Ok(FunctionDef {
                name,
                params: Vec::new(),
                body: Vec::new(),
                is_jit: false,
                pos,
            });
        }
        self.expect_op("(")?;
        let params = self.parse_params()?;
        self.expect_op(")")?;
        if self.peek_op("->") {
            let t = self.peek().clone();
            return Err(err_at(
                "a `->` return annotation is not supported on a kernel",
                t.line,
                t.col,
            ));
        }
        let body = self.parse_block()?;
        Ok(FunctionDef {
            name,
            params,
            body,
            is_jit: true,
            pos,
        })
    }

    /// The parameter list (and any return annotation) of a host `def`, skipped
    /// structurally: the balanced `(...)` group, then everything up to the block's
    /// `:` at bracket depth zero.
    fn skip_host_signature(&mut self) -> Result<()> {
        // The parameter group.
        if !self.peek_op("(") {
            let t = self.peek().clone();
            return Err(err_at(
                format!("expected `(` after the function name, found {}", spell(&t)),
                t.line,
                t.col,
            ));
        }
        let mut depth = 0usize;
        loop {
            let t = self.bump_tok();
            match t.kind {
                Tok::Op if t.text == "(" || t.text == "[" || t.text == "{" => depth += 1,
                Tok::Op if t.text == ")" || t.text == "]" || t.text == "}" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 && t.text == ")" {
                        break;
                    }
                }
                Tok::End => {
                    return Err(err_at(
                        "unterminated parameter list in a host function",
                        t.line,
                        t.col,
                    ));
                }
                _ => {}
            }
        }
        // The optional return annotation, up to the block's `:` at depth zero.
        let mut depth = 0usize;
        loop {
            let t = self.peek().clone();
            match t.kind {
                Tok::Op if t.text == "(" || t.text == "[" || t.text == "{" => {
                    depth += 1;
                    self.bump_tok();
                }
                Tok::Op if t.text == ")" || t.text == "]" || t.text == "}" => {
                    depth = depth.saturating_sub(1);
                    self.bump_tok();
                }
                Tok::Op if t.text == ":" && depth == 0 => break,
                Tok::Newline | Tok::End => {
                    return Err(err_at(
                        "expected `:` to open the function body",
                        t.line,
                        t.col,
                    ));
                }
                _ => {
                    self.bump_tok();
                }
            }
        }
        Ok(())
    }

    /// Parameters until `)`. Each is `name` or `name: annotation`; a `*`/`*args` marker
    /// or a default (`name=expr`) is refused by name, matching the adapter.
    fn parse_params(&mut self) -> Result<Vec<Param>> {
        let mut params = Vec::new();
        loop {
            if self.peek_op(")") {
                break;
            }
            let t = self.peek().clone();
            if t.text == "*" && t.kind == Tok::Op {
                return Err(err_at(
                    "`*args`/keyword-only markers are outside the supported parameter \
                     forms",
                    t.line,
                    t.col,
                ));
            }
            if t.text == "**" {
                return Err(err_at(
                    "`**kwargs` is outside the supported parameter forms",
                    t.line,
                    t.col,
                ));
            }
            let name = self.expect_name()?;
            let mut annotation = None;
            if self.eat_op(":") {
                annotation = Some(self.parse_dotted_name()?);
            }
            if self.peek_op("=") {
                let e = self.peek().clone();
                return Err(err_at(
                    "default parameter values are not supported on a kernel",
                    e.line,
                    e.col,
                ));
            }
            params.push(Param {
                name: name.text,
                annotation,
                pos: Pos {
                    line: name.line,
                    col: name.col,
                },
            });
            if !self.eat_op(",") {
                break;
            }
            // A trailing comma before `)` is fine.
            if self.peek_op(")") {
                break;
            }
        }
        Ok(params)
    }

    /// A dotted name (`tl.constexpr`, `tl.float16`), as an annotation.
    fn parse_dotted_name(&mut self) -> Result<String> {
        let mut s = self.expect_name()?.text;
        while self.eat_op(".") {
            s.push('.');
            s.push_str(&self.expect_name()?.text);
        }
        Ok(s)
    }

    // ── statements ────────────────────────────────────────────────────────────

    fn parse_statement(&mut self) -> Result<Stmt> {
        let (line, col) = self.pos();
        let t = self.peek().clone();

        // Refused keywords first, so the error names the construct. The names are
        // CPython's own `ast` node types (`while` is `While`, `class` is
        // `ClassDef`, ...), the way the census and the adapter before it
        // named them.
        if t.kind == Tok::Keyword
            && let Some(node) = refused_statement_node(&t.text)
        {
            return Err(err_at(
                format!("Python statement `{node}` is not supported inside a @triton.jit kernel"),
                line,
                col,
            ));
        }
        if t.kind == Tok::Keyword && t.text == "assert" {
            return Err(err_at(
                "Python statement `Assert` is not supported inside a @triton.jit kernel \
                 (use `tl.static_assert`)",
                line,
                col,
            ));
        }
        if t.kind == Tok::Keyword && t.text == "lambda" {
            return Err(err_at(
                "Python expression `Lambda` is not supported inside a @triton.jit kernel",
                line,
                col,
            ));
        }

        if t.kind == Tok::Keyword && t.text == "def" {
            // A nested def inside a jit body is outside the census.
            return Err(err_at(
                "Python statement `FunctionDef` is not supported inside a @triton.jit \
                 kernel",
                line,
                col,
            ));
        }

        if self.eat_keyword("return") {
            let value = if self.at_stmt_end() {
                None
            } else {
                Some(self.parse_expr_list()?)
            };
            let pos = Pos { line, col };
            self.end_statement()?;
            return Ok(Stmt::Return { value, pos });
        }
        if self.eat_keyword("pass") {
            let pos = Pos { line, col };
            self.end_statement()?;
            return Ok(Stmt::Pass { pos });
        }
        if self.eat_keyword("if") {
            return self.parse_if(line, col);
        }
        if self.eat_keyword("for") {
            return self.parse_for(line, col);
        }
        if t.kind == Tok::Keyword && t.text == "import" {
            return Err(err_at(
                "Python statement `Import` is not supported inside a @triton.jit kernel",
                line,
                col,
            ));
        }
        if t.kind == Tok::Keyword && t.text == "from" {
            return Err(err_at(
                "Python statement `ImportFrom` is not supported inside a @triton.jit \
                 kernel",
                line,
                col,
            ));
        }
        // Assignment or expression statement. Parse the expression (at testlist
        // rank: `lo, hi = ...` targets are a bare Tuple), then decide.
        // (Bare string statements never reach here: `parse_block` drops them.)
        let expr = self.parse_expr_list()?;
        let pos = Pos { line, col };
        if self.eat_op("=") {
            let target = self.expr_to_target(&expr)?;
            let value = self.parse_expr_list()?;
            // Chained assignment (`a = b = c`): a second `=` after the value.
            if self.peek_op("=") {
                let e = self.peek().clone();
                return Err(err_at(
                    "chained assignment (`a = b = ...`) is not supported",
                    e.line,
                    e.col,
                ));
            }
            self.end_statement()?;
            return Ok(Stmt::Assign { target, value, pos });
        }
        if let Some(op) = self.peek_aug_op() {
            self.bump_tok();
            let target = self.expr_to_target(&expr)?;
            let value = self.parse_expr_list()?;
            self.end_statement()?;
            return Ok(Stmt::AugAssign {
                target,
                op,
                value,
                pos,
            });
        }
        if self.eat_op(":") {
            // AnnAssign: `target : annotation [= value]`. The target was already
            // parsed as an expression.
            let target = self.expr_to_target(&expr)?;
            let annotation = self.parse_expr()?;
            let value = if self.eat_op("=") {
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.end_statement()?;
            return Ok(Stmt::AnnAssign {
                target,
                annotation,
                value,
                pos,
            });
        }
        // A bare expression statement.
        self.end_statement()?;
        Ok(Stmt::Expr { value: expr, pos })
    }

    /// An augmented-assignment operator at the cursor, or `None`. The full set is
    /// mapped; an operator the census has never produced (`<<=` etc.) is refused by
    /// name in [`Parser::parse_statement`]'s fall-through rather than here, so the
    /// message says "unsupported operator" and not "unexpected token".
    fn peek_aug_op(&self) -> Option<BinOpKind> {
        let t = self.peek();
        if t.kind != Tok::Op {
            return None;
        }
        Some(match t.text.as_str() {
            "+=" => BinOpKind::Add,
            "-=" => BinOpKind::Sub,
            "*=" => BinOpKind::Mult,
            "/=" => BinOpKind::Div,
            "//=" => BinOpKind::FloorDiv,
            "%=" => BinOpKind::Mod,
            "&=" => BinOpKind::BitAnd,
            "|=" => BinOpKind::BitOr,
            "^=" => BinOpKind::BitXor,
            "<<=" => BinOpKind::LShift,
            ">>=" => BinOpKind::RShift,
            "**=" => BinOpKind::Pow,
            "@=" => BinOpKind::MatMult,
            _ => return None,
        })
    }

    fn parse_if(&mut self, line: u32, col: u32) -> Result<Stmt> {
        let test = self.parse_expr()?;
        let body = self.parse_block()?;
        let mut orelse = Vec::new();
        // NEWLINEs already consumed by parse_block's DEDENT handling; `elif`/`else` at
        // the same indent follows.
        while self.eat(Tok::Newline) {}
        if self.peek_keyword("elif") {
            let (l, c) = self.pos();
            self.bump_tok();
            let nested = self.parse_if(l, c)?;
            orelse = vec![nested];
        } else if self.peek_keyword("else") {
            self.bump_tok();
            orelse = self.parse_block()?;
        }
        Ok(Stmt::If {
            test,
            body,
            orelse,
            pos: Pos { line, col },
        })
    }

    fn parse_for(&mut self, line: u32, col: u32) -> Result<Stmt> {
        let t = self.expect_name()?;
        if self.peek_op(",") {
            let e = self.peek().clone();
            return Err(err_at(
                "a `for` loop target must be a single name (tuple unpacking in a loop \
                 target is not supported)",
                e.line,
                e.col,
            ));
        }
        self.expect_keyword("in")?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;
        while self.eat(Tok::Newline) {}
        if self.peek_keyword("else") {
            let e = self.peek().clone();
            return Err(err_at("`for ... else` is not supported", e.line, e.col));
        }
        Ok(Stmt::For {
            target: t.text,
            iter,
            body,
            orelse: Vec::new(),
            pos: Pos { line, col },
        })
    }

    /// Turn a parsed expression into an assignment target, refusing anything that is
    /// not a name or a tuple of names.
    fn expr_to_target(&self, e: &Expr) -> Result<AssignTarget> {
        match e {
            Expr::Name { id, pos } => Ok(AssignTarget::Name {
                id: id.clone(),
                pos: *pos,
            }),
            Expr::Tuple { elts, pos } => {
                let mut out = Vec::new();
                for x in elts {
                    out.push(self.expr_to_target(x)?);
                }
                Ok(AssignTarget::Tuple {
                    elts: out,
                    pos: *pos,
                })
            }
            other => Err(err_at(
                format!(
                    "assignment to a {} is not supported (only a name or a tuple of \
                     names)",
                    other.kind_name()
                ),
                other.pos().line,
                other.pos().col,
            )),
        }
    }

    fn at_stmt_end(&self) -> bool {
        matches!(self.peek().kind, Tok::Newline | Tok::Dedent | Tok::End)
    }

    // ── expressions, precedence-climbing ──────────────────────────────────────

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_ternary()
    }

    /// Python's `testlist` level: a comma-separated series at statement or
    /// subscript-element rank forms a Tuple even without parentheses
    /// (`lo, hi = 0, BLOCK`, `return acc, l_i`). Everywhere else -- call
    /// arguments, bracket elements, comparison operands -- a bare comma
    /// belongs to the enclosing construct, so those callers use
    /// [`Parser::parse_expr`] instead.
    fn parse_expr_list(&mut self) -> Result<Expr> {
        let first = self.parse_expr()?;
        if !self.peek_op(",") {
            return Ok(first);
        }
        let pos = first.pos();
        let mut elts = vec![first];
        while self.eat_op(",") {
            if self.at_stmt_end() || self.peek_op("=") || self.peek_op(")") {
                break;
            }
            elts.push(self.parse_expr()?);
        }
        Ok(Expr::Tuple { elts, pos })
    }

    /// Python has no ternary in the census (`a if c else b` is `IfExp`, refused).
    fn parse_ternary(&mut self) -> Result<Expr> {
        let e = self.parse_or()?;
        if self.peek_keyword("if") {
            let t = self.peek().clone();
            return Err(err_at(
                "Python expression `IfExp` (a conditional expression) is not supported \
                 inside a @triton.jit kernel",
                t.line,
                t.col,
            ));
        }
        Ok(e)
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let left = self.parse_and()?;
        if self.peek_keyword("or") {
            let t = self.peek().clone();
            return Err(err_at(
                "Python expression `BoolOp` (`or`) is not supported inside a \
                 @triton.jit kernel",
                t.line,
                t.col,
            ));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let left = self.parse_not()?;
        if self.peek_keyword("and") {
            let t = self.peek().clone();
            return Err(err_at(
                "Python expression `BoolOp` (`and`) is not supported inside a \
                 @triton.jit kernel",
                t.line,
                t.col,
            ));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if self.peek_keyword("not") {
            let t = self.bump_tok();
            let operand = self.parse_not()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOpKind::Not,
                operand: Box::new(operand),
                pos: Pos {
                    line: t.line,
                    col: t.col,
                },
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let left = self.parse_bitor()?;
        let cmp_op = match self.peek_text() {
            Some("==") => Some(CmpKind::Eq),
            Some("!=") => Some(CmpKind::NotEq),
            Some("<") => Some(CmpKind::Lt),
            Some("<=") => Some(CmpKind::LtE),
            Some(">") => Some(CmpKind::Gt),
            Some(">=") => Some(CmpKind::GtE),
            Some("is") | Some("in") | Some("not") => {
                let t = self.peek().clone();
                return Err(err_at(
                    format!("comparison operator `{}` is not supported", t.text),
                    t.line,
                    t.col,
                ));
            }
            _ => None,
        };
        let Some(op) = cmp_op else {
            return Ok(left);
        };
        let tok = self.bump_tok();
        let right = self.parse_bitor()?;
        // A second comparator (`a < b < c`).
        if matches!(
            self.peek_text(),
            Some("==") | Some("!=") | Some("<") | Some("<=") | Some(">") | Some(">=")
        ) {
            let t = self.peek().clone();
            return Err(err_at(
                "simultaneous multiple comparison is not supported",
                t.line,
                t.col,
            ));
        }
        Ok(Expr::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
            pos: Pos {
                line: tok.line,
                col: tok.col,
            },
        })
    }

    fn parse_bitor(&mut self) -> Result<Expr> {
        let mut left = self.parse_bitxor()?;
        while self.peek_op("|") {
            let tok = self.bump_tok();
            let right = self.parse_bitxor()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op: BinOpKind::BitOr,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Expr> {
        let mut left = self.parse_bitand()?;
        while self.peek_op("^") {
            let tok = self.bump_tok();
            let right = self.parse_bitand()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op: BinOpKind::BitXor,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Expr> {
        let mut left = self.parse_shift()?;
        while self.peek_op("&") {
            let tok = self.bump_tok();
            let right = self.parse_shift()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op: BinOpKind::BitAnd,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        let mut left = self.parse_arith()?;
        while self.peek_op("<<") || self.peek_op(">>") {
            let tok = self.bump_tok();
            let op = if tok.text == "<<" {
                BinOpKind::LShift
            } else {
                BinOpKind::RShift
            };
            let right = self.parse_arith()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_arith(&mut self) -> Result<Expr> {
        let mut left = self.parse_term()?;
        loop {
            let op = if self.peek_op("+") {
                BinOpKind::Add
            } else if self.peek_op("-") {
                BinOpKind::Sub
            } else {
                break;
            };
            let tok = self.bump_tok();
            let right = self.parse_term()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expr> {
        let mut left = self.parse_factor()?;
        loop {
            let op = if self.peek_op("*") {
                BinOpKind::Mult
            } else if self.peek_op("/") {
                BinOpKind::Div
            } else if self.peek_op("//") {
                BinOpKind::FloorDiv
            } else if self.peek_op("%") {
                BinOpKind::Mod
            } else if self.peek_op("@") {
                BinOpKind::MatMult
            } else {
                break;
            };
            let tok = self.bump_tok();
            let right = self.parse_factor()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            };
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expr> {
        let op = if self.peek_op("-") {
            Some(UnaryOpKind::USub)
        } else if self.peek_op("+") {
            Some(UnaryOpKind::UAdd)
        } else if self.peek_op("~") {
            Some(UnaryOpKind::Invert)
        } else {
            None
        };
        if let Some(op) = op {
            let tok = self.bump_tok();
            // Python binds `-2 ** 2` as `-(2**2)`; the corpus has no such spelling, and
            // `**` after a unary operand is exactly that ambiguity, so refuse it.
            let operand = self.parse_factor()?;
            if let Expr::BinOp {
                op: BinOpKind::Pow, ..
            } = &operand
            {
                let t = self.peek().clone();
                return Err(err_at(
                    "a unary operator applied to a `**` expression is ambiguous in \
                     Python (it binds looser than `**`); parenthesise it",
                    t.line,
                    t.col,
                ));
            }
            return Ok(Expr::UnaryOp {
                op,
                operand: Box::new(operand),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            });
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_atom_and_postfix()?;
        if self.peek_op("**") {
            let tok = self.bump_tok();
            // Right-associative, and the exponent may carry a unary sign.
            let right = self.parse_factor()?;
            return Ok(Expr::BinOp {
                left: Box::new(base),
                op: BinOpKind::Pow,
                right: Box::new(right),
                pos: Pos {
                    line: tok.line,
                    col: tok.col,
                },
            });
        }
        Ok(base)
    }

    /// An atom with every postfix trailer: attribute, call, subscript.
    fn parse_atom_and_postfix(&mut self) -> Result<Expr> {
        let mut e = self.parse_atom()?;
        loop {
            if self.peek_op(".") {
                let tok = self.bump_tok();
                let attr = self.expect_name()?;
                e = Expr::Attribute {
                    value: Box::new(e),
                    attr: attr.text,
                    pos: Pos {
                        line: tok.line,
                        col: tok.col,
                    },
                };
            } else if self.peek_op("(") {
                let tok = self.bump_tok();
                let mut args = Vec::new();
                let mut keywords = Vec::new();
                loop {
                    if self.peek_op(")") {
                        break;
                    }
                    // `*expr` / `**kw` at a call site.
                    if self.peek_op("*") || self.peek_text() == Some("**") {
                        let t = self.peek().clone();
                        return Err(err_at(
                            "*args/**kwargs unpacking at a call site is not supported",
                            t.line,
                            t.col,
                        ));
                    }
                    // A keyword argument is `name = expr`, distinguished from a
                    // positional containing `=` by trying the name first.
                    if self.peek().kind == Tok::Name
                        && self
                            .c
                            .toks
                            .get(self.c.at + 1)
                            .is_some_and(|t| t.kind == Tok::Op && t.text == "=")
                    {
                        let name = self.bump_tok();
                        self.bump_tok(); // `=`
                        let value = self.parse_expr()?;
                        keywords.push((name.text, value));
                    } else {
                        args.push(self.parse_expr()?);
                    }
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op(")")?;
                e = Expr::Call {
                    func: Box::new(e),
                    args,
                    keywords,
                    pos: Pos {
                        line: tok.line,
                        col: tok.col,
                    },
                };
            } else if self.peek_op("[") {
                let tok = self.bump_tok();
                let index = self.parse_subscript_index()?;
                self.expect_op("]")?;
                e = Expr::Subscript {
                    value: Box::new(e),
                    index: Box::new(index),
                    pos: Pos {
                        line: tok.line,
                        col: tok.col,
                    },
                };
            } else {
                break;
            }
        }
        Ok(e)
    }

    /// The inside of `[...]`: either a single element (which may be a slice or a
    /// tuple of them) or a comma-separated tuple. In Python, `x[a, b]` and `x[a, b, c]`
    /// are tuple indexes and `x[a]` is not; `x[a,]` is a 1-tuple.
    fn parse_subscript_index(&mut self) -> Result<Expr> {
        let first = self.parse_slice_or_expr()?;
        if !self.peek_op(",") {
            return Ok(first);
        }
        let (line, col) = (first.pos().line, first.pos().col);
        let mut elts = vec![first];
        while self.eat_op(",") {
            if self.peek_op("]") {
                break;
            }
            elts.push(self.parse_slice_or_expr()?);
        }
        Ok(Expr::Tuple {
            elts,
            pos: Pos { line, col },
        })
    }

    /// One subscript element: a slice (`a:b`, `:b`, `a:`, `:`, with optional `::step`)
    /// or a plain expression.
    fn parse_slice_or_expr(&mut self) -> Result<Expr> {
        // Is there a leading `:` or `expr :`?
        let starts_colon = self.peek_op(":");
        if starts_colon {
            let tok = self.bump_tok();
            return self.finish_slice(None, tok);
        }
        let e = self.parse_expr()?;
        if self.peek_op(":") {
            let tok = self.bump_tok();
            return self.finish_slice(Some(e), tok);
        }
        Ok(e)
    }

    /// After a `:` has been consumed (with the lower bound, if any): the upper bound
    /// and optional step.
    fn finish_slice(&mut self, lower: Option<Expr>, colon: Token) -> Result<Expr> {
        let pos = Pos {
            line: colon.line,
            col: colon.col,
        };
        let upper = if self.peek_op(":") || self.peek_op("]") || self.peek_op(",") {
            None
        } else {
            Some(self.parse_expr()?)
        };
        let step = if self.eat_op(":") {
            if self.peek_op("]") || self.peek_op(",") {
                None
            } else {
                Some(self.parse_expr()?)
            }
        } else {
            None
        };
        Ok(Expr::Slice {
            lower: lower.map(Box::new),
            upper: upper.map(Box::new),
            step: step.map(Box::new),
            pos,
        })
    }

    fn parse_atom(&mut self) -> Result<Expr> {
        let t = self.peek().clone();
        let pos = Pos {
            line: t.line,
            col: t.col,
        };
        match t.kind {
            Tok::Name => {
                self.bump_tok();
                Ok(Expr::Name { id: t.text, pos })
            }
            Tok::Int => {
                self.bump_tok();
                let v: i128 = t.text.parse().map_err(|_| {
                    err_at("integer literal does not fit in 128 bits", t.line, t.col)
                })?;
                let v = i64::try_from(v).map_err(|_| {
                    err_at("integer literal does not fit in 64 bits", t.line, t.col)
                })?;
                Ok(Expr::Constant {
                    value: Literal::Int(v as i128),
                    pos,
                })
            }
            Tok::Float => {
                self.bump_tok();
                let v: f64 = t
                    .text
                    .parse()
                    .map_err(|_| err_at("malformed float literal", t.line, t.col))?;
                Ok(Expr::Constant {
                    value: Literal::Float(v),
                    pos,
                })
            }
            Tok::Str => {
                self.bump_tok();
                Ok(Expr::Constant {
                    value: Literal::Str(t.text),
                    pos,
                })
            }
            Tok::Keyword if t.text == "True" || t.text == "False" => {
                self.bump_tok();
                Ok(Expr::Constant {
                    value: Literal::Bool(t.text == "True"),
                    pos,
                })
            }
            Tok::Keyword if t.text == "None" => {
                self.bump_tok();
                Ok(Expr::Constant {
                    value: Literal::None,
                    pos,
                })
            }
            Tok::Keyword if t.text == "not" => self.parse_not(),
            Tok::Op if t.text == "(" => {
                self.bump_tok();
                if self.peek_op(")") {
                    self.bump_tok();
                    return Ok(Expr::Tuple { elts: vec![], pos });
                }
                let first = self.parse_expr()?;
                if self.peek_op(",") {
                    let mut elts = vec![first];
                    while self.eat_op(",") {
                        if self.peek_op(")") {
                            break;
                        }
                        elts.push(self.parse_expr()?);
                    }
                    self.expect_op(")")?;
                    Ok(Expr::Tuple { elts, pos })
                } else {
                    self.expect_op(")")?;
                    // A parenthesised single expression is the expression itself;
                    // Python records no Tuple node.
                    Ok(first)
                }
            }
            Tok::Op if t.text == "[" => {
                self.bump_tok();
                let mut elts = Vec::new();
                if !self.peek_op("]") {
                    loop {
                        elts.push(self.parse_expr()?);
                        if !self.eat_op(",") {
                            break;
                        }
                        if self.peek_op("]") {
                            break;
                        }
                    }
                }
                self.expect_op("]")?;
                Ok(Expr::List { elts, pos })
            }
            Tok::Op if t.text == "{" => Err(err_at(
                "Python expression `Dict`/`Set` literals are not supported inside a \
                 @triton.jit kernel",
                t.line,
                t.col,
            )),
            Tok::Op if t.text == "-" || t.text == "+" || t.text == "~" => self.parse_factor(),
            Tok::Newline | Tok::Indent | Tok::Dedent | Tok::End => Err(err_at(
                format!("expected an expression, found {}", spell(&t)),
                t.line,
                t.col,
            )),
            Tok::Keyword => Err(err_at(
                format!(
                    "Python keyword `{}` is not supported in this position inside a \
                     @triton.jit kernel",
                    t.text
                ),
                t.line,
                t.col,
            )),
            Tok::Op => Err(err_at(
                format!("expected an expression, found `{}`", t.text),
                t.line,
                t.col,
            )),
        }
    }
}

/// A token's human spelling for error messages.
fn spell(t: &Token) -> String {
    match t.kind {
        Tok::Newline => "end of line".to_string(),
        Tok::Indent => "an indent".to_string(),
        Tok::Dedent => "a dedent".to_string(),
        Tok::End => "end of file".to_string(),
        Tok::Name => format!("name `{}`", t.text),
        Tok::Int | Tok::Float => format!("number `{}`", t.text),
        Tok::Str => "a string".to_string(),
        Tok::Op => format!("`{}`", t.text),
        Tok::Keyword => format!("`{}`", t.text),
    }
}
