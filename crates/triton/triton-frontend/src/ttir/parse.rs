//! Read TTIR text into a [`Module`].
//!
//! # THIS IS A TEST INSTRUMENT, NOT A PIPELINE STAGE.
//!
//! Its ONLY job is to read the `.ttir_raw.mlir` goldens the existing Python toolchain
//! emits, so `crate::diff` can compare them against what this crate builds. No compile
//! path calls it. Bridge two receives a [`Module`] value directly and never parses text.
//!
//! It is deliberately a SUBSET parser, covering exactly the op forms the goldens contain
//! (censused with:
//! `cat tests/goldens/*.ttir_raw.mlir | grep -v '^#loc' | sed ... | sort | uniq -c`).
//! Anything else is an `Err` QUOTING THE LINE -- never a skip, because a parser that
//! silently drops lines makes the diff pass while comparing nothing.

use std::collections::HashMap;
use std::rc::Rc;

use super::{
    Attr, Block, F64Bits, FloatKind, Func, Loc, Module, Op, Region, Signedness, Type, ValueId,
    ValueInfo, Visibility,
};

#[derive(Debug)]
pub struct ParseError {
    pub line_no: usize,
    pub line: String,
    pub why: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ttir parse error at line {}: {}\n  |{}",
            self.line_no, self.why, self.line
        )
    }
}

macro_rules! bail {
    ($ctx:expr, $($arg:tt)*) => {
        return Err(ParseError { line_no: $ctx.0, line: $ctx.1.to_string(), why: format!($($arg)*) })
    };
}

// ===----------------------------------------------------------------------===//
//                                Type parsing
// ===----------------------------------------------------------------------===//

/// Parse a type spelling: `i32`, `f16`, `!tt.ptr<f16>`, `tensor<64x128xf16>`,
/// `!tt.tensordesc<64xf16>`.
pub fn parse_type(s: &str) -> Option<Type> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("!tt.ptr<") {
        let inner = rest.strip_suffix('>')?;
        return Some(Type::Ptr(Rc::new(parse_type(inner)?), 1));
    }
    if let Some(rest) = s.strip_prefix("!tt.tensordesc<") {
        let inner = rest.strip_suffix('>')?;
        let (shape, elem) = parse_shaped(inner)?;
        return Some(Type::TensorDesc(shape, Rc::new(elem)));
    }
    if let Some(rest) = s.strip_prefix("tensor<") {
        let inner = rest.strip_suffix('>')?;
        let (shape, elem) = parse_shaped(inner)?;
        return Some(Type::Tensor(shape, Rc::new(elem)));
    }
    // A bare `<f16>` / `<64x128xf16>` appears in tt.make_tensor_descriptor's tail form.
    if let Some(rest) = s.strip_prefix('<') {
        let inner = rest.strip_suffix('>')?;
        if inner.contains('x') {
            let (shape, elem) = parse_shaped(inner)?;
            return Some(Type::Tensor(shape, Rc::new(elem)));
        }
        return parse_type(inner);
    }
    if let Some(bits) = s.strip_prefix('i') {
        if let Ok(b) = bits.parse::<u32>() {
            return Some(Type::Int(
                b,
                if b == 1 { Signedness::Signless } else { Signedness::Signed },
            ));
        }
    }
    // `si32` / `ui32`: the SIGNED spelling, which MLIR uses inside a `!tt.tensordesc<>` block
    // type and nowhere else in this IR. See `ttir::desc_block_elem` for the measurement.
    // Without these two arms `embedding.ttir_raw.mlir`'s `<64xsi32>` does not parse at all,
    // and a golden that does not parse is a diff that compares nothing.
    for (prefix, sign) in [("si", Signedness::Signed), ("ui", Signedness::Unsigned)] {
        if let Some(bits) = s.strip_prefix(prefix) {
            if let Ok(b) = bits.parse::<u32>() {
                return Some(Type::Int(b, sign));
            }
        }
    }
    let fk = match s {
        "f16" => FloatKind::F16,
        "bf16" => FloatKind::BF16,
        "f32" => FloatKind::F32,
        "f64" => FloatKind::F64,
        "f8E4M3FN" => FloatKind::F8E4M3FN,
        "f8E5M2" => FloatKind::F8E5M2,
        _ => return None,
    };
    Some(Type::Float(fk))
}

/// Strip MLIR's quoting from a symbol name.
///
/// MLIR prints `@name` when the symbol is a bare identifier and `@"name"` when it is not.
/// Triton's mangled names are not bare whenever a float constexpr reaches the exponent form --
/// `c1e-05` contains a `-` -- so both spellings occur in the goldens for the same kind of
/// symbol. The quotes are escaping; the SYMBOL is what is inside them.
pub fn unquote_symbol(s: &str) -> String {
    let s = s.trim();
    match s.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        Some(inner) => inner.to_string(),
        None => s.to_string(),
    }
}

/// `64x128xf16` -> ([64,128], f16). `f16` alone -> ([], f16).
fn parse_shaped(s: &str) -> Option<(Vec<i64>, Type)> {
    let mut dims = Vec::new();
    let mut rest = s;
    loop {
        match rest.split_once('x') {
            Some((head, tail)) => match head.parse::<i64>() {
                Ok(d) => {
                    dims.push(d);
                    rest = tail;
                }
                Err(_) => break,
            },
            None => break,
        }
    }
    Some((dims, parse_type(rest)?))
}

// ===----------------------------------------------------------------------===//
//                             Small text helpers
// ===----------------------------------------------------------------------===//

/// Split on `sep` at depth 0 with respect to `<>`, `()`, `[]`, `{}`.
fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            _ if c == sep && depth == 0 => {
                out.push(cur.trim().to_string());
                cur = String::new();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Where an op's trailing type list begins, and how many bytes to skip.
///
/// Normally the separator is `" : "`, but an op with NO OPERANDS prints as
/// `ub.poison : i32`, and after the name is split off the remainder is `": i32"` with no
/// leading space. Missing that case silently left such ops with no result type -- found by
/// `tests/no_parser.rs::every_raw_golden_parses`, which is why that test exists.
fn find_type_sep(s: &str) -> Option<(usize, usize)> {
    if let Some(i) = find_top(s, " : ") {
        return Some((i, 3));
    }
    let t = s.trim_start();
    if t.starts_with(": ") {
        return Some((s.len() - t.len(), 2));
    }
    None
}

/// Find `needle` at bracket depth 0.
fn find_top(s: &str, needle: &str) -> Option<usize> {
    let bytes: Vec<char> = s.chars().collect();
    let n: Vec<char> = needle.chars().collect();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 && i + n.len() <= bytes.len() && bytes[i..i + n.len()] == n[..] {
            // Byte offset, not char offset.
            return Some(bytes[..i].iter().map(|c| c.len_utf8()).sum());
        }
        i += 1;
    }
    None
}

/// Strip a trailing `loc(#locN)` / `loc(unknown)` / `loc("x"(#loc))`, returning it.
fn take_loc(s: &str) -> (String, Option<String>) {
    let t = s.trim_end();
    if !t.ends_with(')') {
        return (t.to_string(), None);
    }
    // Walk back to the matching `loc(`.
    let chars: Vec<char> = t.chars().collect();
    let mut depth = 0i32;
    let mut i = chars.len();
    while i > 0 {
        i -= 1;
        match chars[i] {
            ')' => depth += 1,
            '(' => {
                depth -= 1;
                if depth == 0 {
                    // Is the token before `(` exactly `loc`?
                    if i >= 3 && chars[i - 3..i].iter().collect::<String>() == "loc" {
                        let start: usize = chars[..i - 3].iter().map(|c| c.len_utf8()).sum();
                        let loc = t[start..].to_string();
                        return (t[..start].trim_end().to_string(), Some(loc));
                    }
                    return (t.to_string(), None);
                }
            }
            _ => {}
        }
    }
    (t.to_string(), None)
}

// ===----------------------------------------------------------------------===//
//                                 The parser
// ===----------------------------------------------------------------------===//

struct Parser<'a> {
    lines: Vec<(usize, &'a str)>,
    pos: usize,
    /// `#locN` -> Loc
    locs: HashMap<String, Loc>,
    m: Module,
    /// SSA text name -> value id, per function.
    scope: HashMap<String, ValueId>,
}

impl<'a> Parser<'a> {
    fn new_value(&mut self, ty: Type, loc: Loc) -> ValueId {
        let id = ValueId(self.m.values.len() as u32);
        self.m.values.push(ValueInfo { ty, loc });
        id
    }

    fn loc_ref(&self, s: &Option<String>) -> Loc {
        match s {
            None => Loc::Unknown,
            Some(l) => {
                let inner = l.trim();
                // loc(#loc12)
                if let Some(r) = inner.strip_prefix("loc(#").and_then(|r| r.strip_suffix(')')) {
                    return self.locs.get(&format!("#{r}")).cloned().unwrap_or(Loc::Unknown);
                }
                if inner == "loc(unknown)" {
                    return Loc::Unknown;
                }
                // loc("name"(#loc))
                if let Some(r) = inner.strip_prefix("loc(").and_then(|r| r.strip_suffix(')')) {
                    return self.parse_loc_body(r);
                }
                Loc::Unknown
            }
        }
    }

    /// Body of a `loc(...)`: `unknown`, `"file":L:C`, `"name"(#locN)`, `#locN`.
    fn parse_loc_body(&self, s: &str) -> Loc {
        let s = s.trim();
        if s == "unknown" {
            return Loc::Unknown;
        }
        if let Some(r) = s.strip_prefix('#') {
            return self.locs.get(&format!("#{r}")).cloned().unwrap_or(Loc::Unknown);
        }
        if let Some(rest) = s.strip_prefix('"') {
            let close = rest.find('"').unwrap_or(0);
            let name = &rest[..close];
            let after = rest[close + 1..].trim();
            if let Some(inner) = after.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
                return Loc::Name(name.to_string(), Box::new(self.parse_loc_body(inner)));
            }
            // "file":line:col
            if let Some(nums) = after.strip_prefix(':') {
                let parts: Vec<&str> = nums.split(':').collect();
                if parts.len() == 2 {
                    return Loc::File {
                        file: Rc::from(name),
                        line: parts[0].parse().unwrap_or(0),
                        col: parts[1].parse().unwrap_or(0),
                    };
                }
            }
            return Loc::Name(name.to_string(), Box::new(Loc::Unknown));
        }
        Loc::Unknown
    }
}

/// Parse a whole TTIR text module.
pub fn parse_module(text: &str) -> Result<Module, ParseError> {
    let raw: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.trim().is_empty())
        .collect();

    let mut p = Parser {
        lines: raw,
        pos: 0,
        locs: HashMap::new(),
        m: Module::default(),
        scope: HashMap::new(),
    };

    // PASS 1: the `#locN = loc(...)` definitions, which appear both before and after the
    // module body. Two sweeps, because a definition can forward-reference a later one.
    for _ in 0..2 {
        for (_, line) in p.lines.clone() {
            let t = line.trim();
            if let Some((lhs, rhs)) = t.split_once(" = ") {
                if lhs.starts_with("#loc") {
                    if let Some(body) = rhs.trim().strip_prefix("loc(").and_then(|x| x.strip_suffix(')'))
                    {
                        let loc = p.parse_loc_body(body);
                        p.locs.insert(lhs.trim().to_string(), loc);
                    }
                }
            }
        }
    }
    // The bare `#loc` (no number) is the module/file location.
    if let Some(l) = p.locs.get("#loc").cloned() {
        p.m.loc = l;
    }

    // PASS 2: the module body.
    while p.pos < p.lines.len() {
        let (no, line) = p.lines[p.pos];
        let t = line.trim();
        p.pos += 1;
        if t.starts_with("#loc") || t == "module {" {
            continue;
        }
        if t.starts_with('}') {
            continue;
        }
        if t.starts_with("tt.func") {
            let f = parse_func(&mut p, no, t)?;
            p.m.funcs.push(f);
            continue;
        }
        bail!((no, line), "unexpected top-level line");
    }
    Ok(p.m)
}

fn parse_func(p: &mut Parser, no: usize, line: &str) -> Result<Func, ParseError> {
    // tt.func public @name(%a: T loc(...), ...) [-> (T, ...)] attributes {noinline = false} {
    let visibility = if line.contains("tt.func public") {
        Visibility::Public
    } else if line.contains("tt.func private") {
        Visibility::Private
    } else {
        bail!((no, line), "tt.func with no public/private visibility")
    };
    let at = match line.find('@') {
        Some(i) => i,
        None => bail!((no, line), "tt.func with no @name"),
    };
    let open = match line[at..].find('(') {
        Some(i) => at + i,
        None => bail!((no, line), "tt.func with no argument list"),
    };
    // MLIR QUOTES a symbol name that is not a bare identifier, and a mangled name can contain
    // a `-` (from a float constexpr like `1e-05`), so `@"decoder_block._decoder_layer__...
    // c1e-05..."` is what the golden holds. The quotes are MLIR's escaping, not part of the
    // symbol, so they come off here -- otherwise the same function compares as two.
    let name = unquote_symbol(line[at + 1..open].trim());
    // Matching close paren for the argument list.
    let close = {
        let mut depth = 0i32;
        let mut found = None;
        for (i, c) in line[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        found = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        match found {
            Some(i) => i,
            None => bail!((no, line), "unterminated tt.func argument list"),
        }
    };
    let args_src = &line[open + 1..close];
    let tail = &line[close + 1..];

    let noinline = tail.contains("noinline = true");
    let mut ret_types = Vec::new();
    if let Some(arrow) = tail.find("->") {
        let after = tail[arrow + 2..].trim();
        let tys = after
            .split(" attributes")
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches('{')
            .trim()
            .trim_end_matches(')')
            .to_string();
        for t in split_top(&tys, ',') {
            if let Some(ty) = parse_type(&t) {
                ret_types.push(ty);
            }
        }
    }

    p.scope.clear();
    let mut arg_types = Vec::new();
    let mut block_args = Vec::new();
    for a in split_top(args_src, ',') {
        if a.is_empty() {
            continue;
        }
        let (decl, loc_s) = take_loc(&a);
        let (nm, ty_s) = match decl.split_once(':') {
            Some(x) => x,
            None => bail!((no, line), "tt.func argument without a type: {a}"),
        };
        let ty = match parse_type(ty_s) {
            Some(t) => t,
            None => bail!((no, line), "unparsable argument type {ty_s:?}"),
        };
        let loc = p.loc_ref(&loc_s);
        let id = p.new_value(ty.clone(), loc);
        p.scope.insert(nm.trim().trim_start_matches('%').to_string(), id);
        arg_types.push(ty);
        block_args.push(id);
    }

    let (_, floc) = take_loc(line);
    let func_loc = p.loc_ref(&floc);

    let mut region = Region::default();
    region.blocks.push(Block {
        args: block_args,
        ops: Vec::new(),
    });
    parse_region_until_close(p, &mut region)?;

    Ok(Func {
        name,
        visibility,
        noinline,
        arg_types,
        ret_types,
        body: region,
        loc: func_loc,
    })
}

/// Consume lines until the `}` that closes the current region, filling in its BLOCKS.
///
/// A `^bbN...` label starts a new block. Both kinds occur in the goldens and both are real
/// structure, not decoration:
///   * `^bb1:  // no predecessors` -- the unreachable block `visit_Return` creates so the
///     terminator ends its own block; every generated `standard.*` helper has one.
///   * `^bb0(%a: f32, %b: f32):`   -- a `tt.reduce` region's entry block and its arguments.
/// Returns the CLOSING LINE, which is not incidental: MLIR prints a region-carrying op's
/// location AFTER its body, so `scf.for`'s `loc(...)` is on the `}` line and the opening line
/// has none. Reading only the opening line left every `scf.for` with an unnamed location and
/// showed up in the golden diff as four `loc_name` findings.
fn parse_region_until_close(
    p: &mut Parser,
    region: &mut Region,
) -> Result<Option<String>, ParseError> {
    if region.blocks.is_empty() {
        region.blocks.push(Block::default());
    }
    while p.pos < p.lines.len() {
        let (no, line) = p.lines[p.pos];
        let t = line.trim();
        if t.starts_with('}') {
            p.pos += 1;
            return Ok(Some(t.to_string()));
        }
        p.pos += 1;
        if t.starts_with("^bb") {
            let mut args = Vec::new();
            if let Some(open) = t.find('(') {
                let close = t.rfind(')').unwrap_or(open);
                for a in split_top(&t[open + 1..close], ',') {
                    let (decl, loc_s) = take_loc(&a);
                    if let Some((nm, ty_s)) = decl.split_once(':') {
                        let ty = match parse_type(ty_s) {
                            Some(ty) => ty,
                            None => bail!((no, line), "unparsable block-argument type {ty_s:?}"),
                        };
                        let loc = p.loc_ref(&loc_s);
                        let id = p.new_value(ty, loc);
                        p.scope
                            .insert(nm.trim().trim_start_matches('%').to_string(), id);
                        args.push(id);
                    }
                }
            }
            // MLIR prints the ENTRY block's label only when it has arguments, and the entry
            // block we were handed may already be empty and unlabelled. Reuse it in that
            // case rather than leaving a stray empty block, which would shift every block
            // index the diff reports.
            let reuse = region.blocks.len() == 1
                && region.blocks[0].ops.is_empty()
                && region.blocks[0].args.is_empty();
            if reuse {
                region.blocks[0].args = args;
            } else {
                region.blocks.push(Block {
                    args,
                    ops: Vec::new(),
                });
            }
            continue;
        }
        if let Some(op) = parse_op_line(p, no, t)? {
            region
                .blocks
                .last_mut()
                .expect("a region always has a block")
                .ops
                .push(op);
        }
    }
    Ok(None)
}

/// Apply what a region-carrying op prints on its CLOSING line: its location, and -- for the
/// generic form -- its type signature.
///
/// MLIR puts both after the body:
///
/// ```text
/// %acc_31 = scf.for ... -> (tensor<64x128xf16>)  : i32 {   <- types on the OPENING line
/// } loc(#loc45)                                            <- location on the CLOSING line
///
/// %0 = "tt.reduce"(%x) <{axis = 1 : i32}> ({                <- NO types on the opening line
/// }) : (tensor<64x64xf32>) -> tensor<64xf32> loc(...)       <- both on the CLOSING line
/// ```
///
/// Reading only the opening line left `tt.reduce` with its OPERAND's type as its result --
/// `tensor<64x64xf32>` instead of `tensor<64xf32>` -- which the golden diff caught as a
/// result-type finding on every reduction. Keeps the invariant that a result's location is
/// its op's (see `ttir::Op`).
fn apply_close_line(p: &mut Parser, op: &mut Op, close: Option<String>) {
    let Some(close) = close else { return };
    let (body, loc_s) = take_loc(&close);
    if let Some(loc_s) = &loc_s {
        let loc = p.loc_ref(&Some(loc_s.clone()));
        op.loc = loc.clone();
        for r in &op.results {
            p.m.values[r.0 as usize].loc = loc.clone();
        }
    }
    // `}) : (operands) -> results`: the result types belong to the op.
    //
    // The leading `}` and `)` must be stripped BEFORE the depth-aware search, or they drive
    // the bracket depth negative and `" -> "` is never seen at depth 0 -- which is why the
    // first attempt at this left every `tt.reduce` with its operand's type.
    let body = body
        .trim_start()
        .trim_start_matches('}')
        .trim_start()
        .trim_start_matches(')')
        .to_string();
    if let Some(i) = find_top(&body, " -> ") {
        let rhs = body[i + 4..].trim();
        let rhs = rhs.trim_start_matches('(').trim_end_matches(')');
        let tys: Vec<Type> = split_top(rhs, ',').iter().filter_map(|t| parse_type(t)).collect();
        if tys.len() == op.results.len() {
            for (r, t) in op.results.iter().zip(tys) {
                p.m.values[r.0 as usize].ty = t;
            }
        }
    }
}

/// Parse one op line (possibly opening a region).
fn parse_op_line(p: &mut Parser, no: usize, t: &str) -> Result<Option<Op>, ParseError> {
    // `}) : (tensor<...>) -> tensor<...> loc(...)` closes a tt.reduce; handled by the
    // region walker via the `}` prefix check, so it should not arrive here.
    let opens_region = t.ends_with('{');
    let body = if opens_region {
        t[..t.len() - 1].trim_end()
    } else {
        t
    };
    let (body, loc_s) = take_loc(body);
    let loc = p.loc_ref(&loc_s);

    // Split `%res = rest` / `%res:2 = rest`.
    let (result_spec, rest) = match find_top(&body, " = ") {
        Some(i) if body.starts_with('%') => (Some(body[..i].to_string()), body[i + 3..].to_string()),
        _ => (None, body.clone()),
    };

    // Op name is the first token, minus an optional quoted form (`"tt.reduce"(...)`).
    let rest = rest.trim().to_string();
    let (name, args_src) = if let Some(r) = rest.strip_prefix('"') {
        let close = match r.find('"') {
            Some(i) => i,
            None => bail!((no, t), "unterminated quoted op name"),
        };
        (r[..close].to_string(), r[close + 1..].trim().to_string())
    } else {
        match rest.find(' ') {
            Some(i) => (rest[..i].to_string(), rest[i + 1..].trim().to_string()),
            None => (rest.clone(), String::new()),
        }
    };

    let mut op = Op::new(name.clone(), loc.clone());

    // ---- attributes ------------------------------------------------------------
    // `{axis = 1 : i32}`, `<{axis = 1 : i32}>`, `{end = 64 : i32, start = 0 : i32}`,
    // `{order = array<i32: 1, 0>}`.
    let mut work = args_src.clone();
    if let Some(s) = extract_braced(&work) {
        for kv in split_top(&s.inner, ',') {
            if let Some((k, v)) = kv.split_once('=') {
                op.attrs.insert(k.trim().to_string(), parse_attr_value(v.trim()));
            }
        }
        work = format!("{}{}", &work[..s.start], &work[s.end..]);
    }

    // ---- tt.call's callee ------------------------------------------------------
    // `tt.call @triton.language.standard.zeros__Tc64_c64T_cfp16() : () -> tensor<...>`.
    // The symbol IS the semantic content of the op (it encodes the callee's constexpr
    // arguments), so it becomes an attribute the diff compares.
    if name == "tt.call" {
        let w = work.trim();
        if let Some(rest) = w.strip_prefix('@') {
            // A QUOTED callee runs to its closing quote; a bare one to the argument list. The
            // quoted form appears whenever the mangled name is not a bare identifier -- see
            // `unquote_symbol`.
            let sym = if let Some(inner) = rest.strip_prefix('"') {
                match inner.find('"') {
                    Some(end) => inner[..end].to_string(),
                    None => bail!((no, t), "tt.call with an unterminated quoted callee"),
                }
            } else {
                let end = rest
                    .find(['(', ' '])
                    .unwrap_or(rest.len());
                rest[..end].to_string()
            };
            op.attrs.insert("callee".to_string(), Attr::Str(sym));
        } else {
            bail!((no, t), "tt.call without an @symbol");
        }
    }

    // ---- arith.constant's value ------------------------------------------------
    if name == "arith.constant" {
        let v = work.trim();
        op.attrs.insert("value".to_string(), parse_attr_value(v));
    }

    // ---- comparison predicate --------------------------------------------------
    if name == "arith.cmpi" || name == "arith.cmpf" {
        let first = work.trim().split(',').next().unwrap_or("").trim().to_string();
        let pred: &'static str = match first.as_str() {
            "eq" => "eq",
            "ne" => "ne",
            "slt" => "slt",
            "sle" => "sle",
            "sgt" => "sgt",
            "sge" => "sge",
            "ult" => "ult",
            "ule" => "ule",
            "ugt" => "ugt",
            "uge" => "uge",
            "oeq" => "oeq",
            "ogt" => "ogt",
            "oge" => "oge",
            "olt" => "olt",
            "ole" => "ole",
            "one" => "one",
            "une" => "une",
            _ => bail!((no, t), "unknown compare predicate {first:?}"),
        };
        op.attrs.insert("predicate".to_string(), Attr::Pred(pred));
        work = work
            .trim()
            .strip_prefix(&first)
            .unwrap_or(&work)
            .trim_start()
            .trim_start_matches(',')
            .to_string();
    }

    // ---- tt.get_program_id / num_programs axis ---------------------------------
    if name == "tt.get_program_id" || name == "tt.get_num_programs" {
        let ax = work.trim().split(&[' ', ':'][..]).next().unwrap_or("x");
        let axis: &'static str = match ax {
            "x" => "x",
            "y" => "y",
            "z" => "z",
            _ => bail!((no, t), "unknown program-id axis {ax:?}"),
        };
        op.attrs.insert("axis".to_string(), Attr::Axis(axis));
    }

    // ---- scf.for, whose block arguments are declared ON THE OP LINE -------------
    //
    //   %acc_31 = scf.for %n = %0 to %1 step %2 iter_args(%acc_33 = %acc)
    //                 -> (tensor<64x128xf16>)  : i32 {
    //
    // `%n` and `%acc_33` are DEFINITIONS (the region's entry-block arguments), while `%0`,
    // `%1`, `%2` and `%acc` are USES. The generic operand scan below cannot tell them apart
    // and would record all six as operands, so this form is handled on its own.
    if name == "scf.for" {
        let (lb, ub, step, iter_pairs, iv_name, iv_ty) = match parse_scf_for_head(&work) {
            Some(x) => x,
            None => bail!((no, t), "could not read the scf.for header"),
        };
        for nm in [lb, ub, step].iter().chain(iter_pairs.iter().map(|(_, r)| r)) {
            match p.scope.get(nm.as_str()).copied() {
                Some(id) => op.operands.push(id),
                None => bail!((no, t), "scf.for refers to an unknown value %{nm}"),
            }
        }
        let result_types = infer_result_types(&name, &work, &op, p);
        if result_types.len() != iter_pairs.len() {
            bail!(
                (no, t),
                "scf.for has {} iter_args but {} result type(s)",
                iter_pairs.len(),
                result_types.len()
            );
        }
        // Bind the results in the ENCLOSING scope.
        if let Some(spec) = &result_spec {
            let spec = spec.trim();
            let (base, count) = match spec.split_once(':') {
                Some((b, c)) => (
                    b.trim_start_matches('%').to_string(),
                    c.trim().parse::<usize>().unwrap_or(1),
                ),
                None => (spec.trim_start_matches('%').to_string(), 1),
            };
            for k in 0..count.min(result_types.len()) {
                let id = p.new_value(result_types[k].clone(), loc.clone());
                op.results.push(id);
                if count == 1 {
                    p.scope.insert(base.clone(), id);
                } else {
                    p.scope.insert(format!("{base}#{k}"), id);
                }
            }
        }
        // The region's entry block: induction variable, then one argument per carry, typed
        // by the matching result.
        let mut args = Vec::new();
        let ivt = parse_type(&iv_ty).unwrap_or_else(Type::i32);
        let iv_id = p.new_value(ivt, Loc::Unknown);
        p.scope.insert(iv_name, iv_id);
        args.push(iv_id);
        for (k, (lhs, _)) in iter_pairs.iter().enumerate() {
            let id = p.new_value(result_types[k].clone(), Loc::Unknown);
            p.scope.insert(lhs.clone(), id);
            args.push(id);
        }
        if !opens_region {
            bail!((no, t), "scf.for without a body region");
        }
        let mut r = Region::default();
        r.blocks.push(Block {
            args,
            ops: Vec::new(),
        });
        let close = parse_region_until_close(p, &mut r)?;
        op.regions.push(r);
        apply_close_line(p, &mut op, close);
        return Ok(Some(op));
    }

    // ---- operands --------------------------------------------------------------
    // Every `%name` / `%name#K` token before the type separator is an operand.
    let operand_src = match find_type_sep(&work) {
        Some((i, _)) => work[..i].to_string(),
        None => work.clone(),
    };
    // A HYPHEN IS PART OF AN SSA NAME. `arith.constant`'s self-naming produces
    // `%c-2147483648_i64` for the overflow check's lower bound, and splitting at the `-`
    // dropped that operand -- which showed up as an `arith.cmpi` with one operand where ours
    // had two, on the golden side, in four places in the causal attention configuration.
    for tok in operand_src
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '%' || c == '#' || c == '-'))
    {
        if let Some(r) = tok.strip_prefix('%') {
            let base = r.split('#').next().unwrap_or(r);
            // `%V#1` selects result 1 of a multi-result op; the scope stores each result
            // under `base#k`, and `base` alone for single-result ops.
            let key = if r.contains('#') { r.to_string() } else { base.to_string() };
            if let Some(id) = p.scope.get(&key).copied() {
                op.operands.push(id);
            } else if let Some(id) = p.scope.get(base).copied() {
                op.operands.push(id);
            }
        }
    }

    // ---- result types ----------------------------------------------------------
    let result_types = infer_result_types(&name, &work, &op, p);

    // ---- bind results ----------------------------------------------------------
    if let Some(spec) = &result_spec {
        let spec = spec.trim();
        let (base, count) = match spec.split_once(':') {
            Some((b, c)) => (b.trim_start_matches('%').to_string(), c.trim().parse::<usize>().unwrap_or(1)),
            None => (spec.trim_start_matches('%').to_string(), 1),
        };
        if result_types.len() < count {
            bail!(
                (no, t),
                "op {name} declares {count} results but {} type(s) were recoverable",
                result_types.len()
            );
        }
        for k in 0..count {
            let id = p.new_value(result_types[k].clone(), loc.clone());
            op.results.push(id);
            if count == 1 {
                p.scope.insert(base.clone(), id);
            } else {
                p.scope.insert(format!("{base}#{k}"), id);
            }
        }
    }

    // ---- regions ---------------------------------------------------------------
    if opens_region {
        let mut r = Region::default();
        let close = parse_region_until_close(p, &mut r)?;
        op.regions.push(r);
        apply_close_line(p, &mut op, close);
        // A `tt.reduce`'s close line is `}) : (T) -> T loc(...)`, which the region walker
        // consumed as the closing `}`; its trailing type list is already reflected in the
        // result types recovered above, so nothing further is needed here.
    }

    Ok(Some(op))
}

/// Pull apart an `scf.for` header.
///
/// Returns `(lb, ub, step, [(carry_arg, init)], induction_var, induction_type)`, all names
/// with the leading `%` stripped.
#[allow(clippy::type_complexity)]
fn parse_scf_for_head(
    work: &str,
) -> Option<(String, String, String, Vec<(String, String)>, String, String)> {
    // `%n = %0 to %1 step %2 iter_args(...) -> (...)  : i32`
    let (iv, rest) = work.trim().split_once(" = ")?;
    let (lb, rest) = rest.split_once(" to ")?;
    let (ub, rest) = rest.split_once(" step ")?;
    // The step runs to `iter_args`, `->`, or the trailing type.
    let mut iter_pairs = Vec::new();
    let step_end = rest
        .find(" iter_args(")
        .or_else(|| find_top(rest, " -> "))
        .or_else(|| find_type_sep(rest).map(|(i, _)| i))
        .unwrap_or(rest.len());
    let step = rest[..step_end].trim();
    if let Some(i) = rest.find(" iter_args(") {
        let after = &rest[i + " iter_args(".len()..];
        // Matching close paren.
        let mut depth = 1i32;
        let mut end = after.len();
        for (j, c) in after.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        for pair in split_top(&after[..end], ',') {
            let (l, r) = pair.split_once('=')?;
            iter_pairs.push((
                l.trim().trim_start_matches('%').to_string(),
                r.trim().trim_start_matches('%').to_string(),
            ));
        }
    }
    // The induction variable's type is the trailing `: i32`.
    let iv_ty = match find_type_sep(work) {
        Some((i, skip)) => work[i + skip..].trim().to_string(),
        None => "i32".to_string(),
    };
    Some((
        lb.trim().trim_start_matches('%').to_string(),
        ub.trim().trim_start_matches('%').to_string(),
        step.trim().trim_start_matches('%').to_string(),
        iter_pairs,
        iv.trim().trim_start_matches('%').to_string(),
        iv_ty,
    ))
}

struct Braced {
    inner: String,
    start: usize,
    end: usize,
}

/// Extract the first `{...}` (or `<{...}>`) group, if the line has one.
fn extract_braced(s: &str) -> Option<Braced> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    for (i, c) in s[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = start + i + 1;
                    return Some(Braced {
                        inner: s[start + 1..start + i].to_string(),
                        start,
                        end,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_attr_value(v: &str) -> Attr {
    let v = v.trim();
    if v == "true" {
        return Attr::Bool(true);
    }
    if v == "false" {
        return Attr::Bool(false);
    }
    if let Some(rest) = v.strip_prefix("array<i32:") {
        let inner = rest.trim_end_matches('>');
        let items = inner
            .split(',')
            .filter_map(|x| x.trim().parse::<i128>().ok())
            .map(|n| Attr::Int(n, Type::i32()))
            .collect();
        return Attr::Array(items);
    }
    if let Some(rest) = v.strip_prefix("dense<") {
        if let Some((val, ty)) = rest.split_once('>') {
            let ty = ty.trim().trim_start_matches(':').trim();
            let ty = parse_type(ty).unwrap_or(Type::i32());
            let inner = parse_scalar_literal(val, ty.scalar().clone());
            return Attr::DenseSplat(Box::new(inner), ty);
        }
    }
    if let Some((val, ty)) = split_last_colon(v) {
        if let Some(ty) = parse_type(&ty) {
            return parse_scalar_literal(&val, ty);
        }
    }
    Attr::Str(v.to_string())
}

/// Decode an IEEE binary16 bit pattern.
fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0f64 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as f64;
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        31 => {
            if mant == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

fn split_last_colon(s: &str) -> Option<(String, String)> {
    let i = find_top(s, " : ")?;
    Some((s[..i].trim().to_string(), s[i + 3..].trim().to_string()))
}

fn parse_scalar_literal(v: &str, ty: Type) -> Attr {
    let v = v.trim();
    if ty.is_floating() {
        // MLIR prints a special float as the HEX BIT PATTERN OF ITS OWN TYPE -- `0x7C00` for
        // an f16 infinity, `0x7F800000` for an f32 one. Reading those as f64 bits produced
        // 1.5e-319 and 1.05e-314, which then differed from our (correct) infinities and looked
        // like a codegen bug rather than a parser one.
        if let Some(hex) = v.strip_prefix("0x") {
            if let Ok(bits) = u64::from_str_radix(hex, 16) {
                let f = match ty.scalar() {
                    Type::Float(FloatKind::F16) => f16_bits_to_f64(bits as u16),
                    Type::Float(FloatKind::BF16) => {
                        f32::from_bits(((bits as u32) & 0xFFFF) << 16) as f64
                    }
                    Type::Float(FloatKind::F32) => f32::from_bits(bits as u32) as f64,
                    _ => f64::from_bits(bits),
                };
                return Attr::Float(F64Bits::new(f), ty);
            }
        }
        if let Ok(f) = v.parse::<f64>() {
            return Attr::Float(F64Bits::new(f), ty);
        }
    }
    if let Ok(i) = v.parse::<i128>() {
        return Attr::Int(i, ty);
    }
    Attr::Str(v.to_string())
}

/// Recover an op's result types from the printed form.
fn infer_result_types(name: &str, work: &str, op: &Op, p: &Parser) -> Vec<Type> {
    // `tt.make_tensor_descriptor`'s custom form prints the POINTEE type and then the BLOCK
    // type: `... : <f16>, <64x128xf16>`. The single result is
    // `!tt.tensordesc<block shape x pointee>` -- neither of the two printed types as
    // written. Special-cased because reading the first type here silently produced `f16`
    // and made three ops per fixture compare as scalars.
    if name == "tt.make_tensor_descriptor" {
        if let Some((i, skip)) = find_type_sep(work) {
            let parts = split_top(work[i + skip..].trim(), ',');
            if let (Some(elem_s), Some(block_s)) = (parts.first(), parts.last()) {
                let elem = parse_type(elem_s);
                let block = parse_type(block_s);
                if let (Some(elem), Some(block)) = (elem, block) {
                    let shape = block.shape().to_vec();
                    let e = if block.shape().is_empty() {
                        elem
                    } else {
                        block.scalar().clone()
                    };
                    return vec![Type::TensorDesc(shape, Rc::new(e))];
                }
            }
        }
        return Vec::new();
    }
    // `A -> B`: the results are on the right.
    if let Some(i) = find_top(work, " -> ") {
        let mut rhs = work[i + 4..].trim();
        // `scf.for` appends the INDUCTION VARIABLE's type after the result list:
        //   `scf.for %n = %0 to %1 step %2 iter_args(...) -> (tensor<64x128xf16>)  : i32 {`
        // so the trailing `: i32` must be cut before the result list is read, or nothing
        // parses. Found by `tests/no_parser.rs::every_raw_golden_parses`.
        if let Some((j, _)) = find_type_sep(rhs) {
            rhs = rhs[..j].trim();
        }
        let rhs = rhs.trim_start_matches('(').trim_end_matches(')');
        return split_top(rhs, ',').iter().filter_map(|t| parse_type(t)).collect();
    }
    // `scf.for ... -> (types)` was handled above. `iter_args` form without `->` cannot
    // happen in MLIR output.
    // `op operands : T` -- for most arith ops the result type equals T.
    if let Some((i, skip)) = find_type_sep(work) {
        let rhs = work[i + skip..].trim();
        // A CAST prints `SRC to DST`; the result is the DESTINATION type. Without this the
        // whole `i32 to i64` fails to parse and the op silently loses its result type,
        // which is exactly how a subset parser turns into a diff that compares nothing.
        if let Some(j) = find_top(rhs, " to ") {
            let dst = rhs[j + 4..].trim();
            return split_top(dst, ',')
                .iter()
                .filter_map(|t| parse_type(t))
                .collect();
        }
        let tys: Vec<Type> = split_top(rhs, ',').iter().filter_map(|t| parse_type(t)).collect();
        // Comparisons narrow to i1 (or a tensor of i1).
        if name == "arith.cmpi" || name == "arith.cmpf" {
            return tys
                .first()
                .map(|t| vec![t.with_element_ty(Type::i1())])
                .unwrap_or_else(|| vec![Type::i1()]);
        }
        return tys;
    }
    // `arith.constant <value>` with the type inside the attribute.
    if let Some(v) = op.attrs.get("value") {
        return match v {
            Attr::Int(_, t) | Attr::Float(_, t) => vec![t.clone()],
            Attr::DenseSplat(_, t) => vec![t.clone()],
            _ => vec![],
        };
    }
    // Fall back to the first operand's type (e.g. a bare `arith.negf %x`).
    op.operands.first().map(|v| vec![p.m.ty(*v).clone()]).unwrap_or_default()
}
