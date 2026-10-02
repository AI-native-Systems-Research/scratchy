//! Render a [`Module`] as TTIR text.
//!
//! # THIS IS A TEST INSTRUMENT, NOT A PIPELINE STAGE.
//!
//! Nothing in the compile path calls this. It exists so a human can read what the front
//! end built, and so a failing golden diff can show both sides. The diff itself compares
//! [`Module`] against [`Module`] (see `crate::diff`) -- the golden text is read by
//! `super::parse`, never by printing ours and comparing strings. That ordering matters: a
//! text comparison would fail on MLIR's SSA numbering and `#locN` bookkeeping, which are
//! not facts about the program.
//!
//! The SSA naming here MIMICS MLIR's asm printer closely enough to read side by side with
//! a golden -- name from the `NameLoc` if present, else `arith.constant`'s own
//! `%c<value>_<type>` / `%cst` convention, else `%N` -- but nothing depends on it being
//! exact.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::{
    Attr, Block, F64Bits, FloatKind, Func, Loc, Module, Op, Region, Type, ValueId, Visibility,
};

/// Format a float the way MLIR's `arith.constant` does.
///
/// # The six-digit form is MLIR's FIRST CHOICE, not its only one.
///
/// MLIR prints `1.200000e+01` and `0.000000e+00` -- six mantissa digits -- but when six digits
/// do not round-trip AT THE VALUE'S OWN TYPE it falls back to a full-precision spelling, which
/// is why the same golden also contains `9.99999974E-6` (f32 `1e-05`). Printing the six-digit
/// form unconditionally therefore LOSES PRECISION: `9.999999e-06` reads back as a different f32,
/// and `print -> parse` stopped being idempotent for any module carrying such a constant.
///
/// So: try six digits, read it back at `ty`, and fall back to the shortest round-tripping form
/// when it does not agree.
fn fmt_float(bits: F64Bits, ty: &Type) -> String {
    let short = fmt_float_6(bits);
    // Does the short form read back as the same value AT THIS TYPE?
    if let Ok(back) = short.parse::<f64>() {
        if F64Bits::rounded(back, ty) == F64Bits::rounded(bits.get(), ty) {
            return short;
        }
    }
    let v = bits.get();
    match ty.scalar() {
        // An f32 needs at most 9 significant digits; Rust's `{:e}` on the f32 gives the
        // shortest spelling that round-trips as an f32.
        Type::Float(FloatKind::F32) => fmt_exp_upper(v as f32 as f64),
        _ => fmt_exp_upper(v),
    }
}

/// MLIR's full-precision fallback spelling: shortest round-tripping digits, capital `E`, and an
/// unpadded exponent -- `9.99999974E-6`. Rust's `{:E}` already omits the fractional part when
/// the mantissa is integral, so no re-spelling is needed.
fn fmt_exp_upper(v: f64) -> String {
    format!("{v:E}")
}

fn fmt_float_6(bits: F64Bits) -> String {
    let v = bits.get();
    if v.is_nan() {
        return "0x7FF8000000000000".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "0x7FF0000000000000"
        } else {
            "0xFFF0000000000000"
        }
        .to_string();
    }
    // MLIR prints e.g. 0.000000e+00, 1.000000e+00, 1.442700e+00.
    let s = format!("{v:e}");
    // Rust's `{:e}` gives `0e0` / `1e0` / `1.4427e0`; normalize to MLIR's 6-digit
    // mantissa + signed 2-digit exponent.
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let mant = mant.trim_start_matches('-');
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let mut frac = frac_part.to_string();
    while frac.len() < 6 {
        frac.push('0');
    }
    format!(
        "{}{}.{}e{}{:02}",
        if neg { "-" } else { "" },
        int_part,
        &frac[..6],
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

fn fmt_attr(a: &Attr) -> String {
    match a {
        Attr::Int(v, ty) => format!("{v} : {ty}"),
        Attr::Float(bits, ty) => format!("{} : {}", fmt_float(*bits, ty), ty),
        Attr::DenseSplat(inner, ty) => {
            let v = match &**inner {
                Attr::Float(bits, _) => fmt_float(*bits, ty),
                Attr::Int(v, _) => v.to_string(),
                Attr::Bool(b) => b.to_string(),
                other => fmt_attr(other),
            };
            format!("dense<{v}> : {ty}")
        }
        Attr::Bool(b) => b.to_string(),
        Attr::Str(s) => format!("\"{s}\""),
        Attr::Unit => String::new(),
        Attr::Pred(p) => p.to_string(),
        Attr::Axis(x) => x.to_string(),
        // MLIR prints a dense i32 array as `array<i32: 1, 0>` -- `tt.trans`'s `order`. The
        // bracketed form this used to print reads back as a STRING, so `tt.trans`'s order was
        // lost on the way through `parse`.
        Attr::Array(items) => {
            let ints: Option<Vec<String>> = items
                .iter()
                .map(|i| match i {
                    Attr::Int(v, _) => Some(v.to_string()),
                    _ => None,
                })
                .collect();
            match ints {
                Some(vs) => format!("array<i32: {}>", vs.join(", ")),
                None => {
                    let inner: Vec<String> = items.iter().map(fmt_attr).collect();
                    format!("[{}]", inner.join(", "))
                }
            }
        }
        Attr::Type(t) => t.to_string(),
    }
}

/// Assigns `%name` strings the way MLIR's asm printer does, per function.
struct Namer {
    /// name -> whether the bare form has been handed out already
    taken: HashMap<String, bool>,
    /// MLIR's single disambiguation counter, shared across all colliding names.
    next_suffix: u32,
    /// The separate counter for values with no name at all.
    next_anon: u32,
    assigned: HashMap<ValueId, String>,
}

impl Namer {
    fn new() -> Namer {
        Namer {
            taken: HashMap::new(),
            next_suffix: 0,
            next_anon: 0,
            assigned: HashMap::new(),
        }
    }

    /// The name `arith.constant` gives itself when nothing else names it, from
    /// `arith::ConstantOp::getAsmResultNames`: integers get `c<value>_<type>`, everything
    /// else `cst`.
    fn constant_hint(op: &Op) -> Option<String> {
        if op.name != "arith.constant" {
            return None;
        }
        match op.attrs.get("value") {
            Some(Attr::Int(v, ty)) => Some(if *v < 0 {
                format!("c_{}_{}", -v, ty)
            } else {
                format!("c{v}_{ty}")
            }),
            _ => Some("cst".to_string()),
        }
    }

    fn assign(&mut self, v: ValueId, hint: Option<String>) -> String {
        let name = match hint {
            Some(h) => {
                let bare_used = self.taken.get(&h).copied().unwrap_or(false);
                if bare_used {
                    let s = format!("{}_{}", h, self.next_suffix);
                    self.next_suffix += 1;
                    s
                } else {
                    self.taken.insert(h.clone(), true);
                    h
                }
            }
            None => {
                let s = format!("{}", self.next_anon);
                self.next_anon += 1;
                s
            }
        };
        self.assigned.insert(v, name.clone());
        name
    }

    fn has(&self, v: ValueId) -> bool {
        self.assigned.contains_key(&v)
    }

    fn get(&self, v: ValueId) -> String {
        self.assigned
            .get(&v)
            .cloned()
            .unwrap_or_else(|| format!("<undef:{}>", v.0))
    }
}

/// Collects `#locN` definitions in first-appearance order.
struct LocTable {
    order: Vec<Loc>,
    index: HashMap<Loc, usize>,
}

impl LocTable {
    fn new() -> LocTable {
        LocTable {
            order: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn intern(&mut self, loc: &Loc) -> usize {
        if let Some(i) = self.index.get(loc) {
            return *i;
        }
        // A name loc's inner loc gets its own entry, as MLIR prints it.
        if let Loc::Name(_, inner) = loc {
            self.intern(inner);
        }
        let i = self.order.len();
        self.order.push(loc.clone());
        self.index.insert(loc.clone(), i);
        i
    }

    /// A reference as it appears on an op: `loc(#locN)`.
    fn render_ref(&self, loc: &Loc) -> String {
        match self.index.get(loc) {
            Some(i) => format!("loc(#loc{i})"),
            None => "loc(unknown)".to_string(),
        }
    }

    /// A reference as it appears INSIDE another loc definition: `#locN`.
    fn render_inner(&self, loc: &Loc) -> String {
        match self.index.get(loc) {
            Some(i) => format!("#loc{i}"),
            None => "#loc".to_string(),
        }
    }

    fn render_def(&self, loc: &Loc) -> String {
        match loc {
            Loc::Unknown => "loc(unknown)".to_string(),
            Loc::File { file, line, col } => format!("loc(\"{file}\":{line}:{col})"),
            Loc::Name(n, inner) => format!("loc(\"{}\"({}))", n, self.render_inner(inner)),
        }
    }
}

/// Print a whole module as TTIR text.
pub fn print_module(m: &Module) -> String {
    let mut locs = LocTable::new();
    for f in &m.funcs {
        locs.intern(&f.loc);
        intern_region_locs(m, &f.body, &mut locs);
    }

    let mut body = String::new();
    let _ = writeln!(body, "module {{");
    for f in &m.funcs {
        print_func(m, f, &locs, &mut body);
    }
    let _ = writeln!(body, "}} {}", locs.render_ref(&m.loc));

    // Header: the loc definitions, then the module.
    let mut out = String::new();
    for (i, loc) in locs.order.iter().enumerate() {
        let _ = writeln!(out, "#loc{} = {}", i, locs.render_def(loc));
    }
    out.push_str(&body);
    out
}

fn intern_region_locs(m: &Module, r: &Region, locs: &mut LocTable) {
    for b in &r.blocks {
        for a in &b.args {
            locs.intern(m.loc_of(*a));
        }
        for op in &b.ops {
            locs.intern(&op.loc);
            for res in &op.results {
                locs.intern(m.loc_of(*res));
            }
            for sub in &op.regions {
                intern_region_locs(m, sub, locs);
            }
        }
    }
}

fn print_func(m: &Module, f: &Func, locs: &LocTable, out: &mut String) {
    let mut namer = Namer::new();
    let vis = match f.visibility {
        Visibility::Public => "public",
        Visibility::Private => "private",
    };
    let entry = f.body.blocks.first();
    let mut args = Vec::new();
    if let Some(b) = entry {
        for a in &b.args {
            let hint = m.loc_of(*a).name().map(|s| s.to_string());
            let name = namer.assign(*a, hint);
            args.push(format!(
                "%{}: {} {}",
                name,
                m.ty(*a),
                locs.render_def(m.loc_of(*a))
            ));
        }
    }
    let rets = if f.ret_types.is_empty() {
        String::new()
    } else {
        let tys: Vec<String> = f.ret_types.iter().map(Type::to_string).collect();
        format!(" -> ({})", tys.join(", "))
    };
    let _ = writeln!(
        out,
        "  tt.func {} @{}({}){} attributes {{noinline = {}}} {{",
        vis,
        symbol(&f.name),
        args.join(", "),
        rets,
        f.noinline
    );
    // EVERY BLOCK, not just the entry one. A generated `standard.*` helper -- and any function
    // whose `visit_Return` split the block -- has a second, unreachable block holding a
    // `ub.poison` and its own `tt.return`. Printing only the entry block LOST those two ops,
    // which `no_parser.rs::print_then_parse_round_trips_structurally` caught the moment it was
    // pointed at a golden with a helper in it.
    for (i, b) in f.body.blocks.iter().enumerate() {
        if i > 0 {
            let mut decls = Vec::new();
            for a in &b.args {
                let hint = m.loc_of(*a).name().map(|s| s.to_string());
                let name = namer.assign(*a, hint);
                decls.push(format!("%{}: {}", name, m.ty(*a)));
            }
            if decls.is_empty() {
                // MLIR's own annotation, and it is the form the goldens carry.
                let _ = writeln!(out, "  ^bb{i}:  // no predecessors");
            } else {
                let _ = writeln!(out, "  ^bb{i}({}):", decls.join(", "));
            }
        }
        print_block_ops(m, b, locs, &mut namer, 4, out);
    }
    let _ = writeln!(out, "  }} {}", locs.render_ref(&f.loc));
}

/// A symbol reference, quoted the way MLIR quotes one.
///
/// MLIR prints `@name` for a bare identifier and `@"name"` otherwise. Triton's mangled names
/// stop being bare as soon as a float constexpr reaches exponent form (`c1e-05` has a `-`), so
/// both forms occur; `ttir::parse::unquote_symbol` is the other half.
fn symbol(name: &str) -> String {
    let bare = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '.');
    if bare {
        name.to_string()
    } else {
        format!("\"{name}\"")
    }
}

fn print_block_ops(
    m: &Module,
    b: &Block,
    locs: &LocTable,
    namer: &mut Namer,
    indent: usize,
    out: &mut String,
) {
    for op in &b.ops {
        print_op(m, op, locs, namer, indent, out);
    }
}

fn print_op(
    m: &Module,
    op: &Op,
    locs: &LocTable,
    namer: &mut Namer,
    indent: usize,
    out: &mut String,
) {
    let pad = " ".repeat(indent);
    // Name the results first: MLIR names a result before printing its own operands only
    // in the sense that the result name is on the left.
    let mut res_names = Vec::new();
    for r in &op.results {
        let hint = m
            .loc_of(*r)
            .name()
            .map(|s| s.to_string())
            .or_else(|| Namer::constant_hint(op));
        res_names.push(namer.assign(*r, hint));
    }
    let lhs = if res_names.is_empty() {
        String::new()
    } else {
        let refs: Vec<String> = res_names.iter().map(|n| format!("%{n}")).collect();
        format!("{} = ", refs.join(", "))
    };

    let operands: Vec<String> = op
        .operands
        .iter()
        .map(|v| format!("%{}", namer.get(*v)))
        .collect();

    // Attributes other than the ones folded into the assembly form.
    let inline_attrs: Vec<String> = op
        .attrs
        .iter()
        // `value`, `predicate` and `callee` are folded into the assembly form below. `axis` is
        // folded ONLY for `tt.get_program_id`, where it is an [`Attr::Axis`] (`x`/`y`/`z`);
        // `tt.expand_dims` and `tt.reduce` carry an INTEGER axis that has to be printed, and
        // filtering by KEY alone silently dropped it from every one of them.
        .filter(|(k, v)| match k.as_str() {
            "value" | "predicate" | "callee" => false,
            "axis" => !matches!(v, Attr::Axis(_)),
            _ => true,
        })
        .map(|(k, v)| {
            if matches!(v, Attr::Unit) {
                k.clone()
            } else {
                format!("{k} = {}", fmt_attr(v))
            }
        })
        .collect();
    let attr_dict = if inline_attrs.is_empty() {
        String::new()
    } else {
        format!(" {{{}}}", inline_attrs.join(", "))
    };

    let res_tys: Vec<String> = op.results.iter().map(|r| m.ty(*r).to_string()).collect();

    // A few ops have a custom assembly form worth reproducing so the text reads like the
    // golden; everything else uses a generic-ish form. This is presentation only.
    let core = match op.name.as_str() {
        "arith.constant" => {
            let v = op.attrs.get("value").map(fmt_attr).unwrap_or_default();
            format!("arith.constant {v}")
        }
        "tt.get_program_id" => {
            let ax = match op.attrs.get("axis") {
                Some(Attr::Axis(a)) => *a,
                _ => "x",
            };
            format!("tt.get_program_id {ax} : {}", res_tys.join(", "))
        }
        "arith.cmpi" | "arith.cmpf" => {
            let p = match op.attrs.get("predicate") {
                Some(Attr::Pred(p)) => *p,
                _ => "?",
            };
            // MLIR prints a COMMA after the predicate: `arith.cmpi sle, %a, %b : i64`.
            format!(
                "{} {}, {} : {}",
                op.name,
                p,
                operands.join(", "),
                m.ty(op.operands[0])
            )
        }
        // MLIR's custom form prints the POINTEE type and then the BLOCK type, and the
        // shape/stride operands in brackets:
        //   tt.make_tensor_descriptor %p, [%n], [%s] : <f16>, <64xf16>
        // Printing the generic `: !tt.tensordesc<...>` instead made print -> parse
        // non-idempotent (the parser then read the tensordesc as the pointee), which
        // `tests/no_parser.rs::print_then_parse_round_trips_structurally` caught.
        "tt.make_tensor_descriptor" => {
            let n = (op.operands.len() - 1) / 2;
            let base = &operands[0];
            let shape = operands[1..1 + n].join(", ");
            let strides = operands[1 + n..].join(", ");
            let (elem, block) = match m.ty(op.results[0]) {
                Type::TensorDesc(s, e) => {
                    let dims: Vec<String> = s.iter().map(|d| format!("{d}x")).collect();
                    // The POINTEE is signless (`<i32>`) and the BLOCK element carries the
                    // signedness (`<64xsi32>`) -- MLIR prints those two differently on the
                    // same line. See `ttir::desc_block_elem`.
                    (
                        e.to_string(),
                        format!("{}{}", dims.join(""), crate::ttir::desc_block_elem(e)),
                    )
                }
                other => (other.to_string(), other.to_string()),
            };
            format!(
                "tt.make_tensor_descriptor {base}, [{shape}], [{strides}] : <{elem}>, <{block}>"
            )
        }
        // MLIR prints a call as `tt.call @sym(args) : (operand types) -> result types`, with
        // the callee as a SYMBOL rather than an attribute in the trailing dict. Printing it the
        // generic way put `{callee = "..."}` there instead, which `parse` then refused with
        // "tt.call without an @symbol" -- so the printer could not round-trip ANY module with a
        // user function call, attention included. Caught by extending
        // `no_parser.rs::print_then_parse_round_trips_structurally` past `vector_add`.
        // `scf.for` DECLARES ITS REGION'S BLOCK ARGUMENTS ON ITS OWN LINE:
        //
        //   %r = scf.for %n = %lb to %ub step %st iter_args(%acc_1 = %acc) -> (T)  : i32 {
        //
        // so the induction variable and the carries have to be named HERE, before the region is
        // walked, and the region walk must not rename them. Printing the generic form instead
        // produced `scf.for %0, %1, %2, %acc : T {`, which `parse` refuses ("could not read the
        // scf.for header") -- so the printer could not round-trip any looping module either.
        "scf.for" => {
            let region_args: Vec<ValueId> = op
                .regions
                .first()
                .and_then(|r| r.blocks.first())
                .map(|b| b.args.clone())
                .unwrap_or_default();
            let mut names = Vec::new();
            for a in &region_args {
                let hint = m.loc_of(*a).name().map(|s| s.to_string());
                names.push(namer.assign(*a, hint));
            }
            let iv = names.first().cloned().unwrap_or_else(|| "iv".to_string());
            let iv_ty = region_args
                .first()
                .map(|a| m.ty(*a).to_string())
                .unwrap_or_else(|| "i32".to_string());
            // operands are [lb, ub, step, init...]; region args are [iv, carry...].
            let inits = &operands[3.min(operands.len())..];
            let carries: Vec<String> = names
                .iter()
                .skip(1)
                .zip(inits.iter())
                .map(|(a, init)| format!("%{a} = {init}"))
                .collect();
            let iter_args = if carries.is_empty() {
                String::new()
            } else {
                format!(" iter_args({})", carries.join(", "))
            };
            let rets = if res_tys.is_empty() {
                String::new()
            } else {
                format!(" -> ({})", res_tys.join(", "))
            };
            format!(
                "scf.for %{iv} = {} to {} step {}{iter_args}{rets}  : {iv_ty}",
                operands.first().cloned().unwrap_or_default(),
                operands.get(1).cloned().unwrap_or_default(),
                operands.get(2).cloned().unwrap_or_default(),
            )
        }
        "tt.call" => {
            let sym = match op.attrs.get("callee") {
                Some(Attr::Str(s)) => symbol(s),
                _ => "<no callee>".to_string(),
            };
            let opnd_tys: Vec<String> =
                op.operands.iter().map(|v| m.ty(*v).to_string()).collect();
            let rets = if res_tys.is_empty() {
                "()".to_string()
            } else {
                res_tys.join(", ")
            };
            format!(
                "tt.call @{sym}({}) : ({}) -> {rets}",
                operands.join(", "),
                opnd_tys.join(", ")
            )
        }
        "tt.descriptor_load" => {
            let idx = operands[1..].join(", ");
            format!(
                "tt.descriptor_load {}[{}] : {} -> {}",
                operands[0],
                idx,
                m.ty(op.operands[0]),
                res_tys.join(", ")
            )
        }
        // The INDIRECT read. MLIR prints the operand types as a parenthesised list, unlike
        // `tt.descriptor_load`'s single type:
        //   tt.descriptor_gather %d[%ids, %y] : (!tt.tensordesc<1x128xf16>, tensor<64xi32>,
        //                                        i32) -> tensor<64x128xf16>
        "tt.descriptor_gather" => {
            let opnd_tys: Vec<String> = op.operands.iter().map(|v| m.ty(*v).to_string()).collect();
            format!(
                "tt.descriptor_gather {}[{}] : ({}) -> {}",
                operands[0],
                operands[1..].join(", "),
                opnd_tys.join(", "),
                res_tys.join(", ")
            )
        }
        "tt.descriptor_store" => {
            let last = operands.len() - 1;
            let idx = operands[1..last].join(", ");
            format!(
                "tt.descriptor_store {}[{}], {} : {}, {}",
                operands[0],
                idx,
                operands[last],
                m.ty(op.operands[0]),
                m.ty(op.operands[last])
            )
        }
        "arith.extsi" | "arith.extui" | "arith.trunci" | "arith.extf" | "arith.truncf"
        | "arith.sitofp" | "arith.uitofp" | "arith.fptosi" | "arith.fptoui" | "arith.bitcast" => {
            format!(
                "{} {} : {} to {}",
                op.name,
                operands.join(", "),
                m.ty(op.operands[0]),
                res_tys.join(", ")
            )
        }
        _ if op.results.is_empty() => {
            if operands.is_empty() {
                op.name.clone()
            } else {
                let opnd_tys: Vec<String> =
                    op.operands.iter().map(|v| m.ty(*v).to_string()).collect();
                format!(
                    "{} {} : {}",
                    op.name,
                    operands.join(", "),
                    opnd_tys.join(", ")
                )
            }
        }
        _ => {
            if operands.is_empty() {
                format!("{} : {}", op.name, res_tys.join(", "))
            } else {
                format!(
                    "{} {} : {}",
                    op.name,
                    operands.join(", "),
                    res_tys.join(", ")
                )
            }
        }
    };

    if op.regions.is_empty() {
        let _ = writeln!(
            out,
            "{pad}{lhs}{core}{attr_dict} {}",
            locs.render_ref(&op.loc)
        );
    } else {
        let _ = writeln!(out, "{pad}{lhs}{core}{attr_dict} {{");
        for r in &op.regions {
            for (bi, b) in r.blocks.iter().enumerate() {
                // Region block arguments (e.g. scf.for's induction var + carries).
                // `scf.for` NAMES THEM ALREADY, on its own header line, so re-assigning here
                // would hand the body a different name from the one the header declared.
                let mut decls = Vec::new();
                for a in &b.args {
                    if !namer.has(*a) {
                        let hint = m.loc_of(*a).name().map(|s| s.to_string());
                        let nm = namer.assign(*a, hint);
                        decls.push(format!("%{}: {} {}", nm, m.ty(*a), locs.render_def(&Loc::Unknown)));
                    }
                }
                // MLIR prints a region's entry-block label ONLY when the block has arguments,
                // and `tt.reduce`'s does: `^bb0(%arg1: f16, %arg2: f16):`. Omitting it lost
                // those arguments on the way back through `parse`, which left the combiner's
                // `tt.call` with no operands at all.
                if !decls.is_empty() {
                    let _ = writeln!(
                        out,
                        "{}^bb{bi}({}):",
                        " ".repeat(indent + 2),
                        decls.join(", ")
                    );
                }
                print_block_ops(m, b, locs, namer, indent + 2, out);
            }
        }
        let _ = writeln!(out, "{pad}}} {}", locs.render_ref(&op.loc));
    }
}
