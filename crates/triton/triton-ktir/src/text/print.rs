// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! A TEST INSTRUMENT: [`Module`] -> KTIR text.
//!
//! Two jobs, and the second is why this is more than a debug dump:
//!
//! 1. Make a golden diff READABLE side by side with the C++ output.
//! 2. Produce text `triton-superdsc-lower` can actually READ, so the port's own KTIR
//!    can be driven end to end through `dxp_standalone`. That is the only check that
//!    says the output is not merely structurally equal to the C++'s but
//!    DOWNSTREAM-VALID.
//!
//! Job 2 is what forces the per-op CUSTOM ASSEMBLY FORMATS below. `ktdp.*` and
//! `ktdf.*` do not print in the generic form -- `construct_memory_view` puts its
//! extents in `sizes:`/`strides:` keywords, `corelet_plan` puts its pattern before the
//! body, `corelet` puts its index there -- and a printer that emits the generic form
//! produces text the next stage refuses. It is not aiming at byte equality with MLIR
//! (value numbering and constant order are MLIR's internals); [`super::diff`] is what
//! makes the comparison rigorous.

use std::collections::{HashMap, HashSet};

use crate::ir::*;

/// Print a whole module.
pub fn print(module: &Module) -> String {
    let names = Names::of(module);
    let aliases = Aliases::of(module);
    let mut out = String::new();
    // Affine maps and sets print as ALIASES declared before the module, the way MLIR
    // does it -- and the way `triton-superdsc-lower` REQUIRES: it resolves
    // `access_tile_order` through an alias table and refuses an inline `affine_map<..>`
    // by name ("the affine_map alias was not found in this KTIR"). Emitting the inline
    // form makes text that is valid MLIR and that the next stage cannot read.
    out.push_str(&aliases.declarations());
    let attrs = print_attr_dict_of(&module.attrs, &[], &aliases);
    if attrs.is_empty() {
        out.push_str("module {\n");
    } else {
        out.push_str(&format!("module attributes {{{attrs}}} {{\n"));
    }
    for op in &module.ops {
        print_op(module, &names, &aliases, op, 1, &mut out);
    }
    out.push_str("}\n");
    out
}

/// AFFINE MAP / SET ALIASES, named the way MLIR names them: `#map`, `#map1`, ... and
/// `#set`, `#set1`, ..., in first-use order.
struct Aliases {
    map: HashMap<String, String>,
    /// Declaration lines, in the order they are numbered.
    decls: Vec<String>,
}

impl Aliases {
    fn of(module: &Module) -> Aliases {
        let mut a = Aliases { map: HashMap::new(), decls: Vec::new() };
        let mut maps = 0usize;
        let mut sets = 0usize;
        let mut visit = |attrs: &[(AttrKey, Attr)], a: &mut Aliases| {
            for (_, v) in attrs {
                let (kind, body) = match v {
                    Attr::AffineMap(b) => ("map", b.clone()),
                    Attr::AffineSet(b) => ("set", b.clone()),
                    Attr::AffineMapList(bs) => {
                        for b in bs {
                            let key = format!("map:{b}");
                            if !a.map.contains_key(&key) {
                                let name = if maps == 0 {
                                    "#map".to_string()
                                } else {
                                    format!("#map{maps}")
                                };
                                maps += 1;
                                a.decls.push(format!("{name} = affine_map<{b}>\n"));
                                a.map.insert(key, name);
                            }
                        }
                        continue;
                    }
                    _ => continue,
                };
                let key = format!("{kind}:{body}");
                if a.map.contains_key(&key) {
                    continue;
                }
                let (n, name) = if kind == "map" {
                    let n = maps;
                    maps += 1;
                    (n, if n == 0 { "#map".to_string() } else { format!("#map{n}") })
                } else {
                    let n = sets;
                    sets += 1;
                    (n, if n == 0 { "#set".to_string() } else { format!("#set{n}") })
                };
                let _ = n;
                a.decls.push(format!("{name} = affine_{kind}<{body}>\n"));
                a.map.insert(key, name);
            }
        };
        visit(&module.attrs, &mut a);
        for op in module.ops_deep() {
            let attrs = op.attrs.clone();
            visit(&attrs, &mut a);
        }
        a
    }

    fn declarations(&self) -> String {
        self.decls.concat()
    }

    /// The alias for one affine attribute, or its inline form when it has none.
    fn name(&self, kind: &str, body: &str) -> String {
        match self.map.get(&format!("{kind}:{body}")) {
            Some(n) => n.clone(),
            None => format!("affine_{kind}<{body}>"),
        }
    }
}

/// UNIQUE PRINTED NAMES.
///
/// [`Module::hints`] is a diagnostic, not an identity: several values legitimately
/// carry the SAME hint, because a pass that rewrites `%desc_q` into a cast, a view and
/// a second cast names all three after the value they came from -- which is what MLIR
/// does too. MLIR then disambiguates ON PRINT with a `_N` suffix.
///
/// Skipping that step produces text in which three different values are all
/// `%desc_q`, and the next stage resolves whichever it saw last. That is not a
/// cosmetic defect: it made the grid loop read `%start_m to %start_m step %start_m`
/// and the SuperDSC lowering refuse with "`scf.for` step must be positive" -- a true
/// complaint about a printer bug.
struct Names {
    printed: HashMap<Ssa, String>,
}

impl Names {
    fn of(module: &Module) -> Names {
        let mut printed: HashMap<Ssa, String> = HashMap::new();
        let mut taken: HashSet<String> = HashSet::new();
        let mut assign = |v: Ssa, printed: &mut HashMap<Ssa, String>| {
            if printed.contains_key(&v) {
                return;
            }
            let base = module.hint(v);
            let mut name = base.clone();
            let mut k = 0usize;
            while taken.contains(&name) {
                name = format!("{base}_{k}");
                k += 1;
            }
            taken.insert(name.clone());
            printed.insert(v, name);
        };
        // Function arguments first, then every definition in program order, then each
        // region's block arguments -- the order MLIR names them in.
        for op in module.ops_deep() {
            if op.kind == OpKind::TtFunc {
                if let Some(r) = op.regions.first() {
                    for (v, _) in &r.args {
                        assign(*v, &mut printed);
                    }
                }
            }
        }
        for op in module.ops_deep() {
            for r in &op.results {
                assign(*r, &mut printed);
            }
            for region in &op.regions {
                for (v, _) in &region.args {
                    assign(*v, &mut printed);
                }
            }
        }
        Names { printed }
    }

    fn get(&self, v: Ssa) -> String {
        match self.printed.get(&v) {
            Some(s) => s.clone(),
            None => format!("undef{}", v.0),
        }
    }
}

fn indent(n: usize, out: &mut String) {
    for _ in 0..n {
        out.push_str("  ");
    }
}

/// The type of an operand, for the formats that state it.
fn ty_of(module: &Module, v: Ssa) -> String {
    module.type_of(v).map(|t| print_type(&t)).unwrap_or_else(|| "<?>".into())
}

/// An access tile's ABBREVIATED spelling, which is what `ktdp.load`/`store` use.
fn abbrev(module: &Module, v: Ssa) -> String {
    match module.type_of(v) {
        Some(IrType::AccessTile { dims }) => format!(
            "<{}xindex>",
            dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
        ),
        other => other.map(|t| print_type(&t)).unwrap_or_else(|| "<?>".into()),
    }
}

fn print_op(
    module: &Module,
    names: &Names,
    aliases: &Aliases,
    op: &Op,
    depth: usize,
    out: &mut String,
) {
    let n = |v: Ssa| format!("%{}", names.get(v));
    let list = |vs: &[Ssa]| vs.iter().map(|v| n(*v)).collect::<Vec<_>>().join(", ");

    indent(depth, out);
    if !op.results.is_empty() {
        out.push_str(&list(&op.results));
        out.push_str(" = ");
    }

    // Whether this op opens a `{ ... }` body after its header.
    let mut has_body = false;

    match op.kind {
        OpKind::TtFunc => {
            let name = op.attr(&AttrKey::SymName).and_then(|a| a.as_str()).unwrap_or("kernel");
            let args: Vec<String> = op
                .regions
                .first()
                .map(|r| {
                    r.args.iter().map(|(v, t)| format!("{}: {}", n(*v), print_type(t))).collect()
                })
                .unwrap_or_default();
            out.push_str(&format!(
                "tt.func public @{name}({}) attributes {{noinline = false}}",
                args.join(", ")
            ));
            has_body = true;
        }
        // `func.func` prints the same shape `tt.func` does, minus the `public` and the
        // `noinline` attribute: `func.func @name(%a: T) attributes {grid = ...} {`.
        // Without this arm the body SILENTLY DROPPED (the default arm prints the
        // attribute dictionary and no region), which made the text instrument useless
        // on every post-`ToSchedulerKTIR` module.
        OpKind::FuncFunc => {
            let name = op.attr(&AttrKey::SymName).and_then(|a| a.as_str()).unwrap_or("kernel");
            let args: Vec<String> = op
                .regions
                .first()
                .map(|r| {
                    r.args.iter().map(|(v, t)| format!("{}: {}", n(*v), print_type(t))).collect()
                })
                .unwrap_or_default();
            out.push_str(&format!("func.func @{name}({})", args.join(", ")));
            has_body = true;
        }
        OpKind::ScfFor => {
            let r = &op.regions[0];
            out.push_str(&format!(
                "scf.for {} = {} to {} step {}",
                n(r.args[0].0),
                n(op.operands[0]),
                n(op.operands[1]),
                n(op.operands[2])
            ));
            if op.operands.len() > 3 {
                let pairs: Vec<String> = r.args[1..]
                    .iter()
                    .zip(&op.operands[3..])
                    .map(|((a, _), init)| format!("{} = {}", n(*a), n(*init)))
                    .collect();
                let tys: Vec<String> = op.result_types.iter().map(print_type).collect();
                out.push_str(&format!(
                    " iter_args({}) -> ({})  : {}",
                    pairs.join(", "),
                    tys.join(", "),
                    print_type(&r.args[0].1)
                ));
            }
            // An INDEX-typed loop with no iter_args prints NO trailing type:
            // `scf.for %i = %a to %b step %c {`. Emitting `: index` is not MLIR's form.
            has_body = true;
        }
        OpKind::ArithConstant => {
            let val = match op.attr(&AttrKey::Value) {
                Some(Attr::Int(i)) => i.to_string(),
                Some(Attr::Float(f)) => print_float(*f),
                Some(Attr::SplatFloat(f)) => format!("dense<{}>", print_float(*f)),
                Some(Attr::Verbatim(s)) => s.clone(),
                _ => "<?>".into(),
            };
            out.push_str(&format!(
                "arith.constant {val} : {}",
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::UnrealizedConversionCast => {
            out.push_str(&format!(
                "builtin.unrealized_conversion_cast {} : {} to {}",
                list(&op.operands),
                ty_of(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::KtdpGetComputeTileId => out.push_str("ktdp.get_compute_tile_id : index"),
        OpKind::KtdpConstructMemoryView => {
            let ints = |k: &AttrKey| {
                op.attr(k)
                    .and_then(|a| a.as_int_list())
                    .map(|s| {
                        s.iter()
                            .map(|x| if *x == DYNAMIC { "?".into() } else { x.to_string() })
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default()
            };
            // THE DYNAMIC EXTENTS PRINT INSIDE `sizes:`, not in the operand list, which is
            // where the C++'s custom printer puts them: `%base, sizes: [%n_as_index]`. The
            // static ones print as literals in the same list, so a mixed view interleaves --
            // no fixture has one, and `build_base_memory_view` emits all-static or all-dynamic
            // for a given descriptor, so the two lists are concatenated rather than merged.
            let dynamic_ops: Vec<String> =
                op.operands[1..].iter().map(|v| n(*v)).collect();
            let sizes = {
                let statics = ints(&AttrKey::Shape);
                let mut parts: Vec<String> = Vec::new();
                if !dynamic_ops.is_empty() {
                    parts.extend(dynamic_ops.iter().cloned());
                }
                if !statics.is_empty() {
                    parts.push(statics);
                }
                parts.join(", ")
            };
            out.push_str(&format!(
                "ktdp.construct_memory_view {}, sizes: [{sizes}], strides: [{}]",
                n(op.operands[0]),
                ints(&AttrKey::Strides)
            ));
            let mut parts: Vec<String> = Vec::new();
            let d = print_attr_dict_of(
                &op.attrs,
                &[AttrKey::Shape, AttrKey::Strides, AttrKey::MemorySpace],
                aliases,
            );
            if !d.is_empty() {
                parts.push(d);
            }
            if let Some(s) = op.attr(&AttrKey::MemorySpace).and_then(|a| a.as_str()) {
                parts.push(format!("memory_space = #ktdp.spyre_memory_space<{s}>"));
            }
            if !parts.is_empty() {
                out.push_str(&format!(" {{{}}}", parts.join(", ")));
            }
            out.push_str(&format!(
                " : {}",
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::KtdpConstructAccessTile => {
            out.push_str(&format!(
                "ktdp.construct_access_tile {}[{}]",
                n(op.operands[0]),
                list(&op.operands[1..])
            ));
            // `base_map` is ELIDED when it is the identity, matching the custom
            // printer -- the parser synthesizes it back.
            let d = print_attr_dict_of(&op.attrs, &[AttrKey::BaseMap], aliases);
            if !d.is_empty() {
                out.push_str(&format!(" {{{d}}}"));
            }
            out.push_str(&format!(
                " : {} -> {}",
                ty_of(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::KtdpLoad => {
            out.push_str(&format!(
                "ktdp.load {} : {} -> {}",
                n(op.operands[0]),
                abbrev(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::KtdpStore => {
            out.push_str(&format!(
                "ktdp.store {}, {} : {}, {}",
                n(op.operands[0]),
                n(op.operands[1]),
                ty_of(module, op.operands[0]),
                abbrev(module, op.operands[1])
            ));
        }
        OpKind::KtdfCoreletPlan => {
            let p = op.attr(&AttrKey::Pattern).and_then(|a| a.as_str()).unwrap_or("split");
            out.push_str(&format!("ktdf.corelet_plan pattern = \"{p}\""));
            has_body = true;
        }
        OpKind::KtdfCorelet => {
            let i = op.attr(&AttrKey::Index).and_then(|a| a.as_int()).unwrap_or(0);
            let d = print_attr_dict_of(&op.attrs, &[AttrKey::Index], aliases);
            out.push_str(&format!("ktdf.corelet {i} {{{d}}}"));
        }
        OpKind::LinalgMatmul => {
            let ins: Vec<String> = op.operands[..2].iter().map(|v| n(*v)).collect();
            let in_tys: Vec<String> = op.operands[..2].iter().map(|v| ty_of(module, *v)).collect();
            out.push_str(&format!(
                "linalg.matmul ins({} : {}) outs({} : {}) -> {}",
                ins.join(", "),
                in_tys.join(", "),
                n(op.operands[2]),
                ty_of(module, op.operands[2]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::TtReduce => {
            let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(0);
            out.push_str(&format!(
                "\"tt.reduce\"({}) <{{axis = {axis} : i32}}> ({{\n",
                list(&op.operands)
            ));
            let r = &op.regions[0];
            indent(depth, out);
            let args: Vec<String> =
                r.args.iter().map(|(v, t)| format!("{}: {}", n(*v), print_type(t))).collect();
            out.push_str(&format!("^bb0({}):\n", args.join(", ")));
            for inner in &r.ops {
                print_op(module, names, aliases, inner, depth + 1, out);
            }
            indent(depth, out);
            out.push_str(&format!(
                "}}) : ({}) -> {}\n",
                ty_of(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
            return;
        }
        OpKind::TtReduceReturn => {
            out.push_str(&format!(
                "tt.reduce.return {} : {}",
                list(&op.operands),
                ty_of(module, op.operands[0])
            ));
        }
        OpKind::TtExpandDims => {
            let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(0);
            out.push_str(&format!(
                "tt.expand_dims {} {{axis = {axis} : i32}} : {} -> {}",
                list(&op.operands),
                ty_of(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::TtBroadcast
        | OpKind::TtTrans
        | OpKind::ArithExtf
        | OpKind::ArithTruncf
        | OpKind::ArithIndexCast => {
            // `tt.broadcast`/`tt.trans` use `->`; the arith casts use `to`.
            let sep = match op.kind {
                OpKind::TtBroadcast | OpKind::TtTrans => "->",
                _ => "to",
            };
            let d = print_attr_dict_of(&op.attrs, &[], aliases);
            let dict = if d.is_empty() { String::new() } else { format!(" {{{d}}}") };
            out.push_str(&format!(
                "{}{dict} {} : {} {sep} {}",
                op.kind.spelling(),
                list(&op.operands),
                ty_of(module, op.operands[0]),
                op.result_type().map(print_type).unwrap_or_else(|| "?".into())
            ));
        }
        OpKind::ScfYield | OpKind::TtReturn => {
            if op.operands.is_empty() {
                out.push_str(op.kind.spelling());
            } else {
                let tys: Vec<String> = op.operands.iter().map(|v| ty_of(module, *v)).collect();
                out.push_str(&format!(
                    "{} {} : {}",
                    op.kind.spelling(),
                    list(&op.operands),
                    tys.join(", ")
                ));
            }
        }
        _ => {
            // The default: `dialect.op %a, %b {attrs} : T`, which is the form every
            // `arith.*` / `math.*` / `tensor.splat` op uses.
            out.push_str(op.kind.spelling());
            if !op.operands.is_empty() {
                out.push(' ');
                out.push_str(&list(&op.operands));
            }
            let d = print_attr_dict_of(&op.attrs, &[], aliases);
            if !d.is_empty() {
                out.push_str(&format!(" {{{d}}}"));
            }
            if let Some(t) = op.result_type() {
                out.push_str(&format!(" : {}", print_type(t)));
            }
        }
    }

    if has_body {
        out.push_str(" {\n");
        for r in &op.regions {
            for inner in &r.ops {
                print_op(module, names, aliases, inner, depth + 1, out);
            }
        }
        indent(depth, out);
        out.push('}');
        // A POST-BODY attribute dictionary, for the ops whose assemblyFormat puts
        // `attr-dict` after `$body` -- `ktdf.corelet_plan`'s `work_division`.
        if op.kind == OpKind::KtdfCoreletPlan {
            let d = print_attr_dict_of(&op.attrs, &[AttrKey::Pattern], aliases);
            if !d.is_empty() {
                out.push_str(&format!(" {{{d}}}"));
            }
        }
        out.push('\n');
    } else {
        out.push('\n');
    }
}

/// An attribute dictionary, skipping the keys a custom format already printed.
fn print_attr_dict_of(
    attrs: &[(AttrKey, Attr)],
    skip: &[AttrKey],
    aliases: &Aliases,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in attrs {
        if skip.contains(k) || matches!(k, AttrKey::SymName | AttrKey::Value | AttrKey::Noinline) {
            continue;
        }
        parts.push(match v {
            Attr::Unit => k.spelling().to_string(),
            other => format!("{} = {}", k.spelling(), print_attr_aliased(other, aliases)),
        });
    }
    parts.sort();
    parts.join(", ")
}

/// Like [`print_attr`], but affine maps and sets print as their ALIAS.
fn print_attr_aliased(a: &Attr, aliases: &Aliases) -> String {
    match a {
        Attr::AffineMap(s) => aliases.name("map", s),
        Attr::AffineSet(s) => aliases.name("set", s),
        Attr::AffineMapList(v) => format!(
            "[{}]",
            v.iter().map(|s| aliases.name("map", s)).collect::<Vec<_>>().join(", ")
        ),
        other => print_attr(other),
    }
}

pub fn print_attr(a: &Attr) -> String {
    match a {
        Attr::Unit => String::new(),
        Attr::Int(i) => i.to_string(),
        Attr::IntList(v) => format!(
            "array<i64: {}>",
            v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
        ),
        Attr::Float(f) | Attr::SplatFloat(f) => print_float(*f),
        Attr::Str(s) => format!("\"{s}\""),
        Attr::StrList(v) => format!("[{}]", v.join(", ")),
        Attr::Bool(b) => b.to_string(),
        Attr::AffineMap(s) => format!("affine_map<{s}>"),
        Attr::AffineMapList(v) => format!(
            "[{}]",
            v.iter().map(|s| format!("affine_map<{s}>")).collect::<Vec<_>>().join(", ")
        ),
        Attr::AffineSet(s) => format!("affine_set<{s}>"),
        Attr::Verbatim(s) => s.clone(),
    }
}

/// MLIR's float spelling.
///
/// MEASURED FROM THE GOLDEN, not guessed, because two plausible rules disagree on
/// exactly the constant that matters. The f16 nearest `0.127517432` is
/// `0.1275634765625`; a plain `%.6e` of that prints `1.275635e-01`, but the C++ KTIR
/// prints **`1.275630e-01`**. So MLIR rounds to SIX SIGNIFICANT DIGITS (`1.27563`) and
/// then pads the mantissa out to six decimal places -- which is
/// `APFloat::toString(/*FormatPrecision=*/6, /*FormatMaxPadding=*/0)` followed by the
/// printer's fixed-width exponent form. The same rule reproduces the fixture's other
/// constants (`1.000000e+00`, `0.000000e+00`) and the hex form for a special
/// (`0xFC00`, the flash kernel's `m_i` init).
///
/// Getting this wrong is not cosmetic: the diff would report a constant difference on
/// every scaled attention run and the port would look broken where it is not.
pub fn print_float(f: FloatBits) -> String {
    let v = f.as_f64();
    if v.is_nan() || v.is_infinite() {
        return format!("0x{:04X}", f.bits);
    }
    let six = format!("{v:.5e}");
    let (mantissa, exp) = six.split_once('e').expect("Rust always emits an exponent");
    let exp: i32 = exp.parse().unwrap_or(0);
    let mut m = mantissa.to_string();
    if !m.contains('.') {
        m.push('.');
    }
    let decimals = m.len() - m.find('.').expect("just ensured") - 1;
    for _ in decimals..6 {
        m.push('0');
    }
    format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

pub fn print_type(t: &IrType) -> String {
    let dims = |d: &[i64]| -> String {
        d.iter()
            .map(|x| if *x == DYNAMIC { "?".to_string() } else { x.to_string() })
            .collect::<Vec<_>>()
            .join("x")
    };
    match t {
        IrType::Tensor { dims: d, elem } => {
            if d.is_empty() {
                format!("tensor<{}>", elem.spelling())
            } else {
                format!("tensor<{}x{}>", dims(d), elem.spelling())
            }
        }
        IrType::MemRef { dims: d, elem } => format!("memref<{}x{}>", dims(d), elem.spelling()),
        IrType::AccessTile { dims: d } => format!("!ktdp.access_tile<{}xindex>", dims(d)),
        IrType::Ptr { elem } => format!("!tt.ptr<{}>", elem.spelling()),
        IrType::TensorDesc { dims: d, elem } => {
            format!("!tt.tensordesc<{}x{}>", dims(d), elem.spelling())
        }
        IrType::Index => "index".to_string(),
        IrType::Scalar(d) => d.spelling().to_string(),
        IrType::Verbatim(s) => s.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    #[test]
    fn mlir_float_spelling_round_trips_the_flash_constants() {
        assert_eq!(print_float(FloatBits::f16_from_f32(0.127_517_43)), "1.275630e-01");
        // And the f32 the ttir carries, before the re-round.
        assert_eq!(print_float(FloatBits::f32(0.127_517_43)), "1.275170e-01");
        assert_eq!(print_float(FloatBits::f16_from_f32(0.0)), "0.000000e+00");
        assert_eq!(print_float(FloatBits::f16_from_f32(1.0)), "1.000000e+00");
        assert_eq!(print_float(FloatBits { bits: 0xFC00, width: 16 }), "0xFC00");
    }

    #[test]
    fn types_print_the_way_mlir_prints_them() {
        assert_eq!(
            print_type(&IrType::Tensor { dims: vec![64, 128], elem: DType::F16 }),
            "tensor<64x128xf16>"
        );
        assert_eq!(
            print_type(&IrType::AccessTile { dims: vec![128, 64] }),
            "!ktdp.access_tile<128x64xindex>"
        );
        assert_eq!(
            print_type(&IrType::MemRef { dims: vec![512, 128], elem: DType::F16 }),
            "memref<512x128xf16>"
        );
    }

    /// THE REGRESSION THIS PRINTER EXISTS NOT TO HAVE. Several values share a hint, so
    /// the printed names must be uniquified -- otherwise the grid loop reads
    /// `%start_m to %start_m step %start_m` and the next stage refuses with "step must
    /// be positive", which is a true complaint about a printer bug.
    #[test]
    fn values_sharing_a_hint_get_distinct_printed_names() {
        let mut m = Module::new();
        let a = m.fresh_named("start_m");
        let b = m.fresh_named("start_m");
        let c = m.fresh_named("start_m");
        let iv = m.fresh_named("start_m");
        m.ops.push(
            Op::new(OpKind::TtFunc)
                .with_attr(AttrKey::SymName, Attr::Str("k".into()))
                .with_region(Region {
                    args: vec![],
                    ops: vec![
                        Op::new(OpKind::KtdpGetComputeTileId).with_result(a, IrType::Index),
                        Op::new(OpKind::ArithConstant)
                            .with_result(b, IrType::Index)
                            .with_attr(AttrKey::Value, Attr::Int(8)),
                        Op::new(OpKind::ArithConstant)
                            .with_result(c, IrType::Index)
                            .with_attr(AttrKey::Value, Attr::Int(32)),
                        Op::new(OpKind::ScfFor).with_operands([a, b, c]).with_region(Region {
                            args: vec![(iv, IrType::Index)],
                            ops: vec![],
                        }),
                        Op::new(OpKind::TtReturn),
                    ],
                }),
        );
        let text = print(&m);
        let names = Names::of(&m);
        let set: std::collections::HashSet<String> =
            [a, b, c, iv].iter().map(|v| names.get(*v)).collect();
        assert_eq!(set.len(), 4, "four values, four distinct names: {set:?}");
        // The three loop operands must be three DIFFERENT tokens. Checking for the
        // substring `"%start_m to %start_m"` would not do: `%start_m to %start_m_0`
        // contains it and is perfectly well-formed.
        let loop_line = text
            .lines()
            .find(|l| l.contains("scf.for"))
            .expect("the loop is printed");
        let toks: Vec<&str> = loop_line
            .split_whitespace()
            .filter(|t| t.starts_with('%'))
            .collect();
        assert_eq!(toks.len(), 4, "iv + three bounds: {toks:?}");
        let uniq: std::collections::HashSet<&&str> = toks.iter().collect();
        assert_eq!(uniq.len(), 4, "every loop operand distinct: {loop_line}");
    }

    /// A ROUND TRIP: parse, print, parse again, and the STRUCTURE agrees. If the
    /// printer dropped a field or an op, the second parse differs -- which is what
    /// makes the printer usable as a measuring instrument.
    #[test]
    fn parse_print_parse_is_structurally_stable() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %s = arith.constant 1.275630e-01 : f16
    %sp = tensor.splat %s : tensor<64xf16>
    %v = ktdp.construct_memory_view %q, sizes: [64], strides: [1] {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<64xf16>
    %t = ktdp.construct_access_tile %v[%c0] {access_tile_order = affine_map<(d0) -> (d0)>, access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>} : memref<64xf16> -> !ktdp.access_tile<64xindex>
    %l = ktdp.load %t : <64xindex> -> tensor<64xf16>
    ktdp.store %l, %t : tensor<64xf16>, <64xindex>
    tt.return
  }
}
";
        let a = parse::parse(src).unwrap();
        let printed = print(&a);
        let b = parse::parse(&printed).unwrap();
        assert_eq!(
            a.census(),
            b.census(),
            "the census must survive a print/parse round trip\n--- printed ---\n{printed}"
        );
        assert_eq!(printed, print(&b), "printing is idempotent\n{printed}");
        let d = super::super::diff::diff(&a, &b);
        assert!(d.is_empty(), "the round trip changed the structure: {d:?}\n{printed}");
    }
}
