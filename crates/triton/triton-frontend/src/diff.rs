//! Structural comparison of two TTIR [`Module`]s, reported FIELD BY FIELD.
//!
//! # What is compared, and what is deliberately not.
//!
//! COMPARED, because all of it is a fact about the program the AST describes:
//!   * the set of functions, their visibility, `noinline`, argument and result types;
//!   * per function, the OP SEQUENCE in order, and for each op:
//!       - the op name (`arith.addf`, `tt.descriptor_load`, ...);
//!       - every result type, spelled out (element type AND shape);
//!       - every attribute, by key, including constant values and compare predicates;
//!       - operand LINKAGE, normalized to `arg N` / `op K result R` so def-use is
//!         compared without depending on printed names;
//!       - the `loc` NAME, which is the Python variable the value was assigned to
//!         (attached by Triton's `_maybe_set_loc_to_name`) and so is AST-derived;
//!       - nested regions, recursively;
//!   * the OP HISTOGRAM, so a count mismatch is reported even when the first positional
//!     difference would otherwise mask it.
//!
//! NOT COMPARED, because none of it is a fact about the program:
//!   * SSA value names and MLIR's `_0`/`_1` disambiguation suffixes;
//!   * `#locN` numbering, and the file path and line/column inside a location (the goldens
//!     embed an absolute path from the machine that generated them);
//!   * whitespace and printed op assembly form.
//!
//! A diff that passes while comparing nothing is the failure mode that matters here, so
//! `tests/golden_diff.rs` plants a difference of each kind and asserts this module NAMES
//! it. Without that control the rest of the file proves nothing.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ttir::{Attr, Loc, Module, Op, Type, ValueId};

/// How a value is referred to, independent of its printed name.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Ref {
    /// Function (or region) argument N.
    Arg(usize),
    /// Result R of the op at linearized index K.
    Res(usize, usize),
    /// A value defined nowhere the walk reached.
    Dangling,
}

impl std::fmt::Display for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ref::Arg(n) => write!(f, "arg{n}"),
            Ref::Res(k, r) => write!(f, "op{k}#{r}"),
            Ref::Dangling => f.write_str("<dangling>"),
        }
    }
}

/// One op, flattened for comparison.
#[derive(Clone, Debug)]
struct FlatOp {
    /// `""` for a top-level op; `"3.0"` for region 0 of the op at index 3, etc.
    path: String,
    depth: usize,
    name: String,
    result_types: Vec<String>,
    operands: Vec<Ref>,
    attrs: BTreeMap<String, String>,
    /// The name on the OP's own location.
    loc_name: Option<String>,
    /// The name on each RESULT's location.
    ///
    /// Compared SEPARATELY from `loc_name` rather than as a fallback. They are the same
    /// object everywhere the front end sets them, so a fallback looked equivalent -- but a
    /// fallback means a wrong result name is invisible whenever the op name is right, and a
    /// planted-difference control caught exactly that. Compare both.
    result_loc_names: Vec<Option<String>>,
}

fn attr_text(a: &Attr) -> String {
    match a {
        Attr::Int(v, t) => format!("int {v} : {t}"),
        // Floats compare AT THEIR TYPE. MLIR prints an f32 constant with only as many digits
        // as f32 needs (`1.44269502`), so parsing that gives a different f64 from the one our
        // builder holds (`1.4426950216293335`) even though both denote the SAME f32. Rounding
        // both sides to the attribute's type before formatting is what makes the comparison
        // about the value rather than about the printer. NaN is compared by bit pattern,
        // because no two NaNs are `==`.
        Attr::Float(b, t) => {
            let v = crate::ttir::F64Bits::rounded(b.get(), t).get();
            if v.is_nan() {
                format!("float nan:{:#x} : {t}", b.0)
            } else {
                format!("float {v:?} : {t}")
            }
        }
        Attr::DenseSplat(inner, t) => format!("dense<{}> : {t}", attr_text(inner)),
        Attr::Bool(b) => format!("bool {b}"),
        Attr::Str(s) => format!("str {s:?}"),
        Attr::Unit => "unit".to_string(),
        Attr::Pred(p) => format!("pred {p}"),
        Attr::Axis(a) => format!("axis {a}"),
        Attr::Array(items) => {
            let inner: Vec<String> = items.iter().map(attr_text).collect();
            format!("[{}]", inner.join(", "))
        }
        Attr::Type(t) => format!("type {t}"),
    }
}

struct Flattener<'a> {
    m: &'a Module,
    out: Vec<FlatOp>,
    /// value -> how to refer to it
    refs: std::collections::HashMap<ValueId, Ref>,
}

impl<'a> Flattener<'a> {
    fn walk_region(&mut self, r: &crate::ttir::Region, path: &str, depth: usize) {
        for (bi, b) in r.blocks.iter().enumerate() {
            // The BLOCK INDEX is part of an op's path, so an op that moved between blocks --
            // for instance a `tt.return` that ended up in the unreachable block instead of
            // the reachable one -- is reported rather than silently accepted.
            let path = if r.blocks.len() > 1 {
                format!("{path}^bb{bi}")
            } else {
                path.to_string()
            };
            let path = path.as_str();
            for (i, a) in b.args.iter().enumerate() {
                // A region's block arguments continue the argument numbering of their own
                // block, which is what makes scf.for's `(iv, carries...)` comparable.
                self.refs.entry(*a).or_insert(Ref::Arg(i));
            }
            for op in &b.ops {
                let k = self.out.len();
                for (ri, res) in op.results.iter().enumerate() {
                    self.refs.insert(*res, Ref::Res(k, ri));
                }
                let flat = FlatOp {
                    path: path.to_string(),
                    depth,
                    name: op.name.clone(),
                    result_types: op.results.iter().map(|r| self.m.ty(*r).to_string()).collect(),
                    operands: op
                        .operands
                        .iter()
                        .map(|v| self.refs.get(v).cloned().unwrap_or(Ref::Dangling))
                        .collect(),
                    attrs: op
                        .attrs
                        .iter()
                        .map(|(k, v)| (k.clone(), attr_text(v)))
                        .collect(),
                    loc_name: op.loc.name().map(|s| s.to_string()),
                    result_loc_names: op
                        .results
                        .iter()
                        .map(|r| self.m.loc_of(*r).name().map(|s| s.to_string()))
                        .collect(),
                };
                self.out.push(flat);
                for (rj, sub) in op.regions.iter().enumerate() {
                    let sub_path = if path.is_empty() {
                        format!("{k}.{rj}")
                    } else {
                        format!("{path}/{k}.{rj}")
                    };
                    self.walk_region(sub, &sub_path, depth + 1);
                }
            }
        }
    }
}

fn flatten(m: &Module, fname: &str) -> Option<Vec<FlatOp>> {
    let f = m.func(fname)?;
    let mut fl = Flattener {
        m,
        out: Vec::new(),
        refs: std::collections::HashMap::new(),
    };
    if let Some(b) = f.body.blocks.first() {
        for (i, a) in b.args.iter().enumerate() {
            fl.refs.insert(*a, Ref::Arg(i));
        }
    }
    fl.walk_region(&f.body, "", 0);
    Some(fl.out)
}

fn histogram(ops: &[FlatOp]) -> BTreeMap<String, usize> {
    let mut h = BTreeMap::new();
    for o in ops {
        *h.entry(o.name.clone()).or_insert(0) += 1;
    }
    h
}

/// The outcome of a structural comparison.
pub struct Report {
    /// Every difference found, each naming the field that differs.
    pub findings: Vec<String>,
    /// Ops compared and found identical.
    pub matched_ops: usize,
    /// Op histogram of the expected (golden) side.
    pub expected_hist: BTreeMap<String, usize>,
    /// Op histogram of the actual side.
    pub actual_hist: BTreeMap<String, usize>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.findings.is_empty()
    }

    /// A field-by-field rendering suitable for a test failure message.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "{} op(s) matched, {} finding(s)",
            self.matched_ops,
            self.findings.len()
        );
        for f in &self.findings {
            let _ = writeln!(s, "  DIFF {f}");
        }
        let mut keys: Vec<&String> = self
            .expected_hist
            .keys()
            .chain(self.actual_hist.keys())
            .collect();
        keys.sort();
        keys.dedup();
        let _ = writeln!(s, "  op counts (expected -> actual):");
        for k in keys {
            let e = self.expected_hist.get(k).copied().unwrap_or(0);
            let a = self.actual_hist.get(k).copied().unwrap_or(0);
            let mark = if e == a { ' ' } else { '!' };
            let _ = writeln!(s, "   {mark} {k:<28} {e:>4} -> {a:>4}");
        }
        s
    }
}

/// Compare `actual` against `expected` structurally.
pub fn compare(expected: &Module, actual: &Module) -> Report {
    let mut findings = Vec::new();
    let mut matched_ops = 0usize;

    // ---- the function set ------------------------------------------------------
    let mut exp_fns: Vec<&str> = expected.funcs.iter().map(|f| f.name.as_str()).collect();
    let mut act_fns: Vec<&str> = actual.funcs.iter().map(|f| f.name.as_str()).collect();
    exp_fns.sort();
    act_fns.sort();
    for name in &exp_fns {
        if !act_fns.contains(name) {
            findings.push(format!("function `{name}`: present in golden, MISSING in ours"));
        }
    }
    for name in &act_fns {
        if !exp_fns.contains(name) {
            findings.push(format!("function `{name}`: EXTRA in ours, absent from golden"));
        }
    }

    let mut expected_hist = BTreeMap::new();
    let mut actual_hist = BTreeMap::new();

    for name in exp_fns.iter().filter(|n| act_fns.contains(n)) {
        let ef = expected.func(name).unwrap();
        let af = actual.func(name).unwrap();

        if ef.visibility != af.visibility {
            findings.push(format!(
                "function `{name}`.visibility: golden {:?}, ours {:?}",
                ef.visibility, af.visibility
            ));
        }
        if ef.noinline != af.noinline {
            findings.push(format!(
                "function `{name}`.noinline: golden {}, ours {}",
                ef.noinline, af.noinline
            ));
        }
        let et: Vec<String> = ef.arg_types.iter().map(Type::to_string).collect();
        let at: Vec<String> = af.arg_types.iter().map(Type::to_string).collect();
        if et != at {
            findings.push(format!(
                "function `{name}`.arg_types: golden [{}], ours [{}]",
                et.join(", "),
                at.join(", ")
            ));
        }
        let er: Vec<String> = ef.ret_types.iter().map(Type::to_string).collect();
        let ar: Vec<String> = af.ret_types.iter().map(Type::to_string).collect();
        if er != ar {
            findings.push(format!(
                "function `{name}`.ret_types: golden [{}], ours [{}]",
                er.join(", "),
                ar.join(", ")
            ));
        }

        let eops = flatten(expected, name).unwrap_or_default();
        let aops = flatten(actual, name).unwrap_or_default();
        for (k, v) in histogram(&eops) {
            *expected_hist.entry(k).or_insert(0) += v;
        }
        for (k, v) in histogram(&aops) {
            *actual_hist.entry(k).or_insert(0) += v;
        }

        if eops.len() != aops.len() {
            findings.push(format!(
                "function `{name}`.op_count: golden {}, ours {}",
                eops.len(),
                aops.len()
            ));
        }

        for i in 0..eops.len().min(aops.len()) {
            let e = &eops[i];
            let a = &aops[i];
            let at_ = format!("`{name}` op[{i}]");
            let mut op_ok = true;
            if e.name != a.name {
                findings.push(format!("{at_}.name: golden `{}`, ours `{}`", e.name, a.name));
                // A name mismatch desynchronizes the walk; everything after is noise.
                findings.push(format!(
                    "{at_}: op sequence diverged here -- later findings suppressed"
                ));
                break;
            }
            if e.depth != a.depth || e.path != a.path {
                findings.push(format!(
                    "{at_} `{}`.nesting: golden depth {} path {:?}, ours depth {} path {:?}",
                    e.name, e.depth, e.path, a.depth, a.path
                ));
                op_ok = false;
            }
            if e.result_types != a.result_types {
                findings.push(format!(
                    "{at_} `{}`.result_types: golden [{}], ours [{}]",
                    e.name,
                    e.result_types.join(", "),
                    a.result_types.join(", ")
                ));
                op_ok = false;
            }
            if e.operands != a.operands {
                let es: Vec<String> = e.operands.iter().map(|r| r.to_string()).collect();
                let as_: Vec<String> = a.operands.iter().map(|r| r.to_string()).collect();
                findings.push(format!(
                    "{at_} `{}`.operands: golden [{}], ours [{}]",
                    e.name,
                    es.join(", "),
                    as_.join(", ")
                ));
                op_ok = false;
            }
            if e.attrs != a.attrs {
                let mut keys: Vec<&String> = e.attrs.keys().chain(a.attrs.keys()).collect();
                keys.sort();
                keys.dedup();
                for k in keys {
                    let ev = e.attrs.get(k);
                    let av = a.attrs.get(k);
                    if ev != av {
                        findings.push(format!(
                            "{at_} `{}`.attr[{k}]: golden {:?}, ours {:?}",
                            e.name, ev, av
                        ));
                    }
                }
                op_ok = false;
            }
            if e.loc_name != a.loc_name {
                findings.push(format!(
                    "{at_} `{}`.loc_name: golden {:?}, ours {:?}",
                    e.name, e.loc_name, a.loc_name
                ));
                op_ok = false;
            }
            if e.result_loc_names != a.result_loc_names {
                findings.push(format!(
                    "{at_} `{}`.result_loc_names: golden {:?}, ours {:?}",
                    e.name, e.result_loc_names, a.result_loc_names
                ));
                op_ok = false;
            }
            if op_ok {
                matched_ops += 1;
            }
        }
    }

    Report {
        findings,
        matched_ops,
        expected_hist,
        actual_hist,
    }
}

/// Compare against golden TEXT, parsing it first. The parse failing is itself a finding,
/// never a pass.
pub fn compare_to_golden_text(golden: &str, actual: &Module) -> Report {
    match crate::ttir::parse::parse_module(golden) {
        Ok(expected) => compare(&expected, actual),
        Err(e) => Report {
            findings: vec![format!("golden did not parse: {e}")],
            matched_ops: 0,
            expected_hist: BTreeMap::new(),
            actual_hist: BTreeMap::new(),
        },
    }
}

/// Unused-name helper kept out of the public surface.
#[allow(dead_code)]
fn _loc_unused(_: &Loc, _: &Op) {}
