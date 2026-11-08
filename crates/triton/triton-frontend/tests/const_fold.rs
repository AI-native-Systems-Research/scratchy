//! The float constant folds and identities of `make_ttir` (`src/opt.rs`: `fold_constant_arith`
//! and `fold_float_identities`), each EXACT under IEEE-754 or not taken.
//!
//! Every fold has its match on a scalar and on a splat tensor, and every law that holds only on
//! some values has the near-miss that must stay an op: a non-constant operand, the NaN-keeping
//! `maxnumf` where the NaN-propagating `maximumf` folds, and `x * 0` where `x` may be NaN,
//! infinite or of either sign. The fixtures are TTIR text, so each states exactly the ops under
//! test; the last two run the shipped decode-batch kernel, whose one-chunk walk is the
//! measurement the identities were written for.

mod common;

use std::collections::HashMap;

use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::ttir::{Attr, Module, Op, ValueId};
use triton_frontend::{opt, ttir};

/// One public function of `args` returning one `ret`, whose body is `body` and whose result
/// is `%r`.
fn kernel(args: &str, ret: &str, body: &str) -> String {
    format!(
        "module {{\n  tt.func public @k({args}) -> {ret} attributes {{noinline = false}} {{\n\
         {body}\n    tt.return %r : {ret}\n  }}\n}}\n"
    )
}

fn make_ttir(text: &str) -> Module {
    let mut m = ttir::parse::parse_module(text).expect("the fixture parses");
    opt::make_ttir(&mut m).expect("make_ttir");
    m
}

/// What the function returns once `make_ttir` is done.
#[derive(Debug, PartialEq)]
enum Returned {
    /// Its argument at this index.
    Arg(usize),
    /// A float constant (scalar or splat), by its bits, so the sign of a zero is compared.
    Const(u64),
    /// The result of an op of this name.
    Op(String),
}

fn find_def(ops: &[Op], v: ValueId) -> Option<&Op> {
    ops.iter().find_map(|op| {
        if op.results.contains(&v) {
            return Some(op);
        }
        op.regions
            .iter()
            .flat_map(|r| &r.blocks)
            .find_map(|b| find_def(&b.ops, v))
    })
}

fn returned(m: &Module) -> Returned {
    let block = &m.funcs[0].body.blocks[0];
    let ret = block.ops.last().expect("a body");
    assert_eq!(ret.name, "tt.return");
    let v = ret.operands[0];
    if let Some(i) = block.args.iter().position(|a| *a == v) {
        return Returned::Arg(i);
    }
    let def = find_def(&block.ops, v).expect("the result has a definition");
    if def.name == "arith.constant" {
        match def.attrs.get("value") {
            Some(Attr::Float(b, _)) => return Returned::Const(b.0),
            Some(Attr::DenseSplat(inner, _)) => {
                if let Attr::Float(b, _) = &**inner {
                    return Returned::Const(b.0);
                }
            }
            _ => {}
        }
    }
    Returned::Op(def.name.clone())
}

fn bits(v: f64) -> Returned {
    Returned::Const(v.to_bits())
}

fn is_nan_const(r: &Returned) -> bool {
    matches!(r, Returned::Const(b) if f64::from_bits(*b).is_nan())
}

/// A scalar f32 spelling of `v` and a `tensor<4xf32>` splat spelling of it, as MLIR prints
/// each (special values as their f32 bit pattern).
fn spell(v: f64) -> String {
    if v.is_nan() || v.is_infinite() || (v == 0.0 && v.is_sign_negative()) {
        format!("0x{:08X}", (v as f32).to_bits())
    } else {
        format!("{v:e}")
    }
}

const SCALAR: &str = "f32";
const SPLAT: &str = "tensor<4xf32>";

/// `%c = arith.constant <v> : ty` at either form.
fn constant(name: &str, v: f64, ty: &str) -> String {
    if ty == SCALAR {
        format!("    %{name} = arith.constant {} : f32", spell(v))
    } else {
        format!("    %{name} = arith.constant dense<{}> : {ty}", spell(v))
    }
}

/// `op(constant a, constant b)` (or `op(constant a)` when `b` is `None`), at both forms.
fn fold_constants(op: &str, a: f64, b: Option<f64>) -> [Returned; 2] {
    [SCALAR, SPLAT].map(|ty| {
        let mut body = constant("a", a, ty);
        body.push('\n');
        match b {
            Some(b) => {
                body.push_str(&constant("b", b, ty));
                body.push_str(&format!("\n    %r = {op} %a, %b : {ty}"));
            }
            None => body.push_str(&format!("    %r = {op} %a : {ty}")),
        }
        returned(&make_ttir(&kernel("", ty, &body)))
    })
}

/// `op(x, constant c)` with `x` an ARGUMENT -- nothing known of it, NaN and -0 included.
fn identity_on_argument(op: &str, c: f64, constant_left: bool) -> [Returned; 2] {
    [SCALAR, SPLAT].map(|ty| {
        let mut body = constant("c", c, ty);
        let (l, r) = if constant_left {
            ("%c", "%x")
        } else {
            ("%x", "%c")
        };
        body.push_str(&format!("\n    %r = {op} {l}, {r} : {ty}"));
        returned(&make_ttir(&kernel(&format!("%x: {ty}"), ty, &body)))
    })
}

/// `op(x, constant c)` with `x` computed from the argument `%y` by `x_def` (whose result is
/// `%x`, and which may name the constants `%zero` = +0).
fn identity_on(x_def: &str, op: &str, c: f64) -> [Returned; 2] {
    [SCALAR, SPLAT].map(|ty| {
        let mut body = constant("zero", 0.0, ty);
        body.push('\n');
        body.push_str(&constant("c", c, ty));
        body.push('\n');
        body.push_str(&x_def.replace("{ty}", ty));
        body.push_str(&format!("\n    %r = {op} %x, %c : {ty}"));
        returned(&make_ttir(&kernel(&format!("%y: {ty}"), ty, &body)))
    })
}

fn op(name: &str) -> [Returned; 2] {
    [Returned::Op(name.into()), Returned::Op(name.into())]
}

fn arg0() -> [Returned; 2] {
    [Returned::Arg(0), Returned::Arg(0)]
}

// ---------------------------------------------------------------------------------------------
// Constant folds
// ---------------------------------------------------------------------------------------------

#[test]
fn exp_folds_where_its_value_is_exact() {
    let ninf = f64::NEG_INFINITY;
    assert_eq!(
        fold_constants("math.exp", ninf, None),
        [bits(0.0), bits(0.0)]
    );
    assert_eq!(
        fold_constants("math.exp", 0.0, None),
        [bits(1.0), bits(1.0)]
    );
    assert_eq!(
        fold_constants("math.exp", -0.0, None),
        [bits(1.0), bits(1.0)]
    );
    assert_eq!(
        fold_constants("math.exp", f64::INFINITY, None),
        [bits(f64::INFINITY), bits(f64::INFINITY)]
    );
    assert_eq!(
        fold_constants("math.exp2", ninf, None),
        [bits(0.0), bits(0.0)]
    );
    assert!(
        fold_constants("math.exp", f64::NAN, None)
            .iter()
            .all(is_nan_const)
    );
}

#[test]
fn exp_of_any_other_constant_stays_an_op() {
    // e^1 and e^-1 are roundings, which are the device's to make.
    assert_eq!(fold_constants("math.exp", 1.0, None), op("math.exp"));
    assert_eq!(fold_constants("math.exp", -1.0, None), op("math.exp"));
    assert_eq!(fold_constants("math.exp2", 0.5, None), op("math.exp2"));
}

#[test]
fn maximumf_and_minimumf_fold_with_their_own_nan_and_zero_laws() {
    let ninf = f64::NEG_INFINITY;
    assert_eq!(
        fold_constants("arith.maximumf", ninf, Some(3.0)),
        [bits(3.0), bits(3.0)]
    );
    assert_eq!(
        fold_constants("arith.minimumf", 3.0, Some(-2.0)),
        [bits(-2.0), bits(-2.0)]
    );
    // A NaN operand yields NaN -- where `maxnumf` yields the other operand (its neighbour fold,
    // unchanged).
    assert!(
        fold_constants("arith.maximumf", f64::NAN, Some(3.0))
            .iter()
            .all(is_nan_const)
    );
    assert_eq!(
        fold_constants("arith.maxnumf", f64::NAN, Some(3.0)),
        [bits(3.0), bits(3.0)]
    );
    // -0 orders below +0.
    assert_eq!(
        fold_constants("arith.maximumf", -0.0, Some(0.0)),
        [bits(0.0), bits(0.0)]
    );
    assert_eq!(
        fold_constants("arith.minimumf", 0.0, Some(-0.0)),
        [bits(-0.0), bits(-0.0)]
    );
}

#[test]
fn extf_of_a_constant_is_the_constant() {
    for ty in ["f16", "tensor<4xf16>"] {
        let wide = ty.replace("f16", "f32");
        let c = if ty == "f16" {
            "    %a = arith.constant 0xFC00 : f16".to_string()
        } else {
            format!("    %a = arith.constant dense<0xFC00> : {ty}")
        };
        let body = format!("{c}\n    %r = arith.extf %a : {ty} to {wide}");
        assert_eq!(
            returned(&make_ttir(&kernel("", &wide, &body))),
            bits(f64::NEG_INFINITY)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Identities that hold for every operand
// ---------------------------------------------------------------------------------------------

#[test]
fn identities_that_hold_for_every_value_fold_on_an_argument() {
    assert_eq!(identity_on_argument("arith.mulf", 1.0, false), arg0());
    assert_eq!(identity_on_argument("arith.mulf", 1.0, true), arg0());
    assert_eq!(identity_on_argument("arith.divf", 1.0, false), arg0());
    assert_eq!(identity_on_argument("arith.addf", -0.0, false), arg0());
    assert_eq!(identity_on_argument("arith.addf", -0.0, true), arg0());
    assert_eq!(identity_on_argument("arith.subf", 0.0, false), arg0());
    assert_eq!(
        identity_on_argument("arith.maximumf", f64::NEG_INFINITY, false),
        arg0()
    );
    assert_eq!(
        identity_on_argument("arith.minimumf", f64::INFINITY, true),
        arg0()
    );
    assert_eq!(
        identity_on_argument("arith.maxnumf", f64::NAN, false),
        arg0()
    );
}

#[test]
fn identities_that_need_a_fact_stay_ops_on_an_argument() {
    // `-0 + +0` is `+0`.
    assert_eq!(
        identity_on_argument("arith.addf", 0.0, false),
        op("arith.addf")
    );
    // `-0 - -0` is `+0`.
    assert_eq!(
        identity_on_argument("arith.subf", -0.0, false),
        op("arith.subf")
    );
    // `maxnum(NaN, -inf)` is `-inf`, not NaN.
    assert_eq!(
        identity_on_argument("arith.maxnumf", f64::NEG_INFINITY, false),
        op("arith.maxnumf")
    );
    assert_eq!(
        identity_on_argument("arith.minnumf", f64::INFINITY, true),
        op("arith.minnumf")
    );
    // `NaN * 0`, `inf * 0` and `-1 * 0` are not `+0`.
    assert_eq!(
        identity_on_argument("arith.mulf", 0.0, false),
        op("arith.mulf")
    );
    assert_eq!(
        identity_on_argument("arith.mulf", 0.0, true),
        op("arith.mulf")
    );
    // The non-commuting ops with the constant on the left: `0 - x` and `1 / x`.
    assert_eq!(
        identity_on_argument("arith.subf", 0.0, true),
        op("arith.subf")
    );
    assert_eq!(
        identity_on_argument("arith.divf", 1.0, true),
        op("arith.divf")
    );
}

// ---------------------------------------------------------------------------------------------
// Identities that hold on the values the operand can take
// ---------------------------------------------------------------------------------------------

/// `exp(-maxnum(y, 0))`: finite, `+0` or above, never NaN -- an online softmax's rescale.
const RESCALE: &str = "    %m = arith.maxnumf %y, %zero : {ty}\n    %n = arith.negf %m : {ty}\n    \
                       %x = math.exp %n : {ty}";

#[test]
fn x_times_zero_folds_where_x_is_finite_and_its_sign_bit_clear() {
    assert_eq!(
        identity_on(RESCALE, "arith.mulf", 0.0),
        [bits(0.0), bits(0.0)]
    );
    assert_eq!(
        identity_on(RESCALE, "arith.mulf", -0.0),
        [bits(-0.0), bits(-0.0)]
    );
}

#[test]
fn x_times_zero_stays_an_op_where_x_can_be_nan_infinite_or_negative() {
    // `exp(y)` can be +inf (and NaN).
    let unbounded = "    %x = math.exp %y : {ty}";
    assert_eq!(identity_on(unbounded, "arith.mulf", 0.0), op("arith.mulf"));
    // `exp(-exp(y))` is in [0, 1] -- but NaN at a NaN `y`.
    let maybe_nan = "    %e = math.exp %y : {ty}\n    %n = arith.negf %e : {ty}\n    \
                     %x = math.exp %n : {ty}";
    assert_eq!(identity_on(maybe_nan, "arith.mulf", 0.0), op("arith.mulf"));
    // `-rescale` is finite and never NaN, but its sign bit is set: `-1 * 0` is `-0`.
    let negative = "    %m = arith.maxnumf %y, %zero : {ty}\n    %n = arith.negf %m : {ty}\n    \
                    %e = math.exp %n : {ty}\n    %x = arith.negf %e : {ty}";
    assert_eq!(identity_on(negative, "arith.mulf", 0.0), op("arith.mulf"));
}

#[test]
fn x_plus_zero_folds_where_x_is_never_negative_zero() {
    // `exp` never returns -0.
    let exp = "    %x = math.exp %y : {ty}";
    assert_eq!(identity_on(exp, "arith.addf", 0.0), op("math.exp"));
    assert_eq!(identity_on(exp, "arith.subf", -0.0), op("math.exp"));
    // `-exp(y)` can be -0 (at `exp(y)` underflowing to +0).
    let neg = "    %e = math.exp %y : {ty}\n    %x = arith.negf %e : {ty}";
    assert_eq!(identity_on(neg, "arith.addf", 0.0), op("arith.addf"));
}

#[test]
fn maxnum_of_minus_infinity_folds_where_x_is_never_nan() {
    // `maxnum(y, 0)` is never NaN.
    let non_nan = "    %x = arith.maxnumf %y, %zero : {ty}";
    assert_eq!(
        identity_on(non_nan, "arith.maxnumf", f64::NEG_INFINITY),
        op("arith.maxnumf")
    );
    let m = make_ttir(&kernel(
        "%y: f32",
        "f32",
        "    %zero = arith.constant 0.000000e+00 : f32\n    \
         %c = arith.constant 0xFF800000 : f32\n    %x = arith.maxnumf %y, %zero : f32\n    \
         %r = arith.maxnumf %x, %c : f32",
    ));
    let maxes = m.funcs[0].body.blocks[0]
        .ops
        .iter()
        .filter(|o| o.name == "arith.maxnumf")
        .count();
    assert_eq!(maxes, 1, "the outer maxnum is folded away, the inner kept");
    // `exp(y)` can be NaN.
    let maybe_nan = "    %x = math.exp %y : {ty}";
    assert_eq!(
        identity_on(maybe_nan, "arith.maxnumf", f64::NEG_INFINITY),
        op("arith.maxnumf")
    );
}

// ---------------------------------------------------------------------------------------------
// The kernel the identities were measured on
// ---------------------------------------------------------------------------------------------

/// `batched_decode_head_major_kernel` over 4 sequences, `bpc` block slots a chunk, `chunks`
/// chunks.
fn batched_decode(bpc: i128, chunks: i128) -> Module {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("query", "*fp16"),
        ("rep_row_ids", "*i32"),
        ("k_pages", "*fp16"),
        ("v_pages", "*fp16"),
        ("chunk_page_ids", "*i32"),
        ("mask_by_chunk", "*fp16"),
        ("out", "*fp16"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("spec"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("NUM_TOKENS", Val::Int(4)),
        ("NUM_SEQS", Val::Int(4)),
        ("BLOCKS_PER_CHUNK", Val::Int(bpc)),
        ("NUM_CHUNKS", Val::Int(chunks)),
        ("NUM_KV_HEADS", Val::Int(8)),
        ("NUM_QUERIES_PER_KV", Val::Int(4)),
        ("BLOCK_SIZE", Val::Int(64)),
        ("HEAD_SIZE", Val::Int(64)),
        ("NUM_PAGES", Val::Int(2048)),
        ("ROWS_PER_PAGE", Val::Int(1)),
        ("SCALE", Val::Float(0.125)),
    ] {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
    }
    let spec = KernelSpec {
        kernel: "batched_decode_head_major_kernel".to_string(),
        signature,
        constexprs,
        file: "batched_decode_head_major.py".to_string(),
    };
    let src = common::fixture_src("spyre/batched_decode_head_major");
    let mut m = codegen::compile(&src, &spec, Target::spyre()).expect("compiles");
    opt::make_ttir(&mut m).expect("make_ttir");
    m
}

/// Every op in the module, regions included.
fn all_ops(m: &Module) -> Vec<&Op> {
    fn walk<'a>(ops: &'a [Op], out: &mut Vec<&'a Op>) {
        for op in ops {
            out.push(op);
            for r in &op.regions {
                for b in &r.blocks {
                    walk(&b.ops, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut out);
        }
    }
    out
}

fn count(m: &Module, name: &str) -> usize {
    all_ops(m).iter().filter(|o| o.name == name).count()
}

/// The non-constant ops whose result the kernel names `var` (its Python variable).
fn computing(m: &Module, var: &str) -> Vec<String> {
    all_ops(m)
        .iter()
        .filter(|o| o.name != "arith.constant")
        .filter(|o| {
            o.results
                .iter()
                .any(|r| m.values[r.0 as usize].loc.name() == Some(var))
        })
        .map(|o| o.name.clone())
        .collect()
}

/// ONE CHUNK: the walk's loop runs once, its carries enter at `(-inf, 0, 0)`, and the online
/// softmax's rescale is the identity -- `exp(-maxnum(chunk_max + inf, 0))` multiplies a zero
/// sum and a zero output, so the rescale and the running sum's update are gone. Two ops of the
/// identity stay, because neither is exact on every value: `maxnum(chunk_max, -inf)` (a NaN
/// `chunk_max` gives `-inf`) and `chunk_out + 0` (a `-0` element gives `+0`).
#[test]
fn a_one_chunk_decode_batch_drops_its_rescale_chain() {
    let m = batched_decode(4, 1);
    assert_eq!(count(&m, "scf.for"), 0, "the one-trip walk is promoted");
    assert_eq!(computing(&m, "rescale"), Vec::<String>::new());
    assert_eq!(computing(&m, "tile_sum"), Vec::<String>::new());
    assert_eq!(count(&m, "arith.negf"), 0);
    assert_eq!(
        count(&m, "math.exp"),
        1,
        "only the probabilities' exp is left"
    );
    assert_eq!(count(&m, "arith.mulf"), 1, "only the scores' scale is left");
    assert_eq!(computing(&m, "tile_output"), vec!["arith.addf".to_string()]);
    assert_eq!(computing(&m, "new_max"), vec!["arith.maxnumf".to_string()]);
}

/// MANY CHUNKS: the carries are loop block arguments, of which nothing is known, so every op of
/// the rescale stays.
#[test]
fn a_many_chunk_decode_batch_keeps_its_rescale_chain() {
    let m = batched_decode(8, 8);
    assert_eq!(count(&m, "scf.for"), 1);
    assert_eq!(
        computing(&m, "rescale"),
        [
            "arith.extf",
            "arith.subf",
            "arith.maxnumf",
            "arith.negf",
            "math.exp",
            "arith.truncf"
        ]
        .map(String::from)
        .to_vec()
    );
    assert_eq!(
        computing(&m, "tile_sum").len(),
        2,
        "the running sum's mulf and addf"
    );
    assert_eq!(
        count(&m, "math.exp"),
        3,
        "rescale, probabilities and the slots' merge"
    );
}

/// `page_attn_head_major_decode_kernel` -- one sequence's decode -- at granite-3.1-2b's tp/8
/// head slice (one kv head, four query heads) walking `blocks` blocks.
fn sequence_decode(blocks: i128) -> Module {
    let mut signature = HashMap::new();
    for (k, v) in [
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("k_pages", "*fp16"),
        ("v_pages", "*fp16"),
        ("page_index_table", "*i32"),
        ("kv_row_pool", "*i32"),
        ("mask_stack", "*fp16"),
        ("out", "*fp16"),
    ] {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("spec"));
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("NUM_TOKENS", Val::Int(1)),
        ("NUM_HEADS", Val::Int(4)),
        ("NUM_KV_HEADS", Val::Int(1)),
        ("HEAD_SIZE", Val::Int(64)),
        ("BLOCK_SIZE", Val::Int(128)),
        ("NUM_PAGES", Val::Int(256)),
        ("NUM_BLOCKS", Val::Int(blocks)),
        ("SCALE", Val::Float(0.125)),
    ] {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
    }
    let spec = KernelSpec {
        kernel: "page_attn_head_major_decode_kernel".to_string(),
        signature,
        constexprs,
        file: "page_attn_head_major_decode.py".to_string(),
    };
    let src = common::fixture_src("spyre/page_attn_head_major_decode");
    let mut m = codegen::compile(&src, &spec, Target::spyre()).expect("compiles");
    opt::make_ttir(&mut m).expect("make_ttir");
    m
}

/// The same law on the one-sequence decode kernel, whose walk is one block at the shortest
/// rungs: the folds are over ops, so every kernel whose walk the promotion removes gets them.
#[test]
fn a_one_block_sequence_decode_drops_its_rescale_chain() {
    let m = sequence_decode(1);
    assert_eq!(count(&m, "scf.for"), 0);
    assert_eq!(computing(&m, "rescale"), Vec::<String>::new());
    assert_eq!(computing(&m, "tile_sum"), Vec::<String>::new());
    assert_eq!(count(&m, "arith.negf"), 0);
    assert_eq!(count(&m, "math.exp"), 1);
    assert_eq!(count(&m, "arith.mulf"), 1);
}

#[test]
fn a_many_block_sequence_decode_keeps_its_rescale_chain() {
    let m = sequence_decode(2);
    assert_eq!(count(&m, "scf.for"), 1);
    assert_eq!(computing(&m, "rescale").len(), 6);
    assert_eq!(count(&m, "math.exp"), 2);
}
