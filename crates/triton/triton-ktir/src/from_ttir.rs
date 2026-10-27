// SPDX-License-Identifier: Apache-2.0

//! THE LAST HOP: bridge one's ttir VALUE -> this crate's [`crate::ir::Module`].
//!
//! ```text
//!   triton_frontend::codegen::compile   Python source -> RAW ttir  (value)
//!   triton_frontend::opt::make_ttir     the six-pass pipeline      (value)
//!   from_ttir::convert                  <- HERE                   (value)
//!   triton_ktir::make_ktir              the seven KTDP/KTDF passes (value)
//! ```
//!
//! Nothing is printed and nothing is parsed at any of those arrows. [`crate::text`] stays
//! what its header says it is -- a test instrument -- and this module is why: without it,
//! the only way to get bridge one's output into bridge two would have been to print and
//! reparse it, which is the defect the no-text rule exists to prevent.
//!
//! # IT REFUSES BY NAME RATHER THAN FALLING BACK, AND THAT IS THE WHOLE DESIGN
//!
//! A reader with a generic fallback does not fail: it GUESSES, and the guess is often
//! well-formed enough to look right. That has now bitten this tree three times -- most
//! recently `ktdp.construct_indirect_access_tile`, whose generic arm typed the gather with
//! its INDEX VECTOR's type because `indirect(%ids : memref<256xsi32>)` puts a ` : ` inside
//! the operand list, and then `apply_closer` skipped the real type because
//! `result_types` was already non-empty. Nothing errored; a consumer asking for the
//! gathered tile's shape simply got the wrong one.
//!
//! So this module has NO fallback arm. [`OP_SET`], [`ATTR_KEYS`] and the type and
//! attribute mappings are explicit allow-lists, MEASURED from the sixteen post-`make_ttir`
//! goldens rather than imagined, and anything outside them is a [`crate::Refusal`] naming
//! the construct. Two ops are deliberately mapped to [`ir::OpKind::Other`] --
//! `arith.negf` and `math.rsqrt`, which have no variant -- and they are LISTED, so a
//! genuinely new op is refused instead of quietly joining them.
//!
//! # SSA IDENTITY IS AN INDEX, NOT A SPELLING
//!
//! `ttir::ValueId` and [`ir::Ssa`] are both `u32` indices into a module-wide arena, so the
//! mapping is the identity and def-use is carried across unchanged. That is not a
//! convenience: MLIR's SSA NAMES are region-scoped, and keying anything on them collides.
//! `attention_flash_causal` has two sibling `scf.for`s that both define `%start_n`, and
//! that collision has now been found in three separate places. The printed spelling goes
//! into [`ir::Module::hints`], out of band, where no pass can read it.
//!
//! # WHAT IT DOES NOT CARRY, AND WHY THAT IS SOUND
//!
//! LOCATIONS. [`ir`] has no location field at all -- not a dropped one, there is nowhere
//! to put one. `ttir::Loc`'s NAME survives as a hint because MLIR derives a printed SSA
//! name from it and that is what makes a golden diff legible; the file, line and column do
//! not survive, and nothing downstream of here has ever had them.

use triton_frontend::ttir::{self, FloatKind, Signedness, ValueId};

use crate::ir::{self, Attr, AttrKey, DType, FloatBits, IrType, Module, Op, OpKind, Region, Ssa};
use crate::{Refusal, Result};

const PASS: &str = "from-ttir";

/// EVERY ttir op this adapter accepts, with the [`OpKind`] it becomes.
///
/// MEASURED, not designed: this is the exact op set of the sixteen
/// `tests/goldens/<name>.ttir.mlir` files, which is what `triton_frontend::opt::make_ttir`
/// produces. An op absent from this table is refused by name -- including one that
/// `OpKind::from_spelling` would happily turn into an `Other`, because "we do not model
/// this" and "we model this as Other on purpose" must not look the same.
///
/// THE FIVE DELIBERATE `Other`s: `arith.negf`, `math.rsqrt`, `arith.cmpf`,
/// `arith.sitofp`, and `arith.fptosi` have no `OpKind` variant.
/// They are listed here so the choice is visible, and it is safe because a consumer keys
/// on `OpKind::spelling()` for exactly these -- `triton-superdsc-lower`'s op map reads
/// `arith.negf` (its SiLU recogniser's head) and `math.rsqrt` through the spelling, and
/// `crate::text::parse` produces the same `Other` from the C++'s own text, so the golden
/// side and this side agree. The three router-index spellings ride the same rule with the
/// same justification: `to_ktir_emit`'s Other arm names their `KOpKind`, and the compare's
/// `predicate` is an attribute carried verbatim, so no consumer is left keying on a guess.
pub const OP_SET: &[&str] = &[
    // structure
    "tt.func",
    "tt.return",
    "scf.for",
    "scf.yield",
    // arith
    "arith.constant",
    "arith.addf",
    "arith.addi",
    "arith.subf",
    "arith.mulf",
    "arith.muli",
    "arith.divf",
    "arith.divsi",
    "arith.remsi",
    "arith.maxnumf",
    // The MoE kernel's fp8 clamps (`moe.py`): `tl.minimum` beside `tl.maximum`. `ArithMinnumf`
    // is modelled one to one, as `ArithMaxnumf` is.
    "arith.minnumf",
    "arith.extf",
    "arith.truncf",
    "arith.negf", // Other, on purpose
    // THE ROUTER-INDEX EXTENSION (Phase 2b): the MoE router rows spell a
    // 0/1 compare mask and slot selection, which have no compare-free form,
    // and their index arithmetic integerizes through fp<->int casts. The
    // front end's `compare`/`where`/`cast` emit these five spellings; the
    // emulator has served `ArithCmpf`/`ArithCmpi`/`ArithSelect`/
    // `ArithSitofp`/`ArithFptosi` all along, and `predicate` rides
    // `ATTR_KEYS` below (`Attr::Pred` is carried by `attr_value`, whose own
    // comment says that hole was found by reading the reader). `cmpf`/
    // `sitofp`/`fptosi` ride `Other` exactly like `negf`: `to_ktir_emit`'s
    // Other arm names their variant.
    "arith.cmpf", // Other, on purpose
    "arith.cmpi",
    "arith.select",
    "arith.sitofp", // Other, on purpose
    "arith.fptosi", // Other, on purpose
    // math
    "math.exp",
    "math.exp2",
    "math.rsqrt", // Other, on purpose
    "math.tanh",
    // `tl.abs` on a float (`moe.py`'s per-token `amax(|x|)`). Rides `Other` like `rsqrt`:
    // `to_ktir_emit`'s Other arm names `KOpKind::MathAbsf`.
    "math.absf", // Other, on purpose
    // `x.to(tl.float8e4nv)` and back (`semantic.py:840`): the fp8 weight widening and the
    // MoE kernel's activation codes. [`fp_to_fp_as_arith`] admits it AS `arith.extf` /
    // `arith.truncf` at the fp8 element type, so no `tt.fp_to_fp` exists past this adapter.
    "tt.fp_to_fp",
    // triton
    "tt.get_program_id",
    // `tl.arange` -- the lane vector the router index kernels select with. Its
    // `start`/`end` ride `ATTR_KEYS` below as `AttrKey::Other` (the escape hatch
    // for a key with no variant), and `to_ktir`'s `convert_make_range` rewrites
    // the op into the `tensor.generate` index-grid group the emulator's
    // `generate` handler runs -- the branch's own `index_grid` shape.
    "tt.make_range",
    "tt.make_tensor_descriptor",
    "tt.descriptor_load",
    "tt.descriptor_store",
    "tt.descriptor_gather",
    "tt.descriptor_scatter",
    // The routed expert id of `expert_slab.ttir.mlir`: one SCALAR read through a pointer,
    // `tt.addptr %ids, %p : !tt.ptr<i32>, i32` then `tt.load %ptr : !tt.ptr<i32>` -- no
    // mask, no other, no attributes (the default `isVolatile` prints as nothing). Both
    // types map one to one (`IrType::Ptr`, `IrType::Scalar`). `convert_ttir_to_ktdp`
    // consumes the read as the INDIRECT offset of a slab load; a raw TILE load stays
    // refused by name there (`refuse_raw_tile_access`).
    "tt.addptr",
    "tt.load",
    "tt.dot",
    "tt.trans",
    // ⭐ THE FIRST ENTRY ADDED SINCE THE CENSUS, deliberately. `tl.reshape` is the front
    // end's own `semantic.rs::reshape` (numel-preserving flat reinterpretation, NO
    // attributes -- the destination extents ride the result type, which `ty` maps for any
    // rank), and it appears in no golden because no fixture reshapes a COMPUTED value --
    // the spliced `reshape.py` states its relayout as load/store pairs instead. The batched
    // paged-decode kernel reshapes gather results and arithmetic tiles, so the op crosses
    // this bridge as itself: one operand, one result, the result's type the destination.
    // `to_ktir`'s `convert_reshape` is what turns it into a scheduler-dialect op.
    "tt.reshape",
    "tt.reduce",
    "tt.reduce.return",
    "tt.expand_dims",
    "tt.broadcast",
    // A splat of a RUNTIME scalar argument (`tt.splat %scale : tensor<64xf16>`), which
    // `make_ttir` leaves standing exactly because the operand has no value to fold --
    // every CONSTANT splat is already a `dense<>` `arith.constant` by then, so this
    // spelling appears if and only if the splatted value is an argument. `to_ktir`'s
    // `convert_splat_of_argument` is the only consumer that may accept it.
    "tt.splat",
];

/// EVERY attribute key this adapter accepts. Same census, same reasoning.
///
/// `tt.divisibility` is here because exactly one golden carries it
/// (`attention_flash_causal`'s `arith.muli`), and it is the one attribute whose VALUE has
/// no modelled form -- see [`attr_value`].
pub const ATTR_KEYS: &[&str] = &[
    // `arith.constant`'s payload. It is NOT printed in a `{...}` dictionary, which is how
    // it was missing from the first version of this list -- the census that built the list
    // grepped for `{key =` and an `arith.constant` prints `42 : i32` bare. The allow-list
    // caught that omission on the first run, which is the argument for having one.
    "value",
    "axis",
    "order",
    "noinline",
    "callee",
    "tt.divisibility",
    // `arith.cmpf`/`arith.cmpi`'s predicate (`olt`, `oeq`, ...), printed inside the op's
    // own parens, not a `{...}` dictionary. `attr_value` maps `ttir::Attr::Pred` to
    // `Attr::Str` verbatim and `to_ktir_emit` re-keys it `KAttrKey::Predicate`; the
    // emulator and the door both read it to pick the comparison.
    "predicate",
    // `tt.make_range`'s bounds (`start = 0 : i32`, `end = 64 : i32`), the two integers
    // that name the lane vector's extent. They have no `AttrKey` variant -- the
    // fingerprint-carrying enum stays untouched -- so they ride `AttrKey::Other`,
    // which `from_spelling` builds for any spelling not in its table and
    // `to_ktir`'s `convert_make_range` reads back by name when it rewrites the op
    // into a `tensor.generate` group.
    "start",
    "end",
];

/// Convert a post-`make_ttir` ttir module into a KTIR module.
///
/// `Err` names the construct. There is no partial success: a module this cannot represent
/// exactly is not handed on in a form a pass would then interpret.
pub fn convert(m: &ttir::Module) -> Result<Module> {
    let mut out = Module::new();
    // THE ARENA IS SHARED. `ValueId(n)` becomes `Ssa(n)`, so def-use crosses unchanged and
    // `next_ssa` starts past every value bridge one made -- a pass minting a value cannot
    // collide with one of them.
    out.next_ssa = m.values.len() as u32;
    for (i, v) in m.values.iter().enumerate() {
        if let Some(name) = v.loc.name() {
            out.hints.insert(Ssa(i as u32), name.to_string());
        }
    }

    if m.funcs.is_empty() {
        return Err(Refusal::new(PASS, "the ttir module has no functions"));
    }
    // `make_ttir`'s symbol-DCE leaves exactly one, and `ir::Module::kernel()` refuses
    // anything else -- so say it HERE, where the reason is still visible, rather than
    // three passes later.
    if m.funcs.len() != 1 {
        return Err(Refusal::new(
            PASS,
            format!(
                "the ttir module has {} functions ({}). `make_ttir`'s inliner and \
                 symbol-DCE reduce a kernel to one; more than one means the pipeline did \
                 not run, and every pass downstream reads `Module::kernel()`, which \
                 refuses anything but one.",
                m.funcs.len(),
                m.funcs
                    .iter()
                    .map(|f| format!("@{}", f.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }

    for f in &m.funcs {
        out.ops.push(func(m, f)?);
    }
    Ok(out)
}

fn func(m: &ttir::Module, f: &ttir::Func) -> Result<Op> {
    let mut op = Op::new(OpKind::TtFunc);
    op.set_attr(AttrKey::SymName, Attr::Str(f.name.clone()));
    // `noinline` is on every golden's `tt.func` and the text reader reads it, so the two
    // sides must agree that it is there.
    op.set_attr(AttrKey::Noinline, Attr::Bool(f.noinline));

    if f.body.blocks.len() != 1 {
        return Err(Refusal::new(
            PASS,
            format!(
                "`@{}`'s body has {} blocks. `ir::Region` models ONE, and \
                 `make_ttir`'s unreachable-block removal is what reduces a function to \
                 it -- so more than one here means that pass did not run, and truncating \
                 silently would delete code.",
                f.name,
                f.body.blocks.len()
            ),
        ));
    }
    let block = &f.body.blocks[0];
    if block.args.len() != f.arg_types.len() {
        return Err(Refusal::new(
            PASS,
            format!(
                "`@{}` declares {} argument type(s) but its entry block takes {}",
                f.name,
                f.arg_types.len(),
                block.args.len()
            ),
        ));
    }
    let mut region = Region::default();
    for (a, t) in block.args.iter().zip(f.arg_types.iter()) {
        region
            .args
            .push((Ssa(a.0), ty(m, t, "a function argument")?));
    }
    for o in &block.ops {
        region.ops.push(operation(m, o)?);
    }
    op.regions.push(region);
    Ok(op)
}

/// `tt.fp_to_fp` AS THE `arith` CAST IT IS — `arith.extf` widening, `arith.truncf`
/// narrowing — at the SAME element types, so the fp8 side stays `f8E4M3FN` on the value.
///
/// ⭐ AT THE ENTRY, NOT IN A LATER PASS. Before the fp8 cast had its own spelling, the
/// front end emitted the fp8 weight's widening as `arith.extf`, and every pass between here
/// and `to_ktir` keys on that op: `dot_to_linalg`'s fp8 peel, `legalize_types`, the fused
/// segment planner's charge for the widened tile. Mapping `tt.fp_to_fp` here hands every one
/// of them the op it already reads, so `matmul_fp8.py` lowers exactly as
/// before (byte-identical descriptors) and nothing downstream needs a second spelling.
///
/// `tt.fp_to_fp` exists in Triton because MLIR's `arith` cannot spell a non-default
/// rounding or the custom fp8 formats; at KTIR the cast the kernels spell is exactly an
/// e4m3 narrowing under round-to-nearest-even and a widening back. ⛔ ONLY those: a
/// narrowing whose `rounding` is anything but `rtne`, or an fp8 format other than
/// `f8E4M3FN`, is refused by name — no consumer rounds to it.
fn fp_to_fp_as_arith(m: &ttir::Module, o: &ttir::Op) -> Result<Option<OpKind>> {
    if o.name != "tt.fp_to_fp" {
        return Ok(None);
    }
    let elem = |v: ttir::ValueId| m.ty(v).scalar().clone();
    let src = o
        .operands
        .first()
        .map(|v| elem(*v))
        .ok_or_else(|| Refusal::new(PASS, "tt.fp_to_fp without an operand"))?;
    let dst = o
        .results
        .first()
        .map(|v| elem(*v))
        .ok_or_else(|| Refusal::new(PASS, "tt.fp_to_fp without a result"))?;
    let fp8 = |t: &ttir::Type| matches!(t, ttir::Type::Float(ttir::FloatKind::F8E4M3FN));
    let float = |t: &ttir::Type| matches!(t, ttir::Type::Float(_));
    match (fp8(&src), fp8(&dst)) {
        (false, true) if float(&src) => {
            match o.attrs.get("rounding") {
                Some(ttir::Attr::Pred("rtne")) => {}
                other => {
                    return Err(Refusal::new(
                        PASS,
                        format!(
                            "tt.fp_to_fp narrowing to e4m3 with rounding {other:?}: only \
                             `rtne` (Triton's default downcast, `semantic.py:826`) has a \
                             consumer"
                        ),
                    ));
                }
            }
            Ok(Some(OpKind::ArithTruncf))
        }
        (true, false) if float(&dst) => Ok(Some(OpKind::ArithExtf)),
        _ => Err(Refusal::new(
            PASS,
            format!(
                "tt.fp_to_fp from {src} to {dst}: only the e4m3 round trip (f16/f32 <-> \
                 f8E4M3FN) is lowered"
            ),
        )),
    }
}

fn operation(m: &ttir::Module, o: &ttir::Op) -> Result<Op> {
    if !OP_SET.contains(&o.name.as_str()) {
        return Err(Refusal::new(
            PASS,
            format!(
                "ttir operation `{}` is not in this adapter's measured op set, so there is \
                 no KTIR form for it here. The set is the exact census of the sixteen \
                 post-make_ttir goldens (see `from_ttir::OP_SET`); adding an op means \
                 adding it there deliberately, with its type and attribute mapping \
                 checked, rather than letting `OpKind::from_spelling` turn it into an \
                 `Other` whose fields nobody verified.",
                o.name
            ),
        ));
    }
    let kind = fp_to_fp_as_arith(m, o)?.unwrap_or_else(|| OpKind::from_spelling(&o.name));
    let mut out = Op::new(kind);
    out.operands = o.operands.iter().map(|v| Ssa(v.0)).collect();
    for r in &o.results {
        let t = m.ty(*r);
        // A result of type `()` is how the front end spells "no result"; it never appears
        // on an op that has a `results` entry, so it is a malformed module rather than an
        // unsupported one.
        if matches!(t, ttir::Type::Void) {
            return Err(Refusal::new(
                PASS,
                format!("`{}` has a result of type `()`", o.name),
            ));
        }
        out.results.push(Ssa(r.0));
        out.result_types.push(ty(m, t, &o.name)?);
    }
    // BTreeMap iterates in key order, which is the order the text reader sees them in a
    // printed dictionary, so the two sides' attribute lists agree without sorting here.
    for (k, v) in &o.attrs {
        // `tt.fp_to_fp`'s `rounding` is CONSUMED by `fp_to_fp_as_arith`, which admitted only
        // `rtne` — `arith.truncf`'s own semantics — so it carries nothing the cast's kind
        // does not already state, and no later reader has a key for it.
        if o.name == "tt.fp_to_fp" && k == "rounding" {
            continue;
        }
        if !ATTR_KEYS.contains(&k.as_str()) {
            return Err(Refusal::new(
                PASS,
                format!(
                    "`{}` carries the attribute `{k}`, which is not in this adapter's \
                     measured key set (`from_ttir::ATTR_KEYS`). An attribute whose value \
                     form has not been checked against what `text::parse` produces from \
                     the C++'s own output would make the golden diff report a difference \
                     whose cause is in this adapter.",
                    o.name
                ),
            ));
        }
        let (key, value) = attr(m, &o.name, k, v)?;
        out.set_attr(key, value);
    }
    for r in &o.regions {
        if r.blocks.len() != 1 {
            return Err(Refusal::new(
                PASS,
                format!(
                    "`{}` has a region with {} blocks; `ir::Region` models one",
                    o.name,
                    r.blocks.len()
                ),
            ));
        }
        let b = &r.blocks[0];
        let mut nr = Region::default();
        for a in &b.args {
            nr.args.push((Ssa(a.0), ty(m, m.ty(*a), &o.name)?));
        }
        for inner in &b.ops {
            nr.ops.push(operation(m, inner)?);
        }
        out.regions.push(nr);
    }
    Ok(out)
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// An element type, refusing every width [`DType`] does not model.
///
/// `in_desc` is the one place SIGNEDNESS survives, and it is a MEASUREMENT rather than a
/// choice. MLIR drops Triton's signedness in most positions (`i32`), but NOT inside a
/// `!tt.tensordesc<>` block type: `tt.make_tensor_descriptor` over an `!tt.ptr<i32>`
/// prints `<64xsi32>`, and `ConvertTTIRToKTDP` then carries that spelling onto the memory
/// view it builds (`memref<256xsi32>` in `embedding_granite`'s KTIR). `DType::SI32` exists
/// for exactly that, so a descriptor's element has to be asked for differently from a
/// tensor's.
fn dtype(t: &ttir::Type, in_desc: bool, what: &str) -> Result<DType> {
    Ok(match t {
        ttir::Type::Int(1, _) => DType::I1,
        ttir::Type::Int(32, s) => {
            if in_desc && *s == Signedness::Signed {
                DType::SI32
            } else if *s == Signedness::Unsigned {
                return Err(Refusal::new(
                    PASS,
                    format!(
                        "{what}: an UNSIGNED i32. MLIR has no unsigned spelling in these \
                         dialects and `DType` models signed and signless only, so \
                         accepting it would silently change the operation's signedness"
                    ),
                ));
            } else {
                DType::I32
            }
        }
        ttir::Type::Int(64, _) => DType::I64,
        ttir::Type::Int(b, _) => {
            return Err(Refusal::new(
                PASS,
                format!("{what}: `i{b}` has no `DType`; the modelled widths are 1, 32 and 64"),
            ));
        }
        ttir::Type::Float(FloatKind::F16) => DType::F16,
        ttir::Type::Float(FloatKind::F32) => DType::F32,
        // OCP E4M3, the one fp8 the device reads (Triton's `fp8e4nv`; scratchy's W8A8
        // chain is card-proven on it). The LEGAL position is the weight VIEW's elem:
        // `tl.make_tensor_descriptor` over an fp8 pointer builds a tensordesc whose
        // elem is `f8E4M3FN`, and the memory view stamped from it must carry the
        // attribute `ktir-superdsc`'s `regions()` reads (`is_fp8`). Computed values are
        // f16 (the frontend refuses a mixed `tl.dot`, so the kernel widens with
        // `.to(tl.float16)`), so this arm is exercised by views, not by results.
        ttir::Type::Float(FloatKind::F8E4M3FN) => DType::Fp8E4m3,
        ttir::Type::Float(k) => {
            return Err(Refusal::new(
                PASS,
                format!(
                    "{what}: `{}` has no `DType`; the modelled float types are f16, f32 and \
                     f8E4M3FN (E4M3 weights only -- computed values are f16)",
                    k.mlir()
                ),
            ));
        }
        other => {
            return Err(Refusal::new(
                PASS,
                format!("{what}: `{other}` is not an element type"),
            ));
        }
    })
}

fn ty(m: &ttir::Module, t: &ttir::Type, what: &str) -> Result<IrType> {
    let _ = m;
    Ok(match t {
        ttir::Type::Tensor(shape, elem) => IrType::Tensor {
            dims: shape.clone(),
            elem: dtype(elem, false, what)?,
        },
        // The BLOCK shape, which is what the descriptor patterns read.
        ttir::Type::TensorDesc(shape, elem) => IrType::TensorDesc {
            dims: shape.clone(),
            elem: dtype(elem, true, what)?,
        },
        ttir::Type::Ptr(elem, space) => {
            if *space != 1 {
                return Err(Refusal::new(
                    PASS,
                    format!(
                        "{what}: a pointer in address space {space}. `IrType::Ptr` carries \
                         no space, so a non-default one would be silently dropped"
                    ),
                ));
            }
            IrType::Ptr {
                elem: dtype(elem, false, what)?,
            }
        }
        ttir::Type::Int(..) | ttir::Type::Float(..) => IrType::Scalar(dtype(t, false, what)?),
        ttir::Type::Void => {
            return Err(Refusal::new(PASS, format!("{what}: `()` has no KTIR type")));
        }
    })
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

fn attr(m: &ttir::Module, op_name: &str, key: &str, v: &ttir::Attr) -> Result<(AttrKey, Attr)> {
    let k = AttrKey::from_spelling(key);
    Ok((k, attr_value(m, op_name, key, v)?))
}

/// # EVERY ARM HERE IS PINNED TO WHAT `text::parse` PRODUCES FROM THE C++'s OWN OUTPUT
///
/// The golden diff compares this adapter's attributes against ones the text reader built
/// from the C++ KTIR. So "the right answer" for each arm is not "a faithful
/// representation" -- it is "the SAME `Attr` variant the reader makes", or the diff
/// reports a difference whose cause is here. Each arm names the reader path it matches.
fn attr_value(m: &ttir::Module, op_name: &str, key: &str, v: &ttir::Attr) -> Result<Attr> {
    let _ = m;
    Ok(match v {
        // `axis = 1 : i32` -> `parse_attr_value`'s `" : "` split then `parse_int`.
        ttir::Attr::Int(i, _) => Attr::Int(narrow(*i, op_name, key)?),
        // An `arith.constant`'s float payload. `parse_constant` builds `FloatBits` AT THE
        // ELEMENT WIDTH, so a f16 constant is 16 bits of half, not 64 bits of double.
        ttir::Attr::Float(bits, t) => Attr::Float(float_bits(bits.get(), t, op_name, key)?),
        ttir::Attr::DenseSplat(inner, t) => match &**inner {
            ttir::Attr::Float(bits, et) => {
                Attr::SplatFloat(float_bits(bits.get(), et, op_name, key)?)
            }
            // AN INTEGER DENSE SPLAT HAS NO MODELLED FORM, and the reader agrees: for
            // `tt.divisibility = dense<64> : tensor<1xi32>` -- the one in tree, on
            // `attention_flash_causal`'s `arith.muli` -- `parse_attr_value` falls through
            // every arm and returns `Attr::Verbatim("dense<64> : tensor<1xi32>")`. So this
            // reproduces that string EXACTLY rather than inventing a `SplatInt`, because
            // the diff compares the two and a better representation on one side is a
            // reported difference.
            ttir::Attr::Int(i, _) => Attr::Verbatim(format!("dense<{i}> : {t}")),
            other => {
                return Err(Refusal::new(
                    PASS,
                    format!(
                        "`{op_name}`'s `{key}` is a dense splat of {other:?}, which has no \
                         KTIR attribute form"
                    ),
                ));
            }
        },
        ttir::Attr::Bool(b) => Attr::Bool(*b),
        ttir::Attr::Str(s) => Attr::Str(s.clone()),
        ttir::Attr::Unit => Attr::Unit,
        // `tt.get_program_id x` -- THE AXIS IS A BARE KEYWORD in MLIR, and the reader has
        // a hand-written arm turning it into `Attr::Int(0|1|2)`. Matching that is not
        // cosmetic: `DistributeWork` reads this to decide whether the kernel is
        // multi-axis, and every axis arriving as 0 makes a two-axis grid look like one.
        ttir::Attr::Axis(a) => Attr::Int(match *a {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            other => {
                return Err(Refusal::new(
                    PASS,
                    format!("`{op_name}`'s axis is `{other}`; MLIR has x, y and z"),
                ));
            }
        }),
        // `arith.cmpi slt` -- also a bare keyword, also a hand-written reader arm. No
        // GOLDEN has a compare (the hole was found by reading the reader, not by a
        // failing test); the router-index kernels send live ones through here now.
        ttir::Attr::Pred(p) => Attr::Str(p.to_string()),
        // `order = array<i32: 1, 0>` -> the reader's `array<` arm, an `IntList`.
        ttir::Attr::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    ttir::Attr::Int(i, _) => out.push(narrow(*i, op_name, key)?),
                    other => {
                        return Err(Refusal::new(
                            PASS,
                            format!(
                                "`{op_name}`'s `{key}` is an array containing {other:?}; \
                                 `Attr::IntList` is the only list form, so a non-integer \
                                 element has nowhere to go"
                            ),
                        ));
                    }
                }
            }
            Attr::IntList(out)
        }
        ttir::Attr::Type(t) => {
            return Err(Refusal::new(
                PASS,
                format!(
                    "`{op_name}`'s `{key}` is the type attribute `{t}`; `ir::Attr` has no \
                     type variant"
                ),
            ));
        }
    })
}

/// `i128` -> `i64`, refusing rather than truncating.
fn narrow(v: i128, op_name: &str, key: &str) -> Result<i64> {
    i64::try_from(v).map_err(|_| {
        Refusal::new(
            PASS,
            format!(
                "`{op_name}`'s `{key}` is {v}, which does not fit in the `i64` \
                 `ir::Attr::Int` holds"
            ),
        )
    })
}

/// A float constant AT ITS TYPE's width, matching `text::parse`'s `parse_float`.
///
/// Storing an f16 constant as 64 bits of double is the trap this exists to avoid:
/// `FloatBits` compares and hashes by BITS, so a width mismatch makes two constants that
/// denote the same number compare unequal, and `LegalizeTypes`'s re-round then has nothing
/// to re-round from.
fn float_bits(v: f64, t: &ttir::Type, op_name: &str, key: &str) -> Result<FloatBits> {
    match t.scalar() {
        ttir::Type::Float(FloatKind::F16) => Ok(FloatBits::f16_from_f32(v as f32)),
        ttir::Type::Float(FloatKind::F32) => Ok(FloatBits::f32(v as f32)),
        other => Err(Refusal::new(
            PASS,
            format!(
                "`{op_name}`'s `{key}` is a float of type `{other}`; `FloatBits` is \
                 modelled at 16 and 32 bits"
            ),
        )),
    }
}

/// A census of what came in and what went out, so a test asserts on numbers rather than
/// on a printed module.
///
/// IN MUST EQUAL OUT. The adapter neither adds nor removes an operation -- it is a change
/// of representation and nothing else -- so a difference here is the bug, not a
/// simplification.
pub fn census(m: &ttir::Module) -> (usize, usize) {
    fn walk(ops: &[ttir::Op], n: &mut usize) {
        for o in ops {
            *n += 1;
            for r in &o.regions {
                for b in &r.blocks {
                    walk(&b.ops, n);
                }
            }
        }
    }
    let mut n = 0;
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut n);
        }
    }
    (m.funcs.len(), n)
}

/// `census`, for the converted side. The function op itself is not counted, so the two
/// numbers are comparable.
pub fn census_ktir(m: &Module) -> (usize, usize) {
    let funcs = m
        .ops
        .iter()
        .filter(|o| o.kind == OpKind::TtFunc || o.kind == OpKind::FuncFunc)
        .count();
    let ops = m.ops_deep().len() - funcs;
    (funcs, ops)
}

/// Every value the module defines, so a test can check the arena crossed intact.
pub fn value_ids(m: &ttir::Module) -> Vec<ValueId> {
    (0..m.values.len() as u32).map(ValueId).collect()
}

const _: Option<fn(&ir::Module)> = None;
