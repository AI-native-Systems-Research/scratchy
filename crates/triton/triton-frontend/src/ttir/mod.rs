//! # THE MODULE BOUNDARY: the in-memory TTIR value.
//!
//! **This module is the interface between bridge one (Python -> TTIR) and bridge two
//! (TTIR -> KTIR), and it is the ONLY module either side shares.** It is deliberately
//! small and deliberately dumb: types, attributes, ops, regions, and a value arena.
//! There is no builder logic here, no semantics, no Python, and no target policy.
//!
//! ## NOTHING IS SERIALIZED AND NOTHING IS PARSED.
//!
//! There is no text form of a program at any point in the compile path. `print.rs`
//! renders this value to TTIR text and `parse.rs` reads TTIR text back, but BOTH ARE
//! TEST INSTRUMENTS: `print` exists for the golden diff and for debugging, `parse` exists
//! only to read the goldens the existing toolchain emits so they can be compared. No
//! stage of the compiler calls either one. If you find yourself printing a module so
//! another stage can parse it, that is the bug.
//!
//! ## WHY THE BOUNDARY IS ONE MODULE.
//!
//! The KTIR types downstream may become scratchy's vendored `ktir-core` rather than
//! ours; that decision is open. Confining the shared vocabulary to this module means
//! swapping it is a contained change. What depends on it:
//!
//!   * `crate::semantic` and `crate::codegen` CONSTRUCT it (through `Builder`, below).
//!   * `crate::ttir::print` and `crate::ttir::parse` RENDER and READ it (tests only).
//!   * bridge two CONSUMES it.
//!
//! Nothing else. In particular `crate::py` (the Python AST) never mentions a TTIR type,
//! and this module never mentions a Python one.

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

pub mod parse;
pub mod print;

// ===----------------------------------------------------------------------===//
//                                    Types
// ===----------------------------------------------------------------------===//

/// Integer signedness. Triton tracks this in the language type even though MLIR's `iN`
/// does not, because it decides `divsi` vs `divui`, `extsi` vs `extui`, and the compare
/// predicates. See `python/triton/language/core.py`'s `dtype.SIGNEDNESS`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
    /// `i1` / MLIR-level integers with no language signedness.
    Signless,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    F64,
    F8E4M3FN,
    F8E5M2,
}

impl FloatKind {
    pub fn mlir(self) -> &'static str {
        match self {
            FloatKind::F16 => "f16",
            FloatKind::BF16 => "bf16",
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::F8E4M3FN => "f8E4M3FN",
            FloatKind::F8E5M2 => "f8E5M2",
        }
    }

    pub fn bitwidth(self) -> u32 {
        match self {
            FloatKind::F16 | FloatKind::BF16 => 16,
            FloatKind::F32 => 32,
            FloatKind::F64 => 64,
            FloatKind::F8E4M3FN | FloatKind::F8E5M2 => 8,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Type {
    /// `iN`. Signedness is language-level; the MLIR spelling ignores it.
    Int(u32, Signedness),
    Float(FloatKind),
    /// `!tt.ptr<T>` (address space is always 1 for the kernels here; kept explicit so a
    /// non-default space cannot be silently dropped).
    Ptr(Rc<Type>, u32),
    /// `tensor<AxBxT>`. An empty shape is NOT a legal tensor here -- a rank-0 value is
    /// the scalar `T` itself, which is how Triton models scalars.
    Tensor(Vec<i64>, Rc<Type>),
    /// `!tt.tensordesc<AxBxT>`
    TensorDesc(Vec<i64>, Rc<Type>),
    /// The result-less `tt.return` / `tt.descriptor_store` case.
    Void,
}

impl Type {
    pub fn i(bits: u32) -> Type {
        Type::Int(bits, Signedness::Signed)
    }
    pub fn i1() -> Type {
        Type::Int(1, Signedness::Signless)
    }
    pub fn i32() -> Type {
        Type::Int(32, Signedness::Signed)
    }
    pub fn i64() -> Type {
        Type::Int(64, Signedness::Signed)
    }
    pub fn f16() -> Type {
        Type::Float(FloatKind::F16)
    }
    pub fn f32() -> Type {
        Type::Float(FloatKind::F32)
    }
    pub fn tensor(shape: &[i64], elem: Type) -> Type {
        Type::Tensor(shape.to_vec(), Rc::new(elem))
    }
    pub fn desc(shape: &[i64], elem: Type) -> Type {
        Type::TensorDesc(shape.to_vec(), Rc::new(elem))
    }
    pub fn ptr(elem: Type) -> Type {
        Type::Ptr(Rc::new(elem), 1)
    }

    /// The element type of a tensor, or the type itself for a scalar. Triton's
    /// `type.scalar`.
    pub fn scalar(&self) -> &Type {
        match self {
            Type::Tensor(_, e) => e,
            other => other,
        }
    }

    /// `[]` for a scalar. Triton's `type.shape` (which raises for scalars; here an empty
    /// slice is the honest answer and callers that care check `is_tensor`).
    pub fn shape(&self) -> &[i64] {
        match self {
            Type::Tensor(s, _) | Type::TensorDesc(s, _) => s,
            _ => &[],
        }
    }

    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self.scalar(), Type::Int(..))
    }

    pub fn is_floating(&self) -> bool {
        matches!(self.scalar(), Type::Float(..))
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self.scalar(), Type::Ptr(..))
    }

    pub fn int_bitwidth(&self) -> Option<u32> {
        match self.scalar() {
            Type::Int(b, _) => Some(*b),
            _ => None,
        }
    }

    pub fn signedness(&self) -> Signedness {
        match self.scalar() {
            Type::Int(_, s) => *s,
            _ => Signedness::Signless,
        }
    }

    /// Replace the element type, keeping the shape. Triton's
    /// `type.with_element_ty(...)`.
    pub fn with_element_ty(&self, elem: Type) -> Type {
        match self {
            Type::Tensor(s, _) => Type::Tensor(s.clone(), Rc::new(elem)),
            Type::TensorDesc(s, _) => Type::TensorDesc(s.clone(), Rc::new(elem)),
            _ => elem,
        }
    }

    /// Number of elements; 1 for a scalar.
    pub fn numel(&self) -> i64 {
        self.shape().iter().product::<i64>().max(1)
    }
}

/// The element spelling MLIR uses INSIDE a `!tt.tensordesc<>` block type.
///
/// # This is the one place signedness IS printed, and it is measured, not chosen.
///
/// Everywhere else the MLIR spelling of an integer drops Triton's signedness (`i32`), which
/// is why [`Type`]'s `Display` prints `i32` for a tensor element and for a pointer's
/// pointee. A tensordesc's block type does NOT: `tt.make_tensor_descriptor` over an
/// `!tt.ptr<i32>` prints `: <i32>, <64xsi32>` -- signless pointee, SIGNED block element --
/// and its result type as `!tt.tensordesc<64xsi32>`. That is in
/// `tests/goldens/embedding.ttir_raw.mlir`, the first golden in the tree with a non-float
/// descriptor; every earlier one is f16, where signedness cannot show.
///
/// MEASURED rather than extrapolated from the one case, by compiling a kernel that makes one
/// descriptor per width (`make_ir`, not this crate): `*u32 -> <64xui32>`, `*i16 -> <64xsi16>`,
/// `*i8 -> <64xsi8>`, `*i64 -> <64xsi64>`, `*fp32 -> <64xf32>`. The `i1` arm below is a
/// FALLBACK and not a measurement: Triton refuses an i1 descriptor outright, at
/// `make_tensor_descriptor` -- "Descriptor block shape must have at least 16 bytes in the
/// last dimension, but got 64 * 0 = 0 bytes" -- so no spelling for one exists to reproduce.
///
/// Printing `i32` here instead is not cosmetic: `tests/no_parser.rs` asserts that every
/// golden parses, so the reader has to accept `si32`, and the diff compares result types by
/// their printed spelling, so both sides must agree on which one it is.
pub fn desc_block_elem(t: &Type) -> String {
    match t {
        Type::Int(1, _) => "i1".to_string(),
        Type::Int(b, Signedness::Signed) => format!("si{b}"),
        Type::Int(b, Signedness::Unsigned) => format!("ui{b}"),
        other => other.to_string(),
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // MLIR has no unsigned integer spelling in these dialects: `i32` covers both,
            // and the OPERATION carries the signedness (divsi/divui). Printing `ui32`
            // here would produce text MLIR cannot round-trip.
            Type::Int(b, _) => write!(f, "i{b}"),
            Type::Float(k) => f.write_str(k.mlir()),
            Type::Ptr(e, _) => write!(f, "!tt.ptr<{e}>"),
            Type::Tensor(s, e) => {
                f.write_str("tensor<")?;
                for d in s {
                    write!(f, "{d}x")?;
                }
                write!(f, "{e}>")
            }
            Type::TensorDesc(s, e) => {
                f.write_str("!tt.tensordesc<")?;
                for d in s {
                    write!(f, "{d}x")?;
                }
                write!(f, "{}>", desc_block_elem(e))
            }
            Type::Void => f.write_str("()"),
        }
    }
}

// ===----------------------------------------------------------------------===//
//                                 Attributes
// ===----------------------------------------------------------------------===//

/// A float constant kept in a form that PRINTS DETERMINISTICALLY.
///
/// MLIR prints `arith.constant` floats in a fixed scientific form (`0.000000e+00`,
/// `1.000000e+00`) and the golden diff compares attributes, so the bit pattern is what is
/// stored and the formatting is done once, in `print.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct F64Bits(pub u64);

impl F64Bits {
    pub fn new(v: f64) -> F64Bits {
        F64Bits(v.to_bits())
    }
    pub fn get(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Round `v` to what the target float type can actually hold, then keep it as an f64.
    ///
    /// # This is not tidiness: an unrounded constant does not match the oracle.
    ///
    /// MLIR stores an `arith.constant`'s value AT ITS TYPE, so an f32 constant built from the
    /// Python literal `1.44269504` prints as `1.44269502` -- the nearest f32. Storing the full
    /// double made every such constant differ from the golden by a few ULP while looking
    /// correct.
    ///
    /// f16 is rounded through a manual encode/decode because Rust has no `f16` here; it
    /// handles subnormals, overflow to infinity, and NaN.
    pub fn rounded(v: f64, ty: &Type) -> F64Bits {
        match ty.scalar() {
            Type::Float(FloatKind::F32) => F64Bits::new(v as f32 as f64),
            Type::Float(FloatKind::F16) => F64Bits::new(f16_round(v)),
            // bf16 keeps the top 16 bits of an f32.
            Type::Float(FloatKind::BF16) => {
                let bits = (v as f32).to_bits() & 0xFFFF_0000;
                F64Bits::new(f32::from_bits(bits) as f64)
            }
            _ => F64Bits::new(v),
        }
    }
}

/// Round an f64 to the nearest IEEE binary16 value, returned as an f64.
fn f16_round(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() {
        return v;
    }
    let f = v as f32;
    let bits = f.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    let sgn = if sign == 1 { -1.0f64 } else { 1.0f64 };
    // Unbiased exponent; f16's range is 2^-24 (smallest subnormal) to 65504.
    let e = exp - 127;
    if e > 15 {
        return sgn * f64::INFINITY;
    }
    if e < -24 {
        return sgn * 0.0;
    }
    // Number of mantissa bits available at this exponent: 10 normally, fewer when subnormal.
    let keep: i32 = if e < -14 { 10 - (-14 - e) } else { 10 };
    let shift = 23 - keep;
    let step = 1u32 << shift;
    let half = step / 2;
    let rem = mant & (step - 1);
    let mut rounded = mant & !(step - 1);
    // Round half to even, as IEEE and MLIR both do.
    if rem > half || (rem == half && (rounded & step) != 0) {
        rounded += step;
    }
    let out = f32::from_bits((sign << 31) | ((exp as u32) << 23) | (rounded & 0x007F_FFFF));
    // A mantissa carry can push the exponent up; from_bits above already absorbed it when
    // `rounded` overflowed into the exponent field.
    let carried = if rounded > 0x007F_FFFF {
        f32::from_bits((sign << 31) | (((exp + 1) as u32) << 23))
    } else {
        out
    };
    if carried.is_infinite() {
        return sgn * f64::INFINITY;
    }
    carried as f64
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Attr {
    Int(i128, Type),
    Float(F64Bits, Type),
    /// `dense<V> : tensor<...>` -- a splat. Triton only ever emits splat dense attrs from
    /// the AST (`tl.zeros`, `tl.full`); a non-splat dense would come from a pass.
    DenseSplat(Box<Attr>, Type),
    Bool(bool),
    Str(String),
    /// A bare `unit` attribute, e.g. `tt.flatten`.
    Unit,
    /// A comparison predicate keyword: `arith.cmpi`'s `sle`, `arith.cmpf`'s `olt`, ...
    Pred(&'static str),
    /// `tt.get_program_id`'s axis: `x` / `y` / `z`.
    Axis(&'static str),
    Array(Vec<Attr>),
    Type(Type),
}

// ===----------------------------------------------------------------------===//
//                                  Locations
// ===----------------------------------------------------------------------===//

/// A source location.
///
/// The NAME carried by a `Loc::Name` is not cosmetic and is not the printer's invention:
/// it is the Python variable the value was assigned to, attached by Triton's
/// `_maybe_set_loc_to_name` (`code_generator.py:438`), and MLIR's asm printer derives the
/// `%ssa` name from it. That makes it real AST-derived content, so the golden diff
/// compares it. What the diff does NOT compare is the `#locN` numbering or the SSA
/// suffixes MLIR appends to disambiguate -- those are printer bookkeeping.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub enum Loc {
    #[default]
    Unknown,
    File {
        file: Rc<str>,
        line: u32,
        col: u32,
    },
    Name(String, Box<Loc>),
}

impl Loc {
    /// The OUTERMOST name, if any. This is the one MLIR's asm printer uses for the SSA
    /// name, so it is the one the diff compares.
    pub fn name(&self) -> Option<&str> {
        match self {
            Loc::Name(n, _) => Some(n),
            _ => None,
        }
    }

    /// Attach a name, REPLACING one already present.
    ///
    /// This models `CodeGenerator.visit`'s
    /// `set_loc(create_name_loc(name_loc_as_prefix, here_loc))` -- `here_loc` is always a
    /// freshly built file location, so nothing is ever nested there.
    pub fn named(&self, name: &str) -> Loc {
        let inner = match self {
            Loc::Name(_, inner) => (**inner).clone(),
            other => other.clone(),
        };
        Loc::Name(name.to_string(), Box::new(inner))
    }

    /// Attach a name, NESTING inside whatever is already there.
    ///
    /// This models `_maybe_set_loc_to_name` (`code_generator.py:438`), which does
    /// `val.set_loc(create_name_loc(name, val.get_loc()))` -- it wraps the value's CURRENT
    /// location, so naming a value twice produces a nested chain.
    ///
    /// MEASURED, and it is not a detail one would guess: the inner `scf.for` of
    /// `swiglu_mlp` carries two values, `g` and `u`, and its location in the golden is
    /// `#loc62 = loc("u"(#loc48))` with `#loc48 = loc("g"(#loc13))`. Both carries were named
    /// in turn onto the same location and the names stacked, with `u` outermost because it
    /// was named last.
    pub fn wrap_name(&self, name: &str) -> Loc {
        Loc::Name(name.to_string(), Box::new(self.clone()))
    }
}

// ===----------------------------------------------------------------------===//
//                              Values and ops
// ===----------------------------------------------------------------------===//

/// An index into the module's value arena. Def-use is carried by these ids, not by
/// printed names, so the diff can normalize names away without losing the graph.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ValueId(pub u32);

#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub ty: Type,
    pub loc: Loc,
}

/// An operation. `name` is the full MLIR spelling (`arith.addf`, `tt.descriptor_load`).
///
/// # INVARIANT A CONSUMER MAY RELY ON: a result's location IS its op's location.
///
/// In MLIR an operation has exactly one `Location`, and `OpResult::setLoc` sets the
/// **owning operation's** location -- so every result of a multi-result op shares one
/// location, and naming one result renames the op. Triton leans on this: naming a loop's
/// carried results through `_maybe_set_loc_to_name` is what gives `scf.for` its own printed
/// location.
///
/// This crate stores locations per value (block arguments genuinely have their own), so the
/// invariant is maintained rather than structural: everything that sets an op's location
/// sets its results' too, via [`crate::semantic::Semantic::name_op_results`]. Read either;
/// they agree.
///
/// A single string rather than an enum is a decision: the set of ops the front end emits
/// is fixed by `semantic.rs`, and every one is written in exactly one place there, so an
/// enum would add a layer without adding a check. What DOES check the set is
/// `src/py/census.rs` on the input side and the golden diff on the output side.
#[derive(Clone, Debug)]
pub struct Op {
    pub name: String,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    /// Sorted by key so the printer and the diff see a stable order.
    pub attrs: BTreeMap<String, Attr>,
    pub regions: Vec<Region>,
    pub loc: Loc,
}

impl Op {
    pub fn new(name: impl Into<String>, loc: Loc) -> Op {
        Op {
            name: name.into(),
            operands: Vec::new(),
            results: Vec::new(),
            attrs: BTreeMap::new(),
            regions: Vec::new(),
            loc,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub args: Vec<ValueId>,
    pub ops: Vec<Op>,
}

/// A region: one or more blocks, in order.
///
/// # A CONSUMER MUST HANDLE MORE THAN ONE BLOCK.
///
/// A function body is not always a single block, and the extra ones are not exotic --
/// every `@triton.jit` helper with an explicit `return` has two. `visit_Return`
/// (`code_generator.py:544`) records the return and then **creates a fresh block** so the
/// terminator ends its own block; `handle_returns` (`code_generator.py:606`) later emits the
/// real `tt.return` into the recorded block and, into the last (unreachable) one, a
/// `ub.poison` per result plus another `tt.return`. That is the
/// `^bb1: // no predecessors` block visible in every generated `standard.*` helper in the
/// goldens.
///
/// So a consumer walking a region must iterate `blocks`, and must expect a trailing block
/// that is dead. It is dead but well formed: correctly typed, correctly terminated, and MLIR
/// keeps it until a pass removes it.
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub blocks: Vec<Block>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub visibility: Visibility,
    pub noinline: bool,
    pub arg_types: Vec<Type>,
    pub ret_types: Vec<Type>,
    pub body: Region,
    pub loc: Loc,
}

/// A whole TTIR module: the value bridge one hands bridge two.
#[derive(Clone, Debug, Default)]
pub struct Module {
    pub values: Vec<ValueInfo>,
    pub funcs: Vec<Func>,
    pub loc: Loc,
}


impl Module {
    pub fn ty(&self, v: ValueId) -> &Type {
        &self.values[v.0 as usize].ty
    }
    pub fn loc_of(&self, v: ValueId) -> &Loc {
        &self.values[v.0 as usize].loc
    }
    pub fn set_loc(&mut self, v: ValueId, loc: Loc) {
        self.values[v.0 as usize].loc = loc;
    }
    pub fn func(&self, name: &str) -> Option<&Func> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
