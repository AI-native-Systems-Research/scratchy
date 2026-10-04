//! A port of `python/triton/language/semantic.py`: the type rules and the op emission.
//!
//! Every rule here is sourced to a DEFINITION in that file, cited by line, and the port is
//! faithful except where [`crate::target`] says otherwise. Two habits this file keeps
//! deliberately, because both were needed to match the oracle:
//!
//!   * **Dead ops are emitted where Triton emits them.** `binary_op_type_checking_impl`
//!     materializes a scalar operand as a constant, then `scalar_constant` materializes it
//!     AGAIN, leaving the first one dead. The canonicalizer deletes it. Not reproducing
//!     that would make the raw TTIR diff fail on op counts, so it is reproduced.
//!   * **Emission order is the Python call order.** The overflow check for `a * b` emits
//!     six ops BEFORE the multiply it is checking, because
//!     `binary_op_sanitize_overflow_impl` runs before `create_mul`.

use crate::target::Target;
use crate::ttir::{
    Attr, Block, F64Bits, FloatKind, Loc, Module, Op, Region, Signedness, Type, ValueId, ValueInfo,
};
use crate::{Error, Result};

/// A language-level value: either an IR value or a compile-time constant.
///
/// Triton's `constexpr` and `tensor` distinction, which decides whether an operation
/// happens at compile time or becomes an op.
#[derive(Clone, Debug)]
pub enum Val {
    /// An SSA value in the module (scalar or tensor or descriptor).
    Ir(ValueId),
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    None,
    /// `tl.float16` and friends, used as an argument to `.to()` / `tl.zeros` / `tl.full`.
    Dtype(Type),
    /// A Python list or tuple of values.
    Seq(Vec<Val>),
    /// A slice appearing in a subscript, e.g. the `:` of `x[:, None]`.
    Slice,
    /// A TENSOR DESCRIPTOR, which is a COMPOSITE value: the `!tt.tensordesc` handle plus one
    /// IR value per shape entry and per stride entry.
    ///
    /// # Why this is not just a `ValueId`, and why it matters at a call boundary.
    ///
    /// Triton's `tensor_descriptor` is a `base_value` whose type flattens to MORE THAN ONE
    /// IR value (`core.py:1502`): the base type's handle, then `shape_type`, then
    /// `strides_type`. Inside one function that is invisible -- `tt.descriptor_load` takes
    /// only the handle -- but a descriptor passed to another `@triton.jit` function
    /// contributes `1 + 2 * rank` arguments.
    ///
    /// Measured in `attention_flash`'s golden, whose `_attn_fwd_inner` takes three rank-2
    /// descriptors and therefore FIFTEEN parameters for them:
    ///
    /// ```text
    /// %desc_k: !tt.tensordesc<64x128xf16>, %desc_k.shape.0: i32, %desc_k.shape.1: i32,
    ///                                      %desc_k.stride.0: i64, %desc_k.stride.1: i64,
    /// ```
    ///
    /// The mangled name counts the descriptor ONCE (`TDfp16S64_128S`), because mangling is
    /// over the language type, not the flattened one -- so the two must not be conflated.
    Desc {
        handle: ValueId,
        shape: Vec<ValueId>,
        strides: Vec<ValueId>,
    },
}

impl Val {
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Val::Int(v) => Some(*v),
            Val::Bool(b) => Some(*b as i128),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Float(v) => Some(*v),
            Val::Int(v) => Some(*v as f64),
            Val::Bool(b) => Some(*b as i32 as f64),
            _ => None,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(
            self,
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Str(_) | Val::None
        )
    }

    /// Triton's `numbers.Number` test in `binary_op_type_checking_impl`: a bool, int or
    /// float constant participates in promotion as a SCALAR. A string or None does not.
    pub fn is_number(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Float(_) | Val::Bool(_))
    }

    /// `_is_triton_value` (`code_generator.py:44`): whether this is a `base_value`, i.e.
    /// something backed by IR rather than a compile-time constant.
    ///
    /// A `language.tuple` IS a base value even when all its members are constexpr, so a
    /// sequence counts -- it simply flattens to no handles.
    pub fn is_triton_value(&self) -> bool {
        matches!(self, Val::Ir(_) | Val::Desc { .. } | Val::Seq(_))
    }

    /// `flatten_values_to_ir`: the IR handles this value contributes, IN ORDER.
    ///
    /// This is what `_find_carries` compares to decide whether a loop changed a value
    /// (`code_generator.py:492`), and what a `tt.call` passes as operands. A descriptor
    /// contributes `1 + 2 * rank` of them -- see [`Val::Desc`].
    pub fn ir_handles(&self) -> Vec<ValueId> {
        match self {
            Val::Ir(id) => vec![*id],
            Val::Desc {
                handle,
                shape,
                strides,
            } => {
                let mut out = vec![*handle];
                out.extend_from_slice(shape);
                out.extend_from_slice(strides);
                out
            }
            Val::Seq(items) => items.iter().flat_map(|v| v.ir_handles()).collect(),
            _ => Vec::new(),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Val::Ir(_) => "tensor",
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Bool(_) => "bool",
            Val::Str(_) => "str",
            Val::None => "None",
            Val::Dtype(_) => "dtype",
            Val::Seq(_) => "sequence",
            Val::Slice => "slice",
            Val::Desc { .. } => "tensor descriptor",
        }
    }
}

/// The dtype "kind" ordering rule 0 of `computation_type_impl` uses:
/// `bool < uint < int < fp` (`core.py`'s `dtype.kind()`).
fn kind_rank(t: &Type) -> u8 {
    match t.scalar() {
        Type::Int(1, _) => 0,
        Type::Int(_, Signedness::Unsigned) => 1,
        Type::Int(..) => 2,
        Type::Float(_) => 3,
        _ => 4,
    }
}

/// Where ops are being appended: a region under construction.
///
/// A stack of these models Triton's insertion point. Each frame holds a LIST OF BLOCKS
/// because a region is not always one block -- `visit_Return` starts a fresh one so the
/// terminator ends its own block, which is where the goldens' `^bb1: // no predecessors`
/// comes from. Ops go into the LAST block.
struct Frame {
    blocks: Vec<Block>,
}

impl Frame {
    fn new() -> Frame {
        Frame {
            blocks: vec![Block::default()],
        }
    }
}

/// The IR builder plus the semantic rules.
pub struct Semantic {
    pub module: Module,
    pub target: Target,
    /// The file every `Loc::File` refers to. The goldens embed an absolute path from the
    /// machine that generated them, which is why `crate::diff` does not compare it.
    pub file: std::rc::Rc<str>,
    frames: Vec<Frame>,
    /// The location new ops get, set from the AST node being visited.
    pub loc: Loc,
    /// The name prefix `_name_loc_prefix` installs while visiting an assignment's RHS.
    pub name_prefix: Option<String>,
}

impl Semantic {
    pub fn new(file: &str, target: Target) -> Semantic {
        Semantic {
            module: Module::default(),
            target,
            file: std::rc::Rc::from(file),
            frames: vec![Frame::new()],
            loc: Loc::Unknown,
            name_prefix: None,
        }
    }

    // -- insertion points ----------------------------------------------------

    pub fn push_frame(&mut self) {
        self.frames.push(Frame::new());
    }

    /// Finish the current region and return its blocks.
    pub fn pop_frame(&mut self) -> Vec<Block> {
        self.frames.pop().expect("frame underflow").blocks
    }

    /// Discard a frame entirely. `_find_carries` builds a block, dry-visits the loop body
    /// into it, then ERASES it (`code_generator.py:474`) -- the ops are thrown away and only
    /// the set of changed names survives. Values allocated during the dry run stay in the
    /// arena, unreferenced, exactly as MLIR leaves them, which is why the goldens' SSA
    /// numbering jumps across a loop.
    pub fn discard_frame(&mut self) {
        self.frames.pop();
    }

    /// Start a new block in the current region and make it the insertion point. Returns its
    /// index, so a caller can come back and append to an earlier block the way
    /// `handle_returns` does.
    pub fn start_block(&mut self, args: Vec<ValueId>) -> usize {
        let f = self.frames.last_mut().expect("no frame");
        f.blocks.push(Block {
            args,
            ops: Vec::new(),
        });
        f.blocks.len() - 1
    }

    /// The index of the block ops are currently going into.
    pub fn current_block(&self) -> usize {
        self.frames.last().expect("no frame").blocks.len() - 1
    }

    fn emit(&mut self, op: Op) {
        let f = self.frames.last_mut().expect("no frame");
        f.blocks.last_mut().expect("no block").ops.push(op);
    }

    /// Append an already-built op. Used for `tt.return` and `scf.yield`, which carry no
    /// results and are built by the code generator rather than by a rule here.
    pub fn emit_raw(&mut self, op: Op) {
        self.emit(op);
    }

    /// Append to a SPECIFIC block of the current region, which is how `handle_returns`
    /// writes each `tt.return` back into the block its `return` statement was in.
    pub fn emit_in_block(&mut self, block: usize, op: Op) {
        let f = self.frames.last_mut().expect("no frame");
        f.blocks[block].ops.push(op);
    }

    /// Set an op's location AND its results' locations together, keeping the invariant that
    /// a result's location is its op's (see [`crate::ttir::Op`]). `nest` selects
    /// `_maybe_set_loc_to_name`'s wrapping behaviour over plain replacement.
    pub fn name_op_results(&mut self, block: usize, op_index: usize, name: &str, nest: bool) {
        let f = self.frames.last_mut().expect("no frame");
        let op = &mut f.blocks[block].ops[op_index];
        let next = if nest {
            op.loc.wrap_name(name)
        } else {
            op.loc.named(name)
        };
        op.loc = next.clone();
        let results = op.results.clone();
        for r in results {
            self.module.values[r.0 as usize].loc = next.clone();
        }
    }

    /// Index of the last op in the current block, for a caller that needs to rename it.
    pub fn last_op_index(&self) -> usize {
        let f = self.frames.last().expect("no frame");
        f.blocks.last().expect("no block").ops.len() - 1
    }

    pub fn file_loc(&self, line: u32, col: u32) -> Loc {
        Loc::File {
            file: self.file.clone(),
            line,
            col,
        }
    }

    /// The location an op gets, with the pending assignment name applied.
    fn cur_loc(&self) -> Loc {
        match &self.name_prefix {
            Some(n) => self.loc.named(n),
            None => self.loc.clone(),
        }
    }

    // -- value creation ------------------------------------------------------

    fn new_value(&mut self, ty: Type, loc: Loc) -> ValueId {
        let id = ValueId(self.module.values.len() as u32);
        self.module.values.push(ValueInfo { ty, loc });
        id
    }

    pub fn ty(&self, v: ValueId) -> Type {
        self.module.ty(v).clone()
    }

    /// Emit an op with one result.
    fn emit1(
        &mut self,
        name: &str,
        operands: &[ValueId],
        res_ty: Type,
        attrs: &[(&str, Attr)],
    ) -> ValueId {
        let loc = self.cur_loc();
        let res = self.new_value(res_ty, loc.clone());
        let mut op = Op::new(name, loc);
        op.operands = operands.to_vec();
        op.results = vec![res];
        for (k, v) in attrs {
            op.attrs.insert((*k).to_string(), v.clone());
        }
        self.emit(op);
        res
    }

    /// Emit an op with no results.
    fn emit0(&mut self, name: &str, operands: &[ValueId], attrs: &[(&str, Attr)]) {
        let loc = self.cur_loc();
        let mut op = Op::new(name, loc);
        op.operands = operands.to_vec();
        for (k, v) in attrs {
            op.attrs.insert((*k).to_string(), v.clone());
        }
        self.emit(op);
    }

    /// Emit an op carrying a region, whose blocks come from a popped frame.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_with_region(
        &mut self,
        name: &str,
        operands: &[ValueId],
        res_tys: &[Type],
        attrs: &[(&str, Attr)],
        blocks: Vec<Block>,
        loc: Loc,
    ) -> Vec<ValueId> {
        let results: Vec<ValueId> = res_tys
            .iter()
            .map(|t| self.new_value(t.clone(), loc.clone()))
            .collect();
        let mut op = Op::new(name, loc);
        op.operands = operands.to_vec();
        op.results = results.clone();
        for (k, v) in attrs {
            op.attrs.insert((*k).to_string(), v.clone());
        }
        op.regions.push(Region { blocks });
        self.emit(op);
        results
    }

    /// `ub.poison`: `visit_For`'s induction-variable placeholder, and `handle_returns`'
    /// terminator for the unreachable block.
    pub fn poison(&mut self, ty: Type) -> ValueId {
        self.emit1("ub.poison", &[], ty, &[])
    }

    /// `tt.call` to an already-generated function.
    pub fn call(&mut self, callee: &str, args: &[ValueId], res_tys: &[Type]) -> Vec<ValueId> {
        let loc = self.cur_loc();
        let results: Vec<ValueId> = res_tys
            .iter()
            .map(|t| self.new_value(t.clone(), loc.clone()))
            .collect();
        let mut op = Op::new("tt.call", loc);
        op.operands = args.to_vec();
        op.results = results.clone();
        op.attrs
            .insert("callee".to_string(), Attr::Str(callee.to_string()));
        self.emit(op);
        results
    }

    /// A splat: `tt.splat`, except that MLIR's builder FOLDS a constant operand into a
    /// dense splat constant at build time.
    ///
    /// MEASURED: `tl.full([64, 64], 1.0, tl.float16)` is `splat(make_scalar(...))`, and the
    /// golden shows `arith.constant 1.000000e+00 : f16` followed by
    /// `arith.constant dense<1.000000e+00> : tensor<64x64xf16>` -- a second CONSTANT, not a
    /// `tt.splat`. The scalar constant is left dead. Emitting `tt.splat` here instead would
    /// differ from the oracle by one op name and one dead value.
    pub fn splat(&mut self, v: ValueId, shape: &[i64]) -> Result<ValueId> {
        let ty = self.ty(v);
        if ty.is_tensor() {
            return Err(Error::new(
                format!("cannot splat an already-shaped value of type {ty}"),
                0,
                0,
            ));
        }
        if shape.is_empty() {
            return Ok(v);
        }
        let out = Type::Tensor(shape.to_vec(), std::rc::Rc::new(ty.clone()));
        if let Some(attr) = self.constant_value_of(v) {
            return Ok(self.emit1(
                "arith.constant",
                &[],
                out.clone(),
                &[("value", Attr::DenseSplat(Box::new(attr), out))],
            ));
        }
        Ok(self.emit1("tt.splat", &[v], out, &[]))
    }

    /// The `value` attribute of `v` if it is defined by an `arith.constant` in the block
    /// currently being built. Used only by [`Self::splat`] to reproduce MLIR's fold.
    fn constant_value_of(&self, v: ValueId) -> Option<Attr> {
        let f = self.frames.last()?;
        for b in f.blocks.iter().rev() {
            for op in b.ops.iter().rev() {
                if op.results.first() == Some(&v) {
                    if op.name != "arith.constant" {
                        return None;
                    }
                    return op.attrs.get("value").cloned();
                }
            }
        }
        None
    }

    pub fn new_block_arg(&mut self, ty: Type, loc: Loc) -> ValueId {
        self.new_value(ty, loc)
    }

    // ===------------------------------------------------------------------===//
    //                              Constants
    // ===------------------------------------------------------------------===//

    /// `to_tensor_type` (`semantic.py:129`): the dtype a Python literal takes on.
    pub fn literal_type(&self, v: &Val) -> Result<Type> {
        Ok(match v {
            Val::Bool(_) => Type::i1(),
            Val::Int(x) => {
                if (-(1i128 << 31)..(1i128 << 31)).contains(x) {
                    Type::i32()
                } else if (1i128 << 31..1i128 << 32).contains(x) {
                    Type::Int(32, Signedness::Unsigned)
                } else if (-(1i128 << 63)..(1i128 << 63)).contains(x) {
                    Type::i64()
                } else {
                    Type::Int(64, Signedness::Unsigned)
                }
            }
            // `to_tensor_type` picks f32 for anything in f32's normal range (including 0
            // and inf) and f64 otherwise.
            Val::Float(x) => {
                let a = x.abs();
                let min_f32 = 2f64.powi(-126);
                let max_f32 = (2.0 - 2f64.powi(-23)) * 2f64.powi(127);
                if a.is_infinite() || a == 0.0 || x.is_nan() || (min_f32..=max_f32).contains(&a) {
                    Type::f32()
                } else {
                    Type::Float(FloatKind::F64)
                }
            }
            other => {
                return Err(Error::new(
                    format!("cannot convert a {} to a tensor", other.kind_name()),
                    0,
                    0,
                ))
            }
        })
    }

    /// `scalar_constant` (`semantic.py:601`): an `arith.constant` of the given dtype.
    pub fn scalar_constant(&mut self, v: &Val, ty: Type) -> Result<ValueId> {
        let attr = if ty.is_floating() {
            let f = v.as_f64().ok_or_else(|| {
                Error::new(
                    format!("cannot make a {} constant from a {}", ty, v.kind_name()),
                    0,
                    0,
                )
            })?;
            Attr::Float(F64Bits::rounded(f, &ty), ty.clone())
        } else {
            let i = v.as_int().ok_or_else(|| {
                Error::new(
                    format!("cannot make a {} constant from a {}", ty, v.kind_name()),
                    0,
                    0,
                )
            })?;
            Attr::Int(i, ty.clone())
        };
        Ok(self.emit1("arith.constant", &[], ty, &[("value", attr)]))
    }

    /// A splat constant: `arith.constant dense<V> : tensor<...>`.
    pub fn splat_constant(&mut self, v: &Val, ty: Type) -> Result<ValueId> {
        let elem = ty.scalar().clone();
        let inner = if elem.is_floating() {
            Attr::Float(
                F64Bits::rounded(v.as_f64().unwrap_or(0.0), &elem),
                elem.clone(),
            )
        } else {
            Attr::Int(v.as_int().unwrap_or(0), elem.clone())
        };
        Ok(self.emit1(
            "arith.constant",
            &[],
            ty.clone(),
            &[("value", Attr::DenseSplat(Box::new(inner), ty))],
        ))
    }

    /// `to_tensor` (`semantic.py:118`): make an IR value out of a literal, leaving an IR
    /// value alone.
    pub fn to_tensor(&mut self, v: &Val) -> Result<ValueId> {
        match v {
            Val::Ir(id) => Ok(*id),
            other => {
                let ty = self.literal_type(other)?;
                self.scalar_constant(other, ty)
            }
        }
    }

    // ===------------------------------------------------------------------===//
    //                             Type promotion
    // ===------------------------------------------------------------------===//

    /// `integer_promote_impl` (`semantic.py:50`): C's usual arithmetic conversions.
    pub fn integer_promote(&self, a: &Type, b: &Type) -> Result<Type> {
        let (ab, bb) = (
            a.int_bitwidth().unwrap_or(0),
            b.int_bitwidth().unwrap_or(0),
        );
        let (asn, bsn) = (a.signedness(), b.signedness());
        if asn == bsn {
            return Ok(if ab > bb { a.clone() } else { b.clone() });
        }
        if asn == Signedness::Unsigned {
            return Ok(if ab >= bb { a.clone() } else { b.clone() });
        }
        if bsn == Signedness::Unsigned {
            return Ok(if bb >= ab { b.clone() } else { a.clone() });
        }
        Ok(if ab > bb { a.clone() } else { b.clone() })
    }

    /// `computation_type_impl` (`semantic.py:68`).
    ///
    /// # THE ONE DELIBERATE DIVERGENCE, gated on the target.
    ///
    /// Rules 0, 3 and 4 promote `/` and `%` on f16/bf16 to f32. Triton's stated reason is
    /// that they "do not exist natively in PTX for fp16" -- a PTX fact, FALSE on this
    /// device, where divide is the templated `REALDIV`. When
    /// `target.div_promotes_narrow_floats` is false the promotion does not happen and the
    /// narrow type is kept. See [`crate::target::Target::div_promotes_narrow_floats`] for
    /// the full measurement.
    pub fn computation_type(
        &self,
        a: &Type,
        a_is_scalar: bool,
        b: &Type,
        b_is_scalar: bool,
        div_or_mod: bool,
    ) -> Result<Type> {
        // The narrow-float promotion, applied only where the target says it should be.
        let narrow_div = |t: &Type| -> bool {
            div_or_mod
                && self.target.div_promotes_narrow_floats
                && matches!(
                    t.scalar(),
                    Type::Float(FloatKind::F16) | Type::Float(FloatKind::BF16)
                )
        };

        // 0) a scalar of lower-or-equal kind does not participate in the promotion.
        if a_is_scalar != b_is_scalar {
            let (scalar_ty, tensor_ty) = if a_is_scalar { (a, b) } else { (b, a) };
            if kind_rank(scalar_ty) <= kind_rank(tensor_ty) {
                if narrow_div(tensor_ty) {
                    return Ok(Type::f32());
                }
                return Ok(tensor_ty.scalar().clone());
            }
        }
        let (asc, bsc) = (a.scalar(), b.scalar());
        // 1) fp64 wins.
        if matches!(asc, Type::Float(FloatKind::F64)) || matches!(bsc, Type::Float(FloatKind::F64)) {
            return Ok(Type::Float(FloatKind::F64));
        }
        // 2) fp32 wins.
        if matches!(asc, Type::Float(FloatKind::F32)) || matches!(bsc, Type::Float(FloatKind::F32)) {
            return Ok(Type::f32());
        }
        // 3) fp16.
        let a16 = matches!(asc, Type::Float(FloatKind::F16));
        let b16 = matches!(bsc, Type::Float(FloatKind::F16));
        if a16 || b16 {
            if div_or_mod && self.target.div_promotes_narrow_floats {
                return Ok(Type::f32());
            }
            return Ok(Type::f16());
        }
        // 4) bf16.
        let abf = matches!(asc, Type::Float(FloatKind::BF16));
        let bbf = matches!(bsc, Type::Float(FloatKind::BF16));
        if abf && bbf {
            if div_or_mod && self.target.div_promotes_narrow_floats {
                return Ok(Type::f32());
            }
            return Ok(Type::Float(FloatKind::BF16));
        }
        if abf || bbf {
            return Ok(Type::f32());
        }
        // 5) fp8 pairs.
        let is_fp8 = |t: &Type| {
            matches!(
                t,
                Type::Float(FloatKind::F8E4M3FN) | Type::Float(FloatKind::F8E5M2)
            )
        };
        if is_fp8(asc) && is_fp8(bsc) {
            return Ok(if asc == bsc { asc.clone() } else { Type::f16() });
        }
        if !asc.is_int() || !bsc.is_int() {
            return Err(Error::new(
                format!("unexpected types {asc} and {bsc} in an arithmetic operation"),
                0,
                0,
            ));
        }
        // 6) integer promotion.
        if div_or_mod && asc.signedness() != bsc.signedness() {
            return Err(Error::new(
                format!(
                    "cannot use /, // or % with {asc} and {bsc} because they have \
                     different signedness; cast them to the same signedness"
                ),
                0,
                0,
            ));
        }
        self.integer_promote(asc, bsc)
    }

    // ===------------------------------------------------------------------===//
    //                                 Casts
    // ===------------------------------------------------------------------===//

    /// `cast` (`semantic.py:811`). Returns the input unchanged when the scalar types
    /// already agree, which is why an `x.to(same_dtype)` emits nothing.
    pub fn cast(&mut self, v: ValueId, dst: &Type) -> Result<ValueId> {
        let src_ty = self.ty(v);
        let src = src_ty.scalar().clone();
        let dst_sca = dst.scalar().clone();
        if src == dst_sca {
            return Ok(v);
        }
        let dst_ty = src_ty.with_element_ty(dst_sca.clone());

        let sw = match &src {
            Type::Float(k) => k.bitwidth(),
            Type::Int(b, _) => *b,
            _ => 0,
        };
        let dw = match &dst_sca {
            Type::Float(k) => k.bitwidth(),
            Type::Int(b, _) => *b,
            _ => 0,
        };

        // bf16 <=> (not fp32) routes through fp32 (`semantic.py:847`).
        let is_narrow_fp = |t: &Type| {
            matches!(
                t,
                Type::Float(FloatKind::F16) | Type::Float(FloatKind::BF16)
            )
        };
        if is_narrow_fp(&src) && dst_sca.is_floating() && !matches!(dst_sca, Type::Float(FloatKind::F32))
        {
            let mid = self.cast(v, &Type::f32())?;
            return self.cast(mid, &dst_sca);
        }

        if src.is_floating() && dst_sca.is_floating() {
            if sw > dw {
                return Ok(self.emit1("arith.truncf", &[v], dst_ty, &[]));
            }
            if sw < dw {
                return Ok(self.emit1("arith.extf", &[v], dst_ty, &[]));
            }
        }
        if src.is_int() && dst_sca.is_int() {
            // `create_int_cast` picks extsi/extui/trunci by width and signedness; when the
            // widths match but the signedness does not, MLIR emits a bitcast.
            let sign_extend = src.signedness() == Signedness::Signed && sw != 1;
            let name = if dw > sw {
                if sign_extend {
                    "arith.extsi"
                } else {
                    "arith.extui"
                }
            } else if dw < sw {
                "arith.trunci"
            } else {
                "arith.bitcast"
            };
            return Ok(self.emit1(name, &[v], dst_ty, &[]));
        }
        if src.is_floating() && dst_sca.is_int() {
            let name = if dst_sca.signedness() == Signedness::Unsigned {
                "arith.fptoui"
            } else {
                "arith.fptosi"
            };
            return Ok(self.emit1(name, &[v], dst_ty, &[]));
        }
        if src.is_int() && dst_sca.is_floating() {
            let name = if src.signedness() == Signedness::Unsigned || sw == 1 {
                "arith.uitofp"
            } else {
                "arith.sitofp"
            };
            return Ok(self.emit1(name, &[v], dst_ty, &[]));
        }
        Err(Error::new(
            format!("cannot cast {src} to {dst_sca}"),
            0,
            0,
        ))
    }

    /// `create_int_cast` used directly (not through `cast`), which is what `visit_For`
    /// does for the loop bounds -- and why the goldens contain
    /// `arith.bitcast %x : i32 to i32`.
    pub fn int_cast(&mut self, v: ValueId, dst: &Type, sign_extend: bool) -> ValueId {
        let src_ty = self.ty(v);
        let sw = src_ty.int_bitwidth().unwrap_or(0);
        let dw = dst.int_bitwidth().unwrap_or(0);
        let name = if dw > sw {
            if sign_extend {
                "arith.extsi"
            } else {
                "arith.extui"
            }
        } else if dw < sw {
            "arith.trunci"
        } else {
            "arith.bitcast"
        };
        let dst_ty = src_ty.with_element_ty(dst.scalar().clone());
        self.emit1(name, &[v], dst_ty, &[])
    }

    // ===------------------------------------------------------------------===//
    //                            Broadcasting
    // ===------------------------------------------------------------------===//

    /// `broadcast_impl_value`: make two values the same shape, inserting `tt.splat` for a
    /// scalar and `tt.broadcast` for a size-1 axis.
    pub fn broadcast_pair(&mut self, a: ValueId, b: ValueId) -> Result<(ValueId, ValueId)> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        match (ta.is_tensor(), tb.is_tensor()) {
            (false, false) => Ok((a, b)),
            // Both of these go through `splat`, NOT a bare `tt.splat`, so MLIR's
            // constant-operand fold applies. Emitting `tt.splat` directly here left the
            // overflow check's i64 bounds as `tt.splat` where the oracle has
            // `arith.constant dense<2147483647> : tensor<64xi64>`.
            (true, false) => {
                let nb = self.splat(b, ta.shape())?;
                Ok((a, nb))
            }
            (false, true) => {
                let na = self.splat(a, tb.shape())?;
                Ok((na, b))
            }
            (true, true) => {
                let (sa, sb) = (ta.shape().to_vec(), tb.shape().to_vec());
                if sa == sb {
                    return Ok((a, b));
                }
                if sa.len() != sb.len() {
                    return Err(Error::new(
                        format!(
                            "cannot broadcast rank-{} against rank-{} (shapes [{}] and [{}]); \
                             Triton requires equal ranks -- insert an explicit `[:, None]`",
                            sa.len(),
                            sb.len(),
                            join(&sa),
                            join(&sb)
                        ),
                        0,
                        0,
                    ));
                }
                let mut want = Vec::with_capacity(sa.len());
                for i in 0..sa.len() {
                    if sa[i] == sb[i] {
                        want.push(sa[i]);
                    } else if sa[i] == 1 {
                        want.push(sb[i]);
                    } else if sb[i] == 1 {
                        want.push(sa[i]);
                    } else {
                        return Err(Error::new(
                            format!(
                                "cannot broadcast shapes [{}] and [{}]: axis {i} is \
                                 {} against {}",
                                join(&sa),
                                join(&sb),
                                sa[i],
                                sb[i]
                            ),
                            0,
                            0,
                        ));
                    }
                }
                let na = if sa == want {
                    a
                } else {
                    let t = Type::Tensor(want.clone(), std::rc::Rc::new(ta.scalar().clone()));
                    self.emit1("tt.broadcast", &[a], t, &[])
                };
                let nb = if sb == want {
                    b
                } else {
                    let t = Type::Tensor(want.clone(), std::rc::Rc::new(tb.scalar().clone()));
                    self.emit1("tt.broadcast", &[b], t, &[])
                };
                Ok((na, nb))
            }
        }
    }

    /// `expand_dims`: `tt.expand_dims` with an axis attribute.
    pub fn expand_dims(&mut self, v: ValueId, axis: i64) -> Result<ValueId> {
        let ty = self.ty(v);
        let mut shape = ty.shape().to_vec();
        let ax = if axis < 0 {
            (shape.len() as i64 + 1 + axis) as usize
        } else {
            axis as usize
        };
        if ax > shape.len() {
            return Err(Error::new(
                format!("expand_dims axis {axis} is out of range for rank {}", shape.len()),
                0,
                0,
            ));
        }
        shape.insert(ax, 1);
        let out = Type::Tensor(shape, std::rc::Rc::new(ty.scalar().clone()));
        Ok(self.emit1(
            "tt.expand_dims",
            &[v],
            out,
            &[("axis", Attr::Int(ax as i128, Type::i32()))],
        ))
    }

    /// `broadcast_impl_shape`: grow a value to an explicit shape.
    pub fn broadcast_to(&mut self, v: ValueId, shape: &[i64]) -> Result<ValueId> {
        let ty = self.ty(v);
        if !ty.is_tensor() {
            return self.splat(v, shape);
        }
        if ty.shape() == shape {
            return Ok(v);
        }
        let out = Type::Tensor(shape.to_vec(), std::rc::Rc::new(ty.scalar().clone()));
        Ok(self.emit1("tt.broadcast", &[v], out, &[]))
    }

    // ===------------------------------------------------------------------===//
    //                          Binary arithmetic
    // ===------------------------------------------------------------------===//

    /// `binary_op_type_checking_impl` (`semantic.py:170`).
    ///
    /// NOTE THE DOUBLE MATERIALIZATION, which is faithful and load-bearing for the diff: a
    /// scalar operand is turned into a constant by `to_tensor` for the type inspection, and
    /// then `scalar_constant` builds a SECOND constant of the promoted type. The first is
    /// left dead. Reproduced because the oracle contains it.
    #[allow(clippy::too_many_arguments)]
    pub fn binary_op_type_checking(
        &mut self,
        lhs: &Val,
        rhs: &Val,
        allow_lhs_ptr: bool,
        allow_rhs_ptr: bool,
        arithmetic_check: bool,
        div_or_mod: bool,
    ) -> Result<(ValueId, ValueId)> {
        let lhs_is_scalar = lhs.is_number();
        let rhs_is_scalar = rhs.is_number();
        let mut l = self.to_tensor(lhs)?;
        let mut r = self.to_tensor(rhs)?;
        let lt = self.ty(l);
        let rt = self.ty(r);
        self.check_ptr_type(&lt, &rt, allow_lhs_ptr)?;
        self.check_ptr_type(&rt, &lt, allow_rhs_ptr)?;
        if arithmetic_check && !lt.is_ptr() && !rt.is_ptr() {
            let want = self.computation_type(
                lt.scalar(),
                lhs_is_scalar,
                rt.scalar(),
                rhs_is_scalar,
                div_or_mod,
            )?;
            l = if lhs_is_scalar {
                self.scalar_constant(lhs, want.clone())?
            } else {
                self.cast(l, &want)?
            };
            r = if rhs_is_scalar {
                self.scalar_constant(rhs, want.clone())?
            } else {
                self.cast(r, &want)?
            };
        }
        self.broadcast_pair(l, r)
    }

    fn check_ptr_type(&self, a: &Type, b: &Type, allow: bool) -> Result<()> {
        if a.is_ptr() {
            if !allow {
                return Err(Error::new(
                    format!("invalid operand types {a} and {b}"),
                    0,
                    0,
                ));
            }
            if b.is_ptr() && a != b {
                return Err(Error::new(
                    format!("cannot combine pointers of different types: {a} and {b}"),
                    0,
                    0,
                ));
            }
            if b.is_floating() {
                return Err(Error::new(
                    format!("cannot combine a pointer {a} with a float {b}"),
                    0,
                    0,
                ));
            }
        }
        Ok(())
    }

    /// `binary_op_sanitize_overflow_impl` (`semantic.py:212`).
    ///
    /// Emits SIX ops before the operation it guards: two `extsi` to i64, the widened
    /// operation, the two bound constants, two `cmpi`, one `andi`. The `device_assert` that
    /// consumes the result emits nothing unless `options.debug` is on
    /// (`semantic.py:1810` returns early), so the `andi` is dead -- which is exactly what
    /// the oracle shows.
    fn sanitize_overflow(&mut self, l: ValueId, r: ValueId, kind: OverflowOp) -> Result<()> {
        if !self.target.sanitize_overflow {
            return Ok(());
        }
        let lt = self.ty(l);
        let bits = match lt.int_bitwidth() {
            Some(b) => b,
            None => return Ok(()),
        };
        if bits >= 64 {
            return Ok(());
        }
        let i64t = Type::i64();
        let wl = self.cast(l, &i64t)?;
        let wr = self.cast(r, &i64t)?;
        let wide_ty = self.ty(wl);
        let name = match kind {
            OverflowOp::Add => "arith.addi",
            OverflowOp::Sub => "arith.subi",
            OverflowOp::Mul => "arith.muli",
        };
        let ret = self.emit1(name, &[wl, wr], wide_ty, &[]);
        let signed = lt.signedness() != Signedness::Unsigned;
        let (max_v, min_v) = if signed {
            (
                (1i128 << (bits - 1)) - 1,
                -(1i128 << (bits - 1)),
            )
        } else {
            ((1i128 << bits) - 1, 0)
        };
        let max_c = self.scalar_constant(&Val::Int(max_v), i64t.clone())?;
        let min_c = self.scalar_constant(&Val::Int(min_v), i64t.clone())?;
        let le = self.compare_ir(ret, max_c, CmpOp::Le)?;
        let ge = self.compare_ir(ret, min_c, CmpOp::Ge)?;
        let i1 = self.ty(le);
        let _ = self.emit1("arith.andi", &[le, ge], i1, &[]);
        // device_assert emits nothing: options.debug is off.
        Ok(())
    }

    /// `add` (`semantic.py:229`).
    pub fn add(&mut self, lhs: &Val, rhs: &Val, sanitize: bool) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, true, true, true, false)?;
        let lt = self.ty(l);
        let rt = self.ty(r);
        if lt.is_ptr() && rt.is_ptr() {
            return Err(Error::new("cannot add pointers together", 0, 0));
        }
        // `offset + ptr` is normalized to `ptr + offset`.
        let (l, r, lt) = if rt.is_ptr() && !lt.is_ptr() {
            (r, l, rt)
        } else {
            (l, r, lt)
        };
        if lt.is_ptr() {
            return Ok(self.emit1("tt.addptr", &[l, r], lt, &[]));
        }
        if lt.is_floating() {
            return Ok(self.emit1("arith.addf", &[l, r], lt, &[]));
        }
        if lt.is_int() {
            if sanitize {
                self.sanitize_overflow(l, r, OverflowOp::Add)?;
            }
            return Ok(self.emit1("arith.addi", &[l, r], lt, &[]));
        }
        Err(Error::new(format!("unexpected type {lt} for +"), 0, 0))
    }

    /// `sub` (`semantic.py:261`).
    pub fn sub(&mut self, lhs: &Val, rhs: &Val, sanitize: bool) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, true, false, true, false)?;
        let lt = self.ty(l);
        if lt.is_ptr() {
            let neg = self.minus_ir(r)?;
            return self.add(&Val::Ir(l), &Val::Ir(neg), false);
        }
        if lt.is_floating() {
            return Ok(self.emit1("arith.subf", &[l, r], lt, &[]));
        }
        if lt.is_int() {
            if sanitize {
                self.sanitize_overflow(l, r, OverflowOp::Sub)?;
            }
            return Ok(self.emit1("arith.subi", &[l, r], lt, &[]));
        }
        Err(Error::new(format!("unexpected type {lt} for -"), 0, 0))
    }

    /// `mul` (`semantic.py:278`).
    pub fn mul(&mut self, lhs: &Val, rhs: &Val, sanitize: bool) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, false)?;
        let lt = self.ty(l);
        if lt.is_floating() {
            return Ok(self.emit1("arith.mulf", &[l, r], lt, &[]));
        }
        if lt.is_int() {
            if sanitize {
                self.sanitize_overflow(l, r, OverflowOp::Mul)?;
            }
            return Ok(self.emit1("arith.muli", &[l, r], lt, &[]));
        }
        Err(Error::new(format!("unexpected type {lt} for *"), 0, 0))
    }

    /// `truediv` (`semantic.py:293`). `arithmetic_check` ON, so the narrow-float promotion
    /// policy applies -- see [`Self::computation_type`].
    pub fn truediv(&mut self, lhs: &Val, rhs: &Val) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, true)?;
        let lt = self.ty(l);
        let rt = self.ty(r);
        // After type checking both sides already share the computation type, so the only
        // remaining cases are float/float and int/int.
        if lt.is_floating() && rt.is_floating() {
            return Ok(self.emit1("arith.divf", &[l, r], lt, &[]));
        }
        if lt.is_int() && rt.is_int() {
            let name = if lt.signedness() == Signedness::Unsigned {
                "arith.divui"
            } else {
                "arith.divsi"
            };
            return Ok(self.emit1(name, &[l, r], lt, &[]));
        }
        Err(Error::new(
            format!("unexpected types {lt} and {rt} for /"),
            0,
            0,
        ))
    }

    /// `fdiv`: the same divide with `arithmetic_check` OFF, which is what makes
    /// `tl.fdiv(f16, f16) -> f16` while `f16 / f16 -> f32` upstream.
    pub fn fdiv(&mut self, lhs: &Val, rhs: &Val) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, false, false)?;
        let lt = self.ty(l);
        if !lt.is_floating() {
            return Err(Error::new(
                format!("tl.fdiv requires floating operands, got {lt}"),
                0,
                0,
            ));
        }
        Ok(self.emit1("arith.divf", &[l, r], lt, &[]))
    }

    /// `floordiv` (`semantic.py`): integer only.
    pub fn floordiv(&mut self, lhs: &Val, rhs: &Val) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, true)?;
        let lt = self.ty(l);
        if lt.is_int() {
            let name = if lt.signedness() == Signedness::Unsigned {
                "arith.divui"
            } else {
                "arith.divsi"
            };
            return Ok(self.emit1(name, &[l, r], lt, &[]));
        }
        Err(Error::new(
            format!("// requires integer operands, got {lt}"),
            0,
            0,
        ))
    }

    /// `mod` (`semantic.py`).
    pub fn modulo(&mut self, lhs: &Val, rhs: &Val) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, true)?;
        let lt = self.ty(l);
        if lt.is_floating() {
            return Ok(self.emit1("arith.remf", &[l, r], lt, &[]));
        }
        if lt.is_int() {
            let name = if lt.signedness() == Signedness::Unsigned {
                "arith.remui"
            } else {
                "arith.remsi"
            };
            return Ok(self.emit1(name, &[l, r], lt, &[]));
        }
        Err(Error::new(format!("unexpected type {lt} for %"), 0, 0))
    }

    /// Bitwise and / or / xor and the shifts.
    pub fn bitwise(&mut self, lhs: &Val, rhs: &Val, which: Bitwise) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, false)?;
        let lt = self.ty(l);
        let name = match which {
            Bitwise::And => "arith.andi",
            Bitwise::Or => "arith.ori",
            Bitwise::Xor => "arith.xori",
            Bitwise::Shl => "arith.shli",
            Bitwise::Shr => {
                if lt.signedness() == Signedness::Unsigned {
                    "arith.shrui"
                } else {
                    "arith.shrsi"
                }
            }
        };
        Ok(self.emit1(name, &[l, r], lt, &[]))
    }

    /// `minus`: unary negation.
    pub fn minus_ir(&mut self, v: ValueId) -> Result<ValueId> {
        let ty = self.ty(v);
        if ty.is_floating() {
            return Ok(self.emit1("arith.negf", &[v], ty, &[]));
        }
        if ty.is_int() {
            let zero = self.scalar_constant(&Val::Int(0), ty.scalar().clone())?;
            let (z, v2) = self.broadcast_pair(zero, v)?;
            return Ok(self.emit1("arith.subi", &[z, v2], ty, &[]));
        }
        Err(Error::new(
            format!("unary minus is not defined on {ty}"),
            0,
            0,
        ))
    }

    /// Comparisons on two already-typed IR values.
    pub fn compare_ir(&mut self, l: ValueId, r: ValueId, op: CmpOp) -> Result<ValueId> {
        self.compare(&Val::Ir(l), &Val::Ir(r), op)
    }

    /// `less_than` / `less_equal` / ... (`semantic.py`).
    pub fn compare(&mut self, lhs: &Val, rhs: &Val, op: CmpOp) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, false)?;
        let lt = self.ty(l);
        let res_ty = lt.with_element_ty(Type::i1());
        if lt.is_floating() {
            let pred = match op {
                CmpOp::Eq => "oeq",
                CmpOp::Ne => "une",
                CmpOp::Lt => "olt",
                CmpOp::Le => "ole",
                CmpOp::Gt => "ogt",
                CmpOp::Ge => "oge",
            };
            return Ok(self.emit1(
                "arith.cmpf",
                &[l, r],
                res_ty,
                &[("predicate", Attr::Pred(pred))],
            ));
        }
        if lt.is_int() {
            let unsigned = lt.signedness() == Signedness::Unsigned;
            let pred = match (op, unsigned) {
                (CmpOp::Eq, _) => "eq",
                (CmpOp::Ne, _) => "ne",
                (CmpOp::Lt, false) => "slt",
                (CmpOp::Lt, true) => "ult",
                (CmpOp::Le, false) => "sle",
                (CmpOp::Le, true) => "ule",
                (CmpOp::Gt, false) => "sgt",
                (CmpOp::Gt, true) => "ugt",
                (CmpOp::Ge, false) => "sge",
                (CmpOp::Ge, true) => "uge",
            };
            return Ok(self.emit1(
                "arith.cmpi",
                &[l, r],
                res_ty,
                &[("predicate", Attr::Pred(pred))],
            ));
        }
        Err(Error::new(
            format!("comparison is not defined on {lt}"),
            0,
            0,
        ))
    }

    // ===------------------------------------------------------------------===//
    //                         The tl.* surface
    // ===------------------------------------------------------------------===//

    /// `tl.program_id` -> `tt.get_program_id`.
    pub fn program_id(&mut self, axis: i128) -> Result<ValueId> {
        let ax: &'static str = match axis {
            0 => "x",
            1 => "y",
            2 => "z",
            other => {
                return Err(Error::new(
                    format!("tl.program_id axis must be 0, 1 or 2, got {other}"),
                    0,
                    0,
                ))
            }
        };
        Ok(self.emit1(
            "tt.get_program_id",
            &[],
            Type::i32(),
            &[("axis", Attr::Axis(ax))],
        ))
    }

    /// `tl.num_programs` -> `tt.get_num_programs`.
    pub fn num_programs(&mut self, axis: i128) -> Result<ValueId> {
        let ax: &'static str = match axis {
            0 => "x",
            1 => "y",
            2 => "z",
            other => {
                return Err(Error::new(
                    format!("tl.num_programs axis must be 0, 1 or 2, got {other}"),
                    0,
                    0,
                ))
            }
        };
        Ok(self.emit1(
            "tt.get_num_programs",
            &[],
            Type::i32(),
            &[("axis", Attr::Axis(ax))],
        ))
    }

    /// `tl.arange` -> `tt.make_range`.
    pub fn arange(&mut self, start: i128, end: i128) -> Result<ValueId> {
        if end <= start {
            return Err(Error::new(
                format!("tl.arange requires end > start, got start={start} end={end}"),
                0,
                0,
            ));
        }
        let n = end - start;
        if n <= 0 || (n & (n - 1)) != 0 {
            return Err(Error::new(
                format!(
                    "tl.arange requires a power-of-two length, got {n} (start={start}, end={end})"
                ),
                0,
                0,
            ));
        }
        let ty = Type::tensor(&[n as i64], Type::i32());
        Ok(self.emit1(
            "tt.make_range",
            &[],
            ty,
            &[
                ("start", Attr::Int(start, Type::i32())),
                ("end", Attr::Int(end, Type::i32())),
            ],
        ))
    }

    /// `tt.make_tensor_descriptor`. Returns the COMPOSITE descriptor value, since the shape
    /// and stride values are part of it -- see [`Val::Desc`].
    pub fn make_tensor_descriptor(
        &mut self,
        base: ValueId,
        shape: &[ValueId],
        strides: &[ValueId],
        block_shape: &[i64],
    ) -> Result<Val> {
        let base_ty = self.ty(base);
        let elem = match &base_ty {
            Type::Ptr(e, _) => (**e).clone(),
            other => {
                return Err(Error::new(
                    format!(
                        "tl.make_tensor_descriptor needs a pointer as its first argument, \
                         got {other}"
                    ),
                    0,
                    0,
                ))
            }
        };
        let mut operands = vec![base];
        operands.extend_from_slice(shape);
        operands.extend_from_slice(strides);
        let ty = Type::desc(block_shape, elem);
        let handle = self.emit1("tt.make_tensor_descriptor", &operands, ty, &[]);
        Ok(Val::Desc {
            handle,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        })
    }

    /// `desc.load(offsets)` -> `tt.descriptor_load`.
    pub fn descriptor_load(&mut self, desc: ValueId, offsets: &[ValueId]) -> Result<ValueId> {
        let dty = self.ty(desc);
        let (shape, elem) = match &dty {
            Type::TensorDesc(s, e) => (s.clone(), (**e).clone()),
            other => {
                return Err(Error::new(
                    format!("`.load` needs a tensor descriptor, got {other}"),
                    0,
                    0,
                ))
            }
        };
        if offsets.len() != shape.len() {
            return Err(Error::new(
                format!(
                    "`.load` needs one offset per descriptor axis: descriptor has rank {} \
                     but {} offset(s) were given",
                    shape.len(),
                    offsets.len()
                ),
                0,
                0,
            ));
        }
        let mut operands = vec![desc];
        operands.extend_from_slice(offsets);
        let ty = Type::Tensor(shape, std::rc::Rc::new(elem));
        Ok(self.emit1("tt.descriptor_load", &operands, ty, &[]))
    }

    /// `desc.gather(x_offsets, y_offset)` -> `tt.descriptor_gather` (`semantic.py:1143`).
    ///
    /// # THE ONE INDIRECT ADDRESS in this front end, and every check here is Triton's own.
    ///
    /// The row index is a LOADED VALUE, not an affine function of `program_id`, so this is a
    /// distinct op rather than a load with a cleverer offset. `semantic.py`'s
    /// `descriptor_gather` guards it with five asserts and each one is reproduced below,
    /// because an assert in the oracle is a REFUSAL: a kernel that violates one does not
    /// compile there, so compiling it here would make this front end more permissive than
    /// the thing it is a port of.
    ///
    ///   * the descriptor is 2D and `block_shape[0] == 1` -- the result's row count comes
    ///     from the index vector, so a multi-row block has no meaning;
    ///   * `x_offsets` is 1D;
    ///   * its dtype is i16 or i32 (not i64, not f16);
    ///   * it has at least 8 rows;
    ///   * `block_shape[1] >= 32 / bitwidth * 8` columns, which is 16 for f16 and 8 for f32.
    ///
    /// The result is `tensor<x_offsets.shape[0] x block_shape[1] x elem>`.
    pub fn descriptor_gather(
        &mut self,
        desc: ValueId,
        x_offsets: ValueId,
        y_offset: ValueId,
    ) -> Result<ValueId> {
        let dty = self.ty(desc);
        let (block, elem) = match &dty {
            Type::TensorDesc(s, e) => (s.clone(), (**e).clone()),
            other => {
                return Err(Error::new(
                    format!("`.gather` needs a tensor descriptor, got {other}"),
                    0,
                    0,
                ))
            }
        };
        if block.len() != 2 {
            return Err(Error::new(
                format!(
                    "`.gather` needs a 2D descriptor, but this one has rank {} (block [{}])",
                    block.len(),
                    join(&block)
                ),
                0,
                0,
            ));
        }
        if block[0] != 1 {
            return Err(Error::new(
                format!(
                    "`.gather`'s descriptor block must have EXACTLY ONE ROW, but this one is \
                     [{}]. The gathered row count comes from the index vector, so a \
                     multi-row block has no meaning -- make the descriptor's block \
                     [1, cols].",
                    join(&block)
                ),
                0,
                0,
            ));
        }
        let xty = self.ty(x_offsets).clone();
        let xshape = xty.shape().to_vec();
        if xshape.len() != 1 {
            return Err(Error::new(
                format!(
                    "`.gather`'s index vector must be 1D, got [{}]",
                    join(&xshape)
                ),
                0,
                0,
            ));
        }
        match xty.scalar() {
            Type::Int(16, _) | Type::Int(32, _) => {}
            other => {
                return Err(Error::new(
                    format!(
                        "`.gather`'s index vector must have dtype int16 or int32, got {other}"
                    ),
                    0,
                    0,
                ))
            }
        }
        if xshape[0] < 8 {
            return Err(Error::new(
                format!(
                    "`.gather` needs at least 8 indices, got {}. Triton asserts this \
                     (`descriptor gather must have at least 8 rows`).",
                    xshape[0]
                ),
                0,
                0,
            ));
        }
        // `min_cols = 32 // dtype.primitive_bitwidth * 8`, Triton's own expression -- integer
        // division FIRST, so f16 gives (32/16)*8 = 16 and f32 gives (32/32)*8 = 8. Writing it
        // as 256/bitwidth would agree on those two and disagree on f64.
        let bits = match &elem {
            Type::Float(k) => k.bitwidth(),
            Type::Int(b, _) => *b,
            other => {
                return Err(Error::new(
                    format!("`.gather` on a descriptor of {other} has no defined width"),
                    0,
                    0,
                ))
            }
        };
        let min_cols = i64::from(32 / bits * 8);
        if block[1] < min_cols {
            return Err(Error::new(
                format!(
                    "`.gather` of {elem} needs at least {min_cols} columns, but the \
                     descriptor block is [{}]",
                    join(&block)
                ),
                0,
                0,
            ));
        }
        let ty = Type::Tensor(vec![xshape[0], block[1]], std::rc::Rc::new(elem));
        Ok(self.emit1(
            "tt.descriptor_gather",
            &[desc, x_offsets, y_offset],
            ty,
            &[],
        ))
    }

    /// `desc.store(offsets, value)` -> `tt.descriptor_store`.
    pub fn descriptor_store(
        &mut self,
        desc: ValueId,
        offsets: &[ValueId],
        value: ValueId,
    ) -> Result<()> {
        let dty = self.ty(desc);
        let shape = match &dty {
            Type::TensorDesc(s, _) => s.clone(),
            other => {
                return Err(Error::new(
                    format!("`.store` needs a tensor descriptor, got {other}"),
                    0,
                    0,
                ))
            }
        };
        if offsets.len() != shape.len() {
            return Err(Error::new(
                format!(
                    "`.store` needs one offset per descriptor axis: descriptor has rank {} \
                     but {} offset(s) were given",
                    shape.len(),
                    offsets.len()
                ),
                0,
                0,
            ));
        }
        let vty = self.ty(value);
        if vty.shape() != shape.as_slice() {
            return Err(Error::new(
                format!(
                    "`.store` value shape [{}] does not match the descriptor block shape [{}]",
                    join(vty.shape()),
                    join(&shape)
                ),
                0,
                0,
            ));
        }
        // `descriptor_store` IMPLICITLY CASTS the value to the descriptor's element type
        // (`semantic.py:1088`). That is not a formality: `attention_flash.py`'s epilogue
        // divides in f32 (under Triton's own promotion rule) and stores through an f16
        // descriptor, so the cast is the `arith.truncf` the golden shows just before the store.
        let elem = match &dty {
            Type::TensorDesc(_, e) => (**e).clone(),
            _ => unreachable!("checked above"),
        };
        let value = self.cast(value, &elem)?;
        let mut operands = vec![desc];
        operands.extend_from_slice(offsets);
        operands.push(value);
        self.emit0("tt.descriptor_store", &operands, &[]);
        Ok(())
    }

    /// `tl.dot` -> `tt.dot` (`semantic.py:1429`).
    ///
    /// # It emits a DEAD constant first, and that is faithful.
    ///
    /// `semantic.dot` computes `_0` -- a zero of the result's scalar type -- unconditionally,
    /// and then uses it only when `acc is None` (`semantic.py:1496-1505`):
    ///
    /// ```text
    /// _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
    /// ...
    /// if acc is None:
    ///     acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
    /// ```
    ///
    /// Every `tl.dot` in the fixtures passes an accumulator, so every one leaves an unused
    /// `arith.constant 0.000000e+00 : f16` behind. Measured in `swiglu_mlp`'s golden: three
    /// `tl.dot` calls, three dead zero constants (`%g_48`, `%u_51`, `%acc_43`), each
    /// immediately before its `tt.dot`.
    pub fn dot(
        &mut self,
        a: ValueId,
        b: ValueId,
        acc: Option<ValueId>,
        out_dtype: Option<Type>,
    ) -> Result<ValueId> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        let (sa, sb) = (ta.shape(), tb.shape());
        if sa.len() != 2 || sb.len() != 2 {
            return Err(Error::new(
                format!(
                    "tl.dot needs two rank-2 operands, got ranks {} and {}",
                    sa.len(),
                    sb.len()
                ),
                0,
                0,
            ));
        }
        if sa[1] != sb[0] {
            return Err(Error::new(
                format!(
                    "tl.dot inner dimensions disagree: [{}] against [{}]",
                    join(sa),
                    join(sb)
                ),
                0,
                0,
            ));
        }
        if ta.scalar() != tb.scalar() {
            return Err(Error::new(
                format!(
                    "tl.dot operands must have the same dtype. Got {} and {}",
                    ta.scalar(),
                    tb.scalar()
                ),
                0,
                0,
            ));
        }
        let want = vec![sa[0], sb[1]];

        // `out_dtype = tl.float32 if acc is None else acc.type.element_ty`
        // (`semantic.py:1467`).
        let out_dtype = match (out_dtype, acc) {
            (Some(t), _) => t,
            (None, Some(id)) => self.ty(id).scalar().clone(),
            (None, None) => Type::f32(),
        };
        if !ta.scalar().is_floating() {
            return Err(Error::new(
                format!(
                    "tl.dot on {} operands is not lowered: the integer path narrows the result \
                     to i32 and builds its zero differently (`semantic.py:1482`), and no \
                     fixture uses it",
                    ta.scalar()
                ),
                0,
                0,
            ));
        }
        // `ret_scalar_ty` on the standard float path is `out_dtype`, except that an f32 or
        // bf16 input forces f32 and f64 forces f64 (`semantic.py:1489`).
        let ret_scalar_ty = if matches!(
            ta.scalar(),
            Type::Float(FloatKind::F32) | Type::Float(FloatKind::BF16)
        ) {
            Type::f32()
        } else if matches!(ta.scalar(), Type::Float(FloatKind::F64)) {
            Type::Float(FloatKind::F64)
        } else {
            out_dtype.clone()
        };
        let ret_ty = Type::Tensor(want.clone(), std::rc::Rc::new(ret_scalar_ty));

        // `_0`, built UNCONDITIONALLY and used only when there is no accumulator
        // (`semantic.py:1496`). With an accumulator it is left dead, which is why every
        // `tl.dot` in the goldens is preceded by an unused `arith.constant 0.0`.
        let zero_ty = if matches!(out_dtype.scalar(), Type::Float(FloatKind::F16)) {
            Type::f16()
        } else {
            Type::f32()
        };
        let zero = self.scalar_constant(&Val::Int(0), zero_ty)?;

        let acc_id = match acc {
            Some(id) => {
                let tacc = self.ty(id);
                if tacc.shape() != want.as_slice() {
                    return Err(Error::new(
                        format!(
                            "tl.dot accumulator shape [{}] does not match the product shape [{}]",
                            join(tacc.shape()),
                            join(&want)
                        ),
                        0,
                        0,
                    ));
                }
                id
            }
            // No accumulator: splat the zero to the result shape. `create_splat` folds a
            // constant operand, so this becomes a second `arith.constant dense<...>` rather
            // than a `tt.splat`.
            None => self.splat(zero, &want)?,
        };
        Ok(self.emit1("tt.dot", &[a, b, acc_id], ret_ty, &[]))
    }

    /// `tt.trans` with an explicit permutation.
    pub fn trans(&mut self, v: ValueId, order: &[i64]) -> Result<ValueId> {
        let ty = self.ty(v);
        let shape = ty.shape();
        if order.len() != shape.len() {
            return Err(Error::new(
                format!(
                    "tt.trans order has {} entries for a rank-{} value",
                    order.len(),
                    shape.len()
                ),
                0,
                0,
            ));
        }
        let new_shape: Vec<i64> = order.iter().map(|i| shape[*i as usize]).collect();
        let out = Type::Tensor(new_shape, std::rc::Rc::new(ty.scalar().clone()));
        let items: Vec<Attr> = order
            .iter()
            .map(|i| Attr::Int(*i as i128, Type::i32()))
            .collect();
        Ok(self.emit1("tt.trans", &[v], out, &[("order", Attr::Array(items))]))
    }

    /// A unary math op on a float tensor: `math.exp`, `math.exp2`, `math.rsqrt`.
    ///
    /// # THE WIDTH GATE IS THE ORACLE'S, and leaving it out was a real gap.
    ///
    /// `tl.exp`, `tl.math.exp2` and `tl.rsqrt` are each decorated
    /// `@_check_dtype(dtypes=["fp32", "fp64"])` in `python/triton/language/math.py`, so on an
    /// f16 tensor Triton REFUSES:
    ///
    /// ```text
    /// ValueError: Expected dtype ['fp32', 'fp64'] but got fp16
    /// ```
    ///
    /// That is why `swiglu_mlp.py` writes its sigmoid as a widen / exp / truncate island
    /// (its delta 5) and `rmsnorm.py` its reciprocal-sqrt the same way (its delta 1) --
    /// the island is FORCED, not chosen. This function used to accept ANY float, which made
    /// the port more permissive than its own oracle on a construct no fixture reaches from
    /// the wrong side. `tests/math_width.rs` is the control.
    ///
    /// The message deliberately echoes Triton's wording so a kernel author who hits it here
    /// and there recognises the same refusal.
    pub fn math_unary(&mut self, name: &str, v: ValueId) -> Result<ValueId> {
        let ty = self.ty(v);
        if !ty.is_floating() {
            return Err(Error::new(
                format!("{name} requires a floating operand, got {ty}"),
                0,
                0,
            ));
        }
        match ty.scalar() {
            Type::Float(FloatKind::F32) | Type::Float(FloatKind::F64) => {}
            other => {
                return Err(Error::new(
                    format!(
                        "{name}: Expected dtype ['fp32', 'fp64'] but got {}. Triton's \
                         `@_check_dtype` refuses this too -- widen with `.to(tl.float32)`, \
                         call it, and truncate straight back; that island is the shape \
                         LegalizeTypes collapses.",
                        match other {
                            Type::Float(FloatKind::F16) => "fp16",
                            Type::Float(FloatKind::BF16) => "bf16",
                            o => return Err(Error::new(format!("{name} on {o}"), 0, 0)),
                        }
                    ),
                    0,
                    0,
                ))
            }
        }
        Ok(self.emit1(name, &[v], ty, &[]))
    }

    /// `maximum` / `minimum` -> `arith.maxnumf` / `arith.minnumf` (or the integer forms).
    pub fn max_min(&mut self, lhs: &Val, rhs: &Val, want_max: bool) -> Result<ValueId> {
        let (l, r) = self.binary_op_type_checking(lhs, rhs, false, false, true, false)?;
        let lt = self.ty(l);
        let name = if lt.is_floating() {
            if want_max {
                "arith.maxnumf"
            } else {
                "arith.minnumf"
            }
        } else if lt.signedness() == Signedness::Unsigned {
            if want_max {
                "arith.maxui"
            } else {
                "arith.minui"
            }
        } else if want_max {
            "arith.maxsi"
        } else {
            "arith.minsi"
        };
        Ok(self.emit1(name, &[l, r], lt, &[]))
    }

    /// `tt.reduce` (`semantic.reduction`, `semantic.py:1666`) with its combiner REGION.
    ///
    /// The region's entry block takes TWO arguments of the input's element type -- the two
    /// partial values being combined -- and must end in `tt.reduce.return`. Golden:
    ///
    /// ```text
    /// %0 = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    /// ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
    ///   %2 = tt.call @triton.language.standard._elementwise_max__fp32_fp32(%a, %b)
    ///          : (f32, f32) -> f32
    ///   tt.reduce.return %2 : f32
    /// }) : (tensor<64x64xf32>) -> tensor<64xf32>
    /// ```
    ///
    /// The block arguments carry `loc(unknown)`: they are created by the builder, not by any
    /// AST node.
    pub fn reduce_with_region(
        &mut self,
        input: ValueId,
        axis: usize,
        blocks: Vec<Block>,
        loc: Loc,
    ) -> Result<ValueId> {
        let ret_ty = self.reduce_result_type(input, axis)?;
        let results = self.emit_with_region(
            "tt.reduce",
            &[input],
            std::slice::from_ref(&ret_ty),
            &[("axis", Attr::Int(axis as i128, Type::i32()))],
            blocks,
            loc,
        );
        Ok(results[0])
    }

    /// `multiple_of` (`semantic.py:1775`): NOT an op, an ATTRIBUTE on the value's DEFINING OP.
    ///
    /// ```text
    /// x.handle.set_attr("tt.divisibility", ir.make_attr(values, ...))
    /// return x
    /// ```
    ///
    /// So `start_n = tl.multiple_of(start_n, BLOCK_N)` adds nothing to the op stream and
    /// instead stamps the op that produced `start_n`. Measured in the attention golden:
    /// `%3 = ub.poison {tt.divisibility = dense<64> : tensor<1xi32>} : i32` -- the attribute
    /// lands on the loop's induction PLACEHOLDER, because that is what the loop variable is
    /// still bound to while the body is being visited.
    ///
    /// A value with no defining op in the current region (a block argument, say) simply keeps
    /// no attribute, which is what `set_attr` on a block argument amounts to.
    pub fn set_divisibility(&mut self, v: ValueId, values: &[i128]) {
        let attr = Attr::DenseSplat(
            Box::new(Attr::Int(values.first().copied().unwrap_or(1), Type::i32())),
            Type::tensor(&[values.len() as i64], Type::i32()),
        );
        // Search OUTWARD through the frame stack, not just the innermost. The value being
        // stamped is often defined in an enclosing region: `tl.multiple_of(start_n, BLOCK_N)`
        // sits inside the loop body but `start_n` is the induction PLACEHOLDER, emitted before
        // the loop. Searching only the current frame silently found nothing and dropped the
        // attribute.
        for f in self.frames.iter_mut().rev() {
            for b in f.blocks.iter_mut().rev() {
                for op in b.ops.iter_mut().rev() {
                    if op.results.contains(&v) {
                        op.attrs.insert("tt.divisibility".to_string(), attr);
                        return;
                    }
                }
            }
        }
    }

    /// Replace every use of `from` with `to` inside `blocks`.
    ///
    /// `visit_For` binds the loop variable to a `ub.poison` PLACEHOLDER while the body is
    /// visited and only afterwards does `iv_placeholder.replace_all_uses_with(iv)`
    /// (`code_generator.py:1324`). That ordering is observable -- it is why
    /// `tl.multiple_of` on the induction variable stamps the poison rather than the real
    /// induction variable -- so it is reproduced rather than short-circuited.
    pub fn replace_uses(blocks: &mut [Block], from: ValueId, to: ValueId) {
        fn walk(ops: &mut [Op], from: ValueId, to: ValueId) {
            for op in ops.iter_mut() {
                for o in op.operands.iter_mut() {
                    if *o == from {
                        *o = to;
                    }
                }
                for r in op.regions.iter_mut() {
                    for b in r.blocks.iter_mut() {
                        walk(&mut b.ops, from, to);
                    }
                }
            }
        }
        for b in blocks.iter_mut() {
            walk(&mut b.ops, from, to);
        }
    }

    /// `tt.reduce.return`, the combiner region's terminator.
    pub fn reduce_return(&mut self, v: ValueId, loc: Loc) {
        let mut op = Op::new("tt.reduce.return", loc);
        op.operands = vec![v];
        self.emit_raw(op);
    }

    /// The result type of a reduction over `axis`.
    pub fn reduce_result_type(&self, v: ValueId, axis: usize) -> Result<Type> {
        let ty = self.ty(v);
        let shape = ty.shape();
        if axis >= shape.len() {
            return Err(Error::new(
                format!("reduction axis {axis} is out of range for rank {}", shape.len()),
                0,
                0,
            ));
        }
        let mut out: Vec<i64> = shape.to_vec();
        out.remove(axis);
        Ok(if out.is_empty() {
            ty.scalar().clone()
        } else {
            Type::Tensor(out, std::rc::Rc::new(ty.scalar().clone()))
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum OverflowOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug)]
pub enum Bitwise {
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

fn join(xs: &[i64]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
