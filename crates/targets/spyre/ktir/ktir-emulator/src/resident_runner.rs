// Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
//
//! Drive a SINGLE-FUNCTION example program through the PRODUCTION resident /
//! segmented Metal executor ([`crate::segmented::execute_segmented`]).
//!
//! The fused / segmented path threads logical tensors by id (`t<id>`) and runs
//! each non-attention node as a `[1,1]` [`Segment::Fused`] whose body carries the
//! GPU offloads (NAX K-loop GEMM reconstruction, fused map windows, fused decode
//! attention). It was built for the OPTIMIZER's emitted IR, where every launch
//! scalar (tile-loop bound, block size, `K`) is already a literal `arith.constant`
//! and the only function arguments are tensor pointers.
//!
//! The hand-written `examples/*.mlir` programs instead carry their launch scalars
//! as `index`/`i32`/`f16` FUNCTION ARGUMENTS (`%K`, `%BLOCK_SIZE_M`, `%n_rows`,
//! `%scale`, ...). Fusion only keeps pointer args, so those scalar SSA values go
//! dangling ("undefined SSA value %n0_BLOCK_SIZE_M"). [`ResidentRunner`] closes
//! that gap with ONE principled, decomposition-agnostic rewrite: SPECIALIZE the
//! kernel to its concrete launch scalars — replace each scalar function argument
//! with an `arith.constant` of the bound value, prepended to the body, and drop it
//! from the signature. That produces exactly the literal-bound form the optimizer
//! already emits, so the unmodified `execute_segmented` then runs it end-to-end on
//! the Metal path. (It is NOT a model-specific hack: it specializes ANY function's
//! scalar args, recognizing no op pattern.)
//!
//! A program with NO scalar args needs no rewrite; one whose tensors live at
//! hardcoded HBM addresses (the RFC `hbm_seed` fixtures) cannot be expressed as a
//! marshalled-arg `ProgramSpec` and is reported as not-drivable (see the diff
//! CLI), not faked.

use crate::attrkey::AttrKey;
use crate::dtypes::DType;
use crate::interpreter::{Arg, Output};
use crate::ir::{Attr, IRFunction, Operation, Scalar, Ssa};
use crate::irtype::IrType;
use crate::opkind::OpKind;
use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use std::collections::{HashMap, HashSet};

/// One tensor argument of an example kernel: the function arg name (no `%`), the
/// raw little-endian bytes already in `dtype` layout, its shape, and whether the
/// kernel WRITES it (an output to read back) or only reads it (an input source).
pub struct TensorArg {
    /// The function ARGUMENT this tensor binds.
    pub arg: Ssa,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub is_output: bool,
}

/// One scalar argument: the function ARGUMENT it binds and its value. These are
/// specialized into `arith.constant` ops, NOT threaded as tensors.
pub struct ScalarArg {
    pub arg: Ssa,
    pub value: Scalar,
}

/// The plan that drives a single example function through `execute_segmented`:
/// the scalar-specialized module + the one-node `ProgramSpec` + the synthetic
/// tensor-id assignment, so the runner can marshal inputs / read back outputs by
/// the ORIGINAL arg name.
pub struct ResidentRunner {
    funcs: &'static [IRFunction<'static>],
    spec: ProgramSpec,
    /// function argument -> synthetic tensor id.
    arg_tid: HashMap<Ssa, u64>,
    /// the tensors, in argument order.
    tensors: Vec<TensorArg>,
}

/// A scalar value -> the `value` attribute an `arith.constant` carries. Integer /
/// index scalars become `Attr::Int` (the constant handler binds `Scalar::I64`,
/// which every index consumer — `scf.for` bounds, `arith.muli`, access-tile index
/// operands — coerces exactly like a literal `arith.constant : index`); floats
/// become `Attr::Float`; bool becomes `Attr::Bool`.
fn scalar_value_attr(s: &Scalar) -> Attr<'static> {
    match s {
        Scalar::I32(v) => Attr::Int(*v as i64),
        Scalar::I64(v) => Attr::Int(*v),
        Scalar::F32(v) => Attr::Float(*v as f64),
        Scalar::Bool(b) => Attr::Bool(*b),
    }
}

/// The result TYPE of a specialized scalar constant. Index-typed scalars (the
/// common loop-bound / block-size case) keep `index`.
fn scalar_result_type(s: &Scalar) -> IrType<'static> {
    match s {
        Scalar::I32(_) => IrType::Scalar(DType::I32),
        Scalar::I64(_) => IrType::Index,
        Scalar::F32(_) => IrType::Scalar(DType::F16),
        Scalar::Bool(_) => IrType::Scalar(DType::Bool),
    }
}

impl ResidentRunner {
    /// Build a runner for one example function. `module` is the parsed example
    /// program; `func` its function name. `tensors` are the pointer args (in order)
    /// and `scalars` the launch scalars to specialize. Tensor ids are assigned in
    /// the given tensor order (`t0, t1, ...`).
    pub fn new(
        original: &IRFunction<'static>,
        tensors: Vec<TensorArg>,
        scalars: Vec<ScalarArg>,
    ) -> Result<Self, String> {
        let a = crate::arena::Arena::global();

        // The resident executor is an ALL-F16 path: it sizes every stick and binds
        // every pointer arg with the single model dtype (F16), and `set_sources`
        // re-encodes any non-F16 host bytes THROUGH f16. So a program with a
        // non-F16 tensor arg — an i64/i32 GATHER-INDEX tensor (indexed_add's
        // `index`, paged_attention's `block_tables`) or an f32 data tensor
        // (vector_add_dynamic) — would have its indices/data silently rounded to
        // f16 and read the wrong rows. Rather than DIVERGE silently, report it as
        // not-drivable through the resident path (it stays correct on the default
        // mixed-dtype `execute_function` CPU path). Honest, not faked.
        if let Some(t) = tensors.iter().find(|t| t.dtype != DType::F16) {
            return Err(format!(
                "resident path is all-F16; tensor arg %{} is {:?} (non-F16 index/data \
                 tensors are not drivable here — the stick/base binding and set_sources \
                 both assume the model dtype). Runs correctly on the default CPU path.",
                t.arg.0, t.dtype
            ));
        }

        // SPECIALIZE: drop every scalar arg from the signature and prepend an
        // `arith.constant` binding `%<arg>` to its value, so the body's uses
        // resolve to a literal (the form fusion expects).
        let scalar_args: HashSet<Ssa> = scalars.iter().map(|s| s.arg).collect();
        let new_args: Vec<(Ssa, IrType<'static>)> = original
            .arguments
            .iter()
            .filter(|(v, _)| !scalar_args.contains(v))
            .copied()
            .collect();
        let mut operations: Vec<Operation<'static>> = Vec::new();
        for s in &scalars {
            let op = Operation::new(a, Some(s.arg), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                scalar_value_attr(&s.value),
            );
            operations.push(Operation {
                result_type: Some(scalar_result_type(&s.value)),
                ..op
            });
        }
        operations.extend(original.operations.iter().copied());

        let specialized = IRFunction {
            name: original.name,
            arguments: a.args(new_args),
            operations: a.ops(operations),
            grid: original.grid,
            return_type: original.return_type,
        };
        let funcs = a.funcs(vec![specialized]);

        // Assign a synthetic tensor id per pointer arg, in order, and build the
        // one-node ProgramSpec. Bindings use the REAL arg name; sources = inputs,
        // results = outputs.
        let mut arg_tid: HashMap<Ssa, u64> = HashMap::new();
        let mut bindings: Vec<Binding> = Vec::new();
        let mut sources: HashSet<u64> = HashSet::new();
        let mut results: HashSet<u64> = HashSet::new();
        for (i, t) in tensors.iter().enumerate() {
            let tid = i as u64;
            arg_tid.insert(t.arg, tid);
            bindings.push(Binding {
                arg: t.arg,
                tensor: tid,
                is_output: t.is_output,
            });
            if t.is_output {
                results.insert(tid);
            } else {
                sources.insert(tid);
            }
        }
        // A tensor that is BOTH read and written (e.g. reduce_generic's arg0,
        // sdpa-style in-place) is bound once with is_output per the caller; mark it
        // a source too so the buffer is seeded with the caller bytes (not zeroed).
        for t in &tensors {
            // Always seed the output buffer with the caller's bytes (zeros for a
            // pure output, real data for an in-place arg): make it a source so the
            // resident executor marshals it from `args` rather than zeroing it.
            if t.is_output
                && let Some(&tid) = arg_tid.get(&t.arg)
            {
                sources.insert(tid);
            }
        }

        let spec = ProgramSpec {
            nodes: vec![NodeSpec {
                func: original.name.to_string(),
                bindings,
            }],
            sources,
            results,
        };

        Ok(ResidentRunner {
            funcs,
            spec,
            arg_tid,
            tensors,
        })
    }

    /// Run the program end-to-end through the PRODUCTION resident executor
    /// ([`crate::resident::ResidentExecutor::new_native`]) at the kernel's native
    /// grid, reading back every output tensor keyed by its ORIGINAL arg name.
    ///
    /// Native-grid (not `[1,1]`-fused) because the hand-written example kernels are
    /// SPMD-tiled: each compute-tile writes a DISJOINT output slice keyed off
    /// `ktdp.get_compute_tile_id`, so only the native grid computes the WHOLE
    /// output. The resident HBM, weight cache, per-segment seg-plan (K-loop GEMM
    /// reconstruction where recognizable) and per-op Metal offloads
    /// (`metal_gemm_or_blas`, fused map windows) all ride along — this is the real
    /// resident Metal path, not the gate-forced `execute_function` shortcut. Each
    /// tensor arg (incl. zero-seeded outputs and in-place args) is marshaled as a
    /// SOURCE so its stick holds the caller bytes before the run.
    ///
    /// The executor is keyed by TENSOR ID throughout. This used to `format!("t{tid}")`
    /// on the way in and re-`format!` the same string on the way out to look the
    /// result back up — a spelling of a number both sides already held.
    pub fn run(&self) -> Result<HashMap<Ssa, Output>, String> {
        let args: Vec<(u64, Arg)> = self
            .tensors
            .iter()
            .map(|t| {
                (
                    self.arg_tid[&t.arg],
                    Arg::TensorBytes {
                        data: t.data.clone(),
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                    },
                )
            })
            .collect();
        let out_ids: Vec<u64> = self
            .tensors
            .iter()
            .filter(|t| t.is_output)
            .map(|t| self.arg_tid[&t.arg])
            .collect();

        let mut exec = crate::resident::ResidentExecutor::new_native(self.funcs, &self.spec)?;
        exec.set_sources(&args)?;
        let raw = exec.run(&out_ids)?;

        // Re-key from tensor id back to the output ARGUMENT it binds.
        let mut out: HashMap<Ssa, Output> = HashMap::new();
        for t in &self.tensors {
            if !t.is_output {
                continue;
            }
            if let Some(o) = raw.get(&self.arg_tid[&t.arg]) {
                out.insert(t.arg, o.clone());
            }
        }
        Ok(out)
    }

    /// The specialized functions + spec (for callers that want the resident
    /// executor directly).
    pub fn funcs(&self) -> &'static [IRFunction<'static>] {
        self.funcs
    }
    pub fn spec(&self) -> &ProgramSpec {
        &self.spec
    }
    pub fn func(&self) -> &'static str {
        self.funcs[0].name
    }
}
