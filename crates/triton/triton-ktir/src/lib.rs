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

//! BRIDGE TWO: **ttir value -> KTIR value**, 100% Rust.
//!
//! This crate PORTS THE BEHAVIOUR of the C++ passes in
//! `third_party/spyre/lib/Dialect/KTDP/Transforms/` and
//! `.../KTDF/Transforms/PlanCorelets.cpp`. It does not invoke them. The pass
//! ORDER is `make_ktir`'s (`third_party/spyre/backend/compiler.py`), and that
//! order is part of the spec:
//!
//! ```text
//!   DotToLinalg              tt.dot -> linalg.matmul (+ the P1.0 canonical gate)
//!   ConvertTTIRToKTDP        descriptors -> memory view / access tile / load-store,
//!                            then tt.trans folded into access_tile_order
//!   DistributeWork           tt.get_program_id -> get_compute_tile_id + grid loop
//!   LegalizeTypes            collapse Triton's f32 stability-widening island
//!   canonicalize             CSE + fold + DCE + constant hoist
//!   DecomposeDenseConstants  splat dense<> -> scalar constant + tensor.splat
//!   PlanCorelets             the two-corelet coordination plan
//! ```
//!
//! and then, because THE TRANSLATION IS NOT FINISHED THERE:
//!
//! ```text
//!   ToSchedulerKTIR          tt.* OUT -- func.func + a 1-D grid, !tt.ptr -> index,
//!                            tt.reduce -> linalg.generic with a reduction iterator,
//!                            tt.broadcast -> a yield-only linalg.generic, the corelet
//!                            plan dropped, the grid loop folded
//! ```
//!
//! The pipeline is Triton -> ttir -> **Triton-free KTIR** -> backend, and the backend
//! knows nothing about Triton. `make_ktir` alone leaves `tt.*` in the output, and the
//! tools downstream do not register the `tt` dialect at all -- so its KTIR cannot even
//! be PARSED by them. [`passes::to_ktir`] finishes the translation; that is
//! all it is, and it is not optional.
//!
//! # NO TEXT IN THE PIPELINE
//!
//! The compile path is value-in, value-out: [`ir::Module`] goes in carrying `tt.*`
//! ops and comes out carrying `ktdp.*`/`ktdf.*`/`linalg.*` ops. Nothing in
//! [`passes`] serializes or parses.
//!
//! [`text`] is a TEST INSTRUMENT and is the only place a string form exists:
//!
//! * [`text::parse`] reads MLIR generic-ish text so a golden `.mlir` file can be
//!   turned into an input VALUE in a test fixture. It is how this crate is
//!   exercised before bridge one (Python -> ttir) is finished.
//! * [`text::print`] writes KTIR text so the result can be diffed, field by
//!   field, against what the C++ toolchain emits.
//!
//! Neither is reachable from [`passes`]. If you find yourself wanting to parse in
//! a pass, the thing you want is a field on [`ir::Op`].
//!
//! # THE SWAP POINT IS [`ir`], AND ONLY [`ir`]
//!
//! There is an open decision about whether these types become scratchy's vendored
//! `ktir-core` (13 files, dependency-free, Apache-2.0, and its emulator then
//! becomes usable as an oracle) or stay ours. **[`ir`] is the one module that
//! decision touches.** Everything else -- every pass, the printer, the parser --
//! goes through [`ir`]'s API and names no representation detail of its own.
//!
//! Where agreeing was cheap, this module already USES scratchy's shapes so the
//! swap is mechanical rather than a rewrite:
//!
//! | scratchy `ktir::ir`                  | here                     |
//! |--------------------------------------|--------------------------|
//! | `Ssa(pub u32)`                       | [`ir::Ssa`], identical   |
//! | `OpKind` (a variant per op spelling) | [`ir::OpKind`], same shape |
//! | `AttrKey` (a variant per attr name)  | [`ir::AttrKey`], same shape |
//! | `Attr<'a>` (a tagged attribute)      | [`ir::Attr`], same variants |
//! | `IrType<'a>` (dims + elem, not text) | [`ir::IrType`], same shape |
//! | `Operation<'a>` arena-borrowed, `Copy` | [`ir::Op`], **owned** |
//!
//! THE ONE DELIBERATE DIVERGENCE is that last row, and it is a divergence with a
//! reason rather than an oversight. scratchy's `Operation<'a>` is `Copy` and
//! immutable, built once into an `Arena` and thereafter only rebuilt
//! (`with_attr` returns a new op). That is right for a macro that BAKES a program
//! into a binary. It is wrong for nine passes whose entire job is to erase,
//! insert, retype and rewire ops in place -- `LegalizeTypes` alone runs a forward
//! fixed point that mutates result types until nothing changes. So [`ir::Op`]
//! owns its `Vec`s and passes mutate it. An arena-borrowed `'static` form is
//! recoverable from an owned one whenever the vendoring decision lands; the
//! reverse is not, which is why the mutable form is the one that is written here.

//! # WHAT IS DONE, AND WHAT IS NOT
//!
//! DONE, and measured against the C++ chain per fixture (see
//! `third_party/spyre/test/goldens/ktir/`):
//!
//! * the seven `make_ktir` stages above, for `attention_flash` (non-causal, causal,
//!   and the unit-scale variant that folds) and `swiglu_mlp` -- census and structure
//!   both matching;
//! * the refusals, by name: `vector_add` and `mul` on the one-stick corelet split;
//! * END TO END -- the port's own KTIR through `triton-superdsc-lower` and
//!   `dxp_standalone` reaches a **byte-identical** `init_binary.bin` to the one the
//!   C++ path produces, for all three fixtures that have a downstream. Run
//!   `bash third_party/spyre/test/goldens/ktir/e2e.sh`.
//!
//! NOT DONE, stated plainly rather than left to be discovered:
//!
//! * **The KTIR -> SuperDSC leg is no longer ours.** scratchy's `ktir_to_superdsc`
//!   walks `ktir_core::ir` directly, and our text-reading `triton-superdsc-lower` is
//!   superseded -- so the 35 text sites in it are NOT work to do, they are work that
//!   was deleted. What replaces it is emitting `ktir_core` values at our boundary.
//!   [`consumer`] measures how far that is, per fixture, and the answer is gated on
//!   the next bullet: `ktir_core::OpKind` has no `tt.*` variants, so our stage-1
//!   output is not merely unsupported by their walker but UNREPRESENTABLE in their
//!   type. `ToSchedulerKTIR` is what removes those ops, which makes it the
//!   prerequisite rather than the leftover.
//! * **THREE PASSES ARE DELIBERATELY NOT PORTED, AND MUST NOT BE**: `LaneMajorLayout`,
//!   `CarriedValuesToMemory` and `GridIndexChains`. Recorded here so nobody re-derives
//!   the reasoning and quietly "finishes" the port by adding them.
//!
//!   All three exist to satisfy deeptools' **V1 dataflow-scheduler** -- its compute-group
//!   model, its transfer laws, and a hole in its own `QueryMapArithCollapse`. THIS PATH
//!   ROUTES AROUND THAT SCHEDULER: SuperDSC is reached directly. Two of them would be
//!   ACTIVELY WRONG here, not merely unnecessary:
//!
//!   * `CarriedValuesToMemory` builds a FOLD TREE because V1 has no contraction op.
//!     SuperDSC has a native `MACC`, so the tree is work that exists to avoid a
//!     capability the target has. Their walker also handles the loop-carried accumulator
//!     its own way -- one `scf.for` iter_arg becomes a caller-seeded "carried" bundle
//!     argument -- so the transformation is not merely redundant, it removes the shape the
//!     consumer is looking for.
//!   * `LaneMajorLayout` inserts TRANSPOSES for V1's transfer model. `bmm.ddl` wants
//!     Triton's own layout with nothing prepared, so those transposes are data movement
//!     added for a consumer that is not on this path. (It is also why the stage-3
//!     reduction check in `tests/golden_sched.rs` is structural rather than
//!     axis-for-axis: the C++ golden reduces along the transposed axis BECAUSE that pass
//!     ran.)
//!   * `GridIndexChains` rebuilds tile-index expressions in the one form V1's
//!     `QueryMapArithCollapse` can fold into a per-core constant. It is a workaround for
//!     a hole in that pass, and the hole is not ours.
//!
//!   The two REFUSALS their goldens record still stand and are still tested: `swiglu_mlp`
//!   at stage 2 ("not a contiguous flat block") and `attention_flash_causal` at stage 4
//!   ("bounds are not all constant") -- measurements of the C++ chain, kept so causal's
//!   position-dependent trip count cannot be misread as a defect here.

pub mod consumer;
pub mod from_ttir;

pub mod ir;
pub mod passes;
pub mod text;

pub use ir::{Module, Op, OpKind, Ssa};

/// Every way this crate refuses.
///
/// FAIL CLOSED: an unsupported construct is an error that NAMES it. A port that
/// silently lowers part of a kernel is the failure mode this type exists to make
/// unrepresentable -- there is no `Option`-returning "best effort" entry point.
///
/// `pass` is the pass that refused, so a diagnostic reads like the C++ one it is
/// ported from (`"spyre-dot-to-linalg: ..."`, `"PlanCorelets: ..."`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub pass: &'static str,
    pub message: String,
}

impl Refusal {
    pub fn new(pass: &'static str, message: impl Into<String>) -> Refusal {
        Refusal { pass, message: message.into() }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.pass, self.message)
    }
}

pub type Result<T> = std::result::Result<T, Refusal>;

/// Run `make_ktir`'s pipeline, in `make_ktir`'s order, on a ttir value.
///
/// `grid` is the launch grid, fastest-varying axis first -- the same list
/// `spyre.passes.add_ktir_lowering(pm, list(options.grid))` threads into the C++
/// pipeline. An empty grid keeps `DistributeWork`'s single-axis heuristic, and a
/// multi-axis kernel then RED-stops rather than guessing a block count.
pub fn make_ktir(module: &mut Module, grid: &[i64]) -> Result<()> {
    passes::dot_to_linalg::run(module)?;
    passes::convert_ttir_to_ktdp::run(module)?;
    passes::distribute_work::run(module, grid)?;
    passes::legalize_types::run(module)?;
    passes::canonicalize::run(module)?;
    passes::decompose_dense_constants::run(module)?;
    passes::plan_corelets::run(module)?;
    Ok(())
}
