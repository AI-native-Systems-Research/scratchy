// SPDX-License-Identifier: Apache-2.0

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
use ktir_core::ddl_binds;
use std::num::NonZeroU32;

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
        Refusal {
            pass,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.pass, self.message)
    }
}

pub type Result<T> = std::result::Result<T, Refusal>;

/// ⭐ THE LAUNCH GRID A KERNEL IS COMPILED AT — one, two or three program-id axes,
/// fastest-varying first (Triton's `grid`, `program_id(0)` the first extent), every
/// extent at least one by type.
///
/// It is a required argument of the pipeline ([`make_ktir`], `to_ktir::run`): the
/// caller that launches the kernel states it, so nothing downstream infers a block
/// count from the IR or refuses for want of one. Its rank is the caller's statement of
/// how many axes the kernel reads; a kernel reading an axis past it is refused by
/// `DistributeWork`, naming both.
///
/// A literal grid is [`LaunchGrid::x`] / [`LaunchGrid::xy`] / [`LaunchGrid::xyz`],
/// whose zero extent fails const evaluation; a grid read from outside the program (a
/// fixture's `grid.txt`, a command line) is [`LaunchGrid::of_extents`], the one place
/// such a list is checked.
///
/// ```compile_fail,E0080
/// let _ = triton_ktir::LaunchGrid::xy::<4, 0>();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchGrid {
    X(NonZeroU32),
    XY(NonZeroU32, NonZeroU32),
    XYZ(NonZeroU32, NonZeroU32, NonZeroU32),
}

impl LaunchGrid {
    /// One program: the grid every whole-node kernel launches at.
    pub const ONE: LaunchGrid = LaunchGrid::X(NonZeroU32::MIN);

    const fn nz(n: u32) -> NonZeroU32 {
        match NonZeroU32::new(n) {
            Some(v) => v,
            None => panic!("a launch-grid extent is at least one program"),
        }
    }

    /// A one-axis literal grid of `X` programs.
    pub const fn x<const X: u32>() -> LaunchGrid {
        LaunchGrid::X(const { Self::nz(X) })
    }

    /// A two-axis literal grid, `X` fastest.
    pub const fn xy<const X: u32, const Y: u32>() -> LaunchGrid {
        LaunchGrid::XY(const { Self::nz(X) }, const { Self::nz(Y) })
    }

    /// A three-axis literal grid, `X` fastest.
    pub const fn xyz<const X: u32, const Y: u32, const Z: u32>() -> LaunchGrid {
        LaunchGrid::XYZ(
            const { Self::nz(X) },
            const { Self::nz(Y) },
            const { Self::nz(Z) },
        )
    }

    /// A grid read from outside the program — a fixture's `grid.txt`, a command line.
    ///
    /// # Errors
    /// No extent, more than three, or one that is not a positive `u32`.
    pub fn of_extents(extents: &[i64]) -> Result<LaunchGrid> {
        let axis = |e: i64| {
            u32::try_from(e)
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| {
                    Refusal::new(
                        "LaunchGrid",
                        format!("a launch-grid extent is a positive u32 program count; got {e}"),
                    )
                })
        };
        match *extents {
            [x] => Ok(LaunchGrid::X(axis(x)?)),
            [x, y] => Ok(LaunchGrid::XY(axis(x)?, axis(y)?)),
            [x, y, z] => Ok(LaunchGrid::XYZ(axis(x)?, axis(y)?, axis(z)?)),
            _ => Err(Refusal::new(
                "LaunchGrid",
                format!(
                    "a launch grid has one to three axes (Triton's program_id 0..=2); got {} \
                     extents {extents:?}",
                    extents.len()
                ),
            )),
        }
    }

    /// The extents, fastest-varying axis first — one per stated axis.
    pub fn extents(&self) -> Vec<i64> {
        match *self {
            LaunchGrid::X(x) => vec![x.get().into()],
            LaunchGrid::XY(x, y) => vec![x.get().into(), y.get().into()],
            LaunchGrid::XYZ(x, y, z) => vec![x.get().into(), y.get().into(), z.get().into()],
        }
    }

    /// The program count — the product of the extents.
    pub fn programs(&self) -> i64 {
        self.extents().iter().product()
    }
}

/// One pass of a pipeline table: a named step plus the function that runs it.
/// Both drivers' tables are `[(name, Pass); N]`, walked by one runner.
pub type Pass = fn(&mut Module, &LaunchGrid) -> Result<()>;

/// Run `make_ktir`'s pipeline, in `make_ktir`'s order, on a ttir value.
///
/// After every pass the SSA verifier runs ([`passes::verify_defs`]): a module
/// that uses a value nothing defines is refused the moment the pass that left
/// it that way completes, naming that pass.
///
/// `grid` is the launch grid, fastest-varying axis first -- the same list
/// `spyre.passes.add_ktir_lowering(pm, list(options.grid))` threads into the C++
/// pipeline, stated by the caller as a [`LaunchGrid`].
/// One runner, one pipeline table: each row is a pass, and the runner runs the
/// SSA verifier after EVERY entry, so no pass can be added to `make_ktir`
/// without a verify line following it and none can be quietly skipped. A row's
/// comment says why the pass sits where it sits.
pub fn make_ktir(module: &mut Module, grid: &LaunchGrid) -> Result<()> {
    let pipeline: [(&'static str, Pass); 7] = [
        ("make_ktir:dot_to_linalg", |m, _| {
            passes::dot_to_linalg::run(m)
        }),
        ("make_ktir:convert_ttir_to_ktdp", |m, _| {
            passes::convert_ttir_to_ktdp::run(m)
        }),
        ("make_ktir:distribute_work", |m, g| {
            passes::distribute_work::run(m, g)
        }),
        // The BAKE'S ENTRY chooses the generation ONCE: every reader below is
        // generic over `G`, and this row — the one place the pipeline enters
        // them — names the baked generation.
        ("make_ktir:legalize_types", |m, _| {
            passes::legalize_types::run::<ddl_binds::Baked>(m)
        }),
        ("make_ktir:canonicalize", |m, _| {
            passes::canonicalize::run(m)
        }),
        ("make_ktir:decompose_dense_constants", |m, _| {
            passes::decompose_dense_constants::run(m)
        }),
        ("make_ktir:plan_corelets", |m, _| {
            passes::plan_corelets::run(m)
        }),
    ];
    for (name, pass) in pipeline {
        pass(module, grid)?;
        passes::verify_defs::verify_module_defs(module, name).map_err(|v| v.refusal())?;
    }
    Ok(())
}

#[cfg(test)]
mod launch_grid_tests {
    use super::LaunchGrid;

    /// The one boundary a grid read from outside the program crosses: one to three
    /// positive extents, fastest first, and nothing else.
    #[test]
    fn a_grid_from_outside_is_one_to_three_positive_extents() {
        assert_eq!(LaunchGrid::of_extents(&[16]), Ok(LaunchGrid::x::<16>()));
        assert_eq!(
            LaunchGrid::of_extents(&[2, 4]),
            Ok(LaunchGrid::xy::<2, 4>())
        );
        assert_eq!(
            LaunchGrid::of_extents(&[2, 4, 3]),
            Ok(LaunchGrid::xyz::<2, 4, 3>())
        );
        for bad in [&[][..], &[0], &[4, -1], &[1, 2, 3, 4], &[1 << 33]] {
            assert!(
                LaunchGrid::of_extents(bad).is_err(),
                "{bad:?} is no launch grid"
            );
        }
    }

    #[test]
    fn a_grid_states_its_extents_and_program_count() {
        let g = LaunchGrid::xy::<2, 4>();
        assert_eq!(g.extents(), vec![2, 4]);
        assert_eq!(g.programs(), 8);
        assert_eq!(LaunchGrid::ONE.extents(), vec![1]);
    }
}
