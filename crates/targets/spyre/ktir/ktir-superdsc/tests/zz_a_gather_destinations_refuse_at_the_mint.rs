// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A GATHER COPY WHOSE DESTINATION CROSSES ITS SCRATCH REFUSES AT THE MINT — the fail-closed
//! half of the Win lockdown's gather door, after the pointwise (#734), reduce (#736) and
//! restickify/convert mints.
//!
//! The gather copy writes `mb × out` elements at `entry_elems · dest_entry`, and until this door
//! NOTHING bounded that window against the scratch it lands in. That is exactly #717's class: a
//! page-granular op's entries advance a whole stick per op while its destination advances one
//! entry, so a base derived from the wrong one puts every op's rows past the scratch entirely — a
//! clean bake whose overrun is the next intermediate's bytes, read back as keys or page numbers.
//! Both doors below bound the window where it lands:
//! 1. THE RAW DOOR ([`gather_copy_opspec`]) — the element bound against the destination's own
//!    declared `[rows, cols]`, layout-free, at the copy's own df;
//! 2. THE PLACEMENT DOOR — the same window resolved by NAME against the layout's placement
//!    footprint, the "512 reserved against 2048 declared" class. It runs inside the crate, on the
//!    shipped copy legs (`batched_gathered::copy_opspecs`), so its pins live in that module's own
//!    tests (`a_copy_leg_past_its_destination_placement_refuses`,
//!    `a_copy_leg_into_a_core_local_destination_has_no_offset_to_take`).
//!
//! ⭐ AND THE CORE-LOCAL LX FORMS ARE THEIR OWN WINDOW KIND, NOT AN EXEMPTION. A copy that lands in
//! a core-local LX tensor states [`GatherDest::CoreLocalLx`]: each core's slice sits at the pin, so
//! the destination has NO OFFSET TO TAKE (a run at a nonzero destination entry is a build `Err`),
//! and the layout must agree the name is core-local — the minted law and the layout are two
//! authorities for one fact, and their disagreement is a build failure rather than a copy that
//! addresses LX bytes as HBM. The per-core slice itself is the read law's bound, checked against
//! the pin at emit (`core_local_reads_agree`).
//!
//! ⛔ THE fp8 LAYOUT HALF RIDES WITH THE MoE CALLER: the fp8 expert-slab copy states its own df at
//! the raw door, pinned green in `zz_the_gather_op_is_kernel_less` at the fp8 stick.
//!
//! Every refusal below has a control one variable away, and each control BUILDS.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec;
use ktir_superdsc::sdsc_abstract::GatherDest;
use ktir_superdsc::superdsc_opspec::{
    DestEntry, Df, EntryBase, GatherIndex, GatherRun, KernelAxis, OpSpec, PageExtent,
};

/// The vendor-shaped copy every raw-door leg starts from — one-stick sub-rows of `out`, one
/// position per entry, global block numbers — varying only the destination's law and where the run
/// starts. `mb`/`out`/`df` are the fp16 fold's own shape.
fn copy_opspec(mb: u32, run: GatherRun, dest: GatherDest) -> Result<OpSpec, String> {
    gather_copy_opspec(
        "KtPool",
        "KtScratch",
        GatherIndex {
            name: "BlockTable".to_string(),
            entry_dim: KernelAxis::Batch,
            page: PageExtent::single_position(),
            per_position: None,
            first_entry: EntryBase::ZERO,
        },
        ktir_superdsc::sdsc_abstract::GatherCopySpec {
            rows: mb,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
            df: Df::Fp16,
            // Global block numbers — no source base (the `GatherCopy` law).
            run,
            dest,
        },
    )
}

/// ⭐ THE RAW DOOR — a run whose `mb × out` window crosses the scratch it states. The control is the
/// exact-fit scratch: 8×64 holds the 512-element run with nothing left over, which is the law every
/// `copies()` mint guarantees at the LAST stick of a pass.
#[test]
fn a_run_past_its_declared_scratch_refuses_at_the_raw_door() {
    copy_opspec(
        8,
        GatherRun::CONTIGUOUS,
        GatherDest::Scratch {
            rows: 8,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
        },
    )
    .expect("the exact-fit scratch builds — the control");

    let err = copy_opspec(
        8,
        GatherRun::CONTIGUOUS,
        GatherDest::Scratch {
            rows: 4,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
        },
    )
    .expect_err("a 512-element run cannot land in a 256-element scratch");
    assert!(
        err.contains("gather_copy_opspec('KtPool' -> 'KtScratch'): the destination")
            && err.contains("[0, 512)")
            && err.contains("runs past the end its handle declares"),
        "the refusal names the copy, its window and the bound: {err}"
    );
}

/// ⭐ THE BASE, NOT JUST THE WIDTH — a run at a nonzero destination entry whose window crosses the
/// scratch's end. At one position per entry the run at entry 4 starts at element 256 and writes 512
/// more: `[256, 768)` of a 512-element scratch. The control is the exact-fit law the zero-fill
/// guarantee states: a 12×64 scratch ends EXACTLY where this run does, and it builds.
#[test]
fn a_destination_entry_past_the_scratch_refuses() {
    let run = GatherRun::split(DestEntry::of_entries(4), 0);
    let err = copy_opspec(
        8,
        run,
        GatherDest::Scratch {
            rows: 8,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
        },
    )
    .expect_err("the run at entry 4 ends at 768, past the 512-element scratch");
    assert!(
        err.contains("[256, 768)") && err.contains("runs past the end its handle declares"),
        "the refusal names the window the base put past the end: {err}"
    );

    copy_opspec(
        8,
        GatherRun::split(DestEntry::of_entries(4), 0),
        GatherDest::Scratch {
            rows: 12,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
        },
    )
    .expect("the run that ends exactly at the scratch's end builds — the control");
}

/// ⭐ THE CORE-LOCAL FORM'S OWN LAW, AT THE RAW DOOR — each core's slice sits at the pin, so a
/// core-local destination has no offset to take: a run at a nonzero destination entry is a build
/// `Err` however large anything is. The control is the same copy at entry 0: it builds.
#[test]
fn a_core_local_destination_has_no_offset_to_take() {
    copy_opspec(8, GatherRun::CONTIGUOUS, GatherDest::CoreLocalLx)
        .expect("a core-local destination at entry 0 builds — the control");

    let err = copy_opspec(
        8,
        GatherRun::split(DestEntry::of_entries(1), 0),
        GatherDest::CoreLocalLx,
    )
    .expect_err("a core-local destination has no offset to take");
    assert!(
        err.contains("CORE-LOCAL LX tensor") && err.contains("no offset to take"),
        "the refusal states the core-local law, not a size: {err}"
    );
}
