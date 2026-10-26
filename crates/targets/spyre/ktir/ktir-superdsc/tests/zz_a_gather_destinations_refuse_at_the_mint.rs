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
//! 2. THE PLACEMENT DOOR ([`assemble_gather_copy`]) — the same window resolved by NAME against the
//!    layout's placement footprint, the "512 reserved against 2048 declared" class.
//!
//! ⭐ AND THE CORE-LOCAL LX FORMS ARE THEIR OWN WINDOW KIND, NOT AN EXEMPTION. A copy that lands in
//! a core-local LX tensor states [`GatherDest::CoreLocalLx`]: each core's slice sits at the pin, so
//! the destination has NO OFFSET TO TAKE (a run at a nonzero destination entry is a build `Err`),
//! and the layout must agree the name is core-local — the minted law and the layout are two
//! authorities for one fact, and their disagreement is a build failure rather than a copy that
//! addresses LX bytes as HBM. The per-core slice itself is the read law's bound, checked against
//! the pin at emit (`core_local_reads_agree`).
//!
//! ⛔ THE fp8 LAYOUT HALF RIDES WITH THE MoE CALLER: [`assemble_gather_copy`] is the fp16 KV door
//! (every gathered fold to date copies fp16 KV planes); the fp8 expert-slab copy states its own df
//! at the raw door, pinned green in `zz_the_gather_op_is_kernel_less` at the fp8 stick.
//!
//! Every refusal below has a control one variable away, and each control BUILDS.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{assemble_gather_copy, gather_copy_opspec};
use ktir_superdsc::place::{PlaceId, SynthRole};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    ChunkLx, ChunkRowKind, ChunkScratch, GatherCopy, GatherDest, LxDest, PageScratch, PagedKvPool,
    QueryRowCount,
};
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

/// A layout placing `t{tid}` as an activation of `size` bytes — the same rig the pointwise,
/// reduce and restickify/convert refusal tests drive their placement doors with.
fn layout_placing(tid: u32, size: u64) -> BundleLayout {
    let mut layout = BundleLayout::default();
    let name = PlaceId::Act(tid).to_string();
    layout
        .ids
        .borrow_mut()
        .insert(name.clone(), PlaceId::Act(tid));
    layout.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 0,
            size,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    layout
}

/// The panic text out of a refused build — the assemblers panic with `format!`ed refusal messages,
/// so the refusal is observable as the payload.
fn refused(r: Result<EmittedOp, Box<dyn std::any::Any + Send>>) -> String {
    let payload = match r {
        Ok(_) => panic!("the build was accepted — the refusal under test did not fire"),
        Err(p) => p,
    };
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        panic!("a panic payload that is not a string")
    }
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

/// The rig's other two operands — the pool the copy reads and the table it reads it through —
/// declared to the layout as synthetics at the copy's own extents, so the destination's doors are
/// the only variables the legs below leave. Returns the two names to build the copy with.
fn declare_gather_rig(layout: &BundleLayout, cp: GatherCopy) -> (String, String) {
    let src_id = PlaceId::Synth {
        of: 5,
        role: SynthRole::Rot,
    };
    let idx_id = PlaceId::Synth {
        of: 6,
        role: SynthRole::Xc,
    };
    // The pool's window for one run is the run's own `mb × out` elements (an index step advances
    // `skip_addr = page × out`, and the run covers `mb / page` of them); the table's is the run's
    // entries from the copy's own index base.
    layout.synth(src_id, &[cp.dims().mb(), cp.dims().out()]);
    let idx_entries = cp.index_base().entries()
        + cp.dims()
            .entries()
            .expect("the page divides `mb` by construction");
    layout.synth_df(idx_id, &[idx_entries, 1], Df::SenUint32);
    (src_id.to_string(), idx_id.to_string())
}

/// ⭐ THE PLACEMENT DOOR — the handle states the truth and the placement does not. The run is the
/// shipped decode door's own first copy ([`PageScratch::of_pass`]), whose carried law fits its
/// declared scratch exactly; the LAYOUT holds a placement half the run's bytes, which is the "512
/// reserved against 2048 declared" class: a clean bake whose overrun is another tensor's bytes. The
/// control is the placement the copy's own law states: it builds.
#[test]
fn a_destination_past_its_placement_refuses() {
    let pool = PagedKvPool::new(8, 64);
    let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(8))
        .expect("granite's geometry admits the page copy");
    let cp = scratch
        .copies()
        .next()
        .expect("a pass has at least one run");
    let GatherDest::Scratch { rows, cols } = cp.dest() else {
        panic!("the page scratch's first run lands in the HBM scratch")
    };
    let window_bytes = u64::from(cp.dims().mb()) * u64::from(cp.dims().out()) * 2;

    let small = layout_placing(7, window_bytes / 2);
    let (src, index) = declare_gather_rig(&small, cp);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        assemble_gather_copy("gdst_small", &src, "t7", cp, &index, &mut sym, Some(&small))
    }));
    let msg = refused(r);
    assert!(
        msg.contains("assemble_gather_copy gdst_small: the destination")
            && msg.contains("exceeds its placement footprint"),
        "the refusal names the copy and the placement it would overrun: {msg}"
    );

    let true_layout = layout_placing(7, u64::from(rows) * u64::from(cols) * 2);
    let (src, index) = declare_gather_rig(&true_layout, cp);
    let mut sym = 0i64;
    let built = assemble_gather_copy(
        "gdst_true",
        &src,
        "t7",
        cp,
        &index,
        &mut sym,
        Some(&true_layout),
    );
    assert!(
        built.op.is_some(),
        "the placement the copy's own law states builds — the control"
    );
}

/// The first ladder width whose value leg stays in LX, with one of its core-local copies — the
/// shipped chunk form past 8 pages, minted by the same `lx_form` the emission calls.
fn an_lx_form() -> (ChunkLx, GatherCopy) {
    let pool = PagedKvPool::new(8, 64);
    for &pages in &ChunkScratch::LADDER {
        let Some(lx) = ChunkScratch::of_chunk(pool, pages)
            .and_then(|c| c.lx_form(ChunkRowKind::OneRequest, 0, 0))
        else {
            continue;
        };
        if let Some((_, cp)) = lx.copies().find(|(d, _)| matches!(d, LxDest::Lx(_))) {
            return (lx, cp);
        }
    }
    panic!("no ladder width mints an LX-resident value copy")
}

/// ⭐ THE TWO-AUTHORITY LAW — the copy states a CORE-LOCAL destination and the layout does not hold
/// the name core-local. The disagreement leg holds the very same name as a plain HBM activation:
/// two authorities, one fact, opposite answers, and the copy would address HBM bytes as a pin (or
/// LX bytes as HBM). The control is the layout the shipped chunk emission synth's — the same name,
/// core-local, at the pin size the chunk's own law states — and it builds.
#[test]
fn a_core_local_destination_the_layout_does_not_hold_refuses() {
    let (lx, cp) = an_lx_form();

    let hbm = layout_placing(7, 1 << 20);
    let (src, index) = declare_gather_rig(&hbm, cp);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        assemble_gather_copy("glx_hbm", &src, "t7", cp, &index, &mut sym, Some(&hbm))
    }));
    let msg = refused(r);
    assert!(
        msg.contains("assemble_gather_copy glx_hbm")
            && msg.contains("does not hold 't7' as core-local LX"),
        "the refusal names the name the two authorities disagree about: {msg}"
    );

    let core_id = PlaceId::Synth {
        of: 7,
        role: SynthRole::VRep,
    };
    let lx_layout = BundleLayout::default();
    lx_layout.synth_lx_core_local(core_id, lx.per_core_bytes());
    let (src, index) = declare_gather_rig(&lx_layout, cp);
    let mut sym = 0i64;
    let built = assemble_gather_copy(
        "glx_true",
        &src,
        &core_id.to_string(),
        cp,
        &index,
        &mut sym,
        Some(&lx_layout),
    );
    assert!(
        built.op.is_some(),
        "the core-local destination the chunk's own law pins builds — the control"
    );
}
