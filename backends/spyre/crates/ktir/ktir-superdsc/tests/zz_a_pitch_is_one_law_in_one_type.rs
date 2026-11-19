// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ THE BACK-GAP LAW IS ONE TYPE — `Pitch::past` builds a pitch's gap ONCE, from the axis's
//! WHOLE swept extent and the allocation's declared one, and every reader takes the value.
//!
//! The law (plan 2026-11-19 § A1): an HBM view's outer step on an axis is `N_[axis] +
//! backGapCore_[axis]` (`dsc/dsc2.cpp:2900-2926`; an HBM view's capacity is `N_`, `:3626-3632`),
//! so a view whose declared extent on an axis exceeds the op's whole sweep states
//! `backGapCore_[axis] = {"-1": "<declared − swept>"}`, the gap a JSON string. torch-spyre states
//! the same law (`superdsc.py:1686-1691`, `backGap = dev_dim_size − it_dim_size`), the stick dim
//! rounded up to whole sticks (`:1630-1633`). The swept extent is the WHOLE `N_`, never one core's
//! — MEASURED on the card 2026-11-08 (branch `tp-allreduce` `5d64673767`): a per-core gap read the
//! gap fill wherever the cores split the swept axis.
//!
//! Ported from `tp-allreduce` `50f600e0b7`'s per-axis near-miss suite (there in the leading-axis
//! sum's fixture, a feature this tree does not carry), restated over the one type: the law at,
//! past and short of the sweep; the stick-dim law on the view, card-proven by step 0 of the rope
//! plan (2026-11-19: 40/40 cells bit-exact, `{"out": {"-1": "128"}}`), and the df a stick pitch
//! freezes; the per-trip pitch a time tile re-builds through the same constructor; one axis, one
//! mechanism, in either order and through the rank-erased writers; the residency half (no LX view
//! carries a pitch, in either order); and the wire spelling dxp parses, keyed by the axis marker
//! the pitch was stated with.

use ktir_superdsc::superdsc_opspec::{
    Allocation, AnyTensorArg, CoreBudget, DeviceAxis, Df, DimName, InAxis, ItDim, MbAxis, OutAxis,
    Pitch, PitchAxis, Role, Scale, TensorArg, WorkPlan, YAxis,
};
use ktir_superdsc::wire::BackGap;
use std::num::NonZeroU32;

fn dim(name: &'static str, size: u32, is_stick: bool) -> ItDim {
    ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    }
}

/// A `[y = 3, mb = 4, out = 128]` plan, unsplit — the view's own extents, nothing else.
fn plan() -> WorkPlan {
    WorkPlan::divide(
        &[
            dim("y", 3, false),
            dim("mb", 4, false),
            dim("out", 128, true),
        ],
        CoreBudget::whole(),
        |_, _| Default::default(),
    )
    .expect("an unsplit plan")
}

fn view(plan: &WorkPlan, allocation: Allocation) -> TensorArg<3> {
    TensorArg::<3>::new(
        true,
        "t".to_string(),
        Role::Output,
        [Scale::Active; 3],
        plan.iter_syms(["y", "mb", "out"]),
        ["y", "mb", "out"],
        "out",
        allocation,
    )
    .expect("a whole-stick view")
}

fn axes(t: TensorArg<3>) -> Vec<DeviceAxis> {
    AnyTensorArg::R3(t).view().device_extent.to_vec()
}

fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("a nonzero gap")
}

/// The pitch `Pitch::past` builds for the `mb` rows axis — the one way to spell an expected value.
fn rows_pitch(swept: u32, declared: u32) -> Pitch {
    Pitch::past(DimName::of::<MbAxis>(), PitchAxis::Rows, swept, declared)
        .expect("a pitch at or past the sweep")
        .expect("a pitch past the sweep")
}

/// The pitch `Pitch::past` builds for the fp16 `out` stick axis.
fn stick_pitch(swept: u32, declared: u32) -> Option<Pitch> {
    Pitch::past(
        DimName::of::<OutAxis>(),
        PitchAxis::Stick(Df::Fp16),
        swept,
        declared,
    )
    .expect("whole sticks")
}

/// ⭐ THE LAW ITSELF: `gap = declared − whole swept`, built once. At the sweep there is no pitch
/// at all; short of it is refused by name. The pitch remembers the axis it was stated on.
#[test]
fn the_gap_is_the_declared_extent_past_the_whole_sweep() {
    let p = rows_pitch(4, 10);
    assert_eq!((p.declared(), p.gap()), (10, nz(6)));
    assert_eq!(p.dim(), DimName::of::<MbAxis>());
    let mb = DimName::of::<MbAxis>();
    assert_eq!(
        Pitch::past(mb, PitchAxis::Rows, 4, 4).expect("at the sweep"),
        None,
        "a declared extent AT the sweep is no pitch"
    );
    let short = Pitch::past(mb, PitchAxis::Rows, 4, 3).unwrap_err().0;
    assert!(short.contains("shorter than the extent"), "{short}");
}

/// ⭐ THE STICK-DIM LAW, WRITTEN ONCE (purity item 2): the swept extent rounds UP to whole sticks
/// before the gap is taken, and the declared extent is whole sticks. On the view it is card-proven
/// by step 0 of the rope plan (2026-11-19: 40/40 cells bit-exact, the gap `{"out": {"-1": "128"}}`
/// in ELEMENTS, accepted by dxp).
#[test]
fn the_stick_axis_rounds_the_sweep_up_to_whole_sticks() {
    // The rope plan's stick factor at gemma-4's sliding head: each half-move sweeps `hd/2 = 128`
    // of every `hd = 256` head.
    let half = stick_pitch(128, 256).expect("a pitch");
    assert_eq!(half.gap(), nz(128));
    // A sub-stick sweep rounds up: 96 elements sweep two whole 64-lane sticks.
    assert_eq!(stick_pitch(96, 256).expect("a pitch").gap(), nz(128));
    assert_eq!(stick_pitch(100, 128), None, "100 rounds to 128: no pitch");
    // The stick is the operand's format's: fp8 sticks are 128 lanes.
    let out = DimName::of::<OutAxis>();
    let fp8 = Pitch::past(out, PitchAxis::Stick(Df::Fp8), 128, 512)
        .expect("whole sticks")
        .expect("a pitch");
    assert_eq!(fp8.gap(), nz(384));
    // An allocation is whole sticks.
    let frac = Pitch::past(out, PitchAxis::Stick(Df::Fp16), 64, 200)
        .unwrap_err()
        .0;
    assert!(frac.contains("whole 64-element sticks"), "{frac}");
    // ⭐ ON THE VIEW: the half-move's own form — `out` sweeps 128 of a 256-element allocation —
    // states the step-0 wire exactly, the gap in ELEMENTS at the HBM core key.
    let p = plan();
    let stick = AnyTensorArg::R3(
        view(&p, Allocation::Hbm)
            .with_pitch::<OutAxis>(&p, 256)
            .unwrap(),
    );
    assert_eq!(
        stick.view().device_extent,
        [
            DeviceAxis::Swept,
            DeviceAxis::Swept,
            DeviceAxis::Pitch(half)
        ]
    );
    assert_eq!(
        serde_json::to_value(BackGap::of_hbm_view(&stick.view()).expect("a stick pitch"))
            .expect("the gap serializes"),
        serde_json::json!({"out": {"-1": "128"}})
    );
}

/// ⛔ A STICK-AXIS PITCH FREEZES THE DF — its gap was rounded in that df's sticks, so neither df
/// writer (`with_df`, `set_df` — one writer underneath) may re-format the view under it; re-stating
/// the same df passes, and a rows pitch freezes nothing.
#[test]
fn a_stick_pitch_freezes_the_df_it_was_stated_in() {
    let p = plan();
    let stuck = view(&p, Allocation::Hbm)
        .with_pitch::<OutAxis>(&p, 256)
        .unwrap();
    let e = stuck.clone().with_df(Df::Fp8).unwrap_err().0;
    assert!(e.contains("stick axis 'out' states a pitch"), "{e}");
    let mut erased = AnyTensorArg::R3(stuck);
    let e = erased.set_df(Df::Fp32).unwrap_err().0;
    assert!(e.contains("another format's sticks"), "{e}");
    erased.set_df(Df::Fp16).expect("the same df is no change");
    assert_eq!(erased.view().df, Df::Fp16);
    // The df stated BEFORE the pitch is the one the pitch rounds in.
    view(&p, Allocation::Hbm)
        .with_df(Df::Fp16)
        .unwrap()
        .with_pitch::<OutAxis>(&p, 256)
        .expect("df first, then the pitch");
    // A rows pitch is not in sticks: the df stays free.
    let mut rows = AnyTensorArg::R3(
        view(&p, Allocation::Hbm)
            .with_pitch::<MbAxis>(&p, 10)
            .unwrap(),
    );
    rows.set_df(Df::Fp32).expect("a rows pitch freezes nothing");
}

/// ⭐ A TIME-TILED TRIP'S PITCH IS RE-BUILT BY THE ONE CONSTRUCTOR — `Pitch::per_trip` is
/// `Pitch::past` over the pitch's own axis and declared extent at the trip's sweep: the gap grows
/// by what the trip no longer sweeps, the stick law still rounds, the axis is the one the pitch
/// was stated on, and a trip that sweeps the whole allocation states no pitch.
#[test]
fn a_trip_re_builds_its_pitch_through_past() {
    let half = stick_pitch(128, 256).expect("a pitch");
    assert_eq!(half.per_trip(64).unwrap(), stick_pitch(64, 256));
    assert_eq!(half.per_trip(64).unwrap().unwrap().gap(), nz(192));
    assert_eq!(half.per_trip(96).unwrap().unwrap().gap(), nz(128));
    assert_eq!(
        half.per_trip(64).unwrap().unwrap().dim(),
        DimName::of::<OutAxis>()
    );
    assert_eq!(half.per_trip(256).unwrap(), None);
    assert_eq!(rows_pitch(4, 10).per_trip(2).unwrap().unwrap().gap(), nz(8));
}

/// ⭐ ONE AXIS, ONE MECHANISM — a window and a pitch are two device mechanisms for one slot, and
/// every near-miss refuses by name at build, in either order and through the rank-erased writers.
#[test]
fn a_pitch_is_one_mechanism_per_axis_and_its_near_misses_refuse_by_name() {
    let p = plan();
    // A pitch past the extent states it; a pitch AT the extent is no pitch.
    assert_eq!(
        axes(
            view(&p, Allocation::Hbm)
                .with_pitch::<MbAxis>(&p, 10)
                .unwrap()
        ),
        [
            DeviceAxis::Swept,
            DeviceAxis::Pitch(rows_pitch(4, 10)),
            DeviceAxis::Swept
        ]
    );
    assert_eq!(
        axes(
            view(&p, Allocation::Hbm)
                .with_pitch::<MbAxis>(&p, 4)
                .unwrap()
        ),
        [DeviceAxis::Swept; 3]
    );
    // The near-miss the law exists for: a pitch SHORTER than the sweep.
    let short = view(&p, Allocation::Hbm)
        .with_pitch::<MbAxis>(&p, 3)
        .unwrap_err()
        .0;
    assert!(short.contains("shorter than the extent"), "{short}");
    // An axis the view does not have.
    let absent = view(&p, Allocation::Hbm)
        .with_pitch::<InAxis>(&p, 8)
        .unwrap_err()
        .0;
    assert!(absent.contains("not in layout"), "{absent}");
    // One axis, one mechanism — in either order.
    let both = view(&p, Allocation::Hbm)
        .with_device_extent("mb", 16)
        .unwrap()
        .with_pitch::<MbAxis>(&p, 10)
        .unwrap_err()
        .0;
    assert!(both.contains("one axis states one"), "{both}");
    let both = view(&p, Allocation::Hbm)
        .with_pitch::<MbAxis>(&p, 10)
        .unwrap()
        .with_device_extent("mb", 16)
        .unwrap_err()
        .0;
    assert!(both.contains("one axis states one"), "{both}");
    // Different axes may each state their own.
    assert_eq!(
        axes(
            view(&p, Allocation::Hbm)
                .with_device_extent("y", 8)
                .unwrap()
                .with_pitch::<MbAxis>(&p, 10)
                .unwrap()
        ),
        [
            DeviceAxis::Window(8),
            DeviceAxis::Pitch(rows_pitch(4, 10)),
            DeviceAxis::Swept
        ]
    );
    // The rank-erased writers state the same law over the same slot.
    let mut erased = AnyTensorArg::R3(view(&p, Allocation::Hbm));
    erased.set_pitch::<MbAxis>(&p, 10).expect("a rows pitch");
    let both = erased.set_device_extent("mb", 16).unwrap_err().0;
    assert!(both.contains("one axis states one"), "{both}");
    let absent = erased.set_device_extent("in", 16).unwrap_err().0;
    assert!(absent.contains("not in layout"), "{absent}");
    // A pitch re-stated over its own slot is the same law again: at the sweep the axis is swept
    // whole (the row-sliced contraction's standalone block).
    erased.set_pitch::<MbAxis>(&p, 4).expect("a re-statement");
    assert_eq!(erased.view().device_extent, [DeviceAxis::Swept; 3]);
}

/// ⛔ NO LX VIEW CARRIES A PITCH (purity item 3) — the residency half of the law, refused by name
/// at both of its writers: a pitch on an LX view, and an LX residency on a pitched view (which
/// leaves the view as it was). A window is not a pitch; a declared extent AT the sweep states none.
#[test]
fn an_lx_view_carries_no_pitch_in_either_order() {
    let p = plan();
    for lx in [Allocation::Lx, Allocation::LxCoreLocal] {
        let e = view(&p, lx).with_pitch::<MbAxis>(&p, 10).unwrap_err().0;
        assert!(e.contains("LX-resident"), "{lx:?}: {e}");
        assert_eq!(
            axes(view(&p, lx).with_pitch::<MbAxis>(&p, 4).unwrap()),
            [DeviceAxis::Swept; 3],
            "{lx:?}: at the sweep there is no pitch to refuse"
        );
        let mut pitched = AnyTensorArg::R3(
            view(&p, Allocation::Hbm)
                .with_pitch::<MbAxis>(&p, 10)
                .unwrap(),
        );
        let e = pitched.set_allocation(lx).unwrap_err().0;
        assert!(e.contains("LX-resident"), "{lx:?}: {e}");
        assert_eq!(
            pitched.view().allocation,
            Allocation::Hbm,
            "{lx:?}: a refused residency leaves the view as it was"
        );
        let mut windowed = AnyTensorArg::R3(
            view(&p, Allocation::Hbm)
                .with_device_extent("mb", 16)
                .unwrap(),
        );
        windowed
            .set_allocation(lx)
            .expect("a window is not a pitch");
        assert_eq!(windowed.view().allocation, lx);
    }
}

/// ⭐ THE WIRE: the back gap is READ off the view's pitches, never re-derived, keyed by the marker
/// each pitch was stated with, and spelled as dxp's parser reads it — `{"<axis>": {"-1":
/// "<gap>"}}`, HBM keyed at core `-1`, the gap a JSON STRING (`dsc/dsc2.cpp:1782-1791`,
/// `std::stoi(gap.string_value())`), its axes sorted by their wire names. A view with no pitch —
/// swept, a window, or a pitch at its sweep — carries no `backGapCore_` at all.
#[test]
fn the_back_gap_is_read_off_the_pitch_and_spelled_as_dxp_parses_it() {
    let p = plan();
    let pitched = AnyTensorArg::R3(
        view(&p, Allocation::Hbm)
            .with_pitch::<MbAxis>(&p, 10)
            .unwrap(),
    );
    let gap = BackGap::of_hbm_view(&pitched.view()).expect("a pitched view carries a back gap");
    assert_eq!(gap.gap::<MbAxis>(), Some(nz(6)));
    assert_eq!(gap.gap::<YAxis>(), None);
    assert_eq!(
        serde_json::to_value(&gap).expect("the gap serializes"),
        serde_json::json!({"mb": {"-1": "6"}})
    );
    // Two pitched axes, each keyed by its own marker.
    let two = AnyTensorArg::R3(
        view(&p, Allocation::Hbm)
            .with_pitch::<OutAxis>(&p, 256)
            .unwrap()
            .with_pitch::<MbAxis>(&p, 10)
            .unwrap(),
    );
    let gap = BackGap::of_hbm_view(&two.view()).expect("two pitches");
    assert_eq!(
        (gap.gap::<MbAxis>(), gap.gap::<OutAxis>()),
        (Some(nz(6)), Some(nz(128)))
    );
    assert_eq!(
        serde_json::to_string(&gap).expect("the gap serializes"),
        r#"{"mb":{"-1":"6"},"out":{"-1":"128"}}"#
    );
    for t in [
        view(&p, Allocation::Hbm),
        view(&p, Allocation::Hbm)
            .with_device_extent("mb", 16)
            .unwrap(),
        view(&p, Allocation::Hbm)
            .with_pitch::<MbAxis>(&p, 4)
            .unwrap(),
    ] {
        assert!(BackGap::of_hbm_view(&AnyTensorArg::R3(t).view()).is_none());
    }
}
