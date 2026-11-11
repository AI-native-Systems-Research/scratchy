// SPDX-License-Identifier: Apache-2.0
//! ⭐ A ROW BAND OF A TALLER PARENT STATES THE PARENT'S PITCH AS THE WALK'S BACK GAP.
//!
//! A pointwise op over rows `[j·r, (j+1)·r)` of a stick-major `[p, c]` parent (`c` past one stick)
//! must stride the parent's stick-column planes at `p·64`, not at the band's own rows. dxp's walk is
//! a size cap it never reads as a stride, so the pitch is `backGapCore_[mb] = p − r` on every
//! banded HBM operand (an HBM view's capacity includes the core fold, so the gap is past the
//! band's whole sweep) — MEASURED on the card (microbenchmark, 2026-11-08): the pinned walk placed
//! stick `j` at `j·r·64` ([1,4096] of [4,4096], [1,8192] of [4,8192], [16,384] of [64,384]), and
//! this gap read and wrote them bit-exact. The output (a whole tensor) carries none, and no view states a pinned walk for it.

use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, elementwise};
use ktir_superdsc::ktir_node::Elementwise;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

type W = TpWorld<1>;

const PARENT: u32 = 10;
const OUT: u32 = 11;

fn layout(p: u32, r: u32, c: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, rows) in [(PARENT, p), (OUT, r)] {
        let size = u64::from(rows) * u64::from(c) * 2;
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: off,
                size,
            },
        );
        off += size.next_multiple_of(128);
    }
    l.segment_bytes[3] = off;
    l
}

/// The `add` of bands 0 and 1 of the parent into a whole `[r, c]` output; returns each operand's
/// allocate node, `numCoresUsed_` and the rows each core sweeps.
fn band_add(p: u32, r: u32, c: u32) -> (Vec<serde_json::Value>, u64, u64) {
    let l = layout(p, r, c);
    let band = |j: u32| Region {
        tid: PARENT,
        v_rows: p,
        v_cols: c,
        r_start: j * r,
        r_len: r,
        r_cover: (j * r, (j + 1) * r),
        c_start: 0,
        c_len: c,
        c_cover: (0, c),
        is_out: false,
        is_fp8: false,
        band: None,
    };
    let o = Region {
        tid: OUT,
        v_rows: r,
        v_cols: c,
        r_start: 0,
        r_len: r,
        r_cover: (0, r),
        c_start: 0,
        c_len: c,
        c_cover: (0, c),
        is_out: true,
        is_fp8: false,
        band: None,
    };
    let mut sym = 0i64;
    let ops = elementwise::<W>(
        "band",
        Elementwise::Add,
        &[band(0), band(1), o],
        &[],
        &mut sym,
        Some(&l),
    )
    .unwrap_or_else(|e| panic!("[{r}, {c}] of [{p}, {c}]: {}", e.message));
    assert_eq!(ops.len(), 1, "one descriptor");
    let v = serde_json::to_value(&ops[0].op).unwrap();
    let d = v["dscs_"][0]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .clone();
    let cores = d["numCoresUsed_"].as_u64().unwrap();
    // The band's whole row sweep — what the HBM unit view's capacity spans (core fold included).
    let swept = d["N_"]["mb_"].as_u64().expect("a swept mb");
    (d["scheduleTree_"].as_array().unwrap().clone(), cores, swept)
}

/// ⭐ THE LAW, at every core count the division reaches (1, 8 and 32 among them): both banded
/// operands carry `backGapCore_ {mb: {"-1": p − r}}`, the output carries none, and no operand pins
/// its walk to the parent (`maxDimSizes_` is the swept walk or `-1`).
#[test]
fn a_multi_stick_band_states_the_parent_pitch_as_its_back_gap() {
    let mut seen = std::collections::BTreeSet::new();
    let mut cases = Vec::new();
    for (p, r, c) in [
        (4u32, 1u32, 4096u32),
        (4, 1, 8192),
        (64, 16, 384),
        (1024, 512, 128),
        (16, 8, 128),
        (8, 2, 256),
        (2, 1, 128),
        (6, 3, 192),
        (3, 1, 320),
        // 37 sticks — prime past 32, so the stick axis divides over ONE core.
        (2, 1, 37 * 64),
    ] {
        let (nodes, cores, rpc) = band_add(p, r, c);
        seen.insert(cores);
        cases.push(format!(
            "[{r},{c}]/[{p},{c}] -> {cores} core(s), {rpc} row(s) swept"
        ));
        let gap = (u64::from(p) - rpc).to_string();
        for (i, n) in nodes.iter().enumerate() {
            let what = format!("[{r}, {c}] of [{p}, {c}] on {cores} core(s), operand {i}");
            if i < 2 {
                assert_eq!(
                    n["backGapCore_"],
                    serde_json::json!({"mb": {"-1": gap}}),
                    "{what}: the parent's pitch past the band's rows"
                );
            } else {
                assert!(
                    n.get("backGapCore_").is_none(),
                    "{what}: a whole output has no gap"
                );
            }
            let walk = n["maxDimSizes_"].as_array().unwrap();
            assert!(
                walk.iter()
                    .all(|e| e.as_i64() != Some(i64::from(p)) || p == r),
                "{what}: no walk is pinned to the parent's rows: {walk:?}"
            );
        }
    }
    for want in [1u64, 8, 32] {
        assert!(
            seen.contains(&want),
            "the cases reach {want} core(s): {cases:#?}"
        );
    }
}

/// A ONE-STICK band never crosses a column plane: no declaration, no gap — byte-identical to the
/// form every shipped one-stick band already emits.
#[test]
fn a_one_stick_band_states_no_gap() {
    let (nodes, _, _) = band_add(8, 2, 64);
    for n in &nodes {
        assert!(n.get("backGapCore_").is_none(), "{n}");
    }
}

/// ⭐ THE PITCH LAW ON ITS OWN, over any view: stated over the view's extents alone, one mechanism
/// per axis, and every near-miss refused by name at build.
#[test]
fn a_pitch_is_one_mechanism_per_axis_and_its_near_misses_refuse_by_name() {
    use ktir_superdsc::superdsc_opspec::{
        Allocation, CoreBudget, DeviceAxis, Df, ItDim, PastExtent, Pitch, Role, Scale, TensorArg,
        Window, WorkPlan,
    };
    // The pitch a view states is the one value `Pitch::past` builds — read back whole.
    let pitch = |swept, declared, stick| match Pitch::past(swept, declared, stick).unwrap() {
        PastExtent::Pitch(p) => p,
        PastExtent::AtExtent => panic!("{declared} is no pitch past {swept}"),
    };
    let dim = |name: &'static str, size: u32, is_stick: bool| ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    let plan = WorkPlan::divide(
        &[
            dim("y", 3, false),
            dim("mb", 4, false),
            dim("out", 128, true),
        ],
        CoreBudget::whole(),
        |_, _| Default::default(),
    )
    .unwrap();
    let view = || {
        TensorArg::<3>::new(
            true,
            "t".to_string(),
            Role::Output,
            [Scale::Active; 3],
            plan.iter_syms(["y", "mb", "out"]),
            ["y", "mb", "out"],
            "out",
            Allocation::Hbm,
        )
        .unwrap()
    };
    let axes = |t: TensorArg<3>| -> Vec<DeviceAxis> {
        ktir_superdsc::superdsc_opspec::AnyTensorArg::R3(t)
            .view()
            .device_extent
            .to_vec()
    };
    // Each axis's device statement goes through the ONE setter, over the axis's own typed index.
    let pitched = |t: TensorArg<3>, dim: &'static str, declared: u32| {
        let at = t.axis(dim).unwrap();
        at.pitch(&plan, declared).map(|p| t.with_axis(at, p))
    };
    // A pitch past the extent states it; a pitch AT the extent is no pitch.
    assert_eq!(
        axes(pitched(view(), "mb", 10).unwrap()),
        [
            DeviceAxis::Swept,
            DeviceAxis::Pitch(pitch(4, 10, None)),
            DeviceAxis::Swept
        ]
    );
    assert_eq!(
        axes(pitched(view(), "mb", 4).unwrap()),
        [DeviceAxis::Swept; 3]
    );
    // The near-miss the law exists for: a pitch SMALLER than the extent.
    let short = pitched(view(), "mb", 3).unwrap_err().0;
    assert!(short.contains("shorter than the extent"), "{short}");
    // The stick axis states a pitch of whole sticks, and refuses a gap inside one.
    assert_eq!(
        axes(pitched(view(), "out", 256).unwrap()),
        [
            DeviceAxis::Swept,
            DeviceAxis::Swept,
            DeviceAxis::Pitch(pitch(128, 256, Some(64)))
        ]
    );
    let stick = pitched(view(), "out", 160).unwrap_err().0;
    assert!(stick.contains("stick axis"), "{stick}");
    // A dim the view does not have has no index.
    let absent = view().axis("in").unwrap_err().0;
    assert!(absent.contains("not in layout"), "{absent}");
    // ONE AXIS, ONE STATEMENT: the axis's slot holds one `DeviceAxis`, so a later statement on the
    // same axis replaces the earlier one — there is no second mechanism left beside it to disagree.
    let v = view();
    let mb = v.axis("mb").unwrap();
    assert_eq!(
        axes(
            v.with_axis(mb, Window::cap(4, 32, 16).unwrap())
                .with_axis(mb, mb.pitch(&plan, 10).unwrap())
        ),
        [
            DeviceAxis::Swept,
            DeviceAxis::Pitch(pitch(4, 10, None)),
            DeviceAxis::Swept
        ]
    );
    // Different axes each state their own.
    let v = view();
    let (y, mb) = (v.axis("y").unwrap(), v.axis("mb").unwrap());
    assert_eq!(
        axes(
            v.with_axis(y, Window::cap(3, 16, 8).unwrap())
                .with_axis(mb, mb.pitch(&plan, 10).unwrap())
        ),
        [
            Window::cap(3, 16, 8).unwrap(),
            DeviceAxis::Pitch(pitch(4, 10, None)),
            DeviceAxis::Swept
        ]
    );
}

/// ⭐ THE WINDOW LAW ON ITS OWN: a cap is built ONCE by [`Window::cap`] from (swept, declared,
/// phys), fits its allocation, covers the sweep, and collapses to `Swept` at the sweep — every
/// near-miss refused by name at the mint.
#[test]
fn a_window_cap_is_built_once_and_its_near_misses_refuse_by_name() {
    use ktir_superdsc::superdsc_opspec::{DeviceAxis, Window};
    // The match: swept ≤ phys ≤ declared, and the reader takes the value back whole.
    let DeviceAxis::Window(w) = Window::cap(4, 32, 16).unwrap() else {
        panic!("a view of 16 past a sweep of 4, in an allocation of 32, is a window")
    };
    assert_eq!(w.phys(), 16);
    // The view may BE the allocation — the statement is the residency extent itself.
    let DeviceAxis::Window(w) = Window::cap(4, 16, 16).unwrap() else {
        panic!("a view of 16 that IS its allocation is a window, not a pitch")
    };
    assert_eq!(w.phys(), 16);
    // At the sweep the window IS the sweep — `Swept`, the same statement in its second
    // spelling (one fact, one source), proven byte-identical on card.
    assert_eq!(Window::cap(4, 32, 4).unwrap(), DeviceAxis::Swept);
    // A view that exceeds its allocation is out of bounds.
    let past = Window::cap(4, 16, 24).unwrap_err().0;
    assert!(past.contains("exceeds its allocation"), "{past}");
    // A cap that fails to cover the sweep: the walk would describe less memory than the op reads.
    let short = Window::cap(4, 32, 3).unwrap_err().0;
    assert!(short.contains("less memory than the op reads"), "{short}");
}

/// ⭐ THE PITCH IS BUILT ONCE, AND ONLY WHEN THE WALK CAN CARRY IT: [`Pitch::past`] over (swept,
/// declared, stick) is the one constructor, and its gap is the back gap verbatim.
#[test]
fn a_pitch_is_built_once_from_swept_declared_and_stick() {
    use ktir_superdsc::superdsc_opspec::{PastExtent, Pitch};
    // The match: the gap is declared − swept, the declared extent is what an address reads.
    let PastExtent::Pitch(p) = Pitch::past(4, 16, None).unwrap() else {
        panic!("16 past 4 is a pitch")
    };
    assert_eq!((p.declared(), p.gap()), (16, 12));
    // A whole-stick gap on a stick axis is a pitch.
    let PastExtent::Pitch(s) = Pitch::past(2048, 512 * 2048, Some(64)).unwrap() else {
        panic!("a whole-stick gap is a pitch")
    };
    assert_eq!(s.gap(), 511 * 2048);
    // AT the extent: no pitch, the axis is swept whole.
    assert_eq!(Pitch::past(4, 4, None).unwrap(), PastExtent::AtExtent);
    assert_eq!(
        Pitch::past(128, 128, Some(64)).unwrap(),
        PastExtent::AtExtent
    );
    // The near-misses, each refused by name at construction:
    let short = Pitch::past(16, 4, None).unwrap_err().0;
    assert!(short.contains("shorter than the extent"), "{short}");
    let sub_stick = Pitch::past(128, 160, Some(64)).unwrap_err().0;
    assert!(sub_stick.contains("whole 64-element sticks"), "{sub_stick}");
    // The stick only binds the stick axis: the same gap is a pitch on a row axis.
    assert!(matches!(
        Pitch::past(128, 160, None),
        Ok(PastExtent::Pitch(_))
    ));
}
