// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE WINDOW CAP'S OWN LAW: a cap is built ONCE by [`Window::cap`] from (swept, declared,
//! phys), fits its allocation, covers the sweep, and collapses to `Swept` at the sweep — every
//! near-miss refused by name at the mint. What each refusal means and where it binds is the
//! constructor's contract (see `Window`'s doc in `superdsc_opspec`); this file pins it.

/// A cap is built once, fits its allocation, covers the sweep, and collapses at the sweep.
#[test]
fn a_window_cap_is_built_once_and_its_near_misses_refuse_by_name() {
    use ktir_superdsc::superdsc_opspec::{DeviceAxis, Window};
    // The match: swept ≤ phys ≤ declared, and the reader takes the value back whole.
    let DeviceAxis::Window(w) = Window::cap(4, 32, 16).unwrap() else {
        panic!("a view of 16 past a sweep of 4, in an allocation of 32, is a window")
    };
    assert_eq!(w.phys(), 16);
    assert_eq!(DeviceAxis::Window(w).declared(), Some(16));
    // The view may BE the allocation — the statement is the residency extent itself.
    let DeviceAxis::Window(w) = Window::cap(4, 16, 16).unwrap() else {
        panic!("a view of 16 that IS its allocation is a window")
    };
    assert_eq!(w.phys(), 16);
    // At the sweep the window IS the sweep — `Swept`, the same statement in its second
    // spelling (one fact, one source): every reader falls back to the op's own iteration
    // extent, which is the same number, so the emission is byte for byte what a
    // `Some(swept)` stated.
    assert_eq!(Window::cap(4, 32, 4).unwrap(), DeviceAxis::Swept);
    assert_eq!(DeviceAxis::Swept.declared(), None);
    // A view that exceeds its allocation is out of bounds.
    let past = Window::cap(4, 16, 24).unwrap_err().0;
    assert!(past.contains("exceeds its allocation"), "{past}");
    // A cap that fails to cover the sweep: the walk would describe less memory than the op
    // reads — the mis-declaration the `Option<u32>` this replaces accepted silently.
    let short = Window::cap(4, 32, 3).unwrap_err().0;
    assert!(short.contains("less memory than the op reads"), "{short}");
}

/// ONE AXIS, ONE STATEMENT: the axis's slot holds one `DeviceAxis`, so a later statement on the
/// same axis replaces the earlier one — there is no second mechanism left beside it to disagree.
#[test]
fn one_axis_holds_one_statement_and_the_absent_dim_refuses() {
    use ktir_superdsc::superdsc_opspec::{
        Allocation, AnyTensorArg, DeviceAxis, Df, ItDim, MaxCores, Role, Scale, TensorArg, Window,
        WorkPlan,
    };
    let dim = |name: &'static str, size: u32, is_stick: bool| ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    let plan = WorkPlan::divide(
        &[dim("mb", 4, false), dim("out", 128, true)],
        MaxCores::<32>,
        |_, _| Default::default(),
    )
    .unwrap();
    let t = TensorArg::<2>::new(
        true,
        "t".to_string(),
        Role::Output,
        [Scale::Active; 2],
        plan.iter_syms(["mb", "out"]),
        ["mb", "out"],
        "out",
        Allocation::Hbm,
    )
    .unwrap();
    let axes = |t: &TensorArg<2>| -> Vec<DeviceAxis> {
        AnyTensorArg::R2(t.clone()).view().device_extent.to_vec()
    };
    let mb = t.axis("mb").unwrap();
    let stated = t
        .with_axis(mb, Window::cap(4, 32, 16).unwrap())
        .unwrap()
        .with_axis(mb, Window::cap(4, 32, 8).unwrap())
        .unwrap();
    let restated = Window::cap(4, 32, 8).unwrap();
    assert_eq!(
        axes(&stated),
        [restated, DeviceAxis::Swept],
        "the later statement replaces the earlier one — one slot, one mechanism"
    );
    // A dim the view does not have has no index — refused by name, not panicked on.
    let absent = stated.axis("in").unwrap_err().0;
    assert!(absent.contains("not in layout"), "{absent}");
}
