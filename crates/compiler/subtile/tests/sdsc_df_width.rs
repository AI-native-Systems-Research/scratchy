//! Rung-1 substrate guard (integration test — lives in `tests/` so it compiles via the `--test`
//! target, NOT the crate's lib unit-tests). It needs BOTH `spyre` (the activation switch,
//! matching the sibling `sdsc_*_layout` tests) and
//! `superdsc` (which gates `sdsc_abstract`/`superdsc_opspec`). Run:
//!   cargo test -p scratchy-subtile --features spyre,superdsc --test sdsc_df_width
//!
//! WHAT IT LOCKS (rung 1: thread `Df` into the live `StickLayout` so stick width is DERIVED from the
//! operand's device format). Stick width has no stored field — `lanes()` is `df.elems_per_stick()` —
//! so "stick width disagrees with format" is UNCONSTRUCTABLE (the compile-time guarantee). This test
//! pins the two behaviours the on-card gate rests on:
//!   (1) fp16 addressing is byte-identical to the old hardcoded-64 free `dev_off`  ⇒ zero bundle drift.
//!   (2) fp8 is a real 128-lane / 1-byte packed residency whose address actually differs ⇒ the width
//!       bit reaches addressing (real substrate for W8A8, not a cosmetic field).
//! Re-hardcoding the stick back to 64 makes the fp8 asserts go RED.
#![cfg(feature = "spyre")]

use scratchy_subtile::sdsc_abstract::{StickLayout, dev_off, dev_off_stk};
use scratchy_subtile::superdsc_opspec::Df;

#[test]
fn stick_width_is_derived_from_df() {
    // fp16 — the only format the emitter builds today. 64-lane stick; addressing unchanged.
    let k16 = StickLayout::kernel(2048, 512);
    assert_eq!(k16.df, Df::Fp16);
    assert_eq!(k16.lanes(), 64);
    for &(r, c) in &[(0usize, 0usize), (1, 63), (1, 64), (3, 511), (7, 128)] {
        assert_eq!(
            k16.dev_off(r, c),
            dev_off(&[2048, 512], 1, &[r, c]),
            "fp16 dev_off must be byte-identical to the const-STK free fn"
        );
    }

    // fp8 — the substrate this rung adds: 128-lane / 1-byte packed weight residency, now EXPRESSIBLE
    // in the live layout. `lanes()` is 128 and `dev_off` tiles on 128, not 64.
    let k8 = StickLayout::kernel_df(2048, 512, Df::Fp8);
    assert_eq!(k8.lanes(), 128);
    assert_eq!(
        k8.dev_off(1, 128),
        dev_off_stk(&[2048, 512], 1, &[1, 128], 128)
    );
    assert_ne!(
        k8.dev_off(1, 128),
        k16.dev_off(1, 128),
        "the fp8 stick width must reach addressing (else the field is cosmetic)"
    );
}

/// The disk-order re-tile map must land every element where the transposed one does — same gather,
/// read from the orientation safetensors stores rather than from a transposed copy. If these ever
/// disagree, the transpose is load-bearing after all and eliminating it corrupts weights.
#[test]
fn disk_order_stride_map_gathers_the_same_elements() {
    use scratchy_subtile::superdsc_opspec::{DeviceTileLayout, Fp16};
    let (in_elems, out_elems) = (512usize, 256usize);
    let t =
        DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[in_elems as u64, out_elems as u64])
            .expect("2-D stick-last kernel tile");
    let (dev, tr, dk) = (t.device_size(), t.stride_map(), t.stride_map_disk_order());
    let stk = 64usize;
    for tt in 0..dev[0] as usize {
        for i in 0..dev[1] as usize {
            for s in [0usize, 1, stk - 1] {
                // logical element (in = i, out = tt*stk + s)
                let o = tt * stk + s;
                let via_transposed = tt * tr[0] as usize + i * tr[1] as usize + s * tr[2] as usize;
                let via_disk = tt * dk[0] as usize + i * dk[1] as usize + s * dk[2] as usize;
                assert_eq!(
                    via_transposed,
                    i * out_elems + o,
                    "transposed map reads [in,out]"
                );
                assert_eq!(
                    via_disk,
                    o * in_elems + i,
                    "disk map reads [out,in] — the SAME element"
                );
            }
        }
    }
}
