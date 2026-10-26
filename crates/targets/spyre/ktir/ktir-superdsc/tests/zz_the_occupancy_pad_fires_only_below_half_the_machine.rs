// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE FULL-OCCUPANCY PAD BUYS CORES WITH WEIGHT BYTES, SO IT FIRES ONLY WHERE THE CORES ARE WORTH IT.
//!
//! `bump_sticks_to_splittable` is the one width rule the emitter's device width, the layout's
//! placement sizing and the worker's weight zero-pad share. At decode every weight byte it adds is
//! streamed every step, so the pad to a whole multiple of the machine pays only while the unpadded
//! split leaves the gemm on less than half the cores. These are granite's real widths, pinned through
//! the public constructors the three consumers call.

use ktir_superdsc::superdsc_opspec::MAX_CORES;
use ktir_superdsc::work::{
    DeviceWidth, FP16_ELEMS_PER_STICK, Fp16Sticks, UTIL_FLOOR_CORES, bump_sticks_to_splittable,
    core_split,
};

const STICK: u32 = FP16_ELEMS_PER_STICK;

/// granite's vocab, 49155, rounds to 769 sticks: a prime, which splits onto ONE core. The pad to 800
/// sticks (+4%) puts it on all 32. This is the case the pad exists for, and it stays.
#[test]
fn the_lm_head_keeps_its_full_occupancy_pad() {
    let n64 = 49155u32.next_multiple_of(STICK);
    assert_eq!(n64 / STICK, 769);
    assert_eq!(core_split(769, MAX_CORES), 1);
    let padded = bump_sticks_to_splittable(Fp16Sticks::from_elems(n64));
    assert_eq!(padded, 800 * STICK);
    assert_eq!(core_split(padded / STICK, MAX_CORES), MAX_CORES);
    // The lm_head is a [1, 49155] x [49155, 4096]-class gemm far past the util-floor MAC count.
    assert_eq!(DeviceWidth::for_output(1, 49155, 4096).get(), 800 * STICK);
    assert_eq!(DeviceWidth::for_pointwise(49155).get(), 800 * STICK);
}

/// granite-3.1-8b's gate/up is 12800 wide: 200 sticks, already split onto 25 of 32 cores. Padding it to
/// 224 sticks was +12% on the model's largest weights for the last 7 cores; it stays at 12800.
#[test]
fn the_8b_gate_up_keeps_its_unpadded_split() {
    assert_eq!(12800 / STICK, 200);
    let cur = core_split(200, MAX_CORES);
    assert_eq!(cur, 25);
    assert!(
        cur * 2 >= MAX_CORES,
        "25 of 32 cores is past half the machine"
    );
    assert_eq!(bump_sticks_to_splittable(Fp16Sticks::from_elems(12800)), 12800);
    assert_eq!(DeviceWidth::for_output(1, 12800, 4096).get(), 12800);
}

/// Every other granite width (2b and 8b) already fills the machine or is too narrow for the pad's
/// 1/8 bound, so the rule leaves it alone.
#[test]
fn the_other_granite_widths_are_untouched() {
    for n in [512u32, 1024, 2048, 4096, 8192] {
        assert_eq!(
            bump_sticks_to_splittable(Fp16Sticks::from_elems(n)),
            n,
            "width {n}"
        );
    }
}

/// The law's properties over every stick count a weight can have, stated as outcomes rather than as a
/// copy of the rule:
/// * a split at or past half the machine is never padded;
/// * a pad never loses cores, and any pad lands on at least the util floor;
/// * a pad that reaches the whole machine was bought from below half of it, for at most 1/8 more bytes.
#[test]
fn the_pad_fires_only_below_half_the_machine() {
    for sticks in 1..=4096u32 {
        let cur = core_split(sticks, MAX_CORES);
        let got = bump_sticks_to_splittable(Fp16Sticks::from_elems(sticks * STICK)) / STICK;
        let after = core_split(got, MAX_CORES);
        if cur * 2 >= MAX_CORES {
            assert_eq!(got, sticks, "{sticks} sticks split onto {cur}: unpadded");
            continue;
        }
        assert!(after >= cur, "{sticks} sticks: the pad never loses cores");
        if got != sticks {
            assert!(
                after >= UTIL_FLOOR_CORES,
                "{sticks} sticks: a pad reaches the floor"
            );
        }
        if got != sticks && after == MAX_CORES && got > sticks.next_multiple_of(8) {
            assert!(
                (got - sticks) * 8 <= sticks,
                "{sticks} sticks: pad within 1/8"
            );
        }
    }
}
