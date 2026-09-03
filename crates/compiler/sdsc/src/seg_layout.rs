// SPDX-License-Identifier: Apache-2.0
//! SegLayout — the TWELFTH island: place a segment's tensors at SEQUENTIAL, 128-byte-aligned bases. The
//! full wire address of a tensor element is `base[t] + dev_off(...)` — `AddrIR` owns the intra-tensor
//! `dev_off`; this island owns the per-tensor `base`. Placement is `base[0]=0`, `base[i+1] =
//! align128(base[i] + footprint[i])`, where `footprint[t] = rows·pad64(cols)·2` (the device bytes AddrIR's
//! offsets stay within — Kani `dev_off_general_in_footprint`).
//!
//! ISLAND obligation (Kani, <2s): the placed ranges are NON-OVERLAPPING — for `i<j`, `base[i]+foot[i] ≤
//! base[j]` — so distinct tensors never clobber each other (inter-tensor safety), composing AddrIR's
//! intra-tensor injectivity into whole-segment injectivity. Only a CONCRETE `/128` (align), no symbolic ÷.

/// Round up to the 128-byte HBM boundary the SDSC allocator uses (`= 64 fp16`).
pub fn align128(x: u64) -> u64 {
    x.div_ceil(128) * 128
}

/// Sequential 128-aligned bases for `footprints` (device bytes per tensor). `bases[0] = 0`; each next base
/// is the previous end rounded up to 128. Monotonic ⇒ ranges are ordered and disjoint (proven below).
pub fn place(footprints: &[u64]) -> Vec<u64> {
    let mut bases = Vec::with_capacity(footprints.len());
    let mut cur = 0u64;
    for &f in footprints {
        bases.push(cur);
        cur = align128(cur + f);
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_is_disjoint_and_aligned() {
        let foot = [98432u64, 4096 * 49664 * 2, 8192, 128 * 2, 0, 776 * 64 * 2];
        let bases = place(&foot);
        for i in 0..foot.len() {
            assert_eq!(bases[i] % 128, 0, "base[{i}] not 128-aligned");
            for j in (i + 1)..foot.len() {
                assert!(bases[i] + foot[i] <= bases[j], "tensor {i} overlaps {j}");
            }
        }
    }
}

// ── Kani: placement disjoint + aligned, symbolic footprints, fixed small N (align128 = concrete /128). ──
#[cfg(kani)]
mod kani_seg {
    use super::*;

    /// `align128(x) ≥ x` and is a 128-multiple — the atom the whole placement rests on.
    #[kani::proof]
    fn align128_rounds_up() {
        let x: u64 = kani::any();
        kani::assume(x <= 1u64 << 42); // well within HBM; keeps the field small
        let a = align128(x);
        assert!(a >= x);
        assert!(a % 128 == 0);
        assert!(a < x + 128); // rounds up by less than one boundary
    }

    /// TRANSCRIPTION-GAP CLOSER: the on-card bundle packing (emitter `pack` closure, lower_subtile…:1339)
    /// advances each segment base with the EMITTER's OWN `align128` (bitmask `(n+127)&!127`), NOT this
    /// island's `div_ceil(128)*128`. The disjointness proof (`placement_disjoint_4`) rests on the latter,
    /// so prove the two are IDENTICAL on the realistic byte range ⇒ the disjointness transfers to the
    /// ACTUAL packing. A divergence (e.g. an emitter off-by-one in the mask) is caught here.
    #[kani::proof]
    fn align128_emitter_equals_model() {
        let x: u64 = kani::any();
        kani::assume(x <= 1u64 << 42); // realistic bundle bytes; (x+127) can't overflow u64
        let emitter = scratchy_target_spyre::lower_subtile_tape_to_superdsc::align128(x); // ACTUAL on-card fn
        assert!(emitter == align128(x)); // == this island's proven-disjoint round-up
    }

    /// A 4-tensor segment: any symbolic footprints place into NON-OVERLAPPING, 128-aligned ranges.
    #[kani::proof]
    fn placement_disjoint_4() {
        let f: [u64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let mut i = 0;
        while i < 4 {
            kani::assume(f[i] <= 1u64 << 40); // bounded footprint (≤ 1 TiB), sum can't overflow u64
            i += 1;
        }
        let bases = place(&f);
        // every tensor's range [base, base+foot) ends at or before the next tensor's base ⇒ disjoint.
        let mut a = 0;
        while a < 4 {
            assert!(bases[a] % 128 == 0);
            let mut b = a + 1;
            while b < 4 {
                assert!(bases[a] + f[a] <= bases[b]);
                b += 1;
            }
            a += 1;
        }
    }
}
