// SPDX-License-Identifier: Apache-2.0
//! Device-tile weight staging — the pure walk that lays a host weight out in the device TILE
//! layout the PT systolic array reads, lifted out of C++ (`sdsc_stage_weight_tiled`).
//!
//! The walk is DERIVED ENTIRELY from the `RetileDescriptor` the emitter wrote into
//! `bundle_layout.json` `kernel_weights` (itself emitted solely from the Rust `DeviceTileLayout`
//! witness — the same one the per-core address derives from, so producer and consumer cannot
//! diverge). Iterate device coordinates in contiguous `device_size` order (`lin` = device offset);
//! host offset = Σ coord·stride_map. For a `[in,out]` KERNEL: device_size `[out/64, in, 64]`,
//! stride_map `[64, out, 1]` → device(t,i,s) = host(i, t·64+s), the contiguous out-stick-tile
//! layout. No hardcoded geometry.
//!
//! 🛑 THIS IS A PORT, so it reproduces the original walk exactly, including the run-length fast path:
//! when the INNERMOST device axis is contiguous in the source (`stride_map[rank-1] == 1`), a whole
//! row is one contiguous source run — the offset is computed once per ROW and the row is a
//! `memcpy`/row-convert, so the odometer turns over `total/inner` times instead of `total`.

use crate::sen_convert::{IeeeF16, ieee_to_sen};

/// The element width of one staged weight, decoded ONCE at the FFI boundary.
///
/// - [`Element::Fp8`] (word_length 1): the on-disk fp8 code (E4M3) IS the device format
///   (SEN143_FP8, same bit layout), so the tile walk is a VERBATIM 1-byte gather — no convert.
///   This is the real 1-byte HBM residency (the decode bandwidth win).
/// - [`Element::F16`] (word_length 2): each element converts IEEE-fp16 → SEN169 on the way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Element {
    Fp8,
    F16,
}

impl Element {
    /// Decode the manifest's `word_length`. Anything but 1 or 2 is a manifest the walk was never
    /// written for — refuse at the boundary, don't guess.
    pub fn from_word_length(word_length: u32) -> Option<Element> {
        match word_length {
            1 => Some(Element::Fp8),
            2 => Some(Element::F16),
            _ => None,
        }
    }
}

/// One weight's device-tile walk: (device extent, host-element stride) per device axis —
/// ONE slice, because an axis's extent and its stride are one fact, and two separate lists
/// could disagree in length. A slice, not an owned Vec — the caller owns the descriptor (at
/// runtime that is the bake's [`crate::bundle_code::KernelWeight::axes`]; the C++ shim consumes
/// the serialized `RetileDescriptor`, a separate artifact).
#[derive(Clone, Copy, Debug)]
pub struct RetileWalk<'a> {
    pub axes: &'a [(u64, u64)],
}

/// The staging-arm verdict, stated per weight by the bake — see [`crate::bundle_code::WalkArm`].
/// Re-exported here so the walker's law and the artifact's field name one type.
pub use crate::bundle_code::WalkArm;

impl<'a> RetileWalk<'a> {
    /// Total device elements = prod(extents).
    pub fn total_elems(&self) -> usize {
        self.axes.iter().map(|&(e, _)| e).product::<u64>() as usize
    }

    fn inner(&self) -> usize {
        self.axes.last().map(|&(e, _)| e).unwrap_or(1) as usize
    }

    /// The walk's own contiguity statement: the innermost device axis is a run of consecutive
    /// host elements (innermost stride 1, extent > 1), so the walk can copy whole rows.
    pub(crate) fn inner_contig(&self) -> bool {
        let inner = self.inner();
        !self.axes.is_empty() && self.axes.last().map(|&(_, s)| s) == Some(1) && inner > 1
    }

    /// Host element offset for the device coordinate `coord` (dot with the strides, excluding
    /// the last `skip_tail` axes — the fast path computes per-row offsets with the innermost
    /// axis at 0).
    fn host_off(&self, coord: &[u64], skip_tail: usize) -> usize {
        coord
            .iter()
            .zip(self.axes.iter().map(|&(_, s)| s))
            .take(coord.len() - skip_tail)
            .map(|(&c, s)| c * s)
            .sum::<u64>() as usize
    }

    /// Odometer increment over the leading `rank - skip_tail` axes; the C++ walk's carry loop.
    fn advance(&self, coord: &mut [u64], skip_tail: usize) {
        for r in (0..coord.len() - skip_tail).rev() {
            coord[r] += 1;
            if coord[r] < self.axes[r].0 {
                break;
            }
            coord[r] = 0;
        }
    }

    /// The walk's axes in HOST-SCAN order (ascending stride), when the stride map is a
    /// mixed-radix PERMUTATION: each axis's stride equals the product of the extents of every
    /// smaller-stride axis. Positional notation then makes the host offsets EXACTLY `0..total`
    /// in that order — the whole tensor is one linear host scan, and only the device side
    /// needs coordinate math (incremental; no per-element dot product, no per-row odometer
    /// over the full rank). This is the boot walks' own law, not a special case: the measured
    /// table (`SCRATCHY_SDSC_STAGE_TIME`, 2b fp8) shows every slow walk — the 280 fp8 items
    /// at 0.33 GB/s/thread whose "row copy" is TWO BYTES (`[128,1024,64,2]` /
    /// `[131072,2,2048,1]`) and the 200 MiB f16 embedding walked per-element
    /// (`[800,2048,64]` / `[131072,1,2048]`) — satisfies strides==products exactly. A walk
    /// with gather/broadcast structure (repeated or gapped strides) is NOT a permutation and
    /// takes the arms below.
    pub fn permutation(&self) -> Option<Vec<usize>> {
        if self.axes.is_empty() {
            return None; // rank-0: one element, host[0] → dev[0]; the per-element arm's case
        }
        let mut order: Vec<usize> = (0..self.axes.len()).collect();
        order.sort_by_key(|&i| self.axes[i].1);
        let mut run = 1u64;
        for &ax in &order {
            if self.axes[ax].1 != run {
                return None;
            }
            run = run.saturating_mul(self.axes[ax].0);
        }
        Some(order)
    }

    /// Which arm stages this walk — ONE statement: the bake reads it to state
    /// [`crate::bundle_code::KernelWeight::arm`], and `stage_weight_tiled`'s own branch follows it
    /// for callers that hold a walk but not a bundle.
    pub fn arm(&self) -> WalkArm {
        if self.permutation().is_some() {
            WalkArm::Permutation
        } else if self.inner_contig() {
            WalkArm::InnerContig
        } else {
            WalkArm::PerElement
        }
    }
}

/// Increment the odometer over the OUTER axes (host-scan order, `order[1..]`), adjusting the
/// device offset INCREMENTALLY: `dev_stride[ax]` per step, minus the axis's whole span on
/// carry. Returns false when every axis has wrapped — the walk is complete. `d` is always
/// the device offset of the current outer coordinate at inner coordinate 0, so the caller's
/// block/stride loop can key off it directly.
fn advance_outer(
    coord: &mut [u64],
    d: &mut usize,
    order: &[u8],
    dev_stride: &[u64],
    axes: &[(u64, u64)],
) -> bool {
    for &ax in order.iter().skip(1) {
        let ax = ax as usize;
        coord[ax] += 1;
        *d += dev_stride[ax] as usize;
        if coord[ax] < axes[ax].0 {
            return true;
        }
        coord[ax] = 0;
        *d -= dev_stride[ax] as usize * axes[ax].0 as usize;
    }
    false
}

/// Fixtures and harnesses spell walks the way the emitter does — an extent list, then a
/// stride list — and this pairs them into the walk's one slice.
#[cfg(any(test, kani))]
fn axes_of(ds: &[u64], sm: &[u64]) -> Vec<(u64, u64)> {
    ds.iter().zip(sm).map(|(&d, &s)| (d, s)).collect()
}

/// Stage one weight: read `host` (the on-disk layout, IEEE-fp16 or fp8 bytes) and write `dev`
/// (the device tile layout, SEN bytes) via the descriptor's pure walk.
///
/// `host` and `dev` are in ELEMENTS of `elem` width: both must hold `walk.total_elems()`
/// elements (the caller has already validated the host size against prod(device_size), which for
/// a permutation walk bounds every host offset).
pub fn stage_weight_tiled(host: &[u8], dev: &mut [u8], walk: RetileWalk<'_>, elem: Element) {
    // ⭐ A MIXED-RADIX PERMUTATION ([`RetileWalk::permutation`]) IS A LINEAR HOST SCAN — the
    // host offsets are exactly 0..total in ascending-stride order, so the walk never needs a
    // dot product or a full-rank odometer: read the host linearly, write the device at offsets
    // the outer odometer carries incrementally. This is the arm the 2b's boot weights live on
    // (measured: the fp8 packed-tile walk's "row copy" was TWO BYTES — a dot product and a
    // 4-axis odometer per 2 bytes, 0.33 GB/s per thread).
    if let Some(order) = walk.permutation() {
        assert!(
            order.len() <= u8::MAX as usize,
            "stage_weight_tiled: a rank-{} walk exceeds the axis order's u8",
            order.len()
        );
        let order: Vec<u8> = order.iter().map(|&i| i as u8).collect();
        stage_weight_permuted(host, dev, walk.axes, &order, elem);
        return;
    }
    let rank = walk.axes.len();
    let total = walk.total_elems();
    let inner = walk.inner();
    let inner_contig = walk.inner_contig();
    let mut coord = vec![0u64; rank];

    match elem {
        Element::Fp8 => {
            if inner_contig {
                let mut lin = 0;
                while lin < total {
                    let hoff = walk.host_off(&coord, 1);
                    dev[lin..lin + inner].copy_from_slice(&host[hoff..hoff + inner]);
                    walk.advance(&mut coord, 1);
                    lin += inner;
                }
            } else {
                for d in dev.iter_mut().take(total) {
                    let hoff = walk.host_off(&coord, 0);
                    *d = host[hoff];
                    walk.advance(&mut coord, 0);
                }
            }
        }
        Element::F16 => {
            // Reinterpret both buffers as 16-bit elements (the FFI hands byte pointers; the safe
            // callers hand properly-sized slices).
            let h: Vec<u16> = host
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            let conv = |x: u16| ieee_to_sen(IeeeF16(x)).0;
            let mut write = |lin: usize, v: u16| {
                dev[lin * 2..lin * 2 + 2].copy_from_slice(&v.to_le_bytes());
            };
            if inner_contig {
                let mut lin = 0;
                while lin < total {
                    let hoff = walk.host_off(&coord, 1);
                    for j in 0..inner {
                        write(lin + j, conv(h[hoff + j]));
                    }
                    walk.advance(&mut coord, 1);
                    lin += inner;
                }
            } else {
                for lin in 0..total {
                    let hoff = walk.host_off(&coord, 0);
                    write(lin, conv(h[hoff]));
                    walk.advance(&mut coord, 0);
                }
            }
        }
    }
}

/// The permutation walk ([`RetileWalk::permutation`]): the host is ONE linear scan (offsets
/// `0..total` in `order`'s mixed-radix order), and the device offset is carried incrementally
/// by [`advance_outer`] — `dev_stride[ax]` per outer step, the axis's whole span subtracted on
/// carry. The host-innermost axis (`order[0]`) is the inner run: when its device stride is 1
/// the run is contiguous on BOTH sides and copies/converts as one block; otherwise the run's
/// device positions step by `dev_stride[order[0]]` and the loop writes them element-wise —
/// still no dot product, still no full-rank odometer.
///
/// `order` is the bake's stated host-scan axis order ([`crate::bundle_code::KernelWeight::order`],
/// u8 axis indices); `stage_weight_tiled` derives it itself for callers without a bundle.
pub(crate) fn stage_weight_permuted(
    host: &[u8],
    dev: &mut [u8],
    axes: &[(u64, u64)],
    order: &[u8],
    elem: Element,
) {
    let rank = axes.len();
    // Device row-major stride per axis: the last axis is 1, each earlier axis the product of
    // the extents after it.
    let mut dev_stride = vec![1u64; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dev_stride[i] = dev_stride[i + 1] * axes[i + 1].0;
    }
    let inner = order[0] as usize;
    let run = axes[inner].0 as usize;
    let block = dev_stride[inner] == 1;
    let dstep = dev_stride[inner] as usize;
    let mut coord = vec![0u64; rank];
    let mut h = 0usize; // host ELEMENT offset — linear
    let mut d = 0usize; // device ELEMENT offset — incremental, base of the inner run
    loop {
        if block {
            match elem {
                Element::Fp8 => dev[d..d + run].copy_from_slice(&host[h..h + run]),
                Element::F16 => {
                    for j in 0..run {
                        let x = u16::from_le_bytes([host[(h + j) * 2], host[(h + j) * 2 + 1]]);
                        let v = ieee_to_sen(IeeeF16(x)).0;
                        dev[(d + j) * 2..(d + j) * 2 + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        } else {
            match elem {
                Element::Fp8 => {
                    for j in 0..run {
                        dev[d + j * dstep] = host[h + j];
                    }
                }
                Element::F16 => {
                    for j in 0..run {
                        let x = u16::from_le_bytes([host[(h + j) * 2], host[(h + j) * 2 + 1]]);
                        let v = ieee_to_sen(IeeeF16(x)).0;
                        let at = (d + j * dstep) * 2;
                        dev[at..at + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        h += run;
        if !advance_outer(&mut coord, &mut d, order, &dev_stride, axes) {
            break;
        }
    }
}

// ── Kani proofs: the walk over SYMBOLIC shapes/strides, which the fixed-shape unit tests below
//    cannot pin. (`sen_convert` needs no Kani — its domain is 2^16 and the unit tests enumerate
//    ALL of it, which is complete verification.) Run with `cargo kani -p scratchy-target-spyre`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Reference walk: per-element dot-product offset, no fast path — definitionally the layout.
    fn naive(host: &[u8], walk: RetileWalk<'_>, elem: Element) -> Vec<u8> {
        let wl = match elem {
            Element::Fp8 => 1,
            Element::F16 => 2,
        };
        let total = walk.total_elems();
        let mut dev = vec![0u8; total * wl];
        let rank = walk.device_size.len();
        let mut coord = vec![0u64; rank];
        for lin in 0..total {
            let hoff: usize = coord
                .iter()
                .zip(walk.axes.iter().map(|&(_, s)| s))
                .map(|(&c, &s)| c * s)
                .sum::<u64>() as usize;
            match elem {
                Element::Fp8 => dev[lin] = host[hoff],
                Element::F16 => {
                    let x = u16::from_le_bytes([host[hoff * 2], host[hoff * 2 + 1]]);
                    let v = crate::sen_convert::ieee_to_sen(crate::sen_convert::IeeeF16(x)).0;
                    dev[lin * 2..lin * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            for r in (0..rank).rev() {
                coord[r] += 1;
                if coord[r] < walk.device_size[r] {
                    break;
                }
                coord[r] = 0;
            }
        }
        dev
    }

    /// The run-length fast path (`stride_map[last]==1` ⇒ per-ROW offset + contiguous run) produces
    /// BYTE-IDENTICAL output to the per-element definition, for every rank-3 shape/stride within
    /// bounds — including shapes that don't take the fast path at all, and with no OOB access
    /// (an OOB index would be a Kani failure). fp8 exercises the memcpy run.
    #[kani::proof]
    #[kani::unwind(30)]
    fn fast_path_equals_naive_fp8() {
        const R: usize = 3;
        let mut ds = [0u64; R];
        let mut sm = [0u64; R];
        for r in 0..R {
            let d: u64 = kani::any();
            kani::assume((1..=3).contains(&d));
            ds[r] = d;
            let s: u64 = kani::any();
            kani::assume((0..=4).contains(&s));
            sm[r] = s;
        }
        // Host must cover the largest reachable offset: Σ (dim-1)·stride + 1.
        let max_off: u64 = (0..R).map(|r| (ds[r] - 1) * sm[r]).sum::<u64>() + 1;
        let host: Vec<u8> = (0..max_off as u8).collect();
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        let mut dev = vec![0u8; walk.total_elems()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8));
    }

    /// Same equivalence for the f16 branch (row-convert fast path vs per-element convert), over
    /// symbolic rank-2 shapes and symbolic element bits.
    #[kani::proof]
    #[kani::unwind(12)]
    fn fast_path_equals_naive_f16() {
        const R: usize = 2;
        let mut ds = [0u64; R];
        let mut sm = [0u64; R];
        for r in 0..R {
            let d: u64 = kani::any();
            kani::assume((1..=2).contains(&d));
            ds[r] = d;
            let s: u64 = kani::any();
            kani::assume((0..=3).contains(&s));
            sm[r] = s;
        }
        let max_off: u64 = (0..R).map(|r| (ds[r] - 1) * sm[r]).sum::<u64>() + 1;
        let mut host = vec![0u8; (max_off as usize) * 2];
        for b in host.iter_mut() {
            *b = kani::any();
        }
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        let mut dev = vec![0u8; walk.total_elems() * 2];
        stage_weight_tiled(&host, &mut dev, walk, Element::F16);
        assert_eq!(dev, naive(&host, walk, Element::F16));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference walk: per-element, no fast path — what the C++ non-contig branch does.
    fn naive(host: &[u8], walk: RetileWalk<'_>, elem: Element) -> Vec<u8> {
        let wl = match elem {
            Element::Fp8 => 1,
            Element::F16 => 2,
        };
        let total = walk.total_elems();
        let mut dev = vec![0u8; total * wl];
        let rank = walk.axes.len();
        let mut coord = vec![0u64; rank];
        for lin in 0..total {
            let hoff: usize = coord
                .iter()
                .zip(walk.axes.iter().map(|&(_, s)| s))
                .map(|(&c, s)| c * s)
                .sum::<u64>() as usize;
            match elem {
                Element::Fp8 => dev[lin] = host[hoff],
                Element::F16 => {
                    let x = u16::from_le_bytes([host[hoff * 2], host[hoff * 2 + 1]]);
                    let v = ieee_to_sen(IeeeF16(x)).0;
                    dev[lin * 2..lin * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            for r in (0..rank).rev() {
                coord[r] += 1;
                if coord[r] < walk.axes[r].0 {
                    break;
                }
                coord[r] = 0;
            }
        }
        dev
    }

    /// The kernel layout the doc comment names: `[in, out] = [4, 8]` host, device
    /// `[out/64→2, in→4, 64→4]` with stride_map `[4, 8, 1]` (stick 4 to keep the test small) —
    /// device(t,i,s) = host(i, t·4+s). Fast path (innermost stride 1) must equal the naive walk.
    #[test]
    fn f16_tile_matches_naive() {
        let (inn, out, stick) = (4u64, 8u64, 4u64);
        let host: Vec<u8> = (0..(inn * out) as u16)
            .flat_map(|i| (0x3C00 + i).to_le_bytes()) // distinct finite fp16 values
            .collect();
        let ds = [out / stick, inn, stick];
        let sm = [stick, out, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::F16);
        assert_eq!(dev, naive(&host, walk, Element::F16));
        // spot-check the permutation itself: device(t=1, i=2, s=3) = host(i=2, o=1·4+3=7)
        let lin = ((inn + 2) * stick + 3) as usize;
        let hoff = (2 * out + 7) as usize;
        let want = ieee_to_sen(IeeeF16(0x3C00 + hoff as u16)).0;
        assert_eq!(u16::from_le_bytes([dev[lin * 2], dev[lin * 2 + 1]]), want);
    }

    /// ⭐ THE PERMUTATION SCAN — the 2b boot walks' own law, pinned at the measured shape's
    /// structure: the fp8 packed-tile walk `[n/64, k/2, 64, 2]` / `[64k, 2, k, 1]` whose
    /// "row copy" was TWO BYTES (a dot product and a 4-axis odometer per 2 bytes, 0.33 GB/s
    /// per thread, 280 items / 2320 MiB on the card). Strides==products exactly ⇒ the host is
    /// one linear scan; the scan must equal the naive walk. Scaled to keep the test fast —
    /// the stride law is the claim, not the magnitude.
    #[test]
    fn permutation_scan_matches_naive_fp8() {
        let ds = [4u64, 8, 4, 2];
        let sm = [64u64, 2, 16, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        assert_eq!(walk.arm(), WalkArm::Permutation);
        // Distinct bytes: identity is host offset — a wrong law reads a different byte.
        let host: Vec<u8> = (0..ds.iter().product::<u64>() as usize)
            .map(|x| (x % 251) as u8)
            .collect();
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8));
    }

    /// The STRIDED permutation arm — the f16 embedding's shape (`t726`, 200 MiB walked
    /// per-element on the card): `[800, 2048, 64]` / `[131072, 1, 2048]`, where the
    /// host-innermost axis (stride 1, extent 2048) has DEVICE stride 64, so the scan writes
    /// a strided run. Scaled; both element widths.
    #[test]
    fn strided_permutation_scan_matches_naive() {
        let ds = [3u64, 4, 2];
        let sm = [8u64, 1, 4];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        assert_eq!(walk.arm(), WalkArm::Permutation);
        // device stride of the host-innermost axis (axis 1) is 2 ≠ 1 ⇒ the strided arm.
        let host8: Vec<u8> = (0..24u8).collect();
        let mut dev8 = vec![0u8; 24];
        stage_weight_tiled(&host8, &mut dev8, walk, Element::Fp8);
        assert_eq!(dev8, naive(&host8, walk, Element::Fp8));

        let host16: Vec<u8> = (0..24u16)
            .flat_map(|i| (0x4000 + i).to_le_bytes())
            .collect();
        let mut dev16 = vec![0u8; 48];
        stage_weight_tiled(&host16, &mut dev16, walk, Element::F16);
        assert_eq!(dev16, naive(&host16, walk, Element::F16));
    }

    /// The CONTIGUOUS-BLOCK f16 arm — the scan's other half: the host-innermost axis (stride 1,
    /// extent 2) is also the DEVICE-LAST axis (`dev_stride == 1`), so each inner run is contiguous
    /// on both sides and converts as one block. The fp8 packed tile's shape with f16 elements:
    /// no granite-2b walk lands here (measured: its f16 permutation items are all the strided
    /// arm — the shapes are in the stage-time table), so this fixture is the arm's coverage.
    #[test]
    fn contiguous_block_permutation_scan_matches_naive_f16() {
        let ds = [4u64, 8, 4, 2];
        let sm = [64u64, 2, 16, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        assert_eq!(walk.arm(), WalkArm::Permutation);
        // Distinct finite fp16 values: the convert is in the claim, so every element must
        // round-trip through ieee_to_sen exactly as the naive walk converts it.
        let host: Vec<u8> = (0..ds.iter().product::<u64>() as u16)
            .flat_map(|i| (0x3C00 + i).to_le_bytes())
            .collect();
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::F16);
        assert_eq!(dev, naive(&host, walk, Element::F16));
    }

    /// A NEAR-MISS is LEFT UNCLAIMED: one stride off the mixed-radix law and the walk is not
    /// a permutation — it must fall to the older arms (here the inner-contig row-copy, and
    /// with a non-1 tail stride the per-element walk) and still equal the naive walk. A walk
    /// with gather/broadcast structure can never take the scan.
    #[test]
    fn a_near_miss_permutation_falls_to_the_older_arms() {
        let ds = [4u64, 8, 4, 2];
        // Off by one on axis 2: 17 where the mixed-radix law says 16.
        let miss_sm = [64u64, 2, 17, 1];
        let miss_axes = axes_of(&ds, &miss_sm);
        let miss = RetileWalk { axes: &miss_axes };
        assert_eq!(miss.arm(), WalkArm::InnerContig);
        // A broadcast stride (two axes at stride 1) is not a permutation either.
        let bcast_axes = axes_of(&[4u64, 8, 4], &[32u64, 1, 1]);
        let bcast = RetileWalk { axes: &bcast_axes };
        assert_eq!(bcast.arm(), WalkArm::InnerContig);
        // A non-1 innermost stride: per-element.
        let tail_axes = axes_of(&[4u64, 8, 4, 2], &[64u64, 2, 16, 3]);
        let tail = RetileWalk { axes: &tail_axes };
        assert_eq!(tail.arm(), WalkArm::PerElement);
        // And each of them still stages exactly the naive walk. The host stays whole: a
        // near-miss walk can reach offsets past total (that is exactly what makes it not a
        // permutation), so neither the walker nor the naive reference may be handed a slice.
        let host: Vec<u8> = (0..1024usize).map(|x| (x % 251) as u8).collect();
        for walk in [&miss, &bcast, &tail] {
            let total = walk.total_elems();
            let mut dev = vec![0u8; total];
            stage_weight_tiled(&host, &mut dev, *walk, Element::Fp8);
            assert_eq!(dev, naive(&host, *walk, Element::Fp8));
        }
    }

    /// fp8 is a VERBATIM byte gather — no conversion.
    #[test]
    fn fp8_gather_matches_naive_and_converts_nothing() {
        let host: Vec<u8> = (0..128u8).collect();
        let ds = [2u64, 4, 16];
        let sm = [16u64, 32, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8));
        let mut sorted = dev.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted, host,
            "gather must be a permutation of the source bytes"
        );
    }

    /// A non-contiguous innermost axis takes the per-element branch; both element widths.
    #[test]
    fn non_contig_inner_axis() {
        let ds = [4u64, 4];
        let sm = [1u64, 4]; // innermost stride 4 ⇒ no fast path
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        let host8: Vec<u8> = (0..16u8).collect();
        let mut dev8 = vec![0u8; 16];
        stage_weight_tiled(&host8, &mut dev8, walk, Element::Fp8);
        assert_eq!(dev8, naive(&host8, walk, Element::Fp8));

        let host16: Vec<u8> = (0..16u16)
            .flat_map(|i| (0x4000 + i).to_le_bytes())
            .collect();
        let mut dev16 = vec![0u8; 32];
        stage_weight_tiled(&host16, &mut dev16, walk, Element::F16);
        assert_eq!(dev16, naive(&host16, walk, Element::F16));
    }

    /// ⭐⭐⭐⭐⭐ THE HEAD-MAJOR PERMUTED WALK — the o_proj copy's staging law, pinned against an
    /// INDEPENDENT re-derivation of the feature order. The contract is stated from the MODEL's
    /// semantics, not from the emitter's stride map (which is what these literals would otherwise
    /// just mirror):
    ///
    /// the attention finalize writes `out` SLAB-MAJOR — `out[s·nqh·64 + h·64 + d]` holds head
    /// `h`'s feature `s·64 + d` — so the permuted o_proj copy must put, at every device
    /// contraction position `k`, the weight column of exactly that feature. The kernel reads the
    /// staged buffer through its unchanged `[n/64, k/2, 64, 2]` view, which is itself half the
    /// claim: the restructure moves bytes at STAGING only.
    ///
    /// Geometry: granite-3.1-8b's head dim (hd=128, two feature slabs) scaled to nqh=2 — the
    /// multi-slab head dim is the whole point, so hd stays real.
    #[test]
    fn headmajor_permuted_walk_meets_the_slab_major_order() {
        const HD: u64 = 128;
        const NQH: u64 = 2;
        const K: u64 = NQH * HD; // 256
        const N: u64 = 128;
        // A row-AND-feature-dependent byte pattern: a wrong law reads a different host offset,
        // and with 32k comparisons only a collision-everywhere law could still pass.
        let host: Vec<u8> = (0..N * K)
            .map(|i| ((i / K) * 131 + (i % K) * 197) as u8)
            .collect();
        // The two walks, VERBATIM as the emitter writes them: the fp8 packed tile, and its
        // head-major refinement. `32` is POOL_STICK/2 — the k-pairs in one feature slab.
        let norm_ds = [N / 64, K / 2, 64, 2];
        let norm_sm = [64 * K, 2, K, 1];
        let perm_ds = [N / 64, HD / 64, NQH, 32, 64, 2];
        let perm_sm = [64 * K, 64, HD, 2, K, 1];
        let norm_axes = axes_of(&norm_ds, &norm_sm);
        let norm = RetileWalk { axes: &norm_axes };
        let perm_axes = axes_of(&perm_ds, &perm_sm);
        let perm = RetileWalk { axes: &perm_axes };
        assert_eq!(norm.total_elems(), perm.total_elems());
        let mut norm_dev = vec![0u8; norm.total_elems()];
        let mut perm_dev = vec![0u8; perm.total_elems()];
        stage_weight_tiled(&host, &mut norm_dev, norm, Element::Fp8);
        stage_weight_tiled(&host, &mut perm_dev, perm, Element::Fp8);

        // The KERNEL-VIEW linear offset — the same addressing for both buffers, which is the
        // "the device program is untouched" half of the claim.
        let lin = |n_stick: u64, k: u64, n_in: u64| -> usize {
            (((n_stick * (K / 2) + k / 2) * 64 + n_in) * 2 + k % 2) as usize
        };
        // The slab-major feature each device k position owns, derived from the MODEL's order:
        // k = s·nqh·64 + h·64 + d is head h's feature s·64 + d, i.e. host feature h·hd + s·64 + d.
        let pi = |k: u64| -> u64 {
            let s = k / (NQH * 64);
            let rem = k % (NQH * 64);
            let h = rem / 64;
            let d = rem % 64;
            h * HD + s * 64 + d
        };
        for n_stick in 0..N / 64 {
            for k in 0..K {
                for n_in in 0..64u64 {
                    let row = 64 * n_stick + n_in;
                    let at = lin(n_stick, k, n_in);
                    assert_eq!(
                        perm_dev[at],
                        host[(row * K + pi(k)) as usize],
                        "perm: device k={k} (row {row}) must hold head-major feature {}'s column",
                        pi(k)
                    );
                    assert_eq!(
                        norm_dev[at],
                        host[(row * K + k) as usize],
                        "norm: device k={k} (row {row}) must hold feature {k}'s column"
                    );
                }
            }
        }
        // ⛔ THE CONTRACTION IDENTITY — the restructure's whole claim as one number: o_proj's
        // output for a row is UNCHANGED. `a_slab` is what the finalize writes (the token-stream
        // activation of feature pi(k), relocated); against the permuted walk it must contract to
        // the same value the token-stream activation contracts to against the normal walk.
        let a_tok: Vec<u32> = (0..K).map(|f| (f * 73 % 251) as u32).collect();
        let a_slab: Vec<u32> = (0..K).map(|k| a_tok[pi(k) as usize]).collect();
        let row = 17u64;
        let (n_stick, n_in) = (row / 64, row % 64);
        let (mut via_perm, mut via_norm) = (0u64, 0u64);
        for k in 0..K {
            let at = lin(n_stick, k, n_in);
            via_perm += u64::from(a_slab[k as usize]) * u64::from(perm_dev[at]);
            via_norm += u64::from(a_tok[k as usize]) * u64::from(norm_dev[at]);
        }
        assert_eq!(
            via_perm, via_norm,
            "the slab-major contraction must equal the token-stream one"
        );
    }

    /// ⭐ AT ONE SLAB THE PERMUTED WALK **IS** THE FP8 PACKED WALK, byte for byte — the property
    /// that makes the mint's multi-slab gate a gate and not a behavior change: an hd=64 model's
    /// o_proj stages identically through either descriptor, so a copy minted by mistake (or a
    /// future caller at one slab) cannot corrupt anything.
    #[test]
    fn headmajor_permuted_walk_degenerates_to_the_fp8_tile_at_one_slab() {
        const HD: u64 = 64;
        const NQH: u64 = 4;
        const K: u64 = NQH * HD; // 256
        const N: u64 = 128;
        let host: Vec<u8> = (0..N * K).map(|i| (i * 37 % 251) as u8).collect();
        let norm_ds = [N / 64, K / 2, 64, 2];
        let norm_sm = [64 * K, 2, K, 1];
        let perm_ds = [N / 64, HD / 64, NQH, 32, 64, 2];
        let perm_sm = [64 * K, 64, HD, 2, K, 1];
        let norm_axes = axes_of(&norm_ds, &norm_sm);
        let norm = RetileWalk { axes: &norm_axes };
        let perm_axes = axes_of(&perm_ds, &perm_sm);
        let perm = RetileWalk { axes: &perm_axes };
        let mut norm_dev = vec![0u8; norm.total_elems()];
        let mut perm_dev = vec![0u8; perm.total_elems()];
        stage_weight_tiled(&host, &mut norm_dev, norm, Element::Fp8);
        stage_weight_tiled(&host, &mut perm_dev, perm, Element::Fp8);
        assert_eq!(perm_dev, norm_dev);
    }

    #[test]
    fn word_length_is_a_closed_set() {
        assert_eq!(Element::from_word_length(1), Some(Element::Fp8));
        assert_eq!(Element::from_word_length(2), Some(Element::F16));
        assert_eq!(Element::from_word_length(4), None);
    }

    /// ⭐⭐⭐⭐⭐ THE EXPERT BANK'S PACKED SLABS ARE DEVICE-CONTIGUOUS — the property the MoE
    /// gather copy's ENTRY UNIT is. The bank is the expert-major stack `[E·out, in]` (the spyre
    /// `SwitchGluExpertsLayer` loader stacks the per-expert fp8 files), and the manifest emits the
    /// SAME dense-fp8 descriptor for it at `n = E·out`:
    ///
    /// ```text
    ///   device_size [ (E·out)/64, in/2, 64, 2 ]   stride_map [ 64·in, 2, in, 1 ]
    /// ```
    ///
    /// `host[o·in + i]` with `o ∈ [0, E·out)` resolves every coordinate, and expert e's slab —
    /// host rows `[e·out, (e+1)·out)`, a WHOLE number of 64-row n-sticks — lands at DEVICE
    /// elements `[e·out·in, (e+1)·out·in)` as ONE contiguous range. The bytes inside that range
    /// are the PACKED order (k-pairs interleaved), and that is exactly what the consumer needs:
    /// the range is byte-identical to a STANDALONE retile of expert e alone, which is the layout
    /// a `matmulfp8` kernel operand for that expert states. So one gather entry = one expert's
    /// whole packed slab (`gather_copy_opspec`'s `page` = the slab's stick-rows, `cols` = one
    /// 128-elem fp8 stick), and the scratch the copy fills reads as a legal `[out, in]` fp8
    /// kernel operand with no relayout — if the map ever scattered a slab across the range
    /// boundary, the copy would hand the matmul fragments of two experts from a clean bake.
    ///
    /// Pinned at a tiny26-shaped geometry (E=8 experts, out=128 rows, in=128). The dense
    /// single-expert form is the `e = 0` control the granite on-card run already covers.
    #[test]
    fn the_expert_banks_packed_slabs_are_device_contiguous() {
        let (e, out, inn) = (8u64, 128, 128);
        let n = e * out;
        let ds = [n / 64, inn / 2, 64, 2];
        let sm = [64 * inn, 2, inn, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        // Distinct host bytes: host[o * inn + i] = ((o*inn + i) mod 251) — 251 prime, so a wrong
        // map cannot pass by permutation luck; every element's identity is its own host offset.
        let host: Vec<u8> = (0..(n * inn) as usize).map(|x| (x % 251) as u8).collect();
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8), "fast path = naive");
        // ⭐ THE CLAIM: expert e's device RANGE is byte-identical to the STANDALONE packed retile
        // of that expert's host rows — same descriptor shape, local n-sticks [0, out/64).
        let slab = (out * inn) as usize;
        let one_ds = [out / 64, inn / 2, 64, 2];
        let one_sm = [64 * inn, 2, inn, 1];
        let one_axes = axes_of(&one_ds, &one_sm);
        let one = RetileWalk { axes: &one_axes };
        for e in 0..e as usize {
            let base = e * slab;
            let standalone = naive(&host[base..base + slab], one, Element::Fp8);
            assert_eq!(
                &dev[base..base + slab],
                &standalone,
                "expert {e}'s device range is its standalone packed retile, contiguous at {base}"
            );
        }
    }

    /// ⭐⭐⭐⭐⭐ THE CHUNK-MAJOR EXPERT BANK — the 26b's LX wall, answered on the staging side.
    ///
    /// A whole 26b gate/up slab (704·2816 fp8 ≈ 3.96 MB) exceeds per-core LX under the gather's
    /// whole-entry mb split, so the door cuts the slab into 64-column N-stick CHUNKS and the
    /// manifest stages the bank CHUNK-MAJOR: expert `e`'s chunk `χ` at device elements
    /// `(χ·E + e)·chunk_elems`. The walk is the 6-axis form:
    ///
    /// ```text
    ///   device_size [ chunks, E, chunk_groups, in/2, 64, 2 ]
    ///   stride_map  [ chunk_n·in, n·in, 64·in, 2, in, 1 ]
    /// ```
    ///
    /// against the SAME expert-major host `[E·n, in]` (`host[o·in + i]`). The law this pins has
    /// three halves, each the contract one consumer reads:
    ///
    /// * **CHUNK CONTIGUITY** — expert e's chunk χ is ONE contiguous device range at
    ///   `(χ·E + e)·chunk_elems`, byte-identical to a STANDALONE packed retile of that chunk's
    ///   host rows (expert e's rows `[e·n + χ·chunk_n, +chunk_n)`) — the gather's page unit;
    /// * **INDEX COMPOSITION** — the chunk's expert stride within its chunk plane is exactly
    ///   `chunk_elems` (expert e at `e·chunk_elems` inside plane χ), which is the `skip_addr`
    ///   dxp derives for the raw expert-id index; combined with the per-chunk source base
    ///   `χ·E·chunk_elems` it composes `e·chunk + χ·E·chunk = (χ·E + e)·chunk` EXACTLY;
    /// * **TOTALITY** — `prod(device_size) == E·n·in`, the same element count (and therefore
    ///   the same staged-buffer `nbytes`) as the 4-axis expert-major walk.
    ///
    /// At `chunks == 1` the 4-axis descriptor stays the emitted form (byte-identical), pinned
    /// by the test above; this is the `chunks > 1` sibling, pinned at a 26b-shaped geometry
    /// scaled down to keep the test fast (E=8, n=192 = 3 groups ⇒ 3 chunks of 64 columns,
    /// in=128).
    #[test]
    fn the_chunk_major_expert_bank_places_each_chunk_contiguously() {
        let (e, groups, inn) = (8u64, 3u64, 128u64);
        let out = groups * 64; // ONE EXPERT's slab width — the e-stride's own unit
        let n = e * out; // the bank's whole row count (the host's rows)
        let chunk_groups = 1u64; // the 26b gate/up law: prime group count ⇒ one group per chunk
        let chunks = groups / chunk_groups;
        let chunk_n = chunk_groups * 64;
        let chunk_elems = chunk_n * inn;
        let ds = [chunks, e, chunk_groups, inn / 2, 64, 2];
        let sm = [chunk_n * inn, out * inn, 64 * inn, 2, inn, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        // ⭐ TOTATALITY first: the 6-axis walk covers exactly the bank's own elements.
        assert_eq!(
            ds.iter().product::<u64>(),
            n * inn,
            "prod(device_size) == E·n·in — the staged buffer's size is unchanged"
        );
        // Distinct host bytes: host[o·in + i] = ((o·in + i) mod 251) — identity is offset.
        let host: Vec<u8> = (0..(n * inn) as usize).map(|x| (x % 251) as u8).collect();
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8), "fast path = naive");
        // ⭐ THE CLAIM, both halves: expert e's chunk χ is contiguous at `(χ·E + e)·chunk_elems`
        // and byte-identical to the STANDALONE packed retile of that chunk's host rows — the
        // local form of the 4-axis map at `[chunk_groups, in/2, 64, 2]`.
        let one_ds = [chunk_groups, inn / 2, 64, 2];
        let one_sm = [64 * inn, 2, inn, 1];
        let one_axes = axes_of(&one_ds, &one_sm);
        let one = RetileWalk { axes: &one_axes };
        for chi in 0..chunks as usize {
            for exp in 0..e as usize {
                let dev_base = (chi * e as usize + exp) * chunk_elems as usize;
                let host_base = (exp * out as usize + chi * chunk_n as usize) * inn as usize;
                let standalone = naive(
                    &host[host_base..host_base + chunk_elems as usize],
                    one,
                    Element::Fp8,
                );
                assert_eq!(
                    &dev[dev_base..dev_base + chunk_elems as usize],
                    &standalone,
                    "expert {exp}'s chunk {chi} is its standalone packed retile, contiguous at \
                     ({chi}·E + {exp})·chunk_elems = {dev_base}"
                );
            }
        }
        // ⭐ THE INDEX COMPOSITION: inside plane χ, expert e sits exactly `e·chunk_elems` past
        // the plane's start — pinned by the main loop already (expert e's chunk χ at
        // `(χ·E + e)·chunk_elems` IS `e·chunk_elems` past plane χ's start), and the plane
        // spacing is exactly `E·chunk_elems` — the two terms of
        // `idx·skip_addr + χ·E·chunk_elems = (χ·E + e)·chunk_elems` with no relayout and no
        // index remap.
    }

    /// ⭐⭐⭐⭐⭐ THE GROUP-MAJOR SCALE BANK — the prefill (m>1) door's per-(slot, group) scale
    /// gathers read the bank at the STICK LAW `addr = e·64 + g·E·64` (the copy's derived
    /// `skip_addr` at `page = single_position` is one 64-elem fp16 stick, so the RAW expert
    /// id composes with a GROUP-MAJOR plane base). The manifest's retile —
    ///
    /// ```text
    ///   device_size [ n/64, E, 64 ]     stride_map [ 64, n, 1 ]     stick Fp16
    /// ```
    ///
    /// against the SAME expert-major host `[E·n]` stack (`host[e·n + c]`, expert e's n
    /// scales in one contiguous row). The law this pins has three halves:
    ///
    /// * **STICK CONTIGUITY** — device stick `(g, e)` (64 consecutive elements at
    ///   `(g·E + e)·64`) IS expert e's group-g scale row `host[e·n + g·64 .. +64)`, one
    ///   contiguous host range — the gather's page unit, read whole;
    /// * **INDEX COMPOSITION** — the expert stride within a group plane is exactly 64
    ///   (stick `(g, e)` at `e·64` inside plane g), which is the `skip_addr` the raw
    ///   expert-id index derives; combined with the plane base `g·E·64` it composes
    ///   `e·64 + g·E·64 = (g·E + e)·64` EXACTLY;
    /// * **TOTALITY** — `prod(device_size) == E·n`, the same element count (and therefore
    ///   the same staged-buffer `nbytes`) as the flat decode staging: a restage, not a
    ///   resize.
    ///
    /// At `rows == 1` (decode) NO descriptor is inserted for the scale bank — the flat
    /// staging is byte-identical to before this walk existed (the door's m=1 dequant reads
    /// the bank row-major through the gathered scale scratch). This is the `rows > 1`
    /// sibling, pinned at a 26b-shaped geometry scaled down to keep the test fast (E=8,
    /// n=192 = 3 groups).
    #[test]
    fn the_group_major_scale_bank_places_each_stick_contiguously() {
        let (e, groups) = (8u64, 3u64);
        let n = groups * 64; // ONE EXPERT's scale row — the e-stride's own unit
        let ds = [groups, e, 64];
        let sm = [64, n, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        // ⭐ TOTALITY first: the 3-axis walk covers exactly the bank's own elements.
        assert_eq!(
            ds.iter().product::<u64>(),
            e * n,
            "prod(device_size) == E·n — the staged buffer's size is unchanged"
        );
        // Distinct finite fp16 host values: host[j] = 0x3C00 + j — identity is offset.
        let host: Vec<u8> = (0..(e * n) as u16)
            .flat_map(|i| (0x3C00 + i).to_le_bytes())
            .collect();
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::F16);
        assert_eq!(dev, naive(&host, walk, Element::F16), "fast path = naive");
        // ⭐ THE STICK LAW, both halves: device stick (g, e) is contiguous at (g·E + e)·64
        // and IS expert e's group-g host row — the local form the gather's page reads.
        for g in 0..groups as usize {
            for exp in 0..e as usize {
                let dev_base = (g * e as usize + exp) * 64;
                let host_base = exp * n as usize + g * 64;
                for lane in 0..64usize {
                    let want = ieee_to_sen(IeeeF16(0x3C00 + (host_base + lane) as u16)).0;
                    let got = u16::from_le_bytes([
                        dev[(dev_base + lane) * 2],
                        dev[(dev_base + lane) * 2 + 1],
                    ]);
                    assert_eq!(
                        got, want,
                        "stick ({g}, {exp}) lane {lane}: expert {exp}'s group-{g} host row, \
                         contiguous at (g·E + e)·64 = {dev_base}"
                    );
                }
            }
        }
        // ⭐ THE INDEX COMPOSITION: inside plane g, expert e's stick sits exactly `e·64`
        // elements past the plane's start, and the planes are exactly `E·64` apart — the
        // two terms of `e·64 + g·E·64 = (g·E + e)·64` with no relayout and no index remap.
    }

    /// ⭐⭐⭐⭐⭐ THE PADDED-IN CHUNK-MAJOR BANK — the 26b DOWN sibling of the test above, at
    /// the in-pad the door's DOWN projection needs (in=192 → in_pad=256 here; the real door's
    /// 704 → 768). The HOST is the worker's PADDED REBUILD, not the disk buffer: `[E·n,
    /// in_pad]` with each row's `[in, in_pad)` tail zero-filled on the host (marked 0xFF here
    /// so a pad lane can never pass as a real byte — real lanes are `% 251`). The walk is the
    /// same 6-axis chunk-major law with EVERY `in` term at the padded width:
    ///
    /// ```text
    ///   device_size [ chunks, E, chunk_groups, in_pad/2, 64, 2 ]
    ///   stride_map  [ chunk_n·in_pad, out·in_pad, 64·in_pad, 2, in_pad, 1 ]
    /// ```
    ///
    /// Three laws, the padded twins of the sibling's:
    /// * **TOTATALITY AT THE PAD** — `prod(device_size) == E·n·in_pad`, the PADDED count: the
    ///   staging guard the worker's rebuild satisfies (`prod(device_size)·word_length ==
    ///   src.len()`) only closes because BOTH sides state the pad — one rule, the manifest's
    ///   `k` and the worker's rebuild, like the dense K-pad;
    /// * **CHUNK CONTIGUITY AT THE PADDED CHUNK** — expert e's chunk χ is one contiguous
    ///   device range at `(χ·E + e)·chunk_elems` with `chunk_elems = chunk_n·in_pad`,
    ///   byte-identical to the standalone packed retile of that chunk's PADDED host rows;
    /// * **THE PAD LANES ARE STAGED BYTES** — host row o's pad column lands at the same
    ///   packed offset law as a real column, carrying the fill value: the bytes the
    ///   contraction's `0·w` inertness needs to EXIST and be finite, and the count the
    ///   staging guard needs enumerated.
    #[test]
    fn the_padded_in_chunk_major_bank_stages_the_pad_lanes() {
        let (e, groups, inn) = (8u64, 3u64, 192u64); // inn = 1.5 fp8 sticks — the pad's own case
        let in_pad = 256u64;
        let out = groups * 64; // ONE EXPERT's slab width — the e-stride's own unit
        let n = e * out; // the bank's whole row count (the padded host's rows)
        let chunk_groups = 1u64;
        let chunks = groups / chunk_groups;
        let chunk_n = chunk_groups * 64;
        let chunk_elems = chunk_n * in_pad; // the chunk's fp8 elements AT THE PADDED K
        let ds = [chunks, e, chunk_groups, in_pad / 2, 64, 2];
        let sm = [chunk_n * in_pad, out * in_pad, 64 * in_pad, 2, in_pad, 1];
        let walk_axes = axes_of(&ds, &sm);
        let walk = RetileWalk { axes: &walk_axes };
        // ⭐ TOTALITY AT THE PAD: the walk covers exactly the PADDED rebuild's elements.
        assert_eq!(
            ds.iter().product::<u64>(),
            n * in_pad,
            "prod(device_size) == E·n·in_pad — the padded rebuild's own count"
        );
        // The worker's padded rebuild: real lanes are the disk row's bytes (identity is the
        // UNPADDED offset), pad lanes carry the fill marker (0xFF — never a `% 251` value).
        let mut host = vec![0xFFu8; (n * in_pad) as usize];
        for row in 0..n as usize {
            for i in 0..inn as usize {
                host[row * in_pad as usize + i] = ((row * inn as usize + i) % 251) as u8;
            }
        }
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::Fp8);
        assert_eq!(dev, naive(&host, walk, Element::Fp8), "fast path = naive");
        // ⭐ CHUNK CONTIGUITY AT THE PADDED CHUNK: expert e's chunk χ is its standalone packed
        // retile of the chunk's PADDED host rows, at `(χ·E + e)·chunk_elems`.
        let one_ds = [chunk_groups, in_pad / 2, 64, 2];
        let one_sm = [64 * in_pad, 2, in_pad, 1];
        let one_axes = axes_of(&one_ds, &one_sm);
        let one = RetileWalk { axes: &one_axes };
        for chi in 0..chunks as usize {
            for exp in 0..e as usize {
                let dev_base = (chi * e as usize + exp) * chunk_elems as usize;
                let host_base = (exp * out as usize + chi * chunk_n as usize) * in_pad as usize;
                let standalone = naive(
                    &host[host_base..host_base + chunk_elems as usize],
                    one,
                    Element::Fp8,
                );
                assert_eq!(
                    &dev[dev_base..dev_base + chunk_elems as usize],
                    &standalone,
                    "expert {exp}'s chunk {chi} is its standalone padded packed retile, \
                     contiguous at ({chi}·E + {exp})·chunk_elems = {dev_base}"
                );
            }
        }
        // ⭐ THE PAD LANES ARE STAGED BYTES: host row o's PAD column i (192 ≤ i < 256) lands
        // at the same packed offset law as a real column — `(χ·E + e)·chunk_elems +
        // g·64·in_pad + k_outer·128 + n_in·2 + k_inner` — carrying the fill value, and its
        // real neighbor carries the disk row's own byte.
        let (chi, exp, g, n_in, i_pad, i_real) = (2u64, 1u64, 0u64, 3u64, 200u64, 100u64);
        let row = exp * out + chi * chunk_n + g * 64 + n_in;
        let packed = |i: u64| -> usize {
            ((chi * e + exp) * chunk_elems + g * 64 * in_pad + (i / 2) * 128 + n_in * 2 + i % 2)
                as usize
        };
        assert_eq!(
            host[row as usize * in_pad as usize + i_pad as usize],
            0xFF,
            "the pad lane carries the fill marker"
        );
        assert_eq!(
            dev[packed(i_pad)],
            0xFF,
            "the pad lane is staged at the packed law"
        );
        assert_eq!(
            dev[packed(i_real)],
            ((row * inn + i_real) % 251) as u8,
            "the real lane is the disk row's own byte at the same law"
        );
    }
}
