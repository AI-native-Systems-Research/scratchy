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

/// One weight's device-tile walk: device axis extents + the host-element stride per device axis.
/// Slices, not owned Vecs — the caller (Rust or the C++ shim) owns the descriptor.
#[derive(Clone, Copy, Debug)]
pub struct RetileWalk<'a> {
    pub device_size: &'a [u64],
    pub stride_map: &'a [u64],
}

impl<'a> RetileWalk<'a> {
    /// Total device elements = prod(device_size).
    pub fn total_elems(&self) -> usize {
        self.device_size.iter().product::<u64>() as usize
    }

    fn inner(&self) -> usize {
        self.device_size.last().copied().unwrap_or(1) as usize
    }

    fn inner_contig(&self) -> bool {
        let inner = self.inner();
        !self.device_size.is_empty() && self.stride_map.last() == Some(&1) && inner > 1
    }

    /// Host element offset for the device coordinate `coord` (dot with `stride_map`, excluding
    /// the last `skip_tail` axes — the fast path computes per-row offsets with the innermost
    /// axis at 0).
    fn host_off(&self, coord: &[u64], skip_tail: usize) -> usize {
        coord
            .iter()
            .zip(self.stride_map)
            .take(coord.len() - skip_tail)
            .map(|(&c, &s)| c * s)
            .sum::<u64>() as usize
    }

    /// Odometer increment over the leading `rank - skip_tail` axes; the C++ walk's carry loop.
    fn advance(&self, coord: &mut [u64], skip_tail: usize) {
        for r in (0..coord.len() - skip_tail).rev() {
            coord[r] += 1;
            if coord[r] < self.device_size[r] {
                break;
            }
            coord[r] = 0;
        }
    }
}

/// Stage one weight: read `host` (the on-disk layout, IEEE-fp16 or fp8 bytes) and write `dev`
/// (the device tile layout, SEN bytes) via the descriptor's pure walk.
///
/// `host` and `dev` are in ELEMENTS of `elem` width: both must hold `walk.total_elems()`
/// elements (the caller has already validated the host size against prod(device_size), which for
/// a permutation walk bounds every host offset).
pub fn stage_weight_tiled(host: &[u8], dev: &mut [u8], walk: RetileWalk<'_>, elem: Element) {
    let rank = walk.device_size.len();
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
                .zip(walk.stride_map)
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let rank = walk.device_size.len();
        let mut coord = vec![0u64; rank];
        for lin in 0..total {
            let hoff: usize = coord
                .iter()
                .zip(walk.stride_map)
                .map(|(&c, &s)| c * s)
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
                if coord[r] < walk.device_size[r] {
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
        let mut dev = vec![0u8; host.len()];
        stage_weight_tiled(&host, &mut dev, walk, Element::F16);
        assert_eq!(dev, naive(&host, walk, Element::F16));
        // spot-check the permutation itself: device(t=1, i=2, s=3) = host(i=2, o=1·4+3=7)
        let lin = ((1 * inn + 2) * stick + 3) as usize;
        let hoff = (2 * out + 7) as usize;
        let want = ieee_to_sen(IeeeF16(0x3C00 + hoff as u16)).0;
        assert_eq!(u16::from_le_bytes([dev[lin * 2], dev[lin * 2 + 1]]), want);
    }

    /// fp8 is a VERBATIM byte gather — no conversion.
    #[test]
    fn fp8_gather_matches_naive_and_converts_nothing() {
        let host: Vec<u8> = (0..128u8).collect();
        let ds = [2u64, 4, 16];
        let sm = [16u64, 32, 1];
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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

    #[test]
    fn word_length_is_a_closed_set() {
        assert_eq!(Element::from_word_length(1), Some(Element::Fp8));
        assert_eq!(Element::from_word_length(2), Some(Element::F16));
        assert_eq!(Element::from_word_length(4), None);
    }
}
