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
        let lin = ((inn + 2) * stick + 3) as usize;
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let one = RetileWalk {
            device_size: &one_ds,
            stride_map: &one_sm,
        };
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let one = RetileWalk {
            device_size: &one_ds,
            stride_map: &one_sm,
        };
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let walk = RetileWalk {
            device_size: &ds,
            stride_map: &sm,
        };
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
        let one = RetileWalk {
            device_size: &one_ds,
            stride_map: &one_sm,
        };
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
