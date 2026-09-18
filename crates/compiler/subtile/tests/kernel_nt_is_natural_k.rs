// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! ⭐⭐⭐⭐⭐ A `KernelNt` LAYOUT OVER THE NATURAL-K PLANE ADDRESSES EXACTLY WHAT THE POOL SAYS NATURAL K
//! IS — so the score matmul's kernel can BE natural K, and the resident Kᵀ plane is not a hardware
//! requirement.
//!
//! The resident Kᵀ plane exists because [`StickKind::Kernel`] is `[in, out]` stick-blocked on `out`:
//! the contraction is the ROW axis, by construction, so the only shape the score matmul could name was
//! `[hd, cap]` — and natural K, which the cache write produces directly, had to be re-transposed into
//! it. Three planes per page, a whole-page restickify, and ⅓ of the KV pool spent on a transpose source.
//!
//! ⛔ NONE OF THAT WAS THE DEVICE. `dev_off` already addresses BOTH orientations — it is one function
//! with a `stick_idx == 1` tiled branch, and the dims decide which layout you get — and this tree's own
//! `stride_map_disk_order` already reads GEMM weights in their on-disk orientation on the strength of
//! "two swapped strides, not a data movement". What was missing was a KIND that says "the contraction is
//! the sticked axis", so `StickKind::KernelNt` is that, and this file is the proof that it lands on
//! natural K's bytes.
//!
//! ⛔ AND IT IS NOT `Kernel` WITH SWAPPED ARGUMENTS, which is the trap this pins shut:
//! `StickLayout::kernel(k_in, n_out)` puts the contraction on `rows`, so `kernel(cap, hd)` tells the
//! matmul to reduce over SLOTS. Same shapes, same checks, wrong math. The last test here is that the two
//! kinds are NOT address-equal at the same shape, so nothing can quietly substitute one for the other.

use scratchy_subtile::sdsc_abstract::{
    FeatIdx, KvCoord, KvHead, KvPlane, KvSlot, PagedKvPool, StickKind, StickLayout,
};
use std::num::NonZeroU32;

const NKVH: usize = 8;

fn head(i: u32) -> KvHead {
    KvHead::new(i, NonZeroU32::new(NKVH as u32).unwrap()).expect("kv head in range")
}

/// ⭐ THE MEASUREMENT: element for element, a `KernelNt` `[PAGE_SLOTS, hd]` IS the pool's natural-K plane.
///
/// Checked against `PagedKvPool::addr` — the one address law — rather than against a formula written
/// here, so this cannot pass by restating the arithmetic it is supposed to be checking.
#[test]
fn a_kernel_nt_layout_addresses_the_natural_k_plane() {
    for hd in [64usize, 128, 256] {
        let pool = PagedKvPool::new(NKVH, hd);
        let nt = StickLayout::kernel_nt(PagedKvPool::PAGE_SLOTS, hd);
        for h in [0u32, 3, 7] {
            let block = pool.addr(KvCoord::block(KvPlane::Knat, head(h)));
            for slot in [0usize, 1, 63, 64, 65, 127, 200, PagedKvPool::PAGE_SLOTS - 1] {
                for feat in [0usize, 1, 63, hd / 2, hd - 1] {
                    let want = pool.addr(
                        KvCoord::block(KvPlane::Knat, head(h))
                            .at_slot(KvSlot::new(slot as u32))
                            .at_feat(FeatIdx::new(feat as u32)),
                    ) - block;
                    assert_eq!(
                        nt.dev_off(slot, feat),
                        want as usize,
                        "hd={hd} kvh={h} slot={slot} feat={feat}: a KernelNt [PAGE_SLOTS, hd] must land \
                         on the pool's natural-K element, or the score matmul would read a plane the \
                         cache write never filled"
                    );
                }
            }
        }
    }
}

/// And the mirror: a plain `Kernel` `[hd, PAGE_SLOTS]` is the Kᵀ plane. Both orientations come out of ONE
/// address law; the kind and the dims are the whole difference.
#[test]
fn a_kernel_layout_addresses_the_transposed_plane() {
    for hd in [64usize, 128, 256] {
        let pool = PagedKvPool::new(NKVH, hd);
        let kt = StickLayout::kernel(hd, PagedKvPool::PAGE_SLOTS);
        let block = pool.addr(KvCoord::block(KvPlane::Kt, head(2)));
        for slot in [0usize, 1, 63, 64, 255] {
            for feat in [0usize, 1, hd - 1] {
                let want = pool.addr(
                    KvCoord::block(KvPlane::Kt, head(2))
                        .at_slot(KvSlot::new(slot as u32))
                        .at_feat(FeatIdx::new(feat as u32)),
                ) - block;
                assert_eq!(kt.dev_off(feat, slot), want as usize, "hd={hd} Kᵀ ({slot},{feat})");
            }
        }
    }
}

/// ⛔ THE CONTRACTION COMES FROM THE KIND, NOT FROM A POSITION — which is what makes the two safe to
/// share one matmul path.
#[test]
fn the_kind_says_which_axis_reduces() {
    let hd = 64usize;
    let cap = PagedKvPool::PAGE_SLOTS;
    let kt = StickLayout::kernel(hd, cap);
    let nt = StickLayout::kernel_nt(cap, hd);
    assert_eq!(kt.contraction_elems(), Some(hd), "Kᵀ reduces over hd");
    assert_eq!(nt.contraction_elems(), Some(hd), "natural K ALSO reduces over hd");
    assert_eq!(kt.output_elems(), Some(cap), "and both emit cap score columns");
    assert_eq!(nt.output_elems(), Some(cap));
    assert_eq!(kt.kind, StickKind::Kernel);
    assert_eq!(nt.kind, StickKind::KernelNt);
}

/// ⛔⛔⛔ THE `kernel(cap, hd)` MISTAKE IS A CONTRACTION ERROR, NOT AN ADDRESS ERROR — and this test
/// exists because I first wrote it the other way round and it FAILED, correctly.
///
/// I asserted that a `Kernel[cap,hd]` and a `KernelNt[cap,hd]` are not address-equal. At `hd == 64` they
/// ARE: `cols == lanes`, so there is exactly one stick group, the stick-blocking degenerates and both
/// layouts map element `(r,c)` to the same byte. `addr_eq` is right and my assertion was wrong.
///
/// Which sharpens what the new kind is FOR. The hazard in swapping `kernel`'s arguments was never that
/// the bytes move — at one stick they do not — it is that `Kernel` means "reduce over rows". Same
/// buffer, same bytes, every shape check green, and the matmul contracts over SLOTS instead of the head
/// dim. That is exactly the class this codebase keeps paying for, and no address check can catch it:
/// only the KIND carries it, which is why `contraction_elems` asks the kind and not a position.
#[test]
fn the_swap_is_a_contraction_error_that_no_address_check_can_catch() {
    let cap = PagedKvPool::PAGE_SLOTS;
    for hd in [64usize, 128, 256] {
        let swapped = StickLayout::kernel(cap, hd); // the MISTAKE: reduces over cap
        let correct = StickLayout::kernel_nt(cap, hd); // reduces over hd
        assert_eq!(
            swapped.contraction_elems(),
            Some(cap),
            "hd={hd}: `kernel(cap, hd)` contracts over SLOTS — the whole reason this is a kind"
        );
        assert_eq!(correct.contraction_elems(), Some(hd), "hd={hd}");
        // ⚠️ SKIPPED AT `hd == cap`, and that coincidence is worth naming: at head_dim 256 the head dim
        // and a page's slots are BOTH 256, so the two kinds agree on the contraction LENGTH while still
        // reducing over different axes. A test that asserted inequality there failed for the wrong
        // reason — the numbers coincide, the meanings do not — which is the same "two quantities, one
        // value" trap `POOL_STICK` is documented against.
        if hd != cap {
            assert_ne!(
                swapped.contraction_elems(),
                correct.contraction_elems(),
                "hd={hd}: the two disagree about what reduces, which is the error the kind exists to name"
            );
        }
        // AND AT ONE STICK THE BYTES ARE IDENTICAL, so the address authority cannot be the guard.
        if hd == 64 {
            assert!(
                swapped.addr_eq(&correct),
                "at hd == lanes there is one stick group and both layouts are the same bytes — so a \
                 wrong contraction is INVISIBLE to `addr_eq`, which is the point"
            );
        }
    }
}
