#![cfg(feature = "spyre")]
//! ⛔ ONE LAW OR TWO? The prefill WRITE and the fold READ must address the same byte.
//!
//! Isolated on the card 2026-08-09: 200 tokens of context reached by DECODING reads back correctly, while a
//! 175-word PROMPT (3 chunks of `n_new=96`) produces garbage after the first token. So the fold's read path
//! is sound and the fault is in multi-chunk PREFILL — and the one thing the prefill write uses that the
//! decode fold does not is `kt_write_off` / `v_write_off`, whose head term changed when the pool's request
//! dimension (`ROWS`) was deleted.
//!
//! Two functions computing one address is the bug class this pins: `addr` is the law, and these two are a
//! second derivation of it. If they agree for slot 0 and diverge later, chunk 0 lands correctly and every
//! later chunk does not — exactly the measured boundary.
//!
//! Checked at hd=64 AND hd=128, because a stick width standing in for a head width is the other way this
//! family of bugs shows up, and those two are equal only at 64.

use scratchy_subtile::sdsc_abstract::{FeatIdx, KvCoord, KvHead, KvPlane, KvSlot, PagedKvPool};
use std::num::NonZeroU32;

fn head(i: u32, nkvh: u32) -> KvHead {
    KvHead::new(i, NonZeroU32::new(nkvh).unwrap()).expect("head in range")
}

#[test]
fn the_kt_write_offset_is_the_same_address_the_fold_reads() {
    for (nkvh, hd) in [(8usize, 64usize), (8, 128), (4, 128)] {
        let p = PagedKvPool::new(nkvh, hd);
        for kvh in 0..nkvh {
            // Slots spanning several prefill chunks (96 rows each) and several 64-slot fold blocks.
            for slot in [0usize, 1, 63, 64, 95, 96, 128, 191, 192, 255] {
                for d in [0usize, 1, 63, hd - 1] {
                    let write = p.kt_write_off(kvh, slot, d);
                    let read = p.addr(
                        KvCoord::block(KvPlane::Kt, head(kvh as u32, nkvh as u32))
                            .at_slot(KvSlot::new(slot as u32))
                            .at_feat(FeatIdx::new(d as u32)),
                    ) as usize;
                    assert_eq!(
                        write, read,
                        "Kt nkvh={nkvh} hd={hd} kvh={kvh} slot={slot} d={d}: the write and the fold's \
                         read disagree — two derivations of one law"
                    );
                }
            }
        }
    }
}

#[test]
fn the_v_write_offset_is_the_same_address_the_fold_reads() {
    for (nkvh, hd) in [(8usize, 64usize), (8, 128), (4, 128)] {
        let p = PagedKvPool::new(nkvh, hd);
        for kvh in 0..nkvh {
            for slot in [0usize, 1, 63, 64, 95, 96, 128, 191, 192, 255] {
                for d in [0usize, 1, 63, hd - 1] {
                    let write = p.v_write_off(kvh, slot, d);
                    let read = p.addr(
                        KvCoord::block(KvPlane::V, head(kvh as u32, nkvh as u32))
                            .at_slot(KvSlot::new(slot as u32))
                            .at_feat(FeatIdx::new(d as u32)),
                    ) as usize;
                    assert_eq!(
                        write, read,
                        "V nkvh={nkvh} hd={hd} kvh={kvh} slot={slot} d={d}: the write and the fold's read \
                         disagree — two derivations of one law"
                    );
                }
            }
        }
    }
}
