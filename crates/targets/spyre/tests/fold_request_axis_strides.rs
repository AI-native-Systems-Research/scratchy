// SPDX-License-Identifier: Apache-2.0
//! ⛔⛔⛔⛔⛔ WHY A REQUEST AXIS ON A **POOL** OPERAND IS NOT EXPRESSIBLE — the pool law has no request
//! coordinate, so there is no stride for `y` to step, and the constant three attempts baked is not one.
//!
//! ## What this file used to assert, greenly, for five tests
//! *"Collapsing the prefix fold from `pages × requests` launches to `pages` needs the score and value
//! KERNELS to step one request per unit of the batch axis, i.e. a per-`y` step of exactly
//! `PagedKvPool::request_stride` = `hd * PAGE_SLOTS`."*
//!
//! It measured that step off the emitted per-core addresses and found it exact — and it was exact,
//! because the test **defined `req = HD * PAGE_SLOTS` itself** and then checked the emitter reproduced
//! the number the `device_extent` override had just told it to use. A spot-check against its own
//! parameters (`a-spot-check-against-its-own-parameters-verifies-nothing`). It never compared that
//! number against the pool's own address law, which is the only oracle here.
//!
//! ## The pool law, quoted from `PagedKvPool::addr`'s own doc (`sdsc_abstract.rs:5333`)
//! ```text
//! 2. WHICH KV HEAD — and nothing else. There is no request term: a request is a set of SLOTS
//!    (reached through the host's page map), never a coordinate the device computes with.
//! ```
//! `PagedKvPool::request_stride` **no longer exists**: `sdsc_abstract.rs:5361` records that the
//! per-kv-head block distance *"replaced `block_index(kvh) = kvh * ROWS` and `request_stride`"*. So the
//! baked step is not a request stride, and a `y` of 1 does not advance to request 1.
//!
//! ⛔ IT IS **EXACTLY** THE KV-HEAD STRIDE, AND A NEAR-MISS READING OF THAT IS REFUTED BELOW. The
//! distance to the next kv head is `plane_block_elems() = hd * PLANE_SLOTS` — PHYSICAL slots — while
//! `hd * PAGE_SLOTS` is the ADDRESSABLE count, so the natural guess is that the baked constant falls
//! SHORT of kv head 1 by `hd * WRITE_SLACK`. It does not: `WRITE_SLACK` is 0 today, both quantities are
//! 16384, and the first test here records that assertion failing. The step is not approximately the
//! wrong axis, it is precisely the wrong one — see
//! [`the_pinned_request_stride_is_exactly_the_kv_head_stride`].
//!
//! So a `y` of 1 advances one KV HEAD, and with `nkvh = 8` at a batch of 8 the batch axis is ALIASED
//! ONTO the head axis: request `r` scored against kv head `r`'s keys. Which is exactly the reported
//! symptom of all 3-4 attempts: fluent text assembled from another conversation's keys.
//!
//! ## ⭐ WHAT A REQUEST AXIS WOULD NEED, AND WHY THE POOL CANNOT BE IT
//! A `y`-batched matmul steps a stride it DERIVES from a declared extent, so the operand must HAVE a
//! uniform per-request pitch. The pool does not, and by its own law never will — the law's three axes
//! are kv_head, slot and feat_slab, and a request is a SET OF SLOTS chosen by the host's page map, not
//! a coordinate the device multiplies. So the collapse needs a DIFFERENT operand, one whose per-request
//! pitch this compiler chooses; it is not reachable by picking a better constant for the pool, which is
//! the move every attempt made. The three tests below fence off the constant so a fifth attempt cannot
//! start from the same false evidence.
//!
//! ## ⭐ THE SHAPE THAT CAN CARRY A REQUEST AXIS
//! A `y`-batched matmul steps a stride it DERIVES from a declared extent, so the operand must HAVE a
//! uniform per-request pitch. The pool does not and by its own law will not. A **contiguous gather
//! destination** does: `gather_copy_opspec` copies each row's page out of the pool through the index
//! into a scratch whose pitch this compiler chooses, so the `y` step is the scratch's own declared
//! pitch and the request→slots mapping stays where it is known — the host's page map, delivered as
//! index entries. That is why the gather is not an optimisation of the collapse but its precondition.
//!
//! ⛔ WHICH ALSO MEANS THE TWO CANNOT LAND SEPARATELY. `fold_plan::fold_delta` returns
//! `kv = page_base_bytes(s, rp)` — an ABSOLUTE per-pass KV base applied as a segment shift — while a
//! gathered read computes `addr = idx * skip_addr + base` from that already-shifted base. Wiring the
//! gather in without dropping the shift composes two bases and lands inside neither. One atomic change.

use ktir_superdsc::sdsc_abstract::{FeatIdx, KvCoord, KvHead, KvPlane, KvSlot, PagedKvPool};

const HD: usize = 64;
const NKVH: usize = 8;

/// ⭐⭐⭐ THE PINNED CONSTANT **IS** THE KV-HEAD STRIDE, EXACTLY — so `y = r` addresses kv head `r`.
///
/// ⛔ I FIRST WROTE THIS TEST ASSERTING THE CONSTANT WAS *SHORT* of the kv-head stride by
/// `hd * WRITE_SLACK`, reasoning from the addressable-vs-physical split that really did put a padded
/// chunk write on the next head's keys. **The assertion failed: both are 16384.** `WRITE_SLACK` is `0`
/// today (`sdsc_abstract.rs:5195`, and `:90` says so outright), so `PLANE_SLOTS == PAGE_SLOTS` and the
/// two readings coincide. The hypothesis is recorded as refuted because the near-miss story it tells is
/// more forgiving than the truth: the step is not approximately the wrong axis, it is EXACTLY the
/// wrong axis.
///
/// ⚠️ AND THE IDENTITY IS A COINCIDENCE WITH AN EXPIRY. `sdsc_abstract.rs:113` records that a non-zero
/// `WRITE_SLACK` "arrived, and broke head_dim 128". Whenever it is non-zero again, `hd * PAGE_SLOTS`
/// stops being any axis at all — so this test pins the equality rather than the constant, and fails
/// loudly the day the two split.
#[test]
fn the_pinned_request_stride_is_exactly_the_kv_head_stride() {
    let pool = PagedKvPool::new(NKVH, HD);
    let pinned = HD * PagedKvPool::PAGE_SLOTS;
    assert_eq!(
        pool.plane_block_elems(),
        HD * PagedKvPool::PLANE_SLOTS,
        "the pool's kv-head distance is PHYSICAL slots, whatever PAGE_SLOTS says"
    );
    assert_eq!(
        pinned,
        pool.plane_block_elems(),
        "`hd * PAGE_SLOTS` — the constant three request-axis attempts baked as the deleted \
         `PagedKvPool::request_stride` — is the KV-HEAD stride. A `y` step of one advances one kv \
         head. If this ever stops holding (WRITE_SLACK != 0) the constant becomes no axis at all, \
         which is worse, not better."
    );
    assert_eq!(
        PagedKvPool::WRITE_SLACK,
        0,
        "the equality above holds only while WRITE_SLACK is 0 — when it is not, `hd * PAGE_SLOTS` \
         names nothing in the law and this file's second test is the one that still holds"
    );
}

/// ⭐⭐⭐ A STEP OF THE PINNED SIZE ADVANCES A **KV HEAD**, NOT A REQUEST — so `y = r` reads head `r`.
///
/// This is the mechanism behind "another conversation's keys": with `nkvh = 8` and a batch of 8, the
/// batch axis sweeps exactly the kv-head axis, so request `r`'s score is computed against kv head
/// `r`'s keys. Every row gets real, well-formed keys belonging to the wrong head of the wrong request.
#[test]
fn stepping_the_batch_axis_by_the_pinned_constant_walks_the_kv_head_axis() {
    let pool = PagedKvPool::new(NKVH, HD);
    let nkvh_nz = std::num::NonZeroU32::new(NKVH as u32).expect("nkvh > 0");
    let head0 = KvHead::new(0, nkvh_nz).expect("kv head 0");
    let head1 = KvHead::new(1, nkvh_nz).expect("kv head 1");
    let step = pool.addr(KvCoord::block(KvPlane::Kt, head1))
        - pool.addr(KvCoord::block(KvPlane::Kt, head0));
    assert_eq!(
        step as usize,
        pool.plane_block_elems(),
        "one unit of the kv-head axis IS `plane_block_elems` — the axis the pinned constant \
         approximates. There is no other axis of that magnitude in the law."
    );
}

/// ⛔⛔⛔ AND THERE IS NO REQUEST COORDINATE TO STEP: the SAME `KvCoord` is the same address, whichever
/// request is being served.
///
/// The structural proof is that `KvCoord` has no request field — `block(plane, kvh).at_slot(..)
/// .at_feat(..)` is the whole vocabulary — so this asserts the consequence: two requests reaching the
/// same logical position produce ONE address. A request is separated by which SLOTS the host's page map
/// gave it, and the device never computes with that.
#[test]
fn the_only_axis_the_pinned_constant_matches_is_the_kv_head_one() {
    let pool = PagedKvPool::new(NKVH, HD);
    let nkvh_nz = std::num::NonZeroU32::new(NKVH as u32).expect("nkvh > 0");
    let kvh0 = KvHead::new(0, nkvh_nz).expect("kv head 0");
    let base = pool.addr(KvCoord::block(KvPlane::Kt, kvh0));
    // EVERY axis the coordinate vocabulary offers, as its own one-unit step. `KvCoord` has exactly
    // these — `block(plane, kvh)`, `.at_slot()`, `.at_feat()` — and no request among them.
    let kv_head = pool.addr(KvCoord::block(
        KvPlane::Kt,
        KvHead::new(1, nkvh_nz).expect("kv head 1"),
    )) - base;
    let slot = pool.addr(KvCoord::block(KvPlane::Kt, kvh0).at_slot(KvSlot::new(1))) - base;
    let feat = pool.addr(KvCoord::block(KvPlane::Kt, kvh0).at_feat(FeatIdx::of_slab(1))) - base;
    let pinned = (HD * PagedKvPool::PAGE_SLOTS) as u32;
    // The KV-HEAD axis is the one it matches, and matching it is the defect: the batch axis was
    // aliased onto the head axis, so request `r` scored against kv head `r`'s keys. With nkvh=8 and a
    // batch of 8 the alias is total — every row reads real, well-formed keys from the wrong head.
    assert_eq!(
        kv_head, pinned,
        "the kv-head axis is the axis the request axis was actually walking"
    );
    // ⛔ AND NEITHER OF THE OTHER TWO AXES IS A CANDIDATE, so there was never a third reading in which
    // the constant meant something per-request. `slot` is where a request's separation actually lives —
    // and it is the HOST's page map that says which slots, never a stride the device multiplies.
    for (name, step) in [("slot", slot), ("feat_slab", feat)] {
        assert_ne!(
            step, pinned,
            "the pool's `{name}` axis steps {step}, not the pinned {pinned}"
        );
    }
    assert!(
        slot > 0 && feat > 0 && slot != feat,
        "the law's remaining axes must both be live and distinct (slot={slot}, feat={feat}), or this \
         enumeration is not covering the vocabulary it claims to"
    );
}
