// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ THE CAPACITY LAW — the KV pool's size, stated ONCE for both of its readers.
//!
//! **THE LAW (user order, 2026-11-17): the KV pool is EVERY WHOLE PAGE the funded budget holds
//! beyond mandatory residency, FLOORED at the model's whole context.** The model's context is
//! the FLOOR, never the cap: a bake that cannot fund the model's whole context REFUSES at the
//! bake (the compile-time check — serving need is a law, retention past it is the pool's
//! purpose); everything the funded budget holds past that is the pool's to take. Bank count
//! derives arithmetically, exactly as the KV banking already derives it (`bank_kv_segment`
//! splits at layer boundaries into as many regions as the pool takes).
//!
//! The two readers, and why the law lives between them:
//!
//! * **THE BAKE** (`scratchy-target-spyre-compiler`) lowers every bundle at a capacity and
//!   CANNOT change it later — the pool is baked per-layer placements, launch bank+shift tables
//!   and a page count minted from the lowered IR, so growth at load is structurally impossible.
//!   The bake therefore runs the law as a descent: lower at the floor, measure the census,
//!   grow to the law's pages, re-lower, until the capacity is the law's own fixed point.
//! * **THE LOAD CENSUS** (`spyre_load.rs`) measures the REAL card and the built bundles, and
//!   keeps its two refusals (the funded check and the per-bank allocation check) plus the
//!   anchor's reserve-or-borrow placement decision — all stated over THIS module's terms, never
//!   over a second copy of the arithmetic.
//!
//! Both readers hold the same bundles' layouts, so the census terms are stated here over
//! [`RungFacts`] — the projection of EITHER layout (the bake's compile-time
//! [`crate::placement::BundleLayout`] or the runtime `bundle_code::BundleLayout`) that the
//! census reads: segment bytes, the weight and KV segments' extra banks, and the weight-tail
//! verdict.
//!
//! # The funded budget is a compile-time fact, not a probe
//!
//! Every term is a bake fact: the card's memory is [`DeviceGen::MEMORY_BYTES`] (the gen1
//! platform table's own row, measured on dd2), the carve into allocatable tensor capacity is
//! [`tensor_capacity`] (regions × tensor regions, one region reserved for program memory), and
//! the fraction is [`DEFAULT_GPU_MEMORY_UTILIZATION`]. Nothing reads the card at boot to size
//! the pool — that is torch's free-memory probe, and it is not what this is. A serve that
//! passes a SMALLER `--gpu-memory-utilization` than the default is refused by the load census's
//! funded check (the pool is already baked at the default's size); a larger one simply leaves
//! the difference unspent.
//!
//! # What the law is NOT (the three violations, each ruled out)
//!
//! * **Not a config fact** — no cell, model or utilization keying: the law is stated over the
//!   census's own terms (weights, rung charge, decode's private scratch) and the platform
//!   table, the same statement for every model.
//! * **Not unbounded** — the funded budget bounds it, and mandatory residency is charged
//!   FIRST: the pool takes exactly the remainder, in whole pages.
//! * **Not a runtime probe** — see above; a smaller real card is a load-time refusal, never a
//!   silently-shrunk pool.

use ktir_core::ddl_binds::{Baked, DeviceGen};

use crate::placement::WeightTail;

/// The segment bytes of ONE rung, indexed by segment id — the census's unit of account.
pub type SegBytes = [u64; <Baked as DeviceGen>::NUM_SEGMENTS];

/// The WEIGHT segment's index — per-layer resident weights, aliased by every borrower.
pub const SEG_WEIGHT: usize = crate::placement::SegRole::Weight as usize;
/// The KV segment's index — the resident pool this law sizes.
pub const SEG_KV: usize = crate::placement::SegRole::Kv as usize;

/// ⭐ THE PREFILL FAMILY'S SHARED-SCRATCH SEGMENTS — the segments whose scratch ONE family of
/// mutually-exclusive launches shares (one arena per segment at the family's widest extent):
/// INTERMEDIATE (seg0) plus the two intermediate color segments (seg5/seg6).
///
/// ⛔ THE EXCLUSION LAW that licenses sharing at all: prefill-shaped launches COLLECT IN-CALL,
/// one at a time, serialized by the executor — a launch's scratch is dead before the next
/// family member's begins — while the DECODE launch is submitted-then-deferred and genuinely
/// overlaps a prefill launch, so decode's scratch stays private and no decode segment appears
/// here. The card crate's typed table (`superdsc_exec::SEG_SHARED_SCRATCH`) is THIS table in
/// its executor's own type, pinned equal by const assert there.
///
/// seg5/seg6 are the coloring pass's two spare intermediate colors, not [`crate::placement::SegRole`]
/// variants — they are named here as indices because that is what a layout's `segment_bytes`
/// is indexed by.
pub const SEG_SHARED_SCRATCH: [usize; 3] = [crate::placement::SegRole::Intermediate as usize, 5, 6];

/// The fraction of the allocatable tensor capacity a serve funds — the serving layer's own
/// `--gpu-memory-utilization` default (the value every run that names nothing else runs at),
/// stated here because the BAKE's funded budget must be the same budget the load census
/// checks against. A run that passes a smaller fraction is refused by the load census (the
/// pool is baked at this size); a larger one leaves the difference unspent.
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.9;

/// ⛔ THE ALLOCATOR'S REGION CARVE, AS THE BAKE STATES IT — the card is cut into
/// [`PLATFORM_REGIONS`] equal regions (each capped at [`DeviceGen::MAX_SEGMENT_BYTES`]), the
/// first [`PLATFORM_PROGRAM_REGIONS`] of them reserved for program memory, and a tensor
/// allocation is served from the rest.
///
/// The region count is the executor's own segment count (one region per occupied segment —
/// the card crate pins flex-rs's `DEFAULT_MAX_REGIONS` equal to it); the program count is
/// flex-rs's `DEFAULT_NUM_PROGRAM_REGIONS`, pinned by the same card-crate assert. This is the
/// ONE derivation: the card's `CardMemory::tensor_capacity` delegates here.
pub const PLATFORM_REGIONS: u64 = <Baked as DeviceGen>::NUM_SEGMENTS as u64;
/// The regions the allocator reserves for PROGRAM memory — see [`PLATFORM_REGIONS`].
pub const PLATFORM_PROGRAM_REGIONS: u64 = 1;

/// One region's size, exactly as the allocator's pre-allocation computes it: an equal share
/// of the card, capped at the hardware ceiling.
pub const fn region_bytes(card_bytes: u64) -> u64 {
    let per = card_bytes / PLATFORM_REGIONS;
    if per < <Baked as DeviceGen>::MAX_SEGMENT_BYTES {
        per
    } else {
        <Baked as DeviceGen>::MAX_SEGMENT_BYTES
    }
}

/// ⛔ THE BYTES A TENSOR ALLOCATION CAN ACTUALLY BE SERVED FROM — not the card. Measured on
/// dd2: a budget derived from the whole card over-states what exists by one region's worth
/// (the allocator's own OOM diagnostic reported 96 GiB where the card reports 114,688 MB).
pub const fn tensor_capacity(card_bytes: u64) -> u64 {
    region_bytes(card_bytes) * (PLATFORM_REGIONS - PLATFORM_PROGRAM_REGIONS)
}

/// The funded budget: the allocatable tensor capacity times the utilization fraction. The
/// bake computes it from [`DeviceGen::MEMORY_BYTES`] and [`DEFAULT_GPU_MEMORY_UTILIZATION`];
/// the load census from the measured card and the run's own fraction. One arithmetic.
pub fn funded_bytes(card_bytes: u64, utilization: f64) -> u64 {
    (tensor_capacity(card_bytes) as f64 * utilization) as u64
}

/// ⭐ THE POOL'S PAGE STRIDE — its whole bytes over its own page count. One law for both ends:
/// the bake's census states it over the baked layout (`RungFacts::kv_total_bytes` over the
/// capacity the pool was baked at), and the load states it over the runtime bundle's own
/// bytes — the same division, never a second arithmetic at either end. 0 pages is no pool
/// (nothing paged), not a division.
pub fn page_stride_bytes(kv_total_bytes: u64, pages: u32) -> u64 {
    if pages == 0 {
        0
    } else {
        kv_total_bytes / u64::from(pages)
    }
}

/// One rung's census facts — the projection of EITHER layout (the bake's compile-time
/// `placement::BundleLayout` or the runtime `bundle_code::BundleLayout`) that every term of
/// the capacity law reads. A projection, not a copy: each field is read from the layout that
/// already owns it, never re-derived.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RungFacts {
    /// Bytes occupied per segment, index = segment id. For the weight segment this is BANK 0's
    /// bytes — the extra banks are [`Self::weight_bank_bytes`], and both are charged.
    pub segment_bytes: SegBytes,
    /// The weight segment's extra banks (banks `1..N`), empty when the weights fit one region.
    pub weight_bank_bytes: Vec<u64>,
    /// The KV segment's extra banks (banks `1..N`), empty when every layer's pools fit one
    /// region — same law as the weight banks, over the pool this law sizes.
    pub kv_bank_bytes: Vec<u64>,
    /// The segment budget's verdict on the non-per-layer weight tail — the per-borrower copy
    /// charge and the load's copy work list both walk [`WeightTail::spilled_span`].
    pub weight_tail: WeightTail,
}

impl RungFacts {
    /// What the bundle already owns on the card, charged as weights-and-overhead: every
    /// segment except the KV one (whose bytes are what the pool is about to size), PLUS the
    /// weight segment's extra banks, which `segment_bytes` does not count. Charged ONCE, not
    /// per session — a bank IS aliasable, which is the difference between this and the
    /// spilled tail.
    pub fn non_kv_bytes(&self) -> u64 {
        self.segment_bytes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != SEG_KV)
            .map(|(_, b)| *b)
            .sum::<u64>()
            + self.weight_bank_bytes.iter().sum::<u64>()
    }

    /// The POOL's whole bytes — the KV segment's bank 0 plus its extra banks, exactly what the
    /// owner session allocates and what a funded budget must hold. One reading, both census
    /// ends: the bake's layout and the runtime bundle carry the same two fields.
    pub fn kv_total_bytes(&self) -> u64 {
        self.segment_bytes[SEG_KV] + self.kv_bank_bytes.iter().sum::<u64>()
    }

    /// Every segment except the two resident ones (weights and KV) — what a borrowing session
    /// allocates for itself. The decode rungs are charged exactly this (they share nothing);
    /// the prefill family is charged [`family_non_resident_bytes`] instead.
    pub fn non_resident_bytes(&self) -> u64 {
        self.segment_bytes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != SEG_WEIGHT && *i != SEG_KV)
            .map(|(_, b)| *b)
            .sum()
    }

    /// The spilled weight tail's bytes — charged ONCE PER SESSION beyond the owner, because a
    /// spill slot's co-tenant intermediate colour is m-dependent and every borrowing session
    /// takes its own copy (measured: 27 sessions × 419,430,400 B = 10.55 GiB this budget once
    /// could not see).
    pub fn tail_bytes(&self) -> u64 {
        self.weight_tail.spilled_span().map(|(_, b)| b).unwrap_or(0)
    }
}

/// ⭐⭐ THE PREFILL FAMILY'S SHARED-SCRATCH ARENAS, AS ONE COMPUTATION FOR BOTH CONSUMERS —
/// the census (which must charge the bytes the build will allocate) and the load wave (which
/// must pick the rung that allocates them). Every shared segment ([`SEG_SHARED_SCRATCH`]) gets
/// ONE arena at the family's WIDEST extent: the prefill family's launches are collected
/// in-call one at a time (the exclusion law on that table), so the widest rung's region can
/// back every narrower one, and each rung addresses its own bake-proven placement inside its
/// own `segment_bytes`.
///
/// Returns, per shared segment, `(the family's widest extent, the index of the family entry
/// that holds it)`. An empty family returns `(0, usize::MAX)` — nothing is charged and nothing
/// is shared; the callers treat `usize::MAX` as "no arena".
///
/// ⛔ THE ENTRY ORDER IS THE CALLER'S, AND BOTH CALLERS MUST FEED THE SAME FAMILY: the census
/// walks the manifest's prefill-rung and prefix-rung lists, the wave walks the fan-out's
/// `ladder` and `prefix_jobs` — the same rungs by the invariant the census already rests on
/// ("charged as the build will allocate it"). A rung that appears in one and not the other is
/// a pre-existing defect in that invariant, not a fact this function re-derives.
pub fn scratch_arena_owners(lays: &[&RungFacts]) -> [(u64, usize); SEG_SHARED_SCRATCH.len()] {
    let mut out = [(0u64, usize::MAX); SEG_SHARED_SCRATCH.len()];
    for (li, lay) in lays.iter().enumerate() {
        for (si, seg) in SEG_SHARED_SCRATCH.iter().enumerate() {
            let bytes = lay.segment_bytes[*seg];
            if bytes > out[si].0 {
                out[si] = (bytes, li);
            }
        }
    }
    out
}

/// ⭐⭐ THE PREFILL FAMILY'S NON-RESIDENT CHARGE — what the build will allocate for the
/// family's rungs, which is what the census must count: every PRIVATE non-resident segment
/// per rung (seg3/seg4 — no family member shares those), plus ONE arena per shared scratch
/// segment at the family's widest extent ([`scratch_arena_owners`]).
///
/// The degenerate laws a charge must hold: a family of ONE is charged exactly its whole
/// non-resident sum (the arenas are its own regions, nothing is shared), and a family of many
/// is charged at most that whole sum — strictly less whenever two rungs use a shared segment.
/// Both are pinned by test.
pub fn family_non_resident_bytes(lays: &[&RungFacts]) -> u64 {
    lays.iter()
        .map(|lay| {
            lay.segment_bytes
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    *i != SEG_WEIGHT && *i != SEG_KV && !SEG_SHARED_SCRATCH.iter().any(|s| s == i)
                })
                .map(|(_, b)| *b)
                .sum::<u64>()
        })
        .sum::<u64>()
        + scratch_arena_owners(lays)
            .iter()
            .map(|(widest, _)| widest)
            .sum::<u64>()
}

/// The census's inputs — the rung lists exactly as both readers hold them. The ANCHOR is the
/// batched-prefill bundle the decode session is placed beside (the narrowest multi-token
/// prefill rung, or the single prefill bundle when the ladder has no multi-token rung); the
/// FAMILY is every other prefill rung and prefix rung that builds; the DECODE RUNGS are the
/// decode batch ladder's entries past one request.
pub struct CensusInputs<'a> {
    /// The decode owner bundle — the session every other rung borrows from.
    pub decode: &'a RungFacts,
    /// The batched-prefill anchor, when the model baked one.
    pub anchor: Option<&'a RungFacts>,
    /// The prefill family past the anchor: its plain rungs and its prefix rungs.
    pub family: &'a [&'a RungFacts],
    /// The decode batch rungs past one request — they share nothing, charged whole.
    pub decode_rungs: &'a [&'a RungFacts],
    /// The session count that takes its own copy of the spilled weight tail: every rung in
    /// the manifest's prefill and decode ladders, plus the decode owner. An UPPER BOUND from
    /// the declared lists (a rung may fail to build and be skipped): over-charging shrinks
    /// the pool, under-charging hands out memory that does not exist.
    pub borrowers: u64,
}

/// The census's terms — mandatory residency and the placement-luxury facts stated over it.
/// The load census keeps its policy decisions (the anchor's reserve-or-borrow, the funded
/// refusal) over THESE fields, never over a second derivation of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    /// The decode owner's non-KV segments plus its weight banks.
    pub decode_non_kv_bytes: u64,
    /// The spilled weight tail's bytes × borrowers.
    pub tail_charge: u64,
    /// The anchor's live segments (everything but its twin weight home and twin KV pool —
    /// its build is fatal, so these are mandatory).
    pub anchor_live_bytes: u64,
    /// The anchor's spare (a full twin of the weight home) — PLACEMENT LUXURY, never
    /// mandatory: taken only when the card funds it (see the load census's reserve policy).
    pub anchor_spare_bytes: u64,
    /// The ladder's rung charge: the prefill family's shared-scratch arenas plus the decode
    /// batch rungs' whole non-resident bytes.
    pub ladder_rung_bytes: u64,
}

impl Residency {
    /// MANDATORY RESIDENCY — what must be on the card before the pool takes its remainder:
    /// the decode owner, the spilled-tail copies, the anchor's live segments, and the ladder
    /// the spare has to survive. The anchor's SPARE is deliberately absent (luxury, not law).
    pub fn mandatory_bytes(&self) -> u64 {
        self.decode_non_kv_bytes
            + self.tail_charge
            + self.anchor_live_bytes
            + self.ladder_rung_bytes
    }
}

/// THE CENSUS — mandatory residency over the model's own baked bundles, the one computation
/// both the bake's descent and the load's funded check read. Every term is documented at the
/// [`RungFacts`] helper it is stated over; this function only adds them up.
pub fn mandatory_residency(c: &CensusInputs<'_>) -> Residency {
    let (anchor_live_bytes, anchor_spare_bytes) = match c.anchor {
        Some(a) => (a.non_resident_bytes(), {
            a.segment_bytes[SEG_WEIGHT] + a.weight_bank_bytes.iter().sum::<u64>()
        }),
        None => (0, 0),
    };
    Residency {
        decode_non_kv_bytes: c.decode.non_kv_bytes(),
        tail_charge: c.decode.tail_bytes() * c.borrowers,
        anchor_live_bytes,
        anchor_spare_bytes,
        ladder_rung_bytes: family_non_resident_bytes(c.family)
            + c.decode_rungs
                .iter()
                .map(|r| r.non_resident_bytes())
                .sum::<u64>(),
    }
}

/// ⭐ THE POOL LAW — the KV pool in WHOLE PAGES: every page the funded budget holds beyond
/// mandatory residency, FLOORED at the model's whole context.
///
/// Returns `None` when the funded budget cannot hold the model's whole context beside
/// mandatory residency — the bake's refusal (a compile-time check, by order: a model whose
/// context does not fund does not bake, it does not serve degraded). The floor is never the
/// cap: everything past it that the budget holds is the pool's, in whole pages.
pub fn pool_pages(
    funded_bytes: u64,
    mandatory_bytes: u64,
    page_stride_bytes: u64,
    context_pages: u32,
) -> Option<u32> {
    let stride = page_stride_bytes.max(1);
    let budget_pages = funded_bytes.saturating_sub(mandatory_bytes) / stride;
    if budget_pages < u64::from(context_pages) {
        return None;
    }
    Some(budget_pages.min(u32::MAX as u64) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(seg_bytes: SegBytes) -> RungFacts {
        RungFacts {
            segment_bytes: seg_bytes,
            ..RungFacts::default()
        }
    }

    fn whole_non_resident(f: &RungFacts) -> u64 {
        f.segment_bytes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != SEG_WEIGHT && *i != SEG_KV)
            .map(|(_, b)| *b)
            .sum()
    }

    /// One arena per shared segment (seg0/5/6), each at the family's widest extent and owned
    /// by the rung that holds it — possibly a DIFFERENT rung per segment, which is the whole
    /// point of picking per segment.
    #[test]
    fn the_arena_is_one_per_shared_segment_at_its_widest_rung() {
        // Rung 0: widest seg0. Rung 1: widest seg6. Rung 2: widest neither.
        let a = facts([100, 0, 0, 10, 0, 7, 3]);
        let b = facts([60, 0, 0, 20, 0, 5, 9]);
        let c = facts([40, 0, 0, 30, 0, 6, 4]);
        let fam = [&a, &b, &c];
        let owners = scratch_arena_owners(&fam);
        assert_eq!(
            SEG_SHARED_SCRATCH,
            [0, 5, 6],
            "the shared table is seg0/seg5/seg6"
        );
        assert_eq!(owners[0], (100, 0), "seg0's arena is rung 0's extent");
        assert_eq!(owners[1], (7, 0), "seg5's arena is rung 0's extent");
        assert_eq!(
            owners[2],
            (9, 1),
            "seg6's arena is rung 1's extent — per segment"
        );
    }

    /// An empty family mints no arena: nothing is charged, nothing is shared.
    #[test]
    fn an_empty_family_mints_no_arena() {
        let owners = scratch_arena_owners(&[]);
        assert!(
            owners
                .iter()
                .all(|&(bytes, oi)| bytes == 0 && oi == usize::MAX),
            "no rungs, no arenas: {owners:?}"
        );
        assert_eq!(family_non_resident_bytes(&[]), 0, "nothing is charged");
    }

    /// The degenerate family of ONE is charged exactly its whole non-resident sum: the arenas
    /// are its own regions and nothing is shared — the charge law reduces to the old one.
    #[test]
    fn a_family_of_one_is_charged_its_whole_non_resident_sum() {
        let a = facts([100, 999, 999, 10, 4, 7, 3]);
        let fam = [&a];
        assert_eq!(
            family_non_resident_bytes(&fam),
            whole_non_resident(&a),
            "one rung owns every arena, so it is charged everything it places"
        );
    }

    /// The charge is what the build allocates, never more: per-rung private segments plus one
    /// arena per shared segment is at most the whole per-rung sum, and strictly less the
    /// moment two rungs use a shared segment — which is the bytes this law exists to free.
    #[test]
    fn the_family_charge_is_at_most_the_whole_sum_and_strictly_less_when_shared() {
        let a = facts([100, 0, 0, 10, 0, 7, 3]);
        let b = facts([60, 0, 0, 20, 0, 5, 9]);
        let fam = [&a, &b];
        let whole: u64 = fam.iter().map(|l| whole_non_resident(l)).sum();
        let charged = family_non_resident_bytes(&fam);
        // Private (seg3) sums per rung: 10 + 20. Arenas: max(100,60) + max(7,5) + max(3,9).
        assert_eq!(
            charged,
            30 + 100 + 7 + 9,
            "private per rung + one arena each"
        );
        assert!(
            charged < whole,
            "two rungs sharing three segments frees {}/{} bytes",
            whole - charged,
            whole
        );
    }

    /// The census charges the decode owner's banks and the tail per borrower, the anchor's
    /// live segments, and the ladder — and nothing else: the anchor's SPARE is absent from
    /// mandatory residency because it is placement luxury.
    #[test]
    fn the_census_sums_its_terms_and_excludes_the_spare() {
        let decode = RungFacts {
            segment_bytes: [10, 1000, 5000, 10, 4, 0, 0],
            weight_bank_bytes: vec![300],
            kv_bank_bytes: vec![2000],
            weight_tail: WeightTail::Spilled {
                segment: 6,
                bytes: 25,
            },
        };
        let anchor = facts([50, 800, 4000, 5, 2, 0, 0]);
        let rung = facts([30, 0, 0, 7, 0, 3, 1]);
        let family = [&rung];
        let decode_rungs = [&rung];
        let res = mandatory_residency(&CensusInputs {
            decode: &decode,
            anchor: Some(&anchor),
            family: &family,
            decode_rungs: &decode_rungs,
            borrowers: 5,
        });
        // decode: non-KV segments (10+1000+10+4) + bank 300. tail: 25 × 5.
        assert_eq!(res.decode_non_kv_bytes, 10 + 1000 + 10 + 4 + 300);
        assert_eq!(res.tail_charge, 125);
        // anchor live: everything but its weight home and twin pool: 50+5+2.
        assert_eq!(res.anchor_live_bytes, 57);
        // The family of one and the decode rung are each charged their whole non-resident
        // sum: [30,0,0,7,0,3,1] → 30+7+0+3+1 = 41.
        assert_eq!(family_non_resident_bytes(&family), 41);
        // decode rung: whole non-resident = 30+7+0+3+1 = 41.
        assert_eq!(decode_rungs[0].non_resident_bytes(), 41);
        assert_eq!(res.ladder_rung_bytes, 41 + 41);
        assert_eq!(
            res.mandatory_bytes(),
            res.decode_non_kv_bytes
                + res.tail_charge
                + res.anchor_live_bytes
                + res.ladder_rung_bytes
        );
        // The spare is carried for the placement policy, never folded into mandatory
        // residency: `mandatory_bytes()` above does not name it.
        assert_eq!(
            res.anchor_spare_bytes, 800,
            "the spare is the twin weight home, the policy's to spend"
        );
    }

    /// The carve: 7 regions, 1 program, 16 GiB cap — the dd2 numbers.
    #[test]
    fn the_carve_matches_the_measured_card() {
        // 114,688 MB = 120,259,084,288 B: /7 = 17,179,869,184 > 16 GiB → capped.
        assert_eq!(
            region_bytes(<Baked as DeviceGen>::MEMORY_BYTES),
            <Baked as DeviceGen>::MAX_SEGMENT_BYTES
        );
        // 6 tensor regions × 16 GiB = 96 GiB — the number the allocator's own OOM diagnostic
        // reported where a whole-card budget printed 114,688 MB.
        assert_eq!(
            tensor_capacity(<Baked as DeviceGen>::MEMORY_BYTES),
            6 * <Baked as DeviceGen>::MAX_SEGMENT_BYTES
        );
        // A small card is NOT capped: the equal share is what one region holds.
        assert_eq!(region_bytes(7 * 1024), 1024);
        assert_eq!(tensor_capacity(7 * 1024), 6 * 1024);
    }

    /// THE LAW ITSELF: every whole page the funded budget holds beyond mandatory residency,
    /// floored at the model's whole context — and the refusal when the floor does not fund.
    #[test]
    fn the_pool_takes_the_remainder_and_refuses_an_unfundable_context() {
        let stride = 20 * 1024 * 1024;
        // granite-8b-shaped: funded 88,473 MB − mandatory 15,975 MB = 72,498 MB of remainder.
        let funded = 88_473 * 1024 * 1024;
        let mandatory = 15_975 * 1024 * 1024;
        let pages = pool_pages(funded, mandatory, stride, 1296).expect("the floor funds");
        assert_eq!(
            u64::from(pages),
            (funded - mandatory) / stride,
            "every whole page"
        );
        // The stride itself is the pool's whole bytes over its pages — one law, both ends.
        assert_eq!(page_stride_bytes(pages as u64 * stride, pages), stride);
        assert_eq!(
            page_stride_bytes(pages as u64 * stride, 0),
            0,
            "no pages, no pool"
        );
        // A context the budget cannot fund is None — the bake's refusal, never a clamp.
        assert!(pool_pages(funded, mandatory, stride, 100_000).is_none());
        // The floor is never the cap: a tiny context still takes the whole remainder.
        assert_eq!(
            pool_pages(funded, mandatory, stride, 1).expect("funds"),
            pages
        );
        // Mandatory past funded: nothing left, and a one-page context still refuses.
        assert!(pool_pages(mandatory - 1, mandatory, stride, 1).is_none());
    }
}
