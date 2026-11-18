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
//! * **THE LOAD CENSUS** (`spyre_load.rs`) measures the REAL card — the one term the bake
//!   cannot know — and checks it against the CARRIED census ([`BakedCensus`]): the bake's own
//!   terms (mandatory residency, page stride, the utilization it funded at), minted once and
//!   read back, never re-derived.
//!
//! The census's terms are stated here over [`RungFacts`] — the projection of the bake's
//! compile-time [`crate::placement::BundleLayout`] that the census reads: segment bytes, the
//! weight and KV segments' extra banks, and the weight-tail verdict.
//!
//! # The funded budget is a compile-time fact, not a probe
//!
//! Every term is a bake fact: the card's memory is [`DeviceGen::MEMORY_BYTES`] (the gen1
//! platform table's own row, measured on dd2), the carve into allocatable tensor capacity is
//! [`tensor_capacity`] (regions × tensor regions, one region reserved for program memory), and
//! the fraction is [`GpuMemoryUtilization::DEFAULT`] — the serving config's own one constant,
//! consumed here, never restated. Nothing reads the card at boot to size the pool — that is
//! torch's free-memory probe, and it is not what this is. The utilization is a BAKE INPUT:
//! the pool is baked at the fraction the config states, and the load checks the measured card
//! against that same baked budget ([`funded_bytes`] over [`BakedCensus::utilization`]).
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

/// A byte quantity of the capacity law — a funded budget, a residency term, a pool. Newtyped
/// so no bare `u64` can stand in for one of them (or for a page count) at any boundary this
/// module owns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

/// A count of whole KV pool pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pages(pub u32);

/// The pool's whole bytes over its own page count. ONE derivation owns this number — the
/// KTIR lowering's own division of the summed per-launch KV spans by the paged step's page
/// count (`lower_subtile_tape_to_ktir`), carried by the census and by the bundle's reroll
/// meta; neither census end ever re-derives it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageStride(pub u64);

/// The fraction of the allocatable tensor capacity a bake funds — a BAKE INPUT: the pool is
/// baked at this fraction, and the load checks the measured card against the same budget
/// ([`BakedCensus::utilization`] carries the fraction the bake actually used).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMemoryUtilization(pub f64);

impl GpuMemoryUtilization {
    /// The one fraction every bake that names nothing else funds at — the serving config's
    /// own constant, consumed here and never restated.
    pub const DEFAULT: Self = Self(scratchy_core_config::cache::DEFAULT_GPU_MEMORY_UTILIZATION);
}

/// ⭐ THE PREFILL FAMILY'S SHARED-SCRATCH SEGMENTS — the segments whose scratch ONE family of
/// mutually-exclusive launches shares (one arena per segment at the family's widest extent):
/// INTERMEDIATE (seg0) plus the two intermediate color segments (seg5/seg6).
///
/// ⛔ THE EXCLUSION LAW that licenses sharing at all: prefill-shaped launches COLLECT IN-CALL,
/// one at a time, serialized by the executor — a launch's scratch is dead before the next
/// family member's begins — while the DECODE launch is submitted-then-deferred and genuinely
/// overlaps a prefill launch, so decode's scratch stays private and no decode segment appears
/// here.
///
/// seg5/seg6 are the coloring pass's two spare intermediate colors, not [`crate::placement::SegRole`]
/// variants — they are named by this enum because that is what a layout's `segment_bytes` is
/// indexed by. The enum is the ONE table: its [`SharedScratchSeg::index`] is the only place
/// the segment ids are stated, and every reader (this law's census arithmetic, the executor's
/// arena sharing, the load wave's borrower marking) consumes the enum — a segment that is not
/// a shared-scratch segment is not nameable at those boundaries at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedScratchSeg {
    /// seg0 — activations and the online-softmax state.
    Intermediate,
    /// seg5 — the coloring pass's first spare intermediate colour.
    ColorA,
    /// seg6 — the coloring pass's second spare intermediate colour.
    ColorB,
}

/// Every shared-scratch segment, in segment order — the census's and the wave's one list.
pub const SHARED_SCRATCH_SEGS: [SharedScratchSeg; 3] = [
    SharedScratchSeg::Intermediate,
    SharedScratchSeg::ColorA,
    SharedScratchSeg::ColorB,
];

impl SharedScratchSeg {
    /// The segment id a layout's `segment_bytes` is indexed by — the ONE statement of the
    /// table; the card crate renders this into its own `SegIdx` by const evaluation of the
    /// same call, never by a second table.
    pub const fn index(self) -> usize {
        match self {
            SharedScratchSeg::Intermediate => crate::placement::SegRole::Intermediate as usize,
            SharedScratchSeg::ColorA => 5,
            SharedScratchSeg::ColorB => 6,
        }
    }
}

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
/// bake computes it from [`DeviceGen::MEMORY_BYTES`] and [`GpuMemoryUtilization::DEFAULT`];
/// the load computes it from the MEASURED card and the utilization the census carries. One
/// arithmetic.
pub fn funded_bytes(card_bytes: u64, utilization: GpuMemoryUtilization) -> Bytes {
    Bytes((tensor_capacity(card_bytes) as f64 * utilization.0) as u64)
}

/// One rung's census facts — the projection of the bake's compile-time
/// `placement::BundleLayout` that every term of the capacity law reads. A projection, not a
/// copy: each field is read from the layout that already owns it, never re-derived.
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

/// One shared-scratch segment's family arena, as the census states it: the family's WIDEST
/// extent for that segment and the family entry that holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arena {
    /// The widest extent any family rung places in the segment.
    pub bytes: u64,
    /// The index of the family entry that holds the widest extent — the arena's owner.
    pub owner: usize,
}

/// ⭐⭐ THE PREFILL FAMILY'S SHARED-SCRATCH ARENAS — ONE computation, run ONCE at the bake by
/// the census (which charges the bytes the build will allocate) and carried in
/// [`BakedCensus::arena_owners`] for the load wave (which marks every non-owner a borrower).
/// Every shared segment ([`SHARED_SCRATCH_SEGS`]) gets ONE arena at the family's WIDEST
/// extent: the prefill family's launches are collected in-call one at a time (the exclusion
/// law on that table), so the widest rung's region can back every narrower one, and each
/// rung addresses its own bake-proven placement inside its own `segment_bytes`.
///
/// Returns, per shared segment (position ↔ [`SHARED_SCRATCH_SEGS`]), the arena — `None` when
/// the family places nothing in that segment: no arena is charged and no rung borrows it.
pub fn scratch_arena_owners(lays: &[&RungFacts]) -> [Option<Arena>; SHARED_SCRATCH_SEGS.len()] {
    let mut out: [Option<Arena>; SHARED_SCRATCH_SEGS.len()] = [None; SHARED_SCRATCH_SEGS.len()];
    for (li, lay) in lays.iter().enumerate() {
        for (si, seg) in SHARED_SCRATCH_SEGS.iter().enumerate() {
            let bytes = lay.segment_bytes[seg.index()];
            if bytes > out[si].map_or(0, |a| a.bytes) {
                out[si] = Some(Arena { bytes, owner: li });
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
                    *i != SEG_WEIGHT
                        && *i != SEG_KV
                        && !SHARED_SCRATCH_SEGS.iter().any(|s| s.index() == *i)
                })
                .map(|(_, b)| *b)
                .sum::<u64>()
        })
        .sum::<u64>()
        + scratch_arena_owners(lays)
            .iter()
            .map(|arena| arena.map_or(0, |a| a.bytes))
            .sum::<u64>()
}

/// The census's inputs — the rung lists exactly as the manifest holds them, resolved ONCE at
/// the bake. The ANCHOR is the batched-prefill bundle the decode session is placed beside
/// (the narrowest multi-token prefill rung, or the single prefill bundle when the ladder has
/// no multi-token rung); the FAMILY is every other prefill rung and prefix rung that builds;
/// the DECODE RUNGS are the decode batch ladder's entries past one request.
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
/// Minted ONCE by the bake ([`mandatory_residency`]) and carried in [`BakedCensus`]; the
/// load reads them and combines only the terms the bake cannot know (the measured card).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    /// The decode owner's non-KV segments plus its weight banks.
    pub decode_non_kv_bytes: Bytes,
    /// The spilled weight tail's bytes × borrowers.
    pub tail_charge: Bytes,
    /// The anchor's live segments (everything but its twin weight home and twin KV pool —
    /// its build is fatal, so these are mandatory).
    pub anchor_live_bytes: Bytes,
    /// The anchor's spare (a full twin of the weight home) — PLACEMENT LUXURY, never
    /// mandatory: the pool takes every page beyond mandatory residency, so a budget that
    /// funds the pool has nothing left to fund the spare with. Carried as the census's own
    /// statement of what was deliberately NOT charged, never as a load-time decision.
    pub anchor_spare_bytes: Bytes,
    /// The ladder's rung charge: the prefill family's shared-scratch arenas plus the decode
    /// batch rungs' whole non-resident bytes.
    pub ladder_rung_bytes: Bytes,
}

impl Residency {
    /// MANDATORY RESIDENCY — what must be on the card before the pool takes its remainder:
    /// the decode owner, the spilled-tail copies, the anchor's live segments, and the ladder.
    /// The anchor's SPARE is deliberately absent (luxury, not law).
    pub fn mandatory_bytes(&self) -> Bytes {
        Bytes(
            self.decode_non_kv_bytes.0
                + self.tail_charge.0
                + self.anchor_live_bytes.0
                + self.ladder_rung_bytes.0,
        )
    }
}

/// THE CENSUS — mandatory residency over the model's own baked bundles, computed ONCE at the
/// bake (the load reads the carried [`BakedCensus`], it never re-derives these terms). Every
/// term is documented at the [`RungFacts`] helper it is stated over; this function only adds
/// them up.
pub fn mandatory_residency(c: &CensusInputs<'_>) -> Residency {
    let (anchor_live_bytes, anchor_spare_bytes) = match c.anchor {
        Some(a) => (
            Bytes(a.non_resident_bytes()),
            Bytes(a.segment_bytes[SEG_WEIGHT] + a.weight_bank_bytes.iter().sum::<u64>()),
        ),
        None => (Bytes(0), Bytes(0)),
    };
    Residency {
        decode_non_kv_bytes: Bytes(c.decode.non_kv_bytes()),
        tail_charge: Bytes(c.decode.tail_bytes() * c.borrowers),
        anchor_live_bytes,
        anchor_spare_bytes,
        ladder_rung_bytes: Bytes(
            family_non_resident_bytes(c.family)
                + c.decode_rungs
                    .iter()
                    .map(|r| r.non_resident_bytes())
                    .sum::<u64>(),
        ),
    }
}

/// ⭐ THE WHOLE PAGES a funded budget holds beyond mandatory residency — the ONE combination
/// of the law's terms. The bake's descent takes its pool from it (floored at the model's
/// whole context); the load's funded check compares the baked pool against it over the
/// MEASURED card. One arithmetic, two readers, stated once here.
pub fn budget_pages(funded: Bytes, mandatory: Bytes, stride: PageStride) -> Pages {
    Pages((funded.0.saturating_sub(mandatory.0) / stride.0.max(1)).min(u32::MAX as u64) as u32)
}

/// Whether a funded budget holds a baked pool beside mandatory residency — the load's check
/// over the measured card, stated over the SAME combination the bake sized the pool by
/// ([`budget_pages`]). A card that reports less than the platform table fails here.
pub fn funded_holds(funded: Bytes, mandatory: Bytes, stride: PageStride, pool: Pages) -> bool {
    pool <= budget_pages(funded, mandatory, stride)
}

/// ⭐ THE POOL LAW — the KV pool in WHOLE PAGES: every page the funded budget holds beyond
/// mandatory residency, FLOORED at the model's whole context.
///
/// Returns `None` when the funded budget cannot hold the model's whole context beside
/// mandatory residency — the bake's refusal (a compile-time check, by order: a model whose
/// context does not fund does not bake, it does not serve degraded). The floor is never the
/// cap: everything past it that the budget holds is the pool's, in whole pages.
pub fn pool_pages(
    funded: Bytes,
    mandatory: Bytes,
    stride: PageStride,
    context_pages: Pages,
) -> Option<Pages> {
    let budget = budget_pages(funded, mandatory, stride);
    if budget < context_pages {
        return None;
    }
    Some(budget)
}

/// ⭐⭐⭐ THE BAKED CENSUS — the capacity law's terms, minted ONCE by the bake from the
/// manifest's own rung lists and carried in the bundle for the load to READ. The load
/// re-derives nothing on this list; the only terms it combines are the ones the bake cannot
/// know — the MEASURED card's bytes, through [`funded_bytes`] and [`funded_holds`].
///
/// The census names the family by FINGERPRINT, not by position: the load wave marks a rung a
/// borrower by comparing its bundle's fp against [`Self::arena_owners`], so no positional
/// coupling between the bake's list and the load's build order exists to drift.
pub struct BakedCensus {
    /// The decode owner's bundle fingerprint — the session every other rung borrows from.
    pub decode_fp: &'static str,
    /// The batched-prefill anchor's fingerprint, when the model baked one; the load resolves
    /// the anchor bundle by it (the anchor selection is stated once, at the bake).
    pub anchor_fp: Option<&'static str>,
    /// The prefill family past the anchor — its plain rungs and its prefix rungs — as the
    /// census resolved it. The load's ladder and prefix walks keep only the rungs on this
    /// list.
    pub family_fps: &'static [&'static str],
    /// The pool's page stride — the KTIR lowering's own division, carried; see
    /// [`PageStride`].
    pub page_stride: PageStride,
    /// The utilization the bake funded the pool at — a bake input, carried so the load
    /// checks the measured card against the SAME budget the pool was sized by.
    pub utilization: GpuMemoryUtilization,
    /// Mandatory residency — the law's own terms ([`Residency`]), minted once at the bake.
    pub residency: Residency,
    /// Per shared segment (position ↔ [`SHARED_SCRATCH_SEGS`]), the family entry that owns
    /// the arena — the census's own `scratch_arena_owners`, rendered as the owner's
    /// fingerprint. `None` mints no arena and marks no borrower.
    pub arena_owners: [Option<&'static str>; SHARED_SCRATCH_SEGS.len()],
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
            SHARED_SCRATCH_SEGS.map(|s| s.index()),
            [0, 5, 6],
            "the shared table is seg0/seg5/seg6"
        );
        assert_eq!(
            owners[0],
            Some(Arena {
                bytes: 100,
                owner: 0
            }),
            "seg0's arena is rung 0's extent"
        );
        assert_eq!(
            owners[1],
            Some(Arena { bytes: 7, owner: 0 }),
            "seg5's arena is rung 0's extent"
        );
        assert_eq!(
            owners[2],
            Some(Arena { bytes: 9, owner: 1 }),
            "seg6's arena is rung 1's extent — per segment"
        );
    }

    /// An empty family mints no arena: nothing is charged, nothing is shared.
    #[test]
    fn an_empty_family_mints_no_arena() {
        let owners = scratch_arena_owners(&[]);
        assert!(
            owners.iter().all(|a| a.is_none()),
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
        assert_eq!(res.decode_non_kv_bytes, Bytes(10 + 1000 + 10 + 4 + 300));
        assert_eq!(res.tail_charge, Bytes(125));
        // anchor live: everything but its weight home and twin pool: 50+5+2.
        assert_eq!(res.anchor_live_bytes, Bytes(57));
        // The family of one and the decode rung are each charged their whole non-resident
        // sum: [30,0,0,7,0,3,1] → 30+7+0+3+1 = 41.
        assert_eq!(family_non_resident_bytes(&family), 41);
        // decode rung: whole non-resident = 30+7+0+3+1 = 41.
        assert_eq!(decode_rungs[0].non_resident_bytes(), 41);
        assert_eq!(res.ladder_rung_bytes, Bytes(41 + 41));
        assert_eq!(
            res.mandatory_bytes(),
            Bytes(
                res.decode_non_kv_bytes.0
                    + res.tail_charge.0
                    + res.anchor_live_bytes.0
                    + res.ladder_rung_bytes.0
            )
        );
        // The spare is carried as the census's own statement of what was NOT charged, never
        // folded into mandatory residency: `mandatory_bytes()` above does not name it.
        assert_eq!(
            res.anchor_spare_bytes,
            Bytes(800),
            "the spare is the twin weight home — never charged"
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
    /// The load's funded check is the SAME combination (`budget_pages`), pinned here over a
    /// budget the pool does fit and one it does not.
    #[test]
    fn the_pool_takes_the_remainder_and_refuses_an_unfundable_context() {
        let stride = PageStride(20 * 1024 * 1024);
        // granite-8b-shaped: funded 88,473 MB − mandatory 15,975 MB = 72,498 MB of remainder.
        let funded = Bytes(88_473 * 1024 * 1024);
        let mandatory = Bytes(15_975 * 1024 * 1024);
        let pages = pool_pages(funded, mandatory, stride, Pages(1296)).expect("the floor funds");
        assert_eq!(
            pages,
            Pages(((funded.0 - mandatory.0) / stride.0) as u32),
            "every whole page"
        );
        // The load's check over the SAME budget holds for the baked pool — one combination.
        assert!(funded_holds(funded, mandatory, stride, pages));
        // A card that reports less than the bake funded fails the same check: 4 GiB less is
        // ~200 pages short.
        let smaller = Bytes(funded.0 - 4 * 1024 * 1024 * 1024);
        assert!(!funded_holds(smaller, mandatory, stride, pages));
        // A context the budget cannot fund is None — the bake's refusal, never a clamp.
        assert!(pool_pages(funded, mandatory, stride, Pages(100_000)).is_none());
        // The floor is never the cap: a tiny context still takes the whole remainder.
        assert_eq!(
            pool_pages(funded, mandatory, stride, Pages(1)).expect("funds"),
            pages
        );
        // Mandatory past funded: nothing left, and a one-page context still refuses.
        assert!(pool_pages(Bytes(mandatory.0 - 1), mandatory, stride, Pages(1)).is_none());
    }
}
