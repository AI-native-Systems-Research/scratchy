// SPDX-License-Identifier: Apache-2.0
//! THE RESERVED TID SPACE — the tensor ids no model produces, declared, bounded, and proven
//! disjoint at COMPILE TIME.
//!
//! ⭐ THIS IS THE PRODUCER↔LOWERING ABI, which is why it belongs in this crate rather than behind a
//! trait. Each id names a tensor the lowering invents or requires and the caller binds: a rung's
//! scratch, the stick constants its descriptors read, the paged step's host tables, a packed
//! prefill's window tables, the tensor-parallel receive window. A third-party KTIR producer has to
//! agree with these numbers to bind anything, so they are part of the interface.
//!
//! ⛔ AND THE ARITHMETIC IS THE POINT. Every id is a slot of a
//! [`TidRegion`](crate::reserved_tids::TidRegion), every region is in
//! [`RESERVED_REGIONS`](crate::reserved_tids::RESERVED_REGIONS), and the regions are proven
//! disjoint at compile time — two tensors on one tid would share one placement, on a device with
//! no stack to attach the fault to.

/// ⭐ THE RESERVED RANGE'S OWN FLOOR — tensor ids at or above this are the producer's one-off
/// sentinels, never kernel parameters: the runner that decides "reserved vs parameter" reads it
/// here.
pub const RESERVED_TID_FLOOR: u32 = u32::MAX - 1024;

/// RESERVED tensor id for the attention's additive PREFIX length-mask `[mq, cap]` (0 on valid
/// prefix columns, −inf elsewhere): a per-step activation the worker binds, shared across layers.
pub const ATTN_MASK_TID: u32 = u32::MAX - 3;

/// RESERVED tensor id for a rung's scratch: the one device-only region every program's
/// intermediates are packed into, in the intermediate segment.
pub const SCRATCH_TID: u32 = u32::MAX - 6;

/// RESERVED tensor id for a prefill rung's ANSWER ROWS `[S, hidden]` — each request's last prompt
/// row of the final-norm output, extracted on card so the lm-head tail runs at the answer rows
/// inside the prefill bundle (`lmlast`). Written and read inside the rung; never worker-bound.
pub const LAST_HIDDEN_TID: u32 = u32::MAX;

/// One reserved region of the tid space: a base and how many slots it owns, counting DOWN. A
/// region cannot be added or resized without [`RESERVED_REGIONS_ARE_DISJOINT`] re-proving the
/// whole layout, and no index can leave its region without [`Self::at`] refusing during the bake.
#[derive(Clone, Copy, Debug)]
pub struct TidRegion {
    /// For diagnostics — named so a refusal says WHICH region overflowed.
    pub name: &'static str,
    /// Highest tid in the region; slots count downward from here.
    pub base: u32,
    pub slots: u32,
}

impl TidRegion {
    /// Lowest tid this region owns.
    pub const fn floor(self) -> u32 {
        self.base - (self.slots - 1)
    }

    /// The `idx`-th tid, or a BUILD FAILURE. `idx >= slots` would silently alias the region below.
    pub const fn at(self, idx: u32) -> u32 {
        assert!(
            idx < self.slots,
            "reserved tid region overflow: this index is past the region's last slot and would \
             alias the region below it, giving two tensors one placement",
        );
        self.base - idx
    }

    const fn overlaps(self, other: TidRegion) -> bool {
        self.floor() <= other.base && other.floor() <= self.base
    }
}

/// ⭐⭐⭐ THE PAGED STEP's HOST SOURCES — spyre-inference's own attention operands besides the pools:
/// the mask stack, the page index table, the KV write index, the query row index, and a decode
/// batch's `rep_row_ids`, `chunk_page_ids` and `mask_by_chunk`. A resident session holds the decode
/// and the prefill programs, whose graphs differ in size, so a source numbered past one graph could
/// be a real tensor of the other: the reserved region names the same ids for every bundle, and no
/// graph produces one. Slots 4 and 8 are held, unnamed, so the regions below keep their ids.
pub const PAGED_STEP_BASE: u32 = u32::MAX - 2_000_171;

/// The slots one attention class's paged step owns.
pub const PAGED_STEP_SLOTS: u32 = 9;

/// How many attention classes the paged step's tables are reserved for: a uniform model's one, and
/// a hybrid model's local (sliding) layers beside it — each class's step its own tables, since its
/// kv heads, head dim and mask are its own.
pub const PAGED_CLASSES: u32 = 2;

/// ⭐ CLASS `class`'s COPY OF PAGED-STEP SLOT `slot`. **Class 0's are the `paged_step` sentinels**
/// (`PAGED_MASK_TID` ..); every later class takes the same slot of its own block of the
/// `paged_class` region.
pub const fn paged_class_tid(class: u32, slot: u32) -> u32 {
    assert!(
        class < PAGED_CLASSES,
        "an attention class past PAGED_CLASSES"
    );
    assert!(
        slot < PAGED_STEP_SLOTS,
        "a paged-step slot past the step's own"
    );
    match class {
        0 => reserved_region("paged_step").at(slot),
        c => reserved_region("paged_class").at((c - 1) * PAGED_STEP_SLOTS + slot),
    }
}

/// The paged step's mask stack — see [`PAGED_STEP_BASE`].
pub const PAGED_MASK_TID: u32 = reserved_region("paged_step").at(0);
/// The paged step's page table — see [`PAGED_STEP_BASE`].
pub const PAGED_PAGE_TABLE_TID: u32 = reserved_region("paged_step").at(1);
/// The paged step's KV write index — see [`PAGED_STEP_BASE`].
pub const PAGED_WRITE_ROWS_TID: u32 = reserved_region("paged_step").at(2);
/// The paged step's query row index — see [`PAGED_STEP_BASE`].
pub const PAGED_QUERY_ROWS_TID: u32 = reserved_region("paged_step").at(3);
/// A decode batch's `rep_row_ids` — see [`PAGED_STEP_BASE`].
pub const PAGED_REP_ROWS_TID: u32 = reserved_region("paged_step").at(5);
/// A decode batch's `chunk_page_ids` — see [`PAGED_STEP_BASE`].
pub const PAGED_CHUNK_PAGES_TID: u32 = reserved_region("paged_step").at(6);
/// A decode batch's `mask_by_chunk` — see [`PAGED_STEP_BASE`].
pub const PAGED_CHUNK_MASKS_TID: u32 = reserved_region("paged_step").at(7);

/// Every reserved region, in one place. Order is high tid → low.
pub const RESERVED_REGIONS: [TidRegion; 6] = [
    // The one-off sentinels: the attention mask (MAX-3) and the scratch (MAX-6).
    TidRegion {
        name: "sentinels",
        base: u32::MAX - 1,
        slots: 6,
    },
    // ⭐ THE STICK CONSTANTS — one slot per value a rung's descriptors read, the slot keyed by
    // the value itself (see [`stick_const_tid`]).
    TidRegion {
        name: "stick_const",
        base: u32::MAX - 1_000_171,
        slots: 1_000_000,
    },
    // ⭐ THE PAGED STEP's host sources — see [`PAGED_STEP_BASE`].
    TidRegion {
        name: "paged_step",
        base: PAGED_STEP_BASE,
        slots: PAGED_STEP_SLOTS,
    },
    // ⭐ THE TENSOR-PARALLEL RECEIVE WINDOW'S BLOCKS — see [`tp_recv_block_tid`].
    TidRegion {
        name: "tp_recv",
        base: TP_RECV_TID,
        slots: TP_WINDOW_CUTS * TP_MAX_WORLD,
    },
    // ⭐ THE PACKED PREFILL'S WINDOW TABLES — see [`paged_window_tid`]. The FIRST window's tids are
    // the `paged_step` sentinels (window 0 IS the step's own tables), so the region holds only the
    // windows PAST the first, three tables each, bounded by [`PAGED_MAX_WINDOWS`].
    TidRegion {
        name: "paged_window",
        base: PAGED_WINDOW_BASE,
        slots: PAGED_WINDOW_TABLES * (PAGED_MAX_WINDOWS - 1) * PAGED_CLASSES,
    },
    // ⭐ THE LATER ATTENTION CLASSES' STEP TABLES — see [`paged_class_tid`]. Class 0's are the
    // `paged_step` sentinels; the region holds every later class's copy of the same slots.
    TidRegion {
        name: "paged_class",
        base: PAGED_CLASS_BASE,
        slots: PAGED_STEP_SLOTS * (PAGED_CLASSES - 1),
    },
];

/// ⭐ RESERVED tid for the TENSOR-PARALLEL RECEIVE WINDOW — its first block
/// ([`tp_recv_block_tid`]), and the base of the one `(segment, offset, size)` triple per card
/// (`scratchy_spyre_bundle::collective::RecvWindow`) whose blocks every collective's consume reads.
/// The first tids below `paged_step`'s floor.
pub const TP_RECV_TID: u32 = PAGED_STEP_BASE - PAGED_STEP_SLOTS;

/// How many disjoint cuts the receive window is. The forward's `k`-th collective uses cut
/// `k % TP_WINDOW_CUTS` (`collective::WindowCut`), so a peer's send for collective `k + 1` never
/// lands where this rank still reads `k`.
pub const TP_WINDOW_CUTS: u32 = 2;

/// The widest world the window's tids are reserved for — the card's 64 write-done sids over the
/// cuts, one landing slot per (sender, cut) at the least (`collective::MAX_WORLD`, which pins
/// itself equal).
pub const TP_MAX_WORLD: u32 = 64 / TP_WINDOW_CUTS;

/// ⭐ THE RECEIVE WINDOW'S BLOCKS — one tid per (cut, sending rank) of the ONE window: rank `s`'s
/// block of cut `c`, placed at the cut's base plus `s` block-widths (`compute_bundle_layout`). A
/// consume reads sender `s`'s fragment as this tensor from its first byte, so every sender's block
/// is addressed the same way at every world width. Every block is a placement OF the one window,
/// never a second window that could disagree with it.
pub const fn tp_recv_block_tid(cut: u32, sender: u32) -> u32 {
    assert!(cut < TP_WINDOW_CUTS, "the window has TP_WINDOW_CUTS cuts");
    assert!(
        sender < TP_MAX_WORLD,
        "a sender past the widest reserved world"
    );
    TP_RECV_TID - (cut * TP_MAX_WORLD + sender)
}

/// Base of the RESERVED tid block for a PACKED PREFILL's window tables — the first tid below
/// `tp_recv`'s floor.
pub const PAGED_WINDOW_BASE: u32 = TP_RECV_TID - (TP_WINDOW_CUTS * TP_MAX_WORLD);

/// The later attention classes' step tables — see [`paged_class_tid`]. Below `paged_window`'s floor.
pub const PAGED_CLASS_BASE: u32 =
    PAGED_WINDOW_BASE - PAGED_WINDOW_TABLES * (PAGED_MAX_WINDOWS - 1) * PAGED_CLASSES;

/// How many per-request tables a packed prefill window owns: the query row index, the page index
/// table, and the mask stack — the same three the `paged_step` sentinels name for window 0.
pub const PAGED_WINDOW_TABLES: u32 = 3;

/// The widest packed capacity the window tids are reserved for. The bake's packed rung table is
/// bounded by this the way `tp_recv`'s slots are bounded by [`TP_MAX_WORLD`]: a pair asking for
/// more windows than this is a bake bug the tid mint's own assert names, never a wrap onto
/// another region's tids.
pub const PAGED_MAX_WINDOWS: u32 = 8;

/// ⭐ A PACKED PREFILL WINDOW'S TABLES — attention class `class`'s window `r` copy of one of the
/// three per-request tables the attention reads. **Window 0 IS the class's own step tables**
/// ([`paged_class_tid`]): class 0's window 0 answers `PAGED_QUERY_ROWS_TID` / `PAGED_PAGE_TABLE_TID`
/// / `PAGED_MASK_TID`, so a one-window (solo) rung binds the step's own tables. The windows
/// past the first count DOWN from [`PAGED_WINDOW_BASE`], three tids each, class by class, bounded by
/// [`PAGED_MAX_WINDOWS`].
///
/// Which table a window's slot names:
/// `0` the query row index, `1` the page index table, `2` the mask stack.
pub const fn paged_window_tid(class: u32, window: u32, table: u32) -> u32 {
    assert!(table < PAGED_WINDOW_TABLES, "a window has three tables");
    assert!(
        class < PAGED_CLASSES,
        "an attention class past PAGED_CLASSES"
    );
    match window {
        0 => paged_class_tid(
            class,
            match table {
                0 => 3,
                1 => 1,
                _ => 0,
            },
        ),
        w => {
            assert!(
                w < PAGED_MAX_WINDOWS,
                "a packed rung asked for a window past PAGED_MAX_WINDOWS — the bake's packed \
                 table outgrew the reserved tid space"
            );
            PAGED_WINDOW_BASE
                - ((class * (PAGED_MAX_WINDOWS - 1) + w - 1) * PAGED_WINDOW_TABLES + table)
        }
    }
}

/// The region named `name` — the ONE way a reader reaches a region, so inserting a region never
/// repoints a reader the way indexing the table by position would.
pub const fn reserved_region(name: &str) -> TidRegion {
    let want = name.as_bytes();
    let mut i = 0;
    while i < RESERVED_REGIONS.len() {
        let have = RESERVED_REGIONS[i].name.as_bytes();
        if have.len() == want.len() {
            let mut k = 0;
            let mut same = true;
            while k < have.len() {
                if have[k] != want[k] {
                    same = false;
                }
                k += 1;
            }
            if same {
                return RESERVED_REGIONS[i];
            }
        }
        i += 1;
    }
    panic!("no reserved tid region by that name — a region was renamed or removed");
}

// Naming each proof is what makes the compiler EVALUATE it: an unused `const` item is not
// const-evaluated, so a lock nothing mentions is a lock that never runs.
const _: () = SENTINELS_ARE_INSIDE_THEIR_REGION;
const _: () = RESERVED_REGIONS_ARE_DISJOINT;
const _: () = EVERY_NAMED_REGION_RESOLVES;
const _: () = PAGED_WINDOW_0_IS_THE_SENTINELS;

/// Every region a reader names resolves, at the base its door states.
pub const EVERY_NAMED_REGION_RESOLVES: () = {
    assert!(reserved_region("sentinels").base == u32::MAX - 1);
    assert!(reserved_region("stick_const").slots > 0);
    assert!(reserved_region("paged_step").base == PAGED_STEP_BASE);
    assert!(reserved_region("tp_recv").base == TP_RECV_TID);
    assert!(reserved_region("paged_window").base == PAGED_WINDOW_BASE);
    assert!(reserved_region("paged_class").base == PAGED_CLASS_BASE);
};

/// Window 0 of class 0 is the paged step's own tables, and the window and class regions' floors
/// hold exactly the last tid their mints admit.
pub const PAGED_WINDOW_0_IS_THE_SENTINELS: () = {
    assert!(paged_window_tid(0, 0, 0) == PAGED_QUERY_ROWS_TID);
    assert!(paged_window_tid(0, 0, 1) == PAGED_PAGE_TABLE_TID);
    assert!(paged_window_tid(0, 0, 2) == PAGED_MASK_TID);
    // The region's floor holds the LAST class's widest window's LAST table, and nothing past it.
    let r = reserved_region("paged_window");
    assert!(
        paged_window_tid(
            PAGED_CLASSES - 1,
            PAGED_MAX_WINDOWS - 1,
            PAGED_WINDOW_TABLES - 1
        ) == r.floor(),
        "the widest window's last table is not the paged_window region's floor — the region \
         does not hold every window the tid mint admits"
    );
    // And the last class's step tables end at the paged_class region's floor.
    assert!(
        paged_class_tid(PAGED_CLASSES - 1, PAGED_STEP_SLOTS - 1)
            == reserved_region("paged_class").floor()
    );
};

/// The one-off sentinels sit inside their region, and the region sits above the stick constants.
pub const SENTINELS_ARE_INSIDE_THEIR_REGION: () = {
    let r = reserved_region("sentinels");
    assert!(r.base == u32::MAX - 1, "sentinels start at MAX-1");
    assert!(
        ATTN_MASK_TID <= r.base && ATTN_MASK_TID >= r.floor(),
        "the attention mask has left the sentinel region"
    );
    assert!(
        SCRATCH_TID <= r.base && SCRATCH_TID >= r.floor(),
        "the scratch has left the sentinel region"
    );
    assert!(
        STICK_CONST_BASE < r.floor(),
        "the stick-const region overlaps the one-off sentinels",
    );
};

/// Every region is disjoint from every other, and none wraps past zero.
pub const RESERVED_REGIONS_ARE_DISJOINT: () = {
    let mut i = 0;
    while i < RESERVED_REGIONS.len() {
        let mut j = i + 1;
        while j < RESERVED_REGIONS.len() {
            assert!(
                !RESERVED_REGIONS[i].overlaps(RESERVED_REGIONS[j]),
                "two reserved tid regions overlap — some tid names two different tensors",
            );
            j += 1;
        }
        // Regions descend, and none may run below the last one's floor into REAL tensor ids.
        assert!(
            RESERVED_REGIONS[i].floor() < RESERVED_REGIONS[i].base + 1,
            "a reserved region wrapped past zero",
        );
        i += 1;
    }
};

/// The region of a rung's stick constants: the `i`-th distinct constant its descriptors read is
/// staged at `stick_const_tid(i)` (see [`crate::StickConst`]).
pub const STICK_CONST_BASE: u32 = reserved_region("stick_const").base;

/// The tid the `i`-th distinct stick constant is placed under.
pub fn stick_const_tid(idx: usize) -> u32 {
    reserved_region("stick_const").at(idx as u32)
}
