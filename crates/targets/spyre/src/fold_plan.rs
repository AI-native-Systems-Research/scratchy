// SPDX-License-Identifier: Apache-2.0
//! Where a launched op reads and writes KV — the arithmetic the shim's launch loop does per op, and
//! per page of resident prefix, lifted out of C++.
//!
//! This is the whole of the fold loop's *decision making*: which request an op belongs to, how many
//! times its group is re-launched, and what that shifts each segment base by. None of it touches the
//! SDK — the SDK part (building a shifted `CompositeAddress` and submitting it) stays in C++, which
//! is why this can move without moving the launch itself.
//!
//! 🛑 THIS WAS A PORT, AND IT REPRODUCES THE C++ BEHAVIOUR EXACTLY, quirks included — see
//! [`slab_delta`], whose slab index comes from the LAUNCH position rather than the op's own. A port
//! that "fixes" something on the way is indistinguishable from a port that broke something, and an
//! earlier attempt at this file garbled on the pod. The tests below are the executable statement of
//! what it does; changing behaviour is a separate commit that changes them.

/// The KV cache's segment index. The cache write, the page base and the slab shift all land here.
pub const SEG_KV: usize = 2;
/// The prefix-validity mask's segment index. It rides in the activation segment precisely so a fold
/// can shift it without disturbing the running softmax state or any other activation.
pub const SEG_MASK: usize = 3;
/// Slots per slab in the incremental Kᵀ restickify — the fp16 stick, and the slab's slot count.
const SLAB_SLOTS: i64 = 64;

/// WHICH ROW OF THIS LAUNCH — a tensor row, not a request identity.
///
/// ⛔ THIS WAS `RowIdx`, "which request of a batched launch an op's KV belongs to". The rename is the
/// point: all it does now is select which of the launch's rows a PAGE is being resolved for, through the
/// host's block table. It contributes no address stride of its own — a page holds slots, so there is no
/// "distance between two requests" left to add — and nothing below the host treats it as an identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowIdx(u64);

impl RowIdx {
    /// ⭐⭐⭐ THE ONLY WAY TO NAME A LAUNCH ROW — and it says which of the three meanings it carries.
    ///
    /// ⛔ THE FIELD WAS `pub`, SO ANY INTEGER IN SCOPE COULD BECOME A ROW. In one step of a batched decode there
    /// were THREE different "which row" numbers: the KV ROW a request held for life (`KvRow` in
    /// `sdsc_abstract`), the LAUNCH SLOT it occupies this forward ([`LaunchSlot`], compacted over the LIVE set so
    /// it is NOT the row), and this — the index the fold's pass arithmetic walks. The first two were distinct
    /// types precisely because "they were one `u32` for a day and it cost a session"; this was a third spelling
    /// with a public field, filled across the FFI from whatever number the shim passed.
    ///
    /// ⭐ THERE ARE TWO NOW: `KvRow` IS DELETED. A request has no identity in the pool — the host allocates
    /// every page a batched launch writes, including the shared write page — so the only remaining question is
    /// "which row of THIS launch", which is what this type and `LaunchSlot` answer. The guard is still worth
    /// having: those two are the same number and different meanings (a position in the fold's walk vs a
    /// position in the batch), and this constructor names which one it takes.
    ///
    /// `from_launch_row` is named after what the fold actually means by it: the position within THIS launch,
    /// which is what the block table is keyed by (`set_block_table(slot, row, …)` installs a row's pages AT a
    /// slot). A caller that wants to pass a KV row has to notice that it cannot.
    pub fn from_launch_row(i: u64) -> RowIdx {
        RowIdx(i)
    }

    /// The index, for the block-table lookup that is the only consumer. Named, not `.0`, so the sites are
    /// countable.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// HOW MANY ROWS THIS LAUNCH BOUND — the fact that decides whether a per-row cursor exists at all.
///
/// A launch installs one page map and one cursor per row it is about to run (`set_block_table`, which
/// truncates at row 0 so the count is this forward's, never a high-water mark). So:
/// - `Single` — one map. Every prefill, and every unbatched decode. The launch's own position is the
///   only cursor in existence, so "row 0's cursor" is not a thing that can be asked for.
/// - `Batched` — one map per row. Row 0 has its own cursor exactly like every other row.
///
/// This exists so the two cases are decided ONCE, from the maps that were installed, instead of each
/// consumer inferring a batch from `request != 0`. Inferring it that way is what made row 0's write
/// cursor depend on the ORDER the worker presented the batch in: while batches were sorted
/// longest-first, row 0's cursor happened to equal the launch position, so falling back to the launch
/// was invisibly correct. Ordering by KV row instead (a request's row is its identity for its whole
/// life — that is what makes its KV base affine in `r`, and therefore what lets one launch address the
/// whole batch) makes the longest row any row, and the fallback silently wrong for row 0.
///
/// The match on it is exhaustive with no `_ =>`, so a third width cannot be added without every
/// consumer restating what it means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchWidth {
    /// One page map: the launch position is the only cursor.
    Single,
    /// `rows` page maps, `rows >= 2`: every row carries its own cursor.
    Batched { rows: usize },
}

impl LaunchWidth {
    /// Classify a launch from the maps it installed. An empty table is `Single` — a bundle that never
    /// paged installs nothing, and it still has the one position it was launched with.
    pub fn of(s: &SessionKv) -> LaunchWidth {
        let rows = s.request_positions.rows();
        if rows >= 2 {
            LaunchWidth::Batched { rows }
        } else {
            LaunchWidth::Single
        }
    }
}

/// A page index into a request's OWN block table, not a physical page in the pool.
///
/// ⭐⭐⭐⭐⭐ CONSTRUCTIBLE ONLY FROM A ROW THAT ACTUALLY HOLDS THE PAGE, which is what makes the silent
/// wrong-page read unexpressible rather than merely logged.
///
/// ⛔ WHAT THIS REPLACES. `page_base_bytes` used to answer an out-of-range lookup with `Bytes(0)` — and after a
/// separate, correct change removed the request term from its base, that zero became **ABSOLUTE page 0, i.e. row
/// 0's keys**. Its own doc had justified the fallback by saying it "keeps the REQUEST term", so the safety
/// argument outlived the property by many commits. Two defensible edits, one prose-only invariant, one silent
/// cross-row read.
///
/// ⛔ AND THE FIX IS NOT AN `Option` RETURN. Four call sites return `Bytes`, so an `Option` there just moves the
/// decision to four places that all want an address. Making the INDEX unconstructable moves it to ONE place: you
/// cannot name a page a row does not hold, so the lookup below cannot miss.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogicalPage(i64);

/// ⭐⭐⭐⭐ EVERY BOUND ROW'S OWN WRITE CURSOR, readable only BY ROW — never by a bare index.
///
/// ⛔ IT WAS A `Vec<i64>` INDEXED BY `req.get()` WITH A SILENT FALLBACK, and the fallback's comment cited "the
/// same discipline `page_base_bytes` follows for a page" — a discipline that no longer exists, because that
/// function's three fallbacks were exactly the defect (`Bytes(0)` = row 0's keys) and are now deleted. A comment
/// citing a precedent that has been removed is how the next one survives.
///
/// The fallback itself was not wrong: a ragged batch DOES ask for rows it does not own, and the launch's own
/// position is a meaningful answer (it is this forward's cursor, not another row's). What was wrong is that
/// nothing distinguished "this row has no cursor" from "this row's cursor is 0". `of_row` returns `Option`, so the
/// caller states which it means.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowCursors(Vec<i64>);

impl RowCursors {
    /// The cursors this forward bound, in row order. Named so the vector cannot be built from any `Vec<i64>` that
    /// happens to be in scope.
    pub fn bound(cursors: Vec<i64>) -> RowCursors {
        RowCursors(cursors)
    }

    /// `None` when this launch bound no cursor for that row — a ragged batch asking for a row it does not own.
    pub fn of_row(&self, row: RowIdx) -> Option<SlotPos> {
        self.0
            .get(row.get() as usize)
            .copied()
            .map(SlotPos::of_launch)
    }

    /// How many rows this forward bound — the fact that decides whether the launch is batched at all.
    pub fn rows(&self) -> usize {
        self.0.len()
    }

    /// Keep only the first `n` rows. The launch installs exactly this forward's rows, never a high-water mark.
    pub fn truncate(&mut self, n: usize) {
        self.0.truncate(n);
    }

    /// Set row `i`'s cursor, growing the table to reach it. The one mutator, so "which row" is stated once.
    pub fn set(&mut self, i: usize, pos: i64) {
        if self.0.len() <= i {
            self.0.resize(i + 1, 0);
        }
        self.0[i] = pos;
    }
}

/// ⭐⭐⭐⭐ WHERE A ROW SITS IN **THIS** LAUNCH — the key the block table is installed under.
///
/// ⛔ THE PARAMETER THAT RECEIVES THIS WAS NAMED `req`. `set_block_table(req, row, pos, pages)` takes a SLOT in
/// its first argument — the worker passes `slot.index()` — while the second is the KV ROW the request holds for
/// life. Two `i64`s, adjacent, with the first one misnamed, and slots are COMPACTED over the live set so they are
/// not equal whenever admission is staggered.
///
/// A distinct type from [`RowIdx`] because the whole defect class here is that these two are interchangeable at
/// the call site. Swapping them now fails to compile instead of installing one row's pages under another row's key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LaunchSlotIdx(i64);

impl LaunchSlotIdx {
    /// The slot a request occupies in the forward being built. Named for what it is, so it cannot be filled from
    /// a KV row by position.
    pub fn of_launch(i: i64) -> LaunchSlotIdx {
        LaunchSlotIdx(i)
    }

    pub fn get(self) -> i64 {
        self.0
    }
}

/// ⭐⭐⭐⭐⭐ A ROW AND A PAGE **THAT ROW ACTUALLY HOLDS**, validated together — so resolving it to an address
/// cannot fail, and therefore cannot fall back.
///
/// ⛔ THE PAIR IS THE UNIT BECAUSE THE PAIR IS WHAT WAS WRONG. `page_base_bytes(s, lp, req)` took two independent
/// values and had three ways to give up — no table for that row, a negative page, a page past the end — each
/// answering with `Bytes(0)`, which is absolute page 0, which is ROW 0's KEYS. Validating them separately cannot
/// help: the question is not "is this a page" and "is this a row" but "does THIS row hold THIS page".
///
/// With the pair minted once, the resolver is total. No `Option` in its signature, no fallback in its body, and
/// nothing for a later commit to quietly make unsafe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowPage {
    row: RowIdx,
    page: LogicalPage,
}

impl RowPage {
    /// `None` when this launch row does not hold that page — the caller's cue to skip the pass or refuse the
    /// launch, which is a decision that belongs at the call site and nowhere else.
    pub fn of(s: &SessionKv, row: RowIdx, i: i64) -> Option<RowPage> {
        let table = s.block_tables.get(row.get() as usize)?;
        Some(RowPage {
            row,
            page: LogicalPage::of_row(table, i)?,
        })
    }

    /// The row, for the callers that still report it.
    pub fn row(self) -> RowIdx {
        self.row
    }

    /// The page, for the callers that still report it.
    pub fn page(self) -> LogicalPage {
        self.page
    }
}

impl LogicalPage {
    /// The FIRST page of any row that holds at least one — the one index that needs no bound.
    pub const FIRST: LogicalPage = LogicalPage(0);

    /// `None` when this row does not hold page `i`. The ONLY door, so an out-of-range page has no value.
    ///
    /// Takes the row's OWN table rather than an index into a table, because "which row" and "which page of that
    /// row" are the pair that was being confused: with the table in hand there is nothing left to disagree about.
    pub fn of_row(table: &[i64], i: i64) -> Option<LogicalPage> {
        (i >= 0 && (i as usize) < table.len()).then_some(LogicalPage(i))
    }

    /// The index, for the one lookup that consumes it.
    pub fn get(self) -> i64 {
        self.0
    }
}

/// An absolute write position in a sequence — the count of tokens already resident.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SlotPos(i64);

impl SlotPos {
    /// ⛔ NOT THE SAME THING AS `sdsc_abstract::SeqPos`, AND THE DIFFERENCE IS THE BATCH.
    ///
    /// `SeqPos` is where a request is in its OWN conversation — what RoPE rotates by. This is where THIS LAUNCH
    /// writes, which for a batch is one shared slot past every live row's history. For a single request they are
    /// the same number, which is exactly why two crates named the same concept differently and nothing noticed:
    /// the batch is where they diverge, and a short row rotated at the launch's position instead of its own reads
    /// as that row drifting toward the longest row's topic.
    ///
    /// Named rather than a tuple constructor so the two cannot be swapped by position in an argument list.
    pub fn of_launch(pos: i64) -> SlotPos {
        SlotPos(pos)
    }

    pub fn get(self) -> i64 {
        self.0
    }
}

/// Positions per page. The write slot is a position WITHIN a page; the page comes from a block table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PageSlots(i64);

impl PageSlots {
    /// Positions per page — a CAPACITY, not a position and not a count of pages. Named because `SlotPos`,
    /// `PageSlots` and a page COUNT are three `i64`s that appear together in this file's arithmetic.
    pub fn per_page(n: i64) -> PageSlots {
        PageSlots(n)
    }

    pub fn get(self) -> i64 {
        self.0
    }
}

/// A byte quantity — an offset, a stride, or a base.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Bytes(pub u64);

impl std::ops::Add for Bytes {
    type Output = Bytes;
    fn add(self, o: Bytes) -> Bytes {
        Bytes(self.0.wrapping_add(o.0))
    }
}

/// The per-op manifest fields the launch arithmetic reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpKv {
    /// `kv_request`: whose KV this op touches. 0 for every op of an unbatched bundle.
    pub request: u64,
    /// `kv_page_fold`: this group folds ONE page of resident prefix and is re-launched per page.
    pub page_fold: bool,
    /// `slot_write`: this op appends the new token's KV at the write cursor.
    pub slot_write: bool,
    /// `kv_page_slots`: the write-slot modulus, and the signal that the bundle is paged.
    pub page_slots: u64,
    /// Bytes between consecutive write slots.
    pub slot_stride_bytes: u64,
    /// `slab_write`: the incremental Kᵀ restickify, which re-transposes the current slab.
    pub slab_write: bool,
    /// Bytes between consecutive slabs.
    pub slab_stride_bytes: u64,
    /// THIS OP WAS BAKED WITH A REQUEST AXIS — its kernel steps one request per unit of that axis, so
    /// ONE launch computes every row of the batch and the fold no longer needs a pass per request.
    ///
    /// Declared by the emitter, not inferred, for the same reason as [`SessionKv::int_rep_stride_bytes`]:
    /// what axes an op was baked with is not visible from the session, and guessing it wrong turns
    /// `pages x requests` passes into `pages` while the kernel still only reads one request's K — every
    /// row but one silently attending the wrong history. Every bundle baked so far leaves it false and
    /// is bit-for-bit unaffected.
    ///
    /// It is only HALF the precondition. The other half is a pool layout that admits a single stride,
    /// which is a runtime property of the maps the host installed and so is checked, not declared —
    /// see [`LaunchPages`].
    pub batched_requests: bool,
}

/// The session state the arithmetic reads.
#[derive(Clone, Debug, Default)]
pub struct SessionKv {
    pub paged: bool,
    /// Positions per page (the session's, which is NOT always the one an op divides by — see
    /// [`nonfold_page_delta`]).
    pub page_slots: PageSlots,
    /// Bytes per page = all layers of one page.
    pub page_stride_bytes: Bytes,
    // ⛔ NO REQUEST CONCEPT HERE ANY MORE. `request_stride_bytes` and `request_rows` used to live in this
    // struct: the distance between two requests' KV inside one page+layer, and the number of requests a
    // page holds. Both are GONE because a request is no longer a coordinate below the host — a pool cell
    // is `(plane, kvh, slot, feat)` with the slot ABSOLUTE, and a request is reached only by its PAGE,
    // through the host's block table. Request identity lives in the host mask and nowhere else.
    //
    // They were also both DEAD, which is how a request concept survives a rebuild: `request_stride_bytes`
    // had no reader left (the address builders say "NO REQUEST TERM. It used to be
    // `kv_rows[req] * request_stride_bytes`"), and `request_rows` gated a bound check that could never
    // fire because the emitter publishes `kv_request_rows = 0u32` unconditionally — "ALWAYS 0. There is
    // no request dimension in a page any more". An inert guard reads as protection and is not.
    // ⛔⛔⛔ AND `kv_rows` IS GONE TOO — the pool row was the LAST request concept in this struct.
    //
    // It was carried "because a request's row is its identity for as long as it lives, and it is what
    // the request term of the address multiplies". There is no request term any more: every address
    // builder here says so ("NO REQUEST TERM. It used to be `kv_rows[req] * request_stride_bytes`"),
    // and a grep for readers found NONE — the field was written by `install_row` on every step of every
    // batched forward and read by nothing. Dead state that names a forbidden concept reads as proof the
    // concept is still needed.
    //
    // What actually distinguishes two requests below the host is their PAGES, and those arrive in
    // `block_tables` keyed by the LAUNCH SLOT (`set_block_table` installs at
    // `RowIdx::from_launch_row(slot)`). The row never entered an address; it only rode along.
    /// One page map per request. A batched launch installs one per row; an unbatched one has a single
    /// entry, which is why an untagged op and request 0 land in the same place.
    pub block_tables: Vec<Vec<i64>>,
    /// Each request's own write cursor, one per row this forward bound. Its LENGTH is what says
    /// whether the launch is batched at all ([`LaunchWidth`]), and at two or more rows index 0 is row
    /// 0's cursor like any other row's — a batch is ordered by KV row, so row 0 is not the longest.
    pub request_positions: RowCursors,
    /// How many requests the prefix fold must sweep. ONE launched op has ONE page base, so B
    /// requests cannot share a pass: the fold runs `pages x requests` times, and each pass names
    /// both. 0 or 1 ⇒ the single-request sweep, which is what every prefill bundle wants.
    ///
    /// INFERRED from the installed page maps rather than declared, because the worker already
    /// installs exactly one per row of the rung it is about to run, every step.
    pub fold_requests: i64,
    /// Bytes between consecutive fold passes in the MASK segment. 0 ⇒ derive `page_slots * 2`, one
    /// page of ONE row, which is what a broadcast mask wants. A per-request mask is `nqh * mq` rows
    /// deep, so its pass stride is that many times larger and only the worker knows it.
    pub mask_rep_stride_bytes: u64,
    /// BLOCKS THE MASK ACTUALLY HAS — declared by the worker that staged it.
    ///
    /// ⛔ `reps` is derived from the LAUNCH POSITION and the mask is staged by the HOST, and nothing
    /// tied the two together: a pass past the last block reads bytes nobody staged, which are ZERO, and
    /// zero in an additive mask means VALID. So the failure mode of a mismatch is a row attending
    /// whatever the pool holds — fluent, wrong, no fault. 0 means "not declared", which keeps every
    /// bundle that never staged a per-pass mask behaving exactly as before.
    pub mask_blocks: u64,
    /// ⭐ THE HOST'S `FoldPages` — how many pages the fold must walk, decided ONCE on the Rust side from
    /// the batch's `BatchSlot` and sent across. 0 means "not declared", which leaves every existing
    /// bundle behaving exactly as before.
    ///
    /// ⛔ WHY THIS EXISTS. The executor computes the same quantity itself
    /// (`superdsc_exec::Executor::n_fold_pages`), while the worker blocks the prefix mask by ITS
    /// derivation. Two derivations of one number, agreeing only because `ensure_pages` grows every
    /// row to the shared write slot — a fact stated in neither. `FoldPages` made the WORKER side
    /// single; this makes a divergence between the two LOUD.
    ///
    /// A mismatch cannot be allowed to proceed: the mask is blocked `row * pages + page` and the fold
    /// inverts it `(rep / pages, rep % pages)`, so a differing `pages` makes every row but row 0 read
    /// another row's page — and an unstaged additive-mask byte reads as ZERO, i.e. VALID. Fluent, wrong,
    /// no fault. [`reps`] therefore refuses (returns -1) rather than folding on a disagreement.
    pub fold_pages: u64,
    /// Bytes between consecutive REQUESTS' row blocks in the INTERMEDIATE segment (seg0). 0 ⇒ the
    /// fold's ops span every row of the batch and need no shift, which is every bundle baked so far.
    ///
    /// Non-zero says the opposite: this bundle's fold ops are emitted for ONE request's `nqh` rows,
    /// so each pass must be pointed at its own request's block — `nqh * stick * 2` bytes apart. That
    /// is the whole point of the row-batched fold: `pages x requests` passes over `nqh` rows each,
    /// instead of over `nqh * mq` rows of which one request's are kept and the rest masked away.
    ///
    /// Declared, not inferred, because the row count an op was BAKED at is not visible here — only
    /// the emitter knows whether it wrote per-request ops, and getting this wrong silently folds one
    /// request's attention into another's rows.
    pub int_rep_stride_bytes: u64,
}

impl Default for PageSlots {
    fn default() -> Self {
        PageSlots::per_page(0)
    }
}

/// EVERYTHING A LAUNCH NEEDS TO ADDRESS ONE REQUEST'S KV, resolved once and derived from thereafter.
///
/// The reason this type exists: `kv_request` was a bare `u32` on the op, and SEVEN independent
/// consumers each had to remember to honour it — the page base, the write cursor, the slab index, the
/// mask block, the group boundary, the pass count, and the bind slot. Nothing tied them together, so
/// every site that forgot produced a SILENT wrong token: no crash, no shape error, fluent output. Six
/// separate bugs of that exact shape were fixed one at a time (fold width as a session high-water
/// mark, the slab shift taking the launch position, a duplicate C++ block-table store, the fused body
/// losing its fold, the split bodies allocated on pages alone, and a re-transpose that wrote row 0's
/// page for every request). Patching a seventh site is not the fix; removing the possibility is.
///
/// So a `RequestSlot` is built ONCE per (launch, request) from the maps that own the answer, and every
/// address below is a method on it. Two consumers cannot disagree about which page or which cursor a
/// request has, because they read one value instead of each re-deriving it.
///
/// THE CURSOR COMES FROM [`LaunchWidth`], NOT FROM WHETHER THE REQUEST INDEX IS ZERO. A batched launch
/// gives every row its own cursor, row 0 included; a single-row launch has only the position it was
/// launched with. Which of the two applies is read off the installed maps once, here, rather than
/// inferred from `req != 0` at each call site — see [`LaunchWidth`] for why inferring it made row 0's
/// cache write depend on the order the worker presented the batch in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RequestSlot {
    req: RowIdx,
    /// The write cursor THIS request appends at — its own row's, or the launch's when unbatched.
    pos: SlotPos,
}

impl RequestSlot {
    /// Resolve request `req`'s slot for a launch given at `launch`.
    pub fn resolve(s: &SessionKv, req: RowIdx, launch: SlotPos) -> RequestSlot {
        let pos = match LaunchWidth::of(s) {
            // Nothing to read: one map means one cursor, and it is the one we were given. Keeping this
            // arm free of the table is what makes every prefill and every unbatched decode
            // byte-identical to what they were before batching existed.
            LaunchWidth::Single => launch,
            // A row past the end is a ragged batch asking for a row it does not own; the mask already
            // marks it invalid, so fall back rather than fault — the same discipline `page_base_bytes`
            // follows for a page.
            LaunchWidth::Batched { rows } => {
                // `of_row` IS the bound check now: `None` means this launch bound no cursor for that row, and
                // the launch's own position is the deliberate answer — a meaningful cursor for this forward,
                // never another row's.
                let _ = rows;
                s.request_positions.of_row(req).unwrap_or(launch)
            }
        };
        RequestSlot { req, pos }
    }

    /// The request this slot addresses.
    pub fn request(&self) -> RowIdx {
        self.req
    }

    /// The absolute write cursor — positions already resident for this request.
    pub fn pos(&self) -> SlotPos {
        self.pos
    }

    /// Byte base of this request's logical page `i`, or `None` if this request does not hold it.
    ///
    /// The `Option` is HERE and not inside the resolver: this is where "which page of which row" is first
    /// asserted, so it is where the answer can be absent. Inward of the mint there is nothing to fail.
    pub fn page_base(&self, s: &SessionKv, i: i64) -> Option<Bytes> {
        RowPage::of(s, self.req, i).map(|rp| page_base_bytes(s, rp))
    }

    /// The logical page this request's NEXT write lands in.
    pub fn writing_page(&self, s: &SessionKv) -> LogicalPage {
        let per = s.page_slots.get().max(1);
        LogicalPage(self.pos.get() / per)
    }

    /// The slot WITHIN its page that the next write lands at. The absolute position would run past it.
    pub fn slot_in_page(&self, page_slots: u64) -> i64 {
        if page_slots > 0 {
            self.pos.get() % (page_slots as i64)
        } else {
            self.pos.get()
        }
    }

    /// TWO SLOTS ARE FUSABLE ONLY IF THEY ARE THE SAME SLOT.
    ///
    /// A group is ONE launch and a launch resolves ONE page table and ONE cursor, so ops for two
    /// different requests cannot share a group — a single per-group base shift cannot express both.
    /// Asking the slot makes that a property of the addressing instead of something each new
    /// `GroupKind` has to remember.
    pub fn fusable_with(&self, other: &RequestSlot) -> bool {
        self.req == other.req && self.pos == other.pos
    }
}

impl SessionKv {
    /// Install request `req`'s page map, its POOL ROW and its write cursor — the one place a
    /// forward's rows enter this state.
    ///
    /// ⛔ SLOT 0 STARTS A FORWARD, SO IT RETIRES THE LAST ONE'S MAPS. [`fold_requests`] is inferred
    /// from how many maps are installed, and this vector only ever GREW — so the count was a
    /// HIGH-WATER MARK over the session's life, not a statement about the forward about to run. One
    /// batched forward installing eight maps left every LATER forward, including a single request's,
    /// folding `pages * 8` times: sweeping seven other requests' pages with a mask staged for its own
    /// rows. The signature is a CORRECT FIRST TOKEN (prefill writes its own KV) and garbage from the
    /// second on.
    ///
    /// Truncating here is what keeps the inference honest, rather than adding a declared count that
    /// could disagree with it. Every forward binds its rows from slot 0 upward, so slot 0 IS the
    /// start of a new forward and anything above the rows this one binds is stale by definition.
    pub fn install_row(&mut self, req: RowIdx, pos: SlotPos, pages: &[i64]) {
        let r = req.get() as usize;
        if r == 0 {
            self.block_tables.truncate(1);
            self.request_positions.truncate(1);
        }
        if self.block_tables.len() <= r {
            self.block_tables.resize(r + 1, Vec::new());
        }
        self.block_tables[r] = pages.to_vec();
        // `set` grows the table to reach the row, so no separate length-then-resize dance.
        self.request_positions.set(r, pos.get());
        // The fold sweeps as many requests as there are page maps. Inferred rather than declared:
        // the worker installs exactly one per row of the rung it is about to run, every step, so the
        // count is already being stated — asking for it again is a second source that can disagree.
        self.fold_requests = self.block_tables.len() as i64;
    }
}

/// Byte base of logical page `lp` of request `req` — BOTH terms of the address: which page, and where
/// the request sits inside it.
///
/// Out-of-range is NOT an error here: it returns the request's OWN page 0 and lets the caller's mask
/// decide. That is deliberate and load-bearing — a batch is ragged, so a short request is asked for
/// pages it does not own every step, and its own prefix mask already marks those columns invalid.
/// Returning a base keeps the address inside the pool; refusing would fault on data that is masked
/// away anyway. Note that the fallback keeps the REQUEST term, so a page a request does not own reads
/// its own keys rather than request 0's — under the run-per-row pool "page 0" was row 0's first page,
/// which was safe only because the mask happened to cover it.
pub fn page_base_bytes(s: &SessionKv, rp: RowPage) -> Bytes {
    if !s.paged {
        return Bytes(0);
    }
    // ⭐ TOTAL, AND THAT IS THE POINT. There is no `else` here any more: `RowPage` was minted from this session's
    // own block table, so the row exists, the page is non-negative, and the page is within that row's map. The
    // three `return Bytes(base)` fallbacks this function used to have are gone — each one resolved to ABSOLUTE
    // PAGE 0, which is row 0's keys, and the doc that called them safe justified it with a request term that a
    // later commit deleted.
    //
    // ⛔ NO REQUEST TERM, still: a row is isolated by owning its own PAGES, so the whole of a row's base comes
    // from the page map. That is exactly why a missing entry could not be papered over with a zero, and why the
    // index had to become unconstructable instead.
    let phys = s.block_tables[rp.row().get() as usize][rp.page().get() as usize];
    Bytes((phys as u64).wrapping_mul(s.page_stride_bytes.0))
}

/// WHETHER ONE LAUNCH CAN REACH EVERY LIVE ROW'S KV, decided from the page maps the host installed.
///
/// The fold costs `pages x requests` passes today, and a pass is a FIXED cost (measured: 256 rows per
/// pass and 32 rows per pass price the same, and four times the swept width costs +6%), so the passes
/// ARE the batched-decode scaling curve — 2.05 ms per row at bs<=8. Collapsing them to `pages` is the
/// whole win, and it is legal exactly when a launch can step from one request's page to the next by a
/// single constant: `base(lp, r) == base(lp, 0) + r * stride`.
///
/// THAT IS A PROPERTY OF THE POOL, NOT OF THE BATCH, so it is CHECKED here rather than declared.
///
/// ⛔⛔⛔ THIS DOC USED TO SAY THE POOL *"separates them INSIDE each page by
/// `PagedKvPool::request_stride` — a bake constant, which is the half that matters"*. **THAT CONSTANT
/// DOES NOT EXIST.** `sdsc_abstract.rs:5197` records that the per-kv-head block distance *"replaced
/// `block_index(kvh) = kvh * ROWS` and `request_stride`"*, and the pool's address law states the
/// consequence outright at `:5169`: *"There is no request term: a request is a set of SLOTS (reached
/// through the host's page map), never a coordinate the device computes with."*
///
/// So the affinity this type checks is a property of the HOST's `page_base_bytes` — a launch-time fact
/// — and NOT of any stride a matmul could bake. `hd * PAGE_SLOTS`, the number three request-axis
/// attempts baked believing it was `request_stride`, is MEASURED
/// (`tests/fold_request_axis_strides.rs`) to be exactly the KV-HEAD stride: a `y` step of one advances
/// one kv head, so request `r` scored against kv head `r`'s keys — real, well-formed, wrong.
///
/// ⛔ WHICH MEANS [`Affine`](LaunchPages::Affine) IS NECESSARY BUT NOT SUFFICIENT for a pool operand.
/// It says the host CAN reach every row by one stride; it does not give the device an axis to step,
/// because the law has none. A request axis needs an operand with a UNIFORM PER-REQUEST PITCH, and the
/// pool is not one — so the collapse is not reachable by picking a better constant for the pool, which
/// is the move every attempt made. It needs a different operand whose pitch this compiler chooses, and
/// with it the removal of the absolute per-pass KV base [`fold_delta`] applies as a segment shift: a
/// read that computes its own address off an already-shifted base composes two bases and lands inside
/// neither — fluent garbage, no fault. That is one atomic change, and none of it is present today.
///
/// Holding still also depends on which POOL ROWS are live: a request's row is its
/// identity for life, so four requests holding rows 0,1,3,7 have no single stride even though the pool
/// is perfectly regular. Reading the answer off the installed state is what makes the irregular case
/// cost correctness nothing: it falls back to [`Table`](LaunchPages::Table), which is exactly today's
/// arithmetic.
///
/// 🛑 The candidate stride is taken from one pair of rows and then VERIFIED against every row and every
/// page. Trusting the first difference is the bug this type exists to prevent — rows 0,1,3 agree on a
/// stride of 1 for the first two rows and then diverge, so a launch built on the unverified stride
/// reads row 2's KV for row 3 and returns fluent, wrong tokens.
///
/// ⛔ AND THE PAGE MAPS ARE NO LONGER WHERE THE ROW LIVES. This used to read the stride out of
/// `block_tables[1][0] - block_tables[0][0]`, which under the run-per-row pool was the row stride in
/// pages. Every table is now identical, so that probe would see 0 and refuse every batch — the
/// collapse would never fire, silently, and the only symptom would be that it was not faster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchPages {
    /// Every live row's page `lp` sits `stride` bytes past the row before it, so a launch stepping that
    /// stride addresses the whole batch: ONE pass per page.
    ///
    /// ⭐ THE STRIDE IS A **PAGE** STRIDE, `pages_per_row * page_stride_bytes`. It used to be
    /// `stride_rows * request_stride_bytes` — a distance between requests inside a shared page, which is
    /// exactly the concept that was removed. Striping the pool by row keeps the collapse (one launch for
    /// the batch) while the device still only steps a stride over TENSOR ROWS.
    Affine { stride: Bytes },
    /// The maps admit no single stride (or there is no batch to stride over). The fold re-launches per
    /// request, one page table per pass.
    Table,
}

impl LaunchPages {
    /// Classify the installed maps. Total, and `Table` for anything it cannot prove.
    pub fn of(s: &SessionKv) -> LaunchPages {
        // ⭐ THE STRIDE IS NOW A **PAGE** STRIDE BETWEEN ROWS, read off the maps the host installed.
        //
        // This used to ask whether the KV ROWS were consecutive and multiply by `request_stride_bytes` —
        // the distance between two requests sharing a page's slot axis. That dimension is gone. What
        // matters instead is whether the host STRIPED the pool by row: row `r`'s logical page `lp` must be
        // physical page `first + r * pages_per_row + lp`. If it did, one launch steps `pages_per_row *
        // page_stride_bytes` and covers the batch; if it did not, the fold re-launches per row.
        if !s.paged {
            return LaunchPages::Table;
        }
        let rows = s.block_tables.len();
        // Fewer than two rows has no second row to stride to, and the single pass it already runs IS the
        // collapsed form.
        if rows < 2 {
            return LaunchPages::Table;
        }
        let pages = s.block_tables[0].len();
        if pages == 0 || s.page_stride_bytes.0 == 0 {
            return LaunchPages::Table;
        }
        // A RAGGED BATCH IS NORMAL, but a launch has ONE page count per pass: a row holding fewer pages
        // than row 0 would be swept past its own history.
        if s.block_tables.iter().any(|bt| bt.len() != pages) {
            return LaunchPages::Table;
        }
        // THE STRIPE, from rows 0 and 1's first pages, then VERIFIED for every row and every page rather
        // than inferred from the pair.
        let stripe = s.block_tables[1][0] - s.block_tables[0][0];
        if stripe <= 0 {
            // Non-positive means two rows share a page (they would overwrite each other's keys) or the
            // stripe runs backwards out of the pool. Both are refused rather than expressed.
            return LaunchPages::Table;
        }
        for (r, bt) in s.block_tables.iter().enumerate() {
            for (lp, &phys) in bt.iter().enumerate() {
                if phys < 0 || phys != s.block_tables[0][0] + (r as i64) * stripe + lp as i64 {
                    return LaunchPages::Table;
                }
            }
        }
        LaunchPages::Affine {
            stride: Bytes((stripe as u64).wrapping_mul(s.page_stride_bytes.0)),
        }
    }
}

/// The write cursor an op uses — [`RequestSlot::resolve`], reached from an op instead of from a
/// request index. One arithmetic, so the two cannot drift.
pub fn op_slot_pos(op: &OpKv, s: &SessionKv, launch: SlotPos) -> SlotPos {
    RequestSlot::resolve(s, RowIdx::from_launch_row(op.request), launch).pos()
}

/// Whether this op's group is the prefix fold — re-launched once per resident page.
pub fn is_paged_fold(op: &OpKv, s: &SessionKv) -> bool {
    op.page_fold && s.paged
}

/// How many requests the fold sweeps — at least one, however the geometry was left.
pub fn fold_requests(s: &SessionKv) -> i64 {
    s.fold_requests.max(1)
}

/// How many times this op's group is launched: once per resident page PER REQUEST for the fold,
/// else once.
///
/// A launch resolves exactly one page table, so a batch cannot be served by one pass over pages —
/// every request needs its own pass over its own pages, and the mask is what keeps a pass from
/// leaking into the rows it does not belong to.
///
/// UNLESS BOTH HALVES OF THE PRECONDITION HOLD, in which case one pass serves the whole batch and the
/// `x requests` term — the entire batched-decode scaling curve — disappears:
///   1. the OP was baked with a request axis ([`OpKv::batched_requests`], declared by the emitter,
///      because what a kernel steps over is not visible from the session), and
///   2. the live rows admit a single stride ([`LaunchPages::Affine`], checked against the installed
///      maps, because which rows are live is a runtime fact).
///
/// Needing both is the point: either alone is a silent wrong-token bug. A request axis over a pool with
/// no single stride reads the wrong rows' pages; a strided pool with a kernel that has no request axis
/// computes one request's attention and hands it to all of them.
pub fn reps(op: &OpKv, s: &SessionKv, n_fold_pages: i64) -> i64 {
    if !is_paged_fold(op, s) {
        return 1;
    }
    // ⛔ THE HOST DECIDED THIS NUMBER; the shim's own ceiling must agree or nothing may fold.
    // `FoldPages::covering(BatchSlot, PAGE_SLOTS)` on the Rust side and
    // `(seq_pos + kv_page_slots - 1) / kv_page_slots` in the shim are two derivations of one quantity, in
    // two languages, agreeing only via `ensure_pages` — so the disagreement is checked HERE, where the
    // number is about to be turned into passes, instead of being discovered as a wrong answer. Refusing
    // (-1) is what the caller already reports for a mask/pass mismatch.
    if s.fold_pages > 0 && n_fold_pages != s.fold_pages as i64 {
        return -1;
    }
    let n = if collapsed(op, s) {
        n_fold_pages
    } else {
        n_fold_pages * fold_requests(s)
    };
    // ⛔ NEVER MORE PASSES THAN THE MASK HAS BLOCKS. Past the last block the fold reads bytes the host
    // never staged; they are zero, and zero is VALID in an additive mask, so the extra passes fold in
    // whatever the pool holds. Refusing here (a negative count the caller reports) makes the mismatch
    // loud instead of a plausible answer. `mask_blocks == 0` = not declared = unchanged behaviour.
    if s.mask_blocks > 0 && n > s.mask_blocks as i64 {
        return -1;
    }
    n
}

/// WHETHER THIS OP'S PASSES COVER THE WHOLE BATCH — both halves of the precondition, in one place.
///
/// Read by [`reps`] to decide HOW MANY passes and by [`fold_pass`] to decide WHAT a pass is. Those two
/// answers have to come from the same predicate: `reps` collapsing while `fold_pass` still divides
/// `rep` by the page count makes pass 1 of a 3-page context claim to be request 0's page 1 AND get
/// only one third of the launches — a wrong history and a truncated sweep at once.
pub fn collapsed(op: &OpKv, s: &SessionKv) -> bool {
    op.batched_requests && matches!(LaunchPages::of(s), LaunchPages::Affine { .. })
}

/// ⭐⭐⭐⭐⭐ A PROOF THAT EVERY LIVE ROW HOLDS THE SAME NUMBER OF PAGES, carrying that number.
///
/// ⛔ WHY A WITNESS AND NOT AN ASSERT. [`fold_pass`] decomposes a pass index by DIVIDING by one
/// pages-per-row, which is exact only when every row holds that many. The worker guarantees it —
/// `ensure_pages(req, batch_slot + 1, …)` runs for every live request before the launch — but that is a fact
/// about a different crate, and a `debug_assert` here would be a runtime check of a property that can be made
/// UNEXPRESSIBLE instead: `fold_pass` takes this type, so a ragged pool cannot reach the decomposition at all.
/// (A first attempt at this invariant WAS a `debug_assert`, which is the wrong kind of guard by this
/// codebase's own P0 rule, and it was replaced by this.)
///
/// ⛔ AND THE FAILURE IT PREVENTS IS INVISIBLE TO A UNIFORM BATCH: with rows holding `[1, 3, 5, 8]` and a
/// divisor of 8, pass 1 claims to be row 0's page 1 and row 0 holds one page. Every row having the same count
/// makes the division exact, which is why the uniform control passed byte-identically for as long as it did.
///
/// The only constructor inspects the installed tables. Empty tables are ignored — a padded launch row has no
/// pages of its own and is masked off entirely, so it constrains nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UniformPages(i64);

impl UniformPages {
    /// ⭐ THE REAL CONTRACT, WHICH AN EQUALITY CHECK GOT WRONG: `n_fold_pages` is how many pages this step
    /// SWEEPS, and that is legitimately FEWER than a row HOLDS — a short context folds one page while the row
    /// still owns two of capacity. So the witness carries the SWEPT count, and minting it requires both halves:
    /// the rows agree about what they hold, AND the sweep fits inside that.
    ///
    /// ⛔ AN EARLIER VERSION DEMANDED `swept == held` AND THE ABI TEST REFUSED IT — correctly. That is the value
    /// of threading a type through the tests as well as production: the over-strict version could not survive a
    /// fixture that already encoded the real relationship.
    pub fn sweeping(n_fold_pages: i64, s: &SessionKv) -> Option<UniformPages> {
        let held = UniformPages::of(s)?.get();
        let swept = n_fold_pages.max(1);
        (swept <= held).then_some(UniformPages(swept))
    }

    /// `None` when the installed rows do NOT agree about how many pages they hold — the caller's cue to refuse
    /// the launch rather than compute an address from a divisor that does not describe the pool.
    pub fn of(s: &SessionKv) -> Option<UniformPages> {
        let mut per: Option<i64> = None;
        for t in &s.block_tables {
            if t.is_empty() {
                continue;
            }
            let n = t.len() as i64;
            match per {
                None => per = Some(n),
                Some(p) if p == n => {}
                Some(_) => return None,
            }
        }
        Some(UniformPages(per.unwrap_or(1).max(1)))
    }

    /// A SINGLE row's page count — the unbatched case, where "every row agrees" is trivial.
    pub fn solo() -> UniformPages {
        UniformPages(1)
    }

    /// The divisor. The only exit, and it exists because the arithmetic below needs an integer.
    pub fn get(self) -> i64 {
        self.0
    }
}

/// Which (request, page) fold pass `rep` is, given how many pages each request sweeps.
///
/// Request-major, so a request's pages are consecutive: pass `r * pages + p` is request `r`'s page
/// `p`. The worker lays the mask out in the same order, and that agreement is the whole contract —
/// a pass reads one request's page and must be masked off for every row that is not that request.
/// `None` when this pass names a page its row does not hold — which a caller turns into NO SHIFT, never into
/// page 0. A delta of zero means "stay where the caller put you"; an address of zero means "row 0's keys". The
/// old code could not tell those apart because both were `Bytes(0)`.
pub fn fold_pass(
    op: &OpKv,
    s: &SessionKv,
    rep: i64,
    pages_per_request: UniformPages,
) -> Option<RowPage> {
    let per = pages_per_request.get();
    // COLLAPSED: a pass IS a page and it serves every request, so there is no request to name. The KV
    // base is request 0's and the kernel's own batch axis walks from there — which is the entire point,
    // and why naming a request here would double-count the very stride the kernel is stepping.
    if collapsed(op, s) || fold_requests(s) <= 1 {
        // A collapsed pass IS a page and serves every row, so row 0 names it — and it is still MINTED, so a
        // context that sweeps past what row 0 holds yields `None` rather than an address.
        return RowPage::of(s, RowIdx::from_launch_row(0), rep);
    }
    RowPage::of(
        s,
        RowIdx::from_launch_row((rep / per).max(0) as u64),
        rep % per,
    )
}

/// Fold pass `rep`: the KV segment moves to that pass's page, and the mask to that pass's block.
///
/// 🛑 The KV page comes from the PASS's request, not from the op's `kv_request`. Attention ops carry
/// no request — there is one emitted set of them for the whole batch — so reading the op here is
/// what made every row attend request 0's history.
pub fn fold_delta(
    s: &SessionKv,
    op: &OpKv,
    rep: i64,
    // The WITNESS, not a number: this function's whole job is to turn a pass index into addresses, and it cannot
    // do that correctly for a pool whose rows disagree about their page count.
    pages_per_request: UniformPages,
) -> (Bytes, Bytes, Bytes) {
    // ⭐ THE PASS ITSELF IS THE PROOF. `fold_pass` returns a `RowPage`, so this pass demonstrably names a page its
    // own row holds — there is no address to compute for a pass that does not, and no zero to fall back to.
    // ⛔ NO PASS, NO SHIFT. `fold_pass` yields `None` when the pass names a page its row does not hold, and these
    // are DELTAS — zero means "stay where the caller put you", which is the only safe answer. It is emphatically
    // NOT the same as an absolute base of zero, which is row 0's keys, and conflating those two zeros is precisely
    // the defect this type family removes.
    let Some(rp) = fold_pass(op, s, rep, pages_per_request) else {
        return (Bytes(0), Bytes(0), Bytes(0));
    };
    let req = rp.row();
    let kv = page_base_bytes(s, rp);
    // One page of one row unless the worker declared otherwise — a per-request mask is `nqh * mq`
    // rows deep, and stepping it by one row's page would land inside the first row's data.
    let stride = if s.mask_rep_stride_bytes > 0 {
        s.mask_rep_stride_bytes
    } else {
        (s.page_slots.get() as u64).wrapping_mul(2)
    };
    let mask = Bytes((rep as u64).wrapping_mul(stride));
    // THE PASS'S OWN ROW BLOCK, in the INTERMEDIATE segment (seg0).
    //
    // A pass that computes only its own request's `nqh` rows has to be pointed AT them, and every
    // buffer it needs is in the one segment: `qs`, the score block, the whole online-softmax state
    // (`bmax`/`newm`/`corr`/`expb`/`bsum`/`otmp`/`ov`/`run_m`/`run_l`/`run_o`) and `out` are all
    // produced-and-consumed, so all are `SegRole::Intermediate`. One delta moves the lot, and
    // because it is applied per-op per-rep it moves nothing else in the layer.
    //
    // BY REQUEST, NOT BY `rep` — this is the one place it differs from the mask. A request's pages
    // all fold into the SAME rows (that is what makes the online softmax a running state), so page
    // 2 of request 1 must land on request 1's rows, not two blocks further along. Stepping this by
    // `rep` would make a multi-page request accumulate into whatever rows came after it.
    //
    // ZERO until the worker declares a stride, which it only does for a bundle whose fold passes
    // are actually per-request. Every existing bundle leaves it 0 and is bit-for-bit unaffected.
    let intermediate = Bytes((req.0).wrapping_mul(s.int_rep_stride_bytes));
    (kv, mask, intermediate)
}

/// The new token's KV lands at the write cursor's slot WITHIN its page; the page itself comes from
/// the block table. Using the absolute position here would write straight past the page.
///
/// Divides by the OP's `page_slots`, not the session's — they agree in every bundle, but the op is
/// what the C++ reads, so the port reads it too.
pub fn slot_write_delta(op: &OpKv, pos: SlotPos) -> Bytes {
    if !op.slot_write || pos.get() <= 0 {
        return Bytes(0);
    }
    let slot_in_page = if op.page_slots > 0 {
        pos.get() % op.page_slots as i64
    } else {
        pos.get()
    };
    Bytes((slot_in_page as u64).wrapping_mul(op.slot_stride_bytes))
}

/// The page a NON-fold op works in — the one holding its write cursor.
///
/// A tagged op resolves its OWN request's page; an untagged one keeps the base the caller computed,
/// so an unbatched launch is unchanged down to the zero-base early-out.
pub fn nonfold_page_delta(op: &OpKv, s: &SessionKv, pos: SlotPos, caller_base: Bytes) -> Bytes {
    if !s.paged || is_paged_fold(op, s) {
        return Bytes(0);
    }
    if op.request > 0 {
        let lp = if s.page_slots.get() != 0 {
            pos.get() / s.page_slots.get()
        } else {
            0
        };
        // ⭐ THE `None` ARM IS THE CALLER'S BASE, NOT ZERO — and that distinction is the whole bug this type
        // family exists to prevent. A tagged op asking for a page its row does not hold keeps the base the caller
        // computed (which is inside the pool and belongs to this launch); it does NOT get absolute page 0, which
        // is row 0's keys. The old code could not express the difference because both were `Bytes(0)`.
        RowPage::of(s, RowIdx::from_launch_row(op.request), lp)
            .map_or(caller_base, |rp| page_base_bytes(s, rp))
    } else {
        caller_base
    }
}

/// The incremental Kᵀ restickify re-transposes the CURRENT slab, so its segment base moves by one
/// slab's stride. Slab 0 is the baked base and does not shift.
///
/// 🛑 TAKES THE LAUNCH POSITION, NOT THE OP'S. Every op of a batched launch therefore restickifies
/// row 0's slab. Preserved verbatim: it is what the C++ does, and every other batched-decode defect
/// found so far was invisible until a second request existed, so changing it here — untested, in the
/// same commit as a port — is precisely how the last attempt at this file ended up garbling.
pub fn slab_delta(op: &OpKv, pos: SlotPos) -> Bytes {
    if !op.slab_write {
        return Bytes(0);
    }
    let slab = pos.get() / SLAB_SLOTS;
    if slab <= 0 {
        return Bytes(0);
    }
    Bytes((slab as u64).wrapping_mul(op.slab_stride_bytes))
}

/// Everything the launch loop adds to the segment bases for one op on one pass, in the C++'s order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SegDeltas {
    pub kv: Bytes,
    pub mask: Bytes,
    /// The shift into the INTERMEDIATE segment (seg0) — a row-batched fold pass pointing itself at
    /// its own request's `nqh` rows. Always 0 for a fold that spans the whole batch.
    pub intermediate: Bytes,
}

/// THE port's entry point: the segment shifts for op `op` on pass `rep`.
pub fn seg_deltas(
    op: &OpKv,
    s: &SessionKv,
    rep: i64,
    // ⭐ THE WITNESS ENTERS HERE, at the port's own entry point — which is the right boundary for it. The FFI
    // below receives a bare `i64` from the shim (C++ has no way to hold a Rust proof), so the conversion from
    // "a number the shim believes" to "a number the installed pool agrees with" happens exactly once, at the
    // edge, and everything inward is proved.
    pages_per_request: UniformPages,
    launch: SlotPos,
    caller_page_base: Bytes,
) -> SegDeltas {
    let pos = op_slot_pos(op, s, launch);
    let mut kv = Bytes(0);
    let mut mask = Bytes(0);
    // Only a fold pass moves within the intermediate segment, and only when the emitter baked its
    // fold ops per-request. Every other op reads seg0 exactly where it was placed.
    let mut intermediate = Bytes(0);
    if is_paged_fold(op, s) {
        let (k, m, i) = fold_delta(s, op, rep, pages_per_request);
        kv = kv + k;
        mask = mask + m;
        intermediate = intermediate + i;
    }
    kv = kv + slot_write_delta(op, pos);
    kv = kv + nonfold_page_delta(op, s, pos, caller_page_base);
    kv = kv + slab_delta(op, pos);
    SegDeltas {
        kv,
        mask,
        intermediate,
    }
}

#[cfg(test)]
mod tests {
    // ⛔ THIS MODULE, NOT `cursor_proofs`. That one is `#[cfg(kani)]` and NEVER compiles under `cargo test` — this
    // file's own history records ~100 commits of "the proofs hold" that were VACUOUS for exactly that reason, and
    // a test appended into it a moment ago reported "0 passed; 19 filtered out" rather than failing. A test that
    // cannot run is worse than no test: it reads as coverage.
    use super::*;

    /// A paged session with two requests holding distinct physical pages.
    fn sess() -> SessionKv {
        SessionKv {
            paged: true,
            // NOT DECLARED, which is the default and means "the shim's ceiling is the only one" — these
            // tests drive `reps` with an explicit `n_fold_pages`, so pinning a host value here would make
            // every one of them refuse (-1) on the cross-check rather than exercise the arithmetic.
            fold_pages: 0,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(1000),
            // request 0 -> physical pages 5,6 ; request 1 -> 9,10
            block_tables: vec![vec![5, 6], vec![9, 10]],
            request_positions: RowCursors::bound(vec![0, 300]),
            // One request unless a test says otherwise: the shape every prefill bundle runs, and
            // the one an unbatched decode must keep bit-for-bit.
            fold_requests: 1,
            mask_rep_stride_bytes: 0,
            mask_blocks: 0,
            int_rep_stride_bytes: 0,
            // DENSE ROWS FROM 0, and there is NO request term left to set: `request_stride_bytes` and
            // `request_rows` are gone from `SessionKv` entirely, because a request is not a coordinate
            // below the host. The old comment here said "a test that wants the request term sets
            // `request_stride_bytes` itself" — no test can, and none should.
        }
    }

    #[test]
    fn a_page_base_is_the_physical_page_times_the_page_stride() {
        let s = sess();
        let base = |row: u64, i: i64| {
            RowPage::of(&s, RowIdx::from_launch_row(row), i).map(|rp| page_base_bytes(&s, rp))
        };
        assert_eq!(base(0, 0), Some(Bytes(5000)));
        assert_eq!(base(0, 1), Some(Bytes(6000)));
        assert_eq!(base(1, 0), Some(Bytes(9000)));
    }

    /// ⭐⭐⭐ AN OUT-OF-RANGE PAGE NO LONGER HAS AN ADDRESS — IT HAS NO VALUE AT ALL.
    ///
    /// ⛔ THIS TEST USED TO ASSERT THE OPPOSITE, and its name said so: `…falls_back_to_the_pool_base`, with
    /// `Bytes(0)` expected for a page past the end, a negative page, and a row with no table. That fallback was
    /// documented safe because it "keeps the REQUEST term" — and after a later commit removed the request term
    /// from the base, `Bytes(0)` became ABSOLUTE PAGE 0, which is ROW 0's KEYS. The prose outlived the property.
    ///
    /// Now the pair cannot be minted, so `page_base_bytes` is total and has no fallback to make unsafe. What the
    /// caller does with `None` is a decision at the call site: `fold_delta` returns ZERO DELTAS (stay where the
    /// caller put you), which is a different thing from an absolute base of zero — and being unable to express
    /// that difference is what the old code's single `Bytes(0)` cost.
    #[test]
    fn an_out_of_range_page_or_row_cannot_be_minted_at_all() {
        let s = sess();
        assert!(
            RowPage::of(&s, RowIdx::from_launch_row(0), 7).is_none(),
            "page past the end of the row's map"
        );
        assert!(
            RowPage::of(&s, RowIdx::from_launch_row(0), -1).is_none(),
            "negative page"
        );
        assert!(
            RowPage::of(&s, RowIdx::from_launch_row(9), 0).is_none(),
            "row with no installed table"
        );
        // And the pairs that DO exist still resolve, so the constructor is not merely refusing everything.
        assert!(RowPage::of(&s, RowIdx::from_launch_row(0), 0).is_some());
        assert!(RowPage::of(&s, RowIdx::from_launch_row(1), 1).is_some());
    }

    /// EVERY ROW OF A BATCH READS ITS OWN CURSOR, row 0 included. Exhaustively proved over symbolic
    /// tables by `cursor_proofs::every_row_of_a_batch_reads_its_own_cursor`; this is the worked example.
    #[test]
    fn every_row_of_a_batch_reads_its_own_cursor() {
        let mut s = sess();
        s.request_positions = RowCursors::bound(vec![7777, 300]);
        let op0 = OpKv {
            request: 0,
            ..Default::default()
        };
        let op1 = OpKv {
            request: 1,
            ..Default::default()
        };
        assert_eq!(
            op_slot_pos(&op0, &s, SlotPos::of_launch(42)),
            SlotPos::of_launch(7777)
        );
        assert_eq!(
            op_slot_pos(&op1, &s, SlotPos::of_launch(42)),
            SlotPos::of_launch(300)
        );
        // A request past the end of the table falls back, rather than indexing out of range.
        let op9 = OpKv {
            request: 9,
            ..Default::default()
        };
        assert_eq!(
            op_slot_pos(&op9, &s, SlotPos::of_launch(42)),
            SlotPos::of_launch(42)
        );
    }

    /// What keeps every unbatched launch byte-identical: with ONE map installed there is no table to
    /// consult, so the launch position is the cursor whatever the table happens to hold.
    #[test]
    fn a_single_row_launch_uses_the_launch_position() {
        let mut s = sess();
        s.request_positions = RowCursors::bound(vec![7777]);
        s.block_tables.truncate(1);
        let op0 = OpKv {
            request: 0,
            ..Default::default()
        };
        assert_eq!(
            op_slot_pos(&op0, &s, SlotPos::of_launch(42)),
            SlotPos::of_launch(42)
        );
        assert_eq!(LaunchWidth::of(&s), LaunchWidth::Single);
        s.request_positions = RowCursors::bound(Vec::new());
        assert_eq!(
            op_slot_pos(&op0, &s, SlotPos::of_launch(42)),
            SlotPos::of_launch(42)
        );
        assert_eq!(LaunchWidth::of(&s), LaunchWidth::Single);
    }

    #[test]
    fn only_a_fold_group_repeats_and_only_when_paged() {
        let s = sess();
        let fold = OpKv {
            page_fold: true,
            ..Default::default()
        };
        let plain = OpKv {
            page_fold: false,
            ..Default::default()
        };
        assert_eq!(reps(&fold, &s, 4), 4);
        assert_eq!(reps(&plain, &s, 4), 1);
        let unpaged = SessionKv {
            paged: false,
            ..sess()
        };
        assert_eq!(reps(&fold, &unpaged, 4), 1, "an unpaged bundle never folds");
    }

    /// ONE REQUEST: the fold walks that request's pages, exactly as it always did.
    #[test]
    fn an_unbatched_fold_walks_the_one_requests_pages() {
        let s = sess();
        let attn_fold = OpKv {
            page_fold: true,
            request: 0,
            ..Default::default()
        };
        assert_eq!(reps(&attn_fold, &s, 2), 2, "one request, two pages");
        let (kv, mask, _int) = fold_delta(
            &s,
            &attn_fold,
            1,
            UniformPages::of(&s).expect("fixture rows agree"),
        );
        assert_eq!(kv, Bytes(6000), "request 0's page 1 is physical 6");
        assert_eq!(mask, Bytes(256 * 2));
    }

    /// THE FIX. A batched fold sweeps (request, page) request-major, so each pass names BOTH — and
    /// it takes the request from the PASS, not from the op, because attention ops carry none. Before
    /// this, every pass read request 0's pages and every row attended request 0's history.
    #[test]
    fn a_batched_fold_sweeps_every_request_not_just_the_first() {
        let mut s = sess();
        s.fold_requests = 2;
        let attn_fold = OpKv {
            page_fold: true,
            request: 0,
            ..Default::default()
        };
        assert_eq!(reps(&attn_fold, &s, 2), 4, "2 requests x 2 pages");
        // Request-major: passes 0,1 are request 0's pages; 2,3 are request 1's.
        let pages = |rep| {
            fold_delta(
                &s,
                &attn_fold,
                rep,
                UniformPages::of(&s).expect("fixture rows agree"),
            )
            .0
        };
        assert_eq!(pages(0), Bytes(5000), "req 0 page 0 = physical 5");
        assert_eq!(pages(1), Bytes(6000), "req 0 page 1 = physical 6");
        assert_eq!(pages(2), Bytes(9000), "req 1 page 0 = physical 9");
        assert_eq!(pages(3), Bytes(10000), "req 1 page 1 = physical 10");
        for rep in 0..4 {
            let rp = fold_pass(
                &OpKv::default(),
                &s,
                rep,
                UniformPages::of(&s).expect("fixture rows agree"),
            )
            .expect("every rep of this fixture names a page its row holds");
            assert_eq!((rp.row().get(), rp.page().get()), (rep as u64 / 2, rep % 2));
        }
    }

    /// The mask must step by a whole per-request block, not by one row's page — the second half of
    /// the gap. A worker that declares no stride keeps the old one-row step.
    #[test]
    fn the_mask_steps_by_the_stride_the_worker_declares() {
        let mut s = sess();
        s.fold_requests = 2;
        let f = OpKv {
            page_fold: true,
            ..Default::default()
        };
        assert_eq!(
            fold_delta(&s, &f, 3, UniformPages::of(&s).expect("fixture rows agree")).1,
            Bytes(3 * 256 * 2),
            "derived: one page of one row"
        );
        // nqh*mq rows deep: 32 heads x 2 rows x 256 slots x 2 bytes per pass — minted by the
        // mask's own shape law, so the fixture cannot pin a stride the staging never writes.
        s.mask_rep_stride_bytes = {
            use scratchy_subtile::sdsc_abstract::{
                POOL_STICK, PagedKvPool, PrefixMaskShape, RungWidth,
            };
            const PER_PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
            PrefixMaskShape::<POOL_STICK, PER_PAGE>::new(
                32,
                RungWidth::of_baked_rows(2).expect("two rows"),
            )
            .expect("nonzero heads")
            .rep_stride_bytes()
        };
        assert_eq!(
            fold_delta(&s, &f, 3, UniformPages::of(&s).expect("fixture rows agree")).1,
            Bytes(3 * 32 * 2 * 256 * 2)
        );
    }

    /// ⭐⭐ THE FAILING GEOMETRY, ON THE HOST: 4 ROWS x 8 PAGES, EVERY PASS TO rep 31.
    ///
    /// The card says a row at 8 pages collapses in a 4-wide batch and not in a 2-wide one, and not at 6
    /// pages in a 4-wide one. In rep terms every PASSING configuration stays below rep 24
    /// (`w2x8 = 0..15`, `w4x6 = 0..23`) and the only FAILING one crosses it (`w4x8 = 0..31`) — and the row
    /// that collapses is the one whose passes sit at reps >= 24, which with `fold_pass(rep) = (rep/pages,
    /// rep%pages)` is row 3.
    ///
    /// So: does the HOST address rep 24..31 correctly? This costs milliseconds and needs no card, where each
    /// on-card trial costs ~5 minutes — and if it fails at 24 the bug is pure arithmetic here rather than
    /// anywhere on the device.
    #[test]
    fn every_pass_of_a_four_row_eight_page_fold_lands_on_its_own_page() {
        let mut s = sess();
        // The failing shape: 4 requests, 8 pages each, page tables deliberately NON-CONTIGUOUS because the
        // free list interleaves them (`f5e26a3e`) — row 3's pages are the ones reps 24..31 must reach.
        s.block_tables = (0..4i64)
            .map(|r| (0..8i64).map(|p| 11 + r * 13 + p).collect::<Vec<i64>>())
            .collect();
        s.request_positions = RowCursors::bound(vec![1986, 1986, 1986, 1986]);
        s.fold_requests = 4;
        let f = OpKv {
            page_fold: true,
            ..Default::default()
        };
        let pages: i64 = 8;

        for rep in 0..(pages * 4) {
            let rp = fold_pass(
                &f,
                &s,
                rep,
                UniformPages::of(&s).expect("fixture rows agree"),
            )
            .expect("every rep of this fixture names a page its row holds");
            let (row, page) = (rp.row(), rp.page());
            // THE DECOMPOSITION, first: row-major, so row `r` owns reps `[r*8, r*8+8)`.
            assert_eq!(
                (row.get(), page.get()),
                ((rep / pages) as u64, rep % pages),
                "rep {rep} must be row {} page {}",
                rep / pages,
                rep % pages
            );
            // THEN THE ADDRESS: the KV shift must be the PHYSICAL page this row's block table names,
            // times the page stride. Nothing else is a legal answer — a row reading another row's page is
            // the silent-wrong-answer failure this whole hunt is about.
            let want_phys = s.block_tables[row.0 as usize][page.0 as usize];
            let (kv, _mask, _int) = fold_delta(
                &s,
                &f,
                rep,
                UniformPages::of(&s).expect("fixture rows agree"),
            );
            assert_eq!(
                kv,
                Bytes(want_phys as u64 * s.page_stride_bytes.0),
                "rep {rep} (row {}, page {}) must land on physical page {want_phys}",
                row.0,
                page.0
            );
        }
    }

    /// THE INTERMEDIATE SHIFT STEPS BY REQUEST, NOT BY PASS — the one way it differs from the mask.
    ///
    /// A request's pages all fold into the SAME rows; that running accumulation IS the online
    /// softmax. So request 1's page 0 and its page 1 must both land on request 1's row block, while
    /// the mask moves on every pass because each pass masks a different page. Stepping this by `rep`
    /// instead would make a request's second page accumulate into whatever rows follow it — one
    /// request's attention written into another's state, which is fluent and unnoticeable.
    #[test]
    fn the_intermediate_shift_names_the_request_not_the_pass() {
        let mut s = sess();
        s.fold_requests = 2;
        // 32 heads of one request, one stick wide, f16 — minted by the regime law, the same
        // derivation the worker's launch hands `set_int_stride`.
        s.int_rep_stride_bytes =
            scratchy_subtile::sdsc_abstract::FoldRowRegime::PerRequest.int_rep_stride_bytes(32);
        let f = OpKv {
            page_fold: true,
            ..Default::default()
        };
        // 2 pages per request ⇒ passes 0,1 are request 0 and passes 2,3 are request 1.
        for (rep, want_req) in [(0i64, 0u64), (1, 0), (2, 1), (3, 1)] {
            assert_eq!(
                fold_delta(
                    &s,
                    &f,
                    rep,
                    UniformPages::of(&s).expect("fixture rows agree")
                )
                .2,
                Bytes(want_req * 32 * 64 * 2),
                "pass {rep} belongs to request {want_req}, and both its pages share that block"
            );
        }
        // Unset ⇒ a whole-batch fold, which must not be shifted at all.
        s.int_rep_stride_bytes = 0;
        for rep in 0..4 {
            assert_eq!(
                fold_delta(
                    &s,
                    &f,
                    rep,
                    UniformPages::of(&s).expect("fixture rows agree")
                )
                .2,
                Bytes(0),
                "pass {rep} of a whole-batch fold"
            );
        }
    }

    /// Only a FOLD pass moves in the intermediate segment. A cache write or a plain op reads seg0
    /// where the emitter placed it, however the fold is configured.
    #[test]
    fn a_non_fold_op_never_shifts_the_intermediate_segment() {
        let mut s = sess();
        s.fold_requests = 2;
        s.int_rep_stride_bytes =
            scratchy_subtile::sdsc_abstract::FoldRowRegime::PerRequest.int_rep_stride_bytes(32);
        for op in [
            OpKv {
                slot_write: true,
                page_slots: 256,
                slot_stride_bytes: 128,
                request: 1,
                ..Default::default()
            },
            OpKv {
                slab_write: true,
                slab_stride_bytes: 64,
                ..Default::default()
            },
            OpKv::default(),
        ] {
            let d = seg_deltas(
                &op,
                &s,
                1,
                UniformPages::of(&s).expect("fixture rows agree"),
                SlotPos::of_launch(300),
                Bytes(0),
            );
            assert_eq!(d.intermediate, Bytes(0), "non-fold op {op:?}");
        }
    }

    /// A BATCHED FORWARD MUST NOT WIDEN THE NEXT ONE'S FOLD. The map count IS the fold width, so a
    /// vector that only grew made it a high-water mark: after a batch of eight, a single request's
    /// forward folded eight times, sweeping seven other requests' pages. Correct first token,
    /// garbage from the second on.
    #[test]
    fn a_narrower_forward_does_not_inherit_the_last_one_s_fold_width() {
        let mut s = SessionKv {
            paged: true,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(4096),
            ..Default::default()
        };
        let f = OpKv {
            page_fold: true,
            ..Default::default()
        };
        // A batched forward: eight requests, slot 0 upward, exactly as the worker binds them.
        for r in 0..8i64 {
            s.install_row(
                RowIdx::from_launch_row(r as u64),
                SlotPos::of_launch(10 * r),
                &[7],
            );
        }
        assert_eq!(
            reps(&f, &s, 1),
            8,
            "eight maps installed ⇒ the fold sweeps eight requests"
        );
        // Now ONE request. Binding slot 0 retires the other seven.
        s.install_row(RowIdx::from_launch_row(0), SlotPos::of_launch(3), &[7]);
        assert_eq!(
            reps(&f, &s, 1),
            1,
            "one map installed ⇒ ONE pass; inheriting 8 folds in seven other requests' pages"
        );
        // And it grows again for the next batched forward, so the truncation is not a ratchet.
        for r in 0..3i64 {
            s.install_row(
                RowIdx::from_launch_row(r as u64),
                SlotPos::of_launch(r),
                &[7],
            );
        }
        assert_eq!(reps(&f, &s, 1), 3);
    }

    #[test]
    fn a_write_lands_at_its_slot_within_the_page_never_the_absolute_position() {
        let op = OpKv {
            slot_write: true,
            page_slots: 256,
            slot_stride_bytes: 128,
            ..Default::default()
        };
        // Position 300 is slot 44 of page 1 — 300*128 would be a page and a half past the page.
        assert_eq!(
            slot_write_delta(&op, SlotPos::of_launch(300)),
            Bytes(44 * 128)
        );
        assert_eq!(
            slot_write_delta(&op, SlotPos::of_launch(0)),
            Bytes(0),
            "slot 0 is the baked base"
        );
        let unpaged = OpKv {
            page_slots: 0,
            ..op
        };
        assert_eq!(
            slot_write_delta(&unpaged, SlotPos::of_launch(300)),
            Bytes(300 * 128)
        );
        let not_a_write = OpKv {
            slot_write: false,
            ..op
        };
        assert_eq!(
            slot_write_delta(&not_a_write, SlotPos::of_launch(300)),
            Bytes(0)
        );
    }

    #[test]
    fn a_tagged_non_fold_op_resolves_its_own_page_an_untagged_one_keeps_the_callers() {
        let s = sess();
        let tagged = OpKv {
            request: 1,
            ..Default::default()
        };
        // position 300 is page 1 of request 1 => physical 10.
        assert_eq!(
            nonfold_page_delta(&tagged, &s, SlotPos::of_launch(300), Bytes(4242)),
            Bytes(10000)
        );
        let untagged = OpKv {
            request: 0,
            ..Default::default()
        };
        assert_eq!(
            nonfold_page_delta(&untagged, &s, SlotPos::of_launch(300), Bytes(4242)),
            Bytes(4242)
        );
        let fold = OpKv {
            request: 1,
            page_fold: true,
            ..Default::default()
        };
        assert_eq!(
            nonfold_page_delta(&fold, &s, SlotPos::of_launch(300), Bytes(4242)),
            Bytes(0),
            "a fold op takes its page from the fold pass, not from here"
        );
    }

    /// THE SLAB FOLLOWS THE REQUEST WHOSE K IT IS TRANSPOSING, not the launch.
    ///
    /// The port deliberately preserved the C++'s asymmetry here: the slab index came from the LAUNCH
    /// position while the write SLOT beside it came from the op's own request. That is invisible at one
    /// request, where they are the same number, and wrong for every other row of a ragged batch —
    /// request r's incremental Kt restickify landed in whatever slab row 0 happened to be in, so its
    /// new K was transposed into another request's slab and every later step read it back as garbage.
    ///
    /// This is the "separate, later commit" the module header says a behaviour change belongs in: the
    /// port is finished, and `seg_deltas` already resolves the op's own position for the write slot, so
    /// the slab now reads the same number rather than a second, disagreeing one.
    #[test]
    fn the_slab_shift_follows_the_requests_own_position() {
        let s = sess();
        let op = OpKv {
            slab_write: true,
            slab_stride_bytes: 8192,
            request: 1,
            ..Default::default()
        };
        // Request 1 sits at 300 (slab 4); the launch is at 100 (slab 1). The shift must be request 1's.
        assert_eq!(
            op_slot_pos(&op, &s, SlotPos::of_launch(100)),
            SlotPos::of_launch(300)
        );
        assert_eq!(
            slab_delta(&op, op_slot_pos(&op, &s, SlotPos::of_launch(100))),
            Bytes(4 * 8192)
        );
        assert_eq!(
            slab_delta(&op, SlotPos::of_launch(63)),
            Bytes(0),
            "slab 0 is the baked base"
        );
        // And the composite the shim actually calls agrees. Stated as its parts rather than a literal,
        // because a non-fold op also carries its request's page base and hard-coding the sum hides
        // which term moved: the point is that the slab term is request 1's slab 4, not the launch's 1.
        let d = seg_deltas(
            &op,
            &s,
            0,
            UniformPages::of(&s).expect("fixture rows agree"),
            SlotPos::of_launch(100),
            Bytes(0),
        );
        let own = op_slot_pos(&op, &s, SlotPos::of_launch(100));
        assert_eq!(
            d.kv,
            slab_delta(&op, own) + nonfold_page_delta(&op, &s, own, Bytes(0)),
            "the launch's slab would contribute 8192 here instead of {}",
            4 * 8192
        );
    }

    /// The composite, in the C++'s order, for the two shapes that actually launch today.
    #[test]
    fn the_composed_deltas_match_the_launch_loops_order() {
        let s = sess();
        // An unbatched cache write: no fold, request 0, caller-supplied page base kept.
        let cachewr = OpKv {
            slot_write: true,
            page_slots: 256,
            slot_stride_bytes: 128,
            ..Default::default()
        };
        // ONE MAP INSTALLED — that is what makes it unbatched, and what makes the launch position the
        // cursor. `sess()` is a two-row fixture, so say so explicitly rather than relying on the
        // request index being 0: a batched launch's row 0 has its own cursor.
        let solo = SessionKv {
            block_tables: vec![s.block_tables[0].clone()],
            request_positions: RowCursors::bound(vec![0]),
            fold_requests: 1,
            ..s.clone()
        };
        let d = seg_deltas(
            &cachewr,
            &solo,
            0,
            UniformPages::of(&s).expect("fixture rows agree"),
            SlotPos::of_launch(300),
            Bytes(5000),
        );
        assert_eq!(d.kv, Bytes(44 * 128 + 5000));
        assert_eq!(d.mask, Bytes(0));

        // A batched cache write for request 1: its own cursor, its own page, no caller base.
        let batched = OpKv {
            request: 1,
            ..cachewr
        };
        let d = seg_deltas(
            &batched,
            &s,
            0,
            UniformPages::of(&s).expect("fixture rows agree"),
            SlotPos::of_launch(0),
            Bytes(5000),
        );
        assert_eq!(
            d.kv,
            Bytes(44 * 128 + 10000),
            "slot 44 of request 1's page 1 (physical 10)"
        );
        assert_eq!(d.mask, Bytes(0));
    }

    /// The composed answer for a BATCHED launch: each op resolves its own request's cursor and page.
    /// This was pinned through the C ABI while one existed; the executor now calls exactly these
    /// functions, so it is pinned here.
    #[test]
    fn a_batched_launch_resolves_each_request_s_own_cursor_and_page() {
        let mut s = SessionKv {
            paged: true,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(1000),
            ..Default::default()
        };
        s.install_row(RowIdx::from_launch_row(0), SlotPos::of_launch(0), &[5, 6]);
        s.install_row(
            RowIdx::from_launch_row(1),
            SlotPos::of_launch(300),
            &[9, 10],
        );
        let per = UniformPages::sweeping(1, &s).expect("both rows hold two pages");

        // Batched cache write for request 1: its own cursor (300) and its own page (physical 10).
        let w = OpKv {
            request: 1,
            slot_write: true,
            page_slots: 256,
            slot_stride_bytes: 128,
            ..Default::default()
        };
        let d = seg_deltas(&w, &s, 0, per, SlotPos::of_launch(0), Bytes(5000));
        assert_eq!(d.kv, Bytes(44 * 128 + 10000));
        assert_eq!(d.mask, Bytes(0));

        // A fold pass takes its page from the pass index and shifts the mask by one page.
        let f = OpKv {
            page_fold: true,
            ..Default::default()
        };
        let per2 = UniformPages::sweeping(2, &s).expect("a two-page sweep fits");
        let d = seg_deltas(&f, &s, 1, per2, SlotPos::of_launch(0), Bytes(0));
        assert_eq!(d.kv, Bytes(6000));
        assert_eq!(d.mask, Bytes(256 * 2));

        // Two page maps were installed, so the fold sweeps two requests over four pages each.
        assert_eq!(reps(&f, &s, 4), 8);
        assert_eq!(reps(&OpKv::default(), &s, 4), 1);
    }

    /// Installing rows OUT OF ORDER must not lose one, and a row asking for a page it does not hold
    /// falls back to the pool base rather than another row's keys — its own mask covers those
    /// columns, and a ragged batch asks for them every step.
    #[test]
    fn rows_install_out_of_order_and_an_unheld_page_falls_back() {
        let mut s = SessionKv {
            paged: true,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(1000),
            ..Default::default()
        };
        s.install_row(RowIdx::from_launch_row(3), SlotPos::of_launch(900), &[77]);
        assert_eq!(
            s.block_tables.len(),
            4,
            "rows 1 and 2 exist as empty maps, row 3 is installed"
        );
        let per =
            UniformPages::sweeping(1, &s).expect("only row 3 holds pages, so it sets the count");
        // Asked as a NON-fold op, which is the one that reads the op's own tag — a fold pass is
        // chosen by `rep`, so it would answer for request 0 here whatever the op says.
        let w = OpKv {
            request: 3,
            slot_write: true,
            page_slots: 256,
            slot_stride_bytes: 128,
            ..Default::default()
        };
        let d = seg_deltas(&w, &s, 0, per, SlotPos::of_launch(0), Bytes(0));
        // Position 900 is slot 132 of page 3; request 3 holds ONE page, so page 3 is outside its map
        // and falls back to the pool base, leaving just the slot term.
        assert_eq!(d.kv, Bytes(132 * 128));
    }
    // ⛔ A TEST WAS HERE AND THE TYPE REPLACED IT. It built a ragged pool (`[1,3,5,8]`), called `fold_pass` with
    // the max divisor, and asserted the decomposition named pages rows do not own — `#[should_panic]`, i.e. a
    // RUNTIME statement of an invariant. `UniformPages` now makes that call unexpressible: the witness cannot be
    // minted from a ragged pool, so there is no way to reach the arithmetic with one. A test that cannot be
    // written is a stronger guarantee than a test that passes, which is the whole point of the exercise.
}

/// WHERE A REQUEST'S WRITE CURSOR COMES FROM, proved over every table a launch can install.
///
/// This is the one arithmetic in this file whose answer depended on the ORDER the worker presented a
/// batch in, and these proofs are what makes that dependence impossible rather than documented.
#[cfg(kani)]
mod cursor_proofs {
    use super::*;

    /// A session whose position table holds `rows` symbolic cursors.
    fn table(rows: usize, p: [i64; 4]) -> SessionKv {
        let mut request_positions = p.to_vec();
        request_positions.truncate(rows);
        SessionKv {
            paged: true,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(1024),
            block_tables: vec![vec![0]; rows],
            request_positions: RowCursors::bound(request_positions),
            fold_requests: rows as i64,
            // ⛔ AGREES WITH THE BLOCK-TABLE WIDTH (one page per row here). `reps` refuses when the mask's
            // `pages` and the fold's disagree: the mask is blocked `row * pages + page` and the fold inverts
            // it, so a mismatch makes every row but row 0 read another row's page — and an unstaged
            // additive-mask byte reads as ZERO, i.e. VALID. Fluent, wrong, no fault.
            fold_pages: 1,
            mask_rep_stride_bytes: 0,
            mask_blocks: 0,
            int_rep_stride_bytes: 0,
        }
    }

    /// EVERY ROW OF A BATCH READS ITS OWN CURSOR — ROW 0 INCLUDED, and none of them reads the launch
    /// position.
    ///
    /// Row 0 used to fall back to the launch, which was only ever correct because the worker sorted a
    /// batch longest-first, making row 0's cursor the maximum the launch was given. A batch is now
    /// ordered by KV row (a request's row is its identity for its whole life, since one launch can
    /// address B rows only if row r's KV base is affine in r), so the longest row is any row, and
    /// row 0's cache write would land at another request's slot. This proof is what forbids that.
    #[kani::proof]
    #[kani::unwind(6)]
    fn every_row_of_a_batch_reads_its_own_cursor() {
        let rows: usize = kani::any();
        kani::assume(rows >= 2 && rows <= 4);
        let p: [i64; 4] = kani::any();
        for c in p.iter() {
            kani::assume(*c >= 0 && *c <= 1_000_000);
        }
        let s = table(rows, p);
        let launch: i64 = kani::any();
        kani::assume(launch >= 0 && launch <= 1_000_000);
        let req: usize = kani::any();
        kani::assume(req < rows);

        let slot = RequestSlot::resolve(&s, RowIdx(req as u64), SlotPos::of_launch(launch));
        assert!(slot.pos() == SlotPos::of_launch(p[req]));
        // Said the other way, because it is the property that matters: the answer does not depend on
        // the launch position at all, so it cannot depend on how the batch was ordered.
        let other: i64 = kani::any();
        kani::assume(other >= 0 && other <= 1_000_000);
        assert!(
            RequestSlot::resolve(&s, RowIdx(req as u64), SlotPos::of_launch(other)).pos()
                == slot.pos()
        );
        // `op_slot_pos` is the same answer — one arithmetic, not two that can drift.
        let op = OpKv {
            request: req as u64,
            ..Default::default()
        };
        assert!(op_slot_pos(&op, &s, SlotPos::of_launch(launch)) == slot.pos());
    }

    /// A SINGLE-ROW LAUNCH IGNORES THE TABLE ENTIRELY — every prefill, and every unbatched decode.
    ///
    /// A launch that installed one map has one cursor, the one it was launched with, so there is no
    /// table entry that could disagree with it. Keeping this arm untouched is what makes the fix above
    /// byte-identical for every bundle that is not a batched decode.
    #[kani::proof]
    #[kani::unwind(6)]
    fn a_single_row_launch_uses_the_launch_position() {
        let p: [i64; 4] = kani::any();
        let s = table(1, p);
        let launch: i64 = kani::any();
        kani::assume(launch >= 0 && launch <= 1_000_000);
        assert!(
            RequestSlot::resolve(&s, RowIdx::from_launch_row(0), SlotPos::of_launch(launch)).pos()
                == SlotPos::of_launch(launch)
        );
        let op = OpKv {
            request: 0,
            ..Default::default()
        };
        assert!(op_slot_pos(&op, &s, SlotPos::of_launch(launch)) == SlotPos::of_launch(launch));
        // An empty table is the same launch: a bundle that never paged installs nothing.
        let empty = table(0, p);
        assert!(
            RequestSlot::resolve(
                &empty,
                RowIdx::from_launch_row(0),
                SlotPos::of_launch(launch)
            )
            .pos()
                == SlotPos::of_launch(launch)
        );
    }

    /// A session whose `rows` page maps each hold `pages` symbolic PHYSICAL pages.
    fn pool(rows: usize, pages: usize, phys: [i64; 6]) -> SessionKv {
        let mut block_tables = Vec::new();
        for r in 0..rows {
            let mut bt = Vec::new();
            for lp in 0..pages {
                bt.push(phys[(r * pages + lp) % 6]);
            }
            block_tables.push(bt);
        }
        SessionKv {
            paged: true,
            page_slots: PageSlots::per_page(256),
            page_stride_bytes: Bytes(1024),
            block_tables,
            request_positions: RowCursors::bound(vec![0; rows]),
            fold_requests: rows as i64,
            // AGREES WITH THE MAP WIDTH — see `table`.
            fold_pages: pages as u64,
            mask_rep_stride_bytes: 0,
            mask_blocks: 0,
            int_rep_stride_bytes: 0,
        }
    }

    /// A pool cut into equal consecutive runs: row `r`'s logical page `lp` is physical
    /// `first_row * per + r * per + lp`. This is what `PoolSplit` hands out.
    fn runs(rows: usize, pages: usize, per: i64, first_row: i64) -> SessionKv {
        let mut block_tables = Vec::new();
        for r in 0..rows {
            let mut bt = Vec::new();
            for lp in 0..pages {
                bt.push((first_row + r as i64) * per + lp as i64);
            }
            block_tables.push(bt);
        }
        SessionKv {
            block_tables,
            request_positions: RowCursors::bound(vec![0; rows]),
            fold_requests: rows as i64,
            ..pool(0, 0, [0; 6])
        }
    }

    /// 🛑 THE PROOF THE COLLAPSED FOLD RESTS ON: when the maps are classified `Affine`, the ONE stride
    /// a launch steps by reaches EVERY live row's EVERY page — so a single pass addressing
    /// `base + r * stride` reads exactly what `pages x requests` passes read one at a time.
    ///
    /// This is what makes it safe to drop the `x requests` term from [`reps`]. Without it the collapse
    /// is a guess: a launch would step by a stride derived from two rows and read whatever the other
    /// rows' pages happen to be, which produces no crash and no shape error — just fluent wrong tokens.
    /// Written over SYMBOLIC physical pages rather than over runs, so it also covers the pools nobody
    /// designed: it constrains the classifier, not just the allocator.
    #[kani::proof]
    #[kani::unwind(8)]
    fn an_affine_launch_reaches_every_live_row() {
        let rows: usize = kani::any();
        let pages: usize = kani::any();
        kani::assume(rows >= 2 && rows <= 3);
        kani::assume(pages >= 1 && pages <= 2);
        let phys: [i64; 6] = kani::any();
        for p in phys.iter() {
            kani::assume(*p >= 0 && *p <= 4096);
        }
        let s = pool(rows, pages, phys);
        // ⛔ NO `else { return }`. That escape made this proof VACUOUS the moment the classification
        // changed: `LaunchPages::of` now reads a PAGE stripe rather than a request stride, and a proof that
        // quietly returns when its premise stops holding proves nothing. `pool()` builds a striped map by
        // construction, so `Affine` is the claim, not the precondition.
        // ⛔⛔ THE CLAIM IS CONDITIONAL, AND DEMANDING OTHERWISE MADE THIS PROOF **FAIL** — measured, 1 of 703
        // checks, on `assert!(false, "a striped pool must classify Affine")`.
        //
        // An earlier version had `else { return }`, which a comment called VACUOUS and replaced with that
        // assert. Both were wrong, in opposite directions, and the assert was worse: `pool()` fills the map
        // with ARBITRARY SYMBOLIC physical pages (`phys` is `kani::any()`), so it does NOT build a striped map
        // — the "by construction" claim described `runs()`, a different helper. Most instances genuinely are
        // not affine, and the classifier declining them is CORRECT.
        //
        // So assume the classification and prove the REACH. That is not vacuous, because
        // `equal_consecutive_runs_are_affine` proves separately that the striped pools `PoolSplit` actually
        // hands out DO classify `Affine` — the premise is inhabited by another harness, which is the only
        // thing that makes an assumed premise honest. What this adds is the reach guarantee for EVERY pool the
        // classifier accepts, including ones nobody designed.
        let classified = LaunchPages::of(&s);
        kani::assume(matches!(classified, LaunchPages::Affine { .. }));
        let LaunchPages::Affine { stride } = classified else {
            unreachable!("just assumed Affine")
        };

        let r: usize = kani::any();
        let lp: usize = kani::any();
        kani::assume(r < rows && lp < pages);
        // ⭐ THE PAGE IS MINTED FROM THE SESSION NOW rather than passed as a loose (page, row) pair — which
        // is what `RowPage` is for: the row exists, the page is non-negative, and the page lies inside that
        // row's own map, so `page_base_bytes` has no fallback left that could resolve to absolute page 0.
        let rp0 =
            RowPage::of(&s, RowIdx::from_launch_row(0), lp as i64).expect("row 0 owns lp < pages");
        let rpr =
            RowPage::of(&s, RowIdx(r as u64), lp as i64).expect("row r < rows owns lp < pages");
        let base = page_base_bytes(&s, rp0);
        let want = Bytes(base.0.wrapping_add((r as u64).wrapping_mul(stride.0)));
        assert!(page_base_bytes(&s, rpr) == want);
    }

    /// A POOL OF EQUAL CONSECUTIVE RUNS IS ALWAYS AFFINE, AND THE STRIDE IS THE RUN — the classifier
    /// recognises what `PoolSplit` hands out, so the fast path is the normal case rather than a lucky
    /// one. Includes a `first_row` offset because a batch's rows need not start at 0: three requests
    /// holding rows 5,6,7 are as strided as rows 0,1,2.
    #[kani::proof]
    #[kani::unwind(8)]
    fn equal_consecutive_runs_are_affine() {
        let rows: usize = kani::any();
        let pages: usize = kani::any();
        let per: i64 = kani::any();
        let first_row: i64 = kani::any();
        kani::assume(rows >= 2 && rows <= 3);
        kani::assume(pages >= 1 && pages <= 2);
        kani::assume(per >= pages as i64 && per <= 8);
        kani::assume(first_row >= 0 && first_row <= 4);
        let s = runs(rows, pages, per, first_row);
        assert!(
            LaunchPages::of(&s)
                == LaunchPages::Affine {
                    stride: Bytes((per as u64) * 1024)
                }
        );
    }

    /// 🛑 THE FAIL-FIRST CASE: ROWS THAT AGREE ON A STRIDE AND THEN DIVERGE MUST BE `Table`.
    ///
    /// Rows at physical pages 0,1,3 are the counterexample the whole verification loop exists for — the
    /// first two rows agree on a stride of one page, so a classifier that trusts the first difference
    /// calls this affine and a launch built on it reads page 2 for the row that owns page 3. Deleting
    /// the loop in [`LaunchPages::of`] and leaving the candidate stride unverified makes this proof, and
    /// `an_affine_launch_reaches_every_live_row` with it, go RED.
    #[kani::proof]
    #[kani::unwind(8)]
    fn rows_that_diverge_from_the_stride_are_not_affine() {
        let gap: i64 = kani::any();
        kani::assume(gap >= 2 && gap <= 4);
        let mut s = runs(3, 1, 1, 0);
        // 0, 1, then anything but 2.
        s.block_tables[2][0] = 1 + gap;
        assert!(LaunchPages::of(&s) == LaunchPages::Table);
    }

    /// TWO ROWS SHARING A PAGE IS NOT A STRIDE OF ZERO, IT IS A REFUSAL. A zero stride would let one
    /// launch "reach" every row by reading the same page for all of them — two requests writing each
    /// other's KV, which is the same class of bug `SlotMap`'s duplicate-row check refuses up front.
    #[kani::proof]
    #[kani::unwind(8)]
    fn rows_sharing_a_page_are_not_affine() {
        let s = runs(2, 1, 0, 0);
        assert!(LaunchPages::of(&s) == LaunchPages::Table);
    }

    /// THE FALLBACK IS EXACTLY TODAY'S ARITHMETIC, so an unrecognised pool costs correctness nothing and
    /// performance only what it costs now. Also: an op with no request axis keeps its per-request passes
    /// even on a perfectly strided pool — a kernel that cannot step requests must not be asked to.
    #[kani::proof]
    #[kani::unwind(8)]
    fn table_and_an_unbatched_op_both_keep_the_per_request_passes() {
        let n: i64 = kani::any();
        kani::assume(n >= 1 && n <= 4);
        let fold = OpKv {
            page_fold: true,
            batched_requests: false,
            ..Default::default()
        };
        let strided = runs(3, 1, 1, 0);
        assert!(
            LaunchPages::of(&strided)
                == LaunchPages::Affine {
                    stride: Bytes(1024)
                }
        );
        // Affine pool, no request axis ⇒ unchanged.
        assert!(reps(&fold, &strided, n) == n * 3);
        // Request axis, affine pool ⇒ collapsed.
        let batched = OpKv {
            batched_requests: true,
            ..fold
        };
        assert!(reps(&batched, &strided, n) == n);
        // Request axis, unrecognised pool ⇒ unchanged.
        let mut frag = runs(3, 1, 1, 0);
        frag.block_tables[2][0] = 9;
        assert!(reps(&batched, &frag, n) == n * 3);
        // Not the fold at all ⇒ one launch, whatever the pool.
        let plain = OpKv {
            batched_requests: true,
            ..Default::default()
        };
        assert!(reps(&plain, &strided, n) == 1);
    }

    /// A ROW PAST THE TABLE FALLS BACK RATHER THAN INDEXING OUT OF RANGE. A ragged batch asks for rows
    /// it does not own every step, and the mask is what makes that harmless — the same discipline
    /// [`page_base_bytes`] already follows for a page.
    #[kani::proof]
    #[kani::unwind(6)]
    fn a_row_past_the_table_falls_back_to_the_launch() {
        let rows: usize = kani::any();
        kani::assume(rows >= 2 && rows <= 4);
        let p: [i64; 4] = kani::any();
        let s = table(rows, p);
        let launch: i64 = kani::any();
        kani::assume(launch >= 0 && launch <= 1_000_000);
        let req: usize = kani::any();
        kani::assume(req >= rows && req <= 9);
        assert!(
            RequestSlot::resolve(&s, RowIdx(req as u64), SlotPos::of_launch(launch)).pos()
                == SlotPos::of_launch(launch)
        );
    }
}
