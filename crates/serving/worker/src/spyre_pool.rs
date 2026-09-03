// SPDX-License-Identifier: Apache-2.0
//! THE PAGED KV POOL — cutting it, and binding a request's pages into a launch.
//!
//! Target-specific because the addressing is (`bind_launch_slot`, `PageMapCtx`, the fold's page
//! maps), model-NEUTRAL because a page is a page: none of this changes when the architecture does.

use scratchy_subtile::sdsc_abstract::{
    BlockTable, LaunchSlot, PagedKvPool, PoolPages, PoolPartition, RowPages, SlotCount,
};
// ⭐ THE CARD PATH NO LONGER PARSES A MANIFEST. `Manifest` survives only for
// the KTIR-emulator session, whose `new_multi` takes `&Manifest` to thread its
// HBM buffers. Under `sendnn` every fact it carried comes from the GENERATED
// `SUPERDSC_WIRINGS` static instead, so the type is not even in scope.
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::Manifest;
// `--target sendnn` swaps the KTIR emulator runner for the on-silicon sendnn
// runner; the bundle type + session type are cfg-selected, everything else
// (weight load, dynamic sources, KV loop, sampling) is shared.
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::KtirBundle;
#[cfg(feature = "sendnn")]
use scratchy_target_spyre::manifest::SengraphBundle;
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "sendnn")]
use scratchy_target_spyre::sdsc_runner::SuperDscSession;

use crate::error::ExecutorResult;
use crate::spyre_types::*;
use crate::spyre_worker::*;

/// ⭐ INSTALL THE PAGES THE HOST GAVE `req`, and refuse if they do not reach the slot the forward is
/// about to write. Called before every forward.
///
/// 🛑 **IT USED TO ALLOCATE.** `ensure_pages` searched the pool for a page no live request held and
/// pushed it onto the request's map, growing the map to whatever slot the batch needed. That is a second
/// allocator over a pool the SCHEDULER already owns and refcounts, and it is exactly why prefix caching
/// was off: a cache hit is the scheduler saying "these blocks hold that prefix", and pages the worker
/// picked for itself hold it nowhere. The host's `block_ids` are the truth now, and [`BlockTable`] has no
/// constructor that does not take them.
///
/// ⛔ SO THE FAILURE MODE MOVES, ON PURPOSE. Where the old code would silently draw another page, this
/// refuses: the request's slot span has outrun the blocks the host allocated for it. That happens when
/// the two disagree about how many slots a request occupies, which is one number the worker reports every
/// step ([`ReqState::kv_extent`]) — so a refusal here names a reporting bug, and it is a bug report
/// instead of a request reading a page nobody wrote.
#[cfg(feature = "sendnn")]
pub(crate) fn install_host_blocks(
    session: &mut SendnnSession,
    req: &mut ReqState,
    // SLOTS THIS FORWARD WILL HAVE OCCUPIED — a [`SlotCount`], not a `usize`, because the only thing done
    // with it is a division by a page ([`RowPages::holding`]), and the page count that division produces
    // was the same machine type as a slot count at all three sites that used it.
    span: SlotCount,
    // ⛔ NO `held: &HashSet<KvRow>`: this took the set of rows live requests were holding so it could claim a
    // free one. Nothing is claimed here now.
    // The host's list for this request (borrowed from `InputBatch`, the one store) plus the pool's owner
    // split — see [`PageMapCtx`] for why the two travel together.
    ctx: PageMapCtx<'_>,
) -> ExecutorResult<()> {
    // ONE variant, so this binds rather than matches — the sengraph families it used to fall
    // through for are gone.
    let SendnnSession::SuperDsc(sb) = session;
    let Some(split) = sb.kv_split() else {
        if sb.decode.is_paged() {
            return Err(werr(format!(
                // A page holds SLOTS, not requests, so the only way this fails is an empty pool or a
                // batch wider than the PAGES available — one page each, minimum. The old message also
                // blamed "the widest baked rung is wider than the requests a page holds", which was the
                // request-dimension bound and no longer exists.
                "superdsc paged: a pool of {} page(s) cannot serve a {}-wide batch — every sequence needs \
                 at least one page of its own, so raise SUPERDSC_POOL_PAGES or admit fewer requests",
                sb.pool_pages,
                sb.decode_rungs
                    .iter()
                    .map(|r| r.seqs.get())
                    .max()
                    .unwrap_or(1),
            )));
        }
        return Ok(());
    };
    // ⛔⛔⛔ NOTHING IS CLAIMED HERE ANY MORE. This used to take the lowest KV row no live request held
    // (`free_row`) and keep it for the request's life, because the row named the pages that backed its
    // batched write. The host allocates that page now (`own pages + 1`), so a request needs no identity in
    // the pool at all: its pages are the scheduler's block ids and nothing else. `PagedKv`, `KvRow`,
    // `AffineRows`, `free_row` and `rows_held` are gone with it, and so is the refusal that fired when every
    // row was held — a concurrency limit that existed only because rows were a scarce per-request resource.
    let _ = &split;
    // ⛔ ONE PAGE, ONE MEANING. The bundle carries its own `page_slots` while the pool's mask geometry is
    // baked from the const; they are the same quantity from two sources, so they are compared once here
    // instead of being used interchangeably.
    if sb.decode.page_slots != PagedKvPool::PAGE_SLOTS {
        return Err(werr(format!(
            "superdsc paged: the bundle says a page holds {} slot(s) but the pool's baked geometry says {} \
             — the mask is blocked per (row, page) from the const, so nothing can honour both",
            sb.decode.page_slots,
            PagedKvPool::PAGE_SLOTS,
        )));
    }
    // ⛔ THE CONTEXT CEILING. `MAX_PAGES_PER_ROW` exists because the prefix mask costs
    // `pages * width^2 * nqh * 512` bytes against a BAKED segment (`decode_mask_bytes`), so a request
    // deeper than the ceiling makes the fold read mask blocks the host never staged — zero bytes, VALID in
    // an additive mask, and the row answers with one EOS token. A loud refusal is a bug report; a collapse
    // is a wrong answer that reads as fluent.
    let Some(want) = RowPages::holding(span).within_mask_reach() else {
        return Err(werr(format!(
            "superdsc paged: request needs {} slot(s) = {} page(s) but a row may hold at most {} page(s). \
             That ceiling is the PREFIX MASK's, not the pool's: the mask costs `pages * width^2` blocks \
             against a baked segment, so at the widest baked rung ({}) only {} page(s) fit. Serve a shorter \
             context, or make the mask cost one block per PAGE instead of per (row, page).",
            span.get(),
            RowPages::holding(span).get(),
            PagedKvPool::MAX_PAGES_PER_ROW,
            PagedKvPool::WIDEST_BATCH_RUNG,
            PagedKvPool::MAX_PAGES_PER_ROW,
        )));
    };
    // ⭐ AND THE MAP MUST BE BUILDABLE — the host's blocks at the logical pages this request holds keys in,
    // one shared scratch page for the fully-masked ones. `None` now means only one thing: the host granted
    // fewer blocks than those pages need.
    //
    // ⛔ THE MESSAGE NO LONGER OFFERS `HOLE_PAGES_PER_ROW` AS THE REMEDY, because there is no reserve to
    // raise. This is a DISAGREEMENT between the scheduler's allocation and the launch's page count, and the
    // scheduler's rule is `KvSlotSpan::blocks_this_step` = own pages + one for the shared write page.
    if req.page_map(ctx, want).is_none() {
        return Err(werr(format!(
            "superdsc paged: cannot map {} slot(s) = {} page(s): the host granted {} block(s). The pages a \
             row holds keys in must all be host blocks — the scheduler allocates its own pages plus ONE for \
             the batch's shared write page (`KvSlotSpan::blocks_this_step`), so this says that allocation \
             and this launch's page count disagree, which is a reporting bug in `ReqState::kv_extent`",
            span.get(),
            want.get(),
            ctx.host.len(),
        )));
    }
    Ok(())
}

/// ⭐ EVERYTHING NEEDED TO DERIVE ONE REQUEST'S PAGE MAP, as a single borrowed value: the host's block
/// list — borrowed from [`InputBatch`], the ONE store — and the pool's owner split.
///
/// One parameter rather than two because they are only ever correct TOGETHER: `map_row` bounds every host
/// id by `part.host_blocks()`, so a list from one pool checked against another partition would admit an id
/// that names a reserved hole page. Threading them separately is what lets those two drift.
#[cfg(feature = "sendnn")]
#[derive(Clone, Copy)]
pub(crate) struct PageMapCtx<'a> {
    /// The scheduler's block ids for this request, block `j` = its TOKEN page `j`.
    pub(crate) host: &'a [usize],
    pub(crate) part: PoolPartition,
}

/// THE POOL'S OWNER SPLIT for the loaded bundle: which pages the host allocates from, and which back the
/// batched write's hole. One derivation, so the count reported to the host
/// (`kv_cache_num_blocks_override`) and the bound `map_row` checks ids against cannot disagree.
#[cfg(feature = "sendnn")]
pub(crate) fn pool_partition(sb: &SuperDscBundle) -> ExecutorResult<PoolPartition> {
    let pool = PoolPages::of_pool(sb.pool_pages)
        .ok_or_else(|| werr("superdsc paged: the pool has no pages — nothing can be served"))?;
    PoolPartition::of_pool(pool, sb.pool_rows).ok_or_else(|| {
        werr(format!(
            "superdsc paged: a {}-page pool cannot serve {} launch row(s): the pool reserves {} shared \
             scratch page and the host still needs a page per row. Give the pool a bigger memory budget, or \
             bake a narrower decode ladder",
            sb.pool_pages,
            sb.pool_rows.get().get(),
            PoolPartition::reserve(),
        ))
    })
}

/// Point ONE session at the pages a request owns. Needed for EVERY session that forwards for it:
/// prefill rungs are separate sessions sharing seg2 by alias, each with its own page map.
#[cfg(feature = "sendnn")]
pub(crate) fn bind_request_pages(
    session: &mut SuperDscSession,
    req: &ReqState,
    ctx: PageMapCtx<'_>,
) -> ExecutorResult<()> {
    let at = scratchy_subtile::sdsc_abstract::BatchSlot::solo(&req.kv_hist);
    bind_request_pages_at(session, req, LaunchSlot::solo(), at, ctx)
}

/// Install ONE LAUNCH SLOT: which pool row it addresses, which pages, and the KV slot it writes at.
///
/// The row is passed rather than read off a request because a launch has slots a request does not own —
/// a rung is the smallest baked width holding the live count, so the tail is padding — and the baked
/// addressing gives slot `i` row `row0 + i` whether or not anyone lives there. Installing what the
/// bundle actually does is what lets `fold_plan::LaunchPages` see a single stride and collapse the
/// fold's passes; installing live 0's row for the padding made the row list `row0, row0+1, …, row0`,
/// which admits no stride and refuses the collapse with no symptom but the missing speed.
#[cfg(feature = "sendnn")]
/// EVERY PRINT-ONLY FLAG THIS FILE READS ON A PER-STEP OR PER-LAUNCH PATH, RESOLVED ONCE PER PROCESS.
///
/// ⛔⛔⛔ `std::env::var_os` LOCKS THE ENVIRONMENT AND WALKS IT, and these sit in `bind_request_pages_at` (per
/// request per step), `run_prefill_batch` (per prefill chunk AND per batched decode step — six of them) and
/// `superdsc_forward_chunk`. That is ~10 environment walks per token here, on top of the hundreds
/// `fold_plan::reps` and `trace_pass` were doing per op and per pass. Host time paid every token to answer a
/// question whose answer cannot change: nothing mutates the environment mid-run.
///
/// ⭐ `superdsc_exec::Diag` ALREADY DID EXACTLY THIS (one `OnceLock`, ten flags) — these two files were the
/// outliers, and both are on the path the C++ shim used to own. The port made every token pay for its traces.
/// An instrument may print; it must not cost. See [[never-env-gates-no-reward-hacking]].
// ⛔ `sendnn`-ONLY, restored from 8bd5c755b. My branch was cut BEFORE that commit added
// these gates, and resolving the rebase conflict in favour of the split took the whole
// file — which discarded them. They are what makes the KTIR-only build compile.
#[cfg(feature = "sendnn")]
pub(crate) struct Flags {
    pub(crate) phase_time: bool,
    pub(crate) timing: bool,
}

// ⛔ `sendnn`-ONLY, restored from 8bd5c755b. My branch was cut BEFORE that commit added
// these gates, and resolving the rebase conflict in favour of the split took the whole
// file — which discarded them. They are what makes the KTIR-only build compile.
#[cfg(feature = "sendnn")]
pub(crate) fn flags() -> &'static Flags {
    static F: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        let on = |k: &str| std::env::var_os(k).is_some();
        Flags {
            phase_time: on("SCRATCHY_SDSC_PHASE_TIME"),
            timing: on("SCRATCHY_SUPERDSC_TIMING"),
        }
    })
}

// ⛔ `sendnn`-ONLY, restored from 8bd5c755b. My branch was cut BEFORE that commit added
// these gates, and resolving the rebase conflict in favour of the split took the whole
// file — which discarded them. They are what makes the KTIR-only build compile.
#[cfg(feature = "sendnn")]
pub(crate) fn bind_launch_slot(
    session: &mut SuperDscSession,
    slot: LaunchSlot,
    at: scratchy_subtile::sdsc_abstract::BatchSlot,
    // THE MAP THIS LAUNCH BINDS — derived by the caller from the host's list and this row's history
    // (`ReqState::page_map`), never a stored list.
    pages: &BlockTable,
) -> ExecutorResult<()> {
    // ⛔⛔⛔ THIS TOOK AN `AffineRows` WITNESS, AND IT WAS PROVING SOMETHING NOTHING NEEDED.
    //
    // The argument existed so a caller could not reach this function without having shown the batch's KV rows
    // were CONSECUTIVE (`kv_rows[0] + i`), and the row it derived was then used for... the page trace below.
    // `set_block_table` takes a launch slot, a write slot and a page list; it has never taken a row. With the
    // reserve gone there are no rows at all, so the witness, the consecutiveness proof and the
    // "N live request(s) do not hold consecutive KV rows — decoding them one at a time" fallback go together.
    if !session.is_paged() {
        return Ok(());
    }
    if pages.is_empty() {
        return Err(werr(
            "superdsc paged: launch slot has no pages — the scheduler's block table must be installed \
             before the first forward (an empty map would address another request's KV)"
                .to_string(),
        ));
    }
    let pages: Vec<i64> = pages.as_i64();
    session
        .set_block_table(
            scratchy_target_spyre::fold_plan::LaunchSlotIdx::of_launch(slot.index() as i64),
            i64::from(at.get()),
            &pages,
        )
        .map_err(|e| werr(format!("superdsc paged: set_block_table: {e}")))
}

/// Install request `slot` of a batch: its page map and the KV slot its next token writes at.
///
/// Called once per scheduled request before a batched forward.
///
/// ⛔ `at` IS A [`BatchSlot`], NOT A POSITION, and that is the whole point of the type: this call is the
/// only way a write slot reaches the device, and it now cannot be reached with a token count. A batch
/// binds every one of its requests at the SAME slot — a launch resolves one slot shift for every trip
/// inside it, so per-request slots mean one launch per request — and `BatchSlot` is the value that can
/// only have come from the live requests' own histories.
#[cfg(feature = "sendnn")]
pub(crate) fn bind_request_pages_at(
    session: &mut SuperDscSession,
    req: &ReqState,
    slot: LaunchSlot,
    at: scratchy_subtile::sdsc_abstract::BatchSlot,
    ctx: PageMapCtx<'_>,
) -> ExecutorResult<()> {
    let pos = i64::from(at.get());
    if !session.is_paged() {
        return Ok(());
    }
    // ⛔ THERE IS NO ROW TO FETCH FIRST. This began by reading the request's `PagedKv` and refusing if it
    // held none — "request holds no KV row" — because the map's write page came from that row's reserved
    // pages. The map is now a function of the host's list and this request's history alone, so there is no
    // pre-forward state to be missing and no such refusal to make.
    let want = RowPages::holding(SlotCount::new(at.get().saturating_add(1)));
    let map = req.page_map(ctx, want).ok_or_else(|| {
        werr(format!(
            "superdsc paged: cannot map {} page(s): the host granted {} block(s). Every page holding a key \
             must be a host block — the scheduler allocates this request's own pages plus ONE for the \
             batch's shared write page, so this says its allocation and this launch disagree",
            want.get(),
            ctx.host.len(),
        ))
    })?;
    let pages: Vec<i64> = map.as_i64();
    session
        .set_block_table(
            scratchy_target_spyre::fold_plan::LaunchSlotIdx::of_launch(slot.index() as i64),
            pos,
            &pages,
        )
        .map_err(|e| werr(format!("superdsc paged: set_block_table: {e}")))
}

/// Index of the SMALLEST prefill rung that can hold `real` query rows. `rungs` is ascending, and the
/// caller caps its chunk at the top rung, so this only returns `None` for an EMPTY ladder.
#[cfg(feature = "sendnn")]
pub(crate) fn prefill_rung_for(rungs: &[(usize, SuperDscSession)], real: usize) -> Option<usize> {
    rungs
        .iter()
        .position(|(m, _)| *m >= real)
        .or_else(|| rungs.len().checked_sub(1))
}

/// The decode-slot sentinel prefix the SuperDSC codegen stamps into
/// `group.decode.graph_json` in place of a real sengraph JSON. The bytes after the
/// prefix are the dxp-bundle fingerprint `fp` (the `bundle_code` registry key).
#[cfg(feature = "sendnn")]
pub(crate) const SUPERDSC_SENTINEL: &str = "SUPERDSC_BUNDLE:";

/// True iff the baked bundle is the SUPERDSC dxp bundle. The SuperDSC emitter does
/// NOT produce a DeepTools-compilable sengraph JSON; instead the codegen stamps the
/// single group's decode-slot `graph_json` with the `SUPERDSC_BUNDLE:<fp>` sentinel
/// (the dxp bundle is a DIRECTORY of per-supernode dxp blobs + plan, recovered at
/// load via `bundle_code::bundle(fp)`). This MUST be checked BEFORE
/// [`is_superdsc_bundle`]: the sentinel is not a
/// real graph, so those predicates' `PagedAttn` / `prefill==decode` substring tests
/// would misclassify it (e.g. default-mode also has one group with prefill==decode).
#[cfg(feature = "sendnn")]
pub(crate) fn is_superdsc_bundle(bundle: &SengraphBundle) -> bool {
    bundle.groups.len() == 1
        && bundle
            .groups
            .first()
            .is_some_and(|g| g.decode.graph_json.starts_with(SUPERDSC_SENTINEL))
}
