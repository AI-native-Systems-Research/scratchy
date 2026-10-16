//! THE ADDRESSING WELL-FORMEDNESS PASS — the general compile-time lockdown.
//!
//! The campaign's standing directive: **no addressing faults at runtime — the code
//! cannot fault.** Every value the device turns into an address (a gather operand,
//! an `idx2addr` composition) must be provably inside the buffer it names, at BUILD
//! time, with the op, the law and the numbers in the refusal. This module is that
//! proof obligation, stated once, over the graph — run from the bake heads (the
//! spyre target's `lower_subtile_tape_to_ktir`: both its unrolled entry and its
//! re-rolled tape entry) so every bake runs UNDER it before a single program is
//! minted.
//!
//! ## The layering — three halves, one lockdown
//!
//! 1. **The binding half** (landed, M6 `972185f72`): the splice's `ExpertMatmul`
//!    binding walk requires the routing to descend from the router's `RouteTopK`
//!    with `k <= E` — the PRODUCER-side provenance. `RouteTopK` is the only mint of
//!    in-range ids (the argsort's `+inf` pad lanes sort last), so a hand-fed source
//!    can hold any id and is refused there by name.
//! 2. **The door half** (landed, step 4 `d13f6e8e3` + the doors' own laws):
//!    `expert_matmul`'s `k <= e_count` at the bank; `attach_gather_index`'s
//!    structural refusals (index==value, a paged dim the op lacks, partial entries);
//!    the one-index-stick ceiling (`CopyDims::ENTRIES_PER_OP`); the entry-axis-only
//!    split law; the `"index"`-typed `GatherIndexConversion` EAR-range law.
//! 3. **THIS PASS — the graph half**, the general sweep the two guards generalize
//!    into: enumerate every reader of every INDEX-MINTED tensor, bound each mint,
//!    and verify the composition against the budgeted extents. What it adds beyond
//!    the two landed guards:
//!    - **`E <= e_count`** — the id RANGE bound (the router's expert count) against
//!      the bank's BUDGET (the experts its rows pack). The landed guards bound the
//!      COUNT (`k <= E`, `k <= e_count`) but never the range: at `E > e_count` an
//!      id in `[e_count, E)` composes a weight-gather address past the bank's end —
//!      a clean bake that faults the card's FETCH (the wall-10 class).
//!    - **The closed consumer table** — a `SubOp` reading an index-minted tensor at
//!      a position with no stated read law is an E0004-style refusal: a new
//!      index-consumer class needs a bound proof HERE before it can bake, not a
//!      silent fall-through to whatever its door happens to do with the values.
//!
//! ## The closed class tables
//!
//! **Index mints** — graph tensors whose VALUES become addresses:
//! - `RouteTopK`'s output: expert ids `< E` (its router's expert count).
//! - `ExpertSort`'s output: the SAME ids, rearranged — the sort permutes pairs, it
//!   never mints an id, so its output carries the top-k's bound unchanged.
//!
//! **Door-internal mints** (no graph tensor; stated here so the enumeration is
//! closed): the `dl16tofp32`/`fp32toint32` convert chain is range-preserving over
//! the routing buffer it reads; the fine-index gather mints `xsub` from the
//! `[E, sub]` const bank whose own rows bound it. Both are inside one
//! `expert_matmul` door call, guarded by the door's own arithmetic.
//!
//! **Runtime-bound index tables** (not graph tensors): the KV block table (the
//! worker's allocator mints only pool pages — the reserved `kv_block_index` tid),
//! and the embedding/sample-row tables (`EmbeddingGather` & co. are refused as
//! target-unsupported on this ladder, so no proof is owed yet). A future door that
//! gathers by a runtime table states its law here when it lands.
//!
//! **The consumer table** — who may read an index-minted tensor, and how:
//! - `ExpertMatmul[1]` — as IDS: the addressing composition, checked below
//!   (the bank budget vs the mint's bound).
//! - `ExpertMatmul[0]` — as data (the pair rows the sort wrote).
//! - `ExpertSort[0]`/`[1]` — the tokens as data; the ids DATA-INDEPENDENTLY (the
//!   sort's door never reads the values — the copy is data-independent, M6's own
//!   citation).
//! - `ExpertUnsort[0]` — as data; `[1]` NOT AT ALL (the door binds one data
//!   operand; the routing rides in the graph).
//! - `RouteGatherScores[0]`/`[1]` — the scores as data; the ids by COMPARE-SELECT
//!   (an id past `E` selects no row and the output reads 0 — wrong output, never a
//!   fault; the identity-table rows it does address are `j < k <= E <= W`).
//! - `RouteExpertScale[0]`/`[1]`/`[2]` — the same compare-select law on the ids,
//!   the scores and scale row as data.
//!
//! Anything else reading a minted tensor refuses. The front end's every MoE arm
//! (gemma's, the fused Mixtral/Qwen arm's, gpt-oss's) builds exactly the tabled
//! chain — `route_top_k` → `RouteGatherScores`/`RouteExpertScale` → `experts()`'s
//! sort/matmul/unsort/combine — so an honest model never meets the refusal; a NEW
//! consumer class meets it with its own name in the message.

use std::collections::HashSet;

use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR, SubtileNode, TensorId};

/// Verify every index→address composition in `ir` — the bake's first act.
///
/// Walks the whole graph once: every node that READS an index-minted tensor must be
/// in the consumer table (`read_law` below), and every `ExpertMatmul` must compose
/// its routing's bound (`ids < E`, `k <= E` — the mint's laws, restated here at the
/// bake head where they run first) against the experts its two banks actually pack.
/// Every refusal names the op, the law and the numbers.
pub fn verify_addressing<F: RopeForm>(ir: &SubtileIR<F>) -> Result<(), String> {
    // The mints — collected BEFORE the walk so the sweep asks one question per read
    // ("is this tensor's producer a mint?") and never re-derives it per node.
    let mints: HashSet<TensorId> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::RouteTopK { .. } | SubOp::ExpertSort { .. }))
        .map(|n| n.output.tensor)
        .collect();
    for node in &ir.nodes {
        sweep_consumers(node, &mints)?;
        if let SubOp::ExpertMatmul { n, .. } = &node.op {
            expert_matmul_composes_within_its_banks(node, ir, *n)?;
        }
    }
    Ok(())
}

/// The read law for `(op, position)` — `None` means "no bound proof covers this
/// read", which [`sweep_consumers`] refuses. ⛔ TOTAL MATCH, NO `_` ARM: a new
/// `SubOp` that reads a minted tensor must state its law HERE to bake — the
/// E0004 discipline the addressing lockdown owes every new index consumer.
fn read_law<F: RopeForm>(op: &SubOp<F>, pos: usize) -> Option<&'static str> {
    match (op, pos) {
        (SubOp::ExpertMatmul { .. }, 0) => Some("data — the pair rows"),
        (SubOp::ExpertMatmul { .. }, 1) => Some("ids — the bank composition, checked below"),
        (SubOp::ExpertSort { .. }, 0) => Some("data — the tokens"),
        (SubOp::ExpertSort { .. }, 1) => {
            Some("ids — data-independently (the sort's door never reads the values)")
        }
        (SubOp::ExpertUnsort, 0) => Some("data — the pair rows"),
        (SubOp::ExpertUnsort, 1) => Some("routing — never read (the door binds one data operand)"),
        (SubOp::RouteGatherScores, 0) => Some("data — the scores"),
        (SubOp::RouteGatherScores, 1) => {
            Some("ids — compare-select (an id past E selects no row, never a fault)")
        }
        (SubOp::RouteExpertScale { .. }, 0) => Some("data — the scores"),
        (SubOp::RouteExpertScale { .. }, 1) => {
            Some("ids — compare-select (an id past E selects no row, never a fault)")
        }
        (SubOp::RouteExpertScale { .. }, 2) => Some("data — the scale row"),
        _ => None,
    }
}

/// One node's reads against the consumer table — the E0004 sweep. A minted tensor
/// at a position with no [`read_law`] is refused with both names (the reader and
/// the mint) and the table's members, so the author of a new consumer class knows
/// exactly where the proof goes.
fn sweep_consumers<F: RopeForm>(
    node: &SubtileNode<F>,
    mints: &HashSet<TensorId>,
) -> Result<(), String> {
    for (pos, input) in node.inputs.iter().enumerate() {
        if !mints.contains(&input.tensor) || read_law(&node.op, pos).is_some() {
            continue;
        }
        return Err(format!(
            "addressing: {:?} t{} reads t{} at input {pos}, and that tensor is an INDEX MINT \
             (a RouteTopK's or ExpertSort's output — its values become gather addresses). That \
             read has no bound proof in the consumer table, so it cannot bake: a new \
             index-consumer class states its read law in the splice's `addressing` pass first. \
             The tabled readers are ExpertMatmul[0]/[1] (pair rows as data; ids as the bank \
             composition), ExpertSort[1] (data-independent), ExpertUnsort[1] (never read), and \
             RouteGatherScores[1]/RouteExpertScale[1] (compare-select)",
            node.op,
            node.output.tensor.index() as u32,
            input.tensor.index() as u32,
        ));
    }
    Ok(())
}

/// The `ExpertMatmul` composition — the one graph-level read whose values become
/// ADDRESSES on the card (the door's weight/scale bank gathers and the m>1 gathered
/// matmuls all compose `bank_base + id · skip`, with no bound the card enforces).
///
/// The chain, restated at the bake head (the binding-time twin is `bindings_for`'s
/// M6 walk, which also RESOLVES the binding and stays the guard at mint time):
/// `routing <- ExpertSort <- RouteTopK`, then
///
/// - `k <= E` (the mint's law: past it the argsort's `+inf` pad lanes enter the
///   top-k and their ids run past the bank — the wall-10 FETCH fault), and
/// - `E <= e_count` for BOTH banks — the budget the landed guards never tied:
///   `k <= e_count` bounds how MANY ids each token carries, not how LARGE they are.
fn expert_matmul_composes_within_its_banks<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    n: u32,
) -> Result<(), String> {
    let out_tid = node.output.tensor.index() as u32;
    if n == 0 {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} states n=0 — a zero-width projection packs \
             its banks into zero experts and the id bound is undefined"
        ));
    }
    // ⛔ THE SPLICE'S BANK FORM BINDS EXACTLY FOUR — [rows, routing, codes, scales].
    // `bindings_for` indexes inputs[2]/[3] unguarded, so a shorter arity is a PANIC
    // there today; refused here, by name, it is a build error instead.
    if node.inputs.len() != 4 {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} takes {} input(s) — the splice's bank form \
             binds exactly four ([rows, routing, codes, scales]), and a different arity has no \
             binding law and no bound proof",
            node.inputs.len()
        ));
    }
    // ── 1. the mint's laws: the routing descends from a top-k over E experts ──
    let routing = node.inputs[1].tensor;
    let Some(sort) = crate::producer_of(ir, routing) else {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} reads its routing from t{} which no node \
             produces — a source tensor is not the router's RouteTopK, the only mint of \
             in-range ids, so the gather would read whatever it holds past the bank",
            routing.index() as u32
        ));
    };
    let SubOp::ExpertSort { experts, .. } = &sort.op else {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} reads its routing from t{} whose producer is \
             not an ExpertSort — the expert ids cannot be bound, and a projection without them \
             would read slab 0 for every pair",
            routing.index() as u32
        ));
    };
    let e_sort = experts.get();
    if sort.inputs.len() != 2 {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} routes through an ExpertSort with {} input(s) \
             — the sort takes [tokens, ids], and a different arity has no ids to bound",
            sort.inputs.len()
        ));
    }
    let ids = sort.inputs[1].tensor;
    let Some(topk) = crate::producer_of(ir, ids) else {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} reads its expert ids from t{} which no node \
             produces — a source tensor is not the router's RouteTopK, the only mint of \
             in-range ids, so the gather would read whatever it holds past the bank",
            ids.index() as u32
        ));
    };
    let SubOp::RouteTopK { k } = &topk.op else {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} reads its expert ids from t{} whose producer \
             is not the router's RouteTopK — RouteTopK is the only mint of in-range ids (the \
             argsort's +inf pad lanes sort last), so anything else can put an out-of-range id \
             in the gather and fault the card's FETCH",
            ids.index() as u32
        ));
    };
    if k.get() > e_sort {
        return Err(format!(
            "addressing: ExpertMatmul t{out_tid} routes its pairs through a top-k of {} over \
             {e_sort} experts — the top-k would have to include the argsort's +inf pad lanes, \
             and a pad lane's id runs past the bank and faults the card's FETCH. The routing \
             is only in range while k <= the expert count",
            k.get()
        ));
    }
    // ── 2. the budget: the mint's bound vs the experts each bank packs ──
    for (pos, which) in [(2usize, "codes"), (3, "scales")] {
        let bank = node.inputs[pos].tensor;
        let shape = ir.shape(bank);
        let e_count = shape.rows / n;
        if e_sort > e_count {
            return Err(format!(
                "addressing: ExpertMatmul t{out_tid} routes through a top-k over {e_sort} \
                 experts but its {which} bank t{} is [{}, {}] — at n={n} rows per expert that \
                 packs only {e_count}, so an id in [{e_count}, {e_sort}) composes a \
                 weight-gather address past the bank's end and faults the card's FETCH. The \
                 landed guards bound the id COUNT (k <= E and k <= e_count); this is the RANGE: \
                 the composition is only in range while the router's expert count <= the \
                 bank's ({e_sort} <= {e_count})",
                bank.index() as u32,
                shape.rows,
                shape.cols
            ));
        }
    }
    Ok(())
}
