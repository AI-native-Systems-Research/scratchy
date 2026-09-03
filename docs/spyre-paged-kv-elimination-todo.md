# TODO: eliminate the spyre-specific paged-KV structs

**Status: steps 1-2 done and card-verified; step 3 is next.** Four structs remain. This document exists so that is not forgotten again, and so the
next attempt starts from what is measured rather than from what sounds right.

## What has already moved to the common crates

| was | now |
|---|---|
| `ReqState::host_blocks` / worker-side free list | `InputBatch::block_tables` (the store cuda + metal use) |
| `ReqState::tokens`, `ReqState::prompt_len` | `InputBatch::prompt` + `generated`; `prompt_len()` is DERIVED |
| `ReqState::n_computed` | `InputBatch::tokens_in_pool` (+ `set_tokens_in_pool`) |
| `SessionKv::kv_rows` (the pool row, in the DEVICE layer) | deleted — it had no readers (`5e319159`) |
| `SuperDscBundle::prefill_pending` | deleted — unreachable (`1d56d5e0`) |
| four behaviour env-gates | deleted; the surviving vars only PRINT |

`ReqState` is down to `kv_hist` + the KV row + the emulator's host mirrors.

## What remains, and the one thing keeping it

- `KvHistory` — runs, not a length, because a batched step appends every row at ONE shared slot, so a shorter
  request carries a masked hole and slot != token.
- `KvRow` / `PagedKv` — the row identity.
- `PoolPartition` — the hole reserve (`rows * HOLE_PAGES_PER_ROW + 1` pages) plus one shared scratch page.
- `BlockTable::map_row` — derives the page map because slot-page != token-page once a hole exists.

All four hang on **one** question: who owns the page a batched step writes when a request's next key lands past
its own tokens?

- **A (today)** the worker, from `PoolPartition::hole(row, i)`.
- **B** the host, by allocating every live request out to the step's reach — which deletes all four.

### What is measured about that choice

`BlockTable::hole_pages_used()` counts A actually firing (`1294e03e`). Two measurement rounds, and **they are
not comparable to each other** — the probe sets differ, which is the whole reason the numbers are written down
with their probes:

| probes | firings |
|---|---|
| ragged + real cache hits, 367..669 tok (7 runs) | 0 |
| long ragged, 1775..3025 tok (ON and OFF) | 0 |
| the 6-probe set, 76..1840 tok (ON and OFF) | 5 per run, 22 hole pages |
| **the gate's four axes at `--max-num-seqs 4`, 2026-08-14** | **2-4 pages on EVERY batch run; 0 on every solo run** |

The last row is the current instrument. It fires on `hits`, `short` and `longhits` alike — so **A is
load-bearing on every batched axis**, not just on a wide length spread.

⚠️ It is NOT caused by sizing the pool from the declaration. Verified by composition: the same four axes produce
byte-identical chunk and hit counts before and after that change (`21/3`, `32/0`, `120/0`, `61/3`), so scheduling
did not move. The earlier zeroes were measured on different probes, before the `longhits` axis existed.

`95e98df6` reported a pool-wide reach so the host could own more of it, and that cut the caching-off case from
22 hole pages to 16 — half the window. **The residual window is exactly:** a request ADMITTED this step raises
the pool reach (its prompt joins the max) for requests already running, whose allocation was sized by a report
made before that request existed.

`tests/zz_the_hole_reserve_is_cheaper_than_host_allocation.rs` compares the costs (A flat in depth, B linear):

```
rows=4  deepest=1840 tok    A=9 pages    B=24 pages
rows=4  deepest=3667 tok    A=9 pages    B=45 pages
rows=32 deepest=1840 tok    A=65 pages   B=248 pages
rows=4  deepest= 512 tok    A=9 pages    B=6 pages     <- B wins below ~768 tok
```

## ⛔ AND WHY THAT COST TABLE IS NOT A REASON TO STOP

Those are ratios **inside a pool whose size nobody asked for**:

```rust
const DEFAULT_POOL_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;   // hardcoded
let afford = (DEFAULT_POOL_BUDGET_BYTES / stride) as usize;
let pages  = afford.clamp(widest, DEFAULT_POOL_PAGES * widest);   // [32, 256]
```

`--max-model-len` was not read. `--max-num-seqs` was not read until the hole-reserve change. And then the
declared context length was silently truncated to fit the constant:
`Capping max_model_len 131072 -> 4096 (paged-attention addressable limit)`.

**Replaced by `PoolDemand::of_declaration`**, and measured on granite-3.1-8b fp8:

```
superdsc paged: KV pool 37 page(s) = --max-model-len 4096 (16 page(s) per request) x 2 launch row(s)
                + reserve — 2220 MB of a 8192 MB budget at 60 MB per page
```

73 pages at `--max-num-seqs 4`, where the constant took **121 regardless of what was asked for**. Two Kani
proofs hold the conservation law that the reserve's two callers — one sizes a pool to include it, one splits a
pool by it — cannot drift apart: `a_pool_sized_by_its_demand_always_splits` (383 checks) and
`the_reserve_is_exactly_what_the_split_withholds` (352). ⛔ They need `--features superdsc` or kani finds ZERO
harnesses and says `no harnesses matched`, which reads like a typo.

**So the honest order of work is:**

1. ✅ **DONE (card-verified on granite-8b fp8).** **Size the pool from the declared workload**: `admitted_width * pages_for(max_model_len)`, from the CLI, not
   a constant. The worker does not currently receive the CLI `max_model_len` — it reads
   `hf_config.max_position_embeddings` (the model's capability). `WorkerCreateConfig::max_model_len` already
   exists and needs plumbing to `SpyreWorker`, the same way `set_admitted` plumbs `max_num_seqs`.
2. ✅ **DONE.** **Refuse at load** if the memory budget cannot hold that, naming both numbers — do not serve 4096 tokens
   when 131072 was asked for. (See `capping-is-not-validating`.)
3. ⬅ **NEXT, and it is a scheduler change touching cuda/metal — its own commit, its own test round on both
   models.** The scheduler can allocate every request out to the step's reach without exhausting the pool: it
   already knows every prompt length, so it takes `max(reported reach, max prompt over the requests it is about
   to schedule)` before allocating any of them. This is a change to the scheduling loop, which allocates for
   every backend — cuda cannot be compiled on the dev laptop (no `nvcc`), so it needs care, not avoidance.
4. **Verify** `hole_pages_used()` is 0 across every axis of `scripts/spyre-batch-gate.sh`.
5. **Delete** `PoolPartition`'s reserve, `KvRow`, `AffineRows`, `free_row`, the "decode one at a time" fallback,
   and reduce `map_row` to the identity. `KvHistory` collapses to a length once slot == token.

## Gate for any of this

`scripts/spyre-batch-gate.sh <pod> [dir] [max-num-seqs]` — axes `hits`, `short`, `long`, `longhits`. All pass on
granite-3.1-2b fp8 (hd=64) and on granite-3.1-8b fp8 (hd=128) except the 8b's near-tie probes, which are
explained by the prefill not being chunk-width invariant (a property, recorded in memory, not a defect).

✅ **The gate measures latency now** — the `itl` axis (in the default set) runs five `--bench` passes and reports
**p50 grouped by the weight segment's memory region**, never a mean, failing on more than one region across runs,
any `RAS::FLEXALLOCATOR` line, or >2% p50 spread. A mean over a bimodal distribution is how a 4% decode
regression survived ~100 commits.
