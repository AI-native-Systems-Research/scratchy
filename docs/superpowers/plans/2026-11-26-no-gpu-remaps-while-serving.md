# No GPU memory remaps while serving (KV, snapshots, live state)

## Problem

A server for a hybrid model (Mamba-2 / Gated DeltaNet) freezes for seconds — and freezes the whole
GPU, the window server included — whenever a step maps or unmaps GPU memory under load. Measured on
an M5 Max 64 GB, nemotron-3-nano-30b-a3b, the benchmark's 805-prompt cells, commit before #1257:

| step that remaps | at 0.9 (46.7 GiB) | at 0.7 (36.3 GiB) |
|---|---|---|
| 3–4 KV granules unmapped for a snapshot unit (#525's trade) | 42.7 s, 41.7 s | 4.3 s, 10.2 s |
| live GDN state slots mapped as concurrency rises | 0.3 s, 2.4 s (6–8 s in a b649e08e run) | 0.002 s, 0.7 s |

The same ops take 1–13 ms when the GPU is idle. Headroom below the GPU cap shrinks the stall 4–10×;
it does not remove it. No GPU reset or jetsam event is logged. The cause is #525: it made KV and the
snapshots ONE dynamically-traded budget — granules mapped and unmapped, units mapped and unmapped,
live slots mapped lazily — and every trade under load is a sparse map/unmap the GPU serializes.
#1257's disk tier removed the reason for any of it: an evicted snapshot is not lost.

## Fix — allocate once at startup, never remap

The model's sizes are constants and the device's budget is known at startup. One split, made once:

- **Live state**: one slot per sequence (`max_num_seqs`), all mapped before the server reports
  ready. Its bytes come out of the KV budget (today they are unmetered — the over-consumption the
  0.9 run froze on). A Mac too small for `max_num_seqs` slots lowers `max_num_seqs`, as the
  resolver's `recurrent_per_slot_bytes` already does.
- **The snapshot pool**, all mapped at startup: the snapshots that fit beside the KV of a longest
  prompt — `state_budget / (slot_bytes + max_model_len_kv_bytes)` slots, at least one, none when
  even one does not fit. Past the pool, #1257's disk tier holds every snapshot; a save past the pool
  evicts to disk (the existing path), a hit reads back (~10 ms at uncached-disk speed, overlapped
  with the step).
- **KV**: everything the budget leaves, the whole pool mapped at startup.

Then, deleted — not disabled:

- #525's ledger and every trade: `SharedMemory`, `make_room_for_kv`, `grow_before_evicting`,
  `map_snapshot_unit`, `release_unit`, the `SharedMemoryOps` step messages, and the worker's
  `SharedKv` mirror and `apply_shared_memory`.
- The lazy live-slot mapping (`grow_to(live_reach)` per step) and the granule-apart KV mapping
  (`SparseStores::apart` / `map_apart` / `release`, `with_tail`): live slots and snapshot slots are
  ONE grown range, mapped once; the KV pool is grown once to its capacity.
- The idle KV shrink (`KV_SHRINK_IDLE_AFTER`): it unmaps after 30 s idle and re-faults multi-second
  on the next request. The budget is scratchy's for the process's life, as it is for vLLM.

The disk tier writes around the page cache (`UncachedFile`, already on this branch), and its staging
buffers are reused, not reallocated, after the first steps.

Non-recurrent models keep their KV growth (doubling, never unmapping) — their stalls were never
measured; full startup mapping for them is a follow-up, not this change.

## Cost, stated before building

The live pool was ALREADY metered on main — `gdn_reserve` folds its bytes into `weights_and_overhead`
before the KV budget is computed — so the split's only new cost is the snapshot pool carved out of
what is left. M5 Max, nemotron at 0.9: KV ≈ 21 GiB beside a pool of ~15-20 slots (~1 GiB), against
main's 22.2 GiB that apart-maps its snapshots. A Mac where a longest prompt's KV plus a slot does
not fit what is left — qwen3.6 on this box, whose 262k-token context prices near 10 GiB — floors the
pool at one slot beside the disk tier, as its real life today. Both stated against main in the proof
run. (An earlier draft of this paragraph, and the first implementation, subtracted the live slots a
second time from the KV budget — the law stated twice; found on the qwen3.6 proof run, which
measured 2.07 GiB of KV, and fixed: `gdn_reserve` is the one meter.)

## Proof

- nemotron and qwen3.6, the benchmark's 1/2/4-user cells on this Mac: zero steps over 100 ms that
  remap anything (there are no remaps to time), no window-server freeze; TTFT/TPOT no worse than
  main; context capacity stated against main.
- The scheduler tests #525 added for the trades are replaced by ones that assert the step output
  carries no memory ops, and that a save past the pool evicts to the disk tier.
