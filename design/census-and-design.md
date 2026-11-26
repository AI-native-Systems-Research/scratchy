# Prefill CB pipelining (problem 2) — census and design

Front: every prompt chunk of a prefill in flight back to back, through session switches and
engine-step ends — proven like decode's #916 (step depth ≥ 2, card busy ≈ 99% through the whole
prompt), with no TTFT/TPOT regression in any serve cell including c4/c8.

Worktree: `.claude/worktrees/prefill-cb-pipelining`, branch `prefill-cb-p2`, based on
`prefill-host-tables` (ebbb4168a6, problem 1's tree — the future base). This front is measured
ALONE on its own branch, never bundled with problem 1 (PR #1414). No env toggles, no `#[allow]`,
no model/arch special cases, no fallbacks; a regression is attributed only from an A/B that
isolates one change.

All file:line citations below are against this worktree (prefill-host-tables + main lineage;
the step path is identical on `main` — the host-tables diff touches only what a bind writes,
never when a step waits).

---

## 1. THE CENSUS

### 1.1 The belief check, first: "current main already CB-pipelines the solo ~10k prompt at depth ~2" — REFUTED for prefill

The numbers behind that belief are real but they are **decode steps, not prompt chunks**. The
step-segs instrument on this tree (`crates/serving/worker/src/step_segs.rs`) records ONLY the
pipelined decode arm: `record_submit` has exactly one caller, the
`LogitsWanted::PerRequestDeferred` arm of `run_card_launch`
(`backends/spyre-hw/crates/worker/src/card/forward.rs:506-575`, record at `:537`), and the table
prints "N pipelined decode step(s)" (`step_segs.rs:221-225`).

The current A/B runs on pod nickm4-569cb766b-cn5xb (`/work/pht-runs/`, problem 1's A/B:
A = base `efd226276bc4`, B = prefill-host-tables `c879116e64b5`, granite-3.1-8b-instruct,
both sides main-lineage — neither contains any chunk pipelining):

- tp/4 solo ~10k chat (`ab-pht10k-20261126T182848Z`), every run, both sides:
  `[step-segs] 127 pipelined decode step(s) … depth mean 1.99, max 2, distribution 1:1 2:126
  … busy 126/127 = 99%` — the 127 decode steps AFTER the prompt. The ~10k prompt's ~21 chunks
  appear nowhere in the table.
- tp/1 solo ~10k (`ab-pht1-10k-20261126T195657Z`): same decode-only table, `depth mean 1.99,
  busy 126/127 = 99%`, fence mean 131.5 ms.

So decode is pipelined at depth 2 with the card busy 99% — and the prefill that precedes it is
**fully serial**, chunk by chunk, by construction (§1.4). The full ~10k prefill on the current
tree measures **5628.8 ms mean (A) / 5178.5 ms (B) on tp/4** and **11103.2 / 10428.0 ms on
tp/1** (5 pairs each, `full_prefill = 1st token − startup`, from `/work/pht-logs/pht.log`).
The pipelined floor the killed line reached was ~4.6–4.9 s (§1.7) — i.e. **~0.7–1.0 s of the
tp/4 prefill is host exposure between chunks**, which is what this front exists to delete.

### 1.2 What `f7fa378cb5` landed, and where it is

`f7fa378cb5` "perf(spyre): every chunk of a prompt is CB-pipelined on the card's one launch
stream" is the head of **PR #1403 (CLOSED 2026-11-25T20:51Z, never merged)**, branch
`prefill-cb-pipelining`, based on `6eeb0ba546` (main of 11-25 morning). `git branch --contains`
names only `origin/prefill-cb-pipelining`; nothing of it is on `main` — no squash-merge, no
re-land. The close reason, in the PR: *"Closed: nuked by the owner. #1399 and #1403 are being
redone properly — the step logic moves into senlib-rs and scratchy uses it. Nothing from this
branch is reused."*

What it had done (from the PR body and the pod's `/work/p2cb-scripts/plan-entry.md`): one
launch stream per card (`CardRuntime::launches`), one step door
`SuperDscSession::submit_step(n_new, seq_pos, SuffixLaunch)` with a `Landing`, `run_card_launch`
generic over `LaunchEnd` (`Picks | Forwards`), a prompt chunk `Picks` iff it finishes its prompt
(collected in-call — the one host wait a prompt meets), plus a build-time check that a forward
stopping before its suffix ends on a whole receive-window cut cycle. Its measured result:
21-chunk depth `mean 1.90 (1:2 2:19)`, busy `19/21 = 90%`, full ~10k prefill
5620/5688/5661 → 4587/4618/4924 ms (median −18.4%), decode 126/127 unchanged.

Also closed, not merged: **#1380** "a prompt's middle chunks are pipelined, as decode steps
are" (round 1; only middle chunks skipped the wait, drained at every session switch and
engine-step end, shipped on a −13% TTFT with no depth/busy proof — the failure mode this front
must kill). And the **"world step" (ws1)** successor line on the pod
(`/work/ws-scripts/`, `/work/p2cb-runs/ab-*-ws1-*`): 10k full prefill 4827.6 ms vs #1408's
5153.2 — also killed. None of these branches' code is reused; they are census evidence only.

### 1.3 The chunk ladder and its session kinds

A prompt runs as a loop of chunks, each on the narrowest rung that holds it
(`backends/spyre-hw/crates/worker/src/card/forward.rs:1421-1534`, `superdsc_forward_chunk`):

- **First chunk** (pos == 0): the smallest rung of the batched-prefill ladder `rungs`
  (`forward.rs:1459-1463`) — the loader's `PREFILL ladder = [7, 11, …, 512]` (pod log:
  "chunk runs on the smallest rung that holds it; ceiling 512 rows/chunk").
- **Continuation chunks** (pos > 0): `prefix.rung_for(left)` — the narrowest SOLO rung of the
  prefix ladder that holds the rows left (`forward.rs:1465-1471`;
  `backends/spyre-hw/crates/worker/src/card/types.rs:287-298`). The tail lands on a narrower
  rung by the same law.
- **`want_logits`**: `LastRow` iff this chunk finishes the prompt, else `None`
  (`forward.rs:1484-1508`).
- A `BatchOut::Deferred` from a chunk is a **refusal**: "a prompt chunk deferred its read —
  only a decode can" (`forward.rs:1514-1520`; the packed wave's twin at `:1615-1619`). This
  refusal is the type-level statement that prefill cannot pipeline on this tree.

Every rung owns its **own session**: `SuperDscSession` wraps `PerRank<Executor>`
(`backends/spyre-hw/crates/card/src/sdsc_runner.rs:176-188`), and each `Executor` owns its own
flex `Stream` (`backends/spyre-hw/crates/card/src/superdsc_exec.rs:1302,
stream: Option<Stream>`). The prefix ladder's rungs each carry `sess: SuperDscSession`
(`types.rs:230-240`). So the first chunk, the 512-row continuation chunks, the tail, the packed
waves (`types.rs:310-335`), and the decode seats are **different sessions on different
per-rank streams**. There is no device-side ordering across them; today ordering is bought
with the host drain of §1.4.

### 1.4 The submit path per chunk, and the wait points

One chunk launch is `run_card_launch` (`forward.rs:225-628`): stage the paged step's tables and
acts (page table, mask stack, seat laws — the ~185 ms/chunk of host work, §1.7), then:

- **Middle chunk** (`LogitsWanted::None`, `forward.rs:585-594`): `pf.run_bound(&binds, mq,
  seq_pos, StepLogits::Discarded)`.
- **Prompt-final chunk** (`LogitsWanted::LastRow`, `forward.rs:598-617`): `run_bound(…
  LeftOnDevice)` then `read_sampled_tokens` in-call (`forward.rs:1035-1056` — defer the idx
  stick read, then `finish_deferred_read_with` immediately).
- **Decode step** (`PerRequestDeferred`, `forward.rs:506-575`): `submit_bound` (bind all
  ranks, doorbell all ranks in one parallel section, **wait nothing** —
  `sdsc_runner.rs:918-936`), defer the idx read onto the read ring, return
  `BatchOut::Deferred`. The collect half (`collect_decode_tokens`, `forward.rs:1064-1116`)
  runs either in-call or from the drain — the loop's ONE fence
  (`finish_deferred_read_with`, `sdsc_runner.rs:962-982`: the lead's read + each peer's
  `settle_peer_step` shadow quiet point, `superdsc_exec.rs:5077-5084` — never a whole-stream
  sync).

`run_bound` (`sdsc_runner.rs:1078-1107`) is the eager door: `bind_all` → every rank
`predict_submit` in one parallel section → **every rank `predict_finish`**, and
`predict_finish` is a full-stream synchronize (`superdsc_exec.rs:5089-5100`:
`self.stream_ref()?.synchronize()`). **That synchronize is the per-chunk drain**: the host
returns from a chunk only after the card retired it, then stages the next chunk while the card
sits idle. Every chunk of every prompt pays it on both sides of the current A/B.

The shadow's own law, for any redesign: `quiet_shadow` (`superdsc_exec.rs:5200-5221`) — a host
write into a segment shadow may only happen once every in-flight H2D of that session has
executed (FIFO retires the prefix behind the last-enqueued one). On main this is subsumed by
the full drain; in a pipelined design it becomes the binding wait (§2, S7).

### 1.5 The engine-step path: how many steps, and the depth-2 machinery prefill does not use

- The scheduler caps a step at `max_num_batched_tokens` (default **8192**,
  `crates/serving/api/src/llm.rs:1588`; `max_num_scheduled_tokens` defaults to it,
  `crates/serving/scheduler/src/scheduler/core.rs:2918-2920`; `long_prefill_token_threshold`
  defaults 0 — no extra cap). A ~10k prompt is therefore **two engine steps**: 8192 tokens,
  then the remainder. (Step 1 = 16 × 512-row chunks; step 2 = the rest, including the tail.)
- The worker's `execute_model` (`backends/spyre-hw/crates/worker/src/spyre_exec.rs:182-810`)
  runs a step's prompts **synchronously**: new requests with n > 1 are gathered into the wave
  (`:247-339`); a wave of one (the solo case) or leftovers run the solo chunk loop
  (`:456-465` → `run_request_step`, `spyre_forward.rs:138-307`, whose own for-loop feeds whole
  steps to `superdsc_forward_chunk`, `spyre_forward.rs:180-185`); packed groups run
  `superdsc_forward_packed` (`forward.rs:1565-1647`) — all through the eager doors of §1.4.
  `execute_model` returns only after the card is drained.
- Decode's depth-2 is the pairing of: (a) the engine's pre-schedule-to-2 —
  `crates/serving/api/src/engine.rs:2386-2412` (step 3 of `step_loop_async`, `gpu_in_flight <
  2`) and the same law in `crates/serving/engine/src/core_client.rs:436-452`
  (`get_output_pipelined`) — with (b) the worker's deferral: a **pure** decode step (no new
  requests, every cached req a decoder — `spyre_exec.rs:707-715`) returns a deferred
  `ModelRunnerOutput` whose resolver knocks the drain door (`:723-735`); async scheduling is
  on by default (`crates/serving/api/src/init.rs:1488`, `:1793-1794`). A **mixed** step — any
  prefill chunk ran, or any per-request forward will — collects its decode in-call
  (`spyre_exec.rs:736-749`), draining the decode stream including the still-in-flight previous
  step.

### 1.6 Measured evidence (pod nickm4-569cb766b-cn5xb, read-only)

Current tree (problem 1's A/B, 2026-11-26, `/work/pht-runs/`, `/work/pht-logs/pht.log`):

| cell | A (base) | B (host tables) |
|---|---|---|
| tp/4 solo ~10k full prefill, mean / median ms | 5628.8 / 5622.5 | 5178.5 / 5158.6 |
| tp/4 mean ITL ms | 38.600 | 38.600 |
| tp/1 solo ~10k full prefill, mean ms | 11103.2 | 10428.0 |
| tp/1 mean ITL ms | 132.100 | 132.100 |
| tp/4 decode step-segs | depth 1.99, busy 126/127 = 99% | same |
| tp/1 decode step-segs | depth 1.99, busy 126/127 = 99% | same |

Serve sweep, tp/4 (`/work/pht-runs/absweep-pht-20261126T183417Z/pair.txt`, A side, rep2):
c4-123 TPOT 40.1 ms, c4-1k 38.1, c4-10k **51.0**, c8-1k 43.5 — the concurrent-prefill TPOT
spread this front must not regress (c4-10k decode rows currently wait peer prefill).

Killed line's diagnostics (`/work/p2cb-runs/diag-pipe2-p10k-r1-20261125T140432Z/` — the only
chunk-level instrumentation that has ever run on these cards, on the #1403 build):

```
[step-segs] 21 pipelined prompt chunk(s):
  bind   75.474 / 177.138 / 1585.0 ms   (the shadow writes, with the shadow's quiet point)
  submit 110.083 / 235.686 / 2311.7 ms  (submit_step: the doorbell + refill)
  total  185.564 / 338.231 / 3896.8 ms
  depth  mean 1.90, max 2, distribution 1:2 2:19
  busy   19/21 = 90%
```

plus plan-entry facts: a long chunk is more CBs than the card's queue holds beside another
(32768), so a deep-context chunk's submit is fed as the card retires the chunk ahead (submit
100–235 ms at depth, bind ~26–50 ms); continuation chunks wait the shadow quiet point instead
(bind 120–186 ms); either way the card stays owed work — card-bound. One A/B run measured
17/19 busy with bind max 274 ms (the 4924 ms outlier). Base vs piped full prefill:
5620/5688/5661 → 4587/4618/4924 ms (median −18.4%). ws1 (world step): 4827.6 vs 5153.2 (#1408).

**The card-bound floor for a tp/4 ~10k prefill is therefore ~4.6–4.9 s; main pays ~5.6 s; the
gap (~0.7–1.0 s, ~13–18%) is pure host exposure between chunks.** No chunk-depth
instrumentation exists on the current tree — the #916-style bar is unmeasurable as the tree
stands, which the proof plan (§4) fixes first.

---

## 2. THE STALL LIST

Every point where a chunk is not in flight back to back, cause named from the code:

**S1 — Every middle chunk is drained in-call.**
`run_card_launch`'s `LogitsWanted::None` arm (`forward.rs:585-594`) → `run_bound`
(`sdsc_runner.rs:1078-1107`) → per-rank `predict_finish` → full-stream `synchronize()`
(`superdsc_exec.rs:5089-5100`). The host stages chunk j+1 (mean 185.6 ms/chunk: bind 75.5 +
submit 110.1) only after chunk j fully retired; the card idles through every staging span.
Cost: the ~0.7–1.0 s tp/4 gap (§1.6); on tp/1 (~11.1 s prefill) proportionally more.

**S2 — The prompt-final chunk's pick is read in-call.**
`forward.rs:598-617` → `read_sampled_tokens` (`forward.rs:1035-1056`). One wait per prompt by
design (the first generated token is a host fact the reply needs) — but today it is preceded by
`run_bound`'s full drain of the same chunk, so the final chunk's forward and its pick read are
two serial waits where one would do.

**S3 — Session switches between chunk kinds.**
First chunk (batched-ladder rung session), 512-row continuations (prefix w=512 solo session),
tail (narrower prefix rung session) — each a different `SuperDscSession` with per-rank streams
of their own (`sdsc_runner.rs:176-188`, `types.rs:225-251`, `superdsc_exec.rs:1302`). There is
no cross-session device ordering; on main the S1 drain buys it implicitly. #1380's failure was
exactly here: it pipelined only within one session and drained at every switch. Any design
that keeps per-session streams re-creates #1380's failure mode at every rung boundary
(first→prefix, 512→tail, wave→solo).

**S4 — Engine-step ends.**
The 8192-token scheduler cap (§1.5) splits a ~10k prompt across two engine steps;
`execute_model` runs a step's chunks through the eager doors and returns only after the card
is drained, then the engine finalizes, schedules, and submits the next step
(`core_client.rs:436-469` / `engine.rs:2386-2412`). The card idles through the whole round
trip. Measured as the killed line's two depth-1 chunks (distribution `1:2 2:19`) and the
17/19-busy run: the boundary chunk is the one that drops.

**S5 — Mixed steps collect their decode in-call (the c4/c8 TPOT coupling).**
The `pure` gate (`spyre_exec.rs:707-715`) defers only a step that is nothing but decode; any
step that also ran a prefill chunk drains its just-submitted decode step in-call
(`spyre_exec.rs:736-749`), waiting the decode stream out including the previous in-flight
step. Under concurrent prefill, decode pipelining collapses at every mixed step — the
mechanism behind c4-10k TPOT 51.0 ms vs solo ITL 38.6 (§1.6; see also the memory record
"conc-4 TPOT collapse is prefill steps"). A prefill-pipelining change that leaves this gate
alone will move TTFT and TPOT in opposite directions — the exact #1380-shaped trade this
front is barred from shipping.

**S6 — The packed wave front is eager too.**
`superdsc_forward_packed` (`forward.rs:1565-1647`) runs `LastRow` launches through
`run_bound` + `read_sampled_tokens` in-call, group after group (`spyre_exec.rs:431-455`). In
c4/c8 the wave IS the prefill; its drains are S1 at concurrency.

**S7 — The shadow quiet point (the ceiling, not a main-tree stall).**
`quiet_shadow` (`superdsc_exec.rs:5200-5221`): every bind waits the session's last in-flight
H2D. On main it is subsumed by S1's full sync. In any pipelined design it becomes the binding
wait — and because a chunk's H2D sits in FIFO behind the previous chunk's compute, waiting it
is the pipeline's own cadence (card-bound, the killed line's bind 120–186 ms), not an
artifact. The design must state this law once and keep it.

---

## 3. THE DESIGN

### 3.0 The seam, per the owner's nuke ruling

PR #1403 was closed with a direction, not just a verdict: *"the step logic moves into senlib-rs
and scratchy uses it. Nothing from this branch is reused."* The design below therefore places
the ordering and submit/land primitives on the senlib-rs (flex-rs) side of the seam — flex-rs
already owns `Stream`/`StreamId` (`flex-rs/src/stream.rs:52`), `synchronize` (`:1497`), and
the event surface (`flex-rs/src/event.rs`) — and scratchy consumes them through its existing
doors. Nothing below re-introduces #1403's code; the laws are restated because they are the
mechanism's own, and their home changes.

### 3.1 One launch stream per card (kills S1's wait and S3)

**The law: a card's work is one ordered queue.** Device-side ordering of a card's launches is
a property of the CARD, not of the session that submits. Each card owns ONE launch stream,
minted at bring-up by the card runtime (a flex-rs fact); every session the loader builds —
every prefill rung, every prefix rung solo and packed, every decode seat — borrows that
stream for its steps. Submission order is then execution order across all of them, and no
cross-session handoff ever needs a host drain.

- **Typed shape:** the stream is not a field an `Executor` mints (`superdsc_exec.rs:1302`
  today) but a borrow of the card's, handed down at session construction — the same
  target-ABI-fact shape the architecture law already prescribes ("facts the shared passes
  take as input — never logic"). A session's type states it rides a card's stream; nothing
  below the session can ask which stream it is on.
- **No chunk-kind special case:** the law is stated over launches, not over chunks. The
  ladder's rung selection (`rung_for` / smallest-rung-holding-it) is untouched — a chunk still
  runs on the narrowest rung that holds it; what changes is only that rung A's chunk and rung
  B's chunk are device-ordered without a drain between them.

### 3.2 One step door with a landing (kills S1)

**The law: a step is submitted, not run.** The eager chunk doors (`run_bound` for
`Discarded`/`LeftOnDevice`) are replaced by the same submit/defer shape decode already uses —
as a senlib-rs step door (`submit_step(n_new, seq_pos, suffix)` arming a landing latch) that
scratchy spells once per session. A forward's last submission arms its landing; the step hands
back what it owes:

- **Typed shape — `ForwardLanding`**: `Nothing` (a middle chunk: no suffix ran, nothing read —
  a value, not the refusal at `forward.rs:1514-1520` today) or `Picks` (the launch finished a
  prompt: the idx-stick read is deferred onto the ring exactly like a decode's, and collected
  in-call through `collect_decode_tokens` — the ONE host wait a prompt meets, S2 kept by
  design and reduced to one wait instead of today's drain-plus-read).
- The chunk loop (`superdsc_forward_chunk`'s `while pos < end`) changes only in that it does
  not wait: chunk j+1 stages and submits behind chunk j. The staging wait it does take is the
  shadow's own quiet point (S7) — the card's cadence, stated once in the executor, unchanged.
- The CB-queue fact (32768 CBs; a 512-row chunk is more CBs than the queue holds beside
  another) means the host's enqueue of chunk j+1 is fed as the card retires chunk j — the
  submission blocks on queue space, not on completion. The card stays owed work throughout;
  the **busy row is the bar**, the depth row its twin (the step-segs doc's own law,
  `step_segs.rs:53-62`).

### 3.3 Steps defer; the engine's depth-2 machinery carries prefill (kills S4 and S5)

**The law: a step owes its outputs, not its completion.** `execute_model` returns when its
forwards are submitted; what it owes rides the existing deferred-output shape:

- The in-flight queue generalizes from `InFlightDecode` to one `InFlightStep` type — a
  submitted step's landing and deferred reads, popped by the drain **in submission order**
  (the law the #916 queue already states for decode, widened to steps, not restated beside
  it). The knock/drain-door/`DrainCell` machinery is reused as-is (`spyre_exec.rs:716-735`,
  `core_client.rs:186-252`).
- The `pure` gate (`spyre_exec.rs:707-715`) generalizes from "the step is nothing but decode"
  to "the driver can drain" (the drain door is present): a step whose prefill chunks are
  submitted-not-collected defers exactly like a pure decode step, because a mid-prompt chunk
  step owes no tokens at all, and a prompt-final step owes one pick through the resolver —
  the shape decode already proved. A mixed step (c4/c8: prefill chunks + a decode batch)
  defers the same way, and its in-call collect arm remains only for drivers with no drain
  door, composed through the same pop — one path, never two.
- With that, the engine's pre-schedule-to-2 (`core_client.rs:436-452`, `engine.rs:2386-2412`)
  queues engine step 2's chunks behind engine step 1's tail **on the same card stream,
  device-ordered** — the 8192-token boundary costs one schedule message, zero card idle. No
  scheduler change: the 8192 split stays exactly as the scheduler states it.

### 3.4 The wave front rides the same law (kills S6)

`superdsc_forward_packed`'s launches are launches: same stream, same landing, `Picks` iff a
tail finishes (already the semantics — only finishing requests' picks are kept,
`forward.rs:1633-1645`). Group after group (`spyre_exec.rs:431-455`) submits back to back. No
wave-specific code: the question "would this line exist for a non-wave launch?" answers itself.

### 3.5 Problem 1's step-path guarantee, stated explicitly

A single step's submission keeps its exact shape: **one bind pass, one parallel submit** —
`submit_bound`'s body (`sdsc_runner.rs:918-936`: `bind_all` on every card, then every card's
doorbell in one parallel section, waiting nothing) is unchanged, and the per-card fp16
host-table writes (problem 1's work, inside `StepBinds::tables` / `bind_all`) are untouched.
The pipelining changes only what happens AFTER a submission returns — no `predict_finish`
follows a chunk — and WHEN steps are submitted relative to each other. A step's submission
looks the same to the card whether it is the first in flight or the second.

### 3.6 The compile-time check that must ride along

On one stream, a forward whose pick nobody reads (a middle chunk) is followed by the next
forward with no collect between them; that forward's first collective must not land mid
receive-window cut cycle. #1403 carried this as a **build error** (a forward that stops before
its suffix ends on a whole receive-window cut cycle, or the program does not build — the
`WINDOW_CUTS` law in the bake's collective lowering), audited sound: with 2 all-reduces per
layer every dense model passes, and the 8b tp/4 build has 80. This design keeps it as a
build-time check in the bake — a fact the bake knows, refused at compile time, never probed at
runtime.

### 3.7 Instrumentation (measurement, not mechanism)

`step_segs` gains the prompt-chunk table the killed line had — per-kind tables (prompt chunks
beside decode steps), same rows (bind/submit/defer/total/depth/busy), same laws (an instrument
may print; it must not cost; gated by `--bench`, no env toggles). Without it the §4 bar is
unmeasurable. This is the first thing the front types, and it lands alone against base to
establish the BEFORE numbers.

---

## 4. THE PROOF PLAN

All cells on pod nickm4's cards (tp/4 = 6b 6f 7a 7b; tp/1 spot cells on 6b), granite-3.1-8b,
A/B interleaved, same build path, one change isolated: A = this branch's base
(prefill-host-tables alone), B = base + this front. The harness is the existing
`chat_ab.sh` / `serve_ab.sh` / sweep pair (`/work/pht-scripts/`), re-pointed; the tp/1 serve
sweep currently running supplies tp/1 baselines.

1. **Instrumentation-first leg (no mechanism yet):** chunk step-segs on A. Expected: depth 1
   everywhere, busy 0/21 — the BEFORE row that makes the bar falsifiable, and the refutation
   of §1.1 on the record.
2. **Solo ~10k chat, tp/4, ≥5 pairs:** full_prefill mean/median (base 5628.8/5622.5 ms);
   bar = **depth ≥ 2 at every chunk past the prompt's first, busy ≈ (chunks−1)/chunks through
   the whole prompt — including the first chunk's handoff, the 512-row mid-chunks, the tail,
   and the chunk that opens engine step 2 (the 8192 boundary — #1380's exact failure point,
   named by step index in the table)**. full_prefill must approach the ~4.6–4.9 s card-bound
   floor; ITL unchanged at 38.6 ms; replies in the same md5 families as A (multi-chunk greedy
   nondeterminism is a pre-existing main bug — #1403's audit reproduced it on unpatched main —
   so the bar is family match, with the nondeterminism issue tracked separately).
3. **Short-prompt chat, tp/4, 10 pairs:** md5 identical on both sides (the short prompt is
   one/two chunks — the guard that nothing changed where pipelining has nothing to do), ITL
   and TTFT level.
4. **Serve sweep, tp/4 — the 9 cells (c4/c8 × 123/1k/10k, 2 reps):** TPOT watched per cell —
   **no regression anywhere**, specifically c4-10k TPOT (51.0 ms base) and c8-10k; the mixed
   step's decode submits must stay busy through peer prefill (the chunk table's busy row now
   covers steps that carry both). TTFT ≤ base in every 10k cell; throughput level.
5. **tp/1 spot cells:** solo ~10k chat (base 11103.2 ms — the biggest host-exposure win) plus
   one serve cell; same bars.
6. **No-regression bar, per cell:** every TTFT/TPOT/ITL/throughput delta within the noise the
   harness's own pair repeats show on base-vs-base; full_prefill strictly better in the 10k
   cells; any cell that regresses is attributed only from an A/B that isolates this front
   alone (the discipline law — problem 1 is never in the same build as this front's
   measurement).

Gate order: leg 1 (instrumentation) → legs 2–3 (solo proof) → leg 4 (c4/c8 TPOT guard) → leg 5.
A failure at leg 4 is a stop-and-redesign, not a tune: the `pure`-gate generalization (§3.3)
is the piece that makes c4/c8 move with, not against, the solo win — if it does not, the
design is wrong, not the cells.
