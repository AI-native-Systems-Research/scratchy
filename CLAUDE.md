# Spyre tensor parallelism — the rules for this worktree

The job: **CB-pipelined tensor parallelism (TP) for spyre.** Every change on
this branch answers to the rules below. There is no Spyre card on this
machine, so everything is checked by compiling: Rust builds and tests, plus
dxp bakes through
`/Users/moosevan/tmp/dt_src/build/dxp/dxp_standalone`.

## The TP rules

1. **CB-pipelined.** A collective is control blocks in the *same* CB
   stream as the compute CBs that produce and consume its operands:
   producer compute CB → p2p DMA CBs → write-done wait → reduce compute CB.
   The host never sits between a producer, its collective and its consumer:
   no D2H, no host sum, no host barrier, no per-collective submit
   round-trip.
2. **AoT as much as possible.** These are all compile-time constants,
   emitted by `#[forward]`:
   - each tensor's shard layout,
   - where every collective sits on the tape,
   - the inbox buffers and their per-source slots,
   - the reduction order,
   - each DMA CB's segment-relative source and destination,
   - the CB sequence itself.

   The one thing bound at load is which physical peer card sits at each
   rank, i.e. the p2p window base. That is a load-time constant, never a
   per-step computation.
3. **dxp_standalone and runtime errors are Rust type invariants.** Every
   refusal `dxp_standalone` can raise for a TP shape, and every runtime
   error the TP path can hit, is unrepresentable in the types or fails the
   build. That covers:
   - shard divisibility,
   - stick and segment alignment,
   - LX capacity,
   - message size cap (`MAX_HDMA_MESSAGE_BYTES`),
   - peer rank out of bounds,
   - sending to self,
   - an unattached peer,
   - a window that is too small,
   - mismatched world sizes across the bake and the load.

   Use typed constructors, const generics and `const { assert!(..) }`.
   A runtime check stands in only where the type system genuinely cannot
   express the fact, and then it comes with a test that triggers it.
4. **P2P RDMA, never host RDMA.** Cross-card bytes move card-to-card over
   the cross-port IOMMU mapping (`P2pProtocol::P2pRdma`,
   `senlib_ffi::p2p_create_mapping`). Never stage through host memory
   (`HostDma` / HDMA), and never fall back to it: a topology without p2p is
   a refusal, not a slower path.
5. **Collectives are woven into the tape by analysis, never by hacking.**
   Where an AllReduce or AllGather goes falls out of sharding propagation
   over the dataflow graph:
   - a contraction over a split axis yields a partial sum;
   - a partial sum that meets a consumer needing a whole value gets an
     AllReduce;
   - a split value that meets a consumer needing the full axis gets an
     AllGather.

   Shard layouts are derived from the graph too. Banned: matching on weight
   names (`o_proj`/`down_proj`), per-arch shard tables, and hand-placed
   collectives.

## The design record: issue #1007

[#1007](https://github.com/AI-native-Systems-Research/scratchy/issues/1007) holds
the lessons from the earlier attempt. Read it before touching emission or the
runtime. Its §3 (on-card protocol), §4 (hardware facts) and §5 (proof standard)
bind this work:

- **Recv window:** one per (card, peer), in seg1, sized at bake; cap 4 MiB
  (the RDMA length field).
- **Sends:**
  - F-send singlecast only;
  - one sid per (sending rank, landing slot): an F-send lands only where its
    sid's `XSEG` entry points (#1009, E8a), so every slot of the window has
    its own sid and entry, programmed once at link. A fixed budget per rank
    (`LandingSlot::SLOTS`), never one sid per collective;
  - strict per-collective ring order: send → wait → consume;
  - wdone is a counter.
- **The send CB needs three facts:** xlat completeness, the compensated xseg
  word, and the receiver-programmed XSEG register.
- **Engine compensation:** the engine writes one flit (128 B) below the named
  destination.
- **Consumes:**
  - `TpConsume` is a singleton launch group;
  - tp consumers launch through `tp_recv_where` (`byte_off[SEG_WEIGHT] = 0`,
    bank 0);
  - `TpIdentity { rank, world }` is one fact.
- **Gates:**
  - tp=1 byte-identity is the hard gate;
  - a static audit of every collective operand's address must pass before any
    card run.

Its §1–2 *sharding mechanism* (a hand-declared bounds list, and name-matching
inserts in `tp_lowering.rs`) is superseded here by rule 5. Collectives come
from `scratchy_subtile::tensor_parallel::shard`.

## Architecture invariants (unchanged, still binding)

- `dsl_math + weights → weight_loader_fn() + forward_fn() + instruction_tape`.
  `forward_fn()` plays the tape; everything is a compile-time constant.
- **Everything common is shared.** The sharding analysis and collective
  insertion are target-neutral passes; at world size 1 they are a strict
  no-op. Target-ABI facts are const tables the shared passes take as input.
- **Per-target surface = opcode lowering only.** A collective is a new
  opcode: one registry row (`crates/compiler/subtile/src/ops.rs`) plus one
  lowering arm per target that runs it.
- **One bake serves every rank (SPMD).** Rank-dependence lives only in DMA CB
  destination addresses, bound at load; never in a baked program.
- **Every rank computes bit-identical replicated values.** Reduction order is
  canonical, by absolute rank, on every rank.
- No new env-var behavior toggles; no `#[allow]`; no weakening tests
  ([`docs/TESTING.md`](docs/TESTING.md)).
- Spyre-specific code is limited to: lowering to SuperDSC, launch, and
  allocs + d2h + h2d (here: the p2p DMA CBs).

## Build and verify

- Scope every build to one model: `-Fmodel/<stem>` (e.g.
  `-Fmodel/granite-3.1-8b-instruct` or `-Fmodel/smollm2-135m`). Never
  `model/all` to test one thing.
- Commits and PR titles use Conventional Commits scoped to the crate or area
  (`feat(spyre): ...`).
- See [`docs/BUILD.md`](docs/BUILD.md) and [`README.md`](README.md) for the
  general build.
