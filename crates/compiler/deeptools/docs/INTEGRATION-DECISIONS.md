# Integration decisions: where two campaigns ported the same thing differently

The four campaigns each forked from the same mid-bridge-2 commit (`77c193921`) and all four write to
the shared islands, because a bridge that emits an op needs that op declared. crustify's unit names
are also repo-wide rather than campaign-scoped, so the same reference function could be scheduled
twice. The result is rival ports of the same declaration, and **integration is the first time they
are held against each other.**

Every entry below is decided by reading the C++ authority (`/Users/nickm/git/deeptools-src` @
`a0d29abbed`), never by preferring a branch.

## The method

- **A bridge directory belongs to its campaign.** `bridges/dataflow_ir_to_sentient/**` keeps bridge
  2's version outright: bridge 1/3/4 touched those files only to keep the crate compiling while
  porting a different bridge. A lost compile fix comes back as a build error against the
  authoritative version, which is where it should be repaired.
- **`islands/**` is decided per declaration, against the reference.** This is where the real
  disagreements are.
- **Per-branch, in one 3-way pass, not per-commit.** Replaying bridge 1's 22 commits made the same
  island collision resolvable 22 times over (~370 decisions across the four branches). Every
  intermediate state is superseded by the branch's final one, so the only decisions that matter are
  between the two final trees. The campaign branches are untouched and remain the per-unit audit
  trail.

## Decided

### `islands/dataflow_ir/print.rs` — `elem()` → bridge 2
Bridge 2 refactored the `ElemType` match onto `ElemType::spelling()`; bridge 1 kept it inline. Bridge
2's is a strict superset — it also handles `MxInt`, which bridge 1's match omits.

### `islands/dataflow_ir/dialects/agen.rs` — bridge 2's, plus bridge 1's `Access`
Bridge 2's `Op` is a strict superset in variants (bridge 1 adds none it lacks), uses newtypes, and is
the model its 105 reviewed `AgenToSentient` units read. Taken as the base. Then:

- **`Access` added from bridge 1** — a real gap, not a preference. Bridge 2's `VectorLoad`/
  `VectorStore` carry no `load_set`/`store_set` field and derive it with `access_set(view_ty, lanes)`.
  A transfer's set is `constructLoadOrStoreSet`'s — chunked by `unitTimeTransferChunkSize_`, strided,
  repeated (`SNTransferLowering.cpp:135-222`) — which no view type implies. Bridge 2's consumers never
  hit this because they read a set that is already text; bridge 1 is the producer that must state it.
  So `Access::{OfView, Stated}` and an `access` field on both variants, with `emit` printing
  `access.set(..)`.
- **`MaskCounts` (bridge 1) dropped for `MaskPattern` (bridge 2).** Same thing modelled twice; bridge
  1 used raw `i32`, bridge 2 the `Elements` newtype. The crate's own law decides it ("NEWTYPES, never
  raw scalars"), and bridge 2 also cites the divisibility invariant (`Agen.td:1085-1086`).
- **`Yielded` / `Yield { values }` (bridge 1) dropped.** Both emit identical MLIR: bridge 1 puts the
  stored value on the terminator, bridge 2 on the op (`CompositeStoreSource::Region { stored }`), and
  both print `agen.yield %v : ty` for the store side and bare `agen.yield` otherwise. Design, not
  correctness — so the model 105 reviewed consumer units read wins. Bridge 1's producer adapts.
- `CompositeLoad`/`CompositeStore` structs (bridge 1) are bridge 2's `CompositeAccess`/
  `CompositeStoreAccess`; bridge 2's kept.

### `islands/dataflow_ir/dialects/vectorchain.rs` — BOTH campaigns had a defect

- ⛔ **`ConstantBitstreamOp` printing — BRIDGE 2 IS WRONG.** `ConstantBitstreamOp::print` writes the
  hand-rolled `{value = [0x..]}` **only** when `is_symbol` is absent or false; when it is true the op
  prints its generic attribute dictionary instead — decimal `N : i64`, `is_symbol` first because a
  `DictionaryAttr` sorts by name (`VectorChain.cpp:85-101`). Bridge 2 emitted
  `{is_symbol, value = [0x..]}`, which is **neither form the reference can produce**. Bridge 1's
  two-branch version taken.
- ⛔ **`vectorchain.shuffle`'s operand types — BRIDGE 2 IS WRONG.** Bridge 2 prints the trailing type
  list as `", index".repeat(variable.len() + pad.len())`, on the reading that every variable and pad
  is an `index`. The op declares
  `Variadic<AnyTypeOf<[Index, AnyInteger, AnyFloat]>>:$variable` and the same for `$pad`
  (`VectorChain.td:459-460`), and the printer emits each operand's own type
  (`VectorChain.cpp:387-395`). So bridge 1's per-scalar `ShuffleVariable { val, ty }` is necessary.
- ⛔ **`$mask` — BRIDGE 1 IS WRONG (by omission).** `Optional<AnyVectorOfAnyRank>:$mask`
  (`VectorChain.td:461`) is printed as `mask(%m)` plus its type (`VectorChain.cpp:381-382, 393-394`);
  bridge 1's `Shuffle` has no mask field, so it cannot print a masked shuffle. Bridge 2's
  `mask: Option<Predicate>` kept.
- **`$dbgName` — bridge 1 kept.** `OptionalAttr<StrAttr>:$dbgName` (`VectorChain.td:462`); bridge 2's
  `Shuffle` has no field for it.
- **`Predicate::of_operand` — bridge 1's addition kept.** Needed where a `SELECT` hands
  `ElementWiseSelectionOp` an input operand rather than a mask this lowering minted
  (`SNComputeLowering.cpp:1168-1171`); `$cond` is `AnyVectorOfAnyRank` (`VectorChain.td:517`).

So `Op::Shuffle` is the union: bridge 1's `variable`/`pad`/`dbg_name`, bridge 2's `mask`.

## Still open

`islands/dataflow_ir/dialects/mod.rs` (21 hunks) and `bridges/subtile_to_dataflow_ir/tape.rs`, then
the same pass for bridge 3 (`islands/sentient/**`, `islands/progir/**`) and bridge 4.
