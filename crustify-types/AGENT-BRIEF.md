# crustify-types — port the TYPES that carry facts between the ported functions

You are porting **C++ class definitions and their fields**, not algorithms. Read this whole file
before you edit anything.

## 1. The authority

`/Users/nickm/git/deeptools-src/<file>:<line>`, revision `a0d29abbed`. Cite `path:line` in every
doc comment you write.

⛔ `/Users/nickm/git/deeptools` is a DIFFERENT revision. Do not use it.
⛔ Your own tree's Rust doc comments are **not** evidence. Ten defects in this stack were found by
reading IBM's source instead of our comments, and every one of the ten was self-documented as
deliberate. Two cited line numbers were off by 410 and 48.

## 2. Why this campaign exists — read this or you will port the wrong thing

Every prior campaign scoped FUNCTIONS, and each one excluded field accessors on the stated grounds
that they are *"a struct field or a method on an existing Rust type, not a function to port"* — that
exact sentence is in `crustify-capacity/scope-config.json`. The exclusion was right about what it
said and wrong about what followed: **the functions were ported one at a time and the types that
carry facts between them never were.** Nobody owned them, because a per-function scope has no unit
for a field.

The measured cost, in `crates/compiler/deeptools/src/schedule/stages/*.rs`: **160 live stubs, of
which 101 are field-read (37), field-MISSING (49) or borrow (15)** — not one an algorithm. 129 of the
130 present today are a bare `todo!()` with no citation, because there is no C++ line to cite: they
are methods of a trait with no counterpart in the reference.

**The metric, measured 2026-09-16 — `DesignSpaceConfig` has 49 fields in C++
(`dsc/designSpaceConfig.h`):**

| where it lives in Rust | count |
|---|---|
| on the Rust `DesignSpaceConfig` (`schedule/l3/dsc.rs:515`) | **9** |
| somewhere else in the crate | 19 |
| nowhere in the crate | 21 |

The 19 "elsewhere" are the workarounds. `scheduleTree_` is one of them — that is the "tree has two
homes" defect by name. `name_` and `computeOp_` are two more, and `stages/ddc_state.rs`'s own header
lists them as *"facts `l3::dsc` does not carry"*, handed in as construction arguments.

⚠️ Of the 21 absent, **some are a deliberate documented narrowing** — `StageDims::compound` records
`zi_/zj_/si_/sj_/r_/c_` as unspellable as `PrimaryDim`. **Open each one's citation before pricing
it.** The load-bearing absences are `loopOrder_`, `auxLoopOrder_`, `loopProperties_`, `pdsRelation_`,
`unpadN_`, `dscN_`.

## 3. ⛔⛔ THE ACCEPTANCE TEST IS THAT THE CARRIER LAYER GOES AWAY

Not that its stubs get bodies. `stages/ddc_state.rs`'s header states the defect against itself:

> `currDsc` in the reference IS `sdsc.dscs_.at(idx)`; here it is a CLONE taken at construction.
> `select_and_parse_ddl_template` writes through `sdsc.dscs_mut().at_mut(idx)` (`ddc/v1.rs:6489`)
> and those writes do **NOT** reach this clone. … **no caller can close it.**

The cause is in the ported signature: `run_v1(sdsc: &mut SuperDsc, sites: &mut P, ..)`
(`ddc/v1.rs:6532`) takes **two independent `&mut`** where the reference has **one object**, so the
borrow checker forces the clone and the write is silently dropped — and it compiles. Same class as
review 382's `L3RunInputs` cut.

So: when `SuperDsc`, `DesignSpaceConfig` and the `ScheduleNode` hierarchy carry their own fields, the
algorithms take `&mut SuperDsc` as the reference does, and
`stages/{ddc_store,ddc_store2,ddc_state,ddc_sites,ddc_tree,ddc_reads,offsets,env}.rs` are **deleted,
not completed**. A further 54 stubs are already unreachable behind four getters in `reads.rs` that
`return None`; deleting the layer retires those too.

## ⛔⛔ BUT THAT DELETION IS **NOT YOUR UNIT**, AND NOT IN THIS CAMPAIGN

The sentence above says where the architecture is going. It is **not** a licence to delete carrier
code, and `run_v1`'s signature is not yours to change either: `Ddc::run_v1` lives in
`ddc/ddcv1.cpp`, outside this campaign's oracle target, and the de-severing needs every type landed
first. It gets its own campaign.

**What that means for you concretely.** Changing a type's shape breaks the carriers that read its old
fields — the first wave hit exactly this, `error[E0560]: struct ConditionNode has no field named
then_region` in `stages/ddc_tree.rs` and `stages/ddc_store2.rs`. You must leave the crate compiling,
and the ONLY acceptable way is to **adapt the call site to the new shape, carrying every effect
across**. Never by deleting the logic, never by making a live call `todo!()`, never by leaving a
branch out because the new shape made it awkward.

⭐ **THE STANDARD TO MATCH, from this campaign's own first wave** (`acdf5a01b`, `e012_ConditionNode`):
the old `then_region: Vec<SchedNode>` forced a runtime refusal — `let SchedNode::Block(block) = entry
else { return None }`. The port replaced it with `CondRegions`, which encodes the reference's own
*"ConditionNode only accepts 2 BlockNodes as children"* (`dsc/dsc2.cpp:2143`) in the type, so the
refusal became UNREPRESENTABLE rather than deleted, and the missing-region case cites the
nullptr-returning getters at `dsc/dsc2.h:707` and `:713`. Every caller was rewritten to the new shape
with its loop intact. That is what "adapt, don't delete" means.

⛔ Your report must state, for every carrier file you touched, what you changed and why the effect is
preserved. A shrinking line count in `stages/*.rs` is not progress in this campaign; it is the thing
a reviewer will come looking for.

⭐ **A filled carrier method is a port of the wrong thing.** If your unit's only effect is to give a
trait method a body, stop and say so in your report.

## 4. Do NOT port these — measured dead or already done

- `getBlockTransferSizePerDim` (128L) — its own doc records it as unreached on scratchy's path.
- `getCoordinateCategoryOfPos` (`dsc/dsc2.h:153`) — reached only from the deferred `ELEM_ARR_COORD`
  arm; it lands with that arm, not before.
- `DsTrackInMem` (`util/memtracker/mem_track.h:24`, 703L) — **ALREADY PORTED**, in-tree at
  `schedule/memtrack/{mod,tracker,bundle}.rs`. `ddc_state.rs`'s 16 stubs are that ported tracker not
  being reachable through the trait: a carrier artefact, not missing C++.
- `PaddingFormType::getPadding` — already ported as `PaddingForm::padding`
  (`schedule/ddc/transformation_util.rs:192`).

## 5. ⛔ FOUR NAMES PRIOR STUBS CITE DO NOT EXIST — do not hunt them

`allocNode->paddingSizes_` (it is `DataStructDims::paddingSizes_`, `dsc/dims.h:219`), `isExternalDs`,
`externalDataStreams_` (zero hits tree-wide), and `getTransferType` never touches `constantInfo_`.

⭐ **15 of ~55 stub estimates were wrong, and 25 claim `tree::Kind` has no `Compute` arm when it has
had one since `c1f5c63fa`** — 21 of those collapse from "port a body" to "read a field" the moment
you open the citation. One file's header denies the arm its own function matches on. **A stale doc is
worth hours of misclassified work. Open the citation first, every time.**

## 6. Rules that are not negotiable

- **Pure-logic port.** No bindgen, no `-sys`, no `extern "C"`, no `unsafe`, no wrapped-layout
  `Foo`/`FooRef`/`FooMut`, whatever the generic C-porting conventions say.
- **Never a runtime refusal** — `crates/compiler/deeptools/CLAUDE.md` is the pre-eminent rule of the
  crate: `Err(`, `.ok_or` and `Result<` are frozen at ZERO in the DataflowIR bridge. `panic!`/`todo!`
  are capped and ratcheted DOWN only.
- ⛔ **NEVER `#[should_panic(expected = "<stub>")]` on a `todo!`.** It makes the gate green while
  porting nothing, and what it asserts is that the port is unported. The tell is having to rewrite
  the expected string whenever the frontier moves. A red gate on a `todo!` is information — leave it
  red.
- ⛔ **No tombstone comments.** Do not narrate what a stub used to claim, which earlier note was
  wrong, or why an attribute was removed. Cite the authority line and port the body.
- ⛔ **Edit/Write only — never python or sed to modify a file.** A splice once cut 36 of 475 lines and
  reported success.
- **Newtypes, never raw scalars.** Transposing two extents must be `E0308`.
- **Your gate is `cargo check -p deeptools` + `cargo test -p deeptools`.** Never the workspace and
  never the acceptance build from an agent worktree — 6 GB of `target/` each.
- ⛔ **Never commit to the branch the driver is writing.** That stranded 90 units on one campaign and
  67 on another.

## 7. What "done" means, per unit

Report **per unit, never per campaign** — a campaign cannot audit its own boundary, and "18/18" once
meant 238 stubs. A unit is done only with all three:

1. zero `todo!` in its body for the arms this campaign scopes,
2. a **real non-test caller**, named in your report,
3. a field count: how many of the C++ type's declared fields your Rust type now carries, and which
   ones it still does not, each with its `path:line`.

Point (3) is this campaign's burndown. It is the number whose absence let 101 stubs accumulate.
