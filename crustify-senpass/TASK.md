# Campaign: the D29–D75 in-place SentientIR passes

`dcc/src/Transform/Sentient` → `crates/compiler/deeptools/src/transform/sentient/`

The largest single stage of the deeptools compiler and the last unported one on the
SuperDSC → `init_binary` path. Bridges 1–4 are COMPLETE; this is bridge 5 in all but name, except
that it is not a bridge: **these passes rewrite SentientIR in place.** Input and output are both
`src/islands/sentient/`, so the campaign EXTENDS that island rather than emitting into a new one.

## Authority

    /Users/nickm/git/deeptools-src        repo_info.txt: deeptools|master|a0d29abbedfa2dd44ec7255e59440b06a429118c

⛔ `/Users/nickm/git/deeptools` is a DIFFERENT revision. ⛔ The pod is not reachable from this host.

## Why it is not optional

Bridge 2's ported SentientIR emission assigns **no registers** — `index: None` in 39 of 41 sites —
and that is *faithful*: the reference's own SentientIR at that point carries `regIndex = -1 : i32`
in all 54 occurrences of the committed golden corpus (`tests/sentient_corpus/`). These passes are
what turn `-1` into a real register file and index. Without them ProgIR gets `-1` where an
instruction needs a register and the backend refuses with `Register initialization out of boundary`
— already observed on `lxsu0:LRF0` and `l3lu:LBR2`.

The register chain specifically: `RegisterTypeAssignment`, `AddressRegisterPrecisionAssignment`,
`RegisterInitialization`, `VectorRegisterInitialization`, `SmartRegisterAllocation`,
`ReadOnlyRegisterRenumbering`, `RegisterPacking`, `PortAssignment`, `AddressPinningAndToggle`.

## There is no subset

Measured on the shipped driver (`dcc/tools/dcc-standalone/dcc-standalone-main.cpp`) with a
brace/conditional tracker: between `createAgenToSentientPass` at `:271` and
`createSentientToProgIRPass` at `:755`, **every one of the 48 passes is added unconditionally.**
None is behind a flag or an option. Do not look for a "just the required ones" subset.

## Scope, as enumerated

| | |
|---|---|
| files scanned | 63 (`*.cpp` 32,766 lines + `*.hpp`/`*.h` 1,672) |
| definitions found | 1,029 |
| **units in scope** | **656** (23,059 body lines) |
| excluded, with a reason each | 373 — see `EXCLUSIONS.tsv` |
| pass modules | 52 |
| Rust home files | 95 (real nested submodules) |
| dependency levels | 0..12 over the SCC condensation |

Enumeration was **by brace matching** from each signature's opening paren, comment- and
string-aware — never by a signature regex, which undercounted bridge 2 by 72%. Headers were scanned
too: on bridge 4 `progir.h` held 18 of 33 units and a `.cpp`-only scan would have missed every one.
Levels are computed over the **SCC condensation** (Tarjan): bridge 3's plain longest-path fixpoint
printed "level 266" on a mutually recursive pair. 8 non-trivial SCCs, largest 17 members.

Exclusions: 263 one-to-three-line field accessors, 97 MLIR pass factories, 12 LLVM RTTI `classof`
hooks, 1 trivial ctor/dtor. **Nothing is excluded for being hard.**

## The consolidated source

`cpp/sentient.cpp` — 29,444 lines, one self-contained TU, bodies **verbatim**, each preceded by a
banner giving its unit symbol, level, SCC, size and original `file:line`, in dependency order, plus
`cpp/prelude.inc` (a plain-C++ name inventory: no MLIR/LLVM/dcc header anywhere, out of scope,
nothing in it to be ported).

**Extraction was verified independently.** `tools/verify_extract.py` does not import the extractor's
scanner; it re-derives every body's end with its own character state machine, in both the extract and
the authority file, from the banner's citation alone — never re-slicing with the extractor's own
`(file, line, length)`, which is exactly how bridge 2's check passed over 366 truncated bodies. It
compares lengths then bytes, asserts each body's last line closes its function and that the brace
depth returns to zero exactly at the final character, and **never appends `}`**.

    656/656 bodies byte-identical, 962,619 bytes compared
    negative control (a) last line of a body removed  -> DETECTED
    negative control (b) an inner `}` blanked         -> DETECTED

## What is NOT in scope, and what to do about it

**43 of the 52 passes consume an analysis defined outside this scope; 122 of the 656 units name one.**
Measured from the passes' own `#include` lines and the analysis class names their bodies mention —
per pass in `OUTSIDE-DEPS.tsv`, per unit in `OUTSIDE-UNITS.tsv`. The two out-of-scope
subdirectories total ~11,700 further lines:

* `Analyses/` (39 files) — `Liveness`, `PropagationAnalysis`, `GraphColoring`,
  `ExpressionEvaluatorUtils`, `InstructionEstimation`, `CorrelationAnalysis`, `TimeStamps`,
  `RegisterPressureAnalysis`, `AddressPinningScheme`, `XRFRegisterAnalyzer`, `LoopGraph`,
  `BurstUtils`, `LiveRange`, `RedundantDefinitionEliminationTree`,
  `CFGSSentientLevelConditionalTree`, `CFGDeepMergingConditionalTree`, `UniformGroupAnalysis`
* `RegisterInitialization/` (12 files) — `Candidate`, `Collector`, `Evaluator`, `Selector`,
  `Transformer`, `UniformGrouper`: the whole register-init candidate pipeline

⭐ `todo!` naming the missing analysis is the correct action and is allowed here. ⛔ Do not invent
the analysis, do not inline a guess at what it would have returned, and do not substitute a constant
for its result. Extending `src/islands/sentient/` is a different case and IS expected.

## What "ported" means

**The whole function including its effect.** For an in-place pass the effect IS the port: which ops
are rewritten, which attributes are set to what, in what order. A hand attempt on bridge 2 extracted
each function's decision rule into a documented predicate, omitted the part that changed the IR, and
reported it done — nothing called any of it. **A predicate is not a port.** Droppable: only the
mechanism for reaching operands. If the island cannot express a result, extend the island — deciding
a function is unnecessary is not the porter's call.

**Independent review is mandatory**: every unit reviewed by a different agent instance than ported it.

## Schedule

| sub-campaign | levels | units | waves | port batches |
|---|---|---|---|---|
| sc1 level0-leaves | 0 | 256 | 1 | 41 |
| sc2 level1-helpers | 1 | 142 | 1 | 20 |
| sc3 levels2-3-pass-bodies | 2,3 | 148 | 2 | 22 |
| sc4 levels4-6-drivers | 4,5,6 | 90 | 3 | 12 |
| sc5 levels7-plus-roots | 7..12 | 20 | 6 | 7 |

`max_syms` 8 port / 24 review. Batches pack whole home files so a file is usually owned by one
batch. All fifteen schedules pass `crustify translate --dry-run`, and `crustify crates validate`
passes on the placement oracle.

## Gate

`cargo check -p deeptools` + `cargo test -p deeptools`, **once per batch**. ⛔ Agents must never run
the workspace or acceptance build: ~6 GB of `target/` per agent worktree.
