# Regenerating the bridge 2 corpus

The goldens are the **reference's own** SentientIR for **our own** emitted DataflowIR. Both halves
come from a real bake, so regenerating needs the pod.

## 1. Emit the DataflowIR

Sync this tree to the pod and bake. The bake stages DataflowIR *before* invoking `dbo-opt`, so it
still produces the input even though the pod's `dbo-opt` is unpatched and refuses `--from-dfir`:

```bash
rsync -a --delete --exclude=target --exclude=.git -e <kubectl-rsh> ./ <pod>:/work/scratchy/
kubectl exec <pod> -- bash -lc 'cd /work/scratchy && \
  export PATH=/work/.cargo/bin:$PATH HOME=/work CARGO_HOME=/work/.cargo && \
  DBO_OPT=/project_src/deeptools/build/dbo/tools/dbo-opt/dbo-opt SCRATCHY_SKIP_SENDNN_CXX=1 \
  cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel -vv'
```

Groups land at `/tmp/superdsc-stage-<pid>/<kernel>/group_N/group.mlir` — 32 groups, 417 programs.

## 2. Ask the reference for its SentientIR

`mkgolden.py` beside this file extracts each named program module as a **bare, non-private**
`func.func @dataflowProgram()` and runs:

```
dcc_standalone <prog>.mlir -kEmitProgIR --mlir-disable-threading \
    --mlir-print-ir-after=dcc-dataflow-to-sentient
```

⛔ THE BARE SHAPE IS REQUIRED. The nested `module { module { … } }` shape is what `dbo-opt` consumes;
under `dcc-opt` `SymbolDCE` deletes a `private` body whose only caller lives in another symbol table,
and the pipeline then runs 25 passes over an empty module and reports success.

⛔ AND `SENARCH=mpw4` MUST BE SET, as the lit tests set it.

Point `K` at the new staging directory and run it on the pod. Last run: **417 goldens, 0 parse
failures, 0 missing dumps.**

## 3. Commit a subset

417 pairs is 6.3 MB. Committed here are the three smallest of each of the six program kinds — `mul`,
`matmul`, `add`, `rsqrt`, `mean`, `batchmatmul` — 18 pairs, 240 KB.

⚠️ When copying a selection, do not drive the loop with `while read` over a file lacking a trailing
newline: it silently drops the last entry, and that is how one golden went missing here. The pairing
test in `tests/the_bridge_matches_the_reference.rs` exists because of it.

## Provenance of the committed set

* C++ reference: pod `/project_src/deeptools` at `a0d29abbedfa2dd44ec7255e59440b06a429118c`
  (`stable-2026_07_24-142907-715-ga0d29abbed`).
* Emitter: this worktree, `granite-3.1-2b-instruct` + `quant/fp8-dynamic-per-channel`.
* One model at one preset — see the header of the test file for why that is not coverage.

## ⛔ REGENERATE WHEN TILING LANDS — the committed set is a pre-tiling snapshot

The emitter did not tile when these were captured. Across all 417 programs: **45** `affine.for`; in
the 18 committed: **9**, all inside the three `batchmatmul` programs. Every `mul`, `matmul`, `add`,
`rsqrt` and `mean` here has **no loop at all**.

So the corpus exercises transfers, computes, sends, receives and program units in volume, and the loop
machinery **not at all** — `TransformLoopToLegalizeForSentientLowering` (D7), `MutableAddrSplitting`
(D11, 1,358 lines), `LoopUnrollForShuffleOp` (D14), and the loop-bound and stride paths of
`AgenToSentient`.

When tiling lands the DataflowIR changes shape — nests appear, transfers become chunked and strided,
and `Exploit::FITS_LX` begins deciding whether a tiling loop exists — and every golden here becomes
wrong while still being *comparable*, which is the failure mode that matters. Re-run steps 1-3 and
replace the whole set; do not merge new pairs into the old ones.

## The end-to-end gate — D29 re-entry, measured

`reentry.py` beside this file is what established that bridge 2 can be verified end to end rather
than only against a text dump.

For each pair it strips the golden's dump banner to leave a clean module, then runs **both** the
original DataflowIR and its post-D28 SentientIR through:

```
dcc_standalone <in>.mlir "-kEmitProgIR=progir-format=senprog dump-progir=true"
```

and diffs the senprog. Result: **144 byte-identical, 0 differences, 0 re-entry failures.**

⭐ SO THE D1-D28 CONVERSIONS IDLE OVER AN ALREADY-LOWERED MODULE. There is no `-disable` flag for
any of them — `dbo/docs/pass_pipeline.md` says *"the ones without a flag are the ones nothing below
them survives"* — but they are pattern-driven, so a module holding only `sentient.*` gives them
nothing to match. That was plausible and untested; it is now measured.

⛔ THE OUTPUT WAS CHECKED FOR CONTENT, NOT JUST EQUALITY. Two empty outputs are also identical. The
smallest program emits 1,998 bytes of real instructions — `L3_LDMU`, `LX_LDSTI`, `L3_RETURN` — plus a
`reg_initial.txt` block per unit.

⛔ IT COVERS 144 OF 417, NOT ALL. The other 273 fail *before* bridge 2 is involved, in D29-D76, all
with one error: `Register initialization out of boundary` on `lxsu0:LRF0` (165) or `l3lu:LBR2` (108),
with byte addresses like `0x4020`, `0x10020`, `0x40020`, `0x64f000` written into registers whose
fields cannot encode them. That is the emitter's address arithmetic, not this bridge's — but it means
the end-to-end gate is available for a third of the corpus until it is fixed. The SentientIR text
diff still covers all 417.
