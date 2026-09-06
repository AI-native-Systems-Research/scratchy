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
