# Spyre: known issues

## dxp refuses a group at `Liveness.cpp:820`, nondeterministically

**Symptom.** A spyre bake fails with a `dxp refused <stage>/group_N` panic whose message ends
`DtException: operand_to_liverange_.count(val1) and operand_to_liverange_.count(val2), file
.../dcc/src/Transform/Sentient/Analyses/Liveness.cpp line 820`.

**It is a vendor nondeterminism, not a defect in the emitted bundle.** The same group directory,
byte-for-byte, compiles cleanly on most attempts and refuses on a few. Measured on one gemma-4-12b
fp8 group (`dxp_standalone --bundle -d <group> -b sentient`, `DT_PARALLEL_THREADS=1`, the bake's own
environment):

| how the group was compiled | refused |
|---|---|
| one at a time, nothing else running | 2 / 60 |
| 8 concurrent on 20 cores | 1 / 48 |
| 16 concurrent on 20 cores | 8 / 96 |
| 20 concurrent on 20 cores (the bake's own width on a 20-CPU pod) | 13 / 120 |
| 24 concurrent on 20 cores | 9 / 144 |
| 32 concurrent on 20 cores | 11 / 192 |

So it is roughly 2–10% per compile of an affected group, with contention raising the rate. It happens with
one `dxp_standalone` at a time on an idle pod (each uses at most two threads), so it is not a race in the
bake, and there is no concurrency below which it stops — which rules out capping the bake's dxp width as a
fix.

**A refusal is loud and is never a miscompile.** Every compile that passed — under every width above
and solo — produced a `spyreCodeDir/` (`init_binary.bin`, `spyrecode.json`) byte-identical to every
other passing compile of the same group. So the dxp cache (`SCRATCHY_SUPERDSC_CACHE`) cannot persist a
bad program from this: a refused group writes nothing, and a passing one writes the one program.

**What to do: re-run the bake.** The cache keeps every group that already compiled, so a re-run only
recompiles the group that refused, and almost always gets through. Do not add a retry loop to the bake
or an environment toggle to dodge it — the refusal is the only signal this exists.

**Where it fails.** In dxp's register-initialization coalescing, a liveness query asks for the live
ranges of two values that the liveness map does not hold. The map is keyed by the values' identity (an
address), which is consistent with the coverage depending on heap layout — matching "same input, a
few percent, worse under load". It is reported to the vendor; until it is fixed, re-running is the
answer.
