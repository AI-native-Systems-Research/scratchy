# triton-numeric - numeric verification harness

## OVERVIEW
`.py` -> TTIR -> KTIR -> `ktir-emulator`, scored against checked-in torch bytes in `../test-numeric/<config>/`.
Earned its file: distinct domain with its own test law and the only emulator edge (score ~17).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Lower a config (values only, no goldens) | `src/lib.rs` `lower(config) -> Lowered` |
| Run on emulator | `execute(&Lowered, &[Binding]) -> Run`; `Run::f64s` |
| Report | `Comparison` (`within_bound`, `report`) |
| Error envelopes (derive the bound) | `src/bounds.rs` (`Envelope`, `U_F16` = 2^-11, `band_up`) |
| Negative controls | `src/mutants.rs` (`Mutant`, `bracket`, `MUTANT_MIN_ROW_FRACTION`) |
| Reference data reader | `src/data.rs` (`Fixture::load`, `verify_sha`, `staged_dir`) |
| Accumulator refold before emulation | `refold_matmul_accumulators` (private, lib.rs) |
| Data generator / encodings | `../test-numeric/gen_numeric_data.py` |

## THE THREE RULES EVERY TEST OWES (lib.rs)
1. The tolerance is derived BEFORE the run, with the algebra written in `bounds`.
2. Every comparison has a control that fails: algorithm mutants plus a bracket (1.5x bound must exceed it, 0.5x must stay within).
3. Report `max|err` AND the bound. "PASSES" is not a result.

## CONVENTIONS
- The emulator is a CORRECTNESS oracle, not a fidelity one: f32 contraction rounded once, per-step rounding to tile dtype, RNE, no sigmoid, no Newton rsqrt.
- `Refusal { stage, message }` prints "REFUSED at {stage}", so a failure never shows up as a blank row.
- The fixture is authoritative over our derivation. SHA-256 is self-tested against FIPS 180-4 vectors.
- Data encodings announce themselves by suffix (`.sparse` SPRSTBL1 absent rows = zero, `.rep` = `repeat_interleave`, `.bin` f16 LE).
- Oversized (>8 MiB) fixtures are staged: `TRITON_NUMERIC_STAGED_<CONFIG>` (preferred) or `TRITON_NUMERIC_STAGED`.
- `ktir-emulator` uses `default-features = false` (no `optimizer`). There is deliberately no `accelerate` feature (macOS links it via blas-src).

## ANTI-PATTERNS
- NEVER read a checked-in ttir/KTIR golden: every stage is computed.
- NEVER copy pod tolerances (`test/pod/*_eps_derivation.py`); re-derive them (a copied one was measured vacuous).
- NEVER bind an argument prefix: refuse on a count mismatch.
- An absent staged dir is NOT a pass; a neighbour config's reference IS the failure mode; a truncated `.bin` must refuse.
- A non-discriminating mutant is reported, never hidden. A one-sided control proves nothing.
- Envelope generalizations must not move a shipped number (`bounds.rs` near the end).
- `tests/embedding_bm128_lx.rs` holds ONE test on purpose: `KTIR_LX_CAPACITY_MB` is process-global.
- In gen_numeric_data.py: no arithmetic a fixture already does, and a CONFIGS disagreement with `cases.rs` is a bug in the script.

## COMMANDS
```bash
cargo test -p triton-numeric
TRITON_NUMERIC_STAGED_SWIGLU_MLP_GRANITE_TILED_K=<dir> TRITON_NUMERIC_STAGED_SWIGLU_MLP_GRANITE_FLAT=<dir> \
  cargo test -p triton-numeric --test executed -- --ignored     # Granite width, 315 MiB
cargo run -p triton-numeric --example ktir_dump -- <config> [ssa]
cargo run -p triton-numeric --example attn_emu
python3 ../test-numeric/gen_numeric_data.py --list | <config> <out-dir>
```
