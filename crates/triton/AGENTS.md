# crates/triton - the Triton ladder (re-hosted)

## OVERVIEW
Triton `.py` kernel -> TTIR -> KTIR -> SuperDSC, all pure Rust, plus the splice that feeds
the spyre target. Re-hosted from triton-spyre; `PROVENANCE.md` is the authority on origin and copy rules.
Earned its file: 5 workspace crates + shared test data, cross-crate rules no single crate states.

## STRUCTURE
```
crates/triton/
├── triton-frontend/       # bridge one: .py -> ttir::Module (owned parser, zero deps)
├── triton-ktir/           # bridge two: ttir value -> KTIR value (ported C++ passes)
├── triton-ktir-superdsc/  # adapter: KTIR -> ktir-superdsc KtirNode; cases.rs; bake
├── triton-numeric/        # numeric harness: lower + ktir-emulator vs torch bytes
├── splice/                # scratchy-triton-splice: SubOp -> spliced kernel rows (spyre)
├── test-fixtures/         # kernel .py sources (ladder inputs)
├── test-goldens/ktir/     # KTIR/TTIR goldens per config (+ grid.txt)
├── test-experiment1/      # C++ chain's checked-in KTIR
└── test-numeric/          # torch reference bytes + meta.json/sha256.txt per config
```

## DEPENDENCY ARROW
frontend <- ktir <- ktir-superdsc <- numeric; splice uses all three plus
`scratchy-subtile` (feature `spyre`). KTIR/SuperDSC definitions live in
`crates/targets/spyre/ktir/{ktir-core,ktir-superdsc,ktir-emulator}`, not here.
frontend's only reverse edge (`triton-ktir`) is a dev-dep, which avoids a cycle.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Origin, ownership split, stale-golden pins | `PROVENANCE.md` |
| Add/adjust a test configuration | `triton-ktir-superdsc/src/cases.rs` (stated ONCE) |
| Regenerate numeric data | `test-numeric/gen_numeric_data.py` (CONFIGS mirrors cases.rs) |
| Spyre consumer of the ladder | `crates/targets/spyre/src/lower_subtile_tape_to_{ktir,superdsc}.rs` |
| CI gate | `.github/workflows/rust.yml` job `linux-spyre-triton` |

## CONVENTIONS (ladder-wide)
- Every stage is a VALUE. No stage prints text for another to parse; each crate's
  `print`/`parse`/`text` modules are test instruments, unreachable from the lowering path.
- Fail closed: unsupported construct = named refusal (construct + line:col / stage), never partial lowering.
- Edits happen HERE and flow out to triton-spyre; never edit their copy.
- No `[workspace]` table in any crate (scratchy workspace members).
- Test data path: one `.parent()` from `CARGO_MANIFEST_DIR` -> `crates/triton/`; `test-*` are siblings, never inside a crate.
- Zero crates.io deps on the lowering path (pod runs `cargo --offline`); `serde_json` is the documented exception (ktir-superdsc, numeric).
- Comment style: ALL-CAPS thesis lines plus the concrete past defect behind each rule. Keep it when you edit.

## ANTI-PATTERNS
- NEVER "clean up" float literals (`0.011271055`, `1.44269504`, `0.0078125`): their bits are golden-visible.
- NEVER copy upstream (PTX/C++) semantics where Spyre differs; the f16 division divergence is in `triton-frontend/src/target.rs`.
- NEVER let a test pass comparing nothing: planted-difference / mutant / negative controls are part of every oracle.
- NEVER skip on a missing golden or staged dir: missing = failure, panic with the regen command.
- NEVER un-pin `attention_flash_noncausal`'s stale TTIR golden alone; the three pins move together (PROVENANCE.md).
- Not here, by design: their `triton-superdsc*` crates and pod scorers.

## COMMANDS
```bash
# exactly CI's two triton steps (pure Rust, no hardware, no model feature needed)
cargo clippy -p triton-frontend -p triton-ktir -p triton-ktir-superdsc -p triton-numeric \
  -p scratchy-triton-splice --all-targets -- -D warnings
cargo test -p triton-frontend -p triton-ktir -p triton-ktir-superdsc -p triton-numeric -p scratchy-triton-splice
```
CI's workspace clippy has no `--all-targets`, so only `linux-spyre-triton` compiles these crates' tests and examples.
