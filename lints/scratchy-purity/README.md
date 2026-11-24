# scratchy_purity

The scratchy purity rules (`.claude/skills/scratchy-purity-audit/SKILL.md`) as rustc lints, run
by dylint. Every lint is `Forbid`: no site opts out, and a finding is the verdict.

## Lints

- `p11_escape_hatch_constructor` (P11): a `fn` named `from_untyped` or `conflated_with_*`.
- `p11_tombstone_comment` (P11): a comment that narrates history (`moved from`, `used to be`, ...).
- `p11_tmp_path_comment` (P11): a comment that cites a `/tmp/` path.
- `p11_lint_opt_out` (P11): a `[lints]` table that sets a purity lint's level.
- `p12_compile_fail_without_code` (P12): a `compile_fail` doctest that names no error code, in a doc comment or a `#[doc = ".."]` string.

## Running

From the repository root:

```sh
scripts/purity.sh                   # every scope this host builds
scripts/purity.sh --scope <scope>   # root | metal | metal-all | spyre | spyre-hw | all
scripts/purity.sh --file <path>     # the package that owns <path>
```

The lints' own tests (unit tests and `ui/` tests) run with `cargo test` in this directory.
