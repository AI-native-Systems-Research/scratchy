---
name: Bug report
about: Something compiles or runs incorrectly
title: 'fix(<crate>): '
labels: bug
---

## What happened

<!-- Observed vs expected. -->

## Reproduction

**Feature scope is required** — scratchy specializes per model/quant/backend, so
a repro without the exact `--features` is not reproducible.

```bash
cargo build --release -p scratchy-cli --features <backend>,model/<stem>
# then:
```

## Which layer?

<!--
Best guess is fine, but say why — it's the most useful thing in the report.
-->

- [ ] Macro expansion — wrong tape/layout/arena emitted
- [ ] Shared target-neutral pass
- [ ] Target opcode lowering (metal / cuda / spyre)
- [ ] Serving layer (scheduler, engine, wire API)
- [ ] Build/feature scoping
- [ ] Don't know

## Environment

- Backend + features:
- Host (e.g. M1 Max / 32 GiB, H100, Spyre AIU):
- OS / toolchain (`rustc -Vv`):
- Commit:

## Logs / diagnostics

<!-- Compiler diagnostic, panic + backtrace, or numeric diff vs reference. -->
