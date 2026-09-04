---
name: Feature request
about: New capability, model architecture, or performance work
title: 'feat(<crate>): '
labels: enhancement
---

## Problem

<!-- The problem and use case, not the solution. -->

## Proposed approach

<!-- Optional. If it touches the compiler, say where it lands. -->

## Planning tier

<!--
See docs/CONTRIBUTING.md step 3. New architectures, compiler-pass changes,
anything touching the architecture invariants, and tracking issues need a
spec + plan under docs/superpowers/ before code.
-->

- [ ] Tier 1 — micro-plan is enough
- [ ] Tier 2 — needs a spec + plan pair

## Acceptance criteria

<!-- How do we know this is done? -->

## Invariant impact

<!--
See docs/CONTRIBUTING.md. Anything requiring runtime analysis, a mirror type, an
env-var toggle, ISel on metal/spyre, or net line growth in a target crate needs
its case made here first.
-->

- [ ] Stays within the existing invariants
- [ ] Needs an invariant discussion (explain below)

## Evidence

<!--
For perf work: measured numbers, hardware, methodology, and the baseline. State
hypotheses as hypotheses. Include counts/paths you derived yourself — and note
that reviewers will re-derive them.
-->
