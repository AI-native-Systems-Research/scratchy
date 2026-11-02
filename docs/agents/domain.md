# Domain Docs

What the engineering skills should read about this repo's domain before exploring the code.

## Before exploring, read these

- **`CLAUDE.md`**: the build model, feature flags, and where things live.
- **`CONTRIBUTING.md`**: the workflow, and where specs and plans go.
- **`docs/COMPILER.md`**: the compiler pipeline and its vocabulary.
- **`docs/BUILD.md`**, **`docs/MODELS.md`**, **`docs/TESTING.md`**, **`docs/BENCHMARKING.md`**: read the one that matches the area you're working in.
- **`docs/superpowers/specs/`** and **`docs/superpowers/plans/`**: any spec or plan that touches your area. They're live documents; if your work contradicts one, update it in the same PR.

## Glossary and ADRs

This repo has no `GLOSSARY.md` or `docs/adr/` yet. The `/domain-modeling` skill creates them at the repo root when a term or decision actually gets resolved; don't create them upfront or flag their absence. Once they exist, read them too, use the glossary's terms in your output, and say so explicitly when your output contradicts an ADR.
