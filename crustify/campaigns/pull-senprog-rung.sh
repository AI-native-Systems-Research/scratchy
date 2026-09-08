#!/bin/bash
# Pull the SenProg -> InitPacket -> init_binary rung from `deeptools-islands` into this branch.
#
# ⭐ THE ISA IS NOT DUPLICATED. That branch keeps the ISA at `crate::isa::{fields, operand, regfile,
# values, encode, unit}`; this branch keeps it in the `sys-arch-spec` CRATE, deduped 12,209 -> 2,549
# lines behind a byte-equality oracle. Four of those six modules exist in sys-arch-spec already, so
# `src/isa/mod.rs` is written as a FACADE re-exporting them, plus the two that are genuinely missing
# (`encode`, `unit`) pulled across. The 4,355 lines of pulled code then compile with their imports
# UNCHANGED, and there is still exactly one ISA.
set -e
cd /Users/nickm/git/scratchy/.claude/worktrees/bridge2
S=crates/compiler/deeptools/src
B=deeptools-islands

mkdir -p $S/isa/encode $S/packet

for f in encode/mod encode/direction encode/forward_pair unit; do
  git show $B:$S/isa/$f.rs > $S/isa/$f.rs
done

for f in mod block header ibuff lrf patch spr; do
  git show $B:$S/packet/$f.rs > $S/packet/$f.rs
done

git show $B:$S/reginit.rs > $S/reginit.rs

wc -l $S/isa/unit.rs $S/isa/encode/*.rs $S/packet/*.rs $S/reginit.rs | tail -1
