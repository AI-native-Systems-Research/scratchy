// SPDX-License-Identifier: Apache-2.0
//! THE ISA, AS ONE SOURCE OF TRUTH — a facade, not a second copy.
//!
//! # 🛑 WHY THIS MODULE IS A RE-EXPORT AND NOT A TREE OF FILES
//!
//! ⛔⛔ THE `deeptools-islands` BRANCH KEEPS THE ISA HERE, AT `crate::isa::*`. This branch keeps it in
//! the **`sys-arch-spec` crate**, deduped 12,209 → 2,549 lines behind a byte-equality snapshot oracle
//! (`sys-arch-spec/tests/the_tables_are_unchanged.rs`, written *before* the dedup). Pulling that
//! branch's `isa/` across wholesale would give this crate TWO opcode tables, two field tables and two
//! register files — the "one quantity computed twice" defect, in the one place where a divergence is
//! guaranteed to be silent: a field offset that differs by one produces an instruction the hardware
//! decodes as something else.
//!
//! ⭐ SO FOUR OF THE SIX MODULES ARE RE-EXPORTS of `sys_arch_spec`'s own, and the 4,355 lines of
//! SenProg/InitPacket code pulled from that branch compile with their `crate::isa::…` imports
//! UNCHANGED. Nothing was rewritten to fit, and nothing was copied twice.
//!
//! ⛔ TWO ARE GENUINELY ABSENT from `sys-arch-spec` and are pulled across: [`encode`] (the
//! instruction-word encoders) and [`unit`] (executor identity, PT row indices, forwarding). They are
//! the only new ISA code on this branch, and if `sys-arch-spec` ever grows them, they must be deleted
//! here rather than kept in parallel.

/// The instruction-word encoders — ⛔ NOT in `sys-arch-spec`; pulled from `deeptools-islands`.
pub mod encode;
/// Executor identity, PT row indices and forwarding — ⛔ NOT in `sys-arch-spec`; pulled.
pub mod unit;

/// ⭐ `sys_arch_spec::fields` — the field tables, and the `ENC_*` consts the dedup introduced.
pub use sys_arch_spec::fields;
/// ⭐ `sys_arch_spec::operand` — the operand table.
pub use sys_arch_spec::operand;
/// ⭐ `sys_arch_spec::regfile` — components, register types, `max_ibuff_entries`.
pub use sys_arch_spec::regfile;
/// ⭐ `sys_arch_spec::values` — the value/opcode-unit tables.
pub use sys_arch_spec::values;
