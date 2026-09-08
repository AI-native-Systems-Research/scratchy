// SPDX-License-Identifier: Apache-2.0
//! `ProgIR -> SenProg` — bridge 4, ported from `sys-arch-spec/{dpc,progir,isa}`.
//!
//! ⭐ THE EMISSION'S CLOSURE IS FOUR FILES, NOT ONE. `dpc.cpp` holds only 3 of the 33 units;
//! `progir.h` holds 18 as IN-CLASS definitions, which a `.cpp`-only scan would miss entirely.

pub mod isa_fields;
pub mod progir_inline;
pub mod progir_print;
pub mod senprog_writer;
