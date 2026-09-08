// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/dpc/dpc.cpp` — 3 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

// crustify:todo: e010_checkProgFormatCompatibility
//   authority: sys-arch-spec/dpc/dpc.cpp:87  (10 lines)  `Dpc::checkProgFormatCompatibility`

// crustify:todo: e028_strToupper
//   authority: sys-arch-spec/dpc/dpc.cpp:50  (4 lines)  `strToupper`

// crustify:todo: e032_convertIr2Senprog
//   authority: sys-arch-spec/dpc/dpc.cpp:615  (163 lines)  `Dpc::convertIr2Senprog`

