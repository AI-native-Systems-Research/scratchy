// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

// crustify:todo: e009_hasVariablesInInstr
//   authority: sys-arch-spec/progir/progir.cpp:677  (18 lines)  `ProgramAndStateInfo::hasVariablesInInstr`

// crustify:todo: e014_getSenComponent
//   authority: sys-arch-spec/progir/progir.cpp:612  (7 lines)  `ProgramAndStateInfo::getSenComponent`

// crustify:todo: e017_isGraphSimple
//   authority: sys-arch-spec/progir/progir.cpp:532  (14 lines)  `ProgIrGraph::isGraphSimple`

// crustify:todo: e030_tagToLCCR
//   authority: sys-arch-spec/progir/progir.cpp:720  (88 lines)  `ProgramAndStateInfo::tagToLCCR`

// crustify:todo: e031_tagToPC
//   authority: sys-arch-spec/progir/progir.cpp:696  (23 lines)  `ProgramAndStateInfo::tagToPC`

// crustify:todo: e033_print
//   authority: sys-arch-spec/progir/progir.cpp:25  (38 lines)  `OperandAttr::print`

