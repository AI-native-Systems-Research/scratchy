// SPDX-License-Identifier: Apache-2.0
//
// The TurboQuant codebook, compiled in: a function of the code width and the head dim alone
// (`scratchy_layers::turboquant::PolarQuantizer` at `TQ_CODEBOOK_SEED`), set by the lowering's
// `TqCodebookConstants` / `TqSignConstants`:
//   40-55  the N(0,1) Lloyd-Max centroids, `1 << bits` of them;
//   56-70  the boundaries between neighbouring centroids, one fewer;
//   71-86  the random diagonal's signs, one bit an element of the head (set: -1), 32 to a word,
//          `head_dim / 32` words.
// A kernel carries the parts it reads; the rest are unset, and each part's count says how many
// slots its kernel carries, which the shader checks against its own bits and head dim.
//
// No `#pragma once`: it declares constants, so every kernel of a bake batch re-reads it.

#include "turboquant_encode.h"

SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C0, 40);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C1, 41);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C2, 42);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C3, 43);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C4, 44);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C5, 45);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C6, 46);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C7, 47);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C8, 48);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C9, 49);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C10, 50);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C11, 51);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C12, 52);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C13, 53);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C14, 54);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_C15, 55);
constant constexpr float TQ_CB_CENTROIDS[16] = {
    TQ_CB_C0, TQ_CB_C1, TQ_CB_C2,  TQ_CB_C3,  TQ_CB_C4,  TQ_CB_C5,  TQ_CB_C6,  TQ_CB_C7,
    TQ_CB_C8, TQ_CB_C9, TQ_CB_C10, TQ_CB_C11, TQ_CB_C12, TQ_CB_C13, TQ_CB_C14, TQ_CB_C15};
constant constexpr uint TQ_CB_CENTROID_COUNT =
    TQ_CB_C0_SET + TQ_CB_C1_SET + TQ_CB_C2_SET + TQ_CB_C3_SET + TQ_CB_C4_SET + TQ_CB_C5_SET +
    TQ_CB_C6_SET + TQ_CB_C7_SET + TQ_CB_C8_SET + TQ_CB_C9_SET + TQ_CB_C10_SET + TQ_CB_C11_SET +
    TQ_CB_C12_SET + TQ_CB_C13_SET + TQ_CB_C14_SET + TQ_CB_C15_SET;

SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B0, 56);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B1, 57);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B2, 58);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B3, 59);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B4, 60);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B5, 61);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B6, 62);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B7, 63);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B8, 64);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B9, 65);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B10, 66);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B11, 67);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B12, 68);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B13, 69);
SCRATCHY_CONSTANT_OPTIONAL(float, TQ_CB_B14, 70);
constant constexpr float TQ_CB_BOUNDARIES[15] = {
    TQ_CB_B0, TQ_CB_B1, TQ_CB_B2,  TQ_CB_B3,  TQ_CB_B4,  TQ_CB_B5,  TQ_CB_B6, TQ_CB_B7,
    TQ_CB_B8, TQ_CB_B9, TQ_CB_B10, TQ_CB_B11, TQ_CB_B12, TQ_CB_B13, TQ_CB_B14};
constant constexpr uint TQ_CB_BOUNDARY_COUNT =
    TQ_CB_B0_SET + TQ_CB_B1_SET + TQ_CB_B2_SET + TQ_CB_B3_SET + TQ_CB_B4_SET + TQ_CB_B5_SET +
    TQ_CB_B6_SET + TQ_CB_B7_SET + TQ_CB_B8_SET + TQ_CB_B9_SET + TQ_CB_B10_SET + TQ_CB_B11_SET +
    TQ_CB_B12_SET + TQ_CB_B13_SET + TQ_CB_B14_SET;

SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S0, 71);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S1, 72);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S2, 73);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S3, 74);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S4, 75);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S5, 76);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S6, 77);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S7, 78);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S8, 79);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S9, 80);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S10, 81);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S11, 82);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S12, 83);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S13, 84);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S14, 85);
SCRATCHY_CONSTANT_OPTIONAL(uint, TQ_CB_S15, 86);
constant constexpr uint TQ_CB_SIGN_WORDS[16] = {
    TQ_CB_S0, TQ_CB_S1, TQ_CB_S2,  TQ_CB_S3,  TQ_CB_S4,  TQ_CB_S5,  TQ_CB_S6,  TQ_CB_S7,
    TQ_CB_S8, TQ_CB_S9, TQ_CB_S10, TQ_CB_S11, TQ_CB_S12, TQ_CB_S13, TQ_CB_S14, TQ_CB_S15};
constant constexpr uint TQ_CB_SIGN_WORD_COUNT =
    TQ_CB_S0_SET + TQ_CB_S1_SET + TQ_CB_S2_SET + TQ_CB_S3_SET + TQ_CB_S4_SET + TQ_CB_S5_SET +
    TQ_CB_S6_SET + TQ_CB_S7_SET + TQ_CB_S8_SET + TQ_CB_S9_SET + TQ_CB_S10_SET + TQ_CB_S11_SET +
    TQ_CB_S12_SET + TQ_CB_S13_SET + TQ_CB_S14_SET + TQ_CB_S15_SET;

// The diagonal's sign at element `e` of the head: ±1.
inline float tq_cb_sign(uint e) { return tq_sign(TQ_CB_SIGN_WORDS, e); }
