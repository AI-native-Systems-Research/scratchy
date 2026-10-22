// SPDX-License-Identifier: Apache-2.0
//
// The TurboQuant codebook, compiled in: a function of the code width and the head dim alone
// (`scratchy_layers::turboquant::PolarQuantizer` at `TQ_CODEBOOK_SEED`), set by the lowering's
// `TqCodebookPart`s, every value its own kernel constant:
//   40-55  the N(0,1) Lloyd-Max centroids, `1 << bits` of them;
//   56-70  the boundaries between neighbouring centroids, one fewer;
//   71-86  the random diagonal's signs, one bit an element of the head (set: -1), 32 to a word,
//          `head_dim / 32` words.
// A kernel carries the parts it reads; the rest are unset, and each part's count says how many
// slots its kernel carries, which the shader checks against its own bits and head dim. The
// accessors below read the constants themselves — select trees and compares on them — never an
// array of them.
//
// No `#pragma once`: it declares constants, so every kernel of a bake batch re-reads it.

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
constant constexpr uint TQ_CB_SIGN_WORD_COUNT =
    TQ_CB_S0_SET + TQ_CB_S1_SET + TQ_CB_S2_SET + TQ_CB_S3_SET + TQ_CB_S4_SET + TQ_CB_S5_SET +
    TQ_CB_S6_SET + TQ_CB_S7_SET + TQ_CB_S8_SET + TQ_CB_S9_SET + TQ_CB_S10_SET + TQ_CB_S11_SET +
    TQ_CB_S12_SET + TQ_CB_S13_SET + TQ_CB_S14_SET + TQ_CB_S15_SET;

// The centroid of code `c` (< 16): a select tree on its bits over the centroid constants.
inline float tq_cb_centroid(uint c) {
  const bool b0 = (c & 1u) != 0u, b1 = (c & 2u) != 0u, b2 = (c & 4u) != 0u;
  const float c0 = b0 ? TQ_CB_C1 : TQ_CB_C0, c1 = b0 ? TQ_CB_C3 : TQ_CB_C2;
  const float c2 = b0 ? TQ_CB_C5 : TQ_CB_C4, c3 = b0 ? TQ_CB_C7 : TQ_CB_C6;
  const float c4 = b0 ? TQ_CB_C9 : TQ_CB_C8, c5 = b0 ? TQ_CB_C11 : TQ_CB_C10;
  const float c6 = b0 ? TQ_CB_C13 : TQ_CB_C12, c7 = b0 ? TQ_CB_C15 : TQ_CB_C14;
  const float d0 = b1 ? c1 : c0, d1 = b1 ? c3 : c2, d2 = b1 ? c5 : c4, d3 = b1 ? c7 : c6;
  const float e0 = b2 ? d1 : d0, e1 = b2 ? d3 : d2;
  return (c & 8u) != 0u ? e1 : e0;
}

// The code of `y`: the boundaries it exceeds, each compared against its constant (an unset one
// counts nothing).
inline uint tq_cb_code(float y) {
  return uint(TQ_CB_B0_SET && y > TQ_CB_B0) + uint(TQ_CB_B1_SET && y > TQ_CB_B1) +
         uint(TQ_CB_B2_SET && y > TQ_CB_B2) + uint(TQ_CB_B3_SET && y > TQ_CB_B3) +
         uint(TQ_CB_B4_SET && y > TQ_CB_B4) + uint(TQ_CB_B5_SET && y > TQ_CB_B5) +
         uint(TQ_CB_B6_SET && y > TQ_CB_B6) + uint(TQ_CB_B7_SET && y > TQ_CB_B7) +
         uint(TQ_CB_B8_SET && y > TQ_CB_B8) + uint(TQ_CB_B9_SET && y > TQ_CB_B9) +
         uint(TQ_CB_B10_SET && y > TQ_CB_B10) + uint(TQ_CB_B11_SET && y > TQ_CB_B11) +
         uint(TQ_CB_B12_SET && y > TQ_CB_B12) + uint(TQ_CB_B13_SET && y > TQ_CB_B13) +
         uint(TQ_CB_B14_SET && y > TQ_CB_B14);
}

// Sign word `w` (< 16): a select tree on its bits over the sign-word constants.
inline uint tq_cb_sign_word(uint w) {
  const bool b0 = (w & 1u) != 0u, b1 = (w & 2u) != 0u, b2 = (w & 4u) != 0u;
  const uint s0 = b0 ? TQ_CB_S1 : TQ_CB_S0, s1 = b0 ? TQ_CB_S3 : TQ_CB_S2;
  const uint s2 = b0 ? TQ_CB_S5 : TQ_CB_S4, s3 = b0 ? TQ_CB_S7 : TQ_CB_S6;
  const uint s4 = b0 ? TQ_CB_S9 : TQ_CB_S8, s5 = b0 ? TQ_CB_S11 : TQ_CB_S10;
  const uint s6 = b0 ? TQ_CB_S13 : TQ_CB_S12, s7 = b0 ? TQ_CB_S15 : TQ_CB_S14;
  const uint t0 = b1 ? s1 : s0, t1 = b1 ? s3 : s2, t2 = b1 ? s5 : s4, t3 = b1 ? s7 : s6;
  const uint u0 = b2 ? t1 : t0, u1 = b2 ? t3 : t2;
  return (w & 8u) != 0u ? u1 : u0;
}

// The diagonal's sign at element `e` of the head (< 512): ±1.
inline float tq_cb_sign(uint e) {
  const uint bit = (tq_cb_sign_word(e >> 5) >> (e & 31u)) & 1u;
  return as_type<float>(0x3f800000u | bit << 31);
}

// This kernel's codebook as a type, for the guarded encode (`turboquant_encode.h`), which is
// compiled once ahead of a bake batch and reads each kernel's constants through it.
struct TqCodebook {
  static float sign(uint e) { return tq_cb_sign(e); }
  static uint code(float y) { return tq_cb_code(y); }
};
