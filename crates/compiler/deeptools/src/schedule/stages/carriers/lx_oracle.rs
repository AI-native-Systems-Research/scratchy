// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE WIRED TRACKER AGAINST THE REFERENCE'S OWN LX ADDRESSES — every LX allocation of all 187
//! programs the reference pipeline placed for scratchy's own staged bundle.
//!
//! # WHERE THE NUMBERS COME FROM
//!
//! `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_<N>/sdsc.json`, `N` in `0..187` — the SCHEDULED
//! output of `dxp`'s own run over `g0/sdsc_<N>.json`. For each `nodeType_ == "allocate"` node with
//! `component_ == "lx"` the table below carries three of its fields:
//!
//!   * `ldsIdx_` — the key `ldsIdxAndAllocNode` is ordered by, which is the tie-break.
//!   * `numBuffers_ * bufferOffsetCoreCorelet_` — the FOOTPRINT, i.e. `mySize`
//!     (`L3DlOpsScheduler.cpp:5556-5561` asks for it, `:5655-5657` writes the offset back as
//!     `kv.second / numBuffers`, so the product is the size that was committed).
//!   * `startAddressCoreCorelet_.data_` — THE PLACED ADDRESS, which is what this test checks.
//!
//! ⛔⛔ `bufferOffsetCoreCorelet_` IS A SIZE, NOT AN ADDRESS AND NOT A STRIDE BETWEEN NODES. It is
//! 256 on every LX node of `sdsc_0` and on all 32 cores, so a check written against it would pass for
//! a tracker that allocated everything at 256. The rule that catches this: print a fixture field
//! across several nodes before trusting it — measured over all 187, it takes seven distinct values
//! (256, 512, 1024, 4096, 8192, 16384, 262144) while `startAddressCoreCorelet_.data_` takes ten.
//!
//! ⭐ ONE ADDRESS PER NODE IS THE WHOLE ANSWER, MEASURED: over the 588 LX nodes' 7,801
//! `[core, corelet, time]` samples, every node's address is identical at every coordinate — 0
//! exceptions. This test still places on all 32 cores, because each core has its OWN tracker and a
//! wiring that shared one would place core 1 above core 0.
//!
//! # WHAT THIS TEST DOES AND DOES NOT COVER
//!
//! ⭐ IT DRIVES THE SEAM, NOT A RE-DERIVATION: [`ExPhaseTrackers::backup`],
//! [`ExPhaseTrackers::remove`] and [`ExPhaseTrackers::check_and_add`] on a real [`Trackers`], in the
//! order and with the arguments entry 222 uses (`L3DlOpsScheduler.cpp:5537-5645`). The ADDRESS is
//! the port's own answer; the footprint and the expected address are the reference's.
//!
//! ⛔ IT DOES NOT VERIFY THE FOOTPRINT. `getBufferCapacityForNode` (`dsc/dsc2.cpp:3977`) IS PORTED —
//! [`crate::schedule::l3::capacity::buffer_capacity`], with
//! [`crate::schedule::l3::capacity::DscSizing`] answering its `SizeDsc` seam — but
//! [`crate::schedule::l3::dl_ops::L3Placement::buffer_capacity_even_sticks`] is not handed the
//! `&DesignSpaceConfig` the reference calls it on, so [`super::Placement`] refuses and the SIZE of
//! each request is taken from the reference's output rather than computed. ⭐ THAT REFUSAL IS NOW
//! STAGE 2A'S FRONTIER ON EVERY PROGRAM: entry 222 finds
//! its allocate node (the port's separate allocate-node map is gone) and stops on the capacity
//! instead. This test says nothing about it.
//!
//! ⛔ AND WHAT THE 187 DO NOT EXERCISE, so a green run here is not a verified allocator: every one
//! packs consecutively from the base with zero gaps, so there is no fragmented free list, no
//! `allocFromBack`, no non-zero `margin`, no `eps.size() > 10 && opt_frag_` arm and no `DOESNT_FIT`
//! (the worst program consumes 266,752 of the 406,272 bytes left above the reservation).

use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use super::{ExecutionStep, LxAvailFraction, Trackers, reserved_frontend};
use crate::arch::{Bytes, Dd2};
use crate::schedule::ddc::v1;
use crate::schedule::l3::dl_ops::{ExPhase, ExPhaseTrackers, L3TrackerSite};
use crate::schedule::memtrack::bundle::BundleSite;
use crate::schedule::memtrack::memory::Capacity;
use crate::schedule::memtrack::tracker::DsAddress;
use crate::units::{Core, Corelet, Row};

/// ONE LX ALLOCATION AS THE REFERENCE PLACED IT — `(ldsIdx_, numBuffers_ * bufferOffsetCoreCorelet_,
/// startAddressCoreCorelet_.data_)`.
type Placed = (u32, u64, u64);

/// LX's whole space — `lxCapacity` (`sys-arch-spec/sysdef.cpp:211`), which is
/// [`crate::arch::Arch::LX_CAPACITY`].
const LX_CAPACITY: u64 = 2_031_616;

/// ⭐ THE BASE — `reserved-frontend`'s 1,625,292 bytes rounded up to a stick
/// (`dbo/src/Transforms/ProgramLayout.cpp:92-93`, `mem_track.cpp:428-429`).
const BASE: u64 = 1_625_344;

/// ⭐⭐ EVERY LX ALLOCATION OF ALL 187 REFERENCE PROGRAMS — `(sdsc index, program name, the placed
/// nodes by `ldsIdx_` ascending, the `_internalInput` aliases)`.
///
/// ⛔ THE ALIASES ARE THE FOURTH COLUMN BECAUSE THEY ARE NOT TRACKER REQUESTS. Eight programs carry
/// an `allocate_lds0_lx_internalInput` node at a SECOND `ldsIdx_` whose address is its principal's
/// and which consumes nothing — see [`an_internal_input_alias_is_not_a_tracker_request`].
///
/// ⛔ NOT IN ADDRESS ORDER, AND THAT IS THE POINT: `15_t729_fq_mm` places `ldsIdx_` 1 (262,144 bytes)
/// at the base and `ldsIdx_` 0 (4,096) above it, so a tracker that walked `ldsIdx_` ascending would
/// answer 1,625,344 where the reference answers 1,887,488.
///
/// ⛔ `rustfmt::skip` SO ONE PROGRAM STAYS ONE LINE — reflowed, this table is 1,700 lines and no
/// reviewer can read the corpus off it.
#[rustfmt::skip]
const ORACLE: &[(u32, &str, &[Placed], &[Placed])] = &[
    (0, "0_rmsq_o728", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (1, "1_rmmean_o728", &[(0, 8_192, 1_625_344), (2, 512, 1_633_536)], &[]),
    (2, "2_rmeps_o728", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (3, "3_rmrsqrt_o728", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (4, "4_rmxn_o728", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (5, "5_rmg_o728", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (6, "6_t729_fq_absx_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (7, "7_t729_fq_amax_op", &[(0, 8_192, 1_625_344), (2, 512, 1_633_536)], &[]),
    (8, "8_t729_fq_amaxfl_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (9, "9_t729_fq_ascale_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (10, "10_t729_fq_invs_op", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (11, "11_t729_fq_sc_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (12, "12_t729_fq_chi_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (13, "13_t729_fq_cl_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (14, "14_t729_fq_afp8_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (15, "15_t729_fq_mm", &[(0, 4_096, 1_887_488), (1, 262_144, 1_625_344), (4, 512, 1_891_584)], &[]),
    (16, "16_t729_fq_dqa_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (17, "17_t729_fq_dqw_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (18, "18_t730_fq_mm", &[(0, 4_096, 1_887_488), (1, 262_144, 1_625_344), (4, 512, 1_891_584)], &[]),
    (19, "19_t730_fq_dqa_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (20, "20_t730_fq_dqw_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (21, "21_t731_fq_mm", &[(0, 4_096, 1_887_488), (1, 262_144, 1_625_344), (4, 512, 1_891_584)], &[]),
    (22, "22_t731_fq_dqa_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (23, "23_t731_fq_dqw_op", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (24, "24_rope_rot_o732", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (25, "25_rope_xc_o732", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (26, "26_rope_rs_o732", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (27, "27_rope_add_o732", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (28, "28_rope_rot_o733", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (29, "29_rope_xc_o733", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (30, "30_rope_rs_o733", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (31, "31_rope_add_o733", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (32, "32_attn_kzero0_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (33, "33_attn_vzero0_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (34, "34_attn_kzero1_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (35, "35_attn_vzero1_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (36, "36_attn_kzero2_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (37, "37_attn_vzero2_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (38, "38_attn_kzero3_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (39, "39_attn_vzero3_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (40, "40_attn_kzero4_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (41, "41_attn_vzero4_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (42, "42_attn_kzero5_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (43, "43_attn_vzero5_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (44, "44_attn_kzero6_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (45, "45_attn_vzero6_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (46, "46_attn_kzero7_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (47, "47_attn_vzero7_o734", &[(0, 1_024, 1_625_344), (1, 1_024, 1_626_368)], &[]),
    (48, "48_attn_qs_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (49, "49_attn_nks_o734", &[(0, 4_096, 1_625_344), (1, 256, 1_633_536), (2, 4_096, 1_629_440)], &[]),
    (50, "50_attn_newkt0_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (51, "51_attn_newkt1_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (52, "52_attn_newkt2_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (53, "53_attn_newkt3_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (54, "54_attn_newkt4_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (55, "55_attn_newkt5_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (56, "56_attn_newkt6_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (57, "57_attn_newkt7_o734", &[(0, 16_384, 1_625_344), (4, 16_384, 1_641_728)], &[(1, 16_384, 1_625_344)]),
    (58, "58_attn_nsc_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (59, "59_attn_nsc_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (60, "60_attn_nsc_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (61, "61_attn_nsc_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (62, "62_attn_nsc_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (63, "63_attn_nsc_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (64, "64_attn_nsc_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (65, "65_attn_nsc_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (66, "66_attn_nbmax_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (67, "67_attn_nesub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (68, "68_attn_nee_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (69, "69_attn_nbsum_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (70, "70_attn_nov_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (71, "71_attn_nov_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (72, "72_attn_nov_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (73, "73_attn_nov_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (74, "74_attn_nov_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (75, "75_attn_nov_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (76, "76_attn_nov_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (77, "77_attn_nov_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (4, 512, 1_642_240)], &[]),
    (78, "78_attn_p0sc_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (79, "79_attn_p0sc_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (80, "80_attn_p0sc_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (81, "81_attn_p0sc_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (82, "82_attn_p0sc_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (83, "83_attn_p0sc_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (84, "84_attn_p0sc_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (85, "85_attn_p0sc_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (86, "86_attn_p0bmax_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (87, "87_attn_p0newm_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (88, "88_attn_p0corrsub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (89, "89_attn_p0corre_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (90, "90_attn_p0mset_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (91, "91_attn_p0esub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (92, "92_attn_p0ee_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (93, "93_attn_p0bsum_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (94, "94_attn_p0ocorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (95, "95_attn_p0ov_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (96, "96_attn_p0ov_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (97, "97_attn_p0ov_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (98, "98_attn_p0ov_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (99, "99_attn_p0ov_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (100, "100_attn_p0ov_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (101, "101_attn_p0ov_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (102, "102_attn_p0ov_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (103, "103_attn_p0lcorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (104, "104_attn_p0ladd_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (105, "105_attn_p1sc_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (106, "106_attn_p1sc_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (107, "107_attn_p1sc_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (108, "108_attn_p1sc_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (109, "109_attn_p1sc_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (110, "110_attn_p1sc_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (111, "111_attn_p1sc_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (112, "112_attn_p1sc_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (113, "113_attn_p1bmax_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (114, "114_attn_p1newm_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (115, "115_attn_p1corrsub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (116, "116_attn_p1corre_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (117, "117_attn_p1mset_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (118, "118_attn_p1esub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (119, "119_attn_p1ee_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (120, "120_attn_p1bsum_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (121, "121_attn_p1ocorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (122, "122_attn_p1ov_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (123, "123_attn_p1ov_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (124, "124_attn_p1ov_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (125, "125_attn_p1ov_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (126, "126_attn_p1ov_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (127, "127_attn_p1ov_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (128, "128_attn_p1ov_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (129, "129_attn_p1ov_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (130, "130_attn_p1lcorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (131, "131_attn_p1ladd_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (132, "132_attn_p2sc_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (133, "133_attn_p2sc_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (134, "134_attn_p2sc_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (135, "135_attn_p2sc_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (136, "136_attn_p2sc_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (137, "137_attn_p2sc_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (138, "138_attn_p2sc_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (139, "139_attn_p2sc_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (140, "140_attn_p2bmax_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (141, "141_attn_p2newm_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (142, "142_attn_p2corrsub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (143, "143_attn_p2corre_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (144, "144_attn_p2mset_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (145, "145_attn_p2esub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (146, "146_attn_p2ee_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (147, "147_attn_p2bsum_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (148, "148_attn_p2ocorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (149, "149_attn_p2ov_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (150, "150_attn_p2ov_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (151, "151_attn_p2ov_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (152, "152_attn_p2ov_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (153, "153_attn_p2ov_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (154, "154_attn_p2ov_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (155, "155_attn_p2ov_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (156, "156_attn_p2ov_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (157, "157_attn_p2lcorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (158, "158_attn_p2ladd_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (159, "159_attn_p3sc_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (160, "160_attn_p3sc_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (161, "161_attn_p3sc_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (162, "162_attn_p3sc_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (163, "163_attn_p3sc_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (164, "164_attn_p3sc_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (165, "165_attn_p3sc_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (166, "166_attn_p3sc_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (167, "167_attn_p3bmax_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (168, "168_attn_p3newm_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (169, "169_attn_p3corrsub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (170, "170_attn_p3corre_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (171, "171_attn_p3mset_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856)], &[]),
    (172, "172_attn_p3esub_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (173, "173_attn_p3ee_o734", &[(0, 512, 1_625_344), (3, 512, 1_625_856)], &[]),
    (174, "174_attn_p3bsum_o734", &[(0, 512, 1_625_344), (2, 512, 1_625_856)], &[]),
    (175, "175_attn_p3ocorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (176, "176_attn_p3ov_g0_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (177, "177_attn_p3ov_g1_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (178, "178_attn_p3ov_g2_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (179, "179_attn_p3ov_g3_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (180, "180_attn_p3ov_g4_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (181, "181_attn_p3ov_g5_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (182, "182_attn_p3ov_g6_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (183, "183_attn_p3ov_g7_o734", &[(0, 512, 1_641_728), (1, 16_384, 1_625_344), (2, 512, 1_642_240), (5, 512, 1_642_752)], &[]),
    (184, "184_attn_p3lcorr_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (185, "185_attn_p3ladd_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (2, 512, 1_626_368)], &[]),
    (186, "186_attn_o_o734", &[(0, 512, 1_625_344), (1, 512, 1_625_856), (3, 512, 1_626_368)], &[]),
];

/// The site entry 222 reaches for `(LX, core)` — corelet and row are its own `0` proxies
/// (`L3DlOpsScheduler.cpp:5532-5534`).
fn lx_site(core: Core) -> L3TrackerSite {
    L3TrackerSite {
        memory: SenComponent::Lx,
        core,
        corelet: Corelet::at::<0>(),
        row: Row::at::<0>(),
    }
}

/// `currDsc->labeledDs_.at(anode->ldsIdx_).dsName_` (`L3DlOpsScheduler.cpp:5497`) — the tracker key
/// is a NAME, and only distinctness within a program decides anything here. Measured over the 187:
/// no program holds two LX allocations at one `ldsIdx_`, so `ldsIdx_` names them apart.
fn ds_name(lds: u32) -> v1::StorageName {
    v1::StorageName(format!("lds{lds}"))
}

/// `nodeAndSize` IN THE ORDER ENTRY 222 ASKS IN — built over `ldsIdxAndAllocNode`, a
/// `std::map<int, AllocateNode *>` (`L3DlOpsScheduler.h:162`), then sorted by size DESCENDING
/// (`:5593-5598`).
///
/// ⛔ THE TIE-BREAK IS THE BUILD ORDER AND `sort_by` IS STABLE — the same pairing the port itself
/// performs (`l3/dl_ops.rs:9271`), over a table written in `ldsIdx_`-ascending order.
fn requested(placed: &[Placed]) -> Vec<Placed> {
    let mut order = placed.to_vec();
    order.sort_by(|left, right| right.1.cmp(&left.1));
    order
}

/// Entry 222's own sequence for one `(comp, core)` site: back the tracker up, remove every candidate,
/// then place them largest first — and hand back the address each one got
/// (`L3DlOpsScheduler.cpp:5537-5645`).
fn place(trackers: &mut Trackers, core: Core, placed: &[Placed]) -> Vec<(u32, Option<v1::Placed>)> {
    let at = lx_site(core);
    trackers.backup(at);
    for &(lds, _, _) in placed {
        trackers.remove(at, &ds_name(lds));
    }
    requested(placed)
        .into_iter()
        .map(|(lds, footprint, _)| {
            (
                lds,
                trackers.check_and_add(at, ExPhase(0), &ds_name(lds), Bytes(footprint)),
            )
        })
        .collect()
}

/// ⭐⭐ THE BASE IS THE FRONT END'S RESERVATION, ROUNDED UP TO A STICK — and it is the number the
/// whole corpus is placed above: `(int64_t)(2031616 * (1 - 0.2))` = 1,625,292
/// (`dbo/src/Transforms/ProgramLayout.cpp:92`), which `checkAndAddDsAtAddr` rounds to 1,625,344
/// (`mem_track.cpp:428-429`) and pins at address 0.
#[test]
fn the_base_is_the_front_end_reservation_rounded_up_to_a_stick() {
    assert_eq!(
        LxAvailFraction::DEFAULT.reserved(Capacity(2_031_616)),
        Capacity(1_625_292),
        "`memCapacity * (1 - lx_avail_frac)`, truncated"
    );

    let trackers = Trackers::at_step::<Dd2>(ExecutionStep::default());
    let core = Core::checked(0).expect("core 0");

    // ⛔ THE CAPACITY IS THE TRACKER'S OWN `memCapacity`, not a constant in the carrier.
    assert_eq!(
        trackers.capacity(lx_site(core)),
        Bytes(LX_CAPACITY),
        "`lxCapacity` (`sysdef.cpp:211`)"
    );

    // The reservation is a real block: `[0, 1625344)`, held under `reserved-frontend`.
    let tracker = trackers.bundle.tracker(BundleSite::Lx(core));
    assert_eq!(
        tracker.addr_of_ds(ExPhase(0), &reserved_frontend()),
        DsAddress::At(crate::schedule::memtrack::memory::Address::ZERO),
        "`checkAndAddDsAtAddr(\"reserved-frontend\", reserved, phases, 0)`"
    );
    assert_eq!(
        tracker.cap_of_ds(ExPhase(0), &reserved_frontend()),
        Some(Capacity(1_625_344)),
        "the ROUNDED capacity `free_` was debited by"
    );
    assert_eq!(
        tracker.free_cap_at_eps(ExPhase(0)),
        Capacity(406_272),
        "2,031,616 - 1,625,344"
    );

    // ...so the first real allocation lands exactly at the base, on every one of the 32 cores.
    for core in (0..).map_while(Core::checked) {
        let mut trackers = Trackers::at_step::<Dd2>(ExecutionStep::default());
        assert_eq!(
            place(&mut trackers, core, &[(0, 512, BASE)]),
            vec![(0, Some(v1::Placed::At(Bytes(BASE))))],
            "core {}'s first LX allocation",
            core.get()
        );
    }
}

/// ⭐⭐ 187/187 PROGRAMS, 580 REQUESTS, ALL 32 CORES — the wired tracker's address for every LX
/// allocation of the reference's own output, against the reference's own `startAddressCoreCorelet_`.
///
/// ⛔ THE FAILURE CARRIES VALUES: every mismatch is reported as
/// `program/ldsIdx footprint: got -> want`, never as a count.
#[test]
fn every_lx_allocation_of_all_187_reference_programs_lands_where_the_reference_put_it() {
    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0_usize;
    let mut programs = 0_usize;

    for &(sdsc, name, placed, _) in ORACLE {
        programs += 1;
        // ONE bundle per program, as the reference has one phase per SDSC node and places each
        // program into its own (`MemTrackerInit.cpp:85-99`).
        let mut trackers = Trackers::at_step::<Dd2>(ExecutionStep::default());
        for core in (0..).map_while(Core::checked) {
            let want: BTreeMap<u32, u64> = placed.iter().map(|&(lds, _, at)| (lds, at)).collect();
            for (lds, got) in place(&mut trackers, core, placed) {
                checked += 1;
                let expected = want.get(&lds).copied().expect("a node of this program");
                if got != Some(v1::Placed::At(Bytes(expected))) {
                    wrong.push(format!(
                        "sdsc_{sdsc} {name} lds{lds} core{}: {got:?} -> want {expected}",
                        core.get()
                    ));
                }
            }
        }
    }

    assert_eq!(programs, 187, "every program of `g0/`");
    assert_eq!(
        checked,
        580 * 32,
        "580 tracker requests (588 LX nodes less the 8 `_internalInput` aliases) on 32 cores"
    );
    assert!(
        wrong.is_empty(),
        "{} of {checked} placements diverge from the reference:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// ⭐⭐ AN `_internalInput` ALIAS IS NOT A TRACKER REQUEST — it carries its principal's address and
/// consumes nothing, and the negative control shows what treating it as a request would cost.
///
/// `50_attn_newkt0_o734` (`g0/debug/sdsc_50/sdsc.json`) holds THREE LX allocate nodes:
/// `allocate_lds0_lx` (`ldsIdx_` 0) and `allocate_lds0_lx_internalInput` (`ldsIdx_` 1) BOTH at
/// 1,625,344, and `allocate_lds1_lx` (`ldsIdx_` 4) at 1,641,728 = 1,625,344 + 16,384. One 16,384-byte
/// block was consumed, not two.
#[test]
fn an_internal_input_alias_is_not_a_tracker_request() {
    let core = Core::checked(0).expect("core 0");
    let (_, name, placed, aliases) = ORACLE
        .iter()
        .find(|(sdsc, ..)| *sdsc == 50)
        .copied()
        .expect("sdsc_50");
    assert_eq!(name, "50_attn_newkt0_o734");
    assert_eq!(aliases, [(1, 16_384, 1_625_344)], "the one alias");

    // Every alias of the corpus repeats an address a principal of the same program already holds.
    for &(sdsc, name, placed, aliases) in ORACLE {
        for &(lds, _, at) in aliases {
            assert!(
                placed.iter().any(|&(_, _, principal)| principal == at),
                "sdsc_{sdsc} {name} lds{lds} sits at {at}, which no principal holds"
            );
        }
    }

    // The two REAL requests, in the reference's own order and at the reference's own addresses.
    let mut trackers = Trackers::at_step::<Dd2>(ExecutionStep::default());
    assert_eq!(
        place(&mut trackers, core, placed),
        vec![
            (0, Some(v1::Placed::At(Bytes(1_625_344)))),
            (4, Some(v1::Placed::At(Bytes(1_641_728)))),
        ],
    );

    // ⛔ THE NEGATIVE CONTROL. Asking for the alias as a third block of the same footprint pushes
    // `ldsIdx_` 4 to 1,658,112 — 16,384 above where the reference put it.
    let mut with_alias = Trackers::at_step::<Dd2>(ExecutionStep::default());
    let mut all: Vec<Placed> = placed.iter().chain(aliases).copied().collect();
    // `ldsIdxAndAllocNode` is a `std::map<int, ..>` (`L3DlOpsScheduler.h:162`), so an alias that WAS
    // a candidate would be visited between `ldsIdx_` 0 and 4 rather than after them.
    all.sort_by_key(|&(lds, _, _)| lds);
    assert_eq!(
        place(&mut with_alias, core, &all),
        vec![
            (0, Some(v1::Placed::At(Bytes(1_625_344)))),
            (1, Some(v1::Placed::At(Bytes(1_641_728)))),
            (4, Some(v1::Placed::At(Bytes(1_658_112)))),
        ],
        "a third request moves lds4 off the reference's 1,641,728"
    );
}

/// ⭐⭐ A TRIAL PLACEMENT LEAVES NO TRACE — `allocAllMem` backs the tracker up, tries, and
/// `restoreEps`es every phase when it did not commit (`L3DlOpsScheduler.cpp:5539-5542`, `:5739-5742`),
/// so the NEXT attempt gets the same addresses rather than the ones above them.
///
/// ⛔ THE VALUES ARE THE WHOLE POINT: without the restore the second attempt would answer 1,633,536
/// and 1,634,048, which is what a tracker handing out an address twice-over looks like from the
/// caller's side.
#[test]
fn a_probe_that_is_restored_places_the_same_addresses_again() {
    let core = Core::checked(0).expect("core 0");
    let nodes: &[Placed] = &[(0, 4_096, 1_625_344), (1, 4_096, 1_629_440)];
    let first = vec![
        (0, Some(v1::Placed::At(Bytes(1_625_344)))),
        (1, Some(v1::Placed::At(Bytes(1_629_440)))),
    ];

    let mut trackers = Trackers::at_step::<Dd2>(ExecutionStep::default());
    assert_eq!(place(&mut trackers, core, nodes), first, "the probe");
    trackers.restore_all();
    assert_eq!(
        trackers
            .bundle
            .tracker(BundleSite::Lx(core))
            .free_cap_at_eps(ExPhase(0)),
        Capacity(406_272),
        "the restore gave every byte the probe took back"
    );
    assert_eq!(
        place(&mut trackers, core, nodes),
        first,
        "the same addresses, not the ones above them"
    );

    // ⛔ AND WITHOUT THE RESTORE THEY MOVE — the caller's `removeDs` of every candidate
    // (`:5608-5610`) is what keeps a retry from colliding with itself, so a repeat WITHOUT the
    // restore lands right back where it was.
    let mut committed = Trackers::at_step::<Dd2>(ExecutionStep::default());
    assert_eq!(place(&mut committed, core, nodes), first);
    assert_eq!(
        place(&mut committed, core, nodes),
        first,
        "`removeDs` first, so a re-place is not a second allocation"
    );
    // A DIFFERENT name on the same tracker is what stacks above it.
    assert_eq!(
        place(&mut committed, core, &[(2, 4_096, 0)]),
        vec![(2, Some(v1::Placed::At(Bytes(1_633_536))))],
        "1,625,344 + 2 x 4,096 — the two committed blocks are still there"
    );
}
