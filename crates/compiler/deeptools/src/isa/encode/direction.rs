// SPDX-License-Identifier: Apache-2.0
//! A DESTINATION FIELD THAT *IS* A DIRECTION — its value says which of the compute's outputs travels that way.
//!
//! `setSentientComputeOutputProgIROperands` picks the field FROM the direction (`ConstructProgIRHelper.cpp:1319-1360`):
//! on the PT `"east"`/`"south"` give `tgte`/`tgts`, and on the PE/SFP up to RCUDD1A `"lx"` gives `tgtlx`, `"sfp"`
//! gives `tgtsfp`, and `"pe"`/`"pt"`/`"l0"`/`"sfpring"` give `tgtpe`/`tgtpt`/`tgtl0`/`tgtdatafifo`. The list in each
//! is which output goes there — `{no:0, src0:1, src2:2, result:3}` (`isa.cpp:398`).
//!
//! ⛔ A DIRECTION FIELD IS NOT ONE CHIP'S SHAPE, so this module is not arch-gated. The SEN1P5 PE's type 20 states
//! `tgtpe` — a direction field, narrowed to `{no:0, result:1}` — right beside `tgtencoding` and `fwdencoding`
//! (`isa.cpp:477-479`), so both shapes are live on that arch and only the pair itself is gated. What separates the
//! two modules is what a field MEANS: see [`super::forward_pair`].

use crate::isa::unit::TgtEncoding;
use crate::islands::progir::{Forward, FwdEncoding};
use crate::islands::progir::{PhysicalTarget, PhysicalTargetOrUnstated};

/// WHAT ONE DIRECTION FIELD CARRIES — this compute's output if it forwards this way, and the field's own `no` if
/// it does not.
///
/// ⛔ THE NUMBERS ARE THE FIELD'S, PASSED IN. `build.rs` reads them out of that field's own `encodeList` for the
/// component and instruction type being emitted; the sets differ per component, so a table written here would be
/// right on one and wrong on the other.
///
/// ⛔ AND A FORWARD TO A DIFFERENT UNIT WRITES THIS FIELD'S `no`, not nothing: each direction is its own field, so
/// the ones this compute does not use must still say so.
/// THE FOUR VALUES ONE DIRECTION FIELD'S LIST STATES — `{no: 0, src0: 1, src2: 2, result: 3}`
/// (`isa.cpp:398`).
///
/// ⛔ A STRUCT BECAUSE FOUR ADJACENT `i64`s IS FOUR SWAPS. They are all small integers from one encode
/// list, and a transposed pair encodes a DIFFERENT OUTPUT down the same wire — an instruction that is
/// well-formed and forwards the wrong operand. It was two adjacent ones before this and the third and
/// fourth are what made it untenable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectionValues {
    /// `no` — this compute does not forward this way.
    pub no: i64,
    /// `src0` — it forwards its FIRST input, unchanged.
    pub src0: i64,
    /// `src2` — it forwards its third input.
    pub src2: i64,
    /// `result` — it forwards what it computed.
    pub result: i64,
}

pub fn direction_bits(
    target: PhysicalTargetOrUnstated,
    this_field: TgtEncoding,
    values: DirectionValues,
) -> u64 {
    let DirectionValues {
        no,
        src0,
        src2,
        result,
    } = values;
    let forward = match target {
        PhysicalTargetOrUnstated::Unstated => return no as u64,
        // A register-only target forwards nowhere — and neither does a SPELLED one, which is `tgtrf` naming a
        // file rather than indexing one (`ConstructProgIRHelper.cpp:1333-1335` routes `xrf` to `tgtrf`).
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Reg(_) | PhysicalTarget::Spelled(_)) => {
            return no as u64;
        }
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(forward))
        | PhysicalTargetOrUnstated::Stated(PhysicalTarget::Both { forward, .. }) => forward,
    };
    // ⭐ THIS FIELD CARRIES THE OUTPUT ONLY IF IT IS THIS COMPUTE'S DIRECTION. Every other direction field writes
    // its own `no`, which is why all of them are emitted rather than only the one used:
    // `setSentientComputeOutputProgIROperands` sets one field per output (`ConstructProgIRHelper.cpp:1331-1360`) and
    // leaves the rest to the closing default.
    //
    // ⛔ THE TEST IS EQUALITY, NOT A CROSS PRODUCT. It was written as one arm per (via, field) pair, which is
    // total but quadratic — and a fourth direction turns 3 pairs into 12, every one of which says only "these two
    // are the same" or "these two differ". One equality states the rule the comment above already gives, and it
    // stays total when the set grows.
    match forward.via == this_field {
        true => {
            match forward.to {
                FwdEncoding::Result => result as u64,
                // 🔑 THE OPERAND ITSELF, PASSED THROUGH — see [`FwdEncoding::Src0`].
                FwdEncoding::Src0 => src0 as u64,
                FwdEncoding::Src2 => src2 as u64,
                // ⭐ `out0` IS A HOLE THE TEMPLATE FILLS, AND `build.rs` CLOSES IT AT PARSE. A body writes
                // `fwdencoding=out0` and its invoking statement says `params={"out0"="result"}` or `="no"`, so the
                // splice resolves it to a forward or removes the half. The variant survives only because the
                // slot's value set is censused from the template SPELLINGS.
                FwdEncoding::Out0 => panic!(
                    "`fwdencoding=out0` is a template HOLE, closed at build time from the statement's \
                     `params[\"out0\"]` — reaching it in the encoder means the `.smc` splice emitted a hole \
                     instead of filling it"
                ),
            }
        }
        // Forwarding some other way: this direction carries nothing.
        false => no as u64,
    }
}

/// The same, for an opcode whose target is a bare [`Forward`] — a `tgt<direction>` on every instruction and no
/// `tgtrf`.
pub fn direction_bits_forward(
    forward: Forward,
    this_field: TgtEncoding,
    values: DirectionValues,
) -> u64 {
    direction_bits(
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(forward)),
        this_field,
        values,
    )
}
