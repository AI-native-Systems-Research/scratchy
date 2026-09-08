// SPDX-License-Identifier: Apache-2.0
//! THE `tgtencoding` + `fwdencoding` PAIR — SEN1P5's second way to say where a result goes.
//!
//! From SEN1P5 on, `setSentientComputeOutputProgIROperands` writes two fields for a forward to `pe`/`pt`/`l0`/
//! `sfp`/`lx` (`ConstructProgIRHelper.cpp:1361-1377`): `tgtencoding` names the UNIT and `fwdencoding` names WHICH
//! output goes there. The fields are `isa.cpp:479` (`{lx:1, sfp:2}` on the PE, `{l0:0, lx:1, pe:2, pt:3}` on the
//! SFP) and `isa.cpp:478` (`{no:0, src0:1, src2:2, result:3}`).
//!
//! ⛔ IT DOES NOT REPLACE THE DIRECTION FIELDS, IT JOINS THEM. The same SEN1P5 PE type states `tgtpe`
//! (`isa.cpp:477`), so [`super::direction`] is live on this arch too. Only this pair is arch-gated: `defineField`
//! states it in the `else` of `coreArch <= RCUDD1A_ISA` and nowhere else.

use crate::islands::progir::{Forward, FwdEncoding};
use crate::islands::progir::{PhysicalTarget, PhysicalTargetOrUnstated};

/// A bare [`Forward`] as the wrapper the two encoders take — for the opcodes that state `tgtencoding` on every
/// instruction and no `tgtrf`, whose target slot is the forward itself.
fn stated(forward: Forward) -> PhysicalTargetOrUnstated {
    PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(forward))
}

/// `tgtencoding` — WHICH UNIT the result leaves by.
///
/// ⛔ THE TWO NUMBERS ARE THIS FIELD'S OWN, AND ONE OF THEM MAY BE ABSENT: the PE states `{lx, sfp}` and the SFP
/// states `{l0, lx, pe, pt}`, because "tgtencoding can't be itself" (`ConstructProgIRHelper.cpp:1366`). So a via
/// this field cannot spell is a refusal, never the other one's number.
///
/// ⛔ AND NOT FORWARDING IS ZERO BITS. The field states no `no`, so the C++ leaves it unset and the encoder ORs
/// nothing in (`dpc.cpp:1516`) — inert only because `fwdencoding` says `no` alongside it.
pub fn tgtencoding_bits(
    target: PhysicalTargetOrUnstated,
    lx: Option<i64>,
    sfp: Option<i64>,
) -> u64 {
    let forward = match target {
        PhysicalTargetOrUnstated::Unstated => return 0,
        // A register-only target forwards nowhere — and neither does a SPELLED one, which is `tgtrf` naming a
        // file (the PT's XRF) rather than indexing one.
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Reg(_) | PhysicalTarget::Spelled(_)) => {
            return 0;
        }
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(forward))
        | PhysicalTargetOrUnstated::Stated(PhysicalTarget::Both { forward, .. }) => forward,
    };
    let (stated, spelling) = match forward.via {
        TgtEncoding::Lx => (lx, "lx"),
        TgtEncoding::Sfp => (sfp, "sfp"),
        // ⛔ THE PAIR IS PE/SFP-ONLY, AND THE PT'S SOUTH LINK IS NOT IN IT. `tgtencoding` states `{lx, sfp}` on the
        // PE and `{l0, lx, pe, pt}` on the SFP (`isa.cpp:479`, `:708`) — no `south`, because
        // `setSentientComputeOutputProgIROperands` reaches this pair only in the PE/SFP branch and handles the PT's
        // `tgte`/`tgts` before it (`ConstructProgIRHelper.cpp:1319-1330` against `:1361-1377`).
        TgtEncoding::South => panic!(
            "a `south` forward is the PT's own `tgts` field, not this pair: `tgtencoding` states `{{lx, sfp}}` on \
             the PE and `{{l0, lx, pe, pt}}` on the SFP (`isa.cpp:479`, `:708`), and the PT never reaches this \
             branch (`ConstructProgIRHelper.cpp:1319-1330`)"
        ),
    };
    match stated {
        Some(value) => value as u64,
        None => panic!(
            "`tgtencoding` does not state `{spelling}` on this component, so this compute cannot forward that way \
             — the DDL chose a direction the field has no encoding for"
        ),
    }
}

/// `fwdencoding` — WHICH OUTPUT is forwarded.
pub fn fwdencoding_bits(target: PhysicalTargetOrUnstated, no: i64, result: i64) -> u64 {
    let forward = match target {
        PhysicalTargetOrUnstated::Unstated => return no as u64,
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Reg(_) | PhysicalTarget::Spelled(_)) => {
            return no as u64;
        }
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(forward))
        | PhysicalTargetOrUnstated::Stated(PhysicalTarget::Both { forward, .. }) => forward,
    };
    match forward.to {
        FwdEncoding::Result => result as u64,
        FwdEncoding::Out0 => panic!(
            "`fwdencoding=out0` is a template HOLE, closed at build time from the statement's `params[\"out0\"]` — \
             reaching it in the encoder means the `.smc` splice emitted a hole instead of filling it"
        ),
    }
}

/// [`tgtencoding_bits`] for an opcode whose target slot is a bare [`Forward`].
pub fn tgtencoding_bits_forward(forward: Forward, lx: Option<i64>, sfp: Option<i64>) -> u64 {
    tgtencoding_bits(stated(forward), lx, sfp)
}

/// [`fwdencoding_bits`] for an opcode whose target slot is a bare [`Forward`].
pub fn fwdencoding_bits_forward(forward: Forward, no: i64, result: i64) -> u64 {
    fwdencoding_bits(stated(forward), no, result)
}
