// SPDX-License-Identifier: Apache-2.0
//! WHAT ONE SLOT'S VALUE IS, AS BITS — the per-slot half of the encoder.
//!
//! `build.rs` writes the SHIFTS (each field's `bitPos` from `defineField`, resolved per component and instruction
//! type); these say what goes into them. Split that way because the shift is a fact about the instruction type and
//! the value is a fact about the slot, and the two come from different places in the C++.
//!
//! Every function here is total over its slot type, so a new variant is a build error rather than a zero.
//!
//! # What lives here and what lives in a submodule
//!
//! ⛔ ONE FIELD PER SLOT BELONGS HERE; A SLOT SPREAD ACROSS SEVERAL FIELDS DOES NOT. `mask`, `mode`, `unroll`,
//! `be`, the `unrlfld*` flags, the three sources and `tgtrf` are each one field, so their encoders take the value
//! and the numbers `build.rs` read out of that field's `encodeList` — the component and the arch enter as those
//! numbers and nowhere else.
//!
//! A compute's DESTINATION is spread. Which field a forward lands in is chosen from the DIRECTION
//! (`ConstructProgIRHelper.cpp:1319-1360`), so every direction the machine has is its own field and the ones this
//! compute does not use must still say `no` — that shape is [`direction`]. From SEN1P5 there is a second shape
//! beside it, the `tgtencoding` + `fwdencoding` pair (`:1361-1377`), which splits one forward across two fields
//! and is the only arch-gated part of the encoder: [`forward_pair`].

/// A destination field that IS a direction — its value names the output.
pub mod direction;
/// SEN1P5's `tgtencoding` + `fwdencoding` pair — one field for the unit, one for the output.
#[cfg(feature = "arch-sen1p5")]
pub mod forward_pair;

use crate::isa::fields::{ImmWidth, Sign};
use crate::isa::unit::{
    Be, BeOrUnstated, Mask, MaskOrUnstated, Precision, PrecisionOrUnstated, Unroll, UnrollField,
    UnrollFieldOrUnstated, UnrollOrUnstated,
};
use crate::islands::progir::{PhysicalTarget, PhysicalTargetOrUnstated};

// ⛔ A SOURCE SLOT'S ENCODE IS NOT HERE. Its port table, its constant table and its register base all belong to
// one FIELD — `defineField`'s encodeList for one (component, instruction type, position) — so `build.rs` writes
// that field's match out at the term that uses it. A shared encoder here could only receive those tables as
// arguments, which turns a constant into a parameter and the field's identity into a string.

/// A `tgtrf` target's register, or the field's own `no` when this compute writes none.
///
/// ⛔⛔ `no` IS A VALUE, NOT AN ABSENCE, AND IT IS NOT ZERO. `ConstructFMAInstr` closes every compute with
/// `if (!super_instr.hasCommonField("tgtrf")) setCommonField("tgtrf", "no")`
/// (`ConstructProgIRHelper.cpp:1586-1589`), and the field states `no` at 15 up to RCUDD1A (`isa.cpp:400`) and at 0
/// from SEN1P5 (`isa.cpp:482`). So the number is passed in from that field's own encodeList: a 0 written here
/// would name register 0 on RCUDD1A, and the instruction would clobber it.
/// ⛔⛔⛔ AND THE REGISTER CARRIES ITS FAMILY'S BASE, EXACTLY AS A SOURCE SLOT DOES. `tgtrf`'s encodeList is
/// `{{"Regs", numRegs}, …}` (`isa.cpp:600`, `:711`), so `R<i>` is `i + regOffset` here too — this field was
/// missed when the base was restored to the source slots because it has its own encoder.
///
/// ⛔ THE COST IS SILENT, NOT LOUD. `SFP::executePC` decides whether to run an instruction AT ALL with
/// `if (isReg(tgtrf) || is_any_of(3, tgtlx, tgtpe, tgtpt, tgtl0) || tgtdatafifo == 1 || …)`
/// — *"execute instruction iff the output is used"* (`computeElement.cpp:3589-3592`). `isReg` is
/// `s >= num_regs && s < num_regs * 2`, which a bare index never satisfies, so a compute whose only output is a
/// register looked to have NO output and the model SKIPPED it — every such instruction, silently.
/// ⛔ AND A SPELLED TARGET TAKES ITS NUMBER FROM THE SAME encodeList THE REGISTERS COME FROM. `tgtrf` carries a
/// `Regs` family and named entries together — `{{"Regs", numRegs}, {"irf0", 12}, {"xrf", 14}, {"no", 15}}`
/// (`isa.cpp:329`) — so `xrf` is looked up exactly like `no` is, per component and instruction type.
///
/// ⛔⛔ `spelled` IS `None` FOR A FIELD WITH NO SUCH ENTRY, AND REACHING IT IS A REFUSAL. Only the PT's `tgtrf`
/// states `xrf`; a PE or SFP target that arrived spelled would be an operand this crate bound to a file that
/// unit does not have, and writing `no` there would silently drop the write.
pub fn tgtrf_bits(
    target: PhysicalTargetOrUnstated,
    base: crate::isa::fields::RegBase,
    no: i64,
    spelled: Option<i64>,
) -> u64 {
    let reg = |reg: crate::islands::progir::RegIndex| u64::from(base.encoding_of(u32::from(reg.get())));
    match target {
        PhysicalTargetOrUnstated::Unstated => no as u64,
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Reg(held)) => reg(held),
        // ⭐ BOTH HALVES GO THROUGH THE SAME TWO CASES a plain write does — see [`Written`]. A `Both` whose
        // written half is a SPELLING is the PT storing into its XRF and forwarding that same store south
        // (`SNTransferLowering.cpp:2587-2604`); `tgtrf` takes the spelling and `tgts` the direction, two
        // FIELDS of one instruction.
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Both { writes, .. }) => match writes {
            crate::islands::progir::Written::Reg(held) => reg(held),
            crate::islands::progir::Written::Spelled(constant) => spelled_bits(constant, spelled),
        },
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Forward(_)) => no as u64,
        PhysicalTargetOrUnstated::Stated(PhysicalTarget::Spelled(constant)) => {
            spelled_bits(constant, spelled)
        }
    }
}

/// THE BITS A SPELLED FILE TAKES IN `tgtrf` — one place, because two arms of [`tgtrf_bits`] reach it.
///
/// ⛔ `None` MEANS THE FIELD HAS NO SUCH ENTRY, AND REACHING IT IS A REFUSAL. Only the PT's `tgtrf` states
/// `xrf`; a PE or SFP target that arrived spelled would be an operand this crate bound to a file that unit
/// does not have, and writing `no` there would silently drop the write.
fn spelled_bits(constant: crate::generated::OperandConstant, spelled: Option<i64>) -> u64 {
    let value = spelled.unwrap_or_else(|| {
        panic!(
            "a compute targets {constant:?}, and this `tgtrf` states no such entry in its encodeList — \
             only the PT's names `xrf` (`isa.cpp:329`). Writing `no` here would drop the write"
        )
    });
    match constant {
        crate::generated::OperandConstant::Xrf => value as u64,
        other => panic!(
            "a compute targets {other:?}; the only file `tgtrf` spells rather than indexes is the XRF \
             (`ConstructProgIRHelper.cpp:1333-1335`)"
        ),
    }
}

/// ⛔⛔ THE MASK FIELD IS THE COMPLEMENT OF THE STATED VALUE: `255 - value`
/// (`ConstructProgIRHelper.cpp:1497` and `:1631`). So the DDL's 255 — the default every compute takes
/// (`dsc2.h:916`, `compute_mask_ = 255`) — encodes as **0**, and writing the stated number straight into the field
/// would mask off every lane of every compute.
pub fn mask_bits(mask: MaskOrUnstated) -> u64 {
    match mask {
        MaskOrUnstated::Unstated => 0,
        MaskOrUnstated::M255 => stated(255),
        MaskOrUnstated::M1 => stated(1),
    }
}

/// THE MASK FIELD IS THE STATED NUMBER, NOT ITS COMPLEMENT.
///
/// ⛔⛔ THIS WAS `255 - value`, CITING `ConstructProgIRHelper.cpp:1497`, AND THAT CITATION IS REAL BUT INVERTS A
/// DIFFERENT QUANTITY. The C++ writes `setCommonField("mask", 255 - mask_constant.getValue())` where
/// `mask_constant` is `fma_op.getMask()` — the SENTIENT op's mask OPERAND, not the `.ddl`'s `mask=` attribute.
/// dxp's own ProgIR listing settles the direction: `SFP_FMA mask=255 mode=fp16 …` (`debug/sdsc_0/smc.txt` under
/// `CODEGEN_DUMP_IRS=1`) prints the field AFTER that subtraction, so `255 - mask_constant == 255` — the sentient
/// operand is 0 and the FIELD is 255. Subtracting the template's own 255 gave a field of 0, masked every lane off,
/// and `senulator` reported the resulting hang as a `QuiesceError`.
///
/// ⭐ VERIFIED FOR THE CASE THAT MATTERS: 523 of the 526 vendored `mask=` statements say 255 and dxp's field for
/// those is 255. `mask=1` (3 statements) follows the same identity and is NOT independently confirmed.
const fn stated(value: u8) -> u64 {
    lanes(value)
}

/// A COMPUTE INSTRUCTION'S MASK FIELD, WHICH MAY NOT BE ZERO.
///
/// ⛔⛔⛔ A ZERO MASK MASKS EVERY LANE OFF, so the unit consumes NOTHING from its input FIFO and then quiesces
/// with data still queued — a hang, not a wrong answer. IBM's own simulator names it exactly:
/// `senulator -p . -no-mem` on such an image raises
/// `UnitID( sfp:1:0 ) Type: QuiesceError, Msg: SFP quiesced, but still have data in LX-to-SFP FIFO with mask=0.`
/// (`senulator/computeElement.cpp:4673`).
///
/// ⭐ SO IT IS CHECKED ON THE ENCODED VALUE, NOT ON THE STATED ONE. The bug that produced it was a perfectly
/// well-stated `mask=255` turned into a zero FIELD, so a guard on the island-2 value would have passed.
const fn lanes(value: u8) -> u64 {
    assert!(
        value != 0,
        "a compute instruction's `mask` field encodes ZERO, which masks every lane off: the unit consumes \
         nothing from its input FIFO and then quiesces with data still queued. `senulator` reports it as \
         `QuiesceError ... with mask=0.` (`senulator/computeElement.cpp:4673`)"
    );
    value as u64
}

/// The same, for an opcode that states `mask=` on every one of its instructions.
pub const fn mask_bits_always(mask: Mask) -> u64 {
    match mask {
        Mask::M255 => stated(255),
        Mask::M1 => stated(1),
    }
}

/// ⛔⛔⛔ THE DEFAULT MASK ENCODES 255, AND THIS IS A COMPILE ERROR IF IT DOES NOT. 523 of the 526 vendored
/// `mask=` statements say 255, and dxp's own ProgIR listing prints `SFP_FMA mask=255` for that case
/// (`debug/sdsc_0/smc.txt` under `CODEGEN_DUMP_IRS=1`) — so the field for an all-lanes compute is 255, full stop.
///
/// ⭐ A `const` PROBE RATHER THAN A RUNTIME REFUSAL, because the value is a constant: `lanes` can only panic when
/// the encoder actually runs, and by then the build has already spent minutes. This fails at `cargo check`.
const _: () = assert!(
    mask_bits_always(Mask::M255) == 255,
    "the all-lanes mask must encode 255; a zero field masks every lane off and the unit quiesces with its input \
     FIFO still full (`senulator`'s `QuiesceError ... with mask=0.`)"
);

/// `mode=` — the compute precision. `{"fp16", 0}, {"fp32", 1}` (`isa.cpp`'s type-13 and type-118 SFP fields).
pub fn mode_bits(mode: PrecisionOrUnstated) -> u64 {
    match mode {
        PrecisionOrUnstated::Unstated => 0,
        PrecisionOrUnstated::Fp16 => 0,
        PrecisionOrUnstated::Fp32 => 1,
    }
}

/// The same, for an opcode that states `mode=` on every instruction.
pub fn mode_bits_always(mode: Precision) -> u64 {
    match mode {
        Precision::Fp16 => 0,
        Precision::Fp32 => 1,
    }
}

/// `unroll=` — `{"x1", 0}, {"x2", 1}, {"x4", 2}, {"x8", 3}` (the type-20 field, `isa.cpp:173`).
///
/// ⛔ THE VALUE IS THE LOG, NOT THE FACTOR. `x4` encodes as 2; writing 4 would name `x8`'s neighbour.
pub fn unroll_bits(unroll: UnrollOrUnstated) -> u64 {
    match unroll {
        UnrollOrUnstated::Unstated => 0,
        UnrollOrUnstated::X1 => 0,
        UnrollOrUnstated::X2 => 1,
        // ⭐ THE LOG, as this function's own note says: `x4` is 2 and `x8` is 3, straight off
        // `defineField(20, "unroll", 45, {{"x1",0},{"x2",1},{"x4",2},{"x8",3}})`.
        UnrollOrUnstated::X4 => 2,
        UnrollOrUnstated::X8 => 3,
        // ⛔⛔ `u0` IS A VARIABLE NAME, NOT A FACTOR — so there is no number to encode it as, and guessing 0
        // would silently make it `x1`.
        //
        // `SMCLineToInstrInfo` classifies a field value in three steps (`dpc.cpp:818-846`): in the field's
        // encodeList, it is `DESCRIPTIVE`; numeric, it is an INT or a FLOAT; otherwise it is a
        // `Type::VARIABLE` HOLDING THE NAME. `u0` is in the unroll field's list ({x1,x2,x4,x8}) under none of
        // those, and is not numeric, so it is the third.
        //
        // The binder is `ConstructProgIRHelper.cpp:3995-4010`: the name is looked up in the opaque's read
        // registers, then its read-write registers, then its `param` attributes, and a miss is
        // `"OPAQUE was not provided with value for variable "`. For `OperandT::unroll` alone the substituted
        // string is then given an `x` if it does not already start with one (`:4016-4019`), so a param of `2`
        // becomes `x2`.
        //
        // ⭐ SO PORTING IT IS A BIND, NOT AN ENCODING, and it is unreachable today: `u0` appears in exactly two
        // bodies (`pt_slice_mask_{arf,xrf}_write.smc`) and both are refused before this by
        // `UnmodelledSlot::Tgts`. Whoever models `tgts` reaches this next.
        UnrollOrUnstated::U0 => panic!(
            "`unroll=u0` names a VARIABLE, not a factor: `SMCLineToInstrInfo` makes a value that is neither in \
             the encodeList nor numeric a `Type::VARIABLE` holding the name (`dpc.cpp:818-846`), which \
             `ConstructProgIRHelper.cpp:3995-4019` resolves from the opaque's params and then prefixes with `x`. \
             Binding it is bridge 1's, not this encoder's"
        ),
    }
}

/// The same, for an opcode that states `unroll=` on every instruction.
pub fn unroll_bits_always(unroll: Unroll) -> u64 {
    match unroll {
        Unroll::X1 => 0,
        Unroll::X2 => 1,
        Unroll::X4 => 2,
        Unroll::X8 => 3,
        Unroll::U0 => panic!(
            "`unroll=u0` names a VARIABLE the opaque's params bind (`dpc.cpp:818-846`, \
             `ConstructProgIRHelper.cpp:3995-4019`), not a factor this encoder can spell"
        ),
    }
}

/// `unrlfld*` — `{"yes", 1}, {"no", 0}`.
pub fn unrlfld_bits(field: UnrollFieldOrUnstated) -> u64 {
    match field {
        UnrollFieldOrUnstated::Unstated => 0,
        UnrollFieldOrUnstated::Stated(UnrollField::No) => 0,
        UnrollFieldOrUnstated::Stated(UnrollField::Yes) => 1,
    }
}

/// The same, for an opcode that states the flag on every instruction.
pub fn unrlfld_bits_always(field: UnrollField) -> u64 {
    match field {
        UnrollField::No => 0,
        UnrollField::Yes => 1,
    }
}

/// `be` — `{"be", 1}`, the block end. Its absence is 0 legitimately: a field with one encodable value says only
/// whether it is set.
pub fn be_bits(be: BeOrUnstated) -> u64 {
    match be {
        BeOrUnstated::Unstated => 0,
        BeOrUnstated::Off => 0,
        BeOrUnstated::On => 1,
    }
}

/// The same, for an opcode that states `be=` on every instruction.
pub fn be_bits_always(be: Be) -> u64 {
    match be {
        Be::Off => 0,
        Be::On => 1,
    }
}

/// PLACE AN IMMEDIATE IN A FIELD OF `bits` BITS — the masking half of `defineField`'s immediate spec
/// (`isa.cpp:190-205`, read back by `getImmFieldStartLength` at `:81-92`), with the range check `immSizeValid`
/// makes (`progir.cpp:842-853`).
///
/// ⛔⛔ A NEGATIVE IMMEDIATE IS A REAL VALUE AND IT MUST NOT SIGN-EXTEND PAST ITS FIELD.
/// `createOffsetInstrDtTransfer` subtracts the level below's walk from an outer level's stride
/// (`dcgbeCodegen.cpp:4685-4690`), so an `ADDEARIMM` that steps back is ordinary — but placed by `as u64` it sets
/// every bit above the field, overwriting whatever fields sit there and widening the word past its stream.
///
/// ⭐ THE GUARD IS THE FIELD'S OWN SIGNEDNESS, which the field STATES — [`ImmSpec::Imm`] carries a width and a
/// [`Sign`], both from `defineField`. SIGNED admits `[-2^(n-1), 2^(n-1))`, UNSIGNED `[0, 2^n)`, and
/// MODULO_UNSIGNED `[-2^n, 2^n]` — three arms, and the third is why the value is MASKED rather than asserted to
/// fit: `2^n` is in range and encodes as 0.
///
/// ⛔ THE THREE ARMS BELONG TO THE ISA, SO THEY ARE WRITTEN ONCE. `immSizeValid` is one rule over a width and a
/// signedness; a caller that assembled the comparison itself would be re-porting `progir.cpp:842-853` per field.
/// A field passes only what `defineField` gave it.
///
/// ⛔ AND A VALUE OUT OF RANGE IS A BUILD ERROR, not a truncation. dxp answers `ErrorType::IMMEDIATE`
/// (`progir.cpp:1292-1297`); here the value came from extents scratchy stated, so it is a tiling this crate
/// cannot serve rather than a datum it can round.
pub fn imm_bits(value: i64, width: ImmWidth, sign: Sign) -> u64 {
    let bits = width.get();
    let value = i128::from(value);
    // `immSizeValid` (`progir.cpp:842-853`), whose three arms are the three `Signedness` cases.
    let in_range = match sign {
        Sign::Signed => -(1i128 << (bits - 1)) <= value && value < (1i128 << (bits - 1)),
        Sign::Unsigned => 0 <= value && value < (1i128 << bits),
        Sign::ModuloUnsigned => -(1i128 << bits) <= value && value <= (1i128 << bits),
    };
    assert!(
        in_range,
        "an immediate of {value} does not fit its {bits}-bit {sign:?} field; dxp answers `ErrorType::IMMEDIATE` \
         for that (`immSizeValid`, `progir.cpp:842-853`, checked at `:1292`)"
    );
    // The field's own width is the modulus — two's complement for a negative value, identity for a positive one.
    let mask = match bits >= 64 {
        true => u64::MAX,
        false => (1u64 << bits) - 1,
    };
    (value as u64) & mask
}
