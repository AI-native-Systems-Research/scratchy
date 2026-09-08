//! THE SHARED HELPERS — the on-the-fly conversion check, the proper-consumer walk, the address
//! wraparound and the fold-mode value.
//!
//! ⛔ `verifyOnTheFlyConversions` IS A CHECK — see `reg_def_tracker.rs`'s note. Return the
//! offenders, never refuse.
//!
//! 5 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e038_verifyOnTheFlyConversions` | 0 | 56 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:27` |
//! | `e039_getProperConsumer` | 0 | 14 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:86` |
//! | `e040_updateProperConsumer` | 0 | 14 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:102` |
//! | `e041_getAddrWraparounded` | 0 | 10 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:117` |
//! | `e080_setFCValueFromFoldMode` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:128` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::{Arch, IsaGen};
use crate::islands::progir::ty::{FoldId, Operand, OperandValue};
use crate::islands::sentient::dialects::sentient::Precision;

/// WHICH COMPUTE UNIT AN FMA/FMUL/FNMS RUNS ON — the two `is_any_of(comp, PE, SFP)` admits at every
/// callsite (`ConstructProgIRHelper.cpp:1449`, `:1795`, `:1804`).
///
/// ⛔ THE PT IS NOT ONE OF THEM: `ConstructBinaryInstr` errors out for it (`:1815`), so there is no
/// on-the-fly conversion question to ask about the matrix unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnit {
    /// `PE`.
    Pe,
    /// `SFP`.
    Sfp,
}

/// WHICH OF THE THREE OPERANDS — `opA`, `opB` or `opC`, which is what `src0_operand_idx`'s 0/1/2
/// names (`Utils.hpp:31-36`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrcOperand {
    /// `opA`, `src0_operand_idx == 0`.
    OpA,
    /// `opB`, `== 1`.
    OpB,
    /// `opC`, `== 2`.
    OpC,
}

/// THE THREE INPUT PRECISIONS OF ONE FMA — named so two of them cannot be swapped by position
/// (`ConstructProgIRHelper.cpp:1427-1432`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputPrecisions {
    /// `opAPrecision`.
    pub op_a: Precision,
    /// `opBPrecision`.
    pub op_b: Precision,
    /// `opCPrecision` — ⭐ THE COMPUTE PRECISION AT THE TWO BINARY CALLSITES, which have no third
    /// operand and pass it deliberately (`:1797`, `:1807`), so the opC test never fires there.
    pub op_c: Precision,
}

/// A PRECISION COMBINATION THE COMPUTE UNITS HAVE NO ON-THE-FLY CONVERSION FOR — one of
/// `verifyOnTheFlyConversions`' four refusals (`Utils.cpp:27-82`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedConversion {
    /// *"Unsupported result precision conversion in DD2"* (`:33-35`).
    ResultOnRcudd1a {
        /// `$ComputePrecision`.
        compute: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// *"Unsupported input on the fly conversion for FMA/FMUL/FNMS"* (`:39-49`).
    Input {
        /// Which of `opA`/`opB`/`opC`.
        operand: SrcOperand,
        /// What it arrives in.
        precision: Precision,
        /// What the unit computes in.
        compute: Precision,
    },
    /// *"Unsupported compute precision for FMA/FMUL/FNMS"* (`:72-74`).
    ComputePrecision {
        /// What the unit computes in.
        compute: Precision,
    },
    /// *"Unsupported output on the fly conversion for FMA/FMUL/FNMS"* (`:75-82`).
    Output {
        /// What the unit computes in.
        compute: Precision,
        /// What it is asked to write.
        result: Precision,
    },
}

/// Replaces: e038_verifyOnTheFlyConversions
///
/// Every precision of one FMA/FMUL/FNMS the hardware cannot convert on the fly — ⛔ THE OFFENDERS,
/// never a refusal, and never a bool.
/// ⛔ `src0` IS AN `Option` BECAUSE THE REFERENCE'S SENTINEL IS `-1`: no operand carrying port 0
/// leaves it there (`ConstructProgIRHelper.cpp:1442-1448`), which fails every `== N` test below.
/// ⛔ AND A 24-BIT INPUT IS SRC0-ON-THE-PE ONLY, ON ITS OWN INDEX — opA needs `src0 == 0`, opB
/// `== 1`, opC `== 2`, each for itself, so the three arms are not one test on `src0`.
#[must_use]
pub fn unsupported_on_the_fly_conversions<A: Arch>(
    comp: ComputeUnit,
    inputs: InputPrecisions,
    src0: Option<SrcOperand>,
    compute: Precision,
    result: Precision,
) -> Vec<UnsupportedConversion> {
    let mut offenders = Vec::new();
    if matches!(A::GEN, IsaGen::Rcudd1a) {
        // `:32-36` — DD2 converts a result on the way out only by narrowing it to an integer.
        if result != compute && !matches!(result, Precision::Int4 | Precision::Int8) {
            offenders.push(UnsupportedConversion::ResultOnRcudd1a { compute, result });
        }
        return offenders;
    }
    // `:37-62` — Sentient 1.5, one arm per input operand, in operand order.
    for (operand, precision) in [
        (SrcOperand::OpA, inputs.op_a),
        (SrcOperand::OpB, inputs.op_b),
        (SrcOperand::OpC, inputs.op_c),
    ] {
        if precision == compute {
            continue;
        }
        let converts = if matches!(precision, Precision::Fp24 | Precision::Int24) {
            src0 == Some(operand) && comp == ComputeUnit::Pe
        } else {
            precision == Precision::Fp16 && compute == Precision::Fp32
        };
        if !converts {
            offenders.push(UnsupportedConversion::Input {
                operand,
                precision,
                compute,
            });
        }
    }
    if compute != result {
        // `:71-74` — the compute precision itself, checked before what it converts to.
        if !matches!(compute, Precision::Fp16 | Precision::Fp32) {
            offenders.push(UnsupportedConversion::ComputePrecision { compute });
        }
        // `:75-82` — a half result must come out of an fp32 compute; anything else must be an
        // integer narrowing.
        let converts = if matches!(result, Precision::Fp16 | Precision::Bf16) {
            compute == Precision::Fp32
        } else {
            matches!(result, Precision::Int4 | Precision::Int8)
        };
        if !converts {
            offenders.push(UnsupportedConversion::Output { compute, result });
        }
    }
    offenders
}

/// WHICH UNIT CONSUMES A TRANSFER — the four `updateProperConsumer`'s DT_CHECK admits
/// (`Utils.cpp:104`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumerUnit {
    /// `PE`.
    Pe,
    /// `PT`.
    Pt,
    /// `SFP`.
    Sfp,
    /// `L0SU`.
    L0su,
}

impl ConsumerUnit {
    /// The `consumertag` this unit is named by — `senComponentsToString` over its generic component
    /// (`sys-arch-spec/arch_enums.cpp:20-23`, `:16`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            ConsumerUnit::Pe => "pe",
            ConsumerUnit::Pt => "pt",
            ConsumerUnit::Sfp => "sfp",
            ConsumerUnit::L0su => "l0su",
        }
    }
}

/// WHAT A LOAD'S `consumertag` MAY NAME — a [`ConsumerUnit`], or the cross-partition link.
///
/// ⛔⛔ THE TWO PATHS ADMIT DIFFERENT SETS, AND THAT IS WHY THIS TYPE IS SEPARATE.
/// `getProperConsumer` accepts the link and folds it into the SFP on SEN1P5 (`Utils.cpp:90`);
/// `updateProperConsumer`'s DT_CHECK does not accept it on either generation (`:104`), so the
/// per-fold path takes the four-set and a link there is an E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadConsumer {
    /// One of the four.
    Unit(ConsumerUnit),
    /// `CROSSPTNLINK` — the link out of this partition, spelled `"crossptnlink"`
    /// (`sys-arch-spec/arch_enums.cpp:30`).
    CrossPtnLink,
}

impl LoadConsumer {
    /// Its `consumertag` spelling — see [`ConsumerUnit::spelling`].
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            LoadConsumer::Unit(unit) => unit.spelling(),
            LoadConsumer::CrossPtnLink => "crossptnlink",
        }
    }
}

/// Replaces: e039_getProperConsumer
///
/// The `consumertag` a load writes: the PT and the L0SU take their data THROUGH the SFP, and so does
/// the cross-partition link from SEN1P5 — every other consumer names itself.
/// ⛔ RCUDD1A HAS NO CROSS LINK. The reference's fallthrough DT_CHECKs `{PE, SFP}` (`:94`), so a link
/// there is an abort; it names itself here, and no RCUDD1A program binds one to ask about.
/// ⛔ THE `< RCUDD1A` ARM IS MPW4's, a generation [`IsaGen`] deliberately does not model.
#[must_use]
pub fn proper_consumer<A: Arch>(consumer: LoadConsumer) -> Operand {
    let tag = match consumer {
        // `:89` — reached through the SFP, so the tag names the SFP.
        LoadConsumer::Unit(ConsumerUnit::Pt | ConsumerUnit::L0su) => ConsumerUnit::Sfp.spelling(),
        // `:90` — and so is the link, from SEN1P5.
        LoadConsumer::CrossPtnLink => match A::GEN {
            IsaGen::Sen1p5 => ConsumerUnit::Sfp.spelling(),
            IsaGen::Rcudd1a => consumer.spelling(),
        },
        LoadConsumer::Unit(unit) => unit.spelling(),
    };
    Operand::every(OperandValue::Descriptive(tag.to_owned()))
}

/// Replaces: e040_updateProperConsumer
///
/// The same substitution, written into an operand that may already hold one value per fold —
/// `addEntryToOperandMap`'s `unit_name` mode (`UniformInstrAndBlock.cpp:126-132`).
/// ⛔ NO CROSS LINK HERE AT ALL — see [`LoadConsumer`], and note that this makes the arch irrelevant:
/// both generations this crate models are `>= RCUDD1A`, and the four-set's rule is the same on each.
/// ⛔ AND A COMMON WRITE THROWS AN EXISTING PER-FOLD MAP AWAY — that is [`Operand::set`]'s asymmetry,
/// which the reference's `fold_id ? id : std::nullopt` (`:130`) reaches deliberately.
pub fn update_proper_consumer(value: &mut Operand, consumer: ConsumerUnit, fold: Option<FoldId>) {
    let tag = match consumer {
        // `:106-108`.
        ConsumerUnit::Pt | ConsumerUnit::L0su => ConsumerUnit::Sfp,
        ConsumerUnit::Pe | ConsumerUnit::Sfp => consumer,
    };
    value.set(fold, OperandValue::Descriptive(tag.spelling().to_owned()));
}

// crustify:todo: e041_getAddrWraparounded
// crustify:todo: e080_setFCValueFromFoldMode

#[cfg(test)]
mod unit_tests {
    use super::{
        ComputeUnit, ConsumerUnit, InputPrecisions, LoadConsumer, SrcOperand,
        UnsupportedConversion, proper_consumer, unsupported_on_the_fly_conversions,
        update_proper_consumer,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::progir::ty::{FoldId, Operand, OperandValue, PerFold};
    use crate::islands::sentient::dialects::sentient::Precision;

    fn descriptive(name: &str) -> Operand {
        Operand::every(OperandValue::Descriptive(name.to_owned()))
    }

    /// e038: DD2 narrows a result only to an integer; SEN1P5 widens an fp16 input into an fp32
    /// compute, takes a 24-bit input only as src0 on the PE, and names every other combination.
    #[test]
    fn the_unsupported_conversions_come_back_as_offenders() {
        let fp16 = InputPrecisions {
            op_a: Precision::Fp16,
            op_b: Precision::Fp16,
            op_c: Precision::Fp16,
        };
        // `:32-36` — DD2, where int8 out of an fp16 compute is the one conversion allowed.
        assert!(
            unsupported_on_the_fly_conversions::<Dd2>(
                ComputeUnit::Pe,
                fp16,
                None,
                Precision::Fp16,
                Precision::Int8
            )
            .is_empty()
        );
        assert_eq!(
            unsupported_on_the_fly_conversions::<Dd2>(
                ComputeUnit::Pe,
                fp16,
                None,
                Precision::Fp16,
                Precision::Bf16
            ),
            vec![UnsupportedConversion::ResultOnRcudd1a {
                compute: Precision::Fp16,
                result: Precision::Bf16,
            }]
        );
        // SEN1P5: three fp16 inputs into an fp32 compute is the widening path, and it is clean.
        assert!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Sfp,
                fp16,
                None,
                Precision::Fp32,
                Precision::Fp32
            )
            .is_empty()
        );
        // ⛔ A 24-BIT INPUT IS SRC0-ON-THE-PE ONLY: the same operand on the SFP is an offender.
        let fp24_b = InputPrecisions {
            op_b: Precision::Fp24,
            ..fp16
        };
        assert!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Pe,
                fp24_b,
                Some(SrcOperand::OpB),
                Precision::Fp32,
                Precision::Fp32
            )
            .is_empty()
        );
        assert_eq!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Sfp,
                fp24_b,
                Some(SrcOperand::OpB),
                Precision::Fp32,
                Precision::Fp32
            ),
            vec![UnsupportedConversion::Input {
                operand: SrcOperand::OpB,
                precision: Precision::Fp24,
                compute: Precision::Fp32,
            }]
        );
        // ⛔ AND ON ANOTHER OPERAND'S INDEX IT IS STILL ONE: `src0 == 0` does not license opB.
        assert_eq!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Pe,
                fp24_b,
                Some(SrcOperand::OpA),
                Precision::Fp32,
                Precision::Fp32
            ),
            vec![UnsupportedConversion::Input {
                operand: SrcOperand::OpB,
                precision: Precision::Fp24,
                compute: Precision::Fp32,
            }]
        );
        // `:71-74` — the compute precision itself, with every input already in it.
        let fp8 = InputPrecisions {
            op_a: Precision::Fp8,
            op_b: Precision::Fp8,
            op_c: Precision::Fp8,
        };
        assert_eq!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Pe,
                fp8,
                None,
                Precision::Fp8,
                Precision::Int8
            ),
            vec![UnsupportedConversion::ComputePrecision {
                compute: Precision::Fp8
            }]
        );
        // `:75-82` — a bf16 result needs an fp32 compute, not an fp16 one.
        assert_eq!(
            unsupported_on_the_fly_conversions::<Sen1p5>(
                ComputeUnit::Pe,
                fp16,
                None,
                Precision::Fp16,
                Precision::Bf16
            ),
            vec![UnsupportedConversion::Output {
                compute: Precision::Fp16,
                result: Precision::Bf16,
            }]
        );
    }

    /// e039: the PT and the L0SU are tagged as the SFP; so is the cross link, but only from SEN1P5.
    #[test]
    fn the_proper_consumer_routes_the_pt_the_l0su_and_the_cross_link_through_the_sfp() {
        for consumer in [ConsumerUnit::Pt, ConsumerUnit::L0su] {
            assert_eq!(
                proper_consumer::<Sen1p5>(LoadConsumer::Unit(consumer)),
                descriptive("sfp")
            );
            assert_eq!(
                proper_consumer::<Dd2>(LoadConsumer::Unit(consumer)),
                descriptive("sfp")
            );
        }
        assert_eq!(
            proper_consumer::<Sen1p5>(LoadConsumer::CrossPtnLink),
            descriptive("sfp")
        );
        // ⛔ RCUDD1A HAS NO CROSS LINK, and the reference aborts rather than answering.
        assert_eq!(
            proper_consumer::<Dd2>(LoadConsumer::CrossPtnLink),
            descriptive("crossptnlink")
        );
        // The two that name themselves.
        assert_eq!(
            proper_consumer::<Sen1p5>(LoadConsumer::Unit(ConsumerUnit::Pe)),
            descriptive("pe")
        );
        assert_eq!(
            proper_consumer::<Sen1p5>(LoadConsumer::Unit(ConsumerUnit::Sfp)),
            descriptive("sfp")
        );
    }

    /// e040: the same substitution merged per fold — and a common write afterwards discards the map.
    #[test]
    fn updating_the_proper_consumer_merges_one_tag_per_fold() {
        let mut value = Operand::default();
        update_proper_consumer(&mut value, ConsumerUnit::Pt, Some(FoldId(1)));
        update_proper_consumer(&mut value, ConsumerUnit::Pe, Some(FoldId(2)));
        assert_eq!(
            value.value,
            PerFold::ByFold(vec![
                (FoldId(1), OperandValue::Descriptive("sfp".to_owned())),
                (FoldId(2), OperandValue::Descriptive("pe".to_owned())),
            ])
        );
        update_proper_consumer(&mut value, ConsumerUnit::L0su, None);
        assert_eq!(
            value.value,
            PerFold::Every(OperandValue::Descriptive("sfp".to_owned()))
        );
    }
}
