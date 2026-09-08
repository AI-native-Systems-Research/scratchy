//! HOW WIDE ONE ELEMENT OF EACH FORMAT IS.
//!
//! ⭐ IBM'S OWN TABLE, TRANSCRIBED: `EnumsConversion::dataFormatsToBitWidth`
//! (`util/sendefs/sendefs.cpp:129-141`). It is the map `dataType.h:16` reads to answer the same
//! question, so it is the source of truth rather than a convention.
//!
//! ⛔⛔ IN BITS, NOT BYTES, BECAUSE TWO OF THE FORMATS ARE SUB-BYTE. `SENINT4` and `SEN121_FP4` are
//! four bits each; a `bytes_per_element` returning a whole number would have to answer 0 or 1 for
//! them, and both are wrong. Everything downstream that wants bytes divides, and the division is
//! visible where it happens.

use crate::generated::DataType;

/// A WIDTH IN BITS.
///
/// ⛔ A NEWTYPE BECAUSE THE UNIT IS THE WHOLE QUESTION HERE. `bit_width=8` also appears in the
/// templates as a PACKING width and means something else entirely — see [`DataType::BITS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bits(pub u32);

impl DataType {
    /// HOW MANY BITS ONE ELEMENT OCCUPIES.
    ///
    /// ⛔⛔ `Senint24` IS SIXTEEN BITS, NOT TWENTY-FOUR. `{DataFormats::SENINT24, 16}`
    /// (`sendefs.cpp:135`), sitting between `SENINT16, 16` and `IEEE_INT64, 64` where a reader
    /// scanning for the pattern would supply 24 without noticing. Deriving this width from the
    /// name — which is what "SENINT24" invites — gets it wrong by 50%, and every address computed
    /// from it lands in the wrong place.
    ///
    /// ⛔ AND THIS IS NOT THE TEMPLATES' `bit_width=`. That attribute is a PACKING width: the
    /// vendored templates carry both `{data_type="SENINT4", bit_width=16}` and
    /// `{data_type="SENINT4", bit_width=8}` — one data type, two `bit_width`s — so it cannot be
    /// the element width and must not be read as one.
    #[must_use]
    pub const fn bits(self) -> Bits {
        Bits(match self {
            Self::Sen169Fp16 | Self::Bfloat16 => 16,
            Self::IeeeFp32 | Self::Senuint32 => 32,
            Self::Sen143Fp8 | Self::Sen080Fp8 | Self::Sen053Fp8 | Self::Senint8 | Self::Bool => 8,
            Self::Sen121Fp4 | Self::Senint4 => 4,
            // ⛔ SIXTEEN. See above.
            Self::Senint24 => 16,
        })
    }

    /// HOW MANY ELEMENTS OF THIS FORMAT FILL A STICK OF `stick_bits` BITS.
    ///
    /// ⭐ THIS IS THE ARITHMETIC THAT MAKES fp8 LOOK LIKE 128 LANES, AND IT IS NOT THE LANE COUNT.
    /// A 128-byte stick holds 128 fp8 elements, but how many a transfer moves per time step is
    /// `getVectorLanes`, which reads the arch's declared SIMD map and answers ONE for a format the
    /// device does not list. Packing and lanes are different questions; see
    /// [`crate::bridges::subtile_to_dataflow_ir::transfer::Lanes`].
    #[must_use]
    pub const fn per_stick(self, stick_bits: u32) -> u32 {
        stick_bits / self.bits().0
    }
}

/// WHICH FORMAT A VALUE IS ON THE WIRE — `DataFormats` (`util/sendefs/sendefs.h:22-40`).
///
/// ⛔⛔ NOT [`DataType`], AND THE TWO ARE EASY TO MISTAKE FOR ONE ENUM. [`DataType`] is censused by
/// `build.rs` from the formats the `.ddl` templates actually name; this is `util/sendefs`' own closed
/// set, which is what `OperandAttr::senDataType_` records (`progir.h:271`) and what
/// `addPESFPLRFImmcopyToRegInit` dispatches its bit packing on. This one has `SENINT2` and
/// `IEEE_INT32`, which the census does not; the census has `Bfloat16`, two more fp8 spellings and
/// `Sen121Fp4`, which this does not. Neither is a superset, so neither can stand in for the other.
///
/// ⛔ `INVALID` IS NOT A VARIANT. It is what `hasSenDataType` tests for (`progir.h:119`), so it is
/// `Option::None` at the use site and cannot be recorded as if it were a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataFormat {
    /// `SEN169_FP16` — the 1/6/9 half the compute units run on.
    Sen169Fp16,
    /// `IEEE_FP32`.
    IeeeFp32,
    /// `SEN143_FP8`.
    Sen143Fp8,
    /// `SEN152_FP8`.
    Sen152Fp8,
    /// `SEN153_FP9`.
    Sen153Fp9,
    /// `SENINT2`.
    Senint2,
    /// `SENINT4`.
    Senint4,
    /// `SENINT8`.
    Senint8,
    /// `SENINT16`.
    Senint16,
    /// `SENINT24` — ⛔ SIXTEEN BITS WIDE, not twenty-four; see [`DataType::bits`].
    Senint24,
    /// `IEEE_INT64`.
    IeeeInt64,
    /// `IEEE_INT32`.
    IeeeInt32,
    /// `SENUINT32`.
    Senuint32,
    /// `SENUINT2`.
    Senuint2,
    /// `BOOL`.
    Bool,
}

#[cfg(test)]
mod tests {
    use super::Bits;
    use crate::generated::DataType;

    /// ⭐⭐ IBM'S TABLE, ROW FOR ROW, FOR EVERY FORMAT THIS CRATE HAS.
    ///
    /// `EnumsConversion::dataFormatsToBitWidth` (`util/sendefs/sendefs.cpp:129-141`). Carried as
    /// VALUES rather than as a relation: "the fp8 formats agree with each other" passes on a table
    /// where all three are wrong together.
    #[test]
    fn matches_ibms_bit_width_table() {
        for (format, want) in [
            (DataType::Sen169Fp16, 16),
            (DataType::Bfloat16, 16),
            (DataType::IeeeFp32, 32),
            (DataType::Senuint32, 32),
            (DataType::Sen143Fp8, 8),
            (DataType::Sen080Fp8, 8),
            (DataType::Sen053Fp8, 8),
            (DataType::Senint8, 8),
            (DataType::Bool, 8),
            (DataType::Sen121Fp4, 4),
            (DataType::Senint4, 4),
            // ⛔ THE ROW THE NAME LIES ABOUT.
            (DataType::Senint24, 16),
        ] {
            assert_eq!(
                format.bits(),
                Bits(want),
                "{format:?} disagrees with dataFormatsToBitWidth"
            );
        }
    }

    /// ⛔ A 128-BYTE STICK HOLDS DIFFERENT NUMBERS OF DIFFERENT FORMATS, and the numbers are the
    /// point: 64 fp16, 128 fp8, 256 int4. A stick is not "64 things".
    #[test]
    fn a_stick_holds_what_the_format_says() {
        const STICK_BITS: u32 = 128 * 8;
        assert_eq!(DataType::Sen169Fp16.per_stick(STICK_BITS), 64);
        assert_eq!(DataType::Sen143Fp8.per_stick(STICK_BITS), 128);
        assert_eq!(DataType::Senint4.per_stick(STICK_BITS), 256);
        assert_eq!(DataType::IeeeFp32.per_stick(STICK_BITS), 32);
        // ⛔ AND SENINT24 PACKS AS SIXTEEN BITS, so a stick holds 64 of them, not 42.
        assert_eq!(DataType::Senint24.per_stick(STICK_BITS), 64);
    }
}
