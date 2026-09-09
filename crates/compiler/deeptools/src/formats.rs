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

/// AN ELEMENT FORMAT AS THE SCHEDULER'S OWN VOCABULARY SPELLS IT — `enum class DataFormats`
/// (`util/sendefs/sendefs.h:30`), in the vendor's declaration order.
///
/// ⛔⛔ THIS IS A SUPERSET OF [`DataType`], AND THE DIFFERENCE IS LOAD-BEARING. `DataType` is a
/// CENSUS of our own templates' `data_type=`; the scheduler compares a datastage's format against
/// formats no template of ours declares. `GCVTF16F8PackAction::_out_format`
/// (`ddc/transformations/automatic_shuffle/shuffle.cpp:598`) tests `IEEE_FP16` and `SEN152_FP8`,
/// neither of which the census has, so a port that reasons in `DataType` drops those arms without
/// saying so. [`DataFormat::from`] is the join in the other direction.
///
/// ⛔ `INVALID` AND `NUM_DATA_FORMATS` HAVE NO VARIANT. The vendor's own width table answers `-1`
/// for `INVALID` (`util/sendefs/sendefs.cpp:131`), which is not a width; an absent format is
/// `Option<DataFormat>` here. Dropping it does not disturb the order of the rest, which matters
/// because `AbstractLayout::operator<` compares formats first (`shuffle.h:105-109`) and the derived
/// [`Ord`] here has to agree with that enumerator order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataFormat {
    /// `SEN169_FP16`.
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
    /// `SENINT24` — sixteen bits wide, see [`DataType::bits`].
    Senint24,
    /// `IEEE_INT64`.
    IeeeInt64,
    /// `IEEE_INT32`.
    IeeeInt32,
    /// `SENUINT32`.
    Senuint32,
    /// `SENUINT2`.
    Senuint2,
    /// `IEEE_FP16` — torch's fp16, which is not IBM's [`Self::Sen169Fp16`].
    IeeeFp16,
    /// `BOOL`.
    Bool,
    /// `BFLOAT16`.
    Bfloat16,
    /// `SEN18F_FP24`.
    Sen18fFp24,
    /// `SEN080_FP8` — an MX scale.
    Sen080Fp8,
    /// `SEN053_FP8` — an MX scale.
    Sen053Fp8,
    /// `SEN121_FP4` — MX fp4.
    Sen121Fp4,
}

impl DataFormat {
    /// HOW MANY BITS ONE ELEMENT OCCUPIES — `EnumsConversion::dataFormatsToBitWidth`
    /// (`util/sendefs/sendefs.cpp:129-141`), the same table [`DataType::bits`] transcribes for the
    /// census subset.
    ///
    /// ⛔ `SENINT24` IS SIXTEEN, and `SEN153_FP9` is NINE — a width that is neither a power of two
    /// nor a byte multiple.
    #[must_use]
    pub const fn bits(self) -> Bits {
        Bits(match self {
            Self::IeeeInt64 => 64,
            Self::IeeeFp32 | Self::IeeeInt32 | Self::Senuint32 => 32,
            Self::Sen18fFp24 => 24,
            Self::Sen169Fp16
            | Self::Senint16
            | Self::Senint24
            | Self::IeeeFp16
            | Self::Bfloat16 => 16,
            Self::Sen153Fp9 => 9,
            Self::Sen143Fp8
            | Self::Sen152Fp8
            | Self::Senint8
            | Self::Bool
            | Self::Sen080Fp8
            | Self::Sen053Fp8 => 8,
            Self::Senint4 | Self::Sen121Fp4 => 4,
            Self::Senint2 | Self::Senuint2 => 2,
        })
    }

    /// THE FORMAT A `data_type=` SPELLING NAMES — `FromString<DataFormats>`
    /// (`util/sendefs/sendefs.h:251-297`), all twenty-one spellings it answers.
    ///
    /// ⛔⛔ AN UNKNOWN SPELLING IS `None`, WHERE THE VENDOR ANSWERS `INVALID` — and that difference
    /// is deliberate, because `INVALID` is what makes `DatatypeOp::verify` accept a typo: the width
    /// table answers `-1` for it (`sendefs.cpp:131`), so the "can not find this type" arm is
    /// unreachable and `-1 > bit_width` is false — see `DatatypeOp::verify` (`DdlOps.cpp:149`).
    ///
    /// ⛔ EXACT, NOT CASE-INSENSITIVE: the vendor compares `s == "SEN169_FP16"` and nothing folds
    /// case on the way in, so `"sen169_fp16"` is `INVALID` there and `None` here.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        Some(match spelling {
            "SEN169_FP16" => Self::Sen169Fp16,
            "IEEE_FP32" => Self::IeeeFp32,
            "SEN143_FP8" => Self::Sen143Fp8,
            "SEN152_FP8" => Self::Sen152Fp8,
            "SEN153_FP9" => Self::Sen153Fp9,
            "SENINT2" => Self::Senint2,
            "SENINT4" => Self::Senint4,
            "SENINT8" => Self::Senint8,
            "SENINT16" => Self::Senint16,
            "SENINT24" => Self::Senint24,
            "IEEE_INT64" => Self::IeeeInt64,
            "IEEE_INT32" => Self::IeeeInt32,
            "SENUINT32" => Self::Senuint32,
            "SENUINT2" => Self::Senuint2,
            "IEEE_FP16" => Self::IeeeFp16,
            "BOOL" => Self::Bool,
            "BFLOAT16" => Self::Bfloat16,
            "SEN18F_FP24" => Self::Sen18fFp24,
            "SEN080_FP8" => Self::Sen080Fp8,
            "SEN053_FP8" => Self::Sen053Fp8,
            "SEN121_FP4" => Self::Sen121Fp4,
            _ => return None,
        })
    }
}

impl From<DataType> for DataFormat {
    /// THE CENSUS AS A VENDOR FORMAT — every `data_type=` our templates declare is one of the
    /// vendor's, spelled the same way.
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Sen169Fp16 => Self::Sen169Fp16,
            DataType::Bfloat16 => Self::Bfloat16,
            DataType::IeeeFp32 => Self::IeeeFp32,
            DataType::Senuint32 => Self::Senuint32,
            DataType::Sen143Fp8 => Self::Sen143Fp8,
            DataType::Sen080Fp8 => Self::Sen080Fp8,
            DataType::Sen053Fp8 => Self::Sen053Fp8,
            DataType::Senint8 => Self::Senint8,
            DataType::Bool => Self::Bool,
            DataType::Sen121Fp4 => Self::Sen121Fp4,
            DataType::Senint4 => Self::Senint4,
            DataType::Senint24 => Self::Senint24,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Bits, DataFormat};
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

    /// ⭐ THE JOIN BETWEEN THE TWO FORMAT VOCABULARIES, ON THE ONE THING BOTH ANSWER.
    ///
    /// ⛔ CARRIED AS THE WIDTH, NOT AS "THE TWO AGREE": a `From` that mapped every census format to
    /// `SENINT8` would satisfy any relation stated between the two tables. Twelve census formats,
    /// twelve widths.
    #[test]
    fn every_census_format_is_the_same_vendor_width() {
        for data_type in [
            DataType::Sen169Fp16,
            DataType::Bfloat16,
            DataType::IeeeFp32,
            DataType::Senuint32,
            DataType::Sen143Fp8,
            DataType::Sen080Fp8,
            DataType::Sen053Fp8,
            DataType::Senint8,
            DataType::Bool,
            DataType::Sen121Fp4,
            DataType::Senint4,
            DataType::Senint24,
        ] {
            assert_eq!(
                DataFormat::from(data_type).bits(),
                data_type.bits(),
                "{data_type:?} changes width crossing into the vendor's DataFormats"
            );
        }
    }

    /// ⛔ THE TWO FORMATS THE CENSUS CANNOT SPELL, which is the whole reason [`DataFormat`] exists:
    /// `_out_format` (`ddc/transformations/automatic_shuffle/shuffle.cpp:598`) names both.
    #[test]
    fn the_vendor_has_formats_the_census_does_not() {
        assert_eq!(DataFormat::IeeeFp16.bits(), Bits(16));
        assert_eq!(DataFormat::Sen152Fp8.bits(), Bits(8));
    }
    /// ⭐ EVERY SPELLING `FromString<DataFormats>` ANSWERS, AND ONE IT DOES NOT.
    ///
    /// ⛔ CARRIED AS THE SPELLING-TO-VARIANT PAIR, not as a count: a `from_spelling` that mapped
    /// two spellings to one variant would still answer 21 spellings.
    #[test]
    fn every_vendor_spelling_names_its_own_format() {
        for (spelling, want) in [
            ("SEN169_FP16", DataFormat::Sen169Fp16),
            ("IEEE_FP32", DataFormat::IeeeFp32),
            ("SEN143_FP8", DataFormat::Sen143Fp8),
            ("SEN152_FP8", DataFormat::Sen152Fp8),
            ("SEN153_FP9", DataFormat::Sen153Fp9),
            ("SENINT2", DataFormat::Senint2),
            ("SENINT4", DataFormat::Senint4),
            ("SENINT8", DataFormat::Senint8),
            ("SENINT16", DataFormat::Senint16),
            ("SENINT24", DataFormat::Senint24),
            ("IEEE_INT64", DataFormat::IeeeInt64),
            ("IEEE_INT32", DataFormat::IeeeInt32),
            ("SENUINT32", DataFormat::Senuint32),
            ("SENUINT2", DataFormat::Senuint2),
            ("IEEE_FP16", DataFormat::IeeeFp16),
            ("BOOL", DataFormat::Bool),
            ("BFLOAT16", DataFormat::Bfloat16),
            ("SEN18F_FP24", DataFormat::Sen18fFp24),
            ("SEN080_FP8", DataFormat::Sen080Fp8),
            ("SEN053_FP8", DataFormat::Sen053Fp8),
            ("SEN121_FP4", DataFormat::Sen121Fp4),
        ] {
            assert_eq!(
                DataFormat::from_spelling(spelling),
                Some(want),
                "{spelling}"
            );
        }
        // ⛔ `INVALID` IS A SPELLING THE VENDOR'S ENUM HAS AND THIS ONE DOES NOT, and the lowercase
        // form of a real name is not a real name.
        assert_eq!(DataFormat::from_spelling("INVALID"), None);
        assert_eq!(DataFormat::from_spelling("sen169_fp16"), None);
    }
}
