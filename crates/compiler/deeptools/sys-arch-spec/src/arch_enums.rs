//! The architecture's enumerations — `sys-arch-spec/arch_enums.h`.
//!
//! The register files, the LX segments, and the `{unit, storage}` pair every location is named by.

/// 🛑🛑🛑 `MAX_VALUE` IS NOT THE MAXIMUM.
///
/// ```text
/// enum RegType {
///   LRF, LAR, LBR, EAR, EBR, GTR, JCR, ERAT, MVR, XRF, SPR, ARF, IRF,
///   STATE,
///   SCALE,
///   MAX_VALUE = STATE
/// };
/// ```
///
/// `SCALE` is declared AFTER `STATE` and therefore has the larger value, while `MAX_VALUE` is
/// pinned to `STATE`. So `MAX_VALUE` is one less than the largest enumerator.
///
/// 🛑 Every `for (int i = 0; i <= MAX_VALUE; ++i)`, every `std::array<T, MAX_VALUE + 1>` and every
/// bounds check written against it silently excludes the scale register file — the one that a
/// quantised model needs.
///
/// 🔑 The alias is the LAST entry in the list, so `SCALE` was added above it rather than below,
/// which is the placement that keeps the alias's own line unchanged.
pub const MAX_VALUE_IS_NOT_THE_MAXIMUM: &str = "arch_enums.h :: RegType::MAX_VALUE";

/// A register file — `RegType`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u8)]
pub enum RegType {
    /// `LRF = 0` — the local register file.
    Lrf = 0,
    /// `LAR = 1`.
    Lar,
    /// `LBR = 2`.
    Lbr,
    /// `EAR = 3`.
    Ear,
    /// `EBR = 4`.
    Ebr,
    /// `GTR = 5`.
    Gtr,
    /// `JCR = 6`.
    Jcr,
    /// `ERAT = 7`.
    Erat,
    /// `MVR = 8`.
    Mvr,
    /// `XRF = 9`.
    Xrf,
    /// `SPR = 10`.
    Spr,
    /// `ARF = 11`.
    Arf,
    /// `IRF = 12`.
    Irf,
    /// `STATE = 13` — also what `MAX_VALUE` names.
    State,
    /// `SCALE = 14` — 🛑 above `MAX_VALUE`.
    Scale,
}

impl RegType {
    /// All fifteen, in declaration order.
    pub const ALL: [Self; 15] = [
        Self::Lrf,
        Self::Lar,
        Self::Lbr,
        Self::Ear,
        Self::Ebr,
        Self::Gtr,
        Self::Jcr,
        Self::Erat,
        Self::Mvr,
        Self::Xrf,
        Self::Spr,
        Self::Arf,
        Self::Irf,
        Self::State,
        Self::Scale,
    ];

    /// What the header's `MAX_VALUE` is worth.
    pub const MAX_VALUE: u8 = Self::State as u8;

    /// The value of the largest enumerator, which is not `MAX_VALUE`.
    pub const TRUE_MAX: u8 = Self::Scale as u8;

    /// Whether a loop bounded by `MAX_VALUE` reaches it.
    #[must_use]
    pub const fn is_reached_by_a_max_value_loop(self) -> bool {
        (self as u8) <= Self::MAX_VALUE
    }
}

const _: () = assert!(
    RegType::MAX_VALUE < RegType::TRUE_MAX,
    "the enumerator called MAX_VALUE is smaller than the largest enumerator"
);
const _: () = assert!(
    !RegType::Scale.is_reached_by_a_max_value_loop(),
    "the scale register file is the one every MAX_VALUE-bounded loop skips"
);

/// 🛑 TWO SEGMENT SCHEMES IN ONE ENUM.
///
/// ```text
/// enum class LdsSegment {
///   OUTPUT, INPUT, STACK, MODEL, HEAP, RESERVE1, RESERVE2, CONST,
///   // For SNT1.5 : 4 segments per HMI * 8
///   SEG0_HMI0, SEG1_HMI0, SEG2_HMI0, SEG3_HMI0,
///   …
///   SEG3_HMI7
/// };
/// ```
///
/// The first eight are named by ROLE — output, input, stack, model, heap, two reserves and
/// constants. The next thirty-two are named by POSITION — four segments on each of eight HMIs —
/// and the comment says which generation they are for.
///
/// 🛑 So a value of this enum is in one scheme or the other and nothing in the type says which,
/// while `RESERVE1` and `RESERVE2` are placeholders in the first that the second does not need.
///
/// 🔑 Eight roles and eight HMIs is a coincidence of counts, not a correspondence: the roles do
/// not map onto the HMIs.
pub const TWO_SEGMENT_SCHEMES_IN_ONE_ENUM: &str = "arch_enums.h :: LdsSegment";

/// Which naming scheme an `LdsSegment` value belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SegmentScheme {
    /// The eight role-named segments.
    ByRole,
    /// The thirty-two `SEGn_HMIm` segments.
    ByPosition,
}

impl SegmentScheme {
    /// Both.
    pub const ALL: [Self; 2] = [Self::ByRole, Self::ByPosition];

    /// How many enumerators it covers.
    #[must_use]
    pub const fn enumerators(self) -> usize {
        match self {
            Self::ByRole => 8,
            Self::ByPosition => 4 * 8,
        }
    }

    /// Which scheme an enumerator index belongs to.
    #[must_use]
    pub const fn of_index(index: usize) -> Option<Self> {
        if index < Self::ByRole.enumerators() {
            Some(Self::ByRole)
        } else if index < Self::ByRole.enumerators() + Self::ByPosition.enumerators() {
            Some(Self::ByPosition)
        } else {
            None
        }
    }
}

/// How many segments the enum declares in total.
pub const LDS_SEGMENTS: usize = 40;

/// How many of the role-named ones are placeholders.
///
/// 🔑 `RESERVE1` and `RESERVE2`.
pub const RESERVED_SEGMENTS: usize = 2;

const _: () = assert!(
    SegmentScheme::ByRole.enumerators() + SegmentScheme::ByPosition.enumerators() == LDS_SEGMENTS,
    "eight roles and thirty-two positions"
);

/// A segment addressed by position — four per HMI, eight HMIs.
///
/// 🔑 The C++ writes out all thirty-two names; here the two coordinates are the type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct HmiSegment {
    segment: u8,
    hmi: u8,
}

impl HmiSegment {
    /// Segments on one HMI.
    pub const SEGMENTS_PER_HMI: u8 = 4;

    /// How many HMIs there are.
    pub const HMIS: u8 = 8;

    /// Build one, or `None` outside the grid.
    #[must_use]
    pub const fn new(segment: u8, hmi: u8) -> Option<Self> {
        if segment < Self::SEGMENTS_PER_HMI && hmi < Self::HMIS {
            Some(Self { segment, hmi })
        } else {
            None
        }
    }

    /// Its index within the position-named part of the enum.
    #[must_use]
    pub const fn position_index(self) -> usize {
        (self.hmi * Self::SEGMENTS_PER_HMI + self.segment) as usize
    }

    /// Its index in the whole enum.
    #[must_use]
    pub const fn enum_index(self) -> usize {
        SegmentScheme::ByRole.enumerators() + self.position_index()
    }
}

const _: () = assert!(
    (HmiSegment::SEGMENTS_PER_HMI * HmiSegment::HMIS) as usize
        == SegmentScheme::ByPosition.enumerators(),
    "the grid's size is the enumerator count the comment states"
);

/// 🛑 A PAIR OF ONE ENUM, WITH THE TWO ROLES TOLD APART BY POSITION.
///
/// ```text
/// struct DataLocation {
///   SenComponents unit_ = SenComponents::NO_COMPONENT;     // e.g. LXLU, PT_ROW0
///   SenComponents storage_ = SenComponents::NO_COMPONENT;  // e.g. LX, PT_XRF
/// };
///
/// template <> struct std::hash<DataLocation> {
///   size_t operator()(const DataLocation& x) const {
///     return std::hash<SenComponents>()(x.storage_) ^
///            (std::hash<SenComponents>()(x.unit_) << 1);
///   }
/// };
/// ```
///
/// Both members are the same type and the difference between a unit and a storage is a comment. So
/// `{LX, LXLU}` is as constructible as `{LXLU, LX}` and nothing rejects it.
///
/// 🛑 The hash treats them ASYMMETRICALLY — storage unshifted, unit shifted left by one — so the
/// two orderings do hash differently, and a swapped pair is a different key rather than an error.
///
/// 🔑 It is the `hash ^ (hash << 1)` idiom, in a tree whose `util/utils.h` has a real
/// `hash_combine`; the shift by ONE also means two units differing in their top bit collide with
/// two storages differing in the next.
///
/// 🛑 `std::equal_to<DataLocation>` is specialised too, instead of the struct defining
/// `operator==` — so the type compares equal inside an `unordered_map` and does not compare at all
/// anywhere else.
pub const A_PAIR_OF_ONE_ENUM: &str = "arch_enums.h :: DataLocation";

/// Where data is — a unit and the storage it reaches.
///
/// 🔑 The two roles are separate fields here as in the C++, but the constructor is the only way in,
/// so the pair cannot be built from an unordered tuple by accident.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct DataLocation {
    /// The unit — `e.g. LXLU, PT_ROW0`.
    pub unit: SenComponent,
    /// The storage it reaches — `e.g. LX, PT_XRF`.
    pub storage: SenComponent,
}

impl DataLocation {
    /// What both members default to.
    pub const UNSET: Self = Self {
        unit: SenComponent::NoComponent,
        storage: SenComponent::NoComponent,
    };

    /// The C++ hash, written out.
    ///
    /// 🛑 `hash(storage) ^ (hash(unit) << 1)`, over the enumerator values — reproduced so the
    /// asymmetry is visible.
    #[must_use]
    pub const fn cpp_hash(self) -> u64 {
        let storage = self.storage as i32 as u64;
        let unit = self.unit as i32 as u64;
        storage ^ (unit << 1)
    }

    /// The same location with its two members swapped.
    #[must_use]
    pub const fn swapped(self) -> Self {
        Self {
            unit: self.storage,
            storage: self.unit,
        }
    }
}

const _: () = assert!(
    DataLocation::UNSET.cpp_hash() == 1,
    "NO_COMPONENT is -1, so the default location hashes to (-1) ^ (-1 << 1), which is 1"
);

/// 🎯🎯🎯 THE OP-FUNC VOCABULARY — `OpFuncs`, `sys-arch-spec/arch_enums.h` 135-312.
///
/// Every compute op in the machine, PT and SFP alike, names itself with one of these. It is the
/// widest enum in the C++ and the ONE place an op is spelled; [`crate::pt_compute::PtOpFunc`] is
/// the 34-entry subset `doPtCompute` dispatches on.
///
/// 🛑 `NONE` and `UNDEF` are BOTH members — two spellings of absence, at opposite ends of the
/// enum, and only one of them (`NONE`) is the declared default of `ComputeOpInfo::opFuncName`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u8)]
pub enum OpFunc {
    /// `NONE`.
    None,
    /// `ADD`.
    Add,
    /// `ADD_I32_TO_I32`.
    AddI32ToI32,
    /// `ADD_I64_TO_I64`.
    AddI64ToI64,
    /// `STRIDED_ADD`.
    StridedAdd,
    /// `MUL`.
    Mul,
    /// `MUL_I32_TO_I32`.
    MulI32ToI32,
    /// `SUB`.
    Sub,
    /// `REVSUB`.
    Revsub,
    /// `REALDIV`.
    Realdiv,
    /// `QUANT_SCALE_PER_TOKEN`.
    QuantScalePerToken,
    /// `QUANT_SCALE_PER_TOKEN_FP8`.
    QuantScalePerTokenFp8,
    /// `MASK_2BIT`.
    Mask2Bit,
    /// `FNMS`.
    Fnms,
    /// `ROPE64P1_FWD`.
    Rope64P1Fwd,
    /// `ROPE64P2_FWD`.
    Rope64P2Fwd,
    /// `WHERE3`.
    Where3,
    /// `SUM_NONSTICK`.
    SumNonstick,
    /// `MEAN_NONSTICK`.
    MeanNonstick,
    /// `MAX_NONSTICK`.
    MaxNonstick,
    /// `ABSMAX_NONSTICK`.
    AbsmaxNonstick,
    /// `MIN_NONSTICK`.
    MinNonstick,
    /// `PROD_NONSTICK`.
    ProdNonstick,
    /// `SUM`.
    Sum,
    /// `TOPK_VALUE`.
    TopkValue,
    /// `TOPK_INDEX`.
    TopkIndex,
    /// `MASK_BY_INDEX`.
    MaskByIndex,
    /// `MEAN`.
    Mean,
    /// `MAX`.
    Max,
    /// `ABSMAX`.
    Absmax,
    /// `MIN`.
    Min,
    /// `EXX2`.
    Exx2,
    /// `EXX2_ZEROMEAN`.
    Exx2Zeromean,
    /// `LAYERNORM_SCALE`.
    LayernormScale,
    /// `LAYERNORM_NORM`.
    LayernormNorm,
    /// `LAYERNORM_BWDNORM`.
    LayernormBwdnorm,
    /// `RSQRT`.
    Rsqrt,
    /// `RECIPROCAL`.
    Reciprocal,
    /// `BIASADD`.
    Biasadd,
    /// `RELU_FWD`.
    ReluFwd,
    /// `RELU6_FWD`.
    Relu6Fwd,
    /// `LEAKYRELU_FWD`.
    LeakyreluFwd,
    /// `DL16TOFP32`.
    Dl16Tofp32,
    /// `FP32TODL16`.
    Fp32Todl16,
    /// `FP8TODL16`.
    Fp8Todl16,
    /// `DL16TOBF16`.
    Dl16Tobf16,
    /// `GELU_FWD`.
    GeluFwd,
    /// `GELU_BWD`.
    GeluBwd,
    /// `EXP_FWD`.
    ExpFwd,
    /// `FAST_EXP_FWD`.
    FastExpFwd,
    /// `ERF_FWD`.
    ErfFwd,
    /// `SQRT_FWD`.
    SqrtFwd,
    /// `TANH_FWD`.
    TanhFwd,
    /// `TANH_BWD`.
    TanhBwd,
    /// `SIGMOID_FWD`.
    SigmoidFwd,
    /// `FAST_SIGMOID_FWD`.
    FastSigmoidFwd,
    /// `SILU_FWD`.
    SiluFwd,
    /// `MISH_FWD`.
    MishFwd,
    /// `CLIP_FWD`.
    ClipFwd,
    /// `LSTMACTP1_FWD`.
    Lstmactp1Fwd,
    /// `LSTMACTP2_FWD`.
    Lstmactp2Fwd,
    /// `BATCHNORM_FWD`.
    BatchnormFwd,
    /// `LOG_FWD`.
    LogFwd,
    /// `SOFTMAX`.
    Softmax,
    /// `SOFTPLUS`.
    Softplus,
    /// `MAXPOOL_FWD`.
    MaxpoolFwd,
    /// `AVGPOOL_FWD`.
    AvgpoolFwd,
    /// `AVGPOOL_NMAP_FWD`.
    AvgpoolNmapFwd,
    /// `CONV2D_FWD`.
    Conv2DFwd,
    /// `CONV2D_FP8_FWD`.
    Conv2DFp8Fwd,
    /// `CONV2D_INT8_FWD`.
    Conv2DInt8Fwd,
    /// `CONV2D_INT4_FWD`.
    Conv2DInt4Fwd,
    /// `CONV2D_FWD_GENKG3`.
    Conv2DFwdGenkg3,
    /// `CONV2D_FP8_FWD_GENKG3`.
    Conv2DFp8FwdGenkg3,
    /// `CONV2D_INT8_FWD_GENKG3`.
    Conv2DInt8FwdGenkg3,
    /// `CONV2D_INT4_FWD_GENKG3`.
    Conv2DInt4FwdGenkg3,
    /// `CONV2D_FWD_SPARSEKG3`.
    Conv2DFwdSparsekg3,
    /// `CONV2D_FP8_FWD_SPARSEKG3`.
    Conv2DFp8FwdSparsekg3,
    /// `CONV2D_INT8_FWD_SPARSEKG3`.
    Conv2DInt8FwdSparsekg3,
    /// `CONV2D_INT4_FWD_SPARSEKG3`.
    Conv2DInt4FwdSparsekg3,
    /// `MATMUL_FWD`.
    MatmulFwd,
    /// `MATMUL_FP8_FWD`.
    MatmulFp8Fwd,
    /// `MATMUL_INT8_FWD`.
    MatmulInt8Fwd,
    /// `MATMUL_INT4_FWD`.
    MatmulInt4Fwd,
    /// `BATCHMATMUL_FWD`.
    BatchmatmulFwd,
    /// `BATCHMATMULV2`.
    Batchmatmulv2,
    /// `BATCHMATMUL_MXFP4W_FWD`.
    BatchmatmulMxfp4WFwd,
    /// `BATCHMATMUL_FP8_FWD`.
    BatchmatmulFp8Fwd,
    /// `BATCHMATMUL_FP8_FWD_MB`.
    BatchmatmulFp8FwdMb,
    /// `BATCHMATMUL_INT8_FWD`.
    BatchmatmulInt8Fwd,
    /// `BATCHMATMUL_INT8_FWD_MBKG3`.
    BatchmatmulInt8FwdMbkg3,
    /// `BATCHMATMUL_INT4_FWD`.
    BatchmatmulInt4Fwd,
    /// `BATCHMATMUL_FWD_SPARSEKG3`.
    BatchmatmulFwdSparsekg3,
    /// `BATCHMATMUL_FP8_FWD_SPARSEKG3`.
    BatchmatmulFp8FwdSparsekg3,
    /// `BATCHMATMUL_INT8_FWD_SPARSEKG3`.
    BatchmatmulInt8FwdSparsekg3,
    /// `BATCHMATMUL_INT4_FWD_SPARSEKG3`.
    BatchmatmulInt4FwdSparsekg3,
    /// `BATCHMATMUL_XRF_FWD`.
    BatchmatmulXrfFwd,
    /// `BATCHMATMUL_XRF_FP8_FWD`.
    BatchmatmulXrfFp8Fwd,
    /// `BATCHMATMUL_XRF_INT8_FWD`.
    BatchmatmulXrfInt8Fwd,
    /// `BATCHMATMUL_XRF_INT4_FWD`.
    BatchmatmulXrfInt4Fwd,
    /// `BATCHMATMUL_XRFCH_FWD`.
    BatchmatmulXrfchFwd,
    /// `BATCHMATMUL_XRFCH_FP8_FWD`.
    BatchmatmulXrfchFp8Fwd,
    /// `BATCHMATMUL_XRFCH_INT8_FWD`.
    BatchmatmulXrfchInt8Fwd,
    /// `BATCHMATMUL_XRFCH_INT4_FWD`.
    BatchmatmulXrfchInt4Fwd,
    /// `SCALED_GROUP_MATMUL_FP4_FWD`.
    ScaledGroupMatmulFp4Fwd,
    /// `CONV2D_FWD_OS1`.
    Conv2DFwdOs1,
    /// `CONV2D_FWD_GEN_OS1`.
    Conv2DFwdGenOs1,
    /// `CONV2D_INT8_FWD_OS1`.
    Conv2DInt8FwdOs1,
    /// `CONV2D_XRF_INT8_FWD_OS1`.
    Conv2DXrfInt8FwdOs1,
    /// `IDENTITY`.
    Identity,
    /// `SHUFFLE`.
    Shuffle,
    /// `DEPTHWISE_CONV_FWD`.
    DepthwiseConvFwd,
    /// `SQDIFF_FWD`.
    SqdiffFwd,
    /// `LSTMBLOCKCELL`.
    Lstmblockcell,
    /// `Q_FP8`.
    QFp8,
    /// `Q_FP8_CH`.
    QFp8Ch,
    /// `Q_FP8_CHIL`.
    QFp8Chil,
    /// `Q_FP8_WT`.
    QFp8Wt,
    /// `Q_FP8_MB`.
    QFp8Mb,
    /// `CSQ_INT8`.
    CsqInt8,
    /// `CSQ_INT8_V2`.
    CsqInt8V2,
    /// `CSQ_INT8_CH`.
    CsqInt8Ch,
    /// `CSQ_INT8_WT`.
    CsqInt8Wt,
    /// `CSQ_INT8_CHIL`.
    CsqInt8Chil,
    /// `CSQ_INT8_MB`.
    CsqInt8Mb,
    /// `CSQ_INT8_MB_V2`.
    CsqInt8MbV2,
    /// `CSQ_INT4`.
    CsqInt4,
    /// `CSQ_INT4_WT`.
    CsqInt4Wt,
    /// `CSQ_INT4_CHIL`.
    CsqInt4Chil,
    /// `SFP_ReadLX_TRANSPOSE_FWDL0`.
    SfpReadLxTransposeFwdl0,
    /// `PT_BLK_TRANSPOSE_LOAD`.
    PtBlkTransposeLoad,
    /// `STCDPOpLx`.
    StcdpOpLx,
    /// `STCDPOpHBM`.
    StcdpOpHbm,
    /// `ResizeNNHBM`.
    ResizeNnhbm,
    /// `ResizeNNLX`.
    ResizeNnlx,
    /// `GatherOpHBM`.
    GatherOpHbm,
    /// `APEOpLX`.
    ApeOpLx,
    /// `APEOpHBM`.
    ApeOpHbm,
    /// `ReStickifyOpLx`.
    ReStickifyOpLx,
    /// `ReStickifyOpHBM`.
    ReStickifyOpHbm,
    /// `ReStickifyOpWithPTLx`.
    ReStickifyOpWithPtLx,
    /// `ReStickifyOpWithPTHBM`.
    ReStickifyOpWithPthbm,
    /// `XRFWriteHBM`.
    XrfWriteHbm,
    /// `XRFWriteLX`.
    XrfWriteLx,
    /// `Nop`.
    Nop,
    /// `StickifyOpHBM`.
    StickifyOpHbm,
    /// `ConstPadOpLX`.
    ConstPadOpLx,
    /// `ConstPadOpHBM`.
    ConstPadOpHbm,
    /// `PESFP_Collate_2B_Writes`.
    PesfpCollate2BWrites,
    /// `BNPRECZEROSHFT`.
    Bnpreczeroshft,
    /// `DBL_BUF_MNI`.
    DblBufMni,
    /// `ScatterOpHBM`.
    ScatterOpHbm,
    /// `ITOF`.
    Itof,
    /// `ITOFHBM`.
    Itofhbm,
    /// `ABS`.
    Abs,
    /// `NEG`.
    Neg,
    /// `EQUAL`.
    Equal,
    /// `NOTEQUAL`.
    Notequal,
    /// `GREATEREQUAL`.
    Greaterequal,
    /// `GREATERTHAN`.
    Greaterthan,
    /// `LESSEREQUAL`.
    Lesserequal,
    /// `LESSERTHAN`.
    Lesserthan,
    /// `AllGather`.
    AllGather,
    /// `AllReduce`.
    AllReduce,
    /// `AllShuffle`.
    AllShuffle,
    /// `MAXIMUM`.
    Maximum,
    /// `MINIMUM`.
    Minimum,
    /// `GENERIC_PARTIAL_REDUCTION`.
    GenericPartialReduction,
    /// `INTERSLICETRANSPOSE_FP16`.
    InterslicetransposeFp16,
    /// `INTERSLICETRANSPOSE_FP8`.
    InterslicetransposeFp8,
    /// `FLOOR`.
    Floor,
    /// `BATCHMATMUL_MXFP8_FWD`.
    BatchmatmulMxfp8Fwd,
    /// `SINKCORRECTIONFACTOR`.
    Sinkcorrectionfactor,
    /// `INT32IDXTOADDR`.
    Int32Idxtoaddr,
    /// `STZ_LATCH`.
    StzLatch,
    /// `UNDEF`.
    Undef,
}

impl OpFunc {
    /// Every op-func, in the order the enum declares them.
    pub const ALL: [Self; 176] = [
        Self::None,
        Self::Add,
        Self::AddI32ToI32,
        Self::AddI64ToI64,
        Self::StridedAdd,
        Self::Mul,
        Self::MulI32ToI32,
        Self::Sub,
        Self::Revsub,
        Self::Realdiv,
        Self::QuantScalePerToken,
        Self::QuantScalePerTokenFp8,
        Self::Mask2Bit,
        Self::Fnms,
        Self::Rope64P1Fwd,
        Self::Rope64P2Fwd,
        Self::Where3,
        Self::SumNonstick,
        Self::MeanNonstick,
        Self::MaxNonstick,
        Self::AbsmaxNonstick,
        Self::MinNonstick,
        Self::ProdNonstick,
        Self::Sum,
        Self::TopkValue,
        Self::TopkIndex,
        Self::MaskByIndex,
        Self::Mean,
        Self::Max,
        Self::Absmax,
        Self::Min,
        Self::Exx2,
        Self::Exx2Zeromean,
        Self::LayernormScale,
        Self::LayernormNorm,
        Self::LayernormBwdnorm,
        Self::Rsqrt,
        Self::Reciprocal,
        Self::Biasadd,
        Self::ReluFwd,
        Self::Relu6Fwd,
        Self::LeakyreluFwd,
        Self::Dl16Tofp32,
        Self::Fp32Todl16,
        Self::Fp8Todl16,
        Self::Dl16Tobf16,
        Self::GeluFwd,
        Self::GeluBwd,
        Self::ExpFwd,
        Self::FastExpFwd,
        Self::ErfFwd,
        Self::SqrtFwd,
        Self::TanhFwd,
        Self::TanhBwd,
        Self::SigmoidFwd,
        Self::FastSigmoidFwd,
        Self::SiluFwd,
        Self::MishFwd,
        Self::ClipFwd,
        Self::Lstmactp1Fwd,
        Self::Lstmactp2Fwd,
        Self::BatchnormFwd,
        Self::LogFwd,
        Self::Softmax,
        Self::Softplus,
        Self::MaxpoolFwd,
        Self::AvgpoolFwd,
        Self::AvgpoolNmapFwd,
        Self::Conv2DFwd,
        Self::Conv2DFp8Fwd,
        Self::Conv2DInt8Fwd,
        Self::Conv2DInt4Fwd,
        Self::Conv2DFwdGenkg3,
        Self::Conv2DFp8FwdGenkg3,
        Self::Conv2DInt8FwdGenkg3,
        Self::Conv2DInt4FwdGenkg3,
        Self::Conv2DFwdSparsekg3,
        Self::Conv2DFp8FwdSparsekg3,
        Self::Conv2DInt8FwdSparsekg3,
        Self::Conv2DInt4FwdSparsekg3,
        Self::MatmulFwd,
        Self::MatmulFp8Fwd,
        Self::MatmulInt8Fwd,
        Self::MatmulInt4Fwd,
        Self::BatchmatmulFwd,
        Self::Batchmatmulv2,
        Self::BatchmatmulMxfp4WFwd,
        Self::BatchmatmulFp8Fwd,
        Self::BatchmatmulFp8FwdMb,
        Self::BatchmatmulInt8Fwd,
        Self::BatchmatmulInt8FwdMbkg3,
        Self::BatchmatmulInt4Fwd,
        Self::BatchmatmulFwdSparsekg3,
        Self::BatchmatmulFp8FwdSparsekg3,
        Self::BatchmatmulInt8FwdSparsekg3,
        Self::BatchmatmulInt4FwdSparsekg3,
        Self::BatchmatmulXrfFwd,
        Self::BatchmatmulXrfFp8Fwd,
        Self::BatchmatmulXrfInt8Fwd,
        Self::BatchmatmulXrfInt4Fwd,
        Self::BatchmatmulXrfchFwd,
        Self::BatchmatmulXrfchFp8Fwd,
        Self::BatchmatmulXrfchInt8Fwd,
        Self::BatchmatmulXrfchInt4Fwd,
        Self::ScaledGroupMatmulFp4Fwd,
        Self::Conv2DFwdOs1,
        Self::Conv2DFwdGenOs1,
        Self::Conv2DInt8FwdOs1,
        Self::Conv2DXrfInt8FwdOs1,
        Self::Identity,
        Self::Shuffle,
        Self::DepthwiseConvFwd,
        Self::SqdiffFwd,
        Self::Lstmblockcell,
        Self::QFp8,
        Self::QFp8Ch,
        Self::QFp8Chil,
        Self::QFp8Wt,
        Self::QFp8Mb,
        Self::CsqInt8,
        Self::CsqInt8V2,
        Self::CsqInt8Ch,
        Self::CsqInt8Wt,
        Self::CsqInt8Chil,
        Self::CsqInt8Mb,
        Self::CsqInt8MbV2,
        Self::CsqInt4,
        Self::CsqInt4Wt,
        Self::CsqInt4Chil,
        Self::SfpReadLxTransposeFwdl0,
        Self::PtBlkTransposeLoad,
        Self::StcdpOpLx,
        Self::StcdpOpHbm,
        Self::ResizeNnhbm,
        Self::ResizeNnlx,
        Self::GatherOpHbm,
        Self::ApeOpLx,
        Self::ApeOpHbm,
        Self::ReStickifyOpLx,
        Self::ReStickifyOpHbm,
        Self::ReStickifyOpWithPtLx,
        Self::ReStickifyOpWithPthbm,
        Self::XrfWriteHbm,
        Self::XrfWriteLx,
        Self::Nop,
        Self::StickifyOpHbm,
        Self::ConstPadOpLx,
        Self::ConstPadOpHbm,
        Self::PesfpCollate2BWrites,
        Self::Bnpreczeroshft,
        Self::DblBufMni,
        Self::ScatterOpHbm,
        Self::Itof,
        Self::Itofhbm,
        Self::Abs,
        Self::Neg,
        Self::Equal,
        Self::Notequal,
        Self::Greaterequal,
        Self::Greaterthan,
        Self::Lesserequal,
        Self::Lesserthan,
        Self::AllGather,
        Self::AllReduce,
        Self::AllShuffle,
        Self::Maximum,
        Self::Minimum,
        Self::GenericPartialReduction,
        Self::InterslicetransposeFp16,
        Self::InterslicetransposeFp8,
        Self::Floor,
        Self::BatchmatmulMxfp8Fwd,
        Self::Sinkcorrectionfactor,
        Self::Int32Idxtoaddr,
        Self::StzLatch,
        Self::Undef,
    ];

    /// `EnumsConversion::opFuncsToString` - `arch_enums.cpp` 322-499, in the enum's order.
    ///
    /// The spellings are NOT a mechanical lowering of the names: most are lower-cased with the
    /// underscores and any `_FWD` suffix dropped, but a dozen keep their camel case verbatim
    /// (`APEOpHBM`, `ReStickifyOpLx`, and `PESFP_Collate_2B_Writes` with its underscores intact), so neither side can be derived from the other.
    ///
    /// Positional against [`Self::ALL`], which [`OP_FUNC_SPELLINGS_ARE_IN_ENUM_ORDER`] pins.
    const SPELLINGS: [&'static str; 176] = [
        "none",
        "add",
        "addi32toi32",
        "addi64toi64",
        "stridedadd",
        "mul",
        "muli32toi32",
        "sub",
        "revsub",
        "realdiv",
        "quantscalepertoken",
        "quantscalepertokenfp8",
        "mask2bit",
        "fnms",
        "rope64p1",
        "rope64p2",
        "where3",
        "sumnonstick",
        "meannonstick",
        "maxnonstick",
        "absmaxnonstick",
        "minnonstick",
        "prodnonstick",
        "sum",
        "topkvalue",
        "topkindex",
        "maskbyindex",
        "mean",
        "max",
        "absmax",
        "min",
        "exx2",
        "exx2_zeromean",
        "layernormscale",
        "layernormnorm",
        "layernormbackwardnorm",
        "rsqrt",
        "reciprocal",
        "biasadd",
        "relufwd",
        "relu6fwd",
        "leakyrelufwd",
        "dl16tofp32",
        "fp32todl16",
        "fp8todl16",
        "dl16tobf16",
        "gelufwd",
        "gelubackward",
        "exp",
        "fastexp",
        "erf",
        "sqrt",
        "tanh",
        "tanhbackward",
        "sigmoid",
        "fastsigmoid",
        "silu",
        "mish",
        "clip",
        "lstmactp1",
        "lstmactp2",
        "batchnormfwd",
        "log",
        "softmax",
        "softplus",
        "maxpoolfwd",
        "avgpoolfwd",
        "avgpoolnmapfwd",
        "conv2d",
        "conv2dfp8",
        "conv2dint8",
        "conv2dint4",
        "conv2dgenkg3",
        "conv2dfp8genkg3",
        "conv2dint8genkg3",
        "conv2dint4genkg3",
        "conv2dsparsekg3",
        "conv2dfp8sparsekg3",
        "conv2dint8sparsekg3",
        "conv2dint4sparsekg3",
        "matmul",
        "matmulfp8",
        "matmulint8",
        "matmulint4",
        "batchmatmul",
        "batchmatmulv2",
        "batchmatmulmxfp4w",
        "batchmatmulfp8",
        "batchmatmulfp8mb",
        "batchmatmulint8",
        "batchmatmulint8mbkg3",
        "batchmatmulint4",
        "batchmatmulsparsekg3",
        "batchmatmulfp8sparsekg3",
        "batchmatmulint8sparsekg3",
        "batchmatmulint4sparsekg3",
        "batchmatmulxrf",
        "batchmatmulxrffp8",
        "batchmatmulxrfint8",
        "batchmatmulxrfint4",
        "batchmatmulxrfch",
        "batchmatmulxrfchfp8",
        "batchmatmulxrfchint8",
        "batchmatmulxrfchint4",
        "scaledgroupmatmulfp4",
        "conv2dos1",
        "conv2dgenos1",
        "conv2dint8os1",
        "conv2dxrfint8os1",
        "identity",
        "shuffle",
        "depthwiseconv2dnative",
        "squareddifference",
        "lstmblockcell",
        "qfp8",
        "qfp8ch",
        "qfp8chil",
        "qfp8wt",
        "qfp8mb",
        "csqint8",
        "csqint8v2",
        "csqint8ch",
        "csqint8wt",
        "csqint8chil",
        "csqint8mb",
        "csqint8mbv2",
        "csqint4",
        "csqint4wt",
        "csqint4chil",
        "sfp_readlx_transpose_fwdl0",
        "pt_blk_transpose_load",
        "STCDPOpLx",
        "STCDPOpHBM",
        "ResizeNNHBM",
        "ResizeNNLX",
        "GatherOpHBM",
        "APEOpLX",
        "APEOpHBM",
        "ReStickifyOpLx",
        "ReStickifyOpHBM",
        "ReStickifyOpWithPTLx",
        "ReStickifyOpWithPTHBM",
        "XRFWriteHBM",
        "XRFWriteLX",
        "nop",
        "StickifyOpHBM",
        "ConstPadOpLX",
        "ConstPadOpHBM",
        "PESFP_Collate_2B_Writes",
        "bnpreczeroshft",
        "double_buffer_mni",
        "ScatterOpHBM",
        "ITOF",
        "ITOFHBM",
        "abs",
        "neg",
        "equal",
        "notequal",
        "greaterequal",
        "greaterthan",
        "lesserequal",
        "lesserthan",
        "allgather",
        "allreduce",
        "allshuffle",
        "maximum",
        "minimum",
        "genericpartialreduction",
        "interslicetranspose_fp16",
        "interslicetranspose_fp8",
        "floor",
        "batchmatmulmxfp8",
        "sinkcorrectionfactor",
        "int32idxtoaddr",
        "StzLatch",
        "undef",
    ];

    /// This op-func's spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        Self::SPELLINGS[self as usize]
    }

    /// `EnumsConversion::stringToOpFuncs` — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.spelling() == text)
    }
}

/// The spelling table is POSITIONAL against [`OpFunc::ALL`], so the table and the enum must
/// agree on order - the C++ map is keyed and cannot drift, this one can, so it is pinned.
///
/// 176 names, 176 spellings, and every name has one: `opFuncsToString.at()` cannot throw.
pub const OP_FUNC_SPELLINGS_ARE_IN_ENUM_ORDER: &str =
    "sys-arch-spec/arch_enums.cpp :: EnumsConversion::opFuncsToString";

const _: () = {
    let mut i = 0;
    while i < OpFunc::ALL.len() {
        assert!(
            OpFunc::ALL[i] as usize == i,
            "ALL is out of declaration order"
        );
        i += 1;
    }
    assert!(OpFunc::ALL.len() == OpFunc::SPELLINGS.len());
};

/// A hardware unit, register file, or routing endpoint — `SenComponents` in `arch_enums.h`.
///
/// The discriminants are load-bearing in the C++ (`senCompToRowId` and friends key on them), so they
/// are written out here rather than left implicit.
/// 🔑 `Ord` because two of the C++'s maps are keyed by a `SenComponents` or a pair of them —
/// `syncSendRecvAll`, `syncFifos`, `PtRowSFPFifos` — and a `BTreeMap` needs it. The order is the
/// declaration order, which is `SenComponents`' own numbering; nothing depends on it beyond giving
/// those maps a deterministic iteration.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(i32)]
#[rustfmt::skip]
pub enum SenComponent {
    /// No component.
    NoComponent = -1,
    /// High-bandwidth memory.
    Hbm = 0,
    /// Per-core scratchpad.
    Lx = 1,
    /// Per-corelet scratchpad.
    L0 = 2,
    /// Special function processor.
    Sfp = 3,
    /// Processing element.
    Pe = 4,
    /// Processing tile.
    Pt = 5,
    /// The inter-core ring.
    Ring = 6,
    /// The constant zero source.
    Zero = 7,
    /// Level-3 memory.
    L3 = 8,
    /// L3 load unit.
    L3lu = 9,
    /// L3 store unit.
    L3su = 10,
    /// LX load unit, corelet 0.
    Lxlu0 = 11,
    /// LX store unit, corelet 0.
    Lxsu0 = 12,
    /// LX load unit, corelet 1.
    Lxlu1 = 13,
    /// LX store unit, corelet 1.
    Lxsu1 = 14,
    /// L0 load unit, corelet 0.
    L0lu0 = 15,
    /// L0 store unit, corelet 0.
    L0su0 = 16,
    /// L0 load unit, corelet 1.
    L0lu1 = 17,
    /// L0 store unit, corelet 1.
    L0su1 = 18,
    /// SFP, corelet 0.
    Sfp0 = 19,
    /// SFP, corelet 1.
    Sfp1 = 20,
    /// PE, corelet 0.
    Pe0 = 21,
    /// PE, corelet 1.
    Pe1 = 22,
    /// PT row 0, corelet 0.
    Ptrow0_0 = 23,
    /// PT row 1, corelet 0.
    Ptrow1_0 = 24,
    /// PT row 2, corelet 0.
    Ptrow2_0 = 25,
    /// PT row 3, corelet 0.
    Ptrow3_0 = 26,
    /// PT row 4, corelet 0.
    Ptrow4_0 = 27,
    /// PT row 5, corelet 0.
    Ptrow5_0 = 28,
    /// PT row 6, corelet 0.
    Ptrow6_0 = 29,
    /// PT row 7, corelet 0.
    Ptrow7_0 = 30,
    /// PT row 0, corelet 1.
    Ptrow0_1 = 31,
    /// PT row 1, corelet 1.
    Ptrow1_1 = 32,
    /// PT row 2, corelet 1.
    Ptrow2_1 = 33,
    /// PT row 3, corelet 1.
    Ptrow3_1 = 34,
    /// PT row 4, corelet 1.
    Ptrow4_1 = 35,
    /// PT row 5, corelet 1.
    Ptrow5_1 = 36,
    /// PT row 6, corelet 1.
    Ptrow6_1 = 37,
    /// PT row 7, corelet 1.
    Ptrow7_1 = 38,
    /// The LX load/store FIFO.
    Lxlusufifo = 39,
    /// L0 load unit.
    L0lu = 40,
    /// L0 store unit.
    L0su = 41,
    /// LX load unit.
    Lxlu = 42,
    /// LX store unit.
    Lxsu = 43,
    /// L3 inbound buffer.
    L3ibr = 44,
    /// PT cross-register file.
    Ptxrf = 45,
    /// PT index register file.
    Ptirf = 46,
    /// Generic LRF register file — kept for DSC- and arch-level compatibility. New IR ops use the
    /// unit-prefixed variants [`Self::PeLrfreg`], [`Self::SfpLrfreg`] and [`Self::PtLrfreg`].
    Lrfreg = 47,
    /// PT row 0.
    Ptrow0 = 48,
    /// PT row 1.
    Ptrow1 = 49,
    /// PT row 2.
    Ptrow2 = 50,
    /// PT row 3.
    Ptrow3 = 51,
    /// PT row 4.
    Ptrow4 = 52,
    /// PT row 5.
    Ptrow5 = 53,
    /// PT row 6.
    Ptrow6 = 54,
    /// PT row 7.
    Ptrow7 = 55,
    /// PT northward link.
    Ptnorth = 56,
    /// PT westward link.
    Ptwest = 57,
    /// PT southward link.
    Ptsouth = 58,
    /// The SFP ring.
    Sfpring = 59,
    /// Broadcast to all.
    All = 60,
    /// The constant one source.
    One = 61,
    /// L0 load unit, row 0.
    L0lurow0 = 62,
    /// L0 load unit, row 1.
    L0lurow1 = 63,
    /// L0 load unit, row 2.
    L0lurow2 = 64,
    /// L0 load unit, row 3.
    L0lurow3 = 65,
    /// L0 load unit, row 4.
    L0lurow4 = 66,
    /// L0 load unit, row 5.
    L0lurow5 = 67,
    /// L0 load unit, row 6.
    L0lurow6 = 68,
    /// L0 load unit, row 7.
    L0lurow7 = 69,
    /// A compute latch.
    Latch = 70,
    /// A constant operand.
    Constant = 71,
    /// Neighbour forward 0.
    Nfwd0 = 72,
    /// Neighbour forward 2.
    Nfwd2 = 73,
    /// PE local register file.
    Pelrf = 74,
    /// SFP local register file.
    Sfplrf = 75,
    /// PT accumulator register file.
    Ptarf = 76,
    /// L0 load unit, row 0, corelet 0.
    L0lurow0_0 = 77,
    /// L0 load unit, row 1, corelet 0.
    L0lurow1_0 = 78,
    /// L0 load unit, row 2, corelet 0.
    L0lurow2_0 = 79,
    /// L0 load unit, row 3, corelet 0.
    L0lurow3_0 = 80,
    /// L0 load unit, row 4, corelet 0.
    L0lurow4_0 = 81,
    /// L0 load unit, row 5, corelet 0.
    L0lurow5_0 = 82,
    /// L0 load unit, row 6, corelet 0.
    L0lurow6_0 = 83,
    /// L0 load unit, row 7, corelet 0.
    L0lurow7_0 = 84,
    /// L0 load unit, row 0, corelet 1.
    L0lurow0_1 = 85,
    /// L0 load unit, row 1, corelet 1.
    L0lurow1_1 = 86,
    /// L0 load unit, row 2, corelet 1.
    L0lurow2_1 = 87,
    /// L0 load unit, row 3, corelet 1.
    L0lurow3_1 = 88,
    /// L0 load unit, row 4, corelet 1.
    L0lurow4_1 = 89,
    /// L0 load unit, row 5, corelet 1.
    L0lurow5_1 = 90,
    /// L0 load unit, row 6, corelet 1.
    L0lurow6_1 = 91,
    /// L0 load unit, row 7, corelet 1.
    L0lurow7_1 = 92,
    /// LX virtual inbound buffer.
    Lxvirtualibr = 93,
    /// L3 load-unit inbound buffer.
    L3luibr = 94,
    /// L3 store-unit inbound buffer.
    L3suibr = 95,
    /// SFP state register.
    Sfpstate = 96,
    /// PE state register.
    Pestate = 97,
    /// Cross-PT north link.
    Crossptnlink = 98,
    /// L0 scale storage.
    L0Scale = 99,
    /// LX load-unit scale register.
    Lxluscalereg = 100,
    /// LX load-unit value port.
    Lxluvalue = 101,
    /// PE local register file, unit-prefixed.
    PeLrfreg = 102,
    /// SFP local register file, unit-prefixed.
    SfpLrfreg = 103,
    /// PT local register file, unit-prefixed.
    PtLrfreg = 104,
    /// Queue-group interface.
    Qgi = 105,
}

impl SenComponent {
    /// `EnumsConversion::senComponentsToString` (`sys-arch-spec/arch_enums.cpp:11`).
    ///
    /// 107 names, 107 spellings, and every name has one — verified against the C++ map, so the
    /// reference's `senComponentsToString.at(comp)` cannot throw and this is infallible.
    /// ⛔ `Lxlusufifo` really does spell `"lxlusufifio"`; the typo is the reference's and it is what
    /// reaches the IR text, so it is preserved.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::NoComponent => "no_component",
            Self::Hbm => "hbm",
            Self::Lx => "lx",
            Self::L0 => "l0",
            Self::Sfp => "sfp",
            Self::Pe => "pe",
            Self::Pt => "pt",
            Self::Ring => "ring",
            Self::Zero => "zero",
            Self::L3 => "l3",
            Self::L3lu => "l3lu",
            Self::L3su => "l3su",
            Self::Lxlu0 => "lxlu0",
            Self::Lxsu0 => "lxsu0",
            Self::Lxlu1 => "lxlu1",
            Self::Lxsu1 => "lxsu1",
            Self::L0lu0 => "l0lu0",
            Self::L0su0 => "l0su0",
            Self::L0lu1 => "l0lu1",
            Self::L0su1 => "l0su1",
            Self::Sfp0 => "sfp0",
            Self::Sfp1 => "sfp1",
            Self::Pe0 => "pe0",
            Self::Pe1 => "pe1",
            Self::Ptrow0_0 => "ptrow0_0",
            Self::Ptrow1_0 => "ptrow1_0",
            Self::Ptrow2_0 => "ptrow2_0",
            Self::Ptrow3_0 => "ptrow3_0",
            Self::Ptrow4_0 => "ptrow4_0",
            Self::Ptrow5_0 => "ptrow5_0",
            Self::Ptrow6_0 => "ptrow6_0",
            Self::Ptrow7_0 => "ptrow7_0",
            Self::Ptrow0_1 => "ptrow0_1",
            Self::Ptrow1_1 => "ptrow1_1",
            Self::Ptrow2_1 => "ptrow2_1",
            Self::Ptrow3_1 => "ptrow3_1",
            Self::Ptrow4_1 => "ptrow4_1",
            Self::Ptrow5_1 => "ptrow5_1",
            Self::Ptrow6_1 => "ptrow6_1",
            Self::Ptrow7_1 => "ptrow7_1",
            Self::Lxlusufifo => "lxlusufifio",
            Self::L0lu => "l0lu",
            Self::L0su => "l0su",
            Self::Lxlu => "lxlu",
            Self::Lxsu => "lxsu",
            Self::L3ibr => "l3ibr",
            Self::Ptxrf => "ptxrf",
            Self::Ptirf => "ptirf",
            Self::Lrfreg => "lrfreg",
            Self::Ptrow0 => "ptrow0",
            Self::Ptrow1 => "ptrow1",
            Self::Ptrow2 => "ptrow2",
            Self::Ptrow3 => "ptrow3",
            Self::Ptrow4 => "ptrow4",
            Self::Ptrow5 => "ptrow5",
            Self::Ptrow6 => "ptrow6",
            Self::Ptrow7 => "ptrow7",
            Self::Ptnorth => "ptnorth",
            Self::Ptwest => "ptwest",
            Self::Ptsouth => "ptsouth",
            Self::Sfpring => "sfpring",
            Self::All => "all",
            Self::One => "one",
            Self::L0lurow0 => "l0lurow0",
            Self::L0lurow1 => "l0lurow1",
            Self::L0lurow2 => "l0lurow2",
            Self::L0lurow3 => "l0lurow3",
            Self::L0lurow4 => "l0lurow4",
            Self::L0lurow5 => "l0lurow5",
            Self::L0lurow6 => "l0lurow6",
            Self::L0lurow7 => "l0lurow7",
            Self::Latch => "latch",
            Self::Constant => "constant",
            Self::Nfwd0 => "nfwd0",
            Self::Nfwd2 => "nfwd2",
            Self::Pelrf => "pelrf",
            Self::Sfplrf => "sfplrf",
            Self::Ptarf => "ptarf",
            Self::L0lurow0_0 => "l0lurow0_0",
            Self::L0lurow1_0 => "l0lurow1_0",
            Self::L0lurow2_0 => "l0lurow2_0",
            Self::L0lurow3_0 => "l0lurow3_0",
            Self::L0lurow4_0 => "l0lurow4_0",
            Self::L0lurow5_0 => "l0lurow5_0",
            Self::L0lurow6_0 => "l0lurow6_0",
            Self::L0lurow7_0 => "l0lurow7_0",
            Self::L0lurow0_1 => "l0lurow0_1",
            Self::L0lurow1_1 => "l0lurow1_1",
            Self::L0lurow2_1 => "l0lurow2_1",
            Self::L0lurow3_1 => "l0lurow3_1",
            Self::L0lurow4_1 => "l0lurow4_1",
            Self::L0lurow5_1 => "l0lurow5_1",
            Self::L0lurow6_1 => "l0lurow6_1",
            Self::L0lurow7_1 => "l0lurow7_1",
            Self::Lxvirtualibr => "lxvirtualibr",
            Self::L3luibr => "l3luibr",
            Self::L3suibr => "l3suibr",
            Self::Sfpstate => "sfpstate",
            Self::Pestate => "pestate",
            Self::Crossptnlink => "crossptnlink",
            Self::L0Scale => "l0scale",
            Self::Lxluscalereg => "lxluscalereg",
            Self::Lxluvalue => "lxluvalue",
            Self::PeLrfreg => "pe_lrfreg",
            Self::SfpLrfreg => "sfp_lrfreg",
            Self::PtLrfreg => "pt_lrfreg",
            Self::Qgi => "qgi",
        }
    }

    /// `EnumsConversion::senCompToGenericComp` (`sys-arch-spec/arch_enums.cpp:123-211`) — the
    /// component a numbered, per-row or per-corelet instance STANDS FOR, and itself for one that is
    /// already generic.
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `.at()` THROW. 20 of the 107 components are not keys of that
    /// map — `NO_COMPONENT`, `L0`, `L3`, `RING`, `ZERO`, `LXLUSUFIFO`, `PTIRF`, the three PT compass
    /// ports, `SFPRING`, `ALL`, `ONE`, `LATCH`, `CONSTANT`, `NFWD0`, `NFWD2`, `L3LUIBR`, `L3SUIBR`
    /// and `L0_SCALE` — and asking it for one aborts.
    #[must_use]
    pub const fn generic(self) -> Option<Self> {
        Some(match self {
            Self::Sfp0 | Self::Sfp1 | Self::Sfp => Self::Sfp,
            Self::Ptrow0_0
            | Self::Ptrow1_0
            | Self::Ptrow2_0
            | Self::Ptrow3_0
            | Self::Ptrow4_0
            | Self::Ptrow5_0
            | Self::Ptrow6_0
            | Self::Ptrow7_0
            | Self::Ptrow0_1
            | Self::Ptrow1_1
            | Self::Ptrow2_1
            | Self::Ptrow3_1
            | Self::Ptrow4_1
            | Self::Ptrow5_1
            | Self::Ptrow6_1
            | Self::Ptrow7_1
            | Self::Ptrow0
            | Self::Ptrow1
            | Self::Ptrow2
            | Self::Ptrow3
            | Self::Ptrow4
            | Self::Ptrow5
            | Self::Ptrow6
            | Self::Ptrow7
            | Self::Pt => Self::Pt,
            Self::Lrfreg | Self::PeLrfreg | Self::SfpLrfreg | Self::PtLrfreg => Self::Lrfreg,
            Self::Sfplrf => Self::Sfplrf,
            Self::Pelrf => Self::Pelrf,
            Self::Sfpstate => Self::Sfpstate,
            Self::Pestate => Self::Pestate,
            Self::Ptarf => Self::Ptarf,
            Self::Ptxrf => Self::Ptxrf,
            Self::Pe0 | Self::Pe1 | Self::Pe => Self::Pe,
            Self::Lxlu0 | Self::Lxlu1 | Self::Lxlu => Self::Lxlu,
            Self::Lxluscalereg => Self::Lxluscalereg,
            Self::Lxluvalue => Self::Lxluvalue,
            Self::Lxsu0 | Self::Lxsu1 | Self::Lxsu => Self::Lxsu,
            Self::Lx => Self::Lx,
            Self::L0lu0
            | Self::L0lu1
            | Self::L0lurow0
            | Self::L0lurow1
            | Self::L0lurow2
            | Self::L0lurow3
            | Self::L0lurow4
            | Self::L0lurow5
            | Self::L0lurow6
            | Self::L0lurow7
            | Self::L0lurow0_0
            | Self::L0lurow1_0
            | Self::L0lurow2_0
            | Self::L0lurow3_0
            | Self::L0lurow4_0
            | Self::L0lurow5_0
            | Self::L0lurow6_0
            | Self::L0lurow7_0
            | Self::L0lurow0_1
            | Self::L0lurow1_1
            | Self::L0lurow2_1
            | Self::L0lurow3_1
            | Self::L0lurow4_1
            | Self::L0lurow5_1
            | Self::L0lurow6_1
            | Self::L0lurow7_1
            | Self::L0lu => Self::L0lu,
            Self::L0su0 | Self::L0su1 | Self::L0su => Self::L0su,
            Self::L3lu => Self::L3lu,
            Self::L3su => Self::L3su,
            Self::Hbm => Self::Hbm,
            Self::L3ibr => Self::L3ibr,
            Self::Lxvirtualibr => Self::Lxvirtualibr,
            Self::Crossptnlink => Self::Crossptnlink,
            Self::Qgi => Self::Qgi,
            Self::NoComponent
            | Self::Ring
            | Self::Zero
            | Self::L3
            | Self::L0
            | Self::Lxlusufifo
            | Self::Ptirf
            | Self::Ptnorth
            | Self::Ptwest
            | Self::Ptsouth
            | Self::Sfpring
            | Self::All
            | Self::One
            | Self::Latch
            | Self::Constant
            | Self::Nfwd0
            | Self::Nfwd2
            | Self::L3luibr
            | Self::L3suibr
            | Self::L0Scale => return None,
        })
    }
}

/// A SEGMENT OF THE LDS — `LdsSegment` (`arch_enums.h:338`), in BOTH of its naming schemes.
///
/// 🔑 The C++ writes out forty enumerators; here the eight ROLE-named ones are variants and the
/// thirty-two POSITION-named ones are the [`HmiSegment`] grid, so which scheme a value belongs to is
/// the variant rather than a range check on an integer — see [`TWO_SEGMENT_SCHEMES_IN_ONE_ENUM`].
///
/// ⭐ IT IS A MAP KEY: `MemTrackBundle::hbmTrack` is a `std::map<LdsSegment, DsTrackInMem>`
/// (`sys-arch-spec/memtracker/mem_track_bundle.h:22`) whose `begin()` is read for the phase count
/// (`mem_track_bundle.cpp:52`), which is why [`Ord`] here is the enumerator order and not derived.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum LdsSegment {
    /// `OUTPUT`.
    Output,
    /// `INPUT`.
    Input,
    /// `STACK`.
    Stack,
    /// `MODEL`.
    Model,
    /// `HEAP`.
    Heap,
    /// `RESERVE1` — one of the two placeholders the position scheme does not need.
    Reserve1,
    /// `RESERVE2`.
    Reserve2,
    /// `CONST`.
    Const,
    /// One of the thirty-two `SEGn_HMIm` segments.
    Positioned(HmiSegment),
}

impl LdsSegment {
    /// The eight role-named segments, in declaration order.
    pub const BY_ROLE: [Self; SegmentScheme::ByRole.enumerators()] = [
        Self::Output,
        Self::Input,
        Self::Stack,
        Self::Model,
        Self::Heap,
        Self::Reserve1,
        Self::Reserve2,
        Self::Const,
    ];

    /// Its index in the C++ enum — what a `std::map<LdsSegment, ..>` orders by.
    #[must_use]
    pub const fn enum_index(self) -> usize {
        match self {
            Self::Output => 0,
            Self::Input => 1,
            Self::Stack => 2,
            Self::Model => 3,
            Self::Heap => 4,
            Self::Reserve1 => 5,
            Self::Reserve2 => 6,
            Self::Const => 7,
            Self::Positioned(segment) => segment.enum_index(),
        }
    }

    /// Which naming scheme it belongs to.
    #[must_use]
    pub const fn scheme(self) -> SegmentScheme {
        match self {
            Self::Positioned(_) => SegmentScheme::ByPosition,
            _ => SegmentScheme::ByRole,
        }
    }
}

/// 🔑 THE ORDER IS THE C++ ENUM'S OWN, NOT [`HmiSegment`]'S FIELD ORDER. A derived `Ord` would order
/// the position-named segments segment-major over `(segment, hmi)`, where the C++ declares them
/// hmi-major (`SEG0_HMI0 .. SEG3_HMI0, SEG0_HMI1 ..`), so a `BTreeMap` keyed by this would iterate in
/// a different order than the `std::map` it stands for.
impl Ord for LdsSegment {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.enum_index().cmp(&other.enum_index())
    }
}

impl PartialOrd for LdsSegment {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

const _: () = assert!(
    LdsSegment::BY_ROLE.len() + SegmentScheme::ByPosition.enumerators() == LDS_SEGMENTS,
    "the role variants plus the HMI grid are the whole enum"
);
