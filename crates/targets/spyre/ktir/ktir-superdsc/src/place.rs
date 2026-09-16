// SPDX-License-Identifier: Apache-2.0
//! OPERAND IDENTITY — what a lowered operand IS, and the one site that spells it.
//!
//! ⛔ THIS IS THE LOWERING'S OWN VOCABULARY, WHICH IS WHY IT IS HERE AND NOT IN THE BUNDLE. Every
//! [`SynthRole`] names an intermediate *this lowering invents* (rope's `rot`/`xc`/`rs`, the online
//! softmax's `newkt`/`bmax`/`corr`, the fp8 activation quantizer's `fq_*`). A third-party KTIR
//! producer does not supply them and cannot: they exist only because a node decomposes into
//! several device ops. So the vocabulary travels with the decomposition: a consumer's bake crate,
//! which needs a placement for each one, re-exports these two types from here rather than declaring
//! its own, so the bake and the lowering cannot drift.
//!
//! ⭐ AND IT IS THE NAMING DOOR the extraction needed. `superdsc` references an operand by NAME, so
//! something must render one; [`PlaceId`]'s [`Display`](std::fmt::Display) is that one site, and
//! `syn` is the one call to it. Nothing parses the rendering back — the identity is the value.

use std::fmt;

/// What a synthetic intermediate IS — the role it plays for the tensor it derives from.
///
/// ⛔ THESE ARE NOT NAME SUFFIXES. Every one of these was a `format!("{out}_rot")`-style
/// string built at one site and matched at another, which is a join on a spelling. The role
/// is a fact about the lowering, so it is a value; the spelling exists only in [`Display`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SynthRole {
    // ── rope ──
    Rot,
    Xc,
    Rs,
    // ── ffn ──
    Silu,
    // ── attention scratch ──
    Qs,
    KRep,
    VRep,
    NewKRep,
    NewVRep,
    NewKScaled,
    // ── fp8 activation quantization ──
    FqAbsX,
    FqAfp8,
    FqAmax,
    FqAmaxFl,
    FqAscale,
    FqChi,
    FqCl,
    FqInvS,
    FqSc,
    FqDqA,
    FqMm,
    FqRaw,
    // ── rmsnorm scratch ──
    Sq16,
    Mean,
    Meps,
    Rinv,
    Xn,
    // ── flash-attention online-softmax state ──
    NewKt,
    Sc,
    BMax,
    NewM,
    Corr,
    CorrSubT,
    ExpB,
    ESubT,
    BSum,
    OTmp,
    LTmp,
    RunM,
    RunL,
    RunO,
    // ── K-split / down-projection blocking (the only INDEXED roles) ──
    Blk(u32),
    Acc(u32),
    LBlk(u32),
    LAcc(u32),
}

impl fmt::Display for SynthRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rot => f.write_str("rot"),
            Self::Xc => f.write_str("xc"),
            Self::Rs => f.write_str("rs"),
            Self::Silu => f.write_str("silu"),
            Self::Qs => f.write_str("qs"),
            Self::KRep => f.write_str("krep"),
            Self::VRep => f.write_str("vrep"),
            Self::NewKRep => f.write_str("newkrep"),
            Self::NewVRep => f.write_str("newvrep"),
            Self::NewKScaled => f.write_str("newkscaled"),
            Self::FqAbsX => f.write_str("fq_absx"),
            Self::FqAfp8 => f.write_str("fq_afp8"),
            Self::FqAmax => f.write_str("fq_amax"),
            Self::FqAmaxFl => f.write_str("fq_amaxfl"),
            Self::FqAscale => f.write_str("fq_ascale"),
            Self::FqChi => f.write_str("fq_chi"),
            Self::FqCl => f.write_str("fq_cl"),
            Self::FqInvS => f.write_str("fq_invs"),
            Self::FqSc => f.write_str("fq_sc"),
            Self::FqDqA => f.write_str("fq_dqa"),
            Self::FqMm => f.write_str("fq_mm"),
            Self::FqRaw => f.write_str("fq_raw"),
            Self::Sq16 => f.write_str("sq16"),
            Self::Mean => f.write_str("mean"),
            Self::Meps => f.write_str("meps"),
            Self::Rinv => f.write_str("rinv"),
            Self::Xn => f.write_str("xn"),
            Self::NewKt => f.write_str("newkt"),
            Self::Sc => f.write_str("sc"),
            Self::BMax => f.write_str("bmax"),
            Self::NewM => f.write_str("newm"),
            Self::Corr => f.write_str("corr"),
            Self::CorrSubT => f.write_str("corrsubt"),
            Self::ExpB => f.write_str("expb"),
            Self::ESubT => f.write_str("esubt"),
            Self::BSum => f.write_str("bsum"),
            Self::OTmp => f.write_str("otmp"),
            Self::LTmp => f.write_str("ltmp"),
            Self::RunM => f.write_str("m"),
            Self::RunL => f.write_str("l"),
            Self::RunO => f.write_str("o"),
            Self::Blk(i) => write!(f, "blk{i}"),
            Self::Acc(i) => write!(f, "acc{i}"),
            Self::LBlk(i) => write!(f, "lblk{i}"),
            Self::LAcc(i) => write!(f, "lacc{i}"),
        }
    }
}

/// A placed tensor's IDENTITY — the join key between what the bake placed and what the host binds.
///
/// ⛔ THIS REPLACES `t{tid}`. The old form carried `name: Cow<str>` *and* `tid: Option<u32>`: the
/// name was `format!("t{tid}")` for a real tensor, so one field was a rendering of the other, and
/// every bind was a string round-trip through a spelling both ends had to agree on by convention.
/// Nothing ever PARSED it back — the string existed only to be a map key — so the key is now the
/// number it was always standing in for.
///
/// [`Synth`] keeps a place here because a synthetic occupies segment space and the alias-equality
/// check must see it; it is never bound, which is why it needs no separate tid.
///
/// [`Synth`]: Self::Synth
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaceId {
    /// A buffer the CALLER numbered — the only kind the host ever binds.
    ///
    /// The number is the caller's, carried in on
    /// [`KtirNode::bindings`](crate::ktir_node::KtirNode::bindings) as a
    /// [`BufferId`](crate::ktir_node::BufferId); this crate only names it and looks it up in the
    /// caller's [`BundleLayout`](crate::placement::BundleLayout). Whatever the number MEANS to the
    /// producer — a graph tensor index, a Triton buffer slot — is a fact of that side of the door and
    /// not of this type.
    Act(u32),
    /// An intermediate the lowering invented, derived from `of`. Device-internal.
    Synth { of: u32, role: SynthRole },
}

impl PlaceId {
    /// The caller-numbered buffer this names, or that it derives from.
    pub fn tid(self) -> u32 {
        match self {
            Self::Act(t) | Self::Synth { of: t, .. } => t,
        }
    }
    /// Is this a tensor the host may bind? False for a device-internal synthetic.
    pub fn is_bindable(self) -> bool {
        matches!(self, Self::Act(_))
    }
    /// The id the host binds under, or `None` for a device-internal synthetic.
    pub fn bindable(self) -> Option<u32> {
        match self {
            Self::Act(t) => Some(t),
            Self::Synth { .. } => None,
        }
    }
    /// Derive a synthetic from this tensor.
    pub fn synth(self, role: SynthRole) -> Self {
        Self::Synth {
            of: self.tid(),
            role,
        }
    }
}

impl fmt::Display for PlaceId {
    /// THE ONE SPELLING SITE. For diagnostics and the emitted program's own operand references;
    /// nothing parses this back, so it is a rendering, not an encoding.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Act(t) => write!(f, "t{t}"),
            Self::Synth { of, role } => write!(f, "t{of}_{role}"),
        }
    }
}

/// The emitted PROGRAM's operand spelling for tensor `tid`.
///
/// ⛔ THIS IS A RENDERING, NOT AN IDENTITY. The identity is
/// [`PlaceId`], and that is what the bake and the host now join on —
/// nothing parses this string back. It survives only because a SuperDSC op
/// references its operands by name, so the emitted program needs a spelling;
/// [`PlaceId`]'s `Display` is where that spelling is decided, and this
/// is the one call to it.
#[inline]
pub fn act_name(tid: u32) -> String {
    PlaceId::Act(tid).to_string()
}
