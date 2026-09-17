// SPDX-License-Identifier: Apache-2.0
//! Typed FRONTEND core for the SuperDSC (SDSC) emitter — a Rust port of
//! torch-spyre's `torch_spyre/_inductor/op_spec.py` (`OpSpec`/`TensorArg`) and
//! the FRONTEND half of `codegen/compute_ops.py`.
//!
//! The whole point of this module is to make the historical SuperDSC bug
//! classes UNREPRESENTABLE rather than runtime-checked:
//!
//! * (a) a sub-stick split of the matmul N (output stick) / K (reduction stick)
//!   extents — [`StickExtent`] can only be constructed for a multiple of the
//!   data-format stick (`fp16 = 64`), and the work-division divides its STICK
//!   COUNT, so a per-core extent is a whole multiple of 64 by arithmetic.
//! * (b)/(f) over-subscription of the 32 cores or a frontend corelet split —
//!   [`WorkPlan`] is a SEALED proof produced ONLY by [`WorkPlan::divide`], which
//!   asserts `product(splits) ≤ 32` and `≥1 stick/core` for every stick dim, and
//!   carries no corelet field at all (corelet is the const [`ACTIVE_CORELETS`]).
//! * (c) a `startAddr` fold whose `[core,corelet]` factors disagree with the
//!   declared `coreFoldProp_`/`coreletFoldProp_` — [`SdscFoldSet`] is the single
//!   source of truth for all fold factors and [`AddrFold::new`] copies them from
//!   it; there is no public field setter to diverge them.
//! * (d) a `scale_`/`device_dims`/layout rank mismatch — [`TensorArg`] is generic
//!   over the device rank `const D: usize`, so `scale: [Scale; D]`,
//!   `device_dims: [IterSym; D]` and the layout `[&'static str; D]` share the
//!   SAME `D` (a mismatch is a compile error). Erased via [`AnyTensorArg`].
//! * (e) an `IterSym` that is not a key of the op's iteration space — `IterSym`
//!   is ONLY mintable from a [`WorkPlan`].
//! * (f)/(#6) a hand-set `exUnit` on a matmul — [`ExUnit`] is a sealed newtype
//!   produced ONLY by [`OpFunc::ex_unit`].
//! * register-file `memOrg_` (the DtException 1535 root cause) — [`MemOrg`] is a
//!   sealed struct with exactly `hbm`/`lx`; there is no API to add a register
//!   file in the frontend.
//!
//! The typed core does NOT derive `Serialize`. The wire structs in [`crate::wire`] do;
//! [`emit_sdsc`](crate::emit::emit_sdsc) is the sole bridge. (That module used to be in the target
//! crate this leaf must not depend on, so this could not be a link; it has since moved in, so it is
//! one again.)

#![allow(non_snake_case)]

use crate::superdsc_error::SuperDscError;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::num::NonZeroU32;

// ───────────────────────────────────────────────────────────────────────────
// Hardware constants (dd2). Single source of truth shared with the wire module.
// ───────────────────────────────────────────────────────────────────────────

/// Spyre dd2: 32 cores.
pub const MAX_CORES: u32 = 32;
/// Corelets declared per core in the FRONTEND SDSC = 1 (matches torch-spyre's
/// `compute_ops.py` `coreletFoldProp_`={factor_:1} / `numCoreletsUsed_`:1). The
/// L3 scheduler decides the real corelet split — the frontend does NOT
/// pre-specify it. Promoted from a convention to a const the [`WorkPlan`] cannot
/// influence (witness (f)): there is no corelet field anywhere in the frontend.
pub const ACTIVE_CORELETS: u32 = 1;

// ───────────────────────────────────────────────────────────────────────────
// LX scratchpad capacity — the binding limit for the time-tiling (coarse_tile)
// pass. SINGLE SOURCE OF TRUTH, shared with the wire module. A per-core tile
// that does not fit `USABLE_LX_BYTES` is an on-card DtException 1535
// (register-file / scratchpad over-subscription, L3DlOpsScheduler) — the whole
// point of the typed [`TimeTile`] witness is to turn that into a `cargo build`
// `Err` BEFORE the bundle is ever baked.
// ───────────────────────────────────────────────────────────────────────────

/// LX scratchpad bytes PER CORE on dd2 = 2 MiB. Confirmed against torch-spyre
/// `scratchpad/allocator.py:305` (`size = int((2 << 20) * (1.0 - dxp_lx_frac_avail))`)
/// and `sentient_dd2_sysconfig.json`.
pub const LX_CAPACITY_BYTES: u64 = 2 * 1024 * 1024;

/// The DXP-reserved fraction of the LX scratchpad (torch-spyre
/// `config.py:41 dxp_lx_frac_avail = 0.2`). The frontend may only PLAN against
/// `1 - DXP_LX_FRAC_AVAIL` of the pad; the rest is the toolchain's working set.
pub const DXP_LX_FRAC_AVAIL: f64 = 0.2;

/// LX bytes the frontend may PLAN a per-core tile into. A clearly-named TUNABLE
/// const so a future double-buffer ping-pong (halve) or 1× start (full pad) is a
/// one-line change, not a scattered float at the call site. Today =
/// `floor(LX_CAPACITY_BYTES * (1 - DXP_LX_FRAC_AVAIL))` = `floor(2 MiB * 0.8)` =
/// `1_677_721` B, mirroring torch-spyre `allocator.py:305` exactly. (Start the
/// fit check at this 0.8× value; if on-card still hits 1535, halve it for the
/// double-buffer overlap — a tunable, not a structural change. See design risk
/// #3.)
pub const USABLE_LX_BYTES: u64 = 1_677_721;

/// fp16 = 2 bytes / element. The emitter is fp16-only ([`Fp16::WORD_LENGTH`]);
/// this alias documents the residency-byte arithmetic in [`WorkPlan::time_tile_for_lx`].
pub const FP16_BYTES: u64 = 2;

// ───────────────────────────────────────────────────────────────────────────
// (a) DataFormat + StickExtent — the fp16-stick / multiple-of-stick witness.
// ───────────────────────────────────────────────────────────────────────────

/// Sealed data-format trait carrying the per-stick element count. fp16 = 64
/// elems / 128 B stick. (fp32 = 32, fp8/int8 = 128 — present for completeness;
/// the emitter is fp16-only today.)
pub trait DataFormat: private::SealedDf {
    const ELEMS_PER_STICK: u32;
    /// DeepTools `dataFormat_` string.
    const NAME: &'static str;
    /// `wordLength` (bytes per element). 128-byte stick / elems-per-stick.
    const WORD_LENGTH: u32;
    /// The runtime [`Df`] this type-level marker corresponds to — the bridge that
    /// lets a `DF`-generic emitter stamp the erased `ItDim`/`TensorArg` fields from
    /// its COMPILE-TIME format, so the stick basis a work-division sees can only be
    /// this format's (fp8 ⇒ 128), never a hand-picked constant.
    const DF: Df;
}

/// fp16 — the storage/compute format for the matmul PE + activations. 64 elems / 128 B stick.
#[derive(Clone, Copy, Debug)]
pub enum Fp16 {}
impl DataFormat for Fp16 {
    const ELEMS_PER_STICK: u32 = 64;
    const NAME: &'static str = "SEN169_FP16";
    const WORD_LENGTH: u32 = 2;
    const DF: Df = Df::Fp16;
}

/// fp32 (IEEE_FP32) — 32 elems / 128 B stick, 4 bytes/elem. Used ONLY by the fp32-SFP-merge split-K
/// accumulation path: the matmul PE stays fp16 (RCUDD1A has no fp32 PE — SEN1P5-only), but the split-K
/// partials are merged with fp32 SFP `add` ops (`broadcast_ops.ddl` binds "add" for [fp16, fp32]) so the
/// accumulation approaches the fp32 golden (metal `gemm.metal c_scratch: float`). 4-byte, so the byte/stride
/// math must NOT reuse `Fp16::WORD_LENGTH` for fp32 tensors (the `_bf16`-marker shortcut fails here).
#[derive(Clone, Copy, Debug)]
pub enum Fp32 {}
impl DataFormat for Fp32 {
    const ELEMS_PER_STICK: u32 = 32;
    const NAME: &'static str = "IEEE_FP32";
    const WORD_LENGTH: u32 = 4;
    const DF: Df = Df::Fp32;
}

/// SENINT8 — signed 8-bit integer weight residency (torch-spyre `DataFormats::SENINT8`,
/// `csrc/module.cpp:303`). **128 elems / 128-byte stick, 1 byte/elem** — exactly HALF the
/// fp16 residency (64 elems / 128-byte stick, 2 bytes/elem), which IS the decode
/// HBM-bandwidth win: the weight stays PACKED (1 byte) in HBM and is DMA'd packed, then
/// dequantized ON-CARD (SFP `int8·scale→f16`, or a SENINT8 PE-matmul input — both listed
/// supported: `dataflow_architecture.md:162,268` "int8 on SFP+PE"; the int8 stick is 128
/// elems / 128 B per `work_division_planning.md:118 device_dtype.elems_per_stick()`).
/// A resident int8 weight that instead materialized f16-dense (2 bytes) would be the
/// REJECTED dequant-on-load reward-hack — `senint8_resident_is_half_fp16_footprint`
/// (in the sdsc crate) is the fail-first guard against exactly that (RED if WORD_LENGTH==2).
#[derive(Clone, Copy, Debug)]
pub enum SenInt8 {}
impl DataFormat for SenInt8 {
    const ELEMS_PER_STICK: u32 = 128;
    const NAME: &'static str = "SENINT8";
    const WORD_LENGTH: u32 = 1; // 1 byte/elem — the packed int8 residency (½ f16 = the bandwidth win)
    const DF: Df = Df::SenInt8;
}

/// SEN143_FP8 — E4M3 (1-4-3) 8-bit float weight/activation residency (torch-spyre `DataFormats`, hw doc
/// "int/fp8 on SFP+PE"; DDL `%type_fp8 = SEN143_FP8`). 128 elems / 128-byte stick, 1 byte/elem — HALF the
/// fp16 residency (the ÷2 decode-bandwidth win), same stick geometry as [`SenInt8`]. IBM PR #2401 (89ac601)
/// wired the fp8 quant/dequant (qfp8ch, fp8todl16, quantize/dequantize_fp8_with_scale); the DDL
/// `matmulfp8`/`batchmatmulfp8` kernels (bmm.ddl:55,60) consume fp8×fp8 → fp16 (%ptsum_fp accumulate). A
/// tensor's fp8-ness is the arg's typed [`Df::Fp8`] (`TensorArg::with_df`), driving its `wordLength` /
/// `stickSize_` / `dataFormat_` — NOT a `_fp8` name substring.
#[derive(Clone, Copy, Debug)]
pub enum Fp8 {}
impl DataFormat for Fp8 {
    const ELEMS_PER_STICK: u32 = 128;
    const NAME: &'static str = "SEN143_FP8";
    const WORD_LENGTH: u32 = 1; // 1 byte/elem — packed fp8 residency (½ f16 = the ÷2 bandwidth win)
    const DF: Df = Df::Fp8;
}

mod private {
    /// Seals [`DataFormat`](super::DataFormat) so only the formats defined in this
    /// module (`Fp16`, `Fp32`, `SenInt8`) can implement it. `Scale`/`OpFunc` are
    /// sealed by being concrete enums — no external crate can add a variant.
    pub trait SealedDf {}
    impl SealedDf for super::Fp16 {}
    impl SealedDf for super::Fp32 {}
    impl SealedDf for super::SenInt8 {}
    impl SealedDf for super::Fp8 {}
}

/// **Value-level** mirror of the [`DataFormat`] sealed marker trait — the ONE dtype a dataspace
/// carries, threaded on [`TensorArg`]/[`ArgView`] so per-tensor dtype is a TYPE the emitter reads,
/// not a substring of the dataspace name. Every constant delegates to the corresponding type-level
/// `DataFormat` impl, so the numbers live in exactly one place (the impls above); this enum is the
/// total by-value dispatch onto them. It REPLACES the old `name.contains("_fp8")` string-sniffing
/// (`word_length_for` / `stick_elems_for` / `dataformat_for`): a missing or mistyped name suffix can
/// no longer silently downgrade a tensor to fp16 (½-residency weight read as 2-byte, `matmulfp8` read
/// as `matmul`) — the dtype is declared once, at the arg, and checked by the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Df {
    /// SEN169_FP16 — the default storage/compute format (matmul PE + activations). 2-byte / 64-stick.
    Fp16,
    /// IEEE_FP32 — 4-byte / 32-stick. The split-K fp32-SFP-merge partials only.
    Fp32,
    /// SEN143_FP8 (E4M3) — 1-byte / 128-stick. Packed fp8 weight/activation (½ fp16 = the ÷2 bandwidth win).
    Fp8,
    /// SENINT8 — 1-byte / 128-stick. Packed int8 weight residency (½ fp16).
    SenInt8,
    /// BF16E — 2-byte / 64-stick (fp16 geometry, wider exponent). DORMANT: no emitter site creates a
    /// bf16 dataspace today (the score path stays SEN169_FP16; bf16-output matmul is dxp-rejected). Kept
    /// representable for the `lower_attn_node` bf16-operand check.
    Bf16,
}

impl Df {
    /// Elements per 128-byte stick — from the type-level [`DataFormat`] impl (single source of truth).
    pub const fn elems_per_stick(self) -> u32 {
        match self {
            Df::Fp16 => <Fp16 as DataFormat>::ELEMS_PER_STICK,
            Df::Fp32 => <Fp32 as DataFormat>::ELEMS_PER_STICK,
            Df::Fp8 => <Fp8 as DataFormat>::ELEMS_PER_STICK,
            Df::SenInt8 => <SenInt8 as DataFormat>::ELEMS_PER_STICK,
            Df::Bf16 => <Fp16 as DataFormat>::ELEMS_PER_STICK, // BF16E shares fp16 geometry (2-byte / 64-stick)
        }
    }
    /// Bytes per element (`wordLength`) — from the type-level [`DataFormat`] impl.
    pub const fn word_length(self) -> u32 {
        match self {
            Df::Fp16 => <Fp16 as DataFormat>::WORD_LENGTH,
            Df::Fp32 => <Fp32 as DataFormat>::WORD_LENGTH,
            Df::Fp8 => <Fp8 as DataFormat>::WORD_LENGTH,
            Df::SenInt8 => <SenInt8 as DataFormat>::WORD_LENGTH,
            Df::Bf16 => <Fp16 as DataFormat>::WORD_LENGTH,
        }
    }
    /// DeepTools `dataFormat_` string — from the type-level [`DataFormat`] impl (`Bf16` has no marker type).
    pub const fn dataformat(self) -> &'static str {
        match self {
            Df::Fp16 => <Fp16 as DataFormat>::NAME,
            Df::Fp32 => <Fp32 as DataFormat>::NAME,
            Df::Fp8 => <Fp8 as DataFormat>::NAME,
            Df::SenInt8 => <SenInt8 as DataFormat>::NAME,
            Df::Bf16 => "BF16E",
        }
    }
}

/// A stick-axis extent whose ONLY constructor succeeds iff `elems` is a whole
/// multiple of `DF::ELEMS_PER_STICK` (witness (a)). Matmul N (output stick) and
/// K (reduction stick) extents are wrapped in this, so a non-multiple-of-64 N/K
/// cannot reach the splitter (it surfaces as a `cargo`-level lowering `Err`, not
/// an on-card DtException 1535 / L3DlOpsScheduler:1040).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StickExtent<DF: DataFormat> {
    elems: u32,
    _df: PhantomData<DF>,
}

impl<DF: DataFormat> StickExtent<DF> {
    /// Build a stick extent. `Err` iff `elems` is not a whole multiple of the
    /// stick — this is the (a) guard, surfaced at SubtileIR-lowering time.
    pub fn new(elems: u32) -> Result<Self, String> {
        if elems == 0 || !elems.is_multiple_of(DF::ELEMS_PER_STICK) {
            return Err(format!(
                "stick extent {elems} is not a whole multiple of the {}-elem {} stick \
                 (sub-stick tiles are rejected by the dxp scheduler, L3DlOpsScheduler:1040)",
                DF::ELEMS_PER_STICK,
                DF::NAME
            ));
        }
        Ok(StickExtent {
            elems,
            _df: PhantomData,
        })
    }

    /// Element extent.
    pub fn elems(&self) -> u32 {
        self.elems
    }

    /// Stick count = `elems / ELEMS_PER_STICK` (≥1 by the ctor invariant).
    pub fn stick_count(&self) -> u32 {
        self.elems / DF::ELEMS_PER_STICK
    }
}

/// The device TILE layout of a tensor — the SINGLE typed source from which the
/// per-core start stride (emitter `per_core_addr`) AND the host re-tile descriptor
/// (the C++ shim's weight staging, via the manifest) BOTH derive, so they cannot
/// silently disagree. That disagreement was the matmul bug: scratchy uploaded the
/// weight row-major flat while the per-core address used a row-major stride, but the
/// PT systolic array reads the DEVICE tile layout — a `[a,b]` weight sticked on `b`
/// lives on-device as `[b/STICK, a, STICK]` (the b-stick-tile OUTERMOST), confirmed
/// against torch-spyre `SpyreTensorLayout` / `test_tensor_layout` ([512,256] fp16 →
/// device_size [4,512,64]). Routing both consumers through this witness makes the
/// "host re-tile ⟺ per-core stride" mismatch unrepresentable (the silent-numeric
/// analogue of the StickExtent DtException guard). The stick extent is validated via
/// [`StickExtent`] so a non-stick-multiple is a `cargo` Err, never an on-card fault.
#[derive(Clone, Debug)]
pub struct DeviceTileLayout<DF: DataFormat> {
    /// Host dim names in layout order (e.g. `["in","out"]` for a matmul KERNEL).
    layout: Vec<&'static str>,
    /// Innermost stick axis (one of `layout`).
    stick: &'static str,
    /// Host extents in layout order.
    host_size: Vec<u64>,
    stick_idx: usize,
    _df: PhantomData<DF>,
}

impl<DF: DataFormat> DeviceTileLayout<DF> {
    /// Sole constructor. `Err` (a `cargo build` failure) if the stick axis is absent
    /// from `layout`, ranks mismatch, or the stick extent is not a whole multiple of
    /// the stick (the last routed through [`StickExtent`], reusing that witness).
    pub fn new(
        layout: &[&'static str],
        stick: &'static str,
        host_size: &[u64],
    ) -> Result<Self, SuperDscError> {
        if layout.len() != host_size.len() {
            return Err(SuperDscError(format!(
                "DeviceTileLayout: layout {layout:?} (rank {}) and host_size {host_size:?} \
                 (rank {}) differ",
                layout.len(),
                host_size.len()
            )));
        }
        let stick_idx = layout.iter().position(|&d| d == stick).ok_or_else(|| {
            SuperDscError(format!(
                "DeviceTileLayout: stick '{stick}' not in layout {layout:?}"
            ))
        })?;
        StickExtent::<DF>::new(host_size[stick_idx] as u32).map_err(SuperDscError)?;
        Ok(Self {
            layout: layout.to_vec(),
            stick,
            host_size: host_size.to_vec(),
            stick_idx,
            _df: PhantomData,
        })
    }

    /// The scratchy `per_core_addr` multiplier for dim `d`: the device per-ELEMENT
    /// stride such that `off += slice_idx · per_core_extent(d) · THIS` lands a core on
    /// the start of its device tile. For the STICK dim, its tile axis is OUTERMOST
    /// on-device, so advancing one stick element crosses the whole inner tile → the
    /// product of all NON-stick host extents (KERNEL `[in,out]` stick=out → `in`).
    /// For a non-stick dim it is the row-major inner product (extents to its right),
    /// unchanged from the pre-fix behaviour so activation INPUT/OUTPUT addresses (m=1)
    /// don't shift. THIS is the matmul cos→1.0 fix, now the sole producer of the stride.
    pub fn per_core_stride_elems(&self, d: &str) -> u64 {
        if d == self.stick {
            self.host_size
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != self.stick_idx)
                .map(|(_, &e)| e)
                .product()
        } else if let Some(i) = self.layout.iter().position(|&x| x == d) {
            self.host_size[i + 1..].iter().product()
        } else {
            1
        }
    }

    /// device_size in dim_map order. For the re-tiled case — a 2-D `[a,b]` sticked on
    /// its LAST dim `b` (the matmul KERNEL) — this is `[b/STICK, a, STICK]` (b-tile
    /// OUTER, a, stick), matching torch-spyre's rank-2 `get_generic_stick_layout`
    /// `{d1,d0,d1}`. Other shapes (activations, never host-re-tiled) get a flat
    /// `[total/STICK, STICK]` that still round-trips byte-for-byte.
    pub fn device_size(&self) -> Vec<u64> {
        let stk = DF::ELEMS_PER_STICK as u64;
        if self.layout.len() == 2 && self.stick_idx == 1 {
            let (a, b) = (self.host_size[0], self.host_size[1]);
            vec![b / stk, a, stk]
        } else {
            let total: u64 = self.host_size.iter().product();
            vec![total / stk, stk]
        }
    }

    /// stride_map: the host-element stride per device axis, so the host re-tile is
    /// `device(coord) = host[Σ coord·stride_map]`. For 2-D `[a,b]` stick=b over device
    /// `[b/STICK, a, STICK]`: `[STICK, b, 1]` → `host_off = t·STICK + i·b + s`.
    pub fn stride_map(&self) -> Vec<u64> {
        let stk = DF::ELEMS_PER_STICK as u64;
        if self.layout.len() == 2 && self.stick_idx == 1 {
            vec![stk, self.host_size[1], 1]
        } else {
            vec![stk, 1]
        }
    }

    /// [`stride_map`](Self::stride_map) against the ON-DISK `[b, a]` buffer instead of the logical
    /// `[a, b]` one — i.e. the same gather, reading the weight in the orientation safetensors stores
    /// it, so nothing has to transpose it first.
    ///
    /// A GEMM weight is stored `[out, in]` and the graph wants `[in, out]`, so the worker transposes
    /// every one at load: 4.4s of granite-3.1-8b's 6.94s `load_weights`, writing a full second copy of
    /// the model. But the re-tile below is ALREADY a strided gather — it reads
    /// `host[Σ coord·stride_map]` per element and never assumes contiguity. Feeding it the disk
    /// orientation is two swapped strides, not a data movement:
    ///
    /// ```text
    /// host [in,out] (transposed):  device(t,i,s) <- host[i·out + t·STK+s]   [STK,    out, 1  ]
    /// host [out,in] (on disk):     device(t,i,s) <- host[(t·STK+s)·in + i]  [STK·in, 1,   in ]
    /// ```
    ///
    /// Same output bytes, same element count, one fewer pass over the model.
    ///
    /// TRADE-OFF, worth measuring rather than assuming: the shim's re-tile has a fast path when the
    /// INNERMOST device axis is contiguous in the source (`stride_map` ending in 1), which lets it
    /// `memcpy` a row at a time. That holds for the transposed map and NOT for this one, where the
    /// innermost stride is `in`. So this trades a whole pass for a slower gather.
    pub fn stride_map_disk_order(&self) -> Vec<u64> {
        let stk = DF::ELEMS_PER_STICK as u64;
        if self.layout.len() == 2 && self.stick_idx == 1 {
            // `host_size` is the LOGICAL `[in, out]`; on disk that buffer is `[out, in]`.
            let in_elems = self.host_size[0];
            vec![stk * in_elems, 1, in_elems]
        } else {
            vec![stk, 1]
        }
    }

    /// The on-device byte size = `prod(device_size)·WORD_LENGTH` — used to size-check
    /// the staged weight against the host buffer.
    pub fn device_bytes(&self) -> u64 {
        self.device_size().iter().product::<u64>() * DF::WORD_LENGTH as u64
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (g) PackedInt8Scale — the SENINT8 per-group dequant contract (sealed).
// ───────────────────────────────────────────────────────────────────────────

/// The per-group scale plan for a PACKED [`SenInt8`] weight — the typed contract that
/// carries the f16 scale shape alongside the 1-byte-resident int8 codes, so the on-card
/// dequant (`w[k] = scale[scale_index_for(k)] · q[k]`, symmetric affine, bias 0 / no
/// zero-point — the math grounded in scratchy's CUDA `affine_dequant_b8_bf16`) has a
/// provably-consistent scale index for every code.
///
/// Fields are PRIVATE; the SOLE constructor [`PackedInt8Scale::new`] enforces the
/// **GroupDivides** invariant — `group_size` MUST divide `k_elems` exactly, so every one
/// of the `k_elems` int8 codes belongs to EXACTLY one of the `groups = k_elems/group_size`
/// scales (no orphan code, no orphan scale). A mismatch is a `cargo build`-surfaced `Err`
/// (the SDSC bundle is baked at build time), NEVER a runtime assert. `group_size` is a
/// [`NonZeroU32`] so a zero group size is unrepresentable at the type level. Per-CHANNEL
/// symmetric int8 is the `group_size == k_elems` case (`groups == 1`); per-GROUP is any
/// exact divisor (e.g. g64). The int8 codes stay 1-byte resident ([`SenInt8::WORD_LENGTH`]
/// == 1) — proven half the f16 footprint by `senint8_resident_is_half_fp16_footprint`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedInt8Scale {
    k_elems: u32,
    group_size: NonZeroU32,
    groups: u32,
}

impl PackedInt8Scale {
    /// Sole constructor. `Err` (a build-time bake failure) unless `group_size` divides
    /// `k_elems` exactly and `k_elems > 0` (the **GroupDivides** guard). No runtime assert.
    pub fn new(k_elems: u32, group_size: NonZeroU32) -> Result<Self, SuperDscError> {
        let g = group_size.get();
        if !Self::group_divides(k_elems, group_size) {
            return Err(SuperDscError(format!(
                "PackedInt8Scale: group_size {g} does not divide k_elems {k_elems} exactly \
                 — a partial final group would leave int8 codes with no scale (or a scale with \
                 no codes). Refusing to construct (GroupDivides witness)."
            )));
        }
        Ok(Self {
            k_elems,
            group_size,
            groups: k_elems / g,
        })
    }

    /// The pure **GroupDivides** predicate [`new`](Self::new) branches on: `group_size`
    /// divides `k_elems` exactly and `k_elems > 0`. `new` returns `Ok` iff this holds and
    /// `Err` otherwise — by construction it is the SOLE branch condition. Factored out as a
    /// pure `const fn` so the guard's LOGIC is Kani-verifiable directly (locked by
    /// `senint8_packed_scale_rejects_indivisible_group`) without the model checker having to
    /// model the `Err`-string `format!` (an I/O leaf, factored out like the float-ALU leaf
    /// the transcendental proofs skip).
    pub const fn group_divides(k_elems: u32, group_size: NonZeroU32) -> bool {
        k_elems != 0 && k_elems.is_multiple_of(group_size.get())
    }

    /// The number of f16 scales this weight carries = `k_elems / group_size` (≥1 by ctor).
    pub fn groups(&self) -> u32 {
        self.groups
    }

    /// The group (scale index) that dequantizes int8 code `k`: `k / group_size`. Provably
    /// in `[0, groups)` for every `k < k_elems` (the covering map — no code is unscaled),
    /// locked by `senint8_dequant_scale_index_is_covering_partition`.
    pub fn scale_index_for(&self, k: u32) -> u32 {
        k / self.group_size.get()
    }

    /// Codes per group (the ctor's `group_size`).
    pub fn group_size(&self) -> u32 {
        self.group_size.get()
    }

    /// Total int8 codes (the reduction / in-feature extent).
    pub fn k_elems(&self) -> u32 {
        self.k_elems
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (g2) Fp8W8A8Dequant — the fp8 W8A8 output-dequant scale contract (sealed).
// ───────────────────────────────────────────────────────────────────────────

/// The fp8 W8A8 output-dequant scale contract: `out[m,n] = raw_fp16[m,n] · w_scale[n] · a_scale[m]`.
/// `w_scale` is PER-CHANNEL (one f16 per output channel `n`, STATIC from the ckpt); `a_scale` is PER-TOKEN
/// (one f16 per row `m`, DYNAMIC, computed on-card via `quantscalepertoken`). BOTH are CONSTANT ALONG the
/// K (reduction) axis, so they FACTOR OUT of the matmul sum ⇒ dequant is a single OUTPUT scale, NOT a
/// per-K-block scale (this is exactly why per-channel/per-token is the turnkey scheme vs blockwise, which
/// varies along K). Sealed: the SOLE constructor [`Fp8W8A8Dequant::new`] enforces `w_scale` sized to
/// `n_channels` and `a_scale` to `m_rows` — a mis-axised scale (e.g. `w_scale` sized to rows) would dequant
/// the WRONG axis, and is a `cargo build`-surfaced `Err` (bundle baked at build time), NEVER runtime
/// garbage. Scheme = the RedHatAI granite fp8-dynamic ckpt; the ÷2 residency win is the fp8 WEIGHT itself
/// (proven by `fp8_resident_half_*`), independent of this scale plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8W8A8Dequant {
    m_rows: u32,
    n_channels: u32,
}

impl Fp8W8A8Dequant {
    /// Sole constructor. `Err` (a build-time bake failure) unless `w_scale_len == n_channels` (PER-CHANNEL
    /// weight scale) AND `a_scale_len == m_rows` (PER-TOKEN activation scale). No runtime assert.
    pub fn new(
        m_rows: u32,
        n_channels: u32,
        w_scale_len: u32,
        a_scale_len: u32,
    ) -> Result<Self, SuperDscError> {
        if !Self::scales_well_shaped(m_rows, n_channels, w_scale_len, a_scale_len) {
            return Err(SuperDscError(format!(
                "Fp8W8A8Dequant: w_scale_len {w_scale_len} must == n_channels {n_channels} (PER-CHANNEL \
                 weight scale) AND a_scale_len {a_scale_len} must == m_rows {m_rows} (PER-TOKEN activation \
                 scale) — a mis-axised scale dequantizes the wrong axis. Refusing to construct."
            )));
        }
        Ok(Self { m_rows, n_channels })
    }

    /// The pure shape predicate [`new`](Self::new) branches on (factored out so the guard LOGIC is
    /// Kani-verifiable without modeling the `format!` I/O leaf — mirrors `PackedInt8Scale::group_divides`).
    pub const fn scales_well_shaped(
        m_rows: u32,
        n_channels: u32,
        w_scale_len: u32,
        a_scale_len: u32,
    ) -> bool {
        w_scale_len == n_channels && a_scale_len == m_rows
    }

    /// The `(w_scale index, a_scale index)` that dequantizes output element `(m, n)`: `w` by output CHANNEL
    /// `n`, `a` by TOKEN `m`. In-range + covering for every `(m < m_rows, n < n_channels)`, locked by
    /// `fp8_w8a8_dequant_scale_indices_covering`.
    pub const fn scale_indices(m: u32, n: u32) -> (u32, u32) {
        (n, m) // (w_scale[n], a_scale[m])
    }

    pub fn m_rows(&self) -> u32 {
        self.m_rows
    }
    pub fn n_channels(&self) -> u32 {
        self.n_channels
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (b)/(f) WorkPlan — sealed proof of the work-division. Sole minter of IterSym.
// ───────────────────────────────────────────────────────────────────────────

/// A named iteration dimension fed to the work-division. `is_stick` carries the
/// stick count (already divided by the per-stick element count) so the split
/// divides STICK COUNT, keeping per-core extents stick-aligned by arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct ItDim {
    pub name: &'static str,
    /// Element extent of the dim.
    pub size: u32,
    /// Reduction dim (matmul K): split last, ≤1 may carry a >1 split.
    pub is_reduction: bool,
    /// Stick (innermost device) axis: split basis is the stick COUNT, not elems.
    pub is_stick: bool,
    /// The device format of the operand this dim indexes — the SOLE source of the
    /// stick basis (`df.elems_per_stick()`: fp16=64, fp8/int8=128). The work-division
    /// [`WorkPlan::divide`] splits a stick dim by its `df`-derived stick COUNT and
    /// rejects (`Err`, at emit) any split leaving a core a sub-`df`-stick tile — so an
    /// fp8 (128-lane) matmul that splits at the fp16 64-granularity is a Rust-surfaced
    /// error, NOT an on-card `DtException L3DlOpsScheduler:1040`. Non-stick dims ignore it.
    pub df: Df,
}

/// A `≤ MAX_CORES` core count with no public constructor — minted only by
/// [`WorkPlan::divide`] (witness (b): the product of the splits is `≤ 32` by
/// type, not by a runtime re-check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoresUsed(u32);
impl CoresUsed {
    pub fn get(&self) -> u32 {
        self.0
    }
}

/// One core's slice INDEX along one split dim — `0..split_of(dim)`, the value
/// `coreIdToWkSlice_` carries per `(core, dim)`. It is an ORDINAL, not an element count:
/// the ONLY way it becomes an address term is [`WorkPlan::corner_elems`], which pairs it
/// with the SAME plan's per-core extent for that dim. A slice index multiplied by any
/// other quantity (a stride, a stick width, another dim's extent) no longer type-checks —
/// that hand product is the mq>1 conflation class this type removes.
///
/// Constructors are NAMED for their provenance: the mixed-radix core decomposition
/// (`core_to_wk_slice`) and the unsplit dim. Serializes as the bare integer, so
/// `coreIdToWkSlice_` is byte-identical on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceIndex(u32);

impl SliceIndex {
    /// The slice index of an UNSPLIT dim — every core holds the whole dim, so its
    /// work-slice corner along it is slice 0.
    pub const UNSPLIT: SliceIndex = SliceIndex(0);

    /// Minted by the mixed-radix decomposition of a core id over the split dims
    /// (`core_to_wk_slice`): the `core_id % split` digit for one dim.
    pub fn of_core_decomposition(digit: u32) -> SliceIndex {
        SliceIndex(digit)
    }
}

impl serde::Serialize for SliceIndex {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

/// `MaxCores<N>` — the core budget, a type-level cap so the splitter cannot be
/// called with a budget other than the hardware's.
#[derive(Clone, Copy, Debug)]
pub struct MaxCores<const N: u32>;

/// An iteration-space symbol. SEALED: the only way to obtain one is
/// [`WorkPlan::iter_sym`] / [`WorkPlan::iter_syms`], so any `IterSym` referenced
/// by a [`TensorArg`] is guaranteed to be a key of the op's iteration space
/// (witness (e), enforced by construction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IterSym {
    name: &'static str,
    // Private unit field so external code cannot brace-construct one.
    _seal: (),
}
impl IterSym {
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// The SEALED work-division proof (witnesses (b) and (f)). Private fields; the
/// only constructor is [`WorkPlan::divide`].
#[derive(Clone, Debug)]
pub struct WorkPlan {
    /// Ordered iteration dims (extent + stick flag), the iteration vocabulary.
    dims: Vec<ItDim>,
    /// dim name → split count (1 = unsplit). `product ≤ MAX_CORES` by ctor.
    splits: BTreeMap<&'static str, u32>,
    cores: CoresUsed,
}

impl WorkPlan {
    /// Build a work-division for `dims`, capped at `N` cores. Runs the supplied
    /// `splitter` (the cost-model / 3-pass distribute_cores ported in the wire
    /// module) and ASSERTS the (b) invariants:
    ///   * `product(splits) ≤ N`,
    ///   * every STICK dim keeps `≥1 stick / core` (stick_count(size) ≥ split).
    ///
    /// A violation is a `cargo`-surfaced `Err`, never an on-card DtException.
    ///
    /// `splitter` returns a `name → split` map (1-splits may be present or
    /// omitted). It is supplied by the caller so the wire module keeps its
    /// existing `matmul_cost_split`/`distribute_cores` logic verbatim — this
    /// type only OWNS the proof, not the heuristic.
    pub fn divide<const N: u32>(
        dims: &[ItDim],
        _budget: MaxCores<N>,
        splitter: impl FnOnce(&[ItDim], u32) -> BTreeMap<&'static str, u32>,
    ) -> Result<WorkPlan, String> {
        let mut splits = splitter(dims, N);
        // Normalise: drop 1-splits (they are the default).
        splits.retain(|_, v| *v > 1);

        let product: u32 = splits.values().product::<u32>().max(1);
        if product > N {
            return Err(format!(
                "work-division product {product} exceeds the {N}-core budget \
                 (splits={splits:?})"
            ));
        }
        // ⛔ A STICK DIM'S PER-CORE SLICE MUST BE A WHOLE NUMBER OF STICKS — not merely ≥1.
        //
        // This clause used to check ONLY `split > sticks` ("at least one stick per core"), which is a
        // necessary condition and not a sufficient one. `out = 12800` at the fp16 64-stick is 200 sticks;
        // a 32-way split satisfies `32 <= 200` and yet hands every core `12800/32 = 400` elements =
        // **6.25 sticks**. dxp cannot schedule a sub-stick-multiple slab and refuses the whole group with
        // `DtException: There must be at least one valid candidate` (L3DlOpsScheduler:1375) — a bake
        // failure whose message names neither the op, the dim, nor the number that is wrong.
        //
        // MEASURED: that is exactly why granite-3.1-8b-instruct-FP8 stopped baking. Its MLP is 12800 wide
        // (200 sticks, and 200 is not divisible by 32), while granite-2b's 8192 is 128 sticks and 128 IS
        // divisible by 32 — so the 2b passed by ARITHMETIC LUCK and the hole stayed invisible. Any model
        // whose split dim is not a multiple of `32 * stick` trips it.
        //
        // Stated as `sticks % split == 0` the wrong split is UNCONSTRUCTABLE: `WorkPlan` is the sealed
        // proof every op's work division is minted from, so no emitter can route around this, and the
        // failure arrives at `cargo build` naming the dim, its stick count, the split and the remainder.
        for d in dims {
            if let Some(&split) = splits.get(d.name)
                && d.is_stick
            {
                let basis = stick_basis(d);
                let sticks = d.size.div_ceil(basis);
                if split > sticks {
                    return Err(format!(
                        "stick dim '{}' (size {}, {sticks} stick(s)) split {split} ways \
                             leaves <1 stick/core (L3DlOpsScheduler:1040)",
                        d.name, d.size
                    ));
                }
                if sticks % split != 0 {
                    return Err(format!(
                        "stick dim '{}' (size {}, {basis}-elem sticks ⇒ {sticks} stick(s)) split \
                             {split} ways gives {} elem(s)/core = {}.{:02} sticks — a per-core slab that \
                             is not a WHOLE number of sticks, which dxp refuses to schedule \
                             (DtException 'There must be at least one valid candidate', \
                             L3DlOpsScheduler:1375). Split the STICK COUNT, not the element count: the \
                             largest divisor of {sticks} that is ≤ the core budget.",
                        d.name,
                        d.size,
                        d.size / split,
                        sticks / split,
                        (sticks % split) * 100 / split
                    ));
                }
            }
        }

        Ok(WorkPlan {
            dims: dims.to_vec(),
            splits,
            cores: CoresUsed(product),
        })
    }

    /// `≤ N` cores used (witness (b) by type).
    pub fn cores_used(&self) -> CoresUsed {
        self.cores
    }

    /// Split count for a named dim (1 = unsplit).
    pub fn split_of(&self, name: &str) -> u32 {
        *self.splits.get(name).unwrap_or(&1)
    }

    /// The split map (name → >1 split), for `numWkSlicesPerDim_`/`coreIdToWkSlice_`.
    pub fn splits(&self) -> &BTreeMap<&'static str, u32> {
        &self.splits
    }

    /// The iteration dims (the vocabulary), for `N_`/`ss_`/coordInfo extents.
    pub fn dims(&self) -> &[ItDim] {
        &self.dims
    }

    /// Element extent of a named dim (≥1), looked up in the iteration space.
    pub fn extent(&self, name: &str) -> u32 {
        self.dims
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.size)
            .unwrap_or(1)
            .max(1)
    }

    /// Mint the `IterSym` for `name` — the SOLE way to obtain one (witness (e)).
    /// `name` MUST be one of the plan's dims (panics otherwise — a programming
    /// error in the builder, not reachable from user input).
    pub fn iter_sym(&self, name: &'static str) -> IterSym {
        assert!(
            self.dims.iter().any(|d| d.name == name),
            "IterSym('{name}') is not a dim of this WorkPlan {:?}",
            self.dims.iter().map(|d| d.name).collect::<Vec<_>>()
        );
        IterSym { name, _seal: () }
    }

    /// Mint several `IterSym`s at once.
    pub fn iter_syms<const D: usize>(&self, names: [&'static str; D]) -> [IterSym; D] {
        names.map(|n| self.iter_sym(n))
    }

    /// Per-core extent of a named dim = `extent / split` (≥1). The work-division
    /// hands each core this much of `name`.
    pub fn per_core_extent(&self, name: &str) -> u32 {
        (self.extent(name) / self.split_of(name).max(1)).max(1)
    }

    /// The element CORNER along `name` of the work-slice at `idx` — the typed
    /// `(axis, slice-index)` pair from which a per-core start offset is derived:
    /// `idx · per_core_extent(name)`, with both factors read from THIS plan. This is
    /// the sole consumer of a [`SliceIndex`], so a slice index can only ever be scaled
    /// by its own dim's per-core extent — never by a stride or another dim's extent.
    pub fn corner_elems(&self, name: &str, idx: SliceIndex) -> u32 {
        idx.0 * self.per_core_extent(name)
    }

    /// LX-fit TIME-TILING (the Rust port of torch-spyre's `coarse_tile`, reduced
    /// to the single-op case). Decide whether the per-core resident set under THIS
    /// work-division fits the [`USABLE_LX_BYTES`] scratchpad; if not, split the
    /// `tiled_dim` (always the OUTPUT stick dim `out`) into `time` trips so each
    /// per-time tile fits.
    ///
    /// `resident_bytes_fn(self, out_per_time)` returns the per-core LX residency
    /// (in bytes) when the tiled dim is sliced to `out_per_time` elements per trip.
    /// It MUST include the batch/x factor (that is why the known bmm is 576 KiB,
    /// not 37 KiB — the KERNEL term dominates and is NOT divided by the mb split).
    ///
    /// Returns:
    ///   * `Ok(None)` — `resident(per_core_out) ≤ USABLE_LX`: time = 1, today's
    ///     single-shot path (already green). No [`TimeTile`] is minted.
    ///   * `Ok(Some(TimeTile{count>1, dim}))` — the SMALLEST `time>1` dividing the
    ///     per-core STICK COUNT of `tiled_dim` for which `out_per_time =
    ///     per_core_out / time` (a) keeps `resident ≤ USABLE_LX` AND (b) is a whole
    ///     64-fp16 stick (the [`StickExtent`] / **TiledStickExtent** invariant — a
    ///     time-tile can NEVER create a sub-stick slab).
    ///   * `Err(SuperDscError)` — NO divisor fits, even tiling `tiled_dim` to single
    ///     sticks. THIS is the whole point: "matmul too big to tile on N alone"
    ///     (it would need K-time PSUM accumulation, out of scope) becomes a `cargo
    ///     build` failure, NEVER an on-card DtException 1535.
    ///
    /// `dim` is minted via [`WorkPlan::iter_sym`], so the returned [`TimeTile`] is
    /// provably a key of this op's iteration space (witness (e)).
    pub fn time_tile_for_lx(
        &self,
        resident_bytes_fn: impl Fn(&WorkPlan, u32) -> u64,
        tiled_dim: &'static str,
    ) -> Result<Option<TimeTile>, crate::superdsc_error::SuperDscError> {
        use crate::superdsc_error::SuperDscError;

        let per_core_out = self.per_core_extent(tiled_dim);
        // time=1 (no split): does the full per-core tile already fit?
        if resident_bytes_fn(self, per_core_out) <= USABLE_LX_BYTES {
            return Ok(None);
        }

        // The STICK of the tiled dim is its OPERAND device format's, NOT a hardcoded 64:
        // fp8/int8 = 128, fp16 = 64. A time-tile of an fp8 (128-lane) dim to a 64-wide slab is a
        // whole fp16 stick but HALF an fp8 stick — the exact `L3DlOpsScheduler:1070 multiple-of-stick`
        // DtException. Deriving the stick from `df` makes such a slab either a whole fp8 stick or a
        // typed `Err` here (build-time), never an on-card fault. fp16 dims are byte-identical (stk=64).
        let df = self
            .dims()
            .iter()
            .find(|d| d.name == tiled_dim)
            .map(|d| d.df)
            .unwrap_or(Df::Fp16);
        let stk = df.elems_per_stick();

        // The per-core stick count of the tiled dim — the divisor search space.
        // `out_per_time = per_core_out / time` must stay a whole DF-stick, so we only
        // ever search divisors of the STICK COUNT (TiledStickExtent).
        let stick_count = per_core_out / stk;
        if !per_core_out.is_multiple_of(stk) || stick_count == 0 {
            return Err(SuperDscError(format!(
                "time_tile_for_lx('{tiled_dim}'): per-core extent {per_core_out} is not a whole \
                 multiple of the {stk}-elem {} stick — a time-tile cannot be derived from a \
                 sub-stick per-core slab (the spatial split should have rejected this; \
                 L3DlOpsScheduler.cpp:1040)",
                df.dataformat()
            )));
        }

        // Search divisors of the stick count ASCENDING for the smallest time>1
        // whose per-time tile fits. `time` divides `stick_count`, so
        // `out_per_time = (stick_count / time) * stk` is a whole DF-stick by arithmetic.
        for time in 2..=stick_count {
            if !stick_count.is_multiple_of(time) {
                continue;
            }
            let out_per_time = (stick_count / time) * stk;
            // Re-assert the DF-stick invariant via the SAME typed constructor the spatial split
            // uses (witness (a) / TiledStickExtent) — a sub-DF-stick slab is a propagated `Err`.
            assert_df_stick_multiple(out_per_time, df).map_err(SuperDscError)?;
            if resident_bytes_fn(self, out_per_time) <= USABLE_LX_BYTES {
                return Ok(Some(TimeTile {
                    count: time,
                    dim: self.iter_sym(tiled_dim),
                    _seal: (),
                }));
            }
        }

        // Even tiling the dim to single DF-sticks (time == stick_count, out_per_time = stk) does not
        // fit. That needs K-time PSUM accumulation (Stage 2) — refuse to bake; this Err is the
        // build-time guard for DtException 1535.
        let single_stick_resident = resident_bytes_fn(self, stk);
        Err(SuperDscError(format!(
            "time_tile_for_lx('{tiled_dim}'): per-core tile does NOT fit the {USABLE_LX_BYTES}-B \
             usable LX scratchpad even after tiling '{tiled_dim}' to single {stk}-elem {} sticks \
             (resident at out_per_time={stk} is {single_stick_resident} B > {USABLE_LX_BYTES} B). \
             This matmul needs K-time PSUM accumulation (Stage 2), which the frontend does not \
             yet emit. Refusing to bake — this would be an on-card DtException 1535 \
             (register-file / LX over-subscription). {}",
            df.dataformat(),
            self.shape_note()
        )))
    }

    /// The iteration extents and per-core slices behind an LX-fit refusal — the numbers that decide
    /// [`Self::time_tile_for_lx`]'s answer, so the message names the shape instead of only the byte
    /// count. Without it the refusal says a matmul is too big without saying which one or how it was
    /// divided, and finding that out costs a whole traced rebuild.
    fn shape_note(&self) -> String {
        let per = self
            .dims
            .iter()
            .map(|d| {
                format!(
                    "{}={}/{}={}",
                    d.name,
                    d.size,
                    self.split_of(d.name),
                    self.per_core_extent(d.name)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("(extent/split=per-core: {per})")
    }

    /// ⭐ THE WHOLE TILER: work-division, LX-fit time-tiling, and — when the cost model's own split
    /// admits NO time-tile that fits — a REDUCTION-SPLIT REPAIR that trades spatial cores for
    /// reduction cores and tries again.
    ///
    /// [`Self::time_tile_for_lx`]'s only lever is the OUTPUT stick dim, and that lever bottoms out at
    /// one stick. Past that point the per-core resident set is
    /// `per_core_mb·per_core_in + per_core_in·stick + per_core_mb·stick` — a function of the SPLIT,
    /// not of the time-tiling — so "does not fit even at one stick" is a statement about the
    /// work-division, and the frontier is the per-core REDUCTION extent. Splitting the reduction axis
    /// `r` ways divides it by `r`: the `r` cores each contract a K-slice into a partial product dxp
    /// PSUM-accumulates into the shared output tile (the output carries no `in`, so the disjoint-write
    /// rule #50 does not apply), and each weight byte is still read exactly once card-wide — unlike an
    /// `mb` split, which hands every core the SAME stationary weight and so multiplies the weight
    /// stream that a decode is bandwidth-bound on.
    ///
    /// ⛔ MEASURED, and it is why this exists: granite-3.1-8b at fp16 has a `k=12800 n=4096` down_proj.
    /// At one decode row the cost model's `out=32` leaves `per_core_in=12800`, whose one-stick weight
    /// slab is 12800·64·2 = 1,638,400 B — 97.7 % of [`USABLE_LX_BYTES`] on its own. It fits with 0.8 %
    /// spare. At TWO rows the activation slab grows by 25,600 B and the same op needs 1,689,856 B, so
    /// every decode-batch rung of the model was a build refusal reading "needs Stage 2" for a shape
    /// one reduction split away from fitting.
    ///
    /// The repair fires ONLY where the build would otherwise fail, and asks for the SMALLEST reduction
    /// split that fits, so every op the cost model already placed is untouched and byte-identical.
    /// Exhausting every divisor of the reduction stick count still yields the Stage-2 `Err` — a shape
    /// that genuinely needs K-TIME (not K-core) accumulation is still a `cargo build` failure, never
    /// an on-card DtException 1535.
    pub fn divide_and_time_tile_for_lx<const N: u32>(
        dims: &[ItDim],
        budget: MaxCores<N>,
        tiled_dim: &'static str,
        splitter: impl Fn(&[ItDim], u32) -> BTreeMap<&'static str, u32>,
        resident_bytes_fn: impl Fn(&WorkPlan, u32) -> u64,
    ) -> Result<(WorkPlan, Option<TimeTile>), crate::superdsc_error::SuperDscError> {
        use crate::superdsc_error::SuperDscError;

        let plan = WorkPlan::divide(dims, budget, &splitter).map_err(SuperDscError)?;
        let refusal = match plan.time_tile_for_lx(&resident_bytes_fn, tiled_dim) {
            Ok(tt) => return Ok((plan, tt)),
            Err(e) => e,
        };

        // ⚠️ THIS REPAIR CAN STILL PRODUCE AN `in > 1` SPLIT, AND ON THE CURRENT DXP IMAGE THAT IS
        // UNSCHEDULABLE — but it is NOT constrained here, deliberately. Two of this crate's own tests
        // assert that the ladder is what PLACES granite-8b's down projection at two decode rows, so
        // gating it here would break a shape the model path depends on for a reason that is about one
        // dxp image. The seal lives instead where the FINAL split is known and a refusal can name the
        // op: `matmul_opspec`'s post-plan check. See [`crate::work::matmul_split_plan`] for the
        // two-sided fixture evidence.
        //
        // The reduction axis is the ONLY per-core extent left to shrink (see the doc above). A
        // non-stick or absent reduction dim has no divisor ladder to walk, so the refusal stands.
        let Some(red) = dims.iter().find(|d| d.is_reduction && d.is_stick) else {
            return Err(refusal);
        };
        let red_sticks = red.size / stick_basis(red).max(1);
        for r in 2..=red_sticks.min(N) {
            if !red_sticks.is_multiple_of(r) {
                continue;
            }
            // Ask the cost model for its own best split of the cores the reduction does NOT take, then
            // pin the reduction to `r` (or to its own larger choice). Re-running the SAME splitter
            // under a smaller budget keeps the spatial shape the model's decision, not this pass's.
            let mut splits = splitter(dims, (N / r).max(1));
            let own = splits.get(red.name).copied().unwrap_or(1);
            splits.insert(red.name, own.max(r));
            // `divide` re-checks the core budget and the whole-stick-per-core law, so an illegal
            // candidate self-eliminates here rather than reaching dxp.
            let Ok(cand) = WorkPlan::divide(dims, budget, |_, _| splits.clone()) else {
                continue;
            };
            if let Ok(tt) = cand.time_tile_for_lx(&resident_bytes_fn, tiled_dim) {
                return Ok((cand, tt));
            }
        }
        Err(SuperDscError(format!(
            "{} A reduction-core split was also tried at every divisor of '{}'s {red_sticks}-stick \
             reduction (2..={}) and none of them fit either — this shape needs K-TIME accumulation, \
             not more K cores.",
            refusal.0,
            red.name,
            red_sticks.min(N)
        )))
    }
}

/// A validated TIME-TILE count for the LX-fit (coarse_tile) pass — the
/// **TimeTile** witness. SEALED (private `_seal` field, no public ctor): the SOLE
/// producer is [`WorkPlan::time_tile_for_lx`], which only mints a `TimeTile` whose
///   * `count > 1` (a 1-trip "tile" is [`Option::None`], NEVER `TimeTile{count:1}`),
///   * `count` divides the per-core STICK COUNT of `dim` exactly (so
///     `per_core_out / count` is a whole 64-fp16 stick — TiledStickExtent), and
///   * resulting per-time per-core resident set fits [`USABLE_LX_BYTES`].
///
/// A shape that cannot satisfy all three is a `cargo build` `Err`, NEVER an
/// on-card DtException 1535. `dim` is an [`IterSym`], hence provably a key of the
/// op's iteration space (witness (e)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeTile {
    count: u32,
    dim: IterSym,
    // Private unit field so external code cannot brace-construct one.
    _seal: (),
}
impl TimeTile {
    /// The number of tile-loop trips (`> 1` by the ctor invariant).
    pub fn count(&self) -> u32 {
        self.count
    }
    /// The tiled iteration-space dim (a key of the op's `WorkPlan`).
    pub fn dim(&self) -> IterSym {
        self.dim
    }
}

/// Stick basis for a dim — the per-stick element count, derived from the dim's
/// operand device format (fp16=64, fp8/int8=128). This is what makes the
/// [`WorkPlan::divide`] stick clause reject a sub-`df`-stick per-core slice at
/// emit time (fp8 split at fp16 granularity ⇒ `Err`, not a DtException).
fn stick_basis(d: &ItDim) -> u32 {
    d.df.elems_per_stick()
}

/// Assert `elems` is a whole multiple of `df`'s stick, via the TYPED [`StickExtent<DF>`] witness
/// (fp8/int8 ⇒ 128, fp32 ⇒ 32, fp16/bf16 ⇒ 64). A sub-stick extent is the sealed constructor's
/// `Err` — the SAME witness the spatial split uses, so the time-tile pass cannot mint a sub-DF-stick
/// slab (an fp8 dim tiled to a 64-wide fp16-stick slab is rejected here, at emit, not on-card).
pub fn assert_df_stick_multiple(elems: u32, df: Df) -> Result<(), String> {
    match df {
        Df::Fp8 => StickExtent::<Fp8>::new(elems).map(|_| ()),
        Df::SenInt8 => StickExtent::<SenInt8>::new(elems).map(|_| ()),
        Df::Fp32 => StickExtent::<Fp32>::new(elems).map(|_| ()),
        Df::Fp16 | Df::Bf16 => StickExtent::<Fp16>::new(elems).map(|_| ()),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (c) SdscFoldSet — the single source of truth for fold factors; AddrFold::new.
// ───────────────────────────────────────────────────────────────────────────

/// The fold factors for one SuperDSC op: `coreFold = num_cores`, `coreletFold =
/// ACTIVE_CORELETS (1)`, `timeFold = 1`. Created ONCE per `SdscOp`; every
/// `startAddr`/coordInfo/`coreFoldProp_`/`coreletFoldProp_` factor is copied
/// from it, so all fold cardinalities are provably equal (witness (c) — converts
/// foldInfrastructure.h:2775 'Different cardinality' into an unbuildable state).
#[derive(Clone, Copy, Debug)]
pub struct SdscFoldSet {
    core_fold: u32,
    corelet_fold: u32,
    time_fold: u32,
}

impl SdscFoldSet {
    /// Build the fold set for an op using `cores` cores. `coreletFold`/`timeFold`
    /// are the frontend constants (1) — there is no setter to diverge them.
    pub fn new(cores: CoresUsed) -> Self {
        SdscFoldSet {
            core_fold: cores.get().max(1),
            corelet_fold: ACTIVE_CORELETS,
            time_fold: 1,
        }
    }

    pub fn core_fold(&self) -> u32 {
        self.core_fold
    }
    pub fn corelet_fold(&self) -> u32 {
        self.corelet_fold
    }
    pub fn time_fold(&self) -> u32 {
        self.time_fold
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (d) Scale (sealed) + Role + Allocation.
// ───────────────────────────────────────────────────────────────────────────

/// `scale_` per device dim. SEALED enum (witness (d)): `Active(1)` |
/// `RedNonStick(-1)` | `RedStick(-2)`. Never a free integer the caller could
/// mis-set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Active dim — present in the output. JSON `1`.
    Active,
    /// Reduction dim, non-stick axis. JSON `-1`.
    RedNonStick,
    /// Reduction dim aligned to the stick axis. JSON `-2`.
    RedStick,
}
impl Scale {
    /// The JSON `scale_` integer.
    pub fn to_i64(self) -> i64 {
        match self {
            Scale::Active => 1,
            Scale::RedNonStick => -1,
            Scale::RedStick => -2,
        }
    }
    /// A reduction-stick dim drives `is_stick_reduction` in coordInfo.
    pub fn is_stick_reduction(self) -> bool {
        matches!(self, Scale::RedStick)
    }
}

/// Tensor role — drives the `(Role, OpFunc) → layout` const table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Input,
    Kernel,
    Output,
}
impl Role {
    pub fn ds_type(self) -> &'static str {
        match self {
            Role::Input => "INPUT",
            Role::Kernel => "KERNEL",
            Role::Output => "OUTPUT",
        }
    }
}

/// Frontend allocation — exactly ONE residency. Sizes stay UNSET (the scheduler
/// fills them). Sealed so "both hbm and lx address" is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allocation {
    /// Resident in HBM (default for matmul operands).
    Hbm,
    /// Resident in the LX scratchpad only (memOrg_ → lx only).
    Lx,
}
impl Allocation {
    pub fn is_lx(self) -> bool {
        matches!(self, Allocation::Lx)
    }
    pub fn component(self) -> &'static str {
        if self.is_lx() { "lx" } else { "hbm" }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (e)/(#6) OpFunc (sealed) → ExUnit (sealed newtype).
// ───────────────────────────────────────────────────────────────────────────

/// The op function. SEALED enum — `exUnit` is derived solely from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpFunc {
    Matmul,
    BatchMatmul,
    /// `interslicetranspose_fp16` — an 8×8 inter-slice block transpose on the PT
    /// (matrix) unit. ONE input, ONE output; the transpose is expressed by the
    /// OUTPUT's `stickDimOrder_` swapping the inner two layout axes vs the INPUT
    /// (input stick=[`out`] 64; output stick=[`out`,`mb`] [8,8]). Used by
    /// in-bundle attention to transpose the current chunk's new-K `[mq,hd]→[hd,mq]`
    /// for the new-token `Q·Kᵀ` score block (free reshape when mq=1).
    Transpose,
    /// `restickify` — a DEPLOYED device dim-reorder / re-tile (vs the UNDEPLOYED
    /// `interslicetranspose`). ONE input, ONE output; the relayout is expressed by
    /// the OUTPUT's `layoutDimOrder_`/`stickDimOrder_` differing from the INPUT (e.g.
    /// `[cap,hd]` hd-sticked → `[hd,cap]` cap-sticked). torch-spyre inserts exactly
    /// this for a TRANSPOSED matmul input (test_padding.py: "restickify reorders x's
    /// device dims"); in-bundle attention uses it to turn the natural-written K cache
    /// `[cap,hd]` into the matmul-kernel layout `[hd,cap]` WITHOUT a fused
    /// bmm-transpose (which SIGABRTs in dxp_standalone — IBM issue #1731).
    /// POD-CONFIRM: opFuncName + exUnit against the SDK `dscdefn.cpp opFuncsToString`.
    Restickify,
    Add,
    Subtract,
    Multiply,
    /// `realdiv` — elementwise a/b of TWO tensors (sfp unit). torch-spyre `spyre__sdpa_overrideable`
    /// normalizes the attention output as `output / denominator` (decompositions.py); this is that op.
    RealDiv,
    /// `abs` — elementwise |x| (unary_parallel.ddl, sfp unit). The dynamic per-token fp8 activation scale is
    /// `amax = max(|x|)`; this is its `|x|` (then a MAX reduce), replacing the L2-norm smooth-max approx.
    Abs,
    Silu,
    Exp,
    Reciprocal,
    Sqrt,
    Rsqrt,
    Sigmoid,
    Gelu,
    Mish,
    Tanh,
    /// `dl16tofp32` / `fp32todl16` — SFP dtype CONVERTS (DeepTools binds both, unit="sfp"). The KSPLIT
    /// fp32-merge converts each f16 block partial → fp32 (bridging the 64→32 elem-stick), does all-fp32 adds
    /// (same-stick, so dxp resolves the shared dim — the mixed [f16,fp32] add failed `lit_dim_size`), then
    /// converts the fp32 accumulator → f16 for the output.
    Dl16ToFp32,
    Fp32ToDl16,
    Sum,
    Max,
    Mean,
    /// `identity` — an elementwise copy (sfp unit). Used by in-bundle attention to
    /// GQA-replicate the new chunk's K/V (nkvh→nqh heads) via sliced-read /
    /// offset-write copies, and as a generic relayout/move.
    Identity,
    /// `maximum` — elementwise max of TWO tensors (sfp unit; distinct from the
    /// `Max` REDUCTION). Used by in-bundle attention's combined softmax to take the
    /// cross-block max (`max(max_prefix, max_new)`) for numerical stabilization.
    Maximum,
    /// `minimum` — elementwise min of TWO tensors (sfp unit; the min twin of [`Maximum`]). Used with
    /// `Maximum` to CLAMP the fp8 activation into `[-448, 448]` before [`Qfp8ch`]: `min(max(x,-448),448)`.
    Minimum,
    /// `qfp8ch` — cast/quantize fp16 → SEN143_FP8 (E4M3), the fp8 W8A8 activation quantizer (SFP unit;
    /// DDL `quantization_*.ddl` opFuncName "qfp8ch", IBM PR #2401). The input MUST be CLAMPED to [-448, 448]
    /// FIRST (`mul-by-1/scale → clamp±448 → qfp8ch`, mirroring IBM's `quantize_fp8_with_scale`): the
    /// "`act/(amax/448)` ≤ 448 by construction" bound holds in REAL arithmetic but NOT necessarily in fp16
    /// (rounding at the row's absmax element can nudge it past 448 ⇒ E4M3 overflow), so the clamp is a real
    /// compute step, not an unproven skip. (If we ever drop the clamp it MUST be gated by a Kani proof of
    /// the fp16 bound.) Output tensor is `_fp8`-named so the residency + `matmulfp8` dispatch activate.
    Qfp8ch,
}

impl OpFunc {
    /// `matmul` for a PLAIN 2-D matmul (batch==1); `batchmatmul` for a true batched matmul
    /// (batch>1, the attention bmm). PROVEN 2026-07-07 by diffing torch-spyre's OWN generate_sdsc:
    /// for a batch=1 mm the reference emits opFuncName "matmul" with 2-D tensors ([mb,in]/[in,out]/
    /// [mb,out], no `y`); for a 3-D batched mm it emits "batchmatmul". The prior ALWAYS-batchmatmul
    /// was the mq>1 PREFILL BUG: bmm.ddl treats the row dim under a batch loop and, at batch=1, wrote
    /// only the FIRST mb row of each core's tile (mb=1 decode tolerated it → never caught; mb>1
    /// prefill scattered rows → the 8→1/even-row collapse). torch-spyre `lower_mm` (2-D) → matmul,
    /// `lower_bmm`/3-D → batchmatmul; this mirrors that. The earlier "plain matmul → orthogonal
    /// product" was with the 3-D `y` tensors still present (matmul opfunc + 3-D rank mismatch); the
    /// reference uses matmul WITH 2-D tensors (`matmul_dims`/`matmul_opspec_split` drop `y` at batch==1).
    pub fn matmul(_batch: u32) -> OpFunc {
        // ALWAYS batchmatmul — PROVEN 2026-07-07 (torch-spyre repo + issues #3064/#617/#984 via the
        // matmul-search workflow): IBM's inductor emits opFuncName "batchmatmul" on EVERY matmul path
        // (lower_mm → BATCH_MATMUL_OP), NEVER "matmul". A plain "matmul" opFuncName hits an untested
        // DeepTools PT microcode path. (My earlier batch==1→"matmul" was based on a HAND-BUILT
        // reference SDSC that wrongly used "matmul"; the REAL lower_mm uses batchmatmul.)
        OpFunc::BatchMatmul
    }

    /// The DeepTools `opFuncName`.
    pub fn name(self) -> &'static str {
        match self {
            OpFunc::Matmul => "matmul",
            OpFunc::BatchMatmul => "batchmatmul",
            OpFunc::Transpose => "interslicetranspose_fp16",
            // dxp's recognized name is "ReStickifyOpHBM" (the DscType op-node name,
            // dscdefn.h:267-271 + restickify.ddl operation_bind opFuncName). Plain
            // "restickify" was REJECTED on-card (DtException Unrecognized opFunc,
            // designSpaceConfig.cpp:7713 — caught at build by build.rs:230).
            OpFunc::Restickify => "ReStickifyOpHBM",
            OpFunc::Add => "add",
            OpFunc::Subtract => "sub",
            // dxp's recognized name is "mul", NOT "multiply" (on-card 2026-06-25:
            // DtException "Unrecognized opFunc: multiply", designSpaceConfig.cpp:7713;
            // recognized set = dscdefn.cpp `opFuncsToString`).
            OpFunc::Multiply => "mul",
            OpFunc::RealDiv => "realdiv",
            OpFunc::Abs => "abs",
            OpFunc::Silu => "silu",
            OpFunc::Exp => "exp",
            OpFunc::Reciprocal => "reciprocal",
            OpFunc::Sqrt => "sqrt",
            OpFunc::Rsqrt => "rsqrt",
            OpFunc::Sigmoid => "sigmoid",
            // dxp's recognized name is "gelufwd", NOT "gelu" (dscdefn.cpp opFuncsToString).
            OpFunc::Gelu => "gelufwd",
            OpFunc::Mish => "mish",
            OpFunc::Tanh => "tanh",
            OpFunc::Dl16ToFp32 => "dl16tofp32",
            OpFunc::Fp32ToDl16 => "fp32todl16",
            OpFunc::Sum => "sum",
            OpFunc::Max => "max",
            OpFunc::Mean => "mean",
            OpFunc::Identity => "identity",
            OpFunc::Maximum => "maximum",
            OpFunc::Minimum => "minimum",
            OpFunc::Qfp8ch => "qfp8ch",
        }
    }

    /// The SOLE producer of [`ExUnit`] (witness #6): `pt` for matmul/bmm, `sfp`
    /// otherwise. It is impossible to put `sfp` on a matmul.
    pub fn ex_unit(self) -> ExUnit {
        let s = match self {
            // ReStickifyOpHBM is an APE-engine relayout, but `APEOpHBM` is a DscType
            // op-NODE, not a SenComponents hardware unit — there is NO "ape" exUnit. The
            // dxp-validated golden sdsc_restickify.json emits exUnit "sfp" (dxp synthesizes
            // the multi-stage relayout dataflow from the dual-stick layout). So Restickify
            // falls through to the "sfp" default below.
            OpFunc::Matmul | OpFunc::BatchMatmul | OpFunc::Transpose => "pt",
            _ => "sfp",
        };
        ExUnit(s)
    }

    /// True for the transcendental sfp ops that need the SFP polynomial constant
    /// table (the on-card `setupVariables`/`map::at` guard). `add`/`multiply`/
    /// `sum`/`max`/`mean` are NOT in this set.
    pub fn needs_sfp_const_table(self) -> bool {
        // NOTE (2026-06-27): rsqrt/sqrt EXCLUDED — their dxp DDL branch (unary_parallel.ddl
        // RSQRT/SQRT) gets its Newton-Raphson bit-trick constants (ffff/maskone/zero) from
        // the DDL's OWN `ddl.define_constant` placed into SFP-LRF; scratchy's 20-entry
        // SfpConstTable may SHADOW/conflict with those (the on-card rsqrt computes 1/x =
        // the unrefined passthrough, exactly what happens if the bit-trick constants are
        // wrong). The working `add` op carries an EMPTY constantInfo, proving empty is
        // valid. Testing whether dropping the table lets rsqrt refine correctly.
        //
        // NOTE (2026-07-02): Silu EXCLUDED — same shadowing pattern, now DDL-CONFIRMED. The `silu`
        // opFuncName's DDL path (unary_parallel.ddl:155-210) computes SIGMOID(x) then FMUL(x,sig),
        // sourcing ALL 6 sigmoid coefficients (`expVal1..5`, `zero`) via `ddl.define_constant`
        // [INTERNAL] and reading ZERO `get_external_constant`. scratchy's SfpConstTable is non-empty
        // AND repeats `expVal1..5` as EXTERNAL constants → collision in SFP-LRF → the sigmoid refines
        // against shadowed coefficients. MEASURED on-card (granite micro-g3.3 layer0): gate/up matmuls
        // exact, but silu·up |.|=69.4 vs golden 81.5 with max PRESERVED (44.9 vs 45.0) — the mid-range
        // under-refinement fingerprint. Empty constantInfo (this exclusion) lets the DDL's own
        // define_constants drive the sigmoid. LOCKED by `silu_constant_info_matches_ddl_contract`.
        // NOTE (2026-07-04): Exp EXCLUDED — same shadowing pattern as silu, DDL-CONFIRMED. The `exp`
        // opFuncName's DDL path (unary_parallel.ddl:98-137) sources ALL its polynomial coefficients
        // expVal1..5 (0x46dc/0x46e2/0x34c5/0x2121/0x3e00) + eps via `ddl.define_constant` [INTERNAL],
        // allocating them into SFP-LRF itself, and reads ZERO `get_external_constant`. scratchy's
        // SfpConstTable REPEATS expVal1..5 as EXTERNAL constants → collision in SFP-LRF → the softmax
        // exp refines against shadowed coefficients (the SAME silu·up under-refinement, but corrupting
        // the decode-attention softmax at EVERY step). Empty constantInfo (this exclusion) lets the
        // DDL's own define_constants drive exp. LOCKED by `exp_constant_info_matches_ddl_contract`.
        // AUDIT (2026-07-04, DDL-CONFIRMED on-pod unary_parallel.ddl): EVERY SFP transcendental lowered
        // through this path — silu(155-210) / exp(98-137) / reciprocal(382-417) / sigmoid / gelu / mish /
        // tanh / rsqrt / sqrt — sources its coefficients via `ddl.define_constant` [INTERNAL] into SFP-LRF and
        // reads ZERO `get_external_constant`. scratchy's external SfpConstTable REPEATS those coefficients
        // (expVal1..5, plus1/minus1, sigmoid coeffs), so attaching it can ONLY collide/shadow them in SFP-LRF
        // — the MEASURED silu·up under-refinement, and the exp/reciprocal softmax corruption fixed this
        // session. Ops that genuinely read external constants (clip: clipMin/clipMax) are not lowered here.
        // ⇒ NO op needs the external table. LOCKED per-op by the {silu,exp,reciprocal,sigmoid,gelu,mish,tanh}
        // `_constant_info_matches_ddl_contract` Kani harnesses. Was `matches!(Exp|Reciprocal|Sigmoid|Gelu|
        // Mish|Tanh)`; Exp+Reciprocal removed (softmax fix, measured), then the rest by DDL audit.
        let _ = self;
        false
    }
}

/// SEALED `exUnit` newtype — no public constructor; minted only by
/// [`OpFunc::ex_unit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExUnit(&'static str);
impl ExUnit {
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

// ───────────────────────────────────────────────────────────────────────────
// (d) TensorArg<D> + AnyTensorArg.
// ───────────────────────────────────────────────────────────────────────────

/// A frontend tensor argument of device rank `D` (witness (d)): `scale`,
/// `device_dims` and the `layout`/`stick` share the SAME `D`, so a rank mismatch
/// is a compile error. `arg_index` is NOT stored — it is the position in
/// `OpSpec.args` (derived at emit time, the op_spec.py invariant).
///
/// Fields are PRIVATE: the SOLE constructor [`TensorArg::new`] enforces the
/// **MaterializedStick** invariant (the stick dim's scale is `Active` or
/// `RedStick`, never the phantom `RedNonStick`/-1). That makes the on-card
/// DtException-1535 chunk-size rule (`Π(dimSize)·wordLength % 128 == 0`) hold BY
/// CONSTRUCTION — a phantom-scaled stick dim collapses the stick to `dimSize=1`,
/// dropping the ×64×2 factor. With [`StickExtent`] already forcing the stick
/// extent to a multiple of 64, a materialized stick ⇒ its `dimSize` is a multiple
/// of 64 ⇒ the per-core LX chunk is ALWAYS a multiple of 128, independent of the
/// other dims' extents. So invalid IR is unconstructible — no emit-time assert and
/// no grep of generated JSON is needed downstream.
#[derive(Clone, Debug)]
pub struct TensorArg<const D: usize> {
    is_input: bool,
    name: String,
    role: Role,
    scale: [Scale; D],
    /// Per-dim iteration symbols (only mintable from the op's `WorkPlan`).
    device_dims: [IterSym; D],
    /// `layoutDimOrder_` — the const role/op layout. Same `D` as `scale`.
    layout: [&'static str; D],
    /// Innermost stick axis name (one of `layout`). torch-spyre `stickDimOrder_`.
    stick: &'static str,
    allocation: Allocation,
    /// SLICE offset into the source dataspace, in ELEMENTS along the stick axis
    /// (0 = whole tensor). A non-zero offset reads `x[offset_elems ..]` — the
    /// AllocNode `startAddr` is bumped by `offset_elems · wordLength`. Used for
    /// RoPE's half-slices (`x[half:]`). Set via [`TensorArg::with_offset`].
    offset_elems: u32,
    /// The dataspace dtype — drives `wordLength` / `stickSize_` / `dataFormat_` at emit. Defaults to
    /// [`Df::Fp16`] (the format of ~every tensor); set to a packed format via [`TensorArg::with_df`] for
    /// the fp8/fp32/int8 tensors. Carrying it here (vs sniffing a `_fp8` name suffix) makes a forgotten
    /// declaration a compile-visible `Df::Fp16` at the call site, not a silent runtime residency mismatch.
    df: Df,
    /// FORCE the RowBlocked (stick-major `[feat/stk, m, stk]`) DEVICE layout regardless of iteration rank —
    /// the minimal `ElementArrangement` (torch-spyre carries a per-tensor device layout independent of the
    /// op's iteration nesting; scratchy otherwise derives layout from rank via `for_view_df`). Needed for the
    /// fp32 RMSNorm island: its ops must stay rank-3 (dxp's per-dim buffer walk faults on a rank-2 narrowing
    /// convert) yet ADDRESS RowBlocked (the residual is RowBlocked, and a flat rank-3 read row-mixes). Default
    /// `false` = rank-based (`for_view_df`). Set via [`TensorArg::with_row_blocked`].
    row_blocked: bool,
    /// Per-dim PHYSICAL device extent, distinct from this op's own iteration-space view of that dim
    /// (`device_dims`/`op.iter.extent`) — torch-spyre's `arg.device_size` (`superdsc.py:447-469`), which
    /// every stride/offset is derived from, NOT from `iteration_space[dim]`. `None` at an index means
    /// "this tensor's physical extent for that dim IS the op's own iteration extent" (the overwhelmingly
    /// common case — a tensor allocated exactly to the shape this one op produces/consumes), so every
    /// existing call site is a no-op (`[None; D]` from `TensorArg::new`). `Some(phys)` declares the tensor
    /// is actually a slice/reuse of a LARGER physical allocation along that dim (`phys > it_dim_size`) —
    /// torch-spyre's `dev_dim_size > it_dim_size` case — so address/stride derivation for that dim must use
    /// `phys`, not the op's logical extent, mirroring torch-spyre's `strides[dim] = strides[dim] //
    /// dev_dim_size * it_dim_size` rescale exactly. Set via [`TensorArg::with_device_extent`].
    device_extent: [Option<u32>; D],
}

impl<const D: usize> TensorArg<D> {
    /// The SOLE constructor (fields are private). Enforces the **MaterializedStick**
    /// invariant: `stick` MUST be one of `layout`, and its `scale` MUST be
    /// `Active` or `RedStick` — never `RedNonStick` (the phantom -1 that collapses
    /// the stick to `dimSize=1` and would cause the on-card DtException 1535
    /// "Incorrect chunk size in bytes"). Returns `Err` (a `cargo build` failure)
    /// otherwise — so a non-128-multiple LX chunk is unrepresentable, not a
    /// runtime crash.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        is_input: bool,
        name: String,
        role: Role,
        scale: [Scale; D],
        device_dims: [IterSym; D],
        layout: [&'static str; D],
        stick: &'static str,
        allocation: Allocation,
    ) -> Result<TensorArg<D>, SuperDscError> {
        let stick_idx = layout.iter().position(|&d| d == stick).ok_or_else(|| {
            SuperDscError(format!(
                "TensorArg '{name}': stick dim '{stick}' is not in layout {layout:?}"
            ))
        })?;
        if matches!(scale[stick_idx], Scale::RedNonStick) {
            return Err(SuperDscError(format!(
                "TensorArg '{name}': stick dim '{stick}' carries scale RedNonStick (-1), which \
                 collapses it to dimSize=1 — the per-core LX chunk would NOT be a multiple of \
                 bytesPerStick (128) → on-card DtException 1535. A stick dim must materialize \
                 (Active or RedStick). Refusing to construct (MaterializedStick witness)."
            )));
        }
        Ok(TensorArg {
            is_input,
            name,
            role,
            scale,
            device_dims,
            layout,
            stick,
            allocation,
            offset_elems: 0,
            df: Df::Fp16,
            row_blocked: false,
            device_extent: [None; D],
        })
    }

    /// Set the SLICE offset (elements along the stick axis) — a builder that reads
    /// `x[offset_elems ..]` (the AllocNode `startAddr` is bumped by `offset_elems ·
    /// wordLength` at emit). For RoPE half-slices. Chains off [`TensorArg::new`].
    pub fn with_offset(mut self, offset_elems: u32) -> TensorArg<D> {
        self.offset_elems = offset_elems;
        self
    }

    /// Set the dataspace dtype (default [`Df::Fp16`]) — for the packed fp8 / int8 / fp32-merge tensors.
    /// Chains off [`TensorArg::new`]. The `wordLength` / `stickSize_` / `dataFormat_` emitted for this
    /// arg all derive from `df`, so declaring it here is the ONLY place a non-fp16 residency is chosen.
    pub fn with_df(mut self, df: Df) -> TensorArg<D> {
        self.df = df;
        self
    }

    /// FORCE the RowBlocked device layout (see the `row_blocked` field). The fp32 RMSNorm island's
    /// per-tensor `ElementArrangement`: stay rank-3 for dxp, address RowBlocked for correctness.
    pub fn with_row_blocked(mut self, row_blocked: bool) -> TensorArg<D> {
        self.row_blocked = row_blocked;
        self
    }

    /// Declare this tensor's TRUE physical device extent for one `layout` dim, distinct from this
    /// op's own iteration-space view of it — torch-spyre's `arg.device_size` (`superdsc.py:447-469`).
    /// `dim` must name one of `layout`'s entries (a `cargo build` `Err` otherwise, same discipline as
    /// [`TensorArg::new`]'s stick-dim witness). Chains off [`TensorArg::new`].
    pub fn with_device_extent(
        mut self,
        dim: &'static str,
        phys_extent: u32,
    ) -> Result<TensorArg<D>, SuperDscError> {
        let idx = self.layout.iter().position(|&d| d == dim).ok_or_else(|| {
            SuperDscError(format!(
                "TensorArg '{}': device_extent dim '{dim}' is not in layout {:?}",
                self.name, self.layout
            ))
        })?;
        self.device_extent[idx] = Some(phys_extent);
        Ok(self)
    }
}

/// Rank-erased tensor arg (ranks ≤ 4 in practice), so `OpSpec.args` can hold a
/// heterogeneous-rank `Vec`.
#[derive(Clone, Debug)]
pub enum AnyTensorArg {
    /// Rank-1: a SINGLE dim that IS the stick (a reduced scalar laid out as one
    /// 64-stick). torch-spyre's REAL rmsnorm reduced-scalar ops (reduce output / add
    /// / rsqrt) are rank-1 `N_={mb:64}` / layout=stick=`["mb"]` — emitting them rank-3
    /// `[mb=1,out=64,y=1]` (the phantom row/y dims wrapping the stick) collapses the
    /// SFP Newton-Raphson to the seed (rsqrt→1/x). The systolic matmul tolerates
    /// rank-3; the SFP transcendental does not.
    R1(TensorArg<1>),
    R2(TensorArg<2>),
    R3(TensorArg<3>),
    R4(TensorArg<4>),
}

/// A uniform VIEW over an `AnyTensorArg`, so `emit_sdsc` reads the fields without
/// matching the rank everywhere. Borrows from the arg.
pub struct ArgView<'a> {
    pub is_input: bool,
    pub name: &'a str,
    pub role: Role,
    pub scale: &'a [Scale],
    pub device_dims: &'a [IterSym],
    pub layout: &'a [&'static str],
    pub stick: &'static str,
    pub allocation: Allocation,
    /// Slice offset in elements along the stick axis (0 = whole tensor).
    pub offset_elems: u32,
    /// The dataspace dtype — the emitter reads `df.word_length()` / `df.elems_per_stick()` /
    /// `df.dataformat()` instead of sniffing the name for a `_fp8`/`_fp32` suffix.
    pub df: Df,
    /// FORCE RowBlocked device addressing regardless of rank (the fp32 island `ElementArrangement`).
    pub row_blocked: bool,
    /// Per-`layout`-dim physical device extent override (torch-spyre `arg.device_size`) — `None` means
    /// "physical extent == this op's own iteration extent for that dim" (see [`TensorArg::device_extent`]).
    pub device_extent: &'a [Option<u32>],
}

impl AnyTensorArg {
    /// Rank-erased [`TensorArg::with_row_blocked`] setter.
    pub fn set_row_blocked(&mut self, rb: bool) {
        match self {
            AnyTensorArg::R1(t) => t.row_blocked = rb,
            AnyTensorArg::R2(t) => t.row_blocked = rb,
            AnyTensorArg::R3(t) => t.row_blocked = rb,
            AnyTensorArg::R4(t) => t.row_blocked = rb,
        }
    }

    /// Set the dtype on the wrapped arg (rank-erased) — the typed way for a matmul builder to declare
    /// its fp8/int8 operands after construction, without a per-rank `with_df`. Mirrors [`TensorArg::with_df`].
    pub fn set_df(&mut self, df: Df) {
        match self {
            AnyTensorArg::R1(t) => t.df = df,
            AnyTensorArg::R2(t) => t.df = df,
            AnyTensorArg::R3(t) => t.df = df,
            AnyTensorArg::R4(t) => t.df = df,
        }
    }

    /// Rank-erased device-buffer NAME setter — for cloning an EXISTING arg's whole shape (scale/
    /// device_dims/layout/stick/allocation/df) onto a DIFFERENT device buffer. The intended use is a
    /// fused epilogue's extra operand (mask/bias/residual): `bmm.ddl` declares `bnA`/`bnB`/`bias`/
    /// `resadd`/`resadd2` with the SAME layout as `%outtensor` (not a fresh shape), so cloning the
    /// real OUTPUT arg and renaming it is the correct construction, not a shortcut around one.
    pub fn set_name(&mut self, name: String) {
        match self {
            AnyTensorArg::R1(t) => t.name = name,
            AnyTensorArg::R2(t) => t.name = name,
            AnyTensorArg::R3(t) => t.name = name,
            AnyTensorArg::R4(t) => t.name = name,
        }
    }

    /// Rank-erased SLICE offset setter (elements along the stick axis). Mirrors [`TensorArg::with_offset`]
    /// — pairs with [`Self::set_name`] to address a per-head/per-group slice of a cloned-shape operand.
    pub fn set_offset_elems(&mut self, offset_elems: u32) {
        match self {
            AnyTensorArg::R1(t) => t.offset_elems = offset_elems,
            AnyTensorArg::R2(t) => t.offset_elems = offset_elems,
            AnyTensorArg::R3(t) => t.offset_elems = offset_elems,
            AnyTensorArg::R4(t) => t.offset_elems = offset_elems,
        }
    }

    /// Rank-erased ROLE setter. The epilogue's extra operand is tagged [`Role::Output`] to match
    /// `bmm.ddl`'s own convention (bias/bnA/bnB/resadd share the OUTPUT dsType/layout bucket, not
    /// INPUT) — confirmed against the golden `sdsc_bmm_lxopt.json` fixture, whose bias arg carries
    /// `dsType_: "OUTPUT"`.
    pub fn set_role(&mut self, role: Role) {
        match self {
            AnyTensorArg::R1(t) => t.role = role,
            AnyTensorArg::R2(t) => t.role = role,
            AnyTensorArg::R3(t) => t.role = role,
            AnyTensorArg::R4(t) => t.role = role,
        }
    }

    /// Rank-erased is_input setter — the epilogue's extra operand is READ, never written, by the
    /// matmul's own compute (only the fused epilogue's SECOND `computeOp_` entry touches it), so it
    /// must clear the `is_input=false` a cloned OUTPUT arg carries or the collision guard in
    /// `emit_sdsc` (which only checks non-input OUTPUT-role args) would wrongly compare its address
    /// against the real output's.
    pub fn set_is_input(&mut self, is_input: bool) {
        match self {
            AnyTensorArg::R1(t) => t.is_input = is_input,
            AnyTensorArg::R2(t) => t.is_input = is_input,
            AnyTensorArg::R3(t) => t.is_input = is_input,
            AnyTensorArg::R4(t) => t.is_input = is_input,
        }
    }

    /// Rank-erased "mark ONE named dim broadcast/reduced" setter — finds `dim` in this arg's OWN
    /// `layout` (never a positional guess: `layout` and `scale` are the SAME `D`-length arrays by
    /// construction, so the position `layout` answers is the ONLY correct index into `scale`) and
    /// overwrites `scale[idx]`. This is what lets a fused epilogue's operand express the SAME
    /// broadcast semantics [`EwOperand::scale`] already gives a STANDALONE pointwise op's operand
    /// (`mb_broadcast` → `RedNonStick`, `out_broadcast` → `RedStick`) while still being a CLONE of the
    /// matmul's own output arg (same `device_dims`/`layout`/stick — proven `sdsc_bmm_lxopt.json`
    /// confirms a matmul's own output layout is `["mb","out","y"]`, the SAME names/order
    /// `pointwise_broadcast_opspec_df`'s rank-3 operand uses, so the SAME dim name means the SAME
    /// axis in both places).
    ///
    /// PANICS if `dim` is not in this arg's layout. An earlier version of this method silently
    /// no-op'd here — reasoning "every current call site names a dim the golden confirms is
    /// present" — and that confidence was WRONG in practice: the pmask fusion's first landing named
    /// only `"mb"` and forgot `"y"` (the GQA-group batch axis a batched score matmul ALSO needs
    /// broadcast, since pmask is head-independent within a group too), and the silent no-op meant
    /// the emitter built a plausible-looking, wrong SDSC JSON with zero signal — garbled generation
    /// on the card was the FIRST anyone noticed. A caller asking to broadcast a dim this arg does
    /// not have is a real bug, and it must fail here, in `cargo build`/local bake, not on hardware.
    pub fn set_scale_for_dim(&mut self, dim: &str, scale: Scale) {
        match self {
            AnyTensorArg::R1(t) => {
                let i = t.layout.iter().position(|&d| d == dim).unwrap_or_else(|| {
                    panic!(
                        "set_scale_for_dim: '{dim}' not in layout {:?} (arg '{}')",
                        t.layout, t.name
                    )
                });
                t.scale[i] = scale;
            }
            AnyTensorArg::R2(t) => {
                let i = t.layout.iter().position(|&d| d == dim).unwrap_or_else(|| {
                    panic!(
                        "set_scale_for_dim: '{dim}' not in layout {:?} (arg '{}')",
                        t.layout, t.name
                    )
                });
                t.scale[i] = scale;
            }
            AnyTensorArg::R3(t) => {
                let i = t.layout.iter().position(|&d| d == dim).unwrap_or_else(|| {
                    panic!(
                        "set_scale_for_dim: '{dim}' not in layout {:?} (arg '{}')",
                        t.layout, t.name
                    )
                });
                t.scale[i] = scale;
            }
            AnyTensorArg::R4(t) => {
                let i = t.layout.iter().position(|&d| d == dim).unwrap_or_else(|| {
                    panic!(
                        "set_scale_for_dim: '{dim}' not in layout {:?} (arg '{}')",
                        t.layout, t.name
                    )
                });
                t.scale[i] = scale;
            }
        }
    }

    pub fn view(&self) -> ArgView<'_> {
        match self {
            AnyTensorArg::R1(t) => ArgView {
                is_input: t.is_input,
                name: &t.name,
                role: t.role,
                scale: &t.scale,
                device_dims: &t.device_dims,
                layout: &t.layout,
                stick: t.stick,
                allocation: t.allocation,
                offset_elems: t.offset_elems,
                df: t.df,
                row_blocked: t.row_blocked,
                device_extent: &t.device_extent,
            },
            AnyTensorArg::R2(t) => ArgView {
                is_input: t.is_input,
                name: &t.name,
                role: t.role,
                scale: &t.scale,
                device_dims: &t.device_dims,
                layout: &t.layout,
                stick: t.stick,
                allocation: t.allocation,
                offset_elems: t.offset_elems,
                df: t.df,
                row_blocked: t.row_blocked,
                device_extent: &t.device_extent,
            },
            AnyTensorArg::R3(t) => ArgView {
                is_input: t.is_input,
                name: &t.name,
                role: t.role,
                scale: &t.scale,
                device_dims: &t.device_dims,
                layout: &t.layout,
                stick: t.stick,
                allocation: t.allocation,
                offset_elems: t.offset_elems,
                df: t.df,
                row_blocked: t.row_blocked,
                device_extent: &t.device_extent,
            },
            AnyTensorArg::R4(t) => ArgView {
                is_input: t.is_input,
                name: &t.name,
                role: t.role,
                scale: &t.scale,
                device_dims: &t.device_dims,
                layout: &t.layout,
                stick: t.stick,
                allocation: t.allocation,
                offset_elems: t.offset_elems,
                df: t.df,
                row_blocked: t.row_blocked,
                device_extent: &t.device_extent,
            },
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// OpInfo — op-specific aux (only the sfp constant-table request today).
// ───────────────────────────────────────────────────────────────────────────

/// Op-specific aux, serialized into `constantInfo_` ONLY when non-empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpInfo {
    /// No aux — `constantInfo_` = `{}`.
    #[default]
    None,
    /// Emit the SFP polynomial constant table (transcendental sfp ops).
    SfpConstTable,
    /// Emit the `scaling_factor` external constant (= 1/N, fp16, replicated into
    /// the u32 word) that the `sum`/`mean`/`max` reduce template
    /// (`summeanmaxexx2.ddl`) requires — else DtException "Missing external constant
    /// in DSC: scaling_factor" (ddl_conversion.cpp:718). `mean` → 1/N; `sum`/`max`
    /// → 1.0. The value is the PACKED fp16 bits (`bits<<16 | bits`).
    ReduceScaling(u32),
    /// Same `scaling_factor` external constant, but for an **fp32** reduce (the
    /// mq>1 rmsnorm mean(x²)). The const's `dataFormat_` MUST match the op's
    /// data_format (torch-spyre `encode_constant(value, data_format)`,
    /// compute_ops.py:127) — a SEN169_FP16 const feeding an fp32 op makes it a
    /// MIXED [fp16,fp32] op → DD2 "Unsupported result precision conversion".
    /// The value is the raw IEEE_FP32 bits (`f32::to_bits`).
    ReduceScalingFp32(u32),
    /// A SECOND `computeOp_` entry chained onto this (matmul) op's own PE/SFP-resident output,
    /// mirroring IBM's own golden fixture (`sdsc_bmm_lxopt.json`'s `MatMul_122`: `batchmatmul` +
    /// `biasadd`, the second reading AND writing the first's own output labeled-ds in place).
    /// `arg_idx` names the extra operand ALREADY inserted into
    /// `OpSpec.args` (its own shape/layout cloned from the real output's, per `bmm.ddl`'s bnA/bnB/
    /// bias/resadd declaring the SAME layout as `%outtensor` — this is not a fresh addressing
    /// derivation, it reuses the output's own proven placement with a different device name). The
    /// epilogue reads `[this op's own output, args[arg_idx]]` and writes back to the SAME output —
    /// no new HBM round trip, matching the DDL's one-`lxsu`-store-at-the-end structure.
    FusedEpilogue {
        arg_idx: usize,
        op_func: EpilogueOpFunc,
        /// The SECOND chained stage, when the op has one — `bmm.ddl`'s `stradd2_op`. `None` is the
        /// single-stage form every shipped fusion uses today.
        second: Option<EpilogueStage>,
    },
}

/// One chained epilogue stage: the aux operand's slot in [`OpSpec::args`] and which DDL aux op reads
/// it. Built ONLY by [`OpSpec::attach_fused_epilogue`], which records `arg_idx` as it inserts the
/// operand — never re-derived from the arg list afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpilogueStage {
    pub arg_idx: usize,
    pub op_func: EpilogueOpFunc,
}

/// What ONE epilogue stage needs from its caller. The aux operand's shape/layout is NOT here: it is
/// cloned from the real output's proven placement (`bmm.ddl` declares `bnA`/`bnB`/`bias`/`resadd` with
/// the SAME layout as `%outtensor`), so a stage contributes a name, an offset, and a broadcast
/// pattern — never a fresh addressing derivation.
pub struct EpilogueSpec<'a> {
    pub operand_name: String,
    pub offset_elems: u32,
    pub op_func: EpilogueOpFunc,
    pub broadcast_dims: &'a [(&'a str, Scale)],
    pub broadcast_batch: bool,
}

/// ⭐⭐⭐ THE CHAIN LENGTH `bmm.ddl` PHYSICALLY HAS, AS A TYPE. The DDL exposes exactly TWO
/// `stridedadd` registers (`stradd_op`, `stradd2_op`); a third stage has nowhere to live, so it is
/// UNREPRESENTABLE here rather than rejected at runtime. Equally, there is no empty variant — an op
/// with no epilogue does not call [`OpSpec::attach_fused_epilogue`] at all, so "attached an epilogue
/// with zero stages" is not a state the emitter can reach.
pub enum EpilogueSpecs<'a> {
    One(EpilogueSpec<'a>),
    Two(EpilogueSpec<'a>, EpilogueSpec<'a>),
}

/// The DDL `opFuncName` a fused epilogue selects — SEALED to the menu `bmm.ddl` actually exposes
/// (verified against IBM's golden `sdsc_bmm_lxopt.json` + the DDL source), never a hand-typed
/// string. A wrong opFuncName string is a runtime `DtException Unrecognized opFunc`, not a
/// build error — this enum is what turns "which DDL aux op fires" back into a closed,
/// compile-time-checked set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpilogueOpFunc {
    /// `bmm.ddl`'s `stridedadd`: `sfp_output_lrf += operand` in place, SFP-resident, one PE/SFP-LRF
    /// register — the shape a full-tensor (never broadcast) second operand needs, per the DDL's own
    /// `bnA`/`bnB`/`bias`/`resadd`/`resadd2` declaring the SAME layout as `%outtensor`. `bmm.ddl` allows
    /// TWO independent `stridedadd` instances (`stradd_op`/`stradd2_op`) chained onto one matmul; this
    /// variant carries one at a time — nothing has needed a second yet.
    StridedAdd,
}

impl EpilogueOpFunc {
    pub fn name(self) -> &'static str {
        match self {
            EpilogueOpFunc::StridedAdd => "stridedadd",
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// OpSpec — the typed frontend op.
// ───────────────────────────────────────────────────────────────────────────

/// The TYPED FRONTEND op. Carries ONLY what the frontend decides; everything the
/// L3 scheduler fills (corelet split, time fold, lx/hbm sizes, register-file
/// residency) is ABSENT by construction. Faithful to op_spec.py `OpSpec`, but the
/// `iteration_space` tuple `(range, work_div)` is REPLACED by the typed
/// [`WorkPlan`] so over-specification is unrepresentable.
#[derive(Clone, Debug)]
pub struct OpSpec {
    pub op: OpFunc,
    pub is_reduction: bool,
    /// The named iteration space + validated work-division (witnesses (a)(b)(f)).
    pub iter: WorkPlan,
    /// Ordered tensor args; `arg_index == position` is a type invariant.
    pub args: Vec<AnyTensorArg>,
    pub op_info: OpInfo,
    /// Iteration symbols tiled by an enclosing loop. Subset-of-`iter` by type.
    /// Populated from `time_tile.dim()` when `time_tile` is `Some` (the LX-fit
    /// coarse-tile pass), empty otherwise.
    pub tiled_symbols: Vec<IterSym>,
    /// The LX-fit TIME-TILE for this op (the **TimeTile** witness). `None` = the
    /// per-core tile fits [`USABLE_LX_BYTES`] in one trip (today's single-shot
    /// path). `Some(TimeTile{count>1, dim})` = the op's `dim` extent is sliced
    /// into `count` per-time trips so each per-time tile fits the scratchpad —
    /// produced ONLY by [`WorkPlan::time_tile_for_lx`], so "doesn't fit LX" is a
    /// `cargo build` `Err`, never an on-card DtException 1535.
    pub time_tile: Option<TimeTile>,
}

impl OpSpec {
    pub fn ex_unit(&self) -> ExUnit {
        self.op.ex_unit()
    }

    /// Attach a fused pointwise epilogue (mask-add / bias-add / residual-add) to THIS matmul, mirroring
    /// `sdsc_bmm_lxopt.json`'s `MatMul_122` golden: the extra operand is inserted as a CLONE of this
    /// op's own output arg (same scale/device_dims/layout/stick/allocation/df — `bmm.ddl` declares
    /// `bnA`/`bnB`/`bias`/`resadd`/`resadd2` with the output's own layout, not a fresh shape), renamed
    /// to `operand_name` and offset to `offset_elems`, then inserted immediately BEFORE the real output
    /// so `out_idx` (always "the last arg") still resolves correctly. Marks it `is_input=true` (read-only
    /// — the epilogue mutates the output in place, not this operand) and `Role::Output` (the golden's own
    /// bias arg carries `dsType_: "OUTPUT"`, not "INPUT").
    ///
    /// `broadcast_dims` marks specific NAMED dims of the cloned operand as broadcast/reduced —
    /// e.g. `[("mb", Scale::RedNonStick)]` for an operand that repeats the SAME row across every row
    /// this op sweeps (the standalone pointwise mechanism's `EwOperand::mb_broadcast`, reused here so
    /// a fused epilogue's operand can express the identical semantics a STANDALONE op's operand
    /// already does — see [`AnyTensorArg::set_scale_for_dim`]). Empty for the common full-tensor case.
    /// Every dim named here MUST exist on the cloned operand's layout (`set_scale_for_dim` panics
    /// otherwise) — this is for dims `matmul_dims` ALWAYS emits (`"mb"`, `"out"`), never `"y"`.
    ///
    /// `broadcast_batch`: mark the op's BATCH axis broadcast too, IF one exists — see
    /// [`Self::batch_dim_name`]. This is NOT a `("y", Scale)` entry in `broadcast_dims`: whether "y"
    /// exists depends on `batch > 1` (`matmul_dims`'s own "NO y at batch==1" rule), a fact this
    /// `OpSpec`'s own `self.iter` ALREADY carries authoritatively — asking the CALLER to separately
    /// recompute "is this op batched" to decide whether "y" is safe to name is exactly the bug this
    /// method shipped once already (attn.rs's pmask fusion forgot the check on its first landing;
    /// `set_scale_for_dim`'s panic caught a SECOND version of the same mistake immediately after).
    /// Folding the presence check into the one place that already knows the answer turns "two
    /// computations that must agree" into "one computation, asked once" — this flag can no longer
    /// disagree with `matmul_dims` about whether "y" exists, because it asks the same `self.iter`.
    ///
    /// Only sound when `self.time() == 1` (single-shot, untiled) — the mask/bias tensor is NOT threaded
    /// through `rewrite_op_for_time_tile`'s per-trip affine-stride machinery, so a tiled matmul must not
    /// call this (the caller decides; this method does not check, since `OpSpec` alone cannot see whether
    /// `emit_sdsc_tiled` will time-tile it).
    /// ⛔ TAKES THE WHOLE CHAIN AT ONCE, NEVER ONE STAGE PER CALL. The old one-stage-per-call form
    /// assigned `self.op_info` outright, so a second call SILENTLY DISCARDED the first stage — the
    /// caller got one epilogue where it asked for two and nothing said so. Handing the chain over in
    /// one call removes the sequence in which that could happen: there is no "already has a stage"
    /// state to clobber, and [`EpilogueSpecs`] makes a third stage (which `bmm.ddl` does not have a
    /// register for) unrepresentable rather than runtime-rejected.
    pub fn attach_fused_epilogue(&mut self, stages: EpilogueSpecs<'_>) {
        // Each stage's operand is inserted BEFORE the output, so the output keeps sliding right and
        // stage `i` lands at `out_idx + i` — recorded as it is built, never re-derived afterwards.
        let mut insert = |spec: &EpilogueSpec<'_>| -> EpilogueStage {
            let out_idx = self.args.len().saturating_sub(1);
            let mut operand = self.args[out_idx].clone();
            operand.set_name(spec.operand_name.clone());
            operand.set_offset_elems(spec.offset_elems);
            operand.set_is_input(true);
            operand.set_role(Role::Output);
            for &(dim, scale) in spec.broadcast_dims {
                operand.set_scale_for_dim(dim, scale);
            }
            if spec.broadcast_batch
                && let Some(batch_dim) = self.batch_dim_name()
            {
                operand.set_scale_for_dim(batch_dim, Scale::RedNonStick);
            }
            self.args.insert(out_idx, operand);
            EpilogueStage {
                arg_idx: out_idx,
                op_func: spec.op_func,
            }
        };
        // Destructured, not iterated: the first stage is present BY TYPE, so there is no empty case
        // to `expect` away.
        let (first, second) = match stages {
            EpilogueSpecs::One(a) => (insert(&a), None),
            EpilogueSpecs::Two(a, b) => {
                let fa = insert(&a);
                (fa, Some(insert(&b)))
            }
        };
        self.op_info = OpInfo::FusedEpilogue {
            arg_idx: first.arg_idx,
            op_func: first.op_func,
            second,
        };
    }

    /// The name of THIS op's own batch axis, if it has one — `Some("y")` when `self.iter` carries
    /// that dim, `None` when it doesn't (`matmul_dims`'s own rule: batch==1 omits the dim entirely,
    /// not merely sizes it 1). The SAME `self.iter` `matmul_dims`/`WorkPlan::divide` already built,
    /// so this can never disagree with the decision that determined whether "y" exists — the ONE
    /// question, asked once, [`Self::attach_fused_epilogue`]'s `broadcast_batch` relies on instead of
    /// making the caller re-derive "is this op batched" (`batch > 1`) in a SECOND place that could
    /// drift out of sync with `matmul_dims`'s own answer. "y" is hardcoded here (not imported from
    /// `matmul::walk::YAxis::NAME`) deliberately: this module is the FOUNDATION `matmul/opspec.rs`
    /// builds on, so importing from it backwards would invert that layering for one string constant.
    pub fn batch_dim_name(&self) -> Option<&'static str> {
        self.iter
            .dims()
            .iter()
            .find(|d| d.name == "y")
            .map(|d| d.name)
    }

    /// The number of time-tile trips (1 when not time-tiled). Drives the
    /// `scf.for` bound in `bundle.mlir` and the per-time `out` divisor in
    /// `emit_sdsc`.
    pub fn time(&self) -> u32 {
        self.time_tile.map(|t| t.count()).unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stick_extent_rejects_sub_stick() {
        assert!(StickExtent::<Fp16>::new(64).is_ok());
        assert!(StickExtent::<Fp16>::new(384).is_ok());
        assert!(StickExtent::<Fp16>::new(65).is_err()); // not a multiple of 64
        assert!(StickExtent::<Fp16>::new(0).is_err());
        assert_eq!(StickExtent::<Fp16>::new(384).unwrap().stick_count(), 6);
    }

    #[test]
    fn device_tile_layout_matches_torch_spyre() {
        // Confirmed reference value (torch-spyre tests/tensor/test_tensor_layout.py):
        // a 2-D [512,256] fp16 tensor → device_size [4,512,64], stride_map [64,256,1].
        let t = DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[512, 256]).unwrap();
        assert_eq!(t.device_size(), vec![4, 512, 64]); // [out/64, in, 64]
        assert_eq!(t.stride_map(), vec![64, 256, 1]); // [stk, out, 1]
        assert_eq!(t.device_bytes(), 4 * 512 * 64 * 2);

        // The per-core multiplier for the split stick dim `out` is the product of the
        // NON-stick extents (= in), NOT in·64 — scratchy folds the 64-elem per_core in
        // separately. The CRITICAL invariant (the factor-64 bug the workflow critic
        // caught): per_core_extent(out) · per_core_stride_elems(out) MUST equal the
        // device tile span prod(device_size[1..]) so a core lands on a tile boundary.
        assert_eq!(t.per_core_stride_elems("out"), 512); // = in, NOT in*64
        let per_core_out = 256 / 4; // out split 4 ways → 64 elems/core (one stick)
        let tile_span: u64 = t.device_size()[1..].iter().product(); // in·64 = 32768
        assert_eq!(
            per_core_out as u64 * t.per_core_stride_elems("out"),
            tile_span
        );

        // q_proj [576,576]: device [9,576,64]; stride 576 (=in); 9 out-tiles.
        let q = DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[576, 576]).unwrap();
        assert_eq!(q.device_size(), vec![9, 576, 64]);
        assert_eq!(q.per_core_stride_elems("out"), 576);
        // a core owning 1 out-stick (64) lands at tile t: 64·576 = 36864 elems/tile.
        assert_eq!(64u64 * q.per_core_stride_elems("out"), 576 * 64);

        // Non-stick-multiple stick extent is a cargo-build Err (StickExtent witness).
        assert!(DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[576, 100]).is_err());
        // Stick axis absent from layout → Err.
        assert!(DeviceTileLayout::<Fp16>::new(&["in", "out"], "zzz", &[576, 576]).is_err());
    }

    #[test]
    fn workplan_caps_cores_and_sticks() {
        let dims = [
            ItDim {
                name: "mb",
                size: 384,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: 384,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            },
        ];
        // A splitter that over-subscribes is rejected (witness b).
        let bad = WorkPlan::divide(&dims, MaxCores::<32>, |_, _| {
            BTreeMap::from([("mb", 16), ("out", 4)]) // 64 > 32
        });
        assert!(bad.is_err());
        // A valid split is accepted and ≤32 by type.
        let ok = WorkPlan::divide(&dims, MaxCores::<32>, |_, _| {
            BTreeMap::from([("mb", 16), ("out", 2)]) // 32
        })
        .unwrap();
        assert_eq!(ok.cores_used().get(), 32);
        // A stick dim split beyond its stick count is rejected (≥1 stick/core).
        let sub = WorkPlan::divide(&dims, MaxCores::<32>, |_, _| {
            BTreeMap::from([("out", 7)]) // 384 = 6 sticks; 7 > 6
        });
        assert!(sub.is_err());
    }

    #[test]
    fn ex_unit_sealed_by_op() {
        assert_eq!(OpFunc::Matmul.ex_unit().as_str(), "pt");
        assert_eq!(OpFunc::BatchMatmul.ex_unit().as_str(), "pt");
        assert_eq!(OpFunc::Add.ex_unit().as_str(), "sfp");
        assert_eq!(OpFunc::Silu.ex_unit().as_str(), "sfp");
    }

    #[test]
    fn fold_set_is_single_source() {
        let f = SdscFoldSet::new(CoresUsed(32));
        assert_eq!(f.core_fold(), 32);
        assert_eq!(f.corelet_fold(), 1);
        assert_eq!(f.time_fold(), 1);
    }

    /// Build the matmul iteration dims (mb=M, out=N stick, in=K reduction-stick,
    /// x=batch) and a validated WorkPlan with an explicit split, for the
    /// time-tiling tests.
    fn matmul_plan(m: u32, n: u32, k: u32, batch: u32, mb_split: u32, out_split: u32) -> WorkPlan {
        WorkPlan::divide(
            &matmul_dims(m, n, k, batch),
            MaxCores::<MAX_CORES>,
            |_, _| BTreeMap::from([("mb", mb_split), ("out", out_split)]),
        )
        .unwrap()
    }

    /// [`matmul_plan`]'s iteration vocabulary on its own — what
    /// [`WorkPlan::divide_and_time_tile_for_lx`] takes, since it owns the division itself.
    fn matmul_dims(m: u32, n: u32, k: u32, batch: u32) -> [ItDim; 5] {
        [
            ItDim {
                name: "mb",
                size: m,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: n,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            },
            ItDim {
                name: "in",
                size: k,
                is_reduction: true,
                is_stick: true,
                df: Df::Fp16,
            },
            ItDim {
                name: "x",
                size: batch,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "y",
                size: 1,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
        ]
    }

    /// The matmul per-core LX residency (a + w + o), fp16 = 2 B, INCLUDING the
    /// batch/x factor. KERNEL `w` is NOT divided by the mb split (it dominates).
    /// `out_per_time` is the per-time slice of the per-core `out` extent.
    fn matmul_resident(plan: &WorkPlan, out_per_time: u32) -> u64 {
        let per_core_mb = plan.per_core_extent("mb") as u64;
        let k = plan.extent("in") as u64;
        let batch = plan.extent("x") as u64;
        let opt = out_per_time as u64;
        let a = per_core_mb * k * batch * FP16_BYTES;
        let w = k * opt * batch * FP16_BYTES;
        let o = per_core_mb * opt * batch * FP16_BYTES;
        a + w + o
    }

    #[test]
    fn time_tile_ffn_up_proj_tiles_to_multiple_of_64() {
        // FFN up-proj 384×2048×2048 (batch 1). The cost split here is mb16×out2
        // → per_core_mb=24, per_core_out=1024. Full per-core resident:
        //   a = 24*2048*2     = 98_304
        //   w = 2048*1024*2   = 4_194_304   (KERNEL dominates — > USABLE_LX alone)
        //   o = 24*1024*2     = 49_152
        //   resident ≈ 4.34 MiB ≫ 1_677_721 → MUST time-tile.
        let plan = matmul_plan(384, 2048, 2048, 1, 16, 2);
        assert_eq!(plan.per_core_extent("out"), 1024);
        let tt = plan
            .time_tile_for_lx(matmul_resident, "out")
            .expect("ffn up-proj must tile, not Err")
            .expect("ffn up-proj must need a time-tile (Some)");
        assert!(tt.count() > 1, "time count must be > 1");
        assert_eq!(tt.dim().name(), "out");
        let out_per_time = 1024 / tt.count();
        assert_eq!(
            out_per_time % 64,
            0,
            "out_per_time must be a whole 64-fp16 stick"
        );
        assert!(
            matmul_resident(&plan, out_per_time) <= USABLE_LX_BYTES,
            "per-time tile must fit USABLE_LX ({} B)",
            USABLE_LX_BYTES
        );
        // It is the SMALLEST such time: time-1 (=count/?) would overflow. Verify
        // the previous divisor does NOT fit.
        // out=1024 sticks=16; divisors ascending: 2,4,8,16. w at time=2 is
        // 2048*512*2 = 2_097_152 > USABLE_LX, so time=2 fails; time=4 (out=256):
        // w = 2048*256*2 = 1_048_576, a+o small → fits. Expect count == 4.
        assert_eq!(tt.count(), 4, "smallest fitting time for ffn up-proj is 4");
    }

    #[test]
    fn time_tile_bmm_fits_in_one_trip() {
        // The known-good bmm: M384 N384 K64 batch16, split mb16×out2 →
        // per_core_mb=24, per_core_out=192. Resident (INCLUDING batch):
        //   a = 24*64*16*2   = 49_152
        //   w = 64*192*16*2  = 393_216
        //   o = 24*192*16*2  = 147_456
        //   resident = 589_824 B (≈576 KiB) < 1_677_721 → time = 1 (None).
        let plan = matmul_plan(384, 384, 64, 16, 16, 2);
        assert_eq!(plan.per_core_extent("out"), 192);
        let resident = matmul_resident(&plan, 192);
        assert_eq!(resident, 589_824, "bmm per-core resident incl. batch");
        assert!(resident < USABLE_LX_BYTES);
        let tt = plan.time_tile_for_lx(matmul_resident, "out").unwrap();
        assert!(
            tt.is_none(),
            "bmm fits LX in one trip — time=1, no TimeTile"
        );
    }

    #[test]
    fn time_tile_impossibly_large_matmul_is_build_err() {
        // An lm_head-scale single core tile that cannot fit even at single sticks:
        // pretend the split leaves per_core_out huge AND K huge so that even
        // out_per_time=64 overflows: w = K * 64 * batch * 2 must exceed USABLE_LX.
        // K=49152, batch=1: w at out=64 = 49152*64*2 = 6_291_456 B ≫ 1.6 MiB → Err.
        let plan = matmul_plan(64, 4096, 49152, 1, 1, 1);
        let r = plan.time_tile_for_lx(matmul_resident, "out");
        assert!(
            r.is_err(),
            "a matmul too big to tile on N alone must be a build Err (DtException-1535 guard), got {r:?}"
        );
    }

    // ── the reduction-split LX repair (`divide_and_time_tile_for_lx`) ──────────────────────────

    /// The residency the LIVE emitter uses (`TileOp::resident_bytes` → `matmul_lx_resident_generic`):
    /// the reduction extent is the PER-CORE one, which is what makes a reduction split a lever on the
    /// per-core tile at all. [`matmul_resident`] above deliberately reads the FULL `in` extent (the
    /// "K is not divided" form the older tests were written against), so it cannot exercise the repair.
    fn matmul_resident_per_core_k(plan: &WorkPlan, out_per_time: u32) -> u64 {
        let per_core_mb = plan.per_core_extent("mb") as u64;
        let k = plan.per_core_extent("in") as u64;
        let batch = plan.extent("x").max(plan.extent("y")).max(1) as u64;
        let opt = out_per_time as u64;
        per_core_mb * k * batch * FP16_BYTES
            + k * opt * batch * FP16_BYTES
            + per_core_mb * opt * batch * FP16_BYTES
    }

    /// Give every core its own `out` stick-slice and never split `mb` — the live decode-batch policy,
    /// reduced to what these tests need. It HONOURS the core budget it is handed, which is what lets
    /// the repair trade cores away from it.
    fn out_only_splitter(dims: &[ItDim], max_cores: u32) -> BTreeMap<&'static str, u32> {
        let out = dims.iter().find(|d| d.name == "out").map_or(1, |d| d.size);
        let sticks = (out / 64).max(1);
        let mut s = max_cores.min(sticks).max(1);
        while s > 1 && !sticks.is_multiple_of(s) {
            s -= 1;
        }
        BTreeMap::from([("out", s)])
    }

    /// ⭐ granite-3.1-8b fp16 down_proj (k=12800, n=4096) at a TWO-ROW decode batch — the shape that
    /// refused to bake. The control comes first: the cost model's own `out=32` division has NO
    /// time-tile that fits (this is the exact `Err` the build reported), and the repair then places it
    /// with the SMALLEST reduction split that does.
    #[test]
    fn lx_repair_places_granite_8b_down_proj_at_two_decode_rows() {
        let dims = matmul_dims(2, 4096, 12800, 1);

        // CONTROL — without the repair this shape is a build refusal, so the assertions below are
        // measuring the repair and not a plan that fitted all along.
        let unrepaired = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, out_only_splitter).unwrap();
        assert_eq!(unrepaired.split_of("out"), 32);
        assert_eq!(
            matmul_resident_per_core_k(&unrepaired, 64),
            1_689_856,
            "the measured per-core resident at one out-stick"
        );
        assert!(
            unrepaired
                .time_tile_for_lx(matmul_resident_per_core_k, "out")
                .is_err(),
            "out-tiling alone must still refuse this shape"
        );

        let (plan, tt) = WorkPlan::divide_and_time_tile_for_lx(
            &dims,
            MaxCores::<MAX_CORES>,
            "out",
            out_only_splitter,
            matmul_resident_per_core_k,
        )
        .expect("the reduction-split repair must place the down_proj");
        assert_eq!(
            plan.split_of("in"),
            2,
            "the SMALLEST reduction split that fits (6400 per core), not a bigger one"
        );
        assert_eq!(plan.per_core_extent("in"), 6400);
        assert_eq!(
            plan.cores_used().get(),
            32,
            "the reduction cores come OUT of the out split — the card stays full"
        );
        let tt = tt.expect("a 6400-deep reduction still needs an out time-tile");
        let out_per_time = plan.per_core_extent("out") / tt.count();
        assert!(
            out_per_time.is_multiple_of(64),
            "whole fp16 sticks per trip"
        );
        assert!(
            matmul_resident_per_core_k(&plan, out_per_time) <= USABLE_LX_BYTES,
            "the repaired per-time tile must actually fit LX"
        );
    }

    /// An op the cost model already placed is returned UNTOUCHED — the repair is reachable only from
    /// a refusal, so every bundle that bakes today keeps its division and its trip count.
    #[test]
    fn lx_repair_leaves_a_fitting_op_byte_identical() {
        let dims = matmul_dims(384, 384, 64, 16);
        let fixed = |_: &[ItDim], _: u32| BTreeMap::from([("mb", 16u32), ("out", 2u32)]);
        let (plan, tt) = WorkPlan::divide_and_time_tile_for_lx(
            &dims,
            MaxCores::<MAX_CORES>,
            "out",
            fixed,
            matmul_resident_per_core_k,
        )
        .expect("the known-good bmm fits");
        assert!(tt.is_none(), "it fitted in one trip before and still does");
        assert_eq!(
            plan.splits(),
            &BTreeMap::from([("mb", 16u32), ("out", 2u32)])
        );
        assert_eq!(plan.split_of("in"), 1, "no reduction split was introduced");
    }

    /// NEGATIVE CONTROL: a shape whose OUTPUT tile alone busts LX (`per_core_mb · stick · 2` =
    /// 2,097,152 B at mb=16384 unsplit) cannot be rescued by any reduction split, because the
    /// reduction is absent from the output term. It stays a `cargo build` failure — the Stage-2 guard
    /// is narrowed by the repair, not removed — and the message says the ladder was walked.
    #[test]
    fn lx_repair_still_refuses_a_shape_that_needs_k_time() {
        let dims = matmul_dims(16384, 4096, 4096, 1);
        let fixed = |_: &[ItDim], _: u32| BTreeMap::new();
        let e = WorkPlan::divide_and_time_tile_for_lx(
            &dims,
            MaxCores::<MAX_CORES>,
            "out",
            fixed,
            matmul_resident_per_core_k,
        )
        .expect_err("no reduction split can shrink the OUTPUT tile");
        assert!(
            e.0.contains("reduction-core split was also tried"),
            "the refusal must say the reduction ladder was walked, got: {}",
            e.0
        );
        assert!(
            e.0.contains("K-TIME accumulation"),
            "and must name what the shape actually needs, got: {}",
            e.0
        );
    }
}
