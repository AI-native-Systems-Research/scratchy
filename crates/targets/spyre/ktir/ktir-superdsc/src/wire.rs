// SPDX-License-Identifier: Apache-2.0
//! THE WIRE LAYER — the SuperDSC type itself, and the ONLY types here that derive [`Serialize`].
//!
//! ⭐ THE SUPERDSC IR IS TWO LAYERS, AND THIS IS THE OUTER ONE. The typed core
//! ([`crate::superdsc_opspec`]) carries the witnesses and derives nothing; these structs are the
//! JSON dxp reads, and `emit_sdsc` is the sole bridge between them. That split is
//! `superdsc_opspec`'s own stated design — it is why the extraction's first move, the typed core,
//! could not bring `Dsc`/`SdscOp` with it.
//!
//! ⛔ THE EXTRACTION PLAN SAID THIS COULD ONLY MOVE WITH `emit_sdsc`. It is wrong, and the
//! correction is what lets the crate own the SuperDSC type early: every struct below is pure data
//! with `pub` fields, and the three generators take extents plus a [`StickLayout`] — nothing here
//! names a placement, a bundle or a tape. So `emit_sdsc` can go on constructing these values from
//! the other side of the boundary until it moves in its turn.
//!
//! ⛔ TRAILING UNDERSCORES ON KEYS ARE MANDATORY — DeepTools reads them verbatim.

// ⛔ THIS TRAVELLED WITH THE STRUCTS; IT IS NOT A NEW SUPPRESSION. Every field name below IS the
// JSON key DeepTools reads (`coreIdToWkSlice_`, `maxDimSizes_`, `dsType_`, …), so renaming one to
// snake_case renames it on the wire. `lower_subtile_tape_to_superdsc.rs:31` carries exactly this
// module-level attribute for exactly these structs; it moves where they moved.
#![allow(non_snake_case)]

use crate::sdsc_abstract::{DeviceWalk, StickLayout};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{ArgView, Role, Scale, SdscFoldSet, WorkPlan};
use serde::Serialize;
use std::collections::BTreeMap;

/// The `N_` iteration space for a matmul/bmm: mb=M, out=N, in=K (reduction),
/// x=batch, y=1 (decoded from the fixture's primaryDsInfo_ layoutDimOrder_).
pub fn matmul_iter_space(m: u32, n: u32, k: u32, batch: u32) -> IterSpace {
    let mut it = IterSpace::empty();
    it.mb_ = m as i64;
    it.out_ = n as i64;
    it.in_ = k as i64;
    it.y_ = 1;
    // The output-tile counts MUST be 1 (single tile), NOT the -1 "unused"
    // sentinel. `bmm.ddl` reads i_/j_/ij_ as its tile-loop bounds; -1 makes the
    // loop walk garbage → output orthogonal to A@W and invariant to addresses
    // (the exact failure signature). Match the dxp-VALIDATED MatMul_49 N_
    // (fp16_32core_nonmx): i_=1, j_=1, ij_=1, with the batch slot x_ left -1 for
    // a plain (non-batched) matmul. Only a true bmm (batch>1) populates x_.
    it.i_ = 1;
    it.j_ = 1;
    it.ij_ = 1;
    if batch > 1 {
        it.x_ = batch as i64;
    }
    it
}

// ───────────────────────────────────────────────────────────────────────────
// SuperDSC JSON wire structs — the FRONTEND-MINIMAL field set mirroring
// torch-spyre `compute_ops.py generate_sdsc`. These are the ONLY types that
// derive `Serialize`; the typed `OpSpec`/`TensorArg`/`WorkPlan` core (the
// witnesses) lowers into them via `emit_sdsc`. The over-CONSTRAINING fields
// (register-file memOrg, hand-set fold factors, hand-set exUnit) are unset
// BY CONSTRUCTION upstream — they cannot reach these structs. Trailing
// underscores on keys are MANDATORY (DeepTools reads them verbatim).
// ───────────────────────────────────────────────────────────────────────────

/// `{"factor_": N, "label_": "core"}` — a fold (core/corelet/time) descriptor.
#[derive(Serialize, Clone, Debug)]
pub struct FoldProp {
    pub factor_: i64,
    pub label_: &'static str,
}

/// Skip an `IterSpace` dim slot when it is the -1 "unused" sentinel — torch-spyre's
/// REAL frontend `N_` emits ONLY the dims an op uses (e.g. `{name_:"n", out_:64}`),
/// NOT the full 21-slot table with -1 fillers. Verified on compiled `torch.rsqrt`/
/// `add`/`mul`. The verbose -1 fillers + empty scheduler maps below were the LAST
/// difference between scratchy's rsqrt SDSC and torch-spyre's (the compiled programs
/// were 99.7% identical — only 6 constant bytes differed); dxp's SFP path reads the
/// empty `peSfpSplit_`/`coreletSplit_` scheduler maps that torch-spyre OMITS.
fn iter_slot_unused(v: &i64) -> bool {
    *v == -1
}
fn map_is_empty_i64(m: &BTreeMap<String, i64>) -> bool {
    m.is_empty()
}
fn map_is_empty_veci64(m: &BTreeMap<String, Vec<i64>>) -> bool {
    m.is_empty()
}

/// The full SuperDSC named iteration space. -1 = unused slot, OMITTED on serialize
/// (torch-spyre emits only used dims). The empty scheduler maps are also omitted —
/// torch-spyre's frontend never emits them (they are scheduler outputs).
#[derive(Serialize, Clone, Debug)]
pub struct IterSpace {
    pub name_: &'static str,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub in_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub out_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub mb_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub i_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub j_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub ki_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub kj_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub x_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub x1_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub y_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub r_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub c_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub ij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub rc_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub kij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub sij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub si_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub sj_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zi_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zj_: i64,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub symbolicDimInfo_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub maxSymbolicVolume_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_veci64")]
    pub coreletSplit_: BTreeMap<String, Vec<i64>>, // {dim: [per_corelet0, per_corelet1]}
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub rowSplit_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub peSfpSplit_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub paddingSizes_: BTreeMap<String, i64>,
}

impl IterSpace {
    /// All slots -1 (unused) except `name_`. Set the dims an op uses afterward.
    pub fn empty() -> Self {
        IterSpace {
            name_: "n",
            in_: -1,
            out_: -1,
            mb_: -1,
            i_: -1,
            j_: -1,
            ki_: -1,
            kj_: -1,
            x_: -1,
            x1_: -1,
            y_: -1,
            r_: -1,
            c_: -1,
            ij_: -1,
            rc_: -1,
            kij_: -1,
            sij_: -1,
            zij_: -1,
            si_: -1,
            sj_: -1,
            zi_: -1,
            zj_: -1,
            symbolicDimInfo_: BTreeMap::new(),
            maxSymbolicVolume_: BTreeMap::new(),
            coreletSplit_: BTreeMap::new(),
            rowSplit_: BTreeMap::new(),
            peSfpSplit_: BTreeMap::new(),
            paddingSizes_: BTreeMap::new(),
        }
    }
}

// ⛔ `pub fn ex_unit(op_func: &str)` WAS HERE AND IS DELETED, NOT DEPRECATED.
//
// It answered `"pt"` for `"matmul" | "batchmatmul"` and `_ => "sfp"`, and its own doc called itself a
// deprecated string helper "kept only so any out-of-module caller still resolves". There were NO
// callers, in this crate or anywhere in the workspace — so it resolved nothing and only waited to be
// found. Three ways it was wrong for whoever found it:
//
//   1. It CONTRADICTED the typed [`OpFunc::ex_unit`] it deferred to. `OpFunc::Transpose.ex_unit()` is
//      `"pt"`; this returned `"sfp"` for its wire name `interslicetranspose_fp16`, because that name
//      is neither of the two it matched.
//   2. The names that actually reach JSON for a quantized matmul are `matmulfp8` / `matmulint8` /
//      `batchmatmulfp8` / `batchmatmulint8` (see `emit_sdsc`'s `opFuncName`), and none of the four
//      matches the bare spellings — so all four would have taken the `_` arm and reported the SFP
//      unit for a PT-unit matmul.
//   3. The `_ => "sfp"` was a SILENT default over an open string domain: a typo answered confidently.
//
// `OpFunc::ex_unit` is the sole producer, returns a sealed `ExUnit` newtype with no public
// constructor, and is exhaustive over the enum. There is nothing a string version can do that is not
// either that call or a bug.

/// Affine-fold leaf for `dim_prop_func` entries. The fixture uses `{"Affine":
/// {alpha_,beta_}}` for linear per-index maps, `{"Const":{}}` for a fixed value,
/// and `{"Map":{}}` for an explicit `data_` lookup table (per-core addresses).
#[derive(Serialize, Clone, Debug)]
pub enum FoldFunc {
    Affine { alpha_: i64, beta_: i64 },
    Const {},
    Map {},
}

/// `memOrg_` component presence for one tensor. FRONTEND-MINIMAL: torch-spyre
/// `compute_ops.py` emits ONLY `{"isPresent": 1}` per component — the size/pad/
/// offset/allocateNode fields are scheduler outputs (over-specifying them was a
/// DtException 1535 source). Just the presence flag here.
#[derive(Serialize, Clone, Debug)]
pub struct MemPresence {
    pub isPresent: u8,
}

/// `memOrg_` = the components a tensor is resident in. SEALED to EXACTLY hbm/lx
/// (witness: NO register-file key like `pelrf` can be added — the frontend never
/// emits one; the L3 scheduler assigns l0/ptxrf/pelrf). Serializes to the minimal
/// `{"hbm":{"isPresent":1},"lx":{"isPresent":1}}` (or hbm-only / lx-only).
#[derive(Clone, Debug)]
pub struct MemOrg {
    hbm: Option<MemPresence>,
    lx: Option<MemPresence>,
}
impl MemOrg {
    /// Default residency: present in BOTH hbm and lx (the matmul/pointwise case).
    pub fn hbm_lx() -> Self {
        MemOrg {
            hbm: Some(MemPresence { isPresent: 1 }),
            lx: Some(MemPresence { isPresent: 1 }),
        }
    }
    /// HBM-only (index tensors must reside in HBM — no LX indirect addressing).
    pub fn hbm_only() -> Self {
        MemOrg {
            hbm: Some(MemPresence { isPresent: 1 }),
            lx: None,
        }
    }
    /// LX-only (an op-local scratchpad dataspace).
    pub fn lx_only() -> Self {
        MemOrg {
            hbm: None,
            lx: Some(MemPresence { isPresent: 1 }),
        }
    }
}
// Custom Serialize so absent components are simply omitted (never a `null`),
// matching torch-spyre's conditional dict; and so NO register-file key exists.
impl Serialize for MemOrg {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let n = self.hbm.is_some() as usize + self.lx.is_some() as usize;
        let mut m = ser.serialize_map(Some(n))?;
        if let Some(p) = &self.hbm {
            m.serialize_entry("hbm", p)?;
        }
        if let Some(p) = &self.lx {
            m.serialize_entry("lx", p)?;
        }
        m.end()
    }
}

/// One tensor descriptor in `labeledDs_`. FRONTEND-MINIMAL: torch-spyre emits
/// only `{ldsIdx_, dsName_, dsType_, scale_, wordLength, dataFormat_, memOrg_}`.
/// The dropped fields (density_/segment_/isStatic_/level/hbm*Size_/lx*Size_/…)
/// are scheduler-filled — emitting them mis-sized the LX chunk (1535). `scale_`:
/// 1 = active, -1 = reduction (non-stick), -2 = stick-aligned reduction (sealed
/// to the `Scale` enum upstream — never a free integer).
#[derive(Serialize, Clone, Debug)]
pub struct LabeledDs {
    pub ldsIdx_: u32,
    pub dsName_: String,
    pub dsType_: &'static str, // "INPUT" | "KERNEL" | "OUTPUT"
    pub scale_: Vec<i64>,
    pub wordLength: u32,           // 2 for fp16
    pub dataFormat_: &'static str, // "SEN169_FP16"
    pub memOrg_: MemOrg,
}

/// `computeOp_` entry. `exUnit` is a sealed `ExUnit` value from `OpFunc::ex_unit`
/// only (witness #6) — never hand-set on a matmul.
#[derive(Serialize, Clone, Debug)]
pub struct ComputeOp {
    pub exUnit: &'static str, // "pt" | "sfp"
    pub opFuncName: String,
    pub attributes_: ComputeAttrs,
    pub location: &'static str, // "Inner"
    // torch-spyre's REAL frontend emits ONLY the 6 core fields above + the two
    // labeledDs lists — VERIFIED on compiled `torch.rsqrt`/`F.rms_norm`/`add`/`mul`
    // (every op = `exUnit/opFuncName/attributes_/location/inputLabeledDs/outputLabeledDs`,
    // NOTHING else). A prior me ADDED the loop-placement flags below citing the dxp
    // `sdsc_silu.json` *fixture*, but the real frontend OMITS them and computes rsqrt
    // CORRECTLY on-card — and an explicit `auxLoopName: ""` plausibly tells dxp "no aux
    // loop" → suppresses the SFP Newton-Raphson refine (rsqrt→1/x). So every field below
    // is Option/skip-if-empty: the default emission is byte-equal to torch-spyre (omitted),
    // and a populated interim/indirect list (paged ops) still serializes when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxLoopName: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isAtMainLoop: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isAtTop: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coreExclude: Vec<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coreClExclude: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opConsts: Option<serde_json::Value>,
    pub inputLabeledDs: Vec<String>, // ["Tensor0-idx0", …]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub interimLabeledDs: Vec<String>,
    pub outputLabeledDs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub indirectAccessIndexLabeledDs: Vec<String>,
}
#[derive(Serialize, Clone, Debug)]
pub struct ComputeAttrs {
    pub dataFormat_: &'static str,
    pub fidelity_: &'static str, // "regular"
}

/// `primaryDsInfo_` entry (per role INPUT/KERNEL/OUTPUT): the tile layout.
/// Matches torch-spyre: `layoutDimOrder_`, `stickDimOrder_`, `stickSize_` only
/// (no `stickRepl_` — torch-spyre does not emit it).
#[derive(Serialize, Clone, Debug)]
pub struct LayoutInfo {
    pub layoutDimOrder_: Vec<&'static str>,
    pub stickDimOrder_: Vec<&'static str>,
    pub stickSize_: Vec<u32>, // [64] for fp16
}

/// `scheduleTree_` allocate node: a per-tensor HBM (or LX) buffer. FRONTEND-
/// MINIMAL, matching torch-spyre's `generate_sdsc` allocate node: `nodeType_`,
/// `name_`, `prev_`, `ldsIdx_`, `component_`, `layoutDimOrder_`, `maxDimSizes_`,
/// the indirect-access fields, `startAddressCoreCorelet_`, `coordinates_`, and
/// `backGapCore_` ONLY when a back-gap is present. The mechanical defaults the
/// bmm SCHEDULER fixture carried (numBuffers_/constIdx_/nonUnifiedAllocInHBM_/…)
/// are dropped — they are scheduler outputs. `isStartAddrSymbolic_` is emitted
/// ONLY for symbolic HBM addresses; `relatedIndirectAccessAlloc_`/
/// `indexTensorType_` ONLY for indirect access (`skip_serializing_if` on the
/// `Option`s keeps byte-faithfulness for both presence and absence).
#[derive(Serialize, Clone, Debug)]
pub struct AllocNode {
    pub nodeType_: &'static str, // "allocate"
    pub name_: String,
    pub prev_: String,
    pub ldsIdx_: u32,
    pub component_: &'static str, // "hbm" | "lx"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isStartAddrSymbolic_: Option<u8>,
    pub layoutDimOrder_: Vec<&'static str>,
    // Opaque: the ONLY way to build one is `StickLayout::device_walk` (from the same layout as the start),
    // so a walk that disagrees with the start — or has the wrong rank — is unconstructable. Serializes to
    // the `[-1; rank]` / pinned-extents array dxp reads.
    pub maxDimSizes_: DeviceWalk,
    pub indirectAllocType_: &'static str, // "no_indirection"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relatedIndirectAccessAlloc_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexTensorType_: Option<&'static str>,
    pub startAddressCoreCorelet_: AddrFold,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backGapCore_: Option<serde_json::Value>,
    pub coordinates_: serde_json::Value, // {coordInfo:{…}, coreIdToWkSlice_:{}}
}
/// The `startAddressCoreCorelet_` fold: `dim_prop_func` = `[Map, Const, Const]`
/// over `[core, corelet, time]`, with the per-core HBM byte address in `data_`
/// keyed `"[c, 0, 0]"` (SPACES, per torch-spyre `_start_addr_data`). The
/// `[core, corelet]` factors are COPIED from the op's [`SdscFoldSet`] via
/// [`AddrFold::new`] (witness (c)): there is no public field setter, so a factor
/// that diverges from `coreFoldProp_`/`coreletFoldProp_` is unrepresentable.
#[derive(Serialize, Clone, Debug)]
pub struct AddrFold {
    pub dim_prop_func: Vec<FoldFunc>,
    pub dim_prop_attr: Vec<FoldProp>,
    /// "[c, 0, 0]" → HBM byte address (a string, per the torch-spyre frontend).
    pub data_: BTreeMap<String, String>,
}
impl AddrFold {
    /// Build the address fold, copying the core/corelet/time factors STRAIGHT from
    /// `folds` (witness (c)). `per_core_addr[c]` is the byte address for core `c`.
    /// CONCRETE-address path (`isStartAddrSymbolic_` absent) — the time=1 case.
    pub fn new(per_core_addr: &[u64], folds: &SdscFoldSet) -> AddrFold {
        AddrFold {
            // torch-spyre emits [Map, Const, Const] over [core, corelet, time].
            dim_prop_func: vec![FoldFunc::Map {}, FoldFunc::Const {}, FoldFunc::Const {}],
            dim_prop_attr: vec![
                FoldProp {
                    factor_: folds.core_fold() as i64,
                    label_: "core",
                },
                FoldProp {
                    factor_: folds.corelet_fold() as i64,
                    label_: "corelet",
                },
                FoldProp {
                    factor_: folds.time_fold() as i64,
                    label_: "time",
                },
            ],
            data_: per_core_addr
                .iter()
                .enumerate()
                .map(|(c, a)| (format!("[{c}, 0, 0]"), a.to_string()))
                .collect(),
        }
    }

    /// Build the SYMBOLIC-address fold for a TIME-TILED HBM tensor: the per-core
    /// `data_` value is a NEGATIVE symbol-id string (e.g. `"-1"`) instead of a
    /// concrete byte address, because the per-iteration HBM advance is computed by
    /// `affine.apply` in `bundle.mlir` (torch-spyre `compute_ops.py:491`,
    /// `local_symbols[addr]` under `use_symbols=True`). The fold factors are still
    /// copied from `folds` (witness (c)); only the `data_` values differ. Pairs
    /// with `isStartAddrSymbolic_: Some(1)` on the AllocNode (the **SymbolicWhenTiled**
    /// witness). `per_core_sym_ids[c]` is the negative id for core `c`.
    pub fn symbolic(per_core_sym_ids: &[i64], folds: &SdscFoldSet) -> AddrFold {
        AddrFold {
            dim_prop_func: vec![FoldFunc::Map {}, FoldFunc::Const {}, FoldFunc::Const {}],
            dim_prop_attr: vec![
                FoldProp {
                    factor_: folds.core_fold() as i64,
                    label_: "core",
                },
                FoldProp {
                    factor_: folds.corelet_fold() as i64,
                    label_: "corelet",
                },
                FoldProp {
                    factor_: folds.time_fold() as i64,
                    label_: "time",
                },
            ],
            data_: per_core_sym_ids
                .iter()
                .enumerate()
                .map(|(c, id)| (format!("[{c}, 0, 0]"), id.to_string()))
                .collect(),
        }
    }
}

/// `dataStageParam_["0"]` = per-core steady-state (`ss_`) + epilogue (`el_`)
/// dim sizes = N_ dims ÷ the work-division split (out 384/2=192, mb 384/16=24).
#[derive(Serialize, Clone, Debug)]
pub struct StageParam {
    pub ss_: IterSpace,
    pub el_: IterSpace,
}

/// `sdscFolds_` — the time-fold descriptor (Affine identity + per-step map).
#[derive(Serialize, Clone, Debug)]
pub struct SdscFolds {
    pub dim_prop_func: Vec<FoldFunc>,
    pub dim_prop_attr: Vec<FoldProp>,
    pub data_: BTreeMap<String, String>,
}

/// One `dscs_` entry — the per-DesignSpaceConfig tile schedule. FRONTEND-MINIMAL,
/// matching torch-spyre's `generate_sdsc` inner dict: `numCoresUsed_`,
/// `numCoreletsUsed_:1`, `coreIdsUsed_`, `N_`, `coordinateMasking_`,
/// `maskingConstId_`, `dataStageParam_`, `primaryDsInfo_`, `scheduleTree_`,
/// `labeledDs_`, `constantInfo_`, `computeOp_`. The bmm SCHEDULER fixture's
/// extras (unpadN_/T_/Tel_/P_/Pel_/B_/ChipD_/CoreD_/CoreletD_/pdsRelation_/pcfg_/
/// dscN_/numCoreletsUsed_DSC2_/ChipletD_/auxLoopOrder_/dimToSymbolMapping_/
/// gtrIdsUsed_/l0TetheredMode_/scheduleTreeHeadDenId_/loopOrder_/loopProperties_)
/// are all scheduler outputs — DROPPED. `coreIdToDscSchedule` stays (op-level).
#[derive(Serialize, Clone, Debug)]
pub struct Dsc {
    pub numCoresUsed_: u32,
    pub numCoreletsUsed_: u32, // 1 (ACTIVE_CORELETS) — frontend constant
    pub coreIdsUsed_: Vec<u32>,
    pub N_: IterSpace,
    pub coordinateMasking_: BTreeMap<String, Vec<[i64; 2]>>,
    pub maskingConstId_: i64,
    pub dataStageParam_: BTreeMap<String, StageParam>,
    pub primaryDsInfo_: BTreeMap<&'static str, LayoutInfo>,
    pub scheduleTree_: Vec<AllocNode>,
    pub labeledDs_: Vec<LabeledDs>,
    // EMPTY → the JSON *string* "{}"; POPULATED → a dict. VERIFIED on torch-spyre's
    // REAL frontend (compiled `torch.rsqrt`/`F.rms_norm`): rsqrt/add/mul emit
    // `"constantInfo_": "{}"` (a str), the `mean` op's scaling_factor a dict. The earlier
    // claim that the string was an "old bug" (citing the bmm SCHEDULER fixture, NOT the
    // frontend) was WRONG and INVERTED: the empty OBJECT {} is falsy in dxp's Python so
    // the SFP constant-table / NR-refine setup is skipped → transcendentals collapse to
    // their seed (rsqrt→1/x). The empty→string conversion lives at the `constant_info`
    // build site. serde_json::Value carries either form.
    pub constantInfo_: serde_json::Value,
    pub computeOp_: Vec<ComputeOp>,
}

/// Top-level SuperDSC op (one `sdsc_{idx}.json`, serialized `{ "{op}_{id}": .. }`).
/// FRONTEND-MINIMAL, matching torch-spyre's `generate_sdsc` outer dict:
/// `sdscFoldProps_` (time=1), `sdscFolds_`, `coreFoldProp_`, `coreletFoldProp_`
/// (factor 1), `numCoresUsed_`, `coreIdToDsc_`, `numWkSlicesPerDim_`,
/// `coreIdToWkSlice_`, `coreIdToDscSchedule`, `dscs_`. The bmm SCHEDULER fixture's
/// extras (folded_sdsc_name_/fold_coord_/unpadN_/N_ at the op level, datadscs_/
/// opFuncsUsed_/target_/symbolDefinitions_/ldsShareInfo_/prodConsList/
/// inputSymbolsAndTags_/dimToSymbolMappingOpcodeCorrection_) are scheduler
/// outputs — DROPPED.
#[derive(Serialize, Clone, Debug)]
pub struct SdscOp {
    pub sdscFoldProps_: Vec<FoldProp>,
    pub sdscFolds_: SdscFolds,
    pub coreFoldProp_: FoldProp,
    pub coreletFoldProp_: FoldProp,
    pub numCoresUsed_: u32,
    pub coreIdToDsc_: BTreeMap<String, u32>,
    pub numWkSlicesPerDim_: BTreeMap<&'static str, u32>,
    /// Per-core slice indices, typed: a wire value here is provably a `SliceIndex` minted
    /// by `core_to_wk_slice`'s decomposition (serializes as the bare integer, wire-identical).
    pub coreIdToWkSlice_:
        BTreeMap<String, BTreeMap<&'static str, crate::superdsc_opspec::SliceIndex>>,
    pub coreIdToDscSchedule: BTreeMap<String, Vec<[i64; 4]>>, // {core:[[-1,0,0,0]]}
    pub dscs_: Vec<BTreeMap<String, Dsc>>,
}

/// `coreIdToDscSchedule`: one schedule entry `[[-1,0,0,0]]` per core (the fixture
/// form — a single dsc-0 step per core, no chunk/time sub-schedule).
pub fn core_dsc_schedule(cores: u32) -> BTreeMap<String, Vec<[i64; 4]>> {
    (0..cores.max(1))
        .map(|c| (c.to_string(), vec![[-1, 0, 0, 0]]))
        .collect()
}

/// `gen_coord_info_value` — PORTED VERBATIM from torch-spyre `compute_ops.py`.
/// The per-dim tile-fold for one allocate-node dim. `size` is the per-core extent
/// (full // split) for an active dim, else 1; `nsplits` is the split for active
/// dims, else 1; `is_stick_reduction` selects the reduction-stick variant.
pub fn gen_coord_info_value(
    size: i64,
    nsplits: i64,
    elems_per_stick: i64,
    is_stick_dim: bool,
    is_stick_reduction: bool,
    // Byte/elem stride between consecutive STICK-GROUPS along this dim. NATURAL layout =
    // `elems_per_stick` (sticks are contiguous). A RowBlocked tensor ([cols/eps, rows, eps])
    // interleaves all `rows` inside each stick-group, so its stick-groups are `rows·eps`
    // apart — a reduce over such a dim MUST step by that, else it sums across the wrong rows
    // (the mq>1 fp32 rmsnorm `mean(x²)` NaN: it read `sq32`'s stick-groups at stride 32 while
    // they physically sit 992 apart, gathering 64 different rows' first sticks + OOB).
    group_stride: i64,
) -> serde_json::Value {
    if !is_stick_dim {
        serde_json::json!({
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": size, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 1, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": size, "label_": "elem_arr_0"},
                ],
            },
        })
    } else {
        serde_json::json!({
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": if is_stick_reduction { elems_per_stick } else { size }, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": group_stride, "beta_": 0}},
                    {"Affine": {"alpha_": if is_stick_reduction { 0 } else { 1 }, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": if is_stick_reduction { 1 } else { size / elems_per_stick }, "label_": "elem_arr_1"},
                    {"factor_": elems_per_stick, "label_": "elem_arr_0"},
                ],
            },
        })
    }
}

/// The `coordinates_` object: `coordInfo` per layout dim (via the ported
/// `gen_coord_info_value`) + an empty `coreIdToWkSlice_`. `arg` is the view of the
/// tensor whose layout this allocate node owns; `plan` provides the extents +
/// splits. Mirrors compute_ops.py's per-dim loop: for an ACTIVE dim (scale == 1)
/// the fold uses the per-core extent (size // split) and the split; for a
/// reduction dim, size=1/nsplits=1 with the reduction-stick variant when -2.
pub fn build_coordinates(
    arg: &ArgView<'_>,
    // THE SAME layout `per_core_addr` gets (view_stick_layout, computed once at the call site) — now
    // DRIVES group_stride (StickLayout::group_stride) instead of a parallel hand-derivation from
    // `arg.scale`/`arg.row_blocked`. One proven decision, not two.
    layout: &StickLayout,
    plan: &WorkPlan,
    fp8_matmul: bool,
    is_reduction: bool,
) -> serde_json::Value {
    // Stick width from the operand's device format (fp8/int8 = 128, fp16/bf16 = 64, fp32 = 32) so the
    // coordInfo fold matches the emitted `stickSize_`; else dxp folds the fp8 dataspace at 64 (wrong).
    let eps = arg.df.elems_per_stick() as i64;
    let mut ci = serde_json::Map::new();
    for (i, &d) in arg.layout.iter().enumerate() {
        let scale = arg.scale[i];
        let split = plan.split_of(d) as i64;
        let full = plan.extent(d) as i64;
        let (size, nsplits) = if scale == Scale::Active {
            ((full / split.max(1)).max(1), split.max(1))
        } else {
            (1, 1)
        };
        let is_stick = arg.stick == d;
        // The fp8 W8A8 matmul operands use the AIU fp8 PE's NATIVE nested tile geometry (verified against
        // IBM's l0_tethering fixture, extracted via L3DlOpsScheduler_standalone): the WEIGHT's N-stick is
        // sub-tiled 8×8 and its K is 2-packed; the ACTIVATION's K-stick (128) is sub-tiled 8×2×8. A flat
        // single-stick fold (the fp16 shape) makes dsc2's fold-contiguity walk demand a "Loop split"
        // (dsc2.cpp:6379). These generators emit the exact nested folds so the walk stays contiguous.
        // Stick-group step: a RowBlocked tensor `[feat/eps, rows, eps]` interleaves all `rows` INSIDE each
        // stick-group, so consecutive stick-groups sit `rows·eps` apart, NOT `eps`. This is true of EVERY
        // mq>1 stick-major activation — the fp16 rank-2 stick-major (RowBlocked via `for_view_df`) and the
        // fp32 `row_blocked` island — for BOTH the producer (pointwise/rmsnorm write) and the consumer
        // (matmul-A / reduce read), not just the `is_reduction` reduce. Emitting `eps` (contiguous) made a
        // core read each row's 2nd..Nth stick from the NEXT row (row-mixing): the mq>1 matmul/reduce then
        // summed a single stick across 32 consecutive rows (+ zero padding) instead of one row's 32 sticks →
        // Σx²≈0 → amax≈0 → every projection inf. `rows==1` (decode) keeps `eps` byte-identical (rb_rows=1).
        // KERNELS are re-tiled by their RetileDescriptor (not this coordInfo), so they stay `eps`. The
        // `is_stick_reduction` accum collapses to `alpha_=0` regardless, so it also stays `eps`.
        // The stick-GROUP step: ONE call, `StickLayout::group_stride`, replacing the old two-fact
        // hand-derivation (`rb_rows` product + the `arg.row_blocked` gate). Proven fp16-safe /
        // fp32-eligible on the SAME layout `per_core_addr` uses for the start — see its doc.
        let group_stride = if is_stick && !scale.is_stick_reduction() {
            layout.group_stride(is_reduction)
        } else {
            eps
        };
        let info = match (fp8_matmul, arg.role, d) {
            (true, Role::Kernel, "out") => gen_fp8_kernel_out_fold(size, nsplits),
            (true, Role::Kernel, "in") => gen_fp8_kernel_in_fold(size, nsplits),
            (true, Role::Input, "in") => gen_fp8_input_in_fold(size, nsplits),
            _ => gen_coord_info_value(
                size,
                nsplits,
                eps,
                is_stick,
                scale.is_stick_reduction(),
                group_stride,
            ),
        };
        ci.insert(d.to_string(), info);
    }
    serde_json::json!({ "coordInfo": serde_json::Value::Object(ci), "coreIdToWkSlice_": {} })
}

/// fp8 WEIGHT `out` (N) fold: the 64-elem N-stick is sub-tiled 8×8 (elem_arr_1×elem_arr_0), with
/// `per_core_N/64` such sticks (elem_arr_2), split `nsplits` ways across cores. Alphas chain
/// contiguously (1, 8, 64, per_core) so dsc2's `alpha[curr]==alpha[next]·card[next]` holds.
fn gen_fp8_kernel_out_fold(per_core: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 3, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": per_core, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 64, "beta_": 0}},
                {"Affine": {"alpha_": 8, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (per_core / 64).max(1), "label_": "elem_arr_2"},
                {"factor_": 8, "label_": "elem_arr_1"},
                {"factor_": 8, "label_": "elem_arr_0"},
            ],
        },
    })
}

/// fp8 WEIGHT `in` (K) fold: K is 2-PACKED (2 fp8 per fp16-width slot) — elem_arr_0=2 (α1), with
/// `K/2` such packs (elem_arr_1, α2). Contiguous (2=1·2).
///
/// `k` is the PER-CORE extent (`full/nsplits`, as build_coordinates passes it) and `nsplits` is the
/// `in` split, so the reconstructed extent is `core_fold · Π elem_arr = nsplits · (k/2) · 2 =
/// nsplits·k` = the full K. `core_fold` was previously HARDCODED to 1, written under the old
/// invariant that the reduction dim is never split; once the cost model started K-splitting, the
/// descriptor claimed 1/nsplits of K while `numWkSlicesPerDim_["in"]` and `per_core_addr` both
/// encoded a real nsplits-way split — the descriptor and the addressing disagreed. Its fp16 sibling
/// `gen_coord_info_value` has always carried the real `nsplits` here.
fn gen_fp8_kernel_in_fold(k: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 2, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": k, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 2, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (k / 2).max(1), "label_": "elem_arr_1"},
                {"factor_": 2, "label_": "elem_arr_0"},
            ],
        },
    })
}

/// fp8 ACTIVATION `in` (K) fold: the 128-elem fp8 K-stick is sub-tiled 8×2×8 (elem_arr_0=8 α1,
/// elem_arr_1=2 α64, elem_arr_2=8 α8), with `K/128` sticks (elem_arr_3 α128).
/// The non-monotonic alphas (1,64,8,128) are the fp8 activation's physical interleave (from IBM's fixture).
///
/// `k` is the PER-CORE extent and `nsplits` the `in` split, so the reconstructed extent is
/// `core_fold · Π elem_arr = nsplits · (k/128) · 8 · 2 · 8 = nsplits·k` = the full K. See
/// [`gen_fp8_kernel_in_fold`] for why `core_fold` was wrongly hardcoded to 1.
fn gen_fp8_input_in_fold(k: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 4, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": k, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 128, "beta_": 0}},
                {"Affine": {"alpha_": 8, "beta_": 0}},
                {"Affine": {"alpha_": 64, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (k / 128).max(1), "label_": "elem_arr_3"},
                {"factor_": 8, "label_": "elem_arr_2"},
                {"factor_": 2, "label_": "elem_arr_1"},
                {"factor_": 8, "label_": "elem_arr_0"},
            ],
        },
    })
}

// (The former `lx_chunk_bytes` / `assert_chunk_stick_multiples` emit-time guard is
// GONE: the DtException-1535 chunk-size rule is now enforced BY CONSTRUCTION in
// `TensorArg::new` (the MaterializedStick invariant — a stick dim can't be phantom-
// scaled), so a non-128-multiple LX chunk is unrepresentable. No need to recompute
// `getBufferCapacityForNode` on the generated SDSC and assert on it.)

/// Per-arg HBM SEGMENT base addresses (torch-spyre `constants.py:44`
/// `SEGMENT_OFFSETS`, 16 GiB = `0x4_0000_0000` stride). Each I/O arg lives in its
/// OWN 16 GiB segment keyed by its `arg_index` (== `ldsIdx_` == the tensor's
/// position in `op.args`), so two cores' tiles of DIFFERENT tensors can never
/// alias, and a tiled op's per-trip tiles of the SAME tensor only ever advance
/// WITHIN one segment (task #50 — this is what lets GUARD #14 lift). There are 7
/// segments; an op with >7 distinct dataspaces is a build-time `Err` (we cannot
/// give every tensor a non-aliasing segment), surfaced by [`segment_base`].
pub const SEGMENT_OFFSETS: [u64; 7] = [
    0x0,
    0x4_0000_0000,
    0x8_0000_0000,
    0xC_0000_0000,
    0x10_0000_0000,
    0x14_0000_0000,
    0x18_0000_0000,
];
/// 16 GiB segment stride (torch-spyre `constants.py:55` `SEGMENT_SIZE`).
pub const SEGMENT_SIZE: u64 = 0x4_0000_0000;

/// Decompose an HBM byte address into its `(segment_index, intra_segment_offset)`.
/// Every HBM address our emitter produces is `SEGMENT_OFFSETS[seg] + intra` with
/// `intra < SEGMENT_SIZE` (each dataspace lives in its own 16 GiB segment), so the
/// decomposition is exact: `seg = addr / SEGMENT_SIZE`, `off = addr % SEGMENT_SIZE`,
/// and `SEGMENT_OFFSETS[seg] + off == addr`. This is what lets the SYMBOLIC bundle
/// emit each address as `arith.addi %segbase_{seg}, off` — the torch-spyre
/// `kernel_derived` scheme (`generate_bundle`), whose SHARED per-value operand SSA
/// (vs a fresh `arith.constant` per per-core symbol) is what dxp's symbol machinery
/// accepts. Kani-verified (`hbm_seg_off_reconstructs_address`).
pub fn hbm_seg_off(addr: u64) -> (u64, u64) {
    (addr / SEGMENT_SIZE, addr % SEGMENT_SIZE)
}

/// The HBM segment base for arg `arg_index`. A build-time `Err` (not a panic /
/// silent wrap) when `arg_index >= 7`: there are only 7 non-aliasing segments, so
/// an 8th distinct dataspace would have to share a segment and could alias — that
/// must be a `cargo build` failure, never a silently-wrong on-card address.
pub fn segment_base(arg_index: usize) -> Result<u64, SuperDscError> {
    SEGMENT_OFFSETS.get(arg_index).copied().ok_or_else(|| {
        SuperDscError(format!(
            "per-core HBM addressing: arg_index {arg_index} exceeds the {} available 16 GiB \
             HBM segments (SEGMENT_OFFSETS, torch-spyre constants.py:44). An op with >{} \
             distinct dataspaces cannot give each its own non-aliasing segment — refusing to \
             bake (would alias on-card). Fuse/spill some operands or split the op.",
            SEGMENT_OFFSETS.len(),
            SEGMENT_OFFSETS.len()
        ))
    })
}
