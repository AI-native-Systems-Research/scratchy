// SPDX-License-Identifier: Apache-2.0
//! The SuperDSC JSON dxp reads — one `sdsc_<i>.json` per device op. Field names are dxp's (hence
//! their spelling); every value is built by [`crate::emit`] from typed facts.

use std::collections::BTreeMap;

use serde::Serialize;

/// `{"factor_": n, "label_": "core"}`. dxp's schema requires every fold factor `>= 1`, so the
/// factor is a nonzero count by type.
#[derive(Clone, Debug, Serialize)]
pub struct Prop {
    #[serde(rename = "factor_")]
    pub factor: std::num::NonZeroU64,
    #[serde(rename = "label_")]
    pub label: &'static str,
}

impl Prop {
    /// A fold of `factor` along `label`.
    ///
    /// # Panics
    /// When `factor` is 0: a fold spans at least one of what it folds, and a zero names a lowering
    /// that miscounted it (dxp's schema refuses the descriptor: `factor_ … Must be >= 1`).
    pub fn new(factor: u64, label: &'static str) -> Prop {
        Prop {
            factor: std::num::NonZeroU64::new(factor)
                .unwrap_or_else(|| panic!("a `{label}` fold of zero — a fold spans at least one")),
            label,
        }
    }
}

/// One fold function.
#[derive(Clone, Debug, Serialize)]
pub enum Func {
    Affine {
        #[serde(rename = "alpha_")]
        alpha: i64,
        #[serde(rename = "beta_")]
        beta: i64,
    },
    Map {},
    Const {},
}

/// An affine per-dimension fold: the functions and their factors, and for a `Map` fold the data.
#[derive(Clone, Debug, Serialize)]
pub struct Folds<D: Serialize> {
    pub dim_prop_func: Vec<Func>,
    pub dim_prop_attr: Vec<Prop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "data_")]
    pub data: Option<D>,
}

/// One dimension's coordinate description in a tensor's allocation.
#[derive(Clone, Debug, Serialize)]
pub struct Coord {
    pub spatial: i64,
    pub temporal: i64,
    #[serde(rename = "elemArr")]
    pub elem_arr: i64,
    pub padding: &'static str,
    pub folds: Folds<()>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Coordinates {
    #[serde(rename = "coordInfo")]
    pub coord_info: BTreeMap<&'static str, Coord>,
    #[serde(rename = "coreIdToWkSlice_")]
    pub core_id_to_wk_slice: BTreeMap<String, BTreeMap<&'static str, i64>>,
}

/// One tensor's `allocate` node in the schedule tree.
#[derive(Clone, Debug, Serialize)]
pub struct Allocate {
    #[serde(rename = "nodeType_")]
    pub node_type: &'static str,
    #[serde(rename = "name_")]
    pub name: String,
    #[serde(rename = "prev_")]
    pub prev: &'static str,
    #[serde(rename = "ldsIdx_")]
    pub lds_idx: usize,
    #[serde(rename = "component_")]
    pub component: &'static str,
    #[serde(rename = "layoutDimOrder_")]
    pub layout_dim_order: Vec<&'static str>,
    #[serde(rename = "maxDimSizes_")]
    pub max_dim_sizes: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "indirectAllocType_")]
    pub indirect_alloc_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "relatedIndirectAccessAlloc_")]
    pub related_indirect_access_alloc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "indexTensorType_")]
    pub index_tensor_type: Option<&'static str>,
    #[serde(rename = "startAddressCoreCorelet_")]
    pub start_address_core_corelet: Folds<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "backGapCore_")]
    pub back_gap_core: Option<BTreeMap<&'static str, BTreeMap<&'static str, String>>>,
    #[serde(rename = "coordinates_")]
    pub coordinates: Coordinates,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemPresence {
    #[serde(rename = "isPresent")]
    pub is_present: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemOrg {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hbm: Option<MemPresence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lx: Option<MemPresence>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LabeledDs {
    #[serde(rename = "ldsIdx_")]
    pub lds_idx: usize,
    #[serde(rename = "dsName_")]
    pub ds_name: String,
    #[serde(rename = "dsType_")]
    pub ds_type: &'static str,
    #[serde(rename = "scale_")]
    pub scale: Vec<i64>,
    #[serde(rename = "wordLength")]
    pub word_length: u64,
    #[serde(rename = "dataFormat_")]
    pub data_format: &'static str,
    #[serde(rename = "memOrg_")]
    pub mem_org: MemOrg,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayoutInfo {
    #[serde(rename = "layoutDimOrder_")]
    pub layout_dim_order: Vec<&'static str>,
    #[serde(rename = "stickDimOrder_")]
    pub stick_dim_order: Vec<&'static str>,
    #[serde(rename = "stickSize_")]
    pub stick_size: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComputeAttrs {
    #[serde(rename = "dataFormat_")]
    pub data_format: &'static str,
    #[serde(rename = "fidelity_")]
    pub fidelity: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComputeOp {
    #[serde(rename = "exUnit")]
    pub ex_unit: &'static str,
    #[serde(rename = "opFuncName")]
    pub op_func_name: &'static str,
    #[serde(rename = "attributes_")]
    pub attributes: ComputeAttrs,
    pub location: &'static str,
    #[serde(rename = "inputLabeledDs")]
    pub input_labeled_ds: Vec<String>,
    #[serde(rename = "outputLabeledDs")]
    pub output_labeled_ds: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(rename = "indirectAccessIndexLabeledDs")]
    pub indirect_access_index_labeled_ds: Vec<String>,
}

/// One constant an op carries (`constantInfo_` entry).
#[derive(Clone, Debug, Serialize)]
pub struct Constant {
    #[serde(rename = "dataFormat_")]
    pub data_format: &'static str,
    #[serde(rename = "name_")]
    pub name: &'static str,
    #[serde(rename = "data_")]
    pub data: Folds<BTreeMap<String, Vec<String>>>,
}

/// An iteration space (`N_`, `ss_`, `el_`): the name and one entry per dimension.
#[derive(Clone, Debug, Serialize)]
pub struct Space {
    #[serde(rename = "name_")]
    pub name: &'static str,
    #[serde(flatten)]
    pub dims: BTreeMap<&'static str, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    #[serde(rename = "ss_")]
    pub ss: Space,
    #[serde(rename = "el_")]
    pub el: Space,
}

/// `constantInfo_`: dxp reads the empty case as the STRING `"{}"`.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Constants {
    Empty(&'static str),
    Some(BTreeMap<String, Constant>),
}

#[derive(Clone, Debug, Serialize)]
pub struct Dsc {
    #[serde(rename = "numCoresUsed_")]
    pub num_cores_used: u32,
    #[serde(rename = "numCoreletsUsed_")]
    pub num_corelets_used: u32,
    #[serde(rename = "coreIdsUsed_")]
    pub core_ids_used: Vec<u32>,
    #[serde(rename = "N_")]
    pub n: Space,
    #[serde(rename = "coordinateMasking_")]
    pub coordinate_masking: BTreeMap<&'static str, Vec<[i64; 2]>>,
    #[serde(rename = "maskingConstId_")]
    pub masking_const_id: i64,
    #[serde(rename = "dataStageParam_")]
    pub data_stage_param: BTreeMap<&'static str, Stage>,
    #[serde(rename = "primaryDsInfo_")]
    pub primary_ds_info: BTreeMap<&'static str, LayoutInfo>,
    #[serde(rename = "scheduleTree_")]
    pub schedule_tree: Vec<Allocate>,
    #[serde(rename = "labeledDs_")]
    pub labeled_ds: Vec<LabeledDs>,
    #[serde(rename = "constantInfo_")]
    pub constant_info: Constants,
    #[serde(rename = "computeOp_")]
    pub compute_op: Vec<ComputeOp>,
}

/// One `sdsc_<i>.json`'s top-level value.
#[derive(Clone, Debug, Serialize)]
pub struct Sdsc {
    #[serde(rename = "sdscFoldProps_")]
    pub sdsc_fold_props: Vec<Prop>,
    #[serde(rename = "sdscFolds_")]
    pub sdsc_folds: Folds<BTreeMap<&'static str, &'static str>>,
    #[serde(rename = "coreFoldProp_")]
    pub core_fold_prop: Prop,
    #[serde(rename = "coreletFoldProp_")]
    pub corelet_fold_prop: Prop,
    #[serde(rename = "numCoresUsed_")]
    pub num_cores_used: u32,
    #[serde(rename = "coreIdToDsc_")]
    pub core_id_to_dsc: BTreeMap<String, u32>,
    #[serde(rename = "numWkSlicesPerDim_")]
    pub num_wk_slices_per_dim: BTreeMap<&'static str, u64>,
    #[serde(rename = "coreIdToWkSlice_")]
    pub core_id_to_wk_slice: BTreeMap<String, BTreeMap<&'static str, u64>>,
    #[serde(rename = "coreIdToDscSchedule")]
    pub core_id_to_dsc_schedule: BTreeMap<String, Vec<[i64; 4]>>,
    #[serde(rename = "dscs_")]
    pub dscs: Vec<BTreeMap<&'static str, Dsc>>,
}
