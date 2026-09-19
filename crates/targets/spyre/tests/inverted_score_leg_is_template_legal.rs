// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! ⭐⭐⭐⭐⭐ THE INVERTED SCORE LEG EMITS THE SHAPE `bmm.ddl` DEMANDS — natural K as the ACTIVATION, Qᵀ as
//! the kernel, so the resident Kᵀ plane has no reader.
//!
//! `bmm.ddl` fixes two things about an fp16 bmm and they point in opposite directions:
//! ```text
//!   :18  %slice_layout_input_16bit  = ddl.layout(%in)  {is_order_fixed=true}   // ACTIVATION sticks on `in`
//!   :23  %slice_layout_kernel_16bit = ddl.layout(%out) {is_order_fixed=true}   // KERNEL sticks on `out`
//! ```
//! Natural K is `[slots, hd]` sticked on `hd`. As a KERNEL that is `{out} ∩ {in} = {}` and dxp refuses it
//! (`DtException: Could not find any suitable dimension mapping`, measured on card). As an ACTIVATION it is
//! `[mb=slots, in=hd]` sticked on `in` — exactly what :18 asks for, with no copy and no extra plane.
//!
//! So `Sᵀ[slots, m] = K_nat[slots, hd] · Qᵀ[hd, m]`: the cache becomes the activation and the TINY Q
//! becomes the kernel. This file pins that the emitted descriptor really carries that shape, before any of
//! the downstream reorientation (the softmax's reduce axis, the score-block transpose the value leg needs)
//! is built — because if the operands are not legal none of that work is worth starting.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::SharedKernelBmmForm;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::matmul_opspec_off;
use ktir_superdsc::sdsc_abstract::{
    BlockCols, Lanes, MatK, MatM, MatN, MatY, PagedKvPool, QueryRowCount,
};
use ktir_superdsc::superdsc_opspec::Fp16;

const HD: u32 = 64;
const MQ_PAD: u32 = 64;

/// The declared `(layoutDimOrder_, stickDimOrder_)` of each primary dataspace in the emitted descriptor.
fn primary_layouts(
    op: &ktir_superdsc::superdsc_opspec::OpSpec,
    name: &str,
) -> Vec<(String, Vec<String>)> {
    let folds = ktir_superdsc::superdsc_opspec::SdscFoldSet::new(op.iter.cores_used());
    let emitted = ktir_superdsc::emit::emit_sdsc_tiled(name, op, &folds, &mut 0, None)
        .expect("the inverted leg must assemble");
    let v = serde_json::to_value(emitted.dsc()).expect("descriptor serialises");
    let info = &v["dscs_"][0][name]["primaryDsInfo_"];
    let mut out = Vec::new();
    for key in ["INPUT", "KERNEL", "OUTPUT"] {
        if let Some(e) = info.get(key) {
            let stick = e["stickDimOrder_"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            out.push((key.to_string(), stick));
        }
    }
    out
}

/// ⭐ THE SHAPE: the activation sticks on `in`, the kernel on `out` — `bmm.ddl:18` and `:23` respectively.
#[test]
fn the_inverted_leg_sticks_the_activation_on_in_and_the_kernel_on_out() {
    // Sᵀ[slots, m] = K_nat[slots, hd] · Qᵀ[hd, m]:
    //   M = the SLOT window (the cache is the activation, so its rows are the mb axis)
    //   N = the query rows  (Qᵀ's output columns)
    //   K = the head dim    (the contraction, and natural K's stick axis)
    let op = matmul_opspec_off::<Fp16>(
        MatM::of_query_rows(QueryRowCount::of_mq(PagedKvPool::PAGE_SLOTS as u32)),
        MatN::of_kv_window(BlockCols::of_one_stick(Lanes::FP16)),
        MatK::of_head_dim(HD),
        MatY::unbatched(),
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(MQ_PAD)),
        "t_kc",
        "t_qt",
        "t_sct",
        0,
        0,
        0,
    )
    .expect("the inverted score leg must be assemblable");

    let layouts = primary_layouts(&op, "inv_score");
    let stick_of = |role: &str| -> Vec<String> {
        layouts
            .iter()
            .find(|(k, _)| k == role)
            .map(|(_, s)| s.clone())
            .unwrap_or_default()
    };
    assert_eq!(
        stick_of("INPUT"),
        vec!["in".to_string()],
        "bmm.ddl:18 fixes an fp16 activation's slice dim to `in`, and natural K sticks on hd == in — so \
         the cache can BE the activation with no copy. Emitted: {layouts:?}"
    );
    assert_eq!(
        stick_of("KERNEL"),
        vec!["out".to_string()],
        "bmm.ddl:23 fixes an fp16 kernel's slice dim to `out`; Qᵀ's `out` is the query rows. Emitted: \
         {layouts:?}"
    );
}
