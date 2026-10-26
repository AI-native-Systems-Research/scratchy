// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ WHETHER dxp's CHUNK-PARAMETER EXPLORER RUNS IS THE GROUP'S, AND ONLY ONE SHAPE TURNS IT OFF.
//!
//! `DISABLE_ABOVE_LX_CHUNK_EXPLORE=1` used to be set on every `dxp_standalone` child, because the
//! token-plane gathered expert gemm (gemma-4's MoE, `GatherIndex::of_expert_planes`) only schedules
//! with the explorer off. Every other group then compiled without the LX chunking the explorer picks.
//! MEASURED, granite-3.1-2b fp16, knob dropped vs set on every group: TTFT 18100.6 → 8846.3 ms at a
//! 35300-token prompt; granite-3.1-8b fp16 decode at 10,086 tokens: ITL p50 305.9 → 233.1 ms. Same
//! descriptors both ways.
//!
//! These tests read the [`GroupInput`] `render_dxp_input` produces, which is what the bake's submit
//! site seals and hands to the compiler (`SealedGroup::of_rendered`) — the explorer setting a child is
//! spawned with comes from nowhere else.

use ktir_superdsc::emit;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::{
    SharedKernelBmmForm, matmul_opspec_off_operands_phys_gathered,
};
use ktir_superdsc::sdsc_abstract::{BlockCols, MatK, MatM, MatN, MatY, QueryRowCount, RowCount};
use ktir_superdsc::superdsc_opspec::{
    DataFormat, Df, EntryBase, Fp8, Fp16, GatherIndex, GatherRun, KernelAxis, PageExtent, Role,
    SdscFoldSet,
};
use scratchy_subtile::sdsc_abstract::{KernelTag, Stk};
use scratchy_target_spyre::dxp_cache::GroupDigest;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
use scratchy_target_spyre::superdsc_bake::{ChunkExplore, GroupId, SealedGroup};

/// A dense projection as every granite layer emits it: an unbatched fp16 matmul, no gather.
fn dense_matmul() -> emit::EmittedOp {
    superdsc::assemble_matmul(
        "matmul_o7",
        256,
        2048,
        2048,
        1,
        &emit::rb(
            "a",
            RowCount::of_token_rows(256),
            BlockCols::of_feature_cols(2048),
        ),
        &Stk::<KernelTag>::kernel(2048_usize, 2048_usize, "w"),
        &emit::rb(
            "o",
            RowCount::of_token_rows(256),
            BlockCols::of_feature_cols(2048),
        ),
        None,
    )
}

/// The token-plane gathered expert gemm, built the way the MoE door builds it: fp16 geometry, every
/// operand but the output and the index painted fp8, gathered through `of_expert_planes`.
fn gathered_expert_matmul() -> emit::EmittedOp {
    let mut spec = matmul_opspec_off_operands_phys_gathered::<Fp16>(
        MatM::of_token_rows(96),
        MatN::of_out_features(64),
        MatK::of_in_features(2816),
        MatY::unbatched(),
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
        "Tensor0",
        "Tensor1",
        "Tensor3",
        0,
        0,
        0,
        <Fp8 as DataFormat>::DF,
        None,
        Some(GatherIndex::of_expert_planes(
            "Tensor2".to_string(),
            EntryBase::ZERO,
        )),
    )
    .expect("the gathered expert matmul builds");
    for arg in &mut spec.args {
        if !matches!(arg.view().role, Role::Output | Role::Index) {
            arg.set_df(<Fp8 as DataFormat>::DF);
        }
    }
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit::emit_sdsc_tiled("emmm_s0_o9", &spec, &folds, &mut 0, None)
        .expect("the gathered expert matmul emits")
}

/// A KERNEL-LESS gather — the vendor's `identity` copy, roles `{OUTPUT, KERNEL_IDX}`. It reads
/// through an index too, but it is not the shape the explorer refuses.
fn gather_copy() -> emit::EmittedOp {
    let spec = gather_copy_opspec(
        "KtPool",
        "KtScratch",
        GatherIndex {
            name: "BlockTable".to_string(),
            entry_dim: KernelAxis::Batch,
            page: PageExtent::single_position(),
            per_position: None,
            first_entry: EntryBase::ZERO,
        },
        ktir_superdsc::sdsc_abstract::GatherCopySpec {
            rows: 8,
            cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
            df: Df::Fp16,
            run: GatherRun::CONTIGUOUS,
            // The destination's own law: the copy's `mb × out` = 512-element window at entry 0, exact.
            dest: ktir_superdsc::sdsc_abstract::GatherDest::Scratch {
                rows: 8,
                cols: ktir_superdsc::sdsc_abstract::POOL_STICK,
            },
        },
    )
    .expect("the gather copy builds");
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit::emit_sdsc_tiled("gather_o3", &spec, &folds, &mut 0, None).expect("the gather copy emits")
}

fn groups(ops: &[emit::EmittedOp]) -> Vec<superdsc::GroupInput> {
    superdsc::render_dxp_input(ops, superdsc::FoldGrouping::Split).expect("the groups render")
}

/// The explorer setting each group's dxp child is spawned with: the render sealed exactly as the
/// bake's submit site seals it.
fn submitted(ops: &[emit::EmittedOp]) -> Vec<ChunkExplore> {
    groups(ops)
        .iter()
        .map(|g| {
            let id = GroupId {
                fp: "test".to_string(),
                group: g.group,
            };
            SealedGroup::of_rendered(std::env::temp_dir(), id, g).explore()
        })
        .collect()
}

#[test]
fn a_dense_group_runs_the_explorer() {
    let jobs = submitted(&[dense_matmul()]);
    assert!(!jobs.is_empty());
    assert!(
        jobs.iter().all(|e| *e == ChunkExplore::Run),
        "a dense matmul has no gathered kernel, so dxp must keep its chunk explorer: {jobs:?}"
    );
}

#[test]
fn a_group_carrying_the_gathered_expert_gemm_skips_the_explorer() {
    assert_eq!(
        submitted(&[gathered_expert_matmul()]),
        [ChunkExplore::Skip],
        "a KERNEL read through a gather index only schedules with the explorer off \
         (`Expect LX in labeledDs memOrg_.`, L3DlOpsScheduler.cpp:2498)"
    );
}

#[test]
fn a_kernel_less_gather_runs_the_explorer() {
    let jobs = submitted(&[gather_copy()]);
    assert!(!jobs.is_empty());
    assert!(
        jobs.iter().all(|e| *e == ChunkExplore::Run),
        "an `identity` copy carries no KERNEL, so the explorer's LX demand never meets an index \
         and the group keeps it: {jobs:?}"
    );
}

/// ⛔ THE SETTING IS COMPILER INPUT, SO IT IS IN THE KEY. Same group, the setting flipped: the
/// memo and the disk cache key on `key`, and a key that ignored the setting would hand a group the
/// program compiled under the other one.
#[test]
fn the_explorer_setting_is_part_of_the_groups_key() {
    let [run] = &groups(&[dense_matmul()])[..] else {
        panic!("one dense matmul is one group")
    };
    let [skip] = &groups(&[gathered_expert_matmul()])[..] else {
        panic!("one gathered matmul is one group")
    };
    assert_ne!(run.key, skip.key);
    // Same files, both settings: the key must still differ — and neither may equal the files'
    // untagged digest, which is what every entry written before the setting was per group is keyed
    // by (all of them compiled with the explorer off).
    let rekey = |g: &superdsc::GroupInput, explore: ChunkExplore| {
        GroupDigest::of_compile(&g.files, explore.digest_tag())
    };
    assert_eq!(rekey(run, ChunkExplore::Run), run.key);
    assert_eq!(rekey(skip, ChunkExplore::Skip), skip.key);
    assert_ne!(
        rekey(run, ChunkExplore::Skip),
        run.key,
        "the same files under the other explorer setting must not share the key"
    );
    assert_ne!(
        GroupDigest::of(&run.files),
        run.key,
        "a Run group must miss every entry keyed before the setting entered the key"
    );
}

/// A group is ONE dxp compile, so one gathered-kernel trip anywhere in it decides for the group —
/// and only the group holding it skips.
#[test]
fn one_gathered_trip_turns_its_group_off_and_no_other() {
    let rendered = groups(&[dense_matmul(), gathered_expert_matmul(), dense_matmul()]);
    for g in &rendered {
        let holds_the_gather = g.files.iter().any(|(_, j)| j.contains("emmm_s0_o9"));
        let want = if holds_the_gather {
            ChunkExplore::Skip
        } else {
            ChunkExplore::Run
        };
        let id = GroupId {
            fp: "test".to_string(),
            group: g.group,
        };
        assert_eq!(
            SealedGroup::of_rendered(std::env::temp_dir(), id, g).explore(),
            want,
            "group {} (holds the gathered matmul: {holds_the_gather})",
            g.group
        );
    }
}
