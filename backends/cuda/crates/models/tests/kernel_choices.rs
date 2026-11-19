// SPDX-License-Identifier: Apache-2.0
//! Every kernel choice a step stores is the one its constants pick: over every tape of every model
//! this build compiled, each GEMV's K split is `gemv_splits` of its shape and pairs, each norm's
//! kernel is `norm_row_takes` of its rows and columns, each decode attention's split is
//! `attn_decode_split` of its head dim, and each GDN step's scan is `gdn_scan_is_recurrent` of
//! its rows. The player launches what the step stores, so these are the functions it no longer
//! calls. No toolchain and no GPU.

// The models register through `inventory`; nothing here names one, so link the crate.
extern crate scratchy_target_cuda_models as _;

use scratchy_target_cuda::registry;
use scratchy_target_cuda::step::{
    AttnDecodeSplit, CudaStep, ExpertGemvBf16, GdnScan, GemvSplit, MatmulKernel, NormKernel,
    RmsNormBf16, attn_decode_split, gdn_scan_is_recurrent, gemv_splits, norm_row_takes,
};
use scratchy_target_cuda::step::{MinTiles, Splits};

/// The split an `n` × `k` GEMV over `pairs` slices takes.
fn split(n: u32, k: u32, pairs: u32) -> GemvSplit {
    GemvSplit {
        splits: Splits(gemv_splits(n, k, pairs)),
    }
}

/// The split of an expert GEMV at its `m · top_k` pairs.
fn expert_split(g: &ExpertGemvBf16) -> GemvSplit {
    let (n, k) = g.n_k();
    split(n, k, g.m.get().saturating_mul(g.top_k.get()))
}

/// The kernel a norm's rows and columns pick.
fn norm_kernel(n: &RmsNormBf16) -> NormKernel {
    match norm_row_takes(n.rows.get(), n.cols.get()) {
        true => NormKernel::Row,
        false => NormKernel::Tiled,
    }
}

#[test]
fn every_stored_kernel_choice_is_its_constants_pick() {
    let mut checked = 0usize;
    let mut entries = 0usize;
    for entry in registry::entries() {
        entries += 1;
        for tape in entry.tapes {
            for (i, s) in tape.steps.iter().enumerate() {
                let at = format!("{} m={} step {i}", entry.stem, tape.bucket_m);
                match s {
                    CudaStep::Gemv(g) | CudaStep::GemvFp8(g) => {
                        assert_eq!(g.split, split(g.gemm.n.get(), g.gemm.k.get(), 1), "{at}");
                    }
                    CudaStep::GemvGroup(g) | CudaStep::GemvGroupFp8(g) => {
                        let members = g.group.members();
                        for (m, stored) in members.iter().zip(g.splits()) {
                            let want = m.map(|m| split(m.n.get(), g.group.k.get(), 1));
                            assert_eq!(stored, want, "{at}");
                        }
                    }
                    CudaStep::RouterLogits(r) => {
                        if let MatmulKernel::Gemv(stored) = r.kernel {
                            assert_eq!(stored, split(r.experts.get(), r.hidden.get(), 1), "{at}");
                        }
                    }
                    CudaStep::GateUpAct(g) => {
                        if let Some(stored) = g.kernel.gemv_split() {
                            assert_eq!(stored, split(g.gate.n.get(), g.gate.k.get(), 1), "{at}");
                        }
                    }
                    CudaStep::GateUpTcFp8(t) => {
                        if let Some(stored) = t.fold.kernel.gemv_split() {
                            assert_eq!(
                                stored,
                                split(t.fold.gate.n.get(), t.fold.gate.k.get(), 1),
                                "{at}"
                            );
                        }
                    }
                    CudaStep::ExpertGemv(g) => {
                        assert_eq!(g.split, expert_split(&g.gemv), "{at}");
                    }
                    CudaStep::ExpertGateUpAct(g) => {
                        assert_eq!(g.split, expert_split(&g.fold.gate), "{at}");
                    }
                    CudaStep::RmsNorm(n) | CudaStep::RouterNorm(n) => {
                        assert_eq!(n.kernel, norm_kernel(n), "{at}");
                    }
                    CudaStep::AddRmsNorm(f) => {
                        assert_eq!(f.norm.kernel, norm_kernel(&f.norm), "{at}");
                    }
                    CudaStep::NormAdd(f) => {
                        assert_eq!(f.norm.kernel, norm_kernel(&f.norm), "{at}");
                    }
                    CudaStep::RmsNormQuantFp8(t) => {
                        assert_eq!(t.norm.kernel, norm_kernel(&t.norm), "{at}");
                        assert_eq!(
                            t.norm.kernel,
                            NormKernel::Row,
                            "{at}: the fold plays the row kernel"
                        );
                    }
                    CudaStep::AddRmsNormQuantFp8(t) => {
                        assert_eq!(t.fold.norm.kernel, norm_kernel(&t.fold.norm), "{at}");
                        assert_eq!(
                            t.fold.norm.kernel,
                            NormKernel::Row,
                            "{at}: the fold plays the row kernel"
                        );
                    }
                    CudaStep::AttnDecode(a) => {
                        let want =
                            attn_decode_split(a.attn.head_dim.get()).map(|(splits, min_tiles)| {
                                AttnDecodeSplit {
                                    splits: Splits(splits),
                                    min_tiles: MinTiles(min_tiles),
                                }
                            });
                        assert_eq!(Some(a.split), want, "{at}");
                    }
                    CudaStep::GatedDeltaNet(g) => {
                        let want = match gdn_scan_is_recurrent(g.gdn.rows.get()) {
                            true => GdnScan::Recurrent,
                            false => GdnScan::Chunked,
                        };
                        assert_eq!(g.scan, want, "{at}");
                    }
                    _ => continue,
                }
                checked += 1;
            }
        }
    }
    assert!(entries > 0, "no CUDA model in this build's scope");
    assert!(checked > 0, "no step stores a kernel choice");
    eprintln!("{checked} stored kernel choices over {entries} models, each its constants' pick");
}
