// SPDX-License-Identifier: Apache-2.0
//! Splash's DFlash2 cycle (~/git/splash @ 1482f6bb runtime/model/Runtime.cpp:2923-3017) drafts
//! its block before its target verifies and commits the context of the rows its target retained to
//! the drafter's rings after its acceptance. Here those are two plays of the drafter's one tape —
//! its result play and its cache-writes play — so every baked tape plays its context commit in the
//! second alone and its block draft in the first alone.
#![cfg(all(feature = "metal", feature = "qwen3.6-35b-a3b-dflash2"))]

use scratchy_target_metal::interpreter::metal::ForwardReads;
use scratchy_target_metal::tape::lowered::KernelId as K;

#[test]
fn a_block_draft_and_its_commit_play_apart() {
    use scratchy_models::dflash2::qwen3_6_35b_a3b_dflash2_mlx_affine_b4_g64_qembed as drafter;
    assert!(!drafter::METAL_BUCKETS.is_empty());
    for spec in drafter::METAL_BUCKETS {
        for classed in spec.tapes {
            let tape = &classed.tape;
            let (commands, live) = (tape.commands_expanded(), tape.live_expanded());
            let plays = |kernel: K, reads: ForwardReads| -> Vec<bool> {
                (commands.iter().zip(&live))
                    .filter(|(c, _)| c.command.kernel == kernel)
                    .map(|(_, l)| l.plays(reads))
                    .collect()
            };
            let at = spec.bucket_m;
            // Its six layers' commits, a cycle's and a prompt's.
            for kernel in [K::DraftContextCommit, K::DraftContextPrefill] {
                let commit = plays(kernel, ForwardReads::CacheWrites);
                assert_eq!(commit, [true; 6], "m={at}: {kernel:?} in the cache play");
                let result = plays(kernel, ForwardReads::Result);
                assert_eq!(result, [false; 6], "m={at}: {kernel:?} in the result play");
                let rebuild = plays(kernel, ForwardReads::Rebuild);
                assert_eq!(rebuild, [true; 6], "m={at}: {kernel:?} in the rebuild play");
            }
            // Its context window (Splash `DFlashContextWindow`): a commit stores its rows, its
            // layers reading them exact; a rebuild loads them back, and plays nothing before.
            for kernel in [K::DraftWindowStoreCommit, K::DraftWindowStorePrefill] {
                assert_eq!(plays(kernel, ForwardReads::CacheWrites), [true], "m={at}");
                assert_eq!(plays(kernel, ForwardReads::Rebuild), [false], "m={at}");
                assert_eq!(plays(kernel, ForwardReads::Result), [false], "m={at}");
            }
            for reads in [
                ForwardReads::CacheWrites,
                ForwardReads::Result,
                ForwardReads::Outputs,
            ] {
                let load = plays(K::DraftWindowLoad, reads);
                assert_eq!(
                    load,
                    [false],
                    "m={at}: the window's load in a {reads:?} play"
                );
            }
            assert_eq!(
                plays(K::DraftWindowLoad, ForwardReads::Rebuild),
                [true],
                "m={at}"
            );
            let load = (commands.iter()).position(|c| c.command.kernel == K::DraftWindowLoad);
            let before = (commands.iter().zip(&live)).take(load.expect("a window load"));
            for (c, l) in before {
                let kernel = c.command.kernel;
                assert!(
                    !l.plays(ForwardReads::Rebuild),
                    "m={at}: {kernel:?} before the load"
                );
            }
            for kernel in [
                K::DraftConv,
                K::DraftAttentionSplit,
                K::DraftSelectTop16,
                K::DraftSelectWalk,
                K::DraftVerifyRows,
            ] {
                let result = plays(kernel, ForwardReads::Result);
                assert!(
                    !result.is_empty() && result.iter().all(|&p| p),
                    "m={at}: {kernel:?} in the result play"
                );
                let cache = plays(kernel, ForwardReads::CacheWrites);
                assert!(
                    cache.iter().all(|&p| !p),
                    "m={at}: {kernel:?} in the cache play"
                );
            }
        }
    }
}

/// Its `target_hidden` rows are its target's eight captured layers side by side
/// (`backbone_hidden_size`, `fc`'s input: Splash's `capturedTargetHidden`), so its context load
/// reads that many columns a row, not its own hidden size; and its target routes experts, so its
/// selector verifies the rows Splash's verify budget chooses (`--verify-rows auto`).
#[test]
fn its_context_rows_and_verify_rows_are_its_targets() {
    use scratchy_forward_compiler::CanonicalParams;
    use scratchy_models::dflash2::qwen3_6_35b_a3b_dflash2_mlx_affine_b4_g64_qembed as drafter;
    const {
        assert!(<drafter::Weights as CanonicalParams>::TARGET_HIDDEN_WIDTH == 8 * 2048);
        assert!(<drafter::Weights as CanonicalParams>::TARGET_ROUTES_EXPERTS);
    }
}

/// Its block draft runs a step's lanes' blocks, at most its target's verify rows (8 here): every
/// bucket past the one holding them plays its context commit alone, and its arena holds none of
/// the block's tiles — the lm_head's logits, a vocabulary a row, above all (Splash's two draft
/// graphs, runtime/model/DFlashDraft.cpp, size their buffers apart).
#[test]
fn buckets_past_its_block_play_its_commit_alone() {
    use scratchy_models::dflash2::qwen3_6_35b_a3b_dflash2_mlx_affine_b4_g64_qembed as drafter;
    for spec in drafter::METAL_BUCKETS {
        let m = spec.bucket_m;
        assert_eq!(
            spec.plays_result,
            m <= 8,
            "m={m}: whether it plays its block draft"
        );
        let logits = u64::from(m) * 248_320 * 2;
        let arena: u64 = spec.arena_bytes.iter().sum();
        assert!(
            spec.plays_result || arena < logits / 4,
            "m={m}: an arena of {arena} bytes beside logits of {logits}"
        );
    }
}
