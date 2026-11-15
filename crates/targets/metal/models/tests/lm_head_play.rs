// SPDX-License-Identifier: Apache-2.0
//! A prompt's rows are scored for `prompt_logprobs` by the lm_head play of the prompt-scoring tile
//! (`ForwardReads::LmHead`, `PROMPT_SCORING_ROWS`): its tape's lm_head half alone, over final
//! hidden states the host writes into the bucket's backbone slot. So in every baked tape the play
//! runs some commands, all of them after every command of the backbone (the lm_head half is the
//! tape's tail), and none of them writes the KV cache; and the model compiles the tile's bucket.
#![cfg(compiled_model = "llama::smollm2_135m")]

use scratchy_target_metal::interpreter::metal::{ForwardReads, PROMPT_SCORING_ROWS};
use scratchy_target_metal::tape::lowered::KernelId as K;

#[test]
fn the_lm_head_play_is_each_tapes_tail_and_writes_no_cache() {
    use scratchy_target_metal_models::llama::smollm2_135m as model;
    assert!(
        (model::METAL_BUCKETS.iter()).any(|s| s.bucket_m == PROMPT_SCORING_ROWS),
        "the model compiles the prompt-scoring tile's bucket"
    );
    for spec in model::METAL_BUCKETS {
        for classed in spec.tapes {
            let tape = &classed.tape;
            let (commands, live) = (tape.commands_expanded(), tape.live_expanded());
            let at = spec.bucket_m;
            let plays: Vec<bool> = live.iter().map(|l| l.plays(ForwardReads::LmHead)).collect();
            let first = plays
                .iter()
                .position(|&p| p)
                .expect("the lm_head play runs commands");
            assert!(
                plays[first..].iter().all(|&p| p),
                "m={at}: the lm_head play is the tape's tail, every command from {first} on"
            );
            let writes_kv =
                |k: Option<K>| matches!(k, Some(K::RopeAppend | K::RopeAppendNormed | K::KvAppend));
            assert!(
                !commands[first..]
                    .iter()
                    .any(|c| writes_kv(c.command.kernel())),
                "m={at}: the lm_head play writes no KV"
            );
            // The whole forward plays the tail too, and the backbone's every command.
            assert!(
                live.iter().all(|l| l.plays(ForwardReads::Outputs)),
                "m={at}"
            );
        }
    }
}
