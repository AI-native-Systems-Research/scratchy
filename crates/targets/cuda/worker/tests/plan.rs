// SPDX-License-Identifier: Apache-2.0
//! The CUDA worker's step planning, with no toolchain: the bucket a step runs in, its padded
//! inputs, the one-request rule, the GDN start, the greedy token, and the sampling it refuses;
//! and the edge prompts planned step by step (one token; one past the largest bucket).

use std::collections::HashMap;

use scratchy_core_common::SamplingParams;
use scratchy_forward_compiler::gdn_slot_allocator::{GdnRow, GdnSlotAllocator};
use scratchy_serving_engine::worker_factory::WorkerFactory;
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, KvClass, KvTape};
use scratchy_target_cuda_worker::plan::{
    PlanError, block_table_u32, check_sampling, choose_bucket, emits_token, fill_inputs, gdn_fresh,
    greedy_token, kv_block_bytes, logits_row, scheduled_request,
};
use scratchy_target_cuda_worker::{CudaWorkerFactory, NO_NVCC};

/// qwen3.5-0.8b's buckets, as `CUDA_TAPES` orders them.
const BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

#[test]
fn a_step_runs_in_the_smallest_bucket_that_holds_it() {
    for (n, want) in [
        (1, 1),
        (2, 512),
        (511, 512),
        (512, 512),
        (513, 1024),
        (4096, 4096),
    ] {
        let (ix, m) = choose_bucket(&BUCKETS, n).unwrap();
        assert_eq!((BUCKETS[ix], m), (want, want), "{n} tokens");
    }
    assert_eq!(
        choose_bucket(&BUCKETS, 4097),
        Err(PlanError::NoBucket {
            tokens: 4097,
            largest: 4096
        })
    );
}

/// The choice reads each tape's `bucket_m`, not its place in the list.
#[test]
fn the_bucket_is_chosen_by_its_rows_not_its_position() {
    let shuffled = [2048, 1, 4096, 512, 1024];
    for (n, want) in [(1, 1), (2, 512), (513, 1024), (3000, 4096)] {
        let (ix, m) = choose_bucket(&shuffled, n).unwrap();
        assert_eq!((shuffled[ix], m), (want, want), "{n} tokens");
    }
}

#[test]
fn inputs_are_padded_to_the_bucket_and_positions_run_through_it() {
    let (mut ids, mut positions) = (vec![9; 3], vec![9; 3]);
    fill_inputs(&[5, 6, 7], 100, 8, &mut ids, &mut positions).unwrap();
    assert_eq!(ids, [5, 6, 7, 0, 0, 0, 0, 0]);
    assert_eq!(positions, (100..108).collect::<Vec<u32>>());
    // A decode: one row, at its own position.
    fill_inputs(&[42], 4095, 1, &mut ids, &mut positions).unwrap();
    assert_eq!(
        (ids.as_slice(), positions.as_slice()),
        (&[42][..], &[4095][..])
    );
    assert_eq!(
        fill_inputs(&[1], u32::MAX, 2, &mut ids, &mut positions),
        Err(PlanError::Positions {
            start: u32::MAX,
            bucket_m: 2
        })
    );
}

#[test]
fn the_block_table_is_the_schedulers_ids_as_u32() {
    let mut out = vec![7];
    block_table_u32(&[3, 1, 128], &mut out).unwrap();
    assert_eq!(out, [3, 1, 128]);
    let past = u32::MAX as usize + 1;
    assert_eq!(
        block_table_u32(&[1, past], &mut out),
        Err(PlanError::Block {
            entry: 1,
            block: past
        })
    );
}

#[test]
fn one_request_per_step() {
    let none: HashMap<String, usize> = [("a".to_string(), 0)].into();
    assert_eq!(scheduled_request(&none), Ok(None));
    let one: HashMap<String, usize> = [("a".to_string(), 0), ("b".to_string(), 7)].into();
    assert_eq!(scheduled_request(&one), Ok(Some(("b", 7))));
    let two: HashMap<String, usize> = [("b".to_string(), 1), ("a".to_string(), 3)].into();
    let e = scheduled_request(&two).unwrap_err();
    assert_eq!(
        e,
        PlanError::ManyRequests {
            req_ids: vec!["a".into(), "b".into()]
        }
    );
    let msg = e.to_string();
    assert!(
        msg.contains("one sequence per step") && msg.contains("max_num_seqs is 1"),
        "{msg}"
    );
}

#[test]
fn only_the_step_that_reaches_the_history_end_samples() {
    // A 5000-token prompt in chunks of 4096 and 904.
    assert!(!emits_token(0, 4096, 5000));
    assert!(emits_token(4096, 904, 5000));
    // A decode of the token sampled last.
    assert!(emits_token(5000, 1, 5001));
}

#[test]
fn the_greedy_token_is_the_argmax_with_ties_to_the_lowest_id() {
    let row = [bf16(0.5), bf16(2.0), bf16(-1.0), bf16(2.0), bf16(1.0)];
    assert_eq!(greedy_token(&row), Ok(1));
    let all_equal = [bf16(3.0); 4];
    assert_eq!(greedy_token(&all_equal), Ok(0));
    let negative = [bf16(-3.0), bf16(-1.0), bf16(-1.0)];
    assert_eq!(greedy_token(&negative), Ok(1));
    let nan = [bf16(1.0), bf16(f32::NAN)];
    assert_eq!(greedy_token(&nan), Err(PlanError::NanLogit { id: 1 }));
}

#[test]
fn the_sampled_row_is_the_last_real_one() {
    // Three rows of a vocab of 2.
    let result = [1, 2, 3, 4, 5, 6];
    assert_eq!(logits_row(&result, 3, 1), Ok(&[3, 4][..]));
    assert_eq!(logits_row(&result, 3, 2), Ok(&[5, 6][..]));
    assert!(logits_row(&result, 3, 3).is_err());
    assert!(logits_row(&result, 4, 0).is_err());
}

/// One K and one V block per KV layer: qwen3.5-0.8b's 6 layers of 2 heads × 64 × 256 bf16 are
/// 786,432 bytes a block.
#[test]
fn a_kv_block_costs_k_and_v_per_layer() {
    let span = ArenaSpan {
        offset: ArenaOffset(0),
        bytes: 0,
    };
    let kv = KvTape {
        layers: &[3, 7, 11, 15, 19, 23],
        kv_heads: 2,
        head_dim: 256,
        block_size: 64,
        max_blocks_per_seq: 128,
        block_table: span,
        slot_mapping: span,
        sequence: span,
        classes: &[KvClass {
            kv_heads: 2,
            head_dim: 256,
            window: None,
        }],
        layer_class: &[0, 0, 0, 0, 0, 0],
    };
    assert_eq!(kv_block_bytes(&kv, 64), Some(786_432));
}

#[test]
fn greedy_sampling_is_served() {
    assert_eq!(check_sampling(Some(&greedy())), Ok(()));
    // Top-k, top-p and min-p keep the argmax at temperature 0.
    let filtered = SamplingParams {
        top_k: 20,
        top_p: 0.95,
        min_p: 0.1,
        ..greedy()
    };
    assert_eq!(check_sampling(Some(&filtered)), Ok(()));
}

/// Every refusal names what to change.
#[test]
fn anything_but_greedy_is_refused_with_the_fix() {
    let cases: Vec<(SamplingParams, PlanError, &str)> = vec![
        (
            SamplingParams {
                temperature: 0.7,
                ..greedy()
            },
            PlanError::NotGreedy { temperature: 0.7 },
            "--temperature 0",
        ),
        (
            SamplingParams::default(),
            PlanError::NotGreedy { temperature: 1.0 },
            "\"temperature\": 0",
        ),
        (
            SamplingParams { n: 2, ..greedy() },
            PlanError::Completions { n: 2 },
            "\"n\": 1",
        ),
        (
            SamplingParams {
                logprobs: Some(5),
                ..greedy()
            },
            PlanError::Logprobs,
            "drop \"logprobs\"",
        ),
        (
            SamplingParams {
                prompt_logprobs: Some(1),
                ..greedy()
            },
            PlanError::PromptLogprobs,
            "drop \"prompt_logprobs\"",
        ),
        (
            SamplingParams {
                repetition_penalty: 1.1,
                ..greedy()
            },
            PlanError::Penalties,
            "\"repetition_penalty\": 1",
        ),
        (
            SamplingParams {
                logit_bias: Some([(3, 1.0)].into()),
                ..greedy()
            },
            PlanError::LogitTransform {
                field: "logit_bias",
            },
            "drop \"logit_bias\"",
        ),
        (
            SamplingParams {
                allowed_token_ids: Some(vec![1]),
                ..greedy()
            },
            PlanError::LogitTransform {
                field: "allowed_token_ids",
            },
            "drop \"allowed_token_ids\"",
        ),
    ];
    for (params, want, fix) in cases {
        let got = check_sampling(Some(&params)).unwrap_err();
        assert_eq!(got, want);
        let msg = got.to_string();
        assert!(
            msg.contains("samples greedily only") && msg.contains(fix),
            "{msg} names {fix}"
        );
    }
    let none = check_sampling(None).unwrap_err().to_string();
    assert!(none.contains("--temperature 0"), "{none}");
}

/// The factory claims `cuda` and `auto` only, addresses KV in 64-token blocks, serves no cached
/// prefix, and without `nvcc` refuses to build a worker with the rebuild command.
#[test]
fn the_factory_claims_cuda_and_auto() {
    let f = CudaWorkerFactory;
    assert!(f.matches("cuda") && f.matches("auto"));
    for other in ["metal", "spyre", "cpu", "tpu", "CUDA", ""] {
        assert!(!f.matches(other), "{other}");
    }
    assert_eq!(f.required_block_size().map(|b| b.get()), Some(64));
    assert!(!f.supports_prefix_caching());
    assert!(NO_NVCC.contains("built without the `nvcc` feature"));
    assert!(NO_NVCC.contains("--features nvcc,model/<stem>"));
}

/// One planned step of a request: its bucket's rows, whether it starts its GDN state from zero,
/// and whether it samples.
#[derive(Debug, PartialEq)]
struct Planned {
    bucket_m: u32,
    fresh: bool,
    samples: bool,
}

/// Plan a request of `prompt` tokens through `decodes` decode steps, chunked by the step budget
/// of 4096, through the shared GDN slot allocator, as the worker plans each step.
fn plan_request(prompt: usize, decodes: usize) -> Vec<Planned> {
    let mut gdn = GdnSlotAllocator::new(1, None);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < prompt {
        let n = (prompt - start).min(4096);
        chunks.push((start, n, prompt));
        start += n;
    }
    for d in 0..decodes {
        chunks.push((prompt + d, 1, prompt + d + 1));
    }
    let (mut ids, mut positions) = (Vec::new(), Vec::new());
    (chunks.into_iter())
        .map(|(start, n, history)| {
            let (_, bucket_m) = choose_bucket(&BUCKETS, n).unwrap();
            fill_inputs(
                &vec![7; n],
                start as u32,
                bucket_m,
                &mut ids,
                &mut positions,
            )
            .unwrap();
            assert_eq!(ids.len(), bucket_m as usize);
            assert_eq!(positions[0], start as u32);
            let (_, codes) = gdn.step([GdnRow::new(42, 0, start as u32)]).unwrap();
            Planned {
                bucket_m,
                fresh: gdn_fresh(codes[0], start as u32).unwrap(),
                samples: emits_token(start, n, history),
            }
        })
        .collect()
}

/// A one-token prompt runs its first, fresh step on the one-row decode tape, samples from it, and
/// decodes on from the slot.
#[test]
fn a_one_token_prompt_starts_fresh_on_the_decode_tape() {
    let plan = plan_request(1, 2);
    let step = |fresh| Planned {
        bucket_m: 1,
        fresh,
        samples: true,
    };
    assert_eq!(plan, [step(true), step(false), step(false)]);
}

/// A 4097-token prompt is a full 4096 chunk that samples nothing and a one-token remainder, on the
/// decode tape, that continues the state and samples the first token.
#[test]
fn a_prompt_one_past_the_largest_bucket_ends_on_a_one_token_chunk() {
    let plan = plan_request(4097, 1);
    assert_eq!(
        plan,
        [
            Planned {
                bucket_m: 4096,
                fresh: true,
                samples: false
            },
            Planned {
                bucket_m: 1,
                fresh: false,
                samples: true
            },
            Planned {
                bucket_m: 1,
                fresh: false,
                samples: true
            },
        ]
    );
}

/// A step code that is not plain, or fresh away from token 0, is refused.
#[test]
fn a_gdn_start_other_than_fresh_at_zero_is_refused() {
    let mut gdn = GdnSlotAllocator::new(1, None);
    let (_, codes) = gdn.step([GdnRow::new(1, 0, 0)]).unwrap();
    assert_eq!(gdn_fresh(codes[0], 0), Ok(true));
    assert_eq!(
        gdn_fresh(codes[0], 5),
        Err(PlanError::GdnStart {
            start: 5,
            code: codes[0]
        })
    );
    let (_, codes) = gdn.step([GdnRow::new(1, 0, 5)]).unwrap();
    assert_eq!(gdn_fresh(codes[0], 5), Ok(false));
    assert!(gdn_fresh(codes[0], 0).is_err());
}

/// The factory states the greedy-only limit to the engine, which checks it when a request is
/// submitted: non-greedy sampling is refused there, before scheduling, with the worker's message.
#[test]
fn the_factory_reports_the_greedy_only_limit() {
    let f = CudaWorkerFactory;
    assert_eq!(f.check_sampling(&greedy()), Ok(()));
    let hot = SamplingParams {
        temperature: 0.7,
        ..greedy()
    };
    let msg = f.check_sampling(&hot).unwrap_err();
    assert_eq!(
        msg,
        PlanError::NotGreedy { temperature: 0.7 }.to_string(),
        "the worker's own message"
    );
    assert!(msg.contains("--temperature 0"), "{msg}");
    let found = scratchy_serving_engine::worker_factory::factory_for("cuda")
        .expect("a registered factory claims --device cuda");
    assert!(found.check_sampling(&hot).is_err(), "the registered one");
}

/// The factory is in the neutral registry `create_worker` walks.
#[test]
fn the_factory_is_registered() {
    let found = inventory::iter::<&dyn WorkerFactory>
        .into_iter()
        .any(|f| f.matches("cuda"));
    assert!(found, "no registered factory claims --device cuda");
}
