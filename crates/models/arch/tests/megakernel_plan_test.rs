// SPDX-License-Identifier: Apache-2.0
//! The decode megakernel compiled into every metal decode tape in scope is well-formed, under the
//! model's KV codec (`--features turboquant` bakes the coded one): ONE kernel plays the whole
//! decode forward — its launch spans every command the gates
//! admit, every admitted command is a step of it, each step names the command it plays, each
//! baked step is spelled once in the kernel text (its loops rolled), the steps' address rows are
//! disjoint and inside the table, and every load constant is declared and read from a step.
//! `--nocapture` prints what the bake found.
#![cfg(feature = "metal")]

use scratchy_target_metal::interpreter::metal::MetalBucketSpec;
use scratchy_target_metal::tape::lowered::{ClassedTape, GateCtx, GenClass};

fn check(name: &str, buckets: &[MetalBucketSpec]) {
    let decode = buckets
        .iter()
        .find(|b| b.bucket_m == 1)
        .expect("a decode bucket");
    let classes: Vec<GenClass> = decode
        .tapes
        .iter()
        .filter(|t| !t.chunked)
        .map(|t| t.gen_class)
        .collect();
    assert_eq!(
        classes,
        [GenClass::M1, GenClass::Mid, GenClass::M5],
        "{name}: every GPU generation's decode tape"
    );
    for classed in decode.tapes.iter().filter(|t| !t.chunked) {
        check_class(&format!("{name} {:?}", classed.gen_class), classed);
    }
}

fn check_class(name: &str, classed: &ClassedTape) {
    let commands = classed.tape.commands_expanded();
    let origins = classed.tape.expanded_origins();
    let [mk] = classed.megakernel else {
        panic!(
            "{name}: {} megakernels baked for the decode tape",
            classed.megakernel.len()
        );
    };
    let ctx = GateCtx::decode_one(false);
    let admitted: Vec<usize> = (0..commands.len())
        .filter(|&i| commands[i].gate.is_none_or(|g| g.admits(ctx)))
        .collect();
    let span = mk.commands.start as usize..mk.commands.end as usize;
    assert!(
        admitted.iter().all(|i| span.contains(i)),
        "{name}: the launch spans every admitted command"
    );
    assert!(
        source_has(mk.source, &format!("void {}(", mk.kernel)),
        "{name}: the kernel is in the source"
    );
    assert!(
        mk.metallib.starts_with(b"MTLB"),
        "{name}: the library is compiled at build time"
    );
    // The work split is the launch's: every lane a function of the GPU's cores (`MK_P`).
    assert!(
        !source_has(mk.source, "mk_tg == ") || source_has(mk.source, "% MK_P;"),
        "{name}: a pinned lane not taken modulo the launch's threadgroups"
    );
    // Each baked step is spelled once — alone, or in a co-issued group's one call.
    for s in mk.steps {
        assert_eq!(
            mk.source.matches(&format!("(baked {})", s.baked)).count(),
            1,
            "{name}: baked step {} is spelled once",
            s.baked
        );
    }
    let calls = mk.source.matches("(s, mk_lane(s, ").count();
    assert!(
        (1..=mk.steps.len()).contains(&calls),
        "{name}: {calls} adapter calls for {} baked steps",
        mk.steps.len()
    );
    let mut instances = vec![0u32; mk.steps.len()];
    for &i in &admitted {
        let c = &commands[i].command;
        let s = mk
            .steps
            .iter()
            .position(|s| s.baked as usize == origins[i].baked)
            .unwrap_or_else(|| panic!("{name}: admitted command {i} ({}) not played", c.function));
        assert_eq!(
            mk.steps[s].function, c.function,
            "{name}: a step names its command"
        );
        let need = c.bindings.iter().map(|b| u32::from(b.binding_index()) + 1);
        assert!(
            need.max().unwrap_or(0) <= mk.steps[s].row_len,
            "{name}: a binding past its address row"
        );
        instances[s] += 1;
    }
    let mut rows: Vec<(u32, u32)> = mk
        .steps
        .iter()
        .zip(&instances)
        .map(|(s, &n)| (s.table_at, s.table_at + n * s.row_len))
        .collect();
    rows.sort();
    assert!(
        rows.windows(2).all(|w| w[0].1 <= w[1].0),
        "{name}: address rows overlap"
    );
    assert!(
        rows.last().is_some_and(|r| r.1 <= mk.table_len),
        "{name}: an address row past the table"
    );
    for (k, l) in mk.load_constants.iter().enumerate() {
        assert!(
            mk.steps.iter().any(|s| s.baked == l.baked),
            "{name}: a load constant of no step"
        );
        let decl = format!("MK_LOAD_{k} [[function_constant({})]]", l.index.get());
        assert!(source_has(mk.source, &decl), "{name}: {decl} not declared");
    }
    // The dispatch path's barriers at decode, which the launch replaces.
    let fences = classed.tape.barriers_expanded();
    let dispatch_barriers = admitted.iter().filter(|&&i| fences[i]).count();
    println!(
        "{name}: {} commands, {} run at decode, ALL in ONE kernel ({} baked \
         steps, loops rolled); {} grid barriers per forward (dispatch: {} launches, \
         {dispatch_barriers} barriers); {} load constants; {} bytes of MSL",
        commands.len(),
        admitted.len(),
        mk.steps.len(),
        mk.grid_barriers,
        admitted.len(),
        mk.load_constants.len(),
        mk.source.len(),
    );
}

fn source_has(source: &str, text: &str) -> bool {
    source.contains(text)
}

#[cfg(feature = "llama-3.2-3b")]
#[test]
fn llama_3_2_3b_decode_megakernel() {
    check(
        "llama-3.2-3b",
        scratchy_models::llama::llama_3_2_3b_mlx_affine_b4_g64::METAL_BUCKETS,
    );
}

#[cfg(feature = "qwen3.5-0.8b")]
#[test]
fn qwen3_5_0_8b_decode_megakernel() {
    check(
        "qwen3.5-0.8b",
        scratchy_models::qwen3_5::qwen3_5_0_8b_mlx_affine_b4_g64::METAL_BUCKETS,
    );
}

#[cfg(feature = "gemma-3-1b-it")]
#[test]
fn gemma_3_1b_decode_megakernel() {
    check(
        "gemma-3-1b-it",
        scratchy_models::gemma3::gemma_3_1b_it_mlx_affine_b4_g64::METAL_BUCKETS,
    );
}

#[cfg(feature = "gemma-4-26b-a4b-it")]
#[test]
fn gemma_4_26b_a4b_decode_megakernel() {
    check(
        "gemma-4-26b-a4b-it",
        scratchy_models::gemma4_moe::gemma_4_26b_a4b_it_mlx_affine_b4_g64::METAL_BUCKETS,
    );
}
