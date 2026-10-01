// SPDX-License-Identifier: Apache-2.0
//! The decode megakernel compiled into every metal decode tape in scope is well-formed, under the
//! model's KV codec (`--features turboquant` bakes the coded one): its segment kernels play the whole decode forward — every command the gates admit is a
//! step of exactly one segment kernel, spelled once, inside that kernel; no segment holds a wait
//! one threadgroup does not order (every in-segment wait is on the waiter's own lane or on a unit
//! every threadgroup plays for itself) and no kernel synchronizes across threadgroups at all; the
//! launches' address blocks are disjoint and inside the table, each step's row inside its block;
//! and every load constant is declared and read from a step. `--nocapture` prints, per model
//! and class, the segments (kernels and launches per forward) and why each launch
//! boundary stands.
#![cfg(feature = "metal")]

use std::collections::{BTreeMap, HashSet};

use scratchy_target_metal::interpreter::metal::MetalBucketSpec;
use scratchy_target_metal::tape::lowered::{
    ClassedTape, GateCtx, GenClass, MegakernelTape, MkCut, MkPlace,
};

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
    let mut printed = HashSet::new();
    for classed in decode.tapes.iter().filter(|t| !t.chunked) {
        check_class(
            &format!("{name} {:?}", classed.gen_class),
            classed,
            &mut printed,
        );
    }
}

/// The text of each segment kernel of `mk`, in segment order.
fn kernel_texts(mk: &MegakernelTape) -> Vec<&'static str> {
    let starts: Vec<usize> = (mk.segments.iter())
        .map(|s| {
            let decl = format!("void {}(", s.kernel);
            let at = mk.source.find(&decl);
            let at = at.unwrap_or_else(|| panic!("{}: {decl} not in the source", mk.library));
            assert_eq!(
                mk.source.matches(&decl).count(),
                1,
                "{}: {decl} declared once",
                mk.library
            );
            at
        })
        .collect();
    assert!(
        starts.windows(2).all(|w| w[0] < w[1]),
        "{}: the kernels follow the segments' order",
        mk.library
    );
    let ends = starts.iter().skip(1).copied().chain([mk.source.len()]);
    starts
        .iter()
        .zip(ends)
        .map(|(&s, e)| &mk.source[s..e])
        .collect()
}

fn check_class(name: &str, classed: &ClassedTape, printed: &mut HashSet<&'static str>) {
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
    assert!(
        mk.metallib.starts_with(b"MTLB"),
        "{name}: the library is compiled at build time"
    );
    let texts = kernel_texts(mk);
    // No kernel synchronizes across threadgroups: no atomic, no device-scope fence, nothing
    // that spins on another threadgroup's writes.
    for (text, segment) in texts.iter().zip(mk.segments) {
        for forbidden in ["atomic", "memory_order", "thread_scope_device", "while"] {
            assert!(
                !text.contains(forbidden),
                "{name}: {} holds `{forbidden}`",
                segment.kernel
            );
        }
    }
    // Each baked step is spelled once — alone, or in a co-issued group's one call — inside its
    // own segment's kernel, and its row lies inside that segment's block.
    for s in mk.steps {
        let spelled = format!("(baked {})", s.baked);
        assert_eq!(
            mk.source.matches(&spelled).count(),
            1,
            "{name}: baked step {} is spelled once",
            s.baked
        );
        let segment = &mk.segments[s.segment as usize];
        assert!(
            texts[s.segment as usize].contains(&spelled),
            "{name}: baked step {} is played by {}",
            s.baked,
            segment.kernel
        );
        assert!(
            s.row_at + s.row_len <= segment.block_len,
            "{name}: baked step {}'s row past its block",
            s.baked
        );
    }
    // THE SEGMENTS HOLD NO CROSS-THREADGROUP WAIT: a unit waits inside its segment only on a
    // unit of its own lane group, or on one every threadgroup plays for itself.
    let mut local = 0;
    for s in mk.steps {
        for &p in s.after {
            let on = &mk.steps[p as usize];
            assert_eq!(
                on.segment, s.segment,
                "{name}: baked step {} waits inside its segment on baked step {} of another",
                s.baked, on.baked
            );
            let ordered = match (s.place, on.place) {
                (_, MkPlace::Everywhere) => true,
                (MkPlace::Lane(a), MkPlace::Lane(b)) => a == b,
                _ => false,
            };
            assert!(
                ordered,
                "{name}: baked step {} ({:?}) waits inside its segment on baked step {} ({:?}), \
                 which no one threadgroup orders",
                s.baked, s.place, on.baked, on.place
            );
            local += 1;
        }
    }
    // Every admitted command is a step of a segment kernel that names it.
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
    // One launch per expanded instance of a segment's opening command, as many as each of its
    // steps has; the blocks the launches bind are disjoint and inside the table.
    let launches: Vec<u32> = (mk.segments.iter())
        .map(|seg| {
            let n = origins
                .iter()
                .filter(|o| o.baked == seg.opens as usize)
                .count();
            n as u32
        })
        .collect();
    for (s, step) in mk.steps.iter().enumerate() {
        assert_eq!(
            instances[s], launches[step.segment as usize],
            "{name}: baked step {} runs once per launch of its segment",
            step.baked
        );
    }
    let mut blocks: Vec<(u32, u32)> = (mk.segments.iter().zip(&launches))
        .map(|(seg, &n)| (seg.table_at, seg.table_at + n * seg.block_len))
        .collect();
    blocks.sort();
    assert!(
        blocks.windows(2).all(|w| w[0].1 <= w[1].0),
        "{name}: address blocks overlap"
    );
    assert!(
        blocks.last().is_some_and(|r| r.1 <= mk.table_len),
        "{name}: an address block past the table"
    );
    for (k, l) in mk.load_constants.iter().enumerate() {
        assert!(
            mk.steps.iter().any(|s| s.baked == l.baked),
            "{name}: a load constant of no step"
        );
        let decl = format!("MK_LOAD_{k} [[function_constant({})]]", l.index.get());
        assert!(mk.source.contains(&decl), "{name}: {decl} not declared");
    }
    // Why each launch boundary stands, per forward: the first launch, a rolled loop's
    // iteration (or its end), or a wait on a result spread over threadgroups or held by
    // single threadgroups.
    let per_forward: u32 = launches.iter().sum();
    let (mut breaks, mut spread, mut single) = (0, 0, 0);
    let mut pairs: BTreeMap<(&str, &str, &str), u32> = BTreeMap::new();
    for (seg, &n) in mk.segments.iter().zip(&launches) {
        let MkCut::Waits(on) = seg.cut else {
            breaks += n * u32::from(seg.cut == MkCut::Break);
            continue;
        };
        assert!(!on.is_empty(), "{name}: {} waits on nothing", seg.kernel);
        let many = on
            .iter()
            .any(|&p| mk.steps[p as usize].place == MkPlace::Spread);
        let (counter, kind) = match many {
            true => (&mut spread, "spread"),
            false => (&mut single, "single-TG"),
        };
        *counter += n;
        let opener = mk.steps.iter().find(|s| s.baked == seg.opens);
        let consumer = opener.expect("a segment opens with a step").function;
        for &p in on {
            *pairs
                .entry((kind, mk.steps[p as usize].function, consumer))
                .or_default() += n;
        }
    }
    assert_eq!(
        1 + breaks + spread + single,
        per_forward,
        "{name}: every launch after the first has one cause"
    );
    let fences = classed.tape.barriers_expanded();
    let dispatch_barriers = admitted.iter().filter(|&&i| fences[i]).count();
    println!(
        "{name}: {} commands, {} run at decode, played by {} segment kernels in {per_forward} \
         launches per forward (dispatch: {} launches, {dispatch_barriers} barriers); \
         boundaries: {breaks} loop iterations, {spread} wait on a result spread over \
         threadgroups, {single} only on single-threadgroup results; {local} in-segment waits, \
         each met by one threadgroup; {} load constants; {} bytes of MSL",
        commands.len(),
        admitted.len(),
        mk.segments.len(),
        admitted.len(),
        mk.load_constants.len(),
        mk.source.len(),
    );
    if printed.insert(mk.library) {
        for ((kind, from, to), n) in pairs {
            println!("  {n:4} x  {kind:9} {from} -> {to}");
        }
    }
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
