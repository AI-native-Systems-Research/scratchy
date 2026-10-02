// SPDX-License-Identifier: Apache-2.0
//! The decode megakernel compiled into every metal decode tape in scope is well-formed, under the
//! model's KV codec (`--features turboquant` bakes the coded one): every command the gates admit
//! is a step of exactly one region, played in every instance of it; every unit of a region opens
//! a run of itself, so the load's plan always tiles it; every run's kernel is declared once and
//! spells each step of its units once; no run holds a wait one threadgroup does not order (every
//! wait inside is on the waiter's own lane group or on a unit every threadgroup plays for itself)
//! and no kernel synchronizes across threadgroups at all; the regions' address blocks are disjoint
//! and inside the table, each step's row inside its block; and every load constant is declared
//! and read from a step. `--nocapture` prints, per model and class, the regions and the runs a
//! launch may play in each.
#![cfg(feature = "metal")]

use std::collections::HashSet;

use scratchy_target_metal::interpreter::metal::MetalBucketSpec;
use scratchy_target_metal::tape::lowered::{
    ClassedTape, GateCtx, GenClass, MegakernelTape, MkPlace,
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

/// The text of the kernel `kernel` of `mk`: from its declaration to the next kernel's.
fn kernel_text(mk: &MegakernelTape, kernel: &str) -> &'static str {
    let decl = format!("void {kernel}(");
    let at = mk.source.find(&decl);
    let at = at.unwrap_or_else(|| panic!("{}: {decl} not in the source", mk.library));
    assert_eq!(
        mk.source.matches(&decl).count(),
        1,
        "{}: {decl} declared once",
        mk.library
    );
    let rest = &mk.source[at + decl.len()..];
    let end = rest
        .find("[[kernel")
        .map_or(mk.source.len(), |e| at + decl.len() + e);
    &mk.source[at..end]
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
    // A region's steps are consecutive, and its units tile them in order.
    let steps_of: Vec<std::ops::Range<u32>> = (0..mk.regions.len())
        .map(|r| {
            let held: Vec<u32> = (0u32..)
                .zip(mk.steps)
                .filter(|(_, s)| s.region as usize == r)
                .map(|(i, _)| i)
                .collect();
            assert!(
                held.windows(2).all(|w| w[1] == w[0] + 1),
                "{name}: region {r}'s steps are consecutive"
            );
            held[0]..held[held.len() - 1] + 1
        })
        .collect();
    let mut runs = 0;
    for (r, region) in mk.regions.iter().enumerate() {
        let mut at = steps_of[r].start;
        for u in region.units {
            assert!(
                u.first == at && u.end > u.first,
                "{name}: region {r}'s units tile its steps"
            );
            at = u.end;
        }
        assert_eq!(
            at, steps_of[r].end,
            "{name}: region {r}'s units cover its steps"
        );
        // Every unit opens the run of itself: whatever the load picks, a tiling exists.
        for u in 0..region.units.len() as u32 {
            assert!(
                region
                    .runs
                    .iter()
                    .any(|run| run.first == u && run.end == u + 1),
                "{name}: unit {u} of region {r} opens no run of itself"
            );
        }
        for run in region.runs {
            runs += 1;
            let text = kernel_text(mk, run.kernel);
            // No kernel synchronizes across threadgroups: no atomic, no device-scope fence,
            // nothing that spins on another threadgroup's writes.
            for forbidden in ["atomic", "memory_order", "thread_scope_device", "while"] {
                assert!(
                    !text.contains(forbidden),
                    "{name}: {} holds `{forbidden}`",
                    run.kernel
                );
            }
            assert_eq!(
                run.places.len() as u32,
                run.end - run.first,
                "{name}: {} places each of its units",
                run.kernel
            );
            // Each step of its units is spelled once in its kernel.
            for u in &region.units[run.first as usize..run.end as usize] {
                for s in &mk.steps[u.first as usize..u.end as usize] {
                    let spelled = format!("(baked {})", s.baked);
                    assert_eq!(
                        text.matches(&spelled).count(),
                        1,
                        "{name}: {} spells baked step {} once",
                        run.kernel,
                        s.baked
                    );
                }
            }
            // THE RUNS HOLD NO CROSS-THREADGROUP WAIT: a unit waits inside its run only on a
            // unit of its own lane group, or on one every threadgroup plays for itself.
            for &(u, w) in run.waits {
                assert!(
                    (run.first..run.end).contains(&u) && (run.first..u).contains(&w),
                    "{name}: {} holds a wait outside it",
                    run.kernel
                );
                let place = |x: u32| run.places[(x - run.first) as usize];
                let ordered = match (place(u), place(w)) {
                    (_, MkPlace::Everywhere) => true,
                    (MkPlace::Lane(a), MkPlace::Lane(b)) => a == b,
                    _ => false,
                };
                assert!(
                    ordered,
                    "{name}: in {}, unit {u} ({:?}) waits on unit {w} ({:?}), which no one \
                     threadgroup orders",
                    run.kernel,
                    place(u),
                    place(w)
                );
            }
        }
    }
    // Every admitted command is a step of the region instance it falls in, in order.
    let mut instances = vec![0u32; mk.regions.len()];
    let mut open: Option<(usize, u32)> = None;
    for &i in &admitted {
        let baked = origins[i].baked as u32;
        if let Some(r) = mk.regions.iter().position(|reg| reg.opens == baked) {
            if let Some((p, held)) = open {
                assert_eq!(
                    held,
                    steps_of[p].end - steps_of[p].start,
                    "{name}: region {p} played whole"
                );
            }
            instances[r] += 1;
            open = Some((r, 0));
        }
        let (r, held) = open.as_mut().expect("the forward opens with a region");
        let s = &mk.steps[(steps_of[*r].start + *held) as usize];
        assert_eq!(
            (s.baked, s.function),
            (baked, commands[i].command.function),
            "{name}: admitted command {i} is its region's next step"
        );
        *held += 1;
        let need = commands[i]
            .command
            .bindings
            .iter()
            .map(|b| u32::from(b.binding_index()) + 1);
        assert!(
            need.max().unwrap_or(0) <= s.row_len,
            "{name}: a binding past its address row"
        );
    }
    // The blocks the region instances bind are disjoint and inside the table, each step's row
    // inside its block.
    let mut blocks: Vec<(u32, u32)> = (mk.regions.iter().zip(&instances))
        .map(|(reg, &n)| (reg.table_at, reg.table_at + n * reg.block_len))
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
    for s in mk.steps {
        assert!(
            s.row_at + s.row_len <= mk.regions[s.region as usize].block_len,
            "{name}: baked step {}'s row past its block",
            s.baked
        );
    }
    for (k, l) in mk.load_constants.iter().enumerate() {
        assert!(
            mk.steps.iter().any(|s| s.baked == l.baked),
            "{name}: a load constant of no step"
        );
        let decl = format!("MK_LOAD_{k} [[function_constant({})]]", l.index.get());
        assert!(mk.source.contains(&decl), "{name}: {decl} not declared");
    }
    let fences = classed.tape.barriers_expanded();
    let dispatch_barriers = admitted.iter().filter(|&&i| fences[i]).count();
    println!(
        "{name}: {} commands, {} run at decode (dispatch: {dispatch_barriers} barriers), in {} \
         regions ({} instances per forward) offering {runs} runs a launch may play; {} load \
         constants; {} bytes of MSL",
        commands.len(),
        admitted.len(),
        mk.regions.len(),
        instances.iter().sum::<u32>(),
        mk.load_constants.len(),
        mk.source.len(),
    );
    if printed.insert(mk.library) {
        for (r, region) in mk.regions.iter().enumerate() {
            let units: Vec<String> = (region.units.iter())
                .map(|u| {
                    let names: Vec<&str> = (u.first..u.end)
                        .map(|s| mk.steps[s as usize].function)
                        .collect();
                    names.join(" > ")
                })
                .collect();
            println!(
                "  region {r} (x{}): {} units, {} runs: {}",
                instances[r],
                region.units.len(),
                region.runs.len(),
                units.join(" | ")
            );
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
