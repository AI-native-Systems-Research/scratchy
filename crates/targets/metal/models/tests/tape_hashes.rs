// SPDX-License-Identifier: Apache-2.0
//! Each baked model's resolved tape, pinned by its hash: every bucket's every rung's commands, read
//! out of its command table, hashed (sha256 of their serialized values) into one line per model in
//! `tape_hashes.txt`. A change that moves a tape fails here, naming each model whose hash moved.
//!
//! The golden is the test's own: on a mismatch it writes `tape_hashes.txt.new` beside it, with
//! every model's line as this build hashes it, and fails. A change that means to move a tape
//! renames `.new` over the golden; nothing edits it by hand.
//!
//! The models are [`scratchy_target_metal_models::METAL_MODELS`]: every model the build bakes a metal tape for,
//! listed by its build script. The golden pins the scope CI runs it at (the metal-tests job's
//! serving-layer step); at any other scope, a model it doesn't pin fails the same way a moved one
//! does.

use scratchy_target_metal::interpreter::metal::MetalBucketSpec;
use scratchy_target_metal_models::METAL_MODELS;
use sha2::{Digest, Sha256};

/// `buckets`' resolved commands, every bucket's every rung in order, hashed.
fn tape_hash(buckets: &[MetalBucketSpec]) -> String {
    let mut h = Sha256::new();
    for spec in buckets {
        h.update(spec.bucket_m.to_le_bytes());
        for classed in spec.tapes {
            let commands = classed.tape.commands_expanded();
            let json = serde_json::to_vec(&commands).expect("a resolved command serializes");
            h.update((json.len() as u64).to_le_bytes());
            h.update(&json);
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_baked_models_resolved_tape_is_its_pinned_hash() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tape_hashes.txt");
    let pinned = std::fs::read_to_string(&path).unwrap_or_default();
    let pin = |model: &str| {
        (pinned.lines())
            .filter_map(|l| l.split_once("  "))
            .find(|(m, _)| *m == model)
            .map(|(_, h)| h.trim().to_string())
    };
    let rows: Vec<(&str, String)> = (METAL_MODELS.iter())
        .map(|(model, buckets)| (*model, tape_hash(buckets)))
        .collect();
    let moved: Vec<&str> = (rows.iter())
        .filter(|(model, hash)| pin(model).as_deref() != Some(hash.as_str()))
        .map(|(model, _)| *model)
        .collect();
    if !moved.is_empty() {
        let new = (rows.iter())
            .map(|(model, hash)| format!("{model}  {hash}\n"))
            .collect::<String>();
        std::fs::write(path.with_extension("txt.new"), new).expect("write tape_hashes.txt.new");
        panic!(
            "the resolved tapes of {moved:?} moved; `tape_hashes.txt.new` holds this build's \
             hashes"
        );
    }
}
