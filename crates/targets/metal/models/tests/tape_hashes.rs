// SPDX-License-Identifier: Apache-2.0
//! Each baked model's resolved tape, pinned by its hash: every bucket's every rung's commands, read
//! out of its command table, hashed (sha256 of their serialized values) into one line per model in
//! `tape_hashes.txt`. A change that moves a tape fails here, naming each model whose hash moved.
//!
//! The golden is the test's own, held by [`scratchy_golden_hash::assert_pinned_hashes`]: on a
//! mismatch it writes `tape_hashes.txt.new` beside it, with this build's line for every model in
//! its scope and the golden's for every pinned model outside it, and fails. A change that means to
//! move a tape renames `.new` over the golden; nothing edits it by hand.
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
    let rows: Vec<(&str, String)> = (METAL_MODELS.iter())
        .map(|(model, buckets)| (*model, tape_hash(buckets)))
        .collect();
    scratchy_golden_hash::assert_pinned_hashes(
        env!("CARGO_MANIFEST_DIR"),
        "tape_hashes.txt",
        &rows,
    );
}
