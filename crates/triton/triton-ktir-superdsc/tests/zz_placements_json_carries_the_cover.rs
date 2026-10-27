//! `write_placements` carries the write-cover — the bake's own statement, not the launcher's
//! default.
//!
//! The launcher (`bundle_run`) reads placements.json through `PlaceRow`, whose `cover` field
//! deserializes with `BindCover`'s own derived `Deserialize`. A bake that omitted the field
//! would leave the launcher's conservative `Full` default to speak for the fold's per-pass
//! tensors — so the FILE must carry the class the layout minted. This proves it does, by
//! reading the file back through that same deserializer and comparing against the layout.

use ktir_superdsc::placement::{BindCover, BundleLayout, SegRole, TensorPlacement};

#[test]
fn the_placements_file_carries_the_bake_s_own_cover() {
    let mut l = BundleLayout::default();
    for (tid, size, cover) in [
        (11u32, 4096u64, BindCover::Full),
        (12u32, 4096u64, BindCover::FoldBlocks { block_bytes: 1024 }),
    ] {
        l.ids
            .borrow_mut()
            .insert(format!("t{tid}"), ktir_superdsc::place::PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: 0,
                size,
                cover,
            },
        );
    }
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zz_placements_cover");
    std::fs::create_dir_all(&dir).unwrap();
    triton_ktir_superdsc::bake::write_placements(&dir, &l).unwrap();
    let text = std::fs::read_to_string(dir.join("placements.json")).unwrap();
    let file: serde_json::Value = serde_json::from_str(&text).unwrap();
    let places = file["places"]
        .as_array()
        .expect("the file states its places");
    assert_eq!(places.len(), 2, "every placed tensor is stated");
    let mut read_back: Vec<(u64, BindCover)> = places
        .iter()
        .map(|p| {
            let cover: BindCover = serde_json::from_value(p["cover"].clone())
                .unwrap_or_else(|e| panic!("cover does not survive its own deserializer: {e}"));
            (p["size"].as_u64().unwrap(), cover)
        })
        .collect();
    read_back.sort();
    assert_eq!(
        read_back,
        vec![
            (4096, BindCover::Full),
            (4096, BindCover::FoldBlocks { block_bytes: 1024 }),
        ],
        "the file carries the bake's own classes, both of them"
    );
    // THE FOLD'S ARITHMETIC, as the launcher would read it: the stated block divides the
    // stated size — the audit's law, seen from the file the launcher actually reads.
    for (size, cover) in &read_back {
        if let BindCover::FoldBlocks { block_bytes } = cover {
            assert!(
                size % block_bytes == 0,
                "the stated block {block_bytes} divides the stated size {size}"
            );
        }
    }
}
