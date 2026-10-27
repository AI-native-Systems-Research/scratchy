// SPDX-License-Identifier: Apache-2.0
//! ⭐ WHAT LETS ONE dxp COMPILE SERVE MANY GROUPS — the content key and the cache it indexes.
//!
//! A group's dxp program is a function of its ops IN ORDER, so two groups holding the same ops must
//! render to the same bytes no matter where in their bundle they start; and a compiled group kept
//! under that key must come back exactly, under the same compiler, and never under a different one.

use std::path::{Path, PathBuf};

use ktir_superdsc::emit;
use scratchy_subtile::sdsc_abstract::{BlockCols, KernelTag, RowCount, Stk};
use scratchy_target_spyre::dxp_cache::{DxpCache, DxpIdentity, GroupDigest};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
use scratchy_target_spyre::superdsc_bake::CompiledGroup;

/// `n` copies of one fitting matmul — one Pure trip each, so they fill groups back to back.
fn identical_matmuls(n: usize) -> Vec<ktir_superdsc::emit::EmittedOp> {
    (0..n)
        .map(|_| {
            superdsc::assemble_matmul(
                "rep_mm",
                64,
                64,
                64,
                1,
                &emit::rb(
                    "rep_a",
                    RowCount::of_token_rows(64),
                    BlockCols::of_feature_cols(64),
                ),
                &Stk::<KernelTag>::kernel(64_usize, 64_usize, "rep_w"),
                &emit::rb(
                    "rep_o",
                    RowCount::of_token_rows(64),
                    BlockCols::of_feature_cols(64),
                ),
                None,
            )
        })
        .collect()
}

/// The `<idx>` of a rendered op's top-level `"<idx>_<op>"` key.
fn op_index(json: &str) -> usize {
    let rest = json
        .strip_prefix("{\"")
        .expect("an op json opens with its key");
    rest[..rest.find('_').expect("the key is <idx>_<op>")]
        .parse()
        .expect("the key's index is a number")
}

/// ⛔ EVERY GROUP NUMBERS ITS OWN OPS FROM 0. dxp reads the index only to order a group's ops, so a
/// bundle-global index made two groups of identical ops render differently — and the content key,
/// which hashes the rendering, then compiled each one. Mutation: restore the global `{idx}` and the
/// second group starts at its bundle offset, failing both asserts.
#[test]
fn a_group_numbers_its_ops_from_zero_and_identical_groups_share_a_key() {
    let ops = identical_matmuls(3 * superdsc::group_size());
    let groups = superdsc::render_dxp_input(&ops, superdsc::FoldGrouping::Split).unwrap();
    assert!(groups.len() >= 3, "three groups' worth of trips");
    for g in &groups {
        let indices: Vec<usize> = g
            .files
            .iter()
            .filter(|(name, _)| name.starts_with("sdsc_"))
            .map(|(_, json)| op_index(json))
            .collect();
        let want: Vec<usize> = (0..indices.len()).collect();
        assert_eq!(indices, want, "group {} numbers its ops from 0", g.group);
    }
    let (a, b) = (&groups[0], &groups[1]);
    assert_eq!(
        a.files.len(),
        b.files.len(),
        "two full groups of the same op"
    );
    assert_eq!(
        a.key, b.key,
        "identical ops at different offsets are one compile"
    );
    assert_eq!(a.files, b.files, "and render to the same bytes");
}

/// ⛔ THE KEY FRAMES EVERY FILE. Moving bytes between two files, or renaming one, is different
/// compiler input, and must be a different key even when the concatenated bytes agree.
#[test]
fn the_group_digest_frames_names_and_bodies() {
    let f = |v: &[(&str, &str)]| -> Vec<(String, String)> {
        v.iter()
            .map(|(n, b)| ((*n).to_string(), (*b).to_string()))
            .collect()
    };
    let base = GroupDigest::of(&f(&[("a", "xy"), ("b", "z")]));
    assert_eq!(base, GroupDigest::of(&f(&[("a", "xy"), ("b", "z")])));
    assert_ne!(
        base,
        GroupDigest::of(&f(&[("a", "x"), ("b", "yz")])),
        "a moved byte"
    );
    assert_ne!(
        base,
        GroupDigest::of(&f(&[("a", "xy"), ("c", "z")])),
        "a renamed file"
    );
    assert_ne!(
        base,
        GroupDigest::of(&f(&[("b", "z"), ("a", "xy")])),
        "a reordered file"
    );
}

/// A scratch directory of this test's own, under cargo's per-target temp dir.
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A stand-in compiler: what these tests need is a binary the platform loader will TRACE. A system
/// binary will not do on macOS — dyld ignores `DYLD_*` for protected executables, even copied ones —
/// so the test builds a trivial one with the C compiler the spyre build already requires.
fn stand_in_compiler() -> PathBuf {
    static BIN: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dxp_stand_in");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("main.c");
        std::fs::write(&src, "int main(void) { return 0; }\n").unwrap();
        let bin = dir.join("dxp_stand_in");
        let ok = std::process::Command::new("cc")
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .status()
            .expect("cc runs")
            .success();
        assert!(ok, "cc built the stand-in");
        bin
    })
    .clone()
}

/// The stand-in's identity under the bake's fixed environment `env`.
fn identity(env: &[(&str, &str)]) -> DxpIdentity {
    DxpIdentity::probe(
        &stand_in_compiler(),
        Path::new(env!("CARGO_TARGET_TMPDIR")),
        env,
        &[],
    )
    .expect("the platform loader names the stand-in's images")
}

/// ⛔ A DIFFERENT COMMAND IS A DIFFERENT COMPILER. The environment the bake sets on dxp is part of
/// the key, so a change to it (a thread count, an output switch) can never be served a program the
/// old command produced.
#[test]
fn the_compiler_identity_follows_the_fixed_environment() {
    let a = identity(&[("DT_PARALLEL_THREADS", "1")]);
    assert_eq!(a, identity(&[("DT_PARALLEL_THREADS", "1")]));
    assert_ne!(a, identity(&[("DT_PARALLEL_THREADS", "2")]));
}

/// A compiled group with distinguishable bytes in every field.
fn compiled(seed: u8) -> CompiledGroup {
    CompiledGroup {
        init_binary: vec![seed, seed.wrapping_add(1), seed.wrapping_add(2)],
        job_bin_ptr: 0x1000 + u64::from(seed),
        correction: vec![seed.wrapping_mul(3)],
    }
}

/// ⭐ A STORED GROUP COMES BACK FIELD FOR FIELD, under the identity that stored it.
#[test]
fn a_stored_group_round_trips() {
    let base = scratch("dxp_cache_round_trip");
    let key = GroupDigest::of(&[("sdsc_0.json".to_string(), "{}".to_string())]);
    let cache = DxpCache::open(&base, identity(&[("DT_PARALLEL_THREADS", "1")])).unwrap();
    assert!(cache.get(key).is_none(), "nothing before the first put");
    cache.put(key, &compiled(7));
    let hit = cache.get(key).expect("the stored group");
    let want = compiled(7);
    assert_eq!(hit.init_binary, want.init_binary);
    assert_eq!(hit.job_bin_ptr, want.job_bin_ptr);
    assert_eq!(hit.correction, want.correction);
}

/// ⛔ ANOTHER COMPILER'S PROGRAM IS NEVER SERVED — AND IS NOT KEPT. Opening the store under a
/// different identity wipes the old one, so going back to the first compiler finds nothing.
#[test]
fn a_different_compiler_wipes_the_store() {
    let base = scratch("dxp_cache_wipe");
    let key = GroupDigest::of(&[("sdsc_0.json".to_string(), "x".to_string())]);
    let first = identity(&[("DT_PARALLEL_THREADS", "1")]);
    DxpCache::open(&base, first).unwrap().put(key, &compiled(1));
    assert!(DxpCache::open(&base, first).unwrap().get(key).is_some());
    let other = DxpCache::open(&base, identity(&[("DT_PARALLEL_THREADS", "2")])).unwrap();
    assert!(
        other.get(key).is_none(),
        "another compiler's program is never served"
    );
    assert!(
        DxpCache::open(&base, first).unwrap().get(key).is_none(),
        "the first compiler's store went with the switch"
    );
}

/// A truncated or foreign entry is a MISS, never a wrong program.
#[test]
fn a_damaged_entry_reads_as_a_miss() {
    let base = scratch("dxp_cache_damaged");
    let id = identity(&[]);
    let key = GroupDigest::of(&[("sdsc_0.json".to_string(), "d".to_string())]);
    let cache = DxpCache::open(&base, id).unwrap();
    cache.put(key, &compiled(9));
    let entry = base.join(id.hex()).join(format!("{}.bin", key.hex()));
    let bytes = std::fs::read(&entry).unwrap();
    std::fs::write(&entry, &bytes[..bytes.len() - 1]).unwrap();
    assert!(cache.get(key).is_none(), "a truncated entry");
    std::fs::write(
        &entry,
        b"not an entry at all, long enough to pass the header",
    )
    .unwrap();
    assert!(cache.get(key).is_none(), "a foreign file");
}
