//! Regression guard for the SPANS BIT-31 CONTRACT (authoritative definition at
//! the flag-set: `crates/serving/worker/src/gpu_worker.rs`, `slot |= 0x8000_0000`).
//!
//! Spans (rope-on-read) OR bit 31 into `slot_mapping` / `block_table` entries to
//! mark relocatable blocks "stored unrotated". EVERY Metal kernel that uses one
//! of those tables AS AN INDEX must strip bit 31 (`& 0x7FFFFFFFu`, mirroring
//! attention's `ATTN_BT_MASK`) BEFORE dividing/indexing. An unmasked read
//! indexes ~2^31 elements out of bounds and silently corrupts the KV cache —
//! but ONLY when spans are active, so it sails through ordinary tests. This bug
//! has now bitten two kernels (the TurboQuant compress/dequant kernels and the
//! fused-QKV V-head); this test makes the next one a CI failure instead.
//!
//! Why a source scan, not a GPU test: the OOB only fires with spans enabled AND
//! a Metal device, neither present in normal CI. Per the project rule "every
//! runtime error needs a compile-time / automated check, not a manual GPU run,"
//! this reads the shader source as a string and runs on any host.
//!
//! It is a tripwire, not a proof: it asserts a mask token appears within a few
//! lines of each tagged-table read (matching the `slot_raw = slots[i]; ... slot
//! = slot_raw & 0x7FFFFFFFu;` idiom, where the raw value is needed for the
//! 0xFFFFFFFF padding check first), not that the masked value is the one used to
//! index. That is the ceiling for a per-shader textual scan.

/// Shaders that consume worker-tagged index tables and write/read the KV cache.
/// Add new such shaders here.
const SHADERS: &[(&str, &str)] = &[
    (
        "turboquant.metal",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/turboquant.metal"
        )),
    ),
    (
        "fused_qkv_rope_cache.metal",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/fused_qkv_rope_cache.metal"
        )),
    ),
    (
        "fused_affine_qkv_rope_cache.metal",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/fused_affine_qkv_rope_cache.metal"
        )),
    ),
];

/// Spellings that count as "bit 31 stripped".
const MASK_TOKENS: &[&str] = &["0x7FFFFFFF", "ATTN_BT_MASK"];

/// Worker-provided, bit-31-tagged index arrays.
const TAGGED_TABLES: &[&str] = &["block_table[", "slot_mapping[", "slots[", "logical_slots["];

/// The mask may legitimately appear a few lines after the raw read (the
/// read-raw-then-mask idiom, possibly with a stern comment in between). A fixed
/// window keeps it honest.
const MASK_WINDOW: usize = 12;

/// True only if the mask appears in CODE — not in a comment. Critical: the stern
/// warnings near these reads themselves mention `0x7FFFFFFF`, so counting
/// comment lines would let a future edit delete the real code mask and still
/// pass (the warning would mask the regression). The mask must be in code.
fn is_masked(line: &str) -> bool {
    if line.trim_start().starts_with("//") {
        return false;
    }
    MASK_TOKENS.iter().any(|m| line.contains(m))
}

/// Host-generated window indices (TurboQuant `WindowMap`) are NOT worker-tagged
/// and must NOT be masked (masking would corrupt them). They are annotated.
fn is_host_window_index(line: &str) -> bool {
    line.contains("SOURCE:") || line.contains("TARGET:") || line.contains("dst_slots[")
}

#[test]
fn kv_kernels_strip_bit31_from_tagged_index_tables() {
    let mut violations: Vec<String> = Vec::new();
    let mut total_masks = 0usize;

    for (name, src) in SHADERS {
        total_masks += src.matches("0x7FFFFFFF").count();
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !TAGGED_TABLES.iter().any(|t| line.contains(t)) {
                continue;
            }
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || line.contains("[[buffer(") {
                continue; // comment or buffer declaration, not an index use
            }
            if is_host_window_index(line) {
                continue; // deliberately-untagged host index
            }
            let end = (i + MASK_WINDOW + 1).min(lines.len());
            if !lines[i..end].iter().any(|l| is_masked(l)) {
                violations.push(format!("  {}:{}: {}", name, i + 1, line.trim()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "SPANS BIT-31 CONTRACT VIOLATION: a KV kernel reads a worker-tagged index \
         table (block_table / slot_mapping / slots / logical_slots) WITHOUT \
         stripping bit 31 (`& 0x7FFFFFFFu`, like attention's ATTN_BT_MASK) within \
         {MASK_WINDOW} lines. Spans (rope-on-read) set bit 31 on these tables; an \
         unmasked read is OOB and silently corrupts the KV cache. Mask it before \
         indexing — or, if this is a host-generated/untagged index, annotate it \
         SOURCE:/TARGET:. See the contract at gpu_worker.rs (slot |= 0x8000_0000).\n\
         Offending:\n{}",
        violations.join("\n")
    );

    // Anti-vacuity: the guard must actually be seeing masked tagged reads, so a
    // future refactor that removes/renames the tagged reads can't make this test
    // pass by having nothing to check. turboquant.metal alone strips 4 (slots,
    // logical_slots, 2x block_table); the fused-QKV K+V heads add more.
    assert!(
        total_masks >= 6,
        "expected >= 6 bit-31 strips across the KV kernels; found {total_masks}. \
         Did a tagged-table read get removed/renamed without updating this guard?"
    );
}
