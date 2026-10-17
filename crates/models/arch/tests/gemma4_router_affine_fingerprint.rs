// SPDX-License-Identifier: Apache-2.0
//! The 26b router's fingerprint rows follow the CONFIG, not a default.
//!
//! `FieldLoad::GemmaRouter` resolves `router.proj` through
//! `affine_role_widths` — the same resolution the fingerprint's affine
//! bit map is built from — so a variant whose merged config leaves the router
//! dense (the fp8 26b's drift file lists `router.proj` in the
//! checkpoint-style `ignore`) lists NO affine row for it and emits NO
//! positive sniff, while the MLX variant keeps its `router.proj` row at the
//! drift-declared 8 bits over the 4-bit section default. The fabrication
//! this replaces (an `unwrap_or(8)` width for every layout) put a row in
//! the fp8 variant's bit map that demanded `.scales` from checkpoints
//! shipping a dense bf16 router — the only cached 26b FP8 weights among
//! them — and try_load refused them all ("no compiled model variant
//! matches this checkpoint").
//!
//! This pins the WIRING end-to-end — config + preset + per-(stem, preset)
//! overrides merged, forward-expanded, emitted — by reading the emission
//! this very build produced (`scratchy-forwards.rs` writes
//! `$OUT_DIR/gemma4_moe.rs`, which `src/lib.rs` includes). Each leg asserts
//! only the variant modules this build actually carries, but at least one
//! 26b variant must be present: a stem-feature build with every variant
//! missing is a build bug, not a skip.
#![cfg(feature = "gemma-4-26b-a4b-it")]

/// The `gemma4_moe.rs` emission this build produced.
fn emission() -> String {
    std::fs::read_to_string(concat!(env!("OUT_DIR"), "/gemma4_moe.rs"))
        .expect("scratchy-forwards.rs writes $OUT_DIR/gemma4_moe.rs")
}

/// One variant module's body: from its `pub mod <name> {` to the next
/// `pub mod` (the emission is a flat list of variant modules; the trailing
/// `{` keeps `gemma_4_26b_a4b_it` from matching its preset-suffixed
/// siblings). `None` when this build's quant scope left the variant out.
fn variant_module<'a>(src: &'a str, variant: &str) -> Option<&'a str> {
    let head = format!("pub mod {variant} {{");
    let start = src.find(&head)? + head.len();
    let body = &src[start..];
    Some(match body.find("pub mod ") {
        Some(next) => &body[..next],
        None => body,
    })
}

#[test]
fn router_affine_rows_follow_the_config_not_a_default() {
    let src = emission();

    // The fp8 variant (quant/fp8-dynamic-per-channel in scope): the drift
    // file's `ignore` leaves the router dense, so NO affine bit map at all —
    // neither the `__AFFINE_WIDTHS` const nor the `ships_affine_row` sniff
    // may appear anywhere in the module. (Pre-fix both appear, fabricated
    // from the `unwrap_or(8)` default.)
    if let Some(fp8) = variant_module(&src, "gemma_4_26b_a4b_it_fp8_dynamic_per_channel") {
        assert!(
            !fp8.contains("__AFFINE_WIDTHS"),
            "the fp8 26b variant lists affine rows but its config leaves every \
             affine role dense — a fabricated row demands `.scales` the \
             checkpoint never ships"
        );
        assert!(
            !fp8.contains("ships_affine_row"),
            "the fp8 26b variant emits the affine positive sniff with no affine \
             rows to sniff"
        );
    }

    // The MLX variant (quant/mlx-affine-b4-g64 in scope): the router stays
    // affine at the drift-declared width — the bit map keeps the
    // `router.proj` row the loader dequantizes at, 8 bits over the 4-bit
    // section default, group 64.
    if let Some(mlx) = variant_module(&src, "gemma_4_26b_a4b_it_mlx_affine_b4_g64") {
        assert!(
            mlx.contains("__AFFINE_WIDTHS"),
            "the MLX 26b variant lost its affine bit map"
        );
        assert!(
            mlx.contains("router.proj.weight"),
            "the MLX 26b variant's affine bit map dropped the router.proj row"
        );
        // The row itself, `("…router.proj.weight", 8, 64)`: bits then group,
        // matched on whitespace-collapsed tokens so pretty-printing cannot
        // dodge it. A resolver that misses the drift file's
        // `bits_overrides` answers the section default (4) here — the
        // emission-level form of the #202 disagreement.
        let toks: Vec<&str> = mlx.split_whitespace().collect();
        let row = toks
            .iter()
            .position(|t| *t == "\"language_model.model.layers.*.router.proj.weight\"")
            .unwrap_or_else(|| panic!("no router.proj row literal in the MLX module"));
        assert_eq!(
            toks[row + 2],
            "8",
            "the MLX router row's bits must be the drift-declared 8, not the section default"
        );
        assert_eq!(
            toks[row + 4],
            "64",
            "the MLX router row's group must be the section's 64"
        );
        // And the load call carries the same pair: the router dequant width
        // and the fingerprint row read ONE resolution.
        let collapsed = toks.join(" ");
        assert!(
            collapsed.contains("Some ( ( 64 , 8 ) )"),
            "the MLX router load must dequant at (group 64, bits 8)"
        );
    }

    // The default (no-preset) variant: a verbatim bf16 model has no affine
    // roles at all — the router's formerly-fabricated row is gone here too.
    if let Some(dense) = variant_module(&src, "gemma_4_26b_a4b_it") {
        assert!(
            !dense.contains("__AFFINE_WIDTHS"),
            "the unquantized 26b variant lists affine rows — verbatim bf16 has \
             none, so any row is fabricated"
        );
    }

    // At least one 26b variant must be in the emission: the stem feature is
    // on, so a build with every variant missing is a build bug, not a skip.
    assert!(
        variant_module(&src, "gemma_4_26b_a4b_it_fp8_dynamic_per_channel").is_some()
            || variant_module(&src, "gemma_4_26b_a4b_it_mlx_affine_b4_g64").is_some()
            || variant_module(&src, "gemma_4_26b_a4b_it").is_some(),
        "no gemma-4-26b-a4b-it variant module in the emission — the stem \
         feature is on, so the build dropped every variant"
    );
}
