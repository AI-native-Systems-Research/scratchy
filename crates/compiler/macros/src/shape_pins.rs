// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE SHAPE RATCHET, PER MODEL (#1094): the shape doors each model's expansion leaves open,
//! pinned, and held to the pin by the expansion itself — so every build of a model checks its
//! row, CI's full-scope clippy every CI-scope model's. A change that types one lowers its model's
//! count and must lower the pin with it; one that opens a door fails the build. A model no row
//! pins (a preset or head no pinned scope compiles) holds the doors closed at zero — every weight
//! family typed, every affine matmul over a typed slot — and leaves its dims asserts, Phase F's
//! open door, unheld. The source side (asserts and doors in the passes themselves) is the metal
//! crate's `shape_ratchet` test.

use proc_macro2::TokenStream;
use quote::quote;

/// `(arch::model, untyped weight families, untyped affine matmul commands, n/k dims asserts)`:
/// a model's weight families with no typed handle, its affine matmul commands whose weight is no
/// typed slot's, and its weights' n/k (and dense GEMM dims) tied to the bake's numbers by a
/// `const` assert rather than type equality (Phase F lowers these to zero).
const PINS: &[(&str, u32, u32, u32)] = &[
    ("commandr::c4ai_command_r_v01_parity_tiny", 0, 0, 6),
    ("commandr::c4ai_command_r_v01", 0, 0, 6),
    ("commandr::command_r_1_layer", 0, 0, 6),
    ("dflash2::qwen3_6_35b_a3b_dflash2", 0, 0, 9),
    (
        "dflash2::qwen3_6_35b_a3b_dflash2_mlx_affine_b4_g64_qembed",
        0,
        0,
        11,
    ),
    ("dflash2::qwen3_8_27b_dflash2", 0, 0, 9),
    ("gemma2::gemma2_27b_parity_tiny", 0, 0, 6),
    ("gemma2::gemma2_27b", 0, 0, 6),
    ("gemma2::gemma2_2b", 0, 0, 6),
    ("gemma2::gemma2_9b", 0, 0, 6),
    ("gemma3::gemma_3_12b_it_parity_tiny", 0, 0, 6),
    ("gemma3::gemma_3_12b_it", 0, 0, 6),
    ("gemma3::gemma_3_1b_it", 0, 0, 6),
    ("gemma3::gemma_3_27b_it", 0, 0, 6),
    ("gemma3::gemma_3_4b_it", 0, 0, 6),
    ("gemma4_assistant::gemma_4_26b_a4b_it_assistant", 0, 0, 8),
    (
        "gemma4_assistant::gemma_4_26b_a4b_it_qat_q4_0_unquantized_assistant",
        0,
        0,
        8,
    ),
    ("gemma4_moe::gemma_4_26b_a4b_it_parity_tiny", 0, 0, 9),
    ("gemma4_moe::gemma_4_26b_a4b_it", 0, 0, 9),
    ("gemma4::gemma_4_12b_it_parity_tiny_allglobal", 0, 0, 5),
    ("gemma4::gemma_4_12b_it_parity_tiny", 0, 0, 9),
    ("gemma4::gemma_4_12b_it", 0, 0, 9),
    ("gemma4::gemma_4_31b_it", 0, 0, 9),
    ("glm4_moe::glm_4_5_air_parity_tiny", 0, 0, 11),
    ("glm4_moe::glm_4_5_air", 0, 0, 11),
    ("gpt_oss::gpt_oss_120b_3bit", 0, 0, 5),
    ("gpt_oss::gpt_oss_120b_parity_tiny", 0, 0, 5),
    ("gpt_oss::gpt_oss_120b", 0, 0, 5),
    (
        "gpt_oss::gpt_oss_20b_mlx_affine_b2_g64_attn4_router8_qembed",
        0,
        0,
        5,
    ),
    ("gpt_oss::gpt_oss_20b_mlx_affine_b2_g64_qembed", 0, 0, 5),
    (
        "gpt_oss::gpt_oss_20b_mlx_affine_b3_g64_attn4_router8_qembed",
        0,
        0,
        5,
    ),
    ("gpt_oss::gpt_oss_20b_mlx_affine_b4_g64_qembed", 0, 0, 5),
    ("gpt_oss::gpt_oss_20b", 0, 0, 5),
    ("granite::granite_3_0_2b_base_parity_tiny", 0, 0, 6),
    ("granite::granite_3_0_2b_base", 0, 0, 6),
    ("granite::granite_3_0_2b_instruct", 0, 0, 6),
    ("granite::granite_3_0_8b_base", 0, 0, 6),
    ("granite::granite_3_0_8b_instruct", 0, 0, 6),
    ("granite::granite_3_1_2b_base", 0, 0, 6),
    ("granite::granite_3_1_2b_instruct", 0, 0, 6),
    ("granite::granite_3_1_8b_base", 0, 0, 6),
    ("granite::granite_3_1_8b_instruct", 0, 0, 6),
    ("granite::granite_3_2_2b_instruct", 0, 0, 6),
    ("granite::granite_3_2_8b_instruct", 0, 0, 6),
    ("granite::granite_3_3_2b_base", 0, 0, 6),
    ("granite::granite_3_3_2b_instruct", 0, 0, 6),
    ("granite::granite_3_3_8b_base", 0, 0, 6),
    ("granite::granite_3_3_8b_instruct", 0, 0, 6),
    ("granite::granite_4_1_30b", 0, 0, 6),
    ("granite::granite_4_1_3b", 0, 0, 6),
    ("granite::granite_4_1_8b", 0, 0, 6),
    ("granite::granite_4_2_30b", 0, 0, 6),
    ("granite::granite_4_2_3b_mlx_affine_b4_g32", 0, 0, 8),
    ("granite::granite_4_2_3b_mlx_affine_b4_g64_qembed", 0, 0, 8),
    ("granite::granite_4_2_3b_mlx_affine_b4_g64", 0, 0, 8),
    ("granite::granite_4_2_3b", 0, 0, 6),
    ("granite::granite_4_2_8b", 0, 0, 6),
    ("granite::micro_g3_3_8b_instruct_1b", 0, 0, 6),
    ("llama::llama_2_13b_parity_tiny", 0, 0, 6),
    ("llama::llama_2_13b", 0, 0, 6),
    ("llama::llama_2_7b", 0, 0, 6),
    ("llama::llama_3_1_70b", 0, 0, 6),
    ("llama::llama_3_1_8b", 0, 0, 6),
    ("llama::llama_3_2_1b_mlx_affine_b4_g64_qembed", 0, 0, 8),
    ("llama::llama_3_2_1b_mlx_affine_b4_g64", 0, 0, 8),
    ("llama::llama_3_2_1b", 0, 0, 6),
    ("llama::llama_3_2_3b", 0, 0, 6),
    ("llama::llama_3_70b", 0, 0, 6),
    ("llama::llama_3_8b", 0, 0, 6),
    ("llama::smollm2_135m_mlx_affine_b4_g64_qembed", 0, 0, 8),
    ("llama::smollm2_135m_mlx_affine_b4_g64", 0, 0, 8),
    ("llama::smollm2_135m", 0, 0, 6),
    ("llama::smollm2_360m", 0, 0, 6),
    ("llama::tinyllama_1_1b", 0, 0, 6),
    ("locateanything::locateanything_3b_parity_tiny", 4, 0, 9),
    ("locateanything::locateanything_3b", 4, 0, 9),
    ("mistral::mistral_7b_instruct_v0_3_parity_tiny", 0, 0, 6),
    ("mistral::mistral_7b_instruct_v0_3", 0, 0, 6),
    ("mistral::mistral_nemo_instruct_2407", 0, 0, 6),
    ("mixtral::mixtral_1_layer_parity_tiny", 0, 0, 0),
    ("mixtral::mixtral_1_layer", 0, 0, 0),
    ("mixtral::mixtral_8x7b_instruct_v0_1", 0, 0, 0),
    ("modernbert::modernbert_base_1_layer_parity_tiny", 0, 0, 5),
    ("modernbert::modernbert_base_1_layer", 0, 0, 5),
    ("modernbert::modernbert_base", 0, 0, 5),
    (
        "nemotron_h_latent::nemotron_3_super_120b_a12b_parity_tiny",
        0,
        0,
        11,
    ),
    ("nemotron_h_latent::nemotron_3_super_120b_a12b", 0, 0, 0),
    ("nemotron_h_mtp::nemotron_3_super_120b_a12b", 0, 0, 0),
    (
        "nemotron_h::nemotron_3_nano_30b_a3b_mamba_parity_tiny",
        0,
        0,
        7,
    ),
    (
        "nemotron_h::nemotron_3_nano_30b_a3b_mlx_affine_b4_g64_qembed",
        0,
        0,
        9,
    ),
    ("nemotron_h::nemotron_3_nano_30b_a3b_parity_tiny", 0, 0, 9),
    ("nemotron_h::nemotron_3_nano_30b_a3b", 0, 0, 0),
    ("phi3::phi_3_5_mini_instruct_parity_tiny", 0, 0, 6),
    ("phi3::phi_3_5_mini_instruct", 0, 0, 6),
    ("phi3::phi_3_medium_128k_instruct", 0, 0, 6),
    ("phi3::phi_3_medium_4k_instruct", 0, 0, 6),
    ("phi3::phi_3_mini_128k_instruct", 0, 0, 6),
    ("phi3::phi_3_mini_4k_instruct", 0, 0, 6),
    ("phi3::phi_4_mini_instruct", 0, 0, 6),
    ("phi3::phi_4_mini_reasoning", 0, 0, 6),
    ("phi3::phi_4_reasoning", 0, 0, 6),
    ("phi3::phi_4", 0, 0, 6),
    ("qwen2_5_vl::qwen2_5_vl_3b_instruct_parity_tiny", 1, 0, 6),
    ("qwen2_5_vl::qwen2_5_vl_3b_instruct", 1, 0, 6),
    ("qwen2_5_vl::qwen2_5_vl_3b_mlx", 1, 0, 6),
    ("qwen2_5_vl::qwen2_5_vl_72b_instruct", 1, 0, 6),
    ("qwen2_5_vl::qwen2_5_vl_7b_instruct", 1, 0, 6),
    ("qwen2_moe::slimed_qwen_3_parity_tiny", 0, 0, 0),
    ("qwen2_moe::slimed_qwen_3", 0, 0, 0),
    ("qwen2_vl::qwen2_vl_2b_instruct_parity_tiny", 4, 0, 7),
    ("qwen2_vl::qwen2_vl_2b_instruct", 4, 0, 7),
    ("qwen2_vl::qwen2_vl_2b_mlx", 4, 0, 7),
    ("qwen2_vl::qwen2_vl_72b_instruct", 4, 0, 7),
    ("qwen2_vl::qwen2_vl_7b_instruct", 4, 0, 7),
    ("qwen2::locateanything_3b", 0, 0, 6),
    ("qwen2::qwen2_0_5b", 0, 0, 6),
    ("qwen2::qwen2_1_5b", 0, 0, 6),
    ("qwen2::qwen2_5_0_5b", 0, 0, 6),
    ("qwen2::qwen2_5_1_5b", 0, 0, 6),
    ("qwen2::qwen2_5_14b", 0, 0, 6),
    ("qwen2::qwen2_5_32b", 0, 0, 6),
    ("qwen2::qwen2_5_3b", 0, 0, 6),
    ("qwen2::qwen2_5_7b", 0, 0, 6),
    ("qwen2::qwen2_5_vl_3b_mlx", 0, 0, 6),
    ("qwen2::qwen2_5_vl_3b", 0, 0, 6),
    ("qwen2::qwen2_72b", 0, 0, 6),
    ("qwen2::qwen2_7b_parity_tiny", 0, 0, 6),
    ("qwen2::qwen2_7b", 0, 0, 6),
    ("qwen2::qwen2_vl_2b_mlx", 0, 0, 6),
    ("qwen2::qwen2_vl_2b", 0, 0, 6),
    ("qwen3_5_dense_mtp::qwen3_5_0_8b_parity_tiny", 0, 0, 7),
    ("qwen3_5_dense_mtp::qwen3_5_0_8b", 0, 0, 7),
    ("qwen3_5_dense_mtp::qwen3_5_9b", 0, 0, 7),
    ("qwen3_5_dense_mtp::qwen3_6_27b", 0, 0, 7),
    ("qwen3_5_moe::qwen3_5_122b_a10b", 0, 0, 0),
    ("qwen3_5_moe::qwen3_5_35b_a3b_parity_tiny", 0, 0, 0),
    ("qwen3_5_moe::qwen3_5_35b_a3b", 0, 0, 0),
    (
        "qwen3_5_moe::qwen3_6_35b_a3b_mlx_affine_b4_g64_qembed",
        0,
        0,
        14,
    ),
    ("qwen3_5_moe::qwen3_6_35b_a3b", 0, 0, 0),
    (
        "qwen3_5_mtp::qwen3_6_35b_a3b_mlx_affine_b4_g64_qembed",
        0,
        0,
        10,
    ),
    ("qwen3_5_mtp::qwen3_5_122b_a10b", 0, 0, 0),
    ("qwen3_5_mtp::qwen3_5_35b_a3b_parity_tiny", 0, 0, 0),
    ("qwen3_5_mtp::qwen3_5_35b_a3b", 0, 0, 0),
    ("qwen3_5_mtp::qwen3_6_35b_a3b", 0, 0, 0),
    ("qwen3_5_vl::qwen3_5_9b_parity_tiny", 3, 0, 9),
    ("qwen3_5_vl::qwen3_5_9b", 3, 0, 9),
    ("qwen3_5_vl::qwen3_6_27b", 3, 0, 9),
    ("qwen3_5::qwen3_5_0_8b_parity_tiny", 0, 0, 7),
    ("qwen3_5::qwen3_5_0_8b", 0, 0, 11),
    ("qwen3_5::qwen3_5_9b", 0, 0, 11),
    ("qwen3_5::qwen3_6_27b", 0, 0, 11),
    ("qwen3_5::qwen3_8_27b", 0, 0, 11),
    (
        "qwen3_moe::qwen3_30b_a3b_instruct_2507_parity_tiny",
        0,
        0,
        0,
    ),
    ("qwen3_moe::qwen3_30b_a3b_instruct_2507", 0, 0, 0),
    ("qwen3_moe::qwen3_moe_1_layer", 0, 0, 0),
    ("qwen3::qwen3_0_6b_parity_tiny", 0, 0, 6),
    ("qwen3::qwen3_0_6b", 0, 0, 6),
    ("qwen3::qwen3_1_7b", 0, 0, 6),
    ("qwen3::qwen3_4b", 0, 0, 6),
    ("qwen3::qwen3_8b", 0, 0, 6),
];

/// A model's open shape doors, as its expansion counted them: a [`PINS`] row's numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Doors {
    pub untyped_weight_families: u32,
    pub untyped_affine_commands: u32,
    pub dims_asserts: u32,
}

/// `model`'s doors held to its pin: nothing when they are at it, a `compile_error!` naming the
/// model, each door off it and its row otherwise (and for a model not pinned).
pub(crate) fn check(model: &str, doors: Doors) -> TokenStream {
    let Doors {
        untyped_weight_families: untyped,
        untyped_affine_commands: affine,
        dims_asserts: asserts,
    } = doors;
    let row = format!("(\"{model}\", {untyped}, {affine}, {asserts}),");
    // An unpinned model is held to the closed doors, its dims asserts to themselves.
    let (pu, pa, pd) = match PINS.iter().find(|p| p.0 == model) {
        Some(&(_, pu, pa, pd)) => (pu, pa, pd),
        None => (0, 0, asserts),
    };
    let doors = [
        ("untyped weight families", untyped, pu),
        ("untyped affine matmul commands", affine, pa),
        ("n/k dims asserts", asserts, pd),
    ];
    let off: Vec<String> = (doors.iter())
        .filter(|(_, at, pin)| at != pin)
        .map(|(door, at, pin)| {
            let why = match at > pin {
                true => "a door was opened",
                false => "lower the pin",
            };
            format!("{door} {at}, pinned at {pin} ({why})")
        })
        .collect();
    if off.is_empty() {
        return TokenStream::new();
    }
    let msg = format!(
        "shape ratchet (#1094): `{model}`: {}; its row: {row}",
        off.join("; ")
    );
    quote!(::core::compile_error!(#msg);)
}
