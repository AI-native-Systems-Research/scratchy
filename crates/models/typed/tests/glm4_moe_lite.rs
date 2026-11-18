// SPDX-License-Identifier: Apache-2.0
//! GLM-4.7-Flash's typed program: its carrier calls the multi-head latent attention decoder state's ops,
//! whose signatures state the latent attention's widths.

use scratchy_models_typed::traced;

/// GLM-4.7-Flash's latent attention, as its carrier's types state it: the attention is every
/// head's value, the compressed KV is `kv_lora_rank` wide, and the shared rope key one rope width.
#[test]
fn glm_attention_is_heads_of_values() {
    use scratchy_forward_compiler_frontend::dim::Dim;
    let shapes = traced("glm4_moe_lite").expect("GLM-4.7-Flash's typed program");
    let bound = |name: &str| Dim::Bound(name.to_string());
    let width = |local: &str| -> Vec<Dim> {
        let widths: Vec<Dim> = (shapes.program.iter().enumerate())
            .filter(|(_, name)| *name == local)
            .map(|(id, _)| {
                let id = scratchy_forward_compiler_frontend::classified::LocalId(id as u32);
                let shape = shapes.locals.get(&id).expect("a traced local");
                shape.last().expect("a feature dim").clone()
            })
            .collect();
        assert!(!widths.is_empty(), "no local `{local}`");
        widths
    };
    for w in width("attn") {
        assert_eq!(
            w,
            Dim::Mul(vec![bound("num_attention_heads"), bound("v_head_dim")])
        );
    }
    for w in width("kv_latent") {
        assert_eq!(w, bound("kv_lora_rank"));
    }
    for w in width("k_pe") {
        assert_eq!(w, bound("qk_rope_head_dim"));
    }
}
