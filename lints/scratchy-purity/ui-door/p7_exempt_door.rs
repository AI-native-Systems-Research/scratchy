// Built as the crate `scratchy_forward_compiler`, so the door below is the item exceptions.toml names.

pub mod arch_registry {
    pub struct ArchRegistration {
        pub hf_arches: &'static [&'static str],
    }

    impl ArchRegistration {
        // Clean: the sanctioned door, the checkpoint-config to compiled-model lookup.
        pub fn claims(&self, arch_hint: &str) -> bool {
            arch_hint == "LlamaForCausalLM" || self.hf_arches.contains(&arch_hint)
        }

        // Flagged: a sibling of the door is no door.
        pub fn claims_all(&self, arch_hint: &str) -> bool {
            arch_hint == "LlamaForCausalLM"
        }
    }
}

fn main() {
    let row = arch_registry::ArchRegistration { hf_arches: &[] };
    let _ = (row.claims("x"), row.claims_all("x"));
}
