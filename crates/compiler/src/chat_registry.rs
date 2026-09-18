// SPDX-License-Identifier: Apache-2.0
//! The scratchy-side registry for compiled chat templates.
//!
//! ⛔ WHY THIS LIVES HERE AND NOT IN THE COMPILER CRATE.
//! `scratchy-chat-template-compiler` is a leaf crate — no `scratchy-*`
//! dependency, publishable on its own. It therefore must not know about
//! `inventory`, models, arches or serving. It emits a plain
//! [`CompiledTemplate`] and stops. Pairing that with a registry is scratchy's
//! concern, so the registration type lives on this side of the boundary.
//!
//! This mirrors [`crate::arch_registry`]: the build script emits one
//! `inventory::submit!` per compiled asset, and the consumer iterates. Note the
//! registration carries no function pointer of its own and no `dyn` — the fn
//! pointer sits inside `CompiledTemplate`, and there is exactly one entry per
//! enabled model, read once at init and never per request.

use scratchy_chat_template_compiler::CompiledTemplate;

/// One compiled chat template, emitted per `(arch, stem)` that has a vendored
/// `<stem>.chat.jinja` and whose model feature is enabled.
pub struct ChatTemplateRegistration {
    /// Architecture directory the template was vendored under (`"llama"`).
    pub arch: &'static str,
    /// Model stem (`"smollm2-135m"`).
    pub stem: &'static str,
    /// The generated renderer plus the exact source it was generated from.
    pub compiled: &'static CompiledTemplate,
}

inventory::collect!(ChatTemplateRegistration);

/// Find a compiled renderer valid for `resolved`.
///
/// ⛔ THE LOOKUP *IS* THE DRIFT GATE. Matching is byte equality against the
/// template the renderer was generated from, so a compiled function can only
/// ever be used with its own source. If the served model ships a different
/// template — it was updated upstream, the operator passed `--chat-template`, or
/// it came from GGUF metadata rather than `tokenizer_config.json` — nothing
/// matches and the caller falls back to the interpreter.
///
/// That is deliberately conservative: rendering a prompt with the wrong
/// template produces confidently wrong output that nothing downstream can
/// detect, whereas falling back is merely slower.
pub fn find_compiled(resolved: &str) -> Option<&'static CompiledTemplate> {
    inventory::iter::<ChatTemplateRegistration>()
        .map(|r| r.compiled)
        .find(|c| c.matches(resolved))
}

/// Every compiled template this binary carries. For diagnostics — `scr model
/// info`-style reporting and the startup log line that says whether the
/// compiled path is in use.
pub fn compiled_templates() -> impl Iterator<Item = &'static ChatTemplateRegistration> {
    inventory::iter::<ChatTemplateRegistration>()
}
