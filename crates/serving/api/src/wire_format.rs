// SPDX-License-Identifier: Apache-2.0
//! The tool-call and reasoning wire formats a model speaks, read from its chat template.
//!
//! A model's template is what teaches it the shape of a tool call and of its thinking: the
//! markers it renders for an assistant turn are the markers the model emits. So the template the
//! engine loads decides which parser reads the output, one rule per marker, and a model with no
//! template gets no parser (an explicit `--tool-call-parser` or `--reasoning-parser` still sets
//! one).

/// The parsers a template's markers select, by the names [`crate::tool_parser::get_tool_parser`]
/// and [`crate::reasoning_parser::get_reasoning_parser`] take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WireFormats {
    /// The tool-call parser, when the template renders tool calls in a format one reads.
    pub tool: Option<&'static str>,
    /// The reasoning parser, when the template renders thinking in a format one reads.
    pub reasoning: Option<&'static str>,
}

/// A tool call opened by `<tool_call>` with a JSON object for its body, as the template
/// writes it inside a Jinja string (`\n{\"`) or as raw text.
fn json_tool_call(template: &str) -> bool {
    template.contains(r#"<tool_call>\n{\""#) || template.contains("<tool_call>\n{\"")
}

/// The tool-call format of a template.
fn tool(template: &str) -> Option<&'static str> {
    if template.contains("<|tool_call>") {
        // `<|tool_call>call:name{..}<tool_call|>`.
        Some("gemma4")
    } else if template.contains("<function=") {
        // `<tool_call>\n<function=name>\n<parameter=..>` XML calls.
        Some("qwen3_coder")
    } else if template.contains("[TOOL_CALLS]") {
        Some("mistral")
    } else if template.contains("<|start_of_role|>")
        && (template.contains("<|tool_call|>") || template.contains("<tool_call>"))
    {
        // Role-tagged turns, with calls after `<|tool_call|>` or `<tool_call>`.
        Some("granite")
    } else if template.contains("<|python_tag|>") {
        Some("llama3_json")
    } else if json_tool_call(template) {
        // `<tool_call>\n{"name": .., "arguments": ..}\n</tool_call>`.
        Some("hermes")
    } else {
        None
    }
}

/// The reasoning format of a template.
fn reasoning(template: &str) -> Option<&'static str> {
    if template.contains("<|channel|>") {
        // Harmony channels: `<|channel|>analysis<|message|>..`.
        Some("gpt_oss")
    } else if template.contains("<|channel>") {
        // Thought channels: `<|channel>thought ..<channel|>`.
        Some("gemma4")
    } else if template.contains("<think>") && template.contains("reasoning_content") {
        // A `<think>` block the template renders from an assistant turn's
        // `reasoning_content`. A template that only names `<think>` in a system prompt's
        // instructions renders no thinking of its own.
        Some("qwen3")
    } else {
        None
    }
}

/// The wire formats of the chat template the engine loaded, or none without one.
pub fn wire_formats(template: Option<&str>) -> WireFormats {
    template.map_or_else(WireFormats::default, |t| WireFormats {
        tool: tool(t),
        reasoning: reasoning(t),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each fixture is a marker snippet copied from a published chat template.
    fn fixture(name: &str) -> &'static str {
        match name {
            "gemma4" => include_str!("../tests/fixtures/wire_format/gemma4.jinja"),
            "qwen3_coder" => include_str!("../tests/fixtures/wire_format/qwen3_coder.jinja"),
            "mistral" => include_str!("../tests/fixtures/wire_format/mistral.jinja"),
            "granite_role_bar" => {
                include_str!("../tests/fixtures/wire_format/granite_role_bar.jinja")
            }
            "granite_role_xml" => {
                include_str!("../tests/fixtures/wire_format/granite_role_xml.jinja")
            }
            "llama3_json" => include_str!("../tests/fixtures/wire_format/llama3_json.jinja"),
            "hermes_think" => include_str!("../tests/fixtures/wire_format/hermes_think.jinja"),
            "hermes" => include_str!("../tests/fixtures/wire_format/hermes.jinja"),
            "harmony" => include_str!("../tests/fixtures/wire_format/harmony.jinja"),
            "xml_args_think" => include_str!("../tests/fixtures/wire_format/xml_args_think.jinja"),
            "think_instructions" => {
                include_str!("../tests/fixtures/wire_format/think_instructions.jinja")
            }
            "plain" => include_str!("../tests/fixtures/wire_format/plain.jinja"),
            _ => unreachable!("no fixture {name}"),
        }
    }

    fn formats(name: &str) -> (Option<&'static str>, Option<&'static str>) {
        let f = wire_formats(Some(fixture(name)));
        (f.tool, f.reasoning)
    }

    #[test]
    fn tool_call_channel_markers_select_gemma4_both_ways() {
        assert_eq!(formats("gemma4"), (Some("gemma4"), Some("gemma4")));
    }

    #[test]
    fn xml_function_calls_with_rendered_thinking_select_qwen3_coder_and_qwen3() {
        assert_eq!(formats("qwen3_coder"), (Some("qwen3_coder"), Some("qwen3")));
    }

    #[test]
    fn bracketed_tool_calls_select_mistral() {
        assert_eq!(formats("mistral"), (Some("mistral"), None));
    }

    #[test]
    fn role_tagged_turns_select_granite_for_either_call_marker() {
        assert_eq!(formats("granite_role_bar"), (Some("granite"), None));
        assert_eq!(formats("granite_role_xml"), (Some("granite"), None));
    }

    #[test]
    fn python_tag_calls_select_llama3_json() {
        assert_eq!(formats("llama3_json"), (Some("llama3_json"), None));
    }

    #[test]
    fn json_tool_call_bodies_select_hermes() {
        assert_eq!(formats("hermes"), (Some("hermes"), None));
        assert_eq!(formats("hermes_think"), (Some("hermes"), Some("qwen3")));
    }

    #[test]
    fn harmony_channels_select_gpt_oss_reasoning_only() {
        assert_eq!(formats("harmony"), (None, Some("gpt_oss")));
    }

    #[test]
    fn a_tool_call_without_a_json_body_selects_no_tool_parser() {
        assert_eq!(formats("xml_args_think"), (None, Some("qwen3")));
    }

    #[test]
    fn think_named_only_in_instructions_selects_no_reasoning_parser() {
        assert_eq!(formats("think_instructions"), (Some("granite"), None));
    }

    #[test]
    fn a_template_with_no_markers_selects_nothing() {
        assert_eq!(formats("plain"), (None, None));
        assert_eq!(wire_formats(Some("")), WireFormats::default());
    }

    #[test]
    fn no_template_selects_nothing() {
        assert_eq!(wire_formats(None), WireFormats::default());
    }
}
