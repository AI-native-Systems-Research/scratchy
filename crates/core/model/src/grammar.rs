// SPDX-License-Identifier: Apache-2.0
//! Grammar-guided constrained decoding.
//!
//! Wraps `llguidance` to compile JSON schemas, regex patterns, or Lark/EBNF
//! grammars into token masks for constrained decoding. Each request with a
//! `GuidedGrammar` gets its own `GrammarGuide` that tracks parser state and
//! provides the set of allowed tokens at each step.

use std::fmt::Write;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use scratchy_core_common::sampling::GuidedGrammar;
use toktrie::TokenId;
use toktrie_hf_tokenizers::ByteTokenizer;

// Re-export for consumers (the GPU workers).
pub use llguidance::ParserFactory as LlgParserFactory;

/// A compiled grammar guide that tracks parser state for constrained decoding.
///
/// Created once per request (when `guided_grammar` is set), then queried
/// and advanced at each decode step.
///
/// Matches Python's `GuidanceGrammar` in `backend_guidance.py`:
/// - `terminated` is set when EOS is consumed while the matcher is stopped.
/// - Until terminated, `allowed_tokens()` returns EOS-only when stopped
///   (via `compute_mask_or_eos`), allowing the model to naturally sample EOS.
/// - After terminated, `allowed_tokens()` returns `None` (no mask needed).
pub struct GrammarGuide {
    matcher: Matcher,
    /// Set to `true` once EOS has been consumed while the matcher is stopped.
    /// Mirrors Python's `GuidanceGrammar.terminated`.
    terminated: bool,
}

/// A generic JSON schema that matches any valid JSON object.
const JSON_OBJECT_SCHEMA: &str = r#"{"type": "object"}"#;

// ---------------------------------------------------------------------------
// Structural tag → Lark grammar conversion
// ---------------------------------------------------------------------------

/// A single structural tag definition.
///
/// Mirrors Python's `llguidance.StructTag`.
#[derive(Debug, Clone)]
struct StructTag {
    trigger: String,
    begin: String,
    grammar: serde_json::Value,
    end: String,
}

/// JSON-escape a string for use as a Lark terminal literal (double-quoted).
fn json_escape(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s))
}

/// Convert a structural tag spec (JSON string) into a Lark grammar string.
///
/// This is a faithful port of Python's `llguidance.StructTag.to_grammar()`.
/// The spec comes from the API's `response_format` field after JSON serialization
/// of either `LegacyStructuralTagResponseFormat` or `NewStructuralTagResponseFormat`.
///
/// The returned string is either:
/// - A Lark grammar string (when no side grammars are needed)
/// - A JSON `{"grammars": [...]}` string (when tags use Lark sub-grammars)
pub fn structural_tag_to_grammar(spec_json: &str) -> Result<String, String> {
    let spec: serde_json::Value =
        serde_json::from_str(spec_json).map_err(|e| format!("invalid structural_tag JSON: {e}"))?;

    // Parse into StructTag list. Support both legacy (structures+triggers) and new (format) shapes.
    let tags = parse_struct_tags(&spec)?;
    if tags.is_empty() {
        return Err("structural_tag must contain at least one tag".into());
    }

    // Validate: begin must start with trigger
    for tag in &tags {
        if !tag.begin.starts_with(&tag.trigger) {
            return Err(format!(
                "structural tag begin {:?} must start with trigger {:?}",
                tag.begin, tag.trigger
            ));
        }
    }

    // Build the Lark grammar — matching Python's StructTag.to_grammar() exactly.
    let text_regex = r"(.|\n)*";
    let assume_special = true;

    let tag_options: Vec<String> = (0..tags.len()).map(|i| format!("tag_{i}")).collect();
    let tag_options_str = tag_options.join(" | ");

    let mut lark = String::new();
    writeln!(lark, "%llguidance {{}}").unwrap();
    writeln!(lark, "start: ({tag_options_str})* tag_end").unwrap();
    writeln!(lark, "tag_end: TAG_TEXT").unwrap();
    writeln!(lark, "TAG_TEXT: /{text_regex}/").unwrap();

    let mut side_grammars: Vec<serde_json::Value> = Vec::new();

    for (idx, tag) in tags.iter().enumerate() {
        lark.push('\n');
        let tag_rule = format!("tag_{idx}");

        // Determine grammar directive
        let grammar_str = match &tag.grammar {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)
                .map_err(|e| format!("failed to serialize grammar: {e}"))?,
        };

        let grm = if grammar_str.trim_start().starts_with('{') {
            // JSON schema
            format!("%json {grammar_str}")
        } else {
            // Lark sub-grammar — needs side grammar
            let gname = format!("{tag_rule}_grm");
            side_grammars.push(serde_json::json!({
                "name": gname,
                "lark_grammar": grammar_str
            }));
            format!("@{gname}")
        };

        let beg = &tag.begin[tag.trigger.len()..];
        let beg_escaped = if beg.is_empty() {
            String::new()
        } else {
            json_escape(beg)
        };
        let end_escaped = if tag.end.is_empty() {
            String::new()
        } else {
            json_escape(&tag.end)
        };

        let body = format!("{beg_escaped} {grm} {end_escaped}")
            .trim()
            .to_string();

        if assume_special && tag.trigger.starts_with('<') && tag.trigger.ends_with('>') {
            // Special token trigger — use directly
            writeln!(lark, "{tag_rule}: TAG_TEXT {} {body}", tag.trigger).unwrap();
        } else {
            // Text trigger — use lazy lexeme
            let trig_escaped = json_escape(&tag.trigger);
            writeln!(lark, "{tag_rule}_trig[lazy]: TAG_TEXT {trig_escaped}").unwrap();
            writeln!(lark, "{tag_rule}: {tag_rule}_trig {body}").unwrap();
        }
    }

    let lark = lark.trim_start().to_string();

    if side_grammars.is_empty() {
        Ok(lark)
    } else {
        // Wrap in JSON grammars array with the main grammar first
        side_grammars.insert(
            0,
            serde_json::json!({
                "name": "struct_tag",
                "lark_grammar": lark
            }),
        );
        serde_json::to_string(&serde_json::json!({ "grammars": side_grammars }))
            .map_err(|e| format!("failed to serialize grammars: {e}"))
    }
}

/// Parse the structural tag spec into a list of `StructTag`s.
fn parse_struct_tags(spec: &serde_json::Value) -> Result<Vec<StructTag>, String> {
    // Legacy format: { type: "structural_tag", structures: [...], triggers: [...] }
    if let Some(structures) = spec.get("structures") {
        let structures = structures
            .as_array()
            .ok_or("structural_tag 'structures' must be an array")?;

        let triggers: Vec<String> = spec
            .get("triggers")
            .and_then(|t| t.as_array())
            .ok_or("structural_tag 'triggers' must be an array")?
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or("trigger must be a string".to_string())
                    .map(|s| s.to_string())
            })
            .collect::<Result<_, _>>()?;

        let mut tags = Vec::new();
        for s in structures {
            let begin = s
                .get("begin")
                .and_then(|v| v.as_str())
                .ok_or("structure 'begin' must be a string")?;
            let end = s
                .get("end")
                .and_then(|v| v.as_str())
                .ok_or("structure 'end' must be a string")?;
            // Accept both "schema" and "structural_tag_schema" keys
            let grammar = s
                .get("schema")
                .or_else(|| s.get("structural_tag_schema"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            let trigger = triggers
                .iter()
                .find(|t| begin.starts_with(t.as_str()))
                .ok_or_else(|| {
                    format!(
                        "no trigger found for begin {:?} in triggers {:?}",
                        begin, triggers
                    )
                })?;
            tags.push(StructTag {
                trigger: trigger.clone(),
                begin: begin.to_string(),
                grammar,
                end: end.to_string(),
            });
        }
        return Ok(tags);
    }

    // New format: { type: "structural_tag", format: { ... } }
    if let Some(format_val) = spec.get("format") {
        // The format field can itself contain structures+triggers
        return parse_struct_tags(format_val);
    }

    Err("structural_tag spec must contain 'structures' or 'format'".into())
}

impl GrammarGuide {
    /// Build a grammar guide from a structural tag spec (JSON string).
    ///
    /// Converts the spec to a Lark grammar via `structural_tag_to_grammar()`
    /// then compiles it.
    pub fn from_structural_tag(spec: &str, factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let lark = structural_tag_to_grammar(spec)?;
        // If the result is JSON (grammars array), use from_lark_or_grammar_list
        if lark.trim_start().starts_with('{') {
            let tlg =
                TopLevelGrammar::from_lark_or_grammar_list(&lark).map_err(|e| e.to_string())?;
            let parser = factory.create_parser(tlg).map_err(|e| e.to_string())?;
            Ok(Self {
                matcher: Matcher::new(Ok(parser)),
                terminated: false,
            })
        } else {
            Self::from_ebnf(&lark, factory)
        }
    }

    /// Build a grammar guide from a JSON schema.
    pub fn from_json_schema(
        schema: &serde_json::Value,
        factory: &Arc<ParserFactory>,
    ) -> Result<Self, String> {
        let grammar = TopLevelGrammar::from_json_schema(schema.clone());
        let parser = factory.create_parser(grammar).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
            terminated: false,
        })
    }

    /// Build a grammar guide from an arbitrary regex pattern.
    pub fn from_regex(pattern: &str, factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let grammar = TopLevelGrammar::from_regex(pattern);
        let parser = factory.create_parser(grammar).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
            terminated: false,
        })
    }

    /// Build a grammar guide for generic JSON object output.
    pub fn from_json_object(factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let schema: serde_json::Value =
            serde_json::from_str(JSON_OBJECT_SCHEMA).map_err(|e| e.to_string())?;
        Self::from_json_schema(&schema, factory)
    }

    /// Build a grammar guide from a Lark/EBNF grammar string.
    pub fn from_ebnf(grammar: &str, factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let tlg = TopLevelGrammar::from_lark(grammar.to_string());
        let parser = factory.create_parser(tlg).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
            terminated: false,
        })
    }

    /// Create a `GrammarGuide` from a `GuidedGrammar` specification.
    pub fn from_guided_grammar(
        grammar: &GuidedGrammar,
        factory: &Arc<ParserFactory>,
    ) -> Result<Self, String> {
        match grammar {
            GuidedGrammar::Json => Self::from_json_object(factory),
            GuidedGrammar::JsonSchema { schema } => Self::from_json_schema(schema, factory),
            GuidedGrammar::Regex { pattern } => Self::from_regex(pattern, factory),
            GuidedGrammar::Ebnf { grammar } => Self::from_ebnf(grammar, factory),
            GuidedGrammar::StructuralTag { spec } => Self::from_structural_tag(spec, factory),
        }
    }

    /// Get the set of token IDs allowed at the current state.
    ///
    /// Returns `None` only when terminated (EOS already consumed after grammar
    /// stopped). When the grammar is stopped but not yet terminated, returns
    /// EOS-only mask so the model naturally samples EOS.
    ///
    /// Matches Python's `_fill_bitmasks` + `fill_next_token_bitmask` flow:
    /// `compute_mask_or_eos()` returns EOS-only when stopped, normal mask otherwise.
    pub fn allowed_tokens(&mut self) -> Option<Vec<TokenId>> {
        if self.terminated {
            return None;
        }
        match self.matcher.compute_mask_or_eos() {
            Ok(mask) => {
                let mut tokens = Vec::new();
                mask.iter_set_entries(|idx| tokens.push(idx as TokenId));
                Some(tokens)
            }
            Err(_) => None,
        }
    }

    /// Advance the parser to the next state given a sampled token.
    ///
    /// Mirrors Python's `GuidanceGrammar.accept_tokens()`:
    /// - If the matcher is already stopped, marks as terminated and returns
    ///   early (the EOS token was the signal to stop).
    /// - Otherwise, consumes the token normally.
    ///
    /// Returns `true` if the transition was valid, `false` on error.
    pub fn advance(&mut self, token_id: TokenId) -> bool {
        if self.matcher.is_stopped() {
            // Grammar already stopped — this token is EOS (or garbage after
            // stopped). Mark terminated so `allowed_tokens()` returns None.
            self.terminated = true;
            return true;
        }
        self.matcher.consume_token(token_id).is_ok()
    }

    /// Check if the parser is in a finished/stopped state.
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Check if the grammar has been terminated (EOS consumed after stop).
    ///
    /// Mirrors Python's `GuidanceGrammar.is_terminated()`.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }
}

/// Mask logits so that only `allowed` token IDs survive.
///
/// Sets all logits for tokens NOT in `allowed` to `-inf`.
/// This is applied early in the sampling pipeline (before penalties/temperature)
/// so that hard grammar constraints are not softened by later steps.
pub fn apply_grammar_mask(logits: &mut [f32], allowed: &[TokenId]) {
    // Mark allowed positions, then sweep all logits.
    let mut mask = vec![false; logits.len()];
    for &tid in allowed {
        let idx = tid as usize;
        if idx < mask.len() {
            mask[idx] = true;
        }
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !mask[i] {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Build a `TokEnv` and `ParserFactory` from raw tokenizer.json bytes.
///
/// Workers call this once at model-load time and cache the result.
/// The `ParserFactory` is expensive to create (builds sliced bias computer)
/// but is reused across all requests.
pub fn build_parser_factory(tokenizer_json: &[u8]) -> Result<Arc<ParserFactory>, String> {
    let byte_tok = ByteTokenizer::from_json_bytes(tokenizer_json).map_err(|e| e.to_string())?;
    let tok_env = byte_tok.into_tok_env(None).map_err(|e| e.to_string())?;
    let mut factory = ParserFactory::new_simple(&tok_env).map_err(|e| e.to_string())?;
    factory.set_stderr_log_level(0);
    Ok(Arc::new(factory))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_factory() -> Arc<ParserFactory> {
        // Use a minimal BPE tokenizer JSON for testing.
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 99, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "vocab": {
                    "{": 1,
                    "}": 2,
                    "\"": 3,
                    ":": 4,
                    ",": 5,
                    " ": 6,
                    "0": 7,
                    "1": 8,
                    "2": 9,
                    "3": 10,
                    "a": 11,
                    "b": 12,
                    "n": 13,
                    "u": 14,
                    "l": 15,
                    "t": 16,
                    "r": 17,
                    "e": 18,
                    "f": 19,
                    "s": 20,
                    "[": 21,
                    "]": 22,
                    ".": 23,
                    "-": 24,
                    "4": 25,
                    "5": 26,
                    "6": 27,
                    "7": 28,
                    "8": 29,
                    "9": 30
                },
                "merges": []
            }
        }"#;
        build_parser_factory(tokenizer_json.as_bytes()).unwrap()
    }

    #[test]
    fn test_grammar_mask_basic() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let allowed = vec![1u32, 3];
        apply_grammar_mask(&mut logits, &allowed);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_grammar_mask_empty_allowed() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let allowed: Vec<u32> = vec![];
        apply_grammar_mask(&mut logits, &allowed);

        assert!(logits.iter().all(|&l| l == f32::NEG_INFINITY));
    }

    #[test]
    fn test_grammar_mask_all_allowed() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let allowed = vec![0u32, 1, 2];
        apply_grammar_mask(&mut logits, &allowed);

        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_build_parser_factory() {
        let factory = make_test_factory();
        // Factory should be usable for creating parsers.
        let grammar = TopLevelGrammar::from_regex("[0-9]+");
        assert!(factory.create_parser(grammar).is_ok());
    }

    #[test]
    fn test_from_regex_digit_pattern() {
        let factory = make_test_factory();
        let mut guide = GrammarGuide::from_regex("[0-9]+", &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_from_guided_grammar_regex_variant() {
        let factory = make_test_factory();
        let grammar = GuidedGrammar::Regex {
            pattern: "[0-9]+".to_string(),
        };
        let mut guide = GrammarGuide::from_guided_grammar(&grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_advance_and_finish() {
        let factory = make_test_factory();
        // Use a simple regex: exactly one digit.
        let mut guide = GrammarGuide::from_regex("[0-9]", &factory).unwrap();

        // At initial state, we should have allowed tokens.
        let allowed = guide.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());

        // Advance with "1" (token_id=8 in our vocab).
        assert!(guide.advance(8));

        // After one digit, the guide should be finished (regex fully matched).
        // The matcher is stopped, but NOT terminated — allowed_tokens should
        // return EOS-only mask (matching Python's compute_mask_or_eos behavior).
        assert!(guide.is_finished());
        assert!(!guide.is_terminated());
        let eos_allowed = guide.allowed_tokens();
        assert!(
            eos_allowed.is_some(),
            "stopped but not terminated should return EOS-only mask"
        );
        let eos_tokens = eos_allowed.unwrap();
        assert!(
            !eos_tokens.is_empty(),
            "EOS-only mask should contain at least one token"
        );

        // Advance with EOS — now terminated.
        let eos_token = eos_tokens[0];
        assert!(guide.advance(eos_token));
        assert!(guide.is_terminated());
        assert!(
            guide.allowed_tokens().is_none(),
            "terminated grammar should return None"
        );
    }

    #[test]
    fn test_from_ebnf_grammar() {
        let factory = make_test_factory();
        // Simple Lark grammar for digits.
        let grammar = r#"start: /[0-9]+/"#;
        let mut guide = GrammarGuide::from_ebnf(grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_from_guided_grammar_ebnf_variant() {
        let factory = make_test_factory();
        let grammar = GuidedGrammar::Ebnf {
            grammar: r#"start: /[0-9]+/"#.to_string(),
        };
        let mut guide = GrammarGuide::from_guided_grammar(&grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    // ---------------------------------------------------------------
    // structural_tag_to_grammar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_structural_tag_special_token_trigger() {
        // Special token trigger (starts and ends with <>) — should use direct token reference.
        // This matches Python's StructTag.to_grammar() with assume_special=True.
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<function=get_weather>",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                "end": "</function>"
            }],
            "triggers": ["<function"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();

        // With a special token trigger, it should use the trigger directly (not lazy)
        // Python produces: tag_0: TAG_TEXT <function "=get_weather>" %json {...} "</function>"
        assert!(
            result.contains("%llguidance {}"),
            "missing llguidance header: {result}"
        );
        assert!(
            result.contains("start: (tag_0)* tag_end"),
            "missing start rule: {result}"
        );
        assert!(
            result.contains("tag_end: TAG_TEXT"),
            "missing tag_end: {result}"
        );
        assert!(
            result.contains(r"TAG_TEXT: /(.|\n)*/"),
            "missing TAG_TEXT regex: {result}"
        );
        // The trigger <function doesn't end with >, so it should use lazy lexeme
        assert!(
            result.contains("tag_0_trig[lazy]"),
            "missing lazy lexeme: {result}"
        );
        assert!(result.contains("%json"), "missing json directive: {result}");
        assert!(
            result.contains("\"</function>\""),
            "missing end tag: {result}"
        );
    }

    #[test]
    fn test_structural_tag_special_token_trigger_angle_brackets() {
        // Trigger that both starts AND ends with <> — true special token
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<|python_tag|>{\"name\":\"foo\",\"parameters\":",
                "schema": {"type": "object"},
                "end": "}"
            }],
            "triggers": ["<|python_tag|>"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();

        // <|python_tag|> starts with < and ends with > — assume_special applies
        assert!(result.contains("tag_0: TAG_TEXT <|python_tag|>"));
        // Should NOT have lazy lexeme for this trigger
        assert!(!result.contains("tag_0_trig[lazy]"));
    }

    #[test]
    fn test_structural_tag_text_trigger_uses_lazy_lexeme() {
        // Text trigger (not special token) — should use lazy lexeme.
        let spec = serde_json::json!({
            "structures": [{
                "begin": "TOOL_CALL: get_weather(",
                "schema": {"type": "object"},
                "end": ")"
            }],
            "triggers": ["TOOL_CALL:"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();

        // Text trigger — should use lazy lexeme
        assert!(result.contains("tag_0_trig[lazy]: TAG_TEXT \"TOOL_CALL:\""));
        assert!(result.contains("tag_0: tag_0_trig"));
    }

    #[test]
    fn test_structural_tag_multiple_tags() {
        // Multiple structural tags with different triggers
        let spec = serde_json::json!({
            "structures": [
                {
                    "begin": "<function=get_weather>",
                    "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "end": "</function>"
                },
                {
                    "begin": "<function=search>",
                    "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "end": "</function>"
                }
            ],
            "triggers": ["<function"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();

        assert!(result.contains("start: (tag_0 | tag_1)* tag_end"));
        assert!(result.contains("tag_0"));
        assert!(result.contains("tag_1"));
    }

    #[test]
    fn test_structural_tag_empty_end() {
        // Empty end string — should not produce empty quotes
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<|tool|>",
                "schema": {"type": "object"},
                "end": ""
            }],
            "triggers": ["<|tool|>"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();
        assert!(result.contains("%json"));
        // Should not have trailing empty string literal
        assert!(!result.contains("\"\""));
    }

    #[test]
    fn test_structural_tag_new_format() {
        // New format: { type: "structural_tag", format: { structures: [...], triggers: [...] } }
        let spec = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "structures": [{
                    "begin": "<fn=test>",
                    "schema": {"type": "object"},
                    "end": "</fn>"
                }],
                "triggers": ["<fn"]
            }
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();
        assert!(result.contains("tag_0"));
        assert!(result.contains("%json"));
    }

    #[test]
    fn test_structural_tag_error_empty_tags() {
        let spec = serde_json::json!({
            "structures": [],
            "triggers": ["<fn"]
        });
        let result = structural_tag_to_grammar(&spec.to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one tag"));
    }

    #[test]
    fn test_structural_tag_error_no_matching_trigger() {
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<function=test>",
                "schema": {"type": "object"},
                "end": "</function>"
            }],
            "triggers": ["<tool"]
        });
        let result = structural_tag_to_grammar(&spec.to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no trigger found"));
    }

    #[test]
    fn test_structural_tag_error_begin_not_starting_with_trigger() {
        let spec = serde_json::json!({
            "structures": [{
                "begin": "wrong_prefix",
                "schema": {"type": "object"},
                "end": ""
            }],
            "triggers": ["<tool"]
        });
        let result = structural_tag_to_grammar(&spec.to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_structural_tag_error_invalid_json() {
        let result = structural_tag_to_grammar("not valid json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid structural_tag JSON"));
    }

    #[test]
    fn test_structural_tag_error_missing_structures_and_format() {
        let spec = serde_json::json!({"type": "structural_tag"});
        let result = structural_tag_to_grammar(&spec.to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_structural_tag_json_escaping() {
        // Verify that begin/end strings with special chars are properly escaped
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<fn=\"test\">",
                "schema": {"type": "object"},
                "end": "</fn>"
            }],
            "triggers": ["<fn"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();
        // The begin suffix should be properly JSON-escaped
        assert!(result.contains("=\\\"test\\\">")); // escaped quotes
    }

    #[test]
    fn test_structural_tag_lark_sub_grammar_produces_grammars_json() {
        // When grammar is a Lark grammar (not JSON), should produce grammars JSON array
        let spec = serde_json::json!({
            "structures": [{
                "begin": "<fn=calc>",
                "schema": "start: /[0-9]+/",
                "end": "</fn>"
            }],
            "triggers": ["<fn"]
        });
        let result = structural_tag_to_grammar(&spec.to_string()).unwrap();

        // Should be JSON with grammars array
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let grammars = parsed["grammars"].as_array().unwrap();
        assert_eq!(grammars.len(), 2); // main + side grammar
        assert_eq!(grammars[0]["name"], "struct_tag");
        assert_eq!(grammars[1]["name"], "tag_0_grm");
        assert!(
            grammars[1]["lark_grammar"]
                .as_str()
                .unwrap()
                .contains("[0-9]+")
        );
    }

    #[test]
    fn test_from_structural_tag_creates_grammar_guide() {
        // Test that from_structural_tag produces a working GrammarGuide
        let factory = make_test_factory();
        let spec = serde_json::json!({
            "structures": [{
                "begin": "CALL:",
                "schema": {"type": "object"},
                "end": ";END"
            }],
            "triggers": ["CALL:"]
        });
        let result = GrammarGuide::from_structural_tag(&spec.to_string(), &factory);
        assert!(
            result.is_ok(),
            "from_structural_tag failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_guided_grammar_structural_tag_variant() {
        // Test via the GuidedGrammar dispatch
        let factory = make_test_factory();
        let spec = serde_json::json!({
            "structures": [{
                "begin": "FN:",
                "schema": {"type": "object"},
                "end": ";END"
            }],
            "triggers": ["FN:"]
        });
        let grammar = GuidedGrammar::StructuralTag {
            spec: spec.to_string(),
        };
        let result = GrammarGuide::from_guided_grammar(&grammar, &factory);
        assert!(
            result.is_ok(),
            "from_guided_grammar(StructuralTag) failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_structural_tag_serde_roundtrip() {
        // Verify GuidedGrammar::StructuralTag serializes/deserializes correctly
        let grammar = GuidedGrammar::StructuralTag {
            spec: r#"{"structures":[]}"#.to_string(),
        };
        let json = serde_json::to_string(&grammar).unwrap();
        let parsed: GuidedGrammar = serde_json::from_str(&json).unwrap();
        match parsed {
            GuidedGrammar::StructuralTag { spec } => {
                assert_eq!(spec, r#"{"structures":[]}"#);
            }
            other => panic!("expected StructuralTag, got {other:?}"),
        }
    }
}
