// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Chat template application using HuggingFace Jinja2 templates.
//!
//! HuggingFace models store a Jinja2 chat template in `tokenizer_config.json`.
//! This module parses that template and applies it to a list of chat messages,
//! producing the formatted prompt string that the model expects.
//!
//! Uses `minijinja` (a Rust Jinja2 engine) for template rendering.
//! The `Environment` and compiled template are built once at construction time
//! and reused for every request, avoiding per-request template compilation.

use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};

use crate::error::ServeError;

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

/// A parsed chat template that can format messages for a specific model.
///
/// The minijinja `Environment` (with the compiled template) is built once
/// at construction time and reused on every `apply()` call.
pub struct ChatTemplate {
    /// Pre-compiled minijinja environment with the "chat" template loaded.
    env: Environment<'static>,
    /// Optional BOS token string (e.g. "<s>", "<|begin_of_text|>").
    bos_token: Option<String>,
    /// Optional EOS token string (e.g. "</s>", "<|end_of_text|>").
    eos_token: Option<String>,
    /// The raw Jinja2 template string (kept for `template_str()` accessor).
    template_str: String,
}

/// A single chat message for template rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

/// Partial representation of `tokenizer_config.json` for extracting the
/// chat template and special tokens.
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<serde_json::Value>,
    bos_token: Option<serde_json::Value>,
    eos_token: Option<serde_json::Value>,
}

/// Build a minijinja Environment with the given template string compiled as "chat".
fn build_env(template_str: &str) -> Result<Environment<'static>, ServeError> {
    let mut env = Environment::new();

    // HuggingFace's `apply_chat_template` renders with `trim_blocks=True`
    // and `lstrip_blocks=True`. Templates are written assuming both —
    // block tags `{% ... %}` live on their own lines and the surrounding
    // whitespace is expected to be stripped. Without these the trailing
    // `\n` after every `{% ... %}` line leaks into the rendered prompt,
    // which then re-tokenizes differently than the model's training
    // distribution. For TinyLlama-Chat-v1.0 the bare `<` of `<|user|>`
    // is token 529 in the trained tokenization but lands as 29966 once a
    // stray `\n` runs precede it; the model has effectively never seen
    // that input, so generation produces nonsense fragments.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // Enable Python string/dict/list methods (startswith, endswith, etc.)
    // that HuggingFace Jinja2 chat templates commonly use.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Add a `raise_exception` function that Jinja2 templates often use.
    env.add_function("raise_exception", raise_exception);

    // Add `strftime_now` — used by HuggingFace transformers chat templates
    // (e.g. granite, llama4) to inject the current date/time.
    env.add_function("strftime_now", strftime_now);

    env.add_template_owned("chat", template_str.to_owned())
        .map_err(|e| ServeError::Internal(format!("invalid chat template: {e}")))?;

    Ok(env)
}

impl ChatTemplate {
    /// Create a `ChatTemplate` from a raw Jinja2 template string.
    pub fn new(template_str: String) -> Result<Self, ServeError> {
        let env = build_env(&template_str)?;
        Ok(Self {
            env,
            bos_token: None,
            eos_token: None,
            template_str,
        })
    }

    /// Set the BOS token string used by the template.
    pub fn with_bos_token(mut self, bos: String) -> Self {
        self.bos_token = Some(bos);
        self
    }

    /// Set the EOS token string used by the template.
    pub fn with_eos_token(mut self, eos: String) -> Self {
        self.eos_token = Some(eos);
        self
    }

    /// Fallback: load the template from a sibling `chat_template.jinja`
    /// file (the Gemma 4 repo convention — HF transformers#45205 moved
    /// the template out of tokenizer_config.json into a standalone
    /// file, and conversions often omit re-embedding it).
    fn from_sibling_jinja(tokenizer_config_path: &Path) -> Result<Option<Self>, ServeError> {
        let jinja = tokenizer_config_path
            .parent()
            .map(|d| d.join("chat_template.jinja"))
            .filter(|p| p.exists());
        let Some(jinja) = jinja else {
            return Ok(None);
        };
        let template_str = std::fs::read_to_string(&jinja).map_err(|e| {
            ServeError::Internal(format!("failed to read {}: {e}", jinja.display()))
        })?;
        if template_str.trim().is_empty() {
            return Ok(None);
        }
        let mut tpl = Self::new(template_str)?;
        // bos/eos still come from tokenizer_config.json when present —
        // the .jinja file carries only the template body, and Gemma 4's
        // template opens with `{{ bos_token }}` (an empty render here
        // silently drops <bos> and shifts every prompt token).
        if tokenizer_config_path.exists()
            && let Ok(data) = std::fs::read_to_string(tokenizer_config_path)
            && let Ok(config) = serde_json::from_str::<TokenizerConfig>(&data)
        {
            if let Some(bos) = extract_token_string(config.bos_token) {
                tpl = tpl.with_bos_token(bos);
            }
            if let Some(eos) = extract_token_string(config.eos_token) {
                tpl = tpl.with_eos_token(eos);
            }
        }
        Ok(Some(tpl))
    }

    /// Load a `ChatTemplate` from a `tokenizer_config.json` file.
    ///
    /// Returns `None` if the file doesn't exist or doesn't contain a
    /// `chat_template` field (falls back to a sibling
    /// `chat_template.jinja` in both cases).
    pub fn from_tokenizer_config(path: &Path) -> Result<Option<Self>, ServeError> {
        if !path.exists() {
            return Self::from_sibling_jinja(path);
        }

        let data = std::fs::read_to_string(path)
            .map_err(|e| ServeError::Internal(format!("failed to read {}: {e}", path.display())))?;

        let config: TokenizerConfig = serde_json::from_str(&data).map_err(|e| {
            ServeError::Internal(format!("failed to parse {}: {e}", path.display()))
        })?;

        let template_str = match config.chat_template {
            Some(serde_json::Value::String(s)) => s,
            Some(serde_json::Value::Array(arr)) => {
                // Some models have an array of templates; take the first
                // (or the one named "default").
                arr.iter()
                    .find_map(|v| {
                        let obj = v.as_object()?;
                        let name = obj.get("name")?.as_str()?;
                        if name == "default" {
                            obj.get("template")?.as_str().map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        // Fall back to the first template.
                        arr.first()
                            .and_then(|v| {
                                v.as_object()
                                    .and_then(|obj| obj.get("template"))
                                    .and_then(|t| t.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| arr.first().and_then(|v| v.as_str()).map(|s| s.to_string()))
                    })
                    .unwrap_or_default()
            }
            // Gemma 4 convention (HF transformers#45205): the template
            // ships as a SEPARATE `chat_template.jinja` file beside
            // tokenizer_config.json instead of an embedded field.
            _ => return Self::from_sibling_jinja(path),
        };

        if template_str.is_empty() {
            return Ok(None);
        }

        let bos_token = extract_token_string(config.bos_token);
        let eos_token = extract_token_string(config.eos_token);

        let mut tpl = ChatTemplate::new(template_str)?;
        if let Some(bos) = bos_token {
            tpl = tpl.with_bos_token(bos);
        }
        if let Some(eos) = eos_token {
            tpl = tpl.with_eos_token(eos);
        }

        Ok(Some(tpl))
    }

    /// Apply the chat template to rich JSON messages, with optional tool definitions
    /// and extra template kwargs (e.g. `enable_thinking`).
    ///
    /// Messages are `serde_json::Value` objects so templates can access any field
    /// (`tool_calls`, `tool_call_id`, `name`, etc.) without needing Rust struct changes.
    ///
    /// Returns the formatted prompt string ready for tokenization.
    pub fn apply(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String, ServeError> {
        self.apply_with_kwargs(messages, add_generation_prompt, tools, None)
    }

    /// Like [`apply`] but with additional template keyword arguments.
    ///
    /// `extra_kwargs` are merged into the Jinja context so the template can
    /// access them (e.g. `enable_thinking`, `reasoning_effort`).
    pub fn apply_with_kwargs(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
        extra_kwargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<String, ServeError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| ServeError::Internal(format!("failed to get template: {e}")))?;

        // Today's date string — used by some templates (e.g. LLaMA 3.1).
        let date_string = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // Simple UTC date: days since epoch.
            let days = now / 86400;
            // 1970-01-01 is a Thursday (day 4).
            let (year, month, day) = days_to_ymd(days);
            format!(
                "{:02} {month_name} {year}",
                day,
                month_name = MONTH_NAMES[month as usize - 1],
                year = year
            )
        };

        // Build the base context.
        let mut ctx_map = std::collections::BTreeMap::<String, minijinja::Value>::new();
        ctx_map.insert(
            "messages".into(),
            minijinja::Value::from_serialize(messages),
        );
        ctx_map.insert(
            "add_generation_prompt".into(),
            minijinja::Value::from(add_generation_prompt),
        );
        ctx_map.insert(
            "bos_token".into(),
            minijinja::Value::from(self.bos_token.as_deref().unwrap_or("")),
        );
        ctx_map.insert(
            "eos_token".into(),
            minijinja::Value::from(self.eos_token.as_deref().unwrap_or("")),
        );
        if let Some(tools) = tools {
            ctx_map.insert("tools".into(), minijinja::Value::from_serialize(tools));
        }
        ctx_map.insert("date_string".into(), minijinja::Value::from(date_string));

        // Merge extra kwargs (e.g. enable_thinking, reasoning_effort).
        if let Some(kwargs) = extra_kwargs {
            for (key, value) in kwargs {
                ctx_map.insert(key.clone(), minijinja::Value::from_serialize(value));
            }
        }

        let ctx = minijinja::Value::from_serialize(&ctx_map);

        let rendered = tmpl
            .render(ctx)
            .map_err(|e| ServeError::Internal(format!("chat template render failed: {e}")))?;

        Ok(rendered)
    }

    /// Convenience wrapper: apply with simple `TemplateMessage` slices and no tools.
    ///
    /// Used by tests and callers that don't need tool support.
    pub fn apply_simple(
        &self,
        messages: &[TemplateMessage],
        add_generation_prompt: bool,
    ) -> Result<String, ServeError> {
        let values: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        self.apply(&values, add_generation_prompt, None)
    }

    /// Get the raw template string.
    pub fn template_str(&self) -> &str {
        &self.template_str
    }
}

/// Extract a token string from the `bos_token` / `eos_token` field in
/// tokenizer_config.json. These can be either a plain string or an object
/// with a `content` field.
fn extract_token_string(value: Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Object(mut obj)) => obj.remove("content").and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        }),
        _ => None,
    }
}

const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const FULL_MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const DAY_NAMES: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

const FULL_DAY_NAMES: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (simplified Euclidean).
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400))
}

/// A raise_exception function for Jinja2 compatibility.
/// Many HF chat templates use `{% raise_exception(...) %}` for error handling.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// `strftime_now(format)` — returns the current UTC time formatted with the
/// given strftime format string. Used by HuggingFace transformers chat templates
/// (granite, llama4, etc.) to inject the current date.
///
/// Supports the directives those templates actually emit
/// (`%Y %y %m %d %H %M %S %B %b %A %a %j %p %%`); anything else passes
/// through literally rather than panicking, since a chat template must
/// never fail to render over a date format quirk.
fn strftime_now(format: String) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86_400;
    let secs_of_day = now % 86_400;
    let (year, month, day) = days_to_ymd(days);
    let (hour, minute, second) = (
        secs_of_day / 3600,
        (secs_of_day / 60) % 60,
        secs_of_day % 60,
    );
    // 1970-01-01 (day 0) was a Thursday (index 4).
    let weekday = ((days + 4) % 7) as usize;
    let day_of_year = day_of_year(year, month, day);

    let mut out = String::with_capacity(format.len());
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&year.to_string()),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('m') => out.push_str(&format!("{:02}", month)),
            Some('d') => out.push_str(&format!("{:02}", day)),
            Some('H') => out.push_str(&format!("{:02}", hour)),
            Some('M') => out.push_str(&format!("{:02}", minute)),
            Some('S') => out.push_str(&format!("{:02}", second)),
            Some('B') => out.push_str(FULL_MONTH_NAMES[month as usize - 1]),
            Some('b') => out.push_str(MONTH_NAMES[month as usize - 1]),
            Some('A') => out.push_str(FULL_DAY_NAMES[weekday]),
            Some('a') => out.push_str(DAY_NAMES[weekday]),
            Some('j') => out.push_str(&format!("{:03}", day_of_year)),
            Some('p') => out.push_str(if hour < 12 { "AM" } else { "PM" }),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// 1-indexed day of year for a given (year, month, day).
fn day_of_year(year: u64, month: u64, day: u64) -> u64 {
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    month_days[..month as usize - 1].iter().sum::<u64>() + day
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_template() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}".to_string(),
        ).unwrap();

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let result = tpl.apply_simple(&messages, true).unwrap();
        assert!(result.contains("user: Hello!"));
        assert!(result.contains("assistant:"));
    }

    #[test]
    fn test_chatml_template() {
        // ChatML format used by Qwen, Yi, etc.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![
            TemplateMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            TemplateMessage {
                role: "user".to_string(),
                content: "Hi!".to_string(),
            },
        ];

        let result = tpl.apply_simple(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHi!<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_bos_eos_tokens() {
        let tpl = ChatTemplate::new(
            "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}".to_string(),
        ).unwrap()
        .with_bos_token("<s>".to_string())
        .with_eos_token("</s>".to_string());

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let result = tpl.apply_simple(&messages, false).unwrap();
        assert_eq!(result, "<s>Hello</s>");
    }

    #[test]
    fn test_no_generation_prompt() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}GENERATE{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let with = tpl.apply_simple(&messages, true).unwrap();
        let without = tpl.apply_simple(&messages, false).unwrap();

        assert!(with.contains("GENERATE"));
        assert!(!without.contains("GENERATE"));
    }

    #[test]
    fn test_empty_messages() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();

        let result = tpl.apply_simple(&[], false).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_multi_turn_conversation() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}[{{ message.role|upper }}] {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![
            TemplateMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
            TemplateMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
            },
            TemplateMessage {
                role: "user".to_string(),
                content: "Thanks!".to_string(),
            },
        ];

        let result = tpl.apply_simple(&messages, false).unwrap();
        assert!(result.contains("[USER] What is 2+2?"));
        assert!(result.contains("[ASSISTANT] 4"));
        assert!(result.contains("[USER] Thanks!"));
    }

    #[test]
    fn test_from_tokenizer_config_missing_file() {
        let result =
            ChatTemplate::from_tokenizer_config(Path::new("/nonexistent/tokenizer_config.json"))
                .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_token_string_plain() {
        let val = Some(serde_json::Value::String("<s>".to_string()));
        assert_eq!(extract_token_string(val), Some("<s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_object() {
        let obj = serde_json::json!({"content": "</s>", "lstrip": false});
        let val = Some(obj);
        assert_eq!(extract_token_string(val), Some("</s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_none() {
        assert_eq!(extract_token_string(None), None);
    }

    #[test]
    fn test_tools_passed_to_template() {
        // Template that dumps tools via tojson.
        let tpl = ChatTemplate::new(
            "{% if tools %}TOOLS:{{ tools | tojson }}{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}
        ]);

        let result = tpl.apply(&messages, false, Some(&tools)).unwrap();
        assert!(result.contains("TOOLS:"));
        assert!(result.contains("get_weather"));
    }

    #[test]
    fn test_tool_calls_in_message() {
        // Template that accesses tool_calls on a message.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}:{% if message.tool_calls %} CALL={{ message.tool_calls[0].function.name }}{% endif %} {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "NYC"}}
                }]
            }),
            serde_json::json!({
                "role": "tool",
                "content": "{\"temp\": 72}",
                "tool_call_id": "call_1"
            }),
        ];

        let result = tpl.apply(&messages, false, None).unwrap();
        assert!(result.contains("CALL=get_weather"));
        assert!(result.contains("tool:"));
    }

    #[test]
    fn test_no_tools_omitted_from_template() {
        let tpl =
            ChatTemplate::new("{% if tools %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}".to_string())
                .unwrap();

        let result = tpl.apply(&[], false, None).unwrap();
        assert!(result.contains("NO_TOOLS"));

        let tools = serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
        let result = tpl.apply(&[], false, Some(&tools)).unwrap();
        assert!(result.contains("HAS_TOOLS"));
    }

    #[test]
    fn test_date_string_in_template() {
        let tpl = ChatTemplate::new("DATE:{{ date_string }}".to_string()).unwrap();
        let result = tpl.apply(&[], false, None).unwrap();
        // Should contain a date like "28 Feb 2026".
        assert!(result.starts_with("DATE:"));
        assert!(result.len() > 5);
    }

    #[test]
    fn test_extra_kwargs_enable_thinking_true() {
        // Simplified Qwen3-style template that checks enable_thinking.
        let tpl = ChatTemplate::new(
            "{% set enable_thinking = enable_thinking | default(true) %}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];

        // No extra kwargs → enable_thinking defaults to true → <think> present.
        let result = tpl.apply(&messages, true, None).unwrap();
        assert!(
            result.contains("<think>"),
            "expected <think> with default enable_thinking=true, got: {result}"
        );

        // Explicit enable_thinking=true.
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        let result = tpl
            .apply_with_kwargs(&messages, true, None, Some(&kwargs))
            .unwrap();
        assert!(
            result.contains("<think>"),
            "expected <think> with enable_thinking=true, got: {result}"
        );
    }

    #[test]
    fn test_extra_kwargs_enable_thinking_false() {
        // Simplified Qwen3-style template that checks enable_thinking.
        let tpl = ChatTemplate::new(
            "{% set enable_thinking = enable_thinking | default(true) %}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];

        // enable_thinking=false → no <think>.
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        let result = tpl
            .apply_with_kwargs(&messages, true, None, Some(&kwargs))
            .unwrap();
        assert!(
            !result.contains("<think>"),
            "expected no <think> with enable_thinking=false, got: {result}"
        );
    }

    #[test]
    fn test_extra_kwargs_do_not_override_builtins() {
        // Extra kwargs should be able to add new variables but built-in context
        // (messages, add_generation_prompt, etc.) should still work.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}{% if custom_var %}CUSTOM{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("custom_var".to_string(), serde_json::Value::Bool(true));
        let result = tpl
            .apply_with_kwargs(&messages, false, None, Some(&kwargs))
            .unwrap();
        assert!(result.contains("Hello"), "messages should still render");
        assert!(result.contains("CUSTOM"), "custom_var should be accessible");
    }

    #[test]
    fn test_apply_with_kwargs_none_is_same_as_apply() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let a = tpl.apply(&messages, false, None).unwrap();
        let b = tpl.apply_with_kwargs(&messages, false, None, None).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_extra_kwargs_empty_map() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let kwargs = std::collections::HashMap::new();
        let result = tpl
            .apply_with_kwargs(&messages, false, None, Some(&kwargs))
            .unwrap();
        assert_eq!(result, "Hi");
    }

    #[test]
    fn test_strftime_now_common_directives() {
        let out = strftime_now("%Y-%m-%d %H:%M:%S".to_string());
        // "YYYY-MM-DD HH:MM:SS" is 19 chars; every field is fixed-width.
        assert_eq!(out.len(), 19);
        assert_eq!(out.as_bytes()[4], b'-');
        assert_eq!(out.as_bytes()[7], b'-');
        assert_eq!(out.as_bytes()[10], b' ');
        assert_eq!(out.as_bytes()[13], b':');
        assert_eq!(out.as_bytes()[16], b':');
    }

    #[test]
    fn test_strftime_now_literal_passthrough() {
        // Templates like llama's `%d %b %Y` mix literals with directives.
        let out = strftime_now("%d %b %Y".to_string());
        let parts: Vec<&str> = out.split(' ').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].len(), 2);
        assert!(MONTH_NAMES.contains(&parts[1]));
        assert_eq!(parts[2].len(), 4);
    }

    #[test]
    fn test_strftime_now_unknown_directive_passes_through() {
        let out = strftime_now("%Q".to_string());
        assert_eq!(out, "%Q");
    }

    #[test]
    fn test_day_of_year_epoch() {
        assert_eq!(day_of_year(1970, 1, 1), 1);
        assert_eq!(day_of_year(1970, 12, 31), 365);
        assert_eq!(day_of_year(2024, 12, 31), 366); // leap year
    }
}
