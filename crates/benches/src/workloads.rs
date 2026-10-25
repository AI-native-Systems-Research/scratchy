// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! One place for a dataset task: its prompt as the chat messages of a plain request, the span query
//! of the same request (each document a span its cache can be reused from wherever it sits), its
//! answer cap, and how its answer is scored ([`crate::scoring`]). The dataset benchmarks
//! (`scr bench multihop`, `hotpotqa`, `musique`, `msmarco`, `longbench`) build their in-process
//! requests from it, and `scr bench serve --dataset-name rag|rag-spans|longbench` sends the same
//! tasks over HTTP, so both send byte-for-byte the same prompts. Each loader keeps only its parsing.
//!
//! Every engine gets the same messages, so the comparison is of the same prompts. The span form
//! is scratchy's own (`POST /v1/query/execute`): it is timed as a row of its own, never in place
//! of the plain one.

use rand::SeedableRng;
use rand::seq::SliceRandom;

/// One dataset task, ready to send.
#[derive(Clone)]
pub(crate) struct Task {
    /// The plain request's chat messages, in order: `{"role", "content"}`.
    pub(crate) messages: Vec<serde_json::Value>,
    /// The same request as a span query's input, where its documents are separate spans:
    /// `cross[system, plus[document...], user]`. None where the task has no documents to reuse.
    pub(crate) spans_input: Option<serde_json::Value>,
    /// The answer cap: the dataset benchmark's own.
    pub(crate) max_tokens: usize,
    pub(crate) score: Score,
}

/// How a task's answer is scored: a text comparison, so it costs no run time.
#[derive(Clone)]
pub(crate) enum Score {
    /// The answer, or one of its aliases, is in the response, else token F1 at least 0.5 (the
    /// multi-document QA benchmarks' rule), as `rule` says its source scores it.
    Contains {
        answer: String,
        aliases: Vec<String>,
        rule: Rule,
    },
    /// LongBench v2: the letter the response picks, A to D.
    Choice(String),
    /// Nothing to score: a chat turn or a synthetic question with no single right answer.
    None,
}

impl Score {
    /// The reference answer, as the scorer reads it.
    pub(crate) fn reference(&self) -> String {
        match self {
            Score::Contains {
                answer, aliases, ..
            } if aliases.is_empty() => answer.clone(),
            Score::Contains {
                answer, aliases, ..
            } => {
                format!("{answer} (or {})", aliases.join(", "))
            }
            Score::Choice(letter) => letter.clone(),
            Score::None => String::new(),
        }
    }

    /// Whether the task has a right answer to score.
    pub(crate) fn scored(&self) -> bool {
        !matches!(self, Score::None)
    }

    /// 1.0 for a right answer, 0.0 for a wrong one; 0.0 for a task with nothing to score, which
    /// [`Score::scored`] leaves out of accuracy.
    pub(crate) fn score(&self, response: &str) -> f64 {
        match self {
            Score::None => 0.0,
            // Each source scored as its own benchmark scores it ([`crate::scoring`]), whether or
            // not this task has aliases.
            Score::Contains {
                answer,
                aliases,
                rule,
            } => {
                let right = match rule {
                    Rule::OneAnswer => crate::scoring::contains_or_f1(response, answer),
                    Rule::AnyAnswer => {
                        let all: Vec<&str> = std::iter::once(answer.as_str())
                            .chain(aliases.iter().map(String::as_str))
                            .collect();
                        crate::scoring::any_contains_or_f1(response, &all)
                    }
                };
                if right { 1.0 } else { 0.0 }
            }
            Score::Choice(letter) => {
                let picked = crate::longbench::extract_answer(response);
                if picked.as_deref() == Some(letter.as_str()) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Which of [`crate::scoring`]'s rules a multi-document QA source scores by. They differ only where
/// an expected answer has no tokens (punctuation only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Rule {
    /// 2WikiMultihopQA's and HotpotQA's: [`crate::scoring::contains_or_f1`], one answer. Their
    /// loaders need the `datasets` feature.
    #[cfg_attr(not(any(feature = "datasets", test)), allow(dead_code))]
    OneAnswer,
    /// MuSiQue's and MS MARCO's: [`crate::scoring::any_contains_or_f1`], any of the answers.
    AnyAnswer,
}

/// How a multi-document QA dataset words its prompt: the system prompt, and each document's line,
/// `"{label} {j} ({source}): {text}"` for document j of the dataset's (source, text) pairs.
pub(crate) struct RagPrompt {
    pub(crate) system: &'static str,
    pub(crate) label: &'static str,
}

/// MuSiQue's, 2WikiMultihopQA's and HotpotQA's: documents by title.
pub(crate) const DOCUMENTS: RagPrompt = RagPrompt {
    system: "You are a helpful assistant. Answer the question based only on the provided \
             documents. Be concise — answer with just the entity, date, or fact requested.",
    label: "Document",
};

/// MS MARCO's: passages by URL. Its loader needs the `datasets` feature, as MS MARCO's does.
#[cfg(any(feature = "datasets", test))]
pub(crate) const PASSAGES: RagPrompt = RagPrompt {
    system: "You are a helpful assistant. Answer the question based only on the provided \
             passages. Be concise — answer with just the entity, date, number, or fact requested.",
    label: "Passage",
};

/// The answer cap. The multi-document QA benchmarks use 64, but the matrix's models open with a
/// preamble (Granite 4.2: "Based solely on the provided documents..."), and at 64 tokens most
/// answers ended before the answer: 1 of 6 right on MuSiQue. A reply still ends where the model
/// ends it, so the cap only bounds a long one.
pub(crate) const RAG_MAX_TOKENS: usize = 256;

/// A multi-document QA task as the benchmarks send it: the system prompt, each document as its
/// own user message, then the question. `answers` is every accepted answer, the first the main
/// one; `max_tokens` the answer cap.
pub(crate) fn rag_task(
    prompt: &RagPrompt,
    documents: &[(String, String)],
    question: &str,
    answers: &[String],
    rule: Rule,
    max_tokens: usize,
) -> Task {
    let label = prompt.label;
    let doc = |j: usize, source: &str, text: &str| format!("{label} {j} ({source}): {text}");
    let mut messages = vec![serde_json::json!({"role": "system", "content": prompt.system})];
    messages.extend(
        (documents.iter().enumerate())
            .map(|(j, (t, x))| serde_json::json!({"role": "user", "content": doc(j, t, x)})),
    );
    messages.push(serde_json::json!({"role": "user", "content": question}));
    let docs: Vec<serde_json::Value> = (documents.iter().enumerate())
        .map(|(j, (t, x))| serde_json::json!({"user": doc(j, t, x)}))
        .collect();
    let (answer, aliases) = answers
        .split_first()
        .map_or((String::new(), vec![]), |(a, rest)| {
            (a.clone(), rest.to_vec())
        });
    Task {
        messages,
        spans_input: Some(serde_json::json!({
            "cross": [{"system": prompt.system}, {"plus": docs}, {"user": question}]
        })),
        max_tokens,
        score: Score::Contains {
            answer,
            aliases,
            rule,
        },
    }
}

impl Task {
    /// The plain request's messages as the in-process engine takes them.
    pub(crate) fn chat_messages(&self) -> Vec<scratchy_serving_api::llm::ChatMessage> {
        use scratchy_serving_api::llm::ChatMessage;
        (self.messages.iter())
            .map(|m| {
                let text = m["content"].as_str().unwrap_or_default();
                match m["role"].as_str() {
                    Some("system") => ChatMessage::system(text),
                    Some("assistant") => ChatMessage::assistant(text),
                    _ => ChatMessage::user(text),
                }
            })
            .collect()
    }

    /// The span query of the same request, as `POST /v1/query/execute` and the in-process
    /// `LLM::execute_query` take it; None for a task with no documents to reuse.
    pub(crate) fn span_query(
        &self,
        model: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Option<serde_json::Value> {
        let input = self.spans_input.clone()?;
        Some(serde_json::json!({"g": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "input": input,
        }}))
    }
}

/// The longest ShareGPT reply a multi-turn task asks for, in tokens: a turn's cap is its recorded
/// reply's length, at most this.
pub(crate) const CHAT_MAX_TOKENS: usize = 256;

/// The turns per ShareGPT conversation a multi-turn workload sends: the first `CHAT_TURNS`.
pub(crate) const CHAT_TURNS: usize = 4;

/// One ShareGPT conversation as multi-turn tasks: turn k sends the human and gpt turns before it,
/// as the dataset recorded them, then human turn k. So every engine gets the same prompts, and
/// turn k's prompt opens with turn k-1's, for the cache to serve. Its answer cap is the recorded
/// reply's length in tokens (`reply_tokens`), at most [`CHAT_MAX_TOKENS`]. The caller sends them
/// as one unit, in order.
pub(crate) fn chat_tasks(
    turns: &[(String, String)],
    reply_tokens: &dyn Fn(&str) -> usize,
) -> Vec<Task> {
    let mut history: Vec<serde_json::Value> = Vec::new();
    let mut tasks = Vec::new();
    for (human, gpt) in turns.iter().take(CHAT_TURNS) {
        history.push(serde_json::json!({"role": "user", "content": human}));
        tasks.push(Task {
            messages: history.clone(),
            spans_input: None,
            max_tokens: reply_tokens(gpt).clamp(1, CHAT_MAX_TOKENS),
            score: Score::None,
        });
        history.push(serde_json::json!({"role": "assistant", "content": gpt}));
    }
    tasks
}

/// The answer cap of a shared-prefix question: a short answer.
pub(crate) const SHARED_PREFIX_MAX_TOKENS: usize = 32;

/// A synthetic shared-prefix workload, as vLLM's `prefix_repetition` and SGLang's
/// `generated-shared-prefix` build it: `groups` system prompts of `prefix` text, each with
/// `per_group` short questions. With `reuse` false, every question gets a system prompt of its
/// own of the same length: the control, whose cache saves nothing. `text(stream)` is seeded text
/// for that stream ([`crate::datasets::seeded_text`]): the prefix and each question its own.
pub(crate) fn shared_prefix_tasks(
    groups: usize,
    per_group: usize,
    reuse: bool,
    prefix: &dyn Fn(u64) -> String,
    question: &dyn Fn(u64) -> String,
) -> Vec<Task> {
    let mut tasks = Vec::with_capacity(groups * per_group);
    for g in 0..groups {
        for q in 0..per_group {
            let n = (g * per_group + q) as u64;
            let system = if reuse {
                prefix(g as u64)
            } else {
                prefix(1_000_000 + n)
            };
            tasks.push(Task {
                messages: vec![
                    serde_json::json!({"role": "system", "content": system}),
                    serde_json::json!({"role": "user", "content": question(n)}),
                ],
                spans_input: None,
                max_tokens: SHARED_PREFIX_MAX_TOKENS,
                score: Score::None,
            });
        }
    }
    tasks
}

/// `n` tasks, seeded: the same tasks for the same seed, on every engine and every run. Drawn
/// without replacement from `all`; fewer if `all` has fewer.
pub(crate) fn sample<T>(mut all: Vec<T>, n: usize, seed: u64) -> Vec<T> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    all.shuffle(&mut rng);
    all.truncate(n);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rag_answer_is_right_by_substring_alias_or_f1() {
        let s = Score::Contains {
            answer: "Barack Obama".into(),
            aliases: vec!["Obama".into()],
            rule: Rule::AnyAnswer,
        };
        assert_eq!(s.score("It was barack obama."), 1.0);
        assert_eq!(s.score("Obama"), 1.0);
        assert_eq!(s.score("President Barack Hussein Obama"), 1.0);
        assert_eq!(s.score("George Bush"), 0.0);
    }

    /// A source scores by its own benchmark's rule whether or not a task has aliases: the two
    /// differ where neither the expected answer nor the response has a word (an empty answer to a
    /// punctuation-only one), and a MuSiQue task without aliases keeps MuSiQue's.
    #[test]
    fn a_source_keeps_its_own_rule_without_aliases() {
        let by = |rule| Score::Contains {
            answer: "!".into(),
            aliases: vec![],
            rule,
        };
        let response = "";
        assert_eq!(
            by(Rule::AnyAnswer).score(response),
            crate::scoring::any_contains_or_f1(response, &["!"]) as u8 as f64
        );
        assert_eq!(
            by(Rule::OneAnswer).score(response),
            crate::scoring::contains_or_f1(response, "!") as u8 as f64
        );
        assert_ne!(
            by(Rule::AnyAnswer).score(response),
            by(Rule::OneAnswer).score(response),
            "the rules differ here, so the test tells them apart"
        );
    }

    #[test]
    fn a_longbench_answer_is_the_letter_it_picks() {
        assert_eq!(
            Score::Choice("B".into()).score("The correct answer is (B)"),
            1.0
        );
        assert_eq!(
            Score::Choice("B".into()).score("The correct answer is (C)"),
            0.0
        );
    }

    #[test]
    fn a_rag_task_sends_the_same_documents_both_ways() {
        let docs = vec![("Paris".into(), "Capital of France.".into())];
        let t = rag_task(
            &DOCUMENTS,
            &docs,
            "What is the capital?",
            &["Paris".into()],
            Rule::OneAnswer,
            RAG_MAX_TOKENS,
        );
        assert_eq!(t.messages.len(), 3, "system, one document, the question");
        assert_eq!(t.messages[0]["role"], "system");
        assert_eq!(
            t.messages[1]["content"],
            "Document 0 (Paris): Capital of France."
        );
        let spans = t.spans_input.unwrap();
        assert_eq!(
            spans["cross"][1]["plus"][0]["user"],
            t.messages[1]["content"]
        );
        assert_eq!(spans["cross"][2]["user"], "What is the capital?");
        assert_eq!(t.max_tokens, RAG_MAX_TOKENS);
    }

    /// The requests each dataset benchmark built by hand before it built them from a [`Task`],
    /// kept word for word: the plain messages and the span query.
    fn old_plain(
        system: &str,
        label: &str,
        docs: &[(String, String)],
        question: &str,
    ) -> Vec<(String, String)> {
        let mut msgs = vec![("system".to_string(), system.to_string())];
        for (j, (source, text)) in docs.iter().enumerate() {
            msgs.push((
                "user".to_string(),
                format!("{label} {j} ({source}): {text}"),
            ));
        }
        msgs.push(("user".to_string(), question.to_string()));
        msgs
    }

    fn old_spans(
        system: &str,
        label: &str,
        docs: &[(String, String)],
        question: &str,
        model: &str,
        max: usize,
    ) -> serde_json::Value {
        let doc_json: Vec<serde_json::Value> = (docs.iter().enumerate())
            .map(|(j, (source, text))| serde_json::json!({ "user": format!("{label} {j} ({source}): {text}") }))
            .collect();
        serde_json::json!({
            "g": {
                "model": model,
                "max_tokens": max,
                "temperature": 0.0,
                "input": {
                    "cross": [
                        { "system": system },
                        { "plus": doc_json },
                        { "user": question }
                    ]
                }
            }
        })
    }

    /// Every dataset benchmark sends, through the shared [`Task`], byte for byte the plain messages
    /// and the span query it built by hand before: for each wording (documents by title, passages
    /// by URL), over documents with punctuation, unicode, quotes and braces, none, one and several.
    #[test]
    fn the_benchmarks_send_the_prompts_they_built_by_hand() {
        let texts = [
            "",
            "Plain text.",
            "Ünïcödé — dash, \"quotes\", {braces} and {j}",
            "a\nnew line",
        ];
        let doc_sets: Vec<Vec<(String, String)>> = vec![
            vec![],
            vec![("Paris".into(), "Capital of France.".into())],
            texts
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("Title {i} (x)"), t.to_string()))
                .collect(),
            vec![
                ("https://example.com/a?b=c".into(), "A passage.".into()),
                ("".into(), "".into()),
            ],
        ];
        for (prompt, label) in [(&DOCUMENTS, "Document"), (&PASSAGES, "Passage")] {
            for docs in &doc_sets {
                for question in ["Who?", "", "What is {title}?"] {
                    let t = rag_task(prompt, docs, question, &["x".into()], Rule::OneAnswer, 64);
                    let new_plain: Vec<(String, String)> = (t.messages.iter())
                        .map(|m| {
                            (
                                m["role"].as_str().unwrap().into(),
                                m["content"].as_str().unwrap().into(),
                            )
                        })
                        .collect();
                    assert_eq!(
                        new_plain,
                        old_plain(prompt.system, label, docs, question),
                        "{label} plain"
                    );
                    assert_eq!(t.chat_messages().len(), new_plain.len());
                    assert_eq!(
                        t.span_query("m", 64, 0.0).unwrap(),
                        old_spans(prompt.system, label, docs, question, "m", 64),
                        "{label} spans"
                    );
                }
            }
        }
    }

    /// Turn k of a conversation sends the recorded turns before it and then its own human turn, so
    /// turn k's prompt opens with turn k-1's; its cap is the recorded reply's length, bounded.
    #[test]
    fn a_conversations_turns_carry_its_recorded_history() {
        let turns: Vec<(String, String)> = (0..6)
            .map(|k| (format!("question {k}"), "reply ".repeat(k * 200)))
            .collect();
        let words = |t: &str| t.split_whitespace().count();
        let tasks = chat_tasks(&turns, &words);
        assert_eq!(tasks.len(), CHAT_TURNS, "the first CHAT_TURNS turns");
        assert_eq!(tasks[0].messages.len(), 1);
        assert_eq!(
            tasks[2].messages.len(),
            5,
            "two exchanges, then the question"
        );
        assert_eq!(tasks[2].messages[1]["role"], "assistant");
        assert_eq!(tasks[2].messages[1]["content"], turns[0].1);
        assert_eq!(tasks[2].messages[4]["content"], "question 2");
        assert_eq!(
            tasks[1].messages[..1],
            tasks[2].messages[..1],
            "turn 2 opens with turn 1"
        );
        assert_eq!(
            tasks[0].max_tokens, 1,
            "an empty recorded reply still asks for a token"
        );
        assert_eq!(tasks[1].max_tokens, 200);
        assert_eq!(tasks[3].max_tokens, CHAT_MAX_TOKENS, "bounded");
        assert!(!tasks[0].score.scored());
    }

    /// A group's questions share its system prompt; the control gives each its own.
    #[test]
    fn shared_prefix_shares_its_prompt_and_the_control_does_not() {
        let text = |s: u64| format!("text {s}");
        let shared = shared_prefix_tasks(2, 3, true, &text, &text);
        assert_eq!(shared.len(), 6);
        let system = |t: &Task| t.messages[0]["content"].clone();
        assert_eq!(system(&shared[0]), system(&shared[2]), "one group");
        assert_ne!(system(&shared[0]), system(&shared[3]), "two groups");
        assert_ne!(
            shared[0].messages[1], shared[1].messages[1],
            "each question its own"
        );
        let control = shared_prefix_tasks(2, 3, false, &text, &text);
        let prompts: std::collections::HashSet<_> = control.iter().map(system).collect();
        assert_eq!(prompts.len(), 6, "the control shares nothing");
        assert_eq!(
            control[4].messages[1], shared[4].messages[1],
            "the same questions"
        );
    }

    #[test]
    fn a_sample_is_the_same_for_the_same_seed() {
        let all: Vec<u32> = (0..100).collect();
        let a = sample(all.clone(), 8, 7);
        assert_eq!(a, sample(all.clone(), 8, 7));
        assert_ne!(a, sample(all.clone(), 8, 8));
        assert_eq!(a.len(), 8);
        assert_eq!(
            sample(all, 500, 7).len(),
            100,
            "fewer if the dataset has fewer"
        );
    }
}
