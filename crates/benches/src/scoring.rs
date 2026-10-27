// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! How the dataset benchmarks score an answer, in one place: the multi-document QA rule
//! (`multihop`, `hotpotqa`, `musique`, `msmarco`, and `bench serve`'s dataset workloads) and the
//! token F1 they report beside it. Each was written out in every benchmark; they are the same rule,
//! and a test below holds each to the behaviour the benchmark had on its own.

/// Tokens in `text` under the model's tokenizer, 0 where it does not encode: how the dataset
/// benchmarks size a prompt.
pub(crate) fn token_len(text: &str, tokenizer: &scratchy_serving_api::Tokenizer) -> usize {
    tokenizer
        .encode(text, false)
        .map(|ids| ids.len())
        .unwrap_or(0)
}

/// Lowercase alphanumeric runs: the tokens F1 compares.
pub(crate) fn normalize_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Token F1 of `actual` against `expected`, both already lowercase: 1.0 when neither has a token,
/// 0.0 when one has none, else the harmonic mean of precision and recall over shared tokens.
pub(crate) fn token_f1_lower(expected: &str, actual: &str) -> f64 {
    let et = normalize_tokens(expected);
    let at = normalize_tokens(actual);
    if et.is_empty() && at.is_empty() {
        return 1.0;
    }
    if et.is_empty() || at.is_empty() {
        return 0.0;
    }
    let common: usize = et.iter().filter(|t| at.contains(t)).count();
    if common == 0 {
        return 0.0;
    }
    let p = common as f64 / at.len() as f64;
    let r = common as f64 / et.len() as f64;
    2.0 * p * r / (p + r)
}

/// Token F1 of `actual` against `expected`, case-folded first.
pub(crate) fn token_f1(expected: &str, actual: &str) -> f64 {
    token_f1_lower(&expected.to_lowercase(), &actual.to_lowercase())
}

/// The best token F1 of `actual` against any of `answers`: the F1 a benchmark with several
/// accepted answers reports.
pub(crate) fn best_token_f1(answers: &[String], actual: &str) -> f64 {
    answers
        .iter()
        .map(|a| token_f1(a, actual))
        .fold(0.0_f64, f64::max)
}

/// Whether `response` answers with `expected`: it contains it, case-folded, or its token F1 against
/// it is at least 0.5 (answers are short entities a model may wrap in a sentence). An expected
/// answer with no tokens (punctuation only) is right only as a substring; an empty one is a
/// substring of anything. This is `multihop`'s and `hotpotqa`'s rule, F1 computed as they compute
/// it, which differs from [`token_f1`] only where the expected answer has no tokens.
pub(crate) fn contains_or_f1(response: &str, expected: &str) -> bool {
    let (resp, exp) = (response.to_lowercase(), expected.to_lowercase());
    if resp.contains(&exp) {
        return true;
    }
    let (et, rt) = (normalize_tokens(&exp), normalize_tokens(&resp));
    if et.is_empty() {
        return false;
    }
    let common = et.iter().filter(|t| rt.contains(t)).count();
    if common == 0 {
        return false;
    }
    let p = common as f64 / rt.len().max(1) as f64;
    let r = common as f64 / et.len() as f64;
    2.0 * p * r / (p + r) >= 0.5
}

/// Whether `response` answers with any of `answers`, by the rule `musique` and `msmarco` apply to
/// each: a substring, or token F1 at least 0.5 as [`token_f1`] computes it.
pub(crate) fn any_contains_or_f1(response: &str, answers: &[&str]) -> bool {
    let resp = response.to_lowercase();
    answers.iter().any(|a| {
        let a = a.to_lowercase();
        resp.contains(&a) || token_f1_lower(&a, &resp) >= 0.5
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The benchmarks' scorers as they were written in each, before this module: the reference
    // every shared rule is held to.

    fn multihop_evaluate(response: &str, expected: &str) -> f64 {
        let resp_lower = response.to_lowercase();
        let exp_lower = expected.to_lowercase();
        if resp_lower.contains(&exp_lower) {
            return 1.0;
        }
        let exp_tokens = normalize_tokens(&exp_lower);
        let resp_tokens = normalize_tokens(&resp_lower);
        if exp_tokens.is_empty() {
            return 0.0;
        }
        let common: usize = exp_tokens
            .iter()
            .filter(|t| resp_tokens.contains(t))
            .count();
        if common == 0 {
            return 0.0;
        }
        let precision = common as f64 / resp_tokens.len().max(1) as f64;
        let recall = common as f64 / exp_tokens.len() as f64;
        let f1 = 2.0 * precision * recall / (precision + recall);
        if f1 >= 0.5 { 1.0 } else { 0.0 }
    }

    fn musique_evaluate(response: &str, expected: &str, aliases: &[String]) -> f64 {
        let resp_lower = response.to_lowercase();
        let all: Vec<&str> = std::iter::once(expected)
            .chain(aliases.iter().map(|a| a.as_str()))
            .collect();
        for ans in &all {
            let ans_lower = ans.to_lowercase();
            if resp_lower.contains(&ans_lower) {
                return 1.0;
            }
            if token_f1_lower(&ans_lower, &resp_lower) >= 0.5 {
                return 1.0;
            }
        }
        0.0
    }

    fn msmarco_evaluate(response: &str, answers: &[String]) -> f64 {
        let resp_lower = response.to_lowercase();
        for ans in answers {
            let ans_lower = ans.to_lowercase();
            if resp_lower.contains(&ans_lower) {
                return 1.0;
            }
            if token_f1_lower(&ans_lower, &resp_lower) >= 0.5 {
                return 1.0;
            }
        }
        0.0
    }

    /// Strings for the comparison: every short combination of a few words, case, punctuation and
    /// the empty and punctuation-only edges.
    fn cases() -> Vec<String> {
        let parts = [
            "",
            "Paris",
            "paris",
            "Barack Obama",
            "obama",
            "1912",
            "the",
            "France.",
            "!",
            ", ",
            "New York City",
            "york",
        ];
        let mut out: Vec<String> = Vec::new();
        for a in parts {
            out.push(a.to_string());
            for b in parts {
                out.push(format!("{a} {b}"));
                for c in ["", " is the answer", "."] {
                    out.push(format!("{a}{b}{c}"));
                }
            }
        }
        out
    }

    #[test]
    fn the_shared_rule_scores_as_multihop_and_hotpotqa_did() {
        let cs = cases();
        for r in &cs {
            for e in &cs {
                let was = multihop_evaluate(r, e) == 1.0;
                assert_eq!(contains_or_f1(r, e), was, "response {r:?}, expected {e:?}");
            }
        }
    }

    #[test]
    fn the_shared_rule_scores_as_musique_and_msmarco_did() {
        let cs = cases();
        let lists: Vec<Vec<String>> = vec![
            vec!["Paris".into()],
            vec!["Barack Obama".into(), "Obama".into()],
            vec!["".into()],
            vec!["!".into(), "1912".into()],
            vec![],
        ];
        for r in &cs {
            for answers in &lists {
                let refs: Vec<&str> = answers.iter().map(String::as_str).collect();
                let now = any_contains_or_f1(r, &refs);
                assert_eq!(
                    now,
                    msmarco_evaluate(r, answers) == 1.0,
                    "msmarco {r:?} {answers:?}"
                );
                if let Some((first, rest)) = answers.split_first() {
                    assert_eq!(
                        now,
                        musique_evaluate(r, first, rest) == 1.0,
                        "musique {r:?} {answers:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_shared_f1_is_the_one_each_benchmark_reported() {
        let cs = cases();
        for a in &cs {
            for b in &cs {
                // multihop's and hotpotqa's token_f1 case-folded both sides, then the shared rule.
                assert_eq!(
                    token_f1(a, b),
                    token_f1_lower(&a.to_lowercase(), &b.to_lowercase())
                );
            }
        }
        assert_eq!(best_token_f1(&["paris".into(), "lyon".into()], "Lyon"), 1.0);
        assert_eq!(best_token_f1(&[], "x"), 0.0);
    }
}
