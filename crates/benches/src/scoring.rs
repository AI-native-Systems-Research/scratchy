// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! How the dataset benchmarks score an answer, in one place: the multi-document QA rule
//! (`multihop`, `hotpotqa`, `musique`, `msmarco`, and `bench serve`'s dataset workloads), right or
//! wrong, and the token F1 they report beside it.

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

/// Whether `answer` gives one of `accepted`: all of that answer's words, in order and side by
/// side, are among the answer's ([`normalize_tokens`], case-folded), so "The answer is **Paris**."
/// gives "Paris". Words, never characters: "I don't know" does not give "no", nor "Berlin" "in", nor
/// "1922" "22". Every word: "John Adams" does not give "John Quincy Adams". An accepted answer with
/// no words, only punctuation, is given by none.
pub(crate) fn gives_any(answer: &str, accepted: &[&str]) -> bool {
    let words = normalize_tokens(&answer.to_lowercase());
    accepted.iter().any(|a| {
        let wanted = normalize_tokens(&a.to_lowercase());
        !wanted.is_empty() && (words.windows(wanted.len())).any(|w| w == wanted.as_slice())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// An answer is right when it names an accepted answer, whatever else it says, and wrong when
    /// it names part of one or something that only shares its letters.
    #[test]
    fn an_answer_is_right_when_it_names_an_accepted_answer_in_words() {
        for (answer, accepted) in [
            ("Paris", "Paris"),
            ("The answer is **Paris**.", "paris"),
            ("February\u{202f}7,\u{202f}2018", "February 7, 2018"),
            ("\"Niagara\" (the wedding episode)", "``Niagara ''"),
            ("Born in 1922.", "1922"),
        ] {
            assert!(
                gives_any(answer, &[accepted]),
                "{answer:?} gives {accepted:?}"
            );
        }
        for (answer, accepted) in [
            ("I don't know", "no"),
            ("Berlin", "in"),
            ("1922", "22"),
            ("John Adams", "John Quincy Adams"),
            ("1763", "10 February 1763"),
            ("Barack Clinton", "Barack Obama"),
            ("!", "!"),
            ("Paris", ""),
        ] {
            assert!(
                !gives_any(answer, &[accepted]),
                "{answer:?} gives {accepted:?}"
            );
        }
        assert!(
            gives_any("Obama", &["Barack Obama", "Obama"]),
            "any one of them"
        );
        assert!(!gives_any("Obama", &[]));
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
