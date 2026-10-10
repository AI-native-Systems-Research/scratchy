//! Syntax colouring for the forward DSL (Python syntax), shared by the
//! architectures page and the landing page's snippet. The token classes are
//! the `.c-*` rules in styles.css.

use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Class {
    Comment,
    Decorator,
    Str,
    Keyword,
    Call,
    Number,
}

impl Class {
    fn css(self) -> &'static str {
        match self {
            Class::Comment => "c-cm",
            Class::Decorator => "c-at",
            Class::Str => "c-s",
            Class::Keyword => "c-kw",
            Class::Call => "c-fn",
            Class::Number => "c-n",
        }
    }
}

const KEYWORDS: [&str; 17] = [
    "def", "for", "in", "if", "elif", "else", "while", "return", "import", "from", "as", "and",
    "or", "not", "True", "False", "None",
];

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// One line as (class, text) runs, plain text unclassed. Left to right, first
/// match wins: a comment runs to the end of the line, `@name` is a decorator, a
/// double-quoted string honours backslash escapes, a keyword is a whole word,
/// any other name followed by `(` is a call, and a word starting with a digit
/// is a number.
pub fn tokens(line: &str) -> Vec<(Option<Class>, &str)> {
    let b = line.as_bytes();
    let mut out: Vec<(Option<Class>, &str)> = Vec::new();
    let (mut plain, mut i) = (0, 0);
    let emit = |out: &mut Vec<_>, plain: &mut usize, start: usize, end: usize, class: Class| {
        if *plain < start {
            out.push((None, &line[*plain..start]));
        }
        out.push((Some(class), &line[start..end]));
        *plain = end;
    };
    while i < b.len() {
        let c = b[i];
        let at_boundary = i == 0 || !is_word(b[i - 1]);
        if c == b'#' {
            emit(&mut out, &mut plain, i, b.len(), Class::Comment);
            i = b.len();
        } else if c == b'@'
            && b.get(i + 1)
                .is_some_and(|&n| n.is_ascii_alphabetic() || n == b'_')
        {
            let end = (i + 1..b.len())
                .find(|&j| !is_word(b[j]))
                .unwrap_or(b.len());
            emit(&mut out, &mut plain, i, end, Class::Decorator);
            i = end;
        } else if c == b'"' {
            let mut j = i + 1;
            let close = loop {
                match b.get(j) {
                    None => break None,
                    Some(b'"') => break Some(j + 1),
                    Some(b'\\') if j + 1 < b.len() => j += 2,
                    Some(_) => j += 1,
                }
            };
            match close {
                Some(end) => {
                    emit(&mut out, &mut plain, i, end, Class::Str);
                    i = end;
                }
                None => i += 1,
            }
        } else if at_boundary && (c.is_ascii_alphabetic() || c == b'_') {
            let end = (i..b.len()).find(|&j| !is_word(b[j])).unwrap_or(b.len());
            let word = &line[i..end];
            if KEYWORDS.contains(&word) {
                emit(&mut out, &mut plain, i, end, Class::Keyword);
            } else if line[end..].trim_start().starts_with('(') {
                emit(&mut out, &mut plain, i, end, Class::Call);
            }
            i = end;
        } else if at_boundary && c.is_ascii_digit() {
            let mut end = (i..b.len())
                .find(|&j| !is_word(b[j]) && b[j] != b'.')
                .unwrap_or(b.len());
            // A number ends on a word character: a trailing '.' is punctuation.
            while b[end - 1] == b'.' {
                end -= 1;
            }
            emit(&mut out, &mut plain, i, end, Class::Number);
            i = end;
        } else {
            i += 1;
        }
    }
    if plain < b.len() {
        out.push((None, &line[plain..]));
    }
    out
}

pub fn line(text: &str) -> Element {
    rsx! {
        for (class, run) in tokens(text) {
            if let Some(class) = class {
                span { class: class.css(), "{run}" }
            } else {
                "{run}"
            }
        }
    }
}

/// Several lines, newline-separated, for a <pre>-like container.
pub fn block(text: &str) -> Element {
    rsx! {
        for (n, l) in text.lines().enumerate() {
            if n > 0 { "\n" }
            {line(l)}
        }
    }
}
