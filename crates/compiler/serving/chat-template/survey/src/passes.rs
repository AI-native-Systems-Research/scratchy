// SPDX-License-Identifier: Apache-2.0
//! The analysis passes behind ../SPEC.md's numbers.
//!
//! Each pass answers one falsifiable question about rendered OUTPUT. None of them
//! inspect Jinja source: the whole premise of the spec is that the encoding is
//! diverse while the format is not, so reading the encoding would beg the question.
//!
//! ⛔ THE METHOD THAT MAKES THIS MECHANICAL: message contents are unique sentinels
//! (`@@C0@@`, `@@C1@@`, …). Whatever surrounds a sentinel in the output IS the
//! template's literal scaffolding, so cutting each render at the sentinels turns
//! classification into arithmetic instead of 64 KB of reading.

use minijinja::{Environment, Value};
use std::collections::{BTreeMap, BTreeSet};

/// ⛔ MUST MATCH `crates/serving/api/src/chat_template.rs::build_env` EXACTLY.
/// trim_blocks/lstrip_blocks change the rendered bytes, pycompat decides whether
/// `.startswith` resolves at all, and a live clock makes the survey
/// unreproducible. A divergence here means every number below describes a
/// language production does not render.
pub fn build_env(src: &str) -> Result<Environment<'static>, String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |m: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                m,
            ))
        },
    );
    // Frozen clock. A real one would make every run differ.
    env.add_function("strftime_now", |_f: String| "01 Jan 2026".to_string());
    env.add_template_owned("chat", src.to_owned())
        .map_err(|e| e.to_string())?;
    Ok(env)
}

pub fn msg(role: &str, content: &str) -> serde_json::Value {
    serde_json::json!({ "role": role, "content": content })
}

pub fn tools() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "search", "description": "Look things up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
    }])
}

pub fn render(
    env: &Environment,
    messages: &[serde_json::Value],
    gp: bool,
    with_tools: bool,
) -> Result<String, String> {
    let mut c: BTreeMap<String, Value> = BTreeMap::new();
    c.insert("messages".into(), Value::from_serialize(messages));
    c.insert("add_generation_prompt".into(), Value::from(gp));
    if with_tools {
        c.insert("tools".into(), Value::from_serialize(tools()));
    }
    c.insert("bos_token".into(), Value::from("<BOS>"));
    c.insert("eos_token".into(), Value::from("<EOS>"));
    c.insert("date_string".into(), Value::from("01 Jan 2026"));
    env.get_template("chat")
        .map_err(|e| e.to_string())?
        .render(Value::from_serialize(&c))
        .map_err(|e| e.to_string())
}

/// Longest common byte prefix.
fn common(a: &str, b: &str) -> usize {
    let n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    let mut n = n;
    while !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Sentinel-cut a render into the byte ranges of each message's content.
/// `None` means a content went missing or was reordered, which is itself a result.
fn cuts(out: &str, n: usize) -> Option<Vec<(usize, usize)>> {
    let mut v = Vec::with_capacity(n);
    let mut from = 0usize;
    for i in 0..n {
        let pat = format!("@@C{i}@@");
        let rel = out[from..].find(&pat)?;
        let s = from + rel;
        v.push((s, s + pat.len()));
        from = s + pat.len();
    }
    Some(v)
}

fn sentinel_msgs(seq: &[&str]) -> Vec<serde_json::Value> {
    seq.iter()
        .enumerate()
        .map(|(i, r)| msg(r, &format!("@@C{i}@@")))
        .collect()
}

/// Role sequences used by the grammar passes. Includes repeats and a trailing
/// system message: the degenerate shapes are where two templates turned out to
/// coalesce turns, and a real client can send them.
pub const SEQUENCES: &[&[&str]] = &[
    &["user"],
    &["user", "assistant"],
    &["user", "assistant", "user"],
    &["user", "assistant", "user", "assistant"],
    &["user", "assistant", "user", "assistant", "user"],
    &["system", "user"],
    &["system", "user", "assistant"],
    &["system", "user", "assistant", "user"],
    &["system", "assistant"],
    &["user", "user"],
    &["assistant", "assistant"],
    &["assistant", "user"],
];

// ---------------------------------------------------------------------------
// Pass: hypothesis H
// ---------------------------------------------------------------------------

pub struct FitResult {
    pub name: String,
    pub holds: bool,
    pub observations: usize,
    pub pairs: usize,
    pub notes: Vec<String>,
}

/// SPEC.md §1. H: the bytes between two adjacent contents are a function of the
/// two roles ALONE. A disagreement for one role pair, or a dropped content, is a
/// counterexample.
pub fn fit(name: &str, src: &str) -> FitResult {
    let mut r = FitResult {
        name: name.into(),
        holds: true,
        observations: 0,
        pairs: 0,
        notes: Vec::new(),
    };
    let Ok(env) = build_env(src) else {
        r.holds = false;
        r.notes.push("template failed to parse".into());
        return r;
    };

    let mut seps: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    let mut heads: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut tails: BTreeMap<(String, bool), BTreeSet<String>> = BTreeMap::new();

    for seq in SEQUENCES {
        for gp in [false, true] {
            let msgs = sentinel_msgs(seq);
            let Ok(out) = render(&env, &msgs, gp, false) else {
                continue;
            };
            let Some(cu) = cuts(&out, seq.len()) else {
                r.holds = false;
                r.notes.push(format!(
                    "seq={seq:?} gp={gp}: a message's content is MISSING or REORDERED\n    {out:?}"
                ));
                continue;
            };
            r.observations += 1;
            heads
                .entry(seq[0].into())
                .or_default()
                .insert(out[..cu[0].0].into());
            for i in 0..seq.len() - 1 {
                seps.entry((seq[i].into(), seq[i + 1].into()))
                    .or_default()
                    .insert(out[cu[i].1..cu[i + 1].0].into());
            }
            tails
                .entry((seq[seq.len() - 1].into(), gp))
                .or_default()
                .insert(out[cu[seq.len() - 1].1..].into());
        }
    }

    r.pairs = seps.len();
    for ((a, b), v) in &seps {
        if v.len() > 1 {
            r.holds = false;
            r.notes.push(format!(
                "separator for {a:?}->{b:?} is not unique, {} variants:\n{}",
                v.len(),
                v.iter()
                    .map(|s| format!("      {s:?}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    for (k, v) in &heads {
        if v.len() > 1 {
            r.holds = false;
            r.notes.push(format!(
                "head for first-role {k:?} is not unique ({})",
                v.len()
            ));
        }
    }
    for ((role, gp), v) in &tails {
        if v.len() > 1 {
            r.holds = false;
            r.notes.push(format!(
                "tail for {role:?} gp={gp} is not unique ({})",
                v.len()
            ));
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Pass: does the separator table factor into per-role open/close?
// ---------------------------------------------------------------------------

pub struct FactorResult {
    pub name: String,
    pub factors: bool,
    pub open: BTreeMap<String, String>,
    pub close: BTreeMap<String, String>,
    pub residuals: Vec<String>,
}

/// SPEC.md §1. `sep[a→b] == close[a] + open[b]` is what makes a declaration
/// O(roles) instead of O(roles²).
///
/// ⛔ DO NOT ESTIMATE `close[a]` AS THE ROW'S COMMON PREFIX. That over-assigns —
/// it swallows the shared head of `open[b]` — and reports almost nothing as
/// factoring. (It did, on the first attempt: 1 of 20 instead of 21 of 23.) The
/// factorisation gives each ROW a single split offset `k_a`, so search the offsets
/// and check cross-row consistency.
pub fn factor(name: &str, src: &str) -> FactorResult {
    let mut r = FactorResult {
        name: name.into(),
        factors: false,
        open: BTreeMap::new(),
        close: BTreeMap::new(),
        residuals: Vec::new(),
    };
    let Ok(env) = build_env(src) else {
        r.residuals.push("template failed to parse".into());
        return r;
    };

    let mut seps: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for seq in SEQUENCES {
        let msgs = sentinel_msgs(seq);
        let Ok(out) = render(&env, &msgs, false, false) else {
            continue;
        };
        let Some(cu) = cuts(&out, seq.len()) else {
            continue;
        };
        for i in 0..seq.len() - 1 {
            seps.entry((seq[i].into(), seq[i + 1].into()))
                .or_default()
                .insert(out[cu[i].1..cu[i + 1].0].into());
        }
    }
    // Ambiguous cells are H failures, already reported by `fit`; skip them here.
    let flat: BTreeMap<(String, String), String> = seps
        .iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(k, v)| (k.clone(), v.iter().next().unwrap().clone()))
        .collect();

    let roles: BTreeSet<String> = flat
        .keys()
        .flat_map(|(a, b)| [a.clone(), b.clone()])
        .collect();
    let rows: Vec<String> = roles
        .iter()
        .filter(|x| flat.keys().any(|(a, _)| a == *x))
        .cloned()
        .collect();
    let cands: Vec<Vec<usize>> = rows
        .iter()
        .map(|a| {
            let cells: Vec<&String> = flat
                .iter()
                .filter(|((x, _), _)| x == a)
                .map(|(_, v)| v)
                .collect();
            let lim = cells.iter().map(|c| c.len()).min().unwrap_or(0);
            (0..=lim)
                .filter(|k| cells.iter().all(|c| c.is_char_boundary(*k)))
                .collect()
        })
        .collect();

    fn search(
        i: usize,
        rows: &[String],
        cands: &[Vec<usize>],
        flat: &BTreeMap<(String, String), String>,
        close: &mut BTreeMap<String, String>,
        open: &mut BTreeMap<String, String>,
    ) -> bool {
        if i == rows.len() {
            return true;
        }
        let a = &rows[i];
        for &k in &cands[i] {
            let mut added: Vec<String> = Vec::new();
            let mut ok = true;
            for ((_, b), v) in flat.iter().filter(|((x, _), _)| x == a) {
                let o = v[k..].to_string();
                match open.get(b) {
                    Some(prev) if prev != &o => {
                        ok = false;
                        break;
                    }
                    Some(_) => {}
                    None => {
                        open.insert(b.clone(), o);
                        added.push(b.clone());
                    }
                }
            }
            if ok {
                let first = flat
                    .iter()
                    .find(|((x, _), _)| x == a)
                    .map(|(_, v)| v)
                    .unwrap();
                close.insert(a.clone(), first[..k].to_string());
                if search(i + 1, rows, cands, flat, close, open) {
                    return true;
                }
                close.remove(a);
            }
            for b in added {
                open.remove(&b);
            }
        }
        false
    }

    if !search(0, &rows, &cands, &flat, &mut r.close, &mut r.open) {
        r.residuals
            .push("no consistent (open, close) assignment exists".into());
        return r;
    }
    for ((a, b), v) in &flat {
        let predicted = format!(
            "{}{}",
            r.close.get(a).cloned().unwrap_or_default(),
            r.open.get(b).cloned().unwrap_or_default()
        );
        if &predicted != v {
            r.residuals.push(format!(
                "sep[{a}->{b}] = {v:?} but close+open = {predicted:?}"
            ));
        }
    }
    r.factors = r.residuals.is_empty();
    r
}

// ---------------------------------------------------------------------------
// Pass: monotonicity
// ---------------------------------------------------------------------------

pub struct MonoResult {
    pub monotonic: bool,
    pub fixed_tail: Option<String>,
    pub notes: Vec<String>,
}

/// Is the render append-only as the conversation grows? A `no` means something
/// non-local rewrites bytes already in the prompt — which also makes prefix KV
/// cache reuse unsound, so this is worth knowing beyond the DSL.
///
/// ⛔ A FIXED TRAILING SUFFIX IS NOT A VIOLATION, AND NAIVE `starts_with` SAYS IT
/// IS. Some templates emit `eos_token` when `add_generation_prompt` is off, so that
/// tail sits after the last block and reappears further along in the longer
/// render. Discriminator: with `c` the longest common prefix, the break is a
/// benign re-emitted tail iff `a[c..]` is also a SUFFIX of `b` — those bytes did
/// not change, they moved behind the new block. Getting this wrong flags five
/// templates instead of three; over-correcting (accepting any single break) hides
/// the two real ones.
pub fn monotonic(src: &str, with_tools: bool) -> MonoResult {
    let mut r = MonoResult {
        monotonic: true,
        fixed_tail: None,
        notes: Vec::new(),
    };
    let Ok(env) = build_env(src) else {
        r.monotonic = false;
        r.notes.push("template failed to parse".into());
        return r;
    };

    let chain: Vec<Vec<serde_json::Value>> = vec![
        vec![msg("user", "U1")],
        vec![msg("user", "U1"), msg("assistant", "A1")],
        vec![msg("user", "U1"), msg("assistant", "A1"), msg("user", "U2")],
        vec![
            msg("user", "U1"),
            msg("assistant", "A1"),
            msg("user", "U2"),
            msg("assistant", "A2"),
        ],
    ];
    let rs: Vec<Result<String, String>> = chain
        .iter()
        .map(|m| render(&env, m, false, with_tools))
        .collect();

    for i in 0..rs.len() - 1 {
        let (Ok(a), Ok(b)) = (&rs[i], &rs[i + 1]) else {
            break;
        };
        if b.starts_with(a.as_str()) {
            continue;
        }
        let c = common(a, b);
        let t = &a[c..];
        if b.ends_with(t) {
            r.fixed_tail = Some(t.to_string());
        } else {
            r.monotonic = false;
            r.notes.push(format!(
                "n={} is not a prefix of n={} and the divergent bytes are not \
                 re-emitted at the end — the prompt was rewritten:\n    \
                 n={} from @{c}: {:?}\n    n={} from @{c}: {:?}",
                i + 1,
                i + 2,
                i + 1,
                t,
                i + 2,
                &b[c..]
            ));
            break;
        }
    }
    r
}

/// Where a tools block lands, and whether it moves as the conversation grows.
pub fn tools_placement(src: &str) -> String {
    let Ok(env) = build_env(src) else {
        return "parse failed".into();
    };
    let short = [msg("user", "U1")];
    let long = [msg("user", "U1"), msg("assistant", "A1"), msg("user", "U2")];
    let off = |m: &[serde_json::Value]| -> Option<usize> {
        let a = render(&env, m, false, false).ok()?;
        let b = render(&env, m, false, true).ok()?;
        if a == b {
            return None;
        }
        Some(common(&a, &b))
    };
    match (off(&short), off(&long)) {
        (None, None) => "ignores tools".into(),
        (Some(a), Some(c)) if a == c => format!("head (@{a})"),
        (Some(a), Some(c)) => format!("MOVES (@{a} -> @{c})"),
        _ => "mixed".into(),
    }
}

// ---------------------------------------------------------------------------
// Pass: the derived grammar table
// ---------------------------------------------------------------------------

/// Print head/sep/tail for one template — the table a declaration is read off.
pub fn table(src: &str) -> Vec<String> {
    let Ok(env) = build_env(src) else {
        return vec!["parse failed".into()];
    };
    let mut heads: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut tails: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for seq in SEQUENCES {
        for gp in [false, true] {
            let msgs = sentinel_msgs(seq);
            let Ok(out) = render(&env, &msgs, gp, false) else {
                continue;
            };
            let Some(cu) = cuts(&out, seq.len()) else {
                continue;
            };
            heads
                .entry(seq[0].into())
                .or_default()
                .insert(out[..cu[0].0].into());
            for i in 0..seq.len() - 1 {
                seps.entry(format!("{}->{}", seq[i], seq[i + 1]))
                    .or_default()
                    .insert(out[cu[i].1..cu[i + 1].0].into());
            }
            tails
                .entry(format!("{} gp={gp}", seq[seq.len() - 1]))
                .or_default()
                .insert(out[cu[seq.len() - 1].1..].into());
        }
    }

    let mut lines = Vec::new();
    for (label, m) in [("head", &heads), ("sep", &seps), ("tail", &tails)] {
        for (k, v) in m {
            let flag = if v.len() > 1 { "  !AMBIG" } else { "" };
            for s in v {
                lines.push(format!("  {label}[{k}] = {s:?}{flag}"));
            }
        }
    }
    lines
}

// ---------------------------------------------------------------------------
// Pass: the non-grammar dimensions
// ---------------------------------------------------------------------------

pub struct Tally {
    pub name: String,
    pub bytes: usize,
    pub default_system: bool,
    pub tools: bool,
    pub tool_result: &'static str,
    pub reasoning: &'static str,
    pub list_content: &'static str,
    pub two_tool_results: &'static str,
}

/// SPEC.md §4 and §6: everything the turn grammar does not cover.
pub fn tally(name: &str, src: &str) -> Tally {
    let mut t = Tally {
        name: name.into(),
        bytes: src.len(),
        default_system: false,
        tools: false,
        tool_result: "?",
        reasoning: "?",
        list_content: "?",
        two_tool_results: "?",
    };
    let Ok(env) = build_env(src) else { return t };

    let base = [msg("user", "<<U1>>")];
    let plain = render(&env, &base, true, false);
    if let Ok(p) = &plain {
        t.default_system = p.contains("system") && !p.contains("<<S>>");
        t.tools = render(&env, &base, true, true).ok().as_ref() != Some(p);
    }

    let call = serde_json::json!({
        "role": "assistant", "content": "",
        "tool_calls": [{"type": "function", "id": "call_1",
            "function": {"name": "search", "arguments": "{\"q\": \"<<ARG>>\"}"}}]
    });
    let tr = render(
        &env,
        &[msg("user", "<<U1>>"), call.clone(), msg("tool", "<<TRES>>")],
        true,
        true,
    );
    t.tool_result = match &tr {
        Err(_) => "rejects",
        Ok(s) if !s.contains("<<TRES>>") => "DROPS",
        Ok(s) if s.contains("tool_response") || s.contains("tool_output") => "wrap+re-role",
        Ok(s) if s.contains("tool") => "tool role",
        Ok(_) => "bare",
    };

    let two = render(
        &env,
        &[
            msg("user", "<<U1>>"),
            call,
            msg("tool", "<<T1>>"),
            msg("tool", "<<T2>>"),
        ],
        true,
        true,
    );
    t.two_tool_results = match &two {
        Err(_) => "rejects",
        Ok(s) if s.contains("<<T1>>") && s.contains("<<T2>>") => "both kept",
        Ok(_) => "LOSES one",
    };

    let think = render(
        &env,
        &[
            msg("user", "<<U1>>"),
            msg("assistant", "<think>\n<<THINK>>\n</think>\n\n<<A1>>"),
            msg("user", "<<U2>>"),
        ],
        true,
        false,
    );
    t.reasoning = match &think {
        Err(_) => "rejects",
        Ok(s) if !s.contains("<<THINK>>") => "transforms",
        Ok(_) => "passthrough",
    };

    let parts = serde_json::json!({"role": "user", "content": [
        {"type": "text", "text": "<<P1>>"}, {"type": "image"},
        {"type": "text", "text": "<<P2>>"}]});
    let lc = render(&env, &[parts], true, false);
    t.list_content = match &lc {
        Err(_) => "rejects",
        Ok(s) if s.contains("{\"") || s.contains("{'") => "JSON-dumped",
        Ok(s) if s.contains("<<P1>>") && s.contains("<<P2>>") => "handled",
        Ok(s) if !s.contains("<<P1>>") && !s.contains("<<P2>>") => "SILENT EMPTY",
        Ok(_) => "partial",
    };
    t
}
