// SPDX-License-Identifier: Apache-2.0
//! Re-derive the measurements behind ../SPEC.md.
//!
//! ```text
//! cargo run -p scratchy-chat-template-survey -- sync       # fetch + verify pins
//! cargo run -p scratchy-chat-template-survey -- coverage   # arch families vs fixtures
//! cargo run -p scratchy-chat-template-survey -- fit        # hypothesis H
//! cargo run -p scratchy-chat-template-survey -- factor     # per-role open/close
//! cargo run -p scratchy-chat-template-survey -- monotonic  # append-only + tools placement
//! cargo run -p scratchy-chat-template-survey -- table [name…]
//! cargo run -p scratchy-chat-template-survey -- tally      # the non-grammar dimensions
//! cargo run -p scratchy-chat-template-survey -- all        # everything but `table`
//! ```
//!
//! `sync` needs the network; every other pass reads the cache under
//! `target/chat-survey-cache/`.

mod fixtures;
mod passes;

use fixtures::Outcome;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("all");
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();

    match cmd {
        "sync" => sync(),
        "coverage" => coverage(),
        "fit" => fit(),
        "factor" => factor(),
        "monotonic" => monotonic(),
        "table" => table(&rest),
        "tally" => tally(),
        "all" => {
            coverage();
            fit();
            factor();
            monotonic();
            tally();
        }
        other => {
            eprintln!("unknown pass `{other}`");
            eprintln!("known: sync, coverage, fit, factor, monotonic, table, tally, all");
            std::process::exit(2);
        }
    }
}

fn require_cache() -> Vec<(&'static str, String)> {
    let loaded = fixtures::loaded();
    if loaded.is_empty() {
        eprintln!(
            "no templates cached in {}\nrun `cargo run -p scratchy-chat-template-survey -- sync` first",
            fixtures::cache_dir().display()
        );
        std::process::exit(1);
    }
    loaded
}

fn sync() {
    println!("cache: {}\n", fixtures::cache_dir().display());
    let mut ok = 0usize;
    let mut drift = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    let mut unpinned: Vec<(&str, String, usize)> = Vec::new();

    for (f, outcome) in fixtures::sync() {
        match outcome {
            Outcome::Cached => {
                ok += 1;
                println!("  {:<13} cached", f.name);
            }
            Outcome::Fetched { repo, sha } => {
                ok += 1;
                println!("  {:<13} fetched  {repo}", f.name);
                if f.sha256.is_empty() {
                    let bytes =
                        std::fs::read(fixtures::cache_dir().join(format!("{}.jinja", f.name)))
                            .map(|b| b.len())
                            .unwrap_or(0);
                    unpinned.push((f.name, sha, bytes));
                }
            }
            Outcome::Drift { repo, got } => {
                drift += 1;
                println!(
                    "  {:<13} DRIFT    {repo}\n      pinned {}\n      got    {got}",
                    f.name, f.sha256
                );
            }
            Outcome::BadLengthPin { sha, got } => {
                drift += 1;
                println!(
                    "  {:<13} BAD PIN  hash matches nothing in the table\n                           bytes pinned {} but got {got}\n      sha256 {sha}",
                    f.name, f.bytes
                );
            }
            Outcome::Unavailable => {
                missing.push(f.name);
                println!("  {:<13} unavailable (gated on every listed repo)", f.name);
            }
        }
    }

    println!(
        "\n  {ok} available, {drift} drifted, {} unavailable",
        missing.len()
    );
    if !missing.is_empty() {
        println!("  unavailable: {}", missing.join(", "));
    }
    if !unpinned.is_empty() {
        println!("\n⚠ unpinned fixtures — paste these into fixtures.rs so the sample is frozen:");
        for (name, sha, bytes) in &unpinned {
            println!("    {name:<13} bytes: {bytes:<6} sha256: \"{sha}\"");
        }
    }
    if drift > 0 {
        println!(
            "\n⚠ DRIFT means upstream edited a template after the survey ran. The numbers in\n  \
             SPEC.md describe the PINNED bytes. Re-run the passes and update the spec together,\n  \
             or restore the pin — do not silently accept the new template."
        );
    }
}

/// Reconcile the fixture table against the real `configs/` tree.
///
/// ⛔ TEMPLATES AND ARCH FAMILIES ARE NOT THE SAME COUNT, and conflating them
/// overstates coverage. Three fixtures (smollm2, tinyllama, llama32) all live under
/// the `llama` arch; two more pairs share an arch. And a fixture whose template
/// could not be fetched covers nothing, however much it is declared here. So this
/// reports families with an ACTUALLY CACHED template, separately from the fixture
/// count.
fn coverage() {
    let dir = fixtures::configs_dir();
    let families = fixtures::arch_families(&dir);
    if families.is_empty() {
        println!("== coverage ==\n  cannot read {}\n", dir.display());
        return;
    }

    let cached: std::collections::BTreeSet<&str> =
        fixtures::loaded().iter().map(|(n, _)| *n).collect();
    // Only fixtures whose template is on disk count towards a family.
    let measured: std::collections::BTreeSet<&str> = fixtures::FIXTURES
        .iter()
        .filter(|f| cached.contains(f.name))
        .map(|f| f.arch)
        .collect();
    let declared: std::collections::BTreeSet<&str> =
        fixtures::FIXTURES.iter().map(|f| f.arch).collect();
    let excused: std::collections::BTreeMap<&str, &str> = fixtures::NOT_CHAT_MODELS
        .iter()
        .chain(fixtures::SHARED_LINEAGE.iter())
        .copied()
        .collect();

    let mut n_measured = 0usize;
    let mut unreachable: Vec<&String> = Vec::new();
    let mut gaps: Vec<&String> = Vec::new();
    println!("== coverage ==");
    for fam in &families {
        let f = fam.as_str();
        if measured.contains(f) {
            n_measured += 1;
        } else if declared.contains(f) {
            // A fixture exists but its template could not be obtained.
            unreachable.push(fam);
        } else if let Some(why) = excused.get(f) {
            println!("  {fam:<16} excused — {why}");
        } else {
            gaps.push(fam);
        }
    }
    for fam in &unreachable {
        println!("  {fam:<16} UNREACHABLE — fixture declared, template not obtainable");
    }

    let distinct: std::collections::BTreeSet<String> = fixtures::loaded()
        .iter()
        .map(|(_, src)| {
            use sha2::{Digest, Sha256};
            Sha256::digest(src.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        })
        .collect();

    println!(
        "\n  {} templates cached ({} distinct by content)",
        cached.len(),
        distinct.len()
    );
    println!(
        "  covering {n_measured} of {} arch families directly, {} excused, {} unreachable",
        families.len(),
        families.len() - n_measured - unreachable.len() - gaps.len(),
        unreachable.len()
    );
    if !gaps.is_empty() {
        println!(
            "  ⚠ UNACCOUNTED: {}",
            gaps.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    add a fixture, or list it in NOT_CHAT_MODELS / SHARED_LINEAGE with a reason."
        );
    }
    println!();
}

fn fit() {
    let loaded = require_cache();
    println!("== hypothesis H: sep(role_a, role_b) depends on the two roles alone ==");
    let mut holds = 0usize;
    let mut failures = Vec::new();
    for (name, src) in &loaded {
        let r = passes::fit(name, src);
        if r.holds {
            holds += 1;
            println!(
                "  {:<13} HOLDS  ({} renders, {} role-pair separators)",
                r.name, r.observations, r.pairs
            );
        } else {
            println!("  {:<13} FAILS  ({} clean renders)", r.name, r.observations);
            failures.push(r);
        }
    }
    println!("\n  H holds for {holds}/{}", loaded.len());
    for r in &failures {
        println!("\n  --- {} ---", r.name);
        for n in &r.notes {
            println!("  {n}");
        }
    }
    println!();
}

fn factor() {
    let loaded = require_cache();
    println!("== sep[a->b] == close[a] + open[b] ==");
    let mut n = 0usize;
    for (name, src) in &loaded {
        let r = passes::factor(name, src);
        if r.factors {
            n += 1;
            println!("  {:<13} FACTORS", r.name);
            for role in r
                .open
                .keys()
                .chain(r.close.keys())
                .collect::<std::collections::BTreeSet<_>>()
            {
                println!(
                    "      {role:<10} open={:?} close={:?}",
                    r.open.get(role).cloned().unwrap_or_default(),
                    r.close.get(role).cloned().unwrap_or_default()
                );
            }
        } else {
            println!("  {:<13} DOES NOT FACTOR", r.name);
            for x in &r.residuals {
                println!("      {x}");
            }
        }
    }
    println!("\n  factors for {n}/{}\n", loaded.len());
}

fn monotonic() {
    let loaded = require_cache();
    println!("== append-only as the conversation grows, and tools placement ==");
    println!(
        "  {:<13} {:<26} {:<26} tools placement",
        "template", "no tools", "with tools"
    );
    let mut a = 0usize;
    let mut b = 0usize;
    let mut notes = Vec::new();
    for (name, src) in &loaded {
        let m0 = passes::monotonic(src, false);
        let m1 = passes::monotonic(src, true);
        if m0.monotonic {
            a += 1;
        }
        if m1.monotonic {
            b += 1;
        }
        let fmt = |m: &passes::MonoResult| match (&m.monotonic, &m.fixed_tail) {
            (true, Some(t)) => format!("yes, fixed tail {t:?}"),
            (true, None) => "yes".into(),
            (false, _) => "NO".into(),
        };
        println!(
            "  {:<13} {:<26} {:<26} {}",
            name,
            fmt(&m0),
            fmt(&m1),
            passes::tools_placement(src)
        );
        for (tag, m) in [("no tools", &m0), ("with tools", &m1)] {
            for n in &m.notes {
                notes.push(format!("  {name} ({tag}): {n}"));
            }
        }
    }
    println!("\n  monotonic without tools: {a}/{}", loaded.len());
    println!("  monotonic with tools:    {b}/{}", loaded.len());
    for n in &notes {
        println!("\n{n}");
    }
    println!();
}

fn table(which: &[&str]) {
    let loaded = require_cache();
    for (name, src) in &loaded {
        if !which.is_empty() && !which.contains(name) {
            continue;
        }
        println!("== {name} ({} B) ==", src.len());
        for line in passes::table(src) {
            println!("{line}");
        }
        println!();
    }
}

fn tally() {
    let loaded = require_cache();
    println!("== dimensions the turn grammar does not cover ==");
    let w = [13, 7, 9, 7, 13, 13, 14, 12];
    let hdr = [
        "template",
        "bytes",
        "dflt-sys",
        "tools",
        "tool-result",
        "reasoning",
        "list-content",
        "2 results",
    ];
    println!(
        "  {}",
        hdr.iter()
            .zip(w)
            .map(|(h, n)| format!("{h:<n$}"))
            .collect::<String>()
    );

    let mut rows = Vec::new();
    for (name, src) in &loaded {
        rows.push(passes::tally(name, src));
    }
    for t in &rows {
        let cells = [
            t.name.clone(),
            t.bytes.to_string(),
            if t.default_system {
                "yes".into()
            } else {
                "no".into()
            },
            if t.tools { "yes".into() } else { "no".into() },
            t.tool_result.into(),
            t.reasoning.into(),
            t.list_content.into(),
            t.two_tool_results.into(),
        ];
        println!(
            "  {}",
            cells
                .iter()
                .zip(w)
                .map(|(c, n)| format!("{c:<n$}"))
                .collect::<String>()
        );
    }

    let count = |f: &dyn Fn(&passes::Tally) -> String| {
        let mut m: std::collections::BTreeMap<String, usize> = Default::default();
        for t in &rows {
            *m.entry(f(t)).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = m.into_iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        v.iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!();
    println!(
        "  default system: {}",
        count(&|t| if t.default_system {
            "yes".into()
        } else {
            "no".into()
        })
    );
    println!(
        "  tools block:    {}",
        count(&|t| if t.tools { "yes".into() } else { "no".into() })
    );
    println!("  tool-result:    {}", count(&|t| t.tool_result.into()));
    println!("  reasoning:      {}", count(&|t| t.reasoning.into()));
    println!("  list-content:   {}", count(&|t| t.list_content.into()));
    println!(
        "  two results:    {}",
        count(&|t| t.two_tool_results.into())
    );
    println!(
        "  total bytes:    {}",
        rows.iter().map(|t| t.bytes).sum::<usize>()
    );
    println!();
}
