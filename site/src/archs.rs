//! architectures.html, from crates/models/arch/dsl/*.py.
//!
//! The DSL files stay the single source of truth: this reads them at site-build
//! time, so the page cannot drift from the compiler input.

use std::collections::BTreeMap;
use std::fs;

use dioxus::prelude::*;
use serde::Serialize;

use crate::carbon::{CodeSnippet, Column, Grid, Heading, SideNav, SideNavLink, Span, Stack, Tile};
use crate::chrome::{self, Content, Head, Library, Root, Tab, Theme};
use crate::highlight;
use crate::{Site, thousands};

/// Python isn't one of diff-view-element's built-in languages; the page's
/// script loads this grammar into its PrismJS instance on first use.
const PRISM_PYTHON: &str = "https://cdn.jsdelivr.net/npm/prism-esm/components/prism-python.js";

/// Selection, diffing, and the #left..right deep links.
const SCRIPT: &str = include_str!("archs.js");

/// The DSL files, relative to the repo root.
pub const DSL: &str = "crates/models/arch/dsl";

/// The architecture every other one is diffed and ranked against.
const BASELINE: &str = "llama";

#[derive(Serialize)]
struct Arch {
    path: String,
    raw: Vec<String>,
    html: Vec<String>,
    lines: usize,
    del: usize,
    add: usize,
    distance: usize,
}

/// What the page's script reads: every architecture, the ranking, and the
/// initial selection (the baseline, diffed against nothing).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Data<'a> {
    base: &'a str,
    order: &'a [String],
    archs: &'a BTreeMap<String, Arch>,
    left: &'a str,
    right: &'a str,
    prism_python_url: &'a str,
}

/// Every carrier names its entry point after itself, `def llama():`. That's a
/// nominal difference, not a structural one — left alone it shows up as a diff
/// line (and inflates the ranking's distance from baseline) on every single
/// comparison. Rewritten for display and diffing only; the compiler's DSL
/// files are untouched.
fn normalize_entry_fn(stem: &str, line: &str) -> String {
    let entry = format!("def {}()", stem.replace('-', "_"));
    match line.strip_prefix(&entry) {
        Some(rest) => format!("def forward(){rest}"),
        None => line.to_string(),
    }
}

/// Comparison key: whitespace-insensitive, comments ignored.
fn norm(line: &str) -> String {
    let code = line.split('#').next().unwrap_or_default();
    code.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// (removed, added) code lines under an LCS alignment. A rewritten line counts
/// once on each side, which is what a side-by-side view shows.
fn diff_count(a: &[String], b: &[String]) -> (usize, usize) {
    let a: Vec<String> = a
        .iter()
        .map(|l| norm(l))
        .filter(|l| !l.is_empty())
        .collect();
    let b: Vec<String> = b
        .iter()
        .map(|l| norm(l))
        .filter(|l| !l.is_empty())
        .collect();
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let common = dp[0][0];
    (n - common, m - common)
}

pub fn build(site: &Site) -> Result<String, String> {
    let dsl = site.repo.join(DSL);
    let mut files: Vec<_> = fs::read_dir(&dsl)
        .map_err(|e| format!("{}: {e}", dsl.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "py"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no DSL files under {}", dsl.display()));
    }

    let mut archs = BTreeMap::new();
    for f in &files {
        let stem = f
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("{}: bad name", f.display()))?;
        let text = fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let raw: Vec<String> = text.lines().map(|l| normalize_entry_fn(stem, l)).collect();
        let html = raw
            .iter()
            .map(|l| dioxus_ssr::render_element(highlight::line(l)))
            .collect();
        let path = f
            .strip_prefix(&site.repo)
            .unwrap_or(f)
            .display()
            .to_string();
        archs.insert(
            stem.to_string(),
            Arch {
                path,
                lines: raw.len(),
                raw,
                html,
                del: 0,
                add: 0,
                distance: 0,
            },
        );
    }

    let base = if archs.contains_key(BASELINE) {
        BASELINE.to_string()
    } else {
        archs.keys().next().cloned().unwrap_or_default()
    };
    let base_raw = archs[&base].raw.clone();
    for a in archs.values_mut() {
        (a.del, a.add) = diff_count(&base_raw, &a.raw);
        a.distance = a.del + a.add;
    }

    // Sorted by distance from the baseline: the ordering is the argument.
    let mut order: Vec<String> = archs.keys().cloned().collect();
    order.sort_by(|x, y| archs[x].distance.cmp(&archs[y].distance).then(x.cmp(y)));
    let total: usize = archs.values().map(|a| a.lines).sum();

    let data = serde_json::to_string(&Data {
        base: &base,
        order: &order,
        archs: &archs,
        left: &base,
        right: "",
        prism_python_url: PRISM_PYTHON,
    })
    .map_err(|e| e.to_string())?;

    let page = page(&archs, &order, &base, total, &data)?;
    let out = site.out.join("architectures.html");
    fs::write(&out, page).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(format!(
        "architectures page: {} models, {} DSL lines -> {}",
        archs.len(),
        thousands(total),
        out.display()
    ))
}

fn page(
    archs: &BTreeMap<String, Arch>,
    order: &[String],
    base: &str,
    total: usize,
    data: &str,
) -> Result<String, String> {
    let count = archs.len();
    let total = thousands(total);
    let body = rsx! {
        {chrome::header(Root(0), Some(Tab::Models))}

        SideNav { label: "Model architectures", id: "model-nav",
            for n in order {
                SideNavLink { href: "#{n}", data_arch: n.clone(), active: n == base, "{n}" }
            }
        }

        Content {
            div {
              Stack { gap: 7,
                Stack { gap: 5,
                    Heading { "Scratchy model architectures" }
                    p {
                        "{count} architectures, {total} lines of DSL between them. Each one is the model's math, \
                         written once; the compiler turns it into the code for every target. Pick a model from the \
                         list on the left, then optionally diff it against another below."
                    }
                }

                Grid {
                    Column { span: Span::THIRD,
                        cds-select { "id": "right", "label-text": "Diff against", "value": "",
                            cds-select-item { "value": "", "selected": true, "— none —" }
                            for n in order {
                                cds-select-item { "value": "{n}", "{n} — {archs[n].lines} lines" }
                            }
                        }
                    }
                }

                // The baseline is rendered here, so the page is right before its
                // script runs; the script swaps in the model picked.
                Tile { id: "archpane",
                    Stack { gap: 3,
                        p { code { id: "path", "{archs[base].path}" } " · " span { id: "meta", "{archs[base].lines} lines" } }
                        CodeSnippet { id: "code", {highlight::block(&archs[base].raw.join("\n"))} }
                    }
                }

                Tile { id: "diffpane", style: "display: none",
                    Stack { gap: 3,
                        p { id: "diffhead" }
                        diff-view-element { "id": "diffview", "language": "python", "disable-line-numbers": "" }
                    }
                }
              }
            }
        }

        script { r#type: "application/json", id: "archdata", dangerous_inner_html: data }
        script { dangerous_inner_html: SCRIPT }
    };
    chrome::document(
        Head {
            title: "Model architectures — scratchy",
            description: Some(
                "Every model architecture scratchy supports, written in its math DSL, diffable side by side.",
            ),
            og: None,
            root: Root(0),
            libraries: &[Library::DiffView],
            theme: Theme::FollowSystem,
        },
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<String> {
        s.lines().map(str::to_string).collect()
    }

    #[test]
    fn diff_count_ignores_comments_and_whitespace() {
        let a = lines("x = f(a)\n# note\ny = g(x)");
        let b = lines("x  =  f(a)  # renamed\nz = h(x)\ny = g(x)");
        assert_eq!(diff_count(&a, &b), (0, 1));
    }

    #[test]
    fn entry_fn_is_renamed_only_at_its_definition() {
        assert_eq!(
            normalize_entry_fn("gemma4-moe", "def gemma4_moe():"),
            "def forward():"
        );
        assert_eq!(normalize_entry_fn("llama", "    llama()"), "    llama()");
    }
}
