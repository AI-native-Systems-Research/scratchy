//! scratchy over every run a Mac has published: site/data/metal/ and site/data/metal/history/.
//!
//! The page proper shows each Mac's two newest runs; a run the next one replaces moves to
//! history/, and is read here for one number only. So history/ is read loosely: a run file of
//! any age, from before a field existed, still gives its date, its commit and each model's warm
//! single-user speed, and nothing else is asked of it.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dioxus::prelude::*;
use serde::Deserialize;

use crate::charts::{Axes, Axis, Chart, ChartOptions, Colour, Enabled, Themed, chart};

/// The runs a newer one replaced, relative to site/: only this page reads them.
pub const HISTORY: &str = "data/metal/history";

/// A run file, as much of it as the history needs.
#[derive(Deserialize)]
struct Loose {
    machine: LooseMachine,
    generated_utc: String,
    repo: LooseRepo,
    #[serde(default)]
    config: LooseConfig,
    models: Vec<LooseModel>,
}

#[derive(Deserialize)]
struct LooseMachine {
    chip: String,
}

#[derive(Deserialize)]
struct LooseRepo {
    sha: String,
}

#[derive(Default, Deserialize)]
struct LooseConfig {
    startup_request: Option<StartupRequest>,
}

/// The warm single-user request a run timed scratchy at: every run from 2026-10-06 on records
/// 64 tokens in, 32 out, 20 requests. The two before it do not say.
#[derive(Clone, Copy, Deserialize, PartialEq)]
struct StartupRequest {
    input_len: u32,
    output_len: u32,
}

#[derive(Deserialize)]
struct LooseModel {
    stem: String,
    warm_serving: Option<Warm>,
}

#[derive(Deserialize)]
struct Warm {
    output_throughput: Option<f64>,
}

/// One run's warm single-user tok/s for one model, scratchy alone.
#[derive(Clone, PartialEq, Debug)]
pub struct Point {
    pub when: String,
    pub sha8: String,
    pub tok_s: f64,
    /// The request it was timed at was not recorded (runs before 2026-10-06).
    pub unrecorded: bool,
}

/// Per Mac, per model: its points, oldest first.
pub type Timelines = BTreeMap<String, Models>;

/// One Mac's models: each one's points, oldest first.
pub type Models = BTreeMap<String, Vec<Point>>;

/// Every run file in `dirs` (relative to `root`), read loosely. Fails naming each file that will
/// not read even so: a file here is published data, and a typo in one should not vanish.
pub fn load(root: &Path, dirs: &[&str]) -> Result<Timelines, String> {
    let mut out = Timelines::new();
    let mut errors = Vec::new();
    for dir in dirs {
        let dir = root.join(dir);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for p in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if p.extension().is_none_or(|x| x != "json") {
                continue;
            }
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            let run: Loose = match fs::read_to_string(&p)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
            {
                Ok(r) => r,
                Err(e) => {
                    errors.push(format!("{}/{name}: {e}", dir.display()));
                    continue;
                }
            };
            add(&mut out, run);
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    for models in out.values_mut() {
        for points in models.values_mut() {
            points.sort_by(|a, b| a.when.cmp(&b.when));
            points.dedup_by(|a, b| a.when == b.when);
        }
    }
    Ok(out)
}

/// The request every recorded run timed: a run at another would not belong on the same line.
const WARM: StartupRequest = StartupRequest {
    input_len: 64,
    output_len: 32,
};

fn add(out: &mut Timelines, run: Loose) {
    let request = run.config.startup_request;
    if request.is_some_and(|r| r != WARM) {
        return;
    }
    let models = out.entry(run.machine.chip).or_default();
    for m in run.models {
        let Some(tok_s) = m.warm_serving.and_then(|w| w.output_throughput) else {
            continue;
        };
        if tok_s <= 0.0 {
            continue;
        }
        models.entry(m.stem).or_default().push(Point {
            when: run.generated_utc.clone(),
            sha8: run.repo.sha.chars().take(8).collect(),
            tok_s,
            unrecorded: request.is_none(),
        });
    }
}

/// A run's name on the axis: its date, and its commit, so two runs on one day stay apart.
fn run_label(p: &Point) -> String {
    format!("{} {}", &p.when[5..10], &p.sha8[..p.sha8.len().min(7)])
}

/// A model's change from its first run to its newest, as a percentage.
pub fn change(points: &[Point]) -> Option<f64> {
    let (first, last) = (points.first()?, points.last()?);
    (points.len() > 1).then(|| (last.tok_s / first.tok_s - 1.0) * 100.0)
}

/// The colours the lines take in turn: Carbon's categorical palette, the first eight.
const LINES: [&str; 8] = [
    "#6929c4", "#1192e8", "#005d5d", "#9f1853", "#fa4d56", "#570408", "#198038", "#002d9c",
];

/// One Mac's chart: a line a model through its runs, by date. A model a later round dropped
/// stops where its runs stop; one a round added starts there.
fn machine_chart(chip: &str, models: &Models) -> Option<Element> {
    let lines: Vec<(&String, &Vec<Point>)> = models.iter().filter(|(_, p)| p.len() > 1).collect();
    if lines.is_empty() {
        return None;
    }
    // A run on the axis at the time it ran, so the gaps between points are the time between runs:
    // two runs hours apart on one day sit close, a week's
    // gap wide. Each point keeps its date and commit for the tooltip.
    let data: Vec<serde_json::Value> = (lines.iter())
        .flat_map(|(stem, ps)| {
            ps.iter().map(move |p| {
                serde_json::json!({
                    "group": stem,
                    "date": p.when,
                    "run": run_label(p),
                    "value": (p.tok_s * 10.0).round() / 10.0,
                })
            })
        })
        .collect();
    let spec = Chart::Line {
        data,
        options: ChartOptions {
            title: String::new(),
            axes: Axes {
                bottom: Axis {
                    maps_to: "date",
                    title: "Run date (UTC)",
                    scale_type: Some("time"),
                    ticks: None,
                    include_zero: None,
                    visible: None,
                    domain: None,
                },
                left: Axis {
                    maps_to: "value",
                    title: "tok/s, one user (higher is better)",
                    scale_type: None,
                    ticks: None,
                    include_zero: Some(true),
                    visible: None,
                    domain: None,
                },
            },
            color: Some(Colour::Scale {
                scale: (lines.iter().enumerate())
                    .map(|(i, (stem, _))| (stem.as_str(), Themed::both(LINES[i % LINES.len()])))
                    .collect(),
            }),
            heatmap: None,
            height: "16rem".to_string(),
            width: None,
            legend: Some(Enabled { enabled: true }),
            locale: None,
            combo: None,
            bounds: None,
            bubble: None,
            toolbar: Enabled { enabled: false },
        },
    };
    Some(chart(
        &spec,
        &format!("scratchy's one-user speed on {chip} over its runs, a line a model"),
        "",
        None,
    ))
}

/// One Mac's history: its chart and, under it, each model's first and newest run and the change.
/// None when no model has two runs on the Mac.
pub fn block(chip: &str, models: &Models) -> Option<Element> {
    let chart = machine_chart(chip, models)?;
    Some(rsx! {
        div {
            {chart}
            ul { style: "font-size: 0.875rem; margin: 0.5rem 0 0",
                for (stem, ps) in models.iter().filter(|(_, p)| p.len() > 1) {
                    li { {summary(stem, ps)} }
                }
            }
        }
    })
}

fn summary(stem: &str, ps: &[Point]) -> String {
    let (first, last) = (&ps[0], &ps[ps.len() - 1]);
    let mark = |p: &Point| if p.unrecorded { "†" } else { "" };
    let pct = change(ps).unwrap_or_default();
    format!(
        "{stem}: {:.1} tok/s{} on {} ({}) to {:.1} on {} ({}), {}{:.0}% over {} runs",
        first.tok_s,
        mark(first),
        &first.when[..10],
        first.sha8,
        last.tok_s,
        &last.when[..10],
        last.sha8,
        if pct >= 0.0 { "+" } else { "" },
        pct,
        ps.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chip: &str, when: &str, request: Option<(u32, u32)>, models: &[(&str, f64)]) -> Loose {
        Loose {
            machine: LooseMachine { chip: chip.into() },
            generated_utc: when.into(),
            repo: LooseRepo {
                sha: "0123456789abcdef".into(),
            },
            config: LooseConfig {
                startup_request: request.map(|(input_len, output_len)| StartupRequest {
                    input_len,
                    output_len,
                }),
            },
            models: (models.iter())
                .map(|&(stem, t)| LooseModel {
                    stem: stem.into(),
                    warm_serving: Some(Warm {
                        output_throughput: Some(t),
                    }),
                })
                .collect(),
        }
    }

    /// A model's line takes every run that timed it at the warm request, oldest first; a run that
    /// did not record its request is kept and marked; one at another request is left out.
    #[test]
    fn a_line_a_model_oldest_first_at_one_request() {
        let mut t = Timelines::new();
        add(
            &mut t,
            run("M3", "2026-10-13T01:05:19Z", Some((64, 32)), &[("g", 73.6)]),
        );
        add(
            &mut t,
            run("M3", "2026-10-02T03:15:29Z", None, &[("g", 52.8)]),
        );
        add(
            &mut t,
            run(
                "M3",
                "2026-10-10T15:47:09Z",
                Some((128, 64)),
                &[("g", 99.0)],
            ),
        );
        for models in t.values_mut() {
            for ps in models.values_mut() {
                ps.sort_by(|a, b| a.when.cmp(&b.when));
            }
        }
        let g = &t["M3"]["g"];
        assert_eq!(g.len(), 2);
        assert!(g[0].unrecorded && !g[1].unrecorded);
        assert_eq!((g[0].tok_s, g[1].tok_s), (52.8, 73.6));
        let pct = change(g).unwrap();
        assert!((pct - 39.4).abs() < 0.1, "{pct}");
    }

    /// The chart puts each point at the time its run ran, not at an evenly spaced label: two runs
    /// hours apart and two a week apart are as far apart as that.
    #[test]
    fn a_point_sits_at_the_time_its_run_ran() {
        let p = |when: &str| Point {
            when: when.into(),
            sha8: "abcdef12".into(),
            tok_s: 1.0,
            unrecorded: false,
        };
        let mut models = Models::new();
        models.insert(
            "m".into(),
            vec![
                p("2026-10-02T03:00:00Z"),
                p("2026-10-09T09:00:00Z"),
                p("2026-10-09T21:00:00Z"),
            ],
        );
        let html = dioxus_ssr::render_element(machine_chart("Apple M3 Pro", &models).unwrap());
        let html = html.replace("&#34;", "\"");
        assert!(
            html.contains("\"scaleType\":\"time\""),
            "a time axis: {html}"
        );
        for when in [
            "2026-10-02T03:00:00Z",
            "2026-10-09T09:00:00Z",
            "2026-10-09T21:00:00Z",
        ] {
            assert!(
                html.contains(&format!("\"date\":\"{when}\"")),
                "{when} on the axis"
            );
        }
        assert!(
            html.contains("\"run\":\"10-09 abcdef1\""),
            "its date and commit kept"
        );
    }
}
