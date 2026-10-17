//! The chart machinery the performance pages share: the specs charts.js
//! hands Carbon Charts, the ratio heat maps' colour scale, and the sparkline
//! that fills the gap between two lines — blue where scratchy is ahead, red
//! where it is behind. metal.html and spyre.html render these with their own
//! data; nothing here knows a page's own numbers.

use std::collections::BTreeMap;

use dioxus::prelude::*;
use serde::Serialize;

use crate::carbon::{Alignment, Column, Definition, Grid, Gutter, Span};

/// Draws every chart from its rendered spec.
pub const CHARTS_JS: &str = include_str!("charts.js");

/// Which way a measurement is better: higher for a rate, lower for a time.
#[derive(Clone, Copy, PartialEq)]
pub enum Better {
    Higher,
    Lower,
}

impl Better {
    /// How many times better scratchy's value is than its rival's.
    pub fn faster(self, scratchy: Option<f64>, other: Option<f64>) -> Option<f64> {
        let (s, o) = (
            scratchy.filter(|v| *v != 0.0)?,
            other.filter(|v| *v != 0.0)?,
        );
        Some(match self {
            Better::Higher => s / o,
            Better::Lower => o / s,
        })
    }
}

/// A series' colour, fixed per role: blue, orange, aqua from the data-viz
/// reference palette, validated as a set (all pairs, CVD and normal vision)
/// against Carbon's white and g100 surfaces. Aqua is under 3:1 on white, so
/// every chart has a legend and a table view. A previous run is a light grey
/// so it reads as behind.
pub fn colour(class: &str) -> &'static str {
    match class {
        "s1" => "#2a78d6",
        "s2" => "#d95926",
        "s3" => "#1baf7a",
        _ => "#a8a8a8",
    }
}

/// What charts.js hands Carbon Charts: which chart, its points, and its
/// options, in Carbon Charts' own field names.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Chart<'a> {
    /// Carbon's GroupedBarChart: a bar per group, side by side, at each key.
    #[serde(rename = "groupedbar")]
    GroupedBar {
        data: Vec<BarPoint<'a>>,
        options: ChartOptions<'a>,
    },
    Heatmap {
        data: Vec<HeatPoint>,
        options: ChartOptions<'a>,
        /// The values are log2(ratio): charts.js shows them as "×1.23".
        ratio: bool,
    },
    /// A sparkline: Carbon's ComboChart, two lines and the fills between them.
    Combo {
        data: Vec<SparkPoint>,
        options: ChartOptions<'a>,
    },
}

/// One point of a sparkline. On the two lines, a side's value; on the fills,
/// also the band from the lower line to the upper. A None value breaks the
/// line or fill there.
#[derive(Serialize)]
pub struct SparkPoint {
    pub group: &'static str,
    pub key: f64,
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower: Option<f64>,
    /// On a highlighted step's fill: how many times faster scratchy is there,
    /// which charts.js writes in the fill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,
}

/// One measured bar. A None value (present but not measured: no stream)
/// leaves no bar.
#[derive(Clone, Serialize)]
pub struct BarPoint<'a> {
    pub group: &'a str,
    pub key: String,
    pub value: Option<f64>,
}

/// One cell of a heatmap. A highlight has its factor written in it
/// (charts.js).
#[derive(Serialize)]
pub struct HeatPoint {
    pub x: String,
    pub y: String,
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub highlight: bool,
}

#[derive(Serialize)]
pub struct ChartOptions<'a> {
    pub title: String,
    pub axes: Axes<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<Colour<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heatmap: Option<HeatmapOptions>,
    pub height: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legend: Option<Enabled>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<Locale>,
    #[serde(rename = "comboChartTypes", skip_serializing_if = "Option::is_none")]
    pub combo: Option<Vec<ComboLayer>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Bounds>,
    // The page is always Carbon's light theme.
    pub theme: &'static str,
    pub toolbar: Enabled,
}

#[derive(Serialize)]
pub struct Axes<'a> {
    pub bottom: Axis<'a>,
    pub left: Axis<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Axis<'a> {
    pub maps_to: &'static str,
    pub title: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ticks: Option<Ticks>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_zero: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<AxisDomain>,
}

/// An axis's fixed extent.
#[derive(Serialize)]
#[serde(untagged)]
pub enum AxisDomain {
    /// A label axis's labels in order; Carbon puts the first at the left of a
    /// horizontal axis and at the bottom of a vertical one.
    Labels(Vec<String>),
    /// A numeric axis's [from, to]. Given, it is not rounded out: Carbon's
    /// own log range goes to powers of ten (1 to 16 users became 1 to 100,
    /// the line stopping 60% of the way across).
    Values([f64; 2]),
}

/// One layer of a combo chart: a chart type and the series it draws.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComboLayer {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub corresponding_datasets: Vec<&'static str>,
}

/// Which point fields a bounded area fills between. Carbon draws bounds only
/// for an area layer holding a single series.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub upper_bound_maps_to: &'static str,
    pub lower_bound_maps_to: &'static str,
}

#[derive(Serialize)]
pub struct Ticks {
    pub values: Vec<f64>,
    /// What to print at each of `values` instead of the number: charts.js
    /// makes these Carbon's tick formatter, which only a function can be.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
}

/// Carbon Charts' colour options: a colour per series, or the steps of a
/// heatmap's quantized scale.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Colour<'a> {
    Scale {
        scale: BTreeMap<&'a str, &'static str>,
    },
    Gradient {
        gradient: Gradient,
    },
}

#[derive(Serialize)]
pub struct Gradient {
    pub colors: &'static [&'static str],
}

/// The widest speedup the ratio heatmaps tell apart, either way: ×4 faster
/// and ×0.25 (4× slower). Anything beyond takes the end colour.
pub const RATIO_SPAN: f64 = 4.0;

/// The ratio heatmaps plot log2 of the ratio, so a factor of two is the same
/// distance from ×1 whichever way it goes: ±log2(RATIO_SPAN).
pub const RATIO_RANGE: Domain = Domain {
    min: -2.0,
    max: 2.0,
};

/// The ratio heatmaps' 33 steps, slowest to fastest, from Carbon's red, gray
/// and cyan. 33 equal steps over ±2 put ×0.96 to ×1.04 (about the run-to-run
/// noise band) in the gray 40 middle step. Carbon's own diverging palette
/// fades to white in the middle, down to 1.0:1 against the page and tiles, so
/// the cells near ×1, most of them, vanish; here the steps past the middle
/// start at red/cyan 40 and darken outward to 100, every one at least 2.37:1
/// on white and 2.15:1 on the grey tile (#f4f4f4).
pub const RATIO_STEPS: [&str; 33] = [
    "#2d0709", // red 100: ×0.25 or slower
    "#520408", // red 90
    "#520408", // red 90
    "#750e13", // red 80
    "#750e13", // red 80
    "#a2191f", // red 70
    "#a2191f", // red 70
    "#a2191f", // red 70
    "#da1e28", // red 60
    "#da1e28", // red 60
    "#da1e28", // red 60
    "#fa4d56", // red 50
    "#fa4d56", // red 50
    "#fa4d56", // red 50
    "#ff8389", // red 40
    "#ff8389", // red 40
    "#a8a8a8", // gray 40: ×0.96 to ×1.04, within noise
    "#33b1ff", // cyan 40
    "#33b1ff", // cyan 40
    "#1192e8", // cyan 50
    "#1192e8", // cyan 50
    "#1192e8", // cyan 50
    "#0072c3", // cyan 60
    "#0072c3", // cyan 60
    "#0072c3", // cyan 60
    "#00539a", // cyan 70
    "#00539a", // cyan 70
    "#00539a", // cyan 70
    "#003a6d", // cyan 80
    "#003a6d", // cyan 80
    "#012749", // cyan 90
    "#012749", // cyan 90
    "#061727", // cyan 100: ×4 or faster
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeatmapOptions {
    pub color_legend: ColorLegend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_domain: Option<Domain>,
}

#[derive(Serialize)]
pub struct ColorLegend {
    pub title: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

#[derive(Serialize)]
pub struct Domain {
    pub min: f64,
    pub max: f64,
}

/// Carbon's heatmap tooltip lists the axis values, then the cell's value
/// under `locale.translations.total` (Carbon's default: "Total"; it outranks
/// `tooltip.totalLabel`).
#[derive(Serialize)]
pub struct Locale {
    pub translations: Translations,
}

#[derive(Serialize)]
pub struct Translations {
    pub total: String,
}

impl Locale {
    pub fn total(label: String) -> Option<Self> {
        Some(Locale {
            translations: Translations { total: label },
        })
    }
}

#[derive(Serialize)]
pub struct Enabled {
    pub enabled: bool,
}

/// A chart's mount point, styled by `mount`: charts.js draws `spec` into it.
/// `scale` is the id of the ratio scale a heatmap's squares highlight on hover.
pub fn chart(spec: &Chart, label: &str, mount: &str, scale: Option<&str>) -> Element {
    rsx! {
        div { role: "img", "aria-label": label, style: "{mount}", "data-scale": scale,
            "data-chart": serde_json::to_string(spec).unwrap_or_default(),
        }
    }
}

/// The ratio heatmaps' colour scale, upright: RATIO_STEPS fastest at the top,
/// labelled at the ends and the middle. Carbon's heatmap only draws its legend
/// as a bar underneath.
/// The overview's ratio scale.
pub const GLANCE_SCALE: &str = "scale-glance";

pub fn ratio_scale(fit: ScaleFit, id: &str) -> Element {
    let rows = RATIO_STEPS.len();
    let middle = rows / 2 + 1;
    let (place, track) = match fit {
        ScaleFit::Fixed => (String::new(), "0.5rem"),
        ScaleFit::Beside => (
            // Stretched to the rows beside it: from the top of the first
            // row's grids (below its model names, 1.5rem) to the bottom of
            // the last's (above its two label lines, 2.25rem). Stacked under
            // them below the medium breakpoint (charts_beside_scale), it
            // keeps a readable height.
            "align-self: stretch; margin-block: 1.5rem 2.25rem; min-height: 12rem; ".to_string(),
            "1fr",
        ),
    };
    rsx! {
        div { id: "{id}", style: "{place}display: grid; grid-template-columns: 0.75rem max-content; grid-template-rows: repeat({rows}, {track}); column-gap: 0.5rem; font-size: 0.75rem; line-height: 1",
            // Fastest at the top; `data-step` counts from the slowest, the way
            // the heatmap's colours do, so charts.js can find a square's step.
            for (row, colour) in RATIO_STEPS.iter().rev().enumerate() {
                div { style: "grid-column: 1; grid-row: {row + 1}; background: {colour}",
                    "data-step": "{rows - 1 - row}",
                }
            }
            div { style: "grid-column: 2; grid-row: 1; align-self: start",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at least {RATIO_SPAN} times as fast as the engine the map compares it with; anything faster takes this colour too.",
                    strong { "×{RATIO_SPAN}" }
                }
            }
            div { style: "grid-column: 2; grid-row: {middle}; align-self: center",
                Definition { alignment: Alignment::Left,
                    definition: "The same speed, within run-to-run noise: ×0.96 to ×1.04.",
                    strong { "×1" }
                }
            }
            div { style: "grid-column: 2; grid-row: {rows}; align-self: end",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at most a quarter as fast as the engine the map compares it with, {RATIO_SPAN} times slower; anything slower takes this colour too.",
                    strong { "×{1.0 / RATIO_SPAN}" }
                }
            }
        }
    }
}

/// Where the ratio scale goes: fixed-size steps beside a model's grids, or
/// stretched to the height of the overview's rows beside it.
#[derive(Clone, Copy, PartialEq)]
pub enum ScaleFit {
    Fixed,
    Beside,
}

/// A glance's charts beside their shared ratio scale, on Carbon's grid: the
/// charts take the wide column, the scale the narrow one, and the two stack
/// below the medium breakpoint — Carbon's own columns, no layout of ours.
/// `charts` is the wide column's content and lays itself out.
pub fn charts_beside_scale(charts: Element, scale_id: &str) -> Element {
    rsx! {
        Grid { gutter: Gutter::Cards,
            Column { span: Span { sm: 4, md: 6, lg: 14 },
                {charts}
            }
            Column { span: Span { sm: 4, md: 2, lg: 2 },
                {ratio_scale(ScaleFit::Beside, scale_id)}
            }
        }
    }
}

/// A ratio as a heatmap value: log2, so a factor of two either way is the
/// same distance from ×1.
pub fn log2_ratio(r: f64) -> f64 {
    (r.log2() * 1000.0).round() / 1000.0
}

/// The RATIO_STEPS colour a ratio heatmap gives a ratio, as Carbon's quantize
/// scale picks it: RATIO_RANGE in equal slices, beyond it the end colours.
pub fn ratio_step(r: f64) -> usize {
    let n = RATIO_STEPS.len();
    let at = (log2_ratio(r) - RATIO_RANGE.min) / (RATIO_RANGE.max - RATIO_RANGE.min);
    // Truncation is the floor here: clamped to 0 first.
    ((at.max(0.0) * n as f64) as usize).min(n - 1)
}

/// One side of a comparison chart: the name of its line, and of its fill
/// where it is the faster of the two.
#[derive(Clone, Copy)]
pub struct Side {
    pub line: &'static str,
    pub fill: &'static str,
}

/// One step along a sparkline's dimension: where, each side's value, and
/// whether it is one of its panel's highlights (its factor written in its
/// fill).
#[derive(Clone, Copy)]
pub struct Step {
    pub x: f64,
    pub scratchy: Option<f64>,
    pub rival: Option<f64>,
    pub highlight: bool,
}

/// The lines' points, and the two fills' points: each fill present where its
/// side is ahead, the two meeting where the lines cross (found on the log
/// x axis, on which the chart draws straight segments).
pub fn spark_points(pts: &[Step], better: Better, scratchy: Side, rival: Side) -> Vec<SparkPoint> {
    let line = |group, key, value| SparkPoint {
        group,
        key,
        value,
        upper: None,
        lower: None,
        factor: None,
    };
    let band = |group, key, a: f64, b: f64| SparkPoint {
        group,
        key,
        value: Some(a.max(b)),
        upper: Some(a.max(b)),
        lower: Some(a.min(b)),
        factor: None,
    };
    let gap = |group, key| SparkPoint {
        group,
        key,
        value: None,
        upper: None,
        lower: None,
        factor: None,
    };
    // Positive where scratchy is ahead: more tokens a second, or less time.
    let lead = |s: f64, r: f64| match better {
        Better::Higher => s - r,
        Better::Lower => r - s,
    };
    let mut out: Vec<SparkPoint> = Vec::new();
    for &Step {
        x,
        scratchy: sv,
        rival: rv,
        ..
    } in pts
    {
        out.push(line(scratchy.line, x, sv));
        out.push(line(rival.line, x, rv));
    }
    let mut fills: Vec<SparkPoint> = Vec::new();
    for (
        k,
        &Step {
            x,
            scratchy: sv,
            rival: rv,
            highlight,
        },
    ) in pts.iter().enumerate()
    {
        match (sv, rv) {
            (Some(sv), Some(rv)) => {
                let d = lead(sv, rv);
                let factor = if highlight {
                    better.faster(Some(sv), Some(rv))
                } else {
                    None
                };
                fills.push(if d > 0.0 {
                    SparkPoint {
                        factor,
                        ..band(scratchy.fill, x, sv, rv)
                    }
                } else {
                    gap(scratchy.fill, x)
                });
                fills.push(if d < 0.0 {
                    SparkPoint {
                        factor,
                        ..band(rival.fill, x, sv, rv)
                    }
                } else {
                    gap(rival.fill, x)
                });
                // Where the next step's lead has the other sign, the lines
                // cross between: both fills end there, at no width.
                if let Some(&Step {
                    x: x1,
                    scratchy: Some(s1),
                    rival: Some(r1),
                    ..
                }) = pts.get(k + 1)
                {
                    let d1 = lead(s1, r1);
                    if d * d1 < 0.0 {
                        let t = d / (d - d1);
                        let xc = x * (x1 / x).powf(t);
                        let yc = sv + t * (s1 - sv);
                        fills.push(band(scratchy.fill, xc, yc, yc));
                        fills.push(band(rival.fill, xc, yc, yc));
                    }
                }
            }
            _ => {
                fills.push(gap(scratchy.fill, x));
                fills.push(gap(rival.fill, x));
            }
        }
    }
    out.extend(fills);
    out
}

/// What a sparkline follows: which way its values are better, and its axes'
/// titles, which are the tooltip's labels.
#[derive(Clone, Copy)]
pub struct SparkAxes {
    pub better: Better,
    pub along: &'static str,
    pub value: &'static str,
}

/// A sparkline: scratchy's and its rival's values along one dimension, the gap
/// between them filled blue where scratchy is ahead and red where it is behind.
pub fn sparkline(
    pts: &[Step],
    axes: SparkAxes,
    scratchy: Side,
    rival: Side,
    names: Option<Vec<String>>,
    height: &str,
    label: &str,
) -> Element {
    let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
    // From the power of two at or below the first step to the one at or above
    // the last: users and lengths are powers of two already, so they span the
    // chart; model sizes sit inside (Carbon misdraws a log axis given the
    // sizes' own ends, 1.4 to 15.3 GB, and draws 1 to 16 right).
    let range = match (xs.first(), xs.last()) {
        (Some(&a), Some(&b)) => Some(AxisDomain::Values([
            a.log2().floor().exp2(),
            b.log2().ceil().exp2(),
        ])),
        _ => None,
    };
    // The axes' titles are the tooltip's labels; the hidden one's too.
    let bottom = Axis {
        maps_to: "key",
        title: axes.along,
        scale_type: Some("log"),
        ticks: Some(Ticks {
            values: xs,
            labels: names,
        }),
        include_zero: Some(false),
        visible: None,
        domain: range,
    };
    let left = Axis {
        maps_to: "value",
        title: axes.value,
        scale_type: None,
        ticks: None,
        include_zero: Some(false),
        visible: Some(false),
        domain: None,
    };
    let spec = Chart::Combo {
        data: spark_points(pts, axes.better, scratchy, rival),
        options: ChartOptions {
            title: String::new(),
            // Users, both lengths and model sizes grow by a factor per step:
            // on a log axis the steps sit evenly.
            axes: Axes { bottom, left },
            color: Some(Colour::Scale {
                scale: [
                    (scratchy.line, colour("s1")),
                    (rival.line, colour("s2")),
                    // The heat maps' cyan and red 50.
                    (scratchy.fill, "#1192e8"),
                    (rival.fill, "#fa4d56"),
                ]
                .into_iter()
                .collect(),
            }),
            heatmap: None,
            height: height.to_string(),
            width: None,
            legend: Some(Enabled { enabled: false }),
            locale: None,
            // Carbon draws an area's bounds only when the layer holds one series.
            combo: Some(vec![
                ComboLayer {
                    kind: "area",
                    corresponding_datasets: vec![scratchy.fill],
                },
                ComboLayer {
                    kind: "area",
                    corresponding_datasets: vec![rival.fill],
                },
                ComboLayer {
                    kind: "line",
                    corresponding_datasets: vec![scratchy.line, rival.line],
                },
            ]),
            bounds: Some(Bounds {
                upper_bound_maps_to: "upper",
                lower_bound_maps_to: "lower",
            }),
            theme: "white",
            toolbar: Enabled { enabled: false },
        },
    };
    chart(&spec, label, "width: 100%", None)
}

/// How many of each sparkline panel's steps, at each end, have their factor
/// written in them: the ones scratchy is furthest ahead at, and the ones it
/// is furthest behind at (or least ahead), over every machine, model and
/// metric.
pub const HIGHLIGHTS: usize = 3;

/// Marks one sparkline panel's highlights, from all its steps in order: the
/// HIGHLIGHTS where scratchy is furthest ahead and those where it is furthest
/// behind (or least ahead).
pub fn highlight<'s>(steps: impl Iterator<Item = (Better, &'s mut Step)>) {
    let mut ranked: Vec<(f64, &mut Step)> = steps
        .filter_map(|(better, st)| Some((better.faster(st.scratchy, st.rival)?, st)))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = ranked.len();
    for (k, (_, st)) in ranked.iter_mut().enumerate() {
        if k < HIGHLIGHTS || k + HIGHLIGHTS >= n {
            st.highlight = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Better, RATIO_RANGE, RATIO_SPAN, RATIO_STEPS, Side, Step, ratio_step, spark_points,
    };

    const SCRATCHY: Side = Side {
        line: "scratchy",
        fill: "scratchy ahead",
    };
    const RIVAL: Side = Side {
        line: "mlx-lm",
        fill: "mlx-lm ahead",
    };

    #[test]
    fn the_fills_switch_where_the_lines_cross() {
        // Ahead at 1 user (10 tok/s against 5), behind at 4 (5 against 10).
        let steps = [
            Step {
                x: 1.0,
                scratchy: Some(10.0),
                rival: Some(5.0),
                highlight: false,
            },
            Step {
                x: 4.0,
                scratchy: Some(5.0),
                rival: Some(10.0),
                highlight: false,
            },
        ];
        let pts = spark_points(&steps, Better::Higher, SCRATCHY, RIVAL);
        let fill = |group| -> Vec<(f64, Option<(f64, f64)>)> {
            pts.iter()
                .filter(|p| p.group == group)
                .map(|p| (p.key, p.lower.zip(p.upper)))
                .collect()
        };
        // Halfway along the lead, halfway along the log axis: x = 2, y = 7.5.
        assert_eq!(
            fill(SCRATCHY.fill),
            [
                (1.0, Some((5.0, 10.0))),
                (2.0, Some((7.5, 7.5))),
                (4.0, None)
            ]
        );
        assert_eq!(
            fill(RIVAL.fill),
            [
                (1.0, None),
                (2.0, Some((7.5, 7.5))),
                (4.0, Some((5.0, 10.0)))
            ]
        );
        // For time, less is ahead: the same numbers swap the fills.
        let ttft = spark_points(&steps, Better::Lower, SCRATCHY, RIVAL);
        assert!(
            ttft.iter()
                .any(|p| p.group == RIVAL.fill && p.key == 1.0 && p.upper.is_some())
        );
    }

    #[test]
    fn ratio_steps_split_the_range_evenly_and_clamp() {
        let last = RATIO_STEPS.len() - 1;
        assert_eq!(ratio_step(1.0), last / 2);
        assert_eq!(ratio_step(RATIO_SPAN), last);
        assert_eq!(ratio_step(1.0 / RATIO_SPAN), 0);
        assert_eq!(ratio_step(100.0), last);
        assert_eq!(ratio_step(0.01), 0);
        // The gray middle step holds the noise band, ×0.96 to ×1.04.
        assert_eq!(ratio_step(0.96), last / 2);
        assert_eq!(ratio_step(1.04), last / 2);
        assert_eq!(ratio_step(1.1), last / 2 + 1);
    }

    #[test]
    fn ratio_range_is_the_span_either_way() {
        assert_eq!(RATIO_RANGE.max, RATIO_SPAN.log2());
        assert_eq!(RATIO_RANGE.min, -RATIO_SPAN.log2());
    }

    #[test]
    fn ratio_steps_mirror_around_a_gray_middle() {
        let n = RATIO_STEPS.len();
        assert_eq!(n % 2, 1);
        assert_eq!(RATIO_STEPS[n / 2], "#a8a8a8");
    }
}
