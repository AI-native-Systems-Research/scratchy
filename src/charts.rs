//! The chart machinery the performance pages share: the specs charts.js
//! hands Carbon Charts, the ratio heat maps' colour scale, the "At a glance"
//! small multiples laid out beside it, and the sparkline that fills the gap
//! between two lines — blue where scratchy is ahead, red where it is behind.
//! metal.html and spyre.html render these with their own data; nothing here
//! knows a page's own numbers.

use std::collections::BTreeMap;
use std::fmt;

use dioxus::prelude::*;
use serde::Serialize;

use crate::carbon::{Alignment, Definition, Stack};

/// Draws every chart from its rendered spec.
pub const CHARTS_JS: &str = include_str!("charts.js");

/// A colour in each of the two Carbon themes a page follows the reader's
/// system setting between, white and g100 (chrome.rs). charts.js hands Carbon
/// Charts the one the page is in; in CSS it is `light-dark()`, which follows
/// the `color-scheme` Carbon's theme classes set.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Themed {
    pub light: &'static str,
    pub dark: &'static str,
}

impl Themed {
    /// The same colour in both themes.
    pub const fn both(colour: &'static str) -> Self {
        Themed {
            light: colour,
            dark: colour,
        }
    }
}

impl fmt::Display for Themed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "light-dark({}, {})", self.light, self.dark)
    }
}

/// One of Carbon's colour ramps, its grades 10 (lightest) to 100 (darkest).
#[derive(Clone, Copy)]
pub enum Ramp {
    Red,
    Cyan,
    Gray,
}

impl Ramp {
    const fn grades(self) -> [&'static str; 10] {
        match self {
            Ramp::Red => [
                "#fff1f1", "#ffd7d9", "#ffb3b8", "#ff8389", "#fa4d56", "#da1e28", "#a2191f",
                "#750e13", "#520408", "#2d0709",
            ],
            Ramp::Cyan => [
                "#e5f6ff", "#bae6ff", "#82cfff", "#33b1ff", "#1192e8", "#0072c3", "#00539a",
                "#003a6d", "#012749", "#061727",
            ],
            Ramp::Gray => [
                "#f4f4f4", "#e0e0e0", "#c6c6c6", "#a8a8a8", "#8d8d8d", "#6f6f6f", "#525252",
                "#393939", "#262626", "#161616",
            ],
        }
    }

    /// The ramp at GRADE on white, and at the mirrored grade, 100 − GRADE, on
    /// g100, the way Carbon's g100 mirrors white: a colour that stands out
    /// from white by its darkness stands out from g100 by its lightness. On
    /// g100 never lighter than 20: red and cyan 10 are all but white, too
    /// alike to tell faster from slower (OKLab ΔE 3.6, where red and cyan
    /// 100, white's darkest, are 9.4 apart and 20s are 9.7).
    pub const fn at<const GRADE: usize>(self) -> Themed {
        const {
            assert!(
                matches!(GRADE, 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100),
                "Carbon's grades are 10, 20, … 100"
            )
        };
        let dark = if GRADE >= 80 { 20 } else { 100 - GRADE };
        let grades = self.grades();
        Themed {
            light: grades[GRADE / 10 - 1],
            dark: grades[dark / 10 - 1],
        }
    }
}

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
/// against Carbon's white and g100 surfaces, so the same in both themes. Aqua
/// is under 3:1 on white, so every chart has a legend and a table view. A
/// previous run is a grey nearer the page than the series, so it reads as
/// behind.
pub fn colour(class: &str) -> Themed {
    match class {
        "s1" => Themed::both("#2a78d6"),
        "s2" => Themed::both("#d95926"),
        "s3" => Themed::both("#1baf7a"),
        _ => Ramp::Gray.at::<40>(),
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
    /// Carbon's ScatterChart: one dot a measurement.
    Scatter {
        data: Vec<DotPoint<'a>>,
        options: ChartOptions<'a>,
    },
}

/// One dot: its series, where along the bottom axis, and its value.
#[derive(Serialize)]
pub struct DotPoint<'a> {
    pub group: &'a str,
    pub key: f64,
    pub value: f64,
}

/// A scatter chart's dots: Carbon's `points` option.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Points {
    pub radius: f64,
    pub fill_opacity: f64,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<Points>,
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
/// heatmap's quantized scale. Each colour is Themed: charts.js gives Carbon
/// the page's.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Colour<'a> {
    Scale { scale: BTreeMap<&'a str, Themed> },
    Gradient { gradient: Gradient },
}

#[derive(Serialize)]
pub struct Gradient {
    pub colors: &'static [Themed],
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
/// start at red/cyan 40 and darken outward to 100, every one at least 2.36:1
/// on white and 2.15:1 on the grey tile (#f4f4f4). On g100 each grade is
/// mirrored (Ramp::at): the middle is gray 60, and the steps past it start at
/// red/cyan 60 and lighten outward to 20, every step from ×2.5 out (80 to 100
/// on white) 20, every one at least 3.6:1 on g100's page (#161616) and 3.0:1
/// on its tile (#262626).
pub const RATIO_STEPS: [Themed; 33] = {
    use Ramp::{Cyan, Gray, Red};
    [
        Red.at::<100>(), // ×0.25 or slower
        Red.at::<90>(),
        Red.at::<90>(),
        Red.at::<80>(),
        Red.at::<80>(),
        Red.at::<70>(),
        Red.at::<70>(),
        Red.at::<70>(),
        Red.at::<60>(),
        Red.at::<60>(),
        Red.at::<60>(),
        Red.at::<50>(),
        Red.at::<50>(),
        Red.at::<50>(),
        Red.at::<40>(),
        Red.at::<40>(),
        Gray.at::<40>(), // ×0.96 to ×1.04, within noise
        Cyan.at::<40>(),
        Cyan.at::<40>(),
        Cyan.at::<50>(),
        Cyan.at::<50>(),
        Cyan.at::<50>(),
        Cyan.at::<60>(),
        Cyan.at::<60>(),
        Cyan.at::<60>(),
        Cyan.at::<70>(),
        Cyan.at::<70>(),
        Cyan.at::<70>(),
        Cyan.at::<80>(),
        Cyan.at::<80>(),
        Cyan.at::<90>(),
        Cyan.at::<90>(),
        Cyan.at::<100>(), // ×4 or faster
    ]
};

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

impl Chart<'_> {
    fn options(&self) -> &ChartOptions<'_> {
        match self {
            Chart::GroupedBar { options, .. }
            | Chart::Heatmap { options, .. }
            | Chart::Combo { options, .. }
            | Chart::Scatter { options, .. } => options,
        }
    }
}

/// A chart's mount point, styled by `mount`: charts.js draws `spec` into it
/// once it nears the screen. Sized already as Carbon sizes it, to the
/// options' height and any width, so a chart drawn late moves nothing.
/// `scale` is the id of the ratio scale a heatmap's squares highlight on hover.
pub fn chart(spec: &Chart, label: &str, mount: &str, scale: Option<&str>) -> Element {
    let options = spec.options();
    let width = options
        .width
        .as_deref()
        .map(|w| format!("; width: {w}"))
        .unwrap_or_default();
    let height = &options.height;
    rsx! {
        div { role: "img", "aria-label": label, style: "{mount}{width}; height: {height}", "data-scale": scale,
            "data-chart": serde_json::to_string(spec).unwrap_or_default(),
        }
    }
}

/// The overview's ratio scale.
const GLANCE_SCALE: &str = "scale-glance";

/// The ratio heatmaps' colour scale, upright: RATIO_STEPS fastest at the top,
/// labelled at the ends and the middle. Carbon's heatmap only draws its legend
/// as a bar underneath.
pub fn ratio_scale(fit: ScaleFit, id: &str) -> Element {
    let rows = RATIO_STEPS.len();
    let middle = rows / 2 + 1;
    let (place, track) = match fit {
        ScaleFit::Fixed => (String::new(), "0.5rem"),
        ScaleFit::Beside => (
            // Stretched to the rows beside it: from the top of the first
            // row's grids (below its model names, 1.5rem) to the bottom of
            // the last's (above its two label lines, 2.25rem). Beside a
            // single row, it keeps a readable height.
            "flex: none; align-self: stretch; margin-block: 1.5rem 2.25rem; min-height: 12rem; "
                .to_string(),
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
            // Each label of no height, its text hanging from its step's top,
            // centred on its middle, or standing on its bottom: so the labels
            // never size the steps, and stretched over a single row the scale
            // is no taller than its rows (or its minimum height) makes it.
            div { style: "grid-column: 2; grid-row: 1; align-self: start; height: 0; display: flex; align-items: flex-start",
                Definition { alignment: Alignment::Left,
                    definition: "scratchy at least {RATIO_SPAN} times as fast as the engine the map compares it with; anything faster takes this colour too.",
                    strong { "×{RATIO_SPAN}" }
                }
            }
            div { style: "grid-column: 2; grid-row: {middle}; align-self: center; height: 0; display: flex; align-items: center",
                Definition { alignment: Alignment::Left,
                    definition: "The same speed, within run-to-run noise: ×0.96 to ×1.04.",
                    strong { "×1" }
                }
            }
            div { style: "grid-column: 2; grid-row: {rows}; align-self: end; height: 0; display: flex; align-items: flex-end",
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

/// A ratio map's two axes, read like a table: `across` left to right, `down`
/// top to bottom, each its title and its labels in order.
pub struct TableAxes<'a> {
    pub across: (&'a str, Vec<String>),
    pub down: (&'a str, Vec<String>),
}

impl<'a> TableAxes<'a> {
    /// As Carbon's label axes: across on x, down on y.
    pub fn axes(self, visible: Option<bool>) -> Axes<'a> {
        let labels = |maps_to, title, domain: Vec<String>| Axis {
            maps_to,
            title,
            scale_type: Some("labels"),
            ticks: None,
            include_zero: None,
            visible,
            domain: Some(AxisDomain::Labels(domain)),
        };
        let (across, (down, mut rows)) = (self.across, self.down);
        // Carbon draws a vertical axis's first label at the bottom.
        rows.reverse();
        Axes {
            bottom: labels("x", across.0, across.1),
            left: labels("y", down, rows),
        }
    }
}

/// One square of a ratio map: its place on the map's axes, how many times
/// faster scratchy is there (None: no ratio), and whether charts.js writes
/// that factor in it.
pub struct RatioSquare {
    pub across: String,
    pub down: String,
    pub ratio: Option<f64>,
    pub highlight: bool,
}

/// A grid of ratios as a tiny heatmap, colour only: no title, axes or
/// legend, what is around it names it. Its box is as wide as it is given and
/// as tall as square cells make it.
pub fn mini_heatmap(
    grid: TableAxes,
    data: Vec<HeatPoint>,
    total_label: String,
    label: &str,
) -> Element {
    let aspect = format!("{} / {}", grid.across.1.len(), grid.down.1.len());
    let spec = Chart::Heatmap {
        data,
        options: ChartOptions {
            title: String::new(),
            axes: grid.axes(Some(false)),
            color: Some(Colour::Gradient {
                gradient: Gradient {
                    colors: &RATIO_STEPS,
                },
            }),
            heatmap: Some(HeatmapOptions {
                color_legend: ColorLegend {
                    title: String::new(),
                    kind: "quantize",
                },
                color_domain: Some(RATIO_RANGE),
            }),
            // Its box decides.
            height: "100%".to_string(),
            width: Some("100%".to_string()),
            legend: Some(Enabled { enabled: false }),
            locale: Locale::total(total_label),
            combo: None,
            bounds: None,
            points: None,
            toolbar: Enabled { enabled: false },
        },
        ratio: true,
    };
    chart(
        &spec,
        label,
        &format!("width: 100%; aspect-ratio: {aspect}"),
        Some(GLANCE_SCALE),
    )
}

/// One of a glance's small multiples: `squares` as a mini_heatmap, `name`
/// over the range of its ratios under it, or over `none` when no square has
/// one: so a grid with nothing to show keeps its place and its shape, every
/// square blank. Half its cell (glance_maps), however wide the cell is.
/// `about` says whose grid it is.
pub fn glance_mini(
    grid: TableAxes,
    squares: Vec<RatioSquare>,
    total_label: String,
    name: &str,
    none: &str,
    about: &str,
) -> Element {
    let ratios: Vec<f64> = squares.iter().filter_map(|s| s.ratio).collect();
    let (range, label) = match (
        ratios.iter().copied().reduce(f64::min),
        ratios.iter().copied().reduce(f64::max),
    ) {
        (Some(lo), Some(hi)) => {
            let range = format!("×{lo:.2}–×{hi:.2}");
            let label = format!("{about}: scratchy {range}");
            (range, label)
        }
        _ => (none.to_string(), format!("{about}: {none}")),
    };
    let data = squares
        .into_iter()
        .map(|s| HeatPoint {
            x: s.across,
            y: s.down,
            value: s.ratio.map(log2_ratio),
            highlight: s.highlight,
        })
        .collect();
    rsx! {
        div { style: "flex: 1 1 0; min-width: 0",
            Stack { gap: 2,
                {mini_heatmap(grid, data, total_label, &label)}
                // Name over range, centred under the grid.
                div { style: "text-align: center",
                    div { strong { "{name}" } }
                    div { "{range}" }
                }
            }
        }
    }
}

/// A glance's heat maps beside their ratio scale: a row per `rows` (a
/// machine on metal.html, a dtype on spyre.html), its name, then a cell per
/// `columns` (a model), each the column's
/// name over `cell(row, column)`, that row's grids of that model as
/// glance_minis. The cells are one width, 11.5rem (two 88px grids and the 8px
/// between them), and on a narrow screen shrink together rather than wrap, as
/// the sparkline panels' columns do: so every grid on the page is one size,
/// the columns line up down the rows, and the scale stands beside them all.
/// Each row's names share a track and its grids the next, so a name that
/// wraps onto two lines leaves its grids level with their neighbours'.
pub fn glance_maps(
    rows: &[&str],
    columns: &[&str],
    column_width: &str,
    scale: bool,
    cell: impl Fn(&str, &str) -> Element,
) -> Element {
    let grid = format!(
        "flex: 0 1 auto; min-width: 0; align-self: start; display: grid; \
         grid-template-columns: 7rem repeat({}, {column_width}); gap: 1.5rem",
        columns.len()
    );
    rsx! {
        div { style: "display: flex; gap: 2rem",
            div { style: "{grid}",
                for row in rows {
                    div { style: "grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; row-gap: 0.5rem",
                        div { style: "grid-row: span 2", "{row}" }
                        for column in columns {
                            div { style: "align-self: end; text-align: center", "{column}" }
                        }
                        for column in columns {
                            div { style: "display: flex; gap: 0.5rem; align-items: flex-start",
                                {cell(row, column)}
                            }
                        }
                    }
                }
            }
            if scale { {ratio_scale(ScaleFit::Beside, GLANCE_SCALE)} }
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
///
/// Carrying (`carried`, a timeline): through steps only scratchy was measured
/// at, the rival's value carries forward from its last measurement — not a
/// new one. The rival's own line still breaks there; its carried level draws
/// as a second line dataset under `carried`'s name (dashed by charts.js,
/// which Carbon's options cannot ask for), bridging the measured steps either
/// side, and the fills and the highlights' factors follow it, so a step past
/// the rival's newest measurement still pairs with the level it is compared
/// against everywhere else on the page.
pub fn spark_points(
    pts: &[Step],
    better: Better,
    scratchy: Side,
    rival: Side,
    carried: Option<&'static str>,
) -> Vec<SparkPoint> {
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
    // The rival's value at each step, carried forward where only scratchy was
    // measured: the level of its last measurement, not a new one.
    let eff: Vec<Option<f64>> = {
        let mut last = None;
        pts.iter()
            .map(|st| {
                if st.rival.is_some() {
                    last = st.rival;
                }
                match (carried, st.scratchy, st.rival) {
                    (Some(_), Some(_), None) => last,
                    (_, _, rv) => rv,
                }
            })
            .collect()
    };
    // A step whose rival value is carried, not measured.
    let is_carried = |j: usize| carried.is_some() && pts[j].rival.is_none() && eff[j].is_some();
    let mut out: Vec<SparkPoint> = Vec::new();
    for (
        k,
        &Step {
            x,
            scratchy: sv,
            rival: rv,
            ..
        },
    ) in pts.iter().enumerate()
    {
        out.push(line(scratchy.line, x, sv));
        out.push(line(rival.line, x, rv));
        if let Some(id) = carried {
            if is_carried(k) {
                out.push(line(id, x, eff[k]));
            } else if rv.is_some()
                && ((k > 0 && is_carried(k - 1)) || k + 1 < pts.len() && is_carried(k + 1))
            {
                // The measured step beside a carried one: the dotted line
                // starts at, and comes back to, the line it continues.
                out.push(line(id, x, rv));
            }
        }
    }
    let mut fills: Vec<SparkPoint> = Vec::new();
    for (
        k,
        &Step {
            x,
            scratchy: sv,
            highlight,
            ..
        },
    ) in pts.iter().enumerate()
    {
        let rv = eff[k];
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
                if k + 1 < pts.len()
                    && let (Some(s1), Some(r1)) = (pts[k + 1].scratchy, eff[k + 1])
                {
                    let d1 = lead(s1, r1);
                    if d * d1 < 0.0 {
                        let x1 = pts[k + 1].x;
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

/// What a sparkline follows: which way its values are better, its axes'
/// titles, which are the tooltip's labels — and whether the rival's line
/// carries forward, dotted, through steps only scratchy was measured at
/// (a timeline: no PR re-measures torch, so its newest measurement is the
/// level a newer scratchy value pairs with).
#[derive(Clone, Copy)]
pub struct SparkAxes {
    pub better: Better,
    pub along: &'static str,
    pub value: &'static str,
    pub carry: bool,
}

/// A sparkline: scratchy's and its rival's values along one dimension, the gap
/// between them filled blue where scratchy is ahead and red where it is behind.
/// Carrying (SparkAxes), the rival's line continues dotted past its last
/// measurement through steps only scratchy was measured at.
pub fn sparkline(
    pts: &[Step],
    axes: SparkAxes,
    scratchy: Side,
    rival: Side,
    names: Option<Vec<String>>,
    height: &str,
    label: &str,
) -> Element {
    // The rival's carried continuation is its own line dataset — Carbon has
    // no way to dash part of one series, so charts.js dashes this one by its
    // name — in the rival's own colour. The id must live as long as the
    // spec; the build is a one-shot process, so one leak a chart is sound.
    let carried: Option<&'static str> = axes
        .carry
        .then(|| Box::leak(format!("{} (carried)", rival.line).into_boxed_str()) as &'static str);
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
        data: spark_points(pts, axes.better, scratchy, rival, carried),
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
                    (scratchy.fill, Ramp::Cyan.at::<50>()),
                    (rival.fill, Ramp::Red.at::<50>()),
                ]
                .into_iter()
                .chain(carried.map(|id| (id, colour("s2"))))
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
                    corresponding_datasets: match carried {
                        Some(id) => vec![scratchy.line, rival.line, id],
                        None => vec![scratchy.line, rival.line],
                    },
                },
            ]),
            bounds: Some(Bounds {
                upper_bound_maps_to: "upper",
                lower_bound_maps_to: "lower",
            }),
            points: None,
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
        Axis, AxisDomain, Better, RATIO_RANGE, RATIO_SPAN, RATIO_STEPS, Ramp, Side, Step,
        TableAxes, ratio_step, spark_points,
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
        let pts = spark_points(&steps, Better::Higher, SCRATCHY, RIVAL, None);
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
        let ttft = spark_points(&steps, Better::Lower, SCRATCHY, RIVAL, None);
        assert!(
            ttft.iter()
                .any(|p| p.group == RIVAL.fill && p.key == 1.0 && p.upper.is_some())
        );
    }

    #[test]
    fn carrying_extends_the_rival_dotted_and_the_fill() {
        // Both stacks measured, then two steps only scratchy was: the rival's
        // line breaks, its carried level draws dotted from the last measured
        // point, and the fill and the highlight's factor follow that level.
        let mut steps = [
            Step {
                x: 1.0,
                scratchy: Some(10.0),
                rival: Some(5.0),
                highlight: false,
            },
            Step {
                x: 2.0,
                scratchy: Some(12.0),
                rival: None,
                highlight: false,
            },
            Step {
                x: 4.0,
                scratchy: Some(14.0),
                rival: None,
                highlight: true,
            },
        ];
        let pts = spark_points(
            &steps,
            Better::Higher,
            SCRATCHY,
            RIVAL,
            Some("mlx-lm (carried)"),
        );
        let at = |group| -> Vec<(f64, Option<f64>)> {
            pts.iter()
                .filter(|p| p.group == group)
                .map(|p| (p.key, p.value))
                .collect()
        };
        // The rival's own line breaks at the steps it was not measured at.
        assert_eq!(at(RIVAL.line), [(1.0, Some(5.0)), (2.0, None), (4.0, None)]);
        // The carried line: a bridge at the last measured point, then the
        // carried level at each step past it.
        assert_eq!(
            at("mlx-lm (carried)"),
            [(1.0, Some(5.0)), (2.0, Some(5.0)), (4.0, Some(5.0))]
        );
        // The fill continues over the carried steps, against that level.
        let fill = |group| -> Vec<(f64, Option<(f64, f64)>)> {
            pts.iter()
                .filter(|p| p.group == group)
                .map(|p| (p.key, p.lower.zip(p.upper)))
                .collect()
        };
        assert_eq!(
            fill(SCRATCHY.fill),
            [
                (1.0, Some((5.0, 10.0))),
                (2.0, Some((5.0, 12.0))),
                (4.0, Some((5.0, 14.0)))
            ]
        );
        // The highlight's factor pairs the newest value with the carried
        // level — the same pairing the page makes everywhere else.
        assert_eq!(
            pts.iter()
                .find(|p| p.group == SCRATCHY.fill && p.key == 4.0)
                .and_then(|p| p.factor),
            Some(14.0 / 5.0)
        );
        // Without carrying, the same steps leave the fill broken and no
        // carried line at all.
        steps[2].highlight = false;
        let plain = spark_points(&steps, Better::Higher, SCRATCHY, RIVAL, None);
        assert!(plain.iter().all(|p| p.group != "mlx-lm (carried)"));
        assert_eq!(
            plain
                .iter()
                .filter(|p| p.group == SCRATCHY.fill)
                .map(|p| (p.key, p.lower.zip(p.upper)))
                .collect::<Vec<_>>(),
            [(1.0, Some((5.0, 10.0))), (2.0, None), (4.0, None)]
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
        assert_eq!(RATIO_STEPS[n / 2], Ramp::Gray.at::<40>());
        assert_eq!(RATIO_STEPS[n / 2].dark, "#6f6f6f");
    }

    /// WCAG's contrast ratio of two "#rrggbb" colours.
    fn contrast(a: &str, b: &str) -> f64 {
        let luminance = |hex: &str| {
            let channel = |at: usize| {
                let v = f64::from(u8::from_str_radix(&hex[at..at + 2], 16).unwrap()) / 255.0;
                if v <= 0.03928 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5)
        };
        let (x, y) = (luminance(a), luminance(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    #[test]
    fn every_ratio_step_stands_out_from_its_themes_page_and_tile() {
        for step in RATIO_STEPS {
            assert!(contrast(step.light, "#ffffff") >= 2.36, "{step:?} on white");
            assert!(
                contrast(step.light, "#f4f4f4") >= 2.15,
                "{step:?} on white's tile"
            );
            assert!(contrast(step.dark, "#161616") >= 3.6, "{step:?} on g100");
            assert!(
                contrast(step.dark, "#262626") >= 3.0,
                "{step:?} on g100's tile"
            );
        }
    }

    #[test]
    fn a_table_reads_its_first_row_on_top() {
        let labels = |ls: &[&str]| ls.iter().map(|l| l.to_string()).collect::<Vec<_>>();
        let axes = TableAxes {
            across: ("context", labels(&["~123", "~1k", "~10k"])),
            down: ("dtype", labels(&["fp16", "fp8"])),
        }
        .axes(None);
        let domain = |a: &Axis| match &a.domain {
            Some(AxisDomain::Labels(ls)) => ls.clone(),
            _ => Vec::new(),
        };
        assert_eq!(domain(&axes.bottom), labels(&["~123", "~1k", "~10k"]));
        // Carbon draws a vertical axis's first label at the bottom.
        assert_eq!(domain(&axes.left), labels(&["fp8", "fp16"]));
    }
}
