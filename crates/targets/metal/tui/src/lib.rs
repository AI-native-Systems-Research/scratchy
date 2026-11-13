//! scratchy-tui — a terminal agent UI that drives goose's core `Agent` with
//! scratchy as an in-process provider. goose's agent brain (loop, tools/MCP,
//! sessions) linked as a library + scratchy's engine in process. No goose CLI,
//! no ACP, no server.
//!
//! Library crate: the `scr` CLI drives it via [`run`] (behind its `tui`
//! feature) — `scr tui [MODEL] ["<prompt>"]`.

mod provider;

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use goose::agents::{Agent, AgentEvent, SessionConfig};
use goose::config::{ExtensionConfig, GooseMode};
use goose::conversation::message::{Message, MessageContent, ToolRequest, ToolResponse};
use goose::session::session_manager::SessionType;
use goose_provider_types::model::ModelConfig;
use provider::ScratchyProvider;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use scratchy_core_common::forward_telemetry::{
    ForwardRecord, ForwardTelemetry, KernelKind, TapeEntry,
};
use scratchy_core_common::sampler_telemetry::{SamplerRecord, SamplerTelemetry};
use scratchy_serving_api::engine::LiveStats;

/// Build an Agent pointed at the in-process scratchy provider + the `developer`
/// builtin (shell/editor tools), in Auto mode (tool calls auto-approved).
async fn setup(model: &str) -> Result<(Agent, String, Arc<ScratchyProvider>)> {
    let agent = Agent::new();

    let session = agent
        .config
        .session_manager
        .clone()
        .create_session(
            std::env::current_dir()?,
            "scratchy-tui".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let session_id = session.id.clone();

    // Construct the scratchy provider directly (it lives in this crate) and hand
    // it to the agent — no goose-side registry, no goose dependency on scratchy.
    let provider = Arc::new(ScratchyProvider::from_env().await?);
    agent
        .update_provider(provider.clone(), ModelConfig::new(model), &session_id)
        .await?;
    agent
        .update_goose_mode(GooseMode::Auto, &session_id)
        .await?;

    // Builtin platform extensions, loaded in-process (no external MCP server):
    //   developer — write/edit/shell/tree/read_image
    //   todo      — a scratch task list the agent maintains
    //   analyze   — tree-sitter code structure (overviews, symbols, call graphs)
    for (name, display, description) in [
        ("developer", "Developer", "Files and shell"),
        ("todo", "Todo", "A todo list the agent keeps"),
        (
            "analyze",
            "Analyze",
            "Code structure analysis (tree-sitter)",
        ),
    ] {
        let ext = ExtensionConfig::Builtin {
            name: name.to_string(),
            description: description.to_string(),
            display_name: Some(display.to_string()),
            timeout: None,
            bundled: None,
            available_tools: Vec::new(),
        };
        if let Err(e) = agent.add_extension(ext, &session_id).await {
            eprintln!("[scratchy-tui] {name} extension failed to load: {e}");
        }
    }

    Ok((agent, session_id, provider))
}

fn session_config(id: &str) -> SessionConfig {
    SessionConfig {
        id: id.to_string(),
        schedule_id: None,
        max_turns: Some(50),
        retry_config: None,
    }
}

/// Collapse whitespace and hard-cap a string to a single display line.
fn one_line(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > max {
        let head: String = flat.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    } else {
        flat
    }
}

/// A `width`-cell progress bar as (bright filled, dim empty) spans, with
/// sub-cell resolution so even a few-percent fill shows a visible sliver.
fn bar_spans(frac: f64, width: usize) -> Vec<Span<'static>> {
    const EIGHTHS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let frac = frac.clamp(0.0, 1.0);
    let eighths = (frac * (width * 8) as f64).round() as usize;
    let full = (eighths / 8).min(width);
    let rem = eighths % 8;
    let mut filled = "█".repeat(full);
    let mut cells = full;
    if full < width && rem > 0 {
        filled.push(EIGHTHS[rem]);
        cells += 1;
    }
    let empty = "░".repeat(width.saturating_sub(cells));
    vec![
        Span::styled(filled, Style::new().fg(Color::Indexed(75))),
        Span::styled(empty, Style::new().fg(Color::Indexed(238))),
    ]
}

/// Format an integer with thousands separators (3412 → "3,412").
fn commas(n: u64) -> String {
    let s = n.to_string();
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*c as char);
    }
    out
}

/// Human-readable byte size (1_610_612_736 → "1.5 GB").
fn fmt_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = n as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{n} B")
    }
}

/// Compact label for a shard file in the download modal: prefer the
/// `NNNNN-of-MMMMM` index HF shard names carry (`model-00002-of-00005…` →
/// `02/05`), else the filename trimmed of its extension, capped in width.
fn shard_label(name: &str) -> String {
    if let Some(idx) = name.find("-of-") {
        // Walk back over the digits before "-of-" for the shard number.
        let pre = &name[..idx];
        let k: String = pre
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let rest = &name[idx + 4..];
        let m: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !k.is_empty() && !m.is_empty() {
            // Trim leading zeros but keep at least one digit.
            let trim = |s: String| s.trim_start_matches('0').to_string();
            let k = trim(k.clone());
            let m = trim(m.clone());
            return format!(
                "{}/{}",
                if k.is_empty() { "0".into() } else { k },
                if m.is_empty() { "0".into() } else { m }
            );
        }
    }
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    one_line(stem, 16)
}

/// A `w`×`h` rectangle centered within `area` (clamped to `area`'s size), for
/// modal overlays.
fn centered_rect(w: u16, h: u16, area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    ratatui::layout::Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

/// Compact magnitude form for large counts (12431 → "12.4k", 2_100_000 → "2.1M").
fn kmg(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

/// A unicode block sparkline of `vals`, each scaled against `cap` (clamped).
fn sparkline(vals: &[f32], cap: f32) -> String {
    const B: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    vals.iter()
        .map(|&v| {
            let f = (v / cap).clamp(0.0, 1.0);
            B[((f * 7.0).round() as usize).min(7)]
        })
        .collect()
}

/// Escape a decoded token for display so control characters stay visible
/// (`\n`, `\t`), without wrapping it in quotes.
fn display_token(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => out.push('·'),
            c => out.push(c),
        }
    }
    out
}

/// Truncate a display string to `max` character cells, appending `…` if cut.
fn truncate_cells(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars()
        .take(max.saturating_sub(1))
        .chain(std::iter::once('…'))
        .collect()
}

/// Palette for the forward-pass tape, one distinct 256-color per kernel family.
fn kind_color(k: KernelKind) -> Color {
    use KernelKind::*;
    Color::Indexed(match k {
        Embed => 213,
        Norm => 244,
        Rope => 179,
        Attention => 75,
        GatedDeltaNet => 141,
        Gemm => 42,
        Moe => 208,
        Mlp => 79,
        Sample => 201,
        Vision => 168,
        Elementwise => 240,
        Other => 245,
    })
}

/// Build the scrolling tape ribbon. Each row is one kernel dispatch:
/// `[▸][bar…] label[+]` — a left `▸` marks a compiler-inserted barrier before
/// the kernel, a bright `+` after the label marks a fused kernel, and the bar's
/// color is the family. Consecutive same-color rows are consecutive dispatches.
fn tape_lines(tape: &[TapeEntry], head: usize, height: usize, w: usize) -> Vec<Line<'static>> {
    let len = tape.len().max(1);
    // layout: barrier(1) + space(1) + bar(w-9) + space(1) + label(6)
    let bar_w = w.saturating_sub(9);
    let barrier_style = Style::new().fg(Color::Indexed(209));
    let fuse_style = Style::new()
        .fg(Color::Indexed(220))
        .add_modifier(Modifier::BOLD);
    let mut lines = Vec::with_capacity(height);
    for r in 0..height {
        let e = tape[(head + r) % len];
        let color = kind_color(e.kind);
        let mut spans = vec![
            Span::styled(if e.barrier { "▸" } else { " " }, barrier_style),
            Span::raw(" "),
            Span::styled("█".repeat(bar_w), Style::new().fg(color)),
            Span::raw(" "),
            Span::styled(
                format!("{:<5}", e.kind.label()),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ),
        ];
        spans.push(if e.fused {
            Span::styled("+", fuse_style)
        } else {
            Span::raw(" ")
        });
        lines.push(Line::from(spans));
    }
    lines
}

/// Horizontal histogram of inter-token latencies (ms), one bin per row, ascending.
/// Binned over a robust [p2, p95] range so a few slow tokens don't squish the
/// bulk; the top bin is open-ended (`+`) and catches the clipped tail.
fn itl_hist_lines(
    samples: &[f32],
    generating: bool,
    height: usize,
    width: usize,
) -> Vec<Line<'static>> {
    let dim = Style::new().fg(Color::DarkGray);
    if samples.len() < 4 {
        // Only claim "collecting" while a turn is actually running; when idle
        // (e.g. after Esc) nothing more is coming, so don't imply activity.
        let msg = if generating {
            "  collecting…"
        } else {
            "  —"
        };
        return vec![Line::from(Span::styled(msg, dim))];
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let pct = |p: f32| sorted[((p * (n - 1) as f32).round() as usize).min(n - 1)];
    let lo = pct(0.02);
    let hi = pct(0.95).max(lo + 1.0);
    let nbins = height.clamp(1, 8);
    let binw = (hi - lo) / nbins as f32;
    let mut counts = vec![0usize; nbins];
    for &v in samples {
        let b = (((v - lo) / binw).floor().max(0.0) as usize).min(nbins - 1);
        counts[b] += 1;
    }
    let maxc = counts.iter().copied().max().unwrap_or(1).max(1);
    let bar_w = width.saturating_sub(11);
    counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let edge = lo + binw * i as f32;
            let mark = if i == nbins - 1 { "+" } else { " " };
            let fill = ((c as f32 / maxc as f32) * bar_w as f32).round() as usize;
            Line::from(vec![
                Span::styled(format!("{edge:>4.0}{mark} "), dim),
                Span::styled("█".repeat(fill), Style::new().fg(Color::Indexed(75))),
                Span::styled(format!(" {c}"), Style::new().fg(Color::Gray)),
            ])
        })
        .collect()
}

/// Per-family kernel distribution (by dispatch count) as sorted horizontal bars.
fn kernel_dist_lines(tape: &[TapeEntry], height: usize, width: usize) -> Vec<Line<'static>> {
    let mut rows: Vec<(KernelKind, usize)> = KernelKind::ALL
        .iter()
        .map(|&k| (k, tape.iter().filter(|e| e.kind == k).count()))
        .filter(|(_, c)| *c > 0)
        .collect();
    rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let maxc = rows.iter().map(|(_, c)| *c).max().unwrap_or(1).max(1);
    let bar_w = width.saturating_sub(11);
    rows.into_iter()
        .take(height)
        .map(|(k, c)| {
            let fill = ((c as f32 / maxc as f32) * bar_w as f32).round() as usize;
            Line::from(vec![
                Span::styled(
                    format!("{:<5}", k.label()),
                    Style::new().fg(kind_color(k)).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled("█".repeat(fill), Style::new().fg(kind_color(k))),
                Span::styled(format!(" {c}"), Style::new().fg(Color::Gray)),
            ])
        })
        .collect()
}

/// Sampler candidate bars + entropy sparkline, sized to `width`.
fn sampler_lines(
    cands: &[(String, f32, bool)],
    entropy_hist: &[f32],
    height: usize,
    width: usize,
) -> Vec<Line<'static>> {
    let dim = Style::new().fg(Color::DarkGray);
    let accent = Style::new().fg(Color::Indexed(75));
    let label_w = (width / 3).clamp(5, 12);
    let bar_w = width.saturating_sub(2 + label_w + 1 + 4 + 1);
    let mut lines = Vec::with_capacity(height);
    for (disp, prob, sampled) in cands.iter().take(height.saturating_sub(1)) {
        let lbl = truncate_cells(disp, label_w);
        let lbl_style = if *sampled {
            Style::new().fg(Color::Gray).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(Color::Gray)
        };
        let mut spans = vec![
            Span::styled(
                if *sampled { "● " } else { "  " },
                if *sampled { accent } else { dim },
            ),
            Span::styled(format!("{lbl:<label_w$}"), lbl_style),
            Span::raw(" "),
            Span::styled(format!("{prob:.2}"), dim),
            Span::raw(" "),
        ];
        spans.extend(bar_spans(*prob as f64, bar_w));
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(vec![
        Span::styled("entropy ", dim),
        Span::styled(sparkline(entropy_hist, 6.0), accent),
    ]));
    lines
}

/// A tool call split into `(tool_name, salient_argument)` — for shell the arg
/// is the command line, for the editor the op + path, else compact JSON.
fn fmt_call_parts(req: &ToolRequest) -> (String, String) {
    match &req.tool_call {
        Ok(call) => {
            let name = call.name.to_string();
            let args = call
                .arguments
                .as_ref()
                .map(|a| {
                    let path = a.get("path").and_then(|v| v.as_str());
                    match a.get("command").and_then(|v| v.as_str()) {
                        Some(cmd) => match path {
                            Some(p) => format!("{cmd}  {p}"),
                            None => cmd.to_string(),
                        },
                        None => match path {
                            Some(p) => p.to_string(),
                            None => serde_json::to_string(a).unwrap_or_default(),
                        },
                    }
                })
                .unwrap_or_default();
            (name, args)
        }
        Err(e) => ("⚠".to_string(), format!("unparseable: {e}")),
    }
}

/// Flattened one-line form of a call — used only for loop detection.
fn fmt_call(req: &ToolRequest) -> String {
    let (name, args) = fmt_call_parts(req);
    if args.is_empty() {
        name
    } else {
        format!("{name}  {}", one_line(&args, 240))
    }
}

/// Flattened one-line form of a result — used only for loop detection.
fn fmt_result(resp: &ToolResponse) -> String {
    one_line(&result_full(resp), 240)
}

/// The full, multi-line result text (newlines preserved) for display.
fn result_full(resp: &ToolResponse) -> String {
    match &resp.tool_result {
        Ok(r) => {
            let text = r
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.clone()))
                .collect::<Vec<_>>()
                .join("\n");
            if text.trim().is_empty() {
                "(no output)".to_string()
            } else {
                text
            }
        }
        Err(e) => format!("error: {e}"),
    }
}

fn history_file() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".scratchy-tui-history"))
}

fn load_history() -> Vec<String> {
    history_file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| {
            s.lines()
                .map(str::to_string)
                .filter(|l| !l.trim().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn append_history(line: &str) {
    use std::io::Write;
    if let Some(p) = history_file()
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// One turn's worth of transcript in the conversation pane.
enum Entry {
    User(String),
    Model(String),
    Tool {
        name: String,
        args: String,
    },
    ToolResult(String),
    /// The agent's `todo` tool write — its plan, rendered as a checklist.
    TodoPlan(String),
    Error(String),
    /// The user cancelled the in-flight turn (Esc).
    Cancelled,
}

/// If this tool call is the `todo` extension's write, pull out the plan
/// (`content`) so the TUI can pretty-print it instead of showing raw args.
fn todo_plan_content(req: &ToolRequest) -> Option<String> {
    let call = req.tool_call.as_ref().ok()?;
    if !call.name.ends_with("todo_write") {
        return None;
    }
    call.arguments
        .as_ref()?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Greedy word-wrap to `width` columns, preserving explicit newlines and
/// hard-splitting any single word longer than the width.
fn wrap_text(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    for para in s.split('\n') {
        let mut line = String::new();
        for word in para.split(' ').filter(|w| !w.is_empty()) {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
            while line.chars().count() > width {
                let head: String = line.chars().take(width).collect();
                out.push(head);
                line = line.chars().skip(width).collect();
            }
        }
        out.push(line);
    }
    out
}

/// Minimal markdown → styled, width-wrapped lines: bold/italic/inline-code,
/// headings, bullet/ordered lists, code blocks, blockquotes.
/// Lazily-built (once) syntect syntax set + theme for code highlighting.
fn highlight_assets() -> &'static (syntect::parsing::SyntaxSet, syntect::highlighting::Theme) {
    static A: std::sync::OnceLock<(syntect::parsing::SyntaxSet, syntect::highlighting::Theme)> =
        std::sync::OnceLock::new();
    A.get_or_init(|| {
        let ss = syntect::parsing::SyntaxSet::load_defaults_newlines();
        let ts = syntect::highlighting::ThemeSet::load_defaults();
        let theme = ts
            .themes
            .get("base16-ocean.dark")
            .or_else(|| ts.themes.values().next())
            .cloned()
            .expect("a default syntect theme");
        (ss, theme)
    })
}

/// Highlight a fenced code block into styled lines, prefixed with `indent`.
fn highlight_code(code: &str, lang: &str, indent: &str) -> Vec<Line<'static>> {
    use syntect::easy::HighlightLines;
    use syntect::util::LinesWithEndings;
    let (ss, theme) = highlight_assets();
    let syntax = (!lang.is_empty())
        .then(|| ss.find_syntax_by_token(lang))
        .flatten()
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut hl = HighlightLines::new(syntax, theme);
    let mut out = Vec::new();
    for line in LinesWithEndings::from(code) {
        let ranges = hl.highlight_line(line, ss).unwrap_or_default();
        let mut spans: Vec<Span<'static>> = Vec::new();
        if !indent.is_empty() {
            spans.push(Span::raw(indent.to_string()));
        }
        for (sty, text) in ranges {
            let text = text.trim_end_matches('\n');
            if text.is_empty() {
                continue;
            }
            let fg = sty.foreground;
            spans.push(Span::styled(
                text.to_string(),
                Style::new().fg(Color::Rgb(fg.r, fg.g, fg.b)),
            ));
        }
        out.push(Line::from(spans));
    }
    out
}

/// Truncate + pad a table cell to `w` cells, honoring the column alignment.
fn pad_cell(s: &str, w: usize, align: pulldown_cmark::Alignment) -> String {
    use pulldown_cmark::Alignment;
    let t = truncate_cells(s, w);
    let pad = w.saturating_sub(t.chars().count());
    match align {
        Alignment::Right => format!("{}{t}", " ".repeat(pad)),
        Alignment::Center => {
            let l = pad / 2;
            format!("{}{t}{}", " ".repeat(l), " ".repeat(pad - l))
        }
        _ => format!("{t}{}", " ".repeat(pad)),
    }
}

/// Accumulated markdown table (plain cell text; inline styling within cells is
/// dropped for v1).
#[derive(Default)]
struct MdTable {
    aligns: Vec<pulldown_cmark::Alignment>,
    rows: Vec<Vec<String>>,
    header_rows: usize,
    cur_row: Vec<String>,
    cur_cell: String,
}

struct Md {
    width: usize,
    lines: Vec<Line<'static>>,
    cur: Vec<Span<'static>>,
    cur_w: usize,
    bold: u32,
    italic: u32,
    strike: u32,
    code: bool,
    indent: String,
    pending: Option<String>,
    list: Vec<Option<u64>>,
    in_code_block: bool,
    code_lang: String,
    code_buf: String,
    table: Option<MdTable>,
    need_blank: bool,
}

impl Md {
    fn new(width: usize) -> Self {
        Self {
            width: width.max(8),
            lines: Vec::new(),
            cur: Vec::new(),
            cur_w: 0,
            bold: 0,
            italic: 0,
            strike: 0,
            code: false,
            indent: String::new(),
            pending: None,
            list: Vec::new(),
            in_code_block: false,
            code_lang: String::new(),
            code_buf: String::new(),
            table: None,
            need_blank: false,
        }
    }

    fn style(&self) -> Style {
        let mut m = Modifier::empty();
        if self.bold > 0 {
            m |= Modifier::BOLD;
        }
        if self.italic > 0 {
            m |= Modifier::ITALIC;
        }
        if self.strike > 0 {
            m |= Modifier::CROSSED_OUT;
        }
        let s = Style::new().add_modifier(m);
        if self.code {
            s.fg(Color::Indexed(180))
        } else {
            s
        }
    }

    fn eff_width(&self) -> usize {
        self.width
            .saturating_sub(self.indent.chars().count())
            .max(4)
    }

    fn flush(&mut self) {
        if self.cur.is_empty() && self.pending.is_none() {
            return;
        }
        let prefix = self.pending.take().unwrap_or_else(|| self.indent.clone());
        let mut spans = Vec::with_capacity(self.cur.len() + 1);
        if !prefix.is_empty() {
            spans.push(Span::raw(prefix));
        }
        spans.append(&mut self.cur);
        self.lines.push(Line::from(spans));
        self.cur_w = 0;
    }

    fn blank(&mut self) {
        if self.need_blank && !self.lines.is_empty() {
            self.lines.push(Line::default());
        }
        self.need_blank = false;
    }

    fn word(&mut self, w: &str) {
        let ww = w.chars().count();
        if self.cur_w > 0 && self.cur_w + 1 + ww > self.eff_width() {
            self.flush();
        }
        if self.cur_w > 0 {
            self.cur.push(Span::raw(" "));
            self.cur_w += 1;
        }
        let st = self.style();
        self.cur.push(Span::styled(w.to_string(), st));
        self.cur_w += ww;
    }

    fn text(&mut self, t: &str) {
        if self.in_code_block {
            // Buffered whole so the block can be syntax-highlighted at its end.
            self.code_buf.push_str(t);
        } else {
            for w in t.split_whitespace() {
                self.word(w);
            }
        }
    }

    /// Render an accumulated table into `self.lines` as an aligned grid, fit to
    /// the available width.
    fn render_table(&mut self, t: MdTable) {
        if t.rows.is_empty() {
            return;
        }
        let ncols = t.rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if ncols == 0 {
            return;
        }
        let mut widths = vec![0usize; ncols];
        for r in &t.rows {
            for (i, c) in r.iter().enumerate() {
                widths[i] = widths[i].max(c.chars().count());
            }
        }
        // Shrink proportionally if the natural width overflows the pane.
        let overhead = (ncols + 1) + 2 * ncols;
        let budget = self.eff_width().saturating_sub(overhead).max(ncols);
        let total: usize = widths.iter().sum();
        if total > budget && total > 0 {
            for w in widths.iter_mut() {
                *w = (((*w as f64) * (budget as f64) / (total as f64)).floor() as usize).max(3);
            }
        }
        let indent = self.indent.clone();
        let border = Style::new().fg(Color::Indexed(238));
        self.blank();
        for (idx, row) in t.rows.iter().enumerate() {
            let header = idx < t.header_rows;
            let mut spans: Vec<Span<'static>> =
                vec![Span::raw(indent.clone()), Span::styled("│", border)];
            for (i, &w) in widths.iter().enumerate() {
                let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                let align = t
                    .aligns
                    .get(i)
                    .copied()
                    .unwrap_or(pulldown_cmark::Alignment::None);
                let st = if header {
                    Style::new().add_modifier(Modifier::BOLD)
                } else {
                    Style::new()
                };
                spans.push(Span::raw(" "));
                spans.push(Span::styled(pad_cell(cell, w, align), st));
                spans.push(Span::raw(" "));
                spans.push(Span::styled("│", border));
            }
            self.lines.push(Line::from(spans));
            if idx + 1 == t.header_rows {
                let mut s = String::from("├");
                for (i, w) in widths.iter().enumerate() {
                    s.push_str(&"─".repeat(w + 2));
                    s.push(if i + 1 < ncols { '┼' } else { '┤' });
                }
                self.lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(s, border),
                ]));
            }
        }
        self.need_blank = true;
    }

    fn event(&mut self, ev: pulldown_cmark::Event) {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};
        // While inside a table, route cell text + structural events to the
        // accumulator (inline styling within cells is dropped for v1).
        if self.table.is_some() {
            match ev {
                Event::Start(Tag::TableHead) | Event::Start(Tag::TableRow) => {
                    self.table.as_mut().unwrap().cur_row.clear();
                }
                Event::Start(Tag::TableCell) => self.table.as_mut().unwrap().cur_cell.clear(),
                Event::Text(t) | Event::Code(t) => {
                    self.table.as_mut().unwrap().cur_cell.push_str(&t);
                }
                Event::End(TagEnd::TableCell) => {
                    let tbl = self.table.as_mut().unwrap();
                    let cell = tbl.cur_cell.trim().to_string();
                    tbl.cur_row.push(cell);
                }
                Event::End(TagEnd::TableRow) => {
                    let tbl = self.table.as_mut().unwrap();
                    let row = std::mem::take(&mut tbl.cur_row);
                    tbl.rows.push(row);
                }
                Event::End(TagEnd::TableHead) => {
                    let tbl = self.table.as_mut().unwrap();
                    let row = std::mem::take(&mut tbl.cur_row);
                    tbl.rows.push(row);
                    tbl.header_rows = tbl.rows.len();
                }
                Event::End(TagEnd::Table) => {
                    let tbl = self.table.take().unwrap();
                    self.render_table(tbl);
                }
                _ => {}
            }
            return;
        }
        match ev {
            Event::Start(Tag::Table(aligns)) => {
                self.flush();
                self.table = Some(MdTable {
                    aligns,
                    ..Default::default()
                });
            }
            Event::Start(Tag::Paragraph) => self.blank(),
            Event::Start(Tag::Heading { .. }) => {
                self.blank();
                self.bold += 1;
            }
            Event::Start(Tag::Strong) => self.bold += 1,
            Event::Start(Tag::Emphasis) => self.italic += 1,
            Event::Start(Tag::Strikethrough) => self.strike += 1,
            Event::Start(Tag::CodeBlock(kind)) => {
                self.flush();
                self.blank();
                self.in_code_block = true;
                self.code_lang = match kind {
                    CodeBlockKind::Fenced(l) => l.to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                self.code_buf.clear();
            }
            Event::Start(Tag::BlockQuote(_)) => {
                self.blank();
                self.indent.push_str("│ ");
            }
            Event::Start(Tag::List(start)) => {
                if !self.list.is_empty() {
                    self.indent.push_str("  ");
                }
                self.list.push(start);
            }
            Event::Start(Tag::Item) => {
                self.flush();
                let bullet = match self.list.last_mut() {
                    Some(Some(n)) => {
                        let s = format!("{n}. ");
                        *n += 1;
                        s
                    }
                    _ => "• ".to_string(),
                };
                self.pending = Some(format!("{}{}", self.indent, bullet));
            }
            Event::End(TagEnd::Paragraph) => {
                self.flush();
                self.need_blank = true;
            }
            Event::End(TagEnd::Heading(_)) => {
                self.flush();
                self.bold = self.bold.saturating_sub(1);
                self.need_blank = true;
            }
            Event::End(TagEnd::Strong) => self.bold = self.bold.saturating_sub(1),
            Event::End(TagEnd::Emphasis) => self.italic = self.italic.saturating_sub(1),
            Event::End(TagEnd::Strikethrough) => self.strike = self.strike.saturating_sub(1),
            Event::End(TagEnd::CodeBlock) => {
                self.in_code_block = false;
                let code = std::mem::take(&mut self.code_buf);
                let lang = std::mem::take(&mut self.code_lang);
                let indent = self.indent.clone();
                for line in highlight_code(&code, &lang, &indent) {
                    self.lines.push(line);
                }
                self.need_blank = true;
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                self.flush();
                let n = self.indent.len().saturating_sub("│ ".len());
                self.indent.truncate(n);
                self.need_blank = true;
            }
            Event::End(TagEnd::List(_)) => {
                self.list.pop();
                if !self.list.is_empty() {
                    let n = self.indent.len().saturating_sub(2);
                    self.indent.truncate(n);
                }
                self.need_blank = true;
            }
            Event::End(TagEnd::Item) => self.flush(),
            Event::Text(t) => self.text(&t),
            Event::Code(t) => {
                self.code = true;
                self.text(&t);
                self.code = false;
            }
            Event::HardBreak => self.flush(),
            Event::TaskListMarker(checked) => {
                let mark = if checked { "☑ " } else { "☐ " };
                let base = self
                    .pending
                    .take()
                    .map(|p| p.strip_suffix("• ").map(str::to_string).unwrap_or(p))
                    .unwrap_or_else(|| self.indent.clone());
                self.pending = Some(format!("{base}{mark}"));
            }
            Event::Rule => {
                self.blank();
                self.lines.push(
                    Line::from("─".repeat(self.width.min(24)))
                        .style(Style::new().fg(Color::DarkGray)),
                );
                self.need_blank = true;
            }
            _ => {}
        }
    }
}

fn markdown_lines(src: &str, width: usize) -> Vec<Line<'static>> {
    use pulldown_cmark::{Options, Parser};
    let opts = Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS | Options::ENABLE_STRIKETHROUGH;
    let mut md = Md::new(width);
    for ev in Parser::new_ext(src, opts) {
        md.event(ev);
    }
    md.flush();
    md.lines
}

/// Headless: run one turn, print the streamed events, exit. Used for testing.
async fn run_once(agent: &'static Agent, session_id: &str, prompt: String) -> Result<()> {
    let mut stream = agent
        .reply(
            Message::user().with_text(prompt),
            session_config(session_id),
            None,
        )
        .await?;
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Message(m) => {
                for c in &m.content {
                    match c {
                        MessageContent::Text(t) => {
                            let s = t.text.trim();
                            if !s.is_empty() {
                                println!("{s}");
                            }
                        }
                        MessageContent::Thinking(th) => {
                            println!("💭 {}", one_line(&th.thinking, 200))
                        }
                        MessageContent::ToolRequest(req) => println!("{}", fmt_call(req)),
                        MessageContent::ToolResponse(resp) => {
                            println!("  ↳ {}", fmt_result(resp))
                        }
                        _ => {}
                    }
                }
            }
            AgentEvent::Usage(u) => {
                eprintln!(
                    "[usage: {} output tokens]",
                    u.usage.output_tokens.unwrap_or(0)
                )
            }
            _ => {}
        }
    }
    Ok(())
}

/// A single-line prompt buffer with emacs/readline editing and command history.
/// `cursor` is a character index in `[0, buf.chars().count()]`.
#[derive(Default)]
struct Editor {
    buf: String,
    cursor: usize,
    history: Vec<String>,
    browsing: Option<usize>,
    draft: String,
}

impl Editor {
    fn len(&self) -> usize {
        self.buf.chars().count()
    }

    fn byte_at(&self, char_idx: usize) -> usize {
        self.buf
            .char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(self.buf.len())
    }

    fn char_before(&self, char_idx: usize) -> char {
        self.buf
            .chars()
            .nth(char_idx.saturating_sub(1))
            .unwrap_or(' ')
    }

    fn insert(&mut self, ch: char) {
        let at = self.byte_at(self.cursor);
        self.buf.insert(at, ch);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            let at = self.byte_at(self.cursor - 1);
            self.buf.remove(at);
            self.cursor -= 1;
        }
    }

    fn delete_forward(&mut self) {
        if self.cursor < self.len() {
            let at = self.byte_at(self.cursor);
            self.buf.remove(at);
        }
    }

    fn kill_to_end(&mut self) {
        let at = self.byte_at(self.cursor);
        self.buf.truncate(at);
    }

    fn kill_line(&mut self) {
        self.buf.clear();
        self.cursor = 0;
    }

    fn kill_word(&mut self) {
        let mut c = self.cursor;
        while c > 0 && self.char_before(c).is_whitespace() {
            c -= 1;
        }
        while c > 0 && !self.char_before(c).is_whitespace() {
            c -= 1;
        }
        let (start, end) = (self.byte_at(c), self.byte_at(self.cursor));
        self.buf.replace_range(start..end, "");
        self.cursor = c;
    }

    fn home(&mut self) {
        self.cursor = 0;
    }
    fn end(&mut self) {
        self.cursor = self.len();
    }
    fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    fn right(&mut self) {
        if self.cursor < self.len() {
            self.cursor += 1;
        }
    }

    fn load(&mut self, i: usize) {
        self.buf = self.history[i].clone();
        self.cursor = self.len();
    }

    fn prev_history(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.browsing {
            None => {
                self.draft = std::mem::take(&mut self.buf);
                let i = self.history.len() - 1;
                self.browsing = Some(i);
                self.load(i);
            }
            Some(i) if i > 0 => {
                self.browsing = Some(i - 1);
                self.load(i - 1);
            }
            Some(_) => {}
        }
    }

    fn next_history(&mut self) {
        match self.browsing {
            Some(i) if i + 1 < self.history.len() => {
                self.browsing = Some(i + 1);
                self.load(i + 1);
            }
            Some(_) => {
                self.browsing = None;
                self.buf = std::mem::take(&mut self.draft);
                self.cursor = self.len();
            }
            None => {}
        }
    }

    fn submit(&mut self) -> String {
        let line = std::mem::take(&mut self.buf);
        self.cursor = 0;
        self.browsing = None;
        self.draft.clear();
        let trimmed = line.trim();
        if !trimmed.is_empty() && self.history.last().map(String::as_str) != Some(line.as_str()) {
            self.history.push(line.clone());
        }
        line
    }
}

/// Enable the kitty keyboard protocol's escape-code disambiguation, so Ctrl+H
/// and Ctrl+M arrive as distinct keys rather than Backspace/Enter. Returns
/// whether it took effect (false on terminals that don't support it).
fn enable_key_disambiguation() -> bool {
    use crossterm::event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
    if crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false) {
        // DISAMBIGUATE distinguishes ^H/^M from Backspace/Enter; the other two
        // deliver modifier-key press/release events so the UI can tell when Ctrl
        // is held (for the contextual header hints).
        crossterm::execute!(
            std::io::stdout(),
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
            )
        )
        .is_ok()
    } else {
        false
    }
}

fn disable_key_disambiguation() {
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PopKeyboardEnhancementFlags
    );
}

/// Interactive ratatui chat loop.
async fn run_tui(
    agent: &'static Agent,
    session_id: String,
    model: String,
    provider: Arc<ScratchyProvider>,
) -> Result<()> {
    use crossterm::event::{
        Event, EventStream, KeyCode, KeyEventKind, KeyModifiers, ModifierKeyCode,
    };
    use ratatui::prelude::*;
    use ratatui::widgets::{Block, Borders, Clear, Gauge, Paragraph};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    // Cancel a turn after this many consecutive identical tool calls (a small
    // model stuck echoing the same command).
    const LOOP_LIMIT: usize = 3;
    // Forward-pass "tape player": width of the side strip, and how many kernels
    // the read-head advances per 120ms tick (a watchable replay of a real tape).
    const TAPE_WIDTH: u16 = 22;
    const TAPE_SCROLL_PER_TICK: usize = 3;
    // Height of the sampler "soul" pane (border + candidates + entropy line).
    const SOUL_HEIGHT: u16 = 10;

    enum Msg {
        Event(AgentEvent),
        Err(String),
        Done,
    }

    // What the model is doing right now, for the status spinner.
    #[derive(PartialEq)]
    enum Phase {
        Prefill,
        Generating,
        ToolRunning,
    }

    let mut terminal = ratatui::init();
    // Ctrl+H and Ctrl+M are Backspace/Enter at the byte level; the kitty
    // keyboard protocol disambiguates them so those toggles can fire. No-op on
    // terminals that don't support it (they keep the legacy mapping).
    let kb_enhanced = enable_key_disambiguation();
    let mut ed = Editor {
        history: load_history(),
        ..Editor::default()
    };
    let mut convo: Vec<Entry> = Vec::new();
    let mut thinking = String::new();
    let mut generating = false;
    let mut aborted = false;
    let mut cancel: Option<CancellationToken> = None;
    let mut last_call: Option<String> = None;
    let mut repeats = 0usize;
    let mut last_result: Option<String> = None;
    let mut result_repeats = 0usize;
    let mut scroll_off = 0u16;
    let mut show_full_tools = false;
    let mut quit_armed = false;
    // User toggles for chrome (Ctrl+M metrics, Ctrl+H help, Ctrl+T thinking).
    let mut metrics_on = true;
    let mut help_on = true;
    let mut thinking_on = true;
    // Streaming state: the in-progress model entry being appended to, the phase,
    // the running tool's name, and an animation counter for the spinner.
    let mut stream_idx: Option<usize> = None;
    let mut phase = Phase::Prefill;
    let mut current_tool: Option<String> = None;
    let mut spin = 0usize;
    // Tier-1 "vitals": live context size, session output total, and a rough
    // tokens/sec for the last turn (engine-authoritative rate lands in Tier 2).
    let mut ctx_tokens = 0u64;
    let mut session_tokens = 0u64;
    let mut turn_start: Option<std::time::Instant> = None;
    let mut turn_tokens = 0u64;
    let mut last_tok_s = 0.0f64;
    // Tool activity this session: invocation count and total characters of tool
    // output that fed back into the model's context.
    let mut tool_calls = 0u64;
    let mut tool_bytes = 0u64;
    // Engine-authoritative snapshot, sampled each tick.
    let mut engine_stats: Option<LiveStats> = None;
    let mut max_ctx = 0u64;
    // In-flight model-download progress (first-run cold download); drives the
    // centered progress modal. `None` whenever nothing is downloading.
    let mut download: Option<provider::DownloadSnapshot> = None;
    // Last model-build/download failure (e.g. out of disk); drives the error
    // modal until the user dismisses it with a keypress.
    let mut dl_error: Option<String> = None;
    // Tier-3a "tape player": the live forward-pass kernel-dispatch strip. On by
    // default; the read-head scrolls only while a forward is actually running.
    let mut tape_open = true;
    let mut tape: Option<Arc<ForwardRecord>> = None;
    let mut tape_seq = 0u64;
    let mut tape_head = 0usize;
    // Sampler "soul" view (opt-in: enabling it makes the sampler spill top-k +
    // compute entropy per token). Off by default so the sampler path is
    // untouched until asked. `soul_cands` is the decoded top-k for the latest
    // token; `entropy_hist` is a rolling window (bits) for the sparkline.
    let mut soul_open = true;
    let mut soul: Option<Arc<SamplerRecord>> = None;
    let mut soul_seq = 0u64;
    let mut soul_cands: Vec<(String, f32, bool)> = Vec::new();
    let mut entropy_hist: Vec<f32> = Vec::new();
    // Session-wide inter-token latencies (ms) for the dashboard's ITL histogram,
    // accumulated across turns. `itl_absorbed` tracks how many of the current
    // turn's samples we've already folded in (reset per turn).
    let mut itl_samples: Vec<f32> = Vec::new();
    let mut itl_absorbed = 0usize;
    SamplerTelemetry::global().set_enabled(soul_open);
    // The kernel-distribution column reads the same tape as the strip, so the
    // forward tape records whenever EITHER the strip or the dashboard is open.
    ForwardTelemetry::global().set_enabled(tape_open || soul_open);
    // Whether Ctrl is currently held (kitty protocol) — reveals per-panel toggle
    // shortcuts in the panel headers while down.
    let mut ctrl_held = false;

    let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(120));

    // Start building the engine (and downloading weights) now, in the
    // background, rather than waiting for the first prompt — the download modal
    // shows immediately and the model is often ready by the time the user hits
    // enter. `stream`'s `ensure_engine` reuses whatever this builds.
    provider.prewarm(model.clone());

    // Bottom-anchored (auto-tailing), no-wrap paragraph for a pane. Long lines
    // are clipped at the right edge so scroll math stays exact.
    fn pane<'a>(title: &'a str, body: &[String], height: u16) -> Paragraph<'a> {
        let visible = height.saturating_sub(2);
        let scroll = (body.len() as u16).saturating_sub(visible);
        Paragraph::new(body.join("\n"))
            .block(Block::bordered().title(title))
            .scroll((scroll, 0))
    }

    let res = loop {
        if let Err(e) = terminal.draw(|f| {
            // Panes only claim space once they have something to show: the
            // Thinking pane when there's thinking, the vitals row and the tape
            // strip once the engine has produced data this session.
            let show_thinking = thinking_on && !thinking.trim().is_empty();
            // Like the tape/metrics, the soul pane only claims space once a token
            // has actually been sampled this session.
            let show_soul = soul_open && soul.is_some();
            let show_stats = metrics_on && (session_tokens > 0 || engine_stats.is_some());
            let show_tape = tape_open && tape.is_some();
            // Keep the hint row when armed to quit, even with help toggled off, so
            // the "press ^C again" safety prompt is never hidden.
            let show_hint = help_on || quit_armed;
            let mut constraints = vec![Constraint::Min(3)];
            if show_thinking {
                constraints.push(Constraint::Length(8));
            }
            if show_soul {
                constraints.push(Constraint::Length(SOUL_HEIGHT));
            }
            constraints.push(Constraint::Length(3)); // input box
            if show_stats {
                constraints.push(Constraint::Length(1)); // ⚙ stats line
            }
            if show_hint {
                constraints.push(Constraint::Length(1)); // hint
            }
            // Reserve a full-height column on the right for the tape player.
            let (body, tape_area) = if show_tape {
                let cols =
                    Layout::horizontal([Constraint::Min(20), Constraint::Length(TAPE_WIDTH)])
                        .split(f.area());
                (cols[0], Some(cols[1]))
            } else {
                (f.area(), None)
            };
            let rows = Layout::vertical(constraints).split(body);
            let convo_area = rows[0];
            let mut ri = 1;
            let thinking_area = show_thinking.then(|| {
                let a = rows[ri];
                ri += 1;
                a
            });
            let soul_area = show_soul.then(|| {
                let a = rows[ri];
                ri += 1;
                a
            });
            let input_area = rows[ri];
            ri += 1;
            let stats_area = show_stats.then(|| {
                let a = rows[ri];
                ri += 1;
                a
            });
            let hint_area = show_hint.then(|| rows[ri]);

            let inner_w = convo_area.width as usize;
            let text_w = inner_w.saturating_sub(2);
            let user_bg = Style::new().bg(Color::Indexed(235)).fg(Color::Gray);
            let bullet = |lines: &mut Vec<Line>, marker: Span<'static>, s: &str| {
                for (j, wl) in wrap_text(s, text_w).iter().enumerate() {
                    if j == 0 {
                        lines.push(Line::from(vec![marker.clone(), Span::raw(wl.clone())]));
                    } else {
                        lines.push(Line::from(format!("  {wl}")));
                    }
                }
            };

            let dim = Style::new().fg(Color::DarkGray);
            let mut lines: Vec<Line> = Vec::new();
            for (i, e) in convo.iter().enumerate() {
                // Blank line before each block; a tool result stays attached to
                // the call above it.
                if i > 0 && !matches!(e, Entry::ToolResult(_)) {
                    lines.push(Line::default());
                }
                match e {
                    Entry::User(s) => {
                        for (j, wl) in wrap_text(s, text_w).iter().enumerate() {
                            let marker = if j == 0 { "❯ " } else { "  " };
                            let raw = format!("{marker}{wl}");
                            let pad = inner_w.saturating_sub(raw.chars().count());
                            lines.push(
                                Line::from(format!("{raw}{}", " ".repeat(pad))).style(user_bg),
                            );
                        }
                    }
                    Entry::Model(s) => {
                        for (j, mut ln) in markdown_lines(s, text_w.saturating_sub(2))
                            .into_iter()
                            .enumerate()
                        {
                            let mut spans = std::mem::take(&mut ln.spans);
                            let marker = if j == 0 { "● " } else { "  " };
                            let mut ns = Vec::with_capacity(spans.len() + 1);
                            ns.push(Span::raw(marker));
                            ns.append(&mut spans);
                            lines.push(Line::from(ns).style(ln.style));
                        }
                    }
                    Entry::Tool { name, args } => {
                        let mut spans = vec![Span::styled("● ", Style::new().fg(Color::Green))];
                        // For `shell` the command line is self-explanatory, so
                        // drop the redundant tool name and show just the command.
                        let show_name = name != "shell";
                        if show_name {
                            spans.push(Span::raw(name.clone()));
                        }
                        if !args.is_empty() {
                            let used = if show_name { name.chars().count() + 4 } else { 2 };
                            if show_name {
                                spans.push(Span::raw("  "));
                            }
                            spans.push(Span::styled(
                                one_line(args, text_w.saturating_sub(used)),
                                Style::new().fg(Color::Indexed(245)),
                            ));
                        }
                        lines.push(Line::from(spans));
                    }
                    Entry::TodoPlan(content) => {
                        let amber = Color::Indexed(220);
                        lines.push(Line::from(vec![
                            Span::styled("◇ ", Style::new().fg(amber)),
                            Span::styled("plan", Style::new().fg(amber).add_modifier(Modifier::BOLD)),
                        ]));
                        for ln in markdown_lines(content, text_w.saturating_sub(2)) {
                            let mut spans = vec![Span::raw("  ")];
                            spans.extend(ln.spans);
                            lines.push(Line::from(spans).style(ln.style));
                        }
                    }
                    Entry::ToolResult(full) => {
                        let body: Vec<&str> =
                            full.lines().filter(|l| !l.trim().is_empty()).collect();
                        if show_full_tools {
                            let mut first = true;
                            for src in full.lines() {
                                for wl in wrap_text(src, text_w.saturating_sub(4)) {
                                    let prefix = if first { "  ↳ " } else { "    " };
                                    lines.push(Line::from(format!("{prefix}{wl}")).style(dim));
                                    first = false;
                                }
                            }
                            if first {
                                lines.push(Line::from("  ↳ (no output)").style(dim));
                            }
                        } else {
                            let head = body.first().copied().unwrap_or("(no output)");
                            let mut spans = vec![Span::styled(
                                format!("  ↳ {}", one_line(head, text_w.saturating_sub(24))),
                                dim,
                            )];
                            let extra = body.len().saturating_sub(1);
                            if extra > 0 {
                                spans.push(Span::styled(
                                    format!("  (+{extra} lines · ^O)"),
                                    Style::new().fg(Color::Indexed(240)),
                                ));
                            }
                            lines.push(Line::from(spans));
                        }
                    }
                    Entry::Error(s) => {
                        bullet(&mut lines, Span::styled("● ", Style::new().fg(Color::Red)), s)
                    }
                    Entry::Cancelled => {
                        lines.push(Line::from(Span::styled("⎋ cancelled", dim)));
                    }
                }
            }
            let convo_visible = convo_area.height as usize;
            let max_scroll = lines.len().saturating_sub(convo_visible) as u16;
            scroll_off = scroll_off.min(max_scroll);
            f.render_widget(
                Paragraph::new(lines).scroll((max_scroll - scroll_off, 0)),
                convo_area,
            );

            if let Some(area) = thinking_area {
                // Pre-wrap to the pane width so long lines wrap instead of
                // clipping (and the pane's tail-scroll math stays exact).
                let think_lines = wrap_text(&thinking, (area.width as usize).saturating_sub(2));
                f.render_widget(pane(" 🧠 Thinking ", &think_lines, area.height), area);
            }

            // Live dashboard: ITL distribution │ kernel distribution │ sampler.
            if let Some(area) = soul_area {
                let border = Style::new().fg(Color::Indexed(238));
                let dim = Style::new().fg(Color::DarkGray);
                let chip = Style::new().fg(Color::Black).bg(Color::Indexed(214));
                let cols = Layout::horizontal([
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                    Constraint::Ratio(1, 3),
                ])
                .split(area);

                // Col 1 — inter-token latency distribution.
                let itl_title: Line = if ctrl_held {
                    Line::from(vec![Span::raw(" ITL ms "), Span::styled(" ^S ", chip)])
                } else {
                    Line::from(" ITL ms ")
                };
                let b1 = Block::bordered().border_style(border).title(itl_title);
                let i1 = b1.inner(cols[0]);
                f.render_widget(b1, cols[0]);
                f.render_widget(
                    Paragraph::new(itl_hist_lines(
                        &itl_samples,
                        generating,
                        i1.height as usize,
                        i1.width as usize,
                    )),
                    i1,
                );

                // Col 2 — kernel distribution (by dispatch count this forward).
                let b2 = Block::bordered().border_style(border).title(" kernels ");
                let i2 = b2.inner(cols[1]);
                f.render_widget(b2, cols[1]);
                let klines = tape
                    .as_ref()
                    .map(|t| kernel_dist_lines(&t.tape, i2.height as usize, i2.width as usize))
                    .unwrap_or_default();
                f.render_widget(Paragraph::new(klines), i2);

                // Col 3 — sampler soul (top-k candidates + entropy).
                let bottom = match &soul {
                    Some(r) if !r.greedy && !r.max_prob.is_nan() => format!(
                        " H {:.2}b · {:.0}% ",
                        r.entropy_nats * std::f32::consts::LOG2_E,
                        r.max_prob * 100.0
                    ),
                    _ => String::new(),
                };
                let mut b3 = Block::bordered().border_style(border).title(" ◇ sampler ");
                if !bottom.is_empty() {
                    b3 = b3.title_bottom(bottom);
                }
                let i3 = b3.inner(cols[2]);
                f.render_widget(b3, cols[2]);
                let slines = match &soul {
                    Some(r) if r.greedy || r.max_prob.is_nan() => {
                        vec![Line::from(Span::styled("  greedy · no distribution", dim))]
                    }
                    Some(_) if !soul_cands.is_empty() => sampler_lines(
                        &soul_cands,
                        &entropy_hist,
                        i3.height as usize,
                        i3.width as usize,
                    ),
                    _ => vec![Line::from(Span::styled("  …", dim))],
                };
                f.render_widget(Paragraph::new(slines), i3);
            }

            let input_block = Block::new()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(Style::new().fg(Color::Indexed(240)))
                .title(Line::from(format!(" {model} ")).right_aligned());
            let inner = input_block.inner(input_area);
            f.render_widget(input_block, input_area);

            let prompt_line = if generating {
                const FRAMES: [&str; 10] =
                    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let frame = FRAMES[spin % FRAMES.len()];
                let (label, color) = match phase {
                    Phase::Prefill => ("prefilling…".to_string(), Color::Indexed(214)),
                    Phase::Generating => ("generating…".to_string(), Color::Indexed(75)),
                    Phase::ToolRunning => (
                        format!("running {}…", current_tool.as_deref().unwrap_or("tool")),
                        Color::Green,
                    ),
                };
                Line::from(vec![
                    Span::styled(format!("{frame} "), Style::new().fg(color)),
                    Span::styled(label, Style::new().fg(color)),
                ])
            } else {
                Line::from(vec![Span::raw("❯ "), Span::raw(ed.buf.clone())])
            };
            f.render_widget(Paragraph::new(prompt_line), inner);
            if !generating {
                let caret_x = inner.x + 2 + ed.cursor as u16;
                let max_x = inner.x + inner.width.saturating_sub(1);
                f.set_cursor_position((caret_x.min(max_x), inner.y));
            }

            // ⚙ vitals line — engine-authoritative when a request is live,
            // otherwise the last turn's figures.
            let dim = Style::new().fg(Color::DarkGray);
            let val = Style::new().fg(Color::Gray);
            let sep = Style::new().fg(Color::Indexed(238));
            let cyan = Style::new().fg(Color::Indexed(75));

            let ctx = engine_stats
                .map(|s| s.prompt_tokens as u64)
                .unwrap_or(ctx_tokens);
            // Real decode rate while generating; fall back to the turn estimate.
            let tok_s = engine_stats
                .filter(|s| s.decode_tok_s > 0.0)
                .map(|s| s.decode_tok_s)
                .unwrap_or(last_tok_s);
            let rate = if tok_s > 0.0 {
                format!("{tok_s:.1} tok/s")
            } else {
                "—".to_string()
            };

            let mut sp = vec![
                Span::styled("  ⚙ ", cyan),
                Span::styled("context ", dim),
                Span::styled(commas(ctx), val),
            ];
            if max_ctx > 0 {
                sp.push(Span::styled(format!("/{}", commas(max_ctx)), dim));
                sp.push(Span::raw(" "));
                sp.extend(bar_spans(ctx as f64 / max_ctx as f64, 8));
            }
            sp.push(Span::styled("  ·  ", sep));
            sp.push(Span::styled(rate, cyan));
            if let Some(t) = engine_stats.and_then(|s| s.ttft_s) {
                sp.push(Span::styled("  ·  ", sep));
                sp.push(Span::styled("ttft ", dim));
                sp.push(Span::styled(format!("{:.0}ms", t * 1000.0), val));
            }
            if let Some(c) = engine_stats.map(|s| s.cached_tokens).filter(|c| *c > 0) {
                sp.push(Span::styled("  ·  ", sep));
                sp.push(Span::styled("prefix ", dim));
                sp.push(Span::styled(commas(c as u64), Style::new().fg(Color::Green)));
                sp.push(Span::styled(" cached", dim));
            }
            if tool_calls > 0 {
                sp.push(Span::styled("  ·  ", sep));
                sp.push(Span::styled("tools ", dim));
                sp.push(Span::styled(commas(tool_calls), val));
                sp.push(Span::styled("  ·  ", sep));
                sp.push(Span::styled(kmg(tool_bytes), val));
                sp.push(Span::styled(" from tools", dim));
            }
            if let Some(stats_area) = stats_area {
                f.render_widget(Paragraph::new(Line::from(sp)), stats_area);
            }

            if let Some(hint_area) = hint_area {
                let state = if generating { "working…" } else { "ready" };
                let tools_state = if show_full_tools { "full" } else { "compact" };
                let tape_state = if tape_open { "on" } else { "off" };
                let stats_st = if soul_open { "on" } else { "off" };
                let metrics_state = if metrics_on { "on" } else { "off" };
                let think_state = if thinking_on { "on" } else { "off" };
                let (hint, hint_style) = if quit_armed {
                    (
                        "  press ^C again to quit".to_string(),
                        Style::new().fg(Color::Indexed(209)),
                    )
                } else {
                    (
                        format!("  ⏵⏵ auto · {state} · ↑↓ scroll · ^O tools:{tools_state} · ^T think:{think_state} · ^G tape:{tape_state} · ^S stats:{stats_st} · ^M metrics:{metrics_state} · ^H help"),
                        Style::new().fg(Color::DarkGray),
                    )
                };
                f.render_widget(Paragraph::new(hint).style(hint_style), hint_area);
            }

            // Tier-3a tape player: a full-height ribbon of the real forward-pass
            // kernel dispatch sequence, colored by family, scrolling while live.
            // Only reserved once a forward has published a tape, so `tape` is Some.
            if let (Some(area), Some(t)) = (tape_area, &tape) {
                let barriers = t.tape.iter().filter(|e| e.barrier).count();
                let fused = t.tape.iter().filter(|e| e.fused).count();
                let disp_title = format!(" ⚙ {} disp ", commas(t.tape.len() as u64));
                let title: Line = if ctrl_held {
                    Line::from(vec![
                        Span::raw(disp_title),
                        Span::styled(
                            " ^G ",
                            Style::new().fg(Color::Black).bg(Color::Indexed(214)),
                        ),
                    ])
                } else {
                    Line::from(disp_title)
                };
                let block = Block::bordered()
                    .border_style(sep)
                    .title(title)
                    .title_bottom(format!(" {barriers}▸ · {fused}+ "));
                let inner = block.inner(area);
                f.render_widget(block, area);
                if !t.tape.is_empty() && inner.height > 0 {
                    let lines =
                        tape_lines(&t.tape, tape_head, inner.height as usize, inner.width as usize);
                    f.render_widget(Paragraph::new(lines), inner);
                }
            }

            // First-run model download: a centered modal with one gauge per
            // shard downloading in parallel (up to 8 concurrent), drawn last so
            // it sits on top of the chat/panels. Cleared automatically once the
            // download finishes (`download` goes back to `None`).
            if let Some(dl) = download.as_ref() {
                // Show only the shards still downloading (up to 8 in parallel);
                // a shard's bar is dropped the moment it completes. Already
                // sorted by name in the snapshot, so the bars stay put.
                const MAX_BARS: usize = 8;
                let ordered: Vec<&provider::ShardStat> =
                    dl.shards.iter().filter(|s| !s.is_done()).collect();
                let shown = ordered.len().min(MAX_BARS);
                let truncated = ordered.len().saturating_sub(shown);

                let pct_all = if dl.total_bytes > 0 {
                    (dl.done_bytes as f64 / dl.total_bytes as f64 * 100.0).round() as u16
                } else {
                    0
                };
                let header = format!(
                    "{} / {}   ·   {}%",
                    fmt_bytes(dl.done_bytes),
                    fmt_bytes(dl.total_bytes),
                    pct_all,
                );

                // Body rows: model id, aggregate line, N shard bars, optional
                // "+N more" footer. Height = body + 2 for the border.
                let body = 2 + shown + usize::from(truncated > 0);
                let area = centered_rect(58, body as u16 + 2, f.area());
                f.render_widget(Clear, area);
                let block = Block::bordered()
                    .border_style(Style::new().fg(Color::Cyan))
                    .title(" ⬇ Downloading model ");
                let inner = block.inner(area);
                f.render_widget(block, area);
                if inner.height >= 2 {
                    let mut constraints = vec![Constraint::Length(1), Constraint::Length(1)];
                    constraints.extend(std::iter::repeat_n(Constraint::Length(1), shown));
                    if truncated > 0 {
                        constraints.push(Constraint::Length(1));
                    }
                    let rows = Layout::vertical(constraints).split(inner);
                    f.render_widget(
                        Paragraph::new(Line::from(Span::styled(
                            one_line(&model, inner.width as usize),
                            Style::new().fg(Color::Gray),
                        ))),
                        rows[0],
                    );
                    f.render_widget(
                        Paragraph::new(Line::from(Span::styled(
                            header,
                            Style::new().fg(Color::DarkGray),
                        ))),
                        rows[1],
                    );
                    for (i, s) in ordered.iter().take(shown).enumerate() {
                        let r = if s.total_bytes > 0 {
                            (s.done_bytes as f64 / s.total_bytes as f64).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        // Label in its own left column so the bar itself stays
                        // clean (no text overlaid on the fill).
                        let cols = Layout::horizontal([
                            Constraint::Length(12),
                            Constraint::Min(4),
                        ])
                        .split(rows[2 + i]);
                        let label =
                            format!("{:>5} {:>3}%", shard_label(&s.name), (r * 100.0).round() as u16);
                        f.render_widget(
                            Paragraph::new(Line::from(Span::styled(
                                label,
                                Style::new().fg(Color::Gray),
                            ))),
                            cols[0],
                        );
                        f.render_widget(
                            Gauge::default()
                                .gauge_style(Style::new().fg(Color::Cyan))
                                .ratio(r)
                                .label(Span::raw("")),
                            cols[1],
                        );
                    }
                    if truncated > 0 {
                        f.render_widget(
                            Paragraph::new(Line::from(Span::styled(
                                format!("  … +{truncated} more"),
                                Style::new().fg(Color::DarkGray),
                            ))),
                            rows[2 + shown],
                        );
                    }
                }
            }

            // Model build/download failure (e.g. out of disk): a centered red
            // modal with the full error, dismissed by any key. Drawn last so it
            // sits on top of everything, including a stale progress modal.
            if let Some(err) = dl_error.as_ref() {
                let w = f.area().width.saturating_sub(8).clamp(24, 70);
                let wrapped = wrap_text(err, w.saturating_sub(4) as usize);
                let body = 1 + wrapped.len().min(8) + 2; // title line + msg + spacer + hint
                let area = centered_rect(w, body as u16 + 2, f.area());
                f.render_widget(Clear, area);
                let block = Block::bordered()
                    .border_style(Style::new().fg(Color::Red))
                    .title(" ✖ Model load failed ");
                let inner = block.inner(area);
                f.render_widget(block, area);
                if inner.height >= 2 {
                    let mut lines: Vec<Line> = wrapped
                        .iter()
                        .take(8)
                        .map(|l| Line::from(Span::styled(l.clone(), Style::new().fg(Color::White))))
                        .collect();
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "press any key to dismiss",
                        Style::new().fg(Color::DarkGray),
                    )));
                    f.render_widget(Paragraph::new(lines), inner);
                }
            }
        }) {
            break Err(e.into());
        }

        tokio::select! {
            _ = ticker.tick() => {
                if generating {
                    spin = spin.wrapping_add(1);
                }
                engine_stats = provider.live_stats();
                download = provider.download_progress();
                // Don't overwrite a locally-dismissed error until it's cleared
                // on the provider side too.
                if dl_error.is_none() {
                    dl_error = provider.download_error();
                }
                if max_ctx == 0 {
                    max_ctx = provider.max_context().unwrap_or(0) as u64;
                }
                // The strip and the dashboard's kernel column both read the tape.
                if tape_open || soul_open {
                    let ft = ForwardTelemetry::global();
                    let s = ft.seq();
                    if s != tape_seq {
                        tape_seq = s;
                        tape = ft.latest();
                    }
                    if generating && let Some(t) = &tape {
                        tape_head = (tape_head + TAPE_SCROLL_PER_TICK) % t.tape.len().max(1);
                    }
                }
                if soul_open {
                    // Fold this turn's new ITL samples into the session history
                    // (monotonic within a turn; `itl_absorbed` resets on submit).
                    let cur = provider.recent_itls();
                    if cur.len() > itl_absorbed {
                        itl_samples.extend_from_slice(&cur[itl_absorbed..]);
                        itl_absorbed = cur.len();
                        if itl_samples.len() > 8192 {
                            let drop = itl_samples.len() - 8192;
                            itl_samples.drain(0..drop);
                        }
                    }
                    let st = SamplerTelemetry::global();
                    let s = st.seq();
                    if s != soul_seq && let Some(rec) = st.latest() {
                        soul_seq = s;
                        soul_cands = rec
                            .top_k
                            .iter()
                            .map(|c| {
                                let disp = match provider.decode_token(c.token_id) {
                                    Some(t) if !t.is_empty() && !t.contains('\u{FFFD}') => {
                                        display_token(&t)
                                    }
                                    _ => format!("‹{}›", c.token_id),
                                };
                                (disp, c.prob, c.token_id == rec.sampled_token_id)
                            })
                            .collect();
                        if !rec.greedy && !rec.entropy_nats.is_nan() {
                            entropy_hist.push(rec.entropy_nats * std::f32::consts::LOG2_E);
                            if entropy_hist.len() > 32 {
                                entropy_hist.remove(0);
                            }
                        }
                        soul = Some(rec);
                    }
                }
            }
            maybe_key = keys.next() => {
                match maybe_key {
                    Some(Ok(Event::Key(k))) => {
                        use KeyCode::*;
                        // Track Ctrl-held (kitty protocol) for the contextual
                        // header hints; swallow modifier-only + key-release events
                        // so nothing else double-fires on them.
                        if let KeyCode::Modifier(m) = k.code {
                            if matches!(
                                m,
                                ModifierKeyCode::LeftControl | ModifierKeyCode::RightControl
                            ) {
                                ctrl_held = k.kind != KeyEventKind::Release;
                            }
                            continue;
                        }
                        if k.kind == KeyEventKind::Release {
                            continue;
                        }
                        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                        // Both overlays are truly modal: while either is up the
                        // text entry (and everything else) is disabled. Only
                        // quit (Ctrl+C) and suspend (Ctrl+Z) ever pass through.
                        let passthrough = ctrl && matches!(k.code, Char('c') | Char('z'));
                        // Error modal: any other key dismisses it (clearing the
                        // provider too, so the tick doesn't re-surface it).
                        if dl_error.is_some() && !passthrough {
                            dl_error = None;
                            provider.clear_download_error();
                            continue;
                        }
                        // Download modal: any other key is simply swallowed.
                        if download.is_some() && !passthrough {
                            continue;
                        }
                        if ctrl && k.code == Char('c') {
                            if generating && !aborted {
                                // Cancel the running turn (first Ctrl-C).
                                if let Some(tok) = &cancel {
                                    tok.cancel();
                                }
                                aborted = true;
                                quit_armed = true;
                            } else if !ed.buf.is_empty() {
                                // Idle with typed text: clear the line, don't quit.
                                ed.kill_line();
                                quit_armed = false;
                            } else if quit_armed {
                                break Ok(());
                            } else {
                                quit_armed = true;
                            }
                            continue;
                        }
                        // Any other key disarms a pending double-Ctrl-C quit.
                        quit_armed = false;
                        if ctrl && k.code == Char('z') {
                            if kb_enhanced {
                                disable_key_disambiguation();
                            }
                            ratatui::restore();
                            #[cfg(unix)]
                            unsafe {
                                libc::raise(libc::SIGTSTP);
                            }
                            terminal = ratatui::init();
                            if kb_enhanced {
                                enable_key_disambiguation();
                            }
                            continue;
                        }
                        if k.code == Esc {
                            if generating && !aborted {
                                // goose's token stops its loop between steps, but
                                // only aborting the engine request stops the
                                // in-flight generation — so do both.
                                if let Some(tok) = &cancel {
                                    tok.cancel();
                                }
                                aborted = true;
                                let p = provider.clone();
                                tokio::spawn(async move { p.abort().await });
                                convo.push(Entry::Cancelled);
                            }
                            continue;
                        }
                        if ctrl && k.code == Char('o') {
                            show_full_tools = !show_full_tools;
                            continue;
                        }
                        if ctrl && k.code == Char('g') {
                            tape_open = !tape_open;
                            ForwardTelemetry::global().set_enabled(tape_open || soul_open);
                            tape_head = 0;
                            if !tape_open && !soul_open {
                                tape = None;
                            }
                            continue;
                        }
                        if ctrl && k.code == Char('s') {
                            soul_open = !soul_open;
                            SamplerTelemetry::global().set_enabled(soul_open);
                            ForwardTelemetry::global().set_enabled(tape_open || soul_open);
                            if !soul_open {
                                soul = None;
                                soul_cands.clear();
                                entropy_hist.clear();
                                // Keep itl_samples: it's the session-wide history.
                                soul_seq = 0;
                                if !tape_open {
                                    tape = None;
                                }
                            }
                            continue;
                        }
                        if ctrl && k.code == Char('m') {
                            metrics_on = !metrics_on;
                            continue;
                        }
                        if ctrl && k.code == Char('t') {
                            thinking_on = !thinking_on;
                            continue;
                        }
                        if ctrl && k.code == Char('h') {
                            help_on = !help_on;
                            continue;
                        }
                        match k.code {
                            Up => { scroll_off = scroll_off.saturating_add(1); continue; }
                            Down => { scroll_off = scroll_off.saturating_sub(1); continue; }
                            PageUp => { scroll_off = scroll_off.saturating_add(10); continue; }
                            PageDown => { scroll_off = scroll_off.saturating_sub(10); continue; }
                            _ => {}
                        }
                        if !generating {
                            match (ctrl, k.code) {
                                (true, Char('a')) | (false, Home) => ed.home(),
                                (true, Char('e')) | (false, End) => ed.end(),
                                (true, Char('b')) | (false, Left) => ed.left(),
                                (true, Char('f')) | (false, Right) => ed.right(),
                                (true, Char('p')) => ed.prev_history(),
                                (true, Char('n')) => ed.next_history(),
                                (true, Char('d')) => ed.delete_forward(),
                                (true, Char('k')) => ed.kill_to_end(),
                                (true, Char('u')) => ed.kill_line(),
                                (true, Char('w')) => ed.kill_word(),
                                (false, Backspace) => ed.backspace(),
                                (false, Char(ch)) => ed.insert(ch),
                                (false, Enter) => {
                                    let hist_len = ed.history.len();
                                    let prompt = ed.submit();
                                    if !prompt.trim().is_empty() {
                                        if ed.history.len() > hist_len {
                                            append_history(&prompt);
                                        }
                                        // `/clear` clears goose's session history; mirror it in
                                        // the TUI's own display buffers (and the thinking pane).
                                        if prompt.trim() == "/clear" {
                                            convo.clear();
                                            thinking.clear();
                                            session_tokens = 0;
                                            ctx_tokens = 0;
                                            last_tok_s = 0.0;
                                            tool_calls = 0;
                                            tool_bytes = 0;
                                            itl_samples.clear();
                                        } else {
                                            convo.push(Entry::User(prompt.clone()));
                                        }
                                        scroll_off = 0;
                                        generating = true;
                                        aborted = false;
                                        stream_idx = None;
                                        phase = Phase::Prefill;
                                        current_tool = None;
                                        // New turn: restart the per-turn ITL cursor.
                                        itl_absorbed = 0;
                                        turn_start = Some(std::time::Instant::now());
                                        turn_tokens = 0;
                                        last_call = None;
                                        repeats = 0;
                                        last_result = None;
                                        result_repeats = 0;
                                        let token = CancellationToken::new();
                                        cancel = Some(token.clone());
                                        let cfg = session_config(&session_id);
                                        let tx = tx.clone();
                                        tokio::spawn(async move {
                                            match agent.reply(Message::user().with_text(prompt), cfg, Some(token)).await {
                                                Ok(mut s) => {
                                                    while let Some(ev) = s.next().await {
                                                        match ev {
                                                            Ok(e) => { let _ = tx.send(Msg::Event(e)); }
                                                            Err(e) => { let _ = tx.send(Msg::Err(e.to_string())); }
                                                        }
                                                    }
                                                    let _ = tx.send(Msg::Done);
                                                }
                                                Err(e) => {
                                                    let _ = tx.send(Msg::Err(e.to_string()));
                                                    let _ = tx.send(Msg::Done);
                                                }
                                            }
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Err(e)) => break Err(e.into()),
                    _ => {}
                }
            }
            Some(msg) = rx.recv() => {
                match msg {
                    Msg::Done => {
                        generating = false;
                        aborted = false;
                        cancel = None;
                        if let Some(t) = turn_start.take() {
                            let secs = t.elapsed().as_secs_f64();
                            if secs > 0.05 && turn_tokens > 0 {
                                last_tok_s = turn_tokens as f64 / secs;
                            }
                        }
                    }
                    // Once a turn is aborted (loop guard or Esc), drop the
                    // in-flight stragglers until it actually ends.
                    _ if aborted => {}
                    Msg::Event(AgentEvent::Usage(u)) => {
                        if let Some(inp) = u.usage.input_tokens {
                            ctx_tokens = inp.max(0) as u64;
                        }
                        let out = u.usage.output_tokens.unwrap_or(0).max(0) as u64;
                        session_tokens += out;
                        turn_tokens += out;
                    }
                    Msg::Event(AgentEvent::Message(m)) => {
                        // A stuck small model repeats the same call (or varies the
                        // args but keeps getting the same result). Catch both.
                        for c in &m.content {
                            match c {
                                MessageContent::ToolRequest(req) => {
                                    let call = fmt_call(req);
                                    if last_call.as_ref() == Some(&call) {
                                        repeats += 1;
                                    } else {
                                        repeats = 1;
                                        last_call = Some(call);
                                    }
                                }
                                MessageContent::ToolResponse(resp) => {
                                    let res = fmt_result(resp);
                                    if last_result.as_ref() == Some(&res) {
                                        result_repeats += 1;
                                    } else {
                                        result_repeats = 1;
                                        last_result = Some(res);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if (repeats >= LOOP_LIMIT || result_repeats >= LOOP_LIMIT)
                            && let Some(tok) = &cancel
                            && !tok.is_cancelled()
                        {
                            tok.cancel();
                            aborted = true;
                            convo.push(Entry::Error(
                                "stopped — model stuck in a tool-call loop".to_string(),
                            ));
                        }
                        // Accumulate streamed deltas: text tokens append to the
                        // current model block, thinking to the pane; tool calls
                        // and results are whole.
                        for c in &m.content {
                            match c {
                                MessageContent::Text(t) if !t.text.is_empty() => {
                                    match stream_idx {
                                        Some(i)
                                            if matches!(convo.get(i), Some(Entry::Model(_))) =>
                                        {
                                            if let Some(Entry::Model(s)) = convo.get_mut(i) {
                                                s.push_str(&t.text);
                                            }
                                        }
                                        _ => {
                                            convo.push(Entry::Model(t.text.clone()));
                                            stream_idx = Some(convo.len() - 1);
                                        }
                                    }
                                    phase = Phase::Generating;
                                }
                                MessageContent::Thinking(th) => {
                                    thinking.push_str(&th.thinking);
                                    phase = Phase::Generating;
                                }
                                MessageContent::ToolRequest(req) => {
                                    let (name, args) = fmt_call_parts(req);
                                    current_tool = Some(name.clone());
                                    match todo_plan_content(req) {
                                        Some(plan) => convo.push(Entry::TodoPlan(plan)),
                                        None => convo.push(Entry::Tool { name, args }),
                                    }
                                    tool_calls += 1;
                                    stream_idx = None;
                                    phase = Phase::ToolRunning;
                                }
                                MessageContent::ToolResponse(resp) => {
                                    let full = result_full(resp);
                                    tool_bytes += full.chars().count() as u64;
                                    convo.push(Entry::ToolResult(full));
                                    stream_idx = None;
                                    current_tool = None;
                                    phase = Phase::Prefill;
                                }
                                _ => {}
                            }
                        }
                    }
                    Msg::Event(_) => {}
                    Msg::Err(e) => convo.push(Entry::Error(e)),
                }
            }
        }
    };

    if kb_enhanced {
        disable_key_disambiguation();
    }
    ratatui::restore();
    res
}

/// Load `model` in-process and either run one headless turn (`once`) or the
/// interactive TUI. Driven on the caller's tokio runtime (needs multi-thread).
pub async fn run(model: String, once: Option<String>) -> Result<()> {
    eprintln!("[scratchy-tui] loading {model} (in-process) …");
    let (agent, session_id, provider) = setup(&model).await?;
    // The agent lives for the whole program; leak it so reply() streams are
    // 'static and can be driven on spawned tasks in the TUI loop.
    let agent: &'static Agent = Box::leak(Box::new(agent));

    match once {
        Some(prompt) => run_once(agent, &session_id, prompt).await,
        None => {
            let res = run_tui(agent, session_id, model, provider).await;
            // A first-run model build/download runs on a `spawn_blocking` OS
            // thread that can't be cancelled. Returning normally would make the
            // `#[tokio::main]` runtime's shutdown block on that thread until the
            // download finished — the "have to hit Ctrl-C a third time" hang.
            // The terminal is already restored (run_tui does that on the way
            // out) and history is persisted incrementally, so there is nothing
            // left to flush: exit the process directly.
            match res {
                Ok(()) => std::process::exit(0),
                Err(e) => {
                    eprintln!("[scratchy-tui] {e:#}");
                    std::process::exit(1);
                }
            }
        }
    }
}
