//! Typed wrappers over the Carbon Web Components the site uses
//! (https://web-components.carbondesignsystem.com). The components themselves
//! stay IBM's: each wrapper renders the same `cds-*` tag, but its attributes
//! are Rust types, so a misspelt attribute or an unknown `kind` fails to compile
//! rather than rendering a silently default component.

use dioxus::prelude::*;

/// One module of IBM's Carbon Web Components CDN, one per component family.
/// A page's head loads exactly the modules for the tags its body renders
/// (`Module::used_by`), so a component can never render undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Module {
    Accordion,
    Button,
    CodeSnippet,
    DataTable,
    Grid,
    Heading,
    Layer,
    Link,
    List,
    Select,
    Stack,
    Tag,
    Tile,
    ToggleTip,
    UiShell,
}

/// Which module defines each `cds-*` tag family: the tag itself and every
/// `<family>-...` tag.
const FAMILIES: [(&str, Module); 21] = [
    ("cds-accordion", Module::Accordion),
    ("cds-button", Module::Button),
    ("cds-clickable-tile", Module::Tile),
    ("cds-code-snippet", Module::CodeSnippet),
    ("cds-column", Module::Grid),
    ("cds-grid", Module::Grid),
    ("cds-header", Module::UiShell),
    ("cds-heading", Module::Heading),
    ("cds-layer", Module::Layer),
    ("cds-link", Module::Link),
    ("cds-list-item", Module::List),
    ("cds-ordered-list", Module::List),
    ("cds-section", Module::Heading),
    ("cds-select", Module::Select),
    ("cds-side-nav", Module::UiShell),
    ("cds-stack", Module::Stack),
    ("cds-table", Module::DataTable),
    ("cds-tag", Module::Tag),
    ("cds-tile", Module::Tile),
    ("cds-toggletip", Module::ToggleTip),
    ("cds-unordered-list", Module::List),
];

impl Module {
    const CDN: &str = "https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest";

    fn of_tag(tag: &str) -> Option<Module> {
        FAMILIES
            .iter()
            .find(|(family, _)| {
                tag.strip_prefix(family)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
            })
            .map(|&(_, m)| m)
    }

    /// The modules that define every `cds-*` tag in `html`, or the first tag
    /// no module is known to define.
    pub fn used_by(html: &str) -> Result<Vec<Module>, String> {
        let mut used = Vec::new();
        for (at, _) in html.match_indices("<cds-") {
            let tag: String = html[at + 1..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            let m = Module::of_tag(&tag)
                .ok_or_else(|| format!("<{tag}>: no Carbon module known to define it"))?;
            if !used.contains(&m) {
                used.push(m);
            }
        }
        used.sort();
        Ok(used)
    }

    pub fn src(self) -> String {
        let file = match self {
            Module::Accordion => "accordion",
            Module::Button => "button",
            Module::CodeSnippet => "code-snippet",
            Module::DataTable => "data-table",
            Module::Grid => "grid",
            Module::Heading => "heading",
            Module::Layer => "layer",
            Module::Link => "link",
            Module::List => "list",
            Module::Select => "select",
            Module::Stack => "stack",
            Module::Tag => "tag",
            Module::Tile => "tile",
            Module::ToggleTip => "toggle-tip",
            Module::UiShell => "ui-shell",
        };
        format!("{}/{file}.min.js", Self::CDN)
    }
}

/// `cds-button`'s `kind`, the ones the site uses.
#[derive(Clone, Copy, PartialEq)]
pub enum ButtonKind {
    Primary,
    Tertiary,
}

#[component]
pub fn Button(kind: ButtonKind, #[props(into)] href: String, children: Element) -> Element {
    let kind = match kind {
        ButtonKind::Primary => "primary",
        ButtonKind::Tertiary => "tertiary",
    };
    rsx! { cds-button { "kind": kind, "href": href, {children} } }
}

#[component]
pub fn Tile(
    #[props(into)] class: Option<String>,
    #[props(into)] id: Option<String>,
    #[props(into)] style: Option<String>,
    children: Element,
) -> Element {
    rsx! { cds-tile { "id": id, "class": class, "style": style, {children} } }
}

#[component]
pub fn ClickableTile(
    #[props(into)] class: Option<String>,
    #[props(into)] href: String,
    children: Element,
) -> Element {
    rsx! { cds-clickable-tile { "class": class, "href": href, {children} } }
}

/// A multi-line `cds-code-snippet`. Its text is rendered verbatim, so the
/// children carry their own newlines.
#[component]
pub fn CodeSnippet(#[props(into)] id: Option<String>, children: Element) -> Element {
    rsx! { cds-code-snippet { "id": id, "type": "multi", {children} } }
}

/// Where a toggletip's popover opens, relative to its button.
#[derive(Clone, Copy, PartialEq)]
pub enum Alignment {
    Bottom,
}

/// A Carbon toggletip: an (i) button opening a short note, so explanations
/// sit where the question comes up instead of pushing the numbers down.
/// Carbon fixes its width at 18rem, so keep the body to a few sentences. The
/// body is a <span>, not a <p>: toggletips sit inside paragraphs, and a <p>
/// in a <p> makes the parser close the outer one and spill the text out.
/// No `autoalign`: Carbon then calls scrollIntoView on the button every time
/// it renders, page load included, so a page of toggletips jumps to the last
/// one (Firefox lands at the bottom). A fixed alignment never scrolls.
#[component]
pub fn Toggletip(
    #[props(into, default)] label: String,
    alignment: Alignment,
    children: Element,
) -> Element {
    let alignment = match alignment {
        Alignment::Bottom => "bottom",
    };
    let button_label = if label.is_empty() {
        "What this means".to_string()
    } else {
        label.clone()
    };
    rsx! {
        cds-toggletip { "alignment": alignment, "button-label": button_label,
            "{label}"
            span { "slot": "body-text", {children} }
        }
    }
}

/// Which way a `cds-stack` lays out its children.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum Orientation {
    #[default]
    Vertical,
    Horizontal,
}

/// A `cds-stack`: its children spaced by one step of Carbon's spacing scale
/// (1 to 13), in a column unless `orientation` says a row.
#[component]
pub fn Stack(gap: u8, #[props(default)] orientation: Orientation, children: Element) -> Element {
    let orientation = match orientation {
        Orientation::Vertical => "vertical",
        Orientation::Horizontal => "horizontal",
    };
    rsx! { cds-stack { "gap": "{gap}", "orientation": orientation, {children} } }
}

/// A `cds-grid`: Carbon's 16-column responsive grid (4 columns small, 8
/// medium). Full width: by default Carbon caps a grid at 99rem and centres
/// it, which in a wide content area leaves it floating in from the left.
#[component]
pub fn Grid(#[props(default)] gutter: Gutter, children: Element) -> Element {
    let (condensed, row_gap) = match gutter {
        Gutter::Condensed => (Some(""), None),
        Gutter::Cards => (None, Some("")),
    };
    rsx! { cds-grid { "condensed": condensed, "with-row-gap": row_gap, "full-width": "", {children} } }
}

/// The space between a grid's columns (and rows), as Carbon sets it.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum Gutter {
    /// 1px columns, no row gap: one block, e.g. inside a tile.
    #[default]
    Condensed,
    /// Carbon's standard 2rem gutter, and the same between rows: separate
    /// cards.
    Cards,
}

/// A page's content: one Carbon grid, at its default 99rem max width with
/// its own page margins, holding a single full-width column. Grids inside it
/// are Carbon subgrids, so their columns line up with the page's.
#[component]
pub fn Page(children: Element) -> Element {
    rsx! {
        cds-grid {
            cds-column { "sm": "4", "md": "8", "lg": "16", {children} }
        }
    }
}

/// A `cds-link`: Carbon's link, for a call to action standing on its own.
#[component]
pub fn Link(#[props(into)] href: String, children: Element) -> Element {
    rsx! { cds-link { "href": href, {children} } }
}

/// How many grid columns a `cds-column` spans at each breakpoint.
#[derive(Clone, Copy, PartialEq)]
pub struct Span {
    pub sm: u8,
    pub md: u8,
    pub lg: u8,
}

impl Span {
    /// Full width on small screens, half on medium, a third on large.
    pub const THIRD: Span = Span {
        sm: 4,
        md: 4,
        lg: 5,
    };
    /// Full width on small screens, half on medium, a quarter on large.
    pub const QUARTER: Span = Span {
        sm: 4,
        md: 4,
        lg: 4,
    };
    /// A text column beside a wider one (with WIDE), stacked below large.
    pub const NARROW: Span = Span {
        sm: 4,
        md: 8,
        lg: 7,
    };
    pub const WIDE: Span = Span {
        sm: 4,
        md: 8,
        lg: 9,
    };
    /// Full width on small screens, half from medium up.
    pub const HALF: Span = Span {
        sm: 4,
        md: 4,
        lg: 8,
    };
}

#[component]
pub fn Column(span: Span, children: Element) -> Element {
    rsx! { cds-column { "sm": "{span.sm}", "md": "{span.md}", "lg": "{span.lg}", {children} } }
}

/// A `cds-section`: a level of the page's outline, always given. Left out,
/// a section asks its parent section in the browser.
#[component]
pub fn Section(#[props(into)] id: Option<String>, level: u8, children: Element) -> Element {
    rsx! { cds-section { "id": id, "level": "{level}", {children} } }
}

/// A `cds-heading`: h(level) of its section (h1 outside any), in Carbon's
/// matching type style. Carbon's stylesheet defines its type tokens only
/// inside components, so this is how a heading gets Carbon type here.
#[component]
pub fn Heading(children: Element) -> Element {
    rsx! { cds-heading { {children} } }
}

/// A `cds-layer`: sets Carbon's layer tokens for what is inside it, so tiles,
/// tables and fields there take the next background step (level 0 to 2:
/// layer-01, -02, -03), alternating grey and white in the light theme. It
/// paints nothing itself (its `with-background` selector is malformed in this
/// release); the components inside do.
#[component]
pub fn Layer(level: u8, children: Element) -> Element {
    rsx! { cds-layer { "level": "{level}", {children} } }
}

/// A small `cds-table` with one header row.
#[component]
pub fn Table(headers: Vec<String>, children: Element) -> Element {
    // Scrolls sideways in its own box on a narrow screen, rather than
    // widening the page around it.
    rsx! {
      div { style: "max-width: 100%; overflow-x: auto",
        cds-table { "size": "sm",
            cds-table-head {
                cds-table-header-row {
                    for h in headers { cds-table-header-cell { "{h}" } }
                }
            }
            cds-table-body { {children} }
        }
      }
    }
}

#[component]
pub fn TableRow(children: Element) -> Element {
    rsx! { cds-table-row { {children} } }
}

#[component]
pub fn TableCell(#[props(default)] colspan: Option<u8>, children: Element) -> Element {
    rsx! { cds-table-cell { "colspan": colspan.map(|n| n.to_string()), {children} } }
}

/// `cds-tag`'s colour, the ones the site uses.
#[derive(Clone, Copy, PartialEq)]
pub enum TagKind {
    Green,
    Red,
    Gray,
}

/// A small `cds-tag`; `title` says in words what its colour means.
#[component]
pub fn Tag(kind: TagKind, #[props(into)] title: String, children: Element) -> Element {
    let kind = match kind {
        TagKind::Green => "green",
        TagKind::Red => "red",
        TagKind::Gray => "gray",
    };
    rsx! { cds-tag { "type": kind, "size": "sm", "title": title, {children} } }
}

/// A collapsed Carbon accordion item, for detail most readers skip.
#[component]
pub fn Fold(#[props(into)] title: String, children: Element) -> Element {
    rsx! {
        cds-accordion {
            cds-accordion-item { "title": title, {children} }
        }
    }
}

#[component]
pub fn SideNav(
    #[props(into)] label: String,
    #[props(into)] id: Option<String>,
    top: Element,
    #[props(default)] menu_only: bool,
    children: Element,
) -> Element {
    rsx! {
        cds-side-nav { "aria-label": label, "id": id, "is-not-persistent": menu_only.then_some(""),
            cds-side-nav-items {
                // The header's links, which Carbon hides on small screens: shown
                // here instead, above the page's own, below 66rem only.
                cds-header-side-nav-items { "has-divider": "", {top} }
                {children}
            }
        }
    }
}

#[component]
pub fn SideNavLink(
    #[props(into)] href: String,
    #[props(default)] active: bool,
    #[props(into)] data_arch: Option<String>,
    children: Element,
) -> Element {
    rsx! {
        cds-side-nav-link { "href": href, "data-arch": data_arch, "active": active.then_some(""), {children} }
    }
}

/// A side-nav group, always expanded.
#[component]
pub fn SideNavMenu(#[props(into)] title: String, children: Element) -> Element {
    rsx! { cds-side-nav-menu { "title": title, "expanded": "", {children} } }
}

#[component]
pub fn SideNavMenuItem(#[props(into)] href: String, children: Element) -> Element {
    rsx! { cds-side-nav-menu-item { "href": href, {children} } }
}

#[cfg(test)]
mod tests {
    use super::Module;

    #[test]
    fn modules_come_from_the_tags_rendered() {
        let html =
            "<cds-tile><cds-table-row></cds-table-row><cds-tag></cds-tag><cds-grid><cds-column>";
        assert_eq!(
            Module::used_by(html),
            Ok(vec![
                Module::DataTable,
                Module::Grid,
                Module::Tag,
                Module::Tile
            ])
        );
    }

    #[test]
    fn an_unknown_tag_fails_the_build() {
        assert!(Module::used_by("<cds-tooltip>").is_err());
    }
}
