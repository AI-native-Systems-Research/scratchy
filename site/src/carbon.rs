//! Typed wrappers over the Carbon Web Components the site uses
//! (https://web-components.carbondesignsystem.com). The components themselves
//! stay IBM's: each wrapper renders the same `cds-*` tag, but its attributes
//! are Rust types, so a misspelt attribute or an unknown `kind` fails to compile
//! rather than rendering a silently default component.

use dioxus::prelude::*;

/// One module of IBM's Carbon Web Components CDN, one per component family.
/// A page lists the families it renders; the head loads exactly those.
#[derive(Clone, Copy, PartialEq)]
pub enum Module {
    Accordion,
    Button,
    CodeSnippet,
    Select,
    Tile,
    ToggleTip,
    Tooltip,
    UiShell,
}

impl Module {
    const CDN: &str = "https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest";

    pub fn src(self) -> String {
        let file = match self {
            Module::Accordion => "accordion",
            Module::Button => "button",
            Module::CodeSnippet => "code-snippet",
            Module::Select => "select",
            Module::Tile => "tile",
            Module::ToggleTip => "toggle-tip",
            Module::Tooltip => "tooltip",
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
    #[props(into)] class: String,
    #[props(into)] id: Option<String>,
    #[props(into)] style: Option<String>,
    children: Element,
) -> Element {
    rsx! { cds-tile { "id": id, "class": class, "style": style, {children} } }
}

#[component]
pub fn ClickableTile(
    #[props(into)] class: String,
    #[props(into)] href: String,
    children: Element,
) -> Element {
    rsx! { cds-clickable-tile { "class": class, "href": href, {children} } }
}

/// A multi-line `cds-code-snippet`. Its text is rendered verbatim, so the
/// children carry their own newlines.
#[component]
pub fn CodeSnippet(children: Element) -> Element {
    rsx! { cds-code-snippet { "type": "multi", {children} } }
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
        cds-toggletip { "class": "minfo", "alignment": alignment, "button-label": button_label,
            "{label}"
            span { "slot": "body-text", class: "minfo-body", {children} }
        }
    }
}

/// A Carbon tooltip, opened by hovering (and, if `focusable`, focusing) the
/// trigger. cds-tooltip takes its first child as the trigger and hides the
/// content from assistive tech, so the trigger names it with aria-describedby:
/// `id` must be unique on the page.
#[component]
pub fn Tooltip(
    #[props(into)] id: String,
    #[props(into)] class: String,
    #[props(default)] focusable: bool,
    trigger: Element,
    children: Element,
) -> Element {
    rsx! {
        cds-tooltip { "align": "top",
            span { class, "tabindex": focusable.then_some("0"), "aria-describedby": id.clone(), {trigger} }
            cds-tooltip-content { "id": id, {children} }
        }
    }
}

/// A collapsed Carbon accordion item, for detail most readers skip.
#[component]
pub fn Fold(#[props(into)] title: String, children: Element) -> Element {
    rsx! {
        cds-accordion { "class": "mfold",
            cds-accordion-item { "title": title, {children} }
        }
    }
}

#[component]
pub fn SideNav(
    #[props(into)] label: String,
    #[props(into)] id: Option<String>,
    children: Element,
) -> Element {
    rsx! {
        cds-side-nav { "aria-label": label, "class": "docs-side-nav", "id": id,
            cds-side-nav-items { {children} }
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
