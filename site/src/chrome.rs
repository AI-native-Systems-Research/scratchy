//! The site chrome, defined once: every page's head, its header, and the
//! document around them, so the nav can never disagree with itself.

use std::fmt;

use dioxus::prelude::*;

use crate::carbon::{Module, SideNav, SideNavLink, SideNavMenu, SideNavMenuItem};

#[macro_export]
macro_rules! repo {
    ($path:literal) => {
        concat!(
            "https://github.com/AI-native-Systems-Research/scratchy",
            $path
        )
    };
}
pub const REPO: &str = repo!("");

/// How many directories below `_site/` a page sits, rendered as the relative
/// prefix back to the site root ("", "../", "../../").
#[derive(Clone, Copy, PartialEq)]
pub struct Root(pub usize);

impl fmt::Display for Root {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.0).try_for_each(|_| f.write_str("../"))
    }
}

/// The header's top-level tabs, in nav order.
#[derive(Clone, Copy, PartialEq)]
pub enum Tab {
    Models,
    Docs,
    Performance,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Models, Tab::Docs, Tab::Performance];

    fn href(self) -> &'static str {
        match self {
            Tab::Models => "architectures.html",
            Tab::Docs => "book/index.html",
            Tab::Performance => "metal.html",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Tab::Models => "Models",
            Tab::Docs => "Docs",
            Tab::Performance => "Performance",
        }
    }
}

/// Which Carbon theme class the page carries.
#[derive(Clone, Copy, PartialEq)]
pub enum Theme {
    /// Carbon's g100 or white, following the reader's system setting.
    FollowSystem,
    /// Always Carbon's light theme: for pages whose colours are chosen and
    /// checked against it.
    AlwaysLight,
}

impl Theme {
    fn script(self) -> &'static str {
        match self {
            Theme::FollowSystem => FOLLOW_SYSTEM_JS,
            Theme::AlwaysLight => ALWAYS_LIGHT_JS,
        }
    }
}

const FOLLOW_SYSTEM_JS: &str = "
(function () {
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  function apply(dark) {
    document.documentElement.classList.remove('cds--g100', 'cds--white');
    document.documentElement.classList.add(dark ? 'cds--g100' : 'cds--white');
  }
  apply(mq.matches);
  mq.addEventListener('change', function (e) { apply(e.matches); });
})();
";

const ALWAYS_LIGHT_JS: &str = "
// Always Carbon's light theme: the heat-map steps and series colours are
// chosen and checked against it, whatever the reader's system setting.
document.documentElement.classList.add('cds--white');
";

/// Carbon Design System tokens: defines the --cds-* custom properties per
/// theme class. styles.css and the cds-* components both read these.
const CARBON_STYLES: &str = "https://cdn.jsdelivr.net/npm/@carbon/styles@1/css/styles.min.css";

/// Open Graph card for a page that is shared on its own.
pub struct OpenGraph {
    pub title: &'static str,
    pub description: &'static str,
}

/// A third-party library a page loads from a CDN, pinned to its major version.
#[derive(Clone, Copy, PartialEq)]
pub enum Library {
    /// Carbon Charts (https://charts.carbondesignsystem.com): the `Charts` global.
    CarbonCharts,
    /// diff-view-element, a PrismJS-backed diff web component
    /// (https://konnorrogers.github.io/diff-view-element/).
    DiffView,
    /// zero-md, renders a Markdown file in place (https://zerodevx.github.io/zero-md/).
    ZeroMd,
}

impl Library {
    /// Its script, and whether that is an ES module.
    fn script(self) -> (&'static str, bool) {
        match self {
            Library::CarbonCharts => (
                "https://cdn.jsdelivr.net/npm/@carbon/charts@1/dist/umd/bundle.umd.js",
                false,
            ),
            Library::DiffView => (
                "https://cdn.jsdelivr.net/npm/diff-view-element@1/cdn/exports/components/diff-view-element/diff-view-element-register.js",
                true,
            ),
            Library::ZeroMd => ("https://cdn.jsdelivr.net/npm/zero-md@3?register", true),
        }
    }

    fn stylesheet(self) -> Option<&'static str> {
        match self {
            Library::CarbonCharts => {
                Some("https://cdn.jsdelivr.net/npm/@carbon/charts@1/dist/styles.css")
            }
            Library::DiffView | Library::ZeroMd => None,
        }
    }
}

pub struct Head<'a> {
    pub title: &'a str,
    pub description: Option<&'static str>,
    pub og: Option<OpenGraph>,
    pub root: Root,
    pub modules: &'a [Module],
    pub libraries: &'a [Library],
    pub theme: Theme,
}

fn head(h: &Head) -> Element {
    let root = h.root;
    rsx! {
        meta { charset: "utf-8" }
        meta { name: "viewport", content: "width=device-width, initial-scale=1" }
        title { "{h.title}" }
        if let Some(d) = h.description {
            meta { name: "description", content: d }
        }
        if let Some(og) = &h.og {
            meta { "property": "og:title", content: og.title }
            meta { "property": "og:description", content: og.description }
            meta { "property": "og:type", content: "website" }
        }
        link { rel: "icon", r#type: "image/png", href: "{root}favicon.png" }
        link { rel: "stylesheet", href: CARBON_STYLES }
        // Before styles.css, so the site's own rules win.
        for css in h.libraries.iter().filter_map(|l| l.stylesheet()) {
            link { rel: "stylesheet", href: css }
        }
        link { rel: "stylesheet", href: "{root}styles.css" }
        for m in h.modules {
            script { r#type: "module", src: m.src() }
        }
        for (src, module) in h.libraries.iter().map(|l| l.script()) {
            script { r#type: if module { "module" }, src }
        }
        script { dangerous_inner_html: h.theme.script() }
    }
}

/// A whole page: the head, then `body`. dioxus-html has no `<html>` element,
/// so the document shell around the two is fixed text.
pub fn document(h: Head, body: Element) -> String {
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n{}\n</head>\n<body>\n{}\n</body>\n</html>\n",
        dioxus_ssr::render_element(head(&h)),
        dioxus_ssr::render_element(body),
    )
}

/// The GitHub octicon "mark-github", inlined so the global action needs no
/// extra request.
const GITHUB_MARK: &str = "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258\
.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 \
17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 \
1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 \
0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 \
1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 \
2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 \
2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297\
c0-6.627-5.373-12-12-12";

/// The header every page shares. cds-header-nav-item's reflected attribute
/// is `is-active` (CDSHeaderNavItem); plain `active` belongs to other
/// components (e.g. cds-side-nav-link) and does nothing here.
/// cds-header-global-action *is* CDSButton (kind=ghost, size=lg, tooltip below
/// by default) with a panel toggle that only fires when `panel-id` is set:
/// without it, and with `href`, it renders a real <a> sized and tooltipped
/// like every other header action, from ui-shell alone.
pub fn header(root: Root, active: Option<Tab>) -> Element {
    let home = if root.0 == 0 {
        ".".to_string()
    } else {
        root.to_string()
    };
    rsx! {
        cds-header { "aria-label": "scratchy",
            cds-header-menu-button { "button-label-active": "Close menu", "button-label-inactive": "Open menu" }
            cds-header-name { "href": home, "prefix": "▚", "scratchy" }
            cds-header-nav { "menu-bar-label": "scratchy navigation",
                for tab in Tab::ALL {
                    cds-header-nav-item {
                        "href": "{root}{tab.href()}",
                        "is-active": (active == Some(tab)).then_some(""),
                        "{tab.label()}"
                    }
                }
            }
            div { class: "cds--header__global",
                cds-header-global-action { "href": REPO, "target": "_blank", "rel": "noopener",
                    "tooltip-text": "GitHub", "tooltip-alignment": "end",
                    svg { "slot": "icon", width: "20", height: "20", view_box: "0 0 24 24",
                        fill: "currentColor", "aria-hidden": "true",
                        path { d: GITHUB_MARK }
                    }
                }
            }
        }
    }
}

/// Pages under the Performance tab, in left-nav order. A new page joins the
/// left nav by adding one row here, without touching the others.
const PERFORMANCE: [(&str, &str); 1] = [("metal.html", "Metal")];

/// The Performance left nav: the page being built expands to its own in-page
/// sections, given as (anchor, label); every other page is a link.
pub fn perf_side_nav(active: &str, sections: &[(String, String)]) -> Element {
    rsx! {
        SideNav { label: "Performance",
            for (href, label) in PERFORMANCE {
                if href == active {
                    SideNavMenu { title: label,
                        for (anchor, text) in sections {
                            SideNavMenuItem { href: "#{anchor}", "{text}" }
                        }
                    }
                } else {
                    SideNavLink { href, "{label}" }
                }
            }
        }
    }
}
