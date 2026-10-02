"""Guards on the site-wide invariants a new page can quietly break.

Each of these has cost this repo something before: the top nav lives in two
unconnected places, and a Carbon custom element that is not in styles.css's
hide-until-upgraded list flashes unstyled on load.
"""

import re
import unittest
from pathlib import Path

import fixture  # noqa: F401 - puts site/ on sys.path

import build
import build_leaderboard as L

SITE = Path(__file__).resolve().parent.parent
STYLES = (SITE / "styles.css").read_text()
INDEX = (SITE / "index.html").read_text()


class NavParity(unittest.TestCase):
    """build.NAV feeds every generated page; index.html hand-writes the same
    list. Nothing connected them, so they drifted (commit 5b26f981 fixed it by
    hand). check_nav_parity() and this test are the two halves of the guard."""

    def items(self):
        return [
            (h, l.strip()) for h, l in re.findall(
                r'<cds-header-nav-item\s+href="([^"]+)"[^>]*>(.*?)</cds-header-nav-item>',
                INDEX, re.S)
        ]

    def test_every_nav_entry_is_in_index_html(self):
        for item in build.NAV:
            self.assertIn(item, self.items())

    def test_index_html_has_no_extra_entries(self):
        for item in self.items():
            self.assertIn(item, [tuple(n) for n in build.NAV])

    def test_the_leaderboard_is_in_the_nav(self):
        self.assertIn(("leaderboard.html", "Leaderboard"), [tuple(n) for n in build.NAV])

    def test_the_build_time_check_agrees(self):
        self.assertFalse(build.check_nav_parity())


class FoucGuard(unittest.TestCase):
    """Every custom element the page uses must be hidden until upgraded, or it
    renders unboxed on first paint."""

    def listed(self):
        block = STYLES.split("visibility: hidden;")[0]
        return set(re.findall(r"^([a-z]+(?:-[a-z]+)+):not\(:defined\)", block, re.M))

    def test_every_custom_element_on_the_page_is_listed(self):
        html, problems, _ = L.page("", L.load())
        self.assertEqual(problems, [])
        used = set(re.findall(r"<(cds-[a-z-]+|zero-md|diff-view-element)\b", html))
        # cds-select-item only exists inside cds-select, which is listed; it is
        # never a top-level box of its own. Same convention as build_archs.py.
        used.discard("cds-select-item")
        missing = used - self.listed()
        self.assertEqual(missing, set(), f"not in the :not(:defined) list: {missing}")

    def test_the_chart_is_not_a_custom_element(self):
        # Deliberate: the FOUC list hides until upgraded, and a chart that must
        # paint with no JavaScript has to stay out of it.
        html, _problems, _ = L.page("", L.load())
        self.assertIn("<svg", html)
        self.assertNotIn("<lb-chart", html)


class Palette(unittest.TestCase):
    """The series colours are a reviewed constant.

    The dataviz validator is a node script outside this repo, so CI pins the
    values it approved rather than re-running it; re-run it by hand when the
    palette changes.
    """

    def test_the_three_series_hexes_are_the_validated_ones(self):
        for name, hex_ in (("--lb-s1", "#8a3ffc"), ("--lb-s2", "#009d9a"),
                           ("--lb-s3", "#ba4e00")):
            self.assertRegex(STYLES, rf"{name}:\s*{hex_};")

    def test_the_validation_record_is_recorded_beside_them(self):
        # If someone changes a hex, the numbers next to it must stop being true,
        # and this is what makes that obvious in review.
        for needle in ("purple-60", "teal-50", "orange-70", "mode-INVARIANT",
                       "#33b1ff", "2.31:1"):
            self.assertIn(needle, STYLES)

    def test_no_light_dark_branching_was_introduced(self):
        # The stylesheet's own header promises this: theme comes entirely from
        # the cds--g100/cds--white class on <html>.
        self.assertNotIn("prefers-color-scheme", STYLES)

    def test_charts_never_sit_on_the_panel_surface(self):
        # --panel resolves to --cds-layer-02, where --lb-s1 measures 2.31:1.
        fig = re.search(r"\.lbfig \{(.*?)\}", STYLES, re.S)
        self.assertIsNotNone(fig)
        self.assertNotIn("var(--panel)", fig.group(1))

    def test_series_colour_is_bound_to_marks_not_to_bare_classes(self):
        # A bare `.lb-s1 { fill: ... }` would leak the series colour into label
        # text, which must wear ink. Every selector mentioning a series class has
        # to qualify it with the mark it paints.
        bare = re.compile(r"(?m)^\s*\.lb-s[123]\s*(?:,|\{)")
        self.assertIsNone(bare.search(STYLES),
                          "a bare .lb-sN selector would leak series colour into text")

    def test_that_guard_would_catch_a_bare_selector(self):
        bare = re.compile(r"(?m)^\s*\.lb-s[123]\s*(?:,|\{)")
        self.assertIsNotNone(bare.search(".lb-s1 { fill: red; }\n"))
        self.assertIsNone(bare.search(".lb-dot.lb-s1 { fill: red; }\n"))


class CarbonTokenFallbacks(unittest.TestCase):
    """Carbon exposes only its colour theme tokens as custom properties.

    Measured with getComputedStyle against the built page: --cds-background and
    --cds-support-* resolve, but --cds-spacing-* and the type tokens are
    UNDEFINED. A bare var(--cds-spacing-05) is therefore an invalid value and the
    whole declaration is dropped — which silently made this page's `gap` fall back
    to `normal`. So every spacing and type token in the leaderboard block carries
    an explicit fallback.

    (The same is true elsewhere in this stylesheet and on the other pages. That is
    pre-existing and deliberately left alone here: adding fallbacks above this
    block would change the rhythm of every page at once.)
    """

    def block(self):
        return STYLES[STYLES.index("Launch Claude Leaderboard"):]

    def test_no_bare_spacing_token(self):
        bare = re.findall(r"var\(--cds-spacing-\d\d\)", self.block())
        self.assertEqual(bare, [], f"spacing tokens with no fallback: {set(bare)}")

    def test_no_bare_type_token(self):
        bare = re.findall(r"var\(--cds-(?:heading|body|label|code)-[^,)]*\)",
                          self.block())
        self.assertEqual(bare, [], f"type tokens with no fallback: {set(bare)}")

    def test_colour_tokens_need_no_fallback(self):
        # These do resolve, so requiring a fallback would just duplicate Carbon's
        # own theme values and risk them drifting apart.
        self.assertIn("var(--cds-support-success)", self.block())


class NoHorizontalOverflow(unittest.TestCase):
    """Every wide table must scroll inside its own box.

    Measured: without this the board set the document width to 810px in a 500px
    viewport and the whole page scrolled sideways. CI has no browser, so the
    guard is on the rule that fixed it — any container holding a data table needs
    `overflow-x`.
    """

    def containers(self):
        """Classes that wrap a <table> in the generated page."""
        html, problems, _ = L.page("", L.load())
        self.assertEqual(problems, [])
        out = set()
        for m in re.finditer(r'class="([^"]*)"[^>]*>\s*(?:<summary[^>]*>.*?</summary>\s*)?<table',
                             html, re.S):
            out.update(c for c in m.group(1).split() if c.startswith("lb"))
        return out

    def test_every_table_wrapper_scrolls_itself(self):
        wrappers = self.containers()
        self.assertTrue(wrappers, "no table wrappers found at all")
        block = STYLES[STYLES.index("Launch Claude Leaderboard"):]
        scrolls = set()
        for m in re.finditer(r"((?:\.[a-zA-Z0-9-]+,?\s*)+)\{([^}]*)\}", block):
            if "overflow-x" in m.group(2):
                scrolls.update(s.strip().lstrip(".").rstrip(",")
                               for s in m.group(1).split())
        missing = wrappers - scrolls
        self.assertEqual(missing, set(),
                         f"table wrappers with no overflow-x: {missing}")


class DataFilesAreUnderSite(unittest.TestCase):
    def test_data_lives_where_pages_yml_already_watches(self):
        # pages.yml filters on 'site/**', so data here redeploys the site; data
        # outside it silently would not.
        self.assertTrue((SITE / "data" / "leaderboard" / "board.json").exists())
        self.assertTrue(list((SITE / "data" / "leaderboard" / "runs").glob("*.json")))

    def test_the_generator_is_imported_by_the_site_build(self):
        self.assertIn("import build_leaderboard", (SITE / "build.py").read_text())


if __name__ == "__main__":
    unittest.main()
