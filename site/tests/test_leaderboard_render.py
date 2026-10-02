"""Render tests: structural invariants, not byte goldens.

A golden file churns on every wording change and would fight the repo's
no-net-line-growth culture. These assert the properties that must hold — no
placeholder numbers, no blank without a reason, no value reachable only by
hovering — so they survive editing the prose.
"""

import copy
import re
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import fixture

import build_leaderboard as L

HEADER = '<cds-header aria-label="scratchy"></cds-header>'
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class TagStack(HTMLParser):
    """Minimal well-formedness check: every non-void tag closes, in order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def render(board=None):
    b = fixture.indexed(board if board is not None else fixture.board())
    html, problems, counts = L.page(HEADER, b)
    assert not problems, problems
    return html, counts


def empty_board():
    """Every declared cell a reasoned gap, no measurements anywhere."""
    b = fixture.board()
    for run in b["runs"]:
        lane = run["lane"]
        run["status"] = "example"
        run["ablations"] = []
        run["gaps"] = [
            {"lane": lane, "metric": m, "engine": e, "model": "m1",
             "machine": "h1", "rung": "R1", "state": "not-run",
             "reason": "the harness has not run this cell yet", "issue": 166}
            for m in b["lanes"][lane]["owns"] if m in b["expects"]["metrics"]
            for e in b["expects"]["engines"]
        ]
        run["measurements"] = []
    return b


class WellFormed(unittest.TestCase):
    def test_every_svg_parses(self):
        html, _ = render()
        svgs = re.findall(r"<svg\b.*?</svg>", html, re.S)
        self.assertTrue(svgs, "no SVG was rendered at all")
        for s in svgs:
            ET.fromstring(s)

    def test_html_tags_balance(self):
        html, _ = render()
        p = TagStack()
        p.feed(html)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.stack, [])

    def test_committed_page_svgs_parse(self):
        b = L.load()
        html, problems, _ = L.page(HEADER, b)
        self.assertEqual(problems, [])
        for s in re.findall(r"<svg\b.*?</svg>", html, re.S):
            ET.fromstring(s)

    def test_committed_page_tags_balance(self):
        b = L.load()
        html, _problems, _ = L.page(HEADER, b)
        p = TagStack()
        p.feed(html)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.stack, [])


class EmptyBoard(unittest.TestCase):
    def test_says_why_it_is_empty(self):
        html, _ = render(empty_board())
        self.assertIn("nothing yet", html)

    def test_no_hero_and_no_stat_tiles(self):
        # With nothing measured there is no hero figure at all, rather than a
        # dash set in display type.
        html, _ = render(empty_board())
        self.assertNotIn("lbkpi", html)

    def test_no_data_figures(self):
        html, _ = render(empty_board())
        self.assertNotIn("<svg", html)
        # ...and an empty figure is a dashed note, not an empty axis frame.
        self.assertIn("lbfig-empty", html)
        self.assertNotIn("lb-grid", html)

    def test_every_declared_cell_still_has_a_row(self):
        html, counts = render(empty_board())
        self.assertEqual(counts["measured"], 0)
        self.assertEqual(counts["gaps"], counts["rows"])
        self.assertEqual(counts["rows"], 6)  # 3 metrics x 2 engines x 1 model

    def test_no_digit_appears_in_a_display_number(self):
        html, _ = render(empty_board())
        for m in re.finditer(r'class="lbkpi-value[^"]*"[^>]*>([^<]*)<', html):
            self.assertNotRegex(m.group(1), r"\d")


class GapsCarryReasons(unittest.TestCase):
    def test_every_empty_cell_has_a_non_empty_reason(self):
        html, _ = render(empty_board())
        cells = re.findall(r'<td class="[^"]*\bempty\b[^"]*"([^>]*)>', html)
        self.assertTrue(cells)
        for attrs in cells:
            m = re.search(r'title="([^"]*)"', attrs)
            self.assertIsNotNone(m, f"empty cell with no reason: {attrs}")
            self.assertTrue(m.group(1).strip())

    def test_empty_cells_are_not_sortable_as_zero(self):
        # A dash must never sort as a number; omitting data-sort keeps it last.
        html, _ = render(empty_board())
        for attrs in re.findall(r'<td class="[^"]*\bempty\b[^"]*"([^>]*)>', html):
            self.assertNotIn("data-sort", attrs)

    def test_state_word_is_visible_not_just_a_colour(self):
        html, _ = render(empty_board())
        self.assertIn("not run", html)

    def test_committed_board_footnotes_every_gap(self):
        b = L.load()
        html, _problems, _ = L.page(HEADER, b)
        self.assertIn("lb-notes", html)
        self.assertIn("No energy measurement exists in the tree", html)


class LanePurity(unittest.TestCase):
    def test_lane_a_table_holds_no_lane_b_metric(self):
        b = fixture.indexed()
        a = L.board_table(b, "A", "m1", "h1", "R1")
        self.assertIn("TTFT", a)
        self.assertNotIn("task pass", a)
        self.assertNotIn("tool-call failures", a)

    def test_lane_b_table_holds_no_lane_a_metric(self):
        b = fixture.indexed()
        bt = L.board_table(b, "B", "m1", "h1", "R1")
        self.assertIn("task pass", bt)
        self.assertNotIn("TTFT", bt)

    def test_each_lane_names_itself_in_its_caption(self):
        b = fixture.indexed()
        self.assertIn("Lane A", L.board_table(b, "A", "m1", "h1", "R1"))
        self.assertIn("Lane B", L.board_table(b, "B", "m1", "h1", "R1"))


class Winner(unittest.TestCase):
    def s(self, median, p10, p90):
        return {"stat": {"median": median, "p10": p10, "p90": p90}, "n": 5}

    def test_overlapping_spreads_are_not_a_win(self):
        a = self.s(100, 90, 120)
        b = self.s(110, 100, 130)
        self.assertIsNone(L.winner(a, b, True))

    def test_separated_spreads_pick_the_lower_when_lower_is_better(self):
        a = self.s(100, 95, 105)
        b = self.s(300, 280, 320)
        self.assertEqual(L.winner(a, b, True), "a")
        self.assertEqual(L.winner(b, a, True), "b")

    def test_higher_is_better_flips_it(self):
        a = self.s(1.0, 1.0, 1.0)
        b = self.s(0.2, 0.1, 0.3)
        self.assertEqual(L.winner(a, b, False), "a")

    def test_missing_rival_is_not_a_win(self):
        self.assertIsNone(L.winner(self.s(1, 1, 1), None, True))

    def test_a_tie_is_not_a_win(self):
        a = {"stat": {"median": 5}, "n": 1}
        self.assertIsNone(L.winner(a, dict(a), True))

    def test_no_bold_on_a_board_whose_spreads_overlap(self):
        b = fixture.board()
        for m in b["runs"][0]["measurements"]:
            m["stat"] = {"median": 100, "p10": 50, "p90": 500}
        table = L.board_table(fixture.indexed(b), "A", "m1", "h1", "R1")
        self.assertNotIn("win", table)


class CorrectnessGate(unittest.TestCase):
    def test_a_tool_call_failure_de_emphasises_that_engine(self):
        b = fixture.board()
        for m in b["runs"][1]["measurements"]:
            if m["metric"] == "tool_call_parse_failure_rate" and m["engine"] == "ollama":
                m["stat"] = {"median": 0.02, "p10": 0.0, "p90": 0.04}
        idx = fixture.indexed(b)
        self.assertEqual(L.suspect_engines(idx, "m1", "h1", "R1"), {"ollama"})
        # Rule 7: you cannot win a row with a fast wrong answer.
        self.assertIn("suspect", L.board_table(idx, "A", "m1", "h1", "R1"))

    def test_a_clean_row_has_no_suspects(self):
        self.assertEqual(L.suspect_engines(fixture.indexed(), "m1", "h1", "R1"),
                         set())


class TooltipsNeverGate(unittest.TestCase):
    def test_every_hover_value_also_appears_in_a_table_view(self):
        b = fixture.indexed()
        facets, payload = L.ttft_facets(b)
        self.assertTrue(payload)
        views = re.findall(r"<details class=\"lb-tableview\".*?</details>",
                           facets, re.S)
        blob = " ".join(views)
        self.assertTrue(blob, "a figure shipped without its table view")
        for fig in payload.values():
            for s in fig["series"]:
                for p in s["points"]:
                    self.assertIn(str(p["vy"]), blob)

    def test_every_figure_carries_a_table_view(self):
        html, _ = render()
        figs = re.findall(r"<figure\b.*?</figure>", html, re.S)
        self.assertTrue(figs)
        for fig in figs:
            if "lbfig-empty" in fig.split(">")[0]:
                continue  # nothing to tabulate yet
            self.assertIn('<details class="lb-tableview"', fig,
                          f"figure without a table view: {fig[:120]}")


class ExampleMarking(unittest.TestCase):
    """Example data must be unmistakable in a cropped screenshot, not merely
    disclaimed at the top of the page."""

    def committed(self):
        b = L.load()
        html, problems, _ = L.page(HEADER, b)
        assert not problems, problems
        return html

    def test_the_committed_data_is_example_data(self):
        self.assertTrue(L.is_example(L.load()))

    def test_page_carries_a_banner(self):
        self.assertIn("lbbanner", self.committed())

    def test_every_svg_is_marked(self):
        html = self.committed()
        for s in re.findall(r"<svg\b.*?</svg>", html, re.S):
            self.assertIn("lb-example", s.split(">")[0],
                          "an example figure rendered unmarked")

    def test_every_plot_watermarks_itself(self):
        html = self.committed()
        # Sparklines are too small for a watermark and carry the dash plus the
        # spoken <desc>; every full-size plot gets the watermark.
        plots = [s for s in re.findall(r"<svg\b.*?</svg>", html, re.S)
                 if "lb-svg" in s.split(">")[0]]
        self.assertTrue(plots)
        for s in plots:
            self.assertIn("EXAMPLE DATA", s)

    def test_the_marking_is_spoken_not_only_drawn(self):
        html = self.committed()
        for s in re.findall(r"<svg\b.*?</svg>", html, re.S):
            self.assertIn("<desc>", s)
            self.assertIn("not a measurement", s)

    def test_table_views_say_so_too(self):
        html = self.committed()
        for cap in re.findall(r"<caption>(.*?)</caption>", html, re.S):
            if "TTFT in ms per turn" in cap or "pass rate" in cap \
                    or "Ablation deltas with" in cap:
                self.assertIn("Example data", cap)


class WorksWithoutJavaScript(unittest.TestCase):
    """Nothing the page asserts may depend on a script running.

    styles.css hides an un-upgraded custom element, so any number inside one
    disappears when its module fails to load. Measured: the headline vanished
    entirely while the stat tiles were cds-tile.
    """

    def page_without_scripts(self):
        b = L.load()
        html, problems, _ = L.page(HEADER, b)
        assert not problems, problems
        return re.sub(r"<script\b.*?</script>", "", html, flags=re.S)

    def test_headline_numbers_survive(self):
        stripped = self.page_without_scripts()
        self.assertIn("lbkpi-value", stripped)
        self.assertRegex(
            re.search(r'class="lbkpi-value[^"]*"[^>]*>([^<]*)<', stripped).group(1),
            r"\d")

    def test_no_number_sits_inside_a_custom_element(self):
        html = self.page_without_scripts()
        # cds-select is the one custom element left, and it is a control whose
        # default view CSS already renders correctly; it holds no values.
        for m in re.finditer(r"<(cds-(?!select)[a-z-]+)\b.*?</\1>", html, re.S):
            self.assertNotRegex(m.group(0), r">\s*[\d.]+\s*<",
                                f"a value is hidden inside <{m.group(1)}>")

    def test_the_turn_class_default_needs_no_script(self):
        # body carries the default, and CSS decides which cell shows.
        html = self.page_without_scripts()
        self.assertIn('<body data-turnclass="all">', html)

    def test_table_views_are_details_not_script_driven(self):
        html = self.page_without_scripts()
        self.assertIn('<details class="lb-tableview"', html)


class Structure(unittest.TestCase):
    def test_exactly_one_hero(self):
        html, _ = render()
        self.assertEqual(html.count("lbkpi-value lb-hero"), 1)

    def test_every_series_class_has_a_legend_entry(self):
        html, _ = render()
        for cls in ("lb-s1", "lb-s2"):
            self.assertIn(f'class="lb-key lb-key-line {cls}"', html)

    def test_every_table_has_one_caption(self):
        html, _ = render()
        tables = html.count('<table class="cds--data-table') + \
            html.count('<table class="lb-datatable')
        self.assertEqual(tables, html.count("<caption>"))

    def test_disclosures_come_before_the_headline(self):
        # The fairness rules are the product here; a reader meets them first.
        html, _ = render()
        self.assertLess(html.index("Fairness disclosures"), html.index("Headline"))

    def test_no_inline_styles(self):
        html, _ = render()
        self.assertNotIn('style="', html)

    def test_the_cell_format_is_the_published_one(self):
        # median (p10-p90) xreps, with an en dash and a multiplication sign.
        # The terminal harness prints the same shape in ASCII at
        # crates/benches/src/startup_exec/mod.rs:734-749; both spellings are
        # pinned so neither drifts alone.
        metric = {"label": "TTFT", "unit": "ms", "digits": 1,
                  "lower_is_better": True}
        rec = {"stat": {"median": 412.6, "p10": 388.1, "p90": 501.3}, "n": 12}
        out = L.cell(rec, metric)
        self.assertIn("412.6 ms", out)
        self.assertIn("(388.1–501.3)", out)
        self.assertIn("×12", out)

    def test_a_single_rep_prints_no_spread(self):
        metric = {"label": "TTFT", "unit": "ms", "digits": 1}
        out = L.cell({"stat": {"median": 5.0}, "n": 1}, metric)
        self.assertIn("×1", out)
        self.assertNotIn("–", out)

    def test_turn_class_variants_are_rendered_for_switching(self):
        b = L.load()
        html, _problems, _ = L.page(HEADER, b)
        for tc in ("all", "tool-call", "file-write"):
            self.assertIn(f'data-facet="{tc}"', html)

    def test_hover_payload_is_valid_json_in_the_page(self):
        html, _ = render()
        m = re.search(
            r'<script type="application/json" id="lbfigures">(.*?)</script>',
            html, re.S)
        self.assertIsNotNone(m)
        import json
        json.loads(m.group(1))


class BuildRefusesBadData(unittest.TestCase):
    def test_page_returns_problems_and_no_html(self):
        b = fixture.board()
        b["runs"][0]["measurements"][0].pop("stat")
        html, problems, _ = L.page(HEADER, fixture.indexed(b))
        self.assertIsNone(html)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
