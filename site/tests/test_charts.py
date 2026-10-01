"""Geometry tests. Every coordinate the page draws is computed in Python, so all
of this is checkable with no browser and no dependencies."""

import math
import unittest
import xml.etree.ElementTree as ET

import fixture  # noqa: F401 - puts site/ on sys.path

import charts


class NiceTicks(unittest.TestCase):
    def test_steps_are_one_two_or_five_per_decade(self):
        for lo, hi in ((0, 10), (0, 1), (0, 3), (0, 7), (0, 900), (0, 0.04),
                       (12, 4831), (-50, 50)):
            ticks = charts.nice_ticks(lo, hi)
            step = ticks[1] - ticks[0]
            mantissa = step / 10 ** math.floor(math.log10(step))
            self.assertAlmostEqual(
                min(abs(mantissa - m) for m in (1, 2, 5, 10)), 0, places=6,
                msg=f"step {step} for [{lo},{hi}] is not 1/2/5 per decade")

    def test_ticks_are_evenly_spaced(self):
        for lo, hi in ((0, 97), (-40, 12), (0.004, 0.9)):
            ticks = charts.nice_ticks(lo, hi)
            gaps = [round(b - a, 9) for a, b in zip(ticks, ticks[1:])]
            self.assertEqual(len(set(gaps)), 1, f"uneven ticks {ticks}")

    def test_bounds_span_the_data(self):
        for lo, hi in ((3, 97), (0.004, 0.9), (-7, 12)):
            t = charts.nice_ticks(lo, hi)
            self.assertLessEqual(t[0], lo)
            self.assertGreaterEqual(t[-1], hi)

    def test_zero_is_a_tick_when_the_range_straddles_it(self):
        # diverging_bars reads its zero rule off this list, so an exact 0.0 must
        # be present and comparable by ==.
        for lo, hi in ((-40, 12), (-1, 1), (-0.5, 3), (-100, 5), (-3, 0)):
            t = charts.nice_ticks(lo, hi)
            self.assertIn(0.0, t, f"no exact zero in {t} for [{lo},{hi}]")

    def test_no_negative_zero(self):
        for t in charts.nice_ticks(-40, 12):
            if t == 0:
                self.assertEqual(repr(t), "0.0")

    def test_flat_range(self):
        self.assertEqual(charts.nice_ticks(0, 0), [0.0, 1.0])
        t = charts.nice_ticks(5, 5)
        self.assertGreater(len(t), 1)
        self.assertLessEqual(t[0], 5)
        self.assertGreaterEqual(t[-1], 5)

    def test_all_negative_range(self):
        t = charts.nice_ticks(-90, -10)
        self.assertLessEqual(t[0], -90)
        self.assertGreaterEqual(t[-1], -10)

    def test_reversed_bounds_are_tolerated(self):
        self.assertEqual(charts.nice_ticks(100, 0), charts.nice_ticks(0, 100))


class Scale(unittest.TestCase):
    def test_exact_at_both_endpoints(self):
        sc = charts.scale(0, 100, 10, 210)
        self.assertAlmostEqual(charts.project(sc, 0), 10)
        self.assertAlmostEqual(charts.project(sc, 100), 210)
        self.assertAlmostEqual(charts.project(sc, 50), 110)

    def test_flat_domain_does_not_divide_by_zero(self):
        sc = charts.scale(7, 7, 0, 100)
        self.assertAlmostEqual(charts.project(sc, 7), 50)

    def test_flat_zero_domain(self):
        sc = charts.scale(0, 0, 0, 100)
        self.assertAlmostEqual(charts.project(sc, 0), 50)

    def test_inverted_pixel_range(self):
        # y axes run downward: px0 > px1 is the normal case, not an error.
        sc = charts.scale(0, 10, 200, 0)
        self.assertAlmostEqual(charts.project(sc, 0), 200)
        self.assertAlmostEqual(charts.project(sc, 10), 0)


class Escaping(unittest.TestCase):
    HOSTILE = 'gemma<&>"4'

    def test_svg_escapes_its_title_and_desc(self):
        out = charts.svg(10, 10, "", self.HOSTILE, self.HOSTILE)
        self.assertNotIn("<&>", out)
        ET.fromstring(out)  # parses, i.e. the escaping is well-formed

    def test_end_label_escapes_its_text(self):
        sc = charts.scale(0, 1, 0, 10)
        out = charts.end_label(sc, sc, (0, 0), self.HOSTILE, "lb-s1")
        self.assertNotIn("<&>", out)
        ET.fromstring(out)

    def test_table_view_escapes_cells(self):
        out = charts.table_view(["h"], [[self.HOSTILE]], "c")
        self.assertNotIn("<&>", out)


class Sparkline(unittest.TestCase):
    def test_too_few_points_renders_nothing(self):
        self.assertEqual(charts.sparkline([]), "")
        self.assertEqual(charts.sparkline([5]), "")
        self.assertEqual(charts.sparkline([None, None]), "")

    def test_two_points_render(self):
        out = charts.sparkline([1, 2])
        self.assertIn("<polyline", out)
        ET.fromstring(out)

    def test_example_marks_itself(self):
        out = charts.sparkline([1, 2], example=True)
        self.assertIn("lb-example", out)
        self.assertIn(charts.EXAMPLE_DESC, out)


class DotRange(unittest.TestCase):
    def test_zero_width_spread(self):
        out = charts.dot_range(
            [{"label": "a", "median": 1.0, "p10": 1.0, "p90": 1.0}], "t")
        self.assertIn("<circle", out)
        ET.fromstring(out)

    def test_missing_spread(self):
        out = charts.dot_range([{"label": "a", "median": 1.0}], "t")
        self.assertIn("<circle", out)
        self.assertNotIn("lb-whisker", out)

    def test_empty_rows_render_nothing(self):
        self.assertEqual(charts.dot_range([], "t"), "")


class DivergingBars(unittest.TestCase):
    def rows(self, *vals):
        return [{"label": f"r{i}", "value": v, "badge": "optimization"}
                for i, v in enumerate(vals)]

    def test_zero_rule_is_at_the_right_edge_when_all_values_are_negative(self):
        out = charts.diverging_bars(self.rows(-10, -20, -30), "t")
        root = ET.fromstring(out)
        zero = [e for e in root.iter() if e.get("class") == "lb-zero"][0]
        xs = [float(e.get("x1")) for e in root.iter()
              if e.get("class") == "lb-grid"]
        # Every value is below zero, so the zero rule is the rightmost gridline.
        self.assertAlmostEqual(float(zero.get("x1")), max(xs), places=3)

    def test_zero_rule_is_interior_when_signs_are_mixed(self):
        out = charts.diverging_bars(self.rows(-10, 20), "t")
        root = ET.fromstring(out)
        zero = float([e for e in root.iter()
                      if e.get("class") == "lb-zero"][0].get("x1"))
        xs = [float(e.get("x1")) for e in root.iter()
              if e.get("class") == "lb-grid"]
        self.assertGreater(zero, min(xs))
        self.assertLess(zero, max(xs))

    def test_pole_follows_the_sign(self):
        out = charts.diverging_bars(self.rows(-10, 20), "t")
        self.assertIn("lb-pole-fast", out)
        self.assertIn("lb-pole-slow", out)

    def test_value_is_always_written_out(self):
        # The bar is the comparison; the number is the fact, and it must not
        # depend on a tooltip.
        out = charts.diverging_bars(self.rows(-10), "t")
        self.assertIn("-10%", out)

    def test_example_gets_a_hatch_and_a_watermark(self):
        out = charts.diverging_bars(self.rows(-10), "t", example=True)
        self.assertIn("lb-bar-hatch", out)
        self.assertIn("EXAMPLE DATA", out)
        self.assertIn('fill="url(#hxd)"', out)
        ET.fromstring(out)

    def test_empty_rows_render_nothing(self):
        self.assertEqual(charts.diverging_bars([], "t"), "")


class LineFigure(unittest.TestCase):
    SERIES = [
        {"label": "scratchy", "cls": "lb-s1", "points": [(1, 500), (2, 100), (3, 100)]},
        {"label": "ollama", "cls": "lb-s2", "points": [(1, 520), (2, 300), (3, 300)]},
    ]

    def test_payload_pixels_match_the_drawn_marks(self):
        markup, payload = charts.line_figure(self.SERIES, "t")
        root = ET.fromstring(markup)
        drawn = {
            (round(float(c.get("cx")), 1), round(float(c.get("cy")), 1))
            for c in root.iter() if c.tag == "circle"
        }
        # The hover layer does no scale math, so every pixel it will use must be
        # a pixel Python actually drew.
        for s in payload["series"]:
            for p in s["points"]:
                self.assertIn((p["x"], p["y"]), drawn)

    def test_y_axis_includes_zero(self):
        _markup, _payload = charts.line_figure(self.SERIES, "t")
        ticks = charts.nice_ticks(0.0, 520, want=5)
        self.assertEqual(ticks[0], 0.0)

    def test_no_series_renders_nothing(self):
        markup, payload = charts.line_figure([], "t")
        self.assertEqual(markup, "")
        self.assertEqual(payload, {})

    def test_example_dashes_and_watermarks(self):
        markup, _ = charts.line_figure(self.SERIES, "t", example=True)
        self.assertIn("lb-example", markup)
        self.assertIn("EXAMPLE DATA", markup)
        self.assertIn(charts.EXAMPLE_DESC, markup)


class Fmt(unittest.TestCase):
    def test_trims_trailing_zeros(self):
        self.assertEqual(charts.fmt(100.0, "ms", 1), "100 ms")
        self.assertEqual(charts.fmt(0.05, "", 3), "0.05")

    def test_none_is_a_dash(self):
        self.assertEqual(charts.fmt(None), "—")

    def test_zero_stays_zero(self):
        self.assertEqual(charts.fmt(0.0, "", 3), "0")


class Legend(unittest.TestCase):
    def test_every_entry_carries_its_series_class(self):
        out = charts.legend([("scratchy", "lb-s1"), ("ollama", "lb-s2")])
        self.assertIn("lb-s1", out)
        self.assertIn("lb-s2", out)
        self.assertEqual(out.count("<li"), 2)


if __name__ == "__main__":
    unittest.main()
