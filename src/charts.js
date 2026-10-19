// Draws every [data-chart] with Carbon Charts, from the data and options the
// site build rendered. A ratio heatmap stores log2(ratio), so a factor of two
// either way is the same distance from ×1; its tooltip shows the ratio
// itself, "×1.23", through Carbon's valueFormatter.
//
// Not until the page has loaded: this script runs while the page is still
// being parsed, but the Carbon components that lay it out (cds-stack,
// cds-grid, cds-tile) are ES modules, which run only after parsing. A chart
// drawn before then sizes itself to a layout that is not final, and can keep
// that wrong geometry (a heatmap's columns came out uneven) even after it is
// resized. `load` waits for those modules and the stylesheets; the fonts can
// still move text, so wait for them too.
//
// The page follows the reader's system theme, Carbon's white or g100 class on
// <html> (chrome.rs), and so does every chart: the build gives each colour
// for both, and when the class changes the charts redraw in the new one.
(function () {
  // Carbon Charts' name for the page's theme, and the key of its colours.
  function theme() {
    return document.documentElement.classList.contains('cds--g100')
      ? { carbon: 'g100', key: 'dark' }
      : { carbon: 'white', key: 'light' };
  }
  // The spec's options in the theme: its colours, each {light, dark}, as
  // the one Carbon draws there.
  function themed(options, t) {
    var out = Object.assign({}, options, { theme: t.carbon });
    var color = options.color;
    if (color && color.scale) {
      var scale = {};
      Object.keys(color.scale).forEach(function (group) { scale[group] = color.scale[group][t.key]; });
      out.color = { scale: scale };
    } else if (color && color.gradient) {
      out.color = { gradient: { colors: color.gradient.colors.map(function (c) { return c[t.key]; }) } };
    }
    return out;
  }
  function draw() {
    var charts = { groupedbar: Charts.GroupedBarChart, combo: Charts.ComboChart, heatmap: Charts.HeatmapChart };
    var drawn = [];
    var current = theme();
    document.querySelectorAll('[data-chart]').forEach(function (el) {
      var spec = JSON.parse(el.getAttribute('data-chart'));
      if (spec.ratio) {
        var total = spec.options.locale.translations.total;
        spec.options.tooltip = {
          valueFormatter: function (value, label) {
            return label === total ? '×' + Math.pow(2, value).toFixed(2) : value;
          },
        };
      }
      // A log axis's ticks are the measured steps: print them as they are,
      // not in d3's SI style (4096 as "4.096k"), or as the build named them
      // (model sizes as "2b"). Carbon takes a formatter only as a function,
      // which the JSON cannot carry.
      var bottom = spec.options.axes && spec.options.axes.bottom;
      if (bottom && bottom.scaleType === 'log' && bottom.ticks) {
        var at = bottom.ticks.values, names = bottom.ticks.labels;
        bottom.ticks.formatter = names
          ? function (v) { var k = at.indexOf(v); return k >= 0 ? names[k] : String(v); }
          : String;
      }
      var chart = new charts[spec.kind](el, { data: spec.data, options: themed(spec.options, current) });
      drawn.push({ chart: chart, options: spec.options });
      var scale = el.dataset.scale && document.getElementById(el.dataset.scale);
      if (scale) highlightSteps(chart, spec, scale);
      if (spec.kind === 'combo') {
        labelGaps(el, chart, spec.data);
        chart.services.events.addEventListener('render-finished', function () { labelGaps(el, chart, spec.data); });
      }
      if (spec.ratio) {
        labelFactors(el, spec.data);
        chart.services.events.addEventListener('render-finished', function () { labelFactors(el, spec.data); });
      }
    });
    new MutationObserver(function () {
      var t = theme();
      if (t.key === current.key) return;
      current = t;
      drawn.forEach(function (d) { d.chart.model.setOptions(themed(d.options, t)); });
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  }
  // Writes the factor, "×2.4", in each square the build marked a highlight,
  // to two figures so it fits the overview's small squares; a quarter of the
  // square's width, up to 14px in the machines' larger ones. Carbon's heatmap
  // has no cell labels and empties its container on every draw (resizes too),
  // so this runs after each: the label goes in the square's own group
  // (Carbon's g.cell, d3-bound to {value, index}, index into the data), in
  // the square's own units. Dark text on a light square, white on a dark one;
  // the label lets the pointer through to the square, so its tooltip and the
  // scale highlight still work.
  function labelFactors(el, data) {
    el.querySelectorAll('g.cell').forEach(function (cell) {
      var point = cell.__data__;
      var square = cell.querySelector('rect.heat');
      if (!square || !point || point.value === null || point.index === -1) return;
      if (!data[point.index].highlight) return;
      var w = Number(square.getAttribute('width'));
      var h = Number(square.getAttribute('height'));
      var text = cell.querySelector('text') || cell.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'text'));
      text.textContent = '×' + Math.pow(2, point.value).toPrecision(2);
      text.setAttribute('x', w / 2);
      text.setAttribute('y', h / 2);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('pointer-events', 'none');
      text.style.fontSize = Math.min(14, w / 4) + 'px';
      text.style.fontWeight = 'bold';
      text.style.fill = luminance(getComputedStyle(square).fill) > 0.4 ? '#161616' : '#ffffff';
    });
  }
  // Writes the factor, "×3.1", in a sparkline's fill at each step the build
  // marked a highlight (that step's fill point carries it), midway between
  // the lines; at the first and last steps, inside the chart. Carbon draws no
  // labels on areas either: placed with the chart's own scales, in the group
  // its fills are drawn in, again after every draw.
  function labelGaps(el, chart, data) {
    var area = el.querySelector('path.area');
    if (!area) return;
    var layer = area.parentNode;
    layer.querySelectorAll('text[data-factor]').forEach(function (t) { t.remove(); });
    var x = chart.services.cartesianScales.getMainXScale();
    var y = chart.services.cartesianScales.getMainYScale();
    var keys = data.map(function (p) { return p.key; });
    var first = Math.min.apply(null, keys), last = Math.max.apply(null, keys);
    data.forEach(function (p) {
      if (p.factor === undefined) return;
      var text = layer.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'text'));
      text.setAttribute('data-factor', '');
      text.textContent = '×' + p.factor.toPrecision(2);
      var anchor = p.key === first ? 'start' : p.key === last ? 'end' : 'middle';
      text.setAttribute('x', x(p.key) + (anchor === 'start' ? 2 : anchor === 'end' ? -2 : 0));
      text.setAttribute('y', y((p.upper + p.lower) / 2));
      text.setAttribute('text-anchor', anchor);
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('pointer-events', 'none');
      text.style.fontSize = '9px';
      text.style.fontWeight = 'bold';
      text.style.fill = 'var(--cds-text-primary)';
    });
  }
  // The relative luminance of an "rgb(r, g, b)" colour, 0 black to 1 white.
  function luminance(rgb) {
    var c = (rgb.match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(function (v) {
      v = v / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  // While the pointer is on a heatmap square, outline its step in the colour
  // scale: the same equal slices of the colour range the heatmap colours by.
  function highlightSteps(chart, spec, scale) {
    var range = spec.options.heatmap.colorDomain;
    var steps = scale.querySelectorAll('[data-step]').length;
    function clear() {
      scale.querySelectorAll('[data-active]').forEach(function (s) { s.removeAttribute('data-active'); });
    }
    chart.services.events.addEventListener('heatmap-mouseover', function (e) {
      clear();
      var value = e.detail.datum && e.detail.datum.value;
      if (value === null || value === undefined) return;
      var step = Math.floor((value - range.min) / (range.max - range.min) * steps);
      step = Math.min(steps - 1, Math.max(0, step));
      var swatch = scale.querySelector('[data-step="' + step + '"]');
      if (swatch) swatch.setAttribute('data-active', '');
    });
    chart.services.events.addEventListener('heatmap-mouseout', clear);
  }
  function whenLaidOut() {
    document.fonts.ready.then(draw);
  }
  if (document.readyState === 'complete') whenLaidOut();
  else window.addEventListener('load', whenLaidOut, { once: true });
})();
