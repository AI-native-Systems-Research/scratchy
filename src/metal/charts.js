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
(function () {
  function draw() {
    var charts = { line: Charts.LineChart, heatmap: Charts.HeatmapChart };
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
      new charts[spec.kind](el, { data: spec.data, options: spec.options });
    });
  }
  function whenLaidOut() {
    document.fonts.ready.then(draw);
  }
  if (document.readyState === 'complete') whenLaidOut();
  else window.addEventListener('load', whenLaidOut, { once: true });
})();
